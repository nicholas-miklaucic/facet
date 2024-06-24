from functools import cached_property
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import jax
import jax.numpy as jnp
import optax
import pyrallis
from flax import linen as nn
from flax.struct import dataclass
from pyrallis.fields import field

from eins.elementwise import ElementwiseOp
from eins import ElementwiseOps as E
from avid import layers
from avid.diffusion import DiffusionBackbone, DiffusionModel, DiT, KumaraswamySchedule
from avid.diled import Category, DiLED, EFormCategory, EncoderDecoder, SpaceGroupCategory
from avid.encoder import Downsample, ReduceSpeciesEmbed, SpeciesEmbed
from avid.layers import EquivariantMixerMLP, Identity, LazyInMLP, MLPMixer
from avid.mlp_mixer import MLPMixerRegressor, O3ImageEmbed
from avid.utils import ELEM_VALS
from avid.vit import (
    AddPositionEmbs,
    Encoder,
    ImageEmbed,
    SingleImageEmbed,
    ViTRegressor,
)

pyrallis.set_config_type('toml')


@dataclass
class VoxelizerConfig:
    """Controls how crystals are encoded into a grid."""

    # The max number of species in a single sample. Due to batching, this affects performance and
    # memory consumption.
    max_unique_species: int = 5

    # The grid size per axis.
    n_grid: int = 24

    # The power to raise the distance to when computing probabilities. 2 is a Gaussian distribution,
    # 1 is Laplace, and higher values produce tighter spheres around the center.
    distance_power: float = 2.0

    # Constant multiplied by the atom radius to get the standard deviation of the generalized
    # normal.
    eff_scale: float = 0.7

    # The number of cells in each direction to extend outwards for computing the probabilities.
    # Higher values are more numerically precise but take longer to process, although there's no
    # impact on the runtime of training or inference beyond the preprocessing step.
    num_cells: int = 4

    @property
    def n_points(self) -> int:
        """The total number of points in a single unit cell."""
        return self.n_grid**3


@dataclass
class DataConfig:
    # The name of the dataset to use.
    dataset_name: str = 'jarvis_dft3d_cleaned'

    # Folder of raw data files.
    raw_data_folder: Path = Path('data/')

    # Folder of processed data files.
    data_folder: Path = Path('precomputed/')

    # Seed for dataset shuffling. Controls the order batches are given to train.
    shuffle_seed: int = 1618

    # Train split.
    train_split: int = 21
    # Test split.
    test_split: int = 2
    # Valid split.
    valid_split: int = 2

    # Data augmentations
    # If False, disables all augmentations.
    do_augment: bool = False
    # Random seed for augmentations.
    augment_seed: int = 12345
    # Whether to apply SO(3) augmentations: proper rotations
    so3: bool = True
    # Whether to apply O(3) augmentations: includes SO(3) and also reflections.
    o3: bool = True
    # Whether to apply T(3) augmentations: origin shifts.
    t3: bool = True

    @property
    def metadata(self) -> Mapping[str, Any]:
        import json
        with open(self.dataset_folder / 'metadata.json', 'r') as metadata:
            metadata = json.load(metadata)
        return metadata


    def __post_init__(self):
        num_splits = self.train_split + self.test_split + self.valid_split
        num_batches = self.metadata['data_size'] // self.metadata['batch_size']
        if num_batches % num_splits != 0:
            msg = f'Data is split {num_splits} ways, which does not divide {num_batches}'
            raise ValueError(msg)

    @property
    def dataset_folder(self) -> Path:
        """Folder where dataset-specific files are stored."""
        return self.data_folder / self.dataset_name


@dataclass
class DataTransformConfig:
    # Density power to transform as.
    density_power: float = 1 / 9

    def density_transform(self) -> ElementwiseOp:
        return E.from_func(lambda x: x ** self.density_power)



class LoggingLevel(Enum):
    """The logging level."""

    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


@dataclass
class CLIConfig:
    # Verbosity of output.
    verbosity: LoggingLevel = LoggingLevel.info
    # Whether to show progress bars.
    show_progress: bool = True

    def set_up_logging(self):
        from rich.logging import RichHandler
        from rich.pretty import install as pretty_install
        from rich.traceback import install as traceback_install

        pretty_install(crop=True, max_string=100, max_length=10)
        traceback_install(show_locals=False)

        import flax.traceback_util as ftu

        ftu.hide_flax_in_tracebacks()

        logging.basicConfig(
            level=self.verbosity.value,
            format='%(message)s',
            datefmt='[%X]',
            handlers=[
                RichHandler(
                    rich_tracebacks=False,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                )
            ],
        )


@dataclass
class DeviceConfig:
    # Either 'cpu', 'gpu', or 'tpu'
    device: str = 'gpu'

    # Limits the number of GPUs used. 0 means no limit.
    max_gpus: int = 1

    # IDs of GPUs to use.
    gpu_ids: list[int] = field(default_factory=list)

    @property
    def jax_device(self):
        devs = jax.devices(self.device)
        if self.device == 'gpu' and self.max_gpus != 0:
            idx = list(range(len(devs)))
            order = [x for x in self.gpu_ids if x in idx] + [
                x for x in idx if x not in self.gpu_ids
            ]
            devs = [devs[i] for i in order[: self.max_gpus]]

        if len(devs) > 1:
            return jax.sharding.PositionalSharding(devs)
        else:
            return devs[0]

    def __post_init__(self):
        import os

        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

        import jax

        jax.config.update('jax_default_device', self.jax_device)


@dataclass
class Layer:
    """Serializable layer representation. Works for any named layer in layers.py or flax.nn."""

    # The name of the layer.
    name: str

    def build(self) -> Callable:
        """Makes a new layer with the given values, or returns the function if it's a function."""
        if self.name == 'Identity':
            return Identity()

        for module in (nn, layers):
            if hasattr(module, self.name):
                layer = getattr(module, self.name)
                if isinstance(layer, nn.Module):
                    return getattr(module, self.name)()
                else:
                    # something like relu
                    return layer

        msg = f'Could not find {self.name} in flax.linen or avid.layers'
        raise ValueError(msg)


@dataclass
class MLPConfig:
    """Settings for MLP configuration."""

    # Inner dimensions for the MLP.
    inner_dims: list[int] = field(default_factory=list)

    # Inner activation.
    activation: str = 'gelu'

    # Final activation.
    final_activation: str = 'Identity'

    # Output dimension. None means the same size as the input.
    out_dim: Optional[int] = None

    # Dropout.
    dropout: float = 0.1

    # Number of heads, for equivariant layer.
    num_heads: int = 1

    # Whether to make the layer equivariant.
    equivariant: bool = False

    def build(self, equivariant=None) -> nn.Module:
        """Builds the head from the config."""
        equivariant = self.equivariant if equivariant is None else equivariant
        if equivariant:
            return EquivariantMixerMLP(
                num_heads=self.num_heads,
                dropout_rate=self.dropout,
                activation=Layer(self.activation).build(),
            )
        else:
            return LazyInMLP(
                inner_dims=self.inner_dims,
                out_dim=self.out_dim,
                inner_act=Layer(self.activation).build(),
                final_act=Layer(self.final_activation).build(),
                dropout_rate=self.dropout,
            )


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    # 208 = 2^4 * 13 steps per epoch with batch of 1: evenly dividing this is nice.
    logs_per_epoch: int = 8


@dataclass
class ViTInputConfig:
    """Controls how images are processed for ViT."""

    # Patch size: p, where image is broken up into p x p x p cubes.
    patch_size: int = 4
    # Patch inner dimension.
    patch_latent_dim: int = 384
    # Position embedding initialization. 'legendre' uses custom Legendre basis, concatenating with
    # the patch embedding. 'learned' uses standard learned embeddings that are added with the patch
    # embedding. 'identity' doesn't apply any positional embeddings.
    pos_embed_type: str = 'learned'
    # Equivariance: ensures patches are invariant to rotations.
    equivariant: bool = True
    # Patch heads.
    patch_heads: int = 8

    def build(self):
        if self.pos_embed_type == 'learned':
            pos_embed = AddPositionEmbs(name='pos_embed')
        elif self.pos_embed_type == 'identity':
            pos_embed = layers.Identity()
        else:
            raise ValueError('Other position embedding types not implemented yet.')

        if self.equivariant:
            im_embed = O3ImageEmbed(
                patch_size=self.patch_size,
                patch_latent_dim=self.patch_latent_dim,
                patch_heads=self.patch_heads,
                pos_embed=pos_embed,
            )
        else:
            im_embed = SingleImageEmbed(
                patch_size=self.patch_size,
                patch_latent_dim=self.patch_latent_dim,
                pos_embed=pos_embed,
            )

        return ImageEmbed(
            im_embed,
            name='image_embed',
        )


@dataclass
class EncoderConfig:
    """Settings for vision transformer encoder: either transformer or MLP mixer."""

    # Number of encoder layers.
    num_layers: int = 2

    # Number of heads for MHSA. Unused in MLP mixer.
    num_heads: int = 16

    # Dropout rate applied to inputs during channel mixing.
    enc_dropout_rate: float = 0.1

    # Dropout rate applied to inputs during token mixing/attention.
    enc_attention_dropout_rate: float = 0.1

    # MLP config after attention layer. The out dim doesn't matter: it's constrained to the input.
    mlp: MLPConfig = field(default_factory=MLPConfig)

    # Token mixer MLP config. Equivariance is taken from the param in this config.
    token_mixer: MLPConfig = field(default_factory=MLPConfig)

    # Whether equivariant attention is used.
    equivariant: bool = True

    # encoder type
    encoder_type: str = "vit"

    def build(self) -> Encoder | MLPMixer:
        if self.encoder_type == 'vit':
            return Encoder(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout_rate=self.enc_dropout_rate,
                attention_dropout_rate=self.enc_attention_dropout_rate,
                mlp=self.mlp.build(),
                equivariant=self.equivariant,
            )
        elif self.encoder_type == 'mlp':
            return MLPMixer(
                num_layers=self.num_layers,
                dropout_rate=self.enc_dropout_rate,
                attention_dropout_rate=self.enc_attention_dropout_rate,
                tokens_mlp=self.token_mixer.build(equivariant=self.equivariant),
                channels_mlp=self.mlp.build(),
            )
        else:
            raise ValueError


@dataclass
class SpeciesEmbedConfig:
    # Remember that this network will be applied for every voxel of every input point: it's how the
    # data that downsamplers or any downstream encoders can use is generated. These are very
    # flop-intensive parameters.

    # Embedding dimension of the species.
    species_embed_dim: int = 32
    # MLP that embeds species embed + density to a new embedding.
    spec_embed: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=64))
    # Whether to use the above MLP config or a simple weighted average.
    use_simple_weighting: bool = False

    def build(self, data: DataConfig) -> ReduceSpeciesEmbed:
        return ReduceSpeciesEmbed(
            SpeciesEmbed(
                len(data.metadata['elements']),
                self.species_embed_dim,
                self.spec_embed.build(),
                self.use_simple_weighting,
            ),
            name='species_embed',
        )


@dataclass
class DownsampleConfig:
    # This is the majority of the runtime!

    # Downsampling factor: if 2, then new image is 1/8 the size.
    factor: list[int] = field(default_factory=lambda: [2])

    # Output channels.
    channels_out: list[int] = field(default_factory=lambda: [256])

    # Kernel size: Fairly flop-expensive to increase. Must be at least 2 * factor - 1 to actually
    # use every input data point.
    kernel_size: list[int] = field(default_factory=lambda: [3])

    def __post_init__(self):
        if not (len(self.factor) == len(self.channels_out) == len(self.kernel_size)):
            msg = f'Configuration {self} has mismatched lengths!'
            raise ValueError(msg)

        for kernel, factor in zip(self.kernel_size, self.factor):
            if (kernel - 1) // 2 < (factor - 1):
                msg = f'Configuration {self} would skip data!'
                raise ValueError(msg)

    def build(self) -> nn.Module:
        if len(self.factor) == 0:
            return Identity()

        downsamples = []
        for factor, channels, kernel in zip(self.factor, self.channels_out, self.kernel_size):
            downsamples.append(Downsample(factor, channels, kernel))
        return nn.Sequential(downsamples, name='downsample')


@dataclass
class ViTRegressorConfig:
    """Composite config for ViT regressor, including all parts."""

    vit_input: ViTInputConfig = field(default_factory=ViTInputConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: MLPConfig = field(default_factory=MLPConfig)
    species_embed: SpeciesEmbedConfig = field(default_factory=SpeciesEmbedConfig)
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    equivariant: bool = False
    out_dim: int = 1

    def build(self, data: DataConfig) -> ViTRegressor:
        if self.equivariant and self.vit_input.pos_embed_type != 'identity':
            msg = 'Need no positional embeddings for equivariant encoder'
            raise ValueError(msg)

        head = self.head.build()
        head.out_dim = self.out_dim
        return ViTRegressor(
            spec_embed=self.species_embed.build(data),
            downsample=self.downsample.build(),
            head=head,
            encoder=self.encoder.build(),
            im_embed=self.vit_input.build(),
            equivariant=self.equivariant,
        )

@dataclass
class MLPMixerRegressorConfig:
    """Composite config for MLP Mixer regressor, including all parts."""

    # Patch size: p, where image is broken up into p x p x p cubes.
    patch_size: int = 3
    # Number of inner transformations to apply for each patch.
    patch_heads: int = 8
    # Patch inner dimension.
    patch_latent_dim: int = 512
    token_mixer: MLPConfig = field(
        default=MLPConfig(activation='gelu', equivariant=True), is_mutable=True
    )
    channel_mixer: MLPConfig = field(default_factory=MLPConfig)
    num_layers: int = 4
    num_heads: int = 4
    head: MLPConfig = field(default_factory=MLPConfig)
    species_embed: SpeciesEmbedConfig = field(default_factory=SpeciesEmbedConfig)
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    out_dim: int = 1

    def build(self, data: DataConfig) -> MLPMixerRegressor:
        head = self.head.build()
        head.out_dim = self.out_dim
        return MLPMixerRegressor(
            spec_embed=self.species_embed.build(data),
            downsample=self.downsample.build(),
            head=head,
            mixer=MLPMixer(
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                tokens_mlp=self.token_mixer.build(),
                channels_mlp=self.channel_mixer.build(),
            ),
            im_embed=ImageEmbed(
                O3ImageEmbed(self.patch_size, self.patch_latent_dim, self.patch_heads, Identity())
            ),
        )


@dataclass
class DiTConfig:
    """Diffusion transformer configuration."""

    condition_mlp_dims: list[int] = field(default_factory=lambda: [128])
    time_dim: int = 64
    time_mlp: MLPConfig = field(default_factory=MLPConfig)
    label_dim: int = 64
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    condition_dropout: float = 0.0

    def build(self, num_classes: int, hidden_dim: int) -> DiT:
        return DiT(
            condition_mlp_dims=self.condition_mlp_dims,
            time_dim=self.time_dim,
            time_mlp=self.time_mlp.build(),
            num_classes=num_classes,
            label_dim=self.label_dim,
            encoder=self.encoder.build(),
            condition_dropout=self.condition_dropout,
            hidden_dim=hidden_dim,
        )


@dataclass
class DiffusionConfig:
    """Diffusion model configuration."""

    # Noise schedule params. (1, 1) is linear, (1.7, 1.9) is near cosine.
    schedule_a: float = 1.7
    schedule_b: float = 1.9
    # Time steps.
    timesteps: int = 100
    # Class dropout rate.
    class_dropout: float = 0.5

    def build(self, model: DiffusionBackbone) -> DiffusionModel:
        return DiffusionModel(
            model=model,
            schedule=KumaraswamySchedule(
                a=self.schedule_a, b=self.schedule_b, timesteps=self.timesteps
            ),
        )


@dataclass
class DiffusionCategoryConfig:
    # Category loss type. 'e_form', 'bandgap', 'space_group', 'cubic_space_group'
    category_loss_type: str = 'cubic_space_group'

    # Number of categories, for continuous targets.
    num_bins: int = 8

    def build(self, data: DataConfig) -> Category:
        if self.category_loss_type == 'e_form':
            return EFormCategory(num_cats=self.num_bins)
        elif self.category_loss_type == 'bandgap':
            pass
        elif self.category_loss_type == 'space_group':
            return SpaceGroupCategory(just_cubic=False)
        elif self.category_loss_type == 'cubic_space_group':
            return SpaceGroupCategory(just_cubic=True)



@dataclass
class DiLEDConfig:
    """DiLED configuration."""

    patch_latent_dim: int = 384
    patch_conv_sizes: list[int] = field(default_factory=lambda: [7])
    patch_conv_strides: list[int] = field(default_factory=lambda: [3])
    patch_conv_features: list[int] = field(default_factory=lambda: [384])
    use_dec_conv: bool = False
    species_embed_dim: int = 128
    species_embed_type: str = 'lossy'
    backbone: DiTConfig = field(default_factory=DiTConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    category: DiffusionCategoryConfig = field(default_factory=DiffusionCategoryConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    head: MLPConfig = field(default_factory=MLPConfig)
    perm_head: MLPConfig = field(default_factory=MLPConfig)
    w: float = 1

    def build(self, data: DataConfig) -> DiLED:
        enc_dec = EncoderDecoder(
            patch_latent_dim=self.patch_latent_dim,
            patch_conv_strides=self.patch_conv_strides,
            patch_conv_features=self.patch_conv_features,
            patch_conv_sizes=self.patch_conv_sizes,
            species_embed_dim=self.species_embed_dim,
            species_embed_type=self.species_embed_type,
            n_species=len(data.metadata['elements']),
            use_dec_conv=self.use_dec_conv,
        )

        category = self.category.build(data)

        dit = self.backbone.build(
            num_classes=category.num_categories, hidden_dim=self.patch_latent_dim
        )
        return DiLED(
            encoder_decoder=enc_dec,
            diffusion=self.diffusion.build(dit),
            category=self.category.build(data),
            class_dropout=self.diffusion.class_dropout,
            encoder=self.encoder.build(),
            head=self.head.build(),
            perm_encoder=self.perm_head.build(),
            w=self.w,
        )


@dataclass
class RegressionLossConfig:
    """Config defining the loss function."""

    # delta for smooth_l1_loss. delta = 0 is L1 loss, and high delta behaves like L2 loss.
    loss_delta: float = 0.1

    # Whether to use RMSE loss instead.
    use_rmse: bool = False

    def regression_loss(self, preds, targets):
        return jax.lax.cond(
            self.use_rmse,
            lambda: jnp.sqrt(optax.losses.squared_error(preds, targets).mean()),
            lambda: (
                optax.losses.huber_loss(preds, targets, delta=self.loss_delta) / self.loss_delta
            ).mean(),
        )


@dataclass
class LossConfig:
    """Config defining the loss function."""

    reg_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)

    def regression_loss(self, preds, targets):
        return self.reg_loss.regression_loss(preds, targets)


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Loss function.
    loss: LossConfig = field(default_factory=LossConfig)

    # Learning rate schedule: 'cosine' for warmup+cosine annealing, 'finder' for a linear schedule
    # that goes up to 20 times the base learning rate.
    lr_schedule_kind: str = 'cosine'

    # Initial learning rate, as a fraction of the base LR.
    start_lr_frac: float = 0.1

    # Base learning rate.
    base_lr: float = 4e-3

    # Final learning rate, as a fraction of the base LR.
    end_lr_frac: float = 0.04

    # Weight decay. AdamW interpretation, so multiplied by the learning rate.
    weight_decay: float = 0.03

    # Beta 1 for Adam.
    beta_1: float = 0.9

    # Beta 2 for Adam.
    beta_2: float = 0.999

    # Nestorov momentum.
    nestorov: bool = True

    # Gradient norm clipping.
    max_grad_norm: float = 1.0

    def lr_schedule(self, num_epochs: int, steps_in_epoch: int):
        if self.lr_schedule_kind == 'cosine':
            warmup_steps = steps_in_epoch * min(5, num_epochs // 2)
            return optax.warmup_cosine_decay_schedule(
                init_value=self.base_lr * self.start_lr_frac,
                peak_value=self.base_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_epochs * steps_in_epoch,
                end_value=self.base_lr * self.end_lr_frac,
            )
        else:
            raise ValueError('Other learning rate schedules not implemented yet')

    def optimizer(self, learning_rate):
        return optax.chain(
            optax.adamw(
                learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
                nesterov=self.nestorov,
            ),
            optax.clip_by_global_norm(self.max_grad_norm),
        )


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 32 * 2

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 100

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Folder to initialize the encoders and downsampling.
    encoder_start_from: Optional[Path] = None

    voxelizer: VoxelizerConfig = field(default_factory=VoxelizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    data_transform: DataTransformConfig = field(default_factory=DataTransformConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    vit: ViTRegressorConfig = field(default_factory=ViTRegressorConfig)
    # diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    diled: DiLEDConfig = field(default_factory=DiLEDConfig)
    mlp: MLPMixerRegressorConfig = field(default_factory=MLPMixerRegressorConfig)

    regressor: str = 'vit'

    task: str = 'e_form'

    def __post_init__(self):
        if self.batch_size % self.data.metadata['batch_size'] != 0:
            raise ValueError(
                'Training batch size should be multiple of data batch size: {} does not divide {}'.format(
                    self.batch_size, self.data.metadata['batch_size']
                )
            )

        self.cli.set_up_logging()
        import warnings

        warnings.filterwarnings(message='Explicitly requested dtype', action='ignore')
        if not self.log.log_dir.exists():
            raise ValueError(f'Log directory {self.log.log_dir} does not exist!')

        from jax.experimental.compilation_cache.compilation_cache import set_cache_dir

        set_cache_dir('/tmp/jax_comp_cache')

    @property
    def train_batch_multiple(self) -> int:
        """How many files should be loaded per training step."""
        return self.batch_size // self.data.metadata['batch_size']

    def build_vit(self) -> ViTRegressor:
        return self.vit.build()

    # def build_diffusion(self) -> DiffusionModel:
    #     diffuser = MLPMixerDiffuser(
    #         embed_dims=self.diffusion.embed_dim,
    #         embed_max_freq=self.diffusion.unet.embed_max_freq,
    #         embed_min_freq=self.diffusion.unet.embed_min_freq,
    #         mixer=self.build_mlp().mixer,
    #     )
    #     return self.diffusion.diffusion.build(diffuser)

    def build_mlp(self) -> MLPMixerRegressor:
        return self.mlp.build()

    def build_regressor(self):
        if self.regressor == 'vit':
            return self.build_vit()
        elif self.regressor == 'mlp':
            return self.build_mlp()
        else:
            raise ValueError

    def build_diled(self):
        return self.diled.build(self.data)


if __name__ == '__main__':
    from pathlib import Path

    from rich.prompt import Confirm

    if Confirm.ask('Generate configs/defaults.toml and configs/minimal.toml?'):
        default_path = Path('configs') / 'defaults.toml'
        minimal_path = Path('configs') / 'minimal.toml'

        default = MainConfig()

        with open(default_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile)

        with open(minimal_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile, omit_defaults=True)

        with default_path.open('r') as conf:
            pyrallis.cfgparsing.load(MainConfig, conf)
