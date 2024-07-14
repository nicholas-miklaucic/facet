from functools import cached_property
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import e3nn_jax
import jax
import jax.numpy as jnp
import optax
import pyrallis
from flax import linen as nn
from flax.struct import dataclass
from pyrallis.fields import field

from eins.elementwise import ElementwiseOp
from eins import ElementwiseOps as E
from cdv import layers
from cdv.diffusion import DiffusionBackbone, DiffusionModel, DiT, KumaraswamySchedule
from cdv.diled import Category, DiLED, EFormCategory, EncoderDecoder, SpaceGroupCategory
from cdv.encoder import Downsample, ReduceSpeciesEmbed, SpeciesEmbed
from cdv.gnn import (
    GN,
    OldBessel1DBasis,
    Bessel2DBasis,
    DimeNetPP,
    DimeNetPPOutput,
    Fishnet,
    GaussBasis,
    InputEncoder,
    LearnedSpecEmb,
    MLPMessagePassing,
    NodeAggReadout,
    SegmentReduction,
    TripletAngleEmbedding,
)
from cdv.layers import Identity, LazyInMLP, MLPMixer
from cdv.mace import MaceModel
from cdv.mlp_mixer import MLPMixerRegressor, O3ImageEmbed
from cdv.utils import ELEM_VALS
from cdv.vae import Encoder

pyrallis.set_config_type('toml')


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    # 208 = 2^4 * 13 steps per epoch with batch of 1: evenly dividing this is nice.
    logs_per_epoch: int = 8


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
    train_split: int = 22
    # Test split.
    test_split: int = 2
    # Valid split.
    valid_split: int = 2

    # Number of nodes in each batch to pad to.
    batch_n_nodes: int = 512
    # Number of edges in each batch to pad to.
    batch_n_edges: int = 9872
    # Number of graphs in each batch to pad to.
    batch_n_graphs: int = 64

    @property
    def graph_shape(self) -> tuple[int, int, int]:
        return (self.batch_n_nodes, self.batch_n_edges, self.batch_n_graphs)

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

    @property
    def num_species(self) -> int:
        """Number of unique elements."""
        return len(self.metadata['elements'])


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
    activation: str = 'swish'

    # Final activation.
    final_activation: str = 'Identity'

    # Output dimension. None means the same size as the input.
    out_dim: Optional[int] = None

    # Dropout.
    dropout: float = 0.0

    # Whether to add residual.
    residual: bool = False

    # Number of heads, for equivariant layer.
    num_heads: int = 1

    def build(self) -> LazyInMLP:
        """Builds the head from the config."""
        return LazyInMLP(
            inner_dims=self.inner_dims,
            out_dim=self.out_dim,
            inner_act=Layer(self.activation).build(),
            final_act=Layer(self.final_activation).build(),
            dropout_rate=self.dropout,
            residual=self.residual,
        )


@dataclass
class GaussBasisConfig:
    lo: float = 0
    hi: float = 8
    sd: float = 1
    emb: int = 32

    def build(self) -> GaussBasis:
        return GaussBasis(self.lo, self.hi, self.sd, self.emb)


@dataclass
class SegmentReductionConfig:
    reduction: str = 'mean'
    kind: str = 'simple'
    fishnet_mod: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    fishnet_inner: int = 32

    def build(self) -> SegmentReduction:
        if self.kind == 'fishnet':
            return Fishnet(
                self.reduction, net_templ=self.fishnet_mod.build(), inner_dim=self.fishnet_inner
            )
        elif self.kind == 'simple':
            return SegmentReduction(self.reduction)
        else:
            raise ValueError('Unknown kind')


@dataclass
class CoGNConfig:
    # Edge initial embeddings.
    dist_enc: GaussBasisConfig = field(default_factory=GaussBasisConfig)
    # Node embedding dimension.
    node_dim: int = 128
    # Edge embedding dimension:
    edge_dim: int = 128
    # MLP for message passing.
    msg_layer: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    # MLP for node update.
    node_layer: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    # Number of processing blocks.
    num_blocks: int = 5
    # Reduction for node update.
    node_update_reduction: SegmentReductionConfig = field(default_factory=SegmentReductionConfig)
    # Node reduction for readout.
    node_agg_reduction: SegmentReductionConfig = field(default_factory=SegmentReductionConfig)
    # Readout head
    readout_head: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=1))

    def build(self, num_species: int) -> GN:
        input_enc = InputEncoder(
            self.dist_enc.build(),
            nn.Dense(self.edge_dim),
            LearnedSpecEmb(num_species, self.node_dim),
        )
        block_templ = MLPMessagePassing(
            node_reduction=self.node_update_reduction.build(),
            node_emb=self.node_dim,
            msg_dim=self.node_dim,
            msg_templ=self.msg_layer.build(),
            node_templ=self.node_layer.build(),
        )
        readout = NodeAggReadout(self.readout_head.build().copy(out_dim=1))
        return GN(input_enc, self.num_blocks, block_templ, readout)


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
class DimeNetPPConfig:
    num_radial: int = 8
    num_spherical: int = 7
    envelope_exp: int = 6
    cutoff: float = 7
    freq_trainable: bool = True
    species_emb: int = 64
    num_interaction_blocks: int = 4
    act: Layer = field(default_factory=lambda: Layer('sigmoid'))
    initial_embed: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_dist_enc: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_ang_enc: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_down_proj: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_up_proj: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_pre_skip: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    int_post_skip: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))
    edge2node: SegmentReductionConfig = field(default_factory=SegmentReductionConfig)
    node2graph: SegmentReductionConfig = field(default_factory=SegmentReductionConfig)
    head: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=None))

    def build(self, num_species: int) -> DimeNetPP:
        distance_enc = OldBessel1DBasis(
            num_basis=self.num_radial,
            cutoff=self.cutoff,
            envelope_exp=self.envelope_exp,
            freq_trainable=self.freq_trainable,
        )

        return DimeNetPP(
            input_enc=InputEncoder(
                distance_enc=distance_enc.copy(),
                distance_projector=Identity(),
                species_emb=LearnedSpecEmb(num_specs=num_species, embed_dim=self.species_emb),
            ),
            sbf=TripletAngleEmbedding(
                Bessel2DBasis(
                    num_radial=self.num_radial,
                    num_spherical=self.num_spherical,
                    cutoff=self.cutoff,
                    envelope_exp=self.envelope_exp,
                )
            ),
            initial_embed_mlp=self.initial_embed.build(),
            output=DimeNetPPOutput(
                head=self.head.build(),
                edge2node=self.edge2node.build(),
                node2graph=self.node2graph.build(),
            ),
            int_dist_enc=self.int_dist_enc.build(),
            int_ang_enc=self.int_ang_enc.build(),
            int_down_proj_mlp=self.int_down_proj.build(),
            int_up_proj_mlp=self.int_up_proj.build(),
            int_pre_skip_mlp=self.int_pre_skip.build(),
            int_post_skip_mlp=self.int_post_skip.build(),
            num_interaction_blocks=self.num_interaction_blocks,
        )


@dataclass
class MACEConfig:
    max_ell: int = 3
    num_interactions: int = 2
    hidden_irreps: str = '256x0e + 256x1o'
    # hidden_irreps = '16x0e + 16x1o'
    correlation: int = 3  # 4 is better but 5x slower
    readout_mlp_irreps: str = '16x0e'

    def build(self, num_species: int, output_irreps: str) -> MaceModel:
        return MaceModel(
            max_ell=self.max_ell,
            num_interactions=self.num_interactions,
            hidden_irreps=str(e3nn_jax.Irreps(self.hidden_irreps)),
            readout_mlp_irreps=str(e3nn_jax.Irreps(self.readout_mlp_irreps)),
            output_irreps=str(e3nn_jax.Irreps(output_irreps)),
            num_species=num_species,
            correlation=self.correlation,
        )


@dataclass
class RegressionLossConfig:
    """Config defining the loss function."""

    # delta for smoother_l1_loss: the switch point between the smooth version and pure L1 loss.
    loss_delta: float = 1 / 64

    # Whether to use RMSE loss instead.
    use_rmse: bool = False

    def smoother_l1_loss(self, preds, targets):
        a = -5 / 2 / 8
        b = 63 / 8 / 6
        c = -35 / 4 / 4
        d = 35 / 8 / 2
        x = (preds - targets) / self.loss_delta
        x2 = x * x
        x_abs = jnp.abs(x)
        y = jnp.where(x_abs < 1, x2 * d + (x2 * c + (x2 * b + (x2 * a))), x_abs)
        return y * self.loss_delta

    def regression_loss(self, preds, targets, mask):
        mask = mask.reshape(preds.shape)
        if self.use_rmse:
            return jnp.sqrt(optax.losses.squared_error(preds, targets).mean(where=mask))
        else:
            return self.smoother_l1_loss(preds, targets).mean(where=mask)


@dataclass
class LossConfig:
    """Config defining the loss function."""

    reg_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)

    def regression_loss(self, preds, targets, mask):
        return self.reg_loss.regression_loss(preds, targets, mask)


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Loss function.
    loss: LossConfig = field(default_factory=LossConfig)

    # Learning rate schedule: 'cosine' for warmup+cosine annealing
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

    # Schedule-free.
    schedule_free: bool = False

    # Prodigy.
    prodigy: bool = False

    def lr_schedule(self, num_epochs: int, steps_in_epoch: int):
        if self.lr_schedule_kind == 'cosine':
            base_lr = self.base_lr
            if self.prodigy:
                base_lr = 1
            warmup_steps = steps_in_epoch * min(5, num_epochs // 2)
            return optax.warmup_cosine_decay_schedule(
                init_value=base_lr * self.start_lr_frac,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_epochs * steps_in_epoch,
                end_value=base_lr * self.end_lr_frac,
            )
        else:
            raise ValueError('Other learning rate schedules not implemented yet')

    def optimizer(self, learning_rate):
        if self.prodigy:
            tx = optax.contrib.prodigy(
                learning_rate,
                betas=(self.beta_1, self.beta_2),
                weight_decay=self.weight_decay,
                estim_lr_coef=self.base_lr / 4e-3,
            )
        else:
            tx = optax.adamw(
                learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
                nesterov=self.nestorov,
            )
        return optax.chain(tx, optax.clip_by_global_norm(self.max_grad_norm))


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 63 * 1

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 20

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Folder to initialize the encoders and downsampling.
    encoder_start_from: Optional[Path] = None

    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    cogn: CoGNConfig = field(default_factory=CoGNConfig)
    dimenet: DimeNetPPConfig = field(default_factory=DimeNetPPConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)

    regressor: str = 'dimenet'

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

    # def build_diffusion(self) -> DiffusionModel:
    #     diffuser = MLPMixerDiffuser(
    #         embed_dims=self.diffusion.embed_dim,
    #         embed_max_freq=self.diffusion.unet.embed_max_freq,
    #         embed_min_freq=self.diffusion.unet.embed_min_freq,
    #         mixer=self.build_mlp().mixer,
    #     )
    #     return self.diffusion.diffusion.build(diffuser)

    def build_vae(self):
        # output irreps gets changed
        enc = Encoder(self.mace.build(self.data.num_species, '0e'))
        return enc

    def build_regressor(self):
        if self.regressor == 'cogn':
            return self.cogn.build(self.data.num_species)
        elif self.regressor == 'dimenet':
            return self.dimenet.build(self.data.num_species)
        elif self.regressor == 'mace':
            return self.mace.build(self.data.num_species, '0e')
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
