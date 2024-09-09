from collections.abc import Sequence
from json import JSONDecodeError
import logging
from enum import Enum, EnumType
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Union

import e3nn_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
from flax import linen as nn
from flax.struct import dataclass
from pyrallis.fields import field

from cdv import layers
from cdv.e3.activations import S2Activation
from cdv.layers import E3Irreps, Identity, LazyInMLP
from cdv.mace.e3_layers import LinearAdapter, NonlinearAdapter
from cdv.mace.edge_embedding import (
    BesselBasis,
    PolynomialCutoff,
    RadialBasis,
    RadialEmbeddingBlock,
    GaussBasis,
    ExpCutoff,
    Envelope,
)
from cdv.mace.mace import (
    MaceModel,
)
from cdv.mace.message_passing import (
    NodeFeatureMLPWeightedConv,
    SevenNetConv,
    SimpleInteraction,
    SimpleMixMLPConv,
)
from cdv.mace.node_embedding import LinearNodeEmbedding, SevenNetEmbedding
from cdv.mace.self_connection import (
    GateSelfConnection,
    LinearSelfConnection,
    MLPSelfGate,
    S2SelfConnection,
)
from cdv.regression import EFSLoss, EFSWrapper
from cdv.schedule_free import schedule_free_adamw
from cdv.vae import VAE, Decoder, Encoder, LatticeVAE, PropertyPredictor

pyrallis.set_config_type('toml')


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    logs_per_epoch: int = 8

    # Checkpoint every n epochs:
    epochs_per_ckpt: int = 2

    # Test every n epochs:
    epochs_per_valid: float = 2

    # Neptune tags.
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    # The name of the dataset to use.
    dataset_name: str = 'mptrj'

    # Folder of raw data files.
    raw_data_folder: Path = Path('data/')

    # Folder of processed data files.
    data_folder: Path = Path('precomputed/')

    # Seed for dataset shuffling. Controls the order batches are given to train.
    shuffle_seed: int = 1618

    # Train split.
    train_split: int = 30
    # Test split.
    test_split: int = 3
    # Valid split.
    valid_split: int = 3

    # Batches per group to take, 0 means everything. Should only be used for testing.
    batches_per_group: int = 0

    # Number of nodes in each batch to pad to.
    batch_n_nodes: int = 1024
    # Number of neighbors per node.
    k: int = 16
    # Number of graphs in each batch to pad to.
    batch_n_graphs: int = 32

    @property
    def graph_shape(self) -> tuple[int, int, int]:
        return (self.batch_n_nodes, self.k, self.batch_n_graphs)

    @property
    def metadata(self) -> Mapping[str, Any] | None:
        import json

        path = self.dataset_folder / 'metadata.json'

        if not path.exists():
            return None

        try:
            with open(path, 'r') as metadata:
                metadata = json.load(metadata)
                return metadata
        except JSONDecodeError:
            return None

    def __post_init__(self):
        pass
        # num_splits = self.train_split + self.test_split + self.valid_split
        # num_batches = self.metadata['data_size'] // self.metadata['batch_size']
        # if num_batches % num_splits != 0:
        #     msg = f'Data is split {num_splits} ways, which does not divide {num_batches}'
        #     raise ValueError(msg)

    @property
    def dataset_folder(self) -> Path:
        """Folder where dataset-specific files are stored."""
        return self.data_folder / self.dataset_name

    @property
    def num_species(self) -> int:
        """Number of unique elements."""
        return len(self.metadata['elements'])

    def avg_num_neighbors(self, cutoff: float):
        """Estimates average number of neighbors, given a certain cutoff."""
        dists = self.metadata['distances']
        n_nodes = self.metadata['n_nodes']
        return sum([c for b, c in zip(dists['bins'], dists['counts']) if b < cutoff]) / n_nodes

    def avg_dist(self, cutoff: float):
        """Estimates average distance, given a certain cutoff."""
        dists = self.metadata['distances']

        bins = np.array(dists['bins'])
        counts = np.array(dists['counts'])
        return np.average(bins[bins < cutoff], weights=counts[bins < cutoff])


class LoggingLevel(Enum):
    """The logging level."""

    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


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
        if preds.shape != targets.shape:
            msg = f'Incorrect input shapes: {preds.shape} != {targets.shape}'
            raise ValueError(msg)
        if preds.ndim == 2:
            mask = mask[:, None]
        if self.use_rmse:
            return jnp.sqrt(optax.losses.squared_error(preds, targets).mean(where=mask))
        else:
            return self.smoother_l1_loss(preds, targets).mean(where=mask)


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
    max_gpus: int = 0

    # IDs of GPUs to use.
    gpu_ids: list[int] = field(default_factory=list)

    def devices(self):
        devs = jax.devices(self.device)
        if self.device == 'gpu' and self.max_gpus != 0:
            idx = list(range(len(devs)))
            order = [x for x in self.gpu_ids if x in idx] + [
                x for x in idx if x not in self.gpu_ids
            ]
            devs = [devs[i] for i in order[: self.max_gpus]]

        return devs

    def jax_device(self):
        devs = self.devices()

        if len(devs) > 1:
            from jax.experimental import mesh_utils
            from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

            jax.config.update('jax_threefry_partitionable', True)

            d = len(devs)
            mesh = Mesh(mesh_utils.create_device_mesh((d,), devices=jax.devices()[:d]), 'batch')
            sharding = NamedSharding(mesh, P('batch'))
            # replicated_sharding = NamedSharding(mesh, P())
            return sharding
        else:
            return devs[0]

    def __post_init__(self):
        import os

        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

        # import jax
        # if self.max_gpus == 1:
        #     jax.config.update('jax_default_device', self.jax_device())


@dataclass
class Layer:
    """Serializable layer representation. Works for any named layer in layers.py or flax.nn."""

    # The name of the layer.
    name: str

    def build(self) -> Callable:
        """Makes a new layer with the given values, or returns the function if it's a function."""
        if self.name == 'Identity':
            return Identity()

        for module in (nn, layers, jnp):
            if hasattr(module, self.name):
                layer = getattr(module, self.name)
                if isinstance(layer, nn.Module):
                    return getattr(module, self.name)()
                else:
                    # something like relu
                    return layer

        msg = f'Could not find {self.name} in flax.linen or cdv.layers'
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

    # Number of heads, for mixer.
    num_heads: int = 1

    # Whether to use a bias. Also applies to LayerNorm.
    use_bias: bool = False

    # normalization: 'layer', 'weight', 'none'
    normalization: str = 'layer'

    def build(self) -> LazyInMLP:
        """Builds the head from the config."""
        return LazyInMLP(
            inner_dims=self.inner_dims,
            out_dim=self.out_dim,
            inner_act=Layer(self.activation).build(),
            final_act=Layer(self.final_activation).build(),
            dropout_rate=self.dropout,
            residual=self.residual,
            use_bias=self.use_bias,
            normalization=self.normalization,
        )


class Constant:
    value: Any = None

    @classmethod
    def _decode(cls, x):
        if x == cls.value:
            return x
        else:
            raise ValueError(f'{x} != {cls.value}')

    def _encode(self):
        return self.value


pyrallis.decode.register(Constant, lambda t, x: t._decode(x), include_subclasses=True)
pyrallis.encode.register(Constant, Constant._encode)


def Const(val) -> type[Constant]:
    class ConstantImpl(Constant):
        value = val

    return ConstantImpl


@dataclass
class RadialBasisConfig:
    num_basis: int = 10

    def build(self) -> RadialBasis:
        raise NotImplementedError()


@dataclass
class GaussBasisConfig(RadialBasisConfig):
    kind: Const('gauss') = 'gauss'
    mu_max: float = 7
    sd: float = 1

    def build(self) -> GaussBasis:
        return GaussBasis(self.num_basis, self.mu_max, self.sd)


@dataclass
class BesselBasisConfig(RadialBasisConfig):
    kind: Const('bessel') = 'bessel'
    freq_trainable: bool = True

    def build(self) -> BesselBasis:
        return BesselBasis(num_basis=self.num_basis, freq_trainable=self.freq_trainable)


@dataclass
class EnvelopeConfig:
    def build(self) -> Envelope:
        raise NotImplementedError


@dataclass
class ExpCutoffConfig:
    kind: Const('exp') = 'exp'
    cutoff_start: float = 0.9
    c: float = 0.1

    def build(self) -> ExpCutoff:
        return ExpCutoff(c=self.c, cutoff_start=self.cutoff_start)


@dataclass
class RadialEmbeddingConfig:
    r_max: float = 5
    r_max_trainable: bool = False
    radial_basis: Union[GaussBasisConfig, BesselBasisConfig] = field(
        default_factory=BesselBasisConfig
    )
    envelope: Union[ExpCutoffConfig] = field(default_factory=ExpCutoffConfig)
    radius_transform: str = 'Identity'

    def build(self):
        return RadialEmbeddingBlock(
            r_max=self.r_max,
            r_max_trainable=self.r_max_trainable,
            basis=self.radial_basis.build(),
            envelope=self.envelope.build(),
            radius_transform=Layer(self.radius_transform).build(),
        )


@dataclass
class MessageConfig:
    pass


@dataclass
class SimpleMixMessageConfig(MessageConfig):
    kind: Const('simple-mix-mlp-conv') = 'simple-mix-mlp-conv'
    avg_num_neighbors: float = 14
    max_ell: int = 3
    radial_mix: MLPConfig = field(default_factory=MLPConfig)

    def build(self) -> SimpleMixMLPConv:
        return SimpleMixMLPConv(
            irreps_out=None,
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            radial_mix=self.radial_mix.build(),
        )


@dataclass
class SevenNetConvConfig(MessageConfig):
    kind: Const('sevennet-conv') = 'sevennet-conv'
    avg_num_neighbors: float = 14
    max_ell: int = 2
    radial_weight: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            inner_dims=[64, 64],
            final_activation='shifted_softplus',
            out_dim=0,
            use_bias=False,
        )
    )

    def build(self) -> SevenNetConv:
        if (
            self.radial_weight.final_activation != 'Identity'
            and Layer(self.radial_weight.final_activation).build()(0.0) != 0
        ):
            raise ValueError('Final activation for radial MLP must go through origin.')

        if self.radial_weight.use_bias:
            raise ValueError('Radial MLP cannot use bias.')

        return SevenNetConv(
            irreps_out=None,
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            radial_weight=self.radial_weight.build(),
        )


@dataclass
class NodeFeatureMLPWeightedConfig(MessageConfig):
    kind: Const('node-feature-mlp-weighted') = 'node-feature-mlp-weighted'
    avg_num_neighbors: float = 14
    max_ell: int = 2
    node_feature_mlp: MLPConfig = field(default=MLPConfig)

    def build(self) -> NodeFeatureMLPWeightedConv:
        return NodeFeatureMLPWeightedConv(
            irreps_out=None,
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            node_feature_mlp=self.node_feature_mlp.build(),
        )


@dataclass
class InteractionConfig:
    message: Union[SimpleMixMessageConfig, SevenNetConvConfig, NodeFeatureMLPWeightedConfig] = (
        field(default_factory=SimpleMixMessageConfig)
    )


@dataclass
class SimpleInteractionBlockConfig(InteractionConfig):
    kind: Const('simple') = 'simple'

    def build(self) -> SimpleInteraction:
        return SimpleInteraction(irreps_out=None, conv=self.message.build())


@dataclass
class NodeEmbeddingConfig:
    embed_dim: int = 64

    def build(self, num_species: int, elem_indices: Sequence[int]) -> LinearNodeEmbedding:
        raise NotImplementedError


@dataclass
class LinearNodeEmbeddingConfig(NodeEmbeddingConfig):
    kind: Const('linear') = 'linear'

    def build(self, num_species: int, elem_indices: Sequence[int]) -> LinearNodeEmbedding:
        return LinearNodeEmbedding(
            f'{self.embed_dim}x0e', num_species=num_species, element_indices=jnp.array(elem_indices)
        )


@dataclass
class ReadoutConfig:
    pass


@dataclass
class LinearReadoutConfig(ReadoutConfig):
    kind: Const('linear') = 'linear'

    def build(self) -> LinearAdapter:
        return LinearAdapter(irreps_out=None)


@dataclass
class SelfConnectionConfig:
    pass


@dataclass
class GateConfig(SelfConnectionConfig):
    kind: Const('gate') = 'gate'

    def build(self) -> GateSelfConnection:
        return GateSelfConnection(irreps_out=None)


@dataclass
class S2ActivationConfig:
    activation: str = 'silu'
    res_beta: int = 16
    res_alpha: int = 15
    normalization: str = 'integral'
    quadrature: str = 'soft'
    use_fft: bool = True

    def build(self) -> S2Activation:
        return S2Activation(
            activation=Layer(self.activation).build(),
            res_beta=self.res_beta,
            res_alpha=self.res_alpha,
            normalization=self.normalization,  # type: ignore
            quadrature=self.quadrature,  # type: ignore
            fft=self.use_fft,
        )


@dataclass
class S2MLPMixerConfig(SelfConnectionConfig):
    kind: Const('s2-mlp-mixer') = 's2-mlp-mixer'

    s2_grid: S2ActivationConfig = field(default_factory=S2ActivationConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)

    def build(self) -> S2SelfConnection:
        return S2SelfConnection(
            irreps_out=None,
            act=self.s2_grid.build(),
            mlp=self.mlp.build(),
            num_heads=self.mlp.num_heads,
        )


@dataclass
class MLPSelfGateConfig(SelfConnectionConfig):
    kind: Const('mlp-gate') = 'mlp-gate'
    num_hidden_layers: int = 1
    mlp: MLPConfig = field(default_factory=MLPConfig)

    def build(self) -> MLPSelfGate:
        return MLPSelfGate(
            irreps_out=None, num_hidden_layers=self.num_hidden_layers, mlp_templ=self.mlp.build()
        )


@dataclass
class IrrepsConfig:
    kind: Const('derived') = 'derived'
    # Total dimension. May be off by one or two due to rounding.
    dim: int = 384
    # Max degree of tensors in representation.
    max_degree: int = 2
    # Decay for allocating dimensions to the different kinds of tensor. Dimension d+1 gets gamma
    # times the number of dimension d tensors.
    gamma: float = 1

    # Number of layers.
    num_layers: int = 2

    # Minimum GCD of the chosen numbers of tensors.
    min_gcd: int = 2

    def build(self) -> str:
        ells = np.arange(self.max_degree + 1)
        props = float(self.gamma) ** ells
        props /= sum(props)
        props *= self.dim
        tensor_dim = 2 * ells + 1
        num_tensors = [round(x / self.min_gcd) * self.min_gcd for x in props / tensor_dim]

        if any(t == 0 for t in num_tensors):
            logging.warn(f'One given input has no assigned channels: {num_tensors}')

        return [
            ' + '.join([f'{num}x{ell}e' for num, ell in zip(num_tensors, ells)])
        ] * self.num_layers


@dataclass
class MACEConfig:
    node_embed: Union[LinearNodeEmbeddingConfig] = field(default_factory=LinearNodeEmbeddingConfig)
    edge_embed: RadialEmbeddingConfig = field(default_factory=RadialEmbeddingConfig)
    interaction: Union[SimpleInteractionBlockConfig] = field(
        default_factory=SimpleInteractionBlockConfig
    )
    readout: Union[LinearReadoutConfig] = field(default_factory=LinearReadoutConfig)
    self_connection: Union[GateConfig, MLPSelfGateConfig, S2MLPMixerConfig] = field(
        default_factory=GateConfig
    )
    head: MLPConfig = field(default_factory=MLPConfig)

    residual: bool = False
    resid_init: str = 'zeros'
    hidden_irreps: Union[IrrepsConfig, tuple[str, ...]] = (
        '128x0e + 64x1o + 32x2e',
        '128x0e + 64x1o + 32x2e',
    )
    outs_per_node: int = 64
    interaction_reduction: str = 'last'
    share_species_embed: bool = True

    def build(
        self,
        num_species: int,
        elem_indices: Sequence[int],
        precision: str,
    ) -> MaceModel:
        if isinstance(self.hidden_irreps, IrrepsConfig):
            hidden_irreps = self.hidden_irreps.build()
        else:
            hidden_irreps = self.hidden_irreps
        return MaceModel(
            hidden_irreps=hidden_irreps,
            node_embedding=self.node_embed.build(num_species, elem_indices),
            edge_embedding=self.edge_embed.build(),
            interaction=self.interaction.build(),
            readout=self.readout.build(),
            head_templ=self.head.build(),
            self_connection=self.self_connection.build(),
            # self_connection=LinearSelfConnection(irreps_out=None),
            outs_per_node=self.outs_per_node,
            share_species_embed=self.share_species_embed,
            interaction_reduction=self.interaction_reduction,
            residual=self.residual,
            precision=precision,
            resid_init=Layer(self.resid_init).build(),
        )


@dataclass
class LossConfig:
    """Config defining the loss function."""

    reg_loss: RegressionLossConfig = field(default_factory=RegressionLossConfig)
    energy_weight: float = 1
    force_weight: float = 0.1
    stress_weight: float = 0.01

    def regression_loss(self, preds, targets, mask):
        return self.reg_loss.regression_loss(preds, targets, mask)

    @property
    def efs_wrapper(self) -> EFSWrapper:
        compute_fs = not (self.force_weight == 0 and self.stress_weight == 0)
        return EFSWrapper(compute_fs=compute_fs)

    @property
    def efs_loss(self) -> EFSLoss:
        return EFSLoss(
            loss_fn=self.reg_loss.regression_loss,
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            stress_weight=self.stress_weight,
        )


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
    max_grad_norm: float = 3.0

    # Schedule-free.
    schedule_free: bool = False

    # Prodigy.
    prodigy: bool = False

    def lr_schedule(self, num_epochs: int, steps_in_epoch: int):
        num_steps = num_epochs * steps_in_epoch
        warmup_steps = round(num_steps / 10)
        if self.lr_schedule_kind == 'cosine':
            # warmup_steps = steps_in_epoch * max(1, round(num_epochs / 5))
            if self.prodigy:
                base_lr = 1.0
                warmup_frac = 0.2
                warmup_steps = round(num_steps * warmup_frac)
                other_steps = num_steps - warmup_steps
                sched = optax.join_schedules(
                    [
                        optax.polynomial_schedule(
                            init_value=base_lr * self.start_lr_frac,
                            end_value=base_lr,
                            power=1,
                            transition_steps=warmup_steps,
                        ),
                        optax.polynomial_schedule(
                            init_value=base_lr,
                            end_value=base_lr * self.end_lr_frac,
                            power=1,
                            transition_steps=other_steps,
                        ),
                    ],
                    boundaries=[warmup_steps],
                )

                return sched
            else:
                base_lr = self.base_lr
                return optax.warmup_cosine_decay_schedule(
                    init_value=base_lr * self.start_lr_frac,
                    peak_value=base_lr,
                    warmup_steps=warmup_steps,
                    decay_steps=num_steps,
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
                estim_lr_coef=self.base_lr,
            )
        elif self.schedule_free:
            # tx = schedule_free_adamw(
            #     learning_rate=learning_rate,
            #     b1=self.beta_1,
            #     b2=self.beta_2,
            #     weight_decay=self.weight_decay,
            #     eps=1e-6,
            # )
            tx = optax.adamw(
                learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
                nesterov=self.nestorov,
            )
            tx = optax.contrib.mechanize(tx)
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
    batch_size: int = 32 * 1

    # Number of stacked batches to process at once, if not given by the number of devices. Useful
    # for mocking multi-GPU training batches with a single GPU.
    stack_size: int = 1

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 20

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Folder to initialize the encoders and downsampling.
    encoder_start_from: Optional[Path] = None

    # Precision: 'f32' or 'bf16'.
    precision: str = 'bf16'

    # Debug mode: turns off mid-run checkpointing and Neptune tracking.
    debug_mode: bool = False

    # Display kind for training runs: One of 'dashboard', 'progress', or 'quiet'.
    display: str = 'dashboard'

    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    mace: MACEConfig = field(default_factory=MACEConfig)

    regressor: str = 'mace'

    task: str = 'e_form'

    def __post_init__(self):
        if (
            self.data.metadata is not None
            and self.batch_size % self.data.metadata['batch_size'] != 0
        ):
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
        if self.data.metadata is None:
            return 1
        else:
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
        return VAE(
            Encoder(
                self.mace.build(self.data.num_species, '0e', None),
                latent_dim=256,
                latent_space=LatticeVAE(),
            ),
            PropertyPredictor(LazyInMLP([256], dropout_rate=0.5)),
            Decoder(self.mace.build(self.data.num_species, '0e', None)),
            prop_reg_loss=self.train.loss.regression_loss,
        )

    def build_regressor(self):
        if self.regressor == 'mace':
            return self.mace.build(
                self.data.num_species,
                self.data.metadata['element_indices'],
                self.precision,
            )
        else:
            raise ValueError(f'{self.regressor} not supported')

    def as_dict(self):
        """Serializes the relevant values into a dictionary suitable for e.g., Neptune logging."""
        cfg: dict = pyrallis.encode(self)
        for key in ('do_profile', 'cli', 'log', 'debug_mode', 'display'):
            cfg.pop(key)

        for key in ('raw_data_folder', 'data_folder', 'batch_n_nodes', 'batch_n_graphs'):
            cfg['data'].pop(key)

        def convert_leaves(leaf):
            if isinstance(leaf, dict):
                return {k: convert_leaves(v) for k, v in leaf.items()}
            elif leaf is None:
                return 'None'
            elif isinstance(leaf, (tuple, list)):
                return {i: convert_leaves(v) for i, v in enumerate(leaf)}
            else:
                return leaf

        cfg = convert_leaves(cfg)

        return cfg


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
            cfg = pyrallis.cfgparsing.load(MainConfig, conf)

        print(cfg.as_dict())
