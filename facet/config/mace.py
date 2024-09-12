from collections.abc import Sequence
import logging
from typing import Union

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from pyrallis.fields import field

from facet.e3.activations import S2Activation
from facet.mace.e3_layers import LinearAdapter
from facet.mace.edge_embedding import (
    BesselBasis,
    RadialBasis,
    RadialEmbeddingBlock,
    GaussBasis,
    ExpCutoff,
    Envelope,
)
from facet.mace.mace import (
    LearnedSpeciesWiseRescale,
    MaceModel,
)
from facet.mace.message_passing import (
    NodeFeatureMLPWeightedConv,
    SevenNetConv,
    SimpleInteraction,
    SimpleMixMLPConv,
)
from facet.mace.node_embedding import LinearNodeEmbedding
from facet.mace.self_connection import (
    GateSelfConnection,
    MLPSelfGate,
    S2SelfConnection,
)
from facet.config.utils import Const, Layer, MLPConfig


@dataclass
class RadialBasisConfig:
    num_basis: int = 16

    def build(self) -> RadialBasis:
        raise NotImplementedError()


@dataclass
class GaussBasisConfig(RadialBasisConfig):
    kind: Const('gauss') = 'gauss'
    mu_max: float = 7
    sd: float = 0.25
    mu_trainable: bool = True
    sd_trainable: bool = True

    def build(self) -> GaussBasis:
        return GaussBasis(
            self.num_basis, self.mu_trainable, self.sd_trainable, self.mu_max, self.sd
        )


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
    cutoff_start: float = 0.8
    c: float = 0.1

    def build(self) -> ExpCutoff:
        return ExpCutoff(c=self.c, cutoff_start=self.cutoff_start)


@dataclass
class RadialEmbeddingConfig:
    r_max: float = 7
    r_max_trainable: bool = True
    radial_basis: Union[GaussBasisConfig, BesselBasisConfig] = field(
        default_factory=GaussBasisConfig
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
    message: Union[SevenNetConvConfig, SimpleMixMessageConfig, NodeFeatureMLPWeightedConfig] = (
        field(default_factory=SevenNetConvConfig)
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
    res_beta: int = 18
    res_alpha: int = 17
    normalization: str = 'integral'
    quadrature: str = 'soft'
    use_fft: bool = False

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
    gamma: float = 0.5

    # Number of layers.
    num_layers: int = 2

    # Minimum GCD of the chosen numbers of tensors.
    min_gcd: int = 2

    def build(self) -> Sequence[str]:
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
        default_factory=S2MLPMixerConfig
    )
    head: MLPConfig = field(default_factory=MLPConfig)

    residual: bool = False
    resid_init: str = 'zeros'
    hidden_irreps: Union[IrrepsConfig, tuple[str, ...]] = field(default_factory=IrrepsConfig)
    outs_per_node: int = 64
    block_reduction: str = 'last'
    share_species_embed: bool = True

    def build(
        self,
        num_species: int,
        elem_indices: Sequence[int],
        precision: str,
        out_shift_scale: tuple[float, float],
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
            rescale=LearnedSpeciesWiseRescale(num_species, jnp.array(elem_indices), True, True),
            head_templ=self.head.build(),
            self_connection=self.self_connection.build(),
            outs_per_node=self.outs_per_node,
            share_species_embed=self.share_species_embed,
            block_reduction=self.block_reduction,
            residual=self.residual,
            precision=precision,  # type: ignore
            resid_init=Layer(self.resid_init).build(),
            out_shift_scale=out_shift_scale,
        )
