"""
MACE network code. Adapted from https://github.com/ACEsuit/mace-jax.
"""

from collections.abc import Sequence
import functools
import math
from typing import Callable, Optional
from typing import Set, Union
from flax import linen as nn
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int

from eins import EinsOp

from cdv.databatch import CrystalGraphs
from cdv.e3_layers import Linear, LinearReadoutBlock, NonlinearReadoutBlock
from cdv.layers import SegmentReduction, SegmentReductionKind
from cdv.layers import Context, E3NormNorm, LazyInMLP, E3Irreps, E3IrrepsArray, edge_vecs
from cdv.utils import debug_stat, debug_structure
from cdv.self_connection import EquivariantProductBasisBlock, LinearSelfConnection


def safe_norm(x: jnp.ndarray, axis: int | None = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


class LinearNodeEmbedding(nn.Module):
    num_species: int
    element_indices: Int[Array, ' max_species']
    irreps_out: E3Irreps

    def setup(self):
        self.irreps_out_calc = E3Irreps(self.irreps_out).filter('0e').regroup()
        self.out_dim = E3Irreps(self.irreps_out_calc).dim
        self.embed = nn.Embed(self.num_species, self.out_dim)

    def __call__(self, node_species: Int[Array, ' batch']) -> E3IrrepsArray:
        return E3IrrepsArray(self.irreps_out_calc, self.embed(self.element_indices[node_species]))


class RadialEmbeddingBlock(nn.Module):
    r_max: float
    basis_functions: Callable[[jnp.ndarray], jnp.ndarray]
    envelope_function: Callable[[jnp.ndarray], jnp.ndarray]
    avg_r_min: Optional[float] = None

    def __call__(self, edge_lengths: jnp.ndarray) -> E3IrrepsArray:
        """batch -> batch num_basis"""

        def func(lengths):
            basis = self.basis_functions(lengths, self.r_max)  # [n_nodes,k, num_basis]
            cutoff = self.envelope_function(lengths, self.r_max)  # [n_nodes, k]
            return basis * cutoff[..., None]  # [n_edges, num_basis]

        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(self.avg_r_min, self.r_max, 1000, dtype=jnp.float64)
                factor = jnp.mean(func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[..., None], 0.0, func(edge_lengths)
        )  # [n_edges, num_basis]
        return E3IrrepsArray(f'{embedding.shape[-1]}x0e', jnp.array(embedding))


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: E3Irreps
    max_ell: int
    activation: Callable
    mix: str = 'mix'

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
    ) -> E3IrrepsArray:
        """-> n_nodes irreps"""
        assert node_feats.ndim == 2

        messages_broadcast = node_feats[
            jnp.repeat(jnp.arange(node_feats.shape[0])[..., None], 16, axis=-1)
        ]
        # debug_structure(msgs=messages, vecs=vectors)

        msg_prefix = messages_broadcast.filter(self.target_irreps)
        vec_harms = e3nn.tensor_product(
            messages_broadcast,
            e3nn.spherical_harmonics(range(1, self.max_ell + 1), vectors, True),
            filter_ir_out=self.target_irreps,
        )

        # debug_structure(
        #     msg=messages_broadcast, vecs=vectors, msg_pref=msg_prefix, vec_harm=vec_harms
        # )

        messages = e3nn.concatenate(
            [msg_prefix, vec_harms],
            axis=-1,
        ).regroup()  # [n_nodes, irreps]

        if self.mix == 'mix':
            radial = LazyInMLP(
                # np.rint(np.linspace(radial_embedding.shape[-1], messages.irreps.num_irreps, 3)[1:-1])
                # .astype(int)
                # .tolist(),
                [],
                out_dim=messages.irreps.num_irreps,
                normalization='none',
                name='radial_mix',
            )(radial_embedding, ctx)  # [n_nodes, k, num_irreps]

            # debug_structure(messages=messages, mix=radial, rad=radial_embedding.array)
            # debug_stat(messages=messages.array, mix=mix.array, rad=radial_embedding.array)
            # radial = nn.sigmoid(radial.array)
            messages = messages * radial  # [n_nodes, k, irreps]

            # debug_stat(messages=messages, radial=radial)

            zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
            # TODO flip this perhaps?
            node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
            # node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)
        elif self.mix == 'mlpa':
            radial = LazyInMLP(
                # np.rint(np.linspace(radial_embedding.shape[-1], messages.irreps.num_irreps, 3)[1:-1])
                # .astype(int)
                # .tolist(),
                [32],
                out_dim=64,
                inner_act=self.activation,
                normalization='none',
                name='radial_msg',
            )(radial_embedding.array, ctx)  # [n_edges, 64]

            x = jnp.concat(
                [messages.filter('0e').array, radial], axis=-1
            )  # [n_edges, num_scalars + 64]

            z = LazyInMLP(
                # np.rint(np.linspace(x.shape[-1], messages.irreps.num_irreps, 3)[1:-1])
                # .astype(int)
                # .tolist(),
                [],
                out_dim=messages.irreps.num_irreps,
                inner_act=self.activation,
                normalization='none',
                name='msg_attention',
            )(x, ctx)

            a = jnp.exp(z)  # [n_edges, num_irreps]

            normalization = jnp.zeros((node_feats.shape[0], a.shape[-1]), jnp.float32)
            normalization = normalization.at[receivers].add(a)  # [n_nodes, num_irreps]

            att = a / (normalization[receivers] + 1e-6)

            # debug_stat(att=att, a=a, norm=normalization)

            zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
            node_feats = zeros.at[receivers].add(messages * att)  # [n_nodes, irreps]

        return node_feats


class InteractionBlock(nn.Module):
    conv: MessagePassingConvolution

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges, ]
        ctx: Context,
    ) -> tuple[E3IrrepsArray, E3IrrepsArray]:
        """-> n_nodes irreps"""
        # assert node_feats.ndim == 2
        # assert vectors.ndim == 2
        # assert radial_embedding.ndim == 2

        new_node_feats = Linear(node_feats.irreps, name='linear_up')(node_feats)
        # debug_stat(up=node_feats.array)
        new_node_feats = E3NormNorm()(new_node_feats)

        new_node_feats = self.conv(vectors, new_node_feats, radial_embedding, receivers, ctx)
        new_node_feats = E3NormNorm()(new_node_feats)
        # debug_stat(conv=node_feats.array)

        new_node_feats = Linear(self.conv.target_irreps, name='linear_down')(new_node_feats)
        new_node_feats = E3NormNorm()(new_node_feats)
        # debug_stat(down=node_feats.array)

        if new_node_feats.irreps == node_feats.irreps:
            node_feats = new_node_feats + node_feats
        else:
            node_feats = new_node_feats

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]


class MACELayer(nn.Module):
    first: bool
    last: bool
    num_features: int
    interaction_irreps: E3Irreps
    hidden_irreps: E3Irreps
    activation: Callable
    num_species: int
    name: Optional[str]
    # InteractionBlock:
    max_ell: int
    avg_num_neighbors: float
    # EquivariantProductBasisBlock:
    correlation: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool
    # ReadoutBlock:
    output_irreps: E3Irreps
    readout_mlp_irreps: E3Irreps

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges]
        ctx: Context,
        species_embed: Float[Array, 'num_species num_embed'] | None = None,
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ):
        """-> (n_nodes output_irreps, n_nodes features*hidden_irreps)"""
        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        hidden_irreps = E3Irreps(self.hidden_irreps)
        output_irreps = E3Irreps(self.output_irreps)
        interaction_irreps = E3Irreps(self.interaction_irreps)
        readout_mlp_irreps = E3Irreps(self.readout_mlp_irreps)

        node_feats = InteractionBlock(
            MessagePassingConvolution(
                target_irreps=self.num_features * interaction_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                max_ell=self.max_ell,
                activation=self.activation,
            )
        )(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            ctx=ctx,
        )

        # node_feats /= jnp.sqrt(self.avg_num_neighbors)

        # if self.first:
        #     # Selector TensorProductGraphs
        #     node_feats = Linear(
        #         self.num_features * interaction_irreps,
        #         num_indexed_weights=self.num_species,
        #         gradient_normalization='path',
        #         name='skip_tp_first',
        #     )(node_species, node_feats)
        #     node_feats = profile(f'{self.name}: skip_tp_first', node_feats, node_mask[:, None])

        irreps_out = self.num_features * hidden_irreps
        if False:
            self_interact_block = EquivariantProductBasisBlock(
                target_irreps=self.num_features * hidden_irreps,
                correlation=self.correlation,
                num_species=self.num_species,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
            )
        else:
            self_interact_block = LinearSelfConnection(irreps_out=irreps_out)

        node_feats = self_interact_block(
            node_feats=node_feats, node_specie=node_species, species_embed=species_embed, ctx=ctx
        )

        if not self.last:
            node_outputs = LinearReadoutBlock(output_irreps)(
                node_feats, ctx
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonlinearReadoutBlock(
                hidden_irreps=readout_mlp_irreps,
                irreps_out=output_irreps,
                activation=self.activation,
            )(node_feats, ctx)  # [n_nodes, output_irreps]

        return node_outputs, node_feats


class MACE(nn.Module):
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]
    output_irreps: E3Irreps  # Irreps of the output, default 1x0e
    r_max: float
    avg_r_min: float
    num_interactions: int  # Number of interactions (layers), default 2
    hidden_irreps: E3Irreps  # 256x0e or 128x0e + 128x1o
    readout_mlp_irreps: E3Irreps  # Hidden irreps of the MLP in last readout, default 16x0e
    avg_num_neighbors: float
    num_species: int
    elem_indices: Sequence[int]
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]
    max_ell: int
    correlation: int
    gate: Callable  #  = jax.nn.silu
    symmetric_tensor_product_basis: bool  #  = True
    off_diagonal: bool  #  = False
    interaction_irreps: str | E3Irreps  #  = 'o3_restricted'  # or o3_full
    node_embedding_type: type[nn.Module]  #  = LinearNodeEmbedding
    share_species_embed: bool
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: Optional[int]

    global_proj_templ: LazyInMLP = LazyInMLP([])

    def setup(self):
        self.output_irreps_calc = E3Irreps(self.output_irreps)
        self.hidden_irreps_calc = E3Irreps(self.hidden_irreps)
        self.readout_mlp_irreps_calc = E3Irreps(self.readout_mlp_irreps)

        if self.num_features is None:
            self.num_features_calc = functools.reduce(
                math.gcd, (mul for mul, _ in self.hidden_irreps_calc)
            )
        else:
            self.num_features_calc = self.num_features

        self.hidden_irreps_calc = E3Irreps(
            [(mul // self.num_features_calc, ir) for mul, ir in self.hidden_irreps_calc]
        )

        if self.interaction_irreps == 'o3_restricted':
            self.interaction_irreps_calc = E3Irreps.spherical_harmonics(self.max_ell)
        elif self.interaction_irreps == 'o3_full':
            self.interaction_irreps_calc = E3Irreps(e3nn.Irrep.iterator(self.max_ell))
        else:
            self.interaction_irreps_calc = E3Irreps(self.interaction_irreps)

        # Embeddings
        self.node_embedding = self.node_embedding_type(
            self.num_species,
            self.elem_indices,
            self.num_features_calc * self.hidden_irreps_calc,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            avg_r_min=self.avg_r_min,
            basis_functions=self.radial_basis,
            envelope_function=self.radial_envelope,
        )

        layers = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1
            hidden_irreps = (
                E3Irreps(self.hidden_irreps_calc)
                if not last
                else E3Irreps(self.hidden_irreps_calc).filter(self.output_irreps_calc)
            )

            # to output just a vector, there needs to be enough scalars to do the gating. I'm not
            # sure why the above code filters the way it does.
            hidden_irreps = E3Irreps(self.hidden_irreps_calc)
            layers.append(
                MACELayer(
                    first=first,
                    last=last,
                    num_features=self.num_features_calc,
                    interaction_irreps=self.interaction_irreps_calc,
                    hidden_irreps=hidden_irreps,
                    max_ell=self.max_ell,
                    avg_num_neighbors=self.avg_num_neighbors,
                    activation=self.gate,
                    num_species=self.num_species,
                    correlation=self.correlation,
                    output_irreps=self.output_irreps_calc,
                    readout_mlp_irreps=self.readout_mlp_irreps_calc,
                    symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                    off_diagonal=self.off_diagonal,
                    name=f'layer_{i}',
                )
            )

        self.layers = layers

        self.global_proj_mlp = self.global_proj_templ.copy(out_dim=self.node_embedding.out_dim)

    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
        extra_node_features: jnp.ndarray | None = None,
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ) -> E3IrrepsArray:
        """
        global_features: latent vector to be incorporated into initial node embeddings
        -> n_nodes num_interactions output_irreps
        """
        # assert vectors.ndim == 3 and vectors.shape[-1] == 3
        # assert node_species.ndim == 1
        # assert receivers.ndim == 2
        # assert vectors.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_species).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        species_embs = node_feats

        if extra_node_features is not None:
            node_feat_arr = self.global_proj_mlp(
                jnp.concat([node_feats.array, extra_node_features], axis=-1), ctx
            )
            node_feats = E3IrrepsArray(node_feats.irreps, node_feat_arr)

        # print(node_feats)

        if not (hasattr(vectors, 'irreps') and hasattr(vectors, 'array')):
            vectors = E3IrrepsArray('1o', vectors)

        radial_embedding = self.radial_embedding(safe_norm(vectors.array, axis=-1))
        # debug_structure(radial_embedding=radial_embedding)

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            node_outputs, node_feats = self.layers[i](
                vectors,
                node_feats,
                node_species,
                radial_embedding,
                receivers,
                node_mask=node_mask,
                ctx=ctx,
                species_embed=species_embs.array if self.share_species_embed else None,
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        # print([k.shape for k in outputs])
        return e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MaceModel(nn.Module):
    """Graph network that wraps MACE."""

    num_species: int
    elem_indices: Sequence[int]
    output_graph_irreps: str  # output irreps, 1x0e for scalar
    output_node_irreps: str | None  # output by-node irreps
    hidden_irreps: str  # 256x0e or 128x0e + 128x1o
    readout_mlp_irreps: str  # Hidden irreps of the MLP in last readout, default 16x0e

    scalar_mean: float = 0.0
    scalar_std: float = 1.0

    num_interactions: int = 2  # Number of interactions (layers), default 2

    # How to combine the outputs of different interaction blocks.
    # 'last' is special: it means the last block.
    interaction_reduction: str = 'last'
    # Node reduction.
    node_reduction: SegmentReductionKind = 'mean'

    num_radial_embeds: int = 8
    max_r: float = 5.0
    avg_r_min: float = 1.5
    radial_envelope_scale: float = 2
    radial_envelope_intercept: float = 1.2

    avg_num_neighbors: float = 20.0

    max_ell: int = 3  # Max spherical harmonic degree, default 3
    correlation: int = 2  # Correlation order at each layer (~ node_features^correlation), default 3

    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    interaction_irreps: Union[str, E3Irreps] = 'o3_restricted'  # or o3_full

    share_species_embed: bool = True
    num_features: Optional[int] = None

    gate: Callable = nn.tanh

    def setup(self):
        def bessel_basis(length, max_length):
            return e3nn.bessel(length, self.num_radial_embeds, max_length)

        def soft_envelope(length, max_length):
            return e3nn.soft_envelope(
                length,
                max_length,
                arg_multiplicator=self.radial_envelope_scale,
                value_at_origin=self.radial_envelope_intercept,
            )

        self.mace = MACE(
            output_irreps=self.output_irreps,
            r_max=self.max_r,
            num_interactions=self.num_interactions,
            avg_r_min=self.avg_r_min,
            hidden_irreps=self.hidden_irreps,
            readout_mlp_irreps=self.readout_mlp_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            num_species=self.num_species,
            elem_indices=self.elem_indices,
            max_ell=self.max_ell,
            correlation=self.correlation,
            gate=self.gate,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
            interaction_irreps=self.interaction_irreps,
            node_embedding_type=LinearNodeEmbedding,
            radial_basis=bessel_basis,
            radial_envelope=soft_envelope,
            share_species_embed=self.share_species_embed,
            num_features=self.num_features,
        )

        self.node_reduction_mods = [
            SegmentReduction(self.node_reduction)
            for _chunk in E3Irreps(self.output_graph_irreps).slices()
        ]

    def __call__(
        self,
        cg: CrystalGraphs,
        ctx: Context,
        global_feats: Float[Array, 'graphs latent'] | None = None,
    ):
        vecs = edge_vecs(cg).astype(jnp.bfloat16)

        if global_feats is None:
            extra_node_feats = None
        else:
            extra_node_feats = global_feats[cg.nodes.graph_i]

        # shape [n_nodes, n_interactions, output_irreps]
        out = self.mace(
            vecs,
            cg.nodes.species,
            cg.receivers,
            ctx=ctx,
            extra_node_features=extra_node_feats,
        )

        def collect_chunk(x, i):
            filtered_outs = x
            if self.interaction_reduction == 'last':
                filtered_outs = filtered_outs[:, -1, :, :]
            else:
                filtered_outs = EinsOp(
                    'nodes blocks mul outs -> nodes mul outs', reduce=self.interaction_reduction
                )(filtered_outs)

            if i < len(self.node_reduction_mods):
                # part of the global outputs
                return self.node_reduction_mods[i](
                    filtered_outs, cg.nodes.graph_i, cg.n_total_graphs, ctx
                )
            else:
                # global output
                return filtered_outs

        chunks = [collect_chunk(chunk, i) for i, chunk in enumerate(out.chunks)]

        out_ir = out.irreps

        if self.output_graph_irreps is None:
            graph_arr = None
        else:
            graph_arr = e3nn.IrrepsArray.from_list(
                self.output_graph_irreps,
                chunks[: len(self.node_reduction_mods)],
                leading_shape=(cg.n_total_graphs,),
            )

        if self.output_node_irreps is None:
            node_arr = None
        else:
            node_arr = e3nn.IrrepsArray.from_list(
                self.output_node_irreps,
                chunks[len(self.node_reduction_mods) :],
                leading_shape=(cg.n_total_nodes,),
            )

        if out_ir.is_scalar() and out_ir.num_irreps == 1:
            # special case for regression
            return graph_arr.array * self.scalar_std + self.scalar_mean
        else:
            return graph_arr, node_arr

    @property
    def output_irreps(self) -> str:
        if self.output_node_irreps is None:
            return self.output_graph_irreps
        elif self.output_graph_irreps is None:
            return self.output_node_irreps
        else:
            return f'{self.output_graph_irreps} + {self.output_node_irreps}'
