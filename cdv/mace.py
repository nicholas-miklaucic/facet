"""
MACE network code. Adapted from https://github.com/ACEsuit/mace-jax.
"""

from collections.abc import Sequence
import functools
import math
from typing import Callable, Optional
from typing import Set, Union
from flax import linen as nn
from flax import struct
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int

from eins import EinsOp
import numpy as np

from cdv.databatch import CrystalGraphs
from cdv.gnn import SegmentReduction, SegmentReductionKind
from cdv.layers import Context, E3NormNorm, Identity, LazyInMLP, E3Irreps, E3IrrepsArray, edge_vecs
from cdv.utils import debug_stat, debug_structure, flax_summary, ELEM_VALS


def Linear(*args, **kwargs):
    # return nn.WeightNorm(e3nn.flax.Linear(*args, **kwargs))
    return e3nn.flax.Linear(*args, **kwargs)


class LinearNodeEmbedding(nn.Module):
    num_species: int
    element_indices: Int[Array, 'max_species']
    irreps_out: E3Irreps

    def setup(self):
        def skipatom_init(key, shape, dtype):
            import pickle
            import pandas as pd

            with open('data/mp_2020_10_09.dim250.keras.model', 'rb') as fin:
                data = pickle.load(fin)

            atoms = list(
                pd.read_csv(
                    'https://raw.githubusercontent.com/lantunes/skipatom/main/data/atoms.txt',
                    header=None,
                ).values.reshape(-1)
            )

            embed_i = list(map(atoms.index, ELEM_VALS))
            embeds = jnp.array(data[embed_i], dtype=dtype)
            curr_dim = embeds.shape[-1]
            *batch, out_dim = shape

            pad_shape = (*batch, out_dim - curr_dim)

            return jnp.concat((embeds, jax.random.normal(key, pad_shape, dtype) * 0.01), axis=-1)

        self.irreps_out_calc = E3Irreps(self.irreps_out).filter('0e').regroup()
        self.out_dim = E3Irreps(self.irreps_out_calc).dim
        self.embed = nn.Embed(
            self.num_species,
            self.out_dim,
            # embedding_init=skipatom_init
        )

    def __call__(self, node_species: Int[Array, ' batch']) -> E3IrrepsArray:
        return E3IrrepsArray(self.irreps_out_calc, self.embed(self.element_indices[node_species]))


class LinearReadoutBlock(nn.Module):
    output_irreps: E3Irreps

    @nn.compact
    def __call__(self, x: E3IrrepsArray) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        return Linear(self.output_irreps)(x)


class NonLinearReadoutBlock(nn.Module):
    hidden_irreps: E3Irreps
    output_irreps: E3Irreps
    activation: Optional[Callable] = None
    gate: Optional[Callable] = None

    @nn.compact
    def __call__(self, x: E3IrrepsArray) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        hidden_irreps = E3Irreps(self.hidden_irreps)
        output_irreps = E3Irreps(self.output_irreps)
        num_vectors = hidden_irreps.filter(
            drop=['0e', '0o']
        ).num_irreps  # Multiplicity of (l > 0) irreps
        # print(x.irreps)
        x = Linear(
            (hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify(),
        )(x)
        # print((hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify())
        # print(x.irreps)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return Linear(output_irreps)(x)

    # [n_nodes, output_irreps]


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


A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]

import functools as ft

reduced_symmetric_tensor_product_basis = e3nn.reduced_symmetric_tensor_product_basis
reduced_tensor_product_basis = e3nn.reduced_tensor_product_basis
# e3nn.reduced_symmetric_tensor_product_basis


class SymmetricContraction(nn.Module):
    correlation: int
    keep_irrep_out: Union[Set[e3nn.Irrep], str]
    num_species: int
    gradient_normalization: Union[str, float, None] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @nn.compact
    def __call__(
        self,
        input: E3IrrepsArray,  # n_nodes, feats, irreps
        index: jnp.ndarray,
        ctx: Context,
        species_embed: Float[Array, 'num_species embed_dim'] | None = None,
    ) -> E3IrrepsArray:
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config('gradient_normalization')
        if isinstance(gradient_normalization, str):
            gradient_normalization = {'element': 0.0, 'path': 1.0}[gradient_normalization]

        if isinstance(self.keep_irrep_out, str):
            keep_irrep_out = E3Irreps(self.keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)
        else:
            keep_irrep_out = self.keep_irrep_out

        keep_irrep_out_set = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        W = {}

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        num_rbf = 32

        use_half = True
        dtype = jnp.bfloat16 if use_half else jnp.float32

        if species_embed is None:
            species_embed = nn.Embed(
                self.num_species, num_rbf, name='species_embed', param_dtype=dtype
            )
            # species_ind = index
            species_ind = species_embed(index)
        else:
            species_embed_mlp = LazyInMLP(
                [], out_dim=num_rbf, name='species_radial_mlp', normalization='layer'
            )
            species_embed = species_embed_mlp(
                species_embed.astype(jnp.bfloat16) if use_half else species_embed, ctx=ctx
            )
            species_ind = species_embed[index]

        # print(input.shape, index.shape)

        for order in range(self.correlation, 0, -1):  # correlation, ..., 1
            if self.symmetric_tensor_product_basis:
                U = reduced_symmetric_tensor_product_basis(
                    input.irreps, order, keep_ir=keep_irrep_out_set
                )
            else:
                U = reduced_tensor_product_basis([input.irreps] * order, keep_ir=keep_irrep_out_set)
            # U = U / order  # normalization TODO(mario): put back after testing
            # NOTE(mario): The normalization constants (/order and /mul**0.5)
            # has been numerically checked to be correct.

            # TODO(mario) implement norm_p

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            for (mul, ir_out), u in zip(U.irreps, U.list):
                name = f'w{order}_{ir_out}'
                W[name] = self.param(
                    name,
                    # nn.initializers.normal(stddev=(mul**-0.5) ** (1.0 - gradient_normalization)),
                    nn.initializers.normal(
                        stddev=1, dtype=jnp.bfloat16 if use_half else jnp.float32
                    ),
                    (num_rbf, mul, input.shape[-2]),
                )

        # - This operation is parallel on the feature dimension (but each feature has its own parameters)
        # This operation is an efficient implementation of
        # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
        # up to x power self.correlation
        # assert input.ndim == 2  # [num_features, irreps_x.dim]
        # assert index.ndim == 0  # int

        out = dict()

        for order in range(self.correlation, 0, -1):  # correlation, ..., 1
            if self.off_diagonal:
                roll = lambda arr: jnp.roll(arr, A025582[order - 1])
                x_ = jax.vmap(roll)(input.array)
            else:
                x_ = input.array

            if self.symmetric_tensor_product_basis:
                U = reduced_symmetric_tensor_product_basis(
                    input.irreps, order, keep_ir=keep_irrep_out_set
                )
            else:
                U = reduced_tensor_product_basis([input.irreps] * order, keep_ir=keep_irrep_out_set)
            # U = U / order  # normalization TODO(mario): put back after testing
            # NOTE(mario): The normalization constants (/order and /mul**0.5)
            # has been numerically checked to be correct.

            # TODO(mario) implement norm_p

            # ((w3 x + w2) x + w1) x
            #  \-----------/
            #       out

            einsum_kwargs = {
                'precision': jax.lax.Precision.DEFAULT,
                'preferred_element_type': jnp.bfloat16 if use_half else jnp.float32,
            }
            if use_half:
                x_ = x_.astype(jnp.bfloat16)

            for (mul, ir_out), u in zip(U.irreps, U.list):
                u = u.astype(x_.dtype)
                # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                # print(self)
                name = f'w{order}_{ir_out}'
                w = jnp.einsum('be,e...->b...', species_ind, W[name], **einsum_kwargs)

                # w = W[name][species]

                # [multiplicity, num_features]

                # w = w * (mul**-0.5) ** gradient_normalization  # normalize weights

                if ir_out not in out:
                    # debug_structure(u=u, w=w, x=x_)
                    out[ir_out] = (
                        'special',
                        jnp.einsum('...jki,bkc,bcj->bc...i', u, w, x_, **einsum_kwargs),
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                else:
                    out[ir_out] += jnp.einsum(
                        '...ki,bkc->bc...i', u, w, **einsum_kwargs
                    )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \----------------/
            #         out (in the normal case)

            for ir_out in out:
                if isinstance(out[ir_out], tuple):
                    out[ir_out] = out[ir_out][1]
                    continue  # already done (special case optimization above)

                out[ir_out] = jnp.einsum(
                    'bc...ji,bcj->bc...i', out[ir_out], x_, **einsum_kwargs
                )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

            # ((w3 x + w2) x + w1) x
            #  \-------------------/
            #           out

        # out[irrep_out] : [num_features, ir_out.dim]
        irreps_out = E3Irreps(sorted(out.keys()))
        # for k, v in out.items():
        #     debug_structure(**{str(k): v})
        return E3IrrepsArray.from_list(
            irreps_out,
            [out[ir][..., None, :] for (_, ir) in irreps_out],
            input.shape[:-1],
        )


class EquivariantProductBasisBlock(nn.Module):
    target_irreps: E3Irreps
    correlation: int
    num_species: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @nn.compact
    def __call__(
        self,
        node_feats: E3IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
        ctx: Context,
        species_embed: Float[Array, 'num_species embed_dim'] | None = None,
    ) -> E3IrrepsArray:
        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=self.correlation,
            num_species=self.num_species,
            gradient_normalization='element',  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats, node_specie, ctx=ctx, species_embed=species_embed)
        node_feats = node_feats.axis_to_mul()
        return Linear(self.target_irreps, name='proj_out')(node_feats)


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
        radial_embedding: jnp.ndarray,  # [n_nodes, k, radial_embedding_dim]
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

        # one = E3IrrepsArray.ones("0e", edge_attrs.shape[:-1])
        # messages = e3nn.tensor_product(
        #     messages, e3nn.concatenate([one, edge_attrs.filter(drop="0e")])
        # ).filter(self.target_irreps)

        # mix = LazyInMLP(
        #     np.rint(np.linspace(radial_embedding.shape[-1], messages.irreps.num_irreps, 3)[1:-1])
        #     .astype(int)
        #     .tolist(),
        #     out_dim=messages.irreps.num_irreps,
        #     inner_act=self.activation,
        #     dropout_rate=0.2,
        # )(radial_embedding, ctx)  # [n_edges, num_irreps]

        if self.mix == 'mix':
            radial = LazyInMLP(
                # np.rint(np.linspace(radial_embedding.shape[-1], messages.irreps.num_irreps, 3)[1:-1])
                # .astype(int)
                # .tolist(),
                [],
                out_dim=messages.irreps.num_irreps,
                inner_act=self.activation,
                normalization='none',
                name='radial_mix',
            )(radial_embedding, ctx)  # [n_edges, num_irreps]

            # debug_structure(messages=messages, mix=radial, rad=radial_embedding.array)
            # debug_stat(messages=messages.array, mix=mix.array, rad=radial_embedding.array)
            radial = nn.LayerNorm(param_dtype=radial.dtype)(radial.array)
            messages = messages * radial  # [n_nodes, k, irreps]

            # debug_structure(messages=messages)

            zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
            # TODO flip this perhaps?
            node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
            node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)
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

        node_feats = Linear(node_feats.irreps, name='linear_up')(node_feats)
        # debug_stat(up=node_feats.array)
        node_feats = E3NormNorm()(node_feats)

        node_feats = self.conv(vectors, node_feats, radial_embedding, receivers, ctx)
        node_feats = E3NormNorm()(node_feats)
        # debug_stat(conv=node_feats.array)

        node_feats = Linear(self.conv.target_irreps, name='linear_down')(node_feats)
        node_feats = E3NormNorm()(node_feats)
        # debug_stat(down=node_feats.array)

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]


try:
    from profile_nn_jax import profile
except ImportError:

    def profile(_, x, __=None):
        return x


class MACELayer(nn.Module):
    first: bool
    last: bool
    num_features: int
    interaction_irreps: E3Irreps
    hidden_irreps: E3Irreps
    activation: Callable
    num_species: int
    epsilon: Optional[float]
    name: Optional[str]
    # InteractionBlock:
    max_ell: int
    avg_num_neighbors: float
    # EquivariantProductBasisBlock:
    correlation: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool
    soft_normalization: Optional[float]
    # ReadoutBlock:
    output_irreps: E3Irreps
    readout_mlp_irreps: E3Irreps
    skip_connection_first_layer: bool = False

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

        sc = None
        # if not self.first or self.skip_connection_first_layer:
        #     sc = Linear(
        #         self.num_features * hidden_irreps,
        #         num_indexed_weights=self.num_species,
        #         gradient_normalization='path',
        #         name='skip_tp',
        #     )(node_species, node_feats)  # [n_nodes, feature * hidden_irreps]
        #     sc = E3NormNorm()(sc)
        #     sc = profile(f'{self.name}: self-connexion', sc, node_mask[:, None])

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

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        node_feats = profile(f'{self.name}: interaction', node_feats, node_mask[:, None])

        # if self.first:
        #     # Selector TensorProduct
        #     node_feats = Linear(
        #         self.num_features * interaction_irreps,
        #         num_indexed_weights=self.num_species,
        #         gradient_normalization='path',
        #         name='skip_tp_first',
        #     )(node_species, node_feats)
        #     node_feats = profile(f'{self.name}: skip_tp_first', node_feats, node_mask[:, None])

        node_feats = EquivariantProductBasisBlock(
            target_irreps=self.num_features * hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_species, species_embed=species_embed, ctx=ctx)

        node_feats = profile(f'{self.name}: tensor power', node_feats, node_mask[:, None])

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(node_feats, [phi] * len(node_feats.irreps))

            node_feats = profile(f'{self.name}: soft normalization', node_feats, node_mask[:, None])
        else:
            node_feats = node_feats

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutBlock(output_irreps)(node_feats)  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutBlock(
                readout_mlp_irreps,
                output_irreps,
                activation=self.activation,
            )(node_feats)  # [n_nodes, output_irreps]

        node_outputs = profile(f'{self.name}: output', node_outputs, node_mask[:, None])
        return node_outputs, node_feats


def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


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
    # Number of zero derivatives at small and large distances, default 4 and 2
    # If both are None, it uses a smooth C^inf envelope function
    max_ell: int = 3  # Max spherical harmonic degree, default 3
    epsilon: Optional[float] = None
    correlation: int = 3  # Correlation order at each layer (~ node_features^correlation), default 3
    gate: Callable = jax.nn.silu  # activation function
    soft_normalization: Optional[float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    interaction_irreps: Union[str, E3Irreps] = 'o3_restricted'  # or o3_full
    node_embedding_type: type[nn.Module] = LinearNodeEmbedding
    share_species_embed: bool = True
    skip_connection_first_layer: bool = False
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: Optional[int] = None

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
                    epsilon=self.epsilon,
                    correlation=self.correlation,
                    output_irreps=self.output_irreps_calc,
                    readout_mlp_irreps=self.readout_mlp_irreps_calc,
                    symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                    off_diagonal=self.off_diagonal,
                    soft_normalization=self.soft_normalization,
                    skip_connection_first_layer=self.skip_connection_first_layer,
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
                species_embed=node_feats.array if self.share_species_embed else None,
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
    interaction_reduction: str = 'mean'
    # Node reduction.
    node_reduction: SegmentReductionKind = 'mean'

    num_radial_embeds: int = 10
    max_r: float = 7.0
    avg_r_min: float = 1.0
    radial_envelope_scale: float = 2
    radial_envelope_intercept: float = 1.2

    avg_num_neighbors: float = 20.0

    max_ell: int = 3  # Max spherical harmonic degree, default 3
    correlation: int = 2  # Correlation order at each layer (~ node_features^correlation), default 3

    # If a float, soft norms values to stay in this range.
    soft_normalization: Optional[float] = None

    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    interaction_irreps: Union[str, E3Irreps] = 'o3_restricted'  # or o3_full
    skip_connection_first_layer: bool = False
    # Number of features per node, default gcd of hidden_irreps multiplicities
    num_features: Optional[int] = None

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
            num_features=self.num_features,
            max_ell=self.max_ell,
            epsilon=None,
            correlation=self.correlation,
            gate=nn.tanh,
            soft_normalization=self.soft_normalization,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
            interaction_irreps=self.interaction_irreps,
            node_embedding_type=LinearNodeEmbedding,
            skip_connection_first_layer=self.skip_connection_first_layer,
            radial_basis=bessel_basis,
            radial_envelope=soft_envelope,
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


if __name__ == '__main__':
    from cdv.config import MainConfig
    import pyrallis
    from cdv.dataset import load_file, dataloader
    from eins import EinsOp

    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    symmetric_tensor_product_basis = True  # symmetric is slightly worse but slightly faster
    max_ell = 2
    num_interactions = 2
    hidden_irreps = '256x0e + 256x1o'
    # hidden_irreps = '16x0e + 16x1o'
    correlation = 2  # 4 is better but 5x slower
    readout_mlp_irreps = '128x0e + 16x1o'
    output_irreps = '1x0e'

    mace = MaceModel(
        output_irreps=output_irreps,
        num_interactions=num_interactions,
        hidden_irreps=hidden_irreps,
        readout_mlp_irreps=readout_mlp_irreps,
        num_species=config.data.num_species,
        max_ell=max_ell,
        correlation=correlation,
    )

    cg = load_file(config, 20)

    key = jax.random.key(12345)
    ctx = Context(training=True)

    # with jax.check_tracer_leaks(True):
    # flax_summary(mace, cg=cg, ctx=ctx)

    with jax.debug_nans(False):
        with jax.log_compiles(False):
            contributions, params = mace.init_with_output(key, cg=cg, ctx=ctx)

    # debug_structure(contributions)
    # debug_stat(contributions)
    # debug_stat(params)

    # steps_per_epoch, dl = dataloader(config, split='train', infinite=True)

    @jax.jit
    def loss(params, batch):
        cg = batch
        preds = mace.apply(params, cg=cg, ctx=Context(training=True))
        return config.train.loss.regression_loss(
            preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask
        )

    res = jax.value_and_grad(loss)(params, cg)
    debug_stat(res)
    # debug_structure(res)
    # jax.block_until_ready(res)

    # from ctypes import cdll
    # libcudart = cdll.LoadLibrary('libcudart.so')

    # libcudart.cudaProfilerStart()
    # for i in range(1):
    #     batch = next(dl)
    #     res = jax.value_and_grad(loss)(params, batch)
    #     jax.block_until_ready(res)
    # libcudart.cudaProfilerStop()

    # debug_stat(value=res[0], grad=res[1])

    # debug_structure(batch=batch, module=model, out=out)
    # debug_stat(batch=batch, module=model, out=out)
    # flax_summary(mace, vecs, cg.nodes.species, cg.edges.sender, cg.edges.receiver)
