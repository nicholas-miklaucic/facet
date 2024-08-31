from typing import Callable
from flax import linen as nn
import jax.experimental
from cdv.mace.e3_layers import E3Irreps, E3IrrepsArray, IrrepsModule, Linear
import jax.numpy as jnp
from e3nn_jax.legacy import FunctionalTensorProduct
import e3nn_jax as e3nn
from cdv.layers import Context, LazyInMLP
from cdv.utils import debug_stat, debug_structure
import jax
import functools as ft
import operator


class MPConv(IrrepsModule):
    avg_num_neighbors: float
    max_ell: int

    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
    ) -> E3IrrepsArray:
        raise NotImplementedError


class SevenNetConv(MPConv):
    """
    SevenNet convolution: like NequIP, but without the per-element weights.

    Code 'based on' Deepmind's implementation:
    https://github.com/google-deepmind/materials_discovery/blob/7372c3a2f90d480e62d61ec386451a321045cea7/model/nequip.py#L77

    Their sources, in their words:

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.
    """

    radial_weight: LazyInMLP

    # @nn.compact
    # def __call__(
    #     self,
    #     node_features: IrrepsArray,
    #     node_attributes: IrrepsArray,
    #     edge_sh: Array,
    #     edge_src: Array,
    #     edge_dst: Array,
    #     edge_embedded: Array,
    # ) -> IrrepsArray:

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: E3IrrepsArray,  # [n_nodes, k, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
    ) -> E3IrrepsArray:
        # Nequip outline, just the parts we're doing here:
        # TP + aggregate
        # divide by average number of neighbors
        # Concatenation

        # edge_sh is not given in this interface: perhaps that should be cached? is it memory-bound
        # or FLOP-bound?
        edge_sh = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.max_ell, p=-1), vectors, normalize=True
        )

        # map node features onto edges for tp
        edge_features = node_feats  # [n_nodes, irreps]

        # we gather the instructions for the tp as well as the tp output irreps
        mode = 'uvu'
        trainable = True
        irreps_after_tp = []
        instructions = []

        # iterate over both arguments, i.e. node irreps and edge irreps
        # if they have a valid TP path for any of the target irreps,
        # add to instructions and put in appropriate position
        # we use uvu mode (where v is a single-element sum) and weights will
        # be provide externally by the scalar MLP
        # this triple for loop is similar to the one used in e3nn and nequip
        for i, (mul_in1, irreps_in1) in enumerate(node_feats.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in self.ir_out:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # we will likely have constructed irreps in a non-l-increasing order
        # so we sort them to be in a l-increasing order
        irreps_after_tp, p, _inv = E3Irreps(irreps_after_tp).sort()

        # if we sort the target irreps, we will have to sort the instructions
        # acoordingly, using the permutation indices
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        # TP between spherical harmonics embedding of the edge vector
        # Y_ij(\hat{r}) and neighboring node h_j, weighted on a per-element basis
        # by the radial network R(r_ij)
        tp = FunctionalTensorProduct(
            irreps_in1=edge_features.irreps,
            irreps_in2=edge_sh.irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
        )

        # scalar radial network, number of output neurons is the total number of
        # tensor product paths, nonlinearity must have f(0)=0 and MLP must not
        # have biases
        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += ft.reduce(operator.mul, ins.path_shape, 1)

        # build radial MLP R(r) that maps from interatomic distances to TP weights
        # must not use bias to that R(0)=0
        # fc = nn_util.MLP(
        #     (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
        #     self.radial_net_nonlinearity,
        #     use_bias=False,
        #     scalar_mlp_std=self.scalar_mlp_std,
        # )

        fc: LazyInMLP = self.radial_weight.copy(out_dim=n_tp_weights)

        # the TP weights (v dimension) are given by the FC
        weight = fc(radial_embedding.array, ctx=ctx)

        # debug_structure(weight=weight, edge_features=edge_features, sh=edge_sh)

        # tp between node features that have been mapped onto edges and edge RSH
        # weighted by FC weight, we vmap over the dimension of the edges
        edge_features = e3nn.vmap(e3nn.vmap(tp.left_right, in_axes=(0, None, 0)))(
            weight, edge_features, edge_sh
        )
        # TODO: It's not great that e3nn_jax automatically upcasts internally,
        # but this would need to be fixed at the e3nn level.
        edge_features = jax.tree.map(lambda x: x.astype(node_feats.dtype), edge_features)

        h = node_feats
        # aggregate edges onto nodes after tp using e3nn-jax's index_add
        h_type = h.dtype

        e = edge_features.remove_zero_chunks().simplify()
        h = e3nn.index_add(receivers, e, out_dim=h.shape[0])
        h = h.astype(h_type)

        # normalize by the average (not local) number of neighbors
        h = h / self.avg_num_neighbors

        return h


class SimpleMixMLPConv(MPConv):
    radial_mix: LazyInMLP

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
        messages_broadcast = node_feats[
            jnp.repeat(jnp.arange(node_feats.shape[0])[..., None], 16, axis=-1)
        ]
        # debug_structure(msgs=messages, vecs=vectors)

        inner_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)

        msg_prefix = messages_broadcast.filter(inner_irreps)
        vec_harms = e3nn.tensor_product(
            messages_broadcast,
            e3nn.spherical_harmonics(range(1, self.max_ell + 1), vectors, True),
            filter_ir_out=inner_irreps,
        )

        # debug_structure(
        #     msg=messages_broadcast, vecs=vectors, msg_pref=msg_prefix, vec_harm=vec_harms
        # )

        messages = e3nn.concatenate(
            [msg_prefix, vec_harms],
            axis=-1,
        ).regroup()  # [n_nodes, irreps]

        radial = self.radial_mix.copy(out_dim=messages.irreps.num_irreps)(
            radial_embedding, ctx
        )  # [n_nodes, k, num_irreps]

        # debug_structure(messages=messages, mix=radial, rad=radial_embedding.array)
        # debug_stat(messages=messages.array, mix=mix.array, rad=radial_embedding.array)
        # radial = nn.sigmoid(radial.array)
        # radial = nn.tanh(radial.array)
        messages = messages * radial  # [n_nodes, k, irreps]

        # debug_stat(messages=messages, radial=radial)

        zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        # TODO flip this perhaps?
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
        node_feats = node_feats / self.avg_num_neighbors

        return node_feats


class SimpleInteraction(IrrepsModule):
    conv: MPConv

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
        new_node_feats = Linear(node_feats.irreps, name='linear_intro')(node_feats)
        new_node_feats = self.conv.copy(irreps_out=self.ir_out)(
            vectors, new_node_feats, radial_embedding, receivers, ctx
        )
        new_node_feats = Linear(self.ir_out, name='linear_outro', force_irreps_out=True)(
            new_node_feats
        )

        return new_node_feats  # [n_nodes, target_irreps]


# class InteractionBlock(IrrepsModule):
#     conv: MessagePassingConvolution

#     @nn.compact
#     def __call__(
#         self,
#         vectors: E3IrrepsArray,  # [n_edges, 3]
#         node_feats: E3IrrepsArray,  # [n_nodes, irreps]
#         radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
#         receivers: jnp.ndarray,  # [n_edges, ]
#         ctx: Context,
#     ) -> tuple[E3IrrepsArray, E3IrrepsArray]:
#         """-> n_nodes irreps"""
#         # assert node_feats.ndim == 2
#         # assert vectors.ndim == 2
#         # assert radial_embedding.ndim == 2
#         # new_node_feats = Linear(node_feats.irreps, name='linear_up')(node_feats)
#         new_node_feats = node_feats
#         new_node_feats = self.conv(vectors, new_node_feats, radial_embedding, receivers, ctx)
#         new_node_feats = Linear(self.ir_out, name='linear_down')(new_node_feats)

#         if new_node_feats.irreps == node_feats.irreps:
#             node_feats = new_node_feats + node_feats
#         else:
#             node_feats = new_node_feats

#         assert node_feats.ndim == 2
#         return node_feats  # [n_nodes, target_irreps]

if __name__ == '__main__':
    import jax.random as jr

    mod = SevenNetConv('64x0e + 32x1o + 16x2e', 14, 2, LazyInMLP(inner_dims=[]))

    rng = jr.key(29205)

    vecs = e3nn.normal('1o', rng, (128, 16))
    node_feats = e3nn.normal('64x0e + 32x1o + 16x2e', rng, (128,))
    receivers = jr.randint(rng, (128, 16), 0, 128)
    radial_embedding = e3nn.normal('10x0e', rng, (128, 16))

    out, params = mod.init_with_output(
        rng,
        vectors=vecs,
        node_feats=node_feats,
        receivers=receivers,
        radial_embedding=radial_embedding,
        ctx=Context(training=True),
    )

    debug_structure(out)
