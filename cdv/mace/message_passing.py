from typing import Callable
from flax import linen as nn
from cdv.mace.e3_layers import E3Irreps, E3IrrepsArray, IrrepsModule, Linear
import jax.numpy as jnp
import e3nn_jax as e3nn
from cdv.layers import Context, LazyInMLP


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: E3Irreps
    max_ell: int

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

        radial = LazyInMLP(
            [],
            out_dim=messages.irreps.num_irreps,
            normalization='none',
            name='radial_mix',
        )(radial_embedding, ctx)  # [n_nodes, k, num_irreps]

        # debug_structure(messages=messages, mix=radial, rad=radial_embedding.array)
        # debug_stat(messages=messages.array, mix=mix.array, rad=radial_embedding.array)
        # radial = nn.sigmoid(radial.array)
        # radial = nn.tanh(radial.array)
        messages = messages * radial  # [n_nodes, k, irreps]

        # debug_stat(messages=messages, radial=radial)

        zeros = E3IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        # TODO flip this perhaps?
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats


class InteractionBlock(IrrepsModule):
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
        new_node_feats = self.conv(vectors, new_node_feats, radial_embedding, receivers, ctx)
        new_node_feats = Linear(self.conv.target_irreps, name='linear_down')(new_node_feats)

        if new_node_feats.irreps == node_feats.irreps:
            node_feats = new_node_feats + node_feats
        else:
            node_feats = new_node_feats

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]
