"""Functionality for graph neural networks."""

from gc import garbage
from typing import Callable, Literal, Optional
import eins.reduction
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, Bool
import eins
from eins import Reductions as R
from eins.reduction import Reduction
from eins import EinsOp
import pyrallis


from cdv.databatch import CrystalGraphs
from cdv.dataset import dataloader, load_file
from cdv.layers import (
    Context,
    Identity,
    LazyInMLP,
    GaussBasis,
    DistanceEncoder,
    OldBessel1DBasis,
    Bessel2DBasis,
    Envelope,
)
from cdv.utils import debug_stat, debug_structure, flax_summary


class Graphs(struct.PyTreeNode):
    """Generic graphs."""

    node_emb: Float[Array, 'nodes node_emb']
    incoming: Int[Array, 'nodes max_in']
    incoming_pad: Bool[Array, 'nodes max_in']
    outgoing: Int[Array, 'nodes max_out']
    outgoing_pad: Bool[Array, 'nodes max_out']
    carts: Float[Array, 'nodes 3']
    dists: Float[Array, 'edges']
    vecs: Float[Array, 'edges 3']
    senders: Int[Array, 'edges']
    receivers: Int[Array, 'edges']
    edge_emb: Float[Array, 'edges edge_emb']
    graph_emb: Float[Array, 'graphs graph_emb']
    n_nodes: Int[Array, 'graphs']
    n_edges: Int[Array, 'graphs']
    node_graph_i: Int[Array, 'nodes']
    edge_graph_i: Int[Array, 'edges']
    padding_mask: Bool[Array, 'graphs']

    @property
    def n_total_nodes(self) -> int:
        return len(self.node_graph_i)

    @property
    def n_total_edges(self) -> int:
        return len(self.edge_graph_i)

    @property
    def n_total_graphs(self) -> int:
        return len(self.n_nodes)


class SpeciesEmbedding(nn.Module):
    """Species embedding for the nodes."""

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'nodes node_emb']:
        """Embeds the nodes."""
        raise NotImplementedError


class LearnedSpecEmb(nn.Module):
    """Normal trainable species embedding."""

    num_specs: int
    embed_dim: int

    def setup(self):
        self.embed = nn.Embed(self.num_specs, self.embed_dim)

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'nodes node_emb']:
        """Embeds the nodes."""
        return self.embed(cg.nodes.species)


class InputEncoder(nn.Module):
    """Converts crystal graphs into generic graphs by encoding distances and species."""

    distance_enc: DistanceEncoder
    distance_projector: nn.Module
    species_emb: SpeciesEmbedding

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Graphs:
        send_pos = cg.nodes.cart[cg.senders]
        offsets = EinsOp('e abc xyz, e abc -> e xyz')(
            cg.graph_data.lat[cg.edges.graph_i], cg.edges.to_jimage
        )
        recv_pos = cg.nodes.cart[cg.receivers] + offsets

        vecs = recv_pos - send_pos

        dist = EinsOp('edge 3 -> edge', reduce='l2_norm')(vecs)
        # debug_stat(dist=dist)
        dist_emb = self.distance_projector(self.distance_enc(dist, ctx))
        # debug_structure(dist=dist, dist_emb=dist_emb)

        node_emb = self.species_emb(cg, ctx)

        return Graphs(
            node_emb=node_emb,
            carts=cg.nodes.cart,
            vecs=vecs,
            dists=dist,
            incoming=cg.nodes.incoming,
            outgoing=cg.nodes.outgoing,
            incoming_pad=cg.nodes.incoming_pad,
            outgoing_pad=cg.nodes.outgoing_pad,
            senders=cg.senders,
            receivers=cg.receivers,
            edge_emb=dist_emb,
            graph_emb=jnp.zeros((cg.n_total_graphs, 0)),
            n_nodes=cg.n_node,
            n_edges=cg.n_edge,
            node_graph_i=cg.nodes.graph_i,
            edge_graph_i=cg.edges.graph_i,
            padding_mask=cg.padding_mask,
        )


# def angle(a, b, c):
#     """Gets angle ABC."""
#     ab = a - b
#     cb = c - b

#     ab = ab / (jnp.linalg.norm(ab) + 1e-8)
#     cb = cb / (jnp.linalg.norm(ab) + 1e-8)

#     return jnp.arccos(jnp.dot(ab, cb))

SegmentReductionKind = Literal['max', 'min', 'prod', 'sum', 'mean']


def segment_mean(data, segment_ids, **kwargs):
    return jax.ops.segment_sum(data, segment_ids, **kwargs) / (
        1e-6 + jax.ops.segment_sum(jnp.ones_like(data), segment_ids, **kwargs)
    )


def segment_reduce(reduction: SegmentReductionKind, data, segment_ids, **kwargs):
    try:
        fn = getattr(jax.ops, f'segment_{reduction}')
    except AttributeError:
        if reduction == 'mean':
            fn = segment_mean
        else:
            raise ValueError('Cannot find reduction')

    return fn(data, segment_ids, **kwargs)


class SegmentReduction(nn.Module):
    reduction: SegmentReductionKind = 'sum'

    def __call__(self, data, segments, num_segments, ctx):
        return segment_reduce(self.reduction, data, segments, num_segments=num_segments)


class Fishnet(SegmentReduction):
    net_templ: LazyInMLP = LazyInMLP([])
    inner_dim: Optional[int] = None
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, data, segments, num_segments, ctx):
        if self.reduction not in ('sum', 'mean'):
            raise ValueError('Invalid reduction for fishnet')

        out_dim = self.out_dim or data.shape[-1]
        inner_dim = self.inner_dim or out_dim

        # if out_dim = o
        # score has o elements
        # F is o x o, but the Cholesky decomposition is just o(o + 1)/2
        net_out_dim = (inner_dim * (inner_dim + 1)) // 2 + inner_dim

        net = self.net_templ.copy(out_dim=net_out_dim, kernel_init=nn.initializers.normal())

        net_output = net(data, ctx)

        # split into t and F_chol
        t = net_output[..., :inner_dim]
        F_chol = net_output[..., inner_dim:]

        def un_chol(y):
            L = jnp.zeros((inner_dim, inner_dim), dtype=y.dtype)
            L = L.at[jnp.tril_indices_from(L)].set(jax.nn.tanh(y))
            # softplus to ensure these are positive
            diag = jnp.diag_indices_from(L)
            L = L.at[diag].set(jax.nn.softplus(L[diag]))
            return L @ L.T

        F = jax.vmap(un_chol)(F_chol.reshape(-1, F_chol.shape[-1]))
        F = F.reshape(*F_chol.shape[:-1], inner_dim, inner_dim)

        F_reduced = SegmentReduction(self.reduction)(F, segments, num_segments, ctx)
        t_reduced = SegmentReduction(self.reduction)(t, segments, num_segments, ctx)

        # debug_structure(F=F_reduced)
        # debug_stat(F=F_reduced)

        # empty segments will have all 0s, which will cause a divide by 0
        # because any all-0 will also have all-0 score, the multiplication will reset them, so we
        # just need to set them to anything that won't error

        missing = (jnp.abs(t_reduced) < 0.1).all(axis=-1) & (jnp.abs(F_reduced) < 0.1).all(
            axis=(-2, -1)
        )

        # There are a lot of numerical stabilty problems here. If the inverse is too small, the
        # gradient is all over the place. Adding 1 to the diagonals solves this, although it doesn't
        # really do what it's supposed to any more.

        # F_masked = F_reduced + missing[:, None, None] * jnp.eye(inner_dim, inner_dim)
        F_masked = F_reduced + jnp.eye(inner_dim, inner_dim)

        # print(F_masked.sum(axis=(-1, -2)))
        # print(missing)

        F_inv = jax.vmap(jnp.linalg.inv)(F_masked)
        # F_inv = F_masked

        # debug_stat(F=F_reduced, F_m=F_masked, F_inv=F_inv)

        theta = jnp.einsum('...ij,...j->...j', F_inv, t_reduced)

        if inner_dim != out_dim:
            theta = nn.Dense(out_dim, use_bias=False, name='proj')(theta)

        return theta


class ProcessingBlock(nn.Module):
    """Block that processes graphs."""

    def __call__(self, g: Graphs, ctx: Context) -> Graphs:
        raise NotImplementedError


class MessagePassing(ProcessingBlock):
    """Message passing block."""

    # How to reduce nodes. We're limited by Jax here, for the time being.
    node_reduction: SegmentReduction

    def node_update(
        self, node: Float[Array, 'node_emb'], message: Float[Array, 'msg_dim'], ctx: Context
    ) -> Float[Array, 'node_emb']:
        """Updates the node information using the reduced message."""
        raise NotImplementedError

    def message(
        self,
        edge: Float[Array, 'edge_emb'],
        sender: Float[Array, 'node_emb'],
        receiver: Float[Array, 'node_emb'],
        ctx: Context,
    ) -> Float[Array, 'msg_dim']:
        """Computes the message for a given edge."""
        raise NotImplementedError

    def __call__(self, g: Graphs, ctx: Context) -> Graphs:
        edge_messages = jax.vmap(self.message, in_axes=(0, 0, 0, None))(
            g.edge_emb, g.node_emb[g.senders], g.node_emb[g.receivers], ctx
        )  # shape: edges msg_dim

        # aggregate incoming messages for each node
        # shape nodes msg_dim
        node_messages = self.node_reduction(
            edge_messages, g.receivers, num_segments=g.n_total_nodes, ctx=ctx
        )

        node_emb = jax.vmap(self.node_update, in_axes=(0, 0, None))(g.node_emb, node_messages, ctx)
        return g.replace(node_emb=node_emb)


class MLPMessagePassing(MessagePassing):
    """Message passing using MLPs."""

    node_emb: int
    msg_dim: int
    msg_templ: LazyInMLP
    node_templ: LazyInMLP

    def setup(self):
        self.msg = self.msg_templ.copy(out_dim=self.msg_dim, name='msg')
        self.node_mlp = self.node_templ.copy(out_dim=self.node_emb, name='node')

    def node_update(
        self, node: Float[Array, 'node_emb'], message: Float[Array, 'msg_dim'], ctx: Context
    ) -> Float[Array, 'node_emb']:
        """Updates the node information using the reduced message."""
        return node + self.node_mlp(jnp.concat((node, message)), ctx)

    def message(
        self,
        edge: Float[Array, 'edge_emb'],
        sender: Float[Array, 'node_emb'],
        receiver: Float[Array, 'node_emb'],
        ctx: Context,
    ) -> Float[Array, 'msg_dim']:
        """Computes the message for a given edge."""
        return self.msg(jnp.concat((edge, sender, receiver)), ctx)


class Readout(nn.Module):
    """Readout block in GNN."""

    def __call__(self, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        raise NotImplementedError


class NodeAggReadout(Readout):
    """Aggregates node features."""

    head: nn.Module
    graph_reduction: SegmentReduction = SegmentReduction('mean')

    def node_transform(
        self, node: Float[Array, 'node_emb'], ctx: Context
    ) -> Float[Array, 'node_reduce_emb']:
        return node

    def __call__(self, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        transformed = jax.vmap(self.node_transform, in_axes=(0, None))(g.node_emb, ctx)
        graph_embs = self.graph_reduction(
            transformed, g.node_graph_i, num_segments=g.n_total_graphs, ctx=ctx
        )
        return self.head(graph_embs, ctx=ctx)


class TripletAngleEmbedding(nn.Module):
    """
    Generates angle embeddings for each triplet.
    """

    distance_angle_enc: Bessel2DBasis

    @nn.compact
    def __call__(self, g: Graphs, ctx: Context) -> Float[Array, 'nodes in out emb']:
        #     ab = ab / (jnp.linalg.norm(ab) + 1e-8)
        #     cb = cb / (jnp.linalg.norm(ab) + 1e-8)

        #     return jnp.arccos(jnp.dot(ab, cb))

        unit_vecs = g.vecs / (g.dists[:, None] + 1e-8)

        vec_ij = jnp.take(unit_vecs, g.incoming, axis=0)
        vec_ji = -vec_ij
        vec_jk = jnp.take(unit_vecs, g.outgoing, axis=0)

        # angle between vectors is arccosine dot product for unit vecs
        # debug_structure(vec_ji=vec_ji, vec_jk=vec_jk)
        dots = EinsOp('node in 3, node out 3 -> (node in out)')(vec_ji, vec_jk)
        # debug_stat(dots=dots)
        angles = jnp.arccos(jnp.clip(dots, -1, 1))

        d_ij = jnp.take(g.dists, g.incoming).reshape(-1)  # (nodes in)

        s_out = g.outgoing.shape[-1]
        s_nodes, s_in = g.incoming.shape

        angle_embeds = self.distance_angle_enc(
            jnp.repeat(d_ij, s_out), angles, ctx
        )  # (nodes in out) emb
        angle_embeds = angle_embeds.reshape(s_nodes, s_in, s_out, -1)  # nodes in out emb

        return angle_embeds


class DimeNetPPOutput(nn.Module):
    head: LazyInMLP
    edge2node: SegmentReduction = SegmentReduction('mean')
    node2graph: SegmentReduction = SegmentReduction('mean')
    out_dim: int = 1

    @nn.compact
    def __call__(self, base_g: Graphs, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        msg_dim = g.edge_emb.shape[-1]
        dist_proj = nn.Dense(msg_dim, use_bias=False, name='dist_out_proj')

        lin_rbf = dist_proj(base_g.edge_emb)
        edge_msgs = g.edge_emb * lin_rbf
        node_msgs = self.edge2node(edge_msgs, g.receivers, num_segments=g.n_total_nodes, ctx=ctx)

        head = self.head.copy(out_dim=self.out_dim, name='out_head')

        node_outs = head(node_msgs, ctx)
        graph_outs = self.node2graph(
            node_outs, g.node_graph_i, num_segments=g.n_total_graphs, ctx=ctx
        )

        return graph_outs


class DimeNetPP(nn.Module):
    input_enc: InputEncoder
    sbf: TripletAngleEmbedding
    initial_embed_mlp: LazyInMLP
    output: DimeNetPPOutput
    int_dist_enc: LazyInMLP
    int_ang_enc: LazyInMLP
    int_down_proj_mlp: LazyInMLP
    int_up_proj_mlp: LazyInMLP
    int_pre_skip_mlp: LazyInMLP
    int_post_skip_mlp: LazyInMLP

    msg_reduction: SegmentReduction = SegmentReduction('mean')

    act: Callable = nn.sigmoid
    out_dim: int = 1
    initial_embed_distance_dim: int = 64
    message_dim: int = 128
    down_proj_dim: int = 64
    num_interaction_blocks: int = 2

    @nn.compact
    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        g = self.input_enc(cg, ctx)
        a_sbf = self.sbf(g, ctx).astype(jnp.bfloat16)  # nodes in out emb

        # debug_stat(e_rbf=g.edge_emb, a_sbf=a_sbf)

        # first, generate initial message embeddings
        edge_proj = nn.Dense(self.initial_embed_distance_dim, use_bias=False, name='edge_proj')
        # combine e_ij with z_i and z_j
        e_ij = edge_proj(g.edge_emb)
        z_i = jnp.take(g.node_emb, g.senders, axis=0)
        z_j = jnp.take(g.node_emb, g.receivers, axis=0)

        x_ij = jnp.concat((e_ij, z_i, z_j), axis=1)
        init_embed_mlp = self.initial_embed_mlp.copy(out_dim=self.message_dim, name='init_embed')
        m_ij = init_embed_mlp(x_ij, ctx)

        curr_g = g.replace(edge_emb=m_ij)

        output_blocks = []
        for i in range(self.num_interaction_blocks + 1):
            output_blocks.append(self.output.copy(out_dim=self.out_dim, name=f'output_{i}'))

        outs = output_blocks[0](g, curr_g, ctx)

        for i, output_block in enumerate(output_blocks[1:]):
            # interaction
            z_rbf = self.int_dist_enc.copy(out_dim=self.message_dim, name=f'int_dist_enc_{i}')(
                g.edge_emb, ctx
            ).astype(jnp.bfloat16)  # edges msg_dim
            msg_proj = self.act(
                nn.Dense(self.message_dim, name=f'msg_proj_{i}')(curr_g.edge_emb)
            ).astype(jnp.bfloat16)  # edges msg_dim

            # Hadamard multiplication of z_ji and msg_kj
            z_ji = jnp.take(z_rbf, g.outgoing, axis=0)  # nodes out msg_dim
            m_kj = jnp.take(msg_proj, g.incoming, axis=0)  # nodes in msg_dim

            msg_dist = EinsOp('nodes out msg, nodes in msg -> nodes in out msg')(z_ji, m_kj)

            msg_dist_down_proj = self.int_down_proj_mlp.copy(
                out_dim=self.down_proj_dim, name=f'int_down_proj_{i}'
            )(msg_dist, ctx)  # nodes in out down
            z_sbf = self.int_ang_enc.copy(out_dim=self.down_proj_dim, name=f'int_ang_enc_{i}')(
                a_sbf, ctx
            )  # nodes in out down
            z_sbf = (
                z_sbf * g.incoming_pad[..., None, None]
            )  # zero out contributions from padding edges

            msg_all_down = EinsOp('nodes in out down, nodes in out down -> nodes out down')(
                msg_dist_down_proj, z_sbf
            )
            msg_all = self.act(
                nn.Dense(self.message_dim, name=f'int_up_proj_{i}')(msg_all_down)
            )  # nodes out msg_dim
            msg_all = msg_all * g.outgoing_pad[..., None]

            msg_all = EinsOp('nodes out msg_dim -> (nodes out) msg_dim')(msg_all)

            prev_msg_enc = self.act(
                nn.Dense(self.message_dim, name=f'int_prev_msg_{i}')(curr_g.edge_emb)
            )

            combo_msg = prev_msg_enc + self.msg_reduction(
                msg_all, g.outgoing.reshape(-1), num_segments=g.n_total_edges, ctx=ctx
            )

            pre_skip = self.int_pre_skip_mlp.copy(
                out_dim=self.message_dim, name=f'int_pre_skip_{i}'
            )(combo_msg, ctx)

            new_msg = self.int_post_skip_mlp.copy(
                out_dim=self.message_dim, name=f'int_post_skip_{i}'
            )(pre_skip + prev_msg_enc, ctx)

            curr_g = curr_g.replace(edge_emb=new_msg)

            outs = outs + output_block(g, curr_g, ctx)

        return outs


class GN(nn.Module):
    input_enc: InputEncoder
    num_blocks: int
    block_templ: ProcessingBlock
    readout: Readout

    @nn.compact
    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        g = self.input_enc(cg, ctx)
        for _i in range(self.num_blocks):
            block = self.block_templ.copy()
            g = block(g, ctx)

        return self.readout(g, ctx)


if __name__ == '__main__':
    from cdv.config import MainConfig

    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    node_emb = 128
    edge_emb = 128
    msg_dim = node_emb

    input_enc = InputEncoder(
        OldBessel1DBasis(num_basis=7),
        nn.Dense(node_emb),
        LearnedSpecEmb(config.data.num_species, node_emb),
    )

    block = MLPMessagePassing(
        node_reduction=Fishnet(reduction='mean', net_templ=LazyInMLP([256]), inner_dim=32),
        node_emb=node_emb,
        msg_dim=msg_dim,
        node_templ=LazyInMLP([]),
        msg_templ=LazyInMLP([]),
    )
    readout = NodeAggReadout(LazyInMLP([], out_dim=1))
    model = GN(input_enc, 2, block, readout)

    # model = DimeNetPP(
    #     input_enc=input_enc,
    #     sbf=TripletAngleEmbedding(Bessel2DBasis()),
    #     initial_embed_mlp=LazyInMLP([]),
    #     output=DimeNetPPOutput(head=LazyInMLP([])),
    #     int_dist_enc=LazyInMLP([]),
    #     int_ang_enc=LazyInMLP([]),
    #     int_down_proj_mlp=LazyInMLP([]),
    #     int_up_proj_mlp=LazyInMLP([]),
    #     int_pre_skip_mlp=LazyInMLP([]),
    #     int_post_skip_mlp=LazyInMLP([])
    # )

    batch = load_file(config, 300)

    # debug_structure(batch)
    # print(batch.padding_mask.devices())

    key = jax.random.key(12345)
    ctx = Context(training=True)

    with jax.debug_nans(True):
        out, params = model.init_with_output(key, batch, ctx)

    steps_per_epoch, dl = dataloader(config, split='train', infinite=True)

    @jax.jit
    def loss(params, batch):
        preds = model.apply(params, batch, ctx=Context(training=False))
        return config.train.loss.regression_loss(
            preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask
        )

    res = jax.value_and_grad(loss)(params, batch)
    jax.block_until_ready(res)

    # from ctypes import cdll
    # libcudart = cdll.LoadLibrary('libcudart.so')

    # libcudart.cudaProfilerStart()
    # for i in range(1):
    #     batch = next(dl)
    #     res = jax.value_and_grad(loss)(params, batch)
    #     jax.block_until_ready(res)
    # libcudart.cudaProfilerStop()

    debug_stat(value=res[0], grad=res[1])

    # debug_structure(batch=batch, module=model, out=out)
    # debug_stat(batch=batch, module=model, out=out)
    flax_summary(model, cg=batch, ctx=ctx)
