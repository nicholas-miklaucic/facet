"""Functionality for graph neural networks."""
from typing import Literal
import eins.reduction
from flax import struct
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
import eins
from eins import Reductions as R
from eins.reduction import Reduction
from eins import EinsOp
import pyrallis


from cdv.databatch import CrystalGraphs
from cdv.dataset import load_file
from cdv.layers import Context, Identity, LazyInMLP
from cdv.utils import debug_stat, debug_structure, flax_summary

class Graphs(struct.PyTreeNode):
    """Generic graphs."""
    node_emb: Float[Array, 'nodes node_emb']
    senders: Int[Array, 'edges']
    receivers: Int[Array, 'edges']
    edge_emb: Float[Array, 'edges edge_emb']    
    graph_emb: Float[Array, 'graphs graph_emb']
    n_nodes: Int[Array, 'graphs']
    n_edges: Int[Array, 'graphs']
    node_graph_i: Int[Array, 'nodes']
    edge_graph_i: Int[Array, 'edges']

    @property
    def n_total_nodes(self) -> int:
        return len(self.node_graph_i)
    
    @property
    def n_total_edges(self) -> int:
        return len(self.edge_graph_i)
    
    @property
    def n_total_graphs(self) -> int:
        return len(self.n_nodes)


class DistanceEncoder(nn.Module):
    """Converts a scalar distance to an embedding."""
    def __call__(self, d: Float[Array, 'batch'], ctx: Context) -> Float[Array, 'batch emb']:
        raise NotImplementedError
    

class GaussBasis(DistanceEncoder):
    """Uses equispaced Gaussian RBFs, as in coGN."""    
    lo: float = 0
    hi: float = 8
    sd: float = 1
    emb: int = 32
    projector: nn.Module = Identity()

    def setup(self):
        self.locs = jnp.linspace(self.lo, self.hi, self.emb)

    def __call__(self, d: Float[Array, 'batch'], ctx: Context) -> Float[Array, 'batch emb']:
        z = d[:, None] - self.locs[None, :]
        y = jnp.exp(-(z ** 2) / (2 * self.sd ** 2))
        return self.projector(y)
    

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
    species_emb: SpeciesEmbedding

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Graphs:
        send_pos = cg.nodes.cart[cg.senders]
        offsets = cg.graph_data.abc[cg.edges.graph_i] * cg.edges.to_jimage
        recv_pos = cg.nodes.cart[cg.receivers] + offsets

        dist = EinsOp('edge 3, edge 3 -> edge', combine='add', reduce='l2_norm')(send_pos, recv_pos)
        dist_emb = self.distance_enc(dist, ctx)
        # debug_stat(dist=dist, dist_emb=dist_emb)

        node_emb = self.species_emb(cg, ctx)

        return Graphs(
            node_emb=node_emb,
            senders=cg.senders,
            receivers=cg.receivers,
            edge_emb=dist_emb,
            graph_emb=jnp.zeros((cg.n_total_graphs, 0)),
            n_nodes=cg.n_node,
            n_edges=cg.n_edge,
            node_graph_i=cg.nodes.graph_i,
            edge_graph_i=cg.edges.graph_i
        )


def segment_mean(data, segment_ids, **kwargs):
    return jax.ops.segment_sum(data, segment_ids, **kwargs) / (1e-6 + jax.ops.segment_sum(jnp.ones_like(data), segment_ids, **kwargs))

SegmentReduction = Literal['max', 'min', 'prod', 'sum', 'mean']

def segment_reduce(reduction: SegmentReduction, data, segment_ids, **kwargs):
    try:
        fn = getattr(jax.ops, f'segment_{reduction}')
    except AttributeError:
        if reduction == 'mean':
            fn = segment_mean
        else:
            raise ValueError('Cannot find reduction')

    return fn(data, segment_ids, **kwargs)

class ProcessingBlock(nn.Module):
    """Block that processes graphs."""
    def __call__(self, g: Graphs, ctx: Context) -> Graphs:
        raise NotImplementedError

class MessagePassing(ProcessingBlock):
    """Message passing block."""
    # How to reduce nodes. We're limited by Jax here, for the time being.
    node_reduction: SegmentReduction

    def node_update(self, node: Float[Array, 'node_emb'], message: Float[Array, 'msg_dim'], ctx: Context) -> Float[Array, 'node_emb']:
        """Updates the node information using the reduced message."""
        raise NotImplementedError
    
    def message(self, edge: Float[Array, 'edge_emb'], sender: Float[Array, 'node_emb'], receiver: Float[Array, 'node_emb'], ctx: Context) -> Float[Array, 'msg_dim']:
        """Computes the message for a given edge."""
        raise NotImplementedError    
    
    def __call__(self, g: Graphs, ctx: Context) -> Graphs:
        edge_messages = jax.vmap(self.message, in_axes=(0, 0, 0, None))(
            g.edge_emb,
            g.node_emb[g.senders],
            g.node_emb[g.receivers],
            ctx
        ) # shape: edges msg_dim

        # aggregate incoming messages for each node
        # shape nodes msg_dim
        node_messages = segment_reduce(self.node_reduction, edge_messages, g.receivers, num_segments=g.n_total_nodes)

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

    def node_update(self, node: Float[Array, 'node_emb'], message: Float[Array, 'msg_dim'], ctx: Context) -> Float[Array, 'node_emb']:
        """Updates the node information using the reduced message."""
        return self.node_mlp(jnp.concat((node, message)), ctx)
    
    def message(self, edge: Float[Array, 'edge_emb'], sender: Float[Array, 'node_emb'], receiver: Float[Array, 'node_emb'], ctx: Context) -> Float[Array, 'msg_dim']:
        """Computes the message for a given edge."""
        return self.msg(jnp.concat((edge, sender, receiver)), ctx)


class Readout(nn.Module):
    """Readout block in GNN."""
    def __call__(self, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        raise NotImplementedError

class NodeAggReadout(Readout):
    """Aggregates node features."""    
    head: nn.Module
    graph_reduction: SegmentReduction = 'mean'    

    def node_transform(self, node: Float[Array, 'node_emb'], ctx: Context) -> Float[Array, 'node_reduce_emb']:
        return node
    
    def __call__(self, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        transformed = jax.vmap(self.node_transform, in_axes=(0, None))(g.node_emb, ctx)
        graph_embs = segment_reduce(self.graph_reduction, transformed, g.node_graph_i, num_segments=g.n_total_graphs)
        return self.head(graph_embs, ctx=ctx)
    

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
        GaussBasis(),
        LearnedSpecEmb(config.data.num_species, node_emb)
    )
    block = MLPMessagePassing(
        node_reduction='sum',
        node_emb=node_emb,
        msg_dim=msg_dim,
        node_templ=LazyInMLP([], residual=True),
        msg_templ=LazyInMLP([])
    )    
    readout = NodeAggReadout(
        LazyInMLP([], out_dim=1)
    )
    model = GN(input_enc, 2, block, readout)    

    batch = load_file(config, 300)

    key = jax.random.key(12345)        
    ctx = Context(training=True)    

    out, params = model.init_with_output(key, batch, ctx)    
        
    def loss(params):
        preds = model.apply(params, batch, ctx=Context(training=False))
        return config.train.loss.regression_loss(preds, batch.graph_data.e_form.reshape(-1, 1))

    debug_stat(jax.value_and_grad(lambda x: jnp.mean(loss(x)))(params))

    debug_structure(batch=batch, module=model, out=out)
    debug_stat(batch=batch, module=model, out=out)
    flax_summary(model, cg=batch, ctx=ctx)