"""Functionality for graph neural networks."""
from gc import garbage
from typing import Callable, Literal
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
from cdv.dataset import load_file
from cdv.layers import Context, Identity, LazyInMLP
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

    def setup(self):
        self.locs = jnp.linspace(self.lo, self.hi, self.emb)

    def __call__(self, d: Float[Array, 'batch'], ctx: Context) -> Float[Array, 'batch emb']:
        z = d[:, None] - self.locs[None, :]
        y = jnp.exp(-(z ** 2) / (2 * self.sd ** 2))
        return y


# different than
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
# it seems like what I do follows the paper better?

class Envelope(nn.Module):
    """Polynomial envelope that goes to 0 at a cutoff smoothly."""
    # they seem to take p - 1 as input, which seems to me a little harebrained and fails for nans
    # we just take in p and use the formula directly
    p: int
    def setup(self):
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def __call__(self, x):
        # our model shouldn't return nan, even for 0 input, so we have to change this
        return 1 - x ** self.p * (self.a + x * (self.b + x * self.c))


class Bessel1DBasis(DistanceEncoder):
    """Uses spherical Bessel functions with a cutoff, as in DimeNet++."""
    num_basis: int = 7
    cutoff: float = 7
    # Controls how fast the envelope goes to 0 at the cutoff.
    envelope_exp: int = 6

    def setup(self):
        def freq_init(rng):
            return jnp.arange(self.num_basis, dtype=jnp.float32) + 1
        self.freq = self.param('freq', freq_init)
        self.envelope = Envelope(self.envelope_exp)

    def __call__(self, x, ctx: Context):
        dist = x[..., None] / self.cutoff
        env = self.envelope(dist)        

        # e(d) = sqrt(2/c) * sin(fπd/c)/d
        # we use sinc so it's defined at 0
        # jnp.sinc is sin(πx)/(πx)
        # e(d) = sqrt(2/c) * sin(πfd/c)/(fd/c) * f/c
        # e(d) = sqrt(2/c) * sinc(πfd/c)) * πf/c

        e_d = jnp.sqrt(2 / self.cutoff) * jnp.sinc(self.freq * dist) * (jnp.pi * self.freq / self.cutoff)

        debug_stat(e_d=e_d, env=env, dist=dist)
        x123
        return env * e_d



class Bessel2DBasis(nn.Module):
    num_radial: int = 7
    num_spherical: int = 7
    cutoff: float = 7
    # Controls how fast the envelope goes to 0 at the cutoff.
    envelope_exp: int = 6

    def setup(self):
        self.envelope = Envelope(self.envelope_exp)
        self.radial = Bessel1DBasis(num_basis=self.num_radial, cutoff=self.cutoff, envelope_exp=self.envelope_exp)

    def __call__(self, d, alpha, ctx):
        # TODO implement this for real
        dist_emb = self.radial(d, ctx) / self.radial(d * 0, ctx)  # batch radial
        ang_emb = jnp.cos(alpha[:, None] * jnp.arange(self.num_spherical)) # batch spherical
        return EinsOp('batch radial, batch spherical -> batch (radial spherical)')(dist_emb, ang_emb)
    

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
        print(cg.graph_data.lats[10], cg.graph_data.dataset_i[10])
        offsets = EinsOp('e abc xyz, e abc -> e xyz')(cg.graph_data.lats[cg.edges.graph_i], cg.edges.to_jimage)
        recv_pos = cg.nodes.cart[cg.receivers] + offsets

        vecs = recv_pos - send_pos

        dist = EinsOp('edge 3 -> edge', reduce='l2_norm')(vecs)
        debug_stat(dist=dist)
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
            padding_mask=cg.padding_mask
        )
    

# def angle(a, b, c):
#     """Gets angle ABC."""
#     ab = a - b
#     cb = c - b

#     ab = ab / (jnp.linalg.norm(ab) + 1e-8)
#     cb = cb / (jnp.linalg.norm(ab) + 1e-8)

#     return jnp.arccos(jnp.dot(ab, cb))



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

        debug_structure(g)

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
        return node + self.node_mlp(jnp.concat((node, message)), ctx)
    
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

        d_ij = jnp.take(g.dists, g.incoming).reshape(-1) # (nodes in)

        s_out = g.outgoing.shape[-1]
        s_nodes, s_in = g.incoming.shape  

        angle_embeds = self.distance_angle_enc(jnp.repeat(d_ij, s_out), angles, ctx) # (nodes in out) emb        
        angle_embeds = angle_embeds.reshape(s_nodes, s_in, s_out, -1) # nodes in out emb

        return angle_embeds


class DimeNetPPOutput(nn.Module):
    head: LazyInMLP   
    edge2node: SegmentReduction = 'mean'
    node2graph: SegmentReduction = 'mean'
    out_dim: int = 1    
    @nn.compact
    def __call__(self, base_g: Graphs, g: Graphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        msg_dim = g.edge_emb.shape[-1]
        dist_proj = nn.Dense(msg_dim, use_bias=False, name='dist_out_proj')

        lin_rbf = dist_proj(base_g.edge_emb)
        edge_msgs = g.edge_emb * lin_rbf
        node_msgs = segment_reduce(self.edge2node, edge_msgs, g.receivers, num_segments=g.n_total_nodes)

        head = self.head.copy(out_dim=self.out_dim, name='out_head')
        
        node_outs = head(node_msgs, ctx)
        graph_outs = segment_reduce(self.node2graph, node_outs, g.node_graph_i, num_segments=g.n_total_graphs)

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
    
    
    act: Callable = nn.sigmoid
    out_dim: int = 1
    initial_embed_distance_dim: int = 64
    message_dim: int = 128
    down_proj_dim: int = 64
    num_interaction_blocks: int = 2

    @nn.compact
    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'graphs out_dim']:
        g = self.input_enc(cg, ctx)
        a_sbf = self.sbf(g, ctx) # nodes in out emb

        debug_stat(e_rbf=g.edge_emb, a_sbf=a_sbf)

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
            z_rbf = self.int_dist_enc.copy(out_dim=self.message_dim, name=f'int_dist_enc_{i}')(g.edge_emb, ctx) # edges msg_dim
            msg_proj = self.act(nn.Dense(self.message_dim, name=f'msg_proj_{i}')(curr_g.edge_emb)) # edges msg_dim

            # Hadamard multiplication of z_ji and msg_kj
            z_ji = jnp.take(z_rbf, g.outgoing, axis=0)  # nodes out msg_dim
            m_kj = jnp.take(msg_proj, g.incoming, axis=0) # nodes in msg_dim            

            msg_dist = EinsOp('nodes out msg, nodes in msg -> nodes in out msg')(z_ji, m_kj)

            msg_dist_down_proj = self.int_down_proj_mlp.copy(out_dim=self.down_proj_dim, name=f'int_down_proj_{i}')(msg_dist, ctx) # nodes in out down            
            z_sbf = self.int_ang_enc.copy(out_dim=self.down_proj_dim, name=f'int_ang_enc_{i}')(a_sbf, ctx) # nodes in out down
            z_sbf = z_sbf * g.incoming_pad[..., None, None]  # zero out contributions from padding edges            

            msg_all_down = EinsOp('nodes in out down, nodes in out down -> nodes out down')(msg_dist_down_proj, z_sbf)
            msg_all = self.act(nn.Dense(self.message_dim, name=f'int_up_proj_{i}')(msg_all_down))  # nodes out msg_dim
            msg_all = msg_all * g.outgoing_pad[..., None]

            msg_all = EinsOp('nodes out msg_dim -> (nodes out) msg_dim')(msg_all)

            prev_msg_enc = self.act(nn.Dense(self.message_dim, name=f'int_prev_msg_{i}')(curr_g.edge_emb))

            combo_msg = prev_msg_enc + segment_reduce('sum', msg_all, g.outgoing.reshape(-1), num_segments=g.n_total_edges)

            pre_skip = self.int_pre_skip_mlp.copy(out_dim=self.message_dim, name=f'int_pre_skip_{i}')(combo_msg, ctx)

            new_msg = self.int_post_skip_mlp.copy(out_dim=self.message_dim, name=f'int_post_skip_{i}')(pre_skip + prev_msg_enc, ctx)

            curr_g = curr_g.replace(edge_emb=new_msg)

            outs = outs + output_block(g, curr_g, ctx)

        return outs




class Edge2NodeAgg(ProcessingBlock):
    """
    Dimenet++ output block. Aggregates edge embeddings together. 
    """
    distance_embed_dim: int    
    head: LazyInMLP
    edge_to_node: SegmentReduction = 'mean'

    @nn.compact
    def __call__(self, g: Graphs, ctx: Context) -> Graphs:
        total_hidden_dim = g.edge_emb.shape[-1]
        hidden_dim = total_hidden_dim - self.distance_embed_dim
        edge_proj = nn.Dense(hidden_dim, use_bias=False)
        dist_proj = edge_proj(g.edge_emb[..., :self.distance_embed_dim])
        combined_embed = dist_proj * g.edge_emb[..., self.distance_embed_dim:]
        node_embs = segment_reduce(self.edge_to_node, combined_embed, g.receivers, num_segments=g.n_total_nodes)
        node_out = self.head(node_embs, ctx=ctx)
        return g.replace(node_emb=node_out), ctx
        
    

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
        Bessel1DBasis(num_basis=7),
        nn.Dense(node_emb),
        LearnedSpecEmb(config.data.num_species, node_emb)
    )
    # block = MLPMessagePassing(
    #     node_reduction='sum',
    #     node_emb=node_emb,
    #     msg_dim=msg_dim,
    #     node_templ=LazyInMLP([]),
    #     msg_templ=LazyInMLP([])
    # )    
    # readout = nn.Sequential([
    #     Edge2NodeAgg(3, head=LazyInMLP([], out_dim=node_emb)),
    #     NodeAggReadout(LazyInMLP([], out_dim=1))])
    # model = GN(input_enc, 2, block, readout)

    model = DimeNetPP(
        input_enc=input_enc,
        sbf=TripletAngleEmbedding(Bessel2DBasis()),
        initial_embed_mlp=LazyInMLP([]),
        output=DimeNetPPOutput(head=LazyInMLP([])),
        int_dist_enc=LazyInMLP([]),
        int_ang_enc=LazyInMLP([]),
        int_down_proj_mlp=LazyInMLP([]),
        int_up_proj_mlp=LazyInMLP([]),
        int_pre_skip_mlp=LazyInMLP([]),
        int_post_skip_mlp=LazyInMLP([])
    )

    batch = load_file(config, 300)

    key = jax.random.key(12345)        
    ctx = Context(training=True)    

    with jax.debug_nans(True):
        out, params = model.init_with_output(key, batch, ctx)    
        
    def loss(params):
        preds = model.apply(params, batch, ctx=Context(training=False))
        return config.train.loss.regression_loss(preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask)

    res = jax.value_and_grad(loss)(params)
    debug_stat(res)

    # debug_structure(batch=batch, module=model, out=out)
    # debug_stat(batch=batch, module=model, out=out)    
    flax_summary(model, cg=batch, ctx=ctx)