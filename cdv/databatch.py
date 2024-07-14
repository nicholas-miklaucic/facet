from typing import Sequence
from flax import struct
from collections import defaultdict
from jaxtyping import Float, Array, Int, Bool
from cdv.utils import debug_structure
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

DIMENSIONALITIES = [
    '3D-bulk',
    'intercalated ion',
    '2D-bulk',
    '0D-bulk',
    '1D-bulk',
    'na',
    'intercalated molecule',
]

e3nn.reduced_tensor_product_basis(['1o', '1o']).array.transpose(2, 0, 1).round(2)


class NodeData(struct.PyTreeNode):
    species: Int[Array, 'nodes']
    frac: Float[Array, 'nodes 3']
    cart: Float[Array, 'nodes 3']
    incoming: Int[Array, 'nodes max_in']
    incoming_pad: Bool[Array, 'nodes max_in']
    outgoing: Int[Array, 'nodes max_out']
    outgoing_pad: Bool[Array, 'nodes max_out']
    graph_i: Int[Array, 'nodes']

    @classmethod
    def new_empty(cls, nodes: int) -> 'NodeData':
        return cls(
            species=jnp.empty(nodes, dtype=jnp.int16),
            frac=jnp.empty((nodes, 3)),
            cart=jnp.empty((nodes, 3)),
            incoming=jnp.empty((nodes, 32), dtype=jnp.int16),
            outgoing=jnp.empty((nodes, 20), dtype=jnp.int16),
            incoming_pad=jnp.empty((nodes, 32), dtype=jnp.bool),
            outgoing_pad=jnp.empty((nodes, 20), dtype=jnp.bool),
            graph_i=jnp.empty(nodes, dtype=jnp.int16),
        )


class EdgeData(struct.PyTreeNode):
    to_jimage: Int[Array, 'edges 3']
    graph_i: Int[Array, 'edges']
    sender: Int[Array, 'edges']
    receiver: Int[Array, 'edges']

    @classmethod
    def new_empty(cls, edges: int) -> 'EdgeData':
        return cls(
            to_jimage=jnp.empty((edges, 3), dtype=jnp.int8),
            sender=jnp.empty(edges, dtype=jnp.int32),
            receiver=jnp.empty(edges, dtype=jnp.int32),
            graph_i=jnp.empty(edges, dtype=jnp.int16),
        )


class CrystalData(struct.PyTreeNode):
    dataset_i: Int[Array, 'graphs']
    abc: Float[Array, 'graphs 3']
    angles_rad: Float[Array, 'graphs 3']
    lat: Float[Array, 'graphs 3 3']
    e_form: Float[Array, 'graphs']
    bandgap: Float[Array, 'graphs']
    e_total: Float[Array, 'graphs']
    ehull: Float[Array, 'graphs']
    dimensionality: Int[Array, 'graphs']
    density: Float[Array, 'graphs']
    space_group: Int[Array, 'graphs']
    magmom: Float[Array, 'graphs']
    num_spec: Int[Array, 'graphs']

    @classmethod
    def new_empty(cls, graphs: int) -> 'CrystalData':
        return cls(
            abc=jnp.empty((graphs, 3)),
            angles_rad=jnp.empty((graphs, 3)),
            lat=jnp.empty((graphs, 3, 3)),
            e_form=jnp.empty(graphs),
            bandgap=jnp.empty(graphs),
            e_total=jnp.empty(graphs),
            ehull=jnp.empty(graphs),
            dimensionality=jnp.empty(graphs),
            density=jnp.empty(graphs),
            space_group=jnp.empty(graphs),
            magmom=jnp.empty(graphs),
            num_spec=jnp.empty(graphs),
            dataset_i=jnp.empty(graphs, dtype=jnp.int32),
        )


class CrystalGraphs(struct.PyTreeNode):
    """Batched/padded graphs. Should be able to sub in for jraph.GraphsTuple."""

    nodes: NodeData
    edges: EdgeData
    n_node: Int[Array, 'graphs']
    n_edge: Int[Array, 'graphs']
    padding_mask: Bool[Array, 'graphs']
    graph_data: CrystalData

    @property
    def senders(self) -> Int[Array, 'edges']:
        return self.edges.sender

    @property
    def receivers(self) -> Int[Array, 'edges']:
        return self.edges.receiver

    @property
    def globals(self):
        return self.graph_data

    @property
    def n_total_nodes(self) -> int:
        return len(self.nodes.graph_i)

    @property
    def n_total_edges(self) -> int:
        return len(self.edges.graph_i)

    @property
    def n_total_graphs(self) -> int:
        return len(self.n_node)

    @property
    def e_form(self) -> Float[Array, 'graphs']:
        return self.graph_data.e_form

    @property
    def metric_tensor(self) -> Float[Array, 'n_graphs 3 3']:
        return jnp.einsum('bij,bkj->bik', self.graph_data.lat, self.graph_data.lat)

    @property
    def metric_tensor_irreps(self) -> e3nn.IrrepsArray:
        basis = e3nn.reduced_tensor_product_basis(['1o', '1o']).array.transpose(2, 1, 0)
        # metric tensor always symmetric, but this is the easiest way to get the actual basis
        # now we filter
        return e3nn.IrrepsArray(
            ['0e', '1e', '2e'],
            jnp.einsum('ai, bi -> ba', basis.reshape(-1, 9), self.metric_tensor.reshape(-1, 9)),
        ).filter(drop='1e')

    def __add__(self, other: 'CrystalGraphs') -> 'CrystalGraphs':
        """Collates both objects together, taking care to deal with index offsets."""
        other_nodes = other.nodes.replace(
            graph_i=other.nodes.graph_i + self.n_total_graphs,
            incoming=other.nodes.incoming + self.n_total_edges,
            outgoing=other.nodes.outgoing + self.n_total_edges,
        )
        other_edges = other.edges.replace(
            graph_i=other.edges.graph_i + self.n_total_graphs,
            sender=other.edges.sender + self.n_total_nodes,
            receiver=other.edges.receiver + self.n_total_nodes,
        )

        other = other.replace(nodes=other_nodes, edges=other_edges)
        return jax.tree.map(lambda x, y: jnp.concatenate((x, y)), self, other)

    def padded(self, n_node: int, n_edge: int, n_graph: int):
        """Pad the graph to the given shape. Adds a single padding graph
        with the required extra nodes and edges, then adds empty graphs."""
        pad_n_node = int(n_node - self.n_total_nodes)
        pad_n_edge = int(n_edge - self.n_total_edges)
        pad_n_graph = int(n_graph - self.n_total_graphs)

        # https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py#L604
        if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
            raise RuntimeError(
                'Given graph is too large for the given padding.\n'
                f'Current shape: {self.n_total_nodes} nodes, {self.n_total_edges} edges, {self.n_total_graphs} graphs\n'
                f'Desired shape: {n_node} nodes, {n_edge} edges, {n_graph} graphs'
            )

        return self + CrystalGraphs.new_empty(pad_n_node, pad_n_edge, pad_n_graph)

    @classmethod
    def new_empty(cls, nodes: int, edges: int, graphs: int) -> 'CrystalGraphs':
        return cls(
            nodes=NodeData.new_empty(nodes),
            edges=EdgeData.new_empty(edges),
            graph_data=CrystalData.new_empty(graphs),
            n_node=jnp.concat((jnp.array([nodes]), jnp.zeros(graphs - 1, jnp.int16))),
            n_edge=jnp.concat((jnp.array([edges]), jnp.zeros(graphs - 1, jnp.int16))),
            padding_mask=jnp.zeros(graphs, dtype=jnp.bool),
        )

    def rotate(self, seed: int) -> tuple['CrystalGraphs', Float[Array, 'n_graph 3 3']]:
        """Rotate the coordinates using random rotation matrices. Returns the rotated outputs and
        the matrices."""
        rots = e3nn.rand_matrix(jax.random.key(seed), shape=(self.n_total_graphs,))
        lat_rot_m = jnp.einsum('bij,bjk->bik', self.globals.lat, rots)
        new_carts = jnp.einsum('bik,bi->bk', lat_rot_m[self.nodes.graph_i], self.nodes.frac)
        new_nodes = self.nodes.replace(cart=new_carts)
        new_graph_data = self.graph_data.replace(lat=lat_rot_m)
        new_batch = self.replace(nodes=new_nodes, graph_data=new_graph_data)
        return new_batch, rots


def collate(graphs: Sequence[CrystalGraphs]) -> CrystalGraphs:
    """Collates the batches into a new Graphs object."""
    return sum(graphs[1:], start=graphs[0])
