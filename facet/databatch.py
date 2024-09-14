from typing import Sequence
from flax import struct
from jaxtyping import Float, Array, Int, Bool
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp


class NodeData(struct.PyTreeNode):
    species: Int[Array, 'nodes']
    cart: Float[Array, 'nodes 3']
    graph_i: Int[Array, 'nodes']

    @classmethod
    def new_empty(cls, nodes: int) -> 'NodeData':
        return cls(
            species=jnp.empty(nodes, dtype=jnp.int16),
            cart=jnp.empty((nodes, 3)),
            graph_i=jnp.empty(nodes, dtype=jnp.int16),
        )


class EdgeData(struct.PyTreeNode):
    to_jimage: Int[Array, 'nodes k 3']
    receiver: Int[Array, 'nodes k']

    @classmethod
    def new_empty(cls, nodes: int, k: int) -> 'EdgeData':
        return cls(
            to_jimage=jnp.empty((nodes, k, 3), dtype=jnp.int8),
            receiver=jnp.empty((nodes, k), dtype=jnp.uint32),
        )


class CrystalData(struct.PyTreeNode):
    dataset_id: Int[Array, 'graphs']
    abc: Float[Array, 'graphs 3']
    angles_rad: Float[Array, 'graphs 3']
    lat: Float[Array, 'graphs 3 3']

    @classmethod
    def new_empty(cls, graphs: int) -> 'CrystalData':
        return cls(
            abc=jnp.empty((graphs, 3)),
            angles_rad=jnp.empty((graphs, 3)),
            lat=jnp.empty((graphs, 3, 3)),
            dataset_id=jnp.empty(graphs, dtype=jnp.int32),
        )


class MPTrjTarget(struct.PyTreeNode):
    # corrected energy per atom
    e_form: Float[Array, 'graphs']

    # forces
    force: Float[Array, 'nodes 3']

    # stress tensor
    stress: Float[Array, 'graphs 3 3']

    @classmethod
    def new_empty(cls, graphs: int, nodes: int) -> 'MPTrjTarget':
        return cls(
            e_form=jnp.empty((graphs,)),
            force=jnp.empty((nodes, 3)),
            stress=jnp.empty((graphs, 3, 3)) + jnp.eye(3),
        )


class CrystalGraphs(struct.PyTreeNode):
    """Batched/padded graphs. Should be able to sub in for jraph.GraphsTuple."""

    nodes: NodeData
    edges: EdgeData
    n_node: Int[Array, 'graphs']
    padding_mask: Bool[Array, 'graphs']
    graph_data: CrystalData
    target_data: MPTrjTarget

    @property
    def receivers(self) -> Int[Array, 'nodes k']:
        return self.edges.receiver

    @property
    def globals(self):
        return self.graph_data

    @property
    def n_total_nodes(self) -> int:
        return len(self.nodes.graph_i)

    @property
    def n_total_graphs(self) -> int:
        return len(self.n_node)

    @property
    def cart(self) -> Float[Array, 'nodes 3']:
        return self.nodes.cart

    @property
    def frac(self) -> Float[Array, 'nodes 3']:
        lat_inv = jax.vmap(jnp.linalg.inv)(self.globals.lat + jnp.eye(3) * 1e-5)
        return jnp.einsum('bij,bi->bj', lat_inv[self.nodes.graph_i], self.cart)

    @property
    def metric_tensor(self) -> Float[Array, 'n_graphs 3 3']:
        return jnp.einsum('bij,bkj->bik', self.graph_data.lat, self.graph_data.lat)

    @property
    def e_form(self) -> Float[Array, 'graphs']:
        return self.target_data.e_form

    def __add__(self, other: 'CrystalGraphs') -> 'CrystalGraphs':
        """Collates both objects together, taking care to deal with index offsets."""
        other_nodes = other.nodes.replace(
            graph_i=other.nodes.graph_i + self.n_total_graphs,
        )
        other_edges = other.edges.replace(
            receiver=other.edges.receiver + self.n_total_nodes,
        )

        other = other.replace(nodes=other_nodes, edges=other_edges)
        return jax.tree.map(lambda x, y: jnp.concatenate((x, y)), self, other)

    def padded(self, n_node: int, k: int, n_graph: int):
        """Pad the graph to the given shape. Adds a single padding graph
        with the required extra nodes and edges, then adds empty graphs."""
        pad_n_node = int(n_node - self.n_total_nodes)
        pad_n_graph = int(n_graph - self.n_total_graphs)

        # https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py#L604
        if pad_n_node <= 0 or pad_n_graph <= 0:
            raise RuntimeError(
                'Given graph is too large for the given padding.\n'
                f'Current shape: {self.n_total_nodes} nodes, {self.n_total_graphs} graphs\n'
                f'Desired shape: {n_node} nodes, {n_graph} graphs'
            )

        return self + CrystalGraphs.new_empty(pad_n_node, k, pad_n_graph)

    @classmethod
    def new_empty(cls, nodes: int, k: int, graphs: int) -> 'CrystalGraphs':
        return cls(
            nodes=NodeData.new_empty(nodes),
            edges=EdgeData.new_empty(nodes, k),
            graph_data=CrystalData.new_empty(graphs),
            target_data=MPTrjTarget.new_empty(graphs, nodes),
            n_node=jnp.concat((jnp.array([nodes]), jnp.zeros(graphs - 1, jnp.int16))),
            padding_mask=jnp.zeros(graphs, dtype=jnp.bool),
        )

    def rotate(self, seed: int) -> tuple['CrystalGraphs', Float[Array, 'n_graph 3 3']]:
        """Rotate the coordinates using random rotation matrices. Returns the rotated outputs and
        the matrices."""
        rots = e3nn.rand_matrix(jax.random.key(seed), shape=(self.n_total_graphs,))
        lat_rot_m = jnp.einsum('bij,bjk->bik', self.globals.lat, rots)
        new_carts = jnp.einsum('bik,bi->bk', lat_rot_m[self.nodes.graph_i], self.frac)
        new_nodes = self.nodes.replace(cart=new_carts)
        new_graph_data = self.graph_data.replace(lat=lat_rot_m)
        new_batch = self.replace(nodes=new_nodes, graph_data=new_graph_data)
        return new_batch, rots

    def __rich_repr__(self):
        yield 'nodes', self.nodes.cart.shape
        yield 'edges', self.edges.receiver.shape
        yield 'graphs', self.globals.dataset_id.shape


def collate(graphs: Sequence[CrystalGraphs]) -> CrystalGraphs:
    """Collates the batches into a new Graphs object."""
    return sum(graphs[1:], start=graphs[0])
