from typing import Sequence
from flax import struct
from collections import defaultdict
from jaxtyping import Float, Array, Int
from cdv.utils import debug_structure
import jax
import jax.numpy as jnp

DIMENSIONALITIES = ['3D-bulk', 'intercalated ion', '2D-bulk', '0D-bulk', '1D-bulk', 'na', 'intercalated molecule']

@struct.dataclass
class NodeData:
    species: Int[Array, 'nodes']
    frac: Float[Array, 'nodes 3']
    cart: Float[Array, 'nodes 3']
    graph_i: Int[Array, 'nodes']

    @classmethod
    def new_empty(cls, nodes: int) -> 'NodeData':
        return cls(
            species=jnp.empty(nodes),
            frac=jnp.empty((nodes, 3)),
            cart=jnp.empty((nodes, 3)),
            graph_i=jnp.empty(nodes)
        )

@struct.dataclass
class EdgeData:
    to_jimage: Int[Array, 'edges 3']
    graph_i: Int[Array, 'edges']
    sender: Int[Array, 'edges']
    receiver: Int[Array, 'edges']

    @classmethod
    def new_empty(cls, edges: int) -> 'EdgeData':
        return cls(
            to_jimage=jnp.empty((edges, 3)),
            sender=jnp.empty(edges),
            receiver=jnp.empty(edges),
            graph_i=jnp.empty(edges)
        )

@struct.dataclass
class CrystalData:
    dataset_i: Int[Array, 'graphs']
    abc: Float[Array, 'graphs 3']
    angles_rad: Float[Array, 'graphs 3']
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
            e_form=jnp.empty(graphs),
            bandgap=jnp.empty(graphs),
            e_total=jnp.empty(graphs),
            ehull=jnp.empty(graphs),
            dimensionality=jnp.empty(graphs),
            density=jnp.empty(graphs),
            space_group=jnp.empty(graphs),
            magmom=jnp.empty(graphs),
            num_spec=jnp.empty(graphs),
            dataset_i=jnp.empty(graphs)
        )


@struct.dataclass
class Graphs:
    """Batched/padded graphs. Should be able to sub in for jraph.GraphsTuple."""
    nodes: NodeData
    edges: EdgeData
    n_node: Int[Array, 'graphs']
    n_edge: Int[Array, 'graphs']
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
    
    def __add__(self, other: 'Graphs') -> 'Graphs':
        """Collates both objects together, taking care to deal with index offsets."""
        print('Hi')
        other_nodes = other.nodes.replace(graph_i=other.nodes.graph_i + self.n_total_graphs)
        other_edges = other.edges.replace(
            graph_i=other.edges.graph_i + self.n_total_graphs,
            sender=other.edges.sender + self.n_total_nodes,
            receiver=other.edges.receiver + self.n_total_nodes
        )

        other = other.replace(nodes=other_nodes, edges=other_edges)
        return jax.tree.map(lambda x, y: jnp.concatenate((x, y)), self, (other,))
    
    @classmethod
    def new_empty(cls, nodes: int, edges: int, graphs: int) -> 'Graphs':
        return cls(
            nodes=NodeData.new_empty(nodes),
            edges=EdgeData.new_empty(edges),
            graph_data=CrystalData.new_empty(graphs),
            n_node=jnp.empty(graphs),
            n_edge=jnp.empty(graphs),
        )

    

def collate(graphs: Sequence[Graphs]) -> Graphs:
    """Collates the batches into a new Graphs object."""
    return sum(graphs[1:], start=graphs[0])