from collections import defaultdict
from jaxtyping import Array, Float, PRNGKeyArray, Float32, Int
import jax
import jax.numpy as jnp
import functools as ft
from einops import rearrange, reduce
import flax
import flax.linen as nn
import numpy as np
import pandas as pd
from cdv.utils import ELEM_VALS
from cdv.databatch import CrystalGraphs, NodeData, EdgeData, CrystalData, DIMENSIONALITIES, TripletData


def process_structure_graph(graph_is, df):
    n_node = []
    n_edge = []
    nodes = defaultdict(list)
    senders = []
    receivers = []
    edge_features = defaultdict(list)
    graph_data = defaultdict(list)

    for batch_samp_i, (graph_i, metadata) in enumerate(zip(graph_is, df)):
        sg = graphs[graph_i]

        crystal = sg.structure
        graph = sg.graph

        
        n_node.append(len(graph.nodes))
        # every edge goes in both directions
        n_edge.append(2 * len(graph.edges))
        nodes['species'].extend([ELEM_VALS.index(spec.symbol) for spec in crystal.species])
        nodes['frac'].extend(crystal.frac_coords)
        nodes['cart'].extend(crystal.cart_coords)
        nodes['graph_i'].extend([batch_samp_i] * len(graph.nodes))

        node_i_offset = len(senders)

        for i, j, pos_offset in graph.edges(data='to_jimage'):
            neg_offset = tuple(-x for x in pos_offset)
            for sender, receiver, offset in ((i, j, pos_offset), (j, i, neg_offset)):
                senders.append(node_i_offset + sender)
                receivers.append(node_i_offset + receiver)
                edge_features['to_jimage'].append(offset)
                edge_features['graph_i'].append(batch_samp_i)
                edge_features['sender'].append(senders[-1])
                edge_features['receiver'].append(receivers[-1])

        graph_data['dataset_i'].append(graph_i)
        graph_data['abc'].append(crystal.lattice.parameters[:3])
        graph_data['angles_rad'].append(np.deg2rad(crystal.lattice.parameters[3:]))

        for k in metadata.index:
            if k in ('atoms', 'num_atoms', 'formula'):
                pass
            elif k == 'dimensionality':
                graph_data[k].append(DIMENSIONALITIES.index(metadata[k]))
            else:
                graph_data[k].append(metadata[k])


    dtypes = {
        'species': jnp.uint8,
        'graph_i': jnp.uint16,
        'to_jimage': jnp.int8,
        'sender': jnp.uint16,
        'receiver': jnp.uint16
    }
    for d in nodes, edge_features, graph_data:
         for k in d:
            dtype = dtypes.get(k, None)       
            d[k] = jnp.array(d[k], dtype=dtype)

    G = CrystalGraphs(
        nodes=NodeData(**nodes), 
        edges=EdgeData(**edge_features),         
        graph_data=CrystalData(**graph_data),
        n_node=jnp.array(n_node), 
        n_edge=jnp.array(n_edge),
        padding_mask=jnp.ones(len(graph_is), dtype=jnp.bool)
    )
    return G


def process_raw_graph(graph_is, df):
    """Processes raw graph information (edge list, images) instead of the StructureGraph."""
    n_node = []
    n_edge = []
    nodes = defaultdict(list)
    senders = []
    receivers = []
    edge_features = defaultdict(list)
    graph_data = defaultdict(list)

    for batch_samp_i, (graph_i, metadata) in enumerate(zip(graph_is, df)):
        ijs, ims = graphs[graph_i]

        crystal = metadata.atoms

        n_nodes = len(crystal.sites)
        
        n_node.append(n_nodes)
    
        n_edge.append(ijs.shape[0])
        nodes['species'].extend([ELEM_VALS.index(spec.symbol) for spec in crystal.species])
        nodes['frac'].extend(crystal.frac_coords)
        nodes['cart'].extend(crystal.cart_coords)
        nodes['graph_i'].extend([batch_samp_i] * n_nodes)

        node_i_offset = len(senders)

        for (i, j), pos_offset in zip(ijs, ims):
            # directed and asymmetric: don't automatically duplicate edges
            for sender, receiver, offset in ((i, j, pos_offset),):
                senders.append(node_i_offset + sender)
                receivers.append(node_i_offset + receiver)
                edge_features['to_jimage'].append(offset)
                edge_features['graph_i'].append(batch_samp_i)
                edge_features['sender'].append(senders[-1])
                edge_features['receiver'].append(receivers[-1])                

        graph_data['dataset_i'].append(graph_i)
        graph_data['abc'].append(crystal.lattice.parameters[:3])
        graph_data['angles_rad'].append(np.deg2rad(crystal.lattice.parameters[3:]))

        for k in metadata.index:
            if k in ('atoms', 'num_atoms', 'formula'):
                pass
            elif k == 'dimensionality':
                graph_data[k].append(DIMENSIONALITIES.index(metadata[k]))
            else:
                graph_data[k].append(metadata[k])


    dtypes = {
        'species': jnp.uint8,
        'graph_i': jnp.uint16,
        'to_jimage': jnp.int8,
        'sender': jnp.uint16,
        'receiver': jnp.uint16
    }
    for d in nodes, edge_features, graph_data:
         for k in d:
            dtype = dtypes.get(k, None)       
            d[k] = jnp.array(d[k], dtype=dtype)
    
    triplet_ij, triplet_jk = np.where(edge_features['receiver'][:, None] == edge_features['sender'][None, :])

    G = CrystalGraphs(
        nodes=NodeData(**nodes), 
        edges=EdgeData(**edge_features),
        triplets=TripletData(triplet_ij, triplet_jk),
        graph_data=CrystalData(**graph_data),
        n_node=jnp.array(n_node), 
        n_edge=jnp.array(n_edge),
        padding_mask=jnp.ones(len(graph_is), dtype=jnp.bool)
    )
    return G

from tqdm import tqdm
from flax.serialization import to_bytes

if __name__ == '__main__':
    from rich.prompt import Confirm
    if not Confirm.ask('Regenerate batched data files?'):
        raise RuntimeError('Aborted')
    
    clean = pd.read_pickle('precomputed/jarvis_dft3d_cleaned/dataframe.pkl')    

    elements = set()
    for struct in clean['atoms']:
        elements.update(set(struct.elements))

    for e in elements:
        if e.symbol not in ELEM_VALS:
            raise RuntimeError('ELEM_VALS needs to be updated: ', e)
    
    # from graphs_processing.ipynb
    batches = jnp.load('precomputed/jarvis_dft3d_cleaned/batches.npy')

    import pickle
    with open('precomputed/jarvis_dft3d_cleaned/graphs.pkl', 'rb') as infile:
        graphs = pickle.load(infile)

    is_structure_graph = hasattr(graphs[0], 'graph')

    with jax.default_device(jax.devices('cuda')[1]):
        for i, batch in tqdm(enumerate(batches.T), total=batches.shape[1]):            
            if is_structure_graph:
                data = process_structure_graph(batch, [clean.iloc[b.item()] for b in batch])
            else:
                data = process_raw_graph(batch, [clean.iloc[b.item()] for b in batch])
            with open(f'precomputed/jarvis_dft3d_cleaned/batches/batch{i}.mpk', 'wb') as out:
                out.write(to_bytes(data))            