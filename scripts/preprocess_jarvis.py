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
from facet.databatch import CrystalGraphs, NodeData, EdgeData, CrystalData, DIMENSIONALITIES

ELEM_VALS = (
    'K Rb Ba Na Sr Li Ca La Tb Yb Ce Pr Nd Sm Dy Y Ho Er Tm Hf Mg Zr Sc U Ta Ti Mn Be Nb Al Tl V Zn Cr Cd'
    ' In Ga Fe Co Cu Si Ni Ag Sn Hg Ge Bi B Sb Te Mo As P H Ir Os Pd Ru Pt Rh Pb W Au C Se S I Br N Cl O F'
).split(' ')

MAX_IN = 32
MAX_OUT = 20

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

        node_i_offset = len(nodes['species'])
        edge_i_offset = len(senders)

        crystal = metadata.atoms

        n_nodes = len(crystal.sites)
        
        n_node.append(n_nodes)
    
        n_edge.append(ijs.shape[0])
        
        nodes['species'].extend([ELEM_VALS.index(spec.symbol) for spec in crystal.species])
        nodes['frac'].extend(crystal.frac_coords)
        nodes['cart'].extend(crystal.cart_coords)
        nodes['graph_i'].extend([batch_samp_i] * n_nodes)
        nodes['incoming'].extend([[] for _ in range(n_nodes)])
        nodes['outgoing'].extend([[] for _ in range(n_nodes)])        

        for (i, j), pos_offset in zip(ijs, ims):
            # directed and asymmetric: don't automatically duplicate edges
            for sender, receiver, offset in ((i, j, pos_offset),):
                senders.append(node_i_offset + sender)
                receivers.append(node_i_offset + receiver)
                edge_features['to_jimage'].append(offset)
                edge_features['graph_i'].append(batch_samp_i)
                edge_features['sender'].append(senders[-1])
                edge_features['receiver'].append(receivers[-1])
                edge_i = len(senders) - 1
                nodes['outgoing'][senders[-1]].append(edge_i)
                nodes['incoming'][receivers[-1]].append(edge_i)

                

        graph_data['dataset_i'].append(graph_i)
        graph_data['abc'].append(crystal.lattice.parameters[:3])
        graph_data['angles_rad'].append(np.deg2rad(crystal.lattice.parameters[3:]))
        graph_data['lat'].append(crystal.lattice.matrix)

        for k in metadata.index:
            if k in ('atoms', 'num_atoms', 'formula'):
                pass
            elif k == 'dimensionality':
                graph_data[k].append(DIMENSIONALITIES.index(metadata[k]))
            else:
                graph_data[k].append(metadata[k])
    
    for k, limit in zip(('incoming', 'outgoing'), (MAX_IN, MAX_OUT)):
        curr_max = max(map(len, nodes[k]))
        if curr_max > limit:
            raise ValueError(f'Max {k} degree too small: {limit} < {curr_max}')
        for arr in nodes[k]:
            pad_len = limit - len(arr)
            nodes[k + '_pad'].append([True] * len(arr) + [False] * pad_len)
            arr.extend([0] * pad_len)

    dtypes = {
        'species': jnp.uint8,
        'graph_i': jnp.uint16,
        'to_jimage': jnp.int8,
        'sender': jnp.uint16,
        'receiver': jnp.uint16,
        'incoming': jnp.uint16,
        'outgoing': jnp.uint16,
        'incoming_pad': jnp.bool,
        'outgoing_pad': jnp.bool,    
        'dataset_i': jnp.uint32,
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

from tqdm import tqdm
from flax.serialization import to_bytes

if __name__ == '__main__':
    from rich.prompt import Confirm
    if not Confirm.ask('Regenerate batched data files?'):
        raise RuntimeError('Aborted')
    
    from pathlib import Path

    batch_dir = Path('precomputed/jarvis_dft3d_cleaned/batches/')

    for existing in batch_dir.glob('batch*.mpk'):
        existing.unlink()
    

    
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

    with jax.default_device(jax.devices('cuda')[1]):
        for i, batch in tqdm(enumerate(batches.T), total=batches.shape[1]):            
            data = process_raw_graph(batch, [clean.iloc[b.item()] for b in batch])
            with open(batch_dir / f'batch{i}.mpk', 'wb') as out:
                out.write(to_bytes(data))            