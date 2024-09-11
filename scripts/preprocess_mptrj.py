"""Processes the MPTrj dataset."""

from tqdm import tqdm
from facet.databatch import CrystalGraphs, CrystalData, EdgeData, NodeData, MPTrjTarget
from pymatgen.core import Structure
import numpy as np
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
import pickle
import multiprocessing

from facet.utils import save_pytree

num_batch = 32
k = 16
num_atoms = 32
target_batch_size = num_batch * num_atoms - 1

overwrite = False
num_processes = 1

data_folder = Path('precomputed') / 'mptrj'
graphs_folder = Path('/home/nmiklaucic/mat-graph/crystallographic_graph/knns')

if num_processes > 1:
    tqdm = lambda x, **kwargs: x

# def knn_graph(struct: Structure, r_start=8, k=k):
#     """returns (ijs, ims)
#     ijs: nodes k
#     ims: nodes k 3"""
#     graph_ijs = []
#     graph_ims = []
#     if r_start > np.sqrt(np.sum(np.array(struct.lattice.abc) ** 2)) * 4:
#         struct.to('test.cif')
#         raise ValueError(f'{r_start}')
#     r = r_start
#     for i, nbs in enumerate(struct.get_all_neighbors(r)):
#         sites, dists, idxs, ims = zip(*nbs)
#         if len(dists) < k:
#             # print('Not enough neighbors, using r =', 2 * r_start)
#             return knn_graph(struct, r_start=r * 2, k=k)
        
#         chosen = np.argsort(dists)[:k]

#         graph_ijs.append(np.array(idxs)[chosen])
#         graph_ims.append(np.array(ims)[chosen])

#     graph_ijs = np.stack(graph_ijs).astype(np.uint16)
#     graph_ims = np.stack(graph_ims).astype(np.int8)

#     return EdgeData(jnp.array(graph_ims), jnp.array(graph_ijs))


def create_graph(row, data_id, graph_data):
    struct: Structure = row['structure']
    nodes = NodeData(
        species=jnp.array(struct.atomic_numbers, dtype=np.uint8),
        cart=jnp.array(struct.cart_coords, dtype=np.float32),
        graph_i=jnp.zeros((struct.num_sites,), dtype=np.uint16)
    )

    data = CrystalData(
        dataset_id=jnp.array([data_id], dtype=np.uint32),
        abc=jnp.array([struct.lattice.abc]),
        angles_rad=jnp.deg2rad(jnp.array([struct.lattice.angles])),
        lat=jnp.array([struct.lattice.matrix])
    )
    target = MPTrjTarget(
        e_form=jnp.array([row['energy_per_atom']]),    
        force=jnp.array(row['force']),
        stress=jnp.array([row['stress']]),
    )
    # diag = np.sqrt(np.sum(np.array(struct.lattice.abc) ** 2)) + 1
    # edges = knn_graph(struct, r_start=diag * np.cbrt(k / struct.num_sites))
    edges = EdgeData(to_jimage=jnp.array(graph_data['ims']), receiver=jnp.array(graph_data['ijs']))

    return CrystalGraphs(nodes, edges, n_node=jnp.array([struct.num_sites], dtype=np.uint16), padding_mask=jnp.ones((1,), dtype=np.bool_), graph_data=data, target_data=target)

def get_parts(numbers, batch, chunk_size):
    """Splits the numbers into batches of length batch with as equal a split as possible."""
    # assert len(numbers) % (batch * chunk_size) == 0
    n_batches = len(numbers) // batch
    parts = np.zeros((batch, n_batches), dtype=np.int32)
    part_sizes = np.array([0 for _ in range(n_batches)])
    
    chunk_i = 0
    for sample_is in np.argsort(-numbers).reshape(batch // chunk_size, chunk_size * n_batches):
        sample_sizes = numbers[sample_is]
        n_filled = np.zeros((n_batches,), dtype=np.int32)
        for sample_i, sample_size in zip(sample_is, sample_sizes):
            next_i = np.argmin(part_sizes + 10000 * (n_filled == chunk_size))
            parts[chunk_i * chunk_size + n_filled[next_i], next_i] += sample_i
            n_filled[next_i] += 1
            part_sizes[next_i] += sample_size
        chunk_i += 1

    return parts, part_sizes

def padded_parts(sizes, extra=0):
    part_size = num_batch - 1
    data_pad = -len(sizes) % part_size + part_size * extra
    pad_sizes = np.array(list(sizes) + [0] * data_pad)
    parts, part_sizes = get_parts(pad_sizes, part_size, part_size)
    if max(part_sizes) > target_batch_size:
        return padded_parts(sizes, extra + 4)
    else:
        return parts, part_sizes 


def process_batch(batch_name):
    path = data_folder / 'raw' / f'batch_{batch_name}.pkl'
    df = pd.read_pickle(path)

    graph_path = graphs_folder / f'batch_{batch_name}.pkl'
    with open(graph_path, 'rb') as graph_f:
        graphs = pickle.load(graph_f)
        

    out_path = data_folder / 'batches' / f'group_{batch_name}'
    out_path.mkdir(exist_ok=True, parents=True)

    sizes = [s.num_sites for s in df['structure']]
    
    data_id = df['mp_id'].str.replace('mp-', '1').str.replace('mvc-', '2').astype(int) * 10_000
    data_id += df['calc'] * 1_000
    data_id += df['step']

    orig_size = len(sizes)
    parts, part_sizes = padded_parts(sizes)

    for part_i, partition in tqdm(enumerate(parts.T), total=len(parts.T)):
        out_fn = out_path / f'{part_i:05}.mpk'
        if out_fn.exists() and not overwrite:
            continue
        cgs = [create_graph(df.iloc[i], data_id.iloc[i], graphs[i]) for i in partition if i < orig_size]

        cg = sum(cgs[1:], start=cgs[0])
        cg = cg.padded(num_batch * num_atoms, k, num_batch)
        save_pytree(cg, out_fn)        

    return batch_name


if __name__ == '__main__':
    # import multiprocessing            
    import gc
    from rich.prompt import Confirm
    if not Confirm.ask('Regenerate batched MPTrj files?'):
        raise RuntimeError('Aborted')
    
    if num_processes > 1:
        pool = multiprocessing.Pool(num_processes)
        map = pool.imap_unordered

    max_sizes = []
    batches = sorted((data_folder / 'raw').glob('batch_*.pkl'))    
    names = [batch_fn.stem.removeprefix('batch_') for batch_fn in batches]
        
    for res in map(process_batch, names):
        print(f'Finished {res}')
        gc.collect()
