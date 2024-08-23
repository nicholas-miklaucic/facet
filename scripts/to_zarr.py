"""Converts the dataset to zarr files."""

import jax
import jax.numpy as jnp
import zarr
from collections.abc import Mapping
from cdv.utils import load_pytree
from pathlib import Path
from operator import add


dataset_folder = Path('/home/nmiklaucic/cdv/precomputed/mptrj/batches/')

template_batch = load_pytree(dataset_folder / 'group_0000' / '00000.mpk')

def node_to_zarr(state, n=None, store=None):
    if n is None:
        n = zarr.group(store=store, overwrite=True)
    if isinstance(state, Mapping):        
        for k, v in state.items():
            if isinstance(v, Mapping):
                child = n.create_group(k)
                node_to_zarr(v, child)
            else:
                n.array(k, v)

    return n

def set_path(data, path, value):
    if not hasattr(value, 'get_basic_selection'):
        return
    if isinstance(path, str):
        path = path.split('/')
    if len(path) == 1:        
        data[path[0]] = value.get_basic_selection()
    else:
        set_path(data[path[0]], path[1:], value)

def zarr_to_pytree(zb, template):    
    data = jax.tree_map(lambda _x: None, template)
    zb.visititems(lambda n, v, data=data: set_path(data, n, v))
    return data


if __name__ == '__main__':
    from rich.progress import track
    for folder in track(sorted(dataset_folder.glob('group_*'))):
        for fn in sorted(folder.glob('*.mpk')):
            batch = load_pytree(fn)
            new_fn = str(fn).replace('.mpk', '.zip')
            Path(new_fn).unlink(missing_ok=True)
            store = zarr.ZipStore(new_fn)
            node_to_zarr(batch, store=store)
            store.close()

            store = zarr.ZipStore(new_fn)
            zb = zarr.open(store, mode='r', zarr_version=2)
            data = zarr_to_pytree(zb, template_batch)
            total_diff = jax.tree.reduce(add, jax.tree.map(lambda x, y: jnp.sum(jnp.abs(x.astype(float) - y.astype(float))), data, batch))
            assert total_diff < 1e-3, (total_diff, fn)

    print('Done!')