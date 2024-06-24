"""Script to parse metadata information, attached to the Structure but not part of the voxelized
representation."""

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
import pyrallis
from jaxtyping import Array, Float, UInt8
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from rich.progress import track
from rich.prompt import Confirm

from avid.config import MainConfig


class Metadata(eqx.Module):
    e_form: Float[Array, 'n_b b']
    lat_abc_angles: Float[Array, 'n_b b 6']
    space_groups: UInt8[Array, 'n_b b']

    @classmethod
    def new_empty(cls, n_batches: int, batch_size: int):
        return Metadata(jnp.empty((n_batches, batch_size)), jnp.empty((n_batches, batch_size, 6)), jnp.empty((n_batches, batch_size), dtype=jnp.uint8))


def space_group_int(struct: Structure) -> int:
    ana = SpacegroupAnalyzer(struct, symprec=0.3, angle_tolerance=5)
    return ana.get_space_group_number()

if __name__ == '__main__':
    if not Confirm.ask('Regenerate metadata files?'):
        raise ValueError('Aborted')

    config = pyrallis.argparsing.parse(MainConfig, 'configs/defaults.toml')

    # matbench perov dataset
    df = pd.read_json(config.data.raw_data_folder / 'castelli.json', orient='split')
    df['struct'] = [Structure.from_dict(s) for s in df['structure']]

    n_data = len(df.index)
    bs = config.data.data_batch_size
    n_batches = n_data // bs

    data = {
        'e_form': jnp.array(df['e_form'], dtype=jnp.float32).reshape(n_batches, bs),
        'lat_abc_angles': [],
        'space_groups': []
    }
    for batch_i in track(range(0, n_data, bs), description='Processing...'):
        batch: list[Structure] = list(df.iloc[batch_i : batch_i + bs]['struct'])
        data['lat_abc_angles'].append([struct.lattice.parameters for struct in batch])
        data['space_groups'].append([space_group_int(struct) for struct in batch])

    data['lat_abc_angles'] = jnp.array(data['lat_abc_angles'], dtype=jnp.float32)
    data['space_groups'] = jnp.array(data['space_groups'], dtype=jnp.uint8)

    metadata = Metadata(**data)
    print('Space groups:', jnp.unique(metadata.space_groups.reshape(-1)))
    eqx.tree_serialise_leaves(config.data.data_folder / 'metadata.eqx', metadata)
    print('Done!')
