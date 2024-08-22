"""Code to load the processed data."""

from collections.abc import Sequence
import functools

from functools import partial
from itertools import batched
from os import PathLike
from pathlib import Path
from typing import Literal
from warnings import filterwarnings
import functools as ft

from flax.serialization import from_state_dict, msgpack_restore
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from einops import rearrange
from eins import EinsOp

from cdv.databatch import CrystalGraphs, collate
from cdv.metadata import Metadata
from cdv.utils import debug_stat, debug_structure, load_pytree

filterwarnings('ignore', category=BeartypeDecorHintPep585DeprecationWarning)


def load_raw(config: 'MainConfig', group_num=0, file_num=0):
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    data_folder = config.data.dataset_folder
    fn = data_folder / 'batches' / f'group_{group_num:04}' / f'{file_num:05}.mpk'

    return load_pytree(fn)


@ft.partial(jax.jit)
def process_raw(raw_data) -> CrystalGraphs:
    data: CrystalGraphs = from_state_dict(
        CrystalGraphs.new_empty(1024, 16, 32),
        raw_data,
    )
    data = jax.tree.map(jnp.array, data)

    return data


def load_file(config: 'MainConfig', group_num=0, file_num=0) -> CrystalGraphs:
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    return process_raw(load_raw(config, group_num, file_num))


@jax.jit
def stack_trees(cgs: Sequence[CrystalGraphs]) -> CrystalGraphs:
    return jax.tree_map(lambda *args: jnp.stack(args), *cgs)


def dataloader_base(
    config: 'MainConfig', split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever: otherwise, stops when all data has been yielded."""
    data_folder = config.data.dataset_folder
    groups = sorted((data_folder / 'batches').glob('group_*'))

    splits = np.cumsum([config.data.train_split, config.data.valid_split, config.data.test_split])
    total = splits[-1]
    split_inds = np.zeros(total)
    split_inds[: splits[0]] = 0
    split_inds[splits[0] : splits[1]] = 1
    split_inds[splits[1] :] = 2

    split_i = ['train', 'valid', 'test'].index(split)

    shuffle_rng = np.random.default_rng(config.data.shuffle_seed)

    split_idx = shuffle_rng.permutation(len(groups))
    split_idx = split_idx[split_inds[split_idx % total] == split_i]

    shuffle_rng = np.random.default_rng(config.data.shuffle_seed)

    device = config.device.jax_device()
    if isinstance(device, jax.sharding.Sharding):
        num_devices = len(device.addressable_devices)
    else:
        num_devices = config.stack_size

    split_groups = [groups[i] for i in split_idx]

    group_files = []

    for group in split_groups:
        group_num = int(group.stem.removeprefix('group_'))
        for file in group.glob('*.mpk'):
            group_files.append((group_num, int(file.stem)))

    batch_inds = np.split(
        shuffle_rng.permutation(len(group_files) - len(group_files) % config.train_batch_multiple),
        len(group_files) // config.train_batch_multiple,
    )

    yield len(batch_inds)

    split_files = {}

    # assert len(batch_inds) % num_devices == 0, f'{len(batch_inds)} % {num_devices} != 0'

    # file_data = list(map(functools.partial(load_raw, config), split_idx))
    # byte_data = dict(zip(split_idx, file_data))

    for batches in batched(batch_inds, num_devices):
        for device_batch in batches:
            for i in device_batch:
                split_files[i] = load_file(config, *group_files[i])
        collated = [collate([split_files[i] for i in batch]) for batch in batches]
        stacked = stack_trees(collated)
        yield jax.device_put(stacked, device)

    while infinite:
        batch_inds = np.split(
            shuffle_rng.permutation(split_idx),
            len(split_idx) // config.train_batch_multiple,
        )
        for batches in batched(batch_inds, num_devices):
            collated = [collate([split_files[i] for i in batch]) for batch in batches]
            stacked = stack_trees(collated)
            yield jax.device_put(stacked, device)


def dataloader(
    config: 'MainConfig', split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    dl = dataloader_base(config, split, infinite)
    steps_per_epoch = next(dl)
    return (steps_per_epoch, dl)


def num_elements_class(batch):
    # 2, 3, 4, 5 are values
    # map to 0, 1, 2, 3
    return (
        jax.nn.one_hot(batch['species'], jnp.max(batch['species']).item(), dtype=jnp.int16)
        .max(axis=1)
        .sum(axis=1)
    ) - 2


if __name__ == '__main__':
    from cdv.config import MainConfig
    import numpy as np

    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    from tqdm import tqdm

    steps_per_epoch, dl = dataloader(config, split='train', infinite=True)
    f2 = next(dl)
    debug_structure(conf=next(dl))

    e_forms = []
    n_nodes = []
    for _i in tqdm(np.arange(steps_per_epoch * 2)):
        batch = next(dl)
        e_forms.append(batch.target_data.e_form.mean())
        n_nodes.extend(batch.n_node.tolist())

    jax.debug.visualize_array_sharding(batch.target_data.e_form)

    print(jnp.array(e_forms).mean())
    print(np.unique(n_nodes, return_counts=True))
