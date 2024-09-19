"""Code to load the processed data."""

import functools as ft
from collections.abc import Sequence
from itertools import batched, cycle
from typing import Generator, Literal
from warnings import filterwarnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
import zarr  # type: ignore
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from flax.serialization import from_state_dict, to_state_dict

from facet.data.databatch import CrystalGraphs, collate
from facet.utils import debug_structure, load_pytree

filterwarnings('ignore', category=BeartypeDecorHintPep585DeprecationWarning)


def load_raw(config: 'MainConfig', group_num=0, file_num=0):
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    data_folder = config.data.dataset_folder
    fn = data_folder / 'batches' / f'group_{group_num:04}' / f'{file_num:05}.mpk'

    return load_pytree(fn)


@ft.partial(jax.jit)
def process_raw(raw_data) -> CrystalGraphs:
    nodes, k = raw_data['edges']['receiver'].shape
    graphs = raw_data['padding_mask'].shape[0]
    data: CrystalGraphs = from_state_dict(
        CrystalGraphs.new_empty(nodes, k, graphs),
        raw_data,
    )  # type: ignore
    # data = jax.tree.map(jnp.array, data)

    return data


def load_file(config: 'MainConfig', group_num=0, file_num=0) -> CrystalGraphs:
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    # the documentation mistakenly listed this as eV/atom, and I didn't check, but it's total
    # eventually I should redo it
    cg: CrystalGraphs = process_raw(load_raw(config, group_num, file_num))
    if config.data.dataset_name == 'mp2022':
        cg = cg.replace(
            target_data=cg.target_data.replace(
                e_form=cg.target_data.e_form / jnp.clip(cg.n_node, 1.0, None)
            )
        )

    return cg


def set_path(data, path, value):
    if not hasattr(value, 'get_basic_selection'):
        return
    if isinstance(path, str):
        path = path.split('/')
    if len(path) == 1:
        data[path[0]] = value.get_basic_selection()
    else:
        set_path(data[path[0]], path[1:], value)


template = to_state_dict(CrystalGraphs.new_empty(1, 1, 1))


def zarr_to_pytree(zb):
    data = jax.tree.map(lambda _: None, template)
    zb.visititems(lambda n, v, data=data: set_path(data, n, v))
    return data


def load_file_zarr(config: 'MainConfig', group_num=0, file_num=0):
    data_folder = config.data.dataset_folder
    fn = data_folder / 'batches' / f'group_{group_num:04}' / f'{file_num:05}.zip'
    store = zarr.ZipStore(fn)
    zb = zarr.open(store, mode='r', zarr_version=2)
    tree = zarr_to_pytree(zb)
    return process_raw(tree)


@jax.jit
@chex.assert_max_traces(1)
def stack_trees(cgs: Sequence[CrystalGraphs]) -> CrystalGraphs:
    return jax.tree_map(lambda *args: jnp.stack(args), *cgs)


def dataloader_base(
    config: 'MainConfig',
    split: Literal['train', 'test', 'valid'] = 'train',
    infinite: bool = False,
    use_zarr: bool = False,
    allow_padding: bool = True,
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever:
    otherwise, stops when all data has been yielded."""
    file_load_fn = load_file_zarr if use_zarr else load_file
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
    split_idx = split_idx[split_inds[np.arange(len(split_idx)) % total] == split_i]

    # print(split_idx)

    shuffle_rng = np.random.default_rng(config.data.shuffle_seed)

    device = config.device.jax_device()
    if isinstance(device, jax.sharding.Sharding):
        stack_total = len(device.addressable_devices) * config.stack_size
    else:
        stack_total = config.stack_size

    split_groups = [groups[i] for i in split_idx]

    group_files = []

    if config.data.batches_per_group == 0:
        limit = cycle([0])
    else:
        limit = range(config.data.batches_per_group)  # type: ignore

    for group in split_groups:
        group_num = int(group.stem.removeprefix('group_'))
        for file, _i in zip(sorted(group.glob('*.mpk')), limit):
            group_files.append((group_num, int(file.stem)))

    split_idx = np.arange(len(group_files))

    shuffle = shuffle_rng.permutation(split_idx)

    add_length = -len(group_files) % (config.train_batch_multiple * stack_total)
    if add_length != 0 and not allow_padding:
        raise ValueError(
            f'{len(group_files)} does not evenly divide {config.train_batch_multiple} * {stack_total // config.stack_size} * {config.stack_size}'
        )

    shuffle = np.hstack((shuffle, shuffle[:add_length]))

    batch_inds = np.split(
        shuffle,
        len(shuffle) // config.train_batch_multiple,
    )

    yield len(batch_inds)

    split_files = {}

    # assert len(batch_inds) % num_devices == 0, f'{len(batch_inds)} % {num_devices} != 0'

    # file_data = list(map(functools.partial(load_raw, config), split_idx))
    # byte_data = dict(zip(split_idx, file_data))

    for batches in batched(batch_inds, stack_total):
        for device_batch in batches:
            for i in device_batch:
                # print(group_files[i])
                split_files[i] = file_load_fn(config, *group_files[i])
        collated = [collate([split_files[i] for i in batch]) for batch in batches]
        # debug_structure(collated)
        stacked = stack_trees(collated)
        yield jax.device_put(stacked, device)

    # for i in shuffle[trunc_length:]:
    #     split_files[i] = file_load_fn(config, *group_files[i])

    while infinite:
        shuffle = shuffle_rng.permutation(split_idx)
        shuffle = np.hstack((shuffle, shuffle[:add_length]))

        batch_inds = np.split(
            shuffle,
            len(shuffle) // config.train_batch_multiple,
        )

        for batches in batched(batch_inds, stack_total):
            collated = [collate([split_files[i] for i in batch]) for batch in batches]
            stacked = stack_trees(collated)
            yield jax.device_put(stacked, device)


def dataloader(
    config: 'MainConfig',
    split: Literal['train', 'test', 'valid'] = 'train',
    infinite: bool = False,
    use_zarr: bool = False,
) -> tuple[int, Generator[CrystalGraphs, CrystalGraphs, None]]:
    dl = dataloader_base(config, split, infinite, use_zarr)
    steps_per_epoch = next(dl)
    return (steps_per_epoch, dl)  # type: ignore


if __name__ == '__main__':
    import numpy as np

    from facet.config import MainConfig
    from facet.utils import debug_stat

    config = pyrallis.parse(config_class=MainConfig)  # type: ignore
    config.cli.set_up_logging()

    from tqdm import tqdm

    steps_per_epoch, dl = dataloader(config, split='train', infinite=True, use_zarr=False)
    f2 = next(dl)
    debug_structure(conf=next(dl))

    e_forms = []
    n_nodes = []
    n_real_nodes = []
    for _i in tqdm(np.arange(steps_per_epoch * 2)):
        batch = next(dl)
        e_forms.append(np.mean(batch.target_data.e_form))
        n_nodes.extend(batch.n_node.tolist())
        n_real_nodes.append(jnp.sum(jnp.array(batch.n_node), where=batch.padding_mask).item())

    jax.debug.visualize_array_sharding(batch.target_data.e_form)

    print(np.mean(e_forms))
    # print(np.unique(n_nodes, return_counts=True))
    debug_stat(jnp.array(n_real_nodes))
