"""Code to load the processed data."""

import functools
from multiprocessing import Pool

from functools import partial
from os import PathLike
from pathlib import Path
from typing import Literal
from warnings import filterwarnings

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


def load_raw(config: 'MainConfig', file_num=0):
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing.
    If pad, pads the batch to the expected size."""    
    data_folder = config.data.dataset_folder
    fn = data_folder / 'batches' / f'batch{file_num}.mpk'

    with open(fn, 'rb') as file:
        return msgpack_restore(file.read())

def process_raw(raw_data, pad=None) -> CrystalGraphs:
    data: CrystalGraphs = from_state_dict(CrystalGraphs.new_empty(1, 1, 1), raw_data)
    data = jax.tree.map(jnp.array, data)

    # debug_structure(data)
    if pad is not None:        
        # debug_structure(d=data, dp=data.padded(*pad))
        data = data.padded(*pad)

    return data

def load_file(config: 'MainConfig', file_num=0, pad=True) -> CrystalGraphs:
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing.
    If pad, pads the batch to the expected size."""    
    if pad:
        pad = config.data.graph_shape
    else:
        pad = None
    return process_raw(load_raw(config, file_num), pad)


def dataloader_base(
    config: 'MainConfig', split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever: otherwise, stops when all data has been yielded."""
    data_folder = config.data.dataset_folder
    files = sorted(list(data_folder.glob('batches/batch*')))

    splits = np.cumsum([config.data.train_split, config.data.valid_split, config.data.test_split])
    total = splits[-1]
    split_inds = np.zeros(total)
    split_inds[: splits[0]] = 0
    split_inds[splits[0] : splits[1]] = 1
    split_inds[splits[1] :] = 2

    split_i = ['train', 'valid', 'test'].index(split)    

    split_idx = np.arange(len(files))
    split_idx = split_idx[split_inds[split_idx % total] == split_i]

    yield len(split_idx) // config.train_batch_multiple

    split_files = np.array([None for _ in range(len(files))])

    shuffle_rng = np.random.default_rng(config.data.shuffle_seed)

    device = config.device.jax_device

    if isinstance(device, jax.sharding.PositionalSharding):
        # rearrange to fit shape of databatch
        device = device.reshape(-1, 1, 1, 1, 1)

    batch_inds = np.split(
        shuffle_rng.permutation(split_idx),
        len(split_idx) // config.train_batch_multiple,
    )

        
    file_data = list(map(functools.partial(load_raw, config), split_idx))
    byte_data = dict(zip(split_idx, file_data))

    # first batch doesn't augment: that's the base on which future augmentations happen. It may make
    # sense in the future to have limited, imperfect augmentations, and we don't want those to be
    # stacked on top of themselves.
    with jax.default_device(jax.devices('cpu')[0]):        
        for batch in batch_inds:
            data_files = [process_raw(byte_data[i], pad=config.data.graph_shape) for i in batch]
            split_files[batch] = data_files
            batch_data = split_files[batch]
            collated = collate(batch_data)
            yield jax.device_put(collated, device)

    while infinite:
        batch_inds = np.split(
            shuffle_rng.permutation(split_idx),
            len(split_idx) // config.train_batch_multiple,
        )
        for batch in batch_inds:
            batch_data = split_files[batch]
            collated = collate(batch_data)
            yield jax.device_put(collated, device)


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
    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    f1 = load_file(config, 300)    
    debug_structure(conf=f1)

    from tqdm import tqdm

    steps_per_epoch, dl = dataloader(config, split='train', infinite=True)

    dens = []
    e_forms = []
    for _i in tqdm(np.arange(steps_per_epoch * 2)):
        batch = next(dl)
        dens.append(batch.graph_data.density.mean())
        e_forms.append(batch.graph_data.e_form.mean())


    debug_structure(conf=next(dl))

    print(jnp.mean(jnp.array(dens)))
    print(jnp.array(e_forms).mean())

