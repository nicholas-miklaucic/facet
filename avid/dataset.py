"""Code to load the processed data."""

from functools import partial
from os import PathLike
from pathlib import Path
from typing import Literal
from warnings import filterwarnings

from flax.serialization import msgpack_restore
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from einops import rearrange
from eins import EinsOp

from avid.augmentations import randomly_augment
from avid.config import MainConfig
from avid.databatch import DataBatch
from avid.metadata import Metadata
from avid.utils import debug_stat, debug_structure, load_pytree

filterwarnings('ignore', category=BeartypeDecorHintPep585DeprecationWarning)


def load_file(config: MainConfig, file_num=0) -> DataBatch:
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    data_folder = config.data.dataset_folder
    fn = data_folder / 'densities' / f'batch{file_num}.mpk'

    data: dict = load_pytree(fn) # type: ignore

    dens_transform = config.data_transform.density_transform()

    # if np.any(np.isnan(data['density'])):
    #     print(data['index'][28])
    #     debug_structure(density=data['density'])
    #     print(np.where(np.isnan(data['density'])))
    #     debug_stat(density=data['density'])
    data['density'] = dens_transform(data['density'])


    for k, v in data.items():
        if v.dtype == np.float32:
            if np.any(np.isnan(v)):
                print(k)
                raise ValueError
            data[k] = jnp.array(v, dtype=jnp.bfloat16)
        else:
            data[k] = jnp.array(v)

    # print(jnp.mean(data['density']), '±', jnp.std(data['density']))

    return DataBatch(**data)


def dataloader_base(
    config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever: otherwise, stops when all data has been yielded."""
    data_folder = config.data.dataset_folder
    files = sorted(list(data_folder.glob('densities/batch*')))

    splits = np.cumsum([config.data.train_split, config.data.valid_split, config.data.test_split])
    total = splits[-1]
    split_inds = np.zeros(total)
    split_inds[: splits[0]] = 0
    split_inds[splits[0] : splits[1]] = 1
    split_inds[splits[1] :] = 2

    split_i = ['train', 'valid', 'test'].index(split)

    # data = [eqx.tree_deserialise_leaves(file, data_templ) for file in files]

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

    if config.data.do_augment:
        aug_rng = np.random.default_rng(config.data.augment_seed)
        transform = partial(
            randomly_augment,
            so3=config.data.so3,
            o3=config.data.o3,
            t3=config.data.t3,
            n_grid=config.voxelizer.n_grid,
            rng=aug_rng,
        )
    else:
        transform = lambda x: x

    # first batch doesn't augment: that's the base on which future augmentations happen. It may make
    # sense in the future to have limited, imperfect augmentations, and we don't want those to be
    # stacked on top of themselves.
    with jax.default_device(jax.devices('cpu')[0]):
        for batch in batch_inds:
            split_files[batch] = [load_file(config, i) for i in batch]

            batch_data = split_files[batch]
            collated = jax.tree_map(lambda *args: jnp.concat(args, axis=0), *batch_data)
            yield jax.device_put(collated, device)

    while infinite:
        batch_inds = np.split(
            shuffle_rng.permutation(split_idx),
            len(split_idx) // config.train_batch_multiple,
        )
        for batch in batch_inds:
            batch_data = [transform(split_files[i]) for i in batch]
            collated = jax.tree_map(lambda *args: jnp.concat(args, axis=0), *batch_data)
            yield jax.device_put(collated, device)


def dataloader(
    config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
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
    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    f1 = load_file(config, 300)
    debug_structure(conf=f1)

    from tqdm import tqdm

    steps_per_epoch, dl = dataloader(config, split='valid', infinite=True)

    means = []
    e_forms = []
    for _i in tqdm(np.arange(steps_per_epoch * 2)):
        batch = next(dl)
        means.append(batch.density.mean())
        e_forms.append(batch.e_form.mean())


    debug_structure(conf=next(dl))

    print(jnp.mean(jnp.array(means)))
    print(jnp.array(e_forms))

