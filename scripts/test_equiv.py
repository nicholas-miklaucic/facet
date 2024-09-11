"""Tests equivariance in models during training."""

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
from jax.lib import xla_client
from flax import linen as nn

from facet.config import MainConfig
from facet.databatch import CrystalGraphs
from facet.dataset import dataloader
from facet.layers import Context
from facet.mace.e3_layers import IrrepsModule
from facet.regression import EFSLoss, EFSWrapper
from facet.utils import debug_stat, debug_structure, flax_summary, intercept_stat, callable_name, signature

import e3nn_jax as e3nn

def add_with_duplicated_name(d: dict, k, v):
    prefix = 0
    while f'{prefix}_{k}' in d:
        prefix += 1
    d[f'{prefix}_{k}'] = v


class EquivComparator:
    def __init__(self, seed = None):
        if seed is None:
            self.seed = np.random.randint(1, 1000)
        else:
            self.seed = seed

        self.rot = -e3nn.rand_matrix(jax.random.key(self.seed), (), dtype=jnp.float32)
        self.outs = [{}, {}]
        self.using_rot = False    

    def prep_for_rot(self):
        self.using_rot = True

    def intercept_record_out(self, next_fun, args, kwargs, context):
        # sig = signature(next_fun)
        # bound = sig.bind(*args, **kwargs)
        # has_printed = False
        # for name, val in bound.arguments.items():
        #     if hasattr(val, 'shape'):
        #         if not has_printed:
        #             print()
        #             print(context.module.name)
        #             print(callable_name(next_fun))
        #             has_printed = True
        #         debug_stat(**{name: val})

        name = str(context.module.name) + '_' + callable_name(next_fun)
        name = name.removesuffix('.__call___')
        out = next_fun(*args, **kwargs)
        if hasattr(out, 'transform_by_matrix'):
            if self.using_rot:
                add_with_duplicated_name(self.outs[1], name, out.transform_by_matrix(self.rot).array)
            else:
                add_with_duplicated_name(self.outs[0], name, out.array)

        return out


def test_equiv(config: MainConfig, make_hlo_dot=False, do_profile=False, show_stat=False):
    # print(config.data.avg_dist(6), config.data.avg_num_neighbors(6))
    kwargs = dict(ctx=Context(training=False))
    num_batches, dl = dataloader(config, split='train', infinite=True)
    for i, b in zip(range(2), dl):
        batch = b

    batch = jax.device_put(batch, config.device.devices()[0])

    # debug_structure(batch=batch)

    if config.task == 'e_form':
        mod = config.build_regressor()
        rngs = {}
    elif config.task == 'vae':
        mod = config.build_vae()
        rngs = {'noise': jax.random.key(123)}

    rngs['params'] = jax.random.key(0)
    rngs['dropout'] = jax.random.key(1)
    b1 = jax.tree_map(lambda x: x[0], batch)
    # for k, v in rngs.items():
    #     rngs[k] = jax.device_put(v, )

    params = mod.init(rngs, b1, **kwargs)
    batch: CrystalGraphs = jax.tree_map(lambda x: x[0], batch)
    # params = jax.device_put_replicated(params, config.device.devices())

    base_apply_fn = lambda p, b: config.train.loss.efs_wrapper(mod.apply, p, b, rngs=rngs, **kwargs)

    apply_fn = jax.jit(base_apply_fn)

    def loss_fn(params, preds=None):
        if preds is None:
            preds = apply_fn(params, batch)
        if config.task == 'e_form':
            return jax.tree_map(jnp.mean, jax.vmap(config.train.loss.efs_loss)(batch, preds))
        else:
            return preds

    rngs.pop('params')

    comparator = EquivComparator()
    # comparator.rot = e3nn.angles_to_matrix(jnp.pi, jnp.pi / 2, jnp.pi / 4)


    lat_rot_m = jnp.einsum('bij,jk->bik', batch.globals.lat, comparator.rot, precision=jax.lax.Precision.HIGHEST)
    new_carts = jnp.einsum('bik,bi->bk', lat_rot_m[batch.nodes.graph_i], batch.frac,
    precision=jax.lax.Precision.HIGHEST)
    # lat_rot_m = jnp.einsum('bij,jk->bik', batch.globals.lat, comparator.rot, precision=jax.lax.Precision.HIGHEST)
    # new_carts = jnp.einsum('ij, bj->bi', comparator.rot, batch.nodes.cart, precision=jax.lax.Precision.HIGHEST)
    new_nodes = batch.nodes.replace(cart=new_carts)
    new_graph_data = batch.graph_data.replace(lat=lat_rot_m)
    rot_batch = batch.replace(nodes=new_nodes, graph_data=new_graph_data)

    with nn.intercept_methods(comparator.intercept_record_out):
        out = base_apply_fn(params, batch)

    comparator.prep_for_rot()
    with nn.intercept_methods(comparator.intercept_record_out):
        rot_out = base_apply_fn(params, rot_batch)

    debug_stat(jax.tree.map(lambda x, y: jnp.abs(x - y), *comparator.outs))


if __name__ == '__main__':
    pyrallis.argparsing.wrap()(test_equiv)()
