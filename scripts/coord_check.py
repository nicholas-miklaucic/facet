"""Checks model scaling as a function of model width."""

"""Tests equivariance in models during training."""

import subprocess
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
from jax.lib import xla_client
from flax import linen as nn
import rich
from rich.pretty import pprint
from tqdm import tqdm

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


class CoordChecker:
    def __init__(self, widths: Sequence[int]):
        self.widths = widths
        self.curr_out = 0
        self.outs = [[] for _ in self.widths]
        self.outs[0].append({})

    def next_step(self):
        self.outs[self.curr_out].append({})

    def next_width(self):
        self.curr_out += 1
        if self.curr_out < len(self.outs):
            self.outs[self.curr_out].append({})

    def intercept_record_out(self, next_fun, args, kwargs, context):
        name = str(context.module.name) + '_' + callable_name(next_fun)
        name = name.removesuffix('.__call__')
        out = next_fun(*args, **kwargs)
        if hasattr(out, 'transform_by_matrix') and not hasattr(out, 'grid_values'):
            add_with_duplicated_name(self.outs[self.curr_out][-1], name, jnp.mean(jnp.abs(out.array)))

        return out


def setup_for_coord_check(config: MainConfig) -> MainConfig:
    dct = pyrallis.encode(config)
    dct['batch_size'] = 32
    dct['stack_size'] = 1
    dct['debug_mode'] = True
    dct['train']['base_lr'] = 3e-2
    return pyrallis.decode(MainConfig, dct)

def set_width(config: MainConfig, width: int) -> MainConfig:
    dct = pyrallis.encode(config)
    dct['mace']['hidden_irreps']['dim'] = width
    return pyrallis.decode(MainConfig, dct)

def coord_check(config: MainConfig):    
    widths = [32, 64, 128, 256]
    num_steps = 4


    config = setup_for_coord_check(config)
    kwargs = dict(ctx=Context(training=False))
    num_batches, dl = dataloader(config, split='train', infinite=True)
    batches = []
    for i, batch in zip(range(num_steps), dl):    
        batch = jax.device_put(batch, config.device.devices()[0])    
        batches.append(jax.tree_map(lambda x: x[0], batch))
    
    rngs = {}    
    rngs['dropout'] = jax.random.key(1)
    
    checker = CoordChecker(widths)

    losses = []
    for width in widths:
        losses.append([])
        width_config = set_width(config, width)
        mod = width_config.build_regressor()
        optim = width_config.train.optimizer(width_config.train.base_lr)
    
        params = mod.init({**rngs, 'params': jax.random.key(0)}, batches[0], **kwargs)    
        opt_state = optim.init(params)
        base_apply_fn = lambda p, b: width_config.train.loss.efs_wrapper(mod.apply, p, b, rngs=rngs, **kwargs)
        # apply_fn = jax.jit(base_apply_fn)
        apply_fn = base_apply_fn
        
        def loss_fn(params, preds=None, batch=None):
            if preds is None:
                preds = apply_fn(params, batch)
            if config.task == 'e_form':
                return jnp.mean(width_config.train.loss.efs_loss(batch, preds)['loss'])
            else:
                return preds
            
        grad_loss_fn = jax.jit(jax.grad(loss_fn))
        for step, batch in tqdm(enumerate(batches), total=num_steps):
            with nn.intercept_methods(checker.intercept_record_out):
                losses[-1].append(loss_fn(params, batch=batch))
            grads = grad_loss_fn(params, batch=batch)
            updates, opt_state = jax.jit(optim.update)(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            checker.next_step()

        checker.next_width()
            
    

    for step in range(num_steps):
        step_outs = [outs[step] for outs in checker.outs]
        step_curves = jax.tree.map(lambda *x: jnp.stack(x).tolist(), *step_outs)
        pprint({f'{k:>50}': [f'{v:05.3f}' for v in vs] for k, vs in step_curves.items()})


if __name__ == '__main__':
    pyrallis.argparsing.wrap()(coord_check)()
