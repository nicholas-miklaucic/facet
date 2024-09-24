"""Shows a large summary of a model."""

import subprocess

import jax
import jax.numpy as jnp
import pyrallis
from jax.lib import xla_client
from flax import linen as nn
from flax.serialization import to_state_dict

from facet.config import MainConfig
from facet.data.dataset import dataloader, load_file, stack_trees
from facet.layers import Context, edge_vecs
from facet.regression import EFSLoss, EFSWrapper
from facet.utils import debug_stat, debug_structure, flax_summary, intercept_stat


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


def show_model(config: MainConfig, make_hlo_dot=False, do_profile=False, show_stat=False):
    # print(config.data.avg_dist(6), config.data.avg_num_neighbors(6))
    config.batch_size = 32
    config.model.resid_init = 'ones'
    kwargs = dict(ctx=Context(training=False))
    num_batches, dl = dataloader(config, split='train', infinite=True)
    for i, b in zip(range(2), dl):
        batch = b

    # batch = stack_trees([load_file(config, i, 0) for i in range(3)])

    batch = jax.device_put(batch, config.device.devices()[0])

    # debug_stat(batch=batch)

    mod = config.build_regressor()
    rngs = {}

    rngs['params'] = jax.random.key(0)
    rngs['dropout'] = jax.random.key(1)
    b1 = jax.tree.map(lambda x: x[0], batch)

    # for k, v in rngs.items():
    #     rngs[k] = jax.device_put(v, )

    params = mod.init(rngs, b1, **kwargs)
    batch = jax.tree.map(lambda x: x[:1], batch)
    # params = jax.device_put_replicated(params, config.device.devices())

    base_apply_fn = jax.vmap(
        lambda p, b, t: config.train.loss.efs_wrapper(
            mod.apply, p, b, rngs=rngs, ctx=Context(training=t)
        ),
        in_axes=(None, 0, None),
    )
    apply_fn = jax.jit(base_apply_fn, static_argnums=2)

    def loss_fn(params, preds=None, training=True):
        if preds is None:
            preds = apply_fn(params, batch, training)
        return jax.tree.map(jnp.mean, jax.vmap(config.train.loss.efs_loss)(batch, preds))

    with jax.debug_nans():
        out = apply_fn(params, batch, True)
        loss = loss_fn(params, out)

    kwargs['cg'] = b1

    debug_structure(out=out, loss=loss)
    debug_stat(input=batch, out=out, loss=loss, vecs=edge_vecs(b1))
    rngs.pop('params')

    rot_batch1, rots1 = jax.vmap(lambda x: x.rotate(123))(batch)
    rot_batch2, rots2 = jax.vmap(lambda x: x.rotate(234))(rot_batch1)
    # debug_structure(rots=rots)

    if show_stat:
        with nn.intercept_methods(intercept_stat):
            rot_out1 = base_apply_fn(params, rot_batch1, False)
    else:
        rot_out1 = apply_fn(params, rot_batch1, False)

    rot_out2 = apply_fn(params, rot_batch2, False)

    debug_stat(
        equiv_error=jax.tree.map(
            lambda x, y: jnp.abs(x - y),
            rot_out2,
            jax.vmap(lambda o, r, c: o.rotate(r, c))(rot_out1, rots2, rot_batch1),
        )
    )

    flax_summary(mod, rngs=rngs, console_kwargs={'width': 200}, **kwargs)

    val_and_grad = jax.jit(jax.value_and_grad(lambda x: jnp.mean(loss_fn(x)['loss'])))
    cost_analysis = val_and_grad.lower(params).compile().cost_analysis()[0]
    if cost_analysis is not None and 'flops' in cost_analysis:
        cost = cost_analysis['flops']
    else:
        cost = 0

    cost /= 1e9

    print(f'Total cost: {cost:.3f} GFLOPs')

    if do_profile:
        with jax.profiler.trace('/tmp/jax-trace', create_perfetto_trace=True):
            val, grad = val_and_grad(params)
            jax.block_until_ready(grad)
    else:
        with jax.debug_nans():
            val, grad = val_and_grad(params)
    debug_stat(loss=val, grad=grad, tree_depth=6)

    if not make_hlo_dot:
        return cost

    grad_loss = jax.xla_computation(jax.value_and_grad(loss))(params)
    with open('model.hlo', 'w') as f:
        f.write(grad_loss.as_hlo_text())
    with open('model.dot', 'w') as f:
        f.write(grad_loss.as_hlo_dot_graph())

    grad_loss_opt = jax.jit(jax.value_and_grad(loss)).lower(params).compile()
    with open('model_opt.hlo', 'w') as f:
        f.write(grad_loss_opt.as_text())
    with open('model_opt.dot', 'w') as f:
        f.write(to_dot_graph(grad_loss_opt.as_text()))

    # debug_structure(grad_loss_opt.cost_analysis())

    for f in ('model.dot', 'model_opt.dot'):
        subprocess.run(['sfdp', f, '-Tsvg', '-O', '-x'])

    return cost


if __name__ == '__main__':
    pyrallis.argparsing.wrap()(show_model)()
