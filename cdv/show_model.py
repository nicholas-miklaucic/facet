import subprocess

import jax
import jax.numpy as jnp
import pyrallis
from jax.lib import xla_client
from flax import linen as nn

from cdv.config import MainConfig
from cdv.dataset import dataloader
from cdv.layers import Context
from cdv.regression import EFSLoss, EFSWrapper
from cdv.utils import debug_stat, debug_structure, flax_summary, intercept_stat


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


def show_model(config: MainConfig, make_hlo_dot=False, do_profile=False, show_stat=False):
    # print(config.data.avg_dist(6), config.data.avg_num_neighbors(6))
    kwargs = dict(ctx=Context(training=True))
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
    batch = jax.tree_map(lambda x: x[:1], batch)
    # params = jax.device_put_replicated(params, config.device.devices())

    base_apply_fn = jax.vmap(
        lambda p, b: EFSWrapper()(mod.apply, p, b, rngs=rngs, **kwargs), in_axes=(None, 0)
    )
    apply_fn = jax.jit(base_apply_fn)

    def loss_fn(params, preds=None):
        if preds is None:
            preds = apply_fn(params, batch)
        if config.task == 'e_form':
            return jax.tree_map(jnp.mean, jax.vmap(config.train.loss.efs_loss)(batch, preds))
        else:
            return preds

    with jax.debug_nans():
        out = apply_fn(params, batch)
        loss = loss_fn(params, out)

    # kwargs['cg'] = b1
    # print(params['params']['edge_proj']['kernel'].devices())
    debug_structure(module=mod, out=out, loss=loss)
    debug_stat(input=batch, out=out, loss=loss)
    rngs.pop('params')

    rot_batch, rots = jax.vmap(lambda x: x.rotate(123))(batch)
    # debug_structure(rots=rots)

    if show_stat:
        with nn.intercept_methods(intercept_stat):
            rot_out = base_apply_fn(params, rot_batch)
    else:
        rot_out = apply_fn(params, rot_batch)

    debug_stat(
        equiv_error=jax.tree.map(
            lambda x, y: jnp.abs(x - y),
            rot_out,
            jax.vmap(lambda o, r, c: o.rotate(r, c))(out, rots, batch),
        )
    )

    flax_summary(mod, rngs=rngs, cg=b1, **kwargs)

    val_and_grad = jax.jit(jax.value_and_grad(lambda x: jnp.mean(loss_fn(x)['loss'])))
    cost_analysis = val_and_grad.lower(params).cost_analysis()
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
