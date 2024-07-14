import subprocess

import jax
import jax.numpy as jnp
import pyrallis
from jax.lib import xla_client

from cdv.config import MainConfig
from cdv.dataset import dataloader, load_file
from cdv.layers import Context
from cdv.utils import debug_stat, debug_structure, flax_summary
from cdv.vae import vae_loss


# https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html
def to_dot_graph(x):
    return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))


@pyrallis.argparsing.wrap()
def show_model(config: MainConfig, make_hlo_dot=False, do_profile=False):
    kwargs = dict(ctx=Context(training=True))
    num_batches, dl = dataloader(config, split='train')
    for i, b in zip(range(3), dl):
        batch = b

    if config.task == 'e_form':
        mod = config.build_regressor()
        enc_batch = {'cg': batch}
        rngs = {}
    elif config.task == 'vae':
        mod = config.build_vae()
        enc_batch = {'cg': batch}
        rngs = {}
    elif config.task == 'diled':
        mod = config.build_diled()
        enc_batch = {
            'cg': batch,
        }
        rngs = {'noise': jax.random.key(123), 'time': jax.random.key(234)}

    rngs['params'] = jax.random.key(0)
    rngs['dropout'] = jax.random.key(1)
    for k, v in rngs.items():
        rngs[k] = jax.device_put(v, list(batch.e_form.devices())[0])
    kwargs.update(enc_batch)
    out, params = mod.init_with_output(rngs, **kwargs)
    # print(params['params']['edge_proj']['kernel'].devices())
    debug_structure(module=mod)
    # debug_stat(input=batch)
    debug_stat(out=out)
    rngs.pop('params')
    # flax_summary(mod, rngs=rngs, **kwargs)

    if config.task == 'e_form':
        kwargs['cg'], rots = kwargs['cg'].rotate(123)
        rot_out = mod.apply(params, kwargs['cg'], rngs=rngs, ctx=Context(training=True))
        debug_stat(
            equiv_error=jnp.abs(rot_out - out)
            / (jnp.maximum(jnp.abs(out), jnp.abs(rot_out)) + 1e-10)
        )

    def loss(params):
        preds = mod.apply(params, batch, rngs=rngs, ctx=Context(training=True))
        if config.task == 'e_form':
            return {
                'loss': config.train.loss.regression_loss(
                    preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask
                )
            }
        elif config.task == 'vae':
            return vae_loss(config.train.loss, batch, preds)
        else:
            return preds

    if do_profile:
        with jax.profiler.trace('/tmp/jax-trace', create_perfetto_trace=True):
            val, grad = jax.value_and_grad(lambda x: jnp.mean(loss(x)['loss']))(params)
            jax.block_until_ready(grad)
    else:
        val, grad = jax.value_and_grad(lambda x: jnp.mean(loss(x)['loss']))(params)
    debug_stat(val=val, grad=grad)

    if not make_hlo_dot:
        return

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


if __name__ == '__main__':
    show_model()
