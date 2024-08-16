import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import time
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import rho_plus as rp

from cdv.dataset import load_file
from cdv.layers import LazyInMLP
from cdv.utils import debug_stat, debug_structure
from cdv.vae import VAE, Decoder, Encoder, LatticeVAE, PropertyPredictor

is_dark = False
theme, cs = rp.mpl_setup(is_dark)


from pathlib import Path
import pyrallis
from cdv.config import MainConfig
import orbax.checkpoint as ocp

from cdv.training_state import TrainingRun
from cdv.checkpointing import best_ckpt

run_dir = Path('logs') / 'mace_vae_rec_407'

with open(run_dir / 'config.toml') as conf_file:
    config = pyrallis.cfgparsing.load(MainConfig, conf_file)

self = config
model =  VAE(
    Encoder(
        self.mace.build(self.data.num_species, '0e', None),
        latent_dim=128,
        latent_space=LatticeVAE(),
    ),
    PropertyPredictor(LazyInMLP([256], dropout_rate=0.3)),
    Decoder(self.mace.build(self.data.num_species, '0e', None)),
    prop_reg_loss=self.train.loss.regression_loss,
)

ckpt = best_ckpt(run_dir)
# ckpt = jax.tree.map(lambda x: x if isinstance(x, (float, int)) else x.astype(jnp.bfloat16), ckpt)
# model = model.bind(ckpt['state']['params'])

cg1 = load_file(config, 1, pad=True)
cg2 = load_file(config, 2, pad=True)
cg3 = load_file(config, 3, pad=True)
cgs = jax.tree_map(lambda x, y, z: jnp.stack([x, y, z]), cg1, cg2, cg3)


from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

variables = ckpt['state']['params']
rngs = {
    'noise': jr.key(123)
}
from cdv.layers import Context, LazyInMLP

d = 3
mesh = Mesh(mesh_utils.create_device_mesh((d,), devices=jax.devices()[:d]), 'batch')
sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())

variables = jax.device_put(variables, replicated_sharding)

jax.config.update('jax_threefry_partitionable', True)

@jax.jit
def execute(variables, cgs):
    return jax.vmap(lambda cg: model.apply(variables, cg, ctx=Context(training=False), rngs=rngs))(cgs)


m = 20
n = 30
cgs = [load_file(config, i) for i in range(n)]

print('Starting timing now')
start = time.monotonic()

tot = 0
for _ in range(m):
    for i in range(0, n, d):
        cg = jax.tree_map(lambda *args: jnp.stack(args), *cgs[i:i+d])
        cg = jax.device_put(cg, sharding)

        out = execute(variables, cg)['loss']
        tot = tot + out

end = time.monotonic()

print(f'Time: {end - start:.2f}')

print(jax.debug.visualize_array_sharding(tot))
debug_structure(out)