import jax 
import jax.numpy as jnp
import jax.random as jr

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

d = 3
mesh = Mesh(mesh_utils.create_device_mesh((d,), devices=jax.devices()[:d]), 'batch')
sharding = NamedSharding(mesh, P('batch', None))

tree = {
    'a': jr.normal(jr.key(123), (3, 64)),
    'b': jr.normal(jr.key(123), (3, 32, 32)),
    'c': jr.normal(jr.key(123), (3, 1))
}

tree = jax.device_put(tree, sharding)

jax.debug.visualize_array_sharding(tree['a'])
jax.debug.visualize_array_sharding(tree['a'][:1])
jax.debug.visualize_array_sharding(tree['a'][:, 0])


