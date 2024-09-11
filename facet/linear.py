"""Linear layer implementations."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from flax import linen as nn

from facet.layers import Context

class AbstractLinear(nn.Module):
    """Linear layer."""
    in_dim: int
    out_dim: int

    def forward(self, x: Float[Array, 'in'], ctx: Context) -> Float[Array, ' out']:
        """Forward transformation."""
        raise NotImplementedError()

    def __call__(self, x: Float[Array, '*batch in'], ctx: Context) -> Float[Array, '*batch out']:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f'Shape is incorrect: {x.shape} does not match {self.in_dim}')
        
        xr = x.reshape(-1, x.shape[-1])
        yr = jax.vmap(lambda x: self.forward(x, ctx=ctx))(xr)

        return yr.reshape(*x.shape[:-1], yr.shape[-1])
    

class DenseLinear(AbstractLinear):
    param_dtype: jnp.dtype = jnp.float32
    def setup(self):
        self.dense = nn.Dense(self.out_dim, use_bias=False, param_dtype=self.param_dtype)

    def forward(self, x: Float[Array, 'in'], ctx: Context) -> Float[Array, ' out']:
        return self.dense(x)
    
    
class LowRank(AbstractLinear):
    param_dtype: jnp.dtype = jnp.float32
    rank: int | str = 'sqrt'
    def setup(self):
        if self.rank == 'sqrt':
            rank = max(2, round(self.out_dim ** 0.5))
        else:
            rank = self.rank

        self.down = nn.Dense(rank, use_bias=False, param_dtype=self.param_dtype)
        self.up = nn.Dense(self.out_dim, use_bias=False, param_dtype=self.param_dtype)

    def forward(self, x: Float[Array, 'in'], ctx: Context) -> Float[Array, ' out']:
        return self.up(self.down(x))
    

# kaleidoscope