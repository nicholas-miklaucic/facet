"""
Modules that perform node embedding: given species information, constructing embeddings.
Global data would also be incorporated here, but the two are mostly orthogonal.
"""

from flax import linen as nn
from cdv.mace.e3_layers import E3Irreps, E3IrrepsArray, IrrepsModule
from cdv.layers import Context
from jaxtyping import Float, Int, Array
import json
import numpy as np
import jax
import jax.numpy as jnp


class NodeEmbedding(IrrepsModule):
    """Initializes node embeddings."""

    def __call__(self, species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        """nodes -> nodes irreps_out"""
        raise NotImplementedError


class LinearNodeEmbedding(NodeEmbedding):
    """Standard embedding layer."""

    num_species: int
    element_indices: Int[Array, ' max_species']

    def setup(self):
        if self.ir_out.lmax > 0:
            raise ValueError(f'Irreps {self.irreps_out} should just be scalars for node embedding.')
        self.out_dim = self.irreps_out.dim
        self.embed = nn.Embed(self.num_species, self.out_dim)

    def __call__(self, node_species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        return E3IrrepsArray(self.ir_out, self.embed(self.element_indices[node_species]))


class SevenNetEmbedding(NodeEmbedding):
    """Embedding locked to a projection from the SevenNet parameters."""

    def setup(self):
        if self.ir_out.lmax > 0:
            raise ValueError(
                f'Irreps {self.ir_out.regroup()} should just be scalars for node embedding.'
            )
        with open('data/sevennet_stats.json', 'r') as stats_file:
            stats = json.load(stats_file)

        with jax.ensure_compile_time_eval():
            self.inds = jnp.zeros((max(stats['atomic_numbers']) + 1,), dtype=jnp.uint32)
            self.inds = self.inds.at[jnp.array(stats['atomic_numbers'], dtype=jnp.uint32)].set(
                jnp.arange(len(stats['atomic_numbers']), dtype=jnp.uint32)
            )
            self.emb = jnp.array(np.load('data/sevennet_embs.npy'))

        self.proj = nn.Dense(self.ir_out.dim, use_bias=True)

    def __call__(self, node_species: Int[Array, ' nodes'], ctx: Context) -> E3IrrepsArray:
        return E3IrrepsArray(self.ir_out, self.proj(self.emb[self.inds[node_species]]))
