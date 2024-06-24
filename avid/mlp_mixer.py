from typing import Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import linen as nn
from jaxtyping import Array, Float

from avid.databatch import DataBatch
from avid.encoder import ReduceSpeciesEmbed
from avid.layers import DeepSetEncoder, LazyInMLP, MLPMixer, PermInvariantEncoder
from avid.utils import tcheck
from avid.vit import ABY_IDENTITY

# equivariance to rotations/reflections distinguishes only a few points: the center, the face centers, the
# corners, and the diagonals that are parallel to a face.

kinds = np.array(
    [
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
        [[2, 3, 2], [3, 0, 3], [2, 3, 2]],
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
    ]
).reshape(-1)

basis = [kinds == k for k in range(max(kinds) + 1)]
basis = jnp.array(basis, dtype=jnp.bfloat16)


class O3Patchify(nn.Module):
    """Patchify block in 3D, with lazy patch size. Invariant to rotations/reflections."""

    dim_out: int
    inner_dim: int = 4
    kernel_init: Callable = nn.initializers.xavier_normal()
    bias_init: Callable = nn.initializers.normal()

    @tcheck
    @nn.compact
    def __call__(self, patch: Float[Array, 'p p p chan_in']) -> Float[Array, '_dimout']:
        patch = rearrange(patch, 'p1 p2 p3 chan_in -> chan_in (p1 p2 p3)')

        kernel = self.param('kernel', self.kernel_init, (self.inner_dim, basis.shape[0]))
        # QK is out_dim x out_dim
        # is this the most efficient way to do this?
        out = einops.einsum(basis, kernel, patch, 'sub d1, inner sub, chan d1 -> chan inner')
        out = out.reshape(-1)

        patch_embed = nn.Dense(
            self.dim_out,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name='patch_proj',
            dtype=jnp.bfloat16,
        )

        return patch_embed(out)


class O3ImageEmbed(nn.Module):
    """Embeds a 3D image into a sequnce of tokens."""

    patch_size: int
    patch_latent_dim: int
    patch_heads: int
    pos_embed: nn.Module

    @tcheck
    @nn.compact
    def __call__(self, im: Float[Array, 'I I I C']) -> Float[Array, '_ip3 _dimout']:
        i = im.shape[0]
        p = self.patch_size
        c = im.shape[-1]
        assert i % p == 0
        n = i // p

        patches = im.reshape(n, p, n, p, n, p, c)
        patches = rearrange(patches, 'n1 p1 n2 p2 n3 p3 c -> (n1 n2 n3) p1 p2 p3 c')

        embed = jax.vmap(O3Patchify(self.patch_latent_dim, inner_dim=self.patch_heads))(patches)
        # legendre
        # pos_embed = self.pos_embed(rearrange(embed, '(i1 i2 i3) c -> i1 i2 i3 c', i1=n, i2=n,
        # i3=n))
        # return rearrange(pos_embed, 'i1 i2 i3 c -> (i1 i2 i3) c', i1=n, i2=n, i3=n)

        # learned
        pos_embed = self.pos_embed(embed)
        return pos_embed


class MLPMixerRegressor(nn.Module):
    spec_embed: ReduceSpeciesEmbed
    downsample: nn.Module
    im_embed: O3ImageEmbed
    mixer: MLPMixer
    head: LazyInMLP
    perm_head: LazyInMLP

    @nn.compact
    def __call__(self, im: DataBatch, training: bool, abys=ABY_IDENTITY):
        out = self.spec_embed(im, training=training)
        out = self.downsample(out)
        out = self.im_embed(out)
        out = self.mixer(out, training=training, abys=abys)

        # out = reduce(out, 'batch seq dim -> batch dim', 'mean')
        out = DeepSetEncoder(phi=self.perm_head)(out, training=training)
        # debug_structure(out=out)
        out = self.head(out, training=training)
        return out
