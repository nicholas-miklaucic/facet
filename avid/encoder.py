"""Encoder/decoder for crystal data."""

from typing import Callable

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jaxtyping import Array, Bool, Float, Int

from avid.databatch import DataBatch
from avid.layers import LazyInMLP
from avid.utils import ELEM_VALS, debug_structure, flax_summary, tcheck


class SpeciesEmbed(nn.Module):
    """Species density embedding."""

    n_species: int
    species_embed_dim: int
    embed_module: LazyInMLP
    use_simple_weighting: bool = False

    @tcheck
    @nn.compact
    def __call__(
        self,
        x: Float[Array, ''],
        spec: Int[Array, ''],
        mask: Bool[Array, ''],
        training: bool,
    ):
        spec_embed = nn.Embed(
            self.n_species, self.species_embed_dim, name='species_embed', dtype=x.dtype
        )

        if self.use_simple_weighting:
            return spec_embed(spec) * x
        else:
            input_embeds = jnp.concat([spec_embed(spec), x[..., None]], axis=-1)
            return self.embed_module(input_embeds, training) * mask.astype(x.dtype)


class ReduceSpeciesEmbed(nn.Module):
    species_embed: SpeciesEmbed

    @tcheck
    @nn.compact
    def __call__(self, data: DataBatch, training: bool) -> Float[Array, 'batch nx ny nz _dim_out']:
        spec1 = jax.vmap(
            lambda x, spec, mask: self.species_embed(x=x, spec=spec, mask=mask, training=training)
        )

        spec2 = jax.vmap(spec1, in_axes=-1, out_axes=-1)

        x = jax.lax.collapse(data.density, 1, 4)
        nx, ny, nz = data.density.shape[1:4]
        # debug_structure(x=x, spec=data.species, mask=data.mask)
        return rearrange(
            jax.vmap(spec2, in_axes=(1, None, None), out_axes=1)(x, data.species, data.mask).sum(
                axis=-1
            ),
            'b (nx ny nz) c -> b nx ny nz c',
            nx=nx,
            ny=ny,
            nz=nz,
        )


class Downsample(nn.Module):
    downsample_factor: int
    channel_out: int
    kernel_size: int
    kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.1)
    bias_init: Callable = nn.initializers.truncated_normal()

    def setup(self):
        if (self.kernel_size - 1) // 2 < (self.downsample_factor - 1):
            raise ValueError(f'Configuration {self} would skip data!')

        self.conv = nn.Conv(
            features=self.channel_out,
            kernel_size=[self.kernel_size for _ in range(3)],
            strides=[self.downsample_factor for _ in range(3)],
            padding='CIRCULAR',
            name='conv',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @tcheck
    def __call__(
        self, x: Float[Array, '*batch D H W C']
    ) -> Float[Array, '*batch _d_less _h_less _w_less _c_out']:
        return self.conv(x)


if __name__ == '__main__':
    from avid.config import MainConfig
    from avid.dataset import load_file

    config = MainConfig()
    embed_mlp = LazyInMLP(
        inner_dims=(32,),
    )
    spec_embed = SpeciesEmbed(
        len(ELEM_VALS), species_embed_dim=16, dim_out=8, embed_module=embed_mlp
    )
    downsample = Downsample(downsample_factor=2, channel_out=10, kernel_size=3)
    key = jax.random.key(12345)

    batch = load_file(config, 0)

    kwargs = dict(training=True)

    mod = nn.Sequential([ReduceSpeciesEmbed(spec_embed), downsample])
    out, params = mod.init_with_output(key, data=batch, training=True)

    debug_structure(batch=batch, module=mod, out=out)
    flax_summary(mod, batch, **kwargs)
