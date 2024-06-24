"""Simple module to predict e_form from latent representation. A quick way to benchmark an
encoder."""

from typing import Callable

import jax
import jax.numpy as jnp
import pyrallis
from einops import rearrange
from flax import linen as nn
from jaxtyping import Array, Float

from avid.coord_embeddings import legendre_grid_embeds
from avid.databatch import DataBatch
from avid.encoder import ReduceSpeciesEmbed
from avid.layers import EquivariantMHA, LazyInMLP, PermInvariantEncoder
from avid.utils import debug_structure, flax_summary, tcheck


class Patchify(nn.Module):
    """Patchify block in 3D, with lazy patch size."""

    dim_out: int
    kernel_init: Callable = nn.initializers.xavier_normal()
    bias_init: Callable = nn.initializers.normal()

    @tcheck
    @nn.compact
    def __call__(self, patch: Float[Array, 'p p p chan_in']) -> Float[Array, '_dimout']:
        patch = rearrange(patch, 'p1 p2 p3 chan_in -> (p1 p2 p3 chan_in)')

        patch_embed = nn.Dense(
            self.dim_out,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name='patch_proj',
            dtype=jnp.bfloat16,
        )

        return patch_embed(patch)


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs.

    Attributes:
    posemb_init: positional embedding initializer.
    """

    posemb_init: Callable = nn.initializers.normal(stddev=0.02, dtype=jnp.bfloat16)

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionEmbs module.

        Args:
            inputs: Inputs to the layer.

        Returns:
            Output tensor with shape `(timesteps, in_dim)`.
        """
        pos_emb_shape = inputs.shape
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        return inputs + pe


class LegendrePosEmbed(nn.Module):
    """Module to apply fixed Legendre positional encodings."""

    max_input_size: int
    dim_embed: int

    @tcheck
    @nn.compact
    def __call__(
        self, patches: Float[Array, 'I I I patch_d']
    ) -> Float[Array, 'I I I _patchd_posd']:
        embed = self.param(
            'embed', lambda *args: legendre_grid_embeds(patches.shape[0], self.dim_embed)
        )
        step = embed.shape[0] // patches.shape[0]
        return jnp.concat([patches, embed[::step, ::step, ::step, :]], axis=-1)


class SingleImageEmbed(nn.Module):
    """Embeds a 3D image into a sequnce of tokens."""

    patch_size: int
    patch_latent_dim: int
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

        embed = jax.vmap(Patchify(self.patch_latent_dim))(patches)
        # legendre
        # pos_embed = self.pos_embed(rearrange(embed, '(i1 i2 i3) c -> i1 i2 i3 c', i1=n, i2=n,
        # i3=n))
        # return rearrange(pos_embed, 'i1 i2 i3 c -> (i1 i2 i3) c', i1=n, i2=n, i3=n)

        # learned
        pos_embed = self.pos_embed(embed)
        return pos_embed


class ImageEmbed(nn.Module):
    inner: SingleImageEmbed

    @tcheck
    @nn.compact
    def __call__(self, im: Float[Array, 'batch I I I C']) -> Float[Array, 'batch _ip3 _dim_out']:
        return jax.vmap(self.inner)(im)


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    num_heads: int
    mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    equivariant: bool = False

    @nn.compact
    def __call__(self, inputs, abys, *, training: bool):
        """Applies Encoder1DBlock module.

        abys is a six-element array α1, β1, γ1, α2, β2, γ2, that applies affine scaling.
        [1, 0, 1, 1, 0, 1] is the identity. Of shape 6 1 1 dim_tok, or 6 1 1 1.
        """

        a1, b1, y1, a2, b2, y2 = abys

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        # zero out initializations: will just apply the identity function at first
        x = nn.LayerNorm(scale_init=nn.zeros, dtype=inputs.dtype)(inputs)
        x = x * y1 + b1

        if self.equivariant:
            x = EquivariantMHA(num_heads=self.num_heads)(x)
        else:
            x = nn.MultiHeadDotProductAttention(
                kernel_init=nn.initializers.xavier_uniform(),
                broadcast_dropout=False,
                deterministic=not training,
                dropout_rate=self.attention_dropout_rate,
                num_heads=self.num_heads,
                dtype=jnp.bfloat16,
            )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = x * a1
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(scale_init=nn.zeros, dtype=x.dtype)(x)
        y = y * y2 + b2

        mlp = self.mlp.copy(out_dim=x.shape[2])
        y = jax.vmap(jax.vmap(lambda yy: mlp(yy, training=training)))(y)
        y = y * a2

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    num_heads: int
    mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    equivariant: bool = False

    @nn.compact
    def __call__(self, x, abys, *, training: bool):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          abys: Affine scaling.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f'encoderblock_{lyr}',
                num_heads=self.num_heads,
                mlp=self.mlp.copy(),
                equivariant=self.equivariant,
            )(x, abys, training=training)
        encoded = nn.LayerNorm(name='encoder_norm', dtype=x.dtype)(x)

        return encoded


ABY_IDENTITY = jnp.array([1, 0, 1, 1, 0, 1]).reshape(6, 1, 1, 1)


class ViTRegressor(nn.Module):
    spec_embed: ReduceSpeciesEmbed
    downsample: nn.Module
    im_embed: nn.Module
    encoder: Encoder
    head: LazyInMLP
    equivariant: bool = False

    @nn.compact
    def __call__(self, im: DataBatch, training: bool, abys=ABY_IDENTITY):
        out = self.spec_embed(im, training=training)
        out = self.downsample(out)

        out = self.im_embed(out)

        encoder = self.encoder.copy(equivariant=self.equivariant)
        out = encoder(out, abys, training=training)

        # batch seq dim -> batch dim*2
        out = PermInvariantEncoder()(out, axis=1, keepdims=False)
        out = nn.LayerNorm(dtype=out.dtype)(out)
        # debug_structure(out=out)
        out = jax.vmap(lambda x: self.head(x, training=training))(out)
        return out


if __name__ == '__main__':
    from avid.config import MainConfig
    from avid.dataset import load_file

    config = pyrallis.argparsing.parse(MainConfig, 'configs/smoke_test.toml')

    if config.do_profile:
        jax.profiler.start_trace('/tmp/jax-trace', create_perfetto_link=True)

    kwargs = dict(training=False)
    batch = load_file(config, 0)
    mod = config.vit.build()
    out, params = mod.init_with_output(jax.random.key(0), im=batch, **kwargs)
    debug_structure(batch=batch, module=mod, out=out)
    flax_summary(mod, batch, **kwargs)

    if config.do_profile:
        jax.profiler.stop_trace()
