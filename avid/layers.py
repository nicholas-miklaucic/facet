"""Layers useful in different contexts."""

import functools
from typing import Callable, Optional, Sequence

import einops
from eins import EinsOp
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float

from avid.utils import debug_structure, tcheck


class Identity(nn.Module):
    """Module that does nothing, used so model summaries accurately communicate the purpose."""

    @nn.compact
    def __call__(self, x):
        return x


class LazyInMLP(nn.Module):
    """Customizable MLP with input dimension inferred at runtime."""

    inner_dims: Sequence[int]
    out_dim: Optional[int] = None
    inner_act: Callable = nn.gelu
    final_act: Callable = Identity()
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.truncated_normal()

    @tcheck
    @nn.compact
    def __call__(self, x: Float[Array, 'n_in'], training: bool):
        _curr_dim = x.shape[-1]
        if self.out_dim is None:
            out_dim = _curr_dim
        else:
            out_dim = self.out_dim

        for next_dim in self.inner_dims:
            x = nn.Dense(
                next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
            )(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm(dtype=x.dtype)(x)
            _curr_dim = next_dim

        x = nn.Dense(
            out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
        )(x)
        x = self.final_act(x)
        return x


# 8^6 x subspace_dim

with jax.default_device(jax.devices('cpu')[0]):
    Q = jnp.load('precomputed/equiv_basis_8.npy')
    ng6, subspace_dim = Q.shape
    ng = round(ng6 ** (1 / 6))
    assert ng**6 == ng6
    Q = Q.reshape(ng**3, ng**3, subspace_dim).astype(jnp.bfloat16)
    conv_kernel = Q[0, ...].reshape(ng, ng, ng, 35)
    counts = jnp.sum(conv_kernel, axis=(0, 1, 2))


with jax.default_device(jax.devices('cpu')[0]):
    import numpy as np
    batch = 4
    chan = 3
    n = 8
    dtype = jnp.bfloat16
    
    ijk = jnp.transpose(jnp.mgrid[0:n, 0:n, 0:n].astype(jnp.int16)).reshape(-1, 3)
    sub = ijk[:, None, :] - ijk[None, :, :]

    idx = jnp.sort(jnp.minimum(jnp.abs(sub), n - jnp.abs(sub)), axis=-1)
    orig = idx.shape

    # uniq, inv = jnp.unique(idx.reshape(-1, 3), axis=0, return_inverse=True)
    # print(uniq[:5, ...])
    # print(uniq.shape)

    dmax = (n) // 2
    inv2 = (idx @ jnp.array([dmax ** 2, dmax, 1])).astype(jnp.int16)
    inv2_i, inv2, counts = jnp.unique(inv2.reshape(-1), return_inverse=True, return_counts=True)
    # jnp.allclose(inv2, inv)

    basis = jax.nn.one_hot(inv2, inv2_i.shape[0], dtype=jnp.bool_).reshape(*orig[:-1], inv2_i.shape[0])
    conv_kernel = basis[0, :, :]


class EquivariantMHA(nn.Module):
    """Equivariant multi-head attention."""

    num_heads: int
    normalize_qk: bool = True

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Equivariant attention.

        Parameters
        ----------
        x : Array[batch, seq, dim]
            Data.
        training : bool, optional
            Training mode.

        Returns
        -------
            Array[batch, seq, dim]
        """
        # https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#MultiHeadAttention

        in_features = x.shape[-1]
        head_dim = in_features // self.num_heads
        if in_features % self.num_heads != 0:
            msg = f'Input dim {in_features} not divisible by number of heads {self.num_heads}'
            raise ValueError(msg)

        dense = functools.partial(
            nn.DenseGeneral, axis=-1, features=(self.num_heads, head_dim), dtype=x.dtype
        )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name='query')(x),
            dense(name='key')(x),
            dense(name='value')(x),
        )

        if self.normalize_qk:
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = nn.LayerNorm(name='query_ln', use_bias=False, dtype=x.dtype)(query)
            key = nn.LayerNorm(name='key_ln', use_bias=False, dtype=x.dtype)(key)

        # equivariant attention bias, different for each head
        relative_attn = self.param(
            'relative_attn',
            nn.initializers.truncated_normal(0.02, dtype=x.dtype),
            (subspace_dim, self.num_heads),
        )

        relative_attn_expanded = einops.einsum(
            relative_attn, Q, 'subspace heads, seq1 seq2 subspace -> heads seq1 seq2'
        )

        # debug_structure(x=x, q=query, attn_exp=relative_attn_expanded)

        x = nn.dot_product_attention(query, key, value, bias=relative_attn_expanded)

        out = nn.DenseGeneral(in_features, axis=(-2, -1), dtype=x.dtype, name='out')(x)
        return out


class EquivariantLinear(nn.Module):
    kernel_init: Callable = nn.initializers.normal(stddev=1e-2, dtype=jnp.bfloat16)
    num_heads: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, 'batch seq chan']) -> Float[Array, 'batch seq chan']:
        """
        """
        batch, out_dim, chan = x.shape

        assert chan % self.num_heads == 0, f'{chan} must divide {self.num_heads}: {x.shape}'
        assert out_dim**2 == ng6, f'{x.shape} is not valid!'

        conv = nn.Conv(
            features=1,
            kernel_size=(ng, ng, ng),
            use_bias=False,
            padding='VALID',
            dtype=jnp.bfloat16,
        )

        # debug_structure(x=x)
        x_im = EinsOp('batch (8 8 8) ch*h -> h (batch ch) 8 8 8', symbol_values={'h': self.num_heads})(x)
        
        kn = 2 * ng - 1
        x_im = jnp.tile(x_im, (1, 1, 2, 2, 2))[:, :, :kn, :kn, :kn, None]        
        # x_im = einops.rearrange(
        #     x, 'batch (n1 n2 n3) chan -> batch n1 n2 n3 chan', n1=ng, n2=ng, n3=ng
        # )
        kernel = self.param('kernel', self.kernel_init, (subspace_dim, self.num_heads), jnp.bfloat16)        
        kernel_expanded = EinsOp('(n=8 n n) s, s h -> h n n n 1 1')(conv_kernel, kernel)
        

        def conv(lhs, rhs):
            return jax.lax.conv_general_dilated(
                lhs,
                rhs,
                window_strides=(1, 1, 1),
                padding='VALID',
                dimension_numbers=('NXYZC', 'XYZIO', 'NXYZC')
            )

        conv_out = jax.vmap(conv)(x_im, kernel_expanded).squeeze(-1)
        # conv_out = jnp.roll(conv_out, (ng // 2, ng // 2, ng // 2), (-3, -2, -1))
        # conv_out = conv.apply({'params': {'kernel': kernel_expanded[:, :, :, 1, :, :]}}, x_im)        
        out_op = EinsOp('h (batch ch) n n n -> batch (n n n) (ch*h)', symbol_values={'batch': batch})        
        out = out_op(conv_out)                        
        return out


class EquivariantMixerMLP(nn.Module):
    num_heads: int = 1
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Float[Array, 'batch seq chan'], training: bool) -> Float[Array, 'batch seq chan']:
        x = nn.LayerNorm(dtype=x.dtype)(x)
        x = EquivariantLinear(num_heads=self.num_heads)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = self.activation(x)
        x = EquivariantLinear(num_heads=self.num_heads)(x)
        return x


class MixerBlock(nn.Module):
    """
    A Flax Module to act as the mixer block layer for the MLP-Mixer Architecture.

    Attributes:
        tokens_mlp_dim: MLP Block 1
        channels_mlp_dim: MLP Block 2
    """

    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs: Float[Array, 'batch seq chan'], abys: Float[Array, '6 chan'], training: bool) -> Float[Array, 'batch seq chan']:
        a1, b1, y1, a2, b2, y2 = abys
        x = nn.LayerNorm(scale_init=nn.zeros, dtype=inputs.dtype)(inputs)
        x = x * y1 + b1

        x = self.tokens_mlp(x, training=training)
        x = nn.Dropout(rate=self.attention_dropout_rate)(x, deterministic=not training)
        x = x * a1
        x = x + inputs


        y = nn.LayerNorm(dtype=x.dtype, scale_init=nn.zeros)(x)
        y = y * y2 + b2
        y = self.channels_mlp(y, training=training)
        y = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        y = y * a2

        return x + y


class MLPMixer(nn.Module):
    """
    Flax Module for the MLP-Mixer Architecture.

    Attributes:
        patches: Patch configuration
        num_classes: No of classes for the output head
        num_blocks: No of Blocks of Mixers to use
        hidden_dim: No of Hidden Dimension for the Patch-Wise Convolution Layer
        tokens_mlp_dim: No of dimensions for the MLP Block 1
        channels_mlp_dim: No of dimensions for the MLP Block 2
        approximate: If True, uses the approximate formulation of GELU in each MLP Block
        dtype: the dtype of the computation (default: float32)
    """

    num_layers: int
    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, abys, *, training: bool) -> Array:
        x = inputs
        # Num Blocks x Mixer Blocks
        for _ in range(self.num_layers):
            x = MixerBlock(
                tokens_mlp=self.tokens_mlp.copy(),
                channels_mlp=self.channels_mlp.copy(),
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
            )(x, abys, training=training)
        # Output Head
        x = nn.LayerNorm(dtype=x.dtype, name='pre_head_layer_norm')(x)
        return x


class DeepSetEncoder(nn.Module):
    """Deep Sets with several types of pooling. Permutation-invariant encoder."""
    phi: nn.Module

    @nn.compact
    def __call__(self, x: Float[Array, 'batch token chan'], training: bool) -> Float[Array, 'batch out_dim']:
        phi_x = self.phi(x, training=training)
        phi_x = EinsOp('batch token out_dim -> batch out_dim token')(phi_x)
        op = 'batch out_dim token -> batch out_dim'
        phi_x_mean = EinsOp(op, reduce='mean')(phi_x)
        phi_x_std = EinsOp(op, reduce='std')(phi_x)
        phi_x = jnp.concatenate([phi_x_mean, phi_x_std], axis=-1)
        normed = nn.LayerNorm(dtype=x.dtype)(phi_x)
        return normed


class PermInvariantEncoder(nn.Module):
    """A la Deep Sets, constructs a permutation-invariant representation of the inputs based on aggregations.
    Uses mean, std, and differentiable quantiles."""

    @nn.compact
    def __call__(self, x: Float[Array, 'batch token chan'], axis=-1, keepdims=True) -> Float[Array, 'batch out_dim']:
        x = EinsOp('batch token chan -> batch chan token')(x)
        x_mean = jnp.mean(x, axis=axis, keepdims=keepdims)
        # x_std = jnp.std(x, axis=axis, keepdims=keepdims)

        # x_whiten = (x - x_mean) / (x_std + 1e-8)

        # x_quants = []
        # for power in jnp.linspace(1, 3, 6):
        #     x_quants.append(
        #         jnp.mean(
        #             jnp.sign(x_whiten) * jnp.abs(x_whiten**power),
        #             axis=-1,
        #             keepdims=True,
        #         )
        #         ** (1 / power),
        #     )

        # This has serious numerical stability problems in the backward pass. Instead, I'll use something else.
        # eps = 0.02
        # quants = jnp.linspace(eps, 1 - eps, 14, dtype=jnp.bfloat16)
        # from ott.tools.soft_sort import quantile
        # x_quants = quantile(x, quants, axis=-1, weight=10 / x.shape[-1])
        return jnp.concat([x_mean], axis=-1)
