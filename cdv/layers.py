"""Layers useful in different contexts."""

import functools
from typing import Callable, Optional, Sequence

import einops
from eins import EinsOp
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float

from cdv.utils import debug_structure, tcheck


class Context(struct.PyTreeNode):
    training: bool

class Identity(nn.Module):
    """Module that does nothing, used so model summaries accurately communicate the purpose."""

    @nn.compact
    def __call__(self, x):
        return x


class LazyInMLP(nn.Module):
    """Customizable MLP with input dimension inferred at runtime."""

    inner_dims: Sequence[int]
    out_dim: Optional[int] = None
    residual: bool = False
    inner_act: Callable = nn.gelu
    final_act: Callable = Identity()
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.truncated_normal()

    @tcheck
    @nn.compact
    def __call__(self, x: Float[Array, 'n_in'], ctx: Context):
        orig_x = x
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
            x = nn.Dropout(self.dropout_rate, deterministic=not ctx.training)(x)
            x = nn.LayerNorm(dtype=x.dtype)(x)
            _curr_dim = next_dim

        x = nn.Dense(
            out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
        )(x)        
        if self.residual:
            if orig_x.shape[-1] != out_dim:
                x_resid = nn.Dense(out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype)(orig_x)
            else:
                x_resid = orig_x
            
            x = x + x_resid

        x = self.final_act(x)
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
    def __call__(self, inputs: Float[Array, 'batch seq chan'], abys: Float[Array, '6 chan'], ctx: Context) -> Float[Array, 'batch seq chan']:
        training = ctx.training
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
    def __call__(self, inputs, abys, *, ctx: Context) -> Array:
        x = inputs
        # Num Blocks x Mixer Blocks
        for _ in range(self.num_layers):
            x = MixerBlock(
                tokens_mlp=self.tokens_mlp.copy(),
                channels_mlp=self.channels_mlp.copy(),
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
            )(x, abys, training=ctx.training)
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
