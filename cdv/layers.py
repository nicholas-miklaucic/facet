"""Layers useful in different contexts."""

from typing import Callable, Literal, Optional, Sequence

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from eins import EinsOp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float

from cdv.utils import tcheck

E3Irreps = e3nn.Irreps
E3IrrepsArray = e3nn.IrrepsArray


import yaml
from yaml import Node, SafeDumper


def represent_irreps(d: SafeDumper, e: E3Irreps) -> Node:
    return d.represent_str(str(e))


def represent_irrep_array(d: SafeDumper, e: E3IrrepsArray) -> Node:
    return d.represent_dict({'irreps': e.irreps, 'array': e.array})


yaml.SafeDumper.add_representer(E3Irreps, represent_irreps)
yaml.SafeDumper.add_representer(E3IrrepsArray, represent_irrep_array)


def edge_vecs(cg):
    """CG -> nodes k xyz"""
    # print(cg)
    send_pos = cg.nodes.cart[:, None, :]  # nodes 1 3
    offsets = EinsOp('nodes abc xyz, nodes k abc -> nodes k xyz')(
        cg.graph_data.lat[cg.nodes.graph_i], cg.edges.to_jimage
    )
    recv_pos = cg.nodes.cart[cg.receivers] + offsets  # nodes k 3
    vecs = recv_pos - send_pos
    return vecs


class Context(struct.PyTreeNode):
    training: bool


SegmentReductionKind = Literal['max', 'min', 'prod', 'sum', 'mean']


def segment_mean(data, segment_ids, **kwargs):
    return jax.ops.segment_sum(data, segment_ids, **kwargs) / (
        1e-6 + jax.ops.segment_sum(jnp.ones_like(data), segment_ids, **kwargs)
    )


def segment_reduce(reduction: SegmentReductionKind, data, segment_ids, **kwargs):
    try:
        fn = getattr(jax.ops, f'segment_{reduction}')
    except AttributeError:
        if reduction == 'mean':
            fn = segment_mean
        else:
            raise ValueError('Cannot find reduction')

    return fn(data, segment_ids, **kwargs)


class SegmentReduction(nn.Module):
    reduction: SegmentReductionKind = 'sum'

    def __call__(self, data, segments, num_segments, ctx):
        return segment_reduce(self.reduction, data, segments, num_segments=num_segments)


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
    normalization: Literal['layer', 'weight', 'none'] = 'layer'

    @tcheck
    @nn.compact
    def __call__(self, x: Float[Array, ' n_in'], ctx: Context):
        if isinstance(x, e3nn.IrrepsArray):
            if not x.irreps.is_scalar():
                raise ValueError('MLP only works on scalar (0e) input.')
            x = x.array
            output_irrepsarray = True
        else:
            output_irrepsarray = False

        orig_x = x
        _curr_dim = x.shape[-1]
        if self.out_dim is None:
            out_dim = _curr_dim
        else:
            out_dim = self.out_dim

        if self.normalization == 'weight':
            Dense = lambda *args, **kwargs: nn.WeightNorm(nn.Dense(*args, **kwargs))
        else:
            Dense = nn.Dense

        for next_dim in self.inner_dims:
            x = Dense(
                next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
            )(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not ctx.training)(x)
            if self.normalization == 'layer':
                x = nn.LayerNorm(dtype=x.dtype)(x)
            _curr_dim = next_dim

        x = nn.Dense(
            out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
        )(x)
        if self.residual:
            if orig_x.shape[-1] != out_dim:
                x_resid = Dense(
                    out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=x.dtype
                )(orig_x)
            else:
                x_resid = orig_x

            x = x + x_resid

        x = self.final_act(x)

        if output_irrepsarray:
            x = e3nn.IrrepsArray(e3nn.Irreps(f'{x.shape[-1]}x0e'), x)

        return x


class E3NormNorm(nn.Module):
    eps: float = 1e-6

    def __call__(self, x: 'E3IrrepsArray') -> 'E3IrrepsArray':
        """
        Normalizes the norm of each irrep to be mean 1.
        """
        return x
        normed = []

        for chunk in x.chunks:
            if chunk is None:
                normed.append(None)
            else:
                norm = jnp.sum(jnp.conj(chunk) * chunk, axis=-1, keepdims=True)
                norm = jnp.nanmean(
                    jnp.where(norm < self.eps**2, jnp.nan, jnp.sqrt(norm + self.eps)), keepdims=True
                )
                normed.append(chunk / (norm + self.eps))

        return e3nn.from_chunks(
            x.irreps,
            normed,
            x.shape[:-1],
            x.dtype,
        )


class DistanceEncoder(nn.Module):
    """Converts a scalar distance to an embedding."""

    def __call__(self, d: Float[Array, ' batch'], ctx: Context) -> Float[Array, 'batch emb']:
        raise NotImplementedError


class GaussBasis(DistanceEncoder):
    """Uses equispaced Gaussian RBFs, as in coGN."""

    lo: float = 0
    hi: float = 8
    sd: float = 1
    emb: int = 32

    def setup(self):
        self.locs = jnp.linspace(self.lo, self.hi, self.emb)

    def __call__(self, d: Float[Array, ' batch'], ctx: Context) -> Float[Array, 'batch emb']:
        z = d[:, None] - self.locs[None, :]
        y = jnp.exp(-(z**2) / (2 * self.sd**2))
        return y


# different than
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
# it seems like what I do follows the paper better?


class Envelope(nn.Module):
    """Polynomial envelope that goes to 0 at a cutoff smoothly."""

    # they seem to take p - 1 as input, which seems to me a little harebrained and fails for nans
    # we just take in p and use the formula directly
    p: int

    def setup(self):
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def __call__(self, x):
        # our model shouldn't return nan, even for 0 input, so we have to change this
        return 1 + x**self.p * (self.a + x * (self.b + x * self.c))


class OldBessel1DBasis(DistanceEncoder):
    """Uses spherical Bessel functions with a cutoff, as in DimeNet++."""

    num_basis: int = 7
    cutoff: float = 7
    # Controls how fast the envelope goes to 0 at the cutoff.
    envelope_exp: int = 6
    freq_trainable: bool = True

    def setup(self):
        def freq_init(rng):
            return jnp.arange(self.num_basis, dtype=jnp.float32) + 1

        if self.freq_trainable:
            self.freq = self.param('freq', freq_init)
        else:
            self.freq = freq_init(None)
        self.envelope = Envelope(self.envelope_exp)

    def __call__(self, x, ctx: Context):
        dist = x[..., None] / self.cutoff
        env = self.envelope(dist)

        # e(d) = sqrt(2/c) * sin(fπd/c)/d
        # we use sinc so it's defined at 0
        # jnp.sinc is sin(πx)/(πx)
        # e(d) = sqrt(2/c) * sin(πfd/c)/(fd/c) * f/c
        # e(d) = sqrt(2/c) * sinc(πfd/c)) * πf/c

        e_d = (
            jnp.sqrt(2 / self.cutoff)
            * jnp.sinc(self.freq * dist)
            * (jnp.pi * self.freq / self.cutoff)
        )

        # debug_stat(e_d=e_d, env=env, dist=dist)
        return env * e_d


class Bessel2DBasis(nn.Module):
    num_radial: int = 7
    num_spherical: int = 7
    cutoff: float = 7
    # Controls how fast the envelope goes to 0 at the cutoff.
    envelope_exp: int = 6

    def setup(self):
        self.envelope = Envelope(self.envelope_exp)
        self.radial = OldBessel1DBasis(
            num_basis=self.num_radial, cutoff=self.cutoff, envelope_exp=self.envelope_exp
        )

    def __call__(self, d, alpha, ctx):
        # TODO implement this for real
        dist_emb = self.radial(d, ctx) / self.radial(d * 0, ctx)  # batch radial
        ang_emb = jnp.cos(alpha[:, None] * jnp.arange(self.num_spherical))  # batch spherical
        return EinsOp('batch radial, batch spherical -> batch (radial spherical)')(
            dist_emb, ang_emb
        )


def shifted_softplus(x):
    """Shifted softplus activation used for SevenNet weight neural network.
    Not sure why. Equal to softplus(x) - log 2"""
    return jax.nn.softplus(x) - jnp.log(2.0)


# def soft_envelope(length, max_length):
#     return e3nn.soft_envelope(
#         length,
#         max_length,
#         arg_multiplicator=envelope_arg_multiplicator,
#         value_at_origin=envelope_value_at_origin,
#     )

# def polynomial_envelope(length, max_length, degree0: int, degree1: int):
#     return e3nn.poly_envelope(degree0, degree1, max_length)(length)

# def u_envelope(length, max_length, p: int):
#     return e3nn.poly_envelope(p - 1, 2, max_length)(length)


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
    def __call__(
        self, inputs: Float[Array, 'batch seq chan'], abys: Float[Array, '6 chan'], ctx: Context
    ) -> Float[Array, 'batch seq chan']:
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
            )(x, abys, ctx=ctx)
        # Output Head
        x = nn.LayerNorm(dtype=x.dtype, name='pre_head_layer_norm')(x)
        return x


class DeepSetEncoder(nn.Module):
    """Deep Sets with several types of pooling. Permutation-invariant encoder."""

    phi: nn.Module

    @nn.compact
    def __call__(
        self, x: Float[Array, 'batch token chan'], training: bool
    ) -> Float[Array, 'batch out_dim']:
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
    def __call__(
        self, x: Float[Array, 'batch token chan'], axis=-1, keepdims=True
    ) -> Float[Array, 'batch out_dim']:
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
