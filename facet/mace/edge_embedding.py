"""Edge embedding layers. The only trainable part is the radial embedding."""

from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Float, Array

from facet.data.metadata import DatasetMetadata
from facet.layers import Context, E3IrrepsArray, Identity
from facet.utils import get_or_init


class RadialBasis(nn.Module):
    """Set of radial basis functions."""

    num_basis: int

    def __call__(
        self, x: Float[Array, '*batch'], r_max, ctx: Context
    ) -> Float[Array, '*batch num_basis']:
        raise NotImplementedError


class Envelope(nn.Module):
    """Cutoff function that goes to 0 at r_max."""

    def __call__(self, x: Float[Array, '*batch'], r_max, ctx: Context) -> Float[Array, '*batch']:
        raise NotImplementedError


class RadialEmbeddingBlock(nn.Module):
    r_max: float
    r_max_trainable: bool
    basis: RadialBasis
    envelope: Envelope
    radius_transform: nn.Module

    def setup(self):
        self.param_rmax = get_or_init(
            self, 'rmax', jnp.array([self.r_max], dtype=jnp.float32), self.r_max_trainable
        )

    def avg_num_neighbors(self, metadata: DatasetMetadata) -> Float[Array, '1']:
        """Estimates average number of neighbors with given cutoff."""
        return metadata.avg_num_neighbors(self.param_rmax)  # type: ignore

    def __call__(self, edge_lengths: Float[Array, '*batch'], ctx: Context) -> E3IrrepsArray:
        """*batch -> *batch num_basis"""

        edge_lengths = self.radius_transform(edge_lengths)

        embedding = (
            self.basis(edge_lengths, self.param_rmax, ctx)
            * self.envelope(edge_lengths, self.param_rmax, ctx)[..., None]
        )

        return E3IrrepsArray(f'{embedding.shape[-1]}x0e', embedding)


class GaussBasis(RadialBasis):
    """Uses equispaced Gaussian RBFs, as in coGN."""

    mu_trainable: bool
    sd_trainable: bool
    mu_max: float
    sd: float

    def setup(self):
        self.mu = get_or_init(
            self,
            'mu',
            jnp.linspace(0, self.mu_max, self.num_basis, dtype=jnp.float32),
            self.mu_trainable,
        )

        self.sigma = get_or_init(
            self, 'sigma', jnp.array([self.sd], dtype=jnp.float32), self.sd_trainable
        )

    def __call__(
        self, d: Float[Array, ' *batch'], r_max, ctx: Context
    ) -> Float[Array, '*batch emb']:
        z = d[..., None] - self.mu
        y = jnp.exp(-(z**2) / (2 * self.sigma**2))
        return y


# class BesselBasis(RadialBasis):
#     """Uses spherical Bessel functions with a cutoff, as in DimeNet++."""

#     freq_trainable: bool = True

#     def setup(self):
#         self.freq = get_or_init(
#             self, 'freq', jnp.arange(self.num_basis, dtype=jnp.float32) + 1, self.freq_trainable
#         )

#     def __call__(self, x, r_max, ctx: Context):
#         dist = x[..., None] / r_max

#         # e(d) = sqrt(2/c) * sin(fπd/c)/d
#         # we use sinc so it's defined at 0
#         # jnp.sinc is sin(πx)/(πx)
#         # e(d) = sqrt(2/c) * sin(πfd/c)/(fd/c) * f/c
#         # e(d) = sqrt(2/c) * sinc(πfd/c)) * πf/c

#         # e_d = jnp.sqrt(2 / r_max) * jnp.sinc(self.freq * dist) * (jnp.pi * self.freq / r_max)
#         e_d = 2 / r_max * jnp.sinc(self.freq * dist) * (jnp.pi * self.freq / r_max)

#         # debug_stat(e_d=e_d, env=env, dist=dist)
#         return e_d


class SincBasis(RadialBasis):
    """Uses spherical Bessel functions with a cutoff, as in DimeNet++.
    Rewrites to use sinc, for hopefully better numerical performance."""

    freq_trainable: bool = True

    def setup(self):
        self.freq = get_or_init(
            self, 'freq', jnp.arange(self.num_basis, dtype=jnp.float32) + 1, self.freq_trainable
        )

    def __call__(self, x, r_max, ctx: Context):
        dist = x[..., None]

        # e(d) = sqrt(2/c) * sin(fπd/c)/d
        # we use sinc so it's defined at 0
        # jnp.sinc is sin(πx)/(πx)
        # e(d) = sqrt(2/c) * sin(πfd/c)/(fd/c) * f/c
        # e(d) = sqrt(2/c) * sinc(πfd/c)) * πf/c

        # e_d = jnp.sqrt(2 / r_max) * jnp.sinc(self.freq * dist) * (jnp.pi * self.freq / r_max)        
        e_d = 2 / r_max * jnp.sinc(self.freq / jnp.pi * dist) * self.freq

        # debug_stat(e_d=e_d, env=env, dist=dist)
        return e_d


class BesselBasis(RadialBasis):
    """
    Uses spherical Bessel functions with a cutoff, as in DimeNet++.
    This version is numerically unstable: it should be used mainly for compatibility with e.g.,
    SevenNet checkpoints, which are trained to expect these oscillations.
    """

    freq_trainable: bool = True

    def setup(self):
        self.freq = get_or_init(
            self, 'freq', jnp.arange(self.num_basis, dtype=jnp.float32) + 1, self.freq_trainable
        )

    def __call__(self, x, r_max, ctx: Context):
        prefactor = 2.0 / r_max
        dist = x[..., None]

        return prefactor * jnp.sin(dist * self.freq) / dist


class PolynomialCutoff(Envelope):
    # https://github.com/ACEsuit/mace/blob/575af0171369e2c19e04b115140b9901f83eb00c/mace/modules/radial.py#L112

    p: float = 6

    @nn.compact
    def __call__(self, x: Float[Array, '*batch'], r_max, ctx: Context) -> Float[Array, '*batch']:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * jnp.pow(x / r_max, self.p)
            + self.p * (self.p + 2.0) * jnp.pow(x / r_max, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * jnp.pow(x / r_max, self.p + 2)
        )

        return envelope * (x < self.r_max)


class ExpCutoff(Envelope):
    """Original envelope. Designed to have as little effect as possible while keeping higher-order
    derivatives low.

    c should be between 0.1 and 1: higher values have smaller third, fourth derivatives but higher
    first, second derivatives.

    cutoff_start determines when the cutoff starts: values below this are not affected at all. This
    is as a fraction of r_max.
    """

    c: float = 0.1
    cutoff_start: float = 0.6

    @nn.compact
    def __call__(self, x: Float[Array, '*batch'], r_max, ctx: Context) -> Float[Array, '*batch']:
        r_on = self.cutoff_start * r_max
        t = jnp.clip((x - r_on) / (r_max - r_on), 0, 1)

        def exp_func(x):
            return jnp.expm1(x * self.c) / jnp.expm1(self.c) * x * x * jnp.sin(jnp.pi * x / 2)

        envelope = 1 - jnp.where(t < 0.5, exp_func(2 * t) / 2, 1 - exp_func(2 - 2 * t) / 2)

        return envelope


# From SevenNet.
class XPLORCutoff(Envelope):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """    

    cutoff_on: float

    @nn.compact
    def __call__(self, r: Float[Array, '*batch'], r_max, ctx: Context) -> Float[Array, '*batch']:
        r_sq = r * r
        r_on = self.cutoff_on * r_max
        r_on_sq = r_on * r_on
        r_cut_sq = r_max * r_max
        env_out = jnp.where(
            r < r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )

        return jnp.where(r < r_max, env_out, 0)