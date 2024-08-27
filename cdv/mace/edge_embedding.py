"""Edge embedding layers. The only trainable part is the radial embedding."""

from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Float, Array

from cdv.layers import Context, E3IrrepsArray


class RadialBasis(nn.Module):
    """Set of radial basis functions."""

    num_basis: int

    def __call__(self, x: Float[Array, '*batch'], ctx: Context) -> Float[Array, '*batch num_basis']:
        raise NotImplementedError


class Envelope(nn.Module):
    """Cutoff function that goes to 0 at r_max."""

    r_max: float

    def __call__(self, x: Float[Array, '*batch'], ctx: Context) -> Float[Array, '*batch']:
        raise NotImplementedError


class RadialEmbeddingBlock(nn.Module):
    basis: RadialBasis
    envelope: Envelope

    @nn.compact
    def __call__(self, edge_lengths: Float[Array, '*batch'], ctx: Context) -> E3IrrepsArray:
        """*batch -> *batch num_basis"""

        embedding = self.basis(edge_lengths, ctx) * self.envelope(edge_lengths, ctx)[..., None]

        return E3IrrepsArray(f'{embedding.shape[-1]}x0e', embedding)


class GaussBasis(RadialBasis):
    """Uses equispaced Gaussian RBFs, as in coGN."""

    r_max: float
    sd: float = 1.0

    def setup(self):
        self.locs = jnp.linspace(0, self.r_max, self.num_basis)

    def __call__(self, d: Float[Array, ' *batch'], ctx: Context) -> Float[Array, '*batch emb']:
        z = d[..., None] - self.locs
        y = jnp.exp(-(z**2) / (2 * self.sd**2))
        return y


class BesselBasis(RadialBasis):
    """Bessel radial basis functions."""

    # https://github.com/ACEsuit/mace/blob/575af0171369e2c19e04b115140b9901f83eb00c/mace/modules/radial.py#L17

    r_max: float

    def setup(self):
        self.bessel_weights = self.r_max * jnp.linspace(1.0, self.num_basis, self.num_basis)

        self.prefactor = jnp.sqrt(2.0 / self.r_max)

    def __call__(self, edge_lengths: Float[Array, '*batch'], ctx: Context) -> E3IrrepsArray:
        numerator = jnp.sinc(self.bessel_weights * edge_lengths[..., None])
        return self.prefactor * (numerator / (edge_lengths[..., None] + 1e-4))


class PolynomialCutoff(Envelope):
    # https://github.com/ACEsuit/mace/blob/575af0171369e2c19e04b115140b9901f83eb00c/mace/modules/radial.py#L112

    p: float = 6

    @nn.compact
    def __call__(self, x: Float[Array, '*batch'], ctx: Context) -> Float[Array, '*batch']:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * jnp.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * jnp.pow(x / self.r_max, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * jnp.pow(x / self.r_max, self.p + 2)
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
    def __call__(self, x: Float[Array, '*batch'], ctx: Context) -> Float[Array, '*batch']:
        r_on = self.cutoff_start * self.r_max
        t = jnp.clip((x - r_on) / (self.r_max - r_on), 0, 1)

        def exp_func(x):
            return jnp.expm1(x * self.c) / jnp.expm1(self.c) * x * x * jnp.sin(jnp.pi * x / 2)

        envelope = 1 - jnp.where(t < 0.5, exp_func(2 * t) / 2, 1 - exp_func(2 - 2 * t) / 2)

        return envelope
