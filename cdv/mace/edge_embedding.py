"""Edge embedding layers. The only trainable part is the radial embedding."""

from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Float, Array

from cdv.layers import Context, E3IrrepsArray, Identity


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
    """Uses spherical Bessel functions with a cutoff, as in DimeNet++."""

    r_max: float
    freq_trainable: bool = True

    def setup(self):
        def freq_init(rng):
            return jnp.arange(self.num_basis, dtype=jnp.float32) + 1

        if self.freq_trainable:
            self.freq = self.param('freq', freq_init)
        else:
            self.freq = freq_init(None)

    def __call__(self, x, ctx: Context):
        dist = x[..., None] / self.r_max

        # e(d) = sqrt(2/c) * sin(fπd/c)/d
        # we use sinc so it's defined at 0
        # jnp.sinc is sin(πx)/(πx)
        # e(d) = sqrt(2/c) * sin(πfd/c)/(fd/c) * f/c
        # e(d) = sqrt(2/c) * sinc(πfd/c)) * πf/c

        e_d = (
            jnp.sqrt(2 / self.r_max)
            * jnp.sinc(self.freq * dist)
            * (jnp.pi * self.freq / self.r_max)
        )

        # debug_stat(e_d=e_d, env=env, dist=dist)
        return e_d


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


if __name__ == '__main__':
    import jax
    import jax.random as jr
    from cdv.utils import debug_stat, debug_structure

    rng = jr.key(123)
    radii = jr.truncated_normal(rng, lower=-3.4, upper=5, shape=(32, 16), dtype=jnp.bfloat16) + 4
    data = {}
    kwargs = dict(num_basis=10, r_max=7)
    cutoff = ExpCutoff(r_max=kwargs['r_max'], c=0.1, cutoff_start=0.8)

    mods = [cutoff]
    for basis in (GaussBasis(**kwargs), BesselBasis(**kwargs)):
        mods.append(RadialEmbeddingBlock(basis=basis, envelope=cutoff))

    for mod in mods:
        name = mod.basis.__class__.__name__ if hasattr(mod, 'basis') else 'raw'

        def embed(radii):
            out, params = mod.init_with_output(rng, radii, ctx=Context(training=True))
            if hasattr(out, 'array'):
                return out.array
            else:
                return out

        data[name] = embed(radii)
        data[name + '_grad'] = jax.grad(lambda x: jnp.sum(embed(x)))(radii)

    # debug_structure(**data)
    debug_stat(**data)
