"""Generic layers useful for dealing with irreps of E(3)."""

from typing import Callable
import e3nn_jax as e3nn

from cdv.layers import Context, E3Irreps, E3IrrepsArray
from flax import linen as nn


def Linear(*args, **kwargs):
    # return nn.WeightNorm(e3nn.flax.Linear(*args, **kwargs))
    return e3nn.flax.Linear(*args, **kwargs)


class IrrepsModule(nn.Module):
    """Module that outputs irreps."""

    irreps_out: E3Irreps

    @property
    def ir_out(self) -> E3Irreps:
        return E3Irreps(self.irreps_out)


class IrrepsAdapter(IrrepsModule):
    """Generic module that can convert between irreps."""


class LinearReadoutBlock(IrrepsAdapter):
    """Simple linear layer."""

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        return e3nn.flax.Linear(self.ir_out)(x)


class NonlinearReadoutBlock(IrrepsAdapter):
    """Nonlinear readout block: linear layer, gated nonlinearity, then a second linear layer."""

    hidden_irreps: E3Irreps
    activation: Callable = nn.silu
    gate: Callable = nn.sigmoid

    @nn.compact
    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        hidden_irreps = E3Irreps(self.hidden_irreps)
        num_vectors = hidden_irreps.filter(
            drop=['0e', '0o']
        ).num_irreps  # Multiplicity of (l > 0) irreps
        # print(x.irreps)
        x = Linear(
            (hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify(),
        )(x)
        # print((hidden_irreps + E3Irreps(f'{num_vectors}x0e')).simplify())
        # print(x.irreps)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return Linear(self.ir_out)(x)

    # [n_nodes, output_irreps]
