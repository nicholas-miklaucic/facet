"""Equivariant activations for irreps."""

from typing import Callable, Literal, Optional
import e3nn_jax as e3nn
import jax.numpy as jnp
from flax import linen as nn
import rich.table

from facet.layers import Context, E3Irreps, E3IrrepsArray


import yaml
from yaml import Node, SafeDumper


def represent_spherical_signal(d: SafeDumper, s: e3nn.SphericalSignal) -> Node:
    return d.represent_dict({'s2_grid': s.grid_values})


yaml.SafeDumper.add_representer(e3nn.SphericalSignal, represent_spherical_signal)


class S2Activation(nn.Module):
    """
    Pointwise spherical activation.

    Represents its inputs as values on a spherical grid, applies a pointwise nonlinearity, and then
    converts back.
    """

    activation: Callable

    # Number of grid lines on the sphere in the theta (latitude) axis.
    res_beta: int = 16

    # Number of grid lines on the sphere in the phi (longitude) axis.
    res_alpha: int = 15

    # Normalization of the basis.
    normalization: Literal['norm', 'component', 'integral'] = 'integral'

    # Quadrature method. Soft is uniform beta spacing, gauss-legendre is exact for some polynomials.
    quadrature: Literal['soft', 'gausslegendre'] = 'soft'

    # Whether to use FFT.
    fft: bool = True

    def setup(self):
        pass

    def input_signal(self, x: E3IrrepsArray, ctx: Context) -> e3nn.SphericalSignal:
        # lpmn_values does not support bfloat16 right now
        # and there's no way to specify it in the current API
        if x.dtype == jnp.bfloat16:
            x = x.astype(jnp.float32)

        return e3nn.to_s2grid(
            x,
            res_beta=self.res_beta,
            res_alpha=self.res_alpha,
            quadrature=self.quadrature,
            normalization=self.normalization,
            fft=self.fft,
            p_val=1,
            p_arg=1,
        )

    def output_irreps(self, signal: e3nn.SphericalSignal, x_templ: E3IrrepsArray):
        return e3nn.from_s2grid(
            signal,
            irreps=e3nn.Irreps([(1, ir) for ir in x_templ.irreps]),
            normalization=self.normalization,
            lmax_in=x_templ.irreps.lmax,
            fft=self.fft,
            use_s2fft=False,
        ).astype(x_templ.dtype)

    def __call__(self, x: E3IrrepsArray, ctx: Context) -> E3IrrepsArray:
        """Applies the nonlinearity, preserving input shape."""
        signal = self.input_signal(x, ctx)
        signal = signal.apply(self.activation)
        return self.output_irreps(signal, x)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import jax.random as jr
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import rich

    import pandas as pd
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938

    def df_to_table(
        pandas_dataframe: pd.DataFrame,
        rich_table: Table,
        show_index: bool = True,
        index_name: Optional[str] = None,
    ) -> Table:
        if show_index:
            index_name = str(index_name) if index_name else ''
            rich_table.add_column(index_name)

        for column in pandas_dataframe.columns:
            rich_table.add_column(str(column), justify='right')

        for index, value_list in enumerate(pandas_dataframe.values.tolist()):
            row = [str(index)] if show_index else []
            row += [str(x) for x in value_list]
            rich_table.add_row(*row)

        return rich_table

    rng = jr.key(29205)

    x = e3nn.normal('1e', rng, (16,), dtype=jnp.bfloat16) * 3
    print(x)

    res_beta = 64
    res_alpha = 63

    prev_out = x * 0

    df = []

    act_fn = jax.nn.silu

    fn = jax.jit(act_fn)
    flops = fn.lower(x.array).cost_analysis()['flops'] / x.shape[0]
    print('Base FLOPS: ', flops)

    while res_beta >= 4:
        act = S2Activation(activation=act_fn, res_beta=res_beta, res_alpha=res_alpha, fft=False)
        params = act.init(rng, x=x, ctx=Context(training=True))

        fn = jax.jit(lambda x: act.apply(params, rngs=rng, x=x, ctx=Context(training=True)))
        flops = fn.lower(x).cost_analysis()['flops'] / x.shape[0]
        out = fn(x)

        fn_r, r_fn = e3nn.utils.equivariance_test(fn, rng, x)

        equiv_diff = jnp.max(jnp.abs((fn_r - r_fn).array)).item()

        diff = jnp.max(jnp.abs((out - prev_out).array)).item()

        df.append(
            {
                'res_α': res_alpha,
                'res_β': res_beta,
                'flops': flops,
                'equiv_diff': equiv_diff,
                'diff': diff,
            }
        )

        res_beta //= 2
        res_alpha = (res_alpha - 1) // 2
        prev_out = out

    df = pd.DataFrame(df)
    table = Table(highlight=True, show_edge=False)
    rich.print(df_to_table(df, table))
