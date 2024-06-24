"""Module to store different approaches to embed the position."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import pyrallis
from einops import rearrange
from scipy.special import lpmv

from avid.utils import debug_stat, debug_structure


def legendre_first_kind(xyz, m: np.int64, n: np.int64):
    return np.array(lpmv(m, n, np.sin(xyz * 2 * np.pi)))


all_m_n = np.array(list(filter(lambda mn: mn[0] <= mn[1], product(range(20), repeat=2))))
all_m_n = all_m_n[np.argsort(all_m_n.sum(axis=1), kind='stable'), :]
# print(all_m_n)


def legendre_embed(xyz, single_dim_out: int):
    """Embeds the positions using Legendre polynomials, using sin for periodicity."""
    standard = lambda x: x / (max(abs(x)) + 1e-10)
    embeds = np.array(
        [
            [standard(legendre_first_kind(xyz[:, d], m, n)) for m, n in all_m_n[:single_dim_out]]
            for d in range(3)
        ]
    )

    return rearrange(embeds, 'd dim_out N -> N (dim_out d)')


def legendre_grid_embeds(ng: int, dim_embed: int):
    grid_vals = np.linspace(0, 1, ng + 1)[:-1]
    xx, yy, zz = np.meshgrid(grid_vals, grid_vals, grid_vals)
    xyz = rearrange(np.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
    return jnp.array(
        rearrange(
            legendre_embed(xyz, dim_embed),
            '(n1 n2 n3) d -> n1 n2 n3 d',
            n1=ng,
            n2=ng,
            n3=ng,
        )
    )


if __name__ == '__main__':
    from avid.config import MainConfig

    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()
    from einops import rearrange

    N_GRID = 24
    grid_vals = jnp.linspace(0, 1, N_GRID + 1)[:-1]
    xx, yy, zz = jnp.meshgrid(grid_vals, grid_vals, grid_vals)
    xyz = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
    embed = legendre_embed(xyz, 4)
    debug_stat(emb=embed)
    debug_structure(emb=embed)
