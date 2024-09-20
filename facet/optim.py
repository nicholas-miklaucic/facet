"""
Optimizer functionality.
"""

from typing import NamedTuple
from optax._src import base
import jax


class EmaParamsState(NamedTuple):
    """Holds an exponential moving average of past updates."""

    ema: base.Params
    steps_since_update: int


def update_ema(old, target, decay):
    return jax.tree.map(lambda x, y: decay * x + (1 - decay) * y, old, target)


def ema_params(decay: float, update_every: int = 1) -> base.GradientTransformation:
    """
    Store an EMA of past *parameters*, unlike the standard Optax EMA which computes a moving average
    of past *updates*.

    Computes the EMA update every N steps. Adjusts the decay as if it were being applied every step,
    to make that parameter consistent across different update intervals.

    Returns updates unchanged.
    """

    equiv_decay = decay**update_every

    def init_fn(params):
        return EmaParamsState(ema=params, steps_since_update=0)

    def do_ema_update(updates, state, params):
        # update
        new_ema = update_ema(state.ema, params, equiv_decay)
        new_state = EmaParamsState(ema=new_ema, steps_since_update=0)
        return updates, new_state

    def no_ema_update(updates, state, params):
        # update
        new_state = EmaParamsState(ema=state.ema, steps_since_update=state.steps_since_update + 1)
        return updates, new_state

    def update_fn(updates, state, params):
        return jax.lax.cond(
            state.steps_since_update == update_every - 1,
            do_ema_update,
            no_ema_update,
            updates,
            state,
            params,
        )

    return base.GradientTransformation(init_fn, update_fn)
