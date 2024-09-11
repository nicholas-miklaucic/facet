# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Schedule-Free wrapper for faster training & removes the need for lr decay."""

import functools
from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import transform
from optax.schedules import _schedule

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import wrappers


def init_empty_state(params: base.Params) -> base.EmptyState:
    """Init function for a :class:`GradientTransformation` with empty state."""
    del params
    return base.EmptyState()


def add_decayed_weights(
    weight_decay: Union[float, jax.Array] = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """Add parameter scaled by `weight_decay`.

    Args:
      weight_decay: A scalar weight decay rate.
      mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
      A `GradientTransformation` object.
    """

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(base.NO_PARAMS_MSG)
        updates = jtu.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return wrappers.masked(base.GradientTransformation(init_empty_state, update_fn), mask)
    return base.GradientTransformation(init_empty_state, update_fn)


class AddNoiseState(NamedTuple):
    """State for adding gradient noise. Contains a count for annealing."""

    count: chex.Array
    rng_key: chex.PRNGKey


def add_noise(eta: float, gamma: float, seed: int) -> base.GradientTransformation:
    """Add gradient noise.

    References:
      [Neelakantan et al, 2014](https://arxiv.org/abs/1511.06807)

    Args:
      eta: Base variance of the gaussian noise added to the gradient.
      gamma: Decay exponent for annealing of the variance.
      seed: Seed for random number generation.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return AddNoiseState(count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

    def update_fn(updates, state, params=None):  # pylint: disable=missing-docstring
        del params
        count_inc = numerics.safe_int32_increment(state.count)
        standard_deviation = jnp.sqrt(eta / count_inc**gamma)

        rng_key, sample_key = jax.random.split(state.rng_key)
        noise = otu.tree_random_like(sample_key, target_tree=updates, sampler=jax.random.normal)
        updates = otu.tree_add_scalar_mul(tree_x=updates, scalar=standard_deviation, tree_y=noise)
        return updates, AddNoiseState(count=count_inc, rng_key=rng_key)

    return base.GradientTransformation(init_fn, update_fn)


class ScheduleFreeState(NamedTuple):
    """State for schedule_free."""

    b1: chex.Array
    weight_sum: chex.Array
    step_count: chex.Array
    max_lr: chex.Array
    base_optimizer_state: base.OptState
    z: base.Params


def schedule_free_eval_params(state: ScheduleFreeState, params: base.Params):
    """Params for evaluation of :func:`optax.contrib.schedule_free`."""
    return jax.tree_util.tree_map(
        lambda yi, zi: (yi - (1.0 - state.b1) * zi) / state.b1, params, state.z
    )


def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    weight_lr_power: float = 2.0,
    state_dtype=jnp.float32,
) -> base.GradientTransformationExtraArgs:
    r"""Turn base_optimizer schedule_free.

  Accumulates updates returned by the base_optimizer w/o Momentum and
  replaces the momentum of an underlying optimizer with a combination of
  interpolation and averaging. In the case of gradient descent the update is

  .. math::

    \begin{align*}
      y_{t} & = (1-\beta_1)z_{t} + \beta_1 x_{t},\\
      z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
      x_{t+1} & =\left(1-\frac{1}{t}\right)x_{t}+\frac{1}{t}z_{t+1},
    \end{align*}

  Here :math:`x` is the sequence that evaluations of test/val loss should occur
  at,  which differs from the primary iterates :math:`z` and the gradient
  evaluation locations :math:`y`. The updates to :math:`z` correspond to the
  underlying optimizer, in this case a simple gradient step. Note that,
  :math:`\beta_1` corresponds to `b1` in the code.

  As the name suggests, Schedule-Free learning does not require a decreasing
  learning rate schedule, yet typically out-performs, or at worst matches, SOTA
  schedules such as cosine-decay and linear decay. Only two sequences need to be
  stored at a time (the third can be computed from the other two on the fly) so
  this method has the same memory requirements as the base optimizer (parameter
  buffer + momentum).

  In practice, authors recommend tuning :math:`\beta_1`, `warmup_steps` and
  `peak_lr` for each problem seperately. Default for :math:`\beta_1` is 0.9 but
  `0.95` and `0.98` may also work well. Schedule-Free can be wrapped on top of
  any optax optimizer. At test time, the parameters should be evaluated using
  :func:`optax.contrib.schedule_free_eval_params` as presented below.

  For example, change this::

    learning_rate_fn = optax.warmup_cosine_decay_schedule(peak_value=tuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=b1)

  To::

    learning_rate_fn = optax.warmup_constant_schedule(peak_value=retuned_lr)
    optimizer = optax.adam(learning_rate_fn, b1=0.)
    optimizer = optax.contrib.schedule_free(optimizer, learning_rate_fn, b1=b1)
    ..
    params_for_eval = optax.contrib.schedule_free_eval_params(state, params)

  Especially note that is important to switch off Momentum of the base
  optimizer. As of Apr, 2024, schedule_free is tested with SGD and Adam.

  References:
    Defazio et al, `The Road Less Scheduled
    <https://arxiv.org/abs/2405.15682>`_, 2024

    Defazio et al, `Schedule-Free Learning - A New Way to Train
    <https://github.com/facebookresearch/schedule_free/tree/main>`_, 2024

  Args:
    base_optimizer: Base optimizer to compute updates from.
    learning_rate: learning_rate schedule w/o decay but with warmup.
    b1: beta_1 parameter in the y update.
    weight_lr_power: we downweight the weight of averaging using this. This is
      especially helpful in early iterations during warmup.
    state_dtype: dtype for z sequence in the schedule free method.

  Returns:
    A `GradientTransformationExtraArgs` with init and update functions.
  """
    base_optimizer = base.with_extra_args_support(base_optimizer)

    def init_fn(params: base.Params) -> ScheduleFreeState:
        if b1 == 0:
            raise ValueError('The current implementation of schedule_free requires b1 > 0.')
        z = jax.tree_util.tree_map(lambda t: t.astype(state_dtype), params)
        return ScheduleFreeState(
            b1=jnp.array([b1], dtype=jnp.float32),
            weight_sum=jnp.zeros([], dtype=jnp.float32),
            step_count=jnp.ones([], dtype=jnp.int32),
            max_lr=jnp.zeros([], dtype=jnp.float32),
            base_optimizer_state=base_optimizer.init(params),
            z=z,
        )

    def update_fn(
        grads: base.Updates,
        state: ScheduleFreeState,
        params: Optional[base.Params] = None,
        **extra_args,
    ):
        lr = learning_rate
        if callable(learning_rate):
            lr = learning_rate(state.step_count)
        max_lr = jnp.maximum(state.max_lr, lr)

        next_step_count = state.step_count + 1

        weight = max_lr**weight_lr_power
        next_total_weight = state.weight_sum + weight
        # We add this to avoid NaNs in the case of a small learning rate.
        ck = jnp.where(
            jnp.logical_or(jnp.isnan(weight), jnp.isnan(next_total_weight)),
            jnp.full(weight.shape, jnp.nan),
            jnp.nan_to_num(weight / next_total_weight, nan=0.0, posinf=jnp.inf),
        )

        base_updates, next_base_optimizer_state = base_optimizer.update(
            grads,
            state.base_optimizer_state,
            params,
            **extra_args,
        )
        z = jax.tree_util.tree_map(
            lambda pi, ui: jnp.asarray(pi + ui).astype(jnp.asarray(pi).dtype),
            state.z,
            base_updates,
        )

        # Important: recompute x to both save memory and maintain accurate x seq
        # especially if y is modified by another transform wrapped on top.
        prev_x = jax.tree_util.tree_map(lambda yi, zi: (yi - (1.0 - b1) * zi) / b1, params, state.z)

        x = jax.tree_util.tree_map(
            lambda xi, zi: (1.0 - ck) * xi + ck * zi,
            prev_x,
            z,
        )
        new_params = jax.tree_util.tree_map(
            lambda xi, zi: b1 * xi + (1.0 - b1) * zi,
            x,
            z,
        )
        updates = jax.tree_util.tree_map(lambda npi, pi: npi - pi, new_params, params)

        next_state = ScheduleFreeState(
            b1=jnp.array([b1], dtype=jnp.float32),
            weight_sum=next_total_weight,
            step_count=next_step_count,
            max_lr=max_lr,
            base_optimizer_state=next_base_optimizer_state,
            z=z,
        )

        return updates, next_state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def schedule_free_sgd(
    learning_rate: float = 1.0,
    *,
    warmup_steps: int = 0,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    state_dtype=jnp.float32,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free wrapper for SGD.

    Shortcut example for using schedule_free with SGD, which is a common use case.
    Note that this is just an example, and other use cases are possible, e.g.
    using a weight decay mask. Note also that the EMA parameter of the
    schedule free method (b1) must be strictly positive.

    Args:
      learning_rate: SGD learning rate.
      warmup_steps: positive integer, the length of the linear warmup.
      b1: beta_1 parameter in the y update.
      weight_decay: Strength of the weight decay regularization. Note that this
        weight decay is multiplied with the learning rate. This is consistent
        with other frameworks such as PyTorch, but different from
        (Loshchilov et al, 2019) where the weight decay is only multiplied with
        the "schedule multiplier", but not the base learning rate.
      weight_lr_power: we downweight the weight of averaging using this. This is
        especially helpful in early iterations during warmup.
      state_dtype: dtype for z sequence in the schedule free method.

    Returns:
      A `GradientTransformationExtraArgs` with init and update functions.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.contrib.schedule_free_sgd()
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  eval_params = optax.contrib.schedule_free_eval_params(
      ...      opt_state, params)
      ...  print('Objective function: {:.2E}'.format(f(eval_params)))
      Objective function: 1.40E+01
      Objective function: 1.75E-14
      Objective function: 9.96E-01
      Objective function: 8.06E-01
      Objective function: 2.41E-01
    """
    if warmup_steps > 0:
        learning_rate = _schedule.warmup_constant_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )
    optimizer = alias.sgd(learning_rate)
    if weight_decay > 0:
        optimizer = combine.chain(add_decayed_weights(weight_decay), optimizer)
    return schedule_free(
        optimizer,
        learning_rate=learning_rate,
        b1=b1,
        weight_lr_power=weight_lr_power,
        state_dtype=state_dtype,
    )


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    nu: base.Updates


from optax._src.transform import update_moment_per_elem_norm


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    # Kept for backward compatibility, even though ScaleByRmsWithCountState
    # encompasses this state.
    nu: base.Updates


class ScaleByRmsWithCountState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    nu: base.Updates


def tree_full_like(
    tree: Any,
    fill_value: jax.typing.ArrayLike,
    dtype: Optional[jax.typing.DTypeLike] = None,
) -> Any:
    """Creates an identical tree where all tensors are filled with ``fill_value``.

    Args:
      tree: pytree.
      fill_value: the fill value for all tensors in the tree.
      dtype: optional dtype to use for the tensors in the tree.

    Returns:
      an tree with the same structure as ``tree``.
    """
    return jtu.tree_map(lambda x: jnp.full_like(x, fill_value, dtype=dtype), tree)


def tree_update_moment_per_elem_norm(updates, moments, decay, order):
    """Compute the EMA of the `order`-th moment of the element-wise norm."""

    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g**order
        else:
            half_order = order / 2
            # JAX generates different HLO for int and float `order`
            if half_order.is_integer():
                half_order = int(half_order)
            return numerics.abs_sq(g) ** half_order

    return jtu.tree_map(
        lambda g, t: ((1 - decay) * orderth_norm(g) + decay * t if g is not None else None),
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


@functools.partial(jax.jit, inline=True)
def tree_bias_correction(moment, decay, count):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    bias_correction_ = 1 - decay**count

    # Perform division in the original precision.
    return jax.tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)


def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> base.GradientTransformation:
    r"""Rescale updates by the root of the exp. moving avg of the square.

    .. warning::
      Default behavior of optax's RMSprop (``eps_in_sqrt=True``) differs from
      Pytorch's implementation and could impact performance.
      If ``eps_in_sqrt=True``, in the denominator, optax uses
      :math:`\sqrt{v + \epsilon}` in the denominator whereas PyTorch uses
      :math:`\sqrt{v} + \epsilon`.
      Using ``eps_in_sqrt=False`` in optax will match PyTorch's behavior.
      See
      https://github.com/google-deepmind/optax/issues/532 for more detail.

    .. note::
      Using `scale_by_rms(decay=b2, eps_in_sqrt=False, bias_correction=True)`
      will match the behavior of `scale_by_adam(b1=0, b2=b2)`, while sparing the
      memory cost of storing the first moment.

    References:
      Hinton, `Overview of mini-batch gradient descent`
      <www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_, 2012

    Args:
      decay: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      initial_scale: Initial value for second moment.
      eps_in_sqrt: Whether to add ``eps`` in the square root of the
        denominator or outside the square root.
      bias_correction: Whether to apply bias correction to the exponentially
        weighted average of squared grads.

    Returns:
      A `GradientTransformation` object.
    """

    def init_fn(params):
        nu = tree_full_like(params, initial_scale)  # second moment
        if bias_correction:
            return ScaleByRmsWithCountState(count=jnp.zeros([], jnp.int32), nu=nu)
        else:
            return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
        if bias_correction:
            count_inc = numerics.safe_int32_increment(state.count)
            nu_hat = tree_bias_correction(nu, decay, count_inc)
        else:
            count_inc = jnp.asarray(0)
            nu_hat = nu
        if eps_in_sqrt:
            scaling = jtu.tree_map(lambda n: jax.lax.rsqrt(n + eps), nu_hat)
        else:
            scaling = jtu.tree_map(lambda n: 1 / (jnp.sqrt(n) + eps), nu_hat)
        updates = jtu.tree_map(lambda s, g: s * g, scaling, updates)
        if bias_correction:
            new_state = ScaleByRmsWithCountState(count=count_inc, nu=nu)
        else:
            new_state = ScaleByRmsState(nu=nu)
        return updates, new_state

    return base.GradientTransformation(init_fn, update_fn)


def schedule_free_adamw(
    learning_rate: float = 0.0025,
    *,
    warmup_steps: int = 0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    state_dtype=jnp.float32,
) -> base.GradientTransformationExtraArgs:
    """Schedule-Free wrapper for AdamW.

    Shortcut example for using schedule_free with AdamW, which is a common use
    case. Note that this is just an example, and other usecases are possible, e.g.
    using a weight decay mask, nesterov, etc. Note also that the EMA parameter of
    the schedule free method (b1) must be strictly positive.

    Args:
      learning_rate: AdamW learning rate.
      warmup_steps: positive integer, the length of the linear warmup.
      b1: beta_1 parameter in the y update.
      b2: Exponential decay rate to track the second moment of past gradients.
      eps: A small constant applied to denominator outside of the square root
        (as in the Adam paper) to avoid dividing by zero when rescaling.
      weight_decay: Strength of the weight decay regularization.
      weight_lr_power: we downweight the weight of averaging using this. This is
        especially helpful in early iterations during warmup.
      state_dtype: dtype for z sequence in the schedule free method.

    Returns:
      A `GradientTransformationExtraArgs` with init and update functions.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.contrib.schedule_free_adamw(1.0)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  eval_params = optax.contrib.schedule_free_eval_params(
      ...      opt_state, params)
      ...  print('Objective function: {:.2E}'.format(f(eval_params)))
      Objective function: 5.00E+00
      Objective function: 3.05E+00
      Objective function: 1.73E+00
      Objective function: 8.94E-01
      Objective function: 4.13E-01
    """
    if warmup_steps > 0:
        learning_rate = _schedule.warmup_constant_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )
    # The following is the same as adamw, but with the momentum term removed.
    optimizer = combine.chain(
        scale_by_rms(decay=b2, eps=eps, bias_correction=True, eps_in_sqrt=False),
        add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(learning_rate),
    )
    return schedule_free(
        optimizer,
        learning_rate=learning_rate,
        b1=b1,
        weight_lr_power=weight_lr_power,
        state_dtype=state_dtype,
    )
