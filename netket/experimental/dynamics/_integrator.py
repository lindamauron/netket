# Copyright 2021 The NetKet Authors - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from netket.utils.numbers import dtype as _dtype

from netket.utils.mpi.primitives import mpi_all_jax
from netket.utils import struct
from ._utils import maybe_jax_jit
from ._solver import AbstractSolver

from ._utils import (
    LimitsType,
    scaled_error,
    propose_time_step,
    set_flag_jax,
    euclidean_norm,
)

from ._state import IntegratorState, IntegratorFlags


@partial(maybe_jax_jit, static_argnames=["f"])
def general_time_step_fixed(
    solver: AbstractSolver,
    f: Callable,
    state: IntegratorState,
    *,
    max_dt: Optional[float],
    **kwargs,
) -> IntegratorState:
    r"""
    Performs one fixed step from current time.
    Args:
        solver: Instance that solves the ODE
            The solver should contain a method `step(f,dt,t,y_t,solver_state)`
        f: A callable ODE function.
            Given a time `t` and a state `y_t`, it should return the partial
            derivatives in the same format as `y_t`. The dunction should also accept
            supplementary arguments, such as code:`stage`.
        state: IntegratorState containing the current state (t,y), the solver_state and stability information.
        max_dt: The maximal value for the time step `dt`.

    Returns:
        Updated state of the integrator.
    """
    if max_dt is None:
        actual_dt = state.dt
    else:
        actual_dt = jnp.minimum(state.dt, max_dt)

    # Perform the solving step
    y_tp1, solver_state = solver.step(
        f, actual_dt, state.t.value, state.y, state.solver_state
    )

    return state.replace(
        step_no=state.step_no + 1,
        step_no_total=state.step_no_total + 1,
        t=state.t + actual_dt,
        y=y_tp1,
        solver_state=solver_state,
        flags=IntegratorFlags.INFO_STEP_ACCEPTED,
    )


@partial(maybe_jax_jit, static_argnames=["f", "norm_fn", "dt_limits"])
def general_time_step_adaptive(
    solver: AbstractSolver,
    f: Callable,
    state: IntegratorState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType,
    **kwargs,
) -> IntegratorState:
    r"""
    Performs one adaptive step from current time.
    Args:
        solver: Instance that solves the ODE
            The solver should contain a method `step_with_error(f,dt,t,y_t,solver_state)`
        f: A callable ODE function.
            Given a time `t` and a state `y_t`, it should return the partial
            derivatives in the same format as `y_t`. The dunction should also accept
            supplementary arguments, such as code:`stage`.
        state: IntegratorState containing the current state (t,y), the solver_state and stability information.
        atol: The tolerance for the absolute error on the solution.
        rtol: The tolerance for the realtive error on the solution.
        norm_fn: The function used for the norm of the error.
            By default, we use euclidean_norm.
        max_dt: The maximal value for the time-step size `dt`.
        dt_limits: The extremal accepted values for the time-step size `dt`.

    Returns:
        Updated state of the integrator
    """
    flags = IntegratorFlags(0)

    if max_dt is None:
        actual_dt = state.dt
    else:
        actual_dt = jnp.minimum(state.dt, max_dt)

    # Perform the solving step
    y_tp1, y_err, solver_state = solver.step_with_error(
        f, actual_dt, state.t.value, state.y, state.solver_state
    )

    scaled_err, norm_y = scaled_error(
        y_tp1,
        y_err,
        atol,
        rtol,
        last_norm_y=state.last_norm,
        norm_fn=norm_fn,
    )

    # Propose the next time step, but limited within [0.1 dt, 5 dt] and potential
    # global limits in dt_limits. Not used when actual_dt < state.dt (i.e., the
    # integrator is doing a smaller step to hit a specific stop).
    next_dt = propose_time_step(
        actual_dt,
        scaled_err,
        solver.error_order,
        limits=(
            (
                jnp.maximum(0.1 * state.dt, dt_limits[0])
                if dt_limits[0]
                else 0.1 * state.dt
            ),
            (
                jnp.minimum(5.0 * state.dt, dt_limits[1])
                if dt_limits[1]
                else 5.0 * state.dt
            ),
        ),
    )

    # check if next dt is NaN
    flags = set_flag_jax(
        ~jnp.isfinite(next_dt), flags, IntegratorFlags.ERROR_INVALID_DT
    )

    # check if we are at lower bound for dt
    if dt_limits[0] is not None:
        is_at_min_dt = jnp.isclose(next_dt, dt_limits[0])
        flags = set_flag_jax(is_at_min_dt, flags, IntegratorFlags.WARN_MIN_DT)
    else:
        is_at_min_dt = False
    if dt_limits[1] is not None:
        is_at_max_dt = jnp.isclose(next_dt, dt_limits[1])
        flags = set_flag_jax(is_at_max_dt, flags, IntegratorFlags.WARN_MAX_DT)

    # accept if error is within tolerances or we are already at the minimal step
    accept_step = jnp.logical_or(scaled_err < 1.0, is_at_min_dt)
    # accept the time step iff it is accepted by all MPI processes
    accept_step, _ = mpi_all_jax(accept_step)

    return jax.lax.cond(
        accept_step,
        # step accepted
        lambda _: state.replace(
            step_no=state.step_no + 1,
            step_no_total=state.step_no_total + 1,
            y=y_tp1,
            t=state.t + actual_dt,
            dt=jax.lax.cond(
                actual_dt == state.dt,
                lambda _: next_dt,
                lambda _: state.dt,
                None,
            ),
            last_norm=norm_y.astype(state.last_norm.dtype),
            last_scaled_error=scaled_err.astype(state.last_scaled_error.dtype),
            solver_state=solver_state,
            flags=flags | IntegratorFlags.INFO_STEP_ACCEPTED,
        ),
        # step rejected, repeat with lower dt
        lambda _: state.replace(
            step_no_total=state.step_no_total + 1,
            dt=next_dt,
            flags=flags,
        ),
        state,
    )


class Integrator(struct.Pytree, mutable=True):
    r"""
    Ordinary-Differential-Equation integrator.
    Given an ODE-function :math:`dy/dt = f(t, y)`, it integrates the derivatives to obtain the solution
    at the next time step :math:`y_{t+1}`.
    """

    f: Callable = struct.field(pytree_node=False)
    """The ODE function."""

    _state: IntegratorState
    """The state of the integrator, containing informations about the solution."""
    _solver: AbstractSolver
    """The ODE solver."""

    use_adaptive: bool = struct.field(pytree_node=False)
    """Boolean indicating whether to use an adaptative scheme."""
    norm: Callable = struct.field(pytree_node=False)
    """The norm used to estimate the error."""

    atol: float = struct.field(pytree_node=False)
    """Absolute tolerance on the error of the state."""
    rtol: float = struct.field(pytree_node=False)
    """Relative tolerance on the error of the state."""
    dt_limits: Optional[LimitsType] = struct.field()
    """Limits of the time-step size."""

    _do_step: Callable = struct.field(pytree_node=False)
    """The function that performs the time step."""

    def __init__(
        self,
        f: Callable,
        solver: AbstractSolver,
        t0: float,
        y0: struct.Pytree,
        use_adaptive: bool,
        norm: Callable,
        *args,
        **kwargs,
    ):
        self.f = f
        self._solver = solver

        self.use_adaptive = use_adaptive
        if use_adaptive:
            self._do_step = general_time_step_adaptive
        else:
            self._do_step = general_time_step_fixed

        if norm is None:
            norm = euclidean_norm
        self.norm = norm

        self.atol = kwargs.get("atol", 0.0)
        self.rtol = kwargs.get("rtol", 1e-7)
        dt_limits = kwargs.get("dt_limits", None)
        if dt_limits is None:
            dt_limits = (None, 10 * solver.initial_dt)
        self.dt_limits = dt_limits

        self._state = self._init_state(t0, y0, dt=solver.initial_dt)

    def _init_state(self, t0: float, y0: struct.Pytree, dt: float) -> IntegratorState:
        r"""
        Initializes the `IntegratorState` structure containing the solver and state,
        given the necessary information.
        Args:
            t0: The initial time of evolution
            y0: The solution at initial time `t0`
            dt: The initial step size

        Returns:
            An `Integrator` instance intialized with the passed arguments
        """
        t_dtype = jnp.result_type(_dtype(t0), _dtype(dt))

        return IntegratorState(
            t=jnp.array(t0, dtype=t_dtype),
            y=y0,
            dt=jnp.array(dt, dtype=t_dtype),
            solver=self._solver,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=IntegratorFlags(0),
        )

    def step(self, max_dt: float = None):
        """
        Performs one full step by min(self.dt, max_dt).

        Returns:
            A boolean indicating whether the step was successful or
            was rejected by the step controller and should be retried.

            Note that the step size can be adjusted by the step controller
            in both cases, so the integrator state will have changed
            even after a rejected step.
        """
        self._state = self._do_step(
            solver=self._solver,
            f=self.f,
            state=self._state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )
        return self._state.accepted

    @property
    def t(self):
        """The actual time."""
        return self._state.t.value

    @property
    def y(self):
        """The actual state."""
        return self._state.y

    @property
    def dt(self):
        """The actual time-step size."""
        return self._state.dt

    @property
    def solver(self):
        """The ODE solver."""
        return self._solver

    def _get_integrator_flags(self, intersect=IntegratorFlags.NONE) -> IntegratorFlags:
        r"""Returns the currently set flags of the integrator, intersected with `intersect`."""
        # _state.flags is turned into an int-valued DeviceArray by JAX,
        # so we convert it back.
        return IntegratorFlags(int(self._state.flags) & intersect)

    @property
    def errors(self) -> IntegratorFlags:
        r"""Returns the currently set error flags of the integrator."""
        return self._get_integrator_flags(IntegratorFlags.ERROR_FLAGS)

    @property
    def warnings(self) -> IntegratorFlags:
        r"""Returns the currently set warning flags of the integrator."""
        return self._get_integrator_flags(IntegratorFlags.WARNINGS_FLAGS)

    def __repr__(self):
        return "{}(solver={}, state={}, adaptive={}{})".format(
            "Integrator",
            self.solver,
            self._state,
            self.use_adaptive,
            (
                f", norm={self.norm}, atol={self.atol}, rtol={self.rtol}, dt_limits={self.dt_limits}"
                if self.use_adaptive
                else ""
            ),
        )
