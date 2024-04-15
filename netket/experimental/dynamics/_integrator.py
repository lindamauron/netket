from functools import partial, wraps
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import netket as nk
from netket import config
from netket.utils.mpi.primitives import mpi_all_jax
from netket.utils.struct import dataclass, field
from netket.utils.types import Array, PyTree
from netket.utils.numbers import dtype as _dtype

from ._structures import maybe_jax_jit,LimitsType,scaled_error,propose_time_step,set_flag_jax,euclidean_norm, expand_dim
from ._tableau import Tableau
from ._state import IntegratorState, SolverFlags


@partial(maybe_jax_jit, static_argnames=["f", "norm_fn", "dt_limits"])
def general_time_step_adaptive(
    tableau: Tableau,
    f: Callable,
    state: IntegratorState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType, ### change to limtsDType
):
    # return 0
    flags = SolverFlags(0)

    if max_dt is None:
        actual_dt = state.dt
    else:
        actual_dt = jnp.minimum(state.dt, max_dt)

    y_tp1, y_err = tableau.step_with_error(f, state.t.value, actual_dt, state.y, state)

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
        tableau.error_order,
        limits=(
            jnp.maximum(0.1 * state.dt, dt_limits[0])
            if dt_limits[0]
            else 0.1 * state.dt,
            jnp.minimum(5.0 * state.dt, dt_limits[1])
            if dt_limits[1]
            else 5.0 * state.dt,
        ),
    )

    # check if next dt is NaN
    flags = set_flag_jax(~jnp.isfinite(next_dt), flags, SolverFlags.ERROR_INVALID_DT)

    # check if we are at lower bound for dt
    if dt_limits[0] is not None:
        is_at_min_dt = jnp.isclose(next_dt, dt_limits[0])
        flags = set_flag_jax(is_at_min_dt, flags, SolverFlags.WARN_MIN_DT)
    else:
        is_at_min_dt = False
    if dt_limits[1] is not None:
        is_at_max_dt = jnp.isclose(next_dt, dt_limits[1])
        flags = set_flag_jax(is_at_max_dt, flags, SolverFlags.WARN_MAX_DT)

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
            last_norm=norm_y,
            last_scaled_error=scaled_err,
            flags=flags | SolverFlags.INFO_STEP_ACCEPTED,
        ),
        # step rejected, repeat with lower dt
        lambda _: state.replace(
            step_no_total=state.step_no_total + 1,
            dt=next_dt,
            flags=flags,
        ),
        None,
    ), accept_step



@partial(maybe_jax_jit, static_argnames=["f"])
def general_time_step_fixed(
    tableau: Tableau,
    f: Callable,
    state: IntegratorState,
    max_dt: Optional[float],
):
    if max_dt is None:
        actual_dt = state.dt
    else:
        actual_dt = jnp.minimum(state.dt, max_dt)

    y_tp1 = tableau.step(f, state.t.value, actual_dt, state.y, state)

    return state.replace(
        step_no=state.step_no + 1,
        step_no_total=state.step_no_total + 1,
        t=state.t + actual_dt,
        y=y_tp1,
        flags=SolverFlags.INFO_STEP_ACCEPTED
    )


@dataclass(_frozen=False)
class Integrator:
    tableau: Tableau

    f: Callable = field(repr=False)
    t0: float
    y0: Array = field(repr=False)

    initial_dt: float

    use_adaptive: bool
    norm: Callable

    atol: float = 0.0
    rtol: float = 1e-7
    dt_limits: Optional[LimitsType] = None
    
    def __post_init__(self):

        if self.use_adaptive:
            # raise NotImplementedError("Adaptive time steps are not implemented yet.")
            self._do_step = self._do_step_adaptive
        else:
            self._do_step = self._do_step_fixed

        if self.norm is None:
            self.norm = euclidean_norm

        if self.dt_limits is None:
            self.dt_limits = (None, 10 * self.initial_dt)

        t_dtype = jnp.result_type(_dtype(self.t0), _dtype(self.initial_dt))
        setattr(self, "t0", jnp.array(self.t0, dtype=t_dtype))
        setattr(self, "initial_dt", jnp.array(self.initial_dt, dtype=t_dtype))
        
        self._state = IntegratorState(
            step_no=0,
            step_no_total=0,
            t=nk.utils.KahanSum(self.t0),
            y=self.y0,
            dt=self.initial_dt,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=SolverFlags(0),
        )

    def step(self, max_dt=None):
        """
        Perform one full Adams-Bashforht-Moulton step by min(self.dt, max_dt).

        Returns:
            A boolean indicating whether the step was successful or
            was rejected by the step controller and should be retried.

            Note that the step size can be adjusted by the step controller
            in both cases, so the integrator state will have changed
            even after a rejected step.
        """
        self._state = self._do_step(self._state, max_dt)
        return self._state.accepted

    def _do_step_fixed(self, state, max_dt=None):
        return general_time_step_fixed(
            tableau=self.tableau,
            f=self.f,
            state=state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, state, max_dt=None):
        return general_time_step_adaptive(
            tableau=self.tableau,
            f=self.f,
            state=state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )[0]

    @property
    def t(self):
        return self._state.t.value

    @property
    def y(self):
        return self._state.y

    @property
    def dt(self):
        return self._state.dt

    def _get_solver_flags(self, intersect=SolverFlags.NONE) -> SolverFlags:
        """Returns the currently set flags of the solver, intersected with `intersect`."""
        # _state.flags is turned into an int-valued DeviceArray by JAX,
        # so we convert it back.
        return SolverFlags(int(self._state.flags) & intersect)

    @property
    def errors(self) -> SolverFlags:
        """Returns the currently set error flags of the solver."""
        return self._get_solver_flags(SolverFlags.ERROR_FLAGS)

    @property
    def warnings(self) -> SolverFlags:
        """Returns the currently set warning flags of the solver."""
        return self._get_solver_flags(SolverFlags.WARNINGS_FLAGS)

class IntegratorConfig:
    def __init__(self, dt, tableau, *, adaptive=False, **kwargs):

        self.dt = dt
        self.adaptive = adaptive
        self.kwargs = kwargs
        self.tableau = tableau

    def __call__(self, f, t0, y0, *, norm=None):
        return Integrator(
            self.tableau,
            f,
            t0,
            y0,
            initial_dt=self.dt,
            use_adaptive=self.adaptive,
            norm=norm,
            **self.kwargs,
        )

    def __repr__(self):
        return "{}(tableau={}, dt={}, adaptive={}{})".format(
            "IntegratorConfig",
            self.tableau,
            self.dt,
            self.adaptive,
            f", **kwargs={self.kwargs}" if self.kwargs else "",
        )