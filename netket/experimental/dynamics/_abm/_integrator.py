from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import netket as nk
from netket.utils.struct import dataclass

from .._structures import LimitsType
from .._integrator import (
    Integrator,
    general_time_step_adaptive,
    general_time_step_fixed,
)
from .._state import SolverFlags

from ._tableau import TableauABM, expand_dim
from ._state import ABMState


# @partial(maybe_jax_jit, static_argnames=["f", "norm_fn", "dt_limits"])
def abm_time_step_adaptive(
    tableau: TableauABM,
    f: Callable,
    state: ABMState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType,  ### change to limtsDType
):
    r"""
    Performs one adaptive ABM step from current time.
    Args:
        tableau: Integration tableau containing the coefficeints for integration.
            The tableau should contain method step_with_error(f, t, dt, y_t, state).
        f: A callable ODE function.
            Given a time `t` and a state `y_t`, it should return the partial
            derivatives in the same format as `y_t`. The dunction should also accept
            supplementary arguments, such as code:`stage`.
        state: Intagrator state containing the current state (t,y) and stablity information.
        atol: The tolerance for the absolute error on the state.
        rtol: The tolerance for the realtive error on the state.
        norm_fn: The function used for the norm of the error.
            By default, we use euclidean_norm.
        max_dt: The maximal value for the time step `dt`.
        dt_limits: The extremal accepted values for the time-step `dt`.

    Returns:
        Updated state of the integrator.
    """
    state, accept_step = general_time_step_adaptive(
        tableau, f, state, atol, rtol, norm_fn, max_dt, dt_limits
    )

    last_f = f(state.t.value, state.y, stage=tableau.order + 2)
    return (
        jax.lax.cond(
            accept_step,
            lambda _: state.replace(
                F_history=tree_map(
                    lambda H, x: jnp.roll(H, 1, axis=0).at[0].set(x),
                    state.F_history,
                    last_f,
                ),
            ),
            lambda _: state,
            None,
        ),
        accept_step,
    )


# @partial(maybe_jax_jit, static_argnames=["f"])
def abm_time_step_fixed(
    tableau: TableauABM,
    f: Callable,
    state: ABMState,
    max_dt: Optional[float],
):
    r"""
    Performs one fixed ABM step from current time.
    Args:
        tableau: Integration tableau containing the coefficeints for integration.
            The tableau should contain method step_with_error(f, t, dt, y_t, state).
        f: A callable ODE function.
            Given a time `t` and a state `y_t`, it should return the partial
            derivatives in the same format as `y_t`. The dunction should also accept
            supplementary arguments, such as code:`stage`.
        state: Intagrator state containing the current state (t,y) and stablity information.
        max_dt: The maximal value for the time step `dt`.

    Returns:
        Updated state of the integrator.
    """
    state = general_time_step_fixed(tableau, f, state, max_dt)

    last_f = f(state.t.value, state.y, stage=tableau.order + 2)
    return state.replace(
        F_history=tree_map(
            lambda H, x: jnp.roll(H, 1, axis=0).at[0].set(x),
            state.F_history,
            last_f,
        ),
    )


@dataclass(_frozen=False)
class ABMIntegrator(Integrator):
    r"""
    Ordinary-Differential-Equation Adams-Bashforth-Moulton integrator.
    Given an ODE-function f, it integrates the derivatives to obtain the solution
    at the next time step using Adams-Bashforth-Moulton methods.
    """

    def __post_init__(self):
        super().__post_init__()

        # generate the history of derivatives needed
        # there, F_history[0] contains the last derivatives
        history = expand_dim(self.y0, self.tableau.data.order)
        history = tree_map(
            lambda H, x: H.at[0].set(x), history, self.f(self.t0, self.y0, stage=0)
        )

        self._state = ABMState(
            step_no=0,
            step_no_total=0,
            t=nk.utils.KahanSum(self.t0),
            y=self.y0,
            F_history=history,
            dt=self.initial_dt,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=SolverFlags(0),
        )

    def _do_step_fixed(self, state, max_dt=None):
        r"""
        Performs one full ABM step with a fixed time-step value code:`dt`
        """
        return abm_time_step_fixed(
            tableau=self.tableau.data,
            f=self.f,
            state=state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, state, max_dt=None):
        r"""
        Performs one full ABM step with an adaptive time-step value code:`dt`
        """
        return abm_time_step_adaptive(
            tableau=self.tableau.data,
            f=self.f,
            state=state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )[0]
