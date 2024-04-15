from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import netket as nk
from netket.utils.struct import dataclass

from .._structures import maybe_jax_jit,LimitsType
from .._integrator import Integrator,general_time_step_adaptive, general_time_step_fixed
from .._state import SolverFlags

from ._tableau import TableauABM, expand_dim
from ._state import ABMState


@partial(maybe_jax_jit, static_argnames=["f", "norm_fn", "dt_limits"])
def general_abm_adaptive(
    tableau: TableauABM,
    f: Callable,
    state: ABMState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType, ### change to limtsDType
):
    state, flag = general_time_step_adaptive(tableau,f,state,atol,rtol,norm_fn,max_dt,dt_limits)

    return jax.lax.cond(
        flag,
        lambda _: state.replace(
            y_history=jax.tree_map(lambda H, y : jnp.roll(H,1,axis=0).at[0].set(y), state.y_history, state.y),
            t_history=jnp.roll(state.t_history,1,axis=0).at[0].set(state.t.value),
        ),
        lambda _: state,
        None,
    )



@partial(maybe_jax_jit, static_argnames=["f"])
def general_abm_fixed(
    tableau: TableauABM,
    f: Callable,
    state: ABMState,
    max_dt: Optional[float],
):
    state = general_time_step_fixed(tableau,f,state,max_dt)
    new_t = state.t.value
    new_y = state.y

    return state.replace(
        y_history=jax.tree_map(lambda H, y : jnp.roll(H,1,axis=0).at[0].set(y), state.y_history, new_y),
        t_history=jnp.roll(state.t_history,1,axis=0).at[0].set(new_t)
    )


@dataclass(_frozen=False)
class ABMIntegrator(Integrator):
    tableau: TableauABM

    def __post_init__(self):
        super().__post_init__()

        history = expand_dim(self.y0, self.tableau.order)
        history = jax.tree_map(lambda H,y: H.at[0].set( y ), history, self.y0)

        times = jnp.zeros(self.tableau.order)
        times = times.at[0].set( self.t0 )
        
        self._state = ABMState(
            step_no=0,
            step_no_total=0,
            t=nk.utils.KahanSum(self.t0),
            y=self.y0,
            y_history=history,
            t_history=times,
            dt=self.initial_dt,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=SolverFlags(0),
        )


    def _do_step_fixed(self, state, max_dt=None):
        return general_abm_fixed(
            tableau=self.tableau,
            f=self.f,
            state=state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, state, max_dt=None):
        return general_abm_adaptive(
            tableau=self.tableau,
            f=self.f,
            state=state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )
