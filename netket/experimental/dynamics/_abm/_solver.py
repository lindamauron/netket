from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from netket.experimental.dynamics._integrator_state import IntegratorState
from netket.utils.types import Array
from .._utils import expand_dim, maybe_jax_jit
from .._solver import AbstractSolver

from ._tableau import TableauABM, default_dtype
from ._state import ABMState


class ABMSolver(AbstractSolver):
    r"""
    Class representing the tableau of an explicit Adams-Bashforth-Moulton method [1,2,3] of order :math:`s`,
    which, given the ODE :math:`dy/dt = F(t, y)`, updates the solution as

    .. math::
        y_{l}^{(p)} = y_{l-1} + dt*\sum_{j=1}^{s-1} \beta_j F_{l-j}

    where the index l denotes the time step t_l, with the partial derivatives

    .. math::
        F_l = F(t_l, y_t_l).

    The prediction can then be corrected using the Adams-Moulton method:
    .. math::
        y_{l} = y_{l-1} + dt*\sum_{j=0}^{s-2} \alpha_j F_{l-j}

    where F_{l} = f(t_{l},y_{l}^{(p)}).
    If :code:`self.is_adaptive`, the predictor solution is used to estimate the error

    ..math::
        y_{\mathrm{err}} = y_{l} - y_{l}^{(p)}

    [1] J. Stoer and R. Bulirsch, Introduction to Numerical Analysis, Springer NY (2002).
    [2] J. C. Butcher, Numerical Methods for Ordinary Differential Equations, John Wiley & Sons Ltd, 2008
    [3] https://en.wikipedia.org/wiki/Linear_multistep_method
    """

    tableau: TableauABM

    def __init__(self, dt, tableau, adaptive=False, **kwargs):
        self.tableau = tableau
        if adaptive and not tableau.is_adaptive:
            raise AttributeError(f"Tableau of type {tableau} cannot be adaptve.")
        super().__init__(dt=dt, adaptive=adaptive, **kwargs)

    def __repr__(self) -> str:
        return "{}(tableau={}, dt={}, adaptive={}, integrator_parameters={})".format(
            "ABMSolver",
            self.tableau,
            self.dt,
            self.adaptive,
            self.integrator_params,
        )

    @property
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        return self.tableau.is_explicit

    @property
    def is_adaptive(self):
        """Boolean indication whether the integrator can be adaptive."""
        return self.tableau.is_adaptive

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the scheme.
        """
        return self.tableau.stages

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.adaptive:
            return None
        else:
            return self.tableau.error_order

    def _init_state(self, integrator_state: IntegratorState) -> ABMState:
        return ABMState(
            t0=integrator_state.t.value,
            y0=integrator_state.y,
            order=self.tableau.order,
            step_no=integrator_state.step_no,
        )

    def step(self, f: Callable, dt: float, t: float, y_t: Array, state: ABMState):
        """Perform one fixed-size ABM step from `t` to `t + dt`."""

        ## start by initializing the state sice it could not be done in the ABMState
        state = jax.lax.cond(
            state.step_no == 0,
            lambda _state: _state.replace(
                F_history=tree_map(
                    lambda H, x: H.at[0].set(x),
                    _state.F_history,
                    f(t, y_t),
                ),
            ),
            lambda _state: _state,
            state,
        )

        ## we use RK4 for intialization since we need a history of states for the abm method
        ## we then continue with abm
        y_tp1 = jax.lax.cond(
            state.step_no < self.tableau.order + 3,
            lambda y: self._rk4(f, dt, t, y),
            lambda y: self._abm(f, dt, t, y, state.F_history),
            y_t,
        )

        ## return new sol and updated state
        return y_tp1, state.replace(
            F_history=tree_map(
                lambda H, x: jnp.roll(H, 1, axis=0).at[0].set(x),
                state.F_history,
                f(t + dt, y_tp1),
            ),
            step_no=state.step_no + 1,
        )

    def step_with_error(
        self, f: Callable, dt: float, t: float, y_t: Array, state: ABMState
    ):
        """
        Perform one fixed-size ABM step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        ## start by initializing the state sice it could not be done in the ABMState
        state = jax.lax.cond(
            state.step_no == 0,
            lambda state: state.replace(
                F_history=tree_map(
                    lambda H, x: H.at[0].set(x),
                    state.F_history,
                    f(t, y_t),
                ),
            ),
            lambda state: state,
            state,
        )

        ## we use RK4 for intialization since we need a history of states for the abm method
        ## we then continue with abm
        y_tp1, y_err = jax.lax.cond(
            state.step_no < self.stages - 1,
            lambda y: self._rk4_with_error(f, dt, t, y),
            lambda y: self._abm_with_error(f, dt, t, y, state.F_history),
            y_t,
        )

        ## return new sol and updated state
        return (
            y_tp1,
            y_err,
            state.replace(
                F_history=tree_map(
                    lambda H, x: jnp.roll(H, 1, axis=0).at[0].set(x),
                    state.F_history,
                    f(t + dt, y_tp1, stage=1),
                ),
                step_no=state.step_no + 1,
            ),
        )

    @partial(maybe_jax_jit, static_argnames=["f"])
    def _abm_with_error(self, f, dt, t, y_t, past_F):
        """
        Perform one fixed-size ABM step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        ## F[k] = f(t_{n-k}, y_{n-k}), k in [0,s-1] with t_n = t, y_n = y_t

        # Predictor step
        ## Bashforths integration
        y_tilde = tree_map(
            lambda y, F: y
            + jnp.asarray(dt, dtype=y.dtype)
            * jnp.tensordot(
                jnp.asarray(self.tableau.betas, dtype=y.dtype), F, axes=[0, 0]
            ),
            y_t,
            past_F,
        )

        # Corrector step
        ## change the forces by replacing with the one extrapolated
        F = tree_map(
            lambda H, f_l: jnp.roll(H, 1, axis=0).at[0].set(f_l),
            past_F,
            f(t + dt, y_tilde, stage=0),
        )
        ## Moulton integration
        y_tp1 = tree_map(
            lambda y, F: y
            + jnp.asarray(dt, dtype=y.dtype)
            * jnp.tensordot(
                jnp.asarray(self.tableau.alphas, dtype=y.dtype), F, axes=[0, 0]
            ),
            y_t,
            F,
        )

        # Error estimate
        y_err = self.tableau.gamma * tree_map(lambda x, y: x - y, y_tilde, y_tp1)

        return y_tp1, y_err

    @partial(maybe_jax_jit, static_argnames=["self", "f"])
    def _abm(self, f, dt, t, y_t, past_F):
        """
        Perform one fixed-size ABM step from `t` to `t + dt`.
        """
        ## F[k] = f(t_{n-k}, y_{n-k}), k in [0,s-1] with t_n = t, y_n = y_t

        # Predictor step
        ## Bashforths integration
        y_tilde = tree_map(
            lambda y, F: y
            + jnp.asarray(dt, dtype=y.dtype)
            * jnp.tensordot(
                jnp.asarray(self.tableau.betas, dtype=y.dtype), F, axes=[0, 0]
            ),
            y_t,
            past_F,
        )

        # Corrector step
        ## change the forces by replacing with the one extrapolated
        past_F = tree_map(
            lambda H, f_l: jnp.roll(H, 1, axis=0).at[0].set(f_l),
            past_F,
            f(t + dt, y_tilde, stage=0),
        )
        ## Moulton integration
        y_tp1 = tree_map(
            lambda y, F: y
            + jnp.asarray(dt, dtype=y.dtype)
            * jnp.tensordot(
                jnp.asarray(self.tableau.alphas, dtype=y.dtype), F, axes=[0, 0]
            ),
            y_t,
            past_F,
        )

        return y_tp1

    @partial(maybe_jax_jit, static_argnames=["f"])
    def _rk4_with_error(self, f, dt, t, y_t):
        """
        Perform one fixed-size RK4 step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        a = jnp.array(
            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]],
            dtype=default_dtype,
        )

        times = t + jnp.array([0, 1 / 2, 1 / 2, 1], dtype=default_dtype) * dt
        k = expand_dim(y_t, 4)
        for l in range(4):
            dy_l = tree_map(
                lambda k: jnp.tensordot(jnp.asarray(a[l], dtype=k.dtype), k, axes=1),
                k,
            )
            y_l = tree_map(
                lambda y_t, dy_l: jnp.asarray(y_t + dt * dy_l, dtype=dy_l.dtype),
                y_t,
                dy_l,
            )
            k_l = f(times[l], y_l, stage=l)
            k = tree_map(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        b = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=default_dtype)
        y_tp1 = tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(b, dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )

        db = jnp.array([5 / 72, -1 / 12, -1 / 9, 1 / 8], dtype=default_dtype)
        y_err = tree_map(
            lambda k: jnp.asarray(dt, dtype=k.dtype)
            * jnp.tensordot(jnp.asarray(db, dtype=k.dtype), k, axes=1),
            k,
        )

        return y_tp1, y_err

    @partial(maybe_jax_jit, static_argnames=["f"])
    def _rk4(self, f, dt, t, y_t):
        """
        Perform one fixed-size RK4 step from `t` to `t + dt`.
        """
        a = jnp.array(
            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]],
            dtype=default_dtype,
        )

        times = t + jnp.array([0, 1 / 2, 1 / 2, 1], dtype=default_dtype) * dt
        k = expand_dim(y_t, 4)
        for l in range(4):
            dy_l = tree_map(
                lambda k: jnp.tensordot(jnp.asarray(a[l], dtype=k.dtype), k, axes=1),
                k,
            )
            y_l = tree_map(
                lambda y_t, dy_l: jnp.asarray(y_t + dt * dy_l, dtype=dy_l.dtype),
                y_t,
                dy_l,
            )
            k_l = f(times[l], y_l, stage=l)
            k = tree_map(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        b = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=default_dtype)
        y_tp1 = tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(b, dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )

        return y_tp1


def ABM(dt, order, alphas=None, betas=None, **kwargs):
    """
    ABMSolver for a given order.
    Args:
        dt: time-step size
        order: The order of the ABM solver
            The error will scale as dt^order
        alphas, betas: The optional coefficients for the corrector resp. predictor step
            For orders <=10, the coefficients are pre-computed and do not need to be provided.

        adaptive:
        rtol:
        atol:
        dt_limits
    """
    return ABMSolver(
        dt=dt, tableau=TableauABM(order=order, alphas=alphas, betas=betas), **kwargs
    )
