from typing import Callable

import jax
import jax.numpy as jnp

from netket.utils.struct import dataclass
from netket.utils.types import Array
from .._structures import expand_dim
from .._tableau import Tableau, NamedTableau
from ._state import ABMState

default_dtype = jnp.float64


@dataclass
class TableauABM(Tableau):
    r"""
    Class representing the tableau of an explicit Adams-Bashforth-Moulton method [1,2,3],
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
    If :code:`self.is_adaptive`, the predictor solution is used to esimate the error

    ..math::
        y_{\mathrm{err}} = y_{l} - y_{l}^{(p)}

    [1] J. Stoer and R. Bulirsch, Introduction to Numerical Analysis, Springer NY (2002).
    [2] J. C. Butcher, Numerical Methods for Ordinary Differential Equations, John Wiley & Sons Ltd, 2008
    [3] https://en.wikipedia.org/wiki/Linear_multistep_method
    """

    order: int
    """The order of the tableau"""
    betas: jax.numpy.ndarray
    """Coefficients for the predictor step."""
    alphas: jax.numpy.ndarray
    """Coefficients for the corrector step."""

    @property
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        return jnp.isclose(self.alphas, 0).all()

    @property
    def is_adaptive(self):
        """Boolean indication whether the integrator can beå adaptive."""
        return not self.is_explicit

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the scheme.
        """
        return self.order + 1

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            return self.order + 1

    def __repr__(self):
        if self.is_explicit:
            return f"AB{self.order}"
        else:
            return f"ABM{self.order}"

    @property
    def name(self):
        """The name of the tableau."""
        return self.__repr__()

    def step(self, f: Callable, t: float, dt: float, y_t: Array, state: ABMState):
        """Perform one fixed-size ABM step from `t` to `t + dt`."""
        # we use RK4 for intialization since we need a history of states for the abm method
        if state.step_no_total < self.stages - 1:
            y_tp1, _ = self._rk4(f, t, dt, y_t)

        else:
            y_tp1, _ = self._abm(f, t, dt, y_t, state.F_history)

        return y_tp1

    def step_with_error(
        self, f: Callable, t: float, dt: float, y_t: Array, state: ABMState
    ):
        """
        Perform one fixed-size ABM step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        # we use RK4 for intialization since we need a history of states for the abm method
        if state.step_no_total < self.stages - 1:
            y_tp1, y_err = self._rk4(f, t, dt, y_t)

        else:
            y_tp1, y_err = self._abm(f, t, dt, y_t, state.F_history)

        return y_tp1, y_err

    def _abm(self, f, t, dt, y_t, F_history):
        """
        Perform one fixed-size ABM step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        ## F_history[0] corresponds to F(y(t),t)=F(y_n,t_n), we are looking for y(t+dt)=y_n+1

        # F[k] = f(t_{n-1+k}, y_{n-1+k}), k in [0,s-1]
        y_tilde = jax.tree_map(
            lambda y_t, F: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(self.betas, dtype=y_t.dtype), F, axes=[0, 0]),
            y_t,
            F_history,
        )

        # now the corrector step
        if not self.is_explicit:
            # change the forces with by adding the one extrapolated
            F = jax.tree_map(
                lambda H, f_l: jnp.roll(H, 1, axis=0).at[0].set(f_l),
                F_history,
                f(t + dt, y_tilde, stage=self.order),
            )

            y_tp1 = jax.tree_map(
                lambda y_t, F: y_t
                + jnp.asarray(dt, dtype=y_t.dtype)
                * jnp.tensordot(
                    jnp.asarray(self.alphas, dtype=y_t.dtype), F, axes=[0, 0]
                ),
                y_t,
                F,
            )
        else:
            y_tp1 = y_tilde

            # prediction of order from before
            y_tilde = jax.tree_map(
                lambda y_t, F: y_t
                + jnp.asarray(dt, dtype=y_t.dtype)
                * jnp.tensordot(
                    jnp.asarray(self.betas, dtype=y_t.dtype)[:-1], F[:-1], axes=[0, 0]
                ),
                y_t,
                F,
            )

        y_err = jax.tree_map(lambda x, y: x - y, y_tp1, y_tilde)

        return y_tp1, y_err

    def _rk4(self, f, t, dt, y_t):
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
            dy_l = jax.tree_map(
                lambda k: jnp.tensordot(jnp.asarray(a[l], dtype=k.dtype), k, axes=1),
                k,
            )
            y_l = jax.tree_map(
                lambda y_t, dy_l: jnp.asarray(y_t + dt * dy_l, dtype=dy_l.dtype),
                y_t,
                dy_l,
            )
            k_l = f(times[l], y_l, stage=l)
            k = jax.tree_map(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        b = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=default_dtype)
        y_tp1 = jax.tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(b, dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )

        db = jnp.array([5 / 72, -1 / 12, -1 / 9, 1 / 8], dtype=default_dtype)
        y_err = jax.tree_map(
            lambda k: jnp.asarray(dt, dtype=k.dtype)
            * jnp.tensordot(jnp.asarray(db, dtype=k.dtype), k, axes=1),
            k,
        )

        return y_tp1, y_err


r"""
To instantiate a Tableau for any order (not computeed yet), one first needs to find the coefficients :
order = s
:math: `\beta_{s+1-i} = \int_{0}^1 \prod_{l=0 ,l \n eq i}^q \frac{u+l}{l-i} du, i=1,\dots,s`
:math: `\alpha_{i} = \int_{-1}^0 \prod_{l=0 ,l \n eq i}^q \frac{u+l}{l-i} du, i=0,\dots,s-1`

then, instantiate as TableauABM(order=s, betas=jnp.array(betas), alphas=jnp.array(alphas))
"""


# bashforths
betas = {  # f{n-1},....,f_{n-s}
    1: jnp.array([1], default_dtype),
    2: jnp.array([3, -1], default_dtype) / 2,
    3: jnp.array([23, -16, 5], default_dtype) / 12,
    4: jnp.array([55, -59, 37, -9], default_dtype) / 24,
    5: jnp.array([1901, -2774, 2616, -1274, 251], default_dtype) / 720,
    6: jnp.array([4277, -7923, 9982, -7298, 2877, -475], default_dtype) / 1440,
    7: jnp.array(
        [198721, -447288, 705549, -688256, 407139, -134472, 19087], default_dtype
    )
    / 60480,
    8: jnp.array(
        [434241, -1152169, 2183877, -2664477, 2102243, -1041723, 295767, -36799],
        default_dtype,
    )
    / 120960,
    9: jnp.array(
        [
            14097247,
            -43125206,
            95476786,
            -139855262,
            137968480,
            -91172642,
            38833486,
            -9664106,
            1070017,
        ],
        default_dtype,
    )
    / 3628800,
    10: jnp.array(
        [
            30277247,
            -104995189,
            265932680,
            -454661776,
            538363838,
            -444772162,
            252618224,
            -94307320,
            20884811,
            -2082753,
        ],
        default_dtype,
    )
    / 7257600,
}

# moulton
alphas = {  # f_{n},...,f_{n-s+1}
    1: jnp.array([1], default_dtype),
    2: jnp.array([1, 1], default_dtype) / 2,
    3: jnp.array([5, 8, -1], default_dtype) / 12,
    4: jnp.array([9, 19, -5, 1], default_dtype) / 24,
    5: jnp.array([251, 646, -264, 106, -19], default_dtype) / 720,
    6: jnp.array([475, 1427, -798, 482, -173, 27], default_dtype) / 1440,
    7: jnp.array([19087, 65112, -46461, 37504, -20211, 6312, -863], default_dtype)
    / 60480,
    8: jnp.array(
        [36799, 139849, -121797, 123133, -88547, 41499, -11351, 1375], default_dtype
    )
    / 120960,
    9: jnp.array(
        [
            1070017,
            4467094,
            -4604594,
            5595358,
            -5033120,
            3146338,
            -1291214,
            312874,
            -33953,
        ],
        default_dtype,
    )
    / 3628800,
    10: jnp.array(
        [
            2082753,
            9449717,
            -11271304,
            16002320,
            -17283646,
            13510082,
            -7394032,
            2687864,
            -583435,
            57281,
        ],
        default_dtype,
    )
    / 7257600,
}


def abm(order):
    """
    ABM tableau for a given order.
    """
    if order in list(alphas.keys()):
        tab = TableauABM(order=order, alphas=alphas[order], betas=betas[order])
        return NamedTableau(tab, f"ABM{order}")
    else:
        raise NotImplementedError(
            f"The coefficients for a Adams-Bashforth-Moulton of order {order} have not been implemented yet, you need to compute them yourself"
        )


def ab(order):
    """
    AB tableau for a given order.
    """
    if order in list(betas.keys()):
        tab = TableauABM(
            order=order, betas=betas[order], alphas=jnp.zeros(order, default_dtype)
        )
        return NamedTableau(tab, f"AB{order}")
    else:
        raise NotImplementedError(
            f"The coefficients for a Adams-Bashforth of order {order} have not been implemented yet, you need to compute them yourself"
        )
