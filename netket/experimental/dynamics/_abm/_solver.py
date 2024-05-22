from . import _tableau as tb
from .._integrator import IntegratorConfig
from ._integrator import ABMIntegrator
from .._structures import (
    args_adaptive_docstring,
    args_fixed_dt_docstring,
    append_docstring,
)


class ABMConfig(IntegratorConfig):
    r"""
    A configurator for instantiation of the ABM integrator.
    This allows to define the integrator (actually the IntegratorConfig) in a
    first time, pass it as an argument to a driver which will set it by calling it.
    """

    def __call__(self, f, t0, y0, *, norm=None):
        r"""
        Instantiates an ABM integrator given the parameters given in
        the first instance and passed as arguments.
        Args:
            f: The ODE function.
            t0: The initial time.
            y0: The initial state.
            norm: The error norm.

        Returns:
            An Integrator with according parameters.
        """
        return ABMIntegrator(
            self.tableau,
            f,
            t0,
            y0,
            initial_dt=self.dt,
            use_adaptive=self.adaptive,
            norm=norm,
            **self.kwargs,
        )


@append_docstring(
    args_adaptive_docstring
    + r"""
        order: Convergence order :math:`s` of the scheme.
            (i.e. number of operations and order of the error).
        """
)
def ABM(dt, order, **kwargs):
    r"""
    The Adams-Bashforth-Moulton method of order :math:`s`.

    """
    return ABMConfig(dt, tableau=tb.abm(order=order), **kwargs)


@append_docstring(
    args_fixed_dt_docstring
    + r"""
        order: Convergence order :math:`s` of the scheme.
            (i.e. number of operations and order of the error).
        """
)
def AB(dt, order):
    r"""
    The Adams-Bashforth method of order :math:`s`.

    """
    return ABMConfig(dt, tableau=tb.ab(order=order))
