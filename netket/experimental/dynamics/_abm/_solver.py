from . import _tableau as tb
from .._integrator import IntegratorConfig
from ._integrator import ABMIntegrator
from .._structures import (
    args_adaptive_docstring,
    args_fixed_dt_docstring,
    append_docstring,
)


class ABMConfig(IntegratorConfig):
    def __init__(self, dt, tableau, *, adaptive=False, **kwargs):
        self.dt = dt
        self.adaptive = adaptive
        self.kwargs = kwargs
        self.tableau = tableau

    def __call__(self, f, t0, y0, *, norm=None):
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


@append_docstring(args_adaptive_docstring)
def ABM(dt, order, **kwargs):
    r"""
    The Adams-Bashforth-Moulton method of order s.
    """
    return ABMConfig(dt, tableau=tb.abm(order=order), **kwargs)


@append_docstring(args_fixed_dt_docstring)
def AB(dt, order):
    r"""
    The Adams-Bashforth-Moulton method of order s.
    """
    return ABMConfig(dt, tableau=tb.ab(order=order))
