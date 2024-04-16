from netket.utils.struct import dataclass

from .._integrator import (
    general_time_step_adaptive,
    general_time_step_fixed,
    Integrator,
    IntegratorConfig,
)


@dataclass(_frozen=False)
class RKIntegrator(Integrator):
    r"""
    Ordinary-Differential-Equation Runge-Kutta integrator.
    Given an ODE-function f, it integrates the derivatives to obtain the solution
    at the next time step using Runge-Kutta methods.
    """

    def _do_step_fixed(self, rk_state, max_dt=None):
        r"""
        Performs one full step with a fixed time-step value code:`dt`
        """
        return general_time_step_fixed(
            self.tableau.data,
            self.f,
            rk_state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, rk_state, max_dt=None):
        r"""
        Performs one full step with an adaptive time-step value code:`dt`
        """
        return general_time_step_adaptive(
            self.tableau.data,
            self.f,
            rk_state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )[0]


class RKIntegratorConfig(IntegratorConfig):
    r"""
    A configurator for instantiation of the RK-integrator.
    This allows to define the integrator (actually the IntegratorConfig) in a
    first time, pass it as an argument to a driver which will set it by calling it.
    """

    def __init__(self, dt, tableau, *, adaptive=False, **kwargs):
        r"""
        Args:
            dt: The initial time-step of the integrator.
            tableau: The tableau of coefficients for the integration.
            adaptive: A boolean indicator whether to use an daaptive scheme.
        """
        if not tableau.data.is_adaptive and adaptive:
            raise ValueError(
                "Cannot set `adaptive=True` for a non-adaptive integrator."
            )

        super().__init__(dt=dt, tableau=tableau, adaptive=adaptive, kwargs=kwargs)

    def __call__(self, f, t0, y0, *, norm=None):
        r"""
        Instantiates a RK-integrator given the parameters given in
        the first instance and passed as arguments.
        Args:
            f: The ODE function.
            t0: The initial time.
            y0: The initial state.
            norm: The error norm.

        Returns:
            An Integrator with according parameters.
        """
        return RKIntegrator(
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
        return "RK" + super().__repr__()
