from netket.utils.struct import dataclass

from .._integrator import (
    general_time_step_adaptive,
    general_time_step_fixed,
    Integrator,
    IntegratorConfig,
)


@dataclass(_frozen=False)
class RKIntegrator(Integrator):

    def _do_step_fixed(self, rk_state, max_dt=None):
        return general_time_step_fixed(
            self.tableau.data,
            self.f,
            rk_state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, rk_state, max_dt=None):
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

    def __call__(self, f, t0, y0, *, norm=None):
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
