from .._integrator import (
    IntegratorConfig,
)


class RKIntegratorConfig(IntegratorConfig):
    r"""
    A configurator for instantiation of the RK-integrator.
    This allows to define the integrator (actually the IntegratorConfig) in a
    first time, pass it as an argument to a driver which will set it by calling it.
    """

    def __repr__(self):
        return "RK" + super().__repr__()
