from typing import Callable
from netket.utils.struct import dataclass
from netket.utils.types import Array
from abc import abstractmethod

from ._state import IntegratorState


@dataclass(_frozen=True)
class Tableau:
    r"""
    Class representing the general tableaus for various methods for a given the ODE :math:`dy/dt = F(t, y)`

    If :code:`self.is_adaptive`, the tableau also contains the coefficients
    which can be used to estimate the local truncation error (if necessary).
    """

    order: tuple[int, int]
    """The order of the tableau."""

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self):
        return f"Tableau{self.order}"

    @property
    @abstractmethod
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        pass

    @property
    @abstractmethod
    def is_adaptive(self):
        """Boolean indication whether the integrator can beå adaptive."""
        pass

    @property
    @abstractmethod
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function) of the scheme.
        """
        pass

    @property
    @abstractmethod
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            pass

    @abstractmethod
    def step(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """Perform one fixed-size step from `t` to `t + dt`."""
        pass

    @abstractmethod
    def step_with_error(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """
        Perform one fixed-size step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        pass


@dataclass
class NamedTableau:
    """
    Structure holding the name and the coefficients for the RK-Tableau.
    """

    name: str
    """The name of the Tableau."""
    data: Tableau
    """The tableau containing the coefficients."""

    def __repr__(self) -> str:
        return self.name
