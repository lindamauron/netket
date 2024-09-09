from netket.utils.struct import dataclass
from netket.utils.types import Array

from .._state import IntegratorState


@dataclass(_frozen=False)
class ABMState(IntegratorState):
    r"""
    Dataclass containing the state of an ODE solver.
    In particular, it stores the current state of the system, former usefull values
    and information about integration (number of step, errors, etc)
    """

    F_history: Array = None
    """History of ODEs. F_history[0] is the ODE at current time."""
