from netket.utils.struct import dataclass
from netket.utils.types import Array

from .._state import IntegratorState


@dataclass(_frozen=False)
class ABMState(IntegratorState):
    F_history: Array = None
    """History of ODEs. F_history[0] is the ODE at current time."""
