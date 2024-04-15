import jax
from netket.utils.struct import dataclass
from netket.utils.types import Array, PyTree

from .._state import IntegratorState
from .._structures import expand_dim


@dataclass(_frozen=False)
class ABMState(IntegratorState):
    F_history: Array = None
    """History of ODEs. F_history[0] is the ODE at current time."""
