import jax
from netket.utils.struct import dataclass
from netket.utils.types import Array, PyTree

from .._state import IntegratorState
from .._structures import expand_dim

@dataclass(_frozen=False)
class ABMState(IntegratorState):
    y_history: Array = None
    """History of solutions. y_history[0] is the solution at current time."""
    t_history: Array = None
    """History of time of solutions. t_history[0] is the current time."""

    # def __post_init__(self):
        
    #     history = expand_dim(self.y, self.tableau.order)
    #     self.y_history = jax.tree_map(lambda H,y: H.at[0].set( y ), history, self.y)

    #     times = jax.numpy.zeros(self.tableau.order,jax.numpy.float64)
    #     self.t_history = times.at[0].set( self.t.value )