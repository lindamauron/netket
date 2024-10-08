import jax

from netket.utils.types import Array

from .._utils import expand_dim
from .._solver import AbstractSolverState


class ABMState(AbstractSolverState):
    r"""
    Dataclass containing the state of an ABM-ODE solver.
    In particular, it stores the past ODEs to be used in the integration scheme.
    """

    F_history: Array = None
    """History of ODEs. F_history[0] is the solution at current time."""

    step_no: int = 0

    def __init__(self, t0, y0, order, step_no=0):
        self.F_history = expand_dim(y0, order)

        self.step_no = step_no

    def __repr__(self):
        return f"ABMState(step_no={self.step_no}, #ODEs={jax.tree_map(lambda x : x.shape[0], self.F_history)})"
