from enum import IntFlag, auto
from functools import partial, wraps
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import netket as nk
from netket import config
from netket.utils.mpi.primitives import mpi_all_jax
from netket.utils.struct import dataclass, field
from netket.utils.types import Array, PyTree
from netket.utils.numbers import dtype as _dtype

from . import _tableau as rkt
from .._structures import (
    maybe_jax_jit,
    euclidean_norm,
    maximum_norm,
    scaled_error,
    propose_time_step,
    set_flag_jax,
    LimitsType,
)
from .._state import SolverFlags, IntegratorState
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
