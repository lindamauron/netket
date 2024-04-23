# Copyright 2021 The NetKet Authors - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np

import jax
from functools import partial

from netket.experimental.dynamics import (
    Euler,
    Heun,
    Midpoint,
    RK4,
    RK12,
    RK23,
    RK45,
    ABM,
)
from netket.experimental.dynamics._rk._tableau import (
    bt_feuler,
    bt_heun,
    bt_midpoint,
    bt_rk12,
    bt_rk23,
    bt_rk4,
    bt_rk4_dopri,
    bt_rk4_fehlberg,
)

from netket.experimental.dynamics._abm._tableau import abm

from .. import common

pytestmark = common.skipif_mpi


tableaus_rk = {
    "bt_feuler": bt_feuler,
    "bt_heun": bt_heun,
    "bt_midpoint": bt_midpoint,
    "bt_rk12": bt_rk12,
    "bt_rk23": bt_rk23,
    "bt_rk4": bt_rk4,
    "bt_rk4_dopri": bt_rk4_dopri,
    "bt_rk4_fehlberg": bt_rk4_fehlberg,
}

tableaus_abm = [abm(order=k) for k in range(1, 10)]

fixed_step_solvers = {
    "Euler": Euler,
    "Heun": Heun,
    "Midpoint": Midpoint,
    "RK4": RK4,
    "ABM2": partial(ABM, order=2),
    "ABM3": partial(ABM, order=3),
    "ABM4": partial(ABM, order=4),
    "ABM5": partial(ABM, order=5),
}

adaptive_solvers = {
    "RK12": RK12,
    "RK23": RK23,
    "RK45": RK45,
    "ABM2": partial(ABM, order=2),
    "ABM3": partial(ABM, order=3),
    "ABM4": partial(ABM, order=4),
    "ABM5": partial(ABM, order=5),
}

rk_tableaus_params = [pytest.param(obj, id=name) for name, obj in tableaus_rk.items()]
fixed_step_solvers_params = [
    pytest.param(obj, id=name) for name, obj in fixed_step_solvers.items()
]
adaptive_solvers_params = [
    pytest.param(obj, id=name) for name, obj in adaptive_solvers.items()
]


@pytest.mark.parametrize("tableau", rk_tableaus_params)
def test_tableau_rk(tableau: str):
    assert tableau.name != ""

    td = tableau.data

    for x in td.a, td.b, td.c:
        assert np.all(np.isfinite(x))

    assert tableau.a.ndim == 2
    # a should be strictly upper triangular
    np.testing.assert_array_equal(np.triu(td.a), np.zeros_like(td.a))
    # c's should be in [0, 1]
    assert np.all(td.c >= 0.0)
    assert np.all(td.c <= 1.0)

    assert len(td.order) in (1, 2)
    assert len(td.order) == td.b.ndim

    assert td.a.shape[0] == td.a.shape[1]
    assert td.a.shape[0] == td.b.shape[-1]
    assert td.a.shape[0] == td.c.shape[0]
    if len(td.order) == 2:
        assert td.b.shape[0] == 2


@pytest.mark.parametrize("tableau", tableaus_abm)
def test_tableau_abm(tableau: str):
    assert tableau.name != ""

    td = tableau.data

    for x in td.alphas, td.betas:
        assert np.all(np.isfinite(x))

    assert td.alphas.shape == td.betas.shape
    assert td.order == td.alphas.shape[0]

    assert td.alphas.ndim == 1
    assert td.betas.ndim == 1
    # the sum of alphas and betas should be 1
    assert np.isclose(td.alphas.sum(), 1)
    assert np.isclose(td.betas.sum(), 1)


# we skip the last fixed step solvers since they can be used with adaptive time-stepping
@pytest.mark.parametrize("method", fixed_step_solvers_params[:-4])
def test_fixed_adaptive_error(method):
    with pytest.raises(TypeError):
        method(dt=0.01, adaptive=True)


@pytest.mark.parametrize("method", fixed_step_solvers_params)
def test_ode_solver(method):
    def ode(t, x, **_):
        return -t * x

    dt = 0.01
    n_steps = 100
    solver = method(dt=dt)

    y0 = np.array([1.0])
    times = np.linspace(0, n_steps * dt, n_steps, endpoint=False)

    y_ref = y0 * np.exp(-(times**2) / 2)

    solv = solver(ode, 0.0, y0)

    t = []
    y_t = []
    for _ in range(n_steps):
        t.append(solv.t)
        y_t.append(solv.y)
        solv.step()
    y_t = np.asarray(y_t)

    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(y_t))

    np.testing.assert_allclose(t, times)

    # somewhat arbitrary tolerances, that may still help spot
    # errors introduced later
    rtol = {"Euler": 1e-2, "RK4": 5e-4}.get(solver.tableau.name, 1e-3)
    np.testing.assert_allclose(y_t[:, 0], y_ref, rtol=rtol)


def test_ode_repr():
    dt = 0.01

    def ode(t, x, **_):
        return -t * x

    for solver in [RK23(dt, adaptive=True), ABM(dt, order=4, adaptive=True)]:
        y0 = np.array([1.0])
        solv = solver(ode, 0.0, y0)

        assert isinstance(repr(solv), str)
        assert isinstance(repr(solv._state), str)

        @jax.jit
        def _test_jit_repr(x):
            return 1

        _test_jit_repr(solv._state)
        # _test_jit_repr(solv)  # this is broken. should be fixed in the zukumft


def test_solver_t0_is_integer():
    # See issue netket/netket#1735
    # https://github.com/netket/netket/issues/1735

    def df(t, y, stage=None):
        return np.sin(t) ** 2 * y

    for init_config in [
        RK23(dt=0.04, adaptive=True, atol=1e-3, rtol=1e-3, dt_limits=[1e-3, 1e-1]),
        ABM(
            dt=0.04,
            order=4,
            adaptive=True,
            atol=1e-3,
            rtol=1e-3,
            dt_limits=[1e-3, 1e-1],
        ),
    ]:
        integrator = init_config(
            df, 0, np.array([1.0])
        )  # <-- the second argument has to be a float

        integrator.step()
        assert integrator.t > 0.0
        assert integrator.t.dtype == integrator.dt.dtype


@pytest.mark.parametrize("solver", adaptive_solvers_params)
def test_adaptive_solver(solver):
    tol = 1e-7

    def ode(t, x, **_):
        return -t * x

    y0 = np.array([1.0])
    solv = solver(dt=0.2, adaptive=True, atol=0.0, rtol=tol)(ode, 0.0, y0)

    t = []
    y_t = []
    last_step = -1
    while solv.t <= 2.0:
        # print(solv._state)
        if solv._state.step_no != last_step:
            last_step = solv._state.step_no
            t.append(solv.t)
            y_t.append(solv.y)
        solv.step()
    y_t = np.asarray(y_t)
    t = np.asarray(t)

    # print(t)
    y_ref = y0 * np.exp(-(t**2) / 2)

    np.testing.assert_allclose(y_t[:, 0], y_ref, rtol=1e-5)
