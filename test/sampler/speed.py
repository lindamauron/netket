# Copyright 2021 The NetKet Authors - All rights reserved.
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

import flax
from jax import numpy as jnp

from .. import common

import numpy as np
import pytest
from scipy.stats import combine_pvalues, chisquare, multivariate_normal, kstest
import jax
from jax.nn.initializers import normal

import netket as nk
from netket import config
from netket.hilbert import DiscreteHilbert, Particle
from netket.utils import array_in, mpi
from netket.jax.sharding import device_count_per_rank

from netket import experimental as nkx
import time


pytestmark = common.skipif_mpi

nk.config.update("NETKET_EXPERIMENTAL", True)
np.random.seed(1234)

WEIGHT_SEED = 1234
SAMPLER_SEED = 15324


samplers = {}


# TESTS FOR SPIN HILBERT
# Constructing a 1d lattice
g = nk.graph.Hypercube(length=4, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

samplers["Exact: Spin"] = nk.sampler.ExactSampler(hi)

samplers["Metropolis(Local): Spin"] = nk.sampler.MetropolisLocal(hi)

# MultipleRules sampler
samplers["Metropolis(MultipleRules[Local,Local]): Spin"] = nk.sampler.MetropolisSampler(
    hi,
    nk.sampler.rules.MultipleRules(
        [nk.sampler.rules.LocalRule(), nk.sampler.rules.LocalRule()], [0.8, 0.2]
    ),
)
from netket.sampler.rules.old import MultipleRules as oldMultiple
# MultipleRules sampler
samplers["Metropolis(OldMultiple[Local,Local]): Spin"] = nk.sampler.MetropolisSampler(
    hi,
    oldMultiple(
        [nk.sampler.rules.LocalRule(), nk.sampler.rules.LocalRule()], [0.8, 0.2]
    ),
)


if not config.netket_experimental_sharding:
    samplers[
        "Metropolis(MultipleRules[Local,Hamiltonian]): Spin"
    ] = nk.sampler.MetropolisSampler(
        hi,
        nk.sampler.rules.MultipleRules(
            [nk.sampler.rules.LocalRule(), nk.sampler.rules.HamiltonianRule(ha)],
            [0.8, 0.2],
        ),
    )

    samplers[
        "Metropolis(oldMultiple[Local,Hamiltonian]): Spin"
    ] = nk.sampler.MetropolisSampler(
        hi,
        oldMultiple(
            [nk.sampler.rules.LocalRule(), nk.sampler.rules.HamiltonianRule(ha)],
            [0.8, 0.2],
        ),
    )



# The following fixture initialises a model and it's weights
# for tests that require it.
@pytest.fixture
def model_and_weights(request):
    def build_model(hilb, sampler=None):
        if isinstance(sampler, nk.sampler.ARDirectSampler):
            ma = nk.models.ARNNDense(
                hilbert=hilb, machine_pow=sampler.machine_pow, layers=3, features=5
            )
        elif isinstance(hilb, Particle):
            ma = nk.models.Gaussian()
        else:
            # Build RBM by default
            ma = nk.models.RBM(
                alpha=1,
                param_dtype=complex,
                kernel_init=normal(stddev=0.1),
                hidden_bias_init=normal(stddev=0.1),
            )

        # init network
        w = ma.init(jax.random.PRNGKey(WEIGHT_SEED), jnp.zeros((1, hilb.size)))

        return ma, w

    # Do something with the data
    return build_model


# The following fixture returns one sampler at a time (and iterates through)
# all samplers.
# it skips tests according to the --sampler cmdline argument introduced in
# conftest.py
@pytest.fixture(
    params=[pytest.param(sampl, id=name) for name, sampl in samplers.items()]
)
def sampler(request):
    cmdline_sampler = request.config.getoption("--sampler").lower()
    if cmdline_sampler == "":
        return request.param
    elif cmdline_sampler in request.node.name.lower():
        return request.param
    else:
        pytest.skip("skipped from command-line argument")


@pytest.fixture(params=[pytest.param(val, id=f", mpow={val}") for val in [1, 2]])
def set_pdf_power(request):
    def fun(sampler):
        cmdline_mpow = request.config.getoption("--mpow").lower()
        if cmdline_mpow == "all":
            # Nothing to skip
            pass
        elif cmdline_mpow == "single":
            # same sampler leads to same rng
            rng = np.random.default_rng(common.hash_for_seed(sampler))
            exponent = rng.integers(1, 3)  # 1 or 2
            if exponent != request.param:
                pytest.skip(
                    "Running only 1 pdf exponent per sampler. Use --mpow=all to run all pdf exponents."
                )
        elif int(cmdline_mpow) != request.param:
            pytest.skip(f"Running only --mpow={cmdline_mpow}.")

        if isinstance(sampler, nk.sampler.ARDirectSampler) and request.param != 2:
            pytest.skip("ARDirectSampler only supports machine_pow = 2.")

        return sampler.replace(machine_pow=request.param)

    return fun


def findrng(rng):
    if hasattr(rng, "_bit_generator"):
        return rng._bit_generator.state["state"]
    else:
        return rng


@pytest.fixture(
    params=[
        pytest.param(
            sampl,
            id=name,
        )
        for name, sampl in samplers.items()
    ]
)
def sampler_c(request):
    cmdline_sampler = request.config.getoption("--sampler").lower()
    if cmdline_sampler == "":
        return request.param
    elif cmdline_sampler in request.node.name.lower():
        return request.param
    else:
        pytest.skip("skipped from command-line argument")


# Testing that samples generated from direct sampling are compatible with those
# generated by markov chain sampling
# here we use a combination of power divergence tests


# !!WARN!! TODO: Flaky test
# This tests do not take into account the fact that our samplers do not necessarily
# produce samples which are uncorrelated. So unless the autocorrelation time is 0, we
# are bound to fail such tests. We should account for that.
@common.skipif_distributed
def test_correct_sampling(sampler_c, model_and_weights, set_pdf_power):
    sampler = set_pdf_power(sampler_c)

    hi = sampler.hilbert
    if isinstance(hi, DiscreteHilbert):
        n_states = hi.n_states

        ma, w = model_and_weights(hi, sampler)

        n_samples = max(40 * n_states, 100)

        ps = (
            np.absolute(nk.nn.to_array(hi, ma, w, normalize=False))
            ** sampler.machine_pow
        )
        ps /= ps.sum()

        n_rep = 6
        pvalues = np.zeros(n_rep)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)

        for jrep in range(n_rep):
            sampler_state = sampler.reset(ma, w, state=sampler_state)

            # Burnout phase
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples // 100
            )

            assert samples.shape == (
                sampler.n_chains,
                n_samples // 100,
                hi.size,
            )
            t0 = time.time()
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_samples
            )
            t = time.time() - t0

            assert samples.shape == (sampler.n_chains, n_samples, hi.size)

            sttn = hi.states_to_numbers(np.asarray(samples.reshape(-1, hi.size)))
            n_s = sttn.size

            # fill in the histogram for sampler
            unique, counts = np.unique(sttn, return_counts=True)
            hist_samp = np.zeros(n_states)
            hist_samp[unique] = counts

            # expected frequencies
            f_exp = n_s * ps
            statistics, pvalues[jrep] = chisquare(hist_samp, f_exp=f_exp)
            assert 0, f"time : {t}"

        s, pval = combine_pvalues(pvalues, method="fisher")
        assert pval > 0.01 or np.max(pvalues) > 0.01


    elif isinstance(hi, Particle):
        # TODO: Find periodic distribution that can be exactly sampled and do the same test.

        ma, w = model_and_weights(hi, sampler)
        n_samples = 5000
        n_discard = 2 * 1024
        n_rep = 6
        pvalues = np.zeros(n_rep)

        sampler_state = sampler.init_state(ma, w, seed=SAMPLER_SEED)
        for jrep in range(n_rep):
            sampler_state = sampler.reset(ma, w, state=sampler_state)

            # Burnout phase
            samples, sampler_state = sampler.sample(
                ma, w, state=sampler_state, chain_length=n_discard
            )

            assert samples.shape == (
                sampler.n_chains,
                n_discard,
                hi.size,
            )
            samples, sampler_state = sampler.sample(
                ma,
                w,
                state=sampler_state,
                chain_length=n_samples,
            )

            assert samples.shape == (sampler.n_chains, n_samples, hi.size)

            samples = samples.reshape(-1, samples.shape[-1])

            dist = multivariate_normal(
                mean=np.zeros(samples.shape[-1]),
                cov=np.linalg.inv(
                    sampler.machine_pow
                    * np.dot(w["params"]["kernel"].T, w["params"]["kernel"])
                ),
            )
            exact_samples = dist.rvs(size=samples.shape[0])

            counts, bins = np.histogramdd(samples, bins=10)
            counts_exact, _ = np.histogramdd(exact_samples, bins=bins)

            statistics, pvalues[jrep] = kstest(
                counts.reshape(-1), counts_exact.reshape(-1)
            )

        s, pval = combine_pvalues(pvalues, method="fisher")

        assert pval > 0.01 or np.max(pvalues) > 0.01
