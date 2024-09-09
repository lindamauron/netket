# Copyright 2023 The NetKet Authors - All rights reserved.
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

from typing import Any
from functools import partial

import jax
import jax.numpy as jnp

from flax import linen as nn

from netket import config
from netket.utils.types import Array, PyTree, PRNGKeyT
from netket.jax.sharding import sharding_decorator

# Necessary for the type annotation to work
if config.netket_sphinx_build:
    from netket import sampler

from .base import MetropolisRule


class MultipleRules(MetropolisRule):
    r"""A Metropolis sampling rule that can be used to pick a rule from a list of rules
    with a given probability.

    Each `rule[i]` will be selected with a probability `probabilities[i]`.
    """
    rules: tuple[MetropolisRule, ...]
    """List of rules to be selected from."""
    probabilities: jax.Array
    """Corresponding list of probabilities with which every rule can be
    picked."""

    def __init__(
        self, rules: tuple[MetropolisRule, ...], probabilities: Array
    ) -> MetropolisRule:
        r"""A Metropolis sampling rule that can be used to pick a rule from a list of rules
        with a given probability.

        Each `rule[i]` will be selected with a probability `probabilities[i]`.

        Args:
            rules: A list of rules, one for each subspace of the tensor hilbert space.
            probabilities: A list of probabilities, one for each rule.
        """
        probabilities = jnp.asarray(probabilities)

        if not jnp.allclose(jnp.sum(probabilities), 1.0):
            raise ValueError(
                "The probabilities must sum to 1, but they sum to "
                f"{jnp.sum(probabilities)}."
            )

        if not isinstance(rules, tuple | list) or not all(
            isinstance(r, MetropolisRule) for r in rules
        ):
            raise TypeError(
                "The first argument (rules) must be a tuple of `MetropolisRule` "
                f"rules, but you have passed {type(rules)}."
            )

        if len(probabilities) != len(rules):
            raise ValueError(
                "Length mismatch between the probabilities and the rules: probabilities "
                f"has length {len(probabilities)} , rules has length {len(rules)}."
            )

        self.rules = rules
        self.probabilities = probabilities

    def init_state(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        key: PRNGKeyT,
    ) -> Any | None:
        N = len(self.probabilities)
        keys = jax.random.split(key, N)
        return tuple(
            self.rules[i].init_state(sampler, machine, params, keys[i])
            for i in range(N)
        )

    def reset(
        self,
        sampler: "sampler.MetropolisSampler",  # noqa: F821
        machine: nn.Module,
        params: PyTree,
        sampler_state: "sampler.SamplerState",  # noqa: F821
    ) -> Any | None:
        rule_states = []
        for i in range(len(self.probabilities)):
            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])
            rule_states.append(self.rules[i].reset(sampler, machine, params, _state))
        return tuple(rule_states)

    def transition(self, sampler, machine, parameters, state, key, σ):
        n = len(self.probabilities)
        print(n, sampler.n_batches)
        keys = jax.random.split(key, n + 1)

        indices = jax.random.choice(
            keys[-1],
            n,
            shape=(sampler.n_batches,),
            p=self.probabilities,
        )

        keys = jax.random.split(keys[0], sampler.n_batches)

        transitions = [
            lambda k, s: rule.transition(sampler, machine, parameters, state, k, s)
            for rule in self.rules
        ]

        σp, log_prob_corrs = jax.vmap(
            jax.lax.switch,
            in_axes=(0, None, 0, 0),
            out_axes=(1, 0)
        )(indices, transitions, keys, jnp.expand_dims(σ, 1))

        σp = σp.reshape(sampler.n_batches,-1)
        if log_prob_corrs is not None and any(x is not None for x in log_prob_corrs):
            log_prob_corr = jnp.array(
                [
                    x if x is not None else 0.0
                    for x in log_prob_corrs.reshape(sampler.n_batches)
                ],
            )

        else:
            log_prob_corr = None
        

        return σp, log_prob_corr


    def __repr__(self):
        return f"MultipleRules(probabilities={self.probabilities}, rules={self.rules})"
