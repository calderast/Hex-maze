"""
Bayesian (Beta-Binomial) learner for port value learning.

Maintains a Beta(a, b) posterior over each port's reward probability, updated after each port visit.

    reward=1: a += 1   (one more reward)
    reward=0: b += 1   (one more omission)

The expected reward probability is a / (a + b).

The prior Beta(prior_a, prior_b) encodes initial beliefs. When fitting, we set a = b
(prior_strength) for a starting reward probability of 0.5. Larger a and b values make this
prior stronger (need more port visits to change it, analogous to a decreased "learning rate")

An optional decay parameter shrinks (a, b) back toward the prior each trial (modelling forgetting).

The reward probability for each port is learned independently.

With per_start_port=True, there's a separate set of posteriors for each start port (the
port the rat just left), so what the rat learns on a trip from port A only changes its
choices on later trips from port A (fully model free). The session's first trial doesn't
start at a port, so its outcome goes in every set that has the reached port as an option.
"""

import copy

import numpy as np
from scipy import stats

from ...utils import REWARD_PORTS, resolve_port
from .base import PortLearner


class BayesianPortLearner(PortLearner):
    """
    Bayesian learner over reward ports using Beta-Binomial conjugate updates.

    Each port has a Beta(a, b) posterior over its reward probability.

    Update rule:
        reward=1: a += 1
        reward=0: b += 1

    The expected value (mean of the posterior) is a / (a + b).

    Action selection via softmax over posterior means.
    """

    # prior_strength sets both prior_a and prior_b (see _with_params)
    FIT_PARAMS = {"prior_strength": (1.0, "positive"), "decay": (0.05, "unit")}

    def __init__(self, prior_a=1.0, prior_b=1.0, beta=1.0, decay=0.05, per_start_port=False):
        """
        Parameters:
            prior_a (float): Initial alpha parameter for Beta prior (pseudo-count of rewards).
                Default 1.0 gives a uniform prior.
            prior_b (float): Initial beta parameter for Beta prior (pseudo-count of omissions).
                Default 1.0 gives a uniform prior.
            beta (float): Softmax inverse temperature for port selection (not to be confused
                with the Beta prior).
            decay (float): Per-trial decay toward the prior. Each trial:
                a ← a * (1 - decay) + prior_a * decay
                b ← b * (1 - decay) + prior_b * decay
            per_start_port (bool): If True, keep a separate set of posteriors for each start
                port, so posteriors is {start port: {port: {"a", "b"}}} (see the module
                docstring). If False (default), one shared set, so posteriors is
                {port: {"a", "b"}}.
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.beta = beta
        self.decay = decay
        self.per_start_port = per_start_port
        self.reset()

    # The fits make learners with _with_params, so fit prior_strength by setting both prior_a and prior_b
    @classmethod
    def _with_params(cls, params, settings):
        """Make a learner with these fitted parameters, where prior_strength sets both prior_a
        and prior_b (a symmetric prior, so every port starts at 0.5)."""
        params = dict(params)
        prior_strength = params.pop("prior_strength")
        return cls(prior_a=prior_strength, prior_b=prior_strength, **params, **settings)

    def reset(self):
        """Reset posteriors to priors and clear history."""
        def priors():
            return {port: {"a": self.prior_a, "b": self.prior_b} for port in REWARD_PORTS}

        # posteriors[start port][port] with per_start_port, otherwise posteriors[port]
        self.posteriors = {start: priors() for start in REWARD_PORTS} if self.per_start_port else priors()
        self.history = []

    # Helpers for per_start_port. Without it, there's just one set (self.posteriors), and these just give that

    def _tables(self):
        """Get every set of posteriors."""
        return list(self.posteriors.values()) if self.per_start_port else [self.posteriors]

    def _table(self, start_port):
        """Get the posteriors for a trip from start_port ({port: {"a", "b"}})."""
        if not self.per_start_port:
            return self.posteriors
        # The first trial doesn't start at a port. Every set is still the same then, so use the first
        return self.posteriors[REWARD_PORTS[0] if start_port is None else resolve_port(start_port)]

    def _tables_to_update(self, start_port, port):
        """Get the set(s) of posteriors a trip from start_port to port updates."""
        if not self.per_start_port:
            return [self.posteriors]
        # The first trial goes in every set that has this port as an option
        if start_port is None:
            return [self.posteriors[start] for start in REWARD_PORTS if start != port]
        return [self.posteriors[resolve_port(start_port)]]

    def expected_value(self, port, start_port=None):
        """Get the posterior mean (expected reward probability) for a port (on a trip from
        start_port, with per_start_port)."""
        post = self._table(start_port)[resolve_port(port)]
        return post["a"] / (post["a"] + post["b"])

    def update(self, port, reward, start_port=None):
        """
        Update the posterior for a port after observing a reward.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int): Reward received (0 or 1).
            start_port (int or str, optional): The port the trip started from (None for the
                session's first trial). Only used with per_start_port.

        Returns:
            float: Surprise: -log(p(reward | current posterior)).
        """
        # Use 1/2/3 so we can look the port up in the posteriors
        port = resolve_port(port)

        # Decay all posteriors back toward the prior before updating
        for table in self._tables():
            for post in table.values():
                post["a"] = post["a"] * (1 - self.decay) + self.prior_a * self.decay
                post["b"] = post["b"] * (1 - self.decay) + self.prior_b * self.decay

        # Compute surprise before updating so it reflects the pre-update prediction
        # surprise = -log(p(observed outcome)), aka Shannon "self information".
        # (Shannon entropy is the expected surprise across all possible outcomes)
        mean = self.expected_value(port, start_port)
        surprise = -np.log(max(mean if reward else 1 - mean, 1e-10))

        # Conjugate Beta-Binomial update: a counts rewards and b counts omissions, so add 1 to one of them
        for table in self._tables_to_update(start_port, port):
            table[port]["a" if reward else "b"] += 1

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "posteriors": copy.deepcopy(self.posteriors),
        })
        return surprise

    def confidence_interval(self, port, ci=0.95, start_port=None):
        """
        Get the credible interval for a port's reward probability.

        Parameters:
            port (int or str): Which port (1/2/3 or A/B/C).
            ci (float): Credible interval width (default 0.95).
            start_port (int or str, optional): The port the trip starts from. Only used with
                per_start_port.

        Returns:
            tuple of (float, float): (lower, upper) bounds.
        """
        post = self._table(start_port)[resolve_port(port)]
        dist = stats.beta(post["a"], post["b"])
        tail = (1 - ci) / 2
        return (dist.ppf(tail), dist.ppf(1 - tail))

    def get_posteriors(self):
        """Get a copy of all posteriors as {port: {"a": float, "b": float}} (or with
        per_start_port, {start port: {port: {"a": float, "b": float}}})."""
        return copy.deepcopy(self.posteriors)
