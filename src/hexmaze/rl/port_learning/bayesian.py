"""
Bayesian (Beta-Binomial) learner for port-level reward learning.

Maintains a Beta(a, b) posterior for each port's reward probability.
Updates beliefs after each reward/no-reward observation.
"""

import numpy as np
from scipy import stats


class BayesianPortLearner:
    """
    Bayesian learner over reward ports using Beta-Binomial conjugate updates.

    Each port has a Beta(a, b) posterior over its reward probability.

    Update rule:
        reward=1: a += 1
        reward=0: b += 1

    The expected value (mean of the posterior) is a / (a + b).

    Action selection via softmax over posterior means, or Thompson sampling.
    """

    REWARD_PORTS = [1, 2, 3]

    def __init__(
        self,
        prior_a=1.0,
        prior_b=1.0,
        temperature=1.0,
        decay=0.0,
    ):
        """
        Parameters
        ----------
        prior_a : float
            Initial alpha parameter for Beta prior (pseudo-count of successes).
            Default 1.0 gives a uniform prior.
        prior_b : float
            Initial beta parameter for Beta prior (pseudo-count of failures).
            Default 1.0 gives a uniform prior.
        temperature : float
            Softmax temperature for port selection (used in choice_probabilities).
        decay : float
            Per-trial decay toward the prior. Each trial:
            a ← a * (1 - decay) + prior_a * decay
            b ← b * (1 - decay) + prior_b * decay
            Set to 0 for standard Bayesian updating (no forgetting).
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.temperature = temperature
        self.decay = decay

        self.posteriors = {
            port: {"a": prior_a, "b": prior_b} for port in self.REWARD_PORTS
        }
        self.history = []

    def reset(self):
        """Reset posteriors to priors and clear history."""
        self.posteriors = {
            port: {"a": self.prior_a, "b": self.prior_b}
            for port in self.REWARD_PORTS
        }
        self.history = []

    def update(self, port, reward):
        """
        Update the posterior for a port after observing a reward.

        Parameters
        ----------
        port : int
            Which port was visited (1, 2, or 3).
        reward : float
            Reward received (0.0 or 1.0).

        Returns
        -------
        float
            Surprise: -log(p(reward | current posterior)).
        """
        # Apply decay toward prior
        if self.decay:
            for p in self.REWARD_PORTS:
                post = self.posteriors[p]
                post["a"] = post["a"] * (1 - self.decay) + self.prior_a * self.decay
                post["b"] = post["b"] * (1 - self.decay) + self.prior_b * self.decay

        post = self.posteriors[port]
        mean = post["a"] / (post["a"] + post["b"])

        # Compute surprise before updating
        p_reward = mean if reward else (1 - mean)
        surprise = -np.log(max(p_reward, 1e-10))

        # Conjugate update
        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError(f"reward must be 0 or 1, got {reward}")
        if reward:
            post["a"] += 1
        else:
            post["b"] += 1

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "posteriors": {p: dict(v) for p, v in self.posteriors.items()},
        })

        return surprise

    def learn(self, ports, rewards):
        """
        Run updates on a sequence of port visits and rewards.

        Parameters
        ----------
        ports : list of int
            Sequence of ports visited.
        rewards : list of float
            Reward received at each port.

        Returns
        -------
        list of float
            Surprise values for each trial.
        """
        surprises = []
        for port, reward in zip(ports, rewards):
            surprises.append(self.update(port, reward))
        return surprises

    def expected_value(self, port):
        """Return the posterior mean (expected reward probability) for a port."""
        post = self.posteriors[port]
        return post["a"] / (post["a"] + post["b"])

    def confidence_interval(self, port, ci=0.95):
        """
        Return the credible interval for a port's reward probability.

        Parameters
        ----------
        port : int
            Which port (1, 2, or 3).
        ci : float
            Credible interval width (default 0.95).

        Returns
        -------
        tuple of (float, float)
            (lower, upper) bounds.
        """
        post = self.posteriors[port]
        dist = stats.beta(post["a"], post["b"])
        tail = (1 - ci) / 2
        return (dist.ppf(tail), dist.ppf(1 - tail))

    def sample(self, port):
        """Draw a sample from the posterior for a port (for Thompson sampling)."""
        post = self.posteriors[port]
        return np.random.beta(post["a"], post["b"])

    def thompson_choice(self, available_ports=None):
        """
        Choose a port via Thompson sampling.

        Draws a sample from each port's posterior and picks the highest.

        Parameters
        ----------
        available_ports : list of int, optional
            Which ports to choose among. Defaults to all 3.

        Returns
        -------
        int
            The chosen port.
        """
        if available_ports is None:
            available_ports = self.REWARD_PORTS
        samples = {port: self.sample(port) for port in available_ports}
        return max(samples, key=samples.get)

    def choice_probabilities(self, available_ports=None):
        """
        Softmax choice probabilities over posterior means.

        Parameters
        ----------
        available_ports : list of int, optional
            Which ports to choose among. Defaults to all 3.

        Returns
        -------
        dict of {int: float}
            {port: probability}
        """
        if available_ports is None:
            available_ports = self.REWARD_PORTS
        vals = np.array([self.expected_value(p) for p in available_ports])
        scaled = vals / self.temperature
        scaled -= scaled.max()
        exp_v = np.exp(scaled)
        probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, probs.tolist()))

    def get_values(self):
        """Return posterior means as {port: expected_value}."""
        return {port: self.expected_value(port) for port in self.REWARD_PORTS}

    def get_posteriors(self):
        """Return a copy of all posteriors as {port: {"a": float, "b": float}}."""
        return {port: dict(v) for port, v in self.posteriors.items()}

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
