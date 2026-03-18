"""
Bayesian (Beta-Binomial) learner for port value learning.

Maintains a Beta(a, b) posterior over each port's reward probability, updated after each port visit.

    reward=1: a += 1   (one more reward)
    reward=0: b += 1   (one more omission)

The expected reward probability is a / (a + b). 

The prior Beta(prior_a, prior_b) encodes initial beliefs. We enforce a = b for a
starting reward probability of 0.5. Larger a and b values make this prior stronger
(need more port visits to change it, analagous to a decreased "learning rate")

An optional decay parameter shrinks (a, b) back toward the prior each trial (modelling forgetting).

The reward probability for each port is learned independently.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ...utils import REWARD_PORTS, resolve_port


class BayesianPortLearner:
    """
    Bayesian learner over reward ports using Beta-Binomial conjugate updates.

    Each port has a Beta(a, b) posterior over its reward probability.

    Update rule:
        reward=1: a += 1
        reward=0: b += 1

    The expected value (mean of the posterior) is a / (a + b).

    Action selection via softmax over posterior means.
    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(
        self,
        prior_a=1.0,
        prior_b=1.0,
        temperature=1.0,
        decay=0.05,
    ):
        """
        Parameters:
            prior_a (float): Initial alpha parameter for Beta prior (pseudo-count of rewards).
                Default 1.0 gives a uniform prior.
            prior_b (float): Initial beta parameter for Beta prior (pseudo-count of omissions).
                Default 1.0 gives a uniform prior.
            temperature (float): Softmax temperature for port selection.
            decay (float): Per-trial decay toward the prior. Each trial:
                a ← a * (1 - decay) + prior_a * decay
                b ← b * (1 - decay) + prior_b * decay
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.temperature = temperature
        self.decay = decay

        self.posteriors = {
            port: {"a": prior_a, "b": prior_b} for port in REWARD_PORTS
        }
        self.history = []

    def reset(self):
        """Reset posteriors to priors and clear history."""
        self.posteriors = {
            port: {"a": self.prior_a, "b": self.prior_b}
            for port in REWARD_PORTS
        }
        self.history = []

    def update(self, port, reward):
        """
        Update the posterior for a port after observing a reward.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int or float): Reward received (0 or 1).

        Returns:
            float: Surprise: -log(p(reward | current posterior)).
        """
        # Handle ports specified as 1/2/3 or A/B/C
        port = resolve_port(port)

        # Decay all posteriors back toward the prior before updating
        if self.decay:
            for p in REWARD_PORTS:
                post = self.posteriors[p]
                post["a"] = post["a"] * (1 - self.decay) + self.prior_a * self.decay
                post["b"] = post["b"] * (1 - self.decay) + self.prior_b * self.decay

        post = self.posteriors[port]
        mean = post["a"] / (post["a"] + post["b"])

        # Compute surprise before updating so it reflects the pre-update prediction
        p_reward = mean if reward else (1 - mean)
        surprise = -np.log(max(p_reward, 1e-10))

        # Conjugate Beta-Binomial update: increment reward or omission count
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

        Parameters:
            ports (list of int or str): Sequence of ports visited (1/2/3 or A/B/C).
            rewards (list of int or float): Reward received at each port (0 or 1).

        Returns:
            list of float: Surprise values for each trial.
        """
        surprises = []
        for port, reward in zip(ports, rewards):
            # Do posterior update for port visit (returns surprise)
            surprises.append(self.update(port, reward))
        return surprises

    def expected_value(self, port):
        """Return the posterior mean (expected reward probability) for a port."""
        port = resolve_port(port)
        post = self.posteriors[port]
        return post["a"] / (post["a"] + post["b"])

    def confidence_interval(self, port, ci=0.95):
        """
        Return the credible interval for a port's reward probability.

        Parameters:
            port (int or str): Which port (1/2/3 or A/B/C).
            ci (float): Credible interval width (default 0.95).

        Returns:
            tuple of (float, float): (lower, upper) bounds.
        """
        port = resolve_port(port)
        post = self.posteriors[port]
        dist = stats.beta(post["a"], post["b"])
        tail = (1 - ci) / 2
        return (dist.ppf(tail), dist.ppf(1 - tail))

    def choice_probabilities(self, available_ports=None):
        """
        Softmax choice probabilities over posterior means.

        Parameters:
            available_ports (list of int or str, optional): Which ports to choose among
                (1/2/3 or A/B/C). Defaults to all 3.

        Returns:
            dict of {int: float}: {port: probability}
        """
        if available_ports is None:
            available_ports = REWARD_PORTS
        else:
            available_ports = [resolve_port(p) for p in available_ports]

        # Get expected reward probability for each port
        q_vals = np.array([self.expected_value(p) for p in available_ports])

        # Lower temperature = more likely to choose the highest-value port
        # Higher temperature = choices are more random
        q_vals_scaled = q_vals / self.temperature

        # Subtract the max before exp for numerical stability (doesn't change the output)
        q_vals_scaled -= q_vals_scaled.max()
        exp_v = np.exp(q_vals_scaled)
        # Normalize so choice probabilities sum to 1
        port_choice_probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, port_choice_probs.tolist()))

    def get_values(self):
        """Return posterior means as {port: expected_value}."""
        return {port: self.expected_value(port) for port in REWARD_PORTS}

    def get_posteriors(self):
        """Return a copy of all posteriors as {port: {"a": float, "b": float}}."""
        return {port: dict(v) for port, v in self.posteriors.items()}

    def nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of a reward sequence under this
        model's current parameters.

        The posterior mean at each trial is used as the predicted reward probability,
        so the likelihood of each trial is Bernoulli(expected_value(port)).

        Runs the model from scratch with the current params
        (self.prior_a, self.prior_b, self.decay).

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            float: Total negative log-likelihood.
        """
        # Create a fresh instance with the same parameters so we don't modify the current model
        model = BayesianPortLearner(prior_a=self.prior_a, prior_b=self.prior_b,
                                    decay=self.decay)
        total = 0.0
        for port, reward in zip(ports, rewards):
            # Clip away from 0 and 1 so log doesn't blow up (just in case)
            q = np.clip(model.expected_value(port), 1e-10, 1 - 1e-10)
            # Bernoulli log-likelihood of reward at this port given the posterior mean
            # reward * log(q) + (1-reward) * log(1-q)
            total -= reward * np.log(q) + (1 - reward) * np.log(1 - q)
            # Now update the posterior based on this reward
            model.update(port, reward)
        return total

    @classmethod
    def fit(cls, ports, rewards):
        """
        Fit prior_strength and decay to maximise the likelihood of a reward sequence.
        prior_a and prior_b are constrained equal (symmetric prior).

        Returns a fitted instance with best-fit parameters and ``nll_``, ``bic_``,
        and ``result_`` attributes.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            BayesianPortLearner: Fitted instance with attributes:
                - nll_    : NLL at optimum
                - bic_    : BIC (2 params)
                - result_ : raw scipy OptimizeResult
        """
        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        # L-BFGS-B respects the bounds without needing a penalty
        def _obj(params):
            prior_strength, decay = params
            return cls(prior_a=prior_strength, prior_b=prior_strength,
                       decay=decay).nll(ports, rewards)

        # Starting point: prior_strength=1.0 (uniform prior), decay=0.05 (mild forgetting)
        # Bounds keep prior_strength in (0, 30] and decay in [0, 0.5]
        result = minimize(_obj, x0=[1.0, 0.05],
                          bounds=[(0.01, 30.0), (0.0, 0.5)],
                          method='L-BFGS-B')

        # Fitted = model with the best fit prior_strength and decay parameters
        fitted = cls(prior_a=result.x[0], prior_b=result.x[0], decay=result.x[1])
        fitted.nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (prior_strength, decay)
        n = len(rewards)
        fitted.bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.result_ = result
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
