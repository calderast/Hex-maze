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

from ...utils import REWARD_PORTS, resolve_port
from ..fitting import bic, fit_from_starts


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
        beta=1.0,
        decay=0.05,
    ):
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
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.beta = beta
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
        # surprise = -log(p(observed outcome)), aka Shannon "self information". 
        # (Shannon entropy is the expected surprise across all possible outcomes)
        p_outcome = mean if reward else (1 - mean)
        surprise = -np.log(max(p_outcome, 1e-10))

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

        # Higher beta (inverse temperature) = more likely to choose the highest-value port
        # Lower beta = choices are more random
        q_vals_scaled = self.beta * q_vals

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

    def reward_nll(self, ports, rewards):
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

    def choice_nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of the rat's port CHOICES under
        this model's current parameters.

        On each trial the rat chooses between the 2 ports it didn't just visit
        (first trial: all 3 ports available). The choice probability comes from
        softmax over posterior means with self.beta (inverse temperature).

        This is different from nll(), which scores how well the model predicts
        REWARDS. choice_nll() scores how well the model predicts which PORT
        the rat chose.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.

        Returns:
            float: Total negative log-likelihood of the choice sequence.
        """
        model = BayesianPortLearner(prior_a=self.prior_a, prior_b=self.prior_b,
                                    beta=self.beta, decay=self.decay)
        total = 0.0
        prev_port = None
        for port, reward in zip(ports, rewards):
            # First trial: all 3 ports available. After that: exclude previous port.
            if prev_port is None:
                available = REWARD_PORTS
            else:
                available = [p for p in REWARD_PORTS if p != prev_port]

            # Softmax probability of the rat's actual choice BEFORE updating
            probs = model.choice_probabilities(available_ports=available)
            p_choice = probs[resolve_port(port)]
            total -= np.log(max(p_choice, 1e-10))

            # Update model with this trial's outcome
            model.update(port, reward)
            prev_port = resolve_port(port)
        return total

    @classmethod
    def fit_choices(cls, ports, rewards, n_starts=8, seed=0):
        """
        Fit prior_strength, decay, and beta (inverse temperature) to maximize the likelihood
        of the rat's port choices (not rewards).

        See fit_from_starts (in rl/fitting.py) for how the fit works.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.

        Returns:
            BayesianPortLearner: Fitted instance with attributes:
                - choice_nll_    : choice NLL at optimum
                - choice_bic_    : BIC (3 params: prior_strength, decay, beta)
                - choice_result_ : raw scipy OptimizeResult for the best starting point
                - choice_results_: the OptimizeResult from every starting point
        """
        def _obj(params):
            prior_strength, decay, beta = params
            return cls(prior_a=prior_strength, prior_b=prior_strength,
                       beta=beta,
                       decay=decay).choice_nll(ports, rewards)

        # prior_strength and beta are above 0, and decay is between 0 and 1
        values, result, results = fit_from_starts(
            _obj, start_values=[1.0, 0.05, 1.0], kinds=["positive", "unit", "positive"],
            n_starts=n_starts, seed=seed,
        )

        fitted = cls(prior_a=values[0], prior_b=values[0],
                     beta=values[2], decay=values[1])
        fitted.choice_nll_ = result.fun
        fitted.choice_bic_ = bic(result.fun, len(values), len(rewards))
        fitted.choice_result_ = result
        fitted.choice_results_ = results
        return fitted


    @classmethod
    def fit_rewards(cls, ports, rewards, n_starts=8, seed=0):
        """
        Fit prior_strength and decay to maximise the likelihood of a reward sequence.
        prior_a and prior_b are constrained equal (symmetric prior).

        See fit_from_starts (in rl/fitting.py) for how the fit works.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.

        Returns:
            BayesianPortLearner: Fitted instance with attributes:
                - reward_nll_    : NLL at optimum
                - reward_bic_    : BIC (2 params)
                - reward_result_ : raw scipy OptimizeResult for the best starting point
                - reward_results_: the OptimizeResult from every starting point
        """
        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        def _obj(params):
            prior_strength, decay = params
            return cls(prior_a=prior_strength, prior_b=prior_strength,
                       decay=decay).reward_nll(ports, rewards)

        # First starting point: prior_strength=1.0 (uniform prior), decay=0.05 (mild forgetting)
        # prior_strength is above 0, and decay is between 0 and 1
        values, result, results = fit_from_starts(
            _obj, start_values=[1.0, 0.05], kinds=["positive", "unit"], n_starts=n_starts, seed=seed,
        )

        # Fitted = model with the best fit prior_strength and decay parameters
        fitted = cls(prior_a=values[0], prior_b=values[0], decay=values[1])
        fitted.reward_nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (prior_strength, decay)
        fitted.reward_bic_ = bic(result.fun, len(values), len(rewards))
        fitted.reward_result_ = result
        fitted.reward_results_ = results
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
