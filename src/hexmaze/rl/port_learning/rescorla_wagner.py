"""
Rescorla-Wagner learner for port value learning.

After visiting a port and receiving (or not receiving) a reward, the port's value
is updated toward the outcome by the learning rate alpha.

    Q(port) ← Q(port) + α · (reward − Q(port))

The quantity (reward − Q(port)) is the reward prediction error (RPE).

An optional decay parameter pulls all Q-values back toward the initial value (0.5) on
every trial (modelling forgetting / the rat's expectation that reward probabilities can change).

The Q value (reward expectation) for each port is learned independently.
"""

import numpy as np

from ...utils import REWARD_PORTS, resolve_port
from ..fitting import bic, fit_from_starts


class RescorlaWagner:
    """
    Rescorla-Wagner learner over reward ports.

    Update rule:
        Q(port) ← Q(port) + α · [reward - Q(port)]

    Action selection via softmax over port Q-values.
    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(
        self,
        alpha=0.3,
        beta=1.0,
        initial_value=0.5,
        decay=0.05,
    ):
        """
        Parameters:
            alpha (float): Learning rate.
            beta (float): Softmax inverse temperature for port selection.
            initial_value (float): Starting Q-value for all ports.
            decay (float): Per-trial decay toward initial_value applied before each update.
                Q(port) ← Q(port) * (1 - decay) + initial_value * decay
        """
        self.alpha = alpha
        self.beta = beta
        self.initial_value = initial_value
        self.decay = decay

        self.Q = {port: initial_value for port in REWARD_PORTS}
        self.history = []

    def reset(self):
        """Reset Q-values and history."""
        self.Q = {port: self.initial_value for port in REWARD_PORTS}
        self.history = []

    def update(self, port, reward):
        """
        Update Q-value for a port after observing a reward.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int): Reward received (0 or 1).

        Returns:
            float: Prediction error (reward - Q(port)) before the update.
        """
        # Handle ports specified as 1/2/3 or A/B/C
        port = resolve_port(port)

        # Decay all Q-values back toward the initial value before updating
        if self.decay:
            for p in REWARD_PORTS:
                self.Q[p] = self.Q[p] * (1 - self.decay) + self.initial_value * self.decay

        # Reward prediction error
        prediction_error = reward - self.Q[port]

        # Update the Q value of the visited port
        self.Q[port] += self.alpha * prediction_error

        self.history.append({
            "port": port,
            "reward": reward,
            "prediction_error": prediction_error,
            "Q": self.Q.copy(),
        })

        return prediction_error

    def learn(self, ports, rewards):
        """
        Run updates on a sequence of port visits and rewards.

        Parameters:
            ports (list of int or str): Sequence of ports visited (1/2/3 or A/B/C).
            rewards (list of float): Reward received at each port.

        Returns:
            list of float: Prediction errors for each trial.
        """
        errors = []
        for port, reward in zip(ports, rewards):
            # Do Q value update for port visit (returns RPE)
            errors.append(self.update(port, reward))
        return errors

    def choice_probabilities(self, available_ports=None):
        """
        Softmax choice probabilities over ports.

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
        
        # Get Q values for the ports we're choosing between
        q_vals = np.array([self.Q[p] for p in available_ports])

        # Higher beta (inverse temperature) = more likely to choose the port with the highest Q value
        # Lower beta = choices are more random
        q_vals_scaled = self.beta * q_vals

        # Subtract the max before exp for numerical stability (doesn't change the output)
        q_vals_scaled -= q_vals_scaled.max()
        exp_v = np.exp(q_vals_scaled)
        # Normalize so choice probabilities sum to 1
        port_choice_probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, port_choice_probs.tolist()))

    def get_values(self):
        """Return a copy of current Q-values as {port: value}."""
        return self.Q.copy()

    def reward_nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of a reward sequence under this
        model's current parameters.

        Q-values are treated as predicted reward probabilities,
        so the likelihood of each trial is Bernoulli(Q(port)).
        
        Runs the model from scratch with the current params 
        (self.alpha, self.decay, self.initial_value)

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            float: Total negative log-likelihood.
        """
        # Create a fresh instance with the same parameters so we don't modify the current model
        model = RescorlaWagner(alpha=self.alpha, decay=self.decay,
                               initial_value=self.initial_value)
        total = 0.0
        for port, reward in zip(ports, rewards):
            p = resolve_port(port)
            # Clip Q away from 0 and 1 so log doesn't blow up (just in case)
            q = np.clip(model.Q[p], 1e-10, 1 - 1e-10)
            # Bernoulli log-likelihood of reward at this port given the port's Q value
            # reward * log(q) + (1-reward) * log(1-q)
            total -= reward * np.log(q) + (1 - reward) * np.log(1 - q)
            # Now update port Q value based on this reward
            model.update(port, reward)
        return total

    def choice_nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of the rat's port choices under
        this model's current parameters.

        On each trial the rat chooses between the 2 ports it didn't just visit
        (first trial: all 3 ports available). The choice probability comes from
        softmax over Q-values with self.beta (inverse temperature).

        This is different from reward_nll(), which scores how well the model predicts
        reward outcomes. choice_nll() scores how well the model predicts which port
        the rat chose.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.

        Returns:
            float: Total negative log-likelihood of the choice sequence.
        """
        model = RescorlaWagner(alpha=self.alpha, decay=self.decay,
                               initial_value=self.initial_value,
                               beta=self.beta)
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
        Fit alpha, decay, and beta (inverse temperature) to maximize the likelihood of the
        rat's port choices (not rewards).

        See fit_from_starts (in rl/fitting.py) for how the fit works.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.

        Returns:
            RescorlaWagner: Fitted instance with attributes:
                - choice_nll_    : choice NLL at optimum
                - choice_bic_    : BIC (3 params: alpha, decay, beta)
                - choice_result_ : raw scipy OptimizeResult for the best starting point
                - choice_results_: the OptimizeResult from every starting point
        """
        def _obj(params):
            alpha, decay, beta = params
            return cls(alpha=alpha, decay=decay,
                       beta=beta).choice_nll(ports, rewards)

        # alpha and decay are between 0 and 1, and beta is above 0
        values, result, results = fit_from_starts(
            _obj, start_values=[0.3, 0.05, 1.0], kinds=["unit", "unit", "positive"],
            n_starts=n_starts, seed=seed,
        )

        fitted = cls(alpha=values[0], decay=values[1],
                     beta=values[2])
        fitted.choice_nll_ = result.fun
        fitted.choice_bic_ = bic(result.fun, len(values), len(rewards))
        fitted.choice_result_ = result
        fitted.choice_results_ = results
        return fitted


    @classmethod
    def fit_rewards(cls, ports, rewards, n_starts=8, seed=0):
        """
        Fit alpha and decay parameters to maximize the likelihood of a reward sequence.

        See fit_from_starts (in rl/fitting.py) for how the fit works.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.

        Returns:
            RescorlaWagner: Fitted instance with attributes:
                - reward_nll_    : NLL at optimum
                - reward_bic_    : BIC (2 params)
                - reward_result_ : raw scipy OptimizeResult for the best starting point
                - reward_results_: the OptimizeResult from every starting point
        """

        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        def _obj(params):
            alpha, decay = params
            return cls(alpha=alpha, decay=decay).reward_nll(ports, rewards)

        # First starting point: alpha=0.3 (moderate learning rate), decay=0.05 (mild forgetting)
        # Both are between 0 and 1
        values, result, results = fit_from_starts(
            _obj, start_values=[0.3, 0.05], kinds=["unit", "unit"], n_starts=n_starts, seed=seed,
        )

        # Fitted = model with the best fit alpha and decay parameters
        fitted = cls(alpha=values[0], decay=values[1])
        fitted.reward_nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (alpha, decay)
        fitted.reward_bic_ = bic(result.fun, len(values), len(rewards))
        fitted.reward_result_ = result
        fitted.reward_results_ = results
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
