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
from scipy.optimize import minimize

from ...utils import REWARD_PORTS, resolve_port


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
        temperature=1.0,
        initial_value=0.5,
        decay=0.05,
    ):
        """
        Parameters:
            alpha (float): Learning rate.
            temperature (float): Softmax temperature for port selection.
            initial_value (float): Starting Q-value for all ports.
            decay (float): Per-trial decay toward initial_value applied before each update.
                Q(port) ← Q(port) * (1 - decay) + initial_value * decay
        """
        self.alpha = alpha
        self.temperature = temperature
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

        # Lower temperature = more likely to choose the port with the highest Q value
        # Higher temperature = choices are more random
        q_vals_scaled = q_vals / self.temperature

        # Subtract the max before exp for numerical stability (doesn't change the output)
        q_vals_scaled -= q_vals_scaled.max()
        exp_v = np.exp(q_vals_scaled)
        # Normalize so choice probabilities sum to 1
        port_choice_probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, port_choice_probs.tolist()))

    def get_values(self):
        """Return a copy of current Q-values as {port: value}."""
        return self.Q.copy()

    def nll(self, ports, rewards):
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

    @classmethod
    def fit(cls, ports, rewards):
        """
        Fit alpha and decay parameters to maximize the likelihood of a reward sequence.

        Uses L-BFGS-B to minimize negative log-likelihood. Returns a new
        fitted instance with best-fit parameters. The fitted instance also
        has ``nll_``, ``bic_``, and ``result_`` attributes.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            RescorlaWagner: Fitted instance with attributes:
                - nll_    : NLL at optimum
                - bic_    : BIC (2 params)
                - result_ : raw scipy OptimizeResult
        """

        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        # L-BFGS-B respects the bounds without needing a penalty
        def _obj(params):
            alpha, decay = params
            return cls(alpha=alpha, decay=decay).nll(ports, rewards)

        # Starting point: alpha=0.3 (moderate learning rate), decay=0.05 (mild forgetting) 
        # Bounds keep alpha in (0, 1] and decay in [0, 0.5]
        result = minimize(_obj, x0=[0.3, 0.05],
                          bounds=[(1e-3, 1.0), (0.0, 0.5)],
                          method='L-BFGS-B')

        # Fitted = model with the best fit alpha and decay parameters
        fitted = cls(alpha=result.x[0], decay=result.x[1])
        fitted.nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (alpha, decay)
        n = len(rewards)
        fitted.bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.result_ = result
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
