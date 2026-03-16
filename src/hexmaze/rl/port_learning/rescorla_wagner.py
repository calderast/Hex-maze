"""
Rescorla-Wagner (delta rule) learner for port-level reward learning.

Learns Q(port) — the expected value of each reward port from reward outcomes.
"""

import numpy as np

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
        initial_value=0.0,
        decay=0.0,
    ):
        """
        Parameters
        ----------
        alpha : float
            Learning rate.
        temperature : float
            Softmax temperature for port selection.
        initial_value : float
            Starting Q-value for all ports.
        decay : float
            Per-trial decay toward initial_value applied before each update.
            Q(port) ← Q(port) * (1 - decay) + initial_value * decay.
            Set to 0 for standard R-W.
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

        Parameters
        ----------
        port : int or str
            Which port was visited (1/2/3 or A/B/C).
        reward : float
            Reward received (typically 0.0 or 1.0).

        Returns
        -------
        float
            Prediction error (reward - Q(port)) before the update.
        """
        port = resolve_port(port)

        # Apply decay toward initial value
        if self.decay:
            for p in REWARD_PORTS:
                self.Q[p] = self.Q[p] * (1 - self.decay) + self.initial_value * self.decay

        prediction_error = reward - self.Q[port]
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

        Parameters
        ----------
        ports : list of int or str
            Sequence of ports visited (1/2/3 or A/B/C).
        rewards : list of float
            Reward received at each port.

        Returns
        -------
        list of float
            Prediction errors for each trial.
        """
        errors = []
        for port, reward in zip(ports, rewards):
            errors.append(self.update(port, reward))
        return errors

    def choice_probabilities(self, available_ports=None):
        """
        Softmax choice probabilities over ports.

        Parameters
        ----------
        available_ports : list of int or str, optional
            Which ports to choose among (1/2/3 or A/B/C). Defaults to all 3.

        Returns
        -------
        dict of {int: float}
            {port: probability}
        """
        if available_ports is None:
            available_ports = REWARD_PORTS
        else:
            available_ports = [resolve_port(p) for p in available_ports]
        vals = np.array([self.Q[p] for p in available_ports])
        scaled = vals / self.temperature
        scaled -= scaled.max()
        exp_v = np.exp(scaled)
        probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, probs.tolist()))

    def get_values(self):
        """Return a copy of current Q-values as {port: value}."""
        return self.Q.copy()

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
