"""
Rescorla-Wagner learner for port value learning.

After visiting a port and receiving (or not receiving) a reward, the port's value
is updated toward the outcome by the learning rate alpha.

    Q(port) ← Q(port) + α · (reward − Q(port))

The quantity (reward − Q(port)) is the reward prediction error (RPE).

An optional decay parameter pulls all Q-values back toward the initial value (0.5) on
every trial (modelling forgetting / the rat's expectation that reward probabilities can change).

The Q value (reward expectation) for each port is learned independently.

With per_start_port=True, there's a separate table of Q values for each start port (the
port the rat just left), so what the rat learns on a trip from port A only changes its
choices on later trips from port A (fully model free). The session's first trial doesn't
start at a port, so its outcome goes in every table that has the reached port as an option.
"""

import copy

from ...utils import REWARD_PORTS, resolve_port
from .base import PortLearner


class RescorlaWagner(PortLearner):
    """
    Rescorla-Wagner learner over reward ports.

    Update rule:
        Q(port) ← Q(port) + α · [reward - Q(port)]

    Action selection via softmax over port Q-values.
    """

    FIT_PARAMS = {"alpha": (0.3, "unit"), "decay": (0.05, "unit")}

    def __init__(self, alpha=0.3, beta=1.0, initial_value=0.5, decay=0.05, per_start_port=False):
        """
        Parameters:
            alpha (float): Learning rate.
            beta (float): Softmax inverse temperature for port selection.
            initial_value (float): Starting Q-value for all ports.
            decay (float): Per-trial decay toward initial_value applied before each update.
                Q(port) ← Q(port) * (1 - decay) + initial_value * decay
            per_start_port (bool): If True, keep a separate table of Q values for each start
                port, so Q is {start port: {port: value}} (see the module docstring). If False
                (default), one shared table, so Q is {port: value}.
        """
        self.alpha = alpha
        self.beta = beta
        self.initial_value = initial_value
        self.decay = decay
        self.per_start_port = per_start_port
        self.reset()

    def reset(self):
        """Reset Q-values and history."""
        if self.per_start_port:
            # Q[start port][port]
            self.Q = {start: {port: self.initial_value for port in REWARD_PORTS} for start in REWARD_PORTS}
        else:
            self.Q = {port: self.initial_value for port in REWARD_PORTS}
        self.history = []

    # Helpers for per_start_port. Without it, there's just one table (self.Q), and these just give that

    def _tables(self):
        """Get every table of Q values."""
        return list(self.Q.values()) if self.per_start_port else [self.Q]

    def _table(self, start_port):
        """Get the table of Q values for a trip from start_port ({port: value})."""
        if not self.per_start_port:
            return self.Q
        # The first trial doesn't start at a port. Every table is still the same then, so use the first
        return self.Q[REWARD_PORTS[0] if start_port is None else resolve_port(start_port)]

    def _tables_to_update(self, start_port, port):
        """Get the table(s) of Q values a trip from start_port to port updates."""
        if not self.per_start_port:
            return [self.Q]
        # The first trial goes in every table that has this port as an option
        if start_port is None:
            return [self.Q[start] for start in REWARD_PORTS if start != port]
        return [self.Q[resolve_port(start_port)]]

    def expected_value(self, port, start_port=None):
        """Get a port's Q value (on a trip from start_port, with per_start_port)."""
        return self._table(start_port)[resolve_port(port)]

    def update(self, port, reward, start_port=None):
        """
        Update Q-value for a port after observing a reward.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int): Reward received (0 or 1).
            start_port (int or str, optional): The port the trip started from (None for the
                session's first trial). Only used with per_start_port.

        Returns:
            float: Prediction error (reward - Q(port)) before the update.
        """
        # Use 1/2/3 so we can look the port up in the table(s)
        port = resolve_port(port)

        # Decay all Q-values back toward the initial value before updating
        for table in self._tables():
            for p in REWARD_PORTS:
                table[p] = table[p] * (1 - self.decay) + self.initial_value * self.decay

        # Reward prediction error
        prediction_error = reward - self.expected_value(port, start_port)

        # Update the Q value of the visited port (in the start port's table, with per_start_port)
        for table in self._tables_to_update(start_port, port):
            table[port] += self.alpha * (reward - table[port])

        self.history.append({
            "port": port,
            "reward": reward,
            "prediction_error": prediction_error,
            "Q": copy.deepcopy(self.Q),
        })
        return prediction_error
