"""
Hidden state learner for port value learning.

Assumes the reward probabilities are a known set (e.g., [0.9, 0.5, 0.1]) but their assignment
to ports is unknown. Maintains a belief distribution over all 3! = 6 possible assignments (states)
and updates via Bayes' rule after each observation.

    belief[i]  ∝  belief[i] × P(reward | assignment i)

Unlike Rescorla-Wagner and Bayesian, ports are not learned independently (e.g. a reward at port
A also gives information about ports B and C under the assumption that there is one high,
one medium, and one low probability port)

An optional decay parameter allows the model to expect that assignments can change
(e.g., a probability change session). Decay blends the probabilities of being in each state back
towards uniform (this can also be thought of as a state "transition probability")

An optional alpha parameter controls how strongly each observation updates the belief.
alpha=1 is the full Bayesian update; alpha<1 blends the updated belief back with the current
belief, producing more conservative updates (analogous to the learning rate in Rescorla-Wagner).

Note: the reward_set is treated as fixed and known ([0.9, 0.5, 0.1]).
See BayesianHiddenStatePortLearner for a version that also learns the reward probabilities.
"""

from itertools import permutations

import numpy as np

from ...utils import REWARD_PORTS, resolve_port
from .base import PortLearner


class HiddenStatePortLearner(PortLearner):
    """
    Hidden state model for port reward learning.

    Assumes reward probabilities are a permutation of a known set
    (e.g., [0.9, 0.5, 0.1]) assigned to ports 1/2/3 (A/B/C).

    Maintains a posterior over all possible states and updates
    via Bayes' rule after each observation. Optionally allows for
    decay (blend the probabilities of being in each state back towards
    uniform) at each trial.

    A learning rate alpha blends the Bayes-updated belief with the current
    belief: belief = (1-alpha)*belief + alpha*bayes_updated. alpha=1 is the
    standard full Bayesian update; alpha<1 makes each update more conservative.

    There's one belief for every start port, so the start_port arguments aren't used.
    """

    # alpha starts at 0.9 (almost a full Bayesian update), since a parameter fit as its logit
    # can't start at exactly 1
    FIT_PARAMS = {"alpha": (0.9, "unit"), "decay": (0.05, "unit")}

    def __init__(self, reward_set=(0.9, 0.5, 0.1), decay=0.0, alpha=1.0, beta=1.0):
        """
        Parameters:
            reward_set (tuple of float): The set of reward probabilities assigned across ports.
            decay (float): Per-trial probability of switching to a different permutation.
                0 = no switching expected. >0 = changing environment.
            alpha (float): Learning rate for belief updates (0, 1].
                1.0 = full Bayesian update. <1.0 = blend updated belief with current belief,
                producing more conservative updates (analogous to alpha in Rescorla-Wagner).
            beta (float): Softmax inverse temperature for port selection.
        """
        self.reward_set = reward_set
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # All possible assignments: each is a dict {1: p, 2: p, 3: p}
        # e.g. {1: 0.9, 2: 0.5, 3: 0.1}, {1: 0.9, 2: 0.1, 3: 0.5}, ...
        self.states = [dict(zip(REWARD_PORTS, perm)) for perm in permutations(reward_set)]
        self.reset()

    def reset(self):
        """Reset beliefs to uniform (1/6 chance of being in each state) and clear history."""
        self.belief = np.ones(len(self.states)) / len(self.states)
        self.history = []

    def update(self, port, reward, start_port=None):
        """
        Update beliefs after observing reward at a port.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int): Reward received (0 or 1).
            start_port: Not used (there's one belief for every start port).

        Returns:
            float: Surprise: -log(p(reward | current beliefs)).
        """
        # Use 1/2/3 so we can look the port up in each state
        port = resolve_port(port)

        # Blend belief (what state we think we are in) toward uniform before updating
        # Models the possibility that the reward probabilities have changed since the last trial
        # This is essentially the same as decay in bayesian and rescorla-wagner
        # (uniform belief is 1 / number of states for every state)
        self.belief = (1 - self.decay) * self.belief + self.decay / len(self.states)

        # Likelihood of this reward under each state (state=permutation of reward probs)
        # Each state s defines the reward probabilities of each port as s[port]
        # If a reward was received, the likelihood is s[port]
        # If no reward, the likelihood is 1 - s[port]
        # e.g. if state s assigns p=0.9 to port A: reward likelihood = 0.9, omission likelihood = 0.1
        likelihoods = np.array([s[port] if reward else 1 - s[port] for s in self.states])

        # Compute surprise before updating so it reflects the pre-update prediction.
        # The model's overall predicted probability of the observed reward/omission is
        # averaged across all states, weighted by current belief
        # surprise = -log(p(observed outcome)), aka Shannon "self information".
        # (Shannon entropy is the expected surprise across all possible outcomes)
        p_outcome = np.dot(self.belief, likelihoods)
        surprise = -np.log(max(p_outcome, 1e-10))

        # Bayes update: multiply each state's belief by how likely it made this observation,
        # then renormalize so beliefs sum to 1.
        # States that assigned high probability to what actually happened get upweighted.
        # States that assigned low probability get downweighted.
        # e.g. if port A was rewarded: states where A is the high-prob port (0.9) get upweighted,
        # states where A is the low-prob port (0.1) get downweighted.
        updated = self.belief * likelihoods
        updated /= updated.sum()

        # Learning rate: alpha=1 applies the full Bayesian update; alpha<1 blends the
        # Bayes-updated belief with the current belief
        self.belief = (1 - self.alpha) * self.belief + self.alpha * updated

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "belief": self.belief.copy(),
            "expected_values": self.get_values(),
            "expected_value_stds": self.get_stds(),
        })
        return surprise

    def expected_value(self, port, start_port=None):
        """Get the expected reward probability for a port, averaged across all states
        (start_port isn't used)."""
        port = resolve_port(port)
        # sum of p(reward) in each state * likelihood of being in that state
        return float(np.dot(self.belief, [s[port] for s in self.states]))

    def expected_value_std(self, port):
        """
        Standard deviation of the expected reward for a port, based on the spread of
        the belief distribution over different states. Reflects how much the possible
        states disagree about a port's reward probability.
        """
        port = resolve_port(port)
        # Var[X] = Σ P(X=x) · (x − μ)^2, where X is this port's reward probability, which is
        # states[i][port] with probability belief[i], and μ is its expected value
        mean = self.expected_value(port)
        variance = np.dot(self.belief, [(s[port] - mean) ** 2 for s in self.states])
        return float(np.sqrt(variance))

    def get_stds(self):
        """Get the standard deviation of expected reward for each port as {port: std}."""
        return {port: self.expected_value_std(port) for port in REWARD_PORTS}

    def get_state_posteriors(self):
        """
        Get the belief over each state.

        Returns:
            list of dict: Each entry has "assignment" ({1: p, 2: p, 3: p}) and "probability".
        """
        return [{"assignment": dict(s), "probability": float(b)} for s, b in zip(self.states, self.belief)]
