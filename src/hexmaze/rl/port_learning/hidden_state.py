"""
Hidden state learner for port value learning.

Assumes the reward probabilities are a known set (e.g., [0.9, 0.5, 0.1]) but their assignment
to ports is unknown. Maintains a belief distribution over all 3! = 6 possible assignments (states)
and updates via Bayes' rule after each observation.

    belief[i]  ∝  belief[i] × P(reward | assignment i)

Unlike Rescorla-Wagner and Bayesian, ports are not learned independently (e.g. a reward at port 
A also gives information about ports B and C under the assumption that there is one high, 
one medium, and one low probability port)

An optional transition_prob parameter allows the model to expect that assignments can change
(e.g., a probability change session)

Note: the reward_set is treated as fixed and known ([0.9, 0.5, 0.1]).
See BayesianHiddenStatePortLearner for a version that also learns the reward probabilities.
"""

import numpy as np
from itertools import permutations
from scipy.optimize import minimize

from ...utils import REWARD_PORTS, resolve_port


class HiddenStatePortLearner:
    """
    Hidden state model for port reward learning.

    Assumes reward probabilities are a permutation of a known set
    (e.g., [0.9, 0.5, 0.1]) assigned to ports 1/2/3 (A/B/C).

    Maintains a posterior over all possible assignments and updates
    via Bayes' rule after each observation. Optionally allows for
    state transitions (probability change) at each trial.

    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(self, reward_set=(0.9, 0.5, 0.1), transition_prob=0.0):
        """
        Parameters:
            reward_set (tuple of float): The set of reward probabilities assigned across ports.
            transition_prob (float): Per-trial probability of switching to a different permutation.
                0 = no switching expected. >0 = changing environment.
        """
        self.reward_set = reward_set
        self.transition_prob = transition_prob

        # All possible assignments: each is a dict {1: p, 2: p, 3: p}
        self.states = []
        for perm in permutations(reward_set):
            self.states.append({1: perm[0], 2: perm[1], 3: perm[2]})

        n = len(self.states)
        self.belief = np.ones(n) / n  # uniform prior over states
        self.history = []

    def reset(self):
        """Reset beliefs to uniform prior and clear history."""
        self.belief = np.ones(len(self.states)) / len(self.states)
        self.history = []

    def update(self, port, reward):
        """
        Update beliefs after observing reward at a port.

        Parameters:
            port (int or str): Which port was visited (1/2/3 or A/B/C).
            reward (int or float): Reward received (0 or 1).

        Returns:
            float: Surprise: -log(p(reward | current beliefs)).
        """
        # Handle ports specified as 1/2/3 or A/B/C
        port = resolve_port(port)

        if reward not in (0, 1, 0.0, 1.0):
            raise ValueError(f"reward must be 0 or 1, got {reward}")

        # Update belief (what state we think we are in) toward uniform before updating
        # Models the possibility that the reward probabilities have changed since the last trial
        # This is similar to decay in bayesian and rescorla-wagner
        if self.transition_prob > 0:
            n = len(self.states)
            self.belief = (1 - self.transition_prob) * self.belief + \
                          self.transition_prob * np.ones(n) / n

        # Likelihood of this reward under each state (state=permutation of reward probs)
        # Each state s defines the reward probabilites of each port as s[port]
        # If a reward was received, the likelihood is s[port]
        # If no reward, the likelihood is 1 - s[port]
        # e.g. if state s assigns p=0.9 to port A: reward likelihood = 0.9, omission likelihood = 0.1
        likelihoods = np.array([
            s[port] if reward else (1 - s[port])
            for s in self.states
        ])

        # Compute surprise before updating so it reflects the pre-update prediction.
        # probability_of_observation is the model's overall predicted probability of the 
        # observed reward/omission, averaged across all states weighted by current belief
        probability_of_observation = np.dot(self.belief, likelihoods)
        surprise = -np.log(max(probability_of_observation, 1e-10))

        # Bayes update: multiply each state's belief by how likely it made this observation,
        # then renormalize so beliefs sum to 1.
        # States that assigned high probability to what actually happened get upweighted.
        # States that assigned low probability get downweighted.
        # e.g. if port A was rewarded: states where A is the high-prob port (0.9) get upweighted,
        # states where A is the low-prob port (0.1) get downweighted.
        self.belief *= likelihoods
        self.belief /= self.belief.sum()

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "belief": self.belief.copy(),
            "expected_values": self.get_values(),
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
            # Do belief update for port visit (returns surprise)
            surprises.append(self.update(port, reward))
        return surprises

    def expected_value(self, port):
        """Return the expected reward probability for a port, averaged across all assignments."""
        port = resolve_port(port)
        # sum of p(reward) in each state * likelihood of being in that state
        return sum(
            self.belief[i] * self.states[i][port]
            for i in range(len(self.states))
        )

    def get_values(self):
        """Return expected reward probabilities as {port: value}."""
        return {port: self.expected_value(port) for port in REWARD_PORTS}

    def get_state_posteriors(self):
        """
        Return belief over each state.

        Returns:
            list of dict: Each entry has "assignment" ({1: p, 2: p, 3: p}) and "probability".
        """
        return [
            {"assignment": dict(s), "probability": float(self.belief[i])}
            for i, s in enumerate(self.states)
        ]

    def choice_probabilities(self, available_ports=None, temperature=1.0):
        """
        Softmax choice probabilities over expected values.

        Parameters:
            available_ports (list of int or str, optional): Which ports to choose among
                (1/2/3 or A/B/C). Defaults to all 3.
            temperature (float): Softmax temperature.

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
        q_vals_scaled = q_vals / temperature

        # Subtract the max before exp for numerical stability (doesn't change the output)
        q_vals_scaled -= q_vals_scaled.max()
        exp_v = np.exp(q_vals_scaled)
        # Normalize so choice probabilities sum to 1
        port_choice_probs = exp_v / exp_v.sum()
        return dict(zip(available_ports, port_choice_probs.tolist()))

    def nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of a reward sequence under this
        model's current parameters.

        The belief-weighted average reward probability at each trial is used as the
        predicted reward probability, so the likelihood of each trial is
        Bernoulli(expected_value(port)).

        Runs the model from scratch with the current params
        (self.reward_set, self.transition_prob).

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            float: Total negative log-likelihood.
        """
        # Create a fresh instance with the same parameters so we don't modify the current model
        model = HiddenStatePortLearner(reward_set=self.reward_set,
                                       transition_prob=self.transition_prob)
        total = 0.0
        for port, reward in zip(ports, rewards):
            # Clip away from 0 and 1 so log doesn't blow up (just in case)
            q = np.clip(model.expected_value(port), 1e-10, 1 - 1e-10)
            # Bernoulli log-likelihood of reward at this port given the expected value
            # reward * log(q) + (1-reward) * log(1-q)
            total -= reward * np.log(q) + (1 - reward) * np.log(1 - q)
            # Now update beliefs based on this reward
            model.update(port, reward)
        return total

    @classmethod
    def fit(cls, ports, rewards, reward_set=(0.9, 0.5, 0.1)):
        """
        Fit transition_prob to maximise the likelihood of a reward sequence.
        The reward_set is treated as fixed and known.

        Returns a fitted instance with best-fit parameters and ``nll_``, ``bic_``,
        and ``result_`` attributes.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.
            reward_set (tuple of float): The assumed set of reward probabilities.
                Defaults to (0.9, 0.5, 0.1).

        Returns:
            HiddenStatePortLearner: Fitted instance with attributes:
                - nll_    : NLL at optimum
                - bic_    : BIC (1 param)
                - result_ : raw scipy OptimizeResult
        """
        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        # L-BFGS-B respects the bounds without needing a penalty
        def _obj(params):
            return cls(reward_set=reward_set,
                       transition_prob=params[0]).nll(ports, rewards)

        # Starting point: transition_prob=0.05 (small change of probability change)
        # Bounds keep transition_prob in [0, 0.5]
        result = minimize(_obj, x0=[0.05],
                          bounds=[(0.0, 0.5)],
                          method='L-BFGS-B')

        # Fitted = model with the best fit transition_prob
        fitted = cls(reward_set=reward_set, transition_prob=result.x[0])
        fitted.nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=1 (transition_prob)
        n = len(rewards)
        fitted.bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.result_ = result
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
