"""
Bayesian Hidden State learner for port value learning.

Combines HiddenStatePortLearner (belief over which permutation of reward probabilities is assigned
to which ports) with BayesianPortLearner (Beta posterior over the actual probability of each slot).

After each trial:
  1. The belief over permutations is updated via Bayes' rule using the current slot Beta means.
  2. Each slot's Beta posterior is soft-updated proportional to the belief mass assigning
     the visited port to that slot — rather than committing to one assignment, the observation
     is spread across slots weighted by current belief.

Port values are the belief-weighted average of slot Beta posterior means, capturing both
structural uncertainty (which permutation?) and parametric uncertainty (what are the probabilities?).

The prior Beta(prior_strength * slot_prob, prior_strength * (1 - slot_prob)) controls how
flexible the learned slot probabilities are. Larger prior_strength makes it harder to shift
slot probability estimates (analogous to a decreased learning rate).
"""

import numpy as np
from itertools import permutations
from scipy import stats
from scipy.optimize import minimize

from ...utils import REWARD_PORTS, resolve_port


class BayesianHiddenStatePortLearner:
    """
    Bayesian hidden state model for port reward learning.

    Extends HiddenStatePortLearner by treating reward probabilities as unknown.
    Instead of fixed values, each slot (high/mid/low) has a Beta posterior
    that is updated via soft assignment based on the current structural belief.

    Update rule (each trial):
      1. Transition: optionally mix belief toward uniform (models possible swaps).
      2. Structural update: Bayesian update over permutations using current
         slot Beta means as likelihoods.
      3. Soft Beta update: update each slot's Beta proportional to the belief
         mass on states where the visited port occupies that slot.

    Port values are the belief-weighted average of slot Beta posterior means.

    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(
        self,
        slot_probs=(0.9, 0.5, 0.1),
        prior_strength=2.0,
        transition_prob=0.0,
    ):
        """
        Parameters:
            slot_probs (tuple of float): Prior means for each slot's reward probability,
                in descending order. Each becomes the mean of a Beta prior:
                Beta(prior_strength * mean, prior_strength * (1 - mean)).
            prior_strength (float): Total pseudocount for each slot's Beta prior (pseudo-counts
                of rewards and omissions). Higher values make the prior stronger and slot
                probability estimates harder to shift (analogous to a decreased learning rate).
                Use >= 1 to avoid U-shaped priors on extreme slot values.
            transition_prob (float): Per-trial probability of switching to a different permutation.
                0 = no switching expected. >0 = volatile environment.
        """
        self.slot_probs = tuple(slot_probs)
        self.prior_strength = prior_strength
        self.transition_prob = transition_prob
        n_slots = len(slot_probs)

        # All permutations: states[i][port] = slot index
        # Slot 0 = highest prior mean, slot n-1 = lowest
        self.states = []
        for perm in permutations(range(n_slots)):
            self.states.append({port: perm[k] for k, port in enumerate(REWARD_PORTS)})

        n_states = len(self.states)
        self.belief = np.ones(n_states) / n_states  # uniform prior over permutations

        # Beta posteriors per slot index
        self.slot_betas = {
            s: {
                "a": prior_strength * slot_probs[s],
                "b": prior_strength * (1.0 - slot_probs[s]),
            }
            for s in range(n_slots)
        }
        self.history = []

    def _slot_mean(self, slot_idx):
        """Posterior mean (expected value) of the Beta for a given slot."""
        beta = self.slot_betas[slot_idx]
        return beta["a"] / (beta["a"] + beta["b"])

    def reset(self):
        """Reset beliefs and slot Betas to priors, clear history."""
        n_states = len(self.states)
        self.belief = np.ones(n_states) / n_states
        n_slots = len(self.slot_probs)
        self.slot_betas = {
            s: {
                "a": self.prior_strength * self.slot_probs[s],
                "b": self.prior_strength * (1.0 - self.slot_probs[s]),
            }
            for s in range(n_slots)
        }
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

        # Update belief toward uniform before updating — models the possibility
        # that the reward probabilities have changed since the last trial
        # This is similar to decay in bayesian and rescorla-wagner
        if self.transition_prob > 0:
            n = len(self.states)
            self.belief = (
                (1 - self.transition_prob) * self.belief
                + self.transition_prob * np.ones(n) / n
            )

        # Likelihood of this reward under each state (state=permutation of reward probs).
        # Same as HiddenStatePortLearner, except instead of fixed probabilities we use the
        # current Beta posterior mean for each slot — these get updated as data accumulates.
        # states[i][port] gives the slot index assigned to the visited port in state i,
        # and slot_means[slot] gives the current best estimate of that slot's reward probability.
        # If a reward was received, the likelihood is slot_means[slot]
        # If no reward, the likelihood is 1 - slot_means[slot]
        slot_means = {s: self._slot_mean(s) for s in self.slot_betas}
        likelihoods = np.array([
            slot_means[self.states[i][port]] if reward
            else 1.0 - slot_means[self.states[i][port]]
            for i in range(len(self.states))
        ])

        # Compute surprise before updating so it reflects the pre-update prediction.
        # p_obs is the model's overall predicted probability of the observed reward/omission,
        # averaged across all states weighted by current belief.
        p_obs = np.dot(self.belief, likelihoods)
        surprise = -np.log(max(p_obs, 1e-10))

        # Bayes update: multiply each state's belief by how likely it made this observation,
        # then renormalize so beliefs sum to 1.
        # States that assigned high probability to what actually happened get upweighted.
        # States that assigned low probability get downweighted.
        self.belief *= likelihoods
        self.belief /= self.belief.sum()

        # Soft Beta update: since we don't know which state we're in, we can't hard-assign
        # this observation to a single slot. Instead, compute how much total belief currently
        # assigns the visited port to each slot, and use that as the fractional update weight.
        # e.g. if 80% of belief assigns port A to the high slot, the high slot gets 0.8 of
        # this reward/omission added to its Beta.
        n_slots = len(self.slot_probs)
        slot_weights = np.zeros(n_slots)
        for i, state in enumerate(self.states):
            slot_weights[state[port]] += self.belief[i]

        # Increment reward or omission count for each slot, weighted by belief
        for s in range(n_slots):
            w = slot_weights[s]
            if reward:
                self.slot_betas[s]["a"] += w
            else:
                self.slot_betas[s]["b"] += w

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "belief": self.belief.copy(),
            "slot_betas": {s: dict(v) for s, v in self.slot_betas.items()},
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
            # Do belief + slot Beta update for port visit (returns surprise)
            surprises.append(self.update(port, reward))
        return surprises

    def expected_value(self, port):
        """Return the expected reward probability for a port, averaged across all assignments
        and integrated over slot Beta posteriors."""
        port = resolve_port(port)
        return sum(
            self.belief[i] * self._slot_mean(self.states[i][port])
            for i in range(len(self.states))
        )

    def slot_ci(self, slot_idx, ci=0.95):
        """
        Credible interval for a slot's reward probability Beta posterior.

        Parameters:
            slot_idx (int): Slot index (0 = highest prior mean slot, ...).
            ci (float): Credible interval width (default 0.95).

        Returns:
            tuple of (float, float): (lower, upper) bounds.
        """
        beta = self.slot_betas[slot_idx]
        dist = stats.beta(beta["a"], beta["b"])
        tail = (1 - ci) / 2
        return (dist.ppf(tail), dist.ppf(1 - tail))

    def get_values(self):
        """Return expected reward probabilities as {port: value}."""
        return {port: self.expected_value(port) for port in REWARD_PORTS}

    def get_state_posteriors(self):
        """
        Return belief over each permutation with slot assignments.

        Returns:
            list of dict: Each entry has "assignment" ({port: prior_mean}),
                "slot_indices" ({port: slot_idx}), and "probability".
        """
        return [
            {
                "assignment": {p: self.slot_probs[s] for p, s in state.items()},
                "slot_indices": dict(state),
                "probability": float(self.belief[i]),
            }
            for i, state in enumerate(self.states)
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

        The belief-weighted average of slot Beta means at each trial is used as the
        predicted reward probability, so the likelihood of each trial is
        Bernoulli(expected_value(port)).

        Runs the model from scratch with the current params
        (self.slot_probs, self.prior_strength, self.transition_prob).

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            float: Total negative log-likelihood.
        """
        # Create a fresh instance with the same parameters so we don't modify the current model
        model = BayesianHiddenStatePortLearner(
            slot_probs=self.slot_probs,
            prior_strength=self.prior_strength,
            transition_prob=self.transition_prob,
        )
        total = 0.0
        for port, reward in zip(ports, rewards):
            # Clip away from 0 and 1 so log doesn't blow up (just in case)
            q = np.clip(model.expected_value(port), 1e-10, 1 - 1e-10)
            # Bernoulli log-likelihood of reward at this port given the expected value
            # reward * log(q) + (1-reward) * log(1-q)
            total -= reward * np.log(q) + (1 - reward) * np.log(1 - q)
            # Now update beliefs and slot Betas based on this reward
            model.update(port, reward)
        return total

    @classmethod
    def fit(cls, ports, rewards, slot_probs=(0.9, 0.5, 0.1)):
        """
        Fit prior_strength and transition_prob to maximise the likelihood of a
        reward sequence. slot_probs are treated as fixed and known.

        Returns a fitted instance with best-fit parameters and ``nll_``, ``bic_``,
        and ``result_`` attributes.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.
            slot_probs (tuple of float): The assumed prior means for each slot.
                Defaults to (0.9, 0.5, 0.1).

        Returns:
            BayesianHiddenStatePortLearner: Fitted instance with attributes:
                - nll_    : NLL at optimum
                - bic_    : BIC (2 params)
                - result_ : raw scipy OptimizeResult
        """
        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        # L-BFGS-B respects the bounds without needing a penalty
        def _obj(params):
            prior_strength, transition_prob = params
            return cls(slot_probs=slot_probs,
                       prior_strength=prior_strength,
                       transition_prob=transition_prob).nll(ports, rewards)

        # Starting point: prior_strength=2.0 (weak prior), transition_prob=0.05 (mild volatility)
        # Bounds keep prior_strength in [1, 20] and transition_prob in [0, 0.5]
        result = minimize(_obj, x0=[2.0, 0.05],
                          bounds=[(1.0, 20.0), (0.0, 0.5)],
                          method='L-BFGS-B')

        # Fitted = model with the best fit prior_strength and transition_prob
        fitted = cls(slot_probs=slot_probs,
                     prior_strength=result.x[0],
                     transition_prob=result.x[1])
        fitted.nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (prior_strength, transition_prob)
        n = len(rewards)
        fitted.bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.result_ = result
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
