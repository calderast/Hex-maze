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

from itertools import permutations

import numpy as np
from scipy import stats

from ...utils import REWARD_PORTS, resolve_port
from .base import PortLearner


class BayesianHiddenStatePortLearner(PortLearner):
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

    There's one belief for every start port, so the start_port arguments aren't used.
    """

    FIT_PARAMS = {"prior_strength": (2.0, "positive"), "decay": (0.05, "unit")}

    def __init__(self, slot_probs=(0.9, 0.5, 0.1), prior_strength=2.0, decay=0.0, beta=1.0):
        """
        Parameters:
            slot_probs (tuple of float): Prior means for each slot's reward probability,
                in descending order. Each becomes the mean of a Beta prior:
                Beta(prior_strength * mean, prior_strength * (1 - mean)).
            prior_strength (float): Total pseudocount for each slot's Beta prior (pseudo-counts
                of rewards and omissions). Higher values make the prior stronger and slot
                probability estimates harder to shift (analogous to a decreased learning rate).
                Use >= 1 to avoid U-shaped priors on extreme slot values.
            decay (float): Per-trial probability of switching to a different permutation.
                0 = no switching expected. >0 = volatile environment.
            beta (float): Softmax inverse temperature for port selection.
        """
        self.slot_probs = tuple(slot_probs)
        self.prior_strength = prior_strength
        self.decay = decay
        self.beta = beta

        # All permutations: states[i][port] = which slot that port has in state i
        # (slot 0 = highest prior mean, e.g. 0.9, and slot 2 = lowest, e.g. 0.1)
        self.states = [dict(zip(REWARD_PORTS, perm)) for perm in permutations(range(len(slot_probs)))]
        self.reset()

    def reset(self):
        """Reset beliefs (to uniform over permutations) and slot Betas (to their priors), and clear history."""
        self.belief = np.ones(len(self.states)) / len(self.states)
        # Each slot starts at Beta(prior_strength * p, prior_strength * (1 - p)), which has mean p
        self.slot_betas = {
            slot: {"a": self.prior_strength * p, "b": self.prior_strength * (1 - p)}
            for slot, p in enumerate(self.slot_probs)
        }
        self.history = []

    def _slot_mean(self, slot):
        """Posterior mean (expected value) of the Beta for a given slot."""
        beta = self.slot_betas[slot]
        return beta["a"] / (beta["a"] + beta["b"])

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

        # Blend belief toward uniform before updating — models the possibility
        # that the reward probabilities have changed since the last trial
        # This is similar to decay in bayesian and rescorla-wagner
        # (uniform belief is 1 / number of states for every state)
        self.belief = (1 - self.decay) * self.belief + self.decay / len(self.states)

        # Likelihood of this reward under each state (state=permutation of reward probs).
        # Same as HiddenStatePortLearner, except instead of fixed probabilities we use the
        # current Beta posterior mean for each slot — these get updated as data accumulates.
        # state[port] gives the slot assigned to the visited port in that state.
        # If a reward was received, the likelihood is the slot's mean
        # If no reward, the likelihood is 1 - the slot's mean
        slot_means = [self._slot_mean(state[port]) for state in self.states]
        likelihoods = np.array([mean if reward else 1 - mean for mean in slot_means])

        # Compute surprise before updating so it reflects the pre-update prediction.
        # The model's overall predicted probability of the observed reward/omission is
        # averaged across all states, weighted by current belief.
        p_outcome = np.dot(self.belief, likelihoods)
        surprise = -np.log(max(p_outcome, 1e-10))

        # Bayes update: multiply each state's belief by how likely it made this observation,
        # then renormalize so beliefs sum to 1.
        # States that assigned high probability to what actually happened get upweighted.
        # States that assigned low probability get downweighted.
        self.belief = self.belief * likelihoods
        self.belief /= self.belief.sum()

        # Soft Beta update: since we don't know which state we're in, we can't hard-assign
        # this observation to a single slot. Instead, compute how much total belief currently
        # assigns the visited port to each slot, and use that as the fractional update weight.
        # e.g. if 80% of belief assigns port A to the high slot, the high slot gets 0.8 of
        # this reward/omission added to its Beta (a counts rewards and b counts omissions).
        for state, belief in zip(self.states, self.belief):
            self.slot_betas[state[port]]["a" if reward else "b"] += belief

        self.history.append({
            "port": port,
            "reward": reward,
            "surprise": surprise,
            "belief": self.belief.copy(),
            "slot_betas": {slot: dict(beta) for slot, beta in self.slot_betas.items()},
            "expected_values": self.get_values(),
        })
        return surprise

    def expected_value(self, port, start_port=None):
        """Get the expected reward probability for a port, averaged across all assignments
        and integrated over slot Beta posteriors (start_port isn't used)."""
        port = resolve_port(port)
        return float(np.dot(self.belief, [self._slot_mean(state[port]) for state in self.states]))

    def slot_ci(self, slot, ci=0.95):
        """
        Get the credible interval for a slot's reward probability Beta posterior.

        Parameters:
            slot (int): Slot index (0 = highest prior mean slot, ...).
            ci (float): Credible interval width (default 0.95).

        Returns:
            tuple of (float, float): (lower, upper) bounds.
        """
        beta = self.slot_betas[slot]
        dist = stats.beta(beta["a"], beta["b"])
        tail = (1 - ci) / 2
        return (dist.ppf(tail), dist.ppf(1 - tail))

    def get_state_posteriors(self):
        """
        Get the belief over each permutation with slot assignments.

        Returns:
            list of dict: Each entry has "assignment" ({port: prior_mean}),
                "slot_indices" ({port: slot}), and "probability".
        """
        return [
            {
                "assignment": {port: self.slot_probs[slot] for port, slot in state.items()},
                "slot_indices": dict(state),
                "probability": float(belief),
            }
            for state, belief in zip(self.states, self.belief)
        ]
