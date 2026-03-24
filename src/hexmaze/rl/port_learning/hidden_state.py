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
towards uniform (this can also be though of as a state "transition probability")

An optional alpha parameter controls how strongly each observation updates the belief.
alpha=1 is the full Bayesian update; alpha<1 blends the updated belief back with the current
belief, producing more conservative updates (analogous to the learning rate in Rescorla-Wagner).

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

    Maintains a posterior over all possible states and updates
    via Bayes' rule after each observation. Optionally allows for
    decay (blend the probabilites of being in each state back towards
    uniform) at each trial.

    A learning rate alpha blends the Bayes-updated belief with the current
    belief: belief = (1-alpha)*belief + alpha*bayes_updated. alpha=1 is the
    standard full Bayesian update; alpha<1 makes each update more conservative.

    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(self, reward_set=(0.9, 0.5, 0.1), decay=0.0, alpha=1.0):
        """
        Parameters:
            reward_set (tuple of float): The set of reward probabilities assigned across ports.
            decay (float): Per-trial probability of switching to a different permutation.
                0 = no switching expected. >0 = changing environment.
            alpha (float): Learning rate for belief updates (0, 1].
                1.0 = full Bayesian update. <1.0 = blend updated belief with current belief,
                producing more conservative updates (analogous to alpha in Rescorla-Wagner).
        """
        self.reward_set = reward_set
        self.decay = decay
        self.alpha = alpha

        # All possible assignments: each is a dict {1: p, 2: p, 3: p}
        self.states = []
        for perm in permutations(reward_set):
            self.states.append({1: perm[0], 2: perm[1], 3: perm[2]})

        n = len(self.states)
        self.belief = np.ones(n) / n  # uniform prior (1/6 chance of being in each state)
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
        # This is essentially the same as decay in bayesian and rescorla-wagner
        if self.decay > 0:
            n = len(self.states)
            self.belief = (1 - self.decay) * self.belief + \
                          self.decay * np.ones(n) / n

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

        # surprise = -log(p(observed outcome)), aka Shannon "self information". 
        # (Shannon entropy is the expected surprise across all possible outcomes)
        surprise = -np.log(max(probability_of_observation, 1e-10))

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
        """Return the expected reward probability for a port, averaged across all states."""
        port = resolve_port(port)
        # sum of p(reward) in each state * likelihood of being in that state
        return sum(
            self.belief[i] * self.states[i][port]
            for i in range(len(self.states))
        )

    def expected_value_std(self, port):
        """
        Standard deviation of the expected reward for a port, based on the spread of
        the belief distribution over different states. Reflects how much the possible 
        states disagree about a port's reward probability.
        """
        port = resolve_port(port)
        ev = self.expected_value(port)
        # Variance: Var[X] = E[(X - μ)^2]
        # Here Var[X] = Σ P(X=x) · (x − μ)^2
        # X = reward probability of this port, which takes value p_i = states[i][port] with probability belief[i]
        # μ = expected_value(port)
        variance = sum(
            self.belief[i] * (self.states[i][port] - ev) ** 2
            for i in range(len(self.states))
        )
        # sqrt(variance) gives standard deviation
        return float(np.sqrt(variance))

    def get_values(self):
        """Return expected reward probabilities as {port: value}."""
        return {port: self.expected_value(port) for port in REWARD_PORTS}

    def get_stds(self):
        """Return standard deviation of expected reward for each port as {port: std}."""
        return {port: self.expected_value_std(port) for port in REWARD_PORTS}

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

    def reward_nll(self, ports, rewards):
        """
        Compute the negative log-likelihood of a reward sequence under this
        model's current parameters.

        The belief-weighted average reward probability at each trial is used as the
        predicted reward probability, so the likelihood of each trial is
        Bernoulli(expected_value(port)).

        Runs the model from scratch with the current params
        (self.reward_set, self.alpha, self.decay).

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.

        Returns:
            float: Total negative log-likelihood.
        """
        # Create a fresh instance with the same parameters so we don't modify the current model
        model = HiddenStatePortLearner(reward_set=self.reward_set,
                                       alpha=self.alpha,
                                       decay=self.decay)
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

    def choice_nll(self, ports, rewards, temperature=None):
        """
        Compute the negative log-likelihood of the rat's port CHOICES under
        this model's current parameters.

        On each trial the rat chooses between the 2 ports it didn't just visit
        (first trial: all 3 ports available). The choice probability comes from
        softmax over expected values with the given temperature.

        This is different from nll(), which scores how well the model predicts
        REWARDS. choice_nll() scores how well the model predicts which PORT
        the rat chose.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.
            temperature (float, optional): Softmax temperature. If None, uses 1.0.

        Returns:
            float: Total negative log-likelihood of the choice sequence.
        """
        if temperature is None:
            temperature = getattr(self, '_choice_temperature', 1.0)
        model = HiddenStatePortLearner(reward_set=self.reward_set,
                                       alpha=self.alpha, decay=self.decay)
        total = 0.0
        prev_port = None
        for port, reward in zip(ports, rewards):
            # First trial: all 3 ports available. After that: exclude previous port.
            if prev_port is None:
                available = REWARD_PORTS
            else:
                available = [p for p in REWARD_PORTS if p != prev_port]

            # Softmax probability of the rat's actual choice BEFORE updating
            probs = model.choice_probabilities(available_ports=available,
                                               temperature=temperature)
            p_choice = probs[resolve_port(port)]
            total -= np.log(max(p_choice, 1e-10))

            # Update model with this trial's outcome
            model.update(port, reward)
            prev_port = resolve_port(port)
        return total

    @classmethod
    def fit_choices(cls, ports, rewards, reward_set=(0.9, 0.5, 0.1)):
        """
        Fit alpha, decay, and temperature to maximize the likelihood of the
        rat's port choices (not rewards).

        Uses L-BFGS-B to minimize choice_nll(). Returns a fitted instance with
        best-fit parameters and ``choice_nll_``, ``choice_bic_``, and
        ``choice_result_`` attributes.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int or float): Reward received on each trial.
            reward_set (tuple of float): The assumed set of reward probabilities.
                Defaults to (0.9, 0.5, 0.1).

        Returns:
            HiddenStatePortLearner: Fitted instance with attributes:
                - choice_nll_    : choice NLL at optimum
                - choice_bic_    : BIC (3 params: alpha, decay, temperature)
                - choice_result_ : raw scipy OptimizeResult
                - temperature    : best-fit softmax temperature
        """
        def _obj(params):
            alpha, decay, temperature = params
            model = cls(reward_set=reward_set, alpha=alpha, decay=decay)
            return model.choice_nll(ports, rewards, temperature=temperature)

        result = minimize(_obj, x0=[1.0, 0.05, 1.0],
                          bounds=[(1e-3, 1.0), (0.0, 0.5), (0.01, 10.0)],
                          method='L-BFGS-B')

        fitted = cls(reward_set=reward_set, alpha=result.x[0],
                     decay=result.x[1])
        fitted.temperature = result.x[2]
        fitted._choice_temperature = result.x[2]
        fitted.choice_nll_ = result.fun
        n = len(rewards)
        fitted.choice_bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.choice_result_ = result
        return fitted


    @classmethod
    def fit_rewards(cls, ports, rewards, reward_set=(0.9, 0.5, 0.1)):
        """
        Fit alpha and decay to maximise the likelihood of a reward sequence.
        The reward_set is treated as fixed and known.

        Returns a fitted instance with best-fit parameters and ``reward_nll_``,
        ``reward_bic_``, and ``reward_result_`` attributes.

        Parameters:
            ports (list of int or str): Port sequence.
            rewards (list of int or float): Reward sequence.
            reward_set (tuple of float): The assumed set of reward probabilities.
                Defaults to (0.9, 0.5, 0.1).

        Returns:
            HiddenStatePortLearner: Fitted instance with attributes:
                - reward_nll_    : NLL at optimum
                - reward_bic_    : BIC (2 params)
                - reward_result_ : raw scipy OptimizeResult
        """
        # Objective: construct a fresh model for each candidate parameter set and compute NLL
        # L-BFGS-B respects the bounds without needing a penalty
        def _obj(params):
            alpha, decay = params
            return cls(reward_set=reward_set,
                       alpha=alpha,
                       decay=decay).reward_nll(ports, rewards)

        # Starting point: alpha=1.0 (full Bayesian update), decay=0.05
        # Bounds keep alpha in (0, 1] and decay in [0, 0.5]
        result = minimize(_obj, x0=[1.0, 0.05],
                          bounds=[(1e-3, 1.0), (0.0, 0.5)],
                          method='L-BFGS-B')

        # Fitted = model with the best fit alpha and decay
        fitted = cls(reward_set=reward_set, alpha=result.x[0], decay=result.x[1])
        fitted.reward_nll_ = result.fun
        # Compute Bayesian Information Criterion (BIC) as a metric for how good this model is
        # BIC = k*ln(n) + 2*NLL, where k = number of free parameters and n = number of trials
        # The k*ln(n) term penalises model complexity, so models with different
        # numbers of parameters can be compared on the same scale (lower is better)
        # This model has k=2 (alpha, decay)
        n = len(rewards)
        fitted.reward_bic_ = len(result.x) * np.log(n) + 2 * result.fun
        fitted.reward_result_ = result
        return fitted

    def get_history(self):
        """Return the full learning history."""
        return list(self.history)
