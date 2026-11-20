"""
What every port learner shares: learning from a session, softmax choice probabilities,
NLLs of the rat's rewards and choices, and fitting. Also port_values_per_trial, which gets
each port's value at the start of every trial (the values the rat chose with) from any
port learner.

Each learner (e.g. RescorlaWagner) only has to say how it updates after a port visit
(update) and what it currently thinks each port is worth (expected_value), and list the
parameters it fits in FIT_PARAMS.
"""

import copy

import numpy as np
import pandas as pd

from ...utils import REWARD_PORTS, resolve_port
from ..fitting import bic, fit_from_starts

PORT_NAMES = {1: "A", 2: "B", 3: "C"}


class PortLearner:
    """
    Base class for port learners. Reward ports can be specified as 1, 2, 3 or "A", "B", "C".

    Each learner has:
        - update(port, reward, start_port=None): learn from one port visit
        - expected_value(port, start_port=None): what it thinks the port is worth
        - reset(): go back to its starting values and clear its history
        - FIT_PARAMS: {name: (starting value, kind)} for the parameters fit_rewards fits,
          where kind is "unit" (between 0 and 1) or "positive" (above 0). fit_choices also
          fits beta.

    start_port is the port the rat just left (None on the session's first trial). Only
    learners with per_start_port use it.
    """

    # Each learner fills these in (e.g. RescorlaWagner fits {"alpha": (0.3, "unit"), "decay": (0.05, "unit")})
    FIT_PARAMS = {}
    # Learners that keep a separate value table for each start port set this to True in __init__
    per_start_port = False

    @classmethod
    def _with_params(cls, params, settings):
        """Make a learner with these fitted parameters and fixed settings, e.g.
        params={"alpha": 0.2, "decay": 0.01} and settings={"per_start_port": True}."""
        return cls(**params, **settings)

    def _fresh_copy(self):
        """Get a copy of this learner with the same parameters, back at its starting values."""
        # Copy everything (so the parameters come along), then forget what the copy has learned
        model = copy.deepcopy(self)
        model.reset()
        return model

    def learn(self, ports, rewards):
        """
        Run updates on a sequence of port visits and rewards.

        Parameters:
            ports (list of int or str): Sequence of ports visited (1/2/3 or A/B/C).
            rewards (list of int): Reward received at each port (0 or 1).

        Returns:
            list of float: What update returned on each trial (e.g. the reward prediction error).
        """
        outputs = []
        # Each trip starts from the port the last one ended at (none for the first trial)
        start_port = None
        for port, reward in zip(ports, rewards):
            outputs.append(self.update(port, reward, start_port))
            start_port = port
        return outputs

    def get_values(self):
        """Get each port's expected value as {port: value} (or with per_start_port,
        {start port: {port: value}})."""
        if self.per_start_port:
            return {start: {port: self.expected_value(port, start) for port in REWARD_PORTS}
                    for start in REWARD_PORTS}
        return {port: self.expected_value(port) for port in REWARD_PORTS}

    def get_history(self):
        """Get the full trial-by-trial record (one dict per update)."""
        return list(self.history)

    def choice_probabilities(self, available_ports=None, start_port=None):
        """
        Softmax choice probabilities over the ports' expected values.

        Parameters:
            available_ports (list of int or str, optional): Which ports to choose among
                (1/2/3 or A/B/C). Defaults to all 3.
            start_port (int or str, optional): The port the trip starts from. Only used with
                per_start_port, to choose with that port's values.

        Returns:
            dict of {int: float}: {port: probability}
        """
        if available_ports is None:
            available_ports = REWARD_PORTS
        else:
            available_ports = [resolve_port(p) for p in available_ports]
        # Each port's value (from the start port's table, with per_start_port)
        values = np.array([self.expected_value(p, start_port) for p in available_ports])

        # Softmax: p(port) = exp(beta * value) / sum over available ports of exp(beta * value)
        # Higher beta (inverse temperature) = more likely to choose the port with the highest value
        # Lower beta = choices are more random
        scaled = self.beta * values
        # Subtract the max before exp so it can't overflow (doesn't change the probabilities)
        exp_values = np.exp(scaled - scaled.max())
        probabilities = exp_values / exp_values.sum()
        return dict(zip(available_ports, probabilities.tolist()))

    def reward_nll(self, ports, rewards):
        """
        Get the negative log-likelihood of a reward sequence under this learner's parameters.

        Each port's expected value is its predicted reward probability, so the likelihood of
        each trial is Bernoulli(expected value of the port the rat ran to). Runs a fresh copy
        of the learner, so this one isn't changed.

        Parameters:
            ports (list of int or str): Port the rat ran to on each trial.
            rewards (list of int): Reward received on each trial (0 or 1).

        Returns:
            float: Total negative log-likelihood.
        """
        model = self._fresh_copy()
        total = 0.0
        start_port = None
        for port, reward in zip(ports, rewards):
            # The predicted reward probability, kept away from 0 and 1 so the log is finite
            p_reward = np.clip(model.expected_value(port, start_port), 1e-10, 1 - 1e-10)
            # Log probability of what happened: p(reward) if rewarded, 1 - p(reward) if not
            total -= np.log(p_reward if reward else 1 - p_reward)
            # Now learn from this trial
            model.update(port, reward, start_port)
            start_port = port
        return total

    def choice_nll(self, ports, rewards):
        """
        Get the negative log-likelihood of the rat's port choices under this learner's
        parameters (with choice_probabilities, so beta matters here).

        On the first trial the rat can choose any port, and after that any port but the one
        it just left. Runs a fresh copy of the learner, so this one isn't changed.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int): Reward received on each trial (0 or 1).

        Returns:
            float: Total negative log-likelihood of the choices.
        """
        model = self._fresh_copy()
        total = 0.0
        start_port = None
        for port, reward in zip(ports, rewards):
            # Use 1/2/3 so we can look the port up in the choice probabilities
            port = resolve_port(port)
            available = [p for p in REWARD_PORTS if p != start_port]
            # Probability of the rat's choice before learning from this trial (kept above 0 so
            # the log is finite)
            probabilities = model.choice_probabilities(available, start_port)
            total -= np.log(max(probabilities[port], 1e-10))
            model.update(port, reward, start_port)
            start_port = port
        return total

    @classmethod
    def _fit(cls, fit_params, nll, ports, rewards, n_starts, seed, settings):
        """Fit fit_params by minimizing nll (reward_nll or choice_nll) with fit_from_starts.
        Returns the fitted learner, the best OptimizeResult, and every OptimizeResult."""
        names = list(fit_params)

        # What the optimizer minimizes: make a learner with these parameter values and get its
        # NLL. nll is cls.reward_nll or cls.choice_nll, so nll(model, ...) is model.reward_nll(...)
        def objective(values):
            return nll(cls._with_params(dict(zip(names, values)), settings), ports, rewards)

        values, result, results = fit_from_starts(
            objective, start_values=[fit_params[name][0] for name in names],
            kinds=[fit_params[name][1] for name in names], n_starts=n_starts, seed=seed,
        )
        return cls._with_params(dict(zip(names, values)), settings), result, results

    @classmethod
    def fit_rewards(cls, ports, rewards, n_starts=8, seed=0, **settings):
        """
        Fit the learner's FIT_PARAMS (e.g. alpha and decay) to maximize the likelihood of a
        reward sequence (see reward_nll). See fit_from_starts (in rl/fitting.py) for how the fit works.

        Parameters:
            ports (list of int or str): Port the rat ran to on each trial.
            rewards (list of int): Reward received on each trial (0 or 1).
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.
            **settings: Fixed settings for the learner (e.g. per_start_port=True, or reward_set
                for HiddenStatePortLearner).

        Returns:
            The fitted learner, with attributes (a trailing _ means the fit set it, as in scikit-learn):
                - reward_nll_    : NLL at the best fit
                - reward_bic_    : BIC (see bic in rl/fitting.py)
                - reward_result_ : scipy OptimizeResult for the best starting point
                - reward_results_: the OptimizeResult from every starting point
        """
        fitted, result, results = cls._fit(cls.FIT_PARAMS, cls.reward_nll, ports, rewards, n_starts, seed, settings)
        # result.fun is the lowest NLL the optimizer found
        fitted.reward_nll_ = result.fun
        fitted.reward_bic_ = bic(result.fun, len(cls.FIT_PARAMS), len(rewards))
        fitted.reward_result_ = result
        fitted.reward_results_ = results
        return fitted

    @classmethod
    def fit_choices(cls, ports, rewards, n_starts=8, seed=0, **settings):
        """
        Fit the learner's FIT_PARAMS and beta (inverse temperature) to maximize the likelihood
        of the rat's port choices (see choice_nll). See fit_from_starts (in rl/fitting.py) for
        how the fit works.

        Parameters:
            ports (list of int or str): Port the rat chose on each trial.
            rewards (list of int): Reward received on each trial (0 or 1).
            n_starts (int): How many starting points to fit from (default 8).
            seed (int): Seed for the random starting points, so fits are reproducible.
            **settings: Fixed settings for the learner (e.g. per_start_port=True, or reward_set
                for HiddenStatePortLearner).

        Returns:
            The fitted learner, with attributes (a trailing _ means the fit set it, as in scikit-learn):
                - choice_nll_    : NLL at the best fit
                - choice_bic_    : BIC (see bic in rl/fitting.py)
                - choice_result_ : scipy OptimizeResult for the best starting point
                - choice_results_: the OptimizeResult from every starting point
        """
        # Choices also depend on beta, so fit it too (starting at 1, and above 0)
        fit_params = {**cls.FIT_PARAMS, "beta": (1.0, "positive")}
        fitted, result, results = cls._fit(fit_params, cls.choice_nll, ports, rewards, n_starts, seed, settings)
        # result.fun is the lowest NLL the optimizer found
        fitted.choice_nll_ = result.fun
        fitted.choice_bic_ = bic(result.fun, len(fit_params), len(rewards))
        fitted.choice_result_ = result
        fitted.choice_results_ = results
        return fitted


def port_values_per_trial(model, ports, rewards):
    """
    Run a port learner on a session and get each port's value at the start of every trial
    (before that trial's update, so the values the rat chose with).

    Works for any port learner (RescorlaWagner, BayesianPortLearner, HiddenStatePortLearner,
    BayesianHiddenStatePortLearner), e.g. one from fit_choices or fit_rewards. Runs on a fresh
    copy of the model, so the model itself isn't changed.

    Parameters:
        model: A port learner with the parameters to use.
        ports (list of int or str): Port the rat chose on each trial (1/2/3 or A/B/C).
        rewards (list of int): Reward received on each trial (0 or 1).

    Returns:
        pd.DataFrame: One row per trial (indexed by trial number, starting at 1), with columns:
            - "start port", "port", "reward": the trial (start port is None for the first trial)
            - "A", "B", "C": each port's value at the start of the trial. With per_start_port,
              these are from the start port's table, and the start port's own column is NaN
              (it wasn't an option)
            - "p(A)", "p(B)", "p(C)": the model's probability of choosing each port (NaN for
              the port the rat just left, since it wasn't an option)
            - With per_start_port, also "B from A", "C from A", "A from B", ... : every table's
              value for each port at the start of the trial
    """
    # Start from scratch, with the same parameters
    model = model._fresh_copy()

    rows = []
    start_port = None
    for port, reward in zip(ports, rewards):
        # Use 1/2/3 so we can look the port up in the model's values
        port = resolve_port(port)
        # The rat can choose any port on the first trial, and any port but the one it just left after that
        available = [p for p in REWARD_PORTS if p != start_port]
        probabilities = model.choice_probabilities(available, start_port)

        # (start port is None on the first trial)
        row = {"start port": PORT_NAMES.get(start_port), "port": PORT_NAMES[port], "reward": reward}
        for p in REWARD_PORTS:
            # With per_start_port, the start port has no value on a trip from itself
            if model.per_start_port and p == start_port:
                row[PORT_NAMES[p]] = np.nan
            else:
                row[PORT_NAMES[p]] = model.expected_value(p, start_port)
        for p in REWARD_PORTS:
            row[f"p({PORT_NAMES[p]})"] = probabilities.get(p, np.nan)
        if model.per_start_port:
            # Every table's values (a table never has its own start port as an option)
            for start, values in model.get_values().items():
                for p, value in values.items():
                    if p != start:
                        row[f"{PORT_NAMES[p]} from {PORT_NAMES[start]}"] = value
        rows.append(row)

        # Now learn from this trial
        model.update(port, reward, start_port)
        start_port = port

    return pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="trial"))
