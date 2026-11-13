"""
hierarchical.py

Fit the same model to many sessions at once, hierarchically. Each session gets its own
parameters, but they're all drawn from one group distribution, which we fit too.

This is the expectation-maximization (EM) method with a Laplace approximation from
Huys et al. 2011 ("Disentangling the roles of approach, activation and valence in
instrumental and pavlovian responding"), the same thing the Daw lab's em package does
(https://github.com/ndawlab/em). tests/test_hierarchical.py checks that we get the same
answers as em on its own reference data.

Why fit hierarchically: when we fit every session on its own, a parameter that one
session can't pin down can run off to an extreme (e.g. a learning rate of 1). Fitting
hierarchically pulls each session's parameters toward what's typical across sessions.
How much depends on how well that session pins them down: a session with a lot of
information about a parameter barely moves, and one with very little gets pulled most
of the way to the group.

How it works:
- Each parameter is fit as an unbounded number, the same as in fit_from_starts (the
  logit of a parameter between 0 and 1, and the log of a parameter above 0). Across
  sessions, these are normally distributed with covariance Sigma (diagonal by default,
  so each parameter has its own spread).
- The group mean can depend on things we know about each session (covariates, e.g. which
  rat it is, or whether it's a barrier change session): each session's group mean is
  mu = X @ effects, where X is the session's row of the design matrix (a 1 for the
  intercept, then its covariates). With no covariates, every session has the same mean.
- Starting from a broad group distribution, we repeat:
    1. E-step: for each session, find the parameters that minimize
       NLL - log N(x; mu, Sigma), i.e. the best fit given the group distribution
       (the MAP), and the Hessian there. The Hessian says how well the session pins
       down its parameters: its posterior is approximately normal, centered on the
       MAP with covariance inverse(Hessian) (the Laplace approximation).
    2. M-step: regress the sessions' MAPs on X to get the effects (with no covariates,
       this is just the mean of the MAPs). Sigma is the spread of the MAPs around their
       group means plus the sessions' posterior covariances, divided by
       (number of sessions - number of columns of X).
  until the effects and Sigma stop changing (by less than 0.1%).

To compare models, use the integrated BIC (iBIC), the same way em does:
    iBIC = -log marginal likelihood + (number of group parameters) / 2 * ln(number of sessions)
where the log marginal likelihood (how likely the data is given the group distribution,
integrating over each session's parameters) comes from the same Laplace approximation.
Lower is better. The group parameters are the effects and Sigma (so with no covariates
and a diagonal Sigma, a mean and a spread for each fitted parameter). To check whether a covariate matters, compare the
iBIC with and without it.

Note this iBIC is half the scale of BIC (bic in fitting.py), so don't compare the two.
(Huys et al. 2011 use -2 * log marginal likelihood, and the total number of choices
instead of the number of sessions.)
"""

import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fitting import _FROM_UNBOUNDED, _from_unbounded, _to_unbounded, fit_from_starts

__all__ = ["fit_hierarchical", "HierarchicalFit"]

# The group distribution we start from: centered on the starting values, with a standard
# deviation of 10 in unbounded numbers (so broad that the first E-step is basically
# fitting every session on its own)
_INITIAL_GROUP_SD = 10.0

# Step size (in unbounded numbers) for the finite differences we take the Hessian with
_HESSIAN_STEP = 1e-3


def _design_matrix(covariates, categories):
    """
    Get the design matrix X from the covariates: an intercept column of 1s, then a column
    for each number covariate, and for each category covariate (e.g. rat), a 0/1 column
    for each category except the first (so the intercept is the first category, and each
    column is how different that category is from it).

    Parameters:
        covariates (pd.DataFrame): One row per session, one column per covariate
        categories (dict): {covariate: [its categories, in order]} for the category covariates

    Returns:
        pd.DataFrame: X, with the same rows as covariates
    """
    # Every session gets a 1 in the intercept column
    columns = {"intercept": np.ones(len(covariates))}
    for name, values in covariates.items():
        if name in categories:
            # A category covariate (e.g. rat) gets a 0/1 column for each category but the first,
            # e.g. "rat[IM-1594]" is 1 for IM-1594's sessions and 0 for every other session
            for category in categories[name][1:]:
                columns[f"{name}[{category}]"] = (values == category).to_numpy(dtype=float)
        else:
            # A number covariate (e.g. day) is its own column
            columns[name] = values.to_numpy(dtype=float)
    return pd.DataFrame(columns, index=covariates.index)


@dataclass
class HierarchicalFit:
    """
    The result of a hierarchical fit (see fit_hierarchical).

    Everything about the group distribution is in the optimizer's unbounded numbers
    (see fit_from_starts), since that's the space the group distribution is normal in.
    group_params() turns it into normal units.

    Attributes:
        param_names (list of str): The fitted parameters, in order
        kinds (list of str): "unit" or "positive" for each parameter
        group_effects (pd.DataFrame): How each column of the design matrix (the intercept,
            then the covariates) shifts the group mean of each parameter. One row per
            column of the design matrix, one column per parameter. With no covariates,
            this is just the group mean (the intercept)
        group_covariance (np.ndarray): Sigma, the covariance of the group distribution
        covariates (pd.DataFrame or None): The covariates the fit was given
        categories (dict): {covariate: [its categories, in order]} for the category covariates
        session_values (dict): {session key: [the session's parameter values]}, in normal units
        session_nll (dict): {session key: the session's NLL at its parameter values}
        log_marginal_likelihood (float): How likely all the data is given the group
            distribution (Laplace approximation, summed over sessions)
        ibic (float): The integrated BIC (see the module docstring)
        n_iterations (int): How many EM iterations were run
        converged (bool): Whether the effects and Sigma stopped changing before max_iterations
        history (list of dict): For each iteration, the log marginal likelihood (given the
            group distribution the iteration started with), the group effects and sd it
            ended with, and how much they changed (to check that EM converged)
        learners (dict): {session key: fitted learner}, filled in by the learners' own
            hierarchical fits (e.g. BaseHexLearner.fit_choices_hierarchical)
    """

    param_names: list
    kinds: list
    group_effects: pd.DataFrame
    group_covariance: np.ndarray
    covariates: pd.DataFrame
    categories: dict
    session_values: dict
    session_nll: dict
    log_marginal_likelihood: float
    ibic: float
    n_iterations: int
    converged: bool
    history: list
    learners: dict = field(default_factory=dict)

    @property
    def group_sd(self):
        """The standard deviation of each parameter across sessions (in unbounded numbers)."""
        return np.sqrt(np.diag(self.group_covariance))

    def group_mean(self, **covariate_values):
        """
        Get the group mean of each parameter (in unbounded numbers) for sessions with the
        given covariate values, e.g. group_mean(rat="IM-1871", day=3). A covariate that
        isn't given is at its first category, or 0 for a number covariate. With no
        covariates, this is the same for every session.
        """
        # Get the value of each covariate (the given one, or the first category, or 0)
        covariates = {} if self.covariates is None else {
            name: covariate_values.get(name, self.categories[name][0] if name in self.categories else 0)
            for name in self.covariates.columns
        }
        # Make the design matrix row for a session with these covariates
        row = _design_matrix(pd.DataFrame([covariates]), self.categories)
        # The group mean is that row times the effects: mu = X @ effects
        return (row.to_numpy() @ self.group_effects.to_numpy())[0]

    def group_params(self, **covariate_values):
        """
        Get the group distribution of each parameter in normal units, for sessions with
        the given covariate values (see group_mean), e.g. group_params(rat="IM-1871").

        The group distribution is normal in unbounded numbers, so in normal units its
        center is the median (the group mean turned back into normal units), and 90% of
        sessions are between the 5th and 95th percentiles.

        Returns:
            pd.DataFrame: One row per parameter, with its median, 5th and 95th
                percentiles across sessions (normal units), and the group mean and sd
                (unbounded numbers)
        """
        rows = {}
        group_mean = self.group_mean(**covariate_values)
        for name, kind, mean, sd in zip(self.param_names, self.kinds, group_mean, self.group_sd):
            # expit for a parameter between 0 and 1, exp for a parameter above 0
            to_normal_units = _FROM_UNBOUNDED[kind]
            # 5% of a normal distribution is more than 1.645 sd below the mean (and 5% more
            # than 1.645 sd above). Turning these back into normal units keeps them in order,
            # so they're still the 5th and 95th percentiles
            rows[name] = {
                "median": to_normal_units(mean),
                "5%": to_normal_units(mean - 1.645 * sd),
                "95%": to_normal_units(mean + 1.645 * sd),
                "mean (unbounded)": mean,
                "sd (unbounded)": sd,
            }
        return pd.DataFrame(rows).T

    def session_params(self):
        """
        Get each session's parameters in normal units.

        Returns:
            pd.DataFrame: One row per session, one column per parameter
        """
        return pd.DataFrame(self.session_values, index=self.param_names).T


class _SessionPosterior:
    """
    One session's NLL - log N(x; mu, Sigma) (leaving out the constant
    0.5 * log|2 pi Sigma|), the thing the E-step minimizes for that session.

    __call__ takes parameter values (for fit_from_starts), and of_unbounded takes the
    optimizer's unbounded numbers (for the Hessian). This is a class instead of a
    function so it can be sent to another process.
    """

    def __init__(self, nll, kinds, group_mean, group_covariance):
        # The session's NLL (takes parameter values in normal units)
        self.nll = nll
        # "unit" or "positive" for each parameter
        self.kinds = kinds
        # This session's group mean, mu (unbounded numbers)
        self.group_mean = group_mean
        # inverse(Sigma), computed once here since the penalty uses it on every call
        self.group_precision = np.linalg.inv(group_covariance)

    def penalty(self, x):
        """Get how far x is from the group mean, as 0.5 * (x - mu)' inverse(Sigma) (x - mu)."""
        # x - mu, for each parameter (unbounded numbers)
        difference = np.asarray(x) - self.group_mean
        # -log N(x; mu, Sigma), leaving out the constant. This is bigger the further x is
        # from the group mean, relative to how spread out the group is
        return 0.5 * difference @ self.group_precision @ difference

    def of_unbounded(self, x):
        """Get NLL - log N(x; mu, Sigma), for the optimizer's unbounded numbers x."""
        return self.nll(_from_unbounded(x, self.kinds)) + self.penalty(x)

    def __call__(self, values):
        """Get NLL - log N(x; mu, Sigma), for parameter values (normal units)."""
        return self.nll(values) + self.penalty(_to_unbounded(values, self.kinds))


def _hessian(f, x, step=_HESSIAN_STEP):
    """Get the Hessian (matrix of second derivatives) of f at x, with central finite differences."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    hessian = np.zeros((n, n))
    f_x = f(x)
    # steps[i] is a small step in parameter i only
    steps = np.eye(n) * step
    for i in range(n):
        # d2f/dxi2: how much the slope in parameter i changes across x
        hessian[i, i] = (f(x + steps[i]) - 2 * f_x + f(x - steps[i])) / step**2
        for j in range(i + 1, n):
            # d2f/dxi dxj: how much the slope in parameter i changes with parameter j
            # (the Hessian is symmetric, so this fills in both [i, j] and [j, i])
            hessian[i, j] = hessian[j, i] = (
                f(x + steps[i] + steps[j]) - f(x + steps[i] - steps[j])
                - f(x - steps[i] + steps[j]) + f(x - steps[i] - steps[j])
            ) / (4 * step**2)
    return hessian


def _fit_session(job):
    """
    Run the E-step for one session: find its MAP given the group distribution, the
    Hessian there, and its log marginal likelihood (Laplace approximation).

    This is at module level so each session can run in its own process.
    """
    posterior, start_values, n_starts, seed = job

    # Warnings from the data (e.g. tracking errors) would come up on every one of the
    # hundreds of times we score the session, so hide them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Find the MAP: the parameters that minimize NLL - log N(x; mu, Sigma)
        values, best, _ = fit_from_starts(
            posterior, start_values, posterior.kinds, n_starts=n_starts, seed=seed,
        )
        # The MAP in unbounded numbers (the space the group distribution is normal in)
        x = best.x
        # How sharply NLL - log N(x; mu, Sigma) curves up around the MAP. Sharper means
        # the session pins its parameters down better
        hessian = _hessian(posterior.of_unbounded, x)
        # The session's NLL alone at the MAP (without the group penalty)
        nll = posterior.nll(values)

    # log p(data | mu, Sigma) is approximately
    #     -NLL(x) - 0.5 * (x - mu)' inverse(Sigma) (x - mu) - 0.5 * log|Sigma| - 0.5 * log|Hessian|
    # (the integral of a normal centered on the MAP, see the module docstring)
    log_marginal_likelihood = (
        # How well the MAP fits the data, and how close it is to the group mean
        -nll - posterior.penalty(x)
        # -0.5 * log|Sigma| (slogdet gives log|inverse(Sigma)|, which is -log|Sigma|)
        + 0.5 * np.linalg.slogdet(posterior.group_precision)[1]
        # -0.5 * log|Hessian|: the width of the posterior around the MAP
        - 0.5 * np.linalg.slogdet(hessian)[1]
    )
    return {
        "values": values, "x": x, "hessian": hessian, "nll": nll,
        "log_marginal_likelihood": log_marginal_likelihood,
    }


def _run_e_step(jobs, pool):
    """Run the E-step for every session (in parallel, unless pool is None)."""
    if pool is None:
        return [_fit_session(job) for job in jobs]
    return list(pool.map(_fit_session, jobs))


def fit_hierarchical(session_nlls, start_values, kinds, param_names=None, covariates=None, full_covariance=False,
                     n_starts=8, seed=0, max_iterations=100, tolerance=1e-3, max_workers=None, checkpoint=None,
                     verbose=False):
    """
    Fit the same model to many sessions hierarchically, with EM (see the module docstring).

    This is the general version, which works for any model: give it one NLL function per
    session. The learners have their own versions that set these up for you (e.g.
    BaseHexLearner.fit_choices_hierarchical).

    Parameters:
        session_nlls (dict): {session key: a function that takes a list of parameter values
            and returns that session's NLL}. To fit sessions in parallel, each has to be
            something we can send to another process (e.g. a method of an object, not a
            lambda). The key can be anything (e.g. (rat, date))
        start_values (list of float): The starting values of the parameters (normal units)
        kinds (list of str): "unit" (between 0 and 1) or "positive" (above 0) for each parameter
        param_names (list of str, optional): The name of each parameter (default "param 0", ...)
        covariates (pd.DataFrame, optional): Things we know about each session that the
            group mean can depend on. One row per session (indexed by session key), one
            column per covariate. A column of text (e.g. rat, or "barrier"/"probability")
            gets its own group mean for each category, and a column of numbers (e.g. day)
            shifts the group mean in proportion to it (so center it, e.g. day - mean day, to
            make the intercept the average session). None (default) uses one group mean
            for every session
        full_covariance (bool): If True, fit the full covariance of the parameters across
            sessions (so they can be correlated). If False (default), each parameter has
            its own spread, and they're independent. A full covariance needs a lot of
            sessions to fit
        n_starts (int): How many starting points to fit each session from on the first
            iteration (see fit_from_starts). After that, each session starts from where it
            ended up on the iteration before, which is much faster
        seed (int): Seed for the random starting points
        max_iterations (int): The most EM iterations to run
        tolerance (float): Stop once no group parameter (each effect and entry of Sigma)
            changes by more than this fraction between iterations (or by more than this,
            for ones under 0.01). The default 1e-3 is em's
        max_workers (int or None): How many sessions to fit at once, one process each.
            None (default) uses one per core, and 1 fits them one at a time
        checkpoint (str, optional): A file to save the fit to after every iteration. If
            the file already exists (e.g. the fit stopped partway), we pick up where it
            left off instead of starting over. Use a new file for each fit
        verbose (bool): If True, print the log marginal likelihood and how much the group
            parameters changed after each iteration

    Returns:
        HierarchicalFit: The group distribution, each session's parameters, and the
            log marginal likelihood and iBIC
    """
    keys = list(session_nlls)
    param_names = list(param_names) if param_names is not None else [f"param {i}" for i in range(len(kinds))]

    # The design matrix X: one row per session, an intercept column, then the covariates
    if covariates is None:
        # No covariates, so X is just the intercept (every session has the same group mean)
        categories = {}
        X = pd.DataFrame({"intercept": np.ones(len(keys))}, index=keys)
    else:
        # Put the covariate rows in the same order as the sessions
        covariates = covariates.loc[keys]
        # Every text column is a category covariate. Sort its categories, so the first one
        # (the one the intercept is) is always the same
        categories = {
            name: sorted(values.unique()) for name, values in covariates.items()
            if not pd.api.types.is_numeric_dtype(values)
        }
        X = _design_matrix(covariates, categories)
    design = X.to_numpy()
    # n_regressors is the number of columns of X (the intercept plus the covariate columns)
    n_sessions, n_regressors = design.shape

    # Start from a broad group distribution, centered on the starting values (the
    # intercept is the starting values, and every covariate's effect is 0)
    # effects has one row per column of X and one column per parameter
    effects = np.zeros((n_regressors, len(kinds)))
    effects[0] = _to_unbounded(start_values, kinds)
    # Sigma starts diagonal, with an sd of _INITIAL_GROUP_SD for every parameter
    group_covariance = np.eye(len(kinds)) * _INITIAL_GROUP_SD**2
    # Where to start fitting each session from (after the first iteration, this is where
    # the session ended up the iteration before)
    session_starts = {key: list(start_values) for key in keys}

    # Pick up where a fit that stopped partway left off
    history = []
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            effects, group_covariance, session_starts, history = pickle.load(f)
        if verbose:
            print(f"picking up from {checkpoint} after iteration {len(history)}")

    def packed(effects, covariance):
        # All the group parameters in one array (the diagonal of Sigma, unless it's full)
        return np.concatenate([effects.ravel(), (covariance if full_covariance else np.diag(covariance)).ravel()])

    # Fit the sessions in parallel, with the same processes for every iteration (starting
    # new ones each iteration would be slow)
    with ProcessPoolExecutor(max_workers=max_workers) if max_workers != 1 else nullcontext() as pool:
        converged = False
        # (Picking up from a checkpoint starts at the iteration after the last one saved)
        for iteration in range(len(history) + 1, max_iterations + 1):
            # E-step: fit each session given its group mean
            # Each session's group mean, mu = X @ effects (one row per session)
            group_means = design @ effects
            # One job per session: what to minimize, where to start, how many starting
            # points (n_starts on the first iteration, then just 1), and a different seed for
            # each session's random starting points
            jobs = [
                (_SessionPosterior(session_nlls[key], kinds, group_means[i], group_covariance),
                 session_starts[key], n_starts if iteration == 1 else 1, seed + i)
                for i, key in enumerate(keys)
            ]
            estimates = dict(zip(keys, _run_e_step(jobs, pool)))
            # How likely all the data is given the group distribution this iteration started with
            log_marginal_likelihood = sum(estimate["log_marginal_likelihood"] for estimate in estimates.values())

            # M-step: the effects are the regression of the sessions' MAPs on X (with no
            # covariates, the mean of the MAPs). The group covariance is the spread of the MAPs
            # around their group means plus each session's posterior covariance, over
            # (number of sessions - number of columns of X) since we estimated the effects
            # from them too
            # Each session's MAP (unbounded numbers), one row per session
            maps = np.array([estimates[key]["x"] for key in keys])
            # How unsure each session's MAP is: inverse(Hessian) (the Laplace approximation)
            posterior_covariances = [np.linalg.inv(estimates[key]["hessian"]) for key in keys]
            # Least squares regression of the MAPs on X: the effects that best predict each
            # session's MAP from its covariates
            new_effects = np.linalg.lstsq(design, maps, rcond=None)[0]
            # How far each session's MAP is from its new group mean
            residuals = maps - design @ new_effects
            # Sigma = (spread of the MAPs around their group means + how unsure each MAP is),
            # averaged over sessions
            new_covariance = (
                (residuals.T @ residuals + np.sum(posterior_covariances, axis=0)) / (n_sessions - n_regressors)
            )
            # Keep only the diagonal (each parameter's own spread), unless we're fitting
            # how parameters covary too
            if not full_covariance:
                new_covariance = np.diag(np.diag(new_covariance))

            # How much the group parameters changed: relative change, or absolute change for
            # ones under 0.01 (so ones near 0 can still converge)
            old, new = packed(effects, group_covariance), packed(new_effects, new_covariance)
            change = np.max(np.abs(new - old) / np.maximum(np.abs(old), 0.01))

            # The next iteration uses the new group distribution
            effects, group_covariance = new_effects, new_covariance
            # and starts each session from where it ended up this iteration
            session_starts = {key: estimates[key]["values"] for key in keys}
            # Keep a record of every iteration (to check that EM converged)
            history.append({
                "iteration": iteration, "log_marginal_likelihood": log_marginal_likelihood,
                "group_effects": effects.copy(), "group_sd": np.sqrt(np.diag(group_covariance)),
                "change": change,
            })
            if checkpoint is not None:
                # Write to a temporary file first, so stopping partway through writing
                # can't break the checkpoint
                with open(f"{checkpoint}.tmp", "wb") as f:
                    pickle.dump((effects, group_covariance, session_starts, history), f)
                os.replace(f"{checkpoint}.tmp", checkpoint)
            if verbose:
                print(f"iteration {iteration}: log marginal likelihood {log_marginal_likelihood:.2f}, "
                      f"change {change:.2g}")

            if change < tolerance:
                converged = True
                break

    # iBIC = -log marginal likelihood + (number of group parameters) / 2 * ln(number of sessions)
    n_params = len(kinds)
    # The group parameters are every effect, plus every entry of Sigma we fit (the
    # diagonal, or with a full covariance, the diagonal and everything above it)
    n_group_params = n_regressors * n_params + (n_params * (n_params + 1) // 2 if full_covariance else n_params)
    ibic = -log_marginal_likelihood + n_group_params / 2 * np.log(n_sessions)

    # Each session's parameters are from the last E-step. Once EM has converged, the last
    # M-step barely changes the group distribution, so these are what it gives
    return HierarchicalFit(
        param_names=param_names,
        kinds=list(kinds),
        group_effects=pd.DataFrame(effects, index=X.columns, columns=param_names),
        group_covariance=group_covariance,
        covariates=covariates,
        categories=categories,
        session_values={key: estimates[key]["values"] for key in keys},
        session_nll={key: estimates[key]["nll"] for key in keys},
        log_marginal_likelihood=log_marginal_likelihood,
        ibic=ibic,
        n_iterations=iteration,
        converged=converged,
        history=history,
    )
