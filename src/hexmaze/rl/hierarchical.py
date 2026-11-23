"""
hierarchical.py

Fit the same model to many sessions at once, hierarchically. Each session gets its own
parameters, but they're all drawn from a group distribution, which we fit too.

Why fit hierarchically: when we fit every session on its own, a parameter that one
session can't pin down can run off to an extreme (e.g. a learning rate of 1). Fitting
hierarchically pulls each session's parameters toward what's typical across sessions.
How much depends on how well that session pins them down: a session with a lot of
information about a parameter barely moves, and one with very little gets pulled most
of the way to the group.

Where this comes from:
- The method is EM with a Laplace approximation from Huys et al. 2011 ("Disentangling
  the roles of approach, activation and valence in instrumental and pavlovian
  responding").
- We replicate the Daw lab's em package (https://github.com/ndawlab/em, the Julia version,
  in src/emcore.jl). Each piece that replicates em says which em function it matches
  (e.g. "em: mstep!"), and tests/test_hierarchical.py checks that we get the same answers
  as em on em's own reference data.
- Krausz et al. 2023 ("Dual credit assignment processes underlie dopamine signals in a
  complex spatial environment") fit their models this way. From their methods, the pieces
  we match are:
    "Session-level parameters were themselves modeled as arising from a distinct
    population-level Gaussian distribution over sessions for each rat"
        -> groups (each rat gets its own group mean and its own spread)
    "an expectation-maximization algorithm with a Laplace approximation to the
    session-level marginal likelihoods"
        -> the EM below
    "For hypothesis testing on population-level parameters ... we computed an estimate of
    the information matrix over the population-level parameters, taking account of the
    so-called 'missing information' due to optimization in the E-step"
        -> group_effect_tests
  They approximated the missing information with the Hessian of a single Newton-Raphson
  step. We use em's current version instead (Louis's 1982 formula with the Laplace
  approximation), which gets the missing information straight from each session's
  posterior covariance from the E-step, and which we can check against em's reference
  answers.

How it works:
- Each parameter is fit as an unbounded number, the same as in fit_from_starts (the
  logit of a parameter between 0 and 1, and the log of a parameter above 0). Across
  sessions, these are normally distributed with covariance Sigma (diagonal by default,
  so each parameter has its own spread).
- The group mean can depend on things we know about each session (covariates, e.g.
  whether it's a barrier change session): each session's group mean is mu = X @ effects,
  where X is the session's row of the design matrix (a 1 for the intercept, then its
  covariates). With no covariates, every session has the same mean.
- Starting from a broad group distribution, we repeat:
    1. E-step (em: estep!): for each session, find the parameters that minimize
       NLL - log N(x; mu, Sigma), i.e. the best fit given the group distribution
       (the MAP), and the Hessian there. The Hessian says how well the session pins
       down its parameters: its posterior is approximately normal, centered on the
       MAP with covariance inverse(Hessian) (the Laplace approximation).
    2. M-step (em: mstep!): regress the sessions' MAPs on X to get the effects (with no
       covariates, this is just the mean of the MAPs). Sigma is the spread of the MAPs
       around their group means plus the sessions' posterior covariances, divided by
       (number of sessions - number of columns of X).
  until the effects and Sigma stop changing (by less than 0.1%, em's stopping rule).

Where we're different from em (we still get em's answers on its reference data):
- We take the Hessian with finite differences (em uses automatic differentiation).
- If a session's Hessian isn't positive definite, we clip its eigenvalues (see
  _fit_session). em uses inverse(Sigma) instead.
- We start from a group sd of 10 (em starts at 1), and fit each session from several
  starting points on the first iteration.
- groups (em fits one group distribution).

Each rat its own group distribution (Krausz et al. 2023): pass groups ({session key: its
rat}) and each rat gets its own group mean and its own spread. This is the same as
calling fit_hierarchical once per rat, but every rat's sessions are fit in the same
E-step, so they all run in parallel. Each rat stops once its own group distribution
stops changing. (A rat covariate is different: it gives each rat its own group mean, but
one spread for every rat.)

To compare models, use the integrated BIC (iBIC, em: ibic):
    iBIC = -log marginal likelihood + (number of group parameters) / 2 * ln(number of sessions)
where the log marginal likelihood (how likely the data is given the group distribution,
integrating over each session's parameters) comes from the same Laplace approximation.
Lower is better. The group parameters are the effects and Sigma (so with no covariates
and a diagonal Sigma, a mean and a spread for each fitted parameter). With groups, add up
every group's iBIC. Note this iBIC is half the scale of BIC (bic in fitting.py), so don't
compare the two. (Huys et al. 2011 use -2 * log marginal likelihood, and the total number
of choices instead of the number of sessions.)

To test whether a group effect is different from 0 (e.g. whether a covariate matters),
use group_effect_tests (em: emerrors):
- The information matrix over the group parameters (the effects and Sigma) says how well
  the data pins them down. If we knew every session's parameters exactly, it would be
  the standard one for a normal distribution (the "complete information").
- But we only know each session's parameters up to its posterior covariance from the
  E-step, so we take away the information we're missing because of that (the "missing
  information").
- The standard errors are the square roots of the diagonal of its inverse, and the
  p-values are from a t test with (number of sessions - number of columns of X) degrees
  of freedom.
These are tests on unbounded numbers, so the intercept's test is whether the group mean
is 0 in unbounded numbers (e.g. a median of 0.5 for a parameter between 0 and 1), which
usually isn't what we want to know. The covariates' tests are the useful ones.
"""

import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

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


def _sessions_design_matrix(covariates, keys):
    """
    Get the design matrix X for these sessions (see _design_matrix), and the categories of
    each category covariate.

    Parameters:
        covariates (pd.DataFrame or None): One row per session, one column per covariate
        keys (list): The sessions, in order

    Returns:
        tuple: X (pd.DataFrame, one row per session in keys), and {covariate: [its
            categories, in order]} for the category covariates
    """
    if covariates is None:
        # No covariates, so X is just the intercept (every session has the same group mean)
        return pd.DataFrame({"intercept": np.ones(len(keys))}, index=keys), {}
    # Put the covariate rows in the same order as the sessions
    covariates = covariates.loc[keys]
    # Every text column is a category covariate. Sort its categories, so the first one
    # (the one the intercept is) is always the same
    categories = {
        name: sorted(values.unique()) for name, values in covariates.items()
        if not pd.api.types.is_numeric_dtype(values)
    }
    return _design_matrix(covariates, categories), categories


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
        group_effect_tests (pd.DataFrame): Whether each group effect is different from 0
            (see the module docstring). One row per effect (indexed by the column of the
            design matrix and the parameter), with its estimate (the same as in
            group_effects), standard error, t value, and p-value
        covariates (pd.DataFrame or None): The covariates the fit was given (for this fit's
            sessions)
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

    Note session_values, session_nll, log_marginal_likelihood and ibic come from the last
    E-step, and group_effects and group_covariance come from the M-step after it. Once EM
    has converged, that last M-step barely changes the group distribution.
    """

    param_names: list
    kinds: list
    group_effects: pd.DataFrame
    group_covariance: np.ndarray
    group_effect_tests: pd.DataFrame
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
        names = [] if self.covariates is None else list(self.covariates.columns)
        # Make sure every covariate and category actually exists
        for name, value in covariate_values.items():
            if name not in names:
                raise ValueError(f"There's no covariate called {name!r} (the covariates are {names})")
            if name in self.categories and value not in self.categories[name]:
                raise ValueError(f"{value!r} isn't a category of {name!r} (they're {self.categories[name]})")

        # Get the value of each covariate (the given one, or the first category, or 0)
        covariates = {
            name: covariate_values.get(name, self.categories[name][0] if name in self.categories else 0)
            for name in names
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
    0.5 * log|2 pi Sigma|), the thing the E-step minimizes for that session
    (em: gaussianprior).

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
    Hessian there, and its log marginal likelihood (Laplace approximation). This is
    em's estep! for one session.

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

    # At a minimum, the Hessian is positive definite (the objective curves up in every
    # direction). If it isn't (e.g. the MAP is on a flat stretch, or the finite differences
    # are noisy), its inverse would have negative variances. So clip its eigenvalues to at
    # least the smallest eigenvalue of inverse(Sigma), i.e. the session is never less sure
    # of its parameters than the group distribution alone would make it. (This is
    # different from em, which uses inverse(Sigma) itself as the Hessian then)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    hessian_clipped = eigenvalues.min() <= 0
    if hessian_clipped:
        floor = np.linalg.eigvalsh(posterior.group_precision).min()
        hessian = eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T

    # log p(data | mu, Sigma) is approximately
    #     -NLL(x) - 0.5 * (x - mu)' inverse(Sigma) (x - mu) - 0.5 * log|Sigma| - 0.5 * log|Hessian|
    # (the integral of a normal centered on the MAP, em: lml for one session)
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
        "log_marginal_likelihood": log_marginal_likelihood, "hessian_clipped": hessian_clipped,
    }


def _run_e_step(jobs, pool):
    """Run the E-step for every session (in parallel, unless pool is None)."""
    if pool is None:
        return [_fit_session(job) for job in jobs]
    return list(pool.map(_fit_session, jobs))


def _m_step(design, maps, posterior_covariances, full_covariance):
    """
    Run the M-step for one group (em: mstep!): the effects are the regression of the
    sessions' MAPs on X (with no covariates, the mean of the MAPs). The group covariance is
    the spread of the MAPs around their group means plus each session's posterior
    covariance, over (number of sessions - number of columns of X) since we estimated the
    effects from them too.

    Parameters:
        design (np.ndarray): X, one row per session
        maps (np.ndarray): Each session's MAP (unbounded numbers), one row per session
        posterior_covariances (list of np.ndarray): How unsure each session's MAP is
            (inverse(Hessian), the Laplace approximation)
        full_covariance (bool): Whether to fit how the parameters covary (if not, keep only
            the diagonal of the covariance)

    Returns:
        tuple: The new effects (one row per column of X, one column per parameter) and the
            new group covariance
    """
    n_sessions, n_regressors = design.shape
    # Least squares regression of the MAPs on X: the effects that best predict each
    # session's MAP from its covariates
    effects = np.linalg.lstsq(design, maps, rcond=None)[0]
    # How far each session's MAP is from its new group mean
    residuals = maps - design @ effects
    # Sigma = (spread of the MAPs around their group means + how unsure each MAP is),
    # averaged over sessions
    covariance = (residuals.T @ residuals + np.sum(posterior_covariances, axis=0)) / (n_sessions - n_regressors)
    # Keep only the diagonal (each parameter's own spread), unless we're fitting how
    # parameters covary too
    if not full_covariance:
        covariance = np.diag(np.diag(covariance))
    return effects, covariance


def _group_effect_tests(design, effects, covariance, maps, posterior_covariances, full_covariance):
    """
    Get the standard error, t value and p-value of each group effect, from the information
    matrix over the group parameters, corrected for the missing information (see the module
    docstring).

    This is em's emerrors (with groupinformation, informationmatrixsigma and
    missing_information), and it's what Krausz et al. 2023 used for "hypothesis testing on
    population-level parameters".

    Parameters:
        design (np.ndarray): X, one row per session
        effects (np.ndarray): The group effects, one row per column of X, one column per parameter
        covariance (np.ndarray): Sigma, the group covariance
        maps (np.ndarray): Each session's MAP (unbounded numbers), one row per session
        posterior_covariances (list of np.ndarray): Each session's posterior covariance
        full_covariance (bool): Whether every entry of Sigma was fit, or just its diagonal

    Returns:
        tuple of np.ndarray: The standard error, t value and p-value of each effect, in
            the order of effects.ravel() (each column of X, then each parameter)
    """
    n_sessions, n_regressors = design.shape
    n_params = effects.shape[1]
    precision = np.linalg.inv(covariance)

    # The entries of Sigma we fit: the diagonal, or the diagonal and everything above it.
    # sigma_directions[k] is how Sigma changes when the k-th of them goes up by 1 (an
    # off-diagonal entry is in Sigma twice, at [i, j] and [j, i])
    pairs = [(i, j) for i in range(n_params) for j in range(i, n_params) if full_covariance or i == j]
    sigma_directions = []
    for i, j in pairs:
        direction = np.zeros((n_params, n_params))
        direction[i, j] = direction[j, i] = 1.0
        sigma_directions.append(direction)
    n_effects = n_regressors * n_params

    # The complete information (as if we knew every session's parameters exactly, em:
    # groupinformation). The effects and Sigma don't share any information, so it's 2 blocks:
    complete = np.zeros((n_effects + len(pairs),) * 2)
    # For the effects, the inverse of the regression's covariance, inverse(X'X) kron Sigma
    complete[:n_effects, :n_effects] = np.linalg.inv(np.kron(np.linalg.inv(design.T @ design), covariance))
    # For Sigma, (n - number of columns of X) / 2 * trace(inverse(Sigma) dSigma_k inverse(Sigma) dSigma_l)
    # (n - number of columns of X to match the M-step). em: informationmatrixsigma, which
    # writes out each case of this trace
    for k, direction_k in enumerate(sigma_directions):
        for l, direction_l in enumerate(sigma_directions):
            complete[n_effects + k, n_effects + l] = (n_sessions - n_regressors) / 2 * np.trace(
                precision @ direction_k @ precision @ direction_l
            )

    # The missing information (Louis's formula, em: missing_information): for each session,
    # J V J', where V is its posterior covariance and J is how the gradient of its
    # log N(x; mu, Sigma) with respect to each group parameter changes with x
    missing = np.zeros_like(complete)
    for row, x, posterior_covariance in zip(design, maps, posterior_covariances):
        residual = x - row @ effects
        J = np.vstack(
            # One row for each effect (each column of X, then each parameter)
            [row[r] * precision[:, p] for r in range(n_regressors) for p in range(n_params)]
            # One row for each entry of Sigma we fit
            + [precision @ direction @ precision @ residual for direction in sigma_directions]
        )
        missing += J @ posterior_covariance @ J.T

    # The effects' covariance is their block of the inverse of the information matrix
    effect_covariance = np.linalg.inv(complete - missing)[:n_effects, :n_effects]
    # A negative variance means the information matrix isn't positive definite (e.g. EM
    # hasn't converged), so there's no standard error
    variances = np.diag(effect_covariance)
    standard_errors = np.sqrt(np.where(variances > 0, variances, np.nan))
    t_values = effects.ravel() / standard_errors
    # Two sided, with (number of sessions - number of columns of X) degrees of freedom (em: emerrors)
    p_values = 2 * t_distribution.sf(np.abs(t_values), n_sessions - n_regressors)
    return standard_errors, t_values, p_values


def fit_hierarchical(session_nlls, start_values, kinds, param_names=None, covariates=None, groups=None,
                     full_covariance=False, n_starts=8, seed=0, max_iterations=100, tolerance=1e-3, max_workers=None,
                     checkpoint=None, verbose=False):
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
        groups (dict or pd.Series, optional): {session key: its group} (e.g. its rat), to fit
            a separate group distribution for each group (see the module docstring). The
            covariates then work within each group (e.g. a session_type covariate gives each
            rat its own barrier vs probability effect). None (default) fits one group
            distribution for every session
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
            for ones under 0.01). The default 1e-3 is em's. With groups, each group stops
            on its own
        max_workers (int or None): How many sessions to fit at once, one process each.
            None (default) uses one per core, and 1 fits them one at a time
        checkpoint (str, optional): A file to save the fit to after every iteration. If
            the file already exists (e.g. the fit stopped partway), we pick up where it
            left off instead of starting over (or if it already finished, just return it).
            Use a new file for each fit (picking up from a different fit's file raises an error)
        verbose (bool): If True, print the log marginal likelihood and how much the group
            parameters changed after each iteration

    Returns:
        HierarchicalFit: The group distribution, each session's parameters, the log
            marginal likelihood and iBIC, and the tests of the group effects. With groups,
            {group: HierarchicalFit} instead, one for each group (to compare models across
            every group, add up their iBICs)
    """
    keys = list(session_nlls)
    param_names = list(param_names) if param_names is not None else [f"param {i}" for i in range(len(kinds))]

    # Which group each session is in (e.g. its rat, so each rat gets its own group distribution,
    # like Krausz et al. 2023). With no groups, every session is in one group (None), like em
    group_of = {key: None if groups is None else groups[key] for key in keys}
    group_names = [None] if groups is None else sorted(set(group_of.values()))
    # The sessions in each group, in order
    group_keys = {group: [key for key in keys if group_of[key] == group] for group in group_names}

    # The design matrix X for each group: one row per session, an intercept column, then the covariates
    design_matrices, categories = {}, {}
    for group in group_names:
        design_matrices[group], categories[group] = _sessions_design_matrix(covariates, group_keys[group])
        # The M-step divides by (number of sessions - number of columns of X), so we need more
        # sessions than that (e.g. at least 3 sessions for 2 rats)
        n_sessions, n_regressors = design_matrices[group].shape
        if n_sessions <= n_regressors:
            raise ValueError(
                f"{n_sessions} sessions{'' if group is None else f' in group {group!r}'} isn't enough to fit a "
                f"group distribution with {n_regressors} group means per parameter "
                f"({list(design_matrices[group].columns)}). We need more sessions than that"
            )
        # Each covariate has to tell us something the others don't (e.g. a number covariate
        # that's the same for every session in a group can't be told apart from the intercept)
        if np.linalg.matrix_rank(design_matrices[group].to_numpy()) < n_regressors:
            raise ValueError(
                f"The covariates{'' if group is None else f' in group {group!r}'} can't be told apart "
                f"({list(design_matrices[group].columns)}), e.g. a covariate is the same for every session. "
                f"Leave it out (for this group)"
            )

    # Start each group from a broad group distribution, centered on the starting values (the
    # intercept is the starting values, and every covariate's effect is 0). Each group keeps:
    # effects (one row per column of X, one column per parameter), Sigma, a record of every
    # iteration, its last E-step, and whether it's converged
    fits = {}
    for group in group_names:
        effects = np.zeros((design_matrices[group].shape[1], len(kinds)))
        effects[0] = _to_unbounded(start_values, kinds)
        fits[group] = {
            "effects": effects,
            # Sigma starts diagonal, with an sd of _INITIAL_GROUP_SD for every parameter
            "group_covariance": np.eye(len(kinds)) * _INITIAL_GROUP_SD**2,
            "history": [], "estimates": None, "converged": False,
        }
    # Where to start fitting each session from (after the first iteration, this is where
    # the session ended up the iteration before)
    session_starts = {key: list(start_values) for key in keys}
    # How many EM iterations we've run
    iteration = 0

    # What this fit is, saved with the checkpoint so we never pick up from a different fit
    setup = {
        "sessions": keys, "param_names": param_names, "kinds": list(kinds), "groups": group_of,
        "full_covariance": full_covariance,
        "design matrix columns": {group: list(X.columns) for group, X in design_matrices.items()},
    }

    # Pick up where a fit that stopped partway left off
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            saved = pickle.load(f)
        different = [name for name in setup if saved["setup"].get(name) != setup[name]]
        if different:
            raise ValueError(f"{checkpoint} is from a different fit (its {different} don't match this fit's). "
                             f"Use a new checkpoint file for each fit")
        fits, session_starts, iteration = saved["fits"], saved["session_starts"], saved["iteration"]
        if verbose:
            print(f"picking up from {checkpoint} after iteration {iteration}")

    def packed(effects, covariance):
        # All the group parameters in one array (the diagonal of Sigma, unless it's full)
        return np.concatenate([effects.ravel(), (covariance if full_covariance else np.diag(covariance)).ravel()])

    # Fit the sessions in parallel, with the same processes for every iteration (starting
    # new ones each iteration would be slow)
    with ProcessPoolExecutor(max_workers=max_workers) if max_workers != 1 else nullcontext() as pool:
        while iteration < max_iterations:
            # The groups that haven't converged yet (once a group converges, we stop fitting it)
            fitting = [group for group in group_names if not fits[group]["converged"]]
            if not fitting:
                break
            iteration += 1

            # E-step: fit each session given its group mean, for every group at once
            # One job per session: what to minimize, where to start, how many starting
            # points (n_starts on the first iteration, then just 1), and a different seed for
            # each session's random starting points
            jobs, job_keys = [], []
            for group in fitting:
                # Each session's group mean, mu = X @ effects (one row per session)
                group_means = design_matrices[group].to_numpy() @ fits[group]["effects"]
                for key, group_mean in zip(group_keys[group], group_means):
                    posterior = _SessionPosterior(session_nlls[key], kinds, group_mean, fits[group]["group_covariance"])
                    n_session_starts = n_starts if iteration == 1 else 1
                    jobs.append((posterior, session_starts[key], n_session_starts, seed + keys.index(key)))
                    job_keys.append(key)
            estimates = dict(zip(job_keys, _run_e_step(jobs, pool)))
            clipped = [key for key in job_keys if estimates[key]["hessian_clipped"]]
            if clipped:
                warnings.warn(f"Iteration {iteration}: the Hessian wasn't positive definite for sessions {clipped}, "
                              f"so we clipped its eigenvalues (see _fit_session). Check those sessions' fits")

            for group in fitting:
                fit = fits[group]
                fit["estimates"] = {key: estimates[key] for key in group_keys[group]}
                # How likely this group's data is given the group distribution this iteration started with
                log_marginal_likelihood = sum(
                    estimate["log_marginal_likelihood"] for estimate in fit["estimates"].values()
                )

                # M-step
                effects, group_covariance = _m_step(
                    design_matrices[group].to_numpy(),
                    np.array([estimates[key]["x"] for key in group_keys[group]]),
                    [np.linalg.inv(estimates[key]["hessian"]) for key in group_keys[group]],
                    full_covariance,
                )

                # How much the group parameters changed: relative change, or absolute change for
                # ones under 0.01 (so ones near 0 can still converge). This is em's stopping rule
                old, new = packed(fit["effects"], fit["group_covariance"]), packed(effects, group_covariance)
                change = np.max(np.abs(new - old) / np.maximum(np.abs(old), 0.01))

                # The next iteration uses the new group distribution
                fit["effects"], fit["group_covariance"] = effects, group_covariance
                # Keep a record of every iteration (to check that EM converged)
                fit["history"].append({
                    "iteration": iteration, "log_marginal_likelihood": log_marginal_likelihood,
                    "group_effects": effects.copy(), "group_sd": np.sqrt(np.diag(group_covariance)),
                    "change": change,
                })
                fit["converged"] = change < tolerance
                if verbose:
                    print(f"iteration {iteration}{'' if group is None else f', {group}'}: "
                          f"log marginal likelihood {log_marginal_likelihood:.2f}, change {change:.2g}")

            # The next iteration starts each session from where it ended up this iteration
            session_starts.update({key: estimates[key]["values"] for key in job_keys})
            if checkpoint is not None:
                # Write to a temporary file first, so stopping partway through writing
                # can't break the checkpoint
                with open(f"{checkpoint}.tmp", "wb") as f:
                    pickle.dump({"setup": setup, "fits": fits, "session_starts": session_starts,
                                 "iteration": iteration}, f)
                os.replace(f"{checkpoint}.tmp", checkpoint)

    results = {}
    for group in group_names:
        fit = fits[group]
        X = design_matrices[group]
        n_sessions, n_regressors = X.shape
        estimates = fit["estimates"]
        maps = np.array([estimates[key]["x"] for key in group_keys[group]])
        posterior_covariances = [np.linalg.inv(estimates[key]["hessian"]) for key in group_keys[group]]

        # The log marginal likelihood from the last E-step (this run's, or the checkpoint's)
        log_marginal_likelihood = fit["history"][-1]["log_marginal_likelihood"]

        # iBIC = -log marginal likelihood + (number of group parameters) / 2 * ln(number of sessions)
        # (em: ibic)
        n_params = len(kinds)
        # The group parameters are every effect, plus every entry of Sigma we fit (the
        # diagonal, or with a full covariance, the diagonal and everything above it)
        n_group_params = n_regressors * n_params + (n_params * (n_params + 1) // 2 if full_covariance else n_params)
        ibic = -log_marginal_likelihood + n_group_params / 2 * np.log(n_sessions)

        # Test whether each group effect is different from 0
        standard_errors, t_values, p_values = _group_effect_tests(
            X.to_numpy(), fit["effects"], fit["group_covariance"], maps, posterior_covariances, full_covariance,
        )
        group_effect_tests = pd.DataFrame(
            {"estimate": fit["effects"].ravel(), "se": standard_errors, "t": t_values, "p": p_values},
            index=pd.MultiIndex.from_product([X.columns, param_names], names=["effect", "parameter"]),
        )

        # Each session's parameters are from the last E-step (see the HierarchicalFit docstring)
        results[group] = HierarchicalFit(
            param_names=param_names,
            kinds=list(kinds),
            group_effects=pd.DataFrame(fit["effects"], index=X.columns, columns=param_names),
            group_covariance=fit["group_covariance"],
            group_effect_tests=group_effect_tests,
            covariates=None if covariates is None else covariates.loc[group_keys[group]],
            categories=categories[group],
            session_values={key: estimates[key]["values"] for key in group_keys[group]},
            session_nll={key: estimates[key]["nll"] for key in group_keys[group]},
            log_marginal_likelihood=log_marginal_likelihood,
            ibic=ibic,
            n_iterations=len(fit["history"]),
            converged=fit["converged"],
            history=fit["history"],
        )
    return results[None] if groups is None else results
