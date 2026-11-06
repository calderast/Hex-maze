"""
fitting.py

How every learner fits its parameters (fit_from_starts), and how to fit many hex
learners at once (fit_many).

fit_from_starts fits parameters the way the Daw lab's em package does:
- Each parameter is fit as an unbounded number, so the optimizer never gets stuck
  against a bound. Parameters between 0 and 1 ("unit") are fit as their logit, and
  parameters above 0 ("positive") are fit as their log.
- The likelihood can have more than one local optimum, so we fit from several starting
  points (the given starting values, then random ones) and keep the best fit.

Fitting one hex learner means walking through the session a few hundred times, so
fitting a whole grid of models one by one is slow. None of the fits depend on each
other, so fit_many runs them in parallel: build a dict of jobs and pass it to fit_many,
which returns the fitted learners under the same keys.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

__all__ = ["fit_from_starts", "bic", "fit_many"]

# Turn a parameter into an unbounded number for the optimizer, and back
_TO_UNBOUNDED = {"unit": logit, "positive": np.log}
_FROM_UNBOUNDED = {"unit": expit, "positive": np.exp}

# The range the random starting points are drawn from, for each kind of parameter
_RANDOM_START_RANGE = {"unit": (0.05, 0.95), "positive": (0.5, 30.0)}


def _to_unbounded(values, kinds):
    """Turn parameter values into the optimizer's unbounded numbers."""
    return [_TO_UNBOUNDED[kind](value) for kind, value in zip(kinds, values)]


def _from_unbounded(x, kinds):
    """Turn the optimizer's unbounded numbers back into parameter values."""
    return [_FROM_UNBOUNDED[kind](value) for kind, value in zip(kinds, x)]


class _UnboundedObjective:
    """The objective as the optimizer sees it (taking unbounded numbers). This is a
    class instead of a lambda so it can be sent to another process."""

    def __init__(self, objective, kinds):
        self.objective = objective
        self.kinds = kinds

    def __call__(self, x):
        return self.objective(_from_unbounded(x, self.kinds))


def _minimize_from(job):
    """Run the optimizer from one starting point, hiding warnings from the data (since
    the optimizer calls the objective hundreds of times). This is at module level so
    each starting point can run in its own process."""
    objective, x0 = job
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return minimize(objective, x0=x0, method="L-BFGS-B")


def fit_from_starts(objective, start_values, kinds, n_starts=8, seed=0, max_workers=1):
    """
    Get the parameter values that minimize objective, fitting from n_starts starting
    points (start_values, then random ones) and keeping the best fit.

    Parameters:
        objective (callable): Takes a list of parameter values and returns the number to
            minimize (e.g. the NLL). To fit in parallel, it has to be something we can
            send to another process (e.g. a method of an object, not a lambda)
        start_values (list of float): The first starting point
        kinds (list of str): "unit" (between 0 and 1) or "positive" (above 0) for each parameter
        n_starts (int): How many starting points to fit from
        seed (int): Seed for the random starting points, so fits are reproducible
        max_workers (int or None): How many starting points to fit at once, one process
            each. 1 (default) fits them one at a time, and None uses one per core

    Returns:
        tuple: (the best parameter values, the scipy OptimizeResult for the best starting
            point, the OptimizeResult for every starting point). Note an OptimizeResult's
            x is in the optimizer's unbounded numbers
    """
    rng = np.random.default_rng(seed)
    starts = [_to_unbounded(start_values, kinds)]
    for _ in range(n_starts - 1):
        starts.append(_to_unbounded([rng.uniform(*_RANDOM_START_RANGE[kind]) for kind in kinds], kinds))

    jobs = [(_UnboundedObjective(objective, kinds), x0) for x0 in starts]
    if max_workers == 1 or len(jobs) == 1:
        results = [_minimize_from(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_minimize_from, jobs))

    best = min(results, key=lambda result: result.fun)
    return _from_unbounded(best.x, kinds), best, results


def bic(nll, n_params, n_observations):
    """
    Get the BIC (Bayesian Information Criterion) of a fit: BIC = k*ln(n) + 2*NLL, where k
    is the number of fitted parameters and n is the number of observations (e.g. trials,
    or scored choices). The k*ln(n) term penalizes extra parameters, so models with
    different numbers of parameters can be compared on the same scale (lower is better).
    """
    return n_params * np.log(n_observations) + 2 * nll


def _run_fit(job):
    """Run one (learner class, fit_choices arguments) job. Each job already runs in its
    own process, so it fits its starting points one at a time (max_workers=1). This is at
    module level so the worker processes can import it."""
    learner_class, arguments = job
    return learner_class.fit_choices(**{"max_workers": 1, **arguments})


def fit_many(jobs, max_workers=None):
    """
    Fit a batch of hex learners in parallel, one process per core.

    Each fit runs in its own process, so a warnings filter set here won't hide warnings
    from the data (each fit shows them once, when it scores its best fit).

    Parameters:
        jobs (dict): {key: (hex learner class, {argument: value} for fit_choices)}. The
            key can be anything, and is what the fitted learner is returned under
        max_workers (int, optional): How many processes to run at once. None (default)
            uses one per core

    Returns:
        dict: {key: the fitted learner}, with the keys of `jobs`
    """
    keys = list(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        fitted = pool.map(_run_fit, [jobs[key] for key in keys])
    return dict(zip(keys, fitted))
