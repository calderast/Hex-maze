"""
fitting.py

Fit many models at once, in parallel (one per core).

Fitting one learner to a session means walking through that session a few hundred
times (once per parameter set the optimizer tries), so fitting a whole grid of
models one by one is slow. None of the fits depend on each other, so we can run
them in parallel. Instead of looping over `fit_choices` calls, build a dict of
jobs in the loop and pass it to `fit_many`, which returns the fitted learners
under the same keys.
"""

from concurrent.futures import ProcessPoolExecutor

__all__ = ["fit_many"]


def _run_fit(job):
    """Run one (learner class, fit_choices arguments) job. This has to be defined at
    module level so the worker processes can import it."""
    learner_class, arguments = job
    return learner_class.fit_choices(**arguments)


def fit_many(jobs, max_workers=None):
    """
    Fit a batch of models in parallel, one process per core.

    Each job is a learner class plus the arguments to pass to its fit_choices.
    Note that the fits run in separate processes, so any warnings raised while
    fitting (e.g. a tracking error in a trajectory) are printed by the worker
    process. When running from a notebook, these show up in the terminal running
    Jupyter, not in the notebook!

    Parameters:
        jobs (dict): {key: (learner class, {argument: value} for fit_choices)}. The
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
