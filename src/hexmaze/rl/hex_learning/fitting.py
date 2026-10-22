"""
fitting.py

Fit many models at once, one per core.

Fitting one learner to a session means walking that session a few hundred times,
once per parameter set the optimizer tries, and a grid of models is a few hundred
of those walks per cell with no cell depending on any other. `fit_many` takes the
fits as a dict of jobs and hands back the fitted learners keyed the same way, so a
grid written as a loop of `fit_choices` calls becomes that loop collecting jobs,
then one call.
"""

from concurrent.futures import ProcessPoolExecutor

__all__ = ["fit_many"]


def _run_fit(job):
    """Run one (learner class, fit_choices arguments) job. At module level so that a
    worker process can import it."""
    learner_class, arguments = job
    return learner_class.fit_choices(**arguments)


def fit_many(jobs, max_workers=None):
    """
    Fit a batch of models in parallel, one process per core.

    Each job is a learner class and the arguments to fit that class with, exactly
    as fit_choices takes them. The fits happen in separate processes, so a warning
    raised while fitting (a tracking error in a trajectory, say) is printed by the
    worker process, which from a notebook shows up in the terminal running Jupyter.

    Parameters:
        jobs (dict): {key: (learner class, {argument: value} for fit_choices)}, where
            the key is whatever the caller wants the fitted learner filed under
        max_workers (int, optional): How many processes to run at once. None (default)
            uses one per core

    Returns:
        dict: {key: the fitted learner}, with the keys of `jobs`
    """
    keys = list(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        fitted = pool.map(_run_fit, [jobs[key] for key in keys])
    return dict(zip(keys, fitted))
