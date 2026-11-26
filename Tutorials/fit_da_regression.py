"""
fit_da_regression.py

Does the rat's dopamine follow the value of the hex it's in, or the value of the hex it's representing?

For each session, we get the value of where the rat is, V(here), and of the hex it's representing, V(represented), at
every DA sample, from the best choice model (krausz_no_sweeps: KrauszDualLearner, TD(0) + memory trace, directional
states). Then we fit 3 regressions of DA on those values, on the same samples (the ones with a decoded hex):
    here          DA = b0 + b_here * V(here) + noise
    represented   DA = b0 + b_rep * V(represented) + noise
    both          DA = b0 + b_here * V(here) + b_rep * V(represented) + noise

Each regression is fit 2 ways:
    choice fit    the session's learning parameters from the hierarchical choice fit (nothing is fit to DA, so this is
                  the cleanest test of whether the values that explain the rat's choices also explain its DA)
    DA fit        the learning parameters (gamma, alpha, gamma_mb, a_T, a_mb) fit to the DA, starting from the choice
                  fit (beta only affects choices, so it isn't fit). This asks which values explain DA best, if they
                  can be different from the ones that explain choices
For any learning parameters, the regression weights and the noise sd have an exact least squares solution, so the
optimizer only searches over the learning parameters.

To compare the regressions within a session, use BIC (lower is better) or R^2. Neighboring DA samples are strongly
correlated, so the per-sample likelihood (and BIC) treats the session as having far more independent data than it
has: compare the regressions by how consistently one wins across sessions, not by the size of the BIC difference
in one session. DOWNSAMPLE (e.g. to about 10 Hz) helps with this and makes the DA fits much faster.

To run it:
    1. Copy this file, tutorial_utils.py, the session files (example_data/trial_representation), and
       results/hierarchical_conditions/choice_fit_params.csv to the server, next to a checkout of the hexmaze repo
       (with the resolve_snapshot_entry_directions fix, or value_timeseries crashes on some sessions)
    2. Fill in load_da below
    3. Run: python fit_da_regression.py
Results go in da_regression.csv (one row per session, regression, and parameter source).
"""
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hexmaze.rl import KrauszDualLearner  # noqa: E402
from hexmaze.rl.fitting import fit_from_starts  # noqa: E402
from tutorial_utils import load_session  # noqa: E402

# The session files, the choice fit parameters, and where to save the results
SESSION_FILES = f"{HERE}/example_data/trial_representation"
CHOICE_FIT_PARAMS = f"{HERE}/results/hierarchical_conditions/choice_fit_params.csv"
OUT = f"{HERE}/da_regression.csv"

# The decoding each epoch decoded more than once was fit with (DECODE_CHOICE in fit_conditions_hierarchical.py), so
# the DA is lined up with the same paths as the choice fit
DECODE_CHOICE = {
    "IM-1871_20250804_ep0": "43b3990e",
    "IM-1971_20260619_ep0": "be9b0acb",
    "Lily20251221_ep5": "043c1411",
    "Toby20250316_ep1": "804d4fd0",
    "Toby20250316_ep7": "f40538a7",
    "Toby20250318_ep1": "71ce3fbf",
    "Toby20250318_ep3": "2e2777bb",
    "Toby20250318_ep5": "04cda9c4",
    "Toby20250318_ep7": "f28ed7b7",
    "Toby20250327_ep3": "198ac087",
    "Toby20250327_ep5": "e7b6f591",
    "Toby20250327_ep7": "b94510ec",
}

# The model (the best one by iBIC and held-out NLL), and the learning parameters to fit to DA
CONDITION = "krausz_no_sweeps"
LEARNING_PARAMS = {"gamma": "unit", "alpha": "unit", "gamma_mb": "unit", "a_T": "unit", "a_mb": "unit"}

# Each regression: which values it uses
REGRESSIONS = {"here": ["value"], "represented": ["represented_value"], "both": ["value", "represented_value"]}

# How many samples DA lags behind the values (e.g. for the sensor's kinetics): DA at sample i + LAG_SAMPLES is
# regressed on the values at sample i. Try a few to see how the answer depends on it
LAG_SAMPLES = 0

# Keep every DOWNSAMPLE-th sample (1 keeps them all)
DOWNSAMPLE = 1

# How many starting points to fit each regression to DA from (the first is the choice fit)
N_STARTS = 4


def load_da(session_name):
    """
    FILL THIS IN: get one session's samples, split by trial in the same order as the session file's trials.

    Parameters:
        session_name (str): The epoch, e.g. "IM-1871_20250806_ep0"

    Returns:
        tuple: (sampled_hexes, represented_hexes, da), each a list with one list per trial, with one entry per sample:
            sampled_hexes: the hex the rat is in. With consecutive repeats removed, each trial's samples should
                follow that trial's hex_path (small tracking glitches are fine)
            represented_hexes: the decoded hex, or None for a sample with nothing decoded
            da: the DA at that sample (z-scored within the session, so the regression weights compare across
                sessions)
        or None to skip this session (e.g. no DA recorded)
    """
    raise NotImplementedError("Fill in load_da with how to get each session's samples and DA")


def session_files():
    """Get {epoch name: session file}, using the decoding each epoch was fit with (see DECODE_CHOICE)."""
    files = {}
    for path in sorted(glob.glob(f"{SESSION_FILES}/*.csv")):
        epoch_name, merge_id = path.split("/")[-1].removesuffix(".csv").rsplit("_", 1)
        if DECODE_CHOICE.get(epoch_name, merge_id) == merge_id:
            files[epoch_name] = path
    return files


def regression_nll(table, columns):
    """
    Fit DA = b0 + the given values' weights * the values + normal noise by least squares.

    Returns:
        tuple: (NLL at the least squares fit, R^2, {weight name: weight})
    """
    X = np.column_stack([np.ones(len(table))] + [table[column].to_numpy() for column in columns])
    y = table["da"].to_numpy()
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual_sum = np.sum((y - X @ weights) ** 2)
    n = len(y)
    # The normal NLL at the maximum likelihood noise variance (residual_sum / n)
    nll = n / 2 * (np.log(2 * np.pi * residual_sum / n) + 1)
    r_squared = 1 - residual_sum / np.sum((y - y.mean()) ** 2)
    return nll, r_squared, dict(zip(["b0"] + [f"b_{column}" for column in columns], weights))


class SessionDA:
    """One session's data, and how to score a set of learning parameters (kept together so sessions can be fit in
    their own processes)."""

    def __init__(self, session, sampled_hexes, represented_hexes, da):
        self.session = session
        self.sampled_hexes = sampled_hexes
        self.represented_hexes = represented_hexes
        self.da = da

    def values(self, learning_params):
        """Get V(here), V(represented) and DA at every sample we score, with these learning parameters."""
        learner = KrauszDualLearner(self.session["mazes"][0], directional_states=True, lam=0.0,
                                    **learning_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = learner.value_timeseries(
                self.session["trajectories"], self.session["rewards"], self.sampled_hexes, self.represented_hexes,
                mazes=self.session["mazes"], trials_per_block=self.session["trials_per_block"], values="step",
            )
        # Line up each sample's DA (LAG_SAMPLES later, within the same trial) with its values
        da = []
        for trial_da in self.da:
            trial_da = np.asarray(trial_da, dtype=float)
            da.append(np.concatenate([trial_da[LAG_SAMPLES:], np.full(min(LAG_SAMPLES, len(trial_da)), np.nan)]))
        table["da"] = np.concatenate(da)
        # Score the samples with a decoded hex and DA (the same samples for every regression)
        table = table.dropna(subset=["value", "represented_value", "da"])
        return table.iloc[::DOWNSAMPLE]

    def nll(self, columns, values):
        """Get the regression's NLL with these learning parameter values (for fit_from_starts)."""
        return regression_nll(self.values(dict(zip(LEARNING_PARAMS, values))), columns)[0]


class RegressionNLL:
    """One regression's NLL as a function of the learning parameters (a class so it can go to another process)."""

    def __init__(self, session_da, columns):
        self.session_da = session_da
        self.columns = columns

    def __call__(self, values):
        return self.session_da.nll(self.columns, values)


def fit_session(job):
    """Fit every regression to one session both ways, and get one row for each."""
    session_name, path, rat, params = job
    loaded = load_da(session_name)
    if loaded is None:
        return []
    sampled_hexes, represented_hexes, da = loaded
    session = load_session(path)
    choice_fit = {name: params[name] for name in LEARNING_PARAMS}
    session_da = SessionDA(session, sampled_hexes, represented_hexes, da)

    rows = []
    # Loop through regressions to fit each one with the choice fit, then fit to DA
    for regression, columns in REGRESSIONS.items():
        table = session_da.values(choice_fit)
        n = len(table)
        nll, r_squared, weights = regression_nll(table, columns)
        n_regression_params = len(columns) + 2  # the weights, the intercept, and the noise sd
        rows.append({"rat": rat, "session": session_name, "regression": regression, "params": "choice fit",
                     "samples": n, "nll": nll, "bic": 2 * nll + n_regression_params * np.log(n),
                     "r_squared": r_squared, **weights, **choice_fit})

        fitted, _, _ = fit_from_starts(RegressionNLL(session_da, columns), list(choice_fit.values()),
                                       list(LEARNING_PARAMS.values()), n_starts=N_STARTS, seed=0)
        fitted = dict(zip(LEARNING_PARAMS, fitted))
        nll, r_squared, weights = regression_nll(session_da.values(fitted), columns)
        n_params = n_regression_params + len(LEARNING_PARAMS)
        rows.append({"rat": rat, "session": session_name, "regression": regression, "params": "DA fit",
                     "samples": n, "nll": nll, "bic": 2 * nll + n_params * np.log(n),
                     "r_squared": r_squared, **weights, **fitted})
        print(f"{session_name} {regression}: choice fit R^2 {rows[-2]['r_squared']:.4f}, "
              f"DA fit R^2 {r_squared:.4f}", flush=True)
    return rows


if __name__ == "__main__":
    params = pd.read_csv(CHOICE_FIT_PARAMS)
    params = params[(params["condition"] == CONDITION) & (params["session"] != "group median")]
    files = session_files()

    # One job per session that has a choice fit (with the session file it was fit with)
    jobs = [(row["session"], files[row["session"]], row["rat"], row.to_dict()) for _, row in params.iterrows()]

    # Fit every session in parallel, one process each
    with ProcessPoolExecutor() as pool:
        rows = [row for session_rows in pool.map(fit_session, jobs) for row in session_rows]
    results = pd.DataFrame(rows)
    results.to_csv(OUT, index=False)

    # Which regression wins each session (lowest BIC), with each set of parameters
    winners = results.loc[results.groupby(["session", "params"])["bic"].idxmin()]
    print("\nsessions where each regression has the lowest BIC:")
    print(pd.crosstab(winners["params"], winners["regression"]))
    print("\nmedian R^2:")
    print(results.groupby(["params", "regression"])["r_squared"].median().unstack().round(4))
