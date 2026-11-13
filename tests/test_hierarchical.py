"""
Check that our hierarchical fitting (src/hexmaze/rl/hierarchical.py) gets the same answers
as the Daw lab's em package.

The reference answers in em_reference/ are from em (Julia), and come with em-jax
(https://github.com/ndawlab/em-jax, see em_reference/README.md). They fit a simple
Q-learning model to simulated subjects, so we fit the same model to the same data here.

Run with: pytest tests/
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit, ndtr

from hexmaze.rl.hierarchical import _SessionPosterior, _design_matrix, _fit_session, fit_hierarchical

EM_REFERENCE = Path(__file__).parent / "em_reference"


class QLearningNLL:
    """
    em's Q-learning model (qlik in em, qlik_nll in em-jax): 2 options, rewards of 1 or -1,
    and 2 parameters, an inverse temperature (beta) and a learning rate (fit as its probit).

    em fits these as any number, but our hierarchical fit takes parameters between 0 and 1
    ("unit") and fits their logit. So we fit both as "unit" parameters, and take their
    logit here to get em's numbers back. That way our unbounded numbers are exactly em's
    parameters, and the fits should match.
    """

    def __init__(self, choices, rewards):
        self.choices = choices
        self.rewards = rewards

    def __call__(self, values):
        beta, learning_rate_probit = (float(value) for value in logit(values))
        learning_rate = float(ndtr(learning_rate_probit))
        Q = [0.0, 0.0]
        nll = 0.0
        for choice, reward in zip(self.choices, self.rewards):
            # -log softmax probability of the choice (with math instead of numpy, since
            # the fits call this a lot and math is much faster on single numbers)
            a, b = beta * Q[0], beta * Q[1]
            nll -= beta * Q[choice - 1] - (max(a, b) + math.log1p(math.exp(-abs(a - b))))
            Q[choice - 1] = (1 - learning_rate) * Q[choice - 1] + reward
        return nll


KINDS = ["unit", "unit"]


def load_subjects(filename):
    """Get {subject: QLearningNLL} for one of em's data files."""
    data = pd.read_csv(EM_REFERENCE / filename)
    return {subject: QLearningNLL(trials["c"].tolist(), trials["r"].tolist())
            for subject, trials in data.groupby("sub")}


def load_reference_fit(filename):
    """Get {name: values} for one of em's reference fits (skipping each subject's x)."""
    reference = {}
    for line in open(EM_REFERENCE / filename):
        name, *values = line.strip().split(",")
        if name != "x":
            reference[name] = np.array(values, dtype=float)
    return reference


def fit_40_subjects(full_covariance):
    """Fit em's 40 subjects with its covariate, the same way em's reference fits do."""
    subjects = load_subjects("reference_data_40_subjects.csv")
    covariates = pd.read_csv(EM_REFERENCE / "reference_covariate_40_subjects.csv").set_index("sub")
    return fit_hierarchical(
        subjects, start_values=expit([1.0, 0.0]), kinds=KINDS, covariates=covariates,
        full_covariance=full_covariance, n_starts=1, tolerance=1e-4, max_iterations=200,
    )


def test_q_learning_nll_matches_em():
    # Make sure we have em's model right before comparing fits
    subjects = load_subjects("reference_data_5_subjects.csv")
    for line in open(EM_REFERENCE / "reference_nll.txt"):
        subject, beta, learning_rate_probit, nll = map(float, line.split(","))
        assert subjects[int(subject)](expit([beta, learning_rate_probit])) == pytest.approx(nll, abs=1e-8)


def test_e_step_matches_em():
    # Fit each subject given a fixed group distribution: each subject's MAP, its posterior
    # covariance (inverse Hessian), and NLL - log N(x; mu, Sigma) there should match em's
    subjects = load_subjects("reference_data_5_subjects.csv")
    group_mean, group_covariance = np.array([0.8, 0.1]), np.diag([2.0, 1.5])

    # em includes the constant we leave out of NLL - log N(x; mu, Sigma)
    constant = np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(group_covariance))

    for line in open(EM_REFERENCE / "reference_estep.txt"):
        subject, x1, x2, h11, h12, h22, objective = map(float, line.split(","))
        posterior = _SessionPosterior(subjects[int(subject)], KINDS, group_mean, group_covariance)
        estimate = _fit_session((posterior, expit(group_mean), 1, 0))
        posterior_covariance = np.linalg.inv(estimate["hessian"])

        np.testing.assert_allclose(estimate["x"], [x1, x2], atol=1e-4)
        np.testing.assert_allclose(
            [posterior_covariance[0, 0], posterior_covariance[0, 1], posterior_covariance[1, 1]],
            [h11, h12, h22], atol=1e-4,
        )
        assert posterior(estimate["values"]) + constant == pytest.approx(objective, abs=1e-6)


def test_em_matches_em():
    # The whole EM fit, with a covariate: the effects, Sigma, and log marginal likelihood
    fit = fit_40_subjects(full_covariance=False)
    reference = load_reference_fit("reference_fit.txt")

    assert fit.converged
    np.testing.assert_allclose(fit.group_effects.to_numpy().ravel(), reference["betas"], atol=1e-3)
    np.testing.assert_allclose(np.diag(fit.group_covariance), reference["sigma"], atol=1e-3)
    # em reports -log marginal likelihood
    assert -fit.log_marginal_likelihood == pytest.approx(reference["lml"][0], abs=1e-2)


def test_em_with_full_covariance_matches_em():
    fit = fit_40_subjects(full_covariance=True)
    reference = load_reference_fit("reference_fit_full_cov.txt")

    assert fit.converged
    np.testing.assert_allclose(fit.group_effects.to_numpy().ravel(), reference["betas"], atol=1e-3)
    covariance = fit.group_covariance
    np.testing.assert_allclose([covariance[0, 0], covariance[0, 1], covariance[1, 1]], reference["sigma"], atol=1e-3)
    assert -fit.log_marginal_likelihood == pytest.approx(reference["lml"][0], abs=1e-2)


def test_category_covariates():
    # A text covariate (e.g. rat) gets a 0/1 column for each category but the first, so
    # the intercept is the first category
    covariates = pd.DataFrame({"rat": ["B", "A", "C", "A"], "day": [0, 1, 2, 3]}, index=["s1", "s2", "s3", "s4"])
    X = _design_matrix(covariates, {"rat": ["A", "B", "C"]})

    assert list(X.columns) == ["intercept", "rat[B]", "rat[C]", "day"]
    np.testing.assert_array_equal(X.loc["s1"], [1, 1, 0, 0])
    np.testing.assert_array_equal(X.loc["s2"], [1, 0, 0, 1])
    np.testing.assert_array_equal(X.loc["s3"], [1, 0, 1, 2])


def test_checkpoint_picks_up_where_it_left_off(tmp_path):
    # A fit that stops partway and picks up from its checkpoint gives the same answer as one that doesn't stop
    subjects = load_subjects("reference_data_5_subjects.csv")
    arguments = dict(start_values=expit([1.0, 0.0]), kinds=KINDS, n_starts=1, max_workers=1)
    checkpoint = tmp_path / "checkpoint.pkl"

    uninterrupted = fit_hierarchical(subjects, **arguments)
    stopped = fit_hierarchical(subjects, **arguments, max_iterations=3, checkpoint=checkpoint)
    resumed = fit_hierarchical(subjects, **arguments, checkpoint=checkpoint)

    assert not stopped.converged
    assert resumed.n_iterations == uninterrupted.n_iterations
    np.testing.assert_allclose(resumed.group_effects, uninterrupted.group_effects)
    np.testing.assert_allclose(resumed.group_covariance, uninterrupted.group_covariance)
