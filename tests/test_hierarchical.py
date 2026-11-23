"""
Check our hierarchical fitting (src/hexmaze/rl/hierarchical.py).

Replicating the Daw lab's em package: the reference answers in em_reference/ are from em
(Julia), and come with em-jax (https://github.com/ndawlab/em-jax, see em_reference/README.md).
They fit a simple Q-learning model to simulated subjects, so we fit the same model to the
same data here, and check we get em's answers:
    test_q_learning_nll_matches_em          the model itself
    test_e_step_matches_em                  each subject's fit given a group distribution (em: estep!)
    test_em_matches_em                      the whole fit: effects, Sigma, log marginal likelihood, and the
                                            standard errors and p-values of the effects (em: emerrors)
    test_em_with_full_covariance_matches_em the same with a full covariance

What we added on top of em (em has nothing to check these against):
    test_groups_match_fitting_each_group_on_its_own   each rat its own group distribution (Krausz et al. 2023)
    test_category_covariates                          text covariates (e.g. session type)
    test_checkpoint_*                                 picking up a fit that stopped partway
    test_fit_choices_hierarchical_*                   the hex learners hooking up to EM

Run with: pytest tests/
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit, ndtr

from hexmaze import create_empty_hex_maze
from hexmaze.rl import HexMazeTDLearner
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
    # The standard errors and p-values of the effects (em: emerrors)
    np.testing.assert_allclose(fit.group_effect_tests["se"], reference["ses"], rtol=1e-3)
    np.testing.assert_allclose(fit.group_effect_tests["p"], reference["pvalues"], rtol=1e-2)


def test_em_with_full_covariance_matches_em():
    fit = fit_40_subjects(full_covariance=True)
    reference = load_reference_fit("reference_fit_full_cov.txt")

    assert fit.converged
    np.testing.assert_allclose(fit.group_effects.to_numpy().ravel(), reference["betas"], atol=1e-3)
    covariance = fit.group_covariance
    np.testing.assert_allclose([covariance[0, 0], covariance[0, 1], covariance[1, 1]], reference["sigma"], atol=1e-3)
    assert -fit.log_marginal_likelihood == pytest.approx(reference["lml"][0], abs=1e-2)
    # The standard errors and p-values of the effects (em: emerrors)
    np.testing.assert_allclose(fit.group_effect_tests["se"], reference["ses"], rtol=1e-3)
    np.testing.assert_allclose(fit.group_effect_tests["p"], reference["pvalues"], rtol=1e-2)


def test_groups_match_fitting_each_group_on_its_own():
    # Fitting with groups gives each group the same fit as fitting just its sessions (with a
    # covariate, so the covariates have to work within each group too)
    subjects = load_subjects("reference_data_40_subjects.csv")
    covariates = pd.read_csv(EM_REFERENCE / "reference_covariate_40_subjects.csv").set_index("sub")
    groups = {subject: "first half" if subject <= 20 else "second half" for subject in subjects}
    arguments = dict(start_values=expit([1.0, 0.0]), kinds=KINDS, covariates=covariates, n_starts=1, max_workers=1)

    fits = fit_hierarchical(subjects, **arguments, groups=groups)

    assert set(fits) == {"first half", "second half"}
    # The groups take a different number of iterations, so this also checks that each group
    # stops once it converges (instead of running until every group does)
    assert fits["first half"].n_iterations != fits["second half"].n_iterations
    for group, fit in fits.items():
        group_subjects = {subject: nll for subject, nll in subjects.items() if groups[subject] == group}
        alone = fit_hierarchical(group_subjects, **arguments)
        assert set(fit.session_values) == set(group_subjects)
        assert fit.n_iterations == alone.n_iterations
        np.testing.assert_allclose(fit.group_effects, alone.group_effects)
        np.testing.assert_allclose(fit.group_covariance, alone.group_covariance)
        np.testing.assert_allclose(fit.group_effect_tests, alone.group_effect_tests)
        assert fit.ibic == pytest.approx(alone.ibic)


def test_checkpoint_with_groups_picks_up_where_it_left_off(tmp_path):
    # The same as test_checkpoint_picks_up_where_it_left_off, with groups
    subjects = load_subjects("reference_data_40_subjects.csv")
    groups = {subject: subject % 2 for subject in subjects}
    arguments = dict(start_values=expit([1.0, 0.0]), kinds=KINDS, groups=groups, n_starts=1, max_workers=1)
    checkpoint = tmp_path / "checkpoint.pkl"

    uninterrupted = fit_hierarchical(subjects, **arguments)
    fit_hierarchical(subjects, **arguments, max_iterations=3, checkpoint=checkpoint)
    resumed = fit_hierarchical(subjects, **arguments, checkpoint=checkpoint)

    for group in uninterrupted:
        assert resumed[group].n_iterations == uninterrupted[group].n_iterations
        np.testing.assert_allclose(resumed[group].group_effects, uninterrupted[group].group_effects)


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


def test_checkpoint_of_finished_fit_is_returned(tmp_path):
    # Running a fit again with its checkpoint gives back the same fit, whether it stopped at
    # max_iterations or converged (there's nothing left to run either way)
    subjects = load_subjects("reference_data_5_subjects.csv")
    arguments = dict(start_values=expit([1.0, 0.0]), kinds=KINDS, n_starts=1, max_workers=1)

    for max_iterations in [3, 100]:
        checkpoint = tmp_path / f"checkpoint_{max_iterations}.pkl"
        first = fit_hierarchical(subjects, **arguments, max_iterations=max_iterations, checkpoint=checkpoint)
        again = fit_hierarchical(subjects, **arguments, max_iterations=max_iterations, checkpoint=checkpoint)

        assert again.n_iterations == first.n_iterations
        assert again.converged == first.converged
        assert again.ibic == first.ibic
        np.testing.assert_allclose(again.group_effects, first.group_effects)
        assert again.session_values.keys() == first.session_values.keys()


def test_fit_choices_hierarchical_runs():
    # Fit a few short simulated sessions, just to check that the hex learners hook up to EM
    # (simulated with no_backtrack=False, since a fit assumes the agent could take any move)
    maze = create_empty_hex_maze()
    sessions = {}
    for i in range(3):
        trials = HexMazeTDLearner(maze, alpha=0.3, gamma=0.9, beta=5.0, reward_probs=[0.9, 0.5, 0.1],
                                  no_backtrack=False).simulate(start_state=1, n_trials=8)
        sessions[f"session {i}"] = dict(maze=maze, trajectories=[trial["path"] for trial in trials],
                                        rewards=[trial["reward"] for trial in trials])

    fit = HexMazeTDLearner.fit_choices_hierarchical(sessions, lam=0.0, n_starts=1, max_iterations=2, max_workers=1)

    assert fit.param_names == ["gamma", "beta", "alpha"]
    assert set(fit.learners) == set(sessions)
    for key, learner in fit.learners.items():
        assert learner.choice_nll_ == fit.session_nll[key]
        assert learner.choices_scored_ > 0


def test_fit_choices_hierarchical_with_groups():
    # With groups, each session's learner is in its own group's fit, with that fit's parameters
    maze = create_empty_hex_maze()
    sessions = {}
    for i in range(4):
        trials = HexMazeTDLearner(maze, alpha=0.3, gamma=0.9, beta=5.0, reward_probs=[0.9, 0.5, 0.1],
                                  no_backtrack=False).simulate(start_state=1, n_trials=8)
        sessions[(f"rat {i % 2}", i)] = dict(maze=maze, trajectories=[trial["path"] for trial in trials],
                                             rewards=[trial["reward"] for trial in trials])

    fits = HexMazeTDLearner.fit_choices_hierarchical(
        sessions, groups={key: key[0] for key in sessions}, lam=0.0, n_starts=1, max_iterations=2, max_workers=1,
    )

    assert set(fits) == {"rat 0", "rat 1"}
    for rat, fit in fits.items():
        assert set(fit.learners) == {key for key in sessions if key[0] == rat}
        for key, learner in fit.learners.items():
            assert [learner.gamma, learner.beta, learner.alpha] == fit.session_values[key]
            assert learner.choice_nll_ == fit.session_nll[key]
