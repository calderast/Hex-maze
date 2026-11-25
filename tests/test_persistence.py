"""
Check PersistenceRule (src/hexmaze/rl/hex_learning/choice_rules.py), the bias toward repeating the move the rat
made the last time it was in the same state:
    test_persistence_favors_the_last_turn_at_a_junction   the softmax gets exactly the persistence bonus, and only
                                                          when coming through the junction the same way
    test_fit_recovers_persistence                         fitting gets back the persistence sessions were
                                                          simulated with

Run with: pytest tests/
"""

import random

import numpy as np
import pytest

from hexmaze import create_empty_hex_maze
from hexmaze.rl import BaseHexLearner, PersistenceRule, TDLambdaRule


def persistent_learner(maze, persistence, **kwargs):
    """
    A learner with persistence whose values never change (alpha=0, so every value stays 0), so the only thing that
    affects its choices is persistence.
    """
    return BaseHexLearner(maze, rules=[TDLambdaRule()], choice_rules=[PersistenceRule()], alpha=0.0, gamma=0.9,
                          lam=0.0, beta=1.0, persistence=persistence, **kwargs)


def first_junction_turn(learner, path):
    """Get the first left/right turn on a path: (the hex the rat came from, the junction, the hex it turned into)."""
    for step in range(1, len(path) - 1):
        entry_hex, junction, next_hex = path[step - 1], path[step], path[step + 1]
        candidates = learner.junction_candidates(junction, entry_hex)
        # A junction (3 neighbors), where the rat took one of the 2 ways forward
        if candidates is not None and next_hex in candidates:
            return entry_hex, junction, next_hex


def test_persistence_favors_the_last_turn_at_a_junction():
    random.seed(0)
    np.random.seed(0)
    maze = create_empty_hex_maze()
    learner = persistent_learner(maze, persistence=2.0)

    # Run one trial, and learn from it (so the learner remembers each turn the rat made)
    path = BaseHexLearner(maze, no_backtrack=False).simulate(start_state=1, n_trials=1)[0]["path"]
    entry_hex, junction, turn = first_junction_turn(learner, path)
    learner.learn([path], [0.0])

    # Coming through the junction the same way, the turn the rat made gets the bonus. Every value is 0, so the
    # logits are 2 for that turn and 0 for the other, and P(turn) = e^2 / (e^2 + 1)
    candidates = learner.junction_candidates(junction, entry_hex)
    probabilities = learner.softmax_probabilities(junction, candidates, None, entry_hex=entry_hex)
    assert dict(zip(candidates, probabilities))[turn] == pytest.approx(np.exp(2.0) / (np.exp(2.0) + 1))

    # Coming into the junction from a neighbor the rat never came from, it's a different state, so there's
    # nothing to repeat and both ways are equally likely
    moves_made = set(zip(path, path[1:]))
    other_entry = next(neighbor for neighbor in learner.graph.neighbors(junction)
                       if (neighbor, junction) not in moves_made
                       and learner.junction_candidates(junction, neighbor) is not None)
    other_candidates = learner.junction_candidates(junction, other_entry)
    probabilities = learner.softmax_probabilities(junction, other_candidates, None, entry_hex=other_entry)
    np.testing.assert_allclose(probabilities, [0.5, 0.5])


def test_fit_recovers_persistence():
    # Simulate an agent whose choices only depend on persistence, then fit persistence (with everything else fixed
    # at the values it was simulated with)
    random.seed(0)
    np.random.seed(0)
    maze = create_empty_hex_maze()
    trials = persistent_learner(maze, persistence=1.5, reward_probs=[0.9, 0.5, 0.1],
                                no_backtrack=False).simulate(start_state=1, n_trials=40)

    fitted = BaseHexLearner.fit_choices(
        maze, [trial["path"] for trial in trials], [trial["reward"] for trial in trials],
        rules=[TDLambdaRule()], choice_rules=[PersistenceRule()], alpha=0.0, gamma=0.9, lam=0.0, beta=1.0,
        n_starts=1, max_workers=1,
    )
    assert fitted.persistence == pytest.approx(1.5, abs=0.3)
