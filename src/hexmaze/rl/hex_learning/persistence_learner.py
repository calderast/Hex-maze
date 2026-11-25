"""
persistence_learner.py

Hex agent with no value learning, only persistence (see choice_rules.py).

This is a BaseHexLearner with no update rules and one choice rule:

    PersistenceRule:   a bias toward repeating the move the rat made the last time
                       it was in the same state (e.g. the same turn the last time it
                       came through a junction the same way).

No values are ever learned, so every value stays 0 (with the default priors=None,
otherwise values always stay at priors), every move is worth the same (with flat priors), 
and the agent picks each move with probability

    P(move) ∝ exp( persistence * [move = the last move from this state] )

This is the habit-only baseline: compare it to a learner with persistence (e.g.
KrauszDualLearner(choice_rules=[PersistenceRule()])) to see if value learning
explains choices on top of habit.
"""

from .base_learner import BaseHexLearner
from .choice_rules import PersistenceRule

__all__ = ["PersistenceLearner"]


class PersistenceLearner(BaseHexLearner):
    """
    Persistence-only learner: repeat the last move from each state, with no value learning.

    Parameters:
        maze: The hex maze represented in any valid format
        persistence (float): How strongly the last move from a state is favored.
            Repeating it multiplies its odds (against each other move) by
            e^persistence, and a negative persistence is a bias toward switching.
            Defaults to 0.0 (random choices)

    persistence is the only parameter fit_choices fits. With every value at 0,
    gamma and beta don't do anything, so they aren't fit.

    All other arguments are passed through to BaseHexLearner, including the
    simulation options `reward_probs` and `no_backtrack`.
    """

    # Only PersistenceRule's persistence gets fit (not gamma or beta)
    FIT_PARAMS = {}

    def __init__(self, maze, persistence=0.0, **kwargs):
        super().__init__(
            maze,
            rules=[],
            choice_rules=[PersistenceRule()],
            persistence=persistence,
            **kwargs,
        )
