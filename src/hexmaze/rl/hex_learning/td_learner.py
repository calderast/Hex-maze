"""
td_learner.py

TD(lambda) hex value agent for the hex maze.

A hex maze agent that learns values by temporal-difference learning.
It is BaseHexLearner holding a single TDLambdaRule, so the state
representation, policy, fitting and simulation are all as described there,
and the value update is the one in TDLambdaRule.

Value is learned over maze locations via TD learning with eligibility traces.
The single ``lam`` (lambda) controls:

    lam = 0.0  -> pure TD(0): one-step bootstrapping, value propagates
                 backward one hex per repeated traversal (e.g. Krausz 2023)
    lam = 1.0  -> Monte-Carlo: full discounted return assigned along the
                 whole path within a single trial.
    0 < lam<1  -> eligibility-trace blend of all intermediate horizons.

See BaseHexLearner for the state representation options (``directional_states``,
``goal_conditioned_states``) and the rest of the constructor arguments.
"""

from .base_learner import BaseHexLearner
from .update_rules import TDLambdaRule

__all__ = ["HexMazeTDLearner"]


class HexMazeTDLearner(BaseHexLearner):
    """
    TD(lambda) value learning over hexes.

    Parameters:
        maze: The hex maze represented in any valid format
        alpha (float): Learning rate. Defaults to 0.3
        gamma (float): Discount factor. Defaults to 0.95
        lam (float): Eligibility trace decay. 0.0 is TD(0), 1.0 is Monte-Carlo.
            Defaults to 0.0

    All other arguments are passed through to BaseHexLearner, including the
    simulation options `reward_probs` and `no_backtrack`.
    """

    def __init__(self, maze, alpha=0.3, gamma=0.95, lam=0.0, **kwargs):
        super().__init__(
            maze,
            rules=[TDLambdaRule()],
            gamma=gamma,
            alpha=alpha,
            lam=lam,
            **kwargs,
        )

