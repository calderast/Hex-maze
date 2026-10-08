"""
td_learner.py

TD(lambda) hex value agent for the hex maze.

Thin wrapper over BaseHexLearner that plugs in TDLambdaRule, so all of the
graph, state, policy, fitting and simulation machinery lives in the base class
and only the TD(lambda) update itself lives in the rule.

Value is learned over maze locations via TD learning with eligibility traces.
The single ``lam`` (lambda) controls:

    lam = 0.0  -> pure TD(0): one-step bootstrapping, value propagates
                 backward one hex per repeated traversal (e.g. Krausz 2023)
    lam = 1.0  -> Monte-Carlo: full discounted return assigned along the
                 whole path within a single trial.
    0 < lam<1  -> eligibility-trace blend of all intermediate horizons.

See BaseHexLearner for the state representation options (``directional``,
``goal_conditioned``) and the rest of the constructor arguments.
"""

from .base_learner import BaseHexLearner
from .update_rules import TDLambdaRule

__all__ = ["HexMazeTDLearner"]


class HexMazeTDLearner(BaseHexLearner):
    """
    TD(lambda) value learning over hexes.

    Parameters:
        maze: The hex maze represented in any valid format
        reward_probs (list): Reward probabilities [pA, pB, pC] for the 3 ports
        alpha (float): Learning rate. Defaults to 0.3
        gamma (float): Discount factor. Defaults to 0.95
        lam (float): Eligibility trace decay. 0.0 is TD(0), 1.0 is Monte-Carlo.
            Defaults to 0.0

    All other arguments are passed through to BaseHexLearner.
    """

    def __init__(self, maze, reward_probs, alpha=0.3, gamma=0.95, lam=0.0, **kwargs):
        super().__init__(
            maze,
            reward_probs,
            rules=[TDLambdaRule()],
            gamma=gamma,
            alpha=alpha,
            lam=lam,
            **kwargs,
        )

    def filtered_update_log(self, update):
        """
        Log entries from an update dict with a real, displayable change
        (>= 0.00005, i.e. not just "0.0000" after rounding -- covers both
        negligible eligibility and negligible delta), sorted by eligibility
        descending. Shared by format_update_text and animate_learning's
        outline so the two always agree on which hexes actually changed.
        """
        if update is None:
            return []
        log = [row for row in update["log"] if abs(row["new_value"] - row["old_value"]) >= 0.00005]
        log.sort(key=lambda row: -row["eligibility"])
        return log
