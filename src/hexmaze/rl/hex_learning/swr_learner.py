"""
swr_learner.py

A hex maze agent that learns from running the maze and from replaying it.

Runs a model-free and a replay-driven update side by side on the same value
table:

    TDLambdaRule   model-free TD, defaulting to TD(0): on every step, the hex
                   being left is pulled toward the value of the hex being
                   entered.
    SWRRule        replay update: on arrival at a reward port, the hexes
                   replayed during that trial's sharp-wave ripples are each
                   pulled toward the reward that was delivered.

Both rules write to the same V, so a hex can be updated by either mechanism
(or both) on the same trial. See update_rules.py for each rule's equations,
and BaseHexLearner for the state representation and policy options.
"""

from .base_learner import BaseHexLearner
from .update_rules import TDLambdaRule, SWRRule

__all__ = ["SWRDualLearner"]


class SWRDualLearner(BaseHexLearner):
    """
    Learner combining model-free TD with value learning from replayed hexes.

    Parameters:
        maze: The hex maze represented in any valid format
        alpha (float): Model-free TD learning rate. Defaults to 0.3
        gamma (float): Discount factor. Defaults to 0.95
        lam (float): Eligibility trace decay for the model-free rule.
            Defaults to 0.0, which is TD(0)
        a_swr (float): Learning rate for replayed hexes. 0.0 leaves replay with
            no effect on value, leaving a pure TD learner. Defaults to 0.1

    All other arguments are passed through to BaseHexLearner, including the
    simulation options `reward_probs` and `no_backtrack`.

    The hexes replayed on each trial are given to `learn` alongside the
    trajectories and rewards, rather than to this constructor.
    """

    def __init__(
        self,
        maze,
        alpha=0.3,
        gamma=0.95,
        lam=0.0,
        a_swr=0.1,
        **kwargs,
    ):
        super().__init__(
            maze,
            rules=[TDLambdaRule(), SWRRule()],
            gamma=gamma,
            alpha=alpha,
            lam=lam,
            a_swr=a_swr,
            **kwargs,
        )
