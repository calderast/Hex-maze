"""
dual_learner.py

Dual-process hex value agent for the hex maze (Krausz et al.).

Runs a model-free and a model-based update side by side on the same value
table, which is the full dual-process model rather than either half alone:

    TDLambdaRule   -- model-free TD, defaulting to TD(0): on every step, the
                      pre-transition state bootstraps from the next state.
    ModelBasedRule -- model-based port-entry sweep: a recency-weighted memory
                      trace m is kept over the trial, averaged into a persistent
                      map T on arrival at a port, and every state weighted by T
                      is then swept toward the reward received.

Both rules write to the same V, so a state can be updated by either mechanism
(or both) on the same trial. See update_rules.py for each rule's equations,
and BaseHexLearner for the state representation and policy options.
"""

from .base_learner import BaseHexLearner
from .update_rules import TDLambdaRule, ModelBasedRule

__all__ = ["KrauszDualLearner"]


class KrauszDualLearner(BaseHexLearner):
    """
    Dual-process learner combining model-free TD with a model-based port-entry sweep.

    Parameters:
        maze: The hex maze represented in any valid format
        reward_probs (list): Reward probabilities [pA, pB, pC] for the 3 ports
        alpha (float): Model-free TD learning rate. Defaults to 0.3
        gamma (float): Discount factor. Defaults to 0.95
        lam (float): Eligibility trace decay for the model-free rule.
            Defaults to 0.0, which is TD(0)
        gamma_mb (float): Decay of the model-based memory trace per step.
            Defaults to 0.9
        a_T (float): Learning rate for the persistent port-to-state map T.
            Defaults to 0.1
        a_mb (float): Model-based value learning rate. 0.0 disables the
            model-based sweep, leaving a pure TD learner. Defaults to 0.1

    All other arguments are passed through to BaseHexLearner.
    """

    def __init__(
        self,
        maze,
        reward_probs,
        alpha=0.3,
        gamma=0.95,
        lam=0.0,
        gamma_mb=0.9,
        a_T=0.1,
        a_mb=0.1,
        **kwargs,
    ):
        super().__init__(
            maze,
            reward_probs,
            rules=[TDLambdaRule(), ModelBasedRule()],
            gamma=gamma,
            alpha=alpha,
            lam=lam,
            gamma_mb=gamma_mb,
            a_T=a_T,
            a_mb=a_mb,
            **kwargs,
        )
