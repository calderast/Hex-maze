"""
dual_learner.py

Dual-process hex value agent for the hex maze (Krausz et al.).

This is a BaseHexLearner with two rules that update the same value table,
which together make up the dual-process model:

    TDLambdaRule   -- model-free TD, defaulting to TD(0): on every step, the
                      state the rat is leaving bootstraps from the next state.
    ModelBasedRule -- model-based update at the port: we keep a recency-weighted
                      memory trace m over the trial, average it into a persistent
                      map T when the rat reaches a port, then move every state
                      weighted by T toward the reward received.

Both rules write to the same V, so a state can be updated by either rule
(or both) on the same trial. See update_rules.py for each rule's equations,
and BaseHexLearner for the state representation and policy options.
"""

from .base_learner import BaseHexLearner
from .update_rules import TDLambdaRule, ModelBasedRule

__all__ = ["KrauszDualLearner"]


class KrauszDualLearner(BaseHexLearner):
    """
    Dual-process learner: model-free TD plus a model-based update at the port.

    Parameters:
        maze: The hex maze represented in any valid format
        alpha (float): Model-free TD learning rate. Defaults to 0.3
        gamma (float): Discount factor. Defaults to 0.95
        lam (float): Eligibility trace decay for the model-free rule.
            Defaults to 0.0, which is TD(0)
        gamma_mb (float): Decay of the model-based memory trace per step.
            Defaults to 0.9
        a_T (float): Learning rate for the persistent port-to-state map T.
            Defaults to 0.1
        a_mb (float): Model-based value learning rate. 0.0 turns off the
            model-based update, leaving a pure TD learner. Defaults to 0.1

    All other arguments are passed through to BaseHexLearner, including the
    simulation options `reward_probs` and `no_backtrack`.
    """

    def __init__(
        self,
        maze,
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
            rules=[TDLambdaRule(), ModelBasedRule()],
            gamma=gamma,
            alpha=alpha,
            lam=lam,
            gamma_mb=gamma_mb,
            a_T=a_T,
            a_mb=a_mb,
            **kwargs,
        )
