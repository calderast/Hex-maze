"""
choice_rules.py

Choice rules bias the agent's choices directly, instead of through the values it has
learned. The agent picks each move with probability

    P(move) ∝ exp( beta * value(move) + the bonus each choice rule gives that move )

so each choice rule adds its own term next to beta * value. It isn't scaled by beta,
and it doesn't change any values.

This is the third kind of rule a learner can have, next to:
- update rules (update_rules.py), which change the values V the agent has learned
- the retrieval rule (retrieval_rules.py), which turns V into a value for each move

Choice rules are for things that affect the rat's choices but aren't value, like a
habit of repeating the same turn (PersistenceRule). A learner can have any number of
them (`choice_rules=[...]`). Each one sees every move the rat makes, so it can remember
what the rat did, and lists the parameters it reads from the learner in FIT_PARAMS so
they get fit along with the rest of the model.

ChoiceRule (the base class that defines when a rule gets called) comes first, followed
by PersistenceRule.
"""

import numpy as np

__all__ = ["ChoiceRule", "PersistenceRule"]


class ChoiceRule:
    """
    Base interface for a bias on the agent's choices that doesn't come from the values
    (see the module docstring).

    Every hook here does nothing by default, so a rule only needs to implement the ones
    it cares about. Each learner gets its own copy of its choice rules, so a rule can
    keep what it remembers on itself.

    Each rule lists the parameters it reads from the learner in FIT_PARAMS, so fit_choices
    knows what to fit: {name: (first starting value, kind)} (see UpdateRule.FIT_PARAMS).
    """

    FIT_PARAMS = {}

    def on_reset(self, learner):
        """
        Called when the learner is reset, including once when it's built. A rule that
        remembers something over a session should clear it here.
        """

    def on_step(self, learner, context, state, next_state):
        """
        Called once for every hex-to-hex move the rat makes, from `state` into `next_state`.

        This is called after that move's choice is scored, so the bonus for a choice only
        ever uses the moves the rat made before it.
        """

    def bonus(self, learner, context, state, neighbors):
        """
        Get the bonus this rule gives each possible move, for a choice made from `state`
        (see BaseHexLearner.softmax_probabilities).

        A bonus of b on one move multiplies its odds against each other move by e^b.

        Parameters:
            learner (BaseHexLearner): The learner making the choice
            context (int or None): The value table the choice is made in
            state: The state the agent is choosing from (see BaseHexLearner.state_key)
            neighbors (list[int]): The hexes it could move to

        Returns:
            np.ndarray or None: One bonus per neighbor, or None for no bonus on any of them
        """
        return None


class PersistenceRule(ChoiceRule):
    """
    A bias toward repeating the move the rat made the last time it was in the same state.

    Hyperparameters, read from the learner:
        learner.persistence   how strongly the last move is favored (default 0, no bias)

        P(move) ∝ exp( beta * value(move) + persistence * [move = the last move from this state] )

    where [move = the last move from this state] is 1 for the move the rat made the last
    time it was in this state, and 0 for every other move. So persistence multiplies the
    odds of repeating that move (against each other move) by e^persistence. A negative
    persistence is a bias toward switching.

    What "the same state" means depends on the learner's states:
    - directional states (the default): arriving at a hex from a particular neighbor. So
      at a junction, the bias is toward the turn (left or right) the rat made the last time
      it came through that junction the same way. Coming in from a different neighbor is a
      different state, with its own last move
    - non-directional states: being at a hex, whichever way the rat came in
    With goal-conditioned states, each value table (start port) keeps its own memory.

    The memory starts empty each session, and keeps each state's most recent move for the
    rest of the session (across trials). It records every move the rat makes, including
    ones that aren't scored. So there's no bias:
    - the first time the rat is in a state
    - when the move the rat made last time isn't one of the moves being scored now (e.g.
      it turned back the last time, and only left vs right is scored), or it's now
      blocked by a barrier
    The values don't change, so the bias captures choices the rat repeats out of habit,
    leaving the values to explain the rest.
    """

    DEFAULT_PERSISTENCE = 0.0
    FIT_PARAMS = {"persistence": (1.0, "real")}

    def on_reset(self, learner):
        # {(value table, state): the hex the rat moved to the last time it was in that state}
        self.last_moves = {}

    def on_step(self, learner, context, state, next_state):
        # Remember where the rat went from this state (the hex it moved into)
        self.last_moves[(context, state)] = learner.hex_of_state(next_state)

    def bonus(self, learner, context, state, neighbors):
        last_move = self.last_moves.get((context, state))
        # The rat hasn't been in this state yet this session, so there's nothing to repeat
        if last_move is None:
            return None
        persistence = getattr(learner, "persistence", self.DEFAULT_PERSISTENCE)
        # persistence for the move the rat made last time, and 0 for every other move
        return persistence * (np.asarray(neighbors) == last_move)
