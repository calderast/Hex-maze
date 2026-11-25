"""
choice_rules.py

Choice rules bias the agent's choices directly, on top of the values. Each one adds a
number to each possible move's logit in the softmax:

    logit(move) = beta * value(move) + the bonus from each choice rule

This is the third kind of rule a learner can have, next to:
- update rules (update_rules.py), which change the values V the agent has learned
- the retrieval rule (retrieval_rules.py), which turns V into a value for each move

A choice rule doesn't touch V, and its bonus isn't scaled by beta. That makes it the
place for things that affect choices but aren't value, like a habit of repeating the
same turn (PersistenceRule).

A learner holds a list of choice rules (`choice_rules=`), and each one can see every move
the rat makes (on_step), so it can remember what the rat did (e.g. its last turn at
each junction). Like update rules, each choice rule lists the parameters it reads from
the learner in FIT_PARAMS, so fit_choices fits them too.

ChoiceRule (the base class that defines when a rule gets called) comes first, followed
by PersistenceRule.
"""

import numpy as np

__all__ = ["ChoiceRule", "PersistenceRule"]


class ChoiceRule:
    """
    Base interface for something that biases choices directly, instead of through the
    values (e.g. a habit of repeating the last turn).

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
        Called once per hex-to-hex move the rat makes (after the move's choice is scored,
        so a choice never sees the move it's about to make), from `state` into `next_state`.
        """

    def bonus(self, learner, context, state, neighbors):
        """
        Called for every choice (see BaseHexLearner.softmax_probabilities). `state` is the
        state the agent is choosing from, and `neighbors` are the hexes it could move to.

        Returns:
            np.ndarray or None: One number per neighbor, added to beta * value(move)
                before the softmax (so a bonus of b makes a move e^b times as likely,
                all else equal), or None for no bonus
        """
        return None


class PersistenceRule(ChoiceRule):
    """
    A bias toward repeating the move the rat made the last time it was in the same state.

    Hyperparameters, read from the learner:
        learner.persistence   how much the last move is favored (default 0, no bias)

    For a directional learner, a state is arriving at a hex from a particular hex, so at
    a junction this is the turn (left or right) the rat made the last time it came
    through that junction the same way. For a non-directional learner, it's the move the
    rat made the last time it was at that hex. Each value table (start port) keeps its
    own memory.

        logit(move) = beta * value(move) + persistence * [move is the last move from here]

    so the last move is e^persistence times as likely as it would be from the values
    alone. A negative persistence is a bias toward switching. The values don't change, so
    this captures choices the rat repeats out of habit, leaving the values to explain the
    rest.

    The memory starts empty each session (on_reset), and keeps every state's last move
    for the whole session. After a barrier change, a last move into what is now a
    barrier just never matches a possible move.
    """

    DEFAULT_PERSISTENCE = 0.0
    FIT_PARAMS = {"persistence": (1.0, "real")}

    def on_reset(self, learner):
        # {(context, state): the hex the rat moved to the last time it was in that state}
        self.last_moves = {}

    def on_step(self, learner, context, state, next_state):
        self.last_moves[(context, state)] = learner.hex_of_state(next_state)

    def bonus(self, learner, context, state, neighbors):
        last_move = self.last_moves.get((context, state))
        # The rat hasn't been in this state yet this session, so there's nothing to repeat
        if last_move is None:
            return None
        persistence = getattr(learner, "persistence", self.DEFAULT_PERSISTENCE)
        # persistence for the move the rat made last time, and 0 for the others
        return persistence * (np.asarray(neighbors) == last_move)
