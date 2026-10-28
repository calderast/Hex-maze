"""
retrieval_rules.py

Retrieval rules decide how a hex maze agent values each possible move.

Each rule is one theory of what information the rat uses to make a choice.
A learner holds one retrieval rule, which is used everywhere values get turned
into choices: the policy, the simulated agent, and the likelihood of the rat's
observed choices.

The rules can use the sweeps the rat made while at a hex (see
BaseHexLearner.choice_nll for how to pass in sweeps for each trial). Each sweep
is a list of the hexes it represented, in the order they were represented, so a
sweep is basically a route the rat "ran" in its head. The first hex in a sweep
tells us which move the sweep was evaluating (if the rat sweeps down both
arms of a junction, that's two separate sweeps).

For a directional learner, each swept hex is read in the direction the sweep
arrived at it from (the hex before it in that same sweep).

RetrievalRule (the base class) comes first, followed by LocalRetrieval
(the default), which values a move by the hex it steps into.
"""

import numpy as np


__all__ = [
    "RetrievalRule", "LocalRetrieval",
    "BestSweptRetrieval", "BestSweptOrCachedRetrieval", "FurthestSweptRetrieval",
]


class RetrievalRule:
    """
    Base interface for how the agent values each possible move.

    A RetrievalRule decides which values the agent looks at when it makes a
    choice. This is where sweeps come into the model: a rat representing hexes
    beyond its current location can evaluate a move based on what's down that way.

    A learner holds exactly one retrieval rule, since the softmax needs a single
    value for each possible move.

    Sweeps can be given in one of two formats, and every method here accepts both:
    a list of sweeps (each a list of hexes in the order they were represented),
    or a single list of every hex represented while the rat was at that hex.
    Sweeps are None when the rat didn't represent anything beyond its current location.
    """

    # Describes which hex credited_hexes picks out, so a plot can label it.
    # A rule that doesn't credit any hex leaves this as None.
    credit_label = None

    def action_values(self, learner, hex, neighbors, context, sweeps):
        """
        Get the value of each possible move.

        Parameters:
            learner (BaseHexLearner): The learner making the choice
            hex (int): The hex the agent is choosing from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read from
            sweeps (list[list[int]] or list[int] or None): The sweeps the rat made
                from this hex, or a single list of represented hexes. None if the rat
                didn't represent anything beyond its current location

        Returns:
            np.ndarray: One value per neighbor, in the order given
        """
        raise NotImplementedError

    def credited_hexes(self, learner, hex, neighbors, context, sweeps):
        """
        Get the represented hex that each move's value came from.

        Only includes moves whose value came from a sweep (a move valued by the
        hex it steps into has no entry). Used to show on the maze what the
        agent's choice was based on.

        Parameters:
            learner (BaseHexLearner): The learner making the choice
            hex (int): The hex the agent is choosing from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read from
            sweeps (list[list[int]] or list[int] or None): The sweeps the rat made
                from this hex, or a single list of represented hexes. None if the rat
                didn't represent anything beyond its current location

        Returns:
            dict: {move: the represented hex its value came from}
        """
        return {}


class LocalRetrieval(RetrievalRule):
    """
    The agent values a move by the hex it would step into.

    This ignores sweeps entirely, so this agent chooses based only on cached
    values next to its current location.
    """

    def action_values(self, learner, hex, neighbors, context, sweeps):
        return np.array(
            [learner.V[context][learner.state_key(hex, neighbor)] for neighbor in neighbors]
        )


def steps_out_to(beyond, neighbor, swept):
    """
    Count how many hexes out from a move a hex is.

    Parameters:
        beyond (dict): {hex: the hex it is entered from}, walking out from the move
        neighbor (int): The hex this move steps into
        swept (int): The hex to measure

    Returns:
        int: Number of steps from `neighbor` out to `swept`
    """
    steps = 0
    while swept != neighbor:
        swept = beyond[swept]
        steps += 1
    return steps


def sweeps_along_move(learner, sweeps, hex, neighbor):
    """
    Get the hexes the rat swept to down one move, as (hex, the hex it was reached from).

    If sweeps are given as a list of sweeps, a sweep belongs to the move it starts
    with. If a sweep skips hexes, we fill in the direction it arrived from by
    walking out from the last hex it did hit.

    If sweeps are given as a single list of every represented hex, we use the maze
    to decide which move each hex belongs to: a hex counts for this move if you can
    reach it by taking this move and continuing outward without backtracking past
    the rat. This leaves out the rat's own hex and the hexes behind it, and the
    remaining hexes are sorted by how far out they are. A hex reachable either way
    (e.g. if the 2 paths reconnect later) counts for both moves.

    Parameters:
        learner (BaseHexLearner): The learner making the choice
        sweeps (list[list[int]] or list[int] or None): The sweeps the rat made from
            `hex`, each a list of hexes in the order they were represented, or
            a single list of hexes in the order they were represented
        hex (int): The hex the rat is choosing from
        neighbor (int): The hex this move would step into

    Returns:
        list[list[tuple]]: One list per sweep down this move, with each of its
            hexes paired with the hex it was reached from
    """
    if not sweeps:
        return []

    # A single list of represented hexes (not split into sweeps)
    if not isinstance(sweeps[0], (list, tuple, set)):
        beyond = learner.bfs_entry_directions(neighbor, prev_hex=hex)
        reached = [
            (represented, beyond[represented])
            for represented in dict.fromkeys(sweeps)   # keep first appearance, drop repeats
            if represented in beyond
        ]
        reached.sort(key=lambda pair: steps_out_to(beyond, neighbor, pair[0]))
        return [reached] if reached else []

    along = []
    for sweep in sweeps:
        if not sweep or sweep[0] != neighbor:
            continue

        reached = []
        previous = hex
        for swept in sweep:
            # Skip barrier hexes (no value to read) and repeats of the same hex
            if swept not in learner.graph or swept == previous:
                continue
            if learner.graph.has_edge(previous, swept):
                reached.append((swept, previous))
            else:
                # The sweep skips from `previous` to `swept` without the hexes
                # in between, so we don't know the direction it arrived from.
                # Walk out from `previous` to find the neighbor of `swept` on the
                # way there, which is the direction it would have come from if
                # we had caught every hex.
                on_the_way = learner.bfs_entry_directions(previous)
                reached.append((swept, on_the_way[swept]))
            previous = swept

        # This is never empty, since a sweep down this move starts with the hex it steps into
        along.append(reached)
    return along


class BestSweptRetrieval(RetrievalRule):
    """
    A move's value is the best hex the rat swept to down that way.

        value(move) = max value among the hexes of the sweeps down that way

    The value comes entirely from the sweep, so if the rat only sweeps to
    low-value hexes down one way, that move gets a low value (regardless of
    what the rat had learned about the hex it would step into). A move with no
    sweeps down it gets the value of the hex it steps into (so if the rat
    doesn't sweep at all, this is the same as LocalRetrieval).
    """

    credit_label = "best swept-to hex"

    def action_values(self, learner, hex, neighbors, context, sweeps):
        values = []
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            if along:
                values.append(max(
                    learner.V[context][learner.state_key(reached_from, swept)]
                    for sweep in along for swept, reached_from in sweep
                ))
            else:
                values.append(learner.V[context][learner.state_key(hex, neighbor)])
        return np.array(values)

    def credited_hexes(self, learner, hex, neighbors, context, sweeps):
        credited = {}
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            reached = [pair for sweep in along for pair in sweep]
            if reached:
                swept, reached_from = max(
                    reached,
                    key=lambda pair: learner.V[context][learner.state_key(pair[1], pair[0])],
                )
                credited[neighbor] = swept
        return credited


class BestSweptOrCachedRetrieval(RetrievalRule):
    """
    A move's value is the better of the cached value and the best swept-to hex.

        value(move) = max(value of the hex the move steps into,
                          max value among the hexes of the sweeps down that way)

    Sweeping can only raise a move's value (by finding something good down that
    way), never lower it. So what the rat learned about a path over many trials
    isn't wiped out by a single sweep into a dead end.
    """

    credit_label = "swept-to hex that beat the value the rat already had"

    def action_values(self, learner, hex, neighbors, context, sweeps):
        values = []
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            cached = learner.V[context][learner.state_key(hex, neighbor)]
            values.append(max(
                [cached] + [
                    learner.V[context][learner.state_key(reached_from, swept)]
                    for sweep in along for swept, reached_from in sweep
                ]
            ))
        return np.array(values)

    def credited_hexes(self, learner, hex, neighbors, context, sweeps):
        credited = {}
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            reached = [pair for sweep in along for pair in sweep]
            if not reached:
                continue
            cached = learner.V[context][learner.state_key(hex, neighbor)]
            swept, reached_from = max(
                reached,
                key=lambda pair: learner.V[context][learner.state_key(pair[1], pair[0])],
            )
            # Only credit the swept hex if it beat the cached value
            if learner.V[context][learner.state_key(reached_from, swept)] > cached:
                credited[neighbor] = swept
        return credited


class FurthestSweptRetrieval(RetrievalRule):
    """
    A move's value is the value of where the rat's sweep down that way ended.

        value(move) = value of the last hex of the sweeps down that way

    This treats a sweep like a rollout: the rat runs the route in its head, and
    whatever it finds at the end is the value of that way. If the rat made
    several sweeps down the same way, we take the best ending. A move with no
    sweeps down it gets the value of the hex it steps into.
    """

    credit_label = "furthest swept-to hex"

    def action_values(self, learner, hex, neighbors, context, sweeps):
        values = []
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            if along:
                # The last hex of each sweep down this way, paired with the hex it
                # was reached from
                endings = [sweep[-1] for sweep in along]
                values.append(max(
                    learner.V[context][learner.state_key(reached_from, swept)]
                    for swept, reached_from in endings
                ))
            else:
                values.append(learner.V[context][learner.state_key(hex, neighbor)])
        return np.array(values)

    def credited_hexes(self, learner, hex, neighbors, context, sweeps):
        credited = {}
        for neighbor in neighbors:
            along = sweeps_along_move(learner, sweeps, hex, neighbor)
            endings = [sweep[-1] for sweep in along]
            if endings:
                swept, reached_from = max(
                    endings,
                    key=lambda pair: learner.V[context][learner.state_key(pair[1], pair[0])],
                )
                credited[neighbor] = swept
        return credited
