"""
retrieval_rules.py

How a hex maze agent works out what each move on offer is worth.

Each rule below implements one theory of what the rat consults at a choice
point. A learner holds one of them, and everything that turns values into
choices reads it: the policy, the simulated agent, and the likelihood of the
rat's observed choices.

The rules here are given the sweeps the rat made while standing on a hex (see
BaseHexLearner.choice_nll for how sweeps are supplied per trial). Each sweep is the
hexes it represented, in the order they were represented, so a sweep reads as a
route the rat ran out in its head. The hex a sweep opens with is the move it
set off down, which is how each rule knows which sweep speaks for which move,
and a rat that sweeps down both arms makes one sweep for each.

For a directional learner, each swept hex is read in the direction the sweep
arrived at it from, which is the hex before it in that same sweep.

RetrievalRule, what every rule provides, comes first, followed by
LocalRetrieval, the default, which judges a move by the hex it steps into.
"""

import numpy as np


__all__ = [
    "RetrievalRule", "LocalRetrieval",
    "BestSweptRetrieval", "BestSweptOrCachedRetrieval", "FurthestSweptRetrieval",
]


class RetrievalRule:
    """
    How the agent works out what each move on offer is worth.

    A RetrievalRule says which values the agent consults when it chooses. This
    is where sweeping enters the model: a rat representing hexes beyond its own
    location can evaluate a move by what lies down that way.

    A learner holds exactly one of these, since each candidate move needs a
    single number to be compared by the softmax.

    Sweeps are given in one of two forms, and every method here accepts both:
    a list of sweeps, each a sequence of hexes in the order they were
    represented, or one run of every hex represented while the rat stood there.
    They are None when the rat represented nothing beyond its own location.
    """

    # Names the hex credited_hexes picks out, so a plot can say what it marked.
    # A rule that credits no hex leaves this None.
    credit_label = None

    def action_values(self, learner, hex, neighbors, context, sweeps):
        """
        The value compared for each candidate move.

        Parameters:
            learner (BaseHexLearner): The learner doing the choosing
            hex (int): The hex the agent is deciding from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read
            sweeps (list[list[int]] or list[int] or None): The sweeps the rat made
                from this hex, or one run of represented hexes. None when the rat
                represented nothing beyond its own location

        Returns:
            np.ndarray: One value per neighbor, in the order given
        """
        raise NotImplementedError

    def credited_hexes(self, learner, hex, neighbors, context, sweeps):
        """
        Which represented hex each move's value was read from.

        A move valued by the hex it steps into has no entry here, so this holds
        only the moves a sweep spoke for. Used to show on the maze what the
        agent's choice rested on.

        Parameters:
            learner (BaseHexLearner): The learner doing the choosing
            hex (int): The hex the agent is deciding from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read
            sweeps (list[list[int]] or list[int] or None): The sweeps the rat made
                from this hex, or one run of represented hexes. None when the rat
                represented nothing beyond its own location

        Returns:
            dict: {move: the represented hex its value came from}
        """
        return {}


class LocalRetrieval(RetrievalRule):
    """
    The agent judges a move by the hex it would step into.

    Whatever the rat is representing elsewhere leaves the choice alone, so this
    is the agent that decides from cached value at its own location.
    """

    def action_values(self, learner, hex, neighbors, context, sweeps):
        return np.array(
            [learner.V[context][learner.state_key(hex, neighbor)] for neighbor in neighbors]
        )


def steps_out_to(beyond, neighbor, swept):
    """
    How many hexes out from a move a hex sits, following the way in.

    Parameters:
        beyond (dict): {hex: the hex it is entered from}, walking out from the move
        neighbor (int): The hex this move steps into
        swept (int): The hex to measure

    Returns:
        int: Steps from `neighbor` out to `swept`
    """
    steps = 0
    while swept != neighbor:
        swept = beyond[swept]
        steps += 1
    return steps


def sweeps_along_move(learner, sweeps, hex, neighbor):
    """
    What the rat swept down one move, as (hex, the hex it was reached from).

    Given sweeps, a sweep speaks for the move it opens with, and a sweep that
    skips hexes has the direction it arrived from filled in by walking out from
    the last hex it did catch.

    Given one run of everything the rat represented, the maze decides instead:
    a hex counts for this move if it lies beyond it, which is everything
    reachable by taking this move and carrying on outward without doubling back
    past the rat. The rat's own hex and the hexes behind it are left out by that
    walk, and the hexes that remain come back ordered by how far out they sit.
    A hex the maze lets the rat reach either way round lies beyond both moves
    and counts for both.

    Parameters:
        learner (BaseHexLearner): The learner doing the choosing
        sweeps (list[list[int]] or list[int] or None): The sweeps the rat made from
            `hex`, each a sequence of hexes in the order they were represented, or
            one run of hexes in the order it represented them
        hex (int): The hex the rat is deciding from
        neighbor (int): The hex this move would step into

    Returns:
        list[list[tuple]]: One list per sweep down this move, holding each of its
            hexes paired with the hex it was reached from
    """
    if not sweeps:
        return []

    # One run of represented hexes, with no sweeps marked out in it
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
            # A barrier hex has no value to read, and a hex represented twice in a
            # row is still the same place along the sweep
            if swept not in learner.graph or swept == previous:
                continue
            if learner.graph.has_edge(previous, swept):
                reached.append((swept, previous))
            else:
                # The sweep skips from `previous` to `swept` without the hexes
                # between, so the direction it arrived from was not recorded.
                # Walking out from `previous` gives the neighbour of `swept` that
                # lies on the way there, which is the direction it would have
                # come from had every hex been caught.
                on_the_way = learner.bfs_entry_directions(previous)
                reached.append((swept, on_the_way[swept]))
            previous = swept

        # Never empty, since a sweep down this move opens with the hex it steps into
        along.append(reached)
    return along


class BestSweptRetrieval(RetrievalRule):
    """
    A move is worth the best hex the rat swept to down that way.

        value(move) = max value among the hexes of the sweeps that set off down it

    The sweep is the whole evaluation, so sweeping only to poor hexes down one
    way makes that move worth little, whatever the rat had learned about the hex
    it would step into. A move no sweep set off down is worth that hex, which is
    what every move is worth when the rat sweeps nowhere.
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
    A move is worth the better of what the rat already knows and what it swept to.

        value(move) = max(value of the hex the move steps into,
                          max value among the hexes of the sweeps down that way)

    Sweeping can raise a move, by finding something good down that way, and
    leaves it alone otherwise, so what the rat learned about a branch over many
    trials survives a single sweep into a dead end.
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
            # Only a sweep that beat what the rat already knew is what the value rests on
            if learner.V[context][learner.state_key(reached_from, swept)] > cached:
                credited[neighbor] = swept
        return credited


class FurthestSweptRetrieval(RetrievalRule):
    """
    A move is worth where the rat's sweep down that way ended up.

        value(move) = value of the last hex of the sweeps that set off down it

    A sweep read as a rollout: the rat runs the route out in its head, and what
    it finds at the end is what that way is worth. Where the rat made several
    sweeps down the same move, the best of their endings is taken. A move no
    sweep set off down is worth the hex it steps into.
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
