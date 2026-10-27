"""
base_learner.py

A hex maze agent that learns a value for each maze hex from the paths
a rat runs, and chooses where to go next from those values.

What the agent learns is held in V, a value per state. What counts as a
state depends on how the agent is set up: one value per hex, or one value
per direction of travel through a hex (previous hex + current hex). 
When the agent is "goal-conditioned" it keeps a separate value table for 
each start port (the idea being that if the rat starts from e.g. port A,
only ports B and C are valid goals, so the hex values should reflect that.
I only really use this for "simulate" mode so the rat doesn't turn around
and run back towards the port it just left).

How the state values change is supplied by a list of UpdateRule objects (see
update_rules.py). A rule is called at three potential moments as a trial is 
replayed hex by hex: 
- once before the first step (e.g. to reset a per-trial memory trace vector)
- once per hex-to-hex transition (e.g. for a TD(lambda) update)
- once after the rat reaches the port (e.g. for a model-based or SWR update)

Each call returns a list of UpdateEvent describing which states moved and 
how to display that, which is what lets animate_learning show any rule's 
updates using only `event.changed` and `event.describe()`. This makes it
easy for us to implement new rules and animate everything to visualize/debug 
what's going on each time a value changes.

Several rules can run at once on the same value table, so a dual-process
agent is a learner holding two rules (e.g. both a model-free TD(0) update
on each hex transition plus an SWR replay update at reward).

How the agent turns values into a choice is supplied by a single RetrievalRule
(see retrieval_rules.py). By default a move is worth the value of the hex it
steps into. A retrieval rule can also read the hexes the rat swept to while
deciding, and value a move by what it found by sweeping down that way.
"""

import copy
import random
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize

from ...utils import create_empty_hex_maze, maze_to_graph
from ...core import get_safe_hex_distance, get_pairwise_distance_matrix, get_critical_choice_points
from ...utils import REWARD_PORTS, resolve_port
from ...plotting import plot_hex_maze
from .retrieval_rules import LocalRetrieval
from .update_rules import TDLambdaRule, ModelBasedRule, SWRRule


# Named outline color -> the colormap that shades a changed hex's outline by its update's
# weight (see UpdateEvent.changed and build_outlines inside animate_learning). Each runs
# dim (lowest weight) to bright (highest), so a trace decaying back along the path fades
# as it goes, and the hexes an update moved most stay legible against a maze that is dark
# wherever value is still low. An UpdateEvent names one of these.
_COLOR_TO_CMAP = {
    name: LinearSegmentedColormap.from_list("outline", [dim, bright])
    for name, (dim, bright) in {
        "red": ("#7f1d1d", "#ff8585"),
        "green": ("#14532d", "#6ee7a0"),
        "lightgreen": ("#2f7d3a", "#9dff8a"),
    }.items()
}


def represented_hexes(sweeps):
    """
    Every hex in what the rat represented, however the sweeps were given.

    Parameters:
        sweeps (list[list[int]] or list[int]): The sweeps the rat made from one
            hex, each a sequence of hexes, or one run of represented hexes

    Returns:
        set[int]: The hexes represented
    """
    if not sweeps:
        return set()
    if isinstance(sweeps[0], (list, tuple, set)):
        return {hex for sweep in sweeps for hex in sweep}
    return set(sweeps)


class BaseHexLearner:
    """
    Hex-maze value learner driven by a list of UpdateRule objects (see
    module docstring). The learner holds the value table V, walks trials hex
    by hex, and calls each rule at the start of a trial, on every step, and at
    the port. How values change is decided by the rules alone.

    Each rule's hyperparameters (e.g. TDLambdaRule's alpha/lam,
    ModelBasedRule's gamma_mb/a_T/a_mb) are passed to the constructor and
    stored as attributes of the learner, so every parameter of a model lives
    in one place, and each rule reads the ones it needs from the learner.
    `gamma` is a named argument here because it also sets how quickly prior
    values fall off with distance from a port, whichever rules are active.

    HexMazeTDLearner, KrauszDualLearner and SWRDualLearner each fix their own
    rules. Any other combination is built here directly, e.g.:

        BaseHexLearner(
            maze,
            rules=[TDLambdaRule(), ModelBasedRule()],
            alpha=0.3, gamma=0.95, lam=0.3,      # TDLambdaRule reads these
            gamma_mb=0.8, a_T=0.1, a_mb=0.05,    # ModelBasedRule reads these
            goal_conditioned_states=True,
        )
    """

    # Starting value and (lower, upper) bounds for each parameter fit_choices can fit
    _FIT_PARAM_DEFAULTS = {
        # Every learner
        "gamma": (0.9, (0.0, 0.999)), "temperature": (0.5, (0.01, 10.0)),
        # TDLambdaRule
        "alpha": (0.3, (1e-3, 1.0)), "lam": (0.3, (0.0, 1.0)),
        # ModelBasedRule
        "gamma_mb": (0.9, (0.0, 0.999)), "a_T": (0.1, (1e-3, 1.0)), "a_mb": (0.1, (1e-3, 1.0)),
        # SWRRule
        "a_swr": (0.1, (1e-3, 1.0)),
    }

    def __init__(
        self,
        maze,
        rules=None,
        gamma=0.95,
        temperature=1.0,
        directional_states=True,
        goal_conditioned_states=False,
        priors=None,
        no_backtrack=True,
        retrieval=None,
        reward_probs=None,
        **rule_kwargs,
    ):
        """
        Parameters:
            maze (set, frozenset, list, np.ndarray, str, networkx.Graph, or list of
                these): The hex maze in any valid format. A list of mazes describes a
                session with barrier changes, and the first one is what the agent
                starts on (see learn).
            rules (list of UpdateRule): The active value-update mechanisms, e.g.
                [TDLambdaRule()] or [TDLambdaRule(), ModelBasedRule()]. At each moment of a
                trial the rules are called in this order, so a later rule reads any value
                an earlier one has just changed, and their animation frames come in this
                order.
            gamma (float): Discount factor used for prior-value computation, and reused by
                any rule (like TDLambdaRule) that wants a shared discount rate.
            temperature (float): Softmax temperature for action selection (policy-level, not
                tied to any one update rule).
            directional_states (bool): If True, states are directed edges (prev_hex, cur_hex); else
                hexes.
            goal_conditioned_states (bool): If True, keep one value function per start port (3
                value tables); else a single shared value function. A learner is set up
                with exactly one of directional_states and goal_conditioned_states.
            priors (None, "uniform", ("flat", value), or list of 3 floats): Hex value
                initialization strategy (see distance_priors/build_prior_table).
            no_backtrack (bool): If True, the simulated agent avoids revisiting hexes
                within a trial where it can (see choose_action/simulate). Applies to
                simulated behavior alone, so scoring and the update rules see every
                available neighbor. Defaults to True, which keeps simulated trials
                close in length to the rats' runs. A simulated trial ends only when
                the agent reaches a port, and an agent free to turn around wanders
                back and forth far longer on the way there.
            retrieval (RetrievalRule): How the agent works out what each move on offer
                is worth. Defaults to LocalRetrieval, which judges a move by the hex it
                would step into.
            reward_probs (list of float): [p1, p2, p3] reward probability at ports 1/A,
                2/B, 3/C. Read when simulating only, to sample whether a reward is delivered.
            **rule_kwargs: Rule-specific hyperparameters (e.g. alpha=, lam= for
                TDLambdaRule; gamma_mb=, a_T=, a_mb= for ModelBasedRule), set as plain
                attributes on this instance.
        """
        # A list of mazes describes a session with barrier changes. The agent starts
        # on the first one, which is what its graph and priors are built from, and
        # learn swaps in the later ones. A list of plain ints is a single maze,
        # given as its barrier hexes.
        if isinstance(maze, (list, tuple)) and not all(
            isinstance(hex, (int, np.integer)) for hex in maze
        ):
            maze = maze[0]

        self.graph = maze_to_graph(maze)
        
        # A learner with no rules just holds its priors and never updates them
        self.rules = [] if rules is None else rules
        self.gamma = gamma
        self.temperature = temperature
        self.directional_states = directional_states
        self.goal_conditioned_states = goal_conditioned_states
        self.priors = priors
        
        # Reward probabilities and no backtrack are used when simulating only
        self.reward_probs = {} if reward_probs is None else {i + 1: reward_probs[i] for i in range(3)}
        self.no_backtrack = no_backtrack
        self.retrieval = LocalRetrieval() if retrieval is None else retrieval

        for key, value in rule_kwargs.items():
            setattr(self, key, value)

        # Contexts: one value table per start port, or a single shared one.
        self.contexts = list(REWARD_PORTS) if goal_conditioned_states else [None]

        self.prior_table = None  # {context: {hex: value}} or None
        self._hex_distances = None
        self._choice_points = {}
        self.V = {}
        self.reset()

    #  Setup / priors

    def hex_distances(self):
        """
        The number of hexes between every pair of open hexes in this maze.

        Built the first time it is needed and kept until the maze changes.
        SWRRule reads it to find which directions of travel through a replayed
        hex lead toward the port.

        Returns:
            pd.DataFrame: Distance matrix indexed by hex
        """
        if self._hex_distances is None:
            self._hex_distances = get_pairwise_distance_matrix(self.graph)
        return self._hex_distances

    def critical_choice_points(self, start_port=None):
        """
        The hexes where the ways to this maze's reward ports part, cached per maze.

        Setting out from a port, this is where the shortest paths to the other two
        ports stop sharing hexes, so it is the hex at which the rat commits to one
        port over the other. Every other 3-way junction on the way is a branch that
        rejoins, where the two ways lead to the same place.

        Parameters:
            start_port (int or str, optional): Which port the rat set out from
                (1/2/3 or A/B/C). None gives the choice points from every port

        Returns:
            set[int]: The hexes that are critical choice points
        """
        if start_port not in self._choice_points:
            self._choice_points[start_port] = get_critical_choice_points(self.graph, start_port)
        return self._choice_points[start_port]

    def all_states(self):
        """
        Every state this learner holds a value for.

        For a learner keeping one value per hex, that is each open hex. For a
        directional learner, whose states are directed edges, it is each way of
        arriving at a hex, plus arriving at the hex a trip starts from, which is
        written (None, hex).

        Returns:
            list: Every state in this learner's representation
        """
        if not self.directional_states:
            return list(self.graph.nodes())

        # The state a trip is in before the rat has moved anywhere
        states = [(None, hex) for hex in self.graph.nodes()]

        # Every way of arriving at a hex from one of its neighbors
        states += [
            (prev, cur) for cur in self.graph.nodes() for prev in self.graph.neighbors(cur)
        ]
        return states

    def reset(self):
        """Clear all value tables and start every state at its prior value."""
        self.V = {context: {} for context in self.contexts}
        self.build_prior_table(self.priors)

        # No trial has been walked yet, so nothing has been replayed
        self.replayed_hexes_this_trial = []

        # Every state begins at what the rat is assumed to expect there before
        # any experience of this maze, which is 0 when no priors were given
        for context in self.contexts:
            for state in self.all_states():
                self.V[context][state] = self.prior_for_hex(context, self.hex_of_state(state))
        # Only the session's very first trial may lack a real start port.
        self._is_first_trial = True

        for rule in self.rules:
            rule.on_reset(self)

    def build_prior_table(self, priors):
        """Precompute {context: {hex: prior value}} (or None for all-zeros)."""
        # Every hex starts at 0, so the agent has learned nothing about the maze
        if priors is None:
            self.prior_table = None

        # Ports hexes start at 50% probability of reward, decaying with (empty maze)
        # distance from the ports
        # IMO, this is closest to the "true" priors the rats have: they know where the
        # ports are and to expect reward at ports, but they don't yet know which ports 
        # are better or worse (all start at 50%) and which paths to take (value decays
        # using empty maze distances, because rats haven't seen where the paths and
        # dead ends are yet)
        elif priors == "uniform":
            self.prior_table = self.distance_priors([0.5, 0.5, 0.5])

        # Every hex starts at the same value, with no sense of where ports are
        # This matches Tim's paper (all states initialized to 0.2)
        # Every hex of the empty maze gets a prior, so a hex that a later barrier
        # change opens up has one too
        elif isinstance(priors, tuple) and len(priors) == 2 and priors[0] == "flat":
            value = priors[1]
            self.prior_table = {
                context: {hex: value for hex in create_empty_hex_maze().nodes()}
                for context in self.contexts
            }

        # Port hexes start at given probabilities of reward, decaying with (empty maze)
        # distance from the ports. We don't really use this one (the rats should not have
        # prior knowledge of what ports are more rewarding) but it is sometimes useful for
        # simulate mode (starting with well-learned values and seeing how long it takes to
        # update them after a probability shift)
        elif isinstance(priors, (list, tuple)) and len(priors) == 3:
            self.prior_table = self.distance_priors(list(priors))
        else:
            raise ValueError(
                f"priors must be None, 'uniform', ('flat', value), or [p1, p2, p3], got {priors!r}"
            )

    def distance_priors(self, port_values):
        """
        Build distance-discounted priors per context.

        For each context, value(hex) = max over that context's goal ports of
        port_value[goal] * gamma^(distance from hex to goal), using the empty maze
        for distances. A context's goals are all ports except its key (the start
        port); a non-goal-conditioned (None) context uses all three ports as goals.
        """
        # Distances are measured on the maze with no barriers, so these priors
        # describe what the rat expects from general knowledge of the maze task
        # (e.g. reward ports are at the 3 corners) before learning which paths 
        # this particular barrier set leaves open
        empty_maze = create_empty_hex_maze()
        port_value = {i + 1: port_values[i] for i in range(3)}

        table = {}
        for context in self.contexts:
            # Context is the start port for the trial, because the rat cannot get 
            # reward at the port it just visited, so the other two ports are the 
            # only valid goals. 
            # A learner keeping a single shared table (context=None) counts all 
            # three ports as potential goals.
            goals = [port for port in REWARD_PORTS if port != context]
            table[context] = {}

            # Every hex of the empty maze gets a prior, so a hex that a later
            # barrier change opens up has one too
            for hex in empty_maze.nodes():
                # The prior for the port hex is the expected p(reward) at that port
                if hex in goals:
                    table[context][hex] = port_value[hex]
                else:
                    # Every other hex is worth the best port it can reach,
                    # discounted by how many hexes away that port is:
                    #     prior(hex) = max over goals of  p(goal) * gamma ^ distance(hex, goal)
                    table[context][hex] = max(
                        (port_value[goal] * (self.gamma ** get_safe_hex_distance(empty_maze, start_hex=hex, target_hex=goal))
                         for goal in goals),
                        default=0.0,
                    )
        return table

    def prior_for_hex(self, context, hex):
        """Prior value for a hex in a context (0.0 when no priors set)."""
        if self.prior_table is None:
            return 0.0
        return self.prior_table[context].get(hex, 0.0)

    def set_graph(self, new_maze):
        """
        Swap the maze (e.g. after a barrier change). Accepts a maze in any valid
        format (barrier set, string, networkx graph, etc.), converted via
        ``maze_to_graph``. States the new maze no longer has are dropped from V,
        states it opens up are added at their prior value, and each rule drops
        what it has learned about the removed states too (see
        UpdateRule.on_maze_changed).
        
        TODO: How do we want to deal with the values of newly open states
        (on barrier change)? Currently they are set to whatever the prior would
        have been. 
        """
        self.graph = maze_to_graph(new_maze)

        # Distances and choice points belong to the old maze
        self._hex_distances = None
        self._choice_points = {}

        # The states the new maze allows. A value learned for a hex that is now
        # a barrier is dropped, so if that hex reopens later the agent has to
        # learn about it again, and a hex the new maze opens up starts at its
        # prior the same way it would at the beginning of a session.
        valid = set(self.all_states())
        for context in self.contexts:
            for state in list(self.V[context]):
                if state not in valid:
                    del self.V[context][state]
            for state in valid:
                if state not in self.V[context]:
                    self.V[context][state] = self.prior_for_hex(
                        context, self.hex_of_state(state)
                    )

        # Each rule drops whatever it has learned about those hexes as well
        for rule in self.rules:
            rule.on_maze_changed(self)

    #  State helpers

    def context_for_port(self, start_port):
        """The value table key for a trip starting at start_port."""
        return start_port if self.goal_conditioned_states else None

    def state_key(self, prev_hex, cur_hex):
        """State key for arriving at `cur_hex` from `prev_hex` (None at trip start)."""
        return (prev_hex, cur_hex) if self.directional_states else cur_hex

    def hex_of_state(self, state):
        """The maze hex underlying a state key."""
        return state[1] if self.directional_states else state

    def format_state(self, state):
        """Human-readable label for a state key, e.g. '48→43', 'start→43' or '43'."""
        if self.directional_states:
            prev, cur = state
            return f"{prev if prev is not None else 'start'}→{cur}"
        return str(state)

    #  Rule-driven learning

    def maze_schedule(self, trajectories, mazes=None, trials_per_block=None):
        """
        Which maze each trial is run on, as a list with one entry per trial.

        Only the trial where a barrier change happens carries a maze. Every other
        trial is None, meaning the maze the rat is already running.

        Parameters:
            trajectories (list of list of int): The session's trials
            mazes (list): The maze for each block of trials, in order
            trials_per_block (list of int): How many trials are run on each maze

        Returns:
            list: One entry per trial, with a maze on the first trial of each block 
                and None elsewhere
        """
        schedule = [None] * len(trajectories)
        if mazes is None:
            return schedule

        trial = 0
        for maze, block_length in zip(mazes, trials_per_block):
            schedule[trial] = maze
            trial += block_length
        return schedule

    #  One trial, moment by moment
    #
    # A trial is learned as it happens: start_trial before the first move,
    # learn_step after each hex-to-hex move, and end_trial once the rat is at the
    # port. Anything that reads values to make or score a choice does so between
    # these calls, so every choice is made from the maze as it stands at that moment.

    def start_trial(self, context, replayed_hexes=None):
        """
        Get every rule ready for a new trial.

        Parameters:
            context (int or None): The value table this trial updates, meaning the
                port the trial started from, or None for a single shared table
            replayed_hexes (list[int]): The hexes replayed during this trial's
                sharp-wave ripples, for a learner holding an SWRRule
        """
        # What was replayed this trial, which SWRRule reads when the rat
        # reaches the port
        self.replayed_hexes_this_trial = replayed_hexes or []

        # Let each rule clear whatever it accumulates within a trial
        for rule in self.rules:
            rule.on_trial_start(self, context)

    def is_possible_step(self, prev_hex, cur_hex, next_hex):
        """
        Whether a hex-to-hex move from the data could have happened in this maze.

        A move through a barrier, or between two hexes that do not border each
        other, is a position tracking error. It is warned about and left out of
        both learning and scoring.

        Parameters:
            prev_hex (int or None): The hex the rat arrived at cur_hex from, or None
                on a trial's first step
            cur_hex (int): The hex the rat is leaving
            next_hex (int): The hex the rat is entering

        Returns:
            bool: True if the move is possible
        """
        # A hex that is a barrier in this maze cannot have been run through,
        # so it is a position tracking error (or a trial run against the wrong maze!)
        if cur_hex not in self.graph or next_hex not in self.graph:
            missing = [hex for hex in (cur_hex, next_hex) if hex not in self.graph]
            warnings.warn(
                f"hex {missing} is a barrier in this maze! "
                f"This is likely a position tracking error. Skipping this step."
            )
            return False

        # A move between two hexes that do not border each other cannot
        # have happened, so it is a position tracking error
        if not self.graph.has_edge(cur_hex, next_hex):
            warnings.warn(
                f"at hex {cur_hex}, next_hex={next_hex} is not a "
                f"graph neighbor ({sorted(self.graph.neighbors(cur_hex))})! "
                f"This is likely a position tracking error. Skipping this step."
            )
            return False

        # A directional learner's state also records which hex the rat came
        # from, so a tracking error on the way in leaves no state to update
        if self.directional_states and prev_hex is not None and not self.graph.has_edge(prev_hex, cur_hex):
            warnings.warn(
                f"arrived at hex {cur_hex} from hex {prev_hex}, which is not "
                f"a graph neighbor. This is likely a position tracking error. "
                f"Skipping this step."
            )
            return False
        return True

    def learn_step(self, prev_hex, cur_hex, next_hex, context, reward=None):
        """
        Update values for one hex-to-hex move, with every active rule.

        Parameters:
            prev_hex (int or None): The hex the rat arrived at cur_hex from, or None
                on a trial's first step
            cur_hex (int): The hex the rat is leaving
            next_hex (int): The hex the rat is entering
            context (int or None): The value table to update
            reward (float or None): The reward delivered, given only on the move
                that arrives at the reward port. None on every other move

        Returns:
            list[UpdateEvent]: The updates this move caused, across every rule
        """
        # The states being left and entered, which are hexes for a learner
        # holding one value per hex, and directed edges for a directional one
        state = self.state_key(prev_hex, cur_hex)
        next_state = self.state_key(cur_hex, next_hex)

        # A reward is given only on the move that arrives at the reward port
        is_terminal = reward is not None

        # Every rule sees every transition, so several mechanisms can learn
        # from the same step
        events = []
        for rule in self.rules:
            events += rule.on_step(
                self, context, state, next_state, reward if is_terminal else 0.0, is_terminal
            )
        return events

    def end_trial(self, path, reward, context):
        """
        Let every rule learn from the trial as a whole, once the rat is at the port.

        Parameters:
            path (list[int]): The hexes the rat ran this trial, ending at the port
            reward (float): The reward delivered at that port
            context (int or None): The value table to update

        Returns:
            list[UpdateEvent]: The updates made at the port, across every rule
        """
        # Rules that learn from the trial as a whole, such as a replay or
        # model-based update, update once the rat has reached the port
        events = []
        for rule in self.rules:
            events += rule.on_trial_end(self, path, reward, context)
        return events

    def record_events(self, history, path, step_index, events):
        """
        Add one snapshot per update to `history`, each carrying its UpdateEvent
        under "event", so two updates on the same step (two rules, or
        TD(lambda)'s step update followed by its reward update) each get their
        own animation frame and description. Every snapshot from one step shows
        the maze after all of that step's updates.

        Parameters:
            history (list): The snapshots so far, added to in place
            path (list[int]): The trial's path, at least up to the rat's hex
            step_index (int): Where along the path the rat is (see snapshot)
            events (list[UpdateEvent]): The updates to record
        """
        # The snapshot before a trial's first move carries no event (None)
        for event in events:
            snap = self.snapshot(path, step_index)
            snap["event"] = event
            history.append(snap)

    def learn_path(self, path, reward, context, replayed_hexes=None, record=False,
                   before_step=None):
        """
        Walk a single recorded path within one context, updating values after
        each move in the order the rat ran them. Reward is delivered at the
        terminal state (path[-1]).

        Parameters:
            path (list[int]): The hexes the rat ran this trial, in order, ending
                at the reward port
            reward (float): The reward delivered at that port
            context (int or None): The value table to update, meaning the port
                the trial started from, or None for a single shared table
            replayed_hexes (list[int]): The hexes replayed during this trial's
                sharp-wave ripples, for a learner holding an SWRRule
            record (bool): Collect a snapshot of the maze's values after every
                update, for animating how learning unfolded (see record_events).
                The first snapshot is the maze before the trial, with "event" None
            before_step (callable, optional): Called as before_step(step) just
                before the move from path[step] to path[step + 1] is learned from,
                for every move that could have happened. This is where a choice
                is scored, against the values in force when the rat made it

        Returns:
            list or None: One snapshot per update when record=True
        """
        self.start_trial(context, replayed_hexes)

        history = []
        if record:
            # The maze as it stood before this trial changed anything
            self.record_events(history, path, 0, [None])

        # Walk the trial one hex-to-hex transition at a time, in the order the
        # rat ran them, so values evolve exactly as they would have in the session
        for step in range(len(path) - 1):
            # The hex before the current one, which is what makes a state
            # directional. There is no previous hex on the first step.
            prev_hex = path[step - 1] if step > 0 else None
            cur_hex, next_hex = path[step], path[step + 1]

            # A tracking error is warned about and skipped
            if not self.is_possible_step(prev_hex, cur_hex, next_hex):
                continue

            # The values here are the ones in force as the rat made this move
            if before_step is not None:
                before_step(step)

            # The reward is delivered on the move that arrives at the reward port
            is_terminal = step == len(path) - 2
            events = self.learn_step(
                prev_hex, cur_hex, next_hex, context, reward if is_terminal else None
            )
            if record:
                self.record_events(history, path, step + 1, events)

        events = self.end_trial(path, reward, context)
        if record:
            self.record_events(history, path, len(path) - 1, events)

        return history if record else None

    #  Learn from supplied trajectories

    def resolve_context(self, path):
        """
        The value-table context for a trip, derived from path[0].

        With goal_conditioned_states=False there is a single shared table, so
        path[0] never needs to be a real port. With goal_conditioned_states=True,
        the context is path[0] if it's a reward port. If not, a placeholder
        is only allowed on the session's very first trial (which starts
        wherever the rat happened to be, before ever reaching a port): one
        of the other two ports, excluding the trip's end port (path[-1]).
        Any later trial without a real start port raises, since every trial
        after the first should start where the previous one ended.
        """
        is_first_trial = self._is_first_trial
        self._is_first_trial = False

        if not self.goal_conditioned_states:
            return None
        if path[0] in REWARD_PORTS:
            return path[0]
        if not is_first_trial:
            raise ValueError(
                f"path[0]={path[0]} is not a reward port, and this isn't the "
                f"session's first trial (a placeholder start port is only "
                f"allowed once). Check that trajectories are passed in "
                f"session order and that each trial's path starts where the "
                f"previous one ended."
            )
        placeholder = next(port for port in REWARD_PORTS if port != path[-1])
        warnings.warn(
            f"path[0]={path[0]} is not a reward port on the session's first "
            f"trial; using placeholder start port {placeholder}."
        )
        return placeholder

    def learn(self, trajectories, rewards, replayed_hexes=None, mazes=None,
              trials_per_block=None, record=None):
        """
        Run updates over a batch of externally-provided trajectories.

        Parameters:
            trajectories (list of list of int): Hex path for each trial [s0, s1, ...,
                s_terminal]. The first and last hex of each trial should be a reward port
                (except for the very first trial, which is allowed to not start at a port)
            rewards (list of float): Reward for each trajectory.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, one list per trial, in the same order as `trajectories`. Used
                by a learner holding an SWRRule
            mazes (list, optional): The maze for each block of trials, for a session
                with barrier changes. Learned values carry across a change: hexes that
                become barriers are dropped, and hexes the new maze opens start at
                their prior
            trials_per_block (list of int, optional): How many trials are run on each
                maze in `mazes`
            record ({None, "trial", "step"}, optional): Snapshot granularity to return.
                Each snapshot is the same shape as snapshot(); collapse one with
                snapshot_values()
                - None (default): no return value, just runs the updates
                - "trial": one snapshot per trial, taken after that trial's last event
                - "step": every event's snapshot, across every trial

        Returns:
            None, or a list of snapshots, depending on `record`.
        """
        if record not in (None, "trial", "step"):
            raise ValueError(f"record must be None, 'trial', or 'step', got {record!r}")

        # With no replay given, every trial replays nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        history = []
        for path, reward, trial_replay, trial_maze in zip(
            trajectories, rewards, replayed_hexes, maze_schedule
        ):
            # A barrier change swaps the maze in before this trial is learned
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            trial_history = self.learn_path(
                path, reward, self.resolve_context(path), trial_replay, record=bool(record)
            )
            if record == "step":
                history.extend(trial_history)
            elif record == "trial":
                history.append(trial_history[-1])

        return history if record else None

    #  Reading values along a timeseries

    def value_timeseries(self, trajectories, rewards, sampled_hexes, represented_hexes=None,
                         replayed_hexes=None, mazes=None, trials_per_block=None, values="trial"):
        """
        The value of where the rat is, and of what it is representing, sample by sample.

        Walks the session exactly as `learn` does, and reads a value off the maze for
        every sample of a finer-grained timeseries: `sampled_hexes[trial][i]` is the
        hex the rat was in at sample i of that trial, and `represented_hexes[trial][i]`
        is the hex it was representing then. Both are read from the same value table,
        so the two are directly comparable.

        `values` picks the table a sample is read from:

            "trial"  the maze as the trial began, so a hex's value holds still within
                     a trial and what the rat is running toward is the only thing
                     moving the timeseries.
            "step"   the maze as it stood while the rat was in that hex, so updates
                     that already fired this trial are in it. A hex's value then jumps
                     at the moment the rat leaves it, which is that step's own update
                     and says nothing about where the rat went.

        For a directional learner, each hex is valued in the direction the rat would
        enter it (see resolve_snapshot_entry_directions), which is the same view
        animate_learning colors the maze by. Under "trial" that is walking out from
        where the trial began, and under "step" from where the rat is standing.

        Parameters:
            trajectories (list of list of int): Hex path for each trial (see learn)
            rewards (list of float): Reward for each trial
            sampled_hexes (list[list[int]]): The hex the rat was in at each sample of
                each trial, one list per trial, in the same order as `trajectories`
            represented_hexes (list[list[int]], optional): The hex the rat was
                representing at each sample, one list per trial, each the same length
                as that trial's `sampled_hexes`. A sample with nothing decoded is None
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial (see learn)
            mazes (list, optional): The maze for each block of trials, for a session
                with barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            values ({"trial", "step"}): Which value table each sample is read from, as above

        Returns:
            pd.DataFrame: One row per sample, with the trial it belongs to, the sample's
                index within that trial, how far along the path the rat was, the hex it
                was in and that hex's value, and the hex it was representing and that
                hex's value. A hex that is a barrier in this maze has no value, and comes
                back as nan
        """
        # With no replay given, every trial replays nothing, and with nothing
        # decoded, every sample represents nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]
        if represented_hexes is None:
            represented_hexes = [None for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        rows = []
        for trial, (path, reward, samples, represented, trial_replay, trial_maze) in enumerate(zip(
            trajectories, rewards, sampled_hexes, represented_hexes, replayed_hexes, maze_schedule
        )):
            # A barrier change swaps the maze in before this trial is walked
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue

            context = self.resolve_context(path)

            # How far along the path the rat was at each sample. A trajectory is the
            # samples with consecutive repeats collapsed, so the rat has taken a step
            # wherever a sample moves on to the next hex of the path. A sample that
            # matches neither the hex the rat is on nor the next one is a tracking
            # error, and stays on the step it was on.
            steps, step = [], 0
            for hex in samples:
                if step + 1 < len(path) and hex == path[step + 1]:
                    step += 1
                steps.append(step)

            if represented is None:
                represented = [None] * len(samples)

            if values == "trial":
                # The maze as it stood before this trial changed anything, which
                # every sample of the trial is then read from
                table = self.snapshot(path, 0)["values"][context]
                self.learn_path(path, reward, context, trial_replay)
                table_for_step = [table] * len(path)
            elif values == "step":
                history = self.learn_path(path, reward, context, trial_replay, record=True)

                # The table in force while the rat stood on each hex of the path,
                # which is the last snapshot taken as it arrived there. A step where
                # no rule changed anything keeps the table from the step before
                table_at_step = {snap["step"]: snap["values"][context] for snap in history}
                table_for_step, table = [], history[0]["values"][context]
                for step in range(len(path)):
                    table = table_at_step.get(step, table)
                    table_for_step.append(table)
            else:
                raise ValueError(f'values must be "trial" or "step", got {values!r}')

            for sample, (hex, represented_hex, step) in enumerate(zip(samples, represented, steps)):
                table = table_for_step[step]
                rows.append({
                    "trial": trial,
                    "sample": sample,
                    "step": step,
                    "hex": hex,
                    "value": table.get(hex, float("nan")),
                    "represented_hex": represented_hex,
                    "represented_value": table.get(represented_hex, float("nan")),
                })

        # A sample with nothing decoded leaves represented_hex empty, which would
        # otherwise read the hex columns as floats
        return pd.DataFrame(rows).astype({"hex": "Int64", "represented_hex": "Int64"})

    #  Fitting to observed choices

    def junction_candidates(self, cur_hex, entry_hex):
        """
        The two non-backward neighbors at cur_hex if it's a 3-way junction 
        (aka a hex with 3 neighbors: back, left, and right).
        If the hex is a hex on a straight path, this returns None. 
        
        entry_hex (the hex arrived from, or None at a trial's first hex) is 
        excluded as a candidate because we assume the rat is making a left/right
        choice between the 2 forward options. See choice_nll for details.
        """
        all_neighbors = list(self.graph.neighbors(cur_hex))
        if len(all_neighbors) != 3:
            return None
        # A trial that starts on a junction (only possible for the session's first
        # trial) has no way back, so all three neighbors are candidates
        return [n for n in all_neighbors if n != entry_hex]

    def junction_choice_info(self, cur_hex, entry_hex, next_hex, context, sweeps=None):
        """
        Binary junction-choice info for one step, matching choice_nll's
        score_at="junctions" scoring.

        Returns None if cur_hex isn't a real 3-way junction, or next_hex is
        entry_hex, meaning the rat backtracked. (This is valid, just not a binary
        left/right choice, so we follow Tim's paper in not scoring it).
        Also returns None if next_hex isn't one of the junction candidates at all,
        which is a position tracking error.
        Otherwise returns:
        {"candidates": [hexA, hexB],
        "values": {hex: the value the retrieval rule gave that move},
        "credited": {hex: the swept hex its value came from},
        "probabilities": {hex: p},
        "choice": next_hex,
        "probability": p_choice}.
        """
        candidates = self.junction_candidates(cur_hex, entry_hex)
        if candidates is None or next_hex == entry_hex or next_hex not in candidates:
            return None
        probabilities = self.softmax_probabilities(cur_hex, candidates, context, sweeps)
        probability_by_neighbor = dict(zip(candidates, probabilities.tolist()))

        # What the agent compared, which is what the probabilities came from. With
        # a sweep down one of these ways, that is what the sweep found there.
        values = dict(zip(
            candidates, self.retrieval.action_values(self, cur_hex, candidates, context, sweeps)
        ))

        # Where each of those values was read from, which is a swept hex only for
        # the ways the retrieval rule credited a sweep (see RetrievalRule.credited_hexes)
        credited = self.retrieval.credited_hexes(self, cur_hex, candidates, context, sweeps)
        return {
            "candidates": candidates,
            "values": values,
            "credited": credited,
            "probabilities": probability_by_neighbor,
            "choice": next_hex,
            "probability": probability_by_neighbor[next_hex],
        }

    def choice_nll(self, trajectories, rewards, replayed_hexes=None, swept_hexes=None,
                   mazes=None, trials_per_block=None, record=False, score_at="steps"):
        """
        Negative log-likelihood of the rat's hex-to-hex choices under this
        model's current parameters.

        Walks each trajectory move by move. Each move is scored by the softmax
        probability of the hex the rat actually stepped to, using the values as
        they stand at that moment, and then learned from, so every later choice
        (including later in the same trial) is scored against updated values.

        Parameters:
            trajectories (list of list of int): Hex path for each trial [s0, s1, ...,
                s_terminal]. The first and last hex of each trial should be a reward port
                (except for the very first trial, which is allowed to not start at a port)
            rewards (list of float): Reward for each trajectory.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, one list per trial, in the same order as `trajectories`. Used
                by a learner holding an SWRRule
            swept_hexes (list[list[list[list[int]]]], optional): The sweeps the rat made
                at each point along each trial: one entry per hex of that trial's path,
                holding the sweeps made while standing there, each a sequence of hexes
                in the order they were represented (or one run of represented hexes,
                see RetrievalRule). Read by a learner whose retrieval rule uses sweeps
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            record (bool, optional): If True, also return a list of per-choice records, one
                per scored hex-to-hex move, in trajectory order: {"trial": index into
                trajectories/rewards, "entry": prev_hex (None at a trial's first step),
                "hex": cur_hex, "choice": next_hex, "probability": p_choice,
                "probabilities": {neighbor: prob, ...}}.
            score_at ({"steps", "junctions", "choice_points"}): Which of the rat's moves
                are scored, from every move to only the ones that decide a trial
                - "steps" (default): every step, against ALL of cur_hex's graph
                  neighbors, backward included
                - "junctions": genuine binary choices, meaning steps where the rat is at a
                  real 3-way intersection (cur_hex has exactly 3 graph neighbors) and exits
                  through one of the two non-backward neighbors (left/right). A junction
                  where the rat instead backtracked, and every non-junction step, is
                  skipped silently
                - "choice_points": those binary choices at the critical choice point of the
                  port the rat set out from (see critical_choice_points), which is the one
                  hex of the trip where the two remaining ports part ways. Every junction
                  whose ways rejoin is skipped

        Returns:
            float, or (float, list of dict) if record=True: Total negative log-likelihood of
                the scored hex choices, and (if record=True) the per-choice records.
        """
        # With no replay given, every trial replays nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        # With no sweeps given, the rat represents nothing beyond its own location
        if swept_hexes is None:
            swept_hexes = [None] * len(trajectories)

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        # -log p(choice) for every scored choice, summed at the end
        neg_log_likelihoods = []
        choices = [] if record else None
        for trial_index, (path, reward, trial_replay, trial_sweeps, trial_maze) in enumerate(
            zip(trajectories, rewards, replayed_hexes, swept_hexes, maze_schedule)
        ):
            # A barrier change swaps the maze in before this trial is scored
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            context = self.resolve_context(path)

            # Scoring only choice points needs the hexes where the ways to this trip's two
            # remaining ports part. A trial that does not start at a port (the session's
            # first) is scored against the choice points of every port
            choice_points = self.critical_choice_points(
                path[0] if path[0] in REWARD_PORTS else None
            ) if score_at == "choice_points" else set()

            def score_choice(step):
                """Score the move from path[step] to path[step + 1], against the values
                as they stand when the rat makes it. A move that is a tracking error
                never reaches here (see learn_path)."""
                cur_hex, next_hex = path[step], path[step + 1]
                entry_hex = path[step - 1] if step > 0 else None

                if score_at in ("junctions", "choice_points"):
                    if score_at == "choice_points" and cur_hex not in choice_points:
                        return  # a junction whose ways rejoin (outside this analysis)

                    candidates = self.junction_candidates(cur_hex, entry_hex)
                    if candidates is None:
                        return  # not a junction (outside this analysis)
                    if next_hex == entry_hex:
                        return  # backtracked at the junction (valid, just not a binary L/R choice)
                elif score_at == "steps":
                    candidates = list(self.graph.neighbors(cur_hex))
                else:
                    raise ValueError(
                        f'score_at must be "steps", "junctions" or "choice_points", got {score_at!r}'
                    )

                # What the rat was representing while standing on this hex
                step_sweeps = trial_sweeps[step] if trial_sweeps is not None else None
                probabilities = self.softmax_probabilities(
                    cur_hex, candidates, context, step_sweeps
                )
                probability_by_neighbor = dict(zip(candidates, probabilities.tolist()))
                p_choice = probability_by_neighbor[next_hex]

                # A probability of exactly 0 would make the likelihood infinite, so
                # it is floored at 1e-10
                neg_log_likelihoods.append(-np.log(max(p_choice, 1e-10)))
                if record:
                    choices.append({
                        "trial": trial_index,
                        "entry": entry_hex,
                        "hex": cur_hex,
                        "choice": next_hex,
                        "probability": p_choice,
                        "probabilities": probability_by_neighbor,
                    })

            # Each choice is scored just before the rat's move updates the values,
            # so it is scored against what the rat had learned up to that moment
            self.learn_path(path, reward, context, trial_replay, before_step=score_choice)

        total = float(np.sum(neg_log_likelihoods))
        return (total, choices) if record else total

    @classmethod
    def fit_choices(cls, maze, trajectories, rewards, reward_probs=None,
                     alpha=None, gamma=None, lam=None, temperature=None,
                     gamma_mb=None, a_T=None, a_mb=None, a_swr=None, replayed_hexes=None,
                     swept_hexes=None,
                     mazes=None, trials_per_block=None,
                     rules=None, score_at="steps", **kwargs):
        """
        Fit hyperparameters to maximize the likelihood of the rat's
        hex-to-hex choices.

        Which parameters are actually fit depends on which rule types are in
        `rules`: alpha/lam only matter (and are only fit) if a TDLambdaRule
        is present; gamma_mb/a_T/a_mb only if a ModelBasedRule is present;
        a_swr only if an SWRRule is present. gamma and temperature are always
        fit, since gamma also sets how prior values fall off with distance and
        temperature sets how the values become choices whichever value-update
        rules are active. A parameter with no effect on the likelihood (e.g.
        gamma_mb for a TD-only agent) is left out, so it neither costs the
        optimizer effort nor counts against the model in BIC.

        Parameters:
            maze (see __init__.):
            trajectories (list of list of int):
            rewards (list of float):
            reward_probs (see __init__.):
            gamma, temperature (float or None): Every learner's parameters (see above).
                A value fixes the parameter at that value; None (default) fits it.
            alpha, lam (float or None): TDLambdaRule's parameters. Same fix-or-fit
                convention, fit only if a TDLambdaRule is present.
            gamma_mb, a_T, a_mb (float or None): ModelBasedRule's parameters. Same
                fix-or-fit convention.
            a_swr (float or None): SWRRule's learning rate for replayed hexes. Same
                fix-or-fit convention.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, passed to choice_nll. Used by a learner holding an SWRRule
            swept_hexes (list[list[list[list[int]]]], optional): The sweeps the rat made
                at each point along each trial (see choice_nll), passed to choice_nll
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            rules (list of UpdateRule or None): The rules to fit with. None (default) uses
                whichever rules the class itself builds, so HexMazeTDLearner fits TD
                parameters and KrauszDualLearner fits the model-based ones too;
                BaseHexLearner requires them to be passed. If given, a fresh deep copy is
                built for every candidate parameter set evaluated during optimization (and
                for the final returned instance), so no rule's internal state (e.g.
                ModelBasedRule's persistent T map) leaks between evaluations.
            score_at ({"steps", "junctions", "choice_points"}): Which of the rat's moves
                are scored, passed to choice_nll.
            **kwargs: Extra constructor flags held fixed during fitting (e.g. directional_states,
                goal_conditioned_states, priors, no_backtrack).

        Returns:
            BaseHexLearner subclass instance: Fresh instance built with the best-fit (and
                any fixed) parameters (and the fixed **kwargs), carrying:
                - choice_nll_ : choice NLL at optimum
                - choice_bic_ : BIC, counting only the *fitted* params and the choices
                  actually scored
                - choices_scored_ : how many of the rat's choices were scored, which is
                  what turns the summed nll into a per-choice one
                - choice_result_ : raw scipy OptimizeResult, or None if every parameter
                  was fixed
                - score_at_ : the score_at value used to fit
        """
        # A learner class that fixes its own rules (HexMazeTDLearner,
        # KrauszDualLearner) knows them already, so build one and read them off
        # it. BaseHexLearner takes its rules as an argument.
        resolved_rules = rules if rules is not None else cls(maze, reward_probs=reward_probs, **kwargs).rules
        has_td = any(isinstance(rule, TDLambdaRule) for rule in resolved_rules)
        has_mb = any(isinstance(rule, ModelBasedRule) for rule in resolved_rules)
        has_swr = any(isinstance(rule, SWRRule) for rule in resolved_rules)

        candidates = {"gamma": gamma, "temperature": temperature}
        if has_td:
            candidates["alpha"] = alpha
            candidates["lam"] = lam
        if has_mb:
            candidates["gamma_mb"] = gamma_mb
            candidates["a_T"] = a_T
            candidates["a_mb"] = a_mb
        if has_swr:
            candidates["a_swr"] = a_swr

        fixed = {name: value for name, value in candidates.items() if value is not None}
        free_names = [name for name, value in candidates.items() if value is None]

        def _build(free_values):
            params = dict(fixed)
            params.update(zip(free_names, free_values))
            if rules is None:
                # The class builds its own fresh rules on every call
                return cls(maze, reward_probs=reward_probs, **params, **kwargs)
            return cls(
                maze, rules=copy.deepcopy(rules), reward_probs=reward_probs, **params, **kwargs
            )

        if free_names:
            x0 = [cls._FIT_PARAM_DEFAULTS[name][0] for name in free_names]
            bounds = [cls._FIT_PARAM_DEFAULTS[name][1] for name in free_names]
            result = minimize(
                lambda x: _build(x).choice_nll(
                    trajectories, rewards, replayed_hexes, swept_hexes, mazes, trials_per_block,
                    score_at=score_at,
                ),
                x0=x0, bounds=bounds, method='L-BFGS-B',
            )
            final_values = result.x
        else:
            # Nothing to fit (every parameter was pinned)
            result = None
            final_values = []

        # One evaluation with record=True gives both the NLL and the number of
        # choices actually scored, which BIC needs (scoring only junctions or
        # choice points scores far fewer choices than there are hex-to-hex steps).
        # The learner that walked the session is then discarded, and the one
        # returned is a fresh, untrained learner at the same parameters.
        nll, choices = _build(final_values).choice_nll(
            trajectories, rewards, replayed_hexes, swept_hexes, mazes, trials_per_block,
            record=True, score_at=score_at,
        )
        fitted = _build(final_values)
        fitted.choice_nll_ = nll
        n_choices = len(choices)
        fitted.choice_bic_ = (
            len(free_names) * np.log(n_choices) + 2 * nll if n_choices > 0 else float("nan")
        )
        fitted.choices_scored_ = n_choices
        fitted.choice_result_ = result
        fitted.score_at_ = score_at
        return fitted

    #  Self-generated simulation

    def simulate(self, start_state, n_trials=65, record_history=False):
        """
        Run n_trials of self-generated exploration with rule updates. Each
        trial runs until the agent reaches one of its goal ports, and the next
        trial starts from that port. Returns a list of {"path", "reward",
        "start_port"} dicts.

        When record_history=True, each result dict also carries a "history":
        the per-event list of value snapshots (see snapshot()) captured as
        the update(s) propagate along that trial's path. Use this to build
        step-by-step learning animations (snapshot_values() collapses a
        snapshot to {hex: value}).
        """
        results = []
        current_hex = start_state
        for _ in range(n_trials):
            if current_hex in REWARD_PORTS:
                start_port = current_hex
                goal_hexes = [port for port in REWARD_PORTS if port != current_hex]
            else:
                start_port = REWARD_PORTS[0]
                goal_hexes = list(REWARD_PORTS)

            path, reward, history = self.run_trial(
                current_hex, start_port, goal_hexes, record=record_history
            )
            result = {"path": path, "reward": reward, "start_port": start_port}
            if record_history:
                result["history"] = history
            results.append(result)
            current_hex = path[-1]
        return results

    def run_trial(self, start_hex, start_port, goal_hexes, record=False):
        """
        Roll out one trial under the current policy until the agent reaches a
        goal port, learning from each move as it is made, so every choice is
        made from the values as they stand at that moment.

        Returns (path, reward, history), where history is the per-event value
        snapshot list when record=True, else None.
        """
        context = self.context_for_port(start_port)
        path = [start_hex]
        visited = {start_hex}
        reward = None

        self.start_trial(context)
        history = []
        if record:
            # The maze as it stood before this trial changed anything
            self.record_events(history, path, 0, [None])

        # Keep stepping until the agent arrives at one of the trial's goal ports
        while path[-1] not in goal_hexes:
            prev_hex = path[-2] if len(path) > 1 else None
            cur_hex = path[-1]
            next_hex = self.choose_action(cur_hex, context, visited)
            path.append(next_hex)
            visited.add(next_hex)

            # Reward is delivered (or not) on arrival at a goal port, with that port's
            # reward probability. With no reward_probs given, reward is always 0
            if next_hex in goal_hexes:
                reward = 1.0 if random.random() < self.reward_probs.get(next_hex, 0.0) else 0.0

            events = self.learn_step(prev_hex, cur_hex, next_hex, context, reward)
            if record:
                self.record_events(history, path, len(path) - 1, events)

        events = self.end_trial(path, reward, context)
        if record:
            self.record_events(history, path, len(path) - 1, events)
        return path, reward, history if record else None

    #  Action selection

    def choose_action(self, hex, context, visited):
        """
        Pick the simulated agent's next hex by softmax over the value of each move.

        With no_backtrack, only hexes not yet visited this trial are on offer,
        falling back to every neighbor at a dead end or wherever all of them have
        been visited.

        Parameters:
            hex (int): The hex the agent is deciding from
            context (int or None): The value table to read
            visited (set[int]): The hexes visited so far this trial

        Returns:
            int: The hex the agent moves to
        """
        neighbors = list(self.graph.neighbors(hex))
        if self.no_backtrack:
            unvisited = [neighbor for neighbor in neighbors if neighbor not in visited]
            neighbors = unvisited or neighbors
        probabilities = self.softmax_probabilities(hex, neighbors, context)
        return int(np.random.choice(neighbors, p=probabilities))

    def softmax_probabilities(self, hex, neighbors, context, sweeps=None):
        """
        The probability of choosing each neighbor, from the values being compared.

        The value of each move comes from this learner's retrieval rule, and every
        choice the model makes or scores goes through here: the simulated agent,
        the choice likelihood, and the junction choices shown in animations.

        Temperature sets how strongly the agent prefers the better option: near
        0 it almost always takes the highest-valued neighbor, and at large
        values it chooses closer to random.

        Parameters:
            hex (int): The hex the agent is deciding from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read
            sweeps (list[list[int]] or list[int], optional): The sweeps the rat made
                from this hex, or one run of represented hexes (see RetrievalRule).
                None when the rat represented nothing beyond its own location

        Returns:
            np.ndarray: One probability per neighbor, summing to 1
        """
        values = self.retrieval.action_values(self, hex, neighbors, context, sweeps)
        scaled = values / self.temperature

        # Subtracting the largest value leaves every probability unchanged,
        # and keeps exp() from overflowing when values are divided by a small
        # temperature
        scaled -= scaled.max()

        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum()

    #  Inspection

    def get_state_values(self, start_port):
        """
        Per-hex values {hex: value} for a context, viewed as if walking
        straight out from start_port (see bfs_entry_directions).
        """
        port = resolve_port(start_port)
        context = self.context_for_port(port)
        entry_directions = self.bfs_entry_directions(port) if self.directional_states else None
        return self.forward_hex_values(context, entry_directions)

    def bfs_entry_directions(self, origin, prev_hex=None):
        """
        BFS outward from `origin`, having just arrived from `prev_hex` (or
        None if `origin` has no real predecessor, e.g. a trip's starting
        port). Returns {hex: predecessor}: the direction you'd enter each
        reachable hex from if you kept walking straight out from `origin`
        without doubling back through `prev_hex`. Hexes only reachable by
        turning back through `prev_hex` aren't included.
        """
        entry_from = {origin: prev_hex}
        visited = {origin} | ({prev_hex} if prev_hex is not None else set())
        frontier = [origin]
        while frontier:
            next_frontier = []
            for hex in frontier:
                for neighbor in self.graph.neighbors(hex):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        entry_from[neighbor] = hex
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return entry_from

    def resolve_snapshot_entry_directions(self, path, step_index):
        """
        {hex: prev_hex} entry directions for a snapshot taken after
        step_index steps into path (path[step_index] is the rat's current
        hex), combining three tiers in priority order (lowest first, so
        later entries overwrite earlier ones):
            1. every hex, via a BFS rooted at the trip's start (path[0]);
            2. hexes already entered earlier in this same trial, using the
               real direction they were entered with, taken from path;
            3. hexes ahead of the rat, via a BFS from its current position
               (path[step_index]) that excludes backtracking into where it
               just came from.
        """
        start_directions = self.bfs_entry_directions(path[0])
        # A hex entered more than once this trial keeps its most recent direction
        experienced = {path[i]: (path[i - 1] if i > 0 else None) for i in range(step_index + 1)}
        cur_hex = path[step_index]
        prev_hex = path[step_index - 1] if step_index > 0 else None
        ahead = self.bfs_entry_directions(cur_hex, prev_hex)
        return {**start_directions, **experienced, **ahead}

    def plot_entry_directions(self, path, step_index, ax=None, **plot_kwargs):
        """
        Visualize the entry-direction heuristic (resolve_snapshot_entry_directions)
        used for directional-state display: draws a gray arrow from each hex's
        assigned predecessor to that hex, for the snapshot taken after
        step_index steps into path. The rat is drawn at its current position
        (path[step_index]) and the real path walked so far is overlaid as
        black arrows (hex_path), so any hex already visited this trial should
        show a black arrow exactly on top of its gray one -- only hexes not
        yet visited (guessed via BFS) should show gray-only arrows.

        Only meaningful with directional_states=True; with directional_states=False every
        hex has a single state regardless of entry direction.
        """
        entry_directions = self.resolve_snapshot_entry_directions(path, step_index)
        arrows = {}
        for hex, prev_hex in entry_directions.items():
            if prev_hex is not None:
                arrows.setdefault(prev_hex, []).append(hex)

        return plot_hex_maze(
            self.graph,
            arrows=arrows,
            hex_path=path[: step_index + 1],
            rat=path[step_index],
            rat_from=path[step_index - 1] if step_index > 0 else None,
            ax=ax,
            **plot_kwargs,
        )

    def forward_hex_values(self, context, entry_directions):
        """
        {hex: value} for a context, given a precomputed {hex: prev_hex}
        entry-direction map (see resolve_snapshot_entry_directions). With
        directional_states=False there's only one state per hex, so entry
        directions don't matter and this just reads V[context] directly.
        """
        if not self.directional_states:
            return {hex: self.V[context][hex] for hex in self.graph.nodes()}
        # A hex with no entry direction is one the walk out could not reach, so it
        # shows its prior
        return {
            hex: (
                self.V[context][(entry_directions[hex], hex)]
                if hex in entry_directions
                else self.prior_for_hex(context, hex)
            )
            for hex in self.graph.nodes()
        }

    def snapshot(self, path, step_index):
        """
        One history entry: the rat's current hex (path[step_index]), how far
        along the path that is, and per-context value maps (see
        resolve_snapshot_entry_directions).
        """
        entry_directions = self.resolve_snapshot_entry_directions(path, step_index)
        return {
            "state": path[step_index],
            "step": step_index,
            "entry_directions": entry_directions,
            "values": {context: self.forward_hex_values(context, entry_directions) for context in self.contexts},
        }

    def snapshot_values(self, snapshot, start_port=None):
        """
        Collapse a recorded snapshot to a single {hex: value} map for plotting.

        With start_port given, returns that context's map; otherwise returns the
        max value across all contexts.
        """
        per_context = snapshot["values"]
        if start_port is not None:
            return dict(per_context[self.context_for_port(resolve_port(start_port))])
        return {
            hex: max(table.get(hex, 0.0) for table in per_context.values())
            for hex in self.graph.nodes()
        }

    #  Animation text

    def format_header_text(self, cur_hex, path_so_far=None):
        """Shared header line(s) for animate_learning's text box: the path
        so far (optional) and which hex the rat is in."""
        header = f"rat at hex {cur_hex}"
        if path_so_far:
            header = f"path: {path_so_far}\n{header}"
        return header

    def format_junction_text(self, junction_hex, info):
        """
        Text block describing one binary junction choice (see
        junction_choice_info): the two candidate hexes with their retrieved
        values and choice probabilities, which one was actually taken, and for a
        way the retrieval rule credited a sweep, the swept hex the value came from.
        """
        choosing_between = " and ".join(str(hex) for hex in info["candidates"])
        lines = [f"junction at hex {junction_hex}  (choosing between {choosing_between}):"]
        for hex in info["candidates"]:
            marker = "  ← chosen" if hex == info["choice"] else ""
            # A value read from a sweep belongs to a hex further out, so say which
            credited = info["credited"].get(hex)
            source = f"  from swept hex {credited}" if credited is not None else ""
            lines.append(
                f"  hex {hex}  V={info['values'][hex]:.4f}{source}  p={info['probabilities'][hex]:.3f}{marker}"
            )
        return "\n".join(lines)

    #  Animation

    def animate_learning(
        self,
        trajectories,
        rewards,
        replayed_hexes=None,
        swept_hexes=None,
        mazes=None,
        trials_per_block=None,
        start_port=None,
        panels=False,
        show_updates=True,
        show_choices=True,
        choice_color="yellow",
        show_sweeps=True,
        sweep_color="white",
        credit_color="orange",
        show_trial_info=True,
        show_path=True,
        show_equation=True,
        colormap="viridis",
        vmin=0,
        vmax=1,
        interval=150,
        save_to=None,
        fps=3,
        show_hex_labels=False,
        show_barriers=False,
        ax=None,
        **plot_kwargs,
    ):
        """
        Animate learning, one frame per rule event (see learn_path).

        Runs learn_path over each (path, reward) trial in order -- exactly as
        learn() would -- capturing a value snapshot after every event any
        active rule fires. Each frame colors the maze by hex value (see
        snapshot_values), places the rat at its current hex facing the
        direction it came from, and (on a trial's last frame, when it ends at
        a real reward port) shows a reward droplet or no-reward X there.

        This runs real updates on self as it builds frames, so call reset()
        first for a from-scratch replay.

        Note: this bumps the ``animation.embed_limit`` rcParam (used by
        ``to_jshtml()``) up to at least 512 MB. The matplotlib default is
        20 MB, which most real sessions blow through -- and when it's hit,
        frames are silently dropped (only a warning is logged), so an
        embedded animation can quietly end after just the first few trials.

        Parameters:
            trajectories, rewards (see learn().):
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, passed to learn_path. Used by a learner holding an SWRRule
            swept_hexes (list[list[list[list[int]]]], optional): The sweeps the rat made
                at each point along each trial (see choice_nll), used for the junction
                choice shown on each frame
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            start_port (None, a port (1/2/3 or "A"/"B"/"C"), or "trial"): Which context's
                value table to color by (see snapshot_values). None (default): max value
                across all contexts -- can mask a real decrease in one context if another
                context's value is higher. A port: always that one fixed context's table.
                "trial": whichever context the currently-animated trial actually belongs to.
                Ignored when panels=True.
            panels (bool): If True, draw a subplot per context (self.contexts) side by side,
                each always showing that context's own values, with the rat and reward
                marker only on the panel for the trial's own context. For a learner holding
                one value per direction of travel there is a single context, so the two
                panels are that value table and the entry directions it is being read in
                (see resolve_snapshot_entry_directions). `start_port` and `ax` are ignored
                in this mode.
            show_updates (bool): If True (default), outline whichever hex(es) actually had
                their value change on this event (see UpdateEvent.changed), in the event's
                own color (see UpdateEvent.color).
            show_choices (bool): If True (default), whenever the *previous* step was a real
                3-way junction choice (see junction_choice_info, "back" always excluded
                regardless of no_backtrack), outline the two candidate hexes and print their
                retrieved values and choice probabilities, with the one actually taken
                marked.
            choice_color (str): Outline color for the two junction-candidate hexes. Defaults
                to 'yellow'.
            show_sweeps (bool): If True (default), outline the hexes the rat swept to while
                standing on its current hex, so what it was representing is visible
                alongside the values it was choosing between.
            sweep_color (str): Outline color for swept hexes. Defaults to 'white'.
            credit_color (str): Outline color for the swept hex whose value the retrieval
                rule credited to each way out of a junction (see credited_hexes), named in
                the choice text alongside it. Defaults to 'orange'.
            show_trial_info (bool): If True (default), title each frame with the trial
                number, start port, step within the trial, and (once known) the reward
                outcome.
            show_path (bool): If True (default), add a line above the trial info showing the
                hex-by-hex path walked so far this trial.
            show_equation (bool): If True (default), draw a text box each frame with which
                hex the rat is in and the firing event's own description (see
                UpdateEvent.describe).
            colormap (str): Matplotlib colormap used to color hexes by value.
            vmin, vmax (float or None): Range of hex values the colormap spans.
                Defaults to 0 and 1. None fits that end of the scale to the values
                this animation actually reaches, measured once across every frame so
                the scale holds still while it plays.
            interval (int): Milliseconds between frames while playing in a notebook.
            save_to (str or pathlib.Path): Where to write the animation, so a long one
                can be built once and watched later. The file extension picks the format,
                ".mp4" for a video or ".gif" for an animated image. Any folders in the
                path are created. Left out, the animation is returned without being saved
            fps (int): Frames per second in the saved file.
            show_hex_labels, show_barriers, ax, **plot_kwargs :
                Forwarded to plot_hex_maze.

        Returns:
            matplotlib.animation.FuncAnimation
        """
        fixed_port = None if start_port in (None, "trial") else start_port
        port_labels = {1: "A", 2: "B", 3: "C", None: "shared"}

        # The path line is wrapped by hand to the figure's width, which is known
        # before the figure is built (see the figsize math below), since
        # matplotlib's own wrapping does not handle pre-formatted multi-line
        # monospace text.
        if not panels:
            panel_count = 1
        elif self.directional_states:
            # The one value table, and the directions it is being read in
            panel_count = 2
        else:
            panel_count = len(self.contexts)
        fig_width_inches = 6 * panel_count
        path_wrap_width = max(20, int(fig_width_inches * 9))

        # With no replay given, every trial replays nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        # With no sweeps given, the rat represents nothing beyond its own location
        if swept_hexes is None:
            swept_hexes = [None] * len(trajectories)

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        # A trial with fewer than two hexes has no step to animate, and is left out
        # of the trial count shown in each title
        n_animated_trials = sum(len(path) >= 2 for path in trajectories)

        def values_for(snap):
            if panels:
                return {c: dict(snap["values"][c]) for c in self.contexts}
            elif start_port == "trial":
                return dict(snap["values"][context])
            else:
                return self.snapshot_values(snap, start_port=fixed_port)

        def path_line(up_to_index):
            if not show_path:
                return None
            raw_path = " → ".join(str(h) for h in path[: up_to_index + 1])
            return "\n".join(textwrap.wrap(raw_path, width=path_wrap_width, break_long_words=False))

        def trial_title(step_label, suffix=""):
            if not show_trial_info:
                return None
            return (
                f"Trial {trial_index + 1}/{n_animated_trials}  "
                f"start {port_labels.get(context, context)}  "
                f"step {step_label}/{n_steps}{suffix}"
            )

        frames = []
        trial_index = -1
        for path, reward, trial_replay, trial_sweeps, trial_maze in zip(
            trajectories, rewards, replayed_hexes, swept_hexes, maze_schedule
        ):
            # A barrier change swaps the maze in before this trial is animated,
            # including a change scheduled on a trial too short to animate
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            trial_index += 1

            context = self.resolve_context(path)

            # The junction choice behind each move, worked out just before the move
            # is learned from, so it shows the values the rat chose from
            junction_infos = {}

            def note_junction_choice(step):
                entry = path[step - 1] if step > 0 else None
                sweeps = trial_sweeps[step] if trial_sweeps is not None else None
                junction_infos[step] = self.junction_choice_info(
                    path[step], entry, path[step + 1], context, sweeps
                )

            history = self.learn_path(
                path, reward, context, trial_replay, record=True,
                before_step=note_junction_choice if show_choices else None,
            )
            last_snapshot = len(history) - 1
            show_reward = path[-1] in REWARD_PORTS
            n_steps = len(path) - 1  # hex-to-hex transitions in this trial
            previous_step = None
            for snapshot_index, snap in enumerate(history):
                cur_hex = snap["state"]

                # How far along the path the rat is. Several updates can land on the
                # same step (TD's reward update, a rule's update at the port), and
                # those frames show the rat where it already is
                step_label = snap["step"]
                moved = step_label != previous_step
                previous_step = step_label

                # Decision frame: inserted before a move out of a
                # 3-way junction, so the choice_color highlight
                # appears while the rat is still at the junction hex deciding.
                # Uses the pre-move snapshot's values (history[snapshot_index-1]),
                # matching the state the choice was actually made from.
                if show_choices and moved and step_label >= 1:
                    junction_hex = path[step_label - 1]
                    junction_entry = path[step_label - 2] if step_label >= 2 else None
                    junction_sweeps = (
                        trial_sweeps[step_label - 1] if trial_sweeps is not None else None
                    )
                    junction_info = junction_infos.get(step_label - 1)
                    if junction_info is not None:
                        junction_text = self.format_junction_text(junction_hex, junction_info)
                        decision_text = self.format_header_text(
                            junction_hex, path_so_far=path_line(step_label - 1)
                        ) + "\n" + junction_text
                        # The hexes outlined in credit_color are the ones the values
                        # above were read from, so name what the rule picked them out for
                        if junction_info["credited"]:
                            decision_text += (
                                f"\n  ({credit_color} outline: the "
                                f"{self.retrieval.credit_label} credited to each way)"
                            )
                        frames.append({
                            "graph": self.graph,
                            "hex": junction_hex,
                            "rat_from": junction_entry,
                            "context": context,
                            "values": values_for(history[snapshot_index - 1]),
                            "changed_hexes": {},
                            "junction_candidates": set(junction_info["candidates"]),
                            "swept_hexes": represented_hexes(junction_sweeps),
                            "credited_hexes": set(junction_info["credited"].values()),
                            "entry_directions": history[snapshot_index - 1]["entry_directions"],
                            "path_so_far": path[:step_label],
                            "title": trial_title(step_label - 1, "  (deciding)"),
                            "equation_text": decision_text,
                            "reward": None,
                        })

                # Arrival frame: the rat is at cur_hex, facing away from the hex it
                # came in from
                rat_from = path[step_label - 1] if step_label > 0 else None

                # Each changed state is outlined in its event's color, shaded by how
                # strongly the update included it (see UpdateEvent.changed)
                event = snap["event"]
                changed_hexes = {}
                if show_updates and event is not None:
                    changed_hexes = {
                        state: (event.color, weight)
                        for state, (_, _, weight) in event.changed.items()
                    }

                is_reward_step = snapshot_index == last_snapshot and show_reward
                title = trial_title(step_label, "  →  " + ("rewarded" if reward else "omission") if is_reward_step else "")

                # Which hex the rat is in, then the update's own equation and the
                # states it moved, written by the rule that made it
                equation_text = None
                if show_equation:
                    equation_text = self.format_header_text(cur_hex, path_so_far=path_line(step_label)) + "\n" + (
                        "(no update yet this trial)" if event is None else event.describe()
                    )

                frames.append({
                    "graph": self.graph,
                    "hex": cur_hex,
                    "rat_from": rat_from,
                    "context": context,
                    "values": values_for(snap),
                    "changed_hexes": changed_hexes,
                    "junction_candidates": set(),
                    "swept_hexes": represented_hexes(
                        trial_sweeps[step_label] if trial_sweeps is not None else None
                    ),
                    "credited_hexes": set(),
                    "entry_directions": snap["entry_directions"],
                    "path_so_far": path[:step_label + 1],
                    "title": title,
                    "equation_text": equation_text,
                    "reward": (path[-1], bool(reward)) if is_reward_step else None,
                })

        if not frames:
            raise ValueError("No frames to animate -- check trajectories/rewards.")

        # A vmin or vmax of None is fitted to the values this animation actually
        # reaches. It is measured once over every frame, so the color scale is
        # the same in all of them and a hex only changes color when its value does.
        if vmin is None or vmax is None:
            animated_values = []
            for frame in frames:
                if panels:
                    for context_values in frame["values"].values():
                        animated_values.extend(context_values.values())
                else:
                    animated_values.extend(frame["values"].values())
            if vmin is None:
                vmin = min(animated_values)
            if vmax is None:
                vmax = max(animated_values)

        if plt.rcParams["animation.embed_limit"] < 512:
            plt.rcParams["animation.embed_limit"] = 512

        owns_fig = panels or ax is None
        # Reserve dedicated space below the maze for the equation box when we
        # own the figure, so it doesn't overlap the plotted hexes. If the
        # caller passed their own `ax`, we can't safely resize their
        # figure/layout, so the text falls back to drawing inside the axes.
        extra_height = 2.6 if ((show_equation or show_choices) and owns_fig) else 0
        if panels:
            fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 6 + extra_height))
            axes = list(np.atleast_1d(axes))
        else:
            fig = plt.figure(figsize=(6, 6 + extra_height)) if owns_fig else ax.figure
            axes = [ax if ax is not None else fig.add_subplot(111)]
        if extra_height:
            fig.subplots_adjust(bottom=0.30)

        # Fix the axes to a single extent up front: plot_hex_maze widens
        # xlim/ylim to make room whenever `reward` is set, so if left alone
        # the maze visibly shrinks and grows between reward and non-reward
        # frames. Probing with a reward marker captures the wider extent,
        # which every panel/frame is then pinned to (same physical maze).
        plot_hex_maze(
            self.graph, reward=(REWARD_PORTS[0], True),
            show_hex_labels=show_hex_labels, show_barriers=show_barriers,
            ax=axes[0], **plot_kwargs,
        )
        fixed_xlim, fixed_ylim = axes[0].get_xlim(), axes[0].get_ylim()

        equation_bbox = dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.4")

        # fig.text() adds another text box on every call, and nothing clears it
        # between frames, so the box below the maze is made once here and its
        # text replaced each frame. With a caller-supplied `ax` there is no room
        # reserved below the maze (single-axis mode only -- panels always owns
        # its figure), so set_equation_text() draws inside the axes, which
        # ax.clear() wipes at the start of every frame.
        equation_artist = None
        if show_equation and owns_fig:
            equation_artist = fig.text(
                0.5, 0.02, "", ha="center", va="bottom", fontsize=8, family="monospace",
                bbox=equation_bbox,
            )

        def build_outlines(frame):
            """(outline_hexes, outline_colors) lists. Each changed hex gets
            its own single-hex group, shaded along its event's color ramp by its
            weight (TD's eligibility, model-based's T) normalized against the
            max weight among hexes of that same color in this frame, so the
            eligibility/memory-trace decay is visible directly as an outline
            fading from bright to dim. The junction-candidate group 
            (`choice_color`) stays a single flat color, since it's always exactly 
            two hexes with no decay to show."""
            by_color = {}
            for state, (color, weight) in frame["changed_hexes"].items():
                by_color.setdefault(color, []).append((self.hex_of_state(state), weight))

            groups, colors = [], []
            for color_name, entries in by_color.items():
                cmap = _COLOR_TO_CMAP[color_name]
                max_weight = max(weight for _, weight in entries) or 1.0
                for hex, weight in entries:
                    shade = 0.35 + 0.65 * min(weight / max_weight, 1.0)
                    groups.append({hex})
                    colors.append(cmap(shade))

            # The hexes the rat was representing, drawn flat since a sweep
            # either reached a hex or it did not
            if show_sweeps and frame["swept_hexes"]:
                groups.append(frame["swept_hexes"])
                colors.append(sweep_color)

            if frame["junction_candidates"]:
                groups.append(frame["junction_candidates"])
                colors.append(choice_color)

            # The one hex per way whose value the retrieval rule credited, drawn
            # after the others so it stays visible where the groups overlap
            if show_sweeps and frame["credited_hexes"]:
                groups.append(frame["credited_hexes"])
                colors.append(credit_color)
            return (groups, colors) if groups else (None, None)

        def draw_panel(panel_ax, frame, values, is_active):
            """Draw one axes' worth of the maze for a frame. is_active gates
            the rat/reward marker/outline -- in panels mode only the trial's
            own context should show them; in single-axis mode it's always True."""
            panel_ax.clear()
            outline, outline_colors = build_outlines(frame) if is_active else (None, None)
            # Each frame is drawn on the maze in force at that moment of the session
            plot_hex_maze(
                frame["graph"],
                color_by=values,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
                rat=frame["hex"] if is_active else None,
                rat_from=frame["rat_from"] if is_active else None,
                reward=frame["reward"] if is_active else None,
                outline_hexes=outline,
                outline_colors=outline_colors,
                show_hex_labels=show_hex_labels,
                show_barriers=show_barriers,
                ax=panel_ax,
                **plot_kwargs,
            )
            panel_ax.set_xlim(fixed_xlim)
            panel_ax.set_ylim(fixed_ylim)

        def draw_entry_directions(panel_ax, frame):
            """Which direction each hex's value is being read in: a gray arrow from the
            hex it would be entered from (see resolve_snapshot_entry_directions), with
            the path the rat has run this trial drawn over it in black, so a hex it has
            already been through shows its real direction and the rest are the walk
            outward from where it stands."""
            panel_ax.clear()
            arrows = {}
            for hex, entered_from in frame["entry_directions"].items():
                if entered_from is not None:
                    arrows.setdefault(entered_from, []).append(hex)
            plot_hex_maze(
                frame["graph"],
                arrows=arrows,
                hex_path=frame["path_so_far"],
                rat=frame["hex"],
                rat_from=frame["rat_from"],
                show_hex_labels=show_hex_labels,
                show_barriers=show_barriers,
                ax=panel_ax,
                **plot_kwargs,
            )
            panel_ax.set_xlim(fixed_xlim)
            panel_ax.set_ylim(fixed_ylim)
            panel_ax.set_title("Entry directions")

        def set_equation_text(fallback_ax, text):
            if equation_artist is not None:
                equation_artist.set_text(text or "")
            elif text:
                # Caller-supplied ax: no reserved margin available, so draw
                # inside the axes bounds (may overlap the maze).
                fallback_ax.text(0.02, 0.02, text, transform=fallback_ax.transAxes,
                                  ha="left", va="bottom", fontsize=8, family="monospace",
                                  bbox=equation_bbox)

        def draw(frame_index):
            frame = frames[frame_index]
            if panels and self.directional_states:
                draw_panel(axes[0], frame, frame["values"][self.contexts[0]], True)
                axes[0].set_title("Hex values", fontweight="bold")
                draw_entry_directions(axes[1], frame)
                if frame["title"]:
                    fig.suptitle(frame["title"], wrap=True, fontsize=10)
                set_equation_text(None, frame["equation_text"])
            elif panels:
                for context, panel_ax in zip(self.contexts, axes):
                    is_active = context == frame["context"]
                    draw_panel(panel_ax, frame, frame["values"][context], is_active)
                    label = f"Start port {port_labels.get(context, context)}"
                    panel_ax.set_title(label + ("  (active)" if is_active else ""),
                                        fontweight="bold" if is_active else "normal")
                if frame["title"]:
                    fig.suptitle(frame["title"], wrap=True, fontsize=10)
                set_equation_text(None, frame["equation_text"])
            else:
                draw_panel(axes[0], frame, frame["values"], True)
                if frame["title"]:
                    axes[0].set_title(frame["title"], wrap=True, fontsize=10)
                set_equation_text(axes[0], frame["equation_text"])

        anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=interval)

        # Written before the figure is closed, since saving renders every frame
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, fps=fps)

        if owns_fig:
            plt.close(fig)
        return anim
