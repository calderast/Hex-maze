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
This state space is especially useful for "simulate" mode so the rat doesn't 
turn around and run back towards the port it just left).

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


# Map each outline color name to a colormap used to shade a changed hex's outline by
# its update weight (see UpdateEvent.changed and build_outlines in animate_learning).
# Each goes from dim (lowest weight) to bright (highest), so we can see a trace fading
# back along the path, and the hexes that changed most stand out even where the maze
# is dark (low value). Each UpdateEvent uses one of these color names.
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
    Get the set of all hexes the rat represented (in either sweep format).

    Parameters:
        sweeps (list[list[int]] or list[int]): The sweeps the rat made from one
            hex (each a list of hexes), or a single list of represented hexes

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
    module docstring). The learner holds the value table V, steps through trials
    hex by hex, and calls each rule at the start of a trial, on every step, and at
    the port. The rules decide how the values change.

    Each rule's hyperparameters (e.g. TDLambdaRule's alpha/lam,
    ModelBasedRule's gamma_mb/a_T/a_mb) are passed to the constructor and
    stored as attributes of the learner, so all of a model's parameters live
    in one place, and each rule reads the ones it needs from the learner.
    `gamma` is a named argument here because it also sets how quickly prior
    values decay with distance from a port, regardless of which rules are active.

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
                within a trial when it can (see choose_action/simulate). This only
                applies to simulated behavior, so scoring and the update rules still
                see every neighbor. Defaults to True, which keeps simulated trials
                closer in length to the rats' runs (a simulated trial only ends when
                the agent reaches a port, and an agent allowed to turn around wanders
                back and forth for a lot longer before getting there).
            retrieval (RetrievalRule): How the agent values each possible move.
                Defaults to LocalRetrieval, which values a move by the hex it
                would step into.
            reward_probs (list of float): [p1, p2, p3] reward probability at ports 1/A,
                2/B, 3/C. Read when simulating only, to sample whether a reward is delivered.
            **rule_kwargs: Rule-specific hyperparameters (e.g. alpha=, lam= for
                TDLambdaRule; gamma_mb=, a_T=, a_mb= for ModelBasedRule), set as plain
                attributes on this instance.
        """
        # A list of mazes is a session with barrier changes. The agent starts on
        # the first maze (its graph and priors are built from this one), and learn
        # swaps in the later ones. A list of ints is a single maze (barrier hexes).
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
        Get the distance (in hexes) between every pair of open hexes in this maze.

        This is computed the first time it's needed and cached until the maze
        changes. SWRRule uses it to find which directions of travel through a
        replayed hex lead toward the port.

        Returns:
            pd.DataFrame: Distance matrix indexed by hex
        """
        if self._hex_distances is None:
            self._hex_distances = get_pairwise_distance_matrix(self.graph)
        return self._hex_distances

    def critical_choice_points(self, start_port=None):
        """
        Get the critical choice point(s) for this maze (cached per maze).

        Starting from a port, the critical choice point is where the shortest paths
        to the other two ports split, so it's the hex where (theoretically) the rat 
        commits to one port over the other.

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
        Get every state this learner has a value for.

        For a non-directional learner (one value per hex), this is each open hex.
        For a directional learner (states are directed edges), this is each way
        of arriving at a hex, plus the starting state for each hex, (None, hex).

        Returns:
            list: Every state in this learner's representation
        """
        if not self.directional_states:
            return list(self.graph.nodes())

        # The starting state of a trip (before the rat has moved)
        states = [(None, hex) for hex in self.graph.nodes()]

        # Every way of arriving at each hex from one of its neighbors
        states += [
            (prev, cur) for cur in self.graph.nodes() for prev in self.graph.neighbors(cur)
        ]
        return states

    def reset(self):
        """Clear all value tables and start every state at its prior value."""
        self.V = {context: {} for context in self.contexts}
        self.build_prior_table(self.priors)

        # No trials yet, so nothing has been replayed
        self.replayed_hexes_this_trial = []

        # Initialize every state to its prior (0 if no priors were given)
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
        # Every hex in the empty maze gets a prior, so if a barrier change opens
        # up a new hex later, that hex has a prior too
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
        # Use distances in the empty maze (no barriers), so these priors reflect
        # what the rat knows about the task in general (e.g. reward ports are at
        # the 3 corners) before it learns which paths are open in this maze
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

            # Every hex in the empty maze gets a prior, so if a barrier change
            # opens up a new hex later, that hex has a prior too
            for hex in empty_maze.nodes():
                # The prior for the port hex is the expected p(reward) at that port
                if hex in goals:
                    table[context][hex] = port_value[hex]
                else:
                    # Every other hex gets the value of the best port it can reach,
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

        # Clear the cached distances and choice points (they were for the old maze)
        self._hex_distances = None
        self._choice_points = {}

        # Get the states that exist in the new maze. Drop values for hexes that
        # are now barriers (so if that hex opens up again later, the agent has to
        # learn it from scratch), and start newly opened hexes at their prior
        # (same as they would be at the start of a session).
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

        # Let each rule drop whatever it learned about those hexes too
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
        Get which maze each trial is run on, as a list with one entry per trial.

        Only the trial where a barrier change happens has a maze. Every other
        trial is None (meaning keep using the current maze).

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

    #  One trial, step by step
    #
    # Each trial is learned in order: start_trial before the first move,
    # learn_step after each hex-to-hex move, and end_trial once the rat reaches the
    # port. Anything that makes or scores a choice reads values between these calls,
    # so every choice uses the values as they were at that moment in the trial.

    def start_trial(self, context, replayed_hexes=None):
        """
        Set up every rule for a new trial.

        Parameters:
            context (int or None): The value table this trial updates (the port
                the trial started from, or None for a single shared table)
            replayed_hexes (list[int]): The hexes replayed during this trial's
                sharp-wave ripples (used by SWRRule)
        """
        # Store what was replayed this trial (SWRRule reads this when the rat
        # reaches the port)
        self.replayed_hexes_this_trial = replayed_hexes or []

        # Let each rule clear whatever it accumulates within a trial
        for rule in self.rules:
            rule.on_trial_start(self, context)

    def is_possible_step(self, prev_hex, cur_hex, next_hex):
        """
        Check if a hex-to-hex move from the data is possible in this maze.

        A move through a barrier, or between two hexes that aren't neighbors,
        is a position tracking error. We warn about it and skip it for both
        learning and scoring.

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

        # A move between two hexes that aren't neighbors can't have happened,
        # so it is a position tracking error
        if not self.graph.has_edge(cur_hex, next_hex):
            warnings.warn(
                f"at hex {cur_hex}, next_hex={next_hex} is not a "
                f"graph neighbor ({sorted(self.graph.neighbors(cur_hex))})! "
                f"This is likely a position tracking error. Skipping this step."
            )
            return False

        # A directional state also includes the hex the rat came from, so if
        # the move into cur_hex was a tracking error, there's no valid state to update
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
        # Get the states being left and entered (hexes for a non-directional
        # learner, directed edges for a directional one)
        state = self.state_key(prev_hex, cur_hex)
        next_state = self.state_key(cur_hex, next_hex)

        # Reward is only given on the move into the reward port
        is_terminal = reward is not None

        # Every rule sees every transition, so multiple rules can learn
        # from the same step
        events = []
        for rule in self.rules:
            events += rule.on_step(
                self, context, state, next_state, reward if is_terminal else 0.0, is_terminal
            )
        return events

    def end_trial(self, path, reward, context):
        """
        Let every rule learn from the whole trial, once the rat reaches the port.

        Parameters:
            path (list[int]): The hexes the rat ran this trial, ending at the port
            reward (float): The reward delivered at that port
            context (int or None): The value table to update

        Returns:
            list[UpdateEvent]: The updates made at the port, across every rule
        """
        # Rules that learn from the whole trial (e.g. SWR replay or the
        # model-based update) update once the rat has reached the port
        events = []
        for rule in self.rules:
            events += rule.on_trial_end(self, path, reward, context)
        return events

    def record_events(self, history, path, step_index, events):
        """
        Add one snapshot per update to `history`, with its UpdateEvent stored
        under "event". This way, if there are multiple updates on the same step
        (e.g. two rules, or TD(lambda)'s step update followed by its reward update),
        each one gets its own animation frame and description. All snapshots from
        the same step show the maze after all of that step's updates.

        Parameters:
            history (list): The snapshots so far, added to in place
            path (list[int]): The trial's path, at least up to the rat's hex
            step_index (int): Where along the path the rat is (see snapshot)
            events (list[UpdateEvent]): The updates to record
        """
        # The snapshot before a trial's first move has no event (None)
        for event in events:
            snap = self.snapshot(path, step_index)
            snap["event"] = event
            history.append(snap)

    def learn_path(self, path, reward, context, replayed_hexes=None, record=False,
                   before_step=None):
        """
        Step through a single recorded path within one context, updating values
        after each move (in the order the rat ran them). Reward is delivered at
        the terminal state (path[-1]).

        Parameters:
            path (list[int]): The hexes the rat ran this trial, in order, ending
                at the reward port
            reward (float): The reward delivered at that port
            context (int or None): The value table to update (the port the trial
                started from, or None for a single shared table)
            replayed_hexes (list[int]): The hexes replayed during this trial's
                sharp-wave ripples (used by SWRRule)
            record (bool): If True, save a snapshot of the maze's values after every
                update, for animating learning (see record_events). The first
                snapshot is the maze before the trial, with "event" None
            before_step (callable, optional): Called as before_step(step) right
                before learning from the move from path[step] to path[step + 1]
                (for every valid move). This is where we score a choice, using
                the values from when the rat made it

        Returns:
            list or None: One snapshot per update when record=True
        """
        self.start_trial(context, replayed_hexes)

        history = []
        if record:
            # Snapshot of the maze before this trial changed anything
            self.record_events(history, path, 0, [None])

        # Step through the trial one hex-to-hex transition at a time, in the order
        # the rat ran them, so values change exactly as they would have in the session
        for step in range(len(path) - 1):
            # The previous hex (needed for directional states).
            # There is no previous hex on the first step.
            prev_hex = path[step - 1] if step > 0 else None
            cur_hex, next_hex = path[step], path[step + 1]

            # Warn about and skip tracking errors
            if not self.is_possible_step(prev_hex, cur_hex, next_hex):
                continue

            # Values haven't been updated for this move yet, so they are the
            # values the rat had when it made this move
            if before_step is not None:
                before_step(step)

            # Reward is delivered on the move into the reward port
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
        Learn from a list of recorded trajectories (e.g. the rat's hex paths).

        Parameters:
            trajectories (list of list of int): Hex path for each trial [s0, s1, ...,
                s_terminal]. The first and last hex of each trial should be a reward port
                (except for the very first trial, which is allowed to not start at a port)
            rewards (list of float): Reward for each trajectory.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, one list per trial, in the same order as `trajectories`. Used
                by SWRRule
            mazes (list, optional): The maze for each block of trials, for a session
                with barrier changes. Learned values carry over across a change: hexes
                that become barriers are dropped, and newly opened hexes start at
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

        # If no replay is given, no hexes are replayed on any trial
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        history = []
        for path, reward, trial_replay, trial_maze in zip(
            trajectories, rewards, replayed_hexes, maze_schedule
        ):
            # If there's a barrier change, swap in the new maze before learning this trial
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
        Get the value of the rat's current hex, and of the hex it is representing,
        at every sample of a timeseries.

        This steps through the session exactly like `learn` does, and gets values for
        every sample of a finer-grained timeseries: `sampled_hexes[trial][i]` is the
        hex the rat was in at sample i of that trial, and `represented_hexes[trial][i]`
        is the hex it was representing at that sample. Both are read from the same
        value table, so they can be compared directly.

        `values` picks which value table each sample is read from:

            "trial"  the values at the start of the trial, so a hex's value doesn't
                     change within a trial (the only thing that changes the
                     timeseries is which hex the rat is at or representing).
            "step"   the values while the rat was in that hex, including any updates
                     from earlier in this trial. Note a hex's value then jumps when
                     the rat leaves it (bc of that step's own update), which doesn't
                     tell us anything about where the rat went.

        For a directional learner, each hex is valued in the direction the rat would
        enter it (see resolve_snapshot_entry_directions), same as the values
        animate_learning shows. For "trial" this means walking out from where the
        trial started, and for "step" from where the rat is right now.

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
            pd.DataFrame: One row per sample, with columns for the trial, the sample's
                index within that trial, the step along the path, the hex the rat was
                in and its value, and the hex it was representing and its value.
                Barrier hexes have no value, so their value is nan
        """
        # If no replay is given, no hexes are replayed on any trial. If no
        # represented hexes are given, the rat represents nothing on any sample
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]
        if represented_hexes is None:
            represented_hexes = [None for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        rows = []
        for trial, (path, reward, samples, represented, trial_replay, trial_maze) in enumerate(zip(
            trajectories, rewards, sampled_hexes, represented_hexes, replayed_hexes, maze_schedule
        )):
            # If there's a barrier change, swap in the new maze before this trial
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue

            context = self.resolve_context(path)

            # Get how far along the path the rat was at each sample. The trajectory
            # is just the samples with consecutive repeats removed, so the rat takes
            # a step whenever a sample moves on to the next hex in the path. A sample
            # that doesn't match the current hex or the next one is a tracking error,
            # so it stays on the current step.
            steps, step = [], 0
            for hex in samples:
                if step + 1 < len(path) and hex == path[step + 1]:
                    step += 1
                steps.append(step)

            if represented is None:
                represented = [None] * len(samples)

            if values == "trial":
                # Read every sample in this trial from the values before the trial
                table = self.snapshot(path, 0)["values"][context]
                self.learn_path(path, reward, context, trial_replay)
                table_for_step = [table] * len(path)
            elif values == "step":
                history = self.learn_path(path, reward, context, trial_replay, record=True)

                # Get the values while the rat was on each hex of the path (the last
                # snapshot from when it arrived there). If no rule changed anything
                # on a step, use the values from the step before
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

        # Make the hex columns nullable ints (samples with nothing decoded have
        # represented_hex None, which would otherwise make the column floats)
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

        # The values the agent compared to get the probabilities (if the rat swept
        # down one of these ways, this is the value from the sweep)
        values = dict(zip(
            candidates, self.retrieval.action_values(self, cur_hex, candidates, context, sweeps)
        ))

        # The swept hex each value came from (only for moves where the retrieval
        # rule used a sweep, see RetrievalRule.credited_hexes)
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

        Steps through each trajectory move by move. Each move is scored by the
        softmax probability of the hex the rat actually stepped to (using the
        current values), and then learned from, so every later choice (including
        later in the same trial) is scored using the updated values.

        Parameters:
            trajectories (list of list of int): Hex path for each trial [s0, s1, ...,
                s_terminal]. The first and last hex of each trial should be a reward port
                (except for the very first trial, which is allowed to not start at a port)
            rewards (list of float): Reward for each trajectory.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, one list per trial, in the same order as `trajectories`. Used
                by SWRRule
            swept_hexes (list[list[list[list[int]]]], optional): The sweeps the rat made
                at each point along each trial: one entry per hex of that trial's path,
                with the sweeps made while the rat was at that hex, each a list of hexes
                in the order they were represented (or a single list of represented
                hexes, see RetrievalRule). Used by retrieval rules that use sweeps
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            record (bool, optional): If True, also return a list of per-choice records, one
                per scored hex-to-hex move, in trajectory order: {"trial": index into
                trajectories/rewards, "entry": prev_hex (None at a trial's first step),
                "hex": cur_hex, "choice": next_hex, "probability": p_choice,
                "probabilities": {neighbor: prob, ...}}.
            score_at ({"steps", "junctions", "choice_points"}): Which of the rat's moves
                are scored (from every move, to only the moves that decide the trial)
                - "steps" (default): every step, against ALL of cur_hex's graph
                  neighbors, backward included
                - "junctions": only real binary choices, meaning steps where the rat is at
                  a 3-way junction (cur_hex has exactly 3 graph neighbors) and leaves
                  through one of the two forward neighbors (left/right). Junctions where
                  the rat backtracked, and all non-junction steps, are skipped
                - "choice_points": only the binary choices at the critical choice point for
                  the port the rat started from (see critical_choice_points), which is the
                  hex where the paths to the other two ports split. Junctions where the
                  paths reconnect are skipped

        Returns:
            float, or (float, list of dict) if record=True: Total negative log-likelihood of
                the scored hex choices, and (if record=True) the per-choice records.
        """
        # If no replay is given, no hexes are replayed on any trial
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        # If no sweeps are given, the rat doesn't represent anything beyond its current location
        if swept_hexes is None:
            swept_hexes = [None] * len(trajectories)

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        # -log p(choice) for every scored choice, summed at the end
        neg_log_likelihoods = []
        choices = [] if record else None
        for trial_index, (path, reward, trial_replay, trial_sweeps, trial_maze) in enumerate(
            zip(trajectories, rewards, replayed_hexes, swept_hexes, maze_schedule)
        ):
            # If there's a barrier change, swap in the new maze before scoring this trial
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            context = self.resolve_context(path)

            # For score_at="choice_points", get the critical choice point(s) for the port
            # this trial started from. If the trial doesn't start at a port (only possible
            # for the session's first trial), use the choice points from every port
            choice_points = self.critical_choice_points(
                path[0] if path[0] in REWARD_PORTS else None
            ) if score_at == "choice_points" else set()

            def score_choice(step):
                """Score the move from path[step] to path[step + 1], using the values
                from when the rat made it. Tracking errors are already skipped before
                this is called (see learn_path)."""
                cur_hex, next_hex = path[step], path[step + 1]
                entry_hex = path[step - 1] if step > 0 else None

                if score_at in ("junctions", "choice_points"):
                    if score_at == "choice_points" and cur_hex not in choice_points:
                        return  # a junction that is not a critical choice point (outside this analysis)

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

                # What the rat was representing while at this hex
                step_sweeps = trial_sweeps[step] if trial_sweeps is not None else None
                probabilities = self.softmax_probabilities(
                    cur_hex, candidates, context, step_sweeps
                )
                probability_by_neighbor = dict(zip(candidates, probabilities.tolist()))
                p_choice = probability_by_neighbor[next_hex]

                # Floor the probability at 1e-10 so a probability of 0 doesn't
                # make the NLL infinite
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

            # Score each choice right before that move updates the values, so it's
            # scored using what the rat had learned up to that point
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
        fit, since gamma also sets how prior values decay with distance and
        temperature sets how values are turned into choices (regardless of which
        update rules are active). Parameters that don't affect the likelihood
        (e.g. gamma_mb for a TD-only agent) are left out, so they don't slow down
        the optimizer or count against the model in BIC.

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
                trial, passed to choice_nll. Used by SWRRule
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
                - choices_scored_ : how many of the rat's choices were scored (divide
                  choice_nll_ by this to get the NLL per choice)
                - choice_result_ : raw scipy OptimizeResult, or None if every parameter
                  was fixed
                - score_at_ : the score_at value used to fit
        """
        # Learner classes like HexMazeTDLearner and KrauszDualLearner set their own
        # rules, so build one and get the rules from it. BaseHexLearner needs the
        # rules passed in.
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

        # Run choice_nll once more with record=True to get both the NLL and the number
        # of choices actually scored, which we need for BIC (scoring only junctions or
        # choice points scores way fewer choices than there are hex-to-hex steps).
        # Then throw that learner away and return a fresh, untrained learner with the
        # same parameters.
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
        Simulate one trial using the current policy until the agent reaches a
        goal port. The agent learns from each move as it makes it, so every
        choice uses the most up-to-date values.

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
            # Snapshot of the maze before this trial changed anything
            self.record_events(history, path, 0, [None])

        # Keep stepping until the agent reaches one of the trial's goal ports
        while path[-1] not in goal_hexes:
            prev_hex = path[-2] if len(path) > 1 else None
            cur_hex = path[-1]
            next_hex = self.choose_action(cur_hex, context, visited)
            path.append(next_hex)
            visited.add(next_hex)

            # When the agent reaches a goal port, sample whether it gets rewarded
            # using that port's reward probability (always 0 if no reward_probs given)
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

        If no_backtrack is True, the agent can only choose hexes it hasn't visited
        yet this trial (unless it's at a dead end or has visited every neighbor,
        in which case it can choose any neighbor).

        Parameters:
            hex (int): The hex the agent is choosing from
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
        Get the softmax probability of choosing each neighbor.

        The value of each move comes from this learner's retrieval rule. Every
        choice the model makes or scores goes through here: the simulated agent,
        the choice likelihood, and the junction choices shown in animations.

        Temperature sets how strongly the agent prefers the better option: near
        0 it almost always picks the highest-valued neighbor, and for large
        values its choices are closer to random.

        Parameters:
            hex (int): The hex the agent is choosing from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read from
            sweeps (list[list[int]] or list[int], optional): The sweeps the rat made
                from this hex, or a single list of represented hexes (see RetrievalRule).
                None if the rat didn't represent anything beyond its current location

        Returns:
            np.ndarray: One probability per neighbor, summing to 1
        """
        values = self.retrieval.action_values(self, hex, neighbors, context, sweeps)
        scaled = values / self.temperature

        # Subtract the max value to keep exp() from overflowing when temperature
        # is small (this doesn't change the probabilities)
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
        # A hex with no entry direction couldn't be reached by the BFS, so just
        # show its prior
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
        values and choice probabilities, which one the rat actually chose, and (if
        the retrieval rule used a sweep) the swept hex each value came from.
        """
        choosing_between = " and ".join(str(hex) for hex in info["candidates"])
        lines = [f"junction at hex {junction_hex}  (choosing between {choosing_between}):"]
        for hex in info["candidates"]:
            marker = "  ← chosen" if hex == info["choice"] else ""
            # If the value came from a sweep, show which swept hex it came from
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

        Runs learn_path over each (path, reward) trial in order (exactly like
        learn() does), and saves a value snapshot after every update from any
        active rule. Each frame colors the maze by hex value (see
        snapshot_values), shows the rat at its current hex facing away from the
        hex it came from, and (on a trial's last frame, if it ends at a real
        reward port) shows a reward droplet or no-reward X there.

        Note this actually updates the learner's values as it builds frames, so
        call reset() first if you want to animate learning from scratch.

        Also note: this bumps the ``animation.embed_limit`` rcParam (used by
        ``to_jshtml()``) up to at least 512 MB. The matplotlib default is
        20 MB, which most real sessions go way over. When it's hit, frames
        are silently dropped (only a warning is logged), so an embedded
        animation can end after just the first few trials!

        Parameters:
            trajectories, rewards (see learn().):
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, passed to learn_path. Used by SWRRule
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
                marker only on the panel for the trial's own context. A directional learner
                only has a single context, so instead the two panels show the values and
                the entry directions used to read them (see
                resolve_snapshot_entry_directions). `start_port` and `ax` are ignored
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
                at its current hex, so we can see what it was representing alongside the
                values it was choosing between.
            sweep_color (str): Outline color for swept hexes. Defaults to 'white'.
            credit_color (str): Outline color for the swept hex that the retrieval rule
                used for each move out of a junction (see credited_hexes). This hex is
                also listed in the choice text. Defaults to 'orange'.
            show_trial_info (bool): If True (default), title each frame with the trial
                number, start port, step within the trial, and (once known) the reward
                outcome.
            show_path (bool): If True (default), add a line above the trial info showing the
                hex-by-hex path walked so far this trial.
            show_equation (bool): If True (default), draw a text box each frame with which
                hex the rat is in and the firing event's own description (see
                UpdateEvent.describe).
            colormap (str): Matplotlib colormap used to color hexes by value.
            vmin, vmax (float or None): Range of hex values for the colormap.
                Defaults to 0 and 1. If None, that end of the scale is set to the min/max
                value across all frames of this animation (so the scale stays the same
                for the whole animation).
            interval (int): Milliseconds between frames while playing in a notebook.
            save_to (str or pathlib.Path): Where to save the animation (useful for long
                animations so we can build it once and watch it later). The file
                extension sets the format: ".mp4" for a video or ".gif" for an animated
                image. Any folders in the path are created. If None (default), the
                animation is returned without being saved
            fps (int): Frames per second in the saved file.
            show_hex_labels, show_barriers, ax, **plot_kwargs :
                Forwarded to plot_hex_maze.

        Returns:
            matplotlib.animation.FuncAnimation
        """
        fixed_port = None if start_port in (None, "trial") else start_port
        port_labels = {1: "A", 2: "B", 3: "C", None: "shared"}

        # Wrap the path line by hand to the figure's width (we know this before
        # the figure is built, see the figsize math below), since matplotlib's
        # own wrapping doesn't handle pre-formatted multi-line monospace text.
        if not panels:
            panel_count = 1
        elif self.directional_states:
            # One panel for the values, one for the entry directions
            panel_count = 2
        else:
            panel_count = len(self.contexts)
        fig_width_inches = 6 * panel_count
        path_wrap_width = max(20, int(fig_width_inches * 9))

        # If no replay is given, no hexes are replayed on any trial
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        # If no sweeps are given, the rat doesn't represent anything beyond its current location
        if swept_hexes is None:
            swept_hexes = [None] * len(trajectories)

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        # Trials with fewer than two hexes have no steps to animate, so leave them
        # out of the trial count shown in each title
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
            # If there's a barrier change, swap in the new maze before animating this
            # trial (even if this trial is too short to animate)
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            trial_index += 1

            context = self.resolve_context(path)

            # Junction choice info for each move, computed right before learning from
            # that move, so it shows the values the rat was choosing from
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

                # How far along the path the rat is. Multiple updates can happen on the
                # same step (e.g. TD's reward update, or a rule's update at the port),
                # and the rat stays in the same place for those frames
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
                        # Explain what the credit_color outlines mean (the hexes the
                        # values above came from)
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

                # Outline each changed state in its event's color, shaded by its
                # update weight (see UpdateEvent.changed)
                event = snap["event"]
                changed_hexes = {}
                if show_updates and event is not None:
                    changed_hexes = {
                        state: (event.color, weight)
                        for state, (_, _, weight) in event.changed.items()
                    }

                is_reward_step = snapshot_index == last_snapshot and show_reward
                title = trial_title(step_label, "  →  " + ("rewarded" if reward else "omission") if is_reward_step else "")

                # Text box: which hex the rat is in, then the update's equation and
                # the states it changed (from the rule's describe())
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

        # If vmin or vmax is None, set it from the values across every frame, so
        # the color scale is the same for all frames (a hex only changes color
        # when its value changes)
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

        # fig.text() adds a new text box every time it's called, and nothing clears
        # it between frames, so make the text box below the maze once here and just
        # replace its text each frame. If the caller passed their own `ax`, there's
        # no room below the maze (this only happens in single-axis mode, since panels
        # always makes its own figure), so set_equation_text() draws inside the axes
        # instead (ax.clear() wipes it at the start of every frame).
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

            # The hexes the rat was representing (one flat color, since a hex
            # was either swept to or it wasn't)
            if show_sweeps and frame["swept_hexes"]:
                groups.append(frame["swept_hexes"])
                colors.append(sweep_color)

            if frame["junction_candidates"]:
                groups.append(frame["junction_candidates"])
                colors.append(choice_color)

            # The hex the retrieval rule used for each move. Draw these last so they
            # stay visible where groups overlap
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
            # Draw each frame on the maze at that point in the session (in case of barrier changes)
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
            """Draw which direction each hex's value is read in: a gray arrow from the
            hex it would be entered from (see resolve_snapshot_entry_directions), with
            the rat's path so far this trial drawn on top in black. Hexes the rat has
            already visited show the real direction it entered from, and the rest are
            guessed by walking outward from where the rat is now."""
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

        # Save before closing the figure, since saving renders every frame
        if save_to is not None:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, fps=fps)

        if owns_fig:
            plt.close(fig)
        return anim
