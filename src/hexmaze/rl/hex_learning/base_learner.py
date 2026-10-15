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
on each hex transition plus a model-based SWR update at reward).
"""

import copy
import random
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize

from ...utils import create_empty_hex_maze, maze_to_graph
from ...core import get_safe_hex_distance
from ...utils import REWARD_PORTS, resolve_port
from ...plotting import plot_hex_maze

# Named outline color -> single-hue colormap, used to shade a changed hex's
# outline by its update's weight (see UpdateEvent.changed and build_outlines
# inside animate_learning). Falls back to grayscale for an unrecognized name.
_COLOR_TO_CMAP = {
    "red": "Reds", "green": "Greens", "blue": "Blues",
    "yellow": "YlOrBr", "orange": "Oranges", "purple": "Purples",
}


@dataclass
class UpdateEvent:
    """
    One value update, recorded so it can be displayed (used for animation).

    Attributes:
        changed: {state: (old_value, new_value, weight)} for the states whose
            value moved by a visible amount. `weight` is how strongly the rule
            included that state in this update (which is the eligibility trace
            for TD learning and T(port, state) for a model-based update). 
            animate_learning scales it to shade each hex's outline, so
            a trace decaying back along the path is visible on the maze.
        describe: A function of no arguments returning this update's text
            block: its equation and the states it moved.
        color: Outline color family for the changed hexes, which is how an
            animation tells one rule's updates from another's.
    """
    changed: dict
    describe: Callable[[], str]
    color: str


class UpdateRule:
    """
    Base interface for a pluggable value-update mechanism (letting us simulate/test
    different theories of how hex values should be updates based on the rat's experience). 
    
    See update_rules.py for the currently implemented rules:
    - TD(lambda): Classic temporal-difference learning with an eligibility trace.
        Hex values are updated on each hex transition.
    - Model-based memory trace (from Krausz et al.): on reaching a reward port, 
        updates the values of every state on paths that have led into that port, 
        weighted by a persistent, recency-averaged map T learned online.
    - SWR update: on reaching a reward port, updates the values of hexes represented
        in sharp-wave-ripples

    A learner holds a list of these and calls every one of them at each
    moment below, so combining value update mechanisms (e.g. TD learning plus replay)
    is a matter of listing two rules. Every hook here does nothing, so a rule
    implements only the moments it cares about.
    """

    def on_trial_start(self, learner, path, context):
        """
        Called before a trial's steps are walked (e.g. to reset a per-trial 
        eligibility/memory trace). 
        No return value.
        """

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        """
        Called once per hex-to-hex transition, in the order the rat ran them.
        `is_terminal` is True for the transition into the reward port, where
        `reward` is what was delivered (1 or 0 for reward/omission). 
        Returns a list of UpdateEvent.
        """
        return []

    def on_trial_end(self, learner, path, reward, context):
        """
        Called after the rat reaches the port, for a rule that learns from the
        trial as a whole (e.g. SWR replay or model-based update). 
        Returns a list of UpdateEvent.
        """
        return []

    def on_maze_changed(self, learner):
        """
        Called when learner.set_graph() swaps in a new maze, which is how a
        barrier change is applied mid-session. A rule keeping its own
        hex-keyed quantities should drop the entries for hexes that are now
        barriers, so a hex that later reopens starts fresh. The value table V
        has already been cleaned up when this runs.
        """

    def on_reset(self, learner):
        """
        Called when the learner's value tables are cleared, including once
        when the learner is built. A rule that accumulates something over a
        session (like a memory trace), clears it here, so a reset learner 
        starts the session over from the beginning.
        """


class BaseHexLearner:
    """
    Hex-maze value learner driven by a list of UpdateRule objects (see
    module docstring). Everything here is update-rule-agnostic: it reads
    values via `state_value`/`V` and drives rules through their hooks, but
    never assumes a particular update mechanism.

    Rule-specific hyperparameters (e.g. TDLambdaRule's alpha/lam,
    ModelBasedRule's gamma_mb/a_T/a_mb) are passed as constructor kwargs and
    set as plain attributes on the learner (not hidden inside rule objects),
    so any rule can read whatever it needs off the `learner` argument each
    hook already receives. `gamma` is the one exception kept as an explicit
    parameter here, since it's also used for prior-value discounting
    independent of which rules are active.
    """

    _FIT_PARAM_DEFAULTS = {
        # TDLambdaRule
        "alpha": (0.3, (1e-3, 1.0)), "gamma": (0.9, (0.0, 0.999)),
        "lam": (0.3, (0.0, 1.0)), "temperature": (0.5, (0.01, 10.0)),
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
        no_backtrack=False,
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
                [TDLambdaRule()] or [TDLambdaRule(), ModelBasedRule()]. Order matters only
                in that each rule's events become separate, sequential animation frames when
                two rules fire on the same step.
            gamma (float): Discount factor used for prior-value computation, and reused by
                any rule (like TDLambdaRule) that wants a shared discount rate.
            temperature (float): Softmax temperature for action selection (policy-level, not
                tied to any one update rule).
            directional_states (bool): If True, states are directed edges (prev_hex, cur_hex); else
                hexes.
            goal_conditioned_states (bool): If True, keep one value function per start port (3
                value tables); else a single shared value function.
            priors (None, "uniform", ("flat", value), or list of 3 floats): Hex value
                initialization strategy (see distance_priors/build_prior_table).
            no_backtrack (bool): If True, the simulated agent avoids revisiting states
                within a trial where it can (see choose_action/simulate). Applies to
                simulated behavior alone, so scoring and the update rules see every
                available neighbor.
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

        for key, value in rule_kwargs.items():
            setattr(self, key, value)

        # Contexts: one value table per start port, or a single shared one.
        self.contexts = list(REWARD_PORTS) if goal_conditioned_states else [None]

        self.prior_table = None  # {context: {hex: value}} or None
        self.V = {}
        self.reset()

    #  Setup / priors

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
        elif isinstance(priors, tuple) and len(priors) == 2 and priors[0] == "flat":
            value = priors[1]
            self.prior_table = {
                context: {hex: value for hex in self.graph.nodes()} for context in self.contexts
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

            for hex in self.graph.nodes():
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
        ``maze_to_graph``. States are lazily re-created against the new graph;
        stale states are dropped from V, and each rule gets a chance to prune
        its own persistent state too (see UpdateRule.on_maze_changed).
        
        TODO: How do we want to deal with the values of newly open states
        (on barrier change)? Currently they are set to whatever the prior would
        have been. 
        """
        self.graph = maze_to_graph(new_maze)

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
        """Human-readable label for a state key, e.g. 'V(48→43)' or 'V(43)'."""
        if self.directional_states:
            prev, cur = state
            return f"{prev if prev is not None else 'start'}→{cur}"
        return str(state)

    def state_value(self, context, state):
        """
        Read a state's value.

        Parameters:
            context (int or None): Which value table to read, meaning the port
                the trip started from
            state (int or tuple): The hex, or the directed edge, to read

        Returns:
            float: That state's current value
        """
        return self.V[context][state]

    #  Rule-driven learning

    def maze_schedule(self, trajectories, mazes=None, trials_per_block=None):
        """
        Which maze each trial is run on, as a list with one entry per trial.

        Only the trial where a barrier change happens carries a maze; every other
        trial is None, meaning the maze the rat is already running.

        Parameters:
            trajectories (list of list of int): The session's trials
            mazes (list): The maze for each block of trials, in order
            trials_per_block (list of int): How many trials are run on each maze

        Returns:
            list: One entry per trial, a maze on the first trial of each block and
                None elsewhere
        """
        schedule = [None] * len(trajectories)
        if mazes is None:
            return schedule

        trial = 0
        for maze, block_length in zip(mazes, trials_per_block):
            schedule[trial] = maze
            trial += block_length
        return schedule

    def learn_path(self, path, reward, context, replayed_hexes=None, record=False):
        """
        Walk a single path within one context, driving every active rule's
        hooks in order. Reward is delivered at the terminal state (path[-1]). 

        Returns a list of per-event snapshots when record=True, else None. 

        Each snapshot (other than the initial one) carries an
        "event" (an UpdateEvent) describing exactly what that snapshot
        reflects: one snapshot per event, so two rules firing on the same
        step become two separate, sequential snapshots (matching how
        TD(lambda)'s own bootstrap-then-reward split already works when
        only one rule is active).

        Parameters:
            path (list[int]): The hexes the rat ran this trial, in order, ending
                at the reward port
            reward (float): The reward delivered at that port
            context (int or None): The value table to update, meaning the port
                the trial started from, or None for a single shared table
            replayed_hexes (list[int]): The hexes replayed during this trial's
                sharp-wave ripples, for a learner holding an SWRRule
            record (bool): Collect a snapshot of the maze's values after every
                update, for animating how learning unfolded

        Returns:
            list or None: One snapshot per update when record=True
        """
        # What was replayed this trial, which SWRRule reads when the rat
        # reaches the port
        self.replayed_hexes_this_trial = replayed_hexes or []

        # Let each rule clear whatever it accumulates within a trial
        for rule in self.rules:
            rule.on_trial_start(self, path, context)

        history = []
        if record:
            # The maze as it stood before this trial changed anything
            init_snap = self.snapshot(path, 0)
            init_snap["event"] = None
            history.append(init_snap)

        # Walk the trial one hex-to-hex transition at a time, in the order the
        # rat ran them, so values evolve exactly as they would have in the session
        for step in range(len(path) - 1):
            # The hex before the current one, which is what makes a state
            # directional. There is no previous hex on the first step.
            prev_hex = path[step - 1] if step > 0 else None
            cur_hex, next_hex = path[step], path[step + 1]

            # The states being left and entered, which are hexes for a learner
            # holding one value per hex, and directed edges for a directional one
            state = self.state_key(prev_hex, cur_hex)
            next_state = self.state_key(cur_hex, next_hex)

            # True for the step that arrives at the reward port
            is_terminal = step == len(path) - 2

            # A hex that is a barrier in this maze cannot have been run through,
            # so it is a position tracking error. Warn and skip.
            if cur_hex not in self.graph or next_hex not in self.graph:
                missing = [h for h in (cur_hex, next_hex) if h not in self.graph]
                warnings.warn(
                    f"learn_path: hex {missing} is a barrier in this maze! "
                    f"This is likely a position tracking error. Skipping this step."
                )
                continue

            # A move between two hexes that do not border each other cannot
            # have happened, so it is a position tracking error. Warn and skip.
            if not self.graph.has_edge(cur_hex, next_hex):
                warnings.warn(
                    f"learn_path: at hex {cur_hex}, next_hex={next_hex} is not a "
                    f"graph neighbor ({sorted(self.graph.neighbors(cur_hex))})! "
                    f"This is likely a position tracking error. Skipping this step."
                )
                continue

            # A directional learner's state also records which hex the rat came
            # from, so a tracking error on the way in leaves no state to update.
            if self.directional_states and prev_hex is not None and not self.graph.has_edge(prev_hex, cur_hex):
                warnings.warn(
                    f"learn_path: arrived at hex {cur_hex} from hex {prev_hex}, which is not "
                    f"a graph neighbor. This is likely a position tracking error. "
                    f"Skipping this step."
                )
                continue

            # Every rule sees every transition, so several mechanisms can learn
            # from the same step
            for rule in self.rules:
                events = rule.on_step(self, context, state, next_state, reward, is_terminal)
                if record:
                    for event in events:
                        snap = self.snapshot(path, step + 1)
                        snap["event"] = event
                        history.append(snap)

        # Rules that learn from the trial as a whole, such as a replay or
        # model-based update, update once the rat has reached the port
        for rule in self.rules:
            events = rule.on_trial_end(self, path, reward, context)
            if record:
                for event in events:
                    snap = self.snapshot(path, len(path) - 1)
                    snap["event"] = event
                    history.append(snap)

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

    def process_trajectory(self, path, reward, replayed_hexes=None):
        """Run one trial's rule updates along a single path."""
        self.learn_path(path, reward, self.resolve_context(path), replayed_hexes)

    def process_trajectory_with_history(self, path, reward, replayed_hexes=None):
        """Same as process_trajectory, but returns the per-event snapshot
        history (see learn_path)."""
        return self.learn_path(
            path, reward, self.resolve_context(path), replayed_hexes, record=True
        )

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
        return [n for n in all_neighbors if n != entry_hex]

    def junction_choice_info(self, cur_hex, entry_hex, next_hex, context):
        """
        Binary junction-choice info for one step, matching choice_nll's
        junctions_only=True scoring: 
        
        Returns None if cur_hex isn't a real 3-way junction, or next_hex is 
        entry_hex, meaning the rat backtracked. (This is valid, just not a binary 
        left/right choice, so we follow Tim's paper in not scoring it).
        Also returns None if next_hex isn't one of the junction candidates at all. 
        (Could happen due to a position tracking error. Callers that need to warn 
        about that, like choice_nll, check it themselves). 
        Otherwise returns:
        {"candidates": [hexA, hexB], 
        "values": {hex: V(hex)}, 
        "probabilities": {hex: p}, 
        "choice": next_hex, 
        "probability": p_choice}.
        """
        candidates = self.junction_candidates(cur_hex, entry_hex)
        if candidates is None or next_hex == entry_hex or next_hex not in candidates:
            return None
        probabilities = self.softmax_probabilities(cur_hex, candidates, context)
        probability_by_neighbor = dict(zip(candidates, probabilities.tolist()))
        values = {c: self.state_value(context, self.state_key(cur_hex, c)) for c in candidates}
        return {
            "candidates": candidates,
            "values": values,
            "probabilities": probability_by_neighbor,
            "choice": next_hex,
            "probability": probability_by_neighbor[next_hex],
        }

    def choice_nll(self, trajectories, rewards, replayed_hexes=None, mazes=None,
                   trials_per_block=None, record=False, junctions_only=False):
        """
        Negative log-likelihood of the rat's hex-to-hex choices under this
        model's current parameters.

        Replays each trajectory, scoring the softmax probability of the hex
        the rat actually stepped to (before that step's update), then runs
        the ordinary rule updates so values evolve as the trajectory proceeds.

        Parameters:
            trajectories (list of list of int): Hex path for each trial [s0, s1, ...,
                s_terminal]. The first and last hex of each trial should be a reward port
                (except for the very first trial, which is allowed to not start at a port)
            rewards (list of float): Reward for each trajectory.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, one list per trial, in the same order as `trajectories`. Used
                by a learner holding an SWRRule
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            record (bool, optional): If True, also return a list of per-choice records, one
                per scored hex-to-hex move, in trajectory order: {"trial": index into
                trajectories/rewards, "entry": prev_hex (None at a trial's first step),
                "hex": cur_hex, "choice": next_hex, "probability": p_choice,
                "probabilities": {neighbor: prob, ...}}.
            junctions_only (bool, optional): If True, restrict scoring to genuine binary
                choices: steps where the rat is at a real 3-way intersection (cur_hex has
                exactly 3 graph neighbors) and exits through one of the two non-backward
                neighbors (left/right). A junction where the rat instead backtracked, and
                every non-junction step, is skipped silently. If False (default), every step
                is scored against ALL of cur_hex's graph neighbors, backward included.

        Returns:
            float, or (float, list of dict) if record=True: Total negative log-likelihood of
                the scored hex choices, and (if record=True) the per-choice records.
        """
        # With no replay given, every trial replays nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        total = 0.0
        choices = [] if record else None
        for trial_index, (path, reward, trial_replay, trial_maze) in enumerate(
            zip(trajectories, rewards, replayed_hexes, maze_schedule)
        ):
            # A barrier change swaps the maze in before this trial is scored
            if trial_maze is not None:
                self.set_graph(trial_maze)

            if len(path) < 2:
                continue
            context = self.resolve_context(path)

            for step in range(len(path) - 1):
                cur_hex, next_hex = path[step], path[step + 1]
                entry_hex = path[step - 1] if step > 0 else None

                # A hex that is a barrier in this maze cannot have been run through,
                # so it is a tracking error, or a trial scored against the wrong maze.
                if cur_hex not in self.graph or next_hex not in self.graph:
                    missing = [hex for hex in (cur_hex, next_hex) if hex not in self.graph]
                    warnings.warn(
                        f"choice_nll: hex {missing} is a barrier in this maze -- likely a "
                        f"tracking error. Skipping this step."
                    )
                    continue

                if junctions_only:
                    candidates = self.junction_candidates(cur_hex, entry_hex)
                    if candidates is None:
                        continue  # not a junction (outside this analysis)
                    if next_hex == entry_hex:
                        continue  # backtracked at the junction (valid, just not a binary L/R choice)
                    if next_hex not in candidates:
                        warnings.warn(
                            f"choice_nll: at junction hex {cur_hex} (entry={entry_hex}), "
                            f"next_hex={next_hex} is not a graph-neighbor -- likely a "
                            f"tracking error. Skipping this step."
                        )
                        continue
                else:
                    candidates = list(self.graph.neighbors(cur_hex))
                    if next_hex not in candidates:
                        warnings.warn(
                            f"choice_nll: at hex {cur_hex}, next_hex={next_hex} is not a "
                            f"graph-neighbor ({candidates}) -- likely a tracking error. "
                            f"Skipping this step."
                        )
                        continue

                probabilities = self.softmax_probabilities(cur_hex, candidates, context)
                probability_by_neighbor = dict(zip(candidates, probabilities.tolist()))
                p_choice = probability_by_neighbor[next_hex]
                total -= np.log(max(p_choice, 1e-10))
                if record:
                    choices.append({
                        "trial": trial_index,
                        "entry": entry_hex,
                        "hex": cur_hex,
                        "choice": next_hex,
                        "probability": p_choice,
                        "probabilities": probability_by_neighbor,
                    })

            self.learn_path(path, reward, context, trial_replay)
        return (total, choices) if record else total

    @classmethod
    def fit_choices(cls, maze, trajectories, rewards, reward_probs=None,
                     alpha=None, gamma=None, lam=None, temperature=None,
                     gamma_mb=None, a_T=None, a_mb=None, a_swr=None, replayed_hexes=None,
                     mazes=None, trials_per_block=None,
                     rules=None, junctions_only=False, **kwargs):
        """
        Fit hyperparameters to maximize the likelihood of the rat's
        hex-to-hex choices (not just reward outcomes).

        Which parameters are actually fit depends on which rule types are in
        `rules`: alpha/lam only matter (and are only fit) if a TDLambdaRule
        is present; gamma_mb/a_T/a_mb only matter (and are only fit) if a
        ModelBasedRule is present. gamma and temperature are always
        eligible, since gamma also affects prior-value discounting and
        temperature always affects action-selection regardless of which
        value-update rules are active. This avoids wasting the optimizer's
        effort on a parameter that has literally zero effect on the
        likelihood (e.g. fitting gamma_mb for a TD-only agent).

        Parameters:
            maze (see __init__.):
            trajectories (list of list of int):
            rewards (list of float):
            reward_probs (see __init__.):
            alpha, gamma, lam, temperature (float or None): TDLambdaRule's parameters
                (gamma/temperature are general-purpose, see above). Fix at the given value
                instead of fitting it; None (default) fits it, if a relevant rule is
                present.
            gamma_mb, a_T, a_mb (float or None): ModelBasedRule's parameters. Same
                fix-or-fit convention.
            a_swr (float or None): SWRRule's learning rate for replayed hexes. Same
                fix-or-fit convention.
            replayed_hexes (list[list[int]], optional): The hexes replayed on each
                trial, passed to choice_nll. Used by a learner holding an SWRRule
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
            junctions_only (bool, optional): Passed to choice_nll.
            **kwargs: Extra constructor flags held fixed during fitting (e.g. directional_states,
                goal_conditioned_states, priors, no_backtrack).

        Returns:
            BaseHexLearner subclass instance: Fresh instance built with the best-fit (and
                any fixed) parameters (and the fixed **kwargs), carrying:
                - choice_nll_ : choice NLL at optimum
                - choice_bic_ : BIC, counting only the *fitted* params and the choices
                  actually scored
                - choice_result_ : raw scipy OptimizeResult, or None if every parameter
                  was fixed
                - junctions_only_ : the junctions_only value used to fit
        """
        from .update_rules import TDLambdaRule, ModelBasedRule, SWRRule  # local import: avoids a cycle

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
                    trajectories, rewards, replayed_hexes, mazes, trials_per_block,
                    junctions_only=junctions_only,
                ),
                x0=x0, bounds=bounds, method='L-BFGS-B',
            )
            final_values = result.x
        else:
            # Nothing to fit (every parameter was pinned)
            result = None
            final_values = []

        # One evaluation with record=True gives both the NLL and the actual
        # number of scored choices (needed for BIC -- under junctions_only
        # that's far fewer than the number of hex-to-hex steps). This model
        # is discarded; the returned `fitted` is a fresh, untrained instance
        # at the same parameters, consistent regardless of whether anything
        # was optimized.
        nll, choices = _build(final_values).choice_nll(
            trajectories, rewards, replayed_hexes, mazes, trials_per_block,
            record=True, junctions_only=junctions_only,
        )
        fitted = _build(final_values)
        fitted.choice_nll_ = nll
        n_choices = len(choices)
        fitted.choice_bic_ = (
            len(free_names) * np.log(n_choices) + 2 * nll if n_choices > 0 else float("nan")
        )
        fitted.choice_result_ = result
        fitted.junctions_only_ = junctions_only
        return fitted

    #  Self-generated simulation

    def simulate(self, start_state, n_trials=65, max_steps=200, record_history=False):
        """
        Run n_trials of self-generated exploration with rule updates. Each
        trial starts from the previous trial's terminal state. Returns a
        list of {"path", "reward", "start_port"} dicts.

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
                current_hex, start_port, goal_hexes, max_steps, record=record_history
            )
            result = {"path": path, "reward": reward, "start_port": start_port}
            if record_history:
                result["history"] = history
            results.append(result)
            current_hex = path[-1]
        return results

    def run_trial(self, start_hex, start_port, goal_hexes, max_steps, record=False):
        """
        Roll out one trial under the current policy, then apply the rule updates.

        Returns (path, reward, history), where history is the per-event value
        snapshot list when record=True, else None.
        """
        context = self.context_for_port(start_port)
        current_hex = start_hex
        path = [current_hex]
        visited = {current_hex}
        reward = 0.0

        for _ in range(max_steps):
            next_hex = self.choose_action(current_hex, context, visited)
            if next_hex is None:
                break
            path.append(next_hex)
            visited.add(next_hex)
            if next_hex in goal_hexes:
                reward = self.sample_reward(next_hex)
                break
            current_hex = next_hex

        history = self.learn_path(path, reward, context, record=record)
        return path, reward, history

    #  Action selection

    def get_neighbors(self, hex, visited=None):
        """Available neighbors of `hex`, respecting no_backtrack."""
        neighbors = list(self.graph.neighbors(hex))
        if self.no_backtrack and visited is not None:
            unvisited = [neighbor for neighbor in neighbors if neighbor not in visited]
            if unvisited:
                return unvisited
        return neighbors

    def choose_action(self, hex, context, visited=None):
        """Pick the next hex via softmax over the value of entering each neighbor."""
        neighbors = self.get_neighbors(hex, visited)
        if not neighbors:
            return None
        probabilities = self.softmax_probabilities(hex, neighbors, context)
        return int(np.random.choice(neighbors, p=probabilities))

    def action_values(self, hex, neighbors, context):
        """
        The value compared for each candidate move: the value of the state the
        agent would reach by moving to that neighbor.

        This is the single place where "what the agent consults when deciding"
        is defined, and everything that turns values into choices goes through
        it -- the policy, the simulated agent, and the choice likelihood -- so
        they cannot disagree.

        Parameters:
            hex (int): The hex the agent is deciding from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read, meaning the port the
                trip started from

        Returns:
            np.ndarray: One value per neighbor, in the order given
        """
        return np.array(
            [self.state_value(context, self.state_key(hex, neighbor)) for neighbor in neighbors]
        )

    def softmax_probabilities(self, hex, neighbors, context):
        """
        The probability of choosing each neighbor, from the values being compared.

        Temperature sets how strongly the agent prefers the better option: near
        0 it almost always takes the highest-valued neighbor, and at large
        values it chooses closer to random.

        Parameters:
            hex (int): The hex the agent is deciding from
            neighbors (list[int]): The hexes it could move to
            context (int or None): The value table to read

        Returns:
            np.ndarray: One probability per neighbor, summing to 1
        """
        values = self.action_values(hex, neighbors, context)
        scaled = values / self.temperature

        # Subtracting the largest value leaves every probability unchanged,
        # and keeps exp() from overflowing when values are divided by a small
        # temperature
        scaled -= scaled.max()

        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum()

    def sample_reward(self, hex):
        """
        Sample a binary reward at a reward port (used when simulating).

        Parameters:
            hex (int): The hex the agent arrived at (1, 2, or 3)

        Returns:
            float: 1.0 if reward was delivered, otherwise 0.0
        """
        if hex in self.reward_probs and random.random() < self.reward_probs[hex]:
            return 1.0
        # If no reward_probs argument was set, reward is always 0 :(
        return 0.0

    #  Inspection

    def action_probabilities(self, hex, start_port):
        """
        Softmax choice probabilities at a hex under a given context.

        Returns {neighbor_hex: probability}.
        """
        context = self.context_for_port(resolve_port(start_port))
        neighbors = list(self.graph.neighbors(hex))
        if not neighbors:
            return {}
        probabilities = self.softmax_probabilities(hex, neighbors, context)
        return dict(zip(neighbors, probabilities.tolist()))

    def get_state_values(self, start_port):
        """
        Per-hex values {hex: value} for a context, viewed as if walking
        straight out from start_port (see bfs_entry_directions).
        """
        port = resolve_port(start_port)
        context = self.context_for_port(port) if self.goal_conditioned_states else None
        entry_directions = self.bfs_entry_directions(port) if self.directional_states else None
        return self.forward_hex_values(context, entry_directions)

    def get_max_state_values(self):
        """{hex: max value across the 3 ports' outbound views}."""
        per_port = [self.get_state_values(port) for port in REWARD_PORTS]
        return {hex: max(table[hex] for table in per_port) for hex in self.graph.nodes()}

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
        return {
            hex: (
                self.state_value(context, (entry_directions[hex], hex))
                if hex in entry_directions
                else self.prior_for_hex(context, hex)
            )
            for hex in self.graph.nodes()
        }

    def snapshot(self, path, step_index):
        """
        One history entry: the rat's current hex (path[step_index]) plus
        per-context value maps (see resolve_snapshot_entry_directions).
        """
        entry_directions = self.resolve_snapshot_entry_directions(path, step_index)
        return {
            "state": path[step_index],
            "values": {context: self.forward_hex_values(context, entry_directions) for context in self.contexts},
        }

    def snapshot_values(self, snapshot, start_port=None):
        """
        Collapse a recorded snapshot to a single {hex: value} map for plotting.

        With start_port given, returns that context's map; otherwise returns the
        max value across all contexts (like get_max_state_values).
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

    def format_update_text(self, cur_hex, event, path_so_far=None):
        """
        Multi-line diagnostic text for one animate_learning frame: the hex
        path walked so far this trial, which hex the rat is in, and (if an
        event just happened) that event's own description -- see
        UpdateEvent.describe. Generic across every UpdateRule: this never
        branches on which rule produced the event.
        """
        header = self.format_header_text(cur_hex, path_so_far)
        if event is None:
            return header + "\n(no update yet this trial)"
        return header + "\n" + event.describe()

    def format_junction_text(self, junction_hex, info):
        """
        Text block describing one binary junction choice (see
        junction_choice_info): the two candidate hexes with their retrieved
        values and choice probabilities, and which one was actually taken.
        Returns None if `info` is None (not a scored junction choice there).
        """
        if info is None:
            return None
        lines = [f"junction at hex {junction_hex}  (choosing between {info['candidates'][0]} and {info['candidates'][1]}):"]
        for hex in info["candidates"]:
            marker = "  ← chosen" if hex == info["choice"] else ""
            lines.append(
                f"  hex {hex}  V={info['values'][hex]:.4f}  p={info['probabilities'][hex]:.3f}{marker}"
            )
        return "\n".join(lines)

    #  Animation

    def animate_learning(
        self,
        trajectories,
        rewards,
        replayed_hexes=None,
        mazes=None,
        trials_per_block=None,
        start_port=None,
        panels=False,
        show_updates=True,
        show_choices=True,
        choice_color="yellow",
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
            mazes (list, optional): The maze for each block of trials, for a session with
                barrier changes (see learn)
            trials_per_block (list of int, optional): How many trials are run on each maze
            start_port (None, a port (1/2/3 or "A"/"B"/"C"), or "trial"): Which context's
                value table to color by (see snapshot_values). None (default): max value
                across all contexts -- can mask a real decrease in one context if another
                context's value is higher. A port: always that one fixed context's table.
                "trial": whichever context the currently-animated trial actually belongs to.
                Ignored when panels=True.
            panels (bool): If True, draw one subplot per context (self.contexts) side by
                side, each always showing that context's own values. The rat and reward
                marker are only drawn on the panel for the trial's own context. `start_port`
                and `ax` are ignored in this mode.
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
            show_trial_info (bool): If True (default), title each frame with the trial
                number, start port, step within the trial, and (once known) the reward
                outcome.
            show_path (bool): If True (default), add a line above the trial info showing the
                hex-by-hex path walked so far this trial.
            show_equation (bool): If True (default), draw a text box each frame with which
                hex the rat is in and the firing event's own description (see
                format_update_text).
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

        # The equation box's figure width is known before the figure itself
        # is built (see the figsize math below), so precompute a wrap width
        # for the path line now rather than fighting matplotlib's wrap=True
        # against pre-formatted multi-line monospace text.
        fig_width_inches = 6 * len(self.contexts) if panels else 6
        path_wrap_width = max(20, int(fig_width_inches * 9))

        # With no replay given, every trial replays nothing
        if replayed_hexes is None:
            replayed_hexes = [[] for _ in trajectories]

        maze_schedule = self.maze_schedule(trajectories, mazes, trials_per_block)

        valid_trials = [
            (path, reward, trial_replay, trial_maze)
            for path, reward, trial_replay, trial_maze in zip(
                trajectories, rewards, replayed_hexes, maze_schedule
            )
            if len(path) >= 2
        ]

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
                f"Trial {trial_index + 1}/{len(valid_trials)}  "
                f"start {port_labels.get(context, context)}  "
                f"step {step_label}/{n_steps}{suffix}"
            )

        frames = []
        for trial_index, (path, reward, trial_replay, trial_maze) in enumerate(valid_trials):
            # A barrier change swaps the maze in before this trial is animated
            if trial_maze is not None:
                self.set_graph(trial_maze)

            context = self.resolve_context(path)
            history = self.learn_path(path, reward, context, trial_replay, record=True)
            last_step = len(history) - 1
            show_reward = path[-1] in REWARD_PORTS
            n_steps = len(path) - 1  # real hex-to-hex transitions (extra
            # end-of-step/end-of-trial events land on the same step index,
            # so this is capped rather than counted)
            last_hex, last_rat_from = None, None
            for step_index, snap in enumerate(history):
                cur_hex = snap["state"]
                moved = not (step_index > 0 and cur_hex == last_hex)
                step_label = min(step_index, n_steps)

                # Decision frame: inserted before a move out of a
                # 3-way junction, so the choice_color highlight
                # appears while the rat is still at the junction hex deciding.
                # Uses the pre-move snapshot's values (history[step_index-1]),
                # matching the state the choice was actually made from.
                if show_choices and moved and step_label >= 1:
                    junction_hex = path[step_label - 1]
                    junction_entry = path[step_label - 2] if step_label >= 2 else None
                    junction_info = self.junction_choice_info(junction_hex, junction_entry, cur_hex, context)
                    if junction_info is not None:
                        junction_text = self.format_junction_text(junction_hex, junction_info)
                        decision_text = self.format_header_text(
                            junction_hex, path_so_far=path_line(step_label - 1)
                        ) + "\n" + junction_text
                        frames.append({
                            "hex": junction_hex,
                            "rat_from": junction_entry,
                            "context": context,
                            "values": values_for(history[step_index - 1]),
                            "changed_hexes": {},
                            "junction_candidates": set(junction_info["candidates"]),
                            "title": trial_title(step_label - 1, "  (deciding)"),
                            "equation_text": decision_text,
                            "reward": None,
                        })

                # Arrival frame: the rat has now moved to cur_hex.
                if step_index == 0:
                    rat_from = None
                elif cur_hex == last_hex:
                    # Same physical hex as the previous snapshot (an
                    # end-of-step/end-of-trial event's extra frame).
                    # The rat hasn't moved, so keep its previous facing.
                    rat_from = last_rat_from
                else:
                    rat_from = last_hex
                last_hex, last_rat_from = cur_hex, rat_from

                event = snap.get("event")
                changed_hexes = {}
                if show_updates and event is not None:
                    changed_hexes = {
                        state: (event.color, entry[2] if len(entry) > 2 else 1.0)
                        for state, entry in event.changed.items()
                    }

                is_reward_step = step_index == last_step and show_reward
                title = trial_title(step_label, "  →  " + ("rewarded" if reward else "omission") if is_reward_step else "")
                equation_text = (
                    self.format_update_text(cur_hex, event, path_so_far=path_line(step_label))
                    if show_equation else None
                )

                frames.append({
                    "hex": cur_hex,
                    "rat_from": rat_from,
                    "context": context,
                    "values": values_for(snap),
                    "changed_hexes": changed_hexes,
                    "junction_candidates": set(),
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
            fig, axes = plt.subplots(1, len(self.contexts), figsize=(6 * len(self.contexts), 6 + extra_height))
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

        # fig.text() adds a new artist every call rather than replacing the
        # previous one, unlike ax.text() after ax.clear() or fig.suptitle()
        # -- so create it once (whenever we own the figure and reserved
        # space for it below the maze) and update its contents in place each
        # frame. With a caller-supplied `ax` we can't reserve that space
        # (single-axis mode only -- panels always owns its figure), so
        # set_equation_text() falls back to ax.text() inside the axes.
        equation_artist = None
        if show_equation and owns_fig:
            equation_artist = fig.text(
                0.5, 0.02, "", ha="center", va="bottom", fontsize=8, family="monospace",
                bbox=equation_bbox,
            )

        def build_outlines(frame):
            """(outline_hexes, outline_colors) lists. Each changed hex gets
            its own single-hex group, shaded along its event's color family
            (e.g. Reds/Greens) by its weight (TD's eligibility, model-based's
            T) normalized against the max weight among hexes of that same
            color in this frame, so the eligibility/memory-trace decay is
            visible directly as an outline gradient. The junction-candidate group 
            (`choice_color`) stays a single flat color, since it's always exactly 
            two hexes with no decay to show."""
            by_color = {}
            for state, (color, weight) in frame["changed_hexes"].items():
                by_color.setdefault(color, []).append((self.hex_of_state(state), weight))

            groups, colors = [], []
            for color_name, entries in by_color.items():
                cmap = plt.get_cmap(_COLOR_TO_CMAP.get(color_name, "Greys"))
                max_weight = max(weight for _, weight in entries) or 1.0
                for hex, weight in entries:
                    shade = 0.35 + 0.65 * min(weight / max_weight, 1.0)
                    groups.append({hex})
                    colors.append(cmap(shade))

            if frame["junction_candidates"]:
                groups.append(frame["junction_candidates"])
                colors.append(choice_color)
            return (groups, colors) if groups else (None, None)

        def draw_panel(panel_ax, frame, values, is_active):
            """Draw one axes' worth of the maze for a frame. is_active gates
            the rat/reward marker/outline -- in panels mode only the trial's
            own context should show them; in single-axis mode it's always True."""
            panel_ax.clear()
            outline, outline_colors = build_outlines(frame) if is_active else (None, None)
            plot_hex_maze(
                self.graph,
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
            if panels:
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


class HexMazeAgent(BaseHexLearner):
    """
    Concrete hex-maze learner with a configurable list of value-update
    rules (see module docstring and update_rules.py). Construct directly
    when you want more than plain TD(lambda), e.g.:

        HexMazeAgent(
            maze,
            rules=[TDLambdaRule(), ModelBasedRule()],
            alpha=0.3, gamma=0.95, lam=0.3,      # TDLambdaRule reads these
            gamma_mb=0.8, a_T=0.1, a_mb=0.05,    # ModelBasedRule reads these
            goal_conditioned_states=True,
        )
    """
