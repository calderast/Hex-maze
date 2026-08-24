"""
td_learner.py

TD(lambda) hex value agent for the hex maze

Value is learned over maze locations via TD learning with eligibility traces.
The single ``lam`` (lambda) controls:

    lam = 0.0  -> pure TD(0): one-step bootstrapping, value propagates
                 backward one hex per repeated traversal (e.g. Krausz 2023)
    lam = 1.0  -> Monte-Carlo: full discounted return assigned along the
                 whole path within a single trial.
    0 < lam<1  -> eligibility-trace blend of all intermediate horizons.

We can choose to represent hex states in a variety of ways:

    directional : bool
        False -> value over hexes (49 states), V[hex].
        True  -> value over directional hex-states, i.e. directed edges
                 (prev_hex, cur_hex) ~ 126 states. (e.g. Krausz 2023)

    goal_conditioned : bool
        False -> a single shared value function (e.g. Krausz 2023). Good
                 for *fitting* observed trajectories, but when used to *generate*
                 behavior the agent will turn around and run back up the value
                 gradient to the port it just left.
        True  -> one value function per start port (3 value tables). This reflects
                 that the start port cannot give reward on the current trial.
                 Use this for simulate().

Reward port hexes are always terminal: reward is delivered on the transition into the
port hex, the port bootstraps value 0, and each trial is an episode with the
eligibility trace reset between trials.

Paper-exact model-free preset:

    HexMazeTDLearner(
        maze, reward_probs,
        lam=0.0, directional=True, goal_conditioned=False,
        priors=("flat", 0.2),
    )

Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
"""

import random
import warnings
import numpy as np
from scipy.optimize import minimize
from ...utils import create_empty_hex_maze, maze_to_graph
from ...core import get_safe_hex_distance
from ...utils import REWARD_PORTS, resolve_port


class HexMazeTDLearner:
    """TD(lambda) hex-value learner. See module docstring for the flags."""

    def __init__(
        self,
        maze,
        reward_probs,
        alpha=0.3,
        gamma=0.95,
        lam=0.0,
        temperature=1.0,
        directional=False,
        goal_conditioned=True,
        priors=None,
        no_backtrack=False,
    ):
        """
        Parameters
        ----------
        maze : set, frozenset, list, np.ndarray, str, or networkx.Graph
            The hex maze in any valid format (a set of barrier hexes, a
            comma-separated string, a networkx graph, etc.).
        reward_probs : list of float
            [p1, p2, p3] reward probability at ports 1/A, 2/B, 3/C.
        alpha : float
            TD learning rate.
        gamma : float
            Discount factor.
        lam : float
            Eligibility-trace decay (TD-lambda). 0 = TD(0), 1 = Monte Carlo.
        temperature : float
            Softmax temperature for action selection.
        directional : bool
            If True, states are directed edges (prev_hex, cur_hex); else hexes.
        goal_conditioned : bool
            If True, keep one value function per start port (3 value tables);
            else a single shared value function.
        priors : None, "uniform", ("flat", value), or list of 3 floats
            Value initialization strategy:
            - None: all zeros
            - "uniform": 0.5-weighted gamma^distance toward goal ports
            - ("flat", value): constant value for every state
            - [p1, p2, p3]: per-port priors with gamma^distance discounting
        no_backtrack : bool
            If True, the agent avoids revisiting states within a trial if possible
            (can be useful for examples in simulate mode).
        """
        self.graph = maze_to_graph(maze)
        self.reward_probs = {i + 1: reward_probs[i] for i in range(3)}
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.temperature = temperature
        self.directional = directional
        self.goal_conditioned = goal_conditioned
        self.no_backtrack = no_backtrack
        self.priors = priors

        # Contexts: one value table per start port, or a single shared one.
        self.contexts = list(REWARD_PORTS) if goal_conditioned else [None]

        self.prior_table = None  # {context: {hex: value}} or None
        self.V = {}
        self.reset()

    #  Setup / priors

    def reset(self):
        """Clear all value tables and rebuild priors. States are created lazily."""
        self.V = {context: {} for context in self.contexts}
        self.build_prior_table(self.priors)
        # Only the session's very first trial may lack a real start port.
        self._is_first_trial = True

    def build_prior_table(self, priors):
        """Precompute {context: {hex: prior value}} (or None for all-zeros)."""
        if priors is None:
            self.prior_table = None
        elif priors == "uniform":
            self.prior_table = self.distance_priors([0.5, 0.5, 0.5])
        elif isinstance(priors, tuple) and len(priors) == 2 and priors[0] == "flat":
            value = priors[1]
            self.prior_table = {
                context: {hex: value for hex in self.graph.nodes()} for context in self.contexts
            }
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
        empty_maze = create_empty_hex_maze()
        port_value = {i + 1: port_values[i] for i in range(3)}
        table = {}
        for context in self.contexts:
            goals = [port for port in REWARD_PORTS if port != context]  # context=None -> all ports
            table[context] = {}
            for hex in self.graph.nodes():
                if hex in goals:
                    table[context][hex] = port_value[hex]
                else:
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
        stale states are dropped.
        """
        self.graph = maze_to_graph(new_maze)
        valid = set(self.graph.nodes())
        for context in self.contexts:
            for state in list(self.V[context]):
                if self.hex_of_state(state) not in valid:
                    del self.V[context][state]

    #  State helpers

    def context_for_port(self, start_port):
        """The value table key for a trip starting at start_port."""
        return start_port if self.goal_conditioned else None

    def state_key(self, prev_hex, cur_hex):
        """State key for arriving at `cur_hex` from `prev_hex` (None at trip start)."""
        return (prev_hex, cur_hex) if self.directional else cur_hex

    def hex_of_state(self, state):
        """The maze hex underlying a state key."""
        return state[1] if self.directional else state

    def state_value(self, context, state):
        """Read a state's value, falling back to its prior if never written."""
        value = self.V[context].get(state)
        return value if value is not None else self.prior_for_hex(context, self.hex_of_state(state))

    #  TD(lambda) core

    def apply_td_error(self, context, state, delta, eligibility):
        """
        Apply one TD error through the eligibility trace: bump the current
        state's trace, update every traced state, then decay all traces.
        """
        eligibility[state] = eligibility.get(state, 0.0) + 1.0
        decay = self.gamma * self.lam
        for traced_state in list(eligibility):
            self.V[context][traced_state] = (
                self.state_value(context, traced_state)
                + self.alpha * delta * eligibility[traced_state]
            )
            eligibility[traced_state] *= decay
            if eligibility[traced_state] < 1e-6:
                del eligibility[traced_state]

    def learn_path(self, path, reward, context, record=False):
        """
        Run a single TD(lambda) pass over a known path within one context.

        Reward is delivered at the terminal state (path[-1]). Returns a list of
        per-step snapshots when record=True, else None.
        """
        history = []
        if record:
            history.append(self.snapshot(path, 0))

        eligibility = {}
        last_step = len(path) - 2  # index of the final transition

        for step in range(len(path) - 1):
            prev_hex = path[step - 1] if step > 0 else None
            cur_hex, next_hex = path[step], path[step + 1]
            state = self.state_key(prev_hex, cur_hex)
            next_state = self.state_key(cur_hex, next_hex)

            if step == last_step:
                # Terminal transition: reward is delivered here and the terminal
                # state is bootstrapped at 0 (ports are terminal, as in the paper;
                # a mid-maze trajectory end is treated the same way).
                self.apply_td_error(context, state, reward - self.state_value(context, state), eligibility)
                # Record the terminal state's reward expectation without bootstrapping from it.
                self.V[context][next_state] = self.state_value(context, next_state) + self.alpha * (
                    reward - self.state_value(context, next_state)
                )
            else:
                # Ordinary rewardless transition: bootstrap from the next state.
                self.apply_td_error(
                    context,
                    state,
                    self.gamma * self.state_value(context, next_state) - self.state_value(context, state),
                    eligibility,
                )

            if record:
                history.append(self.snapshot(path, step + 1))

        return history if record else None

    #  Learn from supplied trajectories

    def resolve_context(self, path):
        """
        The value-table context for a trip, derived from path[0].

        With goal_conditioned=False there is a single shared table, so
        path[0] never needs to be a real port. With goal_conditioned=True,
        the context is path[0] if it's a reward port. If not, a placeholder
        is only allowed on the session's very first trial (which starts
        wherever the rat happened to be, before ever reaching a port): one
        of the other two ports, excluding the trip's end port (path[-1]).
        Any later trial without a real start port raises, since every trial
        after the first should start where the previous one ended.
        """
        is_first_trial = self._is_first_trial
        self._is_first_trial = False

        if not self.goal_conditioned:
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

    def process_trajectory(self, path, reward):
        """
        Run a TD(lambda) update along a single path.

        Parameters
        ----------
        path : list of int
            Sequence of hexes visited. With goal_conditioned=True, the start
            port is taken from path[0] if it's a reward port, else a
            placeholder (see resolve_context); ignored (and not required)
            when goal_conditioned=False.
        reward : float
            Reward obtained at the terminal state (path[-1]).
        """
        self.learn_path(path, reward, self.resolve_context(path))

    def process_trajectory_with_history(self, path, reward):
        """
        Same as process_trajectory, but returns a per-step snapshot of the
        per-hex value tables (one entry per visited hex; see snapshot()).
        """
        return self.learn_path(path, reward, self.resolve_context(path), record=True)

    def learn(self, trajectories, rewards, record=None):
        """
        Run TD updates over a batch of externally-provided trajectories.

        Parameters
        ----------
        trajectories : list of list of int
            Each path [s0, s1, ..., s_terminal]. See resolve_context for how
            the start port is determined when path[0] isn't one.
        rewards : list of float
            Reward for each trajectory.
        record : {None, "trial", "step"}, optional
            Snapshot granularity to return (each snapshot is the same shape
            as snapshot(); collapse one with snapshot_values()):
                - None (default): no return value, just runs the updates.
                - "trial": one snapshot per trial, taken after that trial's
                  update.
                - "step": one snapshot per hex-step, across every trial (the
                  full within-trial history, concatenated over the session).

        Returns
        -------
        None, or a list of snapshots, depending on `record`.
        """
        if record not in (None, "trial", "step"):
            raise ValueError(f"record must be None, 'trial', or 'step', got {record!r}")

        history = []
        for path, reward in zip(trajectories, rewards):
            if len(path) < 2:
                continue
            trial_history = self.learn_path(path, reward, self.resolve_context(path), record=bool(record))
            if record == "step":
                history.extend(trial_history)
            elif record == "trial":
                history.append(trial_history[-1])

        return history if record else None

    #  Fitting to observed choices

    def choice_nll(self, trajectories, rewards, record=False):
        """
        Negative log-likelihood of the rat's hex-to-hex choices under this
        model's current parameters.

        Replays each trajectory, scoring the softmax probability of the hex
        the rat actually stepped to (before that step's TD update), then runs
        the ordinary TD(lambda) update so values evolve as the replay
        proceeds. This is different from scoring reward outcomes: it measures
        how well the model predicts *which way the rat turned*, not whether
        it got rewarded.

        Parameters
        ----------
        trajectories : list of list of int
            Each path [s0, s1, ..., s_terminal]. See resolve_context for how
            the start port is determined when path[0] isn't one.
        rewards : list of float
            Reward for each trajectory.
        record : bool, optional
            If True, also return a list of per-choice records, one per
            hex-to-hex move across the whole session, in trajectory order:
            {"entry": prev_hex (None at a trial's first step), "hex": cur_hex,
            "choice": next_hex, "probability": p_choice, "probabilities":
            {neighbor: prob, ...}}. Combine "entry"/"hex"/"choice" with
            core.get_hex_exit_direction() to label each choice "left"/
            "right"/"back".

        Returns
        -------
        float, or (float, list of dict) if record=True
            Total negative log-likelihood of the observed hex choices, and
            (if record=True) the per-choice records.
        """
        total = 0.0
        choices = [] if record else None
        for path, reward in zip(trajectories, rewards):
            if len(path) < 2:
                continue
            context = self.resolve_context(path)
            visited = {path[0]}

            for step in range(len(path) - 1):
                cur_hex, next_hex = path[step], path[step + 1]
                entry_hex = path[step - 1] if step > 0 else None
                neighbors = self.get_neighbors(cur_hex, visited)
                if next_hex not in neighbors:
                    # The rat took a step the no_backtrack heuristic would have
                    # excluded; fall back to the full neighbor set so scoring
                    # never crashes on real (possibly backtracking) data.
                    neighbors = list(self.graph.neighbors(cur_hex))
                probabilities = self.softmax_probabilities(cur_hex, neighbors, context)
                probability_by_neighbor = dict(zip(neighbors, probabilities.tolist()))
                p_choice = probability_by_neighbor[next_hex]
                total -= np.log(max(p_choice, 1e-10))
                if record:
                    choices.append({
                        "entry": entry_hex,
                        "hex": cur_hex,
                        "choice": next_hex,
                        "probability": p_choice,
                        "probabilities": probability_by_neighbor,
                    })
                visited.add(next_hex)

            self.learn_path(path, reward, context)
        return (total, choices) if record else total

    @classmethod
    def fit_choices(cls, maze, reward_probs, trajectories, rewards, **kwargs):
        """
        Fit alpha, gamma, lam, and temperature to maximize the likelihood of
        the rat's hex-to-hex choices (not just reward outcomes).

        Uses L-BFGS-B to minimize choice_nll(). Any other constructor flags
        (directional, goal_conditioned, priors, no_backtrack, ...) are held
        fixed at the values passed via **kwargs.

        Parameters
        ----------
        maze, reward_probs : see __init__.
        trajectories : list of list of int
            Each path [s0, s1, ..., s_terminal]. See resolve_context for how
            the start port is determined when path[0] isn't one.
        rewards : list of float
            Reward for each trajectory.
        **kwargs
            Extra constructor flags held fixed during fitting (e.g.
            directional, goal_conditioned, priors, no_backtrack).

        Returns
        -------
        HexMazeTDLearner
            Fresh instance built with the best-fit alpha/gamma/lam/temperature
            (and the fixed **kwargs), carrying:
                - choice_nll_    : choice NLL at optimum
                - choice_bic_    : BIC (4 params: alpha, gamma, lam, temperature)
                - choice_result_ : raw scipy OptimizeResult
        """
        def _obj(params):
            alpha, gamma, lam, temperature = params
            model = cls(maze, reward_probs, alpha=alpha, gamma=gamma, lam=lam,
                        temperature=temperature, **kwargs)
            return model.choice_nll(trajectories, rewards)

        result = minimize(_obj, x0=[0.3, 0.9, 0.3, 1.0],
                          bounds=[(1e-3, 1.0), (0.0, 0.999), (0.0, 1.0), (0.01, 10.0)],
                          method='L-BFGS-B')

        alpha, gamma, lam, temperature = result.x
        fitted = cls(maze, reward_probs, alpha=alpha, gamma=gamma, lam=lam,
                    temperature=temperature, **kwargs)
        fitted.choice_nll_ = result.fun
        n_choices = sum(len(path) - 1 for path in trajectories)
        fitted.choice_bic_ = len(result.x) * np.log(n_choices) + 2 * result.fun
        fitted.choice_result_ = result
        return fitted

    #  Self-generated simulation

    def simulate(self, start_state, n_trials=65, max_steps=200, record_history=False):
        """
        Run n_trials of self-generated exploration with TD updates. Each trial
        starts from the previous trial's terminal state. Returns a list of
        {"path", "reward", "start_port"} dicts.

        When record_history=True, each result dict also carries a "history": the
        per-step list of value snapshots (see snapshot()) captured as the TD
        update propagates along that trial's path. Use this to build step-by-step
        learning animations (snapshot_values() collapses a snapshot to {hex: value}).
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
        Roll out one trial under the current policy, then apply a TD(lambda) update.

        Returns (path, reward, history), where history is the per-step value
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

    def softmax_probabilities(self, hex, neighbors, context):
        """Softmax over the value of the state reached by moving to each neighbor."""
        values = np.array([self.state_value(context, self.state_key(hex, neighbor)) for neighbor in neighbors])
        scaled = values / self.temperature
        scaled -= scaled.max()
        exponentiated = np.exp(scaled)
        return exponentiated / exponentiated.sum()

    def sample_reward(self, hex):
        """Sample a binary reward at a reward port."""
        if hex in self.reward_probs and random.random() < self.reward_probs[hex]:
            return 1.0
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
        straight out from start_port (see bfs_entry_directions). Hexes with
        no learned state fall back to their prior.
        """
        port = resolve_port(start_port)
        context = self.context_for_port(port) if self.goal_conditioned else None
        entry_directions = self.bfs_entry_directions(port) if self.directional else None
        return self.forward_hex_values(context, entry_directions)

    def get_max_state_values(self):
        """{hex: max value across the 3 ports' outbound views}."""
        per_port = [self.get_state_values(port) for port in REWARD_PORTS]
        return {hex: max(table[hex] for table in per_port) for hex in self.graph.nodes()}

    def bfs_entry_directions(self, origin, prev_hex=None):
        """
        BFS outward from `origin`, having just arrived from `prev_hex` (or
        None if `origin` has no real predecessor, e.g. a trip's starting
        port). Returns {hex: predecessor} -- the direction you'd enter each
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

    def forward_hex_values(self, context, entry_directions):
        """
        {hex: value} for a context, given a precomputed {hex: prev_hex}
        entry-direction map (see resolve_snapshot_entry_directions). With
        directional=False there's only one state per hex, so entry
        directions don't matter and this just reads V[context] directly.
        """
        if not self.directional:
            return {
                hex: self.V[context].get(hex, self.prior_for_hex(context, hex)) for hex in self.graph.nodes()
            }
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
