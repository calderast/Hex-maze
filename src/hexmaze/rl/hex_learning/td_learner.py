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
import numpy as np
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
        """Read a state's value, lazily initializing it to its prior."""
        table = self.V[context]
        value = table.get(state)
        if value is None:
            value = self.prior_for_hex(context, self.hex_of_state(state))
            table[state] = value
        return value

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
            history.append(self.snapshot(path[0]))

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
                history.append(self.snapshot(next_hex))

        return history if record else None

    #  Learn from supplied trajectories

    def resolve_start_port(self, path, start_port):
        """Resolve start_port from the argument or path[0] (which must be a port)."""
        if start_port is not None:
            return resolve_port(start_port)
        if path[0] in REWARD_PORTS:
            return path[0]
        raise ValueError(
            f"start_port not provided and path[0]={path[0]} is not a reward port. "
            f"Provide start_port explicitly when the trajectory doesn't start at a port."
        )

    def process_trajectory(self, path, reward, start_port=None):
        """
        Run a TD(lambda) update along a single path.

        Parameters
        ----------
        path : list of int
            Sequence of hexes visited.
        reward : float
            Reward obtained at the terminal state (path[-1]).
        start_port : int or str, optional
            Port the trip started from (1/2/3 or A/B/C). Defaults to path[0]
            if it is a reward port.
        """
        start_port = self.resolve_start_port(path, start_port)
        self.learn_path(path, reward, self.context_for_port(start_port))

    def process_trajectory_with_history(self, path, reward, start_port=None):
        """
        Same as process_trajectory, but returns a per-step snapshot of the
        collapsed per-hex value tables (one entry per visited hex).
        """
        start_port = self.resolve_start_port(path, start_port)
        return self.learn_path(path, reward, self.context_for_port(start_port), record=True)

    def learn(self, trajectories, rewards, start_ports=None):
        """
        Run TD updates over a batch of externally-provided trajectories.

        Parameters
        ----------
        trajectories : list of list of int
            Each path [s0, s1, ..., s_terminal].
        rewards : list of float
            Reward for each trajectory.
        start_ports : list of int or str, optional
            Start port for each trajectory; defaults to each path[0].
        """
        if start_ports is None:
            start_ports = [None] * len(trajectories)
        for path, reward, start_port in zip(trajectories, rewards, start_ports):
            if len(path) >= 2:
                self.process_trajectory(path, reward, start_port)

    #  Self-generated simulation

    def simulate(self, start_state, n_trials=65, max_steps=200):
        """
        Run n_trials of self-generated exploration with TD updates. Each trial
        starts from the previous trial's terminal state. Returns a list of
        {"path", "reward", "start_port"} dicts.
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

            path, reward = self.run_trial(current_hex, start_port, goal_hexes, max_steps)
            results.append({"path": path, "reward": reward, "start_port": start_port})
            current_hex = path[-1]
        return results

    def run_trial(self, start_hex, start_port, goal_hexes, max_steps):
        """Roll out one trial under the current policy, then apply a TD(lambda) update."""
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

        self.learn_path(path, reward, context)
        return path, reward

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

    def get_state_values(self, start_port, reduce="max"):
        """
        Collapsed per-hex values {hex: value} for a context.

        With directional states, each hex is aggregated over its incoming
        directed-edge states via `reduce` ("max" or "mean"). Hexes with no
        learned state fall back to their prior.
        """
        context = self.context_for_port(resolve_port(start_port))
        return self.collapse_to_hex_values(context, reduce)

    def get_max_state_values(self, reduce="max"):
        """{hex: max collapsed value across all contexts}."""
        per_context = [self.collapse_to_hex_values(context, reduce) for context in self.contexts]
        return {hex: max(table[hex] for table in per_context) for hex in self.graph.nodes()}

    def collapse_to_hex_values(self, context, reduce="max"):
        """Aggregate a context's value table down to {hex: value}."""
        aggregated = {hex: [] for hex in self.graph.nodes()}
        for state, value in self.V[context].items():
            hex = self.hex_of_state(state)
            if hex in aggregated:
                aggregated[hex].append(value)
        reducer = np.max if reduce == "max" else np.mean
        return {
            hex: (float(reducer(values)) if values else self.prior_for_hex(context, hex))
            for hex, values in aggregated.items()
        }

    def snapshot(self, current_hex):
        """One history entry: current hex plus collapsed per-context value maps."""
        return {
            "state": current_hex,
            "values": {context: self.collapse_to_hex_values(context) for context in self.contexts},
        }
