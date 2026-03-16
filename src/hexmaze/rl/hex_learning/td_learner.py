"""
hex_maze_td_learner.py

Minimal TD value-propagation agent for the hex maze.
Supports TD(0) forward updates, TD(1) backward pass,
self-generated trials, and clamped trajectory learning.
"""

import random
import numpy as np
from ...utils import create_empty_hex_maze
from ...core import get_safe_hex_distance, divide_into_thirds


class HexMazeTDLearner:
    """
    TD learner maintaining 3 V-tables (one per starting port) over a hex maze.

    Two modes of learning:
        - simulate(): agent picks actions via softmax, runs TD updates online
        - learn(): feed in existing maze trajectories with rewards
    """

    REWARD_PORTS = [1, 2, 3]

    def __init__(
        self,
        graph,
        reward_probs,
        td0_alpha=0.3,
        gamma=0.95,
        temperature=1.0,
        td1_alpha=0.1,
        priors=None,
        no_backtrack=False,
    ):
        """
        Parameters
        ----------
        graph : networkx.Graph
            Hex maze graph (hexes 1-49, edges are adjacencies).
        reward_probs : list of float
            [p1, p2, p3] reward probability at ports 1, 2, 3.
        td0_alpha : float
            Learning rate for TD(0). Set to 0 to disable TD(0) updates.
        gamma : float
            Discount factor.
        temperature : float
            Softmax temperature for action selection.
        td1_alpha : float
            Learning rate for TD(1). Set to 0 to disable TD(1) updates.
        priors : None, "uniform", ("flat", value), or list of 3 floats
            V-table initialization strategy:
            - None: all zeros
            - "uniform": 0.5 at goal ports, gamma^distance elsewhere (empty maze)
            - ("flat", value): constant value for ALL hexes including start port
            - [p1, p2, p3]: per-port priors with gamma^distance discounting
        no_backtrack : bool
            If True, agent avoids revisiting states within a trial (simulation mode).
        """
        self.graph = graph
        self.reward_probs = {i + 1: reward_probs[i] for i in range(3)}
        self.td0_alpha = td0_alpha
        self.gamma = gamma
        self.temperature = temperature
        self.td1_alpha = td1_alpha
        self.no_backtrack = no_backtrack
        self._priors = priors

        self.maze_thirds = divide_into_thirds(graph)

        self.V = {}
        self.init_v_tables()
        self.apply_priors(priors)


    def init_v_tables(self):
        """Initialize V[port][hex] for all 3 ports and all hexes."""
        for port in self.REWARD_PORTS:
            self.V[port] = {hex: 0.0 for hex in self.graph.nodes()}

    def apply_priors(self, priors):
        """Prior initialization."""
        if priors is None:
            return
        elif priors == "uniform":
            self.apply_distance_priors([0.5, 0.5, 0.5])
        elif isinstance(priors, tuple) and len(priors) == 2 and priors[0] == "flat":
            self.apply_flat_priors(priors[1])
        elif isinstance(priors, (list, tuple)) and len(priors) == 3:
            self.apply_distance_priors(list(priors))
        else:
            raise ValueError(
                f"priors must be None, 'uniform', ('flat', value), or [p1, p2, p3], got {priors!r}"
            )

    def apply_distance_priors(self, port_values):
        """
        Set V[start_port][hex] = max over goal ports of
        port_value[goal] * gamma^(distance from hex to goal).
        Uses the empty maze for distances.
        """
        empty_maze = create_empty_hex_maze()
        pv = {i + 1: port_values[i] for i in range(3)}

        for start_port in self.REWARD_PORTS:
            goal_ports = [p for p in self.REWARD_PORTS if p != start_port]
            for hex in self.graph.nodes():
                if hex in goal_ports:
                    self.V[start_port][hex] = pv[hex]
                else:
                    best = 0.0
                    for goal in goal_ports:
                        dist = get_safe_hex_distance(empty_maze, start_hex=hex, target_hex=goal)
                        best = max(best, pv[goal] * (self.gamma ** dist))
                    self.V[start_port][hex] = best

    def apply_flat_priors(self, value):
        """Set every state in every V-table to the same constant value."""
        for port in self.REWARD_PORTS:
            for hex in self.graph.nodes():
                self.V[port][hex] = value

    #  Reset values / swap maze graph

    def reset(self):
        """Reset all V-tables and re-apply priors."""
        self.init_v_tables()
        self.apply_priors(self._priors)

    def set_graph(self, new_graph):
        """
        Swap the maze graph (e.g. after barrier changes).
        New hexes get the mean of their neighbors' values (TODO: do we like this?).
        Removed hexes are deleted from V-tables.
        """
        self.graph = new_graph
        self.maze_thirds = divide_into_thirds(new_graph)

        for port in self.REWARD_PORTS:
            for hex in new_graph.nodes():
                if hex not in self.V[port]:
                    nbrs = list(new_graph.neighbors(hex))
                    self.V[port][hex] = (
                        float(np.mean([self.V[port].get(n, 0.0) for n in nbrs]))
                        if nbrs else 0.0
                    )
            for hex in list(self.V[port]):
                if hex not in new_graph:
                    del self.V[port][hex]

    # Table update logic
    # Update tables for path-independent hexes for all ports but the target port
    # Or when the rat is in the initial maze third, update for the start port only

    def get_hex_third(self, hex):
        """Return which third (1, 2, 3) a hex belongs to, or 0 for choice hex."""
        for i, third_set in enumerate(self.maze_thirds):
            if hex in third_set:
                return i + 1
        return 0

    def tables_to_update(self, state, start_port):
        """
        Which V-tables to update for a given state.
        - State in same third as start_port (or choice hex): [start_port]
        - State in a different third T: all ports except T
        """
        third = self.get_hex_third(state)
        if third == start_port or third == 0:
            return [start_port]
        return [p for p in self.REWARD_PORTS if p != third]

    #  TD updates

    def td0_step(self, state, reward, next_state, start_port):
        """TD(0): V(s) += td0_alpha * [r + gamma * V(s') - V(s)]."""
        for port in self.tables_to_update(state, start_port):
            V = self.V[port]
            next_v = 0.0 if next_state in self.REWARD_PORTS else V[next_state]
            V[state] += self.td0_alpha * (reward + self.gamma * next_v - V[state])

    def td0_terminal(self, state, reward, start_port):
        """Update a terminal state: V(s) += td0_alpha * (reward - V(s))."""
        for port in self.tables_to_update(state, start_port):
            V = self.V[port]
            V[state] += self.td0_alpha * (reward - V[state])

    def td1_backward(self, path, reward, start_port):
        """
        TD(1) backward pass: for each state s_t (including terminal),
        V(s_t) += td1_alpha * (gamma^(T-t) * reward - V(s_t)).
        """
        T = len(path) - 1
        for t in range(T, -1, -1):
            state = path[t]
            discounted_return = (self.gamma ** (T - t)) * reward
            for port in self.tables_to_update(state, start_port):
                V = self.V[port]
                V[state] += self.td1_alpha * (discounted_return - V[state])

    #  Learn values based on a given hex path

    def _resolve_start_port(self, path, start_port):
        """Resolve start_port: use given value, or default to path[0] if it's a reward port."""
        if start_port is not None:
            return start_port
        if path[0] in self.REWARD_PORTS:
            return path[0]
        raise ValueError(
            f"start_port not provided and path[0]={path[0]} is not a reward port. "
            f"Provide start_port explicitly when the trajectory doesn't start at a port."
        )

    def process_trajectory(self, path, reward, start_port=None):
        """
        Run TD(0) forward updates along a path, then optionally TD(1) backward.

        Parameters
        ----------
        path : list of int
            Sequence of states visited.
        reward : float
            Reward obtained at terminal state (path[-1]).
        start_port : int, optional
            Which port the trial started from. Defaults to path[0] if it's a reward port.
        """
        start_port = self._resolve_start_port(path, start_port)
        if self.td0_alpha:
            for t in range(len(path) - 1):
                s, s_next = path[t], path[t + 1]
                r = reward if t == len(path) - 2 else 0.0
                self.td0_step(s, r, s_next, start_port)

            terminal = path[-1]
            if terminal in self.REWARD_PORTS:
                self.td0_terminal(terminal, reward, start_port)

        if self.td1_alpha:
            self.td1_backward(path, reward, start_port)

    def process_trajectory_with_history(self, path, reward, start_port=None):
        """
        Same as process_trajectory, but records V-table snapshots at each step.

        Parameters
        ----------
        path : list of int
            Sequence of states visited.
        reward : float
            Reward obtained at terminal state (path[-1]).
        start_port : int, optional
            Which port the trial started from. Defaults to path[0] if it's a reward port.

        Returns
        -------
        list of dict
            One entry per step. Each dict has:
            - "state": current hex
            - "values": {port: {hex: value}} snapshot of all V-tables after the update
        """
        start_port = self._resolve_start_port(path, start_port)
        history = []

        # Record initial values before any updates
        history.append({
            "state": path[0],
            "values": {port: self.V[port].copy() for port in self.REWARD_PORTS},
        })

        if self.td0_alpha:
            for t in range(len(path) - 1):
                s, s_next = path[t], path[t + 1]
                r = reward if t == len(path) - 2 else 0.0
                self.td0_step(s, r, s_next, start_port)
                history.append({
                    "state": s_next,
                    "values": {port: self.V[port].copy() for port in self.REWARD_PORTS},
                })

            terminal = path[-1]
            if terminal in self.REWARD_PORTS:
                self.td0_terminal(terminal, reward, start_port)
                history[-1]["values"] = {port: self.V[port].copy() for port in self.REWARD_PORTS}

        if self.td1_alpha:
            self.td1_backward(path, reward, start_port)
            # Record final state after backward pass
            history.append({
                "state": path[-1],
                "values": {port: self.V[port].copy() for port in self.REWARD_PORTS},
            })

        return history

    def learn(self, trajectories, rewards, start_ports=None):
        """
        Run TD updates on externally-provided trajectories.

        Parameters
        ----------
        trajectories : list of list of int
            Each element is a path [s0, s1, ..., s_terminal].
            May start mid-maze, not necessarily at a port.
        rewards : list of float
            Reward for each trajectory (0.0 or 1.0).
        start_ports : list of int, optional
            Which port each trajectory's trial started from.
            Defaults to each trajectory's first state (must be a reward port).
        """
        if start_ports is None:
            start_ports = [None] * len(trajectories)
        for path, reward, start_port in zip(trajectories, rewards, start_ports):
            if len(path) >= 2:
                self.process_trajectory(path, reward, start_port)


    #  Simulate hex paths using softmax action selection

    def simulate(self, start_state, n_trials=65, max_steps=200):
        """
        Run n_trials of self-generated exploration with online TD updates.
        Each trial starts from the previous trial's terminal state.

        Parameters
        ----------
        start_state : int
            Starting state for the first trial.
        n_trials : int
            Number of trials.
        max_steps : int
            Maximum steps per trial before timeout.

        Returns
        -------
        list of dict
            One dict per trial: {"path": [...], "reward": float, "start_port": int}
        """
        results = []
        current_state = start_state

        for _ in range(n_trials):
            if current_state in self.REWARD_PORTS:
                start_port = current_state
                goal_states = [p for p in self.REWARD_PORTS if p != current_state]
            else:
                start_port = 1
                goal_states = list(self.REWARD_PORTS)

            path, reward = self.run_trial(current_state, start_port, goal_states, max_steps)
            results.append({"path": path, "reward": reward, "start_port": start_port})
            current_state = path[-1]

        return results

    def run_trial(self, start_state, start_port, goal_states, max_steps):
        """Execute one self-generated trial with online TD updates."""
        state = start_state
        path = [state]
        visited = {state}
        reward = 0.0

        for _ in range(max_steps):
            action = self.choose_action(state, start_port, visited)
            if action is None:
                break

            path.append(action)
            visited.add(action)

            if action in goal_states:
                reward = self.sample_reward(action)
                if self.td0_alpha:
                    self.td0_step(state, reward, action, start_port)
                    self.td0_terminal(action, reward, start_port)
                break

            if self.td0_alpha:
                self.td0_step(state, 0.0, action, start_port)
            state = action

        if self.td1_alpha:
            self.td1_backward(path, reward, start_port)

        return path, reward

    #  Action selection

    def get_neighbors(self, state, visited=None):
        """Get available neighbors, respecting no_backtrack."""
        neighbors = list(self.graph.neighbors(state))
        if self.no_backtrack and visited is not None:
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                return unvisited
        return neighbors

    def choose_action(self, state, start_port, visited=None):
        """Pick next state via softmax over V-values. Returns None if stuck."""
        neighbors = self.get_neighbors(state, visited)
        if not neighbors:
            return None
        probs = self.softmax_probs(neighbors, start_port)
        return np.random.choice(neighbors, p=probs)

    def softmax_probs(self, neighbors, start_port):
        """Compute softmax probabilities over neighbor V-values."""
        V = self.V[start_port]
        vals = np.array([V[n] for n in neighbors])
        scaled = vals / self.temperature
        scaled -= scaled.max()
        exp_v = np.exp(scaled)
        return exp_v / exp_v.sum()

    def sample_reward(self, state):
        """Sample binary reward at a reward port."""
        if state in self.reward_probs and random.random() < self.reward_probs[state]:
            return 1.0
        return 0.0

    #  Get state values

    def action_probabilities(self, state, start_port):
        """
        Softmax choice probabilities at a state under a given V-table.

        Parameters
        ----------
        state : int
            Maze hex to evaluate.
        start_port : int
            Which V-table to use (1, 2, or 3).

        Returns
        -------
        dict of {int: float}
            {neighbor_hex: probability}
        """
        neighbors = list(self.graph.neighbors(state))
        if not neighbors:
            return {}
        probs = self.softmax_probs(neighbors, start_port)
        return dict(zip(neighbors, probs.tolist()))

    def get_state_values(self, start_port):
        """Return a copy of V[start_port] as {hex: value}."""
        return self.V[start_port].copy()

    def get_max_state_values(self):
        """Return {hex: max value across all 3 V-tables}."""
        return {
            hex: max(self.V[port][hex] for port in self.REWARD_PORTS)
            for hex in self.V[self.REWARD_PORTS[0]]
        }
