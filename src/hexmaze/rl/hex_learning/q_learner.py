"""
Q-learning agent for the hex maze.

Learns Q(hex, action) values — the value of moving to a specific neighbor
from a given hex — using off-policy Q-learning updates.
"""

import random
import numpy as np
from ...utils import create_empty_hex_maze
from ...core import get_safe_hex_distance, divide_into_thirds
from ...utils import REWARD_PORTS, resolve_port


class HexMazeQLearner:
    """
    Q-learner maintaining 3 Q-tables (one per starting port) over a hex maze.

    Two modes of learning:
        - simulate(): agent picks actions via softmax, runs Q-learning updates online
        - learn(): feed in existing maze trajectories with rewards

    Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
    """

    def __init__(
        self,
        graph,
        reward_probs,
        alpha=0.3,
        gamma=0.95,
        temperature=1.0,
        priors=None,
        no_backtrack=False,
    ):
        """
        Parameters
        ----------
        graph : networkx.Graph
            Hex maze graph (hexes 1-49, edges are adjacencies).
        reward_probs : list of float
            [p1, p2, p3] reward probability at ports 1/A, 2/B, 3/C.
        alpha : float
            Learning rate for Q-learning updates.
        gamma : float
            Discount factor.
        temperature : float
            Softmax temperature for action selection.
        priors : None, "uniform", ("flat", value), or list of 3 floats
            Q-table initialization strategy:
            - None: all zeros
            - "uniform": 0.5 at goal ports, gamma^distance elsewhere (empty maze)
            - ("flat", value): constant value for ALL hex-action pairs
            - [p1, p2, p3]: per-port priors with gamma^distance discounting
        no_backtrack : bool
            If True, agent avoids revisiting hexes within a trial (simulation mode).
        """
        self.graph = graph
        self.reward_probs = {i + 1: reward_probs[i] for i in range(3)}
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.no_backtrack = no_backtrack
        self._priors = priors

        self.maze_thirds = divide_into_thirds(graph)

        self.Q = {}
        self.init_q_tables()
        self.apply_priors(priors)

    def init_q_tables(self):
        """Initialize Q[port][hex][action] for all 3 ports, all hexes and their neighbors."""
        for port in REWARD_PORTS:
            self.Q[port] = {}
            for hex in self.graph.nodes():
                self.Q[port][hex] = {
                    neighbor: 0.0 for neighbor in self.graph.neighbors(hex)
                }

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
        Set Q[start_port][hex][action] based on the value of the destination hex.
        Value of a hex = max over goal ports of port_value[goal] * gamma^distance.
        """
        empty_maze = create_empty_hex_maze()
        pv = {i + 1: port_values[i] for i in range(3)}

        # First compute V-values for each hex under each start port
        V = {}
        for start_port in REWARD_PORTS:
            V[start_port] = {}
            goal_ports = [p for p in REWARD_PORTS if p != start_port]
            for hex in self.graph.nodes():
                if hex in goal_ports:
                    V[start_port][hex] = pv[hex]
                else:
                    best = 0.0
                    for goal in goal_ports:
                        dist = get_safe_hex_distance(empty_maze, start_hex=hex, target_hex=goal)
                        best = max(best, pv[goal] * (self.gamma ** dist))
                    V[start_port][hex] = best

        # Set Q(hex, action) = V(action) — the value of the destination hex
        for start_port in REWARD_PORTS:
            for hex in self.graph.nodes():
                for neighbor in self.graph.neighbors(hex):
                    self.Q[start_port][hex][neighbor] = V[start_port][neighbor]

    def apply_flat_priors(self, value):
        """Set every Q-value to the same constant."""
        for port in REWARD_PORTS:
            for hex in self.graph.nodes():
                for neighbor in self.Q[port][hex]:
                    self.Q[port][hex][neighbor] = value

    # Reset / swap maze graph

    def reset(self):
        """Reset all Q-tables and re-apply priors."""
        self.init_q_tables()
        self.apply_priors(self._priors)

    def set_graph(self, new_graph):
        """
        Swap the maze graph (e.g. after barrier changes).
        New hex-action pairs get mean of neighboring Q-values.
        Removed hexes/actions are deleted.
        """
        old_Q = self.Q
        self.graph = new_graph
        self.maze_thirds = divide_into_thirds(new_graph)

        for port in REWARD_PORTS:
            new_table = {}
            for hex in new_graph.nodes():
                new_table[hex] = {}
                for neighbor in new_graph.neighbors(hex):
                    if hex in old_Q[port] and neighbor in old_Q[port][hex]:
                        new_table[hex][neighbor] = old_Q[port][hex][neighbor]
                    else:
                        # Average over existing Q-values for this hex
                        if hex in old_Q[port] and old_Q[port][hex]:
                            new_table[hex][neighbor] = float(
                                np.mean(list(old_Q[port][hex].values()))
                            )
                        else:
                            new_table[hex][neighbor] = 0.0
            self.Q[port] = new_table

    # Table update logic

    def get_hex_third(self, hex):
        """Return which port's third (1, 2, 3) a hex belongs to, or None for choice hex."""
        for i, third_set in enumerate(self.maze_thirds):
            if hex in third_set:
                return REWARD_PORTS[i]
        return None

    def tables_to_update(self, hex, start_port):
        """
        Which Q-tables to update for a given hex.
        - Hex in same third as start_port (or choice hex): [start_port]
        - Hex in a different third T: all ports except T
        """
        third = self.get_hex_third(hex)
        if third == start_port or third is None:
            return [start_port]
        return [p for p in REWARD_PORTS if p != third]

    # Q-learning update

    def q_update(self, hex, action, reward, next_hex, start_port):
        """
        Q-learning update:
        Q(hex, a) += alpha * [r + gamma * max_a' Q(next_hex, a') - Q(hex, a)]

        For terminal hexes (next_hex is a reward port), max Q(next_hex, a') = 0.
        """
        for port in self.tables_to_update(hex, start_port):
            Q = self.Q[port]
            if next_hex in REWARD_PORTS or not Q.get(next_hex):
                max_next_q = 0.0
            else:
                max_next_q = max(Q[next_hex].values())
            Q[hex][action] += self.alpha * (
                reward + self.gamma * max_next_q - Q[hex][action]
            )

    def q_terminal(self, hex, action, reward, start_port):
        """Update for arriving at a terminal hex: Q(hex, a) += alpha * (r - Q(hex, a))."""
        for port in self.tables_to_update(hex, start_port):
            Q = self.Q[port]
            Q[hex][action] += self.alpha * (reward - Q[hex][action])

    # Learn from given trajectories

    def _resolve_start_port(self, path, start_port):
        """Resolve start_port: use given value, or default to path[0] if it's a reward port."""
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
        Run Q-learning updates along a path.

        Parameters
        ----------
        path : list of int
            Sequence of hexes visited.
        reward : float
            Reward obtained at terminal hex (path[-1]).
        start_port : int or str, optional
            Which port the trial started from (1/2/3 or A/B/C).
            Defaults to path[0] if it's a reward port.
        """
        start_port = self._resolve_start_port(path, start_port)
        for t in range(len(path) - 1):
            hex, next_hex = path[t], path[t + 1]
            r = reward if t == len(path) - 2 else 0.0
            self.q_update(hex, next_hex, r, next_hex, start_port)

    def process_trajectory_with_history(self, path, reward, start_port=None):
        """
        Same as process_trajectory, but records Q-table snapshots at each step.

        Returns
        -------
        list of dict
            One entry per step. Each dict has:
            - "hex": current hex
            - "values": {port: {hex: {neighbor: q_value}}} snapshot after the update
        """
        start_port = self._resolve_start_port(path, start_port)
        history = []

        # Record initial values
        history.append({
            "hex": path[0],
            "values": {
                port: {h: dict(actions) for h, actions in self.Q[port].items()}
                for port in REWARD_PORTS
            },
        })

        for t in range(len(path) - 1):
            hex, next_hex = path[t], path[t + 1]
            r = reward if t == len(path) - 2 else 0.0
            self.q_update(hex, next_hex, r, next_hex, start_port)
            history.append({
                "hex": next_hex,
                "values": {
                    port: {h: dict(actions) for h, actions in self.Q[port].items()}
                    for port in REWARD_PORTS
                },
            })

        return history

    def learn(self, trajectories, rewards, start_ports=None):
        """
        Run Q-learning updates on externally-provided trajectories.

        Parameters
        ----------
        trajectories : list of list of int
            Each element is a path of hexes [h0, h1, ..., h_terminal].
        rewards : list of float
            Reward for each trajectory (0.0 or 1.0).
        start_ports : list of int or str, optional
            Which port each trajectory's trial started from (1/2/3 or A/B/C).
            Defaults to each trajectory's first hex (must be a reward port).
        """
        if start_ports is None:
            start_ports = [None] * len(trajectories)
        for path, reward, start_port in zip(trajectories, rewards, start_ports):
            if len(path) >= 2:
                self.process_trajectory(path, reward, start_port)

    # Simulate hex paths using softmax action selection

    def simulate(self, start_hex, n_trials=65, max_steps=200):
        """
        Run n_trials of self-generated exploration with online Q-learning updates.

        Returns
        -------
        list of dict
            One dict per trial: {"path": [...], "reward": float, "start_port": int}
        """
        results = []
        current_hex = start_hex

        for _ in range(n_trials):
            if current_hex in REWARD_PORTS:
                start_port = current_hex
                goal_hexes = [p for p in REWARD_PORTS if p != current_hex]
            else:
                start_port = REWARD_PORTS[0]
                goal_hexes = list(REWARD_PORTS)

            path, reward = self.run_trial(current_hex, start_port, goal_hexes, max_steps)
            results.append({"path": path, "reward": reward, "start_port": start_port})
            current_hex = path[-1]

        return results

    def run_trial(self, start_hex, start_port, goal_hexes, max_steps):
        """Execute one self-generated trial with online Q-learning updates."""
        hex = start_hex
        path = [hex]
        visited = {hex}
        reward = 0.0

        for _ in range(max_steps):
            action = self.choose_action(hex, start_port, visited)
            if action is None:
                break

            path.append(action)
            visited.add(action)

            if action in goal_hexes:
                reward = self.sample_reward(action)
                self.q_update(hex, action, reward, action, start_port)
                break

            self.q_update(hex, action, 0.0, action, start_port)
            hex = action

        return path, reward

    # Action selection

    def get_neighbors(self, hex, visited=None):
        """Get available neighbors, respecting no_backtrack."""
        neighbors = list(self.graph.neighbors(hex))
        if self.no_backtrack and visited is not None:
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                return unvisited
        return neighbors

    def choose_action(self, hex, start_port, visited=None):
        """Pick next hex via softmax over Q-values. Returns None if stuck."""
        neighbors = self.get_neighbors(hex, visited)
        if not neighbors:
            return None
        probs = self.softmax_probs(hex, neighbors, start_port)
        return np.random.choice(neighbors, p=probs)

    def softmax_probs(self, hex, neighbors, start_port):
        """Compute softmax probabilities over Q-values for neighbors."""
        Q = self.Q[start_port][hex]
        vals = np.array([Q[n] for n in neighbors])
        scaled = vals / self.temperature
        scaled -= scaled.max()
        exp_v = np.exp(scaled)
        return exp_v / exp_v.sum()

    def sample_reward(self, hex):
        """Sample binary reward at a reward port."""
        if hex in self.reward_probs and random.random() < self.reward_probs[hex]:
            return 1.0
        return 0.0

    # Get hex/action values

    def action_probabilities(self, hex, start_port):
        """
        Softmax choice probabilities at a hex under a given Q-table.

        Parameters
        ----------
        hex : int
            Maze hex to evaluate.
        start_port : int or str
            Which Q-table to use (1/2/3 or A/B/C).

        Returns
        -------
        dict of {int: float}
            {neighbor_hex: probability}
        """
        start_port = resolve_port(start_port)
        neighbors = list(self.graph.neighbors(hex))
        if not neighbors:
            return {}
        probs = self.softmax_probs(hex, neighbors, start_port)
        return dict(zip(neighbors, probs.tolist()))

    def get_q_values(self, start_port):
        """Return a copy of Q[start_port] as {hex: {action: value}}."""
        start_port = resolve_port(start_port)
        return {h: dict(actions) for h, actions in self.Q[start_port].items()}

    def get_state_values(self, start_port):
        """Return max Q-value at each hex: {hex: max_a Q(hex, a)}."""
        start_port = resolve_port(start_port)
        return {
            hex: max(actions.values()) if actions else 0.0
            for hex, actions in self.Q[start_port].items()
        }

    def get_max_state_values(self):
        """Return {hex: max value across all 3 Q-tables}."""
        all_hexes = self.Q[REWARD_PORTS[0]].keys()
        return {
            hex: max(
                max(self.Q[port][hex].values()) if self.Q[port][hex] else 0.0
                for port in REWARD_PORTS
            )
            for hex in all_hexes
        }
