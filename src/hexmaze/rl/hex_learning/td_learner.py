"""
td_learner.py

TD(lambda) value-propagation agent for the hex maze.

Model-free component of a Krausz et al. 2023-style hex-value learner
("Dual credit assignment processes underlie dopamine signals in a complex
spatial environment", Neuron). Value is learned over maze locations via
temporal-difference learning with eligibility traces. The single ``lam``
(lambda) knob spans the family:

    lam = 0.0  -> pure TD(0): one-step bootstrapping, value propagates
                 backward one hex per repeated traversal (the paper's
                 model-free process).
    lam = 1.0  -> Monte-Carlo: full discounted return assigned along the
                 whole path within a single trial.
    0 < lam<1  -> eligibility-trace blend of all intermediate horizons.

Two orthogonal representation flags let you trade off paper-fidelity against
sensible self-generated behavior:

    directional : bool
        False -> value over hexes (49 states), V[hex].
        True  -> value over directional hex-states, i.e. directed edges
                 (prev_hex, cur_hex) ~ 126 states. This is the paper's
                 representation and is what produces approach-dependent ramps.

    goal_conditioned : bool
        False -> a single shared value function (the paper's choice). Faithful
                 for *fitting* observed trajectories, but when used to *generate*
                 behavior it will happily run back up the value gradient to the
                 port it just left.
        True  -> one value function per context, keyed by the start/excluded
                 port (3 tables). The just-departed port is not a reward target
                 on the current trip, so its basin is not attractive. Use this
                 for simulate().

Reward ports are always terminal (the paper's treatment): reward is delivered
on the transition into the port, the port bootstraps value 0, and each trip is
an episode with the eligibility trace reset between trips.

Paper-exact model-free preset:

    HexMazeTDLearner(
        graph, reward_probs,
        lam=0.0, directional=True, goal_conditioned=False,
        priors=("flat", 0.2),
    )

Reward ports can be specified as 1, 2, 3 or "A", "B", "C".
"""

import random
import numpy as np
from ...utils import create_empty_hex_maze
from ...core import get_safe_hex_distance
from ...utils import REWARD_PORTS, resolve_port


class HexMazeTDLearner:
    """TD(lambda) hex-value learner. See module docstring for the flags."""

    def __init__(
        self,
        graph,
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
        graph : networkx.Graph
            Hex maze graph (hexes 1-49, edges are adjacencies).
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
            If True, keep one value function per start/excluded port (3 tables);
            else a single shared value function.
        priors : None, "uniform", ("flat", value), or list of 3 floats
            Value initialization strategy:
            - None: all zeros
            - "uniform": 0.5-weighted gamma^distance toward goal ports
            - ("flat", value): constant value for every state
            - [p1, p2, p3]: per-port priors with gamma^distance discounting
        no_backtrack : bool
            If True, the agent avoids revisiting states within a trial (simulate).
        """
        self.graph = graph
        self.reward_probs = {i + 1: reward_probs[i] for i in range(3)}
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.temperature = temperature
        self.directional = directional
        self.goal_conditioned = goal_conditioned
        self.no_backtrack = no_backtrack
        self._priors = priors

        # Contexts: one value table per excluded/start port, or a single shared one.
        self.contexts = list(REWARD_PORTS) if goal_conditioned else [None]

        self._prior_table = None  # {context: {hex: value}} or None
        self.V = {}
        self.reset()

    #  Setup / priors

    def reset(self):
        """Clear all value tables and rebuild priors. States are created lazily."""
        self.V = {ctx: {} for ctx in self.contexts}
        self._build_prior_table(self._priors)

    def _build_prior_table(self, priors):
        """Precompute {context: {hex: prior value}} (or None for all-zeros)."""
        if priors is None:
            self._prior_table = None
        elif priors == "uniform":
            self._prior_table = self._distance_priors([0.5, 0.5, 0.5])
        elif isinstance(priors, tuple) and len(priors) == 2 and priors[0] == "flat":
            value = priors[1]
            self._prior_table = {
                ctx: {h: value for h in self.graph.nodes()} for ctx in self.contexts
            }
        elif isinstance(priors, (list, tuple)) and len(priors) == 3:
            self._prior_table = self._distance_priors(list(priors))
        else:
            raise ValueError(
                f"priors must be None, 'uniform', ('flat', value), or [p1, p2, p3], got {priors!r}"
            )

    def _distance_priors(self, port_values):
        """
        Build distance-discounted priors per context.

        For each context, value(hex) = max over that context's goal ports of
        port_value[goal] * gamma^(distance from hex to goal), using the empty maze
        for distances. A context's goals are all ports except its key (the excluded
        port); a non-goal-conditioned (None) context uses all three ports as goals.
        """
        empty_maze = create_empty_hex_maze()
        pv = {i + 1: port_values[i] for i in range(3)}
        table = {}
        for ctx in self.contexts:
            goals = [p for p in REWARD_PORTS if p != ctx]  # ctx=None -> all ports
            table[ctx] = {}
            for h in self.graph.nodes():
                if h in goals:
                    table[ctx][h] = pv[h]
                else:
                    table[ctx][h] = max(
                        (pv[g] * (self.gamma ** get_safe_hex_distance(empty_maze, start_hex=h, target_hex=g))
                         for g in goals),
                        default=0.0,
                    )
        return table

    def _prior_for_hex(self, ctx, hex):
        """Prior value for a hex in a context (0.0 when no priors set)."""
        if self._prior_table is None:
            return 0.0
        return self._prior_table[ctx].get(hex, 0.0)

    def set_graph(self, new_graph):
        """
        Swap the maze graph (e.g. after a barrier change). States are lazily
        re-created against the new graph; stale states are dropped.
        """
        self.graph = new_graph
        valid = set(new_graph.nodes())
        for ctx in self.contexts:
            for state in list(self.V[ctx]):
                if self._hex_of(state) not in valid:
                    del self.V[ctx][state]

    #  State helpers

    def _ctx(self, start_port):
        """The value table key for a trip starting at start_port."""
        return start_port if self.goal_conditioned else None

    def _state(self, prev, cur):
        """State key for arriving at `cur` from `prev` (None at trip start)."""
        return (prev, cur) if self.directional else cur

    def _hex_of(self, state):
        """The maze hex underlying a state key."""
        return state[1] if self.directional else state

    def _val(self, ctx, state):
        """Read a state's value, lazily initializing it to its prior."""
        table = self.V[ctx]
        v = table.get(state)
        if v is None:
            v = self._prior_for_hex(ctx, self._hex_of(state))
            table[state] = v
        return v

    #  TD(lambda) core

    def _trace_step(self, ctx, state, delta, e):
        """
        Apply one TD error through the eligibility trace: bump the current
        state's trace, update every traced state, then decay all traces.
        """
        e[state] = e.get(state, 0.0) + 1.0
        decay = self.gamma * self.lam
        for s in list(e):
            self.V[ctx][s] = self._val(ctx, s) + self.alpha * delta * e[s]
            e[s] *= decay
            if e[s] < 1e-6:
                del e[s]

    def _learn_path(self, path, reward, ctx, record=False):
        """
        Run a single TD(lambda) pass over a known path within one context.

        Reward is delivered at the terminal state (path[-1]). Returns a list of
        per-step snapshots when record=True, else None.
        """
        history = []
        if record:
            history.append(self._snapshot(path[0]))

        e = {}
        last_t = len(path) - 2  # index of the final transition

        for t in range(len(path) - 1):
            prev = path[t - 1] if t > 0 else None
            cur, nxt = path[t], path[t + 1]
            state = self._state(prev, cur)
            next_state = self._state(cur, nxt)

            if t == last_t:
                # Terminal transition: reward is delivered here and the terminal
                # state is bootstrapped at 0 (ports are terminal, as in the paper;
                # a mid-maze trajectory end is treated the same way).
                self._trace_step(ctx, state, reward - self._val(ctx, state), e)
                # Record the terminal state's reward expectation without bootstrapping from it.
                self.V[ctx][next_state] = self._val(ctx, next_state) + self.alpha * (
                    reward - self._val(ctx, next_state)
                )
            else:
                # Ordinary rewardless transition: bootstrap from the next state.
                self._trace_step(ctx, state, self.gamma * self._val(ctx, next_state) - self._val(ctx, state), e)

            if record:
                history.append(self._snapshot(nxt))

        return history if record else None

    #  Learn from supplied trajectories

    def _resolve_start_port(self, path, start_port):
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
        start_port = self._resolve_start_port(path, start_port)
        self._learn_path(path, reward, self._ctx(start_port))

    def process_trajectory_with_history(self, path, reward, start_port=None):
        """
        Same as process_trajectory, but returns a per-step snapshot of the
        collapsed per-hex value tables (one entry per visited hex).
        """
        start_port = self._resolve_start_port(path, start_port)
        return self._learn_path(path, reward, self._ctx(start_port), record=True)

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
        current_state = start_state
        for _ in range(n_trials):
            if current_state in REWARD_PORTS:
                start_port = current_state
                goal_states = [p for p in REWARD_PORTS if p != current_state]
            else:
                start_port = REWARD_PORTS[0]
                goal_states = list(REWARD_PORTS)

            path, reward = self.run_trial(current_state, start_port, goal_states, max_steps)
            results.append({"path": path, "reward": reward, "start_port": start_port})
            current_state = path[-1]
        return results

    def run_trial(self, start_state, start_port, goal_states, max_steps):
        """Roll out one trial under the current policy, then apply a TD(lambda) update."""
        ctx = self._ctx(start_port)
        state = start_state
        path = [state]
        visited = {state}
        reward = 0.0

        for _ in range(max_steps):
            nxt = self.choose_action(state, ctx, visited)
            if nxt is None:
                break
            path.append(nxt)
            visited.add(nxt)
            if nxt in goal_states:
                reward = self.sample_reward(nxt)
                break
            state = nxt

        self._learn_path(path, reward, ctx)
        return path, reward

    #  Action selection

    def get_neighbors(self, state, visited=None):
        """Available neighbors of `state`, respecting no_backtrack."""
        neighbors = list(self.graph.neighbors(state))
        if self.no_backtrack and visited is not None:
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                return unvisited
        return neighbors

    def choose_action(self, state, ctx, visited=None):
        """Pick the next hex via softmax over the value of entering each neighbor."""
        neighbors = self.get_neighbors(state, visited)
        if not neighbors:
            return None
        probs = self._softmax_probs(state, neighbors, ctx)
        return int(np.random.choice(neighbors, p=probs))

    def _softmax_probs(self, state, neighbors, ctx):
        """Softmax over the value of the state reached by moving to each neighbor."""
        vals = np.array([self._val(ctx, self._state(state, n)) for n in neighbors])
        scaled = vals / self.temperature
        scaled -= scaled.max()
        exp_v = np.exp(scaled)
        return exp_v / exp_v.sum()

    def sample_reward(self, state):
        """Sample a binary reward at a reward port."""
        if state in self.reward_probs and random.random() < self.reward_probs[state]:
            return 1.0
        return 0.0

    #  Inspection

    def action_probabilities(self, state, start_port):
        """
        Softmax choice probabilities at a hex under a given context.

        Returns {neighbor_hex: probability}.
        """
        ctx = self._ctx(resolve_port(start_port))
        neighbors = list(self.graph.neighbors(state))
        if not neighbors:
            return {}
        probs = self._softmax_probs(state, neighbors, ctx)
        return dict(zip(neighbors, probs.tolist()))

    def get_state_values(self, start_port, reduce="max"):
        """
        Collapsed per-hex values {hex: value} for a context.

        With directional states, each hex is aggregated over its incoming
        directed-edge states via `reduce` ("max" or "mean"). Hexes with no
        learned state fall back to their prior.
        """
        ctx = self._ctx(resolve_port(start_port))
        return self._collapse(ctx, reduce)

    def get_max_state_values(self, reduce="max"):
        """{hex: max collapsed value across all contexts}."""
        per_ctx = [self._collapse(ctx, reduce) for ctx in self.contexts]
        return {h: max(d[h] for d in per_ctx) for h in self.graph.nodes()}

    def _collapse(self, ctx, reduce="max"):
        """Aggregate a context's value table down to {hex: value}."""
        agg = {h: [] for h in self.graph.nodes()}
        for state, value in self.V[ctx].items():
            h = self._hex_of(state)
            if h in agg:
                agg[h].append(value)
        reducer = np.max if reduce == "max" else np.mean
        return {
            h: (float(reducer(vals)) if vals else self._prior_for_hex(ctx, h))
            for h, vals in agg.items()
        }

    def _snapshot(self, state_hex):
        """One history entry: current hex plus collapsed per-context value maps."""
        return {
            "state": state_hex,
            "values": {ctx: self._collapse(ctx) for ctx in self.contexts},
        }
