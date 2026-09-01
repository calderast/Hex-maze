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
import textwrap
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize
from ...utils import create_empty_hex_maze, maze_to_graph
from ...core import get_safe_hex_distance
from ...utils import REWARD_PORTS, resolve_port
from ...plotting import plot_hex_maze


class HexMazeTDLearner:
    """TD(lambda) hex-value learner. See module docstring for the flags."""

    _FIT_PARAM_DEFAULTS = {"alpha": (0.3, (1e-3, 1.0)), "gamma": (0.9, (0.0, 0.999)),
                           "lam": (0.3, (0.0, 1.0)), "temperature": (1.0, (0.01, 10.0))}

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

    def format_state(self, state):
        """Human-readable label for a state key, e.g. 'V(48→43)' or 'V(43)'."""
        if self.directional:
            prev, cur = state
            return f"{prev if prev is not None else 'start'}→{cur}"
        return str(state)

    def state_value(self, context, state):
        """Read a state's value, falling back to its prior if never written."""
        value = self.V[context].get(state)
        return value if value is not None else self.prior_for_hex(context, self.hex_of_state(state))

    #  TD(lambda) core

    def apply_td_error(self, context, state, delta, eligibility, log=None):
        """
        Apply one TD error through the eligibility trace: bump the current
        state's trace, update every traced state, then decay all traces.

        With lambda > 0 this one delta can update many states at once (every
        state still in the trace), each scaled by its own eligibility. 
        
        If `log` (a list) is given, appends one dict per
        updated state: {"state", "eligibility", "old_value", "new_value"}.
        """
        eligibility[state] = eligibility.get(state, 0.0) + 1.0
        decay = self.gamma * self.lam
        for traced_state in list(eligibility):
            e = eligibility[traced_state]
            old_value = self.state_value(context, traced_state)
            new_value = old_value + self.alpha * delta * e
            self.V[context][traced_state] = new_value
            if log is not None:
                log.append({
                    "state": traced_state, "eligibility": e,
                    "old_value": old_value, "new_value": new_value,
                })
            eligibility[traced_state] *= decay
            if eligibility[traced_state] < 1e-6:
                del eligibility[traced_state]

    def learn_path(self, path, reward, context, record=False):
        """
        Run a single TD(lambda) pass over a known path within one context.

        Reward is delivered at the terminal state (path[-1]). Returns a list of
        per-step snapshots when record=True, else None. Each snapshot (other
        than the initial one) carries an "update" dict describing exactly the
        TD update that produced it -- see apply_td_error's callers below for
        the two possible "kind"s ("bootstrap" and "reward").
        """
        history = []
        if record:
            init_snap = self.snapshot(path, 0)
            init_snap["update"] = None
            history.append(init_snap)

        eligibility = {}
        last_step = len(path) - 2  # index of the final transition

        for step in range(len(path) - 1):
            prev_hex = path[step - 1] if step > 0 else None
            cur_hex, next_hex = path[step], path[step + 1]
            state = self.state_key(prev_hex, cur_hex)
            next_state = self.state_key(cur_hex, next_hex)

            # Every hex bootstraps from the next hex's value
            # so we update the value of a hex once we leave it
            # (we need to leave to know what the "next hex" is)
            old_value = self.state_value(context, state)
            next_value = self.state_value(context, next_state)
            delta = self.gamma * next_value - old_value
            log = [] if record else None
            self.apply_td_error(context, state, delta, eligibility, log=log)

            if record:
                snap = self.snapshot(path, step + 1)
                snap["update"] = {
                    "kind": "bootstrap",
                    "state": state,
                    "next_state": next_state,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "old_value": old_value,
                    "next_value": next_value,
                    "delta": delta,
                    "new_value": self.state_value(context, state),
                    "log": log,
                }
                history.append(snap)

            if step == last_step:
                # Reward is delivered on arrival at the port: run one more TD
                # update treating that arrival as its own event, so the port's
                # own value tracks the reward directly. This goes through the
                # same eligibility trace as the update above, so under
                # lambda > 0 the reward also propagates back to recently
                # visited hexes, same as any other TD error.
                old_port_value = self.state_value(context, next_state)
                reward_delta = reward - old_port_value
                reward_log = [] if record else None
                self.apply_td_error(context, next_state, reward_delta, eligibility, log=reward_log)

                if record:
                    snap = self.snapshot(path, step + 1)
                    snap["update"] = {
                        "kind": "reward",
                        "state": next_state,
                        "alpha": self.alpha,
                        "lam": self.lam,
                        "reward": reward,
                        "old_value": old_port_value,
                        "delta": reward_delta,
                        "new_value": self.state_value(context, next_state),
                        "log": reward_log,
                    }
                    history.append(snap)

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

    def junction_candidates(self, cur_hex, entry_hex):
        """
        The two non-backward neighbors at cur_hex if it's a real 3-way
        junction (exactly 3 graph-neighbors), else None. entry_hex (the hex
        arrived from, or None at a trial's first hex) is always excluded,
        regardless of no_backtrack -- see choice_nll.
        """
        all_neighbors = list(self.graph.neighbors(cur_hex))
        if len(all_neighbors) != 3:
            return None
        return [n for n in all_neighbors if n != entry_hex]

    def junction_choice_info(self, cur_hex, entry_hex, next_hex, context):
        """
        Binary junction-choice info for one step, matching choice_nll's
        junctions_only=True scoring: None if cur_hex isn't a real 3-way
        junction, or next_hex is entry_hex (backtracked -- valid, just not a
        binary left/right choice), or next_hex isn't one of the two
        candidates at all (a tracking-error case; callers that need to warn
        about that, like choice_nll, check it themselves). Otherwise:
        {"candidates": [hexA, hexB], "values": {hex: V(hex)}, "probabilities":
        {hex: p}, "choice": next_hex, "probability": p_choice}.
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

    def choice_nll(self, trajectories, rewards, record=False, junctions_only=False):
        """
        Negative log-likelihood of the rat's hex-to-hex choices under this
        model's current parameters.

        Replays each trajectory, scoring the softmax probability of the hex
        the rat actually stepped to (before that step's TD update), then runs
        the ordinary TD(lambda) update so values evolve as the replay
        proceeds.

        Parameters
        ----------
        trajectories : list of list of int
            Each path [s0, s1, ..., s_terminal]. See resolve_context for how
            the start port is determined when path[0] isn't one.
        rewards : list of float
            Reward for each trajectory.
        record : bool, optional
            If True, also return a list of per-choice records, one per
            scored hex-to-hex move, in trajectory order: {"entry": prev_hex
            (None at a trial's first step), "hex": cur_hex, "choice":
            next_hex, "probability": p_choice, "probabilities":
            {neighbor: prob, ...}}. Combine "entry"/"hex"/"choice" with
            core.get_hex_exit_direction() to label each choice "left"/
            "right"/"back".
        junctions_only : bool, optional
            If True, restrict scoring to genuine binary choices: steps where
            the rat is at a real 3-way intersection (cur_hex has exactly 3
            graph-neighbors) and exits through one of the two non-backward
            neighbors (left/right) -- a Krausz 2023-style choice-point
            analysis. A junction where the rat instead backtracked, and
            every non-junction step, is skipped silently: not scored, no
            entry in `choices`, no warning (backtracking is a real,
            unremarkable option -- it's just outside this binary-choice
            definition). If False (default), every step is scored against
            ALL of cur_hex's graph-neighbors, backward included.

        Returns
        -------
        float, or (float, list of dict) if record=True
            Total negative log-likelihood of the scored hex choices, and
            (if record=True) the per-choice records.
        """
        total = 0.0
        choices = [] if record else None
        for path, reward in zip(trajectories, rewards):
            if len(path) < 2:
                continue
            context = self.resolve_context(path)

            for step in range(len(path) - 1):
                cur_hex, next_hex = path[step], path[step + 1]
                entry_hex = path[step - 1] if step > 0 else None

                if junctions_only:
                    candidates = self.junction_candidates(cur_hex, entry_hex)
                    if candidates is None:
                        continue  # not a junction -- outside this analysis
                    if next_hex == entry_hex:
                        continue  # backtracked at the junction -- valid, just not a binary L/R choice
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
                        "entry": entry_hex,
                        "hex": cur_hex,
                        "choice": next_hex,
                        "probability": p_choice,
                        "probabilities": probability_by_neighbor,
                    })

            self.learn_path(path, reward, context)
        return (total, choices) if record else total

    @classmethod
    def fit_choices(cls, maze, reward_probs, trajectories, rewards,
                     alpha=None, gamma=None, lam=None, temperature=None,
                     junctions_only=False, **kwargs):
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
        alpha, gamma, lam, temperature : float or None
            Fix this parameter at the given value instead of fitting it (it's
            excluded from the optimization entirely). None (default) fits it.
            E.g. `fit_choices(..., lam=0.0)` fits alpha/gamma/temperature
            with lam held fixed at 0 (pure TD(0)).
        junctions_only : bool, optional
            Passed to choice_nll -- see its docstring. Restricts fitting to
            genuine binary (left/right) junction choices rather than every
            hex-to-hex step.
        **kwargs
            Extra constructor flags held fixed during fitting (e.g.
            directional, goal_conditioned, priors, no_backtrack).

        Returns
        -------
        HexMazeTDLearner
            Fresh instance built with the best-fit (and any fixed) alpha/
            gamma/lam/temperature (and the fixed **kwargs), carrying:
                - choice_nll_    : choice NLL at optimum
                - choice_bic_    : BIC, counting only the *fitted* params and
                  the choices actually scored (fewer than the number of
                  hex-to-hex steps if junctions_only=True)
                - choice_result_ : raw scipy OptimizeResult, or None if
                  every parameter was fixed (nothing to optimize)
        """
        fixed = {"alpha": alpha, "gamma": gamma, "lam": lam, "temperature": temperature}
        free_names = [name for name, value in fixed.items() if value is None]

        def _build(free_values):
            params = dict(fixed)
            params.update(zip(free_names, free_values))
            return cls(maze, reward_probs, **params, **kwargs)

        if free_names:
            x0 = [cls._FIT_PARAM_DEFAULTS[name][0] for name in free_names]
            bounds = [cls._FIT_PARAM_DEFAULTS[name][1] for name in free_names]
            result = minimize(
                lambda x: _build(x).choice_nll(trajectories, rewards, junctions_only=junctions_only),
                x0=x0, bounds=bounds, method='L-BFGS-B',
            )
            final_values = result.x
        else:
            # Nothing to fit -- every parameter was pinned.
            result = None
            final_values = []

        # One evaluation with record=True gives both the NLL and the actual
        # number of scored choices (needed for BIC -- under junctions_only
        # that's far fewer than the number of hex-to-hex steps). This model
        # is discarded; the returned `fitted` is a fresh, untrained instance
        # at the same parameters, consistent regardless of whether anything
        # was optimized.
        nll, choices = _build(final_values).choice_nll(
            trajectories, rewards, record=True, junctions_only=junctions_only,
        )
        fitted = _build(final_values)
        fitted.choice_nll_ = nll
        n_choices = len(choices)
        fitted.choice_bic_ = (
            len(free_names) * np.log(n_choices) + 2 * nll if n_choices > 0 else float("nan")
        )
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

        Only meaningful with directional=True; with directional=False every
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

    def filtered_update_log(self, update):
        """
        Log entries from an update dict with a real, displayable change
        (>= 0.00005, i.e. not just "0.0000" after rounding -- covers both
        negligible eligibility and negligible delta), sorted by eligibility
        descending. Shared by format_update_text and animate_learning's
        outline so the two always agree on which hexes actually changed.
        """
        if update is None:
            return []
        log = [row for row in update["log"] if abs(row["new_value"] - row["old_value"]) >= 0.00005]
        log.sort(key=lambda row: -row["eligibility"])
        return log

    def format_header_text(self, cur_hex, path_so_far=None):
        """Shared header line(s) for animate_learning's text box: the path
        so far (optional) and current hyperparameters + which hex the rat
        is in."""
        header = (
            f"rat at hex {cur_hex}    "
            f"α={self.alpha:.3g}  γ={self.gamma:.3g}  λ={self.lam:.3g}  τ={self.temperature:.3g}"
        )
        if path_so_far:
            header = f"path: {path_so_far}\n{header}"
        return header

    def format_update_text(self, cur_hex, update, path_so_far=None, max_rows=6):
        """
        Multi-line diagnostic text for one animate_learning frame: the hex
        path walked so far this trial, current hyperparameters, which hex
        the rat is in, and (if an update just happened) the shared TD error
        (delta) for this step plus every state it updated through the
        eligibility trace (see apply_td_error's `log`) -- with lambda > 0 a
        single delta updates many states at once, each scaled by its own
        eligibility, not just the one that triggered it. Rows with no real
        eligibility or no delta (so no actual change) are dropped; the rest
        are sorted by eligibility (most-affected first) and capped at
        `max_rows`, with a summary line for the rest.
        """
        header = self.format_header_text(cur_hex, path_so_far)
        if update is None:
            return header + "\n(no update yet this trial)"

        label = self.format_state(update["state"])
        delta, old = update["delta"], update["old_value"]
        decay = self.gamma * self.lam
        formula_line = f"V(s) ← V(s) + α·δ·e(s)   (e decays ×γλ={decay:.3g} per step back)"

        if update["kind"] == "bootstrap":
            next_label = self.format_state(update["next_state"])
            gamma, next_value = update["gamma"], update["next_value"]
            delta_eq = (
                f"δ = γ·V({next_label}) − V({label})\n"
                f"δ = {gamma:.3g}·{next_value:.4f} − {old:.4f} = {delta:.4f}"
            )
        else:  # "reward"
            reward = update["reward"]
            delta_eq = (
                f"δ = reward − V({label})\n"
                f"δ = {reward:.3g} − {old:.4f} = {delta:.4f}"
            )

        log = self.filtered_update_log(update)
        rows = [
            f"  V({self.format_state(row['state'])})  e={row['eligibility']:.3f}  "
            f"{row['old_value']:.4f} → {row['new_value']:.4f}"
            for row in log[:max_rows]
        ]
        if not rows:
            rows = ["  (no visible change)"]
        elif len(log) > max_rows:
            rows.append(f"  ... +{len(log) - max_rows} more (smaller eligibility)")

        body = f"{formula_line}\n{delta_eq}\n" + "\n".join(rows)
        return header + "\n" + body

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

    def animate_learning(
        self,
        trajectories,
        rewards,
        start_port=None,
        panels=False,
        show_updates=True,
        update_color="red",
        show_choices=True,
        choice_color="yellow",
        show_trial_info=True,
        show_path=True,
        show_equation=True,
        colormap="viridis",
        vmin=0,
        vmax=1,
        interval=150,
        show_hex_labels=False,
        show_barriers=False,
        ax=None,
        **plot_kwargs,
    ):
        """
        Animate TD(lambda) learning, one frame per hex transition.

        Runs learn_path over each (path, reward) trial in order -- exactly as
        learn() would -- capturing a value snapshot after every hex
        transition. Each frame colors the maze by hex value (see
        snapshot_values), places the rat at its current hex facing the
        direction it came from, and (on a trial's last frame, when it ends at
        a real reward port) shows a reward droplet or no-reward X there.

        This runs real TD updates on self as it builds frames, so call
        reset() first for a from-scratch replay.

        Note: this bumps the ``animation.embed_limit`` rcParam (used by
        ``to_jshtml()``) up to at least 512 MB. The matplotlib default is
        20 MB, which most real sessions blow through -- and when it's hit,
        frames are silently dropped (only a warning is logged), so an
        embedded animation can quietly end after just the first few trials.

        Parameters
        ----------
        trajectories, rewards : see learn().
        start_port : None, a port (1/2/3 or "A"/"B"/"C"), or "trial"
            Which context's value table to color by (see snapshot_values).
            None (default): max value across all contexts -- can mask a real
            decrease in one context if another context's value is higher.
            A port: always that one fixed context's table.
            "trial": whichever context the currently-animated trial actually
            belongs to, so the coloring follows the rat's own trip instead of
            a fixed table (drops from an omission are visible when they
            happen). Ignored when panels=True.
        panels : bool
            If True, draw one subplot per context (self.contexts) side by
            side, each always showing that context's own values -- so all
            goal-conditioned tables are visible at once instead of collapsed
            into one view. The rat and reward marker are only drawn on the
            panel for the trial's own context. `start_port` and `ax` are
            ignored in this mode (a fresh figure is created).
        show_updates : bool
            If True (default), outline whichever hex(es) actually had their
            value change on this step -- i.e. a real TD update happened
            there, in the trial's own active context (this is independent of
            what's being displayed: with start_port=None/a fixed port, an
            outlined hex's *displayed* color may not visibly move if a
            different context dominates the max, but the outline still shows
            where the real update occurred). In panels mode, only the active
            context's panel gets outlines, since it's the only table
            actually changing that step.
        update_color : str
            Outline color for updated hexes (see outline_hexes/outline_colors
            in plot_hex_maze). Defaults to 'red'.
        show_choices : bool
            If True (default), whenever the *previous* step was a real
            3-way junction choice (see junction_choice_info -- same
            definition as choice_nll(junctions_only=True), "back" always
            excluded regardless of no_backtrack), outline the two candidate
            hexes and print their retrieved values and choice probabilities,
            with the one actually taken marked. Silent (no info) on
            non-junction steps.
        choice_color : str
            Outline color for the two junction-candidate hexes. Defaults to
            'yellow'.
        show_trial_info : bool
            If True (default), title each frame with the trial number, start
            port, step within the trial, and (once known) the reward outcome.
        show_path : bool
            If True (default), add a line above the trial info showing the
            hex-by-hex path walked so far this trial (e.g. "path: 1 → 4 → 6").
        show_equation : bool
            If True (default), draw a text box each frame with the current
            hyperparameters (alpha, gamma, lambda, temperature), which hex
            the rat is in, and the TD update equation for that step in both
            symbolic and substituted-numbers form (see format_update_text).
        colormap, vmin, vmax, show_hex_labels, show_barriers, ax, **plot_kwargs :
            Forwarded to plot_hex_maze.
        interval : int
            Milliseconds between frames.

        Returns
        -------
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

        valid_trials = [(path, reward) for path, reward in zip(trajectories, rewards) if len(path) >= 2]

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
        for trial_index, (path, reward) in enumerate(valid_trials):
            context = self.resolve_context(path)
            history = self.learn_path(path, reward, context, record=True)
            last_step = len(history) - 1
            show_reward = path[-1] in REWARD_PORTS
            n_steps = len(path) - 1  # real hex-to-hex transitions (the terminal
            # step's reward event is split into an extra same-position frame,
            # not an extra transition, so this is capped rather than counted)
            last_hex, last_rat_from = None, None
            for step_index, snap in enumerate(history):
                cur_hex = snap["state"]
                moved = not (step_index > 0 and cur_hex == last_hex)
                step_label = min(step_index, n_steps)

                # Decision frame: inserted BEFORE a real move out of a
                # genuine 3-way junction, so the choice_color highlight
                # appears while the rat is still AT the junction deciding,
                # not retroactively once it's already at the chosen hex. Uses
                # the *pre*-move snapshot's values (history[step_index-1]),
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
                            "changed_hexes": set(),
                            "junction_candidates": set(junction_info["candidates"]),
                            "title": trial_title(step_label - 1, "  (deciding)"),
                            "equation_text": decision_text,
                            "reward": None,
                        })

                # Arrival frame: the rat has now moved to cur_hex. No
                # junction highlight here -- that belonged to the decision
                # frame above, since the choice has already been made.
                if step_index == 0:
                    rat_from = None
                elif cur_hex == last_hex:
                    # Same physical hex as the previous snapshot (the reward
                    # event's extra frame at the terminal step) -- the rat
                    # hasn't moved, so keep its previous facing.
                    rat_from = last_rat_from
                else:
                    rat_from = last_hex
                last_hex, last_rat_from = cur_hex, rat_from

                changed_hexes = set()
                if show_updates:
                    # Derived from the same filtered log used in the equation
                    # text, so the outline always matches what's printed.
                    changed_hexes = {
                        self.hex_of_state(row["state"])
                        for row in self.filtered_update_log(snap.get("update"))
                    }

                is_reward_step = step_index == last_step and show_reward
                title = trial_title(step_label, "  →  " + ("rewarded" if reward else "omission") if is_reward_step else "")
                equation_text = (
                    self.format_update_text(cur_hex, snap.get("update"), path_so_far=path_line(step_label))
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

        if plt.rcParams["animation.embed_limit"] < 512:
            plt.rcParams["animation.embed_limit"] = 512

        owns_fig = panels or ax is None
        # Reserve dedicated space below the maze for the equation box when we
        # own the figure, so it doesn't overlap the plotted hexes (it can get
        # to ~10 lines with lambda > 0's multi-hex log). If the caller passed
        # their own `ax`, we can't safely resize their figure/layout, so the
        # text falls back to drawing inside the axes bounds in that case.
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
            """(outline_hexes, outline_colors) lists combining the TD-update
            group (update_color) and the junction-candidate group
            (choice_color), or (None, None) if neither applies."""
            groups, colors = [], []
            if frame["changed_hexes"]:
                groups.append(frame["changed_hexes"])
                colors.append(update_color)
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
        if owns_fig:
            plt.close(fig)
        return anim
