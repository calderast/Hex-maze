"""
update_rules.py

Concrete UpdateRule implementations for base_learner.HexMazeAgent (see that
module's docstring for the rule-hook interface and why it's shaped this
way). Each rule owns only its own hyperparameter *names* are read off the
learner instance, not stored on the rule -- see BaseHexLearner.__init__.
Genuinely rule-private state (an eligibility trace, a persistent map) lives
on the rule object itself.
"""

from .base_learner import UpdateRule, UpdateEvent


class TDLambdaRule(UpdateRule):
    """
    TD(lambda) with an eligibility trace -- the same mechanism as
    td_learner.HexMazeTDLearner, reimplemented against the UpdateRule
    interface. Reads `learner.alpha` (default 0.3), `learner.gamma`
    (base-class default 0.95), `learner.lam` (default 0.0) -- alpha/lam
    fall back to these defaults if not passed to the learner's constructor,
    matching HexMazeTDLearner's own defaults.

    On every transition, bootstraps the pre-transition hex from the next
    hex's value. On the terminal transition, additionally runs a second,
    separate update treating the port's own arrival as its own event, so
    the port's value tracks the reward directly and (on the very first
    reward there) only the port itself changes -- the pre-terminal hex only
    catches up once the port already carries value from an earlier visit.
    Both updates go through the same eligibility trace, so under lambda > 0
    a single delta can update many recently-visited states at once.
    """

    DEFAULT_ALPHA = 0.3
    DEFAULT_LAM = 0.0

    def _alpha(self, learner):
        return getattr(learner, "alpha", self.DEFAULT_ALPHA)

    def _lam(self, learner):
        return getattr(learner, "lam", self.DEFAULT_LAM)

    def on_trial_start(self, learner, path, context):
        self._eligibility = {}

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        events = []

        old_value = learner.state_value(context, state)
        next_value = learner.state_value(context, next_state)
        delta = learner.gamma * next_value - old_value
        log = []
        self._apply_td_error(learner, context, state, delta, log)
        events.append(self._make_event(
            learner, "bootstrap", state=state, next_state=next_state,
            old_value=old_value, next_value=next_value, delta=delta, log=log,
        ))

        if is_terminal:
            old_port_value = learner.state_value(context, next_state)
            reward_delta = reward - old_port_value
            reward_log = []
            self._apply_td_error(learner, context, next_state, reward_delta, reward_log)
            events.append(self._make_event(
                learner, "reward", state=next_state, reward=reward,
                old_value=old_port_value, delta=reward_delta, log=reward_log,
            ))

        return events

    def _apply_td_error(self, learner, context, state, delta, log):
        """Bump `state`'s trace, update every traced state, then decay all
        traces. With lambda > 0 one delta can update many states at once,
        each scaled by its own eligibility -- not just `state`."""
        self._eligibility[state] = self._eligibility.get(state, 0.0) + 1.0
        decay = learner.gamma * self._lam(learner)
        for traced_state in list(self._eligibility):
            e = self._eligibility[traced_state]
            old_value = learner.state_value(context, traced_state)
            new_value = old_value + self._alpha(learner) * delta * e
            learner.V[context][traced_state] = new_value
            log.append({
                "state": traced_state, "eligibility": e,
                "old_value": old_value, "new_value": new_value,
            })
            self._eligibility[traced_state] *= decay
            if self._eligibility[traced_state] < 1e-6:
                del self._eligibility[traced_state]

    def _make_event(self, learner, kind, **info):
        log = info["log"]
        filtered = sorted(
            (row for row in log if abs(row["new_value"] - row["old_value"]) >= 0.00005),
            key=lambda row: -row["eligibility"],
        )
        changed = {row["state"]: (row["old_value"], row["new_value"], row["eligibility"]) for row in filtered}

        def describe(filtered=filtered, kind=kind, info=info):
            label = learner.format_state(info["state"])
            old, delta = info["old_value"], info["delta"]
            decay = learner.gamma * self._lam(learner)
            formula_line = f"V(s) ← V(s) + α·δ·e(s)   (e decays ×γλ={decay:.3g} per step back)"

            if kind == "bootstrap":
                next_label = learner.format_state(info["next_state"])
                next_value = info["next_value"]
                delta_eq = (
                    f"δ = γ·V({next_label}) − V({label})\n"
                    f"δ = {learner.gamma:.3g}·{next_value:.4f} − {old:.4f} = {delta:.4f}"
                )
            else:  # "reward"
                reward = info["reward"]
                delta_eq = (
                    f"δ = reward − V({label})\n"
                    f"δ = {reward:.3g} − {old:.4f} = {delta:.4f}"
                )

            max_rows = 6
            rows = [
                f"  V({learner.format_state(row['state'])})  e={row['eligibility']:.3f}  "
                f"{row['old_value']:.4f} → {row['new_value']:.4f}"
                for row in filtered[:max_rows]
            ]
            if not rows:
                rows = ["  (no visible change)"]
            elif len(filtered) > max_rows:
                rows.append(f"  ... +{len(filtered) - max_rows} more (smaller eligibility)")

            return f"{formula_line}\n{delta_eq}\n" + "\n".join(rows)

        return UpdateEvent(changed=changed, describe=describe, color="red")


class ModelBasedRule(UpdateRule):
    """
    Model-based port-entry update (Krausz et al.): on reaching a reward
    port, sweeps a value update over every state on paths that have led
    into that port, weighted by a persistent, recency-averaged map T
    learned online. Reads `learner.gamma_mb` (default 0.9), `learner.a_T`
    (default 0.1), `learner.a_mb` (default 0.1) -- each falls back to its
    default if not passed to the learner's constructor.

        m(state) <- gamma_mb * m(state) for all states      [decay, every step]
        m(state_t) <- 1                                     [replacing trace]

        T(port, state) <- (1 - a_T) * T(port, state) + a_T * m(state)   [on port entry]

        V(state) <- V(state) + a_mb * T(port, state) * (R - V(state))   [on port entry]

    `m` (the current trial's memory trace) resets to empty every trial. `T`
    (indexed by *destination* port -- the trial's path[-1] -- not by
    `context`, which is keyed by *start* port) persists across trials.
    """

    DEFAULT_GAMMA_MB = 0.9
    DEFAULT_A_T = 0.1
    DEFAULT_A_MB = 0.1

    def __init__(self):
        self.T = {}   # {port: {state: weight}}, persists across trials
        self.m = {}   # {state: trace}, reset every trial

    def _gamma_mb(self, learner):
        return getattr(learner, "gamma_mb", self.DEFAULT_GAMMA_MB)

    def _a_t(self, learner):
        return getattr(learner, "a_T", self.DEFAULT_A_T)

    def _a_mb(self, learner):
        return getattr(learner, "a_mb", self.DEFAULT_A_MB)

    def _all_states(self, learner):
        """Every state in the learner's representation -- plain hexes, or
        every real (prev, cur) directed edge if directional=True. The T
        update runs over this whole set every port entry (see on_trial_end),
        not just states touched this trial, so states that stop being
        visited actually decay out of T instead of staying frozen."""
        if not learner.directional:
            return list(learner.graph.nodes())
        return [
            (prev, cur)
            for cur in learner.graph.nodes()
            for prev in learner.graph.neighbors(cur)
        ]

    def on_trial_start(self, learner, path, context):
        self.m = {}

    def on_graph_changed(self, learner):
        """Drop T/m entries for states no longer in the graph (e.g. a hex
        that's now a barrier, or -- if directional -- an edge that no
        longer exists). Without this, a barrier change leaves T holding
        stale weights for hexes that are gone, which just sit as dead
        (never read again) entries rather than causing wrong behavior --
        but they'd resurface with a stale weight if that barrier ever moves
        back, so prune them properly."""
        valid = set(self._all_states(learner))
        for port_map in self.T.values():
            for state in list(port_map):
                if state not in valid:
                    del port_map[state]
        for state in list(self.m):
            if state not in valid:
                del self.m[state]

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        decay = self._gamma_mb(learner)
        for tracked_state in list(self.m):
            self.m[tracked_state] *= decay
        self.m[next_state] = 1.0
        return []  # silent -- the visible sweep happens once, on_trial_end

    def on_trial_end(self, learner, path, reward, context):
        port = path[-1]
        # `port` (destination) is purely a lookup key into T -- the V update
        # below must still write into `context` (the trial's *start*-port
        # context), never into `port`, or goal-conditioned mode's separate
        # value tables get silently corrupted.
        T_port = self.T.setdefault(port, {})
        a_t = self._a_t(learner)

        # "for all states", per the paper -- including ones not on this
        # trial's path, whose trace (0.0) still pulls their T value down by
        # (1 - a_t) each time this port is reached without them.
        for state in self._all_states(learner):
            trace = self.m.get(state, 0.0)
            T_port[state] = (1 - a_t) * T_port.get(state, 0.0) + a_t * trace

        a_mb = self._a_mb(learner)
        changed = {}
        rows = []
        for state, weight in T_port.items():
            if weight == 0.0:
                continue
            old_value = learner.state_value(context, state)
            new_value = old_value + a_mb * weight * (reward - old_value)
            learner.V[context][state] = new_value
            if abs(new_value - old_value) >= 0.00005:
                changed[state] = (old_value, new_value, weight)
                rows.append((weight, state, old_value, new_value))

        rows.sort(key=lambda row: -row[0])

        def describe(port=port, reward=reward, rows=rows):
            header = (
                f"model-based update at port {port}  (reward={reward:.3g})\n"
                f"V(s) ← V(s) + a_mb·T({port},s)·(R − V(s))"
            )
            max_rows = 6
            lines = [
                f"  V({learner.format_state(state)})  T={weight:.3f}  {old:.4f} → {new:.4f}"
                for weight, state, old, new in rows[:max_rows]
            ]
            if not lines:
                lines = ["  (no visible change)"]
            elif len(rows) > max_rows:
                lines.append(f"  ... +{len(rows) - max_rows} more (smaller T)")
            return header + "\n" + "\n".join(lines)

        return [UpdateEvent(changed=changed, describe=describe, color="green")]
