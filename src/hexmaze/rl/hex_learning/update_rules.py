"""
update_rules.py

The value update rules a hex maze agent can learn with. Each rule is one
theory of how the rat's experience changes the value of a maze location.
A learner can run several rules at once, all writing to the same value table V.

Each rule reads its hyperparameters from the learner (`learner.alpha`,
`learner.a_mb`, etc), so all of a model's parameters live in one place and can
be set in the learner's constructor or fit by fit_choices. A rule only stores
the things it accumulates while learning (e.g. an eligibility trace, a memory
trace, or a learned map).

UpdateRule (the base class that defines when a rule gets called) and
UpdateEvent (what a rule reports back about each update) come first.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class UpdateEvent:
    """
    One value update, recorded so it can be displayed (used for animation).

    Attributes:
        changed: {state: (old_value, new_value, weight)} for the states whose
            value changed by a visible amount. `weight` is how strongly the rule
            included that state in this update (which is the eligibility trace
            for TD learning and T(port, state) for a model-based update).
            animate_learning uses it to shade each hex's outline, so we can see
            a trace decaying back along the path.
        describe: A function (no arguments) that returns this update's text
            block: its equation and the states it changed.
        color: Name of the outline color ramp for the changed hexes (see
            _COLOR_TO_CMAP in base_learner.py), so we can tell different rules'
            updates apart in the animation.
    """
    changed: dict
    describe: Callable[[], str]
    color: str


class UpdateRule:
    """
    Base interface for a pluggable value-update mechanism (letting us simulate/test
    different theories of how hex values should be updated based on the rat's experience).

    The rules implemented below:
    - TD(lambda): Classic temporal-difference learning with an eligibility trace.
        Hex values are updated on each hex transition.
    - Model-based memory trace (from Krausz et al.): on reaching a reward port,
        updates the values of every state on paths that have led into that port,
        weighted by a persistent, recency-averaged map T learned online.
    - SWR update: on reaching a reward port, updates the values of hexes represented
        in sharp-wave-ripples

    A learner holds a list of these and calls each one at every moment below,
    so combining value update mechanisms (e.g. TD learning plus replay) is
    as easy as passing in two rules. Every hook here does nothing by default,
    so a rule only needs to implement the ones it cares about.
    """

    def on_trial_start(self, learner, context):
        """
        Called before the rat's first move of a trial (e.g. to reset a per-trial
        eligibility/memory trace).
        No return value.
        """

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        """
        Called once per hex-to-hex transition, right after the rat makes it, so
        the rat's next choice is made from the updated values.
        `is_terminal` is True for the transition into the reward port, where
        `reward` is what was delivered (1 or 0 for reward/omission). On every
        other transition `reward` is 0.
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
        Called when learner.set_graph() swaps in a new maze (e.g. on a barrier
        change mid-session). A rule that keeps its own hex-keyed values should
        drop the entries for hexes that are now barriers, so if that hex opens
        up again later it starts fresh. V has already been cleaned up by the
        time this is called.
        """

    def on_reset(self, learner):
        """
        Called when the learner's value tables are cleared, including once
        when the learner is built. A rule that accumulates something over a
        session (like a memory trace) should clear it here, so a reset learner
        starts the session over from the beginning.
        """


class TDLambdaRule(UpdateRule):
    """
    Temporal-difference learning with an eligibility trace: TD(lambda).

    Hyperparameters, read from the learner:
        learner.alpha   learning rate (default 0.3)
        learner.gamma   discount factor (default 0.95)
        learner.lam     eligibility trace decay (default 0.0)

    On each hex-to-hex transition, the value of the hex the rat is leaving
    moves toward the discounted value of the hex it is entering:

        td_error = gamma * V(next hex) - V(current hex)

    On the transition into a reward port, there is a second update toward
    the reward itself, so the port's value tracks how often reward is
    delivered there:

        td_error = reward - V(port)

    Both updates use the same eligibility trace. Each hex's eligibility goes
    up by 1 every time it is visited and decays by gamma * lam on every step
    after that. Every hex is updated in proportion to its eligibility:

        V(hex) <- V(hex) + alpha * td_error * eligibility(hex)

    With lam = 0 only the hex just left is updated, so value propagates backward
    one hex per traversal. With lam = 1 a single reward is credited along the
    whole path within one trial. Values in between blend the two.
    """

    DEFAULT_ALPHA = 0.3
    DEFAULT_LAM = 0.0

    def _alpha(self, learner):
        """Get the learning rate from the learner (or the default)."""
        return getattr(learner, "alpha", self.DEFAULT_ALPHA)

    def _lam(self, learner):
        """Get the eligibility trace decay from the learner (or the default)."""
        return getattr(learner, "lam", self.DEFAULT_LAM)

    def on_trial_start(self, learner, context):
        """Reset the eligibility trace at the start of each trial, so this
        trial's reward is only credited to hexes visited during this trial."""
        self._eligibility = {}

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        events = []

        # Move the value of the hex being left toward the discounted value of
        # the hex being entered:  td_error = gamma * V(next hex) - V(current hex)
        old_value = learner.V[context][state]
        next_value = learner.V[context][next_state]
        td_error = learner.gamma * next_value - old_value
        updated_states = []
        self._apply_td_error(learner, context, state, td_error, updated_states)
        events.append(self._make_event(
            learner, "bootstrap", state=state, next_state=next_state,
            old_value=old_value, next_value=next_value, delta=td_error, log=updated_states,
        ))

        # When the rat reaches a reward port, also move the port's value
        # toward the reward delivered there:
        #     td_error = reward - V(port)
        if is_terminal:
            old_port_value = learner.V[context][next_state]
            reward_td_error = reward - old_port_value
            states_updated_by_reward = []
            self._apply_td_error(learner, context, next_state, reward_td_error, states_updated_by_reward)
            events.append(self._make_event(
                learner, "reward", state=next_state, reward=reward,
                old_value=old_port_value, delta=reward_td_error, log=states_updated_by_reward,
            ))

        return events

    def _apply_td_error(self, learner, context, state, td_error, log):
        """Bump `state`'s trace, update every traced state, then decay all traces.
        With lambda > 0, one prediction error can update many states at once,
        each scaled by its own eligibility.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            context (int or None): Which value table to update (the port this
                trial started from, or None if the learner has a single
                shared table)
            state (int or tuple): The state this prediction error belongs to
                (its eligibility is bumped before the update)
            td_error (float): The prediction error to distribute across traced states
            log (list): Collects one dict per state updated, for display

        Returns:
            None. `learner.V` and `log` are updated in place
        """
        # Each visit to a state adds 1 to its eligibility, so a state visited
        # twice in one trial gets more credit than a state visited once
        self._eligibility[state] = self._eligibility.get(state, 0.0) + 1.0

        # How much eligibility is left after each step back along the path.
        # gamma discounts by distance and lam sets how far back credit reaches.
        # When lam = 0, this is 0 so only the state just visited is updated.
        decay = learner.gamma * self._lam(learner)

        # Loop over a copy of the keys, since we drop states from the trace below
        for traced_state in list(self._eligibility):
            # How eligible this state is for the current prediction error
            e = self._eligibility[traced_state]

            # Update each eligible state in proportion to its eligibility:
            #     V(state) <- V(state) + alpha * td_error * e(state)
            old_value = learner.V[context][traced_state]
            new_value = old_value + self._alpha(learner) * td_error * e
            learner.V[context][traced_state] = new_value

            # Record what changed so we can display the update later
            log.append({
                "state": traced_state, "eligibility": e,
                "old_value": old_value, "new_value": new_value,
            })

            # Decay this state's eligibility by one step, so the longer ago it was
            # visited, the less credit it gets from the next prediction error
            self._eligibility[traced_state] *= decay

            # Drop states whose eligibility is too small to visibly change a
            # value, to keep the trace short
            if self._eligibility[traced_state] < 1e-6:
                del self._eligibility[traced_state]

    def _make_event(self, learner, kind, **info):
        """
        Package one prediction error and the states it changed into an
        UpdateEvent, so we can outline them on the maze and show the update as text.

        Parameters:
            learner (BaseHexLearner): The learner the update was applied to
            kind (str): "bootstrap" for a hex-to-hex transition, or "reward" for
                the update at a reward port (this changes which equation is shown)
            **info: The values this update was computed from (state, old_value,
                delta, log, and either next_state and next_value for "bootstrap",
                or reward for "reward")

        Returns:
            UpdateEvent: The states that changed and how to describe the update
        """
        # Keep only states whose value changed by at least 0.00005, which is the
        # smallest change still visible once values are rounded for display
        log = info["log"]
        filtered = sorted(
            (row for row in log if abs(row["new_value"] - row["old_value"]) >= 0.00005),
            key=lambda row: -row["eligibility"],   # most eligible states first
        )

        # Keep each state's eligibility with its change, so the animation can
        # shade a hex's outline by how much credit it got
        changed = {row["state"]: (row["old_value"], row["new_value"], row["eligibility"]) for row in filtered}

        # This is a function so the text is only built if it actually gets displayed.
        # Values are captured as default arguments so each event keeps its own.
        def describe(filtered=filtered, kind=kind, info=info):
            label = learner.format_state(info["state"])
            old, delta = info["old_value"], info["delta"]
            decay = learner.gamma * self._lam(learner)

            # The update rule itself, shown once at the top of the text block
            formula_line = f"V(s) ← V(s) + α·δ·e(s)   (e decays ×γλ={decay:.3g} per step back)"

            # Where the prediction error came from: the value of the hex just
            # entered, or the reward delivered at the port
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

            # One line per state that changed (cut off after max_rows)
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
    learned online.
    Reads params:
    `learner.gamma_mb` (default 0.9)
    `learner.a_T` (default 0.1)
    `learner.a_mb` (default 0.1)

        m(state) <- gamma_mb * m(state) for all states      [decay, every step]
        m(state_t) <- 1                                     [replacing trace]

        T(port, state) <- (1 - a_T) * T(port, state) + a_T * m(state)   [on port entry]

        V(state) <- V(state) + a_mb * T(port, state) * (R - V(state))   [on port entry]

    `m` (the current trial's memory trace) resets to empty every trial.
    `T` (indexed by destination port, aka the trial's path[-1]) persists across trials.
    """

    DEFAULT_GAMMA_MB = 0.9
    DEFAULT_A_T = 0.1
    DEFAULT_A_MB = 0.1

    def __init__(self):
        self.T = {}   # {port: {state: T(port, state)}}, persists across trials
        self.m = {}   # {state: trace}, reset every trial

    def on_reset(self, learner):
        """Clear the learned port-to-state map T and the current trial's trace m."""
        self.T = {}
        self.m = {}

    def _gamma_mb(self, learner):
        """Get the memory trace decay per step from the learner (or the default)."""
        return getattr(learner, "gamma_mb", self.DEFAULT_GAMMA_MB)

    def _a_t(self, learner):
        """Get the learning rate for the map T from the learner (or the default)."""
        return getattr(learner, "a_T", self.DEFAULT_A_T)

    def _a_mb(self, learner):
        """Get the model-based value learning rate from the learner (or the default)."""
        return getattr(learner, "a_mb", self.DEFAULT_A_MB)

    def on_trial_start(self, learner, context):
        """Reset the memory trace at the start of each trial, so T only learns
        where the rat went on its way to this trial's port."""
        self.m = {}

    def on_maze_changed(self, learner):
        """Drop T/m entries for states no longer in the graph
        (e.g. a hex that's now a barrier, or if directional,
        an edge that no longer exists)."""
        # Get the states that exist in the new maze. Drop anything learned about
        # states that don't, so if that hex opens up again later it starts fresh.
        valid = set(learner.all_states())
        for port_map in self.T.values():
            for state in list(port_map):
                if state not in valid:
                    del port_map[state]
        for state in list(self.m):
            if state not in valid:
                del self.m[state]

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        """
        Update the memory trace of where the rat has been this trial.

        Parameters:
            learner (BaseHexLearner): The learner being updated
            context (int or None): The value table for this trial
            state (int or tuple): The state being left
            next_state (int or tuple): The state being entered
            reward (float): The reward delivered, on the transition into the
                reward port (0 on every other transition)
            is_terminal (bool): True on the transition into the reward port

        Returns:
            list: Empty, since this rule doesn't change any values until the port
        """
        # Decay every state already in the trace, so each state's trace
        # depends on how many hexes ago it was visited:
        #     m(state) <- gamma_mb * m(state)
        decay = self._gamma_mb(learner)
        for tracked_state in list(self.m):
            self.m[tracked_state] *= decay

        # Set the state just entered to full strength (replacing its decayed
        # value if the rat has already been here this trial):
        #     m(state just entered) <- 1
        self.m[next_state] = 1.0

        # Values are only updated once per trial, in on_trial_end
        return []

    def on_trial_end(self, learner, path, reward, context):
        """
        Average this trial's memory trace into the port's map T, then update
        the value of every state T links to that port.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            path (list[int]): The hexes the rat ran this trial, ending at a port
            reward (float): The reward delivered at that port
            context (int or None): The value table for this trial (the port the
                trial started from)

        Returns:
            list[UpdateEvent]: One event with every state whose value changed
        """
        # The port the rat reached decides which map T to update.
        # Values are written to `context`, the value table for the port the
        # trial started from (for a goal-conditioned learner).
        port = path[-1]
        T_for_port = self.T.setdefault(port, {})
        a_T = self._a_t(learner)

        # Average every state in the maze into T, including states the rat
        # didn't visit this trial. Their memory trace is 0, so each time the rat
        # reaches this port without visiting them, their weight decays by
        # (1 - a_T), and a state that stops being visited fades out of T.
        # (The state a trial starts in, (None, hex), is never entered on a step,
        # so it stays at 0 and is never updated.)
        #     T(port, state) <- (1 - a_T) * T(port, state) + a_T * m(state)
        for state in learner.all_states():
            memory_trace = self.m.get(state, 0.0)
            T_for_port[state] = (1 - a_T) * T_for_port.get(state, 0.0) + a_T * memory_trace

        # Move every state that has recently led to this port toward the reward
        # just received, in proportion to how strongly T links it to the port:
        #     V(state) <- V(state) + a_mb * T(port, state) * (reward - V(state))
        a_mb = self._a_mb(learner)
        changed = {}
        rows = []
        for state, T_for_state in T_for_port.items():
            if T_for_state == 0.0:
                continue
            old_value = learner.V[context][state]
            new_value = old_value + a_mb * T_for_state * (reward - old_value)
            learner.V[context][state] = new_value
            if abs(new_value - old_value) >= 0.00005:
                changed[state] = (old_value, new_value, T_for_state)
                rows.append((T_for_state, state, old_value, new_value))

        # Sort by T (highest first), so the text shows the states most strongly
        # linked to this port
        rows.sort(key=lambda row: -row[0])

        # This is a function so the text is only built if it actually gets displayed
        def describe(port=port, reward=reward, rows=rows):
            header = (
                f"model-based update at port {port}  (reward={reward:.3g})\n"
                f"V(s) ← V(s) + a_mb·T({port},s)·(R − V(s))"
            )
            max_rows = 6
            lines = [
                f"  V({learner.format_state(state)})  T={T_for_state:.3f}  {old:.4f} → {new:.4f}"
                for T_for_state, state, old, new in rows[:max_rows]
            ]
            if not lines:
                lines = ["  (no visible change)"]
            elif len(rows) > max_rows:
                lines.append(f"  ... +{len(rows) - max_rows} more (smaller T)")
            return header + "\n" + "\n".join(lines)

        return [UpdateEvent(changed=changed, describe=describe, color="green")]


class SWRRule(UpdateRule):
    """
    Value learning from sharp-wave-ripple replay.

    Hyperparameters, read from the learner:
        learner.a_swr   learning rate for replayed hexes (default 0.1)

    The hexes replayed on each trial are passed to the learner along with that
    trial's trajectory and reward (see BaseHexLearner.learn), and read here as
    `learner.replayed_hexes_this_trial`.

    When the rat reaches a reward port, each hex replayed during that trial's
    ripples moves toward the reward that was delivered:

        V(hex) <- V(hex) + a_swr * (reward - V(hex))

    A hex is updated when it was replayed. The current implementation has all
    hexes be updated the same amount (regardless of how long they were represented
    in the ripple). Hexes absent from the replay keep their current value,
    so the replay content alone decides what is learned.

    For a directional learner, a replayed hex only updates the directions of travel
    that head toward the port the trial ended at (see _states_for_hex).

    TODO:
    - Yang-Sun / Tim's DA ramp data seems to imply that SWRs bump up hex values
        both on reward and omission (DA ramps decay on omission, but decay less with SWR).
        This current implementation moves hexes towards the reward value (so moves
        hex values DOWN on omission). We probably want options for both
    - Yang-Sun / Tim's DA ramp data also seems to imply a value decay when not
        bumped up or maintained by SWRs. Currently values do not decay. We might want
        to add that in the future.
    """

    DEFAULT_A_SWR = 0.1

    def _a_swr(self, learner):
        """Get the learning rate for replayed hexes from the learner (or the default)."""
        return getattr(learner, "a_swr", self.DEFAULT_A_SWR)

    def _states_for_hex(self, learner, hex, port):
        """
        Get the state(s) to update for a replayed hex.

        For a non-directional learner (one value per hex), this is just the hex.
        For a directional learner, the replay tells us which hex but not which
        direction the rat was traveling through it. We assume the replay is of
        a route toward the port the rat just reached, so we update the directions
        that head toward that port: arriving at this hex from a neighbor that is
        one step further from the port. (A hex with two neighbors equally far
        from the port has two of these.)

        Parameters:
            learner (BaseHexLearner): The learner whose states these are
            hex (int): The replayed hex
            port (int): The port this trial ended at

        Returns:
            list: The state keys to update for this replayed hex
        """
        # Non-directional: the hex is the state (a barrier hex has no state)
        if not learner.directional_states:
            return [hex] if hex in learner.graph else []
        if hex not in learner.graph:
            return []

        # Directional: arriving here from a hex further from the port
        # means heading toward the port
        distances = learner.hex_distances()
        return [
            (prev, hex)
            for prev in learner.graph.neighbors(hex)
            if distances[port][prev] == distances[port][hex] + 1
        ]

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        # Replay updates only happen once per trial, in on_trial_end
        return []

    def on_trial_end(self, learner, path, reward, context):
        """
        Move every hex replayed on this trial toward the reward delivered.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            path (list[int]): The hexes the rat ran this trial, ending at a port
            reward (float): The reward delivered at that port
            context (int or None): The value table for this trial (the port the
                trial started from)

        Returns:
            list[UpdateEvent]: One event with every state whose value changed
        """
        a_swr = self._a_swr(learner)
        port = path[-1]

        # A hex replayed several times is only updated once (dict.fromkeys keeps
        # the first appearance of each state, in replay order)
        states = list(dict.fromkeys(
            state
            for hex in learner.replayed_hexes_this_trial
            for state in self._states_for_hex(learner, hex, port)
        ))

        # Move each replayed hex a fraction a_swr of the way to the reward:
        #     V(state) <- V(state) + a_swr * (reward - V(state))
        changed = {}
        rows = []
        for state in states:
            old_value = learner.V[context][state]
            new_value = old_value + a_swr * (reward - old_value)
            learner.V[context][state] = new_value
            # Every replayed hex is updated by the same fraction, so each one
            # is outlined at full weight in the animation
            if abs(new_value - old_value) >= 0.00005:
                changed[state] = (old_value, new_value, 1.0)
                rows.append((state, old_value, new_value))

        # This is a function so the text is only built if it actually gets displayed
        def describe(port=port, reward=reward, rows=rows, n=len(states)):
            header = (
                f"SWR replay update at port {port}  (reward={reward:.3g}, {n} hexes replayed)\n"
                f"V(s) ← V(s) + a_swr·(R − V(s))"
            )
            max_rows = 6
            lines = [
                f"  V({learner.format_state(state)})  {old:.4f} → {new:.4f}"
                for state, old, new in rows[:max_rows]
            ]
            if not lines:
                lines = ["  (no visible change)"]
            elif len(rows) > max_rows:
                lines.append(f"  ... +{len(rows) - max_rows} more")
            return header + "\n" + "\n".join(lines)

        return [UpdateEvent(changed=changed, describe=describe, color="lightgreen")]

