"""
update_rules.py

The value-update mechanisms a hex maze agent can learn with. Each rule
below implements one theory of how experience changes the value of a maze
location, and a learner can run several at once, each writing into the
same value table V.

A rule reads its hyperparameters from the learner instance (`learner.alpha`,
`learner.a_mb`, etc), so every parameter of a fitted model lives in one
place and can be set in the learner's constructor or fit by fit_choices. A
rule stores on itself only the quantities it accumulates while learning: an
eligibility trace, a memory trace, a learned map.

"""

from .base_learner import UpdateRule, UpdateEvent


class TDLambdaRule(UpdateRule):
    """
    Temporal-difference learning with an eligibility trace: TD(lambda).

    Hyperparameters, read from the learner:
        learner.alpha   learning rate (default 0.3)
        learner.gamma   discount factor (default 0.95)
        learner.lam     eligibility trace decay (default 0.0)

    On each hex-to-hex transition, the value of the hex being left is pulled
    toward the discounted value of the hex being entered:

        td_error = gamma * V(next hex) - V(current hex)

    On the transition into a reward port, a second update treats the reward
    itself as the quantity being predicted, so the port's value tracks how
    often reward is delivered there:

        td_error = reward - V(port)

    Both updates are applied through the same eligibility trace. Each hex
    carries an eligibility that is set to 1 when visited and decays by
    gamma * lam on every subsequent step, and every hex is moved in
    proportion to its eligibility:

        V(hex) <- V(hex) + alpha * td_error * eligibility(hex)

    With lam = 0 only the hex just left is updated, and value propagates backward
    one hex per traversal. With lam = 1 a single reward is credited along the
    whole path within one trial. Values in between blend the two.
    """

    DEFAULT_ALPHA = 0.3
    DEFAULT_LAM = 0.0

    def _alpha(self, learner):
        """The learning rate, from the learner if it was given one."""
        return getattr(learner, "alpha", self.DEFAULT_ALPHA)

    def _lam(self, learner):
        """The eligibility trace decay, from the learner if it was given one."""
        return getattr(learner, "lam", self.DEFAULT_LAM)

    def on_trial_start(self, learner, path, context):
        """Start the trial with an empty eligibility trace, so credit for this
        trial's reward reaches only hexes visited during this trial."""
        self._eligibility = {}

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        events = []

        # The hex being left is pulled toward the discounted value of the hex
        # being entered:  td_error = gamma * V(next hex) - V(current hex)
        old_value = learner.state_value(context, state)
        next_value = learner.state_value(context, next_state)
        td_error = learner.gamma * next_value - old_value
        updated_states = []
        self._apply_td_error(learner, context, state, td_error, updated_states)
        events.append(self._make_event(
            learner, "bootstrap", state=state, next_state=next_state,
            old_value=old_value, next_value=next_value, delta=td_error, log=updated_states,
        ))

        # Arriving at a reward port is a second learning event: the port hex's
        # value is pulled toward the reward delivered there.
        #     td_error = reward - V(port)
        if is_terminal:
            old_port_value = learner.state_value(context, next_state)
            reward_td_error = reward - old_port_value
            states_updated_by_reward = []
            self._apply_td_error(learner, context, next_state, reward_td_error, states_updated_by_reward)
            events.append(self._make_event(
                learner, "reward", state=next_state, reward=reward,
                old_value=old_port_value, delta=reward_td_error, log=states_updated_by_reward,
            ))

        return events

    def _apply_td_error(self, learner, context, state, delta, log):
        """Bump `state`'s trace, update every traced state, then decay all traces. 
        With lambda > 0 one delta can update many states at once,
        each scaled by its own eligibility.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            context (int or None): Which value table to write into, meaning the
                port this trial started from, or None when the learner keeps a
                single shared table
            state (int or tuple): The state this prediction error belongs to,
                whose eligibility is bumped before the update
            delta (float): The prediction error to distribute across traced states
            log (list): Collects one dict per state updated, for display

        Returns:
            None. `learner.V` and `log` are updated in place
        """
        # Visiting a state makes it eligible for credit. Adding to the existing
        # eligibility (rather than overwriting) means a state visited twice in
        # one trial carries more credit than a state visited once.
        self._eligibility[state] = self._eligibility.get(state, 0.0) + 1.0

        # How much eligibility survives one step backward in space. gamma
        # discounts by space and lam sets how far credit reaches, so at lam = 0
        # this is 0 and only the state just visited is updated.
        decay = learner.gamma * self._lam(learner)

        # Walking a copy of the keys, since states are dropped from the trace below
        for traced_state in list(self._eligibility):
            # How eligible this state is for the current prediction error
            e = self._eligibility[traced_state]

            # Every eligible state moves a share of the prediction error,
            # scaled by its own eligibility:
            #     V(state) <- V(state) + alpha * delta * e(state)
            old_value = learner.state_value(context, traced_state)
            new_value = old_value + self._alpha(learner) * delta * e
            learner.V[context][traced_state] = new_value

            # Record what moved, so the update can be displayed afterward
            log.append({
                "state": traced_state, "eligibility": e,
                "old_value": old_value, "new_value": new_value,
            })

            # Decay this state's eligibility by one step, so the further back it
            # was visited, the smaller its share of the next prediction error
            self._eligibility[traced_state] *= decay

            # Once a state's eligibility is this small it can no longer change a
            # value by a visible amount, so it is dropped to keep the trace short
            if self._eligibility[traced_state] < 1e-6:
                del self._eligibility[traced_state]

    def _make_event(self, learner, kind, **info):
        """
        Package one prediction error and the states it moved into an
        UpdateEvent, so it can be outlined on the maze and written out as text.

        Parameters:
            learner (BaseHexLearner): The learner the update was applied to
            kind (str): "bootstrap" for a hex-to-hex transition, or "reward" for
                the update at a reward port, which changes the equation shown
            **info: The quantities this update was computed from (state,
                old_value, delta, log, and for each kind the next state and its
                value, or the reward delivered)

        Returns:
            UpdateEvent: The states that moved and how to describe the update
        """
        # Keep only states whose value moved by at least 0.00005, which is the
        # smallest change still visible once values are rounded for display
        log = info["log"]
        filtered = sorted(
            (row for row in log if abs(row["new_value"] - row["old_value"]) >= 0.00005),
            key=lambda row: -row["eligibility"],   # most eligible states first
        )

        # Eligibility is carried alongside each change so the animation can
        # shade a hex's outline by how much credit it received
        changed = {row["state"]: (row["old_value"], row["new_value"], row["eligibility"]) for row in filtered}

        # Built as a function so the text is only assembled if it gets displayed.
        # The quantities are captured as defaults so each event keeps its own.
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

            # One line per state that moved, with the deepest traces truncated
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
        """Clear the learned port-to-state map and the current trial's trace."""
        self.T = {}
        self.m = {}

    def _gamma_mb(self, learner):
        """How much of the memory trace survives each step, from the learner."""
        return getattr(learner, "gamma_mb", self.DEFAULT_GAMMA_MB)

    def _a_t(self, learner):
        """The learning rate for the port-to-state map T, from the learner."""
        return getattr(learner, "a_T", self.DEFAULT_A_T)

    def _a_mb(self, learner):
        """The learning rate for the value sweep at a port, from the learner."""
        return getattr(learner, "a_mb", self.DEFAULT_A_MB)

    def _all_states(self, learner):
        """
        Every state in the learner's representation: each hex, or each
        directed edge (prev, cur) for a directional learner. The map is
        averaged over this whole set on every port entry (see on_trial_end),
        so a state the rat has stopped visiting decays out of the map as its
        memory trace stays at zero.
        """
        if not learner.directional_states:
            return list(learner.graph.nodes())
        return [
            (prev, cur)
            for cur in learner.graph.nodes()
            for prev in learner.graph.neighbors(cur)
        ]

    def on_trial_start(self, learner, path, context):
        """Start the trial with an empty memory trace, so the map records only
        where the rat went on its way to this trial's port."""
        self.m = {}

    def on_maze_changed(self, learner):
        """Drop T/m entries for states no longer in the graph 
        (e.g. a hex that's now a barrier, or if directional, 
        an edge that no longer exists)."""
        # The states the new maze allows. Anything learned about a state the
        # maze no longer has is dropped, so if that hex reopens later it starts
        # from scratch.
        valid = set(self._all_states(learner))
        for port_map in self.T.values():
            for state in list(port_map):
                if state not in valid:
                    del port_map[state]
        for state in list(self.m):
            if state not in valid:
                del self.m[state]

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        """
        Keep the memory trace of where the rat has been this trial.

        Parameters:
            learner (BaseHexLearner): The learner being updated
            context (int or None): The value table for this trial
            state (int or tuple): The state being left
            next_state (int or tuple): The state being entered
            reward (float): The reward delivered at the end of this trial
            is_terminal (bool): True on the transition into the reward port

        Returns:
            list: Empty, since this rule changes no values until the port
        """
        # Every state already in the trace fades, so a state decays proportionally to 
        # how many hexes ago it was visited:
        #     m(state) <- gamma_mb * m(state)
        decay = self._gamma_mb(learner)
        for tracked_state in list(self.m):
            self.m[tracked_state] *= decay

        # The state just entered is set to full strength, replacing whatever it
        # had faded to if the rat has been here before this trial:
        #     m(state just entered) <- 1
        self.m[next_state] = 1.0

        # The value update for this trial happens once, in on_trial_end.
        return []

    def on_trial_end(self, learner, path, reward, context):
        """
        Average this trial's memory trace into the port's map, then do a
        value update for every state the map links to that port.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            path (list[int]): The hexes the rat ran this trial, ending at a port
            reward (float): The reward delivered at that port
            context (int or None): The value table for this trial, meaning the
                port the trial started from

        Returns:
            list[UpdateEvent]: One event holding every state whose value moved
        """
        # The port the rat arrived at selects which learned map to update.
        # Value is written into `context`, the value table belonging to the
        # port the trial started from, which is where this trial's experience
        # belongs when the learner is goal-conditioned.
        port = path[-1]
        T_for_port = self.T.setdefault(port, {})
        a_T = self._a_t(learner)

        # Every state in the maze is averaged into the map, including states
        # the rat did not visit this trial. Their memory trace is 0, so each
        # arrival at this port without visiting them decays their weight by
        # (1 - a_T), and a state that stops being visited fades out of the map.
        #     T(port, state) <- (1 - a_T) * T(port, state) + a_T * m(state)
        for state in self._all_states(learner):
            memory_trace = self.m.get(state, 0.0)
            T_for_port[state] = (1 - a_T) * T_for_port.get(state, 0.0) + a_T * memory_trace

        # Every state that recently lead to this port is moved toward the reward
        # just received, in proportion to how recently (in hex steps) it lead to the port:
        #     V(state) <- V(state) + a_mb * T(port, state) * (reward - V(state))
        a_mb = self._a_mb(learner)
        changed = {}
        rows = []
        for state, T_for_state in T_for_port.items():
            if T_for_state == 0.0:
                continue
            old_value = learner.state_value(context, state)
            new_value = old_value + a_mb * T_for_state * (reward - old_value)
            learner.V[context][state] = new_value
            if abs(new_value - old_value) >= 0.00005:
                changed[state] = (old_value, new_value, T_for_state)
                rows.append((T_for_state, state, old_value, new_value))

        # Most strongly linked states first, so the text shows the states this
        # port's map has learned most about
        rows.sort(key=lambda row: -row[0])

        # Built as a function so the text is only assembled if it gets displayed
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

    The hexes replayed on each trial are given to the learner alongside that
    trial's trajectory and reward (see BaseHexLearner.learn), and read here as
    `learner.replayed_hexes_this_trial`.

    On arrival at a reward port, the hexes replayed during that trial's
    ripples are each moved toward the reward that was delivered:

        V(hex) <- V(hex) + a_swr * (reward - V(hex))

    A hex is updated when it was replayed. The current implementation has all
    hexes be updated the same amount (regardless of how long they were represented
    in the ripple). Hexes absent from the replay keep their current value, 
    so the replay content alone decides what is learned.
    
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
        """The learning rate for replayed hexes, from the learner."""
        return getattr(learner, "a_swr", self.DEFAULT_A_SWR)

    def _states_for_hex(self, learner, hex):
        """
        The states a replayed hex corresponds to. For a learner holding one
        value per hex, that is the hex itself. For a directional learner,
        whose states are directed edges, it is every edge arriving at that
        hex, since a replayed hex identifies a location and the direction of
        travel through it is left open.
        
        TODO: For directional learners, maybe we only want to update the edges
        that reflect movement "inbound" to the goal port? (the current implementation
        does this value update for all edges, which maybe we don't want. It could
        be useful to have both to compare the two.)
        """
        # One value per hex: the replayed hex is the state, and a hex that is a
        # barrier in this maze has no state to update
        if not learner.directional_states:
            return [hex] if hex in learner.graph else []
        if hex not in learner.graph:
            return []

        # One value per direction of travel: every way of arriving at this hex
        return [(prev, hex) for prev in learner.graph.neighbors(hex)]

    def on_step(self, learner, context, state, next_state, reward, is_terminal):
        # Replay is learned from once per trial, in on_trial_end.
        return []

    def on_trial_end(self, learner, path, reward, context):
        """
        Move every hex replayed on this trial toward the reward delivered.

        Parameters:
            learner (BaseHexLearner): The learner whose value table is updated
            path (list[int]): The hexes the rat ran this trial, ending at a port
            reward (float): The reward delivered at that port
            context (int or None): The value table for this trial, meaning the
                port the trial started from

        Returns:
            list[UpdateEvent]: One event holding every state whose value moved
        """
        a_swr = self._a_swr(learner)
        port = path[-1]

        # A hex replayed several times in one ripple is updated only once.
        states = []
        for hex in learner.replayed_hexes_this_trial:
            for state in self._states_for_hex(learner, hex):
                if state not in states:
                    states.append(state)

        # Each replayed hex moves a fraction a_swr of the way to the reward:
        #     V(state) <- V(state) + a_swr * (reward - V(state))
        changed = {}
        rows = []
        for state in states:
            old_value = learner.state_value(context, state)
            new_value = old_value + a_swr * (reward - old_value)
            learner.V[context][state] = new_value
            # Every replayed hex is included equally, so each carries a weight
            # of 1 where other rules carry a trace or a learned strength
            if abs(new_value - old_value) >= 0.00005:
                changed[state] = (old_value, new_value, 1.0)
                rows.append((state, old_value, new_value))

        # Built as a function so the text is only assembled if it gets displayed
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

        return [UpdateEvent(changed=changed, describe=describe, color="purple")]

