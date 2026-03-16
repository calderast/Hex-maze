# hexmaze.rl

Reinforcement learning agents for the hex maze.

## Modules

### `td_learner.py` — `HexMazeTDLearner`

TD (temporal-difference) value learner that maintains 3 V-tables (one per starting port).

- Learns **V(hex)** — the value of being at a given hex
- Supports both **TD(0)** (forward, one-step) and **TD(1)** (backward, full-path) updates
- On-policy: updates use the value of the hex the agent actually moved to

**TD(0) update** (at each step along the trajectory):

```text
V(hex) ← V(hex) + α₀ · [r + γ · V(next_hex) - V(hex)]
```

**TD(1) backward pass** (after the full trajectory, for each hex at time t):

```text
V(hex_t) ← V(hex_t) + α₁ · [γ^(T-t) · R - V(hex_t)]
```

where T is the final step and R (0 or 1) is the reward.

### `q_learner.py` — `HexMazeQLearner`

Q-learning agent that maintains 3 Q-tables (one per starting port).

- Learns **Q(hex, action)** — the value of moving to a specific neighbor from a given hex
- Captures directional preferences (e.g. "from hex 25, moving toward hex 26 is better than hex 24")
- Off-policy: updates use `max Q(next_hex, a')` regardless of the action actually taken

**Q-learning update** (at each step along the trajectory):

```text
Q(hex, a) ← Q(hex, a) + α · [r + γ · max_a' Q(next_hex, a') - Q(hex, a)]
```

At terminal hexes (reward ports), `max Q(next_hex, a') = 0`.

## Common interface

Both learners share the same interface:

| Method | Description |
|---|---|
| `learn(trajectories, rewards, start_ports)` | Run updates on given provided trajectories (rat hex paths) |
| `simulate(start_state, n_trials, max_steps)` | Self-generated trajectories with online updates |
| `process_trajectory(path, reward, start_port)` | Update on a single trajectory |
| `process_trajectory_with_history(...)` | Same as above, but returns hex value snapshots at each step |
| `action_probabilities(hex, start_port)` | Softmax choice probabilities at a hex |
| `get_state_values(start_port)` | Per-hex values under one start port |
| `get_max_state_values()` | Max value across all 3 tables per hex |
| `reset()` | Re-initialize tables and re-apply hex value priors |
| `set_graph(new_graph)` | Swap the maze graph (e.g. after barrier changes) |

### Shared parameters

- **`reward_probs`**: `[p1, p2, p3]` — reward probability at each port
- **`gamma`**: discount factor
- **`temperature`**: softmax temperature for action selection
- **`priors`**: V/Q-table initialization — `None`, `"uniform"`, `("flat", value)`, or `[p1, p2, p3]`
- **`no_backtrack`**: if `True`, agent avoids revisiting states within a trial, when possible (useful for simulation sometimes)

### Table update logic

Both learners use the same rule for deciding which of the 3 tables to update at a given hex:

- If the hex is in the same third as the start port (or is a critical choice point): update only the start port's table
- If the hex is in a different third T: update all tables except T's

TODO: decide if I like this. Figure out what to do with mazes that have more than 1 critical choice point!

### `start_port` defaults

In both learners, `start_port` is optional for `process_trajectory` and `process_trajectory_with_history`. It defaults to `path[0]` if that state is a reward port; otherwise an error is raised.

## Usage

```python
from hexmaze.rl import HexMazeTDLearner, HexMazeQLearner

# Both accept the same constructor pattern
td = HexMazeTDLearner(graph, reward_probs=[0.9, 0.5, 0.1])
ql = HexMazeQLearner(graph, reward_probs=[0.9, 0.5, 0.1])

# Learn from rat trajectories
td.learn(trajectories, rewards)
ql.learn(trajectories, rewards)

# Or simulate
td_results = td.simulate(start_state=1, n_trials=100)
ql_results = ql.simulate(start_state=1, n_trials=100)

# Compare values
td_values = td.get_state_values(start_port=1)
ql_values = ql.get_state_values(start_port=1)  # max Q per hex
ql_q_values = ql.get_q_values(start_port=1)    # full Q(s, a) table
```
