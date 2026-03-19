# hexmaze.rl

Reinforcement learning agents for the hex maze.

** NOTE hex learning agents are still in progress **

```text
hexmaze/rl/
├── hex_learning/        # trajectory-based, learns values over hexes
│   ├── td_learner.py    # HexMazeTDLearner
│   └── q_learner.py     # HexMazeQLearner
└── port_learning/       # outcome-based, learns values over ports
    ├── rescorla_wagner.py       # RescorlaWagner
    ├── bayesian.py              # BayesianPortLearner
    ├── hidden_state.py          # HiddenStatePortLearner
    └── bayesian_hidden_state.py # BayesianHiddenStatePortLearner (in progress)
```

## Port value learning

Learns values over reward ports (1/2/3 or A/B/C) from binary reward outcomes (0 or 1).

See the full port learning [README](port_learning/README.md) for detailed docs on all models,
methods, and code examples.

| Model | Key idea | Free parameters |
| --- | --- | --- |
| **RescorlaWagner** | Q ← Q + α·(reward − Q) | `alpha`, `decay` |
| **BayesianPortLearner** | Beta(a,b) posterior per port | `prior_strength`, `decay` |
| **HiddenStatePortLearner** | Belief over permutations of known reward probs | `alpha`, `decay` |
| **BayesianHiddenStatePortLearner** | Hidden state + Beta posteriors (in progress) | `prior_strength`, `decay` |


### Port value learning common interface

All port learners share the same interface:

| Method | Description |
| --- | --- |
| `update(port, reward)` | Update after one port visit. Returns prediction error (RW) or surprise |
| `learn(ports, rewards)` | Run updates on a full sequence. Returns list of errors/surprises |
| `get_values()` | Current port values as `{port: value}` |
| `get_history()` | Full trial-by-trial record |
| `choice_probabilities(available_ports)` | Softmax probabilities over port values |
| `reset()` | Re-initialize to starting state |
| `nll(ports, rewards)` | Negative log-likelihood of a sequence |
| `fit(ports, rewards)` | (classmethod) Fit free parameters via MLE, returns instance with `.nll_`, `.bic_` |


## Hex value learning

Learns values over individual hexes from maze trajectories (sequences of hex visits). 

### `HexMazeTDLearner` — TD value learning

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

**Parameters:**

- **`td0_alpha`**: learning rate for TD(0) forward updates (default 0.3; set to 0 to disable)
- **`td1_alpha`**: learning rate for TD(1) backward pass (default 0.1; set to 0 to disable)

### `HexMazeQLearner` — Q-learning

Q-learning agent that maintains 3 Q-tables (one per starting port).

- Learns **Q(hex, action)** — the value of moving to a specific neighbor from a given hex
- Captures directional preferences (e.g. "from hex 25, moving toward hex 26 is better than hex 24")
- Off-policy: updates use `max Q(next_hex, a')` regardless of the action actually taken

**Q-learning update** (at each step along the trajectory):

```text
Q(hex, a) ← Q(hex, a) + α · [r + γ · max_a' Q(next_hex, a') - Q(hex, a)]
```

At terminal hexes (reward ports), `max Q(next_hex, a') = 0`.

**Parameters:**

- **`alpha`**: learning rate (default 0.3)

### Hex value learning common interface

Both hex learners share the same interface:

| Method | Description |
| --- | --- |
| `learn(trajectories, rewards, start_ports)` | Run updates on provided trajectories (rat hex paths) |
| `simulate(start_hex, n_trials, max_steps)` | Self-generated trajectories with online updates |
| `process_trajectory(path, reward, start_port)` | Update on a single trajectory |
| `process_trajectory_with_history(...)` | Same as above, but returns hex value snapshots at each step |
| `action_probabilities(hex, start_port)` | Softmax choice probabilities at a hex |
| `get_state_values(start_port)` | Per-hex values under one start port |
| `get_max_state_values()` | Max value across all 3 tables per hex |
| `reset()` | Re-initialize tables and re-apply hex value priors |
| `set_graph(new_graph)` | Swap the maze graph (e.g. after barrier changes) |

**Q-learner only:**

| Method | Description |
| --- | --- |
| `get_q_values(start_port)` | Full Q-table: `{hex: {neighbor: q_value}}` |

**Shared parameters:**

- **`reward_probs`**: `[p1, p2, p3]` — reward probability at each port
- **`gamma`**: discount factor (default 0.95)
- **`temperature`**: softmax temperature for action selection (default 1.0)
- **`priors`**: V/Q-table initialization — `None` (zeros), `"uniform"` (0.5 at ports, γ^dist elsewhere), `("flat", value)`, or `[p1, p2, p3]` (γ^dist × reward_prob)
- **`no_backtrack`**: if `True`, agent avoids revisiting states within a trial (default `False`)

**Table update logic:**

Both learners use the same rule for deciding which of the 3 tables to update at a given hex:

- If the hex is in the same third as the start port (or is a critical choice point): update only the start port's table
- If the hex is in a different third T: update all tables except T's

TODO: decide if I like this. Figure out what to do with mazes that have more than 1 critical choice point!

**`start_port` defaults:**

In both learners, `start_port` is optional for `process_trajectory` and `process_trajectory_with_history`. It defaults to `path[0]` if that hex is a reward port; otherwise an error is raised.

## Quick start

```python
from hexmaze import maze_to_graph, plot_hex_maze
from hexmaze.rl import HexMazeTDLearner, HexMazeQLearner
from hexmaze.rl import RescorlaWagner, BayesianPortLearner, HiddenStatePortLearner

### Hex learning (trajectory-based)

barriers = {37, 7, 39, 41, 14, 46, 20, 23, 30}
graph = maze_to_graph(barriers)

td = HexMazeTDLearner(graph, reward_probs=[0.9, 0.5, 0.1], priors=[0.9, 0.5, 0.1])
ql = HexMazeQLearner(graph, reward_probs=[0.9, 0.5, 0.1], priors=[0.9, 0.5, 0.1])

# Simulate self-generated exploration
td_results = td.simulate(start_state=1, n_trials=100)
ql_results = ql.simulate(start_hex=1, n_trials=100)

# Or learn from rat trajectories
td.reset()
td.learn(trajectories, rewards, start_ports)

# Inspect learned values
td_values = td.get_state_values(start_port=1)       # {hex: V(hex)}
ql_values = ql.get_state_values(start_port=1)        # {hex: max Q(hex, a)}
ql_q_values = ql.get_q_values(start_port=1)          # {hex: {neighbor: Q}}

# Visualize on the maze
plot_hex_maze(barriers, color_by=td_values, colormap='viridis')

### Port learning (outcome-based)

rw = RescorlaWagner(alpha=0.3, decay=0.05)
bayes = BayesianPortLearner(prior_a=1, prior_b=1, decay=0.05)
hs = HiddenStatePortLearner(reward_set=(0.9, 0.5, 0.1), decay=0.03)

# Learn from reward history
ports = ['A', 'B', 'C', 'A', 'B', 'C']
rewards = [1, 0, 0, 1, 1, 0]
rw.learn(ports, rewards)
bayes.learn(ports, rewards)
hs.learn(ports, rewards)

# Compare port values
rw.get_values()      # {1: ..., 2: ..., 3: ...}
bayes.get_values()   # posterior means
hs.get_values()      # belief-weighted expected values

# Fit models to data
rw_fit = RescorlaWagner.fit(ports, rewards)
print(f"RW: NLL={rw_fit.nll_:.2f}, BIC={rw_fit.bic_:.2f}")
```