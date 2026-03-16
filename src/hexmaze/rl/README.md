# hexmaze.rl

Reinforcement learning agents for the hex maze.

```text
hexmaze/rl/
├── hex_learning/        # trajectory-based, learns values over hexes
│   ├── td_learner.py    # HexMazeTDLearner
│   └── q_learner.py     # HexMazeQLearner
└── port_learning/       # outcome-based, learns values over ports
    ├── rescorla_wagner.py  # RescorlaWagner
    └── bayesian.py         # BayesianPortLearner
```

## Hex value learning

Learns values over individual hexes from maze trajectories (sequences of hex visits).

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

### Hex learning common interface

Both hex learners share the same interface:

| Method | Description |
|---|---|
| `learn(trajectories, rewards, start_ports)` | Run updates on given provided trajectories (rat hex paths) |
| `simulate(start_hex, n_trials, max_steps)` | Self-generated trajectories with online updates |
| `process_trajectory(path, reward, start_port)` | Update on a single trajectory |
| `process_trajectory_with_history(...)` | Same as above, but returns hex value snapshots at each step |
| `action_probabilities(hex, start_port)` | Softmax choice probabilities at a hex |
| `get_state_values(start_port)` | Per-hex values under one start port |
| `get_max_state_values()` | Max value across all 3 tables per hex |
| `reset()` | Re-initialize tables and re-apply hex value priors |
| `set_graph(new_graph)` | Swap the maze graph (e.g. after barrier changes) |

**Shared parameters:**

- **`reward_probs`**: `[p1, p2, p3]` — reward probability at each port
- **`gamma`**: discount factor
- **`temperature`**: softmax temperature for action selection
- **`priors`**: V/Q-table initialization — `None`, `"uniform"`, `("flat", value)`, or `[p1, p2, p3]`
- **`no_backtrack`**: if `True`, agent avoids revisiting states within a trial, when possible (useful for simulation sometimes)

**Table update logic:**

Both learners use the same rule for deciding which of the 3 tables to update at a given hex:

- If the hex is in the same third as the start port (or is a critical choice point): update only the start port's table
- If the hex is in a different third T: update all tables except T's

TODO: decide if I like this. Figure out what to do with mazes that have more than 1 critical choice point!

**`start_port` defaults:**

In both learners, `start_port` is optional for `process_trajectory` and `process_trajectory_with_history`. It defaults to `path[0]` if that hex is a reward port; otherwise an error is raised.

## Port value learning

Learns values over reward ports from binary reward outcomes (0 or 1). No maze structure or trajectories needed — just which port was visited and whether reward was received.

### `rescorla_wagner.py` — `RescorlaWagner`

Delta-rule learner that tracks expected value of each port.

**Update rule:**

```text
Q(port) ← Q(port) + α · [reward - Q(port)]
```

- Returns **prediction error** (`reward - Q(port)`) from each update
- Optional **decay** toward `initial_value` for recency weighting

### `bayesian.py` — `BayesianPortLearner`

Maintains a Beta(a, b) posterior over each port's reward probability.

**Update rule:**

```text
reward = 1: a ← a + 1
reward = 0: b ← b + 1
```

Expected value (posterior mean) = `a / (a + b)`.

- Returns **Bayesian surprise** (`-log p(reward)`) from each update (how unlikely the outcome was)
- Provides `confidence_interval(port)` for uncertainty estimates
- Supports **Thompson sampling** via `thompson_choice()` — draws from each port's posterior and picks the highest
- Optional **decay** toward prior for forgetting

### Port learning common interface

Both port learners share the same interface:

| Method | Description |
|---|---|
| `update(port, reward)` | Update values after visiting a port and receiving reward |
| `learn(ports, rewards)` | Run updates on a sequence of port visits |
| `choice_probabilities(available_ports)` | Softmax probabilities over port values |
| `get_values()` | Current port values as `{port: value}` |
| `get_history()` | Full learning history (values, errors/surprise at each step) |
| `reset()` | Re-initialize to starting values |

**Shared parameters:**

- **`temperature`**: softmax temperature for `choice_probabilities`
- **`decay`**: per-trial decay toward initial values (0 = no forgetting)

## Usage

```python
from hexmaze.rl import HexMazeTDLearner, HexMazeQLearner
from hexmaze.rl import RescorlaWagner, BayesianPortLearner

### Hex learning (trajectory-based)
td = HexMazeTDLearner(graph, reward_probs=[0.9, 0.5, 0.1])
ql = HexMazeQLearner(graph, reward_probs=[0.9, 0.5, 0.1])

# Learn from rat trajectories
td.learn(trajectories, rewards)
ql.learn(trajectories, rewards)

# Or simulate
td_results = td.simulate(start_hex=1, n_trials=100)
ql_results = ql.simulate(start_hex=1, n_trials=100)

# Compare values
td_values = td.get_state_values(start_port=1)
ql_values = ql.get_state_values(start_port=1)  # max Q per hex
ql_q_values = ql.get_q_values(start_port=1)    # full Q(hex, action) table

### Port learning (based on reward outcomes only)
rw = RescorlaWagner(alpha=0.3)
bayes = BayesianPortLearner(prior_a=1, prior_b=1)

# Learn from reward history
ports = [1, 2, 1, 3, 2, 1]
rewards = [1, 0, 1, 0, 1, 1]
prediction_errors = rw.learn(ports, rewards)
surprises = bayes.learn(ports, rewards)

# Compare port values
rw.get_values()      # {1: 0.82, 2: 0.38, 3: 0.0}
bayes.get_values()   # {1: 0.75, 2: 0.50, 3: 0.33}

# Choice probabilities
rw.choice_probabilities()       # softmax over Q-values
bayes.choice_probabilities()    # softmax over posterior means
bayes.thompson_choice()         # sample from posteriors, pick highest
```
