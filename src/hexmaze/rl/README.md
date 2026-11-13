# hexmaze.rl

Reinforcement learning agents for the hex maze.

**NOTE: hex learning agents are still in progress**

```text
hexmaze/rl/
├── hex_learning/        # trajectory-based, learns values over hexes
│   ├── base_learner.py     # BaseHexLearner
│   ├── td_learner.py       # HexMazeTDLearner
│   ├── dual_learner.py     # KrauszDualLearner
│   ├── swr_learner.py      # SWRDualLearner
│   ├── update_rules.py     # TDLambdaRule, ModelBasedRule, SWRRule
│   └── retrieval_rules.py  # LocalRetrieval, BestSweptRetrieval, ...
├── port_learning/       # outcome-based, learns values over ports
│   ├── rescorla_wagner.py       # RescorlaWagner
│   ├── bayesian.py              # BayesianPortLearner
│   ├── hidden_state.py          # HiddenStatePortLearner
│   └── bayesian_hidden_state.py # BayesianHiddenStatePortLearner (in progress)
├── fitting.py           # fit_from_starts (how every learner fits its parameters), fit_many
└── hierarchical.py      # fit_hierarchical (fit many sessions at once, hierarchically)
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

Learns values over individual hexes from maze trajectories (the hex-by-hex path the rat ran on each trial)
and whether each trial was rewarded.

See the full hex learning [README](hex_learning/README.md) for detailed docs, and the
[41_Hex_Value_Learning](../../../Tutorials/41_Hex_Value_Learning.ipynb) tutorial for a walkthrough on real data.

A learner is a `BaseHexLearner` with a list of **update rules** (how experience changes hex values) and one
**retrieval rule** (how the agent values each possible move).

| Learner | Update rule(s) |
| --- | --- |
| **HexMazeTDLearner** | TD(λ) |
| **KrauszDualLearner** | TD(λ) + model-based memory trace (Krausz et al. 2023) |
| **SWRDualLearner** | TD(λ) + SWR replay |

| Retrieval rule | Value of a move |
| --- | --- |
| **LocalRetrieval** (default) | the hex the move steps into |
| **BestSweptRetrieval** | the best hex swept to down that way |
| **BestSweptOrCachedRetrieval** | the better of those two |
| **FurthestSweptRetrieval** | the last hex the sweep down that way reached |

States are either directional (one value per direction of travel through a hex, the default) or
goal-conditioned (one value per hex, with a separate table per start port). Use exactly one.

### Hex value learning common interface

| Method | Description |
| --- | --- |
| `learn(trajectories, rewards)` | Learn from the rat's trajectories. `record="trial"`/`"step"` returns value snapshots |
| `fit_choices(maze, trajectories, rewards)` | (classmethod) Fit parameters to the rat's choices (unbounded, from multiple starting points). Returns a fresh learner with `.choice_nll_`, `.choice_bic_` |
| `choice_nll(trajectories, rewards)` | NLL of the rat's choices. `record=True` also returns per-choice probabilities |
| `value_timeseries(trajectories, rewards, sampled_hexes)` | Value of where the rat is (and what it's representing) at each sample |
| `animate_learning(trajectories, rewards)` | Animate learning one update at a time (`save_to=` for .mp4/.gif) |
| `get_state_values(start_port)` | Current values as `{hex: value}` |
| `simulate(start_state, n_trials)` | Let the agent run the maze itself |
| `reset()` | Clear values back to the priors |
| `set_graph(maze)` | Swap in a new maze (e.g. on a barrier change) |

`fit_many(jobs)` fits a dict of models in parallel. Sessions with barrier changes are handled by passing
`mazes` and `trials_per_block` to any of the methods above.

## Quick start

```python
from hexmaze import plot_hex_maze
from hexmaze.rl import KrauszDualLearner
from hexmaze.rl import RescorlaWagner, BayesianPortLearner, HiddenStatePortLearner

### Hex learning (trajectory-based)

# trajectories, rewards, mazes, trials_per_block loaded from a session
# (see Tutorials/41_Hex_Value_Learning.ipynb)
session_data = dict(trajectories=trajectories, rewards=rewards,
                    mazes=mazes, trials_per_block=trials_per_block)

learner = KrauszDualLearner.fit_choices(
    maze=mazes[0], **session_data, lam=0.0, score_at="junctions",
    directional_states=True,  # one value per direction of travel (the default)
)
learner.learn(**session_data)

# Visualize learned values on the maze
plot_hex_maze(learner.graph, color_by=learner.get_state_values(start_port="A"), colormap="viridis")

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