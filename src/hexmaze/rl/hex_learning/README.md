# hexmaze.rl.hex_learning

Learns a value for each hex of the maze from the paths the rat runs, and turns those values into choices.

**NOTE: hex learning agents are still in progress**

Start with the tutorial: [41_Hex_Value_Learning](../../../../Tutorials/41_Hex_Value_Learning.ipynb).
It covers fitting a model to a real session, looking at the learned hex values (and how they change
after each trial), and animating learning, plus how to fit a session with barrier changes, get the value of every
hex after each trial, get the value of where the rat is at every step, and compare models.

```text
hex_learning/
├── base_learner.py     # BaseHexLearner: value table, learning loop, fitting, animation
├── td_learner.py       # HexMazeTDLearner  (TD(λ))
├── dual_learner.py     # KrauszDualLearner (TD(λ) + model-based memory trace)
├── swr_learner.py      # SWRDualLearner    (TD(λ) + SWR replay)
├── update_rules.py     # UpdateRule, TDLambdaRule, ModelBasedRule, SWRRule
├── retrieval_rules.py  # RetrievalRule, LocalRetrieval, BestSwept..., FurthestSweptRetrieval
└── fitting.py          # fit_many (fit models in parallel)
```

## How a learner is built

A learner is a `BaseHexLearner` with:

- a list of **update rules**, which decide how the rat's experience changes hex values
- one **retrieval rule**, which decides how the agent values each possible move when it makes a choice
- a **state space**, which decides what gets a value

All learners share the same value table `V`, learning loop, fitting, and animation code.
Each update rule is called at 3 points in a trial: before the first step, after every hex-to-hex step,
and once the rat reaches the port. Several rules can run on the same value table at once.

## Learners

| Learner | Update rule(s) | Parameters (defaults) |
|---|---|---|
| `HexMazeTDLearner` | TD(λ) | `alpha=0.3`, `gamma=0.95`, `lam=0.0`, `temperature=1.0` |
| `KrauszDualLearner` | TD(λ) + model-based memory trace (Krausz et al. 2023) | + `gamma_mb=0.9`, `a_T=0.1`, `a_mb=0.1` |
| `SWRDualLearner` | TD(λ) + SWR replay | + `a_swr=0.1` |
| `BaseHexLearner` | any list you pass as `rules=` | whatever those rules read |

Any other combination of rules can be built directly:

```python
BaseHexLearner(
    maze,
    rules=[TDLambdaRule(), ModelBasedRule(), SWRRule()],
    alpha=0.3, gamma=0.95, lam=0.0,      # TDLambdaRule reads these
    gamma_mb=0.9, a_T=0.1, a_mb=0.1,     # ModelBasedRule reads these
    a_swr=0.1,                           # SWRRule reads this
    goal_conditioned_states=True, directional_states=False,
)
```

## Update rules

**`TDLambdaRule`**: on every hex-to-hex step, the hex the rat is leaving moves toward the discounted value of
the hex it's entering. At the port, the port's value also moves toward the reward. Both updates go through an
eligibility trace, so with `lam > 0` hexes visited earlier in the trial are updated too.

```text
td_error = gamma * V(next hex) - V(current hex)     # every step
td_error = reward - V(port)                         # also on the step into the port
V(hex) <- V(hex) + alpha * td_error * e(hex)        # e(hex) += 1 on visit, decays by gamma * lam per step
```

`lam = 0` is TD(0): value creeps back one hex per traversal. `lam = 1` is Monte Carlo.

**`ModelBasedRule`** (Krausz et al.): keeps a memory trace `m` of where the rat has been this trial, averages it
into a persistent map `T(port, state)` when the rat reaches a port, then moves every state linked to that port
toward the reward.

```text
m(state) <- gamma_mb * m(state)      # every step;  m(state just entered) <- 1
T(port, state) <- (1 - a_T) * T(port, state) + a_T * m(state)       # at the port
V(state) <- V(state) + a_mb * T(port, state) * (reward - V(state))  # at the port
```

**`SWRRule`**: at the port, each hex replayed during that trial's SWRs moves toward the reward.
Pass the replayed hexes for each trial as `replayed_hexes`.

```text
V(hex) <- V(hex) + a_swr * (reward - V(hex))
```

For a directional learner, a replayed hex updates the directions of travel through it that head toward the port.

## Retrieval rules

Pass one as `retrieval=` (defaults to `LocalRetrieval`). The sweep-based rules need `swept_hexes`:
the sweeps the rat made at each hex of each trial's path.

| Rule | Value of a move |
|---|---|
| `LocalRetrieval` | the value of the hex the move steps into |
| `BestSweptRetrieval` | the best value among the hexes swept to down that way |
| `BestSweptOrCachedRetrieval` | the better of those two (so a sweep can only raise a move's value) |
| `FurthestSweptRetrieval` | the value of the last hex the sweep down that way reached |

A move with no sweeps down it gets the value of the hex it steps into.

## State space

Use exactly one of these:

- **`directional_states=True`** (the default): one value per direction of travel through each hex, `(prev_hex, hex)`.
  The same hex can have a different value depending on which way the rat is going.
- **`goal_conditioned_states=True, directional_states=False`**: one value per hex, with a separate value table for
  each start port (if the rat starts at A, only B and C are possible goals, so the values should reflect that).

## Other options

- **`priors`**: starting values. `None` (all 0, default), `"uniform"` (0.5 at ports, decaying by `gamma` per hex
  of empty-maze distance), `("flat", value)` (every hex starts at `value`, e.g. `("flat", 0.2)` matches Tim's paper),
  or `[pA, pB, pC]` (like `"uniform"` but with your own port values).
- **`temperature`**: softmax temperature for choices.
- **`reward_probs`**: `[pA, pB, pC]`, only used to sample rewards in `simulate`.
- **`no_backtrack`**: if True (default), the simulated agent avoids revisiting hexes in a trial. Only affects `simulate`.

## Data format

- `trajectories`: one list of hexes per trial, e.g. `[1, 4, 5, 7, ..., 2]`. Each trial should start and end at a port
  (the session's first trial is allowed to start somewhere else).
- `rewards`: one 0/1 per trial.
- `mazes`, `trials_per_block`: for sessions with barrier changes, the maze for each block and how many trials are in
  each block. The learner swaps in each new maze at the right trial. Hexes that become barriers lose their values,
  and newly opened hexes start at their prior.
- `replayed_hexes` (optional): one list of replayed hexes per trial (used by `SWRRule`).
- `swept_hexes` (optional): for each trial, one entry per hex of the path, holding the sweeps made at that hex
  (each a list of hexes), or a single list of represented hexes (used by the sweep-based retrieval rules).

Tracking errors in a trajectory (e.g. a jump between two hexes that aren't neighbors) are warned about and skipped.

## Methods

| Method | What it does |
|---|---|
| `learn(trajectories, rewards, ...)` | Learn from the rat's trajectories. `record="trial"` or `"step"` returns value snapshots |
| `fit_choices(maze, trajectories, rewards, ...)` | (classmethod) Fit parameters to the rat's choices. Returns a fresh learner with `.choice_nll_`, `.choice_bic_`, `.choices_scored_` |
| `choice_nll(trajectories, rewards, ...)` | NLL of the rat's choices under the current parameters. `record=True` also returns per-choice probabilities |
| `value_timeseries(trajectories, rewards, sampled_hexes, represented_hexes=None, ...)` | Value of where the rat is (and what it's representing) at every sample, as a DataFrame |
| `animate_learning(trajectories, rewards, ...)` | Animate learning one update at a time. `save_to=` writes an .mp4 or .gif |
| `get_state_values(start_port)` | Current values as `{hex: value}` from one start port |
| `snapshot_values(snapshot, start_port=None)` | Turn a recorded snapshot into `{hex: value}` |
| `simulate(start_state, n_trials)` | Let the agent run the maze itself (using `reward_probs`) |
| `reset()` | Clear all values back to the priors |
| `set_graph(maze)` | Swap in a new maze (e.g. on a barrier change) |
| `critical_choice_points(start_port=None)` | The critical choice point(s) of the current maze |

`fit_many(jobs)` runs a dict of `{key: (learner class, fit_choices arguments)}` fits in parallel and returns
`{key: fitted learner}`.

**`score_at`** (for `fit_choices` and `choice_nll`) picks which of the rat's moves are scored:

- `"steps"` (default): every step, against all neighbors
- `"junctions"`: only left/right choices at 3-way junctions (backtracking at a junction isn't scored)
- `"choice_points"`: only left/right choices at the critical choice point for the trial's start port

## Things to watch out for

- A learner remembers what it has learned and where it is in the session, so call `reset()` before running
  through the session again.
- `reset()` clears values but keeps the current maze. Anything you pass `mazes` to swaps in the right maze itself,
  but if you only use some of the trials, call `set_graph(mazes[0])` first.
- `fit_choices` returns an **untrained** learner, so call `learn` on it to get the learned values.
- `animate_learning` actually updates the learner as it builds frames.
- `fit_many` runs fits in separate processes, so `warnings.catch_warnings()` around it doesn't hide their warnings
  (each worker process prints its own).

## Quick start

```python
from hexmaze import plot_hex_maze
from hexmaze.rl import KrauszDualLearner

# trajectories, rewards, mazes, trials_per_block loaded from a session (see the tutorial)
session_data = dict(trajectories=trajectories, rewards=rewards,
                    mazes=mazes, trials_per_block=trials_per_block)

learner = KrauszDualLearner.fit_choices(
    maze=mazes[0], **session_data, lam=0.0, score_at="junctions",
    directional_states=True,  # one value per direction of travel (the default)
)
print(learner.choice_bic_)

learner.learn(**session_data)
plot_hex_maze(learner.graph, color_by=learner.get_state_values(start_port="A"))
```
