# Port Value Learning

Learns value (expected reward probability) for each port (1/2/3 or A/B/C) from port visits and binary reward outcomes (0 or 1).

```text
port_learning/
├── rescorla_wagner.py       # RescorlaWagner
├── bayesian.py              # BayesianPortLearner
├── hidden_state.py          # HiddenStatePortLearner
└── bayesian_hidden_state.py # BayesianHiddenStatePortLearner (in progress)
```

See [Tutorial 40 — Port Value Models](../../../../Tutorials/40_Port_Value_Models.ipynb) for fitted
examples, parameter explorations, and confidence intervals.

## Models

### `RescorlaWagner` — Q learning

Tracks a Q-value per port, updated with a simple reward prediction error (RPE) rule.

**Update rule:**

```text
Q(port) ← Q(port) + α · [reward − Q(port)]
```

With optional per-trial decay toward `initial_value` (default 0.5):

```text
Q(port) ← (1 − decay) · Q(port) + decay · initial_value
```

- Returns **reward prediction error** (`reward − Q(port)`) from each update
- Free parameters for fitting: `alpha`, `decay`

### `BayesianPortLearner` — Beta-binomial posterior

Maintains a Beta(a, b) posterior over each port's reward probability.

**Update rule:**

```text
reward = 1:  a ← a + 1
reward = 0:  b ← b + 1
```

Expected value (posterior mean) = `a / (a + b)`.

With optional per-trial decay toward the prior:

```text
a ← (1 − decay) · a + decay · prior_a
b ← (1 − decay) · b + decay · prior_b
```

- Returns **surprise** (`−log p(observed outcome)`) from each update (same as Shannon "self-information")
- Provides `confidence_interval(port, ci)` for credible intervals
- Free parameters for fitting: `prior_strength` (sets prior_a = prior_b), `decay`

### `HiddenStatePortLearner` — Bayesian belief over reward assignments

Assumes the three ports have a known set of reward probabilities (e.g. 0.9, 0.5, 0.1) but the
assignment of probabilities to ports is unknown. Maintains a belief distribution over all 6
possible permutations.

**Update rule:**

```text
# Optional decay: drift belief toward uniform
belief ← (1 − decay) · belief + decay · uniform

# Compute Bayesian update
likelihood[i] = p(reward | state_i, port)
updated = belief · likelihood
updated = updated / sum(updated)

# Interpolate with learning rate (alpha=1.0 is full Bayesian)
belief ← (1 − α) · belief + α · updated
```

Expected value for a port = weighted average across all states:

```text
E[reward | port] = Σᵢ belief[i] · state_i[port]
```

- Returns **surprise** (`−log p(observed outcome)`) from each update (same as Shannon "self-information")
- Provides `expected_value_std(port)` — uncertainty from belief spread across permutations
- Free parameters for fitting: `alpha`, `decay`

### `BayesianHiddenStatePortLearner` — Hidden state + Beta posteriors (in progress)

Combines the hidden-state structure with Beta posteriors per slot. Instead of fixed reward
probabilities, each slot has a Beta posterior that gets updated. Uses soft EM — structural
learning (which port has which slot) and parametric learning (what each slot's probability is)
happen simultaneously.

- Free parameters for fitting: `prior_strength`, `decay`

## Common Interface

All port learners share these methods:

| Method | Description |
|---|---|
| `update(port, reward)` | Update after one port visit. Returns prediction error (RW) or surprise (Bayesian/HS) |
| `learn(ports, rewards)` | Run updates on a full sequence. Returns list of errors/surprises |
| `get_values()` | Current port values as `{1: v1, 2: v2, 3: v3}` |
| `get_history()` | Full trial-by-trial record (values, errors/surprise, model internals at each step) |
| `choice_probabilities(available_ports)` | Softmax probabilities over current port values |
| `reset()` | Re-initialize to starting state, keeping parameters |
| `reward_nll(ports, rewards)` | Negative log-likelihood of a reward sequence under current parameters |
| `choice_nll(ports, rewards)` | Negative log-likelihood of a port choice sequence under current parameters |
| `fit_rewards(ports, rewards)` | (classmethod) Fit free parameters by minimizing NLL via L-BFGS-B. Returns fitted instance with `.reward_nll_`, `.reward_bic_` attributes |
| `fit_choices(ports, rewards)` | (classmethod) Fit free parameters by minimizing NLL via L-BFGS-B. Returns fitted instance with `.choice_nll_`, `.choice_bic_` attributes |

**Shared parameters:**

- **`temperature`**: softmax temperature for `choice_probabilities` (RW/Bayesian set at init; HS passes per call)
- **`decay`**: per-trial decay toward initial values (0 = no forgetting)

## Model-Specific Methods

### BayesianPortLearner

| Method | Description |
|---|---|
| `expected_value(port)` | Posterior mean for a port |
| `confidence_interval(port, ci=0.95)` | Credible interval `(lower, upper)` from Beta posterior |
| `get_posteriors()` | Raw Beta parameters as `{port: {"a": float, "b": float}}` |

### HiddenStatePortLearner

| Method | Description |
|---|---|
| `expected_value(port)` | Belief-weighted expected reward probability |
| `expected_value_std(port)` | SD of expected reward from belief spread: `√(Σ belief[i] · (pᵢ − μ)²)` |
| `get_stds()` | SD for all ports as `{port: std}` |
| `get_state_posteriors()` | Belief over each permutation: list of `{"assignment": {port: p}, "probability": float}` |

## Quick Start

```python
from hexmaze.rl.port_learning import (
    RescorlaWagner,
    BayesianPortLearner,
    HiddenStatePortLearner,
)

ports   = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']
rewards = [1,    0,   0,   1,   1,   0,   1,   0  ]

# Fit models to data
rw_fit    = RescorlaWagner.fit_rewards(ports, rewards)
bayes_fit = BayesianPortLearner.fit_rewards(ports, rewards)
hs_fit    = HiddenStatePortLearner.fit_rewards(ports, rewards)

print(f"RW:    NLL={rw_fit.reward_nll_:.2f}, BIC={rw_fit.reward_bic_:.2f}, alpha={rw_fit.alpha:.3f}")
print(f"Bayes: NLL={bayes_fit.reward_nll_:.2f}, BIC={bayes_fit.reward_bic_:.2f}")
print(f"HS:    NLL={hs_fit.reward_nll_:.2f}, BIC={hs_fit.reward_bic_:.2f}")

# Step through trials manually
rw = RescorlaWagner(alpha=0.3, decay=0.05)
for port, reward in zip(ports, rewards):
    pe = rw.update(port, reward)
    print(f"Port {port}, reward={reward}, PE={pe:+.3f}, Q={rw.get_values()}")

# Get uncertainty estimates
bayes = BayesianPortLearner(prior_a=1.0, prior_b=1.0, decay=0.05)
bayes.learn(ports, rewards)
for p in [1, 2, 3]:
    lo, hi = bayes.confidence_interval(p, ci=0.95)
    print(f"Port {p}: {bayes.expected_value(p):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

hs = HiddenStatePortLearner(reward_set=(0.9, 0.5, 0.1), decay=0.03)
hs.learn(ports, rewards)
for p in [1, 2, 3]:
    print(f"Port {p}: {hs.expected_value(p):.3f} ± {hs.expected_value_std(p):.3f}")

# Inspect hidden state beliefs
for state in hs.get_state_posteriors():
    print(f"  {state['assignment']}  P={state['probability']:.4f}")
```

## Model Comparison with BIC

Each model's free parameters are fit by minimizing negative log-likelihood (NLL) over the port visit / reward sequence.

All models expose `.reward_nll_`/`.choice_nll_` and `.reward_bic_`/`.choice_bic_` after fitting. 

We can compare the fit of different models using Bayesian Information Criterion (BIC).

BIC = k·ln(n) + 2·NLL penalizes extra parameters so models with different numbers of free parameters can be compared on the same scale (lower is better).

All of our models only have 2 free parameters (for now) so we can also just compare NLL.


```python
bics = {
    'Rescorla-Wagner': rw_fit.reward_bic_,
    'Bayesian':        bayes_fit.reward_bic_,
    'Hidden State':    hs_fit.reward_bic_,
}
best = min(bics, key=bics.get)
print(f"Best model: {best} (BIC={bics[best]:.1f})")
```
