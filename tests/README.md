# Tests

Run from the repo with:

```sh
pip install -e ".[test]"  # once, to get pytest
pytest tests/
```

`test_hierarchical.py` checks that our hierarchical fitting (`src/hexmaze/rl/hierarchical.py`) gets the same answers as
the Daw lab's em package on em's reference data (the E-step, and the whole EM fit with a covariate, with and without a
full covariance, the standard errors of the effects, and leave-one-subject-out cross validation). It also checks what
we added on top of em: groups (each rat its own group distribution), text covariates, checkpoints, and
`fit_choices_hierarchical` with groups.

`test_persistence.py` checks `PersistenceRule`: the softmax gets exactly the persistence bonus for repeating the
last turn at a junction (and only when coming through it the same way), and fitting gets back the persistence
simulated sessions were made with.

`em_reference/` has em's reference data and answers (see its [README](em_reference/README.md)).
