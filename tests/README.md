# Tests

Run from the repo with:

```sh
pip install -e ".[test]"  # once, to get pytest
pytest tests/
```

`test_hierarchical.py` checks that our hierarchical fitting (`src/hexmaze/rl/hierarchical.py`) gets the same answers as
the Daw lab's em package on em's reference data (the E-step, and the whole EM fit with a covariate, with and without a
full covariance). It also checks checkpoints, and that `fit_choices_hierarchical` runs. Takes about 10 seconds.

`em_reference/` has em's reference data and answers (see its [README](em_reference/README.md)).
