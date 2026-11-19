# Tests

Tests that check our work. Run them from the repo with:

```sh
pip install -e ".[test]"  # once, to get pytest
pytest tests/
```

| Test file | What it checks |
| --- | --- |
| `test_hierarchical.py` | Our hierarchical fitting (`src/hexmaze/rl/hierarchical.py`) gets the same answers as the Daw lab's em package on em's own reference data: each subject's fit given a group distribution (the E-step), and the whole EM fit, with a covariate, with and without a full covariance. Also checks picking up from checkpoints, and that `fit_choices_hierarchical` runs on simulated sessions. Takes about 10 seconds |

`em_reference/` has em's reference data and answers (see its [README](em_reference/README.md)).
