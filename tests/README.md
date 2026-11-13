# Tests

Tests that check our work. Run them from the repo with:

```sh
pytest tests/
```

| Test file | What it checks |
| --- | --- |
| `test_hierarchical.py` | Our hierarchical fitting (`src/hexmaze/rl/hierarchical.py`) gets the same answers as the Daw lab's em package on em's own reference data: each subject's fit given a group distribution (the E-step), and the whole EM fit, with a covariate, with and without a full covariance. Takes a few seconds |

`em_reference/` has em's reference data and answers (see its [README](em_reference/README.md)).
