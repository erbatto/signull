# Signull — random-signature null testing for molecular biomarkers

Given a candidate gene signature, a cohort expression matrix and a **binary** outcome,
answer one question with a defensible number:

> Is this signature more predictive than signatures of the same size drawn at random
> from the same dataset?

Motivated by the finding that 60% of 47 published breast-cancer outcome signatures were
no better than size-matched random signatures, and that >90% of random signatures longer
than 100 genes were significant outcome predictors (Venet et al. 2011, PLoS Comput Biol).

Status: waves 1-2 complete. The data, nulls, scoring and metrics packages are implemented
and tested (87 tests); the orchestration pipeline, CLI, report layer and the Sec. 7
calibration acceptance tests are wave 3.

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

See `docs/` for the statistical design, the prior-art survey and the architecture contract.
