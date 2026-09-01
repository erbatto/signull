# signull — random-signature null testing for molecular biomarkers

Given a candidate gene signature, a cohort expression matrix and a **binary** outcome,
answer one question with a defensible number:

> Is this signature more predictive than signatures of the same size drawn at random
> from the same dataset?

Motivated by the finding that 60% of 47 published breast-cancer outcome signatures were
no better than size-matched random signatures, and that >90% of random signatures longer
than 100 genes were significant outcome predictors (Venet et al. 2011, PLoS Comput Biol).

Status: scaffold. See `docs/` for the design specs produced by the fleet session.
