# Fleet Session: signull

Status: active
Started: 2026-09-01
Direction: Implement a tool that, given a signature and a cohort of patients with a disease,
builds a null distribution of signatures to test whether the candidate signature is more
predictive than chance.

## Confirmed scope (from decision owner)
- Endpoint: **binary outcome** (AUC / average precision), not survival
- Data: **public benchmark first**, own data plugs into the same loader later
- Repo: `/Users/ieo7498/Desktop/AILABS` (pre-existing MarkerDB exports preserved)

## Recorded deviations from the Fleet protocol
1. **No worktree isolation.** The session root (`~/Desktop`) is not a git repository; only
   `AILABS/` is. Agent worktree isolation resolves from the session root and would fail.
   Mitigation: strictly non-overlapping file scopes per agent, enforced in each prompt, and
   scope compliance verified after each wave. Recorded per the skill's fringe-case rule.
2. `.citadel` telemetry scripts are present but the project has no `npm`/node project wiring;
   telemetry logging is attempted best-effort and skipped silently on failure.

## Work Queue
| # | Campaign | Scope | Deps | Status | Wave | Agent type |
|---|----------|-------|------|--------|------|-----------|
| 1 | Statistical design of the null | docs/statistical-design.md | none | pending | 1 | research scout |
| 2 | Prior art + benchmark data acquisition | docs/prior-art-and-data.md, scripts/ | none | pending | 1 | research scout |
| 3 | Architecture + data contracts | docs/architecture.md, src/signull/types.py | none | pending | 1 | architecture scout |
| 4 | Data layer: loaders, signature I/O, entity resolution | src/signull/data/, tests/test_data.py | 3 | pending | 2 | builder |
| 5 | Null engine: sampling schemes, matching, permutation | src/signull/nulls/, tests/test_nulls.py | 1,3 | pending | 2 | builder |
| 6 | Scoring + metrics: signature scoring, AUC/AP, CV | src/signull/scoring/, src/signull/metrics/, tests/test_scoring.py | 1,3 | pending | 2 | builder |
| 7 | Integration, CLI, report, calibration test on real data | src/signull/report/, src/signull/cli.py, tests/test_calibration.py | 4,5,6 | pending | 3 | integrator |

## Shared Context (Discovery Relay — seeded from the prior brainstorm register)
Seeded before Wave 1 so agents do not rediscover it. Source: session `biomarker-validator-01`,
`~/Desktop/biomarker-validation-brainstorm/`.

- Venet 2011 (PLoS Comput Biol): 60% of 47 published breast-cancer signatures were **no better
  than size-matched random signatures**; 23% worse than the median random one; **>90% of random
  signatures over 100 genes** were significant outcome predictors.
- Venet 2011: adjusting for a **proliferation metagene abrogated almost all outcome association
  of published AND random signatures**; >50% of the breast-cancer transcriptome correlates with
  that single axis. A null that ignores the dominant latent axis measures the wrong thing.
- Starmans 2011: the share of random signatures reaching significance ranges **~1% to ~40%
  depending on the dataset** → the null must be **dataset-specific**; a fixed threshold is
  uninterpretable.
- Michiels 2005 (Lancet): signature membership highly unstable across resampled training sets;
  5 of 7 large studies classified no better than chance.
- **SigCheck** (Bioconductor) already implements a random-signature benchmark. Prior art —
  compare against it, do not blindly rebuild.
- Two DIFFERENT nulls, not interchangeable: (a) **random gene-set null** — is this gene set
  special among gene sets? (b) **label-permutation null** — is there any signal at all?
- Null gene sets must match the candidate on **size**, and should be matched on **expression
  level, detection rate and variance** — an unmatched null is misspecified and will flatter the
  candidate.
- The **scoring method** (mean z-score / first PC / ssGSEA / fitted model) must be identical for
  candidate and null signatures, and results are known to be scoring-method dependent.
- Acceptance test for the whole tool: under a **true null (permuted labels)**, the reported
  p-values must be approximately **uniform**. If they are not, the tool is broken.
- If any model fitting happens, feature selection must sit **inside** the cross-validation loop.

## Environment (verified by Fleet, 2026-09-01)
Conda env `signull` at `/Users/ieo7498/miniconda3/envs/signull` — python `/Users/ieo7498/miniconda3/envs/signull/bin/python`
- Python 3.11.16, numpy 2.4.6, scipy 1.17.1, **pandas 3.0.5**, scikit-learn 1.9.0, matplotlib, pytest 9.1.1
- **Relay note for Wave 2:** pandas is 3.x, not 2.x. Copy-on-write is the default and the default
  string dtype changed. Code written against pandas 2 idioms (chained assignment, `inplace=`,
  implicit object dtype for strings) may warn or behave differently. numpy is 2.x as well.
- Smoke check: AUROC on pure noise with n=200 returned 0.511, i.e. the metric path behaves.

## Continuation State
Next wave: 1
Blocked items: none
Auto-continue: true
