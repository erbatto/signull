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
| 1 | Statistical design of the null | docs/statistical-design.md | none | complete | 1 | research scout |
| 2 | Prior art + benchmark data acquisition | docs/prior-art-and-data.md, scripts/ | none | complete | 1 | research scout |
| 3 | Architecture + data contracts | docs/architecture.md, src/signull/types.py | none | complete | 1 | architecture scout |
| 4 | Data layer: loaders, signature I/O, entity resolution | src/signull/data/, tests/test_data.py | 3 | complete | 2 | builder |
| 5 | Null engine: sampling schemes, matching, permutation | src/signull/nulls/, tests/test_nulls.py | 1,3 | complete | 2 | builder |
| 6 | Scoring + metrics: signature scoring, AUC/AP, CV | src/signull/scoring/, src/signull/metrics/, tests/test_scoring.py | 1,3 | complete | 2 | builder |
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

## Wave 1 Results (partial — scout 3 still running)

### Agent: statistical-design  — **complete**, scope clean
`docs/statistical-design.md`, 389 lines. Five nulls, binned matching pseudocode, `(1+r)/(1+K)`
p-values, PC1/meta-PCNA adjustment, both CV paths, 7 acceptance tests, 12 refusals, 21 citations.
Fleet verified three numeric claims independently and all hold: K>=2000 floor gives 9.7% relative
SE at p=0.05; T1 band [0.033,0.070] is ~2.5 binomial SD at n=1000; the add-one estimator returns
P(p<=0.05)=0.048 under a true null and never returns 0.

### Agent: architecture-contract — **complete**, scope clean
`src/signull/types.py` (1367 lines, CONTRACT_VERSION 1.0.0, 42 exports, 11 enums, 5 Protocols,
19 frozen dataclasses, 0 mutable) + `docs/architecture.md`. Imports cleanly under the signull env.
Converged independently with the statistics scout on the add-one p-value, direction symmetrisation
applied to nulls, property-matched nulls, effective-not-nominal size, and dataset-specific universes.

**Design improvement over the stats spec:** the spec's flat N0-N4 null labels are collapsed to two
orthogonal axes — *what is resampled* (`NullType`: RANDOM_GENE_SET | LABEL_PERMUTATION) x *how it is
matched* (`MatchingSpec`). N0 = empty `properties`; N1 = three properties; N3 = N1 plus
`set_level_constraints`. Fewer enum members, same expressive power, explicitly documented.

## Seams found by Fleet at merge (NOT found by either agent) — MUST reach Wave 2
1. **`NullSpec.n_draws` defaults to 1000, but the statistical spec sets default K=10000 with a HARD
   FLOOR of K>=2000 and says refuse to emit a p-value below that.** The contract's default is below
   the spec's own refusal threshold. The nulls builder must reconcile: raise the default to 10000
   and enforce the >=2000 floor, or the tool ships emitting p-values it is specified to refuse.
2. **`MatchingSpec.n_bins` is a single int (default 10), but the statistical spec requires NESTED
   CONDITIONAL bins with a different count per level** — `K_a = clip(|B|/500, 10, 40)` for mean
   expression, then `K_v = 5` for variance, then `K_d = 3` for detection rate. One integer cannot
   express this. The nulls builder must either extend the spec object (coordinating a contract bump)
   or map `n_bins` to the first level and carry the rest in `MatchingSpec.tolerance`/`params`.
   Do not silently flatten the nested scheme to a marginal product grid — the spec rejects that
   explicitly because mean and variance are strongly dependent.

## Risk flagged by the architecture scout — MUST reach Wave 2
`nulls/` owns draw generation plus the p-value estimator ONLY. If the nulls builder also writes a
pipeline/runner it will collide with the Wave 3 integration campaign. Restate in that agent's prompt.

### Agent: prior-art-and-data — **complete**, scope clean
`docs/prior-art-and-data.md` (442 lines) + `scripts/fetch_benchmark.py` (653 lines, verified by execution).

**STRATEGIC FINDING — SigCheck COMPETES.** Bioconductor SigCheck 2.44.0 (Artistic-2.0, in BioC 12
years, build OK) already implements the core feature: `sigCheckRandom()` builds a size-matched
random-signature null and reports an empirical p-value, and it handles **classification as well as
survival**. Fleet verified this independently against the Bioconductor and rdrr.io documentation.
The tool we are building is NOT novel in its core idea. Four differentiators survive and the whole
case now rests on them:
  D1 **Property-matched nulls.** SigCheck and singscore sample uniformly "at random from all
     available features" — unmatched, and the bias runs TOWARD declaring the candidate significant.
     scanpy `score_genes` / Seurat `AddModuleScore` do expression-bin-matched control sets, and
     `nullranges` does covariate matching for intervals, but NOBODY has connected property matching
     to signature significance testing. This is the real gap and the product.
  D2 **Python.** No Python package tests outcome-predictiveness (gseapy, decoupler, pyUCell score
     but do not test).
  D3 **Calibration.** Permuted-label p-value uniformity ships nowhere.
  D4 **Latent-axis / meta-PCNA adjustment.** Offered by no tool in the table.
Also verified by Fleet: SigCheck's `$checkPval` can return exactly 0 ("no random signature performed
as well or better"). The signull add-one estimator refuses to, by construction. Concrete, checkable
superiority claim.
Other verdicts: sigQC **removed from CRAN 2025-12-19** (PORT its metrics, never depend on it);
singscore/camera/nullranges PORT as patterns; GSVA/AUCell/UCell/gseapy/decoupler WRAP as scorers.

**Benchmark chosen: GSE25055** (Hatzis 2011, PMID 21558518), GPL96, 22,283 x 310. Fleet verified the
downloaded data on disk: outcome column is **57 pCR / 249 RD, n=306 usable, prevalence 0.186** —
exactly as reported. Paired validation cohort GSE25065 also fetched. pCR is uncensored and
pathologist-adjudicated; four published predictors' per-sample calls ship in the GEO metadata.

**Fleet's own measurement on the real cohort (for Wave 2 to reproduce):** 200 UNIFORM random
50-probe signatures, mean-z scored, symmetrized AUC -> median 0.548, 95th pct **0.633**, max 0.681.
A candidate must beat ~0.633 to clear the *uniform* null at alpha=0.05 here. The property-matched
null will move that number, and by how much is D1's entire value proposition.

**Agent's own caveats:** GEO characteristics keys are NOT row-aligned across samples (proven false
in GSE20194) — a positional parser silently mislabels outcomes; the script parses `key: value` per
cell. All three candidate cohorts are breast/proliferation-dominated, which is itself the Venet
confound, so they cannot independently validate the meta-PCNA adjustment. DLDA-30 probe list is not
machine-readable and needs manual transcription.

## Contract amendment 1.1.0 (applied by Fleet at merge, so no Wave 2 agent edits types.py)
Both seams above are now CLOSED in `src/signull/types.py`:
- `DEFAULT_DRAWS = 10_000`, `MIN_DRAWS = 2_000` added; `NullSpec.n_draws` now defaults to
  `DEFAULT_DRAWS`. Implementations MUST enforce the floor for gating nulls.
- `MatchingSpec.bins_by_property: Mapping[MatchingProperty,int]` and `MatchingSpec.nested: bool=True`
  added, plus `DEFAULT_BINS_BY_PROPERTY` (VARIANCE=5, DETECTION_RATE=3) and
  `default_mean_expression_bins(n)` = clip(n//500,10,40) — returns 40 for this cohort's 22,283 probes.
Verified: imports clean, 46 exports, frozen-ness intact, `NullSpec().n_draws == 10000 >= MIN_DRAWS`.
**`src/signull/types.py` is FROZEN for Wave 2.** Any further contract change routes through Fleet.

## Wave 2 Results (closed 2026-09-01, second machine)

Wave 2 landed in two parts: the module bodies arrived with commit `b4e7dad`, and this
session closed the remaining gaps, added the package surfaces and wrote the test suite.

### Environment on this machine (Linux, `/media/sergio/Data/Universita/CIBB2026-signull/signull`)
The conda env recorded above is on the **Mac** session root and does not exist here. A
project-local `.venv/` (gitignored) was created from `~/miniconda3` and pinned by
`pyproject.toml`: Python 3.12, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0,
pytest 9.1.1. Same major versions as the relay note, so the pandas-3 / numpy-2 caveats stand.

### What this session added
- `data/resolve.py` — `MatrixIndexResolver` (implements `GeneIdResolver`) plus
  `AnnotationTable`, `load_annotation_table`, `resolve_signature`, `strip_version_suffix`.
  Implements all nine rules of `docs/architecture.md` Sec. 4: matched genes come back in
  **matrix row order** with weights carried along, missing and unmapped are tracked
  separately, ambiguous annotation entries are dropped rather than guessed, the F5 overlap
  floor raises *before* any low-overlap warning, and collapses are enumerated.
- `data/universe.py` — `eligible_background` (Sec. 2.1: sd>0, expression filter with the
  `median` escape route, platform restriction, `AFFX-` controls, optional candidate
  exclusion) and `check_background_floors` (Sec. 8 F4: `|B| >= 2000` and `|B| >= 20 m`,
  raising `BackgroundTooSmallError`).
- `data/matrix.py` — added `load_matrix_tsv`, the missing file-level matrix loader, with a
  sha256 provenance record and a declared (never guessed) orientation.
- `nulls/samplers.py` — `RandomGeneSetNull` and `LabelPermutationNull`, both satisfying the
  `NullModel` protocol, plus `REGISTRY`/`get`. Enforces `MIN_DRAWS` at the point of
  generation (`DrawFloorError`, opt-out for the reduced-resolution supervised path), reuses
  the candidate's weight multiset for signed candidates, records a diagnostic when a draw
  had to reach into neighbouring bins, and implements the `mean_abs_correlation` set-level
  constraint by rejection sampling. Exhaustive permutation enumeration is deliberately NOT
  implemented and the docstring says why: with the F3 cohort floors the smallest reachable
  `C(N, n1)` is `C(30,8) = 5852925`, so the `<= 20000` branch is unreachable dead code.
- `scoring/supervised.py` — `SupervisedModelScorer` (Path B): one sklearn Pipeline fitted
  per fold, gene selection by `|t|` **inside** the training fold, folds derived from
  `CVSpec.seed` and never from the per-draw generator (so fold noise cannot inflate the
  null), out-of-fold probabilities rank-normalised per repeat before averaging.
- Package surfaces: `data/__init__.py`, `nulls/__init__.py`, `scoring/__init__.py`,
  `signull/__init__.py`, plus `REGISTRY`/`get` in `metrics/__init__.py`. All four analysis
  packages now expose the registry the architecture asks for.
- `pyproject.toml` (setuptools, `src` layout, pytest `pythonpath`).
- `tests/conftest.py` + `tests/test_data.py` (28), `tests/test_nulls.py` (32),
  `tests/test_scoring.py` (27). **87 passed in ~2 s.**

### Verified
- Import direction holds: `data`, `nulls`, `scoring`, `metrics` each import `types` and no
  sibling analysis package (grepped, clean).
- `nulls/` contains no evaluation loop, so the Wave-1 collision risk with Wave 3 did not
  materialise.
- End-to-end smoke on a 6000 x 120 synthetic cohort with a planted dominant axis, m = 50
  highest-variance candidate, mean-z scoring, symmetrized AUROC, K = 2000 per null
  (8 s total): observed AUC **0.854**; N0 uniform p = 0.234, N1 matched p = 0.354,
  N2 permutation p = 0.0005. Exactly the Venet configuration — the labels carry real signal,
  and both competitive nulls correctly decline to call the signature special, while the
  self-contained null screams. The three nulls are visibly not interchangeable.

### Notes for Wave 3
1. `resolve_signature` takes an extra `log=` keyword beyond the `GeneIdResolver` protocol
   signature; it is keyword-only with a default, so the protocol still matches.
2. `empirical_p_value` enforces `MIN_DRAWS` by default (`enforce_min_draws=False` opts out)
   and `RandomGeneSetNull`/`LabelPermutationNull` enforce it again at draw time. The
   pipeline should pass the reduced-resolution flag through in exactly one place for the
   supervised path, and the report must render the note.
3. `ScoringMethodName.SSGSEA` has no implementation and is absent from `scoring.REGISTRY`;
   `get` raises rather than substituting. Either implement it or keep the refusal.
4. The T1-T7 acceptance tests of `docs/statistical-design.md` Sec. 7 are Wave 3's
   `tests/test_calibration.py` and are NOT yet written. The smoke run above is not one of
   them.
5. `data/` still has no HDF5 / AnnData loader; only TSV/CSV via `load_matrix_tsv`.

## Continuation State
Next wave: 3
Blocked items: none
Auto-continue: true
