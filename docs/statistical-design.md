# `signull` — Statistical Design Specification

**Status:** normative. An implementer should need no further statistical decisions.
**Scope:** binary patient outcome only. Survival endpoints are out of scope for v1.
**Convention:** where this document says MUST / MUST NOT / DEFAULT, it is a decision, not a suggestion. Open questions are marked **[UNCERTAIN]** and still carry a recommended default.

---

## 0. Notation and inputs

| Symbol | Meaning |
|---|---|
| `X` | expression matrix, `G x N`, log-scale, genes x samples, already normalised |
| `y` | outcome, `y in {0,1}^N`; `n1 = sum(y)`, `n0 = N - n1`, prevalence `pi = n1/N` |
| `S` | candidate signature, a set of gene identifiers; may be signed (`S+`, `S-`) |
| `B` | *eligible background* — the gene pool null sets are drawn from (Sec. 2.1) |
| `m` | `|S ∩ B|`, the **observed** signature size (not the nominal size) |
| `K` | number of null draws |
| `T` | test statistic (Sec. 3.1) |
| `Z` | gene-standardised matrix, `Z[g,j] = (X[g,j] - mean_g) / sd_g` |

**Core invariant (I1).** The scoring function is `score(X, S) -> R^N`. It MUST NOT take `y` as an argument in the default (unsupervised) path. Every null set is pushed through the *identical* function object.

**Core invariant (I2).** The candidate and every null draw are evaluated by the identical statistic, on the identical samples, with the identical adjustment and (if supervised) the identical fold assignment.

---

## 1. Scoring the signature

Scoring is not the subject of the test, but results are scoring-method dependent, so it must be fixed *a priori*.

**S1 — signed signature (DEFAULT when directions are supplied).**
```
z = row-standardise(X over all N samples)     # mean 0, sd 1 per gene
score_j = mean_{g in S+} z[g,j] - mean_{g in S-} z[g,j]
```

**S2 — unsigned signature (DEFAULT when no directions).** Eigengene:
```
Z_S = z[S, :]                       # m x N
u1  = first left singular vector of Z_S (centred over samples)
score = u1' Z_S ; flip sign so corr(score, colMeans(Z_S)) >= 0
```
Rationale: an unsigned arithmetic mean cancels anti-correlated members and systematically *under*-scores heterogeneous signatures while random sets (weakly correlated) are unaffected — a bias against the candidate. The eigengene is `y`-blind, so I1 holds.

**S3 — rank-based (option `--scorer=rank`).** singscore: per-sample gene ranks, score = mean normalised rank of `S+` minus that of `S-` (Foroutan et al. 2018). Use when cross-platform normalisation is untrusted; invariant to any monotone per-sample transform.

The scorer MUST be declared before the run. If the user requests several, all are reported and p-values are BH-adjusted across scorers (Sec. 8, F11).

---

## 2. Null constructions

Four nulls. They answer different questions and are **not** interchangeable.

| Null | Resampled | Held fixed | Hypothesis tested | Question |
|---|---|---|---|---|
| **N0** uniform random gene sets | gene membership, uniform over `B` | `X`, `y`, scorer | `S` is no better than an arbitrary set of `m` measured genes | weak/diagnostic |
| **N1** property-matched random gene sets | gene membership, stratified (Sec. 2.2) | `X`, `y`, scorer, marginal gene properties | `S` is no better than `m` genes with the same expression/variance/detection profile | **"is this gene set special among gene sets?"** — **DEFAULT** |
| **N2** label permutation | `y` | `X`, `S`, scorer | `score(S)` is independent of outcome | **"is there any signal at all?"** — **MANDATORY GATE** |
| **N3** coherence-matched sets | gene membership, matched also on within-set mean \|corr\| | as N1 + set coherence | `S` is no better than an equally *coherent* module of the same size | secondary, reported not gating |
| **N4** annotated-set null | draw real curated gene sets of size in `[0.8m, 1.25m]` | `X`, `y`, scorer | `S` is no better than other real biology of that size | optional, needs MSigDB |

This is the competitive / self-contained distinction of Goeman & Bühlmann (2007): N0/N1/N3/N4 are **competitive** (gene sampling); N2 is **self-contained** (subject sampling). N2 alone inflates with signature size — that is exactly the ">90% of random 100-gene signatures are significant" result of Venet et al. (2011). N1 alone is uninterpretable if there is no signal in the cohort at all.

**Decision — reporting rule.** The headline verdict "this signature is more predictive than size-matched random signatures" requires **both** `p_N2 < alpha` **and** `p_N1 < alpha`. `p_N0` is printed as a diagnostic only; a large gap between `p_N0` and `p_N1` is itself the finding (the signature's advantage was its marginal expression properties, not its identity).

**Decision — N3 is not the default.** Matching on coherence can remove the very property that makes a signature useful (a real pathway *is* coherent). It is reported so the reader can see whether the candidate is "a coherent module" versus "*this particular* coherent module". **[UNCERTAIN]** whether coherence matching over-corrects in practice; default = report, do not gate.

### 2.1 Eligible background `B`

```
B = { g in rows(X)
      : g passes the expression filter
      : g is not in S
      : sd_g > 0
      : g was measurable on the platform the signature was derived from (if declared) }
```
Expression filter (RNA-seq): `median CPM >= 1` OR detected (count>0) in `>= 20%` of samples. Arrays: above the array's declared background in `>= 20%` of samples; if no background call is available, keep all probes and set the detection dimension to constant.

Two traps:
- **Platform restriction.** If the candidate was selected on a 10k-feature platform, `B` MUST be restricted to those 10k. Drawing "random" sets from 20k genes the original authors could never have picked makes the null easier and flatters the candidate.
- **Excluding `S`.** `S` is removed from `B`. Otherwise null draws contain candidate genes and the null drifts toward the candidate.
- Near-duplicate probes MUST be collapsed to one feature per gene at ingest (max-mean probe), or the effective `|B|` is overstated.

### 2.2 Matching scheme (N1) — normative

Per-gene properties, computed on all `N` samples from `X` (all `y`-blind, so no leakage):

- `a_g` = mean log expression
- `v_g` = `log(var_g + eps)`, `eps = 1e-8`
- `d_g` = detection rate = fraction of samples above the platform detection threshold

**Binning: nested (conditional) quantiles, not a product of marginals.** `a` and `v` are strongly dependent (mean–variance trend); a product grid leaves most cells empty.

```
K_a = clip(round(|B| / 500), 10, 40)        # default 20 for a 10k background
K_v = 5
K_d = 3   (or 1 if d is degenerate: sd(d) < 0.01)
```

```
FUNCTION build_bins(B, a, v, d, K_a, K_v, K_d):
    a_bin[g] <- quantile_bin(a over B, K_a)
    FOR each a-bin A:
        v_bin[g in A] <- quantile_bin(v over A, K_v)
        FOR each (a,v)-cell C:
            d_bin[g in C] <- quantile_bin(d over C, K_d)
    RETURN cell[g] = (a_bin, v_bin, d_bin)          # <= 600 cells
```

**Adequacy check and coarsening.** Let `need[c]` = number of candidate genes falling in cell `c`, and `pool[c] = |{g in B : cell[g] = c}|`. Require
```
pool[c] >= C_min(c) := max(50, 10 * need[c])   for every c with need[c] > 0
```
If violated, coarsen in this order — `K_d -> 1`, then `K_v: 5 -> 3 -> 2`, then `K_a` halved (min 5) — re-binning each time, until satisfied or the floor `(K_a,K_v,K_d) = (5,2,1)` is reached. The final configuration MUST be recorded in the report. If the floor is reached and the check still fails, refuse (Sec. 8, F4).

**Drawing one null set.**
```
FUNCTION draw_matched(S, cell, B, rng):
    R <- empty set
    FOR each s in S (in a fixed, seeded order):
        c <- cell[s]
        cand <- pool[c] \ R
        radius <- 0
        WHILE cand is empty:
            radius <- radius + 1
            cand <- union of pool[c'] for all c' with L1(cell-index c', c) == radius, minus R
            IF radius > K_a + K_v + K_d: FAIL "background exhausted"
        R <- R + { uniform_choice(cand, rng) }      # without replacement within this draw
    ASSERT |R| == |S|
    RETURN R
```
Neighbour expansion walks the ordered `(a_bin, v_bin, d_bin)` lattice by L1 distance in bin-index space; ties are resolved by smaller `|Δa_bin|` first, then `|Δv_bin|`, then `|Δd_bin|`. Sampling is without replacement *within* a draw and independent *across* draws.

**Explicitly not matched:** association with `y`. Matching on outcome association would be circular and is forbidden.

**N3 (coherence-matched)** extends `draw_matched` with rejection sampling: compute `rho(S)` = mean pairwise `|Pearson corr|` within `S`; accept a candidate draw `R` only if `|rho(R) - rho(S)| <= 0.02`; after 200 rejections, widen the tolerance by `0.01` and continue; abort at tolerance `0.10` and mark N3 unavailable.

### 2.3 Label permutation (N2)

Permute `y` uniformly at random; `X`, `S`, and the scorer are untouched. The number of distinct labelings is `C(N, n1)`.
- If `C(N, n1) <= 20000`: enumerate exhaustively; `p = r / M` where `r` counts labelings with `T >= T_obs` (the identity labeling is included, so `r >= 1`).
- Otherwise: Monte Carlo with `K_perm` draws and the `+1` formula (Sec. 3.2).

In the unsupervised path, permutation is cheap: the score vector is fixed, so `T` under permutation is a pure function of the ranks of `score` — the entire null can be computed from one sort.

### 2.4 Rotation null (N2-alt, optional)

For very small cohorts the permutation null is granular. A rotation test (ROAST, Wu et al. 2010) replaces label permutation with random rotations in the residual space of the linear model, giving a continuous null while preserving inter-gene correlation. It tests a **self-contained** hypothesis and is therefore a substitute for N2, never for N1. Offer under `--null=rotation`; **not** the default, because it assumes approximate normality of the gene-level residuals which log-CPM data only roughly satisfy.

---

## 3. Test statistic and p-value

### 3.1 Statistic

Primary `T = AUROC(score, y)`.

- **Direction prespecified** (signed signature with a claimed direction): `T = AUC`, one-sided upper tail.
- **Direction not prespecified** (DEFAULT): `T* = max(AUC, 1 - AUC)`. This absorbs the one bit of sign information that would otherwise be fitted on `y`. `T*` MUST be applied to the nulls as well; applying `max()` to the candidate only is a real and easy-to-make leak that inflates significance.

The test is **always one-sided on the upper tail**. "Significantly worse than random" is not a claim this tool makes; it is reported as a *diagnostic flag* (likely inverted signs or a mis-mapped identifier), never as a positive result.

### 3.2 Empirical p-value

```
p_hat = (1 + #{ k in 1..K : T_k >= T_obs }) / (1 + K)
```

**Rationale for `+1`.** The plug-in estimator `r/K` is anti-conservative and can equal 0, which is not a possible p-value for a Monte Carlo test. `(r+1)/(K+1)` is the exactly valid Monte Carlo p-value: under the null it satisfies `P(p_hat <= alpha) <= alpha` for every `K`. (North, Curtis & Sham 2002; Phipson & Smyth 2010, whose title is the rule: permutation p-values should never be zero.)

Ties (`T_k == T_obs`) count as exceedances. With small `n1` the AUC is discrete and ties are common; counting them makes the test conservative, which is the correct direction.

### 3.3 How many draws

- Minimum attainable p is `1/(K+1)`. The tool MUST report `p < 1/(K+1)` rather than a fabricated smaller number.
- Monte Carlo SE `≈ sqrt(p(1-p)/K)`. For 10% relative SE at `p = 0.05`, `K >= (1-p)/(p * 0.01) ≈ 1900`.
- **Defaults:** `K_N1 = K_N0 = 10000`; `K_N2 = 10000` or exhaustive per Sec. 2.3. **Floor: `K >= 2000`;** refuse to emit a p-value below that.
- Supervised path (Sec. 5B) costs `K * R * k` model fits. Default there is `K = 1000` (min p ≈ 1e-3), and the report must state the reduced resolution.
- **Optional** Besag–Clifford sequential stopping (`--sequential`): stop when exceedances reach `h = 20`, report `p = h / k_used`; or at `K_max`. Off by default so runs are bit-reproducible.

### 3.4 What to report alongside p

The p-value alone hides the Starmans (2011) point — the random baseline is dataset-specific and ranges from ~1% to ~40% of random signatures reaching significance. So the report MUST carry:

- `T_obs`, and the null's **median**, IQR, and 95th percentile;
- standardised effect `z = (T_obs - mean(T_null)) / sd(T_null)`;
- the **dataset random-significance rate**: fraction of N1 draws whose own N2 permutation p-value is `< 0.05`. This is the single most informative number in the report — it is the cohort's inflation factor.
- Clopper–Pearson 95% CI on the exceedance proportion `r/K`.

---

## 4. Latent-axis / confounder adjustment

Venet et al. (2011) showed that adjusting for a proliferation metagene abrogated nearly all outcome association of both published and random breast-cancer signatures, and that >50% of that transcriptome correlates with a single axis. Any tool that ignores this will certify noise.

### 4.1 Detection

On `Z` restricted to `B`:
1. SVD; record variance explained by PC1..PC5.
2. **Dominant-axis flag** if `varexp(PC1) >= 0.20` **or** `>= 30%` of background genes have `|corr(g, PC1)| > 0.3`.
3. **Candidate overlap**: `o = mean_{g in S} |corr(g, PC1)|`, converted to a percentile against the same quantity computed on the 10000 N1 draws. Flag if `o` exceeds the 90th percentile.
4. **Proliferation metagene** (`meta-PCNA`, after Venet et al.): rank `B` by Pearson correlation with `PCNA` across the cohort; take the top 1% (minimum 100 genes); metagene = mean of their `z` rows. Fallbacks in order: `MKI67`, then the mean `z` of `HALLMARK_E2F_TARGETS ∩ B`. If none available, mark proliferation adjustment unavailable — do not silently substitute PC1 for it (they are related but not the same object; Dai et al. 2005).

### 4.2 Adjustment

Confounder matrix `C` (columns, each standardised): default `C = [PC1]`; the proliferation-adjusted analysis is a **separate** run with `C = [meta-PCNA]`; `C = [PC1, meta-PCNA]` is available under `--adjust=both`.

Adjusted statistic: **residual-score AUC**.
```
s      = score(X, S)
s_res  = s - C (C'C)^-1 C' s
T_adj  = AUROC(s_res, y)      (or max(., 1-.) if direction not prespecified)
```
Residualise the **score**, not `y`. `C` is estimated from `X` only, so it stays `y`-blind and I1 holds. The identical residualisation is applied to every null draw, so `p_N1_adj` is computed exactly like `p_N1`.

**[UNCERTAIN]** A covariate-stratified AUC (probability a case outranks a control *within the same `C` stratum*, quintiles of `C`) is more principled than residualisation and is the recommended future replacement. Residual-score AUC is the default because it is a drop-in transformation of the score object and therefore trivially identical across candidate and nulls.

### 4.3 Reporting rule — both, always

Report `(T, p_N1, p_N2)` **unadjusted** and **adjusted**, side by side. Never report only one.

| unadjusted `p_N1` | adjusted `p_N1` | verdict string |
|---|---|---|
| `>= alpha` | any | *not distinguishable from size-matched random signatures* |
| `< alpha` | `< alpha` | *outperforms matched random signatures, independent of the dominant axis* |
| `< alpha` | `>= alpha` | *not distinguishable from the cohort's dominant expression axis* |

**Risk of over-adjustment.** If the biology of the outcome *is* proliferation, regressing out the proliferation metagene deletes real signal, and the third row above is a false negative. Therefore the report MUST also print `corr(PC1, y)` and `AUROC(meta-PCNA, y)` on their own. If the confounder alone achieves `AUROC >= 0.65`, print: *"the adjustment target is itself strongly prognostic here; the adjusted analysis is a conservative lower bound, not the headline."* Also check `PC1` against declared technical covariates (batch, site, platform, RIN) — if PC1 is technical, adjustment is a correction; if PC1 is biological and outcome-linked, it is a subtraction of signal.

---

## 5. Cross-validation and leakage

### Path A — unsupervised score (DEFAULT)

Nothing is fitted on `y`, so **no cross-validation is required for validity**. The permutation null is exact and the competitive null is exact. Do not add CV; it only adds variance and invites fold-dependent leaks.

Leakage traps in Path A, all of which have bitten published analyses:
1. **Sign fitting.** Choosing the score's direction by looking at `y` is a 1-bit fit. Either prespecify the direction or use `T* = max(AUC, 1-AUC)` **for candidate and nulls alike**.
2. **`y`-aware background filtering.** The expression filter and the bins MUST be computed without reference to `y`.
3. **Threshold tuning.** Report threshold-free metrics only. Never pick a cut-point on `y` and report accuracy/sensitivity from it.
4. **Scorer shopping.** See Sec. 8, F11.
5. **In-sample signatures.** If `S` was derived on this same cohort, no null construction can rescue the analysis (Michiels et al. 2005: signature membership is highly unstable across resampled training sets, and in-sample performance is optimistic). See Sec. 8, F2.

### Path B — supervised score (`--scorer=logistic|lasso|centroid`)

CV is **mandatory** and everything that touches `y` goes inside the loop.

```
folds <- repeated stratified k-fold, k = 5, R = 20 repeats, seed fixed ONCE
        (leave-pair-out if n1 < 15)
FOR each repeat r, each fold f:
    train = X[:, not f], y[not f]
    fit scaler, any gene selection, any hyperparameter tuning   <- ALL on train only
    (if adjusting) estimate PC1 / meta-PCNA on train only, project test
    predict on fold f
pool out-of-fold predictions within a repeat -> AUC_r
T = mean_r AUC_r
```

Rules:
- **Same folds for candidate and every null draw.** Fix the fold seed globally. Fold randomness must not contribute to the null's spread, or the null is inflated and the test loses power.
- **Every null signature is refit and re-cross-validated.** Reusing the candidate's fitted model with swapped gene sets is not a null.
- **Permutation in Path B** permutes `y` once per replicate and re-runs the *entire* CV including all fitting. Permuting inside the loop, or after fitting, invalidates the test.
- **Nested CV** whenever a hyperparameter is tuned; a single CV loop that both tunes and reports is biased downward in error (Varma & Simon 2006), and gene selection outside the loop is the classic catastrophic version (Ambroise & McLachlan 2002).
- **PC1/metagene estimation** must move inside the fold in Path B. In Path A it may use all samples because it never sees `y`.

---

## 6. Metrics

**Default primary: AUROC.** Rationale: prevalence-invariant, so the candidate and the null draws are on a common scale and the chance value is exactly 0.5 — which is what makes "better than random" legible. It is a U-statistic of ranks, so the permutation null is computable in closed form from a single sort.

**Always also reported: average precision (AP).** Use the step-wise estimator `AP = sum_i (R_i - R_{i-1}) * P_i`; never the trapezoidal interpolation. Report alongside it the baseline `AP_0 = pi` and the normalised `AP_norm = (AP - pi) / (1 - pi)`.

**Under imbalance** (`pi <= 0.10` or `n1 < 20`): report both. Make AP the headline **only** when the stated use is ranking/triage of a top-k list. Do not accept the folk rule that AUPRC is simply better under imbalance: AUPRC and AUROC differ in *what* they reward, and AUPRC can favour gains in higher-prevalence subgroups (McDermott et al. 2024). Saito & Rehmsmeier (2015) remains the case for reporting PR curves; it is not a case for discarding AUROC.

Because the null is empirical and uses the same metric, **the p-value is valid under either metric.** Metric choice affects power and interpretation, not validity.

**Cohort size floors.** Require `n1 >= 8`, `n0 >= 8`, `N >= 30` to emit a p-value. Below that the AUC takes too few distinct values for the permutation null to be usable; emit the null distribution as a descriptive figure and refuse the verdict.

**Confidence intervals.**
- AUROC, fixed score: DeLong CI (DeLong et al. 1988).
- AP: stratified bootstrap percentile CI, 2000 resamples, resampling cases and controls separately.
- CV-AUC (Path B): influence-curve CI of LeDell et al. (2015). A naive CI over per-fold AUCs is wrong — folds are dependent.
- Do **not** bootstrap a CI for the p-value; report its Monte Carlo SE and the Clopper–Pearson interval on `r/K`.

---

## 7. Calibration acceptance tests

These are the test suite. All use fixed seeds. A release that fails T1, T2, or T4 is broken.

**T1 — Uniformity of `p_N2` under permuted labels.** 1000 independent permutations of `y`; for each, run the full pipeline with `K = 999`. Assert: KS test against `U(0,1)` gives `p > 0.01`, and the empirical `P(p <= 0.05)` lies in `[0.033, 0.070]`.

**T2 — Uniformity of `p_N1` and discrimination between nulls.** Two arms on a real cohort:
- (a) candidate drawn *by the matched sampler itself*: `p_N1` must be uniform (KS `p > 0.01`).
- (b) candidate = the `m` highest-variance genes: `p_N0` must be strongly non-uniform and small (`P(p_N0 <= 0.05) > 0.5`), while `p_N1` must remain approximately uniform (`P(p_N1 <= 0.05)` in `[0.02, 0.12]`).
Arm (b) is the central acceptance test of the whole design: it demonstrates that the matched null removes the marginal-property advantage that the uniform null credits to the candidate.

**T3 — Power against a planted signal.** Synthetic cohort `G = 10000`, `N = 200`, `pi = 0.3`. Latent factor `f ~ N(0,1)`; `m0 = 50` genes get `X[g,:] += delta * f`; `logit P(y=1) = a + b*f`. Realistic correlation is induced by 20 additional shared latent factors. Sweep `b` so the true score AUC is `{0.60, 0.65, 0.70, 0.75, 0.80}`. Assert: power at true AUC `0.70` exceeds `0.80` at `alpha = 0.05` for N1; assert monotone increase in power across the sweep. Store the full curve as a regression baseline with tolerances rather than a single hard threshold, to avoid flaky tests.

**T4 — Signature-size growth (the >100-gene effect).** On a cohort with a dominant axis, for `m in {5, 10, 25, 50, 100, 200, 500}` draw 200 uniform-random signatures each. Assert:
- (i) the fraction with `p_N2 < 0.05` **increases** with `m` and exceeds `0.5` by `m = 200` — reproducing Venet et al. (2011);
- (ii) the fraction with `p_N1 < 0.05` stays in `[0.02, 0.12]` for **every** `m`.
(ii) is the correctness claim of the tool: the competitive matched p-value must not inflate with size the way the self-contained permutation p-value does.

**T5 — Dataset specificity.** Across `>= 3` cohorts, assert the N1 null median AUC differs materially between them, and that the reported dataset random-significance rate varies. Structurally: the cached null object carries a hash of `(X, y, scorer, adjustment, K, seed)`; evaluating against a different matrix MUST recompute. Assert a cache miss.

**T6 — Invariance and determinism.** `p` unchanged under permutation of gene order and of sample order; unchanged under a monotone per-sample transform iff `--scorer=rank`; bit-identical across two runs with the same seed; unchanged by the number of worker threads.

**T7 — Degenerate inputs.** Defined behaviour for: constant genes; single-class `y`; `m > |B|`; duplicate identifiers in `S`; `S` genes absent from `X`; `NaN` in `X`; `K` below the floor.

---

## 8. Failure modes and mandatory refusals

**F1 — Candidate overlaps the dominant axis.** Detected by Sec. 4.1(3). Do not refuse to compute; refuse to emit an unqualified verdict. The report carries the flag, both adjusted and unadjusted numbers, and the verdict string from the Sec. 4.3 table.

**F2 — Signature derived on this cohort.** The tool asks for provenance (`independent` / `same-cohort` / `undeclared`). `same-cohort` and `undeclared` both produce the label *in-sample, descriptive only*; the word "validation" MUST NOT appear in the output. Undeclared defaults to the pessimistic reading.

**F3 — Cohort too small.** `n1 < 8`, `n0 < 8`, or `N < 30` -> refuse the p-value (Sec. 6).

**F4 — Background too small.** `|B| < 2000` or `|B| < 20*m`, or the bin adequacy check fails at the coarsening floor -> refuse the competitive null (N0/N1/N3). N2 may still run.

**F5 — Partial signature coverage.** If `|S ∩ B| / |S| < 0.70` -> refuse. If in `[0.70, 1.0)` -> proceed on the observed subset and set the null size to `m = |S ∩ B|`, **not** the nominal `|S|`. Drawing nominal-size nulls against an observed-size candidate is a silent size mismatch and biases the test. Report both numbers.

**F6 — Outcome confounded with batch.** For each declared technical covariate, test association with `y` (chi-square / Cramér's V). If `p < 0.01` or `V > 0.3` -> refuse the verdict and report the confounding instead.

**F7 — Cross-dataset null reuse.** Structurally impossible per T5; if a cache hash mismatch is detected, error out rather than reuse. Starmans et al. (2011) is the reason: the random-signature significance rate ranges from ~1% to ~40% across datasets, so an arbitrary or borrowed cut-off is meaningless.

**F8 — p-value floor.** Never print a p smaller than `1/(K+1)`; print `p < 1/(K+1)`.

**F9 — Multiple candidates.** If `k > 1` signatures are submitted in one call, apply Benjamini–Hochberg across them and report raw and adjusted p-values together.

**F10 — "Worse than random."** Reported as a data-quality flag (inverted signs, mis-mapped identifiers), never as a positive finding.

**F11 — Scorer / adjustment shopping.** The tool MUST NOT select the scorer, adjustment, or metric that yields the smallest p. Any multi-configuration run reports every configuration and BH-adjusts across them.

**F12 — Silent imputation.** Missing values are reported, never mean-imputed without a flag; a gene with `>10%` missing is dropped from `B` and from `S` (counting against F5 coverage).

---

## 9. Report schema (minimum fields)

```
cohort:      N, n1, n0, pi, |B|, platform, provenance
signature:   nominal size, observed size m, coverage, signed?
scorer:      id, parameters
bins:        (K_a, K_v, K_d) after coarsening, min cell pool
axis:        varexp PC1..PC5, dominant-axis flag,
             candidate overlap percentile,
             AUROC(PC1, y), AUROC(meta-PCNA, y)
results:     for each of {unadjusted, PC1-adjusted, proliferation-adjusted}:
               T_obs, AUROC CI, AP, AP_norm, AP CI,
               p_N2 (+ enumerated?), p_N0, p_N1, p_N3,
               null median / IQR / p95, z, Clopper-Pearson CI on r/K
             dataset random-significance rate
verdict:     one string from the Sec. 4.3 table, plus any active F-flags
repro:       seed, K per null, versions, hash of (X, y, scorer, adjustment)
```

---

## 10. References (all verified to resolve)

1. Venet D, Dumont JE, Detours V. Most random gene expression signatures are significantly associated with breast cancer outcome. *PLoS Comput Biol* 2011;7(10):e1002240. https://doi.org/10.1371/journal.pcbi.1002240
2. Starmans MHW, et al. A simple but highly effective approach to evaluate the prognostic performance of gene expression signatures. *PLoS ONE* 2011;6(12):e28320. https://doi.org/10.1371/journal.pone.0028320
3. Michiels S, Koscielny S, Hill C. Prediction of cancer outcome with microarrays: a multiple random validation strategy. *Lancet* 2005;365(9458):488-92. https://doi.org/10.1016/S0140-6736(05)17866-0
4. Beck AH, Knoblauch NW, Hefti MM, et al. Significance analysis of prognostic signatures. *PLoS Comput Biol* 2013;9(1):e1002875. https://doi.org/10.1371/journal.pcbi.1002875
5. Stark R, Norden J. SigCheck: check a gene signature's prognostic performance against random signatures, known signatures, and permuted data/metadata. Bioconductor. https://bioconductor.org/packages/release/bioc/html/SigCheck.html
6. Goeman JJ, Bühlmann P. Analyzing gene expression data in terms of gene sets: methodological issues. *Bioinformatics* 2007;23(8):980-7. https://doi.org/10.1093/bioinformatics/btm051
7. Efron B, Tibshirani R. On testing the significance of sets of genes. *Ann Appl Stat* 2007;1(1):107-29. https://doi.org/10.1214/07-AOAS101
8. Wu D, Smyth GK. Camera: a competitive gene set test accounting for inter-gene correlation. *Nucleic Acids Res* 2012;40(17):e133. https://doi.org/10.1093/nar/gks461
9. Wu D, Lim E, Vaillant F, et al. ROAST: rotation gene set tests for complex microarray experiments. *Bioinformatics* 2010;26(17):2176-82. https://doi.org/10.1093/bioinformatics/btq401
10. Tirosh I, Izar B, Prakadan SM, et al. Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq. *Science* 2016;352(6282):189-96. https://doi.org/10.1126/science.aad0501 (expression-bin-matched control gene sets)
11. Foroutan M, Bhuva DD, Lyu R, et al. Single sample scoring of molecular phenotypes. *BMC Bioinformatics* 2018;19:404. https://doi.org/10.1186/s12859-018-2435-4
12. Phipson B, Smyth GK. Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Stat Appl Genet Mol Biol* 2010;9(1):Article 39. https://doi.org/10.2202/1544-6115.1585
13. North BV, Curtis D, Sham PC. A note on the calculation of empirical P values from Monte Carlo procedures. *Am J Hum Genet* 2002;71(2):439-41. https://doi.org/10.1086/341527
14. Dai H, van't Veer L, Lamb J, et al. A cell proliferation signature is a marker of extremely poor outcome in a subpopulation of breast cancer patients. *Cancer Res* 2005;65(10):4059-66. https://doi.org/10.1158/0008-5472.CAN-04-3953
15. Leek JT, Storey JD. Capturing heterogeneity in gene expression studies by surrogate variable analysis. *PLoS Genet* 2007;3(9):e161. https://doi.org/10.1371/journal.pgen.0030161
16. Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. *PNAS* 2002;99(10):6562-6. https://doi.org/10.1073/pnas.102102699
17. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics* 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
18. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
19. McDermott MBA, Hansen LH, Zhang H, et al. A closer look at AUROC and AUPRC under class imbalance. *NeurIPS* 2024. https://arxiv.org/abs/2401.06091
20. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 1988;44(3):837-45. https://doi.org/10.2307/2531595
21. LeDell E, Petersen M, van der Laan M. Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates. *Electron J Stat* 2015;9(1):1583-607. https://doi.org/10.1214/15-EJS1035
