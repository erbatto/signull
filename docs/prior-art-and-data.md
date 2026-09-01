# Prior art and benchmark data

Research scout output, Wave 1, campaign 2. Everything here was checked against the live
source on **2026-09-01**; versions, licences, sample counts and class balances are
observed values, not recalled ones. Anything I could not verify is flagged as such.

Scope of the tool this feeds: given a candidate gene signature, a cohort expression
matrix and a **binary** patient outcome, decide whether the signature is more predictive
than same-size signatures drawn at random from the same dataset.

---

## Part 1 — Prior art

### 1.1 Verdict summary

| Tool | Lang | Version (2026-09-01) | Maintained? | Licence | Binary outcome? | Random-signature null? | Property-matched null? | Verdict |
|---|---|---|---|---|---|---|---|---|
| **SigCheck** | R / Bioconductor | 2.44.0 (BioC 3.23) | Yes — build OK, in BioC 12 yrs | Artistic-2.0 | **Yes** (classification + survival) | **Yes** — the core feature | No — uniform sample from all features | **COMPETE** |
| **sigQC** | R / CRAN | 0.1.24 | **No — removed from CRAN 2025-12-19** | GPL (>= 3) | Indirectly | Yes — random-set negative controls for its QC metrics | No | **PORT** (metric ideas only) |
| **singscore** | R / Bioconductor | 1.32.0 | Yes | GPL-3 | No — per-sample enrichment | Yes — `generateNull()`, size-matched | No | **PORT** |
| **GSVA / ssGSEA** | R / Bioconductor | 2.6.6 | Yes | Artistic-2.0 | No — scoring only | No | No | **WRAP** (as a scorer) |
| **AUCell** | R / Bioconductor | 1.34.0 | Yes | GPL-3 | No — scoring only | No | No | **IGNORE** (for the null) |
| **UCell** | R / Bioconductor | 2.16.0 | Yes | GPL-3 + file LICENSE | No — scoring only | No | No | **IGNORE** (for the null) |
| **pyUCell** | Python / PyPI | 0.7.3 | Yes | MIT | No — scoring only | No | No | **WRAP** (as a scorer) |
| **scanpy `tl.score_genes`** | Python | scanpy (scverse) | Yes | BSD-3 | No — scoring only | No | **Yes** — expression-bin-matched control genes | **PORT** (the matching scheme) |
| **Seurat `AddModuleScore`** | R / CRAN | Seurat 5.x | Yes | MIT | No — scoring only | No | **Yes** — same bin-matched scheme | **PORT** (the matching scheme) |
| **gseapy** | Python / PyPI | 1.3.1 (2026-07-26) | Yes — active | BSD-3-Clause | No — ranked-list enrichment | Gene-label / phenotype permutation, not size-matched sets | No | **WRAP** (as a scorer) |
| **decoupler** | Python / PyPI | 2.2.0 (2026-07-11) | Yes — scverse | BSD-3-Clause | No — activity inference | Permutation normalisation in some methods | No | **WRAP** (as a scorer) |
| **limma `camera` / `roast`** | R / Bioconductor | limma (current) | Yes | GPL (>= 2) | Via a design matrix | Competitive test with inter-gene-correlation adjustment | Partially (variance inflation, not covariate matching) | **PORT** (the VIF idea) |
| **nullranges** | R / Bioconductor | 1.18.0 | Yes | GPL-3 | n/a — genomic intervals | n/a | **Yes** — covariate-matched null *ranges* | **PORT** (the API design) |
| **genefu** | R / Bioconductor | 2.44.0 | Yes | Artistic-2.0 | n/a | No | No | **Use as a signature source** |

**No Python package was found that answers the actual question** — "is this signature
more predictive of a binary outcome than same-size random signatures from this dataset".
The Python ecosystem covers *scoring* thoroughly and *null construction* barely.

### 1.2 The honest finding, stated plainly

**The random-signature benchmark is not a new idea and one tool already implements it
well.** SigCheck (Rory Stark, Bioconductor since 3.0) is a direct, live, maintained
competitor. `sigCheckRandom()` does exactly what signull's headline feature does:
sample `iterations` signatures of the same length as the candidate, score each, and
report an empirical p-value from the resulting null. `sigCheckPermuted()` gives the
label-permutation null. `sigCheckKnown()` compares against curated signature
collections. It handles **binary classification outcomes**, not just survival. This
space is genuinely served, and any pitch for signull that pretends otherwise is wrong.

So the case for signull is narrow, and it has to be argued on these four points, not on
novelty:

1. **The null is misspecified in every existing implementation.** SigCheck's
   `sigCheckRandom` samples uniformly from all features. singscore's `generateNull`
   samples size-matched sets, also uniformly. Neither matches the candidate on
   expression level, detection rate or variance. Since a random set of highly-expressed,
   high-variance probes is a systematically better predictor than a random set of
   low-expressed ones, an unmatched null is biased *toward* declaring the candidate
   significant — the exact direction of error that motivated the field's concern.
   Interestingly, the machinery to fix this already exists in a neighbouring corner of
   the ecosystem: `scanpy.tl.score_genes` / Seurat `AddModuleScore` bin all genes by mean
   expression and draw control genes from the *same bin*. That is a property-matched
   null gene set — it is just used for score centering, never for significance testing.
   `nullranges` does the same thing rigorously (covariate matching + bootstrapping) for
   genomic intervals. **Nobody has connected property matching to signature
   significance.** That is the defensible gap.
2. **Python.** The gap is real and confirmed by search: SigCheck, sigQC and singscore
   are R-only. Everything Python-side (gseapy, decoupler, pyUCell, scanpy) scores gene
   sets but does not answer the outcome-predictiveness question.
3. **Calibration as a first-class deliverable.** None of the above ship a
   permuted-label uniformity check. The shared-context acceptance test — p-values
   approximately uniform under a true null — should be a test in the repo, not a claim.
4. **The proliferation / dominant-latent-axis confound.** Venet 2011 showed that
   adjusting for a meta-PCNA proliferation metagene abrogated almost all outcome
   association of both published *and* random breast-cancer signatures. No tool in the
   table offers that adjustment as an option. This is arguably the most scientifically
   valuable differentiator and the one most likely to change a user's conclusion.

If those four are dropped, signull is a Python rewrite of SigCheck and should not be
built.

### 1.3 Detailed entries

#### SigCheck — **COMPETE**
- <https://bioconductor.org/packages/release/bioc/html/SigCheck.html>
- R / Bioconductor 3.23, version **2.44.0**, licence **Artistic-2.0**, maintainer Rory
  Stark. Build status **OK** on the release builder (checked 2026-09-01). In
  Bioconductor for 12 years.
- Depends: MLInterfaces, Biobase, e1071, BiocParallel, survival.
- Takes an `ExpressionSet` + a signature. Four checks: `sigCheckRandom` (size-matched
  random signatures, uniform over all features), `sigCheckKnown` (against other
  signatures), `sigCheckPermuted` (permuted data and/or metadata), `sigCheckAll`.
  `iterations` sets null size; the p-value is the empirical rank of the candidate.
- **Binary outcomes: yes.** Classification is a first-class task alongside survival.
- Verdict **COMPETE**. This is the benchmark to beat and the tool signull must be
  validated against on the same cohort. Where results disagree, signull owes an
  explanation. Its interface (one object in, a set of named checks out) is worth copying.

#### sigQC — **PORT** (ideas only; do not depend on it)
- <https://cran.r-project.org/package=sigQC>, source <https://github.com/andrewdhawan/sigQC>
- R, last release **0.1.24 (2024-08-18)**, licence **GPL (>= 3)**, maintainer Andrew Dhawan.
- **Removed from CRAN on 2025-12-19**, "as requires archived package 'biclust'". Only
  reachable from the CRAN Archive. Treat as unmaintained.
- Not a significance tool: it is a QC battery (expression level, variability,
  autocorrelation, standardisation/scoring-method comparison) rendered as plots. It
  *does* include random gene sets of matched length as negative controls to reveal each
  metric's null.
- **Binary outcomes: only indirectly.**
- Verdict **PORT**: its QC metrics (does the signature have enough expression? enough
  variance? is it internally coherent?) are good pre-flight checks for signull and are
  cheap to reimplement. Do not take a dependency on an archived package.

#### singscore — **PORT**
- <https://bioconductor.org/packages/release/bioc/html/singscore.html> — 1.32.0, GPL-3.
- Rank-based single-sample scoring. `generateNull()` builds size-matched random gene
  sets, scores them per sample, and returns an n x B matrix used for per-sample
  empirical p-values.
- **Important distinction:** this null asks "is this gene set enriched *in this
  sample*". signull asks "is this gene set predictive *of an outcome across samples*".
  Same machinery, different question, non-interchangeable conclusions. Worth stating in
  signull's docs so users do not conflate them.
- Verdict **PORT** the null-generation and parallelisation pattern.

#### GSVA / ssGSEA, AUCell, UCell / pyUCell — **WRAP** or **IGNORE**
- GSVA 2.6.6 (Artistic-2.0), AUCell 1.34.0 (GPL-3), UCell 2.16.0 (GPL-3),
  pyUCell 0.7.3 (MIT, active).
- All are *scorers*: gene set -> per-sample score. None construct an outcome-association
  null. They matter to signull only because the shared context is explicit that **the
  scoring method must be identical for the candidate and the null**, and that results
  are scoring-method dependent. So signull needs several scorers, and pyUCell is the
  Python-native way to offer a rank-based one without reimplementing it.
- Verdict **WRAP** pyUCell (optional dependency), **IGNORE** the R ones.

#### scanpy `tl.score_genes` / Seurat `AddModuleScore` — **PORT** (highest-value port)
- <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html>
- Bin all genes by mean expression; for each signature gene, draw control genes from the
  *same* expression bin; subtract the control mean. This is **property-matched null gene
  set construction**, already implemented and battle-tested in Python.
- It is used for score centering, never for significance. Lifting the binning scheme and
  pointing it at the significance question is precisely signull's differentiator.
- Verdict **PORT**. Extend the matching from mean expression alone to
  (mean expression, variance, detection rate) as the shared context requires.

#### gseapy, decoupler — **WRAP**
- gseapy 1.3.1 (BSD-3, released 2026-07-26), decoupler 2.2.0 (BSD-3, scverse,
  released 2026-07-11). Both actively maintained Python packages.
- Their permutation nulls are about enrichment in a *ranked gene list* (gene-label or
  phenotype permutation), not about the outcome-predictiveness of a size-matched
  signature. Different null, different claim.
- Verdict **WRAP** as optional scoring backends (ssGSEA in particular).

#### limma `camera` / `roast`, and `nullranges` — **PORT** (design patterns)
- `camera` is a competitive gene set test that explicitly corrects for **inter-gene
  correlation** via a variance inflation factor. Random signatures drawn from a
  correlated transcriptome are not independent draws; ignoring this inflates
  significance. signull's null needs the same correction or an empirical equivalent.
- `nullranges` 1.18.0 (GPL-3) generates **covariate-matched null genomic ranges**
  (`matchRanges`) and bootstrapped block nulls. It is the closest thing in all of
  Bioconductor to a principled property-matched null, and its API design — declare the
  covariates to match on, get a matched null set back, inspect the match quality — is a
  good template for signull's null engine.

#### genefu — signature source, not a competitor
- 2.44.0, Artistic-2.0. Ships the published breast-cancer signatures as data objects
  with EntrezGene IDs and, for several, Affymetrix HG-U133A probe IDs: `sig.ggi`,
  `sig.gene70`, `sig.gene76`, `sig.genius`, `sig.oncotypedx`, `sig.pik3cags`,
  `sig.tamr13`, `sig.endoPredict`, `pam50`, `scmgene`/`scmod1`/`scmod2`, `claudinLow`.
- Verified sizes: **`sig.ggi` = 128 HG-U133A probesets / 97 unique genes**;
  **`sig.gene70` = 70 Agilent probes / 56 unique genes**.
- Use it (one-off, in R) to export gene lists to TSV for signull's test fixtures. It is
  a data source; it does no null testing.

### 1.4 The two foundational papers (methods, not software)

- **Venet, Dumont & Detours 2011**, *Most random gene expression signatures are
  significantly associated with breast cancer outcome*, PLoS Comput Biol 7:e1002240 —
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002240>.
  Supplementary files all resolve (verified): **Dataset S1** (`.s001`, ZIP, ~87 MB —
  script and data), **Table S1** (`.s002`, PDF — the meta-PCNA signature), **Text S1**
  (`.s003`, PDF, 5.3 MB — supplementary information, containing the gene lists for the
  47 published signatures and the three "meaningless" control signatures).
  **Caveat, verified:** Venet analysed the **NKI (van de Vijver, n=295)** and **Loi**
  cohorts with **survival** endpoints — *not* the pCR cohorts recommended below. Their
  numbers do not transfer directly to a binary-outcome benchmark; do not quote them as
  expected results on GSE25055.
  **Caveat 2:** the gene lists are in a PDF, not a machine-readable table. Extracting
  the three meaningless signatures requires either parsing Text S1 by hand or unpacking
  the 87 MB Dataset S1. Budget for this; it is not a five-minute job.
- **Starmans et al. 2011**, *A Simple but Highly Effective Approach to Evaluate the
  Prognostic Performance of Gene Expression Signatures*, PLoS ONE 6:e28320 —
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0028320>.
  The source of the "significance rate of random signatures varies ~1–40% by dataset"
  finding that forces the null to be dataset-specific.

---

## Part 2 — Benchmark datasets

### 2.1 Selection criteria applied

Every candidate had to clear all of: downloadable with **no credentials and no data
access application**; a **documented binary** clinical outcome present in the
downloadable file itself (not only in a paper supplement); **n >= 100**; a real,
resolvable accession. I additionally preferred cohorts with a **published signature
already applicable to the same platform**, so the tool can be exercised on a real
candidate rather than a toy.

All three recommendations are on **GPL96 (Affymetrix HG-U133A, 22,283 probes)**. That is
deliberate, not lazy: holding the platform fixed means a difference in the random-signature
null between cohorts is a *dataset* effect (Starmans' finding) and not a platform
artefact. It also means one probe-ID namespace for all signature fixtures.

### 2.2 D1 — **GSE25055** (first choice) with **GSE25065** as paired validation

| Field | GSE25055 (discovery) | GSE25065 (validation) |
|---|---|---|
| Accession | [GSE25055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25055) | [GSE25065](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25065) |
| Platform | GPL96, Affymetrix HG-U133A | GPL96, Affymetrix HG-U133A |
| Probes x samples | **22,283 x 310** | **22,283 x 198** |
| Outcome field | `pathologic_response_pcr_rd` | `pathologic_response_pcr_rd` |
| Outcome definition | pathologic complete response (**pCR**) vs residual disease (**RD**) after neoadjuvant taxane–anthracycline chemotherapy, assessed at surgery | identical |
| Class balance | **57 pCR / 249 RD / 4 NA** -> n=306 usable, **18.6% positive** | **42 pCR / 140 RD / 16 NA** -> n=182 usable, **23.1% positive** |
| Matrix format | GEO series matrix, `.txt.gz`, probes x samples, **already log2** (median 8.12, IQR 6.57–9.39, max 17.36), no missing values | same |
| Download size | **37,718,581 bytes** (~36 MB gz) | **24,194,802 bytes** (~23 MB gz) |
| sha256 (gz) | `9f8a94a9226f38d16380d776ab007c0453c3d31c9915d69562d5854baf3d6777` | `adac6710a8452be19e0f73070668b186a282ebeb34f9aaec09c884906034f6e9` |
| Last GEO update | Nov 02 2022 | Nov 02 2022 |
| Citation | Hatzis et al. 2011, JAMA 305:1873, PMID [21558518](https://pubmed.ncbi.nlm.nih.gov/21558518/) | same study |

Direct URL (no credentials, plain HTTPS, verified 200):
`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25055/matrix/GSE25055_series_matrix.txt.gz`

**Why this is first choice.**
- pCR/RD is the cleanest binary endpoint in the whole outcome-signature literature: a
  pathologist's adjudication at a fixed time point, no censoring, no follow-up
  truncation, no arbitrary dichotomisation of a survival curve. For a tool whose scope
  is explicitly binary, this removes an entire class of definitional argument.
- The class imbalance (18.6% positive) is realistic and forces the tool to report
  **average precision alongside AUC** rather than letting AUC alone flatter it.
- A **paired, independent validation cohort of the same design and platform**
  (GSE25065) exists. Starmans' point — that the random-signature significance rate is
  dataset-specific — becomes directly demonstrable rather than merely cited.
- **The GEO metadata already carries the calls of four published predictors**, per
  sample: `dlda30_prediction`, `ggi_class`, `pam50_class`, `set_class`, plus
  `chemosensitivity_prediction` and `rcb_0_i_prediction`. This is unusual and very
  valuable: signull's verdict on a signature can be sanity-checked against the original
  authors' own classifier calls without reimplementing their classifiers.
- Rich covariates for confounder work: `er_status_ihc`, `pr_status_ihc`,
  `her2_status`, `grade`, `clinical_t_stage`, `clinical_nodal_status`, `age_years`.
- Survival fields (`drfs_1_event_0_censored`, `drfs_even_time_years`) are present too,
  so the same cohort supports a later time-to-event extension without a new download.

**Preprocessing gotchas (all observed, not assumed).**
1. **Already log2.** Values range 0.37–17.36 with median 8.12. Do **not** log again. The
   header says MAS5 with global scaling to a trimmed mean of 600, which would be
   linear-scale; the deposited matrix has been log-transformed on top of that. Trust the
   values, not the protocol text. `fetch_benchmark.py` warns if max > 40.
2. **4 samples have `pathologic_response_pcr_rd: NA`** (16 in GSE25065). They must be
   dropped, and the drop must be recorded — n=306, not 310.
3. **`grade` uses `4=Indeterminate`** as a category, not a grade. Naive numeric coercion
   silently creates a fictitious grade 4.
4. **Both cohorts are HER2-negative by design** and all patients received the same
   regimen. Findings are conditional on that; this is a homogeneous treated cohort, not a
   general breast-cancer population.
5. **GSE25055 and GSE25065 are SubSeries of GSE25066.** Pulling GSE25066 gets all 508
   samples mixed together. Fetch the SubSeries individually to keep discovery and
   validation separate — the whole point of having two.
6. **GEO characteristics keys are `key: value` strings inside tab-separated cells and
   are not guaranteed to occupy the same row index for every sample.** They happen to be
   aligned in GSE25055/GSE25065 but are *not* in GSE20194 (see below). Parse per cell,
   never by row position. The fetch script does this unconditionally.
7. Probe IDs are Affymetrix probe set IDs (`1007_s_at`), including 68 `AFFX-` control
   probes. Decide explicitly whether control probes are eligible for the random null —
   they are not genes, and leaving them in the sampling pool quietly contaminates it.
8. No gene symbols in the matrix. Signature gene lists given as symbols need mapping via
   the GPL96 annotation; probe-level signatures (GGI, DLDA-30) avoid the problem entirely
   and are preferable for the first end-to-end test.

**Signatures to test on it.**

| Role | Signature | Size | Availability | Notes |
|---|---|---|---|---|
| Real candidate, strong prior | **GGI** (Genomic Grade Index, Sotiriou et al. 2006) | 128 HG-U133A probesets / 97 genes | `genefu::sig.ggi` — already HG-U133A probe IDs, **no mapping needed** | Best first candidate. Per-sample `ggi_class` calls are in the GEO metadata for cross-checking. Note it is essentially a proliferation signature, so it doubles as the test case for the meta-PCNA confound. |
| Real candidate, purpose-built for *this* endpoint | **DLDA-30** (Hess et al. 2006, JCO 24:4236, PMID [16896004](https://pubmed.ncbi.nlm.nih.gov/16896004/)) | 30 probe sets | Gene list is in the paper's supplement — **not machine-readable; must be transcribed once**. Per-sample `dlda30_prediction` calls *are* in the GEO metadata. | The strongest possible candidate: built to predict pCR under exactly this regimen. If signull cannot distinguish DLDA-30 from random, signull is broken. |
| Real candidate, different biology | **PAM50** (Parker et al. 2009) | 50 genes | `genefu::pam50`; symbols widely available | `pam50_class` calls in the metadata. Subtype, not response — expect weaker pCR association. |
| Wrong-endpoint / weak comparator | **MammaPrint 70-gene** (`genefu::sig.gene70`) | 70 Agilent probes / 56 genes | genefu | Developed for distant-metastasis-free survival in node-negative untreated patients — a **different endpoint in a different population**. Expected to be near-null for pCR. A good "published but not applicable here" control. |
| Wrong-endpoint / weak comparator | **Wang 76-gene** (`genefu::sig.gene76`) | 76 probes | genefu | Derived *on GSE2034* for relapse. On GSE25055 it is out-of-domain. Useful precisely because a naive user would assume a famous signature must work anywhere. |
| **Known-weak / meaningless** | Venet 2011's three control signatures: postprandial laughter, mouse social defeat, skin fibroblast localisation | varies | **Text S1 (PDF) or Dataset S1 (87 MB ZIP)** of the Venet paper — requires manual extraction | The canonical "should not work but does" comparator. Highest-value fixture in the list; also the highest extraction cost. |
| Positive control | **HALLMARK_G2M_CHECKPOINT** / **HALLMARK_E2F_TARGETS** | ~200 genes each | MSigDB hallmark GMT, credential-free: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt` (verified 200) | Proliferation. Should be strongly associated. Doubles as the meta-PCNA proxy if Table S1 is not extracted. |
| **Negative control** | Random draws from the matrix itself | any | generated | Under a correctly specified null, empirical p-values from random candidates must be uniform. This is the calibration acceptance test. |

Empirically observed on the prepared matrix (5 random 50-probe mean-z signatures,
AUC vs pCR): **0.401, 0.474, 0.478, 0.559, 0.605**. A spread that wide around 0.5 on
n=306 is exactly why a size-matched null is needed rather than a fixed AUC threshold.

### 2.3 D2 — **GSE20194** (MAQC-II breast cancer)

- Accession: [GSE20194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20194)
- Platform: **GPL96**, 22,283 probes. **n = 278**.
- Outcome: `pcr_vs_rd`, **56 pCR / 222 RD**, **no missing values**, **20.1% positive**.
- Format: GEO series matrix `.txt.gz`, ~17 MB gz.
  `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/matrix/GSE20194_series_matrix.txt.gz`
- Covariates: age, race, er_status, pr_status, her2 status/IHC/FISH, tbefore, nbefore,
  bmngrd, histology, treatment code.
- **Why include it:** an *independent consortium's* pCR cohort on the same platform with
  the same endpoint definition. This is the cleanest possible test of the Starmans claim
  — same platform, same endpoint, same signature, different dataset. If the
  random-signature significance rate differs materially between GSE25055 and GSE20194,
  the dataset-specific-null argument is demonstrated in-repo rather than cited.
- **Gotcha, confirmed by inspection and the reason to parse defensively:** the
  `!Sample_characteristics_ch1` keys are **not aligned across samples**. Scanning the
  characteristics rows shows the `pcr_vs_rd` token distributed across three different
  rows (1, 9 and 268 occurrences). Any parser that assumes "row 5 is the pCR row" will
  produce a mis-labelled outcome vector for a minority of samples — a silent, severe
  bug. `fetch_benchmark.py` parses `key: value` per cell for exactly this reason.
- Same signature set as D1 applies (same platform, same endpoint). DLDA-30 is again the
  natural strong candidate.

### 2.4 D3 — **GSE2034** (Wang et al., node-negative breast cancer)

- Accession: [GSE2034](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034),
  PMID [15721472](https://pubmed.ncbi.nlm.nih.gov/15721472/).
- Platform: **GPL96**, 22,283 probes. **n = 286**. ~14 MB gz.
- Outcome available in the series matrix: **`bone relapses (1=yes, 0=no)`** —
  **69 yes / 217 no**, 24.1% positive, no missing.
- **Serious gotcha — read before choosing this one.** The headline endpoint of the Wang
  paper is *any* distant metastasis (the series summary states 180 relapse-free vs 106
  with distant metastasis). **That label is NOT in the series matrix.** The only binary
  outcome in the downloadable file is bone relapse. The main relapse label and
  time-to-relapse live in the paper's supplementary "patient clinical parameters sheet",
  which is a separate, less stable artefact. This fails my "documented binary outcome in
  the downloadable file" requirement for the *intended* endpoint, and passes it only for
  bone relapse. **Ranked third for this reason.**
- **Why include it anyway:** (a) it is a *different kind* of binary outcome — a
  dichotomised long-term relapse event rather than a short-term treatment response, so
  it stresses different assumptions; (b) the Wang 76-gene signature was **derived on this
  cohort**, making it a ready-made, clearly-labelled **overfitting demonstration**: a
  signature that will look excellent here for the wrong reason. That is a genuinely
  useful negative teaching fixture for a tool whose whole purpose is scepticism.
- Untreated (no adjuvant systemic therapy), lymph-node-negative — a prognostic rather
  than predictive setting, complementary to D1/D2.

### 2.5 Considered and not chosen

- **NKI / van de Vijver (n=295)** — the cohort Venet actually used. Not on GEO as a
  clean credential-free series matrix; historically distributed via the Rosetta/NKI
  portal and various rehostings of uncertain provenance. Endpoint is survival, not
  binary. **Rejected: fetchability and endpoint.**
- **TCGA-BRCA via cBioPortal / GDC** — open and large, but the natural binary endpoints
  are either subtype (a labelling exercise, not an outcome) or heavily dichotomised
  survival, and the RNA-seq/microarray platform change breaks probe-level signature
  fixtures. **Rejected for the first benchmark; revisit for RNA-seq generalisation.**
- **GSE39582 (colorectal, n=585)** — large and open, plausible binary endpoints (MSI
  status, relapse). Not verified in this pass. **Deferred**, and the obvious candidate
  for a non-breast, non-proliferation-dominated fourth dataset — worth doing, because
  all three recommendations above inherit breast cancer's single dominant proliferation
  axis, which is itself a confound (Venet's meta-PCNA finding).

### 2.6 Recommendation

Develop against **GSE25055**. Calibrate on **GSE25065**. Use **GSE20194** as the
cross-dataset null-variability test. Add **GSE2034** only as an overfitting-demo
fixture. Treat **GSE39582** (or another non-breast cohort) as a known gap — because
three breast cohorts all dominated by the same proliferation axis is not independent
validation, and the tool should not ship claiming otherwise.

---

## Part 3 — `scripts/fetch_benchmark.py`

Self-contained downloader/preparer for D1, with D2's paired validation cohort supported.
Python 3.11 + pandas + numpy only (stdlib for everything else).

```bash
python scripts/fetch_benchmark.py                     # GSE25055 (default)
python scripts/fetch_benchmark.py --accession GSE25065
python scripts/fetch_benchmark.py --force-download    # ignore the cached copy
python scripts/fetch_benchmark.py --strict-checksum   # hard-fail on any sha256 drift
```

Outputs:

| Path | Contents |
|---|---|
| `data/raw/<ACC>_series_matrix.txt.gz` | verbatim GEO download |
| `data/raw/<ACC>_series_matrix.txt.gz.meta.json` | download sidecar: URL, UTC timestamp, size, sha256 |
| `data/processed/<ACC>_expression.tsv.gz` | 22,283 probes x N samples, log2, probe IDs as index, GSM IDs as columns |
| `data/processed/<ACC>_outcome.tsv` | `sample_id`, `outcome` (1=pCR, 0=RD), `outcome_label`, + 21 covariates including the four published-predictor calls |
| `data/processed/<ACC>_provenance.json` | full provenance: accession, URL, download and prepare timestamps, raw sha256, output sha256s, shape, class balance, any warnings |

Idempotency: an existing raw file whose sha256 matches its sidecar is reused; anything
else triggers a re-download. Validation is against the numbers in section 2.2 and fails
loudly (exit 2) on any drift.

### Execution log — verified 2026-09-01, conda env `signull` (Python 3.11.16, pandas 3.0.5, numpy 2.4.6)

Verified **by running**:
- Cold run on GSE25055: downloaded 37,718,581 bytes in ~9 s, sha256 matched, parsed to
  **22,283 x 310**, outcome usable on **306 (57 pCR / 249 RD, 18.6%)**, 4 dropped. All
  three outputs written. Whole run ~19 s.
- Warm re-run: `[skip] ... already present and checksum-verified`, no network fetch,
  identical outputs. Idempotency confirmed.
- GSE25065: downloaded 24,194,802 bytes, sha256 matched, **22,283 x 198**, outcome
  usable on **182 (42 pCR / 140 RD, 23.1%)**, 16 dropped.
- Corruption recovery: truncating the raw `.gz` to 3 MB was detected by the sidecar
  checksum and self-healed by re-downloading.
- All four loud-failure paths, each exiting 2 with an actionable message:
  (a) shape + class-balance drift -> names each mismatched quantity, the GEO URL, and
  the two files to update; (b) outcome field renamed -> lists all 24 available
  characteristic keys; (c) sha256 drift under `--strict-checksum`; (d) HTTP 404 on a
  nonexistent accession -> reports code, reason, URL and the GEO page to check.
- Round-trip read-back with pandas: expression `(22283, 310)`, outcome `(306, 23)`, no
  duplicate columns, `outcome` dtype int64, all values finite after aligning the matrix
  to the outcome index.
- Sanity signal: 5 random 50-probe mean-z signatures scored AUC 0.401–0.605 vs pCR.

Verified **by reading only** (not executed):
- The claim that DLDA-30's per-sample calls in the GEO metadata correspond to the Hess
  2006 classifier — taken from the GEO series description, not independently recomputed.
- genefu's signature contents. The package was not installed or run; sizes come from its
  Bioconductor reference manual. No R was executed in this campaign.
- The contents of Venet's Text S1 / Dataset S1. Only checked that the URLs return 200
  with the right content types; the PDFs and the 87 MB ZIP were not downloaded or parsed.
- Hess 2006's 30-probe list. Confirmed the paper exists and describes DLDA-30; the probe
  list itself was **not** located in machine-readable form and will need manual
  transcription.

---

## Open items for the next wave

1. **Extract the signature fixtures.** GGI probe IDs from `genefu::sig.ggi` (one R
   session), DLDA-30 by hand-transcription from Hess 2006, hallmark sets from the MSigDB
   GMT URL above. Store as TSVs under a `data/signatures/` or `tests/fixtures/` path —
   owner to be decided; outside this campaign's file scope.
2. **Decide the AFFX control-probe policy** for the random sampling pool, and make it an
   explicit, documented parameter rather than an accident of implementation.
3. **Benchmark signull against SigCheck on GSE25055.** This is the honest comparison the
   prior-art finding demands. Disagreement needs an explanation, not a shrug.
4. **The Venet meaningless-signature fixtures** are the single highest-value test asset
   and the most laborious to obtain (PDF or 87 MB ZIP). Decide early whether to pay that
   cost; if not, say so in the README rather than implying the comparator exists.
5. **Add a non-breast cohort** (GSE39582 is the lead candidate, unverified). All three
   recommended datasets share breast cancer's dominant proliferation axis.
