# P6 — Symptom prediction

**Manuscript:** Fig. 4a–b.

**Runnable: yes.** Consumes only P4 and P5 outputs, both published.

Scripts are the working-tree originals; changes are limited to path
configuration (see [Changes](#changes-from-the-working-tree)). Analysis code is
untouched.

---

## Verification status

**Not verified by re-running — by decision, not oversight.** The published
figures and statistics were checked against the manuscript by direct comparison
of the saved results instead:

| Value | Manuscript (Fig. 4a) | `model_comparison_statistics.csv` |
|---|---|---|
| RRB, combined | r = 0.461, R² = 0.212 | r = 0.4610, R² = 0.2117 |
| Social communication, combined | r = 0.432, R² = 0.213 | r = 0.4619, R² = 0.2131 |

Note the communication discrepancy: R² matches, r does not (0.432 vs 0.462).
Worth checking which number Fig. 4a plots — the per-repetition mean, or the
correlation over pooled predictions.

A full re-run is expensive: 5 subscales × 100 repetitions × 20 folds ×
(1000 permutations + SHARP with J = 30, K = 10), each fitting an `ElasticNetCV`.
The original run took roughly six hours across the five subscales.

### Why `_sharp` is the manuscript version

Seven near-duplicate variants of this analysis exist in the working tree
(`10_behavior{,_parallel,_matched,_matched_parallel,_matched_sharp,_asd_opt1*,_asd_opt2*}`).
`10_behavior_sharp.py` is the published one, on two independent grounds:

1. It is the only variant whose figure directory contains the Fig. 4 panels
   (`model_comparison.png`, the five `scatter_*` and `null_dist_*` files),
   all dated 2026-07-20.
2. It is the only variant implementing both statistical tests the manuscript
   cites — the corrected resampled t-test (Nadeau & Bengio, ref. 71) and SHARP
   (Zeng et al., ref. 72). The plain variants cite neither.

The other six are supplementary or superseded and are not distributed.

---

## Usage

```bash
python 10_behavior_sharp.py
```

Loads per-subject PC1 scores and participation ratios, restricts them per fold
to the regions that are significant in that fold's training split, fits
`ElasticNetCV`, and compares three feature sets — semantic, dimensionality, and
their combination — against a permutation null.

Fold-specific region selection is what keeps this leak-free: the TD-vs-ASC GLM
that defines the feature set is refitted inside each training fold rather than
once on the full sample.

Parameters (`config/params.yml`): `n_rep = 100`, `n_cv = 20`,
`n_perm = 1000`, `sharp_J = 30`, `sharp_K = 10`,
`one_sided_combined = true`.

## Inputs

All present and resolved (checked 2026-08-25):

| Input | Source | Size |
|---|---|---|
| `sem_list_{td,asd}.npy` | P4 | 2.4 / 3.1 MB |
| `sig_regions_Whole_sign-True_iter-10.npy` | P4 — 33 parcels | 4 KB |
| `cov_pr_{td,asd}.npy` | P5 | 304 / 392 KB |
| `cov_sig_mmp.npy` | P5 — 95 parcels | 4 KB |
| `participants_df_deepmreye_inc_byhx.xlsx` | Figshare — SRS targets, covariates | 120 KB |

Targets are the five SRS subscales: `SRS_{AWR,COG,COM,MOT,RRB}_T`.

## Outputs

`{pipeline}/10_behavior_sharp/out/default+me/mmp/k-0-50/bias-0.9_verb-1.0/wn-True/sa_sem-True_dim-False/`

| File | Contents |
|---|---|
| `{model}_{item}_results.npz` | per-repetition r, R², predictions — 15 files |
| `{model}_{item}_perm.npy` | permutation null — 15 files |
| `fold_info_{item}.npz` | per-fold region selections — 5 files, 5.5 MB each |
| `model_comparison_statistics.csv` | **the Fig. 4a/b numbers** |
| `model_comparison_sharp.csv` | SHARP test results |
| `model_comparison_trend.csv` | cross-subscale Wilcoxon |

---

## Changes from the working tree

1. **Storage roots → `config/paths.yml`** (`lib/project_config.py`).
2. **`sharp_test` import repointed** to `lib/`, where the module now lives. The
   original relied on it sitting beside the script.
3. **Figures → `figures/p6_prediction/`** instead of the source tree.

The three `os.name == 'nt'` branches in this script were **left alone**: unlike
P4 and P5, they only choose between `plt.show()` and `plt.close()` after the
figure has already been saved. They do not skip analysis, so they are not the
silent-skip pattern fixed elsewhere.

No `UnicodeEncodeError` risk here — no `print()` statement contains non-ASCII.

## Known issues

- Seven near-duplicate variants exist upstream; only `_sharp` is distributed.
  See above for the evidence.
- `fold_info_*.npz` totals ~27 MB and is diagnostic only — not required to
  reproduce the reported statistics.
- The Fig. 4a communication `r` discrepancy noted above is unresolved.
