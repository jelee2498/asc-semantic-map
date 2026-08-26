# P3 — Parcel-level encoding model and performance mask

**Manuscript:** Fig. 1b–d; Methods *Encoding model, 2) Model fitting and
evaluation*.

**Runnable here: no.** Published for inspection. Requires preprocessed HBN fMRI
(Data Use Agreement) and the P2 regressors (derived from copyrighted stimuli).

Its **outputs are published** on Figshare and are the entry point for
reproduction: P4 starts by loading the fold-averaged weights and the performance
mask produced here. See `docs/FIGSHARE_MANIFEST.md`.

Scripts are the working-tree originals with **one change**: storage roots now come
from `config/paths.yml` via `lib/project_config.py`, instead of the hard-coded
drive letters and mount points the analyses were run with. All analysis code is
untouched. `python docs/verify_paths.py` checks that the configuration resolves to
the same paths the originals used.

The PBS launcher reads the project root from `$ASC_PROJECT_ROOT`; set it to
`roots.project` from `config/paths.yml` before `qsub`.

---

## Order of execution

### 0. Prepare fMRI with cross-validation folds — `04_prepare_fmri.py`

Denoises (`default+me`: FD + global signal + linear trend + motion energy),
parcellates to HCP-MMP 360, splits Despicable Me into 9 equal chunks merged into
3 cross-validation folds, and writes train/test arrays per subject. Also fits
DetSRM, though the parcel-level encoding uses `k = 0` (no SRM).

> **Boundary call.** This sits between preprocessing and encoding. It is included
> here rather than excluded with the fMRIPrep/Ciftify stage because the Methods
> describe its cross-validation scheme, and P3's ridge fit reads its output
> directly. Say if you would rather it were dropped.

> **Known path mismatch, carried over from the working tree.** This script writes
> to `17_param_search/00_prepare_data/`, but `06_encoding_model_*` reads from
> `17_param_search/00_prepare_data_release11/`. The discrepancy is documented in
> the script's own docstring and was deliberately not "fixed" during refactoring,
> to keep output paths byte-identical to the original. It must be resolved before
> this stage can be re-run end to end.

### 1. Ridge alpha search — `06_encoding_model_0search_alpha_sub.py`

Per subject. Bootstrapped ridge over 20 candidate alphas
(`np.logspace`), with delayed regressors at 3, 5, 7, 9 TRs (2.4/4.0/5.6/7.2 s)
giving 85 × 4 = 340 columns.

Also estimates the **split-half noise ceiling** used to correct encoding accuracy
(published mean 0.63 ± 0.20).

`06_encoding_model_0search_alpha_pbs.pbs` — PBS array, `#PBS -J 1-246`, one task
per subject. No serial fallback; on a single machine the loop must be driven
manually.

### 2. Refit at the optimal alpha — `06_encoding_model_1optimal_alpha.py`

Pools `boot_corrs` across subjects, takes the single alpha maximising mean
validation correlation, and refits every subject at that alpha. Writes
fold-averaged `weight`, `test_corrs`, `test_corrs_corrected`, `noise_ceiling`,
and per-fold `pred_bold`.

Parallel over subjects with joblib; `--debug_subject` runs one subject serially.

### 3. Performance mask — `06_encoding_model_2perf_mask_avg.py`

Averages observed and predicted BOLD across subjects, correlates them per parcel,
and FDR-corrects.

Produces **`mask_fdr-0.01.npy` — 291 of 360 parcels (80.8%)**, the mask that
restricts every subsequent analysis in the paper.

---

## Published outputs (Figshare)

`…/chunk-9/fold-avg/d-True/k-0/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-True/`

| | |
|---|---|
| `weight/{sub_id}.npy` × 246 | 85 features × 360 parcels |
| `test_corrs/`, `test_corrs_corrected/`, `noise_ceiling/` | Fig. 1c |
| `test_corrs_of_avg.npy`, `test_corrs_pval_of_avg.npy` | |
| `mask_fdr-0.01.npy` | Fig. 1d |
| **total** | **497 MB** |

Note `s_alpha-True` here. The vertex-level branch that feeds P5 uses
`s_alpha-False` (per-vertex alphas). Mixing them is the easiest way to read back
an empty directory.

## Known issues

- The `04_prepare_fmri.py` output-path mismatch described above.
- PBS array submission only; no serial driver for a single-machine run.
