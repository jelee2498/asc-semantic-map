# P7 — Normative model and structural equation model

**Manuscript:** Fig. 5; Fig. 6a–b (normative curves); Supp. Fig. 8–10
(alternative models).

**Runnable: yes.** A five-script Python ↔ R chain.

---

## Verification status

`11_sem_zscore_3analysis.R` re-run on 2026-08-25. Every Fig. 5 number reproduces:

| Value | Manuscript | Reproduced |
|---|---|---|
| χ² / df | 1.559 | 81.090 / 52 = **1.559** |
| RMSEA | 0.051 | **0.0509** |
| CFI | 0.980 | **0.9802** |
| SAL → CEN (β, z) | 0.684 | **0.684** (z = 9.87) |
| CEN → DMN (β, z) | 0.851 | **0.851** (z = 14.72) |
| SAL → DMN (β, z) | 0.067, n.s. | **0.067** (z = 0.95, p = 0.34) |
| DMN → autism symptoms (β) | 0.414 | **0.414** (z = 8.93) |
| DMN → cognitive ability (β) | −0.184 | **−0.184** (z = −3.33) |

Competing models, confirming "alternative models … showed substantially reduced
model fit":

| Model | χ² | df | χ²/df | RMSEA | CFI |
|---|---|---|---|---|---|
| **Model 1** (Fig. 5) | 81.09 | 52 | **1.559** | **0.0509** | **0.9802** |
| Model 2 (Supp. Fig. 8) | 85.53 | 52 | 1.645 | 0.0546 | 0.9771 |
| Model 3 (Supp. Fig. 9) | 107.57 | 52 | 2.069 | 0.0703 | 0.9621 |
| Model 4 (Supp. Fig. 10) | 230.61 | 51 | 4.522 | 0.1277 | 0.8775 |

Model 1 wins on every index. N = 216 (97 TD + 119 ASC after complete-case
filtering on the 9 behavioural indicators).

### A subtlety in the Fig. 5 legend

The legend calls these "standardized path coefficients (β)". They are lavaan's
**unstandardized `est`**, not `std.all`. Because every observed variable is
already a z-score, `est` equals `std.lv` — partially standardised — but
`std.all` additionally rescales by the latent variances and gives visibly
different numbers:

| Path | `est` (Fig. 5) | `std.all` |
|---|---|---|
| SAL → CEN | 0.684 | 0.557 |
| CEN → DMN | 0.851 | 0.753 |
| DMN → SRS | 0.414 | 0.588 |

The figure is internally consistent and the values are correct; the word
"standardized" is just doing double duty. `sem_model1_headline.csv` now records
both columns so there is no ambiguity about which is plotted.

---

## Order of execution

| | Script | Role |
|---|---|---|
| 0 | `11_sem_zscore_0save_raw.py` | triple-network (SAL/CEN/DMN) raw scores from P4 PC1 + P5 PR |
| 1 | `11_sem_zscore_1gam.R` | GAM normative model on TD; z-scores everyone against it |
| 2 | `11_sem_zscore_2save_data.py` | joins z-scores to SRS + NIH Toolbox; flips `sem_dmn`; averages |
| 3 | `11_sem_zscore_3analysis.R` | fits the four competing SEMs; **Fig. 5** |
| 4 | `11_sem_zscore_4results.py` | DMN-vs-behaviour scatter panels |

```bash
python 11_sem_zscore_0save_raw.py
Rscript 11_sem_zscore_1gam.R
python 11_sem_zscore_2save_data.py
Rscript 11_sem_zscore_3analysis.R
python 11_sem_zscore_4results.py
```

Step 1 selects a non-linear age term only when Δ AIC > 2 over the linear model,
with `k = 3` (quadratic). Step 2 flips `sem_dmn` (`-1 ×`) before averaging so the
semantic and dimensionality metrics point the same way — PC1 decreases in ASC
where PR increases.

## Inputs

| Input | Source |
|---|---|
| `sem_list_{td,asd}.npy`, `sig_regions_…npy` | P4 |
| `cov_pr_{td,asd}.npy`, `cov_sig_mmp.npy` | P5 |
| `mask_fdr-0.01.npy` | P3 (Zenodo) |
| `participants_df_deepmreye_inc_byhx.xlsx` | per-subject tier, on request — SRS |
| NIH Toolbox scores | HBN LORIS release (`roots.hbn_phenotype`) |
| EG17 atlas | `templates/WashU/EG17.mat` |

## Outputs

`…/screen-True_overlap-False/`

| File | Contents |
|---|---|
| `{sem,dim}_data_{td,asd}_raw.csv` | raw triple-network scores + demographics |
| `zscore_{td,asd}.csv` | GAM-referenced deviations |
| `all_sem_dim_zscore.csv` | **SEM input** — brain z-scores + 9 behavioural indicators |
| `gam_trajectory_*.csv` | fitted normative curves (Fig. 6a–b) |
| `sem_model_fit_measures.csv` | **new** — all four models |
| `sem_model1_paths_std.csv` | **new** — `est`, `std.lv`, `std.all`, se, z, p |
| `sem_model1_defined_effects.csv` | **new** — mediation / cascade effects |
| `sem_model1_headline.csv` | **new** — the Fig. 5 numbers in one row |

`all_sem_dim_zscore.csv` and `cluster_assignments` feed **P8**.

---

## Changes from the working tree

1. **`11_sem_zscore_3analysis.R` now persists its results.** This was the
   headline problem: the original printed everything with `summary()` and
   `cat()` and saved **nothing**, so Fig. 5's fit indices and path coefficients
   existed only in a terminal scrollback. Four CSVs are now written, mirroring
   what `12_heterogeneity_2sem_zscore.R` already did for the subtype fits.

2. **A hard-coded conda path that made the chain unrunnable elsewhere.**
   `11_sem_zscore_0save_raw.py` resolved the EG17 atlas as
   ```python
   Path('D:/Users/jongeun/anaconda3/envs/fastsrm/Lib/site-packages/…/EG17.mat')
   ```
   with **no fallback** — drive letter, user name, conda install and environment
   name all baked in. Unlike `07_pca.py` and `09_dimensionality.py`, which tried
   a shared directory first, this was the only path attempted, so P7 failed on
   any other machine. Now `template('eg17')`.

3. **Shared config on both sides of the language boundary.** Added
   `lib/project_config.R`, the R counterpart of `project_config.py`, reading the
   same `config/{paths,params}.yml`. The R scripts previously hard-coded
   `store7 <- "S:/"` *and* re-declared every analysis parameter as a literal —
   the precise drift risk the config exists to prevent, since the Python and R
   halves meet through a directory name built from those values. A mismatch
   there does not error; R just reads an empty directory. Verified: the R side
   resolves to the same published folder the original built.

   `yaml` (2.3.7) added to `environment/renv.lock` — 39 packages.

4. **Robust script-location bootstrap.** The R scripts locate `lib/` via
   `--file=`, then `sys.frame()$ofile`, then RStudio, then the working
   directory, so they work under `Rscript`, `source()` and the RStudio console
   alike.

## Known issues

- The NIH Toolbox scores come from the HBN LORIS release, not from
  `participants_df`, so `roots.hbn_phenotype` must be set for step 2. The
  per-subject tier includes `nih_toolbox_scores.csv`, which removes that
  dependency — see `docs/DERIVED_DATA_MANIFEST.md` §4.
- The R scripts have no `--out-root`; they write beside their inputs. Step 3
  only *adds* files, so a re-run cannot damage existing results, but there is no
  scratch-run option as there is for P4/P5.
- Three other `screen-*_overlap-*` parameter branches exist in the working tree
  and are **not** the published analysis. Only `screen-True_overlap-False` is.
