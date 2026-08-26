# P8 — Neurosubtyping

**Manuscript:** Fig. 6c–g; Supp. Fig. 11 (three-cluster solution).

**Runnable: yes.** Consumes P7's z-score tables.

---

## Verification status

Full chain re-run on 2026-08-25. **All 11 outputs byte-identical** to the
published ones, including the relocated cluster assignments.

| Output | Result |
|---|---|
| `cluster_assignments_profile.csv` | byte-identical |
| `sem_input_{pooled,subtype-0,subtype-1,asd_only,all_with_cluster}.csv` | byte-identical |
| `subtype_sample_summary.csv` | byte-identical |
| `subtype_model1_{fit_measures,paths_std,defined_effects,multigroup_fit}.csv` | byte-identical |

| Value | Manuscript | Reproduced |
|---|---|---|
| Optimal k | 2 (silhouette) | **k = 2**, silhouette 0.222 / 0.219 / 0.189 / 0.179 |
| Subtype sizes | 58 / 70 | **58 / 70** |
| Subtype-1 fit | χ²/df 1.566, RMSEA 0.061, CFI 0.965 | **1.566 / 0.0614 / 0.9653** |
| Subtype-2 fit | χ²/df 1.429, RMSEA 0.051, CFI 0.983 | **1.429 / 0.0513 / 0.9825** |
| Subtype-2 direct path | z = 0.60, p = 0.55 | **z = 0.60, p = 0.546** |
| Subtype-1 direct path | z = 1.81, p = 0.07 | z = 1.83, p = 0.068 — see below |

**On that last row.** `subtype_model1_paths_std.csv` reports the *standardised*
solution, whose standard errors come from the delta method and differ slightly
from the unstandardised ones. The manuscript's z = 1.81 is the unstandardised
value; this file gives z = 1.83 for the same path. The file is byte-identical to
the published one, so both numbers come from the same fit — it is a choice of
column, the same nuance documented for Fig. 5 in P7. Either way the path is a
trend (p ≈ 0.07) in Subtype-1 and clearly null (p = 0.55) in Subtype-2, which is
the claim Fig. 6g makes.

### Naming

The code's `subtype-0` / `subtype-1` are the manuscript's **Subtype-1** /
**Subtype-2**. `subtype-0` is the younger, milder cluster (n = 58).

---

## Order of execution

| | Script | Role |
|---|---|---|
| 0 | `12_heterogeneity_0subtype.py` | K-means on triple-network z-scores; **Fig. 6c–f**, Supp. Fig. 11 |
| 1 | `12_heterogeneity_1save_data.py` | builds per-subtype SEM tables (TD shared across both) |
| 2 | `12_heterogeneity_2sem_zscore.R` | refits Model 1 within each subtype; **Fig. 6g** |

```bash
python 12_heterogeneity_0subtype.py
python 12_heterogeneity_1save_data.py
Rscript 12_heterogeneity_2sem_zscore.R
```

Clustering runs on the six triple-network z-scores (`{sem,dim}_{sal,dmn,cen}`)
for the 128 ASC subjects, profile-normalised. The SEM refits use 119 of them
(53 / 66): nine lack one or more of the nine behavioural indicators and are
dropped by complete-case filtering. That, not the subtyping, is why the SEM Ns
are smaller.

**TD subjects are shared between the two subtype tables**, so those two fits are
not independent and a difference between them is not a test. Step 2 uses
`sem_input_asd_only.csv` for the formal multi-group comparison.

## Inputs

| Input | Source |
|---|---|
| `zscore_{td,asd}.csv`, `all_sem_dim_zscore.csv` | P7 |
| NIH Toolbox scores | HBN LORIS (`roots.hbn_phenotype`) |
| Comorbidity table | legacy `25_sensitivity/00_comorbidity_release11` tree |

## Outputs

`{pipeline}/12_heterogeneity/out/screen-True_overlap-False/`

| File | Contents |
|---|---|
| `cluster_assignments_profile.csv` | **subject → cluster; relocated here (see below)** |
| `cluster-kmeans_profile/sem_input_*.csv` | per-subtype SEM tables |
| `cluster-kmeans_profile/subtype_sample_summary.csv` | n / age / network means per subset |
| `cluster-kmeans_profile/subtype_model1_*.csv` | fit, paths, defined effects, multi-group |

---

## Changes from the working tree

1. **Cluster assignments moved out of the figure directory.**
   `12_heterogeneity_0subtype.py` wrote **no data at all** to
   `2_pipeline`: `cluster_assignments_profile.csv` — a required input to step 1 —
   went into `1_code/99_main/claude_figures/`, making a figures folder a
   pipeline dependency and meaning a run modified the source checkout. It now
   writes to the pipeline output directory; step 1 reads that location and falls
   back to the legacy one so earlier runs still resolve. Four further result
   CSVs (trajectory comparison, outlier prevalence, behavioural summaries) were
   redirected the same way, so clearing `figures/` cannot lose a result.

2. **K-means seed surfaced into the configuration.** `random_state = 42` and
   `n_init = 50` existed only as defaults in a function signature, so the
   determinism behind the published 58/70 split was invisible from the config.
   Both are now in `config/params.yml` and the script's `CONFIG`.

3. **HBN LORIS path via `require()`.** The NIH Toolbox CSV was read from a bare
   `store9` root. It now resolves through `roots.hbn_phenotype` and fails with a
   message naming the config key, rather than letting a missing file surface
   later as an empty merge.

4. **Storage roots → `config/{paths,params}.yml`**, Python and R alike
   (`lib/project_config.{py,R}`).

5. **Figures → `figures/p8_subtyping/screen-True_overlap-False/`.**

## Known issues

- The comorbidity table is read from the legacy `25_sensitivity` tree, a
  cross-tree dependency carried over from the original working directory. Same
  pattern as P2 step 3.
- Output paths are keyed on `screen`/`overlap` only, not the full parameter
  tree, because the full tree pushes filenames past the Windows 260-character
  limit. This is deliberate and documented in the scripts.
- The R script has no `--out-root`; it writes beside its inputs.
