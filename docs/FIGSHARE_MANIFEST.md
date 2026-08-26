# Figshare manifest

Derived data required to reproduce the analyses in this repository. Download
into `./derivatives/`, preserving the directory structure below — the paths
encode the analysis parameters (`config/params.yml`) and the scripts resolve
them literally.

**Staged 2026-08-26 to `3_output/Figshare/` — 525 MB, 1540 files**, with
`SHA256SUMS.txt` covering every file (verified). The archive mirrors the project
directory layout so it extracts into place.

Validated end to end: with the bundle as the *only* data source, P4 reproduced
all nine of its published arrays byte-for-byte.

Sizes below are measured, not estimated.

---

## 1. Encoding-model results — entry point for P4

`2_pipeline/99_main/06_encoding_model/out/default+me/mmp/chunk-9/fold-avg/d-True/k-0/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-True/`

| Contents | Size |
|---|---|
| `weight/{sub_id}.npy` × 246 — ridge weights, 85 features × 360 parcels | 136 MB |
| `test_corrs/`, `test_corrs_corrected/`, `noise_ceiling/` — 246 each | 98 MB |
| `test_corrs_of_avg.npy`, `test_corrs_pval_of_avg.npy` | small |
| `mask_fdr-0.01.npy` — **the performance mask, 291 of 360 parcels** | small |
| **Staged total** | **234 MB, 987 files** |

> The `s_alpha-True/` directory on disk is 497 MB, but 263 MB of that is a
> `sem/` subtree of parameter-search leftovers (`mask_bf-0.01`, `sign-False`,
> `iter-2/4/6`) that is **not** the manuscript configuration. Excluded.

This is the first shareable point in the pipeline. Everything upstream is either
governed by the HBN Data Use Agreement (imaging) or copyrighted (stimuli).

## 2. Participation-ratio arrays — entry point for P5

`2_pipeline/99_main/09_dimensionality/out/32k/default+me/det_results_inc_byhx_docu-True/k-50/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-False/`

| File | Size |
|---|---|
| `cov_pr_td.npy` — (108, 360) | 311 KB |
| `cov_pr_asd.npy` — (138, 360) | 397 KB |
| `cov_sig_mmp.npy`, `cov_stat_results.npy`, `cov_stat_results_wholebrain.npy` | ~19 KB |
| `cov_sim_mmp_{td,asd}_avg_list.pkl` — Fig. 3a insets | 40 MB |

P5 enters at the cached-PR path, skipping the vertex-level encoding weights
entirely (~20 GB: 340 × 59,412 float32 per subject × 246). This is the same code
path the analysis takes on Windows, where `overwrite` defaults to `False`.

**Confirmed 2026-08-25.** A run seeded with only `cov_pr_td.npy` and
`cov_pr_asd.npy` reproduced `cov_sig_mmp.npy`, `cov_stat_results.npy` and
`cov_stat_results_wholebrain.npy` **byte-identically**. The ~20 GB of vertex
weights are genuinely not needed.

For the Fig. 3a similarity-matrix insets, add
`cov_sim_mmp_td_avg_list.pkl` and `cov_sim_mmp_asd_avg_list.pkl` (2 × 21 MB) from
the same directory — verified against the code as the files those insets read.
Without them the run completes and only the insets are skipped.

## 3. Spatial-null cache — exact `p_eigen` reproduction

`2_pipeline/99_main/_spatial_nulls_cache/`

| File | Size |
|---|---|
| `emodes_L_midthickness_200.npz` | 25 MB |
| `emodes_R_midthickness_200.npz` | 25 MB |
| `spin_perm_id_mmp360_n2000_seed0.npy` | 2.8 MB |

Without these the null models regenerate and give *statistically equivalent* but
not *identical* p-values. See `docs/STAGE_A_NOTES.md` §3.3 for why exact
reproduction cannot be obtained from a seed alone.

## 4. Phenotypic covariates

`participants_df_deepmreye_inc_byhx.xlsx` — 489 rows × 26 columns.

Supplies the GLM covariates (Age, Sex, Site, Mean_FD_DM), the quality-control
flags that define the analytic cohort, and the SRS targets for P6/P7. Every
pipeline from P4 onward reads it.

Columns: `DX`, `ASD_certainty`, `ASD_document`, `Site`, `Age`, `Sex`,
`SRS_{AWR,COG,COM,DSMRRB,MOT,RRB,SCI,Total}_T`, `SCQ_Total`, `CELF_Total`,
`condition_ok`, `prep_ok` (×2), `Mean_FD_{DM,TP}`, `QC_link_recon_all`,
`Rating_recon_all`, `Remarks`, `Rating_deepmreye_movie{DM,TP}`.

`nih_toolbox_scores.csv` — **staged**. The four NIH Toolbox percentile scores
used as the SEM's cognitive-ability indicators, extracted from the HBN LORIS
release for the same 489 subjects (453 complete on all four). Publishing it
removes P7/P8's dependency on separate LORIS access.

`QC_link_recon_all` (an internal QC URL) was **dropped** — the staged table is
489 × 25 rather than 489 × 26.

> **Still open before release:** both tables are derived from HBN phenotypic data
> keyed to HBN subject IDs. Confirm with the HBN data-access team that
> redistributing derived phenotypic variables is permitted.

## 5. Stimulus annotations and gaze-weighted regressors (P2)

**Decided 2026-08-25: publish.** Earlier drafts of this manifest treated the
regressors as non-redistributable; that was over-cautious. They describe *which
semantic labels are present in each TR* and *how strongly each subject attended
to them* — a time-aligned semantic description of the films, not the films. This
is in line with what comparable naturalistic-encoding papers publish. The frames,
the polygon masks and the per-frame gaze maps remain excluded (see
[Not published](#not-published)).

Total ≈ 202 MB.

| Item | Size | Notes |
|---|---|---|
| `wordnet_gaze_weighted_regressor_…/within/bias-0.9_verb-1.0/` | 87.5 MB, 500 files | per-subject × per-movie, the direct P3 input |
| `results_hbn_td_asd_inc_byhx_edited.pickle` | 79 MB | DeepMREye gaze: 781 runs (399 DM, 382 TP), per-TR `pred_xy` + `uncertainty` |
| `DM_motion_energy.xlsx`, `TP_motion_energy.xlsx` | 32 MB | required by the `default+me` denoising, so P1/P3 are runnable for DUA holders |
| `sync_regressor_DM.xlsx`, `prepend_sync_regressor_TP.xlsx` | 3 MB | pre-WordNet 61-label matrices; make the 61 → 85 expansion checkable |
| `wordnet_sync_regressor_DM.xlsx`, `wordnet_prepend_sync_regressor_TP.xlsx` | 250 KB | the 85-feature × TR matrices, incl. the `Frequency` column |
| `movie_despicable_me_annot.xlsx`, `movie_present_annot.xlsx`, `label_list.txt` ×2 | 72 KB | human annotation + taxonomy (Supp. Fig. 2b) |
| `features_85.csv` | ~5 KB | derived: 85 labels + frequency, all P4/P5 need from the annotations |

**Watch out when assembling this bundle:**

- **`wordnet_sync_regressor_TP.xlsx` does not exist.** DM and TP take different
  code paths in `05_prepare_reg_2wordnet_gaze_weight.py`: DM reads
  `sync_regressor_DM.xlsx`, TP reads `prepend_sync_regressor_TP.xlsx` (TP has 7
  prepended sync frames). The TP file to ship is
  **`wordnet_prepend_sync_regressor_TP.xlsx`**. A `wordnet_sync_regressor_*`
  glob silently ships DM only.
- **Exclude `~$wordnet_prepend_sync_regressor_TP.xlsx`** (165 B), an Excel lock
  file left in the TP directory.
- **`real_xy` in the pickle is all zeros** for movie runs — a DeepMREye
  placeholder, since there is no gaze ground truth during movie watching. Kept
  as-is by decision; documented here so it is not mistaken for measured
  eye-tracking.
- Ship only the `bias-0.9_verb-1.0` cell. The other 18 `(bias, verb)`
  combinations (~1.6 GB) belong to the parameter search, which is not a
  manuscript result.
- The pre-WordNet gaze-weighted regressors (`gaze_weighted_regressor_…`, 57 MB)
  are deliberately **not** shipped: the two files that bracket them are.

## 6. Normative-model and subtyping tables — standalone P7 / P8

`2_pipeline/99_main/11_sem_zscore/out/.../screen-True_overlap-False/`
— `{sem,dim}_data_{td,asd}_raw.csv`, `zscore_{td,asd}.csv`,
`all_sem_dim_zscore.csv`

`2_pipeline/99_main/12_heterogeneity/out/screen-True_overlap-False/cluster-kmeans_profile/`
— `cluster_assignments_profile.csv`, `sem_input_{pooled,subtype-0,subtype-1,asd_only,all_with_cluster}.csv`,
`subtype_model1_{fit_measures,paths_std,defined_effects,multigroup_fit}.csv`,
`subtype_sample_summary.csv`

< 5 MB. Publishing these lets P7 and P8 be checked without running P4/P5 at all,
and they are the files the reported SEM and subtype statistics were read from.

Note the parameter branch: three other `screen-*_overlap-*` combinations exist in
the working tree and are **not** the published analysis. Publish only
`screen-True_overlap-False`.

---

## Not published

| | Why |
|---|---|
| Raw and preprocessed fMRI | HBN Data Use Agreement |
| Movie frames and MP4s | Copyrighted; 51 GB |
| Polygon binary masks | ~24 GB per movie (96 labels x 750 frames); pixel-level derivatives of copyrighted frames |
| Per-frame gaze weight maps | ~180 GB (494 MB x 246 subjects x 2 movies); regenerable from the gaze pickle plus the frames |
| Other 18 (bias, verb) grid cells | ~1.6 GB; parameter search, not a manuscript result |
| `regressor_DM_moten.xlsx` | 106 MB; superseded by `DM_motion_energy.xlsx` |
| Vertex-level encoding weights | ~20 GB; superseded by the PR arrays |
| Per-parcel similarity cache | 4.9 GB; group averages suffice |
