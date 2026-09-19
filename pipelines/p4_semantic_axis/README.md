# P4 — Semantic axis (PCA)

**Manuscript:** Fig. 2a–i; Supp. Fig. 3, 4, 5; Supp. Table 1.

**Runnable: yes.** This is the first reproducible stage. It starts from the
encoding-model results (per-subject tier, on request to HBN DUA holders) rather
than from imaging data.

Scripts are the working-tree originals; changes are limited to path
configuration and the fixes listed under [Changes](#changes-from-the-working-tree).
The computations are unchanged. Change 6 moves the brainspace modifications the
analysis depended on into `lib/brainspace_ext.py` without changing any result.

---

## Verification status

Reproduced on 2026-08-25 against the published outputs, and again on 2026-09-20
with the **released** brainspace 0.1.20 from PyPI after the brainspace changes
were moved into `lib/brainspace_ext.py` (change 6 below), in a fresh environment
installed from `environment/` (isolated from any user site-packages). Both runs
give the results below. The 2026-09-20 run took 161 s with
`--skip-surface-figures`, with 0.84 GB peak memory.

| Output | Result |
|---|---|
| `sem_list_td.npy`, `sem_list_asd.npy` | byte-identical |
| `avg_sem_td.npy`, `avg_sem_asd.npy` | byte-identical |
| `sem_tpl.npy`, `sem_tpl_loading.npy` | byte-identical |
| `tstat_Whole_sign-True_iter-10.npy` | byte-identical |
| `pval_Whole_sign-True_iter-10.npy` | byte-identical |
| `sig_regions_Whole_sign-True_iter-10.npy` | byte-identical |
| `sem_Whole_sign-True_iter-10.mat` | all 8 arrays identical; file bytes differ only in the MAT header timestamp that `scipy.io.savemat` embeds |

Reported values reproduced exactly:

| Value | Manuscript | Reproduced |
|---|---|---|
| Cohort | 108 TD / 138 ASC | 108 / 138 |
| Performance mask | 291 of 360 parcels | 291; template `(291, 10)` |
| FDR-significant parcels | 33 | 33 |
| PC1 / PC2 / PC3 explained variance (Fig. 2a) | 40.83 / 15.4 / 9.21 % | 40.83 / 15.40 / 9.21 % |
| TD-vs-ASC PC1 map correlation (Supp. Fig. 3e) | r = 0.78, p_eigen < 0.001 | r = 0.778, spin p < 0.001, eigenstrap p < 0.001 |

The Supp. Fig. 3e nulls were computed from the cached spin permutations and
geometric eigenmodes (`_spatial_nulls_cache/`), which is what makes `p_eigen`
exactly rather than approximately reproducible.

---

## Order of execution

### 1. `07_pca.py` — main analysis (Fig. 2b–i)

```bash
python 07_pca.py                       # full run, including figures
python 07_pca.py --skip-surface-figures        # skip VTK surface rendering
python 07_pca.py --out-root /some/dir  # write elsewhere, e.g. to verify a run
```

Eleven steps: load participants → load encoding weights → load regressors →
preprocess weights (delay average, neuroCombat site harmonisation, label-frequency
regression, performance mask, WordNet superordinates, z-score) → PCA template →
Procrustes alignment per subject → save semantic spaces → BrainStat GLM →
FDR-significant regions → save → figures.

The GLM is `PC score ~ Group + Site + Age + Sex + MeanFD`, FDR at
`glm_alpha = 0.025` two-tailed (i.e. FDR < 0.05).

Runtime ≈ 4 min with `--skip-surface-figures` on a workstation; the figure stage is
considerably slower.

### 2. `07_pca_significance.py` — Fig. 2a

Compares the explained variance of the semantic-weight PCs against PCs of the raw
stimulus matrix (paired t-test, Bonferroni). This is what establishes that the
first three components carry structure beyond the stimulus.

The script saves only the figure. Verified on 2026-09-20: the version at commit
46708e0, run on the modified brainspace, and the current version, run on
released brainspace 0.1.20, produce identical explained-variance ratios
(246 × 85, stimulus and semantic), t-values and corrected p-values (max |diff|
= 0). Semantic PC1–3 are 40.83 / 15.40 / 9.21 %, as reported.

### 3. `07_pca_global.py` — Supp. Fig. 3

Whole-cortex TD-vs-ASC comparison of the PC1 map, with an
eigenstrapping/spin null for the spatial correlation (published r = 0.78).

Slow: 2000 surrogates. Ship the cached surrogates
(`_spatial_nulls_cache/`, Zenodo) to reproduce `p_eigen` exactly rather than
only statistically — see `docs/STAGE_A_NOTES.md` §3.3.

### Supplementary components

Supp. Fig. 4 and 5 are `07_pca.py` rerun with `comp_no = 2` and `3` in
`CONFIG`. Neither shows a group difference, which is why the paper reports PC1
only.

---

## Inputs

| Input | Source |
|---|---|
| Fold-averaged encoding weights, 246 × (85 × 360) | per-subject tier, on request (P3 output) |
| `mask_fdr-0.01.npy` (291 parcels) | Zenodo (P3 output) |
| `participants_df_deepmreye_inc_byhx.xlsx` | per-subject tier, on request |
| 85 labels + occurrence frequency | Zenodo (`features_85.csv`) |
| MMP atlas, sections, surfaces, Yeo, EG17 | `templates/` (vendored) |
| Wang et al. (2023) semantic ratings, *Sci. Data* 10, 106 | `0_data/semantic_features/` (Fig. 2e) |

## Outputs

`{pipeline}/07_pca/out/default+me/mmp/chunk-9/fold-avg/results/k-0/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-True/sem/mask_fdr-0.01/delay-Avg_super-True/`

| File | Contents | Consumed by |
|---|---|---|
| `sem_list_{td,asd}.npy` | per-subject PC scores, `(n, 291, 10)` | P6, P7 |
| `avg_sem_{td,asd}.npy` | group-mean PC1 map | Supp. Fig. 3 |
| `sem_tpl.npy`, `sem_tpl_loading.npy` | PCA template and loadings | Fig. 2b–c |
| `tstat_…npy`, `pval_…npy` | group comparison, 360-space | Fig. 2f–h |
| `sig_regions_…npy` | 33 significant parcels | **P6, P7** |
| `sem_…iter-10.mat` | design + PC scores for the GLM | P7 |

---

## Changes from the working tree

1. **Storage roots → `config/paths.yml`** (`lib/project_config.py`). The
   original hard-coded `S:/`, `Q:/`, `V:/`, `/Volumes/*`, `/MIPL/store*`.

2. **EG17 atlas: fail instead of silently dropping a panel.** The original
   wrapped the load in `try/except` over `eg_atlas = None`, with a fallback path
   into one specific conda installation
   (`D:/Users/…/site-packages/cbig_network_correspondence/…`). If both failed,
   the run continued and **Fig. 2h quietly disappeared** — exit code 0, no
   warning. Now loads from `templates/WashU/EG17.mat` and raises if absent.

3. **Surface rendering: explicit dependency error instead of a platform gate.**
   The brainspace/VTK imports and both figure blocks were gated on
   `os.name == 'nt'`, so on Linux every figure section was skipped silently.
   Imports are now unconditional; `require_surface_plotting()` raises an
   `ImportError` naming the missing package and pointing at `--skip-surface-figures`.

4. **Removed the dead `'Angular Gyrus'` pathway level.** Commented out, absent
   from the module-level definition, and gated behind
   `include_expanded_networks: False`. It never entered the published Fig. 2g;
   removal is cosmetic and does not change results (confirmed: outputs remain
   byte-identical).

5. **Figures now write to `figures/p4_semantic_axis/`** instead of
   `{project}/1_code/99_main/claude_figures/07_pca/`. The original wrote
   generated output into the source tree, so a run dirtied the code checkout.
   Subdirectories are preserved (`brain/`, `wordcloud/`, `global/`,
   `pca_significance/`).

6. **Locally modified brainspace → `lib/brainspace_ext.py`.** The published
   analysis ran on a copy of brainspace 0.1.20 that had been edited in place, so
   `07_pca.py` failed on the released package (`GradientMaps` has no
   `only_sign` argument and no `loadings_` attribute). Worse, the released
   defaults would silently change the numbers. The edits were:
   - PCA on the input itself: no kernel (the edit changed the default from
     `'normalized_angle'` to `None`), and negative values kept when
     `sparsity=0` (released brainspace sets them to zero);
   - PCA feature loadings stored as `loadings_` (= `PCA.components_`);
   - `only_sign` Procrustes: inside the usual iterative alignment, a
     subject's component is sign-flipped only if its Pearson correlation with
     the reference component is negative **and** significant after FDR
     (Benjamini–Hochberg, q < 0.01, across the 10 components); otherwise it is
     left as is.

   `lib/brainspace_ext.py` reimplements exactly these with numpy, scikit-learn,
   scipy and statsmodels. `07_pca.py` (template, loadings, alignment) and
   `07_pca_significance.py` (explained variance) now call it. brainspace is
   still used for the diffusion-map option (which passes its kernel explicitly)
   and for surface rendering. Checked on the real data: all 246 aligned subject
   maps, the template and the loadings match the modified package with max
   |diff| = 0, and a full run under released brainspace reproduces every output
   in the table above.

Additionally, four `print()` statements used `✓`/`✗`/`⏭`, which raise
`UnicodeEncodeError` on a non-UTF-8 console (this run hit it on cp949) and abort
the script *after* the analysis has completed. Replaced with ASCII.

## Known issues

- Long output paths exceed the Windows 260-character limit when the project root
  is itself deeply nested; keep `roots.project` short on Windows.
- The figure stage contains many `try/except … print("Warning: …")` blocks that
  swallow errors. Fixed for the two cases above; the rest are untouched.
