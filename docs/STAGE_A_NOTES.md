# Stage A — foundation notes

What was captured, what was verified, and what must be fixed before the
downstream pipelines can be reproduced.

---

## 1. Environments

### Python

`fastsrm` conda environment: **Python 3.8.12**, Windows x64, conda 4.9.2.
Only 19 packages came from conda channels (interpreter, pip, MKL stack); the
other 255 came from pip. Captured as `environment/environment.yml` +
`environment/requirements-lock.txt` (267 pinned packages).

Key versions: numpy 1.22.0, scipy 1.10.1, pandas 1.5.3, scikit-learn 1.3.2,
nibabel 5.2.1, nilearn 0.10.2, statsmodels 0.14.1, brainspace 0.1.20,
brainstat 0.4.2, neuroCombat 0.2.12, eigenstrapping 0.1, vtk 9.1.0.

**`conda env export` does not work on this environment.** It aborts with
`InvalidVersionSpec: Invalid version '0.18.0numpy>=1.12scikit.learn>=0.23…'` —
a malformed requirement string in an installed package's metadata (the missing
separators point at `eigenstrapping`). The lock was therefore built from
`conda list` plus `pip list --format=freeze`. Anyone regenerating it must use
the same route.

Two packages were dropped from the lock because they are not importable by any
pipeline here and would break the install:

- `neuromaps==0.0.3rc1+2.g185444b` — a git-describe version with no PyPI
  release. Only its **data** is used, and those four files are now vendored.
- `fastsrm==0.0.4` — conda-only, and belongs to the SRM/preprocessing stages.

**`brainiak` is not installed in this environment at all**, and no in-scope
script imports it. Since brainiak is the hardest dependency in this stack to
build, it is worth stating in the README that reproducing P4–P8 does not need it.

### R

**R 4.1.0 (2021-05-18)**, Windows x64. Packages split across the system library
and a user library at `C:/Users/er/Documents/R/win-library/4.1`, which is why the
SEM scripts carry the note *"Requires: lavaan (user library). Run with
`R_LIBS_USER` set, or from RStudio."*

Versions that matter: **lavaan 0.6-16**, **mgcv 1.8-35**, semhelpinghands 0.1.9,
ggplot2 3.4.2, dplyr 1.1.2.

`renv` is **not installed**, so `environment/renv.lock` was written directly: the
recursive `Depends`/`Imports`/`LinkingTo` closure of the five packages the
in-scope R scripts load, minus base priority packages — 38 entries, validated as
JSON against the renv lockfile schema.

---

## 2. Templates

20 files, ~11 MB, vendored under `templates/` with SHA-256 checksums
(`sha256sum -c SHA256SUMS.txt` passes). Contents and provenance:
`templates/MANIFEST.md`.

The published scripts read these from `S:/jelee/template`, a shared directory
**outside the project tree**. That is the single largest portability problem in
the codebase and is what `config/paths.yml` exists to fix.

---

## 3. Issues found during Stage A

### 3.1 Hard-coded conda path with no fallback — blocks P7

`11_sem_zscore_0save_raw.py:152`

```python
'eg17_atlas': Path('D:/Users/jongeun/anaconda3/envs/fastsrm/Lib/site-packages/'
                   'cbig_network_correspondence/data/atlases/fs_LR_32k/WashU/EG17.mat'),
```

This is the **only** path tried. It hard-codes a drive letter, a user name, a
conda installation and an environment name, so it fails on any other machine —
including this one after an environment rename. P7 cannot run anywhere else until
it is repointed at `templates/WashU/EG17.mat`.

The same absolute path appears in `07_pca.py:347` and
`09_dimensionality.py:860`, but there it is only a **fallback** behind the shared
template directory, so those two currently work. All three should be repointed.

### 3.2 Silent-skip on EG17 load failure — same class as the VTK issue

In `07_pca.py` and `09_dimensionality.py` the EG17 load is wrapped in
`try/except` over `eg_atlas = None`. If it fails, execution continues with
`eg_atlas = None` and the EG17 network panels — **Fig. 2h** and **Fig. 3d** —
are quietly dropped. No warning, non-zero-length output, exit code 0.

This is the same failure mode as the VTK/brainspace skip you asked to convert
into an explicit dependency error, and should be fixed the same way when we reach
P4/P5.

### 3.3 eigenstrapping seed is inert

`SurfaceEigenstrapping(seed=N)` accepts the argument but never uses it:
`rotate_modes(gen=True)` calls `rotate_matrix(M)` without passing the instance
RNG, so it falls through to `check_random_state(None)` — NumPy's **global**
RandomState.

Consequences, all of which the repository has to account for:

1. `lib/spatial_nulls.py` works around it by seeding the global RNG itself.
   **The repository must expose `spatial_nulls.py`, not the library** — anyone
   calling eigenstrapping directly with `seed=` gets different p-values than the
   paper reports.
2. The workaround is process-local. Under `n_jobs > 1` each joblib worker gets
   its own global state and reproducibility is lost. Hence
   `spatial_nulls.es_n_jobs: 1` is pinned in `config/params.yml`, not merely
   defaulted. (Measured on 20 cores: parallelism peaks at ~1.5× and then degrades
   from BLAS oversubscription, so nothing is lost by pinning it.)
3. Even with 1 and 2, exact reproduction depends on the NumPy RNG implementation.
   To make the published `p_eigen` values reproduce **exactly** rather than
   merely statistically, ship the cached surrogates
   (`2_pipeline/99_main/_spatial_nulls_cache/`, 51 MB: two eigenmode `.npz` files
   plus `spin_perm_id_mmp360_n2000_seed0.npy`).

`eigenstrapping==0.1` is pinned for the same reason and must not be upgraded
casually. Its availability on PyPI at that exact version has **not** been
verified — worth confirming before release, since the environment may need a
vendored copy or a git pin.

### 3.4 Cluster assignments live in a figure directory — affects P8

`12_heterogeneity_1save_data.py` reads its cluster labels from

```
1_code/99_main/claude_figures/12_heterogeneity/screen-True_overlap-False/cluster_assignments_profile.csv
```

A figure directory is acting as a pipeline data dependency. It should move to
`2_pipeline/99_main/12_heterogeneity/out/` when we do P8.

---

## 4. Verified against the manuscript during Stage A

| Claim | Source | Status |
|---|---|---|
| Subtypes n = 58 / 70 | `cluster_assignments_profile.csv` | matches Fig. 6c |
| SEM fit χ²=81.090, df=52, RMSEA=0.0509, CFI=0.9802 | `subtype_model1_fit_measures.csv` (pooled row) | matches Fig. 5 |
| Subtype-1 fit χ²/df=1.566, RMSEA=0.0614, CFI=0.965 | same, subtype-0 row | matches Fig. 6g |
| Subtype-2 fit χ²/df=1.429, RMSEA=0.0513, CFI=0.983 | same, subtype-1 row | matches Fig. 6g |
| RRB prediction r=0.461, R²=0.212 | `model_comparison_statistics.csv` | matches Fig. 4a |
| Perf mask = 291/360 parcels | `mask_fdr-0.01.npy` | matches Methods (80.8%) |

Two open discrepancies, both for the manuscript rather than the code:

- **Fig. 4a social communication**: the results file gives combined r = 0.462,
  the manuscript text says r = 0.432 (R² = 0.213 matches). Worth checking which
  number the figure plots.
- **Subtype Ns**: clustering uses 128 ASC, the subtype SEM refits use 119
  (53 / 66) after complete-case filtering on the 9 behavioural indicators. A
  one-line Methods note resolves it.

## 5. Withdrawn concern

The commented-out `'Angular Gyrus'` block in `07_pca.py:3863` was **already
inactive**: it is commented out, absent from the module-level pathway definition
at line 2221, and gated behind `include_expanded_networks: False` regardless.
It never entered the published Fig. 2g. Deleting it is cosmetic cleanup with no
effect on results.
