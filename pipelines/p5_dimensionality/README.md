# P5 — Representational dimensionality

**Manuscript:** Fig. 3a–e; Methods *Encoding model, 4) Dimensionality analysis*.

**Runnable: yes**, and cheaply. The published entry point is the pair of
participation-ratio arrays — **708 KB** — not the ~20 GB of vertex-level
encoding weights they were computed from.

Scripts are the working-tree originals; changes are limited to path
configuration and the fixes under [Changes](#changes-from-the-working-tree).
Analysis code is untouched.

---

## Verification status

Reproduced on 2026-08-25, entering from the cached PR arrays alone.

| Output | Result |
|---|---|
| `cov_sig_mmp.npy` | byte-identical |
| `cov_stat_results.npy` | byte-identical |
| `cov_stat_results_wholebrain.npy` | byte-identical |

The run was seeded with **only** `cov_pr_td.npy` and `cov_pr_asd.npy` in the
output directory; everything else was recomputed. This is exactly the path a
reader takes after downloading the Figshare bundle.

| Value | Manuscript | Reproduced |
|---|---|---|
| PR array shapes | 108 TD / 138 ASC × 360 parcels | `(108, 360)` / `(138, 360)` |
| Significant parcels | 95 (the count the behaviour scripts consume) | 95 of 360 |
| Direction | "increase (but never decrease)" | **95 of 95 t > 0**, range 2.68–5.14 |
| Mean PR | ASC > TD | TD 6.496 ± 1.350, ASC 6.863 ± 1.451 |

---

## Usage

```bash
python 09_dimensionality.py                  # cached PR + GLM + figures
python 09_dimensionality.py --skip-surface-figures   # skip VTK surface rendering
python 09_dimensionality.py --recompute-pr   # full path; needs the vertex weights
python 09_dimensionality.py --out-root DIR   # write elsewhere, e.g. to verify
```

Eleven steps: load subjects → load atlases → **check for cached PR** → load
vertex weights *(skipped when cached)* → performance mask → compute PR
*(skipped when cached)* → BrainStat GLM → save → brain maps → network
aggregation → similarity heatmaps.

The GLM is `PR ~ Group + Site + Age + Sex + MeanFD`, FDR at
`glm_alpha = 0.025`.

### The two entry points

**Cached (default).** If `cov_pr_{td,asd}.npy` are present, Steps 4–6 are
skipped and the GLM runs from them. Seconds, ~1 GB of RAM.

**Full (`--recompute-pr`).** Recomputes PR from per-subject vertex weights —
`(340, 59412)` float32 ≈ 80 MB each, 246 subjects loaded up front, ~20 GB
resident. High-memory node only.

Both produce identical numbers; only the cached path is reproducible from
published data.

### Fig. 3a insets

The V1/V2/MT/TPOJ similarity-matrix insets read
`cov_sim_mmp_{td,asd}_avg_list.pkl` (2 × 21 MB) from the same directory. Without
them the run completes and the insets are skipped. Confirmed against the code:
these two files are what the insets need, and they are listed in
`docs/FIGSHARE_MANIFEST.md`.

---

## Inputs

| Input | Source |
|---|---|
| `cov_pr_{td,asd}.npy` | Figshare — **the entry point** |
| `participants_df_deepmreye_inc_byhx.xlsx` | Figshare |
| `mask_fdr-0.01.npy` (291 parcels) | Figshare (P3 output) |
| `cov_sim_mmp_{td,asd}_avg_list.pkl` | Figshare — Fig. 3a insets only |
| MMP atlas, medial wall, surfaces, Yeo, EG17 | `templates/` (vendored) |
| Ito & Murray multi-task dimensionality | `templates/external/` (Fig. 3b) |
| Vertex-level encoding weights (~20 GB) | **not published** — `--recompute-pr` only |

## Outputs

`{pipeline}/09_dimensionality/out/32k/default+me/det_results_inc_byhx_docu-True/k-50/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-False/`

| File | Contents | Consumed by |
|---|---|---|
| `cov_pr_{td,asd}.npy` | per-subject PR, `(n, 360)` | P6, P7 |
| `cov_stat_results.npy` | `t`, `p`, `p_fdr`, each `(360,)` | Fig. 3c |
| `cov_sig_mmp.npy` | boolean mask, 95 parcels | **P6, P7** |
| `cov_stat_results_wholebrain.npy` | same, outside the performance mask | |

Note `s_alpha-False` here — the vertex branch uses per-vertex alphas, whereas P4
reads `s_alpha-True`. Mixing them silently yields an empty directory.

---

## Changes from the working tree

1. **Storage roots → `config/paths.yml`** (`lib/project_config.py`).

2. **Explicit cache control instead of a platform proxy.** The original set
   ```python
   'overwrite': False if os.name == 'nt' else True
   ```
   using the operating system as a stand-in for "do I have the vertex weights".
   The practical effect was that **a rerun on Windows silently reused the cached
   PR and appeared to reproduce**, while the same command on Linux tried to load
   20 GB. Now `overwrite: False` with an explicit `--recompute-pr` flag.

3. **EG17 atlas: fail instead of silently dropping a panel.** Was `try/except`
   over `eg_atlas = None` with a fallback into one specific conda installation.
   On failure the run continued and **Fig. 3d quietly disappeared**. Now loads
   from `templates/WashU/EG17.mat` and raises.

4. **Surface rendering: capability, not platform.** The imports were gated on
   `os.name == 'nt'` *in addition to* an `ImportError` guard, so on Linux the
   brain-map sections were skipped even where brainspace and VTK were installed.
   Now detected properly, with `require_surface_plotting()` raising an
   actionable error.

5. **`spatial_nulls` import repointed** from the script's own directory to
   `lib/`.

6. **Figures → `figures/p5_dimensionality/`** instead of the source tree.

Two `print()` statements used `±`, which raises `UnicodeEncodeError` on a
non-UTF-8 console and aborts the script *after* the analysis has finished — the
same latent crash found in `07_pca.py`. Replaced with ASCII.

## Known issues

- The post-save stages (network aggregation, multi-task comparison, spatial
  nulls) take considerably longer than the analysis itself. The numeric outputs
  are written at Step 8, before them.
- `--out-root` redirects the numeric outputs; figures go to `figures/` either way.
- Long output paths exceed the Windows 260-character limit when the project root
  is deeply nested; keep `roots.project` short on Windows.
