# Demo: P4 semantic axis on simulated data

The real inputs to the pipelines (per-subject encoding weights, the participants
table) are data about individual Healthy Brain Network participants and cannot be
redistributed. This demo runs the P4 semantic-axis analysis
(`pipelines/p4_semantic_axis/07_pca.py`, manuscript Fig. 2) end to end, unmodified,
on a small simulated dataset with a known group difference planted in it.

Nothing in the simulated data is derived from participant data. The only real
inputs are two public files from the Zenodo deposit, shipped in `demo/inputs/`:

| File | Contents |
|---|---|
| `features_85.csv` | the 85 WordNet feature labels and their frequencies in the movie |
| `mask_fdr-0.01.npy` | the encoding-performance mask (291 of 360 parcels) |

## What is simulated

`make_demo_data.py` writes a project directory with the same file names, layout,
shapes and column schema as the real one:

- 30 TD and 30 ASC participants across two sites. Two are set to fail quality
  control (one for head motion, one for a preprocessing remark), which shows the
  filters working. **29 TD and 29 ASC remain.**
- Encoding weights per subject (4 delays × 85 features × 360 parcels), built from
  three latent semantic components plus a site offset and noise.
- A **planted group effect** in 30 parcels inside the performance mask. In ASC,
  the first component is weakened to 20% in 15 parcels and strengthened in the
  other 15. The strengthening factor keeps the map's total sum of squares
  unchanged, so that the pipeline's z-scoring across parcels does not shift the
  remaining parcels. The parcels and directions are saved as
  `planted_effect_parcels.npy` (360-parcel indices) and
  `planted_effect_direction.npy` (−1 weakened, +1 strengthened).

The generator is seeded (`--seed 0`), and both it and the pipeline are
deterministic. Repeated runs, and runs in the original analysis environment and in
a fresh install from `environment/`, give byte-identical outputs. The only
exception is the creation timestamp that `scipy.io.savemat` writes into the
`.mat` header.

## Run it

From the repository root, in the environment from the main README's
[Quick start](../README.md#quick-start), with the NLTK WordNet data downloaded:

```bash
python demo/make_demo_data.py            # writes demo/project/ (about 60 MB)

# bash
ASC_PROJECT_ROOT=$PWD/demo/project python pipelines/p4_semantic_axis/07_pca.py \
    --skip-surface-figures --out-root demo/output

# Windows PowerShell
$env:ASC_PROJECT_ROOT = "$PWD\demo\project"
python pipelines/p4_semantic_axis/07_pca.py --skip-surface-figures --out-root demo/output
```

`ASC_PROJECT_ROOT` points the pipeline at the simulated project instead of
`roots.project` in `config/paths.yml`. `--out-root` keeps the results out of the
project directory. `--skip-surface-figures` skips the VTK surface rendering, which
does not affect any numeric output. Some matplotlib panels are still written under
`figures/p4_semantic_axis/`.

On Windows, keep the repository path short (under about 80 characters). The
output directory encodes the analysis configuration, and a deeply nested checkout
exceeds the 260-character path limit (`WinError 206`).

## Expected output

The console ends with:

```
  TD aligned: 29
  ASD aligned: 29
...
  Number of significant regions: 29
...
Analysis complete!
```

Results are written to
`demo/output/07_pca/out/default+me/mmp/chunk-9/fold-avg/results/k-0/bias-0.9_verb-1.0/wn-True/seed-0/s_alpha-True/sem/mask_fdr-0.01/delay-Avg_super-True/`:

| File | Shape | Contents |
|---|---|---|
| `sem_tpl.npy`, `sem_tpl_loading.npy` | (291, 10), (85, 10) | group PCA template and feature loadings |
| `sem_list_td.npy`, `sem_list_asd.npy` | (29, 291, 10) each | per-subject aligned PC scores |
| `avg_sem_td.npy`, `avg_sem_asd.npy` | (291, 10) each | group-mean PC scores |
| `tstat_…npy`, `pval_…npy` | (360,) | TD-vs-ASC GLM on PC1, per parcel |
| `sig_regions_…npy` | (360,) | −1 / 0 / +1: FDR-significant parcels and direction |
| `sem_Whole_sign-True_iter-10.mat` | | GLM design and PC scores |

**29 parcels are significant, 28 of which are planted** (14 of 15 weakened and 14
of 15 strengthened). That leaves 2 planted parcels missed and 1 false positive.
Check it with:

```python
import numpy as np
from pathlib import Path
out = next(Path('demo/output').rglob('sig_regions_*.npy'))
sig = np.load(out)
planted = np.load('demo/project/planted_effect_parcels.npy')
print('significant:', np.count_nonzero(sig),
      '| planted recovered:', np.count_nonzero(sig[planted]), 'of', len(planted))
```

In the planted parcels, the ASC group-mean PC1 is about 0.2× (weakened) and 1.3×
(strengthened) the TD mean in magnitude.

## Run time

Measured in a fresh install of the environment on Windows 11, Intel Core
i9-10900 (10 cores), 96 GB RAM, local SSD:

| Step | Time | Peak memory |
|---|---|---|
| `make_demo_data.py` | 8 s | |
| `07_pca.py --skip-surface-figures` | 27 s | 0.53 GB |

No GPU is needed. With the project directory on a network drive, both steps are
slower (about 40 s and 1 min here) because of the file I/O.
