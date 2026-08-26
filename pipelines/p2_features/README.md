# P2 — Gaze-informed semantic feature construction

**Manuscript:** Fig. 1a–b; Methods *Encoding model, 1) Gaze-weighted semantic
features*; Supp. Fig. 2.

**Runnable here: no** — the inputs are the preprocessed HBN fMRI (Data Use
Agreement) and the movie stimuli (copyrighted; 51 GB of frames, plus ~24 GB of
polygon masks per movie). The code is published because the Methods describe
these steps in detail — the Gaussian gaze kernel, the uncertainty scaling, the
WordNet expansion — and the description should be checkable against the
implementation.

**Its outputs are published**, so the stage it feeds (P3) has everything it
needs. See [Outputs](#outputs-consumed-downstream).

Scripts are the working-tree originals with **one change**: storage roots now come
from `config/paths.yml` via `lib/project_config.py`, instead of the hard-coded
drive letters and mount points the analyses were run with. All analysis code is
untouched. `python docs/verify_paths.py` checks that the configuration resolves to
the same paths the originals used.

---

## Order of execution

### 0. DeepMREye gaze inference — `deepmreye/`

`00_prepare_data_release11_0deepmreye_mask_inc_byhx.py`

Extracts the eye mask from each fMRI run, applies the pretrained DeepMREye model
to infer per-TR gaze position and uncertainty, and de-duplicates the runs the
model returns at doubled length (250 → 500 TRs, 750 → 1500 TRs).

Requires the `deepmreye` package. Standalone otherwise.

Produces **two** files that the whole downstream pipeline depends on:

| Output | Role |
|---|---|
| `results_hbn_td_asd_inc_byhx_edited.pickle` (76 MB) | per-TR gaze `pred_xy` + `uncertainty`, consumed by step 1 |
| `participants_df_deepmreye_inc_byhx.xlsx` | **the cohort definition** — read by every pipeline from P3 onward |

This script lives in the older `17_param_search` tree rather than `99_main`
because it was never refactored. It is nevertheless the canonical producer: the
copy of the pickle under `99_main/05_prepare_reg/out/` is byte-identical
(SHA-256 `eec72ae1…`).

The `Rating_deepmreye_movie{DM,TP}`, `Rating_recon_all` and `Remarks` columns it
writes are **manual visual-QC ratings**, not computed. They cannot be regenerated
from code, which is one reason the participants table is published as data.

### 1. Gaze weight maps — `05_prepare_reg_0save_gaze_weight_sub.py`

Per subject, per movie. Places a 2D Gaussian at each predicted gaze coordinate,
with width scaled by the model's own predicted error, and writes one weight map
per frame.

Parameters, from the script's `CONFIG` (these are not all in the Methods):

- kernel size = frame width — `DM: 720 px`, `TP: 1280 px`
- `sigma = sigma_factor × uncertainty`, `sigma_factor` = `DM: 30`, `TP: 45`
- uncertainty clipped to the 20th–80th percentile
- `bias_weight = 0` at this stage; the peripheral-vision floor is added in step 2
- gaze coordinates mapped from visual degrees to screen coordinates by a
  per-site, per-movie scaling table (`RU` and `CBIC` differ)

Submitted via `05_prepare_reg_0save_gaze_weight_pbs.pbs` (PBS array, one task per
subject).

### 2. Apply gaze weights to regressors — `05_prepare_reg_1apply_gaze_weight_sub.py`

Multiplies each frame's object polygon mask by that frame's gaze map to get a
per-feature scalar weight (Methods Eq. 1), rescales to `[0, 1 − bias]` and adds
the `bias_weight` floor (Eq. 2). Action features, which have no polygon mask,
receive a constant `verb_weight`.

Published run: `bias_weight = 0.9`, `verb_weight = 1.0` (see
`config/params.yml`), selected by a 19-cell grid search over
`(bias_weight, verb_weight)`. That search is not a manuscript result and is not
distributed.

Submitted via `05_prepare_reg_1apply_gaze_weight_pbs.pbs`.

### 3. WordNet expansion — `05_prepare_reg_2wordnet_gaze_weight.py`

Adds every superordinate category along each label's first WordNet hypernym path,
merges duplicate superordinates by `nanmean`, and prunes to the canonical label
set — taking the 61 directly-annotated categories to the **85 features** used
throughout the paper.

Requires the NLTK `wordnet31` corpus. Batch script, no PBS.

---

## Outputs consumed downstream

| Output | Consumer | Published |
|---|---|---|
| `participants_df_deepmreye_inc_byhx.xlsx` | P3, P4, P5, P6, P7, P8 | yes |
| `results_hbn_td_asd_inc_byhx_edited.pickle` | step 1 (and Ext. Data Fig. 4) | yes |
| `wordnet_gaze_weighted_regressor_…/bias-0.9_verb-1.0/{sub_id}_regressor_{DM,TP}.xlsx` | P3 | yes |
| 85 labels + occurrence frequency (`features_85.csv`) | P4, P5 | yes |

**These are published on Figshare** (≈ 202 MB; see `docs/FIGSHARE_MANIFEST.md`
§5). They describe which semantic labels occur in each TR and how strongly each
subject attended to them — a time-aligned semantic description of the films, not
the films themselves. The frames, the polygon masks and the per-frame gaze maps
stay unpublished.

So although this pipeline cannot be *run* without the stimuli and the imaging
data, its inputs to P3 are all available, and P4 and P5 need only
`features_85.csv` from it.

## Known issues

- The DeepMREye step is unrefactored: no `CONFIG` block, and it mixes eye-mask
  extraction, model inference, the TR de-duplication fix, and the QC-table build
  into one script.
- The gaze-kernel parameters in step 1 (`sigma_factor`, the percentile clipping,
  the per-site coordinate scaling) are in code but not in the Methods.
- Step 3 reads its canonical label set from the legacy `25_sensitivity` tree, a
  cross-tree dependency carried over from the original working directory.
