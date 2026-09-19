# Pipelines

One directory per analysis stage. Each carries a README giving its manuscript
target, execution order, inputs, outputs, and known issues.

Scripts are the working-tree originals, with storage roots migrated to
`config/paths.yml` (via `lib/project_config.py`). The analysis code itself is
untouched; the published results were produced by these routines. Run
`python docs/verify_paths.py` to confirm the configuration resolves to the same
paths the originals used.

| | Stage | Manuscript | Runnable | Status |
|---|---|---|---|---|
| P2 | `p2_features/` — gaze-informed feature construction | Fig. 1a–b, Supp. Fig. 2 | no | published for inspection |
| P3 | `p3_encoding_parcel/` — encoding model, performance mask | Fig. 1b–d | no | published for inspection; **group outputs on Zenodo** |
| P4 | `p4_semantic_axis/` — semantic axis (PCA) | Fig. 2, Supp. Fig. 3–5 | yes | **verified** — outputs byte-identical |
| P5 | `p5_dimensionality/` — representational dimensionality | Fig. 3 | yes | **verified** — outputs byte-identical |
| P6 | `p6_prediction/` — symptom prediction | Fig. 4 | yes | migrated; results cross-checked, not re-run |
| P7 | `p7_normative_sem/` — normative model + SEM | Fig. 5, Fig. 6a–b, Supp. Fig. 8–10 | yes | **verified** — Fig. 5 values reproduced |
| P8 | `p8_subtyping/` — neurosubtyping | Fig. 6c–g, Supp. Fig. 11 | yes | **verified** — outputs byte-identical |

## Why P2 and P3 are here but not runnable

Their inputs cannot be redistributed — the imaging data is governed by the HBN
Data Use Agreement and the movie stimuli are copyrighted. Their code is published
anyway, because the manuscript Methods describe these steps in detail (the gaze
kernel, the WordNet expansion, the ridge fit with noise-ceiling correction, the
FDR performance mask) and a reader should be able to check the description
against the implementation rather than take it on trust.

P3's outputs *are* published, which is what makes P4 onward reproducible: the
encoding weights are the first point in the pipeline that is both compact and
free of restricted or copyrighted content.

## Not included

**P1 — raw-data screening, fMRIPrep and Ciftify.** Excluded by decision: standard
third-party preprocessing over data that cannot be shared. The versions and
parameters are given in the manuscript Methods.

The one exception is `04_prepare_fmri.py` (denoising, parcellation,
cross-validation folds), which is filed under `p3_encoding_parcel/` because the
Methods describe its CV scheme and P3 reads its output directly.

## Deferred

Replication and specificity analyses — matched subsamples (Ext. Data Fig. 5–6),
the second movie (Ext. Data Fig. 2, 7), the independent McGill cohort (Ext. Data
Fig. 3, 8), sex interaction (Supp. Fig. 6–7), comorbidity (Supp. Fig. 12), and
gaze validation (Ext. Data Fig. 1, 4) — are scheduled after P4–P8.
