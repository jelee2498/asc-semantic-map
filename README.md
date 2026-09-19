# Cortex-wide mapping of visual semantic representation in autism

Code for Lee et al., *How do children with autism perceive the
external world?: Cortex-wide mapping of visual semantic representation*.

> **Status.** All main-analysis pipelines (P2–P8) are migrated. P4, P5, P7 and
> P8 reproduce the published results; see
> [Reproduction status](#reproduction-status). Replication and specificity
> analyses are not yet included.

---

## What this repository reproduces

The analyses map whole-brain semantic representations from movie-watching fMRI in
246 children (108 typically developing, 138 autistic) from the Healthy Brain
Network, using a gaze-informed encoding model.

Reproduction starts from the **encoding-model results**, not from raw imaging.
Neither the fMRI data nor the movie stimuli can be redistributed: the imaging and
phenotypic data are governed by the HBN Data Use Agreement, and the film frames
are copyrighted (51 GB of stimulus assets, in any case). The encoding weights are
the first compact point in the pipeline, and everything downstream of them runs
from this repository. Group-level results are public on Zenodo
(<https://doi.org/10.5281/zenodo.22846400>); the per-subject derived data (encoding weights and
the like) are data about individual subjects under the HBN Data Use Agreement and
are available on request to DUA holders — see [Data availability](#data-availability).

```
  raw fMRI  ──►  preprocessing  ──►  gaze-weighted   ──►  ENCODING MODEL
  (HBN DUA)      (fMRIPrep)         features              (published)
                                    (copyrighted)               │
                                                                ▼
                        semantic axis ──► dimensionality ──► prediction
                              │                 │
                              └────────┬────────┘
                                       ▼
                            normative model + SEM ──► subtyping
```

Stages left of the encoding model are documented in the manuscript Methods and
their code is retained for provenance, but they are not runnable here.

## Repository layout

```
environment/    Python (conda + pip lock) and R (renv.lock) specifications
config/         paths.yml   - storage roots and file locations
                params.yml  - the manuscript configuration (see warning below)
templates/      Vendored atlases and surfaces, SHA-256 pinned (20 files, ~11 MB)
lib/            project_config.{py,R}, spatial_nulls.py, sharp_test.py
pipelines/      One directory per analysis stage, each with its own README
figures/        Generated figures, one subdirectory per pipeline (git-ignored)
docs/           Stage notes, derived-data manifest, path verification
```

Stages upstream of the encoding model (`pipelines/p2_features/`,
`pipelines/p3_encoding_parcel/`) are published for inspection but cannot be run
here — their inputs are restricted or copyrighted. They are included because the
Methods describe those steps in detail, and the description should be checkable
against the implementation. See `pipelines/README.md`.

## Quick start

```bash
# 1. Python
conda env create -f environment/environment.yml
conda activate asc-semantic-map

# 2. R  (R 4.1.0)
Rscript -e 'renv::restore(lockfile = "environment/renv.lock")'

# 3. Verify the vendored templates
cd templates && sha256sum -c SHA256SUMS.txt && cd ..

# 4. Download the public derived data from Zenodo (132 MB) and, if you hold an
#    HBN DUA, the per-subject tier (on request). Extract both into your project
#    directory - they mirror the project layout. See
#    docs/DERIVED_DATA_MANIFEST.md

# 5. Point roots.project in config/paths.yml at that directory
```

`brainiak` (0.11, used for the shared response model) and `fastsrm` are **not**
required. They belong to the SRM and preprocessing stages, which are not
reproduced here — worth knowing, as brainiak
is the most difficult dependency in this stack to build.

## The manuscript configuration

`config/params.yml` holds the parameter values that define the published
analysis. They are not tuning knobs.

Every script encodes these values into its **output directory path**. Changing
one does not raise an error — it silently writes to a different directory and
reads back nothing. A reproduction attempt that quietly produced no output has
been the most common failure mode in this codebase. Change a value only when you
intend to run something other than the published analysis.

## Reproduction status

| | Pipeline | Manuscript | Runnable | Status |
|---|---|---|---|---|
| P0 | Foundation: environments, templates, nulls | — | — | **complete** |
| P2 | Gaze-informed feature construction | Fig. 1a–b, Supp. Fig. 2 | no | published for inspection |
| P3 | Parcel-level encoding + performance mask | Fig. 1b–d | no | published for inspection; group outputs on Zenodo |
| P4 | Semantic axis (PCA) | Fig. 2, Supp. Fig. 3–5 | yes | **verified** — outputs byte-identical |
| P5 | Representational dimensionality | Fig. 3 | yes | **verified** — outputs byte-identical |
| P6 | Symptom prediction | Fig. 4 | yes | migrated; results cross-checked, not re-run |
| P7 | Normative model + SEM | Fig. 5, Fig. 6a–b, Supp. Fig. 8–10 | yes | **verified** — Fig. 5 values reproduced |
| P8 | Neurosubtyping | Fig. 6c–g, Supp. Fig. 11 | yes | **verified** — outputs byte-identical |

Raw-data screening, fMRIPrep and Ciftify (P1) are not included: standard
third-party preprocessing over data that cannot be shared. Versions and
parameters are in the manuscript Methods.

Replication and specificity analyses (matched subsamples, the second movie, the
independent McGill cohort, sex interaction, comorbidity, gaze validation) are
scheduled after the pipelines above.

## Data availability

- **Imaging and phenotypic data**: Healthy Brain Network
  (<https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/>), under
  its Data Use Agreement. The independent validation cohort is held at McGill
  University under a separate agreement.
- **Derived data, public**: Zenodo, <https://doi.org/10.5281/zenodo.22846400> — the
  performance mask and group encoding accuracy, dimensionality group statistics,
  normative trajectories, SEM and subtype model fits, the spatial-null cache, and
  the stimulus annotations and feature matrices. See `docs/DERIVED_DATA_MANIFEST.md`.
- **Derived data, per-subject**: encoding-model weights and test correlations,
  participation-ratio arrays, DeepMREye gaze predictions, gaze-weighted
  regressors, and the phenotypic, QC and SEM input tables. These are data about
  individual subjects, which the CMI Biobank Data Use Agreement restricts; they
  are available to investigators holding a DUA on request to the corresponding
  authors.
- **Stimuli**: the films themselves are copyrighted and not redistributable, nor
  are the frame-level polygon masks derived from them. The frame-by-frame
  semantic annotations and the 85-feature matrices **are** public.

## Licence

Analysis code: MIT (see `LICENSE`). The atlases and surfaces vendored under
`templates/` are third-party and keep their own terms; see
`templates/MANIFEST.md`. Derived data is deposited separately and is subject to
the Healthy Brain Network and CMI Biobank access terms, not to the MIT licence.

## Citation

Citation details will be added on publication.
