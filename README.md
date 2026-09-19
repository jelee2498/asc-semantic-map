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
environment/    Python (conda + pip) and R (renv.lock) specifications
config/         paths.yml   - storage roots and file locations
                params.yml  - the manuscript configuration (see warning below)
templates/      Vendored atlases and surfaces, SHA-256 pinned (20 files, ~11 MB)
lib/            project_config.{py,R}, spatial_nulls.py, sharp_test.py,
                brainspace_ext.py (PCA and alignment as used for the paper)
pipelines/      One directory per analysis stage, each with its own README
demo/           Simulated dataset and instructions for running P4 end to end
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
# 1. Python (see System requirements below)
conda env create -f environment/environment.yml
conda activate asc-semantic-map
pip install --no-deps -r environment/requirements-nodeps.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('wordnet31')"

# 2. R  (R 4.1.0; on Windows, Rtools40 to compile packages)
Rscript -e 'install.packages("renv", repos = "https://cloud.r-project.org")'
Rscript -e 'renv::restore(lockfile = "environment/renv.lock", prompt = FALSE)'

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

## System requirements

- **Operating system.** Developed and tested on Windows 10/11 x64, which is the
  platform the published results were produced on. The Python code has no
  Windows-specific parts, but only Windows has been tested.
- **Software.** Python 3.8 through conda (Anaconda or Miniconda; conda 4.9
  tested) and R 4.1.0 for the SEM stages (P7, P8). Python package versions are
  pinned in `environment/requirements.txt`, R package versions in
  `environment/renv.lock`.
- **Hardware.** A standard desktop computer. No GPU or other non-standard
  hardware is needed. The demo peaks at about 0.5 GB of memory. The full P4 run
  on the real data takes about 3–4 minutes.
- **Windows paths.** Output paths encode the analysis configuration and are long.
  Keep the repository and project directories short (under about 80 characters)
  to stay inside the 260-character path limit.

## Installation

The Quick start above installs everything. The Python environment has three parts:

| File | Role |
|---|---|
| `environment/environment.yml` | Python 3.8.12 and the MKL stack, then `requirements.txt` |
| `environment/requirements.txt` | the packages the pipelines import, pinned to the versions used for the published results; dependencies held to published versions through the snapshot, used as a constraints file |
| `environment/requirements-nodeps.txt` | brainstat, installed with `--no-deps`, because its declared dependencies cannot be resolved together with the published pins (see the file) |
| `environment/requirements-lock.txt` | full snapshot of the original environment, for provenance only; it has conflicting pins and is not installable as a whole |

brainspace is the released 0.1.20. The analysis originally ran on a locally
modified copy. The modifications it depends on are in `lib/brainspace_ext.py`,
and the released package reproduces the published P4 outputs byte for byte (see
`pipelines/p4_semantic_axis/README.md`).

**Install time.** On a desktop with a broadband connection, the Python
environment took about 4 minutes from scratch: `conda env create` 3.7 min, the
`--no-deps` step and the NLTK data 10 s. The R library took about 4 minutes into
an empty library: installing renv 24 s, `renv::restore` 3.6 min. renv installs
33 packages: 28 are built from source, because the locked versions have no R 4.1
binaries (so Windows needs Rtools40), and 5 install as binaries. The other 6 locked
packages ship with R. Both were measured on Windows 11, Intel Core i9-10900,
96 GB RAM, on a broadband connection.

If your machine has packages in the per-user site-packages directory (`pip
install --user`), set `PYTHONNOUSERSITE=1` before creating and using the
environment. Otherwise pip treats those packages as already installed and they
can shadow the pinned versions.

## Demo

The per-subject inputs cannot be shared, so `demo/` provides a simulated dataset
(58 subjects after quality control, with a planted group difference) that runs the
P4 semantic-axis analysis end to end with the unmodified pipeline:

```bash
python demo/make_demo_data.py
ASC_PROJECT_ROOT=$PWD/demo/project python pipelines/p4_semantic_axis/07_pca.py \
    --skip-surface-figures --out-root demo/output
```

Expected output: 29 TD and 29 ASC subjects after quality control and 29
FDR-significant parcels, 28 of which are among the 30 planted ones. Expected run
time is about 10 s to generate the data and about 30 s for the analysis. See
[`demo/README.md`](demo/README.md) for the output files, a check against the
planted effect, and PowerShell syntax.

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
