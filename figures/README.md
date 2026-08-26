# Generated figures

**Empty by design.** Figures are reproducible outputs, not sources: the
repository ships the code that makes them, not the images. Run a pipeline and
this directory fills in.

One subdirectory per pipeline, created on demand by
`project_config.fig_dir(__file__)` (Python) or `asc_figures()` (R):

```
figures/
  p4_semantic_axis/     Fig. 2, Supp. Fig. 3-5
  p5_dimensionality/    Fig. 3
  p6_prediction/        Fig. 4
  p7_normative_sem/     Fig. 5, Fig. 6a-b
  p8_subtyping/         Fig. 6c-g, Supp. Fig. 11
```

Contents are git-ignored; the directory itself is tracked so a fresh clone has
somewhere to write. Set `roots.figures` in `config/paths.yml` to send figures
elsewhere.

## Only figures live here

Result tables and derived brain maps go to the pipeline output directory under
`2_pipeline/`, never here — so clearing this directory can never lose a result.

That was not true of the original: `12_heterogeneity_0subtype.py` wrote its
cluster assignments and four result CSVs into the figure tree, which made a
figures folder a pipeline dependency and meant a run modified the source
checkout. Those now go to `2_pipeline/99_main/12_heterogeneity/out/`.

## Rendering without VTK

The surface maps need `brainspace` and VTK. If that stack is unavailable, pass
`--skip-surface-figures` to `07_pca.py` or `09_dimensionality.py`. The flag
covers the VTK-dependent blocks only — some matplotlib panels and result tables
are still written — and numeric outputs are identical either way.

Scripts fail with an explicit `ImportError` if the rendering stack is missing
and the flag was not given. The original silently skipped every figure section
on non-Windows platforms and still exited 0.
