# Vendored templates

Atlases, surfaces and reference maps required by the analysis pipelines.

These were originally read from a shared directory outside the project tree
(`S:/jelee/template`), which is why the published scripts contain paths that
point outside the repository. They are vendored here so the analyses are
self-contained. `SHA256SUMS.txt` pins every file; verify with:

```bash
cd templates && sha256sum -c SHA256SUMS.txt
```

Total: 20 files, ~11 MB.

## MMP/ — HCP Multi-Modal Parcellation (Glasser et al. 2016, fsLR)

| File | Used by | Purpose |
|---|---|---|
| `Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii` | P4, P5, P0 | 360-parcel atlas; the analysis unit throughout |
| `10k_fs_LR/…10k_fs_LR.dlabel.nii` | P5 (10k branch) | 10k variant; unused at the published `resolution: 32k`, kept so the branch runs |
| `Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii` | P5, P0 | medial-wall exclusion (64,984 → 59,412 vertices) |
| `ResultsRegions_ROI.dlabel.nii` | P4, P5 | 22 MMP section groupings; the lollipop-plot axis |
| `HCP_cortical_subcortical_379.xlsx` | P4, P5, P7 | parcel index → ROI-name table |
| `Q1-Q6_RelatedParcellation210.{L,R}.sphere.32k_fs_LR.surf.gii` | P0 | parcel centroids for spin permutations |
| `Q1-Q6_RelatedParcellation210.{L,R}.midthickness_…surf.gii` | P0 | geometry for eigenstrapping surrogates (`es_surface: midthickness`) |
| `Q1-Q6_RelatedParcellation210.{L,R}.very_inflated_…surf.gii` | P4, P5 | surface rendering |
| `Q1-Q6_RelatedParcellation210.sulc_…dscalar.nii` | P4, P5 | sulcal-depth shading underlay |

## neuromaps-data/atlases/fsLR/

`tpl-fsLR_den-{10k,32k}_hemi-{L,R}_desc-nomedialwall_dparc.label.gii` — non-medial-wall
vertex masks, from the `neuromaps` data distribution.

Vendoring these four small files removes the `neuromaps` **package** from the
dependency set entirely: no pipeline imports it, only its data is used.

## Yeo_JNeurophysiol11_FreeSurfer/32k_fs_LR/

`lh.Yeo2011_{7,17}Networks_order.label.gii` — Yeo 7- and 17-network parcellations.

Left hemisphere only, by design. The scripts mirror it (`np.r_[lh, lh]`) on the
assumption of inter-hemispheric symmetry, so the right-hemisphere files are never
read and are not shipped.

## WashU/

`EG17.mat` — Gordon 17-network parcellation, from the `cbig_network_correspondence`
package. Supplies the network profiling in Fig. 2h and Fig. 3d.

Relocated here from a package-internal path. The published scripts try the shared
template directory first and fall back to an absolute path inside one specific
conda installation; see `docs/STAGE_A_NOTES.md`.

## external/

`analysis1_parcel_cosine_dimensionality_groupavg.csv` — multi-task fMRI
dimensionality (Ito & Murray), the external comparison map in **Fig. 3b**
(r = 0.25, p_eigen < 0.05). Extracted from the `multitaskrepresentations`
repository, the only file needed from it. 360 rows, one per MMP parcel.

## Provenance and licensing

The HCP-MMP atlas and associated surfaces are distributed under the HCP Open
Access Data Use Terms; the Yeo and Gordon parcellations and the multi-task
dimensionality map under their respective original licences. Redistribution here
is for reproduction of the published analyses. Confirm each upstream licence
before the repository is made public.
