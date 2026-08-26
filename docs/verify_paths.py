"""
Check that the shared configuration resolves to the paths the published scripts
used.

The distributed pipeline scripts were migrated off hard-coded storage roots
(``S:/``, ``Q:/``, ``V:/``, ``/MIPL/store7``) onto ``config/paths.yml``.  That
migration is only safe if, under the default configuration, every path resolves
exactly as before - otherwise a "reproduction" silently reads or writes somewhere
new.  This script asserts that, without running any analysis.

    python docs/verify_paths.py

Exit status is 0 if every check passes, 1 otherwise.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'lib'))

import project_config as pc  # noqa: E402


# The roots the original scripts hard-coded, on the machine that produced the
# published results (Windows branch).
ORIGINAL_STORE7 = Path('S:/')
ORIGINAL_PROJECT = ORIGINAL_STORE7 / 'jelee' / '02_asd_semantic_map'
ORIGINAL_TEMPLATE = ORIGINAL_STORE7 / 'jelee' / 'template'

# (label, resolved-by-config, path the original scripts built)
CHECKS = [
    ('project root',   pc.PROJECT,   ORIGINAL_PROJECT),
    ('raw data',       pc.RAW,       ORIGINAL_PROJECT / '0_data' / 'raw'),
    ('code tree',      pc.CODE,      ORIGINAL_PROJECT / '1_code' / '99_main'),
    ('pipeline out',   pc.PIPELINE,  ORIGINAL_PROJECT / '2_pipeline' / '99_main'),
    ('nulls cache',    pc.SPATIAL_NULLS_CACHE,
     ORIGINAL_PROJECT / '2_pipeline' / '99_main' / '_spatial_nulls_cache'),
]

# Task directories, as the original CONFIG blocks named them.
TASK_CHECKS = [
    ('features',        '05_prepare_reg'),
    ('encoding_parcel', '06_encoding_model'),
    ('semantic_axis',   '07_pca'),
    ('encoding_vertex', '08_encoding_model_vertex'),
    ('dimensionality',  '09_dimensionality'),
    ('prediction',      '10_behavior_sharp'),
    ('normative_sem',   '11_sem_zscore'),
    ('subtyping',       '12_heterogeneity'),
]

# Template files, keyed as in paths.yml, against the original shared directory.
# EG17 is the exception: it was read from inside a conda site-packages tree, so
# there is no original path under the template root to compare against.
TEMPLATE_CHECKS = [
    ('mmp_32k_dlabel',
     'MMP/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'),
    ('mmp_10k_dlabel',
     'MMP/10k_fs_LR/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.10k_fs_LR.dlabel.nii'),
    ('medial_wall_32k', 'MMP/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'),
    ('mmp_sections',    'MMP/ResultsRegions_ROI.dlabel.nii'),
    ('mmp_roi_table',   'MMP/HCP_cortical_subcortical_379.xlsx'),
    ('sphere_l',        'MMP/Q1-Q6_RelatedParcellation210.L.sphere.32k_fs_LR.surf.gii'),
    ('brain_mask_32k_l',
     'neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii'),
    ('yeo17',
     'Yeo_JNeurophysiol11_FreeSurfer/32k_fs_LR/lh.Yeo2011_17Networks_order.label.gii'),
]


def main() -> int:
    failures = []

    print(f"repository : {pc.REPO_ROOT}")
    print(f"project    : {pc.PROJECT}")
    print(f"templates  : {pc.TEMPLATES}")
    print(f"figures    : {pc.FIGURES}")
    print()

    print("Resolved paths vs. the original hard-coded ones")
    print("-" * 68)
    for label, got, expected in CHECKS:
        ok = Path(got) == Path(expected)
        print(f"  [{'ok' if ok else 'XX'}] {label:<14} {got}")
        if not ok:
            failures.append(f"{label}: {got} != {expected}")
            print(f"       expected  {expected}")

    print()
    print("Task directories")
    print("-" * 68)
    for key, directory in TASK_CHECKS:
        got = pc.task_dir(key)
        expected = pc.PIPELINE / directory
        ok = got == expected
        print(f"  [{'ok' if ok else 'XX'}] {key:<16} -> {directory}")
        if not ok:
            failures.append(f"task {key}: {got} != {expected}")

    print()
    print("Figure output (per-pipeline, created on demand)")
    print("-" * 68)
    for pipeline in ('p2_features', 'p3_encoding_parcel', 'p4_semantic_axis'):
        probe = REPO / 'pipelines' / pipeline / '_probe.py'
        got = pc.fig_dir(str(probe), create=False)
        ok = got == pc.FIGURES / pipeline
        print(f"  [{'ok' if ok else 'XX'}] {pipeline:<20} -> figures/{got.name}")
        if not ok:
            failures.append(f"fig_dir {pipeline}: {got}")

    print()
    print("Vendored templates (existence + relative layout)")
    print("-" * 68)
    for key, relative in TEMPLATE_CHECKS:
        try:
            got = pc.template(key)
        except (KeyError, FileNotFoundError) as exc:
            print(f"  [XX] {key:<18} {exc}")
            failures.append(f"template {key}: {exc}")
            continue
        ok = got == (pc.TEMPLATES / relative)
        print(f"  [{'ok' if ok else 'XX'}] {key:<18} {relative}")
        if not ok:
            failures.append(f"template {key}: unexpected layout")

    print()
    if failures:
        print(f"FAILED - {len(failures)} check(s):")
        for failure in failures:
            print(f"  - {failure}")
        print()
        print("If roots.project in config/paths.yml points somewhere other than the")
        print("original S:/jelee/02_asd_semantic_map, the first block is expected to")
        print("differ - that is the point of the config. The task and template blocks")
        print("must still pass.")
        return 1

    print("All checks passed. Configuration resolves as the original scripts did.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
