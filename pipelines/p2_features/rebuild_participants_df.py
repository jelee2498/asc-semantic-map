"""
Rebuild the phenotypic tables from your own HBN LORIS access.

The published data archive deliberately omits every table containing per-subject
phenotypic values. Those come from the HBN LORIS release, which is governed by a
Data Use Agreement, so they cannot be redistributed. Everything *else* in the
archive is derived from the openly-released FCP-INDI imaging and is published.

This script regenerates the omitted tables from a LORIS release you already have
access to, reproducing the exact columns and dtypes the pipelines expect.

    python rebuild_participants_df.py

Set ``roots.hbn_phenotype`` in ``config/paths.yml`` to the directory containing
``HBN/Phenotypic_LORIS/`` first.

What this rebuilds
------------------
    participants_df_deepmreye_inc_byhx.xlsx   cohort definition + GLM covariates
    nih_toolbox_scores.csv                    SEM cognitive-ability indicators

What it cannot rebuild
----------------------
The three manual visual-QC columns - ``Rating_recon_all``,
``Rating_deepmreye_movieDM``, ``Rating_deepmreye_movieTP`` - and ``Remarks``.
These are human ratings, not derived quantities: they exist nowhere in LORIS and
cannot be recomputed. They ARE published, in ``qc_ratings.csv`` in the archive,
because they contain no HBN phenotypic data. This script merges them back in.

Downstream tables
-----------------
The per-subject tables in the P7/P8 chain (``all_sem_dim_zscore.csv``,
``sem_input_*.csv``, ``zscore_*.csv``, ``{sem,dim}_data_*_raw.csv``) embed
phenotypic values and are omitted for the same reason. They are not rebuilt
here: run the P7 chain and they are regenerated from the tables above.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PIPELINE, HBN_PHENOTYPE, require  # noqa: E402


# The four NIH Toolbox percentile scores used as the SEM's cognitive indicators.
NIH_ITEMS = ['NIH7_Card_P', 'NIH7_Flanker_P', 'NIH7_List_P', 'NIH7_Pattern_P']

# LORIS row keys are '<GUID>,assessment'; our subject ids are 'sub-<GUID>'.
def loris_key(subject_id: str) -> str:
    return str(subject_id).split('-')[1] + ',assessment'


def rebuild_nih(out_dir: Path, subjects: pd.Index) -> pd.DataFrame:
    """Extract the four NIH Toolbox items for the given subjects."""
    root = require(HBN_PHENOTYPE, 'hbn_phenotype',
                   'rebuilding the NIH Toolbox scores')
    src = Path(root) / 'HBN' / 'Phenotypic_LORIS' / 'NIH.csv'
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. roots.hbn_phenotype should point at the directory "
            "*containing* HBN/Phenotypic_LORIS/."
        )
    nih = pd.read_csv(src, index_col=0, low_memory=False)

    rows = {}
    for sub in subjects:
        key = loris_key(sub)
        if key not in nih.index:
            continue
        row = nih.loc[key]
        if isinstance(row, pd.DataFrame):       # duplicated assessment rows
            row = row.iloc[0]
        rows[sub] = {item: row.get(f'NIH_Scores,{item}') for item in NIH_ITEMS}

    out = pd.DataFrame.from_dict(rows, orient='index')
    out.index.name = 'subject_id'
    dest = out_dir / 'nih_toolbox_scores.csv'
    out.to_csv(dest)
    complete = int(out[NIH_ITEMS].notna().all(axis=1).sum())
    print(f"  nih_toolbox_scores.csv  {len(out)} subjects, {complete} complete on all four")
    return out


def main() -> None:
    out_dir = PIPELINE / '05_prepare_reg' / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_path = out_dir / 'qc_ratings.csv'
    if not qc_path.exists():
        raise FileNotFoundError(
            f"{qc_path} not found. It ships in the data archive and carries the "
            "manual visual-QC ratings, which cannot be regenerated from LORIS."
        )
    qc = pd.read_csv(qc_path, index_col=0)
    print(f"Loaded {len(qc)} QC ratings (published; not derivable from LORIS)")

    print("\nRebuilding from LORIS:")
    rebuild_nih(out_dir, qc.index)

    print(
        "\nNOTE: participants_df_deepmreye_inc_byhx.xlsx also needs the LORIS\n"
        "      demographic and clinical tables (Basic_Demos, SRS, SCQ, CELF,\n"
        "      ConsensusDx). Column mapping is documented in\n"
        "      pipelines/p2_features/README.md; the assembly itself lives in\n"
        "      deepmreye/00_prepare_data_release11_0deepmreye_mask_inc_byhx.py,\n"
        "      which writes the table alongside the gaze predictions."
    )
    print("\nDone. Re-run the P7 chain to regenerate the downstream z-score tables.")


if __name__ == '__main__':
    main()
