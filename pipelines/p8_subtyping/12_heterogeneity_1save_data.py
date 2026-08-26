"""
12_heterogeneity_1save_data.py
Build subtype-specific SEM input tables (TD + ASD subtype-k).

Purpose
-------
11_sem_zscore_3analysis.R fits the four competing SEMs on the pooled sample
(TD + all ASD). This script splits that same table by the ASD subtypes found in
12_heterogeneity_0subtype.py, so 12_heterogeneity_2sem_zscore.R can refit the
identical models within each subtype and check whether the selected model, its
fit, and the mediation/cascade effects hold in both.

Design
------
- The TD group is shared: each subtype file contains ALL TD subjects plus the
  ASD subjects of one cluster. This keeps the normative reference identical
  across the two fits, so a path difference reflects the ASD subtype and not a
  different comparison group.
- Because TD subjects appear in both subtype files, the two fits are NOT
  independent and must not be pooled into a single multi-group model. For a
  formal between-subtype test, an ASD-only table (group = subtype) is written
  as well; 2sem_zscore.R uses it for the multi-group invariance test.
- Behavioral indicators were z-scored on the pooled sample in
  11_sem_zscore_2save_data.py. That scaling is kept by default
  (rescale_behav = False) so subtype estimates stay on the same metric as the
  published pooled fit. Set rescale_behav = True to re-standardize within each
  subtype file instead.

Input
-----
- all_sem_dim_zscore.csv          (11_sem_zscore_2save_data.py)
- cluster_assignments_{mode}.csv  (12_heterogeneity_0subtype.py)

Output (2_pipeline/99_main/12_heterogeneity/out/<param folder>/)
-----------------------------------------------------------------
- sem_input_subtype-{k}.csv       TD + ASD cluster k, one file per cluster
- sem_input_asd_only.csv          ASD only, with `cluster` column
- sem_input_all_with_cluster.csv  everything, `cluster` = 'TD' or the cluster id
- subtype_sample_summary.csv      n / age / network means per subset
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, fig_dir  # noqa: E402


# =============================================================================
# CONFIGURATION (matching 11_sem_zscore_* and 12_heterogeneity_0subtype.py)
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '12_heterogeneity',
    'source_task_zscore': '11_sem_zscore',

    # Data parameters
    'atlas': 'mmp',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option_pc': 0,
    'srm_option_dim': 50,

    # Upstream analysis parameters
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha_semantic': True,
    'perf_method': 'fdr',
    'perf_alpha': 0.01,
    'weight_across_delays': 'Avg',
    'weight_add_superordinate': True,

    # Must match 11_sem_zscore_0save_raw.py / 12_heterogeneity_0subtype.py
    'screen': True,
    'sig_overlap': False,

    # -------------------------------------------------------------------------
    # Subtype source (must match what 12_heterogeneity_0subtype.py produced)
    # -------------------------------------------------------------------------
    # 'profile'  = CONFIG['normalize_profiles'] was True  (pattern subtypes)
    # 'severity' = CONFIG['normalize_profiles'] was False (severity subtypes)
    'cluster_mode': 'profile',

    # 'kmeans' or 'hierarchical'
    'cluster_method': 'kmeans',

    # -------------------------------------------------------------------------
    # SEM variables (must match 11_sem_zscore_3analysis.R)
    # -------------------------------------------------------------------------
    'behav_items_srs': ['SRS_AWR_T', 'SRS_COG_T', 'SRS_COM_T', 'SRS_MOT_T', 'SRS_RRB_T'],
    'behav_items_nih': ['NIH7_Card_P', 'NIH7_Flanker_P', 'NIH7_List_P', 'NIH7_Pattern_P'],
    'network_vars': ['avg_sal', 'avg_dmn', 'avg_cen'],

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------
    # Re-standardize behavioral indicators within each subtype file. False keeps
    # the pooled-sample scaling used by the published SEM (recommended).
    'rescale_behav': False,

    # Drop subjects missing any SEM variable, mirroring lavaan's listwise
    # deletion, so the reported n matches the n lavaan actually fits.
    'complete_cases': True,

    'save_outputs': True,
}


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Construct all project paths from CONFIG with platform detection."""
    # Paths come from config/paths.yml via lib/project_config.py.  The original
    # working tree hard-coded storage roots here; those are machine-specific and
    # are not distributed.
    proj_root = PROJECT

    proj = config['project']
    pipe = config['pipeline']
    task = config['task']

    paths = {
        'code': proj_root / '1_code' / pipe,
        'pipe': proj_root / '2_pipeline' / pipe,
    }

    # Parameter sub-path shared with 11_sem_zscore
    param_path = (
        Path(config['conf_option']) /
        config['atlas'] /
        f"chunk-{config['chunk_option']}" /
        'fold-avg' / 'results' /
        f"k-{config['srm_option_pc']}_{config['srm_option_dim']}" /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"wn-{config['reg_wordnet']}" /
        f"s_alpha-{config['enc_single_alpha_semantic']}" /
        'sem' /
        f"mask_{config['perf_method']}-{config['perf_alpha']}" /
        f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}" /
        f"screen-{config['screen']}_overlap-{config['sig_overlap']}"
    )

    # Where 11_sem_zscore_2save_data.py wrote all_sem_dim_zscore.csv
    paths['zscore_data'] = paths['pipe'] / config['source_task_zscore'] / 'out' / param_path

    # Where 12_heterogeneity_0subtype.py wrote the cluster assignments.
    # It now writes them to the pipeline output directory; the legacy location
    # inside the figure tree is kept as a fallback so runs predating that change
    # still resolve.
    branch = f"screen-{config['screen']}_overlap-{config['sig_overlap']}"
    paths['clusters'] = paths['pipe'] / task / 'out' / branch
    paths['clusters_legacy'] = paths['code'] / 'claude_figures' / task / branch

    # Output for this script (read by 12_heterogeneity_2sem_zscore.R).
    # Deliberately NOT the full parameter tree: appending it to
    # 12_heterogeneity/out pushes the filenames past the Windows 260-character
    # path limit and to_csv fails. This mirrors 12_heterogeneity_0subtype.py,
    # which keys its output folder on screen/overlap only.
    paths['out'] = (
        paths['pipe'] / task / 'out' /
        f"screen-{config['screen']}_overlap-{config['sig_overlap']}" /
        f"cluster-{config['cluster_method']}_{config['cluster_mode']}"
    )
    paths['out'].mkdir(parents=True, exist_ok=True)

    return paths


PATHS = setup_paths(CONFIG)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sem_table(paths: Dict[str, Path]) -> pd.DataFrame:
    """Load the pooled SEM input table written by 11_sem_zscore_2save_data.py."""
    print("\n[1/5] Loading pooled SEM table...")

    csv_path = paths['zscore_data'] / 'all_sem_dim_zscore.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SEM table not found: {csv_path}\n"
            "Run 11_sem_zscore_2save_data.py first."
        )

    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = 'subject_id'

    print(f"  Loaded: {df.shape[0]} subjects x {df.shape[1]} variables")
    print(f"  From: {csv_path}")
    if 'DX' in df.columns:
        counts = df['DX'].value_counts()
        print("  Group counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    return df


def load_cluster_assignments(paths: Dict[str, Path], config: Dict[str, Any]) -> pd.Series:
    """Load ASD cluster assignments from 12_heterogeneity_0subtype.py."""
    print("\n[2/5] Loading cluster assignments...")

    csv_name = f"cluster_assignments_{config['cluster_mode']}.csv"
    csv_path = paths['clusters'] / csv_name
    if not csv_path.exists() and (paths['clusters_legacy'] / csv_name).exists():
        csv_path = paths['clusters_legacy'] / csv_name
        print(f"  (using legacy cluster-assignment location: {csv_path.parent})")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {csv_path}\n"
            "Run 12_heterogeneity_0subtype.py first (with the matching "
            f"normalize_profiles setting for cluster_mode='{config['cluster_mode']}')."
        )

    cluster_df = pd.read_csv(csv_path)

    col = f"{config['cluster_method']}_cluster"
    if col not in cluster_df.columns:
        raise KeyError(
            f"Column '{col}' not found in {csv_path.name}. "
            f"Available: {list(cluster_df.columns)}. "
            "12_heterogeneity_0subtype.py only writes the methods it actually ran "
            "(see its CONFIG['clustering_method'])."
        )

    labels = cluster_df.set_index('subject_id')[col].astype(int)

    print(f"  Loaded: {len(labels)} ASD subjects from {csv_path.name}")
    print(f"  Method column: {col}")
    for cid, n in labels.value_counts().sort_index().items():
        print(f"    Cluster {cid}: {n} subjects")

    return labels


# =============================================================================
# TABLE CONSTRUCTION
# =============================================================================

def attach_clusters(
    df: pd.DataFrame,
    labels: pd.Series,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Attach the cluster label to each ASD row; TD rows get 'TD'."""
    print("\n[3/5] Attaching cluster labels to the SEM table...")

    df = df.copy()

    is_asd = df['DX'] == 'ASD'
    asd_ids = df.index[is_asd]

    # Cluster assignments come from the full ASD z-score sample, while the SEM
    # table is restricted to subjects with complete behavioral data, so the two
    # sets are expected to differ. Report both directions.
    missing_label = [sid for sid in asd_ids if sid not in labels.index]
    unused_label = [sid for sid in labels.index if sid not in asd_ids]

    if missing_label:
        print(f"  WARNING: {len(missing_label)} ASD subjects in the SEM table have "
              "no cluster assignment; they are dropped from the subtype files.")
        print(f"           e.g. {missing_label[:5]}")
    if unused_label:
        print(f"  Note: {len(unused_label)} clustered ASD subjects are absent from the "
              "SEM table (missing behavioral data upstream).")

    df['cluster'] = np.nan
    df.loc[asd_ids, 'cluster'] = [
        labels.get(sid, np.nan) for sid in asd_ids
    ]

    # Readable label used as the grouping variable downstream
    df['cluster_label'] = np.where(
        df['DX'] == 'TD', 'TD',
        np.where(df['cluster'].notna(),
                 'S' + df['cluster'].astype('Float64').astype(str).str.replace('.0', '', regex=False),
                 'unassigned')
    )

    n_assigned = int(df.loc[is_asd, 'cluster'].notna().sum())
    print(f"  ASD subjects with a cluster label: {n_assigned}/{int(is_asd.sum())}")

    return df


def apply_complete_cases(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Drop rows missing any SEM variable, mirroring lavaan's listwise deletion."""
    if not config['complete_cases']:
        return df

    sem_vars = (config['behav_items_srs'] + config['behav_items_nih'] +
                config['network_vars'])
    missing_cols = [c for c in sem_vars if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing SEM columns: {missing_cols}")

    n_before = len(df)
    df = df[df[sem_vars].notna().all(axis=1)].copy()
    print(f"  Complete cases on all {len(sem_vars)} SEM variables: "
          f"{len(df)} (dropped {n_before - len(df)})")

    return df


def rescale_behavioral(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Re-standardize behavioral indicators within this subset."""
    df = df.copy()
    for col in config['behav_items_srs'] + config['behav_items_nih']:
        sd = df[col].std()
        if sd and sd > 0:
            df[col] = (df[col] - df[col].mean()) / sd
    return df


def summarize_subset(df: pd.DataFrame, label: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """One summary row describing a subset that will be fit by lavaan."""
    row: Dict[str, Any] = {
        'subset': label,
        'n_total': len(df),
        'n_td': int((df['DX'] == 'TD').sum()) if 'DX' in df.columns else np.nan,
        'n_asd': int((df['DX'] == 'ASD').sum()) if 'DX' in df.columns else np.nan,
    }

    if 'Age' in df.columns:
        row['age_mean'] = df['Age'].mean()
        row['age_sd'] = df['Age'].std()

    for var in config['network_vars']:
        if var in df.columns:
            row[f'{var}_mean'] = df[var].mean()
            row[f'{var}_sd'] = df[var].std()

    # Mean of the two indicator blocks, as a quick read on behavioral level
    srs = [c for c in config['behav_items_srs'] if c in df.columns]
    nih = [c for c in config['behav_items_nih'] if c in df.columns]
    if srs:
        row['srs_mean'] = df[srs].mean(axis=1).mean()
    if nih:
        row['nih_mean'] = df[nih].mean(axis=1).mean()

    return row


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("12_heterogeneity_1save_data: Build subtype-specific SEM tables")
    print("=" * 70)

    df = load_sem_table(PATHS)
    labels = load_cluster_assignments(PATHS, CONFIG)
    df = attach_clusters(df, labels, CONFIG)

    # -------------------------------------------------------------------------
    # Step 4: Build the subsets
    # -------------------------------------------------------------------------
    print("\n[4/5] Building subsets...")

    df = apply_complete_cases(df, CONFIG)

    td_df = df[df['DX'] == 'TD'].copy()
    asd_df = df[(df['DX'] == 'ASD') & df['cluster'].notna()].copy()
    asd_df['cluster'] = asd_df['cluster'].astype(int)

    cluster_ids = sorted(asd_df['cluster'].unique())
    print(f"  TD subjects (shared across subtype files): {len(td_df)}")
    print(f"  Clusters found: {cluster_ids}")

    if len(cluster_ids) < 2:
        print("  WARNING: fewer than 2 clusters present; the subtype comparison "
              "in 2sem_zscore.R needs at least 2.")

    subsets: Dict[str, pd.DataFrame] = {}
    for cid in cluster_ids:
        subset = pd.concat([td_df, asd_df[asd_df['cluster'] == cid]], axis=0)
        if CONFIG['rescale_behav']:
            subset = rescale_behavioral(subset, CONFIG)
        subsets[f'subtype-{cid}'] = subset
        print(f"    subtype-{cid}: n={len(subset)} "
              f"(TD={len(td_df)}, ASD={int((subset['DX'] == 'ASD').sum())})")

    # Pooled reference (what 11_sem_zscore_3analysis.R fits), restricted to ASD
    # subjects that carry a cluster label so the three fits share a sample frame
    pooled_df = pd.concat([td_df, asd_df], axis=0)
    print(f"    pooled (reference): n={len(pooled_df)} "
          f"(TD={len(td_df)}, ASD={len(asd_df)})")

    # -------------------------------------------------------------------------
    # Step 5: Save
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving outputs...")

    out_path = PATHS['out']

    if CONFIG['save_outputs']:
        for name, subset in subsets.items():
            subset.to_csv(out_path / f'sem_input_{name}.csv')
            print(f"  Saved: sem_input_{name}.csv ({len(subset)} subjects)")

        pooled_df.to_csv(out_path / 'sem_input_pooled.csv')
        print(f"  Saved: sem_input_pooled.csv ({len(pooled_df)} subjects)")

        # ASD only, for the multi-group invariance test (independent groups)
        asd_out = asd_df.copy()
        asd_out.to_csv(out_path / 'sem_input_asd_only.csv')
        print(f"  Saved: sem_input_asd_only.csv ({len(asd_out)} subjects)")

        # Everything, with the grouping column, for ad hoc checks
        df.to_csv(out_path / 'sem_input_all_with_cluster.csv')
        print(f"  Saved: sem_input_all_with_cluster.csv ({len(df)} subjects)")

        # Sample summary
        summary_rows = [summarize_subset(pooled_df, 'pooled', CONFIG)]
        for name, subset in subsets.items():
            summary_rows.append(summarize_subset(subset, name, CONFIG))
        for cid in cluster_ids:
            summary_rows.append(
                summarize_subset(asd_df[asd_df['cluster'] == cid],
                                 f'asd-only-subtype-{cid}', CONFIG)
            )
        summary_rows.append(summarize_subset(td_df, 'td-only', CONFIG))

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_path / 'subtype_sample_summary.csv', index=False)
        print("  Saved: subtype_sample_summary.csv")

        # Provenance: the output folder is keyed on screen/overlap/cluster only,
        # so record which upstream parameter tree these tables came from.
        with open(out_path / 'source_paths.txt', 'w') as f:
            f.write(f"z-score / SEM table : {PATHS['zscore_data']}\n")
            f.write(f"cluster assignments : {PATHS['clusters']}\n")
            f.write(f"cluster file        : cluster_assignments_{CONFIG['cluster_mode']}.csv\n")
            f.write(f"cluster column      : {CONFIG['cluster_method']}_cluster\n")
            f.write(f"rescale_behav       : {CONFIG['rescale_behav']}\n")
            f.write(f"complete_cases      : {CONFIG['complete_cases']}\n")
        print("  Saved: source_paths.txt")

        print(f"\n  Output folder: {out_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Sample summary")
    print("=" * 70)
    print(pd.DataFrame(
        [summarize_subset(pooled_df, 'pooled', CONFIG)] +
        [summarize_subset(s, n, CONFIG) for n, s in subsets.items()]
    )[['subset', 'n_total', 'n_td', 'n_asd', 'age_mean',
       'avg_sal_mean', 'avg_dmn_mean', 'avg_cen_mean']].to_string(index=False))

    print(f"\nBehavioral rescaling: {'within subset' if CONFIG['rescale_behav'] else 'pooled (unchanged)'}")
    print("\nNote: TD subjects are shared between the subtype files, so those two "
          "fits are not independent.\n      2sem_zscore.R uses sem_input_asd_only.csv "
          "for the formal between-subtype test.")

    print("\n" + "=" * 70)
    print("Done! Run 12_heterogeneity_2sem_zscore.R next.")
    print("=" * 70)


if __name__ == '__main__':
    main()
