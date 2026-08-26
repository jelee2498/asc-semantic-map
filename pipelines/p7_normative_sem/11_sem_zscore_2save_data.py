"""
11_sem_zscore_2save_data.py
Combine Z-Scores with Behavioral Data for SEM Analysis.

This script:
1. Loads z-scores from 11_sem_zscore_1gam.R output
2. Loads behavioral data (SRS + NIH cognitive) from participants_df
3. Flips sem_dmn z-score (-1 * sem_dmn) to align direction with dim_dmn
4. Computes averaged network scores: avg_sal, avg_dmn, avg_cen
5. Z-score normalizes behavioral variables (brain z-scores already normalized)
6. Saves CSV for SEM analysis

Key difference from raw approach:
- Z-scores preserve linearity (SEM assumption)
- Averaging z-scores is mathematically valid
- sem_dmn is flipped with -1 before averaging to align with dim_dmn direction
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, template, fig_dir  # noqa: E402


# =============================================================================
# CONFIGURATION (matching previous scripts)
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '11_sem_zscore',

    # Source data pipelines
    'source_pipeline_participants': '99_main',
    'source_task_participants': '05_prepare_reg',

    # Data parameters
    'atlas': 'mmp',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option_pc': 0,
    'srm_option_dim': 50,

    # Upstream analysis parameters (matching 11_sem)
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha_semantic': True,
    'perf_method': 'fdr',
    'perf_alpha': 0.01,
    'weight_across_delays': 'Avg',
    'weight_add_superordinate': True,

    # Age-based screening (matching 0save_raw.py)
    'screen': True,
    'sig_overlap': False,  # Use overlap significance mask (matching 0save_raw.py)

    # Behavioral items
    'behav_items_srs': ['SRS_AWR_T', 'SRS_COG_T', 'SRS_COM_T', 'SRS_MOT_T', 'SRS_RRB_T'],
    'behav_items_nih': ['NIH7_Card_P', 'NIH7_Flanker_P', 'NIH7_List_P', 'NIH7_Pattern_P'],
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
        # Base paths
        'pipe': proj_root / '2_pipeline' / pipe,

        # Participants data
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        # NIH cognitive data
        'nih_data': store9 / 'HBN' / 'Phenotypic_LORIS' / 'NIH.csv',
    }

    # Task output paths (where z-score data was saved)
    paths['zscore_data'] = (
        paths['pipe'] / task / 'out' /
        config['conf_option'] /
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

    paths['out'] = paths['zscore_data']

    return paths


PATHS = setup_paths(CONFIG)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_zscore_data(paths: Dict[str, Path]) -> pd.DataFrame:
    """Load z-score data from GAM output and combine TD + ASD."""
    print("\n[1/5] Loading z-score data from GAM output...")

    zscore_td = pd.read_csv(paths['zscore_data'] / 'zscore_td.csv', index_col=0)
    zscore_asd = pd.read_csv(paths['zscore_data'] / 'zscore_asd.csv', index_col=0)

    print(f"  TD: {zscore_td.shape}")
    print(f"  ASD: {zscore_asd.shape}")

    # Combine
    zscore_all = pd.concat([zscore_td, zscore_asd], axis=0)
    print(f"  Combined: {zscore_all.shape}")

    return zscore_all


def load_participants_data(paths: Dict[str, Path], id_list: List[str]) -> pd.DataFrame:
    """Load participants DataFrame for behavioral data."""
    print("\n[2/5] Loading participants data...")

    participants_df = pd.read_excel(
        paths['participants_df'] / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    # Filter to subjects in z-score data
    participants_df = participants_df.loc[id_list]
    print(f"  Participants: {participants_df.shape}")

    return participants_df


def load_nih_cognitive_data(
    participants_df: pd.DataFrame,
    paths: Dict[str, Path],
    id_list: List[str]
) -> pd.DataFrame:
    """Load NIH cognitive data from LORIS and merge with participants_df."""
    print("\n[3/5] Loading NIH cognitive data...")

    nih_item_list = [
        'NIH_Scores,NIH7_Card_P',
        'NIH_Scores,NIH7_Flanker_P',
        'NIH_Scores,NIH7_List_P',
        'NIH_Scores,NIH7_Pattern_P'
    ]

    # Load NIH data
    nih_path = paths['nih_data']
    if not nih_path.exists():
        print(f"  Warning: NIH data file not found at {nih_path}")
        return participants_df

    nih_df = pd.read_csv(nih_path, index_col=0)
    print(f"  NIH data loaded: {nih_df.shape[0]} rows")

    # Initialize columns
    for item in nih_item_list:
        col_name = item.split(',')[1]
        participants_df[col_name] = np.nan

    # Map subject IDs and merge
    id_list_no_sub_assess = [sub_id.split('-')[1] + ',assessment' for sub_id in id_list]

    n_found = 0
    for sub_id, sub_id_no_sub_assess in zip(id_list, id_list_no_sub_assess):
        if sub_id_no_sub_assess in nih_df.index:
            for item in nih_item_list:
                col_name = item.split(',')[1]
                value = nih_df.loc[sub_id_no_sub_assess, item]
                try:
                    participants_df.loc[sub_id, col_name] = float(value)
                except (ValueError, TypeError):
                    participants_df.loc[sub_id, col_name] = np.nan
            n_found += 1

    print(f"  Matched {n_found}/{len(id_list)} subjects with NIH data")

    return participants_df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to combine z-scores with behavioral data."""
    print("=" * 70)
    print("11_sem_zscore_2save_data: Combine Z-Scores with Behavioral Data")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load z-score data
    # -------------------------------------------------------------------------
    zscore_all = load_zscore_data(PATHS)
    id_list = list(zscore_all.index)

    # -------------------------------------------------------------------------
    # Step 2: Load participants data
    # -------------------------------------------------------------------------
    participants_df = load_participants_data(PATHS, id_list)

    # -------------------------------------------------------------------------
    # Step 3: Load NIH cognitive data
    # -------------------------------------------------------------------------
    participants_df = load_nih_cognitive_data(participants_df, PATHS, id_list)

    # -------------------------------------------------------------------------
    # Step 4: Prepare data for SEM
    # -------------------------------------------------------------------------
    print("\n[4/5] Preparing data for SEM...")

    # Behavioral items
    behav_items = CONFIG['behav_items_srs'] + CONFIG['behav_items_nih']

    # Filter subjects with all behavioral data
    id_list_behav = []
    for sub_id in id_list:
        all_valid = True
        for item in behav_items:
            if item not in participants_df.columns:
                all_valid = False
                break
            if pd.isna(participants_df.loc[sub_id, item]):
                all_valid = False
                break
        if all_valid:
            id_list_behav.append(sub_id)

    print(f"  Subjects with all behavioral data: {len(id_list_behav)}/{len(id_list)}")

    # Create output DataFrame
    behav_df = participants_df.loc[id_list_behav][behav_items].astype(float).copy()

    # Add z-scores
    behav_df['sem_sal'] = zscore_all.loc[id_list_behav, 'sem_sal']
    behav_df['sem_dmn'] = zscore_all.loc[id_list_behav, 'sem_dmn']
    behav_df['sem_cen'] = zscore_all.loc[id_list_behav, 'sem_cen']
    behav_df['dim_sal'] = zscore_all.loc[id_list_behav, 'dim_sal']
    behav_df['dim_dmn'] = zscore_all.loc[id_list_behav, 'dim_dmn']
    behav_df['dim_cen'] = zscore_all.loc[id_list_behav, 'dim_cen']

    # Flip sem_dmn (-1 * sem_dmn) to align direction with dim_dmn
    # Higher dim_dmn = more dimensions = more complex
    # Higher sem_dmn = higher on PC1, but we need to flip to align with dim direction
    print("\n  Flipping sem_dmn z-score (-1 * sem_dmn) to align with dim_dmn direction...")
    behav_df['sem_dmn_flipped'] = -1 * behav_df['sem_dmn']

    # Compute averaged network scores (using flipped sem_dmn)
    behav_df['avg_sal'] = (behav_df['sem_sal'] + behav_df['dim_sal']) / 2
    behav_df['avg_dmn'] = (behav_df['sem_dmn_flipped'] + behav_df['dim_dmn']) / 2
    behav_df['avg_cen'] = (behav_df['sem_cen'] + behav_df['dim_cen']) / 2

    # Drop the intermediate flipped column
    behav_df = behav_df.drop(columns=['sem_dmn_flipped'])

    # Add metadata
    behav_df['DX'] = zscore_all.loc[id_list_behav, 'DX']
    behav_df['Age'] = participants_df.loc[id_list_behav, 'Age']

    # Define columns to z-score (only behavioral - brain vars are already z-scores)
    # Note: Brain z-scores are relative to TD norm, we keep them as-is
    # Only z-score behavioral items for comparability
    cols_to_zscore = behav_items

    # Z-score normalize behavioral variables only
    behav_df_norm = behav_df.copy()
    for col in cols_to_zscore:
        col_mean = behav_df_norm[col].mean()
        col_std = behav_df_norm[col].std()
        if col_std > 0:
            behav_df_norm[col] = (behav_df_norm[col] - col_mean) / col_std

    # Split by group
    behav_df_td = behav_df_norm[behav_df_norm['DX'] == 'TD'].copy()
    behav_df_asd = behav_df_norm[behav_df_norm['DX'] == 'ASD'].copy()

    print(f"\n  Output DataFrame (z-scored): {behav_df_norm.shape}")
    print(f"    TD: {behav_df_td.shape}")
    print(f"    ASD: {behav_df_asd.shape}")

    # -------------------------------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving outputs...")

    out_path = PATHS['out']

    behav_df_norm.to_csv(out_path / 'all_sem_dim_zscore.csv')
    behav_df_td.to_csv(out_path / 'all_sem_dim_zscore_td.csv')
    behav_df_asd.to_csv(out_path / 'all_sem_dim_zscore_asd.csv')

    print(f"  Saved to: {out_path}")
    print(f"    - all_sem_dim_zscore.csv")
    print(f"    - all_sem_dim_zscore_td.csv")
    print(f"    - all_sem_dim_zscore_asd.csv")

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print("\nAveraged network scores (after sem_dmn flip and averaging):")
    print(behav_df_norm[['avg_sal', 'avg_dmn', 'avg_cen']].describe())

    print("\nCorrelations between sem and dim (in z-score space):")
    print(f"  SAL: r = {behav_df['sem_sal'].corr(behav_df['dim_sal']):.3f}")
    # Note: using flipped sem_dmn for correlation check
    sem_dmn_flipped = -1 * behav_df['sem_dmn']
    print(f"  DMN (flipped): r = {sem_dmn_flipped.corr(behav_df['dim_dmn']):.3f}")
    print(f"  CEN: r = {behav_df['sem_cen'].corr(behav_df['dim_cen']):.3f}")

    print("\n" + "=" * 70)
    print("Done! Run 11_sem_zscore_3analysis.R next for SEM analysis.")
    print("=" * 70)


if __name__ == '__main__':
    main()
