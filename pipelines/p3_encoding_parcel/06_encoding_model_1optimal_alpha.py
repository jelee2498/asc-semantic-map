"""
Train final encoding models using optimal alpha from parameter search.

This script uses the optimal ridge regularization parameter (alpha) identified
in Step 0 (06_encoding_model_0search_alpha_sub.py) to train final encoding models
across all subjects. Unlike Step 0, this does NOT perform bootstrap CV or alpha
search - it directly trains on the full training set with the pre-computed optimal alpha.

Reference Files (read-only):
    - S:\\jelee\\02_asd_semantic_map\\1_code\\17_param_search\\02_pca_group_comparison_release11_1optimal_alpha_inc_byhx.py
    - S:\\jelee\\02_asd_semantic_map\\1_code\\17_param_search\\functions.py

Input:
    - boot_corrs from Step 0 (to determine optimal alpha)
    - Preprocessed fMRI data (SRM-aligned, .npy files per fold)
    - Movie regressors (Excel files with semantic features)
    - Participant metadata (participants_df_deepmreye_inc_byhx.xlsx)

Output:
    - weight: Ridge regression weights (features x voxels) fold-averaged
    - test_corrs: Test set correlations fold-averaged
    - noise_ceiling: Split-half reliability estimates (upper bound) fold-averaged
    - test_corrs_corrected: Test correlations corrected by noise ceiling fold-averaged

Usage:
    # Batch mode: Process all subjects in parallel (default)
    python 06_encoding_model_1optimal_alpha.py
    python 06_encoding_model_1optimal_alpha.py --hx_docu True

    # Debug mode: Process single subject for testing
    python 06_encoding_model_1optimal_alpha.py --debug_subject sub-NDARINVXXXXXXX
    python 06_encoding_model_1optimal_alpha.py --hx_docu False --debug_subject sub-NDARINVXXXXXXX

Execution Modes:
    Batch mode (default):
        - Processes all QC-filtered subjects in parallel using joblib
        - Uses CONFIG['n_jobs'] parallel workers (default: 4)
        - Optimal for production runs

    Debug mode (--debug_subject):
        - Processes only the specified subject
        - Serial execution (no parallelization)
        - Prints detailed progress for single subject
        - Optimal for testing and debugging

Project: 02_asd_semantic_map
Pipeline: 99_main
Task: 06_encoding_model
Step: 1_optimal_alpha (final model training)
"""

import os
import random
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import chain
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed


# ============================================================================
# PLATFORM-AWARE PATH CONFIGURATION
# ============================================================================

# Paths come from config/paths.yml via lib/project_config.py.  The original
# working tree hard-coded a storage root here (S:/ on Windows, /MIPL/store7 on
# Linux); that is machine-specific and is not distributed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, RAW, CODE, PIPELINE, TEMPLATES  # noqa: E402


# ============================================================================
# UTILITY FUNCTIONS (from 17_param_search/functions.py)
# ============================================================================

# Z-score function
zs = lambda v: (v - v.mean(0)) / v.std(0)
zs.__doc__ = """Z-scores (standardizes) each column of [v]."""


def make_delayed(REG: np.ndarray, DELAYS: int) -> np.ndarray:
    """
    Make delayed regressor.

    Args:
        REG: Original regressor
        DELAYS: Number of TRs to delay

    Returns:
        Delayed regressor
    """
    D_REG = np.zeros(len(REG))
    if DELAYS != 0:
        D_REG[DELAYS:] = REG[:-DELAYS]
    else:
        D_REG = REG

    return D_REG


def make_delayed_all(REG_ARRAY: np.ndarray, DELAYS: int) -> np.ndarray:
    """
    Make delayed regressor arrays.

    Args:
        REG_ARRAY: Original regressor array
        DELAYS: Number of TRs to delay

    Returns:
        Delayed regressor array
    """
    D_REG_ARRAY = None
    if DELAYS != 0:
        D_REG_ARRAY = np.array([make_delayed(REG_ARRAY[I, :], DELAYS) for I in range(len(REG_ARRAY))])
    else:
        D_REG_ARRAY = REG_ARRAY

    return D_REG_ARRAY


def save_files_sub(DICT: Dict[str, np.ndarray], SUB_ID: str, SAVE_PATH: str, EXIST_OK: bool = True) -> None:
    """
    Save individual results dictionary as individual files.

    Args:
        DICT: Dictionary of results to save
        SUB_ID: Subject ID
        SAVE_PATH: Path to save directory
        EXIST_OK: Whether to allow existing directories
    """
    KEYS = DICT.keys()

    for KEY in KEYS:
        os.makedirs(os.path.join(SAVE_PATH, KEY), exist_ok=EXIST_OK)
        np.save(os.path.join(SAVE_PATH, KEY, SUB_ID), DICT[KEY])


def load_files(KEYS: List[str], ID_LIST: List[str], SAVE_PATH: str) -> Dict[str, List[np.ndarray]]:
    """
    Load saved files for multiple subjects.

    Args:
        KEYS: List of result keys to load
        ID_LIST: List of subject IDs
        SAVE_PATH: Path to saved results directory

    Returns:
        Dictionary mapping keys to lists of arrays (one per subject)
    """
    RESULTS_DICT = {}

    for KEY in KEYS:
        RESULTS = []
        for SUB_ID in ID_LIST:
            FILE_PATH = os.path.join(SAVE_PATH, KEY, f'{SUB_ID}.npy')
            if os.path.exists(FILE_PATH):
                RESULTS.append(np.load(FILE_PATH))
            else:
                raise FileNotFoundError(f"File not found: {FILE_PATH}")
        RESULTS_DICT[KEY] = RESULTS

    return RESULTS_DICT


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG: Dict[str, Any] = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '06_encoding_model',

    # Base paths
    'raw_path': RAW,
    'code_path': CODE,
    'pipe_path': PIPELINE,

    # Source data paths
    'source_pipeline_fmri': '17_param_search',  # fMRI from 17_param_search
    'source_pipeline_reg': '99_main',           # Regressors from 99_main
    'fmri_data_task': '00_prepare_data_release11',
    'reg_data_task': '05_prepare_reg',

    # fMRI preprocessing options
    'conf_option': 'default+me',  # Confound regression strategy
    'atlas': 'mmp',  # Parcellation atlas (Glasser 2016)
    'srm_option': 0,  # SRM components (0 = no SRM)

    # Cross-validation parameters
    'chunk_option': 9,  # Number of chunks (9 or 16)

    # Movie parameters
    'start_ignore_trs': 5,  # TRs to discard at beginning
    'end_ignore_trs': 5,    # TRs to discard at end
    'tr_delays_option': [3, 5, 7, 9],  # Hemodynamic response delays
    'n_dm_trs': 750,  # Total TRs in DM movie (before trimming)

    # Quality control thresholds
    'fd_threshold': 0.5,  # FD threshold in mm

    # Ridge regression parameters
    'alpha_list': np.logspace(1, 4, 20),  # 20 alphas from 10 to 10000 (must match Step 0)
    'motion_energy_option': False,  # Include motion energy features

    # Noise ceiling parameters (split-half reliability)
    'hrf_block_len': 25,  # HRF-aware block length (samples)
    'sb_min_block_len': 12,  # Minimum block length for A/B split

    # Gaze weight grid (bias_weight, verb_weight)
    'gaze_weight_grid': [
        (0.1, 0.5),
        (0.1, 0.55),
        (0.1, 0.1),
        (0.1, 1.0),
        (0.3, 0.5),
        (0.3, 0.65),
        (0.3, 0.3),
        (0.3, 1.0),
        (0.5, 0.5),  # (0.5, 0.5) is overlapping so only once
        (0.5, 0.75),
        (0.5, 1.0),
        (0.7, 0.5),
        (0.7, 0.85),
        (0.7, 0.7),
        (0.7, 1.0),
        (0.9, 0.5),
        (0.9, 0.95),
        (0.9, 0.9),
        (0.9, 1.0)
    ],

    'wordnet_options': [True],  # Use WordNet hierarchy

    # Parallel processing
    'n_jobs': 4,  # Number of parallel jobs

    # Random seed for reproducibility (0, 37, 42)
    'seed': 0,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_paths(hx_docu: str) -> Dict[str, Path]:
    """
    Configure all file paths for the analysis pipeline.

    Args:
        hx_docu: ASD by-history documentation filter ('True' or 'False')

    Returns:
        Dictionary containing:
            - participants_df_path: Path to participants dataframe (from 99_main/05_prepare_reg)
            - fmri_path: Base path to preprocessed fMRI data (from 17_param_search)
            - reg_path: Regressor data path (from 99_main/05_prepare_reg)
            - out_path: Output directory
    """
    proj = CONFIG['project']

    # fMRI data path (from 17_param_search)
    fmri_source_pipe = PROJECT / '2_pipeline' / CONFIG['source_pipeline_fmri']
    fmri_path = fmri_source_pipe / CONFIG['fmri_data_task'] / 'out'

    # Regressor and participant data path (from 99_main/05_prepare_reg)
    reg_source_pipe = PROJECT / '2_pipeline' / CONFIG['source_pipeline_reg']
    reg_path = reg_source_pipe / CONFIG['reg_data_task'] / 'out'
    participants_df_path = reg_path  # Same location

    # Target pipeline paths (99_main)
    proc_path = CONFIG['pipe_path'] / CONFIG['task']
    out_path = proc_path / 'out'

    return {
        'participants_df_path': participants_df_path,
        'fmri_path': fmri_path,
        'reg_path': reg_path,
        'out_path': out_path,
    }


def load_and_filter_participants(
    hx_docu: str,
    paths: Dict[str, Path]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load participants dataframe and apply QC filters.

    Filters applied:
    1. Diagnosis (TD/ASD/all) - currently set to 'all'
    2. ASD by-history documentation (if hx_docu='True')
    3. Preprocessing complete (prep_ok == 1)
    4. No remarks during preprocessing
    5. FD < threshold
    6. DeepMReye quality rating == 1

    Args:
        hx_docu: ASD by-history documentation filter ('True' or 'False')
        paths: Dictionary of file paths from setup_paths()

    Returns:
        Tuple of (filtered_df, subject_id_list)
    """
    print('* Loading and filtering participants')

    # Load participants dataframe
    participants_df = pd.read_excel(
        paths['participants_df_path'] / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    # Filter 1: Diagnosis (set to 'all' to include both TD and ASD)
    dx = 'all'
    if dx == 'TD':
        dx_filt = (participants_df['DX'] == 'TD').values
    elif dx == 'ASD':
        dx_filt = (participants_df['DX'] == 'ASD').values
    elif dx == 'all':
        dx_filt = np.ones(len(participants_df)) > 0
    else:
        raise ValueError(f"Invalid dx option: {dx}")

    # Filter 2: ASD by-history documentation
    if hx_docu == 'True':
        for sub_id in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                print(f"  Exclude {sub_id} due to by-history without documentation")
                dx_filt[participants_df.index == sub_id] = False

    # Filter 3: Preprocessing complete
    prep_filt = (participants_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] == 1).values

    # Filter 4: No remarks during preprocessing
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Combine filters
    id_list_dx_prep_remark = list(participants_df[dx_filt & prep_filt & no_remarks_filt].index)
    participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

    # Filter 5: FD threshold
    fd_filt = (participants_df_dx_prep_remark['Mean_FD_DM'] < CONFIG['fd_threshold']).values

    # Filter 6: DeepMReye quality
    deepmreye_filt = (participants_df_dx_prep_remark['Rating_deepmreye_movieDM'] == 1).values

    # Final subject list
    id_list = list(participants_df_dx_prep_remark[np.multiply(fd_filt, deepmreye_filt)].index)

    # Print participant info
    print(f'  Participants (n={len(id_list)}):')
    dx_counts = participants_df.loc[id_list]['DX'].value_counts()
    print(f'    DX: TD={dx_counts.get("TD", 0)}, ASD={dx_counts.get("ASD", 0)}')
    site_counts = participants_df.loc[id_list]['Site'].value_counts()
    print(f'    Site: CBIC={site_counts.get("CBIC", 0)}, RU={site_counts.get("RU", 0)}')
    sex_counts = participants_df.loc[id_list]['Sex'].value_counts()
    print(f'    Sex: Male={sex_counts.get("Male", 0)}, Female={sex_counts.get("Female", 0)}')

    return participants_df.loc[id_list], id_list


def create_cross_validation_folds(chunk_option: int) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Generate chunk indices and fold assignments for cross-validation.

    Creates evenly-sized chunks across 740 TRs (750 - 10 trimmed), then
    merges chunks into sqrt(chunk_option) folds using a grid pattern.

    Args:
        chunk_option: Number of chunks (9 or 16)

    Returns:
        Tuple of (chunk_list, fold_list):
            - chunk_list: List of arrays with TR indices per chunk
            - fold_list: List of lists with TR indices per test fold

    Raises:
        ValueError: If chunk_option is not 9 or 16

    Example:
        For chunk_option=9:
        - Creates 9 chunks of ~82 TRs each
        - Merges into 3 folds (chunks [0,3,6], [1,4,7], [2,5,8])
    """
    if chunk_option == 9:
        chunk_list = [
            np.linspace(0, 81, 82, dtype=np.int16),
            np.linspace(82, 163, 82, dtype=np.int16),
            np.linspace(164, 246, 83, dtype=np.int16),
            np.linspace(247, 328, 82, dtype=np.int16),
            np.linspace(329, 410, 82, dtype=np.int16),
            np.linspace(411, 493, 83, dtype=np.int16),
            np.linspace(494, 575, 82, dtype=np.int16),
            np.linspace(576, 657, 82, dtype=np.int16),
            np.linspace(658, 739, 82, dtype=np.int16)
        ]
    elif chunk_option == 16:
        chunk_list = [
            np.linspace(0, 45, 46, dtype=np.int16),
            np.linspace(46, 91, 46, dtype=np.int16),
            np.linspace(92, 137, 46, dtype=np.int16),
            np.linspace(138, 184, 47, dtype=np.int16),
            np.linspace(185, 230, 46, dtype=np.int16),
            np.linspace(231, 276, 46, dtype=np.int16),
            np.linspace(277, 322, 46, dtype=np.int16),
            np.linspace(323, 369, 47, dtype=np.int16),
            np.linspace(370, 415, 46, dtype=np.int16),
            np.linspace(416, 461, 46, dtype=np.int16),
            np.linspace(462, 507, 46, dtype=np.int16),
            np.linspace(508, 554, 47, dtype=np.int16),
            np.linspace(555, 600, 46, dtype=np.int16),
            np.linspace(601, 646, 46, dtype=np.int16),
            np.linspace(647, 692, 46, dtype=np.int16),
            np.linspace(693, 739, 47, dtype=np.int16)
        ]
    else:
        raise ValueError(
            f"Invalid chunk_option: {chunk_option}. Must be 9 or 16."
        )

    # Merge chunks into folds (grid pattern)
    n_merge = int(np.sqrt(chunk_option))
    fold_list = []

    for i in range(n_merge):
        # Select chunks with stride n_merge (e.g., [0,3,6] for 9 chunks)
        merge_chunk_ids = [i + n_merge * j for j in range(n_merge)]
        fold_trs = []

        for chunk_idx in merge_chunk_ids:
            fold_trs.extend(chunk_list[chunk_idx].tolist())

        fold_list.append(fold_trs)

    return chunk_list, fold_list


def build_output_path_step0(
    base_path: Path,
    hx_docu: str,
    gaze_weight: Tuple[float, float],
    wordnet: bool
) -> Path:
    """
    Build output path for Step 0 boot_corrs (abbreviated naming, s_alpha=False).

    Args:
        base_path: Base output directory
        hx_docu: ASD by-history documentation filter ('True' or 'False')
        gaze_weight: Tuple of (bias_weight, verb_weight)
        wordnet: Whether WordNet hierarchy is used

    Returns:
        Full output path to Step 0 results (including seed-{N})
    """
    conf = CONFIG['conf_option']
    atlas = CONFIG['atlas']
    chunk = CONFIG['chunk_option']
    srm = CONFIG['srm_option']
    seed = CONFIG['seed']

    # Build base path
    path = (
        base_path /
        conf /
        atlas /
        f'chunk-{chunk}' /
        'fold-avg' /
        f'd-{hx_docu}' /
        f'k-{srm}'
    )

    # Add gaze weight directory
    if not gaze_weight[0]:
        path = path / 'gaze_weight-False'
    else:
        path = path / f'bias-{gaze_weight[0]}_verb-{gaze_weight[1]}'

    # Add WordNet and seed directories
    path = path / f'wn-{wordnet}'
    path = path / f'seed-{seed}'

    return path


def build_output_path_step1(
    base_path: Path,
    hx_docu: str,
    gaze_weight: Tuple[float, float],
    wordnet: bool
) -> Path:
    """
    Build output path for Step 1 final models (abbreviated naming, s_alpha=True).

    Args:
        base_path: Base output directory
        hx_docu: ASD by-history documentation filter ('True' or 'False')
        gaze_weight: Tuple of (bias_weight, verb_weight)
        wordnet: Whether WordNet hierarchy is used

    Returns:
        Full output path for Step 1 results
    """
    # Same as Step 0 but with s_alpha=True appended
    path = build_output_path_step0(base_path, hx_docu, gaze_weight, wordnet)
    path = path / 's_alpha-True'
    return path


def load_optimal_alpha(
    gaze_weight: Tuple[float, float],
    wordnet: bool,
    hx_docu: str,
    id_list: List[str],
    paths: Dict[str, Path]
) -> float:
    """
    Load pre-computed boot_corrs from Step 0 and find optimal alpha.

    Steps:
    1. Load boot_corrs for all subjects (from fold-avg, s_alpha=False path)
    2. Average across subjects: mean(boot_corrs_list)
    3. Find alpha with max mean correlation: argmax(mean_boot_corrs)

    Args:
        gaze_weight: Tuple of (bias_weight, verb_weight)
        wordnet: Whether WordNet hierarchy is used
        hx_docu: ASD by-history documentation filter
        id_list: List of subject IDs
        paths: Dictionary of file paths

    Returns:
        Optimal alpha value (scalar)

    Raises:
        FileNotFoundError: If boot_corrs files not found
        ValueError: If optimal alpha is NaN or invalid
    """
    # Build path to Step 0 boot_corrs (s_alpha=False)
    boot_corrs_path = build_output_path_step0(
        paths['out_path'],
        hx_docu,
        gaze_weight,
        wordnet
    )

    # Load boot_corrs for all subjects
    try:
        results_dict = load_files(['boot_corrs'], id_list, str(boot_corrs_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find boot_corrs from Step 0 at {boot_corrs_path}. "
            f"Please run 06_encoding_model_0search_alpha_sub.py first."
        ) from e

    boot_corrs_list = results_dict['boot_corrs']

    # Average across subjects
    boot_corrs_mean = np.array(boot_corrs_list).mean(axis=0)

    # Find optimal alpha
    alpha_list = CONFIG['alpha_list']
    optimal_alpha_idx = np.argmax(boot_corrs_mean)
    optimal_alpha = alpha_list[optimal_alpha_idx]

    # Validate
    if not np.isfinite(optimal_alpha):
        raise ValueError(f"Optimal alpha is not finite: {optimal_alpha}")

    print(f'  Optimal alpha: {optimal_alpha:.2f} (index {optimal_alpha_idx}/20)')

    return optimal_alpha


def load_regressors(
    gaze_weight: Tuple[float, float],
    wordnet: bool,
    hx_docu: str,
    sub_id: str,
    paths: Dict[str, Path]
) -> Tuple[np.ndarray, List[str]]:
    """
    Load movie regressors with optional gaze weighting and WordNet hierarchy.

    Loads DM movie semantic features from Excel files. Supports four regressor
    types based on gaze_weight and wordnet flags:
    1. Standard regressors (no gaze, no WordNet)
    2. WordNet hierarchy regressors (no gaze, with WordNet)
    3. Gaze-weighted regressors (with gaze, no WordNet)
    4. Gaze-weighted WordNet regressors (with both)

    Args:
        gaze_weight: Tuple of (bias_weight, verb_weight).
                     False means no gaze weighting.
        wordnet: Whether to use WordNet semantic hierarchy
        hx_docu: ASD by-history documentation filter ('True' or 'False')
        sub_id: Subject ID (required for gaze-weighted regressors)
        paths: Dictionary of file paths from setup_paths()

    Returns:
        Tuple of (regressor_array, label_list):
            - regressor_array: (n_features, n_frames) semantic features
            - label_list: List of feature names

    Raises:
        FileNotFoundError: If regressor file not found in either location

    Notes:
        - Gaze-weighted regressors are subject-specific
        - Non-gaze regressors are shared across all subjects
        - Small noise is added later to avoid NaN in ridge regression
    """
    raw_path = CONFIG['raw_path']

    # Case 1 & 2: No gaze weighting
    if not gaze_weight[0]:
        if not wordnet:
            # Case 1: Standard regressors
            reg_df = pd.read_excel(
                raw_path / 'movie' / 'DM' / 'sync_regressor_DM.xlsx',
                index_col=0,
                sheet_name='all (wordnet)',
                engine='openpyxl'
            )
        else:
            # Case 2: WordNet hierarchy
            reg_df = pd.read_excel(
                raw_path / 'movie' / 'DM' / 'wordnet_sync_regressor_DM.xlsx',
                index_col=0,
                engine='openpyxl'
            )

        # Extract feature labels (exclude time column)
        label_list = [label for label in reg_df.index if label != 'Time (sec)']

        # Get frame columns (format: 'f0000', 'f0001', ...)
        frame_list = [col for col in reg_df.columns if 'f' in col]

        # Extract regressor matrix
        reg_array = reg_df.loc[label_list, frame_list].values

    # Case 3 & 4: Gaze weighting
    else:
        bias_w, verb_w = gaze_weight

        if not wordnet:
            # Case 3: Gaze-weighted (no WordNet)
            reg_dir = 'gaze_weighted_regressor'
        else:
            # Case 4: Gaze-weighted WordNet
            reg_dir = 'wordnet_gaze_weighted_regressor'

        # Load from 99_main/05_prepare_reg
        reg_file = (
            paths['reg_path'] /
            f'{reg_dir}_inc_byhx_docu-{hx_docu}' /
            'within' /
            f'bias-{bias_w}_verb-{verb_w}' /
            f'{sub_id}_regressor_DM.xlsx'
        )
        reg_df = pd.read_excel(reg_file, index_col=0, engine='openpyxl')

        # Extract labels and frames
        label_list = list(reg_df.index)
        frame_list = reg_df.columns

        reg_array = reg_df.loc[label_list, frame_list].values

    return reg_array, label_list


def prepare_regressors(
    reg_array: np.ndarray,
    label_list: List[str],
    tr_delays: List[int],
    start_ignore: int,
    end_ignore: int,
    me_option: bool = False
) -> Tuple[np.ndarray, Optional[List[int]], int]:
    """
    Prepare regressors for encoding model: add noise, create delays, trim TRs.

    Processing steps:
    1. Add tiny random noise to avoid NaN values in ridge regression
    2. (Optional) Append motion energy features
    3. Create delayed versions for each TR delay
    4. Trim ignored TRs from beginning and end

    Args:
        reg_array: (n_features, n_frames) raw regressors
        label_list: List of feature names
        tr_delays: List of TR delays for hemodynamic response
        start_ignore: Number of TRs to trim from start
        end_ignore: Number of TRs to trim from end
        me_option: Whether to include motion energy features

    Returns:
        Tuple of (delayed_regressors, me_indices, n_labels):
            - delayed_regressors: (n_features * n_delays, n_trs) array
            - me_indices: Indices of motion energy features (or None)
            - n_labels: Number of original features

    Notes:
        - Noise is uniform random in [0, 1e-10]
        - Motion energy features are NOT delayed (fixed in place)
        - Final shape: (n_features * len(tr_delays), n_trs_trimmed)
    """
    # Step 1: Add small noise to avoid NaN
    reg_noisy = reg_array + np.random.rand(*reg_array.shape) * 1e-10

    # Step 2: Add motion energy if requested
    me_ids = None
    if me_option:
        # Load motion energy features
        raw_path = CONFIG['raw_path']
        me_df = pd.read_excel(
            raw_path / 'movie' / 'DM' / 'DM_motion_energy.xlsx',
            sheet_name='mean',
            index_col=0,
            engine='openpyxl'
        )
        me_mean = me_df.values.flatten()

        # Append to regressors
        reg_noisy = np.vstack([reg_noisy, me_mean])
        label_list.append('Motion_Energy_Mean')

        # Calculate motion energy indices in delayed regressor array
        me_idx = len(label_list) - 1  # Last feature
        me_ids = [me_idx + i * len(label_list) for i in range(len(tr_delays))]

    n_labels = len(label_list)

    # Step 3: Create delayed regressors
    reg_delayed = None
    for i, tr_delay in enumerate(tr_delays):
        delayed = make_delayed_all(reg_noisy, tr_delay)

        if i == 0:
            reg_delayed = delayed
        else:
            reg_delayed = np.vstack([reg_delayed, delayed])

    # Step 4: Trim ignored TRs
    reg_delayed = reg_delayed[:, start_ignore:-end_ignore]

    return reg_delayed, me_ids, n_labels


def compute_noise_ceiling_split_half(
    reg_test_z: np.ndarray,
    fmri_test_z: np.ndarray,
    alpha: float,
    hrf_block_len: int = 25,
    min_block_len: int = 12
) -> np.ndarray:
    """
    Compute noise ceiling using split-half reliability with Spearman-Brown correction.

    This method estimates the theoretical upper bound on model performance by:
    1. Splitting test data into two disjoint halves (A and B) using HRF-aware blocks
    2. Fitting separate ridge models on each half
    3. Reconstructing signals over the full test set
    4. Computing split-half correlation and applying Spearman-Brown correction
    5. Converting to correlation ceiling via sqrt(reliability)

    The split-half approach accounts for temporal autocorrelation by using blocks
    longer than the HRF support (~20-25 samples at TR=0.8s).

    Args:
        reg_test_z: (n_trs, n_features) z-scored test regressors
        fmri_test_z: (n_trs, n_voxels) z-scored test fMRI
        alpha: Ridge regularization parameter (scalar, same for all voxels)
        hrf_block_len: Block length in samples for A/B split (default: 25)
        min_block_len: Minimum block length fallback (default: 12)

    Returns:
        noise_ceiling: (n_voxels,) ceiling values in [0, 1]

    Notes:
        - Uses scalar alpha (optimal alpha from Step 0)
        - Falls back to odd/even split if blocks are too short
        - Negative correlations are clipped to 0
        - Based on split-half reliability theory (Spearman-Brown formula)

    References:
        - Spearman-Brown prophecy formula for test reliability
        - HRF-aware chunking to avoid temporal correlation artifacts
    """

    def _corr_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Column-wise Pearson correlation."""
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        num = (A * B).sum(axis=0)
        den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
        r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(r, -0.999999, 0.999999)

    def _ridge_fit_masked(
        X: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray,
        lam: float
    ) -> np.ndarray:
        """Ridge regression on masked rows (scalar lambda)."""
        Xs = X[mask]  # (n_masked, n_features)
        Ys = Y[mask]  # (n_masked, n_voxels)
        XtX = Xs.T @ Xs
        Xty = Xs.T @ Ys
        P = XtX.shape[0]
        beta = np.linalg.solve(XtX + lam * np.eye(P), Xty)
        return beta

    # Step 1: Create HRF-aware A/B split
    T_test = reg_test_z.shape[0]
    B = max(min_block_len, hrf_block_len)

    # Alternate blocks between A and B
    maskA = np.zeros(T_test, dtype=bool)
    maskB = np.zeros(T_test, dtype=bool)
    toggle = True

    for start in range(0, T_test, B):
        end = min(start + B, T_test)
        if toggle:
            maskA[start:end] = True
        else:
            maskB[start:end] = True
        toggle = not toggle

    # Fallback: if one mask is empty (very short fold), use odd/even
    if (maskA.sum() == 0) or (maskB.sum() == 0):
        maskA = np.zeros(T_test, dtype=bool)
        maskA[::2] = True
        maskB = ~maskA

    # Step 2: Fit ridge models on disjoint samples (scalar alpha)
    betaA = _ridge_fit_masked(reg_test_z, fmri_test_z, maskA, alpha)
    betaB = _ridge_fit_masked(reg_test_z, fmri_test_z, maskB, alpha)

    # Step 3: Reconstruct signals over full test set
    sA = reg_test_z @ betaA  # (T_test, V)
    sB = reg_test_z @ betaB  # (T_test, V)

    # Step 4: Split-half correlation and Spearman-Brown correction
    r_split = _corr_cols(sA, sB)  # (V,)
    r_sb = np.divide(2 * r_split, 1 + r_split + 1e-12)
    r_sb = np.clip(r_sb, 0.0, 1.0)  # Clamp negatives

    # Step 5: Convert to correlation ceiling
    ceiling = np.sqrt(r_sb)

    return ceiling


def train_final_model_fold(
    fold_idx: int,
    test_fold_ids: List[int],
    fmri_train_path: Path,
    fmri_test_path: Path,
    reg_delayed: np.ndarray,
    n_dm_trs: int,
    me_ids: Optional[List[int]],
    optimal_alpha: float
) -> Dict[str, np.ndarray]:
    """
    Train ridge regression model with fixed optimal alpha for one fold.

    No bootstrap, no alpha search - just train on full training set
    and evaluate on test set using SVD-based ridge regression.

    Args:
        fold_idx: Fold index (0-indexed)
        test_fold_ids: TR indices in test fold
        fmri_train_path: Path to training fMRI .npy file
        fmri_test_path: Path to test fMRI .npy file
        reg_delayed: (n_features * n_delays, n_trs) delayed regressors
        n_dm_trs: Total number of TRs (after trimming)
        me_ids: Motion energy feature indices (or None)
        optimal_alpha: Ridge regularization parameter

    Returns:
        Dictionary with keys:
            - weight: (n_features * n_delays, n_voxels) regression weights
            - test_corrs: (n_voxels,) test correlations
            - noise_ceiling: (n_voxels,) split-half reliability estimates
            - test_corrs_corrected: (n_voxels,) test_corrs / noise_ceiling
            - pred_bold: (n_test_trs, n_voxels) predicted BOLD signal (not z-scored)

    Notes:
        - pred_bold is saved per-fold but not aggregated (too large for fold-avg)
        - Unlike Step 0, pred_bold is NOT z-scored in Step 1
    """
    # Load fMRI data
    fmri_train = np.load(fmri_train_path)  # (n_voxels, n_train_trs)
    fmri_test = np.load(fmri_test_path)    # (n_voxels, n_test_trs)

    # Get train fold IDs (complement of test)
    train_fold_ids = np.array(list(set(range(n_dm_trs)) - set(test_fold_ids)))

    # Split regressors
    reg_train = reg_delayed[:, train_fold_ids]
    reg_test = reg_delayed[:, test_fold_ids]

    # Identify non-motion-energy features
    if me_ids:
        non_me_ids = list(set(range(len(reg_train))) - set(me_ids))
    else:
        non_me_ids = list(range(len(reg_train)))

    # Z-score all data
    fmri_train_z = zscore(fmri_train.T, axis=0)  # (n_train_trs, n_voxels)
    fmri_test_z = zscore(fmri_test.T, axis=0)    # (n_test_trs, n_voxels)
    reg_train_z = zscore(reg_train.T, axis=0)    # (n_train_trs, n_features)

    # For test regressors, exclude motion energy from z-scoring
    reg_test_z = zscore(reg_test[non_me_ids].T, axis=0)

    # SVD-based ridge regression
    U, S, Vh = np.linalg.svd(reg_train_z, full_matrices=False)
    UR = np.dot(U.T, np.nan_to_num(fmri_train_z))

    if not me_ids:
        # No motion energy features
        weight = reduce(np.dot, [Vh.T, np.diag(S / (S ** 2 + optimal_alpha)), UR])
        pred = np.dot(reg_test_z, weight)
    else:
        # Motion energy features: exclude from test regressors
        weight = reduce(np.dot, [Vh.T, np.diag(S / (S ** 2 + optimal_alpha)), UR])
        non_me_ids_train = list(set(range(reg_train_z.shape[1])) - set(me_ids))
        weight = weight[non_me_ids_train, :]
        pred = np.dot(reg_test_z, weight)

    # Compute test correlations
    nnpred = np.nan_to_num(pred)
    test_corrs = np.nan_to_num(
        np.array([pearsonr(fmri_test_z[:, ii], nnpred[:, ii].ravel())[0] for ii in range(fmri_test_z.shape[1])])
    )

    # Compute noise ceiling (split-half reliability)
    noise_ceiling = compute_noise_ceiling_split_half(
        reg_test_z,
        fmri_test_z,
        optimal_alpha,
        CONFIG['hrf_block_len'],
        CONFIG['sb_min_block_len']
    )

    # Compute corrected test correlations
    test_corrs_corrected = np.divide(
        test_corrs,
        noise_ceiling,
        out=np.zeros_like(test_corrs),
        where=noise_ceiling > 0
    )

    return {
        'weight': weight,
        'test_corrs': test_corrs,
        'noise_ceiling': noise_ceiling,
        'test_corrs_corrected': test_corrs_corrected,
        'pred_bold': nnpred  # Predicted BOLD (not z-scored in Step 1)
    }


def aggregate_fold_results(
    weight_list: List[np.ndarray],
    test_corrs_list: List[np.ndarray],
    noise_ceiling_list: List[np.ndarray],
    test_corrs_corrected_list: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Aggregate results across cross-validation folds.

    Args:
        weight_list: List of weight arrays per fold
        test_corrs_list: List of test correlation arrays per fold
        noise_ceiling_list: List of noise ceiling arrays per fold
        test_corrs_corrected_list: List of corrected test correlation arrays per fold

    Returns:
        Dictionary with aggregated results:
            - weight: Mean weight across folds
            - test_corrs: Mean test correlations across folds
            - noise_ceiling: Mean noise ceiling across folds
            - test_corrs_corrected: Mean corrected test correlations across folds

    Notes:
        - pred_bold is NOT aggregated (saved per-fold only)
        - pred_bold arrays are too large to store in fold-avg
    """
    weight_mean = np.array(weight_list).mean(axis=0)
    test_corrs_mean = np.array(test_corrs_list).mean(axis=0)
    noise_ceiling_mean = np.array(noise_ceiling_list).mean(axis=0)
    test_corrs_corrected_mean = np.array(test_corrs_corrected_list).mean(axis=0)

    return {
        'weight': weight_mean,
        'test_corrs': test_corrs_mean,
        'noise_ceiling': noise_ceiling_mean,
        'test_corrs_corrected': test_corrs_corrected_mean
    }


def process_subject(
    sub_id: str,
    test_fold_list: List[List[int]],
    optimal_alpha: float,
    gaze_weight: Tuple[float, float],
    wordnet: bool,
    hx_docu: str,
    paths: Dict[str, Path]
) -> None:
    """
    Process one subject: train models across all folds and aggregate.

    This is the main worker function for parallel processing.

    Steps:
    1. Load fMRI paths for this subject
    2. Load and prepare regressors
    3. For each fold:
       - Train model with optimal alpha
       - Save pred_bold per-fold
    4. Aggregate across folds (mean weights, mean test_corrs)
    5. Save fold-averaged results (all outputs per seed)

    Args:
        sub_id: Subject ID
        test_fold_list: List of test fold TR indices
        optimal_alpha: Ridge regularization parameter
        gaze_weight: Tuple of (bias_weight, verb_weight)
        wordnet: Whether WordNet hierarchy is used
        hx_docu: ASD by-history documentation filter
        paths: Dictionary of file paths
    """
    seed = CONFIG['seed']
    print(f'* {sub_id}')

    # Load fMRI paths
    fmri_pattern = (
        paths['fmri_path'] /
        CONFIG['conf_option'] /
        CONFIG['atlas'] /
        f"chunk-{CONFIG['chunk_option']}" /
        'fold-*' /
        'srm_dx-*' /
        f"shared_k-{CONFIG['srm_option']}" /
        f"*_inc_byhx_docu-{hx_docu}" /
        f'{sub_id}.npy'
    )

    fmri_train_paths = sorted(glob(str(fmri_pattern).replace('*_inc', 'train_inc')))
    fmri_test_paths = sorted(glob(str(fmri_pattern).replace('*_inc', 'test_inc')))

    if len(fmri_train_paths) == 0:
        print(f'  WARNING: No fMRI files found for {sub_id}, skipping')
        return

    # Load and prepare regressors
    reg_array, label_list = load_regressors(
        gaze_weight,
        wordnet,
        hx_docu,
        sub_id,
        paths
    )

    reg_delayed, me_ids, n_labels = prepare_regressors(
        reg_array,
        label_list,
        CONFIG['tr_delays_option'],
        CONFIG['start_ignore_trs'],
        CONFIG['end_ignore_trs'],
        CONFIG['motion_energy_option']
    )

    n_dm_trs = reg_delayed.shape[1]  # TRs after trimming

    # Process each fold
    weight_list = []
    test_corrs_list = []
    noise_ceiling_list = []
    test_corrs_corrected_list = []

    for fold_idx, (test_fold_ids, fmri_train_path, fmri_test_path) in enumerate(
        zip(test_fold_list, fmri_train_paths, fmri_test_paths)
    ):
        # Train model with optimal alpha
        results = train_final_model_fold(
            fold_idx,
            test_fold_ids,
            Path(fmri_train_path),
            Path(fmri_test_path),
            reg_delayed,
            n_dm_trs,
            me_ids,
            optimal_alpha
        )

        weight_list.append(results['weight'])
        test_corrs_list.append(results['test_corrs'])
        noise_ceiling_list.append(results['noise_ceiling'])
        test_corrs_corrected_list.append(results['test_corrs_corrected'])

        # Save pred_bold per-fold
        # Construct fold-specific path (similar to build_output_path_step1 but with fold-N)
        conf = CONFIG['conf_option']
        atlas = CONFIG['atlas']
        chunk = CONFIG['chunk_option']
        srm = CONFIG['srm_option']

        fold_output_path = (
            paths['out_path'] /
            conf /
            atlas /
            f'chunk-{chunk}' /
            f'fold-{fold_idx + 1}' /
            f'd-{hx_docu}' /
            f'k-{srm}'
        )

        # Add gaze weight directory
        if not gaze_weight[0]:
            fold_output_path = fold_output_path / 'gaze_weight-False'
        else:
            fold_output_path = fold_output_path / f'bias-{gaze_weight[0]}_verb-{gaze_weight[1]}'

        # Add WordNet, seed, and s_alpha directories
        fold_output_path = fold_output_path / f'wn-{wordnet}' / f'seed-{seed}' / 's_alpha-True'
        fold_output_path.mkdir(parents=True, exist_ok=True)

        # Save only pred_bold (other metrics are saved in fold-avg)
        save_files_sub({'pred_bold': results['pred_bold']}, sub_id, str(fold_output_path))

    # Aggregate across folds
    aggregated = aggregate_fold_results(
        weight_list,
        test_corrs_list,
        noise_ceiling_list,
        test_corrs_corrected_list
    )

    # Save all results (build_output_path_step1 now includes seed)
    output_path = build_output_path_step1(
        paths['out_path'],
        hx_docu,
        gaze_weight,
        wordnet
    )
    output_path.mkdir(parents=True, exist_ok=True)
    save_files_sub(aggregated, sub_id, str(output_path))


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with:
            - hx_docu (str): ASD by-history documentation filter
            - debug_subject (str): Subject ID for single-subject debugging (optional)
    """
    parser = argparse.ArgumentParser(
        description='Train final encoding models using optimal alpha',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects
  python 06_encoding_model_1optimal_alpha.py
  python 06_encoding_model_1optimal_alpha.py --hx_docu True

  # Debug single subject
  python 06_encoding_model_1optimal_alpha.py --debug_subject sub-NDARINVXXXXXXX

Notes:
  - Processes all subjects in parallel using joblib (unless --debug_subject specified)
  - Loads optimal alpha from Step 0 boot_corrs
  - Saves only fold-averaged results (no per-fold)
        """
    )

    parser.add_argument(
        '--hx_docu',
        type=str,
        default='True',
        choices=['True', 'False'],
        help='Filter subjects with ASD by-history requiring documentation'
    )

    parser.add_argument(
        '--debug_subject',
        type=str,
        default=None,
        help='Process only this subject ID for debugging (e.g., sub-NDARINVXXXXXXX)'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution flow (batch mode or single-subject debug mode).

    Workflow:
    1. Parse arguments (optional hx_docu override and debug_subject)
    2. Load and filter participants (QC filters: FD, deepmreye, prep_ok, etc.)
    3. Create cross-validation folds
    4. Set random seed from CONFIG
    5. For each parameter combination (gaze_weight × wordnet):
       - Load optimal alpha from Step 0 boot_corrs (fold-avg, seed-specific)
       - Process subjects:
         * If --debug_subject: process only that subject (no parallelization)
         * Otherwise: process all subjects in parallel using joblib
       - Each subject:
         * Load regressors
         * Train models across folds with optimal alpha
         * Aggregate fold results
         * Save all results per seed
    6. Print completion summary
    """
    # Parse arguments
    args = parse_arguments()

    # Get seed from CONFIG
    SEED = CONFIG['seed']

    print(f'\n{"="*70}')
    print(f'ENCODING MODEL - TRAIN WITH OPTIMAL ALPHA (STEP 1)')
    print(f'{"="*70}')
    print(f'ASD by-history documentation filter: {args.hx_docu}')
    print(f'Random seed: {SEED}')
    print(f'{"="*70}\n')

    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Random seed set to {SEED} for reproducibility.\n")

    # Setup paths
    paths = setup_paths(args.hx_docu)

    # Load and filter participants
    participants_df, id_list = load_and_filter_participants(args.hx_docu, paths)

    # Create cross-validation folds
    print('\n* Creating cross-validation folds')
    chunk_list, test_fold_list = create_cross_validation_folds(CONFIG['chunk_option'])
    print(f'  Created {len(test_fold_list)} folds from {CONFIG["chunk_option"]} chunks')

    # ========================================================================
    # PARAMETER GRID SEARCH
    # ========================================================================
    for gaze_weight in CONFIG['gaze_weight_grid']:
        print(f'\nRegressor - Gaze weight:')
        print(f'  bias_weight={gaze_weight[0]}, verb_weight={gaze_weight[1]}')

        for wordnet in CONFIG['wordnet_options']:
            print(f'  Regressor - WordNet hierarchy: {wordnet}')

            # Load optimal alpha from Step 0 (seed-specific path)
            print('  * Loading optimal alpha from Step 0')
            optimal_alpha = load_optimal_alpha(
                gaze_weight,
                wordnet,
                args.hx_docu,
                id_list,
                paths
            )

            # Process subjects (parallel or single debug subject)
            if args.debug_subject:
                # Debug mode: process single subject
                if args.debug_subject not in id_list:
                    print(f'  WARNING: Debug subject {args.debug_subject} not in filtered ID list')
                    print(f'  Available subjects: {id_list[:5]}... (showing first 5)')
                    continue

                print(f'  * DEBUG MODE: Processing single subject {args.debug_subject}')
                process_subject(
                    args.debug_subject, test_fold_list, optimal_alpha,
                    gaze_weight, wordnet, args.hx_docu, paths
                )
            else:
                # Normal mode: process all subjects in parallel
                print(f'  * Processing {len(id_list)} subjects in parallel')
                with Parallel(n_jobs=CONFIG['n_jobs'], verbose=10) as parallel:
                    parallel(
                        delayed(process_subject)(
                            sub_id, test_fold_list, optimal_alpha,
                            gaze_weight, wordnet, args.hx_docu, paths
                        ) for sub_id in id_list
                    )

            print(f'  * Completed parameter combination (seed={SEED})')

    print(f'\n{"="*70}')
    print(f'COMPLETED SUCCESSFULLY')
    print(f'{"="*70}\n')


if __name__ == "__main__":
    main()
