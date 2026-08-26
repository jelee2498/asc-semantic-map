"""
Ridge regression parameter search for encoding models with noise ceiling correction.

This script performs bootstrapped ridge regression to find optimal regularization
parameters (alphas) for predicting brain responses from movie regressors. It includes
advanced noise ceiling estimation using split-half reliability.

Reference Files (read-only):
    - S:\\jelee\\02_asd_semantic_map\\1_code\\17_param_search\\02_pca_group_comparison_release11_0alpha_inc_byhx_sub.py
    - S:\\jelee\\02_asd_semantic_map\\1_code\\17_param_search\\02_pca_group_comparison_release11_0alpha_inc_byhx_pbs.pbs
    - S:\\jelee\\02_asd_semantic_map\\1_code\\17_param_search\\functions.py (bootstrap_ridge, make_delayed_all)

Input:
    - Preprocessed fMRI data (SRM-aligned, .npy files per fold)
    - Movie regressors (Excel files with semantic features)
    - Optional: Gaze-weighted regressors (subject-specific)

Output:
    - boot_corrs: Bootstrapped validation correlations (20 alphas)
    - weight: Ridge regression weights (features x voxels)
    - test_corrs: Test set correlations
    - noise_ceiling: Split-half reliability estimates
    - test_corrs_corrected: Test correlations / noise ceiling

Usage:
    python 06_encoding_model_0search_alpha_sub.py <sub_id> <hx_docu>

Example:
    python 06_encoding_model_0search_alpha_sub.py sub-NDARINVXXXXXXX True
    python 06_encoding_model_0search_alpha_sub.py sub-NDARINVXXXXXXX False

Project: 02_asd_semantic_map
Pipeline: 99_main
Task: 06_encoding_model
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import chain
from functools import reduce
import itertools as itools

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from glob import glob


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


def mult_diag(d: np.ndarray, mtx: np.ndarray, left: bool = True) -> np.ndarray:
    """
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Args:
        d: 1D (N,) array (contains the diagonal elements)
        mtx: 2D (N,N) array
        left: If True, multiply from left; if False, multiply from right

    Returns:
        mult_diag(d, mts, left=True) == dot(diag(d), mtx)
        mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx


def ridge_corr(
    Rstim: np.ndarray,
    Pstim: np.ndarray,
    Rresp: np.ndarray,
    Presp: np.ndarray,
    alphas: np.ndarray,
    normalpha: bool = False,
    dtype: type = np.single,
    corrmin: float = 0.2,
    singcutoff: float = 1e-10,
    use_corr: bool = True
) -> List[np.ndarray]:
    """
    Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp].
    Then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. Note that the regression weights are NOT returned.

    Args:
        Rstim: Training stimuli with TR time points and N features (TR, N)
        Pstim: Test stimuli with TP time points and N features (TP, N)
        Rresp: Training responses with TR time points and M responses (TR, M)
        Presp: Test responses with TP time points and M responses (TP, M)
        alphas: Ridge parameters to be tested (A,)
        normalpha: Whether ridge parameters should be normalized by the Frobenius norm of Rstim
        dtype: Data type for computation
        corrmin: Display threshold for correlations
        singcutoff: Threshold for removing small singular values
        use_corr: If True, use correlation; if False, use variance explained

    Returns:
        Rcorrs: The correlation between each predicted response and each column of Presp for each alpha (A, M)
    """
    # Calculate SVD of stimulus matrix
    U, S, Vh = np.linalg.svd(Rstim, full_matrices=False)

    # Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize - ngoodS
    U = U[:, :ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]

    # Normalize alpha by the Frobenius norm
    frob = S[0]
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    # Precompute some products for speed
    UR = np.dot(U.T, Rresp)
    PVh = np.dot(Pstim, Vh.T)

    zPresp = zs(Presp)
    Prespvar = Presp.var(0)
    Rcorrs = []

    for na, a in zip(nalphas, alphas):
        D = S / (S ** 2 + na)
        pred = np.dot(mult_diag(D, PVh, left=False), UR)

        if use_corr:
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            # Compute variance explained
            resvar = (Presp - pred).var(0)
            Rcorr = np.clip(1 - (resvar / Prespvar), 0, 1)

        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)

    return Rcorrs


def bootstrap_ridge(
    Rstim: np.ndarray,
    Rresp: np.ndarray,
    Pstim: np.ndarray,
    Presp: np.ndarray,
    alphas: np.ndarray,
    nboots: int,
    chunklen: int,
    nchunks: int,
    me_ids: Optional[List[int]] = None,
    return_pred: bool = False,
    dtype: type = np.single,
    corrmin: float = 0.2,
    joined: Optional[List[np.ndarray]] = None,
    singcutoff: float = 1e-10,
    normalpha: bool = False,
    single_alpha: bool = False,
    use_corr: bool = True,
    return_test_wt: bool = False
) -> Tuple:
    """
    Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.

    Args:
        Rstim: Training stimuli (TR, N)
        Rresp: Training responses (TR, M)
        Pstim: Test stimuli (TP, N)
        Presp: Test responses (TP, M)
        alphas: Ridge parameters to test (A,)
        nboots: Number of bootstrap samples
        chunklen: Length of chunks for held-out validation
        nchunks: Number of chunks to hold out
        me_ids: Motion energy feature indices
        return_pred: Whether to return predictions
        dtype: Data type for computation
        corrmin: Display threshold
        joined: Grouped voxels to share ridge parameters
        singcutoff: Threshold for singular values
        normalpha: Whether to normalize alphas
        single_alpha: Whether to use single alpha for all voxels
        use_corr: Use correlation vs variance explained
        return_test_wt: Whether to return test weights

    Returns:
        wt: Regression weights (N, M)
        corrs: Validation set correlations (M,)
        valphas: Selected alpha per voxel (M,)
        allRcorrs: Bootstrap correlations (A, M, B)
        valinds: Validation indices (TH, B)
        nnpred: Predicted responses (optional)
        test_wt: Test weights (optional)
    """
    nresp, nvox = Rresp.shape
    bestalphas = np.zeros((nboots, nvox))
    valinds = []

    Rcmats = []
    for bi in range(nboots):
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)] * chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds) - set(heldinds))
        valinds.append(heldinds)

        RRstim = Rstim[notheldinds, :]
        PRstim = Rstim[heldinds, :]
        RRresp = Rresp[notheldinds, :]
        PRresp = Rresp[heldinds, :]

        # Run ridge regression using this test set
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                          dtype=dtype, corrmin=corrmin, singcutoff=singcutoff,
                          normalpha=normalpha, use_corr=use_corr)

        Rcmats.append(Rcmat)

    # Find weights for each voxel
    U, S, Vh = np.linalg.svd(Rstim, full_matrices=False)

    # Normalize alpha by the Frobenius norm
    frob = S[0]
    if normalpha:
        nalphas = alphas * frob
    else:
        nalphas = alphas

    allRcorrs = np.dstack(Rcmats)
    if not single_alpha:
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = nalphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                jcorrs = allRcorrs[:, jl, :].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = nalphas[bestalpha]
    else:
        meanbootcorr = allRcorrs.mean(2).mean(1)
        bestalphaind = np.argmax(meanbootcorr)
        bestalpha = alphas[bestalphaind]
        valphas = np.array([bestalpha] * nvox)

    UR = np.dot(U.T, np.nan_to_num(Rresp))
    pred = np.zeros(Presp.shape)

    if not me_ids:  # no motion energy features used
        wt = np.zeros((Rstim.shape[1], Rresp.shape[1]))
        for ai, alpha in enumerate(nalphas):
            selvox = np.nonzero(valphas == alpha)[0]
            awt = reduce(np.dot, [Vh.T, np.diag(S / (S ** 2 + alpha)), UR[:, selvox]])
            pred[:, selvox] = np.dot(Pstim, awt)
            wt[:, selvox] = awt

    else:  # motion energy features are used as regressors
        wt = np.zeros((Pstim.shape[1], Rresp.shape[1]))
        for ai, alpha in enumerate(nalphas):
            selvox = np.nonzero(valphas == alpha)[0]
            awt = reduce(np.dot, [Vh.T, np.diag(S / (S ** 2 + alpha)), UR[:, selvox]])

            non_me_ids = set(list(range(Rstim.shape[1]))) - set(me_ids)

            awt = awt[list(non_me_ids), :]
            pred[:, selvox] = np.dot(Pstim, awt)
            wt[:, selvox] = awt

    # Find test correlations
    nnpred = np.nan_to_num(pred)
    corrs = np.nan_to_num(
        np.array([np.corrcoef(Presp[:, ii], nnpred[:, ii].ravel())[0, 1] for ii in range(Presp.shape[1])]))

    if return_test_wt:
        test_U, test_S, test_Vh = np.linalg.svd(Pstim, full_matrices=False)
        test_UR = np.dot(test_U.T, np.nan_to_num(Presp))

        if not me_ids:
            test_wt = np.zeros_like(wt)
            for ai, alpha in enumerate(nalphas):
                selvox = np.nonzero(valphas == alpha)[0]
                test_awt = reduce(np.dot, [test_Vh.T, np.diag(test_S / (test_S ** 2 + alpha)), test_UR[:, selvox]])
                test_wt[:, selvox] = test_awt
        else:
            raise Exception("Not implemented yet!")

        if return_pred:
            return wt, corrs, valphas, allRcorrs, valinds, nnpred, test_wt
        else:
            return wt, corrs, valphas, allRcorrs, valinds, test_wt
    else:
        if return_pred:
            return wt, corrs, valphas, allRcorrs, valinds, nnpred
        else:
            return wt, corrs, valphas, allRcorrs, valinds


def save_files_sub(DICT: Dict[str, np.ndarray], SUB_ID: str, SAVE_PATH: str, EXIST_OK: bool = True) -> None:
    """
    Save individual results dictionary as individual files to save time.

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
    'tpl_path': TEMPLATES,

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

    # Ridge regression parameters
    'alpha_list': np.logspace(1, 4, 20),  # 20 alphas from 10 to 10000
    'n_boots': 30,  # Bootstrap samples for alpha selection
    'chunk_len': 10,  # Chunk length for bootstrap CV
    'n_heldout_factor': 4,  # n_heldout = n_chunks / 4
    'single_alpha': False,  # Use per-voxel alphas

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
    'motion_energy_option': False,  # Include motion energy features

    # Noise ceiling parameters (split-half reliability)
    'hrf_block_len': 25,  # HRF-aware block length (samples)
    'sb_min_block_len': 12,  # Minimum block length for A/B split

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
            - fmri_path: Base path to preprocessed fMRI data (from 17_param_search)
            - reg_path: Regressor data path (from 99_main/05_prepare_reg)
            - proc_path: Pipeline processing directory
            - out_path: Output directory
            - save_path: Saved results directory
            - tmp_path: Temporary files directory
    """
    proj = CONFIG['project']

    # fMRI data path (from 17_param_search)
    fmri_source_pipe = PROJECT / '2_pipeline' / CONFIG['source_pipeline_fmri']
    fmri_path = fmri_source_pipe / CONFIG['fmri_data_task'] / 'out'

    # Regressor data path (from 99_main/05_prepare_reg)
    reg_source_pipe = PROJECT / '2_pipeline' / CONFIG['source_pipeline_reg']
    reg_path = reg_source_pipe / CONFIG['reg_data_task'] / 'out'

    # Target pipeline paths (99_main)
    proc_path = CONFIG['pipe_path'] / CONFIG['task']
    out_path = proc_path / 'out'
    save_path = proc_path / 'save'
    tmp_path = proc_path / 'tmp'

    return {
        'fmri_path': fmri_path,
        'reg_path': reg_path,
        'proc_path': proc_path,
        'out_path': out_path,
        'save_path': save_path,
        'tmp_path': tmp_path
    }


def create_cross_validation_folds(
    chunk_option: int
) -> Tuple[List[np.ndarray], List[List[int]]]:
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
    alphas: np.ndarray,
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
        alphas: (n_voxels,) ridge parameters per voxel (or scalar)
        hrf_block_len: Block length in samples for A/B split (default: 25)
        min_block_len: Minimum block length fallback (default: 12)

    Returns:
        noise_ceiling: (n_voxels,) ceiling values in [0, 1]

    Notes:
        - Uses per-voxel alphas if provided (more accurate than scalar)
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

    def _ridge_fit_masked_per_voxel(
        X: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray,
        alphas_v: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Ridge regression with per-voxel regularization.

        Uses eigendecomposition of X'X to efficiently solve for all voxels
        with different lambda values in one pass.
        """
        X_masked = X[mask, :]
        Y_masked = Y[mask, :]
        P = X_masked.shape[1]

        # Precompute X'X and X'Y
        XtX = X_masked.T @ X_masked
        XtY = X_masked.T @ Y_masked

        # Eigendecompose X'X (add jitter for stability)
        evals, Q = np.linalg.eigh(XtX + eps * np.eye(P))

        # Solve in eigen-space: beta = Q @ ((Q'X'Y) / (evals + alpha))
        TQ = Q.T @ XtY  # (P, V)
        denom = evals[:, None] + alphas_v[None, :]  # (P, V)
        TQ_scaled = TQ / denom
        beta = Q @ TQ_scaled

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

    # Step 2: Fit ridge models on disjoint samples
    alphas_v = np.asarray(alphas).ravel()

    # Check if we can use scalar lambda (all alphas equal)
    if alphas_v.size == 1 or np.allclose(alphas_v, alphas_v[0], equal_nan=False):
        # Scalar case: more efficient
        lam_scalar = float(alphas_v[0])
        betaA = _ridge_fit_masked(reg_test_z, fmri_test_z, maskA, lam_scalar)
        betaB = _ridge_fit_masked(reg_test_z, fmri_test_z, maskB, lam_scalar)
    else:
        # Per-voxel case: handle NaNs
        if np.any(~np.isfinite(alphas_v)):
            finite_mask = np.isfinite(alphas_v)
            fill_val = np.nanmedian(alphas_v[finite_mask]) if np.any(finite_mask) else 0.0
            alphas_v = np.where(np.isfinite(alphas_v), alphas_v, fill_val)

        betaA = _ridge_fit_masked_per_voxel(reg_test_z, fmri_test_z, maskA, alphas_v)
        betaB = _ridge_fit_masked_per_voxel(reg_test_z, fmri_test_z, maskB, alphas_v)

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


def process_single_fold(
    fold_idx: int,
    test_fold_ids: List[int],
    fmri_train_path: Path,
    fmri_test_path: Path,
    reg_delayed: np.ndarray,
    n_dm_trs: int,
    me_ids: Optional[List[int]],
    alpha_list: np.ndarray,
    n_boots: int,
    chunk_len: int,
    n_heldout_factor: int,
    single_alpha: bool,
    hrf_block_len: int,
    min_block_len: int
) -> Dict[str, np.ndarray]:
    """
    Process one cross-validation fold: train encoding model and compute metrics.

    This is the core computational function that:
    1. Loads train/test fMRI data
    2. Splits regressors into train/test
    3. Z-scores all data
    4. Runs bootstrapped ridge regression
    5. Computes noise ceiling via split-half reliability
    6. Computes corrected test correlations

    Args:
        fold_idx: Fold index (0-indexed)
        test_fold_ids: TR indices in test fold
        fmri_train_path: Path to training fMRI .npy file
        fmri_test_path: Path to test fMRI .npy file
        reg_delayed: (n_features * n_delays, n_trs) delayed regressors
        n_dm_trs: Total number of TRs (after trimming)
        me_ids: Motion energy feature indices (or None)
        alpha_list: Ridge parameters to search
        n_boots: Number of bootstrap samples
        chunk_len: Bootstrap chunk length
        n_heldout_factor: Divisor for n_heldout calculation
        single_alpha: Whether to use single alpha for all voxels
        hrf_block_len: HRF block length for noise ceiling
        min_block_len: Minimum block length for noise ceiling

    Returns:
        Dictionary with keys:
            - boot_corrs: (20,) mean bootstrap correlations per alpha
            - weight: (n_features * n_delays, n_voxels) regression weights
            - test_corrs: (n_voxels,) test correlations
            - noise_ceiling: (n_voxels,) split-half reliability estimates
            - test_corrs_corrected: (n_voxels,) test_corrs / noise_ceiling
            - pred_bold: (n_test_trs, n_voxels) z-scored predicted BOLD signal

    Notes:
        - All data is z-scored before regression
        - Motion energy features (if present) are excluded from test normalization
        - Predicted BOLD is z-scored within each voxel before returning
        - pred_bold is saved per-fold but not aggregated (too large for fold-avg)
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

    # Bootstrap parameters
    n_chunks = int(reg_train_z.shape[0] / chunk_len)
    n_heldout = int(n_chunks / n_heldout_factor)

    # Run bootstrapped ridge regression
    weight, test_corrs, alphas, boot_corrs, _, nnpred = bootstrap_ridge(
        reg_train_z,
        fmri_train_z,
        reg_test_z,
        fmri_test_z,
        alpha_list,
        n_boots,
        chunk_len,
        n_heldout,
        me_ids=me_ids,
        return_pred=True,
        single_alpha=single_alpha
    )

    # Compute noise ceiling (split-half reliability)
    noise_ceiling = compute_noise_ceiling_split_half(
        reg_test_z,
        fmri_test_z,
        alphas,
        hrf_block_len,
        min_block_len
    )

    # Compute corrected test correlations
    test_corrs_corrected = np.divide(
        test_corrs,
        noise_ceiling,
        out=np.zeros_like(test_corrs),
        where=noise_ceiling > 0
    )

    # Z-normalize predicted BOLD within each voxel
    nnpred_z = zscore(nnpred, axis=0)

    return {
        'boot_corrs': boot_corrs.mean((1, 2)),  # (20,) mean across voxels and boots
        'weight': weight,
        'test_corrs': test_corrs,
        'noise_ceiling': noise_ceiling,
        'test_corrs_corrected': test_corrs_corrected,
        'pred_bold': nnpred_z,  # Z-scored predicted BOLD (saved per-fold only)
    }


def aggregate_results(
    results_list: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    Aggregate results across cross-validation folds.

    Combines per-fold results into fold-averaged metrics:
    - boot_corrs: Mean across folds
    - weight: Z-score within voxels, then mean across folds
    - test_corrs: Mean across folds
    - noise_ceiling: Mean across folds
    - test_corrs_corrected: Mean across folds

    Args:
        results_list: List of per-fold result dictionaries

    Returns:
        Dictionary with aggregated results (same keys as input)

    Notes:
        - Weights are z-scored before averaging because each voxel may have
          different optimal alphas across folds
        - pred_bold is NOT aggregated (only saved per-fold)
    """
    # Extract per-fold arrays
    boot_corrs_list = [r['boot_corrs'] for r in results_list]
    weight_list = [r['weight'] for r in results_list]
    test_corrs_list = [r['test_corrs'] for r in results_list]
    noise_ceiling_list = [r['noise_ceiling'] for r in results_list]
    test_corrs_corrected_list = [r['test_corrs_corrected'] for r in results_list]

    # Aggregate boot_corrs
    boot_corrs_mean = np.array(boot_corrs_list).mean(axis=0)

    # Aggregate weights (z-score within voxels first)
    weight_list_z = [zscore(w, axis=0) for w in weight_list]
    weight_mean = np.array(weight_list_z).mean(axis=0)

    # Aggregate test metrics
    test_corrs_mean = np.array(test_corrs_list).mean(axis=0)
    noise_ceiling_mean = np.array(noise_ceiling_list).mean(axis=0)
    test_corrs_corrected_mean = np.array(test_corrs_corrected_list).mean(axis=0)

    return {
        'boot_corrs': boot_corrs_mean,
        'weight': weight_mean,
        'test_corrs': test_corrs_mean,
        'noise_ceiling': noise_ceiling_mean,
        'test_corrs_corrected': test_corrs_corrected_mean
    }


def build_output_path(
    base_path: Path,
    fold_name: str,
    hx_docu: str,
    gaze_weight: Tuple[float, float],
    wordnet: bool,
    single_alpha: bool,
    include_s_alpha: bool = True
) -> Path:
    """
    Build hierarchical output path following abbreviated naming convention.

    Uses shortened names to avoid Windows 260-character path limit:
    - 'd-{hx_docu}' instead of 'results_inc_byhx_docu-{hx_docu}'

    Args:
        base_path: Base output directory
        fold_name: Fold identifier (e.g., 'fold-1', 'fold-avg')
        hx_docu: ASD by-history documentation filter ('True' or 'False')
        gaze_weight: Tuple of (bias_weight, verb_weight)
        wordnet: Whether WordNet hierarchy is used
        single_alpha: Whether single alpha is used
        include_s_alpha: Whether to include 's_alpha-{single_alpha}' in path

    Returns:
        Full output path

    Example:
        base_path / 'default+me' / 'mmp' / 'chunk-9' / 'fold-1' /
        'd-True' / 'k-0' / 'bias-0.5_verb-0.5' / 'wn-True' / 'seed-0' / 's_alpha-False'
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
        fold_name /
        f'd-{hx_docu}' /  # Abbreviated
        f'k-{srm}'
    )

    # Add gaze weight directory
    if not gaze_weight[0]:
        path = path / 'gaze_weight-False'
    else:
        path = path / f'bias-{gaze_weight[0]}_verb-{gaze_weight[1]}'

    # Add WordNet directory
    path = path / f'wn-{wordnet}'

    # Add seed directory
    path = path / f'seed-{seed}'

    # Add single_alpha directory (if requested)
    if include_s_alpha:
        path = path / f's_alpha-{single_alpha}'

    return path


def save_fold_results(
    results: Dict[str, np.ndarray],
    sub_id: str,
    output_path: Path,
    include_pred_bold: bool = False
) -> None:
    """
    Save results for one fold or aggregated results.

    Args:
        results: Dictionary of result arrays
        sub_id: Subject ID
        output_path: Output directory
        include_pred_bold: Whether to include predicted BOLD in saved results

    Notes:
        - Creates output directory if it doesn't exist
        - Uses save_files_sub() which creates subdirectories per key
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results dictionary
    if include_pred_bold and 'pred_bold' in results:
        save_dict = results.copy()
    else:
        # Exclude pred_bold for summary results
        save_dict = {k: v for k, v in results.items() if k != 'pred_bold'}

    # Save using functions.py utility
    save_files_sub(save_dict, sub_id, str(output_path))


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for encoding model parameter search.

    Returns:
        argparse.Namespace with:
            - sub_id (str): Subject ID
            - hx_docu (str): ASD by-history documentation filter
    """
    parser = argparse.ArgumentParser(
        description='Ridge regression parameter search for encoding models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 06_encoding_model_0search_alpha_sub.py sub-NDARINVXXXXXXX True
  python 06_encoding_model_0search_alpha_sub.py sub-NDARINVXXXXXXX False

Notes:
  - Processes one subject (designed for PBS array job submission)
  - Performs grid search over gaze weights and WordNet options
  - Saves results per fold and fold-averaged
  - Includes noise ceiling correction via split-half reliability
        """
    )

    parser.add_argument(
        'sub_id',
        type=str,
        help='Subject ID in BIDS format (e.g., sub-NDARINVXXXXXXX)'
    )

    parser.add_argument(
        'hx_docu',
        type=str,
        choices=['True', 'False'],
        help=(
            'Filter subjects with ASD by-history requiring documentation. '
            'True: only include with documentation. False: include all.'
        )
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution flow for encoding model parameter search.

    Workflow:
    1. Parse command-line arguments
    2. Setup file paths
    3. Create cross-validation folds
    4. Set random seed from CONFIG
    5. Iterate over parameter grid:
       - Load regressors (gaze weight x WordNet)
       - Prepare delayed regressors
       - Process each CV fold:
         * Train encoding model
         * Compute noise ceiling
         * Save per-fold results
       - Aggregate across folds
       - Save fold-averaged results (all outputs per seed)

    Grid search parameters:
    - gaze_weight: 11 options (False, 0.1-0.9 with various verb weights)
    - wordnet: 1 option (True)
    Total: 11 parameter combinations per subject per seed
    """
    # Parse arguments
    args = parse_arguments()

    # Get seed from CONFIG
    SEED = CONFIG['seed']

    print(f'\n{"="*70}')
    print(f'ENCODING MODEL PARAMETER SEARCH')
    print(f'{"="*70}')
    print(f'Subject: {args.sub_id}')
    print(f'ASD by-history documentation filter: {args.hx_docu}')
    print(f'Random seed: {SEED}')
    print(f'{"="*70}\n')

    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Random seed set to {SEED} for reproducibility.\n")

    # Setup paths
    paths = setup_paths(args.hx_docu)

    # Create cross-validation folds
    chunk_list, test_fold_list = create_cross_validation_folds(
        CONFIG['chunk_option']
    )

    # Get fMRI file paths (one per fold)
    fmri_pattern = (
        paths['fmri_path'] /
        CONFIG['conf_option'] /
        CONFIG['atlas'] /
        f"chunk-{CONFIG['chunk_option']}" /
        'fold-*' /
        'srm_dx-*' /
        f"shared_k-{CONFIG['srm_option']}" /
        f"*_inc_byhx_docu-{args.hx_docu}" /
        f'{args.sub_id}.npy'
    )

    fmri_train_paths = sorted(glob(str(fmri_pattern).replace('*_inc', 'train_inc')))
    fmri_test_paths = sorted(glob(str(fmri_pattern).replace('*_inc', 'test_inc')))

    if len(fmri_train_paths) == 0:
        raise FileNotFoundError(
            f'No fMRI files found for {args.sub_id} with hx_docu={args.hx_docu}'
        )

    print(f'Found {len(fmri_train_paths)} CV folds\n')

    # ========================================================================
    # PARAMETER GRID SEARCH
    # ========================================================================
    for gaze_weight in CONFIG['gaze_weight_grid']:
        print(f'\nRegressor - Gaze weight:')
        print(f'  bias_weight={gaze_weight[0]}, verb_weight={gaze_weight[1]}')

        for wordnet in CONFIG['wordnet_options']:
            print(f'  Regressor - WordNet hierarchy: {wordnet}')

            # Load regressors
            print('  * Loading movie regressors')
            reg_array, label_list = load_regressors(
                gaze_weight,
                wordnet,
                args.hx_docu,
                args.sub_id,
                paths
            )

            # Prepare regressors (add noise, create delays, trim TRs)
            print('  * Preparing delayed regressors')
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
            print('  * Training encoding models across folds')
            fold_results = []

            for fold_idx, (test_fold_ids, fmri_train_path, fmri_test_path) in enumerate(
                tqdm(
                    zip(test_fold_list, fmri_train_paths, fmri_test_paths),
                    desc='    Processing folds',
                    total=len(test_fold_list)
                )
            ):
                # Process single fold
                results = process_single_fold(
                    fold_idx,
                    test_fold_ids,
                    Path(fmri_train_path),
                    Path(fmri_test_path),
                    reg_delayed,
                    n_dm_trs,
                    me_ids,
                    CONFIG['alpha_list'],
                    CONFIG['n_boots'],
                    CONFIG['chunk_len'],
                    CONFIG['n_heldout_factor'],
                    CONFIG['single_alpha'],
                    CONFIG['hrf_block_len'],
                    CONFIG['sb_min_block_len']
                )

                fold_results.append(results)

                # Save pred_bold per-fold
                pred_bold_path = build_output_path(
                    paths['out_path'],
                    f'fold-{fold_idx + 1}',
                    args.hx_docu,
                    gaze_weight,
                    wordnet,
                    CONFIG['single_alpha'],
                    include_s_alpha=True
                )
                save_fold_results(
                    {'pred_bold': results['pred_bold']},
                    args.sub_id,
                    pred_bold_path,
                    include_pred_bold=True
                )

            # Aggregate results across folds
            print('  * Aggregating results across folds')
            aggregated = aggregate_results(fold_results)

            # Save all results (boot_corrs at wn level, others at s_alpha level)
            # 1. Save boot_corrs (at wn-{wordnet}/seed-{seed}/ level, before s_alpha)
            boot_corrs_avg_path = build_output_path(
                paths['out_path'],
                'fold-avg',
                args.hx_docu,
                gaze_weight,
                wordnet,
                CONFIG['single_alpha'],
                include_s_alpha=False
            )
            save_fold_results(
                {'boot_corrs': aggregated['boot_corrs']},
                args.sub_id,
                boot_corrs_avg_path,
                include_pred_bold=False
            )

            # 2. Save weight
            weight_path = build_output_path(
                paths['out_path'],
                'fold-avg',
                args.hx_docu,
                gaze_weight,
                wordnet,
                CONFIG['single_alpha'],
                include_s_alpha=True
            )
            save_fold_results(
                {'weight': aggregated['weight']},
                args.sub_id,
                weight_path,
                include_pred_bold=False
            )

            # 3. Save detailed results (test_corrs, noise_ceiling, test_corrs_corrected)
            detailed_avg_path = build_output_path(
                paths['out_path'],
                'fold-avg',
                args.hx_docu,
                gaze_weight,
                wordnet,
                CONFIG['single_alpha'],
                include_s_alpha=True
            )
            aggregated_detailed = {
                k: v for k, v in aggregated.items()
                if k not in ['boot_corrs', 'weight']
            }
            save_fold_results(
                aggregated_detailed,
                args.sub_id,
                detailed_avg_path,
                include_pred_bold=False
            )

            print(f'  * Completed parameter combination (seed={SEED})')

    print(f'\n{"="*70}')
    print(f'COMPLETED SUCCESSFULLY')
    print(f'{"="*70}\n')


if __name__ == "__main__":
    main()
