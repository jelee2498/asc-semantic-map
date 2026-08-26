"""
Encoding Model Performance Mask Analysis
==========================================

Evaluates encoding model performance by calculating correlations between
averaged actual and predicted BOLD responses across participants. Identifies
significantly predicted brain regions using multiple comparison correction.

Analysis workflow:
    1. Filter participants by quality criteria (motion, preprocessing, eye-tracking)
    2. Load actual and predicted BOLD time series from cross-validation folds
    3. Average BOLD responses across all participants
    4. Calculate ROI-wise Pearson correlations between actual and predicted
    5. Apply FDR correction to identify significantly predicted regions
    6. Save correlation maps and significance masks
    7. Visualize results on brain surface (Windows only)

Input requirements:
    - Participant metadata: participants_df_deepmreye_inc_byhx.xlsx
    - Actual BOLD: Preprocessed fMRI data per subject/fold
    - Predicted BOLD: Encoding model predictions per parameter combination
    - MMP atlas: Brain parcellation and surface meshes

Output (saved to 99_main/06_encoding_model/out/):
    - test_corrs_of_avg.npy: Correlation per ROI (n_rois,)
    - test_corrs_pval_of_avg.npy: P-values per ROI (n_rois,)
    - mask_fdr-{alpha}.npy: Significant ROI indices after correction

Usage:
    python 06_encoding_model_2perf_mask_avg.py

Note:
    This is a batch analysis script (no command-line arguments).
    Parameter search loops run over gaze weights, WordNet features, and alpha strategies.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Union
from itertools import chain
from math import sqrt

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.stats import pearsonr
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES  # noqa: E402
from statsmodels.stats.multitest import multipletests

# TODO: Refactor functions.py to modular structure
# For now, we import only what's absolutely needed
import nibabel as nib

# Conditional visualization imports (Windows only)
if os.name == 'nt':
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.vtk_interface import wrap_vtk, serial_connect
    from vtk import vtkPolyDataNormals


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '06_encoding_model',

    # Source data pipelines (where to read inputs from)
    'source_pipeline_fmri': '17_param_search',  # fMRI from 17_param_search
    'source_pipeline_reg': '99_main',           # Regressors from 99_main
    'source_task_fmri': '00_prepare_data_release11',
    'source_task_reg': '05_prepare_reg',

    # Data parameters
    'atlas': 'mmp',                    # Parcellation atlas
    'conf_option': 'default+me',       # Confounds denoising option
    'chunk_option': 9,                 # K-fold chunks (9 or 16)
    'srm_option': 0,                   # SRM features (0 = no SRM)
    'me_option': False,                # Motion energy features

    # TR trimming
    'start_ignore_trs': 5,
    'end_ignore_trs': 5,
    'tr_delays_option': [3, 5, 7, 9],

    # Quality control thresholds
    'fd_thres': 0.5,                   # Framewise displacement (mm)
    'hx_docu': 'True',                 # ASD documentation requirement

    # Analysis parameters
    'dx': 'all',                       # Diagnosis filter ('TD', 'ASD', 'all')
    'param_search': {
        'gaze_weights': [
            (False, False),            # No gaze weight
            (0.1, 0.5), (0.1, 0.55), (0.1, 0.1), (0.1, 1.0),
            (0.3, 0.5), (0.3, 0.65), (0.3, 0.3), (0.3, 1.0),
            (0.5, 0.5), (0.5, 0.75), (0.5, 1.0), 
            (0.7, 0.5), (0.7, 0.85), (0.7, 0.7), (0.7, 1.0),
            (0.9, 0.5), (0.9, 0.95), (0.9, 0.9), (0.9, 1.0)
        ],
        'wordnet_hierarchy': [True],   # WordNet feature
        'single_alpha': [False, True], # Encoding model alpha strategy
    },

    # Statistical testing
    'perf_methods': ['fdr_bh', 'bonferroni'],        # Multiple comparison correction
    'perf_alphas': [0.05, 0.01],             # Significance thresholds

    # Atlas resolution
    'n_rois': 360,                     # MMP parcels
    'n_vertices_32k': 64984,           # fsLR 32k vertices

    # Random seed for reproducibility (0, 37, 42)
    'seed': 0,
}


# =============================================================================
# PATH HANDLING
# =============================================================================

def get_project_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Initialize all project paths with OS-agnostic handling.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of project paths including:
            - Source paths: Read fMRI from 17_param_search, participant data from 99_main/05_prepare_reg
            - Output paths: Write results to 99_main/06_encoding_model
    """
    # Paths come from config/paths.yml via lib/project_config.py.  The original
    # working tree hard-coded a storage root here; that is machine-specific and
    # is not distributed.
    proj_root = PROJECT

    # fMRI source paths (READ from 17_param_search)
    fmri_source_pipe = proj_root / '2_pipeline' / config['source_pipeline_fmri']
    fmri_path = fmri_source_pipe / config['source_task_fmri'] / 'out'

    # Regressor and participant source paths (READ from 99_main/05_prepare_reg)
    reg_source_pipe = proj_root / '2_pipeline' / config['source_pipeline_reg']
    reg_path = reg_source_pipe / config['source_task_reg'] / 'out'

    # Target pipeline paths (WRITE to 99_main/06_encoding_model)
    target_pipe_root = proj_root / '2_pipeline' / config['pipeline']
    target_proc_path = target_pipe_root / config['task']

    return {
        'raw': proj_root / '0_data' / 'raw',
        'code': proj_root / '1_code' / config['pipeline'],
        'template': TEMPLATES,
        'mmp_atlas': TEMPLATES / 'MMP',

        # Source paths (read from different pipelines)
        'participants_df_path': reg_path,
        'fmri_path': fmri_path,

        # Target paths (write to 99_main/06_encoding_model)
        'pipeline': target_pipe_root,
        'proc': target_proc_path,
        'pred_bold_path': target_proc_path / 'out',
        'out_path': target_proc_path / 'out',
    }


# =============================================================================
# PARTICIPANT FILTERING
# =============================================================================

def filter_participants(
    participants_df: pd.DataFrame,
    dx: str,
    fd_thres: float,
    hx_docu: str
) -> Tuple[List[str], pd.DataFrame]:
    """Apply multi-stage quality filters to participants.

    Filtering stages:
        1. Diagnosis (TD, ASD, or all)
        2. By-history ASD documentation (if hx_docu='True')
        3. Preprocessing completed successfully
        4. No preprocessing remarks
        5. Motion (framewise displacement < threshold)
        6. DeepMReye eye-tracking quality rating

    Args:
        participants_df: DataFrame with participant metadata
        dx: Diagnosis filter ('TD', 'ASD', 'all')
        fd_thres: Framewise displacement threshold (mm)
        hx_docu: Include only documented by-history ASD subjects ('True'/'False')

    Returns:
        (filtered_id_list, filtered_dataframe)
    """
    # Filter #1: Diagnosis
    if dx == 'TD':
        dx_filt = (participants_df['DX'] == 'TD').values
    elif dx == 'ASD':
        dx_filt = (participants_df['DX'] == 'ASD').values
    elif dx == 'all':
        dx_filt = np.ones(len(participants_df), dtype=bool)
    else:
        raise ValueError(f"Invalid dx option: {dx}. Must be 'TD', 'ASD', or 'all'")

    # Filter #2: By-history documentation
    # If 'ASD_certainty' is 'by-history', only include subjects with
    # 'ASD_document' == 'documentation provided'
    if hx_docu == 'True':
        for sub_id in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                print(f"Exclude {sub_id} due to 'ASD_certainty' is 'by-history' "
                      f"and 'ASD_document' is not 'documentation provided'")
                dx_filt[participants_df.index == sub_id] = False

    # Filter #3: Preprocessing OK
    prep_filt = (participants_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] == 1).values

    # Filter #4: No remarks during preprocessing
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Apply filters 1-4
    id_list_dx_prep_remark = list(participants_df[dx_filt & prep_filt & no_remarks_filt].index)
    participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

    # Filter #5: Framewise displacement
    fd_filt = (participants_df_dx_prep_remark['Mean_FD_DM'] < fd_thres).values

    # Filter #6: DeepMReye quality
    deepmreye_filt = (participants_df_dx_prep_remark['Rating_deepmreye_movieDM'] == 1).values

    # Final filtered list
    id_list = list(participants_df_dx_prep_remark[fd_filt & deepmreye_filt].index)

    return id_list, participants_df.loc[id_list]


def print_participant_summary(participants_df: pd.DataFrame) -> None:
    """Print demographic summary of filtered participants.

    Args:
        participants_df: Filtered participant DataFrame
    """
    n = len(participants_df)
    print(f'\nParticipants information (n={n}):')

    # Diagnosis breakdown
    print(' - DX')
    n_td = len(participants_df[participants_df['DX'] == 'TD'])
    n_asd = len(participants_df[participants_df['DX'] == 'ASD'])
    print(f"   TD: {n_td}, ASD: {n_asd}")

    # Site breakdown
    print(' - Site')
    n_cbic = len(participants_df[participants_df['Site'] == 'CBIC'])
    n_ru = len(participants_df[participants_df['Site'] == 'RU'])
    print(f"   CBIC: {n_cbic}, RU: {n_ru}")

    # Sex breakdown
    print(' - Sex')
    n_male = len(participants_df[participants_df['Sex'] == 'Male'])
    n_female = len(participants_df[participants_df['Sex'] == 'Female'])
    print(f"   Male: {n_male}, Female: {n_female}")


# =============================================================================
# CROSS-VALIDATION SETUP
# =============================================================================

def create_cv_folds(chunk_option: int) -> List[List[int]]:
    """Create k-fold cross-validation fold indices.

    Splits movie TRs into chunks, then merges chunks into folds.
    For chunk_option=9: Creates 9 chunks, merges into 3 folds
    For chunk_option=16: Creates 16 chunks, merges into 4 folds

    Args:
        chunk_option: 9 or 16 chunks

    Returns:
        List of test fold TR indices (each fold is a list of TR indices)

    Raises:
        ValueError: If chunk_option is not 9 or 16
    """
    if chunk_option == 9:
        dm_chunk_list = [
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
        dm_chunk_list = [
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
        raise ValueError(f"Invalid chunk_option: {chunk_option}. Must be 9 or 16")

    # Merge chunks into folds
    n_merge = int(np.sqrt(chunk_option))
    test_fold_list = []
    for i in range(n_merge):
        merge_chunk_ids = [i + n_merge * ii for ii in range(n_merge)]
        dm_fold_list = [dm_chunk_list[merge_chunk_idx] for merge_chunk_idx in merge_chunk_ids]
        test_fold_list.append(list(chain(*dm_fold_list)))

    return test_fold_list


# =============================================================================
# ATLAS AND VISUALIZATION SETUP
# =============================================================================

def load_mmp_atlas(atlas_path: Path) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Load MMP atlas with medial wall handling.

    Args:
        atlas_path: Path to MMP atlas folder

    Returns:
        (mmp_atlas_full, nonmed_vertex_ids)
        - mmp_atlas_full: Full 32k vertex atlas with medial wall as 0
        - nonmed_vertex_ids: Indices of non-medial wall vertices
    """
    # Load MMP atlas (non-medial wall only)
    mmp_atlas_file = atlas_path / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    mmp_atlas_nonmed = nib.load(str(mmp_atlas_file)).get_fdata()[0].astype(np.int32)

    # Load medial wall mask
    med_wall_file = atlas_path / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
    med_32k = nib.load(str(med_wall_file)).get_fdata()[0].astype(np.int32).nonzero()[0]

    # Create non-medial wall vertex IDs
    nonmed_32k_ids = np.array(list(set(range(64984)) - set(list(med_32k))))

    # Create full atlas (64984 vertices) with medial wall = 0
    mmp_atlas = np.zeros(64984, dtype=np.int32)
    mmp_atlas[nonmed_32k_ids] = mmp_atlas_nonmed

    return mmp_atlas, nonmed_32k_ids


def setup_visualization(
    paths: Dict[str, Path],
    atlas: str
) -> Tuple[Any, Any, npt.NDArray[np.int32]]:
    """Load brain surfaces and atlas for visualization (Windows only).

    Args:
        paths: Dictionary of project paths
        atlas: Atlas name ('mmp', 'sch-400', 'sch-800')

    Returns:
        (surf_lh, surf_rh, label_atlas)
    """
    atlas_path = paths['mmp_atlas']

    # Load surfaces
    lh_surf_file = atlas_path / 'Q1-Q6_RelatedParcellation210.L.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'
    rh_surf_file = atlas_path / 'Q1-Q6_RelatedParcellation210.R.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'

    surf_lh = read_surface(str(lh_surf_file))
    surf_rh = read_surface(str(rh_surf_file))

    # Apply normals
    nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
    surf_lh = serial_connect(surf_lh, nf)
    surf_rh = serial_connect(surf_rh, nf)

    # Load atlas labels
    if atlas == 'mmp':
        label_atlas, _ = load_mmp_atlas(atlas_path)
    elif atlas == 'sch-400':
        sch_file = paths['template'] / 'schaefer/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii'
        label_atlas = nib.load(str(sch_file)).get_fdata().astype(np.int32).squeeze()
    elif atlas == 'sch-800':
        sch_file = paths['template'] / 'schaefer/Parcellations/HCP/fslr32k/cifti/Schaefer2018_800Parcels_17Networks_order.dlabel.nii'
        label_atlas = nib.load(str(sch_file)).get_fdata().astype(np.int32).squeeze()
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    return surf_lh, surf_rh, label_atlas


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bold_predictions(
    id_list: List[str],
    test_fold_list: List[List[int]],
    fmri_path: Path,
    pred_bold_path: Path,
    config: Dict[str, Any],
    param_reg_gaze_weight: Tuple[Union[float, bool], Union[float, bool]],
    param_reg_wordnet: bool,
    param_enc_single_alpha: bool
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load and average actual/predicted BOLD across participants.

    Reads actual BOLD from preprocessed fMRI and predicted BOLD from
    encoding model outputs (from source pipeline 17_param_search).

    Args:
        id_list: Subject IDs
        test_fold_list: CV fold indices
        fmri_path: Path to actual fMRI data (from 17_param_search)
        pred_bold_path: Path to predicted BOLD results (from 17_param_search)
        config: Configuration dictionary
        param_reg_gaze_weight: Gaze weight parameters (bias_weight, verb_weight)
        param_reg_wordnet: WordNet hierarchy flag
        param_enc_single_alpha: Single alpha flag

    Returns:
        (actual_bold_mean, pred_bold_mean) - shape (n_rois, n_trs)
    """
    n_rois = config['n_rois']
    n_trs = int(np.max(list(chain(*test_fold_list)))) + 1  # Max TR index + 1

    actual_bold_list = []
    pred_bold_list = []

    for sub_id in tqdm(id_list, desc='Loading BOLD data'):
        actual_bold = np.zeros((n_rois, n_trs))
        pred_bold = np.zeros((n_rois, n_trs))

        for fold_idx, test_fold in enumerate(test_fold_list):
            # Load actual BOLD
            actual_pattern = str(fmri_path / config['conf_option'] / config['atlas'] /
                               f"chunk-{config['chunk_option']}" / f"fold-{fold_idx+1}" /
                               'srm_dx-*' / f"shared_k-{config['srm_option']}" /
                               f"test_inc_byhx_docu-{config['hx_docu']}" / f"{sub_id}.npy")
            actual_files = glob(actual_pattern)
            if len(actual_files) == 0:
                raise FileNotFoundError(f"No actual BOLD file found for {sub_id} fold {fold_idx+1}")
            actual_bold[:, test_fold] = np.load(actual_files[0])

            # Load predicted BOLD (from source pipeline, now with seed)
            seed = config['seed']
            if param_reg_gaze_weight[0] is False:  # No gaze weight
                pred_file = pred_bold_path / config['conf_option'] / config['atlas'] / \
                           f"chunk-{config['chunk_option']}" / f"fold-{fold_idx+1}" / \
                           f"d-{config['hx_docu']}" / \
                           f"k-{config['srm_option']}" / 'gaze_weight-False' / \
                           f"wn-{param_reg_wordnet}" / f"seed-{seed}" / \
                           f"s_alpha-{param_enc_single_alpha}" / \
                           'pred_bold' / f"{sub_id}.npy"
            else:  # With gaze weight
                pred_file = pred_bold_path / config['conf_option'] / config['atlas'] / \
                           f"chunk-{config['chunk_option']}" / f"fold-{fold_idx+1}" / \
                           f"d-{config['hx_docu']}" / \
                           f"k-{config['srm_option']}" / \
                           f"bias-{param_reg_gaze_weight[0]}_verb-{param_reg_gaze_weight[1]}" / \
                           f"wn-{param_reg_wordnet}" / f"seed-{seed}" / \
                           f"s_alpha-{param_enc_single_alpha}" / \
                           'pred_bold' / f"{sub_id}.npy"

            pred_bold[:, test_fold] = np.load(str(pred_file)).T

        actual_bold_list.append(actual_bold)
        pred_bold_list.append(pred_bold)

    # Average across participants
    actual_bold_mean = np.mean(np.array(actual_bold_list), axis=0)
    pred_bold_mean = np.mean(np.array(pred_bold_list), axis=0)

    return actual_bold_mean, pred_bold_mean


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calculate_correlations(
    actual_bold: npt.NDArray[np.float64],
    pred_bold: npt.NDArray[np.float64],
    alternative: str = 'greater'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate Pearson correlations and p-values per ROI.

    Args:
        actual_bold: Ground truth BOLD (n_rois, n_trs)
        pred_bold: Predicted BOLD (n_rois, n_trs)
        alternative: Hypothesis test direction ('greater', 'less', 'two-sided')

    Returns:
        (correlations, p_values) - shape (n_rois,)
    """
    n_rois = actual_bold.shape[0]
    test_corrs = np.zeros(n_rois)
    test_corrs_pval = np.zeros(n_rois)

    for roi_idx in range(n_rois):
        # Calculate Pearson correlation
        if os.name == 'nt':
            # Windows: scipy supports alternative parameter
            test_corrs[roi_idx], test_corrs_pval[roi_idx] = pearsonr(
                actual_bold[roi_idx, :],
                pred_bold[roi_idx, :],
                alternative=alternative
            )
        else:
            # Linux: manual p-value calculation
            test_corrs[roi_idx] = pearsonr(actual_bold[roi_idx, :], pred_bold[roi_idx, :])[0]

            # Manual p-value calculation using beta distribution
            n = actual_bold.shape[1]
            ab = n / 2 - 1
            dist = stats.beta(ab, ab, loc=-1, scale=2)

            if alternative == 'two-sided':
                prob = 2 * dist.sf(abs(test_corrs[roi_idx]))
            elif alternative == 'less':
                prob = dist.cdf(test_corrs[roi_idx])
            elif alternative == 'greater':
                prob = dist.sf(test_corrs[roi_idx])
            else:
                raise ValueError(f"Invalid alternative: {alternative}")

            test_corrs_pval[roi_idx] = prob

    return test_corrs, test_corrs_pval


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_performance_results(
    out_path: Path,
    test_corrs: npt.NDArray[np.float64],
    test_corrs_pval: npt.NDArray[np.float64],
    config: Dict[str, Any],
    param_reg_gaze_weight: Tuple[Union[float, bool], Union[float, bool]],
    param_reg_wordnet: bool,
    param_enc_single_alpha: bool,
    mask_indices: Optional[npt.NDArray[np.int64]] = None,
    param_perf_method: Optional[str] = None,
    param_perf_alpha: Optional[float] = None
) -> None:
    """Save correlation results and significance masks.

    Args:
        out_path: Output directory base
        test_corrs: Test correlations (n_rois,)
        test_corrs_pval: P-values (n_rois,)
        config: Configuration dictionary
        param_reg_gaze_weight: Gaze weight parameters
        param_reg_wordnet: WordNet hierarchy flag
        param_enc_single_alpha: Single alpha flag
        mask_indices: Significant ROI indices (optional)
        param_perf_method: Multiple comparison method (optional)
        param_perf_alpha: Significance alpha (optional)
    """
    seed = config['seed']

    # Construct base path (now includes seed)
    if param_reg_gaze_weight[0] is False:
        base_path = out_path / config['conf_option'] / config['atlas'] / \
                   f"chunk-{config['chunk_option']}" / 'fold-avg' / \
                   f"d-{config['hx_docu']}" / \
                   f"k-{config['srm_option']}" / 'gaze_weight-False' / \
                   f"wn-{param_reg_wordnet}" / f"seed-{seed}" / \
                   f"s_alpha-{param_enc_single_alpha}"
    else:
        base_path = out_path / config['conf_option'] / config['atlas'] / \
                   f"chunk-{config['chunk_option']}" / 'fold-avg' / \
                   f"d-{config['hx_docu']}" / \
                   f"k-{config['srm_option']}" / \
                   f"bias-{param_reg_gaze_weight[0]}_verb-{param_reg_gaze_weight[1]}" / \
                   f"wn-{param_reg_wordnet}" / f"seed-{seed}" / \
                   f"s_alpha-{param_enc_single_alpha}"

    # Create directory
    base_path.mkdir(parents=True, exist_ok=True)

    # Save correlation results
    np.save(str(base_path / 'test_corrs_of_avg'), test_corrs)
    np.save(str(base_path / 'test_corrs_pval_of_avg'), test_corrs_pval)

    # Save mask if provided
    if mask_indices is not None and param_perf_method is not None and param_perf_alpha is not None:
        method_str = 'fdr' if param_perf_method == 'fdr_bh' else 'bf'
        np.save(str(base_path / f'mask_{method_str}-{param_perf_alpha}'), mask_indices)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_brain_surface(
    data: npt.NDArray[np.float64],
    surf_lh: Any,
    surf_rh: Any,
    label_atlas: npt.NDArray[np.int32],
    cmap: str = 'jet',
    cmin: float = 0.0,
    cmax: float = 0.5
) -> None:
    """Visualize scalar data on brain surface (Windows only).

    Args:
        data: ROI-level values to map (n_rois,)
        surf_lh: Left hemisphere surface
        surf_rh: Right hemisphere surface
        label_atlas: Atlas labels (n_vertices,)
        cmap: Colormap name
        cmin: Color range minimum
        cmax: Color range maximum
    """
    if os.name != 'nt':
        return

    # Map ROI data to vertices
    data_full = map_to_labels(
        data,
        label_atlas,
        mask=label_atlas != 0,
        fill=np.nan
    )

    # Plot on surface
    plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=data_full,
        size=(1100, 500),
        zoom=1.3,
        layout_style='grid',
        cmap=cmap,
        color_bar=True,
        color_range=(cmin, cmax)
    )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """Execute performance mask analysis."""

    # Setup
    config = CONFIG
    paths = get_project_paths(config)

    # Load participant data
    print('* Load ID list of preprocessed data')
    participants_file = paths['participants_df_path'] / 'participants_df_deepmreye_inc_byhx.xlsx'
    participants_df = pd.read_excel(str(participants_file), index_col=0, engine='openpyxl')

    # Filter participants
    id_list, participants_df_filt = filter_participants(
        participants_df,
        config['dx'],
        config['fd_thres'],
        config['hx_docu']
    )

    # Print demographics
    print_participant_summary(participants_df_filt)

    # Setup cross-validation
    print('\n* K-fold cross-validation')
    test_fold_list = create_cv_folds(config['chunk_option'])

    # Setup paths
    fmri_path = paths['fmri_path']
    pred_bold_path = paths['pred_bold_path']
    out_path = paths['out_path']

    # Load atlas and surfaces (Windows only)
    if os.name == 'nt':
        print('\n* Load visualization toolbox')
        surf_lh, surf_rh, label_atlas = setup_visualization(paths, config['atlas'])

    # Parameter search loop
    for param_reg_gaze_weight in config['param_search']['gaze_weights']:
        print(f'\nRegressor - gaze weight: bias_weight-{param_reg_gaze_weight[0]}, '
              f'verb_weight-{param_reg_gaze_weight[1]}')

        for param_reg_wordnet in config['param_search']['wordnet_hierarchy']:
            print(f'\tRegressor - WordNet hierarchy: {param_reg_wordnet}')

            for param_enc_single_alpha in config['param_search']['single_alpha']:
                print(f'\t\tEncoding - Single alpha: {param_enc_single_alpha}')

                # Load BOLD data
                print('\t\t* Average actual BOLD and predicted BOLD of test fold')
                actual_bold_mean, pred_bold_mean = load_bold_predictions(
                    id_list,
                    test_fold_list,
                    fmri_path,
                    pred_bold_path,
                    config,
                    param_reg_gaze_weight,
                    param_reg_wordnet,
                    param_enc_single_alpha
                )

                # Calculate correlations
                print('\t\t* Calculate correlation and find significant regions after '
                      'multiple comparisons correction')
                test_corrs, test_corrs_pval = calculate_correlations(
                    actual_bold_mean,
                    pred_bold_mean,
                    alternative='greater'
                )

                # Save correlation results
                save_performance_results(
                    out_path,
                    test_corrs,
                    test_corrs_pval,
                    config,
                    param_reg_gaze_weight,
                    param_reg_wordnet,
                    param_enc_single_alpha
                )

                # Visualize correlations (Windows only)
                if os.name == 'nt':
                    visualize_brain_surface(
                        test_corrs,
                        surf_lh,
                        surf_rh,
                        label_atlas,
                        cmap='jet',
                        cmin=0.0,
                        cmax=0.5
                    )

                # Multiple comparisons correction
                for param_perf_method in config['perf_methods']:
                    print(f'\t\tPerformance - Method: {param_perf_method}')

                    for param_perf_alpha in config['perf_alphas']:
                        print(f'\t\t\tPerformance - Alpha: {param_perf_alpha}')

                        # Apply correction
                        reject, pval_corrected = multipletests(
                            test_corrs_pval,
                            alpha=param_perf_alpha,
                            method=param_perf_method
                        )[:2]

                        n_sig = reject.sum()
                        n_total = len(test_corrs_pval)
                        print(f'\t\t\t({n_sig}/{n_total}) are significant. <- Corrected')

                        # Save mask
                        mask_indices = reject.nonzero()[0]
                        save_performance_results(
                            out_path,
                            test_corrs,
                            test_corrs_pval,
                            config,
                            param_reg_gaze_weight,
                            param_reg_wordnet,
                            param_enc_single_alpha,
                            mask_indices=mask_indices,
                            param_perf_method=param_perf_method,
                            param_perf_alpha=param_perf_alpha
                        )

                        # Visualize mask (Windows only)
                        if os.name == 'nt':
                            mask_float = (reject * 1).astype(np.float64)
                            visualize_brain_surface(
                                mask_float,
                                surf_lh,
                                surf_rh,
                                label_atlas,
                                cmap='Reds',
                                cmin=0.0,
                                cmax=0.5
                            )


if __name__ == '__main__':
    main()
