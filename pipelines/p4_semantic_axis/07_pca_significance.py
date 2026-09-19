"""
PCA significance testing: Compare explained variance between stimulus and semantic spaces.

Refactored from: 16_pca/05_significance_inc_byhx.py

This script performs:
1. Calculate explained variance ratio of individual stimulus matrix (PCA on regressors)
2. Calculate explained variance ratio of individual semantic space (PCA on brain weights)
3. Statistical comparison (paired t-test with Bonferroni correction)
4. Visualization of explained variance across components

Key insight: Tests whether semantic brain representations capture more variance in
particular components compared to the stimulus features themselves.

All functions are defined locally - this is a standalone script with no external dependencies.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from pprint import pprint

# Scientific computing
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import zscore, ttest_rel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, fig_dir  # noqa: E402
# PCA exactly as the (locally modified) brainspace used for the published
# analysis computed it; see lib/brainspace_ext.py.
import brainspace_ext  # noqa: E402

# Neuroimaging
from neuroCombat import neuroCombat
from nilearn.glm import regression

# NLP
from nltk.corpus import wordnet31 as wn
from nltk.corpus.reader.wordnet import WordNetError

# Utilities
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '07_pca_significance',

    # Source data pipelines (99_main refactored versions)
    'source_pipeline_participants': '99_main',
    'source_pipeline_encoding': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_task_encoding': '06_encoding_model',

    # Data parameters
    'atlas': 'mmp',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option': 0,
    'me_option': False,
    'start_ignore_trs': 5,
    'end_ignore_trs': 5,
    'tr_delays_option': [3, 5, 7, 9],

    # Quality control
    'fd_thres': 0.5,
    'hx_docu': 'True',  # Only include ASD subjects with documentation if by-history

    # Analysis parameters (matching 07_pca.py defaults)
    'bias_weight': 0.9, 
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha': True, 
    'weight_across_delays': 'Avg',
    'perf_method': 'fdr', 
    'perf_alpha': 0.01,
    'weight_add_superordinate': True,
    'pca_template': 'Whole', 
    'pca_only_sign': True,
    'pca_iter': 10, 

    # Statistical parameters
    'pval_thres': 0.001,  # Significance threshold for display
    'clip_components': 10,  # Number of components to visualize
    'n_std': 3,  # Number of standard deviations for error bands

    # Seed parameters
    'seed': 0,  # Random seed for encoding model (0-9)

    # File operations
    'save_outputs': True,   # Enable saving outputs
    'overwrite': False,     # If True, overwrite existing files; if False, skip existing
}


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Construct all project paths from CONFIG with platform detection.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of Path objects for all required directories
    """
    # Platform-specific storage roots
    # Paths come from config/paths.yml via lib/project_config.py.  The original
    # working tree hard-coded storage roots here (S:/, Q:/, V:/ on Windows;
    # /Volumes/* on macOS; /MIPL/store* on Linux); those are machine-specific and
    # are not distributed.
    proj_root = PROJECT

    proj = config['project']
    pipe = config['pipeline']
    task = config['task']

    paths = {
        # Base paths
        'raw': proj_root / '0_data' / 'raw',
        'code': proj_root / '1_code' / pipe,
        'tpl': TEMPLATES,
        'fig': fig_dir(__file__),

        # Task-specific output paths
        'pipe': proj_root / '2_pipeline' / pipe,

        # Source data paths (from 99_main refactored pipelines)
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        'enc_results': proj_root / '2_pipeline' /
                      config['source_pipeline_encoding'] /
                      config['source_task_encoding'] / 'out',

        'reg': proj_root / '2_pipeline' /
              config['source_pipeline_participants'] /
              config['source_task_participants'] / 'out',
    }

    # Task proc paths
    paths['proc'] = paths['pipe'] / task
    paths['out'] = paths['proc'] / 'out'
    paths['save'] = paths['proc'] / 'save'
    paths['tmp'] = paths['proc'] / 'tmp'

    # Create directories
    for key in ['out', 'save', 'tmp']:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


# Initialize paths
PATHS = setup_paths(CONFIG)


# =============================================================================
# HELPER FUNCTIONS: FILE I/O
# =============================================================================

def load_weights_with_seed(
    id_list: List[str],
    path: Path,
    seed: int
) -> List[npt.NDArray]:
    """
    Load encoding model weights from results directory.

    Args:
        id_list: List of subject IDs
        path: Results path (already includes seed-{seed} directory)
        seed: Random seed value (0-9), used for display only

    Returns:
        List of weight arrays per subject

    Note:
        Expected structure: path/weight/{sub_id}.npy
        (seed is already part of the path from build_encoding_results_path)
    """
    weight_path = path / 'weight'
    weight_list = []

    for sub_id in tqdm(id_list, desc=f"Loading weights (seed {seed})"):
        file_path = weight_path / f'{sub_id}.npy'
        if file_path.exists():
            weight_list.append(np.load(file_path))
        else:
            print(f"Warning: Missing {file_path}")
            weight_list.append(None)

    return weight_list


def avg_weight(
    weight_list: List[npt.NDArray],
    N_DELAYS: int
) -> List[npt.NDArray]:
    """
    Average weights across TR delays.

    Args:
        weight_list: List of weight arrays per subject (n_features*n_delays, n_rois)
        N_DELAYS: Number of TR delays

    Returns:
        List of averaged weight arrays (n_features, n_rois)
    """
    return [w.reshape(N_DELAYS, -1, w.shape[-1]).mean(0) for w in weight_list]


def max_weight(
    weight_list: List[npt.NDArray],
    N_DELAYS: int
) -> List[npt.NDArray]:
    """
    Select max absolute weights across TR delays.

    Args:
        weight_list: List of weight arrays per subject (n_features*n_delays, n_rois)
        N_DELAYS: Number of TR delays

    Returns:
        List of max weight arrays (n_features, n_rois)
    """
    result = []
    for w in weight_list:
        w_reshaped = w.reshape(N_DELAYS, -1, w.shape[-1])
        max_idx = np.abs(w_reshaped).argmax(0)
        max_weights = w_reshaped[max_idx, np.arange(w_reshaped.shape[1])[:, None],
                                 np.arange(w_reshaped.shape[2])]
        result.append(max_weights)
    return result


# =============================================================================
# CORE ANALYSIS: DATA LOADING
# =============================================================================

def load_participant_data(
    dx: str,
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load and filter participants by quality control criteria.

    Applies 6 filters:
    1. Diagnosis (TD/ASD/all)
    2. By-history documentation (if config['hx_docu'] == 'True')
    3. Preprocessing completion
    4. No remarks during preprocessing
    5. Motion (FD < threshold)
    6. DeepMReye quality rating

    Args:
        dx: Diagnosis filter ('TD', 'ASD', or 'all')
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Tuple of (id_list, participants_df)
    """
    print(f'* Load ID list - {dx}')

    # Load participants DataFrame
    participants_df = pd.read_excel(
        paths['participants_df'] / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    # Filter #1: Diagnosis
    if dx == 'TD':
        dx_filt = (participants_df['DX'] == 'TD').values
    elif dx == 'ASD':
        dx_filt = (participants_df['DX'] == 'ASD').values
    elif dx == 'all':
        dx_filt = np.ones(len(participants_df), dtype=bool)
    else:
        raise ValueError(f"Invalid dx option: {dx}")

    # Filter #2: By-history documentation
    if config['hx_docu'] == 'True':
        for sub_id in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                print(f"  Exclude {sub_id} (by-history without documentation)")
                dx_filt[participants_df.index == sub_id] = False

    # Filter #3: Preprocessing OK
    prep_filt = (participants_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] == 1).values

    # Filter #4: No remarks
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Combine filters 1-4
    id_list_dx_prep_remark = list(
        participants_df[dx_filt & prep_filt & no_remarks_filt].index
    )
    participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

    # Filter #5: Motion (FD threshold)
    fd_filt = (participants_df_dx_prep_remark['Mean_FD_DM'] < config['fd_thres']).values

    # Filter #6: DeepMReye quality
    deepmreye_filt = (participants_df_dx_prep_remark['Rating_deepmreye_movieDM'] == 1).values

    # Final subject list
    id_list = list(
        participants_df_dx_prep_remark[fd_filt & deepmreye_filt].index
    )

    # Print summary
    print(f'Participants information (n={len(id_list)}):')
    print(' - DX')
    print(f"   TD: {len(participants_df.loc[id_list][participants_df.loc[id_list]['DX']=='TD'])}, "
          f"ASD: {len(participants_df.loc[id_list][participants_df.loc[id_list]['DX']=='ASD'])}")
    print(' - Site')
    print(f"   CBIC: {len(participants_df.loc[id_list][participants_df.loc[id_list]['Site']=='CBIC'])}, "
          f"RU: {len(participants_df.loc[id_list][participants_df.loc[id_list]['Site']=='RU'])}")
    print(' - Sex')
    print(f"   Male: {len(participants_df.loc[id_list][participants_df.loc[id_list]['Sex']=='Male'])}, "
          f"Female: {len(participants_df.loc[id_list][participants_df.loc[id_list]['Sex']=='Female'])}")

    id_list = sorted(id_list)

    return id_list, participants_df


def build_encoding_results_path(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Path:
    """
    Build path to encoding results based on configuration.

    Args:
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Path to encoding results directory

    Note:
        Path structure: .../wn-{wordnet}/seed-{seed}/s_alpha-{single_alpha}
    """
    base = (paths['enc_results'] / config['conf_option'] / config['atlas'] /
            f"chunk-{config['chunk_option']}" / 'fold-avg' /
            f"d-{config['hx_docu']}" / f"k-{config['srm_option']}")

    if not config['bias_weight']:  # No gaze weight
        return (base / 'gaze_weight-False' /
                f"wn-{config['reg_wordnet']}" /
                f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}")
    else:  # Gaze weight
        return (base /
                f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f"wn-{config['reg_wordnet']}" /
                f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}")


def load_regressors(
    id_list: List[str],
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load semantic regressors (wordnet or standard).

    Args:
        id_list: List of subject IDs (uses first subject for gaze-weighted)
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Tuple of (reg_dm_df, label_list)
    """
    print('* Load regressors')

    if not config['bias_weight']:  # No gaze weight
        if not config['reg_wordnet']:  # No wordnet hierarchy
            reg_dm_df = pd.read_excel(
                paths['raw'] / 'movie' / 'DM' / 'sync_regressor_DM.xlsx',
                index_col=0,
                sheet_name='all (wordnet)',
                engine='openpyxl'
            )
        else:  # WordNet hierarchy
            reg_dm_df = pd.read_excel(
                paths['raw'] / 'movie' / 'DM' / 'wordnet_sync_regressor_DM.xlsx',
                index_col=0,
                engine='openpyxl'
            )
        label_list = list(reg_dm_df.index)
    else:  # Gaze weight
        if not config['reg_wordnet']:  # No wordnet hierarchy
            reg_dm_df = pd.read_excel(
                paths['reg'] / f"gaze_weighted_regressor_inc_byhx_docu-{config['hx_docu']}" /
                'within' / f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f'{id_list[0]}_regressor_DM.xlsx',
                index_col=0,
                engine='openpyxl'
            )
            label_list = list(pd.read_excel(
                paths['raw'] / 'movie' / 'DM' / 'sync_regressor_DM.xlsx',
                index_col=0,
                sheet_name='all (wordnet)',
                engine='openpyxl'
            ).index)
        else:  # WordNet hierarchy
            reg_dm_df = pd.read_excel(
                paths['reg'] / f"wordnet_gaze_weighted_regressor_inc_byhx_docu-{config['hx_docu']}" /
                'within' / f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f'{id_list[0]}_regressor_DM.xlsx',
                index_col=0,
                engine='openpyxl'
            )
            label_list = list(reg_dm_df.index)

    return reg_dm_df, label_list


# =============================================================================
# CORE ANALYSIS: PREPROCESSING
# =============================================================================

def process_weights_across_delays(
    weight_list: List[npt.NDArray],
    config: Dict[str, Any]
) -> npt.NDArray:
    """
    Average or max weights across TR delays.

    Args:
        weight_list: List of weight arrays per subject
        config: Configuration dictionary

    Returns:
        Processed weight array
    """
    print(f"* Weight across delays: {config['weight_across_delays']}")

    if config['weight_across_delays'] == 'Avg':
        return np.array(avg_weight(weight_list, N_DELAYS=len(config['tr_delays_option'])))
    else:
        return np.array(max_weight(weight_list, N_DELAYS=len(config['tr_delays_option'])))


def harmonize_weights(
    weights: npt.NDArray,
    participants_df: pd.DataFrame,
    id_list: List[str]
) -> npt.NDArray:
    """
    Harmonize weights w.r.t site using neuroCombat.

    Args:
        weights: Weight array (n_subjects, n_features, n_rois)
        participants_df: Participant metadata
        id_list: List of subject IDs

    Returns:
        Harmonized weight array
    """
    print('* Harmonize weights w.r.t site')

    weight_har = np.zeros_like(weights)

    # Harmonize each feature separately
    for feat_idx in tqdm(range(weights.shape[1]), desc="Harmonizing features"):
        # neuroCombat expects (features, samples)
        weight_har[:, feat_idx, :] = neuroCombat(
            weights[:, feat_idx, :].T,
            covars=participants_df.loc[id_list],
            batch_col='Site',
            categorical_cols=['Sex', 'DX'],
            continuous_cols=['Age'],
            verbose=False
        )['data'].T

    return weight_har


def regress_label_frequency(
    weights: npt.NDArray,
    label_list: List[str],
    paths: Dict[str, Path]
) -> List[npt.NDArray]:
    """
    Regress out label frequency from weights.

    Args:
        weights: Weight array (n_subjects, n_features, n_rois)
        label_list: List of label names
        paths: Path dictionary

    Returns:
        List of frequency-regressed weight arrays
    """
    print('* Regress out label frequency')

    # Load frequencies
    freq = pd.read_excel(
        paths['raw'] / 'movie' / 'DM' / 'wordnet_sync_regressor_DM.xlsx',
        index_col=0,
        engine='openpyxl'
    )['Frequency'].loc[np.array(label_list)].values

    # Regress for each subject
    weight_clean_list = []
    for weight in weights:
        design = freq.reshape(-1, 1)
        model = regression.OLSModel(design)
        results = model.fit(weight)
        weight_clean_list.append(results.residuals)

    return weight_clean_list


def load_performance_mask(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> npt.NDArray:
    """
    Load model performance significance mask.

    Args:
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Performance mask array (ROI indices)
    """
    print('* Load performance mask')

    results_path = build_encoding_results_path(config, paths)
    mask_path = results_path / f"mask_{config['perf_method']}-{config['perf_alpha']}.npy"

    perf_mask = np.load(mask_path)

    return perf_mask


def mask_by_performance(
    weight_list: List[npt.NDArray],
    perf_mask: npt.NDArray
) -> List[npt.NDArray]:
    """
    Apply performance mask to weights.

    Args:
        weight_list: List of weight arrays
        perf_mask: Performance mask (ROI indices)

    Returns:
        List of masked weight arrays
    """
    print('* Mask weight matrix')

    return [weight[:, perf_mask] for weight in weight_list]


def add_superordinate_weights(
    weight_list: List[npt.NDArray],
    label_list: List[str]
) -> List[npt.NDArray]:
    """
    Add WordNet superordinate (hypernym) weights to child concepts.

    Args:
        weight_list: List of weight arrays
        label_list: List of WordNet synset labels

    Returns:
        List of weight arrays with superordinate additions
    """
    print('* Add superordinate weights')

    # Fail loudly if the WordNet 3.1 corpus is missing (see 07_pca.py).
    try:
        wn.ensure_loaded()
    except LookupError as exc:
        raise LookupError(
            "The NLTK WordNet 3.1 corpus is required for the superordinate step. "
            "Install it with: python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('wordnet31')\""
        ) from exc

    weight_wordnet_list = []

    for weight in weight_list:
        weight_df = pd.DataFrame(weight, index=label_list)

        for label in label_list:
            try:
                net = wn.synset(label)
                hyper_list = net.hypernym_paths()[0][:-1]

                for hyper in hyper_list:
                    hyper_name = hyper.name()
                    if hyper_name in label_list:
                        weight_df.loc[label] += weight_df.loc[hyper_name]

            except WordNetError:
                # Skip labels that are not WordNet synsets (none of the 85 are)
                continue

        weight_wordnet_list.append(weight_df.values)

    return weight_wordnet_list


def normalize_weights(
    weight_list: List[npt.NDArray]
) -> List[npt.NDArray]:
    """
    Z-score normalize weights.

    Args:
        weight_list: List of weight arrays

    Returns:
        List of z-scored weight arrays
    """
    print('* Normalize weight matrix')

    return [zscore(weight, axis=1).T for weight in weight_list]


# =============================================================================
# CORE ANALYSIS: EXPLAINED VARIANCE
# =============================================================================

def compute_stimulus_explained_variance(
    id_list: List[str],
    label_list: List[str],
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> npt.NDArray:
    """
    Calculate explained variance ratio of individual stimulus matrix.

    Performs PCA on each subject's stimulus regressor matrix and computes
    the explained variance ratio for each component.

    Args:
        id_list: List of subject IDs
        label_list: List of semantic labels
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Array of explained variance ratios (n_subjects, n_components)
    """
    print("* Calculate explained variance of regressor matrix (stimulus matrix)")

    stim_list = []

    for sub_id in tqdm(id_list, desc="Loading stimulus matrices"):
        # Load regressor for this subject
        if not config['bias_weight']:  # No gaze weight
            if not config['reg_wordnet']:  # No wordnet hierarchy
                reg_dm_df = pd.read_excel(
                    paths['raw'] / 'movie' / 'DM' / 'sync_regressor_DM.xlsx',
                    index_col=0,
                    sheet_name='all (wordnet)',
                    engine='openpyxl'
                )
            else:  # Wordnet hierarchy
                reg_dm_df = pd.read_excel(
                    paths['raw'] / 'movie' / 'DM' / 'wordnet_sync_regressor_DM.xlsx',
                    index_col=0,
                    engine='openpyxl'
                )
        else:  # Gaze weight
            if not config['reg_wordnet']:  # No wordnet hierarchy
                reg_dm_df = pd.read_excel(
                    paths['reg'] / f"gaze_weighted_regressor_inc_byhx_docu-{config['hx_docu']}" /
                    'within' / f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                    f'{sub_id}_regressor_DM.xlsx',
                    index_col=0,
                    engine='openpyxl'
                )
            else:  # Wordnet hierarchy
                reg_dm_df = pd.read_excel(
                    paths['reg'] / f"wordnet_gaze_weighted_regressor_inc_byhx_docu-{config['hx_docu']}" /
                    'within' / f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                    f'{sub_id}_regressor_DM.xlsx',
                    index_col=0,
                    engine='openpyxl'
                )

        # Select frames (remove start/end TRs)
        frame_list = list(reg_dm_df.columns)
        frame_list = [frame for frame in frame_list if frame.startswith('f')]
        frame_list = frame_list[config['start_ignore_trs']:-config['end_ignore_trs']]

        stim_list.append(reg_dm_df.loc[label_list][frame_list].values)

    # Perform PCA on stimulus matrices
    n_components = min(stim_list[0].shape)

    # Normalize within labels before PCA
    stim_z_list = [zscore(stim.T, axis=0) for stim in stim_list]
    stim_lambdas = [
        brainspace_ext.pca_embedding(stim_z, n_components, random_state=0)[1]
        for stim_z in stim_z_list
    ]

    # Calculate explained variance ratio
    stim_exp_var_list = [lambdas / np.sum(lambdas) for lambdas in stim_lambdas]
    stim_exp_var_list = np.array(stim_exp_var_list)

    return stim_exp_var_list


def compute_semantic_explained_variance(
    weight_list_td: List[npt.NDArray],
    weight_list_asd: List[npt.NDArray],
    config: Dict[str, Any]
) -> npt.NDArray:
    """
    Calculate explained variance ratio of individual semantic space.

    Performs PCA on semantic weight matrices (brain responses) and computes
    the explained variance ratio for each component.

    Args:
        weight_list_td: List of TD weight arrays (already normalized)
        weight_list_asd: List of ASD weight arrays (already normalized)
        config: Configuration dictionary

    Returns:
        Array of explained variance ratios (n_subjects, n_components)
    """
    print('* Calculate explained variance of individual semantic space')

    # Determine template
    if config['pca_template'] == 'TD':
        weight_z_list = [zscore(weight, axis=0) for weight in weight_list_td]
    else:  # 'Whole'
        weight_z_list = [zscore(weight, axis=0) for weight in weight_list_td + weight_list_asd]

    # Perform PCA
    n_components = min(weight_z_list[0].shape)
    sem_lambdas = [
        brainspace_ext.pca_embedding(weight_z, n_components, random_state=0)[1]
        for weight_z in weight_z_list
    ]

    # Calculate explained variance ratio
    sem_exp_var_list = [lambdas / np.sum(lambdas) for lambdas in sem_lambdas]
    sem_exp_var_list = np.array(sem_exp_var_list)

    return sem_exp_var_list


def compare_explained_variance(
    stim_exp_var: npt.NDArray,
    sem_exp_var: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Statistical comparison of stimulus vs semantic explained variance.

    Performs paired t-test with Bonferroni correction across components.

    Args:
        stim_exp_var: Stimulus explained variance (n_subjects, n_components)
        sem_exp_var: Semantic explained variance (n_subjects, n_components)

    Returns:
        Tuple of (tstat, pval, pval_corrected)
    """
    print('* Compare explained variance of stimulus and semantic space')

    # Paired t-test
    ttest = ttest_rel(stim_exp_var, sem_exp_var)

    # Bonferroni correction
    pval_corrected = multipletests(ttest[1], method='bonferroni')[1]

    return ttest[0], ttest[1], pval_corrected


def plot_explained_variance(
    stim_exp_var: npt.NDArray,
    sem_exp_var: npt.NDArray,
    pval_corrected: npt.NDArray,
    tstat: npt.NDArray,
    config: Dict[str, Any],
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize explained variance comparison.

    Plots mean explained variance for stimulus and semantic spaces across
    components, with error bands and significance markers.

    Args:
        stim_exp_var: Stimulus explained variance (n_subjects, n_components)
        sem_exp_var: Semantic explained variance (n_subjects, n_components)
        pval_corrected: Corrected p-values
        tstat: T-statistics
        config: Configuration dictionary
        save_path: Optional path to save figure
    """
    print('* Plot explained variance of stimulus and semantic space')

    clip = config['clip_components']
    n_std = config['n_std']
    pval_thres = config['pval_thres']

    # Calculate means and standard deviations
    stim_exp_var_mean = np.mean(stim_exp_var[:, :clip], axis=0)
    stim_exp_var_std = np.std(stim_exp_var[:, :clip], axis=0)
    sem_exp_var_mean = np.mean(sem_exp_var[:, :clip], axis=0)
    sem_exp_var_std = np.std(sem_exp_var[:, :clip], axis=0)

    # Print significant components
    print('Significant components (semantic > stimulus):')
    for i, pval in enumerate(pval_corrected[:clip]):
        if pval < pval_thres and tstat[i] < 0:
            print(f'  Component {i+1}: {sem_exp_var_mean[i]*100:.2f}% '
                  f'(p={pval:.2e}, t={tstat[i]:.2f})')

    # Create plot
    fig, ax = plt.subplots(figsize=(9, 10))

    ax.plot(np.arange(1, clip+1), stim_exp_var_mean*100, 'o-',
            label='Stimulus', markersize=25)
    ax.plot(np.arange(1, clip+1), sem_exp_var_mean*100, 'o-',
            label='Semantic space', markersize=25)

    ax.fill_between(np.arange(1, clip+1),
                     stim_exp_var_mean*100 - stim_exp_var_std*n_std*100,
                     stim_exp_var_mean*100 + stim_exp_var_std*n_std*100,
                     alpha=0.2)
    ax.fill_between(np.arange(1, clip+1),
                     sem_exp_var_mean*100 - sem_exp_var_std*n_std*100,
                     sem_exp_var_mean*100 + sem_exp_var_std*n_std*100,
                     alpha=0.2)

    ax.set_xticks(np.arange(1, clip+1))
    ax.tick_params(tick1On=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  Saved to: {save_path}')

    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main analysis workflow."""
    print("=" * 80)
    print("PCA Significance Testing: Stimulus vs Semantic Explained Variance")
    print("=" * 80)
    print()

    # Print configuration
    if os.name == 'nt':
        print("Configuration:")
        pprint({k: v for k, v in CONFIG.items() if k not in ['project', 'pipeline', 'task']},
               sort_dicts=False)
        print()

    # -------------------------------------------------------------------------
    # STEP 1: Load participant data and calculate stimulus explained variance
    # -------------------------------------------------------------------------

    # Determine cohort for template
    if CONFIG['pca_template'] == 'TD':
        template_dx = 'TD'
    else:
        template_dx = 'all'

    id_list, participants_df = load_participant_data(template_dx, CONFIG, PATHS)

    # Load regressors
    reg_dm_df, label_list = load_regressors(id_list, CONFIG, PATHS)

    # Calculate stimulus explained variance
    stim_exp_var = compute_stimulus_explained_variance(id_list, label_list, CONFIG, PATHS)

    # -------------------------------------------------------------------------
    # STEP 2: Load encoding results for both TD and ASD groups
    # -------------------------------------------------------------------------

    id_list_td_asd = []
    weight_list_td_asd = []

    for dx in ['TD', 'ASD']:
        id_list_dx, _ = load_participant_data(dx, CONFIG, PATHS)
        id_list_td_asd.append(id_list_dx)

        # Load weights
        results_path = build_encoding_results_path(CONFIG, PATHS)
        weight_list = load_weights_with_seed(id_list_dx, results_path, CONFIG['seed'])
        weight_list_td_asd.append(weight_list)

    # Combine TD and ASD lists
    id_list_combined = id_list_td_asd[0] + id_list_td_asd[1]
    weight_list_combined = weight_list_td_asd[0] + weight_list_td_asd[1]

    # -------------------------------------------------------------------------
    # STEP 3: Preprocess encoding weights
    # -------------------------------------------------------------------------

    # Process weights across delays
    weight_avg = process_weights_across_delays(weight_list_combined, CONFIG)

    # Harmonize weights
    weight_har = harmonize_weights(weight_avg, participants_df, id_list_combined)

    # Regress out label frequency
    weight_clean_list = regress_label_frequency(weight_har, label_list, PATHS)

    # Load performance mask
    perf_mask = load_performance_mask(CONFIG, PATHS)

    # Mask by performance
    weight_sig_list = mask_by_performance(weight_clean_list, perf_mask)

    # Add superordinate weights
    if CONFIG['weight_add_superordinate']:
        weight_wordnet_list = add_superordinate_weights(weight_sig_list, label_list)
    else:
        weight_wordnet_list = weight_sig_list

    # Normalize weights
    weight_wordnet_z_list = normalize_weights(weight_wordnet_list)

    # Split by diagnosis
    td_ids = participants_df.loc[id_list_combined]['DX'] == 'TD'
    asd_ids = participants_df.loc[id_list_combined]['DX'] == 'ASD'
    weight_list_td = list(np.array(weight_wordnet_z_list)[td_ids.values.nonzero()[0]])
    weight_list_asd = list(np.array(weight_wordnet_z_list)[asd_ids.values.nonzero()[0]])

    # -------------------------------------------------------------------------
    # STEP 4: Calculate semantic explained variance
    # -------------------------------------------------------------------------

    sem_exp_var = compute_semantic_explained_variance(weight_list_td, weight_list_asd, CONFIG)

    # -------------------------------------------------------------------------
    # STEP 5: Statistical comparison
    # -------------------------------------------------------------------------

    tstat, pval, pval_corrected = compare_explained_variance(stim_exp_var, sem_exp_var)

    # -------------------------------------------------------------------------
    # STEP 6: Visualization
    # -------------------------------------------------------------------------

    save_path = None
    if CONFIG['save_outputs']:
        save_path = (PATHS['fig'] / 'pca_significance' /
                    f"explained_variance_comparison_seed-{CONFIG['seed']}.png")

    plot_explained_variance(stim_exp_var, sem_exp_var, pval_corrected, tstat, CONFIG, save_path)

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
