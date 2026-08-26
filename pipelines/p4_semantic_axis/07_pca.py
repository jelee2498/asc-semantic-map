"""
PCA-based group comparison analysis (TD vs ASD) of semantic brain representations.

Refactored from: 16_pca/04_group_diff_inc_byhx.py

This script performs:
1. Subject filtering by quality control metrics
2. Encoding model weight preprocessing (harmonization, frequency regression)
3. Semantic template computation via PCA
4. Individual space alignment using Procrustes
5. Statistical group comparison (TD vs ASD) using simplified GLM workflow
6. FDR-corrected significance testing across all ROIs
7. Brain interpretation (pathways, sections, semantic features)

Key simplification: Removed positive/negative component separation for more direct
statistical testing. Now runs single GLM on all ROIs simultaneously.

All functions are defined locally - this is a standalone script with no external dependencies.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import gc
import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
from pprint import pprint
from math import pi

# Scientific computing
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import zscore, pointbiserialr, pearsonr, t as t_dist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, template, fig_dir  # noqa: E402
from scipy.io import savemat, loadmat

# Neuroimaging
import nibabel as nib
from brainspace.gradient import GradientMaps
from neuroCombat import neuroCombat
from nilearn.glm import regression

# NLP
from nltk.corpus import wordnet31 as wn
from nltk.corpus.reader.wordnet import WordNetError

# Utilities
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Surface rendering (brainspace + VTK).
#
# The original gated these imports on `os.name == 'nt'`, so on Linux every figure
# section was skipped silently: the run exited 0 having written the numeric
# outputs but none of the brain maps.  Import unconditionally and record whether
# it worked, so a missing dependency is reported as one.
try:
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.vtk_interface import wrap_vtk, serial_connect
    from vtk import vtkPolyDataNormals
    HAS_SURFACE_PLOTTING = True
    _SURFACE_PLOTTING_ERROR = None
except ImportError as _exc:  # pragma: no cover
    HAS_SURFACE_PLOTTING = False
    _SURFACE_PLOTTING_ERROR = _exc


def require_surface_plotting() -> None:
    """Raise with an actionable message if the rendering stack is unavailable."""
    if not HAS_SURFACE_PLOTTING:
        raise ImportError(
            "Surface rendering requires brainspace and VTK, which failed to "
            f"import ({_SURFACE_PLOTTING_ERROR}). Install the environment with "
            "`conda env create -f environment/environment.yml`, or pass "
            "--skip-surface-figures to run without the rendering stack."
        )


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '07_pca',

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

    # Analysis parameters
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
    'comp_no': 1,  # PC component number to analyze
    'glm_alpha': 0.025,  # Alpha for group comparison (0.005=strict, 0.025=default, 0.05=liberal)
    'include_expanded_networks': False,  # Include expanded networks in 11h pathway analysis
                                        # If False, excludes: ventral-'Orbital & Polar Frontal',
                                        # dorsal-'Dorsolateral Prefrontal'/'Precuneus & RSC',
                                        # social-'Precuneus'/'Inferior Frontal & Medial Prefrontal'

    # Seed parameters
    'seed': 0,  # Random seed for encoding model (0, 37, 42)

    # File operations
    'save_outputs': True,   # Enable saving outputs (save_all_outputs())
    'overwrite': True,     # If True, overwrite existing files; if False, skip existing
    'make_figures': True,  # Render surface figures (brainspace + VTK).
                           # --skip-surface-figures turns this off; numeric
                           # outputs are unaffected.
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
# CONSTANTS & ATLASES
# =============================================================================

# MMP section list (22 anatomical sections)
MMP_SECTION_LIST = [
    'Primary visual (V1)', 'Early visual', 'Dorsal stream',
    'Ventral stream', 'MT+ complex and neighboring visual',
    'Somatosensory/motor', 'Paracentral lobular and mid cingulate',
    'Premotor', 'Posterior opercular', 'Early auditory', 'Auditory association',
    'Insular and frontal opercular', 'Medial temporal', 'Lateral temporal',
    'TPOJ', 'Superior parietal', 'Inferior parietal',
    'Posterior cingulate', 'Anterior cingulate and medial prefrontal',
    'Orbital and polar frontal', 'Inferior frontal', 'Dorsolateral prefrontal'
]

# Yeo 7 network annotations
ANNOT_DICT_YEO7 = {
    0: 'medial_wall',
    1: 'visual',
    2: 'somat',
    3: 'dorsal_att',
    4: 'ventral_att',
    5: 'limbic',
    6: 'frontoparietal',
    7: 'default'
}

# Yeo 7 network names (for plotting)
YEO7_LIST = ['Visual', 'Somator', 'Dorsal Att.', 'Ventral Att.',
             'Limbic', 'Frontoparietal', 'Default']

# Yeo 7 colors (for plotting)
YEO7_COLOR_DICT = {
    'Visual': 'indigo',
    'Somator': 'royalblue',
    'Dorsal Att.': 'green',
    'Ventral Att.': 'mediumorchid',
    'Limbic': 'beige',
    'Frontoparietal': 'orange',
    'Default': 'indianred'
}

# Yeo 17 network names
YEO17_LIST = [
    'Visual A', 'Visual B', 'Somator A', 'Somator B',
    'Dorsal Att. A', 'Dorsal Att. B', 'Ventral Att. A', 'Ventral Att. B',
    'Limbic A', 'Limbic B', 'Control C', 'Control A', 'Control B',
    'Temporal Parietal', 'Default C', 'Default A', 'Default B'
]

# Yeo 17 RGB colors
YEO17_RGB_DICT = {
    'Visual A': (120/255, 18/255, 134/255),
    'Visual B': (255/255, 0/255, 0/255),
    'Somator A': (70/255, 130/255, 180/255),
    'Somator B': (42/255, 204/255, 164/255),
    'Dorsal Att. A': (74/255, 155/255, 60/255),
    'Dorsal Att. B': (0/255, 118/255, 14/255),
    'Ventral Att. A': (196/255, 58/255, 250/255),
    'Ventral Att. B': (255/255, 152/255, 213/255),
    'Limbic A': (220/255, 248/255, 164/255),
    'Limbic B': (122/255, 135/255, 50/255),
    'Control C': (119/255, 140/255, 176/255),
    'Control A': (230/255, 148/255, 34/255),
    'Control B': (135/255, 50/255, 74/255),
    'Temporal Parietal': (12/255, 48/255, 255/255),
    'Default C': (0/255, 0/255, 130/255),
    'Default A': (255/255, 255/255, 0/255),
    'Default B': (205/255, 62/255, 78/255)
}

# Evan Gordon 17 network names
EG17_LIST = [
    'Default', 'LatVis', 'FrontPar', 'MedVis', 'DorsAttn', 'Premotor', 'Language',
    'Salience', 'CingOperc', 'HandSM', 'FaceSM', 'Auditory', 'AntMTL', 'PostMTL',
    'ParMemory', 'Context', 'FootSM'
]

# EG17 sorted order (for visualization)
EG17_LIST_SORTED = [
    'Premotor', 'FootSM', 'HandSM', 'FaceSM', 'Auditory', 'FrontPar', 'CingOperc',
    'Salience', 'DorsAttn', 'AntMTL', 'PostMTL', 'MedVis', 'LatVis', 'ParMemory',
    'Context', 'Language', 'Default'
]


def load_atlases(paths: Dict[str, Path]) -> Dict[str, npt.NDArray]:
    """
    Load MMP and Yeo atlases from template directory.

    Args:
        paths: Path dictionary

    Returns:
        Dictionary containing atlas arrays and metadata
    """
    mmp_folder = paths['tpl'] / 'MMP'

    # Load MMP atlas (32k resolution)
    mmp_path = mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    mmp_atlas_nonmed = nib.load(str(mmp_path)).get_fdata()[0].astype(np.int32)

    # Load medial wall mask
    med_path = mmp_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
    med_32k = nib.load(str(med_path)).get_fdata()[0].astype(np.int32).nonzero()[0]
    nonmed_32k_ids = np.array(list(set(range(64984)) - set(med_32k)))

    # Create full MMP atlas (with medial wall as zeros)
    mmp_atlas = np.zeros(64984, dtype=np.int32)
    mmp_atlas[nonmed_32k_ids] = mmp_atlas_nonmed

    # Load MMP section atlas
    section_path = mmp_folder / 'ResultsRegions_ROI.dlabel.nii'
    section_atlas_nonmed = nib.load(str(section_path)).get_fdata()[0].astype(np.int32)
    section_atlas = np.zeros(64984, dtype=np.int32)
    section_atlas[nonmed_32k_ids] = section_atlas_nonmed

    # Load Yeo7 atlas (left hemisphere, then symmetric)
    yeo7_path = paths['tpl'] / 'Yeo_JNeurophysiol11_FreeSurfer' / '32k_fs_LR' / 'lh.Yeo2011_7Networks_order.label.gii'
    yeo7_l = nib.load(str(yeo7_path)).darrays[0].data
    yeo7_atlas = np.r_[yeo7_l, yeo7_l]  # Symmetric (duplicate left to right)

    # Load Yeo17 atlas
    yeo17_path = paths['tpl'] / 'Yeo_JNeurophysiol11_FreeSurfer' / '32k_fs_LR' / 'lh.Yeo2011_17Networks_order.label.gii'
    yeo17_l = nib.load(str(yeo17_path)).darrays[0].data
    yeo17_atlas = np.r_[yeo17_l, yeo17_l]  # Symmetric

    # Load 10k brain mask (non-medial wall vertices)
    non_med_ids_l = nib.load(str(paths['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                 'tpl-fsLR_den-10k_hemi-L_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    non_med_ids_r = nib.load(str(paths['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                 'tpl-fsLR_den-10k_hemi-R_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    brain_mask_10k = np.r_[non_med_ids_l, non_med_ids_r].nonzero()[0]

    # Load 32k brain mask
    non_med_ids_l_32k = nib.load(str(paths['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                     'tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    non_med_ids_r_32k = nib.load(str(paths['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                     'tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    brain_mask_32k = np.r_[non_med_ids_l_32k, non_med_ids_r_32k].nonzero()[0]

    # Load EG17 atlas (Gordon 17 networks) - supplies the network profiling in
    # Fig. 2h.  The original swallowed any failure into `eg_atlas = None`, which
    # silently dropped that panel while still exiting 0; a missing atlas is a
    # configuration error and is raised as one.
    eg_path = template('eg17')
    eg_atlas_mat = loadmat(str(eg_path))
    eg_atlas = np.r_[
        eg_atlas_mat['lh_labels'].squeeze(),
        eg_atlas_mat['rh_labels'].squeeze()
    ].astype(np.int32)

    return {
        'mmp': mmp_atlas,
        'section': section_atlas,
        'yeo7': yeo7_atlas,
        'yeo17': yeo17_atlas,
        'eg': eg_atlas,
        'nonmed_32k_ids': nonmed_32k_ids,
        'brain_mask_10k': brain_mask_10k,
        'brain_mask_32k': brain_mask_32k,
    }


# Load all atlases
ATLASES = load_atlases(PATHS)


# =============================================================================
# HELPER FUNCTIONS: FILE I/O
# =============================================================================

def safe_save(
    data: Union[npt.NDArray, Dict, Any],
    filepath: Path,
    save_func: str = 'numpy',
    overwrite: bool = False,
    verbose: bool = True
) -> bool:
    """
    Conditionally save file with overwrite protection.

    Args:
        data: Data to save
        filepath: Destination path (Path object)
        save_func: Save function type ('numpy', 'matlab', 'cifti')
        overwrite: Allow overwriting existing files
        verbose: Print status messages

    Returns:
        True if saved, False if skipped
    """
    if filepath.exists() and not overwrite:
        if verbose:
            print(f"[skip] Skipping (exists): {filepath.name}")
        return False

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        if save_func == 'numpy':
            np.save(filepath, data)
        elif save_func == 'matlab':
            savemat(str(filepath), data)
        elif save_func == 'cifti':
            # data is already a CIFTI object
            data.to_filename(str(filepath))
        else:
            raise ValueError(f"Unknown save_func: {save_func}")

        if verbose:
            print(f"[ok] Saved: {filepath.name}")
        return True

    except Exception as e:
        print(f"[error] Error saving {filepath.name}: {e}")
        return False


def load_files(
    keys: List[str],
    id_list: List[str],
    path: Path
) -> Dict[str, List[npt.NDArray]]:
    """
    Load saved .npy files for each subject.

    Args:
        keys: List of data keys to load (e.g., ['weight', 'test_corrs'])
        id_list: List of subject IDs
        path: Base path containing data folders (each key has its own subfolder)

    Returns:
        Dictionary with keys from input, values are lists of arrays per subject

    Note:
        Expected structure: path/{key}/{sub_id}.npy
        e.g., path/weight/sub-NDARINV12345678.npy
    """
    results = {key: [] for key in keys}

    for i, sub_id in enumerate(tqdm(id_list, desc="Loading files")):
        for key in keys:
            file_path = path / key / f'{sub_id}.npy'
            if file_path.exists():
                # Explicitly open, load, close, and delete file handle
                f = None
                data = None
                try:
                    f = open(file_path, 'rb')
                    data = np.load(f)
                    results[key].append(data.copy() if hasattr(data, 'copy') else data)
                finally:
                    if f is not None:
                        f.close()
                    del f, data
            else:
                print(f"Warning: Missing {file_path}")
                results[key].append(None)

        # Garbage collection after every subject to free file handles
        gc.collect()

    return results


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

    for i, sub_id in enumerate(tqdm(id_list, desc=f"Loading weights (seed {seed})")):
        file_path = weight_path / f'{sub_id}.npy'
        if file_path.exists():
            # Explicitly open, load, close, and delete file handle
            f = None
            data = None
            try:
                f = open(file_path, 'rb')
                data = np.load(f)
                weight_list.append(data.copy() if hasattr(data, 'copy') else data)
            finally:
                if f is not None:
                    f.close()
                del f, data
        else:
            print(f"Warning: Missing {file_path}")
            weight_list.append(None)

        # Garbage collection after every subject to free file handles
        gc.collect()

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
        # Select weight at max index for each feature-roi pair
        max_weights = w_reshaped[max_idx, np.arange(w_reshaped.shape[1])[:, None],
                                 np.arange(w_reshaped.shape[2])]
        result.append(max_weights)
    return result


# =============================================================================
# HELPER FUNCTIONS: VISUALIZATION
# =============================================================================

def value_to_rgb(
    value: float,
    cmap_name: str,
    vmin: float,
    vmax: float
) -> Tuple[float, float, float]:
    """
    Convert a value to RGB color using matplotlib colormap.

    Args:
        value: Value to convert to color
        cmap_name: Name of the colormap
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization

    Returns:
        Tuple of RGB color values (r, g, b)
    """
    from matplotlib import colormaps

    # Create colormap
    cmap = colormaps.get_cmap(cmap_name)

    # Normalize value between 0 and 1
    norm_value = (value - vmin) / (vmax - vmin)
    norm_value = np.clip(norm_value, 0, 1)  # ensure value is between 0 and 1

    # Get RGB values
    rgb = cmap(norm_value)[:3]  # exclude alpha channel
    return rgb


def plot_radar(
    df: pd.DataFrame,
    column: str,
    value_max: Optional[float] = None,
    value_min: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot radar chart for DataFrame values.

    Args:
        df: DataFrame with categories as index
        column: Column name to plot
        value_max: Maximum value for plot range
        value_min: Minimum value for plot range
        title: Plot title
    """
    # Number of variables
    categories = list(df.index)
    N = len(categories)

    # Repeat first value to close the circular graph
    values = df[column].values.flatten().tolist()
    values += values[:1]

    # Calculate angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialize spider plot
    ax = plt.subplot(111, polar=True)
    fig = plt.gcf()
    fig.set_size_inches(15, 15)

    if title is not None:
        plt.title(title, size=20, y=1.1)

    # Draw axes and labels
    plt.xticks(angles[:-1], categories, color='grey', size=15)

    # Draw ylabels
    ax.set_rlabel_position(0)
    if value_max is None:
        value_max = np.ceil(np.max(np.abs(values))).astype(int)
    if value_min is None:
        value_min = -value_max

    yticks = np.linspace(value_min, value_max, 5).round(2)
    yticks_label = [f'{ytick}' for ytick in yticks]
    plt.yticks(yticks, yticks_label, color="grey", size=10)
    plt.ylim(value_min, value_max)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.show()


def plot_radar_sep(
    df: pd.DataFrame,
    column: str,
    value_max: Optional[float] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot radar chart with positive and negative values separately.

    Args:
        df: DataFrame with categories as index
        column: Column name to plot
        value_max: Maximum value for plot range
        title: Plot title
    """
    # Number of variables
    categories = list(df.index)
    N = len(categories)

    # Separate positive and negative values
    values_positive = df[column].values.copy()
    values_positive[values_positive < 0] = 0
    values_positive = values_positive.flatten().tolist()
    values_positive += values_positive[:1]

    values_negative = df[column].values.copy()
    values_negative[values_negative > 0] = 0
    values_negative = values_negative.flatten().tolist()
    values_negative += values_negative[:1]
    values_negative = np.abs(values_negative).tolist()

    # Calculate angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialize spider plot
    ax = plt.subplot(111, polar=True)
    fig = plt.gcf()
    fig.set_size_inches(15, 15)

    if title is not None:
        plt.title(title, size=20, y=1.1)

    # Draw axes and labels
    plt.xticks(angles[:-1], categories, color='grey', size=15)

    # Draw ylabels
    ax.set_rlabel_position(0)
    if value_max is None:
        value_max = np.ceil(np.max(np.abs(df[column]))).astype(int)

    yticks = np.linspace(0, value_max, 5).round(2)
    yticks_label = [f'{ytick}' for ytick in yticks]
    plt.yticks(yticks, yticks_label, color="grey", size=10)
    plt.ylim(0, value_max)

    # Plot data
    ax.plot(angles, values_positive, linewidth=1, linestyle='solid',
            label='Positive', color='r')
    ax.plot(angles, values_negative, linewidth=1, linestyle='solid',
            label='Negative', color='b')

    # Fill areas
    ax.fill(angles, values_positive, 'r', alpha=0.1)
    ax.fill(angles, values_negative, 'b', alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()


def roiarray_2_32knii(
    roiarray: npt.NDArray,
    mask: npt.NDArray,
    atlas: str,
    output_folder: str,
    output_file: str,
    fig_path: Optional[Path] = None,
    comp_no: Optional[int] = None
) -> None:
    """
    Convert ROI array to 32k CIFTI (.dscalar.nii) file.

    Args:
        roiarray: ROI array data
        mask: Model performance mask (indices)
        atlas: Parcellation atlas ('mmp', 'sch-400', 'sch-800')
        output_folder: Output folder name
        output_file: Output file name (must end with .dscalar.nii)
        fig_path: Figure output path (if None, uses MMP folder)
        comp_no: Component number (unused, for compatibility)
    """
    # Check output file extension
    assert output_file.endswith('.dscalar.nii'), \
        'output_file should end with .dscalar.nii'

    # Load brain mask (32k)
    brain_mask = ATLASES['brain_mask_32k']

    # Load atlas
    mmp_folder = PATHS['tpl'] / 'MMP'

    if atlas == 'mmp':
        atlas_nonmed = nib.load(str(mmp_folder /
            'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
            ).get_fdata()[0].astype(np.int32)
        med_32k = nib.load(str(mmp_folder /
            'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii')
            ).get_fdata()[0].astype(np.int32).nonzero()[0]
        nonmed_32k_ids = np.array(list(set(range(64984)) - set(med_32k)))
        label_atlas = np.zeros(64984, dtype=np.int32)
        label_atlas[nonmed_32k_ids] = atlas_nonmed
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    # Map ROI-based data to 32k array
    if len(mask) == len(roiarray):
        data_temp = np.empty(label_atlas.max())
        data_temp[:] = np.nan
        data_temp[mask] = roiarray
        data_32karray = map_to_labels(data_temp, label_atlas,
                                      mask=label_atlas!=0, fill=np.nan)
    else:  # Skip masking
        data_32karray = map_to_labels(roiarray, label_atlas,
                                      mask=label_atlas!=0, fill=np.nan)

    # Transform 32k array to CIFTI
    dummy_nii_path = mmp_folder / 'Q1-Q6_RelatedParcellation210.sulc_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii'
    dummy_nii = nib.load(str(dummy_nii_path))

    data_32knii = nib.Cifti2Image(
        data_32karray[brain_mask][np.newaxis, :],
        header=dummy_nii.header,
        nifti_header=dummy_nii.nifti_header
    )

    # Save file
    if fig_path is None:
        output_dir = mmp_folder / output_folder
    else:
        output_dir = fig_path / output_folder

    output_dir.mkdir(parents=True, exist_ok=True)
    data_32knii.to_filename(str(output_dir / output_file))


def get_single_color_func_legacy(color: str):
    """
    Create a color function which returns a single RGB color.

    Legacy version for wordcloud compatibility.

    Args:
        color: Color string (e.g., 'deepskyblue', '#00b4d2')

    Returns:
        Function that returns RGB color string
    """
    from PIL import ImageColor

    old_r, old_g, old_b = ImageColor.getrgb(color)

    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        """Return fixed RGB color."""
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(old_r, old_g, old_b)

    return single_color_func


class GroupedColorFunc(object):
    """
    Color function for wordcloud that assigns colors based on word groups.

    Maps specific words to specific colors for wordcloud visualization.

    Parameters:
        color_to_words: Dictionary mapping colors to lists of words
        default_color: Color for words not in any group
    """

    def __init__(self, color_to_words: Dict[str, List[str]], default_color: str):
        self.color_func_to_words = [
            (get_single_color_func_legacy(color), set(words))
            for (color, words) in color_to_words.items()
        ]
        self.default_color_func = get_single_color_func_legacy(default_color)

    def get_color_func(self, word: str):
        """Returns color function associated with the word."""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words
            )
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word: str, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def visualize_beta_wordcloud(
    roi: str,
    weight_mean: List[npt.NDArray],
    label_list: List[str],
    perf_mask: npt.NDArray,
    paths: Dict[str, Path],
    percentile: int = 90,
    n_top_words: int = 10,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize encoding model beta weights as wordcloud for a specific ROI.

    Words are colored by sign (red=positive, blue=negative) and sized by weight magnitude.

    Args:
        roi: ROI name (e.g., 'R_PHA1_ROI')
        weight_mean: Mean weights across subjects (list of arrays per masked ROI)
        label_list: List of semantic labels
        perf_mask: Model performance mask
        paths: Path dictionary
        percentile: Percentile for weight normalization (default: 90)
        n_top_words: Number of top/bottom words to show (default: 10)
        save_path: Optional path to save figure
    """
    try:
        from wordcloud import WordCloud
        from matplotlib import colormaps
    except ImportError:
        print("  WordCloud not available - skipping visualization")
        return

    print(f'* Visualize beta values using wordcloud for {roi}')

    # Load MMP info
    mmp_info_path = paths['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
    mmp_info_df = pd.read_excel(str(mmp_info_path), engine='openpyxl')
    mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]

    # Get ROI index
    roi_idx = mmp_info_df[mmp_info_df['ROI'] == roi].index[0]

    # Check if ROI is in model performance mask
    if roi_idx not in perf_mask:
        print(f'  Warning: {roi} is not in the model performance mask - skipping')
        return

    roi_idx_in_perf_mask = np.where(perf_mask == roi_idx)[0][0]

    # Extract weights for this ROI
    weight_mean_roi = np.array(weight_mean)[roi_idx_in_perf_mask]
    weight_mean_roi_df = pd.DataFrame(weight_mean_roi, index=label_list, columns=['weight'])

    # Normalize weights for color mapping
    weight_mean_roi_df['weight_abs'] = weight_mean_roi_df['weight'].abs()
    weight_mean_roi_abs_percentile = np.percentile(weight_mean_roi_df['weight_abs'], percentile)
    weight_mean_roi_max_percentile = weight_mean_roi_abs_percentile
    weight_mean_roi_min_percentile = -weight_mean_roi_abs_percentile
    weight_mean_roi_df['weight_norm'] = (
        (weight_mean_roi_df['weight'] - weight_mean_roi_min_percentile) /
        (weight_mean_roi_max_percentile - weight_mean_roi_min_percentile)
    )

    print(f'  Weight range: {weight_mean_roi_min_percentile:.3f} to {weight_mean_roi_max_percentile:.3f}')

    # Convert weights to color strings using coolwarm colormap
    cmap = colormaps.get_cmap('coolwarm')
    colors_roi = (cmap(weight_mean_roi_df['weight_norm']) * 255).astype(np.uint8)
    color_strings_roi = [
        "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        for color in colors_roi
    ]

    # Create color-to-words mapping
    color_to_words_roi = {key: [] for key in color_strings_roi}
    for i in range(len(weight_mean_roi_df)):
        color_to_words_roi[color_strings_roi[i]].append(weight_mean_roi_df.index[i])

    # Circle shape mask
    x, y = np.ogrid[:1500, :1500]
    mask = (x - 750) ** 2 + (y - 750) ** 2 > 750 ** 2
    mask = 255 * mask.astype(np.uint8)

    # Create wordcloud
    wc = WordCloud(
        background_color='white',
        width=1500,
        height=1500,
        max_words=100,
        prefer_horizontal=1,
        mask=mask,
        min_font_size=12,
        max_font_size=300
    )

    # Z-score weights for sizing
    weight_mean_roi_df['weight_z_abs'] = zscore(weight_mean_roi_df['weight']).abs()
    weight_mean_roi_df['weight_z_abs'] = np.clip(
        weight_mean_roi_df['weight_z_abs'],
        0,
        np.percentile(weight_mean_roi_df['weight_z_abs'], 100)
    )

    # Select top and bottom words
    top_words = weight_mean_roi_df.sort_values(by='weight', ascending=False).head(n_top_words).index
    bottom_words = weight_mean_roi_df.sort_values(by='weight', ascending=True).head(n_top_words).index
    weight_mean_roi_df_selected = pd.concat([
        weight_mean_roi_df.loc[top_words],
        weight_mean_roi_df.loc[bottom_words]
    ])

    # Generate wordcloud
    wc.generate_from_frequencies(weight_mean_roi_df_selected['weight_z_abs'])

    # Apply grouped coloring
    default_color = 'grey'
    grouped_color_func = GroupedColorFunc(color_to_words_roi, default_color)
    wc.recolor(color_func=grouped_color_func)

    # Plot
    plt.figure(figsize=(16, 16))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f'Beta weights for {roi}', size=20)
    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved to: {save_path}')

    plt.show()
    plt.close()


# =============================================================================
# HELPER FUNCTIONS: ATLAS OPERATIONS
# =============================================================================

def mmp_2_section(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    section_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation (360 ROIs) to MMP section (22 sections).

    Args:
        array_mmp: Array in MMP parcellation (can be 1D or 2D)
        mmp_atlas: MMP atlas labels (64984,), values 1-360
        section_atlas: MMP section atlas labels (64984,), values 1-22
        model_perf_mask: Boolean or index mask for significant ROIs
        sum: If True, sum values within section; if False, average

    Returns:
        Array in section space (22,) or (22, n_features)
    """
    # Ensure 2D
    if len(array_mmp.shape) == 1:
        array_mmp = array_mmp.reshape(-1, 1)

    num_features = array_mmp.shape[1]

    # Find mapping between MMP and section
    mapping_mmp_2_section = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        section_atlas_mmp_ids = section_atlas[mmp_ids]
        section_atlas_mmp_ids_unique, section_atlas_mmp_ids_count = \
            np.unique(section_atlas_mmp_ids, return_counts=True)
        mapping_mmp_2_section[mmp_no] = \
            section_atlas_mmp_ids_unique[np.argmax(section_atlas_mmp_ids_count)]

    # Apply mask if provided
    if model_perf_mask is None:
        array_mmp_full = array_mmp
    else:
        array_mmp_full = np.zeros((360, num_features))
        array_mmp_full[model_perf_mask, :] = array_mmp

    # Convert to sections
    array_mmp_section = np.zeros((22, num_features))
    for section_no in range(22):
        if model_perf_mask is None:
            mmp_atlas_section_ids = np.where(
                mapping_mmp_2_section == section_no + 1)[0]
        else:
            mmp_atlas_section_ids = np.array(sorted(list(
                set(np.where(mapping_mmp_2_section == section_no + 1)[0])
                .intersection(set(model_perf_mask))
            )))

        if mmp_atlas_section_ids.size != 0:
            if not sum:
                array_mmp_section[section_no, :] = \
                    array_mmp_full[mmp_atlas_section_ids, :].mean(axis=0)
            else:
                array_mmp_section[section_no, :] = \
                    array_mmp_full[mmp_atlas_section_ids, :].sum(axis=0)

    array_mmp_section = np.nan_to_num(array_mmp_section)

    return array_mmp_section


def mmp_2_yeo17(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    yeo17_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation to Yeo 17 network.

    Args:
        array_mmp: Array in MMP parcellation (360,)
        mmp_atlas: MMP atlas
        yeo17_atlas: Yeo 17 network atlas
        model_perf_mask: Mask of significant ROIs
        sum: If True, sum; if False, average

    Returns:
        Array in Yeo 17 space (17,)
    """
    # Find mapping between MMP and Yeo 17
    mapping_mmp_2_yeo17 = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        yeo17_atlas_mmp_ids = yeo17_atlas[mmp_ids]
        yeo17_atlas_mmp_ids_unique, yeo17_atlas_mmp_ids_count = \
            np.unique(yeo17_atlas_mmp_ids, return_counts=True)
        mapping_mmp_2_yeo17[mmp_no] = \
            yeo17_atlas_mmp_ids_unique[np.argmax(yeo17_atlas_mmp_ids_count)]

    # Apply mask
    if model_perf_mask is None:
        array_mmp_full = array_mmp
    else:
        array_mmp_full = np.zeros(360)
        array_mmp_full[model_perf_mask] = array_mmp

    # Convert to Yeo 17
    array_mmp_yeo17 = np.zeros(17)
    for yeo17_no in range(17):
        mmp_atlas_yeo17_ids = np.where(mapping_mmp_2_yeo17 == yeo17_no + 1)[0]
        if not sum:
            array_mmp_yeo17[yeo17_no] = array_mmp_full[mmp_atlas_yeo17_ids].mean(0)
        else:
            array_mmp_yeo17[yeo17_no] = array_mmp_full[mmp_atlas_yeo17_ids].sum(0)

    array_mmp_yeo17 = np.nan_to_num(array_mmp_yeo17)

    return array_mmp_yeo17


def mmp_2_eg(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    eg_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation to Evan Gordon 17 network.

    Args:
        array_mmp: Array in MMP parcellation (can be 1D or 2D)
        mmp_atlas: MMP atlas labels (64984,), values 1-360
        eg_atlas: Evan Gordon 17 network atlas (64984,), values 1-17
        model_perf_mask: Boolean or index mask for significant ROIs
        sum: If True, sum values within network; if False, average

    Returns:
        Array in EG17 space (17,) or (17, n_features)
    """
    # Ensure 2D
    if len(array_mmp.shape) == 1:
        array_mmp = array_mmp.reshape(-1, 1)

    num_features = array_mmp.shape[1]

    # Find mapping between MMP and EG17
    mapping_mmp_2_eg17 = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        eg_atlas_mmp_ids = eg_atlas[mmp_ids]
        eg_atlas_mmp_ids_unique, eg_atlas_mmp_ids_count = \
            np.unique(eg_atlas_mmp_ids, return_counts=True)
        mapping_mmp_2_eg17[mmp_no] = \
            eg_atlas_mmp_ids_unique[np.argmax(eg_atlas_mmp_ids_count)]

    # Apply mask
    if model_perf_mask is None:
        array_mmp_full = array_mmp
    else:
        array_mmp_full = np.zeros((360, num_features))
        array_mmp_full[model_perf_mask, :] = array_mmp

    # Convert to EG17
    array_mmp_eg = np.zeros((17, num_features))
    for eg_no in range(17):
        if model_perf_mask is None:
            mmp_atlas_eg_ids = np.where(mapping_mmp_2_eg17 == eg_no + 1)[0]
        else:
            mmp_atlas_eg_ids = np.array(sorted(list(
                set(np.where(mapping_mmp_2_eg17 == eg_no + 1)[0])
                .intersection(set(model_perf_mask))
            )))

        if mmp_atlas_eg_ids.size != 0:
            if not sum:
                array_mmp_eg[eg_no, :] = \
                    array_mmp_full[mmp_atlas_eg_ids, :].mean(axis=0)
            else:
                array_mmp_eg[eg_no, :] = \
                    array_mmp_full[mmp_atlas_eg_ids, :].sum(axis=0)

    array_mmp_eg = np.nan_to_num(array_mmp_eg)

    return array_mmp_eg


# =============================================================================
# HELPER FUNCTIONS: WORDNET SEMANTIC GRAPH
# =============================================================================

# Note: networkx is required for WordNet graph operations
import networkx as nx


def node_set(LABEL_LIST: List[str]) -> nx.Graph:
    """
    Create WordNet graph node set from label list.

    Args:
        LABEL_LIST: List of WordNet synset names

    Returns:
        NetworkX graph with nodes
    """
    G = nx.Graph()
    for LABEL in LABEL_LIST:
        try:
            # Check if hypernym paths exist
            hyper_check = wn.synset(LABEL).hypernym_paths()[0][:-1][0]
            G.add_node(LABEL)
        except IndexError:
            continue
    return G


def add_edge(
    G: nx.Graph,
    NODE: str,
    HYPER_NODE: str,
    HYPERNODE_LIST: List,
    DEPTH: int
) -> None:
    """
    Recursively add edges between nodes and their hypernyms.

    Args:
        G: NetworkX graph
        NODE: Current node
        HYPER_NODE: Hypernym node
        HYPERNODE_LIST: List of hypernym synsets
        DEPTH: Recursion depth
    """
    DEPTH += 1
    NODE_LIST = list(G.nodes)
    HYPERNODE_LIST = HYPERNODE_LIST[:-1]

    try:
        if NODE in NODE_LIST and HYPER_NODE in NODE_LIST:
            G.add_edge(NODE, HYPER_NODE, length=5)
            NODE = HYPER_NODE
            HYPER_NODE = HYPERNODE_LIST[-1].name()
            add_edge(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
        else:
            if NODE in NODE_LIST and HYPER_NODE not in NODE_LIST:
                HYPER_NODE = HYPERNODE_LIST[-1].name()
                add_edge(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
            else:
                NODE = HYPER_NODE
                HYPER_NODE = HYPERNODE_LIST[-1].name()
                add_edge(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
    except IndexError:
        pass


def hyper_edge(G: nx.Graph, NODE: str) -> None:
    """
    Add hypernym edges for a node.

    Args:
        G: NetworkX graph
        NODE: Node to add hypernym edges for
    """
    try:
        HYPERNODE_LIST = wn.synset(NODE).hypernym_paths()[0][:-1]
        HYPER_NODE = HYPERNODE_LIST[-1].name()
        add_edge(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=0)
    except (WordNetError, IndexError):
        pass  # Silently skip nodes without hypernyms


def node_set_with_entity_action(LABEL_LIST: List[str]) -> nx.Graph:
    """
    Create WordNet graph with ENTITY/ACTION root nodes.

    Args:
        LABEL_LIST: List of WordNet synset names

    Returns:
        NetworkX graph with nodes including verbs
    """
    G = nx.Graph()
    for LABEL in LABEL_LIST:
        try:
            hyper_check = wn.synset(LABEL).hypernym_paths()[0][:-1][0]
            G.add_node(LABEL)
        except IndexError:
            if '.v.' in LABEL:  # Include verbs
                G.add_node(LABEL)
        except ValueError:  # 'ENTITY' or 'ACTION'
            G.add_node(LABEL)
    return G


def add_edge_with_entity_action(
    G: nx.Graph,
    NODE: str,
    HYPER_NODE: str,
    HYPERNODE_LIST: List,
    DEPTH: int
) -> None:
    """
    Recursively add edges with ENTITY/ACTION support.

    Args:
        G: NetworkX graph
        NODE: Current node
        HYPER_NODE: Hypernym node
        HYPERNODE_LIST: List of hypernym synsets or strings
        DEPTH: Recursion depth
    """
    DEPTH += 1
    NODE_LIST = list(G.nodes)
    HYPERNODE_LIST = HYPERNODE_LIST[:-1]

    try:
        if NODE in NODE_LIST and HYPER_NODE in NODE_LIST:
            G.add_edge(NODE, HYPER_NODE, length=5)
            NODE = HYPER_NODE
            try:
                HYPER_NODE = HYPERNODE_LIST[-1].name()
            except AttributeError:  # 'ENTITY' or 'ACTION'
                HYPER_NODE = HYPERNODE_LIST[-1]
            add_edge_with_entity_action(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
        else:
            if NODE in NODE_LIST and HYPER_NODE not in NODE_LIST:
                try:
                    HYPER_NODE = HYPERNODE_LIST[-1].name()
                except AttributeError:
                    HYPER_NODE = HYPERNODE_LIST[-1]
                add_edge_with_entity_action(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
            else:
                NODE = HYPER_NODE
                try:
                    HYPER_NODE = HYPERNODE_LIST[-1].name()
                except AttributeError:
                    HYPER_NODE = HYPERNODE_LIST[-1]
                add_edge_with_entity_action(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=DEPTH)
    except IndexError:
        pass


def hyper_edge_with_entity_action(G: nx.Graph, NODE: str) -> None:
    """
    Add hypernym edges with ENTITY/ACTION root support.

    Args:
        G: NetworkX graph
        NODE: Node to process
    """
    try:
        HYPERNODE_LIST = wn.synset(NODE).hypernym_paths()[0][:-1]

        # Add root node (ENTITY or ACTION)
        if '.n.' in NODE:
            HYPERNODE_LIST.insert(0, 'ENTITY')
        elif '.v.' in NODE:
            HYPERNODE_LIST.insert(0, 'ACTION')
        else:
            raise ValueError('NODE should be noun or verb')

        try:
            HYPER_NODE = HYPERNODE_LIST[-1].name()
        except AttributeError:
            HYPER_NODE = HYPERNODE_LIST[-1]

        add_edge_with_entity_action(G, NODE, HYPER_NODE, HYPERNODE_LIST, DEPTH=0)

    except (WordNetError, IndexError):
        pass
    except ValueError:  # 'ENTITY', 'ACTION', or 'WORDNET'
        if NODE == 'ENTITY' or NODE == 'ACTION':
            G.add_edge(NODE, 'WORDNET', length=15)


# =============================================================================
# HELPER FUNCTIONS: STATISTICAL ANALYSIS
# =============================================================================

def surfstat(
    sem_path: Path,
    comp_no: int = 1
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Python version of MATLAB SurfStat using BrainStat.

    Performs group comparison (TD vs ASD) using GLM with covariates.

    Args:
        sem_path: Path to semantic space MATLAB file
        comp_no: PC component number

    Returns:
        Tuple of (tstat_results, pval_corrected)
    """
    from brainstat.stats.SLM import SLM
    from brainstat.stats.terms import FixedEffect
    from statsmodels.stats.multitest import multipletests

    # Load MATLAB file
    sem_demo_dict = loadmat(str(sem_path))

    sem = sem_demo_dict[f'sem{comp_no}']

    # Extract demographics
    group_list = [group[0] for group in sem_demo_dict['group'][0]]
    site_list = [site[0] for site in sem_demo_dict['site'][0]]
    sex_list = [sex[0] for sex in sem_demo_dict['sex'][0]]
    age_list = [age.astype(float) for age in sem_demo_dict['age'][0]]
    meanfd_list = [meanfd.astype(float) for meanfd in sem_demo_dict['meanfd'][0]]

    demo_df = pd.DataFrame(
        [group_list, site_list, sex_list, age_list, meanfd_list],
        index=['Group', 'Site', 'Sex', 'Age', 'MeanFD']
    ).T
    demo_df['Age'] = demo_df['Age'].astype(float)
    demo_df['MeanFD'] = demo_df['MeanFD'].astype(float)

    # Build GLM model
    term_group = FixedEffect(demo_df.Group)
    term_site = FixedEffect(demo_df.Site)
    term_sex = FixedEffect(demo_df.Sex)
    term_age = FixedEffect(demo_df.Age)
    term_meanfd = FixedEffect(demo_df.MeanFD)

    model = term_group + term_site + term_age + term_sex + term_meanfd
    contrast_group = (demo_df.Group == "ASD").astype(int) - \
                     (demo_df.Group == "TD").astype(int)

    # Fit model
    slm = SLM(model, contrast_group, correction=["fdr"], two_tailed=True)
    slm.fit(sem)

    # Get t-statistics
    tstat_results = slm.t.squeeze()

    # Calculate corrected p-values using scipy.stats
    pval = t_dist.sf(np.abs(slm.t), slm.df)
    pval_corrected = multipletests(pval.flatten(), alpha=0.05, method='fdr_bh')[1]

    return tstat_results, pval_corrected


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


def load_encoding_results(
    id_list: List[str],
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Dict[str, List[npt.NDArray]]:
    """
    Load encoding model results for subjects.

    Note:
        - Weights are loaded from seed-specific path: path/seed-{seed}/weight/
        - Other results (test_corrs, noise_ceiling, etc.) are loaded from: path/{key}/

    Args:
        id_list: List of subject IDs
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Dictionary with keys ['weight', 'test_corrs', 'noise_ceiling', 'test_corrs_corrected']
    """
    print('* Load encoding results')

    results_path = build_encoding_results_path(config, paths)

    # Load weights from seed-specific directory
    weight_list = load_weights_with_seed(id_list, results_path, config['seed'])
    gc.collect()  # Release file handles

    # Load other results from non-seeded directories
    other_keys = ['test_corrs', 'noise_ceiling', 'test_corrs_corrected']
    other_results = load_files(other_keys, id_list, results_path)
    gc.collect()  # Release file handles

    # Combine results
    results_dict = {
        'weight': weight_list,
        **other_results
    }

    return results_dict


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

    # Remove '.n.**' or '.v.**' suffixes from labels
    cleaned_label_list = []
    for label in label_list:
        if '.' in label:
            cleaned_label_list.append(label.split('.')[0])
        else:
            cleaned_label_list.append(label)

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
            continuous_cols=['Age']
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

    Uses option #2: Original weights for additions (not updated weights).

    Args:
        weight_list: List of weight arrays
        label_list: List of WordNet synset labels

    Returns:
        List of weight arrays with superordinate additions
    """
    print('* Add superordinate weights')

    weight_wordnet_list = []

    for weight in weight_list:
        weight_df = pd.DataFrame(weight, index=label_list)
        weight_df_original = weight_df.copy()

        for label in label_list:
            try:
                net = wn.synset(label)
                hyper_list = net.hypernym_paths()[0][:-1]

                for hyper in hyper_list:
                    hyper_name = hyper.name()
                    if hyper_name in label_list:
                        # Use original weights for updates
                        weight_df.loc[label] += weight_df_original.loc[hyper_name]

            except Exception as e:
                # Silently skip labels without hypernyms
                pass

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
# CORE ANALYSIS: TEMPLATE COMPUTATION
# =============================================================================

def compute_semantic_template(
    weight_list: List[npt.NDArray],
    method: str = 'pca',
    n_components: int = 10
) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
    """
    Compute semantic space template via PCA or diffusion map.

    Args:
        weight_list: List of weight arrays (one per subject)
        method: 'pca' or 'dm'
        n_components: Number of components to extract

    Returns:
        Tuple of (gradients, loadings) - loadings only for PCA
    """
    print('* Calculate semantic space template')

    weight_mean = np.array(weight_list).mean(0)

    if method == 'pca':
        emb = GradientMaps(
            n_components=min(weight_mean.shape),
            approach=method,
            random_state=0
        )
    elif method == 'dm':
        emb = GradientMaps(
            kernel='normalized_angle',
            n_components=n_components,
            approach=method,
            random_state=0
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit on z-scored mean
    emb.fit(zscore(weight_mean, axis=0), sparsity=0)

    gradients = emb.gradients_[:, :n_components]

    loadings = None
    if method == 'pca':
        loadings = emb.loadings_.T[:, :n_components]

    return gradients, loadings


def get_semantic_output_path(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Path:
    """
    Construct output path for semantic analysis results.

    Args:
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Path to semantic output directory
    """
    base = (paths['out'] / config['conf_option'] / config['atlas'] /
            f"chunk-{config['chunk_option']}" / 'fold-avg' / 'results' /
            f"k-{config['srm_option']}")

    if not config['bias_weight']:  # No gaze weight
        path = (base / 'gaze_weight-False' /
                f"wn-{config['reg_wordnet']}" / f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}" / 'sem' /
                f"mask_{config['perf_method']}-{config['perf_alpha']}" /
                f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}")
    else:  # Gaze weight
        path = (base /
                f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f"wn-{config['reg_wordnet']}" / f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}" / 'sem' /
                f"mask_{config['perf_method']}-{config['perf_alpha']}" /
                f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}")

    path.mkdir(parents=True, exist_ok=True)
    return path


def align_individual_spaces(
    weight_list_td: List[npt.NDArray],
    weight_list_asd: List[npt.NDArray],
    template: npt.NDArray,
    config: Dict[str, Any]
) -> List[List[npt.NDArray]]:
    """
    Align individual semantic spaces to template using Procrustes.

    Args:
        weight_list_td: List of TD weight arrays (already normalized)
        weight_list_asd: List of ASD weight arrays (already normalized)
        template: Template semantic space
        config: Configuration dictionary

    Returns:
        List of [sem_list_td, sem_list_asd]
    """
    print('* Align individual semantic spaces')

    # Align TD group (batch processing)
    print('  Aligning TD group...')
    emb_td = GradientMaps(
        n_components=10,
        approach='pca',
        alignment='procrustes',
        only_sign=config['pca_only_sign'],
        random_state=0
    )
    emb_td.fit(weight_list_td, reference=template, sparsity=0, n_iter=config['pca_iter'])
    sem_list_td = [sem[:, :10] for sem in emb_td.aligned_]

    # Align ASD group (batch processing)
    print('  Aligning ASD group...')
    emb_asd = GradientMaps(
        n_components=10,
        approach='pca',
        alignment='procrustes',
        only_sign=config['pca_only_sign'],
        random_state=0
    )
    emb_asd.fit(weight_list_asd, reference=template, sparsity=0, n_iter=config['pca_iter'])
    sem_list_asd = [sem[:, :10] for sem in emb_asd.aligned_]

    return [sem_list_td, sem_list_asd]


# =============================================================================
# CORE ANALYSIS: GROUP COMPARISON
# =============================================================================

def save_semantic_spaces(
    sem_list_td: List[npt.NDArray],
    sem_list_asd: List[npt.NDArray],
    sem_tpl: npt.NDArray,
    participants_df: pd.DataFrame,
    id_list_td_asd: List[str],
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Path:
    """
    Save semantic spaces as .mat file for surfstat.

    Args:
        sem_list_td: List of TD semantic spaces
        sem_list_asd: List of ASD semantic spaces
        sem_tpl: Template semantic space (unused but kept for compatibility)
        participants_df: Participant metadata DataFrame
        id_list_td_asd: Combined list of TD and ASD subject IDs
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Path to saved semantic space .mat file
    """
    # Combine TD and ASD lists
    sem_list_td_asd_combined = sem_list_td + sem_list_asd
    sem_array = np.array(sem_list_td_asd_combined)

    # Get output path
    sem_path = get_semantic_output_path(config, paths)

    # Construct file paths
    pca_template = config['pca_template']
    pca_only_sign = config['pca_only_sign']
    pca_iter = config['pca_iter']

    sem_mat_path = sem_path / f'sem_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.mat'

    # Prepare subject information
    group_info = participants_df.loc[id_list_td_asd, 'DX'].values
    site_info = participants_df.loc[id_list_td_asd, 'Site'].values
    age_info = participants_df.loc[id_list_td_asd, 'Age'].values
    sex_info = participants_df.loc[id_list_td_asd, 'Sex'].values
    meanfd_info = participants_df.loc[id_list_td_asd, 'Mean_FD_DM'].values

    # Save semantic space as .mat file
    sem_dict = {
        'sem1': sem_array[:, :, 0],
        'sem2': sem_array[:, :, 1],
        'sem3': sem_array[:, :, 2],
        'group': group_info,
        'site': site_info,
        'age': age_info,
        'sex': sex_info,
        'meanfd': meanfd_info
    }
    safe_save(
        sem_dict,
        sem_mat_path,
        save_func='matlab',
        overwrite=config['overwrite']
    )

    return sem_mat_path


def perform_group_comparison(
    sem_path: Path,
    config: Dict[str, Any]
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Perform group comparison (TD vs ASD) using surfstat.

    Args:
        sem_path: Path to semantic space .mat file
        config: Configuration dictionary

    Returns:
        Tuple of (tstat_results, pval_corrected)
    """
    print('* Perform group comparison (TD vs ASD)')

    comp_no = config['comp_no']

    # Perform surfstat
    tstat_results, pval_corrected = surfstat(
        sem_path,
        comp_no=comp_no
    )

    return tstat_results, pval_corrected


def identify_significant_regions(
    tstat_results: npt.NDArray,
    pval_corrected: npt.NDArray,
    config: Dict[str, Any]
) -> npt.NDArray:
    """
    Identify significant regions using FDR-corrected p-values.

    Args:
        tstat_results: T-statistic array (360,) for all ROIs
        pval_corrected: FDR-corrected p-values (360,) for all ROIs
        config: Configuration dictionary

    Returns:
        sig_regions: Significance array (360,) - (-1 for negative, 0 for non-sig, 1 for positive)
    """
    print('* Identify significant regions')

    glm_alpha = config['glm_alpha']

    # Initialize significance array
    sig_regions = np.zeros(360).astype(int)

    # Mark significant regions
    sig_mask = pval_corrected < glm_alpha
    sig_regions[sig_mask] = np.sign(tstat_results[sig_mask]).astype(int)

    print(f'  Number of significant regions: {len(sig_regions.nonzero()[0])}')

    return sig_regions


# =============================================================================
# CORE ANALYSIS: VISUAL PATHWAY DEFINITIONS
# =============================================================================

def define_visual_pathways(
    mmp_info_df: pd.DataFrame
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Define expanded ROI lists for ventral, dorsal, and social visual pathways.

    Args:
        mmp_info_df: MMP atlas metadata DataFrame

    Returns:
        Tuple of dictionaries for ventral/dorsal/social ROI lists and indices
    """
    mmp_roi_list = mmp_info_df['ROI'].tolist()

    # Ventral pathway (expanded)
    ventral_roi_list_expanded = {
        'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
        'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'L_V4_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_V4_ROI'],
        'Ventral Occipitotemporal': [
            'L_FFC_ROI', 'L_PIT_ROI', 'L_V8_ROI', 'L_VMV1_ROI', 'L_VMV2_ROI', 'L_VMV3_ROI', 'L_VVC_ROI',
            'R_FFC_ROI', 'R_PIT_ROI', 'R_V8_ROI', 'R_VMV1_ROI', 'R_VMV2_ROI', 'R_VMV3_ROI', 'R_VVC_ROI'],
        'Inferior Temporal': [
            'L_PHT_ROI', 'L_TE1a_ROI', 'L_TE1m_ROI', 'L_TE1p_ROI', 'L_TE2a_ROI', 'L_TE2p_ROI',
            'R_PHT_ROI', 'R_TE1a_ROI', 'R_TE1m_ROI', 'R_TE1p_ROI', 'R_TE2a_ROI', 'R_TE2p_ROI'],
        'Temporal Pole': [
            'L_TGv_ROI', 'L_TGd_ROI',
            'R_TGv_ROI', 'R_TGd_ROI'],
        'Oribital & Polar Frontal': [
            'L_a47r_ROI', 'L_47m_ROI', 'L_47s_ROI', 'L_10p_ROI', 'L_11l_ROI', 'L_13l_ROI', 'L_OFC_ROI', 'L_pOFC_ROI', 'L_10pp_ROI', 'L_10d_ROI',
            'R_a47r_ROI', 'R_47m_ROI', 'R_47s_ROI', 'R_10p_ROI', 'R_11l_ROI', 'R_13l_ROI', 'R_OFC_ROI', 'R_pOFC_ROI', 'R_10pp_ROI', 'R_10d_ROI']
    }

    ventral_roi_idx_list_expanded = {
        key: [i for i, roi in enumerate(mmp_roi_list) if roi in ventral_roi_list_expanded[key]]
        for key in ventral_roi_list_expanded.keys()
    }

    # Dorsal pathway (expanded)
    dorsal_roi_list_expanded = {
        'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
        'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'L_V3A_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_V3A_ROI'],
        'MT+ complex': [
            'L_LO1_ROI', 'L_LO2_ROI', 'L_LO3_ROI', 'L_MST_ROI', 'L_MT_ROI', 'L_V3CD_ROI', 'L_V4t_ROI',
            'R_LO1_ROI', 'R_LO2_ROI', 'R_LO3_ROI', 'R_MST_ROI', 'R_MT_ROI', 'R_V3CD_ROI', 'R_V4t_ROI'],
        'Intraparietal': [
            'L_AIP_ROI', 'L_LIPd_ROI', 'L_LIPv_ROI', 'L_MIP_ROI', 'L_VIP_ROI', 'L_IP0_ROI', 'L_IP1_ROI', 'L_IP2_ROI',
            'R_AIP_ROI', 'R_LIPd_ROI', 'R_LIPv_ROI', 'R_MIP_ROI', 'R_VIP_ROI', 'R_IP0_ROI', 'R_IP1_ROI', 'R_IP2_ROI'],
        'Area 7': [
            'L_7AL_ROI', 'L_7Am_ROI', 'L_7PC_ROI', 'L_7PL_ROI', 'L_7Pm_ROI',
            'R_7AL_ROI', 'R_7Am_ROI', 'R_7PC_ROI', 'R_7PL_ROI', 'R_7Pm_ROI'],
        'Premotor & Eye Fields': [
            'L_55b_ROI', 'L_6a_ROI', 'L_6d_ROI', 'L_FEF_ROI',
            'R_55b_ROI', 'R_6a_ROI', 'R_6d_ROI', 'R_FEF_ROI'],
        'Dorsolateral Prefrontal': [
            'L_46_ROI', 'L_8Ad_ROI', 'L_8Av_ROI', 'L_8BL_ROI', 'L_8C_ROI', 'L_9-46d_ROI', 'L_9a_ROI', 'L_9p_ROI', 'L_a9-46v_ROI', 'L_i6-8_ROI', 'L_p9-46v_ROI', 'L_s6-8_ROI', 'L_SFL_ROI',
            'R_46_ROI', 'R_8Ad_ROI', 'R_8Av_ROI', 'R_8BL_ROI', 'R_8C_ROI', 'R_9-46d_ROI', 'R_9a_ROI', 'R_9p_ROI', 'R_a9-46v_ROI', 'R_i6-8_ROI', 'R_p9-46v_ROI', 'R_s6-8_ROI', 'R_SFL_ROI'],
        'Precuneus & RSC': [
            'L_31a_ROI', 'L_31pd_ROI', 'L_31pv_ROI', 'L_PCV_ROI', 'L_7m_ROI', 'L_d23ab_ROI', 'L_v23ab_ROI', 'L_RSC_ROI', 'L_POS1_ROI', 'L_POS2_ROI', 'L_DVT_ROI', 'L_ProS_ROI',
            'R_31a_ROI', 'R_31pd_ROI', 'R_31pv_ROI', 'R_PCV_ROI', 'R_7m_ROI', 'R_d23ab_ROI', 'R_v23ab_ROI', 'R_RSC_ROI', 'R_POS1_ROI', 'R_POS2_ROI', 'R_DVT_ROI', 'R_ProS_ROI']
    }

    dorsal_roi_idx_list_expanded = {
        key: [i for i, roi in enumerate(mmp_roi_list) if roi in dorsal_roi_list_expanded[key]]
        for key in dorsal_roi_list_expanded.keys()
    }

    # Social pathway (expanded)
    social_roi_list_expanded = {
        'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
        'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'R_V2_ROI', 'R_V3_ROI'],
        'MT+ complex': [
            'L_LO1_ROI', 'L_LO2_ROI', 'L_LO3_ROI', 'L_V4t_ROI', 'L_MT_ROI', 'L_MST_ROI', 'L_FST_ROI',
            'R_LO1_ROI', 'R_LO2_ROI', 'R_LO3_ROI', 'R_V4t_ROI', 'R_MT_ROI', 'R_MST_ROI', 'R_FST_ROI'],
        'Superior Temporal Sulcus': [
            'L_STSda_ROI', 'L_STSdp_ROI', 'L_STSva_ROI', 'L_STSvp_ROI',
            'R_STSda_ROI', 'R_STSdp_ROI', 'R_STSva_ROI', 'R_STSvp_ROI'],
        'Temporal-Parietal-Occipital Junction': [
            'L_TPOJ1_ROI', 'L_TPOJ2_ROI', 'L_TPOJ3_ROI', 'L_STV_ROI', 'L_PSL_ROI',
            'R_TPOJ1_ROI', 'R_TPOJ2_ROI', 'R_TPOJ3_ROI', 'R_STV_ROI', 'R_PSL_ROI'],
        'Precuneus': [
            'L_31a_ROI', 'L_31pd_ROI', 'L_31pv_ROI', 'L_PCV_ROI', 'L_7m_ROI', 'L_d23ab_ROI', 'L_v23ab_ROI',
            'R_31a_ROI', 'R_31pd_ROI', 'R_31pv_ROI', 'R_PCV_ROI', 'R_7m_ROI', 'R_d23ab_ROI', 'R_v23ab_ROI'],
        'Inferior Frontal & Medial Prefrontal': [
            'L_44_ROI', 'L_45_ROI', 'L_IFJa_ROI', 'L_IFSp_ROI', 'L_IFSa_ROI', 'L_p47r_ROI', 'L_47l_ROI', 'L_8bm_ROI', 'L_9m_ROI', 'L_d32_ROI', 'L_a24_ROI', 'L_p24_ROI', 'L_a32pr_ROI',
            'R_44_ROI', 'R_45_ROI', 'R_IFJa_ROI', 'R_IFSp_ROI', 'R_IFSa_ROI', 'R_p47r_ROI', 'R_47l_ROI', 'R_8bm_ROI', 'R_9m_ROI', 'R_d32_ROI', 'R_a24_ROI', 'R_p24_ROI', 'R_a32pr_ROI']
    }

    social_roi_idx_list_expanded = {
        key: [i for i, roi in enumerate(mmp_roi_list) if roi in social_roi_list_expanded[key]]
        for key in social_roi_list_expanded.keys()
    }

    return (
        ventral_roi_list_expanded,
        ventral_roi_idx_list_expanded,
        dorsal_roi_list_expanded,
        dorsal_roi_idx_list_expanded,
        social_roi_list_expanded,
        social_roi_idx_list_expanded
    )


# =============================================================================
# OUTPUT & VISUALIZATION
# =============================================================================

def save_all_outputs(
    sem_list_td: List[npt.NDArray],
    sem_list_asd: List[npt.NDArray],
    sem_tpl: npt.NDArray,
    sem_tpl_loading: npt.NDArray,
    tstat_results: npt.NDArray,
    pval_corrected: npt.NDArray,
    sig_regions: npt.NDArray,
    config: Dict[str, Any],
    paths: Dict[str, Path],
    atlases: Dict[str, npt.NDArray]
) -> None:
    """
    Save all analysis outputs with conditional overwrite protection.

    Args:
        sem_list_td: List of TD semantic spaces
        sem_list_asd: List of ASD semantic spaces
        sem_tpl: Semantic template (group-level PCA)
        sem_tpl_loading: Semantic template loadings
        tstat_results: T-statistic results
        pval_corrected: FDR-corrected p-values (360,)
        sig_regions: Significant regions array
        config: Configuration dictionary
        paths: Path dictionary
        atlases: Atlas dictionary
    """
    if not config['save_outputs']:
        print('* Output saving disabled')
        return

    print('* Save analysis outputs')
    sem_path = get_semantic_output_path(config, paths)

    pca_template = config['pca_template']
    pca_only_sign = config['pca_only_sign']
    pca_iter = config['pca_iter']
    glm_alpha = config['glm_alpha']

    # Save semantic spaces
    safe_save(
        np.array(sem_list_td),
        sem_path / 'sem_list_td.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )
    safe_save(
        np.array(sem_list_asd),
        sem_path / 'sem_list_asd.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save average semantic spaces
    safe_save(
        np.array(sem_list_td).mean(0),
        sem_path / 'avg_sem_td.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )
    safe_save(
        np.array(sem_list_asd).mean(0),
        sem_path / 'avg_sem_asd.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save semantic template
    safe_save(
        sem_tpl,
        sem_path / 'sem_tpl.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )
    safe_save(
        sem_tpl_loading,
        sem_path / 'sem_tpl_loading.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save t-statistics
    safe_save(
        tstat_results,
        sem_path / f'tstat_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save FDR-corrected p-values (for lenient overlap thresholding in 09_dimensionality.py)
    safe_save(
        pval_corrected,
        sem_path / f'pval_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy',
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save significant regions
    if glm_alpha == 0.025:
        filename = f'sig_regions_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy'
    elif glm_alpha == 0.005:
        filename = f'sig_regions_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}_alpha-0.01.npy'
    elif glm_alpha == 0.05:
        filename = f'sig_regions_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}_alpha-0.1.npy'
    else:
        filename = f'sig_regions_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}_alpha-{int(glm_alpha)*2}.npy'

    safe_save(
        sig_regions,
        sem_path / filename,
        save_func='numpy',
        overwrite=config['overwrite']
    )

    # Save brain maps as CIFTI
    if config.get('make_figures', True):
        require_surface_plotting()
        # T-statistics brain map
        roiarray_2_32knii(
            tstat_results,
            atlases['brain_mask_10k'],
            atlas='mmp',
            output_folder='brain',
            output_file=f'tstat_{pca_template}.dscalar.nii',
            fig_path=paths['fig']
        )

        # Significant regions brain map
        roiarray_2_32knii(
            sig_regions.astype('float32'),
            atlases['brain_mask_10k'],
            atlas='mmp',
            output_folder='brain',
            output_file=f'sig_regions.dscalar.nii',
            fig_path=paths['fig']
        )

    print('  [ok] All outputs saved')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main PCA-based group comparison pipeline.

    Pipeline steps:
    1. Load participants (TD + ASD)
    2. Load encoding results
    3. Load regressors
    4. Preprocess weights (harmonize, regress, mask, normalize)
    5. Compute semantic template (PCA)
    6. Align individual spaces (Procrustes)
    7. Save semantic spaces (single .mat file)
    8. Group comparison (single GLM on all ROIs)
    9. Identify significant regions (FDR-corrected)
    10. Save outputs
    11. Visualize (Windows only)

    Simplified workflow: Removed positive/negative component separation.
    All ROIs are now analyzed together in a single statistical test.
    """
    print("=" * 80)
    print("07_pca.py - PCA-based Group Comparison Analysis")
    print("=" * 80)
    print(f"\nConfiguration: {CONFIG['atlas']} atlas, {CONFIG['conf_option']} denoising")
    print(f"Template: {CONFIG['pca_template']}, Component: {CONFIG['comp_no']}")
    print(f"GLM alpha: {CONFIG['glm_alpha']}, Overwrite: {CONFIG['overwrite']}\n")

    # =========================================================================
    # 1. Load participants
    # =========================================================================
    print("\n[1/11] Loading participants...")
    id_list_td, participants_df_td = load_participant_data('TD', CONFIG, PATHS)
    id_list_asd, participants_df_asd = load_participant_data('ASD', CONFIG, PATHS)

    # Combine participant lists
    id_list_td_asd = id_list_td + id_list_asd
    # Both participants_df_td and participants_df_asd are the same full DataFrame
    participants_df = participants_df_td

    print(f"  TD: {len(id_list_td)} subjects")
    print(f"  ASD: {len(id_list_asd)} subjects")
    print(f"  Total: {len(id_list_td_asd)} subjects")

    # =========================================================================
    # 2. Load encoding results
    # =========================================================================
    print("\n[2/11] Loading encoding results...")
    results_td = load_encoding_results(id_list_td, CONFIG, PATHS)
    results_asd = load_encoding_results(id_list_asd, CONFIG, PATHS)

    weight_list_td = results_td['weight']
    weight_list_asd = results_asd['weight']

    # =========================================================================
    # 3. Load regressors
    # =========================================================================
    print("\n[3/11] Loading regressors...")
    label_list_df, label_list = load_regressors(id_list_td_asd, CONFIG, PATHS)
    print(f"  Loaded {len(label_list)} semantic labels")

    # clean label list by removing prefixes (e.g., 'ball.n.01' -> 'ball')
    label_list_clean = [label.split('.')[0] for label in label_list]

    # =========================================================================
    # 4. Preprocess weights
    # =========================================================================
    print("\n[4/11] Preprocessing weights...")

    # Process across delays
    weight_list_td = process_weights_across_delays(weight_list_td, CONFIG)
    weight_list_asd = process_weights_across_delays(weight_list_asd, CONFIG)

    # Harmonize (neuroCombat)
    weight_concat = np.vstack([np.array(weight_list_td), np.array(weight_list_asd)])
    weight_concat_harmonized = harmonize_weights(weight_concat, participants_df, id_list_td_asd)
    weight_list_td = list(weight_concat_harmonized[:len(id_list_td)])
    weight_list_asd = list(weight_concat_harmonized[len(id_list_td):])

    # Regress label frequency
    weight_list_td = regress_label_frequency(weight_list_td, label_list, PATHS)
    weight_list_asd = regress_label_frequency(weight_list_asd, label_list, PATHS)

    # Mask by performance
    perf_mask = load_performance_mask(CONFIG, PATHS)
    weight_list_td = mask_by_performance(weight_list_td, perf_mask)
    weight_list_asd = mask_by_performance(weight_list_asd, perf_mask)

    # Add superordinate weights
    if CONFIG['weight_add_superordinate']:
        weight_list_td = add_superordinate_weights(weight_list_td, label_list)
        weight_list_asd = add_superordinate_weights(weight_list_asd, label_list)

    # Normalize
    weight_list_td = normalize_weights(weight_list_td)
    weight_list_asd = normalize_weights(weight_list_asd)

    # =========================================================================
    # 5. Compute semantic template
    # =========================================================================
    print(f"\n[5/11] Computing semantic template (mode: {CONFIG['pca_template']})...")

    # Select weight list based on template option
    if CONFIG['pca_template'] == 'Whole':
        weight_list_template = weight_list_td + weight_list_asd
    elif CONFIG['pca_template'] == 'TD':
        weight_list_template = weight_list_td
    elif CONFIG['pca_template'] == 'ASD':
        weight_list_template = weight_list_asd
    else:
        raise ValueError(f"Unknown pca_template option: {CONFIG['pca_template']}")

    sem_tpl, sem_tpl_loading = compute_semantic_template(
        weight_list_template,
        method='pca',
        n_components=10
    )
    print(f"  Template shape: {sem_tpl.shape}")

    comp_no = CONFIG['comp_no']

    # Convert template to section and Yeo17 space
    sem_tpl_section = mmp_2_section(
        sem_tpl[:, comp_no-1],
        ATLASES['mmp'],
        ATLASES['section'],
        model_perf_mask=perf_mask
    ).squeeze()

    # Get sorted section list according to sem_tpl
    sem_tpl_section_df = pd.DataFrame(
        sem_tpl_section,
        index=MMP_SECTION_LIST,
        columns=[f'PC#{comp_no}']
    )
    sem_tpl_section_df = sem_tpl_section_df.sort_values(by=f'PC#{comp_no}', ascending=False)
    mmp_section_list_sorted = list(sem_tpl_section_df.index)

    if CONFIG['comp_no'] == 1:  # Only check for PC1 sign
        if sem_tpl_section_df.loc['TPOJ']['PC#1'] < 0:  # Invert if PC1 is negative in TPOJ
            sem_tpl_section_df[f'PC#{comp_no}'] = -sem_tpl_section_df[f'PC#{comp_no}']
            sem_tpl[:, comp_no-1] = -sem_tpl[:, comp_no-1]
            sem_tpl_loading[:, comp_no-1] = -sem_tpl_loading[:, comp_no-1]
            print("  Note: Inverted PC1 sign to ensure TPOJ is positive")

    # =========================================================================
    # 6. Align individual spaces
    # =========================================================================
    print("\n[6/11] Aligning individual semantic spaces...")
    sem_list_td, sem_list_asd = align_individual_spaces(
        weight_list_td,
        weight_list_asd,
        sem_tpl,
        CONFIG
    )
    print(f"  TD aligned: {len(sem_list_td)}")
    print(f"  ASD aligned: {len(sem_list_asd)}")

    # =========================================================================
    # 7. Save semantic spaces
    # =========================================================================
    print("\n[7/11] Saving semantic spaces for surfstat...")
    sem_path = save_semantic_spaces(
        sem_list_td,
        sem_list_asd,
        sem_tpl,
        participants_df,
        id_list_td_asd,
        CONFIG,
        PATHS
    )

    # =========================================================================
    # 8. Perform group comparison
    # =========================================================================
    print("\n[8/11] Performing group comparison...")
    tstat_masked, pval_masked = perform_group_comparison(
        sem_path,
        CONFIG
    )
    print(f"  T-statistics computed (masked): {tstat_masked.shape}")

    # Expand to full 360 ROI space
    tstat_results = np.zeros(360)
    tstat_results[perf_mask] = tstat_masked

    pval_corrected = np.ones(360)  # Non-significant p-values (1.0) by default
    pval_corrected[perf_mask] = pval_masked
    print(f"  Expanded to full ROI space: {tstat_results.shape}")

    # =========================================================================
    # 9. Identify significant regions
    # =========================================================================
    print("\n[9/11] Identifying significant regions...")
    sig_regions = identify_significant_regions(
        tstat_results,
        pval_corrected,
        CONFIG
    )

    sig_regions_eg = mmp_2_eg(sig_regions, ATLASES['mmp'], ATLASES['eg'], sum=True)
    sig_regions_eg_df = pd.DataFrame(sig_regions_eg, index=EG17_LIST, columns=['Significant Regions'])

    # =========================================================================
    # 10. Save outputs
    # =========================================================================
    print("\n[10/11] Saving outputs...")
    save_all_outputs(
        sem_list_td,
        sem_list_asd,
        sem_tpl,
        sem_tpl_loading,
        tstat_results,
        pval_corrected,
        sig_regions,
        CONFIG,
        PATHS,
        ATLASES
    )

    # =========================================================================
    # 11. Visualize
    # =========================================================================
    print("\n[11/11] Visualization...")
    if CONFIG.get('make_figures', True):
        require_surface_plotting()

        # =====================================================================
        # 11a. Visualize beta wordclouds
        # =====================================================================
        print("* Visualize beta wordclouds")

        # Compute mean weights for visualization
        weight_list_all = weight_list_td + weight_list_asd
        weight_mean = np.array(weight_list_all).mean(0)

        # Example ROIs to visualize (can be customized)
        example_rois = ['R_PHA1_ROI', 'L_PCV_ROI', 'R_FFC_ROI', 'L_TPOJ1_ROI']

        # Visualize beta wordcloud for example ROIs
        for roi in example_rois:
            try:
                visualize_beta_wordcloud(
                    roi=roi,
                    weight_mean=weight_mean,
                    label_list=label_list_clean,
                    perf_mask=perf_mask,
                    paths=PATHS,
                    percentile=90,
                    n_top_words=10,
                    save_path=PATHS['fig'] / 'wordcloud' / f'{roi}_beta_wordcloud.png'
                )
            except Exception as e:
                print(f"  Warning: Could not visualize {roi}: {e}")

        # =====================================================================
        # 11b. Visualize PC scores
        # =====================================================================
        print("* Visualize PC scores")

        cmax = 10
        cmin = -cmax
        cmap = 'coolwarm'

        # Add color column based on PC component
        sem_tpl_section_df[f'PC#{comp_no}_color'] = sem_tpl_section_df[f'PC#{comp_no}'].apply(
            lambda x: value_to_rgb(x, cmap, cmin, cmax)
        )

        # Expand template to full 360 ROI space
        sem_tpl_mmp = np.zeros(360).astype('float32')
        sem_tpl_mmp[perf_mask] = sem_tpl[:, comp_no-1]

        # Save semantic template as dscalar
        pca_template = CONFIG['pca_template']
        sem_path_obj = get_semantic_output_path(CONFIG, PATHS)

        roiarray_2_32knii(
            sem_tpl_mmp,
            perf_mask,
            atlas='mmp',
            output_folder='brain',
            output_file=f'tpl_{pca_template}_PC{comp_no}.dscalar.nii',
            fig_path=PATHS['fig'],
            comp_no=comp_no
        )

        # Save performance mask as 32k dscalar
        perf_mask_array = np.zeros(360, dtype='float32')
        perf_mask_array[perf_mask] = 1.0
        roiarray_2_32knii(
            perf_mask_array,
            ATLASES['brain_mask_10k'],
            atlas='mmp',
            output_folder='brain',
            output_file='perf_mask.dscalar.nii',
            fig_path=PATHS['fig'],
        )

        # Plot brain surface visualization
        try:
            # Load surface meshes
            mmp_folder = PATHS['tpl'] / 'MMP'
            surfs = [None] * 2
            surfs[0] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.L.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
            surf_lh = serial_connect(surfs[0], nf)
            surfs[1] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.R.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
            surf_rh = serial_connect(surfs[1], nf)
            
            # Map template to surface
            mmp_atlas = ATLASES['mmp']
            sem_tpl_32k = map_to_labels(
                sem_tpl_mmp,
                mmp_atlas,
                mask=mmp_atlas != 0,
                fill=np.nan
            )

            # Plot hemispheres
            plot_hemispheres(
                surf_lh,
                surf_rh,
                array_name=sem_tpl_32k,
                size=(1200, 200),
                cmap=cmap,
                nan_color=(0.5, 0.5, 0.5, 1),
                color_bar=True,
                color_range=(cmin, cmax),
                zoom=1.6,
            )

            # Visualize avg_sem_td and avg_sem_asd
            print("  Visualizing average semantic spaces (TD and ASD)...")

            # Compute average semantic spaces for the current component
            avg_sem_td_comp = np.zeros(360).astype('float32')
            avg_sem_asd_comp = np.zeros(360).astype('float32')
            avg_sem_td_comp[perf_mask] = np.array(sem_list_td).mean(0)[:, comp_no-1]
            avg_sem_asd_comp[perf_mask] = np.array(sem_list_asd).mean(0)[:, comp_no-1]

            # Save avg_sem_td as dscalar
            roiarray_2_32knii(
                avg_sem_td_comp,
                perf_mask,
                atlas='mmp',
                output_folder='brain',
                output_file=f'avg_sem_td_{pca_template}_PC{comp_no}.dscalar.nii',
                fig_path=PATHS['fig'],
                comp_no=comp_no
            )

            # Save avg_sem_asd as dscalar
            roiarray_2_32knii(
                avg_sem_asd_comp,
                perf_mask,
                atlas='mmp',
                output_folder='brain',
                output_file=f'avg_sem_asd_{pca_template}_PC{comp_no}.dscalar.nii',
                fig_path=PATHS['fig'],
                comp_no=comp_no
            )

            # Map to 32k surface for visualization
            avg_sem_td_32k = map_to_labels(
                avg_sem_td_comp,
                mmp_atlas,
                mask=mmp_atlas != 0,
                fill=np.nan
            )
            avg_sem_asd_32k = map_to_labels(
                avg_sem_asd_comp,
                mmp_atlas,
                mask=mmp_atlas != 0,
                fill=np.nan
            )

            # Plot TD and ASD average semantic spaces in one window
            print(f"  Plotting avg_sem_td and avg_sem_asd (PC{comp_no})...")
            plot_hemispheres(
                surf_lh,
                surf_rh,
                array_name=[avg_sem_td_32k, avg_sem_asd_32k],
                size=(1200, 400),
                cmap=cmap,
                nan_color=(0.5, 0.5, 0.5, 1),
                color_bar=True,
                color_range=(-3, 3),
                zoom=1.6,
                label_text=['TD', 'ASD'],
            )

            print(f"  Generated brain surface plots for template, avg_sem_td, and avg_sem_asd")

        except Exception as e:
            print(f"  Warning: Could not create brain surface plot: {e}")

        # Plot lollipop chart for MMP sections
        try:
            fig, ax = plt.subplots(figsize=(13, 8))

            plt.title(f'PC#{comp_no} - Semantic template (MMP section)')

            # Create stems
            plt.stem(sem_tpl_section_df[f'PC#{comp_no}'], linefmt='silver', markerfmt=' ', basefmt=' ')

            # Add colored dots
            for i, section in enumerate(mmp_section_list_sorted):
                plt.plot(
                    i,
                    sem_tpl_section_df.loc[section, f'PC#{comp_no}'],
                    marker='o',
                    color=sem_tpl_section_df.loc[section, f'PC#{comp_no}_color'],
                    markersize=24
                )

            # Customize appearance
            plt.xticks(np.arange(len(mmp_section_list_sorted)), mmp_section_list_sorted, rotation=45, ha='right')
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(tick1On=True)
            plt.tight_layout()
            plt.show()

            print(f"  Generated lollipop chart for PC{comp_no}")

        except Exception as e:
            print(f"  Warning: Could not create lollipop chart: {e}")

        # Interpret template using dual stream hypothesis
        print("* Interpret template using dual stream hypothesis")

        try:
            # Load MMP info
            mmp_info_path = PATHS['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
            mmp_info_df = pd.read_excel(str(mmp_info_path), engine='openpyxl')
            mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]
            mmp_roi_list = mmp_info_df['ROI'].tolist()

            # Define ROI lists for each pathway (option #2: second version)
            ventral_roi_list = [
                # Ventral Occipitotemporal
                'L_FFC_ROI', 'L_PIT_ROI', 'L_V8_ROI', 'L_VMV1_ROI', 'L_VMV2_ROI', 'L_VMV3_ROI', 'L_VVC_ROI',
                'R_FFC_ROI', 'R_PIT_ROI', 'R_V8_ROI', 'R_VMV1_ROI', 'R_VMV2_ROI', 'R_VMV3_ROI', 'R_VVC_ROI',
                # Lateral Temporal
                'L_PHT_ROI', 'L_TE1a_ROI', 'L_TE1m_ROI', 'L_TE1p_ROI', 'L_TE2a_ROI', 'L_TE2p_ROI', 'L_TGv_ROI', 'L_TGd_ROI',
                'R_PHT_ROI', 'R_TE1a_ROI', 'R_TE1m_ROI', 'R_TE1p_ROI', 'R_TE2a_ROI', 'R_TE2p_ROI', 'R_TGv_ROI', 'R_TGd_ROI'
            ]
            dorsal_roi_list = [
                # Intraparietal
                'L_AIP_ROI', 'L_LIPd_ROI', 'L_LIPv_ROI', 'L_MIP_ROI', 'L_VIP_ROI', 'L_IP0_ROI', 'L_IP1_ROI', 'L_IP2_ROI',
                'R_AIP_ROI', 'R_LIPd_ROI', 'R_LIPv_ROI', 'R_MIP_ROI', 'R_VIP_ROI', 'R_IP0_ROI', 'R_IP1_ROI', 'R_IP2_ROI',
                # Area 7
                'L_7AL_ROI', 'L_7Am_ROI', 'L_7PC_ROI', 'L_7PL_ROI', 'L_7Pm_ROI',
                'R_7AL_ROI', 'R_7Am_ROI', 'R_7PC_ROI', 'R_7PL_ROI', 'R_7Pm_ROI',
                # Premotor & Eye Fields
                'L_55b_ROI', 'L_6a_ROI', 'L_6d_ROI', 'L_FEF_ROI',
                'R_55b_ROI', 'R_6a_ROI', 'R_6d_ROI', 'R_FEF_ROI'
            ]
            social_roi_list = [
                # Superior Temporal Sulcus
                'L_STSda_ROI', 'L_STSdp_ROI', 'L_STSva_ROI', 'L_STSvp_ROI',
                'R_STSda_ROI', 'R_STSdp_ROI', 'R_STSva_ROI', 'R_STSvp_ROI',
                # Temporal-Parietal-Occipital Junction
                'L_TPOJ1_ROI', 'L_TPOJ2_ROI', 'L_TPOJ3_ROI', 'L_STV_ROI', 'L_PSL_ROI',
                'R_TPOJ1_ROI', 'R_TPOJ2_ROI', 'R_TPOJ3_ROI', 'R_STV_ROI', 'R_PSL_ROI'
            ]

            # Get indices for each pathway
            ventral_roi_idx_list = [i for i, roi in enumerate(mmp_roi_list) if roi in ventral_roi_list]
            dorsal_roi_idx_list = [i for i, roi in enumerate(mmp_roi_list) if roi in dorsal_roi_list]
            social_roi_idx_list = [i for i, roi in enumerate(mmp_roi_list) if roi in social_roi_list]
            others_roi_idx_list = [i for i, roi in enumerate(mmp_roi_list)
                                  if roi not in ventral_roi_list and roi not in dorsal_roi_list
                                  and roi not in social_roi_list]
            pathway_mask = np.array([i for i in range(360) if i not in others_roi_idx_list])

            # Save brain map as dlabel.nii file
            dummy_dlabel_path = PATHS['tpl'] / 'MMP' / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
            dummy_dlabel = nib.load(str(dummy_dlabel_path))
            dummy_32k_nonmed = dummy_dlabel.get_fdata().copy().astype(int)
            label_color_dict = dummy_dlabel.header.get_axis(0).label
            new_label_color_dict = deepcopy(label_color_dict)

            # Define pathway colors
            ventral_rgba = (179/255, 138/255, 236/255, 120/255)  # Purple
            dorsal_rgba = (160/255, 244/255, 164/255, 120/255)   # Green
            social_rgba = (190/255, 44/255, 73/255, 120/255)     # Red
            others_rgba = (222/255, 222/255, 222/255, 120/255)   # Gray

            # Assign colors to each ROI
            for mmp_idx in range(360):
                if new_label_color_dict[0][mmp_idx+1][0] in ventral_roi_list:
                    new_label_color_dict[0][mmp_idx+1] = (new_label_color_dict[0][mmp_idx+1][0], ventral_rgba)
                elif new_label_color_dict[0][mmp_idx+1][0] in dorsal_roi_list:
                    new_label_color_dict[0][mmp_idx+1] = (new_label_color_dict[0][mmp_idx+1][0], dorsal_rgba)
                elif new_label_color_dict[0][mmp_idx+1][0] in social_roi_list:
                    new_label_color_dict[0][mmp_idx+1] = (new_label_color_dict[0][mmp_idx+1][0], social_rgba)
                else:  # others
                    new_label_color_dict[0][mmp_idx+1] = (new_label_color_dict[0][mmp_idx+1][0], others_rgba)

            # Create new CIFTI image with updated colors
            label_axis = dummy_dlabel.header.get_axis(0)
            label_axis.label = new_label_color_dict
            new_header = nib.Cifti2Header.from_axes((label_axis, dummy_dlabel.header.get_axis(1)))
            dlabel_32k_nonmed_cifti = nib.Cifti2Image(dummy_32k_nonmed, header=new_header)

            # Save pathway dlabel file
            pathway_dlabel_path = PATHS['fig'] / 'brain' / 'visual_pathways.dlabel.nii'
            pathway_dlabel_path.parent.mkdir(parents=True, exist_ok=True)
            dlabel_32k_nonmed_cifti.to_filename(str(pathway_dlabel_path))
            print(f"  Saved visual pathways dlabel: {pathway_dlabel_path.name}")

            # Correlation between sem_tpl and pathway map
            # Pathway map: {ventral: 0, dorsal: 0, social: 1}
            pathway_map = np.zeros_like(sem_tpl_mmp)
            pathway_map[ventral_roi_idx_list] = 0
            pathway_map[dorsal_roi_idx_list] = 0
            pathway_map[social_roi_idx_list] = 1

            # Point-biserial correlation
            sem_tpl_comp_corr, sem_tpl_comp_pval = pointbiserialr(
                sem_tpl_mmp[pathway_mask],
                pathway_map[pathway_mask]
            )
            print(f'  Semantic template PC#{comp_no} correlation with pathway map: '
                  f'{sem_tpl_comp_corr:.3f}, p-value: {sem_tpl_comp_pval:.3f}')

            # Draw scatter plot between sem_tpl and pathway map
            fig, ax = plt.subplots(figsize=(9, 12))
            norm_pathway = plt.Normalize(-12, 12)
            cmap_pathway = plt.get_cmap('coolwarm')

            sns.regplot(
                x=pathway_map[pathway_mask],
                y=sem_tpl_mmp[pathway_mask],
                color='black',
                ax=ax,
                scatter_kws={
                    's': 2500,
                    'color': cmap_pathway(norm_pathway(sem_tpl_mmp[pathway_mask])),
                    'alpha': 1
                },
                line_kws={'linestyle': '--'}
            )

            # Customize plot
            ax.set_xlabel('Pathway Map', fontsize=20)
            ax.set_ylabel(f'Semantic Template PC#{comp_no}', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_yticks([-12, -6, 0, 6, 12])
            ax.set_xticks([0, 1])
            ax.set_xlim(-0.5, 1.5)
            plt.title(f'r: {sem_tpl_comp_corr:.4f}, p: {sem_tpl_comp_pval:.4f}', fontsize=20)
            plt.tight_layout()
            plt.show()

            print(f"  Generated dual stream hypothesis scatter plot")

        except Exception as e:
            print(f"  Warning: Could not perform dual stream hypothesis analysis: {e}")
            import traceback
            traceback.print_exc()

        # =====================================================================
        # 11c. Visualize PC loadings (WordNet graph)
        # =====================================================================
        print("* Visualize PC loadings")

        # Get PC loadings
        try:
            # Create loading dataframe
            loading_df = pd.DataFrame(
                np.array(sem_tpl_loading)[:, comp_no-1],
                index=label_list,
                columns=[f'PC#{comp_no}']
            )
            loading_df_sorted = loading_df.sort_values(by=f'PC#{comp_no}', ascending=False)

            # Clean label names for display
            new_label_list_sorted = []
            for label in loading_df_sorted.index:
                if '.' in label:
                    new_label_list_sorted.append(label.split('.')[0])
                else:
                    new_label_list_sorted.append(label)

            # Heatmap visualization
            plt.figure(figsize=(15, 5))
            heatmap_cmap = 'coolwarm'
            sns.heatmap(
                loading_df_sorted.T,
                annot=np.array(new_label_list_sorted).reshape(1, -1),
                fmt='',
                cmap=heatmap_cmap,
                center=0,
                vmin=-0.14,
                vmax=0.14,
                annot_kws={'rotation': 90}
            )
            plt.xticks([])
            plt.tight_layout()
            plt.show()
            print(f"  Generated heatmap for PC{comp_no} loadings")

            # WordNet graph visualization
            print("  Drawing WordNet graph...")

            # Create graph with ENTITY and ACTION nodes
            label_list_with_entity_action = deepcopy(label_list)
            label_list_with_entity_action.extend(['ENTITY', 'ACTION', 'WORDNET'])
            g = node_set_with_entity_action(label_list_with_entity_action)
            node_list = list(g.nodes)

            # Add hypernym edges
            for node in list(g.nodes):
                hyper_edge_with_entity_action(g, node)

            # Create coefficient array for graph nodes
            coef_graph = np.empty(len(node_list))
            for ii, node in enumerate(node_list):
                if node in label_list:
                    coef_graph[ii] = loading_df.loc[node][f'PC#{comp_no}']
                else:
                    coef_graph[ii] = 0

            # Set node colors and sizes
            coef_graph_color = coef_graph
            coef_graph_size = np.ones(len(coef_graph)) * 2500
            coef_graph_df = pd.DataFrame(
                {'color': coef_graph_color, 'size': coef_graph_size},
                index=node_list
            )

            if comp_no == 1:
                abs_max = 0.14
            elif comp_no == 2:
                abs_max = 0.20
            elif comp_no == 3:
                abs_max = 0.20

            # Create clean labels for graph
            new_label_list_with_entity_action = []
            for label in loading_df.index:
                if label in list(g.nodes):
                    if '.' in label:
                        new_label_list_with_entity_action.append(label.split('.')[0])
                    else:
                        new_label_list_with_entity_action.append(label)
            new_label_list_with_entity_action.extend(['ENTITY', 'ACTION', 'WORDNET'])
            coef_graph_df['new_label'] = new_label_list_with_entity_action

            # Draw graph
            fig, ax = plt.subplots(1, 1, figsize=(17, 17))
            ax.set_facecolor('white')
            plt.title(f'PC#{comp_no} - WordNet semantic graph (loadings)', fontsize=20)

            # Calculate layout
            pos = nx.kamada_kawai_layout(g)

            # Draw network
            nx.draw_networkx(
                g,
                pos=pos,
                node_size=coef_graph_df.loc[list(g.nodes)]['size'],
                cmap='coolwarm',
                node_color=coef_graph_df.loc[list(g.nodes)]['color'],
                edge_color='grey',
                vmin=-abs_max,
                vmax=abs_max,
                with_labels=False,
                ax=ax
            )

            # Label top 10% positive and negative nodes
            pos_topn_nodes = coef_graph_df.sort_values(by='color', ascending=False).head(
                int(len(coef_graph_df) * 0.1)
            ).index
            neg_topn_nodes = coef_graph_df.sort_values(by='color', ascending=True).head(
                int(len(coef_graph_df) * 0.1)
            ).index

            for node, coords in pos.items():
                if node in pos_topn_nodes or node in neg_topn_nodes or node in ['ENTITY', 'ACTION', 'WORDNET']:
                    new_label = coef_graph_df.loc[node]['new_label']
                    ax.text(
                        coords[0], coords[1], new_label,
                        fontsize=17,
                        fontfamily='sans-serif',
                        color='black',
                        fontstyle='italic',
                        ha='center',
                        va='center',
                        fontweight='bold'
                    )

            plt.tight_layout()
            plt.show()
            print(f"  Generated WordNet graph for PC{comp_no} loadings")

        except Exception as e:
            print(f"  Warning: Could not create PC loadings visualizations: {e}")

        # =====================================================================
        # 11d. Analyze template loading using semantic features
        # =====================================================================
        print("* Load semantic features")

        try:
            # Load semantic features (BERT-based)
            # Source: https://www.nature.com/articles/s41597-023-01995-6
            sem_feat_df = pd.read_csv(
                PATHS['raw'].parent / 'semantic_features' / 'Main_data' /
                'Estimated_semantic_dimensions_bert_English.csv',
                index_col=0
            )

            # Find matching labels in semantic features
            label_similar_dict = {}
            label_exists_ids = []
            for label in label_list:
                label_name = label.split('.')[0]
                if label_name in sem_feat_df.index:
                    label_similar_dict[label] = label_name
                    label_exists_ids.append(label_list.index(label))
                else:
                    label_similar_dict[label] = None
            label_exists_ids = np.array(label_exists_ids)

            # Create semantic feature dataframe for our labels
            sem_feature_list = list(sem_feat_df.columns)
            sem_feat_label_list_df = pd.DataFrame(
                index=list(np.array(label_list)[label_exists_ids]),
                columns=sem_feature_list,
                dtype=float
            )
            for label in label_list:
                if label_similar_dict[label] is not None:
                    sem_feat_label_list_df.loc[label] = sem_feat_df.loc[label_similar_dict[label]]

            # Remove 'Emotion_abs+1' column if present
            if 'Emotion_abs+1' in sem_feature_list:
                sem_feature_list.remove('Emotion_abs+1')
                sem_feat_df = sem_feat_df.drop('Emotion_abs+1', axis=1)
                sem_feat_label_list_df = sem_feat_label_list_df[sem_feature_list]

            print(f"  Loaded {len(sem_feature_list)} semantic features for {len(label_exists_ids)} labels")

            # =====================================================================
            # Analyze template loading using semantic features
            # =====================================================================
            print("* Analyze the template loading using semantic features")

            from statsmodels.stats.multitest import multipletests

            # Calculate correlation between semantic features and template loading
            sem_feat_corr = sem_feat_label_list_df.corrwith(
                pd.Series(
                    sem_tpl_loading[:, comp_no-1][label_exists_ids],
                    index=sem_feat_label_list_df.index
                )
            )

            # Calculate p-values
            sem_feat_corr_pval = sem_feat_label_list_df.apply(
                lambda x: pearsonr(x, sem_tpl_loading[:, comp_no-1][label_exists_ids])[1]
            )

            # FDR correction
            sem_feat_corr_pval_fdr = multipletests(
                sem_feat_corr_pval, alpha=0.05, method='fdr_bh'
            )[1]

            pos_col, neg_col = 'indianred', 'royalblue'

            # Plot correlation bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            # for i, (corr, pval) in enumerate(zip(sem_feat_corr, sem_feat_corr_pval_fdr)):
            for i, (corr, pval) in enumerate(zip(sem_feat_corr, sem_feat_corr_pval)):  # Use uncorrected p-values for visualization
                if pval < 0.05:
                    if corr > 0:
                        ax.bar(sem_feature_list[i], np.abs(corr), color=pos_col, edgecolor='black')
                    else:
                        ax.bar(sem_feature_list[i], np.abs(corr), color=neg_col, edgecolor='black')
                else:
                    if corr > 0:
                        ax.bar(sem_feature_list[i], np.abs(corr), color=pos_col, edgecolor='black', alpha=0.3)
                    else:
                        ax.bar(sem_feature_list[i], np.abs(corr), color=neg_col, edgecolor='black', alpha=0.3)

            plt.title(f'PC#{comp_no} - Semantic template loading (semantic features)')
            plt.ylabel('Correlation')
            plt.xlabel('Semantic features')
            plt.xticks(rotation=90)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.tick_right()
            ax.tick_params(tick1On=True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            plt.show()
            print(f"  Generated semantic feature correlation bar chart")

            # Create clean label names
            new_label_list_rating = []
            for label in sem_feat_label_list_df.index:
                if '.' in label:
                    new_label_list_rating.append(label.split('.')[0])
                else:
                    new_label_list_rating.append(label)

            # Plot scatter plots for significant features
            sns.set(style="white")
            sig_sem_feat_ids = sem_feat_corr_pval_fdr < 0.05

            for idx in np.where(sig_sem_feat_ids)[0]:
                fig, ax = plt.subplots(figsize=(10, 10))
                if sem_feat_corr.iloc[idx] > 0:
                    sns.regplot(
                        x=sem_feat_label_list_df.iloc[:, idx],
                        y=sem_tpl_loading[:, comp_no-1][label_exists_ids],
                        color=pos_col,
                        scatter_kws={'color': 'gray', 'alpha': 0.8, 's': 600},
                        ax=ax,
                        line_kws={'linewidth': 8}
                    )
                else:
                    sns.regplot(
                        x=sem_feat_label_list_df.iloc[:, idx],
                        y=sem_tpl_loading[:, comp_no-1][label_exists_ids],
                        color=neg_col,
                        scatter_kws={'color': 'gray', 'alpha': 0.8, 's': 600},
                        ax=ax,
                        line_kws={'linewidth': 8}
                    )

                plt.title(
                    f'Correlation: {sem_feat_corr.iloc[idx]:.3f}, '
                    f'p-value: {sem_feat_corr_pval_fdr[idx]:.3f}',
                    fontsize=25
                )
                plt.xlabel(sem_feature_list[idx], fontsize=20)
                plt.ylabel(f'PC#{comp_no} template loading', fontsize=20)
                ax.tick_params(tick1On=False)
                plt.tight_layout()
                plt.show()

            print(f"  Generated {sig_sem_feat_ids.sum()} scatter plots for significant features")

        except Exception as e:
            print(f"  Warning: Could not perform semantic feature analysis: {e}")
            import traceback
            traceback.print_exc()

        # =====================================================================
        # 11e. Visualize significant regions (EG17 lollipop plot)
        # =====================================================================
        print("* Visualize t-statistics and significant regions (EG17)")

        try:
            # Separate positive and negative regions
            sig_regions_pos = sig_regions.copy()
            sig_regions_pos[sig_regions < 0] = 0
            sig_regions_neg = sig_regions.copy()
            sig_regions_neg[sig_regions > 0] = 0

            # Calculate significant regions for EG17 networks
            sig_regions_pos_eg = mmp_2_eg(sig_regions_pos, ATLASES['mmp'], ATLASES['eg'], sum=True)
            sig_regions_neg_eg = mmp_2_eg(sig_regions_neg, ATLASES['mmp'], ATLASES['eg'], sum=True)

            # Create DataFrame for EG17 networks
            sig_regions_pos_eg_df = pd.DataFrame(
                sig_regions_pos_eg,
                index=EG17_LIST,
                columns=[f'PC#{comp_no}']
            )
            sig_regions_neg_eg_df = pd.DataFrame(
                sig_regions_neg_eg,
                index=EG17_LIST,
                columns=[f'PC#{comp_no}']
            )
            sig_regions_pos_eg_df[f'PC#{comp_no}_abs'] = sig_regions_pos_eg_df[f'PC#{comp_no}'].abs()
            sig_regions_neg_eg_df[f'PC#{comp_no}_abs'] = sig_regions_neg_eg_df[f'PC#{comp_no}'].abs()

            # Reorder to sorted list
            sig_regions_pos_eg_df = sig_regions_pos_eg_df.reindex(index=EG17_LIST_SORTED)
            sig_regions_neg_eg_df = sig_regions_neg_eg_df.reindex(index=EG17_LIST_SORTED)

            # Color mapping
            cmax = 1
            cmin = -cmax
            cmap_name = 'Spectral_r'
            sig_regions_pos_eg_df['color'] = sig_regions_pos_eg_df[f'PC#{comp_no}'].apply(
                lambda x: value_to_rgb(x, cmap_name, cmin, cmax)
            )
            sig_regions_neg_eg_df['color'] = sig_regions_neg_eg_df[f'PC#{comp_no}'].apply(
                lambda x: value_to_rgb(x, cmap_name, cmin, cmax)
            )

            # Create lollipop plot for EG17 networks
            fig, ax = plt.subplots(figsize=(12, 5))
            plt.title(f'PC#{comp_no} - Semantic template (EG17 networks)')
            plt.stem(sig_regions_pos_eg_df[f'PC#{comp_no}_abs'], linefmt='silver', markerfmt=' ', basefmt=' ')
            plt.stem(sig_regions_neg_eg_df[f'PC#{comp_no}_abs'], linefmt='silver', markerfmt=' ', basefmt=' ')

            # Plot colored dots for each network
            for i, network in enumerate(EG17_LIST_SORTED):
                plt.plot(
                    i,
                    sig_regions_pos_eg_df.loc[network, f'PC#{comp_no}_abs'],
                    marker='o',
                    color=sig_regions_pos_eg_df.loc[network, 'color'],
                    markersize=24
                )
                plt.plot(
                    i,
                    sig_regions_neg_eg_df.loc[network, f'PC#{comp_no}_abs'],
                    marker='o',
                    color=sig_regions_neg_eg_df.loc[network, 'color'],
                    markersize=24
                )

            # Customize plot appearance
            plt.xticks(np.arange(len(EG17_LIST_SORTED)), EG17_LIST_SORTED, rotation=45, ha='right')
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(tick1On=True)
            plt.tight_layout()
            plt.show()

            print(f"  Generated EG17 lollipop plot for significant regions")

            # Create radar plot for EG17 networks
            categories = EG17_LIST_SORTED
            N = len(categories)

            # Use pre-calculated positive and negative values
            pos_values = sig_regions_pos_eg_df[f'PC#{comp_no}_abs'].values.tolist()
            neg_values = sig_regions_neg_eg_df[f'PC#{comp_no}_abs'].values.tolist()

            # Close the circular plot by repeating first value
            pos_values_c = np.r_[pos_values, pos_values[0]]
            neg_values_c = np.r_[neg_values, neg_values[0]]

            # Calculate angles for radar plot
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles_c = np.r_[angles, angles[0]]

            # Set colors
            pos_color, neg_color = plt.cm.Spectral_r(0.99), plt.cm.Spectral_r(0.01)

            fig = plt.figure(figsize=(12, 12))
            ax = plt.subplot(111, polar=True)

            # Plot line, markers, and fill
            ax.vlines(angles, 0, pos_values, linewidth=3.0, color=pos_color, alpha=0.9)
            ax.vlines(angles, 0, neg_values, linewidth=3.0, color=neg_color, alpha=0.9)
            ax.fill(angles_c, pos_values_c, color=pos_color, alpha=0.4, zorder=2)
            ax.fill(angles_c, neg_values_c, color=neg_color, alpha=0.4, zorder=2)
            ax.scatter(angles, pos_values, s=750, color=pos_color, zorder=10, clip_on=False)
            ax.scatter(angles, neg_values, s=750, color=neg_color, zorder=10, clip_on=False)

            # Black scatter in the center
            ax.scatter(0, 0, s=1000, color='black', zorder=11, clip_on=False)

            # Hide default labels
            ax.set_xticks(angles)
            ax.set_xticklabels([])

            # Set y-ticks
            max_value = 9
            yticks = [0, 3, 6, 9]
            plt.yticks(yticks, [])
            plt.ylim(0, max_value)

            plt.tight_layout()
            plt.show()

            print(f"  Generated EG17 radar plot for significant regions")

        except Exception as e:
            print(f"  Warning: Could not create EG17 visualizations: {e}")

        # =====================================================================
        # 11f. Brain surface visualization for significant regions
        # =====================================================================
        print("* Visualize significant regions on brain surface")

        try:
            # Map significant regions to full 32k surface (combined, not separated)
            sig_regions_full = map_to_labels(
                sig_regions.astype('float32'),
                ATLASES['mmp'],
                mask=ATLASES['mmp'] != 0,
                fill=np.nan
            )

            # Load surface meshes
            mmp_folder = PATHS['tpl'] / 'MMP'
            surfs = [None] * 2
            surfs[0] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.L.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
            surf_lh = serial_connect(surfs[0], nf)
            surfs[1] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.R.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
            surf_rh = serial_connect(surfs[1], nf)

            # Plot t-statistics
            tstat_results_32k = map_to_labels(
                tstat_results,
                ATLASES['mmp'],
                mask=ATLASES['mmp'] != 0,
                fill=np.nan
            )

            cmax = 3.5
            cmin = -cmax
            cmap = 'Spectral_r'

            plot_hemispheres(
                surf_lh,
                surf_rh,
                array_name=tstat_results_32k,
                size=(1200, 200),
                cmap=cmap,
                nan_color=(0.5, 0.5, 0.5, 1),
                color_bar=True,
                color_range=(cmin, cmax),
                zoom=1.6,
            )

            # Plot combined significant regions
            cmax = 1
            cmin = -cmax
            cmap = 'Spectral_r'

            plot_hemispheres(
                surf_lh,
                surf_rh,
                array_name=sig_regions_full,
                size=(1200, 200),
                cmap=cmap,
                color_bar=True,
                color_range=(cmin, cmax),
                zoom=1.6,
            )

            # Save CIFTI file for significant regions
            roiarray_2_32knii(
                sig_regions.astype('float32'),
                perf_mask,
                atlas='mmp',
                output_folder='brain',
                output_file=f'sig_regions_PC{comp_no}.dscalar.nii',
                fig_path=PATHS['fig'],
                comp_no=comp_no
            )

            print(f"  Saved CIFTI file for significant regions")

            # =================================================================
            # Save t-statistics with mask based on non-significant regions as dlabel.nii
            # Non-significant regions are blended toward white for visual distinction
            # =================================================================
            print("* Save t-statistics with significance mask as dlabel.nii")

            from matplotlib import cm

            # Load dummy dlabel for structure
            dummy_dlabel_path = PATHS['tpl'] / 'MMP' / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
            dummy_dlabel = nib.load(str(dummy_dlabel_path))
            dummy_32k_nonmed = dummy_dlabel.get_fdata().copy().astype(int)
            label_color_dict = dummy_dlabel.header.get_axis(0).label
            new_label_color_dict = deepcopy(label_color_dict)

            # Colormap settings for t-statistics
            tstat_cmap = cm.get_cmap('Spectral_r')
            tstat_cmax, tstat_cmin = 3.5, -3.5
            tstat_norm = plt.Normalize(vmin=tstat_cmin, vmax=tstat_cmax)

            # Blending strength (0.35 = 35% blend toward white/black)
            strength = 0.35

            for mmp_idx in range(360):
                # Get t-statistic color
                rgba_tuple = tstat_cmap(tstat_norm(tstat_results[mmp_idx]))[:3]

                if sig_regions[mmp_idx] == 0:
                    # Non-significant: blend toward white for visual distinction
                    new_rgba_tuple = tuple([(1 - strength) * x + strength * 1 for x in rgba_tuple]) + (1.0,)
                else:
                    # Significant: blend toward black for emphasis
                    new_rgba_tuple = tuple([(1 - strength) * x + strength * 0 for x in rgba_tuple]) + (1.0,)

                new_label_color_dict[0][mmp_idx + 1] = (new_label_color_dict[0][mmp_idx + 1][0], new_rgba_tuple)

            # Create new CIFTI image with updated colors
            label_axis = dummy_dlabel.header.get_axis(0)
            label_axis.label = new_label_color_dict
            new_header = nib.Cifti2Header.from_axes((label_axis, dummy_dlabel.header.get_axis(1)))
            dlabel_tstat_masked_cifti = nib.Cifti2Image(dummy_32k_nonmed, header=new_header)

            # Save dlabel file
            tstat_masked_dlabel_path = PATHS['fig'] / 'brain' / f'tstat_PC{comp_no}_sig_masked.dlabel.nii'
            tstat_masked_dlabel_path.parent.mkdir(parents=True, exist_ok=True)
            dlabel_tstat_masked_cifti.to_filename(str(tstat_masked_dlabel_path))
            print(f"  Saved t-statistics with significance mask: {tstat_masked_dlabel_path.name}")

        except Exception as e:
            print(f"  Warning: Could not create brain surface visualizations: {e}")
            import traceback
            traceback.print_exc()

        # =====================================================================
        # 11g. Parallel plot and histogram
        # =====================================================================
        print("* Compute average semantic spaces for parallel plot and histogram")

        try:
            # Compute average semantic spaces for TD and ASD
            comp_no = CONFIG['comp_no']
            sem_avg_td_comp = np.zeros(360).astype(np.float32)
            sem_avg_asd_comp = np.zeros(360).astype(np.float32)
            sem_avg_td_comp[perf_mask] = np.array(sem_list_td).mean(0)[:, comp_no-1]
            sem_avg_asd_comp[perf_mask] = np.array(sem_list_asd).mean(0)[:, comp_no-1]

            # Extract significant regions only
            sem_avg_td_comp_sig = sem_avg_td_comp[perf_mask]
            sem_avg_asd_comp_sig = sem_avg_asd_comp[perf_mask]
            sig_regions_sig = sig_regions[perf_mask]

            # Load MMP info
            mmp_info_path = PATHS['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
            mmp_info_df = pd.read_excel(str(mmp_info_path), engine='openpyxl')
            mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]
            mmp_info_df_sig = mmp_info_df.iloc[perf_mask].reset_index(drop=True)

            # Get mappings for EG17
            mapping_mmp_2_eg17 = np.zeros(360)
            for mmp_no in range(360):
                mmp_ids = np.where(ATLASES['mmp'] == mmp_no + 1)[0]
                eg_atlas_mmp_ids = ATLASES['eg'][mmp_ids]
                eg_atlas_mmp_ids_unique, eg_atlas_mmp_ids_count = \
                    np.unique(eg_atlas_mmp_ids, return_counts=True)
                mapping_mmp_2_eg17[mmp_no] = \
                    eg_atlas_mmp_ids_unique[np.argmax(eg_atlas_mmp_ids_count)]
            mapping_mmp_2_eg17_sig = mapping_mmp_2_eg17[perf_mask]

            print(f"  Computed average semantic spaces for {len(sem_avg_td_comp_sig)} significant regions")

            # =================================================================
            # Parallel plot
            # =================================================================
            print("* Draw parallel plot between TD and ASD semantic spaces")

            from matplotlib.collections import LineCollection

            # Define triple network colors
            dmn_col, sal_col, cen_col = '#F0BE6F', '#1B4543', '#B3D0C6'
            dmn_no_list = [idx+1 for idx, network in enumerate(EG17_LIST)
                          if 'Default' in network or 'Language' in network or 'Context' in network]
            sal_no_list = [idx+1 for idx, network in enumerate(EG17_LIST)
                          if 'Salience' in network]
            cen_no_list = [idx+1 for idx, network in enumerate(EG17_LIST)
                          if 'CingOperc' in network or 'FrontPar' in network]

            # Normalize using shared min/max across both groups
            all_values = np.concatenate([sem_avg_td_comp_sig, sem_avg_asd_comp_sig])
            vmax = 4.5
            vmin = -vmax
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_parallel = plt.get_cmap('coolwarm')

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 4))

            # Define y-coordinates for the two parallel horizontal lines
            y_positions = [4, 1]
            labels = ['TD', 'ASD']

            # Draw horizontal lines
            line_length = max(all_values) + 0.1
            ax.hlines(y_positions[0], 0, line_length, colors='lightgray',
                     linestyles='--', alpha=0.5)
            ax.hlines(y_positions[1], 0, line_length, colors='lightgray',
                     linestyles='--', alpha=0.5)

            # Plot gradient lines between corresponding TD-ASD points
            for i, (td_val, asd_val) in enumerate(zip(sem_avg_td_comp_sig, sem_avg_asd_comp_sig)):
                x_vals = np.linspace(td_val, asd_val, 500)  # high resolution for smooth gradient
                y_vals = np.linspace(y_positions[0], y_positions[1], 500)
                points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Check for significance
                if sig_regions_sig[i] != 0:
                    lw = 5
                    a = 1
                else:
                    lw = 1
                    a = 0.1

                lc = LineCollection(
                    segments,
                    cmap=cmap_parallel,
                    norm=norm,
                    capstyle='round',
                    linewidths=lw,
                    alpha=a
                )
                lc.set_array(x_vals)
                ax.add_collection(lc)

            # Plot TD scatter with colormap
            ax.scatter(
                sem_avg_td_comp_sig,
                [y_positions[0]] * len(sem_avg_td_comp_sig),
                c=cmap_parallel(norm(sem_avg_td_comp_sig)),
                alpha=0.5, s=400, label='TD',
                zorder=3
            )

            # Plot ASD scatter with colormap
            ax.scatter(
                sem_avg_asd_comp_sig,
                [y_positions[1]] * len(sem_avg_asd_comp_sig),
                c=cmap_parallel(norm(sem_avg_asd_comp_sig)),
                alpha=0.5, s=400, label='ASD',
                zorder=3
            )

            # Add outline for triple network regions
            dmn_val_dict = {'TD': [], 'ASD': []}
            sal_val_dict = {'TD': [], 'ASD': []}
            cen_val_dict = {'TD': [], 'ASD': []}

            for i, (td_val, asd_val) in enumerate(zip(sem_avg_td_comp_sig, sem_avg_asd_comp_sig)):
                if sig_regions_sig[i] != 0:
                    if mapping_mmp_2_eg17_sig[i] in dmn_no_list:
                        ax.scatter(td_val, y_positions[0], facecolors='none',
                                 edgecolors=dmn_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        ax.scatter(asd_val, y_positions[1], facecolors='none',
                                 edgecolors=dmn_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        dmn_val_dict['TD'].append(td_val)
                        dmn_val_dict['ASD'].append(asd_val)
                    elif mapping_mmp_2_eg17_sig[i] in sal_no_list:
                        ax.scatter(td_val, y_positions[0], facecolors='none',
                                 edgecolors=sal_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        ax.scatter(asd_val, y_positions[1], facecolors='none',
                                 edgecolors=sal_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        sal_val_dict['TD'].append(td_val)
                        sal_val_dict['ASD'].append(asd_val)
                    elif mapping_mmp_2_eg17_sig[i] in cen_no_list:
                        ax.scatter(td_val, y_positions[0], facecolors='none',
                                 edgecolors=cen_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        ax.scatter(asd_val, y_positions[1], facecolors='none',
                                 edgecolors=cen_col, alpha=0.9, s=500, linewidths=3, zorder=4)
                        cen_val_dict['TD'].append(td_val)
                        cen_val_dict['ASD'].append(asd_val)

            # Draw 'v' marks at the average value for each network
            if dmn_val_dict['TD']:
                ax.scatter(np.mean(dmn_val_dict['TD']), y_positions[0]+0.1,
                         facecolors='none', edgecolors=dmn_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                ax.scatter(np.mean(dmn_val_dict['ASD']), y_positions[1]-0.1,
                         facecolors='none', edgecolors=dmn_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                print(f'  Average value of DMN: {np.mean(dmn_val_dict["TD"])}')
                print(f'  Average value of DMN: {np.mean(dmn_val_dict["ASD"])}')

            if sal_val_dict['TD']:
                ax.scatter(np.mean(sal_val_dict['TD']), y_positions[0]+0.1,
                         facecolors='none', edgecolors=sal_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                ax.scatter(np.mean(sal_val_dict['ASD']), y_positions[1]-0.1,
                         facecolors='none', edgecolors=sal_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                print(f'  Average value of SAL: {np.mean(sal_val_dict["TD"])}')
                print(f'  Average value of SAL: {np.mean(sal_val_dict["ASD"])}')

            if cen_val_dict['TD']:
                ax.scatter(np.mean(cen_val_dict['TD']), y_positions[0]+0.1,
                         facecolors='none', edgecolors=cen_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                ax.scatter(np.mean(cen_val_dict['ASD']), y_positions[1]-0.1,
                         facecolors='none', edgecolors=cen_col, alpha=0.9,
                         s=500, linewidths=3, zorder=4, marker='v')
                print(f'  Average value of CEN: {np.mean(cen_val_dict["TD"])}')
                print(f'  Average value of CEN: {np.mean(cen_val_dict["ASD"])}')

            # Axis settings
            x_min = min(all_values) - 0.5
            x_max = max(all_values) + 0.5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.5, 4.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels)
            ax.set_xlabel(f'PC#{comp_no} Semantic Score')
            ax.set_title(f'Parallel Plot: TD vs ASD (PC#{comp_no})')
            ax.set_xticks(np.linspace(x_min, x_max, 6))

            # Remove grid and outlines
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            plt.tight_layout()
            plt.show()

            print(f"  Generated parallel plot for PC#{comp_no}")

            # =================================================================
            # Histogram plot
            # =================================================================
            print("* Draw histogram of TD and ASD semantic spaces")

            from scipy.stats import gaussian_kde

            # Shared x-range/grid (same as before for fair comparison)
            all_values_full = np.concatenate([sem_avg_td_comp, sem_avg_asd_comp])
            x_min_hist = all_values_full.min() - 0.5
            x_max_hist = all_values_full.max() + 0.5
            x_hist = np.linspace(x_min_hist, x_max_hist, 512)

            # Build KDEs first to lock the shared y-axis
            kde_td = gaussian_kde(sem_avg_td_comp)
            kde_asd = gaussian_kde(sem_avg_asd_comp)

            y_td = kde_td(x_hist)
            y_asd = kde_asd(x_hist)

            y_max_hist = max(y_td.max(), y_asd.max()) * 1.1  # 10% headroom

            def plot_kde_with_gradient(x, y, label, cmap, norm):
                """Plot KDE with gradient fill."""
                fig, ax = plt.subplots(figsize=(12, 3))

                # Gradient fill under the curve
                for i in range(len(x) - 1):
                    x0, x1 = x[i], x[i+1]
                    y0, y1 = y[i], y[i+1]
                    xc = 0.5 * (x0 + x1)
                    ax.fill_between(
                        [x0, x1], [y0, y1], 0,
                        facecolor=cmap(norm(xc)), edgecolor='none'
                    )

                # Gradient-colored line on top for a crisp edge
                pts = np.array([x, y]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2, zorder=3)
                lc.set_array(x)  # color by x using same colormap
                ax.add_collection(lc)

                # Shared axes
                ax.set_xlim(x_min_hist, x_max_hist)
                ax.set_ylim(0, y_max_hist)

                ax.set_xlabel(f"PC#{comp_no} Semantic Score")
                ax.set_ylabel("Density")
                ax.set_title(f"KDE of PC#{comp_no} Scores – {label}")
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                # no x-ticks
                ax.set_xticks([])
                plt.tight_layout()
                plt.show()

            # Plot TD and ASD in separate figures with shared axes
            plot_kde_with_gradient(x_hist, y_td, "TD", cmap_parallel, norm)
            plot_kde_with_gradient(x_hist, y_asd, "ASD", cmap_parallel, norm)

            print(f"  Generated histogram plots for PC#{comp_no}")

        except Exception as e:
            print(f"  Warning: Could not create parallel plot and histogram: {e}")
            import traceback
            traceback.print_exc()

        # =================================================================
        # 11h. Analyze t-statistics using expanded dual stream hypothesis
        # =================================================================
        print("\n" + "=" * 80)
        print("11h. Analyze t-statistics using expanded dual stream hypothesis")
        print("=" * 80)

        try:
            from copy import deepcopy

            # Load MMP info
            mmp_info_df = pd.read_excel(
                str(PATHS['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'),
                engine='openpyxl'
            )
            mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]  # remove indices after 360
            mmp_roi_list = mmp_info_df['ROI'].tolist()

            # Expanded ROI list for each pathway
            ventral_roi_list_expanded = {
                'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
                'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'L_V4_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_V4_ROI'],
                'Ventral Occipitotemporal': [
                    'L_FFC_ROI', 'L_PIT_ROI', 'L_V8_ROI', 'L_VMV1_ROI', 'L_VMV2_ROI', 'L_VMV3_ROI', 'L_VVC_ROI',
                    'R_FFC_ROI', 'R_PIT_ROI', 'R_V8_ROI', 'R_VMV1_ROI', 'R_VMV2_ROI', 'R_VMV3_ROI', 'R_VVC_ROI'],
                'Inferior Temporal': [
                    'L_PHT_ROI', 'L_TE1a_ROI', 'L_TE1m_ROI', 'L_TE1p_ROI', 'L_TE2a_ROI', 'L_TE2p_ROI',
                    'R_PHT_ROI', 'R_TE1a_ROI', 'R_TE1m_ROI', 'R_TE1p_ROI', 'R_TE2a_ROI', 'R_TE2p_ROI'],
                'Temporal Pole': [
                    'L_TGv_ROI', 'L_TGd_ROI',
                    'R_TGv_ROI', 'R_TGd_ROI'],
                'Oribital & Polar Frontal': [
                    'L_a47r_ROI', 'L_47m_ROI', 'L_47s_ROI', 'L_10p_ROI', 'L_11l_ROI', 'L_13l_ROI', 'L_OFC_ROI', 'L_pOFC_ROI', 'L_10pp_ROI', 'L_10d_ROI',
                    'R_a47r_ROI', 'R_47m_ROI', 'R_47s_ROI', 'R_10p_ROI', 'R_11l_ROI', 'R_13l_ROI', 'R_OFC_ROI', 'R_pOFC_ROI', 'R_10pp_ROI', 'R_10d_ROI']}
            ventral_roi_idx_list_expanded = {
                key: [i for i, roi in enumerate(mmp_roi_list) if roi in ventral_roi_list_expanded[key]]
                for key in ventral_roi_list_expanded.keys()}

            dorsal_roi_list_expanded = {
                'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
                'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'L_V3A_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_V3A_ROI'],
                'MT+ complex': [
                    'L_LO1_ROI','L_LO2_ROI','L_LO3_ROI','L_MST_ROI','L_MT_ROI','L_V3CD_ROI','L_V4t_ROI',
                    'R_LO1_ROI','R_LO2_ROI','R_LO3_ROI','R_MST_ROI','R_MT_ROI','R_V3CD_ROI','R_V4t_ROI'],
                'Intraparietal': [
                    'L_AIP_ROI','L_LIPd_ROI','L_LIPv_ROI','L_MIP_ROI','L_VIP_ROI','L_IP0_ROI','L_IP1_ROI','L_IP2_ROI',
                    'R_AIP_ROI','R_LIPd_ROI','R_LIPv_ROI','R_MIP_ROI','R_VIP_ROI','R_IP0_ROI','R_IP1_ROI','R_IP2_ROI'],
                'Area 7':[
                    'L_7AL_ROI','L_7Am_ROI','L_7PC_ROI','L_7PL_ROI','L_7Pm_ROI',
                    'R_7AL_ROI','R_7Am_ROI','R_7PC_ROI','R_7PL_ROI','R_7Pm_ROI'],
                'Premotor & Eye Fields': [
                    'L_55b_ROI','L_6a_ROI','L_6d_ROI','L_FEF_ROI',
                    'R_55b_ROI','R_6a_ROI','R_6d_ROI','R_FEF_ROI'],
                'Dorsolateral Prefrontal': [
                    'L_46_ROI','L_8Ad_ROI','L_8Av_ROI','L_8BL_ROI','L_8C_ROI','L_9-46d_ROI','L_9a_ROI','L_9p_ROI','L_a9-46v_ROI','L_i6-8_ROI','L_p9-46v_ROI','L_s6-8_ROI', 'L_SFL_ROI',
                    'R_46_ROI','R_8Ad_ROI','R_8Av_ROI','R_8BL_ROI','R_8C_ROI','R_9-46d_ROI','R_9a_ROI','R_9p_ROI','R_a9-46v_ROI','R_i6-8_ROI','R_p9-46v_ROI','R_s6-8_ROI', 'R_SFL_ROI'],
                'Precuneus & RSC': [
                    'L_31a_ROI','L_31pd_ROI','L_31pv_ROI','L_PCV_ROI','L_7m_ROI','L_d23ab_ROI','L_v23ab_ROI','L_RSC_ROI','L_POS1_ROI','L_POS2_ROI','L_DVT_ROI','L_ProS_ROI',
                    'R_31a_ROI','R_31pd_ROI','R_31pv_ROI','R_PCV_ROI','R_7m_ROI','R_d23ab_ROI','R_v23ab_ROI','R_RSC_ROI','R_POS1_ROI','R_POS2_ROI','R_DVT_ROI','R_ProS_ROI']}
            dorsal_roi_idx_list_expanded = {
                key: [i for i, roi in enumerate(mmp_roi_list) if roi in dorsal_roi_list_expanded[key]]
                for key in dorsal_roi_list_expanded.keys()}

            social_roi_list_expanded = {
                'Primary Visual': ['L_V1_ROI', 'R_V1_ROI'],
                'Early Visual': ['L_V2_ROI', 'L_V3_ROI', 'R_V2_ROI', 'R_V3_ROI'],
                'MT+ complex': [
                    'L_LO1_ROI','L_LO2_ROI','L_LO3_ROI','L_V4t_ROI','L_MT_ROI','L_MST_ROI','L_FST_ROI',
                    'R_LO1_ROI','R_LO2_ROI','R_LO3_ROI','R_V4t_ROI','R_MT_ROI','R_MST_ROI','R_FST_ROI'],
                'Superior Temporal Sulcus': [
                    'L_STSda_ROI','L_STSdp_ROI','L_STSva_ROI','L_STSvp_ROI',
                    'R_STSda_ROI','R_STSdp_ROI','R_STSva_ROI','R_STSvp_ROI'],
                'Temporal-Parietal-Occipital Junction': [
                    'L_TPOJ1_ROI','L_TPOJ2_ROI','L_TPOJ3_ROI','L_STV_ROI','L_PSL_ROI',
                    'R_TPOJ1_ROI','R_TPOJ2_ROI','R_TPOJ3_ROI','R_STV_ROI','R_PSL_ROI'],
                'Precuneus': [
                    'L_31a_ROI','L_31pd_ROI','L_31pv_ROI','L_PCV_ROI','L_7m_ROI','L_d23ab_ROI','L_v23ab_ROI',
                    'R_31a_ROI','R_31pd_ROI','R_31pv_ROI','R_PCV_ROI','R_7m_ROI','R_d23ab_ROI','R_v23ab_ROI'],
                'Inferior Frontal & Medial Prefrontal': [
                    'L_44_ROI','L_45_ROI','L_IFJa_ROI','L_IFSp_ROI','L_IFSa_ROI','L_p47r_ROI','L_47l_ROI','L_8bm_ROI','L_9m_ROI','L_d32_ROI','L_a24_ROI','L_p24_ROI','L_a32pr_ROI',
                    'R_44_ROI','R_45_ROI','R_IFJa_ROI','R_IFSp_ROI','R_IFSa_ROI','R_p47r_ROI','R_47l_ROI','R_8bm_ROI','R_9m_ROI','R_d32_ROI','R_a24_ROI','R_p24_ROI','R_a32pr_ROI']}
            social_roi_idx_list_expanded = {
                key: [i for i, roi in enumerate(mmp_roi_list) if roi in social_roi_list_expanded[key]]
                for key in social_roi_list_expanded.keys()}

            # Conditionally exclude expanded networks based on config
            if not CONFIG['include_expanded_networks']:
                # Remove expanded networks from each pathway
                ventral_expanded_keys = ['Oribital & Polar Frontal']
                dorsal_expanded_keys = ['Dorsolateral Prefrontal', 'Precuneus & RSC']
                social_expanded_keys = ['Precuneus', 'Inferior Frontal & Medial Prefrontal']

                for key in ventral_expanded_keys:
                    if key in ventral_roi_list_expanded:
                        del ventral_roi_list_expanded[key]
                    if key in ventral_roi_idx_list_expanded:
                        del ventral_roi_idx_list_expanded[key]

                for key in dorsal_expanded_keys:
                    if key in dorsal_roi_list_expanded:
                        del dorsal_roi_list_expanded[key]
                    if key in dorsal_roi_idx_list_expanded:
                        del dorsal_roi_idx_list_expanded[key]

                for key in social_expanded_keys:
                    if key in social_roi_list_expanded:
                        del social_roi_list_expanded[key]
                    if key in social_roi_idx_list_expanded:
                        del social_roi_idx_list_expanded[key]

                print("  Defined core ROI lists for ventral, dorsal, and social pathways (expanded networks excluded)")
            else:
                print("  Defined expanded ROI lists for ventral, dorsal, and social pathways")

            # Save brain map for each pathway as *.dlabel.nii file
            pathway_list = ['ventral', 'dorsal', 'social']
            ventral_rgba = (154/255, 89/255, 181/255, 255/255)
            dorsal_rgba = (45/255, 203/255, 111/255, 255/255)
            social_rgba = (196/255, 25/255, 9/255, 255/255)
            others_rgba = (222/255, 222/255, 222/255, 255/255)
            pathway_color_dict = {'ventral': ventral_rgba, 'dorsal': dorsal_rgba, 'social': social_rgba}

            # Plotting option #2: gradient color within each pathway (lighten the color for later groups)
            MAX_WHITEN_255 = 150  # how much to lighten the last group
            MAX_WHITEN = MAX_WHITEN_255 / 255.0

            # Use full pathway dictionaries for dlabel generation (include all networks like Primary Visual)
            pathway_group_dict = {
                'ventral': ventral_roi_list_expanded,
                'dorsal': dorsal_roi_list_expanded,
                'social': social_roi_list_expanded,}

            # Load dummy dlabel for structure
            dummy_dlabel = nib.load(
                str(PATHS['tpl'] / 'MMP' / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')
            )
            dummy_32k_nonmed = dummy_dlabel.get_fdata().copy().astype(int)
            label_color_dict = dummy_dlabel.header.get_axis(0).label

            for pathway in pathway_list:
                base_rgba = pathway_color_dict[pathway]   # (r,g,b,a) in 0-1
                base_rgb = base_rgba[:3]
                base_a   = base_rgba[3]                   # keep alpha constant
                groups = list(pathway_group_dict[pathway].values())  # ordered groups
                n_groups = len(groups)
                # step so that last group hits MAX_WHITEN
                denom = max(1, n_groups - 1)              # avoid div-by-zero if 1 group
                w_step = MAX_WHITEN / denom               # 0, step, 2*step, ... up to MAX_WHITEN
                # build ROI -> whiten weight
                roi_to_whiten = {}
                for g_idx, roi_group in enumerate(groups):
                    w = w_step * g_idx                    # 0 … MAX_WHITEN
                    for roi in roi_group:
                        roi_to_whiten[roi] = w
                new_label_color_dict = deepcopy(label_color_dict)
                for mmp_idx in range(360):
                    roi_name = new_label_color_dict[0][mmp_idx+1][0]
                    if roi_name in roi_to_whiten:
                        w = roi_to_whiten[roi_name]
                        # blend toward white
                        r = (1 - w) * base_rgb[0] + w * 1.0
                        g = (1 - w) * base_rgb[1] + w * 1.0
                        b = (1 - w) * base_rgb[2] + w * 1.0
                        rgba = (r, g, b, base_a)
                    else:
                        rgba = others_rgba  # constant gray
                    new_label_color_dict[0][mmp_idx+1] = (roi_name, rgba)
                label_axis = dummy_dlabel.header.get_axis(0)
                label_axis.label = new_label_color_dict
                new_header = nib.Cifti2Header.from_axes((label_axis, dummy_dlabel.header.get_axis(1)))
                dlabel_32k_nonmed_cifti = nib.Cifti2Image(dummy_32k_nonmed, header=new_header)

                # Save dlabel file
                expanded_suffix = 'expanded' if CONFIG['include_expanded_networks'] else 'core'
                out_dlabel_path = PATHS['fig'] / 'brain' / f'visual_pathways_{expanded_suffix}_{pathway}_pc{comp_no}.dlabel.nii'
                out_dlabel_path.parent.mkdir(parents=True, exist_ok=True)
                dlabel_32k_nonmed_cifti.to_filename(str(out_dlabel_path))
                print(f"  Saved dlabel file for {pathway} pathway: {out_dlabel_path.name}")

            # Draw boxplot for each pathway (1x3 subplots)
            def _values_for_groups(tstats, roi_idx_dict):
                """
                Given a t-statistic array and a dictionary mapping group names to ROI indices,
                return a dictionary mapping group names to arrays of absolute t-statistic values.
                """
                groups = {}
                for k, idxs in roi_idx_dict.items():
                    # delete indices if not included in perf_mask
                    idxs_sig = np.array([idx for idx in idxs if idx in perf_mask])
                    vals = np.abs(tstats[np.array(idxs_sig, dtype=int)])
                    vals = vals[np.isfinite(vals)]  # drop NaNs/Infs just in case
                    groups[k] = vals
                return groups

            # Get tstat_results masked by perf_mask
            ventral_groups = _values_for_groups(tstat_results, ventral_roi_idx_list_expanded)
            dorsal_groups  = _values_for_groups(tstat_results, dorsal_roi_idx_list_expanded)
            social_groups  = _values_for_groups(tstat_results, social_roi_idx_list_expanded)

            # Filter out groups with no data (e.g., Primary Visual not in perf_mask)
            ventral_groups = {k: v for k, v in ventral_groups.items() if len(v) > 0}
            dorsal_groups = {k: v for k, v in dorsal_groups.items() if len(v) > 0}
            social_groups = {k: v for k, v in social_groups.items() if len(v) > 0}

            ventral_order = list(ventral_groups.keys())
            dorsal_order  = list(dorsal_groups.keys())
            social_order  = list(social_groups.keys())

            # -----------------------------------------------------------------
            # Statistically test whether |t-statistic| is associated with the
            # hierarchical level (ordinal position along the pathway). Each
            # retained network becomes a level 0, 1, 2, ... in pathway order,
            # and every ROI in that network contributes one |t| observation.
            # -----------------------------------------------------------------
            from scipy.stats import spearmanr, kendalltau, linregress, norm as _norm
            import itertools as _itertools

            def _jonckheere_terpstra(samples_ordered):
                """One-sided (increasing) Jonckheere-Terpstra trend test.

                Uses the normal approximation without a tie correction (|t|
                values are continuous, so ties are negligible). Returns the
                J statistic, its z-score, and increasing/two-sided p-values.
                """
                J = 0.0
                for i, j in _itertools.combinations(range(len(samples_ordered)), 2):
                    xi, xj = samples_ordered[i], samples_ordered[j]
                    for a in xi:
                        J += np.count_nonzero(xj > a) + 0.5 * np.count_nonzero(xj == a)
                ns = [len(s) for s in samples_ordered]
                N = sum(ns)
                mean_J = (N ** 2 - sum(n ** 2 for n in ns)) / 4.0
                var_J = (N ** 2 * (2 * N + 3) - sum(n ** 2 * (2 * n + 3) for n in ns)) / 72.0
                z = (J - mean_J) / np.sqrt(var_J) if var_J > 0 else np.nan
                return {'J': J, 'z': z,
                        'p_increasing': float(_norm.sf(z)),
                        'p_two_sided': float(2 * _norm.sf(abs(z)))}

            def _hierarchy_stats(groups_dict, order):
                """Association between pathway hierarchy level and |t-statistic|.

                ROI-level tests use every ROI as an observation (level index vs
                |t|); the group-mean Spearman is a conservative check that treats
                each network as a single point and is insensitive to the number
                of ROIs (and to non-independence of L/R homotopic ROIs).
                """
                levels, vals = [], []
                for lvl, k in enumerate(order):
                    v = np.asarray(groups_dict[k], dtype=float)
                    levels.extend([lvl] * v.size)
                    vals.extend(v.tolist())
                levels = np.asarray(levels, dtype=float)
                vals = np.asarray(vals, dtype=float)
                res = {'n_levels': len(order), 'n_rois': int(vals.size)}
                if len(order) >= 2 and vals.size >= 3:
                    rho, p_rho = spearmanr(levels, vals)
                    tau, p_tau = kendalltau(levels, vals)
                    lr = linregress(levels, vals)
                    jt = _jonckheere_terpstra([np.asarray(groups_dict[k], dtype=float) for k in order])
                    gmeans = np.array([np.mean(groups_dict[k]) for k in order])
                    rho_m, p_rho_m = spearmanr(np.arange(len(order)), gmeans)
                    res.update({
                        'spearman_rho': float(rho), 'spearman_p': float(p_rho),
                        'kendall_tau': float(tau), 'kendall_p': float(p_tau),
                        'ols_slope': float(lr.slope), 'ols_p': float(lr.pvalue), 'ols_r': float(lr.rvalue),
                        'jt_z': float(jt['z']), 'jt_p_increasing': jt['p_increasing'], 'jt_p_two_sided': jt['p_two_sided'],
                        'groupmean_spearman_rho': float(rho_m), 'groupmean_spearman_p': float(p_rho_m),
                    })
                return res

            pathway_stats = {
                'ventral': _hierarchy_stats(ventral_groups, ventral_order),
                'dorsal':  _hierarchy_stats(dorsal_groups, dorsal_order),
                'social':  _hierarchy_stats(social_groups, social_order),
            }

            print("\n  --- Hierarchical level vs |t-statistic| association ---")
            for pw in pathway_list:
                s = pathway_stats[pw]
                print(f"  [{pw}] {s['n_levels']} levels, {s['n_rois']} ROIs")
                if 'spearman_rho' in s:
                    print(f"      Spearman rho = {s['spearman_rho']:+.3f} (p = {s['spearman_p']:.3g})")
                    print(f"      Kendall  tau = {s['kendall_tau']:+.3f} (p = {s['kendall_p']:.3g})")
                    print(f"      OLS slope    = {s['ols_slope']:+.3f} |t|/level (p = {s['ols_p']:.3g}, r = {s['ols_r']:+.3f})")
                    print(f"      Jonckheere-Terpstra z = {s['jt_z']:+.3f} (p_increasing = {s['jt_p_increasing']:.3g})")
                    print(f"      Group-mean Spearman rho = {s['groupmean_spearman_rho']:+.3f} (p = {s['groupmean_spearman_p']:.3g})")
                else:
                    print("      Not enough levels/ROIs for a trend test.")

            # Save the trend statistics to CSV for reporting
            expanded_suffix = 'expanded' if CONFIG['include_expanded_networks'] else 'core'
            stats_rows = [{'pathway': pw, 'comp_no': comp_no, **pathway_stats[pw]} for pw in pathway_list]
            hierarchy_stats_path = PATHS['fig'] / 'brain' / f'visual_pathways_{expanded_suffix}_hierarchy_stats_pc{comp_no}.csv'
            hierarchy_stats_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(stats_rows).to_csv(hierarchy_stats_path, index=False)
            print(f"  Saved hierarchy trend stats: {hierarchy_stats_path.name}")

            fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
            rng = np.random.default_rng(0)
            pathway_label = "expanded" if CONFIG['include_expanded_networks'] else "core"
            triples = [
                (axes[0], ventral_groups, ventral_order, f"Ventral ({pathway_label})", "ventral"),
                (axes[1], dorsal_groups, dorsal_order, f"Dorsal ({pathway_label})", "dorsal"),
                (axes[2], social_groups, social_order, f"Social ({pathway_label})", "social")
            ]

            for ax, groups, order, title, pathway in triples:
                data = [groups[k] for k in order]
                means = [np.mean(vals) if len(vals) > 0 else np.nan for vals in data]
                ax.bar(range(1, len(order)+1), means, color='lightgray', edgecolor='black', alpha=0.7)
                for xi, vals in enumerate(data, start=1):
                    vals = np.asarray(vals)
                    if vals.size == 0: continue
                    jitter = (rng.random(vals.size) - 0.5) * 0.15
                    ax.plot(np.full(vals.shape, xi) + jitter, vals, 'o', ms=5, alpha=0.6, color='black')
                base_rgba = pathway_color_dict[pathway]
                base_rgb = np.array(base_rgba[:3])
                base_a = base_rgba[3]
                n_groups = len(order)
                denom = max(1, n_groups - 1)
                w_step = MAX_WHITEN / denom
                for i in range(n_groups - 1):
                    w = w_step * i
                    color_rgb = (1 - w) * base_rgb + w * 1.0
                    ax.plot([i + 1, i + 2], [means[i], means[i + 1]], color=(*color_rgb, base_a), lw=8, solid_capstyle='round')
                point_colors = [(*((1 - w_step * i) * base_rgb + (w_step * i) * 1.0), base_a) for i in range(n_groups)]
                ax.scatter(range(1, n_groups + 1), means, color=point_colors, edgecolor='black', s=200, zorder=5)
                ax.set_xticks(range(1, len(order) + 1))
                ax.set_xticklabels(order, rotation=20, ha='right')
                ax.set_ylabel('|t-statistic|')
                ax.set_title(title)
                ax.grid(axis='y', alpha=0.5)

                # # Annotate with the hierarchical-level trend statistics
                # s = pathway_stats[pathway]
                # if 'spearman_rho' in s:
                #     ax.text(
                #         0.03, 0.97,
                #         (f"Spearman $\\rho$ = {s['spearman_rho']:+.2f} (p = {s['spearman_p']:.3g})\n"
                #          f"JT trend p = {s['jt_p_increasing']:.3g}\n"
                #          f"slope = {s['ols_slope']:+.2f} |t|/level"),
                #         transform=ax.transAxes, va='top', ha='left', fontsize=11,
                #         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)
                #     )

            ymax = max(ax_i.get_ylim()[1] for ax_i in axes)
            for ax_i in axes:
                ax_i.set_ylim(top=ymax)
                ax_i.spines['top'].set_visible(False)
                ax_i.spines['right'].set_visible(False)
                # number of y-ticks: 5
                ax_i.yaxis.set_major_locator(plt.MaxNLocator(5))

            plt.tight_layout()
            expanded_suffix = 'expanded' if CONFIG['include_expanded_networks'] else 'core'
            plt.savefig(
                PATHS['fig'] / 'brain' / f'visual_pathways_{expanded_suffix}_boxplot_pc{comp_no}.png',
                dpi=300, bbox_inches='tight'
            )
            plt.show()

            print(f"  Generated {expanded_suffix} dual stream boxplot for PC#{comp_no}")
            print(f"  Saved figure: visual_pathways_{expanded_suffix}_boxplot_pc{comp_no}.png")

        except Exception as e:
            print(f"  Warning: Could not create expanded dual stream analysis: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("  Skipping surface figures (--skip-surface-figures); numeric outputs are unaffected")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def _parse_args() -> None:
    """CLI flags.  Defaults reproduce the published run exactly."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-surface-figures', '--skip-figures', dest='skip_surface_figures',
        action='store_true',
        help='Skip the brainspace/VTK surface rendering, for machines without '
             'that stack. Some matplotlib panels and result tables are still '
             'written; this is not a blanket "no output" switch. Numeric '
             'results are identical either way.')
    parser.add_argument(
        '--out-root', type=Path, default=None,
        help='Write outputs under this directory instead of the configured '
             'pipeline location. Use to verify a run without overwriting '
             'existing results.')
    args = parser.parse_args()

    if args.skip_surface_figures:
        CONFIG['make_figures'] = False
    if args.out_root is not None:
        PATHS['proc'] = args.out_root / CONFIG['task']
        PATHS['out'] = PATHS['proc'] / 'out'
        PATHS['save'] = PATHS['proc'] / 'save'
        PATHS['tmp'] = PATHS['proc'] / 'tmp'
        for key in ('out', 'save', 'tmp'):
            PATHS[key].mkdir(parents=True, exist_ok=True)
        print(f"Output redirected to: {PATHS['proc']}")


if __name__ == '__main__':
    _parse_args()
    main()
