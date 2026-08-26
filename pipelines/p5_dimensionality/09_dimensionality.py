"""
Dimensionality Analysis of Semantic Brain Representations (TD vs ASD).

Refactored from: 99_main/backup_251230/figures_dim.ipynb
Weight processing aligned with: 12_rsa/03_dimensionality_release11_1whole_brain_mmp_inc_byhx.py

This script performs comprehensive dimensionality analysis using participation ratio:
1. Subject filtering by quality control metrics (FD, prep quality, deepmreye)
2. Load encoding model weights from upstream pipeline
3. Apply preprocessing: site harmonization, frequency regression, WordNet aggregation, z-scoring
4. Compute participation ratio (PR = (Σλ)² / Σλ²) from FEATURE-FEATURE similarity matrices
5. Statistical group comparison (TD vs ASD) using BrainStat
6. Hierarchical clustering of semantic features
7. Network-level aggregation (Yeo7/17, EG17, MMP sections)
8. Visualization (brain maps, wordclouds, radar plots, lollipop plots)
9. External comparison with multitask dimensionality data

Key metric: Participation ratio measures dimensionality of semantic feature representations
across 360 MMP brain parcels. Each subject has unique PR values (subject-specific).

**IMPORTANT CHANGE**: This version computes subject-specific PR values in FEATURE SPACE,
matching the reference implementation. Previous version computed group-level PR in subject space.

All functions are defined locally - this is a standalone script.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import sys
import platform
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
from pprint import pprint
from math import pi

# Scientific computing
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.stats import zscore, pearsonr, ttest_ind
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

# Machine learning
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import SpectralClustering, KMeans

# Neuroimaging
import nibabel as nib
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from statsmodels.stats.multitest import multipletests
from neuroCombat import neuroCombat
from nilearn.glm import regression

# NLP
from nltk.corpus import wordnet31 as wn

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image, ImageDraw
from kneed import KneeLocator

# Utilities
from tqdm import tqdm

# Repository lib/ - shared configuration and the spatial-null library.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, template, fig_dir  # noqa: E402
import spatial_nulls as snull  # noqa: E402

# Optional imports with graceful degradation
try:
    import cbig_network_correspondence as cnc
    HAS_CBIG = True
except ImportError:
    HAS_CBIG = False


# Surface rendering (brainspace + VTK).
#
# The original additionally gated these imports on `os.name == 'nt'`, so on Linux
# every brain-map section was skipped silently even where the packages were
# installed.  Detect the capability, not the platform.
try:
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.vtk_interface import wrap_vtk, serial_connect
    from vtk import vtkPolyDataNormals
    HAS_BRAINSPACE = True
    _BRAINSPACE_ERROR = None
except ImportError as _exc:  # pragma: no cover
    HAS_BRAINSPACE = False
    _BRAINSPACE_ERROR = _exc


def require_surface_plotting() -> None:
    """Raise with an actionable message if the rendering stack is unavailable."""
    if not HAS_BRAINSPACE:
        raise ImportError(
            "Surface rendering requires brainspace and VTK, which failed to "
            f"import ({_BRAINSPACE_ERROR}). Install the environment with "
            "`conda env create -f environment/environment.yml`, or pass "
            "--skip-surface-figures to run without the rendering stack."
        )


# =============================================================================
# RESOLUTION CONSTANTS
# =============================================================================

RESOLUTION_SPECS = {
    '10k': {
        'n_dimension': 20484,  # full dimension of 10k surface including medial wall
        'n_vertices': 20484,  # same as 'n_dimension' -> needs to be masked
        'mmp_atlas_path': '10k_fs_LR/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.10k_fs_LR.dlabel.nii',
        'brain_mask_l_path': 'neuromaps-data/atlases/fsLR/tpl-fsLR_den-10k_hemi-L_desc-nomedialwall_dparc.label.gii',
        'brain_mask_r_path': 'neuromaps-data/atlases/fsLR/tpl-fsLR_den-10k_hemi-R_desc-nomedialwall_dparc.label.gii',
        'fmri_suffix': '_10k.dtseries.nii',
        'prep_column_suffix': '_10k.dtseries.nii',
    },
    '32k': {
        'n_dimension': 64984,  # full dimension of 32k surface including medial wall
        'n_vertices': 59412,  # 32k surface has 59,412 vertices
        'mmp_atlas_path': 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii',
        'medial_wall_path': 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii',
        'brain_mask_l_path': 'neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii',
        'brain_mask_r_path': 'neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii',
        'fmri_suffix': '_32k.dtseries.nii',
        'fmri_suffix_fallback': '.dtseries.nii',  # For files without resolution suffix
        'prep_column_suffix': '_32k.dtseries.nii',
        'prep_column_suffix_fallback': '.dtseries.nii',
    },
}

def get_resolution_spec(resolution: str) -> Dict[str, Any]:
    """
    Get resolution specifications with validation.

    Args:
        resolution: Surface resolution ('10k' or '32k')

    Returns:
        Dictionary with resolution-specific parameters

    Raises:
        ValueError: If resolution not supported
    """
    if resolution not in RESOLUTION_SPECS:
        raise ValueError(f"Unsupported resolution: {resolution}. Must be '10k' or '32k'.")
    return RESOLUTION_SPECS[resolution]


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '09_dimensionality',

    # Source data pipelines (from refactored 99_main)
    'source_pipeline_participants': '99_main',
    'source_pipeline_encoding': '99_main',
    'source_pipeline_perf_mask': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_task_encoding': '08_encoding_model_vertex',  # Vertex-level weights from 08_encoding_model_vertex_1optimal_alpha.py
    'source_task_perf_mask': '06_encoding_model',  # Performance mask from parcel-level encoding

    # Data parameters
    'atlas': 'mmp',
    'resolution': '32k',  # Surface resolution ('10k' or '32k')
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option': 0,             # SRM components for parcel-level encoding (0=no SRM)
    'srm_option_dim': 50,        # SRM components for vertex-level dimensionality (0=no SRM)
    'srm_type': 'det',           # SRM type ('det' for DetSRM or 'prob' for probabilistic SRM)
    'me_option': False,
    'start_ignore_trs': 5,
    'end_ignore_trs': 5,
    'tr_delays_option': [3, 5, 7, 9],

    # Quality control
    'fd_thres': 0.5,             # FD threshold for motion filtering
    'hx_docu': 'True',           # Only include ASD subjects with documentation

    # Encoding model parameters (inherited from upstream, except 'enc_single_alpha')
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha': False,
    'weight_across_delays': 'Avg',
    'weight_add_superordinate': True,

    # Performance mask parameters
    'perf_method': 'fdr',         # 'bf' (Bonferroni) or 'fdr'
    'perf_alpha': 0.01,

    # Dimensionality parameters
    'dim_metric': 'cov',         # 'cov', 'cosine', 'corr', 'euc'
    'within_perf_mask': True,    # Restrict analysis to significant parcels

    # Spatial null models (see spatial_nulls.py). Dimensionality is estimated in every
    # parcel, so correspondence tests run whole-brain (minus zero-PR small parcels).
    'pvalue_within_perf_mask': False,
    'seed_surro': 0,             # Seed for both null models, independent of 'seed'
    'spin_n_rot': 2000,          # Sphere rotations
    'es_n_surr': 2000,           # Eigenstrapping surrogates per map
    'es_num_modes': 200,         # Geometric eigenmodes per hemisphere
    'es_surface': 'midthickness',
    'es_n_jobs': 1,              # Keep at 1 for reproducibility
    'sens_save_excel': True,     # Sensitivity tables are always CSV; also write .xlsx
    'glm_alpha': 0.025,          # Group comparison alpha (0.005=strict, 0.025=default, 0.05=liberal)
    'overlap_alpha': 0.1,       # Lenient alpha for overlap detection with semantic map

    # PCA/semantic map parameters (for overlap analysis with 07_pca.py results)
    'pca_template': 'Whole',          # Template type for PCA ('Whole' as in 07_pca.py)
    'pca_only_sign': True,            # Only use sign of loadings (boolean, True as in 07_pca.py)
    'pca_iter': 10,                   # Number of permutation iterations (10 as in 07_pca.py)
    'sem_enc_single_alpha': True,     # enc_single_alpha used in 07_pca.py (different from this script)

    # Clustering parameters
    'max_clusters': 10,          # Maximum k for hierarchical clustering
    'min_clusters': 3,           # Minimum k for optimal cluster selection
    'percentile_color': 70,      # Percentile threshold for wordcloud coloring

    # Seed parameters
    'seed': 0,                  # Random seed for encoding model (0, 37, 42)

    # Weight preprocessing parameters (matching reference implementation)
    'weight_site_harmonization': True,     # Apply neuroCombat site harmonization
    'weight_freq_regression': True,        # Regress out semantic category frequency
    'weight_add_superordinate': True,      # Add WordNet parent category weights
    'weight_zscore_vertex': True,          # Z-score within each vertex

    # Caching parameters
    'use_weight_cache': True,              # Use cached parcel weights if available
    'n_vtx_threshold': 10,                 # Minimum vertices per parcel (assign 0 if below)

    # File operations
    'save_outputs': True,                                # Enable saving outputs
    # Reuse participation-ratio arrays if they are already present, instead of
    # recomputing them from the vertex-level encoding weights (~20 GB).  The PR
    # arrays are the published entry point for this stage, so caching is the
    # default; --recompute-pr forces the full path.
    #
    # The original expressed this as `False if os.name == 'nt' else True`, using
    # the platform as a proxy for "do I have the vertex weights" - which meant a
    # rerun on Windows silently reused the cache and *appeared* to reproduce.
    'overwrite': False,
    'make_figures': True,  # Render surface figures (brainspace + VTK).
                           # --skip-surface-figures turns this off.
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

        # Source data paths
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        'enc_results': proj_root / '2_pipeline' /
                      config['source_pipeline_encoding'] /
                      config['source_task_encoding'] / 'out',

        'perf_mask_results': proj_root / '2_pipeline' /
                            config['source_pipeline_perf_mask'] /
                            config['source_task_perf_mask'] / 'out',

        'reg': proj_root / '2_pipeline' /
              config['source_pipeline_participants'] /
              config['source_task_participants'] / 'out',
    }

    # Task output paths
    paths['proc'] = paths['pipe'] / task
    paths['out'] = paths['proc'] / 'out'
    paths['save'] = paths['proc'] / 'save'
    paths['tmp'] = paths['proc'] / 'tmp'

    # Create directories
    for key in ['out', 'save', 'tmp', 'fig']:
        if key in paths:
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

# Z-score utility function
zs = lambda v: (v - v.mean(0)) / v.std(0)
zs.__doc__ = """Z-scores (standardizes) each column of array."""


# =============================================================================
# UTILITY FUNCTIONS: FILE I/O
# =============================================================================
# Functions adapted from functions.py - defined locally for standalone operation

def load_files(
    keys: List[str],
    id_list: List[str],
    path: Path
) -> Dict[str, List[npt.NDArray]]:
    """
    Load saved .npy files for each subject.

    Args:
        keys: List of keys to load
        id_list: List of subject IDs
        path: Path to directory containing saved files

    Returns:
        Dictionary mapping keys to lists of arrays (one per subject)
    """
    results = {key: [] for key in keys}

    for sub_id in id_list:
        sub_file = path / f'{sub_id}.npy'
        if not sub_file.exists():
            raise FileNotFoundError(f"File not found: {sub_file}")

        sub_data = np.load(str(sub_file), allow_pickle=True).item()

        for key in keys:
            if key not in sub_data:
                raise KeyError(f"Key '{key}' not found in {sub_file}")
            results[key].append(sub_data[key])

    return results


def save_files(
    data_dict: Dict[str, npt.NDArray],
    id_list: List[str],
    save_path: Path,
    exist_ok: bool = True
) -> None:
    """
    Save results per subject as .npy files.

    Args:
        data_dict: Dictionary with keys mapping to arrays
        id_list: List of subject IDs
        save_path: Path to save directory
        exist_ok: If True, allow overwriting existing files
    """
    save_path.mkdir(parents=True, exist_ok=True)

    for i, sub_id in enumerate(id_list):
        sub_file = save_path / f'{sub_id}.npy'

        if sub_file.exists() and not exist_ok:
            print(f"Skipping existing file: {sub_file}")
            continue

        sub_data = {key: val[i] for key, val in data_dict.items()}
        np.save(str(sub_file), sub_data)


def save_files_summary(
    data_dict: Dict[str, Any],
    summary_path: Path,
    exist_ok: bool = True
) -> None:
    """
    Save group-level summary results.

    Args:
        data_dict: Dictionary with summary data
        summary_path: Path to summary file (.npy or .pkl)
        exist_ok: If True, allow overwriting existing files
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not exist_ok:
        print(f"Skipping existing file: {summary_path}")
        return

    if summary_path.suffix == '.pkl':
        with open(summary_path, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        np.save(str(summary_path), data_dict)


def avg_weight(
    weight_list: List[npt.NDArray],
    n_delays: int = 4
) -> List[npt.NDArray]:
    """
    Average weights across TR delays.

    Args:
        weight_list: List of weight arrays per subject
                    Each array shape: (n_features * n_delays, n_vertices)
                    Delays are interleaved: [f0_d0, f1_d0, ..., fN_d0, f0_d1, f1_d1, ...]
        n_delays: Number of TR delays (typically 4)

    Returns:
        List of averaged weight arrays per subject
        Each array shape: (n_features, n_vertices)
    """
    n_features = int(weight_list[0].shape[0] / n_delays)
    n_vertices = weight_list[0].shape[1]

    result = []
    for weights in weight_list:
        weight_avg = np.zeros((n_features, n_vertices))

        for feat_idx in range(n_features):
            # Indices for this feature across all delays
            # feat_idx, feat_idx + n_features, feat_idx + 2*n_features, ...
            idx_add = [i * n_features for i in range(n_delays)]
            ids_loi = [feat_idx + add for add in idx_add]

            # Extract weights for this feature across all delays
            weight_loi_list = []
            for idx in ids_loi:
                weight_loi_list.append(weights[idx, :])

            # Average across delays
            weight_avg[feat_idx] = np.array(weight_loi_list).mean(0)

        result.append(weight_avg)

    return result


def max_weight(
    weight_list: List[npt.NDArray],
    n_delays: int = 4
) -> List[npt.NDArray]:
    """
    Select maximum absolute weights across TR delays.

    Args:
        weight_list: List of weight arrays per subject
                    Each array shape: (n_features * n_delays, n_vertices)
                    Delays are interleaved: [f0_d0, f1_d0, ..., fN_d0, f0_d1, f1_d1, ...]
        n_delays: Number of TR delays (typically 4)

    Returns:
        List of max absolute weight arrays per subject
        Each array shape: (n_features, n_vertices)
    """
    n_features = int(weight_list[0].shape[0] / n_delays)
    n_vertices = weight_list[0].shape[1]

    result = []
    for weights in weight_list:
        weight_max = np.zeros((n_features, n_vertices))

        for feat_idx in range(n_features):
            # Indices for this feature across all delays
            idx_add = [i * n_features for i in range(n_delays)]
            ids_loi = [feat_idx + add for add in idx_add]

            # Extract weights for this feature across all delays
            weight_loi_list = []
            for idx in ids_loi:
                weight_loi_list.append(weights[idx, :])

            # Convert to array for processing
            weight_loi_array = np.array(weight_loi_list)  # (n_delays, n_vertices)

            # Find max absolute value but keep original sign
            weight_loi_abs = np.abs(weight_loi_array)
            max_idx = np.argmax(weight_loi_abs, axis=0)  # (n_vertices,)

            # Extract original values (with sign) at max positions
            for vtx_idx in range(n_vertices):
                weight_max[feat_idx, vtx_idx] = weight_loi_array[max_idx[vtx_idx], vtx_idx]

        result.append(weight_max)

    return result


# =============================================================================
# WEIGHT PREPROCESSING FUNCTIONS (matching reference implementation)
# =============================================================================

def build_hypernym_lookup(
    label_list: List[str]
) -> Dict[int, List[int]]:
    """
    Build WordNet hypernym (parent category) lookup dictionary.

    For each semantic label, finds all parent categories in the WordNet hierarchy
    and stores their indices for fast lookup during weight aggregation.

    Args:
        label_list: List of WordNet synset labels (e.g., ['person.n.01', 'walk.v.01'])

    Returns:
        Dictionary mapping label_idx -> [parent_idx1, parent_idx2, ...]
    """
    # Create label to index mapping
    label_to_index = {label: idx for idx, label in enumerate(label_list)}

    # Build hypernym lookup
    hypernym_indices = {}

    for label in label_list:
        try:
            # Get synset for this label
            net = wn.synset(label)

            # Get first hypernym path (excludes the label itself)
            hyper_list = net.hypernym_paths()[0][:-1]

            # Map hypernyms to indices
            hyper_indices = [
                label_to_index[hyper.name()]
                for hyper in hyper_list
                if hyper.name() in label_to_index
            ]

            hypernym_indices[label_to_index[label]] = hyper_indices

        except Exception as e:
            print(f'Warning: Could not build hypernyms for {label}: {e}')
            hypernym_indices[label_to_index[label]] = []

    return hypernym_indices


def apply_site_harmonization(
    weight_array: npt.NDArray,
    participants_df: pd.DataFrame,
    id_list: List[str]
) -> npt.NDArray:
    """
    Apply neuroCombat site harmonization to encoding weights.

    Harmonizes weights across scanning sites while preserving group differences
    (TD vs ASD) and controlling for demographic variables.

    Args:
        weight_array: Weight array (n_subjects, n_features, n_vertices_in_parcel)
        participants_df: Demographics dataframe with columns: Site, Sex, DX, Age
        id_list: List of subject IDs matching weight_array order

    Returns:
        Harmonized weight array with same shape
    """
    n_subjects, n_features, n_vertices = weight_array.shape

    # Prepare harmonized array
    weight_har = np.zeros_like(weight_array)

    # Apply neuroCombat per feature dimension
    for feat_idx in range(n_features):
        # Extract weights for this feature: (n_subjects, n_vertices)
        # Then transpose to (n_vertices, n_subjects) for neuroCombat
        feat_weights = weight_array[:, feat_idx, :].T

        # Apply neuroCombat
        # Note: neuroCombat expects (features, samples) format = (n_vertices, n_subjects)
        # neuroCombat returns data in SAME format: (n_vertices, n_subjects)
        combat_result = neuroCombat(
            dat=feat_weights,
            covars=participants_df.loc[id_list],
            batch_col='Site',
            categorical_cols=['Sex', 'DX'],
            continuous_cols=['Age'],
            verbose=False
        )

        # Store harmonized weights (transpose back to (n_subjects, n_vertices))
        weight_har[:, feat_idx, :] = combat_result['data'].T

    return weight_har


def apply_frequency_regression(
    weight_array: npt.NDArray,
    freq: npt.NDArray
) -> npt.NDArray:
    """
    Regress out semantic category frequency from encoding weights.

    Removes variance explained by word frequency to focus on semantic content.

    Args:
        weight_array: Weight array (n_features, n_vertices_in_parcel) for single subject
        freq: Category frequency values (n_features,)

    Returns:
        Residual weights after removing frequency effects (same shape)
    """
    # Design matrix: frequency as predictor
    # design matrix is assumed to be column ordered with observations in rows
    design = freq.reshape(-1, 1)

    # Fit OLS model using nilearn's regression API
    model = regression.OLSModel(design)
    results = model.fit(weight_array)

    # Return residuals (weights with frequency effects removed)
    return results.residuals


def apply_wordnet_superordinate(
    weight_array: npt.NDArray,
    hypernym_indices: Dict[int, List[int]]
) -> npt.NDArray:
    """
    Add WordNet parent category weights to child categories.

    For each semantic label, adds the weights of all parent categories
    (hypernyms) in the WordNet hierarchy. This captures hierarchical
    semantic relationships.

    Args:
        weight_array: Weight array (n_features, n_vertices_in_parcel) for single subject
        hypernym_indices: Dictionary mapping label_idx -> [parent_idx1, parent_idx2, ...]

    Returns:
        Augmented weight array with parent weights added (same shape)
    """
    n_features = weight_array.shape[0]

    # Create copies for safe iteration
    weight_augmented = weight_array.copy()
    weight_original = weight_array.copy()

    # For each label, add all parent category weights
    for label_idx in range(n_features):
        parent_indices = hypernym_indices.get(label_idx, [])

        for parent_idx in parent_indices:
            weight_augmented[label_idx] += weight_original[parent_idx]

    return weight_augmented


# =============================================================================
# ATLAS LOADING AND CONVERSION FUNCTIONS
# =============================================================================

def load_atlases(
    paths: Dict[str, Path],
    resolution: str = '10k'
) -> Dict[str, npt.NDArray]:
    """
    Load MMP, Yeo, and other brain atlases from template directory.

    IMPORTANT: For mmp_2_* functions (mmp_2_section, mmp_2_yeo7, etc.), all atlases
    must be at 32k resolution (64984 vertices) because they work by finding vertex
    correspondences between MMP parcels and network/section atlases.

    Args:
        paths: Path dictionary
        resolution: Surface resolution ('10k' or '32k') - for brain_mask and n_vertices

    Returns:
        Dictionary containing atlas arrays and metadata:
        - mmp_atlas_32k: MMP 360 parcels at 32k resolution (64984,) - for mmp_2_* functions
        - mmp_atlas: MMP 360 parcels at specified resolution (10k or 32k)
        - section_atlas: MMP 22 sections at 32k (64984,)
        - yeo7_atlas: Yeo 7 networks at 32k (64984,), symmetric
        - yeo17_atlas: Yeo 17 networks at 32k (64984,), symmetric
        - eg_atlas: Evan Gordon 17 networks at 32k (64984,) or None
        - brain_mask: Non-medial wall vertex indices at specified resolution
        - n_vertices: Total vertex count for specified resolution
        - resolution: Resolution string ('10k' or '32k')
    """
    res_spec = get_resolution_spec(resolution)
    mmp_folder = paths['tpl'] / 'MMP'

    # ===== Load 32k MMP atlas (required for mmp_2_* functions) =====
    # All network mapping functions need 32k atlases for vertex correspondence
    mmp_32k_path = mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    med_32k_path = mmp_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'

    if not mmp_32k_path.exists():
        raise FileNotFoundError(f"32k MMP atlas not found: {mmp_32k_path}")
    if not med_32k_path.exists():
        raise FileNotFoundError(f"32k medial wall mask not found: {med_32k_path}")

    # Load 32k MMP atlas with medial wall handling
    mmp_atlas_32k_nonmed = nib.load(str(mmp_32k_path)).get_fdata()[0].astype(np.int32)
    med_32k = nib.load(str(med_32k_path)).get_fdata()[0].astype(np.int32).nonzero()[0]
    nonmed_32k_ids = np.array(list(set(range(64984)) - set(med_32k)))

    mmp_atlas_32k = np.zeros(64984, dtype=np.int32)
    mmp_atlas_32k[nonmed_32k_ids] = mmp_atlas_32k_nonmed

    # ===== Load 10k MMP atlas (resolution-matched) =====
    mmp_10k_path = mmp_folder / '10k_fs_LR' / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.10k_fs_LR.dlabel.nii'
    if mmp_10k_path.exists():
        mmp_atlas_10k = nib.load(str(mmp_10k_path)).get_fdata().squeeze().astype(np.int32)
    else:
        print(f"  Warning: 10k MMP atlas not found: {mmp_10k_path}")
        mmp_atlas_10k = None

    if resolution == '32k':
        mmp_atlas = mmp_atlas_32k
    else:
        mmp_atlas = mmp_atlas_10k

    # Load brain mask at specified resolution (for other purposes)
    brain_mask = load_brain_mask_neuromaps(paths['tpl'], resolution)

    result = {
        'mmp_atlas_32k': mmp_atlas_32k,  # Always 32k for mmp_2_* functions
        'mmp_atlas': mmp_atlas,  # Resolution-matched MMP atlas
        'brain_mask': brain_mask,
        'n_vertices': res_spec['n_vertices'],
        'resolution': resolution,
    }

    # ===== Load 32k network/section atlases =====
    # Section atlas (32k)
    section_path = mmp_folder / 'ResultsRegions_ROI.dlabel.nii'
    if section_path.exists():
        section_atlas_nonmed = nib.load(str(section_path)).get_fdata()[0].astype(np.int32)
        section_atlas = np.zeros(64984, dtype=np.int32)
        section_atlas[nonmed_32k_ids] = section_atlas_nonmed
        result['section_atlas'] = section_atlas

    # Yeo7 atlas (32k, symmetric)
    yeo7_path = paths['tpl'] / 'Yeo_JNeurophysiol11_FreeSurfer' / '32k_fs_LR' / 'lh.Yeo2011_7Networks_order.label.gii'
    if yeo7_path.exists():
        yeo7_l = nib.load(str(yeo7_path)).darrays[0].data
        result['yeo7_atlas'] = np.r_[yeo7_l, yeo7_l]

    # Yeo17 atlas (32k, symmetric)
    yeo17_path = paths['tpl'] / 'Yeo_JNeurophysiol11_FreeSurfer' / '32k_fs_LR' / 'lh.Yeo2011_17Networks_order.label.gii'
    if yeo17_path.exists():
        yeo17_l = nib.load(str(yeo17_path)).darrays[0].data
        result['yeo17_atlas'] = np.r_[yeo17_l, yeo17_l]

    # EG17 atlas (32k) - supplies the network profiling in Fig. 3d.  The original
    # swallowed any failure into `eg_atlas = None`, which silently dropped that
    # panel while still exiting 0; a missing atlas is a configuration error.
    from scipy.io import loadmat
    eg_atlas_mat = loadmat(str(template('eg17')))
    result['eg_atlas'] = np.r_[
        eg_atlas_mat['lh_labels'].squeeze(),
        eg_atlas_mat['rh_labels'].squeeze()
    ].astype(np.int32)

    return result


def load_brain_mask_neuromaps(tpl_path: Path, resolution: str) -> npt.NDArray[np.int64]:
    """
    Load brain mask from neuromaps-data (consistent with encoding weights).

    This function mirrors the implementation in 08_encoding_model_vertex_0vertexlevel_srm.py
    to ensure identical vertex indexing.

    Args:
        tpl_path: Path to template directory
        resolution: '10k' or '32k'

    Returns:
        Brain mask array (non-medial wall vertex indices)
    """
    res_spec = get_resolution_spec(resolution)

    mask_l_path = tpl_path / res_spec['brain_mask_l_path']
    mask_r_path = tpl_path / res_spec['brain_mask_r_path']

    if not mask_l_path.exists() or not mask_r_path.exists():
        raise FileNotFoundError(
            f"Brain mask files not found for {resolution}: {mask_l_path}, {mask_r_path}"
        )

    non_med_ids_l = nib.load(str(mask_l_path)).darrays[0].data
    non_med_ids_r = nib.load(str(mask_r_path)).darrays[0].data
    brain_mask = np.r_[non_med_ids_l, non_med_ids_r].nonzero()[0]

    return brain_mask


def value_to_rgb(value: float, cmap_name: str, vmin: float, vmax: float) -> Tuple[float, float, float]:
    """
    Convert a numeric value to RGB color using matplotlib colormap.

    Args:
        value: Numeric value to map to color
        cmap_name: Matplotlib colormap name (e.g., 'magma', 'Spectral_r')
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization

    Returns:
        Tuple of (r, g, b) values in range [0, 1]
    """
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    normalized = np.clip(norm(value), 0, 1)
    rgba = cmap(normalized)
    return rgba[:3]  # Return RGB, exclude alpha


def mmp_2_eg(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    eg_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum_values: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation to Evan Gordon 17 networks.

    Always operates at whole-brain level (360 parcels). If model_perf_mask is provided,
    non-significant parcels are zeroed out before aggregation.
    Supports both 1D and 2D array_mmp inputs.

    Args:
        array_mmp: Array in MMP space (360,) or (360, n_features)
        mmp_atlas: MMP atlas (64984,) - 32k surface
        eg_atlas: EG17 atlas (64984,) - 32k surface
        model_perf_mask: Optional array of indices for significant parcels.
                         Non-significant parcels will be zeroed out.
        sum_values: If True, sum values; else average

    Returns:
        Array in EG17 space (17,) or (17, n_features)
    """
    # Check if array_mmp is 2D, if not, reshape it
    if len(array_mmp.shape) == 1:
        array_mmp = array_mmp.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    num_features = array_mmp.shape[1]

    # Find the mapping between MMP and EG17
    mapping_mmp_2_eg = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        eg_atlas_mmp_ids = eg_atlas[mmp_ids]
        # Find the most frequent EG17 network for this MMP parcel
        eg_atlas_mmp_ids_unique, eg_atlas_mmp_ids_count = np.unique(
            eg_atlas_mmp_ids, return_counts=True
        )
        mapping_mmp_2_eg[mmp_no] = eg_atlas_mmp_ids_unique[
            np.argmax(eg_atlas_mmp_ids_count)
        ]

    # Zero out non-significant parcels if mask is provided
    if model_perf_mask is not None:
        array_mmp_full = np.zeros((360, num_features))
        array_mmp_full[model_perf_mask, :] = array_mmp[model_perf_mask, :]
    else:
        array_mmp_full = array_mmp

    # Convert from MMP to EG17
    array_eg = np.zeros((17, num_features))
    for eg_no in range(17):
        mmp_atlas_eg_ids = np.where(mapping_mmp_2_eg == eg_no + 1)[0]
        if mmp_atlas_eg_ids.size != 0:
            if sum_values:
                array_eg[eg_no, :] = array_mmp_full[mmp_atlas_eg_ids, :].sum(axis=0)
            else:
                array_eg[eg_no, :] = array_mmp_full[mmp_atlas_eg_ids, :].mean(axis=0)

    # Convert nan to 0
    array_eg = np.nan_to_num(array_eg)

    # Squeeze output if input was 1D
    if squeeze_output:
        array_eg = array_eg.squeeze()

    return array_eg


def mmp_2_section(
    array_mmp: npt.NDArray,
    section_atlas: npt.NDArray,
    mmp_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum_values: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation to MMP sections (22 anatomical sections).

    Always operates at whole-brain level (360 parcels). If model_perf_mask is provided,
    non-significant parcels are zeroed out before aggregation.
    Supports both 1D and 2D array_mmp inputs.

    Args:
        array_mmp: Array in MMP space (360,) or (360, n_features)
        section_atlas: Section atlas (64984,) - 32k surface
        mmp_atlas: MMP atlas (64984,) - 32k surface
        model_perf_mask: Optional array of indices for significant parcels.
                         Non-significant parcels will be zeroed out.
        sum_values: If True, sum values; else average

    Returns:
        Array in section space (22,) or (22, n_features)
    """
    # Check if array_mmp is 2D, if not, reshape it
    if len(array_mmp.shape) == 1:
        array_mmp = array_mmp.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    num_features = array_mmp.shape[1]

    # Find the mapping between MMP and MMP section
    mapping_mmp_2_section = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        section_atlas_mmp_ids = section_atlas[mmp_ids]
        # Find the most frequent section for this MMP parcel
        section_atlas_mmp_ids_unique, section_atlas_mmp_ids_count = np.unique(
            section_atlas_mmp_ids, return_counts=True
        )
        mapping_mmp_2_section[mmp_no] = section_atlas_mmp_ids_unique[
            np.argmax(section_atlas_mmp_ids_count)
        ]

    # Zero out non-significant parcels if mask is provided
    if model_perf_mask is not None:
        array_mmp_full = np.zeros((360, num_features))
        array_mmp_full[model_perf_mask, :] = array_mmp[model_perf_mask, :]
    else:
        array_mmp_full = array_mmp

    # Convert from MMP to MMP section (always use all parcels in each section)
    array_mmp_section = np.zeros((22, num_features))
    for section_no in range(22):
        mmp_atlas_section_ids = np.where(mapping_mmp_2_section == section_no + 1)[0]

        if mmp_atlas_section_ids.size != 0:
            if sum_values:
                array_mmp_section[section_no, :] = array_mmp_full[mmp_atlas_section_ids, :].sum(axis=0)
            else:
                array_mmp_section[section_no, :] = array_mmp_full[mmp_atlas_section_ids, :].mean(axis=0)

    # Convert nan to 0
    array_mmp_section = np.nan_to_num(array_mmp_section)

    # Squeeze output if input was 1D
    if squeeze_output:
        array_mmp_section = array_mmp_section.squeeze()

    return array_mmp_section


def mmp_2_yeo17(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    yeo17_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum_values: bool = False
) -> npt.NDArray:
    """
    Convert MMP parcellation to Yeo 17 networks.

    Always operates at whole-brain level (360 parcels). If model_perf_mask is provided,
    non-significant parcels are zeroed out before aggregation.
    Supports both 1D and 2D array_mmp inputs.

    Args:
        array_mmp: Array in MMP space (360,) or (360, n_features)
        mmp_atlas: MMP atlas (64984,) - 32k surface
        yeo17_atlas: Yeo 17 atlas (64984,) - 32k surface
        model_perf_mask: Optional array of indices for significant parcels.
                         Non-significant parcels will be zeroed out.
        sum_values: If True, sum values; else average

    Returns:
        Array in Yeo17 space (17,) or (17, n_features)
    """
    # Check if array_mmp is 2D, if not, reshape it
    if len(array_mmp.shape) == 1:
        array_mmp = array_mmp.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    num_features = array_mmp.shape[1]

    # Find the mapping between MMP and Yeo17
    mapping_mmp_2_yeo17 = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        yeo17_atlas_mmp_ids = yeo17_atlas[mmp_ids]
        # Find the most frequent Yeo17 network for this MMP parcel
        yeo17_atlas_mmp_ids_unique, yeo17_atlas_mmp_ids_count = np.unique(
            yeo17_atlas_mmp_ids, return_counts=True
        )
        mapping_mmp_2_yeo17[mmp_no] = yeo17_atlas_mmp_ids_unique[
            np.argmax(yeo17_atlas_mmp_ids_count)
        ]

    # Zero out non-significant parcels if mask is provided
    if model_perf_mask is not None:
        array_mmp_full = np.zeros((360, num_features))
        array_mmp_full[model_perf_mask, :] = array_mmp[model_perf_mask, :]
    else:
        array_mmp_full = array_mmp

    # Convert from MMP to Yeo17
    array_yeo17 = np.zeros((17, num_features))
    for yeo17_no in range(17):
        mmp_atlas_yeo17_ids = np.where(mapping_mmp_2_yeo17 == yeo17_no + 1)[0]
        if mmp_atlas_yeo17_ids.size != 0:
            if sum_values:
                array_yeo17[yeo17_no, :] = array_mmp_full[mmp_atlas_yeo17_ids, :].sum(axis=0)
            else:
                array_yeo17[yeo17_no, :] = array_mmp_full[mmp_atlas_yeo17_ids, :].mean(axis=0)

    # Convert nan to 0
    array_yeo17 = np.nan_to_num(array_yeo17)

    # Squeeze output if input was 1D
    if squeeze_output:
        array_yeo17 = array_yeo17.squeeze()

    return array_yeo17


# =============================================================================
# SIMILARITY & DIMENSIONALITY FUNCTIONS
# =============================================================================

def compute_dimensionality_all_parcels(
    weight_list_td: List[npt.NDArray],
    weight_list_asd: List[npt.NDArray],
    n_parcels: int,
    metric: str,
    mmp_atlas: npt.NDArray,
    brain_mask: npt.NDArray,
    participants_df: pd.DataFrame,
    id_list_td: List[str],
    id_list_asd: List[str],
    label_list: List[str],
    freq: npt.NDArray,
    hypernym_indices: Dict[int, List[int]],
    cache_path: Path,
    config: Dict[str, Any],
    use_cache: bool = True,
    sim_save_path: Optional[Path] = None
) -> Tuple[npt.NDArray, npt.NDArray, List[npt.NDArray], List[npt.NDArray]]:
    """
    Calculate participation ratio matching reference implementation.

    This implementation computes feature-feature similarity matrices per subject,
    resulting in subject-specific participation ratio values.

    Args:
        weight_list_td: List of TD weights per subject (n_features * n_delays, n_vertices)
                       RAW weights with delays - averaging/max pooling happens per-parcel
        weight_list_asd: List of ASD weights per subject (n_features * n_delays, n_vertices)
                        RAW weights with delays
        n_parcels: Number of parcels (360)
        metric: Similarity metric ('cov', 'corr', 'cosine', 'euc')
        mmp_atlas: MMP atlas array (64984,) with parcel labels
        brain_mask: Non-medial wall vertex indices
        participants_df: Demographics dataframe
        id_list_td: List of TD subject IDs
        id_list_asd: List of ASD subject IDs
        label_list: List of semantic feature labels
        freq: Category frequency values (n_features,)
        hypernym_indices: WordNet hypernym lookup dictionary
        cache_path: Path for caching intermediate results
        config: Configuration dictionary
        use_cache: Whether to use cached parcel weights
        sim_save_path: Optional path to save similarity matrices (same folder as stat_results)

    Returns:
        Tuple of (pr_td, pr_asd, sim_mmp_td_avg_list, sim_mmp_asd_avg_list):
        - pr_td: (n_subs_td, 360) - subject-specific PR values
        - pr_asd: (n_subs_asd, 360) - subject-specific PR values
        - sim_mmp_td_avg_list: List of 360 averaged similarity matrices for TD
        - sim_mmp_asd_avg_list: List of 360 averaged similarity matrices for ASD
    """
    n_subs_td = len(weight_list_td)
    n_subs_asd = len(weight_list_asd)

    # Initialize output arrays
    pr_td_all = np.zeros((n_subs_td, n_parcels))
    pr_asd_all = np.zeros((n_subs_asd, n_parcels))

    # Initialize similarity accumulation lists (for heatmap visualization)
    sim_mmp_td_avg_list = []
    sim_mmp_asd_avg_list = []

    # Combine subject lists for processing
    weight_list_all = weight_list_td + weight_list_asd
    id_list_all = id_list_td + id_list_asd
    n_subjects_all = len(id_list_all)

    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)

    # Get preprocessing parameters
    n_vtx_thres = config['n_vtx_threshold']
    n_delays = len(config['tr_delays_option'])

    # Validate resolution consistency
    resolution = config.get('resolution', '10k')
    res_spec = get_resolution_spec(resolution)

    if mmp_atlas.shape[0] != res_spec['n_dimension']:
        raise ValueError(
            f"MMP atlas shape {mmp_atlas.shape[0]} doesn't match "
            f"resolution {resolution} (expected {res_spec['n_dimension']})"
        )

    if len(brain_mask) > res_spec['n_vertices']:
        raise ValueError(
            f"Brain mask has {len(brain_mask)} indices but resolution "
            f"{resolution} only has {res_spec['n_vertices']} vertices"
        )

    print(f"Computing participation ratio for {n_parcels} parcels...")
    print(f"  Resolution: {resolution} ({res_spec['n_vertices']} vertices)")
    print(f"  Preprocessing steps enabled:")
    print(f"    - Site harmonization: {config['weight_site_harmonization']}")
    print(f"    - Frequency regression: {config['weight_freq_regression']}")
    print(f"    - WordNet superordinate: {config['weight_add_superordinate']}")
    print(f"    - Z-score vertex: {config['weight_zscore_vertex']}")

    for parcel_idx in tqdm(range(n_parcels), desc="Parcels"):
        # === STEP 1: Load or compute parcel weights ===
        cache_file = cache_path / f'weight_(mmp){parcel_idx+1}_list.npy'

        if cache_file.exists() and use_cache:
            # Load from cache
            weight_parcel_list = list(np.load(str(cache_file), allow_pickle=True))
        else:
            # Extract vertices for this parcel
            # Note: brain_mask is from neuromaps-data (same as used for encoding weights)
            parcel_ids = np.where(mmp_atlas[brain_mask] == parcel_idx + 1)[0]

            # Mask weights to parcel vertices
            weight_parcel_list = []
            for weight in weight_list_all:
                weight_masked = weight[:, parcel_ids]  # (n_features * n_delays, n_vertices_in_parcel)
                weight_parcel_list.append(weight_masked)

            # Save to cache
            np.save(str(cache_file), np.array(weight_parcel_list))

        # === STEP 2: Check parcel vertex count ===
        n_vertices_in_parcel = weight_parcel_list[0].shape[1]

        if n_vertices_in_parcel < n_vtx_thres:
            # Assign zero PR for small parcels
            print(f"  Parcel {parcel_idx+1}: {n_vertices_in_parcel} vertices < {n_vtx_thres} threshold")
            print(f'  Parcel {parcel_idx+1}: PR(TD)=0.00, PR(ASD)=0.00')
            pr_td_all[:, parcel_idx] = 0
            pr_asd_all[:, parcel_idx] = 0
            # Append None for small parcels to maintain indexing
            sim_mmp_td_avg_list.append(None)
            sim_mmp_asd_avg_list.append(None)
            continue

        # === STEP 3: Apply preprocessing pipeline ===
        # 3a. Average/max across TR delays
        if config['weight_across_delays'] == 'Avg':
            weight_avg_list = avg_weight(weight_parcel_list, n_delays=n_delays)
        elif config['weight_across_delays'] == 'Max':
            weight_avg_list = max_weight(weight_parcel_list, n_delays=n_delays)
        else:
            weight_avg_list = weight_parcel_list

        weight_avg_array = np.array(weight_avg_list)  # (n_subjects_all, n_features, n_vertices_in_parcel)

        # 3b. Site harmonization
        if config['weight_site_harmonization']:
            weight_har_array = apply_site_harmonization(
                weight_avg_array, participants_df, id_list_all
            )
        else:
            weight_har_array = weight_avg_array

        # 3c. Regress out category frequency (per subject)
        weight_clean_list = []
        if config['weight_freq_regression']:
            for subject_idx in range(n_subjects_all):
                weight_clean = apply_frequency_regression(
                    weight_har_array[subject_idx],  # (n_features, n_vertices)
                    freq
                )
                weight_clean_list.append(weight_clean)
        else:
            weight_clean_list = list(weight_har_array)

        # 3d. Add WordNet superordinate weights (per subject)
        weight_wordnet_list = []
        if config['weight_add_superordinate']:
            for weight_clean in weight_clean_list:
                weight_wordnet = apply_wordnet_superordinate(weight_clean, hypernym_indices)
                weight_wordnet_list.append(weight_wordnet)
        else:
            weight_wordnet_list = weight_clean_list

        # 3e. Z-score within each vertex (per subject)
        if config['weight_zscore_vertex']:
            weight_proc_list = [zscore(w, axis=0) for w in weight_wordnet_list]
        else:
            weight_proc_list = weight_wordnet_list

        # === STEP 4: Separate TD and ASD groups ===
        dx_labels = participants_df.loc[id_list_all, 'DX']
        td_mask = (dx_labels == 'TD').values
        asd_mask = (dx_labels == 'ASD').values

        weight_list_td_proc = [weight_proc_list[i] for i in td_mask.nonzero()[0]]
        weight_list_asd_proc = [weight_proc_list[i] for i in asd_mask.nonzero()[0]]

        # === STEP 5: Compute similarity PER SUBJECT in feature space ===
        sim_list_td = []
        for weight in weight_list_td_proc:  # weight: (n_features, n_vertices_in_parcel)
            if metric == 'cov':
                sim = np.cov(weight)  # (n_features, n_features)
            elif metric == 'corr':
                corr_matrix = np.corrcoef(weight)
                np.fill_diagonal(corr_matrix, 0)
                sim = np.arctanh(corr_matrix)
            elif metric == 'cosine':
                sim = cosine_similarity(weight)
            elif metric == 'euc':
                sim = euclidean_distances(weight)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sim_list_td.append(sim)

        sim_list_asd = []
        for weight in weight_list_asd_proc:
            if metric == 'cov':
                sim = np.cov(weight)
            elif metric == 'corr':
                corr_matrix = np.corrcoef(weight)
                np.fill_diagonal(corr_matrix, 0)
                sim = np.arctanh(corr_matrix)
            elif metric == 'cosine':
                sim = cosine_similarity(weight)
            elif metric == 'euc':
                sim = euclidean_distances(weight)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sim_list_asd.append(sim)

        # === STEP 5b: Save individual similarity matrices and accumulate averaged ===
        # Save individual (per-subject) similarity matrices per parcel
        if sim_save_path is not None:
            sim_save_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            np.save(sim_save_path / f'sim_(mmp){parcel_idx+1}_list_td.npy', sim_list_td)
            np.save(sim_save_path / f'sim_(mmp){parcel_idx+1}_list_asd.npy', sim_list_asd)

        # Accumulate averaged similarity matrices for heatmap visualization
        sim_mmp_td_avg_list.append(np.array(sim_list_td).mean(0))
        sim_mmp_asd_avg_list.append(np.array(sim_list_asd).mean(0))

        # === STEP 6: Calculate PR per subject ===
        pr_td_parcel = []
        for sim in sim_list_td:
            # Verify symmetry
            if not np.allclose(sim, sim.T):
                print(f"  Warning: Parcel {parcel_idx+1} - similarity matrix not symmetric")

            # Extract eigenvalues
            eigvals = np.linalg.eigvals(sim)
            pos_eigvals = eigvals[eigvals > 0]
            pos_eigvals = np.real(pos_eigvals)  # Take real part only

            # Check sufficient positive eigenvalues
            if len(pos_eigvals) < 10:
                print(f"  Warning: Parcel {parcel_idx+1} TD - only {len(pos_eigvals)} positive eigenvalues")

            # Participation ratio
            if len(pos_eigvals) > 0:
                pr = (np.sum(pos_eigvals))**2 / np.sum(pos_eigvals**2)
            else:
                pr = 0.0

            pr_td_parcel.append(pr)

        pr_asd_parcel = []
        for sim in sim_list_asd:
            # Verify symmetry
            if not np.allclose(sim, sim.T):
                print(f"  Warning: Parcel {parcel_idx+1} - similarity matrix not symmetric")

            # Extract eigenvalues
            eigvals = np.linalg.eigvals(sim)
            pos_eigvals = eigvals[eigvals > 0]
            pos_eigvals = np.real(pos_eigvals)

            # Check sufficient positive eigenvalues
            if len(pos_eigvals) < 10:
                print(f"  Warning: Parcel {parcel_idx+1} ASD - only {len(pos_eigvals)} positive eigenvalues")

            # Participation ratio
            if len(pos_eigvals) > 0:
                pr = (np.sum(pos_eigvals))**2 / np.sum(pos_eigvals**2)
            else:
                pr = 0.0

            pr_asd_parcel.append(pr)

        # Print average participation ratio for this parcel
        print(f'  Parcel {parcel_idx+1} ({n_vertices_in_parcel} vertices): PR(TD)={np.mean(pr_td_parcel):.2f}, PR(ASD)={np.mean(pr_asd_parcel):.2f}')

        # === STEP 7: Store results ===
        pr_td_all[:, parcel_idx] = pr_td_parcel
        pr_asd_all[:, parcel_idx] = pr_asd_parcel

    return pr_td_all, pr_asd_all, sim_mmp_td_avg_list, sim_mmp_asd_avg_list


def compute_similarity_for_roi(
    roi_name: str,
    weight_list_td: List[npt.NDArray],
    weight_list_asd: List[npt.NDArray],
    mmp_atlas: npt.NDArray,
    brain_mask: npt.NDArray,
    participants_df: pd.DataFrame,
    id_list_td: List[str],
    id_list_asd: List[str],
    label_list: List[str],
    freq: npt.NDArray,
    hypernym_indices: Dict[int, List[int]],
    config: Dict[str, Any],
    paths: Dict[str, Path],
    metric: str = 'cov'
) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
    """
    Compute similarity matrix for a specific ROI.

    This function extracts weights for a specific parcel, applies the preprocessing
    pipeline, and computes feature-feature similarity matrices for TD and ASD groups.

    Args:
        roi_name: ROI name (e.g., 'R_V1_ROI', 'L_TPOJ1_ROI')
        weight_list_td: List of TD weights per subject (n_features * n_delays, n_vertices)
        weight_list_asd: List of ASD weights per subject
        mmp_atlas: MMP atlas array
        brain_mask: Non-medial wall vertex indices
        participants_df: Demographics dataframe
        id_list_td: List of TD subject IDs
        id_list_asd: List of ASD subject IDs
        label_list: List of semantic feature labels
        freq: Category frequency values
        hypernym_indices: WordNet hypernym lookup dictionary
        config: Configuration dictionary
        paths: Path dictionary
        metric: Similarity metric ('cov', 'corr', 'cosine', 'euc')

    Returns:
        Tuple of (sim_td_avg, sim_asd_avg, label_list):
        - sim_td_avg: Average similarity matrix for TD group (n_features, n_features)
        - sim_asd_avg: Average similarity matrix for ASD group (n_features, n_features)
        - label_list: List of feature labels
    """
    # Load MMP info to get ROI index
    mmp_info_path = paths['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
    mmp_info_df = pd.read_excel(mmp_info_path, engine='openpyxl')
    mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]

    # Get the index of the ROI
    roi_matches = mmp_info_df[mmp_info_df['ROI'] == roi_name]
    if len(roi_matches) == 0:
        raise ValueError(f"ROI '{roi_name}' not found in MMP atlas. "
                        f"Available ROIs: {list(mmp_info_df['ROI'].values[:10])}...")
    parcel_idx = roi_matches.index[0]

    print(f"Computing similarity matrix for {roi_name} (parcel {parcel_idx + 1})...")

    # Combine subject lists for processing
    weight_list_all = weight_list_td + weight_list_asd
    id_list_all = id_list_td + id_list_asd
    n_subjects_all = len(id_list_all)

    # Get preprocessing parameters
    n_vtx_thres = config['n_vtx_threshold']
    n_delays = len(config['tr_delays_option'])

    # Extract vertices for this parcel
    parcel_ids = np.where(mmp_atlas[brain_mask] == parcel_idx + 1)[0]
    n_vertices_in_parcel = len(parcel_ids)

    print(f"  Parcel has {n_vertices_in_parcel} vertices")

    if n_vertices_in_parcel < n_vtx_thres:
        raise ValueError(f"Parcel {roi_name} has only {n_vertices_in_parcel} vertices "
                        f"(threshold: {n_vtx_thres})")

    # Mask weights to parcel vertices
    weight_parcel_list = []
    for weight in weight_list_all:
        weight_masked = weight[:, parcel_ids]
        weight_parcel_list.append(weight_masked)

    # Apply preprocessing pipeline
    # 3a. Average across TR delays
    if config['weight_across_delays'] == 'Avg':
        weight_avg_list = avg_weight(weight_parcel_list, n_delays=n_delays)
    elif config['weight_across_delays'] == 'Max':
        weight_avg_list = max_weight(weight_parcel_list, n_delays=n_delays)
    else:
        weight_avg_list = weight_parcel_list

    weight_avg_array = np.array(weight_avg_list)

    # 3b. Site harmonization
    if config['weight_site_harmonization']:
        weight_har_array = apply_site_harmonization(
            weight_avg_array, participants_df, id_list_all
        )
    else:
        weight_har_array = weight_avg_array

    # 3c. Regress out category frequency
    weight_clean_list = []
    if config['weight_freq_regression']:
        for subject_idx in range(n_subjects_all):
            weight_clean = apply_frequency_regression(
                weight_har_array[subject_idx], freq
            )
            weight_clean_list.append(weight_clean)
    else:
        weight_clean_list = list(weight_har_array)

    # 3d. Add WordNet superordinate weights
    weight_wordnet_list = []
    if config['weight_add_superordinate']:
        for weight_clean in weight_clean_list:
            weight_wordnet = apply_wordnet_superordinate(weight_clean, hypernym_indices)
            weight_wordnet_list.append(weight_wordnet)
    else:
        weight_wordnet_list = weight_clean_list

    # 3e. Z-score within each vertex
    if config['weight_zscore_vertex']:
        weight_proc_list = [zscore(w, axis=0) for w in weight_wordnet_list]
    else:
        weight_proc_list = weight_wordnet_list

    # Separate TD and ASD groups
    dx_labels = participants_df.loc[id_list_all, 'DX']
    td_mask = (dx_labels == 'TD').values
    asd_mask = (dx_labels == 'ASD').values

    weight_list_td_proc = [weight_proc_list[i] for i in td_mask.nonzero()[0]]
    weight_list_asd_proc = [weight_proc_list[i] for i in asd_mask.nonzero()[0]]

    # Compute similarity matrices
    sim_list_td = []
    for weight in weight_list_td_proc:
        if metric == 'cov':
            sim = np.cov(weight)
        elif metric == 'corr':
            corr_matrix = np.corrcoef(weight)
            np.fill_diagonal(corr_matrix, 0)
            sim = np.arctanh(corr_matrix)
        elif metric == 'cosine':
            sim = cosine_similarity(weight)
        elif metric == 'euc':
            sim = euclidean_distances(weight)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        sim_list_td.append(sim)

    sim_list_asd = []
    for weight in weight_list_asd_proc:
        if metric == 'cov':
            sim = np.cov(weight)
        elif metric == 'corr':
            corr_matrix = np.corrcoef(weight)
            np.fill_diagonal(corr_matrix, 0)
            sim = np.arctanh(corr_matrix)
        elif metric == 'cosine':
            sim = cosine_similarity(weight)
        elif metric == 'euc':
            sim = euclidean_distances(weight)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        sim_list_asd.append(sim)

    # Average across subjects
    sim_td_avg = np.mean(sim_list_td, axis=0)
    sim_asd_avg = np.mean(sim_list_asd, axis=0)

    print(f"  Similarity matrix shape: {sim_td_avg.shape}")

    return sim_td_avg, sim_asd_avg, label_list


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def convert_similarity_to_distance(
    similarity_matrix: npt.NDArray
) -> npt.NDArray:
    """
    Convert a similarity matrix to a distance matrix.

    For similarity matrices, higher values indicate more similarity.
    For distance matrices, lower values indicate more similarity.

    Args:
        similarity_matrix: Symmetric similarity matrix (n_samples, n_samples)

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    # Ensure the matrix is symmetric
    assert np.allclose(similarity_matrix, similarity_matrix.T), "Matrix must be symmetric"

    # Convert similarity to distance (1 - similarity)
    # Normalize to [0, 1] range first if needed
    sim_min = np.min(similarity_matrix)
    sim_max = np.max(similarity_matrix)
    norm_sim = (similarity_matrix - sim_min) / (sim_max - sim_min) if sim_max > sim_min else similarity_matrix

    # Convert to distance
    distance_matrix = 1 - norm_sim

    # Ensure zero diagonal (self-distance is 0)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def find_optimal_clusters_hierarchical(
    similarity_matrix: npt.NDArray,
    max_clusters: int = 15,
    min_clusters: int = 2,
    methods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using multiple metrics.

    This implementation matches the original figures_dim.ipynb notebook and uses:
    1. Silhouette Score (higher is better)
    2. Calinski-Harabasz Index (higher is better)
    3. Davies-Bouldin Index (lower is better)
    4. Gap Statistic (higher is better)
    5. Elbow/Knee Method on clustering metrics

    Args:
        similarity_matrix: Similarity matrix (n_samples, n_samples)
                          Higher values indicate more similarity
        max_clusters: Maximum k to test (default: 15)
        min_clusters: Minimum k to consider for optimal cluster selection (default: 2)
                      Metrics are still computed from k=2, but optimal selection
                      only considers k >= min_clusters
        methods: List of methods to use. If None, all methods are used.
                 Options: 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap', 'elbow'

    Returns:
        Dictionary with:
        - 'Z': Linkage matrix
        - 'distance_matrix': Distance matrix computed from similarity
        - 'metrics': Dictionary of metric scores for each k
        - 'optimal': Dictionary of optimal k per method
        - 'min_clusters': Minimum clusters parameter used
    """
    if methods is None:
        methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap', 'elbow']

    # Validate min_clusters
    if min_clusters < 2:
        print(f"Warning: min_clusters={min_clusters} is less than 2. Setting to 2.")
        min_clusters = 2
    if min_clusters > max_clusters:
        raise ValueError(f"min_clusters ({min_clusters}) cannot be greater than max_clusters ({max_clusters})")

    # Convert similarity to distance
    distance_matrix = convert_similarity_to_distance(similarity_matrix)

    # Flatten the distance matrix for linkage
    condensed_dist = squareform(distance_matrix)

    # Compute linkage matrix
    Z = linkage(condensed_dist, method='ward')

    # Results dictionary
    results = {
        'Z': Z,
        'distance_matrix': distance_matrix,
        'min_clusters': min_clusters,
        'max_clusters': max_clusters,
        'metrics': {
            'k': list(range(2, max_clusters + 1)),
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'gap': []
        },
        'optimal': {}
    }

    # Compute MDS points once if needed for multiple metrics
    points = None
    if any(m in methods for m in ['calinski_harabasz', 'davies_bouldin', 'gap']):
        try:
            from sklearn.manifold import MDS
            n_components = min(distance_matrix.shape[0] - 1, 50)
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
            points = mds.fit_transform(distance_matrix)
        except Exception as e:
            print(f"Warning: Could not compute MDS embedding: {e}")

    # Compute metrics for different numbers of clusters
    for k in range(2, max_clusters + 1):
        # Get cluster labels
        labels = fcluster(Z, k, criterion='maxclust')

        # Silhouette score (higher is better)
        if 'silhouette' in methods:
            try:
                sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')
                results['metrics']['silhouette'].append(sil_score)
            except Exception as e:
                print(f"Error calculating silhouette for k={k}: {e}")
                results['metrics']['silhouette'].append(None)

        # Calinski-Harabasz score (higher is better, requires coordinate space)
        if 'calinski_harabasz' in methods:
            try:
                if points is not None:
                    ch_score = calinski_harabasz_score(points, labels)
                    results['metrics']['calinski_harabasz'].append(ch_score)
                else:
                    results['metrics']['calinski_harabasz'].append(None)
            except Exception as e:
                print(f"Error calculating Calinski-Harabasz for k={k}: {e}")
                results['metrics']['calinski_harabasz'].append(None)

        # Davies-Bouldin score (lower is better, requires coordinate space)
        if 'davies_bouldin' in methods:
            try:
                if points is not None:
                    db_score = davies_bouldin_score(points, labels)
                    results['metrics']['davies_bouldin'].append(db_score)
                else:
                    results['metrics']['davies_bouldin'].append(None)
            except Exception as e:
                print(f"Error calculating Davies-Bouldin for k={k}: {e}")
                results['metrics']['davies_bouldin'].append(None)

        # Gap statistic (higher is better)
        if 'gap' in methods:
            try:
                if points is not None:
                    # Calculate within-cluster dispersion for actual data
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(points)
                    actual_dispersion = np.log(kmeans.inertia_)

                    # Generate reference datasets and calculate their dispersions
                    n_refs = 5  # Number of reference datasets
                    ref_dispersions = []

                    for _ in range(n_refs):
                        # Generate random uniform data over the same range as the original data
                        min_vals = np.min(points, axis=0)
                        max_vals = np.max(points, axis=0)
                        random_data = np.random.uniform(
                            low=min_vals, high=max_vals,
                            size=points.shape
                        )

                        # Fit KMeans to reference data
                        kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans_ref.fit(random_data)
                        ref_dispersions.append(np.log(kmeans_ref.inertia_))

                    # Calculate gap statistic
                    gap = np.mean(ref_dispersions) - actual_dispersion
                    results['metrics']['gap'].append(gap)
                else:
                    results['metrics']['gap'].append(None)
            except Exception as e:
                print(f"Error calculating Gap statistic for k={k}: {e}")
                results['metrics']['gap'].append(None)

    # Find optimal clusters based on metrics (respecting min_clusters constraint)
    # Create filtered pairs that only include k >= min_clusters
    k_values = results['metrics']['k']
    min_idx = min_clusters - 2  # Index offset since k starts at 2

    if 'silhouette' in methods and results['metrics']['silhouette']:
        # Filter to k >= min_clusters
        valid_pairs = [
            (k, s) for k, s in zip(k_values, results['metrics']['silhouette'])
            if s is not None and k >= min_clusters
        ]
        if valid_pairs:
            opt_k = max(valid_pairs, key=lambda x: x[1])[0]  # Higher is better
            results['optimal']['silhouette'] = opt_k

    if 'calinski_harabasz' in methods and results['metrics']['calinski_harabasz']:
        valid_pairs = [
            (k, s) for k, s in zip(k_values, results['metrics']['calinski_harabasz'])
            if s is not None and k >= min_clusters
        ]
        if valid_pairs:
            opt_k = max(valid_pairs, key=lambda x: x[1])[0]  # Higher is better
            results['optimal']['calinski_harabasz'] = opt_k

    if 'davies_bouldin' in methods and results['metrics']['davies_bouldin']:
        valid_pairs = [
            (k, s) for k, s in zip(k_values, results['metrics']['davies_bouldin'])
            if s is not None and k >= min_clusters
        ]
        if valid_pairs:
            opt_k = min(valid_pairs, key=lambda x: x[1])[0]  # Lower is better
            results['optimal']['davies_bouldin'] = opt_k

    if 'gap' in methods and results['metrics']['gap']:
        valid_pairs = [
            (k, s) for k, s in zip(k_values, results['metrics']['gap'])
            if s is not None and k >= min_clusters
        ]
        if valid_pairs:
            opt_k = max(valid_pairs, key=lambda x: x[1])[0]  # Higher is better
            results['optimal']['gap'] = opt_k

    # Elbow/Knee method for appropriate metrics (respecting min_clusters)
    # Note: Silhouette is excluded because it's non-monotonic (elbow detection not meaningful)
    if 'elbow' in methods:
        # Apply elbow method to monotonic metrics only (higher is better)
        for metric_name in ['calinski_harabasz', 'gap']:
            if metric_name in methods and results['metrics'][metric_name]:
                try:
                    values = results['metrics'][metric_name]
                    # Filter out None values and k < min_clusters
                    valid_pairs = [
                        (k, v) for k, v in zip(k_values, values)
                        if v is not None and k >= min_clusters
                    ]
                    if len(valid_pairs) >= 3:  # Need at least 3 points for knee detection
                        x = [p[0] for p in valid_pairs]
                        y = [p[1] for p in valid_pairs]
                        kneedle = KneeLocator(x, y, curve='convex', direction='increasing')
                        elbow_point = kneedle.elbow
                        if elbow_point and elbow_point >= min_clusters:
                            results['optimal'][f'{metric_name}_elbow'] = elbow_point
                except Exception as e:
                    print(f"Error calculating elbow for {metric_name}: {e}")

        # For Davies-Bouldin (lower is better, so the curve is concave)
        if 'davies_bouldin' in methods and results['metrics']['davies_bouldin']:
            try:
                values = results['metrics']['davies_bouldin']
                valid_pairs = [
                    (k, v) for k, v in zip(k_values, values)
                    if v is not None and k >= min_clusters
                ]
                if len(valid_pairs) >= 3:
                    x = [p[0] for p in valid_pairs]
                    y = [p[1] for p in valid_pairs]
                    kneedle = KneeLocator(x, y, curve='concave', direction='decreasing')
                    elbow_point = kneedle.elbow
                    if elbow_point and elbow_point >= min_clusters:
                        results['optimal']['davies_bouldin_elbow'] = elbow_point
            except Exception as e:
                print(f"Error calculating elbow for davies_bouldin: {e}")

    # Calculate a consensus optimal K (simple average of all methods, matching original)
    # Ensure consensus is at least min_clusters
    if results['optimal']:
        consensus = int(round(np.mean(list(results['optimal'].values()))))
        results['optimal']['consensus'] = max(consensus, min_clusters)
    else:
        results['optimal']['consensus'] = min_clusters  # Default fallback respects min_clusters

    return results


def draw_diagonal_rectangles(
    ax: plt.Axes,
    row_labels_sorted: npt.NDArray,
    col_labels_sorted: npt.NDArray,
    edgecolor: str = 'white',
    linewidth: float = 10
) -> None:
    """
    Draw rectangle outlines on the given axis for diagonal bicluster blocks only.

    Each rectangle is drawn only for blocks where the row cluster equals the column cluster.
    This creates visually distinct bounding boxes around clustered regions on the diagonal.

    Args:
        ax: Matplotlib axes to draw on
        row_labels_sorted: Cluster labels for rows (reordered by dendrogram)
        col_labels_sorted: Cluster labels for columns (reordered by dendrogram)
        edgecolor: Color of the rectangle edges
        linewidth: Width of the rectangle edges
    """
    from matplotlib import patches

    unique_labels = np.unique(row_labels_sorted)

    # Get boundaries for rows for each cluster
    row_boundaries = {}
    for label in unique_labels:
        indices = np.where(row_labels_sorted == label)[0]
        row_boundaries[label] = (indices[0], indices[-1])

    # Get boundaries for columns for each cluster
    col_boundaries = {}
    for label in unique_labels:
        indices = np.where(col_labels_sorted == label)[0]
        col_boundaries[label] = (indices[0], indices[-1])

    # Draw a rectangle only for the blocks where the cluster labels match
    for label in unique_labels:
        r_start, r_end = row_boundaries[label]
        c_start, c_end = col_boundaries[label]
        rect = patches.Rectangle(
            (c_start - 0.5, r_start - 0.5),  # position (x, y)
            (c_end - c_start + 1),           # width
            (r_end - r_start + 1),           # height
            fill=False,
            edgecolor=edgecolor,
            lw=linewidth
        )
        ax.add_patch(rect)


def visualize_hierarchical_clustering(
    similarity_matrix: npt.NDArray,
    optimal_k: int,
    linkage_matrix: Optional[npt.NDArray] = None,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = 'magma',
    plot_dendrogram: bool = True,
    plot_reordered: bool = True,
    plot_metrics: bool = True,
    metric_results: Optional[Dict[str, Any]] = None,
    visualize: bool = True
) -> Tuple[Optional[plt.Figure], npt.NDArray, npt.NDArray]:
    """
    Visualize hierarchical clustering results with dendrogram and reordered similarity matrix.

    Args:
        similarity_matrix: Similarity matrix (n_samples, n_samples)
        optimal_k: Optimal number of clusters
        linkage_matrix: Precomputed linkage matrix (optional)
        labels: Labels for the rows/columns of the similarity matrix
        figsize: Figure size
        cmap: Colormap for heatmaps
        plot_dendrogram: Whether to plot the dendrogram
        plot_reordered: Whether to plot the reordered similarity matrix
        plot_metrics: Whether to plot the clustering metrics
        metric_results: Results from find_optimal_clusters_hierarchical
        visualize: Whether to display the figure or just compute the results

    Returns:
        Tuple of (figure, ordered_indices, cluster_labels)
    """
    # Use original labels or create default ones
    if labels is None:
        labels = [f'Item {i+1}' for i in range(similarity_matrix.shape[0])]

    # Compute linkage if not provided
    if linkage_matrix is None:
        distance_matrix = convert_similarity_to_distance(similarity_matrix)
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='ward')

    # Get cluster labels for optimal_k
    cluster_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')

    # Compute dendrogram data without plotting
    dendrogram_data = dendrogram(linkage_matrix, no_plot=True)
    ordered_idx = np.array(dendrogram_data['leaves'])

    # Only create figure and plot if visualize is True
    if not visualize:
        return None, ordered_idx, cluster_labels

    # Set up the figure with appropriate subplots
    n_plots = sum([plot_dendrogram, plot_reordered, plot_metrics and metric_results is not None])
    fig = plt.figure(figsize=figsize)

    plot_idx = 1

    # Plot dendrogram
    if plot_dendrogram:
        ax_dendro = plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1

        dendrogram(
            linkage_matrix,
            labels=labels,
            orientation='left',
            color_threshold=linkage_matrix[-optimal_k+1, 2] if optimal_k > 1 else 0,
            leaf_font_size=8,
            ax=ax_dendro
        )

        ax_dendro.set_title('Hierarchical Clustering Dendrogram')
        ax_dendro.set_xlabel('Distance')

    # Plot reordered similarity matrix
    if plot_reordered:
        ax_heatmap = plt.subplot(1, n_plots, plot_idx)
        plot_idx += 1

        # Reorder the similarity matrix
        reordered_sim = similarity_matrix[ordered_idx][:, ordered_idx]

        # Plot heatmap
        im = ax_heatmap.imshow(reordered_sim, aspect='auto', cmap=cmap)

        # Reorder cluster labels according to dendrogram
        reordered_labels = cluster_labels[ordered_idx]

        # Draw diagonal rectangles around clusters
        draw_diagonal_rectangles(ax_heatmap, reordered_labels, reordered_labels,
                                 edgecolor='white', linewidth=10)

        # Add colorbar
        plt.colorbar(im, ax=ax_heatmap, label='Similarity')

        # Add labels if there aren't too many
        if len(labels) <= 30:
            reordered_labels_text = [labels[i] for i in ordered_idx]
            ax_heatmap.set_xticks(np.arange(len(reordered_labels_text)))
            ax_heatmap.set_xticklabels(reordered_labels_text, rotation=90, fontsize=8)
            ax_heatmap.set_yticks(np.arange(len(reordered_labels_text)))
            ax_heatmap.set_yticklabels(reordered_labels_text, fontsize=8)
        else:
            ax_heatmap.set_xticks([])
            ax_heatmap.set_yticks([])

        ax_heatmap.set_title(f'Reordered Similarity Matrix\n(k={optimal_k} clusters)')

    # Plot metrics if provided
    if plot_metrics and metric_results is not None:
        ax_metrics = plt.subplot(1, n_plots, plot_idx)

        x_values = metric_results['metrics']['k']

        # Define colors and styles for each metric
        metric_styles = {
            'silhouette': {'color': 'b', 'marker': 'o', 'label': 'Silhouette'},
            'calinski_harabasz': {'color': 'g', 'marker': 's', 'label': 'Calinski-Harabasz'},
            'davies_bouldin': {'color': 'r', 'marker': '^', 'label': 'Davies-Bouldin'},
            'gap': {'color': 'm', 'marker': 'd', 'label': 'Gap Statistic'}
        }

        # Plot all available metrics
        for metric_name, style in metric_styles.items():
            if metric_results['metrics'].get(metric_name):
                valid_pairs = [(k, s) for k, s in zip(x_values, metric_results['metrics'][metric_name]) if s is not None]
                if valid_pairs:
                    ks, scores = zip(*valid_pairs)
                    ax_metrics.plot(ks, scores, f"{style['color']}-{style['marker']}", label=style['label'])
                    # Mark optimal (max/min based method)
                    if metric_name in metric_results['optimal']:
                        opt_k = metric_results['optimal'][metric_name]
                        if opt_k in ks:
                            ax_metrics.axvline(x=opt_k, color=style['color'], linestyle='--', alpha=0.5)
                    # Mark elbow point with star marker
                    elbow_key = f'{metric_name}_elbow'
                    if elbow_key in metric_results['optimal']:
                        elbow_k = metric_results['optimal'][elbow_key]
                        if elbow_k in ks:
                            elbow_idx = list(ks).index(elbow_k)
                            elbow_score = scores[elbow_idx]
                            ax_metrics.plot(elbow_k, elbow_score, marker='*', markersize=15,
                                          color=style['color'], markeredgecolor='black',
                                          markeredgewidth=1, zorder=5)

        ax_metrics.set_xlabel('Number of Clusters (k)')
        ax_metrics.set_ylabel('Score')
        ax_metrics.set_title('Clustering Metrics')
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ordered_idx, cluster_labels


def run_hierarchical_clustering_analysis(
    similarity_matrix: Union[npt.NDArray, pd.DataFrame],
    max_clusters: int = 15,
    min_clusters: int = 2,
    labels: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    visualize: bool = True,
    optimal_k: Optional[int] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run comprehensive hierarchical clustering analysis on a similarity matrix.

    This function performs the full clustering workflow:
    1. Find optimal number of clusters using specified methods
    2. Visualize dendrogram, reordered similarity matrix, and metrics
    3. Return cluster assignments and reordered data

    Args:
        similarity_matrix: Similarity matrix (n_samples, n_samples) or DataFrame
                          Higher values indicate more similarity
        max_clusters: Maximum number of clusters to consider (default: 15)
        min_clusters: Minimum number of clusters for optimal selection (default: 2)
        labels: Labels for the rows/columns of the similarity matrix
        methods: Methods to use for determining optimal clusters
                 Options: ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap', 'elbow']
        visualize: Whether to display figures
        optimal_k: Override optimal k (if None, uses consensus from methods)
        save_path: Path to save the figure

    Returns:
        Dictionary with:
        - 'optimal_k': Optimal number of clusters
        - 'evaluation_results': Results from find_optimal_clusters_hierarchical
        - 'linkage_matrix': Hierarchical clustering linkage matrix
        - 'cluster_assignments': Cluster label for each item (1-indexed)
        - 'cluster_mapping': Dictionary mapping item index to cluster label
        - 'ordered_indices': Indices ordered by dendrogram
        - 'ordered_labels': Labels ordered by dendrogram
        - 'reordered_similarity': Similarity matrix reordered by clustering
        - 'figure': Matplotlib figure object (if visualize=True)
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(similarity_matrix, pd.DataFrame):
        labels = list(similarity_matrix.index) if labels is None else labels
        similarity_matrix = similarity_matrix.values

    # Find optimal number of clusters
    if visualize:
        print("Finding optimal number of clusters...")
    results = find_optimal_clusters_hierarchical(similarity_matrix, max_clusters, min_clusters, methods)

    # Get the consensus optimal k
    optimal_k_consensus = results['optimal'].get('consensus', 3)
    if visualize:
        print(f"Optimal number of clusters determined: {optimal_k_consensus}")

    # Print all optimal k values from different methods
    if visualize:
        print("Optimal k values from different methods:")
        for method, k in results['optimal'].items():
            if method != 'consensus':
                print(f"  - {method}: {k}")

    if optimal_k is None:
        optimal_k = optimal_k_consensus

    # Visualize results
    if visualize:
        print("Visualizing clustering results...")
    fig, ordered_idx, cluster_labels = visualize_hierarchical_clustering(
        similarity_matrix, optimal_k,
        linkage_matrix=results['Z'],
        labels=labels,
        metric_results=results,
        visualize=visualize
    )

    if save_path and fig is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved clustering figure to {save_path}")

    # Get reordered similarity matrix
    reordered_sim = similarity_matrix[ordered_idx][:, ordered_idx]

    # Get ordered labels if provided
    if labels is not None:
        if isinstance(labels, np.ndarray):
            ordered_labels = labels[ordered_idx]
        else:
            ordered_labels = [labels[i] for i in ordered_idx]
    else:
        ordered_labels = None

    # Get cluster assignments for each item
    cluster_assignments = fcluster(results['Z'], optimal_k, criterion='maxclust')

    # Create mapping from original indices to cluster labels
    cluster_mapping = {i: cluster_assignments[i] for i in range(len(cluster_assignments))}

    # Print items in each cluster
    if visualize:
        print(f"\nCluster assignments (k={optimal_k}):")
        for cluster_id in range(1, optimal_k + 1):
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    cluster_items = labels[cluster_indices]
                else:
                    cluster_items = [labels[i] for i in cluster_indices]
            else:
                cluster_items = [f"Item {i+1}" for i in cluster_indices]

            print(f"Cluster {cluster_id}: {', '.join(map(str, cluster_items))}")

    # Return comprehensive results
    return {
        'optimal_k': optimal_k,
        'evaluation_results': results,
        'linkage_matrix': results['Z'],
        'cluster_assignments': cluster_assignments,
        'cluster_mapping': cluster_mapping,
        'ordered_indices': ordered_idx,
        'ordered_labels': ordered_labels,
        'reordered_similarity': reordered_sim,
        'figure': fig
    }


def plot_similarity_heatmap_clustered(
    similarity_matrix: Union[npt.NDArray, pd.DataFrame],
    labels: Optional[List[str]] = None,
    n_clusters: Optional[int] = None,
    max_clusters: int = 10,
    min_clusters: int = 2,
    methods: Optional[List[str]] = None,
    cmap: str = 'magma',
    cmax: Optional[float] = None,
    cmin: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 12),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show_labels: bool = True,
    show_cluster_boundaries: bool = True
) -> Dict[str, Any]:
    """
    Draw heatmap of similarity matrix reordered by hierarchical clustering.

    This is a convenience function that combines clustering analysis with
    a clean heatmap visualization focused on the reordered similarity matrix.

    Args:
        similarity_matrix: Similarity matrix (n_samples, n_samples) or DataFrame
        labels: Labels for the rows/columns
        n_clusters: Number of clusters (if None, automatically determined)
        max_clusters: Maximum clusters to consider for automatic determination
        min_clusters: Minimum clusters for optimal selection (default: 2)
        methods: Methods for optimal cluster determination
        cmap: Colormap for heatmap
        cmax: Maximum value for color scale
        cmin: Minimum value for color scale
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        show_labels: Whether to show axis labels
        show_cluster_boundaries: Whether to draw cluster boundary lines

    Returns:
        Dictionary with clustering results and figure
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(similarity_matrix, pd.DataFrame):
        labels = list(similarity_matrix.index) if labels is None else labels
        similarity_matrix = similarity_matrix.values

    if methods is None:
        methods = ['silhouette']

    # Run clustering analysis (without visualization)
    results = run_hierarchical_clustering_analysis(
        similarity_matrix,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        labels=labels,
        methods=methods,
        visualize=False,
        optimal_k=n_clusters
    )

    optimal_k = results['optimal_k']
    ordered_idx = results['ordered_indices']
    cluster_assignments = results['cluster_assignments']

    # Reorder the similarity matrix
    reordered_sim = similarity_matrix[ordered_idx][:, ordered_idx]

    # Create the heatmap figure
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(reordered_sim, cmap=cmap, aspect='auto', vmin=cmin, vmax=cmax)

    # Draw cluster boundaries as diagonal rectangles
    if show_cluster_boundaries:
        reordered_labels = cluster_assignments[ordered_idx]
        draw_diagonal_rectangles(ax, reordered_labels, reordered_labels,
                                 edgecolor='white', linewidth=10)

    # # Add colorbar
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Similarity')

    # Add labels
    if show_labels and labels is not None and len(labels) <= 50:
        reordered_labels_text = [labels[i] for i in ordered_idx]
        ax.set_xticks(np.arange(len(reordered_labels_text)))
        ax.set_xticklabels(reordered_labels_text, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(reordered_labels_text)))
        ax.set_yticklabels(reordered_labels_text, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Similarity Matrix (Hierarchical Clustering, k={optimal_k})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.show()

    # Add figure to results
    results['heatmap_figure'] = fig

    return results


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def surfstat_dim(
    dim_td: npt.NDArray,
    dim_asd: npt.NDArray,
    id_list_td: List[str],
    id_list_asd: List[str],
    demo_df: pd.DataFrame
) -> Dict[str, npt.NDArray]:
    """
    Statistical comparison using BrainStat (Python SurfStat).

    Fits GLM: Y ~ Group + Site + Age + Sex + MeanFD
    Contrast: ASD - TD

    Args:
        dim_td: TD participation ratios (n_subs_td, 360)
        dim_asd: ASD participation ratios (n_subs_asd, 360)
        id_list_td: List of TD subject IDs
        id_list_asd: List of ASD subject IDs
        demo_df: Demographics dataframe with columns: DX, Site, Sex, Age, Mean_FD_DM

    Returns:
        Dictionary with:
        - 't': T-statistics (360,)
        - 'p': P-values (360,)
        - 'p_fdr': FDR-corrected p-values (360,)
    """
    # Prepare demographic dataframe
    id_list_all = id_list_td + id_list_asd

    group_list = [group for group in demo_df.loc[id_list_all, 'DX']]
    site_list = [site for site in demo_df.loc[id_list_all, 'Site']]
    sex_list = [sex for sex in demo_df.loc[id_list_all, 'Sex']]
    age_list = [age for age in demo_df.loc[id_list_all, 'Age']]
    meanfd_list = [meanfd for meanfd in demo_df.loc[id_list_all, 'Mean_FD_DM']]

    surfstat_df = pd.DataFrame({
        'Group': group_list,
        'Site': site_list,
        'Sex': sex_list,
        'Age': age_list,
        'MeanFD': meanfd_list
    })
    surfstat_df['Age'] = surfstat_df['Age'].astype(float)
    surfstat_df['MeanFD'] = surfstat_df['MeanFD'].astype(float)

    # Define model terms
    term_group = FixedEffect(surfstat_df.Group)
    term_site = FixedEffect(surfstat_df.Site)
    term_sex = FixedEffect(surfstat_df.Sex)
    term_age = FixedEffect(surfstat_df.Age)
    term_meanfd = FixedEffect(surfstat_df.MeanFD)

    model = term_group + term_site + term_age + term_sex + term_meanfd

    # Define contrast (ASD - TD)
    contrast_group = (surfstat_df.Group == "ASD").astype(int) - (surfstat_df.Group == "TD").astype(int)

    # Concatenate data
    dim_all = np.concatenate([dim_td, dim_asd], axis=0)  # (n_subs_all, 360)

    # Fit SLM
    slm = SLM(
        model,
        contrast_group,
        correction=["fdr"],
        two_tailed=True
    )
    slm.fit(dim_all)

    # Extract results
    # Use scipy.stats for p-value calculation (more reliable)
    from scipy.stats import t as t_dist

    t_stats = slm.t.flatten()
    df = slm.df
    p_vals = t_dist.sf(np.abs(t_stats), df) * 2  # Two-tailed

    # FDR correction
    _, p_fdr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    return {
        't': t_stats,
        'p': p_vals,
        'p_fdr': p_fdr
    }


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
        roiarray: ROI array data (n_masked_rois,) or (n_rois,)
        mask: Model performance mask (indices of significant ROIs)
        atlas: Parcellation atlas ('mmp', 'sch-400', 'sch-800')
        output_folder: Output folder name
        output_file: Output file name (must end with .dscalar.nii)
        fig_path: Figure output path (if None, uses MMP folder)
        comp_no: Component number (unused, for compatibility)
    """
    if not HAS_BRAINSPACE:
        print("  Warning: brainspace not available. Cannot create CIFTI file.")
        return

    # Check output file extension
    assert output_file.endswith('.dscalar.nii'), \
        'output_file should end with .dscalar.nii'

    # Load brain mask (32k) - use neuromaps-data mask
    brain_mask_l = nib.load(str(PATHS['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                'tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    brain_mask_r = nib.load(str(PATHS['tpl'] / 'neuromaps-data' / 'atlases' / 'fsLR' /
                                'tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii')).darrays[0].data
    brain_mask = np.r_[brain_mask_l, brain_mask_r].nonzero()[0]

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
        # roiarray contains only masked ROIs, expand to full 360
        data_temp = np.empty(label_atlas.max())
        data_temp[:] = np.nan
        data_temp[mask] = roiarray
        data_32karray = map_to_labels(data_temp, label_atlas,
                                      mask=label_atlas != 0, fill=np.nan)
    else:
        # roiarray is already full (360,), just map directly
        data_32karray = map_to_labels(roiarray, label_atlas,
                                      mask=label_atlas != 0, fill=np.nan)

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


def roiarray_2_32kdlabel(
    roiarray: npt.NDArray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    output_folder: str,
    output_file: str,
    fig_path: Optional[Path] = None,
    nan_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
) -> None:
    """
    Save a per-parcel (360,) array as a colored MMP dlabel.nii for Connectome Workbench.

    Rather than relying on Workbench's built-in palettes, this bakes an arbitrary
    matplotlib colormap into the parcellation's label color table: the MMP vertex->parcel
    assignment is kept intact, but each parcel's color is overwritten with the color of
    its value under `cmap_name` normalized to [vmin, vmax]. This makes any matplotlib
    colormap usable in Workbench (same idea as the t-statistic dlabel export in main()).

    Note: because this is a label file, coloring is flat per parcel and Workbench will
    not show a scale bar - generate a colorbar separately with the same cmap/vmin/vmax.

    Args:
        roiarray: Per-parcel values (360,) in MMP order (parcel i -> label key i+1)
        cmap_name: Any matplotlib colormap name (e.g. 'cividis', 'magma', 'Spectral_r')
        vmin: Lower bound for color normalization
        vmax: Upper bound for color normalization
        output_folder: Output subfolder name
        output_file: Output file name (must end with .dlabel.nii)
        fig_path: Base output path (if None, uses template MMP folder)
        nan_color: RGB (0-1) used for parcels whose value is NaN
    """
    assert output_file.endswith('.dlabel.nii'), \
        'output_file should end with .dlabel.nii'

    mmp_folder = PATHS['tpl'] / 'MMP'

    # Load MMP dlabel as a template: provides vertex->parcel integers and the color table
    dummy_dlabel_path = mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    dummy_dlabel = nib.load(str(dummy_dlabel_path))
    dummy_32k_nonmed = dummy_dlabel.get_fdata().copy().astype(int)
    label_color_dict = dummy_dlabel.header.get_axis(0).label
    new_label_color_dict = deepcopy(label_color_dict)

    # Colormap and normalization (any matplotlib colormap works)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Recolor each parcel by its value (label key mmp_idx+1; key 0 is medial wall/background)
    for mmp_idx in range(360):
        value = roiarray[mmp_idx]
        if np.isnan(value):
            rgb = nan_color
        else:
            rgb = cmap(norm(value))[:3]
        new_rgba = tuple(rgb) + (1.0,)
        # Preserve the original parcel name ([0]), overwrite only the color
        new_label_color_dict[0][mmp_idx + 1] = (new_label_color_dict[0][mmp_idx + 1][0], new_rgba)

    # Rebuild header with the new color table and wrap the unchanged label data
    label_axis = dummy_dlabel.header.get_axis(0)
    label_axis.label = new_label_color_dict
    new_header = nib.Cifti2Header.from_axes((label_axis, dummy_dlabel.header.get_axis(1)))
    dlabel_cifti = nib.Cifti2Image(dummy_32k_nonmed, header=new_header)

    # Save file
    if fig_path is None:
        output_dir = mmp_folder / output_folder
    else:
        output_dir = fig_path / output_folder

    output_dir.mkdir(parents=True, exist_ok=True)
    dlabel_cifti.to_filename(str(output_dir / output_file))


# =============================================================================
# BRAIN SURFACE VISUALIZATION FUNCTIONS
# =============================================================================

def load_brain_surfaces(
    paths: Dict[str, Path],
    atlas: str = 'mmp'
) -> Dict[str, Any]:
    """
    Load brain surfaces and label atlas for visualization.

    This function loads the inflated brain surfaces and MMP atlas required
    for brain surface visualization using brainspace.

    Args:
        paths: Path dictionary containing 'tpl' (template) path
        atlas: Atlas name ('mmp', 'sch-400', 'sch-800')

    Returns:
        Dictionary with:
        - 'surf_lh': Left hemisphere surface
        - 'surf_rh': Right hemisphere surface
        - 'label_atlas': Label atlas (64984,)
        - 'available': Boolean indicating if brainspace is available
    """
    if not HAS_BRAINSPACE:
        print("  Warning: brainspace not available. Brain visualization skipped.")
        return {'available': False}

    mmp_folder = paths['tpl'] / 'MMP'

    # Load surfaces
    surfs = [None, None]
    surfs[0] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.L.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
    nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
    surf_lh = serial_connect(surfs[0], nf)

    surfs[1] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.R.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
    nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
    surf_rh = serial_connect(surfs[1], nf)

    # Load atlas
    if atlas == 'mmp':
        atlas_nonmed = nib.load(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')).get_fdata()[0].astype(np.int32)
        med_32k = nib.load(str(mmp_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii')).get_fdata()[0].astype(np.int32).nonzero()[0]
        nonmed_32k_ids = np.array(list(set(range(64984)) - set(list(med_32k))))
        label_atlas = np.zeros(64984).astype(np.int32)
        label_atlas[nonmed_32k_ids] = atlas_nonmed
    elif atlas == 'sch-400':
        label_atlas = nib.load(str(paths['tpl'] / 'schaefer/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii')).get_fdata().astype(np.int32).squeeze()
    elif atlas == 'sch-800':
        label_atlas = nib.load(str(paths['tpl'] / 'schaefer/Parcellations/HCP/fslr32k/cifti/Schaefer2018_800Parcels_17Networks_order.dlabel.nii')).get_fdata().astype(np.int32).squeeze()
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    return {
        'surf_lh': surf_lh,
        'surf_rh': surf_rh,
        'label_atlas': label_atlas,
        'available': True
    }


def visualize_participation_ratio(
    pr_td: npt.NDArray,
    pr_asd: npt.NDArray,
    stat_results: Dict[str, npt.NDArray],
    brain_surfs: Dict[str, Any],
    perf_mask: Optional[npt.NDArray] = None,
    within_perf_mask: bool = False,
    glm_alpha: float = 0.025,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize participation ratio, t-statistics and significance on brain surface.

    Creates three brain surface plots:
    1. Mean participation ratio for TD and ASD groups (ALWAYS full brain)
    2. T-statistics from group comparison (depends on within_perf_mask)
    3. Significant regions (FDR-corrected, depends on within_perf_mask)

    Args:
        pr_td: TD participation ratios (n_subs_td, 360)
        pr_asd: ASD participation ratios (n_subs_asd, 360)
        stat_results: Dictionary with 't', 'p', 'p_fdr' arrays (360,)
        brain_surfs: Dictionary from load_brain_surfaces()
        perf_mask: Performance mask (indices of significant parcels)
        within_perf_mask: Whether t-stat/significance should be masked
        glm_alpha: Significance threshold for FDR correction
        save_path: Path to save figures (optional)
    """
    if not brain_surfs.get('available', False):
        print("  Brain surfaces not available. Skipping visualization.")
        return

    surf_lh = brain_surfs['surf_lh']
    surf_rh = brain_surfs['surf_rh']
    label_atlas = brain_surfs['label_atlas']

    # Compute mean participation ratios (always full brain - 360 parcels)
    pr_td_mean = pr_td.mean(axis=0)  # (360,)
    pr_asd_mean = pr_asd.mean(axis=0)  # (360,)

    # Get t-statistics and significance
    t_mmp = stat_results['t']
    sig_mmp = (stat_results['p_fdr'] < glm_alpha).astype(float)

    # === Plot 1: Mean participation ratio (TD and ASD) - ALWAYS FULL BRAIN ===
    print("  Plotting mean participation ratio (TD vs ASD) - full brain...")
    map_list = []
    map_list.append(map_to_labels(pr_asd_mean, label_atlas, mask=label_atlas != 0, fill=np.nan))
    map_list.append(map_to_labels(pr_td_mean, label_atlas, mask=label_atlas != 0, fill=np.nan))

    cmap = 'cividis'
    cmax, cmin = 8.5, 5.0
    plot_hemispheres(
        surf_lh, surf_rh,
        array_name=map_list,
        size=(1200, 200 * len(map_list)),
        cmap=cmap,
        color_bar=True,
        color_range=(cmin, cmax),
        label_text=['ASD', 'TD'],
        zoom=1.5
    )

    # === Plot 2: T-statistics (depends on within_perf_mask) ===
    print("  Plotting t-statistics...")
    map_list = []

    if within_perf_mask and perf_mask is not None:
        # Mask t-statistics to performance mask
        t_mmp_masked = np.full(360, np.nan)
        t_mmp_masked[perf_mask] = t_mmp[perf_mask]
        map_list.append(map_to_labels(t_mmp_masked, label_atlas, mask=label_atlas != 0, fill=np.nan))
    else:
        map_list.append(map_to_labels(t_mmp, label_atlas, mask=label_atlas != 0, fill=np.nan))

    cmap = 'Spectral_r'
    cmax, cmin = 3, -3
    plot_hemispheres(
        surf_lh, surf_rh,
        array_name=map_list,
        size=(1200, 200 * len(map_list)),
        cmap=cmap,
        color_bar=True,
        color_range=(cmin, cmax),
        label_text=['T-stat'],
        zoom=1.5
    )

    # === Plot 3: Significance (depends on within_perf_mask) ===
    print("  Plotting significant regions...")
    map_list = []

    if within_perf_mask and perf_mask is not None:
        # Mask significance to performance mask
        sig_mmp_masked = np.full(360, np.nan)
        sig_mmp_masked[perf_mask] = sig_mmp[perf_mask]
        map_list.append(map_to_labels(sig_mmp_masked, label_atlas, mask=label_atlas != 0, fill=np.nan))
    else:
        map_list.append(map_to_labels(sig_mmp, label_atlas, mask=label_atlas != 0, fill=np.nan))

    cmap = 'Reds'
    cmax, cmin = 1, 0
    plot_hemispheres(
        surf_lh, surf_rh,
        array_name=map_list,
        size=(1200, 200 * len(map_list)),
        cmap=cmap,
        color_bar=True,
        color_range=(cmin, cmax),
        label_text=['Significant'],
        zoom=1.5
    )

    n_sig = int(sig_mmp.sum()) if not within_perf_mask else int(sig_mmp[perf_mask].sum())
    n_total = 360 if not within_perf_mask else len(perf_mask)
    print(f"  Significant parcels (FDR < {glm_alpha}): {n_sig}/{n_total}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def hex_to_rgba(
    hex_color: str,
    alpha: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Convert hex color to RGBA tuple.

    Args:
        hex_color: Hex color string (e.g., '#FF5733' or 'FF5733')
        alpha: Alpha channel value (0-1)

    Returns:
        RGBA tuple with values in [0, 1]
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert to RGB
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255

    return (r, g, b, alpha)


def plot_wordcloud_clusters(
    label_list: List[str],
    cluster_assignments: npt.NDArray,
    n_clusters: int,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate simple black wordclouds for each cluster.

    Args:
        label_list: List of semantic labels
        cluster_assignments: Cluster labels (n_features,)
        n_clusters: Number of clusters
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Calculate grid size
    ncols = min(n_clusters, 4)
    nrows = (n_clusters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for cluster_idx in range(n_clusters):
        # Get labels in this cluster
        cluster_mask = (cluster_assignments == cluster_idx + 1)
        cluster_labels = [label_list[i] for i in range(len(label_list)) if cluster_mask[i]]

        if len(cluster_labels) == 0:
            axes[cluster_idx].axis('off')
            continue

        # Create frequency dict (all equal frequency)
        freq_dict = {label: 1 for label in cluster_labels}

        # Generate wordcloud
        wc = WordCloud(
            width=400,
            height=400,
            background_color='white',
            color_func=lambda *args, **kwargs: 'black'
        ).generate_from_frequencies(freq_dict)

        # Plot
        axes[cluster_idx].imshow(wc, interpolation='bilinear')
        axes[cluster_idx].axis('off')
        axes[cluster_idx].set_title(f'Cluster {cluster_idx + 1} ({len(cluster_labels)} features)')

    # Hide extra subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig


def plot_radar(
    df: pd.DataFrame,
    column: str,
    value_max: Optional[float] = None,
    value_min: Optional[float] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot radar chart for network comparison.

    Args:
        df: DataFrame with networks as index
        column: Column name to plot
        value_max: Maximum value for axis
        value_min: Minimum value for axis
        title: Plot title

    Returns:
        Figure object
    """
    categories = list(df.index)
    values = df[column].values

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    values = list(values)

    # Complete the circle
    values += values[:1]
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set value range
    if value_min is not None and value_max is not None:
        ax.set_ylim(value_min, value_max)

    # Add title
    if title:
        ax.set_title(title, size=16, y=1.1)

    plt.tight_layout()

    return fig


# =============================================================================
# MAIN WORKFLOW FUNCTIONS
# =============================================================================

def load_data_and_subjects(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Load participants, filter by QC, separate TD/ASD groups, and extract label list.

    Args:
        config: Configuration dictionary
        paths: Path dictionary

    Returns:
        Tuple of (participants_df, id_list_td, id_list_asd, label_list)
    """
    # Load participants dataframe
    participants_file = paths['participants_df'] / 'participants_df_deepmreye_inc_byhx.xlsx'
    participants_df = pd.read_excel(participants_file, index_col=0)

    # === Apply filters in same order as filter_participants() ===

    # Filter #1: Diagnostic group (TD + ASD for dimensionality analysis)
    dx_filt = (participants_df['DX'].isin(['TD', 'ASD'])).values

    # Filter #2: ASD documentation (if by-history and hx_docu='True')
    if config['hx_docu'] == 'True':
        for sub_id in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                print(f"Exclude {sub_id} due to 'ASD_certainty' is 'by-history' "
                      f"and 'ASD_document' is not 'documentation provided'")
                dx_filt[participants_df.index == sub_id] = False

    # Filter #3: Preprocessing complete (MUST match 08_encoding_model_vertex!)
    prep_filt = (participants_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] == 1).values

    # Filter #4: No remarks during preprocessing
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Combine first 4 filters
    id_list_dx_prep_remark = list(participants_df[dx_filt & prep_filt & no_remarks_filt].index)
    participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

    # Filter #5: Framewise displacement
    fd_filt = (participants_df_dx_prep_remark['Mean_FD_DM'] < config['fd_thres']).values

    # Filter #6: DeepMReye quality rating
    deepmreye_filt = (participants_df_dx_prep_remark['Rating_deepmreye_movieDM'] == 1).values

    # Final filtered list
    id_list_all = list(participants_df_dx_prep_remark[np.multiply(fd_filt, deepmreye_filt)].index)

    # Separate TD and ASD
    participants_df_filtered = participants_df.loc[id_list_all]
    id_list_td = list(participants_df_filtered[participants_df_filtered['DX'] == 'TD'].index)
    id_list_asd = list(participants_df_filtered[participants_df_filtered['DX'] == 'ASD'].index)

    print(f"\nFiltered participants: {len(id_list_td)} TD + {len(id_list_asd)} ASD = {len(id_list_all)} total")

    # Load label list from regressor dataframe (same for all subjects with same parameters)
    # Use first TD subject to get label list
    if len(id_list_td) == 0:
        raise ValueError("No TD subjects found after filtering. Cannot load label list.")

    first_subject = id_list_td[0]

    # Construct path to regressor file based on CONFIG parameters
    if config['reg_wordnet']:
        subdir = 'wordnet_gaze_weighted_regressor'
    else:
        subdir = 'gaze_weighted_regressor'

    reg_file = (
        paths['reg'] /
        f"{subdir}_inc_byhx_docu-{config['hx_docu']}" / 'within' /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"{first_subject}_regressor_DM.xlsx"
    )

    print(f"Loading label list from: {reg_file}")

    if not reg_file.exists():
        raise FileNotFoundError(
            f"Regressor file not found: {reg_file}\n"
            f"Make sure to run 99_main/05_prepare_reg first."
        )

    # Load regressor dataframe and extract label list
    reg_dm_df = pd.read_excel(reg_file, index_col=0, engine='openpyxl')
    label_list = list(reg_dm_df.index)

    print(f"Loaded {len(label_list)} semantic features")

    # Load category frequency (for frequency regression preprocessing)
    print("Loading category frequency data...")
    freq_file = paths['raw'] / 'movie' / 'DM' / 'wordnet_sync_regressor_DM.xlsx'

    if not freq_file.exists():
        raise FileNotFoundError(
            f"Frequency file not found: {freq_file}\n"
            f"This file is required for weight preprocessing (frequency regression).\n"
            f"Make sure the file exists at the expected location."
        )

    freq_df = pd.read_excel(freq_file, index_col=0, engine='openpyxl')
    freq = freq_df['Frequency'].loc[np.array(label_list)].values

    print(f"Loaded frequency data for {len(freq)} features")

    # Build WordNet hypernym lookup dictionary
    print("Building WordNet hypernym lookup dictionary...")
    hypernym_indices = build_hypernym_lookup(label_list)
    print(f"Built hypernym lookup for {len(hypernym_indices)} features")

    return participants_df, id_list_td, id_list_asd, label_list, freq, hypernym_indices


def load_encoding_weights(
    id_list_td: List[str],
    id_list_asd: List[str],
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[List[npt.NDArray], List[npt.NDArray]]:
    """
    Load RAW encoding model weights from 08_encoding_model_vertex (with delays intact).

    Path structure matches output from 08_encoding_model_vertex_1optimal_alpha.py:
    {enc_results}/{conf_option}/{srm_type}_results_inc_byhx_docu-{hx_docu}/k-{srm_option_dim}/
      bias-{bias_weight}_verb-{verb_weight}/wn-{reg_wordnet}/seed-{seed}/s_alpha-{enc_single_alpha}/weight/{sub_id}.npy

    NOTE: Weights are returned RAW with delays intact. Averaging/max pooling across delays
    happens per-parcel in compute_dimensionality_all_parcels() during preprocessing.

    Args:
        id_list_td: List of TD subject IDs
        id_list_asd: List of ASD subject IDs
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        Tuple of (weight_list_td, weight_list_asd)
        Each weight array has shape (n_features * n_delays, n_vertices) - RAW weights
    """
    # Construct path to weights (matching 08_encoding_model_vertex_1optimal_alpha.py output structure)
    weight_base_path = (
        paths['enc_results'] / CONFIG['resolution'] / config['conf_option'] /
        f"{config['srm_type']}_results_inc_byhx_docu-{config['hx_docu']}" /
        f"k-{config['srm_option_dim']}" /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"wn-{str(config['reg_wordnet'])}" /
        f"seed-{config['seed']}" /
        f"s_alpha-{config['enc_single_alpha']}" / 'weight'
    )

    print(f"Loading weights from: {weight_base_path}")

    # Check if path exists
    if not weight_base_path.exists():
        raise FileNotFoundError(
            f"Weight path not found: {weight_base_path}\n"
            f"Expected structure: {{enc_results}}/{config['conf_option']}/"
            f"{config['srm_type']}_results_inc_byhx_docu-{config['hx_docu']}/k-{config['srm_option_dim']}/\n"
            f"bias-{config['bias_weight']}_verb-{config['verb_weight']}/"
            f"wn-{str(config['reg_wordnet'])}/seed-{config['seed']}/s_alpha-{config['enc_single_alpha']}/weight/\n"
            f"Make sure to run 08_encoding_model_vertex_1optimal_alpha.py first with matching CONFIG parameters."
        )

    # Load TD weights
    print("Loading TD weights...")
    weight_list_td_raw = []
    for sub_id in tqdm(id_list_td, desc="TD"):
        weight_file = weight_base_path / f'{sub_id}.npy'
        if weight_file.exists():
            # Load weight array directly (saved by save_files_sub, not wrapped in dictionary)
            weights = np.load(str(weight_file), allow_pickle=True)
            weight_list_td_raw.append(weights)
        else:
            print(f"Warning: Missing weights for {sub_id}")

    # Load ASD weights
    print("Loading ASD weights...")
    weight_list_asd_raw = []
    for sub_id in tqdm(id_list_asd, desc="ASD"):
        weight_file = weight_base_path / f'{sub_id}.npy'
        if weight_file.exists():
            # Load weight array directly (saved by save_files_sub, not wrapped in dictionary)
            weights = np.load(str(weight_file), allow_pickle=True)
            weight_list_asd_raw.append(weights)
        else:
            print(f"Warning: Missing weights for {sub_id}")

    # NOTE: Do NOT average/max across delays here!
    # Averaging/max pooling happens per-parcel in compute_dimensionality_all_parcels()
    # during the preprocessing pipeline (step 3a)
    weight_list_td = weight_list_td_raw
    weight_list_asd = weight_list_asd_raw

    print(f"Loaded weights: TD={len(weight_list_td)}, ASD={len(weight_list_asd)}")
    if len(weight_list_td) > 0:
        print(f"Weight shape (RAW with delays): {weight_list_td[0].shape}")

    return weight_list_td, weight_list_asd


def build_encoding_results_path(
    config: Dict[str, Any],
    paths: Dict[str, Path],
    use_perf_mask_base: bool = False
) -> Path:
    """
    Build path to encoding results based on configuration.

    Args:
        config: Configuration dictionary
        paths: Path dictionary
        use_perf_mask_base: If True, use perf_mask_results path (parcel-level),
                           else use enc_results path (vertex-level)

    Returns:
        Path to encoding results directory
    """
    # Choose base path (parcel-level for masks, vertex-level for weights)
    base_path = paths['perf_mask_results'] if use_perf_mask_base else paths['enc_results']

    base = (base_path / config['conf_option'] / config['atlas'] /
            f"chunk-{config['chunk_option']}" / 'fold-avg' /
            f"d-{config['hx_docu']}" / f"k-{config['srm_option']}")
    
    # if not config['bias_weight']:  # No gaze weight
    #     return (base / 'gaze_weight-False' /
    #             f"wn-{config['reg_wordnet']}" /
    #             f"seed-{config['seed']}" /
    #             f"s_alpha-{config['enc_single_alpha']}")
    # else:  # Gaze weight
    #     return (base /
    #             f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
    #             f"wn-{config['reg_wordnet']}" /
    #             f"seed-{config['seed']}" /
    #             f"s_alpha-{config['enc_single_alpha']}")

    if use_perf_mask_base:
        if not config['bias_weight']:  # No gaze weight
            return (base / 'gaze_weight-False' /
                    f"wn-{config['reg_wordnet']}" /
                    f"seed-{config['seed']}" /
                    f"s_alpha-{config['sem_enc_single_alpha']}")
        else:  # Gaze weight
            return (base /
                    f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                    f"wn-{config['reg_wordnet']}" /
                    f"seed-{config['seed']}" /
                    f"s_alpha-{config['sem_enc_single_alpha']}")
    else:
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

    results_path = build_encoding_results_path(config, paths, use_perf_mask_base=True)
    mask_path = results_path / f"mask_{config['perf_method']}-{config['perf_alpha']}.npy"

    perf_mask = np.load(mask_path)

    return perf_mask


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main execution workflow for dimensionality analysis.

    Workflow:
    1. Load and filter subjects (TD/ASD separation)
    2. Load atlases (MMP, Yeo, EG)
    3. Check for existing PR outputs
       - If exists: Load PR from disk, skip Steps 4-6
       - If not exists: Continue with Steps 4-6
    4. Load encoding model weights (SKIPPED if PR outputs exist)
    5. Load performance mask
    6. Compute participation ratio (360 parcels) (SKIPPED if PR outputs exist)
    7. Statistical comparison (BrainStat GLM) - ALWAYS runs
    8. Save outputs (if enabled) - ALWAYS runs
    9. Brain visualization - Visualize participation ratio (full brain), t-statistics
       and significance on brain surface (Windows only, requires brainspace)
    10. Network aggregation (EG17, MMP sections) - ALWAYS runs for visualization
    11. Similarity matrix heatmap visualization - Draw heatmap of similarity matrix
        reordered by hierarchical clustering for specified ROIs (SKIPPED if using cached PR)
    """
    print("="*80)
    print("Dimensionality Analysis: Participation Ratio (TD vs ASD)")
    print("="*80)
    print()

    # Print configuration
    print("Configuration:")
    print("-" * 40)
    for key, val in CONFIG.items():
        print(f"  {key:30s}: {val}")
    print()

    # Step 1: Load data and subjects
    print("="*80)
    print("Step 1: Loading data and subjects")
    print("="*80)
    participants_df, id_list_td, id_list_asd, label_list, freq, hypernym_indices = load_data_and_subjects(CONFIG, PATHS)
    print(f"  TD subjects: {len(id_list_td)}")
    print(f"  ASD subjects: {len(id_list_asd)}")
    print(f"  Semantic features: {len(label_list)}")
    print(f"  Frequency data: {len(freq)} features")
    print(f"  Hypernym lookup: {len(hypernym_indices)} features")
    print()

    # Step 2: Load atlases
    print("="*80)
    print("Step 2: Loading atlases")
    print("="*80)
    atlases = load_atlases(PATHS, CONFIG['resolution'])
    print("  Loaded atlases:", list(atlases.keys()))
    print(f"  Resolution: {CONFIG['resolution']}")
    print(f"  MMP atlas shape: {atlases['mmp_atlas'].shape}")
    print(f"  Brain mask: {len(atlases['brain_mask'])} vertices (from neuromaps-data)")
    print(f"  Expected vertices: {atlases['n_vertices']}")
    print()

    # Step 3: Check if outputs already exist (before loading weights)
    print("="*80)
    print("Step 3: Checking for existing outputs")
    print("="*80)

    # Build detailed output directory path (matching Step 8 save structure)
    out_res_dir = (
        PATHS['out'] /
        CONFIG['resolution'] /
        CONFIG['conf_option'] /
        f"{CONFIG['srm_type']}_results_inc_byhx_docu-{CONFIG['hx_docu']}" /
        f"k-{CONFIG['srm_option_dim']}" /
        f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
        f"wn-{CONFIG['reg_wordnet']}" /
        f"seed-{CONFIG['seed']}" /
        f"s_alpha-{CONFIG['enc_single_alpha']}"
    )
    pr_td_file = out_res_dir / f'{CONFIG["dim_metric"]}_pr_td.npy'
    pr_asd_file = out_res_dir / f'{CONFIG["dim_metric"]}_pr_asd.npy'

    outputs_exist = pr_td_file.exists() and pr_asd_file.exists()
    skip_computation = outputs_exist and not CONFIG['overwrite']

    if skip_computation:
        print("  Found existing PR outputs - loading from disk")
        print(f"  overwrite={CONFIG['overwrite']} - skipping recomputation")
        pr_td = np.load(str(pr_td_file))
        pr_asd = np.load(str(pr_asd_file))
        print(f"  PR TD shape: {pr_td.shape}")
        print(f"  PR ASD shape: {pr_asd.shape}")

        # Also load similarity matrices for heatmap visualization
        sim_td_pkl = out_res_dir / f'{CONFIG["dim_metric"]}_sim_mmp_td_avg_list.pkl'
        sim_asd_pkl = out_res_dir / f'{CONFIG["dim_metric"]}_sim_mmp_asd_avg_list.pkl'
        if sim_td_pkl.exists() and sim_asd_pkl.exists():
            with open(sim_td_pkl, 'rb') as f:
                sim_mmp_td_avg_list = pickle.load(f)
            with open(sim_asd_pkl, 'rb') as f:
                sim_mmp_asd_avg_list = pickle.load(f)
            print(f"  Loaded averaged similarity matrices ({len(sim_mmp_td_avg_list)} parcels)")
        else:
            sim_mmp_td_avg_list = None
            sim_mmp_asd_avg_list = None
            print("  No saved similarity matrices found - heatmap visualization will be skipped")

        print(f"  Skipping Steps 4-6 (loading weights, computing PR)")
        print(f"  Will proceed with Steps 7-8 (statistics, saving)")
        print()

        # Still need performance mask for network aggregation
        print("="*80)
        print("Step 4: Loading performance mask (for network aggregation)")
        print("="*80)
        perf_mask = load_performance_mask(CONFIG, PATHS)
        print()

    else:
        if outputs_exist:
            print("  Found existing PR outputs")
            print(f"  overwrite={CONFIG['overwrite']} - recomputing from scratch")
        else:
            print("  No existing outputs found - computing from scratch")
        print()

        # Step 4: Load encoding weights
        print("="*80)
        print("Step 4: Loading encoding model weights")
        print("="*80)
        weight_list_td, weight_list_asd = load_encoding_weights(
            id_list_td, id_list_asd, PATHS, CONFIG
        )
        print()

        # Step 5: Load performance mask
        print("="*80)
        print("Step 5: Loading performance mask")
        print("="*80)
        perf_mask = load_performance_mask(CONFIG, PATHS)
        print()

        # Step 6: Compute dimensionality
        print("="*80)
        print("Step 6: Computing participation ratio")
        print("="*80)

        # Create cache directory for parcel weights (resolution-specific to avoid mixing)
        cache_path = PATHS['tmp'] / f'parcel_weights_cache_{CONFIG["resolution"]}'

        # Use resolution-matched brain mask (encoding weights created with this mask)
        # NOTE: The reference implementation (12_rsa) uses different masks for SRM vs non-SRM,
        # but in 99_main pipeline, all encoding weights use the same neuromaps-data brain mask
        # (see load_brain_mask() in 08_encoding_model_vertex_0vertexlevel_srm.py)
        brain_mask = atlases['brain_mask']
        print(f"  Using {CONFIG['resolution']} brain mask from neuromaps-data ({len(brain_mask)} vertices)")
        print(f"  SRM option: {CONFIG['srm_option_dim']} (all weights use same mask)")

        pr_td, pr_asd, sim_mmp_td_avg_list, sim_mmp_asd_avg_list = compute_dimensionality_all_parcels(
            weight_list_td=weight_list_td,
            weight_list_asd=weight_list_asd,
            n_parcels=360,
            metric=CONFIG['dim_metric'],
            mmp_atlas=atlases['mmp_atlas'],         # Resolution-matched MMP atlas
            brain_mask=brain_mask,                   # Resolution-matched brain mask from neuromaps-data
            participants_df=participants_df,
            id_list_td=id_list_td,
            id_list_asd=id_list_asd,
            label_list=label_list,
            freq=freq,
            hypernym_indices=hypernym_indices,
            cache_path=cache_path,
            config=CONFIG,                           # Pass full config with resolution
            use_cache=CONFIG['use_weight_cache'] and not CONFIG['overwrite'],
            sim_save_path=out_res_dir                # Save similarity matrices to out_res_dir
        )
        print(f"  PR TD shape: {pr_td.shape}")
        print(f"  PR ASD shape: {pr_asd.shape}")
        print(f"  Mean PR (TD): {pr_td.mean():.3f} +/- {pr_td.std():.3f}")
        print(f"  Mean PR (ASD): {pr_asd.mean():.3f} +/- {pr_asd.std():.3f}")
        print(f"  PR range (TD): [{pr_td.min():.3f}, {pr_td.max():.3f}]")
        print(f"  PR range (ASD): [{pr_asd.min():.3f}, {pr_asd.max():.3f}]")
        print(f"  NOTE: PR values are now subject-specific (not broadcast)")

        # Save averaged similarity matrices (for heatmap visualization)
        out_res_dir.mkdir(parents=True, exist_ok=True)
        sim_td_pkl = out_res_dir / f'{CONFIG["dim_metric"]}_sim_mmp_td_avg_list.pkl'
        sim_asd_pkl = out_res_dir / f'{CONFIG["dim_metric"]}_sim_mmp_asd_avg_list.pkl'
        with open(sim_td_pkl, 'wb') as f:
            pickle.dump(sim_mmp_td_avg_list, f)
        with open(sim_asd_pkl, 'wb') as f:
            pickle.dump(sim_mmp_asd_avg_list, f)
        print(f"  Saved averaged similarity matrices to {out_res_dir}")
        print()

    # Step 7: Statistical comparison (always runs)
    print("="*80)
    print("Step 7: Statistical comparison (BrainStat)")
    print("="*80)

    if not CONFIG['within_perf_mask']:
        # Whole-brain analysis
        print("  Running whole-brain statistical comparison")
        stat_results = surfstat_dim(
            pr_td, pr_asd, id_list_td, id_list_asd, participants_df
        )
    else:
        # Within performance mask analysis
        print(f"  Running statistical comparison within performance mask")

        # Mask PR values to significant parcels only
        # Convert to list format for surfstat_dim
        pr_td_masked = [pr_td[i, perf_mask] for i in range(pr_td.shape[0])]
        pr_asd_masked = [pr_asd[i, perf_mask] for i in range(pr_asd.shape[0])]

        # Run statistical comparison on masked data
        stat_results_masked = surfstat_dim(
            pr_td_masked, pr_asd_masked, id_list_td, id_list_asd, participants_df
        )

        # Create full arrays initialized with zeros/ones
        stat_results = {
            't': np.zeros(360),
            'p': np.ones(360),
            'p_fdr': np.ones(360)
        }

        # Populate masked regions with statistical results
        stat_results['t'][perf_mask] = stat_results_masked['t']
        stat_results['p'][perf_mask] = stat_results_masked['p']
        stat_results['p_fdr'][perf_mask] = stat_results_masked['p_fdr']

    sig_parcels = (stat_results['p_fdr'] < CONFIG['glm_alpha']).sum()
    if not CONFIG['within_perf_mask']:
        print(f"  Significant parcels (FDR < {CONFIG['glm_alpha']}): {sig_parcels}/360")
        print(f"  Max t-stat: {stat_results['t'].max():.3f}")
        print(f"  Min t-stat: {stat_results['t'].min():.3f}")
    else:
        print(f"  Significant parcels within perf mask (FDR < {CONFIG['glm_alpha']}): {sig_parcels}/{len(perf_mask)}")
        print(f"  Max t-stat (within perf mask): {stat_results['t'][perf_mask].max():.3f}")
        print(f"  Min t-stat (within perf mask): {stat_results['t'][perf_mask].min():.3f}")

    # Whole-brain GLM over all 360 parcels. Consumed by the cross-dataset correspondence
    # test in 15_mcgill_perf_mask_10dimensionality.py Step 11: spin rotations move
    # whole-brain data, so parcels left at exactly 0.0 by a masked fit would be swept
    # into the comparison as if they were estimates.
    #
    # This does not perturb the masked results: surfstat_dim is mass-univariate, so t and
    # uncorrected p inside perf_mask are identical to the masked fit. Only FDR differs.
    if CONFIG['within_perf_mask']:
        print("  Refitting whole-brain GLM (360 parcels) for cross-dataset spatial nulls...")
        stat_results_wb = surfstat_dim(
            pr_td, pr_asd, id_list_td, id_list_asd, participants_df
        )
        print(f"    Whole-brain t range: [{stat_results_wb['t'].min():.3f}, "
              f"{stat_results_wb['t'].max():.3f}]")
        print(f"    Matches masked fit inside perf mask: "
              f"{np.allclose(stat_results_wb['t'][perf_mask], stat_results['t'][perf_mask])}")
    else:
        stat_results_wb = stat_results

    print()

    # Step 8: Save outputs (always runs if enabled)
    if CONFIG['save_outputs']:
        print("="*80)
        print("Step 8: Saving outputs")
        print("="*80)

        # Build detailed output directory path (matching 08_encoding_model_vertex structure)
        # Format: out/{resolution}/{conf_option}/{srm_type}_results_inc_byhx_docu-{hx_docu}/k-{srm_option_dim}/
        #         bias-{bias_weight}_verb-{verb_weight}/wn-{reg_wordnet}/seed-{seed}/s_alpha-{enc_single_alpha}/
        out_res_dir = (
            PATHS['out'] /
            CONFIG['resolution'] /
            CONFIG['conf_option'] /
            f"{CONFIG['srm_type']}_results_inc_byhx_docu-{CONFIG['hx_docu']}" /
            f"k-{CONFIG['srm_option_dim']}" /
            f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
            f"wn-{CONFIG['reg_wordnet']}" /
            f"seed-{CONFIG['seed']}" /
            f"s_alpha-{CONFIG['enc_single_alpha']}"
        )
        out_res_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(out_res_dir / f'{CONFIG["dim_metric"]}_pr_td.npy'), pr_td)
        np.save(str(out_res_dir / f'{CONFIG["dim_metric"]}_pr_asd.npy'), pr_asd)
        print(f"  Saved participation ratios to {out_res_dir}")

        # Save statistical results
        np.save(str(out_res_dir / f'{CONFIG["dim_metric"]}_stat_results.npy'), stat_results)
        print(f"  Saved statistical results")

        # Whole-brain GLM, saved alongside (never overwrites the masked results above).
        # Read by 15_mcgill_perf_mask_10dimensionality.py Step 11.
        np.save(str(out_res_dir / f'{CONFIG["dim_metric"]}_stat_results_wholebrain.npy'),
                stat_results_wb)
        print(f"  Saved whole-brain statistical results")

        # Save significance mask (boolean array for behavioral prediction in 10_behavior.py)
        sig_mmp = (stat_results['p_fdr'] < CONFIG['glm_alpha'])
        np.save(str(out_res_dir / f'{CONFIG["dim_metric"]}_sig_mmp.npy'), sig_mmp)
        print(f"  Saved significance mask ({sig_mmp.sum()} significant parcels)")

    else:
        print("="*80)
        print("Step 8: Skipping save (save_outputs=False)")
        print("="*80)
        print()

    print()

    # Step 9: Brain visualization (Windows only, requires brainspace)
    print("="*80)
    print("Step 9: Brain visualization")
    print("="*80)

    if CONFIG.get('make_figures', True):
        require_surface_plotting()
        print("  Loading brain surfaces...")
        brain_surfs = load_brain_surfaces(PATHS, atlas=CONFIG['atlas'])

        if brain_surfs.get('available', False):
            # Determine save path
            vis_save_path = PATHS['fig'] if CONFIG['save_outputs'] else None

            # Visualize participation ratio, t-statistics and significance
            visualize_participation_ratio(
                pr_td=pr_td,
                pr_asd=pr_asd,
                stat_results=stat_results,
                brain_surfs=brain_surfs,
                perf_mask=perf_mask,
                within_perf_mask=CONFIG['within_perf_mask'],
                glm_alpha=CONFIG['glm_alpha'],
                save_path=vis_save_path
            )
        else:
            print("  Brain surfaces could not be loaded. Skipping visualization.")
    else:
        if os.name != 'nt':
            print("  Skipped (not Windows - brainspace visualization requires Windows/VTK)")
        else:
            print("  Skipped (brainspace not available)")

    # Save participation ratio as 32k surface using roiarray_2_32knii
    if CONFIG['save_outputs'] and HAS_BRAINSPACE:
        print("  Saving participation ratio as 32k CIFTI surface...")

        # Compute mean participation ratios
        pr_td_mean = pr_td.mean(axis=0)  # (360,)
        pr_asd_mean = pr_asd.mean(axis=0)  # (360,)

        # Convert to 32k surface and save as CIFTI (.dscalar.nii)
        # Note: roiarray_2_32knii expects full 360-element array when mask length != roiarray length
        roiarray_2_32knii(
            roiarray=pr_td_mean,
            mask=np.arange(360),  # Full brain (no masking)
            atlas=CONFIG['atlas'],
            output_folder='participation_ratio',
            output_file=f'{CONFIG["dim_metric"]}_pr_td.dscalar.nii',
            fig_path=PATHS['fig']
        )
        roiarray_2_32knii(
            roiarray=pr_asd_mean,
            mask=np.arange(360),  # Full brain (no masking)
            atlas=CONFIG['atlas'],
            output_folder='participation_ratio',
            output_file=f'{CONFIG["dim_metric"]}_pr_asd.dscalar.nii',
            fig_path=PATHS['fig']
        )
        print(f"    Saved: {PATHS['fig'] / 'participation_ratio' / (CONFIG['dim_metric'] + '_pr_td.dscalar.nii')}")
        print(f"    Saved: {PATHS['fig'] / 'participation_ratio' / (CONFIG['dim_metric'] + '_pr_asd.dscalar.nii')}")
    elif CONFIG['save_outputs'] and not HAS_BRAINSPACE:
        print("  Skipped saving 32k CIFTI (brainspace not available)")

    # Save participation ratio as colored MMP dlabel.nii (Workbench-ready, any colormap)
    # Bakes the chosen matplotlib colormap into the parcel color table, so it does not
    # depend on brainspace/VTK and works even where the 32k CIFTI export above is skipped.
    if CONFIG['save_outputs']:
        print("  Saving participation ratio as 32k colored dlabel...")

        # --- Colormap settings (edit to change appearance in Workbench) ---
        pr_dlabel_cmap = 'viridis'   # any matplotlib colormap name
        pr_dlabel_vmin = 5.0
        pr_dlabel_vmax = 8.0
        # ------------------------------------------------------------------

        pr_td_mean = pr_td.mean(axis=0)    # (360,)
        pr_asd_mean = pr_asd.mean(axis=0)  # (360,)

        roiarray_2_32kdlabel(
            roiarray=pr_td_mean,
            cmap_name=pr_dlabel_cmap,
            vmin=pr_dlabel_vmin,
            vmax=pr_dlabel_vmax,
            output_folder='participation_ratio',
            output_file=f'{CONFIG["dim_metric"]}_pr_td.dlabel.nii',
            fig_path=PATHS['fig']
        )
        roiarray_2_32kdlabel(
            roiarray=pr_asd_mean,
            cmap_name=pr_dlabel_cmap,
            vmin=pr_dlabel_vmin,
            vmax=pr_dlabel_vmax,
            output_folder='participation_ratio',
            output_file=f'{CONFIG["dim_metric"]}_pr_asd.dlabel.nii',
            fig_path=PATHS['fig']
        )
        print(f"    Saved: {PATHS['fig'] / 'participation_ratio' / (CONFIG['dim_metric'] + '_pr_td.dlabel.nii')}")
        print(f"    Saved: {PATHS['fig'] / 'participation_ratio' / (CONFIG['dim_metric'] + '_pr_asd.dlabel.nii')}")

    print()

    # Step 10: Network-level aggregation
    print("="*80)
    print("Step 10: Network-level aggregation")
    print("="*80)

    pr_td_avg = pr_td.mean(axis=0)  # (360,)
    pr_asd_avg = pr_asd.mean(axis=0)

    # Network-level aggregation always uses whole-brain (360 parcels)
    # Pass perf_mask to zero out non-significant parcels before averaging
    # (independent of within_perf_mask setting which affects statistical comparison)

    # MMP sections
    if 'section_atlas' in atlases:
        pr_section_td = mmp_2_section(pr_td_avg, atlases['section_atlas'], atlases['mmp_atlas_32k'])
        pr_section_asd = mmp_2_section(pr_asd_avg, atlases['section_atlas'], atlases['mmp_atlas_32k'])
        pr_section_td_df = pd.DataFrame({
            'Section': MMP_SECTION_LIST,
            'Participation_Ratio_TD': pr_section_td
        }).set_index('Section')
        pr_section_asd_df = pd.DataFrame({
            'Section': MMP_SECTION_LIST,
            'Participation_Ratio_ASD': pr_section_asd
        }).set_index('Section')
        # sort by participation ratio in TD
        pr_section_td_df = pr_section_td_df.sort_values(by='Participation_Ratio_TD', ascending=False)
        pr_section_asd_df = pr_section_asd_df.loc[pr_section_td_df.index]  # match order
        print(f"  Section dimensionality computed (22 sections)")

    if atlases.get('eg_atlas') is not None:
        # Note: mmp_2_eg expects 64984-vertex mmp_atlas for surface mapping
        pr_eg17_td = mmp_2_eg(pr_td_avg, atlases['mmp_atlas_32k'], atlases['eg_atlas'])
        pr_eg17_asd = mmp_2_eg(pr_asd_avg, atlases['mmp_atlas_32k'], atlases['eg_atlas'])
        pr_eg17_td_df = pd.DataFrame({
            'Network': EG17_LIST,
            'Participation_Ratio_TD': pr_eg17_td
        }).set_index('Network')
        pr_eg17_asd_df = pd.DataFrame({
            'Network': EG17_LIST,
            'Participation_Ratio_ASD': pr_eg17_asd
        }).set_index('Network')
        # sort by participation ratio in TD
        pr_eg17_td_df = pr_eg17_td_df.sort_values(by='Participation_Ratio_TD', ascending=False)
        pr_eg17_asd_df = pr_eg17_asd_df.loc[pr_eg17_td_df.index]  # match order
        print(f"  EG17 dimensionality computed (17 networks)")

    # Visualize dimensionality using lollipop plot (section-level)
    if CONFIG['save_outputs'] and 'section_atlas' in atlases:
        print("  Creating lollipop plot for section-level dimensionality...")

        # Add color based on dimensionality
        cmap_name = 'viridis'
        cmin, cmax = 5, 8
        pr_section_td_df['color'] = pr_section_td_df['Participation_Ratio_TD'].apply(
            lambda x: value_to_rgb(x, cmap_name, cmin, cmax)
        )

        # Create lollipop plot
        fig, ax = plt.subplots(figsize=(13, 8))
        sections = list(pr_section_td_df.index)
        values = pr_section_td_df['Participation_Ratio_TD'].values
        colors = pr_section_td_df['color'].values

        ax.stem(range(len(sections)), values, linefmt='silver', markerfmt=' ', basefmt=' ')
        for i, (sec, val, col) in enumerate(zip(sections, values, colors)):
            ax.plot(i, val, marker='o', color=col, markersize=24)

        ax.set_xticks(range(len(sections)))
        ax.set_xticklabels(sections, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(4, 9.5)
        ax.set_xlabel('MMP section')
        ax.set_ylabel('Participation ratio')
        ax.set_title('Dimensionality (TD)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        save_path = PATHS['fig'] / 'network_aggregation' / f'lollipop_section_{CONFIG["dim_metric"]}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    # Analyze significant parcels based on EG17
    if CONFIG['save_outputs'] and atlases.get('eg_atlas') is not None:
        print("  Analyzing significant parcels for EG17 network visualization...")

        # Get significant parcels with direction (from t-values)
        sig_mask = stat_results['p_fdr'] < CONFIG['glm_alpha']
        sig_mmp = np.zeros(360)
        sig_mmp[sig_mask] = np.sign(stat_results['t'][sig_mask])  # +1 for ASD>TD, -1 for TD>ASD

        # Split into positive and negative
        sig_mmp_pos = sig_mmp.copy()
        sig_mmp_pos[sig_mmp_pos < 0] = 0
        sig_mmp_neg = sig_mmp.copy()
        sig_mmp_neg[sig_mmp_neg > 0] = 0

        # Aggregate to EG17 networks (sum of significant parcels)
        sig_eg17_pos = mmp_2_eg(sig_mmp_pos, atlases['mmp_atlas_32k'], atlases['eg_atlas'], sum_values=True)
        sig_eg17_neg = mmp_2_eg(sig_mmp_neg, atlases['mmp_atlas_32k'], atlases['eg_atlas'], sum_values=True)

        # Create DataFrames with sorted order
        eg17_list_sorted = [
            'Premotor', 'FootSM', 'HandSM', 'FaceSM', 'Auditory', 'FrontPar', 'CingOperc',
            'Salience', 'DorsAttn', 'AntMTL', 'PostMTL', 'MedVis', 'LatVis',
            'ParMemory', 'Context', 'Language', 'Default'
        ]

        sig_eg17_pos_df = pd.DataFrame({'value': sig_eg17_pos, 'abs_value': np.abs(sig_eg17_pos)}, index=EG17_LIST)
        sig_eg17_neg_df = pd.DataFrame({'value': sig_eg17_neg, 'abs_value': np.abs(sig_eg17_neg)}, index=EG17_LIST)
        sig_eg17_pos_df = sig_eg17_pos_df.reindex(index=eg17_list_sorted)
        sig_eg17_neg_df = sig_eg17_neg_df.reindex(index=eg17_list_sorted)

        # Color mapping
        cmap_name = 'Spectral_r'
        cmin, cmax = -1, 1
        sig_eg17_pos_df['color'] = sig_eg17_pos_df['value'].apply(lambda x: value_to_rgb(x, cmap_name, cmin, cmax))
        sig_eg17_neg_df['color'] = sig_eg17_neg_df['value'].apply(lambda x: value_to_rgb(x, cmap_name, cmin, cmax))

        # === Lollipop plot for EG17 significant parcels ===
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.stem(range(len(eg17_list_sorted)), sig_eg17_pos_df['abs_value'].values, linefmt='silver', markerfmt=' ', basefmt=' ')
        ax.stem(range(len(eg17_list_sorted)), sig_eg17_neg_df['abs_value'].values, linefmt='silver', markerfmt=' ', basefmt=' ')

        for i, network in enumerate(eg17_list_sorted):
            ax.plot(i, sig_eg17_pos_df.loc[network, 'abs_value'], marker='o',
                   color=sig_eg17_pos_df.loc[network, 'color'], markersize=24)
            ax.plot(i, sig_eg17_neg_df.loc[network, 'abs_value'], marker='o',
                   color=sig_eg17_neg_df.loc[network, 'color'], markersize=24)

        ax.set_xticks(range(len(eg17_list_sorted)))
        ax.set_xticklabels(eg17_list_sorted, rotation=45, ha='right')
        ax.set_title('Significant parcels (EG17 networks)')
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        save_path = PATHS['fig'] / 'network_aggregation' / f'lollipop_eg17_significant_{CONFIG["dim_metric"]}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved lollipop plot to {save_path}")

        # === Radar plot for EG17 significant parcels ===
        from math import pi

        N = len(eg17_list_sorted)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles_c = np.r_[angles, angles[0]]

        pos_values = sig_eg17_pos_df['abs_value'].values.tolist()
        neg_values = sig_eg17_neg_df['abs_value'].values.tolist()
        pos_values_c = np.r_[pos_values, pos_values[0]]
        neg_values_c = np.r_[neg_values, neg_values[0]]

        pos_color = plt.cm.Spectral_r(0.99)
        neg_color = plt.cm.Spectral_r(0.01)

        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)

        ax.vlines(angles, 0, pos_values, linewidth=3.0, color=pos_color, alpha=0.9)
        ax.vlines(angles, 0, neg_values, linewidth=3.0, color=neg_color, alpha=0.9)
        ax.fill(angles_c, pos_values_c, color=pos_color, alpha=0.4, zorder=2)
        ax.fill(angles_c, neg_values_c, color=neg_color, alpha=0.4, zorder=2)
        ax.scatter(angles, pos_values, s=750, color=pos_color, zorder=4)
        ax.scatter(angles, neg_values, s=750, color=neg_color, zorder=4)
        ax.scatter(0, 0, s=1000, color='black', zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels([])

        max_value = 20 
        ax.set_ylim(0, max_value)
        plt.yticks([0, 5, 10, 15, 20], [])
        plt.title('Significant parcels - EG17 networks (Positive vs Negative)', pad=20)
        plt.tight_layout()

        save_path = PATHS['fig'] / 'network_aggregation' / f'radar_eg17_significant_{CONFIG["dim_metric"]}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved radar plot to {save_path}")

        # === Save t-statistics with significance mask as dlabel.nii ===
        # Non-significant regions are blended toward white for visual distinction
        print("  Saving t-statistics with significance mask as dlabel.nii...")

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

        # Get t-statistics from stat_results
        tstat_results = stat_results['t']
 
        for mmp_idx in range(360):
            # Get t-statistic color
            rgba_tuple = tstat_cmap(tstat_norm(tstat_results[mmp_idx]))[:3]

            if sig_mmp[mmp_idx] == 0:
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
        tstat_masked_dlabel_path = PATHS['fig'] / 'participation_ratio' / f'tstat_{CONFIG["dim_metric"]}_sig_masked.dlabel.nii'
        tstat_masked_dlabel_path.parent.mkdir(parents=True, exist_ok=True)
        dlabel_tstat_masked_cifti.to_filename(str(tstat_masked_dlabel_path))
        print(f"    Saved t-statistics with significance mask: {tstat_masked_dlabel_path.name}")

    print()

    # Step 11: Similarity matrix heatmap visualization
    print("="*80)
    print("Step 11: Similarity matrix heatmap visualization")
    print("="*80)
    print("* Draw heatmap of similarity matrix reordered by hierarchical clustering")

    # ROIs to visualize (can be customized)
    vis_sim_rois = ['R_V1_ROI']  # Options: 'L_TPOJ1_ROI', 'L_MT_ROI', 'R_V1_ROI', 'R_V2_ROI'
    eval_methods = ['silhouette']  # Options: None (all methods), 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap', 'elbow'

    # Create cleaned label list (remove '.n.XX' or '.v.XX' suffixes for display)
    new_label_list = [label.split('.')[0] for label in label_list]

    # Use loaded or computed similarity matrices (works both when computed or cached)
    if sim_mmp_td_avg_list is not None:
        # Load MMP info to get ROI indices
        mmp_info_path = PATHS['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
        mmp_info_df = pd.read_excel(mmp_info_path, engine='openpyxl')
        mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]

        # Get ROI indices for all vis_sim_rois
        vis_sim_roi_idx = []
        for roi in vis_sim_rois:
            roi_matches = mmp_info_df[mmp_info_df['ROI'] == roi]
            if len(roi_matches) > 0:
                vis_sim_roi_idx.append(roi_matches.index[0])
            else:
                print(f"  Warning: ROI '{roi}' not found in MMP atlas")

        if len(vis_sim_roi_idx) == 0:
            print("  No valid ROIs found. Skipping heatmap visualization.")
        else:
            # Print dimensionality of the vis_sim_rois
            pr_td_mean = pr_td.mean(axis=0)  # (360,)
            print(f"  Dimensionality of the vis_sim_roi: {pr_td_mean[vis_sim_roi_idx]}")

            # Visualize the selected ROIs on brain surface (Windows only)
            if CONFIG.get('make_figures', True):
                require_surface_plotting()
                brain_surfs = load_brain_surfaces(PATHS, atlas=CONFIG['atlas'])
                if brain_surfs.get('available', False):
                    dim_vis_mmp = np.zeros(360, dtype=np.float32)
                    dim_vis_mmp[vis_sim_roi_idx] = pr_td_mean[vis_sim_roi_idx]
                    sig_regions_vis_32k = map_to_labels(
                        dim_vis_mmp, brain_surfs['label_atlas'],
                        mask=brain_surfs['label_atlas'] != 0, fill=np.nan
                    )
                    cmap = 'viridis'
                    cmax, cmin = 7, 3
                    plot_hemispheres(
                        brain_surfs['surf_lh'], brain_surfs['surf_rh'],
                        array_name=sig_regions_vis_32k,
                        size=(1100, 500), layout_style='grid',
                        cmap=cmap, color_bar=True, color_range=(cmin, cmax), zoom=1.3
                    )

            # Get similarity matrix averaged across selected ROIs
            sim_matrices_td = [sim_mmp_td_avg_list[idx] for idx in vis_sim_roi_idx if sim_mmp_td_avg_list[idx] is not None]
            sim_matrices_asd = [sim_mmp_asd_avg_list[idx] for idx in vis_sim_roi_idx if sim_mmp_asd_avg_list[idx] is not None]

            if len(sim_matrices_td) == 0:
                print(f"  Warning: No similarity matrices available for selected ROIs (parcel below vertex threshold)")
            else:
                # Average across ROIs if multiple
                sim_td_avg = np.array(sim_matrices_td).mean(0)
                sim_asd_avg = np.array(sim_matrices_asd).mean(0)

                # Create DataFrame for visualization with cleaned labels
                sim_td_df = pd.DataFrame(sim_td_avg, index=new_label_list, columns=new_label_list)

                # Run hierarchical clustering analysis and plot heatmap
                print(f"  Running hierarchical clustering analysis...")
                cluster_results = run_hierarchical_clustering_analysis(
                    sim_td_df,
                    max_clusters=CONFIG['max_clusters'],
                    min_clusters=CONFIG['min_clusters'],
                    methods=eval_methods,
                    visualize=True
                )

                # Save heatmap figure
                if CONFIG['save_outputs']:
                    roi_names_str = ','.join(vis_sim_rois)
                    heatmap_save_path = PATHS['fig'] / 'similarity_heatmap' / f'{roi_names_str}_{CONFIG["dim_metric"]}_TD.png'
                    heatmap_save_path.parent.mkdir(parents=True, exist_ok=True)
                    heatmap_results = plot_similarity_heatmap_clustered(
                        sim_td_df,
                        n_clusters=cluster_results['optimal_k'],
                        max_clusters=CONFIG['max_clusters'],
                        min_clusters=CONFIG['min_clusters'],
                        cmap='viridis',
                        cmax=0.8, cmin=-0.1,
                        figsize=(12, 12),
                        title=f'TD Group Similarity Matrix - {roi_names_str}\n(k={cluster_results["optimal_k"]} clusters)',
                        save_path=heatmap_save_path,
                        show_labels=len(new_label_list) <= 100
                    )
                    print(f"  Saved TD heatmap to {heatmap_save_path}")

                    # Also save ASD heatmap for comparison
                    sim_asd_df = pd.DataFrame(sim_asd_avg, index=new_label_list, columns=new_label_list)
                    heatmap_save_path_asd = PATHS['fig'] / 'similarity_heatmap' / f'{roi_names_str}_{CONFIG["dim_metric"]}_ASD.png'
                    heatmap_results_asd = plot_similarity_heatmap_clustered(
                        sim_asd_df,
                        n_clusters=cluster_results['optimal_k'],  # Use same k as TD for comparison
                        max_clusters=CONFIG['max_clusters'],
                        min_clusters=CONFIG['min_clusters'],
                        cmap='viridis',
                        cmax=0.8, cmin=-0.1,
                        figsize=(12, 12),
                        title=f'ASD Group Similarity Matrix - {roi_names_str}\n(k={cluster_results["optimal_k"]} clusters)',
                        save_path=heatmap_save_path_asd,
                        show_labels=len(new_label_list) <= 100
                    )
                    print(f"  Saved ASD heatmap to {heatmap_save_path_asd}")
    else:
        print("  Skipped (no similarity matrices available)")
        print("  Run with overwrite=True or ensure similarity matrices are saved during computation")

    print()

    # Step 12: Compare dimensionality with multitask representations
    print("="*80)
    print("Step 12: Compare dimensionality with multitask representations")
    print("="*80)

    # Compute average semantic dimensionality across TD subjects
    dim_sem_mmp = pr_td.mean(axis=0)  # (360,)

    # Get indices with zero semantic dimensionality (small parcels)
    zero_sem_idx = np.where(dim_sem_mmp == 0)[0]
    dim_sem_nonzero = np.delete(dim_sem_mmp, zero_sem_idx)
    print(f"  N parcels with zero PR: {len(zero_sem_idx)}")
    print(f"  N parcels for comparison: {len(dim_sem_nonzero)}")

    # Load multitask dimensionality
    multitask_csv = PATHS['code'].parent / 'multitaskrepresentations-main' / \
        'processed_data/analysis1/analysis1_parcel_cosine_dimensionality_groupavg.csv'
    if multitask_csv.exists():
        dim_multitask = pd.read_csv(multitask_csv)['Dimensionality'].values
        dim_multitask_nonzero = np.delete(dim_multitask, zero_sem_idx)

        # Compute correlation
        r, p = pearsonr(dim_sem_nonzero, dim_multitask_nonzero)
        print(f"  Correlation btw semantic and multitask dimensionality: r={r:.4f}, p={p:.4f}")

        # Spatial null models. np.delete() collapses the arrays, so parcel identity is
        # lost - rebuild full 360-parcel maps with NaN at the dropped parcels and let
        # the rotation restrict afterwards, since spinning is only defined over the
        # whole cortex.
        nonzero_idx = np.setdiff1d(np.arange(360), zero_sem_idx)
        dim_sem_full = np.full(360, np.nan)
        dim_multitask_full = np.full(360, np.nan)
        dim_sem_full[nonzero_idx] = dim_sem_nonzero
        dim_multitask_full[nonzero_idx] = dim_multitask_nonzero

        null_row = snull.compare_maps(
            dim_sem_full, dim_multitask_full,
            label='dimensionality_semantic_vs_multitask',
            mask=nonzero_idx,
            tpl_path=PATHS['tpl'],
            n_rot=CONFIG['spin_n_rot'], n_surr=CONFIG['es_n_surr'],
            num_modes=CONFIG['es_num_modes'], surface=CONFIG['es_surface'],
            seed=CONFIG['seed_surro'], n_jobs=CONFIG['es_n_jobs'],
        )
        spin_p = null_row['spin_p']
        eigen_p = null_row['eigenstrap_p']
        print(f"  Spin permutation p={spin_p:.4e} (one-sided)")
        print(f"  Eigenstrapping   p={eigen_p:.4e} (one-sided)")

        if CONFIG['save_outputs']:
            snull.save_sensitivity_table(
                pd.DataFrame([null_row]), PATHS['fig'] / 'sensitivity',
                'null_model_comparison_semantic_vs_multitask',
                also_excel=CONFIG['sens_save_excel'])

        # Save scatter plot
        if CONFIG['save_outputs']:
            fig, ax = plt.subplots(figsize=(12, 12))
            norm = plt.Normalize(3, 10)
            cmap = plt.get_cmap('viridis')
            sns.regplot(x=dim_sem_nonzero, y=dim_multitask_nonzero, color='black', ax=ax,
                        scatter_kws={'s': 700, 'color': cmap(norm(dim_sem_nonzero)), 'alpha': 1},
                        line_kws={'linestyle': '--'})
            ax.set_xlabel('Semantic dimensionality', fontsize=20)
            ax.set_ylabel('Multitask dimensionality', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            plt.title(f'r: {r:.4f}, p: {p:.4f}, Spin p: {spin_p:.4e}', fontsize=20)
            plt.tight_layout()

            save_path = PATHS['fig'] / 'multitask_comparison' / f'semantic_vs_multitask_{CONFIG["dim_metric"]}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved scatter plot to {save_path}")

        # Save multitask dimensionality as colored MMP dlabel.nii (same approach as PR maps)
        if CONFIG['save_outputs']:
            print("  Saving multitask dimensionality as 32k colored dlabel...")
            print(f"    Multitask dimensionality range: "
                  f"[{np.nanmin(dim_multitask):.2f}, {np.nanmax(dim_multitask):.2f}]")

            # --- Colormap settings (edit to change appearance in Workbench) ---
            mt_dlabel_cmap = 'viridis'   # any matplotlib colormap name
            mt_dlabel_vmin = 0.2
            mt_dlabel_vmax = 1.2
            # ------------------------------------------------------------------

            # Grey out the small parcels excluded from the comparison (zero semantic PR),
            # so the map aligns with the semantic dimensionality dlabel
            dim_multitask_masked = dim_multitask.astype(float).copy()
            dim_multitask_masked[zero_sem_idx] = np.nan

            roiarray_2_32kdlabel(
                roiarray=dim_multitask_masked,
                cmap_name=mt_dlabel_cmap,
                vmin=mt_dlabel_vmin,
                vmax=mt_dlabel_vmax,
                output_folder='multitask_comparison',
                output_file='multitask_dimensionality.dlabel.nii',
                fig_path=PATHS['fig']
            )
            print(f"    Saved: {PATHS['fig'] / 'multitask_comparison' / 'multitask_dimensionality.dlabel.nii'}")
    else:
        print(f"  Multitask dimensionality file not found: {multitask_csv}")
        print("  Skipping comparison")

    print()

    # Step 13: Group into unimodal and transmodal areas and compare TD vs ASD
    print("="*80)
    print("Step 13: Group into unimodal and transmodal areas")
    print("="*80)

    # Define unimodal and transmodal MMP sections
    unimodal_sections = [
        'Primary visual (V1)', 'Early visual', 'Posterior opercular',
        'Somatosensory/motor', 'Paracentral lobular and mid cingulate',
        'Early auditory', 'MT+ complex and neighboring visual', 'Premotor'
    ]
    transmodal_sections = [s for s in MMP_SECTION_LIST if s not in unimodal_sections]

    print(f"  Unimodal sections ({len(unimodal_sections)}): {unimodal_sections}")
    print(f"  Transmodal sections ({len(transmodal_sections)}): {transmodal_sections}")

    # Aggregate PR to sections for each subject
    # pr_td: (n_subjects, 360) -> need to aggregate to (n_subjects, 22)
    # Using mmp_2_section which can handle 2D arrays (will transpose as needed)
    if 'section_atlas' in atlases:
        # Transpose to get (360, n_subjects) for mmp_2_section, then transpose back
        dim_section_td = mmp_2_section(pr_td.T, atlases['section_atlas'], atlases['mmp_atlas_32k']).T  # (n_subjects, 22)
        dim_section_asd = mmp_2_section(pr_asd.T, atlases['section_atlas'], atlases['mmp_atlas_32k']).T  # (n_subjects, 22)

        # Get section indices
        unimodal_ids = [MMP_SECTION_LIST.index(s) for s in unimodal_sections if s in MMP_SECTION_LIST]
        transmodal_ids = [MMP_SECTION_LIST.index(s) for s in transmodal_sections if s in MMP_SECTION_LIST]

        # Calculate mean dimensionality for unimodal/transmodal per subject
        dim_unimodal_td = dim_section_td[:, unimodal_ids].mean(axis=1)  # (n_subjects_td,)
        dim_unimodal_asd = dim_section_asd[:, unimodal_ids].mean(axis=1)  # (n_subjects_asd,)
        dim_transmodal_td = dim_section_td[:, transmodal_ids].mean(axis=1)
        dim_transmodal_asd = dim_section_asd[:, transmodal_ids].mean(axis=1)

        # T-tests comparing ASD vs TD
        t_unimodal, p_unimodal = ttest_ind(dim_unimodal_asd, dim_unimodal_td)
        t_transmodal, p_transmodal = ttest_ind(dim_transmodal_asd, dim_transmodal_td)

        print(f"  Unimodal: t={t_unimodal:.4f}, p={p_unimodal:.2e}")
        print(f"  Transmodal: t={t_transmodal:.4f}, p={p_transmodal:.2e}")

        # Create violin plot with stripplot
        if CONFIG['save_outputs']:
            print("  Creating violin plot for unimodal vs transmodal comparison...")

            # Prepare data for seaborn
            sns_df = pd.DataFrame({
                'Dimensionality': np.concatenate([
                    dim_unimodal_td, dim_unimodal_asd,
                    dim_transmodal_td, dim_transmodal_asd
                ]),
                'Group': (
                    ['Unimodal (TD)'] * len(dim_unimodal_td) +
                    ['Unimodal (ASD)'] * len(dim_unimodal_asd) +
                    ['Transmodal (TD)'] * len(dim_transmodal_td) +
                    ['Transmodal (ASD)'] * len(dim_transmodal_asd)
                )
            })

            # Sort order
            sort_order = ['Transmodal (ASD)', 'Transmodal (TD)', 'Unimodal (ASD)', 'Unimodal (TD)']
            sns_df['Group'] = pd.Categorical(sns_df['Group'], categories=sort_order, ordered=True)
            sns_df = sns_df.sort_values('Group')

            # Create figure
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_title('Dimensionality (TD vs. ASD)')
            ax.set_xlabel('MMP section')
            ax.set_ylabel('Participation ratio')
            ax.set_ylim(5, 9)

            # Color palettes
            pastel_palette = sns.color_palette("pastel")
            pastel_pink = pastel_palette[3]
            pastel_blue = pastel_palette[0]
            violin_palette = {
                'Unimodal (ASD)': pastel_pink, 'Unimodal (TD)': pastel_pink,
                'Transmodal (ASD)': pastel_blue, 'Transmodal (TD)': pastel_blue
            }
            strip_palette = {
                'Unimodal (ASD)': 'darkmagenta', 'Unimodal (TD)': 'darkmagenta',
                'Transmodal (ASD)': 'navy', 'Transmodal (TD)': 'navy'
            }

            sns.violinplot(x='Group', y='Dimensionality', data=sns_df, inner=None, palette=violin_palette, ax=ax)
            sns.stripplot(x='Group', y='Dimensionality', data=sns_df, jitter=0.1, size=5, alpha=0.7, palette=strip_palette, ax=ax)
            ax.set_yticks(np.arange(5, 10, 1))
            sns.despine()
            plt.tight_layout()

            save_path = PATHS['fig'] / 'unimodal_transmodal' / f'violin_unimodal_transmodal_{CONFIG["dim_metric"]}.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved violin plot to {save_path}")
    else:
        print("  Skipped (section_atlas not available)")

    print()

    # =========================================================================
    # Step 14: Overlapping parcels with semantic map
    # =========================================================================
    print("="*80)
    print("Step 14: Overlapping parcels with semantic map")
    print("="*80)

    # Build semantic map results path (from 99_main/07_pca.py output)
    # Note: Semantic map uses k-0 SRM and enc_single_alpha=True (different from this script)
    sem_path = (
        PATHS['pipe'] / '07_pca' / 'out' /
        CONFIG['conf_option'] / CONFIG['atlas'] /
        f"chunk-{CONFIG['chunk_option']}" / 'fold-avg' / 'results' /
        f"k-{CONFIG['srm_option']}" /
        f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
        f"wn-{CONFIG['reg_wordnet']}" / f"seed-{CONFIG['seed']}" /
        f"s_alpha-{CONFIG['sem_enc_single_alpha']}" / 'sem' /
        f"mask_{CONFIG['perf_method']}-{CONFIG['perf_alpha']}" /
        f"delay-{CONFIG['weight_across_delays']}_super-{CONFIG['weight_add_superordinate']}"
    )

    # Load semantic map significant parcels
    sem_sig_file = sem_path / f"sig_regions_{CONFIG['pca_template']}_sign-{CONFIG['pca_only_sign']}_iter-{CONFIG['pca_iter']}.npy"

    if sem_sig_file.exists():
        print(f"  Loading semantic map results from: {sem_path}")
        sig_mmp_sr = np.load(str(sem_sig_file))

        # Load p-values for lenient thresholding
        pval_file = sem_path / f"pval_{CONFIG['pca_template']}_sign-{CONFIG['pca_only_sign']}_iter-{CONFIG['pca_iter']}.npy"
        tstat_file = sem_path / f"tstat_{CONFIG['pca_template']}_sign-{CONFIG['pca_only_sign']}_iter-{CONFIG['pca_iter']}.npy"

        if pval_file.exists():
            pval_sr = np.load(str(pval_file))
            tstat_sr = np.load(str(tstat_file))

            # Apply lenient threshold for overlap detection (semantic map)
            sig_mask_sr_lenient = pval_sr < CONFIG['overlap_alpha']
            sig_mmp_sr_lenient = np.zeros(360)
            sig_mmp_sr_lenient[sig_mask_sr_lenient] = np.sign(tstat_sr[sig_mask_sr_lenient])
        else:
            # Fallback to pre-computed sig_regions if p-values not available
            print(f"  Warning: p-values not found, using pre-computed sig_regions at glm_alpha=0.025")
            sig_mmp_sr_lenient = sig_mmp_sr

        # Get dimensionality significant parcels with STRICT threshold (for comparison)
        sig_mask_strict = stat_results['p_fdr'] < CONFIG['glm_alpha']
        sig_mmp_strict = np.zeros(360)
        sig_mmp_strict[sig_mask_strict] = np.sign(stat_results['t'][sig_mask_strict])

        # Get dimensionality significant parcels with LENIENT threshold
        sig_mask_lenient = stat_results['p_fdr'] < CONFIG['overlap_alpha']
        sig_mmp_lenient = np.zeros(360)
        sig_mmp_lenient[sig_mask_lenient] = np.sign(stat_results['t'][sig_mask_lenient])

        # Calculate overlap using STRICT thresholds (original behavior)
        sig_mmp_overlap_strict = sig_mmp_strict * sig_mmp_sr
        n_overlap_strict = (sig_mmp_overlap_strict != 0).sum()
        print(f"  Overlapping parcels (strict alpha={CONFIG['glm_alpha']}): {n_overlap_strict}")

        # Calculate overlap using LENIENT thresholds
        sig_mmp_overlap = sig_mmp_lenient * sig_mmp_sr_lenient
        n_overlap = (sig_mmp_overlap != 0).sum()
        print(f"  Overlapping parcels (lenient alpha={CONFIG['overlap_alpha']}): {n_overlap}")

        # Aggregate to EG17 networks
        if CONFIG['save_outputs'] and atlases.get('eg_atlas') is not None:
            sig_overlap_eg17 = mmp_2_eg(
                sig_mmp_overlap,
                atlases['mmp_atlas_32k'],
                atlases['eg_atlas'],
                sum_values=True
            )

            # Create DataFrame with sorted network order
            sig_overlap_eg17_df = pd.DataFrame({
                'Overlap': sig_overlap_eg17,
                'Overlap_abs': np.abs(sig_overlap_eg17)
            }, index=EG17_LIST)
            sig_overlap_eg17_df = sig_overlap_eg17_df.reindex(index=EG17_LIST_SORTED)

            # Color mapping (Spectral_r: blue for negative, red for positive)
            cmap_name = 'Spectral_r'
            cmin, cmax = -1, 1
            sig_overlap_eg17_df['color'] = sig_overlap_eg17_df['Overlap'].apply(
                lambda x: value_to_rgb(x, cmap_name, cmin, cmax)
            )

            # === Lollipop plot for EG17 overlap ===
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.stem(
                range(len(EG17_LIST_SORTED)),
                sig_overlap_eg17_df['Overlap_abs'].values,
                linefmt='silver',
                markerfmt=' ',
                basefmt=' '
            )

            for i, network in enumerate(EG17_LIST_SORTED):
                ax.plot(
                    i,
                    sig_overlap_eg17_df.loc[network, 'Overlap_abs'],
                    marker='o',
                    color=sig_overlap_eg17_df.loc[network, 'color'],
                    markersize=22
                )

            ax.set_xticks(range(len(EG17_LIST_SORTED)))
            ax.set_xticklabels(EG17_LIST_SORTED, rotation=45, ha='right')
            ax.set_title('Overlap: Dimensionality x Semantic Map (EG17 networks)')
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()

            save_path = PATHS['fig'] / 'overlap' / f"lollipop_eg17_overlap_{CONFIG['dim_metric']}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved EG17 overlap lollipop plot to {save_path}")

        # === Print ROI name of sig_mmp_overlap ===
        print()
        print("  Print ROI name of sig_mmp_overlap")

        # Load MMP info and semantic map t-statistics
        mmp_info_path = PATHS['tpl'] / 'MMP' / 'HCP_cortical_subcortical_379.xlsx'
        mmp_info_df = pd.read_excel(mmp_info_path, engine='openpyxl')
        mmp_info_df = mmp_info_df.loc[mmp_info_df.index < 360]

        tstat_file = sem_path / f"tstat_{CONFIG['pca_template']}_sign-{CONFIG['pca_only_sign']}_iter-{CONFIG['pca_iter']}.npy"
        t_mmp_sr = np.load(str(tstat_file)) if tstat_file.exists() else np.zeros(360)
        t_mmp = stat_results['t']

        # Get mappings
        mapping_mmp_2_yeo17 = atlases.get('mapping_mmp_2_yeo17', np.zeros(360))
        mapping_mmp_2_eg = atlases.get('mapping_mmp_2_eg', np.zeros(360))

        for i in range(360):
            if sig_mmp_overlap[i] != 0:
                yeo17_idx = int(mapping_mmp_2_yeo17[i] - 1) if mapping_mmp_2_yeo17[i] > 0 else 0
                eg17_idx = int(mapping_mmp_2_eg[i] - 1) if mapping_mmp_2_eg[i] > 0 else 0
                yeo17_name = YEO17_LIST[yeo17_idx] if yeo17_idx < len(YEO17_LIST) else 'Unknown'
                eg17_name = EG17_LIST[eg17_idx] if eg17_idx < len(EG17_LIST) else 'Unknown'
                avg_t = np.mean([np.abs(t_mmp_sr[i]), np.abs(t_mmp[i])])
                print(f"  [{mmp_info_df.loc[i, 'ROI']}] pc1:{t_mmp_sr[i]:.2f}, dim:{t_mmp[i]:.2f}, avg:{avg_t:.2f} ({yeo17_name}) ({eg17_name})")

        # === Compare similarity matrix between TD and ASD - ROI (parcel) ===
        print()
        print("  Compare similarity matrix between TD and ASD - ROI (parcel)")

        vis_roi_list = ['R_STSda_ROI']
        vis_roi_idx = [mmp_info_df[mmp_info_df['ROI'] == roi].index[0] for roi in vis_roi_list if roi in mmp_info_df['ROI'].values]

        if len(vis_roi_idx) > 0:
            print(f"  Selected ROIs: {vis_roi_list}")
            results_path = (
                PATHS['out'] / CONFIG['conf_option'] / CONFIG['atlas'] /
                f"chunk-{CONFIG['chunk_option']}" / 'fold-avg' / 'results' /
                f"k-{CONFIG['srm_option_dim']}" /
                f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
                f"wn-{CONFIG['reg_wordnet']}" / f"s_alpha-{CONFIG['enc_single_alpha']}"
            )

            sim_td_list, sim_asd_list = [], []
            for mmp_idx in vis_roi_idx:
                sim_td_file = results_path / f"sim_(mmp){mmp_idx+1}_list_td.npy"
                sim_asd_file = results_path / f"sim_(mmp){mmp_idx+1}_list_asd.npy"
                if sim_td_file.exists() and sim_asd_file.exists():
                    sim_td_list.append(np.load(str(sim_td_file)))
                    sim_asd_list.append(np.load(str(sim_asd_file)))

            if len(sim_td_list) > 0:
                sim_td_avg = np.mean(sim_td_list, axis=0)
                sim_asd_avg = np.mean(sim_asd_list, axis=0)

                dim_td_list = []
                for sim in sim_td_avg:
                    eigvals = np.linalg.eigvals(sim)
                    pos_eigvals = np.real(eigvals[eigvals > 0])
                    if len(pos_eigvals) > 0:
                        dim_td_list.append(np.sum(pos_eigvals)**2 / np.sum(pos_eigvals**2))

                dim_asd_list = []
                for sim in sim_asd_avg:
                    eigvals = np.linalg.eigvals(sim)
                    pos_eigvals = np.real(eigvals[eigvals > 0])
                    if len(pos_eigvals) > 0:
                        dim_asd_list.append(np.sum(pos_eigvals)**2 / np.sum(pos_eigvals**2))

                if len(dim_td_list) > 0 and len(dim_asd_list) > 0:
                    print(f"  ROI dimensionality - TD: {np.mean(dim_td_list):.2f}, ASD: {np.mean(dim_asd_list):.2f}")
                    t_stat, p_val = ttest_ind(dim_asd_list, dim_td_list)
                    print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    else:
        print(f"  Semantic map results not found: {sem_sig_file}")
        print("  Skipping overlap analysis")

    print()

    print("="*80)
    print("Analysis complete!")
    print("="*80)


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
        '--recompute-pr', action='store_true',
        help='Recompute participation ratios from the vertex-level encoding '
             'weights instead of reusing cached PR arrays. Needs the ~20 GB of '
             'vertex weights and a high-memory node; the published PR arrays '
             'are the normal entry point.')
    parser.add_argument(
        '--out-root', type=Path, default=None,
        help='Read and write outputs under this directory instead of the '
             'configured pipeline location. Use to verify a run without '
             'overwriting existing results.')
    args = parser.parse_args()

    if args.skip_surface_figures:
        CONFIG['make_figures'] = False
    if args.recompute_pr:
        CONFIG['overwrite'] = True
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
