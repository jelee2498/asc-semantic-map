"""
11_sem_zscore_0save_raw.py
Save RAW (non-residualized) Triple Network-Level Scores for GAM Z-Score Estimation.

This script prepares brain data (semantic PC scores and dimensionality participation ratio)
at the triple network level (SAL, DMN, CEN) for normative modeling with GAM.

Key differences from 11_sem_0save_data.py:
- Saves RAW scores (not residualized by age/sex)
- Includes age, sex, site, mean_fd columns for GAM fitting
- Outputs separate CSV files for TD and ASD groups

Output: CSV files with raw brain features + demographics for GAM z-score estimation.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, template, fig_dir  # noqa: E402

import nibabel as nib

# Conditional imports (Windows only - for visualization)
if os.name == 'nt':
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.vtk_interface import wrap_vtk, serial_connect
    from vtk import vtkPolyDataNormals


# =============================================================================
# CONFIGURATION (matching 11_sem_0save_data.py)
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '11_sem_zscore',

    # Source data pipelines (from refactored 99_main)
    'source_pipeline_participants': '99_main',
    'source_pipeline_semantic': '99_main',
    'source_pipeline_dimensionality': '99_main',
    'source_pipeline_perf_mask': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_task_semantic': '07_pca',
    'source_task_dimensionality': '09_dimensionality',
    'source_task_perf_mask': '06_encoding_model',

    # Data parameters
    'atlas': 'mmp',
    'resolution': '32k',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option_pc': 0,      # SRM for semantic PC (from 07_pca)
    'srm_option_dim': 50,    # SRM for dimensionality (from 09_dimensionality)
    'srm_type': 'det',       # SRM type for dimensionality

    # Quality control
    'fd_thres': 0.5,
    'hx_docu': 'True',

    # Upstream analysis parameters (matching 11_sem)
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha_semantic': True,
    'enc_single_alpha_dim': False,
    'weight_across_delays': 'Avg',
    'perf_method': 'fdr',
    'perf_alpha': 0.01,
    'weight_add_superordinate': True,
    'pca_template': 'Whole',
    'pca_only_sign': True,
    'pca_iter': 10,
    'comp_no': 1,
    'dim_metric': 'cov',

    # Significance mask options
    'sig_overlap': False,  # If True, use overlapping mask (sem & dim); if False, use separate masks
    'overlap_alpha': 0.1,  # Lenient alpha for overlap detection (matching 09_dimensionality.py)

    # Seed and file operations
    'seed': 0,
    'save_outputs': True,
    'overwrite': False,

    # Age-based screening parameters (from 26_normative/00_save_data)
    'screen': True,  # Whether to apply age-based screening
    'age_childhood_max': 10,
    'age_early_adolescence_max': 14,
    'age_middle_adolescence_max': 18,
    'min_subjects_per_age_group': 5,
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
        'raw': proj_root / '0_data' / 'raw',
        'code': proj_root / '1_code' / pipe,
        'tpl': TEMPLATES,

        # Task-specific output paths
        'pipe': proj_root / '2_pipeline' / pipe,

        # Source data paths (participants)
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        # Performance mask path
        'perf_mask': proj_root / '2_pipeline' /
                    config['source_pipeline_perf_mask'] /
                    config['source_task_perf_mask'] / 'out',

        # EG17 atlas, vendored under templates/.  The original hard-coded an
        # absolute path into one specific conda installation, with no
        # fallback, so this script could not run on any other machine.
        'eg17_atlas': template('eg17'),
    }

    # Semantic results path (from 07_pca)
    paths['semantic_results'] = (
        proj_root / '2_pipeline' /
        config['source_pipeline_semantic'] /
        config['source_task_semantic'] / 'out' /
        config['conf_option'] /
        config['atlas'] /
        f"chunk-{config['chunk_option']}" /
        'fold-avg' / 'results' /
        f"k-{config['srm_option_pc']}" /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"wn-{config['reg_wordnet']}" /
        f"seed-{config['seed']}" /
        f"s_alpha-{config['enc_single_alpha_semantic']}" /
        'sem' /
        f"mask_{config['perf_method']}-{config['perf_alpha']}" /
        f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}"
    )

    # Dimensionality results path (from 09_dimensionality)
    paths['dimensionality_results'] = (
        proj_root / '2_pipeline' /
        config['source_pipeline_dimensionality'] /
        config['source_task_dimensionality'] / 'out' /
        config['resolution'] /
        config['conf_option'] /
        f"{config['srm_type']}_results_inc_byhx_docu-{config['hx_docu']}" /
        f"k-{config['srm_option_dim']}" /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"wn-{config['reg_wordnet']}" /
        f"seed-{config['seed']}" /
        f"s_alpha-{config['enc_single_alpha_dim']}"
    )

    # Task output paths
    paths['proc'] = paths['pipe'] / task
    paths['out'] = paths['proc'] / 'out'
    paths['save'] = paths['proc'] / 'save'
    paths['tmp'] = paths['proc'] / 'tmp'

    # Create directories
    for key in ['out', 'save', 'tmp']:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)

    return paths


# Initialize paths
PATHS = setup_paths(CONFIG)


# =============================================================================
# ATLAS LOADING
# =============================================================================

def load_mmp_atlas(tpl_path: Path) -> npt.NDArray:
    """Load MMP atlas (360 parcels) in 32k fsLR space."""
    mmp_atlas_folder = tpl_path / 'MMP'
    mmp_atlas_path = mmp_atlas_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    mmp_atlas_nonmed = nib.load(str(mmp_atlas_path)).get_fdata()[0].astype(int)

    med_32k_path = mmp_atlas_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
    med_32k = nib.load(str(med_32k_path)).get_fdata()[0].astype(np.int32).nonzero()[0]
    nonmed_32k_ids = np.array(list(set(range(64984)) - set(list(med_32k))))

    mmp_atlas = np.zeros(64984).astype(np.int32)
    mmp_atlas[nonmed_32k_ids] = mmp_atlas_nonmed

    return mmp_atlas


def load_eg17_atlas(eg17_path: Path) -> npt.NDArray:
    """Load EG17 (Evan Gordon 17 networks) atlas."""
    eg_atlas_mat = loadmat(str(eg17_path))
    eg_atlas = np.r_[
        eg_atlas_mat['lh_labels'].squeeze(),
        eg_atlas_mat['rh_labels'].squeeze()
    ].astype(np.int32)
    return eg_atlas


def compute_mmp_to_eg17_mapping(mmp_atlas: npt.NDArray, eg_atlas: npt.NDArray) -> npt.NDArray:
    """
    Compute mapping from 360 MMP parcels to 17 EG networks.

    For each MMP parcel, find the most frequent EG17 network label.
    """
    mapping_mmp_2_eg17 = np.zeros(360)
    for mmp_no in range(360):
        mmp_ids = np.where(mmp_atlas == mmp_no + 1)[0]
        eg_atlas_mmp_ids = eg_atlas[mmp_ids]
        eg_atlas_mmp_ids_unique, eg_atlas_mmp_ids_count = np.unique(
            eg_atlas_mmp_ids, return_counts=True
        )
        mapping_mmp_2_eg17[mmp_no] = eg_atlas_mmp_ids_unique[np.argmax(eg_atlas_mmp_ids_count)]
    return mapping_mmp_2_eg17


# =============================================================================
# TRIPLE NETWORK DEFINITIONS
# =============================================================================

# EG17 network labels (1-indexed in the atlas)
EG17_LIST = [
    'Default', 'LatVis', 'FrontPar', 'MedVis', 'DorsAttn', 'Premotor', 'Language',
    'Salience', 'CingOperc', 'HandSM', 'FaceSM', 'Auditory', 'AntMTL', 'PostMTL',
    'ParMemory', 'Context', 'FootSM'
]

# Triple network mapping (EG17 network names -> triple network)
TRIPLE_NETWORK_MAPPING = {
    'SAL': ['Salience'],                         # Salience network
    'DMN': ['Default', 'Language', 'Context'],   # Default mode network
    'CEN': ['CingOperc', 'FrontPar'],            # Central executive network
}


def get_triple_network_ids():
    """Get EG17 indices (1-indexed) for each triple network."""
    sal_ids = [EG17_LIST.index(n) + 1 for n in TRIPLE_NETWORK_MAPPING['SAL']]
    dmn_ids = [EG17_LIST.index(n) + 1 for n in TRIPLE_NETWORK_MAPPING['DMN']]
    cen_ids = [EG17_LIST.index(n) + 1 for n in TRIPLE_NETWORK_MAPPING['CEN']]
    return sal_ids, dmn_ids, cen_ids


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_participant_data(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load and filter participant data."""
    print("\n[1/8] Loading participants data...")

    participants_df = pd.read_excel(
        paths['participants_df'] / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    id_list_td_asd = []
    for dx in ['TD', 'ASD']:
        print(f"  Filtering {dx} subjects...")

        # Filter #1: dx
        if dx == 'TD':
            dx_filt = (participants_df['DX'] == 'TD').values
        else:
            dx_filt = (participants_df['DX'] == 'ASD').values

        # Filter #2: by-history documentation
        if config['hx_docu'] == 'True':
            for sub_id in participants_df[dx_filt].index:
                if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                    participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                    dx_filt[participants_df.index == sub_id] = False

        # Filter #3: prep_ok
        prep_filt = (participants_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] == 1).values

        # Filter #4: no remarks
        no_remarks_filt = (participants_df['Remarks'] == 'none').values

        # Combine filters
        id_list_dx_prep_remark = list(participants_df[dx_filt & prep_filt & no_remarks_filt].index)
        participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

        # Filter #5: FD
        fd_filt = (participants_df_dx_prep_remark['Mean_FD_DM'] < config['fd_thres']).values

        # Filter #6: deepmreye
        deepmreye_filt = (participants_df_dx_prep_remark['Rating_deepmreye_movieDM'] == 1).values

        id_list = list(participants_df_dx_prep_remark[np.multiply(fd_filt, deepmreye_filt)].index)
        id_list = sorted(id_list)

        print(f"    {dx} subjects: {len(id_list)}")
        id_list_td_asd.append(id_list)

    id_list_td, id_list_asd = id_list_td_asd[0], id_list_td_asd[1]
    print(f"  Total: TD={len(id_list_td)}, ASD={len(id_list_asd)}")

    return participants_df, id_list_td, id_list_asd


def apply_age_screening(
    participants_df: pd.DataFrame,
    id_list_td: List[str],
    id_list_asd: List[str],
    config: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str], npt.NDArray, npt.NDArray]:
    """
    Apply age-based screening to exclude sparse age groups.

    From 26_normative/00_save_data_release11_triple_network.py:
    - Age groups: childhood (≤10), early_adolescence (10-14),
                  middle_adolescence (14-18), late_adolescence (>18)
    - Exclude groups with < MIN_SUBJECTS_PER_AGE_GROUP in either TD or ASD

    Returns:
        Tuple of (filtered_id_list_td, filtered_id_list_asd, excluded_age_groups,
                  mask_td, mask_asd)
        where mask_td/mask_asd are boolean arrays for filtering feature arrays
    """
    n_td_orig = len(id_list_td)
    n_asd_orig = len(id_list_asd)

    if not config.get('screen', True):
        print("\n  Age-based screening: DISABLED")
        return (id_list_td, id_list_asd, [],
                np.ones(n_td_orig, dtype=bool), np.ones(n_asd_orig, dtype=bool))

    print("\n  Applying age-based screening...")

    # Get parameters
    age_childhood_max = config['age_childhood_max']
    age_early_adolescence_max = config['age_early_adolescence_max']
    age_middle_adolescence_max = config['age_middle_adolescence_max']
    min_subjects = config['min_subjects_per_age_group']

    # Define age group labeling function
    def label_age_group(age: float) -> str:
        if age <= age_childhood_max:
            return 'childhood'
        elif age <= age_early_adolescence_max:
            return 'early_adolescence'
        elif age <= age_middle_adolescence_max:
            return 'middle_adolescence'
        else:
            return 'late_adolescence'

    # Get ages and label age groups
    ages_td = participants_df.loc[id_list_td, 'Age'].values
    ages_asd = participants_df.loc[id_list_asd, 'Age'].values

    age_groups_td = np.array([label_age_group(age) for age in ages_td])
    age_groups_asd = np.array([label_age_group(age) for age in ages_asd])

    # Count subjects per age group
    age_group_names = ['childhood', 'early_adolescence', 'middle_adolescence', 'late_adolescence']

    print(f"\n  Age group counts (before screening):")
    print(f"  {'Age Group':<20} {'TD':>6} {'ASD':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*6}")

    excluded_groups = []
    for age_group in age_group_names:
        n_td = np.sum(age_groups_td == age_group)
        n_asd = np.sum(age_groups_asd == age_group)
        exclude_marker = ''

        # Check if either group has too few subjects
        if n_td < min_subjects or n_asd < min_subjects:
            excluded_groups.append(age_group)
            exclude_marker = ' *EXCLUDE*'

        print(f"  {age_group:<20} {n_td:>6} {n_asd:>6}{exclude_marker}")

    # Create boolean masks for filtering
    if excluded_groups:
        print(f"\n  Excluding age groups with < {min_subjects} subjects: {excluded_groups}")

        mask_td = ~np.isin(age_groups_td, excluded_groups)
        mask_asd = ~np.isin(age_groups_asd, excluded_groups)

        filtered_id_list_td = [sub_id for sub_id, keep in zip(id_list_td, mask_td) if keep]
        filtered_id_list_asd = [sub_id for sub_id, keep in zip(id_list_asd, mask_asd) if keep]

        n_excluded_td = n_td_orig - len(filtered_id_list_td)
        n_excluded_asd = n_asd_orig - len(filtered_id_list_asd)

        print(f"  Excluded TD subjects: {n_excluded_td}")
        print(f"  Excluded ASD subjects: {n_excluded_asd}")
    else:
        print(f"\n  No age groups excluded (all have >= {min_subjects} subjects)")
        mask_td = np.ones(n_td_orig, dtype=bool)
        mask_asd = np.ones(n_asd_orig, dtype=bool)
        filtered_id_list_td = id_list_td
        filtered_id_list_asd = id_list_asd

    print(f"\n  After screening:")
    print(f"    TD: {len(filtered_id_list_td)} subjects")
    print(f"    ASD: {len(filtered_id_list_asd)} subjects")

    return filtered_id_list_td, filtered_id_list_asd, excluded_groups, mask_td, mask_asd


def load_semantic_features(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Load semantic PC scores from 07_pca output."""
    print("\n[2/8] Loading semantic PC scores from 07_pca...")

    sem_path = paths['semantic_results']
    comp_no = config['comp_no']

    # Load semantic scores (in perf_mask space)
    sem_list_td = np.load(sem_path / 'sem_list_td.npy')[:, :, comp_no - 1]
    sem_list_asd = np.load(sem_path / 'sem_list_asd.npy')[:, :, comp_no - 1]

    # Load performance mask
    perf_mask_path = (
        paths['perf_mask'] /
        config['conf_option'] /
        config['atlas'] /
        f"chunk-{config['chunk_option']}" /
        'fold-avg' /
        f"d-{config['hx_docu']}" /
        f"k-{config['srm_option_pc']}" /
        f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
        f"wn-{config['reg_wordnet']}" /
        f"seed-{config['seed']}" /
        f"s_alpha-{config['enc_single_alpha_semantic']}" /
        f"mask_{config['perf_method']}-{config['perf_alpha']}.npy"
    )
    perf_mask = np.load(perf_mask_path)

    print(f"  Semantic features loaded: TD={sem_list_td.shape}, ASD={sem_list_asd.shape}")
    print(f"  Performance mask size: {len(perf_mask)} parcels")

    return sem_list_td, sem_list_asd, perf_mask


def load_dimensionality_features(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Load participation ratio values from 09_dimensionality output."""
    print("\n[3/8] Loading dimensionality features from 09_dimensionality...")

    dim_path = paths['dimensionality_results']
    metric = config['dim_metric']

    dim_list_td = np.load(dim_path / f'{metric}_pr_td.npy')
    dim_list_asd = np.load(dim_path / f'{metric}_pr_asd.npy')

    print(f"  Dimensionality features loaded: TD={dim_list_td.shape}, ASD={dim_list_asd.shape}")

    return dim_list_td, dim_list_asd


def visualize_significance_mask(
    sig_mask: npt.NDArray,
    paths: Dict[str, Path],
    title: str,
    cmap: str = 'Spectral_r',
    save_path: Path = None
) -> None:
    """
    Visualize significance mask on brain surface.

    Args:
        sig_mask: Significance mask array (360 parcels for MMP)
        paths: Path dictionary containing template paths
        title: Title for the visualization
        cmap: Colormap name
        save_path: Optional path to save the figure
    """
    if os.name != 'nt':
        print(f"  Skipping visualization (Windows only): {title}")
        return

    try:
        # Load surface meshes
        mmp_folder = paths['tpl'] / 'MMP'
        surfs = [None] * 2
        surfs[0] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.L.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
        nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
        surf_lh = serial_connect(surfs[0], nf)
        surfs[1] = read_surface(str(mmp_folder / 'Q1-Q6_RelatedParcellation210.R.very_inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'))
        nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
        surf_rh = serial_connect(surfs[1], nf)

        # Load MMP atlas for mapping
        mmp_atlas_path = mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
        mmp_atlas_nonmed = nib.load(str(mmp_atlas_path)).get_fdata()[0].astype(int)

        med_32k_path = mmp_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
        med_32k = nib.load(str(med_32k_path)).get_fdata()[0].astype(np.int32).nonzero()[0]
        nonmed_32k_ids = np.array(list(set(range(64984)) - set(list(med_32k))))

        mmp_atlas = np.zeros(64984).astype(np.int32)
        mmp_atlas[nonmed_32k_ids] = mmp_atlas_nonmed

        # Convert sig_mask to float to allow np.nan as fill value
        sig_mask_float = sig_mask.astype(float)

        # Map significance mask to 32k surface
        sig_mask_32k = map_to_labels(
            sig_mask_float,
            mmp_atlas,
            mask=mmp_atlas != 0,
            fill=np.nan
        )

        # Determine color range
        cmin, cmax = -1, 1

        # Plot hemispheres
        print(f"  Visualizing: {title}")
        plot_hemispheres(
            surf_lh,
            surf_rh,
            array_name=sig_mask_32k,
            size=(1200, 200),
            cmap=cmap,
            nan_color=(0.5, 0.5, 0.5, 1),
            color_bar=True,
            color_range=(cmin, cmax),
            zoom=1.6,
        )

        # Save figure if path provided
        if save_path is not None:
            import matplotlib.pyplot as plt
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")

    except Exception as e:
        print(f"  Warning: Could not visualize {title}: {e}")


def load_significance_masks(
    paths: Dict[str, Path],
    config: Dict[str, Any],
    visualize: bool = True
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Load significance masks from group-difference analysis.

    For strict masks (used when sig_overlap=False):
        - Loads pre-computed sig_regions files

    For overlap mask (used when sig_overlap=True):
        - Loads p-values and t-stats
        - Applies lenient alpha (overlap_alpha=0.1) threshold
        - Consistent with 09_dimensionality.py Step 14

    Returns:
        Tuple of (sem_sig_mask, dim_sig_mask, overlap_mask)
    """
    print("\n[4/8] Loading significance masks...")

    sem_path = paths['semantic_results']
    dim_path = paths['dimensionality_results']
    pca_template = config['pca_template']
    pca_only_sign = config['pca_only_sign']
    pca_iter = config['pca_iter']
    metric = config['dim_metric']

    # -------------------------------------------------------------------------
    # Load STRICT significance masks (pre-computed at glm_alpha)
    # -------------------------------------------------------------------------
    # Semantic strict mask
    sem_sig_filename = f"sig_regions_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy"
    sem_sig_mask = np.load(sem_path / sem_sig_filename)

    # Dimensionality strict mask
    dim_sig_mask = np.load(dim_path / f'{metric}_sig_mmp.npy')

    # Count significant regions (strict)
    sem_sig_count = np.sum(sem_sig_mask != 0)
    dim_sig_count = np.sum(dim_sig_mask)

    print(f"  Semantic significance mask (strict): {sem_sig_count} significant regions")
    print(f"  Dimensionality significance mask (strict): {dim_sig_count} significant regions")

    # -------------------------------------------------------------------------
    # Calculate LENIENT overlap mask (using overlap_alpha=0.1)
    # Consistent with 09_dimensionality.py Step 14
    # -------------------------------------------------------------------------
    overlap_alpha = config.get('overlap_alpha', 0.1)
    print(f"\n  Calculating overlap mask with lenient alpha={overlap_alpha}...")

    # Load semantic p-values and t-stats for lenient thresholding
    pval_file = sem_path / f"pval_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy"
    tstat_file = sem_path / f"tstat_{pca_template}_sign-{pca_only_sign}_iter-{pca_iter}.npy"

    if pval_file.exists():
        pval_sr = np.load(pval_file)
        tstat_sr = np.load(tstat_file)

        # Apply lenient threshold for semantic
        sig_mask_sr_lenient = pval_sr < overlap_alpha
        sig_mmp_sr_lenient = np.zeros(360)
        sig_mmp_sr_lenient[sig_mask_sr_lenient] = np.sign(tstat_sr[sig_mask_sr_lenient])
        print(f"    Semantic (lenient): {np.sum(sig_mask_sr_lenient)} significant regions")
    else:
        # Fallback to pre-computed sig_regions if p-values not available
        print(f"    Warning: p-values not found, using pre-computed sig_regions (strict)")
        sig_mmp_sr_lenient = sem_sig_mask.astype(float)

    # Load dimensionality stat_results for lenient thresholding
    dim_stat_file = dim_path / f'{metric}_stat_results.npy'
    dim_stat = np.load(dim_stat_file, allow_pickle=True).item()

    # Apply lenient threshold for dimensionality
    sig_mask_dim_lenient = dim_stat['p_fdr'] < overlap_alpha
    sig_mmp_dim_lenient = np.zeros(360)
    sig_mmp_dim_lenient[sig_mask_dim_lenient] = np.sign(dim_stat['t'][sig_mask_dim_lenient])
    print(f"    Dimensionality (lenient): {np.sum(sig_mask_dim_lenient)} significant regions")

    # Calculate signed overlap mask (product of lenient masks, like 09_dimensionality.py)
    # Values: +1 (both positive), -1 (both negative or opposite signs), 0 (no overlap)
    sig_mmp_overlap = sig_mmp_sr_lenient * sig_mmp_dim_lenient
    overlap_mask = sig_mmp_overlap != 0
    overlap_count = np.sum(overlap_mask)
    n_concordant = np.sum(sig_mmp_overlap > 0)
    n_discordant = np.sum(sig_mmp_overlap < 0)
    print(f"    Overlap (lenient): {overlap_count} significant regions "
          f"(concordant: {n_concordant}, discordant: {n_discordant})")

    # Visualize significance masks on brain surface
    if visualize and os.name == 'nt':
        print("\n  Visualizing significance masks on brain surface...")

        # Create output path for figures
        fig_path = (
            paths['code'] / 'claude_figures' / 'significance_masks' / f"overlap_alpha-{overlap_alpha}"
        )
        fig_path.mkdir(parents=True, exist_ok=True)

        # Visualize semantic significance mask
        # Values: -1 (ASD < TD), 0 (not significant), 1 (ASD > TD)
        visualize_significance_mask(
            sig_mask=sem_sig_mask,
            paths=paths,
            title="Semantic PC Significance Mask (ASD vs TD)",
            cmap='Spectral_r',
            save_path=fig_path / 'sem_sig_mask_brain.png'
        )

        # Visualize dimensionality significance mask
        # Convert boolean to float for visualization
        dim_sig_mask_float = dim_sig_mask.astype(float)
        visualize_significance_mask(
            sig_mask=dim_sig_mask_float,
            paths=paths,
            title="Dimensionality (PR) Significance Mask",
            cmap='Spectral_r',
            save_path=fig_path / 'dim_sig_mask_brain.png'
        )

        # Visualize overlap significance mask (with sign)
        # Values: +1 (concordant: both same direction), -1 (discordant: opposite directions)
        visualize_significance_mask(
            sig_mask=sig_mmp_overlap,
            paths=paths,
            title="Overlap Significance Mask (Sem & Dim, with sign)",
            cmap='Spectral_r',
            save_path=fig_path / 'overlap_sig_mask_brain.png'
        )

    return sem_sig_mask, dim_sig_mask, overlap_mask


# =============================================================================
# TRIPLE NETWORK AGGREGATION
# =============================================================================

def expand_perf_mask_to_360(scores_perf: npt.NDArray, perf_mask: npt.NDArray) -> npt.NDArray:
    """Expand scores from perf_mask space to full 360-space."""
    scores_360 = np.full(360, np.nan)
    scores_360[perf_mask] = scores_perf
    return scores_360


def aggregate_to_triple_network(
    scores_360: npt.NDArray,
    sig_idx: npt.NDArray,
    mapping_mmp_2_eg17: npt.NDArray,
    sal_ids: List[int],
    dmn_ids: List[int],
    cen_ids: List[int]
) -> Tuple[float, float, float]:
    """Aggregate 360-parcel scores to triple network level within significance mask."""
    # Get intersection of significant parcels and each network
    sal_mask = [i for i in sig_idx if mapping_mmp_2_eg17[i] in sal_ids and not np.isnan(scores_360[i])]
    dmn_mask = [i for i in sig_idx if mapping_mmp_2_eg17[i] in dmn_ids and not np.isnan(scores_360[i])]
    cen_mask = [i for i in sig_idx if mapping_mmp_2_eg17[i] in cen_ids and not np.isnan(scores_360[i])]

    # Average within each network (handle empty case)
    sal_avg = scores_360[sal_mask].mean() if len(sal_mask) > 0 else np.nan
    dmn_avg = scores_360[dmn_mask].mean() if len(dmn_mask) > 0 else np.nan
    cen_avg = scores_360[cen_mask].mean() if len(cen_mask) > 0 else np.nan

    return sal_avg, dmn_avg, cen_avg


def compute_triple_network_scores(
    sem_list: npt.NDArray,
    dim_list: npt.NDArray,
    perf_mask: npt.NDArray,
    sem_sig_idx: npt.NDArray,
    dim_sig_idx: npt.NDArray,
    mapping_mmp_2_eg17: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute triple network scores for all subjects."""
    sal_ids, dmn_ids, cen_ids = get_triple_network_ids()
    n_subjects = sem_list.shape[0]

    sem_triple = np.zeros((n_subjects, 3))
    dim_triple = np.zeros((n_subjects, 3))

    for i in range(n_subjects):
        # Semantic: expand from perf_mask to 360-space, then aggregate
        sem_360 = expand_perf_mask_to_360(sem_list[i], perf_mask)
        sem_triple[i] = aggregate_to_triple_network(
            sem_360, sem_sig_idx, mapping_mmp_2_eg17, sal_ids, dmn_ids, cen_ids
        )

        # Dimensionality: already in 360-space
        dim_triple[i] = aggregate_to_triple_network(
            dim_list[i], dim_sig_idx, mapping_mmp_2_eg17, sal_ids, dmn_ids, cen_ids
        )

    return sem_triple, dim_triple


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to save raw triple network data for GAM z-score estimation."""
    print("=" * 70)
    print("11_sem_zscore_0save_raw: Save Raw Triple Network Scores for GAM")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load participants
    # -------------------------------------------------------------------------
    participants_df, id_list_td_orig, id_list_asd_orig = load_participant_data(PATHS, CONFIG)

    # -------------------------------------------------------------------------
    # Step 2: Load semantic features and perf_mask
    # -------------------------------------------------------------------------
    sem_list_td, sem_list_asd, perf_mask = load_semantic_features(PATHS, CONFIG)

    # -------------------------------------------------------------------------
    # Step 3: Load dimensionality features
    # -------------------------------------------------------------------------
    dim_list_td, dim_list_asd = load_dimensionality_features(PATHS, CONFIG)

    # -------------------------------------------------------------------------
    # Step 3.5: Apply age-based screening (if enabled)
    # -------------------------------------------------------------------------
    id_list_td, id_list_asd, excluded_age_groups, mask_td, mask_asd = apply_age_screening(
        participants_df, id_list_td_orig, id_list_asd_orig, CONFIG
    )

    # Filter feature arrays using the masks
    if CONFIG.get('screen', True) and len(excluded_age_groups) > 0:
        print("\n  Filtering feature arrays...")
        sem_list_td = sem_list_td[mask_td]
        sem_list_asd = sem_list_asd[mask_asd]
        dim_list_td = dim_list_td[mask_td]
        dim_list_asd = dim_list_asd[mask_asd]
        print(f"    Semantic: TD={sem_list_td.shape}, ASD={sem_list_asd.shape}")
        print(f"    Dimensionality: TD={dim_list_td.shape}, ASD={dim_list_asd.shape}")

    n_td, n_asd = len(id_list_td), len(id_list_asd)

    # -------------------------------------------------------------------------
    # Step 4: Load significance masks
    # -------------------------------------------------------------------------
    sem_sig_mask, dim_sig_mask, overlap_mask = load_significance_masks(PATHS, CONFIG)

    # Get significant indices based on sig_overlap config
    sem_sig_idx = np.where(sem_sig_mask != 0)[0]
    dim_sig_idx = np.where(dim_sig_mask)[0]
    overlap_sig_idx = np.where(overlap_mask)[0]

    # Determine which indices to use for aggregation
    if CONFIG.get('sig_overlap', False):
        print(f"\n  Using OVERLAP mask for aggregation ({len(overlap_sig_idx)} regions)")
        sem_agg_idx = overlap_sig_idx
        dim_agg_idx = overlap_sig_idx
    else:
        print(f"\n  Using SEPARATE masks for aggregation (sem={len(sem_sig_idx)}, dim={len(dim_sig_idx)})")
        sem_agg_idx = sem_sig_idx
        dim_agg_idx = dim_sig_idx

    # -------------------------------------------------------------------------
    # Step 5: Load atlases and compute mapping
    # -------------------------------------------------------------------------
    print("\n[5/9] Loading atlases and computing MMP to EG17 mapping...")

    mmp_atlas = load_mmp_atlas(PATHS['tpl'])
    eg_atlas = load_eg17_atlas(PATHS['eg17_atlas'])
    mapping_mmp_2_eg17 = compute_mmp_to_eg17_mapping(mmp_atlas, eg_atlas)

    print(f"  MMP atlas loaded: {len(np.unique(mmp_atlas))-1} parcels")
    print(f"  EG17 atlas loaded: {len(np.unique(eg_atlas))-1} networks")

    # -------------------------------------------------------------------------
    # Step 6: Aggregate to triple network level
    # -------------------------------------------------------------------------
    print("\n[6/9] Aggregating to triple network level...")

    # TD subjects
    sem_triple_td, dim_triple_td = compute_triple_network_scores(
        sem_list_td, dim_list_td, perf_mask, sem_agg_idx, dim_agg_idx, mapping_mmp_2_eg17
    )
    print(f"  TD: sem_triple={sem_triple_td.shape}, dim_triple={dim_triple_td.shape}")

    # ASD subjects
    sem_triple_asd, dim_triple_asd = compute_triple_network_scores(
        sem_list_asd, dim_list_asd, perf_mask, sem_agg_idx, dim_agg_idx, mapping_mmp_2_eg17
    )
    print(f"  ASD: sem_triple={sem_triple_asd.shape}, dim_triple={dim_triple_asd.shape}")

    # -------------------------------------------------------------------------
    # Step 7: Create output DataFrames (RAW scores, not residualized)
    # -------------------------------------------------------------------------
    print("\n[7/9] Creating output DataFrames (raw scores)...")

    col_list = ['sal', 'dmn', 'cen']

    # Semantic DataFrames
    sem_data_td_df = pd.DataFrame(sem_triple_td, index=id_list_td, columns=col_list)
    sem_data_asd_df = pd.DataFrame(sem_triple_asd, index=id_list_asd, columns=col_list)

    # Dimensionality DataFrames
    dim_data_td_df = pd.DataFrame(dim_triple_td, index=id_list_td, columns=col_list)
    dim_data_asd_df = pd.DataFrame(dim_triple_asd, index=id_list_asd, columns=col_list)

    # Add demographics
    for df, id_list in [(sem_data_td_df, id_list_td), (sem_data_asd_df, id_list_asd),
                        (dim_data_td_df, id_list_td), (dim_data_asd_df, id_list_asd)]:
        df['age'] = participants_df.loc[id_list]['Age'].values
        # Sex encoding: Male=0, Female=1
        df['sex'] = (participants_df.loc[id_list]['Sex'] == 'Female').astype(int).values
        # Site encoding: CBIC=0, RU=1
        df['site'] = (participants_df.loc[id_list]['Site'] == 'RU').astype(int).values
        df['mean_fd'] = participants_df.loc[id_list]['Mean_FD_DM'].values

    print(f"  TD DataFrames: sem={sem_data_td_df.shape}, dim={dim_data_td_df.shape}")
    print(f"  ASD DataFrames: sem={sem_data_asd_df.shape}, dim={dim_data_asd_df.shape}")

    # -------------------------------------------------------------------------
    # Step 8: Save outputs
    # -------------------------------------------------------------------------
    print("\n[8/9] Saving outputs...")

    out_path = (
        PATHS['out'] /
        CONFIG['conf_option'] /
        CONFIG['atlas'] /
        f"chunk-{CONFIG['chunk_option']}" /
        'fold-avg' / 'results' /
        f"k-{CONFIG['srm_option_pc']}_{CONFIG['srm_option_dim']}" /
        f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
        f"wn-{CONFIG['reg_wordnet']}" /
        f"s_alpha-{CONFIG['enc_single_alpha_semantic']}" /
        'sem' /
        f"mask_{CONFIG['perf_method']}-{CONFIG['perf_alpha']}" /
        f"delay-{CONFIG['weight_across_delays']}_super-{CONFIG['weight_add_superordinate']}" /
        f"screen-{CONFIG['screen']}_overlap-{CONFIG['sig_overlap']}"
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    sem_data_td_df.to_csv(out_path / 'sem_data_td_raw.csv')
    sem_data_asd_df.to_csv(out_path / 'sem_data_asd_raw.csv')
    dim_data_td_df.to_csv(out_path / 'dim_data_td_raw.csv')
    dim_data_asd_df.to_csv(out_path / 'dim_data_asd_raw.csv')

    print(f"  Saved to: {out_path}")
    print(f"    - sem_data_td_raw.csv ({len(sem_data_td_df)} subjects)")
    print(f"    - sem_data_asd_raw.csv ({len(sem_data_asd_df)} subjects)")
    print(f"    - dim_data_td_raw.csv ({len(dim_data_td_df)} subjects)")
    print(f"    - dim_data_asd_raw.csv ({len(dim_data_asd_df)} subjects)")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics (raw scores)")
    print("=" * 70)

    print("\nSemantic TD:")
    print(sem_data_td_df[col_list].describe())

    print("\nSemantic ASD:")
    print(sem_data_asd_df[col_list].describe())

    print("\nDimensionality TD:")
    print(dim_data_td_df[col_list].describe())

    print("\nDimensionality ASD:")
    print(dim_data_asd_df[col_list].describe())

    print("\n" + "=" * 70)
    print("Done! Run 11_sem_zscore_1gam.R next to fit GAM and calculate z-scores.")
    print("=" * 70)


if __name__ == '__main__':
    main()
