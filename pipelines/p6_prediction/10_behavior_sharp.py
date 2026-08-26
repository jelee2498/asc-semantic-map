"""
Behavioral Prediction Analysis using Semantic and Dimensionality Features (TD vs ASD).

Refactored from: 19_behav_prediction/02_combined_fold_surfstat.py

This script performs:
1. Load semantic PC scores from 07_pca.py output
2. Load dimensionality (participation ratio) from 09_dimensionality.py output
3. Cross-validation with fold-specific significant region calculation
4. ElasticNetCV regression for behavioral prediction (SRS subscales)
5. Permutation testing for significance (using feature_type-specific sig regions)
6. Comparative visualization across feature sets

Key features:
- Fold-specific significant region calculation (prevents data leakage)
- Feature-type specific sig_regions for permutation tests:
  - semantic: from 07_pca output
  - dimensionality: from 09_dimensionality output
  - combined: union of both
- Separate enc_single_alpha parameters for semantic vs dimensionality paths

All functions are defined locally - this is a standalone script with no external dependencies.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Scientific computing
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import zscore, wilcoxon, ttest_rel
from scipy.stats import t as t_dist

# Machine learning
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Neuroimaging / Statistics
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from statsmodels.stats.multitest import multipletests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Repository lib/ - shared configuration and the SHARP test module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, fig_dir  # noqa: E402
import sharp_test  # noqa: E402  SHARP (Split-HAlf RePeated) model-comparison test

# Utilities
from tqdm import tqdm
from joblib import Parallel, delayed

os.environ.setdefault("LOKY_PICKLER", "cloudpickle")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '10_behavior_sharp',

    # Source data pipelines (from refactored 99_main)
    'source_pipeline_participants': '99_main',
    'source_pipeline_semantic': '99_main',
    'source_pipeline_dimensionality': '99_main',
    'source_pipeline_perf_mask': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_task_semantic': '07_pca',
    'source_task_dimensionality': '09_dimensionality',
    'source_task_perf_mask': '06_encoding_model',

    # Data parameters (must match upstream pipelines)
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

    # Upstream analysis parameters (for path construction)
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha_semantic': True,   # For 07_pca path construction
    'enc_single_alpha_dim': False,       # For 09_dimensionality path construction
    'weight_across_delays': 'Avg',
    'perf_method': 'fdr',
    'perf_alpha': 0.01,
    'weight_add_superordinate': True,
    'pca_template': 'Whole',
    'pca_only_sign': True,
    'pca_iter': 10,
    'comp_no': 1,
    'dim_metric': 'cov',

    # Statistical parameters
    'glm_alpha': 0.025,  # FDR alpha for two-tailed GLM

    # Cross-validation parameters
    'n_rep': 100,        # Number of repetitions
    'n_cv': 20,          # Number of folds per repetition
    'n_perm': 1000,      # Number of permutations
    'n_jobs': 30,        # Parallel workers for rep/perm loops
    'sharp_K': 10,       # SHARP: K-fold CV within each split-half
    'sharp_J': 30,       # SHARP: number of split-half repetitions
    'corrt_rho': None,   # corrected-t assumed between-fold rho (None=1/K; lower=more lenient)
    'one_sided_combined': True,  # combined-vs-* tested one-sided (greater); sem-vs-dim two-sided
    'min_sig_regions': 10,  # Minimum significant regions required per fold

    # Behavioral items
    'behav_items': ['SRS_AWR_T', 'SRS_COG_T', 'SRS_COM_T', 'SRS_MOT_T', 'SRS_RRB_T'],

    # Seed and file operations
    'seed': 0,
    'save_outputs': True,
    'overwrite': False,
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

        # Source data paths (participants)
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        # Performance mask path
        'perf_mask': proj_root / '2_pipeline' /
                    config['source_pipeline_perf_mask'] /
                    config['source_task_perf_mask'] / 'out',
    }

    # Semantic results path (from 07_pca)
    # Path: 07_pca/out/{conf}/{atlas}/chunk-{chunk}/fold-avg/results/k-{srm}/bias-{b}_verb-{v}/wn-{wn}/seed-{s}/s_alpha-{sa}/sem/mask_{m}-{a}/delay-{d}_super-{sup}/
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
    # Path: 09_dimensionality/out/{resolution}/{conf}/{srm_type}_results_inc_byhx_docu-{hx}/k-{srm}/bias-{b}_verb-{v}/wn-{wn}/seed-{s}/s_alpha-{sa}/
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
    for key in ['out', 'save', 'tmp', 'fig']:
        if key in paths:
            paths[key].mkdir(parents=True, exist_ok=True)

    return paths


# Initialize paths
PATHS = setup_paths(CONFIG)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_save(data: Any, filepath: Path, overwrite: bool = False) -> bool:
    """
    Safely save data with overwrite protection.

    Args:
        data: Data to save (numpy array or dict for npz)
        filepath: Path to save file
        overwrite: If True, overwrite existing files; if False, skip

    Returns:
        True if file was saved, False if skipped
    """
    filepath = Path(filepath)

    if filepath.exists() and not overwrite:
        print(f"    Skipping (exists): {filepath.name}")
        return False

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.npy':
        np.save(filepath, data)
    elif filepath.suffix == '.npz':
        if isinstance(data, dict):
            np.savez(filepath, **data)
        else:
            np.savez(filepath, data=data)
    elif filepath.suffix == '.csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath)
        else:
            pd.DataFrame(data).to_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    print(f"    Saved: {filepath.name}")
    return True


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_participant_data(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load and filter participant data.

    Args:
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        Tuple of (participants_df, id_list_td, id_list_asd)
    """
    print("\n[1/10] Loading participants data...")

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


def load_semantic_features(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load semantic PC scores from 07_pca output.

    Args:
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        Tuple of (sem_list_td, sem_list_asd) - shape (n_subjects, n_roi)
    """
    print("\n[2/10] Loading semantic PC scores from 07_pca...")

    sem_path = paths['semantic_results']
    comp_no = config['comp_no']

    sem_list_td = np.load(sem_path / 'sem_list_td.npy')[:, :, comp_no - 1]
    sem_list_asd = np.load(sem_path / 'sem_list_asd.npy')[:, :, comp_no - 1]

    print(f"  Semantic features loaded: TD={sem_list_td.shape}, ASD={sem_list_asd.shape}")
    print(f"  Source: {sem_path}")

    return sem_list_td, sem_list_asd


def load_dimensionality_features(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load participation ratio values from 09_dimensionality output.

    Args:
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        Tuple of (dim_list_td, dim_list_asd) - shape (n_subjects, 360)
    """
    print("\n[3/10] Loading dimensionality features from 09_dimensionality...")

    dim_path = paths['dimensionality_results']
    metric = config['dim_metric']

    dim_list_td = np.load(dim_path / f'{metric}_pr_td.npy')
    dim_list_asd = np.load(dim_path / f'{metric}_pr_asd.npy')

    print(f"  Dimensionality features loaded: TD={dim_list_td.shape}, ASD={dim_list_asd.shape}")
    print(f"  Source: {dim_path}")

    return dim_list_td, dim_list_asd


def load_performance_mask(
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> npt.NDArray:
    """
    Load performance mask from 06_encoding_model output.

    Args:
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        perf_mask: Array of significant ROI indices
    """
    print("\n[4/10] Loading performance mask from 06_encoding_model...")

    perf_path = (
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

    perf_mask = np.load(perf_path)
    print(f"  Performance mask ROIs: {len(perf_mask)}")
    print(f"  Source: {perf_path}")

    return perf_mask


def load_behavioral_data(
    participants_df: pd.DataFrame,
    id_list_td: List[str],
    id_list_asd: List[str],
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], npt.NDArray]:
    """
    Load behavioral data (SRS subscales) and prepare covariates.

    Args:
        participants_df: Participant metadata DataFrame
        id_list_td: List of TD subject IDs
        id_list_asd: List of ASD subject IDs
        config: Configuration dictionary

    Returns:
        Tuple of (behav_df, id_list_behav, covariates)
    """
    print("\n[5/10] Loading behavioral data and covariates...")

    item_list = config['behav_items']

    # Filter subjects with valid behavioral data
    id_list_behav = []
    for sub_id in id_list_td + id_list_asd:
        behav_data = participants_df.loc[sub_id, item_list]
        if behav_data.apply(lambda x: isinstance(x, str)).any():
            print(f"  Removing {sub_id} due to invalid behavioral data")
        else:
            id_list_behav.append(sub_id)

    behav_df = participants_df.loc[id_list_behav][item_list]
    behav_df = behav_df.astype(float)

    print(f"  Subjects with behavioral data: {len(id_list_behav)}")

    # Prepare covariates (age, sex)
    covariates_df = participants_df.loc[id_list_behav][['Age', 'Sex']]
    covariates_df['Sex_numeric'] = (covariates_df['Sex'] == 'Male').astype(int)
    covariates = covariates_df[['Age', 'Sex_numeric']].values.astype(float)

    print(f"  Covariates shape: {covariates.shape}")
    print(f"  Age range: {covariates[:, 0].min():.1f} - {covariates[:, 0].max():.1f}")
    print(f"  Sex: {np.sum(covariates[:, 1] == 1)} Male, {np.sum(covariates[:, 1] == 0)} Female")

    return behav_df, id_list_behav, covariates


def load_sig_regions_by_feature_type(
    feature_type: str,
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> npt.NDArray:
    """
    Load significant regions based on feature type.

    Args:
        feature_type: 'semantic', 'dimensionality', or 'combined'
        paths: Path dictionary
        config: Configuration dictionary

    Returns:
        sig_regions: Boolean array (360,) indicating significant ROIs
    """
    if feature_type == 'semantic':
        # Load from 07_pca output
        sig_file = (
            paths['semantic_results'] /
            f"sig_regions_{config['pca_template']}_sign-{config['pca_only_sign']}_iter-{config['pca_iter']}.npy"
        )
        sig_regions = np.load(sig_file)

    elif feature_type == 'dimensionality':
        # Load from 09_dimensionality output
        sig_file = paths['dimensionality_results'] / f"{config['dim_metric']}_sig_mmp.npy"
        sig_regions = np.load(sig_file)

    elif feature_type == 'combined':
        # Union of both semantic and dimensionality sig regions
        sem_sig = load_sig_regions_by_feature_type('semantic', paths, config)
        dim_sig = load_sig_regions_by_feature_type('dimensionality', paths, config)
        sig_regions = sem_sig | dim_sig  # Union

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return sig_regions


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def calculate_semantic_sig_regions(
    sem_list_td_train: List[npt.NDArray],
    sem_list_asd_train: List[npt.NDArray],
    id_list_td_train: List[str],
    id_list_asd_train: List[str],
    participants_df: pd.DataFrame,
    perf_mask: npt.NDArray,
    glm_alpha: float = 0.025
) -> npt.NDArray:
    """
    Calculate significant regions for semantic features using BrainStat GLM.

    Args:
        sem_list_td_train: Semantic PC scores for TD training subjects
        sem_list_asd_train: Semantic PC scores for ASD training subjects
        id_list_td_train: Subject IDs for TD training subjects
        id_list_asd_train: Subject IDs for ASD training subjects
        participants_df: Participant metadata
        perf_mask: Performance mask indices
        glm_alpha: FDR alpha threshold

    Returns:
        sig_regions_mask: Boolean mask of significant regions within perf_mask
    """
    # Prepare data for GLM
    id_list_all = id_list_td_train + id_list_asd_train
    sem_all = np.array(list(sem_list_td_train) + list(sem_list_asd_train))

    # Build demographic dataframe
    demo_df = pd.DataFrame({
        'Group': [participants_df.loc[sub_id, 'DX'] for sub_id in id_list_all],
        'Site': [participants_df.loc[sub_id, 'Site'] for sub_id in id_list_all],
        'Sex': [participants_df.loc[sub_id, 'Sex'] for sub_id in id_list_all],
        'Age': [float(participants_df.loc[sub_id, 'Age']) for sub_id in id_list_all],
        'MeanFD': [float(participants_df.loc[sub_id, 'Mean_FD_DM']) for sub_id in id_list_all]
    })

    # Build model
    term_group = FixedEffect(demo_df.Group)
    term_site = FixedEffect(demo_df.Site)
    term_sex = FixedEffect(demo_df.Sex)
    term_age = FixedEffect(demo_df.Age)
    term_meanfd = FixedEffect(demo_df.MeanFD)

    model = term_group + term_site + term_age + term_sex + term_meanfd
    contrast_group = (demo_df.Group == "ASD").astype(int) - (demo_df.Group == "TD").astype(int)

    # Fit SLM
    slm = SLM(model, contrast_group, correction=["fdr"], two_tailed=True)
    slm.fit(sem_all)

    # Calculate p-values using scipy
    pval = t_dist.sf(np.abs(slm.t), slm.df).flatten()
    pval_corrected = multipletests(pval, alpha=glm_alpha, method='fdr_bh')[1]

    # Identify significant regions
    sig_regions_mask = pval_corrected < glm_alpha

    return sig_regions_mask


def calculate_dim_sig_regions(
    dim_list_td_train: List[npt.NDArray],
    dim_list_asd_train: List[npt.NDArray],
    id_list_td_train: List[str],
    id_list_asd_train: List[str],
    participants_df: pd.DataFrame,
    glm_alpha: float = 0.025
) -> npt.NDArray:
    """
    Calculate significant regions for dimensionality features using BrainStat GLM.

    Args:
        dim_list_td_train: Dimensionality scores for TD training subjects
        dim_list_asd_train: Dimensionality scores for ASD training subjects
        id_list_td_train: Subject IDs for TD training subjects
        id_list_asd_train: Subject IDs for ASD training subjects
        participants_df: Participant metadata
        glm_alpha: FDR alpha threshold

    Returns:
        sig_regions_mask: Boolean mask of significant regions (360 MMP regions)
    """
    # Prepare data for GLM
    id_list_all = id_list_td_train + id_list_asd_train
    dim_all = np.array(list(dim_list_td_train) + list(dim_list_asd_train))

    # Build demographic dataframe
    demo_df = pd.DataFrame({
        'Group': [participants_df.loc[sub_id, 'DX'] for sub_id in id_list_all],
        'Site': [participants_df.loc[sub_id, 'Site'] for sub_id in id_list_all],
        'Sex': [participants_df.loc[sub_id, 'Sex'] for sub_id in id_list_all],
        'Age': [float(participants_df.loc[sub_id, 'Age']) for sub_id in id_list_all],
        'MeanFD': [float(participants_df.loc[sub_id, 'Mean_FD_DM']) for sub_id in id_list_all]
    })

    # Build model
    term_group = FixedEffect(demo_df.Group)
    term_site = FixedEffect(demo_df.Site)
    term_sex = FixedEffect(demo_df.Sex)
    term_age = FixedEffect(demo_df.Age)
    term_meanfd = FixedEffect(demo_df.MeanFD)

    model = term_group + term_site + term_age + term_sex + term_meanfd
    contrast_group = (demo_df.Group == "ASD").astype(int) - (demo_df.Group == "TD").astype(int)

    # Fit SLM
    slm = SLM(model, contrast_group, correction=["fdr"], two_tailed=True)
    slm.fit(dim_all)

    # Calculate p-values using scipy
    pval = t_dist.sf(np.abs(slm.t), slm.df).flatten()
    pval_corrected = multipletests(pval, alpha=glm_alpha, method='fdr_bh')[1]

    # Identify significant regions
    sig_regions_mask = pval_corrected < glm_alpha

    return sig_regions_mask


def residualize_data(
    data: npt.NDArray,
    covariates_train: npt.NDArray,
    covariates_test: Optional[npt.NDArray] = None
) -> npt.NDArray:
    """
    Residualize data by regressing out covariates.

    Args:
        data: Data to residualize (n_samples, n_features) or (n_samples,)
        covariates_train: Training covariates (n_train, n_cov)
        covariates_test: Optional test covariates (n_test, n_cov)

    Returns:
        Residualized data
    """
    # Add intercept
    n_train = covariates_train.shape[0]
    X_train_cov = np.column_stack([np.ones(n_train), covariates_train])

    # Handle 1D vs 2D data
    is_1d = (data.ndim == 1)
    if is_1d:
        data = data.reshape(-1, 1)

    if covariates_test is None:
        # Only training data
        data_residual_list = []

        for i in range(data.shape[1]):
            y_train = data[:, i]

            # Fit on training data
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train_cov, y_train)

            # Get residuals
            y_train_pred = reg.predict(X_train_cov)
            y_train_resid = y_train - y_train_pred
            data_residual_list.append(y_train_resid)

        data_residual = np.column_stack(data_residual_list) if len(data_residual_list) > 1 else data_residual_list[0]

    else:
        # Both training and test data
        n_test = covariates_test.shape[0]
        X_test_cov = np.column_stack([np.ones(n_test), covariates_test])

        data_train = data[:n_train]
        data_test = data[n_train:]

        data_residual_list = []

        for i in range(data.shape[1]):
            y_train = data_train[:, i]
            y_test = data_test[:, i]

            # Fit on training data only
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train_cov, y_train)

            # Get residuals for both train and test
            y_train_pred = reg.predict(X_train_cov)
            y_test_pred = reg.predict(X_test_cov)

            y_train_resid = y_train - y_train_pred
            y_test_resid = y_test - y_test_pred

            # Concatenate
            y_resid = np.concatenate([y_train_resid, y_test_resid])
            data_residual_list.append(y_resid)

        data_residual = np.column_stack(data_residual_list) if len(data_residual_list) > 1 else np.array(data_residual_list[0])

    # Convert back to 1D if input was 1D
    if is_1d:
        data_residual = data_residual.flatten()

    return data_residual


# =============================================================================
# CROSS-VALIDATION FUNCTIONS
# =============================================================================

def prepare_fold_info(
    sem_feature_behav: npt.NDArray,
    dim_feature_behav: npt.NDArray,
    id_list_all_behav: List[str],
    id_list_td: List[str],
    id_list_asd: List[str],
    participants_df: pd.DataFrame,
    perf_mask: npt.NDArray,
    behav_df: pd.DataFrame,
    covariates: npt.NDArray,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Pre-compute fold information including significant regions for each fold.

    Quality validation: Only accepts KFold splits where ALL folds have
    >= min_sig_regions in BOTH semantic and dimensionality features.

    Args:
        sem_feature_behav: Semantic features for behavioral subjects (n_behav_subjects, n_roi)
        dim_feature_behav: Dimensionality features for behavioral subjects (n_behav_subjects, 360)
        id_list_all_behav: Subject IDs for behavioral subjects (aligned with feature arrays)
        id_list_td: TD subject IDs (for filtering)
        id_list_asd: ASD subject IDs (for filtering)
        participants_df: Participant metadata
        perf_mask: Performance mask
        behav_df: Behavioral data
        covariates: Covariates array
        config: Configuration dictionary

    Returns:
        fold_info_list: List of dicts with train/test info and sig masks
    """
    n_rep = config['n_rep']
    n_cv = config['n_cv']
    glm_alpha = config['glm_alpha']
    min_sig_regions = config['min_sig_regions']

    fold_info_list = []
    target_folds = n_rep * n_cv
    seed_offset = 0
    n_splits_tried = 0
    n_splits_accepted = 0
    n_splits_rejected = 0
    rejection_reasons = []

    MAX_ATTEMPTS = 5000

    print(f"  Pre-computing fold information... (target: {target_folds} folds)")
    print(f"  Quality threshold: >={min_sig_regions} significant regions in BOTH sem and dim for ALL folds")

    with tqdm(total=target_folds, desc='    Collecting valid folds', leave=False) as pbar:
        while len(fold_info_list) < target_folds:
            if n_splits_tried >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Could not find {target_folds} valid folds after {MAX_ATTEMPTS} attempts.\n"
                    f"Only collected {len(fold_info_list)} folds."
                )

            # Create new KFold with unique random seed
            kf = KFold(n_splits=n_cv, shuffle=True, random_state=42 + seed_offset)
            n_splits_tried += 1

            # Temporary storage
            temp_fold_info = []
            all_valid = True
            min_sem_sig = float('inf')
            min_dim_sig = float('inf')

            # Pre-compute ALL folds in this split
            for train_ids, test_ids in kf.split(range(len(behav_df))):
                # Get subject IDs for training set
                train_subjects = [id_list_all_behav[i] for i in train_ids]
                id_list_td_train = [s for s in train_subjects if s in id_list_td]
                id_list_asd_train = [s for s in train_subjects if s in id_list_asd]

                # Get training indices (indices into behavioral feature arrays)
                td_train_indices = [id_list_all_behav.index(s) for s in id_list_td_train]
                asd_train_indices = [id_list_all_behav.index(s) for s in id_list_asd_train]

                # Extract semantic features (from behavioral-only array)
                sem_list_td_train = [sem_feature_behav[i] for i in td_train_indices]
                sem_list_asd_train = [sem_feature_behav[i] for i in asd_train_indices]

                # Calculate semantic significant regions
                sem_sig_mask = calculate_semantic_sig_regions(
                    sem_list_td_train, sem_list_asd_train,
                    id_list_td_train, id_list_asd_train,
                    participants_df, perf_mask, glm_alpha
                )

                # Extract dimensionality features (from behavioral-only array)
                dim_list_td_train = [dim_feature_behav[i] for i in td_train_indices]
                dim_list_asd_train = [dim_feature_behav[i] for i in asd_train_indices]

                # Calculate dimensionality significant regions
                dim_sig_mask = calculate_dim_sig_regions(
                    dim_list_td_train, dim_list_asd_train,
                    id_list_td_train, id_list_asd_train,
                    participants_df, glm_alpha
                )

                # Count significant regions
                n_sem_sig = np.sum(sem_sig_mask)
                n_dim_sig = np.sum(dim_sig_mask)

                min_sem_sig = min(min_sem_sig, n_sem_sig)
                min_dim_sig = min(min_dim_sig, n_dim_sig)

                if n_sem_sig < min_sig_regions or n_dim_sig < min_sig_regions:
                    all_valid = False

                fold_info = {
                    'train_ids': train_ids,
                    'test_ids': test_ids,
                    'sem_sig_mask': sem_sig_mask,
                    'dim_sig_mask': dim_sig_mask
                }
                temp_fold_info.append(fold_info)

            # Accept or reject this split
            if all_valid:
                fold_info_list.extend(temp_fold_info)
                n_splits_accepted += 1
                pbar.update(n_cv)
                pbar.set_description(
                    f'    Collecting folds: {n_splits_accepted} splits accepted ({n_splits_rejected} rejected)'
                )
            else:
                n_splits_rejected += 1
                rejection_reasons.append(f"min_sem={min_sem_sig}, min_dim={min_dim_sig}")

            seed_offset += 1

    print(f"  Collected {target_folds} valid folds from {n_splits_tried} KFold splits")
    print(f"    Acceptance rate: {n_splits_accepted}/{n_splits_tried} ({100*n_splits_accepted/n_splits_tried:.1f}%)")

    return fold_info_list


def _parallel_tqdm(delayed_tasks, n_jobs, total, desc):
    """Run delayed() tasks in parallel with a progress bar.

    Uses joblib's streaming generator (joblib >= 1.3) for a live completion bar
    when available; otherwise falls back to a plain parallel run (older joblib).
    Results are identical either way.
    """
    tasks = list(delayed_tasks)
    try:
        runner = Parallel(n_jobs=n_jobs, return_as="generator")
    except TypeError:
        runner = None
    if runner is not None:
        return list(tqdm(runner(tasks), total=total, desc=desc, leave=True))
    print(f"  {desc}: {total} tasks (n_jobs={n_jobs})...", flush=True)
    return Parallel(n_jobs=n_jobs)(tasks)


def run_cv_regression_with_fold_info(
    sem_feature_behav: npt.NDArray,
    dim_feature_behav: npt.NDArray,
    perf_mask: npt.NDArray,
    y_behav: pd.DataFrame,
    item_name: str,
    fold_info_list: List[Dict[str, Any]],
    config: Dict[str, Any],
    covariates: Optional[npt.NDArray] = None,
    feature_type: str = 'combined',
    desc_prefix: str = '',
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Run cross-validation with pre-computed fold information.

    Args:
        sem_feature_behav: Semantic features for behavioral subjects
        dim_feature_behav: Dimensionality features for behavioral subjects
        perf_mask: Performance mask
        y_behav: Behavioral data DataFrame
        item_name: Behavioral item name
        fold_info_list: Pre-computed fold information
        config: Configuration dictionary
        covariates: Optional covariates array
        feature_type: 'combined', 'semantic', or 'dimensionality'
        desc_prefix: Description prefix for progress bar
        return_predictions: If True, return predictions from last repetition

    Returns:
        Results dictionary with metrics
    """
    n_rep = config['n_rep']
    n_cv = config['n_cv']

    pearson_list = []
    r2_list = []
    mae_list = []

    desc = f'{desc_prefix}Repetitions' if desc_prefix else 'Repetitions'

    y_pred_final = None
    y_test_final = None

    def _one_rep(rep):
        y_pred_all = []
        y_test_all = []

        for fold in range(n_cv):
            fold_idx = rep * n_cv + fold
            fold_info = fold_info_list[fold_idx]

            train_ids = fold_info['train_ids']
            test_ids = fold_info['test_ids']
            sem_sig_mask = fold_info['sem_sig_mask']
            dim_sig_mask = fold_info['dim_sig_mask']

            # Extract features based on feature_type
            if feature_type == 'semantic':
                sem_full_train = np.zeros((len(train_ids), 360))
                sem_full_test = np.zeros((len(test_ids), 360))
                sem_full_train[:, perf_mask] = sem_feature_behav[train_ids]
                sem_full_test[:, perf_mask] = sem_feature_behav[test_ids]

                X_train = sem_full_train[:, perf_mask[sem_sig_mask]]
                X_test = sem_full_test[:, perf_mask[sem_sig_mask]]

            elif feature_type == 'dimensionality':
                X_train = dim_feature_behav[train_ids][:, dim_sig_mask]
                X_test = dim_feature_behav[test_ids][:, dim_sig_mask]

            elif feature_type == 'combined':
                sem_full_train = np.zeros((len(train_ids), 360))
                sem_full_test = np.zeros((len(test_ids), 360))
                sem_full_train[:, perf_mask] = sem_feature_behav[train_ids]
                sem_full_test[:, perf_mask] = sem_feature_behav[test_ids]

                sem_train = sem_full_train[:, perf_mask[sem_sig_mask]]
                sem_test = sem_full_test[:, perf_mask[sem_sig_mask]]

                dim_train = dim_feature_behav[train_ids][:, dim_sig_mask]
                dim_test = dim_feature_behav[test_ids][:, dim_sig_mask]

                X_train = np.concatenate([sem_train, dim_train], axis=1)
                X_test = np.concatenate([sem_test, dim_test], axis=1)

            # Get behavioral data
            y_train = y_behav.iloc[train_ids][item_name].values
            y_test = y_behav.iloc[test_ids][item_name].values

            # Residualize covariates
            if covariates is not None:
                cov_train = covariates[train_ids]
                cov_test = covariates[test_ids]

                X_combined = np.vstack([X_train, X_test])
                X_combined_resid = residualize_data(X_combined, cov_train, cov_test)
                X_train = X_combined_resid[:len(train_ids)]
                X_test = X_combined_resid[len(train_ids):]

                y_combined = np.concatenate([y_train, y_test])
                y_combined_resid = residualize_data(y_combined, cov_train, cov_test)
                y_train = y_combined_resid[:len(train_ids)]
                y_test = y_combined_resid[len(train_ids):]

            # Z-score normalization
            scaler = StandardScaler()
            X_train_z = scaler.fit_transform(X_train)
            X_test_z = scaler.transform(X_test)

            X_train_z[np.isnan(X_train_z)] = 0
            X_test_z[np.isnan(X_test_z)] = 0

            # Fit ElasticNetCV
            reg = ElasticNetCV(
                cv=5,
                random_state=0,
                l1_ratio=[.1, .3, .5, .7, .9],
                max_iter=10000
            ).fit(X_train_z, y_train)

            y_pred = reg.predict(X_test_z)

            y_pred_all.append(y_pred)
            y_test_all.append(y_test)

        # Aggregate predictions across folds
        y_pred_rep = np.concatenate(y_pred_all)
        y_test_rep = np.concatenate(y_test_all)

        # Calculate metrics for this repetition
        pearson_r = np.corrcoef(y_pred_rep, y_test_rep)[0, 1]
        r2 = r2_score(y_test_rep, y_pred_rep)
        mae = mean_absolute_error(y_test_rep, y_pred_rep)
        return pearson_r, r2, mae, y_pred_rep, y_test_rep

    rep_outputs = _parallel_tqdm(
        (delayed(_one_rep)(rep) for rep in range(n_rep)),
        config.get('n_jobs', -1), n_rep, f'  {desc}'
    )

    results = {
        'pearson': np.array([o[0] for o in rep_outputs]),
        'r2': np.array([o[1] for o in rep_outputs]),
        'mae': np.array([o[2] for o in rep_outputs])
    }

    if return_predictions:
        results['y_pred'] = rep_outputs[-1][3]
        results['y_test'] = rep_outputs[-1][4]

    return results


def run_permutation_test(
    sem_feature_behav: npt.NDArray,
    dim_feature_behav: npt.NDArray,
    perf_mask: npt.NDArray,
    y_behav: pd.DataFrame,
    item_name: str,
    paths: Dict[str, Path],
    config: Dict[str, Any],
    covariates: Optional[npt.NDArray] = None,
    feature_type: str = 'combined'
) -> npt.NDArray:
    """
    Run permutation test using feature_type-specific pre-calculated significant regions.

    Args:
        sem_feature_behav: Semantic features for behavioral subjects
        dim_feature_behav: Dimensionality features for behavioral subjects
        perf_mask: Performance mask
        y_behav: Behavioral data DataFrame
        item_name: Behavioral item name
        paths: Path dictionary
        config: Configuration dictionary
        covariates: Optional covariates array
        feature_type: 'combined', 'semantic', or 'dimensionality'

    Returns:
        Array of permutation correlations
    """
    n_perm = config['n_perm']
    n_cv = config['n_cv']

    # Load significant regions based on feature_type
    if feature_type == 'semantic':
        sem_sig_regions = load_sig_regions_by_feature_type('semantic', paths, config)
        sem_sig_mask = sem_sig_regions.nonzero()[0]
        print(f"    Using SEMANTIC significant regions: {len(sem_sig_mask)} ROIs")

    elif feature_type == 'dimensionality':
        dim_sig_regions = load_sig_regions_by_feature_type('dimensionality', paths, config)
        dim_sig_mask = dim_sig_regions.nonzero()[0]
        print(f"    Using DIMENSIONALITY significant regions: {len(dim_sig_mask)} ROIs")

    elif feature_type == 'combined':
        sem_sig_regions = load_sig_regions_by_feature_type('semantic', paths, config)
        dim_sig_regions = load_sig_regions_by_feature_type('dimensionality', paths, config)
        sem_sig_mask = sem_sig_regions.nonzero()[0]
        dim_sig_mask = dim_sig_regions.nonzero()[0]
        print(f"    Using COMBINED significant regions:")
        print(f"      Semantic: {len(sem_sig_mask)} ROIs")
        print(f"      Dimensionality: {len(dim_sig_mask)} ROIs")
        print(f"      Total features: {len(sem_sig_mask) + len(dim_sig_mask)}")

    # Create independent KFold for permutation testing
    kf_perm = KFold(n_splits=n_cv, shuffle=True, random_state=42)
    splits = list(kf_perm.split(range(len(y_behav))))
    seed = config.get('seed', 0)

    def _one_perm(perm):
        # per-perm RNG so permutations are reproducible under parallel execution
        y_perm = pd.Series(
            np.random.RandomState(seed + perm).permutation(y_behav[item_name].values),
            index=y_behav[item_name].index
        )

        y_pred_perm_all = []
        y_test_perm_all = []

        for train_ids, test_ids in splits:
            # Extract features using pre-calculated significant regions
            if feature_type == 'semantic':
                sem_full_train = np.zeros((len(train_ids), 360))
                sem_full_test = np.zeros((len(test_ids), 360))
                sem_full_train[:, perf_mask] = sem_feature_behav[train_ids]
                sem_full_test[:, perf_mask] = sem_feature_behav[test_ids]

                X_train = sem_full_train[:, sem_sig_mask]
                X_test = sem_full_test[:, sem_sig_mask]

            elif feature_type == 'dimensionality':
                X_train = dim_feature_behav[train_ids][:, dim_sig_mask]
                X_test = dim_feature_behav[test_ids][:, dim_sig_mask]

            elif feature_type == 'combined':
                sem_full_train = np.zeros((len(train_ids), 360))
                sem_full_test = np.zeros((len(test_ids), 360))
                sem_full_train[:, perf_mask] = sem_feature_behav[train_ids]
                sem_full_test[:, perf_mask] = sem_feature_behav[test_ids]

                sem_train = sem_full_train[:, sem_sig_mask]
                sem_test = sem_full_test[:, sem_sig_mask]

                dim_train = dim_feature_behav[train_ids][:, dim_sig_mask]
                dim_test = dim_feature_behav[test_ids][:, dim_sig_mask]

                X_train = np.concatenate([sem_train, dim_train], axis=1)
                X_test = np.concatenate([sem_test, dim_test], axis=1)

            # Get permuted behavioral data
            y_train = y_perm.iloc[train_ids].values
            y_test = y_perm.iloc[test_ids].values

            # Residualize covariates
            if covariates is not None:
                cov_train = covariates[train_ids]
                cov_test = covariates[test_ids]

                X_combined = np.vstack([X_train, X_test])
                X_combined_resid = residualize_data(X_combined, cov_train, cov_test)
                X_train = X_combined_resid[:len(train_ids)]
                X_test = X_combined_resid[len(train_ids):]

                y_combined = np.concatenate([y_train, y_test])
                y_combined_resid = residualize_data(y_combined, cov_train, cov_test)
                y_train = y_combined_resid[:len(train_ids)]
                y_test = y_combined_resid[len(train_ids):]

            # Z-score normalization
            scaler = StandardScaler()
            X_train_z = scaler.fit_transform(X_train)
            X_test_z = scaler.transform(X_test)

            X_train_z[np.isnan(X_train_z)] = 0
            X_test_z[np.isnan(X_test_z)] = 0

            # Fit model on permuted data
            reg = ElasticNetCV(
                cv=5,
                random_state=0,
                l1_ratio=[.1, .3, .5, .7, .9],
                max_iter=10000
            ).fit(X_train_z, y_train)

            y_pred = reg.predict(X_test_z)

            y_pred_perm_all.append(y_pred)
            y_test_perm_all.append(y_test)

        # Aggregate predictions
        y_pred_perm_agg = np.concatenate(y_pred_perm_all)
        y_test_perm_agg = np.concatenate(y_test_perm_all)

        # Correlation for this permutation
        return np.corrcoef(y_pred_perm_agg, y_test_perm_agg)[0, 1]

    pearson_perm_list = _parallel_tqdm(
        (delayed(_one_perm)(perm) for perm in range(n_perm)),
        config.get('n_jobs', -1), n_perm, '  Permutations'
    )
    return np.array(pearson_perm_list)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_null_distribution(
    observed_r: float,
    perm_dist: npt.NDArray,
    pval: float,
    item_name: str,
    save_path: Path,
    color: str = 'steelblue'
) -> None:
    """
    Plot permutation null distribution with observed value.

    Args:
        observed_r: Observed correlation
        perm_dist: Permutation distribution
        pval: p-value
        item_name: Behavioral item name
        save_path: Path to save figure
        color: Plot color
    """
    plt.figure(figsize=(10, 8))

    plt.hist(perm_dist, bins=20, color=color, alpha=0.7, edgecolor='black')
    plt.axvline(observed_r, color=color, linestyle='--', linewidth=5,
                label=f'Observed r = {observed_r:.3f}')

    plt.title(f'{item_name} - p-value = {pval:.3f}', fontsize=15, fontweight='bold')
    plt.xlabel('Pearson correlation coefficient', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if os.name == 'nt':
        plt.show()
    else:
        plt.close()


def plot_scatter_predicted_actual(
    y_pred: npt.NDArray,
    y_test: npt.NDArray,
    pearson_r: float,
    r2: float,
    item_name: str,
    save_path: Path,
    color: str = 'steelblue'
) -> None:
    """
    Plot predicted vs actual behavioral scores.

    Args:
        y_pred: Predicted values
        y_test: Actual values
        pearson_r: Pearson correlation
        r2: R-squared
        item_name: Behavioral item name
        save_path: Path to save figure
        color: Plot color
    """
    plt.figure(figsize=(10, 7))

    sns.regplot(
        x=y_pred,
        y=y_test,
        color=color,
        scatter_kws={'s': 150, 'alpha': 0.6},
        line_kws={'lw': 5}
    )

    plt.xlabel('Predicted behavioral score', fontsize=12)
    plt.ylabel('Actual behavioral score', fontsize=12)
    plt.title(f'{item_name}\nr = {pearson_r:.3f}, r2 = {r2:.3f}',
              fontsize=14, fontweight='bold')

    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if os.name == 'nt':
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    all_results: Dict[str, Dict[str, Any]],
    item_list: List[str],
    save_path: Path
) -> None:
    """
    Plot bar comparison of model performance across behavioral items.

    Horizontal bar plot with seaborn, stripplot overlay, and blue gradient palette.

    Args:
        all_results: Dictionary of results for all items and models
        item_list: List of behavioral items
        save_path: Path to save figure
    """
    # Prepare data as DataFrame for seaborn (individual repetition values)
    comparison_data = []
    for item in item_list:
        item_short = item.replace('SRS_', '').replace('_T', '')

        # Add semantic data points
        sem_pearson = all_results[item]['semantic']['pearson']
        for val in sem_pearson:
            comparison_data.append({
                'Behavioral item': item_short,
                'Feature Set': 'Semantic Only',
                'Pearson r': val
            })

        # Add dimensionality data points
        dim_pearson = all_results[item]['dimensionality']['pearson']
        for val in dim_pearson:
            comparison_data.append({
                'Behavioral item': item_short,
                'Feature Set': 'Dimensionality Only',
                'Pearson r': val
            })

        # Add combined data points
        combined_pearson = all_results[item]['combined']['pearson']
        for val in combined_pearson:
            comparison_data.append({
                'Behavioral item': item_short,
                'Feature Set': 'Semantic + Dim',
                'Pearson r': val
            })

    comparison_df = pd.DataFrame(comparison_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Fix the order so colors map predictably
    item_order = [item.replace('SRS_', '').replace('_T', '') for item in item_list]
    item_order = sorted(item_order)
    hue_order = ['Semantic Only', 'Dimensionality Only', 'Semantic + Dim']

    # Create blue gradient for the 3 feature types (light -> medium -> dark)
    blues_gradient = sns.color_palette("Blues", n_colors=6)
    feature_colors = {
        'Semantic Only': blues_gradient[1],       # Light blue
        'Dimensionality Only': blues_gradient[3], # Medium blue
        'Semantic + Dim': blues_gradient[5],      # Dark blue
    }

    # Plot barplot (let seaborn aggregate across repetitions)
    sns.barplot(
        data=comparison_df,
        y='Behavioral item',
        x='Pearson r',
        hue='Feature Set',
        order=item_order,
        hue_order=hue_order,
        palette=feature_colors,
        ax=ax,
        edgecolor='black',
        linewidth=1.2
    )

    # Overlay individual points (kept neutral so the gradients stay legible)
    sns.stripplot(
        data=comparison_df,
        y='Behavioral item',
        x='Pearson r',
        hue='Feature Set',
        order=item_order,
        hue_order=hue_order,
        ax=ax,
        color='black',
        size=5,
        alpha=0.45,
        dodge=True,
    )

    # Remove legend to reduce clutter
    ax.legend_.remove()

    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=14, fontweight='bold')
    ax.set_ylabel('SRS Subscale', fontsize=14, fontweight='bold')
    ax.set_title('Behavioral Prediction Performance:\nComparison of Feature Sets',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if os.name == 'nt':
        plt.show()
    else:
        plt.close()


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    """Main analysis workflow."""
    print("=" * 80)
    print("BEHAVIORAL PREDICTION: Semantic + Dimensionality Features (TD vs ASD)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  n_rep={CONFIG['n_rep']}, n_cv={CONFIG['n_cv']}, n_perm={CONFIG['n_perm']}")
    print(f"  glm_alpha={CONFIG['glm_alpha']}, min_sig_regions={CONFIG['min_sig_regions']}")

    # [1/10] Load participants
    participants_df, id_list_td, id_list_asd = load_participant_data(PATHS, CONFIG)

    # [2/10] Load semantic features
    sem_list_td, sem_list_asd = load_semantic_features(PATHS, CONFIG)

    # [3/10] Load dimensionality features
    dim_list_td, dim_list_asd = load_dimensionality_features(PATHS, CONFIG)

    # [4/10] Load performance mask
    perf_mask = load_performance_mask(PATHS, CONFIG)

    # [5/10] Load behavioral data and covariates
    behav_df, id_list_behav, covariates = load_behavioral_data(
        participants_df, id_list_td, id_list_asd, CONFIG
    )

    # Prepare features
    print("\n[6/10] Preparing features...")
    id_list_all = id_list_td + id_list_asd
    sem_feature_all = np.array(list(sem_list_td) + list(sem_list_asd))
    dim_feature_all = np.array(list(dim_list_td) + list(dim_list_asd))

    # Get indices for behavioral subjects
    id_list_behav_ids = [i for i, sub_id in enumerate(id_list_all) if sub_id in id_list_behav]
    id_list_all_behav = [id_list_all[i] for i in id_list_behav_ids]

    sem_feature_behav = sem_feature_all[id_list_behav_ids]
    dim_feature_behav = dim_feature_all[id_list_behav_ids]

    print(f"  Semantic features shape: {sem_feature_behav.shape}")
    print(f"  Dimensionality features shape: {dim_feature_behav.shape}")

    # Fixed significant regions (from main analyses) for the SHARP model comparison
    _sem_sig = load_sig_regions_by_feature_type('semantic', PATHS, CONFIG)
    _dim_sig = load_sig_regions_by_feature_type('dimensionality', PATHS, CONFIG)
    _sem_sig_idx = np.nonzero(_sem_sig)[0]
    _sem360 = np.zeros((sem_feature_behav.shape[0], 360))
    _sem360[:, perf_mask] = sem_feature_behav
    sem_feat_sharp = _sem360[:, _sem_sig_idx]
    dim_feat_sharp = dim_feature_behav[:, _dim_sig.astype(bool)]
    print(f"  SHARP fixed features: semantic={sem_feat_sharp.shape[1]}, "
          f"dimensionality={dim_feat_sharp.shape[1]}")

    # Define output path
    results_path = (
        PATHS['out'] /
        CONFIG['conf_option'] /
        CONFIG['atlas'] /
        f"k-{CONFIG['srm_option_pc']}-{CONFIG['srm_option_dim']}" /
        f"bias-{CONFIG['bias_weight']}_verb-{CONFIG['verb_weight']}" /
        f"wn-{CONFIG['reg_wordnet']}" /
        f"sa_sem-{CONFIG['enc_single_alpha_semantic']}_dim-{CONFIG['enc_single_alpha_dim']}"
    )
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"\n  Results path: {results_path}")

    # =========================================================================
    # MAIN ANALYSIS LOOP
    # =========================================================================
    print("\n" + "=" * 80)
    print("[7/10] MAIN ANALYSIS: BEHAVIORAL PREDICTION")
    print("=" * 80)

    item_list = CONFIG['behav_items']
    all_results = {}
    all_pvals = {}

    for item in item_list:
        print(f'\n{"=" * 80}')
        print(f'Processing: {item}')
        print("=" * 80)

        # Prepare fold info
        fold_info_file = results_path / f'fold_info_{item}.npz'

        if not fold_info_file.exists() or CONFIG['overwrite']:
            print("\n  Preparing fold information...")
            fold_info_list = prepare_fold_info(
                sem_feature_behav, dim_feature_behav, id_list_all_behav,
                id_list_td, id_list_asd, participants_df, perf_mask,
                behav_df, covariates, CONFIG
            )
            np.savez(fold_info_file, fold_info_list=fold_info_list)
            print(f"    Saved: {fold_info_file.name}")
        else:
            print(f"\n  Loading existing fold info: {fold_info_file.name}")
            fold_info_list = np.load(fold_info_file, allow_pickle=True)['fold_info_list']

        # Run models
        model_results = {}

        for feature_type in ['combined', 'semantic', 'dimensionality']:
            results_file = results_path / f'{feature_type}_{item}_results.npz'

            if not results_file.exists() or CONFIG['overwrite']:
                print(f"\n  Running {feature_type.upper()} model...")
                cv_results = run_cv_regression_with_fold_info(
                    sem_feature_behav, dim_feature_behav, perf_mask,
                    behav_df, item, fold_info_list, CONFIG,
                    covariates=covariates,
                    feature_type=feature_type,
                    desc_prefix=f'{feature_type.capitalize()} '
                )
                np.savez(results_file, **cv_results)
                print(f"    Saved: {results_file.name}")
            else:
                print(f"\n  Loading {feature_type}: {results_file.name}")
                cv_results = dict(np.load(results_file, allow_pickle=True))

            model_results[feature_type] = cv_results

            print(f'    Pearson r: {np.nanmean(cv_results["pearson"]):.4f}')
            print(f'    R2: {np.nanmean(cv_results["r2"]):.4f}')

        # Run permutation tests for each feature type
        perm_results = {}
        for feature_type in ['combined', 'semantic', 'dimensionality']:
            perm_file = results_path / f'{feature_type}_{item}_perm.npy'

            if not perm_file.exists() or CONFIG['overwrite']:
                print(f"\n  Running permutation test for {feature_type.upper()}...")
                perm_dist = run_permutation_test(
                    sem_feature_behav, dim_feature_behav, perf_mask,
                    behav_df, item, PATHS, CONFIG,
                    covariates=covariates,
                    feature_type=feature_type
                )
                np.save(perm_file, perm_dist)
                print(f"    Saved: {perm_file.name}")
            else:
                print(f"\n  Loading permutation: {perm_file.name}")
                perm_dist = np.load(perm_file)

            perm_results[feature_type] = perm_dist

            observed_r = np.nanmean(model_results[feature_type]['pearson'])
            pval = np.sum(perm_dist > observed_r) / len(perm_dist)
            print(f"    p-value: {pval:.4f}")

        all_results[item] = model_results
        all_pvals[item] = {
            ft: np.sum(perm_results[ft] > np.nanmean(model_results[ft]['pearson'])) / len(perm_results[ft])
            for ft in ['combined', 'semantic', 'dimensionality']
        }

    # =========================================================================
    # FDR CORRECTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("[8/10] FDR CORRECTION")
    print("=" * 80)

    for feature_type in ['combined', 'semantic', 'dimensionality']:
        print(f"\n  {feature_type.upper()} model:")
        pvals = [all_pvals[item][feature_type] for item in item_list]
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

        print(f"  {'Item':<15} | {'p-raw':>8} | {'p-FDR':>8} | {'Sig':>5}")
        print("  " + "-" * 50)
        for i, item in enumerate(item_list):
            sig = "***" if pvals_corrected[i] < 0.001 else "**" if pvals_corrected[i] < 0.01 else "*" if pvals_corrected[i] < 0.05 else "ns"
            print(f"  {item:<15} | {pvals[i]:>8.4f} | {pvals_corrected[i]:>8.4f} | {sig:>5}")

    # =========================================================================
    # MODEL COMPARISON (valid tests under cross-validation; Zeng et al., 2026):
    #   SHARP score test + corrected resampled t-test (Nadeau & Bengio, 2003).
    # Replaces the fold-dependence-ignoring paired Wilcoxon.
    # Uses fixed significant regions (from the main analysis) for all folds.
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"[8.5/10] MODEL COMPARISON (SHARP + corrected resampled t-test, K={CONFIG['sharp_K']}, J={CONFIG['sharp_J']})")
    print("=" * 80)

    rows = sharp_test.run_model_comparison(
        sem_feat_sharp, dim_feat_sharp, behav_df, item_list, covariates, CONFIG)
    trend_rows = sharp_test.cross_subscale_trend_barplot(
        all_results, item_list, CONFIG.get('one_sided_combined', True))

    # Save statistical comparison results
    pd.DataFrame(rows).to_csv(results_path / 'model_comparison_sharp.csv', index=False)
    pd.DataFrame(trend_rows).to_csv(results_path / 'model_comparison_trend.csv', index=False)
    print(f"\n  Saved: {results_path / 'model_comparison_sharp.csv'}")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[9/10] VISUALIZATIONS")
    print("=" * 80)

    fig_path = PATHS['fig']
    palette = sns.color_palette("Blues", n_colors=len(item_list) + 2)[2:]

    # Null distribution plots
    print("\n  Creating null distribution plots...")
    for i, item in enumerate(item_list):
        combined_perm = np.load(results_path / f'combined_{item}_perm.npy')
        observed_r = np.nanmean(all_results[item]['combined']['pearson'])
        pval = all_pvals[item]['combined']

        plot_null_distribution(
            observed_r, combined_perm, pval, item,
            fig_path / f'null_dist_{item}.png',
            color=palette[i]
        )

    # Scatter plots: predicted vs actual
    print("\n  Creating scatter plots of predicted vs actual behavioral scores...")

    # Create config for single repetition (for scatter plot predictions)
    config_single_rep = CONFIG.copy()
    config_single_rep['n_rep'] = 1

    for i, item in enumerate(item_list):
        print(f"    Creating scatter plot for {item}...")

        # Load fold info (only need first n_cv folds for 1 repetition)
        fold_info_file = results_path / f'fold_info_{item}.npz'
        fold_info_list_full = np.load(fold_info_file, allow_pickle=True)['fold_info_list']
        fold_info_list_single = list(fold_info_list_full[:CONFIG['n_cv']])

        # Run CV with predictions returned (only 1 repetition to get predictions)
        cv_results_with_pred = run_cv_regression_with_fold_info(
            sem_feature_behav, dim_feature_behav, perf_mask,
            behav_df, item, fold_info_list_single, config_single_rep,
            covariates=covariates,
            feature_type='combined',
            desc_prefix='',
            return_predictions=True
        )

        y_pred = cv_results_with_pred['y_pred']
        y_test = cv_results_with_pred['y_test']

        # Get metrics from full results
        combined_pearson = all_results[item]['combined']['pearson']
        combined_r2 = all_results[item]['combined']['r2']

        # Create scatter plot
        plot_scatter_predicted_actual(
            y_pred, y_test,
            np.nanmean(combined_pearson),
            np.nanmean(combined_r2),
            item,
            fig_path / f'scatter_{item}.png',
            color=palette[i]
        )

    # Model comparison bar plot
    print("\n  Creating model comparison plot...")
    plot_model_comparison(all_results, item_list, fig_path / 'model_comparison.png')

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("[10/10] SAVING SUMMARY")
    print("=" * 80)

    # Save summary statistics
    summary_data = []
    for item in item_list:
        for feature_type in ['combined', 'semantic', 'dimensionality']:
            summary_data.append({
                'item': item,
                'feature_type': feature_type,
                'pearson_mean': np.nanmean(all_results[item][feature_type]['pearson']),
                'pearson_std': np.nanstd(all_results[item][feature_type]['pearson']),
                'r2_mean': np.nanmean(all_results[item][feature_type]['r2']),
                'r2_std': np.nanstd(all_results[item][feature_type]['r2']),
                'pval_raw': all_pvals[item][feature_type]
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_path / 'model_comparison_statistics.csv', index=False)
    print(f"  Saved: {results_path / 'model_comparison_statistics.csv'}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
