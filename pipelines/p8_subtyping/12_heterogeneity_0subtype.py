"""
12_heterogeneity_0subtype.py
ASD Heterogeneity Analysis using Z-Score Based Triple Network Data.

This script analyzes ASD heterogeneity using z-score data from 11_sem_zscore pipeline.
Adapted from 26_normative/02_analysis_gam_release11_triple_network.py.

Analyses:
1. Outlier prevalence analysis (overall and age-stratified)
2. K-means clustering with elbow method
3. Hierarchical clustering with dendrogram
4. Cluster characterization (demographics + MANOVA + univariate tests)
5. Comorbidity analysis

Input: Z-score CSVs from 11_sem_zscore (6 features: sem_sal/dmn/cen, dim_sal/dmn/cen)
Output: Figures saved to 99_main/claude_figures/heterogeneity/
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import (PROJECT, TEMPLATES, fig_dir,  # noqa: E402
                            HBN_PHENOTYPE, require)
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, fisher_exact, chi2_contingency, pearsonr, shapiro, binomtest
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols as smf_ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '12_heterogeneity',

    # Source data pipelines
    'source_pipeline_participants': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_pipeline_comorbidity': '25_sensitivity',
    'source_task_comorbidity': '00_comorbidity_release11',

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

    # Behavioral items
    'behav_items_srs': ['SRS_AWR_T', 'SRS_COG_T', 'SRS_COM_T', 'SRS_MOT_T', 'SRS_RRB_T'],
    'behav_items_nih': ['NIH7_Card_P', 'NIH7_Flanker_P', 'NIH7_List_P', 'NIH7_Pattern_P'],

    # Heterogeneity analysis parameters
    # K-means determinism, mirroring config/params.yml.  Previously only a
    # function default, so the seed behind the published 58/70 split was
    # invisible from the configuration.
    'random_state': 42,
    'n_init': 50,
    'outlier_threshold': 1.645,  # z-score threshold (options: 1.96(95%), 1.645(90%), 1.44(85%), 1.28(80%))
    'max_k': 5,                 # Maximum clusters to test
    'fdr_alpha': 0.05,          # FDR correction threshold

    # Age stratification (matching original 26_normative script)
    'age_bins': [0, 10, 14, 18, float('inf')],
    'age_labels': ['≤10', '10-14', '14-18', '18+'],

    # Age-based screening (must match 11_sem_zscore_0save_raw.py)
    'screen': True,  # Used for path construction only; actual screening done upstream
    'sig_overlap': False,  # Use overlap significance mask (matching 0save_raw.py)

    # Clustering normalization mode
    'normalize_profiles': True,  # True = profile-normalized (subtype patterns), False = original z-scores (severity-based)

    # Manual cluster number override (set to None to use automatic selection)
    'manual_k': None,  # None = auto (silhouette-optimal), or set integer (e.g., 2, 3, 4, 5)

    # Clustering methods to run: 'both', 'kmeans', or 'hierarchical'
    'clustering_method': 'kmeans',
}

# Triple network definitions
NETWORK_NAMES = ['sal', 'dmn', 'cen']
NETWORK_LABELS = {
    'sal': 'Salience Network',
    'dmn': 'Default Mode Network',
    'cen': 'Central Executive Network'
}
FEATURE_TYPES = ['sem', 'dim']
FEATURE_LABELS = {'sem': 'Semantic', 'dim': 'Dimensional'}

# Z-score variable names
ZSCORE_VARS = ['sem_sal', 'sem_dmn', 'sem_cen', 'dim_sal', 'dim_dmn', 'dim_cen']

# Feature ordering used for radar profile plots (grouped for readability)
# ZSCORE_VARS_RADAR = ['dim_sal', 'dim_dmn', 'sem_dmn', 'sem_sal', 'sem_cen', 'dim_cen']
ZSCORE_VARS_RADAR = ['dim_cen', 'dim_sal', 'sem_sal', 'sem_cen', 'sem_dmn', 'dim_dmn']

# Per-cluster colors, shared by the radar plot and the trajectory scatter/density
# overlays so a cluster keeps the same color across every figure.
CLUSTER_COLORS = ['#2ca02c', '#9467bd', '#195b74', '#b58a1c', '#d62728']


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
        'code': proj_root / '1_code' / pipe,
        'pipe': proj_root / '2_pipeline' / pipe,

        # Participants data
        'participants_df': proj_root / '2_pipeline' /
                          config['source_pipeline_participants'] /
                          config['source_task_participants'] / 'out',

        # Comorbidity data
        'comorbidity': proj_root / '2_pipeline' /
                      config['source_pipeline_comorbidity'] /
                      config['source_task_comorbidity'] / 'out',

        # NIH Toolbox cognition, from the access-controlled HBN LORIS release.
        # require() fails with a message naming the config key rather than
        # letting a missing file surface later as an empty merge.
        'nih_data': require(HBN_PHENOTYPE, 'hbn_phenotype',
                            'the NIH Toolbox scores used by the SEM'
                            ) / 'HBN' / 'Phenotypic_LORIS' / 'NIH.csv',
    }

    # Z-score data path (from 11_sem_zscore)
    paths['zscore_data'] = (
        paths['pipe'] / '11_sem_zscore' / 'out' /
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

    # Figures.  The original wrote these into the source tree
    # (1_code/99_main/claude_figures/), so a run modified the code checkout.
    branch = f"screen-{config['screen']}_overlap-{config['sig_overlap']}"
    paths['figures'] = fig_dir(__file__) / branch
    paths['figures'].mkdir(parents=True, exist_ok=True)

    # Data outputs.  The original had none: cluster_assignments_*.csv - a
    # required input to 12_heterogeneity_1save_data.py - was written into the
    # figure directory, making a figures folder a pipeline dependency.
    # Keyed on screen/overlap only, not the full parameter tree, because the
    # full tree pushes filenames past the Windows 260-character limit (the same
    # reason 12_heterogeneity_1save_data.py gives for its own output path).
    paths['out'] = paths['pipe'] / config['task'] / 'out' / branch
    paths['out'].mkdir(parents=True, exist_ok=True)

    return paths


PATHS = setup_paths(CONFIG)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_zscore_data(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load z-score CSVs from 11_sem_zscore_1gam.R output."""
    print("\n[1/4] Loading z-score data...")

    zscore_td = pd.read_csv(paths['zscore_data'] / 'zscore_td.csv', index_col=0)
    zscore_asd = pd.read_csv(paths['zscore_data'] / 'zscore_asd.csv', index_col=0)

    print(f"  TD: {zscore_td.shape[0]} subjects, {zscore_td.shape[1]} columns")
    print(f"  ASD: {zscore_asd.shape[0]} subjects, {zscore_asd.shape[1]} columns")

    return zscore_td, zscore_asd


def load_behavioral_data(
    paths: Dict[str, Path],
    id_list: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Load behavioral data (SRS + NIH) from participants_df."""
    print("\n[2/4] Loading behavioral data...")

    # Load participants DataFrame
    participants_df = pd.read_excel(
        paths['participants_df'] / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    # Filter to subjects in z-score data
    participants_df = participants_df.loc[id_list]
    print(f"  Participants loaded: {participants_df.shape[0]} subjects")

    # Load NIH cognitive data
    nih_item_list = [
        'NIH_Scores,NIH7_Card_P',
        'NIH_Scores,NIH7_Flanker_P',
        'NIH_Scores,NIH7_List_P',
        'NIH_Scores,NIH7_Pattern_P'
    ]

    if paths['nih_data'].exists():
        nih_df = pd.read_csv(paths['nih_data'], index_col=0)

        # Initialize NIH columns
        for item in nih_item_list:
            col_name = item.split(',')[1]
            participants_df[col_name] = np.nan

        # Map and merge NIH data
        n_found = 0
        for sub_id in id_list:
            sub_id_no_sub_assess = sub_id.split('-')[1] + ',assessment'
            if sub_id_no_sub_assess in nih_df.index:
                for item in nih_item_list:
                    col_name = item.split(',')[1]
                    value = nih_df.loc[sub_id_no_sub_assess, item]
                    try:
                        participants_df.loc[sub_id, col_name] = float(value)
                    except (ValueError, TypeError):
                        participants_df.loc[sub_id, col_name] = np.nan
                n_found += 1

        print(f"  NIH data matched: {n_found}/{len(id_list)} subjects")
    else:
        print(f"  Warning: NIH data not found at {paths['nih_data']}")

    # Report behavioral data availability
    behav_items = config['behav_items_srs'] + config['behav_items_nih']
    print("  Behavioral data availability:")
    for item in behav_items:
        if item in participants_df.columns:
            n_valid = participants_df[item].notna().sum()
            print(f"    {item}: {n_valid}/{len(id_list)} ({100*n_valid/len(id_list):.1f}%)")

    return participants_df


def load_comorbidity_data(
    paths: Dict[str, Path],
    id_list: List[str]
) -> Optional[pd.DataFrame]:
    """Load and categorize comorbidity data."""
    print("\n[3/4] Loading comorbidity data...")

    comorbid_path = paths['comorbidity'] / 'comorbid_df.csv'

    if not comorbid_path.exists():
        print(f"  Warning: Comorbidity data not found at {comorbid_path}")
        return None

    comorbid_df_raw = pd.read_csv(comorbid_path, index_col=0)

    # Define comorbidity categories
    adhd_list = [
        'ADHD-Combined Type',
        'ADHD-Inattentive Type',
        'ADHD-Hyperactive/Impulsive Type',
        'Other Specified Attention-Deficit/Hyperactivity Disorder'
    ]
    internalizing_list = [
        'Generalized Anxiety Disorder',
        'Separation Anxiety',
        'Social Anxiety (Social Phobia)',
        'Specific Phobia',
        'Other Specified Anxiety Disorder',
        'Major Depressive Disorder',
        'Persistent Depressive Disorder (Dysthymia)',
        'Other Specified Depressive Disorder',
        'Adjustment Disorders',
        'Posttraumatic Stress Disorder'
    ]
    externalizing_list = [
        'Oppositional Defiant Disorder',
        'Disruptive Mood Dysregulation Disorder',
        'Other Specified Disruptive, Impulse-Control, and Conduct Disorder',
        'Encopresis',
        'Enuresis',
        'Trichotillomania (Hair-Pulling Disorder)'
    ]
    learning_list = [
        'Specific Learning Disorder with Impairment in Reading',
        'Specific Learning Disorder with Impairment in Mathematics',
        'Specific Learning Disorder with Impairment in Written Expression',
        'Language Disorder',
        'Speech Sound Disorder',
        'Borderline Intellectual Functioning',
        'Intellectual Disability-Mild'
    ]

    # Create simplified 4-category binary flags
    comorbid_df = pd.DataFrame(index=comorbid_df_raw.index)

    # ADHD
    adhd_cols = [col for col in adhd_list if col in comorbid_df_raw.columns]
    comorbid_df['ADHD'] = (comorbid_df_raw[adhd_cols].sum(axis=1) > 0).astype(int) if adhd_cols else 0

    # Internalizing
    int_cols = [col for col in internalizing_list if col in comorbid_df_raw.columns]
    comorbid_df['Internalizing'] = (comorbid_df_raw[int_cols].sum(axis=1) > 0).astype(int) if int_cols else 0

    # Externalizing
    ext_cols = [col for col in externalizing_list if col in comorbid_df_raw.columns]
    comorbid_df['Externalizing'] = (comorbid_df_raw[ext_cols].sum(axis=1) > 0).astype(int) if ext_cols else 0

    # Learning
    learn_cols = [col for col in learning_list if col in comorbid_df_raw.columns]
    comorbid_df['Learning'] = (comorbid_df_raw[learn_cols].sum(axis=1) > 0).astype(int) if learn_cols else 0

    # Filter to ASD subjects
    common_subjects = comorbid_df.index.intersection(id_list)
    comorbid_df = comorbid_df.loc[common_subjects]

    print(f"  Comorbidity data loaded: {len(comorbid_df)} subjects")
    print("  Prevalence:")
    for cat in ['ADHD', 'Internalizing', 'Externalizing', 'Learning']:
        n_pos = comorbid_df[cat].sum()
        pct = 100 * n_pos / len(comorbid_df)
        print(f"    {cat:15s}: {n_pos:3d}/{len(comorbid_df):3d} ({pct:5.1f}%)")

    return comorbid_df


def filter_complete_cases(
    zscore_asd: pd.DataFrame,
    participants_df: pd.DataFrame,
    behav_items: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Filter to complete cases with all behavioral measures."""
    print("\n[4/4] Filtering to complete cases...")

    # Get common subjects
    common_ids = list(zscore_asd.index.intersection(participants_df.index))

    # Make a copy and convert behavioral columns to numeric (handles string 'nan', etc.)
    participants_df = participants_df.copy()
    for item in behav_items:
        if item in participants_df.columns:
            participants_df[item] = pd.to_numeric(participants_df[item], errors='coerce')

    # Filter to complete cases using vectorized operation
    behav_subset = participants_df.loc[common_ids, behav_items]
    complete_case_mask = behav_subset.notna().all(axis=1)
    complete_ids = list(behav_subset[complete_case_mask].index)

    n_excluded = len(common_ids) - len(complete_ids)
    print(f"  Excluded {n_excluded} subjects with missing behavioral data")
    print(f"  Complete cases: {len(complete_ids)}/{len(common_ids)}")

    zscore_filtered = zscore_asd.loc[complete_ids]
    participants_filtered = participants_df.loc[complete_ids]

    # Verify no NaN values remain in behavioral data
    n_nans = participants_filtered[behav_items].isna().sum().sum()
    if n_nans > 0:
        print(f"  WARNING: {n_nans} NaN values still remain!")
    else:
        print(f"  Verified: No NaN values in complete case behavioral data")

    return zscore_filtered, participants_filtered, complete_ids


# =============================================================================
# GAM TRAJECTORY DATA LOADING FUNCTIONS
# =============================================================================

def load_gam_trajectories(paths: Dict[str, Path]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load GAM trajectory predictions from CSV files generated by R.

    Returns:
        trajectories: Dict with structure {feature: {group: DataFrame}}
            where feature is 'sem' or 'dim' and group is 'td' or 'asd'
    """
    print("\n  Loading GAM trajectory predictions...")
    trajectories = {}

    for feature in ['sem', 'dim']:
        trajectories[feature] = {}
        for group in ['td', 'asd']:
            filename = f'gam_trajectory_{group}_male_{feature}.csv'
            filepath = paths['zscore_data'] / filename
            if filepath.exists():
                trajectories[feature][group] = pd.read_csv(filepath)
                print(f"    Loaded: {filename}")
            else:
                print(f"    Warning: {filename} not found")
                trajectories[feature][group] = None

    return trajectories


def load_raw_network_data(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load raw network scores for scatter plot overlay.

    Returns:
        raw_data: Dict with keys like 'sem_td', 'sem_asd', 'dim_td', 'dim_asd'
    """
    print("\n  Loading raw network data for scatter plots...")
    raw_data = {}

    for prefix in ['sem', 'dim']:
        for group in ['td', 'asd']:
            key = f'{prefix}_{group}'
            filename = f'{prefix}_data_{group}_raw.csv'
            filepath = paths['zscore_data'] / filename
            if filepath.exists():
                raw_data[key] = pd.read_csv(filepath, index_col=0)
                print(f"    Loaded: {filename} ({len(raw_data[key])} subjects)")
            else:
                print(f"    Warning: {filename} not found")
                raw_data[key] = None

    return raw_data


# =============================================================================
# OUTLIER ANALYSIS FUNCTIONS
# =============================================================================

def identify_outliers(z_scores: np.ndarray, threshold: float = 1.96) -> np.ndarray:
    """Identify outliers based on z-score threshold."""
    return np.abs(z_scores) > threshold


def calculate_overall_prevalence(
    zscore_asd: pd.DataFrame,
    threshold: float = 1.96
) -> pd.DataFrame:
    """Calculate outlier prevalence for each network/feature."""
    results = []

    for var in ZSCORE_VARS:
        z_scores = zscore_asd[var].values
        outlier_mask = identify_outliers(z_scores, threshold)
        n_outliers = outlier_mask.sum()
        n_total = len(outlier_mask)

        # Parse variable name (e.g., 'sem_sal' -> feature='sem', network='sal')
        parts = var.split('_')
        feature = parts[0]
        network = parts[1]

        results.append({
            'variable': var,
            'feature': feature,
            'network': network,
            'n_outliers': n_outliers,
            'n_total': n_total,
            'prevalence_%': 100 * n_outliers / n_total
        })

    return pd.DataFrame(results)


def calculate_age_stratified_prevalence(
    zscore_asd: pd.DataFrame,
    var_name: str,
    age_bins: List[float],
    age_labels: List[str],
    threshold: float = 1.96
) -> pd.DataFrame:
    """Calculate outlier prevalence by age groups for a single variable."""
    z_scores = zscore_asd[var_name].values
    ages = zscore_asd['age'].values
    outlier_mask = identify_outliers(z_scores, threshold)

    # Create age groups
    age_groups = pd.cut(ages, bins=age_bins, labels=age_labels, right=True, include_lowest=True)

    results = []
    for age_label in age_labels:
        age_mask = (age_groups == age_label)
        n_total = age_mask.sum()
        n_outliers = (outlier_mask & age_mask).sum()
        prevalence = 100 * n_outliers / n_total if n_total > 0 else 0

        results.append({
            'age_group': age_label,
            'n_outliers': n_outliers,
            'n_total': n_total,
            'prevalence_%': prevalence
        })

    return pd.DataFrame(results)


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def create_zscore_matrix_6d(zscore_asd: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Create 6D z-score matrix for clustering."""
    feature_names = ZSCORE_VARS
    zscore_matrix = zscore_asd[feature_names].values
    return zscore_matrix, feature_names


def create_binary_outlier_matrix_6d(
    zscore_asd: pd.DataFrame,
    threshold: float = 1.96
) -> Tuple[np.ndarray, List[str]]:
    """Create 6D binary outlier matrix for clustering."""
    feature_names = ZSCORE_VARS
    n_subjects = len(zscore_asd)
    binary_matrix = np.zeros((n_subjects, len(feature_names)), dtype=int)

    for i, var in enumerate(feature_names):
        z_scores = zscore_asd[var].values
        binary_matrix[:, i] = identify_outliers(z_scores, threshold).astype(int)

    return binary_matrix, feature_names


def create_profile_normalized_matrix(zscore_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Create profile-normalized z-score matrix (per-subject demeaning).

    Removes global severity effect by subtracting each subject's mean z-score
    across all features. This centers each profile at zero, so clustering
    will find patterns (which networks are relatively high/low) rather than
    magnitude (overall deviation from norm).

    Args:
        zscore_df: DataFrame with z-scores for each subject

    Returns:
        profile_matrix: Profile-normalized z-score matrix (n_subjects, n_features)
        feature_names: List of feature names
    """
    feature_names = ZSCORE_VARS
    zscore_matrix = zscore_df[feature_names].values

    # Per-subject demeaning: subtract row mean from each row
    subject_means = zscore_matrix.mean(axis=1, keepdims=True)
    profile_matrix = zscore_matrix - subject_means

    return profile_matrix, feature_names


def kmeans_clustering_with_elbow(
    feature_matrix: np.ndarray,
    max_k: int = 5,
    random_state: int = 42
) -> Tuple[Dict, Dict]:
    """K-means clustering with elbow method evaluation."""
    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores_list = []
    cluster_models = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=50)
        labels = kmeans.fit_predict(feature_matrix)

        inertias.append(kmeans.inertia_)

        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(feature_matrix, labels)
            silhouette_scores_list.append(sil_score)
        else:
            silhouette_scores_list.append(np.nan)

        cluster_models[k] = kmeans

    # Find elbow
    try:
        knee_locator = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
        optimal_k_elbow = knee_locator.elbow
    except:
        optimal_k_elbow = None

    # Best silhouette
    optimal_k_silhouette = k_values[np.nanargmax(silhouette_scores_list)]

    results_dict = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_silhouette': optimal_k_silhouette
    }

    return results_dict, cluster_models


def hierarchical_clustering_with_metrics(
    feature_matrix: np.ndarray,
    max_k: int = 5,
    method: str = 'ward',
    metric: str = 'euclidean'
) -> Tuple[Dict, np.ndarray]:
    """Hierarchical clustering with metrics evaluation."""
    # Compute distance matrix
    dist_matrix = pdist(feature_matrix, metric=metric)

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method=method)

    # Evaluate different k values
    k_values = list(range(2, max_k + 1))
    silhouette_scores_list = []

    for k in k_values:
        labels = fcluster(linkage_matrix, k, criterion='maxclust')

        if len(np.unique(labels)) > 1:
            dist_matrix_full = squareform(dist_matrix)
            sil_score = silhouette_score(dist_matrix_full, labels, metric='precomputed')
            silhouette_scores_list.append(sil_score)
        else:
            silhouette_scores_list.append(np.nan)

    optimal_k_silhouette = k_values[np.nanargmax(silhouette_scores_list)]

    results_dict = {
        'k_values': k_values,
        'silhouette_scores': silhouette_scores_list,
        'optimal_k_silhouette': optimal_k_silhouette,
        'method': method
    }

    return results_dict, linkage_matrix


# =============================================================================
# CLUSTER CHARACTERIZATION FUNCTIONS
# =============================================================================

def characterize_clusters(
    binary_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    feature_names: List[str],
    behav_data: Optional[pd.DataFrame] = None,
    behav_item_list: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Characterize clusters by demographics and behavioral measures."""
    n_clusters = len(np.unique(cluster_labels))

    # Cluster demographics
    cluster_summary = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)

        cluster_summary.append({
            'cluster': cluster_id,
            'n_subjects': cluster_mask.sum(),
            'age_mean': ages[cluster_mask].mean(),
            'age_std': ages[cluster_mask].std(),
            'n_male': (sexes[cluster_mask] == 0).sum(),
            'n_female': (sexes[cluster_mask] == 1).sum()
        })

    cluster_summary_df = pd.DataFrame(cluster_summary)
    cluster_summary_df['sex_ratio_m_f'] = cluster_summary_df['n_male'] / cluster_summary_df['n_female'].replace(0, np.nan)

    # Age test
    if n_clusters == 2:
        cluster_0_ages = ages[cluster_labels == 0]
        cluster_1_ages = ages[cluster_labels == 1]
        stat_age, p_age = ttest_ind(cluster_0_ages, cluster_1_ages)
        test_type_age = 't-test'
    else:
        age_groups = [ages[cluster_labels == i] for i in range(n_clusters)]
        stat_age, p_age = f_oneway(*age_groups)
        test_type_age = 'ANOVA'

    # Sex test
    contingency_sex = pd.crosstab(sexes, cluster_labels)
    if n_clusters == 2 and contingency_sex.shape[0] == 2:
        stat_sex, p_sex = fisher_exact(contingency_sex.values)
        test_type_sex = 'Fisher'
    else:
        stat_sex, p_sex, _, _ = chi2_contingency(contingency_sex)
        test_type_sex = 'Chi-square'

    cluster_summary_df['test_age'] = test_type_age
    cluster_summary_df['stat_age'] = stat_age
    cluster_summary_df['p_age'] = p_age
    cluster_summary_df['test_sex'] = test_type_sex
    cluster_summary_df['stat_sex'] = stat_sex
    cluster_summary_df['p_sex'] = p_sex

    # Feature outlier proportions per cluster
    cluster_features = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        feature_props = binary_matrix[cluster_mask].mean(axis=0) * 100

        feature_dict = {'cluster': cluster_id}
        for feat_name, prop in zip(feature_names, feature_props):
            feature_dict[feat_name] = prop
        cluster_features.append(feature_dict)

    cluster_feature_df = pd.DataFrame(cluster_features)

    # Behavioral analysis
    behav_summary_df = None
    if behav_data is not None and behav_item_list is not None:
        behav_results = []
        unique_clusters = np.unique(cluster_labels)

        # MANOVA analysis
        print("\n" + "=" * 70)
        print("MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)")
        print("=" * 70)
        print("Note: MANOVA assumes multivariate normality and homogeneous covariances.")
        print("      Clinical behavioral data may violate these assumptions.")

        srs_list = [item for item in behav_item_list if 'SRS' in item]
        nih_list = [item for item in behav_item_list if 'NIH' in item]

        domains = {
            'SRS (Social)': srs_list,
            'NIH (Cognitive)': nih_list
        }

        manova_df = behav_data.copy()
        manova_df['Cluster'] = cluster_labels.astype(str)

        for domain_name, var_list in domains.items():
            if len(var_list) == 0:
                continue

            print(f"\n---> Testing Domain: {domain_name}")
            print(f"     Variables ({len(var_list)}): {', '.join(var_list)}")

            missing_cols = [c for c in var_list if c not in manova_df.columns]
            if missing_cols:
                print(f"     Skipping! Missing columns: {missing_cols}")
                continue

            # Check normality assumption (Shapiro-Wilk test)
            print(f"     Normality check (Shapiro-Wilk, p < 0.05 = non-normal):")
            non_normal_vars = []
            for var in var_list:
                if var in manova_df.columns:
                    var_data = manova_df[var].dropna()
                    if len(var_data) >= 3:  # Shapiro-Wilk requires at least 3 samples
                        stat_sw, p_sw = shapiro(var_data)
                        if p_sw < 0.05:
                            non_normal_vars.append(var)
                            print(f"       {var}: W={stat_sw:.3f}, p={p_sw:.4f} *")
                        else:
                            print(f"       {var}: W={stat_sw:.3f}, p={p_sw:.4f}")

            if len(non_normal_vars) > 0:
                print(f"     WARNING: {len(non_normal_vars)}/{len(var_list)} variables violate normality assumption")
                print(f"     Consider interpreting MANOVA results with caution")
            else:
                print(f"     All variables pass normality assumption")

            formula_lhs = " + ".join(var_list)
            formula = f"{formula_lhs} ~ C(Cluster)"

            try:
                manova = MANOVA.from_formula(formula, data=manova_df)
                test_results = manova.mv_test()
                res_table = test_results['C(Cluster)']['stat']

                wilks_lambda = res_table.loc["Wilks' lambda", "Value"]
                wilks_f = res_table.loc["Wilks' lambda", "F Value"]
                wilks_p = res_table.loc["Wilks' lambda", "Pr > F"]
                num_df = res_table.loc["Wilks' lambda", "Num DF"]
                den_df = res_table.loc["Wilks' lambda", "Den DF"]

                print(f"     Wilks' Lambda: {wilks_lambda:.4f}")
                print(f"     F({num_df:.0f}, {den_df:.0f}) = {wilks_f:.3f}, p = {wilks_p:.4f}")

                if wilks_p < 0.05:
                    print(f"     >> Significant multivariate separation")
                else:
                    print(f"     >> No significant multivariate separation")

            except Exception as e:
                print(f"     !! ERROR in MANOVA: {e}")

        print("\n" + "=" * 70)

        # Univariate tests
        print("\nUNIVARIATE ANALYSIS")
        print("-" * 70)

        for item in behav_item_list:
            behav_values = behav_data[item].values

            if len(unique_clusters) == 2:
                group0 = behav_values[cluster_labels == unique_clusters[0]]
                group1 = behav_values[cluster_labels == unique_clusters[1]]
                stat, pval = ttest_ind(group0, group1)
                test_name = 't-test'
            else:
                groups = [behav_values[cluster_labels == c] for c in unique_clusters]
                stat, pval = f_oneway(*groups)
                test_name = 'ANOVA'

            cluster_means = {}
            cluster_stds = {}
            for c in unique_clusters:
                cluster_data = behav_values[cluster_labels == c]
                cluster_means[f'cluster{c}_mean'] = cluster_data.mean()
                cluster_stds[f'cluster{c}_std'] = cluster_data.std()

            item_category = 'SRS' if 'SRS' in item else 'NIH'

            behav_results.append({
                'item': item,
                'category': item_category,
                'n_total': len(behav_values),
                'test': test_name,
                'statistic': stat,
                'p_value': pval,
                **cluster_means,
                **cluster_stds
            })

        behav_summary_df = pd.DataFrame(behav_results)

        # FDR correction by category
        for category in ['SRS', 'NIH']:
            cat_mask = behav_summary_df['category'] == category
            if cat_mask.sum() > 0:
                p_vals = behav_summary_df.loc[cat_mask, 'p_value'].values
                reject, q_vals = fdrcorrection(p_vals, alpha=CONFIG['fdr_alpha'])
                behav_summary_df.loc[cat_mask, 'fdr_reject'] = reject
                behav_summary_df.loc[cat_mask, 'q_value'] = q_vals

    return cluster_summary_df, cluster_feature_df, behav_summary_df


def analyze_comorbidity_prevalence(
    cluster_labels: np.ndarray,
    subject_ids: List[str],
    comorbid_df: pd.DataFrame,
    category_list: List[str] = None,
    fdr_alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze comorbidity prevalence across clusters."""
    if category_list is None:
        category_list = ['ADHD', 'Internalizing', 'Externalizing', 'Learning']

    n_clusters = len(np.unique(cluster_labels))

    # Align data
    common_subjects = [sid for sid in subject_ids if sid in comorbid_df.index]

    if len(common_subjects) != len(subject_ids):
        n_missing = len(subject_ids) - len(common_subjects)
        print(f"  Warning: {n_missing} subjects missing comorbidity data")

        subject_mask = np.array([sid in comorbid_df.index for sid in subject_ids])
        cluster_labels_filtered = cluster_labels[subject_mask]
        subject_ids_filtered = common_subjects
    else:
        cluster_labels_filtered = cluster_labels
        subject_ids_filtered = subject_ids

    comorbid_data = comorbid_df.loc[subject_ids_filtered]

    results = []
    prevalence_data = []

    for category in category_list:
        if category not in comorbid_data.columns:
            continue

        contingency = np.zeros((n_clusters, 2), dtype=int)

        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels_filtered == cluster_id)
            cluster_comorbid = comorbid_data[category].values[cluster_mask]

            n_positive = cluster_comorbid.sum()
            n_negative = len(cluster_comorbid) - n_positive

            contingency[cluster_id, 0] = n_negative
            contingency[cluster_id, 1] = n_positive

            prevalence_data.append({
                'category': category,
                'cluster': cluster_id,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'n_total': len(cluster_comorbid),
                'prevalence_%': 100 * n_positive / len(cluster_comorbid) if len(cluster_comorbid) > 0 else 0
            })

        # Statistical test
        if n_clusters == 2 and contingency.shape == (2, 2):
            oddsratio, p_value = fisher_exact(contingency, alternative='two-sided')
            test_statistic = oddsratio
            test_name = 'Fisher'
            dof = None
            # Debug: print contingency table
            print(f"    {category}: Contingency table:")
            print(f"      Cluster 0: Absent={contingency[0,0]}, Present={contingency[0,1]}")
            print(f"      Cluster 1: Absent={contingency[1,0]}, Present={contingency[1,1]}")
            print(f"      OR={(contingency[0,0]*contingency[1,1])/(contingency[0,1]*contingency[1,0]) if contingency[0,1]*contingency[1,0] > 0 else 'inf':.3f}, p={p_value:.4f}")
        else:
            chi2_stat, p_value, dof, _ = chi2_contingency(contingency)
            test_statistic = chi2_stat
            test_name = 'Chi-square'

        results.append({
            'category': category,
            'test_name': test_name,
            'test_statistic': test_statistic,
            'p_value': p_value,
            'dof': dof,
            'n_subjects': len(cluster_labels_filtered),
            'n_positive_total': comorbid_data[category].sum(),
            'prevalence_total_%': 100 * comorbid_data[category].sum() / len(comorbid_data)
        })

    results_df = pd.DataFrame(results)
    prevalence_df = pd.DataFrame(prevalence_data)

    # FDR correction
    if len(results_df) > 0:
        p_values = results_df['p_value'].values
        reject, q_values = fdrcorrection(p_values, alpha=fdr_alpha)
        results_df['fdr_reject'] = reject
        results_df['q_value'] = q_values

    return results_df, prevalence_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def print_separator(char: str = '=', length: int = 80):
    """Print separator line."""
    print(char * length)


def plot_clustering_metrics(results_dict: Dict, save_dir: Path):
    """Plot elbow curve and silhouette scores for K-means."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    k_values = results_dict['k_values']

    # Elbow plot
    axes[0].plot(k_values, results_dict['inertias'], 'o-', linewidth=2, markersize=8, color='black')
    if results_dict['optimal_k_elbow'] is not None:
        axes[0].axvline(results_dict['optimal_k_elbow'], color='red',
                       linestyle='--', label=f"Elbow: k={results_dict['optimal_k_elbow']}")
        axes[0].legend()
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
    axes[0].set_xticks(k_values)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Silhouette plot
    axes[1].plot(k_values, results_dict['silhouette_scores'], 'o-',
                linewidth=2, markersize=8, color='black')
    axes[1].axvline(results_dict['optimal_k_silhouette'], color='red',
                   linestyle='--', label=f"Best: k={results_dict['optimal_k_silhouette']}")
    axes[1].legend()
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_xticks(k_values)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    filename = 'clustering_metrics_kmeans.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_cluster_feature_heatmap(
    cluster_feature_df: pd.DataFrame,
    save_dir: Path,
    method: str = 'kmeans'
):
    """Plot heatmap of outlier proportions per cluster."""
    fig, ax = plt.subplots(figsize=(8, 4))

    feature_cols = [col for col in cluster_feature_df.columns if col != 'cluster']
    heatmap_data = cluster_feature_df[feature_cols].values

    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
               cbar_kws={'label': 'Outlier Prevalence (%)'},
               xticklabels=feature_cols,
               yticklabels=[f'Cluster {i}' for i in cluster_feature_df['cluster']],
               ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Network-Feature', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    method_label = 'K-means' if method == 'kmeans' else 'Hierarchical'
    ax.set_title(f'Outlier Pattern by Cluster ({method_label})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    filename = f'cluster_outlier_heatmap_{method}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_cluster_zscore_heatmap(
    zscore_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    save_dir: Path,
    method: str = 'kmeans'
):
    """Plot heatmap of MEAN z-scores per cluster (shows severity, not just outlier status)."""
    n_clusters = len(np.unique(cluster_labels))

    # Calculate mean z-scores per cluster
    cluster_means = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        mean_zscores = zscore_df.loc[cluster_mask, feature_names].mean()
        cluster_means.append({'cluster': cluster_id, **mean_zscores.to_dict()})

    cluster_mean_df = pd.DataFrame(cluster_means)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    feature_cols = [col for col in cluster_mean_df.columns if col != 'cluster']
    heatmap_data = cluster_mean_df[feature_cols].values

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Z-Score'},
                xticklabels=feature_cols,
                yticklabels=[f'Cluster {i}' for i in cluster_mean_df['cluster']],
                ax=ax, linewidths=0.5, linecolor='gray',
                vmin=-2, vmax=2)  # Symmetric colorbar

    ax.set_xlabel('Network-Feature', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    method_label = 'K-means' if method == 'kmeans' else 'Hierarchical'
    ax.set_title(f'Mean Z-Score by Cluster ({method_label})', fontsize=12, fontweight='bold')

    plt.tight_layout()
    filename = f'cluster_zscore_heatmap_{method}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_cluster_demographics(
    cluster_labels: np.ndarray,
    ages: np.ndarray,
    sexes: np.ndarray,
    save_dir: Path,
    method: str = 'kmeans'
):
    """Plot demographic comparisons across clusters."""
    n_clusters = len(np.unique(cluster_labels))

    # fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    # fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Age boxplot
    ax = axes[0]
    age_data = [ages[cluster_labels == i] for i in range(n_clusters)]
    cluster_names = [f'Cluster {i}' for i in range(n_clusters)]

    bp = ax.boxplot(age_data, labels=cluster_names, patch_artist=True,
                    widths=0.6, showmeans=True,
                    # meanprops=dict(marker='D', markerfacecolor='red',
                    #               markeredgecolor='red', markersize=6),
                    meanprops=dict(marker='None'),
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(facecolor='lightgray', edgecolor='black', linewidth=1.5))

    for i, age_vals in enumerate(age_data):
        x_jitter = np.random.normal(i+1, 0.04, size=len(age_vals))
        ax.scatter(x_jitter, age_vals, alpha=0.3, s=30, color='black', zorder=3)

    ax.set_ylabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution by Cluster', fontsize=13, fontweight='bold')

    if n_clusters == 2:
        stat, pval = ttest_ind(age_data[0], age_data[1])
        test_name = 't-test'
    else:
        stat, pval = f_oneway(*age_data)
        test_name = 'ANOVA'

    sig_marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ' (n.s.)'
    # ax.text(0.5, 0.98, f'{test_name}: p = {pval:.4f}{sig_marker}', transform=ax.transAxes,
    #        ha='center', va='top', fontsize=10,
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Sex distribution
    ax = axes[1]
    sex_counts = []
    for i in range(n_clusters):
        cluster_sexes = sexes[cluster_labels == i]
        n_male = (cluster_sexes == 0).sum()
        n_female = (cluster_sexes == 1).sum()
        sex_counts.append([n_male, n_female])

    sex_counts = np.array(sex_counts)
    x = np.arange(n_clusters)
    width = 0.35

    bars1 = ax.bar(x - width/2, sex_counts[:, 0], width, label='Male',
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, sex_counts[:, 1], width, label='Female',
                   color='lightcoral', edgecolor='black', linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Sex Distribution by Cluster', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
    # ax.legend(loc='upper right')

    contingency = np.array([sex_counts[:, 0], sex_counts[:, 1]])
    if n_clusters == 2:
        _, pval_sex = fisher_exact(contingency.T)
        test_name_sex = "Fisher's exact"
    else:
        _, pval_sex, _, _ = chi2_contingency(contingency)
        test_name_sex = 'Chi-square'

    sig_marker_sex = '***' if pval_sex < 0.001 else '**' if pval_sex < 0.01 else '*' if pval_sex < 0.05 else ' (n.s.)'
    # ax.text(0.5, 0.98, f'{test_name_sex}: p = {pval_sex:.4f}{sig_marker_sex}', transform=ax.transAxes,
    #        ha='center', va='top', fontsize=10,
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(sex_counts.flatten()) * 1.15)
    ax.tick_params(axis='x', which='both', bottom=True, top=False)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    filename = f'cluster_demographics_{method}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_single_behavioral_measure(
    behav_data: pd.DataFrame,
    cluster_labels: np.ndarray,
    item_name: str,
    save_dir: Path,
    method: str = 'kmeans',
    p_value: Optional[float] = None,
    test_name: Optional[str] = None,
    td_behav_data: Optional[pd.DataFrame] = None
):
    """Plot boxplot for single behavioral measure across clusters.

    If td_behav_data is provided, a TD reference box is drawn as the leftmost
    box (grey, like the cluster boxes) and the omnibus significance test shown
    in the title is recomputed to INCLUDE TD as an additional group (overriding
    the cluster-only p_value/test_name passed in). When the omnibus ANOVA is
    significant (>=3 groups, p < 0.05), Tukey HSD post-hoc pairwise comparisons
    are drawn as significance bars over the boxes.
    """
    if item_name not in behav_data.columns:
        print(f"  Warning: '{item_name}' not found")
        return

    item_values = behav_data[item_name].values
    n_clusters = len(np.unique(cluster_labels))
    data_by_cluster = [item_values[cluster_labels == i] for i in range(n_clusters)]

    # Optional TD reference box + TD-inclusive statistics
    td_vals = None
    if td_behav_data is not None and item_name in td_behav_data.columns:
        td_series = pd.to_numeric(td_behav_data[item_name], errors='coerce').dropna()
        if len(td_series) > 0:
            td_vals = td_series.values

    if td_vals is not None:
        plot_data = [td_vals] + data_by_cluster
        box_labels = ['TD'] + [f'Cluster {i}' for i in range(n_clusters)]
        # Recompute the test to INCLUDE TD as a group (overrides cluster-only stats)
        if len(plot_data) == 2:
            _, p_value = ttest_ind(*plot_data)
            test_name = 't-test (TD vs cluster)'
        else:
            _, p_value = f_oneway(*plot_data)
            test_name = 'ANOVA (TD+clusters)'
    else:
        plot_data = data_by_cluster
        box_labels = [f'Cluster {i}' for i in range(n_clusters)]

    n_boxes = len(plot_data)
    # fig, ax = plt.subplots(figsize=(max(3, 1.1 * n_boxes + 1), 6))
    # fig, ax = plt.subplots(figsize=(5, 6))
    fig, ax = plt.subplots(figsize=(3, 6))

    bp = ax.boxplot(plot_data,
                    labels=box_labels,
                    patch_artist=True,
                    widths=0.6,
                    # showmeans=True,
                    # meanprops=dict(marker='D', markerfacecolor='red',
                    #               markeredgecolor='red', markersize=8),
                    meanprops=dict(marker='None'),
                    medianprops=dict(color='black', linewidth=2.5),
                    boxprops=dict(facecolor='lightgray', edgecolor='black', linewidth=1.5))

    for i, vals in enumerate(plot_data):
        x_jitter = np.random.normal(i+1, 0.04, size=len(vals))
        ax.scatter(x_jitter, vals, alpha=0.4, s=40, color='black', zorder=3)

    # Post-hoc pairwise comparisons (Tukey HSD) when the omnibus ANOVA is significant
    if n_boxes >= 3 and p_value is not None and p_value < 0.05:
        endog = np.concatenate(plot_data)
        group_codes = np.concatenate([np.full(len(g), i) for i, g in enumerate(plot_data)])
        tukey = pairwise_tukeyhsd(endog, group_codes, alpha=0.05)
        pairs = list(itertools.combinations(range(n_boxes), 2))
        sig_pairs = [(pairs[k], tukey.pvalues[k])
                     for k in range(len(pairs)) if tukey.reject[k]]
        # Draw shorter spans first so nested bars stack cleanly
        sig_pairs.sort(key=lambda pp: (pp[0][1] - pp[0][0], pp[0][0]))

        if sig_pairs:
            data_max = max(np.max(g) for g in plot_data)
            data_min = min(np.min(g) for g in plot_data)
            y_range = (data_max - data_min) or 1.0
            tick_h = y_range * 0.02
            step = y_range * 0.07
            for level, ((i, j), p) in enumerate(sig_pairs):
                y = data_max + step * (level + 1)
                ax.plot([i + 1, i + 1, j + 1, j + 1],
                        [y, y + tick_h, y + tick_h, y],
                        lw=1.3, c='black', clip_on=False)
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*'
                ax.text((i + 1 + j + 1) / 2, y + tick_h, stars,
                        ha='center', va='bottom', fontsize=11)
            ax.set_ylim(top=data_max + step * (len(sig_pairs) + 1))

    # Format title
    if item_name.startswith('SRS_'):
        display_name = item_name.replace('SRS_', 'SRS: ').replace('_T', '')
    elif item_name.startswith('NIH'):
        display_name = item_name.replace('NIH7_', 'NIH: ').replace('_P', '')
    else:
        display_name = item_name

    title_text = f'{display_name}'
    if p_value is not None and test_name is not None:
        sig_stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        title_text += f'\n{test_name}: p = {p_value:.4f}{sig_stars}'

    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')

    ax.tick_params(axis='both', which='major', length=6, width=1.5,
                   direction='out', bottom=True, left=True, color='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    clean_name = item_name.replace('_', '').lower()
    filename = f'cluster_behavioral_{clean_name}_{method}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_comorbidity_prevalence(
    prevalence_df: pd.DataFrame,
    results_df: pd.DataFrame,
    save_dir: Path,
    method: str = 'kmeans'
):
    """Plot comorbidity prevalence by cluster."""
    categories = prevalence_df['category'].unique()
    n_clusters = prevalence_df['cluster'].nunique()

    fig, ax = plt.subplots(figsize=(10, 8))

    cluster_alphas = [0.8, 0.4, 0.2][:n_clusters]
    bar_width = 0.8 / n_clusters
    x_positions = np.arange(len(categories))

    for cluster_id in range(n_clusters):
        prevalences = []
        for category in categories:
            cat_data = prevalence_df[
                (prevalence_df['category'] == category) &
                (prevalence_df['cluster'] == cluster_id)
            ]
            if len(cat_data) > 0:
                prevalences.append(cat_data['prevalence_%'].values[0])
            else:
                prevalences.append(0)

        x_offset = (cluster_id - n_clusters/2 + 0.5) * bar_width
        x = x_positions + x_offset

        ax.bar(x, prevalences, bar_width,
              color='black', alpha=cluster_alphas[cluster_id],
              edgecolor='black', linewidth=1.5,
              label=f'Cluster {cluster_id}')

    ax.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Comorbidity Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Comorbidity Prevalence by Cluster ({method.capitalize()})',
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)

    max_prevalence = prevalence_df['prevalence_%'].max()
    ax.set_ylim(0, max_prevalence * 1.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    filename = f'comorbidity_prevalence_{method}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_dendrogram(linkage_matrix: np.ndarray, save_dir: Path, max_k: int = 5):
    """Plot dendrogram for hierarchical clustering."""
    fig, ax = plt.subplots(figsize=(12, 6))

    dend = dendrogram(linkage_matrix, ax=ax, color_threshold=0,
                     above_threshold_color='black', no_labels=True)

    ax.set_xlabel('Subject Index', fontsize=12)
    ax.set_ylabel('Distance (Euclidean)', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')

    colors = ['red', 'blue', 'green', 'orange']
    heights = linkage_matrix[:, 2]

    for i, (k, color) in enumerate(zip(range(2, max_k + 1), colors)):
        cut_height = heights[-(k-1)] if k > 2 else heights[-1]
        if cut_height < ax.get_ylim()[1]:
            ax.axhline(y=cut_height, color=color, linestyle='--', linewidth=1.5,
                      label=f'k={k}', alpha=0.7)

    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = 'dendrogram_hierarchical.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_hierarchical_metrics(results_dict: Dict, save_dir: Path):
    """Plot silhouette scores for hierarchical clustering."""
    fig, ax = plt.subplots(figsize=(5, 5))

    k_values = results_dict['k_values']

    ax.plot(k_values, results_dict['silhouette_scores'], 'o-',
           linewidth=2, markersize=8, color='black')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(f"Hierarchical Clustering ({results_dict['method']} linkage)",
                fontsize=12, fontweight='bold')
    ax.set_xticks(k_values)
    ax.tick_params(axis='x', which='major', length=6, width=1.5, direction='out')
    ax.tick_params(axis='y', which='major', length=6, width=1.5, direction='out')
    ax.tick_params(bottom=True, left=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    y_min, y_max = min(results_dict['silhouette_scores']), max(results_dict['silhouette_scores'])
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    plt.tight_layout()
    filename = 'hierarchical_metrics.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_age_stratified_bars(
    prevalence_df: pd.DataFrame,
    network_name: str,
    feature_type: str,
    save_dir: Path
):
    """Plot bar chart of outlier prevalence by age group."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(prevalence_df))
    bars = ax.bar(x, prevalence_df['prevalence_%'],
                  color=sns.color_palette()[1], alpha=0.7, edgecolor='black')

    for i, (bar, n_out, n_tot) in enumerate(zip(bars,
                                                  prevalence_df['n_outliers'],
                                                  prevalence_df['n_total'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{n_out}/{n_tot}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(prevalence_df['age_group'])
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Outlier Prevalence (%)', fontsize=12)

    network_label = NETWORK_LABELS[network_name]
    feature_label = FEATURE_LABELS[feature_type]
    ax.set_title(f'{network_label} - {feature_label}\nOutlier Prevalence by Age',
                 fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(prevalence_df['prevalence_%']) * 1.2)

    plt.tight_layout()
    filename = f'outlier_age_prevalence_{network_name}_{feature_type}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_cluster_radar(
    zscore_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    save_dir: Path,
    method: str = 'kmeans',
    title_suffix: str = ''
):
    """Plot radar/spider chart showing cluster profiles across features.

    Args:
        zscore_df: DataFrame with z-scores (can be original or profile-normalized)
        cluster_labels: Cluster assignment for each subject
        feature_names: List of feature names to plot
        save_dir: Directory to save figure
        method: Clustering method name for filename
        title_suffix: Additional text for title (e.g., '_profile')
    """
    n_clusters = len(np.unique(cluster_labels))
    n_features = len(feature_names)

    # Calculate mean z-scores per cluster
    cluster_means = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        mean_zscores = zscore_df.loc[cluster_mask, feature_names].mean()
        cluster_means.append(mean_zscores.values)
    cluster_means = np.array(cluster_means)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Color palette for clusters
    colors = CLUSTER_COLORS

    for cluster_id in range(n_clusters):
        values = cluster_means[cluster_id].tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, markersize=10,
                label=f'Cluster {cluster_id} (n={(cluster_labels == cluster_id).sum()})',
                color=colors[cluster_id % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[cluster_id % len(colors)])

    # Set feature labels
    ax.set_xticks(angles[:-1])
    # Create readable labels
    readable_labels = []
    for fname in feature_names:
        parts = fname.split('_')
        feat_type = 'Sem' if parts[0] == 'sem' else 'Dim'
        network = parts[1].upper()
        readable_labels.append(f'{feat_type}\n{network}')
    ax.set_xticklabels(readable_labels, fontsize=10)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels([])  # Hide y-axis labels

    # Add reference circle at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_title(f'Cluster Profiles ({method.capitalize()}{title_suffix})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    filename = f'cluster_radar_{method}{title_suffix}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_cluster_parallel_coords(
    zscore_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
    save_dir: Path,
    method: str = 'kmeans',
    title_suffix: str = ''
):
    """Plot parallel coordinates showing cluster profiles.

    Args:
        zscore_df: DataFrame with z-scores (can be original or profile-normalized)
        cluster_labels: Cluster assignment for each subject
        feature_names: List of feature names to plot
        save_dir: Directory to save figure
        method: Clustering method name for filename
        title_suffix: Additional text for title (e.g., '_profile')
    """
    n_clusters = len(np.unique(cluster_labels))

    # Calculate mean and std per cluster
    cluster_data = []
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        mean_vals = zscore_df.loc[cluster_mask, feature_names].mean()
        std_vals = zscore_df.loc[cluster_mask, feature_names].std()
        cluster_data.append({
            'cluster': cluster_id,
            'n': cluster_mask.sum(),
            'means': mean_vals.values,
            'stds': std_vals.values
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(feature_names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for cluster_id, data in enumerate(cluster_data):
        color = colors[cluster_id % len(colors)]

        # Plot mean line with error band (std)
        ax.plot(x_positions, data['means'], 'o-', linewidth=2.5, markersize=10,
                label=f"Cluster {cluster_id} (n={data['n']})", color=color)
        ax.fill_between(x_positions,
                        data['means'] - data['stds'],
                        data['means'] + data['stds'],
                        alpha=0.2, color=color)

    # Reference line at zero
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Create readable labels
    readable_labels = []
    for fname in feature_names:
        parts = fname.split('_')
        feat_type = 'Semantic' if parts[0] == 'sem' else 'Dimensional'
        network = NETWORK_LABELS[parts[1]]
        readable_labels.append(f'{feat_type}\n{network}')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(readable_labels, fontsize=9, ha='center')
    ax.set_xlabel('Network Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Z-Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster Profiles - Parallel Coordinates ({method.capitalize()}{title_suffix})',
                 fontsize=13, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    filename = f'cluster_parallel_coords_{method}{title_suffix}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def determine_trend_direction(
    zscore_td: pd.DataFrame,
    zscore_asd: pd.DataFrame,
    zscore_col: str,
    significance_threshold: float = 0.05
) -> Dict:
    """
    Determine whether ASD group tends to be above or below TD normative trajectory.

    Uses independent samples t-test to compare mean z-scores between groups.

    Parameters:
    -----------
    zscore_td : pd.DataFrame
        TD z-score data with z-scores for each network-feature combination
    zscore_asd : pd.DataFrame
        ASD z-score data with z-scores for each network-feature combination
    zscore_col : str
        Column name for the z-score variable (e.g., 'sem_sal')
    significance_threshold : float
        P-value threshold for statistical significance (default: 0.05)

    Returns:
    --------
    dict with keys:
        'direction': str - 'below', 'above', or 'no_difference'
        'mean_z_td': float - Mean z-score for TD group
        'mean_z_asd': float - Mean z-score for ASD group
        'diff': float - Mean difference (ASD - TD)
        'p_value': float - P-value from t-test
        't_statistic': float - T-statistic
        'n_td': int - Number of TD subjects
        'n_asd': int - Number of ASD subjects
    """
    try:
        # Extract z-scores (TD z-scores should be ~0 by definition)
        z_scores_td = zscore_td[zscore_col].dropna().values
        z_scores_asd = zscore_asd[zscore_col].dropna().values

        # Check for empty groups
        if len(z_scores_td) == 0 or len(z_scores_asd) == 0:
            return {
                'direction': 'insufficient_data',
                'mean_z_td': np.nan,
                'mean_z_asd': np.nan,
                'diff': np.nan,
                'p_value': np.nan,
                't_statistic': np.nan,
                'n_td': len(z_scores_td),
                'n_asd': len(z_scores_asd)
            }

        # Calculate means
        mean_z_td = np.mean(z_scores_td)
        mean_z_asd = np.mean(z_scores_asd)
        diff = mean_z_asd - mean_z_td

        # Perform Welch's t-test
        t_stat, p_val = ttest_ind(z_scores_asd, z_scores_td, equal_var=False)

        # Determine direction
        if p_val < significance_threshold:
            if diff < 0:
                direction = 'below'
            else:
                direction = 'above'
        else:
            direction = 'no_difference'

        return {
            'direction': direction,
            'mean_z_td': mean_z_td,
            'mean_z_asd': mean_z_asd,
            'diff': diff,
            'p_value': p_val,
            't_statistic': t_stat,
            'n_td': len(z_scores_td),
            'n_asd': len(z_scores_asd)
        }

    except (KeyError, IndexError) as e:
        print(f"  Error in determine_trend_direction for {zscore_col}: {e}")
        return {
            'direction': 'error',
            'mean_z_td': np.nan,
            'mean_z_asd': np.nan,
            'diff': np.nan,
            'p_value': np.nan,
            't_statistic': np.nan,
            'n_td': 0,
            'n_asd': 0
        }


def calculate_directional_adherence(
    raw_data_asd: pd.DataFrame,
    traj_td: pd.DataFrame,
    network: str,
    trend_direction: str,
    age_bounds_buffer: float = 0.5
) -> Dict:
    """
    Calculate percentage of ASD subjects following the identified trend direction.

    For each ASD subject, compares their observed value to the age-matched TD
    trajectory prediction and determines if they follow the trend direction.

    Parameters:
    -----------
    raw_data_asd : pd.DataFrame
        ASD subject data with columns: network (e.g., 'sal'), 'age', 'sex'
    traj_td : pd.DataFrame
        TD trajectory predictions with 'age', '{network}_mu' columns
    network : str
        Network column name ('sal', 'dmn', 'cen')
    trend_direction : str
        Direction from determine_trend_direction(): 'below', 'above', or 'no_difference'
    age_bounds_buffer : float
        Age buffer (years) for trajectory bounds checking (default: 0.5)

    Returns:
    --------
    dict with keys:
        'n_total': int - Total ASD subjects analyzed
        'n_following_trend': int - Number following trend direction
        'n_opposing_trend': int - Number opposing trend direction
        'pct_adherence': float - Percentage following trend (0-100)
        'binomial_p': float - P-value from binomial test (H0: p=0.5)
    """
    # Handle no_difference case
    if trend_direction in ['no_difference', 'insufficient_data', 'error']:
        return {
            'n_total': 0,
            'n_following_trend': 0,
            'n_opposing_trend': 0,
            'pct_adherence': np.nan,
            'binomial_p': np.nan
        }

    try:
        # Filter ASD data by sex (male only, sex=0)
        asd_filtered = raw_data_asd[raw_data_asd['sex'] == 0].copy()

        if len(asd_filtered) == 0:
            return {
                'n_total': 0,
                'n_following_trend': 0,
                'n_opposing_trend': 0,
                'pct_adherence': np.nan,
                'binomial_p': np.nan
            }

        # Extract TD trajectory
        td_ages = traj_td['age'].values
        mu_col = f'{network}_mu'
        td_mu = traj_td[mu_col].values

        # Filter subjects within age range
        age_min, age_max = td_ages.min(), td_ages.max()
        in_bounds = ((asd_filtered['age'] >= age_min - age_bounds_buffer) &
                     (asd_filtered['age'] <= age_max + age_bounds_buffer))

        # Also filter out NaN values
        valid_mask = in_bounds & (~pd.isna(asd_filtered['age'])) & (~pd.isna(asd_filtered[network]))
        asd_clean = asd_filtered[valid_mask].copy()

        if len(asd_clean) == 0:
            return {
                'n_total': 0,
                'n_following_trend': 0,
                'n_opposing_trend': 0,
                'pct_adherence': np.nan,
                'binomial_p': np.nan
            }

        # Get ages and observed values
        asd_ages = asd_clean['age'].values
        asd_observed = asd_clean[network].values

        # Interpolate TD trajectory at ASD ages
        td_predicted = np.interp(asd_ages, td_ages, td_mu)

        # Compare observed vs predicted
        below_trajectory = asd_observed < td_predicted
        above_trajectory = asd_observed > td_predicted

        # Count following trend
        if trend_direction == 'below':
            following_trend = below_trajectory
        elif trend_direction == 'above':
            following_trend = above_trajectory
        else:
            following_trend = np.zeros(len(asd_clean), dtype=bool)

        n_total = len(asd_clean)
        n_following = int(following_trend.sum())
        n_opposing = n_total - n_following
        pct_adherence = 100 * n_following / n_total

        # Binomial test (H0: p=0.5, two-sided)
        binomial_result = binomtest(n_following, n_total, p=0.5, alternative='two-sided')
        binomial_p = binomial_result.pvalue

        return {
            'n_total': n_total,
            'n_following_trend': n_following,
            'n_opposing_trend': n_opposing,
            'pct_adherence': pct_adherence,
            'binomial_p': binomial_p
        }

    except (KeyError, IndexError) as e:
        print(f"  Error in calculate_directional_adherence for {network}: {e}")
        return {
            'n_total': 0,
            'n_following_trend': 0,
            'n_opposing_trend': 0,
            'pct_adherence': np.nan,
            'binomial_p': np.nan
        }


def test_cluster_trajectories(
    raw_data_asd: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_label_index: List[str],
    network: str,
    feature: str
) -> Dict[str, Any]:
    """Test whether ASD cluster subgroups have distinct age trajectories.

    Fits OLS: score ~ age + cluster + age:cluster (male only).
    The interaction term tests slope differences; the cluster main effect
    tests intercept differences.

    Parameters:
    -----------
    raw_data_asd : pd.DataFrame
        Raw ASD data with columns: network score, 'age', 'sex'
    cluster_labels : np.ndarray
        Cluster assignment per subject (same order as cluster_label_index)
    cluster_label_index : list of str
        Subject IDs corresponding to cluster_labels
    network : str
        Network column name ('sal', 'dmn', 'cen')
    feature : str
        Feature type ('sem' or 'dim'), used for raw_data key construction

    Returns:
    --------
    dict with keys:
        'network', 'feature': identifiers
        'n_per_cluster': dict of cluster_id -> n
        'interaction_F', 'interaction_p': age:cluster interaction test
        'cluster_F', 'cluster_p': cluster main effect test
        'age_F', 'age_p': age main effect test
        'r2_full': R-squared of full model
        'slope_per_cluster': dict of cluster_id -> (slope, intercept)
    """
    cluster_map = dict(zip(cluster_label_index, cluster_labels))

    # Filter to male ASD subjects with valid data
    asd_male = raw_data_asd[raw_data_asd['sex'] == 0].copy()
    if len(asd_male) == 0 or network not in asd_male.columns:
        return {'network': network, 'feature': feature, 'error': 'no_data'}

    # Assign cluster labels
    asd_male = asd_male.loc[asd_male.index.isin(cluster_map)]
    asd_male['cluster'] = asd_male.index.map(cluster_map)
    asd_male = asd_male.dropna(subset=[network, 'age'])

    if len(asd_male) < 10:
        return {'network': network, 'feature': feature, 'error': 'insufficient_data'}

    # Prepare for statsmodels
    asd_male = asd_male.rename(columns={network: 'score'})
    asd_male['cluster'] = asd_male['cluster'].astype(str)

    # Fit interaction model: score ~ age * cluster
    try:
        model = smf_ols('score ~ age * C(cluster)', data=asd_male).fit()

        # Type II ANOVA for proper main effects with interaction
        anova_results = anova_lm(model, typ=2)

        result = {
            'network': network,
            'feature': feature,
            'n_total': len(asd_male),
            'n_per_cluster': asd_male.groupby('cluster').size().to_dict(),
            'r2_full': model.rsquared,
            'r2_adj': model.rsquared_adj,
        }

        # Extract ANOVA results for each term
        for term, label in [('age', 'age'), ('C(cluster)', 'cluster'), ('age:C(cluster)', 'interaction')]:
            if term in anova_results.index:
                result[f'{label}_F'] = anova_results.loc[term, 'F']
                result[f'{label}_p'] = anova_results.loc[term, 'PR(>F)']
            else:
                result[f'{label}_F'] = np.nan
                result[f'{label}_p'] = np.nan

        # Per-cluster slopes via simple linear regression
        slopes = {}
        for cid, grp in asd_male.groupby('cluster'):
            if len(grp) >= 2:
                slope, intercept, _, _, _ = stats.linregress(grp['age'], grp['score'])
                slopes[cid] = {'slope': slope, 'intercept': intercept, 'n': len(grp)}
        result['slope_per_cluster'] = slopes

        return result

    except Exception as e:
        return {'network': network, 'feature': feature, 'error': str(e)}


def run_trajectory_comparison(
    raw_data: Dict[str, pd.DataFrame],
    cluster_labels: np.ndarray,
    cluster_label_index: List[str],
    fdr_alpha: float = 0.05
) -> pd.DataFrame:
    """Run trajectory comparison across all network-feature combinations.

    Tests whether ASD cluster subgroups have distinct developmental
    trajectories using OLS interaction models, with FDR correction.

    Returns:
        DataFrame with one row per network-feature combination.
    """
    results = []

    for feature in FEATURE_TYPES:
        raw_asd_key = f'{feature}_asd'
        if raw_data.get(raw_asd_key) is None:
            continue

        for network in NETWORK_NAMES:
            result = test_cluster_trajectories(
                raw_data_asd=raw_data[raw_asd_key],
                cluster_labels=cluster_labels,
                cluster_label_index=cluster_label_index,
                network=network,
                feature=feature
            )
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # FDR correction on interaction p-values
    if 'interaction_p' in df.columns:
        valid = df['interaction_p'].notna()
        if valid.sum() > 0:
            reject, q_vals = fdrcorrection(df.loc[valid, 'interaction_p'].values, alpha=fdr_alpha)
            df.loc[valid, 'interaction_q'] = q_vals
            df.loc[valid, 'interaction_sig'] = reject

    # FDR correction on cluster main effect p-values
    if 'cluster_p' in df.columns:
        valid = df['cluster_p'].notna()
        if valid.sum() > 0:
            reject, q_vals = fdrcorrection(df.loc[valid, 'cluster_p'].values, alpha=fdr_alpha)
            df.loc[valid, 'cluster_q'] = q_vals
            df.loc[valid, 'cluster_sig'] = reject

    return df


def plot_gam_trajectory(
    network: str,
    feature: str,
    trajectories: Dict[str, Dict[str, pd.DataFrame]],
    raw_data: Dict[str, pd.DataFrame],
    zscore_asd: pd.DataFrame,
    save_dir: Path,
    zscore_td: pd.DataFrame = None,
    outlier_threshold: float = 1.96,
    plot_groups: str = 'td_only',
    interval_type: str = 'prediction',
    scatter: bool = True,
    show_adherence: bool = True,
    cluster_labels: Optional[np.ndarray] = None,
    cluster_label_index: Optional[List[str]] = None,
    cluster_method: Optional[str] = None,
    cluster_slopes: Optional[Dict] = None,
    cluster_density: bool = False
):
    """Plot GAM trajectory for a specific network and feature (male only).

    Parameters:
    -----------
    network : str
        Network name ('sal', 'dmn', 'cen')
    feature : str
        Feature type ('sem' or 'dim')
    trajectories : Dict
        Trajectory data from load_gam_trajectories()
    raw_data : Dict
        Raw network scores from load_raw_network_data()
    zscore_asd : pd.DataFrame
        ASD z-score data for outlier identification
    save_dir : Path
        Directory to save plots
    zscore_td : pd.DataFrame, optional
        TD z-score data for trend direction analysis
    outlier_threshold : float
        Z-score threshold for outlier marking (default: 1.96)
    plot_groups : str
        Which trajectory lines to plot: 'td_only', 'asd_only', or 'both'
    interval_type : str
        Type of interval: 'confidence', 'prediction', 'both', or 'none'
        ('none' draws the TD mean line without any shaded normative band)
    scatter : bool
        Whether to show scatter points
    show_adherence : bool
        Whether to calculate and print trend direction & adherence summary
    cluster_labels : np.ndarray, optional
        Cluster assignment for each ASD subject (same order as cluster_label_index).
        When provided, ASD scatter points are color-coded by cluster.
    cluster_label_index : list of str, optional
        Subject IDs corresponding to cluster_labels (e.g., list(zscore_asd.index)).
        Required when cluster_labels is provided.
    cluster_method : str, optional
        Clustering method name for filename suffix (e.g., 'kmeans_profile').
    cluster_slopes : dict, optional
        Per-cluster slope info from test_cluster_trajectories()['slope_per_cluster'].
        When provided, draws per-cluster regression lines on the plot.
    cluster_density : bool
        When True, overlay a 2D Gaussian KDE contour per cluster in age x score
        space, colored with the same CLUSTER_COLORS used by the radar plot.
    """
    # Check if trajectory data is available
    if trajectories[feature]['td'] is None:
        print(f"  Warning: No trajectory data for {feature} - skipping {network}")
        return

    traj_td = trajectories[feature]['td']
    traj_asd = trajectories[feature]['asd']

    # Calculate and print trend direction & adherence summary
    if show_adherence and zscore_td is not None:
        zscore_col = f'{feature}_{network}'

        # Determine trend direction using z-scores
        trend_result = determine_trend_direction(
            zscore_td, zscore_asd, zscore_col
        )

        # Only calculate adherence if there's a clear directional trend
        if trend_result['direction'] not in ['no_difference', 'insufficient_data', 'error']:
            # Get raw ASD data for adherence calculation
            raw_asd_key = f'{feature}_asd'
            if raw_data.get(raw_asd_key) is not None:
                adherence_result = calculate_directional_adherence(
                    raw_data_asd=raw_data[raw_asd_key],
                    traj_td=traj_td,
                    network=network,
                    trend_direction=trend_result['direction']
                )

                # Print summary to console
                print(f"  Trend direction for {network}_{feature}: {trend_result['direction']} "
                      f"(mean_z_asd={trend_result['mean_z_asd']:.2f}, p={trend_result['p_value']:.3f})")
                print(f"  Adherence: {adherence_result['pct_adherence']:.1f}% of {adherence_result['n_total']} "
                      f"ASD males follow trend (p={adherence_result['binomial_p']:.3f})")
        else:
            print(f"  Trend direction for {network}_{feature}: {trend_result['direction']} "
                  f"(mean_z_asd={trend_result['mean_z_asd']:.2f}, p={trend_result['p_value']:.3f})")

    # Column names for this network
    mu_col = f'{network}_mu'
    ci_lower_col = f'{network}_ci_lower'
    ci_upper_col = f'{network}_ci_upper'
    pi_lower_col = f'{network}_pi_lower'
    pi_upper_col = f'{network}_pi_upper'

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    # fig, ax = plt.subplots(figsize=(8, 6))

    # Seaborn default colors
    colors = sns.color_palette()
    color_td = colors[0]   # Blue for TD
    color_asd = colors[1]  # Orange for ASD

    # Plot TD trajectory
    if plot_groups in ['td_only', 'both']:
        # Plot prediction interval (wider)
        if interval_type in ['prediction', 'both']:
            ax.fill_between(traj_td['age'], traj_td[pi_lower_col], traj_td[pi_upper_col],
                           color=color_td, alpha=0.1,
                           label=f'TD PI (±{outlier_threshold}σ)' if interval_type == 'both' else None)

        # Plot confidence interval (narrower)
        if interval_type in ['confidence', 'both']:
            ax.fill_between(traj_td['age'], traj_td[ci_lower_col], traj_td[ci_upper_col],
                           color=color_td, alpha=0.25 if interval_type == 'both' else 0.2,
                           label='TD 95% CI' if interval_type == 'both' else None)

        # Plot mean line
        ax.plot(traj_td['age'], traj_td[mu_col], color=color_td, linewidth=7, label='TD Male')

    # Plot ASD trajectory
    if plot_groups in ['asd_only', 'both'] and traj_asd is not None:
        if interval_type in ['prediction', 'both']:
            ax.fill_between(traj_asd['age'], traj_asd[pi_lower_col], traj_asd[pi_upper_col],
                           color=color_asd, alpha=0.1)

        if interval_type in ['confidence', 'both']:
            ax.fill_between(traj_asd['age'], traj_asd[ci_lower_col], traj_asd[ci_upper_col],
                           color=color_asd, alpha=0.25 if interval_type == 'both' else 0.2)

        ax.plot(traj_asd['age'], traj_asd[mu_col], color=color_asd, linewidth=3, label='ASD Male')

    # Per-cluster ASD male points, reused by the scatter and the density contours
    cluster_points = {}
    raw_asd_key = f'{feature}_asd'
    if (cluster_labels is not None and cluster_label_index is not None
            and raw_data.get(raw_asd_key) is not None):
        asd_male_all = raw_data[raw_asd_key]
        asd_male_all = asd_male_all[asd_male_all['sex'] == 0]
        if len(asd_male_all) > 0 and network in asd_male_all.columns:
            cluster_map = dict(zip(cluster_label_index, cluster_labels))
            matched_ids = asd_male_all.index.intersection(zscore_asd.index)
            matched = asd_male_all.loc[matched_ids]
            matched_clusters = np.array([cluster_map.get(sid, -1) for sid in matched_ids])
            for cid in sorted(set(matched_clusters) - {-1}):
                cmask = matched_clusters == cid
                cluster_points[int(cid)] = (matched['age'].values[cmask],
                                            matched[network].values[cmask])
            unmatched_mask = matched_clusters == -1
            cluster_points[-1] = (matched['age'].values[unmatched_mask],
                                  matched[network].values[unmatched_mask])

    # Plot scatter points (always show both groups when scatter=True)
    if scatter:
        # Get raw data keys
        raw_td_key = f'{feature}_td'

        # Plot TD scatter (male only)
        if raw_data.get(raw_td_key) is not None:
            td_data = raw_data[raw_td_key]
            td_male = td_data[td_data['sex'] == 0]
            if len(td_male) > 0 and network in td_male.columns:
                ax.scatter(td_male['age'], td_male[network],
                          facecolors=to_rgba(color_td, 0.45), s=100,
                          edgecolors='black', linewidths=0.8,
                          label='_nolegend_', zorder=3)

        # Plot ASD scatter (male only) with optional cluster coloring
        if raw_data.get(raw_asd_key) is not None:
            asd_data = raw_data[raw_asd_key]
            asd_male = asd_data[asd_data['sex'] == 0]

            if len(asd_male) > 0 and network in asd_male.columns:
                if cluster_points:
                    # Uniform orange fill + cluster-colored outlines (radar colors)
                    for cid, (cx, cy) in sorted(cluster_points.items()):
                        if cid == -1:
                            continue
                        edge_color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                        ax.scatter(cx, cy,
                                  facecolors=to_rgba(color_asd, 0.45), s=100,
                                  edgecolors=edge_color, linewidths=1.8,
                                  label=f'ASD Cluster {cid}', zorder=5)
                    # Unmatched subjects (if any)
                    ux, uy = cluster_points.get(-1, (np.array([]), np.array([])))
                    if len(ux) > 0:
                        ax.scatter(ux, uy,
                                  facecolors=to_rgba(color_asd, 0.35), s=100,
                                  edgecolors='black', linewidths=0.8,
                                  label='_nolegend_', zorder=4)
                else:
                    # No clustering: uniform color with a plain outline
                    common_ids = asd_male.index.intersection(zscore_asd.index)
                    asd_male_matched = asd_male.loc[common_ids]
                    ax.scatter(asd_male_matched['age'], asd_male_matched[network],
                              facecolors=to_rgba(color_asd, 0.45), s=100,
                              edgecolors='black', linewidths=0.8,
                              label='_nolegend_', zorder=4)

    # Overlay 2D KDE density contours per cluster (radar colors)
    if cluster_density and cluster_points:
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        grid_x, grid_y = np.mgrid[x_lo:x_hi:120j, y_lo:y_hi:120j]
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()])

        for cid, (cx, cy) in sorted(cluster_points.items()):
            if cid == -1 or len(cx) < 5:
                continue
            try:
                kde = stats.gaussian_kde(np.vstack([cx, cy]))
            except np.linalg.LinAlgError:
                continue  # degenerate (collinear) cluster
            density = kde(grid_points).reshape(grid_x.shape)
            fill_color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            # Nested filled bands at fractions of that cluster's peak density,
            # drawn beneath the scatter and getting more opaque toward the peak.
            band_edges = np.array([0.25, 0.5, 0.75, 1.0]) * density.max()
            band_alphas = [0.12, 0.22, 0.35]
            for lo, hi, band_alpha in zip(band_edges[:-1], band_edges[1:], band_alphas):
                ax.contourf(grid_x, grid_y, density, levels=[lo, hi],
                            colors=[fill_color], alpha=band_alpha, zorder=2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    # Draw per-cluster regression lines
    if cluster_slopes is not None:
        CLUSTER_EDGE_COLORS_LINE = CLUSTER_COLORS
        age_range = ax.get_xlim()
        age_line = np.linspace(age_range[0], age_range[1], 100)
        for cid_str, slope_info in sorted(cluster_slopes.items()):
            cid = int(cid_str)
            slope = slope_info['slope']
            intercept = slope_info['intercept']
            fitted = slope * age_line + intercept
            line_color = CLUSTER_EDGE_COLORS_LINE[cid % len(CLUSTER_EDGE_COLORS_LINE)]
            ax.plot(age_line, fitted, color=line_color, linewidth=2.5,
                    linestyle='--', label=f'Cluster {cid} fit', zorder=6)

    # Labels and title
    network_label = NETWORK_LABELS[network]
    feature_label = FEATURE_LABELS[feature]

    # Title suffix based on groups shown
    if plot_groups == 'td_only':
        title_suffix = '(TD Male Only)'
    elif plot_groups == 'asd_only':
        title_suffix = '(ASD Male Only)'
    else:
        title_suffix = '(Male Only)'

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(f'{feature_label} Score', fontsize=12)
    ax.set_title(f'{network_label} - {feature_label} {title_suffix}',
                fontsize=14, fontweight='bold', pad=15)

    # Legend (show when cluster coloring is active)
    # if cluster_labels is not None:
    #     ax.legend(loc='best', frameon=True, fontsize=9, framealpha=0.9)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.tick_params(axis='both', which='major', direction='out', length=6, width=1,
    #               bottom=True, left=True, top=False, right=False)

    # Set y-axis ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()

    # Save figure
    screen_tag = f"screen-{CONFIG['screen']}"

    if plot_groups == 'td_only':
        groups_tag = 'tdonly'
    elif plot_groups == 'asd_only':
        groups_tag = 'asdonly'
    else:
        groups_tag = 'both'

    if interval_type == 'confidence':
        interval_tag = 'ci'
    elif interval_type == 'prediction':
        interval_tag = 'pi'
    elif interval_type == 'none':
        interval_tag = 'noband'
    else:
        interval_tag = 'ci+pi'

    cluster_tag = f'_{cluster_method}' if cluster_method else ''
    filename = f'gam_trajectory_{network}_{feature}_{groups_tag}_{interval_tag}_{screen_tag}{cluster_tag}.png'
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")

    plt.close(fig)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function for heterogeneity analysis."""
    print("=" * 80)
    print("12_heterogeneity: ASD Heterogeneity Analysis")
    print("=" * 80)
    print(f"\nOutput directory: {PATHS['figures']}")

    # Set plotting style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)

    # =========================================================================
    # SECTION 1: Load Data
    # =========================================================================
    print_separator('=')
    print("SECTION 1: LOAD DATA")
    print_separator('=')

    zscore_td, zscore_asd = load_zscore_data(PATHS)

    # Note: Age-based screening is now applied upstream in 11_sem_zscore_0save_raw.py
    # (controlled by CONFIG['screen'] parameter)

    participants_df = load_behavioral_data(PATHS, list(zscore_asd.index), CONFIG)
    comorbid_df = load_comorbidity_data(PATHS, list(zscore_asd.index))

    # Filter to complete cases for behavioral analysis
    behav_items = CONFIG['behav_items_srs'] + CONFIG['behav_items_nih']
    zscore_asd_complete, participants_complete, complete_ids = filter_complete_cases(
        zscore_asd, participants_df, behav_items
    )

    # Load TD behavioral data for TD reference boxes on metric plots
    td_participants = load_behavioral_data(PATHS, list(zscore_td.index), CONFIG)
    td_behav_data = td_participants[behav_items].apply(pd.to_numeric, errors='coerce')

    # =========================================================================
    # SECTION 2: Outlier Prevalence Analysis
    # =========================================================================
    print_separator('=')
    print("SECTION 2: OUTLIER PREVALENCE ANALYSIS")
    print_separator('=')

    threshold = CONFIG['outlier_threshold']
    print(f"\nOutlier threshold: z > {threshold}")

    # Overall prevalence
    print("\n2.1 Overall Prevalence")
    prevalence_df = calculate_overall_prevalence(zscore_asd, threshold)
    print(prevalence_df.to_string(index=False))

    # Age-stratified prevalence
    print("\n2.2 Age-Stratified Prevalence")
    for var in ZSCORE_VARS:
        parts = var.split('_')
        feature_type = parts[0]
        network_name = parts[1]

        age_prev_df = calculate_age_stratified_prevalence(
            zscore_asd, var,
            CONFIG['age_bins'], CONFIG['age_labels'],
            threshold
        )

        print(f"\n  {var}:")
        print(age_prev_df.to_string(index=False))

        # Plot
        plot_age_stratified_bars(age_prev_df, network_name, feature_type, PATHS['figures'])

    # =========================================================================
    # SECTION 2B: GAM Trajectory Plots
    # =========================================================================
    print_separator('=')
    print("SECTION 2B: GAM TRAJECTORY PLOTS")
    print_separator('=')

    # Load trajectory data (generated by 11_sem_zscore_1gam.R)
    trajectories = load_gam_trajectories(PATHS)
    raw_data = load_raw_network_data(PATHS)

    # Check if trajectory data is available
    has_trajectories = any(
        trajectories[f][g] is not None
        for f in ['sem', 'dim']
        for g in ['td', 'asd']
    )

    if has_trajectories:
        print("\n2B.1 Generating GAM trajectory plots...")

        # Generate plots for each network-feature combination
        for network in NETWORK_NAMES:
            for feature in FEATURE_TYPES:
                print(f"\n  Plotting {NETWORK_LABELS[network]} - {FEATURE_LABELS[feature]}...")
                try:
                    plot_gam_trajectory(
                        network=network,
                        feature=feature,
                        trajectories=trajectories,
                        raw_data=raw_data,
                        zscore_asd=zscore_asd,
                        save_dir=PATHS['figures'],
                        zscore_td=zscore_td,
                        outlier_threshold=CONFIG['outlier_threshold'],
                        plot_groups='td_only',
                        interval_type='prediction',
                        scatter=True,
                        show_adherence=True
                    )
                except Exception as e:
                    print(f"    Error: {e}")
    else:
        print("\n  No trajectory data found. Run 11_sem_zscore_1gam.R first to generate trajectory files.")
        print("  Skipping GAM trajectory plots.")

    # =========================================================================
    # Clustering method selection & shared preparation
    # =========================================================================
    run_kmeans = CONFIG['clustering_method'] in ('both', 'kmeans')
    run_hierarchical = CONFIG['clustering_method'] in ('both', 'hierarchical')
    print(f"\nClustering method: {CONFIG['clustering_method']}")

    # Determine normalization mode and create feature matrix
    normalize_profiles = CONFIG['normalize_profiles']
    if normalize_profiles:
        print("\nMode: Profile-normalized (subtype patterns)")
        print("  Note: Per-subject demeaning removes global severity effect.")
        print("        Clusters represent PROFILE PATTERNS, not severity levels.")
        feature_matrix_all, feature_names = create_profile_normalized_matrix(zscore_asd)
        # Also create profile-normalized DataFrame for visualization
        zscore_asd_viz = zscore_asd.copy()
        subject_means = zscore_asd[ZSCORE_VARS].values.mean(axis=1)
        for var in ZSCORE_VARS:
            zscore_asd_viz[var] = zscore_asd[var].values - subject_means
        method_suffix = '_profile'
    else:
        print("\nMode: Original z-scores (severity-based)")
        feature_matrix_all, feature_names = create_zscore_matrix_6d(zscore_asd)
        zscore_asd_viz = zscore_asd  # Use original z-scores for visualization
        method_suffix = ''

    print(f"Feature matrix (all ASD): {feature_matrix_all.shape}")

    # --- Combined silhouette comparison (K-means vs Hierarchical) ---
    # Overlay both silhouette curves on a single plot for method comparison.
    print("\nComputing silhouette scores for K-means and hierarchical clustering...")
    km_metrics, _ = kmeans_clustering_with_elbow(feature_matrix_all, max_k=CONFIG['max_k'])
    hc_metrics, _ = hierarchical_clustering_with_metrics(feature_matrix_all, max_k=CONFIG['max_k'])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(km_metrics['k_values'], km_metrics['silhouette_scores'], 'o-',
            linewidth=2, markersize=8, color='black', label='K-means')
    ax.plot(hc_metrics['k_values'], hc_metrics['silhouette_scores'], 'o-',
            linewidth=2, markersize=8, color='grey', label='Hierarchical')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score: K-means vs Hierarchical', fontsize=12, fontweight='bold')
    ax.set_xticks(km_metrics['k_values'])
    ax.tick_params(axis='both', which='major', length=6, width=1.5,
                   direction='out', bottom=True, left=True, color='black')
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _sil_filename = f'clustering_silhouette_comparison{method_suffix}.png'
    fig.savefig(PATHS['figures'] / _sil_filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {_sil_filename}")
    plt.close(fig)

    # Shared data for characterization
    all_asd_ids = list(zscore_asd.index)
    complete_indices = np.array([i for i, sid in enumerate(all_asd_ids) if sid in complete_ids])
    binary_matrix_complete, _ = create_binary_outlier_matrix_6d(zscore_asd_complete, threshold)
    ages_all = zscore_asd['age'].values
    sexes_all = zscore_asd['sex'].values
    binary_matrix_all, _ = create_binary_outlier_matrix_6d(zscore_asd, threshold)
    ages_complete = zscore_asd_complete['age'].values
    sexes_complete = zscore_asd_complete['sex'].values
    behav_data = participants_complete[behav_items].astype(float)

    # Initialize variables that may be referenced in save section
    kmeans_labels_all = None
    hier_labels_all = None
    behav_summary = None
    hier_behav = None

    # =========================================================================
    # SECTION 3+4: K-Means Clustering & Characterization
    # =========================================================================
    if run_kmeans:
        print_separator('=')
        print("SECTION 3: K-MEANS CLUSTERING")
        print_separator('=')

        # Run K-means on full sample
        print("\n3.1 Running K-means clustering...")
        kmeans_results, kmeans_models = kmeans_clustering_with_elbow(
            feature_matrix_all, max_k=CONFIG['max_k']
        )

        print(f"  Optimal k (elbow): {kmeans_results['optimal_k_elbow']}")
        print(f"  Optimal k (silhouette): {kmeans_results['optimal_k_silhouette']}")
        print(f"  Silhouette scores: {[f'{s:.3f}' for s in kmeans_results['silhouette_scores']]}")

        # Plot metrics
        plot_clustering_metrics(kmeans_results, PATHS['figures'])

        # Use manual k if specified, otherwise use silhouette-optimal k
        if CONFIG['manual_k'] is not None:
            optimal_k = CONFIG['manual_k']
            print(f"\n3.2 Using manual k={optimal_k}")
        else:
            optimal_k = kmeans_results['optimal_k_silhouette']
            print(f"\n3.2 Selected k={optimal_k} (silhouette-optimal)")
        kmeans_labels_all = kmeans_models[optimal_k].fit_predict(feature_matrix_all)
        print(f"  Full sample cluster sizes:")
        for i in range(optimal_k):
            print(f"  Cluster {i}: {(kmeans_labels_all == i).sum()} subjects")

        # Get cluster labels for behavioral-complete subset
        kmeans_labels_complete = kmeans_labels_all[complete_indices]
        print(f"\n  Behavioral-complete subset: {len(kmeans_labels_complete)} subjects")

        # --- Section 4: K-Means Characterization ---
        print_separator('=')
        print("SECTION 4: K-MEANS CLUSTER CHARACTERIZATION")
        print_separator('=')

        # Characterize clusters using ALL subjects (demographics + feature profiles)
        cluster_summary, cluster_features, _ = characterize_clusters(
            binary_matrix_all, kmeans_labels_all, ages_all, sexes_all, feature_names,
            behav_data=None, behav_item_list=None
        )

        # Behavioral analysis using complete cases only
        _, _, behav_summary = characterize_clusters(
            binary_matrix_complete, kmeans_labels_complete, ages_complete, sexes_complete, feature_names,
            behav_data=behav_data, behav_item_list=behav_items
        )

        print(f"\n4.1 Cluster Demographics (all {len(all_asd_ids)} subjects):")
        print(cluster_summary.to_string(index=False))

        print("\n4.2 Cluster Feature Profiles (% outliers):")
        print(cluster_features.to_string(index=False))

        # Plot heatmap using correctly computed cluster_features
        plot_cluster_feature_heatmap(cluster_features, PATHS['figures'], method=f'kmeans{method_suffix}')

        # Plot mean z-score heatmap (shows severity, not just binary outlier status)
        plot_cluster_zscore_heatmap(zscore_asd_viz, kmeans_labels_all, ZSCORE_VARS, PATHS['figures'], method=f'kmeans{method_suffix}')

        # Plot demographics (using all subjects)
        plot_cluster_demographics(kmeans_labels_all, ages_all, sexes_all, PATHS['figures'], method=f'kmeans{method_suffix}')

        # Plot radar and parallel coords for profile visualization
        # Use original z-scores to show actual magnitudes (clustering was done on normalized profiles)
        if normalize_profiles:
            plot_cluster_radar(zscore_asd, kmeans_labels_all, ZSCORE_VARS_RADAR,
                              PATHS['figures'], method='kmeans', title_suffix='_profile')
            plot_cluster_parallel_coords(zscore_asd, kmeans_labels_all, ZSCORE_VARS,
                                        PATHS['figures'], method='kmeans', title_suffix='_profile')

        # Plot behavioral measures
        if behav_summary is not None:
            print("\n4.3 Behavioral Measures:")
            print(behav_summary.to_string(index=False))

            for _, row in behav_summary.iterrows():
                plot_single_behavioral_measure(
                    behav_data, kmeans_labels_complete, row['item'],
                    PATHS['figures'], method=f'kmeans{method_suffix}',
                    p_value=row['p_value'], test_name=row['test'],
                    td_behav_data=td_behav_data
                )
    else:
        print("\n[Skipping K-means clustering]")

    # =========================================================================
    # SECTION 5: Hierarchical Clustering
    # =========================================================================
    if run_hierarchical:
        print_separator('=')
        print("SECTION 5: HIERARCHICAL CLUSTERING")
        print_separator('=')

        # Use full ASD sample for hierarchical clustering (same feature matrix as K-means)
        print("\n5.1 Running hierarchical clustering (Ward linkage)...")
        hier_results, linkage_mat = hierarchical_clustering_with_metrics(
            feature_matrix_all, max_k=CONFIG['max_k']
        )

        print(f"  Optimal k (silhouette): {hier_results['optimal_k_silhouette']}")
        print(f"  Silhouette scores: {[f'{s:.3f}' for s in hier_results['silhouette_scores']]}")

        # Plot dendrogram and metrics
        plot_dendrogram(linkage_mat, PATHS['figures'], CONFIG['max_k'])
        plot_hierarchical_metrics(hier_results, PATHS['figures'])

        # Use manual k if specified, otherwise use silhouette-optimal k
        if CONFIG['manual_k'] is not None:
            hier_k = CONFIG['manual_k']
            print(f"\n5.2 Using manual k={hier_k}")
        else:
            hier_k = hier_results['optimal_k_silhouette']
            print(f"\n5.2 Selected k={hier_k} (silhouette-optimal)")
        hier_labels_all = fcluster(linkage_mat, hier_k, criterion='maxclust') - 1  # 0-indexed

        print(f"  Full sample cluster sizes:")
        for i in range(hier_k):
            print(f"    Cluster {i}: {(hier_labels_all == i).sum()} subjects")

        # Get hierarchical labels for behavioral-complete subset
        hier_labels_complete = hier_labels_all[complete_indices]
        print(f"\n  Behavioral-complete subset: {len(hier_labels_complete)} subjects")

        # Characterize hierarchical clusters using ALL subjects (demographics + feature profiles)
        hier_summary, hier_features, _ = characterize_clusters(
            binary_matrix_all, hier_labels_all, ages_all, sexes_all, feature_names,
            behav_data=None, behav_item_list=None
        )

        # Behavioral analysis using complete cases only
        _, _, hier_behav = characterize_clusters(
            binary_matrix_complete, hier_labels_complete, ages_complete, sexes_complete, feature_names,
            behav_data=behav_data, behav_item_list=behav_items
        )

        print(f"\n5.3 Cluster Demographics (Hierarchical, all {len(all_asd_ids)} subjects):")
        print(hier_summary.to_string(index=False))

        # Plot (using all subjects)
        plot_cluster_demographics(hier_labels_all, ages_all, sexes_all, PATHS['figures'], method=f'hierarchical{method_suffix}')
        plot_cluster_feature_heatmap(hier_features, PATHS['figures'], method=f'hierarchical{method_suffix}')

        # Plot mean z-score heatmap (shows severity, not just binary outlier status)
        plot_cluster_zscore_heatmap(zscore_asd_viz, hier_labels_all, ZSCORE_VARS, PATHS['figures'], method=f'hierarchical{method_suffix}')

        # Plot radar and parallel coords for profile visualization
        # Use original z-scores to show actual magnitudes (clustering was done on normalized profiles)
        if normalize_profiles:
            plot_cluster_radar(zscore_asd, hier_labels_all, ZSCORE_VARS_RADAR,
                              PATHS['figures'], method='hierarchical', title_suffix='_profile')
            plot_cluster_parallel_coords(zscore_asd, hier_labels_all, ZSCORE_VARS,
                                        PATHS['figures'], method='hierarchical', title_suffix='_profile')

        if hier_behav is not None:
            for _, row in hier_behav.iterrows():
                plot_single_behavioral_measure(
                    behav_data, hier_labels_complete, row['item'],
                    PATHS['figures'], method=f'hierarchical{method_suffix}',
                    p_value=row['p_value'], test_name=row['test'],
                    td_behav_data=td_behav_data
                )
    else:
        print("\n[Skipping hierarchical clustering]")

    # =========================================================================
    # SECTION 6: Cluster-Colored GAM Trajectory Plots & Trajectory Comparison
    # =========================================================================
    if has_trajectories and (kmeans_labels_all is not None or hier_labels_all is not None):
        print_separator('=')
        print("SECTION 6: CLUSTER TRAJECTORY ANALYSIS")
        print_separator('=')

        for method_name, labels in [('kmeans', kmeans_labels_all), ('hierarchical', hier_labels_all)]:
            if labels is None:
                continue

            # 6.1 Test whether cluster trajectories are distinct
            print(f"\n  6.1 Trajectory comparison ({method_name}):")
            print(f"      Model: score ~ age + cluster + age:cluster (male only)")
            traj_comparison_df = run_trajectory_comparison(
                raw_data=raw_data,
                cluster_labels=labels,
                cluster_label_index=all_asd_ids,
                fdr_alpha=CONFIG['fdr_alpha']
            )

            if len(traj_comparison_df) > 0 and 'error' not in traj_comparison_df.columns:
                # Print summary table
                print_cols = ['feature', 'network', 'n_total',
                              'cluster_F', 'cluster_p', 'cluster_q',
                              'interaction_F', 'interaction_p', 'interaction_q']
                available_cols = [c for c in print_cols if c in traj_comparison_df.columns]
                print(traj_comparison_df[available_cols].to_string(index=False, float_format='%.4f'))

                # Save results
                traj_filename = f'trajectory_comparison_{method_name}{method_suffix}.csv'
                traj_comparison_df.to_csv(PATHS['out'] / traj_filename, index=False)
                print(f"\n  Saved: {traj_filename}")

            # 6.2 Build slopes lookup for plotting: {(feature, network): slope_dict}
            slopes_lookup = {}
            if len(traj_comparison_df) > 0:
                for _, row in traj_comparison_df.iterrows():
                    if 'slope_per_cluster' in row and isinstance(row.get('slope_per_cluster'), dict):
                        slopes_lookup[(row['feature'], row['network'])] = row['slope_per_cluster']

            # 6.3 Plot cluster-colored GAM trajectories with regression lines
            print(f"\n  6.2 Generating GAM trajectories colored by {method_name} clusters...")
            for network in NETWORK_NAMES:
                for feature in FEATURE_TYPES:
                    try:
                        plot_gam_trajectory(
                            network=network,
                            feature=feature,
                            trajectories=trajectories,
                            raw_data=raw_data,
                            zscore_asd=zscore_asd,
                            save_dir=PATHS['figures'],
                            zscore_td=zscore_td,
                            outlier_threshold=CONFIG['outlier_threshold'],
                            plot_groups='td_only',
                            interval_type='none',   # no shaded normative band
                            scatter=True,
                            show_adherence=False,
                            cluster_labels=labels,
                            cluster_label_index=all_asd_ids,
                            cluster_method=f'{method_name}{method_suffix}',
                            cluster_slopes=None,    # no per-cluster fit lines
                            cluster_density=True    # 2D KDE contours instead
                        )
                    except Exception as e:
                        print(f"    Error ({network}_{feature}): {e}")

    # =========================================================================
    # SECTION 7: Comorbidity Analysis
    # =========================================================================
    if comorbid_df is not None and (run_kmeans or run_hierarchical):
        print_separator('=')
        print("SECTION 7: COMORBIDITY ANALYSIS")
        print_separator('=')

        # Use ALL ASD subjects for comorbidity analysis (independent of behavioral completeness)
        # This matches the original script's approach
        print(f"\n  Using full ASD sample for comorbidity analysis: {len(all_asd_ids)} subjects")

        # K-means clusters (full sample)
        if run_kmeans:
            print("\n7.1 Comorbidity by K-Means Clusters:")
            comorbid_results_km, comorbid_prev_km = analyze_comorbidity_prevalence(
                kmeans_labels_all, all_asd_ids, comorbid_df
            )
            print(comorbid_results_km.to_string(index=False))
            plot_comorbidity_prevalence(comorbid_prev_km, comorbid_results_km, PATHS['figures'], method=f'kmeans{method_suffix}')

        # Hierarchical clusters (full sample)
        if run_hierarchical:
            print("\n7.2 Comorbidity by Hierarchical Clusters:")
            comorbid_results_hier, comorbid_prev_hier = analyze_comorbidity_prevalence(
                hier_labels_all, all_asd_ids, comorbid_df
            )
            print(comorbid_results_hier.to_string(index=False))
            plot_comorbidity_prevalence(comorbid_prev_hier, comorbid_results_hier, PATHS['figures'], method=f'hierarchical{method_suffix}')

    # =========================================================================
    # SECTION 8: Save Results
    # =========================================================================
    print_separator('=')
    print("SECTION 8: SAVE RESULTS")
    print_separator('=')

    # Determine mode label for filenames
    mode_label = 'profile' if normalize_profiles else 'severity'

    # Save cluster assignments for ALL subjects
    assignment_cols = {'subject_id': all_asd_ids}
    if kmeans_labels_all is not None:
        assignment_cols['kmeans_cluster'] = kmeans_labels_all
    if hier_labels_all is not None:
        assignment_cols['hierarchical_cluster'] = hier_labels_all
    cluster_assignments_all = pd.DataFrame(assignment_cols)
    cluster_filename = f'cluster_assignments_{mode_label}.csv'
    # Written to the pipeline output directory.  The original put it beside the
    # figures, which made a figures folder a pipeline dependency.
    cluster_assignments_all.to_csv(PATHS['out'] / cluster_filename, index=False)
    print(f"  Saved: {cluster_filename} (all {len(all_asd_ids)} subjects)")

    # Save prevalence results
    prevalence_df.to_csv(PATHS['out'] / 'outlier_prevalence_overall.csv', index=False)
    print(f"  Saved: outlier_prevalence_overall.csv")

    # Save behavioral results
    if behav_summary is not None:
        kmeans_behav_filename = f'behavioral_analysis_kmeans_{mode_label}.csv'
        behav_summary.to_csv(PATHS['out'] / kmeans_behav_filename, index=False)
        print(f"  Saved: {kmeans_behav_filename}")

    if hier_behav is not None:
        hier_behav_filename = f'behavioral_analysis_hierarchical_{mode_label}.csv'
        hier_behav.to_csv(PATHS['out'] / hier_behav_filename, index=False)
        print(f"  Saved: {hier_behav_filename}")

    print("\n" + "=" * 80)
    print("Done! All figures and results saved to:")
    print(f"  {PATHS['figures']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
