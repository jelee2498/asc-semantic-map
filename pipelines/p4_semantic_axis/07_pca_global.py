"""
Global comparison of TD vs ASD average semantic spaces.

Refactored from: 07_pca.py (visualization sections)

This script performs:
1. Scatter plot between avg_sem_td and avg_sem_asd (per-ROI comparison)
2. Lollipop plots for MMP sections for each group (TD, ASD)

All functions are defined locally - this is a standalone script with no external dependencies.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, fig_dir  # noqa: E402

import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

# Shared spatial-null library (spin permutation + eigenstrapping); lives in lib/,
# already on sys.path via the project_config import above.
import spatial_nulls as snull


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '07_pca',

    # Source data pipelines
    'source_pipeline_participants': '99_main',
    'source_pipeline_encoding': '99_main',
    'source_task_participants': '05_prepare_reg',
    'source_task_encoding': '06_encoding_model',

    # Data parameters
    'atlas': 'mmp',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option': 0,
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha': True,
    'weight_across_delays': 'Avg',
    'weight_add_superordinate': True,

    # Performance mask
    'perf_method': 'fdr',
    'perf_alpha': 0.01,

    # Analysis parameters
    'comp_no': 1,
    'hx_docu': 'True',

    # Seed parameters
    'seed': 0,

    # Spatial null models (see spatial_nulls.py). PCA loadings are only defined
    # inside the performance mask, so the correspondence test stays within it.
    'pvalue_within_perf_mask': True,
    'seed_surro': 0,             # Seed for both null models, independent of 'seed'
    'spin_n_rot': 2000,          # Sphere rotations
    'es_n_surr': 2000,           # Eigenstrapping surrogates per map
    'es_num_modes': 200,         # Geometric eigenmodes per hemisphere
    'es_surface': 'midthickness',
    'es_n_jobs': 1,              # Keep at 1 for reproducibility
    'sens_save_excel': True,     # Sensitivity tables are always CSV; also write .xlsx
}


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Construct all project paths from CONFIG with platform detection."""
    # Paths come from config/paths.yml via lib/project_config.py.  The original
    # working tree hard-coded storage roots here (S:/, Q:/, V:/ on Windows;
    # /Volumes/* on macOS; /MIPL/store* on Linux); those are machine-specific and
    # are not distributed.
    proj_root = PROJECT

    proj = config['project']
    pipe = config['pipeline']
    task = config['task']

    paths = {
        'raw': proj_root / '0_data' / 'raw',
        'code': proj_root / '1_code' / pipe,
        'tpl': TEMPLATES,
        'fig': fig_dir(__file__),
        'pipe': proj_root / '2_pipeline' / pipe,

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

    paths['proc'] = paths['pipe'] / task
    paths['out'] = paths['proc'] / 'out'
    paths['save'] = paths['proc'] / 'save'
    paths['tmp'] = paths['proc'] / 'tmp'

    return paths


# =============================================================================
# CONSTANTS
# =============================================================================

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


# =============================================================================
# ATLAS LOADING (minimal)
# =============================================================================

def load_atlases(paths: Dict[str, Path]) -> Dict[str, npt.NDArray]:
    """Load MMP and section atlases from template directory."""
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

    return {
        'mmp': mmp_atlas,
        'section': section_atlas,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def value_to_rgb(
    value: float,
    cmap_name: str,
    vmin: float,
    vmax: float
) -> Tuple[float, float, float]:
    """Convert a value to RGB color using matplotlib colormap."""
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(cmap_name)
    norm_value = (value - vmin) / (vmax - vmin)
    norm_value = np.clip(norm_value, 0, 1)
    rgb = cmap(norm_value)[:3]
    return rgb


def mmp_2_section(
    array_mmp: npt.NDArray,
    mmp_atlas: npt.NDArray,
    section_atlas: npt.NDArray,
    model_perf_mask: Optional[npt.NDArray] = None,
    sum: bool = False
) -> npt.NDArray:
    """Convert MMP parcellation (360 ROIs) to MMP section (22 sections)."""
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


def build_encoding_results_path(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Path:
    """Build path to encoding results based on configuration."""
    base = (paths['enc_results'] / config['conf_option'] / config['atlas'] /
            f"chunk-{config['chunk_option']}" / 'fold-avg' /
            f"d-{config['hx_docu']}" / f"k-{config['srm_option']}")

    if not config['bias_weight']:
        return (base / 'gaze_weight-False' /
                f"wn-{config['reg_wordnet']}" /
                f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}")
    else:
        return (base /
                f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f"wn-{config['reg_wordnet']}" /
                f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}")


def get_semantic_output_path(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> Path:
    """Construct output path for semantic analysis results."""
    base = (paths['out'] / config['conf_option'] / config['atlas'] /
            f"chunk-{config['chunk_option']}" / 'fold-avg' / 'results' /
            f"k-{config['srm_option']}")

    if not config['bias_weight']:
        path = (base / 'gaze_weight-False' /
                f"wn-{config['reg_wordnet']}" / f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}" / 'sem' /
                f"mask_{config['perf_method']}-{config['perf_alpha']}" /
                f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}")
    else:
        path = (base /
                f"bias-{config['bias_weight']}_verb-{config['verb_weight']}" /
                f"wn-{config['reg_wordnet']}" / f"seed-{config['seed']}" /
                f"s_alpha-{config['enc_single_alpha']}" / 'sem' /
                f"mask_{config['perf_method']}-{config['perf_alpha']}" /
                f"delay-{config['weight_across_delays']}_super-{config['weight_add_superordinate']}")

    return path


def load_performance_mask(
    config: Dict[str, Any],
    paths: Dict[str, Path]
) -> npt.NDArray:
    """Load model performance significance mask."""
    results_path = build_encoding_results_path(config, paths)
    mask_path = results_path / f"mask_{config['perf_method']}-{config['perf_alpha']}.npy"
    perf_mask = np.load(mask_path)
    return perf_mask


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("07_pca_global.py - Global Comparison of TD vs ASD Semantic Spaces")
    print("=" * 80)

    # =========================================================================
    # 1. Initialize
    # =========================================================================
    PATHS = setup_paths(CONFIG)
    ATLASES = load_atlases(PATHS)

    comp_no = CONFIG['comp_no']

    # Figure output directory
    fig_dir = PATHS['fig'] / 'global'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 2. Load data
    # =========================================================================
    print("\n[1] Loading data...")
    sem_path = get_semantic_output_path(CONFIG, PATHS)
    avg_sem_td = np.load(sem_path / 'avg_sem_td.npy')
    avg_sem_asd = np.load(sem_path / 'avg_sem_asd.npy')
    perf_mask = load_performance_mask(CONFIG, PATHS)

    print(f"  avg_sem_td shape: {avg_sem_td.shape}")
    print(f"  avg_sem_asd shape: {avg_sem_asd.shape}")
    print(f"  perf_mask: {len(perf_mask)} ROIs")

    # Extract component
    td_vals = avg_sem_td[:, comp_no - 1]
    asd_vals = avg_sem_asd[:, comp_no - 1]

    # =========================================================================
    # 3. Scatter plot: TD vs ASD per ROI
    # =========================================================================
    print(f"\n[2] Scatter plot (PC#{comp_no})...")

    r_val, p_val = pearsonr(td_vals, asd_vals)
    print(f"  Pearson r={r_val:.3f}, p={p_val:.3e}")

    # Spatial null models. td_vals/asd_vals are mask-length (avg_sem_* is stored over
    # perf_mask ROIs only), but rotations are defined over the whole cortex - so scatter
    # them back into full 360-parcel maps, spin those, and restrict afterwards. Parcels
    # outside the mask stay NaN because PCA loadings are not defined there.
    td_vals_full = np.full(360, np.nan)
    asd_vals_full = np.full(360, np.nan)
    td_vals_full[perf_mask] = td_vals
    asd_vals_full[perf_mask] = asd_vals

    corr_mask = perf_mask if CONFIG['pvalue_within_perf_mask'] else None

    null_row = snull.compare_maps(
        td_vals_full, asd_vals_full,
        label=f'pca_pc{comp_no}_td_vs_asd',
        mask=corr_mask,
        tpl_path=PATHS['tpl'],
        n_rot=CONFIG['spin_n_rot'], n_surr=CONFIG['es_n_surr'],
        num_modes=CONFIG['es_num_modes'], surface=CONFIG['es_surface'],
        seed=CONFIG['seed_surro'], n_jobs=CONFIG['es_n_jobs'],
    )
    spin_p = null_row['spin_p']
    eigen_p = null_row['eigenstrap_p']
    print(f"  Spin permutation p={spin_p:.4e} (one-sided)")
    print(f"  Eigenstrapping   p={eigen_p:.4e} (one-sided)")

    snull.save_sensitivity_table(
        pd.DataFrame([null_row]), fig_dir / 'sensitivity',
        f'null_model_comparison_pca_pc{comp_no}_td_vs_asd',
        also_excel=CONFIG['sens_save_excel'])

    fig, ax = plt.subplots(figsize=(10, 10))
    norm = plt.Normalize(td_vals.min(), td_vals.max())
    cmap_obj = plt.get_cmap('coolwarm')
    sns.regplot(
        x=td_vals, y=asd_vals,
        scatter=True, ax=ax,
        scatter_kws={'s': 600, 'color': cmap_obj(norm(td_vals)), 'alpha': 1},
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2}
    )
    plt.title(f'r={r_val:.3f}, p_spin={spin_p:.4e}', fontsize=25)
    plt.xlabel(f'TD avg sem (PC#{comp_no})', fontsize=20)
    plt.ylabel(f'ASD avg sem (PC#{comp_no})', fontsize=20)
    ax.tick_params(tick1On=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / f'scatter_td_vs_asd_PC{comp_no}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # =========================================================================
    # 4. Lollipop plots: MMP sections for TD and ASD
    # =========================================================================
    print(f"\n[3] Lollipop plots (PC#{comp_no})...")

    cmax = 10
    cmin = -cmax
    cmap_name = 'coolwarm'

    for group_name, avg_sem in [('TD', avg_sem_td), ('ASD', avg_sem_asd)]:
        # Convert to section space
        section_vals = mmp_2_section(
            avg_sem[:, comp_no - 1],
            ATLASES['mmp'],
            ATLASES['section'],
            model_perf_mask=perf_mask
        ).squeeze()

        # Build DataFrame
        section_df = pd.DataFrame(
            section_vals,
            index=MMP_SECTION_LIST,
            columns=[f'PC#{comp_no}']
        )
        section_df = section_df.sort_values(by=f'PC#{comp_no}', ascending=False)
        sorted_sections = list(section_df.index)

        # Add color column
        section_df[f'PC#{comp_no}_color'] = section_df[f'PC#{comp_no}'].apply(
            lambda x: value_to_rgb(x, cmap_name, cmin, cmax)
        )

        # Plot lollipop
        fig, ax = plt.subplots(figsize=(13, 8))
        plt.title(f'PC#{comp_no} - {group_name} avg semantic space (MMP section)')

        plt.stem(section_df[f'PC#{comp_no}'], linefmt='silver', markerfmt=' ', basefmt=' ')

        for i, section in enumerate(sorted_sections):
            plt.plot(
                i,
                section_df.loc[section, f'PC#{comp_no}'],
                marker='o',
                color=section_df.loc[section, f'PC#{comp_no}_color'],
                markersize=24
            )

        plt.xticks(np.arange(len(sorted_sections)), sorted_sections, rotation=45, ha='right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(tick1On=True)
        plt.tight_layout()
        plt.savefig(fig_dir / f'lollipop_{group_name}_PC{comp_no}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"  {group_name}: saved lollipop plot")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
