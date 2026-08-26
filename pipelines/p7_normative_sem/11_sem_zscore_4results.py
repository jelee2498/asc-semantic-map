"""
11_sem_zscore_4results.py
Scatter plots of DMN alteration vs. behavioral outcomes (SEM Model 1 results).

This script:
1. Loads the SEM input table written by 11_sem_zscore_2save_data.py
   (all_sem_dim_zscore.csv)
2. Builds observed proxies for the two SEM latent variables:
     - Autistic symptom  <- 5 SRS T-scores      (srs_beh in the lavaan model)
     - Cognitive ability <- 4 NIH7 percentiles  (cog_beh in the lavaan model)
3. Draws one scatter plot per outcome, with alteration of DMN (avg_dmn) on the
   x-axis, and colors each dot by symptom severity:
     - Autistic symptom  : 'Reds'  , HIGHER value = more severe = darker
     - Cognitive ability : 'Blues' , LOWER  value = more severe = darker

These panels visualize the two behavioral paths of Model 1, which was the
selected model in 11_sem_zscore_3analysis.R:
    srs_beh ~ avg_dmn   (d = 0.414,  z =  8.928)
    cog_beh ~ avg_dmn   (e = -0.184, z = -3.329)

Latent-score option
-------------------
CONFIG['latent_method'] controls how the two y-axis variables are built:
  'mean'         : unit-weighted mean of the (already z-scored) indicators.
                   Default. Requires no extra input.
  'factor_score' : lavaan factor scores, read from 'sem_factor_scores.csv' in
                   the same folder as all_sem_dim_zscore.csv. Expects an index
                   of subject IDs and columns 'srs_beh' and 'cog_beh'. Export
                   them from 11_sem_zscore_3analysis.R with, e.g.:

                       fs <- lavPredict(fit1)
                       fs <- as.data.frame(fs)
                       rownames(fs) <- rownames(triple_data)
                       write.csv(fs, file.path(data_folder, "sem_factor_scores.csv"))

                   (note: lavPredict drops listwise-deleted rows, so align the
                   row names to the cases actually used by the fit)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES, template, fig_dir  # noqa: E402


# =============================================================================
# CONFIGURATION (matching 11_sem_zscore_2save_data.py)
# =============================================================================

CONFIG = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '11_sem_zscore',

    # Data parameters
    'atlas': 'mmp',
    'conf_option': 'default+me',
    'chunk_option': 9,
    'srm_option_pc': 0,
    'srm_option_dim': 50,

    # Upstream analysis parameters (matching 11_sem)
    'bias_weight': 0.9,
    'verb_weight': 1.0,
    'reg_wordnet': True,
    'enc_single_alpha_semantic': True,
    'perf_method': 'fdr',
    'perf_alpha': 0.01,
    'weight_across_delays': 'Avg',
    'weight_add_superordinate': True,

    # Age-based screening (matching 0save_raw.py)
    'screen': True,
    'sig_overlap': False,

    # Behavioral items (the two measurement models of SEM Model 1)
    'behav_items_srs': ['SRS_AWR_T', 'SRS_COG_T', 'SRS_COM_T', 'SRS_MOT_T', 'SRS_RRB_T'],
    'behav_items_nih': ['NIH7_Card_P', 'NIH7_Flanker_P', 'NIH7_List_P', 'NIH7_Pattern_P'],

    # -------------------------------------------------------------------------
    # Analysis options
    # -------------------------------------------------------------------------
    # How to build the y-axis latent proxies: 'mean' or 'factor_score'
    'latent_method': 'mean',

    # Which subjects to plot. The SEM in 3analysis.R is fit on the pooled
    # sample, so 'all' is the panel that matches Fig. 5. Use 'ASD' for the
    # within-ASD supplementary version.
    'group': 'all',  # 'all' | 'ASD' | 'TD'

    # Restrict to the sample the SEM actually used. lavaan defaults to listwise
    # deletion, so fit1 in 3analysis.R drops any subject missing one of the 9
    # behavioral indicators or the 3 network scores. Keeping this True makes
    # both panels share that N; setting it False plots every available case and
    # the two panels will have different N.
    'complete_cases': True,

    # -------------------------------------------------------------------------
    # Plotting options
    # -------------------------------------------------------------------------
    'figsize': (12, 12),
    'scatter_size': 500,      # smaller than the parcel-level plots (n is ~subjects)
    'scatter_alpha': 1.0,
    'scatter_edgecolor': 'black',
    'scatter_edgewidth': 0.5,
    'label_fontsize': 20,
    'tick_fontsize': 20,
    'title_fontsize': 20,
    'n_ticks': 4,
    # Lightest dot uses this fraction of the colormap, so low-severity points
    # stay visible instead of fading into the white background.
    'color_floor': 0.25,
    'show_colorbar': False,   # redundant with the y-axis; on by request
    'dpi': 150,

    'save_outputs': True,
    'show_plots': True,
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
        'pipe': proj_root / '2_pipeline' / pipe,
        'code': proj_root / '1_code' / pipe,
    }

    # Where 11_sem_zscore_2save_data.py wrote all_sem_dim_zscore.csv
    paths['data'] = (
        paths['pipe'] / task / 'out' /
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

    paths['fig'] = (
        paths['code'] / 'claude_figures' / '11_sem_zscore_4results' /
        f"group-{config['group']}_latent-{config['latent_method']}"
    )

    return paths


PATHS = setup_paths(CONFIG)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sem_data(paths: Dict[str, Path], config: Dict[str, Any]) -> pd.DataFrame:
    """Load the SEM input table and restrict it to the requested group."""
    print("\n[1/3] Loading SEM data...")

    csv_path = paths['data'] / 'all_sem_dim_zscore.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SEM data not found: {csv_path}\n"
            "Run 11_sem_zscore_2save_data.py first."
        )

    df = pd.read_csv(csv_path, index_col=0)
    print(f"  Loaded: {df.shape[0]} subjects x {df.shape[1]} variables")
    print(f"  From: {csv_path}")

    if config['group'] != 'all':
        if 'DX' not in df.columns:
            raise KeyError("Column 'DX' not found; cannot filter by group.")
        df = df[df['DX'] == config['group']].copy()
        print(f"  Filtered to DX == '{config['group']}': {df.shape[0]} subjects")
    elif 'DX' in df.columns:
        counts = df['DX'].value_counts()
        print(f"  Group counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    if config['complete_cases']:
        sem_vars = (config['behav_items_srs'] + config['behav_items_nih'] +
                    ['avg_sal', 'avg_cen', 'avg_dmn'])
        missing = [c for c in sem_vars if c not in df.columns]
        if missing:
            raise KeyError(f"Missing SEM columns: {missing}")

        n_before = len(df)
        df = df[df[sem_vars].notna().all(axis=1)].copy()
        n_dropped = n_before - len(df)
        print(f"  Complete cases on all {len(sem_vars)} SEM variables: {len(df)} "
              f"(dropped {n_dropped})")
        if 'DX' in df.columns:
            counts = df['DX'].value_counts()
            print(f"    Group counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    return df


def compute_latent_scores(
    df: pd.DataFrame,
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Add 'autistic_symptom' and 'cognitive_ability' columns to df.

    'mean'         : unit-weighted mean of the already-z-scored indicators.
                     A close approximation of the congeneric factor score when
                     loadings are similar across indicators (they are here).
    'factor_score' : lavaan factor scores read from sem_factor_scores.csv.
    """
    print("\n[2/3] Computing latent-variable scores "
          f"(method = '{config['latent_method']}')...")

    method = config['latent_method']

    if method == 'mean':
        srs_items = config['behav_items_srs']
        nih_items = config['behav_items_nih']

        missing = [c for c in srs_items + nih_items if c not in df.columns]
        if missing:
            raise KeyError(f"Missing behavioral columns: {missing}")

        # Require every indicator to be present, mirroring lavaan's listwise
        # deletion. Without this, a subject missing 2 of 5 SRS items would be
        # averaged over the remaining 3 and silently enter the plot.
        df['autistic_symptom'] = (df[srs_items].mean(axis=1)
                                  .where(df[srs_items].notna().all(axis=1)))
        df['cognitive_ability'] = (df[nih_items].mean(axis=1)
                                   .where(df[nih_items].notna().all(axis=1)))

        print(f"  Autistic symptom  = mean of {len(srs_items)} SRS T-scores")
        print(f"  Cognitive ability = mean of {len(nih_items)} NIH7 percentiles")

    elif method == 'factor_score':
        fs_path = paths['data'] / 'sem_factor_scores.csv'
        if not fs_path.exists():
            raise FileNotFoundError(
                f"Factor scores not found: {fs_path}\n"
                "Export them from 11_sem_zscore_3analysis.R with lavPredict(fit1), "
                "or set CONFIG['latent_method'] = 'mean'."
            )
        fs = pd.read_csv(fs_path, index_col=0)
        for col in ('srs_beh', 'cog_beh'):
            if col not in fs.columns:
                raise KeyError(f"Column '{col}' not found in {fs_path.name}")

        df['autistic_symptom'] = fs['srs_beh'].reindex(df.index)
        df['cognitive_ability'] = fs['cog_beh'].reindex(df.index)

        n_matched = int(df['autistic_symptom'].notna().sum())
        print(f"  Matched factor scores for {n_matched}/{len(df)} subjects")

    else:
        raise ValueError(
            f"Unknown latent_method: {method!r} (expected 'mean' or 'factor_score')"
        )

    return df


# =============================================================================
# PLOTTING
# =============================================================================

def make_severity_cmap(cmap_name: str, floor: float, reverse: bool) -> LinearSegmentedColormap:
    """
    Build a colormap where darker always means 'more severe'.

    floor   : lowest colormap fraction used, so the lightest dot stays visible.
    reverse : True when a LOWER value means more severe (cognitive ability),
              so low values map to the dark end.
    """
    base = plt.get_cmap(cmap_name)
    fractions = np.linspace(floor, 1.0, 256)
    if reverse:
        fractions = fractions[::-1]
    return LinearSegmentedColormap.from_list(f'{cmap_name}_severity', base(fractions))


def plot_dmn_vs_behavior(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    cmap_name: str,
    severe_is_low: bool,
    save_name: str,
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> Tuple[float, float, int]:
    """
    Scatter of avg_dmn (x) vs a behavioral outcome (y), dots colored by severity.

    Returns (pearson_r, p_value, n).
    """
    x_col = 'avg_dmn'

    plot_df = df[[x_col, y_col]].dropna()
    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)
    n = len(plot_df)

    if n < 3:
        raise ValueError(f"Too few complete cases for {y_col}: n = {n}")

    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)

    # Color by the y value, dark = more severe
    cmap = make_severity_cmap(cmap_name, config['color_floor'], reverse=severe_is_low)
    norm = Normalize(np.min(y), np.max(y))
    colors = cmap(norm(y))

    fig, ax = plt.subplots(figsize=config['figsize'])
    sns.regplot(
        x=x, y=y, color='black', ax=ax,
        scatter_kws={
            's': config['scatter_size'],
            'color': colors,
            'alpha': config['scatter_alpha'],
            'edgecolors': config['scatter_edgecolor'],
            'linewidths': config['scatter_edgewidth'],
        },
        line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 3}
    )
    ax.set_xlabel('Alteration of DMN', fontsize=config['label_fontsize'])
    ax.set_ylabel(y_label, fontsize=config['label_fontsize'])
    ax.tick_params(axis='both', which='major', labelsize=config['tick_fontsize'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(config['n_ticks']))
    ax.yaxis.set_major_locator(plt.MaxNLocator(config['n_ticks']))

    if config['show_colorbar']:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(y_label, fontsize=config['label_fontsize'])
        cbar.ax.tick_params(labelsize=config['tick_fontsize'])

    plt.title(f'{y_label}: r={r:.4f}, p={p:.4e} (n={n})',
              fontsize=config['title_fontsize'])
    plt.tight_layout()

    print(f"\n  {y_label} vs. alteration of DMN")
    print(f"    n        = {n}")
    print(f"    Pearson  : r   = {r:+.4f}, p = {p:.4e}")
    print(f"    Spearman : rho = {rho:+.4f}, p = {p_rho:.4e}")

    if config['save_outputs']:
        save_path = paths['fig'] / f'{save_name}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')

        # Keep the exact plotted values alongside the figure
        source_df = plot_df.copy()
        if 'DX' in df.columns:
            source_df['DX'] = df.loc[plot_df.index, 'DX']
        source_df.to_csv(paths['fig'] / f'{save_name}_source_data.csv')

        print(f"    Saved to {save_path}")

    if config['show_plots']:
        plt.show()
    else:
        plt.close(fig)

    return r, p, n


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("11_sem_zscore_4results: DMN Alteration vs. Behavior Scatter Plots")
    print("=" * 70)

    df = load_sem_data(PATHS, CONFIG)
    df = compute_latent_scores(df, PATHS, CONFIG)

    if 'avg_dmn' not in df.columns:
        raise KeyError("Column 'avg_dmn' not found in the SEM data.")

    print("\n[3/3] Drawing scatter plots...")

    results: List[Dict[str, Any]] = []

    # Panel A: DMN -> autistic symptom (Model 1 path d, positive)
    # Higher SRS = more severe -> darker red
    r, p, n = plot_dmn_vs_behavior(
        df=df,
        y_col='autistic_symptom',
        y_label='Autistic symptom',
        cmap_name='Reds',
        severe_is_low=False,
        save_name='scatter_dmn_vs_autistic_symptom',
        paths=PATHS,
        config=CONFIG,
    )
    results.append({'outcome': 'autistic_symptom', 'r': r, 'p': p, 'n': n})

    # Panel B: DMN -> cognitive ability (Model 1 path e, negative)
    # Lower NIH7 = more severe -> darker blue
    r, p, n = plot_dmn_vs_behavior(
        df=df,
        y_col='cognitive_ability',
        y_label='Cognitive ability',
        cmap_name='Blues',
        severe_is_low=True,
        save_name='scatter_dmn_vs_cognitive_ability',
        paths=PATHS,
        config=CONFIG,
    )
    results.append({'outcome': 'cognitive_ability', 'r': r, 'p': p, 'n': n})

    if CONFIG['save_outputs']:
        stats_path = PATHS['fig'] / 'scatter_correlations.csv'
        pd.DataFrame(results).to_csv(stats_path, index=False)
        print(f"\n  Correlation summary saved to {stats_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()
