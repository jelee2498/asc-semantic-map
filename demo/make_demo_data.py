"""
Generate a small simulated dataset for demonstrating the P4 pipeline.

The real inputs to P4 - per-subject encoding weights and the participants
table - are data about individual Healthy Brain Network participants and cannot
be redistributed (see docs/DERIVED_DATA_MANIFEST.md).  This script writes
simulated stand-ins with the same file names, directory layout, shapes and
column schema, so that `pipelines/p4_semantic_axis/07_pca.py` runs end to end
without modification.

Nothing here is derived from participant data.  The only real inputs are two
public files from the Zenodo deposit, shipped in demo/inputs/:
    features_85.csv      the 85 WordNet feature labels and their frequencies
    mask_fdr-0.01.npy    the encoding-performance mask (291 of 360 parcels)

Simulated structure
-------------------
Each subject's weights (4 delays x 85 features, 360 parcels) are generated from
three latent semantic components (feature loadings x parcel maps), scaled by a
haemodynamic delay profile, plus a site offset and Gaussian noise.  For ASC
subjects the first component is weakened in 15 fixed parcels and strengthened in
15 others, so the group comparison has a planted effect to find in both
directions.  The strengthening factor keeps the sum of squares of the first
component's map unchanged: 07_pca.py z-scores each feature across parcels, and a
one-directional effect would change that scaling and shift every other parcel.

    python demo/make_demo_data.py            # writes demo/project/
    python demo/make_demo_data.py --out DIR  # writes DIR instead
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Directory names encode the manuscript configuration (config/params.yml);
# 07_pca.py resolves them literally.
PARTICIPANTS_DIR = Path('2_pipeline/99_main/05_prepare_reg/out')
REGRESSOR_DIR = (PARTICIPANTS_DIR /
                 'wordnet_gaze_weighted_regressor_inc_byhx_docu-True' /
                 'within' / 'bias-0.9_verb-1.0')
ENCODING_DIR = Path('2_pipeline/99_main/06_encoding_model/out/default+me/mmp/'
                    'chunk-9/fold-avg/d-True/k-0/bias-0.9_verb-1.0/wn-True/'
                    'seed-0/s_alpha-True')
RAW_DM_DIR = Path('0_data/raw/movie/DM')

N_PARCELS = 360
N_DELAYS = 4
DELAY_PROFILE = np.array([0.6, 1.0, 0.8, 0.4])   # TR delays 3, 5, 7, 9
N_LATENT = 3
WEIGHT_SCALE = 0.01                              # order of magnitude of real weights


def make_participants(rng: np.random.Generator, n_td: int, n_asd: int) -> pd.DataFrame:
    """Participants table with the columns and codes 07_pca.py filters on."""
    n = n_td + n_asd
    ids = [f'sub-DEMO{i:03d}' for i in range(1, n + 1)]
    dx = ['TD'] * n_td + ['ASD'] * n_asd
    df = pd.DataFrame(index=pd.Index(ids, name='participant_id'))
    df['DX'] = dx
    df['ASD_certainty'] = ['rule-out' if d == 'TD' else 'confirmed' for d in dx]
    df['ASD_document'] = 'no information'
    df['Site'] = rng.permutation(np.resize(['RU', 'CBIC'], n))
    df['Age'] = np.round(rng.uniform(6, 17, n), 2)
    df['Sex'] = rng.choice(['Male', 'Female'], n, p=[0.7, 0.3])
    srs_mean = np.where(np.array(dx) == 'ASD', 68, 49)
    for item in ['AWR', 'COG', 'COM', 'DSMRRB', 'MOT', 'RRB', 'SCI', 'Total']:
        df[f'SRS_{item}_T'] = np.round(rng.normal(srs_mean, 9)).astype(int)
    df['SCQ_Total'] = np.round(rng.normal(np.where(np.array(dx) == 'ASD', 18, 5), 5)).clip(0).astype(int)
    df['CELF_Total'] = np.round(rng.normal(20, 4, n)).astype(int)
    df['condition_ok (movieDM_bold.json && movieDM_bold.nii.gz)'] = 1
    df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)'] = 1
    df['prep_ok (task-movieTP_Atlas_s2_10k.dtseries.nii)'] = 1
    df['Mean_FD_DM'] = np.round(rng.gamma(4, 0.04, n), 4)
    df['Mean_FD_TP'] = np.round(rng.gamma(4, 0.04, n), 4)
    df['Rating_recon_all'] = 5
    df['Remarks'] = 'none'
    df['Rating_deepmreye_movieDM'] = 1.0
    df['Rating_deepmreye_movieTP'] = 1.0
    # Two subjects that the quality-control filters should drop, to show them working.
    df.iloc[1, df.columns.get_loc('Mean_FD_DM')] = 0.62          # motion > 0.5 mm
    df.iloc[n_td + 1, df.columns.get_loc('Remarks')] = 'frontal'  # preprocessing remark
    return df


def make_weights(rng: np.random.Generator, participants: pd.DataFrame,
                 n_features: int, effect_parcels: np.ndarray,
                 effect_direction: np.ndarray, effect_size: float) -> dict:
    """Per-subject weight matrices, (N_DELAYS * n_features, N_PARCELS)."""
    feature_loadings = rng.normal(0, 1, (n_features, N_LATENT))
    parcel_maps = rng.normal(0, 1, (N_LATENT, N_PARCELS))
    parcel_maps[0] *= 2.0                         # a dominant first component
    site_offset = {'RU': rng.normal(0, 0.3, (n_features, N_PARCELS)),
                   'CBIC': rng.normal(0, 0.3, (n_features, N_PARCELS))}

    # ASC scaling of the first component: (1 - effect_size) in the weakened
    # parcels, and in the strengthened ones the factor that restores the sum of
    # squares over all effect parcels.
    weakened = effect_parcels[effect_direction < 0]
    strengthened = effect_parcels[effect_direction > 0]
    ss_weak = np.sum(parcel_maps[0, weakened] ** 2)
    ss_strong = np.sum(parcel_maps[0, strengthened] ** 2)
    gain = np.sqrt(1 + ss_weak * (1 - (1 - effect_size) ** 2) / ss_strong)
    asd_scale = np.ones(N_PARCELS)
    asd_scale[weakened] = 1 - effect_size
    asd_scale[strengthened] = gain

    weights = {}
    for sub_id, row in participants.iterrows():
        maps = parcel_maps * rng.normal(1, 0.1, (N_LATENT, 1))   # individual variation
        if row['DX'] == 'ASD':
            maps = maps.copy()
            maps[0] *= asd_scale
        base = feature_loadings @ maps + site_offset[row['Site']]
        per_delay = [h * base + rng.normal(0, 1.0, base.shape) for h in DELAY_PROFILE]
        weights[sub_id] = (np.vstack(per_delay) * WEIGHT_SCALE).astype(np.float64)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE / 'project',
                        help='Project root to write (default: demo/project).')
    parser.add_argument('--n-td', type=int, default=30)
    parser.add_argument('--n-asd', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    features = pd.read_csv(HERE / 'inputs' / 'features_85.csv')
    labels = list(features['label'])
    perf_mask = np.load(HERE / 'inputs' / 'mask_fdr-0.01.npy')

    # Planted effect: 30 parcels inside the performance mask, the first component
    # weakened in ASC in 15 of them (-1) and strengthened in the other 15 (+1).
    effect_parcels = np.sort(rng.choice(perf_mask, 30, replace=False))
    effect_direction = rng.permutation(np.repeat([-1, 1], 15))

    out = args.out
    for sub in (PARTICIPANTS_DIR, REGRESSOR_DIR, ENCODING_DIR / 'weight', RAW_DM_DIR):
        (out / sub).mkdir(parents=True, exist_ok=True)

    participants = make_participants(rng, args.n_td, args.n_asd)
    participants.to_excel(out / PARTICIPANTS_DIR / 'participants_df_deepmreye_inc_byhx.xlsx')

    # Label frequencies, read by the label-frequency regression step.
    freq_df = pd.DataFrame({'Original Row': range(len(labels)),
                            'Frequency': features['frequency'].values},
                           index=pd.Index(labels))
    freq_df.to_excel(out / RAW_DM_DIR / 'wordnet_sync_regressor_DM.xlsx')

    # Gaze-weighted regressor: 07_pca.py reads only its row labels, from the
    # first subject's file.  Simulated values, 10 TRs, for every subject.
    for sub_id in participants.index:
        reg = pd.DataFrame(rng.random((len(labels), 10)), index=pd.Index(labels),
                           columns=[f'f{i * 19:05d}' for i in range(10)])
        reg.to_excel(out / REGRESSOR_DIR / f'{sub_id}_regressor_DM.xlsx')

    weights = make_weights(rng, participants, len(labels), effect_parcels,
                           effect_direction, effect_size=0.8)
    for sub_id, w in weights.items():
        np.save(out / ENCODING_DIR / 'weight' / f'{sub_id}.npy', w)
    np.save(out / ENCODING_DIR / 'mask_fdr-0.01.npy', perf_mask)
    np.save(out / 'planted_effect_parcels.npy', effect_parcels)
    np.save(out / 'planted_effect_direction.npy', effect_direction)

    print(f'Wrote simulated project to {out}')
    print(f'  {args.n_td} TD + {args.n_asd} ASC subjects '
          f'(2 fail quality control by design), {len(labels)} features, '
          f'{N_PARCELS} parcels, {len(perf_mask)} in the performance mask')
    print(f'  planted ASC effect in {len(effect_parcels)} parcels, '
          f'{np.sum(effect_direction < 0)} weakened and {np.sum(effect_direction > 0)} strengthened '
          f'(planted_effect_parcels.npy, planted_effect_direction.npy)')


if __name__ == '__main__':
    main()
