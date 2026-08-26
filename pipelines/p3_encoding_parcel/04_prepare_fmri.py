"""
Prepare fMRI data (parcellated + SRM-aligned) for parameter search / encoding.

Refactored from: 99_main/backup_251230/04_prepare_fmri.py
  (which itself was @ Original code from
   17_param_search/00_prepare_data_0fmri.py)

This script performs, for each diagnostic group (TD, ASD):
1. Load the screened demographic table and DeepMReye QC; intersect to the
   final subject list (FD < 0.5 for both movies & DeepMReye available)
2. (posix only) Load movieDM CIFTI (s3 10k), confound-clean ('default+me'),
   parcellate to the MMP atlas
3. Split DM into 9 chunks merged into 3 cross-validation folds
4. Per fold, save:
     - k-0   : parcellated (non-SRM) train/test fMRI
     - k-{srm}: DetSRM-aligned (n_iter=10) train/test shared fMRI

================================================================================
OUTPUT PATH PRESERVATION  (do not change - downstream scripts 05- are frozen)
================================================================================
The refactored code writes to the EXACT same location as the original
backup_251230/04_prepare_fmri.py:

    {project}/2_pipeline/17_param_search/
        00_prepare_data/out/{prep_option}/{conf_option}/{atlas}/
        chunk-{chunk}/fold-{i}/srm_dx-{dx}/shared_k-{0|srm}/{train|test}/
        {sub_id}.npy

!!  IMPORTANT DISCREPANCY (surfaced, intentionally NOT changed)  !!
    The downstream 06_encoding_model.py CONFIG reads fMRI from
        17_param_search / '00_prepare_data_release11' / out / ...
    whereas this script (faithful to the original) writes to
        17_param_search / '00_prepare_data'      / out / ...
    Per the instruction "save outputs in the same path as the original and do
    NOT modify downstream 05- scripts", the ORIGINAL '00_prepare_data' task
    folder is preserved here verbatim. If a single shared location is desired,
    that is a deliberate decision for the maintainer to make - this refactor
    does not silently retarget it.

Other fidelity notes:
  * pipeline/task kept as '17_param_search' / '00_prepare_data' (NOT
    '99_main') so the out/ path is byte-identical to the original.
  * Storage roots now come from config/paths.yml (lib/project_config.py); the
    original hard-coded roots (Windows S:/ | Linux /store7) are not distributed.
    Under the default config the resolved paths are unchanged.
  * clean_fmri / parcel_fmri / load_confounds / make_poly_regressors are
    ported verbatim (behaviour-preserving) from the original functions.py so
    the saved arrays are numerically identical. Standalone - no
    `from functions import *`.
  * The heavy fMRI section runs only on posix (os.name == 'posix'), exactly
    as the original; on Windows it prints the cohort summary and skips.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import platform
from glob import glob
from pathlib import Path
from itertools import chain
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.polynomial.legendre import Legendre
import scipy.linalg as la

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, TEMPLATES  # noqa: E402

import nibabel as nib

# brainiak (DetSRM) is only importable / used on the HPC (posix), like 08_*
if os.name == 'posix':
    from brainiak.funcalign.srm import DetSRM


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG: Dict[str, Any] = {
    'project': '02_asd_semantic_map',

    # NOTE: kept identical to original to preserve the out/ output path.
    # (See module docstring re: 06_encoding_model reading '00_prepare_data_release11'.)
    'pipeline': '17_param_search',
    'task': '00_prepare_data',

    # Source preprocessing (re-ciftified) data
    'prep_pipeline': '00_preprocess_fmri',
    'prep_task': '22_fix_dataset',
    'prep_option': 'smooth-3',

    # Demographic / DeepMReye QC sources
    'demo_pipeline': '00_preprocess_fmri',
    'demo_task': '22_fix_dataset',
    'demo_file': 'demo.xlsx',
    'demo_sheet': 'screened',
    'deepmreye_pipeline': '07_deepmreye',
    'deepmreye_task': '05_apply_pretrained',
    'deepmreye_file': 'quality_control_deepmreye_added.xlsx',

    # Analysis parameters (preserved from original)
    'conf_option': 'default+me',
    'atlas': 'mmp',
    'chunk_option': 9,
    'srm_option': 50,        # k in SRM (k-0 also saved)
    'srm_n_iter': 10,        # DetSRM iterations (original used 10)
    'start_ignore_trs': 5,
    'end_ignore_trs': 5,
    'fd_thres': 0.5,

    'dx_list': ['TD', 'ASD'],
}


# =============================================================================
# PATH SETUP
# =============================================================================

def setup_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Construct project paths, preserving the ORIGINAL store roots."""
    # Paths come from config/paths.yml via lib/project_config.py.  The original
    # working tree hard-coded a storage root here; that is machine-specific and
    # is not distributed.
    proj_root = PROJECT

    proj = config['project']
    pipe = config['pipeline']
    task = config['task']

    pipe_path = proj_root / '2_pipeline' / pipe
    proc_path = pipe_path / task

    prep_path = (
        proj_root / '2_pipeline' /
        config['prep_pipeline'] / config['prep_task'] / 'out' /
        config['prep_option'] / 'mri'
    )
    demo_file = (
        proj_root / '2_pipeline' /
        config['demo_pipeline'] / config['demo_task'] / 'out' /
        config['demo_file']
    )
    deepmreye_file = (
        proj_root / '2_pipeline' /
        config['deepmreye_pipeline'] / config['deepmreye_task'] / 'out' /
        config['deepmreye_file']
    )

    paths = {
        'raw': proj_root / '0_data' / 'raw',
        'tpl': TEMPLATES,
        'pipe': pipe_path,
        'proc': proc_path,
        'out': proc_path / 'out',
        'save': proc_path / 'save',
        'tmp': proc_path / 'tmp',
        'prep': prep_path,
        'demo_file': demo_file,
        'deepmreye_file': deepmreye_file,
    }

    return paths


# =============================================================================
# UTILITY FUNCTIONS  (ported verbatim from the original functions.py)
# =============================================================================

# Column-wise z-score (original `zs` lambda)
zs = lambda v: (v - v.mean(0)) / v.std(0)
zs.__doc__ = """Z-scores (standardizes) each column of [v]."""


def load_brain_mask(tpl_path: Path) -> np.ndarray:
    """Load the 10k non-medial-wall vertex mask (neuromaps-data)."""
    non_med_l = nib.load(str(
        tpl_path / 'neuromaps-data' / 'atlases' / 'fsLR' /
        'tpl-fsLR_den-10k_hemi-L_desc-nomedialwall_dparc.label.gii'
    )).darrays[0].data
    non_med_r = nib.load(str(
        tpl_path / 'neuromaps-data' / 'atlases' / 'fsLR' /
        'tpl-fsLR_den-10k_hemi-R_desc-nomedialwall_dparc.label.gii'
    )).darrays[0].data
    return np.r_[non_med_l, non_med_r].nonzero()[0]


def make_poly_regressors(n_volume: int, order: int = 2) -> np.ndarray:
    """Legendre polynomial detrend regressors (verbatim from functions.py)."""
    X = np.ones((n_volume, 1))
    for d in range(order):
        poly = Legendre.basis(d + 1)
        poly_trend = poly(np.linspace(-1, 1, n_volume))
        X = np.hstack((X, poly_trend[:, None]))
    X = X[:, 1:]  # remove bias regressor
    return X


def load_confounds(prep_path: Path, sub_id: str, movie: str) -> pd.DataFrame:
    """Load the fMRIprep confounds TSV for a subject/movie."""
    confounds_file = (
        prep_path / sub_id / 'fmriprep' / sub_id / 'func' /
        f'{sub_id}_task-movie{movie}_desc-confounds_regressors.tsv'
    )
    return pd.read_csv(confounds_file, sep='\t')


def clean_fmri(
    fmri: np.ndarray,
    confounds_df: pd.DataFrame,
    option: str,
    raw_path: Path,
    start_ignore_trs: int = 0,
    end_ignore_trs: int = 0,
) -> np.ndarray:
    """
    Confound-clean fMRI - ported verbatim from the original functions.py.

    Only the branches reachable in this pipeline are implemented:
    'original', 'default', 'default+me' (04 uses 'default+me' on movieDM).
    `raw_path` replaces the original module-global used for the ME table so
    this script stays standalone (numerically identical behaviour).
    """
    if option == 'original':
        return fmri[start_ignore_trs:, :]

    elif option == 'default':
        # FD, global signal, linear trend
        confounds_df = confounds_df[['framewise_displacement', 'global_signal']]
        X = confounds_df.values
        X = np.hstack((X, make_poly_regressors(len(fmri), order=1)))
        X = X[start_ignore_trs:, :]
        X = zs(X)
        X[np.isnan(X)] = 0.

        fmri = fmri[start_ignore_trs:, :]
        fmri_mean = fmri.mean(0)
        fmri = fmri - fmri_mean
        coef, _, _, _ = la.lstsq(X, fmri)
        return fmri - X.dot(coef) + fmri_mean

    elif option == 'default+me':
        # FD, global signal, linear trend + motion energy feature
        confounds_df = confounds_df[['framewise_displacement', 'global_signal']]

        if confounds_df.shape[0] == 750:
            me_df = pd.read_excel(
                raw_path / 'movie' / 'DM' / 'DM_motion_energy.xlsx',
                sheet_name='mean', index_col=0, engine='openpyxl'
            )
        elif confounds_df.shape[0] == 250:
            me_df = pd.read_excel(
                raw_path / 'movie' / 'TP' / 'TP_motion_energy.xlsx',
                sheet_name='mean', index_col=0, engine='openpyxl'
            )
        else:
            raise Exception(f'Unexpected confounds length: {confounds_df.shape[0]}')

        confounds_df = confounds_df.copy()
        confounds_df['motion_energy_feature'] = me_df.values.flatten().tolist()

        X = confounds_df.values
        X = np.hstack((X, make_poly_regressors(len(fmri), order=1)))
        X = X[start_ignore_trs:-end_ignore_trs, :]
        X = zs(X)
        X[np.isnan(X)] = 0.

        fmri = fmri[start_ignore_trs:-end_ignore_trs, :]
        fmri_mean = fmri.mean(0)
        fmri = fmri - fmri_mean
        coef, _, _, _ = la.lstsq(X, fmri)
        return fmri - X.dot(coef) + fmri_mean

    else:
        raise Exception(f"Unsupported clean_fmri option in 04: '{option}'")


def parcel_fmri(
    fmri: np.ndarray,
    label: np.ndarray,
    brain_mask: np.ndarray,
    symmetric_half_label: bool = False,
) -> np.ndarray:
    """
    Parcellate surface fMRI into the given atlas - ported verbatim from the
    original functions.py (10k, non-32k, surface branch used by this script).

    Args:
        fmri: (n_volume, n_vertices) surface fMRI; n_vertices may equal
              len(brain_mask) (expanded to 20484) or 20484 already
        label: 10k atlas label array (values 1..n_parcels)
        brain_mask: 10k non-medial-wall vertex indices
        symmetric_half_label: original flag (False here)

    Returns:
        (n_volume, n_parcels) parcellated fMRI
    """
    if len(fmri.shape) >= 3:
        raise Exception('parcel_fmri: only surface fMRI supported here')

    fmri_full = np.zeros((fmri.shape[0], 20484))
    fmri_full[:, brain_mask] = fmri

    fmri_parcel = np.zeros((fmri.shape[0], label.max()))

    brain_mask_full = np.zeros(20484).astype('int')
    brain_mask_full[brain_mask] = 1  # fine-tuned whole-brain mask

    for roi_idx in range(label.max()):
        if symmetric_half_label:
            roi_mask = (list(np.argwhere(label == roi_idx + 1).squeeze()) +
                        list(np.argwhere(label == roi_idx + 1).squeeze() + 10242))
        else:
            roi_mask = list(np.argwhere(label == roi_idx + 1).squeeze())
        roi_mask = np.array(roi_mask)

        roi_mask_full = np.zeros(20484).astype('int')
        try:
            roi_mask_full[roi_mask] = 1
        except IndexError:
            print('Something wrong with ROI_MASK')

        roi_mask_full_ft = np.multiply(brain_mask_full, roi_mask_full)
        try:
            fmri_parcel[:, roi_idx] = fmri_full[
                :, np.argwhere(roi_mask_full_ft == 1).squeeze()
            ].mean(1)
        except np.AxisError:  # only one vertex assigned as ROI
            fmri_parcel[:, roi_idx] = fmri_full[
                :, np.argwhere(roi_mask_full_ft == 1).squeeze()
            ]

    return fmri_parcel


def create_cv_folds(chunk_option: int) -> List[List[int]]:
    """Build the merged CV test-fold index lists (verbatim chunk boundaries)."""
    if chunk_option == 9:
        dm_chunk_list = [
            np.linspace(0, 81, 82).astype(np.int16),
            np.linspace(82, 163, 82).astype(np.int16),
            np.linspace(164, 246, 83).astype(np.int16),
            np.linspace(247, 328, 82).astype(np.int16),
            np.linspace(329, 410, 82).astype(np.int16),
            np.linspace(411, 493, 83).astype(np.int16),
            np.linspace(494, 575, 82).astype(np.int16),
            np.linspace(576, 657, 82).astype(np.int16),
            np.linspace(658, 739, 82).astype(np.int16),
        ]
    elif chunk_option == 16:
        dm_chunk_list = [
            np.linspace(0, 45, 46).astype(np.int16),
            np.linspace(46, 91, 46).astype(np.int16),
            np.linspace(92, 137, 46).astype(np.int16),
            np.linspace(138, 184, 47).astype(np.int16),
            np.linspace(185, 230, 46).astype(np.int16),
            np.linspace(231, 276, 46).astype(np.int16),
            np.linspace(277, 322, 46).astype(np.int16),
            np.linspace(323, 369, 47).astype(np.int16),
            np.linspace(370, 415, 46).astype(np.int16),
            np.linspace(416, 461, 46).astype(np.int16),
            np.linspace(462, 507, 46).astype(np.int16),
            np.linspace(508, 554, 47).astype(np.int16),
            np.linspace(555, 600, 46).astype(np.int16),
            np.linspace(601, 646, 46).astype(np.int16),
            np.linspace(647, 692, 46).astype(np.int16),
            np.linspace(693, 739, 47).astype(np.int16),
        ]
    else:
        raise Exception("None of the possible options are matched with given cv option!")

    n_merge = np.sqrt(chunk_option).astype(int)
    test_fold_list = []
    for i in range(n_merge):
        merge_chunk_ids = [i + n_merge * ii for ii in range(n_merge)]
        dm_fold_list = [dm_chunk_list[idx] for idx in merge_chunk_ids]
        test_fold_list.append(list(chain(*dm_fold_list)))

    return test_fold_list


# =============================================================================
# COHORT SELECTION
# =============================================================================

def select_cohort(
    dx: str,
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> tuple:
    """
    Select the final subject list for a diagnostic group.

    Filters: DX == dx, FD_TP & FD_DM < fd_thres, intersect with
    DeepMReye-available subjects.

    Returns:
        Tuple of (id_list, demo_df)
    """
    print('* Load ID list of preprocessed data')

    demo_df = pd.read_excel(
        paths['demo_file'], sheet_name=config['demo_sheet'],
        index_col=0, engine='openpyxl'
    )

    dx_filt = (demo_df['DX'] == dx).values if dx != 'all' else np.ones(len(demo_df)) > 0
    fd_thres = config['fd_thres']
    fd_filt = np.multiply(
        (demo_df['Mean_FD_TP'] < fd_thres),
        (demo_df['Mean_FD_DM'] < fd_thres)
    )
    cohort_demo_df = demo_df[np.multiply(dx_filt, fd_filt)]
    id_list_prep = list(cohort_demo_df.index)

    print('* Load ID list of DeepMReye processed data')
    qc_df = pd.read_excel(paths['deepmreye_file'], index_col=0, engine='openpyxl')
    id_list_eye = list(qc_df[qc_df['DeepMReye_available'] == 1].index)

    id_list = list(set(id_list_prep).intersection(set(id_list_eye)))

    print(f'Participants information (n={len(id_list)}):')
    sel = demo_df.loc[id_list]
    print(' - DX')
    print(f"   TD: {len(sel[sel['DX']=='TD'])}, ASD: {len(sel[sel['DX']=='ASD'])}")
    print(' - Site')
    print(f"   CBIC: {len(sel[sel['Site']=='CBIC'])}, RU: {len(sel[sel['Site']=='RU'])}")
    print(' - Sex')
    print(f"   Male: {len(sel[sel['Sex']=='Male'])}, Female: {len(sel[sel['Sex']=='Female'])}")

    return id_list, demo_df


def load_atlas(atlas: str, tpl_path: Path) -> np.ndarray:
    """Load the 10k parcellation atlas label array."""
    if atlas == 'mmp':
        return nib.load(str(
            tpl_path / 'MMP' / '10k_fs_LR' /
            'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.10k_fs_LR.dlabel.nii'
        )).get_fdata().squeeze().astype(np.int32)
    elif atlas == 'sch-400':
        return nib.load(str(
            tpl_path / 'schaefer' / 'Parcellations' / 'HCP' / 'fslr10k' / 'cifti' /
            'Schaefer2018_400Parcels_17Networks_order.dlabel.nii'
        )).get_fdata().astype(np.int32).squeeze()
    elif atlas == 'sch-800':
        return nib.load(str(
            tpl_path / 'schaefer' / 'Parcellations' / 'HCP' / 'fslr10k' / 'cifti' /
            'Schaefer2018_800Parcels_17Networks_order.dlabel.nii'
        )).get_fdata().astype(np.int32).squeeze()
    else:
        raise Exception("None of the possible options are matched with given atlas!")


# =============================================================================
# SAVE SRM-ALIGNED FMRI
# =============================================================================

def process_and_save_dx(
    dx: str,
    id_list: List[str],
    paths: Dict[str, Path],
    config: Dict[str, Any]
) -> None:
    """
    Load/clean/parcellate movieDM, build CV folds, save k-0 and DetSRM fMRI.

    Output paths are byte-identical to the original (see module docstring).
    """
    prep_path = paths['prep']
    raw_path = paths['raw']
    out_path = paths['out']
    conf_option = config['conf_option']
    atlas = config['atlas']
    chunk_option = config['chunk_option']
    srm_option = config['srm_option']
    prep_option = config['prep_option']

    brain_mask = load_brain_mask(paths['tpl'])

    print('* Load DM data')
    fmri_dm_list = []
    for sub_id in tqdm(id_list, desc='MovieDM'):
        fmri_path = glob(str(
            prep_path / sub_id / 'ciftify' / 'sub*' / 'MNINonLinear' /
            'Results' / 'task-movieDM' / 'task-movieDM_Atlas_s3_10k.dtseries.nii'
        ))[0]
        fmri = nib.load(fmri_path).get_fdata()
        fmri_clean = clean_fmri(
            fmri[:, brain_mask],
            load_confounds(prep_path, sub_id, 'DM'),
            conf_option,
            raw_path,
            start_ignore_trs=config['start_ignore_trs'],
            end_ignore_trs=config['end_ignore_trs'],
        )
        fmri_dm_list.append(fmri_clean.T)

    label_atlas = load_atlas(atlas, paths['tpl'])

    fmri_dm_parcel_list = []
    for fmri_dm in tqdm(fmri_dm_list, desc='Parcellate fMRI'):
        fmri_dm_parcel_list.append(
            parcel_fmri(fmri_dm.T, label_atlas, brain_mask,
                        symmetric_half_label=False).T
        )

    n_dm = len(fmri_dm_parcel_list[0].T)

    print('* Split DM and calculate shared response')
    test_fold_list = create_cv_folds(chunk_option)

    base = (out_path / prep_option / conf_option / atlas /
            f'chunk-{chunk_option}')
    os.makedirs(out_path / prep_option / conf_option, exist_ok=True)

    for idx, test_fold_ids in tqdm(enumerate(test_fold_list),
                                   desc='Save shared fMRI for each fold'):
        train_fold_ids = np.array(list(set(range(n_dm)) - set(list(test_fold_ids))))

        fmri_train_list, fmri_test_list = [], []
        for fmri_dm_parcel in fmri_dm_parcel_list:
            fmri_train_list.append(fmri_dm_parcel[:, train_fold_ids])
            fmri_test_list.append(fmri_dm_parcel[:, test_fold_ids])

        fold_dir = base / f'fold-{idx + 1}' / f'srm_dx-{dx}'

        # --- k-0: parcellated (non-SRM) fMRI ---
        for split, data_list in [('train', fmri_train_list), ('test', fmri_test_list)]:
            d = fold_dir / 'shared_k-0' / split
            os.makedirs(d, exist_ok=True)
            for sub_id, arr in zip(id_list, data_list):
                np.save(d / f'{sub_id}.npy', arr)

        # --- k-{srm}: DetSRM-aligned shared fMRI (separate fit per split) ---
        for split, data_list in [('train', fmri_train_list), ('test', fmri_test_list)]:
            d = fold_dir / f'shared_k-{srm_option}' / split
            os.makedirs(d, exist_ok=True)
            detsrm = DetSRM(n_iter=config['srm_n_iter'], features=srm_option)
            detsrm.fit(data_list)
            for sub_id, w in zip(id_list, detsrm.w_):
                np.save(d / f'{sub_id}.npy', w.dot(detsrm.s_))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution flow for fMRI preparation (per diagnostic group)."""
    print('=' * 70)
    print('04_prepare_fmri.py - parcellate + SRM-align movieDM fMRI')
    print('=' * 70)

    paths = setup_paths(CONFIG)

    for dx in CONFIG['dx_list']:
        print(f'\n=== {dx} ===')
        id_list, _ = select_cohort(dx, paths, CONFIG)

        # Heavy fMRI processing only on posix (HPC), exactly as the original.
        if os.name == 'posix':
            process_and_save_dx(dx, id_list, paths, CONFIG)
        else:
            print('  (Skipping fMRI load/SRM - only runs on Linux/HPC, '
                  'matching the original os.name == "posix" guard)')

    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)


if __name__ == '__main__':
    main()
