"""
Prepare WordNet-enhanced gaze-weighted regressors for encoding model analysis.

This script processes gaze-weighted regressors by:
1. Loading movie frame data and regressor information
2. Filtering participants based on QC criteria (FD, deepmreye, preprocessing)
3. Adding superordinate labels from WordNet hierarchy
4. Merging overlapping labels using nanmean aggregation
5. Pruning unnecessary labels
6. Saving per-subject Excel files with WordNet-enhanced regressors

Processing grid:
    - 10 gaze weight combinations (bias_weight, verb_weight)

Dependencies:
    - NLTK WordNet (wordnet31)
    - pandas, numpy
    - openpyxl (for Excel I/O)
    - tqdm (progress bars)

Outputs:
    Excel files per subject with gaze-weighted WordNet regressors:
    {out_path}/wordnet_gaze_weighted_regressor_inc_byhx_docu-{hx_docu}/{setting}/bias-{bias}_verb-{verb}/{sub_id}_regressor_{movie}.xlsx

Original script: 17_param_search/00_prepare_data_release11_4wordnet_gaze_weight_inc_byhx.py
Refactored: 2026-01 for reproducibility and readability
"""

import os
from pathlib import Path
from typing import Tuple, List
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet31 as wn
from nltk.corpus.reader.wordnet import WordNetError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import PROJECT, RAW, CODE, PIPELINE, TEMPLATES  # noqa: E402


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Project metadata
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '05_prepare_reg',

    # Base paths come from config/paths.yml (see below).  The original working
    # tree hard-coded a storage root here (S:/ on Windows, /MIPL/store7 on
    # Linux); that is machine-specific and is not distributed.

    # Analysis parameters
    'fd_thres': 0.5,  # FD threshold for motion filtering
    'hx_docu': 'True',  # Include only ASD subjects with documentation when by-history

    # # Gaze weight grid (bias_weight, verb_weight)
    # 'gaze_weight_grid': [
    #     (0.1, 0.5),
    #     (0.1, 0.55),
    #     (0.1, 0.1),
    #     (0.1, 1.0),
    #     (0.3, 0.5),
    #     (0.3, 0.65),
    #     (0.3, 0.3),
    #     (0.3, 1.0),
    #     (0.5, 0.5),  # (0.5, 0.5) is overlapping so only once
    #     (0.5, 0.75),
    #     (0.5, 1.0),
    #     (0.7, 0.5),
    #     (0.7, 0.85),
    #     (0.7, 0.7),
    #     (0.7, 1.0),
    #     (0.9, 0.5),
    #     (0.9, 0.95),
    #     (0.9, 0.9),
    #     (0.9, 1.0)
    # ],
    # Gaze weight grid (bias_weight, verb_weight) -> temporarily only (0.9, 1.0) for movieTP
    'gaze_weight_grid': [
        (0.9, 1.0)
    ],

    # Movie settings
    'movie': 'TP',  # Despicable Me
    'setting': 'within',  # 'within' or 'cross' movie encoding

    # Filtering options
    'dx': 'all',  # Diagnosis filter: 'TD', 'ASD', or 'all'
}

# Derived paths
project = CONFIG['project']
pipeline = CONFIG['pipeline']
task = CONFIG['task']

raw_path = RAW
code_path = CODE
pipe_path = PIPELINE
proc_path = pipe_path / task
out_path = proc_path / 'out'
save_path = proc_path / 'save'
tmp_path = proc_path / 'tmp'
tpl_path = TEMPLATES

# Regressor option based on setting
if CONFIG['setting'] == 'cross':
    CONFIG['reg_option'] = 'only_41+behav'
elif CONFIG['setting'] == 'within':
    CONFIG['reg_option'] = 'all (wordnet)'
else:
    raise ValueError(f"Invalid setting: {CONFIG['setting']}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_movie_data(config: dict) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Load movie frame data and regressor information.

    Args:
        config: Configuration dictionary containing movie and path settings

    Returns:
        Tuple of:
            - reg_df: DataFrame with regressor values per frame
            - label_list: List of semantic labels
            - frame_list: List of frame identifiers (e.g., 'f0001')
            - file_list: List of image file paths for each frame
    """
    movie = config['movie']
    reg_option = config['reg_option']

    print('* Load movie frames and time information')

    if movie == 'TP':
        reg_file = raw_path / 'movie' / movie / f'prepend_sync_regressor_{movie}.xlsx'
        reg_df = pd.read_excel(
            reg_file,
            sheet_name=reg_option,
            index_col='Row',
            engine='openpyxl'
        )

        # Find exactly 7 prepended frames (columns containing 'f')
        # and track where they end in the column list
        frame_no_list_prepend = []
        after_prepended_frame_idx = 0
        for frame_no in list(reg_df.columns):
            if len(frame_no_list_prepend) == 7:
                break
            else:
                after_prepended_frame_idx += 1
                if len(frame_no_list_prepend) < 7 and 'f' in frame_no:
                    frame_no_list_prepend.append(frame_no.strip('f'))

        # Identify overlapping frames (marked with .1 suffix) after prepend frames
        overlap_ids = [
            i for i, frame_no in enumerate(
                list(reg_df.columns)[after_prepended_frame_idx:]
            )
            if '.1' in frame_no
        ]

        # Renumber overlapping frames (offset by 131)
        frame_no_list_overlap = [
            str(int(frame_no.strip('f').strip('.1')) + 131).zfill(4)
            for i, frame_no in enumerate(
                list(reg_df.columns)[after_prepended_frame_idx:]
            )
            if i in overlap_ids
        ]

        # Non-overlapping frames also need offset by 131
        frame_no_list_nonoverlap = [
            str(int(frame_no.strip('f')) + 131).zfill(4)
            for i, frame_no in enumerate(
                list(reg_df.columns)[after_prepended_frame_idx:]
            )
            if i not in overlap_ids
        ]

        # Combine and sort all frame numbers
        frame_no_list = sorted(
            frame_no_list_prepend + frame_no_list_overlap + frame_no_list_nonoverlap
        )

        label_list = list(reg_df.index)
        frame_list = [f'f{frame_no}' for frame_no in frame_no_list]

        # Build file paths from prepend_sync_frames directory
        file_list = [
            glob(str(raw_path / 'movie' / movie / f'prepend_sync_frames/present_{frame_no}.jpg'))[0]
            for frame_no in frame_no_list
        ]

    else:
        reg_file = raw_path / 'movie' / movie / f'sync_regressor_{movie}.xlsx'
        reg_df = pd.read_excel(
            reg_file,
            sheet_name=reg_option,
            index_col=0,
            engine='openpyxl'
        )

        label_list = list(reg_df.index)
        frame_list = [col for col in reg_df.columns if 'f' in col]

        # Find image files for each frame (exclude 'allframes' folder)
        file_list = []
        for frame_no in frame_list:
            frame_num = frame_no.strip('f')
            pattern = str(raw_path / 'movie' / movie / f'?[!l][!l]*' / f'despicable_me_{frame_num}.jpg')
            matches = glob(pattern)
            if matches:
                file_list.append(matches[0])
            else:
                raise FileNotFoundError(f"Image file not found for frame {frame_no}")

    return reg_df, label_list, frame_list, file_list


def load_and_filter_participants(config: dict) -> Tuple[List[str], pd.DataFrame]:
    """
    Load participants dataframe and apply QC filters.

    Applies the following filters sequentially:
        1. Diagnosis (TD, ASD, or all)
        2. ASD documentation (if hx_docu='True')
        3. Preprocessing completion
        4. No remarks during preprocessing
        5. Framewise displacement (FD) threshold
        6. DeepMReye quality rating

    Args:
        config: Configuration dictionary with filtering parameters

    Returns:
        Tuple of:
            - id_list: List of subject IDs passing all filters
            - participants_df: DataFrame with participant metadata
    """
    print('* Load ID list of preprocessed data')

    dx = config['dx']
    fd_thres = config['fd_thres']
    hx_docu = config['hx_docu']
    movie = config['movie']

    participants_file = out_path / 'participants_df_deepmreye_inc_byhx.xlsx'
    participants_df = pd.read_excel(participants_file, index_col=0, engine='openpyxl')

    # Filter 1: Diagnosis
    if dx == 'TD':
        dx_filt = (participants_df['DX'] == 'TD').values
    elif dx == 'ASD':
        dx_filt = (participants_df['DX'] == 'ASD').values
    elif dx == 'all':
        dx_filt = np.ones(len(participants_df), dtype=bool)
    else:
        raise ValueError(f"Invalid dx option: {dx}")

    # Filter 2: ASD documentation for by-history subjects
    if hx_docu == 'True':
        for sub_id in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id]['ASD_document'] != 'documentation provided'):
                print(f"Exclude {sub_id} due to 'ASD_certainty' is 'by-history' "
                      f"and 'ASD_document' is not 'documentation provided'")
                dx_filt[participants_df.index == sub_id] = False

    # Filter 3: Preprocessing completion
    prep_filt = (participants_df[f'prep_ok (task-movie{movie}_Atlas_s2_10k.dtseries.nii)'] == 1).values

    # Filter 4: No remarks during preprocessing
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Apply first 4 filters
    id_list_dx_prep_remark = list(participants_df[dx_filt & prep_filt & no_remarks_filt].index)
    participants_df_dx_prep_remark = participants_df.loc[id_list_dx_prep_remark]

    # Filter 5: Framewise displacement
    fd_filt = (participants_df_dx_prep_remark[f'Mean_FD_{movie}'] < fd_thres).values

    # Filter 6: DeepMReye quality
    deepmreye_filt = (participants_df_dx_prep_remark[f'Rating_deepmreye_movie{movie}'] == 1).values

    # Final filtered list
    id_list = list(participants_df_dx_prep_remark[fd_filt & deepmreye_filt].index)

    # Print summary statistics
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

    return id_list, participants_df


def add_wordnet_hypernyms(
    weight_reg_df: pd.DataFrame,
    label_list: List[str],
    frame_list: List[str]
) -> pd.DataFrame:
    """
    Add superordinate labels from WordNet hierarchy.

    For each label in label_list, finds its hypernym path in WordNet
    and adds all hypernyms to the DataFrame. Uses only the first
    hypernym path when multiple paths exist.

    Args:
        weight_reg_df: Gaze-weighted regressor DataFrame (labels x frames)
        label_list: Original list of semantic labels
        frame_list: List of frame column identifiers

    Returns:
        DataFrame with original labels plus all hypernym labels added
    """
    temp_reg_df = weight_reg_df.copy()

    for label in label_list:
        try:
            net = wn.synset(label)
            hyper_label_paths = net.hypernym_paths()

            # Use only first hypernym path
            hyper_label = hyper_label_paths[0]
            for hyper in hyper_label[:-1]:  # Exclude the label itself (last element)
                series_label = weight_reg_df.loc[label].copy()
                series_label = series_label.rename(hyper.name())
                temp_reg_df = pd.concat([temp_reg_df, pd.DataFrame(series_label).T])

        except WordNetError:
            print(f'{label} <- not found in WordNet')

    return temp_reg_df


def merge_overlapping_labels(
    temp_reg_df: pd.DataFrame,
    label_list: List[str],
    frame_list: List[str]
) -> pd.DataFrame:
    """
    Merge overlapping superordinate labels using nanmean aggregation.

    When multiple subordinate labels share the same superordinate label,
    this function averages their values (ignoring zeros/NaN) to create
    a single merged representation.

    Args:
        temp_reg_df: DataFrame with original + hypernym labels
        label_list: Original semantic label list
        frame_list: List of frame column identifiers

    Returns:
        DataFrame with merged labels (overlapping labels averaged)
    """
    all_labels = list(temp_reg_df.index)
    superordinate_labels = all_labels.copy()
    del superordinate_labels[0:len(label_list)]  # Remove original labels

    wordnet_reg_df = temp_reg_df.iloc[:len(label_list)].copy()

    for label in all_labels:
        cnt_label = superordinate_labels.count(label)

        if cnt_label > 1:  # Only merge labels with multiple subordinates
            df_label = temp_reg_df.loc[label][frame_list]
            array_label = df_label.values.copy()

            # Convert zeros to NaN to ignore in averaging
            array_label[array_label == 0] = np.nan

            # Convert NaN back to zero if all elements in column are NaN
            for i in range(array_label.shape[1]):
                if np.isnan(array_label[:, i]).all():
                    array_label[:, i] = 0

            # Merge using nanmean (ignoring NaN values)
            merge_label = np.nanmean(array_label, axis=0)

            series_label = pd.Series(merge_label, index=frame_list, name=label)

            # Add or update merged label
            if label in label_list:  # Superordinate label already in original labels
                wordnet_reg_df.loc[label, frame_list] = list(merge_label)
            else:
                wordnet_reg_df = pd.concat([wordnet_reg_df, pd.DataFrame(series_label).T])

            # Remove all instances of this label from superordinate_labels
            while label in superordinate_labels:
                superordinate_labels.remove(label)

    return wordnet_reg_df


def prune_labels(wordnet_reg_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Remove labels not present in reference WordNet regressor.

    This ensures consistency with the canonical WordNet hierarchy
    used in the analysis by removing any spurious labels.

    Args:
        wordnet_reg_df: DataFrame with WordNet-enhanced labels
        config: Configuration dictionary with movie settings

    Returns:
        Pruned DataFrame with only canonical WordNet labels
    """
    movie = config['movie']

    wordnet_label_list = list(wordnet_reg_df.index)
    if 'Time (sec)' in wordnet_label_list:
        wordnet_label_list.remove('Time (sec)')

    # Load reference WordNet regressor (without gaze weighting) for canonical label set.
    # NOTE: this reads from the legacy `25_sensitivity` tree, not from 99_main -
    # a cross-tree dependency carried over from the original working directory.
    ref_file = (PROJECT / '2_pipeline' / '25_sensitivity' /
                '04_movietp' / 'out' / f'regressor_worndet_no_gaze_{movie}.xlsx')
    wordnet_reg_nogaze_df = pd.read_excel(ref_file, index_col=0, engine='openpyxl')

    # Remove labels not in reference
    for label in wordnet_label_list:
        if label not in list(wordnet_reg_nogaze_df.index):
            wordnet_reg_df = wordnet_reg_df.drop([label])

    return wordnet_reg_df


def process_subject_gaze_weights(
    sub_id: str,
    config: dict,
    label_list: List[str],
    frame_list: List[str],
    bias_weight: float,
    verb_weight: float
) -> None:
    """
    Process gaze-weighted regressors for a single subject.

    Loads subject-specific gaze-weighted regressors, adds WordNet
    hypernyms, merges overlapping labels, prunes to canonical set,
    and saves results.

    Args:
        sub_id: Subject identifier (e.g., 'sub-NDARINVXXXXXXX')
        config: Configuration dictionary
        label_list: Original semantic label list
        frame_list: List of frame identifiers
        bias_weight: Gaze bias weight parameter
        verb_weight: Verb weight parameter
    """
    movie = config['movie']
    setting = config['setting']
    hx_docu = config['hx_docu']

    # Define output path
    output_dir = (out_path /
                  f'wordnet_gaze_weighted_regressor_inc_byhx_docu-{hx_docu}' /
                  setting /
                  f'bias-{bias_weight}_verb-{verb_weight}')
    output_file = output_dir / f'{sub_id}_regressor_{movie}.xlsx'

    # Skip if already processed
    if output_file.exists():
        return

    # Load subject's gaze-weighted regressor
    input_dir = (out_path /
                 f'gaze_weighted_regressor_inc_byhx_docu-{hx_docu}' /
                 setting /
                 f'bias-{bias_weight}_verb-{verb_weight}')
    input_file = input_dir / f'{sub_id}_regressor_{movie}.xlsx'

    weight_reg_df = pd.read_excel(input_file, index_col=0, engine='openpyxl')
    weight_reg_df = pd.DataFrame(weight_reg_df.values, index=label_list, columns=frame_list)

    # Add WordNet hypernyms
    temp_reg_df = add_wordnet_hypernyms(weight_reg_df, label_list, frame_list)

    # Merge overlapping labels
    wordnet_reg_df = merge_overlapping_labels(temp_reg_df, label_list, frame_list)

    # Prune to canonical WordNet labels
    wordnet_reg_df = prune_labels(wordnet_reg_df, config)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    wordnet_reg_df.to_excel(output_file)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function.

    Orchestrates the complete workflow:
    1. Load movie data and regressor information
    2. Filter participants based on QC criteria
    3. Process each subject for each gaze weight combination
    """
    print('=' * 70)
    print('WordNet Gaze-Weighted Regressor Preparation')
    print('=' * 70)
    print(f"Movie: {CONFIG['movie']}")
    print(f"Setting: {CONFIG['setting']}")
    print(f"Regressor option: {CONFIG['reg_option']}")
    print(f"Gaze weight combinations: {len(CONFIG['gaze_weight_grid'])}")
    for bias_w, verb_w in CONFIG['gaze_weight_grid']:
        print(f"  - bias={bias_w}, verb={verb_w}")
    print('=' * 70)

    # Load movie data
    reg_df, label_list, frame_list, file_list = load_movie_data(CONFIG)

    # Filter participants
    id_list, participants_df = load_and_filter_participants(CONFIG)

    # Process subjects
    print()
    print('* Add superordinate labels to gaze-weighted regressors according to WordNet hierarchy')

    for bias_weight, verb_weight in CONFIG['gaze_weight_grid']:
        for sub_id in tqdm(id_list, desc=f'bias-{bias_weight}_verb-{verb_weight}'):
            process_subject_gaze_weights(
                sub_id,
                CONFIG,
                label_list,
                frame_list,
                bias_weight,
                verb_weight
            )

    print()
    print('Processing complete!')


if __name__ == '__main__':
    main()
