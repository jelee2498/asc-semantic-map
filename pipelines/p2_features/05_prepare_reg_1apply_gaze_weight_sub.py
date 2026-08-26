"""
Apply gaze attention weights to semantic regressors for encoding models.

This script loads pre-computed gaze weight maps (from step 0) and applies them
to semantic regressor matrices. Noun labels (objects) receive pixel-level gaze
weighting, while verb labels (actions) receive scalar multiplication.

Reference Files (source):
    - S:\jelee\02_asd_semantic_map\1_code\17_param_search\00_prepare_data_release11_3apply_gaze_weight_inc_byhx_sub.py
    - S:\jelee\02_asd_semantic_map\1_code\17_param_search\00_prepare_data_release11_3apply_gaze_weight_inc_byhx_pbs.pbs

Input:
    - Gaze weight maps (.npy files from step 0)
    - Semantic regressor Excel files (movie annotations)
    - Pixel-level label masks (object segmentation)

Output:
    - Gaze-weighted regressor Excel files (ready for encoding model training)
    - One Excel file per subject per movie

Usage:
    python 05_prepare_reg_1apply_gaze_weight_sub.py <sub_id> <movie> <setting> <hx_docu> [--bias_weight <float>] [--verb_weight <float>]

Examples:
    python 05_prepare_reg_1apply_gaze_weight_sub.py sub-NDARAB653ZXP DM within False --bias_weight 0.1 --verb_weight 0.5
    python 05_prepare_reg_1apply_gaze_weight_sub.py sub-NDARAB653ZXP TP within True --bias_weight 0.9 --verb_weight 1.0

Project: 02_asd_semantic_map
Pipeline: 99_main
Task: 05_prepare_reg
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths come from config/paths.yml via lib/project_config.py.  The original
# working tree hard-coded storage roots here (S:/ and Q:/ on Windows,
# /MIPL/store7 and /MIPL/store10 on Linux); those are machine-specific and are
# not distributed.  'store10' was declared but never read, and is dropped.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
from project_config import RAW, CODE, PIPELINE, TEMPLATES  # noqa: E402

CONFIG: Dict[str, Any] = {
    # Project structure
    'project': '02_asd_semantic_map',
    'pipeline': '99_main',
    'task': '05_prepare_reg',

    # Base paths
    'raw_path': RAW,
    'code_path': CODE,
    'pipe_path': PIPELINE,
    'tpl_path': TEMPLATES,

    # Movie metadata (copied from functions.py to avoid wildcard import)
    # movie_dict[movie] = [name, n_trs, fps]
    'movie_dict': {
        'TP': {
            'name': 'present',
            'n_trs': 250,
            'fps': 0.0427458412297326  # Frame duration in seconds (24 fps)
        },
        'DM': {
            'name': 'despicable_me',
            'n_trs': 750,
            'fps': 0.0417072153482552  # Frame duration in seconds (23.98 fps)
        }
    },

    # Regressor options by encoding model setting
    'reg_options': {
        'cross': 'only_41+behav',     # Cross-movie encoding (DM train → TP test)
        'within': 'all (wordnet)'     # Within-movie encoding (DM train → DM test)
    },

    # Default gaze weight parameters
    'bias_weight_default': 0.1,  # Bias weight for noun labels (0-1 range)
    'verb_weight_default': 0.5   # Weight multiplier for verb labels
}

# TODO: Refactor functions.py to use explicit imports instead of wildcard
# Currently we've copied movie_dict into CONFIG to avoid 'from functions import *'


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for gaze weight application.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - sub_id (str): Subject ID in BIDS format
            - movie (str): Movie condition ('TP' or 'DM')
            - setting (str): Encoding model setting ('cross' or 'within')
            - hx_docu (str): ASD by-history documentation filter ('True' or 'False')
            - bias_weight (float): Bias weight for noun labels
            - verb_weight (float): Weight multiplier for verb labels
    """
    parser = argparse.ArgumentParser(
        description='Apply gaze attention weights to semantic regressors for encoding models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_prepare_reg_1apply_gaze_weight_sub.py sub-NDARINVXXXXXXX DM within False --bias_weight 0.1 --verb_weight 0.5
  python 05_prepare_reg_1apply_gaze_weight_sub.py sub-NDARINVXXXXXXX TP cross True --bias_weight 0.2 --verb_weight 0.8

Notes:
  - Processes a single subject (designed for PBS array job submission)
  - Loads pre-computed gaze weights from step 0
  - Noun labels (containing '_') get pixel-level gaze weighting
  - Verb labels get scalar multiplication by verb_weight
  - Outputs Excel file with weighted regressors
        """
    )

    parser.add_argument(
        'sub_id',
        type=str,
        help='Subject ID in BIDS format (e.g., sub-NDARINVXXXXXXX)'
    )

    parser.add_argument(
        'movie',
        type=str,
        choices=['TP', 'DM'],
        help='Movie condition: TP (The Present) or DM (Despicable Me)'
    )

    parser.add_argument(
        'setting',
        type=str,
        choices=['cross', 'within'],
        help=(
            'Encoding model setting: '
            'cross (DM train → TP test) or within (DM train → DM test)'
        )
    )

    parser.add_argument(
        'hx_docu',
        type=str,
        choices=['True', 'False'],
        help=(
            'Filter subjects with ASD by-history requiring documentation. '
            'True: only include ASD by-history with documentation provided. '
            'False: include all ASD by-history subjects.'
        )
    )

    parser.add_argument(
        '--bias_weight',
        type=float,
        default=CONFIG['bias_weight_default'],
        help=f"Bias weight for noun labels (default: {CONFIG['bias_weight_default']})"
    )

    parser.add_argument(
        '--verb_weight',
        type=float,
        default=CONFIG['verb_weight_default'],
        help=f"Weight multiplier for verb labels (default: {CONFIG['verb_weight_default']})"
    )

    return parser.parse_args()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_movie_frames_and_regressors(
    movie: str,
    setting: str
) -> Tuple[pd.DataFrame, List[str], List[Path]]:
    """
    Load movie frames and regressor DataFrame for the specified movie.

    Handles special frame numbering for TP movie which includes:
    - Prepended synchronization frames
    - Overlapping frame indices that need renumbering

    Args:
        movie: Movie identifier ('TP' or 'DM')
        setting: Encoding model setting ('cross' or 'within')

    Returns:
        Tuple of (regressor_df, frame_list, file_list):
            - regressor_df: DataFrame with labels as rows, frames as columns
            - frame_list: List of frame column names from regressor_df
            - file_list: List of Path objects to movie frame JPEG files

    Raises:
        ValueError: If movie or setting is invalid
    """
    print('* Loading movie frames and regressor metadata')

    raw_path = CONFIG['raw_path']
    reg_option = CONFIG['reg_options'][setting]

    if movie == 'TP':
        # Load regressor Excel with prepended synchronization frames
        regressor_df = pd.read_excel(
            raw_path / 'movie' / movie / 'prepend_sync_regressor_TP.xlsx',
            sheet_name=reg_option,
            engine='openpyxl',
            index_col='Original Row'
        )

        # Extract frame numbers with special handling
        # Find exactly 7 prepended frames (columns containing 'f')
        # and track where they end in the column list
        frame_no_list_prepend = []
        after_prepended_frame_idx = 0
        for frame_no in list(regressor_df.columns):
            if len(frame_no_list_prepend) == 7:
                break
            else:
                after_prepended_frame_idx += 1
                if len(frame_no_list_prepend) < 7 and 'f' in frame_no:
                    frame_no_list_prepend.append(frame_no.strip('f'))

        # Identify overlapping frames (marked with .1 suffix) after prepend frames
        overlap_ids = [
            i for i, frame_no in enumerate(
                list(regressor_df.columns)[after_prepended_frame_idx:]
            )
            if '.1' in frame_no
        ]

        # Renumber overlapping frames (offset by 131)
        frame_no_list_overlap = [
            str(int(frame_no.strip('f').strip('.1')) + 131).zfill(4)
            for i, frame_no in enumerate(
                list(regressor_df.columns)[after_prepended_frame_idx:]
            )
            if i in overlap_ids
        ]

        # Non-overlapping frames also need offset by 131
        frame_no_list_nonoverlap = [
            str(int(frame_no.strip('f')) + 131).zfill(4)
            for i, frame_no in enumerate(
                list(regressor_df.columns)[after_prepended_frame_idx:]
            )
            if i not in overlap_ids
        ]

        # Combine and sort all frame numbers
        frame_no_list = sorted(
            frame_no_list_prepend + frame_no_list_overlap + frame_no_list_nonoverlap
        )

        # Build file paths from prepend_sync_frames directory
        frames_dir = raw_path / 'movie' / movie / 'prepend_sync_frames'
        file_list = [
            list(frames_dir.glob(f'present_{frame_no}.jpg'))[0]
            for frame_no in frame_no_list
        ]

        # Extract frame columns from DataFrame for regressor access
        # (these are the actual column names like 'f1', 'f2', etc.)
        frame_list = [col for col in regressor_df.columns if 'f' in col]

    else:  # DM movie
        # Load regressor Excel
        regressor_df = pd.read_excel(
            raw_path / 'movie' / movie / 'sync_regressor_DM.xlsx',
            sheet_name=reg_option,
            engine='openpyxl',
            index_col=1  # Use second column as index
        )

        # Extract frame columns (those starting with 'f')
        frame_list = [col for col in regressor_df.columns if 'f' in col]

        # Build file paths (exclude 'allframes*' directories using negative lookahead pattern)
        file_list = []
        for frame_no in frame_list:
            # Search pattern: ?[!l][!l]*/ excludes directories starting with 'al'
            # This excludes 'allframes' while including numbered directories
            matching_files = list(
                (raw_path / 'movie' / movie).glob(
                    f'?[!l][!l]*/despicable_me_{frame_no.strip("f")}.jpg'
                )
            )
            if matching_files:
                file_list.append(matching_files[0])
            else:
                raise FileNotFoundError(
                    f"Frame {frame_no} not found for movie {movie}"
                )

    print(f'  Loaded {len(file_list)} frames for movie {movie}')
    print(f'  Regressor option: {reg_option}')
    print(f'  Labels in regressor: {len(regressor_df)}')

    return regressor_df, frame_list, file_list


def apply_gaze_weights_to_regressors(
    regressor_df: pd.DataFrame,
    frame_list: List[str],
    file_list: List[Path],
    gaze_weight_path: Path,
    label_weight_path: Path,
    sub_id: str,
    movie: str,
    hx_docu: str,
    verb_weight: float
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply gaze weights to regressors for each movie frame.

    Processing workflow:
    1. For each movie frame, load pre-computed gaze weight map
    2. For noun labels (containing '_'):
       - Load pixel-level label mask
       - Multiply gaze weight map by label mask
       - Calculate mean weight within masked region
    3. For verb labels:
       - Multiply original regressor value by verb_weight

    Args:
        regressor_df: Regressor DataFrame with labels as rows
        frame_list: List of frame column names from regressor_df
        file_list: List of movie frame file paths
        gaze_weight_path: Path to directory containing gaze weight .npy files
        label_weight_path: Path to directory containing pixel-level label masks
        sub_id: Subject ID
        movie: Movie identifier ('TP' or 'DM')
        hx_docu: By-history documentation filter ('True' or 'False')
        verb_weight: Weight multiplier for verb labels

    Returns:
        Tuple of (weight_regressor_df, noun_label_list):
            - weight_regressor_df: DataFrame with gaze-weighted regressors
            - noun_label_list: List of noun labels (for subsequent normalization)

    Notes:
        - Noun labels are identified by presence of '_' in label name
        - Verb labels are identified by absence of '_'
        - Gaze weight maps are float16 .npy files from step 0
        - Label masks are boolean .npy files marking object pixels
    """
    print('* Applying gaze weights to regressors')

    label_list = list(regressor_df.index)
    noun_label_list: List[str] = []

    # Collect all column data first to avoid DataFrame fragmentation
    all_columns_data: Dict[str, Dict[str, float]] = {}

    for file_idx, file_path in enumerate(tqdm(file_list, desc=f'  Processing {sub_id}')):
        print(f'  Frame: {file_path.name}')

        # Load pre-computed gaze weight map
        weight_file_candidates = [
            gaze_weight_path / movie / sub_id / f'{file_path.stem}_weight.npy',
            gaze_weight_path / movie / sub_id / f'{file_path.name.replace(".jpg", "")}_weight.npy'
        ]

        weight_map = None
        for weight_file in weight_file_candidates:
            if weight_file.exists():
                weight_map = np.load(weight_file)
                break

        if weight_map is None:
            raise FileNotFoundError(
                f"Gaze weight file not found for {sub_id}, frame {file_path.name}"
            )

        # Process each label for this frame
        label_weight_dict: Dict[str, float] = {}
        for label in label_list:
            if '_' in label:  # Noun label (object)
                if file_idx == 0:  # Only collect noun labels once
                    noun_label_list.append(label)

                # Load pixel-level label mask
                label_mask_file = (
                    label_weight_path / movie / 'pixel_level_label' / label /
                    f'{file_path.stem}.npy'
                )

                if not label_mask_file.exists():
                    raise FileNotFoundError(
                        f"Label mask not found: {label_mask_file}"
                    )

                label_map = np.load(label_mask_file)

                # Multiply weight map by label mask
                label_weight_map = np.multiply(weight_map, label_map)

                # Calculate mean weight within labeled region
                if len(label_weight_map[label_map == True]) > 0:
                    label_weight_dict[label] = label_weight_map[label_map == True].mean()
                else:
                    # Label not present in this frame
                    label_weight_dict[label] = 0

            else:  # Verb label (action)
                # Simple scalar multiplication of original regressor value
                label_weight_dict[label] = (
                    verb_weight * regressor_df.loc[label, frame_list[file_idx]]
                )

        # Store column data (avoid adding to DataFrame in loop)
        column_name = file_path.stem.replace('.jpg', '')
        all_columns_data[column_name] = label_weight_dict

    # Build DataFrame all at once to avoid fragmentation
    weight_regressor_df = pd.DataFrame(all_columns_data, index=label_list).T.T

    print(f'  Processed {len(file_list)} frames')
    print(f'  Noun labels: {len(noun_label_list)}')
    print(f'  Verb labels: {len(label_list) - len(noun_label_list)}')

    return weight_regressor_df, noun_label_list


def normalize_and_combine_weights(
    weight_regressor_df: pd.DataFrame,
    regressor_df: pd.DataFrame,
    frame_list: List[str],
    noun_label_list: List[str],
    bias_weight: float
) -> pd.DataFrame:
    """
    Apply Min-Max scaling to noun labels and combine with bias weight.

    This normalization ensures that:
    1. Gaze-weighted noun labels are scaled to [0, 1-bias_weight] range
    2. A bias weight is added back from the original regressors
    3. Final range is [bias_weight, 1] for noun labels

    The bias weight prevents complete suppression of non-gazed objects,
    maintaining some baseline activation for all present objects.

    Args:
        weight_regressor_df: DataFrame with raw gaze-weighted regressors
        regressor_df: Original regressor DataFrame (for bias weighting)
        frame_list: List of frame column names
        noun_label_list: List of noun labels to normalize
        bias_weight: Bias weight for combining with original regressors (0-1 range)

    Returns:
        DataFrame with normalized and bias-combined regressors

    Notes:
        - Only noun labels are normalized; verb labels remain unchanged
        - Min-Max scaling is applied across all frames for each label
        - Formula: normalized = (1-bias) * gaze_weight + bias * original
    """
    print('* Normalizing noun label weights with Min-Max scaling')

    # Min-Max scale noun labels to [0, 1-bias_weight] range
    scaler = MinMaxScaler(feature_range=(0, 1 - bias_weight))
    weight_regressor_df.loc[noun_label_list] = scaler.fit_transform(
        weight_regressor_df.loc[noun_label_list]
    )

    # Add bias weight from original regressors
    # Formula: final = (1-bias) * gaze_weight + bias * original
    weight_regressor_df.loc[noun_label_list] = pd.DataFrame(
        weight_regressor_df.loc[noun_label_list].values +
        (regressor_df.loc[noun_label_list, frame_list] * bias_weight).values,
        index=weight_regressor_df.loc[noun_label_list].index,
        columns=weight_regressor_df.loc[noun_label_list].columns
    )

    print(f'  Scaled {len(noun_label_list)} noun labels to [{bias_weight:.2f}, 1.0] range')

    return weight_regressor_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution flow for applying gaze weights to regressors.

    Workflow:
    1. Parse command-line arguments
    2. Build output paths
    3. Load movie frames and regressor DataFrame
    4. Apply gaze weights to regressors (noun and verb labels)
    5. Normalize noun labels and combine with bias weight
    6. Save weighted regressor Excel file

    The script is designed for PBS array job submission, processing one
    subject at a time in parallel across the cluster.
    """
    # Parse command-line arguments
    args = parse_arguments()

    print(f'\n{"="*70}')
    print(f'GAZE WEIGHT APPLICATION TO REGRESSORS')
    print(f'{"="*70}')
    print(f'Subject: {args.sub_id}')
    print(f'Movie: {args.movie}')
    print(f'Setting: {args.setting}')
    print(f'By-history documentation filter: {args.hx_docu}')
    print(f'Bias weight: {args.bias_weight}')
    print(f'Verb weight: {args.verb_weight}')
    print(f'{"="*70}\n')

    # Build paths
    proc_path = CONFIG['pipe_path'] / CONFIG['task']
    out_path = proc_path / 'out'
    gaze_weight_path = out_path / f'gaze_weight_inc_byhx_docu-{args.hx_docu}'
    label_weight_path = CONFIG['raw_path'] / 'movie'

    # Create output directory
    output_dir = (
        out_path /
        f'gaze_weighted_regressor_inc_byhx_docu-{args.hx_docu}' /
        args.setting /
        f'bias-{args.bias_weight}_verb-{args.verb_weight}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load movie frames and regressors
    regressor_df, frame_list, file_list = load_movie_frames_and_regressors(
        args.movie,
        args.setting
    )

    # Apply gaze weights to regressors
    weight_regressor_df, noun_label_list = apply_gaze_weights_to_regressors(
        regressor_df,
        frame_list,
        file_list,
        gaze_weight_path,
        label_weight_path,
        args.sub_id,
        args.movie,
        args.hx_docu,
        args.verb_weight
    )

    # Normalize and combine with bias weight
    weight_regressor_df = normalize_and_combine_weights(
        weight_regressor_df,
        regressor_df,
        frame_list,
        noun_label_list,
        args.bias_weight
    )

    # Save weighted regressor Excel file
    output_file = output_dir / f'{args.sub_id}_regressor_{args.movie}.xlsx'
    weight_regressor_df.to_excel(output_file)

    print(f'\n* Saved weighted regressors to: {output_file}')
    print(f'\n{"="*70}')
    print(f'COMPLETED SUCCESSFULLY')
    print(f'{"="*70}\n')


if __name__ == "__main__":
    main()
