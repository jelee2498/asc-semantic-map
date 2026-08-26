"""
Calculate Gaussian gaze attention weights for movie frames.

This script processes eye-tracking data from DeepMREye predictions to generate
attention weight maps for each movie frame. The weights are calculated using
2D Gaussian kernels centered at predicted gaze positions, with kernel size
scaled by prediction uncertainty.

Reference Files (read-only):
    - S:\jelee\02_asd_semantic_map\1_code\17_param_search\00_prepare_data_release11_2save_gaze_weight_inc_byhx_sub.py
    - S:\jelee\02_asd_semantic_map\1_code\17_param_search\00_prepare_data_release11_2save_gaze_weight_inc_byhx_pbs.pbs
    - S:\jelee\02_asd_semantic_map\1_code\17_param_search\functions.py (lines 144-173 for functions to copy)

Input:
    - DeepMREye prediction results (pickle file)
    - Movie frames (JPEG images)
    - Participant metadata (Excel file)

Output:
    - Gaze weight maps (.npy files, float16) for each movie frame
    - One weight map per frame per subject

Usage:
    python 05_prepare_reg_0save_gaze_weight_sub.py <sub_id> <movie> <hx_docu>

Example:
    python 05_prepare_reg_0save_gaze_weight_sub.py sub-NDARINVXXXXXXX TP True
    python 05_prepare_reg_0save_gaze_weight_sub.py sub-NDARINVXXXXXXX DM False

Project: 02_asd_semantic_map
Pipeline: 99_main
Task: 05_prepare_reg
"""

import pickle
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths come from config/paths.yml via lib/project_config.py.  The original
# working tree hard-coded storage roots here (S:/ on Windows, /MIPL/store7 on
# Linux); those are machine-specific and are not distributed.
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

    # Quality control thresholds
    'fd_threshold': 0.5,  # Framewise displacement threshold in mm
    'uncertainty_percentiles': {'upper': 80, 'lower': 20},  # Bounds for uncertainty clipping

    # Gaze mapping parameters (movie-specific)
    'kernel_sizes': {'TP': 1280, 'DM': 720},  # Gaussian kernel diameter in pixels
    'sigma_factors': {'TP': 45, 'DM': 30},  # Sigma = sigma_factor * uncertainty
    'radius_gaze': {'TP': 10, 'DM': 5},  # Gaze position circle radius in pixels
    'padding_factor': 1,  # Padding = resolution / padding_factor

    # Diagnosis filter
    'dx_filter': 'all',  # 'TD', 'ASD', or 'all'

    # Bias weight (must be 0 at this stage; adjusted in later pipeline steps)
    'bias_weight': 0,

    # Coordinate transformation table (empirically derived from psychophysics calibration)
    # Maps decoded gaze position (visual degrees) to movie screen coordinates
    # Format: {site: {movie: [x_scale, y_scale]}}
    'transform_table': {
        'RU': {
            'TP': [37/29, 20.5/15.5],   # Rutgers - The Present
            'DM': [37/29, 20.5/16.5]    # Rutgers - Despicable Me
        },
        'CBIC': {
            'TP': [33.5/26.5, 26/15],   # CBIC - The Present
            'DM': [33.5/26.5, 26/15]    # CBIC - Despicable Me
        }
    },

    # Movie metadata
    'movie_dict': {
        'TP': {
            'name': 'present',
            'n_trs': 250,
            'tr': 0.8,
            'fps': 0.0416666666666667  # Frame duration in seconds (24 fps)
        },
        'DM': {
            'name': 'despicable_me',
            'n_trs': 750,
            'tr': 0.8,
            'fps': 0.0417014178482068  # Frame duration in seconds (23.98 fps)
        }
    }
}

# Validate bias weight configuration
assert CONFIG['bias_weight'] == 0, (
    'bias_weight must be 0 at this stage. '
    'Bias weighting is applied in subsequent pipeline steps.'
)


# ============================================================================
# HELPER FUNCTIONS (COPIED FROM 17_param_search/functions.py)
# ============================================================================

def gaussain_2d(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Create 2D Gaussian kernel for attention weight mapping.

    Note: Function name retains original typo ('gaussain') to maintain
    compatibility with other scripts in the pipeline.

    Args:
        kernel_size: Size of the kernel in pixels (should be odd)
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        2D numpy array of shape (kernel_size, kernel_size) with normalized
        Gaussian weights
    """
    kernel1d = cv2.getGaussianKernel(kernel_size, sigma)
    kernel2d = np.outer(kernel1d, kernel1d.transpose())
    return kernel2d


def transformed_gaze(
    results_dict: Dict[str, Any],
    sub_id: str,
    site: str,
    movie: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform decoded gaze position from visual degrees to movie screen coordinates.

    The transformation corrects for:
    1. Scale conversion (divide by 10 from DeepMREye output)
    2. Y-axis calibration (multiply by 1.2925 to match ground truth)
    3. Site- and movie-specific screen geometry (from transform_table)

    Args:
        results_dict: DeepMREye prediction results dictionary with keys:
            - '{sub_id}_task-movie{movie}': dict containing 'pred_xy' and 'uncertainty'
        sub_id: Subject ID in BIDS format (e.g., 'sub-NDARINVXXXXXXX')
        site: Acquisition site ('RU' or 'CBIC')
        movie: Movie condition ('TP' or 'DM')

    Returns:
        Tuple of (transformed_coords, uncertainty_values):
            - transformed_coords: (n_frames, 2) array of [x, y] screen coordinates
            - uncertainty_values: (n_frames, 2) array of prediction uncertainties
    """
    # Extract gaze predictions and take median across samples
    gaze_sub = np.median(
        results_dict[f'{sub_id}_task-movie{movie}']['pred_xy'],
        axis=1
    ) / 10  # Scale conversion

    # Apply y-axis calibration factor (empirically determined from ground truth)
    gaze_sub[:, 1] = gaze_sub[:, 1] * 1.2925 * (-1)  # Invert y-axis

    # Extract uncertainty estimates
    uncer_sub = np.median(
        results_dict[f'{sub_id}_task-movie{movie}']['uncertainty'],
        axis=1
    )

    # Apply site- and movie-specific coordinate transformation
    trans_sub = np.zeros_like(gaze_sub)
    transform = CONFIG['transform_table'][site][movie]
    for axis in range(2):
        trans_sub[:, axis] = gaze_sub[:, axis] * transform[axis]

    return trans_sub, uncer_sub


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for gaze weight calculation.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - sub_id (str): Subject ID
            - movie (str): Movie condition ('TP' or 'DM')
            - hx_docu (str): ASD by-history documentation filter ('True' or 'False')
    """
    parser = argparse.ArgumentParser(
        description='Calculate Gaussian gaze attention weights for movie frames based on DeepMREye predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_prepare_reg_0save_gaze_weight_sub.py sub-NDARINVXXXXXXX TP True
  python 05_prepare_reg_0save_gaze_weight_sub.py sub-NDARINVXXXXXXX DM False

Notes:
  - Processes a single subject (designed for PBS array job submission)
  - Outputs gaze weight maps as .npy files (float16 format)
  - Gaussian kernel size scales with prediction uncertainty
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
        'hx_docu',
        type=str,
        choices=['True', 'False'],
        help=(
            'Filter subjects with ASD by-history requiring documentation. '
            'True: only include ASD by-history with documentation provided. '
            'False: include all ASD by-history subjects.'
        )
    )

    return parser.parse_args()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_and_filter_participants(
    out_path: Path,
    dx: str,
    hx_docu: bool,
    fd_threshold: float,
    movie: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load participant data and apply quality control filters.

    Applies the following filters in sequence:
    1. Diagnosis (DX): TD, ASD, or all
    2. ASD by-history documentation (if hx_docu=True)
    3. Preprocessing completion (movie-specific)
    4. No quality control remarks
    5. Framewise displacement (FD) threshold (movie-specific)
    6. DeepMREye quality rating (movie-specific)

    Args:
        out_path: Path to output directory containing participants Excel file
        dx: Diagnosis filter ('TD', 'ASD', or 'all')
        hx_docu: Whether to require documentation for by-history ASD cases
        fd_threshold: Framewise displacement threshold in mm
        movie: Movie condition ('TP' or 'DM') - used for movie-specific QC metrics

    Returns:
        Tuple of (filtered_dataframe, subject_id_list):
            - filtered_dataframe: DataFrame with all filters applied
            - subject_id_list: List of subject IDs passing all filters
    """
    print('* Loading participant metadata and applying QC filters')

    # Load participant metadata
    participants_df = pd.read_excel(
        out_path / 'participants_df_deepmreye_inc_byhx.xlsx',
        index_col=0,
        engine='openpyxl'
    )

    # Filter #1: Diagnosis
    if dx == 'TD':
        dx_filt = (participants_df['DX'] == 'TD').values
    elif dx == 'ASD':
        dx_filt = (participants_df['DX'] == 'ASD').values
    elif dx == 'all':
        dx_filt = np.ones(len(participants_df), dtype=bool)
    else:
        raise ValueError(
            f"Invalid dx option: '{dx}'. Must be 'TD', 'ASD', or 'all'."
        )

    # Filter #2: ASD by-history documentation requirement
    if hx_docu:
        for sub_id_temp in participants_df[dx_filt].index:
            if (participants_df.loc[sub_id_temp]['ASD_certainty'] == 'by-history' and
                participants_df.loc[sub_id_temp]['ASD_document'] != 'documentation provided'):
                print(
                    f"  Excluding {sub_id_temp}: ASD by-history without documentation"
                )
                dx_filt[participants_df.index == sub_id_temp] = False

    # Filter #3: Preprocessing completion (movie-specific)
    prep_filt = (
        participants_df[f'prep_ok (task-movie{movie}_Atlas_s2_10k.dtseries.nii)'] == 1
    ).values

    # Filter #4: No QC remarks during preprocessing
    no_remarks_filt = (participants_df['Remarks'] == 'none').values

    # Combine filters 1-4
    combined_filt = dx_filt & prep_filt & no_remarks_filt
    participants_df_filtered = participants_df.loc[combined_filt]

    # Filter #5: Framewise displacement (movie-specific)
    fd_filt = (participants_df_filtered[f'Mean_FD_{movie}'] < fd_threshold).values

    # Filter #6: DeepMREye quality rating (movie-specific)
    deepmreye_filt = (
        participants_df_filtered[f'Rating_deepmreye_movie{movie}'] == 1
    ).values

    # Final combined filter
    final_filt = fd_filt & deepmreye_filt
    id_list = list(participants_df_filtered[final_filt].index)

    # Print summary statistics
    print(f'Participants passing all filters (n={len(id_list)}):')
    print('  - Diagnosis:')
    dx_counts = participants_df.loc[id_list]['DX'].value_counts()
    print(f"    TD: {dx_counts.get('TD', 0)}, ASD: {dx_counts.get('ASD', 0)}")
    print('  - Site:')
    site_counts = participants_df.loc[id_list]['Site'].value_counts()
    print(f"    CBIC: {site_counts.get('CBIC', 0)}, RU: {site_counts.get('RU', 0)}")
    print('  - Sex:')
    sex_counts = participants_df.loc[id_list]['Sex'].value_counts()
    print(f"    Male: {sex_counts.get('Male', 0)}, Female: {sex_counts.get('Female', 0)}")

    return participants_df, id_list


def load_gaze_predictions(out_path: Path) -> Dict[str, Any]:
    """
    Load DeepMREye prediction results from pickle file.

    Args:
        out_path: Path to output directory containing prediction pickle file

    Returns:
        Dictionary with keys '{sub_id}_task-movie{movie}' containing:
            - 'pred_xy': Predicted gaze coordinates (samples, frames, 2)
            - 'uncertainty': Prediction uncertainty (samples, frames, 2)
    """
    print('* Loading DeepMREye prediction results')

    pickle_path = out_path / 'results_hbn_td_asd_inc_byhx_edited.pickle'
    with open(pickle_path, 'rb') as fp:
        gaze_results = pickle.load(fp)

    return gaze_results


def calculate_uncertainty_bounds(
    gaze_results: Dict[str, Any],
    id_list: List[str],
    movie: str,
    percentiles: Dict[str, int]
) -> Tuple[float, float]:
    """
    Calculate lower and upper uncertainty bounds from all subjects.

    The bounds are used to clip extreme uncertainty values and normalize
    the range of Gaussian kernel sizes across subjects.

    Args:
        gaze_results: DeepMREye prediction results dictionary
        id_list: List of subject IDs to include in calculation
        movie: Movie condition ('TP' or 'DM')
        percentiles: Dictionary with 'upper' and 'lower' percentile values

    Returns:
        Tuple of (lower_bound, upper_bound) for uncertainty clipping
    """
    print('* Calculating uncertainty bounds across all subjects')

    # Collect median uncertainty across frames for all subjects
    uncer_list = []
    for sub_id in tqdm(id_list, desc='Processing subjects'):
        uncer_sub = gaze_results[f'{sub_id}_task-movie{movie}']['uncertainty']
        uncer_list.append(np.median(uncer_sub, axis=1))

    # Calculate percentile bounds
    uncer_array = np.array(uncer_list)
    upper_uncer = np.percentile(uncer_array, percentiles['upper'])
    lower_uncer = np.percentile(uncer_array, percentiles['lower'])

    print(f'  Upper bound (p{percentiles["upper"]}): {upper_uncer:.4f}')
    print(f'  Lower bound (p{percentiles["lower"]}): {lower_uncer:.4f}')

    return lower_uncer, upper_uncer


def load_movie_frames(movie: str) -> List[Path]:
    """
    Load movie frame file paths in correct temporal order.

    Handles special frame numbering for TP movie which includes:
    - Prepended synchronization frames
    - Overlapping frame indices that need renumbering

    Args:
        movie: Movie condition ('TP' or 'DM')

    Returns:
        List of Path objects pointing to movie frame JPEG files in temporal order
    """
    print('* Loading movie frame metadata')

    raw_path = CONFIG['raw_path']

    if movie == 'TP':
        # Load regressor Excel to get frame ordering
        regressor_df = pd.read_excel(
            raw_path / 'movie' / movie / 'prepend_sync_regressor_TP.xlsx',
            sheet_name='only_41+behav',
            engine='openpyxl',
            index_col='Row'
        )

        # Extract frame numbers with special handling for prepended/overlapping frames
        frame_no_list_prepend = [
            frame_no.strip('f') for frame_no in list(regressor_df.columns)[:7]
        ]

        # Identify overlapping frames (marked with .1 suffix)
        overlap_ids = [
            i for i, frame_no in enumerate(list(regressor_df.columns)[7:])
            if '.1' in frame_no
        ]

        # Renumber overlapping frames (offset by 131)
        frame_no_list_overlap = [
            str(int(frame_no.strip('f').strip('.1')) + 131).zfill(4)
            for i, frame_no in enumerate(list(regressor_df.columns)[7:])
            if i in overlap_ids
        ]

        # Non-overlapping frames also need offset
        frame_no_list_nonoverlap = [
            str(int(frame_no.strip('f')) + 131).zfill(4)
            for i, frame_no in enumerate(list(regressor_df.columns)[7:])
            if i not in overlap_ids
        ]

        # Combine and sort frame numbers
        frame_no_list = sorted(
            frame_no_list_prepend + frame_no_list_overlap + frame_no_list_nonoverlap
        )

        # Build file paths
        frames_dir = raw_path / 'movie' / movie / 'prepend_sync_frames'
        file_list = [
            list(frames_dir.glob(f'present_{frame_no}.jpg'))[0]
            for frame_no in frame_no_list
        ]

    else:  # DM movie
        # Load regressor Excel
        regressor_df = pd.read_excel(
            raw_path / 'movie' / movie / 'sync_regressor_DM.xlsx',
            engine='openpyxl',
            index_col='Row'
        )

        # Get frame numbers (strip 'f' prefix)
        frame_columns = [col.strip('f') for col in regressor_df.columns]

        # Build file paths (excluding 'allframes' directories)
        file_list = []
        for frame_no in frame_columns:
            # Search in numbered directories only (exclude 'allframes*')
            matching_files = list(
                (raw_path / 'movie' / movie).glob(
                    f'?[!l][!l]*/despicable_me_{frame_no}.jpg'
                )
            )
            if matching_files:
                file_list.append(matching_files[0])

    print(f'  Loaded {len(file_list)} frames for movie {movie}')

    return file_list


# ============================================================================
# GAZE WEIGHT CALCULATION
# ============================================================================

def calculate_gaze_weights(
    sub_id: str,
    movie: str,
    site: str,
    gaze_results: Dict[str, Any],
    file_list: List[Path],
    weight_path: Path,
    lower_uncer: float,
    upper_uncer: float
) -> None:
    """
    Calculate and save Gaussian gaze attention weights for all movie frames.

    Processing workflow for each frame:
    1. Load movie frame and apply zero-padding for edge cases
    2. Transform gaze coordinates to pixel space
    3. Create 2D Gaussian kernel scaled by uncertainty
    4. Place kernel at gaze position on padded frame
    5. Extract unpadded attention map
    6. Save as float16 .npy file

    Args:
        sub_id: Subject ID
        movie: Movie condition ('TP' or 'DM')
        site: Acquisition site ('RU' or 'CBIC')
        gaze_results: DeepMREye prediction dictionary
        file_list: List of movie frame file paths in temporal order
        weight_path: Directory to save weight files
        lower_uncer: Lower bound for uncertainty clipping
        upper_uncer: Upper bound for uncertainty clipping

    Returns:
        None (saves .npy files to disk)

    Notes:
        - Upper-left corner is array origin (following OpenCV convention)
        - Padding allows Gaussian kernels for out-of-frame gaze points
        - Uncertainty is clipped to [lower_uncer, upper_uncer] range
    """
    print(f'* Calculating gaze weights for {sub_id}')

    # Create output directory
    output_dir = weight_path / movie / sub_id

    # Skip if already processed (all weight files exist)
    existing_weights = list(output_dir.glob('*_weight.npy')) if output_dir.exists() else []
    if len(existing_weights) >= len(file_list):
        print(f'  Already processed ({len(existing_weights)} weight files found). Skipping.')
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get movie-specific parameters
    kernel_size = CONFIG['kernel_sizes'][movie]
    sigma_factor = CONFIG['sigma_factors'][movie]
    radius_gaze = CONFIG['radius_gaze'][movie]
    padding_factor = CONFIG['padding_factor']
    bias_weight = CONFIG['bias_weight']

    # Get transformed gaze coordinates and uncertainty
    trans_sub, uncer_sub = transformed_gaze(gaze_results, sub_id, site, movie)

    # Clip uncertainty to bounds
    uncer_sub[uncer_sub < lower_uncer] = lower_uncer
    uncer_sub[uncer_sub > upper_uncer] = upper_uncer

    # Get frame resolution from first frame
    first_frame = cv2.imread(str(file_list[0]))
    x_res, y_res = first_frame.shape[0], first_frame.shape[1]

    # Precompute gaze position circle mask
    kernel_size_half = kernel_size // 2
    gaze_x, gaze_y = np.ogrid[-radius_gaze:radius_gaze, -radius_gaze:radius_gaze]
    gaze_ids = gaze_x**2 + gaze_y**2 <= radius_gaze**2

    # Precompute circular kernel mask
    kernel_x, kernel_y = np.ogrid[
        -kernel_size_half:kernel_size_half,
        -kernel_size_half:kernel_size_half
    ]
    kernel_ids = kernel_x**2 + kernel_y**2 <= kernel_size_half**2

    # Process each frame
    for frame_idx, file_path in enumerate(tqdm(file_list, desc=f'  Processing {sub_id}')):
        # Load frame
        frame = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create zero-padded frame for out-of-frame gaze points
        pad_rgb = np.zeros(
            (int(x_res + x_res/padding_factor),
             int(y_res + y_res/padding_factor),
             3),
            dtype=np.uint8
        )

        # Place original frame in center of padded array
        pad_rgb[
            int(pad_rgb.shape[0]/2 - x_res/2):int(pad_rgb.shape[0]/2 + x_res/2),
            int(pad_rgb.shape[1]/2 - y_res/2):int(pad_rgb.shape[1]/2 + y_res/2)
        ] = frame_rgb

        # Initialize attention map
        attention = np.zeros(
            (int(x_res + x_res/padding_factor),
             int(y_res + y_res/padding_factor))
        )

        # Convert gaze coordinates to pixel space
        # Note: y-axis is inverted (x_res - ...) due to screen coordinate system
        g_x = x_res - int(x_res/2 + x_res/2 * trans_sub[frame_idx][1])
        g_y = int(y_res/2 + y_res/2 * trans_sub[frame_idx][0])

        # Get frame-specific uncertainty
        uncer_sub_frame = uncer_sub[frame_idx]

        # Create Gaussian kernel scaled by uncertainty
        kernel = gaussain_2d(kernel_size, sigma_factor * uncer_sub_frame)

        # Calculate kernel placement boundaries (handle edge cases)
        uppermost_kernel = 0
        uppermost = int((attention.shape[0] - x_res)/2) + g_x - kernel_size_half
        if uppermost < 0:
            uppermost_kernel = -uppermost
            uppermost = 0

        lowermost_kernel = kernel_size
        lowermost = int((attention.shape[0] - x_res)/2) + g_x + kernel_size_half
        if lowermost > int(x_res + x_res/padding_factor):
            lowermost_kernel = int(x_res + x_res/padding_factor) - uppermost
            lowermost = int(x_res + x_res/padding_factor)

        leftmost_kernel = 0
        leftmost = int((attention.shape[1] - y_res)/2) + g_y - kernel_size_half
        if leftmost < 0:
            leftmost_kernel = -leftmost
            leftmost = 0

        rightmost_kernel = kernel_size
        rightmost = int((attention.shape[1] - y_res)/2) + g_y + kernel_size_half
        if rightmost > int(y_res + y_res/padding_factor):
            rightmost_kernel = int(y_res + y_res/padding_factor) - leftmost
            rightmost = int(y_res + y_res/padding_factor)

        # Extract circular region of kernel that fits in bounds
        kernel_ids_shrink = kernel_ids[
            uppermost_kernel:lowermost_kernel,
            leftmost_kernel:rightmost_kernel
        ]

        # Place Gaussian kernel on attention map
        try:
            attention[uppermost:lowermost, leftmost:rightmost][kernel_ids_shrink] += \
                kernel[uppermost_kernel:lowermost_kernel,
                       leftmost_kernel:rightmost_kernel][kernel_ids_shrink]
        except IndexError:
            # Skip frames where kernel placement fails (edge case)
            pass

        # Extract unpadded frame weight
        frame_weight = attention[
            int(pad_rgb.shape[0]/2 - x_res/2):int(pad_rgb.shape[0]/2 + x_res/2),
            int(pad_rgb.shape[1]/2 - y_res/2):int(pad_rgb.shape[1]/2 + y_res/2)
        ].astype(np.float16)

        # Apply bias weight (currently 0; used in later pipeline steps)
        frame_weight = (1 - bias_weight) * frame_weight + bias_weight * 1

        # Save weight map as .npy file
        output_file = output_dir / f"{file_path.stem}_weight.npy"
        np.save(output_file, frame_weight)

    print(f'  Saved {len(file_list)} weight maps to {output_dir}')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution flow for gaze weight calculation.

    Workflow:
    1. Parse command-line arguments (sub_id, movie, hx_docu)
    2. Load and filter participant metadata
    3. Load DeepMREye gaze predictions
    4. Calculate uncertainty bounds (20th-80th percentile across subjects)
    5. Load movie frame file paths
    6. Calculate and save gaze weight maps for target subject

    The script is designed for PBS array job submission, processing one
    subject at a time in parallel across the cluster.
    """
    # Parse command-line arguments
    args = parse_arguments()

    print(f'\n{"="*70}')
    print(f'GAZE WEIGHT CALCULATION')
    print(f'{"="*70}')
    print(f'Subject: {args.sub_id}')
    print(f'Movie: {args.movie}')
    print(f'ASD by-history documentation filter: {args.hx_docu}')
    print(f'{"="*70}\n')

    # Build paths
    proc_path = CONFIG['pipe_path'] / CONFIG['task']
    out_path = proc_path / 'out'

    # Load and filter participants
    participants_df, id_list = load_and_filter_participants(
        out_path,
        CONFIG['dx_filter'],
        args.hx_docu == 'True',
        CONFIG['fd_threshold'],
        args.movie
    )

    # Load gaze predictions
    gaze_results = load_gaze_predictions(out_path)

    # Calculate uncertainty bounds across all subjects
    lower_uncer, upper_uncer = calculate_uncertainty_bounds(
        gaze_results,
        id_list,
        args.movie,
        CONFIG['uncertainty_percentiles']
    )

    # Load movie frames
    file_list = load_movie_frames(args.movie)

    # Get subject's site information
    site = participants_df.loc[args.sub_id]['Site']

    # Define output path
    weight_path = out_path / f'gaze_weight_inc_byhx_docu-{args.hx_docu}'

    # Calculate and save gaze weights
    calculate_gaze_weights(
        args.sub_id,
        args.movie,
        site,
        gaze_results,
        file_list,
        weight_path,
        lower_uncer,
        upper_uncer
    )

    print(f'\n{"="*70}')
    print(f'COMPLETED SUCCESSFULLY')
    print(f'{"="*70}\n')


if __name__ == "__main__":
    main()
