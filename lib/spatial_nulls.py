"""
Spatial-autocorrelation-preserving null models for parcellated (MMP-360) brain maps.

Two independent nulls for testing the correspondence between two cortical maps:

  spin permutation   - Vasa et al. (2018) rotation of parcel centroids on the sphere
  eigenstrapping     - Koussis et al. (2024) resampling of geometric eigenmodes

Both answer "is this correlation stronger than expected between two maps with this
spatial smoothness?", by different mechanisms. Reporting both is a robustness check:
agreement is evidence the result is a property of the data rather than of the null.

Why not enigmatoolbox.permutation_testing.spin_test
---------------------------------------------------
1. It defaults to parcellation_name='aparc' (68 Desikan-Killiany parcels). Called on
   MMP-360 data it builds a (68, n_rot) permutation, and perm_sphere_p() then
   correlates a 68-element permuted vector against the 360-element target via
   pd.Series.corr(), which aligns on index and silently truncates to the first 68
   parcels (all left hemisphere). The empirical r and its null end up computed on
   different data, with no error raised.
2. It must be given FULL parcellated maps. Passing pre-masked vectors hits the same
   truncation - restriction to an ROI has to happen AFTER the spin, not before.
3. It rotates using numpy's global RNG with no seed parameter.

The spin implementation here reproduces enigmatoolbox's rotation algorithm and
p-value convention, but takes centroids from the native fsLR-32k sphere and this
project's own MMP dlabel (avoiding enigmatoolbox's fsaverage5 resampling of Glasser),
accepts a seed, and depends only on numpy/scipy/nibabel.

Validated against enigmatoolbox (fsaverage5 glasser_360, 1000 rotations):
  PR map, 360 parcels   ours p=0.0000, null sd=0.120  |  enigma p=0.0000, sd=0.107
  t-stats, 291 parcels  ours p=0.0545, null sd=0.072  |  enigma p=0.0595, sd=0.071

Reproducibility caveat (eigenstrapping)
---------------------------------------
SurfaceEigenstrapping's own `seed` does NOT control the surrogates: its
rotate_modes(gen=True) calls rotate_matrix(M) without passing the instance RNG, so
indirect_method() falls through to check_random_state(None) - numpy's GLOBAL
RandomState. eigenstrap_nulls() seeds that global RNG (and restores the caller's
state afterwards), which is what actually makes the draws reproducible. This only
holds in-process: with n_jobs > 1 each joblib worker gets its own global state and
reproducibility is lost. Measured on 20 logical cores (n_surr=200, 360 parcels):
n_jobs = 1/2/4/8 -> 59/45/40/52 s, so parallelism peaks around 1.5x and then
degrades from BLAS oversubscription. Keep n_jobs=1 for any reported p-value.

P-value convention
------------------
The reported p is sign-directed one-sided, matching enigmatoolbox.perm_sphere_p and
the spin-test literature: the fraction of null correlations exceeding the empirical
one in its observed direction, averaged over permuting each map in turn. Note this
chooses its tail from the data rather than a pre-specified hypothesis. Every function
also accepts two_tailed=True for the symmetric alternative.

Public API
----------
build_spin_perm_id(tpl_path, cache_dir, n_rot, seed)      -> (360, n_rot) int array
spin_p(x, y, perm_id, mask=None)                          -> p (or (p, nulls))
build_surface_eigenmodes(tpl_path, cache_dir, num_modes)  -> {'L': (evals, emodes), ...}
eigenstrap_nulls(map_mmp, tpl_path, emode_store, parcels_used, ...) -> (n_surr, n) array
eigenstrap_p(x, y, tpl_path, emode_store, mask=None, ...) -> p (or (p, nulls))
p_from_nulls(r_emp, null_r, two_tailed=False)             -> p
compare_maps(x, y, ...)                                   -> dict of both nulls
save_sensitivity_table(df, sens_dir, name)                -> Path to the CSV
"""

import os
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

try:
    from eigenstrapping import SurfaceEigenstrapping
    from eigenstrapping.geometry import calc_surface_eigenmodes
    HAS_EIGENSTRAPPING = True
except ImportError:
    HAS_EIGENSTRAPPING = False


# =============================================================================
# DEFAULT PATHS
# =============================================================================

# Defaults come from config/paths.yml (see lib/project_config.py).  The original
# working tree derived them from a hard-coded storage root here (S:/ on Windows,
# /Volumes/mipl on macOS, /MIPL/store7 on Linux); those are machine-specific and
# are not distributed.  Both remain overridable per call.
from project_config import TEMPLATES as _TEMPLATES, SPATIAL_NULLS_CACHE as _CACHE

DEFAULT_TPL_PATH = _TEMPLATES

# Shared across every script that imports this module: spin permutations and surface
# eigenmodes depend only on the atlas and the surface, never on the analysis, so all
# callers reuse one cache instead of each rebuilding them.
DEFAULT_CACHE_DIR = _CACHE

N_PARCELS = 360
N_VTX_HEMI = 32492
N_VTX_FULL = 64984

_ATLAS_CACHE: Dict[str, npt.NDArray] = {}


# =============================================================================
# ATLAS
# =============================================================================

def load_mmp_atlas(tpl_path: Union[str, Path] = DEFAULT_TPL_PATH) -> npt.NDArray:
    """
    Load the MMP-360 atlas at 32k with the medial wall restored, memoized.

    Args:
        tpl_path: Template directory containing MMP/

    Returns:
        (64984,) int array, 0 = medial wall, 1-360 = parcels
        (1-180 left hemisphere, 181-360 right)
    """
    key = str(tpl_path)
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]

    mmp_folder = Path(tpl_path) / 'MMP'
    mmp_nonmed = nib.load(str(
        mmp_folder / 'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
    )).get_fdata()[0].astype(np.int32)
    med_32k = nib.load(str(
        mmp_folder / 'Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii'
    )).get_fdata()[0].astype(np.int32).nonzero()[0]
    nonmed_ids = np.array(list(set(range(N_VTX_FULL)) - set(med_32k)))

    atlas = np.zeros(N_VTX_FULL, dtype=np.int32)
    atlas[nonmed_ids] = mmp_nonmed

    _ATLAS_CACHE[key] = atlas
    return atlas


# =============================================================================
# P-VALUE CONVENTION
# =============================================================================

def p_from_nulls(
    r_emp: float,
    null_r: npt.NDArray,
    two_tailed: bool = False
) -> float:
    """
    Convert a null distribution of correlations into a p-value.

    Both conventions derive from the pooled null: because the two directions
    (permuting x, permuting y) contribute equally sized halves, the mean over the
    pooled array equals the average of the two per-direction means used by
    enigmatoolbox.perm_sphere_p.

    Args:
        r_emp: Empirical correlation
        null_r: Pooled null correlations
        two_tailed: If True, fraction of |null r| >= |r_emp|; if False, the
                    sign-directed one-sided fraction

    Returns:
        p-value
    """
    if two_tailed:
        return float(np.nanmean(np.abs(null_r) >= abs(r_emp)))
    if r_emp >= 0:
        return float(np.nanmean(null_r > r_emp))
    return float(np.nanmean(null_r < r_emp))


def _corr(a: npt.NDArray, b: npt.NDArray) -> float:
    """Pearson r dropping non-finite pairs."""
    ok = np.isfinite(a) & np.isfinite(b)
    return pearsonr(a[ok], b[ok])[0] if ok.sum() > 2 else np.nan


# =============================================================================
# SPIN PERMUTATION
# =============================================================================

def mmp_sphere_centroids(
    tpl_path: Union[str, Path] = DEFAULT_TPL_PATH
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    MMP parcel centroids on the native fsLR-32k sphere.

    Args:
        tpl_path: Template directory

    Returns:
        (coord_l, coord_r), each (180, 3): centroids for parcels 1-180 and 181-360
    """
    mmp_folder = Path(tpl_path) / 'MMP'
    atlas = load_mmp_atlas(tpl_path)

    coords = []
    for hemi, vtx_slice, parcel_ids in (
        ('L', slice(0, N_VTX_HEMI), range(1, 181)),
        ('R', slice(N_VTX_HEMI, N_VTX_FULL), range(181, 361)),
    ):
        sphere = nib.load(str(
            mmp_folder / f'Q1-Q6_RelatedParcellation210.{hemi}.sphere.32k_fs_LR.surf.gii'
        )).darrays[0].data
        atlas_hemi = atlas[vtx_slice]
        coords.append(np.vstack([
            sphere[atlas_hemi == p].mean(axis=0) for p in parcel_ids
        ]))

    return coords[0], coords[1]


def rotate_parcellation(
    coord_l: npt.NDArray,
    coord_r: npt.NDArray,
    n_rot: int = 2000,
    seed: Optional[int] = None
) -> npt.NDArray:
    """
    Spin-permutation indices from random rotations of parcel centroids.

    Applies a random rotation to the sphere, then matches each rotated centroid to an
    unrotated one without replacement, in order of "most distant minimum" (the hardest
    region to match is assigned first). The right hemisphere gets the same rotation
    reflected across the Y-Z plane, preserving hemispheric correspondence.

    Faithful to enigmatoolbox.rotate_parcellation, with an explicit RNG and a working
    identity-rotation rejection.

    Args:
        coord_l: Left hemisphere centroids (n_l, 3)
        coord_r: Right hemisphere centroids (n_r, 3)
        n_rot: Number of rotations
        seed: Random seed

    Returns:
        (n_l + n_r, n_rot) array of permuted parcel indices
    """
    rs = np.random.RandomState(seed)
    nroi_l, nroi_r = coord_l.shape[0], coord_r.shape[0]
    nroi = nroi_l + nroi_r

    perm_id = np.zeros((nroi, n_rot), dtype=int)
    reflect = np.diag([-1.0, 1.0, 1.0])
    r = 0

    while r < n_rot:
        # Random rotation via QR, forced to a proper rotation (det = +1)
        A = rs.normal(size=(3, 3))
        rot_l_mat, temp = np.linalg.qr(A)
        rot_l_mat = rot_l_mat @ np.diag(np.sign(np.diag(temp)))
        if np.linalg.det(rot_l_mat) < 0:
            rot_l_mat[:, 0] = -rot_l_mat[:, 0]
        rot_r_mat = reflect @ rot_l_mat @ reflect

        assignments = []
        for coord, n_roi, rot_mat in (
            (coord_l, nroi_l, rot_l_mat),
            (coord_r, nroi_r, rot_r_mat),
        ):
            dist = cdist(coord, coord @ rot_mat)
            ref_idx = np.zeros(n_roi, dtype=int)
            rot_idx = np.zeros(n_roi, dtype=int)

            # Assigned rows/columns are set to NaN; the all-NaN slices this creates
            # are expected and their warnings suppressed.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for i in range(n_roi):
                    ref_i = int(np.nanargmax(np.nanmin(dist, axis=1)))
                    rot_i = int(np.nanargmin(dist[ref_i, :]))
                    ref_idx[i], rot_idx[i] = ref_i, rot_i
                    dist[:, rot_i] = np.nan
                    dist[ref_i, :] = np.nan

            assignments.append((ref_idx, rot_idx))

        ref_lr = np.r_[assignments[0][0], nroi_l + assignments[1][0]]
        rot_lr = np.r_[assignments[0][1], nroi_l + assignments[1][1]]
        rot_sorted = rot_lr[np.argsort(ref_lr)]

        # Exclude the (rare) rotation mapping every parcel to itself
        if not bool(np.all(rot_sorted == np.arange(nroi))):
            perm_id[:, r] = rot_sorted
            r += 1

    return perm_id


def build_spin_perm_id(
    tpl_path: Union[str, Path] = DEFAULT_TPL_PATH,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    n_rot: int = 2000,
    seed: int = 0,
    use_cache: bool = True,
    verbose: bool = True
) -> npt.NDArray:
    """
    Build (and cache) MMP-360 spin permutations.

    Rotation takes ~40 s per 1000, so the result is cached and shared across callers.

    Args:
        tpl_path: Template directory
        cache_dir: Shared cache directory
        n_rot: Number of rotations
        seed: Random seed
        use_cache: Load/save the cached array
        verbose: Print cache activity

    Returns:
        (360, n_rot) array of permuted parcel indices
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f'spin_perm_id_mmp360_n{n_rot}_seed{seed}.npy'

    if use_cache and cache_file.exists():
        if verbose:
            print(f'  Loaded cached spin permutations: {cache_file.name}')
        return np.load(str(cache_file))

    if verbose:
        print(f'  Building {n_rot} spin permutations (MMP-360, fsLR-32k sphere)...')
    coord_l, coord_r = mmp_sphere_centroids(tpl_path)
    perm_id = rotate_parcellation(coord_l, coord_r, n_rot=n_rot, seed=seed)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_file), perm_id)
        if verbose:
            print(f'  Cached to {cache_file.name}')

    return perm_id


def spin_p(
    x: npt.NDArray,
    y: npt.NDArray,
    perm_id: npt.NDArray,
    mask: Optional[npt.NDArray] = None,
    two_tailed: bool = False,
    return_null: bool = False
) -> Union[float, Tuple[float, npt.NDArray]]:
    """
    Spin-permutation p-value for the correlation between two parcellated maps.

    Both maps must be FULL 360-parcel arrays: rotations are defined over the whole
    cortex, so a map has to be spun before it is restricted. Use `mask` to compare
    within a subset; parcels outside should be NaN, and NaN pairs produced by a
    rotation are dropped pairwise.

    Args:
        x, y: Full parcellated maps (360,)
        perm_id: (360, n_rot) from build_spin_perm_id()
        mask: Parcel indices to compare within (None = all 360)
        two_tailed: See p_from_nulls()
        return_null: Also return the pooled null correlations

    Returns:
        p, or (p, null_r) if return_null

    Raises:
        ValueError: If x or y is not a full parcellated map (e.g. pre-masked)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_parcels, n_rot = perm_id.shape

    if x.shape[0] != n_parcels or y.shape[0] != n_parcels:
        raise ValueError(
            f'x ({x.shape[0]}) and y ({y.shape[0]}) must be full {n_parcels}-parcel '
            f'maps. Pass full maps with NaN outside the ROI and use `mask` rather '
            f'than pre-masking - spinning is only defined over the whole cortex.'
        )

    sub = np.arange(n_parcels) if mask is None else np.asarray(mask)
    r_emp = _corr(x[sub], y[sub])

    r_xy = np.empty(n_rot)
    r_yx = np.empty(n_rot)
    for rr in range(n_rot):
        idx = perm_id[:, rr]
        r_xy[rr] = _corr(x[idx][sub], y[sub])   # spin the full map, then restrict
        r_yx[rr] = _corr(x[sub], y[idx][sub])

    null_r = np.r_[r_xy, r_yx]
    p_perm = p_from_nulls(r_emp, null_r, two_tailed=two_tailed)

    return (p_perm, null_r) if return_null else p_perm


# =============================================================================
# EIGENSTRAPPING
# =============================================================================

def build_surface_eigenmodes(
    tpl_path: Union[str, Path] = DEFAULT_TPL_PATH,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    num_modes: int = 200,
    surface: str = 'midthickness',
    use_cache: bool = True,
    verbose: bool = True
) -> Dict[str, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Compute (and cache) geometric eigenmodes of each cortical hemisphere.

    Modes are computed on the full cortex (medial wall excluded) so the same basis can
    be reused for ROI-restricted maps.

    Args:
        tpl_path: Template directory
        cache_dir: Shared cache directory
        num_modes: Eigenmodes per hemisphere
        surface: Surface geometry ('midthickness', 'white', 'pial')
        use_cache: Load/save cached eigenmodes
        verbose: Print progress

    Returns:
        {'L': (evals, emodes), 'R': (evals, emodes)}; evals (num_modes,),
        emodes (32492, num_modes) zero-padded across the medial wall

    Raises:
        ImportError: If eigenstrapping is not installed
    """
    if not HAS_EIGENSTRAPPING:
        raise ImportError(
            'eigenstrapping is required for build_surface_eigenmodes(). '
            'Install it, or use the spin test alone.'
        )

    tpl_path = Path(tpl_path)
    cache_dir = Path(cache_dir)
    mmp_folder = tpl_path / 'MMP'
    atlas = load_mmp_atlas(tpl_path)

    emode_store = {}
    for hemi, vtx_slice in (('L', slice(0, N_VTX_HEMI)),
                            ('R', slice(N_VTX_HEMI, N_VTX_FULL))):
        cache_file = cache_dir / f'emodes_{hemi}_{surface}_{num_modes}.npz'

        if use_cache and cache_file.exists():
            cached = np.load(str(cache_file))
            emode_store[hemi] = (cached['evals'], cached['emodes'])
            if verbose:
                print(f'  Loaded cached eigenmodes: {cache_file.name}')
            continue

        surf_file = str(
            mmp_folder /
            f'Q1-Q6_RelatedParcellation210.{hemi}.{surface}_MSMAll_2_d41_WRN_DeDrift'
            f'.32k_fs_LR.surf.gii'
        )
        cortex = (atlas[vtx_slice] > 0).astype(int)
        if verbose:
            print(f'  [{hemi}] computing {num_modes} eigenmodes on {surface} '
                  f'({cortex.sum()} cortical vertices)...')

        evals, emodes = calc_surface_eigenmodes(
            surf_file, cortex, save_cut=False, num_modes=num_modes
        )
        emode_store[hemi] = (evals, emodes)

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cache_file), evals=evals, emodes=emodes)
            if verbose:
                print(f'  Cached to {cache_file.name}')

    return emode_store


def eigenstrap_nulls(
    map_mmp: npt.NDArray,
    tpl_path: Union[str, Path],
    emode_store: Dict[str, Tuple[npt.NDArray, npt.NDArray]],
    parcels_used: npt.NDArray,
    n_surr: int = 2000,
    seed: int = 0,
    n_jobs: int = 1,
    verbose: bool = True
) -> npt.NDArray:
    """
    Generate eigenstrapping surrogates of a parcellated map.

    The parcel map is projected to vertices, surrogates are generated per hemisphere
    from the eigenmode basis, then averaged back within each parcel.

    Args:
        map_mmp: Full parcellated map (360,)
        tpl_path: Template directory
        emode_store: From build_surface_eigenmodes()
        parcels_used: 1-indexed MMP parcel IDs the map is defined on
        n_surr: Number of surrogates
        seed: Random seed (see module docstring - seeds numpy's GLOBAL RNG)
        n_jobs: Parallel workers; keep at 1 for reproducibility
        verbose: Print the n_jobs warning

    Returns:
        (n_surr, len(parcels_used)) re-parcellated surrogate maps
    """
    if not HAS_EIGENSTRAPPING:
        raise ImportError('eigenstrapping is required for eigenstrap_nulls().')

    atlas = load_mmp_atlas(tpl_path)

    used = np.zeros(N_PARCELS + 1, dtype=bool)
    used[parcels_used] = True

    if n_jobs != 1 and verbose:
        print(f'    Warning: n_jobs={n_jobs} makes eigenstrapping surrogates '
              f'non-reproducible from `seed`. Use n_jobs=1 for a deterministic null.')

    # Restore the caller's RNG state afterwards so seeding here cannot perturb
    # anything else in the pipeline that draws from np.random.
    rng_state_outer = np.random.get_state()

    hemi_out = {}
    for hemi_idx, (hemi, vtx_slice) in enumerate(
            (('L', slice(0, N_VTX_HEMI)), ('R', slice(N_VTX_HEMI, N_VTX_FULL)))):
        atlas_hemi = atlas[vtx_slice]
        evals, emodes = emode_store[hemi]

        # Distinct derived seed per hemisphere, so L and R do not receive the
        # identical sequence of mode rotations.
        np.random.seed((int(seed) * 2 + hemi_idx) % (2 ** 32 - 1))

        # Restrict to cortical vertices belonging to the parcels in use. Passing
        # full-cortex eigenmodes with this mask keeps the mode basis anchored to
        # whole-hemisphere geometry while fitting only inside the ROI.
        keep = used[atlas_hemi] & (atlas_hemi > 0)
        data = np.zeros(N_VTX_HEMI, dtype=float)
        data[keep] = map_mmp[atlas_hemi[keep] - 1]

        es = SurfaceEigenstrapping(
            data=data,
            evals=evals,
            emodes=emodes,
            medial=keep.astype(int),
            num_modes=emodes.shape[1],
            resample=True,       # preserve the empirical value distribution
            seed=seed,
            n_jobs=n_jobs,
        )
        hemi_out[hemi] = (np.atleast_2d(es(n=n_surr)), atlas_hemi, keep)

    np.random.set_state(rng_state_outer)

    # Re-parcellate: mean surrogate value across each parcel's vertices
    nulls = np.full((n_surr, len(parcels_used)), np.nan)
    for j, parcel in enumerate(parcels_used):
        hemi = 'L' if parcel <= 180 else 'R'
        surrs, atlas_hemi, keep = hemi_out[hemi]
        vtx_ids = np.where((atlas_hemi == parcel) & keep)[0]
        nulls[:, j] = np.nanmean(surrs[:, vtx_ids], axis=1)

    return nulls


def eigenstrap_p(
    x: npt.NDArray,
    y: npt.NDArray,
    tpl_path: Union[str, Path],
    emode_store: Dict[str, Tuple[npt.NDArray, npt.NDArray]],
    mask: Optional[npt.NDArray] = None,
    n_surr: int = 2000,
    seed: int = 0,
    n_jobs: int = 1,
    two_tailed: bool = False,
    return_null: bool = False,
    verbose: bool = True
) -> Union[float, Tuple[float, npt.NDArray]]:
    """
    Eigenstrapping p-value for the correlation between two parcellated maps.

    Mirrors spin_p(): both maps must be full 360-parcel arrays, `mask` restricts the
    comparison, and surrogates are generated for each map in turn so the two
    directions can be averaged under the same convention as the spin test.

    Args:
        x, y: Full parcellated maps (360,)
        tpl_path: Template directory
        emode_store: From build_surface_eigenmodes()
        mask: Parcel indices to compare within (None = all 360)
        n_surr: Surrogates per map
        seed: Random seed (y uses seed + 1)
        n_jobs: Parallel workers; keep at 1 for reproducibility
        two_tailed: See p_from_nulls()
        return_null: Also return the pooled null correlations
        verbose: Pass through to eigenstrap_nulls()

    Returns:
        p, or (p, null_r) if return_null

    Raises:
        ValueError: If x or y is not a full parcellated map
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape[0] != N_PARCELS or y.shape[0] != N_PARCELS:
        raise ValueError(
            f'x ({x.shape[0]}) and y ({y.shape[0]}) must be full {N_PARCELS}-parcel '
            f'maps. Pass full maps and use `mask` rather than pre-masking.'
        )

    sub = np.arange(N_PARCELS) if mask is None else np.asarray(mask)
    parcels_used = sub + 1   # eigenstrap_nulls expects 1-indexed parcel IDs

    # NaN outside the ROI would propagate through the eigenmode fit, so the surrogate
    # input carries zeros there; only `sub` is ever compared.
    x_in = np.nan_to_num(x, nan=0.0)
    y_in = np.nan_to_num(y, nan=0.0)

    nulls_x = eigenstrap_nulls(x_in, tpl_path, emode_store, parcels_used,
                               n_surr=n_surr, seed=seed, n_jobs=n_jobs, verbose=verbose)
    nulls_y = eigenstrap_nulls(y_in, tpl_path, emode_store, parcels_used,
                               n_surr=n_surr, seed=seed + 1, n_jobs=n_jobs,
                               verbose=verbose)

    r_emp = _corr(x[sub], y[sub])
    r_xy = np.array([_corr(nulls_x[i], y[sub]) for i in range(n_surr)])
    r_yx = np.array([_corr(x[sub], nulls_y[i]) for i in range(n_surr)])

    null_r = np.r_[r_xy, r_yx]
    p_perm = p_from_nulls(r_emp, null_r, two_tailed=two_tailed)

    return (p_perm, null_r) if return_null else p_perm


# =============================================================================
# CONVENIENCE
# =============================================================================

def compare_maps(
    x: npt.NDArray,
    y: npt.NDArray,
    label: str = 'comparison',
    mask: Optional[npt.NDArray] = None,
    tpl_path: Union[str, Path] = DEFAULT_TPL_PATH,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    n_rot: int = 2000,
    n_surr: int = 2000,
    num_modes: int = 200,
    surface: str = 'midthickness',
    seed: int = 0,
    n_jobs: int = 1,
    perm_id: Optional[npt.NDArray] = None,
    emode_store: Optional[Dict[str, Tuple[npt.NDArray, npt.NDArray]]] = None,
    run_eigenstrapping: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Correlate two parcellated maps and test it against both null models.

    Pass `perm_id` / `emode_store` when comparing several map pairs so the shared
    machinery is built once.

    Args:
        x, y: Full parcellated maps (360,), NaN outside the ROI
        label: Name recorded in the result row
        mask: Parcel indices to compare within (None = whole brain)
        tpl_path, cache_dir: Template and shared cache directories
        n_rot, n_surr, num_modes, surface, seed, n_jobs: Null-model parameters
        perm_id, emode_store: Prebuilt machinery, built here if omitted
        run_eigenstrapping: Set False to report the spin test alone
        verbose: Print progress

    Returns:
        Dict with label, n_parcels, pearson_r, p_parametric, spin_p, eigenstrap_p,
        the null SDs, and the parameters used. Suitable for pd.DataFrame([...]).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sub = np.arange(N_PARCELS) if mask is None else np.asarray(mask)

    ok = np.isfinite(x[sub]) & np.isfinite(y[sub])
    r_emp, p_param = pearsonr(x[sub][ok], y[sub][ok])

    if perm_id is None:
        perm_id = build_spin_perm_id(tpl_path, cache_dir, n_rot=n_rot, seed=seed,
                                     verbose=verbose)
    spin_pval, spin_null = spin_p(x, y, perm_id, mask=mask, return_null=True)

    eig_pval, eig_null = np.nan, np.array([np.nan])
    if run_eigenstrapping and HAS_EIGENSTRAPPING:
        if emode_store is None:
            emode_store = build_surface_eigenmodes(tpl_path, cache_dir,
                                                   num_modes=num_modes,
                                                   surface=surface, verbose=verbose)
        eig_pval, eig_null = eigenstrap_p(x, y, tpl_path, emode_store, mask=mask,
                                          n_surr=n_surr, seed=seed, n_jobs=n_jobs,
                                          return_null=True, verbose=verbose)

    if verbose:
        print(f'  {label}: r={r_emp:.4f} (n={int(ok.sum())})  '
              f'spin p={spin_pval:.4f}  eigenstrap p={eig_pval:.4f}')

    return {
        'comparison': label,
        'scope': 'whole_brain' if mask is None else 'within_mask',
        'n_parcels': int(ok.sum()),
        'pearson_r': float(r_emp),
        'p_parametric': float(p_param),
        'spin_p': float(spin_pval),
        'eigenstrap_p': float(eig_pval),
        'spin_null_sd': float(np.nanstd(spin_null)),
        'eigenstrap_null_sd': float(np.nanstd(eig_null)),
        'p_convention': 'one-sided (sign-directed)',
        'seed_surro': seed,
        'spin_n_rot': int(perm_id.shape[1]),
        'es_n_surr': int(n_surr) if run_eigenstrapping else 0,
        'es_num_modes': int(num_modes) if run_eigenstrapping else 0,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def save_sensitivity_table(
    df: pd.DataFrame,
    sens_dir: Union[str, Path],
    name: str,
    also_excel: bool = True,
    float_format: str = '%.6g'
) -> Path:
    """
    Save a sensitivity table as CSV, optionally with a matching .xlsx.

    CSV is the primary format: plain text, diff-able under version control, and
    readable without openpyxl or Excel.

    Note: columns using 'n/a' as a sentinel are read back as NaN by pandas' default
    NA handling. Use pd.read_csv(..., keep_default_na=False) to keep the literal text.

    Args:
        df: Table to save
        sens_dir: Output directory (created if absent)
        name: File stem, without extension
        also_excel: Also write {name}.xlsx alongside
        float_format: Numeric formatting for the CSV

    Returns:
        Path to the CSV file
    """
    sens_dir = Path(sens_dir)
    sens_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sens_dir / f'{name}.csv'
    df.to_csv(csv_path, index=False, float_format=float_format)

    if also_excel:
        df.to_excel(sens_dir / f'{name}.xlsx', index=False, engine='openpyxl')

    return csv_path
