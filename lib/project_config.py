"""
Shared path and parameter configuration.

Single source of truth for storage roots, derived-data locations, template files
and the manuscript parameter set.  Reads ``config/paths.yml`` and
``config/params.yml`` from the repository root.

Replaces the per-script storage-root definitions of the original working tree,
which hard-coded drive letters and mount points (``S:/``, ``Q:/``, ``V:/``,
``/MIPL/store7``).  Those are machine-specific and meaningless outside the lab
that produced the results.

Under the default ``config/paths.yml`` every path resolves to exactly what the
original scripts resolved to, so behaviour is unchanged; ``verify_paths.py``
checks this.

Usage
-----
    from project_config import PROJECT, TEMPLATES, PIPELINE, params, template

    weights = PIPELINE / 'encoding_parcel' / 'out'
    mmp     = template('mmp_32k_dlabel')
    bias    = params()['features']['bias_weight']

Scripts inside ``pipelines/<stage>/`` reach this module with::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lib'))
    from project_config import ...
"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to read config/paths.yml. "
        "Install the environment with `conda env create -f environment/environment.yml`."
    ) from exc


# =============================================================================
# LOCATE THE REPOSITORY
# =============================================================================

def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward until a directory containing config/paths.yml is found."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / 'config' / 'paths.yml').is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate config/paths.yml above {here}. "
        "project_config.py must stay inside the repository."
    )


REPO_ROOT: Path = _find_repo_root()

_PATHS_YML = REPO_ROOT / 'config' / 'paths.yml'
_PARAMS_YML = REPO_ROOT / 'config' / 'params.yml'


def _load(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not parse to a mapping.")
    return loaded


_PATHS: Dict[str, Any] = _load(_PATHS_YML)
_PARAMS: Dict[str, Any] = _load(_PARAMS_YML)


# =============================================================================
# ROOTS
# =============================================================================

def _resolve_root(value: Optional[str]) -> Optional[Path]:
    """Absolute paths as given; relative paths against the repository root."""
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


_ROOTS = _PATHS.get('roots', {})

#: Project root - the directory containing 0_data/, 1_code/, 2_pipeline/.
PROJECT: Path = _resolve_root(_ROOTS.get('project'))

#: Vendored atlases and surfaces (ships with this repository).
TEMPLATES: Path = _resolve_root(_ROOTS.get('templates'))

#: Derived data (Zenodo public tier + per-subject tier on request).
DERIVATIVES: Path = _resolve_root(_ROOTS.get('derivatives'))

#: Generated figures, one subdirectory per pipeline.
FIGURES: Path = _resolve_root(_ROOTS.get('figures'))

#: HBN phenotypic release (LORIS).  None unless configured; P7 needs it.
HBN_PHENOTYPE: Optional[Path] = _resolve_root(_ROOTS.get('hbn_phenotype'))

#: Local HBN imaging mirror.  None unless configured; preprocessing only.
HBN_MIRROR: Optional[Path] = _resolve_root(_ROOTS.get('hbn_mirror'))


def _expand(template_string: str) -> Path:
    return Path(template_string.format(project=str(PROJECT)))


_PROJECT_DIRS = _PATHS.get('project', {})

#: {project}/0_data/raw - stimuli and annotations (not redistributable).
RAW: Path = _expand(_PROJECT_DIRS['raw'])

#: {project}/1_code/99_main - the original working tree.
CODE: Path = _expand(_PROJECT_DIRS['code'])

#: {project}/2_pipeline/99_main - analysis outputs, one directory per task.
PIPELINE: Path = _expand(_PROJECT_DIRS['pipeline'])

#: Shared cache of spin permutations and geometric eigenmodes.
SPATIAL_NULLS_CACHE: Path = _expand(_PATHS['spatial_nulls_cache'])


# =============================================================================
# ACCESSORS
# =============================================================================

def params() -> Dict[str, Any]:
    """
    The manuscript configuration, as a nested dict (config/params.yml).

    These values are not tuning knobs.  Every script encodes them into its output
    directory path, so a changed value does not raise - it silently writes
    somewhere else and reads back nothing.
    """
    return _PARAMS


def task_dir(task: str) -> Path:
    """
    Output directory for a pipeline stage, e.g. ``task_dir('semantic_axis')``
    -> ``{project}/2_pipeline/99_main/07_pca``.

    Accepts either the logical name from ``pipeline_tasks`` in paths.yml or a
    literal directory name.
    """
    tasks = _PATHS.get('pipeline_tasks', {})
    return PIPELINE / tasks.get(task, task)


def fig_dir(script_file: str, create: bool = True) -> Path:
    """
    Figure output directory for the pipeline a script belongs to.

    Pass ``__file__``; the pipeline is taken from the enclosing directory under
    ``pipelines/``, so ``pipelines/p4_semantic_axis/07_pca.py`` writes to
    ``figures/p4_semantic_axis/``.

    ``create=False`` resolves the path without touching the filesystem - use it
    when only checking configuration, so that merely verifying the setup does
    not litter ``figures/`` with empty directories for pipelines that never
    produce any.
    """
    here = Path(script_file).resolve()
    parents = [q.name for q in here.parents]
    if 'pipelines' in parents:
        # the directory directly beneath pipelines/
        idx = parents.index('pipelines')
        name = here.parents[idx - 1].name if idx > 0 else here.parent.name
    else:
        name = here.parent.name
    out = FIGURES / name
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def template(key: str) -> Path:
    """
    Absolute path to a vendored template file, by key (see paths.yml).

    Raises with the available keys rather than returning a path that does not
    exist - a missing atlas otherwise surfaces much later as an empty result.
    """
    entries = _PATHS.get('templates', {})
    if key not in entries:
        raise KeyError(
            f"Unknown template key {key!r}. Available: {sorted(entries)}"
        )
    resolved = TEMPLATES / entries[key]
    if not resolved.exists():
        raise FileNotFoundError(
            f"Template {key!r} not found at {resolved}. "
            "Check roots.templates in config/paths.yml, and verify the vendored "
            "files with `cd templates && sha256sum -c SHA256SUMS.txt`."
        )
    return resolved


def require(root: Optional[Path], name: str, needed_for: str) -> Path:
    """
    Return a root that may be unconfigured, or fail with a message that says
    what to set and why.

    Used for the access-controlled roots, so a missing one is reported up front
    instead of surfacing as an empty glob deep inside an analysis.
    """
    if root is None:
        raise FileNotFoundError(
            f"roots.{name} is not set in config/paths.yml, but it is required "
            f"for {needed_for}. See docs/DERIVED_DATA_MANIFEST.md for what this "
            f"directory should contain."
        )
    return root


__all__ = [
    'REPO_ROOT', 'PROJECT', 'TEMPLATES', 'DERIVATIVES', 'FIGURES',
    'HBN_PHENOTYPE', 'HBN_MIRROR',
    'RAW', 'CODE', 'PIPELINE', 'SPATIAL_NULLS_CACHE',
    'params', 'task_dir', 'template', 'fig_dir', 'require',
]
