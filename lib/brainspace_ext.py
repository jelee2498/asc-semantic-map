"""PCA embedding and sign-only Procrustes alignment used by the P4 semantic axis.

The published analysis ran on a locally modified brainspace 0.1.20.  Released
brainspace cannot reproduce it, for three reasons:

1. ``GradientMaps(approach='pca')`` builds an affinity matrix first.  With
   ``sparsity=0`` the modified copy passed ``non_negative=False``, so PCA ran on
   the z-scored input itself; released brainspace sets negative values to zero.
   The modified copy also changed the default kernel from 'normalized_angle' to
   None (no kernel), so the input was never turned into a parcel x parcel
   similarity matrix.
2. The modified copy stored the PCA feature loadings as ``loadings_``.
3. The modified copy added ``only_sign`` alignment: each subject's components
   keep their own axes and only flip sign, and only when the component is
   negatively and significantly correlated with the reference (Pearson, FDR
   q < 0.01 across components).

This module reimplements those three pieces with numpy, scipy, scikit-learn and
statsmodels, so the pipelines run on released packages.  It is a faithful copy
of the modified code, including the Procrustes iteration of
``brainspace.gradient.alignment.procrustes_alignment``.
"""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection


def pca_embedding(
    x: npt.NDArray,
    n_components: int,
    random_state: int = 0
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """PCA of ``x`` as the modified ``GradientMaps(approach='pca', kernel=None)``
    with ``fit(x, sparsity=0)`` computed it: no kernel, no negative-value zeroing.

    Args:
        x: Input matrix, shape (n_samples, n_features)
        n_components: Number of components
        random_state: PCA random state

    Returns:
        Tuple of (gradients, lambdas, loadings): the component scores
        (n_samples, n_components), the explained variances (n_components,) and
        the feature loadings, i.e. ``PCA.components_``
        (n_components, n_features), as ``GradientMaps.loadings_`` stored them.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    gradients = pca.fit_transform(x)
    return gradients, pca.explained_variance_, pca.components_


def procrustes_only_sign(
    source: npt.NDArray,
    target: npt.NDArray,
    alpha: float = 0.01
) -> npt.NDArray:
    """Flip the sign of the components of ``source`` that are negatively and
    significantly correlated with the matching component of ``target``.

    Each column pair is tested with a Pearson correlation; the p-values are
    FDR-corrected (Benjamini-Hochberg) across columns at ``alpha``.  Columns that
    are not significant keep their sign whatever the direction of correlation.

    Args:
        source: Components to align, shape (n_samples, n_components)
        target: Reference components, same shape

    Returns:
        ``source`` with the selected columns multiplied by -1
    """
    tests = [pearsonr(t, s) for t, s in zip(target.T, source.T)]
    r = np.array([test[0] for test in tests])
    p = np.array([test[1] for test in tests])
    significant = fdrcorrection(p, alpha=alpha, method='i')[0]
    signs = np.where(significant & (r < 0), -1, 1)
    return source * signs


def procrustes_alignment(
    data: List[npt.NDArray],
    reference: npt.NDArray,
    n_iter: int = 10,
    tol: float = 1e-5,
    only_sign: bool = True
) -> Tuple[List[npt.NDArray], npt.NDArray]:
    """Iterative (generalised) Procrustes alignment to an initial reference.

    Each iteration aligns every dataset in ``data`` to the current reference and
    replaces the reference with the mean of the aligned datasets.  Iteration
    stops after ``n_iter`` passes or when the change in the squared distance
    between successive references falls below ``tol``.

    Args:
        data: Datasets to align, each (n_samples, n_components)
        reference: Initial reference, (n_samples, n_components)
        n_iter: Maximum number of iterations
        tol: Convergence tolerance
        only_sign: Sign flips only (:func:`procrustes_only_sign`); otherwise a
            full orthogonal Procrustes rotation

    Returns:
        Tuple of (aligned datasets, final reference)
    """
    if n_iter <= 0:
        raise ValueError('A positive number of iterations is required.')

    reference = reference.copy()
    aligned: List[npt.NDArray] = []
    dist = np.inf
    for _ in range(n_iter):
        if only_sign:
            aligned = [procrustes_only_sign(d, reference) for d in data]
        else:
            aligned = [_procrustes(d, reference) for d in data]

        new_reference = np.mean(aligned, axis=0)
        new_dist = np.square(reference - new_reference).sum()
        reference = new_reference

        if dist != np.inf and np.abs(new_dist - dist) < tol:
            break
        dist = new_dist

    return aligned, reference


def _procrustes(source: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
    """Orthogonal Procrustes (rotation + reflection) of ``source`` onto ``target``,
    as ``brainspace.gradient.alignment.procrustes`` with its defaults."""
    u, _, vt = np.linalg.svd(target.T.dot(source).T)
    return source.dot(u.dot(vt))


def align_to_reference(
    weight_list: List[npt.NDArray],
    reference: npt.NDArray,
    n_components: int = 10,
    n_iter: int = 10,
    only_sign: bool = True,
    random_state: int = 0
) -> List[npt.NDArray]:
    """Per-subject PCA followed by Procrustes alignment, as the modified
    ``GradientMaps(n_components, approach='pca', alignment='procrustes',
    only_sign=...)`` with ``fit(weight_list, reference=..., sparsity=0,
    n_iter=...)``.

    Args:
        weight_list: One (n_samples, n_features) matrix per subject
        reference: Initial reference, (n_samples, n_components)

    Returns:
        Aligned component scores, one (n_samples, n_components) array per subject
    """
    gradients = [pca_embedding(w, n_components, random_state)[0] for w in weight_list]
    aligned, _ = procrustes_alignment(gradients, reference, n_iter=n_iter,
                                      only_sign=only_sign)
    return aligned
