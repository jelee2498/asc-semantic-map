"""
SHARP (Split-HAlf RePeated) test for valid model comparison under cross-validation.

Implements the score-test variant of Zeng et al. (2026), "Widespread use of invalid
statistical tests in biomedical machine learning" (Supplementary Methods S7.1-S7.6).

Standard cross-validation produces fold-level performance estimates that are
statistically dependent (overlapping training sets), so paired t / Wilcoxon tests
inflate false positives — badly so under repeated CV. SHARP fixes this by, in each
of J repetitions, splitting the data into two DISJOINT halves A and B and running
K-fold CV separately within each half. The two per-half performance-difference
statistics are independent within a repetition (disjoint data) but correlated across
repetitions, which lets us estimate both the variance sigma^2 and the across-rep
correlation rho — the two quantities standard CV cannot disentangle.

This is a normally importable module (valid identifier), so joblib/loky workers can
pickle its functions by reference on any platform.

Public API
----------
sharp_score_test(D_A, D_B)      -> dict with Z, p, Dbar, sigma2_0, rho_0, ...
run_sharp(sem_feat, dim_feat, y_values, covariates, config) -> (results, per_rep)
    results: {'combined_vs_semantic': {...}, 'combined_vs_dimensionality': {...},
              'semantic_vs_dimensionality': {...}}
"""

import os
os.environ.setdefault("LOKY_PICKLER", "cloudpickle")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist, wilcoxon, binom
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **k):
        return x

# Each pair carries the directional alternative used when one-sided testing is on:
# 'greater' tests model1 > model2 (combined > each single-marker model, and dim > semantic).
_PAIRS = [
    ('combined', 'semantic', 'greater'),
    ('combined', 'dimensionality', 'greater'),
    ('dimensionality', 'semantic', 'greater'),
]


def _directional_p(p_two_sided, stat, alternative):
    """Convert a two-sided p-value to a one-sided one given the statistic's sign.
    'greater' (model1 > model2): p = p2/2 if stat>0 else 1 - p2/2. Valid because
    both the normal (SHARP) and t (corrected-t) null distributions are symmetric."""
    if not np.isfinite(p_two_sided) or not np.isfinite(stat):
        return p_two_sided
    if alternative == 'greater':
        return p_two_sided / 2.0 if stat > 0 else 1.0 - p_two_sided / 2.0
    if alternative == 'less':
        return p_two_sided / 2.0 if stat < 0 else 1.0 - p_two_sided / 2.0
    return p_two_sided


# =============================================================================
# SHARP score test (Supplementary Methods S7.1-S7.6)
# =============================================================================

def _build_Sigma(sigma2, rho, J):
    """Covariance of D = [D_A; D_B] (length 2J): sigma^2 [[M, C], [C, M]],
    M = 1 on diag / rho off-diag, C = 0 on diag / rho off-diag (Eq. S7.1)."""
    M = np.full((J, J), rho)
    np.fill_diagonal(M, 1.0)
    C = np.full((J, J), rho)
    np.fill_diagonal(C, 0.0)
    Sigma = np.block([[M, C], [C, M]])
    return sigma2 * Sigma


def _neg_loglik_null(theta, D, J):
    """Negative Gaussian log-likelihood under H0 (mu=0), params theta=[log sigma^2, rho]."""
    sigma2 = np.exp(theta[0])
    rho = theta[1]
    Sigma = _build_Sigma(sigma2, rho, J)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e12
    try:
        sol = np.linalg.solve(Sigma, D)
    except np.linalg.LinAlgError:
        return 1e12
    return 0.5 * logdet + 0.5 * float(D.dot(sol))


def sharp_score_test(D_A, D_B):
    """
    SHARP score test comparing two models from paired split-half statistics.

    D_A, D_B : length-J arrays of per-half performance differences (model1 - model2)
               for halves A and B across J repetitions.

    Returns a dict with the score-test Z, two-sided p-value, and nuisance estimates.
    """
    D_A = np.asarray(D_A, dtype=float)
    D_B = np.asarray(D_B, dtype=float)
    J = len(D_A)
    D = np.concatenate([D_A, D_B])
    Dbar = float(D.mean())

    # Method-of-moments initialisation (Eqs. S7.12-S7.16)
    SA2 = float(np.var(D_A, ddof=1))
    SB2 = float(np.var(D_B, ddof=1))
    sig2_delta = float(np.mean((D_A - D_B) ** 2) / 2.0)   # (1/2J) sum (D_Aj - D_Bj)^2
    if sig2_delta <= 0:
        return {'Z': np.nan, 'p': np.nan, 'Dbar': Dbar, 'sigma2_0': 0.0,
                'rho_0': np.nan, 'rho_mom': np.nan, 'var_dbar': np.nan, 'J': J}
    rho_mom = (sig2_delta - 0.5 * (SA2 + SB2)) / sig2_delta
    sig2_mom = sig2_delta

    # Estimate (sigma^2, rho) under the null by ML (score test, S7.6)
    x0 = [np.log(max(sig2_mom, 1e-8)), float(np.clip(rho_mom, -0.05, 0.95))]
    # rho bounds keep Sigma positive-definite: the SHARP covariance is PD only for
    # -1/(2J-1) < rho < 1/2 (its symmetric-subspace eigenvalue is 1 - 2 rho).
    res = minimize(
        _neg_loglik_null, x0, args=(D, J), method='L-BFGS-B',
        bounds=[(np.log(1e-10), np.log(1e6)), (-1.0 / (2 * J - 1) + 1e-4, 0.5 - 1e-4)],
    )
    sig2_0 = float(np.exp(res.x[0]))
    rho_0 = float(res.x[1])

    # Var(Dbar) = sigma^2 (1/2J + (J-1)/J rho)  (Eq. S7.8); Z = Dbar / sqrt(Var)  (Eq. S7.28)
    var_dbar = sig2_0 * (1.0 / (2 * J) + (J - 1.0) / J * rho_0)
    if var_dbar <= 0 or not np.isfinite(var_dbar):
        Z = np.nan
        p = np.nan
    else:
        Z = Dbar / np.sqrt(var_dbar)
        p = 2.0 * (1.0 - norm.cdf(abs(Z)))
    return {'Z': Z, 'p': p, 'Dbar': Dbar, 'sigma2_0': sig2_0, 'rho_0': rho_0,
            'rho_mom': rho_mom, 'var_dbar': var_dbar, 'J': J}


# =============================================================================
# Split-half repeated cross-validation (Pearson r, pooled within each half)
# =============================================================================

def _residualize(data, cov_train, cov_test):
    """Regress covariates out of `data` (fit on train rows), return residuals.
    `data` stacks train rows then test rows; cov_train/cov_test give the covariates."""
    n_tr = cov_train.shape[0]
    Xtr = np.column_stack([np.ones(n_tr), cov_train])
    Xte = np.column_stack([np.ones(cov_test.shape[0]), cov_test])
    is1d = data.ndim == 1
    if is1d:
        data = data.reshape(-1, 1)
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        reg = LinearRegression(fit_intercept=False).fit(Xtr, data[:n_tr, i])
        out[:n_tr, i] = data[:n_tr, i] - reg.predict(Xtr)
        out[n_tr:, i] = data[n_tr:, i] - reg.predict(Xte)
    return out.ravel() if is1d else out


def _extract(ft, sem_feat, dim_feat, idx):
    if ft == 'semantic':
        return sem_feat[idx]
    if ft == 'dimensionality':
        return dim_feat[idx]
    return np.concatenate([sem_feat[idx], dim_feat[idx]], axis=1)


def _fit_fold(Xtr, Xte, ytr, yte, cov_tr, cov_te):
    """Residualize (fit on train) -> z-score (fit on train) -> ElasticNetCV -> predict.
    Returns (y_pred_test, y_test_residualized). ElasticNetCV n_jobs=1 (outer loop parallel)."""
    Xall = _residualize(np.vstack([Xtr, Xte]), cov_tr, cov_te)
    Xtr, Xte = Xall[:len(ytr)], Xall[len(ytr):]
    yall = _residualize(np.concatenate([ytr, yte]), cov_tr, cov_te)
    ytr, yte = yall[:len(ytr)], yall[len(ytr):]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    Xtr[np.isnan(Xtr)] = 0; Xte[np.isnan(Xte)] = 0
    reg = ElasticNetCV(cv=5, random_state=0, l1_ratio=[.1, .3, .5, .7, .9],
                       max_iter=10000, n_jobs=1).fit(Xtr, ytr)
    return reg.predict(Xte), yte


def _safe_r(a, b):
    """Pearson r that returns 0.0 when either input has zero variance (e.g. an
    ElasticNet prediction that regularized to a constant on a small subsample) or is
    non-finite. A constant predictor has no correlation with the target, so 0 is the
    sensible value; this also avoids the np.corrcoef divide-by-zero RuntimeWarning."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return 0.0 if not np.isfinite(r) else float(r)


def _half_r(sub_idx, sem_feat, dim_feat, y_values, covariates, K, fold_seed):
    """K-fold CV within one half; pool test predictions across folds -> one Pearson r
    per model (combined/semantic/dimensionality), using identical folds for all models."""
    sub_idx = np.asarray(sub_idx)
    kf = KFold(n_splits=K, shuffle=True, random_state=fold_seed)
    preds = {ft: [] for ft in ('combined', 'semantic', 'dimensionality')}
    tests = {ft: [] for ft in ('combined', 'semantic', 'dimensionality')}
    for tr_loc, te_loc in kf.split(sub_idx):
        tr, te = sub_idx[tr_loc], sub_idx[te_loc]
        cov_tr, cov_te = covariates[tr], covariates[te]
        for ft in ('combined', 'semantic', 'dimensionality'):
            yp, yt = _fit_fold(_extract(ft, sem_feat, dim_feat, tr),
                               _extract(ft, sem_feat, dim_feat, te),
                               y_values[tr], y_values[te], cov_tr, cov_te)
            preds[ft].append(yp); tests[ft].append(yt)
    return {ft: _safe_r(np.concatenate(preds[ft]), np.concatenate(tests[ft]))
            for ft in ('combined', 'semantic', 'dimensionality')}


def _parallel(n, func, n_jobs, desc):
    """Version-agnostic parallel map with progress bar (mirrors _sharp files' helper)."""
    tasks = [delayed(func)(i) for i in range(n)]
    try:
        runner = Parallel(n_jobs=n_jobs, return_as="generator")
    except TypeError:
        runner = None
    if runner is not None:
        return list(tqdm(runner(tasks), total=n, desc=desc, leave=True))
    print(f"  {desc}: {n} tasks (n_jobs={n_jobs})...", flush=True)
    return Parallel(n_jobs=n_jobs)(tasks)


def run_sharp(sem_feat, dim_feat, y_values, covariates, config, desc='SHARP'):
    """
    Run the full SHARP procedure for one behavioral item and return score-test
    results for all three model pairs.

    sem_feat, dim_feat : (N, p) fixed-region feature matrices (already restricted).
    y_values           : (N,) behavioral scores.
    covariates         : (N, c) covariates residualized out within each fold.
    config             : needs 'sharp_K', 'sharp_J', 'seed', 'n_jobs'.
    """
    K = config['sharp_K']
    J = config['sharp_J']
    seed = config.get('seed', 0)
    n_jobs = config.get('n_jobs', -1)
    N = len(y_values)
    half = N // 2

    def _one_rep(j):
        rng = np.random.RandomState(seed + j)
        perm = rng.permutation(N)
        A = perm[:half]
        B = perm[half:2 * half]
        rA = _half_r(A, sem_feat, dim_feat, y_values, covariates, K, seed + j)
        rB = _half_r(B, sem_feat, dim_feat, y_values, covariates, K, seed + j)
        return rA, rB

    reps = _parallel(J, _one_rep, n_jobs, f'  {desc}')

    one_sided = config.get('one_sided_combined', True)
    results = {}
    for m1, m2, alt in _PAIRS:
        D_A = np.array([rA[m1] - rA[m2] for rA, rB in reps])
        D_B = np.array([rB[m1] - rB[m2] for rA, rB in reps])
        out = sharp_score_test(D_A, D_B)
        out['alternative'] = alt if one_sided else 'two-sided'
        out['p'] = _directional_p(out['p'], out['Z'], out['alternative'])
        results[f'{m1}_vs_{m2}'] = out
    return results, reps


# =============================================================================
# Corrected resampled t-test (Nadeau & Bengio, 2003; Supplementary Methods S3)
# =============================================================================

def corrected_resampled_ttest(fold_diffs, K, rho=None):
    """Corrected resampled t-test on per-fold performance differences.

    fold_diffs : per-fold (model1 - model2) differences, length J = R*K, from R
                 repetitions of K-fold CV.
    rho        : assumed between-fold correlation. If None (default), uses the
                 Nadeau-Bengio value rho = N2/(N1+N2) = 1/K, giving the correction
                 rho/(1-rho) = 1/(K-1). A SMALLER rho gives a smaller correction
                 (more lenient / higher power, higher FPR); a LARGER rho is more
                 conservative. Must satisfy 0 <= rho < 1.
    Statistic (Eq. S3.9): T = Dbar / sqrt((1/J + rho/(1-rho)) * S^2), df = J-1.
    """
    d = np.asarray(fold_diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    rho_eff = (1.0 / K) if rho is None else float(rho)
    if n < 2:
        return {'T': np.nan, 'p': np.nan, 'Dbar': (float(d.mean()) if n else np.nan),
                'df': max(n - 1, 0), 'n': n, 'rho_assumed': rho_eff}
    Dbar = float(d.mean())
    S2 = float(d.var(ddof=1))
    ratio = rho_eff / (1.0 - rho_eff)           # rho/(1-rho); equals 1/(K-1) when rho=1/K
    var_corr = (1.0 / n + ratio) * S2           # (1/J + N2/N1) * S^2
    if var_corr <= 0:
        return {'T': np.nan, 'p': np.nan, 'Dbar': Dbar, 'df': n - 1, 'n': n,
                'rho_assumed': rho_eff}
    T = Dbar / np.sqrt(var_corr)
    df = n - 1
    p = float(2.0 * t_dist.sf(abs(T), df))
    return {'T': T, 'p': p, 'Dbar': Dbar, 'df': df, 'n': n, 'rho_assumed': rho_eff}


def run_corrected_t(sem_feat, dim_feat, y_values, covariates, config, desc='CorrT'):
    """Repeated K-fold CV on the FULL sample (fixed features), then the corrected
    resampled t-test on per-fold Pearson-r differences for each model pair.

    Reuses config['sharp_K'] as K and config['sharp_J'] as the number of repetitions R
    (so K=10, J=30 gives 300 fold-level statistics, matching the paper's CorrT10-R).
    """
    K = config['sharp_K']
    R = config['sharp_J']
    rho = config.get('corrt_rho', None)
    seed = config.get('seed', 0)
    n_jobs = config.get('n_jobs', -1)
    N = len(y_values)

    def _one_rep(r):
        kf = KFold(n_splits=K, shuffle=True, random_state=seed + r)
        fold_r = {ft: [] for ft in ('combined', 'semantic', 'dimensionality')}
        for tr, te in kf.split(np.arange(N)):
            cov_tr, cov_te = covariates[tr], covariates[te]
            for ft in ('combined', 'semantic', 'dimensionality'):
                yp, yt = _fit_fold(_extract(ft, sem_feat, dim_feat, tr),
                                   _extract(ft, sem_feat, dim_feat, te),
                                   y_values[tr], y_values[te], cov_tr, cov_te)
                fold_r[ft].append(_safe_r(yp, yt))
        return fold_r

    reps = _parallel(R, _one_rep, n_jobs, f'  {desc}')

    one_sided = config.get('one_sided_combined', True)
    results = {}
    for m1, m2, alt in _PAIRS:
        diffs = [fr[m1][k] - fr[m2][k] for fr in reps for k in range(K)]
        out = corrected_resampled_ttest(diffs, K, rho=rho)
        out['alternative'] = alt if one_sided else 'two-sided'
        out['p'] = _directional_p(out['p'], out['T'], out['alternative'])
        results[f'{m1}_vs_{m2}'] = out
    return results, reps


# =============================================================================
# Combined model-comparison driver (SHARP + corrected resampled t-test)
# =============================================================================

def run_model_comparison(sem_feat, dim_feat, behav_df, item_list, covariates, config):
    """Run SHARP and the corrected resampled t-test for every behavioral item and
    model pair, print a combined table, and return result rows (for DataFrame/CSV).
    Both tests are FDR-corrected (Benjamini-Hochberg) across all item x pair
    comparisons, separately per test.
    """
    from statsmodels.stats.multitest import multipletests
    rows = []
    for item in item_list:
        y = behav_df[item].values
        sharp_res, _ = run_sharp(sem_feat, dim_feat, y, covariates, config, desc=f'SHARP {item}')
        corrt_res, _ = run_corrected_t(sem_feat, dim_feat, y, covariates, config, desc=f'CorrT {item}')
        for pair in sharp_res:
            s, c = sharp_res[pair], corrt_res[pair]
            rows.append({'item': item, 'comparison': pair,
                         'alternative': s.get('alternative', 'two-sided'),
                         'mean_diff_r': s['Dbar'], 'sharp_Z': s['Z'], 'sharp_rho': s['rho_0'],
                         'sharp_p': s['p'], 'corrT_T': c['T'], 'corrT_rho': c.get('rho_assumed'),
                         'corrT_p': c['p']})

    for pcol, fcol in (('sharp_p', 'sharp_p_fdr'), ('corrT_p', 'corrT_p_fdr')):
        fdr = multipletests([r[pcol] for r in rows], alpha=0.05, method='fdr_bh')[1]
        for i, r in enumerate(rows):
            r[fcol] = float(fdr[i])

    def _sig(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    _cr = config.get('corrt_rho', None)
    _rho_eff = (1.0 / config['sharp_K']) if _cr is None else float(_cr)
    print(f"\n  corrected-t assumed between-fold rho = {_rho_eff:.3f}"
          f"{' (default 1/K)' if _cr is None else ' (custom corrt_rho)'}")
    if config.get('one_sided_combined', True):
        print("  test direction: combined-vs-* one-sided (greater); "
              "semantic-vs-dimensionality two-sided")
    else:
        print("  test direction: all comparisons two-sided")
    print(f"\n{'Item':<10} | {'Comparison':<26} | {'Delta-r':>8} | {'SHARP Z':>7} | {'SHARP p':>8} | "
          f"{'SHARP pFDR':>12} | {'CorrT T':>7} | {'CorrT p':>8} | {'CorrT pFDR':>12}")
    print("-" * 124)
    for r in rows:
        it = r['item'].replace('SRS_', '').replace('_T', '')
        sfdr = f"{r['sharp_p_fdr']:.4f} {_sig(r['sharp_p_fdr'])}"
        cfdr = f"{r['corrT_p_fdr']:.4f} {_sig(r['corrT_p_fdr'])}"
        print(f"{it:<10} | {r['comparison']:<26} | {r['mean_diff_r']:>8.4f} | {r['sharp_Z']:>7.3f} | "
              f"{r['sharp_p']:>8.4f} | {sfdr:>12} | {r['corrT_T']:>7.3f} | "
              f"{r['corrT_p']:>8.4f} | {cfdr:>12}")
    print("=" * 124)
    n_s = sum(r['sharp_p_fdr'] < 0.05 for r in rows)
    n_c = sum(r['corrT_p_fdr'] < 0.05 for r in rows)
    print(f"Significant (FDR<0.05): SHARP {n_s}/{len(rows)}, corrected-t {n_c}/{len(rows)}")
    return rows


def cross_subscale_trend(rows, source='subscale-level Delta-r'):
    """Test whether one model *consistently* beats another ACROSS behavioral subscales.

    Operates on one summary Delta-r per subscale (from `rows['mean_diff_r']`) with the
    sign test and the one-sample Wilcoxon signed-rank test. This is an
    across-OUTCOME (Demsar-style) test: the units are subscales, not CV folds, and each
    subscale's Delta-r has already absorbed the CV fold dependence via SHARP — so it is
    NOT inflated by cross-validation overlap. It is, however, somewhat anti-conservative
    because the subscales are correlated (same subjects, related constructs); report the
    result as a directional trend. Direction matches each comparison's `alternative`
    (one-sided 'greater' for combined-vs-*, two-sided for semantic-vs-dimensionality).
    """
    from collections import OrderedDict
    by_comp = OrderedDict()
    for r in rows:
        by_comp.setdefault(r['comparison'],
                           {'d': [], 'alt': r.get('alternative', 'two-sided')})['d'].append(r['mean_diff_r'])

    trend = []
    for comp, v in by_comp.items():
        diffs = np.asarray(v['d'], dtype=float)
        alt = v['alt']
        n_total = len(diffs)
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n = n_pos + n_neg  # non-zero
        if n < 1:
            sign_p = np.nan
        elif alt == 'greater':
            sign_p = float(binom.sf(n_pos - 1, n, 0.5))          # P(X >= n_pos)
        elif alt == 'less':
            sign_p = float(binom.cdf(n_pos, n, 0.5))             # P(X <= n_pos)
        else:
            sign_p = float(min(1.0, 2.0 * min(binom.cdf(n_pos, n, 0.5),
                                              binom.sf(n_pos - 1, n, 0.5))))
        try:
            walt = alt if alt in ('greater', 'less') else 'two-sided'
            _, wilcox_p = wilcoxon(diffs, alternative=walt)
            wilcox_p = float(wilcox_p)
        except ValueError:
            wilcox_p = np.nan
        trend.append({'comparison': comp, 'alternative': alt, 'n_subscales': n_total,
                      'n_favoring_first': n_pos, 'mean_delta_r': float(diffs.mean()),
                      'sign_test_p': sign_p, 'wilcoxon_p': wilcox_p})

    # Benjamini-Hochberg FDR across the trend comparisons (separate family from the
    # per-subscale tests), applied within each test; NaNs are left uncorrected.
    from statsmodels.stats.multitest import multipletests

    def _fdr(pvals):
        p = np.asarray(pvals, dtype=float)
        out = np.full(len(p), np.nan)
        mask = np.isfinite(p)
        if mask.any():
            out[mask] = multipletests(p[mask], alpha=0.05, method='fdr_bh')[1]
        return out

    sign_fdr = _fdr([t['sign_test_p'] for t in trend])
    wilcox_fdr = _fdr([t['wilcoxon_p'] for t in trend])
    for i, t in enumerate(trend):
        t['sign_test_p_fdr'] = float(sign_fdr[i])
        t['wilcoxon_p_fdr'] = float(wilcox_fdr[i])

    print(f"\n  Cross-subscale directional trend (sign + Wilcoxon signed-rank on {source};")
    print(f"  BH-FDR across the {len(trend)} comparisons):")
    print(f"  {'Comparison':<28} | {'alt':>9} | {'favor':>6} | {'mean dr':>8} | "
          f"{'sign p':>8} | {'signFDR':>8} | {'wilcox p':>9} | {'wilcFDR':>8}")
    print("  " + "-" * 112)
    for t in trend:
        fav = f"{t['n_favoring_first']}/{t['n_subscales']}"
        print(f"  {t['comparison']:<28} | {t['alternative']:>9} | {fav:>6} | "
              f"{t['mean_delta_r']:>8.4f} | {t['sign_test_p']:>8.4f} | {t['sign_test_p_fdr']:>8.4f} | "
              f"{t['wilcoxon_p']:>9.4f} | {t['wilcoxon_p_fdr']:>8.4f}")
    print("  (across-subscale trend; suggestive - SRS subscales are correlated)")
    return trend


def _per_rep_wilcoxon_rate(all_results, item_list, pairs, one_sided):
    """Fraction of CV repetitions in which the cross-subscale Wilcoxon signed-rank test is
    significant (p < 0.05), per comparison. In each repetition the test is run on that
    repetition's per-subscale Delta-r. DESCRIPTIVE robustness diagnostic only: the
    repetitions are re-randomizations of the same data (dependent), so this is NOT a
    combined test and is deliberately left uncorrected."""
    n_rep = len(all_results[item_list[0]]['combined']['pearson'])
    rates = {}
    for m1, m2, alt in pairs:
        walt = alt if (one_sided and alt in ('greater', 'less')) else 'two-sided'
        n_sig, n_valid = 0, 0
        for j in range(n_rep):
            diffs = np.array([all_results[it][m1]['pearson'][j] - all_results[it][m2]['pearson'][j]
                              for it in item_list], dtype=float)
            diffs = diffs[np.isfinite(diffs)]
            if len(diffs) < 1:
                continue
            n_valid += 1
            try:
                _, p = wilcoxon(diffs, alternative=walt)
                if np.isfinite(p) and p < 0.05:
                    n_sig += 1
            except ValueError:
                pass  # degenerate repetition (e.g. all-zero differences)
        rates[f'{m1}_vs_{m2}'] = (n_sig / n_valid) if n_valid else np.nan
    return rates


def cross_subscale_trend_barplot(all_results, item_list, one_sided=True, central='median'):
    """Cross-subscale trend on the per-subscale difference of the models' cross-validated
    Pearson r, aggregated over repetitions by `central` ('median' by default, or 'mean').

    Uses the CV performance directly (all_results[item][model]['pearson']) rather than
    SHARP's split-half Delta-r: more intuitive but less conservative, so use it for the
    directional-trend statement only. Same sign + Wilcoxon signed-rank test. Note: the
    bar plot draws the MEAN, so with central='median' the trend Delta-r differs slightly
    from the bar heights (set central='mean' to match them exactly).
    """
    agg = np.nanmedian if central == 'median' else np.nanmean
    _pairs = [('combined', 'semantic', 'greater'),
              ('combined', 'dimensionality', 'greater'),
              ('dimensionality', 'semantic', 'greater')]
    rows = []
    for item in item_list:
        mr = {ft: float(agg(all_results[item][ft]['pearson']))
              for ft in ('combined', 'semantic', 'dimensionality')}
        for m1, m2, alt in _pairs:
            rows.append({'item': item, 'comparison': f'{m1}_vs_{m2}',
                         'alternative': (alt if one_sided else 'two-sided'),
                         'mean_diff_r': mr[m1] - mr[m2]})
    trend = cross_subscale_trend(rows, source=f'{central} CV-r Delta-r (across repetitions)')

    rate = _per_rep_wilcoxon_rate(all_results, item_list, _pairs, one_sided)
    print("  robustness - per-repetition Wilcoxon significance rate "
          "(uncorrected; NOT a combined test, repetitions are dependent):")
    for t in trend:
        t['wilcoxon_sig_rate'] = float(rate.get(t['comparison'], np.nan))
        rr = t['wilcoxon_sig_rate']
        rate_str = f"{rr * 100:5.1f}% of reps p<0.05" if np.isfinite(rr) else "n/a"
        print(f"    {t['comparison']:<28}: {rate_str}")
    return trend


# =============================================================================
# Self-test: null false-positive rate + a signal (power) sanity check
# =============================================================================

def _null_fpr_simulation(J=30, sigma=1.0, rho=0.5, n_sim=2000, mu=0.0, seed=0):
    """Sample D ~ N(mu*1, Sigma(sigma^2, rho)) and report rejection rate at alpha=0.05."""
    Sigma = _build_Sigma(sigma ** 2, rho, J)
    L = np.linalg.cholesky(Sigma)
    rng = np.random.RandomState(seed)
    rej = 0
    for _ in range(n_sim):
        D = mu + L.dot(rng.standard_normal(2 * J))
        out = sharp_score_test(D[:J], D[J:])
        if np.isfinite(out['p']) and out['p'] < 0.05:
            rej += 1
    return rej / n_sim


if __name__ == '__main__':
    print("SHARP score-test validation (analytic covariance simulation)")
    print("-" * 60)
    for rho in (0.0, 0.2, 0.4, 0.49):
        fpr = _null_fpr_simulation(J=30, rho=rho, mu=0.0, n_sim=3000, seed=1)
        print(f"  NULL (rho={rho:.2f}): FPR = {fpr:.3f}   (target ~0.05)")
    # power: nonzero mean should reject often
    for mu in (0.05, 0.1, 0.2):
        pw = _null_fpr_simulation(J=30, rho=0.4, mu=mu, n_sim=1500, seed=2)
        print(f"  SIGNAL (mu={mu:.2f}, rho=0.4): reject rate = {pw:.3f}")
    print("-" * 60)
    print("Done.")
