"""Per-fold voxel selection via 2-Gaussian R² mixture posterior.

Selects voxels where ``P(signal | r²) ≥ p_threshold`` (default 0.5)
under a fit_r2_mixture on the training R²s. Falls back to top-N by
R² when the mixture is degenerate (small/noisy ROIs).
"""
from __future__ import annotations

import numpy as np


def fdr_significant_voxels(train_data, train_pred, p_threshold=0.5,
                            min_voxels=100):
    """Voxels passing ``P(signal | r²) ≥ p_threshold``.

    Returns ``(keep, info)`` where ``keep`` is a sorted int array of
    voxel indices and ``info`` is a dict with mixture-fit diagnostics
    + the threshold + a fallback flag (top-N was used).
    """
    from braincoder.utils.stats import (
        fit_r2_mixture, r2_posterior_signal, r2_p_signal_threshold)

    td = np.asarray(train_data, dtype=np.float64)
    tp = np.asarray(train_pred, dtype=np.float64)
    ss_res = np.sum((td - tp) ** 2, axis=0)
    ss_tot = np.sum((td - td.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_safe = np.nan_to_num(r2, nan=-np.inf)

    fit, p_signal, threshold = None, None, float('inf')
    try:
        fit = fit_r2_mixture(r2)
        threshold = r2_p_signal_threshold(fit, p=p_threshold)
        p_signal = r2_posterior_signal(r2, fit)
        keep = np.where(np.isfinite(r2) & (p_signal >= p_threshold))[0]
    except ValueError:
        keep = np.where(np.isfinite(r2) & (r2 > threshold))[0]

    fallback = False
    if len(keep) < min_voxels:
        keep = np.argsort(-r2_safe)[:min_voxels]
        fallback = True

    info = dict(fit) if fit is not None else {}
    info.update(p_threshold=float(p_threshold),
                r2_threshold=float(threshold),
                n_kept=int(len(keep)),
                fallback=bool(fallback),
                r2=r2.astype(np.float32))
    if p_signal is not None:
        info['p_signal'] = p_signal.astype(np.float32)
    return np.sort(keep), info
