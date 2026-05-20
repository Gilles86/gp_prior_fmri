"""Decode test trials via posterior-mean over a stim grid.

Fits a residual noise model (multivariate Student-t with optional
distance-modulated Ω) on training residuals, then evaluates the
posterior over a stim grid for each held-out test trial.

The model used at decode time can be either the default ``LogGaussianPRF``
or any factory-built model (for adapters that use a different
encoding model — e.g., neural_priors' ``LinearScalingModel`` for
multi-range fits).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from braincoder.optimize import ResidualFitter


def _scalar(x):
    """Helper: mean-collapse to a single float (handles arrays + scalars)."""
    x = np.asarray(x)
    return float(x.mean()) if x.size else float('nan')


def decode_test_trials(
    params: pd.DataFrame,
    train_data: pd.DataFrame,
    train_par,
    test_data: pd.DataFrame,
    sig_voxels,
    stim_grid: np.ndarray,
    *,
    max_resid_iter: int = 2000,
    distance_matrix: Optional[np.ndarray] = None,
    use_wwt: bool = True,
    model_factory: Optional[Callable] = None,
):
    """Run the residual-fit + posterior-decode for one (fold, method, ω).

    Returns ``(decoded, omega_stats)``. ``decoded`` is None when
    there are no significant voxels.

    Parameters
    ----------
    params, sig_params : DataFrame
        Per-voxel encoding-model parameters; ``sig_params = params.iloc[sig_voxels]``.
    train_par : Series or DataFrame
        Paradigm for training trials. Passed to ``model_factory`` if
        given; otherwise wrapped as ``train_par.to_frame()`` for the
        default ``LogGaussianPRF``.
    sig_voxels : 1-D array of int
        Voxel indices that pass voxel selection.
    stim_grid : array
        Decoding grid (shape ``(n_grid,)`` for 1-D models, or
        ``(n_grid, n_dims)`` for multi-dim paradigms).
    distance_matrix : array or None
        Voxel-voxel geodesic distance matrix; when given, the
        residual fitter uses the distance-modulated Ω form.
    use_wwt : bool
        Include the σ²·WᵀW tuning-similarity term in Ω. Set False
        as a diagnostic to test redundancy with the distance term.
    model_factory : callable ``(train_par, sig_params) → model`` or None
        How to build the decoder model. Default uses ``LogGaussianPRF``;
        pass a factory for adapters that use a different model.
    """
    if len(sig_voxels) == 0:
        return None, {}

    sig_voxels = np.asarray(sig_voxels, dtype=int)
    sig_params = params.iloc[sig_voxels]
    train_data_sig = train_data.iloc[:, sig_voxels].astype(np.float32)
    test_data_sig = test_data.iloc[:, sig_voxels].astype(np.float32)

    if model_factory is None:
        from braincoder.models import LogGaussianPRF
        m = LogGaussianPRF(paradigm=train_par.to_frame(), parameters=sig_params)
    else:
        m = model_factory(train_par, sig_params)
    m.init_pseudoWWT(stim_grid, sig_params)

    residfit = ResidualFitter(m, train_data_sig, train_par,
                              parameters=sig_params)
    D_sig = None
    if distance_matrix is not None:
        D_sig = np.asarray(distance_matrix)[np.ix_(sig_voxels, sig_voxels)
                                            ].astype(np.float32)

    omega, dof = residfit.fit(
        init_sigma2=0.1, method='t', D=D_sig,
        max_n_iterations=max_resid_iter, learning_rate=0.05,
        use_wwt=use_wwt, progressbar=False)

    op_pars = dict(residfit.fitted_omega_parameters or {})
    omega_stats = {
        'omega_sigma2':   _scalar(op_pars.get('sigma2', np.nan)),
        'omega_rho':      _scalar(op_pars.get('rho',    np.nan)),
        'omega_alpha':    _scalar(op_pars.get('alpha',  np.nan)),
        'omega_beta':     _scalar(op_pars.get('beta',   np.nan)),
        'omega_dof':      _scalar(op_pars.get('dof', dof if dof is not None
                                                       else np.nan)),
        'omega_tau_mean': _scalar(op_pars.get('tau',    np.nan)),
    }

    pdf = m.get_stimulus_pdf(test_data_sig, stim_grid, sig_params,
                              omega=omega, dof=dof)
    cols = pdf.columns.astype(float).values
    decoded = (pdf.values * cols[None, :]).sum(axis=1) \
              / pdf.values.sum(axis=1)
    return decoded.astype(np.float32), omega_stats
