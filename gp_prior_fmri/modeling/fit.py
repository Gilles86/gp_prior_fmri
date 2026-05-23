"""Cross-validated LogGaussianPRF fit with optional GP prior on cortex.

Dataset-agnostic main entry point. The fit pipeline is the standard
Daghlian-style three-stage recipe (classical SSQ → MLE
hyperparameters or skipped → MAP), with options for joint type-II
MAP, shared lengthscale, and prior on any subset of parameters.

Per fold: classical / ML / bayes fits, voxel selection via the
fit_r2_mixture posterior, decode-test-trials with plain + distance ω.

Run via ``python -m gp_prior_fmri.modeling.fit <subject> --adapter ...``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import os.path as op
import pickle
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_PRIOR_PARAMS = ['mu', 'sd', 'amplitude', 'baseline']


# ---------------------------------------------------------------- prior build
def _build_priors(distance_matrix, classical_pars, prior_params):
    """One GeodesicGPPrior per name in ``prior_params``."""
    from braincoder.optimize.gp_prior import GeodesicGPPrior

    if not prior_params:
        return {}
    missing = [n for n in prior_params if n not in classical_pars.columns]
    if missing:
        raise ValueError(
            f"prior_params {missing} not in classical_pars columns "
            f"{list(classical_pars.columns)}")
    offdiag = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)]
    l0 = max(float(np.median(offdiag)) * 0.25, 1.0)
    priors = {}
    for name in prior_params:
        v = max(float(np.var(classical_pars[name].values)), 1e-4)
        priors[name] = GeodesicGPPrior(
            distance_matrix,
            lengthscale_init=l0, variance_init=v,
            nugget_init=max(v * 0.1, 1e-4))
    return priors


def _initial_pars(n_vx, paradigm_x):
    x = np.asarray(paradigm_x, dtype=np.float32)
    mu_init = float(x.mean())
    sd_init = max(float((x.max() - x.min()) / 4.0), 1.0)
    return pd.DataFrame({
        'mu':        np.full(n_vx, mu_init, dtype=np.float32),
        'sd':        np.full(n_vx, sd_init, dtype=np.float32),
        'amplitude': np.ones(n_vx, dtype=np.float32),
        'baseline':  np.zeros(n_vx, dtype=np.float32),
    })


# ---------------------------------------------------------------- per-fold fits
def _fit_classical(model, train_data, train_par, init_pars, max_iter,
                    no_early_stop=False, noise_model='ssq'):
    """Classical per-voxel fit via ``ParameterFitter``.

    Returns ``(pars, train_r2, loss_history)``. The loss history is
    the per-iteration value of the active loss (SSQ-sum or per-voxel
    Gaussian nLL); same length as ``r2_history_``.
    """
    from braincoder.optimize import ParameterFitter
    fitter = ParameterFitter(model, train_data, train_par, log_dir=False)
    kwargs = dict(init_pars=init_pars, max_n_iterations=max_iter,
                   progressbar=False, noise_model=noise_model)
    if no_early_stop:
        kwargs['min_n_iterations'] = max_iter
    pars = fitter.fit(**kwargs)
    return pars, float(fitter.r2.mean()), fitter.loss_history_


def _fit_ml(model, train_data, train_par, init_pars, max_iter):
    """Returns ``(map_estimates, sigma, loss_history)``."""
    from braincoder.optimize.bayesian_fitter import BayesianParameterFitter
    fitter = BayesianParameterFitter(model, train_data, train_par, priors={})
    fitter.classical_estimates = init_pars
    fitter.fit_map(max_n_iterations=max_iter, init_pars=init_pars,
                    progressbar=False)
    return fitter.map_estimates, fitter.map_sigma, fitter.map_history


def _fit_bayes(model, train_data, train_par, D, classical_pars,
                max_iter, prior_params, joint_hyperparams, shared_lengthscale):
    """Returns ``(map_estimates, sigma, hyperpars, loss_history)``."""
    from braincoder.optimize.bayesian_fitter import BayesianParameterFitter
    priors = _build_priors(D, classical_pars, prior_params)
    fitter = BayesianParameterFitter(model, train_data, train_par, priors=priors)
    fitter.classical_estimates = classical_pars
    if joint_hyperparams:
        if shared_lengthscale:
            fitter.tie_lengthscales()
    else:
        fitter.fit_hyperparameters(progressbar=False,
                                    shared_lengthscale=shared_lengthscale)
    fitter.fit_map(max_n_iterations=max_iter, progressbar=False,
                    joint_hyperparams=joint_hyperparams)
    hp = {name: priors[name].hyperparameters for name in prior_params}
    return fitter.map_estimates, fitter.map_sigma, hp, fitter.map_history


# ---------------------------------------------------------------- main
def main(subject, adapter_name='neural_priors', bids_folder=None,
         session=None, roi='NPCr', smoothed=False,
         tag='default', max_iter=2000, debug=False, output_dir=None,
         prior_params=None,
         joint_hyperparams=False, shared_lengthscale=False, use_wwt=True,
         no_early_stop_classical=False,
         classical_noise_model='ssq',
         **adapter_kwargs):
    """Run the full GP-prior pipeline for one subject (+ optional session)."""

    print(f'[fit] subject={subject} adapter={adapter_name} '
          f'session={session} roi={roi} smoothed={smoothed} tag={tag}',
          flush=True)

    # Heavy imports here (keeps `python -m gp_prior_fmri.modeling.fit --help`
    # snappy and avoids GPU init before argparse error paths).
    from braincoder.models import LogGaussianPRF
    from braincoder.utils import get_rsq

    from gp_prior_fmri.adapters import get_adapter
    from gp_prior_fmri.modeling.decode import decode_test_trials
    from gp_prior_fmri.utils.surface import (
        voxel_centroids_mm, load_white_surface,
        cortical_distance_matrix, roi_to_hemi_letter)
    from gp_prior_fmri.utils.voxel_selection import fdr_significant_voxels
    from gp_prior_fmri.utils.manifest import git_sha, write_manifest

    if debug:
        max_iter = 200
    if prior_params is None:
        prior_params = list(DEFAULT_PRIOR_PARAMS)

    adapter = get_adapter(adapter_name,
                           bids_folder=bids_folder
                           or _default_bids(adapter_name))
    if session is None:
        session = adapter.default_session(str(subject))

    # ---- load ----
    bundle = adapter.load_data(str(subject), session=session, roi=roi,
                                smoothed=smoothed, **adapter_kwargs)
    paradigm, data, masker, xyz, sub = bundle

    # For 1-D LogGaussianPRF we expect a single stim-value column.
    if isinstance(paradigm, pd.DataFrame) and 'range' in paradigm.columns:
        # neural_priors paradigm has both 'x' and 'range'; collapse to 'x'.
        paradigm_x = paradigm['x']
    elif isinstance(paradigm, pd.DataFrame):
        # Single-column DataFrame.
        paradigm_x = paradigm.iloc[:, 0]
    else:
        paradigm_x = paradigm
    paradigm_x = paradigm_x.astype(np.float32)
    paradigm_x.name = paradigm_x.name or 'x'

    n_vx = data.shape[1]
    print(f'[fit] {n_vx} voxels, {data.shape[0]} trials')

    init = _initial_pars(n_vx, paradigm_x)
    model = LogGaussianPRF(paradigm=paradigm_x.to_frame())

    # ---- geodesic distance matrix ----
    hemi = roi_to_hemi_letter(roi)
    vertices, faces = load_white_surface(
        adapter.get_white_surface_path(str(subject), hemi))
    D, vtx_idx, snap_dist = cortical_distance_matrix(
        xyz, vertices, faces, progressbar=not debug)
    print(f'[fit] snap median {np.median(snap_dist):.2f} mm | '
          f'D shape {D.shape}, median off-diag '
          f'{np.median(D[D > 0]):.1f} mm')
    if snap_dist.max() > 5.0:
        print('WARNING: max snap > 5 mm — check coregistration')

    # ---- output dir + manifest ----
    if output_dir is None:
        output_dir = adapter.output_dir(
            adapter.bids_folder, roi, tag, smoothed,
            str(subject), session)
    os.makedirs(output_dir, exist_ok=True)

    import braincoder as _bc
    import gp_prior_fmri as _gpf
    manifest = {
        'subject':           str(subject),
        'session':           None if session is None else str(session),
        'adapter':           adapter_name,
        'roi':               roi,
        'smoothed':          bool(smoothed),
        'tag':               tag,
        'shared_lengthscale': bool(shared_lengthscale),
        'joint_hyperparams': bool(joint_hyperparams),
        'use_wwt':           bool(use_wwt),
        'no_early_stop_classical': bool(no_early_stop_classical),
        'classical_noise_model':   classical_noise_model,
        'max_iter':          int(max_iter),
        'debug':             bool(debug),
        'prior_params':      list(prior_params),
        'voxel_selection':   'per_fold_p_signal>=0.5',
        'git_gp_prior_fmri': git_sha(op.dirname(_gpf.__file__)),
        'git_braincoder':    git_sha(op.dirname(_bc.__file__)),
        'run_started':       datetime.datetime.utcnow().isoformat() + 'Z',
        'output_dir':        output_dir,
    }
    write_manifest(output_dir, str(subject), session, manifest)

    suffix = f'_ses-{session}' if session is not None else ''
    np.save(op.join(output_dir,
                     f'sub-{subject}{suffix}_desc-distance.npy'), D)
    np.save(op.join(output_dir,
                     f'sub-{subject}{suffix}_desc-vertex_idx.npy'), vtx_idx)

    # ---- CV loop ----
    folds = adapter.cv_folds(paradigm)
    if debug:
        folds = folds[:2]
        print(f'[fit] DEBUG → first {len(folds)} folds only')

    fold_results = []
    for fold in folds:
        print(f'\n=== Fold {fold} ===')
        test_data = data.loc[fold].astype(np.float32)
        test_par_full = (paradigm.loc[fold] if isinstance(paradigm,
                                                            pd.DataFrame)
                          else paradigm_x.loc[fold])
        train_data = data.drop(fold)
        train_par_full = (paradigm.drop(fold) if isinstance(paradigm,
                                                              pd.DataFrame)
                           else paradigm_x.drop(fold))
        # Use the collapsed 1-D x for the model.
        test_par = paradigm_x.loc[fold].astype(np.float32)
        train_par = paradigm_x.drop(fold).astype(np.float32)

        cls_pars, cls_train_r2, cls_loss = _fit_classical(
            model, train_data, train_par, init, max_iter,
            no_early_stop=no_early_stop_classical,
            noise_model=classical_noise_model)
        cls_test_pred  = model.predict(parameters=cls_pars,
                                         paradigm=test_par.to_frame())
        cls_train_pred = model.predict(parameters=cls_pars,
                                         paradigm=train_par.to_frame())
        cls_cvr2 = get_rsq(test_data, cls_test_pred)

        ml_pars, ml_sigma, ml_loss = _fit_ml(model, train_data, train_par,
                                     init, max_iter)
        ml_test_pred  = model.predict(parameters=ml_pars,
                                        paradigm=test_par.to_frame())
        ml_train_pred = model.predict(parameters=ml_pars,
                                        paradigm=train_par.to_frame())
        ml_cvr2 = get_rsq(test_data, ml_test_pred)

        map_pars, sigma, hp, bayes_loss = _fit_bayes(
            model, train_data, train_par, D, cls_pars,
            max_iter, prior_params, joint_hyperparams, shared_lengthscale)
        map_test_pred  = model.predict(parameters=map_pars,
                                         paradigm=test_par.to_frame())
        map_train_pred = model.predict(parameters=map_pars,
                                         paradigm=train_par.to_frame())
        map_cvr2 = get_rsq(test_data, map_test_pred)

        # Voxel selection + per-omega decoding.
        stim_grid = np.linspace(
            float(paradigm_x.min()), float(paradigm_x.max()), 201,
            dtype=np.float32)
        true_test = test_par.values.astype(np.float32)
        decoding = {}
        decode_iter = 200 if debug else 2000
        for method, train_pred, fit_pars in (
                ('classical', cls_train_pred, cls_pars),
                ('ml',        ml_train_pred,  ml_pars),
                ('bayes',     map_train_pred, map_pars)):
            sig, fdr_info = fdr_significant_voxels(
                train_data.values, train_pred.values)
            for ω, D_arg in (('plain', None), ('distance', D)):
                decoded, omega_stats = decode_test_trials(
                    fit_pars, train_data, train_par, test_data,
                    sig, stim_grid, max_resid_iter=decode_iter,
                    distance_matrix=D_arg, use_wwt=use_wwt)
                key = (method, ω)
                if decoded is None:
                    decoding[key] = dict(
                        n_sig=0, mae=np.nan, median_ae=np.nan,
                        mae_log=np.nan, median_ae_log=np.nan, r=np.nan,
                        decoded=None, fdr_info=fdr_info,
                        omega_stats=omega_stats)
                    continue
                err = np.abs(decoded - true_test)
                err_log = np.abs(np.log(np.clip(decoded, 1e-6, None))
                                  - np.log(np.clip(true_test, 1e-6, None)))
                r_fold = (float('nan')
                          if (np.std(decoded) < 1e-9
                              or np.std(true_test) < 1e-9
                              or len(decoded) < 3)
                          else float(np.corrcoef(decoded, true_test)[0, 1]))
                decoding[key] = dict(
                    n_sig=int(len(sig)),
                    mae=float(err.mean()), median_ae=float(np.median(err)),
                    mae_log=float(err_log.mean()),
                    median_ae_log=float(np.median(err_log)),
                    r=r_fold, decoded=decoded, true=true_test,
                    fdr_info=fdr_info, omega_stats=omega_stats)

        print(f'  classical train R² {cls_train_r2:+.3f}')
        for name, cvr2 in (('classical', cls_cvr2), ('ml', ml_cvr2),
                            ('bayes', map_cvr2)):
            p = decoding.get((name, 'plain'), {})
            d = decoding.get((name, 'distance'), {})
            print(f'  {name:9s}: cvR² {float(cvr2.mean()):+.3f} | '
                  f'p r {p.get("r", float("nan")):+.3f} | '
                  f'd r {d.get("r", float("nan")):+.3f}')
        for name, h in hp.items():
            print(f'    prior[{name}]: l={h["lengthscale"]:.2f} mm '
                  f'v={h["variance"]:.3f} nug={h["nugget"]:.3f}')

        fold_results.append({
            'fold': fold,
            'classical_params': cls_pars, 'ml_params': ml_pars,
            'bayes_params': map_pars,
            'classical_cvr2': cls_cvr2, 'ml_cvr2': ml_cvr2,
            'bayes_cvr2': map_cvr2,
            'hyperparameters': hp,
            'ml_sigma': ml_sigma, 'bayes_sigma': sigma,
            'decoding': decoding,
            'loss_history': {
                'classical': np.asarray(cls_loss, dtype=np.float64),
                'ml':        np.asarray(ml_loss,  dtype=np.float64),
                'bayes':     np.asarray(bayes_loss, dtype=np.float64),
            },
        })

    # ---- write outputs ----
    _write_outputs(output_dir, str(subject), session, fold_results, roi)

    manifest['run_finished'] = datetime.datetime.utcnow().isoformat() + 'Z'
    write_manifest(output_dir, str(subject), session, manifest)
    print(f'\nWrote outputs to {output_dir}')


def _write_outputs(output_dir, subject, session, fold_results, roi):
    """Pickle fold dicts + write cvR², hyperpars, decoding TSVs."""
    suffix = f'_ses-{session}' if session is not None else ''

    with open(op.join(output_dir,
                       f'sub-{subject}{suffix}_desc-folds.pkl'),
              'wb') as f:
        pickle.dump({'subject': subject, 'session': session,
                     'roi': roi, 'folds': fold_results}, f)

    # cvR² long-form
    rows = []
    for r in fold_results:
        for method, cvr2 in (('classical', r['classical_cvr2']),
                              ('ml', r['ml_cvr2']),
                              ('bayes', r['bayes_cvr2'])):
            for vox, val in cvr2.items():
                rows.append(dict(fold=str(r['fold']),
                                  voxel=int(vox), method=method,
                                  cvr2=float(val)))
    pd.DataFrame(rows).to_csv(op.join(
        output_dir, f'sub-{subject}{suffix}_desc-cvr2.tsv'),
        sep='\t', index=False)

    # Hyperparams
    hp_rows = []
    for r in fold_results:
        for pname, h in r['hyperparameters'].items():
            hp_rows.append(dict(fold=str(r['fold']),
                                 parameter=pname, **h))
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv(op.join(
            output_dir,
            f'sub-{subject}{suffix}_desc-hyperpars.tsv'),
            sep='\t', index=False)

    # Loss history: long-form per (fold, method, step) → loss. One TSV
    # per subject so we can inspect convergence across methods and folds.
    loss_rows = []
    for r in fold_results:
        for method, arr in r['loss_history'].items():
            for step, val in enumerate(arr):
                loss_rows.append(dict(fold=str(r['fold']),
                                       method=method,
                                       step=int(step),
                                       loss=float(val)))
    if loss_rows:
        pd.DataFrame(loss_rows).to_csv(op.join(
            output_dir,
            f'sub-{subject}{suffix}_desc-loss_history.tsv'),
            sep='\t', index=False)

    # Decoding long-form
    dec = []
    for r in fold_results:
        for (method, ω), d in r['decoding'].items():
            info = d.get('fdr_info', {}) or {}
            o = d.get('omega_stats', {}) or {}
            dec.append(dict(
                fold=str(r['fold']),
                method=method, omega=ω,
                n_sig_voxels=d['n_sig'],
                mae=d['mae'], median_ae=d['median_ae'],
                mae_log=d.get('mae_log', np.nan),
                median_ae_log=d.get('median_ae_log', np.nan),
                r=d.get('r', np.nan),
                fdr_fallback=info.get('fallback', False),
                fdr_r2_threshold=info.get('r2_threshold', np.nan),
                fdr_noise_mean_r2=info.get('noise_mean_r2', np.nan),
                fdr_signal_mean_r2=info.get('signal_mean_r2', np.nan),
                fdr_signal_weight=info.get('signal_weight', np.nan),
                omega_sigma2=o.get('omega_sigma2', np.nan),
                omega_rho=o.get('omega_rho', np.nan),
                omega_alpha=o.get('omega_alpha', np.nan),
                omega_beta=o.get('omega_beta', np.nan),
                omega_dof=o.get('omega_dof', np.nan),
                omega_tau_mean=o.get('omega_tau_mean', np.nan)))
    pd.DataFrame(dec).to_csv(op.join(
        output_dir, f'sub-{subject}{suffix}_desc-decoding.tsv'),
        sep='\t', index=False)


def _default_bids(adapter_name: str) -> str:
    return {
        'neural_priors': '/data/ds-neuralpriors',
        'tms_risk':      '/data/ds-tmsrisk',
    }.get(adapter_name, '/data')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subject', type=str)
    p.add_argument('--adapter', default='neural_priors',
                    choices=['neural_priors', 'tms_risk'])
    p.add_argument('--bids_folder', default=None)
    p.add_argument('--session', default=None)
    p.add_argument('--roi', default='NPCr')
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--tag', default='default')
    p.add_argument('--prior_params', nargs='+', default=None,
                    help='Subset of [mu, sd, amplitude, baseline]. '
                         'Default = all four.')
    p.add_argument('--joint_hyperparams', action='store_true',
                    help='Type-II MAP: co-fit GP hyperparams with model '
                         'params in stage 3.')
    p.add_argument('--shared_lengthscale', action='store_true')
    p.add_argument('--no_wwt', dest='use_wwt', action='store_false')
    p.add_argument('--no_early_stop_classical', action='store_true',
                    help='Disable classical ParameterFitter R²-plateau early '
                         'stop by setting min_n_iterations=max_iter. '
                         'Diagnostic for whether classical was under-converged '
                         'in the joint_mu_tms vs ML comparison.')
    p.add_argument('--classical_noise_model', default='ssq',
                    choices=['ssq', 'gaussian'],
                    help='Noise model for the classical ParameterFitter loss. '
                         'New in braincoder 0.6: gaussian uses a per-voxel '
                         'σ²ᵥ MLE, equivalent at convergence to ssq but much '
                         'faster on heteroskedastic data. Will become the '
                         'braincoder default in 0.7.')
    # neural_priors-specific:
    p.add_argument('--stim_range', default='both',
                    choices=['narrow', 'wide', 'both'])
    p.add_argument('--max_iter', type=int, default=2000)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--output_dir', default=None)
    a = p.parse_args()

    extra = {}
    if a.adapter == 'neural_priors':
        extra['stim_range'] = a.stim_range

    main(a.subject, adapter_name=a.adapter, bids_folder=a.bids_folder,
         session=a.session, roi=a.roi, smoothed=a.smoothed, tag=a.tag,
         max_iter=a.max_iter, debug=a.debug, output_dir=a.output_dir,
         prior_params=a.prior_params,
         joint_hyperparams=a.joint_hyperparams,
         shared_lengthscale=a.shared_lengthscale,
         use_wwt=a.use_wwt,
         no_early_stop_classical=a.no_early_stop_classical,
         classical_noise_model=a.classical_noise_model,
         **extra)
