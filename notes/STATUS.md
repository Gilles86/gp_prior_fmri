# Status

Last updated: 2026-05-20.

## Done

- Project scaffolded (`setup.py`, `gp_prior_fmri/`, conda envs, SLURM wrapper).
- `DatasetAdapter` ABC + `LoadedFmriData` NamedTuple.
- `neural_priors` adapter (CV: leave-one-(session, run2)-out).
- `tms_risk` adapter (CV: leave-one-run-out; default session = vertex/control).
- Ported `fit_gp_prior` from neural_priors into `gp_prior_fmri/modeling/fit.py`.
  Factored decoder, voxel selection, surface helpers, manifest into `utils/`.
- Moved `GP_EXPERIMENTS.md` here as `EXPERIMENTS.md`.

## In progress

- First end-to-end run on `tms_risk` (joint_mu recipe) — pending.

## Blocked / parked

- Pushing the `DatasetAdapter` interface upstream into
  `braincoder.utils.data`. Wait until the API settles after the
  tms_risk first-fits.

## Followups

- Tutorial notebook for the GP-prior pipeline.
- Drop the legacy `fit_gp_prior.py` in `neural_priors/encoding_model/`
  once the new pipeline has reproduced the indep_l / joint_mu cells
  byte-for-byte.
