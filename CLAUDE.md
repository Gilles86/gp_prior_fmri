# CLAUDE.md

Developer-facing recipes and gotchas for `gp_prior_fmri`.

## What this repo is

Cross-dataset GP-prior pRF fitting and decoding. One pipeline,
multiple neuroimaging datasets, plugged in via small adapters. The
fit code (`gp_prior_fmri/modeling/fit.py`) is dataset-agnostic; each
dataset (e.g., `neural_priors`, `tms_risk`) is wrapped in a
~80-line `DatasetAdapter` subclass under `gp_prior_fmri/adapters/`.

Started from the neural_priors `fit_gp_prior.py` (which itself
reproduces Daghlian et al. 2025 on numerosity NPC). Moved out because
the same pipeline turned out to be useful on multiple BIDS datasets,
and embedding it inside one of them was an awkward fit.

The intent is that the `DatasetAdapter` interface eventually moves
into `braincoder.utils.data` so any braincoder user can plug their
data in without touching this project.

## Paths

| Where | Path |
|---|---|
| Local repo | `~/git/gp_prior_fmri` |
| BIDS roots (per dataset) | `/data/ds-neuralpriors`, `/data/ds-tmsrisk` (local); `/shares/zne.uzh/gdehol/ds-{neuralpriors,tmsrisk}` (cluster) |
| GP-prior derivatives | `<bids>/derivatives/gp_prior_roi-{ROI}[.smoothed]/exp-{tag}/sub-NN[/ses-Y]/func/` |
| Cross-dataset registry | `notes/EXPERIMENTS.md` |

The pipeline writes back into each dataset's `derivatives/` rather
than into a central location, so individual subjects' fits live next
to the dataset they came from. The registry in `notes/EXPERIMENTS.md`
indexes them.

## Subject naming

Subject IDs are strings, zero-padded for `neural_priors` (`'01'`)
and 2-char-padded for `tms_risk`. Adapters accept either `int` or
`str` and normalize internally. The valid-subject list is per-adapter:

- `neural_priors`: 01–10, 12–22, 24–41 (39 valid; 11 + 23 don't exist)
- `tms_risk`: keys of `tms_risk/data/tms_keys.yml` (~55 subjects)

## Environment setup

Three conda envs, all defined in `create_env/` (mac env at top level):

| Env name | YML | Use case |
|---|---|---|
| `gp_prior_fmri` | `environment_apple_silicon.yml` | Local Mac dev |
| `gp_prior_fmri_cpu` | `create_env/environment_cpu.yml` | Cluster CPU jobs |
| `gp_prior_fmri_cuda` | `create_env/environment_cuda.yml` | Cluster GPU jobs (default for fits) |

Canonical stack (2026-05): Python 3.12 + Keras 3.13+ + TF 2.20.
Same as the parent dataset projects so braincoder behaves
identically. All three envs install `neural_priors` and `tms_risk`
editable so the adapters can find their wrapped `Subject` classes.

```bash
conda env create -f environment_apple_silicon.yml
conda activate gp_prior_fmri
pip install -e .
```

Cluster:
```bash
sbatch create_env/create_cpu_env.sh
sbatch create_env/create_gpu_env.sh    # must run on GPU node
```

## Pipeline stages

| Submodule | Inputs | Outputs |
|---|---|---|
| `adapters/` | nothing — pure code | n/a |
| `modeling/fit.py` | one adapter, one subject (+ session) | per-fold TSVs, manifest, parameter NIfTIs |
| `visualize/` | summary TSVs | PDFs in `notes/figures/` |

`modeling/` is the only stage; everything upstream (BIDS, fmriprep,
GLMsingle) is owned by the wrapped dataset projects.

## CLI examples

### Single subject, one cell

```bash
# neural_priors, joint_mu recipe (the current narrow-range winner)
python -m gp_prior_fmri.modeling.fit 01 \
    --adapter neural_priors \
    --bids_folder /data/ds-neuralpriors \
    --joint_hyperparams --prior_params mu \
    --tag joint_mu

# tms_risk, vertex session (default), same recipe
python -m gp_prior_fmri.modeling.fit 01 \
    --adapter tms_risk \
    --bids_folder /data/ds-tmsrisk \
    --joint_hyperparams --prior_params mu \
    --tag joint_mu_tms
```

### Debug (200 iters, first 2 folds only)

```bash
python -m gp_prior_fmri.modeling.fit 02 \
    --adapter neural_priors \
    --bids_folder /data/ds-neuralpriors \
    --joint_hyperparams --prior_params mu \
    --tag DEBUG --debug
```

## SLURM examples

```bash
cd ~/git/gp_prior_fmri/gp_prior_fmri/modeling/slurm_jobs

# neural_priors narrow, joint_mu
sbatch --array=1-10,12-22,24-41%10 fit.sh \
    neural_priors NPCr "--joint_hyperparams --prior_params mu --tag joint_mu"

# tms_risk, both unsmoothed + smoothed, joint_mu
sbatch --array=1-10%10 fit.sh \
    tms_risk NPCr "--joint_hyperparams --prior_params mu --tag joint_mu_tms"
sbatch --array=1-10%10 fit.sh \
    tms_risk NPCr "--joint_hyperparams --prior_params mu --smoothed --tag joint_mu_tms"
```

SLURM internals (account `zne.uzh`, conda activation, walltime, GPU
constraints) live in the **sciencecluster** skill — don't duplicate
here.

## Per-dataset gotchas

- **neural_priors**: stim_range column is mapped `{narrow: False, wide: True}` inside `get_paradigm`. CV unit is `(session, run2)` where `run2 = (run - 1) % 4 + 1`.
- **tms_risk**: each subject has two sessions in different TMS conditions (vertex/ips). Default session is *vertex* (control); pass `--session N` to override. Subject 10's session 1 only has 5 runs (not 6).
- Both: NPCr ROI lives at `<bids>/derivatives/ips_masks/sub-NN/anat/sub-NN_space-T1w_desc-NPCr_mask.nii.gz`.

## Registered experiments

See `notes/EXPERIMENTS.md` for the full table: one row per tag, with
priors-on / stage-2 mode / voxel selection / decoder ω / status / N
landed / key findings. Always **add a row before submitting a new
tag**. Per-subject `_desc-manifest.json` (machine-readable
provenance: git SHAs of `gp_prior_fmri` + `braincoder` + the source
dataset project, CLI args, run timestamps) lives next to every TSV.
