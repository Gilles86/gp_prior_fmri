# gp_prior_fmri

Cross-dataset GP-prior pRF fitting and decoding.

## What this is

A reproduction of [Daghlian et al. 2025](https://github.com/mdaghlian/braincoder_bprf)'s
hierarchical Bayesian pRF method — a Gaussian-Process prior over
pRF parameters indexed by cortical geodesic distance — generalized
from one neuroimaging dataset to many. The fitting pipeline is
written once; each dataset is wrapped in a small `DatasetAdapter`.

Current adapters:

- **`neural_priors`** — 39 subjects, 3T Philips Achieva, NPC numerosity (de Hollander, *in prep*)
- **`tms_risk`** — 55 subjects, 7T + cTBS, parietal magnitude × risk (de Hollander, Moisa & Ruff, *in prep*)

## Quick start

```bash
git clone git@github.com:Gilles86/gp_prior_fmri.git
cd gp_prior_fmri
conda env create -f environment_apple_silicon.yml
conda activate gp_prior_fmri
pip install -e .
```

Then fit one subject:

```bash
python -m gp_prior_fmri.modeling.fit 01 \
    --adapter neural_priors \
    --bids_folder /data/ds-neuralpriors \
    --joint_hyperparams --prior_params mu \
    --tag joint_mu
```

Cluster: see `gp_prior_fmri/modeling/slurm_jobs/fit.sh`.

## What's in this repo

```
gp_prior_fmri/
├── adapters/                # one module per neuroimaging dataset
│   ├── base.py              # DatasetAdapter ABC + LoadedFmriData
│   ├── neural_priors.py
│   └── tms_risk.py
├── modeling/                # the dataset-agnostic fit pipeline
│   ├── fit.py               # main entry point
│   └── slurm_jobs/
├── utils/                   # shared helpers
├── visualize/               # cross-dataset plotting / reports
└── notebooks/               # exploratory work (inside the package)

notes/
├── EXPERIMENTS.md           # canonical experiment registry
├── INDEX.md
├── STATUS.md
└── figures/

create_env/                  # cluster conda envs + sbatch wrappers
```

## Adding a new dataset

1. Subclass `DatasetAdapter` in `gp_prior_fmri/adapters/<dataset>.py`,
   implementing `get_subjects`, `load_data`, `cv_folds`.
2. Add the dataset to `_ADAPTERS` in
   `gp_prior_fmri/adapters/__init__.py`.
3. Optionally add a test in `tests/test_adapters.py` constructing the
   adapter and loading one subject.

The full fit pipeline then works on the new dataset with no
changes to `modeling/fit.py`.

## See also

- `CLAUDE.md` — developer recipes & gotchas
- `notes/EXPERIMENTS.md` — current experiment registry + findings
- [braincoder](https://github.com/Gilles86/braincoder) — the
  underlying encoding-model library
