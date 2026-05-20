"""tms_risk adapter.

Wraps ``tms_risk.utils.Subject`` and the GLMsingle single-trial
estimates pipeline. Paradigm is the single ``n1`` numerosity column;
each (subject, session) is fit independently, and CV is
leave-one-run-out within a session.

``default_session(subject)`` returns the subject's *vertex* (control)
session, looked up in ``tms_risk/data/tms_keys.yml`` — so a no-flag
fit gives the cleanest baseline uncontaminated by parietal TMS.
"""
from __future__ import annotations

from importlib.resources import files
import os.path as op
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
import pandas as pd
import yaml

from .base import DatasetAdapter, LoadedFmriData


def _voxel_centroids_mm(masker):
    mask_img = masker.mask_img_
    affine = mask_img.affine
    mask_data = mask_img.get_fdata().astype(bool)
    ijk = np.argwhere(mask_data)
    return nib.affines.apply_affine(affine, ijk).astype(np.float32)


def _load_tms_keys():
    """Read the tms_risk session→condition map. Returns dict of
    ``{'01': {'2': 'vertex', '3': 'ips'}, ...}``."""
    resource = files('tms_risk').joinpath('data/tms_keys.yml')
    with open(resource) as f:
        return yaml.safe_load(f)


class Adapter(DatasetAdapter):
    name = 'tms_risk'

    DEFAULT_BIDS = '/data/ds-tmsrisk'

    def __init__(self, bids_folder: str = DEFAULT_BIDS,
                 denoise: bool = True):
        super().__init__(bids_folder)
        self.denoise = denoise

    def get_subjects(self) -> Sequence[str]:
        # All subjects with at least one TMS session listed.
        return sorted(_load_tms_keys().keys())

    def get_sessions(self, subject: str) -> Sequence[Optional[str]]:
        subj_key = f'{int(subject):02d}'
        keys = _load_tms_keys()
        if subj_key not in keys:
            return []
        return [str(s) for s in keys[subj_key].keys()]

    def default_session(self, subject: str) -> Optional[str]:
        """Return the subject's vertex (control) session number, as a
        string (e.g., ``'2'``). Falls back to the lowest-numbered
        session if no vertex condition is recorded.
        """
        subj_key = f'{int(subject):02d}'
        keys = _load_tms_keys()
        if subj_key not in keys:
            return None
        for sess_num, cond in keys[subj_key].items():
            if cond == 'vertex':
                return str(sess_num)
        # Fallback: lowest-numbered session.
        return str(min(keys[subj_key].keys()))

    def load_data(self,
                  subject: str,
                  session: Optional[str] = None,
                  roi: str = 'NPCr',
                  smoothed: bool = False,
                  **kwargs) -> LoadedFmriData:
        """Load tms_risk data for one (subject, session).

        Mirrors the path-construction in
        ``tms_risk.modeling.fit_nprf`` exactly so we land on the
        same single-trial estimates the production pipeline uses.
        """
        from tms_risk.utils import Subject

        if session is None:
            session = self.default_session(subject)
            if session is None:
                raise ValueError(
                    f"sub-{subject}: no session info in tms_keys.yml; "
                    f"pass session= explicitly")

        sub = Subject(str(subject), bids_folder=self.bids_folder)

        runs = list(range(1, 7))
        if (str(subject) == '10') and (str(session) == '1'):
            runs = list(range(1, 6))

        # Per-run events.tsv → trial-level paradigm.
        paradigm_parts = []
        for run in runs:
            ev_fn = op.join(
                self.bids_folder, f'sub-{subject}', f'ses-{session}',
                'func',
                f'sub-{subject}_ses-{session}_task-task_'
                f'run-{run}_events.tsv')
            ev = pd.read_csv(ev_fn, sep='\t')
            ev = ev[ev.trial_type == 'stimulus 1'].set_index('trial_nr')
            paradigm_parts.append(ev)
        paradigm = pd.concat(paradigm_parts, keys=runs, names=['run'])
        paradigm = paradigm[['n1']].astype(np.float32)
        paradigm.columns.name = None

        # Single-trial PE NIfTI; key composition mirrors fit_nprf.py.
        key = 'glm_stim1'
        if self.denoise:
            key += '.denoise'
        if smoothed:
            key += '.smoothed'
        pe_fn = op.join(
            self.bids_folder, 'derivatives', key, f'sub-{subject}',
            f'ses-{session}', 'func',
            f'sub-{subject}_ses-{session}_task-task_'
            f'space-T1w_desc-stims1_pe.nii.gz')

        masker = sub.get_volume_mask(roi=roi, session=session,
                                      epi_space=True, return_masker=True)
        data_2d = masker.fit_transform(pe_fn).astype(np.float32)
        data = pd.DataFrame(data_2d, index=paradigm.index)
        data.columns.name = 'voxel'

        xyz = _voxel_centroids_mm(masker)
        return LoadedFmriData(paradigm=paradigm, data=data,
                                masker=masker, xyz=xyz, sub=sub)

    def cv_folds(self, paradigm: pd.DataFrame) -> list:
        # leave-one-run-out within one session
        return list(paradigm.index.get_level_values('run').unique())
