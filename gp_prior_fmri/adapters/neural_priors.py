"""neural_priors adapter.

Wraps ``neural_priors.utils.data.Subject`` and
``neural_priors.encoding_model.fit_model.get_paradigm`` to expose the
``DatasetAdapter`` interface. Paradigm columns are ``['x', 'range']``;
the GP-prior pipeline can request a single stim range, both, or use
model 15's joint narrow+wide fit.

This file replaces the data-loading section of the legacy
``neural_priors/neural_priors/encoding_model/fit_gp_prior.py`` —
business logic (priors, CV, decoding) now lives in
``gp_prior_fmri.modeling.fit``.
"""
from __future__ import annotations

import os.path as op
from typing import Optional, Sequence

import numpy as np
import nibabel as nib
import pandas as pd

from .base import DatasetAdapter, LoadedFmriData


def _voxel_centroids_mm(masker):
    mask_img = masker.mask_img_
    affine = mask_img.affine
    mask_data = mask_img.get_fdata().astype(bool)
    ijk = np.argwhere(mask_data)
    return nib.affines.apply_affine(affine, ijk).astype(np.float32)


class Adapter(DatasetAdapter):
    name = 'neural_priors'

    DEFAULT_BIDS = '/data/ds-neuralpriors'

    # Valid subject IDs (zero-padded). Mirrors the inclusion list
    # used in the legacy SLURM submissions (excludes sub-11 + sub-23,
    # which don't exist on disk).
    _VALID_SUBJECTS = (
        [f'{i:02d}' for i in range(1, 11)]      # 01–10
        + [f'{i:02d}' for i in range(12, 23)]   # 12–22
        + [f'{i:02d}' for i in range(24, 42)]   # 24–41
    )

    def __init__(self, bids_folder: str = DEFAULT_BIDS):
        super().__init__(bids_folder)

    def get_subjects(self) -> Sequence[str]:
        return list(self._VALID_SUBJECTS)

    def get_sessions(self, subject: str) -> Sequence[Optional[str]]:
        # neural_priors is session-less from this pipeline's perspective
        # (data is pooled across sessions internally by the Subject class).
        return [None]

    def default_session(self, subject: str) -> Optional[str]:
        return None

    def load_data(self,
                  subject: str,
                  session: Optional[str] = None,
                  roi: str = 'NPCr',
                  smoothed: bool = False,
                  stim_range: str = 'both',
                  **kwargs) -> LoadedFmriData:
        """Load neural_priors data for one subject.

        Parameters
        ----------
        stim_range : {'narrow', 'wide', 'both'}
            Filter trials by stim range. ``'both'`` is the default and
            is required for fitting model 15 (LinearScalingModel).
        """
        # Heavy import inside the call so the package imports cleanly
        # without neural_priors installed (e.g., on a machine that
        # only has tms_risk).
        from neural_priors.utils.data import Subject
        from neural_priors.encoding_model.fit_model import get_paradigm

        sub = Subject(str(subject), bids_folder=self.bids_folder)

        paradigm_full = get_paradigm(sub, fit_responses=False)
        if stim_range == 'wide':
            keep = np.asarray(paradigm_full['range'] == 1.0)
        elif stim_range == 'narrow':
            keep = np.asarray(paradigm_full['range'] == 0.0)
        elif stim_range == 'both':
            keep = np.ones(len(paradigm_full), dtype=bool)
        else:
            raise ValueError(
                f"stim_range must be wide/narrow/both, got {stim_range!r}")
        paradigm_full = paradigm_full.loc[keep]

        # CV index level: (run-1) % 4 + 1, so runs across sessions
        # get re-grouped into 4 mod-classes for leave-one-out CV.
        runs = paradigm_full.index.get_level_values('run')
        paradigm_full = paradigm_full.set_index(
            pd.Index((runs - 1) % 4 + 1, name='run2'), append=True)
        paradigm_full.index = paradigm_full.index.swaplevel('run', 'run2')
        paradigm_full = paradigm_full.droplevel(['run', 'trial_nr',
                                                  'subject'])

        # For models that consume a single stim column (LogGaussianPRF),
        # caller pulls just paradigm['x']. For multi-range models
        # (LinearScalingModel) the full DataFrame is used.
        paradigm = paradigm_full[['x', 'range']].astype(np.float32)

        # Build the NiftiMasker here rather than via
        # ``sub.get_volume_mask(return_masker=True)`` — the neural_priors
        # version calls ``NiftiMasker(resampling_target='data')`` which
        # has been removed in modern nilearn. Construct it ourselves
        # from the mask Img so we work across nilearn versions.
        from nilearn.maskers import NiftiMasker
        mask_img = sub.get_volume_mask(roi=roi, epi_space=True,
                                        return_masker=False)
        masker = NiftiMasker(mask_img=mask_img)
        masker.fit()
        data_img = sub.get_single_trial_estimates(session=None,
                                                    smoothed=smoothed)
        data_2d = masker.fit_transform(data_img).astype(np.float32)
        # Reindex through the original (un-filtered) paradigm index, then
        # apply the same keep mask so data and paradigm align.
        data_full = pd.DataFrame(data_2d, index=get_paradigm(sub).index)
        data = data_full.iloc[keep].copy()
        data.index = paradigm.index
        data.columns.name = 'voxel'

        xyz = _voxel_centroids_mm(masker)
        return LoadedFmriData(paradigm=paradigm, data=data,
                                masker=masker, xyz=xyz, sub=sub)

    def get_white_surface_path(self, subject: str, hemi: str) -> str:
        # Mirrors neural_priors' get_surf_info logic for the 'inner'
        # (white) entry, without the all-files-must-exist assertion.
        path = op.join(
            self.bids_folder, 'derivatives', 'fmriprep',
            f'sub-{subject}', 'anat',
            f'sub-{subject}_hemi-{hemi}_white.surf.gii')
        if not op.exists(path):
            raise FileNotFoundError(
                f'neural_priors white surface not found: {path}')
        return path

    def cv_folds(self, paradigm: pd.DataFrame) -> list:
        # leave-one-(session, run2)-out
        return list(paradigm.groupby(level=['session', 'run2'])
                                .groups.keys())
