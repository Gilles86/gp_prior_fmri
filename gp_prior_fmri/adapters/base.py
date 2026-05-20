"""Dataset adapter abstract base class.

For the cross-dataset GP-prior pipeline, a "Subject class" (the
single-source-of-truth pattern from the cogneuro-project skill) is
*per-project*, but we need to fit one pipeline across many projects.
The adapter pattern below wraps each project's ``Subject`` class
behind a stable interface so ``gp_prior_fmri.modeling.fit`` stays
dataset-agnostic.

The intent is that this interface eventually moves upstream into
``braincoder.utils.data`` so braincoder users get a standard way to
plug their dataset into braincoder's encoding-model pipeline. Until
then it lives here.

Conventions
-----------
* Subject IDs are strings (zero-padded where the upstream project
  does that). Adapters may accept either ``int`` or ``str`` and
  normalize.
* Session is ``None`` for datasets without sessions. For datasets
  with sessions, it is the BIDS session label (str or int).
* All paths returned must already be valid; if a file is missing the
  adapter should raise rather than return a non-existent path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Sequence, Any
import pandas as pd
import numpy as np


# ---------------------------------------------------------------- return type
class LoadedFmriData(NamedTuple):
    """Bundle of everything ``DatasetAdapter.load_data`` returns.

    NamedTuple so callers can use either positional unpacking or
    attribute access (``bundle.paradigm`` reads cleaner in the fit
    pipeline; ``paradigm, data, masker, xyz, sub = ...`` still works
    for back-compat with the original fit_gp_prior.py code).

    Attributes
    ----------
    paradigm : pd.DataFrame
        Trial-level stimulus DataFrame indexed by (run, trial_nr) or
        whatever multi-index the dataset uses. Columns are the stim
        features the model consumes — for a 1-D Gaussian PRF this is
        a single column (e.g., ``'x'`` or ``'n1'``); for multi-range
        models it is two columns (e.g., ``['x', 'range']``).
    data : pd.DataFrame
        Per-trial neural response, same index as ``paradigm``, columns
        are voxels (column index name ``'voxel'``).
    masker : nilearn.maskers.NiftiMasker
        Already ``fit``-ted to the chosen ROI mask. Used downstream to
        write parameter maps back out as NIfTI and to recover voxel
        coordinates.
    xyz : np.ndarray
        ``(n_voxels, 3)`` scanner-space coordinates of the in-mask
        voxel centroids. Used to project voxels to the cortical
        surface for the geodesic distance matrix.
    sub : Any
        The upstream project's ``Subject`` object. Held so the fit
        script can ask for surface info, etc. Type is intentionally
        loose (``Any``) because each project's ``Subject`` is its own
        class.
    """
    paradigm: pd.DataFrame
    data: pd.DataFrame
    masker: Any
    xyz: np.ndarray
    sub: Any


# ---------------------------------------------------------------- ABC
class DatasetAdapter(ABC):
    """Wrap a neuroimaging dataset for use by the GP-prior pipeline.

    Implementations live in sibling modules (``neural_priors.py``,
    ``tms_risk.py``, …). One adapter = one BIDS dataset; ``bids_folder``
    is captured at construction so call sites stay clean.

    Subclasses must override every ``@abstractmethod``; everything
    else has a sensible default.
    """

    name: str = 'unnamed'   # subclasses set this; used in output paths

    def __init__(self, bids_folder: str):
        self.bids_folder = bids_folder

    # ------------------------------------------------------- subject listing
    @abstractmethod
    def get_subjects(self) -> Sequence[str]:
        """Return the list of subject IDs (zero-padded strings)."""
        raise NotImplementedError

    def get_sessions(self, subject: str) -> Sequence[Optional[str]]:
        """Return the sessions available for a subject. Datasets
        without sessions return ``[None]``.
        """
        return [None]

    def default_session(self, subject: str) -> Optional[str]:
        """Pick a "canonical" session for this subject — used when the
        caller doesn't specify ``--session``. For control-vs-condition
        datasets (e.g., tms_risk), return the control session. For
        session-less datasets, return ``None``.
        """
        sessions = list(self.get_sessions(subject))
        return sessions[0] if sessions else None

    # ------------------------------------------------------- data loading
    @abstractmethod
    def load_data(self,
                  subject: str,
                  session: Optional[str] = None,
                  roi: str = 'NPCr',
                  smoothed: bool = False,
                  **kwargs) -> LoadedFmriData:
        """Load (paradigm, data, masker, xyz, sub) for one subject.

        See :class:`LoadedFmriData` for the return-bundle schema.
        ``**kwargs`` lets each adapter expose dataset-specific flags
        (e.g., neural_priors' ``stim_range``, tms_risk's
        ``denoise``).
        """
        raise NotImplementedError

    # ------------------------------------------------------- surface lookup
    @abstractmethod
    def get_white_surface_path(self, subject: str, hemi: str) -> str:
        """Path to the white-matter GIfTI surface (``.surf.gii``) for
        one hemisphere. ``hemi`` is ``'L'`` or ``'R'``. The file must
        exist; the geodesic-distance helper loads it directly.

        We expose the path (not a path-set or info dict) because the
        wrapped projects' ``get_surf_info()`` methods strictly assert
        that every related file (inflated, pial, curv...) exists,
        which fails on minimally-preprocessed subjects. The white
        surface is the only one this pipeline actually needs.
        """
        raise NotImplementedError

    # ------------------------------------------------------- CV folds
    def cv_folds(self, paradigm: pd.DataFrame) -> list:
        """Return the list of fold keys for leave-one-X-out CV.

        Default = leave-one-run-out keyed by ``'run'`` in the
        paradigm's index. Override when the dataset's CV unit is
        something else (e.g., neural_priors' ``(session, run2)``).
        """
        if 'run' in (paradigm.index.names or ()):
            return list(paradigm.index.get_level_values('run').unique())
        raise NotImplementedError(
            f"{self.name}: paradigm has no 'run' level — override "
            f"cv_folds() to define how this dataset splits.")

    # ------------------------------------------------------- output keying
    def output_dir(self, bids_folder: str, roi: str, tag: str,
                   smoothed: bool, subject: str,
                   session: Optional[str]) -> str:
        """Where outputs for one (subject, session) land. Override if
        you want a different layout; default mirrors neural_priors'
        ``gp_prior_roi-{ROI}[.smoothed]/exp-{tag}/sub-{NN}[/ses-{Y}]/func/``.
        """
        import os.path as op
        key = f'gp_prior_roi-{roi}'
        if smoothed:
            key += '.smoothed'
        parts = [bids_folder, 'derivatives', key,
                 f'exp-{tag}', f'sub-{subject}']
        if session is not None:
            parts.append(f'ses-{session}')
        parts.append('func')
        return op.join(*parts)
