"""Adapter smoke tests.

The data-loading paths require the source-dataset projects to be
importable AND the local BIDS folder to be reachable. When that
isn't the case (e.g., CI), those tests skip cleanly. The pure-code
tests (ABC + subject lists) always run.
"""
from __future__ import annotations

import os.path as op
import pytest

from gp_prior_fmri.adapters import DatasetAdapter, LoadedFmriData, get_adapter


# ---------------------------------------------------------------- ABC
def test_loaded_fmri_data_is_namedtuple():
    """LoadedFmriData should support both positional and attribute access."""
    import numpy as np
    import pandas as pd
    bundle = LoadedFmriData(
        paradigm=pd.DataFrame({'x': [1, 2, 3]}),
        data=pd.DataFrame({0: [0.1, 0.2, 0.3]}),
        masker=None, xyz=np.zeros((1, 3), dtype=np.float32), sub=None)
    assert bundle.paradigm.shape == (3, 1)
    # positional unpacking still works
    paradigm, data, masker, xyz, sub = bundle
    assert paradigm.shape == (3, 1)


def test_get_adapter_rejects_unknown():
    with pytest.raises(ValueError, match='Unknown dataset adapter'):
        get_adapter('not_a_real_dataset')


def test_get_adapter_returns_subclass():
    try:
        adapter = get_adapter('neural_priors')
    except ImportError:
        pytest.skip("neural_priors not installed")
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.name == 'neural_priors'


# ---------------------------------------------------------------- neural_priors
def test_neural_priors_subjects():
    try:
        adapter = get_adapter('neural_priors')
    except ImportError:
        pytest.skip("neural_priors not installed")
    subs = adapter.get_subjects()
    assert '01' in subs
    assert '11' not in subs   # known exclusion
    assert '23' not in subs   # known exclusion
    assert len(subs) == 39


# ---------------------------------------------------------------- tms_risk
def test_tms_risk_default_session_picks_vertex():
    try:
        adapter = get_adapter('tms_risk')
    except ImportError:
        pytest.skip("tms_risk not installed")
    # If tms_risk's data/tms_keys.yml has '01', the default session
    # should be one of {'2', '3'} and correspond to the vertex condition.
    if '01' not in adapter.get_subjects():
        pytest.skip("tms_risk has no sub-01")
    sess = adapter.default_session('01')
    assert sess in ('2', '3')
