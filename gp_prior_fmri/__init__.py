"""Cross-dataset GP-prior pRF fitting and decoding.

The fitting pipeline (``gp_prior_fmri.fit``) is dataset-agnostic. Each
neuroimaging dataset (e.g., ``neural_priors``, ``tms_risk``) is wrapped
in a small ``DatasetAdapter`` subclass under
``gp_prior_fmri.adapters``. The adapter knows how to enumerate
subjects/sessions, load paradigm + single-trial estimates + cortical
surface for one subject — and nothing else.

The current adapter ABC mirrors what could eventually live in
``braincoder.utils.data`` so the abstraction can move upstream once
its API stabilizes.
"""
__version__ = "0.1.0"
