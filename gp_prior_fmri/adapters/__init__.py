"""Dataset adapters.

Each module here implements ``DatasetAdapter`` for one neuroimaging
dataset. ``get_adapter(name)`` returns the right instance.

Adapters are intentionally lazy-imported so installing the package
doesn't require every wrapped repo to be present.
"""
from .base import DatasetAdapter, LoadedFmriData

_ADAPTERS = {
    'neural_priors': '.neural_priors',
    'tms_risk':      '.tms_risk',
}


def get_adapter(name: str, **kwargs) -> DatasetAdapter:
    """Instantiate the named dataset adapter.

    Args:
        name: One of the keys in ``_ADAPTERS`` (e.g., 'neural_priors',
            'tms_risk').
        **kwargs: Passed to the adapter's constructor (most importantly
            ``bids_folder``).
    """
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset adapter {name!r}. "
            f"Available: {sorted(_ADAPTERS)}")
    from importlib import import_module
    mod = import_module(_ADAPTERS[name], package=__name__)
    cls = mod.Adapter
    return cls(**kwargs)


__all__ = ['DatasetAdapter', 'LoadedFmriData', 'get_adapter']
