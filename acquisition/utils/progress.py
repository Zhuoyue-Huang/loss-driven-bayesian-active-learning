"""Progress helpers that degrade gracefully when tqdm is unavailable."""

from __future__ import annotations

try:
    from tqdm import trange as _tqdm_trange
except ModuleNotFoundError:
    _tqdm_trange = None


def trange(*args, **kwargs):
    """Return a tqdm-backed range when available, otherwise plain range."""

    if _tqdm_trange is not None:
        return _tqdm_trange(*args, **kwargs)
    return range(*args)
