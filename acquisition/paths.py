from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _resolve_root(env_var: str, default: Path) -> Path:
    """Resolve a root directory from an environment variable or fallback default."""
    override = os.environ.get(env_var)
    base = Path(override).expanduser() if override else default
    return base.resolve()


DEFAULT_CODE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = _resolve_root("LDAL_CODE_ROOT", DEFAULT_CODE_ROOT)
RESULTS_ROOT = _resolve_root("LDAL_RESULTS_ROOT", CODE_ROOT / "results")
CHECKPOINT_ROOT = _resolve_root("LDAL_CHECKPOINT_ROOT", CODE_ROOT / "checkpoint")


def _ensure_dir(root: Path, parts: Iterable[str | os.PathLike[str]]) -> Path:
    path = root
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_results_dir(*parts: str | os.PathLike[str]) -> Path:
    """Create (if needed) and return a subdirectory rooted at RESULTS_ROOT."""
    return _ensure_dir(RESULTS_ROOT, parts)


def ensure_checkpoint_dir(*parts: str | os.PathLike[str]) -> Path:
    """Create (if needed) and return a subdirectory rooted at CHECKPOINT_ROOT."""
    return _ensure_dir(CHECKPOINT_ROOT, parts)


def results_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a Path rooted at RESULTS_ROOT (without creating parents)."""
    path = RESULTS_ROOT
    for part in parts:
        path = path / part
    return path


def checkpoint_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a Path rooted at CHECKPOINT_ROOT (without creating parents)."""
    path = CHECKPOINT_ROOT
    for part in parts:
        path = path / part
    return path
