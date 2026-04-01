#!/usr/bin/env python3
"""Stage or verify UCI/OpenML datasets for LDAL experiments.

Each dataset specification looks like:
  <name_or_id>[:<version>][@<source1+source2>]

Examples:
  yacht_hydrodynamics
  slump:2@openml
  477@ucimlrepo+pmlb

Run from the repository root, e.g.:
  python scripts/data/stage_datasets.py --dataset yacht_hydrodynamics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.generators import fetch_any_uci


def default_data_home() -> str:
    env_home = os.environ.get("LDAL_DATA_HOME")
    if env_home:
        return os.path.expanduser(env_home)
    return str((Path.home() / ".cache" / "ldal" / "datasets").expanduser())


def _slugify(token: str) -> str:
    cleaned = token.lower().strip()
    out = []
    for char in cleaned:
        if char.isalnum() or char in ("-", "_"):
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "dataset"


def _parse_prefer(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = value.replace("+", ",").split(",")
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned or None


def _parse_spec(spec: str) -> dict:
    raw = spec.strip()
    if not raw or raw.startswith("#"):
        raise ValueError("empty dataset spec")
    prefer = None
    version = None
    core = raw
    if "@" in core:
        core, prefer_part = core.split("@", 1)
        prefer = _parse_prefer(prefer_part)
    if ":" in core:
        core, version_part = core.split(":", 1)
        if version_part:
            version = int(version_part)
    try:
        dataset = int(core)
    except ValueError:
        dataset = core
    return {
        "dataset": dataset,
        "version": version,
        "prefer": prefer,
        "slug": _slugify(raw),
        "raw": raw,
    }


def _load_specs(args: argparse.Namespace) -> List[str]:
    specs: List[str] = []
    specs.extend(args.dataset or [])
    if args.spec_file:
        with open(args.spec_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                specs.append(line)
    if not specs:
        raise SystemExit("Provide at least one --dataset or --spec-file entry")
    return specs


def _load_known_defaults() -> Dict[str, List[str]]:
    """
    Load default source preferences from config/datasets/defaults.json if present.
    Returns a mapping from dataset identifier (string) to prefer list.
    """
    defaults_file = REPO_ROOT / "config/datasets/defaults.json"
    if not defaults_file.exists():
        return {}
    try:
        with defaults_file.open("r", encoding="utf-8") as handle:
            data: Dict[str, List[Dict[str, Any]]] = json.load(handle)  # type: ignore
    except Exception:
        return {}
    mapping: Dict[str, List[str]] = {}
    for group_entries in data.values():
        for entry in group_entries:
            prefer = entry.get("prefer")
            dataset = entry.get("dataset")
            if prefer and dataset is not None:
                mapping[str(dataset)] = prefer
    return mapping


def stage_dataset_specs(
    specs: List[str],
    *,
    data_home: Path | str,
    default_prefer: Optional[List[str]] = None,
    verify_only: bool = False,
    force: bool = False,
) -> None:
    data_home = Path(data_home).expanduser().resolve()
    sentinel_dir = data_home / ".sentinels"
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    known_defaults = _load_known_defaults()

    parsed_specs = [_parse_spec(spec) for spec in specs]

    for spec in parsed_specs:
        slug = spec["slug"]
        sentinel = sentinel_dir / f"{slug}.complete"
        prefer = spec["prefer"]
        if prefer is None:
            prefer = known_defaults.get(str(spec["dataset"])) or known_defaults.get(spec["raw"]) or default_prefer
        label = spec["raw"]

        if verify_only:
            if sentinel.exists():
                print(f"✔ {label} present ({sentinel})")
            else:
                raise SystemExit(f"✖ Missing cache for {label}; expected {sentinel}")
            continue

        if sentinel.exists() and not force:
            print(f"↺ Skipping {label}; cache already prepared ({sentinel})")
            continue

        kwargs = {"data_home": str(data_home)}
        if prefer:
            kwargs["prefer"] = tuple(prefer)
        if spec["version"] is not None:
            kwargs["version"] = spec["version"]

        print(f"⬇ Fetching {label} with prefer={kwargs.get('prefer')} version={kwargs.get('version')}")
        fetch_any_uci(spec["dataset"], **kwargs)
        sentinel.write_text(
            json.dumps(
                {
                    "dataset": spec["dataset"],
                    "version": spec["version"],
                    "prefer": prefer,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                indent=2,
            )
        )
        print(f"✔ Cached {label} → {sentinel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        "-d",
        action="append",
        help="Dataset spec (repeatable). Format: name[:version][@source+...]",
    )
    parser.add_argument(
        "--spec-file",
        help="Path to a text file containing one dataset spec per line",
    )
    parser.add_argument(
        "--data-home",
        default=default_data_home(),
        help="Directory to use as LDAL_DATA_HOME (default: %(default)s)",
    )
    parser.add_argument(
        "--default-prefer",
        help="Comma-separated fallback source preference (e.g. openml,pmlb)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that sentinel files exist; do not download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a sentinel already exists",
    )
    args = parser.parse_args()

    data_home = args.data_home
    default_prefer = _parse_prefer(args.default_prefer)
    specs = _load_specs(args)
    stage_dataset_specs(
        specs,
        data_home=data_home,
        default_prefer=default_prefer,
        verify_only=args.verify_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()
