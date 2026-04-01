#!/usr/bin/env python3
"""Stage pre-defined dataset groups used throughout LDAL experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.stage_datasets import default_data_home, stage_dataset_specs

DEFAULTS_FILE = REPO_ROOT / "config/datasets/defaults.json"


def _load_defaults(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Defaults file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _entry_to_spec(entry: dict) -> str:
    if "spec" in entry:
        return entry["spec"]
    dataset = entry["dataset"]
    spec = str(dataset)
    version = entry.get("version")
    if version is not None:
        spec = f"{spec}:{version}"
    prefer = entry.get("prefer")
    if prefer:
        spec = spec + "@" + "+".join(prefer)
    return spec


def _collect_specs(defaults: dict, groups: Iterable[str]) -> List[str]:
    specs: List[str] = []
    for group in groups:
        if group not in defaults:
            raise KeyError(f"Unknown dataset group '{group}'. Available: {', '.join(sorted(defaults))}")
        entries = defaults[group]
        for entry in entries:
            specs.append(_entry_to_spec(entry))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        "-g",
        action="append",
        help="Dataset group to stage (repeatable). Defaults to all groups defined in defaults file.",
    )
    parser.add_argument(
        "--data-home",
        default=default_data_home(),
        help="Dataset root (passed to LDAL_DATA_HOME). Default: %(default)s",
    )
    parser.add_argument(
        "--defaults-file",
        default=str(DEFAULTS_FILE),
        help="Path to defaults JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check that datasets already exist (delegates to stage_datasets --verify-only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if their sentinel exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset groups and exit",
    )
    args = parser.parse_args()

    defaults_path = Path(args.defaults_file).expanduser().resolve()
    defaults = _load_defaults(defaults_path)

    if args.list:
        print("Available dataset groups:")
        for name, entries in defaults.items():
            labels = [entry.get("name") or str(entry.get("dataset")) for entry in entries]
            print(f"  - {name}: {', '.join(labels)}")
        return

    groups = args.group or list(defaults.keys())
    specs = _collect_specs(defaults, groups)
    if not specs:
        print("No datasets declared for the selected groups; nothing to do.")
        return

    stage_dataset_specs(
        specs,
        data_home=args.data_home,
        default_prefer=None,
        verify_only=args.verify_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()
