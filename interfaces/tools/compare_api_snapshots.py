#!/usr/bin/env python3
"""Fail when regenerated API snapshots differ from checked-in metadata."""

from __future__ import annotations

import argparse
from pathlib import Path


def snapshots(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {directory}")
    result = {path.name: path.read_bytes() for path in sorted(directory.glob("*.json"))}
    if not result:
        raise ValueError(f"snapshot directory is empty: {directory}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    expected = snapshots(args.expected)
    actual = snapshots(args.actual)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    if missing or extra or changed:
        details = [
            f"missing regenerated snapshots: {', '.join(missing) or '-'}",
            f"unexpected regenerated snapshots: {', '.join(extra) or '-'}",
            f"changed snapshots: {', '.join(changed) or '-'}",
        ]
        raise ValueError(
            "API snapshots are stale; regenerate them:\n" + "\n".join(details)
        )


if __name__ == "__main__":
    main()
