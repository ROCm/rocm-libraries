# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run one disjoint cohort of the actual installed platform pytest payload."""

import argparse
from pathlib import Path

import pytest

LAYERS = ("core", "helpers", "instances")
COHORTS = ("other", *LAYERS)


def cohort_paths(root: Path, cohort: str) -> list[Path]:
    if cohort not in COHORTS:
        raise ValueError(f"unknown pytest cohort: {cohort}")
    if cohort != "other":
        path = root / cohort
        if not path.is_dir():
            raise ValueError(f"missing pytest layer: {cohort}")
        return [path]
    paths = [
        child
        for child in sorted(root.iterdir())
        if child.name not in {*LAYERS, "library", "__pycache__"}
        and not child.name.startswith(".")
        and (
            child.is_dir() or (child.name.startswith("test_") and child.suffix == ".py")
        )
    ]
    if not paths:
        raise ValueError("remaining-tree pytest cohort is empty")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=COHORTS)
    args, pytest_args = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parent
    paths = cohort_paths(root, args.cohort)
    # Pin the same root config/markers for every cohort; never inherit a parent
    # workspace's pytest configuration. pytest stays in this interpreter and
    # sees the CTest-supplied native/compiler environment unchanged.
    return int(
        pytest.main(
            [
                "-c",
                str(root / "pytest.ini"),
                *map(str, paths),
                "--ignore",
                str(root / "library"),
                "-m",
                "not integration",
                *pytest_args,
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
