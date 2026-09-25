#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extract pure-Python TensileLite wheels into an installed test artifact."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile


_CANONICAL_PATHS = (
    Path("tensilelite"),
    Path("_tensilelite_client_binding.py"),
    Path("tensilelite_configure_client.py"),
)
_COMPATIBILITY_PATHS = (Path("tensilelite_tensile_compat"),)


def _remove_owned_paths(destination: Path, members: list[zipfile.ZipInfo]) -> None:
    top_levels = {PurePosixPath(member.filename).parts[0] for member in members}
    if "tensilelite" in top_levels:
        owned_paths = _CANONICAL_PATHS
        dist_info_patterns = ("tensilelite-*.dist-info",)
    elif "tensilelite_tensile_compat" in top_levels:
        owned_paths = _COMPATIBILITY_PATHS
        dist_info_patterns = ("tensilelite_tensile_compat-*.dist-info",)
    else:
        raise ValueError("wheel contains no recognized TensileLite package root")

    for relative_path in owned_paths:
        path = destination / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for pattern in dist_info_patterns:
        for path in destination.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def install_wheels(wheels: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.infolist()
            invalid = [
                member.filename
                for member in members
                if PurePosixPath(member.filename).is_absolute()
                or ".." in PurePosixPath(member.filename).parts
            ]
            if invalid:
                raise ValueError(f"wheel contains unsafe paths: {invalid}")
            _remove_owned_paths(destination, members)
            archive.extractall(destination, members)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args(argv)
    install_wheels(args.wheels, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
