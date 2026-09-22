#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Extract pure-Python TensileLite wheels into an installed test artifact."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import zipfile


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
