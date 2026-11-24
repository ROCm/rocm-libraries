#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate a selected generator against its native producer, then unpack it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_release_wheel_contents import validate_identity
from install_release_wheels import install_wheels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--wheel", type=Path)
    selection.add_argument("--manifest", type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and args.destination is None:
        parser.error("--destination is required when preparing a generator")
    wheel = args.wheel
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        wheel = args.manifest.parent / manifest["wheel"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    # Source archives without a revision retain the established pairing policy:
    # validate the package/version, but do not invent a Git identity requirement.
    revision = metadata["source_revision"]
    if revision == "none":
        revision = None
    problems = validate_identity(wheel, metadata["distribution_version"], revision)
    if problems:
        parser.error("Invalid TensileLite generator wheel:\n" + "\n".join(problems))
    if not args.check_only:
        install_wheels([wheel], args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
