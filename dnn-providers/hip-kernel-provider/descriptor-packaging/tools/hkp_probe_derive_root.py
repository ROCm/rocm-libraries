#!/usr/bin/env python3
"""Derive a one-kernel descriptor root from a production descriptor directory.

Copies every file under `--from` unchanged except the named KDP, whose
`kernelDescriptors` list is reduced to the single UKD whose `name` equals
`--instance-name`. The output directory is removed first so a stale derived root
cannot survive.

Exit code 0 on success. Exit code 2, with a one-line `hkp_probe_derive: ...`
message on stderr, when the KDP is missing or the instance name does not match
exactly one descriptor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PREFIX = "hkp_probe_derive:"


def derive_root(src: Path, kdp_name: str, instance_name: str, out: Path) -> str | None:
    """Write the derived root to `out`; return an error message, or None on success."""
    kdp_path = src / kdp_name
    if not kdp_path.is_file():
        return f"{_PREFIX} {kdp_name} not found under {src}"

    kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
    descriptors = kdp["kernelDescriptors"]
    matches = [d for d in descriptors if d.get("name") == instance_name]
    if not matches:
        return (
            f"{_PREFIX} instance '{instance_name}' not found in {kdp_name} "
            f"({len(descriptors)} descriptors)"
        )
    if len(matches) > 1:
        return (
            f"{_PREFIX} instance '{instance_name}' matched {len(matches)} times "
            f"in {kdp_name} ({len(descriptors)} descriptors)"
        )

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    kdp["kernelDescriptors"] = matches
    (out / kdp_name).write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="src", type=Path, required=True)
    parser.add_argument("--kdp", required=True, help="KDP file name under --from")
    parser.add_argument("--instance-name", required=True, help="UKD name to keep")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    error = derive_root(args.src, args.kdp, args.instance_name, args.out)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
