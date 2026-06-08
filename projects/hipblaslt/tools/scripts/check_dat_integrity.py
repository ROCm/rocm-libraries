#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Post-build integrity checks for installed hipBLASLt Tensile .dat files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

try:
    import msgpack
except ImportError:
    msgpack = None

_MASTER_RE = re.compile(r"^TensileLibrary_lazy_(?P<arch>[A-Za-z0-9]+)\.dat$")
_MAPPING_RE = re.compile(
    r"^TensileLiteLibrary_lazy_(?P<arch>[A-Za-z0-9]+)_Mapping\.dat$"
)


def _archDir(libDir: Path, arch: str) -> Path:
    return libDir / arch.split(":")[0]


def _scanArchs(libDir: Path):
    masters, mappings = set(), set()
    for sub in libDir.iterdir():
        if not sub.is_dir() or not sub.name.startswith("gfx"):
            continue
        for f in sub.iterdir():
            if m := _MASTER_RE.match(f.name):
                masters.add(m.group("arch"))
            elif m := _MAPPING_RE.match(f.name):
                mappings.add(m.group("arch"))
    return masters, mappings


def _loadMapping(libDir: Path, arch: str):
    path = _archDir(libDir, arch) / f"TensileLiteLibrary_lazy_{arch}_Mapping.dat"
    with open(path, "rb") as f:
        return msgpack.unpack(f, raw=False, strict_map_key=False)


def validate(libDir: Path) -> List[str]:
    libDir = Path(libDir).resolve()
    if not libDir.is_dir():
        return [f"library dir does not exist or is not a directory: {libDir}"]
    if msgpack is None:
        return ["msgpack is required to read Tensile .dat files but is not installed"]

    violations: List[str] = []
    masters, mappings = _scanArchs(libDir)
    archs = masters & mappings

    if not archs:
        return [
            f"{libDir} contains no matched (master, Mapping) pair; runtime "
            "cannot resolve lazy lookups"
        ]
    for a in sorted(masters - archs):
        violations.append(f"master without a per-arch Mapping: {a}")
    for a in sorted(mappings - archs):
        violations.append(f"per-arch Mapping without a master: {a}")

    fallback_dat_files = list(libDir.glob("*/*_fallback_*.dat"))
    for arch in sorted(archs):
        mapping = _loadMapping(libDir, arch)
        archDir = _archDir(libDir, arch)

        for idx, kernelName in mapping.items():
            if not (archDir / f"{kernelName}.dat").is_file():
                violations.append(
                    f"arch={arch}: Mapping[{idx}] -> {kernelName}.dat is not on disk"
                )
            if not kernelName.endswith("_" + arch):
                violations.append(
                    f"arch={arch}: Mapping[{idx}] -> {kernelName} is not arch-scoped "
                    "(kpack overlay collision risk)"
                )

        arch_has_fallback_files = any(
            f.name.endswith(f"_fallback_{arch}.dat") for f in fallback_dat_files
        )
        mapping_has_fallback = any(
            "_fallback_" in n and n.endswith("_" + arch) for n in mapping.values()
        )
        if arch_has_fallback_files and not mapping_has_fallback:
            violations.append(
                f"arch={arch}: per-arch Mapping is missing fallback entries "
                "(runtime may report 'NO solution found!' for fallback-only dtypes)"
            )

    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "library_dir",
        type=Path,
        help="Installed library dir containing <base-arch>/ subdirs of .dat files",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args(argv)

    violations = validate(args.library_dir)
    if violations:
        print(
            f"[check_dat_integrity] {len(violations)} violation(s) in {args.library_dir}:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"[check_dat_integrity] OK: {args.library_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
