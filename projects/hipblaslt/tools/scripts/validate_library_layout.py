#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate the installed hipBLASLt library tree against the per-base layout."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Set

REQUIRED_PER_BASE_FILES = (
    "hipblasltTransform_{arch}.hsaco",
    "hipblasltExtOpLibrary_{arch}.dat",
    "extop_{arch}.co",
)

TENSILE_MASTER_CANDIDATES = (
    "TensileLibrary_{arch}.dat",
    "TensileLibrary_{arch}.yaml",
)

TENSILE_LAZY_CANDIDATES = (
    "TensileLibrary_lazy_{arch}.dat",
    "TensileLibrary_lazy_{arch}.yaml",
)

PER_ARCH_REQUIRED = {
    "gfx950": ("rr_custom_kernels_gfx950.co",),
}

FORBIDDEN_FLAT_ROOT_BASENAMES = (
    "TensileLibrary.dat",
    "TensileLibrary.yaml",
    "TensileLibrary_lazy.dat",
    "TensileLibrary_lazy.yaml",
    "hipblasltTransform.hsaco",
    "hipblasltExtOpLibrary.dat",
    "extop.co",
)

_GFX_PREFIX_RE = re.compile(r"^gfx[a-z0-9]+(?:[\-:][\-+a-z0-9]+)*$")


def _arch_dir_name_is_base(name: str) -> bool:
    return bool(re.fullmatch(r"gfx[a-z0-9]+", name))


def _filename_arch_matches_dir(filename: str, base_arch: str) -> bool:
    pattern = re.compile(r"(?:^|[._-])(?P<arch>gfx[0-9a-z]+(?:[-+][0-9a-z]+)*)")
    for m in pattern.finditer(filename):
        found = m.group("arch")
        if (
            found == base_arch
            or found.startswith(base_arch + "-")
            or found.startswith(base_arch + "+")
        ):
            return True
    return False


def _library_root(install_root: Path) -> Optional[Path]:
    candidates = (
        install_root / "lib" / "hipblaslt" / "library",
        install_root / "library",
    )
    for c in candidates:
        if c.is_dir():
            return c
    return None


def validate(install_root: Path) -> List[str]:
    install_root = Path(install_root).resolve()
    violations: List[str] = []

    if not install_root.is_dir():
        return [f"install_root does not exist or is not a directory: {install_root}"]

    library_dir = _library_root(install_root)
    if library_dir is None:
        return [
            f"library dir not found under {install_root}; expected "
            f"<root>/lib/hipblaslt/library/ or <root>/library/"
        ]

    for basename in FORBIDDEN_FLAT_ROOT_BASENAMES:
        offender = library_dir / basename
        if offender.is_file():
            violations.append(
                f"flat-root file found (per-base layout violation): {offender}"
            )

    for entry in library_dir.iterdir():
        if entry.is_file() and entry.suffix in (".dat", ".co", ".hsaco", ".yaml"):
            violations.append(
                f"unexpected payload file at library root: {entry} "
                f"(per-base layout requires files in library/<base>/)"
            )

    base_arch_dirs = sorted(
        p for p in library_dir.iterdir() if p.is_dir() and p.name.startswith("gfx")
    )
    if not base_arch_dirs:
        violations.append(f"no per-base gfx* subdirs found under {library_dir}")
        return violations

    for d in base_arch_dirs:
        if not _arch_dir_name_is_base(d.name):
            violations.append(
                f"library subdir name carries target features (must be bare base arch): {d}"
            )

    for arch_dir in base_arch_dirs:
        base = arch_dir.name
        if not _arch_dir_name_is_base(base):
            continue

        entries: Set[str] = {p.name for p in arch_dir.iterdir() if p.is_file()}

        for template in REQUIRED_PER_BASE_FILES:
            wanted = template.format(arch=base)
            if wanted not in entries:
                violations.append(f"missing required file in {arch_dir}: {wanted}")

        master_present = any(
            t.format(arch=base) in entries for t in TENSILE_MASTER_CANDIDATES
        )
        lazy_present = any(
            t.format(arch=base) in entries for t in TENSILE_LAZY_CANDIDATES
        )
        if not (master_present or lazy_present):
            violations.append(
                f"missing TensileLibrary master/lazy file for {base} in {arch_dir} "
                f"(expected one of: TensileLibrary_{base}.{{dat,yaml}} or "
                f"TensileLibrary_lazy_{base}.{{dat,yaml}})"
            )

        for extra in PER_ARCH_REQUIRED.get(base, ()):
            if extra not in entries:
                violations.append(
                    f"missing required {base}-only file in {arch_dir}: {extra}"
                )

        for fname in entries:
            if fname == "metadata.yaml":
                continue
            if not _filename_arch_matches_dir(fname, base):
                violations.append(
                    f"filename arch suffix does not match dir {base}: {arch_dir / fname}"
                )

    for arch, extras in PER_ARCH_REQUIRED.items():
        for fname in extras:
            for d in base_arch_dirs:
                if d.name == arch:
                    continue
                stray = d / fname
                if stray.is_file():
                    violations.append(
                        f"file required only for {arch} found under wrong arch dir: {stray}"
                    )

    return violations


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "install_root",
        type=Path,
        help="Install root containing lib/hipblaslt/library/ (or a build tree "
        "containing library/).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress success message; exit code still reflects result.",
    )
    args = parser.parse_args(argv)

    violations = validate(args.install_root)
    if violations:
        print(
            f"[validate_library_layout] {len(violations)} layout violation(s) "
            f"in {args.install_root}:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"[validate_library_layout] OK: {args.install_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
