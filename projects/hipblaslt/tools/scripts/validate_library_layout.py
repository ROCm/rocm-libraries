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
    "TensileLibrary_{arch}.dat.zlib",
    "TensileLibrary_{arch}.yaml",
)

TENSILE_LAZY_CANDIDATES = (
    "TensileLibrary_lazy_{arch}.dat",
    "TensileLibrary_lazy_{arch}.dat.zlib",
    "TensileLibrary_lazy_{arch}.yaml",
)

PER_ARCH_REQUIRED = {
    "gfx950": ("rr_custom_kernels_gfx950.co",),
}

FORBIDDEN_FLAT_ROOT_BASENAMES = (
    "TensileLibrary.dat",
    "TensileLibrary.dat.zlib",
    "TensileLibrary.yaml",
    "TensileLibrary_lazy.dat",
    "TensileLibrary_lazy.dat.zlib",
    "TensileLibrary_lazy.yaml",
    "hipblasltTransform.hsaco",
    "hipblasltExtOpLibrary.dat",
    "hipblasltExtOpLibrary.dat.zlib",
    "extop.co",
)

_GFX_PREFIX_RE = re.compile(r"^gfx[a-z0-9]+(?:[\-:][\-+a-z0-9]+)*$")


def _arch_dir_name_is_base(name: str) -> bool:
    return bool(re.fullmatch(r"gfx[a-z0-9]+", name))


def _stepping_base(name: str) -> Optional[str]:
    """The architecture a stepping subtree belongs to, or None if it is not one.

    Subtree names are otherwise bare -- target features never reach them -- so a
    hyphen is the stepping, as in library/gfx1250-strict/. The architecture it
    returns is only for rules that are about the shared ISA; the files inside are
    validated against the subtree's own name, since that is the name the runtime
    reports for the silicon and therefore the one it forms filenames from.
    """
    base, sep, _ = name.partition("-")
    return base if sep and _arch_dir_name_is_base(base) else None


_ARCH_IN_FILENAME_RE = re.compile(
    # The trailing [-+] is the feature's state, and it has to be captured: it is
    # what tells gfx942-xnack+ apart from gfx1250-strict.
    r"(?:^|[._-])(?P<arch>gfx[0-9a-z]+(?:[-+][0-9a-z]+[-+]?)*)"
)

# A target feature always carries its state, as in gfx942-xnack+; a stepping
# never does, as in gfx1250-strict. That sign is the only thing separating the
# two spellings, and the distinction decides which subtree a file belongs in.
# A name may carry several features, as in gfx942-sramecc+-xnack-, so every
# token has to be signed -- requiring it of only the first would reject that.
_TARGET_FEATURE_RE = re.compile(r"^[a-z0-9]+[-+](?:-[a-z0-9]+[-+])*$")


def _filename_arch_matches_dir(filename: str, accepted: str) -> bool:
    """Whether the architecture in ``filename`` is the one the subtree accepts.

    A name may carry target features on top of it, but it may not carry a
    stepping: gfx1250-strict is its own architecture with its own subtree, and
    a code object for it cannot load on the gfx1250 silicon whose directory it
    would be sitting in.
    """
    for m in _ARCH_IN_FILENAME_RE.finditer(filename):
        found = m.group("arch")
        if found == accepted:
            return True
        if found.startswith(accepted + "-") and _TARGET_FEATURE_RE.match(
            found[len(accepted) + 1 :]
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


def _has_required_file(entries: Set[str], template: str, arch: str) -> bool:
    """Check if entries contains the required file or its .zlib variant (for .dat only)."""
    wanted = template.format(arch=arch)
    if wanted in entries:
        return True
    if wanted.endswith(".dat"):
        return (wanted + ".zlib") in entries
    return False


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
        if entry.is_file() and entry.suffix in (".dat", ".zlib", ".co", ".hsaco", ".yaml"):
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
        # A stepping subtree is the one legitimate non-bare name.
        if _stepping_base(d.name):
            continue
        if not _arch_dir_name_is_base(d.name):
            violations.append(
                f"library subdir name carries target features (must be bare base arch): {d}"
            )

    for arch_dir in base_arch_dirs:
        # `base` is the architecture a stepping subtree belongs to, used only
        # where a rule is about the ISA. Everything the runtime opens by name is
        # keyed on the subtree's own name instead (see _stepping_base).
        stepping_of = _stepping_base(arch_dir.name)
        base = stepping_of or arch_dir.name
        if not _arch_dir_name_is_base(base):
            continue

        entries: Set[str] = {p.name for p in arch_dir.iterdir() if p.is_file()}

        for fname in sorted(entries):
            if fname.endswith(".dat") and (fname + ".zlib") in entries:
                violations.append(
                    f"both compressed and uncompressed payloads present in {arch_dir}: "
                    f"{fname} and {fname}.zlib (stale .dat shadows the .zlib at runtime; "
                    f"the producer must remove the uncompressed sibling)"
                )

        # ExtOp and Transform are resolved at runtime from gcnArchName, which
        # getExtOpLibraryPath trims only at ':' -- so on a stepping device both
        # the directory and the filename it opens carry the stepping, and the
        # subtree needs its own copies rather than inheriting the
        # architecture's.
        for template in REQUIRED_PER_BASE_FILES:
            if not _has_required_file(entries, template, arch_dir.name):
                violations.append(
                    f"missing required file in {arch_dir}: {template.format(arch=arch_dir.name)}"
                )

        # Every subtree, stepping or not, is named for what the runtime reports
        # and opens the master spelled the same way, so one name serves both.
        master_present = any(t.format(arch=arch_dir.name) in entries for t in TENSILE_MASTER_CANDIDATES)
        lazy_present = any(t.format(arch=arch_dir.name) in entries for t in TENSILE_LAZY_CANDIDATES)
        if not (master_present or lazy_present):
            violations.append(
                f"missing TensileLibrary master/lazy file for {arch_dir.name} in {arch_dir} "
                f"(expected one of: TensileLibrary_{arch_dir.name}.{{dat,dat.zlib,yaml}} or "
                f"TensileLibrary_lazy_{arch_dir.name}.{{dat,dat.zlib,yaml}})"
            )

        for extra in PER_ARCH_REQUIRED.get(base, ()):
            if extra not in entries:
                violations.append(
                    f"missing required {base}-only file in {arch_dir}: {extra}"
                )

        for fname in entries:
            if fname == "metadata.yaml":
                continue
            if not _filename_arch_matches_dir(fname, arch_dir.name):
                if stepping_of:
                    violations.append(
                        f"filename in the {arch_dir.name} subtree is not named for it: "
                        f"{arch_dir / fname} (a stepping is what the runtime reports for "
                        f"the silicon, so it forms filenames from {arch_dir.name}, not "
                        f"from the {base} it shares an ISA with; this file is never opened)"
                    )
                else:
                    violations.append(
                        f"filename arch does not match dir {base}: {arch_dir / fname} "
                        f"(a stepping is a separate architecture with its own subtree; "
                        f"its code objects carry an ELF machine code {base} cannot load)"
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
