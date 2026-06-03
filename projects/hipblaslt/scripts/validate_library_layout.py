#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

"""Validate the installed hipBLASLt library tree against the per-base layout.

The runtime probe in projects/hipblaslt/library/src/amd_detail/rocblaslt/
src/rocblaslt_auxiliary.cpp resolves device-library files under

    <install_prefix>/lib/hipblaslt/library/<base-arch>/<file>_<base-arch>.<ext>

with no fallback to a flat root. A producer that drops files anywhere else
(flat root, wrong arch suffix, missing per-base subdir) makes the runtime
silently miss the file and fall through to the now-removed DEFAULT_CO_PATH
escape hatch -> hipModuleLoad errors at first dispatch.

This script asserts those invariants against an installed tree without
running any GPU code or building anything. Designed to fire:

  * Locally during dev (via pytest, see
    tensilelite/Tensile/Tests/unit/test_libraryLayout.py).
  * In CI from test/therock/test_hipblaslt.py before the gtest, so layout
    bugs surface BEFORE the runtime probe does and the failure message
    points at the actual offending file.

The script is hipblaslt-specific by construction; do not invoke it for
other components.

Usage:

  validate_library_layout.py <install_root>
      install_root = directory containing lib/hipblaslt/library/<arch>/
      Returns nonzero on any violation; prints one line per violation.

  As a module:
      from validate_library_layout import validate
      violations = validate(install_root)
      if violations: ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Set

# Files every per-base subdir MUST contain. Each entry is a filename
# template; "{arch}" is substituted with the base arch name. A missing
# file is a layout violation.
REQUIRED_PER_BASE_FILES = (
    "hipblasltTransform_{arch}.hsaco",
    "hipblasltExtOpLibrary_{arch}.dat",
    "extop_{arch}.co",
)

# At least ONE TensileLibrary master file must be present per base. We
# accept both extensions (msgpack and yaml) because LibraryFormat picks
# one at build time.
TENSILE_MASTER_CANDIDATES = (
    "TensileLibrary_{arch}.dat",
    "TensileLibrary_{arch}.yaml",
)

# At least ONE lazy-load mapping file must be present per base when
# lazy loading is enabled (the default). Some build configs disable
# lazy loading; we skip the lazy check if neither candidate exists AND
# a non-lazy master is present (best-effort distinction).
TENSILE_LAZY_CANDIDATES = (
    "TensileLibrary_lazy_{arch}.dat",
    "TensileLibrary_lazy_{arch}.yaml",
)

# Per-arch files allowed ONLY for that arch. The key is the arch; the
# value is a tuple of additional required filenames. Used for kernels
# gated on specific archs (e.g. rocroller's gfx950-only custom_kernels).
PER_ARCH_REQUIRED = {
    # rocroller's gfx950 custom kernel is gated via add_subdirectory in
    # projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocroller/
    # custom_kernels/CMakeLists.txt — it should only land in library/gfx950/
    "gfx950": ("rr_custom_kernels_gfx950.co",),
}

# Files that MUST NOT exist at the library root. Each entry is the
# legacy flat-layout filename a regression would resurrect. Filenames
# with "{arch}" patterns are checked against every cooked arch we saw
# in subdirs (so a stray gfx942-suffixed file at the root counts too).
FORBIDDEN_FLAT_ROOT_BASENAMES = (
    "TensileLibrary.dat",
    "TensileLibrary.yaml",
    "TensileLibrary_lazy.dat",
    "TensileLibrary_lazy.yaml",
    "hipblasltTransform.hsaco",
    "hipblasltExtOpLibrary.dat",
    "extop.co",
)

# Filename suffix arch must match the dir name. Bundler-target arch
# strings like `gfx942-sramecc+-xnack+` are valid filename suffixes for
# the gfx942 dir (target features survive in the filename); we accept
# any suffix that starts with the dir's base arch.
_GFX_PREFIX_RE = re.compile(r"^gfx[a-z0-9]+(?:[\-:][\-+a-z0-9]+)*$")


def _arch_dir_name_is_base(name: str) -> bool:
    """The dir under library/ must be a bare base arch (no colon-features,
    no dash-features). Cooked variants are filename-only.

    Accept gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950, gfx10xx,
    gfx11xx, gfx12xx, gfx1250 — anything matching `gfx[a-z0-9]+$`.
    """
    return bool(re.fullmatch(r"gfx[a-z0-9]+", name))


def _filename_arch_matches_dir(filename: str, base_arch: str) -> bool:
    """True if filename contains the dir's base arch as a delimited token.

    The arch token can appear:
      - at the end of the stem:           `foo_gfx942.dat`
      - in the middle, delimited by `_`:  `TensileLibrary_lazy_gfx942_Mapping.dat`
      - as a cooked variant:              `kernel_gfx942-sramecc+-xnack+.hsaco`
        (target features survive in the filename; base must still match)
      - dash-delimited inside a dotted name: `Kernels.so-000-gfx90a.hsaco`
        (Tensile/rocRoller code-object naming)

    What we forbid is the WRONG base arch in a sibling dir (e.g. a
    `_gfx950` file under `library/gfx942/`). Anything matching the dir's
    base — alone or carrying additional `-feature[+-]` tokens — passes.
    """
    # Search the WHOLE filename (not just the text before the first dot) for a
    # gfx arch token delimited on the left by start-of-string or one of [._-],
    # capturing the base arch plus any cooked `-feature`/`+feature` tail. A
    # wrong base arch (e.g. a `_gfx950` file under library/gfx942/) matches no
    # token for gfx942 -> violation, preserving the misplacement check.
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
    """Resolve the canonical library directory under install_root.

    Checks both standard layouts:
      - <install_root>/lib/hipblaslt/library/        (rocm-libraries / TheRock install)
      - <install_root>/library/                       (build tree, when HIPBLASLT_TENSILE_LIBPATH is the build root)
    Returns the first that exists, else None.
    """
    candidates = (
        install_root / "lib" / "hipblaslt" / "library",
        install_root / "library",
    )
    for c in candidates:
        if c.is_dir():
            return c
    return None


def validate(install_root: Path) -> List[str]:
    """Walk the hipblaslt library tree and return a list of violations.

    Empty list = layout OK. Each violation is a single human-readable
    string suitable for a CI error log; includes the absolute path of
    the offending file or dir so devs can grep their build tree.
    """
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

    # ---- 1. Flat-root files must not exist ----
    for basename in FORBIDDEN_FLAT_ROOT_BASENAMES:
        offender = library_dir / basename
        if offender.is_file():
            violations.append(
                f"flat-root file found (per-base layout violation): {offender}"
            )

    # Also reject any non-gfx file/dir entries at the root that look like
    # library payload (a .dat / .co / .hsaco at the root is wrong even
    # if its name isn't in the canonical forbidden list).
    for entry in library_dir.iterdir():
        if entry.is_file() and entry.suffix in (".dat", ".co", ".hsaco", ".yaml"):
            violations.append(
                f"unexpected payload file at library root: {entry} "
                f"(per-base layout requires files in library/<base>/)"
            )

    # ---- 2. Discover per-base subdirs ----
    base_arch_dirs = sorted(
        p for p in library_dir.iterdir() if p.is_dir() and p.name.startswith("gfx")
    )
    if not base_arch_dirs:
        violations.append(f"no per-base gfx* subdirs found under {library_dir}")
        return violations

    # Reject dirs whose name is not a bare base arch (e.g. gfx942-sramecc+).
    # Cooked variants must live in the filename, not the directory.
    for d in base_arch_dirs:
        if not _arch_dir_name_is_base(d.name):
            violations.append(
                f"library subdir name carries target features (must be bare base arch): {d}"
            )

    # ---- 3. Per-base required files + filename suffix check ----
    for arch_dir in base_arch_dirs:
        base = arch_dir.name
        if not _arch_dir_name_is_base(base):
            continue  # already reported

        entries: Set[str] = {p.name for p in arch_dir.iterdir() if p.is_file()}

        # Required canonical files
        for template in REQUIRED_PER_BASE_FILES:
            wanted = template.format(arch=base)
            if wanted not in entries:
                violations.append(
                    f"missing required file in {arch_dir}: {wanted}"
                )

        # At least one TensileLibrary master must exist
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

        # Per-arch-required (e.g. rocroller's gfx950 custom kernel)
        for extra in PER_ARCH_REQUIRED.get(base, ()):
            if extra not in entries:
                violations.append(
                    f"missing required {base}-only file in {arch_dir}: {extra}"
                )

        # Filename suffix must match the dir's base arch
        for fname in entries:
            # metadata.yaml is a known sidecar with no arch suffix; allow it.
            if fname == "metadata.yaml":
                continue
            if not _filename_arch_matches_dir(fname, base):
                violations.append(
                    f"filename arch suffix does not match dir {base}: {arch_dir / fname}"
                )

    # ---- 4. Per-arch files must not appear under the WRONG arch ----
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
        "--quiet", "-q", action="store_true",
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
