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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Post-build integrity checks for installed hipBLASLt Tensile .dat files.

These tests run against an *already-built* library directory; they do not
build anything themselves and they do not need a GPU. Set the env var
``HIPBLASLT_TEST_LIBRARY_DIR`` to the directory containing the installed
``Tensile/library`` artifacts to activate them. When the env var is unset
(typical local invocation), the tests skip and the suite passes.

Each assertion is constructed to fail under the *unpatched* producer (whose
per-arch Mapping filter dropped fallback entries) for a specific reason, so
the suite discriminates the regression class rather than just rubber-stamping
whatever the build produced.
"""

import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest

try:
    import msgpack
except ImportError:
    msgpack = None


_LIB_DIR_ENV = "HIPBLASLT_TEST_LIBRARY_DIR"

# Master library filename:    "TensileLibrary_lazy_<arch>.dat"
# Per-arch Mapping filename:  "TensileLiteLibrary_lazy_<arch>_Mapping.dat"
_MASTER_RE = re.compile(r"^TensileLibrary_lazy_(?P<arch>[A-Za-z0-9]+)\.dat$")
_MAPPING_RE = re.compile(r"^TensileLiteLibrary_lazy_(?P<arch>[A-Za-z0-9]+)_Mapping\.dat$")


def _libDirOrSkip() -> Path:
    raw = os.environ.get(_LIB_DIR_ENV)
    if not raw:
        pytest.skip(
            f"set {_LIB_DIR_ENV} to an installed library directory "
            "to run dat-file integrity checks"
        )
    if msgpack is None:
        pytest.skip("msgpack not installed; cannot read .dat files")
    p = Path(raw)
    if not p.is_dir():
        pytest.fail(f"{_LIB_DIR_ENV}={raw!r} is not a directory")
    return p


def _archsPresent(libDir: Path):
    """Return the set of arches that have BOTH a master library file and a
    per-arch Mapping file in the installed directory."""
    masters = {m.group("arch") for f in libDir.iterdir()
               if (m := _MASTER_RE.match(f.name))}
    mappings = {m.group("arch") for f in libDir.iterdir()
                if (m := _MAPPING_RE.match(f.name))}
    return masters & mappings


def _loadMapping(libDir: Path, arch: str):
    path = libDir / f"TensileLiteLibrary_lazy_{arch}_Mapping.dat"
    with open(path, "rb") as f:
        return msgpack.unpack(f, raw=False)

# TODO: Tests always skipped — need markers to run in CI.
# Issue: https://github.com/ROCm/rocm-libraries/issues/7486

# ---------------------------------------------------------------------------
# 1. Discovery: a built directory must have at least one (master, Mapping) pair.
# ---------------------------------------------------------------------------
def test_libraryDirHasMatchingMasterAndMappingFiles():
    libDir = _libDirOrSkip()
    archs = _archsPresent(libDir)
    masters_only = {_MASTER_RE.match(f.name).group("arch")
                    for f in libDir.iterdir() if _MASTER_RE.match(f.name)} - archs
    mappings_only = {_MAPPING_RE.match(f.name).group("arch")
                     for f in libDir.iterdir() if _MAPPING_RE.match(f.name)} - archs

    assert archs, (
        f"{libDir} contains no matched (master, Mapping) pair. "
        "If only TensileLibrary_lazy_<arch>.dat files are present, the "
        "producer never wrote per-arch Mapping files (legacy single shared "
        "Mapping) and runtime will not be able to resolve lazy lookups."
    )
    assert not masters_only, (
        f"masters without matching per-arch Mapping: {sorted(masters_only)}. "
        "These archs cannot resolve any lazy library at runtime."
    )
    assert not mappings_only, (
        f"per-arch Mapping files without matching master: {sorted(mappings_only)}. "
        "Likely a leftover from an earlier build; would cause arch-derivation "
        "lookups to find a Mapping whose master lives elsewhere."
    )


# ---------------------------------------------------------------------------
# 2. Every Mapping value names a real .dat file on disk.
#
# Discriminator: catches the producer dropping fallback entries (would leave
# referenced files behind without Mapping pointers) AND catches the dual case
# where the producer references arch-suffixed files it didn't actually emit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("_dummy", [None])  # parametrize so per-arch IDs show
def test_everyMappingValueResolvesToAFileOnDisk(_dummy):
    libDir = _libDirOrSkip()
    archs = _archsPresent(libDir)
    if not archs:
        pytest.skip("no matched master/Mapping pairs to check")

    missing = defaultdict(list)
    for arch in sorted(archs):
        mapping = _loadMapping(libDir, arch)
        # mapping is {int -> str} where str is a kernel filename stem
        for idx, kernelName in mapping.items():
            datPath = libDir / f"{kernelName}.dat"
            if not datPath.is_file():
                missing[arch].append((idx, kernelName))

    assert not missing, (
        "Mapping references kernel files that do not exist on disk:\n"
        + "\n".join(f"  arch={a}: {entries[:5]}{' ...' if len(entries) > 5 else ''}"
                    for a, entries in missing.items())
    )


# ---------------------------------------------------------------------------
# 3. Per-arch Mapping must contain BOTH tuned and fallback entries.
#
# This is the assertion that would have caught the CI failure: under the
# unpatched producer, Mapping for gfx942 only contains "*_gfx942" entries
# because the filter was `name.endswith("_gfx942")` and fallback names ended
# in "_fallback" (no arch). Asserting fallback presence locks in the fix.
#
# (Skipped when no fallback solutions exist for the build's archs at all,
# which is rare in practice but possible for a single tightly-tuned arch.)
# ---------------------------------------------------------------------------
def test_perArchMappingIncludesFallbackEntries():
    libDir = _libDirOrSkip()
    archs = _archsPresent(libDir)
    if not archs:
        pytest.skip("no matched master/Mapping pairs to check")

    # First scan to decide whether fallback files exist for this build;
    # if not, there's nothing for the per-arch Mapping to point at.
    fallback_dat_files = list(libDir.glob("*_fallback_*.dat"))
    if not fallback_dat_files:
        pytest.skip(
            "no *_fallback_<arch>.dat files in build output; either the "
            "build had no fallback logic for these archs, or the renamed "
            "fallback files were never produced"
        )

    archs_missing_fallback_in_mapping = []
    for arch in sorted(archs):
        mapping = _loadMapping(libDir, arch)
        # values look like "TensileLibrary_..._fallback_<arch>" or "..._<arch>"
        has_fallback = any(
            "_fallback_" in name and name.endswith("_" + arch)
            for name in mapping.values()
        )
        # Confirm fallback files for this arch actually exist; only then is
        # missing fallback in the Mapping a real bug rather than a no-op.
        arch_fallback_files = [
            f for f in fallback_dat_files
            if f.name.endswith(f"_fallback_{arch}.dat")
        ]
        if arch_fallback_files and not has_fallback:
            archs_missing_fallback_in_mapping.append(arch)

    assert not archs_missing_fallback_in_mapping, (
        "Per-arch Mapping is missing fallback entries for: "
        f"{archs_missing_fallback_in_mapping}. This means the runtime cannot "
        "resolve solutions whose dtype lives only in the fallback library "
        "(symptom: 'NO solution found!' for that dtype). The producer "
        "must rename fallback lazy-library names to '*_fallback_<arch>' "
        "before writing the per-arch Mapping; see renameFallbacksPerArch."
    )


# ---------------------------------------------------------------------------
# 4. Every Mapping value's name must end with the Mapping's own arch.
# Sanity check: catches accidental cross-arch references.
# ---------------------------------------------------------------------------
def test_mappingValuesAreArchScoped():
    libDir = _libDirOrSkip()
    archs = _archsPresent(libDir)
    if not archs:
        pytest.skip("no matched master/Mapping pairs to check")

    cross = defaultdict(list)
    for arch in sorted(archs):
        mapping = _loadMapping(libDir, arch)
        for idx, name in mapping.items():
            if not name.endswith("_" + arch):
                cross[arch].append((idx, name))

    assert not cross, (
        "Per-arch Mapping has entries that don't end with the arch suffix:\n"
        + "\n".join(f"  arch={a}: first 3 = {entries[:3]}" for a, entries in cross.items())
        + "\nA mismatch here means kpack overlay collisions can still occur "
        "(two arches' Mapping files would name the same kernel file)."
    )
