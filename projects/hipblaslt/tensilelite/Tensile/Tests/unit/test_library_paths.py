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

"""Tests for the per-base library path helpers in Run.py.

The runtime probe in tensile_host.cpp strips the cooked arch string at the
first colon (`gfx942:xnack+` -> `gfx942`) before opening
`library/<base>/TensileLibrary.dat`. The producer must lay artifacts down in
that same per-base shape. These helpers are the single source of truth for
that path layout, so the invariants tested here -- colon stripping, dedup
across cooked variants, and the single-arch fallback used by auxiliary
tooling -- are the same invariants the runtime depends on.
"""

from pathlib import Path

import pytest

from Tensile.TensileCreateLibrary.Run import (
    _baseArchs,
    _singleArchDir,
    libraryDir,
    libraryRoot,
)

# The path-based auto-marker in Tensile/Tests/conftest.py only fires when
# pytest is invoked from that root; the coverage tox env runs
# `pytest -m unit Tensile/Tests/unit` and would silently skip this file
# without the explicit declaration (matches test_helper_cache.py /
# test_datFileIntegrity.py convention).
pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# libraryRoot / libraryDir: path composition and colon stripping
# ---------------------------------------------------------------------------
def test_libraryRoot_appendsLibrarySegment(tmp_path):
    assert libraryRoot(tmp_path) == tmp_path / "library"


def test_libraryRoot_acceptsStringPath(tmp_path):
    assert libraryRoot(str(tmp_path)) == tmp_path / "library"


def test_libraryDir_bareArch(tmp_path):
    assert libraryDir(tmp_path, "gfx942") == tmp_path / "library" / "gfx942"


@pytest.mark.parametrize(
    "cooked",
    ["gfx942:xnack+", "gfx942:xnack-", "gfx942:sramecc+:xnack-"],
)
def test_libraryDir_stripsTargetFeatures(tmp_path, cooked):
    # Runtime probes library/<base>/, so every cooked variant must collapse
    # to the same base directory.
    assert libraryDir(tmp_path, cooked) == tmp_path / "library" / "gfx942"


def test_libraryDir_acceptsStringPath(tmp_path):
    assert libraryDir(str(tmp_path), "gfx950") == tmp_path / "library" / "gfx950"


# ---------------------------------------------------------------------------
# _baseArchs: dedup + sort
# ---------------------------------------------------------------------------
def test_baseArchs_dedupsCookedVariants():
    # All three collapse to gfx942 -- the fanout would otherwise write the
    # same subdir three times.
    assert _baseArchs(["gfx942", "gfx942:xnack+", "gfx942:xnack-"]) == ["gfx942"]


def test_baseArchs_sortsForDeterminism():
    # Build order must not depend on set iteration order, otherwise the
    # produced .dat ordering would vary across runs and break diffing.
    assert _baseArchs(["gfx1100", "gfx942", "gfx950"]) == [
        "gfx1100",
        "gfx942",
        "gfx950",
    ]


def test_baseArchs_emptyInput():
    assert _baseArchs([]) == []


# ---------------------------------------------------------------------------
# _singleArchDir: fallback used by auxiliary tooling that has no arch in hand
# ---------------------------------------------------------------------------
def test_singleArchDir_returnsLoneSubdir(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "gfx942").mkdir()
    assert _singleArchDir(tmp_path) == lib / "gfx942"


def test_singleArchDir_acceptsStringPath(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "gfx950").mkdir()
    assert _singleArchDir(str(tmp_path)) == lib / "gfx950"


def test_singleArchDir_ignoresNonGfxEntries(tmp_path):
    # Auxiliary metadata (caches, source dirs) lives alongside the gfx
    # subdir; only the gfx* directory should be picked.
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "gfx942").mkdir()
    (lib / "cache").mkdir()
    (lib / "TensileLibrary.dat").write_text("")
    assert _singleArchDir(tmp_path) == lib / "gfx942"


def test_singleArchDir_raisesOnMissingRoot(tmp_path):
    with pytest.raises(FileNotFoundError, match="library root does not exist"):
        _singleArchDir(tmp_path)


def test_singleArchDir_raisesOnZeroGfxSubdirs(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    with pytest.raises(RuntimeError, match="expected exactly one gfx subdir"):
        _singleArchDir(tmp_path)


def test_singleArchDir_raisesOnMultipleGfxSubdirs(tmp_path):
    # The whole point of this helper is to fail loudly when the caller's
    # single-arch assumption is wrong -- if it silently picked one, the
    # auxiliary tool would write to the wrong arch's directory.
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "gfx942").mkdir()
    (lib / "gfx950").mkdir()
    with pytest.raises(RuntimeError, match="Caller must pass the arch explicitly"):
        _singleArchDir(tmp_path)
