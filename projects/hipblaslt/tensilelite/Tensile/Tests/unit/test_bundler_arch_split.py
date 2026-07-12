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

"""Tests for the bundler-target -> (filenameArch, baseArch) split.

A previous bug used ``re.sub(":", "-", rawArch).split("-xnack")[0]`` to
derive the per-base directory. For ``gfx942:sramecc+:xnack+`` this yielded
``gfx942-sramecc+`` instead of ``gfx942``, silently placing kernels in the
wrong subdir. The runtime probe in tensile_host.cpp strips at the first
colon (``gfx942``), so any cooked variant must collapse to that same base.
"""

import pytest

from Tensile.Toolchain.Source import _archNamesFromBundlerTarget

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "rawArch,expectedFilename,expectedBase",
    [
        ("gfx942",                     "gfx942",                     "gfx942"),
        ("gfx942:xnack+",              "gfx942-xnack+",              "gfx942"),
        ("gfx942:xnack-",              "gfx942-xnack-",              "gfx942"),
        ("gfx942:sramecc+",            "gfx942",                     "gfx942"),
        ("gfx942:sramecc+:xnack+",     "gfx942-xnack+",              "gfx942"),
        ("gfx942:sramecc-:xnack-",     "gfx942-xnack-",              "gfx942"),
        ("gfx950:sramecc+:xnack+",     "gfx950-xnack+",              "gfx950"),
        ("gfx1250",                    "gfx1250",                    "gfx1250"),
    ],
)
def test_archNamesFromBundlerTarget(rawArch, expectedFilename, expectedBase):
    filenameArch, baseArch = _archNamesFromBundlerTarget(rawArch)
    assert filenameArch == expectedFilename
    assert baseArch     == expectedBase


def test_baseArch_strips_sramecc_not_just_xnack():
    # The historical bug: arch with sramecc but no xnack split at "-xnack" got
    # nothing stripped, so the directory included `-sramecc+`. The base must
    # strip everything past (and including) the first colon.
    _, base = _archNamesFromBundlerTarget("gfx942:sramecc+")
    assert base == "gfx942", "sramecc must be stripped from the per-base dir"


def test_baseArch_collapses_all_cooked_variants_to_same_dir():
    # Three cooked variants of one base arch must yield the same per-base dir,
    # so xnack+/xnack- code objects co-locate under one library/<base>/.
    cooked = [
        "gfx942",
        "gfx942:xnack+",
        "gfx942:xnack-",
        "gfx942:sramecc+:xnack+",
        "gfx942:sramecc+:xnack-",
    ]
    bases = {_archNamesFromBundlerTarget(c)[1] for c in cooked}
    assert bases == {"gfx942"}


@pytest.mark.parametrize(
    "rawArch",
    [
        "gfx942:sramecc+",
        "gfx942:sramecc+:xnack+",
        "gfx942:sramecc-:xnack-",
        "gfx950:sramecc+:xnack+",
    ],
)
def test_filename_never_carries_sramecc(rawArch):
    """The runtime helper-kernel loader (HipSolutionAdapter / tensile_host) only
    probes ``{"", "-xnack-", "-xnack+"}`` appended to the base arch. A sramecc
    suffix in the produced filename yields a code object the runtime can never
    load -> "named symbol not found" for every source/PostGSU helper kernel.
    """
    filenameArch, _ = _archNamesFromBundlerTarget(rawArch)
    assert "sramecc" not in filenameArch


@pytest.mark.parametrize(
    "rawArch,expectedHelperName",
    [
        ("gfx942",                 "Kernels.so-000-gfx942.hsaco"),
        ("gfx942:sramecc+:xnack+", "Kernels.so-000-gfx942-xnack+.hsaco"),
        ("gfx942:sramecc-:xnack-", "Kernels.so-000-gfx942-xnack-.hsaco"),
    ],
)
def test_helper_kernel_name_matches_runtime_probe(rawArch, expectedHelperName):
    """The runtime builds ``"Kernels.so-000-" + base + {"", "-xnack-", "-xnack+"}``."""
    filenameArch, _ = _archNamesFromBundlerTarget(rawArch)
    assert "Kernels.so-000-{0}.hsaco".format(filenameArch) == expectedHelperName
