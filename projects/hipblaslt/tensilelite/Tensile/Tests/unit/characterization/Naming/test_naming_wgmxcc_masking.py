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
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the WorkGroupMappingXCC token in the key produced
by ``Tensile.SolutionStructs.Naming.getKeyNoInternalArgs``.

The rest of the suite exercises WGMXCC=1, where masking is a no-op and the
folding behavior is invisible in the key. These pin the key-level behavior with
non-1 values: a fixed (non ``-1``) WGMXCC folds to the ``1`` token so kernels
differing only in a fixed WGMXCC dedup together, while the auto value ``-1`` is
preserved as ``n1``.

Note: the ``getKeyNoInternalArgs`` mask (lines 80-81) is redundant with the mask
inside ``_getName`` (lines 160-161) — ``_getName`` re-applies the same fold
before emitting the token, so mutating the ``getKeyNoInternalArgs`` mask alone
does not change this observable. These tests therefore pin the *combined*
end-to-end masking behavior, not that specific line. See SOURCE-FINDINGS.md
(redundant WGMXCC double-mask). Uses the ``make_state`` factory from the suite
``conftest``.
"""

import re

import pytest

from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs

pytestmark = pytest.mark.unit


def _wgmxcc_token(key):
    """Return the WGMXCC value token from a key (e.g. "1", "n1"), or None."""
    m = re.search(r"_WGMXCC(n?\d+)_", key)
    return m.group(1) if m else None


def test_positive_wgmxcc_folds_to_one(make_state):
    """A fixed WGMXCC=8 produces the "1" token in the key."""
    key = getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=8), True)
    assert _wgmxcc_token(key) == "1"


def test_distinct_fixed_wgmxcc_values_share_key(make_state):
    """Two distinct fixed WGMXCC values yield the same key: kernels differing
    only in a fixed WGMXCC share one generated-assembly dedup key."""
    k8 = getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=8), True)
    k5 = getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=5), True)
    assert k8 == k5


def test_auto_wgmxcc_preserved_as_n1(make_state):
    """WGMXCC=-1 (auto) is exempt from the fold and appears as the "n1" token,
    producing a key distinct from any fixed value's key."""
    kauto = getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=-1), True)
    kfixed = getKeyNoInternalArgs(make_state(WorkGroupMappingXCC=8), True)
    assert _wgmxcc_token(kauto) == "n1"
    assert kauto != kfixed
