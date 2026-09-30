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

"""Characterization tests for the pure table builders in
``Tensile.Common.ValidParameters``: ``makeValidWorkGroups`` /
``makeValidWMMA`` / ``makeValidSWMMAC`` / ``makeValidMFMA`` /
``makeValidSMFMA`` / ``makeValidMatrixInstructions``, plus the module-level
``validParameters`` dict.

All Tier A: pure, ``lru_cache``d builders with no environmental coupling. The
small lists are snapshotted whole; the large structures (the ``_format9``
expansions and the 700k-entry master matrix-instruction list) are pinned via a
**normalised summary** (length + a deterministic head/tail sample) so the
``.ambr`` stays reviewable while still detecting any change in shape or content.
"""

import pytest

import Tensile.Common.ValidParameters as VP

pytestmark = pytest.mark.unit


def _summarize_list(lst):
    """Deterministic, reviewable summary of a (possibly huge) list."""
    return {
        "len": len(lst),
        "head": lst[:5],
        "tail": lst[-3:],
    }


# ===========================================================================
# Small builders — snapshot whole
# ===========================================================================

def test_make_valid_work_groups(snapshot):
    # Hundreds of entries -> summary (the construction loop is fully exercised
    # regardless of how much we snapshot).
    assert _summarize_list(VP.makeValidWorkGroups()) == snapshot


def test_make_valid_wmma(snapshot):
    assert VP.makeValidWMMA() == snapshot


def test_make_valid_swmmac(snapshot):
    assert VP.makeValidSWMMAC() == snapshot


# ===========================================================================
# MFMA / SMFMA dict builders — per-key summary
# ===========================================================================

def _summarize_mi_dict(d):
    # Sorted keys, each mapped to a len + head sample. _format9 is large; the
    # base dtype-combo keys are small and their head pins their content.
    return {k: {"len": len(d[k]), "head": d[k][:3]} for k in sorted(d)}


def test_make_valid_mfma(snapshot):
    assert _summarize_mi_dict(VP.makeValidMFMA()) == snapshot


def test_make_valid_smfma(snapshot):
    assert _summarize_mi_dict(VP.makeValidSMFMA()) == snapshot


# ===========================================================================
# Master matrix-instruction list — summary + structural invariants
# ===========================================================================

def test_make_valid_matrix_instructions(snapshot):
    mi = VP.makeValidMatrixInstructions()
    summary = {
        "len": len(mi),
        "starts_with_empty_and_minus_one": mi[:2],
        "head": mi[2:10],
        "tail": mi[-3:],
    }
    assert summary == snapshot


def test_matrix_instructions_lru_cache_identity():
    # @lru_cache: repeated calls return the same cached object.
    assert VP.makeValidMatrixInstructions() is VP.makeValidMatrixInstructions()
    assert VP.makeValidMFMA() is VP.makeValidMFMA()


# ===========================================================================
# validParameters module dict — roster + structural summary
# ===========================================================================

def test_valid_parameters_key_roster(snapshot):
    # Pin the full parameter name roster (sorted) so an added/removed parameter
    # surfaces as a diff.
    assert sorted(VP.validParameters.keys()) == snapshot


def _summarize_value(v):
    if isinstance(v, list):
        return {"type": "list", "len": len(v), "head": v[:3]}
    return {"type": type(v).__name__, "value": v}


def test_valid_parameters_structure(snapshot):
    # Structural summary of every entry (type + length/value) — pins the shape
    # of the whole table without dumping the multi-hundred-thousand-entry lists.
    summary = {k: _summarize_value(VP.validParameters[k]) for k in sorted(VP.validParameters)}
    assert summary == snapshot
