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

"""Characterization tests for the ``Solution`` class (slice 2 of the Solution.py
campaign): construction (a real Solution built from the vendored logic fixture),
the ``Mapping`` interface, identity/hash/equality, and the simpler static
``state``-helpers. The reject-heavy parameter-derivation statics are deferred to
slice 3 (see resistance.md).
"""

import pytest

# `from ... import Solution` correctly yields the class (the package __init__
# re-exports it). The module's static methods are reached via the class.
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# ===========================================================================
# Construction
# ===========================================================================

def test_solution_construction(solution, solution_summary, snapshot):
    # A real Solution parsed from the vendored gfx942 HSS_BH logic fixture.
    assert solution_summary(solution) == snapshot


def test_all_solutions_constructed(solutions, snapshot):
    # Pin how many solutions the fixture yields + each one's name.
    assert {"count": len(solutions), "names": [str(s) for s in solutions]} == snapshot


# ===========================================================================
# Mapping interface
# ===========================================================================

def test_mapping_interface(solution, snapshot):
    summary = {
        "len": len(solution),
        "iter_matches_keys": sorted(iter(solution)) == sorted(solution.keys()),
        "keys_is_list": isinstance(solution.keys(), list),
        "getitem_kernel_language": solution["KernelLanguage"],
        "getattributes_is_state": solution.getAttributes() is solution._state,
    }
    assert summary == snapshot


def test_setitem_roundtrip(solution):
    # __setitem__ writes through to _state (and invalidates the cached name).
    saved = solution["WorkGroupMapping"]
    try:
        solution["WorkGroupMapping"] = 7
        assert solution["WorkGroupMapping"] == 7
    finally:
        solution["WorkGroupMapping"] = saved


# ===========================================================================
# Identity: __str__ / __repr__ / __hash__ / __eq__ / __ne__
# ===========================================================================

def test_str_repr_agree(solution):
    assert repr(solution) == str(solution)


def test_hash_and_equality(solutions):
    import copy

    a = solutions[0]
    assert a == a                       # same name + same DeviceNames -> equal
    assert hash(a) == hash(a)
    # vs a non-Solution -> the isinstance guard returns False.
    assert (a == "not a solution") is False
    assert (a != "not a solution") is True
    # A Solution with a different name -> the str-mismatch branch returns False.
    other = copy.copy(a)
    other._name = "A_DIFFERENT_SOLUTION_NAME"
    assert a != other
    assert hash(a) != hash(other)


# ===========================================================================
# Static helpers (the simple, non-reject ones)
# ===========================================================================

def test_get_mi_output_info(solution, isa_info_map, snapshot):
    assert Solution.getMIOutputInfo(solution._state, isa_info_map) == snapshot


def test_is_direct_to_vgpr_support_data_type(solutions, snapshot):
    result = {str(s)[:40]: Solution.isDirectToVgprSupportDataType(s._state) for s in solutions}
    assert result == snapshot


@pytest.mark.parametrize(
    "group,vector",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["group_vec", "group_novec", "nogroup_vec", "nogroup_novec"],
)
@pytest.mark.parametrize("tc", ["A", "B"], ids=["A", "B"])
def test_get_divisor_name(tc, group, vector, snapshot):
    state = {
        f"GlobalReadCoalesceGroup{tc}": group,
        f"GlobalReadCoalesceVector{tc}": vector,
    }
    assert Solution.getDivisorName(state, tc) == snapshot


# ===========================================================================
# Kernel accessors
# ===========================================================================

def test_get_kernels_shape(solution, snapshot):
    kernels = solution.getKernels()
    summary = {
        "num_kernels": len(kernels),
        "all_dicts_or_solutions": all(hasattr(k, "keys") for k in kernels),
        "first_is_self": kernels[0] is solution,
    }
    assert summary == snapshot


def test_kernel_betaonly_conversion_accessors_need_pipeline_state(solution):
    # getKernelBetaOlnyObjects / getKernelConversionObjects read attributes
    # (betaOnlyKernelObjects / conversionKernelObjects) populated by a later
    # pipeline stage, not by parse/construction -> AttributeError on a freshly
    # built Solution. Pinned as current behaviour; see resistance.md.
    with pytest.raises(AttributeError):
        solution.getKernelBetaOlnyObjects()
    with pytest.raises(AttributeError):
        solution.getKernelConversionObjects()
