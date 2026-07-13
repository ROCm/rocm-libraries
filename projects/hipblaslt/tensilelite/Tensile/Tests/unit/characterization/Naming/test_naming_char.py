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

"""Characterization tests for ``Tensile.SolutionStructs.Naming`` — the kernel /
solution name builders. All pure string-building over a solution ``state`` dict;
snapshot the returned name strings.
"""

import pytest

from Tensile.SolutionStructs.Problem import ProblemType
import Tensile.SolutionStructs.Naming as N

pytestmark = pytest.mark.unit


# ===========================================================================
# getParameterNameAbbreviation — uppercase-letter abbreviation
# ===========================================================================

@pytest.mark.parametrize(
    "name",
    ["MacroTile", "GlobalSplitU", "WorkGroupMapping", "lowercase", "ABC", "ThreadTile0"],
    ids=["macrotile", "gsu", "wgm", "lowercase", "allcaps", "trailingdigit"],
)
def test_get_parameter_name_abbreviation(name, snapshot):
    assert N.getParameterNameAbbreviation(name) == snapshot


# ===========================================================================
# getPrimitiveParameterValueAbbreviation — one per type branch
# ===========================================================================

def test_primitive_abbrev_str(snapshot):
    assert N.getPrimitiveParameterValueAbbreviation("K", "SomeValue") == snapshot


def test_primitive_abbrev_bool(snapshot):
    assert {
        "true": N.getPrimitiveParameterValueAbbreviation("K", True),
        "false": N.getPrimitiveParameterValueAbbreviation("K", False),
    } == snapshot


def test_primitive_abbrev_int(snapshot):
    assert {
        "zero": N.getPrimitiveParameterValueAbbreviation("K", 0),
        "pos": N.getPrimitiveParameterValueAbbreviation("K", 42),
        "neg_one": N.getPrimitiveParameterValueAbbreviation("K", -1),
        "neg_other": N.getPrimitiveParameterValueAbbreviation("K", -5),
    } == snapshot


def test_primitive_abbrev_problem_type(snapshot):
    pt = ProblemType({"DataType": 0}, False)
    assert N.getPrimitiveParameterValueAbbreviation("ProblemType", pt) == snapshot


def test_primitive_abbrev_unhandled_type_returns_none(snapshot):
    # A value matching none of the handled types falls through to implicit None.
    assert N.getPrimitiveParameterValueAbbreviation("K", None) == snapshot


def test_primitive_abbrev_float(snapshot):
    assert {
        "whole": N.getPrimitiveParameterValueAbbreviation("K", 2.0),
        "fraction": N.getPrimitiveParameterValueAbbreviation("K", 1.5),
        "fraction_small": N.getPrimitiveParameterValueAbbreviation("K", 3.07),
    } == snapshot


# ===========================================================================
# getParameterValueAbbreviation — ISA, scalar, tuple, list, dict
# ===========================================================================

def test_value_abbrev_isa(snapshot):
    assert N.getParameterValueAbbreviation("ISA", (9, 4, 10)) == snapshot


def test_value_abbrev_scalar(snapshot):
    assert N.getParameterValueAbbreviation("GlobalSplitU", 4) == snapshot


def test_value_abbrev_tuple(snapshot):
    assert N.getParameterValueAbbreviation("K", (1, 2, 3)) == snapshot


def test_value_abbrev_list(snapshot):
    assert N.getParameterValueAbbreviation("MIWaveTile", [2, 4]) == snapshot


def test_value_abbrev_dict(snapshot):
    assert N.getParameterValueAbbreviation("K", {0: 1, 2: 3}) == snapshot


# ===========================================================================
# _getName via the public wrappers
# ===========================================================================

def test_solution_name_min(make_state, snapshot):
    assert N.getSolutionNameMin(make_state(), splitGSU=False) == snapshot


def test_solution_name_full(make_state, snapshot):
    assert N.getSolutionNameFull(make_state(), splitGSU=False) == snapshot


def test_kernel_name_min(make_state, snapshot):
    assert N.getKernelNameMin(make_state(), splitGSU=False) == snapshot


def test_name_custom_kernel_early_return(make_state, snapshot):
    # CustomKernelName short-circuits _getName.
    s = make_state(CustomKernelName="MyHandwrittenKernel")
    assert {
        "min": N.getSolutionNameMin(s, splitGSU=False),
        "full": N.getSolutionNameFull(s, splitGSU=False),
        "kernel": N.getKernelNameMin(s, splitGSU=False),
    } == snapshot


def test_name_with_thread_tile_no_matrix_inst(make_state, snapshot):
    # No MatrixInstM -> the ThreadTile branch instead of MatrixInstruction.
    s = make_state()
    for k in ("MatrixInstM", "MatrixInstN", "MatrixInstB", "MIWaveTile"):
        s.pop(k, None)
    s["ThreadTile"] = [4, 4]
    assert N.getSolutionNameFull(s, splitGSU=False) == snapshot


def test_name_with_custom_main_loop_schedule(make_state, snapshot):
    s = make_state(UseCustomMainLoopSchedule=True)
    assert N.getSolutionNameFull(s, splitGSU=False) == snapshot


def test_name_no_macrotile(make_state, snapshot):
    # Drop the MacroTile/DepthU triple -> that block is skipped.
    s = make_state()
    for k in ("MacroTile0", "MacroTile1", "DepthU"):
        s.pop(k, None)
    assert N.getSolutionNameFull(s, splitGSU=False) == snapshot


def test_name_with_dict_problem_type(make_state, snapshot):
    # state["ProblemType"] as a raw config dict (not a ProblemType) -> the else
    # branch that constructs ProblemType(pt) on the fly.
    s = make_state(ProblemType={"DataType": 0, "GroupedGemm": False})
    assert N.getSolutionNameFull(s, splitGSU=False) == snapshot


def test_name_gsu_zero_no_discard(make_state, snapshot):
    # ignoreInternalArgs=True, splitGSU=False, GSU==0 -> the `GSU>0` discard is
    # NOT taken (the 160->171 false branch).
    assert N.getKernelNameMin(make_state(GlobalSplitU=0), splitGSU=False) == snapshot


def test_name_space_filling_algo_nonempty(make_state, snapshot):
    # SpaceFillingAlgo present and non-empty -> the discard is skipped and it
    # appears in the name (the 194->197 false branch).
    assert N.getSolutionNameFull(make_state(SpaceFillingAlgo=[0, 1]), splitGSU=False) == snapshot


def test_kernel_name_min_split_gsu_unit(make_state, snapshot):
    # GSU==1 with splitGSU=True: the L155 rewrite leaves GSU==1 (1 is not >1 and
    # not -1), so the subsequent `GSU > 0` discard works.
    assert N.getKernelNameMin(make_state(GlobalSplitU=1), splitGSU=True) == snapshot


@pytest.mark.parametrize("gsu", [4, -1], ids=["gsu4", "gsu_neg1"])
def test_kernel_name_min_split_gsu_typeerror(make_state, gsu):
    # CHARACTERIZED BUG: with ignoreInternalArgs=True (getKernelNameMin) and
    # splitGSU=True, GSU>1 or GSU==-1 is rewritten to the string "M" at
    # Naming.py:155, then compared with `"M" > 0` at Naming.py:160 -> TypeError.
    # Pinned as current behaviour; see resistance.md.
    with pytest.raises(TypeError):
        N.getKernelNameMin(make_state(GlobalSplitU=gsu), splitGSU=True)


def test_name_does_not_mutate_state(make_state):
    # _getName backs up and restores GlobalSplitU and ProblemType.GroupedGemm.
    s = make_state(GlobalSplitU=4)
    gg_before = s["ProblemType"]["GroupedGemm"]
    _ = N.getSolutionNameFull(s, splitGSU=True)
    assert s["GlobalSplitU"] == 4
    assert s["ProblemType"]["GroupedGemm"] == gg_before


# ===========================================================================
# getKeyNoInternalArgs
# ===========================================================================

@pytest.mark.parametrize(
    "splitGSU,gsu",
    [(True, 4), (True, 1), (False, 4), (False, 0)],
    ids=["split_gsu4", "split_gsu1", "nosplit_gsu4", "nosplit_gsu0"],
)
def test_key_no_internal_args(make_state, splitGSU, gsu, snapshot):
    s = make_state(GlobalSplitU=gsu)
    assert N.getKeyNoInternalArgs(s, splitGSU=splitGSU) == snapshot


def test_key_no_internal_args_with_codeobject_and_devices(make_state, snapshot):
    # codeObjectFile + DeviceNames are appended to the key.
    s = make_state(codeObjectFile="kernels.co", DeviceNames="gfx942")
    assert N.getKeyNoInternalArgs(s, splitGSU=False) == snapshot


def test_key_no_internal_args_restores_state(make_state):
    s = make_state(GlobalSplitU=4, WorkGroupMapping=8)
    gg_before = s["ProblemType"]["GroupedGemm"]
    _ = N.getKeyNoInternalArgs(s, splitGSU=True)
    assert s["GlobalSplitU"] == 4
    assert s["WorkGroupMapping"] == 8
    assert s["ProblemType"]["GroupedGemm"] == gg_before


# ===========================================================================
# shortenFileBase / getKernelFileBase
# ===========================================================================

def test_shorten_file_base_short(snapshot):
    # A short custom-kernel name is returned unchanged (<= MAX_FILENAME_LENGTH).
    kernel = {"CustomKernelName": "short_kernel"}
    assert N.shortenFileBase(False, kernel) == snapshot


def test_shorten_file_base_long(snapshot):
    # A name longer than MAX_FILENAME_LENGTH (64) is sha256/base64-shortened.
    kernel = {"CustomKernelName": "A" * 200}
    out = N.shortenFileBase(False, kernel)
    assert {"len_le_limit_plus_hash": len(out) <= 64 + 44, "value": out} == snapshot


def test_kernel_file_base_custom(snapshot):
    kernel = {"CustomKernelName": "ExplicitName"}
    assert N.getKernelFileBase(False, kernel) == snapshot


def test_kernel_file_base_generated(make_state, snapshot):
    # No CustomKernelName -> falls through to shortenFileBase(getKernelNameMin).
    assert N.getKernelFileBase(False, make_state()) == snapshot
