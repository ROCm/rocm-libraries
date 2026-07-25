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

"""Mutation-killing characterization tests for ``_getName`` in
``Tensile.SolutionStructs.Naming``. Each test pins a specific current behaviour
of the name builder with a direct assertion (no snapshots) so that an applied
mutant flips it red. The shared ``make_state`` fixture lives in the sibling
conftest.py.
"""

import pytest

import Tensile.SolutionStructs.Naming as N

pytestmark = pytest.mark.unit


def test_gsu_neg1_discarded_ignore_internal_args(make_state):
    # ignoreInternalArgs=True (getKernelNameMin), splitGSU=False, GSU==-1:
    # the `GSU > 0 or GSU == -1` guard discards GlobalSplitU, so no GSU
    # component appears. Kills the -1 -> +1 and -1 -> -2 comparison mutants.
    name = N.getKernelNameMin(make_state(GlobalSplitU=-1), splitGSU=False)
    assert "GSUn1" not in name


def test_gsu2_split_raises_typeerror(make_state):
    # ignoreInternalArgs=True, splitGSU=True, GSU==2: 2 > 1 is True, so GSU is
    # rewritten to "M" and the later `"M" > 0` comparison raises TypeError.
    # Kills the `GSU > 1` -> `GSU > 2` boundary mutant (which would leave GSU as
    # the int 2 and return a name instead).
    with pytest.raises(TypeError):
        N.getKernelNameMin(make_state(GlobalSplitU=2), splitGSU=True)


def test_grouped_gemm_masked_in_kernel_name(make_state):
    # ignoreInternalArgs=True forces ProblemType.GroupedGemm to False before the
    # ProblemType string is built, so the name is identical whether GroupedGemm
    # starts True or False. Kills the mutants that assign the mask to a bogus
    # key (leaving GroupedGemm True -> a stray "GG" tag).
    s_true = make_state()
    s_true["ProblemType"]["GroupedGemm"] = True
    name_true = N.getKernelNameMin(s_true, splitGSU=False)

    s_false = make_state()
    s_false["ProblemType"]["GroupedGemm"] = False
    name_false = N.getKernelNameMin(s_false, splitGSU=False)

    assert name_true == name_false
    assert "GG" not in name_true


def test_grouped_gemm_restored_after_kernel_name(make_state):
    # _getName masks ProblemType.GroupedGemm to False during computation then
    # restores it. Kills the restore-side mutants that write a bogus key and
    # leave GroupedGemm stuck at False.
    s = make_state()
    s["ProblemType"]["GroupedGemm"] = True
    _ = N.getKernelNameMin(s, splitGSU=False)
    assert s["ProblemType"]["GroupedGemm"] is True


def test_thread_tile_added_when_no_matrix_inst(make_state):
    # No MatrixInstM -> the else branch adds "ThreadTile" to the required set.
    # ThreadTile is NOT in the Min required set, so it only appears via this add.
    # Kills the mutants that add a mis-cased/bogus key instead.
    s = make_state()
    for k in ("MatrixInstM", "MatrixInstN", "MatrixInstB", "MIWaveTile"):
        s.pop(k, None)
    s["ThreadTile"] = [4, 4]
    name = N.getSolutionNameMin(s, splitGSU=False)
    assert "TT4_4" in name


def test_custom_kernel_name_skipped_in_loop(make_state):
    # An empty CustomKernelName does not trigger the early return, and the loop
    # explicitly skips key == "CustomKernelName". Kills the mutants that mis-case
    # that literal (which would emit a stray "CKN" component).
    s = make_state(CustomKernelName="")
    name = N.getSolutionNameFull(s, splitGSU=False)
    assert "CKN" not in name


def test_isa_uses_special_hex_format(make_state):
    # getParameterValueAbbreviation receives key="ISA" and formats the third
    # element as hex ("94a"). Kills the mutant that passes None as the key
    # (which would fall through to the plain tuple join "9410").
    s = make_state(ISA=(9, 4, 10))
    name = N.getSolutionNameMin(s, splitGSU=False)
    assert "ISA94a" in name
    assert "ISA9410" not in name


def test_wgmxcc_restored_after_name(make_state):
    # _getName backs up WorkGroupMappingXCC, forces it to 1 during computation,
    # then restores the original. Kills the backup/restore mutants that leave it
    # as None or stuck at 1.
    s = make_state(WorkGroupMappingXCC=8)
    _ = N.getSolutionNameFull(s, splitGSU=False)
    assert s["WorkGroupMappingXCC"] == 8


def test_macrotile_component_requires_all_three_keys(make_state):
    # The MacroTile component requires MacroTile0 AND MacroTile1 AND DepthU.
    # With MacroTile1 absent the block is skipped. Kills the mutants that turn an
    # `and` into `or` (which would enter the block and KeyError on MacroTile1).
    s = make_state()
    s.pop("MacroTile1", None)
    name = N.getSolutionNameFull(s, splitGSU=False)
    assert isinstance(name, str)
    assert "MT128" not in name
