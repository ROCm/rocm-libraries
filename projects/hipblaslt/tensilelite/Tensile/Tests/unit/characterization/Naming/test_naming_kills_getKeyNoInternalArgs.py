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

"""Mutation-killing characterization tests for
``Tensile.SolutionStructs.Naming.getKeyNoInternalArgs``.

Each test pins an observable behavior of the current implementation: the
``_state`` unwrapping, GroupedGemm masking and restore, GlobalSplitU masking
under ``splitGSU``, and the WorkGroupMappingXCC backup/restore that leaves the
input state (and ProblemType) unpolluted. Uses the ``make_state`` factory from
the suite ``conftest``.
"""

import pytest

from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs

pytestmark = pytest.mark.unit


class _Wrapped:
    """A Solution-like holder exposing a ``_state`` dict but not subscriptable
    itself, so ``getKeyNoInternalArgs`` MUST use ``state._state``."""

    def __init__(self, state):
        self._state = state


def test_unwraps_state_attr(make_state):
    """Kills mutants that break ``hasattr(state, '_state')`` unwrapping
    (mutmut_2/6/7): a wrapper object is not subscriptable, so any path that
    fails to reach ``._state`` raises instead of producing the dict's key."""
    plain = make_state()
    wrapped = _Wrapped(make_state())
    assert getKeyNoInternalArgs(wrapped, True) == getKeyNoInternalArgs(plain, True)


def test_masks_grouped_gemm_in_key(make_state):
    """Kills mutants that write the GroupedGemm mask to the wrong ProblemType
    key (mutmut_26/27/28). The mask is gated on ``SupportUserArgs``: with it off,
    GroupedGemm is forced to False for the key, so a GroupedGemm=True state and a
    GroupedGemm=False state produce the SAME key. With it on the mask is skipped,
    so the two keys DIFFER and the GroupedGemm=True key carries a distinguishing
    ``_GG_`` tag -- pinning the ``if not SupportUserArgs`` guard (kills its flip)."""
    s_true = make_state()
    s_true["ProblemType"]["SupportUserArgs"] = False
    s_true["ProblemType"]["GroupedGemm"] = True
    s_false = make_state()
    s_false["ProblemType"]["SupportUserArgs"] = False
    s_false["ProblemType"]["GroupedGemm"] = False
    assert getKeyNoInternalArgs(s_true, True) == getKeyNoInternalArgs(s_false, True)

    g_true = make_state()
    g_true["ProblemType"]["SupportUserArgs"] = True
    g_true["ProblemType"]["GroupedGemm"] = True
    g_false = make_state()
    g_false["ProblemType"]["SupportUserArgs"] = True
    g_false["ProblemType"]["GroupedGemm"] = False
    key_true = getKeyNoInternalArgs(g_true, True)
    key_false = getKeyNoInternalArgs(g_false, True)
    assert key_true != key_false
    assert "_GG_" in key_true and "_GG_" not in key_false


def test_restores_grouped_gemm(make_state):
    """Kills mutants that restore GroupedGemm to the wrong key (mutmut_80/81/82):
    after the call the ProblemType's GroupedGemm must equal its original value."""
    s = make_state()
    s["ProblemType"]["GroupedGemm"] = True
    getKeyNoInternalArgs(s, True)
    assert s["ProblemType"]["GroupedGemm"] is True


def test_gsu_masking_thresholds_splitgsu(make_state):
    """Kills mutants that shift the GlobalSplitU mask thresholds (mutmut_38:
    ``>1``->``>2``; mutmut_41: ``==-1``->``==-2``). With splitGSU=True, GSU
    values 2, 10 and -1 all mask to "M", so their keys are identical."""
    k2 = getKeyNoInternalArgs(make_state(GlobalSplitU=2), True)
    k10 = getKeyNoInternalArgs(make_state(GlobalSplitU=10), True)
    kn1 = getKeyNoInternalArgs(make_state(GlobalSplitU=-1), True)
    assert k2 == k10 == kn1


def test_restores_state_without_pollution(make_state):
    """Kills mutants around the WorkGroupMappingXCC backup/restore and its
    line-81 masking write (mutmut_21/62/63/64/87/88/89/90): after the call the
    input state must be fully restored (WGMXCC back to its original non-1 value)
    and neither the state dict nor its ProblemType may gain extra keys."""
    s = make_state(WorkGroupMappingXCC=5)
    keys_before = set(s.keys())
    pt_keys_before = set(s["ProblemType"].keys())
    getKeyNoInternalArgs(s, True)
    assert s["WorkGroupMappingXCC"] == 5
    assert set(s.keys()) == keys_before
    assert set(s["ProblemType"].keys()) == pt_keys_before
