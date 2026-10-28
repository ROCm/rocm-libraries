# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from rocisa.asmpass import rocIsaPass, rocIsaPassOption
from rocisa.code import KernelBody, Label, Module, TextBlock
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    SAddU32,
    SBranch,
    SMovB32,
    SMovB64,
    SNop,
    VMovB32,
    _SWaitCnt,
)

# Only already-lowered instructions here: the pass lowers composites such as
# SWaitCnt for whichever ISA is current in rocIsa's global state (gfx12 turns it
# into s_wait_dscnt, which the pass doesn't treat as a reset), and that state is
# whatever an earlier test in the process left behind.


def _run(*items, totalSgprs=16, totalVgprs=8):
    body = Module("body")
    for item in items:
        body.add(item)
    kernel = KernelBody("kernel")
    kernel.addBody(body)
    kernel.totalSgprs = totalSgprs
    kernel.totalVgprs = totalVgprs
    options = rocIsaPassOption()
    options.insertDelayAlu = False
    options.removeDupFunc = False
    options.removeDupAssign = True
    options.getCycles = False
    result = rocIsaPass(kernel, options)
    return body, result


def _names(module):
    return [type(item).__name__ for item in module.items()]


def test_repeated_move_is_removed():
    first = SMovB32(sgpr(4), 7)
    body, _ = _run(first, SMovB32(sgpr(4), 7))
    items = body.items()
    assert len(items) == 1 and items[0] is first


def test_move_of_a_different_value_is_kept():
    body, _ = _run(SMovB32(sgpr(4), 7), SMovB32(sgpr(4), 8))
    assert _names(body) == ["SMovB32", "SMovB32"]


@pytest.mark.parametrize(
    "barrier",
    [lambda: Label("L", ""), lambda: _SWaitCnt(lgkmcnt=0), lambda: SBranch("L")],
    ids=["label", "waitcnt", "branch"],
)
def test_resetting_barrier_keeps_the_repeated_move(barrier):
    body, _ = _run(SMovB32(sgpr(4), 7), barrier(), SMovB32(sgpr(4), 7))
    assert len(body.items()) == 3


def test_non_resetting_barrier_still_removes_the_repeated_move():
    body, _ = _run(SMovB32(sgpr(4), 7), SNop(0), SMovB32(sgpr(4), 7))
    assert _names(body) == ["SMovB32", "SNop"]


def test_one_resetting_barrier_among_several_keeps_the_repeated_move():
    body, _ = _run(
        SMovB32(sgpr(4), 7), SNop(0), _SWaitCnt(lgkmcnt=0), SNop(0), SMovB32(sgpr(4), 7)
    )
    assert len(body.items()) == 5


def test_several_non_resetting_barriers_still_remove_the_repeated_move():
    body, _ = _run(SMovB32(sgpr(4), 7), SNop(0), SNop(0), SMovB32(sgpr(4), 7))
    assert _names(body) == ["SMovB32", "SNop", "SNop"]


def test_reset_applies_to_every_tracked_sgpr():
    body, _ = _run(
        SMovB32(sgpr(4), 7),
        SMovB32(sgpr(5), 1),
        Label("L", ""),
        SMovB32(sgpr(5), 1),
        SMovB32(sgpr(4), 7),
    )
    assert len(body.items()) == 5


def test_write_to_the_sgpr_keeps_the_repeated_move():
    body, _ = _run(
        SMovB32(sgpr(4), 7), SAddU32(sgpr(4), sgpr(4), 1), SMovB32(sgpr(4), 7)
    )
    assert len(body.items()) == 3


def test_write_to_another_sgpr_still_removes_the_repeated_move():
    body, _ = _run(
        SMovB32(sgpr(4), 7), SAddU32(sgpr(5), sgpr(5), 1), SMovB32(sgpr(4), 7)
    )
    assert _names(body) == ["SMovB32", "SAddU32"]


def test_write_to_a_range_covering_the_sgpr_keeps_the_repeated_move():
    body, _ = _run(SMovB32(sgpr(4), 7), SMovB64(sgpr(4, 2), 0), SMovB32(sgpr(4), 7))
    assert len(body.items()) == 3


def test_repeated_move_with_a_comment_becomes_a_comment():
    body, _ = _run(SMovB32(sgpr(4), 7), SMovB32(sgpr(4), 7, "again"))
    items = body.items()
    assert _names(body) == ["SMovB32", "TextBlock"]
    assert isinstance(items[1], TextBlock)
    assert str(items[1]).strip() == "// again (dup assign opt.)"


def test_repeated_move_in_a_nested_module_is_removed():
    inner = Module("inner")
    inner.add(SMovB32(sgpr(4), 7))
    _run(SMovB32(sgpr(4), 7), inner)
    assert inner.items() == []


def test_no_opt_module_is_left_alone():
    noOpt = Module("noOpt")
    noOpt.setNoOpt(True)
    noOpt.add(SMovB32(sgpr(4), 7))
    _run(SMovB32(sgpr(4), 7), noOpt)
    assert _names(noOpt) == ["SMovB32"]


def test_no_opt_module_resets_tracking():
    noOpt = Module("noOpt")
    noOpt.setNoOpt(True)
    noOpt.add(SMovB32(sgpr(4), 7))
    body, _ = _run(noOpt, SMovB32(sgpr(4), 7))
    assert _names(body) == ["Module", "SMovB32"]


@pytest.mark.parametrize(
    "item",
    [lambda: SMovB32(sgpr(16), 0), lambda: VMovB32(vgpr(8), 0)],
    ids=["sgpr", "vgpr"],
)
def test_out_of_range_register_is_rejected(item):
    with pytest.raises(RuntimeError, match="GPR index out of range"):
        _run(item())


def test_max_vgpr_is_one_past_the_highest_vgpr_used():
    _, result = _run(VMovB32(vgpr(5), 0), VMovB32(vgpr(2, 2), 0))
    assert result.maxVgpr == 6
