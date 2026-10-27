# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PGR=2 pre-loop SRD hold in ``emitSrdAdvance``.

The pre-loop issues two prefetch clusters with an SRD advance between them. When
the whole summation fits in one DepthU the pre-loop branches straight to the NLL,
which reads only cluster 1's LDS buffer -- so cluster 2 prefetches a K-block past
the end of the tensor. Those loads still have to issue, because the pre-loop's
partial ``vmcnt`` counts on them, so instead of skipping them the advance is
suppressed and they re-read cluster 1's block. ``Srd+2`` is a fixed tile-boundary
limit rather than a distance to the end of the tensor, so nothing else clips
them; without the hold they run off the end of the allocation. The failure mode
is ``hipErrorIllegalAddress`` on the MX scale tensors, whose allocations are
small enough to end on a mapping boundary.

Tested here rather than against emitted assembly because the interesting part is
a runtime decision. The emitter cannot be asked what the kernel will do, but the
select chain it produces is a decision table over two scalars, and that table can
be executed. ``test_hold_decision_table`` does exactly that: it runs the emitted
instructions against each (tail, LoopCounterL) combination and checks the
increment that reaches the SRD.

CPU-only. No GPU, no assembler, no rocisa kernel state.
"""

from types import SimpleNamespace

import pytest

from rocisa.code import Module
from rocisa.container import RegisterContainer
from rocisa.enum import RegisterType
from rocisa.instruction import SAddCU32, SAddU32, SAndB32, SCmpEQU32, SCmpLeU32, SCSelectB32
from rocisa.register import RegisterPool

from tensilelite.Components.Subtile.SubtileGREmit import emitSrdAdvance

pytestmark = pytest.mark.unit

_DEPTH_U = 256
_INC = 128


class _DataType:
    """The two ``isMxf4SubtilePath`` predicates, without pulling in DataType."""

    def __init__(self, float4=False, bfloat16=False):
        self._float4 = float4
        self._bfloat16 = bfloat16

    def isFloat4(self):
        return self._float4

    def isBFloat16(self):
        return self._bfloat16


def mxf4Kernel(depthU=_DEPTH_U, float4=True, transposeA=True):
    """A kernel dict carrying just what the hold path reads.

    ``float4`` and ``transposeA`` are the two ways to fall out of
    ``isMxf4SubtilePath`` while keeping everything else identical.
    """
    return {
        "DepthU": depthU,
        "UseSubtileImpl": True,
        "ProblemType": {
            "TransposeA": transposeA,
            "TransposeB": False,
            "DataTypeA": _DataType(float4=float4),
            "DataTypeB": _DataType(float4=float4),
            "DestDataType": _DataType(bfloat16=True),
        },
    }


def writer():
    w = SimpleNamespace()
    w.sgprPool = RegisterPool(32, RegisterType.Sgpr, False)
    w.states = SimpleNamespace(unrollIdx=0)
    return w


def advance(w=None, kernel=None, holdOnLastIter=True, tc="A", inc=_INC):
    if w is None:
        w = writer()
    if kernel is None:
        kernel = mxf4Kernel()
    module = Module("srdAdvance")
    emitSrdAdvance(module, tc, inc, w, kernel, holdOnLastIter)
    return list(module.items())


def names(items):
    return [type(i).__name__ for i in items]


# A plain advance: add the increment to the 64-bit SRD base and carry.
_PLAIN = ["SAddU32", "SAddCU32"]


def test_plain_advance_when_the_hold_is_not_requested():
    items = advance(holdOnLastIter=False)
    assert names(items) == _PLAIN
    assert items[0].srcs[1] == _INC, "the increment should reach the SRD as an immediate"


@pytest.mark.parametrize(
    "kernel, why",
    [
        (mxf4Kernel(float4=False), "fp4 inputs"),
        (mxf4Kernel(transposeA=False), "the TN layout"),
    ],
)
def test_plain_advance_off_the_mxf4_subtile_path(kernel, why):
    """A kernel outside the gate advances exactly as it does on develop.

    The hold is requested here -- only the gate keeps it from being built -- so
    this is the assertion that the scope gate, not the call site, is what holds
    the rest of the library at develop's codegen.
    """
    items = advance(kernel=kernel)
    assert names(items) == _PLAIN, f"the hold was built for a kernel without {why}"
    assert items[0].srcs[1] == _INC


def test_hold_makes_the_increment_a_runtime_value():
    items = advance()
    assert names(items) == [
        "SAndB32",      # tail = SizesSum % DepthU
        "SCmpEQU32",    # no tail?
        "SCSelectB32",  # threshold: 1 without a tail, 0 with one
        "SCmpLeU32",    # LoopCounterL <= threshold?
        "SCSelectB32",  # 0 to hold, inc to advance
        "SAddU32",
        "SAddCU32",
    ]
    mask = items[0]
    assert str(mask.srcs[0]) == "s[sgprSizesSum+0]"
    assert mask.srcs[1] == _DEPTH_U - 1, (
        "the tail test must mask with DepthU-1 so it really is SizesSum % DepthU"
    )
    assert str(items[3].srcs[0]) == "s[sgprLoopCounterL]"
    assert isinstance(items[5].srcs[1], RegisterContainer), (
        "the SRD increment must come from the select, not stay an immediate"
    )


def run(items, tail, loopCounter):
    """Execute the emitted select chain and return what reaches the SRD.

    The chain is straight-line scalar ALU over SCC, so a six-line interpreter
    covers it exactly. ``tail`` and ``loopCounter`` seed the two values the
    kernel reads at runtime.
    """
    regs = {
        "s[sgprSizesSum+0]": tail,  # only the low DepthU-1 bits are ever read
        "s[sgprLoopCounterL]": loopCounter,
    }
    scc = None

    def value(src):
        return regs[str(src)] if isinstance(src, RegisterContainer) else src

    for item in items:
        if isinstance(item, SAndB32):
            regs[str(item.dst)] = value(item.srcs[0]) & value(item.srcs[1])
        elif isinstance(item, SCmpEQU32):
            scc = value(item.srcs[0]) == value(item.srcs[1])
        elif isinstance(item, SCmpLeU32):
            scc = value(item.srcs[0]) <= value(item.srcs[1])
        elif isinstance(item, SCSelectB32):
            regs[str(item.dst)] = value(item.srcs[0] if scc else item.srcs[1])
        elif isinstance(item, SAddU32):
            return value(item.srcs[1])
        elif isinstance(item, SAddCU32):
            continue
        else:
            raise AssertionError(f"unmodelled instruction in the hold chain: {item}")
    raise AssertionError("the chain never reached the SRD add")


# (tail, LoopCounterL, hold). LoopCounterL counts the unrolled mainloop
# iterations and tail is SizesSum % DepthU, so the first four rows are the
# summations that end at or before the pre-loop's second prefetch cluster.
_DECISION_TABLE = [
    (0, 0, True),    # K < DepthU, no tail: the NLL is the whole kernel
    (0, 1, True),    # K == DepthU exactly: the NLL is terminal
    (0, 2, False),   # two full K-blocks, cluster 2 reads the second
    (64, 0, True),   # K < DepthU and all of it is tail
    (64, 1, False),  # one unrolled iteration then a tail -- see below
    (64, 2, False),
]


@pytest.mark.parametrize("tail, loopCounter, hold", _DECISION_TABLE)
def test_hold_decision_table(tail, loopCounter, hold):
    """The emitted chain holds the SRD on exactly the summations that need it.

    Row ``(64, 1)`` is the one worth naming: a K-tail following a single
    unrolled iteration. Cluster 2's K-block is the tail's own block -- it exists,
    partially, and the tail loop reads it through this same SRD -- so holding
    there would make the tail re-MAC K-block 0 and return wrong results rather
    than fault. A hold written as the simpler ``LoopCounterL <= 1`` passes every
    other row here and fails this one.
    """
    applied = run(advance(), tail, loopCounter)
    assert applied == (0 if hold else _INC)


def test_hold_requires_a_power_of_two_depth_u():
    """The mask stands in for a modulo, which only works on a power of two."""
    with pytest.raises(AssertionError, match="power of two"):
        advance(kernel=mxf4Kernel(depthU=192))


def test_the_scratch_sgpr_is_returned_to_the_pool():
    """Two advances share one temp, so the first one gave it back.

    The pre-loop emits four of these, one per tensor; a leak would cost four
    SGPRs on every MXF4 kernel and can push a tuned solution past its budget.
    """
    w = writer()
    first = str(advance(w=w)[0].dst)
    second = str(advance(w=w)[0].dst)
    assert first == second, f"the temp leaked: {first} then {second}"
