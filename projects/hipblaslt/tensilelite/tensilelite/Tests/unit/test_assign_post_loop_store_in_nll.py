# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The PostLoopStoreInNll eligibility gate.

``assignPostLoopStoreInNll`` decides, for every solution in the library, whether
the fused post-loop store is built. It runs unconditionally -- including for
solutions loaded from pre-tuned logic files, which predate the parameter and do
not carry it -- so a solution that is ineligible in any one respect has to be
turned back off by exactly one of its arms.

The arms are what these tests are for. Driving the gate with eligible configs
exercises every arm in its passing state and proves nothing about the
auto-disable, which is the gate's whole purpose: an arm that stopped firing
would let the store be built for a config it was never measured on, and the
only visible symptom is an SGPR renumbering in the generated kernel.

So each case below starts from one eligible baseline and changes one thing. The
baseline is asserted eligible first, which is what makes a ``False`` result
attributable to the arm under test rather than to a stale baseline.

Pure dict in, bool out: no emit, no GPU, no toolchain.
"""

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


def state(**overrides):
    """An eligible MXF4 subtile solution, with ``overrides`` applied.

    ``ProblemType`` merges rather than replaces, so a case can change one field
    of it without restating the rest.
    """
    s = {
        "ISA": (9, 5, 0),
        "UseSubtileImpl": True,
        "EnableMatrixInstruction": True,
        "PrefetchGlobalRead": 1,
        "BufferStore": True,
        "WavefrontSize": 64,
        "_GlobalAccumulation": None,
        "StoreRemapVectorWidth": 0,
        "StreamK": 0,
        "StreamKAtomic": 0,
        "MIWaveTile": [8, 8],
        "MacroTile0": 256,
        "MacroTile1": 256,
        "ProblemType": {
            "DataTypeA": DataType("float4"),
            "DataTypeB": DataType("float4"),
            "DestDataType": DataType("bfloat16"),
            "HighPrecisionAccumulate": True,
            "MXBlockA": 32,
            "MXBlockB": 32,
        },
    }
    problemType = overrides.pop("ProblemType", {})
    s.update(overrides)
    s["ProblemType"].update(problemType)
    return s


def resolve(**overrides):
    s = state(**overrides)
    Solution.assignPostLoopStoreInNll(s)
    return s["PostLoopStoreInNll"]


def test_baseline_is_eligible():
    """Every disable case below is a one-field change from this.

    If this stops holding, those cases pass for the wrong reason.
    """
    assert resolve() is True


# (id, overrides, the arm each one is meant to reach)
#
# Two arms have no case of their own because nothing can reach them alone.
# ``isFloat4`` asks for fp4 on A *or* B, while ``pairedStoreAvailable`` goes
# through ``plsinDataTypes``, which asks for A *and* B -- so any input that
# fails ``isFloat4`` has already failed the paired-store arm. And the
# ``PrefetchGlobalRead < 1`` term is repeated by the defensive re-check after
# the gate, so removing either one still disables. Both are covered here only
# in the sense that the config is disabled; see test_subsumed_arms.
_DISABLE_CASES = [
    # structuralFail: the fused store cannot be emitted at all.
    ("bf16 in",               dict(ProblemType={"DataTypeA": DataType("bfloat16"),
                                                "DataTypeB": DataType("bfloat16")})),
    ("only A is fp4",         dict(ProblemType={"DataTypeB": DataType("bfloat16")})),
    ("UseSubtileImpl off",    dict(UseSubtileImpl=False)),
    ("not gfx950",            dict(ISA=(9, 4, 2))),
    ("no matrix instruction", dict(EnableMatrixInstruction=False)),
    ("PrefetchGlobalRead 0",  dict(PrefetchGlobalRead=0)),
    ("BufferStore off",       dict(BufferStore=False)),
    # pairedStoreAvailable, which is four conditions in one arm.
    ("dest is not bf16",      dict(ProblemType={"DestDataType": DataType("float")})),
    ("no HPA",                dict(ProblemType={"HighPrecisionAccumulate": False})),
    ("wave32",                dict(WavefrontSize=32)),
    ("MultipleBuffer",        dict(_GlobalAccumulation="MultipleBuffer")),
    ("MultipleBufferSingleKernel",
                              dict(_GlobalAccumulation="MultipleBufferSingleKernel")),
    # The store mechanism itself.
    ("StoreRemapVectorWidth", dict(StoreRemapVectorWidth=1)),
    ("StreamK with atomics",  dict(StreamK=3, StreamKAtomic=1)),
    # registerFail: it would be emitted, but would not fit.
    ("spills, no large tile", dict(MIWaveTile=[10, 8])),
    ("store overflows VGPR",  dict(MIWaveTile=[4, 16])),
    ("MX skew overflows SGPR", dict(MIWaveTile=[2, 16])),
]


@pytest.mark.parametrize(
    "overrides",
    [pytest.param(o, id=d) for d, o in _DISABLE_CASES],
)
def test_arm_disables(overrides):
    assert resolve(**overrides) is False


# Cases that look like the disable cases above but must stay eligible. Without
# these, an arm that disabled everything it touched would pass the table above.
_STILL_ELIGIBLE_CASES = [
    # largeTileLend is the reason spillFree tolerates a >64 tile product: the
    # large-MT path lends K=0 A/B registers instead of holding them live.
    ("MT320x256 lends past the spill limit", dict(MIWaveTile=[10, 8], MacroTile0=320)),
    ("MT256x320 lends past the spill limit", dict(MIWaveTile=[8, 10], MacroTile1=320)),
    # The SGPR ceiling this arm protects is only reached when scales are read,
    # so the same skew without MX block scaling is fine. This is what separates
    # mxBlockScaleSgprFits from storeFitsVgpr, which [2,16] does not trip.
    ("MX skew without block scaling", dict(MIWaveTile=[2, 16],
                                           ProblemType={"MXBlockA": 0, "MXBlockB": 0})),
    # Only the atomic StreamK reduction is excluded.
    ("StreamK without atomics", dict(StreamK=3, StreamKAtomic=0)),
]


@pytest.mark.parametrize(
    "overrides",
    [pytest.param(o, id=d) for d, o in _STILL_ELIGIBLE_CASES],
)
def test_near_miss_stays_eligible(overrides):
    assert resolve(**overrides) is True


def test_subsumed_arms():
    """Pin the two arms that cannot fire on their own.

    Neither can be reached by any input, so neither is load-bearing today. They
    are asserted rather than removed because the subsumption is a property of
    their neighbours: if ``plsinDataTypes`` ever stopped requiring fp4 on
    both operands, or the re-check after the gate were dropped, these would
    become the only thing disabling a config -- and that should be a decision,
    not a surprise.
    """
    # isFloat4 is "A or B", the paired-store arm is "A and B" -- so a kernel
    # failing the first has always failed the second too.
    assert resolve(ProblemType={"DataTypeB": DataType("bfloat16")}) is False
    assert resolve(ProblemType={"DestDataType": DataType("float")}) is False

    # PGR is tested inside the gate and again straight after it, so a config
    # with no NLL is disabled whichever one runs.
    assert resolve(PrefetchGlobalRead=0) is False
    assert resolve(PostLoopStoreInNll=True, PrefetchGlobalRead=0) is False


def test_absent_key_is_defaulted_then_gated():
    """Pre-PLSIN solutions carry no key at all.

    The gate defaults it so the arms have something to evaluate, which is only
    safe because they can then turn it back off -- so check both halves.
    """
    eligible = state()
    assert "PostLoopStoreInNll" not in eligible
    Solution.assignPostLoopStoreInNll(eligible)
    assert eligible["PostLoopStoreInNll"] is True

    ineligible = state(ISA=(9, 4, 2))
    assert "PostLoopStoreInNll" not in ineligible
    Solution.assignPostLoopStoreInNll(ineligible)
    assert ineligible["PostLoopStoreInNll"] is False


@pytest.mark.parametrize(
    "overrides",
    [pytest.param(o, id=d) for d, o in [("eligible", {})] + _DISABLE_CASES],
)
def test_never_enables(overrides):
    """An explicit False is a decision taken elsewhere, and the gate only ever
    auto-disables. It must not promote one back to True -- not even for a
    config that would otherwise qualify."""
    assert resolve(PostLoopStoreInNll=False, **overrides) is False


@pytest.mark.parametrize(
    "overrides",
    [pytest.param(o, id=d) for d, o in [("eligible", {})] + _DISABLE_CASES],
)
def test_idempotent(overrides):
    """The gate runs from two call sites and may run twice on one solution."""
    s = state(**overrides)
    Solution.assignPostLoopStoreInNll(s)
    once = s["PostLoopStoreInNll"]
    Solution.assignPostLoopStoreInNll(s)
    assert s["PostLoopStoreInNll"] is once
