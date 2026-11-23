# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scope gate for the MXF4 subtile optimizations.

``isMxf4SubtilePath`` and the tile predicates layered on it decide, for every
kernel in the library, whether the fused post-loop store and the partitioned
block schedule are built. Getting the gate wrong in the permissive direction
silently reshapes the epilogue of kernels that were never measured with it, and
that failure mode is invisible in a test that only drives MXF4 configs.

These are pure-predicate tests: no emit, no GPU, no toolchain.
"""

import itertools
from types import SimpleNamespace

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.KernelWriterAssembly import KernelWriterAssembly
from tensilelite.Common.Utilities import (
    isMxf4SubtilePath,
    plsinBlockSchedTile,
    plsinDataTypes,
)

pytestmark = pytest.mark.unit

FP4 = "float4"
BF16 = "bfloat16"
FP32 = "float"
FP16 = "half"


def kern(dtA=FP4, dtB=FP4, dtD=BF16, subtile=True, mt0=256, mt1=256, plsin=True,
         transA=True, transB=False):
    return {
        "UseSubtileImpl": subtile,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "PostLoopStoreInNll": plsin,
        "ProblemType": {
            "DataTypeA": DataType(dtA),
            "DataTypeB": DataType(dtB),
            "DestDataType": DataType(dtD),
            "TransposeA": transA,
            "TransposeB": transB,
        },
    }


# (description, kernel, in scope?)
_SCOPE_CASES = [
    ("mxf4 in, bf16 out, subtile", kern(), True),
    ("fp32 out", kern(dtD=FP32), False),
    ("fp16 out", kern(dtD=FP16), False),
    ("bf16 in", kern(dtA=BF16, dtB=BF16), False),
    ("only A is fp4", kern(dtB=BF16), False),
    ("only B is fp4", kern(dtA=BF16), False),
    ("UseSubtileImpl off", kern(subtile=False), False),
]


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _SCOPE_CASES],
)
def test_is_mxf4_subtile_path(kernel, expected):
    assert isMxf4SubtilePath(kernel) is expected


@pytest.mark.parametrize(
    "transA,transB,expected",
    [
        (True, False, True),    # TN, the layout everything here was measured on
        (False, False, False),  # NN
        (False, True, False),   # NT
        (True, True, False),    # TT
    ],
    ids=["TN", "NN", "NT", "TT"],
)
def test_gate_is_restricted_to_tn(transA, transB, expected):
    """The non-TN layouts reach the subtile path but not these optimizations.

    They use a strided-K global-read pointer update, so the pre-loop SRD hold and
    the fused store reason about a shape those kernels do not have. Opening the
    gate to them miscompares rather than merely running slower.
    """
    assert isMxf4SubtilePath(kern(transA=transA, transB=transB)) is expected


def test_use_subtile_impl_alone_does_not_open_the_gate():
    """The distinction the gate exists to draw.

    ``UseSubtileImpl`` is set on gfx1250 as well as gfx950, and gfx950 MX merely
    requires it rather than being the only thing that sets it, so gating on it
    would pull in every subtile kernel in the library.
    """
    bf16Subtile = kern(dtA=BF16, dtB=BF16, dtD=BF16)
    assert bf16Subtile["UseSubtileImpl"] is True
    assert not isMxf4SubtilePath(bf16Subtile)


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _SCOPE_CASES],
)
def test_plsin_subtile_types_ignores_use_subtile_impl(kernel, expected):
    """``plsinDataTypes`` is the type half of the gate on its own.

    It must answer for the operand types alone, so that the only case where it
    disagrees with the full gate is the one where UseSubtileImpl is off.
    """
    typesOnly = plsinDataTypes(kernel)
    if kernel["UseSubtileImpl"]:
        assert typesOnly is expected
    else:
        assert typesOnly is True


@pytest.mark.parametrize("mt0,mt1,blockSched", [
    (256, 256, True),      # the one geometry block scheduling is measured on
    (128, 128, False),     # PLSIN-eligible, deliberately not block-scheduled
    (192, 256, False),     # the equality excludes it; a <= bound would not
    (256, 192, False),
    (320, 256, False),     # >256: lends its K=0 operand registers to the store
    (256, 320, False),
])
def test_tile_scope(mt0, mt1, blockSched):
    """MT256x256 is an equality, not a bound.

    MT192x256 would pass any ``<= 256`` form and be dragged in untested, and
    MT>256x256 could not take an early store at all -- it lends its K=0 operand
    registers to the store, so one moved earlier would read them mid-flight.
    """
    assert plsinBlockSchedTile(kern(mt0=mt0, mt1=mt1)) is blockSched


def test_block_scheduling_requires_plsin():
    """Block scheduling is the staged *fused* store, so it needs the fused store.

    The tile and the operand types can both be in scope while PostLoopStoreInNll
    has been turned off for an unrelated reason -- StreamK atomic, a non-zero
    StoreRemapVectorWidth, wave32, MultipleBuffer accumulation. Building the
    partitioned schedule there would emit the NGLL/NLL arm split for a store
    that is never fused. Early-store scope is tile-and-type only, so it still
    answers True.
    """
    noPlsin = kern(plsin=False)
    assert isMxf4SubtilePath(noPlsin) is True
    assert plsinBlockSchedTile(noPlsin) is False


@pytest.mark.parametrize("layout,transA,transB", [
    ("TN", True,  False),   # the only layout any of this was built on
    ("NN", False, False),
    ("NT", False, True),
    ("TT", True,  True),
])
def test_tile_scope_requires_tn(layout, transA, transB):
    """The tile predicates carry the TN restriction, not just the emit sites.

    ``assignPostLoopStoreInNll`` resolves ``PostLoopStoreInNll`` from the data
    types, the ISA and the register budget -- it never looks at the layout. So
    the layout restriction has to arrive through the tile predicates, or a
    non-TN fp4/bf16 subtile kernel resolves PLSIN True, builds the block
    schedule, the four-deep tail and absolute row addressing, and then finds
    every store-side site keyed on ``isMxf4SubtilePath`` switched off. The
    kernel name would not even record it: ``Naming`` drops
    ``PostLoopStoreInNll`` from the required set when the path gate is False.

    Nothing in the shipped library is non-TN fp4 -- every F4 solution is
    ``Cijk_Alik_Bljk_*`` -- but ``Solution.assignDerivedParameters`` accepts
    fp4 with TLU=1, so this is one tuning config away.
    """
    kernel = kern(transA=transA, transB=transB)
    inScope = layout == "TN"
    assert isMxf4SubtilePath(kernel) is inScope
    assert plsinBlockSchedTile(kernel) is inScope


def test_tile_scope_requires_the_type_scope():
    """No tile is in scope once the operand types are out of it."""
    outOfScope = kern(dtD=FP32)
    assert isMxf4SubtilePath(outOfScope) is False
    assert plsinBlockSchedTile(outOfScope) is False


def test_block_scheduling_stays_inside_the_mxf4_path():
    """Block scheduling is a strict subset of the scope gate.

    The block-scheduled tail, its staged store and the finer store grid are all
    reached through ``plsinBlockSchedTile``, never through
    ``isMxf4SubtilePath`` directly, so no call site checks the containment. The
    barrier and wait-accounting changes are scoped the other way, on
    ``isMxf4SubtilePath`` itself, and the two only agree as long as the tile
    predicates stay inside it. Loosening the tile predicate -- restating the
    gate's terms instead of calling it, or widening the equality to a bound --
    would start building the block schedule for kernels none of this was
    measured on, and the first sign of it would be a library diff rather than a
    failing test.

    The product sweeps the layout too. It did not, and that is exactly how the
    tile predicate came to be missing the TN restriction the gate carries:
    every case here was TN, so the containment held vacuously on the one axis
    where it did not hold at all.
    """
    tiles = (64, 128, 192, 256, 320)
    for dtA, dtB, dtD, subtile, mt0, mt1, plsin, (tA, tB) in itertools.product(
            (FP4, BF16), (FP4, BF16), (BF16, FP32, FP16), (True, False),
            tiles, tiles, (True, False),
            ((True, False), (False, False), (False, True), (True, True))):
        kernel = kern(dtA=dtA, dtB=dtB, dtD=dtD, subtile=subtile,
                      mt0=mt0, mt1=mt1, plsin=plsin, transA=tA, transB=tB)
        if plsinBlockSchedTile(kernel):
            assert isMxf4SubtilePath(kernel), \
                f"block scheduling escaped the MXF4 path: {kernel}"


@pytest.mark.parametrize("kernel", [
    pytest.param({}, id="empty"),
    pytest.param({"UseSubtileImpl": True}, id="no ProblemType"),
    pytest.param({"UseSubtileImpl": True, "ProblemType": {}}, id="empty ProblemType"),
    pytest.param(
        {"UseSubtileImpl": True,
         "ProblemType": {"DataTypeA": 21, "DataTypeB": 21, "DestDataType": 7}},
        id="raw type values",
    ),
])
def test_partially_built_state_is_not_mxf4(kernel):
    """Naming runs before the data types are DataType objects.

    ``isMxf4SubtilePath`` has to answer there too, and "not MXF4" is the
    conservative answer: it is what keeps ``PostLoopStoreInNll`` out of the name
    hash of kernels that can never set it.
    """
    assert isMxf4SubtilePath(kernel) is False


class TestStagedArmNoActivation:
    """When the staged fused arm may carry only the no-activation store body.

    The arm emits one store body. If that body is the no-activation one and the
    kernel actually wants an activation, the activation is skipped outright --
    there is no second body and no runtime path back to one. So the predicate
    has to be false for every kernel that applies an activation, whether the
    choice is made at runtime or baked in at compile time.

    The compile-time case is the one with no safety net: a runtime-selectable
    activation at least has an ``ActivationType`` SGPR that something could
    branch on, while a kernel built for ``relu`` has nothing to test.
    """

    @staticmethod
    def _call(kernel, sgprs=(), plsin=True):
        """Invoke the predicate against a stub; it reads only these two fields."""
        stub = SimpleNamespace(states=SimpleNamespace(postLoopStoreInNll=plsin),
                               sgprs=dict.fromkeys(sgprs, 0))
        return KernelWriterAssembly._plsinStagedArmNoActivation(stub, kernel)

    def test_no_activation_folds(self):
        k = kern()
        k["ProblemType"]["ActivationType"] = "none"
        k["ActivationFused"] = True
        assert self._call(k) is True

    def test_unfused_activation_folds(self):
        """ActivationFused False means this kernel applies none regardless."""
        k = kern()
        k["ProblemType"]["ActivationType"] = "relu"
        k["ActivationFused"] = False
        assert self._call(k) is True

    def test_fixed_nonnone_activation_does_not_fold(self):
        """The regression: relu baked in at compile time has no SGPR to test.

        Folding here emits the no-activation body and drops the relu with no
        runtime fallback -- wrong results, not a slow path.
        """
        k = kern()
        k["ProblemType"]["ActivationType"] = "relu"
        k["ActivationFused"] = True
        assert self._call(k) is False

    def test_runtime_activation_does_not_fold(self):
        """'all' carries an ActivationType SGPR, so the arm keeps every body.

        The fused/not-fused decision is a pre-loop flag and ActivationType is
        not defined that early, so it cannot be folded into that flag either.
        """
        k = kern()
        k["ProblemType"]["ActivationType"] = "all"
        k["ActivationFused"] = True
        assert self._call(k, sgprs=("ActivationType",)) is False

    def test_requires_staging_eligibility(self):
        """Out of scope for staging, so the question never arises."""
        k = kern(mt0=128, mt1=128)
        k["ProblemType"]["ActivationType"] = "none"
        k["ActivationFused"] = True
        assert self._call(k) is False
        assert self._call(kern_none_act(), plsin=False) is False


def kern_none_act():
    k = kern()
    k["ProblemType"]["ActivationType"] = "none"
    k["ActivationFused"] = True
    return k
