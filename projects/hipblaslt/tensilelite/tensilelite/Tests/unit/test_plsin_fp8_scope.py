# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The scope gates that carry the PLSIN fused store to MXFP8.

``isMxf4SubtilePath`` stays fp4-only on purpose: it carries the whole MXF4
subtile bundle -- swizzled global read, deferred LRA, StreamK, the staged store
and the block schedule -- across some thirty call sites, and widening it to
reach fp8 would move all of them at once. The fused store travels on its own
gates instead, so what these tests pin down is mostly a containment: fp8 gets
PLSIN and nothing else, and no fp8 kernel that did not ask for the fused store
changes shape.

These are pure-predicate tests: no emit, no GPU, no toolchain.
"""

import itertools

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.Common.Utilities import (
    isMxf4SubtilePath,
    isPlsinSubtilePath,
    plsinBlockSchedTile,
    plsinDataTypes,
    plsinStoreDataTypes,
    plsinStoreScope,
)

pytestmark = pytest.mark.unit

FP4 = "float4"
FP8 = "float8"
FP8N = "float8_fnuz"
BF16 = "bfloat16"
FP32 = "float"
FP16 = "half"


def kern(dtA=FP8, dtB=FP8, dtD=BF16, subtile=True, mt0=256, mt1=256, plsin=True,
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


# (description, kernel, store types in scope?)
_TYPE_CASES = [
    ("fp8 in, bf16 out", kern(), True),
    ("fp4 in, bf16 out", kern(dtA=FP4, dtB=FP4), True),
    ("fp4 A, fp8 B", kern(dtA=FP4, dtB=FP8), True),
    ("fp8 A, fp4 B", kern(dtA=FP8, dtB=FP4), True),
    ("bf16 in", kern(dtA=BF16, dtB=BF16), False),
    ("only A is fp8", kern(dtB=BF16), False),
    ("only B is fp8", kern(dtA=BF16), False),
    ("fp32 out", kern(dtD=FP32), False),
    ("fp16 out", kern(dtD=FP16), False),
    ("fp8 in, fp8 out", kern(dtD=FP8), False),
]


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _TYPE_CASES],
)
def test_store_data_types(kernel, expected):
    assert plsinStoreDataTypes(kernel) is expected


def test_store_types_are_plsin_types_widened_by_exactly_fp8():
    """The one claim the widening rests on, stated as a containment.

    ``plsinStoreDataTypes`` exists to be ``plsinDataTypes`` plus fp8 operands,
    nothing else. Were it to drift wider -- picking up an output type, say --
    the fused store would start being built for kernels whose register budget
    was never measured, and the gates below would carry it there silently.
    """
    for dtA, dtB, dtD in itertools.product(
            (FP4, FP8, BF16), (FP4, FP8, BF16), (BF16, FP32, FP16)):
        kernel = kern(dtA=dtA, dtB=dtB, dtD=dtD)
        widened = plsinStoreDataTypes(kernel)
        if plsinDataTypes(kernel):
            assert widened, f"narrowed against plsinDataTypes: {dtA} {dtB} {dtD}"
        elif widened:
            assert FP8 in (dtA, dtB), \
                f"widened by something other than fp8: {dtA} {dtB} {dtD}"


def test_fnuz_fp8_is_out_of_scope():
    """Only the OCP fp8 encoding is in scope, which is the gfx950 one.

    ``isFloat8`` is False for fnuz, so the gate excludes it without a test of
    its own. That is the intended answer rather than an oversight: the MX scale
    path these kernels take is defined on the OCP encoding, and nothing fnuz
    was built or measured. Pinned here because the exclusion is implicit, and a
    future ``isFloat8`` that folded the two encodings together would open the
    gate to fnuz with no other test noticing.
    """
    assert plsinStoreDataTypes(kern(dtA=FP8N, dtB=FP8N)) is False


@pytest.mark.parametrize(
    "kernel,expected",
    [pytest.param(k, e, id=d) for d, k, e in _TYPE_CASES],
)
def test_path_gate_follows_the_types_when_subtile_and_tn(kernel, expected):
    assert isPlsinSubtilePath(kernel) is expected


def test_path_gate_requires_use_subtile_impl():
    assert isPlsinSubtilePath(kern(subtile=False)) is False


@pytest.mark.parametrize("transA,transB,expected", [
    (True, False, True),    # TN, the layout the fused store was measured on
    (False, False, False),  # NN
    (False, True, False),   # NT
    (True, True, False),    # TT
])
def test_path_gate_is_restricted_to_tn(transA, transB, expected):
    """Inherited from the MXF4 gate, and for the same reason.

    The non-TN layouts reach the subtile path through a strided-K global-read
    pointer update, a shape the fused store does not model. ``MXFP8`` makes
    that reachable in a way fp4 never was -- the shipped F8 logic is all
    ``Cijk_Alik_Bljk_*`` today, but fp8 with TLU=1 is accepted by
    ``assignDerivedParameters``, so the restriction has to live in the gate.
    """
    assert isPlsinSubtilePath(kern(transA=transA, transB=transB)) is expected


def test_path_gate_does_not_read_post_loop_store_in_nll():
    """It is asked during derivation, before that key is resolved.

    ``Naming`` uses it to decide whether ``PostLoopStoreInNll`` belongs in the
    kernel name at all, so a gate that read the key would be circular.
    """
    assert isPlsinSubtilePath(kern(plsin=False)) is True
    withoutKey = kern()
    del withoutKey["PostLoopStoreInNll"]
    assert isPlsinSubtilePath(withoutKey) is True


def test_fp8_never_enters_the_mxf4_path():
    """The containment the split exists to preserve.

    Every one of the MXF4 bundle's call sites keys on ``isMxf4SubtilePath``. If
    fp8 answered True there it would acquire the swizzled global read, the
    deferred LRA and the StreamK paths along with the fused store, none of
    which it was built for.
    """
    for dtA, dtB in ((FP8, FP8), (FP4, FP8), (FP8, FP4)):
        assert isMxf4SubtilePath(kern(dtA=dtA, dtB=dtB)) is False


@pytest.mark.parametrize("mt0,mt1", [(256, 256), (128, 128)])
def test_block_scheduling_stays_fp4_only(mt0, mt1):
    """Load-bearing, and the reason the gates are split rather than widened.

    The block-scheduled tail does not model dual-period scale reuse. An MXFP8
    MT256x256 DepthU-128 kernel fails to schedule with it on: a scale global
    read at n+2 finds no local read to collide with, because the scale is
    loaded once per two unroll iterations rather than every one.

    ``plsinBlockSchedTile`` is built on ``isMxf4SubtilePath``, so it answers
    False for fp8 without a type test of its own -- which is exactly the
    property that would be lost if the fp4 gate were widened instead.
    """
    assert plsinBlockSchedTile(kern(mt0=mt0, mt1=mt1)) is False
    assert plsinBlockSchedTile(kern(dtA=FP4, dtB=FP4, mt0=mt0, mt1=mt1)) \
        is (mt0 == 256 and mt1 == 256)


class TestStoreScope:
    """Where the store emitters may act. Deliberately asymmetric in the types.

    The emitters are shared with the plain subtile store, so the two inputs
    cannot be scoped the same way.
    """

    def test_mxf4_scope_is_unconditional(self):
        """Not one fp4 kernel changes shape, PLSIN on or off.

        fp4 has always taken these emitters through the plain subtile store.
        """
        assert plsinStoreScope(kern(dtA=FP4, dtB=FP4, plsin=True)) is True
        assert plsinStoreScope(kern(dtA=FP4, dtB=FP4, plsin=False)) is True

    def test_fp8_joins_only_where_plsin_is_built(self):
        """A non-PLSIN fp8 kernel keeps develop's store path untouched.

        Scoping on the type alone would move every 16-bit subtile store that
        happens to be fp8 and never asked for a fused store.
        """
        assert plsinStoreScope(kern(plsin=True)) is True
        assert plsinStoreScope(kern(plsin=False)) is False

    def test_absent_key_reads_as_off_for_fp8(self):
        withoutKey = kern()
        del withoutKey["PostLoopStoreInNll"]
        assert plsinStoreScope(withoutKey) is False

    def test_out_of_scope_types_are_out_of_scope_however_plsin_is_set(self):
        for plsin in (True, False):
            assert plsinStoreScope(kern(dtA=BF16, dtB=BF16, plsin=plsin)) is False
            assert plsinStoreScope(kern(dtD=FP32, plsin=plsin)) is False

    def test_scope_contains_the_mxf4_path(self):
        """No kernel loses an emitter it had before this change.

        The sweep is the point: the asymmetry between the two disjuncts is easy
        to restate wrongly, and the failure mode is an fp4 kernel quietly
        dropping out of a store path it has always taken.
        """
        tiles = (128, 256)
        for dtA, dtB, dtD, subtile, mt0, mt1, plsin, (tA, tB) in itertools.product(
                (FP4, FP8, BF16), (FP4, FP8, BF16), (BF16, FP32), (True, False),
                tiles, tiles, (True, False),
                ((True, False), (False, False), (False, True), (True, True))):
            kernel = kern(dtA=dtA, dtB=dtB, dtD=dtD, subtile=subtile,
                          mt0=mt0, mt1=mt1, plsin=plsin, transA=tA, transB=tB)
            if isMxf4SubtilePath(kernel):
                assert plsinStoreScope(kernel), \
                    f"fp4 kernel dropped out of the store scope: {kernel}"
            elif plsinStoreScope(kernel):
                assert isPlsinSubtilePath(kernel) and plsin, \
                    f"store scope opened outside PLSIN: {kernel}"


@pytest.mark.parametrize("kernel", [
    pytest.param({}, id="empty"),
    pytest.param({"UseSubtileImpl": True}, id="no ProblemType"),
    pytest.param({"UseSubtileImpl": True, "ProblemType": {}}, id="empty ProblemType"),
    pytest.param(
        {"UseSubtileImpl": True, "PostLoopStoreInNll": True,
         "ProblemType": {"DataTypeA": 21, "DataTypeB": 21, "DestDataType": 7}},
        id="raw type values",
    ),
])
def test_partially_built_state_is_out_of_scope(kernel):
    """All three gates are asked before the solution dict is finished.

    ``Naming`` calls ``plsinStoreDataTypes`` directly rather than through a
    guarded gate, and it runs while the data types are still raw values, so the
    tolerance has to be in the predicate itself. "Out of scope" is the
    conservative answer: it is what keeps ``PostLoopStoreInNll`` out of the
    name hash of kernels that can never set it.
    """
    assert plsinStoreDataTypes(kernel) is False
    assert isPlsinSubtilePath(kernel) is False
    assert plsinStoreScope(kernel) is False
