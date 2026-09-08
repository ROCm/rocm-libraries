# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the count-preserving per-XCD StreamK work-queue index.

These tests pin the per-XCD work-queue self-reset codegen fix: the SK4
(``StreamKDynamic``) and SK5 (``StreamKHybrid``) dynamic work-queue per-XCD
atomic counters self-reset to 0 each launch only if every queue receives exactly
``tiles_q + W_q`` increments, which requires the value feeding ``% numQueues`` to
densely cover ``[0, skGrid)``.  The queue index therefore must come from the RAW
pre-remap launch WG rank (== physical XCD rank), NOT from the remapped
``StreamKIdx`` (whose ``% numQueues`` is not count-preserving when the grid does
not block evenly).

Zero-SGPR carrier reuse: the raw rank is snapshotted into the ALREADY-allocated
persistent ``StreamKTileIdx`` slot -- provably dead in the [prologue, queue-read)
window -- instead of a dedicated ``StreamKQueue`` SGPR, which overflowed the SGPR
file on tuned high-register SKXCC kernels.  The fix covers TWO disjoint remap
regimes:
  * WorkGroupMappingXCC == -1 (dynamic auto-WGM), and
  * StreamKXCCMapping != 0 with WorkGroupMappingXCC > 1 (SKXCC).

Like the sibling StreamK codegen tests, they import rocisa instructions and
inspect emitted modules rather than matching source text, and reason about the
*real* KernelWriter / KernelWriterAssembly source via the AST so the ordering and
gating assertions track the actual code.

Invariants pinned (see per-test notes):
  * Both dynamic auto-WGM and SKXCC (WGMXCC>1) queue indices are
    ``StreamKTileIdx & (numQueues-1)`` -- a single mask of the raw-rank carrier,
    never the post-remap StreamKIdx shifts.
  * Both SK4 and SK5 route their queue index through the shared ``_emitQueueIndex``.
  * The raw-rank snapshot ``s_mov_b32 StreamKTileIdx, WorkGroup0`` is emitted
    BEFORE the wgmXCC workgroup remap.
  * The reused carrier costs ZERO additional persistent SGPRs: ``StreamKQueue``
    is never declared, and the snapshot targets the already-allocated
    ``StreamKTileIdx`` slot.
"""

import ast
import inspect
import textwrap
import types

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module
from rocisa.instruction import (
    SAndB32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SMovB32,
    SSubU32,
)

from Tensile.KernelWriter import KernelWriter
from Tensile.Components.StreamK import (
    StreamK,
    StreamKDynamic,
    StreamKHybrid,
    streamKVariantClass,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes: just enough of a "writer" for the standalone helper methods.
#
# _emitQueueIndex / usesRawQueueRank only touch writer.sgprPool (checkOut) and
# writer.states.archCaps, so a tiny monotonic pool plus a SimpleNamespace of the
# per-arch caps is all that is required -- no KernelWriter, no GPU.
# ---------------------------------------------------------------------------
class _FakeSgprPool:
    def __init__(self, start: int = 100):
        self._next = start

    def checkOut(self, n: int, name: str = "", *args, **kwargs) -> int:
        reg = self._next
        self._next += n
        return reg

    def checkIn(self, *args, **kwargs):
        return None


class _FakeWriter:
    """Minimal writer: gfx942/gfx950-like caps, tunable for scoping tests."""

    def __init__(self, numXCD: int = 8, cacheLineBytes: int = 128,
                 workGroupIdFromTTM: bool = False):
        self.sgprPool = _FakeSgprPool()
        self.states = types.SimpleNamespace(
            archCaps={
                "NumXCD": numXCD,
                "CacheLineBytes": cacheLineBytes,
                "WorkGroupIdFromTTM": workGroupIdFromTTM,
            })


# gfx942/gfx950 both map to 8 queues (power of two) in the per-arch lookup.
def _kernel(streamk: int = 4, wgmXCC: int = -1, skxcc: int = 0,
            isa=(9, 4, 0)) -> dict:
    return {"ISA": isa, "StreamK": streamk, "WorkGroupMappingXCC": wgmXCC,
            "StreamKXCCMapping": skxcc}


def _flat(module: Module) -> list:
    return list(module.flatitems())


def _param_texts(inst) -> list:
    return [str(p) for p in inst.getParams()]


def _refs_sgpr(inst, name: str) -> bool:
    """True if *inst* references the named SGPR (e.g. ``s[sgprStreamKTileIdx]``)."""
    token = "sgpr" + name
    return any(token in p for p in _param_texts(inst))


def _imm_in(inst, value: int) -> bool:
    """True if *inst* carries *value* as an immediate operand (dec or hex)."""
    for p in inst.getParams():
        try:
            if int(str(p), 0) == value:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _emit_queue_index(streamk: int, wgmXCC: int, skxcc: int = 0,
                      workGroupIdFromTTM: bool = False, numXCD: int = 8) -> list:
    """Render the shared ``_emitQueueIndex`` for a StreamK variant."""
    inst = streamKVariantClass(streamk)()
    writer = _FakeWriter(numXCD=numXCD, workGroupIdFromTTM=workGroupIdFromTTM)
    kernel = _kernel(streamk=streamk, wgmXCC=wgmXCC, skxcc=skxcc)
    sQueueIdx = writer.sgprPool.checkOut(1, "QueueIdx")
    wsLog2Queues = 3  # log2(8)
    module = inst._emitQueueIndex(writer, kernel, sQueueIdx, wsLog2Queues)
    return _flat(module)


# The carrier slot that holds the snapshotted raw rank (reused, in-window-dead).
_CARRIER = "StreamKTileIdx"


# ===========================================================================
# 1. Raw-rank paths: the queue index is the reused carrier (StreamKTileIdx)
#    masked % numQueues -- NOT the post-remap StreamKIdx shift derivation.
#    Applies uniformly to SK4 (StreamKDynamic) and SK5 (StreamKHybrid), and to
#    BOTH raw-rank regimes: dynamic auto-WGM (WGMXCC == -1) and SKXCC (WGMXCC>1).
# ===========================================================================
class TestRawRankQueueIndex:
    # (wgmXCC, skxcc) regimes that must derive the queue index from the raw rank.
    RAW_REGIMES = [
        pytest.param(-1, 0, id="auto-wgm"),
        pytest.param(2, 4, id="skxcc-wgmxcc2"),
        pytest.param(8, 8, id="skxcc-wgmxcc8"),
    ]

    @pytest.mark.parametrize("streamk", [4, 5])
    @pytest.mark.parametrize("wgmXCC,skxcc", RAW_REGIMES)
    def test_queue_index_masks_raw_rank_carrier(self, streamk, wgmXCC, skxcc):
        # For every raw-rank regime (dynamic auto-WGM and SKXCC with WGMXCC > 1),
        # the queue index must be a single mask of the raw-rank carrier, not the
        # StreamKIdx shr/shl/sub derivation.
        items = _emit_queue_index(streamk, wgmXCC=wgmXCC, skxcc=skxcc)
        ands = [i for i in items if isinstance(i, SAndB32)]
        assert len(ands) == 1, "raw-rank queue index must be a single mask op"
        assert _refs_sgpr(ands[0], _CARRIER), (
            "the queue index must be derived from the raw-rank %s carrier" % _CARRIER
        )
        # queue = rawWG & (numQueues-1); gfx942/gfx950 => 8 queues => mask 0x7.
        assert _imm_in(ands[0], 0x7), "expected the (numQueues-1) = 0x7 mask"

    @pytest.mark.parametrize("streamk", [4, 5])
    @pytest.mark.parametrize("wgmXCC,skxcc", RAW_REGIMES)
    def test_queue_index_does_not_use_post_remap_streamkidx(self, streamk, wgmXCC, skxcc):
        # On a raw-rank path the queue index must NOT come from the remapped
        # StreamKIdx (StreamKIdx % numQueues), so neither StreamKIdx nor its
        # shift/shift/sub derivation may appear.
        items = _emit_queue_index(streamk, wgmXCC=wgmXCC, skxcc=skxcc)
        assert not any(_refs_sgpr(i, "StreamKIdx") for i in items), (
            "the raw-rank queue index must not reference the post-remap StreamKIdx"
        )
        assert not any(
            isinstance(i, (SLShiftRightB32, SLShiftLeftB32, SSubU32)) for i in items
        ), "the raw-rank path must not emit the StreamKIdx shift/shift/sub derivation"

    @pytest.mark.parametrize("streamk", [4, 5])
    @pytest.mark.parametrize("wgmXCC,skxcc", RAW_REGIMES)
    def test_queue_index_never_uses_dead_streamkqueue(self, streamk, wgmXCC, skxcc):
        # The queue index never references a dedicated StreamKQueue SGPR; the raw
        # rank lives in the reused StreamKTileIdx carrier.
        items = _emit_queue_index(streamk, wgmXCC=wgmXCC, skxcc=skxcc)
        assert not any(_refs_sgpr(i, "StreamKQueue") for i in items), (
            "no dedicated StreamKQueue SGPR; the carrier is the reused StreamKTileIdx"
        )


# ===========================================================================
# 2. Count-preserving else-branch: the fix is deliberately NOT applied when the
#    remap is already count-preserving. The queue index stays the StreamKIdx
#    shift derivation and never reads the carrier as a raw rank. Covers:
#      * fixed non-SKXCC WGMXCC (== 1, or a tuned > 1 without SKXCC),
#      * SKXCC with WGMXCC == 1 (already count-preserving),
#      * gfx12 WorkGroupIdFromTTM (StreamKIdx already the raw id).
# ===========================================================================
class TestCountPreservingFallback:
    @pytest.mark.parametrize("streamk", [4, 5])
    @pytest.mark.parametrize(
        "wgmXCC,skxcc",
        [
            pytest.param(1, 0, id="wgmxcc1"),
            pytest.param(2, 0, id="fixed-wgmxcc2-no-skxcc"),
            pytest.param(1, 4, id="skxcc-wgmxcc1"),
        ],
    )
    def test_count_preserving_uses_streamkidx(self, streamk, wgmXCC, skxcc):
        items = _emit_queue_index(streamk, wgmXCC=wgmXCC, skxcc=skxcc)
        assert any(_refs_sgpr(i, "StreamKIdx") for i in items), (
            "count-preserving paths keep the StreamKIdx-derived queue index"
        )
        assert any(isinstance(i, SSubU32) for i in items), (
            "count-preserving paths keep the shift/shift/sub derivation"
        )
        assert not any(isinstance(i, SAndB32) for i in items), (
            "count-preserving paths must not mask a raw-rank carrier"
        )

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_gfx12_workgroupidfromttm_uses_streamkidx(self, streamk):
        # On WorkGroupIdFromTTM targets StreamKIdx is already the raw id, so the
        # raw-rank snapshot is unnecessary and the carrier is not read as a rank.
        items = _emit_queue_index(streamk, wgmXCC=-1, workGroupIdFromTTM=True)
        assert any(_refs_sgpr(i, "StreamKIdx") for i in items)
        assert not any(isinstance(i, SAndB32) for i in items)


# ===========================================================================
# 3. Scoping predicate: usesRawQueueRank / skUsesRawQueueRank are True only on
#    the raw-rank regimes (NumXCD > 1, not WorkGroupIdFromTTM, StreamK in (4,5))
#    for either WGMXCC == -1 or (StreamKXCCMapping != 0 and WGMXCC > 1).
# ===========================================================================
class _FakeKW:
    """Minimal stand-in for the KernelWriter self used by skUsesRawQueueRank."""

    def __init__(self, numXCD=8, workGroupIdFromTTM=False):
        self.states = types.SimpleNamespace(
            archCaps={"NumXCD": numXCD, "WorkGroupIdFromTTM": workGroupIdFromTTM})


class TestUsesRawQueueRankScoping:
    def test_true_for_dynamic_auto_wgm(self):
        assert StreamK.usesRawQueueRank(_FakeWriter(), _kernel(wgmXCC=-1)) is True

    @pytest.mark.parametrize("wgmXCC", [2, 4, 8])
    def test_true_for_skxcc_fixed_wgmxcc_gt1(self, wgmXCC):
        assert StreamK.usesRawQueueRank(
            _FakeWriter(), _kernel(wgmXCC=wgmXCC, skxcc=4)) is True

    @pytest.mark.parametrize("wgmXCC", [1, 2, 4])
    def test_false_for_fixed_wgmxcc_without_skxcc(self, wgmXCC):
        assert StreamK.usesRawQueueRank(
            _FakeWriter(), _kernel(wgmXCC=wgmXCC, skxcc=0)) is False

    def test_false_for_skxcc_wgmxcc1(self):
        # SKXCC with WGMXCC == 1 is already count-preserving -> cheap else-branch.
        assert StreamK.usesRawQueueRank(
            _FakeWriter(), _kernel(wgmXCC=1, skxcc=4)) is False

    def test_false_for_single_xcd(self):
        w = _FakeWriter(numXCD=1)
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=-1)) is False
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=8, skxcc=8)) is False

    def test_false_for_workgroupidfromttm(self):
        w = _FakeWriter(workGroupIdFromTTM=True)
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=-1)) is False
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=8, skxcc=8)) is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_true_for_dynamic_streamk(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=-1)) is True

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_true_for_skxcc_wgmxcc_gt1(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=8, skxcc=8)) is True

    @pytest.mark.parametrize("streamk", [0, 1, 2, 3])
    def test_kw_predicate_false_for_non_dynamic_streamk(self, streamk):
        # Only the SK4/SK5 dynamic-queue variants use the raw-rank carrier.
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=-1)) is False
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=8, skxcc=8)) is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_false_for_fixed_wgmxcc_without_skxcc(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=1)) is False
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=2, skxcc=0)) is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_false_for_skxcc_wgmxcc1(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=1, skxcc=4)) is False


# ---------------------------------------------------------------------------
# AST helpers: read the *real* KernelWriter / KernelWriterAssembly source.
# ---------------------------------------------------------------------------
def _source_of(func) -> str:
    return inspect.getsource(func)


def _mentions_const(node, value: str) -> bool:
    return any(
        isinstance(sub, ast.Constant) and sub.value == value
        for sub in ast.walk(node)
    )


def _is_sk_raw_rank_guard(test) -> bool:
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "skUsesRawQueueRank"
    )


# ===========================================================================
# 4. Both dynamic fetch helpers route through the shared _emitQueueIndex,
#    rather than each inlining its own StreamKIdx shift/shift/sub derivation.
#    graWorkGroup must call the helper (PAP pop-once) and must not re-inline
#    the queue pop.
# ===========================================================================
class TestSharedHelperRouting:
    @pytest.mark.parametrize(
        "func",
        [StreamKDynamic._fetchWorkItemAndBroadcast, StreamKHybrid._fetchWorkItemAndBroadcast],
    )
    def test_fetch_uses_shared_queue_index_helper(self, func):
        assert "_emitQueueIndex" in _source_of(func), (
            "both SK4 and SK5 must derive the queue index via _emitQueueIndex"
        )

    @pytest.mark.parametrize(
        "func", [StreamKDynamic.graWorkGroup, StreamKHybrid.graWorkGroup]
    )
    def test_grawg_routes_queue_pop_through_fetch_helper(self, func):
        src = _source_of(func)
        assert "_fetchWorkItemAndBroadcast" in src, (
            "PAP-extracted pop must stay in _fetchWorkItemAndBroadcast; "
            "graWorkGroup must not inline the queue pop"
        )
        assert "_emitQueueIndex" not in src, (
            "queue-index math belongs in the fetch helper, not graWorkGroup"
        )


# ===========================================================================
# 5. The raw-rank snapshot is emitted BEFORE wgmXCC rewrites WorkGroup0, and
#    only under the skUsesRawQueueRank guard, into the reused StreamKTileIdx
#    carrier. Verified against the real KernelWriterAssembly.defineAndResources
#    source.
# ===========================================================================
class TestSnapshotBeforeWgmXcc:
    _SNAP = 'SMovB32(dst=sgpr("%s"), src=sgpr("WorkGroup0")' % _CARRIER

    def _src(self) -> str:
        return _source_of(KernelWriterAssembly.defineAndResources)

    def test_snapshot_mov_targets_carrier(self):
        # The raw-rank snapshot targets the already-allocated StreamKTileIdx slot,
        # not a dedicated StreamKQueue SGPR.
        src = self._src()
        assert self._SNAP in src, (
            "the raw pre-remap launch id must be snapshotted into %s" % _CARRIER
        )
        assert 'sgpr("StreamKQueue")' not in src, (
            "StreamKQueue is removed; the snapshot must reuse StreamKTileIdx"
        )

    def test_snapshot_precedes_wgmxcc_reorder(self):
        src = self._src()
        snap = src.index('SMovB32(dst=sgpr("%s")' % _CARRIER)
        remap = src.index("module.add(wgmXCC(")
        assert snap < remap, (
            "the carrier snapshot must be emitted BEFORE the wgmXCC remap "
            "so it captures the RAW (pre-remap) workgroup id"
        )

    def test_snapshot_is_guarded_by_predicate(self):
        # The snapshot must sit inside `if self.skUsesRawQueueRank(kernel):`.
        tree = ast.parse(textwrap.dedent(self._src()))
        guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_sk_raw_rank_guard(node.test):
                if _mentions_const(node, _CARRIER):
                    guarded = True
        assert guarded, (
            "the carrier snapshot must be gated by skUsesRawQueueRank"
        )


# ===========================================================================
# 6. Zero-SGPR carrier reuse (the whole point of the SKXCC extension): NO
#    dedicated StreamKQueue SGPR is ever declared, and the carrier the snapshot
#    reuses (StreamKTileIdx) is an already-allocated persistent slot present for
#    BOTH the SK4 and SK5 dynamic paths -- so extending the fix to SKXCC adds no
#    net persistent SGPR and cannot regress the SGPR-overflow ceiling.
# ===========================================================================
class TestZeroSgprCarrierReuse:
    def _init_src(self) -> str:
        return _source_of(KernelWriter._initKernel)

    def _list_literals_with(self, value: str) -> int:
        """Count list literals (e.g. requiredUnalignedSgprVar += [...]) that
        include *value* as a string element, across the _initKernel source."""
        tree = ast.parse(textwrap.dedent(self._init_src()))
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and any(
                isinstance(e, ast.Constant) and e.value == value
                for e in node.elts
            ):
                count += 1
        return count

    def test_streamkqueue_sgpr_is_gone(self):
        # No StreamKQueue anywhere in the SGPR declaration source: neither an
        # append nor a list-literal element.
        src = self._init_src()
        assert '"StreamKQueue"' not in src and "'StreamKQueue'" not in src, (
            "the dedicated StreamKQueue SGPR must be removed (zero-SGPR reuse)"
        )

    def test_carrier_is_already_allocated_for_both_variants(self):
        # StreamKTileIdx is declared in both the SK4 and SK5 blocks (>= 2 list
        # literals), so the raw-rank snapshot reuses an existing slot for free.
        assert self._list_literals_with(_CARRIER) >= 2, (
            "%s must be an already-allocated persistent slot for SK4 and SK5" % _CARRIER
        )

    def test_no_streamkqueue_append_remains(self):
        # Belt-and-suspenders AST check: no requiredUnalignedSgprVar.append(
        # "StreamKQueue") survives anywhere.
        tree = ast.parse(textwrap.dedent(self._init_src()))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "StreamKQueue"
            ):
                pytest.fail("a StreamKQueue SGPR append still exists")
