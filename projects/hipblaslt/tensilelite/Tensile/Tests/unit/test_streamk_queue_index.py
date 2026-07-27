# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the count-preserving per-XCD StreamK work-queue index.

These tests pin the ``abbf172210`` codegen fix: the SK4 (``StreamKDynamic``) and
SK5 (``StreamKHybrid``) dynamic work-queue per-XCD atomic counters self-reset to
0 each launch only if every queue receives exactly ``tiles_q + W_q`` increments,
which requires the value feeding ``% numQueues`` to densely cover ``[0, skGrid)``.
The queue index therefore must come from the RAW pre-wgmXCC launch WG rank
(snapshotted into the dedicated persistent ``StreamKQueue`` SGPR), NOT from the
wgmXCC-remapped ``StreamKIdx`` (whose ``% numQueues`` is not count-preserving
when the grid does not block evenly).

Like the sibling StreamK codegen tests, they import rocisa instructions and
inspect emitted modules rather than matching source text, and reason about the
*real* KernelWriter / KernelWriterAssembly source via the AST so the ordering and
gating assertions track the actual code.

Invariants pinned (each fails against the pre-fix code -- see per-test notes):
  * The dynamic auto-WGM queue index is ``StreamKQueue & (numQueues-1)`` -- a
    single mask of the raw-rank SGPR, never the post-remap StreamKIdx shifts.
  * Both SK4 and SK5 route their queue index through the shared ``_emitQueueIndex``.
  * The raw-rank snapshot ``s_mov_b32 StreamKQueue, WorkGroup0`` is emitted
    BEFORE the wgmXCC workgroup remap.
  * The ``StreamKQueue`` SGPR is declared only on the dynamic auto-WGM path
    (``skUsesRawQueueRank``), guarding the fixed-WGMXCC SGPR-overflow regression.
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
def _kernel(streamk: int = 4, wgmXCC: int = -1, isa=(9, 4, 0)) -> dict:
    return {"ISA": isa, "StreamK": streamk, "WorkGroupMappingXCC": wgmXCC}


def _flat(module: Module) -> list:
    return list(module.flatitems())


def _param_texts(inst) -> list:
    return [str(p) for p in inst.getParams()]


def _refs_sgpr(inst, name: str) -> bool:
    """True if *inst* references the named SGPR (e.g. ``s[sgprStreamKQueue]``)."""
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


def _emit_queue_index(streamk: int, wgmXCC: int, workGroupIdFromTTM: bool = False,
                      numXCD: int = 8) -> list:
    """Render the shared ``_emitQueueIndex`` for a StreamK variant."""
    inst = streamKVariantClass(streamk)()
    writer = _FakeWriter(numXCD=numXCD, workGroupIdFromTTM=workGroupIdFromTTM)
    kernel = _kernel(streamk=streamk, wgmXCC=wgmXCC)
    sQueueIdx = writer.sgprPool.checkOut(1, "QueueIdx")
    wsLog2Queues = 3  # log2(8)
    module = inst._emitQueueIndex(writer, kernel, sQueueIdx, wsLog2Queues)
    return _flat(module)


# ===========================================================================
# 1. Dynamic auto-WGM (WGMXCC == -1): the queue index is the raw-rank SGPR
#    masked % numQueues -- NOT the post-remap StreamKIdx shift derivation.
#    Applies uniformly to SK4 (StreamKDynamic) and SK5 (StreamKHybrid).
# ===========================================================================
class TestRawRankQueueIndex:
    @pytest.mark.parametrize("streamk", [4, 5])
    def test_queue_index_masks_raw_rank_sgpr(self, streamk):
        # Pre-fix emitted shr/shl/sub of StreamKIdx and NO SAndB32 -- so both the
        # "single SAndB32" and "references StreamKQueue" assertions fail on it.
        items = self._raw(streamk)
        ands = [i for i in items if isinstance(i, SAndB32)]
        assert len(ands) == 1, "raw-rank queue index must be a single mask op"
        assert _refs_sgpr(ands[0], "StreamKQueue"), (
            "the queue index must be derived from the raw-rank StreamKQueue SGPR"
        )
        # queue = rawWG & (numQueues-1); gfx942/gfx950 => 8 queues => mask 0x7.
        assert _imm_in(ands[0], 0x7), "expected the (numQueues-1) = 0x7 mask"

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_queue_index_does_not_use_post_remap_streamkidx(self, streamk):
        # The whole point of the fix: on the dynamic auto-WGM path the queue
        # index must NOT come from the wgmXCC-remapped StreamKIdx. Pre-fix code
        # emitted exactly this shift/shift/sub of StreamKIdx, so this fails on it.
        items = self._raw(streamk)
        assert not any(_refs_sgpr(i, "StreamKIdx") for i in items), (
            "the dynamic-queue index must not reference the post-remap StreamKIdx"
        )
        assert not any(
            isinstance(i, (SLShiftRightB32, SLShiftLeftB32, SSubU32)) for i in items
        ), "the raw-rank path must not emit the StreamKIdx shift/shift/sub derivation"

    def _raw(self, streamk: int) -> list:
        return _emit_queue_index(streamk, wgmXCC=-1)


# ===========================================================================
# 2. Fixed WGMXCC (== 1 or a tuned > 1): the fix is deliberately NOT applied.
#    The queue index stays the StreamKIdx shift derivation and never reads the
#    StreamKQueue SGPR (which is not even declared there -- see class 5), so the
#    register-ceiling fixed-WGMXCC kernels do not overflow the SGPR file.
# ===========================================================================
class TestFixedWgmXccFallback:
    @pytest.mark.parametrize("streamk", [4, 5])
    @pytest.mark.parametrize("wgmXCC", [1, 2])
    def test_fixed_wgmxcc_uses_streamkidx_not_queue_sgpr(self, streamk, wgmXCC):
        items = _emit_queue_index(streamk, wgmXCC=wgmXCC)
        assert not any(_refs_sgpr(i, "StreamKQueue") for i in items), (
            "fixed-WGMXCC kernels must not read the StreamKQueue SGPR "
            "(guards the SGPR-overflow regression)"
        )
        assert any(_refs_sgpr(i, "StreamKIdx") for i in items), (
            "fixed-WGMXCC keeps the StreamKIdx-derived queue index"
        )
        assert any(isinstance(i, SSubU32) for i in items), (
            "fixed-WGMXCC keeps the shift/shift/sub derivation"
        )

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_gfx12_workgroupidfromttm_uses_streamkidx(self, streamk):
        # On WorkGroupIdFromTTM targets StreamKIdx is already the raw id, so the
        # raw-rank snapshot is unnecessary and the StreamKQueue SGPR is not read.
        items = _emit_queue_index(streamk, wgmXCC=-1, workGroupIdFromTTM=True)
        assert not any(_refs_sgpr(i, "StreamKQueue") for i in items)
        assert any(_refs_sgpr(i, "StreamKIdx") for i in items)


# ===========================================================================
# 3. Scoping predicate: usesRawQueueRank / skUsesRawQueueRank are True only on
#    the dynamic auto-WGM path (NumXCD > 1, not WorkGroupIdFromTTM, WGMXCC==-1)
#    for StreamK in (4, 5).
# ===========================================================================
class _FakeKW:
    """Minimal stand-in for the KernelWriter self used by skUsesRawQueueRank."""

    def __init__(self, numXCD=8, workGroupIdFromTTM=False):
        self.states = types.SimpleNamespace(
            archCaps={"NumXCD": numXCD, "WorkGroupIdFromTTM": workGroupIdFromTTM})


class TestUsesRawQueueRankScoping:
    def test_true_for_dynamic_auto_wgm(self):
        assert StreamK.usesRawQueueRank(_FakeWriter(), _kernel(wgmXCC=-1)) is True

    @pytest.mark.parametrize("wgmXCC", [1, 2, 4])
    def test_false_for_fixed_wgmxcc(self, wgmXCC):
        assert StreamK.usesRawQueueRank(_FakeWriter(), _kernel(wgmXCC=wgmXCC)) is False

    def test_false_for_single_xcd(self):
        w = _FakeWriter(numXCD=1)
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=-1)) is False

    def test_false_for_workgroupidfromttm(self):
        w = _FakeWriter(workGroupIdFromTTM=True)
        assert StreamK.usesRawQueueRank(w, _kernel(wgmXCC=-1)) is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_true_for_dynamic_streamk(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=-1)) is True

    @pytest.mark.parametrize("streamk", [0, 1, 2, 3])
    def test_kw_predicate_false_for_non_dynamic_streamk(self, streamk):
        # Only the SK4/SK5 dynamic-queue variants get the StreamKQueue SGPR.
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=-1)) is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_kw_predicate_false_for_fixed_wgmxcc(self, streamk):
        assert KernelWriter.skUsesRawQueueRank(
            _FakeKW(), _kernel(streamk=streamk, wgmXCC=1)) is False


# ---------------------------------------------------------------------------
# AST helpers: read the *real* KernelWriter / KernelWriterAssembly source.
# ---------------------------------------------------------------------------
def _source_of(func) -> str:
    return inspect.getsource(func)


def _is_streamkqueue_append(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "requiredUnalignedSgprVar"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "StreamKQueue"
    )


def _is_sk_raw_rank_guard(test) -> bool:
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == "skUsesRawQueueRank"
    )


# ===========================================================================
# 4. Both dynamic graWorkGroup paths route through the shared _emitQueueIndex.
#    Pre-fix each inlined its own StreamKIdx shift/shift/sub, so the shared
#    helper call is absent -> this fails against the pre-fix source.
# ===========================================================================
class TestSharedHelperRouting:
    @pytest.mark.parametrize(
        "func", [StreamKDynamic.graWorkGroup, StreamKHybrid.graWorkGroup]
    )
    def test_grawg_uses_shared_queue_index_helper(self, func):
        assert "_emitQueueIndex" in _source_of(func), (
            "both SK4 and SK5 must derive the queue index via _emitQueueIndex"
        )


# ===========================================================================
# 5. The raw-rank snapshot is emitted BEFORE wgmXCC rewrites WorkGroup0, and
#    only under the skUsesRawQueueRank guard. Verified against the real
#    KernelWriterAssembly.defineAndResources source.
# ===========================================================================
class TestSnapshotBeforeWgmXcc:
    def _src(self) -> str:
        return _source_of(KernelWriterAssembly.defineAndResources)

    def test_snapshot_mov_is_present(self):
        # Pre-fix there was no snapshot at all -> this fails.
        assert 'SMovB32(dst=sgpr("StreamKQueue"), src=sgpr("WorkGroup0")' in self._src(), (
            "the raw pre-wgmXCC launch id must be snapshotted into StreamKQueue"
        )

    def test_snapshot_precedes_wgmxcc_reorder(self):
        src = self._src()
        snap = src.index('SMovB32(dst=sgpr("StreamKQueue")')
        remap = src.index("module.add(wgmXCC(")
        assert snap < remap, (
            "the StreamKQueue snapshot must be emitted BEFORE the wgmXCC remap "
            "so it captures the RAW (pre-remap) workgroup id"
        )

    def test_snapshot_is_guarded_by_predicate(self):
        # The snapshot must sit inside `if self.skUsesRawQueueRank(kernel):`.
        tree = ast.parse(textwrap.dedent(self._src()))
        guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_sk_raw_rank_guard(node.test):
                if any(
                    isinstance(sub, ast.Constant) and sub.value == "StreamKQueue"
                    for sub in ast.walk(node)
                ):
                    guarded = True
        assert guarded, (
            "the StreamKQueue snapshot must be gated by skUsesRawQueueRank"
        )


# ===========================================================================
# 6. The StreamKQueue SGPR is declared only under skUsesRawQueueRank -- once for
#    the SK4 block and once for the SK5 block -- never unconditionally. This is
#    the def-level guard against the fixed-WGMXCC SGPR-overflow regression.
# ===========================================================================
class TestSgprDeclarationGated:
    def _appends(self):
        func = KernelWriter._initKernel
        tree = ast.parse(textwrap.dedent(_source_of(func)))
        all_appends = [n for n in ast.walk(tree) if _is_streamkqueue_append(n)]
        guarded_ids = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_sk_raw_rank_guard(node.test):
                for sub in ast.walk(node):
                    if _is_streamkqueue_append(sub):
                        guarded_ids.add(id(sub))
        return all_appends, guarded_ids

    def test_streamk_queue_declared_for_both_variants(self):
        # One guarded append in the SK4 block, one in the SK5 block.
        all_appends, _ = self._appends()
        assert len(all_appends) == 2, (
            "StreamKQueue must be declared for both SK4 and SK5 dynamic paths"
        )

    def test_every_streamk_queue_append_is_guarded(self):
        all_appends, guarded_ids = self._appends()
        assert len(guarded_ids) == len(all_appends) and all_appends, (
            "every StreamKQueue SGPR declaration must be gated by "
            "skUsesRawQueueRank (fixed-WGMXCC kernels must not allocate it)"
        )
