# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the single-hop next-neighbor StreamK work-stealing codegen.

These tests assert that the work-stealing assembly is emitted by the helper
methods on the ``StreamK`` base class, and -- crucially -- that those helpers
are only ever reached behind the codegen-time ``StreamKWorkStealing`` toggle.
They import rocisa instructions and inspect emitted modules rather than matching
source text; the toggle gating and the Solution-level validation are verified by
executing the *real* source (via the AST) so the assertions track the actual code.

Emission contract (single-hop next-neighbor + sticky-home + static auto-reset):
  * The steal fires on an empty home fetch with no neighbor structural-extra guard.
  * The steal & home atomic bounds are the predecessor-inclusive self-reset value.
  * A per-WG sticky-empty SGPR gates the home fetch.
  * kernelEnd emits no explicit reset; per-queue counters auto-reset.
"""

import ast
import inspect
import textwrap
import types

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module, Label
from rocisa.instruction import (
    SAddU32,
    SAndB32,
    SAtomicInc,
    SBarrier,
    SCBranchSCC1,
    SCmpGeU32,
    SCmpLtU32,
    SMovB32,
    SSubU32,
)

from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.StreamK import (
    StreamK,
    StreamKDynamic,
    StreamKHybrid,
    streamKVariantClass,
)
from Tensile.SolutionStructs import Solution
from Tensile.SolutionStructs.Utilities import reject


# ---------------------------------------------------------------------------
# Fakes: just enough of a "writer" for the standalone helper methods.
#
# The helpers only touch ``writer.sgprPool`` (checkOut / checkOutAligned /
# checkIn) and emit rocisa instructions via free functions (sgpr/vgpr), so a
# tiny pool that hands out monotonically increasing register indices is all
# that is required -- no KernelWriter, no GPU.
# ---------------------------------------------------------------------------
class _FakeSgprPool:
    def __init__(self, start: int = 100):
        self._next = start

    def checkOut(self, n: int, name: str = "", *args, **kwargs) -> int:
        reg = self._next
        self._next += n
        return reg

    def checkOutAligned(self, n: int, align: int, name: str = "", *args, **kwargs) -> int:
        if self._next % align:
            self._next += align - (self._next % align)
        reg = self._next
        self._next += n
        return reg

    def checkIn(self, *args, **kwargs):
        return None


class _FakeWriter:
    def __init__(self, numXCD: int = 8, cacheLineBytes: int = 128):
        self.sgprPool = _FakeSgprPool()
        # gfx942/gfx950 mirror origami get_default_num_xcds == 8 and origami
        # get_default_cache_line_bytes == 128; the helpers read the per-arch
        # queue count from writer.states.archCaps["NumXCD"] and the per-queue
        # counter stride from writer.states.archCaps["CacheLineBytes"].
        self.states = types.SimpleNamespace(
            archCaps={"NumXCD": numXCD, "CacheLineBytes": cacheLineBytes})


def _mk_label(base: str) -> Label:
    return Label(base, "")


# gfx942/gfx950 both map to 8 queues in the per-arch lookup; the helpers key
# their fast-mask constants off ``kernel["ISA"]``.
_WS_KERNEL = {"ISA": (9, 4, 0)}


def _stream_k_instance(streamk: int) -> StreamK:
    """A concrete StreamK variant (helpers live on the base class)."""
    return streamKVariantClass(streamk)()


def _imm_in(inst, value: int) -> bool:
    """True if *inst* carries *value* as an immediate operand.

    rocisa renders immediates inconsistently -- ints passed straight through
    print as decimal ("7"), while values passed as ``hex(...)`` print as
    "0x..." -- so normalise every param through ``int(p, 0)`` and compare.
    """
    for p in inst.getParams():
        try:
            if int(str(p), 0) == value:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _flat(module: Module) -> list:
    return list(module.flatitems())


# ---------------------------------------------------------------------------
# AST helpers: read the *real* StreamK / Solution source and reason about it.
# ---------------------------------------------------------------------------
def _const_slice(subscript: ast.Subscript):
    s = subscript.slice
    if isinstance(s, ast.Constant):
        return s.value
    return None


def _is_subscript_on(node, name: str, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
        and _const_slice(node) == key
    )


def _ws_guarded_calls(func) -> set:
    """Names of ``self.streamKWorkStealing*`` calls that sit inside an
    ``if kernel["StreamKWorkStealing"]:`` block in *func* (recursing into
    nested closures)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_subscript_on(
            node.test, "kernel", "StreamKWorkStealing"
        ):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if sub.func.attr.startswith("streamKWorkStealing"):
                        guarded.add(sub.func.attr)
    return guarded


def _all_ws_calls(func) -> set:
    """Every ``self.streamKWorkStealing*`` call in *func*, guarded or not."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("streamKWorkStealing"):
                calls.add(node.func.attr)
    return calls


def _source_of(func) -> str:
    return inspect.getsource(func)


def _extract_ws_validation():
    """Compile the *real* ``if state["StreamKWorkStealing"]:`` block out of
    ``Solution.assignDerivedParameters`` into a standalone callable so the
    actual rejection logic can be exercised without a full Solution state.

    The block reads ``isaInfoMap[isa].asmCaps`` (for the HasSAtomic / gfx1250
    gate), so ``isa`` and ``isaInfoMap`` are threaded in as parameters -- the
    real function derives ``isa = tuple(state["ISA"])`` and receives
    ``isaInfoMap`` as an argument."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(Solution.assignDerivedParameters))
    )
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_subscript_on(
            node.test, "state", "StreamKWorkStealing"
        ):
            target = node
            break
    assert target is not None, "could not find StreamKWorkStealing validation block"

    func = ast.FunctionDef(
        name="_validate",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg("state"),
                ast.arg("printRejectionReason"),
                ast.arg("reject"),
                ast.arg("isa"),
                ast.arg("isaInfoMap"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[target],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    mod = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {}
    exec(compile(mod, "<ws-validation>", "exec"), ns)
    return ns["_validate"]


# gfx1250 (12,5,0) has no scalar atomics; gfx942/gfx950 do. Mirror just the
# HasSAtomic capability the work-stealing validation reads.
_NO_SATOMIC_ISAS = {(12, 5, 0)}


def _fake_isa_info_map(isa):
    isa = tuple(isa)
    asm_caps = {"HasSAtomic": isa not in _NO_SATOMIC_ISAS}
    return {isa: types.SimpleNamespace(asmCaps=asm_caps)}


# ===========================================================================
# 1. ValidParameters: the codegen-time toggle exists and is boolean.
# ===========================================================================
class TestValidParameters:
    def test_work_stealing_param_exists(self):
        assert "StreamKWorkStealing" in validParameters

    def test_work_stealing_param_is_zero_one(self):
        assert validParameters["StreamKWorkStealing"] == [0, 1]


# ===========================================================================
# 2. The work-stealing helper methods exist on the StreamK base class.
# ===========================================================================
class TestHelperMethodsExist:
    @pytest.mark.parametrize(
        "name",
        [
            "streamKWorkStealingHomeBound",
            "streamKWorkStealingSteal",
        ],
    )
    def test_method_is_defined_on_base(self, name):
        assert callable(getattr(StreamK, name))


# ===========================================================================
# 3a. Home auto-reset bound: the predecessor's workgroup count is folded into
#     the atomic_inc bound (NOT disabled with 0xFFFFFFFF).
# ===========================================================================
class TestHomeBoundEmission:
    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("home-bound")
        sBound = writer.sgprPool.checkOut(1, "bound")
        sQueueIdx = writer.sgprPool.checkOut(1, "queueIdx")
        sk.streamKWorkStealingHomeBound(
            writer, module, _WS_KERNEL, sBound, sQueueIdx, "skGrid"
        )
        return _flat(module)

    def test_computes_predecessor_index(self):
        items = self._emit()
        # p = (q - 1) & 0x7
        assert any(isinstance(i, SSubU32) and _imm_in(i, 1) for i in items), (
            "expected q-1 to reach the predecessor queue"
        )
        assert any(isinstance(i, SAndB32) and _imm_in(i, 0x7) for i in items), (
            "expected (q-1) & 0x7 wrap for the predecessor index"
        )

    def test_adds_predecessor_workgroups_to_bound(self):
        items = self._emit()
        # The predecessor structural share plus the final fold-in add.
        assert sum(isinstance(i, SAddU32) for i in items) >= 2, (
            "expected the W_(q-1) structural add and the bound fold-in"
        )

    def test_does_not_disable_auto_reset(self):
        items = self._emit()
        assert not any(
            isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF) for i in items
        ), "the home bound must be a finite predecessor-inclusive value"


# ===========================================================================
# 3b. Next-neighbor steal: one unconditional (past home-empty) NEXT-neighbor steal with
#     the static predecessor-inclusive auto-reset bound. No remainder/extra
#     guards.
# ===========================================================================
class TestStealEmission:
    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("steal")
        sQueueIdx = writer.sgprPool.checkOut(1, "queueIdx")
        sWorkItemIdx = writer.sgprPool.checkOut(1, "workItemIdx")
        sk.streamKWorkStealingSteal(
            writer, module, _WS_KERNEL, sQueueIdx, sWorkItemIdx, "skGrid", _mk_label
        )
        return _flat(module)

    def test_neighbor_walk_is_plus_one_then_wrap(self):
        items = self._emit()
        # +1 to advance to the next neighbor ...
        assert any(
            isinstance(i, SAddU32) and _imm_in(i, 1) for i in items
        ), "expected +1 advance to the next queue"
        # ... wrapped within the 8 queues (single-hop next-neighbor) via & 0x7.
        assert any(
            isinstance(i, SAndB32) and _imm_in(i, 0x7) for i in items
        ), "expected (queueIdx+1) & 0x7 wrap"

    def test_no_neighbor_extra_guard(self):
        # The next-neighbor steal always fires on an empty home fetch: the old
        # ">= remainder / neighbor-has-no-structural-extra" guard is removed.
        items = self._emit()
        assert not any(isinstance(i, SCmpGeU32) for i in items), (
            "the next-neighbor steal must NOT guard on a neighbor structural extra"
        )

    def test_exactly_one_atomic_increment(self):
        items = self._emit()
        atomics = [i for i in items if isinstance(i, SAtomicInc)]
        assert len(atomics) == 1, "the steal must emit exactly one atomic"

    def test_atomic_bound_is_predecessor_inclusive_not_all_ones(self):
        items = self._emit()
        # The static bound tiles_s + W_s + W_q - 1 ends in a -1 (SSubU32 imm 1)
        # and must never load the 0xFFFFFFFF disable-auto-reset sentinel.
        assert not any(
            isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF) for i in items
        ), "the stolen atomic must use a finite self-reset bound, not 0xFFFFFFFF"
        assert any(
            isinstance(i, SSubU32) and _imm_in(i, 1) for i in items
        ), "expected the '- 1' that forms the atomic_inc auto-reset bound"

    def test_guards_on_valid_home_fetch(self):
        # A valid home fetch (index < TotalItems) must short-circuit the steal.
        items = self._emit()
        assert any(isinstance(i, SCmpLtU32) for i in items)
        assert any(isinstance(i, SCBranchSCC1) for i in items)

    def test_no_reset_barrier(self):
        items = self._emit()
        assert not any(isinstance(i, SBarrier) for i in items), (
            "the steal helper must not emit a reset barrier"
        )


# ===========================================================================
# 3c. Sticky-home: the home fetch is gated by a persistent StreamKStickyEmpty
#     SGPR, and the flag is latched on an empty home fetch. Verified against
#     the real graWorkGroup source.
# ===========================================================================
class TestStickyHomeGate:
    @pytest.mark.parametrize(
        "func", [StreamKDynamic.graWorkGroup, StreamKHybrid.graWorkGroup]
    )
    def test_home_fetch_is_gated_by_sticky_flag(self, func):
        src = _source_of(func)
        assert "StreamKStickyEmpty" in src, (
            "the home fetch must be gated by the persistent sticky-empty SGPR"
        )
        # A steal-only skip label proves the home s_atomic_inc is bypassed once
        # the WG has gone sticky.
        assert "SK_StealOnly" in src, (
            "sticky WGs must skip the home fetch and steal only"
        )

    @pytest.mark.parametrize(
        "func", [StreamKDynamic.graWorkGroup, StreamKHybrid.graWorkGroup]
    )
    def test_steal_passes_grid_sgpr(self, func):
        # The steal now needs the mode-appropriate grid SGPR for its bound.
        src = _source_of(func)
        grid = "SKGrid" if func is StreamKHybrid.graWorkGroup else "skGrid"
        assert "streamKWorkStealingSteal" in src
        assert grid in src


# ===========================================================================
# 3d. kernelEnd emits no work-stealing reset and no completion counter.
# ===========================================================================
class TestNoExplicitReset:
    @pytest.mark.parametrize("func", [StreamKDynamic.kernelEnd, StreamKHybrid.kernelEnd])
    def test_kernelend_has_no_ws_calls(self, func):
        assert _all_ws_calls(func) == set(), (
            "kernelEnd must not emit any work-stealing reset"
        )

    @pytest.mark.parametrize("func", [StreamKDynamic.kernelEnd, StreamKHybrid.kernelEnd])
    def test_kernelend_has_no_completion_counter(self, func):
        src = _source_of(func)
        assert "0x80" not in src
        assert "completion" not in src.lower() or "no explicit" in src.lower()


# ===========================================================================
# 5. Per-architecture queue-count lookup (C1): fast masking requires a
#    power-of-two queue count; the count is read from the per-arch capability
#    writer.states.archCaps["NumXCD"] (the codegen mirror of origami
#    get_default_num_xcds).
# ===========================================================================
class TestQueueConstants:
    @pytest.mark.parametrize("isa", [(9, 4, 0), (9, 5, 0)])
    def test_supported_arches_use_eight_power_of_two_queues(self, isa):
        sk = _stream_k_instance(4)
        # gfx942/gfx950 mirror origami get_default_num_xcds == 8 and
        # get_default_cache_line_bytes == 128, so the constants tuple is exactly
        # (numQueues=8, mask=7, log2=3, cacheLineLog2=log2(128)=7).
        writer = _FakeWriter(numXCD=8, cacheLineBytes=128)
        assert sk._wsQueueConstants(writer, {"ISA": isa}) == (8, 7, 3, 7)

    def test_non_power_of_two_queue_count_asserts(self):
        # The shift/AND fast masking is only valid for a power-of-two queue
        # count; a non-power-of-two NumXCD cap must trip the guard assert.
        sk = _stream_k_instance(4)
        writer = _FakeWriter(numXCD=6)
        with pytest.raises(AssertionError):
            sk._wsQueueConstants(writer, {"ISA": (9, 9, 0)})


# ===========================================================================
# 3e. Absence-by-toggle: the helpers are only reached behind the
#     ``kernel["StreamKWorkStealing"]`` gate at every callsite. Verified
#     against the real source so "off" provably emits nothing extra.
# ===========================================================================
class TestCallsitesAreToggleGated:
    def test_sk4_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKDynamic.graWorkGroup)
        allcalls = _all_ws_calls(StreamKDynamic.graWorkGroup)
        assert {"streamKWorkStealingHomeBound", "streamKWorkStealingSteal"} <= guarded
        # Nothing slips through ungated.
        assert allcalls == guarded

    def test_sk5_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKHybrid.graWorkGroup)
        allcalls = _all_ws_calls(StreamKHybrid.graWorkGroup)
        assert {"streamKWorkStealingHomeBound", "streamKWorkStealingSteal"} <= guarded
        assert allcalls == guarded


# ===========================================================================
# 4. Solution validation: the real rejection logic from
#    assignDerivedParameters, executed in isolation.
# ===========================================================================
class TestSolutionValidation:
    def setup_method(self):
        self.validate = _extract_ws_validation()

    def _run(self, *, streamk, atomic, work_stealing=1, debug_streamk=0, isa=(9, 4, 2)):
        isa = tuple(isa)
        state = {
            "StreamKWorkStealing": work_stealing,
            "StreamK": streamk,
            "StreamKAtomic": atomic,
            "DebugStreamK": debug_streamk,
            "ISA": isa,
        }
        self.validate(state, False, reject, isa, _fake_isa_info_map(isa))
        return state

    @pytest.mark.parametrize("streamk", [0, 1, 2, 3])
    def test_rejected_when_streamk_not_4_or_5(self, streamk):
        state = self._run(streamk=streamk, atomic=0)
        assert state["Valid"] is False

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_accepted_for_dynamic_and_hybrid_without_atomic(self, streamk):
        state = self._run(streamk=streamk, atomic=0)
        assert state.get("Valid", True) is True

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_rejected_with_atomic(self, streamk):
        state = self._run(streamk=streamk, atomic=1)
        assert state["Valid"] is False

    @pytest.mark.parametrize("debug", [1, 2, 3])
    def test_rejected_with_debug_streamk(self, debug):
        # DebugStreamK overrides can break the W_q>=1 precondition the per-queue
        # auto-reset relies on, so the combination is rejected.
        state = self._run(streamk=4, atomic=0, debug_streamk=debug)
        assert state["Valid"] is False

    def test_off_toggle_is_inert_even_for_bad_combo(self):
        # With the toggle off the guard must not fire, even for a combination
        # that would otherwise be rejected.
        state = self._run(streamk=3, atomic=1, work_stealing=0, debug_streamk=3)
        assert "Valid" not in state

    @pytest.mark.parametrize("streamk", [4, 5])
    def test_rejected_on_gfx1250_no_scalar_atomic(self, streamk):
        # gfx1250 lacks scalar atomics (HasSAtomic=false); the steal path emits
        # s_atomic_inc with no vector fallback, so work stealing is rejected.
        state = self._run(streamk=streamk, atomic=0, isa=(12, 5, 0))
        assert state["Valid"] is False

    @pytest.mark.parametrize("isa", [(9, 4, 2), (9, 5, 0)])
    @pytest.mark.parametrize("streamk", [4, 5])
    def test_accepted_on_scalar_atomic_arches(self, isa, streamk):
        # gfx942/gfx950 have scalar atomics, so work stealing stays accepted.
        state = self._run(streamk=streamk, atomic=0, isa=isa)
        assert state.get("Valid", True) is True

    def test_off_toggle_is_inert_on_gfx1250(self):
        # StreamKWorkStealing=0 on gfx1250 must not fire the reject.
        state = self._run(streamk=4, atomic=0, work_stealing=0, isa=(12, 5, 0))
        assert "Valid" not in state
