# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the single-hop StreamK work-stealing codegen.

These tests assert that the new work-stealing assembly is emitted by the
helper methods on the ``StreamK`` base class, and -- crucially -- that those
helpers are only ever reached behind the codegen-time ``StreamKWorkStealing``
toggle. Following the StreamK=5 hybrid tests, they import rocisa instructions
and inspect emitted modules rather than matching source text; the toggle gating
and the Solution-level validation are verified by executing the *real* source
(via the AST) so the assertions track the actual code, not a copy of it.
"""

import ast
import inspect
import textwrap

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module, Label
from rocisa.instruction import (
    SAddU32,
    SAndB32,
    SAtomicInc,
    SBarrier,
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpGeU32,
    SCmpLtU32,
    SMovB32,
    SStoreB32,
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
# The three helpers only touch ``writer.sgprPool`` (checkOut / checkOutAligned
# / checkIn) and emit rocisa instructions via free functions (sgpr/vgpr), so a
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
    def __init__(self):
        self.sgprPool = _FakeSgprPool()


def _mk_label(base: str) -> Label:
    return Label(base, "")


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


def _extract_ws_validation():
    """Compile the *real* ``if state["StreamKWorkStealing"]:`` block out of
    ``Solution.assignDerivedParameters`` into a standalone callable so the
    actual rejection logic can be exercised without a full Solution state."""
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
            args=[ast.arg("state"), ast.arg("printRejectionReason"), ast.arg("reject")],
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


# ===========================================================================
# 1. ValidParameters: the codegen-time toggle exists and is boolean.
# ===========================================================================
class TestValidParameters:
    def test_work_stealing_param_exists(self):
        assert "StreamKWorkStealing" in validParameters

    def test_work_stealing_param_is_zero_one(self):
        assert validParameters["StreamKWorkStealing"] == [0, 1]


# ===========================================================================
# 2. The three new StreamK helper methods exist and are callable.
# ===========================================================================
class TestHelperMethodsExist:
    @pytest.mark.parametrize(
        "name",
        [
            "streamKWorkStealingHomeNoReset",
            "streamKWorkStealingSteal",
            "streamKWorkStealingKernelEndReset",
        ],
    )
    def test_method_is_defined_on_base(self, name):
        assert callable(getattr(StreamK, name))


# ===========================================================================
# 3a. Presence: the helpers actually emit the work-stealing assembly.
# ===========================================================================
class TestHomeNoResetEmission:
    """Disabling the home auto-reset must mask TotalItems with 0x7 and, when a
    remainder exists, move 0xFFFFFFFF into the auto-reset bound register."""

    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("home-no-reset")
        sBound = writer.sgprPool.checkOut(1, "bound")
        sk.streamKWorkStealingHomeNoReset(writer, module, {}, sBound, _mk_label)
        return _flat(module)

    def test_masks_total_items_with_queue_mask(self):
        items = self._emit()
        masks = [
            i for i in items
            if isinstance(i, SAndB32) and _imm_in(i, 0x7)
        ]
        assert masks, "expected an s_and_b32 with the 0x7 queue mask"

    def test_disables_auto_reset_with_all_ones(self):
        items = self._emit()
        movs = [
            i for i in items
            if isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF)
        ]
        assert movs, "expected 0xFFFFFFFF mov to disable the home auto-reset"


class TestStealEmission:
    """One single-hop NEXT steal: walk to (queueIdx+1) & 0x7, guard against the
    neighbor having no structural extra, then a single auto-reset-disabled
    atomic increment against the stolen queue."""

    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = Module("steal")
        sQueueIdx = writer.sgprPool.checkOut(1, "queueIdx")
        sWorkItemIdx = writer.sgprPool.checkOut(1, "workItemIdx")
        sk.streamKWorkStealingSteal(
            writer, module, {}, sQueueIdx, sWorkItemIdx, _mk_label
        )
        return _flat(module)

    def test_neighbor_walk_is_plus_one_then_wrap(self):
        items = self._emit()
        # +1 to advance to the next neighbor ...
        assert any(
            isinstance(i, SAddU32) and _imm_in(i, 1) for i in items
        ), "expected +1 advance to the next queue"
        # ... wrapped within the 8-queue ring via & 0x7.
        assert any(
            isinstance(i, SAndB32) and _imm_in(i, 0x7) for i in items
        ), "expected (queueIdx+1) & 0x7 wrap"

    def test_skips_when_neighbor_has_no_extra(self):
        items = self._emit()
        assert any(isinstance(i, SCmpGeU32) for i in items), (
            "expected a >= remainder guard so a neighbor without a structural "
            "extra is not robbed"
        )

    def test_exactly_one_atomic_increment(self):
        items = self._emit()
        atomics = [i for i in items if isinstance(i, SAtomicInc)]
        assert len(atomics) == 1, "single-hop steal must emit exactly one atomic"

    def test_atomic_uses_auto_reset_disabled_bound(self):
        items = self._emit()
        assert any(
            isinstance(i, SMovB32) and _imm_in(i, 0xFFFFFFFF)
            for i in items
        ), "the stolen atomic must run with auto-reset disabled (0xFFFFFFFF)"

    def test_guards_on_valid_home_fetch(self):
        # A valid home fetch (index < TotalItems) must short-circuit the steal.
        items = self._emit()
        assert any(isinstance(i, SCmpLtU32) for i in items)
        assert any(isinstance(i, SCBranchSCC1) for i in items)


class TestKernelEndResetEmission:
    """The last WG zeroes the 8 per-queue counters plus the completion counter,
    behind a barrier + wave-0 completion count."""

    def _emit(self):
        sk = _stream_k_instance(4)
        writer = _FakeWriter()
        module = sk.streamKWorkStealingKernelEndReset(writer, {}, "skGrid", _mk_label)
        return _flat(module)

    def test_starts_with_barrier(self):
        items = self._emit()
        assert any(isinstance(i, SBarrier) for i in items)

    def test_resets_eight_queues_plus_completion_counter(self):
        items = self._emit()
        stores = [i for i in items if isinstance(i, SStoreB32)]
        assert len(stores) == StreamK._WS_NUM_QUEUES + 1 == 9, (
            "expected 8 per-queue counter resets + 1 completion counter reset"
        )

    def test_completion_count_uses_atomic_inc(self):
        items = self._emit()
        assert any(isinstance(i, SAtomicInc) for i in items), (
            "wave 0 counts completed WGs via an atomic increment"
        )

    def test_only_last_wg_resets(self):
        # SCBranchSCC0 guards (a) wave-0-only and (b) last-WG-only.
        items = self._emit()
        assert sum(isinstance(i, SCBranchSCC0) for i in items) >= 2


# ===========================================================================
# 3b. Absence-by-toggle: the helpers are only reached behind the
#     ``kernel["StreamKWorkStealing"]`` gate at every callsite. Verified
#     against the real source so "off" provably emits nothing extra.
# ===========================================================================
class TestCallsitesAreToggleGated:
    def test_sk4_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKDynamic.graWorkGroup)
        allcalls = _all_ws_calls(StreamKDynamic.graWorkGroup)
        assert {"streamKWorkStealingHomeNoReset", "streamKWorkStealingSteal"} <= guarded
        # Nothing slips through ungated.
        assert allcalls == guarded

    def test_sk4_kernelend_reset_is_gated(self):
        guarded = _ws_guarded_calls(StreamKDynamic.kernelEnd)
        allcalls = _all_ws_calls(StreamKDynamic.kernelEnd)
        assert "streamKWorkStealingKernelEndReset" in guarded
        assert allcalls == guarded

    def test_sk5_grawg_steal_calls_are_all_gated(self):
        guarded = _ws_guarded_calls(StreamKHybrid.graWorkGroup)
        allcalls = _all_ws_calls(StreamKHybrid.graWorkGroup)
        assert {"streamKWorkStealingHomeNoReset", "streamKWorkStealingSteal"} <= guarded
        assert allcalls == guarded

    def test_sk5_kernelend_reset_is_gated(self):
        guarded = _ws_guarded_calls(StreamKHybrid.kernelEnd)
        allcalls = _all_ws_calls(StreamKHybrid.kernelEnd)
        assert "streamKWorkStealingKernelEndReset" in guarded
        assert allcalls == guarded


# ===========================================================================
# 4. Solution validation: the real rejection logic from
#    assignDerivedParameters, executed in isolation.
# ===========================================================================
class TestSolutionValidation:
    def setup_method(self):
        self.validate = _extract_ws_validation()

    def _run(self, *, streamk, atomic, work_stealing=1):
        state = {
            "StreamKWorkStealing": work_stealing,
            "StreamK": streamk,
            "StreamKAtomic": atomic,
        }
        self.validate(state, False, reject)
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

    def test_off_toggle_is_inert_even_for_bad_combo(self):
        # With the toggle off the guard must not fire, even for a combination
        # that would otherwise be rejected.
        state = self._run(streamk=3, atomic=1, work_stealing=0)
        assert "Valid" not in state
