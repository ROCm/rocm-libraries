# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU-free tests for the register-budget verdict and its deferral to allocation.

``checkResources`` has to judge a kernel before StinkyTofu's register allocator
has touched it, so the pool sizes it judges are a proposal rather than what the
kernel will declare.  Three pieces settle that, and they are what is tested here:

``judgeRegisterBudget``
    Records a count triple on the kernel body, recomputes ``CUOccupancy`` from
    it, and reaches the verdict the triple implies.  One function, two callers.

``deferRegisterOverflowForAllocation``
    Under force apply (``StinkyTofuRegisterAllocation`` 3) reopens that verdict,
    whichever register class it came from, so the kernel reaches allocation.

``resolveDeferredRegisterOverflow``
    Runs the judgement again on the counts the emitted descriptor declares,
    which both settles the deferral and keeps the recorded counts honest when a
    colouring merely moved them.
"""

import ast
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

from tensilelite.KernelWriterAssembly import (
    REGISTER_BUDGET_OVERFLOW_CODES,
    KernelWriterAssembly,
)

_KERNEL_WRITER_ASM_PY = Path(__file__).resolve().parents[2] / "KernelWriterAssembly.py"
_KERNEL_WRITER_PY = Path(__file__).resolve().parents[2] / "KernelWriter.py"

# gfx1250, the one architecture StinkyTofu has a backend for.
_MAX_VGPR = 1024
_MAX_SGPR = 106


class _Pool:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _Mkb:
    """Stands in for rocisa's KernelBody, for the architectures tested here.

    ``getNextFreeVgpr`` has to fold AGPRs exactly the way
    ``SignatureKernelDescriptor::setGprs`` does, because the occupancy-limit
    verdict divides by it: on a target whose AGPRs share the register file the
    descriptor declares ``accum_offset + agprs``, not the VGPR count.  Every ISA
    from gfx10 on takes that branch as soon as an AGPR is present, and the tests
    here are gfx1250.
    """

    def __init__(self):
        self.totalVgprs = 0
        self.totalAgprs = 0
        self.totalSgprs = 0

    def setGprs(self, totalVgprs, totalAgprs, totalSgprs):
        self.totalVgprs = totalVgprs
        self.totalAgprs = totalAgprs
        self.totalSgprs = totalSgprs

    def getNextFreeVgpr(self):
        if self.totalAgprs:
            return int(ceil(self.totalVgprs / 8.0)) * 8 + self.totalAgprs
        return self.totalVgprs

    def getNextFreeSgpr(self):
        return self.totalSgprs


class _RecordingBody:
    """``mkb.body``, enough of it for applyOverflowStub to write into."""

    def __init__(self):
        self.items = []

    def add(self, item, *args):
        self.items.append(item)


class _StubbableMkb(_Mkb):
    def __init__(self):
        super().__init__()
        self.body = _RecordingBody()


def _writer(vgprs=256, agprs=0, sgprs=64, standing=0):
    kw = object.__new__(KernelWriterAssembly)
    kw.states = SimpleNamespace(
        version=(12, 5, 0),
        kernelName="test_kernel",
        overflowedResources=standing,
        overflowPendingRA=0,
        invalidLSUCode=False,
        doubleVgpr=False,
        # An over-budget PLSIN kernel exits loudly rather than being stubbed, so
        # judgeRegisterBudget reads this. Off here: these are gfx1250 kernels.
        postLoopStoreInNll=False,
        regCaps={"MaxVgpr": _MAX_VGPR, "MaxSgpr": _MAX_SGPR,
                 "PhysicalMaxSgpr": 800, "PhysicalMaxVgprCU": 4096 * 32},
        archCaps={"MaxWavesPerSimd": 10, "DeviceLDS": 327680, "LdsGranularity": 1024},
        asmCaps={"HasWMMA": True},
    )
    kw.vgprPool = _Pool(vgprs)
    kw.agprPool = _Pool(agprs)
    kw.sgprPool = _Pool(sgprs)
    kw.debugConfig = SimpleNamespace(printSolutionRejectionReason=False)
    return kw


def _kernel(**overrides):
    kernel = {
        "NumThreads": 256,
        "LdsNumBytes": 0,
        "CUOccupancy": -1,
        "_ScheduleIterAlg": 0,
        "_StinkyTofuOptLevel": 3,
        "StinkyTofuRegisterAllocation": 3,
    }
    kernel.update(overrides)
    return kernel


def _judge(writer, kernel, vgprs, agprs, sgprs):
    mkb = _Mkb()
    writer.judgeRegisterBudget(kernel, mkb, vgprs, agprs, sgprs)
    return mkb


# ── judgeRegisterBudget ───────────────────────────────────────────────────────

class TestJudgeRegisterBudget:
    def test_records_the_counts_it_judged(self):
        mkb = _judge(_writer(), _kernel(), 300, 0, 80)
        assert (mkb.totalVgprs, mkb.totalAgprs, mkb.totalSgprs) == (300, 0, 80)

    def test_occupancy_follows_the_counts(self):
        kernel = _kernel()
        _judge(_writer(), kernel, _MAX_VGPR // 4, 0, 64)
        assert kernel["CUOccupancy"] == 4

        _judge(_writer(), kernel, _MAX_VGPR, 0, 64)
        assert kernel["CUOccupancy"] == 1, (
            "occupancy must be recomputed from the counts handed in, not kept "
            "from the previous judgement"
        )

    @pytest.mark.parametrize("vgprs,sgprs", [
        (_MAX_VGPR + 1, 64),                # over the VGPR cap
        (256, _MAX_SGPR + 1),               # over the SGPR cap
    ])
    def test_an_over_budget_count_reaches_a_deferrable_verdict(self, vgprs, sgprs):
        # Which code: a gfx1250 workgroup of 256 threads is allowed exactly
        # MaxVgpr registers each, so a VGPR count over the cap trips the
        # occupancy limit as well and is reported as that.  Both are budget
        # verdicts, so the deferral treats them alike.
        writer = _writer()
        _judge(writer, _kernel(), vgprs, 0, sgprs)
        assert writer.states.overflowedResources in REGISTER_BUDGET_OVERFLOW_CODES

    def test_too_many_sgprs_is_named_as_such(self):
        writer = _writer()
        _judge(writer, _kernel(), 256, 0, _MAX_SGPR + 1)
        assert writer.states.overflowedResources == 2

    def test_a_fitting_count_reaches_no_verdict(self):
        writer = _writer()
        _judge(writer, _kernel(), 256, 0, 64)
        assert writer.states.overflowedResources == 0

    def test_occupancy_limit_is_a_budget_verdict(self):
        # 512 threads halve the per-thread VGPR budget to 512, so a kernel
        # declaring more than that cannot place even one workgroup.
        writer = _writer()
        _judge(writer, _kernel(NumThreads=512), 600, 0, 64)
        assert writer.states.overflowedResources == 4

    def test_sia2_occupancy_floor_is_a_budget_verdict(self):
        writer = _writer()
        _judge(writer, _kernel(_ScheduleIterAlg=2), _MAX_VGPR, 0, 64)
        assert writer.states.overflowedResources == 6

    def test_a_standing_verdict_is_not_overwritten(self):
        # Code 3 is reached during code generation and no colouring answers it.
        # Were the budget to overwrite it, the deferral below would reopen a
        # verdict that allocation cannot settle and the kernel would be accepted.
        writer = _writer(standing=3)
        _judge(writer, _kernel(), _MAX_VGPR + 1, 0, 64)
        assert writer.states.overflowedResources == 3

    def test_a_fitting_budget_never_clears_a_verdict(self):
        writer = _writer(standing=3)
        _judge(writer, _kernel(), 256, 0, 64)
        assert writer.states.overflowedResources == 3


# ── deferRegisterOverflowForAllocation ────────────────────────────────────────

class TestDeferral:
    @pytest.mark.parametrize("code", sorted(REGISTER_BUDGET_OVERFLOW_CODES))
    def test_every_budget_verdict_is_deferred_under_force_apply(self, code):
        writer = _writer(standing=code)
        writer.deferRegisterOverflowForAllocation(_kernel())
        assert writer.states.overflowedResources == 0
        assert writer.states.overflowPendingRA == code

    def test_vgpr_and_sgpr_are_both_covered(self):
        assert {1, 2} <= REGISTER_BUDGET_OVERFLOW_CODES, (
            "both register classes must be deferrable; allocation colours both"
        )

    @pytest.mark.parametrize("code", [3, 5, 7, 8, 9])
    def test_a_non_budget_verdict_is_never_deferred(self, code):
        writer = _writer(standing=code)
        writer.deferRegisterOverflowForAllocation(_kernel())
        assert writer.states.overflowedResources == code
        assert writer.states.overflowPendingRA == 0

    @pytest.mark.parametrize("mode", [0, 1, 2])
    def test_only_force_apply_defers(self, mode):
        writer = _writer(standing=1)
        writer.deferRegisterOverflowForAllocation(
            _kernel(StinkyTofuRegisterAllocation=mode))
        assert writer.states.overflowedResources == 1
        assert writer.states.overflowPendingRA == 0

    @pytest.mark.parametrize("value,defers", [("force,classes=vs", True),
                                              ("apply,classes=vs", False)])
    def test_a_config_string_defers_by_the_mode_it_starts_with(self, value, defers):
        writer = _writer(standing=1)
        writer.deferRegisterOverflowForAllocation(
            _kernel(StinkyTofuRegisterAllocation=value))
        assert (writer.states.overflowPendingRA == 1) is defers

    def test_no_stinkytofu_path_means_no_deferral(self):
        # Nothing would resolve it: the mode is gated on the same two questions
        # the conversion gate asks.
        writer = _writer(standing=1)
        writer.deferRegisterOverflowForAllocation(_kernel(_StinkyTofuOptLevel=None))
        assert writer.states.overflowedResources == 1
        assert writer.states.overflowPendingRA == 0


# ── resolveDeferredRegisterOverflow ───────────────────────────────────────────

class TestResolution:
    def _resolve(self, writer, kernel, declaredVgprs, declaredSgprs):
        mkb = _Mkb()
        error = writer.resolveDeferredRegisterOverflow(
            kernel, mkb, declaredVgprs, declaredSgprs)
        return error, mkb

    def test_a_deferred_vgpr_overflow_allocation_fixes_is_accepted(self):
        writer = _writer(vgprs=_MAX_VGPR + 200)
        writer.states.overflowPendingRA = 1
        error, mkb = self._resolve(writer, _kernel(), _MAX_VGPR - 24, 64)
        assert error == 0
        assert mkb.totalVgprs == _MAX_VGPR - 24
        assert writer.states.overflowPendingRA == 0

    def test_a_deferred_vgpr_overflow_allocation_cannot_fix_is_reinstated(self):
        writer = _writer(vgprs=_MAX_VGPR + 200)
        writer.states.overflowPendingRA = 1
        error, _ = self._resolve(writer, _kernel(), _MAX_VGPR + 8, 64)
        assert error in REGISTER_BUDGET_OVERFLOW_CODES
        assert writer.states.overflowedResources == error

    def test_a_deferred_sgpr_overflow_allocation_fixes_is_accepted(self):
        writer = _writer(sgprs=_MAX_SGPR + 4)
        writer.states.overflowPendingRA = 2
        error, mkb = self._resolve(writer, _kernel(), 256, _MAX_SGPR - 2)
        assert error == 0
        assert mkb.totalSgprs == _MAX_SGPR - 2

    def test_a_deferred_sgpr_overflow_allocation_cannot_fix_is_reinstated(self):
        writer = _writer(sgprs=_MAX_SGPR + 4)
        writer.states.overflowPendingRA = 2
        error, _ = self._resolve(writer, _kernel(), 256, _MAX_SGPR + 1)
        assert error == 2

    def test_an_unreadable_count_reinstates_the_deferred_verdict(self):
        # Falling back to the pool is what reinstates it: the kernel is judged
        # on the same estimate that was deferred in the first place.
        writer = _writer(sgprs=_MAX_SGPR + 4)
        writer.states.overflowPendingRA = 2
        error, _ = self._resolve(writer, _kernel(), 0, 0)
        assert error == 2

    def test_occupancy_is_recomputed_from_the_declared_counts(self):
        # The headline case: the pool reserved four times what the emitted code
        # declares, so the recorded occupancy was four times too pessimistic.
        kernel = _kernel()
        writer = _writer(vgprs=_MAX_VGPR)
        error, mkb = self._resolve(writer, kernel, _MAX_VGPR // 4, 64)
        assert error == 0
        assert kernel["CUOccupancy"] == 4
        assert mkb.totalVgprs == _MAX_VGPR // 4

    def test_an_undeferred_kernel_is_still_realigned(self):
        # Nothing was deferred, so there is no verdict to settle -- but the
        # counts still have to describe the emitted code.
        kernel = _kernel()
        writer = _writer(vgprs=_MAX_VGPR, sgprs=96)
        error, mkb = self._resolve(writer, kernel, 400, 90)
        assert error == 0
        assert (mkb.totalVgprs, mkb.totalSgprs) == (400, 90)

    def test_agprs_keep_the_pool_count(self):
        # .amdhsa_next_free_vgpr covers both classes where they share the file,
        # so feeding it back as a plain VGPR count would count the AGPRs twice.
        #
        # Driven directly because no architecture reaches this through codegen:
        # gfx1250 is StinkyTofu's only backend and has no AGPRs.  The guard is
        # for the first backend on an architecture that does.
        writer = _writer(vgprs=256, agprs=128)
        _, mkb = self._resolve(writer, _kernel(), 384, 64)
        assert mkb.totalVgprs == 256
        assert mkb.totalAgprs == 128


# ── The direction nothing used to check: counts that went up ──────────────────

class TestCountsThatRose:
    """Allocation and the emit passes raise a count as readily as they lower it.

    ``refreshSgprCount`` says so itself -- prefetch adds SGPRs after compaction
    -- and it applies the result with no cap of its own.  Before the budget was
    re-judged on the declared counts, a kernel whose counts rose past the cap was
    emitted anyway.  These are the cases that reach a verdict no pool estimate
    would have reached, and they do so in every mode, deferral or not.
    """

    def _resolve(self, writer, kernel, declaredVgprs, declaredSgprs):
        return writer.resolveDeferredRegisterOverflow(
            kernel, _Mkb(), declaredVgprs, declaredSgprs)

    def test_sgprs_raised_past_the_cap_are_rejected(self):
        # The pool fitted; the emitted code does not.
        writer = _writer(vgprs=256, sgprs=_MAX_SGPR - 6)
        assert self._resolve(writer, _kernel(), 256, _MAX_SGPR + 1) == 2

    def test_vgprs_raised_past_the_budget_are_rejected(self):
        writer = _writer(vgprs=256)
        error = self._resolve(writer, _kernel(NumThreads=512), 600, 64)
        assert error in REGISTER_BUDGET_OVERFLOW_CODES

    def test_a_raised_count_that_still_fits_is_accepted(self):
        # Rejection has to come from the budget, not from the count moving.
        writer = _writer(vgprs=256, sgprs=64)
        assert self._resolve(writer, _kernel(), 400, 90) == 0

    def test_sia2_occupancy_floor_is_rejudged_upward(self):
        # The pool estimate left room for two workgroups; the declared count
        # does not, and SIA2 requires two.
        writer = _writer(vgprs=_MAX_VGPR // 4)
        error = self._resolve(writer, _kernel(_ScheduleIterAlg=2), _MAX_VGPR, 64)
        assert error == 6

    def test_occupancy_is_lowered_to_match_a_raised_count(self):
        kernel = _kernel()
        writer = _writer(vgprs=_MAX_VGPR // 4)
        self._resolve(writer, kernel, _MAX_VGPR // 2, 64)
        assert kernel["CUOccupancy"] == 2


# ── Wiring: the set, the call site, the stub ──────────────────────────────────

def _funcDef(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("%s not found in %s" % (name, path.name))


class TestWiring:
    def test_every_verdict_the_judgement_assigns_is_deferrable(self):
        # REGISTER_BUDGET_OVERFLOW_CODES is maintained by hand, and a code
        # missing from it is not a failure -- it is a verdict that silently
        # stops being deferrable.  Read the codes back out of the production
        # source so the two cannot drift.
        fn = _funcDef(_KERNEL_WRITER_ASM_PY, "judgeRegisterBudget")
        assigned = {
            node.value.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "verdict" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value != 0
        }
        assert assigned, "judgeRegisterBudget assigns no verdict; the walk went stale"
        assert assigned == set(REGISTER_BUDGET_OVERFLOW_CODES), (
            "judgeRegisterBudget assigns %s but REGISTER_BUDGET_OVERFLOW_CODES holds %s; "
            "a code in one and not the other is a verdict that is judged but never "
            "deferred, or deferred but never judged"
            % (sorted(assigned), sorted(REGISTER_BUDGET_OVERFLOW_CODES))
        )

    def test_the_call_site_reports_and_stubs_a_rejected_resolution(self):
        # kernelBody must not merely drop the error on the floor: a rejected
        # kernel still gets built under ForceGenerateKernel, so the body has to
        # be replaced and the rejection announced as a post-allocation one.
        fn = _funcDef(_KERNEL_WRITER_PY, "kernelBody")
        calls = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "applyOverflowVerdict"
        ]
        assert calls, "kernelBody no longer reports the resolver's verdict"
        assert any(kw.arg == "afterAllocation" for call in calls for kw in call.keywords), (
            "the post-allocation rejection must say so, or its counts read as pool sizes"
        )

    def test_a_rejected_resolution_stubs_the_body(self):
        # The sequence kernelBody performs, end to end: judge, defer, re-judge
        # on counts allocation could not bring under the cap, report.
        writer = _writer(sgprs=_MAX_SGPR + 4)
        kernel = _kernel()
        mkb = _StubbableMkb()

        writer.judgeRegisterBudget(kernel, mkb, 256, 0, _MAX_SGPR + 4)
        assert writer.states.overflowedResources == 2
        writer.deferRegisterOverflowForAllocation(kernel)
        assert writer.states.overflowedResources == 0

        error = writer.resolveDeferredRegisterOverflow(kernel, mkb, 256, _MAX_SGPR + 2)
        assert error == 2
        writer.applyOverflowVerdict(kernel, mkb, afterAllocation=True)
        assert len(mkb.body.items) == 3, (
            "a rejected kernel must have its body replaced by the .if 0 / s_endpgm stub"
        )

    def test_an_accepted_resolution_leaves_the_body_alone(self):
        writer = _writer(sgprs=_MAX_SGPR + 4)
        kernel = _kernel()
        mkb = _StubbableMkb()

        writer.judgeRegisterBudget(kernel, mkb, 256, 0, _MAX_SGPR + 4)
        writer.deferRegisterOverflowForAllocation(kernel)
        error = writer.resolveDeferredRegisterOverflow(kernel, mkb, 256, _MAX_SGPR - 2)
        assert error == 0
        writer.applyOverflowVerdict(kernel, mkb, afterAllocation=True)
        assert mkb.body.items == []


class TestRejectionReporting:
    """The warning is the only record of why a solution vanished from a build."""

    def _report(self, afterAllocation):
        writer = _writer(vgprs=300, sgprs=_MAX_SGPR + 4)
        writer.debugConfig.printSolutionRejectionReason = True
        writer.states.overflowedResources = 2
        mkb = _StubbableMkb()
        mkb.setGprs(totalVgprs=288, totalAgprs=0, totalSgprs=_MAX_SGPR + 2)

        warnings = []
        import tensilelite.KernelWriterAssembly as kwa_module
        original = kwa_module.printWarning
        kwa_module.printWarning = warnings.append
        try:
            writer.applyOverflowVerdict(_kernel(), mkb, afterAllocation=afterAllocation)
        finally:
            kwa_module.printWarning = original
        return warnings[0]

    def test_a_pool_rejection_reports_the_pool(self):
        # Unchanged from before the budget was ever re-judged: on every
        # architecture without a StinkyTofu backend this is the only message.
        text = self._report(afterAllocation=False)
        assert "vgprs=300, sgprs=110" in text, text
        assert "after register allocation" not in text

    def test_a_post_allocation_rejection_reports_both(self):
        # Both numbers, because the question being asked is "did allocation
        # help, and by how much": 110 sgprs down to 108, still over the cap.
        text = self._report(afterAllocation=True)
        assert "after register allocation" in text, text
        assert "vgprs=288, sgprs=108" in text, text
        assert "pool estimate vgprs=300, sgprs=110" in text, text
