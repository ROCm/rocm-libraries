# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail fast once overflowedResources is set: the code sites that reject a kernel (SGPR/VGPR
overflow, LDS scheduling, LDS size, ...) stop code generation right there with the error the
end-of-codegen check gives (getSourceFileString error -2), print the reason under
PrintSolutionRejectionReason, and only record the code under ForceGenerateKernel."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import Tensile
from Tensile.Common import DebugConfig
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

MAX_VGPR = 256
MAX_SGPR = 102


class _Body:
    def __init__(self):
        self.items = []

    def add(self, item, pos=None):
        self.items.append(item)


def _writer(vgprs=128, forceGenerateKernel=False, printSolutionRejectionReason=False):
    kw = object.__new__(KernelWriterAssembly)
    kw.debugConfig = DebugConfig(
        forceGenerateKernel=forceGenerateKernel,
        printSolutionRejectionReason=printSolutionRejectionReason,
    )
    kw.states = SimpleNamespace(
        regCaps={"MaxVgpr": MAX_VGPR, "MaxSgpr": MAX_SGPR, "PhysicalMaxVgprCU": 512 * 256},
        asmCaps={"HasWMMA": False},
        overflowedResources=0,
        invalidLSUCode=False,
        doubleVgpr=False,
        kernelName="KernelUnderTest",
        kernel={},
    )
    kw.vgprPool = SimpleNamespace(size=lambda: vgprs, checkFinalState=lambda: None)
    kw.agprPool = SimpleNamespace(size=lambda: 0)
    kw.sgprPool = SimpleNamespace(size=lambda: 80)
    return kw


def test_setting_a_code_stops_generation():
    kw = _writer()
    with pytest.raises(RuntimeError, match="resulted in error 5$"):
        kw.setOverflowedResources({}, 5)
    assert kw.states.overflowedResources == 5


def test_force_generate_kernel_records_the_code_and_continues():
    kw = _writer(forceGenerateKernel=True)
    kw.setOverflowedResources({}, 5)
    assert kw.states.overflowedResources == 5


def test_stop_without_a_code_continues():
    kw = _writer()
    kw.stopOnOverflowedResources({})
    assert kw.states.overflowedResources == 0


@pytest.mark.parametrize("printReason", [False, True])
def test_reason_printed_only_when_requested(capsys, printReason):
    kw = _writer(printSolutionRejectionReason=printReason)
    with pytest.raises(RuntimeError):
        kw.setOverflowedResources({}, 2)
    out = capsys.readouterr().out
    if printReason:
        assert (
            'KernelUnderTest overflowed resources.  errorCode=2, msg="too many sgprs", '
            "vgprs=128, sgprs=80" in out
        )
    else:
        assert "overflowed resources" not in out


def _initKernelRejecting(kw):
    """_initKernel stand-in that records a rejection the way applyLdsLayout does (code 8)."""

    def _initKernel(kernel, tPA, tPB):
        kw.states.overflowedResources = 8

    return _initKernel


def test_get_source_stops_after_an_init_kernel_rejection(capsys):
    """A code _initKernel records stops generation before anything past it runs."""
    kw = _writer(printSolutionRejectionReason=True)
    kw._initKernel = _initKernelRejecting(kw)

    def _notReached(kernel):
        raise AssertionError("code generation continued past the recorded rejection")

    kw._registerKernelArgs = _notReached
    assert kw.getSourceFileString({"KernelLanguage": "Assembly"}) == (-2, "")
    assert "errorCode=8" in capsys.readouterr().out


def test_get_source_continues_after_an_init_kernel_rejection_with_force_generate_kernel():
    class _Reached(Exception):
        pass

    kw = _writer(forceGenerateKernel=True)
    kw._initKernel = _initKernelRejecting(kw)

    def _reached(kernel):
        raise _Reached()

    kw._registerKernelArgs = _reached
    with pytest.raises(_Reached):
        kw.getSourceFileString({"KernelLanguage": "Assembly"})


def _sgprWriter(**flags):
    kw = _writer(**flags)
    kw.db = {"AssertOnSgprOverflow": False}
    checkedIn = []
    kw.sgprPool = SimpleNamespace(
        size=lambda: MAX_SGPR,
        checkOutAligned=lambda num, alignment, tag, preventOverflow: MAX_SGPR - 1,
        checkIn=checkedIn.append,
    )
    return kw, checkedIn


def test_sgpr_overflow_stops_at_the_allocation():
    kw, checkedIn = _sgprWriter()
    with pytest.raises(RuntimeError, match="resulted in error 2$"):
        with kw.allocTmpSgpr(2):
            raise AssertionError("the overflowing temporary SGPR was handed out")
    assert kw.states.overflowedResources == 2
    assert checkedIn == [MAX_SGPR - 1]


def test_sgpr_overflow_continues_with_force_generate_kernel():
    kw, checkedIn = _sgprWriter(forceGenerateKernel=True)
    with kw.allocTmpSgpr(2) as reg:
        assert reg.idx == MAX_SGPR - 1
    assert kw.states.overflowedResources == 2
    assert checkedIn == [MAX_SGPR - 1]


def _checkResources(kw, nextFreeVgpr):
    kw.getOccupancy = lambda *args: 2
    kw.getLdsSize = lambda kernel: 0
    body = _Body()
    mkb = SimpleNamespace(
        setGprs=lambda **counts: None, getNextFreeVgpr=lambda: nextFreeVgpr, body=body
    )
    kw.checkResources({"NumThreads": 256, "_ScheduleIterAlg": 3}, mkb)
    return body.items


def test_check_resources_stops_on_an_overflow():
    kw = _writer(vgprs=MAX_VGPR + 1)
    with pytest.raises(RuntimeError, match="resulted in error 1$"):
        _checkResources(kw, MAX_VGPR + 1)
    assert kw.states.overflowedResources == 1


def test_check_resources_ends_the_kernel_with_force_generate_kernel():
    kw = _writer(vgprs=MAX_VGPR + 1, forceGenerateKernel=True)
    items = _checkResources(kw, MAX_VGPR + 1)
    assert kw.states.overflowedResources == 1
    assert [type(item).__name__ for item in items] == ["ValueEndif", "SEndpgm", "ValueIf"]


def test_check_resources_accepts_a_kernel_that_fits():
    kw = _writer()
    assert _checkResources(kw, 128) == []
    assert kw.states.overflowedResources == 0


# The only functions allowed to assign overflowedResources directly, keyed by package-relative
# file and enclosing function. Every other site must call setOverflowedResources, so that it stops
# generation; a plain assignment would silently lose the fail-fast for that code.
_DIRECT_ASSIGNMENT_ALLOWLIST = {
    # The central setter.
    ("KernelWriter.py", "KernelWriter.setOverflowedResources"),
    # Set, re-raise under AssertOnSgprOverflow, then stopOnOverflowedResources.
    ("KernelWriterAssembly.py", "KernelWriterAssembly.allocTmpSgpr.overflowListener"),
    ("KernelWriterAssembly.py", "KernelWriterAssembly.allocTmpSgprList.overflowListener"),
    # Evaluates all its checks (a later one may replace the code), then stops.
    ("KernelWriterAssembly.py", "KernelWriterAssembly.checkResources"),
    # Runs inside _initKernel, which ductile calls alone; _getKernelSource stops after it.
    ("Components/Subtile/SubtileLdsLayout.py", "applyLdsLayout"),
}


def _directAssignments():
    root = Path(Tensile.__file__).parent
    found = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] == "Tests":
            continue

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, scope + [child.name])
                    continue
                if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        for sub in ast.walk(target):
                            if isinstance(sub, ast.Attribute) and sub.attr == "overflowedResources":
                                found.add((rel.as_posix(), ".".join(scope)))
                visit(child, scope)

        visit(ast.parse(path.read_text(), str(path)), [])
    return found


def test_overflowed_resources_is_set_only_through_the_stopping_setter():
    found = _directAssignments()
    assert found - _DIRECT_ASSIGNMENT_ALLOWLIST == set(), (
        "assign overflowedResources through KernelWriter.setOverflowedResources so the kernel "
        "stops at the rejection"
    )
    assert _DIRECT_ASSIGNMENT_ALLOWLIST - found == set(), "stale allowlist entry"
