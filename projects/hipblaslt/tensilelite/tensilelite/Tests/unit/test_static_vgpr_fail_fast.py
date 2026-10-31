# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail fast on a static VGPR overflow: once _initKernel has sized the VGPR pool above MaxVgpr,
code generation stops with the same error the end-of-codegen resource check gives (overflowed
resources code 1, getSourceFileString error -2), unless ForceGenerateKernel is set."""

from types import SimpleNamespace

import pytest

from Tensile.Common import DebugConfig
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

MAX_VGPR = 256


def _writer(vgprs, forceGenerateKernel=False, printSolutionRejectionReason=False):
    kw = object.__new__(KernelWriterAssembly)
    kw.debugConfig = DebugConfig(
        forceGenerateKernel=forceGenerateKernel,
        printSolutionRejectionReason=printSolutionRejectionReason,
    )
    kw.states = SimpleNamespace(
        regCaps={"MaxVgpr": MAX_VGPR}, overflowedResources=0, kernelName="KernelUnderTest"
    )
    kw.vgprPool = SimpleNamespace(size=lambda: vgprs)
    kw.sgprPool = SimpleNamespace(size=lambda: 80)
    return kw


@pytest.mark.parametrize("vgprs", [MAX_VGPR + 1, 2 * MAX_VGPR])
def test_static_overflow_rejects(vgprs):
    kw = _writer(vgprs)
    with pytest.raises(RuntimeError, match="resulted in error 1"):
        kw._rejectStaticVgprOverflow({})
    assert kw.states.overflowedResources == 1


@pytest.mark.parametrize("vgprs", [1, MAX_VGPR])
def test_within_max_vgpr_continues(vgprs):
    kw = _writer(vgprs)
    kw._rejectStaticVgprOverflow({})
    assert kw.states.overflowedResources == 0


def test_force_generate_kernel_continues():
    kw = _writer(MAX_VGPR + 1, forceGenerateKernel=True)
    kw._rejectStaticVgprOverflow({})
    assert kw.states.overflowedResources == 0


@pytest.mark.parametrize("printReason", [False, True])
def test_reason_printed_only_when_requested(capsys, printReason):
    kw = _writer(MAX_VGPR + 8, printSolutionRejectionReason=printReason)
    with pytest.raises(RuntimeError):
        kw._rejectStaticVgprOverflow({})
    out = capsys.readouterr().out
    if printReason:
        assert (
            'KernelUnderTest overflowed resources.  errorCode=1, msg="too many vgprs", '
            "vgprs=264, sgprs=80" in out
        )
    else:
        assert "overflowed resources" not in out


def test_get_source_stops_after_init_kernel():
    """getSourceFileString returns error -2 right after _initKernel: nothing past it runs."""
    kw = _writer(MAX_VGPR + 1)
    kw._initKernel = lambda kernel, tPA, tPB: None

    def _notReached(kernel):
        raise AssertionError("code generation continued past the static VGPR overflow")

    kw._registerKernelArgs = _notReached
    assert kw.getSourceFileString({"KernelLanguage": "Assembly"}) == (-2, "")
    assert kw.states.overflowedResources == 1


def test_get_source_continues_with_force_generate_kernel():
    """With ForceGenerateKernel the overflowing kernel is generated past _initKernel."""

    class _Reached(Exception):
        pass

    kw = _writer(MAX_VGPR + 1, forceGenerateKernel=True)
    kw._initKernel = lambda kernel, tPA, tPB: None

    def _reached(kernel):
        raise _Reached()

    kw._registerKernelArgs = _reached
    with pytest.raises(_Reached):
        kw.getSourceFileString({"KernelLanguage": "Assembly"})
