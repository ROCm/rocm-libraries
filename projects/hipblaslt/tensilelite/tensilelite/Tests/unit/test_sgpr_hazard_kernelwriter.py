# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""KernelWriter runs the gfx12 SGPR read-hazard pass exactly when the arch has the hazard."""

from types import SimpleNamespace

import pytest

from tensilelite.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit

# s4 is read by a VALU, rewritten by an SALU and read again: one sa_sdst wait on gfx1200/gfx1201.
HAZARD = "v_mov_b32 v1, s4\ns_mov_b32 s4, 1\nv_mov_b32 v2, s4\ns_endpgm\n"


def _writer(cap):
    # The parts of a KernelWriter that _getKernelSource reads.
    return SimpleNamespace(
        _initKernel=lambda kernel, tPA, tPB: None,
        _registerKernelArgs=lambda kernel: None,
        states=SimpleNamespace(kernelName="test_kernel", archCaps={"HasVALUReadSGPRHazard": cap}),
        kernelArgDefs=[],
        stringIdx=0,
        kernelBody=lambda kernel, tPA, tPB: (0, HAZARD),
        kernelBodySubtile=lambda kernel, tPA, tPB: (0, HAZARD),
        debugConfig=SimpleNamespace(forceGenerateKernel=False),
    )


def _kernel():
    return {
        "UseSubtileImpl": False,
        "TileProcessingStrategy": "None",
        "MacroTile0": 128,
        "MacroTile1": 128,
        "DepthU": 32,
        "NumThreads": 256,
    }


@pytest.mark.parametrize("cap, numWaits", [(1, 1), (0, 0)])
def test_pass_runs_only_with_the_cap(cap, numWaits):
    source = KernelWriter._getKernelSource(_writer(cap), _kernel())
    assert source.count("s_wait_alu depctr_sa_sdst(0)") == numWaits


def test_missing_cap_fails_loudly():
    # A rocisa extension built before the cap existed must not silently skip the mitigation.
    writer = _writer(1)
    writer.states.archCaps = {}
    with pytest.raises(KeyError):
        KernelWriter._getKernelSource(writer, _kernel())
