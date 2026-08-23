################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""TDMWaveSpread=1, the 2/2/4 wave overlay, and why it is refused.

tdmWavePartition describes the overlay correctly -- A and B on every wave, the MX
scales still on the parity pair -- and every site that derives *addresses* from it
is right: the de-aliased descriptors are built on all four waves, the per-component
tile extent and LDS offset follow numComp, and the decoupled LDS swap runs
unguarded because A and B own separate descriptors.

The dispatch does not follow. KernelWriterAssembly._emitTdmDealiasedIssue gates
each fill on bit0 of WaveIdx, so at numComp == numWaves the odd waves' A
components and the even waves' B components are addressed and never transferred:
half of each tile reaches LDS unwritten and the kernel returns wrong results
instead of failing to build. The single parity MulticastMask has the same shape,
and is reachable whenever ClusterDim is not [1, 1].

So the refusal is about the issue path, not about the overlay. These tests pin the
partition that is right, the guards in the order they fire, and that
TDMWaveSpread=0 is untouched.
"""
import copy

import pytest

from Tensile.Common.DecouplePgr import tdmWavePartition
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


def _ks(spread=1, fuse=6, numWaves=4, **overrides):
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "TDMWaveSpread": spread,
        "NumWaves": numWaves,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
        "enableTDMA": True,
        "enableTDMB": True,
    }
    ks.update(overrides)
    return ks


# ---------------------------------------------------------------------------
# The overlay tdmWavePartition describes.
# ---------------------------------------------------------------------------
def test_data_tensors_ride_every_wave_and_the_scales_keep_the_pair():
    assert tdmWavePartition(_ks(), "A") == (4, (0, 1, 2, 3))
    assert tdmWavePartition(_ks(), "B") == (4, (0, 1, 2, 3))
    assert tdmWavePartition(_ks(), "MXSA") == (2, (0, 2))
    assert tdmWavePartition(_ks(), "MXSB") == (2, (1, 3))


def test_the_overlay_is_what_the_issue_gate_cannot_express():
    """The reject exists because these two answers disagree.

    A parity gate admits half the waves. At TDMWaveSpread=0 that is exactly the
    set that carries A, so the gate is right by coincidence; at 1 it is half of
    it, and the component ids the gate drops are the ones already addressed.
    """
    _, spreadWaves = tdmWavePartition(_ks(spread=1), "A")
    _, parityWaves = tdmWavePartition(_ks(spread=0), "A")
    assert parityWaves == (0, 2)
    assert set(parityWaves) < set(spreadWaves)


@pytest.mark.parametrize("numWaves", [2, 4, 8])
def test_zero_keeps_the_two_way_split_at_every_wave_count(numWaves):
    numComp, waves = tdmWavePartition(_ks(spread=0, numWaves=numWaves), "A")
    assert numComp == numWaves // 2
    assert waves == tuple(w for w in range(numWaves) if w % 2 == 0)


# ---------------------------------------------------------------------------
# Solution-level guards, on a real derived gfx1250 solution.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    """The F8F4 TN MT64x512 DepthU256 hero pair, the only shape 1 can reach."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = overrides.pop("MatrixInstruction", [16, 16, 128, 1, 1, 2, 16, 2, 2])
    workGroup = overrides.pop("WorkGroup", [32, 4, 1])
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F4",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    problemType.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": workGroup,
        "WavefrontSize": 32,
        "DepthU": 256,
        "MaxLDS": 327680,
        "KernelLanguage": "Assembly",
        "TDMInst": 3,
        "MXScaleFormat": "InMemorySwizzle",
        "LDSTrInst": True,
        "TDMFuse": 6,
        "TDMWaveSpread": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "LdsPadMetadata": 0,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"], gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


# The control comes first: every reject below is vacuous if the shape it starts
# from is itself rejected.
@pytest.mark.parametrize("pgrA, pgrB, label", [(1, 2, "hero"), (2, 1, "mirror")])
def test_zero_is_accepted_at_both_divergent_pairs(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMWaveSpread=0,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


@pytest.mark.parametrize("pgrA, pgrB, label", [(1, 2, "hero"), (2, 1, "mirror")])
def test_rejected_as_unimplemented_at_both_divergent_pairs(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    """The shape that generated and validated as wrong output before this reject."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is False, label
    assert "TDMWaveSpread=1 is not implemented" in out
    assert "addressed and never transferred" in out


def test_rejects_without_the_dealiased_grouping(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """A more specific objection than the unimplemented one, so it must win.

    With A and B sharing one descriptor set the parity pair fixes A to the even
    waves whatever numComp says, so the overlay cannot be expressed at all --
    a different statement from the issue gate not honouring it.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0)
    assert sol.get("Valid") is False
    assert "requires the de-aliased" in out
    assert "is not implemented" not in out


def test_rejects_one_wave(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Refused, but by the grouping this row depends on rather than by this row.

    TDMWaveSpread=1 needs TDMFuse=6, and TDMFuse=6 declines one wave first, so
    this row's own NumWaves message is unreachable from this direction rather
    than dead -- the predicate half is covered by
    test_zero_keeps_the_two_way_split_at_every_wave_count. Assert the refusal
    that is actually produced.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       MatrixInstruction=[16, 16, 128, 1, 1, 2, 16, 1, 1],
                       WorkGroup=[32, 1, 1])
    assert sol.get("Valid") is False
    assert "TDMFuse=6 de-aliases A from B on the wave-separated path" in out


def test_rejects_without_tdm_on_both_tensors(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMInst=1)
    assert sol.get("Valid") is False
    assert "TDMA and TDMB must be enabled simultaneously" in out
