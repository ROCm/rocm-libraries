# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-validation guard for PrefetchGL2 on Stream-K (gfx1250).

AIHPBLAS-4142: the PrefetchGL2 validator in
``Tensile/SolutionStructs/Solution.py`` was broadened so that PrefetchGL2 is
compatible with DP-first Stream-K (``StreamK==3``) -- both the DP-only
(``StreamKForceDPOnly=1``) and non-DP-only (``StreamKForceDPOnly=0``) variants --
while every other non-zero Stream-K mode (1, 2, 4, 5) stays rejected. The old
guard rejected PrefetchGL2 for *any* non-zero Stream-K:

    if state["StreamK"] != 0:
        reject(..., "PrefetchGL2 does not support Stream-K")

and it now reads:

    if state["StreamK"] != 0 and state["StreamK"] != 3:
        reject(..., "PrefetchGL2 only supports DP-first (StreamK==3) Stream-K")

These tests pin that behaviour end-to-end. The harness mirrors
``test_halfplr_streamk_rejects``: real gfx1250 capability maps from
``makeIsaInfoMap`` (needs ``amdclang++``; skipped if the toolchain cannot target
gfx1250) plus a real assembler feed ``Solution.__init__``, which runs
``assignDerivedParameters`` (and hence the PrefetchGL2 validator) end-to-end. The
reject reason is captured from stdout via ``capsys``.

The base solution is the known-good MXFP4 (F4/F4/S) TN DP-first Stream-K +
PrefetchGL2 candidate from ``sk_mxf4gemm_tdm_pap_prefetchgl2.yaml`` /
``sk_mxf4gemm_pap_prefetchgl2_nodponly.yaml``; each test flips only StreamK /
StreamKForceDPOnly / PrefetchAcrossPersistent / GlobalSplitU. The negative cases
use PrefetchAcrossPersistent=0 so the PAP "requires StreamK=3" guard does not
reject first -- the PrefetchGL2 Stream-K guard must be the deciding criterion.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Exact diagnostic emitted by the (broadened) PrefetchGL2 Stream-K guard, and the
# pre-broadening message. The positive tests assert *neither* fires (the old
# message would fire for StreamK==3 under the pre-AIHPBLAS-4142 guard, so this is
# what makes the tests catch a regression); the negative tests assert the new
# message fires.
GUARD_REASON = "PrefetchGL2 only supports DP-first (StreamK==3) Stream-K"
OLD_GUARD_REASON = "PrefetchGL2 does not support Stream-K"


# Snapshot the pristine process-global defaultSolution at import time (collection
# runs before any test executes). Sibling unit tests mutate it in place, which
# makes Solution.__init__'s `for key in defaultSolution` loop overwrite derived
# objects and break Solution construction in an order-dependent way.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# ---------------------------------------------------------------------------
# Module-scoped toolchain fixtures (real gfx1250 caps + assembler).
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
    if not iim[isa].asmCaps["HasGlobalPrefetch"]:
        pytest.skip("gfx1250 caps report no HasGlobalPrefetch; PrefetchGL2 unavailable")
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
    """Assign process-global parameters for gfx1250; restore after module."""
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


# ---------------------------------------------------------------------------
# Base solution: known-good MXFP4 TN DP-first Stream-K + PrefetchGL2 (from
# sk_mxf4gemm_tdm_pap_prefetchgl2.yaml). Each test flips exactly one knob.
# ---------------------------------------------------------------------------
def _make_params(gfx1250_iim, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    # [M, N, K, B, ?, MIWaveTile0, MIWaveTile1, WaveGroup0, WaveGroup1]
    mi = [16, 16, 128, 1, 1, 2, 2, 2, 2]
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F4",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,   # TN
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,   # PrefetchGL2 rejects general (non-strided) batch.
        "MXBlockA": 32,
        "MXBlockB": 32,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 0,             # PrefetchGL2 rejects GSU > 1 / GSU == -1.
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "TDMInst": 3,
        "LDSTrInst": False,
        "StreamK": 3,
        "StreamKForceDPOnly": 0,
        "PrefetchAcrossPersistent": 1,
        "PrefetchGL2": 1,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "ClusterLocalRead": 0,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim
    )
    params.update(mi_params)
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(gfx1250_iim, **overrides)
    # printSolutionRejectionReason=True so reject() writes the reason to stdout.
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    out = capsys.readouterr().out
    return sol, out


# ---------------------------------------------------------------------------
# Positive: PrefetchGL2 + DP-first Stream-K (StreamK==3) is ACCEPTED, for both
# StreamKForceDPOnly variants. Under the pre-AIHPBLAS-4142 guard these were
# rejected ("PrefetchGL2 does not support Stream-K"), so each assertion catches
# a regression to the old behaviour.
# ---------------------------------------------------------------------------
def test_prefetchgl2_streamk3_dp_only_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """PrefetchGL2 + StreamK==3 + StreamKForceDPOnly==1 + PAP=1 is valid.

    Mirrors sk_mxf4gemm_tdm_pap_prefetchgl2.yaml.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, StreamKForceDPOnly=1)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert GUARD_REASON not in out
    assert OLD_GUARD_REASON not in out


def test_prefetchgl2_streamk3_non_dp_only_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """PrefetchGL2 + StreamK==3 + StreamKForceDPOnly==0 + PAP=1 is valid.

    Mirrors sk_mxf4gemm_pap_prefetchgl2_nodponly.yaml.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, StreamKForceDPOnly=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert GUARD_REASON not in out
    assert OLD_GUARD_REASON not in out


# ---------------------------------------------------------------------------
# Baseline (unchanged by AIHPBLAS-4142): PrefetchGL2 + StreamK==0 is still
# allowed. GlobalSplitU=1 satisfies the "GSU or StreamK must be enabled" gate
# without tripping the PrefetchGL2 GSU>1 guard.
# ---------------------------------------------------------------------------
def test_prefetchgl2_streamk0_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(
        gfx1250_iim, assembler, capsys,
        StreamK=0, GlobalSplitU=1, PrefetchAcrossPersistent=0,
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert GUARD_REASON not in out
    assert OLD_GUARD_REASON not in out


# ---------------------------------------------------------------------------
# Negative: PrefetchGL2 + any non-DP-first non-zero Stream-K (1, 2, 4, 5) is
# still rejected by the PrefetchGL2 Stream-K guard. PrefetchAcrossPersistent=0 so
# the PAP "requires StreamK=3" guard does not reject first, making the PrefetchGL2
# Stream-K guard the deciding criterion. A future over-broadening of the guard
# regresses these.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("streamk", [1, 2, 4, 5])
def test_prefetchgl2_rejects_non_dpfirst_streamk(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, streamk
):
    sol, out = _derive(
        gfx1250_iim, assembler, capsys,
        StreamK=streamk, PrefetchAcrossPersistent=0,
    )
    assert sol.get("Valid") is False
    assert GUARD_REASON in out
