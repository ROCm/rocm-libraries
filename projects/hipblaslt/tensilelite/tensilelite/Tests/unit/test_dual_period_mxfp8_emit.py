################################################################################
# ISA-level regression for the dual-period MXFP8 subtile kernel.
#
# The dual-period path keeps the 128-K data loop while running a decoupled
# 256-K pre-shuffled scale epoch that is reused by two data iterations.  This
# test drives the REAL winning geometry (MT256x256, MIWT 8x4, MIWG 2x4, DU128,
# PGR2/PLR0, 4x2 partitions, N-major, WGM4) through CPU-only codegen and
# asserts the steady-state mainloop honors the scale contract:
#
#   * exactly SIX scale ds_read_b32 per 256-K epoch (4 SA groups + 2 SB groups)
#   * both K phases consumed by the MFMAs (K=0 for data iter 0, K=1 for iter 1)
#   * every scale GR / SRD-advance / LDS-swap fires exactly once per epoch
#   * architectural VGPR stays within the two-wave-slot budget (<= 128)
#
# It needs the gfx950 assembly toolchain (amdclang++); it skips otherwise.
################################################################################

import re
import shutil

import pytest


def _emit_dual_period_kernel(plr=0):
    """Build the real dual-period MXFP8 solution and return its assembly text.

    Partition-directed PLR (next-partition prefetch, n-major traversal, 2x2
    partition grid) is selected automatically from the dual-period MXFP8 solution
    parameters -- no environment knobs are required.  ``plr`` only reaches the
    solution; the schedule pins its own base, so it cannot change the assembly.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")

    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    asm = makeAssemblyToolchain(cxx, bundler, "default").assembler
    assignGlobalParameters({}, iim)

    mi = [16, 16, 128, 1, 1, 8, 4, 2, 4]
    pt = {"OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
          "ComputeDataType": "s", "HighPrecisionAccumulate": True,
          "TransposeA": True, "TransposeB": False, "UseBeta": True,
          "Batched": True, "StridedBatched": True, "MXBlockA": 32, "MXBlockB": 32}
    params = {"ProblemType": pt, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 64, "DepthU": 128,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": plr,
        "ScheduleIterAlg": 3, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0, "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
        "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "DirectToLds": 1, "StreamK": 0, "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": True, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 4}
    params.update(matrixInstructionToMIParameters(
        mi, isa, 64, pt, params["WorkGroup"], iim))
    sol = Solution(params, False, True, False, asm, iim)
    assert sol.get("Valid"), "dual-period MXFP8 solution did not validate"
    assert sol.get("_DualPeriodMXFP8"), "solution did not take the dual-period path"

    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.Common.Types import DebugConfig
    import rocisa

    kernels = generateKernelObjectsFromSolutions([sol])
    kwa = KernelWriterAssembly(asm, DebugConfig())
    for kernel in kernels:
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "dual_period_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        res = processKernelSource(
            kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, "kernel codegen failed (err=%s)" % res.err
        src = res.src
        if isinstance(src, (bytes, bytearray)):
            src = src.decode(errors="replace")
        return src
    pytest.skip("no kernel object generated")


def _mainloop_lines(src):
    """Return the steady-state mainloop body (label_LoopBeginL .. back-branch)."""
    lines = src.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and ln.strip() == "label_LoopBeginL:":
            start = i
        elif start is not None and "s_cbranch_scc0 label_LoopBeginL" in ln:
            end = i
            break
    assert start is not None and end is not None, "could not locate mainloop body"
    return lines[start:end + 1]


@pytest.fixture(scope="module")
def dual_src():
    return _emit_dual_period_kernel()


def test_six_scale_reads_per_epoch(dual_src):
    body = _mainloop_lines(dual_src)
    reads = [ln for ln in body if re.search(r"scaleMXS.*load 4B", ln)]
    assert len(reads) == 6, (
        "expected exactly six scale ds_reads per 256-K epoch, got %d:\n%s"
        % (len(reads), "\n".join(reads)))
    # 4 SA groups (group0..3) + 2 SB groups (group0..1).
    sa = sorted(re.search(r"group(\d+)", r).group(1)
                for r in reads if "MXSA" in r)
    sb = sorted(re.search(r"group(\d+)", r).group(1)
                for r in reads if "MXSB" in r)
    assert sa == ["0", "1", "2", "3"], "scale-A groups: %s" % sa
    assert sb == ["0", "1"], "scale-B groups: %s" % sb


def test_both_scale_phases_consumed(dual_src):
    body = _mainloop_lines(dual_src)
    mfmas = [ln for ln in body if "v_mfma_scale" in ln]
    phases = {}
    for ln in mfmas:
        m = re.search(r"A\[\d+,K=(\d+)\]", ln)
        if m:
            phases[m.group(1)] = phases.get(m.group(1), 0) + 1
    assert set(phases) == {"0", "1"}, "MFMA scale phases present: %s" % phases
    # Symmetric: scaleReuse==2 data iterations => equal MFMA counts per phase.
    assert phases["0"] == phases["1"], "phase MFMA counts asymmetric: %s" % phases
    # Phase 1 MFMAs must select the high scale byte (op_sel_hi:[1,1]).
    for ln in mfmas:
        if re.search(r"A\[\d+,K=1\]", ln):
            assert "op_sel_hi:[1,1]" in ln, "phase-1 MFMA missing hi byte-sel: %s" % ln


def test_scale_cadence_once_per_epoch(dual_src):
    body = "\n".join(_mainloop_lines(dual_src))
    # One SRD advance, one DTL load, one GR write-half swap and one LR read-half
    # swap for EACH of the two scale tensors (MXSA, MXSB) per 256-K epoch.
    assert body.count("Scale SRD update") == 2
    assert body.count("DTL b128 load") == 2
    assert len(re.findall(r"swap MXS\w+ GR m0", body)) == 2
    assert len(re.findall(r"swap MXS\w+ LR vgpr", body)) == 2


def test_vgpr_within_two_wave_slot_budget(dual_src):
    m = re.search(r"\.vgpr_count:\s*(\d+)", dual_src)
    assert m, "could not find .vgpr_count in emitted assembly"
    vgpr = int(m.group(1))
    assert vgpr <= 128, (
        "architectural VGPR %d exceeds the two-wave-slot budget (128)" % vgpr)


def test_prefetch_local_read_is_name_only(dual_src):
    """PrefetchLocalRead never reaches the dual-period schedule.

    The partition-directed prefetch pins its own plr=0 base, the same way
    ForceUnrollSubIter pins numItersPLR/numVgprBuffer for non-subtile kernels,
    so the declared value survives only in the kernel name.  Comments are
    stripped because they quote scheduler state that is allowed to mention the
    declared value.
    """
    def instructions(src):
        return [ln.split("//")[0].rstrip() for ln in _mainloop_lines(src)]

    assert instructions(_emit_dual_period_kernel(plr=1)) == instructions(dual_src)
