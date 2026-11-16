################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
# ISA-level regression for the dual-period MXFP8 subtile kernel.
#
# The dual-period path keeps the 128-K data loop while running a decoupled
# 256-K pre-shuffled scale epoch that is reused by two data iterations.  This
# test drives the REAL winning geometry (MT256x256, MIWT 8x4, MIWG 2x4, DU128,
# PGR2/PLR1, 4x2 partitions, N-major, WGM4) through CPU-only codegen and
# asserts the steady-state mainloop honors the scale contract:
#
#   * exactly SIX scale ds_read_b32 per 256-K epoch (4 SA groups + 2 SB groups)
#   * both K phases consumed by the MFMAs (K=0 for data iter 0, K=1 for iter 1)
#   * every scale GR / SRD-advance / LDS-swap fires exactly once per epoch
#   * architectural VGPR stays within the two-wave-slot budget (<= 128)
#
# It needs the gfx950 assembly toolchain (amdclang++); it skips otherwise.
################################################################################

import os
import re
import shutil

import pytest


# MT256x256, MIWT 8x4, MIWG 2x4: the geometry described above.
_MI_8X4 = (16, 16, 128, 1, 1, 8, 4, 2, 4)
# MT256x320, MIWT 8x10, MIWG 2x2 (library sol 102): its halves are long enough
# for planHalfDisjointBalance to move wrap A tiles into a current-cell read.
_MI_8X10 = (16, 16, 128, 1, 1, 8, 10, 2, 2)


def _emit_dual_period_kernel(plr=1, mi=_MI_8X4):
    """Build the real dual-period MXFP8 solution and return its assembly text.

    Partition-directed PLR (next-partition prefetch, n-major traversal, 2x2
    partition grid) is selected automatically from the dual-period MXFP8 solution
    parameters -- no environment knobs are required.  ``plr`` only reaches the
    solution; the schedule pins its own base, so it cannot change the assembly.
    """
    res = _codegen_kernel(plr=plr, mi=mi)
    assert res.err == 0, "kernel codegen failed (err=%s)" % res.err
    src = res.src
    if isinstance(src, (bytes, bytearray)):
        src = src.decode(errors="replace")
    return src


def _codegen_kernel(plr=1, mx=True, mi=_MI_8X4):
    """Run codegen on the dual-period geometry, or a plain-FP8 kernel when not mx."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults
    from tensilelite.Common.GlobalParameters import assignGlobalParameters
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
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

    mi = list(mi)
    pt = {"OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
          "ComputeDataType": "s", "HighPrecisionAccumulate": True,
          "TransposeA": True, "TransposeB": False, "UseBeta": True,
          "Batched": True, "StridedBatched": True}
    if mx:
        pt.update({"MXBlockA": 32, "MXBlockB": 32})
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
    assert sol.get("Valid"), "solution did not validate"
    assert bool(sol.get("_DualPeriodMXFP8")) == mx, "unexpected dual-period selection"

    from tensilelite.create_library.run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
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
        return processKernelSource(
            kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
    pytest.skip("no kernel object generated")


def _refuse_half_disjoint_plans(monkeypatch):
    from tensilelite.Components.Subtile.LogicalScheduler import LogicalScheduler
    monkeypatch.setattr(LogicalScheduler, "_plan_half_disjoint_loads",
                        lambda self, partitions: None)


def test_refused_half_disjoint_plan_rejects_the_kernel(monkeypatch, capsys):
    _refuse_half_disjoint_plans(monkeypatch)
    res = _codegen_kernel()
    assert res.err != 0
    assert "no dual-period MXFP8 partition candidate has a half-disjoint schedule" \
        in capsys.readouterr().out


def test_plain_fp8_kernel_ignores_the_half_disjoint_plan(monkeypatch):
    _refuse_half_disjoint_plans(monkeypatch)
    assert _codegen_kernel(mx=False, mi=(16, 16, 128, 1, 1, 4, 4, 2, 2)).err == 0


def test_half_disjoint_env_selects_the_base_schedule(monkeypatch):
    from tensilelite.Components.Subtile.Kernel import (
        MXFP8_HALF_DISJOINT_ENV, mxfp8HalfDisjointLds)
    monkeypatch.delenv(MXFP8_HALF_DISJOINT_ENV, raising=False)
    assert mxfp8HalfDisjointLds()
    for value, expected in (("1", True), ("0", False), (" 0 ", False)):
        monkeypatch.setenv(MXFP8_HALF_DISJOINT_ENV, value)
        assert mxfp8HalfDisjointLds() is expected, value
    for value in ("", "2", "yes", "off"):
        monkeypatch.setenv(MXFP8_HALF_DISJOINT_ENV, value)
        with pytest.raises(ValueError, match=MXFP8_HALF_DISJOINT_ENV):
            mxfp8HalfDisjointLds()


def test_uniform_epoch_build_ignores_a_refused_half_disjoint_plan(monkeypatch):
    from tensilelite.Components.Subtile.Kernel import MXFP8_HALF_DISJOINT_ENV
    monkeypatch.setenv(MXFP8_HALF_DISJOINT_ENV, "0")
    _refuse_half_disjoint_plans(monkeypatch)
    assert _codegen_kernel().err == 0


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


def _mfma_gap_lines(body):
    """Code of the instructions after each mainloop MFMA, up to the next."""
    gaps = None
    out = []
    for ln in body:
        code = ln.split("//")[0].strip()
        if not code or code.startswith("/*") or code.endswith(":"):
            continue
        if code.startswith("v_mfma"):
            gaps = []
            out.append(gaps)
        elif gaps is not None:
            gaps.append(code)
    return out


def _mfma_gaps(body):
    """Opcodes of the instructions after each mainloop MFMA, up to the next."""
    return [[code.split()[0] for code in gap] for gap in _mfma_gap_lines(body)]


def _is_valu(op):
    return op.startswith("v_") and not op.startswith("v_mfma")


@pytest.fixture(scope="module")
def dual_src():
    return _emit_dual_period_kernel()


@pytest.fixture(scope="module")
def uniform_src():
    """The 8x4 kernel built on the uniform-epoch base schedule."""
    from tensilelite.Components.Subtile.Kernel import MXFP8_HALF_DISJOINT_ENV
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(MXFP8_HALF_DISJOINT_ENV, "0")
        return _emit_dual_period_kernel()


@pytest.fixture(scope="module", params=["half-disjoint", "uniform-epoch"])
def base_src(request, dual_src, uniform_src):
    """The 8x4 kernel on each base schedule."""
    return dual_src if request.param == "half-disjoint" else uniform_src


@pytest.fixture(scope="module")
def balanced_8x10():
    """The 8x10 kernel, with each balanced-read deferral recorded.

    Returns the assembly and one (body, balanced reads, applied) entry per
    main-loop body, in emit order. The bodies are the scheduler's modules
    after every post-pass, so they hold the final order."""
    from tensilelite.Components.Subtile import InstructionScheduler
    defer = InstructionScheduler.deferBalancedCurrentCellWait
    calls = []

    def record(module, balancedReads, *args):
        applied = defer(module, balancedReads, *args)
        calls.append((module, list(balancedReads), applied))
        return applied

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(InstructionScheduler, "deferBalancedCurrentCellWait", record)
        src = _emit_dual_period_kernel(mi=_MI_8X10)
    return src, calls


@pytest.fixture(scope="module")
def unbalanced_8x10_src():
    from tensilelite.Components.Subtile import LogicalScheduler
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LogicalScheduler, "planHalfDisjointBalance", lambda *a, **k: None)
        return _emit_dual_period_kernel(mi=_MI_8X10)


@pytest.fixture(scope="module", params=["8x4", "8x10"])
def any_src(request, dual_src, balanced_8x10):
    """Geometry-independent checks run on both kernels."""
    return dual_src if request.param == "8x4" else balanced_8x10[0]


def _body_sync(src):
    """(barriers, vmcnt waits) of each main-loop body, C0 then C1."""
    bodies, counts = _mainloop_lines(src), []
    for cell in ("C0", "C1"):
        start = next(i for i, ln in enumerate(bodies) if f"MAINLOOP_{cell} start" in ln)
        end = next(i for i, ln in enumerate(bodies) if f"MAINLOOP_{cell} end" in ln)
        code = [ln.split("//")[0].strip() for ln in bodies[start:end]]
        counts.append((sum(c.startswith("s_barrier") for c in code),
                       sum(c.startswith("s_waitcnt") and "vmcnt" in c for c in code)))
    return counts


def test_half_disjoint_bodies_open_each_half_with_a_vmem_wait_and_barrier(dual_src):
    assert _body_sync(dual_src) == [(2, 2), (2, 2)]


def test_uniform_epoch_bodies_keep_one_vmem_wait_and_three_barriers(uniform_src):
    assert _body_sync(uniform_src) == [(3, 1), (3, 1)]


def test_uniform_epoch_barriers_have_no_valu_behind_them(uniform_src):
    for gap in _mfma_gaps(_mainloop_lines(uniform_src)):
        if "s_barrier" in gap:
            assert not any(_is_valu(op) for op in gap[gap.index("s_barrier"):]), gap


def test_six_scale_reads_per_epoch(base_src):
    body = _mainloop_lines(base_src)
    reads = [ln for ln in body if "ds_read" in ln and re.search(r"// S[AB] MT=", ln)]
    assert len(reads) == 6, (
        "expected exactly six scale ds_reads per 256-K epoch, got %d:\n%s"
        % (len(reads), "\n".join(reads)))
    # 4 SA tiles + 2 SB tiles, each carrying its macro-tile epoch.
    sa = sorted(int(re.search(r"tile=(\d+)", r).group(1))
                for r in reads if "// SA " in r)
    sb = sorted(int(re.search(r"tile=(\d+)", r).group(1))
                for r in reads if "// SB " in r)
    assert sa == [0, 2, 4, 6], "scale-A tiles: %s" % sa
    assert sb == [0, 2], "scale-B tiles: %s" % sb
    assert all("MT=" in r for r in reads)


def test_both_scale_phases_consumed(base_src):
    body = _mainloop_lines(base_src)
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


def test_scale_cadence_once_per_epoch(base_src):
    body = "\n".join(_mainloop_lines(base_src))
    # One SRD advance, one DTL load, one GR write-half swap and one LR read-half
    # swap for EACH of the two scale tensors (MXSA, MXSB) per 256-K epoch.
    assert body.count("Scale SRD update") == 2
    scale_dtl = [
        ln for ln in body.splitlines()
        if "buffer_load" in ln and re.search(r"// S[AB] MT=", ln)
    ]
    assert len(scale_dtl) == 2, "scale DTL loads: %s" % scale_dtl
    assert len(re.findall(r"swap MXS\w+ GR m0", body)) == 2
    assert len(re.findall(r"swap MXS\w+ LR vgpr", body)) == 2


def test_valu_stays_out_of_direct_to_lds_gaps(any_src):
    for i, gap in enumerate(_mfma_gaps(_mainloop_lines(any_src))):
        assert not (any(_is_valu(op) for op in gap)
                    and any(op.startswith("buffer_load") for op in gap)), (i, gap)


def test_no_gap_opens_with_a_valu(any_src):
    for i, gap in enumerate(_mfma_gaps(_mainloop_lines(any_src))):
        assert not (gap and _is_valu(gap[0])), (i, gap)


def test_salu_issues_first_in_each_gap(any_src):
    """No SALU, m0 write or load follows a ds_read or VALU of its gap
    unless a wait, barrier or branch separates them or it uses the
    earlier instruction's result. The loop counter and back-branch after
    the C1 cell are not part of any gap."""
    body = _mainloop_lines(any_src)
    body = body[:next(i for i, ln in enumerate(body) if "MAINLOOP_C1 end" in ln)]
    for i, gap in enumerate(_mfma_gap_lines(body)):
        behind = []
        for code in gap:
            op = code.split()[0]
            if op.startswith(("s_waitcnt", "s_barrier", "s_cbranch", "s_branch")):
                behind = []
            elif op.startswith("ds_read") or _is_valu(op):
                behind.append(code)
            elif op.startswith(("s_", "buffer_load")):
                operands = code[len(op):]
                for earlier in behind:
                    dst = earlier.split()[1].rstrip(",")
                    assert dst in operands, (i, gap, earlier, code)


def test_load_runs_and_salu_per_gap_stay_capped(any_src):
    """At most five consecutive gaps issue a load, and no gap holds more
    SALU and m0 writes than its five follower slots."""
    run = 0
    for i, gap in enumerate(_mfma_gaps(_mainloop_lines(any_src))):
        run = run + 1 if any(op.startswith("buffer_load") for op in gap) else 0
        assert run <= 5, (i, gap)
        salu = [op for op in gap if op.startswith("s_") and not op.startswith(
            ("s_waitcnt", "s_barrier", "s_cbranch", "s_branch"))]
        assert len(salu) <= 5, (i, gap)


def _partition_cell_gaps(body):
    """Instructions per MFMA gap of each partition cell, in body order.

    A cell's gaps sit between its own MFMAs; the gap after its last MFMA
    holds the next cell's prefix."""
    cells = []
    gaps = None
    for ln in body:
        code = ln.split("//")[0].strip()
        if code.startswith("/* partition="):
            cells.append([])
            gaps = None
        elif not code or code.startswith("/*") or code.endswith(":"):
            continue
        elif code.startswith("v_mfma"):
            if cells:
                gaps = []
                cells[-1].append(gaps)
        elif gaps is not None:
            gaps.append(code)
    return [cell[:-1] for cell in cells if cell]


def _half_gaps(body):
    """Instructions per MFMA gap of each half, in body order.

    Partition cells of one half are scheduled together, so a half's gaps
    run across its partition markers; the gap after a cell's last MFMA
    holds the next cell's fence. Each half opens with its barrier."""
    halves = []
    gaps = None
    for ln in body:
        code = ln.split("//")[0].strip()
        if not code or code.startswith("/*") or code.endswith(":"):
            continue
        if code.startswith("v_mfma"):
            gaps = []
            if halves:
                halves[-1].append(gaps)
        elif (code.startswith("s_barrier") and gaps is not None
              and all(prior.startswith("s_waitcnt") for prior in gaps)):
            if halves and halves[-1][-1] is gaps:
                halves[-1].pop()
            halves.append([gaps])
            gaps.append(code)
        elif gaps is not None:
            gaps.append(code)
    return [half[:-1] for half in halves]


def test_c1_scale_refreshes_sit_at_their_war_floors(dual_src):
    """Each next-C0 scale refresh issues in the first legal C1 gap past
    the ninth MFMA after its last C1 reader, and takes the back-edge gap
    only when that is its floor."""
    body = _mainloop_lines(dual_src)
    gap, c1Start, refreshes = -1, None, []
    for ln in body:
        code = ln.split("//")[0].strip()
        if "MAINLOOP_C1 start" in ln:
            c1Start = gap + 1
        if code.startswith("v_mfma"):
            gap += 1
        elif "prior C1 shadow" in ln:
            last = int(re.search(r"last-use=M(\d+)", ln).group(1))
            refreshes.append((gap, c1Start + last + 8))
    backEdge = gap
    assert refreshes
    gaps = _mfma_gap_lines(body)
    for at, floor in refreshes:
        assert floor <= at <= backEdge, (at, floor)
        assert sum(code.startswith("ds_read") for code in gaps[at]) <= 2, gaps[at]
        assert at < backEdge or floor == backEdge, (at, floor)
    assert [floor for _at, floor in refreshes] == [at for at, _floor in refreshes]


def test_loads_share_a_gap_with_reads_only_when_no_free_gap_is_left(any_src):
    """A half puts a load beside a ds_read only when it has fewer gaps
    after its barrier that could take a load alone than loads: gaps with
    no read, wait, barrier, VALU or request, room for m0 and the load,
    and no run past five. The SRD and LDS-base updates that close the
    requests follow the last load, so a gap after the half's last SALU
    does not count."""
    def has(gap, *prefixes):
        return any(code.startswith(prefixes) for code in gap)

    halves = _half_gaps(_mainloop_lines(any_src))
    assert len(halves) == 4
    for c, gaps in enumerate(halves):
        barrier = [i for i, gap in enumerate(gaps) if has(gap, "s_barrier")]
        opened = barrier[0] + 1 if barrier else 0
        loaded = [has(gap, "buffer_load") for gap in gaps]
        lastSalu = max((i for i, gap in enumerate(gaps) if any(
            code.startswith("s_") and not code.startswith(("s_waitcnt", "s_barrier"))
            and code.split()[1:2] != ["m0,"] for code in gap)), default=-1)

        def run(i):
            left = right = i
            while left > 0 and loaded[left - 1]:
                left -= 1
            while right + 1 < len(gaps) and loaded[right + 1]:
                right += 1
            return right - left + 1

        shared = [i for i in range(opened, len(gaps))
                  if loaded[i] and has(gaps[i], "ds_read")]
        free = [i for i in range(opened, lastSalu)
                if not loaded[i] and len(gaps[i]) <= 3 and run(i) <= 5
                and not has(gaps[i], "ds_read", "s_waitcnt", "s_barrier")
                and not any(_is_valu(code.split()[0]) for code in gaps[i])
                and not any(code.split()[1:2] == ["m0,"] for code in gaps[i])]
        assert not shared or len(free) < sum(loaded), (c, shared, free, gaps)


def test_vgpr_within_two_wave_slot_budget(base_src):
    m = re.search(r"\.vgpr_count:\s*(\d+)", base_src)
    assert m, "could not find .vgpr_count in emitted assembly"
    vgpr = int(m.group(1))
    assert vgpr <= 128, (
        "architectural VGPR %d exceeds the two-wave-slot budget (128)" % vgpr)


def test_odd_phase_tail_swaps_scale_write_half(base_src):
    """An odd residual re-enters mid-epoch, so the write half needs a step.

    The scale SRDs are already where the tail wants them -- the kernel seeds
    one epoch in the preloop and advances once per epoch it consumes -- so the
    only thing out of phase is which LDS half the tail's own scale load lands
    in.  Nothing here may touch Srd+0/+1/+2.
    """
    align_start = base_src.index(
        "dual-period tail: test completed-DepthU parity for scale LDS")
    align_end = base_src.index("label_SubtileTailScalePhased:", align_start)
    tail_start = base_src.index("/* TAILLOOP */", align_end)
    alignment = base_src[align_start:align_end]

    parity_test = next(
        ln for ln in base_src.splitlines()
        if "dual-period tail: test completed-DepthU parity for scale LDS" in ln)
    assert parity_test.lstrip().startswith("s_bitcmp1_b32")
    assert re.search(r",\s*7(?:\s|//)", parity_test), parity_test
    assert "s_cbranch_scc0 label_SubtileTailScalePhased" in alignment

    for tc in ("MXSA", "MXSB"):
        swap = next(
            ln for ln in alignment.splitlines()
            if f"sgprLocalWriteBaseAddr{tc}" in ln)
        assert swap.lstrip().startswith("s_xor_b32"), swap
        assert f"sgprSwap{tc}" in swap, swap

    assert not re.search(r"SrdMXS[AB]\+?\d*\]", alignment), (
        "tail scale phase fixup must not touch the scale SRDs:\n%s" % alignment)
    assert align_start < align_end < tail_start


def test_preloop_seeds_one_scale_epoch(base_src):
    """PGR2 seeds two data iterations but only the one epoch that covers them.

    A second scale prefetch would cover data the preloop never fetched, and
    its SRD advance would leave every later epoch load one epoch past the
    data frontier.  So the preloop issues one DTL load per scale tensor, no
    scale SRD advance, and no write-half swap -- the mainloop body owns the
    advance-swap-load cadence from there.
    """
    preloop = []
    for ln in base_src.splitlines():
        if ln.strip() == "label_LoopBeginL:":
            break
        preloop.append(ln)
    preloop = "\n".join(preloop)

    scale_loads = [ln for ln in preloop.splitlines()
                   if ln.startswith("buffer_load") and re.search(r"// S[AB] MT=", ln)]
    assert len(scale_loads) == 2, scale_loads
    assert "Scale SRD update" not in preloop
    assert not re.search(r"swap MXS\w+ GR m0", preloop)

    wait = next(ln for ln in preloop.splitlines()
                if "Wait GR" in ln and "SA=" in ln and "s_branch" not in ln)
    assert "SA=0 SB=0" in wait, (
        "preloop wait must retire the seeded epoch, not leave one in flight: %s"
        % wait)


def test_no_scale_local_read_in_preloop(base_src):
    """The pre-loop prefetches data operands but not scales.

    Scale local reads are epoch-granular: the mainloop, NLL, NGLL and tail
    paths each re-read the whole group set at their head, so a pre-loop copy
    would be overwritten before any MFMA consumes it.  Data local reads are
    partition-granular and must still be there, feeding partition 0's MFMAs.
    """
    preloop = []
    for ln in base_src.splitlines():
        if ln.strip() == "label_LoopBeginL:":
            break
        preloop.append(ln)

    scale_lr = [l for l in preloop if "ds_read" in l and re.search(r"// S[AB] MT=", l)]
    assert not scale_lr, "pre-loop re-reads scales the mainloop reloads: %s" % scale_lr

    data_lr = [l for l in preloop if "ds_read" in l and re.search(r"// [AB] MT=", l)]
    assert data_lr, "pre-loop lost its data operand prefetch"


def test_loads_name_their_macro_tile(base_src):
    """A, B, and scale loads name the macro-tile epoch they fill.

    PGR2 issues the current tile, the next tile, and the tile two ahead.
    The comment is the identity later physical passes use to tell those
    loads apart.  Opcodes and addresses are unchanged.
    """
    src = base_src
    assert re.search(r"// A MT=n ", src)
    assert re.search(r"// B MT=n ", src)
    assert re.search(r"// A MT=n\+1 ", src) or re.search(r"// B MT=n\+1 ", src)
    assert "MT=n+2" in src


def test_8x10_balances_both_main_loop_bodies(balanced_8x10):
    _src, calls = balanced_8x10
    assert len(calls) == 2, "expected the C0 and C1 main-loop bodies"
    for body, reads, applied in calls:
        assert reads, body.name
        assert applied, body.name


def test_8x10_balanced_reads_follow_their_opening_barrier(balanced_8x10):
    from tensilelite.Components.Subtile.InstructionScheduler import _isBarrier
    for body, reads, _applied in balanced_8x10[1]:
        items = list(body.flatitems())
        barrier = next(k for k, inst in enumerate(items) if _isBarrier(inst))
        at = [next(k for k, inst in enumerate(items) if inst is read) for read in reads]
        assert all(k > barrier for k in at), (body.name, barrier, at)


def test_8x10_balanced_reads_have_one_covering_wait(balanced_8x10):
    """Between each balanced read and its first consumer MFMA exactly one
    lgkm wait retires it: none is missing and none is redundant."""
    from tensilelite.Components.Subtile.InstructionScheduler import (
        _incrementsLgkm, _instructionRegisterSpans, _isMfma, _isWaitCnt,
        _lgkmCovers, _readsSpans)
    for body, reads, _applied in balanced_8x10[1]:
        items = list(body.flatitems())
        for read in reads:
            at = next(k for k, inst in enumerate(items) if inst is read)
            dests = [span for role, span in _instructionRegisterSpans(read) if role == "dst"]
            consumer = next(k for k in range(at + 1, len(items))
                            if _isMfma(items[k]) and _readsSpans(items[k], dests))
            covering, issued = 0, 0
            for inst in items[at + 1:consumer]:
                if _isWaitCnt(inst) and _lgkmCovers(inst, issued):
                    covering += 1
                if _incrementsLgkm(inst):
                    issued += 1
            assert covering == 1, (body.name, read.comment, covering)


def _vmcnt_sequence(src):
    return re.findall(r"vmcnt\((\d+)\)", "\n".join(
        ln.split("//")[0] for ln in _mainloop_lines(src)))


def test_8x10_balance_keeps_the_vmcnt_sequence(balanced_8x10, unbalanced_8x10_src):
    balanced = _vmcnt_sequence(balanced_8x10[0])
    assert balanced
    assert balanced == _vmcnt_sequence(unbalanced_8x10_src)
    assert "balanced current-cell" not in "\n".join(_mainloop_lines(unbalanced_8x10_src))
