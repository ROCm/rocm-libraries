# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark of the emitted skinny MXFP8 GEMM (and the split-K reduce).

Prints one SWEEP line per shape. The timing methodology is: before every GEMM
launch, timed or not, a read-only kernel streams a 512 MiB buffer to empty the
LLC (twice the 256 MiB MALL of the MI355X) and the device is synchronised, so
N timed iterations are N flush launches plus N GEMM launches. The HIP events
bracket only the GEMM, never the flush: they are attached to the dispatch with
hipExtModuleLaunchKernel when the symbol loads, which reads the kernel duration
plus about 0.5 us with a floor of about 4 us; the fallback records the events
around hipModuleLaunchKernel on the null stream, which adds 3 to 4 us with a
floor of about 6 us. Kernels below about 3.5 us therefore cannot be told apart.
The reported times are means over the iterations. The split-K reduce is timed
warm, without a flush, after the GEMM iterations, because in the real pipeline
it runs right after its GEMM. The gemm_min_us and gemm_p90_us columns are
informational.

Run from the tensilelite directory:

  PYTHONPATH=build_tmp/tensilelite/src/tensilelite/bindings \\
  LD_LIBRARY_PATH=/opt/rocm/lib \\
  ~/.tensile/bin/python tests/tensilelite/bindings/bench_skinny_gemm.py \\
  --sweep vllm --iters 30
"""

import argparse
import ctypes
import glob
import math
import os
import sys
from contextlib import ExitStack

import yaml

import skinny_gemm_support as support
from skinny_gemm_support import DeviceBuffer, LoadedKernel, Problem, hipCheck

# Twice the 256 MiB MALL of the MI355X: a 256 MiB flush left 1 to 3% of the
# operands warm, while 384 MiB and more are equal within noise.
FLUSH_MB = 512
# Untimed flush plus launch pairs after a kernel is loaded.
POST_LOAD_LAUNCHES = 2
EXT_TIMING = "hipExtModuleLaunchKernel events"
EVENT_TIMING = "hipEventRecord around hipModuleLaunchKernel"
FLUSH_KERNEL_NAME = "flush_llc_kernel"
# The assembly of the LLC flush kernel: each thread of the workgroup loads its
# slice of the buffer one dword at a time. It only reads, so the LLC is left
# clean.
FLUSH_ASM = """
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl flush_llc_kernel
  .p2align 8
  .type flush_llc_kernel,@function
flush_llc_kernel:
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)
  s_mul_i32 s2, s6, {threads}
  v_add_u32 v1, s2, v0
  s_mov_b32 s3, {bytesPerThread}
  v_mul_lo_u32 v2, v1, s3
  s_mov_b32 s2, {elementsPerThread}
.Lloop:
  global_load_dword v3, v2, s[4:5]
  s_waitcnt vmcnt(0)
  v_add_u32 v2, v2, 4
  s_sub_u32 s2, s2, 1
  s_cmp_gt_u32 s2, 0
  s_cbranch_scc1 .Lloop
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel flush_llc_kernel
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size flush_llc_kernel, .Lfunc_end0-flush_llc_kernel

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: flush_llc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .sgpr_spill_count: 0
    .symbol: flush_llc_kernel.kd
    .vgpr_count: 4
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.version:
  - 1
  - 2
---
  .end_amdgpu_metadata
"""
WAVEFRONT = 64


def packArgs(args):
    """The kernel arguments (ctypes values) packed into the object that
    hipModuleLaunchKernel otherwise builds on every launch."""
    try:
        # hip-python 10 (a wrapper over rocm-bindings) moved this helper.
        from rocm.bindings import _hip_helpers
    except ImportError:
        # Earlier releases exposed it directly under hip.
        from hip import _hip_helpers

    return _hip_helpers.HipModuleLaunchKernel_extra(tuple(args))


def allocateFlushBuffer(nbytes: int):
    """The device buffer a flush kernel streams over."""
    try:
        return hipCheck(support._hip().hipMalloc(nbytes))
    except RuntimeError as err:
        raise RuntimeError(
            f"cannot allocate the {nbytes // 2**20} MiB LLC flush buffer: {err}; "
            "is another process using this GPU's memory?"
        ) from err


class LlcFlush:
    """Empties the LLC by reading a buffer larger than it, one workgroup per
    CU, then waits for the device; a context manager that frees the buffer."""

    def __init__(self):
        props = support.getDeviceProperties()
        self.workgroups = props.multiProcessorCount
        elements = FLUSH_MB * 2**20 // 4
        elementsPerThread = elements // (self.workgroups * WAVEFRONT)
        asm = FLUSH_ASM.format(
            threads=WAVEFRONT,
            bytesPerThread=4 * elementsPerThread,
            elementsPerThread=elementsPerThread,
        )
        import tensilelite_jit

        hsaco = tensilelite_jit.assembleToHSACO(asm, support.TARGET)
        self.buffer = None
        self.kernel = LoadedKernel(hsaco, FLUSH_KERNEL_NAME)
        try:
            self.buffer = allocateFlushBuffer(FLUSH_MB * 2**20)
            self.args = packArgs((self.buffer.as_c_void_p(),))
        except BaseException:
            self.close()
            raise

    def flush(self) -> None:
        hip = support._hip()
        hipCheck(
            hip.hipModuleLaunchKernel(
                self.kernel.function,
                self.workgroups,
                1,
                1,
                WAVEFRONT,
                1,
                1,
                sharedMemBytes=0,
                stream=None,
                kernelParams=None,
                extra=self.args,
            )
        )
        hipCheck(hip.hipDeviceSynchronize())

    def close(self) -> None:
        """Frees the buffer and closes the kernel, even if the free raises."""
        try:
            if self.buffer is not None:
                buffer, self.buffer = self.buffer, None
                hipCheck(support._hip().hipFree(buffer))
        finally:
            self.kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback) -> None:
        support.closeOnExit(self, value)


_EXT_UNLOADED = object()
_extLaunch = _EXT_UNLOADED


def loadExtLaunch():
    """hipExtModuleLaunchKernel through ctypes (hip-python lacks the module
    variant), or None if the runtime does not export it."""
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    # The bare soname first, so that ctypes binds to the runtime instance that
    # hip-python already loaded.
    for path in ("libamdhip64.so", os.path.join(root, "lib", "libamdhip64.so")):
        try:
            function = ctypes.CDLL(path).hipExtModuleLaunchKernel
        except (OSError, AttributeError):
            continue
        function.restype = ctypes.c_int
        function.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint32] * 6
            + [ctypes.c_size_t, ctypes.c_void_p]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_uint32]
        )
        return function
    return None


def extLaunchFunction():
    """The cached hipExtModuleLaunchKernel function, or None."""
    global _extLaunch
    if _extLaunch is _EXT_UNLOADED:
        _extLaunch = loadExtLaunch()
    return _extLaunch


def useExt(method: str | None) -> bool:
    return method != "events" and extLaunchFunction() is not None


def timingMethod(method: str | None = None) -> str:
    """The name of the event timing used for method (None or 'ext' prefer the
    hipExtModuleLaunchKernel events, 'events' forces the fallback)."""
    return EXT_TIMING if useExt(method) else EVENT_TIMING


def pointerOf(handle) -> int | None:
    return handle.as_c_void_p().value


def extLaunch(function, kernel, grid, block, packed, start, stop) -> None:
    """One launch with the events attached to the dispatch; the global sizes of
    hipExtModuleLaunchKernel are in threads."""
    err = function(
        pointerOf(kernel.function),
        grid[0] * block[0],
        grid[1] * block[1],
        grid[2] * block[2],
        *block,
        0,
        None,
        None,
        pointerOf(packed),
        pointerOf(start),
        pointerOf(stop),
        0,
    )
    if err != 0:
        raise RuntimeError(f"hipExtModuleLaunchKernel failed: error {err}")


def launch(hip, kernel, grid, block, packed) -> None:
    hipCheck(
        hip.hipModuleLaunchKernel(
            kernel.function,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=packed,
        )
    )


def timedLaunch(hip, ext, kernel, grid, block, packed, start, stop) -> int:
    """One launch bracketed by the events, in ns; ext is the ctypes function of
    hipExtModuleLaunchKernel, or None to record the events around the launch."""
    if ext is not None:
        extLaunch(ext, kernel, grid, block, packed, start, stop)
    else:
        hipCheck(hip.hipEventRecord(start, None))
        launch(hip, kernel, grid, block, packed)
        hipCheck(hip.hipEventRecord(stop, None))
    hipCheck(hip.hipEventSynchronize(stop))
    return int(hipCheck(hip.hipEventElapsedTime(start, stop)) * 1e6)


def timeLaunches(
    kernel: LoadedKernel,
    grid,
    block,
    args,
    iters: int,
    flush: LlcFlush | None,
    method: str | None = None,
) -> list[int]:
    """Times iters launches in ns, each preceded by its own flush (if any) and
    bracketed by HIP events that cover only the launch. The launch arguments
    are packed once: packing them inside the timed region would add the CPU cost
    to the event time. method 'events' forces the event-record fallback."""
    hip = support._hip()
    ext = extLaunchFunction() if useExt(method) else None
    packed = packArgs(args)
    times = []
    start = hipCheck(hip.hipEventCreate())
    try:
        stop = hipCheck(hip.hipEventCreate())
        try:
            for _ in range(iters):
                if flush is not None:
                    flush.flush()
                times.append(
                    timedLaunch(hip, ext, kernel, grid, block, packed, start, stop)
                )
        finally:
            hipCheck(hip.hipEventDestroy(stop))
    finally:
        hipCheck(hip.hipEventDestroy(start))
    return times


def warmLaunches(
    kernel: LoadedKernel, grid, block, args, count: int, flush: LlcFlush | None
) -> None:
    """Count untimed launches, each preceded by its flush (if any) and followed
    by a device synchronisation."""
    hip = support._hip()
    packed = packArgs(args)
    for _ in range(count):
        if flush is not None:
            flush.flush()
        launch(hip, kernel, grid, block, packed)
        hipCheck(hip.hipDeviceSynchronize())


class MissingAsm(Exception):
    """The saved asm of a kernel is not in the asm directory."""


def loadAsmKernel(stack: ExitStack, pattern: str, name: str) -> LoadedKernel:
    """Assembles the first (sorted) file matching pattern and loads the kernel
    name, which the file must define."""
    import tensilelite_jit

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise MissingAsm(f"no saved asm matches {pattern}")
    with open(paths[0]) as stream:
        asm = stream.read()
    if f".globl {name}\n" not in asm:
        raise RuntimeError(f"{paths[0]} does not define {name}")
    hsaco = tensilelite_jit.assembleToHSACO(asm, support.TARGET)
    return stack.enter_context(LoadedKernel(hsaco, name))


def gemmKernelFor(
    stack: ExitStack,
    p: Problem,
    asmDir: str | None,
    codeObjectVersion: int,
    maxPreloadDwords: int,
) -> LoadedKernel:
    if asmDir is None:
        return support.gemmKernel(p, codeObjectVersion, maxPreloadDwords)
    name = support.gemmKernelName(p)
    return loadAsmKernel(stack, f"{asmDir}/{name}_*/kernel.s", name)


def reduceKernelFor(
    stack: ExitStack,
    asmDir: str | None,
    codeObjectVersion: int,
    maxPreloadDwords: int,
) -> LoadedKernel:
    if asmDir is None:
        return support.reduceKernel(codeObjectVersion, maxPreloadDwords)
    name = support.REDUCE_KERNEL_NAME
    return loadAsmKernel(stack, f"{asmDir}/{name}_*/kernel.s", name)


def benchProblem(
    p: Problem,
    iters: int,
    verify: bool,
    asmDir: str | None,
    codeObjectVersion: int,
    maxPreloadDwords: int,
    method: str | None = None,
) -> tuple[list[int], list[int] | None]:
    """Times the GEMM (cold LLC) and, for split-K, the reduce (warm) in ns per
    iteration; with verify the results are checked against the host reference.
    The timed launches follow POST_LOAD_LAUNCHES untimed warm launches."""
    inputs = support.makeInputs(p, seed=42, withReference=verify)
    dBits, ws = support.makeOutputs(p.m, p.n, p.ldD, p.splits)
    arrays = (
        dBits,
        inputs.aBuf,
        inputs.bBuf,
        inputs.sclABuf,
        inputs.sclBBuf,
        ws,
        inputs.cBits,
    )
    reduceTimes = None
    with ExitStack() as stack:
        dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf = (
            stack.enter_context(DeviceBuffer(array)) for array in arrays
        )
        kernel = gemmKernelFor(stack, p, asmDir, codeObjectVersion, maxPreloadDwords)
        flush = stack.enter_context(LlcFlush())
        args = support.gemmArgs(p, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf, cBuf)
        block = support.gemmBlock(p)
        warmLaunches(kernel, p.grid, block, args, POST_LOAD_LAUNCHES, flush)
        gemmTimes = timeLaunches(kernel, p.grid, block, args, iters, flush, method)
        if verify:
            dBuf.download(dBits)
            wsBuf.download(ws)
            support.verifyGemm(p, inputs, dBits, ws)
        if p.splits > 1:
            reducer = reduceKernelFor(
                stack, asmDir, codeObjectVersion, maxPreloadDwords
            )
            grid, block = support.reduceLaunchDims(p.m, p.n)
            args = support.reduceArgs(
                dBuf, wsBuf, cBuf, p.ldD, p.m, p.n, p.splits, p.ldC, p.alpha, p.beta
            )
            warmLaunches(reducer, grid, block, args, POST_LOAD_LAUNCHES, None)
            reduceTimes = timeLaunches(reducer, grid, block, args, iters, None, method)
            if verify:
                dBuf.download(dBits)
                support.verifyD(
                    dBits, support.expectedD(p, inputs), p.m, p.n, p.ldD, p.label
                )
    return gemmTimes, reduceTimes


def mean(values) -> float:
    return sum(values) / len(values)


def percentile(values, fraction: float):
    """The value at sorted index ceil(fraction * n) - 1 (nearest rank)."""
    ordered = sorted(values)
    # The rounding keeps products such as 0.9 * 10 from landing above the integer.
    rank = math.ceil(round(fraction * len(ordered), 9))
    return ordered[max(rank, 1) - 1]


def formatSweepLine(p: Problem, gemmTimes, reduceTimes) -> str:
    """One SWEEP line: gemm_us and reduce_us are means over the iterations in
    us and total_us is the mean of the per-iteration gemm + reduce sums; the min
    and p90 columns of the GEMM are informational."""
    fields = {
        "m": p.m,
        "n": p.n,
        "k": p.k,
        "path": "A" if p.m <= p.n else "B",
        "bm": p.bm,
        "tn": p.tn,
        "lanes_k": p.lanes_k,
        "waves": p.waves,
        "pf": p.pf,
        "spk": p.spk,
        "group": p.group,
        "xcds": p.xcds,
        "up_scales": p.up_scales,
        "blocks": p.blocks,
        "grid3d": int(p.grid3d),
        "pieces": p.plan.pieces,
        "workgroups": p.plan.workgroups,
        **({"mfma": 1} if p.mfma else {}),
        **({"lds": 1} if p.lds else {}),
        **({"use_split_k": 1} if p.use_split_k else {}),
        "gemm_us": f"{mean(gemmTimes) * 1e-3:.2f}",
        "gemm_min_us": f"{min(gemmTimes) * 1e-3:.2f}",
        "gemm_p90_us": f"{percentile(gemmTimes, 0.9) * 1e-3:.2f}",
    }
    if reduceTimes is not None:
        total = mean([g + r for g, r in zip(gemmTimes, reduceTimes)])
        fields["reduce_us"] = f"{mean(reduceTimes) * 1e-3:.2f}"
        fields["total_us"] = f"{total * 1e-3:.2f}"
    return "SWEEP " + " ".join(f"{key}={value}" for key, value in fields.items())


def printAsm(p: Problem, codeObjectVersion: int, maxPreloadDwords: int) -> None:
    """Prints the emitted GEMM kernel's .s, and the split-K reduce's if the
    problem splits K, straight from the emitter (no assembling or GPU)."""
    print(f"# ASM {p.label}", flush=True)
    print(support.gemmAsm(p, codeObjectVersion, maxPreloadDwords))
    if p.splits > 1:
        print(f"# ASM {support.REDUCE_KERNEL_NAME}", flush=True)
        print(support.reduceAsm(codeObjectVersion, maxPreloadDwords))


def sweepShape(
    m: int,
    n: int,
    k: int,
    iters: int,
    verify: bool,
    overrides: dict,
    asmDir,
    printOnly: bool,
    codeObjectVersion: int,
    maxPreloadDwords: int,
    splitRemainder: int = 1,
    method: str | None = None,
) -> None:
    """Runs one shape and prints its SWEEP line, or with printOnly its emitted
    asm instead of benchmarking it; a shape that violates the constraints with
    the overrides, or lacks saved asm, is reported as skipped. splitRemainder is
    the remainder narrowing of the mfma 3-D kernel: 0 picks it for the compute
    units of the device (off when only printing, which needs no device), 1 turns
    it off and N forces N pieces."""
    knobs = {name: None for name in support.KNOB_OVERRIDE_NAMES}
    knobs.update(overrides)
    try:
        p = support.makeProblem(m, n, k, padding=(0, 0, 0), **knobs)
        if printOnly and splitRemainder == 0:
            splitRemainder = 1
        if splitRemainder != 1:
            cus = support.getDeviceProperties().multiProcessorCount
            p = support.resolvePieces(p, splitRemainder, cus)
        if printOnly:
            printAsm(p, codeObjectVersion, maxPreloadDwords)
            return
        gemmTimes, reduceTimes = benchProblem(
            p, iters, verify, asmDir, codeObjectVersion, maxPreloadDwords, method
        )
    except (AssertionError, MissingAsm) as err:
        reason = str(err).splitlines()[0]
        print(f"SWEEP m={m} n={n} k={k} SKIPPED {reason}", flush=True)
        return
    print(formatSweepLine(p, gemmTimes, reduceTimes), flush=True)


def vllmSweepShapes() -> list[tuple]:
    """The primary shape, the vLLM shapes and their transposes, without
    duplicates."""
    shapes = [support.PRIMARY_SHAPE] + support.VLLM_SHAPES
    shapes += [(n, m, k) for m, n, k in support.VLLM_SHAPES]
    return list(dict.fromkeys(shapes))


SWEEPS = {"vllm": vllmSweepShapes}


def makeParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the emitted skinny MXFP8 GEMM"
    )
    parser.add_argument("-m", type=int, default=4, help="Columns of A / rows of D")
    parser.add_argument("-n", type=int, default=8192, help="Columns of B / cols of D")
    parser.add_argument("-k", type=int, default=8192, help="Shared dimension")
    parser.add_argument("--iters", type=int, default=30, help="Timed launches")
    parser.add_argument(
        "--timing",
        choices=("ext", "events"),
        default="ext",
        help="ext attaches the events to the dispatch with "
        "hipExtModuleLaunchKernel (falls back to events if the runtime lacks "
        "it); events records them around hipModuleLaunchKernel, which reads "
        "3 to 4 us more",
    )
    parser.add_argument(
        "--spk",
        type=int,
        default=None,
        help="K elements per split; selects the split-K kernel (default: the "
        "kernel without split-K over the whole K)",
    )
    parser.add_argument("--bm", type=int, default=None, help="Skinny columns per block")
    parser.add_argument("--tn", type=int, default=None, help="Wide columns per lane")
    parser.add_argument("--lanes-k", type=int, default=None, help="Lanes sharing K")
    parser.add_argument("--waves", type=int, default=None, help="Waves per block")
    parser.add_argument("--pf", type=int, default=None, help="Prefetch depth")
    parser.add_argument("--group", type=int, default=None, help="Tile rows per group")
    parser.add_argument("--xcds", type=int, default=None, help="XCDs of the tile order")
    parser.add_argument(
        "--up-scales",
        type=int,
        choices=(0, 1),
        default=None,
        help="1 loads the scales of the split up front, 0 streams them",
    )
    parser.add_argument(
        "--persistent-blocks",
        type=int,
        default=None,
        help="Blocks of a 1-D persistent grid (default: the 3-D grid)",
    )
    parser.add_argument(
        "--use-beta",
        action="store_true",
        help="Run the useBeta GEMM, which prefetches C and adds beta * C",
    )
    parser.add_argument(
        "--mfma",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force the scaled-MFMA kernel (with the untuned plain defaults "
        "unless --lds) or, with --no-mfma, the VALU kernel; by default the "
        "shape picks, and a knob option selects the VALU kernel",
    )
    parser.add_argument(
        "--lds",
        action="store_true",
        default=None,
        help="With --mfma, stage the operands through LDS (coalesced loads) "
        "with the tuned defaults",
    )
    parser.add_argument(
        "--config",
        metavar="MAPPING",
        default=None,
        help="Run one kernel config given as a YAML mapping, as in the "
        "'config' entries of select_skinny_gemm_kernels.py's selection, e.g. "
        "'{bm: 16, tn: 1, lanes_k: 64, waves: 4, pf: 2, up_iters: 0, "
        "group: 1, xcds: 1, grid3d: true, blocks: null, "
        "mfma: true, lds: true, use_beta: true, use_split_k: false}'; "
        "use_split_k true needs --spk and false forbids it; knob options given "
        "explicitly override its entries",
    )
    parser.add_argument(
        "--code-object-version",
        type=int,
        choices=support.CODE_OBJECT_VERSIONS,
        default=support.DEFAULT_CODE_OBJECT_VERSION,
        help="Code object version the kernels are emitted for",
    )
    parser.add_argument(
        "--max-preload-dwords",
        type=int,
        default=0,
        help="Kernarg dwords the kernels preload into SGPRs (0 disables it)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale of A * B")
    parser.add_argument("--beta", type=float, default=0.0, help="Scale of C")
    parser.add_argument(
        "--sweep",
        choices=sorted(SWEEPS),
        default=None,
        help="Run a fixed shape list, ignoring -m/-n/-k; the knob options apply "
        "to every shape and shapes they make invalid are skipped",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check the results against the host reference after the timing",
    )
    parser.add_argument(
        "--asm-dir",
        metavar="DIR",
        default=None,
        help="Run previously saved asm instead of the emitted kernels, e.g. "
        "for an A/B comparison (DIR/<gemm kernel name>_*/kernel.s and "
        "DIR/splitk_reduce_*/kernel.s)",
    )
    parser.add_argument(
        "--split-remainder",
        type=int,
        default=0,
        metavar="N",
        help="Remainder narrowing of the mfma 3-D kernel with lds: 0 picks the "
        "pieces per remainder tile for the CUs of the device, 1 turns it off, "
        "N >= 2 forces N pieces (a power of two up to 16 * tn); with "
        "--print-asm 0 means off",
    )
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Print the emitted kernel(s) .s for the shape(s) instead of "
        "benchmarking; no assembling or GPU is needed",
    )
    return parser


CONFIG_KEYS = frozenset(
    (
        "bm",
        "tn",
        "lanes_k",
        "waves",
        "pf",
        "up_iters",
        "group",
        "xcds",
        "grid3d",
        "blocks",
        "mfma",
        "lds",
        "use_beta",
        "use_split_k",
    )
)


def parseConfig(text: str, spk: int | None = None) -> dict:
    """The makeProblem knobs of a selection config mapping. up_iters is derived
    from the shape, so only whether it is non-zero (the up-front scales) is
    kept; blocks is only meaningful for the persistent grid, which grid3d false
    selects. use_split_k tells the kernel with split-K from the one without, but
    the split size is not in the mapping, so spk (the --spk option) must be
    given exactly when use_split_k is true."""
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise SystemExit(f"--config must be a mapping, got {text!r}")
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise SystemExit(f"--config has unknown keys {unknown}")
    use_split_k = config.get("use_split_k")
    if use_split_k and spk is None:
        raise SystemExit("--config with use_split_k true needs --spk")
    if use_split_k is not None and not use_split_k and spk is not None:
        raise SystemExit(
            "--config with use_split_k false selects the kernel without "
            "split-K; drop --spk"
        )
    grid3d = bool(config.get("grid3d", True))
    if not grid3d and config.get("blocks") is None:
        raise SystemExit("--config with grid3d false needs blocks")
    mfma = bool(config.get("mfma", False))
    overrides = {
        name: config[name]
        for name in ("bm", "tn", "lanes_k", "waves", "pf", "group", "xcds", "lds")
        if config.get(name) is not None
    }
    overrides["mfma"] = mfma
    overrides["blocks"] = None if grid3d else config["blocks"]
    if "up_iters" in config:
        overrides["up_scales"] = 0 if mfma else int(config["up_iters"] > 0)
    if config.get("use_beta"):
        overrides["use_beta"] = True
    return overrides


def knobOverrides(args: argparse.Namespace) -> dict:
    """The knob options that were given, keyed by makeProblem's argument names."""
    values = {
        name: getattr(args, name)
        for name in support.KNOB_OVERRIDE_NAMES
        if name != "blocks"
    }
    values["blocks"] = args.persistent_blocks
    # Only the non-default values are passed on, so the sweep header of a run
    # without them stays as it was.
    if args.use_beta:
        values["use_beta"] = True
    if args.alpha != 1.0:
        values["alpha"] = args.alpha
    if args.beta != 0.0:
        values["beta"] = args.beta
    given = {name: value for name, value in values.items() if value is not None}
    if args.config is None:
        return given
    return {**parseConfig(args.config, args.spk), **given}


def timingBanner(args: argparse.Namespace) -> str:
    """The timing method in effect; notes a fallback from the requested one."""
    method = timingMethod(args.timing)
    if args.timing == "ext" and method != EXT_TIMING:
        print(f"# note: hipExtModuleLaunchKernel is unavailable, using {method}")
    return f"timing={method.replace(' ', '_')} flush_mib={FLUSH_MB}"


def main() -> None:
    args = makeParser().parse_args()
    overrides = knobOverrides(args)
    banner = None if args.print_asm else timingBanner(args)
    if args.sweep is None:
        if banner is not None:
            print(f"# {banner}", flush=True)
        sweepShape(
            args.m,
            args.n,
            args.k,
            args.iters,
            args.verify,
            overrides,
            args.asm_dir,
            args.print_asm,
            args.code_object_version,
            args.max_preload_dwords,
            args.split_remainder,
            args.timing,
        )
        return
    if banner is not None:
        header = f"# sweep={args.sweep} iters={args.iters} verify={int(args.verify)}"
        header += f" cold_llc=1 {banner}"
        for key, value in overrides.items():
            header += f" {key}={value}"
        print(header, flush=True)
    for m, n, k in SWEEPS[args.sweep]():
        sweepShape(
            m,
            n,
            k,
            args.iters,
            args.verify,
            overrides,
            args.asm_dir,
            args.print_asm,
            args.code_object_version,
            args.max_preload_dwords,
            args.split_remainder,
            args.timing,
        )


if __name__ == "__main__":
    sys.exit(main())
