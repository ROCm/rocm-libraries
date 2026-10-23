# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark of the emitted skinny MXFP8 GEMM (and the split-K reduce).

Prints one SWEEP line per shape in the format of aster's
kernels/test_gemm_mxfp8_skinny.py, with the same timing methodology: before
each GEMM launch a read-only kernel streams a 256 MiB buffer to empty the LLC
and the device is synchronised; HIP events on the null stream bracket a single
launch whose kernel arguments were built once; the reported time is the minimum
over the iterations (the first included). The split-K reduce is timed warm,
without the flush, after the GEMM iterations.

Run from the tensilelite directory:

  PYTHONPATH=build_tmp/tensilelite/src/tensilelite/bindings \\
  LD_LIBRARY_PATH=/opt/rocm/lib \\
  ~/.tensile/bin/python tests/tensilelite/bindings/bench_skinny_gemm.py \\
  --sweep vllm --iters 30
"""

import argparse
import glob
import sys
from contextlib import ExitStack

import skinny_gemm_support as support
from skinny_gemm_support import DeviceBuffer, LoadedKernel, Problem, hipCheck

FLUSH_MB = 256
FLUSH_KERNEL_NAME = "flush_llc_kernel"
# The assembly of aster's flush_llc kernel: each thread of the workgroup loads
# its slice of the buffer one dword at a time. Despite the name of the aster
# class it only reads, so the LLC is left clean.
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
    from hip import _hip_helpers

    return _hip_helpers.HipModuleLaunchKernel_extra(tuple(args))


class LlcFlush:
    """Empties the LLC by reading a buffer larger than it, one workgroup per
    CU, then waits for the device; a context manager that frees the buffer."""

    def __init__(self):
        hip = support._hip()
        props = hip.hipDeviceProp_t()
        hipCheck(hip.hipGetDeviceProperties(props, 0))
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
        self.kernel = LoadedKernel(hsaco, FLUSH_KERNEL_NAME)
        self.buffer = hipCheck(hip.hipMalloc(FLUSH_MB * 2**20))
        self.args = packArgs((self.buffer.as_c_void_p(),))

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
        if self.buffer is None:
            return
        hipCheck(support._hip().hipFree(self.buffer))
        self.buffer = None
        self.kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def timeLaunches(
    kernel: LoadedKernel, grid, block, args, iters: int, flush: LlcFlush | None
) -> list[int]:
    """Times iters launches in ns, each bracketed by HIP events on the null
    stream after the flush, if any. The launch arguments are packed once: packing
    them inside the timed region would add the CPU cost to the event time."""
    hip = support._hip()
    packed = packArgs(args)
    start = hipCheck(hip.hipEventCreate())
    stop = hipCheck(hip.hipEventCreate())
    times = []
    try:
        for _ in range(iters):
            if flush is not None:
                flush.flush()
            hipCheck(hip.hipEventRecord(start, None))
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
            hipCheck(hip.hipEventRecord(stop, None))
            hipCheck(hip.hipEventSynchronize(stop))
            milliseconds = hipCheck(hip.hipEventElapsedTime(start, stop))
            times.append(int(milliseconds * 1e6))
    finally:
        hipCheck(hip.hipEventDestroy(start))
        hipCheck(hip.hipEventDestroy(stop))
    return times


class MissingAsm(Exception):
    """The reference asm of a kernel is not in the asm directory."""


def loadAsmKernel(stack: ExitStack, pattern: str, name: str) -> LoadedKernel:
    """Assembles the first (sorted) file matching pattern and loads the kernel
    name, which the file must define."""
    import tensilelite_jit

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise MissingAsm(f"no reference asm matches {pattern}")
    with open(paths[0]) as stream:
        asm = stream.read()
    if f".globl {name}\n" not in asm:
        raise RuntimeError(f"{paths[0]} does not define {name}")
    hsaco = tensilelite_jit.assembleToHSACO(asm, support.TARGET)
    return stack.enter_context(LoadedKernel(hsaco, name))


def gemmKernelFor(stack: ExitStack, p: Problem, asmDir: str | None) -> LoadedKernel:
    if asmDir is None:
        return support.gemmKernel(p)
    name = support.gemmKernelName(p)
    return loadAsmKernel(stack, f"{asmDir}/{name}_*/kernel.s", name)


def reduceKernelFor(stack: ExitStack, asmDir: str | None) -> LoadedKernel:
    if asmDir is None:
        return support.reduceKernel()
    name = support.REDUCE_KERNEL_NAME
    return loadAsmKernel(stack, f"{asmDir}/{name}_*/kernel.s", name)


def benchProblem(
    p: Problem, iters: int, verify: bool, asmDir: str | None
) -> tuple[list[int], list[int] | None]:
    """Times the GEMM (cold LLC) and, for split-K, the reduce (warm) in ns per
    iteration; with verify the results are checked against the host reference."""
    inputs = support.makeInputs(p, seed=42, withReference=verify)
    dBits, ws = support.makeOutputs(p.m, p.n, p.ldD, p.splits)
    arrays = (dBits, inputs.aBuf, inputs.bBuf, inputs.sclABuf, inputs.sclBBuf, ws)
    reduceTimes = None
    with ExitStack() as stack:
        dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf = (
            stack.enter_context(DeviceBuffer(array)) for array in arrays
        )
        kernel = gemmKernelFor(stack, p, asmDir)
        flush = stack.enter_context(LlcFlush())
        args = support.gemmArgs(p, dBuf, aBuf, bBuf, sclABuf, sclBBuf, wsBuf)
        gemmTimes = timeLaunches(
            kernel, p.grid, support.gemmBlock(p), args, iters, flush
        )
        if verify:
            dBuf.download(dBits)
            wsBuf.download(ws)
            support.verifyGemm(p, inputs, dBits, ws)
        if p.splits > 1:
            reducer = reduceKernelFor(stack, asmDir)
            grid, block = support.reduceLaunchDims(p.m, p.n)
            args = support.reduceArgs(dBuf, wsBuf, p.ldD, p.m, p.n, p.splits)
            reduceTimes = timeLaunches(reducer, grid, block, args, iters, None)
            if verify:
                dBuf.download(dBits)
                support.verifyD(dBits, inputs.expected, p.m, p.n, p.ldD, p.label)
    return gemmTimes, reduceTimes


def formatSweepLine(p: Problem, gemmTimes, reduceTimes) -> str:
    """One SWEEP line in the format of the aster driver: times are minima over
    the iterations in us; total_us is the minimum of the per-iteration gemm +
    reduce sums."""
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
        "gemm_us": f"{min(gemmTimes) * 1e-3:.2f}",
    }
    if reduceTimes is not None:
        total = min(g + r for g, r in zip(gemmTimes, reduceTimes))
        fields["reduce_us"] = f"{min(reduceTimes) * 1e-3:.2f}"
        fields["total_us"] = f"{total * 1e-3:.2f}"
    return "SWEEP " + " ".join(f"{key}={value}" for key, value in fields.items())


def sweepShape(
    m: int, n: int, k: int, iters: int, verify: bool, overrides: dict, asmDir
) -> None:
    """Runs one shape and prints its SWEEP line; a shape that violates the
    constraints with the overrides, or lacks reference asm, is reported as
    skipped."""
    knobs = {name: None for name in support.KNOB_OVERRIDE_NAMES}
    knobs.update(overrides)
    try:
        p = support.makeProblem(m, n, k, padding=(0, 0, 0), **knobs)
        gemmTimes, reduceTimes = benchProblem(p, iters, verify, asmDir)
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
    parser.add_argument("--spk", type=int, default=None, help="K elements per split")
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
        help="Harness validation: run the reference asm saved by the aster "
        "driver's --save-tmps (DIR/<gemm kernel name>_*/kernel.s and "
        "DIR/splitk_reduce_*/kernel.s) instead of the emitted kernels",
    )
    return parser


def knobOverrides(args: argparse.Namespace) -> dict:
    """The knob options that were given, keyed by makeProblem's argument names."""
    values = {
        name: getattr(args, name)
        for name in support.KNOB_OVERRIDE_NAMES
        if name != "blocks"
    }
    values["blocks"] = args.persistent_blocks
    return {name: value for name, value in values.items() if value is not None}


def main() -> None:
    args = makeParser().parse_args()
    overrides = knobOverrides(args)
    if args.sweep is None:
        sweepShape(
            args.m, args.n, args.k, args.iters, args.verify, overrides, args.asm_dir
        )
        return
    header = f"# sweep={args.sweep} iters={args.iters} verify={int(args.verify)}"
    header += " cold_llc=1"
    for key, value in overrides.items():
        header += f" {key}={value}"
    print(header, flush=True)
    for m, n, k in SWEEPS[args.sweep]():
        sweepShape(m, n, k, args.iters, args.verify, overrides, args.asm_dir)


if __name__ == "__main__":
    sys.exit(main())
