# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure this device's achievable HBM bandwidth and fp16 WMMA compute peak.

Runbook 3.1 asks you to compare a kernel's arithmetic intensity against the
hardware balance point ``peak_flops / peak_bw``. gfx1250 has no reference under
``dsl_docs/optimization/arch/``, so rather than quote a spec sheet this measures
both ends of the roofline directly with two minimal kernels.

``bw``
    Grid-stride streaming read over a large buffer using 128-bit (``dwordx4``)
    buffer loads, accumulated into registers so nothing is dead-code
    eliminated. Reports achieved read GB/s.

``compute``
    Back-to-back register-resident ``wmma_gfx1250_f32_16x16x32_f16`` over
    ``--chains`` *independent* accumulators, so throughput is bounded by issue
    rate rather than by one accumulator's dependency chain. No memory traffic
    in the loop. Reports achieved fp16 TFLOPS.

Both are deliberately naive. The point is an upper bound to normalize real
kernels against -- not an optimized kernel.

Run on a gfx1250 device::

    export ROCM=/opt/rocm-10.0.0a20260729
    export ROCKE_HIP_LIB=$ROCM/lib/libamdhip64.so LD_LIBRARY_PATH=$ROCM/lib
    PYTHONPATH=platform/python python3 -m rocke.examples.gfx1250.gemm.roofline_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.helpers import compile_kernel

ARCH = "gfx1250"
WAVE = 32
BLOCK = 256  # threads per workgroup, both kernels

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
FLOPS_PER_WMMA = 2 * WMMA_M * WMMA_N * WMMA_K  # 16384, wave-level


# --------------------------------------------------------------------------
# Kernel 1: streaming read bandwidth
# --------------------------------------------------------------------------
def build_bw_kernel(vec: int = 4):
    """Grid-stride streaming read. One f32 written per thread to defeat DCE."""
    b = IRBuilder(f"rocke_bw_read_v{vec}")
    src = b.param("src", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("dst", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    n_chunks = b.param("n_chunks", I32)  # total vec-sized chunks
    stride = b.param("stride", I32)  # grid stride, in chunks
    src_bytes = b.param("src_bytes", I32)
    dst_bytes = b.param("dst_bytes", I32)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    rsrc = b.buffer_rsrc(src, src_bytes)

    loop = b.scf_for_iter(
        gid, n_chunks, stride, [("acc", b.zero_vec_f32(vec))], iv_name="i"
    )
    with loop as (iv, accs):
        off = b.mul(iv, b.const_i32(vec * 4))  # byte offset of this chunk
        v = b.buffer_load_vN(rsrc, off, b.const_i32(0), F32, vec)
        b.scf_yield(b.vector_add(accs[0], v))
    acc = loop.results[0]

    total = b.vec_extract(acc, 0)
    for i in range(1, vec):
        total = b.fadd(total, b.vec_extract(acc, i))
    drsrc = b.buffer_rsrc(dst, dst_bytes)
    b.buffer_store_f32(drsrc, b.mul(gid, b.const_i32(4)), b.const_i32(0), total)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


# --------------------------------------------------------------------------
# Kernel 2: fp16 WMMA issue-rate peak
# --------------------------------------------------------------------------
def build_wmma_peak_kernel(chains: int = 8):
    """``chains`` independent WMMA accumulator chains, no memory in the loop."""
    b = IRBuilder(f"rocke_wmma_peak_c{chains}")
    dst = b.param("dst", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    iters = b.param("iters", I32)
    dst_bytes = b.param("dst_bytes", I32)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())

    # Register-resident operands. Values are irrelevant to issue rate; they are
    # non-zero so nothing can be folded away as an identity.
    a_h = b.trunc_f32_to_f16(b.const_f32(1.0009765625))
    b_h = b.trunc_f32_to_f16(b.const_f32(0.99951171875))
    a_frag = b.vector_splat(a_h, 16)  # <16 x half>
    b_frag = b.vector_splat(b_h, 16)  # <16 x half>

    init = [(f"acc{i}", b.zero_vec_f32(8)) for i in range(chains)]
    loop = b.scf_for_iter(b.const_i32(0), iters, b.const_i32(1), init, iv_name="it")
    with loop as (_iv, accs):
        nxt = [b.wmma_gfx1250_f32_16x16x32_f16(a_frag, b_frag, accs[i]) for i in range(chains)]
        b.scf_yield(*nxt)

    # Reduce every chain into one scalar so none can be eliminated.
    total = b.const_f32(0.0)
    for acc in loop.results:
        for i in range(8):
            total = b.fadd(total, b.vec_extract(acc, i))
    drsrc = b.buffer_rsrc(dst, dst_bytes)
    b.buffer_store_f32(drsrc, b.mul(gid, b.const_i32(4)), b.const_i32(0), total)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


# --------------------------------------------------------------------------
# Kernel 3: fp16 WMMA fed from LDS (operand-supply ceiling)
# --------------------------------------------------------------------------
def build_wmma_lds_kernel(chains: int = 16):
    """WMMA with one ``ds_read_b128`` per WMMA -- the GEMM's operand ratio.

    ``build_wmma_peak_kernel`` holds both operands in registers, so it measures
    the pure issue-rate ceiling. A real GEMM cannot: it streams operands out of
    LDS. The tuned gfx1250 GEMM's main loop issues exactly **one**
    ``ds_read_b128`` per WMMA, so this kernel reproduces that 1:1 ratio while
    keeping everything else identical to the peak kernel.

    The gap between this and :func:`build_wmma_peak_kernel` is therefore the
    cost of operand supply alone -- how much of the register-fed peak an
    LDS-fed kernel can reach at all, before any GEMM-specific overhead.
    """
    from rocke.core.ir import F16

    b = IRBuilder(f"rocke_wmma_lds_c{chains}")
    dst = b.param("dst", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    iters = b.param("iters", I32)
    dst_bytes = b.param("dst_bytes", I32)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    lane = b.thread_id_x()

    # One <16 x half> fragment slot per lane, mirroring an A-tile row.
    smem = b.smem_alloc(F16, [BLOCK, 16], name_hint="frag_lds")
    a_seed = b.trunc_f32_to_f16(b.const_f32(1.0009765625))
    b.smem_store_vN(smem, [lane, b.const_i32(0)], b.vector_splat(a_seed, 8), 8)
    b.smem_store_vN(smem, [lane, b.const_i32(8)], b.vector_splat(a_seed, 8), 8)
    b.sync()

    # Mirror the GEMM's 4x4 sub-tile reuse: 4 A-fragments x 4 B-fragments feed
    # 16 WMMA. Each <16 x half> fragment costs two b128 LDS reads (the widest
    # smem_load_vN for f16), so 8 fragments = 16 reads per 16 WMMA -- exactly
    # the 1:1 ds_read:WMMA ratio the tuned kernel's ISA shows.
    SUB = 4
    assert chains == SUB * SUB, "wmma_lds models a 4x4 sub-tile warp (16 accs)"

    def _frag(slot: int):
        lo = b.smem_load_vN(smem, lane, b.const_i32(0), dtype=F16, n=8)
        hi = b.smem_load_vN(smem, lane, b.const_i32(8), dtype=F16, n=8)
        return b.vec_concat(lo, hi)

    init = [(f"acc{i}", b.zero_vec_f32(8)) for i in range(chains)]
    loop = b.scf_for_iter(b.const_i32(0), iters, b.const_i32(1), init, iv_name="it")
    with loop as (_iv, accs):
        a_frags = [_frag(i) for i in range(SUB)]
        b_frags = [_frag(SUB + i) for i in range(SUB)]
        nxt = []
        for i in range(SUB):
            for j in range(SUB):
                nxt.append(
                    b.wmma_gfx1250_f32_16x16x32_f16(
                        a_frags[i], b_frags[j], accs[i * SUB + j]
                    )
                )
        b.scf_yield(*nxt)

    total = b.const_f32(0.0)
    for acc in loop.results:
        for i in range(8):
            total = b.fadd(total, b.vec_extract(acc, i))
    drsrc = b.buffer_rsrc(dst, dst_bytes)
    b.buffer_store_f32(drsrc, b.mul(gid, b.const_i32(4)), b.const_i32(0), total)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


def build_wmma_hybrid_kernel(chains: int = 16):
    """WMMA with A fed from **global** and B from LDS -- the half-LDS ceiling.

    ``build_wmma_lds_kernel`` models the tuned GEMM's 1:1 ``ds_read``:WMMA
    ratio. That ratio is structural: it is ``(m+n)*2/(m*n)`` for an ``m x n``
    per-warp atom grid, and the only way to improve it is a bigger grid (8x8
    needs 512 accumulator VGPRs against a 256 ceiling) or to source one operand
    from somewhere other than LDS.

    This kernel measures the second option. In RCR the A fragment is
    K-contiguous, so it is contiguous in *global* memory too and can be loaded
    straight to registers with no transpose. Doing so halves the ratio to 0.5.

    The A buffer is deliberately tiny and re-read by every block, so it stays
    resident in cache. That is the honest model of the real scheme: the four
    warps of a column all want the same A rows at the same time, so the
    redundancy is cache traffic, not HBM traffic. This kernel therefore
    measures whether the cache path is any less of a bottleneck than LDS --
    it is not an HBM bandwidth test.

    Reading: land near the register-fed peak and sourcing A outside LDS is
    worth the restructure; land near the LDS-fed number and the cache path is
    just as constrained and the gap is structural on this part.
    """
    from rocke.core.ir import F16

    b = IRBuilder(f"rocke_wmma_hybrid_c{chains}")
    dst = b.param("dst", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    src = b.param("src", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    iters = b.param("iters", I32)
    dst_bytes = b.param("dst_bytes", I32)
    src_bytes = b.param("src_bytes", I32)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(BLOCK)), b.thread_id_x())
    lane = b.thread_id_x()

    # B still comes from LDS, exactly as in the LDS-fed kernel.
    smem = b.smem_alloc(F16, [BLOCK, 16], name_hint="frag_lds")
    a_seed = b.trunc_f32_to_f16(b.const_f32(1.0009765625))
    b.smem_store_vN(smem, [lane, b.const_i32(0)], b.vector_splat(a_seed, 8), 8)
    b.smem_store_vN(smem, [lane, b.const_i32(8)], b.vector_splat(a_seed, 8), 8)
    b.sync()

    srsrc = b.buffer_rsrc(src, src_bytes)
    # Per-lane 16-byte slot, wrapped into a small cache-resident window.
    a_off = b.mul(b.mod(lane, b.const_i32(BLOCK)), b.const_i32(128))

    SUB = 4
    assert chains == SUB * SUB, "wmma_hybrid models a 4x4 sub-tile warp (16 accs)"

    def _frag_lds():
        lo = b.smem_load_vN(smem, lane, b.const_i32(0), dtype=F16, n=8)
        hi = b.smem_load_vN(smem, lane, b.const_i32(8), dtype=F16, n=8)
        return b.vec_concat(lo, hi)

    def _frag_global(slot: int):
        # 32 B per fragment (two b128 loads); stride by 32 so the four
        # fragments do not overlap -- overlapping slots let CSE merge the
        # loads and would understate the operand cost being measured.
        base = b.add(a_off, b.const_i32((slot % SUB) * 32))
        lo = b.buffer_load_vN_f16(srsrc, base, b.const_i32(0), 4)
        hi = b.buffer_load_vN_f16(srsrc, b.add(base, b.const_i32(16)), b.const_i32(0), 4)
        return b.vec_concat(lo, hi)

    init = [(f"acc{i}", b.zero_vec_f32(8)) for i in range(chains)]
    loop = b.scf_for_iter(b.const_i32(0), iters, b.const_i32(1), init, iv_name="it")
    with loop as (_iv, accs):
        a_frags = [_frag_global(i) for i in range(SUB)]   # 8 global b128 loads
        b_frags = [_frag_lds() for _ in range(SUB)]       # 8 ds_read_b128
        nxt = []
        for i in range(SUB):
            for j in range(SUB):
                nxt.append(
                    b.wmma_gfx1250_f32_16x16x32_f16(
                        a_frags[i], b_frags[j], accs[i * SUB + j]
                    )
                )
        b.scf_yield(*nxt)

    total = b.const_f32(0.0)
    for acc in loop.results:
        for i in range(8):
            total = b.fadd(total, b.vec_extract(acc, i))
    drsrc = b.buffer_rsrc(dst, dst_bytes)
    b.buffer_store_f32(drsrc, b.mul(gid, b.const_i32(4)), b.const_i32(0), total)
    b.ret()
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    return b.kernel


# --------------------------------------------------------------------------
# Launch helpers
# --------------------------------------------------------------------------
def _pack(names_types, values) -> bytes:
    """Pack kernel args through rocke's kernarg packer.

    Delegates to :func:`rocke.runtime.packing.pack_args` rather than a local
    ``struct.pack``: the AMDGPU kernarg ABI aligns each field to its own size
    (8 for ptr/i64, 4 for i32/f32), and a hand-rolled packer that pads every
    scalar to 8 bytes silently misplaces every field after the first i32.
    """
    from rocke.runtime.packing import pack_args

    sig = [
        {"name": n, "type": t, "size_bytes": 8 if t.startswith("ptr") else 4}
        for n, t in names_types
    ]
    return pack_args(sig, values)


_PTR = "ptr<f32, global>"


def _time(rt, fn, grid, block, args, warmup, iters) -> float:
    for _ in range(warmup):
        rt.launch(fn, grid, block, args)
    rt.sync()
    e0, e1 = rt.event(), rt.event()
    e0.record()
    for _ in range(iters):
        rt.launch(fn, grid, block, args)
    e1.record()
    e1.synchronize()
    ms = e0.elapsed_to(e1) / iters
    e0.destroy()
    e1.destroy()
    return ms


def _load(rt, blobs: dict, key: str, build):
    """Resolve a kernel to (function, name), from a prebuilt blob if supplied.

    Compiling and launching in one process loads two LLVM copies (comgr's and
    the HIP runtime's) and aborts with a CommandLine option clash, so on a GPU
    host we launch prebuilt HSACOs instead of compiling in-process.
    """
    if key in blobs:
        name, blob = blobs[key]
    else:
        art = compile_kernel(build(), arch=ARCH)
        name, blob = art.kernel_name, art.hsaco
    return rt.load_module(blob).get_function(name), name


def run_bw(rt, mib: int, blocks: int, warmup: int, iters: int, blobs: dict) -> dict:
    vec = 4
    nbytes = mib * 1024 * 1024
    if nbytes >= 2**31:
        # src_bytes is an i32 kernarg (and the buffer descriptor's num_records
        # field is 32-bit), so a >=2 GiB buffer cannot be addressed this way.
        raise ValueError(f"--mib {mib} exceeds the 2 GiB i32 addressing limit")
    n_chunks = nbytes // (vec * 4)
    fn, _ = _load(rt, blobs, "bw", lambda: build_bw_kernel(vec))

    threads = blocks * BLOCK
    src = rt.alloc(nbytes)
    dst_bytes = threads * 4
    dst = rt.alloc(dst_bytes)
    rt.memset(src, 0, nbytes)
    args = _pack(
        [("src", _PTR), ("dst", _PTR), ("n_chunks", "i32"), ("stride", "i32"),
         ("src_bytes", "i32"), ("dst_bytes", "i32")],
        {"src": src, "dst": dst, "n_chunks": n_chunks, "stride": threads,
         "src_bytes": nbytes, "dst_bytes": dst_bytes},
    )
    ms = _time(rt, fn, (blocks, 1, 1), (BLOCK, 1, 1), args, warmup, iters)
    rt.free(src)
    rt.free(dst)
    gbps = nbytes / (ms * 1e-3) / 1e9
    return {"kernel": "bw_read", "mib": mib, "blocks": blocks, "ms": ms, "gbps": gbps}


def run_compute(
    rt, chains: int, loop_iters: int, blocks: int, warmup: int, iters: int, blobs: dict,
    key: Optional[str] = None,
) -> dict:
    _k = key or f"wmma{chains}"
    fn, _ = _load(rt, blobs, _k, lambda: build_wmma_peak_kernel(chains))

    threads = blocks * BLOCK
    dst_bytes = threads * 4
    dst = rt.alloc(dst_bytes)
    args = _pack(
        [("dst", _PTR), ("iters", "i32"), ("dst_bytes", "i32")],
        {"dst": dst, "iters": loop_iters, "dst_bytes": dst_bytes},
    )
    ms = _time(rt, fn, (blocks, 1, 1), (BLOCK, 1, 1), args, warmup, iters)
    rt.free(dst)
    waves = blocks * (BLOCK // WAVE)
    flops = waves * loop_iters * chains * FLOPS_PER_WMMA
    return {
        "kernel": "wmma_peak",
        "chains": chains,
        "loop_iters": loop_iters,
        "blocks": blocks,
        "ms": ms,
        "tflops": flops / (ms * 1e-3) / 1e12,
    }


def run_compute_hybrid(
    rt, chains: int, loop_iters: int, blocks: int, warmup: int, iters: int, blobs: dict
) -> dict:
    """Time the A-from-global / B-from-LDS kernel.

    ``src`` is intentionally small (one block's worth of fragment slots) and is
    re-read by every block, so it stays cache-resident. This measures the cache
    path as an operand source, not HBM bandwidth.
    """
    fn, _ = _load(rt, blobs, f"wmmahybrid{chains}",
                  lambda: build_wmma_hybrid_kernel(chains))
    threads = blocks * BLOCK
    dst_bytes = threads * 4
    src_bytes = BLOCK * 128  # 128 B/lane: 4 fragments x 32 B
    dst = rt.alloc(dst_bytes)
    src = rt.alloc(src_bytes)
    rt.memset(src, 0, src_bytes)
    args = _pack(
        [("dst", _PTR), ("src", _PTR), ("iters", "i32"),
         ("dst_bytes", "i32"), ("src_bytes", "i32")],
        {"dst": dst, "src": src, "iters": loop_iters,
         "dst_bytes": dst_bytes, "src_bytes": src_bytes},
    )
    ms = _time(rt, fn, (blocks, 1, 1), (BLOCK, 1, 1), args, warmup, iters)
    rt.free(src)
    rt.free(dst)
    waves = blocks * (BLOCK // WAVE)
    flops = waves * loop_iters * chains * FLOPS_PER_WMMA
    return {"kernel": "wmma_hybrid", "chains": chains, "ms": ms,
            "tflops": flops / (ms * 1e-3) / 1e12}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--which", choices=("bw", "compute", "both"), default="both")
    p.add_argument("--mib", type=int, default=1024, help="buffer size for the bw test")
    p.add_argument("--bw-blocks", type=int, default=2048)
    p.add_argument("--compute-blocks", type=int, default=1024)
    p.add_argument("--chains", default="", help="comma-separated chain counts; default 1,2,4,8,16")
    p.add_argument("--loop-iters", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--build-to",
        type=Path,
        default=None,
        help="compile the kernels to this dir and exit (run on any host)",
    )
    p.add_argument(
        "--run-from",
        type=Path,
        default=None,
        help="launch prebuilt HSACOs from this dir instead of compiling",
    )
    args = p.parse_args(argv)

    chain_list = ([int(x) for x in args.chains.split(",")] if args.chains
                  else [1, 2, 4, 8, 16])

    if args.build_to:
        args.build_to.mkdir(parents=True, exist_ok=True)
        meta = {}
        specs = [("bw", lambda: build_bw_kernel(4))]
        specs += [(f"wmma{c}", (lambda c=c: build_wmma_peak_kernel(c))) for c in chain_list]
        specs += [("wmmalds16", lambda: build_wmma_lds_kernel(16))]
        specs += [("wmmahybrid16", lambda: build_wmma_hybrid_kernel(16))]
        for key, build in specs:
            art = compile_kernel(build(), arch=ARCH)
            (args.build_to / f"{key}.hsaco").write_bytes(art.hsaco)
            meta[key] = art.kernel_name
        (args.build_to / "kernels.json").write_text(json.dumps(meta, indent=2))
        print(f"[roofline] built {len(meta)} kernels -> {args.build_to}")
        return 0

    blobs = {}
    if args.run_from:
        meta = json.loads((args.run_from / "kernels.json").read_text())
        for key, name in meta.items():
            blobs[key] = (name, (args.run_from / f"{key}.hsaco").read_bytes())

    from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_num_cus

    rt = Runtime()
    arch = get_device_arch()
    cus = get_device_num_cus()
    results = [{"arch": arch, "num_cus": cus}]
    if not args.json:
        print(f"[roofline] device arch={arch} CUs={cus}")

    if args.which in ("bw", "both"):
        for mib in (64, 256, 1024, 1536):
            r = run_bw(rt, mib, args.bw_blocks, args.warmup, args.iters, blobs)
            results.append(r)
            if not args.json:
                print(f"  bw   {mib:5d} MiB  {r['ms']:8.3f} ms  {r['gbps']:8.1f} GB/s")

    if args.which in ("compute", "both"):
        if "wmmalds16" in blobs:
            r = run_compute(rt, 16, args.loop_iters, args.compute_blocks,
                            args.warmup, args.iters, blobs, key="wmmalds16")
            results.append(r)
            if not args.json:
                print(f"  wmma LDS-fed c16  {r['ms']:8.3f} ms  {r['tflops']:8.1f} TFLOPS")
        if "wmma_hybrid16" in blobs or "wmmahybrid16" in blobs:
            r = run_compute_hybrid(rt, 16, args.loop_iters, args.compute_blocks,
                                   args.warmup, args.iters, blobs)
            results.append(r)
            if not args.json:
                print(f"  wmma hybrid c16   {r['ms']:8.3f} ms  {r['tflops']:8.1f} TFLOPS"
                      "   (A from global, B from LDS)")
        for c in chain_list:
            r = run_compute(
                rt, c, args.loop_iters, args.compute_blocks, args.warmup, args.iters, blobs
            )
            results.append(r)
            if not args.json:
                print(
                    f"  wmma chains={c:<3d}    {r['ms']:8.3f} ms  {r['tflops']:8.1f} TFLOPS"
                )

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
