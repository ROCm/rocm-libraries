# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Qwen3-30B-A3B gfx950<->gfx1250 performance-parity scoreboard.

Day-0 bar: everything works AND gfx1250 reaches performance parity with gfx950
on the Qwen3-30B-A3B model. This harness captures the per-operator wall-clock
(plus modeled HBM traffic + achieved GB/s + reference TFLOPs) at the REAL A3B
shapes, on whichever arch it runs on, and emits a machine-readable
``SCOREBOARD_JSON`` line per op so a driver can tabulate the parity ratio
``gfx950_us / gfx1250_us``.

Run the SAME invocation on each arch (gfx950 local, gfx1250 remote):

  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.qwen3_30b_a3b.a3b_parity_scoreboard \
      --arch gfx950  --op moe --preset decode
  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.qwen3_30b_a3b.a3b_parity_scoreboard \
      --arch gfx1250 --op moe --preset decode

Ops are an extensible registry; ``moe`` is implemented first because it is the
dominant (~60%) decode cost and is the only op with a single-launch kernel on
both arches today (gfx950 MFMA mega, gfx1250 WMMA mega). attention / gemm /
norm / routing rows can be added behind the same timing + accounting frame.

A3B geometry (from qwen3_30b_a3b_shapes): hidden H=2048, moe_intermediate
I=768, experts=128, topk=8, batch=2 -> active (token,expert) pairs = 16, so the
fused MoE M dimension is one tile_m=16 block per distinct expert. ``decode``
times one such block (the irreducible per-expert-block unit); ``prefill`` scales
M up so the op enters the compute-bound regime.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct

# A3B fixtures (kept import-light; values mirror qwen3_30b_a3b_shapes.py).
A3B_HIDDEN = 2048  # H  (gate/up contraction K, down output H_out)
A3B_INTER = 768  # I   (gate/up output N, down contraction)
TILE_M = 16
ELEM_BYTES_BF16 = 2


def _build_moe(arch: str, dtype: str, tile_n_inter: int, tile_n_down: int):
    """Return (spec, kernel_def, grid_fn, block_size) for the arch's MoE mega."""
    if arch == "gfx1250":
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            FusedMegaWmmaSpec,
            build_moe_fused_mega_wmma,
            moe_fused_mega_wmma_grid,
        )

        spec = FusedMegaWmmaSpec(
            name=f"sb_{arch}",
            dtype=dtype,
            tile_n_inter=tile_n_inter,
            tile_n_down=tile_n_down,
        )
        return (
            spec,
            build_moe_fused_mega_wmma(spec, arch=arch),
            moe_fused_mega_wmma_grid,
            spec.block_size,
        )
    from rocke.instances.common.moe_fused_mega import (
        FusedMegaKernelSpec,
        build_moe_fused_mega_gemm,
        moe_fused_mega_grid,
    )

    spec = FusedMegaKernelSpec(
        name=f"sb_{arch}",
        dtype=dtype,
        tile_n_inter=tile_n_inter,
        tile_n_down=tile_n_down,
    )
    return (
        spec,
        build_moe_fused_mega_gemm(spec, arch=arch),
        moe_fused_mega_grid,
        spec.block_size,
    )


def _run_moe(args) -> dict:
    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    H = args.hidden  # K (gate/up contraction)
    I = args.inter  # N (gate/up out, down contraction)  # noqa: E741
    H_out = H  # down output
    num_m_blocks = args.m_blocks
    M = num_m_blocks * TILE_M
    E = max(1, num_m_blocks)  # one distinct expert per block (decode realism)
    elem_b = ELEM_BYTES_BF16

    spec, kdef, grid_fn, block_size = _build_moe(
        args.arch, args.dtype, args.tile_n_inter, args.tile_n_down
    )
    art = compile_kernel(kdef, arch=args.arch)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    block_expert_ids = (np.arange(num_m_blocks) % E).astype(np.int32)
    sorted_token_ids = np.arange(M, dtype=np.int32)
    sorted_weights = np.ones(M, np.float32)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    a_nb = M * H * elem_b
    wg_nb = E * I * H * elem_b
    wd_nb = E * H_out * I * elem_b
    y_nb = M * H_out * 4

    ad = rt.alloc(a_nb)
    wgd = rt.alloc(wg_nb)
    wud = rt.alloc(wg_nb)
    wdd = rt.alloc(wd_nb)
    stid = rt.alloc(sorted_token_ids.nbytes)
    swd = rt.alloc(sorted_weights.nbytes)
    beid = rt.alloc(block_expert_ids.nbytes)
    yd = rt.alloc(y_nb)
    for ptr, nb in ((ad, a_nb), (wgd, wg_nb), (wud, wg_nb), (wdd, wd_nb), (yd, y_nb)):
        rt.memset(ptr, 0, nb)
    rt.memcpy_h2d(stid, u8(sorted_token_ids), sorted_token_ids.nbytes)
    rt.memcpy_h2d(swd, u8(sorted_weights), sorted_weights.nbytes)
    rt.memcpy_h2d(beid, u8(block_expert_ids), block_expert_ids.nbytes)

    stride_b_gate = I * H
    stride_b_down = H_out * I
    packed = struct.pack(
        "<8Q10i",
        ad,
        wgd,
        wud,
        wdd,
        stid,
        swd,
        beid,
        yd,
        M,
        I,
        H,
        H_out,
        0,
        stride_b_gate,
        stride_b_gate,
        stride_b_down,
        0,
        M,
    )
    grid = grid_fn(num_m_blocks, I, spec)
    block = (block_size, 1, 1)
    grid_x = grid[0]

    for _ in range(args.warmup):
        rt.launch(fn, grid, block, packed)
    rt.sync()
    start = rt.event()
    end = rt.event()
    start.record()
    for _ in range(args.iters):
        rt.launch(fn, grid, block, packed)
    end.record()
    end.synchronize()
    us = (start.elapsed_to(end) / args.iters) * 1e3
    rt.sync()
    for ptr in (ad, wgd, wud, wdd, stid, swd, beid, yd):
        rt.free(ptr)
    module.unload()

    # FLOPs: gate + up + down (silu negligible).
    flops = 2.0 * M * (2.0 * I * H + H_out * I)
    # Modeled HBM traffic: per m-block one expert's gate+up+down weights are
    # read once (split across grid.x); + activations in, + atomic Y out.
    weight_bytes = num_m_blocks * (2 * I * H + H_out * I) * elem_b
    act_bytes = M * H * elem_b
    out_bytes = grid_x * M * H_out * 4
    total_bytes = weight_bytes + act_bytes + out_bytes
    gbps = total_bytes / (us * 1e-6) / 1e9
    tflops = flops / (us * 1e-6) / 1e12

    row = {
        "op": "moe",
        "arch": args.arch,
        "dtype": args.dtype,
        "preset": args.preset,
        "M": M,
        "m_blocks": num_m_blocks,
        "H": H,
        "I": I,
        "grid": list(grid),
        "block": block_size,
        "us": round(us, 3),
        "tflops": round(tflops, 2),
        "gbps": round(gbps, 1),
        "weight_MB": round(weight_bytes / 1e6, 2),
        "total_MB": round(total_bytes / 1e6, 2),
        "kernel": art.kernel_name,
    }
    if args.hbm_gbps:
        row["pct_peak_bw"] = round(100.0 * gbps / args.hbm_gbps, 1)
    return row


_PRESETS = {
    # decode: one expert-block (irreducible per-expert unit, memory-bound).
    "decode": dict(m_blocks=1),
    # prefill: M large enough to enter the compute-bound regime.
    "prefill": dict(m_blocks=8),
}

# ---------------------------------------------------------------------------
# OP: gemm  (the dense decode/prefill projections: qkv, o_proj, router)
# ---------------------------------------------------------------------------

# decode_gemm_shapes()/prefill_gemm_shapes() return a 3-tuple in this order.
_GEMM_SHAPE_IDX = {"qkv": 0, "o": 1, "router": 2}

# prefill packs 128 tokens into the M dimension (the value the A3B fixtures use).
_PREFILL_TOTAL_TOKENS = 128

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8e4m3": 1, "bf8e5m2": 1}


def _a3b_gemm_shape(gemm_shape: str, preset: str):
    """Resolve one A3B dense-GEMM shape (qkv|o|router) for decode|prefill."""
    from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
        decode_gemm_shapes,
        prefill_gemm_shapes,
    )

    idx = _GEMM_SHAPE_IDX[gemm_shape]
    if preset == "prefill":
        return prefill_gemm_shapes(total_tokens=_PREFILL_TOTAL_TOKENS)[idx]
    return decode_gemm_shapes()[idx]


def _build_gemm_spec(arch: str, shape, dtype: str):
    """Universal-GEMM spec for ``shape`` on ``arch`` (wave32 WMMA / wave64 MFMA).

    gfx1250 reuses the canonical A3B wave32 spec (tile 16x16x32, ``mem``
    pipeline). gfx950 builds the analogous wave64 MFMA spec with the same
    16x16x32 tile + ``mem``/``default`` epilogue so the only difference across
    arches is the lane geometry (validated with ``is_valid_spec``).
    """
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        is_valid_spec,
    )

    if arch == "gfx1250":
        from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
            bf16_universal_gemm_spec,
        )

        spec = bf16_universal_gemm_spec(shape)
    else:
        spec = UniversalGemmSpec(
            name=f"sb_{arch}_{shape.name}",
            tile=TileSpec(
                tile_m=16,
                tile_n=16,
                tile_k=32,
                warp_m=1,
                warp_n=1,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            trait=TraitSpec(
                pipeline="mem",
                epilogue="default",
                pad_m=True,
                pad_n=True,
                pad_k=True,
            ),
            data=DataSpec(
                dtype_a=dtype,
                dtype_b=dtype,
                dtype_c=dtype,
                dtype_acc="fp32",
                layout=shape.layout,
            ),
            wave_size=64,
        )
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise SystemExit(f"[{arch}] invalid GEMM spec for {shape.name}: {why}")
    return spec


def _run_gemm(args) -> dict:
    """Time one A3B dense GEMM via the run_manifest subprocess flow.

    Mirrors ``universal_gemm_verify``: build -> compile_kernel -> manifest ->
    ``python -m rocke.run_manifest <hsaco> <manifest> --shape M,N,K`` and parse
    the ``Perf: <ms> ms, <tflops> TFlops`` line. This avoids hand-packing the
    GEMM kernel ABI. We then layer the same modeled-traffic accounting the MoE
    row uses on top of the measured latency.
    """
    import re
    import subprocess
    import sys
    from pathlib import Path

    from rocke.helpers import compile_kernel, make_gemm_manifest, write_artifact
    from rocke.instances.common.gemm_universal import build_universal_gemm

    shape = _a3b_gemm_shape(args.gemm_shape, args.preset)
    M, N, K = shape.M, shape.N, shape.K
    spec = _build_gemm_spec(args.arch, shape, args.dtype)
    tile = spec.tile

    art = compile_kernel(build_universal_gemm(spec, arch=args.arch), arch=args.arch)

    out = Path(f"/tmp/a3b_sb_gemm_{args.arch}_{args.gemm_shape}_{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    atom_family = "wmma" if spec.wave_size == 32 else "mfma"
    manifest = make_gemm_manifest(
        artifact=art,
        block_m=tile.tile_m,
        block_n=tile.tile_n,
        block_k=tile.tile_k,
        threads_per_block=spec.block_size,
        default_shape=(M, N, K),
        warmup_iters=args.warmup,
        timed_iters=args.iters,
        atoms=[
            f"{atom_family}_f32_{tile.warp_tile_m}x{tile.warp_tile_n}"
            f"x{tile.warp_tile_k}_{spec.data.dtype_a}"
        ],
    )
    write_artifact(art, out, manifest)

    cmd = [
        sys.executable,
        "-m",
        "rocke.run_manifest",
        str(out / f"{art.kernel_name}.hsaco"),
        str(out / "manifest.json"),
        "--shape",
        f"{M},{N},{K}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout)
    m = re.search(r"Perf:\s*([0-9.eE+-]+)\s*ms,\s*([0-9.eE+-]+)\s*TFlops", r.stdout)
    if m is None:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(
            f"[{args.arch}] run_manifest produced no Perf line for {shape.name} "
            f"(rc={r.returncode})"
        )
    ms = float(m.group(1))
    tflops = float(m.group(2))
    us = ms * 1e3

    # Modeled HBM traffic at the GEMM I/O boundary (RCR: A=MxK, B=NxK, C=MxN).
    ab_bytes = _DTYPE_BYTES.get(spec.data.dtype_a, 2)
    c_bytes = _DTYPE_BYTES.get(spec.data.dtype_c, 2)
    total_bytes = (M * K + N * K) * ab_bytes + M * N * c_bytes
    flops = 2.0 * M * N * K  # noqa: F841
    gbps = total_bytes / (us * 1e-6) / 1e9

    row = {
        "op": "gemm",
        "gemm_shape": args.gemm_shape,
        "arch": args.arch,
        "dtype": args.dtype,
        "preset": args.preset,
        "M": M,
        "N": N,
        "K": K,
        "grid": [
            (N + tile.tile_n - 1) // tile.tile_n,
            (M + tile.tile_m - 1) // tile.tile_m,
        ],
        "block": spec.block_size,
        "us": round(us, 3),
        "tflops": round(tflops, 2),
        "gbps": round(gbps, 1),
        "weight_MB": round(N * K * ab_bytes / 1e6, 2),
        "total_MB": round(total_bytes / 1e6, 2),
        "kernel": art.kernel_name,
        "timed": True,
    }
    if args.hbm_gbps:
        row["pct_peak_bw"] = round(100.0 * gbps / args.hbm_gbps, 1)
    return row


# ---------------------------------------------------------------------------
# OP: attention  (paged decode attention; split-KV 3D segment + reduce)
# ---------------------------------------------------------------------------


def _run_attention(args) -> dict:
    """Time (or, if the launch is unavailable, build+lower) decode attention.

    Models the build on ``examples/gfx1250/attention/decode_3d_verify.py``: it
    resolves the arch's split-KV 3D segment + reduce instances through the
    unified dispatcher, builds + lowers + assembles both kernels (proving the
    op is functional on the requested arch), and then attempts a real two-kernel
    timed launch. If the launch path raises, the row is emitted with
    ``timed=false`` and a ``note`` rather than a fragile partial measurement.
    """
    import numpy as np

    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers import compile_kernel
    from rocke.instances.common import attention_unified as au

    HD, NQH, NKVH, BS = 64, 32, 4, 16
    NQK = NQH // NKVH
    num_seqs = 2
    NUM_SEG = 16
    kv_len = args.kv_len
    use_sinks = True
    scale = float(HD**-0.5)
    k_scale = v_scale = 1.0
    total_q = num_seqs  # q_len == 1
    wave_size = 32 if args.arch == "gfx1250" else 64

    au._RESOLVED_ATTENTION_ARCH = args.arch
    problem = au.UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=NQH,
        num_kv_heads=NKVH,
        head_size=HD,
        block_size=BS,
        max_seqlen_q=1,
        max_seqlen_k=kv_len,
        dtype="bf16",
        q_dtype="bf16",
        sliding_window=0,
        use_sinks=use_sinks,
        use_fp8=False,
    )
    ok, why = au.supports_native_unified_attention_3d_tiled(problem)
    if not ok:
        raise SystemExit(f"[{args.arch}] decode3d UNSUPPORTED: {why}")

    Spec3D, ReduceSpec, build_seg, build_red, _ = au._tiled_3d_impl(args.arch)
    base_seg_spec = au._tiled_3d_spec_from_problem(problem)
    from dataclasses import replace as _replace

    seg_spec = _replace(base_seg_spec, num_segments=NUM_SEG)
    red_spec = ReduceSpec(
        head_size=HD,
        num_query_heads=NQH,
        num_kv_heads=NKVH,
        dtype="bf16",
        num_segments=NUM_SEG,
    )
    seg_kdef = build_seg(seg_spec, arch=args.arch)
    red_kdef = build_red(red_spec, arch=args.arch)
    # Build + lower (functional proof) before assembling to a code object.
    lower_kernel_to_llvm(seg_kdef, arch=args.arch)
    lower_kernel_to_llvm(red_kdef, arch=args.arch)
    seg_art = compile_kernel(seg_kdef, arch=args.arch)
    red_art = compile_kernel(red_kdef, arch=args.arch)

    base_row = {
        "op": "attention",
        "arch": args.arch,
        "dtype": "bf16",
        "preset": args.preset,
        "kv_len": kv_len,
        "batch": num_seqs,
        "nhead_q": NQH,
        "nhead_k": NKVH,
        "head_dim": HD,
        "block_size": BS,
        "num_segments": NUM_SEG,
        "seg_kernel": seg_art.kernel_name,
        "reduce_kernel": red_art.kernel_name,
    }

    try:
        import ctypes
        import struct

        from rocke.runtime.hip_module import Runtime

        rng = np.random.default_rng(0xD3C0)
        BLOCK_Q = 16 // NQK
        max_blocks = (kv_len + BS - 1) // BS
        num_blocks = max_blocks * num_seqs + 4
        seq_lens_np = np.full(num_seqs, kv_len, dtype=np.int32)
        cu_q = np.arange(num_seqs + 1, dtype=np.int32)
        block_tables = np.zeros((num_seqs, max_blocks), dtype=np.int32)
        for i in range(num_seqs):
            block_tables[i] = rng.permutation(num_blocks)[:max_blocks]

        rt = Runtime()
        seg_mod = rt.load_module(seg_art.hsaco)
        seg_fn = seg_mod.get_function(seg_art.kernel_name)
        red_mod = rt.load_module(red_art.hsaco)
        red_fn = red_mod.get_function(red_art.kernel_name)

        def u8(a):
            a = np.ascontiguousarray(a)
            return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

        def alloc_copy(a):
            a = np.ascontiguousarray(a)
            d = rt.alloc(max(1, int(a.nbytes)))
            if a.nbytes:
                rt.memcpy_h2d(d, u8(a), a.nbytes)
            return d

        elem = 2  # bf16
        qd = rt.alloc(total_q * NQH * HD * elem)
        kd = rt.alloc(num_blocks * BS * NKVH * HD * elem)
        vd = rt.alloc(num_blocks * BS * NKVH * HD * elem)
        od = rt.alloc(total_q * NQH * HD * elem)
        sink_d = rt.alloc(2 * NQH)
        alibi_d = rt.alloc(4 * NQH)
        qq_d = rt.alloc(4)
        for ptr, nb in (
            (qd, total_q * NQH * HD * elem),
            (kd, num_blocks * BS * NKVH * HD * elem),
            (vd, num_blocks * BS * NKVH * HD * elem),
            (od, total_q * NQH * HD * elem),
            (sink_d, 2 * NQH),
            (alibi_d, 4 * NQH),
            (qq_d, 4),
        ):
            rt.memset(ptr, 0, nb)
        bt_d = alloc_copy(block_tables)
        sl_d = alloc_copy(seq_lens_np)
        cuq_d = alloc_copy(cu_q)

        segm_out_n = total_q * NQH * NUM_SEG * HD
        segm_ml_n = total_q * NQH * NUM_SEG
        segm_out_d = rt.alloc(4 * segm_out_n)
        segm_max_d = rt.alloc(4 * segm_ml_n)
        segm_exp_d = rt.alloc(4 * segm_ml_n)

        total_num_q_blocks = total_q // BLOCK_Q + num_seqs
        seg_grid = (int(total_num_q_blocks), int(NKVH), int(NUM_SEG))
        red_grid = (int(total_q), int(NQH), 1)
        blk = (wave_size, 1, 1)

        seg_packed = struct.pack(
            "<" + "Q" * 12 + "f" * 4 + "i" * 3,
            segm_out_d,
            segm_max_d,
            segm_exp_d,
            qd,
            kd,
            vd,
            sink_d,
            bt_d,
            sl_d,
            alibi_d,
            qq_d,
            cuq_d,
            scale,
            k_scale,
            v_scale,
            0.0,
            num_seqs,
            int(block_tables.shape[1]),
            0,
        )
        red_packed = struct.pack(
            "<" + "Q" * 5, od, segm_out_d, segm_max_d, segm_exp_d, sl_d
        )

        for _ in range(args.warmup):
            rt.launch(seg_fn, seg_grid, blk, seg_packed)
            rt.launch(red_fn, red_grid, blk, red_packed)
        rt.sync()
        ev0, ev1 = rt.event(), rt.event()
        ev0.record()
        for _ in range(args.iters):
            rt.launch(seg_fn, seg_grid, blk, seg_packed)
            rt.launch(red_fn, red_grid, blk, red_packed)
        ev1.record()
        ev1.synchronize()
        us = ev0.elapsed_to(ev1) * 1e3 / args.iters
        rt.sync()

        for ptr in (
            qd,
            kd,
            vd,
            od,
            sink_d,
            alibi_d,
            qq_d,
            bt_d,
            sl_d,
            cuq_d,
            segm_out_d,
            segm_max_d,
            segm_exp_d,
        ):
            rt.free(ptr)
        seg_mod.unload()
        red_mod.unload()

        # FLOPs: QK^T + PV, GQA-expanded over all query heads (2 * 2 * ...).
        flops = 2.0 * 2.0 * num_seqs * NQH * kv_len * HD
        # Modeled traffic: KV cache read dominates decode attention.
        kv_bytes = 2.0 * num_seqs * kv_len * NKVH * HD * elem
        q_bytes = total_q * NQH * HD * elem
        out_bytes = total_q * NQH * HD * elem
        total_bytes = kv_bytes + q_bytes + out_bytes
        gbps = total_bytes / (us * 1e-6) / 1e9
        tflops = flops / (us * 1e-6) / 1e12

        base_row.update(
            {
                "grid": list(seg_grid),
                "block": wave_size,
                "us": round(us, 3),
                "tflops": round(tflops, 2),
                "gbps": round(gbps, 1),
                "total_MB": round(total_bytes / 1e6, 2),
                "timed": True,
            }
        )
        if args.hbm_gbps:
            base_row["pct_peak_bw"] = round(100.0 * gbps / args.hbm_gbps, 1)
        return base_row
    except Exception as e:  # noqa: BLE001 - best-effort timed launch
        base_row.update(
            {
                "timed": False,
                "note": (
                    "built+lowered+assembled split-KV 3D segment+reduce kernels "
                    f"for {args.arch}; timed launch unavailable: {type(e).__name__}: {e}"
                ),
            }
        )
        return base_row


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=("gfx950", "gfx1250"))
    p.add_argument("--op", default="moe", choices=("moe", "gemm", "attention"))
    p.add_argument("--preset", default="decode", choices=tuple(_PRESETS))
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument(
        "--gemm-shape",
        default="qkv",
        choices=tuple(_GEMM_SHAPE_IDX),
        help="dense GEMM to time when --op gemm (qkv|o|router)",
    )
    p.add_argument(
        "--kv-len",
        type=int,
        default=1024,
        help="decode attention KV length when --op attention",
    )
    p.add_argument("--hidden", type=int, default=A3B_HIDDEN)
    p.add_argument("--inter", type=int, default=A3B_INTER)
    p.add_argument("--tile-n-inter", type=int, default=256)
    p.add_argument("--tile-n-down", type=int, default=256)
    p.add_argument("--m-blocks", type=int, default=None, help="override preset")
    p.add_argument(
        "--hbm-gbps", type=float, default=None, help="peak HBM GB/s for %peak"
    )
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    if args.m_blocks is None:
        args.m_blocks = _PRESETS[args.preset]["m_blocks"]

    if args.op == "moe":
        row = _run_moe(args)
        print(
            f"[{row['arch']}] op={row['op']} {row['dtype']} preset={row['preset']} "
            f"M{row['M']} H{row['H']} I{row['I']} grid={tuple(row['grid'])}: "
            f"{row['us']:.2f} us  {row['tflops']:.2f} TFLOPs  {row['gbps']:.1f} GB/s  "
            f"(weights {row['weight_MB']:.1f}MB / {row['total_MB']:.1f}MB)"
            + (f"  {row['pct_peak_bw']:.1f}% peak BW" if "pct_peak_bw" in row else "")
        )
    elif args.op == "gemm":
        row = _run_gemm(args)
        print(
            f"[{row['arch']}] op={row['op']}:{row['gemm_shape']} {row['dtype']} "
            f"preset={row['preset']} M{row['M']} N{row['N']} K{row['K']} "
            f"grid={tuple(row['grid'])}: "
            f"{row['us']:.2f} us  {row['tflops']:.2f} TFLOPs  {row['gbps']:.1f} GB/s  "
            f"(B {row['weight_MB']:.1f}MB / {row['total_MB']:.1f}MB)"
            + (f"  {row['pct_peak_bw']:.1f}% peak BW" if "pct_peak_bw" in row else "")
        )
    elif args.op == "attention":
        row = _run_attention(args)
        if row.get("timed"):
            print(
                f"[{row['arch']}] op={row['op']} {row['dtype']} preset={row['preset']} "
                f"kv_len={row['kv_len']} grid={tuple(row['grid'])}: "
                f"{row['us']:.2f} us  {row['tflops']:.2f} TFLOPs  "
                f"{row['gbps']:.1f} GB/s  ({row['total_MB']:.1f}MB)"
                + (
                    f"  {row['pct_peak_bw']:.1f}% peak BW"
                    if "pct_peak_bw" in row
                    else ""
                )
            )
        else:
            print(
                f"[{row['arch']}] op={row['op']} {row['dtype']} preset={row['preset']} "
                f"kv_len={row['kv_len']} seg={row['seg_kernel']} "
                f"reduce={row['reduce_kernel']}: NOT TIMED ({row['note']})"
            )
    else:  # pragma: no cover - argparse choices guard this
        raise SystemExit(f"unknown op {args.op!r}")
    print("SCOREBOARD_JSON " + json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
