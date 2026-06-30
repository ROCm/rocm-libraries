# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1250 WMMA split-KV 3D decode attention.

Numpy-only harness (no torch) for the gfx1250 box: builds the split-KV segment
kernel + reduce kernel (instances/gfx1250/attention_tiled_3d.py), launches both
with an fp32 partials workspace, and compares the merged output against a numpy
paged decode-attention reference (GQA, causal, optional sinks, fp8/bf16 KV).

Targets the Qwen3-30B-A3B decode contract: batch=2, 32 q-heads / 4 kv-heads
(GQA-8), head_dim=64, block_size=16, q_len=1, kv_len in {512,1024,2048,4096}.

    export PATH=/opt/rocm/bin:$PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=/tmp/rocke_gfx1250/python
    HIP_VISIBLE_DEVICES=3 python -m \\
        rocke.examples.gfx1250.attention.decode_3d_verify --kv-len 1024 --kv-dtype fp8e4m3
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import struct

from dataclasses import replace
import pathlib

import numpy as np

from rocke.helpers import compile_kernel
from rocke.instances.common import attention_unified as au
from rocke.runtime.hip_module import Runtime, get_device_arch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
    _FP8 = ml_dtypes.float8_e4m3fn
except Exception as e:  # pragma: no cover
    raise SystemExit("ml_dtypes required; pip install ml_dtypes into the venv") from e

_HD = 64
_NQH = 32
_NKVH = 4
_NQK = _NQH // _NKVH
_BS = 16
_BLOCK_Q = _BLOCK_M_DIV = 16 // _NQK  # = 2


def _ref_decode(q_f32, kc_f32, vc_f32, *, block_tables, seq_lens, scale, sinks_f32):
    num_seqs = len(seq_lens)
    out = np.zeros((num_seqs, _NQH, _HD), dtype=np.float32)
    for i in range(num_seqs):
        sl = int(seq_lens[i])
        nblk = (sl + _BS - 1) // _BS
        idx = block_tables[i, :nblk]
        k = kc_f32[idx].reshape(-1, _NKVH, _HD)[:sl]
        v = vc_f32[idx].reshape(-1, _NKVH, _HD)[:sl]
        k = np.repeat(k, _NQK, axis=1)  # [sl, NQH, HD]
        v = np.repeat(v, _NQK, axis=1)
        qi = q_f32[i]  # [NQH, HD]
        scores = np.einsum("hd,khd->hk", qi, k) * scale  # [NQH, sl]
        if sinks_f32 is not None:
            scores = np.concatenate([scores, sinks_f32[:, None]], axis=-1)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        if sinks_f32 is not None:
            p = p[..., :-1]
        out[i] = np.einsum("hk,khd->hd", p, v)
    return out


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arch", default=None, help="default: auto-detect the local device"
    )
    ap.add_argument("--kv-len", type=int, default=1024)
    ap.add_argument("--num-seqs", type=int, default=2)
    ap.add_argument("--num-segments", type=int, default=16)
    ap.add_argument("--kv-dtype", default="fp8e4m3", choices=("bf16", "fp8e4m3"))
    ap.add_argument("--no-sinks", action="store_true")
    ap.add_argument("--tol", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0xD3C0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument(
        "--bench-split",
        action="store_true",
        help="with --bench: also print mean segment-only and reduce-only latency (us)",
    )
    ap.add_argument(
        "--csv",
        metavar="PATH",
        help="append one CSV row with timing / correctness summary",
    )
    ap.add_argument("--register-p", action="store_true")
    ap.add_argument("--no-register-p", action="store_true")
    ap.add_argument("--wide-kv-load", action="store_true")
    ap.add_argument("--no-wide-kv-load", action="store_true")
    ap.add_argument(
        "--dtla",
        action="store_true",
        help="force DTLA: async global->LDS V staging + double-buffer + prefetch",
    )
    ap.add_argument("--no-dtla", action="store_true")
    ap.add_argument(
        "--ds-tr",
        action="store_true",
        help="HW transpose-LDS read (ds_load_tr16_b128) for the PV V operand",
    )
    ap.add_argument("--no-ds-tr", action="store_true")
    ap.add_argument(
        "--sw-pipeline",
        action="store_true",
        help="software-pipeline stack: DTLA async-V shadow + iglp/sched cadence",
    )
    ap.add_argument(
        "--ablate-softmax",
        action="store_true",
        help="DEBUG perf-only: drop softmax wave-reduce+exp2 (output garbage)",
    )
    ap.add_argument(
        "--ablate-pv",
        action="store_true",
        help="DEBUG perf-only: drop V staging + PV-GEMM (output garbage); "
        "measures the PV+V-staging ceiling. Implies --plain.",
    )
    ap.add_argument(
        "--plain",
        action="store_true",
        help="force the plain stage_v_tile + compute_pv path (no wide-LDS-reads "
        "or other V levers); matches the ablate-pv baseline",
    )
    ap.add_argument(
        "--dpp-softmax",
        action="store_true",
        help="softmax cross-lane reduction via DPP row_xmask (VALU) instead of "
        "ds_swizzle (LDS port); default on for gfx1250 3D decode",
    )
    ap.add_argument(
        "--no-dpp-softmax",
        action="store_true",
        help="force the ds_swizzle softmax reduction (disable the DPP default)",
    )
    ap.add_argument("--invariant-hoist", action="store_true")
    ap.add_argument("--no-invariant-hoist", action="store_true")
    ap.add_argument("--wmma-spacing", type=int, default=None)
    ap.add_argument(
        "--num-waves",
        type=int,
        default=None,
        help="cooperative multi-wave32 CTA: waves per (q-block, kv-head, segment)",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    if args.register_p and args.no_register_p:
        ap.error("--register-p and --no-register-p are mutually exclusive")
    if args.wide_kv_load and args.no_wide_kv_load:
        ap.error("--wide-kv-load and --no-wide-kv-load are mutually exclusive")
    if args.dtla and args.no_dtla:
        ap.error("--dtla and --no-dtla are mutually exclusive")
    if args.invariant_hoist and args.no_invariant_hoist:
        ap.error("--invariant-hoist and --no-invariant-hoist are mutually exclusive")

    arch = args.arch or get_device_arch()
    num_seqs = args.num_seqs
    NUM_SEG = args.num_segments
    use_sinks = not args.no_sinks
    kv_is_fp8 = args.kv_dtype == "fp8e4m3"
    seq_lens = [args.kv_len] * num_seqs
    total_q = num_seqs  # q_len == 1
    max_blocks = (max(seq_lens) + _BS - 1) // _BS
    num_blocks = max_blocks * num_seqs + 4
    scale = float(_HD**-0.5)
    k_scale = v_scale = 1.0
    wave_size = 32 if arch == "gfx1250" else 64

    # Build the split-KV 3D segment + reduce kernels through the unified
    # dispatcher's per-arch instance so the same harness compares gfx950 (MFMA
    # wave64) and gfx1250 (WMMA wave32) on identical shapes / ABIs.
    au._RESOLVED_ATTENTION_ARCH = arch
    problem = au.UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=_NQH,
        num_kv_heads=_NKVH,
        head_size=_HD,
        block_size=_BS,
        max_seqlen_q=1,
        max_seqlen_k=args.kv_len,
        dtype="bf16",
        q_dtype="bf16",
        sliding_window=0,
        use_sinks=use_sinks,
        use_fp8=kv_is_fp8,
    )
    ok, why = au.supports_native_unified_attention_3d_tiled(problem)
    if not ok:
        print(f"[{arch}] decode3d UNSUPPORTED: {why}")
        return 2
    Spec3D, ReduceSpec, build_seg, build_red, _ = au._tiled_3d_impl(arch)
    seg_updates = {"num_segments": NUM_SEG}
    base_seg_spec = au._tiled_3d_spec_from_problem(problem)
    if hasattr(base_seg_spec, "use_register_p"):
        if args.register_p:
            seg_updates["use_register_p"] = True
        elif args.no_register_p:
            seg_updates["use_register_p"] = False
    if hasattr(base_seg_spec, "use_wide_kv_load"):
        if args.wide_kv_load:
            seg_updates["use_wide_kv_load"] = True
        elif args.no_wide_kv_load:
            seg_updates["use_wide_kv_load"] = False
    if hasattr(base_seg_spec, "use_invariant_hoist"):
        if args.invariant_hoist:
            seg_updates["use_invariant_hoist"] = True
        elif args.no_invariant_hoist:
            seg_updates["use_invariant_hoist"] = False
    if hasattr(base_seg_spec, "wmma_spacing") and args.wmma_spacing is not None:
        seg_updates["wmma_spacing"] = args.wmma_spacing
    if hasattr(base_seg_spec, "num_waves") and args.num_waves is not None:
        seg_updates["num_waves"] = args.num_waves
        if args.num_waves > 1:
            # multi-wave uses the LDS-P single-buffer path; the default
            # wide-LDS-reads is incompatible.
            seg_updates["use_wide_lds_reads"] = False
    if hasattr(base_seg_spec, "use_dtla_prefetch"):
        if args.dtla:
            # DTLA owns the V_lds layout; it is mutually exclusive with the
            # default wide-LDS-reads path (token-major vs dim-major V).
            seg_updates["use_dtla_prefetch"] = True
            seg_updates["use_wide_lds_reads"] = False
        elif args.no_dtla:
            seg_updates["use_dtla_prefetch"] = False
    if hasattr(base_seg_spec, "use_ds_tr_reads"):
        if args.ds_tr:
            # HW transpose read of token-major V; mutually exclusive with the
            # default dim-major wide-LDS-reads path.
            seg_updates["use_ds_tr_reads"] = True
            seg_updates["use_wide_lds_reads"] = False
        elif args.no_ds_tr:
            seg_updates["use_ds_tr_reads"] = False
    if hasattr(base_seg_spec, "use_sw_pipeline") and args.sw_pipeline:
        seg_updates["use_sw_pipeline"] = True
        seg_updates["use_wide_lds_reads"] = False
    if hasattr(base_seg_spec, "ablate_softmax") and args.ablate_softmax:
        seg_updates["ablate_softmax"] = True
    if (args.plain or args.ablate_pv) and hasattr(base_seg_spec, "use_wide_lds_reads"):
        # plain stage_v_tile + compute_pv path: drop the wide-LDS-read V lever
        # so the baseline and the ablate-pv run share the same V path.
        seg_updates["use_wide_lds_reads"] = False
    if hasattr(base_seg_spec, "ablate_pv") and args.ablate_pv:
        seg_updates["ablate_pv"] = True
    if hasattr(base_seg_spec, "use_dpp_softmax"):
        if args.dpp_softmax:
            seg_updates["use_dpp_softmax"] = True
        elif args.no_dpp_softmax:
            seg_updates["use_dpp_softmax"] = False
    seg_spec = replace(base_seg_spec, **seg_updates)
    red_spec = ReduceSpec(
        head_size=_HD,
        num_query_heads=_NQH,
        num_kv_heads=_NKVH,
        dtype="bf16",
        num_segments=NUM_SEG,
    )
    import os as _os

    _seg_compile = compile_kernel
    if _os.environ.get("ROCKE_DECODE3D_HIPCC") == "1":
        from rocke.helpers.compile import compile_kernel_via_hipcc as _seg_compile
    seg_art = _seg_compile(build_seg(seg_spec, arch=arch), arch=arch)
    red_art = compile_kernel(build_red(red_spec, arch=arch), arch=arch)
    print(
        f"[{arch}] wave{wave_size} seg={seg_art.kernel_name} ({seg_art.hsaco_bytes}B) "
        f"reduce={red_art.kernel_name} ({red_art.hsaco_bytes}B)"
    )

    rng = np.random.default_rng(args.seed)
    q_f32 = (rng.standard_normal((total_q, _NQH, _HD)) * 0.3).astype(np.float32)
    if kv_is_fp8:
        kc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
        vc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
        kc_f32 = kc.astype(np.float32) * k_scale
        vc_f32 = vc.astype(np.float32) * v_scale
    else:
        kc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_BF16)
        vc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_BF16)
        kc_f32 = kc.astype(np.float32)
        vc_f32 = vc.astype(np.float32)

    q_bf16 = q_f32.astype(_BF16)
    out = np.zeros((total_q, _NQH, _HD), dtype=_BF16)
    cu_q = np.arange(num_seqs + 1, dtype=np.int32)  # q_len == 1
    seq_lens_np = np.array(seq_lens, dtype=np.int32)
    block_tables = np.zeros((num_seqs, max_blocks), dtype=np.int32)
    for i in range(num_seqs):
        block_tables[i] = rng.permutation(num_blocks)[:max_blocks]
    sinks_bf16 = (rng.standard_normal(_NQH) * 0.5).astype(_BF16) if use_sinks else None
    sinks_f32 = sinks_bf16.astype(np.float32) if use_sinks else None

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

    qd = alloc_copy(q_bf16)
    kd = alloc_copy(kc)
    vd = alloc_copy(vc)
    od = rt.alloc(out.nbytes)
    rt.memset(od, 0, out.nbytes)
    sink_d = alloc_copy(sinks_bf16) if use_sinks else rt.alloc(2 * _NQH)
    bt_d = alloc_copy(block_tables)
    sl_d = alloc_copy(seq_lens_np)
    alibi_d = rt.alloc(4 * _NQH)
    qq_d = rt.alloc(4)
    cuq_d = alloc_copy(cu_q)

    # fp32 partials workspace
    segm_out_n = total_q * _NQH * NUM_SEG * _HD
    segm_ml_n = total_q * _NQH * NUM_SEG
    segm_out_d = rt.alloc(4 * segm_out_n)
    segm_max_d = rt.alloc(4 * segm_ml_n)
    segm_exp_d = rt.alloc(4 * segm_ml_n)
    import os as _os

    if _os.environ.get("ROCKE_DECODE3D_ZERO_WS") == "1":
        rt.memset(segm_out_d, 0, 4 * segm_out_n)
        rt.memset(segm_max_d, 0, 4 * segm_ml_n)
        rt.memset(segm_exp_d, 0, 4 * segm_ml_n)
        rt.sync()

    block_q = _BLOCK_Q
    total_num_q_blocks = total_q // block_q + num_seqs
    seg_grid = (int(total_num_q_blocks), int(_NKVH), int(NUM_SEG))
    seg_waves = int(getattr(seg_spec, "num_waves", 1))
    seg_blk = (wave_size * seg_waves, 1, 1)
    red_blk = (wave_size, 1, 1)

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
    red_grid = (int(total_q), int(_NQH), 1)

    if _os.environ.get("ROCKE_DECODE3D_PROBE_WS") == "1":
        rt.memset(segm_max_d, 0xFF, 4 * segm_ml_n)  # 0xFFFFFFFF -> NaN sentinel
        rt.memset(segm_exp_d, 0xFF, 4 * segm_ml_n)
        rt.memset(segm_out_d, 0xFF, 4 * segm_out_n)
        rt.sync()
        rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
        rt.sync()

        def _readback(ptr, n, shape):
            buf = (ctypes.c_uint8 * (4 * n))()
            rt.memcpy_d2h(buf, ptr, 4 * n)
            return np.frombuffer(bytes(buf), dtype=np.float32).reshape(shape)

        smf = _readback(segm_max_d, segm_ml_n, (total_q, _NQH, NUM_SEG))
        sef = _readback(segm_exp_d, segm_ml_n, (total_q, _NQH, NUM_SEG))
        sof = _readback(segm_out_d, segm_out_n, (total_q, _NQH, NUM_SEG, _HD))
        for name, arr in (
            ("segm_max", smf),
            ("segm_expsum", sef),
            ("segm_output", sof),
        ):
            uw = np.argwhere(np.isnan(arr))
            print(f"[probe] {name}: {len(uw)} unwritten; first: {uw[:6].tolist()}")
        return 0

    rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
    rt.sync()
    rt.launch(red_fn, red_grid, red_blk, red_packed)
    rt.sync()
    rt.memcpy_d2h(u8_out := (ctypes.c_uint8 * out.nbytes)(), od, out.nbytes)
    out = np.frombuffer(bytes(u8_out), dtype=_BF16).reshape(out.shape).copy()

    if _os.environ.get("ROCKE_DECODE3D_DUMP") == "1":

        def _rb(ptr, n, shape):
            buf = (ctypes.c_uint8 * (4 * n))()
            rt.memcpy_d2h(buf, ptr, 4 * n)
            return np.frombuffer(bytes(buf), dtype=np.float32).reshape(shape)

        smf = _rb(segm_max_d, segm_ml_n, (total_q, _NQH, NUM_SEG))
        sef = _rb(segm_exp_d, segm_ml_n, (total_q, _NQH, NUM_SEG))
        sof = _rb(segm_out_d, segm_out_n, (total_q, _NQH, NUM_SEG, _HD))
        # numpy re-implementation of the reduce from the GPU workspace
        om = smf.max(axis=2)  # [tq, nqh]
        fac = np.where(smf > -1e29, np.exp2(smf - om[:, :, None]), 0.0)
        denom = (sef * fac).sum(axis=2)
        inv = np.where(denom == 0, 0.0, 1.0 / denom)
        np_out = (sof * fac[:, :, :, None]).sum(axis=2) * inv[:, :, None]
        ws_huge = np.argwhere(np.abs(sof) > 1e20)
        print(
            f"[dump] workspace |segm_output|>1e20: {len(ws_huge)}; "
            f"segm_max range [{smf.min():.3g},{smf.max():.3g}] "
            f"segm_expsum range [{sef.min():.3g},{sef.max():.3g}]"
        )
        if len(ws_huge):
            print(f"[dump] segm_output huge first: {ws_huge[:12].tolist()}")
            unique_slots, counts = np.unique(ws_huge[:, :3], axis=0, return_counts=True)
            order = np.argsort(-counts)
            summary = [unique_slots[i].tolist() + [int(counts[i])] for i in order[:12]]
            print(f"[dump] huge slot summary [token,head,seg,count]: {summary}")
            first_t, first_h, first_s, _first_d = (int(x) for x in ws_huge[0])
            print(
                "[dump] first huge slot ml: "
                f"token={first_t} head={first_h} seg={first_s} "
                f"m={smf[first_t, first_h, first_s]:.6g} "
                f"l={sef[first_t, first_h, first_s]:.6g} "
                f"dims48_63={sof[first_t, first_h, first_s, 48:64].tolist()}"
            )
        krn = out.astype(np.float32)
        red_mismatch = np.argwhere(np.abs(krn - np_out) > 1e-2)
        print(
            f"[dump] kernel-reduce vs numpy-reduce-of-workspace mismatches: "
            f"{len(red_mismatch)}; first: {red_mismatch[:4].tolist()}"
        )

    bench_us = None
    bench_seg_us = None
    bench_red_us = None
    if args.bench:
        for _ in range(args.warmup):
            rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            rt.launch(red_fn, red_grid, red_blk, red_packed)
        rt.sync()
        ev0, ev1 = rt.event(), rt.event()
        ev0.record()
        for _ in range(args.iters):
            rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            rt.launch(red_fn, red_grid, red_blk, red_packed)
        ev1.record()
        ev1.synchronize()
        bench_us = ev0.elapsed_to(ev1) * 1000.0 / args.iters
        ev0.destroy()
        ev1.destroy()
        rt.sync()
        if args.bench_split:
            for _ in range(args.warmup):
                rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            rt.sync()
            es0, es1 = rt.event(), rt.event()
            es0.record()
            for _ in range(args.iters):
                rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            es1.record()
            es1.synchronize()
            bench_seg_us = es0.elapsed_to(es1) * 1000.0 / args.iters
            es0.destroy()
            es1.destroy()
            rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            rt.sync()
            for _ in range(args.warmup):
                rt.launch(red_fn, red_grid, red_blk, red_packed)
            rt.sync()
            er0, er1 = rt.event(), rt.event()
            er0.record()
            for _ in range(args.iters):
                rt.launch(red_fn, red_grid, red_blk, red_packed)
            er1.record()
            er1.synchronize()
            bench_red_us = er0.elapsed_to(er1) * 1000.0 / args.iters
            er0.destroy()
            er1.destroy()
            rt.sync()

    for ptr in (
        qd,
        kd,
        vd,
        od,
        sink_d,
        bt_d,
        sl_d,
        alibi_d,
        qq_d,
        cuq_d,
        segm_out_d,
        segm_max_d,
        segm_exp_d,
    ):
        rt.free(ptr)
    seg_mod.unload()
    red_mod.unload()

    ref = _ref_decode(
        q_f32,
        kc_f32,
        vc_f32,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        sinks_f32=sinks_f32,
    )
    out_f = out.astype(np.float32)
    diff = np.abs(out_f - ref)
    max_abs = float(diff.max())
    has_nan = bool(np.isnan(out_f).any())
    bad = int(np.count_nonzero(diff > args.tol))
    ok = (not has_nan) and max_abs <= args.tol
    bench_str = f" lat={bench_us:.2f}us" if bench_us is not None else ""
    if bench_seg_us is not None and bench_red_us is not None:
        bench_str += f" seg={bench_seg_us:.2f}us red={bench_red_us:.2f}us"
    if args.csv:
        row = [
            arch,
            args.kv_len,
            num_seqs,
            NUM_SEG,
            args.kv_dtype,
            use_sinks,
            f"{bench_us:.4f}" if bench_us is not None else "",
            f"{bench_seg_us:.4f}" if bench_seg_us is not None else "",
            f"{bench_red_us:.4f}" if bench_red_us is not None else "",
            f"{max_abs:.6e}",
            int(ok),
        ]
        newf = not pathlib.Path(args.csv).exists()
        with open(args.csv, "a", newline="") as fp:
            w = csv.writer(fp)
            if newf:
                w.writerow(
                    [
                        "arch",
                        "kv_len",
                        "num_seqs",
                        "num_segments",
                        "kv_dtype",
                        "sinks",
                        "us_full",
                        "us_seg",
                        "us_red",
                        "max_abs",
                        "ok",
                    ]
                )
            w.writerow(row)
    print(
        f"[{arch}] decode3d kv_len={args.kv_len} seqs={num_seqs} seg={NUM_SEG} "
        f"kv={args.kv_dtype} sinks={use_sinks}: max_abs={max_abs:.3e} nan={has_nan} "
        f"bad={bad}/{out.size} tol={args.tol:.0e}{bench_str} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
