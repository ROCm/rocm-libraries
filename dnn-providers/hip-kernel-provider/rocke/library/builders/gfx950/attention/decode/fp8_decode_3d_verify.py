# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx950 fp8 KV-dequant split-KV 3D decode attention.

Numpy-only harness (no torch) for a gfx950 box: for each shape in the gpt-oss
decode cohort it builds the split-KV segment + reduce kernels
(``_tiled_3d_impl("gfx950")``) at the *shipped* config, launches both with an
fp32 partials workspace, and compares the merged output against an independent
numpy paged decode-attention reference (GQA, optional sinks, fp8 e4m3fn KV).

This is the on-GPU numeric gate for the fp8-long-KV-decode -> 3D routing change:
the routing gate (``_enable_fp8_decode_3d``) moves this cohort onto the 3D
kernels verified here, so a regression on them cannot ship green.

Cohort (gpt-oss decode): D64, 64/8 GQA, block_size=16, Sq=1, flash + sink,
kv_len in {2048, 8192}, batch in {1, 64}. e4m3fn is the gfx950-native (OCP) fp8
dialect, so the format-vs-arch guard admits.

Runs the full cohort by default (non-zero exit if any shape fails); pass
``--kv-len`` / ``--num-seqs`` / ``--flash-only`` / ``--sink-only`` to pin one.

    export PATH=/opt/rocm/bin:$PATH
    export PYTHONPATH=<rocke>/platform/python
    python -m builders.gfx950.attention.decode.fp8_decode_3d_verify
    python -m builders.gfx950.attention.decode.fp8_decode_3d_verify --kv-len 8192 --num-seqs 64
"""

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from rocke.helpers import compile_kernel
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import Runtime, get_device_arch

# Soft import: bf16/fp8 host encodings need ml_dtypes (numpy has neither). Kept
# soft so the arch self-skip below fires FIRST on non-gfx950 boxes -- the driver
# must exit 0 there, not fail on a missing dep it never uses.
try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
    _FP8 = ml_dtypes.float8_e4m3fn  # gfx950-native OCP fp8
    _ML_DTYPES_ERR = None
except Exception as e:  # pragma: no cover
    ml_dtypes = None
    _BF16 = _FP8 = None
    _ML_DTYPES_ERR = e

# gpt-oss decode cohort geometry.
_HD = 64
_NQH = 64
_NKVH = 8
_NQK = _NQH // _NKVH  # 8
_BS = 16
_BLOCK_Q = 16 // _NQK  # = 2 (block_m=16 for num_queries_per_kv<=16)


def _ref_decode(q_f32, kc_f32, vc_f32, *, block_tables, seq_lens, scale, sinks_f32):
    """Independent numpy paged decode-attention reference (GQA, optional sink)."""
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


def _verify_one(arch, *, num_seqs, kv_len, use_sinks, tol, seed):
    """Build + launch the shipped 3D decode kernels for one shape; return
    (ok, max_abs, has_nan, label)."""
    total_q = num_seqs  # q_len == 1
    scale = float(_HD**-0.5)
    k_scale = v_scale = 1.0
    wave_size = 64  # gfx950 MFMA

    au._RESOLVED_ATTENTION_ARCH = arch
    problem = au.UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=_NQH,
        num_kv_heads=_NKVH,
        head_size=_HD,
        block_size=_BS,
        max_seqlen_q=1,
        max_seqlen_k=kv_len,
        dtype="bf16",
        q_dtype="bf16",
        sliding_window=0,
        use_sinks=use_sinks,
        use_fp8=True,
    )
    # Guard: the routing gate must actually pick 3D for this cohort on gfx950.
    assert problem.select_path() == "3d", "cohort did not route to 3D"
    ok_support, why = au.supports_native_unified_attention_3d_tiled(problem)
    if not ok_support:
        raise SystemExit(f"[{arch}] decode3d UNSUPPORTED: {why}")

    # Use the SHIPPED segmentation the dispatcher picks for this shape.
    seg_cfg, _red_cfg = problem.select_3d()
    num_segments = int(seg_cfg.NUM_SEGMENTS_PER_SEQ)

    Spec3D, ReduceSpec, build_seg, build_red, _ = au._tiled_3d_impl(arch)
    seg_spec = au._tiled_3d_spec_from_problem(problem)
    from dataclasses import replace

    seg_spec = replace(seg_spec, num_segments=num_segments)
    red_spec = ReduceSpec(
        head_size=_HD,
        num_query_heads=_NQH,
        num_kv_heads=_NKVH,
        dtype="bf16",
        num_segments=num_segments,
    )
    seg_art = compile_kernel(build_seg(seg_spec, arch=arch), arch=arch)
    red_art = compile_kernel(build_red(red_spec, arch=arch), arch=arch)

    rng = np.random.default_rng(seed)
    max_blocks = (kv_len + _BS - 1) // _BS
    num_blocks = max_blocks * num_seqs + 4
    q_f32 = (rng.standard_normal((total_q, _NQH, _HD)) * 0.3).astype(np.float32)
    kc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    vc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    kc_f32 = kc.astype(np.float32) * k_scale
    vc_f32 = vc.astype(np.float32) * v_scale

    q_bf16 = q_f32.astype(_BF16)
    out = np.zeros((total_q, _NQH, _HD), dtype=_BF16)
    cu_q = np.arange(num_seqs + 1, dtype=np.int32)  # q_len == 1
    seq_lens_np = np.array([kv_len] * num_seqs, dtype=np.int32)
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

    # fp32 partials workspace (segment output + running max/expsum).
    segm_out_n = total_q * _NQH * num_segments * _HD
    segm_ml_n = total_q * _NQH * num_segments
    segm_out_d = rt.alloc(4 * segm_out_n)
    segm_max_d = rt.alloc(4 * segm_ml_n)
    segm_exp_d = rt.alloc(4 * segm_ml_n)

    total_num_q_blocks = total_q // _BLOCK_Q + num_seqs
    seg_grid = (int(total_num_q_blocks), int(_NKVH), int(num_segments))
    seg_waves = int(getattr(seg_spec, "num_waves", 1))
    seg_blk = (wave_size * seg_waves, 1, 1)
    red_blk = (wave_size, 1, 1)
    red_grid = (int(total_q), int(_NQH), 1)

    seg_packed = struct.pack(
        "<" + "Q" * 12 + "f" * 4 + "i" * 3,
        segm_out_d, segm_max_d, segm_exp_d, qd, kd, vd, sink_d, bt_d, sl_d,
        alibi_d, qq_d, cuq_d,
        scale, k_scale, v_scale, 0.0,
        num_seqs, int(block_tables.shape[1]), 0,
    )
    red_packed = struct.pack(
        "<" + "Q" * 5, od, segm_out_d, segm_max_d, segm_exp_d, sl_d
    )

    rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
    rt.sync()
    rt.launch(red_fn, red_grid, red_blk, red_packed)
    rt.sync()
    rt.memcpy_d2h(u8_out := (ctypes.c_uint8 * out.nbytes)(), od, out.nbytes)
    out = np.frombuffer(bytes(u8_out), dtype=_BF16).reshape(out.shape).copy()

    for ptr in (qd, kd, vd, od, sink_d, bt_d, sl_d, alibi_d, qq_d, cuq_d,
                segm_out_d, segm_max_d, segm_exp_d):
        rt.free(ptr)
    seg_mod.unload()
    red_mod.unload()

    ref = _ref_decode(
        q_f32, kc_f32, vc_f32, block_tables=block_tables,
        seq_lens=[kv_len] * num_seqs, scale=scale, sinks_f32=sinks_f32,
    )
    out_f = out.astype(np.float32)
    diff = np.abs(out_f - ref)
    max_abs = float(diff.max())
    has_nan = bool(np.isnan(out_f).any())
    ok = (not has_nan) and max_abs <= tol
    label = f"{'sink' if use_sinks else 'flash'}_b{num_seqs}_kv{kv_len}"
    return ok, max_abs, has_nan, label, num_segments


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default=None, help="default: auto-detect the device")
    ap.add_argument("--kv-len", type=int, default=None, help="pin one kv_len")
    ap.add_argument("--num-seqs", type=int, default=None, help="pin one batch")
    ap.add_argument("--flash-only", action="store_true")
    ap.add_argument("--sink-only", action="store_true")
    ap.add_argument("--tol", type=float, default=5e-2)  # bf16 band (runbook 1.4)
    ap.add_argument("--seed", type=int, default=0x9500)
    args = ap.parse_args()

    arch = args.arch or get_device_arch()
    if arch != "gfx950":
        print(f"[skip] this driver targets gfx950; device is {arch}")
        return 0
    if ml_dtypes is None:
        raise SystemExit(
            "ml_dtypes required on gfx950 for bf16/fp8 host encodings; "
            "pip install ml_dtypes into the venv"
        ) from _ML_DTYPES_ERR

    sinks = (
        (False,) if args.flash_only else (True,) if args.sink_only else (False, True)
    )
    batches = (args.num_seqs,) if args.num_seqs else (1, 64)
    kv_lens = (args.kv_len,) if args.kv_len else (2048, 8192)

    print(f"[{arch}] fp8 e4m3fn decode 3D verify  (D{_HD} {_NQH}x{_NKVH} bs{_BS}) "
          f"tol={args.tol:.0e}")
    failed = False
    for use_sinks in sinks:
        for num_seqs in batches:
            for kv_len in kv_lens:
                ok, max_abs, has_nan, label, nseg = _verify_one(
                    arch, num_seqs=num_seqs, kv_len=kv_len, use_sinks=use_sinks,
                    tol=args.tol, seed=args.seed,
                )
                failed = failed or not ok
                print(f"  {label:<18} seg={nseg:<3} max_abs={max_abs:.3e} "
                      f"nan={has_nan} -> {'PASS' if ok else 'FAIL'}")
    print("FAIL — a shape exceeded tol" if failed else "PASS — all cohort shapes correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
