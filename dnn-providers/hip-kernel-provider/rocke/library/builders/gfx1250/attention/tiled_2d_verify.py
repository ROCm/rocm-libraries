# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1250 WMMA tiled-2D unified-attention forward.

Numpy-only harness (no torch) for the gfx1250 box: builds
``instances/gfx1250/attention_tiled_2d.py``, launches it via the HIP runtime
with paged fp8 K/V + bf16 Q, and compares against a numpy paged-attention
reference (fp8 dequant, GQA-8, causal, optional sliding-window + sinks).

Requires ``ml_dtypes`` for bf16 / fp8e4m3 host encoding:

    /tmp/rocke_gfx1250/venv/bin/python -m pip -q install ml_dtypes

Run on a gfx1250 device:

    export PATH=/opt/rocm/bin:$PATH
    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH=/tmp/rocke_gfx1250/python
    HIP_VISIBLE_DEVICES=3 python -m \\
        rocke.examples.gfx1250.attention.tiled_2d_verify --scenario sw
"""

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from rocke.helpers import compile_kernel
from rocke.instances.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.runtime.hip_module import Runtime

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
    _FP8 = ml_dtypes.float8_e4m3fn
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "ml_dtypes required for bf16/fp8 host encoding; "
        "pip install ml_dtypes into the venv"
    ) from e

_HD = 64
_BS = 32
_NQH = 64
_NKVH = 8
_NQK = _NQH // _NKVH
_BLOCK_Q = 2


def _ref(
    q_f32,
    kc_f32,
    vc_f32,
    *,
    block_tables,
    query_lens,
    seq_lens,
    scale,
    sliding_window,
    sinks_f32,
):
    """Numpy paged-attention reference matching the gfx1250 kernel semantics."""
    num_seqs = len(query_lens)
    outputs = []
    start = 0
    for i in range(num_seqs):
        ql = int(query_lens[i])
        sl = int(seq_lens[i])
        ctx = sl - ql
        qi = q_f32[start : start + ql]  # [ql, NQH, HD]
        nblk = (sl + _BS - 1) // _BS
        idx = block_tables[i, :nblk]
        k = kc_f32[idx].reshape(-1, _NKVH, _HD)[:sl]  # [sl, NKVH, HD]
        v = vc_f32[idx].reshape(-1, _NKVH, _HD)[:sl]
        k = np.repeat(k, _NQK, axis=1)  # -> [sl, NQH, HD]
        v = np.repeat(v, _NQK, axis=1)
        scores = np.einsum("qhd,khd->hqk", qi, k) * scale  # [NQH, ql, sl]
        qpos = np.arange(ql)[:, None]
        kpos = np.arange(sl)[None, :]
        keep = kpos <= (ctx + qpos)
        if sliding_window and sliding_window > 0:
            keep &= (ctx + qpos - kpos) < sliding_window
        mask = ~keep  # [ql, sl]
        scores = np.where(mask[None, :, :], -np.inf, scores)
        if sinks_f32 is not None:
            s_aux = np.broadcast_to(sinks_f32[:, None, None], (_NQH, ql, 1))
            scores = np.concatenate([scores, s_aux], axis=-1)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        p = e / e.sum(axis=-1, keepdims=True)
        if sinks_f32 is not None:
            p = p[..., :-1]
        out = np.einsum("hqk,khd->qhd", p, v)  # [ql, NQH, HD]
        outputs.append(out)
        start += ql
    return np.concatenate(outputs, axis=0)


def _scenario(name: str):
    """Return (query_lens, ctx_lens, sliding_window, use_sinks)."""
    if name == "decode":
        return [1, 1, 1, 1], [256, 512, 1024, 33], 0, True
    if name == "prefill":
        return [16, 32, 8], [16, 64, 100], 0, True
    if name == "sw":
        return [8, 16], [300, 600], 128, True
    if name == "mixed":
        return [1, 16, 1, 4], [1024, 200, 63, 500], 0, True
    if name == "prefill_big":
        return [2048], [0], 0, True
    if name == "prefill_big_sw":
        return [2048], [0], 128, True
    if name == "decode_big":
        return [1] * 256, [1024] * 256, 0, True
    if name == "decode_big_sw":
        return [1] * 256, [1024] * 256, 128, True
    raise SystemExit(f"unknown scenario {name!r}")


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx1250")
    ap.add_argument(
        "--scenario",
        default="prefill",
        choices=(
            "decode",
            "prefill",
            "sw",
            "mixed",
            "prefill_big",
            "prefill_big_sw",
            "decode_big",
            "decode_big_sw",
        ),
    )
    ap.add_argument("--tol", type=float, default=3e-2)
    ap.add_argument("--seed", type=int, default=0xA11E)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time the launch (HIP events)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--waves-per-eu",
        type=int,
        default=None,
        help="override the kernel's waves_per_eu occupancy hint",
    )
    ap.add_argument("--register-p", action="store_true")
    ap.add_argument("--no-register-p", action="store_true")
    args = ap.parse_args()
    if args.register_p and args.no_register_p:
        ap.error("--register-p and --no-register-p are mutually exclusive")

    query_lens, ctx_lens, sliding_window, use_sinks = _scenario(args.scenario)
    num_seqs = len(query_lens)
    seq_lens = [int(query_lens[i] + ctx_lens[i]) for i in range(num_seqs)]
    total_q = int(sum(query_lens))
    max_blocks = max((sl + _BS - 1) // _BS for sl in seq_lens)
    num_blocks = max_blocks * num_seqs + 4

    spec = UnifiedAttention2DTiledSpec(
        head_size=_HD,
        block_size=_BS,
        num_query_heads=_NQH,
        num_kv_heads=_NKVH,
        dtype="bf16",
        use_sinks=use_sinks,
        sliding_window=sliding_window,
        has_softcap=False,
        num_seqs=num_seqs,
        kv_storage_dtype="fp8e4m3",
        tile_size=_BS,
        waves_per_eu=args.waves_per_eu,
        use_register_p=args.register_p,
    )
    kernel = build_unified_attention_2d_tiled(spec, arch=args.arch)
    art = compile_kernel(kernel, arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa})")
    if args.no_verify:
        print(f"[{args.arch}] build OK (verify skipped)")
        return 0

    rng = np.random.default_rng(args.seed)
    scale = float(_HD**-0.5)
    k_scale = 1.0
    v_scale = 1.0

    # Host tensors. K/V are quantized to fp8 then dequantized for the reference
    # so both kernel and reference see the *same* fp8 cache values.
    q_f32 = (rng.standard_normal((total_q, _NQH, _HD)) * 0.3).astype(np.float32)
    kc_q = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    vc_q = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    kc_f32 = kc_q.astype(np.float32) * k_scale
    vc_f32 = vc_q.astype(np.float32) * v_scale

    q_bf16 = q_f32.astype(_BF16)
    out = np.zeros((total_q, _NQH, _HD), dtype=_BF16)

    cu_q = np.zeros(num_seqs + 1, dtype=np.int32)
    cu_q[1:] = np.cumsum(query_lens)
    seq_lens_np = np.array(seq_lens, dtype=np.int32)
    block_tables = np.zeros((num_seqs, max_blocks), dtype=np.int32)
    for i in range(num_seqs):
        block_tables[i] = rng.permutation(num_blocks)[:max_blocks]
    sinks_bf16 = (rng.standard_normal(_NQH) * 0.5).astype(_BF16) if use_sinks else None
    sinks_f32 = sinks_bf16.astype(np.float32) if use_sinks else None

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

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
    kd = alloc_copy(kc_q)
    vd = alloc_copy(vc_q)
    od = rt.alloc(out.nbytes)
    rt.memset(od, 0, out.nbytes)
    sink_d = alloc_copy(sinks_bf16) if use_sinks else rt.alloc(2 * _NQH)
    bt_d = alloc_copy(block_tables)
    sl_d = alloc_copy(seq_lens_np)
    alibi_d = rt.alloc(4 * _NQH)
    qq_d = rt.alloc(4)
    cuq_d = alloc_copy(cu_q)

    # ABI param order from attention_tiled_2d._declare_params:
    #   out, query, key, value, sink, block_tables, seq_lens, alibi, qq_bias,
    #   cu_q, scale, k_scale, v_scale, out_scale, softcap,
    #   num_seqs, block_table_stride, qq_bias_stride_0
    packed = struct.pack(
        "<" + "Q" * 10 + "f" * 5 + "i" * 3,
        od,
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
        1.0,
        0.0,
        num_seqs,
        int(block_tables.shape[1]),
        0,
    )

    total_num_q_blocks = total_q // _BLOCK_Q + num_seqs
    grid = (_NKVH, int(total_num_q_blocks), 1)
    block = (32, 1, 1)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8_out := (ctypes.c_uint8 * out.nbytes)(), od, out.nbytes)
    out = np.frombuffer(bytes(u8_out), dtype=_BF16).reshape(out.shape).copy()

    bench_us = None
    if args.bench:
        for _ in range(args.warmup):
            rt.launch(fn, grid, block, packed)
        rt.sync()
        ev0, ev1 = rt.event(), rt.event()
        ev0.record()
        for _ in range(args.iters):
            rt.launch(fn, grid, block, packed)
        ev1.record()
        ev1.synchronize()
        bench_us = ev0.elapsed_to(ev1) * 1000.0 / args.iters
        ev0.destroy()
        ev1.destroy()
        rt.sync()
        print(
            f"[{args.arch}] bench scenario={args.scenario} grid={grid} "
            f"iters={args.iters}: {bench_us:.2f} us/launch"
        )

    for ptr in (qd, kd, vd, od, sink_d, bt_d, sl_d, alibi_d, qq_d, cuq_d):
        rt.free(ptr)
    module.unload()

    ref = _ref(
        q_f32,
        kc_f32,
        vc_f32,
        block_tables=block_tables,
        query_lens=query_lens,
        seq_lens=seq_lens,
        scale=scale,
        sliding_window=sliding_window,
        sinks_f32=sinks_f32,
    )

    out_f = out.astype(np.float32)
    ref_f = ref.astype(np.float32)
    diff = np.abs(out_f - ref_f)
    max_abs = float(diff.max())
    has_nan = bool(np.isnan(out_f).any())
    bad = int(np.count_nonzero(diff > args.tol))
    ok = (not has_nan) and max_abs <= args.tol
    print(
        f"[{args.arch}] tiled2d scenario={args.scenario} seqs={num_seqs} "
        f"tq={total_q} sw={sliding_window} sinks={use_sinks}: "
        f"max_abs={max_abs:.3e} nan={has_nan} bad={bad}/{out.size} "
        f"tol={args.tol:.0e} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
