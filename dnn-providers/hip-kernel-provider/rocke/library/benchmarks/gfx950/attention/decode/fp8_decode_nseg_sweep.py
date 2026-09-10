"""fp8 decode num_segments sweep (gfx950) -- is the shipped split-KV count optimal?

The 3D split-KV fp8 decode seg kernel is VALU/softmax-bound; ISA inspection showed
the split-KV partial-write overhead (vmem_store) is the one 3D-specific cost. This
sweeps ``num_segments`` (the split count -- a free spec field) for the gpt-oss decode
cohort to see whether a different split beats the shipped value the dispatcher derives
from num_cus. Correctness-gated (numpy paged-decode reference), timed via HIP events.

Both variants are the same kernel family, so this is a pure latency/overhead A/B. If a
non-shipped split wins reproducibly, it justifies an fp8-decode num_segments override;
if all splits are neutral, the cohort is confirmed at its floor.

Measured latencies are for the local decision only -- Confluence, never the repo.

Run (rocke .venv, gfx950 node):
    python fp8_decode_nseg_sweep.py
    python fp8_decode_nseg_sweep.py --warmup 30 --iters 300
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys

import numpy as np

from rocke.helpers import compile_kernel
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import Runtime, get_device_arch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
    _FP8 = ml_dtypes.float8_e4m3fn
    _ML_ERR = None
except Exception as e:  # pragma: no cover
    ml_dtypes = None
    _BF16 = _FP8 = None
    _ML_ERR = e

_HD, _NQH, _NKVH, _BS = 64, 64, 8, 16
_NQK = _NQH // _NKVH
_BLOCK_Q = 16 // _NQK
_TOL = 5e-2
_K_SCALE = _V_SCALE = 1.0
_CANDIDATES = (8, 16, 32, 64, 128)


def _ref_decode(q_f32, kc_f32, vc_f32, *, block_tables, seq_lens, scale, sinks_f32, num_seqs):
    out = np.zeros((num_seqs, _NQH, _HD), dtype=np.float32)
    for i in range(num_seqs):
        sl = int(seq_lens[i])
        nblk = (sl + _BS - 1) // _BS
        idx = block_tables[i, :nblk]
        k = np.repeat(kc_f32[idx].reshape(-1, _NKVH, _HD)[:sl], _NQK, axis=1)
        v = np.repeat(vc_f32[idx].reshape(-1, _NKVH, _HD)[:sl], _NQK, axis=1)
        scores = np.einsum("hd,khd->hk", q_f32[i], k) * scale
        if sinks_f32 is not None:
            scores = np.concatenate([scores, sinks_f32[:, None]], axis=-1)
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        if sinks_f32 is not None:
            p = p[..., :-1]
        out[i] = np.einsum("hk,khd->hd", p, v)
    return out


def _sweep_shape(arch, *, num_seqs, kv_len, use_sinks, seed, warmup, iters, repeat, shipped):
    total_q = num_seqs
    scale = float(_HD ** -0.5)
    wave_size = 64

    au._RESOLVED_ATTENTION_ARCH = arch
    problem = au.UnifiedAttentionProblem(
        total_q=total_q, num_seqs=num_seqs, num_query_heads=_NQH, num_kv_heads=_NKVH,
        head_size=_HD, block_size=_BS, max_seqlen_q=1, max_seqlen_k=kv_len,
        dtype="bf16", q_dtype="bf16", sliding_window=0, use_sinks=use_sinks,
        use_fp8=True, num_cus=256,
    )
    _Spec3D, ReduceSpec, build_seg, build_red, _ = au._tiled_3d_impl(arch)
    from dataclasses import replace

    base_seg = au._tiled_3d_spec_from_problem(problem)

    rng = np.random.default_rng(seed)
    max_blocks = (kv_len + _BS - 1) // _BS
    num_blocks = max_blocks * num_seqs + 4
    q_f32 = (rng.standard_normal((total_q, _NQH, _HD)) * 0.3).astype(np.float32)
    kc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    vc = (rng.standard_normal((num_blocks, _BS, _NKVH, _HD)) * 0.3).astype(_FP8)
    kc_f32 = kc.astype(np.float32) * _K_SCALE
    vc_f32 = vc.astype(np.float32) * _V_SCALE
    q_bf16 = q_f32.astype(_BF16)
    out = np.zeros((total_q, _NQH, _HD), dtype=_BF16)
    cu_q = np.arange(num_seqs + 1, dtype=np.int32)
    seq_lens_np = np.array([kv_len] * num_seqs, dtype=np.int32)
    block_tables = np.zeros((num_seqs, max_blocks), dtype=np.int32)
    for i in range(num_seqs):
        block_tables[i] = rng.permutation(num_blocks)[:max_blocks]
    sinks_bf16 = (rng.standard_normal(_NQH) * 0.5).astype(_BF16) if use_sinks else None
    sinks_f32 = sinks_bf16.astype(np.float32) if use_sinks else None
    ref = _ref_decode(q_f32, kc_f32, vc_f32, block_tables=block_tables,
                      seq_lens=seq_lens_np, scale=scale, sinks_f32=sinks_f32,
                      num_seqs=num_seqs)

    rt = Runtime()

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    def alloc_copy(a):
        a = np.ascontiguousarray(a)
        d = rt.alloc(max(1, int(a.nbytes)))
        if a.nbytes:
            rt.memcpy_h2d(d, u8(a), a.nbytes)
        return d

    qd = alloc_copy(q_bf16); kd = alloc_copy(kc); vd = alloc_copy(vc)
    od = rt.alloc(out.nbytes)
    sink_d = alloc_copy(sinks_bf16) if use_sinks else rt.alloc(2 * _NQH)
    bt_d = alloc_copy(block_tables); sl_d = alloc_copy(seq_lens_np)
    alibi_d = rt.alloc(4 * _NQH); qq_d = rt.alloc(4); cuq_d = alloc_copy(cu_q)
    total_num_q_blocks = total_q // _BLOCK_Q + num_seqs

    rows = []
    for nseg in _CANDIDATES:
        try:
            seg_spec = replace(base_seg, num_segments=nseg)
            red_spec = ReduceSpec(head_size=_HD, num_query_heads=_NQH,
                                  num_kv_heads=_NKVH, dtype="bf16", num_segments=nseg)
            seg_art = compile_kernel(build_seg(seg_spec, arch=arch), arch=arch)
            red_art = compile_kernel(build_red(red_spec, arch=arch), arch=arch)
        except Exception as exc:  # noqa: BLE001
            rows.append((nseg, None, None, f"build ERR: {exc!r}"[:40]))
            continue
        seg_mod = rt.load_module(seg_art.hsaco); seg_fn = seg_mod.get_function(seg_art.kernel_name)
        red_mod = rt.load_module(red_art.hsaco); red_fn = red_mod.get_function(red_art.kernel_name)

        segm_out_n = total_q * _NQH * nseg * _HD
        segm_ml_n = total_q * _NQH * nseg
        segm_out_d = rt.alloc(4 * segm_out_n)
        segm_max_d = rt.alloc(4 * segm_ml_n)
        segm_exp_d = rt.alloc(4 * segm_ml_n)
        rt.memset(od, 0, out.nbytes)

        seg_grid = (int(total_num_q_blocks), int(_NKVH), int(nseg))
        seg_waves = int(getattr(seg_spec, "num_waves", 1))
        seg_blk = (wave_size * seg_waves, 1, 1)
        red_grid = (int(total_q), int(_NQH), 1)
        red_blk = (wave_size, 1, 1)
        seg_packed = struct.pack(
            "<" + "Q" * 12 + "f" * 4 + "i" * 3,
            segm_out_d, segm_max_d, segm_exp_d, qd, kd, vd, sink_d, bt_d, sl_d,
            alibi_d, qq_d, cuq_d, scale, _K_SCALE, _V_SCALE, 0.0,
            num_seqs, int(block_tables.shape[1]), 0,
        )
        red_packed = struct.pack("<" + "Q" * 5, od, segm_out_d, segm_max_d, segm_exp_d, sl_d)

        def _run():
            rt.launch(seg_fn, seg_grid, seg_blk, seg_packed)
            rt.launch(red_fn, red_grid, red_blk, red_packed)

        try:
            _run(); rt.sync()
            rt.memcpy_d2h(ob := (ctypes.c_uint8 * out.nbytes)(), od, out.nbytes)
            got = np.frombuffer(bytes(ob), dtype=_BF16).reshape(out.shape).astype(np.float32)
            err = float(np.abs(got - ref).max())
            nan = bool(np.isnan(got).any())
            for _ in range(warmup):
                _run()
            rt.sync()
            samples = []
            for _ in range(repeat):
                ev0, ev1 = rt.event(), rt.event()
                ev0.record()
                for _ in range(iters):
                    _run()
                ev1.record(); ev1.synchronize()
                samples.append(ev0.elapsed_to(ev1) * 1000.0 / iters)
            samples.sort()
            us = samples[len(samples) // 2]  # median over repeat blocks
            status = "OK" if (not nan and err < _TOL) else ("NaN" if nan else "DIFF")
            rows.append((nseg, err, us, status))
        except Exception as exc:  # noqa: BLE001
            rows.append((nseg, None, None, f"run ERR: {exc!r}"[:40]))
        for ptr in (segm_out_d, segm_max_d, segm_exp_d):
            rt.free(ptr)
        seg_mod.unload(); red_mod.unload()

    for ptr in (qd, kd, vd, od, sink_d, bt_d, sl_d, alibi_d, qq_d, cuq_d):
        rt.free(ptr)

    label = f"{'sink' if use_sinks else 'flash'}_b{num_seqs}_kv{kv_len}"
    ok_rows = [(n, u) for n, e, u, s in rows if s == "OK" and u is not None]
    best = min(ok_rows, key=lambda t: t[1]) if ok_rows else None
    print(f"\n{label}  (shipped num_segments={shipped})")
    for nseg, err, us, status in rows:
        mark = "  <= shipped" if nseg == shipped else ""
        best_mark = "  * best" if best and nseg == best[0] else ""
        us_s = f"{us:8.2f}" if us is not None else "     -- "
        err_s = f"{err:.3e}" if err is not None else "   --   "
        print(f"    seg={nseg:<4} {us_s} us  max_abs={err_s}  {status}{mark}{best_mark}")
    if best and shipped in [n for n, _ in ok_rows]:
        ship_us = dict(ok_rows)[shipped]
        gain = (ship_us - best[1]) / ship_us
        verdict = f"best seg={best[0]} is {gain*100:.1f}% vs shipped" if best[0] != shipped else "shipped is best"
        print(f"    -> {verdict}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1, help="timing blocks to median over")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arch = get_device_arch()
    if arch != "gfx950":
        print(f"[skip] targets gfx950; device is {arch}")
        return 0
    if ml_dtypes is None:
        raise SystemExit("ml_dtypes required on gfx950; pip install ml_dtypes") from _ML_ERR

    print(f"[{arch}] fp8 e4m3fn decode num_segments sweep  candidates={_CANDIDATES}  "
          f"(D{_HD} {_NQH}x{_NKVH} bs{_BS})")
    for use_sinks in (False, True):
        for num_seqs in (1, 64):
            for kv_len in (2048, 8192):
                au._RESOLVED_ATTENTION_ARCH = arch
                p = au.UnifiedAttentionProblem(
                    total_q=num_seqs, num_seqs=num_seqs, num_query_heads=_NQH,
                    num_kv_heads=_NKVH, head_size=_HD, block_size=_BS, max_seqlen_q=1,
                    max_seqlen_k=kv_len, dtype="bf16", use_fp8=True, num_cus=256)
                shipped = au._num_segments(p)  # production value (gfx950 pre-bump clamp)
                _sweep_shape(arch, num_seqs=num_seqs, kv_len=kv_len, use_sinks=use_sinks,
                             seed=args.seed, warmup=args.warmup, iters=args.iters,
                             repeat=args.repeat, shipped=shipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
