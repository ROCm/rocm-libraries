#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Strengthened ckc_engine binding parity check (binding vs STANDALONE emitter).

For every family, for every config the standalone emitter tests/parity/<fam>_emit.c
encodes, this harness:
  1. runs the prebuilt standalone emitter (./emit/<fam> <idx> ll) -> real .ll text;
  2. calls the corresponding ckc_engine binding entry point with a config dict
     that mirrors that emitter config exactly;
  3. byte-compares the two .ll texts.

A config "matches NON-EMPTY" iff both produce identical, non-empty .ll. A config
where the STANDALONE emitter produces empty .ll (a spec the engine itself
rejects) is reported separately and does NOT count toward a family being
"validated"; per the strengthened rule a family is validated only if it has at
least one buildable config and ALL its buildable configs match NON-EMPTY.

Env:
  CKC_WS16B_BUILD   build dir holding ckc_engine*.so  (default /tmp/ckc_ws16b/build)
  CKC_WS16B_EMIT    dir holding the prebuilt standalone emitters (default /tmp/ckc_ws16b/emit)
"""

import hashlib
import os
import subprocess
import sys

BUILD = os.environ.get("CKC_WS16B_BUILD", "/tmp/ckc_ws16b/build")
EMIT = os.environ.get("CKC_WS16B_EMIT", "/tmp/ckc_ws16b/emit")
sys.path.insert(0, BUILD)
import ckc_engine  # noqa: E402


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run_emitter(fam, idx):
    """Run ./emit/<fam> <idx> ll -> stdout text (empty string on reject/error)."""
    exe = os.path.join(EMIT, fam)
    try:
        p = subprocess.run(
            [exe, str(idx), "ll"], capture_output=True, text=True, timeout=120
        )
    except Exception as e:  # noqa: BLE001
        return None, f"emitter exec error: {e}"
    # Non-zero exit with empty stdout == reject (the engine rejected the spec).
    return p.stdout, (p.stderr.strip() if p.returncode != 0 else "")


# ---------------------------------------------------------------------------
# Per-family config tables. Each entry: list of dicts, one per emitter config
# index, mirroring tests/parity/<fam>_emit.c make_spec/make_cfg verbatim. The
# binding lower fn is ckc_engine.<lower_name>(dict, arch=<arch>).
# ---------------------------------------------------------------------------


def cfgs_layernorm2d():
    rows = [
        (4096, 256, 4, "f16", False),
        (4096, 256, 8, "f16", False),
        (4096, 256, 4, "bf16", False),
        (2048, 128, 4, "f16", True),
        (8192, 256, 8, "f16", False),
        (1024, 256, 2, "bf16", True),
        (128, 64, 2, "f16", False),
        (256, 64, 4, "f16", True),
        (512, 64, 8, "bf16", False),
        (128, 64, 2, "bf16", True),
        (4096, 512, 4, "f16", False),
        (8192, 1024, 8, "f16", False),
        (2048, 1024, 2, "bf16", True),
        (1024, 128, 8, "f16", False),
        (4096, 256, 4, "fp16", False),
        (2048, 128, 2, "fp16", True),
        (1536, 256, 2, "f16", False),
        (3072, 256, 4, "bf16", False),
        (5120, 256, 4, "f16", True),
        (1792, 128, 2, "bf16", False),
        (2816, 128, 2, "f16", False),
        (6656, 256, 2, "bf16", True),
        (16384, 256, 8, "f16", False),
        (33280, 256, 2, "f16", False),
        (32768, 256, 8, "bf16", False),
        (65536, 256, 8, "f16", True),
        (131072, 512, 8, "bf16", False),
        (34816, 256, 2, "bf16", True),
        (65536, 1024, 8, "f16", False),
        (133120, 1024, 2, "bf16", False),
        (2048, 256, 2, "f16", False),
        (4096, 256, 4, "f16", True),
        (8192, 256, 8, "bf16", True),
        (66560, 512, 2, "f16", False),
        (133120, 512, 4, "bf16", True),
    ]
    return [
        dict(n_per_block=n, block_size=b, vec=v, dtype=d, save_mean_invstd=s)
        for (n, b, v, d, s) in rows
    ]


def cfgs_rmsnorm2d():
    rows = [
        (1024, 256, 4, "f16", False),
        (2048, 256, 4, "bf16", False),
        (4096, 256, 4, "f16", False),
        (8192, 256, 8, "bf16", False),
        (16384, 256, 8, "f16", True),
        (131072, 256, 4, "f16", False),
    ]
    return [
        dict(n_per_block=n, block_size=b, vec=v, dtype=d, save_inv_rms=s)
        for (n, b, v, d, s) in rows
    ]


def cfgs_add_rmsnorm2d_bf16():
    rows = [
        (1024, 256, 4, "bf16", True, 64),
        (2048, 256, 4, "bf16", True, 64),
        (4096, 256, 4, "bf16", True, 64),
        (8192, 256, 4, "bf16", True, 64),
        (1024, 256, 2, "f16", False, 64),
        (2048, 128, 4, "bf16", True, 64),
    ]
    return [
        dict(n_per_block=n, block_size=b, vec=v, dtype=d, save_residual=s, wave_size=w)
        for (n, b, v, d, s, w) in rows
    ]


def cfgs_add_rmsnorm2d_rdquant():
    rows = [
        (4096, "f16", "i8", 256, 4, True, True, 64),
        (8192, "f16", "fp8e4m3", 256, 8, True, True, 64),
        (2048, "bf16", "bf8e5m2", 128, 4, False, False, 64),
        (16384, "f16", "i8", 256, 4, True, True, 64),
        (1024, "bf16", "i8", 64, 4, True, True, 64),
        (6144, "f16", "fp8e4m3", 256, 2, True, False, 64),
    ]
    return [
        dict(
            n_per_block=n,
            dtype=dt,
            out_dtype=o,
            block_size=b,
            vec=v,
            save_residual=sr,
            save_yscale=sy,
            wave_size=w,
        )
        for (n, dt, o, b, v, sr, sy, w) in rows
    ]


def cfgs_elementwise():
    rows = [
        ("relu", "f16", 256, 8),
        ("relu", "bf16", 256, 8),
        ("add", "f16", 128, 4),
        ("add", "f16", 512, 2),
        ("silu", "bf16", 64, 8),
        ("gelu_tanh", "f16", 1024, 4),
    ]
    return [dict(op=o, dtype=d, block_size=b, vec=v) for (o, d, b, v) in rows]


def cfgs_reduce():
    rows = [
        (4096, "sum", 256, 4, "f16", 64),
        (4096, "max", 256, 4, "f16", 64),
        (4096, "mean", 256, 4, "f16", 64),
        (2048, "sum", 128, 4, "bf16", 64),
        (4096, "sum", 512, 2, "f16", 64),
        (3072, "max", 256, 8, "bf16", 64),
    ]
    return [
        dict(n_per_block=n, op=o, block_size=b, vec=v, dtype=d, wave_size=w)
        for (n, o, b, v, d, w) in rows
    ]


def cfgs_pooling():
    rows = [
        (
            dict(N=1, H=28, W=28, C=64, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1),
            "f16",
            "max",
            256,
            1,
        ),
        (
            dict(N=2, H=56, W=56, C=128, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
            "f16",
            "avg",
            256,
            2,
        ),
        (
            dict(
                N=4, H=112, W=112, C=256, Y=7, X=7, sH=7, sW=7, pH=0, pW=0, dH=1, dW=1
            ),
            "f16",
            "sum",
            256,
            4,
        ),
        (
            dict(N=1, H=224, W=224, C=64, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1),
            "bf16",
            "max",
            256,
            1,
        ),
        (
            dict(N=2, H=32, W=32, C=256, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
            "f16",
            "avg",
            128,
            8,
        ),
        (
            dict(N=1, H=64, W=64, C=128, Y=2, X=2, sH=2, sW=2, pH=0, pW=0, dH=1, dW=1),
            "bf16",
            "sum",
            512,
            4,
        ),
    ]
    return [
        dict(problem=p, dtype=d, op=o, block_size=b, vec=v) for (p, d, o, b, v) in rows
    ]


def cfgs_transpose():
    rows = [
        (16, 16, 2, "f16", 8, "row"),
        (32, 32, 4, "f16", 8, "row"),
        (64, 64, 8, "f16", 8, "row"),
        (64, 64, 8, "bf16", 8, "row"),
        (32, 32, 4, "bf16", 8, "morton"),
        (16, 16, 4, "f16", 8, "row"),
    ]
    return [
        dict(tile_m=m, tile_n=n, vec=v, dtype=d, lds_pad=p, grid_order=g)
        for (m, n, v, d, p, g) in rows
    ]


def cfgs_permute_nd():
    rows = [
        (3, [4, 8, 16], [2, 0, 1], "f16", 256),
        (3, [16, 8, 4], [2, 1, 0], "f16", 256),
        (2, [32, 32], [1, 0], "bf16", 256),
        (4, [8, 8, 8, 8], [3, 2, 1, 0], "f16", 128),
        (3, [64, 64, 64], [1, 2, 0], "bf16", 512),
        (1, [256], [0], "f16", 256),
    ]
    return [
        dict(rank=r, x_shape=xs, perm=pm, dtype=d, block_size=b)
        for (r, xs, pm, d, b) in rows
    ]


def cfgs_smoothquant():
    rows = [
        (1024, "f16", "i8", 256, 4),
        (2048, "bf16", "i8", 256, 8),
        (512, "f16", "fp8e4m3", 128, 4),
        (1024, "bf16", "bf8e5m2", 64, 2),
        (2048, "f16", "i8", 512, 4),
        (256, "bf16", "fp8e4m3", 256, 2),
    ]
    return [
        dict(n_per_block=n, dtype=dt, out_dtype=o, block_size=b, vec=v)
        for (n, dt, o, b, v) in rows
    ]


def cfgs_topk_softmax():
    rows = [
        (32, 1, "f32", "f32", 32),
        (64, 4, "f16", "f32", 64),
        (128, 8, "bf16", "bf16", 64),
        (4096, 16, "f32", "f32", 128),
        (16384, 32, "f32", "f32", 256),
        (768, 2, "f32", "f32", 64),
    ]
    return [
        dict(n_per_row=n, k=k, dtype=d, out_dtype=o, block_size=b)
        for (n, k, d, o, b) in rows
    ]


def cfgs_moe_smoothquant():
    rows = [
        (512, 2, 64, "f16", "i8", 256, 4, None),
        (1024, 4, 128, "bf16", "fp8e4m3", 256, 4, None),
        (2048, 8, 256, "f16", "i8", 256, 4, 256),
        (4096, 1, 8, "f16", "i8", 512, 8, None),
    ]
    out = []
    for n, tk, e, dt, o, b, v, tok in rows:
        d = dict(
            n_per_block=n,
            topk=tk,
            experts=e,
            dtype=dt,
            out_dtype=o,
            block_size=b,
            vec=v,
        )
        if tok is not None:
            d["tokens"] = tok
        out.append(d)
    return out


def cfgs_moe_fused_mega():
    rows = [
        ("moe_mega_baseline", 32, 256, 32, 256, 64, "fp16"),
        ("moe_mega_tuned_m16", 16, 256, 32, 256, 64, "fp16"),
        ("moe_mega_large_k", 32, 256, 64, 256, 128, "fp16"),
        ("moe_mega_wide_n", 32, 512, 32, 512, 64, "fp16"),
        ("moe_mega_fp8", 32, 256, 32, 256, 64, "fp8e4m3"),
        ("moe_mega_bf16", 32, 256, 32, 256, 64, "bf16"),
    ]
    return [
        dict(
            name=nm,
            tile_m=tm,
            tile_n_inter=ni,
            tile_k_gu=kg,
            tile_n_down=nd,
            tile_k_down=kd,
            dtype=d,
        )
        for (nm, tm, ni, kg, nd, kd, d) in rows
    ]


def cfgs_moe_fused_mega_fp8():
    rows = [
        ("moe_fused_mega_fp8_baseline", 16, 256, 32, 32, False, None, False),
        ("moe_fused_mega_fp8_l7_hero", 16, 256, 128, 128, False, None, False),
        ("moe_fused_mega_fp8_l8_dtla", 16, 256, 128, 128, True, "none", False),
        ("moe_fused_mega_fp8_l9_iglp", 16, 256, 128, 128, True, "iglp1", False),
        ("moe_fused_mega_fp8_prod", 16, 256, 128, 128, True, "iglp1", False),
        ("moe_fused_mega_fp8_persistent", 16, 256, 128, 128, True, "iglp1", True),
    ]
    out = []
    for nm, tm, ni, gk, dk, dtla, sched, persist in rows:
        d = dict(
            name=nm,
            tile_m=tm,
            tile_n_inter=ni,
            gate_up_k=gk,
            down_k=dk,
            use_dtla=dtla,
            persistent=persist,
        )
        if sched is not None:
            d["sched_cadence"] = sched
        out.append(d)
    return out


def cfgs_fused_moe():
    # gather phase (the emitter default phase) for all 36 configs.
    rows = [
        (4, 4, 2, 128, 512, "f16", 256, 4),
        (1, 8, 2, 1024, 2048, "f16", 256, 4),
        (256, 16, 4, 4096, 16384, "bf16", 256, 8),
        (128, 32, 2, 2048, 8192, "f16", 512, 4),
        (512, 64, 8, 8192, 32768, "bf16", 1024, 8),
        (16, 4, 1, 256, 1024, "f16", 64, 2),
        (1, 1, 1, 64, 64, "f16", 64, 2),
        (1, 1, 1, 64, 64, "bf16", 64, 2),
        (1, 2, 1, 128, 128, "fp16", 64, 2),
        (7, 13, 3, 256, 256, "f16", 64, 2),
        (101, 17, 5, 512, 1536, "bf16", 256, 4),
        (3, 3, 3, 192, 384, "f16", 64, 2),
        (97, 5, 5, 320, 640, "bf16", 64, 2),
        (4, 4, 2, 64, 64, "f16", 64, 8),
        (4, 4, 2, 128, 128, "bf16", 128, 4),
        (4, 4, 2, 256, 256, "f16", 256, 8),
        (8, 4, 2, 128, 384, "f16", 64, 8),
        (8, 4, 2, 192, 192, "bf16", 64, 4),
        (16, 8, 2, 64, 2048, "f16", 64, 8),
        (16, 8, 2, 4096, 64, "bf16", 64, 8),
        (32, 8, 2, 512, 512, "f16", 512, 2),
        (32, 8, 2, 1024, 1024, "bf16", 1024, 2),
        (32, 8, 2, 2048, 2048, "f16", 1024, 8),
        (10, 6, 6, 384, 768, "bf16", 128, 4),
        (5, 5, 5, 640, 1280, "f16", 128, 8),
        (1024, 128, 8, 16384, 16384, "bf16", 1024, 8),
        (2048, 256, 4, 8192, 28672, "f16", 512, 4),
        (64, 8, 2, 1024, 4096, "f16", 256, 8),
        (64, 8, 2, 1024, 4096, "bf16", 256, 8),
        (64, 8, 2, 1024, 4096, "fp16", 256, 8),
        (64, 8, 2, 768, 1536, "f16", 64, 2),
        (64, 8, 2, 768, 1536, "f16", 64, 4),
        (64, 8, 2, 768, 1536, "f16", 64, 8),
        (4, 4, 2, 448, 704, "bf16", 64, 2),
        (4, 4, 2, 384, 640, "f16", 128, 4),
        (300, 32, 1, 2048, 8192, "bf16", 256, 8),
    ]
    return [
        dict(
            tokens=t,
            experts=e,
            topk=k,
            hidden=h,
            intermediate=i,
            dtype=d,
            block_size=b,
            vec=v,
            phase="gather",
        )
        for (t, e, k, h, i, d, b, v) in rows
    ]


def cfgs_moe_sorting():
    # flat 0..17: phase = flat<6?hist : flat<12?scan : scatter; spec = flat % 6
    specs = [
        (2, 8, 8, 64),
        (16, 4, 32, 256),
        (32, 8, 64, 256),
        (128, 2, 32, 512),
        (8, 16, 16, 128),
        (2, 8, 64, 256),
    ]
    phases = ["histogram", "scan", "scatter"]
    out = []
    for flat in range(18):
        tok, tk, e, b = specs[flat % 6]
        out.append(
            dict(tokens=tok, topk=tk, experts=e, block_size=b, phase=phases[flat // 6])
        )
    return out


def cfgs_moe_gemm_fused():
    pad = dict(epilogue="cshuffle", pad_m=True, pad_n=True)
    return [
        dict(
            kind="gate_up_silu",
            name="moe_gate_up_silu_f16",
            dtype="f16",
            grouped=False,
            tile=dict(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=pad,
        ),
        dict(
            kind="gate_up_silu",
            name="moe_gate_up_silu_grouped_f16",
            dtype="f16",
            grouped=True,
            tile=dict(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=pad,
        ),
        dict(
            kind="interleaved",
            name="moe_interleaved_gate_up_silu_f16",
            dtype="f16",
            grouped=False,
            tile=dict(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=pad,
        ),
        dict(
            kind="down_reduce",
            name="moe_down_reduce_f16",
            dtype="f16",
            grouped=False,
            tile=dict(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=pad,
        ),
        dict(
            kind="interleaved",
            name="moe_interleaved_bf16_grouped",
            dtype="bf16",
            grouped=True,
            tile=dict(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
        ),
        dict(
            kind="down_reduce",
            name="moe_down_grouped",
            dtype="f16",
            grouped=True,
            tile=dict(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_k=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
        ),
    ]


def cfgs_gfx1151_wmma_gemm():
    return [
        dict(name="ck_dsl_wmma_gemm", dtype="fp16", block_x_is_m=True),
        dict(name="wmma_probe_gfx1151", block_x_is_m=True),
        dict(dtype="fp16", block_x_is_m=True),
        dict(name="ck_dsl_wmma_gemm_v2", dtype="fp16", block_x_is_m=True),
        dict(name="wmma_gemm_tile16x16x16", block_x_is_m=False),
        dict(dtype="fp16", name="wmma_f16_16x16x16", block_x_is_m=False),
    ]


def cfgs_gfx1151_wmma_gemm_int8():
    names = [
        "ck_dsl_wmma_gemm_int8",
        "wmma_int8_probe_gfx1151",
        "ck_dsl_wmma_gemm_int8_v2",
        "wmma_gemm_int8_tile16x16x16",
        "wmma_int8_dequant_f16_out",
        "wmma_path_b_int8_f16",
    ]
    return [dict(name=n) for n in names]


def cfgs_gfx1151_wmma_gemm_iu8():
    return [dict(name="ck_dsl_wmma_gemm_iu8") for _ in range(6)]


def cfgs_gfx1151_wmma_gemm_iu8_dequant():
    return [dict() for _ in range(6)]


def cfgs_gfx1201_wmma_gemm():
    return [
        dict(name="ck_dsl_wmma_gemm_gfx12", dtype="fp16"),
        dict(name="wmma_probe_gfx1201"),
        dict(dtype="fp16"),
        dict(name="ck_dsl_wmma_gemm_gfx12_v2", dtype="fp16"),
        dict(name="wmma_gemm_tile16x16x16"),
        dict(dtype="fp16", name="wmma_f16_16x16x16"),
    ]


def cfgs_gfx1151_wmma_fmha_fwd():
    rows = [
        (64, 4, 0, "none", False),
        (128, 8, 0, "none", False),
        (64, 4, 0, "causal", False),
        (256, 8, 2, "none", False),
        (128, 4, 4, "causal", False),
        (64, 6, 0, "none", True),
    ]
    return [
        dict(head_size=h, num_query_heads=q, num_kv_heads=k, mask_mode=m, v_lds_stage=v)
        for (h, q, k, m, v) in rows
    ]


def cfgs_attention_unified():
    rows = [
        (64, 32, "bf16", 64, 8, 128, 512, 512, True, 0.0, 0, 1),
        (128, 32, "bf16", 64, 8, 256, 1024, 1024, True, 0.0, 0, 1),
        (256, 16, "fp16", 32, 4, 64, 256, 256, False, 0.0, 0, 1),
        (64, 16, "bf16", 16, 2, 512, 2048, 2048, True, 0.0, 128, 1),
        (128, 32, "bf16", 64, 8, 1, 1, 4096, False, 0.0, 0, 1),
        (64, 32, "bf16", 128, 16, 1024, 4096, 4096, True, 50.0, 0, 1),
    ]
    return [
        dict(
            head_size=hs,
            block_size=bs,
            dtype=dt,
            num_query_heads=qh,
            num_kv_heads=kv,
            total_q=tq,
            max_seqlen_q=sq,
            max_seqlen_k=sk,
            use_sinks=sink,
            softcap=sc,
            sliding_window=sw,
            num_seqs=ns,
        )
        for (hs, bs, dt, qh, kv, tq, sq, sk, sink, sc, sw, ns) in rows
    ]


def cfgs_fmha_mfma():
    rows = [
        (64, 8, 8, 256, 256, "f16", "none", 0),
        (128, 16, 16, 512, 512, "f16", "none", 0),
        (64, 8, 8, 256, 1024, "f16", "causal", 0),
        (256, 32, 32, 512, 2048, "f16", "sliding_window", 512),
        (192, 12, 12, 128, 512, "f16", "none", 0),
    ]
    return [
        dict(
            head_size=h,
            num_query_heads=q,
            num_kv_heads=k,
            seqlen_q=sq,
            seqlen_k=sk,
            dtype=d,
            mask_mode=m,
            sliding_window=sw,
        )
        for (h, q, k, sq, sk, d, m, sw) in rows
    ]


def cfgs_fmha_fwd_fp8():
    rows = [
        (64, 4, 4, "f16", "none", "fp8e4m3", 16, 64, None),
        (64, 2, 2, "f16", "causal", "fp8e4m3", 32, 128, None),
        (128, 8, 4, "bf16", "none", "bf8e5m2", 16, 64, None),
        (256, 4, 1, "f16", "sliding_window", "fp8e4m3", 48, 256, 4),
        (32, 16, 16, "f16", "none", "fp8e4m3", 64, 256, None),
        (64, 8, 2, "bf16", "none", "bf8e5m2", 32, 512, None),
    ]
    out = []
    for h, q, k, dt, m, kv, sq, sk, weu in rows:
        # all use shape_make(...,16,64); set sliding_window for cfg3
        d = dict(
            head_size=h,
            num_query_heads=q,
            num_kv_heads=k,
            block_size_q=16,
            block_size_k=64,
            dtype=dt,
            mask_mode=m,
            kv_dtype=kv,
            seqlen_q=sq,
            seqlen_k=sk,
        )
        if m == "sliding_window":
            d["sliding_window"] = 32
        if weu is not None:
            d["waves_per_eu"] = weu
        out.append(d)
    return out


def cfgs_fmha_bwd():
    rows = [
        (64, 4, 4, "f16", "none", 16, 16, 0),
        (128, 8, 4, "f16", "none", 32, 64, 0),
        (256, 16, 8, "bf16", "causal", 64, 64, 0),
        (64, 2, 2, "f16", "sliding_window", 128, 128, 16),
        (192, 8, 2, "bf16", "none", 256, 256, 0),
        (128, 1, 1, "f16", "causal", 128, 256, 0),
    ]
    out = []
    for h, q, k, dt, m, sq, sk, sw in rows:
        d = dict(
            head_size=h,
            num_query_heads=q,
            num_kv_heads=k,
            dtype=dt,
            mask_mode=m,
            seqlen_q=sq,
            seqlen_k=sk,
        )
        if m == "sliding_window":
            d["sliding_window"] = sw
        out.append(d)
    return out


def cfgs_fmha_head_grouping():
    rows = [
        (64, 32, 8, "f16", "none", 16, 16, 0),
        (128, 16, 1, "f16", "causal", 32, 32, 0),
        (64, 8, 4, "bf16", "none", 32, 32, 0),
        (256, 32, 4, "f16", "sliding_window", 16, 16, 512),
        (128, 8, 2, "bf16", "none", 48, 64, 0),
        (64, 24, 8, "f16", "causal", 64, 64, 0),
    ]
    out = []
    for h, q, k, dt, m, sq, sk, sw in rows:
        d = dict(
            head_size=h,
            num_query_heads=q,
            num_kv_heads=k,
            dtype=dt,
            mask_mode=m,
            seqlen_q=sq,
            seqlen_k=sk,
        )
        if m == "sliding_window":
            d["sliding_window"] = sw
        out.append(d)
    return out


def cfgs_sparse_attention():
    return [
        dict(
            kind="jenga",
            head_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            dtype="f16",
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=64,
        ),
        dict(
            kind="jenga",
            head_size=128,
            num_query_heads=16,
            num_kv_heads=16,
            dtype="bf16",
            seqlen_q=64,
            seqlen_k=256,
            block_q=2,
            block_k=64,
        ),
        dict(
            kind="vsa",
            head_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            dtype="f16",
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=64,
            max_blocks_per_q=16,
        ),
        dict(
            kind="vsa",
            head_size=128,
            num_query_heads=16,
            num_kv_heads=16,
            dtype="f16",
            seqlen_q=64,
            seqlen_k=256,
            block_q=2,
            block_k=64,
            max_blocks_per_q=32,
            use_wave_ballot_scatter=True,
        ),
        dict(
            kind="jenga",
            head_size=256,
            num_query_heads=32,
            num_kv_heads=8,
            dtype="f16",
            seqlen_q=96,
            seqlen_k=512,
            block_q=4,
            block_k=128,
        ),
        dict(
            kind="vsa",
            head_size=256,
            num_query_heads=32,
            num_kv_heads=32,
            dtype="bf16",
            seqlen_q=128,
            seqlen_k=1024,
            block_q=8,
            block_k=64,
            max_blocks_per_q=24,
            use_wave_ballot_scatter=False,
        ),
    ]


def cfgs_sage_attention():
    def pb(sb, sbatch, sh, sblk):
        return dict(
            layout="per_block",
            scale_block=sb,
            stride_batch=sbatch,
            stride_head=sh,
            stride_block=sblk,
        )

    def ph():
        return dict(
            layout="per_head",
            scale_block=0,
            stride_batch=8,
            stride_head=1,
            stride_block=1,
        )

    return [
        dict(
            head_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="f16",
            mask_mode="none",
            quant_mode="fp16_bf16",
            q_scale=pb(16, 128, 8, 1),
            k_scale=pb(64, 128, 8, 1),
            seqlen_q=16,
            seqlen_k=64,
        ),
        dict(
            head_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="bf16",
            mask_mode="none",
            quant_mode="fp8_bf16",
            q_scale=pb(16, 128, 8, 1),
            k_scale=pb(64, 128, 8, 1),
            seqlen_q=16,
            seqlen_k=64,
        ),
        dict(
            head_size=64,
            num_query_heads=8,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="f16",
            mask_mode="none",
            quant_mode="i8_fp8_bf16",
            q_scale=ph(),
            k_scale=ph(),
            seqlen_q=16,
            seqlen_k=64,
        ),
        dict(
            head_size=128,
            num_query_heads=8,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="bf16",
            mask_mode="none",
            quant_mode="i4_fp8_bf16",
            q_scale=pb(16, 128, 8, 1),
            k_scale=pb(64, 128, 8, 1),
            seqlen_q=32,
            seqlen_k=128,
        ),
        dict(
            head_size=256,
            num_query_heads=16,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="f16",
            mask_mode="causal",
            quant_mode="fp16_bf16",
            q_scale=pb(32, 256, 16, 1),
            k_scale=pb(64, 256, 16, 1),
            seqlen_q=64,
            seqlen_k=64,
        ),
        dict(
            head_size=128,
            num_query_heads=8,
            num_kv_heads=8,
            block_size_q=16,
            block_size_k=64,
            dtype="bf16",
            mask_mode="none",
            quant_mode="fp8_bf16",
            q_scale=ph(),
            k_scale=ph(),
            seqlen_q=32,
            seqlen_k=128,
        ),
    ]


def cfgs_gfx942_attention_tiled_2d():
    rows = [
        (64, 32, 32, 32, "bf16", True, 2048, False),
        (128, 32, 32, 32, "fp16", True, 0, True),
        (64, 128, 16, 4, "bf16", False, 4096, False),
        (128, 64, 32, 8, "fp16", True, 2048, True),
        (256, 32, 32, 32, "bf16", False, 0, False),
    ]
    return [
        dict(
            head_size=h,
            block_size=b,
            num_query_heads=q,
            num_kv_heads=k,
            dtype=d,
            use_sinks=s,
            sliding_window=sw,
            has_softcap=sc,
        )
        for (h, b, q, k, d, s, sw, sc) in rows
    ]


def cfgs_gfx950_attention_tiled_2d():
    """Full 38-config table mirroring gfx950_attention_tiled_2d_emit.c make_spec."""

    def base(h, b, q, k, dt, sink, sw, sc, **extra):
        d = dict(
            head_size=h,
            block_size=b,
            num_query_heads=q,
            num_kv_heads=k,
            dtype=dt,
            use_sinks=sink,
            sliding_window=sw,
            has_softcap=sc,
        )
        d.update(extra)
        return d

    return [
        base(64, 16, 1, 1, "bf16", False, 0, False),  # 0
        base(128, 16, 1, 1, "bf16", False, 0, False),  # 1
        base(256, 16, 1, 1, "bf16", False, 0, False),  # 2
        base(64, 16, 16, 1, "bf16", False, 0, False),  # 3
        base(64, 16, 2, 1, "bf16", False, 0, False),  # 4
        base(64, 32, 32, 32, "bf16", False, 0, False),  # 5
        base(64, 32, 32, 32, "fp16", True, 2048, True),  # 6
        base(64, 32, 32, 32, "bf16", True, 1, False),  # 7
        base(128, 32, 32, 32, "fp16", True, 0, True),  # 8
        base(256, 32, 32, 32, "bf16", False, 0, False),  # 9
        base(64, 64, 32, 32, "bf16", False, 0, False),  # 10
        base(64, 32, 7, 7, "bf16", False, 0, False),  # 11
        base(64, 32, 64, 8, "bf16", False, 0, False),  # 12
        base(64, 32, 40, 8, "bf16", False, 0, False),  # 13
        base(64, 32, 128, 1, "bf16", False, 0, False),  # 14
        base(64, 32, 32, 32, "bf16", False, 0, False, use_qq_bias=True),  # 15
        base(64, 32, 32, 32, "bf16", False, 0, False, use_alibi=True),  # 16
        base(
            64, 32, 32, 32, "fp16", True, 512, True, use_alibi=True, use_qq_bias=True
        ),  # 17
        base(64, 64, 64, 8, "bf16", False, 0, False, num_warps=8),  # 18
        base(64, 32, 32, 32, "bf16", False, 0, False, num_warps=2),  # 19
        base(64, 32, 64, 8, "bf16", False, 0, False, num_warps=4),  # 20
        base(64, 32, 32, 32, "bf16", False, 0, False, tile_size=64),  # 21
        base(64, 32, 32, 32, "bf16", False, 0, False, tile_size=128),  # 22
        base(
            128, 32, 32, 32, "fp16", False, 0, False, num_warps=2, tile_size=128
        ),  # 23
        base(64, 32, 32, 32, "bf16", False, 0, False, waves_per_eu=2),  # 24
        base(64, 32, 32, 32, "bf16", False, 0, False, num_seqs=1),  # 25
        base(64, 32, 32, 32, "bf16", False, 0, False, num_seqs=257),  # 26
        base(
            64,
            32,
            32,
            32,
            "bf16",
            False,
            0,
            False,
            kv_storage_dtype="fp8e4m3",
            use_fp8_mfma_pv=True,
        ),  # 27
        base(64, 32, 32, 32, "bf16", False, 0, False, use_i64_kv_addr=True),  # 28
        base(64, 32, 32, 32, "bf16", False, 0, False, use_register_pv=True),  # 29
        base(64, 32, 32, 32, "bf16", False, 0, False, kv_storage_dtype="fp8e4m3"),  # 30
        base(
            64,
            32,
            32,
            32,
            "bf16",
            False,
            0,
            False,
            kv_storage_dtype="fp8e4m3",
            use_fp8_mfma_qk=True,
        ),  # 31
        base(
            128,
            32,
            32,
            32,
            "fp16",
            False,
            0,
            False,
            use_mfma_32x32=True,
            block_m_per_warp=32,
            tile_size=64,
        ),  # 32
        base(
            64,
            32,
            32,
            32,
            "bf16",
            False,
            0,
            False,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_invariant_hoist=True,
            use_transposed_mask_once=True,
            use_grouped_kv2_softmax=True,
            block_m_per_warp=32,
            tile_size=64,
        ),  # 33
        base(
            128,
            32,
            32,
            32,
            "bf16",
            False,
            0,
            False,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_invariant_hoist=True,
            use_transposed_mask_once=True,
            use_grouped_kv2_softmax=True,
            block_m_per_warp=32,
            tile_size=64,
        ),  # 34
        base(
            64,
            32,
            64,
            8,
            "bf16",
            False,
            0,
            False,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
            use_transposed_invariant_hoist=True,
            use_transposed_mask_once=True,
            use_grouped_kv2_softmax=True,
            num_warps=4,
            block_m_per_warp=32,
            tile_size=64,
        ),  # 35
        base(64, 32, 32, 32, "bf16", False, 0, False, use_early_v_schedule=True),  # 36
        base(
            64,
            32,
            64,
            8,
            "bf16",
            False,
            0,
            False,
            num_warps=4,
            block_m_per_warp=16,
            tile_size=64,
            use_fast_paged_kv_desc=True,
        ),  # 37
    ]


def cfgs_gfx950_attention_tiled_2d_fastkv_regp():
    # make_base + additive deltas, cfgs 0..5
    base = {}
    deltas = [
        {},
        {"use_transposed_half_local_pv": True},
        {"use_mfma32_skip_legacy_qreg": True},
        {"use_transposed_half_local_pv": True, "use_mfma32_skip_legacy_qreg": True},
        {"use_agpr_alloc_zero": True, "use_transposed_half_local_pv": True},
        {"use_grouped_kv2_softmax": True},
    ]
    out = []
    for dl in deltas:
        d = dict(base)
        d.update(dl)
        out.append(d)
    return out


def cfgs_gfx942_attention_tiled_3d():
    rows = [
        (64, 16, 8, 8, "fp16", 8, False, 0, False, None, False, False),
        (128, 32, 16, 16, "bf16", 16, False, 0, False, None, False, False),
        (256, 64, 32, 8, "fp16", 8, False, 4096, False, None, False, False),
        (128, 32, 32, 8, "bf16", 16, True, 0, True, "fp8e4m3", False, False),
        (64, 16, 16, 4, "fp16", 8, False, 2048, False, None, True, False),
    ]
    out = []
    for h, b, q, k, dt, seg, sink, sw, sc, kv, alibi, qqb in rows:
        d = dict(
            head_size=h,
            block_size=b,
            num_query_heads=q,
            num_kv_heads=k,
            dtype=dt,
            num_segments=seg,
            use_sinks=sink,
            sliding_window=sw,
            has_softcap=sc,
            use_alibi=alibi,
            use_qq_bias=qqb,
        )
        if kv is not None:
            d["kv_storage_dtype"] = kv
        out.append(d)
    return out


def cfgs_gfx950_attention_tiled_3d():
    return cfgs_gfx942_attention_tiled_3d()


def cfgs_deep_fused_conv_pool():
    # all 8 emitter configs; 0..5 are reject-only (engine rejects them),
    # 6 (gfx950) + 7 (gfx1201) are the buildable ones.
    def mk(h, w, c, k0, k1, tn, tk, wt, ws):
        return dict(
            n=1,
            h=h,
            w=w,
            c=c,
            k0=k0,
            k1=k1,
            r=3,
            s=3,
            pool_tile_h=4,
            pool_tile_w=8,
            tile_n=tn,
            tile_k=tk,
            warp_m=2,
            warp_n=1,
            warp_tile_m=wt,
            warp_tile_n=wt,
            warp_tile_k=16,
            wave_size=ws,
        )

    return [
        (mk(112, 112, 64, 64, 64, 32, 16, 32, 64), "gfx950"),
        (mk(56, 56, 128, 128, 128, 32, 16, 32, 64), "gfx950"),
        (mk(28, 28, 256, 256, 256, 32, 16, 32, 64), "gfx950"),
        (mk(112, 112, 64, 64, 64, 32, 16, 16, 32), "gfx1201"),
        (
            mk(56, 56, 32, 32, 32, 32, 16, 32, 64) | dict(cache_input_footprint=True),
            "gfx950",
        ),
        (
            mk(28, 28, 64, 64, 64, 32, 16, 32, 64)
            | dict(direct_conv0_from_input_cache=True),
            "gfx950",
        ),
        (mk(64, 128, 8, 16, 16, 16, 16, 16, 64), "gfx950"),
        (mk(64, 128, 8, 16, 16, 16, 16, 16, 32), "gfx1201"),
    ]


# (family, lower_fn_name, configs, arch). arch=None => use per-config arch
# (configs are (dict, arch) tuples). Otherwise configs are plain dicts.
FAMILIES = [
    ("layernorm2d", "layernorm2d_lower_llvm", cfgs_layernorm2d(), "gfx950"),
    ("rmsnorm2d", "rmsnorm2d_lower_llvm", cfgs_rmsnorm2d(), "gfx950"),
    (
        "add_rmsnorm2d_bf16",
        "add_rmsnorm2d_bf16_lower_llvm",
        cfgs_add_rmsnorm2d_bf16(),
        "gfx950",
    ),
    (
        "add_rmsnorm2d_rdquant",
        "add_rmsnorm2d_rdquant_lower_llvm",
        cfgs_add_rmsnorm2d_rdquant(),
        "gfx950",
    ),
    ("elementwise", "elementwise_lower_llvm", cfgs_elementwise(), "gfx950"),
    ("reduce", "reduce_lower_llvm", cfgs_reduce(), "gfx950"),
    ("pooling", "pooling_lower_llvm", cfgs_pooling(), "gfx950"),
    ("transpose", "transpose_lower_llvm", cfgs_transpose(), "gfx950"),
    ("permute_nd", "permute_nd_lower_llvm", cfgs_permute_nd(), "gfx950"),
    ("smoothquant", "smoothquant_lower_llvm", cfgs_smoothquant(), "gfx950"),
    ("topk_softmax", "topk_softmax_lower_llvm", cfgs_topk_softmax(), "gfx950"),
    ("moe_smoothquant", "moe_smoothquant_lower_llvm", cfgs_moe_smoothquant(), "gfx950"),
    ("moe_fused_mega", "moe_fused_mega_lower_llvm", cfgs_moe_fused_mega(), "gfx950"),
    (
        "moe_fused_mega_fp8",
        "moe_fused_mega_fp8_lower_llvm",
        cfgs_moe_fused_mega_fp8(),
        "gfx950",
    ),
    ("fused_moe", "fused_moe_lower_llvm", cfgs_fused_moe(), "gfx950"),
    ("moe_sorting", "moe_sorting_lower_llvm", cfgs_moe_sorting(), "gfx950"),
    ("moe_gemm_fused", "moe_gemm_fused_lower_llvm", cfgs_moe_gemm_fused(), "gfx950"),
    (
        "gfx1151_wmma_gemm",
        "gfx1151_wmma_gemm_lower_llvm",
        cfgs_gfx1151_wmma_gemm(),
        "gfx1151",
    ),
    (
        "gfx1151_wmma_gemm_int8",
        "gfx1151_wmma_gemm_int8_lower_llvm",
        cfgs_gfx1151_wmma_gemm_int8(),
        "gfx1151",
    ),
    (
        "gfx1151_wmma_gemm_iu8",
        "gfx1151_wmma_gemm_iu8_lower_llvm",
        cfgs_gfx1151_wmma_gemm_iu8(),
        "gfx1151",
    ),
    (
        "gfx1151_wmma_gemm_iu8_dequant",
        "gfx1151_wmma_gemm_iu8_dequant_lower_llvm",
        cfgs_gfx1151_wmma_gemm_iu8_dequant(),
        "gfx1151",
    ),
    (
        "gfx1151_wmma_fmha_fwd",
        "gfx1151_wmma_fmha_fwd_lower_llvm",
        cfgs_gfx1151_wmma_fmha_fwd(),
        "gfx1151",
    ),
    (
        "gfx1201_wmma_gemm",
        "gfx1201_wmma_gemm_lower_llvm",
        cfgs_gfx1201_wmma_gemm(),
        "gfx1201",
    ),
    (
        "attention_unified",
        "attention_unified_lower_llvm",
        cfgs_attention_unified(),
        "gfx950",
    ),
    ("fmha_mfma", "fmha_mfma_lower_llvm", cfgs_fmha_mfma(), "gfx950"),
    ("fmha_fwd_fp8", "fmha_fwd_fp8_lower_llvm", cfgs_fmha_fwd_fp8(), "gfx950"),
    ("fmha_bwd", "fmha_bwd_lower_llvm", cfgs_fmha_bwd(), "gfx950"),
    (
        "fmha_head_grouping",
        "fmha_head_grouping_lower_llvm",
        cfgs_fmha_head_grouping(),
        "gfx950",
    ),
    (
        "sparse_attention",
        "sparse_attention_lower_llvm",
        cfgs_sparse_attention(),
        "gfx950",
    ),
    ("sage_attention", "sage_attention_lower_llvm", cfgs_sage_attention(), "gfx950"),
    (
        "gfx942_attention_tiled_2d",
        "gfx942_attention_tiled_2d_lower_llvm",
        cfgs_gfx942_attention_tiled_2d(),
        "gfx942",
    ),
    (
        "gfx950_attention_tiled_2d",
        "gfx950_attention_tiled_2d_lower_llvm",
        cfgs_gfx950_attention_tiled_2d(),
        "gfx950",
    ),
    (
        "gfx950_attention_tiled_2d_fastkv_regp",
        "gfx950_attention_tiled_2d_fastkv_regp_lower_llvm",
        cfgs_gfx950_attention_tiled_2d_fastkv_regp(),
        "gfx950",
    ),
    (
        "gfx942_attention_tiled_3d",
        "gfx942_attention_tiled_3d_lower_llvm",
        cfgs_gfx942_attention_tiled_3d(),
        "gfx942",
    ),
    (
        "gfx950_attention_tiled_3d",
        "gfx950_attention_tiled_3d_lower_llvm",
        cfgs_gfx950_attention_tiled_3d(),
        "gfx950",
    ),
    (
        "deep_fused_conv_pool",
        "deep_fused_conv_pool_lower_llvm",
        cfgs_deep_fused_conv_pool(),
        None,
    ),
]


def main():
    fam_validated = 0
    fam_empty_only = 0
    fam_failed = 0
    total_match = 0
    total_mismatch = 0
    print(
        f"{'family':<40}{'nonempty':<10}{'match':<8}{'mismatch':<10}{'empty':<7}status"
    )
    for fam, lower_name, cfgs, fam_arch in FAMILIES:
        lower = getattr(ckc_engine, lower_name)
        n_nonempty = n_match = n_mismatch = n_empty = 0
        details = []
        for idx, cfg in enumerate(cfgs):
            if fam_arch is None:
                d, arch = cfg
            else:
                d, arch = cfg, fam_arch
            std, err = run_emitter(fam, idx)
            if std is None:
                n_mismatch += 1
                details.append(f"  cfg{idx}: emitter exec error: {err}")
                continue
            if std == "":
                n_empty += 1  # standalone rejects -> not counted toward validation
                continue
            n_nonempty += 1
            try:
                got = lower(d, arch=arch)
            except Exception as e:  # noqa: BLE001
                n_mismatch += 1
                details.append(f"  cfg{idx}: binding raised: {str(e)[:120]}")
                continue
            if got == std:
                n_match += 1
            else:
                n_mismatch += 1
                details.append(
                    f"  cfg{idx}: DIFF cpp={sha(got)} std={sha(std)} "
                    f"(len {len(got)} vs {len(std)})"
                )
        total_match += n_match
        total_mismatch += n_mismatch
        if n_nonempty == 0:
            status = "EMPTY-ONLY (not validated)"
            fam_empty_only += 1
        elif n_mismatch == 0 and n_match == n_nonempty:
            status = "VALIDATED"
            fam_validated += 1
        else:
            status = "FAILED"
            fam_failed += 1
        print(
            f"{fam:<40}{n_nonempty:<10}{n_match:<8}{n_mismatch:<10}{n_empty:<7}{status}"
        )
        for line in details:
            print(line)
    print()
    print(f"families VALIDATED (non-empty, all buildable match): {fam_validated}")
    print(f"families EMPTY-ONLY (engine rejects all configs):    {fam_empty_only}")
    print(f"families FAILED:                                     {fam_failed}")
    print(f"configs matched non-empty: {total_match}   mismatched: {total_mismatch}")
    return 0 if fam_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
