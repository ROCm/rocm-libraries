# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Focused gfx942 bf16 wide-K (32x32x8) transposed flash validation.

Reuses parity_unified_attention.py's data gen + fp32 reference + launch path,
but builds several specs per shape and compares them all vs the fp32 reference
AND measures graph-mode kernel time:

  * narrow      : the current shipped bf16 path (16x16x16, the BEFORE)
  * wide        : the bf16 32x32x8 transposed path with the naive strided-V feed
  * wide_cfonly : wide + conflict-free V store (cfvst) -- the PRODUCTION D64
                  config; ~3% faster than naive-V on D64 (MI300X, steady-state)
  * wide_cf     : cfvst + sliced-K ring -- ported from fp16 but the ring
                  REGRESSES bf16 (kept for A/B); not used in production
  * flash       : PyTorch SDPA FLASH_ATTENTION backend (the bar to beat)

Run:  python rocke/examples/gfx942/attention/bf16_wide_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))  # repo python/ root

import importlib.util  # noqa: E402

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "parity_unified_attention", str(HERE / "parity_unified_attention.py")
)
H = importlib.util.module_from_spec(_spec)
sys.modules["parity_unified_attention"] = H
_spec.loader.exec_module(H)
from rocke import compile_kernel  # noqa: E402
from rocke.instances import build_unified_attention_2d_tiled  # noqa: E402
from rocke.instances.gfx942.attention_tiled_2d import (  # noqa: E402
    UnifiedAttention2DTiledSpec,
)
from rocke.instances.common.attention_unified import _attn_signature  # noqa: E402
from rocke.runtime import KernelLauncher  # noqa: E402

BS = H.BLOCK_SIZE


def _base(s):
    return dict(
        head_size=s.head_size,
        block_size=BS,
        num_query_heads=s.heads,
        num_kv_heads=s.kv_heads,
        dtype=s.dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        tile_size=64,
    )


def make_spec(s, kind):
    b = _base(s)
    if kind == "narrow":
        if s.head_size == 64:
            return UnifiedAttention2DTiledSpec(**b, num_warps=4, block_m_per_warp=32)
        return UnifiedAttention2DTiledSpec(**b, num_warps=2, block_m_per_warp=16)
    if kind == "wide":
        if s.head_size == 128:
            return UnifiedAttention2DTiledSpec(
                **b,
                num_warps=2,
                block_m_per_warp=32,
                use_mfma_32x32x8=True,
                use_transposed_qk_32x32=True,
                use_k_single_buffer=True,
            )
        return UnifiedAttention2DTiledSpec(
            **b,
            num_warps=4,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
        )
    if kind == "wide4":
        # naive-V wide, but nw=4/BLOCK_M=128 so it matches wide_cf geometry
        # (isolates the cfvst/ring effect from the BLOCK_M change on D128).
        # D128 nw=4 naive-V needs K single-buffer to fit LDS.
        if s.head_size == 128:
            return UnifiedAttention2DTiledSpec(
                **b,
                num_warps=4,
                block_m_per_warp=32,
                use_mfma_32x32x8=True,
                use_transposed_qk_32x32=True,
                use_k_single_buffer=True,
            )
        return UnifiedAttention2DTiledSpec(
            **b,
            num_warps=4,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
        )
    if kind == "wide_cfonly":
        # cfvst (conflict-free V store) WITHOUT the sliced-K ring. Isolates the
        # V-feed effect. nw=4 + K single-buffer so D128 fits (no ring to stage K).
        return UnifiedAttention2DTiledSpec(
            **b,
            num_warps=4,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_conflict_free_v_store=True,
            use_k_single_buffer=(s.head_size == 128),
        )
    if kind == "wide_cf":
        # bf16 wide path + conflict-free V store (cfvst) + sliced-K ring,
        # ported from the fp16 transposed-x8 flash family. nw=4, T=64 for both
        # head sizes (the sliced-K ring keeps D128 LDS under the 64 KB cap).
        return UnifiedAttention2DTiledSpec(
            **b,
            num_warps=4,
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_conflict_free_v_store=True,
            use_k_sliced_ring=True,
        )
    raise ValueError(kind)


def build_launcher(s, spec, kind):
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=_attn_signature(
            s.dtype, include_bt_stride=True, include_qq_bias_stride=True
        ),
        cache_key=("bf16wide", spec.kernel_name(), kind),
    )
    return launcher


def run_ck(s, data, launcher, spec, warmup, attempts):
    return H._run_rocke(s, data, launcher, spec, warmup=warmup, attempts=attempts)


def flash_ms(s, data, warmup, attempts):
    # Dense flash SDPA on the equivalent [B,Hq,Sq,D] problem, causal.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # Build dense tensors from shape (uniform per-seq); the packed data["query"]
    # is not needed here since flash runs on the equivalent dense [B,Hq,Sq,D] problem.
    B, Hq, Hkv, D = s.batch, s.heads, s.kv_heads, s.head_size
    Sq, Sk = s.seqlen_q, s.seqlen_k
    dt = torch.float16 if s.dtype == "fp16" else torch.bfloat16
    qd = torch.randn(B, Hq, Sq, D, device="cuda", dtype=dt) * 0.1
    kd = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dt) * 0.1
    vd = torch.randn(B, Hkv, Sk, D, device="cuda", dtype=dt) * 0.1
    if Hkv != Hq:
        kd = kd.repeat_interleave(Hq // Hkv, dim=1)
        vd = vd.repeat_interleave(Hq // Hkv, dim=1)
    scale = D**-0.5

    def once():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(
                qd, kd, vd, is_causal=True, scale=scale
            )

    from rocke.runtime import time_launches

    stream = int(torch.cuda.current_stream().cuda_stream)
    return time_launches(once, warmup=warmup, iters=attempts, stream=stream)


def main():
    shapes = [
        H.Shape("bf16_d64_S1024", "bf16", 1024, 1024, 64, 64, 8, 1, True, "perf"),
        H.Shape("bf16_d64_S2048", "bf16", 2048, 2048, 64, 64, 8, 1, True, "perf"),
        H.Shape("bf16_d64_S4096", "bf16", 4096, 4096, 64, 64, 8, 1, True, "perf"),
        H.Shape("bf16_d128_S1024", "bf16", 1024, 1024, 128, 32, 4, 1, True, "perf"),
        H.Shape("bf16_d128_S2048", "bf16", 2048, 2048, 128, 32, 4, 1, True, "perf"),
        H.Shape("bf16_d128_S4096", "bf16", 4096, 4096, 128, 32, 4, 1, True, "perf"),
    ]
    warmup, attempts = 10, 50
    tol = 4e-2
    print(
        f"{'shape':18} {'kind':7} {'corr':6} {'max_abs':9} {'ms':9} {'TFLOPS':8} {'vs_flash':9}"
    )
    for s in shapes:
        data = H.make_inputs(s)
        ref = H.ref_paged_attn(
            data["query"],
            data["key_cache"],
            data["value_cache"],
            [s.seqlen_q] * s.batch,
            [s.seqlen_k] * s.batch,
            data["block_tables"],
            data["scale"],
        )
        fms = flash_ms(s, data, warmup, attempts)
        results = {}
        for kind in ("narrow", "wide", "wide_cfonly", "wide_cf"):
            try:
                spec = make_spec(s, kind)
                lau = build_launcher(s, spec, kind)
                out, ms = run_ck(s, data, lau, spec, warmup, attempts)
                m = H.compare(ref, out)
                ok = m["max_abs"] <= tol
                tf = H.attention_tflops(s, ms)
                vs = fms / ms if ms > 0 else 0.0
                results[kind] = (ok, m["max_abs"], ms, tf, vs)
            except Exception as e:
                results[kind] = ("ERR", str(e)[:60], 0, 0, 0)
        ftf = H.attention_tflops(s, fms)
        print(
            f"{s.name:18} {'flash':7} {'-':6} {'-':9} {fms:9.4f} {ftf:8.1f} {'1.00':>9}"
        )
        for kind in ("narrow", "wide", "wide_cfonly", "wide_cf"):
            r = results[kind]
            if r[0] == "ERR":
                print(f"{s.name:18} {kind:7} ERR    {r[1]}")
            else:
                ok, ma, ms, tf, vs = r
                print(
                    f"{s.name:18} {kind:7} {('PASS' if ok else 'FAIL'):6} {ma:9.4f} {ms:9.4f} {tf:8.1f} {vs:9.3f}"
                )
        print()


if __name__ == "__main__":
    main()
