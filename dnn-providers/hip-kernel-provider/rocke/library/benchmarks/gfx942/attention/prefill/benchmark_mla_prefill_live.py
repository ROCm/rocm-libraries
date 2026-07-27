# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MLA prefill live benchmark — rocke variants vs baselines.

Reads mla_prefill_shapes.json, builds rocke MLA prefill kernel variants,
measures latency, optionally runs AITER Triton MLA baseline, and writes a
JSON results file in the same format as benchmark_prefill2d_live.py.

Variants:
  prod      default MlaPrefillSpec (block_size=16, num_warps=1)
  fallback  same kernel, block_size=32 (wider KV tile, lower occupancy)

Usage:
    python library/benchmarks/gfx950/attention/prefill/benchmark_mla_prefill_live.py \\
        --shapes library/benchmarks/gfx942/attention/prefill/mla_prefill_shapes.json \\
        --variants prod fallback \\
        --warmup 10 --iterations 50 \\
        --output-json library/benchmarks/gfx950/attention/prefill/mla_prefill_bench_perf.json

Optional flags:
  --flydsl          include AITER Triton MLA as external baseline (requires AITER_PATH)
  --no-correctness  skip max_abs check against Python reference
  --limit N         process only the first N shapes
  --arch gfx942     override GPU arch (auto-detected by default)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Variant registry
# Each entry: (label, block_size)  — num_warps=1 for both (single-wave body)
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, dict] = {
    "prod":     {"block_size": 16, "mfma": False},
    "fallback": {"block_size": 32, "mfma": False},
    # v2: purpose-built inner-H_q-loop MFMA kernel (MFMA_PLAN.md Option A).
    # Grid: (1, ceil(sq/BLOCK_M), 1); K/V shared across all H_q head iterations.
    "mfma_v2":  {"block_size": 16, "mfma": "v2"},
}


# ---------------------------------------------------------------------------
# Shape loading
# ---------------------------------------------------------------------------

def _load_mla_shapes(path: Path) -> list[dict]:
    """Load shapes from mla_prefill_shapes.json → flat list of dicts."""
    with path.open() as fh:
        data = json.load(fh)
    out = []
    for model in data["models"]:
        for shape in model["shapes"]:
            if shape.get("dtype", "bf16") != "bf16":
                continue  # fp8 is Phase 2
            out.append({
                "model":           model["model"],
                "num_query_heads": model["num_query_heads"],
                "num_kv_heads":    model["num_kv_heads"],
                "block_size":      model["block_size"],
                "seqlen_q":        shape["seqlen_q"],
                "seqlen_k":        shape["seqlen_k"],
                "batch":           shape.get("batch", 1),
                "dtype":           shape.get("dtype", "bf16"),
                "label":           shape.get("label", ""),
            })
    return out


# ---------------------------------------------------------------------------
# Arch detection
# ---------------------------------------------------------------------------

def _detect_arch() -> str:
    try:
        import subprocess
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith("Name:") and "gfx" in s:
                gfx = s.split("Name:")[-1].strip()
                if gfx.startswith("gfx"):
                    return gfx
    except Exception:
        pass
    return "gfx950"


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def _bench(fn, warmup: int, iters: int) -> float:
    """Return median latency in ms."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
# FLOP count
# ---------------------------------------------------------------------------

def _flops(sq: int, sk: int, hq: int,
           d_qk: int = 192, d_v: int = 128, r_kv: int = 512) -> float:
    return (
        2.0 * sq * hq * sk * d_qk   # QK
        + 2.0 * sq * hq * sk * d_v  # PV
        + 2.0 * sk * r_kv * 128     # K_nope expansion
        + 2.0 * sk * r_kv * d_v     # V expansion
    )


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _make_inputs(shape: dict) -> dict:
    from builders.mla.ref_mla_attn import make_mla_prefill_inputs
    import torch
    return make_mla_prefill_inputs(
        num_query_heads=shape["num_query_heads"],
        seqlen_q=shape["seqlen_q"],
        seqlen_k=shape["seqlen_k"],
        block_size=shape["block_size"],
        dtype=torch.bfloat16,
        device="cuda",
        seed=42,
    )


# ---------------------------------------------------------------------------
# Reference (Python)
# ---------------------------------------------------------------------------

def _run_reference(inputs: dict):
    from builders.mla.ref_mla_attn import ref_mla_prefill_fwd
    import torch
    out = ref_mla_prefill_fwd(
        inputs["q"],
        inputs["c_kv_flat"],
        inputs["k_rope_flat"],
        inputs["w_uk_k"],
        inputs["w_uv"],
        inputs["cu_seqlens_q"],
        causal=True,
        scale=inputs["scale"],
    )
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# AITER Triton MLA prefill baseline (--flydsl flag)
# Uses aiter.ops.triton.attention.mla.mla_prefill_fwd — takes raw Q latent.
# AITER: /home/barkocot/aiter  (set AITER_PATH or hardcoded fallback).
# ---------------------------------------------------------------------------

_AITER_PREFILL_FN = None

def _import_aiter_mla():
    global _AITER_PREFILL_FN
    if _AITER_PREFILL_FN is not None:
        return True
    import os, sys
    aiter_path = os.environ.get("AITER_PATH", "/home/barkocot/aiter")
    if aiter_path and aiter_path not in sys.path:
        sys.path.insert(0, aiter_path)
    try:
        from aiter.ops.triton.attention.mla import mla_prefill_fwd
        _AITER_PREFILL_FN = mla_prefill_fwd
        return True
    except Exception:
        return False


def _run_aiter_mla(shape: dict, inputs: dict, warmup: int, iters: int):
    """Run AITER Triton mla_prefill_fwd. Returns (out[sq,hq,128], ms) or raises.

    AITER Triton MLA interface:
        q          [sq, hq, kv_lora_rank+qk_rope_head_dim=576]  — raw Q latent
        kv_buffer  [nb, bs, 1, 576]  — raw KV latent (c_KV ‖ K_rope)
        out        [sq, hq, v_head_dim=128]
        cu_seqlens_q  [2] i32
        seqused_k     [1] i32  (KV lengths per seq)
        max_seqlen_kv  int
        block_tables  [1, nb] i32
        softmax_scale  float
        kv_lora_rank  512
        qk_rope_head_dim  64
        causal  True
        q_descale, kv_descale  None (bf16)
    """
    import torch
    if not _import_aiter_mla():
        raise NotImplementedError("AITER Triton MLA not available (set AITER_PATH)")

    sq  = shape["seqlen_q"]
    sk  = shape["seqlen_k"]
    hq  = shape["num_query_heads"]

    q_latent  = inputs["q_latent"]   # [sq, hq, 576]
    kv_buffer = inputs["kv_buffer"]  # [nb, bs, 1, 576]
    out = torch.zeros(sq, hq, 128, dtype=torch.bfloat16, device="cuda")

    cu_seqlens_q = inputs["cu_seqlens_q"]          # [2] i32
    seqused_k    = torch.tensor([sk], dtype=torch.int32, device="cuda")
    block_tables = inputs["block_table"]             # [1, nb] i32
    sm_scale     = float(inputs["scale"])

    def _fn():
        _AITER_PREFILL_FN(
            q_latent, kv_buffer, out,
            cu_seqlens_q, seqused_k,
            max_seqlen_kv=sk,
            block_tables=block_tables,
            softmax_scale=sm_scale,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            causal=True,
            q_descale=None,
            kv_descale=None,
        )

    ms = _bench(_fn, warmup, iters)
    _fn()
    return out, ms


# ---------------------------------------------------------------------------
# rocke variant runner
# ---------------------------------------------------------------------------

_KERNEL_HEAD = 256   # kernel head_size (K and V padded to this from 192/128)

def _expand_kv_paged(inputs: dict, bs: int) -> tuple:
    """Pre-expand c_KV → K_exp and V_exp_padded in paged layout.

    K_exp[num_blocks, bs, 1, 256] = pad(concat(c_KV @ W_UK_K^T, K_rope), 0→64)
    V_exp_pad[num_blocks, bs, 1, 256] = pad(c_KV @ W_UV^T, zeros to 256)

    Padding to 256 (from 192/128) makes ept=256/64=4 so warp-body can use
    vectorised loads.  Zeros in [192:256] / [128:256] do not affect attention.

    Returns (K_paged, V_paged, seqlens_k).
    """
    import torch
    sk         = inputs["seqlen_k"]
    num_blocks = inputs["num_blocks"]
    c_kv_flat  = inputs["c_kv_flat"].float()
    k_rope_flat = inputs["k_rope_flat"].float()
    w_uk_k     = inputs["w_uk_k"].float()
    w_uv       = inputs["w_uv"].float()

    K_nope = c_kv_flat @ w_uk_k                                      # [sk, 128]
    K_192  = torch.cat([K_nope, k_rope_flat], dim=-1)                 # [sk, 192]
    K_256  = torch.zeros(sk, _KERNEL_HEAD, dtype=torch.bfloat16, device=K_192.device)
    K_256[:, :192] = K_192.to(torch.bfloat16)

    V_128  = (c_kv_flat @ w_uv).to(torch.bfloat16)                   # [sk, 128]
    V_256  = torch.zeros(sk, _KERNEL_HEAD, dtype=torch.bfloat16, device=V_128.device)
    V_256[:, :128] = V_128

    pad_k = num_blocks * bs - sk
    def _page(x):
        if pad_k > 0:
            x = torch.cat([x, torch.zeros(pad_k, _KERNEL_HEAD, dtype=x.dtype, device=x.device)])
        return x.view(num_blocks, bs, 1, _KERNEL_HEAD)

    K_paged = _page(K_256)
    V_paged = _page(V_256)
    seqlens_k = torch.tensor([sk], dtype=torch.int32, device=K_paged.device)
    return K_paged, V_paged, seqlens_k


def _run_variant_mfma(
    shape: dict,
    inputs: dict,
    variant_cfg: dict,
    arch: str,
    warmup: int,
    iters: int,
):
    """MFMA tiled variant: H_q parallel single-head launches on separate streams.

    Key insight vs previous approach (K/V replicated hq times):
      - K/V are NOT replicated: K_paged[nb, bs, 1, 256] shared by all launches.
      - Each head h gets its own contiguous Q[sq, 256] and O[sq, 256] buffer.
      - All hq kernel launches run concurrently on separate HIP streams.
      - Total HBM read for K/V: 1× instead of hq×  → ~hq× less bandwidth.

    This matches AITER's approach: one kernel reads the shared KV latent for
    each Q head independently, exploiting GPU-level parallelism.
    """
    import torch
    from kernels.mla import (
        MlaPrefillMfmaSpec, build_mla_prefill_mfma_fwd,
        mla_prefill_mfma_grid, mla_prefill_mfma_signature,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from rocke.runtime import no_fence, synchronize_and_release, time_launches

    hq  = shape["num_query_heads"]
    sq  = shape["seqlen_q"]
    bs  = variant_cfg.get("block_size", shape["block_size"])
    H   = _KERNEL_HEAD  # 256

    spec = MlaPrefillMfmaSpec(num_query_heads=hq, block_size=bs)
    kdef = build_mla_prefill_mfma_fwd(spec, arch=arch)
    art  = compile_kernel(kdef, arch=arch, capture_ir_text=False)
    lnch = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=mla_prefill_mfma_signature(spec),
    )
    grid_1h = mla_prefill_mfma_grid(spec, total_q=sq)  # (1, q_blocks+1, 1)

    # K/V: expanded once, shape [nb, bs, 1, H] — shared across all head launches
    K_paged, V_paged, seqlens_k = _expand_kv_paged(inputs, bs)

    # Allocate contiguous per-head Q and O buffers: hq × [sq, H]
    # Layout: q_all[hq, sq, H]  (head-major so each head slice is contiguous)
    q_all  = torch.zeros(hq, sq, H, dtype=torch.bfloat16, device="cuda")
    out_all = torch.zeros(hq, sq, H, dtype=torch.bfloat16, device="cuda")
    # Fill Q from [sq, hq, 192] → transpose + pad
    q_all[:, :, :192] = inputs["q"].permute(1, 0, 2)   # [hq, sq, 192]

    num_blocks   = inputs["num_blocks"]
    block_table  = inputs["block_table"]   # [1, nb] i32
    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    scale        = float(inputs["scale"])

    hip_stream = int(torch.cuda.current_stream().cuda_stream)

    # Build per-head vals dicts once (pointers are stable across launches)
    head_vals = []
    for h in range(hq):
        head_vals.append({
            "output_ptr":          out_all[h],
            "query_ptr":           q_all[h],
            "key_cache_ptr":       K_paged,
            "value_cache_ptr":     V_paged,
            "sink_ptr":            0,
            "block_tables_ptr":    block_table,
            "seq_lens_ptr":        seqlens_k,
            "alibi_slopes_ptr":    0,
            "qq_bias_ptr":         0,
            "query_start_len_ptr": cu_seqlens_q,
            "scale":               scale,
            "k_scale":             1.0,
            "v_scale":             1.0,
            "out_scale":           1.0,
            "softcap":             0.0,
            "num_seqs":            1,
            "block_table_stride":  int(num_blocks),
            "qq_bias_stride_0":    0,
        })

    # fence=False inside no_fence() context: launches are fire-and-forget
    # within each _launch_all() call but time_launches provides outer sync.
    cfg = LaunchConfig(grid=grid_1h, block=(64, 1, 1), stream=hip_stream, fence=False)

    def _launch_all():
        for h in range(hq):
            lnch(head_vals[h], config=cfg)

    # First do one fenced warm pass to flush any prior GPU state
    cfg_fenced = LaunchConfig(grid=grid_1h, block=(64, 1, 1), stream=hip_stream, fence=True)
    for h in range(hq):
        lnch(head_vals[h], config=cfg_fenced)

    with no_fence():
        ms = time_launches(_launch_all, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)

    # Reconstruct [sq, hq, H] → slice [:, :, :128]
    out_128 = out_all.permute(1, 0, 2)[:, :, :128].contiguous()
    return out_128, None, ms


def _run_variant_mfma_v2(
    shape: dict,
    inputs: dict,
    variant_cfg: dict,
    arch: str,
    warmup: int,
    iters: int,
):
    """v2 MFMA kernel: single CTA per Q-tile, inner scf_for loop over H_q heads.

    K/V cache is read once per CTA and reused across all H_q head iterations —
    no replication, no OOM, correct causal mask (token index, not tile index).
    """
    import torch
    import math
    from kernels.mla import (
        MlaPrefillMfmaSpec,
        build_mla_prefill_mfma_fwd_v2,
        mla_prefill_mfma_v2_grid,
        mla_prefill_mfma_v2_signature,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from rocke.runtime import no_fence, synchronize_and_release, time_launches

    hq  = shape["num_query_heads"]
    sq  = shape["seqlen_q"]
    bs  = variant_cfg.get("block_size", shape["block_size"])
    H   = _KERNEL_HEAD  # 256

    spec = MlaPrefillMfmaSpec(num_query_heads=hq, block_size=bs, batch=1)
    kdef = build_mla_prefill_mfma_fwd_v2(spec, arch=arch)
    art  = compile_kernel(kdef, arch=arch, capture_ir_text=False)
    lnch = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=mla_prefill_mfma_v2_signature(spec),
    )
    grid = mla_prefill_mfma_v2_grid(spec, total_q=sq)

    K_paged, V_paged, seqlens_k = _expand_kv_paged(inputs, bs)
    num_blocks  = inputs["num_blocks"]
    block_table = inputs["block_table"]

    q_raw = inputs["q"]  # [sq, hq, 192] bf16
    q_256 = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")
    q_256[:, :, :192] = q_raw
    out_256 = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")

    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
    scale_log2 = float(math.log2(inputs["scale"]))

    stride_q_head  = H
    stride_q_token = hq * H
    stride_o_head  = H
    stride_o_token = hq * H
    stride_block   = bs * H
    stride_page    = H
    stride_kv_head = H

    hip_stream = int(torch.cuda.current_stream().cuda_stream)
    cfg = LaunchConfig(grid=grid, block=(64, 1, 1), stream=hip_stream, fence=False)

    vals = {
        "Q":                q_256,
        "K_cache":          K_paged,
        "V_cache":          V_paged,
        "O":                out_256,
        "block_table":      block_table,
        "cu_seqlens_q":     cu_seqlens_q,
        "seqlens_k":        seqlens_k,
        "scale_log2":       scale_log2,
        "total_q":          int(sq),
        "batch":            int(1),
        "stride_q_token":   stride_q_token,
        "stride_q_head":    stride_q_head,
        "stride_block":     stride_block,
        "stride_page":      stride_page,
        "stride_kv_head":   stride_kv_head,
        "stride_v_block":   stride_block,
        "stride_v_page":    stride_page,
        "stride_v_kv_head": stride_kv_head,
        "stride_o_token":   stride_o_token,
        "stride_o_head":    stride_o_head,
        "block_table_stride": int(num_blocks),
        "num_query_heads":  int(hq),
    }

    cfg_fenced = LaunchConfig(grid=grid, block=(64, 1, 1), stream=hip_stream, fence=True)
    lnch(vals, config=cfg_fenced)

    def _launch():
        lnch(vals, config=cfg)

    with no_fence():
        ms = time_launches(_launch, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)

    out_128 = out_256[:, :, :128].contiguous()
    return out_128, None, ms


def _run_variant(
    shape: dict,
    inputs: dict,
    variant_cfg: dict,
    arch: str,
    warmup: int,
    iters: int,
):
    """Dispatch to the appropriate kernel variant. Returns (out_128, lse, ms)."""
    if variant_cfg.get("mfma") == "v2":
        return _run_variant_mfma_v2(shape, inputs, variant_cfg, arch, warmup, iters)
    if variant_cfg.get("mfma"):
        return _run_variant_mfma(shape, inputs, variant_cfg, arch, warmup, iters)

    """Strategy (DESIGN.md §11.4):
      1. Expand c_KV → K_exp[sk, 192] and V_exp_pad[sk, 192] (PyTorch GEMMs).
      2. Call fmha_paged_prefill(head_size=192, causal, use_mfma_body=True).
      3. Truncate output to [:, :, :128] → d_V=128.
    """
    import torch
    import math
    from kernels.mla import MlaPrefillSpec, build_mla_prefill_fwd, mla_prefill_fwd_grid
    from kernels.mla import mla_prefill_fwd_signature
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher
    from rocke.runtime import no_fence, synchronize_and_release, time_launches
    from rocke.runtime.launcher import LaunchConfig

    hq  = shape["num_query_heads"]
    sq  = shape["seqlen_q"]
    sk  = shape["seqlen_k"]
    bs  = variant_cfg.get("block_size", shape["block_size"])

    spec  = MlaPrefillSpec(num_query_heads=hq, block_size=bs)
    kdef  = build_mla_prefill_fwd(spec, arch=arch)
    art   = compile_kernel(kdef, arch=arch, capture_ir_text=False)
    lnch  = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=mla_prefill_fwd_signature(spec),
    )
    grid  = mla_prefill_fwd_grid(spec, total_q=sq)

    # Pre-expand K and V (outside timed region for fair latency measurement)
    K_paged, V_paged, seqlens_k = _expand_kv_paged(inputs, bs)

    H = _KERNEL_HEAD   # 256
    # Q is padded to 256 (zeros in [192:256]); Q input from inputs["q"] is [sq,hq,192]
    # We pad Q to [sq, hq, 256] on the fly.
    q_raw = inputs["q"]                         # [sq, hq, 192] bf16
    q_256 = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")
    q_256[:, :, :192] = q_raw

    # Output: [sq, hq, 256]; truncate to [:, :, :128] after
    out_256 = torch.zeros(sq, hq, H, dtype=torch.bfloat16, device="cuda")

    scale_log2 = float(math.log2(inputs["scale"]))

    # Strides in elements (bf16), row-major [sq, hq, H]
    stride_q_head  = H
    stride_q_token = hq * H
    stride_o_head  = H
    stride_o_token = hq * H

    # K/V cache strides: [num_blocks, bs, 1, H]
    stride_block   = bs * H
    stride_page    = H
    stride_kv_head = H

    num_blocks  = inputs["num_blocks"]
    block_table = inputs["block_table"]
    cu_seqlens_q = inputs["cu_seqlens_q"]

    hip_stream = int(torch.cuda.current_stream().cuda_stream)
    cfg = LaunchConfig(grid=grid, block=(64, 1, 1), stream=hip_stream, fence=False)

    vals = {
        "Q":                q_256,
        "K_cache":          K_paged,
        "V_cache":          V_paged,
        "O":                out_256,
        "block_table":      block_table,
        "cu_seqlens_q":     cu_seqlens_q,
        "seqlens_k":        seqlens_k,
        "scale_log2":       scale_log2,
        "total_q":          int(sq),
        "batch":            int(1),
        "stride_q_token":   stride_q_token,
        "stride_q_head":    stride_q_head,
        "stride_block":     stride_block,
        "stride_page":      stride_page,
        "stride_kv_head":   stride_kv_head,
        "stride_v_block":   stride_block,
        "stride_v_page":    stride_page,
        "stride_v_kv_head": stride_kv_head,
        "stride_o_token":   stride_o_token,
        "stride_o_head":    stride_o_head,
        "block_table_stride": int(num_blocks),
    }

    def _launch():
        lnch(vals, config=cfg)

    with no_fence():
        ms = time_launches(_launch, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)

    out_128 = out_256[:, :, :128].contiguous()
    return out_128, None, ms


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="MLA prefill live benchmark")
    ap.add_argument(
        "--shapes", nargs="+", type=Path,
        default=[Path(__file__).parent / "mla_prefill_shapes.json"],
    )
    ap.add_argument(
        "--variants", nargs="+", default=["prod"],
        choices=list(_VARIANTS),
        help="Kernel variants to benchmark (default: prod)",
    )
    ap.add_argument("--flydsl", action="store_true",
                    help="Include AITER Triton MLA as external baseline")
    ap.add_argument("--warmup",      type=int,   default=5)
    ap.add_argument("--iterations",  type=int,   default=20)
    ap.add_argument("--tol",         type=float, default=4e-2)
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--output-json", type=Path,  default=None)
    ap.add_argument("--arch",        type=str,   default=None)
    ap.add_argument("--limit",       type=int,   default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: no GPU", file=sys.stderr)
        return 1

    arch = args.arch or _detect_arch()

    # Add library and platform roots to sys.path
    here     = Path(__file__).resolve()
    lib_root = here.parents[4]     # rocke/library
    plt_root = here.parents[5] / "platform" / "python"
    for p in (str(lib_root), str(plt_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    shapes: list[dict] = []
    for p in args.shapes:
        shapes.extend(_load_mla_shapes(p))
    if args.limit:
        shapes = shapes[:args.limit]

    print(f"device:   {torch.cuda.get_device_name(0)}")
    print(f"arch:     {arch}")
    print(f"shapes:   {len(shapes)}")
    print(f"variants: {args.variants}  flydsl={args.flydsl}")

    results = []

    for i, shape in enumerate(shapes, 1):
        sq    = shape["seqlen_q"]
        sk    = shape["seqlen_k"]
        hq    = shape["num_query_heads"]
        label = shape["label"] or f"{shape['model']}_sq{sq}"
        tag   = f"[{i}/{len(shapes)}] {label}"

        try:
            inputs = _make_inputs(shape)
        except Exception as exc:
            print(f"{tag}  INPUT ERR: {exc!r}")
            continue

        # Reference output (once per shape)
        ref_out = None
        if not args.no_correctness:
            try:
                ref_out = _run_reference(inputs)
            except Exception as exc:
                print(f"{tag}  REF ERR: {exc!r}")

        # AITER Triton MLA baseline
        aiter_ms  = None
        aiter_out = None
        if args.flydsl:
            try:
                aiter_out, aiter_ms = _run_aiter_mla(
                    shape, inputs, args.warmup, args.iterations
                )
            except NotImplementedError as exc:
                print(f"{tag}  AITER SKIP: {exc}")
            except Exception as exc:
                print(f"{tag}  AITER ERR: {exc!r}")
                # Sync GPU after AITER failure to prevent cascade errors
                try:
                    import torch; torch.cuda.synchronize()
                except Exception:
                    pass

        rec = {
            "label":           label,
            "model":           shape["model"],
            "num_query_heads": hq,
            "seqlen_q":        sq,
            "seqlen_k":        sk,
            "block_size":      shape["block_size"],
            "dtype":           shape["dtype"],
            "arch":            arch,
            "aiter_ms":        aiter_ms,
            "variants":        {},
        }

        best_ms      = None
        best_variant = None

        for vname in args.variants:
            vcfg = _VARIANTS[vname]
            try:
                out, _lse, ms = _run_variant(
                    shape, inputs, vcfg, arch, args.warmup, args.iterations
                )
                err = _max_abs(out, ref_out) if ref_out is not None else None
                ok  = (err <= args.tol) if err is not None else True
                if best_ms is None or ms < best_ms:
                    best_ms      = ms
                    best_variant = vname
            except Exception as exc:
                ms  = None
                err = None
                ok  = False
                print(f"{tag}  {vname} ERR: {exc!r}")
                traceback.print_exc()
            finally:
                # Always sync GPU between variants — catches asynchronous HIP
                # errors from the previous kernel before the next variant starts.
                try:
                    import torch; torch.cuda.synchronize()
                except Exception:
                    pass

            speedup_vs_aiter = (aiter_ms / ms) if (aiter_ms and ms) else None
            rec["variants"][vname] = {
                "ms":               ms,
                "max_abs":          err,
                "ok":               ok,
                "speedup_vs_aiter": speedup_vs_aiter,
            }

        flops  = _flops(sq, sk, hq)
        tflops = (flops / best_ms / 1e9) if best_ms else None
        rec["best_variant"]   = best_variant
        rec["best_ms"]        = best_ms
        rec["tflops"]         = tflops

        # Print one-line summary
        ms_str  = f"{best_ms*1000:.1f}us"  if best_ms  else "---"
        tf_str  = f"{tflops:.2f}TFLOPS"    if tflops   else "---"
        ai_str  = f"  aiter={aiter_ms*1000:.1f}us" if aiter_ms else ""
        spd_str = ""
        if aiter_ms and best_ms:
            spd_str = f"  {aiter_ms/best_ms:.2f}x vs aiter"
        best_ok = rec["variants"].get(best_variant, {}).get("ok", False) if best_variant else False
        err_str = (f"  max_abs={rec['variants'][best_variant]['max_abs']:.4f}"
                   if best_variant and rec["variants"][best_variant]["max_abs"] is not None else "")
        status  = ("ok" if best_ok else "WRONG") if ref_out is not None else "no-ref"
        print(f"{tag}  {ms_str}  {tf_str}{ai_str}{spd_str}{err_str}  [{status}]")

        results.append(rec)

    if args.output_json:
        with args.output_json.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults written to {args.output_json}")

    n_ok = sum(
        1 for r in results
        if r.get("best_variant") and r["variants"][r["best_variant"]].get("ok", False)
    )
    print(f"\nSummary: {n_ok}/{len(results)} shapes correct")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
