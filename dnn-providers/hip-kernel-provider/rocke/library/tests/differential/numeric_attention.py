#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# numeric_attention.py -- Attention (FMHA) lane of the L6 numeric harness.
#
# Extracted from platform/tests/instances/differential/numeric.py so that
# library-layer tests can run the attention numeric lane without importing
# platform-internal attention kernel builders from platform code.
#
# Run:
#   python library/tests/differential/numeric_attention.py [--arch gfx950]
#
# Needs GPU access (torch.cuda). Build/compile (comgr) does not need GPU.

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TMP = Path(tempfile.gettempdir()) / "rocke_numeric_attn"
TMP.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Shared infra (duplicated from numeric.py; keep in sync)
# NumericResult and Tol are small value types — duplication is intentional
# since platform/tests/instances/differential/numeric.py is a standalone
# script, not an importable package.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


@dataclass
class NumericResult:
    family: str
    name: str
    status: str  # GREEN | DRIFT | REJECTED | BUILD_FAIL | LAUNCH_FAIL
    dtype: str = ""
    shape: Tuple[int, ...] = ()
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    rtol: float = 0.0
    atol: float = 0.0
    margin: float = 0.0  # worst (|d| - (atol+rtol|ref|)); <=0 => pass
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def passed(self) -> bool:
        return self.status == "GREEN"


def _torch_dtype(torch, dtype: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def _compare(torch, out, ref_f32, tol: Tol) -> Tuple[float, float, float]:
    """Return (max_abs_diff, max_rel_diff, worst allclose margin)."""
    out_f32 = out.to(torch.float32)
    diff = (out_f32 - ref_f32).abs()
    max_abs = float(diff.max().item())
    denom = ref_f32.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    allowed = tol.atol + tol.rtol * ref_f32.abs()
    margin = float((diff - allowed).max().item())
    return max_abs, max_rel, margin


# map spec dtype -> TOL-table key (mirrors _ELEM_TOL_KEY in numeric.py)
_ELEM_TOL_KEY = {"f16": "fp16", "bf16": "bf16"}


# ---------------------------------------------------------------------
# Attention lane (FMHA forward, unified tiled MFMA body)
# ---------------------------------------------------------------------
# Builds kernels.common.fmha_mfma.build_fmha_fwd_mfma through the
# *comgr* (LLVM-IR) path -- the same Python engine the rest of the L6
# numeric harness uses -- and compares against a dense fp32
# softmax-attention reference. The ABI mirrors
# examples/common/fmha_fwd_verify_hip.py:
#
#   args: (Q, K, V, Out : ptr,  scale_log2 : f32,  Sq, Sk : i32,
#          stride_q_token, stride_q_head, stride_k_token, stride_k_head,
#          stride_v_token, stride_v_head, stride_o_token, stride_o_head : i32)
#   layout: Q (B,Sq,Hq,D) / K,V (B,Sk,Hk,D) / Out (B,Sq,Hq,D) row-major;
#           the batch axis is folded in by the grid z dim (block_id_z).
#   grid : fmha_fwd_mfma_grid(spec, batch=B);  block = (wave_size,1,1).
#
# scale_log2 = (1/sqrt(D)) * log2(e): the kernel does the softmax in
# base-2 (exp2), so the host pre-scales the QK scale into log2 space.
@dataclass(frozen=True)
class AttnCfg:
    name: str
    batch: int
    heads: int
    kv_heads: int  # == heads -> MHA
    seqlen_q: int
    seqlen_k: int
    head_size: int
    dtype: str = "f16"
    causal: bool = False


ATTN_CONFIGS: List[AttnCfg] = [
    AttnCfg("fmha_mha_b2_h4_s64_d64", 2, 4, 4, 64, 64, 64),
    AttnCfg("fmha_mha_b1_h8_s128_d64", 1, 8, 8, 128, 128, 64),
    AttnCfg("fmha_causal_b2_h4_s64_d64", 2, 4, 4, 64, 64, 64, causal=True),
    AttnCfg("fmha_gqa_b1_h8kv2_s64_d64", 1, 8, 2, 64, 64, 64),
]

# softmax-attention carries an exp + a length-Sk normalization on top of two
# matmuls; the accumulation order differs from the dense reference, so use the
# attention parity gate's tolerance (2e-2), matching the example harness.
_ATTN_TOL = Tol(rtol=0.0, atol=2e-2)


def _ref_attention_torch(torch, Q, K, V, *, causal: bool):
    """Dense attention ref (fp32 math), Q/K/V shape (Sq|Sk, H, D)."""
    import math as _m

    d = Q.shape[-1]
    q = Q.to(torch.float32)
    k = K.to(torch.float32)
    v = V.to(torch.float32)
    # scores[i,h,j] = sum_d q[i,h,d]*k[j,h,d]
    scores = torch.einsum("ihd,jhd->ihj", q, k) / _m.sqrt(d)
    if causal:
        sq, sk = Q.shape[0], K.shape[0]
        qpos = torch.arange(sq, device=q.device)[:, None, None]
        kpos = torch.arange(sk, device=q.device)[None, None, :]
        scores = torch.where(kpos <= qpos, scores, torch.full_like(scores, -1e30))
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(scores)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.einsum("ihj,jhd->ihd", probs, v)


def run_attn_config(cfg: AttnCfg, arch: str = "gfx950") -> NumericResult:
    import math as _m

    import torch

    from rocke.core.arch import ArchTarget
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from kernels import FmhaCommonSpec, FmhaShape
    from kernels.common.fmha_mfma import (
        FmhaMfmaSpec,
        build_fmha_fwd_mfma,
        fmha_fwd_mfma_grid,
        is_valid_spec,
    )
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    tol_key = _ELEM_TOL_KEY.get(cfg.dtype, cfg.dtype)
    res = NumericResult(
        family="attention",
        name=cfg.name,
        status="GREEN",
        dtype=tol_key,
        shape=(cfg.batch, cfg.seqlen_q, cfg.heads, cfg.head_size),
    )
    tol = _ATTN_TOL
    res.rtol, res.atol = tol.rtol, tol.atol

    common = FmhaCommonSpec(
        FmhaShape(
            head_size=cfg.head_size,
            num_query_heads=cfg.heads,
            num_kv_heads=cfg.kv_heads,
            block_size_q=16,
            block_size_k=64,
        ),
        dtype=cfg.dtype,
        mask_mode="causal" if cfg.causal else "none",
    )
    spec = FmhaMfmaSpec(
        common=common,
        seqlen_q=cfg.seqlen_q,
        seqlen_k=cfg.seqlen_k,
        name=f"rocke_fmha_num_{cfg.name}",
    )

    target = ArchTarget.from_gfx(arch)
    try:
        ok, why = is_valid_spec(spec, arch)
    except Exception as e:  # noqa: BLE001
        res.status = "REJECTED"
        res.detail = f"validate raised: {e}"
        return res
    if not ok:
        res.status = "REJECTED"
        res.detail = f"is_valid_spec: {why}"
        return res

    try:
        kern = build_fmha_fwd_mfma(spec, arch=arch)
        art = compile_kernel(kern, arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"build/compile raised: {e}"
        return res
    res.extra["kernel_name"] = art.kernel_name
    res.extra["hsaco_bytes"] = art.hsaco_bytes

    B, Hq, Hk, D = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_size
    Sq, Sk = cfg.seqlen_q, cfg.seqlen_k
    td = _torch_dtype(torch, tol_key)
    torch.manual_seed(0xA11E)
    Q = (torch.randn((B, Sq, Hq, D), device="cuda", dtype=torch.float32) * 0.3).to(td)
    K = (torch.randn((B, Sk, Hk, D), device="cuda", dtype=torch.float32) * 0.3).to(td)
    V = (torch.randn((B, Sk, Hk, D), device="cuda", dtype=torch.float32) * 0.3).to(td)
    Out = torch.zeros((B, Sq, Hq, D), device="cuda", dtype=td)

    scale_log2 = float(1.0 / _m.sqrt(D) * _m.log2(_m.e))
    sig = (
        SignatureBuilder()
        .ptr("Q", cfg.dtype)
        .ptr("K", cfg.dtype)
        .ptr("V", cfg.dtype)
        .ptr("Out", cfg.dtype)
        .scalar("scale", "f32")
        .scalar("Sq", "i32")
        .scalar("Sk", "i32")
        .scalar("sqt", "i32")
        .scalar("sqh", "i32")
        .scalar("skt", "i32")
        .scalar("skh", "i32")
        .scalar("svt", "i32")
        .scalar("svh", "i32")
        .scalar("sot", "i32")
        .scalar("soh", "i32")
        .build()
    )
    values = {
        "Out": Out,
        "Q": Q,
        "K": K,
        "V": V,
        "scale": scale_log2,
        "Sq": Sq,
        "Sk": Sk,
        "sqt": Hq * D,
        "sqh": D,
        "skt": Hk * D,
        "skh": D,
        "svt": Hk * D,
        "svh": D,
        "sot": Hq * D,
        "soh": D,
    }
    grid = fmha_fwd_mfma_grid(spec, batch=B)
    block = (target.wave_size, 1, 1)

    try:
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        res.status = "LAUNCH_FAIL"
        res.detail = f"launch raised: {e}"
        return res

    # Reference per batch (expand KV heads for GQA).
    ref = torch.empty_like(Out, dtype=torch.float32)
    for bi in range(B):
        if Hk != Hq:
            rep = Hq // Hk
            Kb = K[bi].repeat_interleave(rep, dim=1)
            Vb = V[bi].repeat_interleave(rep, dim=1)
        else:
            Kb, Vb = K[bi], V[bi]
        ref[bi] = _ref_attention_torch(torch, Q[bi], Kb, Vb, causal=cfg.causal)

    max_abs, max_rel, margin = _compare(torch, Out, ref, tol)
    res.max_abs_diff = max_abs
    res.max_rel_diff = max_rel
    res.margin = margin
    res.status = "GREEN" if margin <= 0.0 and math.isfinite(margin) else "DRIFT"
    res.detail = (
        f"causal={cfg.causal} grid={grid} block={block} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"atol={tol.atol:.0e} margin={margin:.3e}"
    )
    return res


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def _check_gpu() -> Optional[str]:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"torch import failed: {e}"
    if not torch.cuda.is_available():
        return (
            "torch.cuda.is_available() is False -- run under sudo -E "
            "(the login user is not in the GPU device group)"
        )
    return None


def run_all(arch: str = "gfx950", only: str = "") -> List[NumericResult]:
    results: List[NumericResult] = []
    subs = [s for s in only.split(",") if s]

    def want(family: str, name: str) -> bool:
        if not subs:
            return True
        return any(s in family or s in name for s in subs)

    for cfg in ATTN_CONFIGS:
        if want("attention", cfg.name):
            results.append(run_attn_config(cfg, arch=arch))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--only", default="", help="comma-separated family/name substrings")
    ap.add_argument("--json", default=str(TMP / "numeric_attn_dashboard.json"))
    args = ap.parse_args(argv)

    gpu_err = _check_gpu()
    if gpu_err is not None:
        sys.stderr.write(f"GPU unavailable: {gpu_err}\n")
        return 3

    import torch

    print(f"L6 NUMERIC ATTN  arch={args.arch}  device={torch.cuda.get_device_name(0)}")
    results = run_all(arch=args.arch, only=args.only)

    rows: List[Dict[str, Any]] = []
    npass = nfail = nrej = nskip = nerr = 0
    for r in results:
        rows.append(
            {
                "family": r.family,
                "name": r.name,
                "status": r.status,
                "dtype": r.dtype,
                "shape": list(r.shape),
                "max_abs_diff": r.max_abs_diff,
                "max_rel_diff": r.max_rel_diff,
                "rtol": r.rtol,
                "atol": r.atol,
                "margin": r.margin,
                "detail": r.detail,
                "extra": r.extra,
            }
        )
        if r.status == "GREEN":
            npass += 1
        elif r.status == "DRIFT":
            nfail += 1
        elif r.status == "REJECTED":
            nrej += 1
        elif r.status == "SKIPPED":
            nskip += 1
        else:
            nerr += 1
        tag = r.status
        line = f"  {tag:11s} {r.family}/{r.name}"
        if r.status in ("GREEN", "DRIFT"):
            line += (
                f"  {r.dtype} {tuple(r.shape)}  "
                f"max_abs={r.max_abs_diff:.3e} max_rel={r.max_rel_diff:.3e} "
                f"margin={r.margin:.3e} (rtol={r.rtol:.0e} atol={r.atol:.0e})"
            )
        elif r.detail:
            line += f"  {r.detail}"
        print(line)

    Path(args.json).write_text(json.dumps(rows, indent=2))
    print(
        f"\n=== L6 NUMERIC ATTN SUMMARY ===\n"
        f"  PASS={npass}  FAIL={nfail}  REJECTED={nrej}  "
        f"SKIPPED={nskip}  ERROR={nerr}"
    )
    return 1 if (nfail or nerr) else 0


if __name__ == "__main__":
    raise SystemExit(main())
