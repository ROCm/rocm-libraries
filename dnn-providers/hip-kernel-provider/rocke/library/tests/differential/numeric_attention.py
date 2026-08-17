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
# Needs GPU access (HIP). Build/compile (comgr) does not need GPU.

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


def _np_dtype(dtype: str):
    """Map a spec dtype key to a numpy host dtype.

    Only fp16/fp32 are supported torch-free: numpy has no native bfloat16 and
    this harness does not pull in ml_dtypes. Every current ATTN_CONFIG is f16,
    so bf16 is a clean, explicit error rather than dead scaffolding -- add a
    real bf16 host encoding here if a bf16 config is ever introduced.
    """
    import numpy as np

    try:
        return {"fp16": np.float16, "fp32": np.float32}[dtype]
    except KeyError:
        raise NotImplementedError(
            f"dtype {dtype!r} has no torch-free numpy host encoding "
            "(numpy lacks bfloat16 and this harness avoids ml_dtypes); "
            "all current ATTN_CONFIGS are f16"
        ) from None


def _compare(out, ref_f32, tol: Tol) -> Tuple[float, float, float]:
    """Return (max_abs_diff, max_rel_diff, worst allclose margin)."""
    import numpy as np

    out_f32 = out.astype(np.float32)
    diff = np.abs(out_f32 - ref_f32)
    max_abs = float(diff.max())
    denom = np.clip(np.abs(ref_f32), 1e-12, None)
    max_rel = float((diff / denom).max())
    allowed = tol.atol + tol.rtol * np.abs(ref_f32)
    margin = float((diff - allowed).max())
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


def run_attn_config(cfg: AttnCfg, arch: str = "gfx950") -> NumericResult:
    import math as _m

    import numpy as np

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
    from rocke.numeric.references import dense_attention_reference
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem, KernelLauncher, LaunchConfig

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
    np_dtype = _np_dtype(tol_key)
    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np_dtype)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np_dtype)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np_dtype)
    Out = np.zeros((B, Sq, Hq, D), dtype=np_dtype)

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
    grid = fmha_fwd_mfma_grid(spec, batch=B)
    block = (target.wave_size, 1, 1)

    # Torch-free device I/O: DeviceMem RAII buffers (freed when they leave
    # scope), numpy host arrays uploaded as raw bytes, and the DeviceMem
    # objects passed straight into the launcher values (pack_args reads their
    # .ptr()). fence=True makes the launch stream-synchronize before returning,
    # so the d2h readback observes a finished kernel.
    rt = Runtime()
    try:
        q_dev = DeviceMem(Q.nbytes)
        k_dev = DeviceMem(K.nbytes)
        v_dev = DeviceMem(V.nbytes)
        o_dev = DeviceMem(Out.nbytes)
        rt.memcpy_h2d(q_dev.ptr(), as_u8_buffer(Q), Q.nbytes)
        rt.memcpy_h2d(k_dev.ptr(), as_u8_buffer(K), K.nbytes)
        rt.memcpy_h2d(v_dev.ptr(), as_u8_buffer(V), V.nbytes)
        rt.memset(o_dev.ptr(), 0, Out.nbytes)
        values = {
            "Out": o_dev,
            "Q": q_dev,
            "K": k_dev,
            "V": v_dev,
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
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        rt.memcpy_d2h(as_u8_buffer(Out), o_dev.ptr(), Out.nbytes)
    except Exception as e:  # noqa: BLE001
        res.status = "LAUNCH_FAIL"
        res.detail = f"device I/O or launch raised: {e}"
        return res

    # Reference per batch (expand KV heads for GQA). out_dtype=None keeps the
    # reference in fp32 -- the full-precision compare this harness has always
    # used (do not truncate the reference to the kernel dtype).
    ref = np.empty((B, Sq, Hq, D), dtype=np.float32)
    for bi in range(B):
        if Hk != Hq:
            rep = Hq // Hk
            Kb = np.repeat(K[bi], rep, axis=1)
            Vb = np.repeat(V[bi], rep, axis=1)
        else:
            Kb, Vb = K[bi], V[bi]
        ref[bi] = dense_attention_reference(Q[bi], Kb, Vb, causal=cfg.causal)

    max_abs, max_rel, margin = _compare(Out, ref, tol)
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
    """Gate this harness on a visible HIP device via the torch-free probe.

    The harness is torch-free: data generation is numpy and device I/O goes
    through DeviceMem + the HIP runtime, so it gates on the resource it actually
    uses -- ``rocke.runtime.hip_module.get_device_arch()``. This is the same
    probe the test *gates* use. (It inverts the old torch-era rationale: when
    the driver drove ``torch.cuda`` in-process, probing HIP first would dlopen
    the system HIP before torch bound its own; with torch gone there is no such
    ordering hazard and the HIP probe is the correct, direct check.)
    """
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch(0) is None:
        return (
            "no HIP device visible (get_device_arch() is None) -- on a GPU box, "
            "ensure the user is in the render/video groups with access to "
            "/dev/kfd and /dev/dri (e.g. run under sudo -E)"
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

    from rocke.runtime.hip_module import get_device_arch, get_device_name

    dev_arch = get_device_arch(0)
    dev_name = get_device_name(0) or "?"
    print(f"L6 NUMERIC ATTN  arch={args.arch}  device={dev_arch} ({dev_name})")
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
