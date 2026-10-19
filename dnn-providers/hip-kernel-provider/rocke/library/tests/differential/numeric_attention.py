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
    """Map a spec dtype key to the numpy host *storage* dtype.

    numpy has no native bfloat16 and this harness does not pull in ml_dtypes,
    so bf16 is carried as raw ``uint16`` and converted through
    ``rocke.numeric.bf16``, which its own tests hold bit-identical to
    ``ml_dtypes.bfloat16``.
    Use ``_decode`` / ``_encode`` rather than ``.astype`` on a bf16 array --
    a raw uint16 reinterpreted as a number is silent garbage, not an upcast.
    """
    import numpy as np

    try:
        return {"fp16": np.float16, "fp32": np.float32, "bf16": np.uint16}[dtype]
    except KeyError:
        raise NotImplementedError(
            f"dtype {dtype!r} has no torch-free numpy host encoding "
            "(supported: fp16, fp32, bf16)"
        ) from None


def _encode(a_f32, dtype: str):
    """fp32 -> host storage array for ``dtype``."""
    from rocke.numeric.bf16 import f32_to_bf16

    return f32_to_bf16(a_f32) if dtype == "bf16" else a_f32.astype(_np_dtype(dtype))


def _decode(a, dtype: str):
    """Host storage array -> fp32."""
    import numpy as np
    from rocke.numeric.bf16 import bf16_to_f32

    return bf16_to_f32(a) if dtype == "bf16" else a.astype(np.float32)


def _compare(out_f32, ref_f32, tol: Tol) -> Tuple[float, float, float]:
    """Return (max_abs_diff, max_rel_diff, worst allclose margin).

    ``out_f32`` must already be decoded to fp32 (see ``_decode``) -- bf16 is
    carried as raw uint16, which ``.astype(float32)`` would silently misread.
    """
    import numpy as np

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

# RDNA WMMA FMHA prefill (kernels/gfx1151/wmma_fmha_fwd_v2.py), which carries
# the bf16 I/O path. These are the on-GPU cases behind
# `--only wmma_fmha_fwd_v2`; without them that scope selects nothing and the
# gate passes vacuously. is_valid_spec is capability-driven: it rejects any
# target whose catalog lacks the wave32 16x16x16 WMMA atom for the dtype (CDNA
# has MFMA instead), so the cases self-skip there rather than failing.
WMMA_CONFIGS: List[AttnCfg] = [
    AttnCfg("wmma_mha_b2_h4_s64_d64_f16", 2, 4, 4, 64, 64, 64),
    AttnCfg("wmma_causal_b2_h4_s64_d64_f16", 2, 4, 4, 64, 64, 64, causal=True),
    AttnCfg("wmma_mha_b2_h4_s64_d64_bf16", 2, 4, 4, 64, 64, 64, dtype="bf16"),
    AttnCfg(
        "wmma_causal_b2_h4_s64_d64_bf16",
        2,
        4,
        4,
        64,
        64,
        64,
        dtype="bf16",
        causal=True,
    ),
    AttnCfg(
        "wmma_gqa_b1_h8kv2_s64_d128_bf16",
        1,
        8,
        2,
        64,
        64,
        128,
        dtype="bf16",
    ),
    AttnCfg(
        "wmma_gqa_causal_b1_h8kv2_s64_d128_f16",
        1,
        8,
        2,
        64,
        64,
        128,
        causal=True,
    ),
]

# Against an *fp32* reference the residual is dominated by the final rounding
# to the I/O dtype, so bf16 (8 significand bits) lands ~8x above fp16 (11).
# Worst observed on gfx1201 with the single-wave predecessor kernel over the
# same shapes: fp16 2.2e-04, bf16 1.8e-03. bf16 therefore gets 1e-2 (~5x
# headroom), while fp16 keeps its historical 2e-2 attention gate. A tolerance
# loose enough to swallow bf16 quantisation noise would also swallow a
# structural bug (TESTING.md gap G5), so bf16 is tightened, not relaxed.
_WMMA_TOL = {"fp16": Tol(rtol=0.0, atol=2e-2), "bf16": Tol(rtol=0.0, atol=1e-2)}


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
    rng = np.random.default_rng(0xA11E)
    # Route host tensors through _encode/_decode rather than .astype: for a
    # dtype with no native numpy type (bf16, carried as raw uint16) .astype
    # would convert the *values* to integers instead of reinterpreting the
    # bits, and the kernel would be gated against garbage inputs.
    Q = _encode((rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float32), tol_key)
    K = _encode((rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), tol_key)
    V = _encode((rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), tol_key)
    Out = np.zeros((B, Sq, Hq, D), dtype=Q.dtype)
    # The reference consumes exactly what the device reads, so input rounding
    # is not counted as kernel error.
    Qf, Kf, Vf = (_decode(t, tol_key) for t in (Q, K, V))

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
            Kb = np.repeat(Kf[bi], rep, axis=1)
            Vb = np.repeat(Vf[bi], rep, axis=1)
        else:
            Kb, Vb = Kf[bi], Vf[bi]
        ref[bi] = dense_attention_reference(Qf[bi], Kb, Vb, causal=cfg.causal)

    max_abs, max_rel, margin = _compare(_decode(Out, tol_key), ref, tol)
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


def run_wmma_config(cfg: AttnCfg, arch: str = "gfx1151") -> NumericResult:
    """On-GPU numeric check for the RDNA WMMA FMHA prefill (fp16 and bf16 I/O).

    Same shape as ``run_attn_config`` but drives the WMMA adapter instead of the
    MFMA one, and routes host tensors through ``_encode``/``_decode`` so bf16
    (raw uint16 on the host) is never silently reinterpreted.
    """
    import math as _m

    import numpy as np
    from kernels.gfx1151.wmma_fmha_fwd_v2 import (
        WmmaFmhaFwdV2Spec,
        build_wmma_fmha_fwd_v2,
        is_valid_spec,
        wmma_fmha_fwd_v2_grid,
        wmma_fmha_fwd_v2_signature,
    )
    from rocke.helpers.compile import compile_kernel
    from rocke.numeric.references import dense_attention_reference
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, KernelLauncher, LaunchConfig

    tol_key = _ELEM_TOL_KEY.get(cfg.dtype, cfg.dtype)
    res = NumericResult(
        family="wmma_fmha_fwd_v2",
        name=cfg.name,
        status="GREEN",
        dtype=tol_key,
        shape=(cfg.batch, cfg.seqlen_q, cfg.heads, cfg.head_size),
    )
    tol = _WMMA_TOL[tol_key]
    res.rtol, res.atol = tol.rtol, tol.atol

    # Only the head-width bucket, dtype and mask class are compiled into the
    # code object; the head counts and widths, strides and sequence lengths
    # below are kernel arguments.
    spec = WmmaFmhaFwdV2Spec(
        dtype=tol_key,
        head_dim=cfg.head_size,
        mask="causal" if cfg.causal else "none",
        name=f"rocke_wmma_num_{cfg.name}",
    )

    try:
        ok, why = is_valid_spec(spec, arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "REJECTED"
        res.detail = f"validate raised: {e}"
        return res
    if not ok:
        # Expected on a target with no wave32 WMMA atom for this dtype (CDNA).
        res.status = "REJECTED"
        res.detail = f"is_valid_spec: {why}"
        return res

    try:
        art = compile_kernel(build_wmma_fmha_fwd_v2(spec, arch=arch), arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"build/compile raised: {e}"
        return res
    res.extra["kernel_name"] = art.kernel_name
    res.extra["hsaco_bytes"] = art.hsaco_bytes

    B, Hq, Hk, D = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_size
    Sq, Sk = cfg.seqlen_q, cfg.seqlen_k
    rng = np.random.default_rng(0xA11E)
    Q = _encode((rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float32), tol_key)
    K = _encode((rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), tol_key)
    V = _encode((rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), tol_key)
    Out = np.zeros((B, Sq, Hq, D), dtype=Q.dtype)
    # The reference consumes exactly what the device reads, so input rounding
    # is not counted as kernel error.
    Qf, Kf, Vf = (_decode(t, tol_key) for t in (Q, K, V))

    scale_log2 = float(1.0 / _m.sqrt(D) * _m.log2(_m.e))
    sig = wmma_fmha_fwd_v2_signature(spec)
    grid, block = wmma_fmha_fwd_v2_grid(spec, batch=B, seqlen_q=Sq, num_query_heads=Hq)

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
        # [B, S, H, D] row-major; strides are in elements. LSE is not written
        # (write_lse=0) but the pointer must still be a valid one. Causal is
        # top-left (window_right=0); no mask leaves both window sides open.
        values = {
            "Q": q_dev,
            "K": k_dev,
            "V": v_dev,
            "O": o_dev,
            "LSE": o_dev,
            "scale_log2": scale_log2,
            "batch": B,
            "seqlen_q": Sq,
            "seqlen_k": Sk,
            "num_query_heads": Hq,
            "num_kv_heads": Hk,
            "num_v_heads": Hk,
            "head_dim_q": D,
            "head_dim_v": D,
            "bottom_right": 0,
            "window_left": -1,
            "window_right": 0 if cfg.causal else -1,
            "write_lse": 0,
            "stride_q_batch": Sq * Hq * D,
            "stride_q_token": Hq * D,
            "stride_q_head": D,
            "stride_k_batch": Sk * Hk * D,
            "stride_k_token": Hk * D,
            "stride_k_head": D,
            "stride_v_batch": Sk * Hk * D,
            "stride_v_token": Hk * D,
            "stride_v_head": D,
            "stride_o_batch": Sq * Hq * D,
            "stride_o_token": Hq * D,
            "stride_o_head": D,
            "stride_lse_batch": 0,
            "stride_lse_token": 0,
            "stride_lse_head": 0,
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

    ref = np.empty((B, Sq, Hq, D), dtype=np.float32)
    for bi in range(B):
        if Hk != Hq:
            rep = Hq // Hk
            Kb = np.repeat(Kf[bi], rep, axis=1)
            Vb = np.repeat(Vf[bi], rep, axis=1)
        else:
            Kb, Vb = Kf[bi], Vf[bi]
        ref[bi] = dense_attention_reference(Qf[bi], Kb, Vb, causal=cfg.causal)

    max_abs, max_rel, margin = _compare(_decode(Out, tol_key), ref, tol)
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


# The families this harness owns. run_checks.py forwards its --op to --only for
# every operator in the repo, most of which live in other harnesses, so "this
# scope selected nothing" has two very different meanings -- see _scope_is_ours.
_OWNED_FAMILIES = ("attention", "wmma_fmha_fwd_v2")


def _scope_is_ours(only: str) -> bool:
    """Could this ``--only`` scope ever select a case here?

    Mirrors ``want()``'s substring matching exactly: a scope is ours if any of
    its terms could match one of our family names or case names. Anything else
    names an operator owned by a different harness (``fmha_bwd``, say), and the
    right answer for that is to stand aside, not to fail.
    """
    subs = [s for s in only.split(",") if s]
    if not subs:
        return True
    names = [c.name for c in (*ATTN_CONFIGS, *WMMA_CONFIGS)]
    return any(
        any(s in fam for fam in _OWNED_FAMILIES) or any(s in n for n in names)
        for s in subs
    )


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
    for cfg in WMMA_CONFIGS:
        if want("wmma_fmha_fwd_v2", cfg.name):
            results.append(run_wmma_config(cfg, arch=arch))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Default to the arch of the visible device rather than a fixed string: the
    # kernels are built for --arch and launched on whatever is present, so a
    # mismatch is a guaranteed LAUNCH_FAIL (and run_checks.py does not pass
    # --arch). "" means "follow the device".
    ap.add_argument("--arch", default="")
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
    arch = args.arch or dev_arch
    print(f"L6 NUMERIC ATTN  arch={arch}  device={dev_arch} ({dev_name})")
    results = run_all(arch=arch, only=args.only)

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

    # A gate that executed nothing has proved nothing. Without this, an --only
    # scope that matches no case (or one whose every case is REJECTED on this
    # arch) exits 0 and reads as a green numeric stage -- the failure mode is
    # indistinguishable from "verified" in CI, which is the worst kind.
    #
    # But only for scopes this harness owns. run_checks.py forwards --op to
    # --only for every operator in the repo, so a scope naming someone else's
    # family (fmha_bwd, its own documented example) is not a failed check -- it
    # is a check that belongs elsewhere. Failing there would make an unrelated
    # operator's numeric stage permanently red.
    if npass + nfail == 0 and args.only and not _scope_is_ours(args.only):
        print(
            f"no case for --only {args.only!r} in this harness "
            f"(owns: {', '.join(_OWNED_FAMILIES)}); nothing to do."
        )
        return 0

    if npass + nfail == 0:
        scope = f"--only {args.only!r}" if args.only else "the full set"
        names = sorted(c.name for c in (*ATTN_CONFIGS, *WMMA_CONFIGS))
        sys.stderr.write(
            f"no numeric case executed for {scope} on arch={arch}: "
            f"{len(results)} selected, {nrej} rejected, {nskip} skipped.\n"
            f"known cases: {', '.join(names)}\n"
        )
        return 2

    return 1 if (nfail or nerr) else 0


if __name__ == "__main__":
    raise SystemExit(main())
