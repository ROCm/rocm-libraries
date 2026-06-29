#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# numeric.py -- L6 NUMERIC lane of the rocke differential harness.
#
# Where L1/L2/L3 (run_diff.py) prove the C engine and the Python engine emit
# byte/structurally-identical IR, L6 proves the *Python-engine* kernel produces
# numerically-correct output on real hardware: build -> compile (comgr) ->
# launch on the gfx950 GPU with torch input tensors -> compare against a torch
# reference within a per-dtype tolerance.
#
# Scope (this lane): "Python-engine kernel vs torch reference". Cross-backend
# C-vs-Python bit-identity is a separate (later) step and is NOT done here.
#
#   canonical build+run path (all from rocke, no hand-rolled hipModuleLaunch):
#     spec  = UniversalGemmSpec(...)              instances/common/gemm_universal.py
#     kern  = build_universal_gemm(spec, arch)    instances/common/gemm_universal.py
#     art   = compile_kernel(kern, arch=...)      helpers/compile.py  -> .hsaco
#     sig   = gemm_args_signature()               helpers/manifest.py
#     L     = KernelLauncher(hsaco, name, sig)    runtime/launcher.py
#     L(values, config=LaunchConfig(grid,block))  runtime/launcher.py (fenced)
#
#   GEMM ABI (authoritative source: instances/common/manifest_runner/gemm.py):
#     args  : (A_ptr, B_ptr, C_ptr, M:i32, N:i32, K:i32)
#     layout: RCR -- A is (M,K) row-major, B is (N,K) row-major (== B^T col-major),
#             C is (M,N) row-major.  reference = A @ B.T
#     grid  : grid_order "NM" (universal-GEMM default) ->
#             gx = ceil(N/block_n), gy = ceil(M/block_m), gz = 1
#     block : (block_size, 1, 1)
#
# NOTE: launching a kernel needs GPU access. If plain python cannot see the
# GPU, run under passwordless sudo with -E so PYTHONPATH survives, e.g.
#     sudo -n -E "$(command -v python)" \
#         <rocKE>/tests/instances/differential/numeric.py
# (build/compile via comgr does NOT need the GPU; only the launch does.)
#
# Build artifacts / dashboards go to /tmp.  No git operations.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # rocKE root (differential -> instances -> tests -> rocKE)
PYROOT = ROCKE / "Python"  # holds rocke
if str(PYROOT) not in sys.path:
    sys.path.insert(0, str(PYROOT))

TMP = Path(tempfile.gettempdir()) / "rocke_numeric"
TMP.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Per-dtype tolerance table
# ---------------------------------------------------------------------
#
# rtol/atol are applied as: pass iff max over elems of
#   |out - ref| <= atol + rtol * |ref|
# i.e. the numpy/torch allclose convention, reported as a single worst-case
# margin so a config that "just" passes is visible.
#
# The inputs are drawn from a small symmetric range (see _make_inputs) so the
# accumulation magnitude is bounded; these tolerances are calibrated against
# that. fp16/bf16 MFMA accumulates in fp32 (dtype_acc="fp32") then rounds the
# store to the I/O dtype, so the dominant error is a single output-rounding
# step plus input quantization, not catastrophic accumulation.
@dataclass(frozen=True)
class Tol:
    rtol: float
    atol: float


TOL: Dict[str, Tol] = {
    "fp32": Tol(rtol=1e-5, atol=1e-6),
    "fp16": Tol(rtol=1e-2, atol=1e-2),
    "bf16": Tol(rtol=1.5e-2, atol=1e-2),
}


# ---------------------------------------------------------------------
# GEMM config table
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GemmCfg:
    name: str
    m: int
    n: int
    k: int
    dtype: str  # "fp16" | "bf16"
    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 16
    pipeline: str = "compv3"
    epilogue: str = "default"
    block_size: int = 0  # 0 -> derived warp_m*warp_n*wave_size
    batch: int = 0  # 0 -> non-batched; >0 -> batched GEMM (block_id_z = batch)
    # Padding (partial-tile) trait flags. Must be set whenever the matching
    # dim (M/N/K) is NOT a multiple of the tile, else the kernel reads/writes
    # the partial tile out of bounds (the masked load/store path is gated on
    # these flags -- see gemm_universal _store_masked / pad_n/pad_m guards).
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    # Documented known-unimplemented path: when set, a numeric DRIFT is
    # reported as XFAIL (expected) instead of a spurious red.
    xfail: str = ""


# The first three reuse the *valid, non-batched* shapes sampled by
# tests/parity/gemm_emit.py (configs 0, 3, 4 there); the rest add dtype
# (bf16) and shape coverage on the supported 16x16x16 MFMA atom. Shapes are
# kept multiples of the tile so no padding path is exercised by this lane
# (padding correctness is a separate axis).
GEMM_CONFIGS: List[GemmCfg] = [
    # gemm_emit cfg0: 128x128x32, compv3/default, fp16
    GemmCfg("emit0_fp16_128t", 512, 512, 256, "fp16", 128, 128, 32, 2, 2),
    # gemm_emit cfg3: 128x256x64, mem/default, fp16
    GemmCfg(
        "emit3_fp16_128x256", 512, 512, 256, "fp16", 128, 256, 64, 4, 1, pipeline="mem"
    ),
    # gemm_emit cfg4: 64x64x64, compv3/cshuffle, fp16
    GemmCfg(
        "emit4_fp16_64t", 256, 256, 128, "fp16", 64, 64, 64, 1, 1, epilogue="cshuffle"
    ),
    # bf16 variants on the supported atom
    GemmCfg("bf16_128t", 512, 512, 256, "bf16", 128, 128, 32, 2, 2),
    GemmCfg(
        "bf16_cshuffle_128t",
        384,
        256,
        192,
        "bf16",
        128,
        128,
        32,
        2,
        2,
        epilogue="cshuffle",
    ),
    # non-square / odd-ish (still tile-aligned) shapes, fp16
    GemmCfg("fp16_skinny_n", 1024, 128, 512, "fp16", 128, 128, 32, 2, 2),
    GemmCfg("fp16_tall_m", 2048, 256, 256, "fp16", 128, 128, 32, 2, 2),
    # --- batched GEMM (block_id_z is the batch index, +stride_a/b/c args) ---
    # gemm_emit "cfg5"-style batched shape: small per-batch GEMM, batch>1 so
    # the z grid axis + per-batch pointer-offset arithmetic is exercised.
    GemmCfg(
        "batched_fp16_b4",
        256,
        256,
        128,
        "fp16",
        64,
        64,
        64,
        1,
        1,
        epilogue="cshuffle",
        batch=4,
    ),
    GemmCfg(
        "batched_bf16_b8",
        128,
        128,
        64,
        "bf16",
        64,
        64,
        64,
        1,
        1,
        epilogue="cshuffle",
        batch=8,
    ),
    # --- padded / non-tile-aligned shapes (exercise the GEMM pad path) ---
    # M, N, K are deliberately NOT multiples of the tile so the partial-tile
    # masked load/store path runs; reference is the same A @ B.T.
    GemmCfg("pad_fp16_m_ragged", 500, 512, 256, "fp16", 128, 128, 32, 2, 2, pad_m=True),
    GemmCfg("pad_fp16_n_ragged", 512, 300, 256, "fp16", 128, 128, 32, 2, 2, pad_n=True),
    GemmCfg(
        "pad_fp16_k_ragged",
        512,
        512,
        200,
        "fp16",
        128,
        128,
        32,
        2,
        2,
        pad_k=True,
    ),
    GemmCfg(
        "pad_fp16_all_ragged",
        260,
        130,
        144,
        "fp16",
        64,
        64,
        64,
        1,
        1,
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
        pad_k=True,
    ),
    GemmCfg(
        "pad_bf16_all_ragged",
        300,
        200,
        176,
        "bf16",
        128,
        128,
        32,
        2,
        2,
        pad_m=True,
        pad_n=True,
        pad_k=True,
    ),
    # K-aligned partial M & N (K multiple of block_k): proves the
    # pad_m+pad_n combo independently of the K-tail path.
    GemmCfg(
        "pad_fp16_mn_ragged",
        260,
        136,
        128,
        "fp16",
        64,
        64,
        64,
        1,
        1,
        epilogue="cshuffle",
        pad_m=True,
        pad_n=True,
    ),
    # Same partial M & N but on the DIRECT (default) epilogue, whose
    # _store_masked is element-granular (no vec-alignment precondition):
    # N=130 is allowed here.
    GemmCfg(
        "pad_fp16_mn_ragged_direct",
        260,
        130,
        128,
        "fp16",
        128,
        128,
        32,
        2,
        2,
        pad_m=True,
        pad_n=True,
    ),
]


# ---------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# torch helpers
# ---------------------------------------------------------------------
def _torch_dtype(torch, dtype: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def _make_inputs(torch, m: int, n: int, k: int, dtype: str, seed: int = 0xC0FFEE):
    """Deterministic RCR inputs: A (M,K) row-major, B (N,K) row-major.

    Values are drawn from a small symmetric integer-ish range and scaled so
    fp16/bf16 round-trips are well-behaved but the test still exercises real
    (non-degenerate) accumulation. Fixed seed makes the run repeatable.
    """
    torch.manual_seed(seed)
    td = _torch_dtype(torch, dtype)
    dev = "cuda"
    # small magnitudes keep the reference within fp16/bf16 representable range
    A = (
        torch.randint(-4, 5, (m, k), device=dev, dtype=torch.int32).to(torch.float32)
        * 0.5
    ).to(td)
    B = (
        torch.randint(-4, 5, (n, k), device=dev, dtype=torch.int32).to(torch.float32)
        * 0.5
    ).to(td)
    return A, B


def _compare(torch, out, ref_f32, tol: Tol) -> Tuple[float, float, float]:
    """Return (max_abs_diff, max_rel_diff, worst allclose margin)."""
    out_f32 = out.to(torch.float32)
    diff = (out_f32 - ref_f32).abs()
    max_abs = float(diff.max().item())
    denom = ref_f32.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    # allclose margin: |d| - (atol + rtol*|ref|), worst (max) over elems
    allowed = tol.atol + tol.rtol * ref_f32.abs()
    margin = float((diff - allowed).max().item())
    return max_abs, max_rel, margin


# ---------------------------------------------------------------------
# GEMM lane
# ---------------------------------------------------------------------
def _build_gemm_spec(cfg: GemmCfg):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name=cfg.name,
        tile=TileSpec(
            tile_m=cfg.tile_m,
            tile_n=cfg.tile_n,
            tile_k=cfg.tile_k,
            warp_m=cfg.warp_m,
            warp_n=cfg.warp_n,
            warp_k=1,
            warp_tile_m=cfg.warp_tile_m,
            warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
        ),
        trait=TraitSpec(
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            pad_m=cfg.pad_m,
            pad_n=cfg.pad_n,
            pad_k=cfg.pad_k,
        ),
        data=DataSpec(
            dtype_a=cfg.dtype,
            dtype_b=cfg.dtype,
            dtype_c=cfg.dtype,
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=64,
        block_size=cfg.block_size,
        batched=cfg.batch > 0,
    )


def run_gemm_config(cfg: GemmCfg, arch: str = "gfx950") -> NumericResult:
    import torch

    from rocke.core.arch import ArchTarget
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.manifest import gemm_args_signature
    from rocke.instances import GemmPipelinePolicy
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    res = NumericResult(
        family="gemm",
        name=cfg.name,
        status="GREEN",
        dtype=cfg.dtype,
        shape=(cfg.m, cfg.n, cfg.k),
    )
    tol = TOL[cfg.dtype]
    res.rtol, res.atol = tol.rtol, tol.atol

    spec = _build_gemm_spec(cfg)

    # Validate first so an unsupported (spec,arch) is reported as a faithful
    # REJECTED rather than a crash -- parallels run_diff's BOTH_REJECTED.
    target = ArchTarget.from_gfx(arch)
    try:
        vr = GemmPipelinePolicy().validate(target, spec)
    except Exception as e:  # noqa: BLE001
        res.status = "REJECTED"
        res.detail = f"validate raised: {e}"
        return res
    if not vr.ok:
        res.status = "REJECTED"
        res.detail = f"validate: {vr.reason}"
        return res

    # Build + compile (comgr) -- no GPU needed for this part.
    try:
        kern = build_universal_gemm(spec, arch=arch)
        art = compile_kernel(kern, arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"build/compile raised: {e}"
        return res
    res.extra["kernel_name"] = art.kernel_name
    res.extra["hsaco_bytes"] = art.hsaco_bytes

    # Launch grid: universal-GEMM "NM" grid order -> gx=ceil(N/bn),
    # gy=ceil(M/bm). For ragged (non-tile-aligned) shapes the ceil-div
    # spawns the extra partial-tile CTA, exercising the masked load/store
    # (pad) path; the reference is the exact A @ B.T either way.
    gx = (cfg.n + cfg.tile_n - 1) // cfg.tile_n
    gy = (cfg.m + cfg.tile_m - 1) // cfg.tile_m
    block = (int(spec.block_size), 1, 1)
    td = _torch_dtype(torch, cfg.dtype)

    if cfg.batch > 0:
        # Batched: A (Z,M,K), B (Z,N,K), C (Z,M,N) contiguous; per-batch
        # element strides = M*K / N*K / M*N. block_id_z indexes the batch.
        bsz = cfg.batch
        torch.manual_seed(0xC0FFEE)
        A = (
            torch.randint(
                -4, 5, (bsz, cfg.m, cfg.k), device="cuda", dtype=torch.int32
            ).to(torch.float32)
            * 0.5
        ).to(td)
        B = (
            torch.randint(
                -4, 5, (bsz, cfg.n, cfg.k), device="cuda", dtype=torch.int32
            ).to(torch.float32)
            * 0.5
        ).to(td)
        C = torch.zeros((bsz, cfg.m, cfg.n), device="cuda", dtype=td)
        ref_f32 = torch.bmm(A.to(torch.float32), B.to(torch.float32).transpose(1, 2))
        grid = (gx, gy, bsz)
        # ABI param order: A,B,C,M,N,K,stride_a,stride_b,stride_c (i32 elems)
        sig = gemm_args_signature() + [
            {"name": "stride_a", "type": "i32", "size_bytes": 4},
            {"name": "stride_b", "type": "i32", "size_bytes": 4},
            {"name": "stride_c", "type": "i32", "size_bytes": 4},
        ]
        values = {
            "A": A,
            "B": B,
            "C": C,
            "M": cfg.m,
            "N": cfg.n,
            "K": cfg.k,
            "stride_a": cfg.m * cfg.k,
            "stride_b": cfg.n * cfg.k,
            "stride_c": cfg.m * cfg.n,
        }
    else:
        A, B = _make_inputs(torch, cfg.m, cfg.n, cfg.k, cfg.dtype)
        C = torch.zeros((cfg.m, cfg.n), device="cuda", dtype=td)
        ref_f32 = A.to(torch.float32) @ B.to(torch.float32).t()
        grid = (gx, gy, 1)
        sig = gemm_args_signature()
        values = {"A": A, "B": B, "C": C, "M": cfg.m, "N": cfg.n, "K": cfg.k}

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

    max_abs, max_rel, margin = _compare(torch, C, ref_f32, tol)
    res.max_abs_diff = max_abs
    res.max_rel_diff = max_rel
    res.margin = margin
    ok = margin <= 0.0 and math.isfinite(margin)
    if ok:
        res.status = "GREEN"
    elif cfg.xfail:
        res.status = "XFAIL"
    else:
        res.status = "DRIFT"
    xnote = f" XFAIL[{cfg.xfail}]" if (cfg.xfail and not ok) else ""
    res.detail = (
        f"grid={grid} block={block} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"rtol={tol.rtol:.0e} atol={tol.atol:.0e} margin={margin:.3e}{xnote}"
    )
    return res


# ---------------------------------------------------------------------
# Elementwise lane (second family, proves the harness generalizes)
# ---------------------------------------------------------------------
# dtype here is the rocke elementwise spec dtype ("f16"/"bf16"); the harness
# TOL table is keyed by "fp16"/"bf16" so we map at compare time.
@dataclass(frozen=True)
class ElemCfg:
    name: str
    n: int  # numel (kept a multiple of block_size*vec -> fast path, no remainder)
    op: str  # binary: add|sub|mul|max|min   unary: relu|silu|...
    dtype: str  # "f16" | "bf16"
    block_size: int = 256
    vec: int = 8


# numel = block_size*vec*<tiles>; 256*8 = 2048 elems/block.
ELEM_CONFIGS: List[ElemCfg] = [
    ElemCfg("add_f16_128k", 2048 * 64, "add", "f16"),
    ElemCfg("mul_f16_128k", 2048 * 64, "mul", "f16"),
    ElemCfg("max_f16_128k", 2048 * 64, "max", "f16"),
    ElemCfg("relu_f16_128k", 2048 * 64, "relu", "f16"),
    ElemCfg("add_bf16_128k", 2048 * 64, "add", "bf16"),
    ElemCfg("silu_bf16_128k", 2048 * 64, "silu", "bf16"),
]

# map spec dtype -> TOL-table key
_ELEM_TOL_KEY = {"f16": "fp16", "bf16": "bf16"}


def _torch_elementwise_ref(torch, op: str, A, B):
    """Reference for one elementwise op, computed in fp32."""
    a = A.to(torch.float32)
    if op == "add":
        return a + B.to(torch.float32)
    if op == "sub":
        return a - B.to(torch.float32)
    if op == "mul":
        return a * B.to(torch.float32)
    if op == "max":
        return torch.maximum(a, B.to(torch.float32))
    if op == "min":
        return torch.minimum(a, B.to(torch.float32))
    if op == "relu":
        return torch.relu(a)
    if op == "silu":
        return a * torch.sigmoid(a)
    if op == "neg":
        return -a
    if op == "abs":
        return a.abs()
    if op == "tanh":
        return torch.tanh(a)
    if op == "sigmoid":
        return torch.sigmoid(a)
    raise ValueError(f"no torch reference for elementwise op {op!r}")


def run_elementwise_config(cfg: ElemCfg, arch: str = "gfx950") -> NumericResult:
    """Second family lane: build+launch an elementwise kernel, compare vs a
    torch reference. Proves the harness generalizes beyond GEMM."""
    import torch

    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.elementwise import (
        ElementwiseSpec,
        build_elementwise,
        elementwise_grid,
        elementwise_signature,
        is_valid_spec,
    )

    tol_key = _ELEM_TOL_KEY.get(cfg.dtype, cfg.dtype)
    res = NumericResult(
        family="elementwise",
        name=cfg.name,
        status="GREEN",
        dtype=tol_key,
        shape=(cfg.n,),
    )
    tol = TOL[tol_key]
    res.rtol, res.atol = tol.rtol, tol.atol

    spec = ElementwiseSpec(
        op=cfg.op,
        dtype=cfg.dtype,
        block_size=cfg.block_size,
        vec=cfg.vec,
        name=f"ew_{cfg.name}",
    )
    ok, reason = is_valid_spec(spec)
    if not ok:
        res.status = "REJECTED"
        res.detail = f"is_valid_spec: {reason}"
        return res

    try:
        kern = build_elementwise(spec)
        art = compile_kernel(kern, arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"build/compile raised: {e}"
        return res
    res.extra["kernel_name"] = art.kernel_name
    res.extra["hsaco_bytes"] = art.hsaco_bytes

    td = _torch_dtype(torch, tol_key)
    torch.manual_seed(0xC0FFEE)
    A = (
        torch.randint(-4, 5, (cfg.n,), device="cuda", dtype=torch.int32).to(
            torch.float32
        )
        * 0.5
    ).to(td)
    is_binary = spec.is_binary()
    B = None
    if is_binary:
        B = (
            torch.randint(-4, 5, (cfg.n,), device="cuda", dtype=torch.int32).to(
                torch.float32
            )
            * 0.5
        ).to(td)
    C = torch.zeros((cfg.n,), device="cuda", dtype=td)

    ref_f32 = _torch_elementwise_ref(torch, cfg.op, A, B if is_binary else None)

    grid = elementwise_grid(cfg.n, spec)
    block = (spec.block_size, 1, 1)
    sig = elementwise_signature(spec)
    values: Dict[str, Any] = {"A": A, "C": C, "N": cfg.n}
    if is_binary:
        values["B"] = B
    try:
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        res.status = "LAUNCH_FAIL"
        res.detail = f"launch raised: {e}"
        return res

    max_abs, max_rel, margin = _compare(torch, C, ref_f32, tol)
    res.max_abs_diff = max_abs
    res.max_rel_diff = max_rel
    res.margin = margin
    res.status = "GREEN" if margin <= 0.0 and math.isfinite(margin) else "DRIFT"
    res.detail = (
        f"op={cfg.op} grid={grid} block={block} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} margin={margin:.3e}"
    )
    return res


# ---------------------------------------------------------------------
# Row-norm / reduce lane (layernorm2d, rmsnorm2d, reduce2d)
# ---------------------------------------------------------------------
# These are one-CTA-per-row kernels (grid = (M,1,1), block = (block_size,
# 1,1)) over an (M, N) activation tensor with N == n_per_block (one CTA
# covers a full row). The spec dtype is "f16"/"bf16"; the TOL table is
# keyed "fp16"/"bf16" so we map at compare time.
#
# Tolerance note: layernorm/rmsnorm divide by a per-row inv_std/inv_rms
# whose magnitude depends on the row variance; relative error in that
# single reciprocal-sqrt dominates, so the per-element rtol below is
# looser than the GEMM table (and matches the manifest_runner's own
# verify thresholds in manifest_runner/simple_ops.py).
@dataclass(frozen=True)
class RowCfg:
    name: str
    family: str  # "layernorm2d" | "rmsnorm2d" | "reduce2d"
    m: int
    n: int  # == n_per_block
    dtype: str  # "f16" | "bf16"
    op: str = ""  # reduce only: sum|max|min|mean
    block_size: int = 256
    vec: int = 8


ROW_CONFIGS: List[RowCfg] = [
    # layernorm2d
    RowCfg("ln_f16_512x4096", "layernorm2d", 512, 4096, "f16"),
    RowCfg("ln_bf16_512x4096", "layernorm2d", 512, 4096, "bf16"),
    RowCfg("ln_f16_1024x2048", "layernorm2d", 1024, 2048, "f16"),
    # rmsnorm2d
    RowCfg("rms_f16_512x4096", "rmsnorm2d", 512, 4096, "f16"),
    RowCfg("rms_bf16_512x4096", "rmsnorm2d", 512, 4096, "bf16"),
    RowCfg("rms_f16_1024x2048", "rmsnorm2d", 1024, 2048, "f16"),
    # reduce2d (sum / max / mean over the N axis)
    RowCfg("red_sum_f16_512x4096", "reduce2d", 512, 4096, "f16", op="sum"),
    RowCfg("red_max_f16_512x4096", "reduce2d", 512, 4096, "f16", op="max"),
    RowCfg("red_mean_f16_512x4096", "reduce2d", 512, 4096, "f16", op="mean"),
    RowCfg("red_sum_bf16_512x4096", "reduce2d", 512, 4096, "bf16", op="sum"),
]

# Per-family tolerance overrides. layernorm/rmsnorm carry a divide by a
# per-row reciprocal so they get a looser rtol than the elementwise table.
_ROW_TOL: Dict[str, Tol] = {
    "layernorm2d": Tol(rtol=2e-1, atol=1e-1),
    "rmsnorm2d": Tol(rtol=5e-2, atol=5e-3),
    # reduce sum over 4096 f16 terms: each add quantizes, error grows ~sqrt(N);
    # keep an absolute floor proportional to N and a modest rtol.
    "reduce2d": Tol(rtol=5e-2, atol=5e-2),
}


def _row_tol(cfg: RowCfg) -> Tol:
    return _ROW_TOL[cfg.family]


def run_row_config(cfg: RowCfg, arch: str = "gfx950") -> NumericResult:
    """Build+launch one row-norm / reduce kernel, compare vs a torch ref."""
    import torch

    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    tol_key = _ELEM_TOL_KEY.get(cfg.dtype, cfg.dtype)
    res = NumericResult(
        family=cfg.family,
        name=cfg.name,
        status="GREEN",
        dtype=tol_key,
        shape=(cfg.m, cfg.n),
    )
    tol = _row_tol(cfg)
    res.rtol, res.atol = tol.rtol, tol.atol

    td = _torch_dtype(torch, tol_key)
    eps = 1e-5

    # --- build spec + kernel for the requested family ---
    try:
        # Thread the running arch's wave size into the row-reduction specs so the
        # cross-wave fold matches the hardware wavefront (wave32 on RDNA, wave64
        # on CDNA). Defaulting to 64 on a wave32 part does a cross-half butterfly
        # over lanes that do not exist and miscounts the waves per CTA, dropping
        # partials -> wrong reduction.
        from rocke.core.arch import ArchTarget as _ArchTarget

        _wave = _ArchTarget.from_gfx(arch).wave_size
        if cfg.family == "layernorm2d":
            from rocke.instances.common.layernorm2d import (
                LayerNorm2DSpec,
                build_layernorm2d,
                is_valid_spec,
                layernorm2d_grid,
                layernorm2d_signature,
            )

            spec = LayerNorm2DSpec(
                n_per_block=cfg.n,
                block_size=cfg.block_size,
                vec=cfg.vec,
                dtype=cfg.dtype,
                wave_size=_wave,
            )
            ok, reason = is_valid_spec(spec, arch=arch)
            builder, grid_fn, sig_fn = (
                build_layernorm2d,
                layernorm2d_grid,
                layernorm2d_signature,
            )
        elif cfg.family == "rmsnorm2d":
            from rocke.instances.common.rmsnorm2d import (
                RMSNorm2DSpec,
                build_rmsnorm2d,
                is_valid_spec,
                rmsnorm2d_grid,
                rmsnorm2d_signature,
            )

            spec = RMSNorm2DSpec(
                n_per_block=cfg.n,
                block_size=cfg.block_size,
                vec=cfg.vec,
                dtype=cfg.dtype,
                wave_size=_wave,
            )
            ok, reason = is_valid_spec(spec, arch=arch)
            builder, grid_fn, sig_fn = (
                build_rmsnorm2d,
                rmsnorm2d_grid,
                rmsnorm2d_signature,
            )
        elif cfg.family == "reduce2d":
            from rocke.instances.common.reduce import (
                Reduce2DSpec,
                build_reduce2d,
                is_valid_spec,
                reduce2d_grid,
                reduce2d_signature,
            )

            spec = Reduce2DSpec(
                n_per_block=cfg.n,
                op=cfg.op,
                block_size=cfg.block_size,
                vec=cfg.vec,
                dtype=cfg.dtype,
                wave_size=_wave,
            )
            ok, reason = is_valid_spec(spec)
            builder, grid_fn, sig_fn = (
                build_reduce2d,
                reduce2d_grid,
                reduce2d_signature,
            )
        else:  # pragma: no cover - config typo guard
            res.status = "REJECTED"
            res.detail = f"unknown row family {cfg.family!r}"
            return res
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"spec import raised: {e}"
        return res

    if not ok:
        res.status = "REJECTED"
        res.detail = f"is_valid_spec: {reason}"
        return res

    try:
        kern = builder(spec)
        art = compile_kernel(kern, arch=arch)
    except Exception as e:  # noqa: BLE001
        res.status = "BUILD_FAIL"
        res.detail = f"build/compile raised: {e}"
        return res
    res.extra["kernel_name"] = art.kernel_name
    res.extra["hsaco_bytes"] = art.hsaco_bytes

    # --- deterministic inputs + torch reference (all in f32) ---
    torch.manual_seed(0xC0FFEE)
    # small magnitudes so f16/bf16 round-trips and accumulation are tame
    X = (torch.randn((cfg.m, cfg.n), device="cuda", dtype=torch.float32) * 0.5).to(td)
    x32 = X.to(torch.float32)
    grid = grid_fn(cfg.m, spec)
    block = (spec.block_size, 1, 1)
    sig = sig_fn(spec)

    if cfg.family == "layernorm2d":
        Gamma = (
            torch.randn((cfg.n,), device="cuda", dtype=torch.float32) * 0.1 + 1.0
        ).to(td)
        Beta = (torch.randn((cfg.n,), device="cuda", dtype=torch.float32) * 0.1).to(td)
        Y = torch.zeros((cfg.m, cfg.n), device="cuda", dtype=td)
        mean = x32.mean(dim=-1, keepdim=True)
        var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
        inv_std = 1.0 / torch.sqrt(var + eps)
        ref_f32 = (x32 - mean) * inv_std * Gamma.to(torch.float32)[None, :] + Beta.to(
            torch.float32
        )[None, :]
        out = Y
        values = {
            "X": X,
            "Gamma": Gamma,
            "Beta": Beta,
            "Y": Y,
            "M": cfg.m,
            "N": cfg.n,
            "eps": eps,
        }
    elif cfg.family == "rmsnorm2d":
        Gamma = (
            torch.randn((cfg.n,), device="cuda", dtype=torch.float32) * 0.1 + 1.0
        ).to(td)
        Y = torch.zeros((cfg.m, cfg.n), device="cuda", dtype=td)
        rms = torch.sqrt((x32**2).mean(dim=-1, keepdim=True) + eps)
        ref_f32 = (x32 / rms) * Gamma.to(torch.float32)[None, :]
        out = Y
        values = {"X": X, "Gamma": Gamma, "Y": Y, "M": cfg.m, "N": cfg.n, "eps": eps}
    else:  # reduce2d
        Y = torch.zeros((cfg.m,), device="cuda", dtype=td)
        if cfg.op == "sum":
            ref_f32 = x32.sum(dim=-1)
        elif cfg.op == "max":
            ref_f32 = x32.max(dim=-1).values
        elif cfg.op == "min":
            ref_f32 = x32.min(dim=-1).values
        elif cfg.op == "mean":
            ref_f32 = x32.mean(dim=-1)
        else:  # pragma: no cover
            res.status = "REJECTED"
            res.detail = f"no torch ref for reduce op {cfg.op!r}"
            return res
        out = Y
        values = {"X": X, "Y": Y, "M": cfg.m, "N": cfg.n}

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

    max_abs, max_rel, margin = _compare(torch, out, ref_f32, tol)
    res.max_abs_diff = max_abs
    res.max_rel_diff = max_rel
    res.margin = margin
    res.status = "GREEN" if margin <= 0.0 and math.isfinite(margin) else "DRIFT"
    extra = f"op={cfg.op} " if cfg.op else ""
    res.detail = (
        f"{extra}grid={grid} block={block} "
        f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"rtol={tol.rtol:.0e} atol={tol.atol:.0e} margin={margin:.3e}"
    )
    return res


# ---------------------------------------------------------------------
# Attention lane (FMHA forward, unified tiled MFMA body)
# ---------------------------------------------------------------------
# Builds rocke.instances.common.fmha_mfma.build_fmha_fwd_mfma through the
# *comgr* (LLVM-IR) path -- the same Python engine the rest of this lane
# uses -- and compares against a dense fp32 softmax-attention reference
# (== torch.nn.functional.scaled_dot_product_attention up to accumulation
# order). The ABI mirrors examples/common/fmha_fwd_verify_hip.py:
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
    from rocke.instances import FmhaCommonSpec, FmhaShape
    from rocke.instances.common.fmha_mfma import (
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

    for cfg in GEMM_CONFIGS:
        if want("gemm", cfg.name):
            results.append(run_gemm_config(cfg, arch=arch))
    for cfg in ELEM_CONFIGS:
        if want("elementwise", cfg.name):
            results.append(run_elementwise_config(cfg, arch=arch))
    for cfg in ROW_CONFIGS:
        if want(cfg.family, cfg.name):
            results.append(run_row_config(cfg, arch=arch))
    for cfg in ATTN_CONFIGS:
        if want("attention", cfg.name):
            results.append(run_attn_config(cfg, arch=arch))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--only", default="", help="comma-separated family/name substrings")
    ap.add_argument("--json", default=str(TMP / "numeric_dashboard.json"))
    args = ap.parse_args(argv)

    gpu_err = _check_gpu()
    if gpu_err is not None:
        sys.stderr.write(f"GPU unavailable: {gpu_err}\n")
        return 3

    import torch

    print(f"L6 NUMERIC  arch={args.arch}  device={torch.cuda.get_device_name(0)}")
    results = run_all(arch=args.arch, only=args.only)

    rows: List[Dict[str, Any]] = []
    npass = nfail = nrej = nskip = nerr = nxfail = 0
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
        elif r.status == "XFAIL":
            nxfail += 1
        else:
            nerr += 1
        tag = r.status
        line = f"  {tag:11s} {r.family}/{r.name}"
        if r.status in ("GREEN", "DRIFT", "XFAIL"):
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
        f"\n=== L6 NUMERIC SUMMARY ===\n"
        f"  PASS={npass}  FAIL={nfail}  XFAIL={nxfail}  REJECTED={nrej}  "
        f"SKIPPED={nskip}  ERROR={nerr}"
    )
    print(f"dashboard: {args.json}")
    # exit nonzero only on a real numeric drift or an unexpected build/launch error
    return 1 if (nfail or nerr) else 0


if __name__ == "__main__":
    raise SystemExit(main())
