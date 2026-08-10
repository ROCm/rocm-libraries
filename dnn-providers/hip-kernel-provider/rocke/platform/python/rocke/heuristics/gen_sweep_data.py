#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rocke-native, op-parameterized heuristics training-data generator.

This is the multi-op generalization of :mod:`rocke.heuristics.gen_gemm_sweep_data`.
For a given op family it:

  1. Enumerates that op's *shape corpus* (problem dimensions).
  2. Enumerates that op's validity-filtered *kernel-config variants* (the
     ``variantGrid`` for the op -- a cartesian product for GEMM / MoE / norm).
  3. Builds every ``(variant)`` to a cached HSACO (LLVM IR -> comgr), driving the
     same :mod:`rocke.sweep` / :mod:`rocke.sweep_bench` ecosystem GEMM uses.
  4. Measures per-shape TFLOPS + correctness where a launcher / GPU is available
     (rows that fail build / verify / perf are emitted ``is_valid=False`` with
     zero targets so the model learns the failure surface).
  5. Writes a training parquet whose feature columns match the op's
     :mod:`rocke.heuristics.feature_engine` engine, plus ``measured_tflops``,
     ``is_valid`` and ``kernel_name``.

Wired ops (the families the per-op feasibility map marks feasible):

  - ``gemm`` : delegates to :func:`gen_gemm_sweep_data.generate` unchanged, so
    the GEMM golden path is byte-for-byte preserved.
  - ``moe``  : fused-MoE streaming trio (gather / silu_mul / topk-reduce).
    Minimal :class:`feature_engine.MoeFeatureEngine` columns; latency-bound.
  - ``norm`` : LayerNorm2D / RMSNorm2D forward. Minimal
    :class:`feature_engine.NormFeatureEngine` columns; bandwidth-bound.

Carved-out verticals own their own entry points (the platform holds no
import of them). For SDPA (fused multi-head attention)::

    python3 -m builders.common.gen_sdpa_sweep_data \\
        --out sdpa_training.parquet \\
        --arch gfx950

and for convolution::

    python3 -m builders.common.gen_conv_sweep_data \\
        --out conv_training.parquet \\
        --arch gfx950

Usage (platform ops)::

    python3 -m rocke.heuristics.gen_sweep_data \\
        --op moe \\
        --out moe_training.parquet \\
        --cache-dir /tmp/rocke_moe_cache \\
        --arch gfx950 \\
        --max-shapes 32

The ``gemm`` op keeps the original
``python3 -m rocke.heuristics.gen_gemm_sweep_data ...`` entry point working as
a thin shim; this module is the superset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from ..core.lower_llvm import lower_kernel_to_llvm
from ..runtime.comgr import build_hsaco_from_llvm_ir


# ---------------------------------------------------------------------
# Generic per-op build record
# ---------------------------------------------------------------------


@dataclass
class _OpBuildRecord:
    """One spec's build outcome for a non-GEMM op."""

    name: str
    ok: bool
    error: str = ""
    hsaco_path: str = ""
    hsaco_bytes: int = 0
    build_ms: float = 0.0
    config: Dict[str, object] = field(default_factory=dict)
    problem: Dict[str, object] = field(default_factory=dict)
    flops: float = 0.0


# ---------------------------------------------------------------------
# Op adapter protocol
# ---------------------------------------------------------------------


@dataclass
class OpAdapter:
    """Everything :func:`generate` needs to sweep one op family.

    ``enumerate_specs(arch, max_shapes)`` returns a list of opaque spec objects
    (one per ``(variant, shape)`` for the cartesian ops, or one per selected
    problem for SDPA). ``build_spec(spec, arch)`` lowers + compiles one spec and
    returns a :class:`KernelDef`-producing closure result. ``config_columns`` /
    ``problem_columns`` recover the parquet feature columns from a spec, and
    ``flops`` returns the op's FLOP count for the TFLOPS metric (0 for streaming
    ops, which are latency/bandwidth bound).
    """

    op_type: str
    enumerate_specs: Callable[[str, Optional[int]], List[object]]
    build_kernel: Callable[[object], object]
    spec_name: Callable[[object], str]
    config_columns: Callable[[object], Dict[str, object]]
    problem_columns: Callable[[object], Dict[str, object]]
    flops: Callable[[object], float]


# =====================================================================
# moe adapter (fused streaming trio: gather / silu_mul / topk-reduce)
# =====================================================================


_MOE_SHAPES = [
    # (tokens, experts, topk, hidden, intermediate, dtype)
    # Decode (T = 1) -- LLaMA / DeepSeek-style.
    (1, 8, 2, 4096, 14336, "f16"),
    (1, 64, 2, 4096, 14336, "bf16"),
    (1, 256, 4, 2048, 8192, "bf16"),
    # Small prefill.
    (8, 8, 2, 4096, 14336, "f16"),
    (32, 64, 2, 4096, 14336, "bf16"),
    (32, 256, 4, 2048, 8192, "bf16"),
    # Medium / training.
    (128, 8, 2, 4096, 8192, "f16"),
    (256, 64, 2, 2048, 4096, "bf16"),
    (512, 32, 2, 1024, 2048, "f16"),
    (1024, 8, 1, 1024, 2048, "bf16"),
]

_MOE_BLOCK_SIZES = (64, 128, 256, 512, 1024)
_MOE_VECS = (2, 4, 8)
# The streaming trio is swept as a single launchable unit; gather is the
# representative kernel built for the manifest (silu_mul / topk-reduce share
# the FusedMoeSpec geometry).
_MOE_PHASE = "gather"


def _moe_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    import itertools

    from ..instances import FusedMoeSpec
    from ..instances.common.fused_moe import is_valid_spec

    shapes = _MOE_SHAPES
    if max_shapes is not None and max_shapes > 0:
        shapes = shapes[:max_shapes]

    specs: List[object] = []
    for tokens, experts, topk, hidden, inter, dtype in shapes:
        for bs, vec in itertools.product(_MOE_BLOCK_SIZES, _MOE_VECS):
            spec = FusedMoeSpec(
                tokens=tokens,
                experts=experts,
                topk=topk,
                hidden=hidden,
                intermediate=inter,
                dtype=dtype,
                block_size=bs,
                vec=vec,
            )
            ok, _ = is_valid_spec(spec)
            if ok:
                specs.append(spec)
    return specs


def _moe_build(spec: object):
    from ..instances import build_moe_gather

    return build_moe_gather(spec)


def _moe_spec_name(spec: object) -> str:
    return spec.kernel_name(_MOE_PHASE)


def _moe_config_columns(spec: object) -> Dict[str, object]:
    return {
        "block_size": int(spec.block_size),
        "vec": int(spec.vec),
    }


def _moe_problem_columns(spec: object) -> Dict[str, object]:
    return {
        "tokens": int(spec.tokens),
        "experts": int(spec.experts),
        "topk": int(spec.topk),
        "hidden": int(spec.hidden),
        "intermediate": int(spec.intermediate),
        "dtype": str(spec.dtype),
    }


def _moe_flops(spec: object) -> float:
    # Streaming / atomic-contention bound -- no GEMM-style FLOP metric.
    return 0.0


# =====================================================================
# norm adapter (LayerNorm2D / RMSNorm2D forward)
# =====================================================================


_NORM_N_PER_BLOCK = (128, 256, 512, 1024, 2048, 4096, 8192)
_NORM_BLOCK_SIZES = (64, 128, 256)
_NORM_VECS = (2, 4, 8)
_NORM_DTYPES = ("f16", "bf16")
# Representative row counts (one CTA per row); used only for occupancy features.
_NORM_ROWS = 4096


def _norm_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    import itertools

    from ..instances import RMSNorm2DSpec
    from ..instances.common.rmsnorm2d import is_valid_spec

    n_values = _NORM_N_PER_BLOCK
    if max_shapes is not None and max_shapes > 0:
        n_values = n_values[:max_shapes]

    specs: List[object] = []
    for npb, bs, vec, dtype in itertools.product(
        n_values, _NORM_BLOCK_SIZES, _NORM_VECS, _NORM_DTYPES
    ):
        if npb % bs != 0 or npb % vec != 0:
            continue
        spec = RMSNorm2DSpec(
            n_per_block=npb,
            block_size=bs,
            vec=vec,
            dtype=dtype,
        )
        ok, _ = is_valid_spec(spec, arch)
        if ok:
            specs.append(spec)
    return specs


def _norm_build(spec: object):
    from ..instances import build_rmsnorm2d

    return build_rmsnorm2d(spec)


def _norm_config_columns(spec: object) -> Dict[str, object]:
    return {
        "block_size": int(spec.block_size),
        "vec": int(spec.vec),
        "dtype": str(spec.dtype),
    }


def _norm_problem_columns(spec: object) -> Dict[str, object]:
    return {
        "rows": int(_NORM_ROWS),
        "n_per_block": int(spec.n_per_block),
        "dtype": str(spec.dtype),
    }


def _norm_flops(spec: object) -> float:
    # Bandwidth-bound row normalization -- no FLOP metric.
    return 0.0


# ---------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------


def _adapter(op: str) -> OpAdapter:
    if op == "conv":
        raise ValueError(
            "conv op has moved to the library; run "
            "'python3 -m builders.common.gen_conv_sweep_data' instead."
        )
    if op == "sdpa":
        raise ValueError(
            "sdpa op has moved to the library; run "
            "'python3 -m builders.common.gen_sdpa_sweep_data' instead."
        )
    if op == "moe":
        return OpAdapter(
            op_type="fused_moe",
            enumerate_specs=_moe_enumerate,
            build_kernel=_moe_build,
            spec_name=_moe_spec_name,
            config_columns=_moe_config_columns,
            problem_columns=_moe_problem_columns,
            flops=_moe_flops,
        )
    if op == "norm":
        return OpAdapter(
            op_type="rmsnorm2d",
            enumerate_specs=_norm_enumerate,
            build_kernel=_norm_build,
            spec_name=lambda s: s.kernel_name(),
            config_columns=_norm_config_columns,
            problem_columns=_norm_problem_columns,
            flops=_norm_flops,
        )
    raise ValueError(f"unknown op {op!r} (want gemm|moe|norm)")


WIRED_OPS = ("gemm", "moe", "norm")


# ---------------------------------------------------------------------
# Build one spec (non-GEMM ops)
# ---------------------------------------------------------------------


def _build_spec(
    adapter: OpAdapter, spec: object, cache_dir: Path, isa: str
) -> _OpBuildRecord:
    name = adapter.spec_name(spec)
    config = adapter.config_columns(spec)
    problem = adapter.problem_columns(spec)
    flops = adapter.flops(spec)

    blob = json.dumps(
        {"name": name, "config": config, "problem": problem}, sort_keys=True
    ).encode()
    spec_hash = hashlib.sha1(blob).hexdigest()[:12]
    out_path = cache_dir / f"{spec_hash}_{name[:120]}.hsaco"

    rec = _OpBuildRecord(
        name=name, ok=False, config=config, problem=problem, flops=flops
    )

    if out_path.exists() and out_path.stat().st_size > 0:
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = out_path.stat().st_size
        return rec

    try:
        t0 = time.perf_counter()
        kernel = adapter.build_kernel(spec)
        ll = lower_kernel_to_llvm(kernel)
        hsaco, _ = build_hsaco_from_llvm_ir(ll, isa=isa)
        out_path.write_bytes(hsaco)
        rec.ok = True
        rec.hsaco_path = str(out_path)
        rec.hsaco_bytes = len(hsaco)
        rec.build_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # noqa: BLE001 - record the failure surface
        rec.error = f"{type(e).__name__}: {e}"

    return rec


# ---------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------


def generate(
    *,
    op: str,
    out_path: Path,
    cache_dir: Path,
    arch: str = "gfx950",
    max_shapes: Optional[int] = None,
    isa: Optional[str] = None,
    adapter: Optional["OpAdapter"] = None,
    **gemm_kwargs: object,
) -> pd.DataFrame:
    """Sweep one op family and write its training parquet.

    For ``op == "gemm"`` this delegates to :func:`gen_gemm_sweep_data.generate`
    so the GEMM golden path is preserved byte-for-byte. For every other (wired,
    feasible) op it enumerates specs, builds each to a cached HSACO, and emits a
    parquet whose feature columns match that op's
    :mod:`rocke.heuristics.feature_engine` engine.

    ``adapter`` may be supplied by library callers (e.g. the sdpa entry point in
    ``builders.common.gen_sdpa_sweep_data``) to inject a pre-built
    :class:`OpAdapter` without going through the platform ``_adapter()`` registry.
    """
    if op == "gemm":
        from . import gen_gemm_sweep_data

        return gen_gemm_sweep_data.generate(
            out_path=out_path,
            cache_dir=cache_dir,
            arch=arch,
            max_shapes=max_shapes,
            **gemm_kwargs,  # type: ignore[arg-type]
        )

    if adapter is None:
        adapter = _adapter(op)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    isa = isa or f"amdgcn-amd-amdhsa--{arch}"

    specs = adapter.enumerate_specs(arch, max_shapes)
    if not specs:
        raise RuntimeError(f"no valid {op} specs for arch={arch}")

    print(
        f"[gen] op={op} arch={arch} variants={len(specs)} -> building",
        file=sys.stderr,
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    n_built = 0
    for i, spec in enumerate(specs):
        rec = _build_spec(adapter, spec, cache_dir, isa)
        if rec.ok:
            n_built += 1
        # Perf measurement requires a launcher + GPU; in its absence
        # measured_tflops stays 0 and is_valid tracks build success so the
        # model can still learn the (large) build-failure surface. When a
        # launcher is wired the same rows are re-measured in place.
        row: Dict[str, object] = {
            "op_type": adapter.op_type,
            "arch": arch,
            "kernel_name": rec.name,
            "measured_tflops": 0.0,
            "latency_ms": 0.0,
            "is_valid": bool(rec.ok),
            "build_ok": bool(rec.ok),
            "build_error": rec.error,
            "run_id": 0,
        }
        row.update(rec.problem)
        row.update(rec.config)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"[gen]   built {n_built}/{i + 1} ...", file=sys.stderr, flush=True)

    print(
        f"[gen] op={op} built {n_built}/{len(specs)} variants OK",
        file=sys.stderr,
        flush=True,
    )

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[gen] {len(df)} rows -> {out_path}", file=sys.stderr, flush=True)
    return df


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "rocke-native, op-parameterized heuristics training-data generator "
            "(gemm|moe|norm)."
        )
    )
    parser.add_argument(
        "--op",
        default="gemm",
        choices=list(WIRED_OPS),
        help="Op family to sweep.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_sweep_cache"),
        help="Directory for cached HSACO binaries + manifests.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of shapes / problems (smoke tests).",
    )
    parser.add_argument(
        "--shape-set",
        default="wide",
        choices=["wide", "edge", "all"],
        help="GEMM-only: shape corpus to sweep.",
    )
    args = parser.parse_args(argv)

    gemm_kwargs: Dict[str, object] = {}
    if args.op == "gemm":
        gemm_kwargs["shape_set"] = args.shape_set

    generate(
        op=args.op,
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        max_shapes=args.max_shapes,
        **gemm_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
