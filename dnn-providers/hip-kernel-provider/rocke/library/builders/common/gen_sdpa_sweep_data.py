#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SDPA (fused multi-head attention) heuristics training-data generator.

Library-side entry point for the SDPA sweep; the platform
:mod:`rocke.heuristics.gen_sweep_data` no longer carries the sdpa adapter.
This module owns the sdpa problem corpus, the problem-driven variant
selector, and the OpAdapter, then calls
:func:`rocke.heuristics.gen_sweep_data.generate` as a service.

Usage::

    python3 -m builders.common.gen_sdpa_sweep_data \\
        --out sdpa_training.parquet \\
        --cache-dir /tmp/rocke_sdpa_cache \\
        --arch gfx950 \\
        --max-shapes 4
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rocke.heuristics.gen_sweep_data import OpAdapter, generate


# =====================================================================
# sdpa problem corpus
# =====================================================================


@dataclass(frozen=True)
class SdpaGraphShape:
    """One SDPA problem, described in the hipDNN ``SdpaGraphConfig`` vocabulary
    (``api/src/tests/dispatcher/SdpaGraphFixture.hpp``) rather than a positional
    tuple. Field names/units match the graph the runtime dispatcher will
    featurize, so a swept shape is expressible as a hipDNN SDPA graph by
    construction -- the eventual consumer, not torch, defines the shape.

    Kernel-capability note: the rocKE 2D tiled kernel
    (``UnifiedAttentionProblem``) carries a SINGLE ``head_size`` and only the
    canonical BSHD layout. So ``head_size_qk`` and ``head_size_v`` MUST be equal
    and ``layout`` MUST be "BSHD" today; :meth:`to_problem` asserts loudly rather
    than silently mis-lowering. The separate QK/V fields are kept so the corpus
    can already *express* MLA-style shapes (they map to the model's hdim_q/hdim_v
    features) the moment the kernel gains support -- at which point the assert
    relaxes, not the schema.
    """

    batch: int
    seqlen_q: int
    seqlen_k: int
    num_query_heads: int
    num_kv_heads: int
    head_size_qk: int
    head_size_v: int
    dtype: str
    sliding_window: int = 0
    layout: str = "BSHD"  # BSHD (canonical) | BHSD -- kernel only does BSHD today
    block_size: int = 16  # paged-KV block; 16 across the supported corpus

    def to_problem(self, UnifiedAttentionProblem):
        """Lower to a UnifiedAttentionProblem, asserting the kernel-unsupported
        axes are within what the tiled kernel accepts (fail loud, never silently
        mis-map). ``num_query_heads`` must be an integer multiple of
        ``num_kv_heads`` (GQA)."""
        if self.head_size_qk != self.head_size_v:
            raise ValueError(
                f"head_size_qk ({self.head_size_qk}) != head_size_v "
                f"({self.head_size_v}); the 2D tiled kernel has a single "
                "head_size. Split-head-dim (MLA) shapes are expressible here but "
                "not yet buildable -- drop them until the kernel supports it."
            )
        if self.layout != "BSHD":
            raise ValueError(
                f"layout {self.layout!r} unsupported; the tiled kernel is "
                "canonical BSHD only."
            )
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be a multiple of "
                f"num_kv_heads ({self.num_kv_heads}) for a valid GQA ratio."
            )
        return UnifiedAttentionProblem(
            total_q=self.batch * self.seqlen_q,
            num_seqs=self.batch,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size_qk,
            block_size=self.block_size,
            max_seqlen_q=self.seqlen_q,
            max_seqlen_k=self.seqlen_k,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
        )


class ShapeUnbuildable(ValueError):
    """A reduced shape record the current tiled sweep cannot build (carries why),
    so ingest can skip it with a counted reason rather than crash the sweep."""


def shape_from_record(rec: dict) -> "SdpaGraphShape":
    """Build an SdpaGraphShape from a compact SdpaProblem record (the JSONL rows
    emitted by the pipeline's graph reducer, core/graph_reduce.py).

    The record carries hipDNN's full field set; the tiled sweep can only build a
    subset today, so this raises ShapeUnbuildable (with a reason) for records the
    sweep can't yet handle -- the caller counts and skips them. Mapping:
      head_size            -> head_size_qk == head_size_v (kernel single head dim)
      mask_mode "none"     -> sliding_window 0
      mask_mode "sliding_window" -> UNBUILDABLE: the reducer does not capture the
                              window magnitude (SdpaGraphAdapter drops bound
                              values), so we can't reconstruct the launch param.
      mask_mode "causal*"  -> UNBUILDABLE: the tiled sweep hardcodes mask=0 today.
    """
    dtype = str(rec["dtype"])
    if dtype not in ("fp16", "bf16"):
        raise ShapeUnbuildable(f"dtype {dtype!r} not built by the tiled sweep")
    layout = str(rec.get("layout", "BSHD"))
    if layout != "BSHD":
        raise ShapeUnbuildable(f"layout {layout!r} not built (BSHD only)")
    mask = str(rec.get("mask_mode", "none"))
    if mask == "none":
        sliding_window = 0
    elif mask == "sliding_window":
        raise ShapeUnbuildable("sliding_window magnitude not captured in the record")
    else:  # causal_top_left / causal_bottom_right
        raise ShapeUnbuildable(f"mask_mode {mask!r} not built by the tiled sweep")
    hd = int(rec["head_size"])
    return SdpaGraphShape(
        batch=int(rec["batch"]),
        seqlen_q=int(rec["seqlen_q"]),
        seqlen_k=int(rec["seqlen_k"]),
        num_query_heads=int(rec["num_query_heads"]),
        num_kv_heads=int(rec["num_kv_heads"]),
        head_size_qk=hd,
        head_size_v=hd,
        dtype=dtype,
        sliding_window=sliding_window,
        layout=layout,
    )


def load_shapes_file(path) -> List["SdpaGraphShape"]:
    """Load a JSONL shape corpus (one SdpaProblem record per line) into
    SdpaGraphShape list. Records the sweep can't build are skipped and reported
    (count by reason) to stderr; a valid line is never silently dropped."""
    import json
    from collections import Counter

    shapes: List[SdpaGraphShape] = []
    skipped: Counter = Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                shapes.append(shape_from_record(rec))
            except ShapeUnbuildable as e:
                skipped[str(e)] += 1
    if skipped:
        print(
            f"[sdpa] shapes-file: loaded {len(shapes)}, skipped "
            f"{sum(skipped.values())} unbuildable:",
            file=sys.stderr,
        )
        for reason, n in skipped.most_common():
            print(f"[sdpa]     {n:>6}  {reason}", file=sys.stderr)
    else:
        print(f"[sdpa] shapes-file: loaded {len(shapes)} shapes", file=sys.stderr)
    return shapes


def _S(
    batch,
    seqlen_q,
    seqlen_k,
    num_query_heads,
    num_kv_heads,
    head_size,
    dtype,
    sliding_window=0,
):
    """Terse builder for the common case (head_size_qk == head_size_v, BSHD,
    block_size 16). Use SdpaGraphShape(...) directly for the exceptions."""
    return SdpaGraphShape(
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size_qk=head_size,
        head_size_v=head_size,
        dtype=dtype,
        sliding_window=sliding_window,
    )


_SDPA_PROBLEMS = [
    # Decode (seqlen_q == 1) across batch + GQA ratios.
    _S(1, 1, 1024, 32, 32, 128, "fp16"),
    _S(8, 1, 2048, 32, 8, 128, "fp16"),
    _S(16, 1, 4096, 32, 8, 128, "bf16"),
    _S(32, 1, 512, 64, 8, 64, "bf16"),
    # Short prefill (q <= 256).
    _S(1, 128, 128, 32, 32, 128, "fp16"),
    _S(4, 256, 256, 32, 8, 128, "bf16"),
    # Medium prefill (256 < q <= 1024).
    _S(1, 512, 512, 32, 32, 64, "fp16"),
    _S(4, 1024, 1024, 32, 8, 128, "bf16"),
    # Long prefill (q > 1024).
    _S(1, 2048, 2048, 16, 16, 128, "bf16"),
    _S(2, 4096, 4096, 32, 4, 64, "bf16"),
    # Sliding-window variants.
    _S(1, 1024, 1024, 32, 8, 128, "bf16", sliding_window=256),
    _S(4, 2048, 2048, 32, 8, 64, "fp16", sliding_window=512),
]


# Config-grid axes, swept per problem around the selector-chosen default and
# filtered by ``supports_tiled_2d`` so only buildable configs survive. A picker
# needs multiple valid configs per shape to have anything to rank; one derived
# spec per problem (the old behaviour) gave it nothing to learn.
_GRID_NUM_WARPS = (1, 2, 4)
_GRID_BLOCK_M_PER_WARP = (16, 32)
# tile_size (T) grid is derived per problem from block_size (T is a multiple of
# the paged-KV block): {1x, 2x, 4x} block_size.
_GRID_TILE_MULT = (1, 2, 4)


@dataclass
class _SdpaCandidate:
    """One (problem, tiled-spec) grid point. The adapter callbacks read config
    columns from ``tiled`` (the actual swept config) and problem columns from
    ``problem``."""

    problem: object
    tiled: object


def _default_tiled_spec(prob: object):
    """The selector-chosen 2D tiled spec for a problem (the grid anchor, with all
    arch-specific flags set correctly)."""
    from kernels.common import attention_unified as au

    return au._tiled_spec_from_problem(prob)


def _grid_tiled_specs(prob: object, arch: str) -> List[object]:
    """Valid tiled-spec grid points for a problem.

    Varies (num_warps, block_m_per_warp, tile_size) around the default spec via
    ``dataclasses.replace`` (so every other subtle flag stays as the selectors
    set it), and keeps only the points ``supports_tiled_2d`` accepts. The default
    spec itself is always included first (deduped) so a problem never yields zero
    candidates even if the grid is fully pruned.
    """
    import dataclasses

    from kernels import supports_tiled_2d

    default = _default_tiled_spec(prob)
    seen: set = set()
    out: List[object] = []

    def _accept(spec) -> None:
        if spec is None:
            return
        key = (spec.num_warps, spec.block_m_per_warp, spec.tile_size)
        if key in seen:
            return
        ok, _reason = supports_tiled_2d(
            head_size=spec.head_size,
            block_size=spec.block_size,
            dtype=spec.dtype,
            num_queries_per_kv=prob.num_queries_per_kv,
            use_alibi=spec.use_alibi,
            use_qq_bias=spec.use_qq_bias,
            use_fp8=prob.use_fp8,
            q_dtype=prob.q_dtype,
            num_warps=spec.num_warps,
            block_m_per_warp=spec.block_m_per_warp,
            kv_storage_dtype=spec.kv_storage_dtype,
            tile_size=spec.tile_size,
            arch=arch,
        )
        if ok:
            seen.add(key)
            out.append(spec)

    # Default first (guaranteed valid — the selectors produced it).
    _accept(default)

    bs = int(prob.block_size)
    for nw in _GRID_NUM_WARPS:
        for bm in _GRID_BLOCK_M_PER_WARP:
            for mult in _GRID_TILE_MULT:
                # dataclasses.replace runs the spec's __post_init__, which raises
                # for internally-inconsistent combos our axis grid can produce
                # (e.g. the default has use_mfma_32x32=True, which requires
                # block_m_per_warp=32, but we also try bm=16). Skip such points --
                # they are simply not valid configs, not a reason to abort the
                # whole sweep. supports_tiled_2d does not cover these cross-flag
                # rules, so the constructor is the authority.
                try:
                    cand = dataclasses.replace(
                        default,
                        num_warps=nw,
                        block_m_per_warp=bm,
                        tile_size=bs * mult,
                    )
                except (ValueError, TypeError):
                    continue
                _accept(cand)
    return out


def _sdpa_enumerate(arch: str, max_shapes: Optional[int]) -> List[object]:
    from kernels.common.attention_unified import UnifiedAttentionProblem

    return _enumerate_from(_SDPA_PROBLEMS, arch, max_shapes)


def _enumerate_from(
    problems: Sequence["SdpaGraphShape"], arch: str, max_shapes: Optional[int]
) -> List[object]:
    from kernels.common.attention_unified import UnifiedAttentionProblem

    if max_shapes is not None and max_shapes > 0:
        problems = problems[:max_shapes]

    specs: List[object] = []
    for shape in problems:
        prob = shape.to_problem(UnifiedAttentionProblem)
        # One candidate per valid grid point (multiple configs per problem).
        for tiled in _grid_tiled_specs(prob, arch):
            specs.append(_SdpaCandidate(problem=prob, tiled=tiled))
    return specs


def _sdpa_build(cand: object):
    from kernels import build_unified_attention_2d_tiled

    # Build the explicit tiled spec for this grid point (arch-dispatched). This
    # is the actual swept config, not a problem-derived default.
    return build_unified_attention_2d_tiled(cand.tiled)


def _sdpa_config_columns(cand: object) -> Dict[str, object]:
    """Recover the 68-feature kernel columns from this grid point's tiled spec.

    The FMHA feature layout treats ``tm0`` as the per-warp query block
    (``block_q`` = block_m_per_warp), ``tn0`` as the tile_size T, ``tk0``/
    ``tk0max`` as head_size, ``tn1`` as hdim_v, and ``tk1`` as T -- exactly
    mirroring the C++ derivation so the Python and runtime feature vectors agree
    field-for-field.
    """
    spec = cand.tiled
    prob = cand.problem
    T = int(getattr(spec, "tile_size", None) or (2 * int(prob.block_size)))
    block_q = int(getattr(spec, "block_m_per_warp", 16))
    num_warps = int(getattr(spec, "num_warps", 1))
    pipeline = 1  # qr_async
    mask = 0
    sink = bool(getattr(spec, "use_sinks", False))

    hd = int(prob.head_size)
    return {
        "pipeline": pipeline,
        "num_warps": num_warps,
        "tile_m0": block_q,
        "tile_n0": T,
        "tile_k0": hd,
        "tile_n1": hd,
        "tile_k1": T,
        "tile_k0max": hd,
        "pad_s": 0,
        "pad_sk": 0,
        "pad_d": 0,
        "pad_dv": 0,
        "mask": mask,
        "bias": 0,
        "lse": 0,
        "dropout": 0,
        "logits": 0,
        "sink": 1 if sink else 0,
        "skip": 0,
        "qscale": 0,
        "paged": 1,
    }


def _sdpa_problem_columns(cand: object) -> Dict[str, object]:
    prob = cand.problem
    return {
        "batch": int(prob.num_seqs),
        "seqlen_q": int(prob.max_seqlen_q),
        "seqlen_k": int(prob.max_seqlen_k),
        "nhead_q": int(prob.num_query_heads),
        "nhead_k": int(prob.num_kv_heads),
        "hdim_q": int(prob.head_size),
        "hdim_v": int(prob.head_size),
        "dtype": str(prob.dtype),
        "sliding_window": int(prob.sliding_window),
    }


def _sdpa_flops(cand: object) -> float:
    prob = cand.problem
    b = prob.num_seqs
    hq = prob.num_query_heads
    sq = prob.max_seqlen_q
    sk = prob.max_seqlen_k
    d = prob.head_size
    return float(2.0 * b * hq * sq * sk * (d + d))


def _sdpa_spec_name(cand: object) -> str:
    prob = cand.problem
    spec = cand.tiled
    return (
        f"sdpa_b{prob.num_seqs}_sq{prob.max_seqlen_q}_sk{prob.max_seqlen_k}"
        f"_hq{prob.num_query_heads}_hk{prob.num_kv_heads}_d{prob.head_size}"
        f"_{prob.dtype}_nw{spec.num_warps}_bm{spec.block_m_per_warp}_T{spec.tile_size}"
    )


# =====================================================================
# GPU benchmark (real TFLOPS + lightweight correctness)
# =====================================================================


def _ua_shape_for(prob: object):
    """Build a UAShape (the benchmark harness's input descriptor) from a
    UnifiedAttentionProblem, computing the paged-KV bookkeeping the trace format
    records (num_blocks, max_blocks_per_seq) that the problem doesn't carry."""
    import math

    from rocke.assets import shape_utils_dir

    sys.path.insert(0, str(shape_utils_dir()))
    from _ua_shape_utils import UAShape  # noqa: E402

    bs = int(prob.block_size)
    max_blocks_per_seq = max(1, math.ceil(int(prob.max_seqlen_k) / bs))
    num_blocks = max_blocks_per_seq * int(prob.num_seqs)
    dt = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}.get(
        str(prob.dtype), "torch.bfloat16"
    )
    win = int(prob.sliding_window)
    return UAShape(
        source_file="gen_sdpa",
        line_idx=0,
        call_idx=0,
        kind="prefill_2d",
        all_decode=(int(prob.max_seqlen_q) == 1),
        num_seqs=int(prob.num_seqs),
        total_q=int(prob.total_q),
        num_query_heads=int(prob.num_query_heads),
        num_kv_heads=int(prob.num_kv_heads),
        head_size=int(prob.head_size),
        block_size=bs,
        num_blocks=num_blocks,
        max_blocks_per_seq=max_blocks_per_seq,
        max_seqlen_q=int(prob.max_seqlen_q),
        max_seqlen_k=int(prob.max_seqlen_k),
        softmax_scale=1.0 / math.sqrt(int(prob.head_size)),
        softcap=float(prob.softcap),
        window_size=((win - 1) if win > 0 else -1, 0),
        has_sinks=bool(prob.use_sinks),
        has_alibi=bool(prob.use_alibi),
        has_output_scale=False,
        q_dtype=dt,
        k_dtype=dt,
        v_dtype=dt,
        out_dtype=dt,
    )


def _sdpa_benchmark(cand: object) -> Dict[str, object]:
    """Time this grid point on the GPU and return {tflops, latency_ms, correct}.

    Builds inputs via the shared UA harness, forces the tiled-2D backend with
    THIS candidate's explicit tiled spec (via ``tiled_spec=`` — the runner builds
    and launches that exact num_warps/T/block_m config, not the selector
    default), times it with rocke's HIP-event timer, and computes TFLOPS from the
    analytic attention FLOP count. Correctness is a lightweight NaN/inf/finite
    check on the output (a full reference compare is AITER-coupled and too heavy
    per grid point); a kernel that returns non-finite output is marked
    correct=False so it can't win the oracle-best selection.
    """
    import sys as _sys

    import torch

    from rocke.assets import shape_utils_dir
    from rocke.runtime import synchronize_and_release, time_launches
    from kernels import UnifiedAttentionProblem, run_unified_attention_torch

    _sys.path.insert(0, str(shape_utils_dir()))
    from _ua_shape_utils import attention_flops, make_inputs  # noqa: E402

    prob = cand.problem
    shape = _ua_shape_for(prob)
    data = make_inputs(shape, seed=0)

    # Rebuild a problem carrying num_sms etc.; dtype/sw already match cand.problem.
    hip_stream = int(torch.cuda.current_stream().cuda_stream)

    def call_once():
        run_unified_attention_torch(
            problem=prob,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=data["output"],
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(prob.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            backend="tiled",
            stream=hip_stream,
            tiled_spec=cand.tiled,  # benchmark THIS grid point's exact config
        )

    latency_ms = time_launches(call_once, warmup=5, iters=20, stream=hip_stream)
    synchronize_and_release(hip_stream)

    out = data["output"]
    correct = bool(torch.isfinite(out.float()).all().item())

    flops = attention_flops(shape, data["query_lens"], data["kv_lens_list"])
    tflops = (flops / 1e12) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    # Bytes moved: Q + O (per-token) + K + V (per KV token), all at dtype width.
    # Attention decode is memory-bound, so this bandwidth head is meaningful — and
    # train.py trains a bandwidth model per op, so it must be > 0 for valid rows.
    bpe = out.element_size()
    d = int(prob.head_size)
    tot_q = int(prob.total_q)
    tot_kv = sum(int(x) for x in data["kv_lens_list"])
    bytes_moved = (
        float(
            tot_q * int(prob.num_query_heads) * d * 2  # Q read + O write
            + tot_kv * int(prob.num_kv_heads) * d * 2  # K + V read
        )
        * bpe
    )
    bandwidth_gb_s = (bytes_moved / 1e9) / (latency_ms / 1e3) if latency_ms > 0 else 0.0

    return {
        "tflops": tflops,
        "latency_ms": latency_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "correct": correct,
    }


# =====================================================================
# Public adapter factory
# =====================================================================


def build_sdpa_adapter(
    shapes: Optional[Sequence["SdpaGraphShape"]] = None,
) -> OpAdapter:
    """Construct the SDPA OpAdapter for use with ``generate()``.

    ``shapes`` overrides the built-in ``_SDPA_PROBLEMS`` corpus (e.g. the
    hipDNN-derived shapes from a --shapes-file). It is closed over so the
    generic ``enumerate_specs(arch, max_shapes)`` contract is preserved.
    """
    if shapes is None:
        enumerate_specs = _sdpa_enumerate
    else:
        shapes = list(shapes)

        def enumerate_specs(arch: str, max_shapes: Optional[int]) -> List[object]:
            return _enumerate_from(shapes, arch, max_shapes)

    return OpAdapter(
        op_type="fmha",
        enumerate_specs=enumerate_specs,
        build_kernel=_sdpa_build,
        spec_name=_sdpa_spec_name,
        config_columns=_sdpa_config_columns,
        problem_columns=_sdpa_problem_columns,
        flops=_sdpa_flops,
        benchmark=_sdpa_benchmark,
    )


# =====================================================================
# CLI
# =====================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "SDPA (fused multi-head attention) heuristics training-data generator. "
            "Library entry point — calls rocke.heuristics.gen_sweep_data.generate() "
            "with the sdpa adapter."
        )
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output training parquet path."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rocke_sdpa_cache"),
        help="Directory for cached HSACO binaries + manifests.",
    )
    parser.add_argument("--arch", default="gfx950", help="GPU architecture.")
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="Limit number of SDPA problems (smoke tests).",
    )
    parser.add_argument(
        "--shapes-file",
        type=Path,
        default=None,
        help=(
            "JSONL of compact SdpaProblem records (one per line, from the "
            "pipeline graph reducer) to sweep INSTEAD of the built-in corpus. "
            "Records the tiled sweep can't build are skipped + reported."
        ),
    )
    args = parser.parse_args(argv)

    shapes = load_shapes_file(args.shapes_file) if args.shapes_file else None
    if args.shapes_file is not None and not shapes:
        print(
            "[sdpa] shapes-file yielded 0 buildable shapes; nothing to sweep",
            file=sys.stderr,
        )
        return 2

    generate(
        op="sdpa",
        out_path=args.out,
        cache_dir=args.cache_dir,
        arch=args.arch,
        max_shapes=args.max_shapes,
        adapter=build_sdpa_adapter(shapes=shapes),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
