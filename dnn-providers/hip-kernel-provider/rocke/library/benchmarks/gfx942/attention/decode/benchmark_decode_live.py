# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic decode attention benchmark: DSL split-KV 3D vs AITER Triton (gfx942).

gfx942 sibling of
``benchmarks.gfx950.attention.decode.benchmark_decode_live``.

Differences vs gfx950:
  * ``num_sms`` sweep defaults to ``30 60 80 120`` (gfx942 has 228 CUs).
  * No FP8 KV-cache (not supported on gfx942).
  * ``dtype`` defaults to ``bf16``; ``fp16`` is also supported.

Run::

    python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \\
        --shapes library/benchmarks/gfx942/attention/decode/qwen3_30b_a3b_shapes.json

    # or with a custom sweep:
    python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \\
        --shapes shapes.json \\
        --num-sms-sweep 30 60 120 \\
        --output-json /tmp/decode_gfx942.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ARCH = "gfx942"


# ---------------------------------------------------------------------------
# Shape loading (identical format to gfx950 sibling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeShape:
    batch: int
    seqlen_q: int
    seqlen_k: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: str
    label: str

    @property
    def signature(self) -> str:
        return (
            f"b{self.batch}_sq{self.seqlen_q}_sk{self.seqlen_k}"
            f"_nhq{self.num_query_heads}_nhk{self.num_kv_heads}"
            f"_hd{self.head_size}_bs{self.block_size}_{self.dtype}"
        )


def load_decode_shapes(paths: List[Path]) -> List[DecodeShape]:
    """Load shapes from one or more JSON files.

    Supported formats:
    - ``{"models": [{model-defaults, "shapes": [...]}]}`` — multi-model file
      (decode_shapes.json); model-level fields serve as defaults for each shape.
    - ``{"meta": {...}, "shapes": [...]}`` — single-model file
      (qwen3_30b_a3b_shapes.json); meta fields are defaults for all shapes.
    - A bare list of self-contained shape dicts.
    """
    shapes = []
    for path in paths:
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            groups = [({}, raw)]
        elif "models" in raw:
            groups = [(m, m.get("shapes", [])) for m in raw["models"]]
        else:
            groups = [(raw.get("meta", {}), raw.get("shapes", []))]
        for defaults, entries in groups:
            for entry in entries:
                merged = {**defaults, **entry}
                shape = DecodeShape(
                    batch=int(merged["batch"]),
                    seqlen_q=int(merged.get("seqlen_q", 1)),
                    seqlen_k=int(merged["seqlen_k"]),
                    num_query_heads=int(merged["num_query_heads"]),
                    num_kv_heads=int(merged["num_kv_heads"]),
                    head_size=int(merged["head_size"]),
                    block_size=int(merged["block_size"]),
                    dtype=str(merged.get("dtype", "bf16")),
                    label=str(merged.get("label", f"kv{merged['seqlen_k']}")),
                )
                shapes.append(shape)
    return shapes


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _make_inputs(
    shape: DecodeShape,
    *,
    cap_blocks: int = 65536,
    softcap: float = 0.0,
    use_alibi: bool = False,
    use_qq_bias: bool = False,
):
    """Allocate paged-KV tensors for one decode shape."""
    import torch

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
    num_blks = (shape.seqlen_k + shape.block_size - 1) // shape.block_size
    pool = min(num_blks * shape.batch + 64, cap_blocks)

    q = (
        torch.randn(
            shape.batch,
            shape.num_query_heads,
            shape.head_size,
            dtype=dtype,
            device="cuda",
        )
        * 0.1
    )
    kc = (
        torch.randn(
            pool,
            shape.block_size,
            shape.num_kv_heads,
            shape.head_size,
            dtype=dtype,
            device="cuda",
        )
        * 0.1
    )
    vc = torch.randn_like(kc)
    cu_q = torch.arange(0, shape.batch + 1, dtype=torch.int32, device="cuda")
    kv_lens = torch.full(
        (shape.batch,), shape.seqlen_k, dtype=torch.int32, device="cuda"
    )
    block_table = torch.randint(
        0, pool, (shape.batch, num_blks), dtype=torch.int32, device="cuda"
    )
    scale = shape.head_size**-0.5

    # ALiBi slopes: one per query head, random small positive values
    alibi_slopes = (
        torch.rand(shape.num_query_heads, dtype=torch.float32, device="cuda") * 0.1
        if use_alibi
        else None
    )
    # QQ-bias: (num_query_heads, total_q, total_q) — for decode total_q == batch
    qq_bias = (
        torch.zeros(
            shape.num_query_heads,
            shape.batch,
            shape.batch,
            dtype=torch.float32,
            device="cuda",
        )
        if use_qq_bias
        else None
    )

    return dict(
        q=q,
        kc=kc,
        vc=vc,
        cu_q=cu_q,
        kv_lens=kv_lens,
        block_table=block_table,
        scale=scale,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
    )


def _run_triton(
    shape: DecodeShape, data: dict, *, warmup: int, iters: int
) -> Optional[float]:
    """Time AITER Triton unified_attention. Returns ms or None on failure."""
    from rocke.runtime import synchronize_and_release, time_launches
    import torch

    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention as tri  # type: ignore
    except ImportError:
        return None

    hip_stream = _bench_stream_handle()
    out = torch.empty_like(data["q"])

    def call_once():
        tri(
            q=data["q"],
            k=data["kc"],
            v=data["vc"],
            out=out,
            cu_seqlens_q=data["cu_q"],
            seqused_k=data["kv_lens"],
            max_seqlen_q=shape.seqlen_q,
            max_seqlen_k=shape.seqlen_k,
            softmax_scale=data["scale"],
            causal=True,
            window_size=(-1, -1),
            block_table=data["block_table"],
            softcap=data["softcap"],
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=data["alibi_slopes"],
            qq_bias=data["qq_bias"],
            sinks=None,
        )

    try:
        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return ms
    except Exception:
        return None


def _run_dsl(shape: DecodeShape, data: dict, num_sms: int, *, warmup: int, iters: int):
    """Time DSL run_unified_attention_torch for one num_sms value (gfx942).

    Uses :func:`~dispatch.attention.dispatch_attention` to select the
    registered kernel candidate (2d-tiled or 3d split-KV) for this shape,
    then exercises the same production path as the provider.

    Returns (ms, path_name) or (None, None) on failure.
    """
    from rocke.runtime import synchronize_and_release, time_launches
    import torch

    try:
        from dispatch.attention import AttentionRequest, dispatch_attention
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch  # type: ignore
    except ImportError:
        return None, None

    hip_stream = _bench_stream_handle()
    out = torch.empty_like(data["q"])

    try:
        req = AttentionRequest(
            batch=shape.batch,
            nhead_q=shape.num_query_heads,
            nhead_k=shape.num_kv_heads,
            seqlen_q=shape.seqlen_q,
            seqlen_k=shape.seqlen_k,
            hdim_q=shape.head_size,
            hdim_v=shape.head_size,
            arch=ARCH,
            dtype=shape.dtype,
            kv_block_size=shape.block_size,
            num_sms=num_sms,
        )
        result = dispatch_attention(req)
        path = result.spec.path  # "2d" or "3d"
        run_backend = "tiled" if path == "2d" else path

        prob = UnifiedAttentionProblem(
            total_q=shape.batch * shape.seqlen_q,
            num_seqs=shape.batch,
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            head_size=shape.head_size,
            block_size=shape.block_size,
            max_seqlen_q=shape.seqlen_q,
            max_seqlen_k=shape.seqlen_k,
            dtype=shape.dtype,
            softcap=data["softcap"],
            use_alibi=data["alibi_slopes"] is not None,
            use_qq_bias=data["qq_bias"] is not None,
            num_sms=num_sms,
        )

        def call_once():
            run_unified_attention_torch(
                problem=prob,
                q=data["q"],
                k=data["kc"],
                v=data["vc"],
                out=out,
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_table"],
                softcap=data["softcap"],
                alibi_slopes=data["alibi_slopes"],
                qq_bias=data["qq_bias"],
                backend=run_backend,
                stream=hip_stream,
            )

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return ms, path
    except Exception:
        return None, None


def _gm(vals: List[float]) -> float:
    vals = [v for v in vals if v and v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Decode attention benchmark: DSL split-KV vs AITER Triton (gfx942)."
    )
    ap.add_argument(
        "--shapes",
        nargs="+",
        type=Path,
        required=True,
        help="One or more shapes JSON files.",
    )
    ap.add_argument(
        "--num-sms-sweep",
        nargs="+",
        type=int,
        default=[30, 60, 80, 120],
        metavar="N",
        help="num_sms values to sweep (default: 30 60 80 120).",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument(
        "--cap-blocks",
        type=int,
        default=65536,
        help="Max paged-KV pool size (blocks). Large values keep KV in HBM.",
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Process only first N shapes."
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write per-shape results to JSON.",
    )
    # Decode-bias options
    ap.add_argument(
        "--softcap",
        type=float,
        default=0.0,
        help="Softcap value (0.0 = disabled). Supported by tiled path on gfx942.",
    )
    ap.add_argument(
        "--alibi",
        action="store_true",
        default=False,
        help="Enable ALiBi slopes. Forces scalar fallback (tiled path unsupported).",
    )
    ap.add_argument(
        "--qq-bias",
        action="store_true",
        default=False,
        help="Enable QQ-bias. Forces scalar fallback (tiled path unsupported).",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1

    shapes = load_decode_shapes(args.shapes)
    if args.limit is not None:
        shapes = shapes[: args.limit]

    bias_tag = (
        "+".join(
            filter(
                None,
                [
                    f"softcap={args.softcap}" if args.softcap else "",
                    "alibi" if args.alibi else "",
                    "qq_bias" if args.qq_bias else "",
                ],
            )
        )
        or "none"
    )

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"arch   : {ARCH}")
    print(f"shapes : {len(shapes)}")
    print(f"num_sms sweep: {args.num_sms_sweep}")
    print(f"bias   : {bias_tag}")
    print()

    header = (
        f"{'label':<22}  {'triton_us':>10}  "
        + "  ".join(f"sms{s:>4}" for s in args.num_sms_sweep)
        + f"  {'best_sms':>8}  {'best_spd':>9}  path"
    )
    print(header)
    print("-" * len(header))

    results: List[Dict] = []
    speedups: List[float] = []

    for i, shape in enumerate(shapes, 1):
        tag = f"[{i}/{len(shapes)}]"
        try:
            data = _make_inputs(
                shape,
                cap_blocks=args.cap_blocks,
                softcap=args.softcap,
                use_alibi=args.alibi,
                use_qq_bias=args.qq_bias,
            )
        except Exception as exc:
            print(f"{tag} {shape.label:<20}  INPUT ERR: {exc!r}")
            continue

        tri_ms = _run_triton(shape, data, warmup=args.warmup, iters=args.iterations)
        tri_us = tri_ms * 1000 if tri_ms else float("nan")

        dsl_results: Dict[int, Dict] = {}
        best_sms: Optional[int] = None
        best_ms: Optional[float] = None
        best_path: str = "n/a"

        for sms in args.num_sms_sweep:
            ms, path = _run_dsl(
                shape, data, sms, warmup=args.warmup, iters=args.iterations
            )
            if ms is not None:
                dsl_results[sms] = {"ms": ms, "path": path}
                if best_ms is None or ms < best_ms:
                    best_ms = ms
                    best_sms = sms
                    best_path = path or "n/a"
            else:
                dsl_results[sms] = {"ms": None, "path": None}

        best_spd = (tri_ms / best_ms) if (tri_ms and best_ms) else float("nan")
        if math.isfinite(best_spd):
            speedups.append(best_spd)

        sms_cols = "  ".join(
            (
                f"{dsl_results[s]['ms'] * 1000:>8.1f}u"
                if dsl_results[s]["ms"]
                else f"{'ERR':>9}"
            )
            for s in args.num_sms_sweep
        )
        print(
            f"{shape.label:<22}  {tri_us:>10.1f}  {sms_cols}"
            f"  {best_sms or '-':>8}  {best_spd:>8.3f}x  {best_path}"
        )

        results.append(
            {
                "label": shape.label,
                "signature": shape.signature,
                "bias": bias_tag,
                "arch": ARCH,
                "batch": shape.batch,
                "seqlen_q": shape.seqlen_q,
                "seqlen_k": shape.seqlen_k,
                "num_query_heads": shape.num_query_heads,
                "num_kv_heads": shape.num_kv_heads,
                "head_size": shape.head_size,
                "block_size": shape.block_size,
                "dtype": shape.dtype,
                "triton_ms": tri_ms,
                "dsl": {str(sms): dsl_results[sms] for sms in args.num_sms_sweep},
                "best_sms": best_sms,
                "best_ms": best_ms,
                "best_speedup": best_spd if math.isfinite(best_spd) else None,
                "best_path": best_path,
            }
        )

    print()
    print(
        f"geomean speedup (DSL best vs Triton): {_gm(speedups):.3f}x  "
        f"({sum(1 for s in speedups if s > 1)}/{len(speedups)} wins)"
    )

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.output_json}  ({len(results)} shapes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
