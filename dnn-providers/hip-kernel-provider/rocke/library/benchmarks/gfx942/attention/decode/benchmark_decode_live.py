# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic decode attention benchmark: DSL split-KV 3D vs AITER Triton (gfx942).

gfx942 sibling of
``benchmarks.gfx950.attention.decode.benchmark_decode_live``.

Differences vs gfx950:
  * ``num_cus`` sweep defaults to ``30 60 80 120`` (gfx942 has 228 CUs).
  * No FP8 KV-cache (not supported on gfx942).
  * ``dtype`` defaults to ``bf16``; ``fp16`` is also supported.

Run::

    python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \\
        --shapes library/benchmarks/gfx942/attention/decode/qwen3_30b_a3b_shapes.json

    # or with a custom sweep:
    python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \\
        --shapes shapes.json \\
        --num-cus-sweep 30 60 120 \\
        --output-json /tmp/decode_gfx942.json

    # --flydsl requires FLYDSL_PATH env var set to the ROCm/FlyDSL repo root:
    FLYDSL_PATH=/path/to/FlyDSL \\
    python -m benchmarks.gfx942.attention.decode.benchmark_decode_live \\
        --shapes shapes.json --flydsl
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
    use_sinks: bool = False
    sliding_window: int = 0

    @property
    def signature(self) -> str:
        sig = (
            f"b{self.batch}_sq{self.seqlen_q}_sk{self.seqlen_k}"
            f"_nhq{self.num_query_heads}_nhk{self.num_kv_heads}"
            f"_hd{self.head_size}_bs{self.block_size}_{self.dtype}"
        )
        if self.use_sinks:
            sig += "_sinks"
        if self.sliding_window:
            sig += f"_sw{self.sliding_window}"
        return sig


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
                    use_sinks=bool(merged.get("use_sinks", False)),
                    sliding_window=int(merged.get("sliding_window", 0)),
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
    sinks = (
        torch.randn(shape.num_query_heads, dtype=dtype, device="cuda") * 0.1
        if shape.use_sinks
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
        sinks=sinks,
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
    window_size = (shape.sliding_window - 1, 0) if shape.sliding_window else (-1, -1)

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
            window_size=window_size,
            block_table=data["block_table"],
            softcap=data["softcap"],
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=data["alibi_slopes"],
            qq_bias=data["qq_bias"],
            sinks=data["sinks"],
        )

    try:
        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return ms
    except Exception:
        return None


def _run_dsl(shape: DecodeShape, data: dict, num_cus: int, *, warmup: int, iters: int):
    """Time DSL run_unified_attention_torch for one num_cus value (gfx942).

    Uses :func:`~dispatch.attention.dispatch_attention` to select the
    registered kernel candidate (2d-tiled or 3d split-KV) for this shape,
    then exercises the same production path as the provider.

    Returns (ms, path_name, kernel_name) or (None, None, None) on failure.
    """
    from rocke.runtime import synchronize_and_release, time_launches
    import torch

    try:
        from dispatch.attention import AttentionRequest, dispatch_attention
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch  # type: ignore
        from kernels.common.attention_unified import _resolve_attention_arch
    except ImportError:
        return None, None, None

    hip_stream = _bench_stream_handle()
    out = torch.empty_like(data["q"])

    try:
        arch = _resolve_attention_arch()
        req = AttentionRequest(
            batch=shape.batch,
            nhead_q=shape.num_query_heads,
            nhead_k=shape.num_kv_heads,
            seqlen_q=shape.seqlen_q,
            seqlen_k=shape.seqlen_k,
            hdim_q=shape.head_size,
            hdim_v=shape.head_size,
            arch=arch,
            dtype=shape.dtype,
            kv_block_size=shape.block_size,
            num_cus=num_cus,
            use_sinks=shape.use_sinks,
            sliding_window=shape.sliding_window,
        )
        result = dispatch_attention(req)
        path = result.spec.path  # "2d" or "3d"
        kernel_name = result.spec.kernel_name()
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
            use_sinks=data["sinks"] is not None,
            sliding_window=shape.sliding_window,
            num_cus=num_cus,
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
                sinks=data["sinks"],
                alibi_slopes=data["alibi_slopes"],
                qq_bias=data["qq_bias"],
                backend=run_backend,
                stream=hip_stream,
            )

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return ms, path, kernel_name
    except Exception:
        return None, None, None


def _run_aoTriton(
    shape: DecodeShape, data: dict, *, warmup: int, iters: int
) -> "Optional[float]":
    """Time AOTriton flash SDPA for decode. Returns ms or None on failure.

    Reconstructs dense [B, H, S_k, D] KV from the paged cache outside the
    timed region. Returns None (skipped) when any operand it cannot express
    is active (softcap, alibi_slopes, qq_bias, attention sinks, or a sliding
    window) because scaled_dot_product_attention does not support them.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from rocke.runtime import synchronize_and_release, time_launches

    try:
        nrep = shape.num_query_heads // shape.num_kv_heads

        ks, vs = [], []
        for bi in range(shape.batch):
            bt = data["block_table"][bi]
            ks.append(
                data["kc"][bt].reshape(-1, shape.num_kv_heads, shape.head_size)[
                    : shape.seqlen_k
                ]
            )
            vs.append(
                data["vc"][bt].reshape(-1, shape.num_kv_heads, shape.head_size)[
                    : shape.seqlen_k
                ]
            )

        kh = torch.stack([k.transpose(0, 1) for k in ks], dim=0)  # [B, nhk, S_k, D]
        vh = torch.stack([v.transpose(0, 1) for v in vs], dim=0)
        kh = kh.repeat_interleave(nrep, dim=1).contiguous()
        vh = vh.repeat_interleave(nrep, dim=1).contiguous()

        # q: [B, nhq, D] -> [B, nhq, 1, D]
        qh = data["q"].unsqueeze(2).contiguous()

        scale = data["scale"]
        alibi_slopes = data["alibi_slopes"]
        softcap = data["softcap"]

        # scaled_dot_product_attention does not support alibi, qq_bias,
        # softcap, attention sinks, or a sliding window.
        unsupported = (
            alibi_slopes is not None
            or data["qq_bias"] is not None
            or softcap
            or shape.use_sinks
            or shape.sliding_window > 0
        )
        if unsupported:
            return None

        # Probe eligibility.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            _ = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=False, scale=scale
            )
        torch.cuda.synchronize()

        hip_stream = _bench_stream_handle()

        def call_once():
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=False, scale=scale
                )

        ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
        synchronize_and_release(hip_stream)
        return ms
    except Exception:
        return None


def _gm(vals: List[float]) -> float:
    vals = [v for v in vals if v and v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# ---------------------------------------------------------------------------
# FlyDSL helpers
# ---------------------------------------------------------------------------

_FLYDSL_FUNC = None
_FLYDSL_KERNELS: dict = {}
_ROCKE_KERNELS: dict = {}


def _import_flydsl():
    global _FLYDSL_FUNC, _FLYDSL_KERNELS, _ROCKE_KERNELS
    if _FLYDSL_FUNC is not None:
        return
    import os

    flydsl_path = os.environ.get("FLYDSL_PATH")
    if not flydsl_path:
        raise RuntimeError(
            "FLYDSL_PATH env var must be set to the ROCm/FlyDSL repo root"
        )
    sys.path.insert(0, flydsl_path)
    _rocke = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "kernels" or k.startswith("kernels.")
    }
    try:
        import kernels.attention.flash_attn_generic  # type: ignore  # noqa: F401

        try:
            import kernels.attention.flash_attn_gfx950  # type: ignore  # noqa: F401
        except ImportError:
            pass
        from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # type: ignore

        _FLYDSL_FUNC = flydsl_flash_attn_func
        _FLYDSL_KERNELS = {
            k: v
            for k, v in sys.modules.items()
            if k == "kernels" or k.startswith("kernels.")
        }
    finally:
        sys.path.remove(flydsl_path)
        for k in [k for k in sys.modules if k == "kernels" or k.startswith("kernels.")]:
            del sys.modules[k]
        sys.modules.update(_rocke)
        _ROCKE_KERNELS = {
            k: v
            for k, v in sys.modules.items()
            if k == "kernels" or k.startswith("kernels.")
        }


def _flydsl_swap(flydsl: bool) -> None:
    for k in [k for k in sys.modules if k == "kernels" or k.startswith("kernels.")]:
        del sys.modules[k]
    sys.modules.update(_FLYDSL_KERNELS if flydsl else _ROCKE_KERNELS)


_FLYDSL_PAGED_PAGE_SIZE = 64


def _flydsl_supported(shape: DecodeShape) -> tuple[bool, str]:
    if shape.head_size not in (64, 128):
        return False, f"head_size={shape.head_size} (need 64 or 128)"
    if shape.dtype not in ("bf16", "fp16"):
        return False, f"dtype={shape.dtype} (need bf16 or fp16)"
    if shape.use_sinks:
        return False, "sinks unsupported"
    if shape.sliding_window:
        return False, "sliding_window unsupported"
    return True, ""


def _run_flydsl(shape: DecodeShape, data: dict, *, warmup: int, iters: int):
    """Time FlyDSL flash attention for decode.

    FlyDSL decode uses 4D Q [B, Sq, H, D] (no cu_seqlens_q).

    block_size == 64: paged path — FlyDSL reads KV natively via block_table.
    block_size != 64: dense path — KV reconstructed as [B, Skv, H, D] outside
        the timed loop, then passed without block_table.

    Returns (ms, kv_path) or (None, None) on failure.
    """
    import torch
    from rocke.runtime import synchronize_and_release, time_launches

    _import_flydsl()
    hip_stream = _bench_stream_handle()

    # Decode q from _make_inputs: [batch, num_query_heads, head_size].
    # FlyDSL dense/paged path expects 4D [B, Sq, H, D].
    q = data["q"].unsqueeze(1)  # [B, 1, H, D]
    kv_lens = data["kv_lens"]
    out = torch.empty_like(q)

    if shape.block_size == _FLYDSL_PAGED_PAGE_SIZE:
        kv_path = "paged"

        def call_once():
            _FLYDSL_FUNC(
                q,
                data["kc"],
                data["vc"],
                causal=True,
                block_table=data["block_table"],
                seqlen_k=kv_lens,
                out=out,
            )

    else:
        # Reconstruct dense KV [B, Skv, nhk, D] outside the timed region.
        kv_path = "dense"
        block_size = shape.block_size
        nheads_k = shape.num_kv_heads
        head_dim = shape.head_size
        seqlen_k = shape.seqlen_k
        k_dense = torch.empty(
            shape.batch, seqlen_k, nheads_k, head_dim, dtype=q.dtype, device=q.device
        )
        v_dense = torch.empty_like(k_dense)
        for bi in range(shape.batch):
            bt = data["block_table"][bi]
            k_dense[bi] = data["kc"][bt].reshape(-1, nheads_k, head_dim)[:seqlen_k]
            v_dense[bi] = data["vc"][bt].reshape(-1, nheads_k, head_dim)[:seqlen_k]

        def call_once():
            _FLYDSL_FUNC(
                q,
                k_dense,
                v_dense,
                causal=True,
                num_kv_heads=nheads_k,
                out=out,
            )

    try:
        _flydsl_swap(flydsl=True)
        try:
            ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
            synchronize_and_release(hip_stream)
        finally:
            _flydsl_swap(flydsl=False)
        return ms, kv_path
    except Exception:
        traceback.print_exc()
        return None, None


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
        "--num-cus-sweep",
        nargs="+",
        type=int,
        default=[30, 60, 80, 120],
        metavar="N",
        help="num_cus values to sweep (default: 30 60 80 120).",
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
    ap.add_argument(
        "--flydsl",
        action="store_true",
        help="Also benchmark FlyDSL flash attention (set FLYDSL_PATH to ROCm/FlyDSL repo root)",
    )
    args = ap.parse_args()

    import os

    if args.flydsl and not os.environ.get("FLYDSL_PATH"):
        ap.error(
            "--flydsl requires the FLYDSL_PATH env var to be set to the ROCm/FlyDSL repo root"
        )

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
    print(f"num_cus sweep: {args.num_cus_sweep}")
    print(f"bias   : {bias_tag}")
    print()

    fly_col = f"  {'fly_us':>10}" if args.flydsl else ""
    header = (
        f"{'label':<22}  {'triton_us':>10}  {'aot_us':>10}{fly_col}  "
        + "  ".join(f"CUs{s:>4}" for s in args.num_cus_sweep)
        + f"  {'best_cus':>8}  {'best_spd':>9}  path  kernel"
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

        aot_ms = _run_aoTriton(shape, data, warmup=args.warmup, iters=args.iterations)
        aot_us = aot_ms * 1000 if aot_ms else float("nan")

        fly_ms = None
        fly_kv_path = None
        fly_skip_reason = None
        fly_error = None
        if args.flydsl:
            fly_ok, fly_skip_reason = _flydsl_supported(shape)
            if fly_ok:
                fly_ms, fly_kv_path = _run_flydsl(
                    shape, data, warmup=args.warmup, iters=args.iterations
                )
                if fly_ms is None:
                    fly_error = "runtime error (see stderr)"

        dsl_results: Dict[int, Dict] = {}
        best_cus: Optional[int] = None
        best_ms: Optional[float] = None
        best_path: str = "n/a"
        best_kernel_name: str = "n/a"

        for cus in args.num_cus_sweep:
            ms, path, kname = _run_dsl(
                shape, data, cus, warmup=args.warmup, iters=args.iterations
            )
            if ms is not None:
                dsl_results[cus] = {"ms": ms, "path": path}
                if best_ms is None or ms < best_ms:
                    best_ms = ms
                    best_cus = cus
                    best_path = path or "n/a"
                    best_kernel_name = kname or "n/a"
            else:
                dsl_results[cus] = {"ms": None, "path": None}

        best_spd = (tri_ms / best_ms) if (tri_ms and best_ms) else float("nan")
        best_spd_aot = (aot_ms / best_ms) if (aot_ms and best_ms) else float("nan")
        if math.isfinite(best_spd):
            speedups.append(best_spd)

        cu_cols = "  ".join(
            (
                f"{dsl_results[s]['ms'] * 1000:>8.1f}u"
                if dsl_results[s]["ms"]
                else f"{'ERR':>9}"
            )
            for s in args.num_cus_sweep
        )
        aot_spd_str = (
            f"{best_spd_aot:>8.3f}x" if math.isfinite(best_spd_aot) else f"{'N/A':>9}"
        )
        fly_col_val = (
            f"  {fly_ms * 1000:>10.1f}"
            if fly_ms is not None
            else (
                (f"  {'SKIP':>10}" if fly_skip_reason else f"  {'ERR':>10}")
                if args.flydsl
                else ""
            )
        )
        print(
            f"{shape.label:<22}  {tri_us:>10.1f}  {aot_us:>10.1f}{fly_col_val}  {cu_cols}"
            f"  {best_cus or '-':>8}  {best_spd:>8.3f}x(tri)  {aot_spd_str}(aot)"
            f"  {best_path}  {best_kernel_name}"
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
                "use_sinks": shape.use_sinks,
                "sliding_window": shape.sliding_window,
                "triton_ms": tri_ms,
                "aoTriton_ms": aot_ms,
                "dsl": {str(cus): dsl_results[cus] for cus in args.num_cus_sweep},
                "best_cus": best_cus,
                "best_ms": best_ms,
                "best_speedup_vs_triton": best_spd if math.isfinite(best_spd) else None,
                "best_speedup_vs_aoTriton": (
                    best_spd_aot if math.isfinite(best_spd_aot) else None
                ),
                "best_path": best_path,
                "flydsl_ms": fly_ms,
                "flydsl_kv_path": fly_kv_path,
                "flydsl_skip_reason": fly_skip_reason,
                "flydsl_error": fly_error,
                "best_speedup_vs_flydsl": (
                    fly_ms / best_ms
                    if (fly_ms is not None and best_ms is not None and best_ms > 0)
                    else None
                ),
            }
        )

    aot_spds = [
        r["best_speedup_vs_aoTriton"]
        for r in results
        if r.get("best_speedup_vs_aoTriton") is not None
    ]
    print()
    print(
        f"geomean speedup (DSL best vs Triton):   {_gm(speedups):.3f}x  "
        f"({sum(1 for s in speedups if s > 1)}/{len(speedups)} wins)"
    )
    if aot_spds:
        print(
            f"geomean speedup (DSL best vs AOTriton): {_gm(aot_spds):.3f}x  "
            f"(n={len(aot_spds)})"
        )
    if args.flydsl:
        fly_ran = [r for r in results if r.get("flydsl_ms") is not None]
        fly_vs_tri = [
            r["flydsl_ms"] / r["triton_ms"]
            for r in fly_ran
            if r.get("triton_ms") and r["triton_ms"] > 0
        ]
        fly_vs_ck = [
            r["best_speedup_vs_flydsl"]
            for r in fly_ran
            if r.get("best_speedup_vs_flydsl") is not None
        ]
        print(f"\nFlyDSL summary ({len(fly_ran)}/{len(results)} shapes ran):")
        if fly_vs_tri:
            wins = sum(1 for x in fly_vs_tri if x > 1)
            print(
                f"  fly vs Triton:      geomean={_gm(fly_vs_tri):.3f}x  wins={wins}/{len(fly_vs_tri)}"
            )
        if fly_vs_ck:
            wins = sum(1 for x in fly_vs_ck if x > 1)
            print(
                f"  rocke vs FlyDSL:    geomean={_gm(fly_vs_ck):.3f}x  wins={wins}/{len(fly_vs_ck)}"
            )

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.output_json}  ({len(results)} shapes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
