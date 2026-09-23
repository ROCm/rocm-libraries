# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA prefill live benchmark — rocke vs aiter comparison.

Three modes:

``--mode resolve`` (default)
    Shape -> ``MLARequest`` -> ``dispatch_mla`` -> spec, grid, block, kernel name.
    CPU only; no GPU, no comgr, no timing.

``--mode build``
    Resolve + build each selection's IR (still CPU-only). Catches lowering
    failures before touching a device.

``--mode time``
    Full live timing: builds the rocke MLA kernel, allocates torch tensors,
    runs warmup + timed iterations with ``time_launches``, and compares against
    the aiter ``mla_prefill_fwd`` absorb-mode baseline (when available).

    FlyDSL note: FlyDSL only ships an MLA *decode* kernel
    (``kernels/attention/mla_fwd_decode.py``). There is no FlyDSL MLA prefill
    implementation as of September 2026, so no ``--flydsl`` flag is provided.
    When FlyDSL adds prefill MLA support, wire it in the same way as
    ``benchmark_prefill2d_live.py`` does.

    aiter baseline: ``aiter.mla.mla_prefill_fwd`` (absorb mode, gfx942 ASM).
    It operates in *latent space*: its query is ``[total_q, H, r_kv+d_rope]``
    (576-wide) while the rocke kernel takes a post-W_UQ query
    ``[total_q, H, d_nope+d_rope]`` (192-wide). Both compute the same MLA
    math; the difference is where the W_UQ / W_UK expansions happen.
    aiter only supports ``nhead in {16, 128}`` on gfx942 (hard constraint from
    its ASM kernel); shapes with other head counts skip the aiter lane.

Run, from ``rocke/library``:

    export AITER_PATH=<path/to/aiter>
    PYTHONPATH="python:${AITER_PATH}" \\
      python -m benchmarks.gfx942.attention.prefill.benchmark_mla_prefill_live \\
        --mode time \\
        --shapes benchmarks/gfx942/attention/prefill/mla_prefill_shapes.json \\
        --limit 8

    # resolve/build only (no GPU):
    python -m ... --mode resolve
    python -m ... --mode build --regime chunked --limit 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from benchmarks.common.mla_shapes import MLAShape, filter_shapes, load_mla_shapes

_ARCH = "gfx942"
_DEFAULT_SHAPES = Path(__file__).resolve().parent / "mla_prefill_shapes.json"

# aiter nhead constraint for the ASM prefill kernel on gfx942
_AITER_SUPPORTED_NHEADS = frozenset({16, 128})

# aiter's ASM prefill kernel is only numerically correct at page_size == 1 (see
# _make_aiter_inputs). Independent of rocke's block_size, which is pinned to 16.
_AITER_PAGE_SIZE = 1


# ---------------------------------------------------------------------------
# Resolve / build path (CPU-only)
# ---------------------------------------------------------------------------


def _resolve(shape: MLAShape, *, arch: str, build: bool) -> dict:
    """Resolve one shape; never raise, so one bad shape does not hide the rest."""
    from dispatch.mla import dispatch_mla

    row: dict = {"shape": shape.signature, "regime": shape.regime}
    try:
        result = dispatch_mla(shape.to_request(arch=arch))
    except ValueError as exc:
        row["status"] = "unsupported"
        row["detail"] = str(exc)
        return row
    row.update(
        status="ok",
        candidate=result.candidate.name,
        kernel=result.spec.fwd_kernel_name(),
        grid=result.grid,
        block=result.block,
        nargs=len(result.signature),
    )
    if build:
        try:
            kernel = result.candidate.built(result.spec, arch)
            row["built"] = kernel.name
        except Exception as exc:  # pragma: no cover - diagnostic  # noqa: BLE001
            row["status"] = "build_failed"
            row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def _report_resolve(rows: list[dict]) -> int:
    width = max((len(r["shape"]) for r in rows), default=0)
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['shape']:<{width}}  {row['regime']:<11}  "
                f"grid={row['grid']} block={row['block']}  {row['kernel']}"
            )
        else:
            print(f"{row['shape']:<{width}}  {row['status'].upper()}: {row['detail']}")
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nSUMMARY {ok}/{len(rows)} shapes resolved on {_ARCH}")
    return 0 if ok == len(rows) else 1


# ---------------------------------------------------------------------------
# Timing path — tensor allocation
# ---------------------------------------------------------------------------


def _bench_stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _make_rocke_inputs(shape: MLAShape, *, seed: int = 0) -> dict[str, Any]:
    """Allocate torch tensors matching the rocke MLA kernel ABI.

    rocke takes *projected* query ``[total_q, H, d_nope+d_rope]`` plus
    separate ``c_kv``, ``k_rope``, and ``w_uk`` tensors. W_UQ projection
    happens on the host before the kernel call (or in a fused pre-kernel that
    is out of scope here); the timing region covers only the attention kernel.
    """
    import torch

    g = shape.geometry
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    page = shape.block_size
    total_q = shape.total_q
    max_k = shape.seqlen_k
    num_pages = (max_k + page - 1) // page * shape.batch
    max_bt_cols = (max_k + page - 1) // page

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16

    def randn(*s):
        return torch.randn(*s, dtype=dtype, device="cuda", generator=rng)

    q = randn(total_q, shape.num_query_heads, g.qk_nope_dim + g.qk_rope_dim)
    c_kv = randn(num_pages, page, g.kv_lora_rank)
    k_rope = randn(num_pages, page, g.qk_rope_dim)
    # W_UK: [H, r_kv, d_nope + d_v]
    w_uk = randn(shape.num_query_heads, g.kv_lora_rank, g.qk_nope_dim + g.v_head_dim)

    # packed varlen offsets
    seqlen_q = shape.seqlen_q
    seqlen_k = shape.seqlen_k
    batch = shape.batch
    cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_q
    cu_k = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_k

    # block table: each sequence owns its own pages in order
    block_table = torch.zeros(batch, max_bt_cols, dtype=torch.int32, device="cuda")
    for bi in range(batch):
        for bk in range(max_bt_cols):
            block_table[bi, bk] = bi * max_bt_cols + bk

    scale = float(1.0 / math.sqrt(g.qk_nope_dim + g.qk_rope_dim))

    return {
        "q": q,
        "c_kv": c_kv,
        "k_rope": k_rope,
        "w_uk": w_uk,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "block_table": block_table,
        "scale": scale,
        "total_q": total_q,
        "max_k": seqlen_k,
    }


def _make_aiter_inputs(shape: MLAShape, *, seed: int = 0) -> dict[str, Any]:
    """Allocate tensors for aiter ``mla_prefill_fwd`` (absorb mode).

    aiter's absorb-mode kernel takes ``q [total_q, H, r_kv+d_rope]`` (the
    query projected into the combined latent+rope space) and a packed
    ``kv_buffer [num_page, page_size, 1, r_kv+d_rope]``. This is a different
    decomposition from rocke's: both compute the same math, but the weight
    matrix absorption happens before the aiter call.

    aiter uses FlashInfer-style indirections:
        qo_indptr  [batch+1] int32 — prefix sums of per-seq query lengths
        kv_indptr  [batch+1] int32 — prefix sums of per-seq KV page counts
        kv_indices [total_kv_pages] int32 — physical page indices
        kv_last_page_lens [batch] int32 — valid tokens in each last page
    """
    import torch

    g = shape.geometry
    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    # aiter's ASM prefill kernel only produces correct results at page_size == 1.
    # Verified in isolation against aiter's own oracle
    # (aiter_meta/csrc/cpp_itfs/mla/asm_mla_decode_fwd_test.py::test_absorb_prefill,
    # which itself pins block_size=1): page=1 -> rel_err 1.9e-03, page=16 -> NaN
    # with 65796 output elements left unwritten. rocke's own block_size is pinned
    # to block_k=16 by its spec, so the two paging schemes must NOT be shared.
    page = _AITER_PAGE_SIZE
    batch = shape.batch
    seqlen_q = shape.seqlen_q
    seqlen_k = shape.seqlen_k
    total_q = shape.total_q

    # aiter absorb qk_head_dim = r_kv + d_rope (not d_nope + d_rope)
    aiter_qk = g.kv_lora_rank + g.qk_rope_dim  # 576 for DeepSeek

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16

    def randn(*s):
        return torch.randn(*s, dtype=dtype, device="cuda", generator=rng)

    # query in latent space
    q = randn(total_q, shape.num_query_heads, aiter_qk)

    # packed KV buffer: [num_page, page_size, 1, r_kv + d_rope]
    pages_per_seq = (seqlen_k + page - 1) // page
    total_pages = batch * pages_per_seq
    kv_buffer = randn(total_pages, page, 1, aiter_qk)

    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seqlen_q
    kv_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.full(
        (batch,),
        seqlen_k - (pages_per_seq - 1) * page,
        dtype=torch.int32,
        device="cuda",
    )
    # Absorb mode rebinds v_head_dim := kv_lora_rank (512), NOT the 128-wide
    # post-W_UV output rocke produces. Allocating g.v_head_dim here under-sizes
    # the buffer the kernel writes.
    out = torch.zeros(
        total_q, shape.num_query_heads, g.kv_lora_rank, dtype=dtype, device="cuda"
    )
    sm_scale = float(1.0 / math.sqrt(aiter_qk))

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "out": out,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "max_seqlen_q": seqlen_q,
        "sm_scale": sm_scale,
    }


# ---------------------------------------------------------------------------
# aiter baseline runner
# ---------------------------------------------------------------------------


def _aiter_supported(shape: MLAShape) -> tuple[bool, str]:
    """Return (supported, reason) for aiter mla_prefill_fwd on gfx942."""
    if shape.dtype not in ("bf16",):
        return False, f"dtype={shape.dtype!r} not supported (bf16 only on gfx942)"
    if shape.num_query_heads not in _AITER_SUPPORTED_NHEADS:
        return (
            False,
            (
                f"nhead={shape.num_query_heads} not in aiter gfx942 supported set "
                f"{sorted(_AITER_SUPPORTED_NHEADS)}"
            ),
        )
    return True, ""


def _import_aiter_mla():
    """Import ``aiter.mla.mla_prefill_fwd``; raise descriptively on failure."""
    try:
        import aiter.mla as _aiter_mla  # type: ignore

        return _aiter_mla.mla_prefill_fwd
    except ImportError as exc:
        raise RuntimeError(
            "aiter not importable. Set AITER_PATH and add it to PYTHONPATH: "
            "PYTHONPATH=python:${AITER_PATH} python -m ..."
        ) from exc


def _run_aiter_live(
    shape: MLAShape, data: dict, *, warmup: int, iters: int
) -> tuple[Any, float]:
    """Time aiter ``mla_prefill_fwd`` and return ``(out, ms)``."""
    from rocke.runtime import synchronize_and_release, time_launches

    mla_prefill_fwd = _import_aiter_mla()
    hip_stream = _bench_stream_handle()

    q = data["q"]
    kv_buffer = data["kv_buffer"]
    out = data["out"].zero_()
    qo_indptr = data["qo_indptr"]
    kv_indptr = data["kv_indptr"]
    kv_indices = data["kv_indices"]
    kv_last_page_lens = data["kv_last_page_lens"]
    max_seqlen_q = data["max_seqlen_q"]
    sm_scale = data["sm_scale"]

    def call_once():
        mla_prefill_fwd(
            q,
            kv_buffer,
            out,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q,
            sm_scale,
        )

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out.clone(), ms


# ---------------------------------------------------------------------------
# rocke MLA kernel runner
# ---------------------------------------------------------------------------


def _build_rocke_kernel(shape: MLAShape, *, arch: str):
    """Build (compile to HSACO) the rocke MLA prefill kernel for ``shape``.

    Returns ``(DispatchResult, Artifact)`` where ``Artifact`` has ``.hsaco``
    and ``.kernel_name`` attributes.
    """
    from dispatch.mla import dispatch_mla
    from rocke import compile_kernel

    req = shape.to_request(arch=arch)
    result = dispatch_mla(req)
    kernel_def = result.candidate.built(result.spec, arch)
    artifact = compile_kernel(kernel_def, arch=arch, capture_ir_text=False)
    return result, artifact


def _run_rocke_live(
    shape: MLAShape,
    data: dict,
    result,
    kernel_obj,
    *,
    warmup: int,
    iters: int,
) -> tuple[Any, float, str]:
    """Time the rocke MLA prefill kernel and return ``(out, ms, kernel_name)``."""
    import torch
    from rocke.runtime import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
        time_launches,
    )

    hip_stream = _bench_stream_handle()
    g = shape.geometry

    dtype = torch.bfloat16 if shape.dtype == "bf16" else torch.float16
    out = torch.zeros(
        shape.total_q, shape.num_query_heads, g.v_head_dim, dtype=dtype, device="cuda"
    )
    lse = torch.zeros(
        shape.total_q, shape.num_query_heads, dtype=torch.float32, device="cuda"
    )

    q = data["q"]
    c_kv = data["c_kv"]
    k_rope = data["k_rope"]
    w_uk = data["w_uk"]
    cu_q = data["cu_q"]
    cu_k = data["cu_k"]
    block_table = data["block_table"]
    scale = data["scale"]
    max_k = data["max_k"]
    n_seqs = shape.batch
    bt_stride = int(block_table.shape[1])

    launcher = KernelLauncher(
        hsaco=kernel_obj.hsaco,
        kernel_name=kernel_obj.kernel_name,
        signature=tuple(result.signature),
        cache_key=("mla_prefill_live_gfx942", shape.signature),
    )

    # pack_args expects a dict keyed by the signature argument names
    # (see mla_prefill_fwd_signature: out_ptr, lse_ptr, q_ptr, c_kv_ptr,
    #  k_rope_ptr, w_uk_ptr, cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    #  block_table_ptr, scale, num_seqs, block_table_stride, max_k)
    vals = {
        "out_ptr": out,
        "lse_ptr": lse,
        "q_ptr": q,
        "c_kv_ptr": c_kv,
        "k_rope_ptr": k_rope,
        "w_uk_ptr": w_uk,
        "cu_seqlens_q_ptr": cu_q,
        "cu_seqlens_k_ptr": cu_k,
        "block_table_ptr": block_table,
        "scale": scale,
        "num_seqs": n_seqs,
        "block_table_stride": bt_stride,
        "max_k": max_k,
    }

    cfg = LaunchConfig(grid=result.grid, block=result.block, stream=hip_stream)

    def call_once():
        launcher(vals, config=cfg)

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=hip_stream)
    synchronize_and_release(hip_stream)
    return out.clone(), ms, kernel_obj.kernel_name


# ---------------------------------------------------------------------------
# TFLOP count for MLA prefill (absorb mode, fused)
# ---------------------------------------------------------------------------


def _mla_prefill_flops(shape: MLAShape) -> float:
    """Approximate arithmetic intensity (TFLOP) for one MLA prefill shape.

    Counts the dominant GEMM work inside the kernel (score + softmax-weighted V
    accumulation). W_UQ and W_UK projections (done outside or fused) are excluded
    so the number is comparable across implementations with different fusion
    boundaries.
    """
    g = shape.geometry
    total_q = shape.total_q
    seqlen_k = shape.seqlen_k
    H = shape.num_query_heads
    # Score: total_q × seqlen_k × head_dim_qk (×2 for multiply-add)
    score_flops = 2.0 * total_q * seqlen_k * g.head_dim_qk * H
    # Value accumulation: total_q × seqlen_k × v_head_dim (×2)
    value_flops = 2.0 * total_q * seqlen_k * g.v_head_dim * H
    return (score_flops + value_flops) / 1e12


def _gm(vals: list[float]) -> float:
    vals = [v for v in vals if v > 0]
    return (
        math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", type=Path, default=_DEFAULT_SHAPES)
    parser.add_argument(
        "--mode", choices=("resolve", "build", "time"), default="resolve"
    )
    parser.add_argument("--regime", choices=("full_prompt", "chunked"), default=None)
    parser.add_argument("--model", default=None, help="substring match on model name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arch", default=_ARCH)
    parser.add_argument(
        "--list-shapes", action="store_true", help="print the shapes and exit"
    )
    # timing-mode args
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="write timing results to this JSON file (--mode time only)",
    )
    parser.add_argument(
        "--no-aiter",
        action="store_true",
        help="skip the aiter mla_prefill_fwd baseline even if available",
    )
    args = parser.parse_args(argv)

    if args.mode in ("resolve", "build"):
        shapes = filter_shapes(
            load_mla_shapes(args.shapes),
            regime=args.regime,
            model=args.model,
            limit=args.limit,
        )
        if not shapes:
            print("no shapes matched the filters", file=sys.stderr)
            return 1
        if args.list_shapes:
            for shape in shapes:
                print(f"{shape.signature}  regime={shape.regime}")
            return 0

        rows = [
            _resolve(shape, arch=args.arch, build=args.mode == "build")
            for shape in shapes
        ]
        return _report_resolve(rows)

    # --mode time
    import torch

    if not torch.cuda.is_available():
        print("no GPU available", file=sys.stderr)
        return 1

    shapes = filter_shapes(
        load_mla_shapes(args.shapes),
        regime=args.regime,
        model=args.model,
        limit=args.limit,
    )
    if not shapes:
        print("no shapes matched the filters", file=sys.stderr)
        return 1
    if args.list_shapes:
        for shape in shapes:
            print(f"{shape.signature}  regime={shape.regime}")
        return 0

    print(f"device:  {torch.cuda.get_device_name(0)}")
    print(f"arch:    {args.arch}")
    print(f"shapes:  {len(shapes)}")
    print(f"warmup:  {args.warmup}  iters: {args.iterations}")
    print(
        f"aiter:   {'disabled (--no-aiter)' if args.no_aiter else 'enabled (bf16, nhead in {16,128})'}"
    )
    print("flydsl:  no MLA prefill kernel in FlyDSL (decode-only as of Sep 2026)")

    results = []
    n_ok = 0
    n_aiter_ran = 0
    n_aiter_supported = 0

    for i, shape in enumerate(shapes, 1):
        tag = f"[{i}/{len(shapes)}] {shape.signature}"
        # TFLOP (not FLOP) -- _mla_prefill_flops already scales by 1e12.
        tflop = _mla_prefill_flops(shape)

        # -- build rocke kernel (once per shape)
        try:
            result, kernel_obj = _build_rocke_kernel(shape, arch=args.arch)
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  ROCKE BUILD FAIL: {exc!r}")
            traceback.print_exc()
            results.append(
                {
                    "signature": shape.signature,
                    "regime": shape.regime,
                    "status": "build_fail",
                    "detail": repr(exc),
                }
            )
            continue

        rocke_ms: float | None = None
        rocke_kname: str | None = None
        try:
            rocke_data = _make_rocke_inputs(shape, seed=args.seed)
            _, rocke_ms, rocke_kname = _run_rocke_live(
                shape,
                rocke_data,
                result,
                kernel_obj,
                warmup=args.warmup,
                iters=args.iterations,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{tag}  ROCKE TIME FAIL: {exc!r}")
            traceback.print_exc()

        rocke_tflops = (
            tflop / (rocke_ms * 1e-3) if (rocke_ms and rocke_ms > 0) else None
        )

        # -- aiter baseline
        aiter_ms: float | None = None
        aiter_ok, aiter_skip_reason = _aiter_supported(shape)
        if not args.no_aiter and aiter_ok:
            n_aiter_supported += 1
            try:
                aiter_data = _make_aiter_inputs(shape, seed=args.seed)
                _, aiter_ms = _run_aiter_live(
                    shape, aiter_data, warmup=args.warmup, iters=args.iterations
                )
                n_aiter_ran += 1
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}  AITER FAIL: {exc!r}")

        aiter_tflops = (
            tflop / (aiter_ms * 1e-3) if (aiter_ms and aiter_ms > 0) else None
        )
        speedup_vs_aiter = (
            aiter_ms / rocke_ms if (aiter_ms and rocke_ms and rocke_ms > 0) else None
        )

        rec = {
            "signature": shape.signature,
            "regime": shape.regime,
            "model": shape.model,
            "batch": shape.batch,
            "seqlen_q": shape.seqlen_q,
            "seqlen_k": shape.seqlen_k,
            "num_query_heads": shape.num_query_heads,
            "dtype": shape.dtype,
            "status": "ok" if rocke_ms is not None else "fail",
            "rocke_ms": rocke_ms,
            "rocke_tflops": rocke_tflops,
            "rocke_kernel": rocke_kname,
            "aiter_ms": aiter_ms,
            "aiter_tflops": aiter_tflops,
            "aiter_skip_reason": aiter_skip_reason if not aiter_ok else None,
            "speedup_rocke_vs_aiter": speedup_vs_aiter,
            "tflop": tflop,
            # FlyDSL: no MLA prefill kernel available (decode-only)
            "flydsl_ms": None,
            "flydsl_skip_reason": "no MLA prefill kernel in FlyDSL (decode-only as of Sep 2026)",
        }
        results.append(rec)

        # per-shape line
        rocke_str = f"rocke={rocke_ms * 1000:.1f}us" if rocke_ms else "rocke=FAIL"
        tf_str = f" {rocke_tflops:.1f}TF" if rocke_tflops else ""
        aiter_str = ""
        if args.no_aiter:
            aiter_str = " aiter=disabled"
        elif not aiter_ok:
            aiter_str = f" aiter=SKIP({aiter_skip_reason})"
        elif aiter_ms:
            spd = f" ({speedup_vs_aiter:.2f}x)" if speedup_vs_aiter else ""
            aiter_str = f" aiter={aiter_ms * 1000:.1f}us{spd}"
        else:
            aiter_str = " aiter=FAIL"
        print(f"{tag}  {rocke_str}{tf_str}{aiter_str}")

    # summary
    print(f"\n=== SUMMARY (arch={args.arch}) ===")
    ok_recs = [r for r in results if r.get("rocke_ms")]
    if ok_recs:
        by_regime: dict[str, list] = {}
        for r in ok_recs:
            by_regime.setdefault(r["regime"], []).append(r)
        for regime, rs in sorted(by_regime.items()):
            rocke_us = [r["rocke_ms"] * 1000 for r in rs]
            tf_vals = [r["rocke_tflops"] for r in rs if r.get("rocke_tflops")]
            aiter_spds = [
                r["speedup_rocke_vs_aiter"]
                for r in rs
                if r.get("speedup_rocke_vs_aiter")
            ]
            lat_part = f"rocke_lat_gm={_gm(rocke_us):.1f}us (n={len(rocke_us)})"
            tf_part = f"  tflops_gm={_gm(tf_vals):.1f}" if tf_vals else ""
            aiter_part = (
                f"  vs_aiter={_gm(aiter_spds):.3f}x (n={len(aiter_spds)})"
                if aiter_spds
                else ""
            )
            print(f"  {regime:<12}  {lat_part}{tf_part}{aiter_part}")
    print(f"\nrocke ok:          {n_ok}/{len(shapes)}")
    print(f"aiter supported:   {n_aiter_supported}  ran: {n_aiter_ran}")
    print("flydsl:            N/A (no MLA prefill kernel)")

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.output_json}  ({len(results)} shapes)")

    return 0 if n_ok == len(shapes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
