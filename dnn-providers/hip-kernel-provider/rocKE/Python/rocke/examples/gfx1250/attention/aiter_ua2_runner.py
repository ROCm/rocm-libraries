# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 unified-attention 2D runner over the AITER trace cohort.

Drives the gfx1250 WMMA tiled-2D unified-attention path
(``instances/gfx1250/attention_tiled_2d.py``) across the real
``aiter_ua_2_shapes.json`` trace (head_size=64, block_size=32, GQA-8, bf16
Q/O, fp8e4m3 paged K/V, sinks, optional sliding-window).

Two modes:

* ``--build-only`` (no GPU, no torch): for each *distinct kernel signature*
  in the cohort, exercise the production dispatch
  (``problem -> supports_native_unified_attention_tiled -> _tiled_spec_from_problem
  -> build_unified_attention_2d_tiled -> lower_kernel_to_llvm``) and report how
  many build+lower cleanly. This validates that the merged tiled leg compiles
  across the trace's shape variety without a GPU.

* default (GPU): materialise bf16 Q/O + fp8 K/V paged-cache inputs, run the
  tiled kernel via ``run_unified_attention_torch(backend="tiled")``, and compare
  against a paged-attention reference (fp8 dequant) reporting max_abs / latency.

Usage (no GPU, runs anywhere with rocke importable):

    PYTHONPATH=Python python -m rocke.examples.gfx1250.attention.aiter_ua2_runner \\
        --build-only --shapes Python/rocke/examples/gfx950/attention/aiter_ua_2_shapes.json

Usage (on the gfx1250 box, venv python + ROCm on LD_LIBRARY_PATH):

    HIP_VISIBLE_DEVICES=<idx> python -m \\
        rocke.examples.gfx1250.attention.aiter_ua2_runner --limit 16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# composablekernel/python on the path so ``rocke`` imports cleanly when the
# file is run as a module from the repo root.
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT / "Python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Python"))

_DEFAULT_SHAPES = (
    _ROOT
    / "Python"
    / "rocke"
    / "examples"
    / "gfx950"
    / "attention"
    / "aiter_ua_2_shapes.json"
)


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------


def _window_to_sliding(window_size) -> int:
    """Map an AITER ``window_size`` field to a rocke ``sliding_window``.

    AITER encodes a left-only causal window as ``(left, 0)`` where ``left =
    sliding_window - 1`` (matches the Triton harness' ``(sw - 1, 0)``).
    ``(-1, -1)`` means full (non-windowed) attention.
    """
    if isinstance(window_size, (list, tuple)):
        left = int(window_size[0])
        if left < 0:
            return 0
        return left + 1
    return 0


def load_shapes(path: Path, limit: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _binary_search_iters(num_seqs: int) -> int:
    if num_seqs <= 0:
        return 32
    return max(1, int(math.ceil(math.log2(num_seqs + 1))))


def shape_signature(rec: Dict) -> Tuple:
    """The tuple that determines a distinct *compiled kernel* for this cohort.

    head_size/block_size/heads/dtypes are fixed across the whole trace, so the
    only fields that change the emitted kernel body are the sliding-window mode
    and the unrolled binary-search depth (a function of ``num_seqs``).
    """
    return (
        int(rec["head_size"]),
        int(rec["block_size"]),
        int(rec["num_query_heads"]),
        int(rec["num_kv_heads"]),
        bool(rec["has_sinks"]),
        _window_to_sliding(rec.get("window_size")),
        _binary_search_iters(int(rec["num_seqs"])),
    )


def build_problem(rec: Dict):
    from rocke.instances import UnifiedAttentionProblem

    q_shape = rec["q_shape"]  # [total_q, num_query_heads, head_size]
    total_q = int(q_shape[0])
    return UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=int(rec["num_seqs"]),
        num_query_heads=int(rec["num_query_heads"]),
        num_kv_heads=int(rec["num_kv_heads"]),
        head_size=int(rec["head_size"]),
        block_size=int(rec["block_size"]),
        max_seqlen_q=int(rec["max_seqlen_q"]),
        max_seqlen_k=int(rec["max_seqlen_k"]),
        dtype="bf16",
        q_dtype="bf16",
        sliding_window=_window_to_sliding(rec.get("window_size")),
        softcap=float(rec.get("softcap", 0.0)),
        use_sinks=bool(rec["has_sinks"]),
        use_alibi=bool(rec.get("has_alibi", False)),
        use_fp8=True,
    )


# ---------------------------------------------------------------------------
# Build-only (no GPU): exercise the production dispatch + LLVM lowering.
# ---------------------------------------------------------------------------


def build_only(rows: List[Dict]) -> int:
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.instances.common import attention_unified as au
    from rocke.instances.gfx1250.attention_tiled_2d import (
        build_unified_attention_2d_tiled,
    )

    prev = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx1250"

    seen: Dict[Tuple, Dict] = {}
    for rec in rows:
        sig = shape_signature(rec)
        seen.setdefault(sig, rec)

    print(
        f"cohort: {len(rows)} records -> {len(seen)} distinct kernel signatures "
        f"(by sliding-window x binary-search depth)"
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0
    try:
        for sig, rec in sorted(seen.items()):
            problem = build_problem(rec)
            ok, why = au.supports_native_unified_attention_tiled(problem)
            if not ok:
                print(f"  SKIP sig={sig}: {why}")
                n_skip += 1
                continue
            try:
                spec = au._tiled_spec_from_problem(problem)
                kernel = build_unified_attention_2d_tiled(spec, arch="gfx1250")
                ll = lower_kernel_to_llvm(kernel, arch="gfx1250")
                bf16_wmma = "llvm.amdgcn.wmma.f32.16x16x32.bf16" in ll
                fp8_cvt = "llvm.amdgcn.cvt.pk.f32.fp8" in ll
                if not (bf16_wmma and fp8_cvt):
                    raise AssertionError(
                        f"missing expected intrinsics (wmma={bf16_wmma} fp8={fp8_cvt})"
                    )
                n_ok += 1
                sw = sig[5]
                print(
                    f"  OK   sw={sw:<4d} bsearch={sig[6]:<2d} name={spec.kernel_name()}"
                )
            except Exception as e:  # pragma: no cover - surfaced in the report
                n_fail += 1
                print(f"  FAIL sig={sig}: {type(e).__name__}: {e}")
    finally:
        au._RESOLVED_ATTENTION_ARCH = prev

    print(f"\nbuild-only: {n_ok} ok, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# GPU correctness + latency
# ---------------------------------------------------------------------------


def _ref_paged_attn_fp8(data, *, sliding_window: int):
    import torch

    q = data["query"].float()
    k = data["key_cache"].float() * data["k_scale"]
    v = data["value_cache"].float() * data["v_scale"]
    block_tables = data["block_tables"].cpu().numpy()
    _, block_size, num_kv_heads, head_size = k.shape
    query_lens = data["query_lens"]
    kv_lens = data["kv_lens_list"]
    sinks = data["sinks"]
    scale = data["scale"]
    outputs = []
    start = 0
    for i in range(len(query_lens)):
        ql = query_lens[i]
        kvl = int(kv_lens[i])
        qi = q[start : start + ql] * scale
        nblk = (kvl + block_size - 1) // block_size
        idx = block_tables[i, :nblk]
        ki = k[idx].reshape(-1, num_kv_heads, head_size)[:kvl]
        vi = v[idx].reshape(-1, num_kv_heads, head_size)[:kvl]
        if qi.shape[1] != ki.shape[1]:
            rep = qi.shape[1] // ki.shape[1]
            ki = torch.repeat_interleave(ki, rep, dim=1)
            vi = torch.repeat_interleave(vi, rep, dim=1)
        attn = torch.einsum("qhd,khd->hqk", qi, ki).float()
        mask = torch.triu(
            torch.ones(ql, kvl, device=qi.device), diagonal=kvl - ql + 1
        ).bool()
        if sliding_window and sliding_window > 0:
            sw = (
                torch.triu(
                    torch.ones(ql, kvl, device=qi.device),
                    diagonal=kvl - (ql + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= sw
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks.float()[:, None, None].repeat_interleave(
                attn.shape[-2], dim=-2
            )
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn.to(vi.dtype), vi)
        outputs.append(out)
        start += ql
    return torch.cat(outputs, dim=0)


def _make_inputs(rec: Dict, seed: int = 0):
    import torch

    q_shape = rec["q_shape"]
    total_q = int(q_shape[0])
    nqh = int(rec["num_query_heads"])
    nkvh = int(rec["num_kv_heads"])
    hd = int(rec["head_size"])
    bs = int(rec["block_size"])
    num_seqs = int(rec["num_seqs"])
    num_blocks = int(rec["k_shape"][0])
    torch.manual_seed(seed)

    # Per-seq query/kv lengths. The trace stores only aggregate shapes, so we
    # synthesise plausible lengths: a uniform split of total_q across seqs with
    # kv_len = context + query_len. Decode shapes (max_seqlen_q==1) get q=1.
    max_q = int(rec["max_seqlen_q"])
    max_k = int(rec["max_seqlen_k"])
    if max_q <= 1:
        query_lens = [1] * num_seqs
    else:
        base = max(1, total_q // num_seqs)
        query_lens = [base] * num_seqs
        query_lens[-1] += total_q - sum(query_lens)
        query_lens = [max(1, x) for x in query_lens]
    # keep total_q consistent
    total_q = sum(query_lens)
    kv_lens_list = [
        max(query_lens[i], min(max_k, query_lens[i] + 64)) for i in range(num_seqs)
    ]

    query = torch.randn(total_q, nqh, hd, dtype=torch.bfloat16, device="cuda")
    kf = torch.randn(num_blocks, bs, nkvh, hd, dtype=torch.float32, device="cuda") * 0.1
    vf = torch.randn(num_blocks, bs, nkvh, hd, dtype=torch.float32, device="cuda") * 0.1
    key_cache = kf.to(torch.float8_e4m3fn)
    value_cache = vf.to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    max_blocks = (max(kv_lens_list) + bs - 1) // bs
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device="cuda"
    )
    sinks = (
        torch.randn(nqh, dtype=torch.bfloat16, device="cuda")
        if rec["has_sinks"]
        else None
    )
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_q": cu_q,
        "kv_lens": kv_lens,
        "query_lens": query_lens,
        "kv_lens_list": kv_lens_list,
        "block_tables": block_tables,
        "scale": float(rec["softmax_scale"]),
        "sinks": sinks,
        "k_scale": 1.0,
        "v_scale": 1.0,
        "max_query_len": max(query_lens),
        "max_kv_len": max(kv_lens_list),
    }


def gpu_run(rows: List[Dict], *, warmup: int, attempts: int) -> int:
    import torch

    from rocke.instances import UnifiedAttentionProblem, run_unified_attention_torch

    if not torch.cuda.is_available():
        print("HIP device unavailable; cannot run GPU mode", file=sys.stderr)
        return 1
    print("device:", torch.cuda.get_device_name(0))

    n_pass = 0
    n_fail = 0
    for i, rec in enumerate(rows):
        sw = _window_to_sliding(rec.get("window_size"))
        data = _make_inputs(rec)
        out = torch.empty_like(data["query"])
        problem = UnifiedAttentionProblem(
            total_q=data["query"].shape[0],
            num_seqs=int(rec["num_seqs"]),
            num_query_heads=int(rec["num_query_heads"]),
            num_kv_heads=int(rec["num_kv_heads"]),
            head_size=int(rec["head_size"]),
            block_size=int(rec["block_size"]),
            max_seqlen_q=data["max_query_len"],
            max_seqlen_k=data["max_kv_len"],
            dtype="bf16",
            q_dtype="bf16",
            sliding_window=sw,
            softcap=0.0,
            use_sinks=data["sinks"] is not None,
            use_fp8=True,
        )
        try:
            ms = run_unified_attention_torch(
                problem=problem,
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=out,
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_tables"],
                softcap=0.0,
                sinks=data["sinks"],
                backend="tiled",
                warmup=warmup,
                attempts=attempts,
                k_scale=data["k_scale"],
                v_scale=data["v_scale"],
            )
            from rocke.runtime import synchronize_and_release

            synchronize_and_release()
            ref = _ref_paged_attn_fp8(data, sliding_window=sw)
            max_abs = float((ref.float() - out.float()).abs().max().item())
            ok = math.isfinite(max_abs) and max_abs < 5e-2
            n_pass += int(ok)
            n_fail += int(not ok)
            lat = f"{ms * 1000:.1f}us" if isinstance(ms, float) else "-"
            print(
                f"  [{i}] sw={sw} ns={rec['num_seqs']} tq={problem.total_q}: "
                f"max_abs={max_abs:.4g} lat={lat} {'PASS' if ok else 'FAIL'}"
            )
        except Exception as e:  # pragma: no cover
            n_fail += 1
            print(
                f"  [{i}] sw={sw} ns={rec['num_seqs']}: ERROR {type(e).__name__}: {e}"
            )

    print(f"\ngpu-run: {n_pass} pass, {n_fail} fail")
    return 1 if n_fail else 0


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=Path, default=_DEFAULT_SHAPES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="no GPU/torch: build + LLVM-lower each distinct kernel signature",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=10)
    args = ap.parse_args()

    if not args.shapes.exists():
        print(f"shapes file not found: {args.shapes}", file=sys.stderr)
        return 2
    rows = load_shapes(args.shapes, limit=args.limit)
    if not rows:
        print("no shapes loaded", file=sys.stderr)
        return 2

    if args.build_only:
        return build_only(rows)
    return gpu_run(rows, warmup=args.warmup, attempts=args.attempts)


if __name__ == "__main__":
    sys.exit(main())
