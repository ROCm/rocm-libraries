# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared utilities for Unified Attention (UA) benchmarks.

Loads aiter trace-shape JSON files (one JSON object per line) captured from
vLLM/aiter's `unified_attention` launches, filters for the "prefill 2D"
subset, and synthesizes the torch tensors required to drive both the AITER
Triton kernel and the CK DSL tiled-2D kernel from the same input.

Schema notes (per record):
  - kind:         "2d" (prefill or sliding) or "3d" (long-context decode split)
  - ALL_DECODE:   True iff every sequence has seqlen_q==1 (pure decode)
  - q_shape:      [total_q, num_query_heads, head_size]
  - k_shape:      [num_blocks, block_size, num_kv_heads, head_size]
  - v_shape:      same as k_shape
  - block_table_shape: [num_seqs, max_blocks_per_seq]
  - max_seqlen_q, max_seqlen_k, num_seqs, head_size, block_size
  - num_query_heads, num_kv_heads, num_queries_per_kv
  - softmax_scale, softcap, window_size: [size, 0]
  - has_sinks, has_alibi, has_output_scale
  - q_dtype/k_dtype/v_dtype/out_dtype: e.g. "torch.bfloat16"

"Prefill 2D" subset = kind == "2d" AND ALL_DECODE == False (i.e. at least
one sequence has seqlen_q > 1; the 2D kernel was selected because Triton's
heuristic kept it on the 2D path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


_DTYPE_MAP = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e4m3fnuz": torch.float8_e4m3fnuz,
}


def _torch_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype string {name!r}")
    return _DTYPE_MAP[name]


@dataclass(frozen=True)
class UAShape:
    """One captured unified-attention launch shape."""

    source_file: str
    line_idx: int
    call_idx: int
    kind: str
    all_decode: bool
    num_seqs: int
    total_q: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    max_blocks_per_seq: int
    max_seqlen_q: int
    max_seqlen_k: int
    softmax_scale: float
    softcap: float
    window_size: tuple[int, int]
    has_sinks: bool
    has_alibi: bool
    has_output_scale: bool
    q_dtype: str
    k_dtype: str
    v_dtype: str
    out_dtype: str

    @classmethod
    def from_record(
        cls, rec: dict[str, Any], source_file: str, line_idx: int
    ) -> "UAShape":
        q_shape = rec["q_shape"]
        k_shape = rec["k_shape"]
        bt_shape = rec["block_table_shape"]
        ws = rec.get("window_size", [-1, -1])
        return cls(
            source_file=source_file,
            line_idx=line_idx,
            call_idx=int(rec.get("call_idx", -1)),
            kind=str(rec.get("kind", "")),
            all_decode=bool(rec.get("ALL_DECODE", False)),
            num_seqs=int(rec["num_seqs"]),
            total_q=int(q_shape[0]),
            num_query_heads=int(rec["num_query_heads"]),
            num_kv_heads=int(rec["num_kv_heads"]),
            head_size=int(rec["head_size"]),
            block_size=int(rec["block_size"]),
            num_blocks=int(k_shape[0]),
            max_blocks_per_seq=int(bt_shape[1]),
            max_seqlen_q=int(rec["max_seqlen_q"]),
            max_seqlen_k=int(rec["max_seqlen_k"]),
            softmax_scale=float(rec["softmax_scale"]),
            softcap=float(rec.get("softcap", 0.0)),
            window_size=(int(ws[0]), int(ws[1])),
            has_sinks=bool(rec.get("has_sinks", False)),
            has_alibi=bool(rec.get("has_alibi", False)),
            has_output_scale=bool(rec.get("has_output_scale", False)),
            q_dtype=str(rec.get("q_dtype", "")),
            k_dtype=str(rec.get("k_dtype", "")),
            v_dtype=str(rec.get("v_dtype", "")),
            out_dtype=str(rec.get("out_dtype", "")),
        )

    @property
    def signature(self) -> str:
        """Compact human-readable id used for logs/CSVs."""
        sw = self.window_size[0] + 1 if self.window_size[0] >= 0 else 0
        return (
            f"d{self.head_size}_b{self.block_size}"
            f"_h{self.num_query_heads}kv{self.num_kv_heads}"
            f"_q{self.max_seqlen_q}_k{self.max_seqlen_k}"
            f"_ns{self.num_seqs}_tq{self.total_q}"
            f"_sw{sw}_sc{self.softcap:g}"
            f"_sinks{int(self.has_sinks)}"
            f"_{self.q_dtype.split('.')[-1]}"
        )


def load_shapes(paths: Iterable[Path]) -> list[UAShape]:
    """Load all UA shape records from one or more ndjson files."""
    shapes: list[UAShape] = []
    for p in paths:
        p = Path(p)
        with p.open() as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                shapes.append(UAShape.from_record(rec, p.name, i))
    return shapes


def filter_prefill_2d(
    shapes: Iterable[UAShape],
    *,
    dtype: str | None = "bf16",
    require_sliding_window: bool | None = None,
) -> list[UAShape]:
    """Filter to the prefill-2D subset.

    Prefill-2D := ``kind == "2d"`` AND ``ALL_DECODE == False``.

    Args:
      dtype: keep only shapes whose q_dtype matches ``"bf16"`` or ``"fp16"``
        (or pass ``None`` to keep all). The aiter ``q_dtype`` strings are
        ``"torch.bfloat16"`` / ``"torch.float16"``.
      require_sliding_window: keep only sliding-window shapes (True),
        only no-window shapes (False), or both (None).
    """
    if dtype is not None:
        wanted = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}.get(dtype)
        if wanted is None:
            raise ValueError(f"unsupported dtype filter {dtype!r}")
    else:
        wanted = None

    out: list[UAShape] = []
    for s in shapes:
        if s.kind != "2d" or s.all_decode:
            continue
        if wanted is not None and s.q_dtype != wanted:
            continue
        has_sw = s.window_size[0] >= 0
        if require_sliding_window is True and not has_sw:
            continue
        if require_sliding_window is False and has_sw:
            continue
        out.append(s)
    return out


def dedupe_shapes(shapes: Iterable[UAShape]) -> list[UAShape]:
    """Drop shapes that are duplicates ignoring call_idx / source / timestamp."""
    seen: set[tuple] = set()
    out: list[UAShape] = []
    for s in shapes:
        key = (
            s.num_seqs,
            s.total_q,
            s.num_query_heads,
            s.num_kv_heads,
            s.head_size,
            s.block_size,
            s.max_seqlen_q,
            s.max_seqlen_k,
            s.softmax_scale,
            s.softcap,
            s.window_size,
            s.has_sinks,
            s.has_alibi,
            s.q_dtype,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _distribute_lens(total: int, num_seqs: int, cap: int) -> list[int]:
    """Distribute ``total`` tokens across ``num_seqs`` so max <= cap."""
    if num_seqs <= 0:
        raise ValueError("num_seqs must be >= 1")
    if total < num_seqs:
        # pad with zero-length sequences; CK DSL/Triton both handle empty seqs
        return [total] + [0] * (num_seqs - 1)
    base, rem = divmod(total, num_seqs)
    lens = [base + (1 if i < rem else 0) for i in range(num_seqs)]
    # Pin the first sequence at ``cap`` (the captured max_seqlen_q) so the
    # kernel sees the trace's actual longest prefill row.
    if cap > 0 and lens[0] < cap and total >= cap:
        deficit = cap - lens[0]
        lens[0] = cap
        # Take the deficit from the largest other slot first.
        idx = max(range(1, num_seqs), key=lambda i: lens[i])
        if lens[idx] >= deficit:
            lens[idx] -= deficit
        else:
            # Fallback: spread the take across remaining slots.
            for j in range(1, num_seqs):
                if deficit == 0:
                    break
                take = min(lens[j], deficit)
                lens[j] -= take
                deficit -= take
    return lens


def make_inputs(
    shape: UAShape,
    *,
    device: str = "cuda",
    seed: int = 0,
    cap_blocks: int | None = 65536,
) -> dict[str, Any]:
    """Synthesize the tensors needed to launch a UA call for this shape.

    Per-sequence query / kv length splits are fabricated to match
    ``total_q``, ``num_seqs``, and ``max_seqlen_q`` / ``max_seqlen_k``
    (the only summary stats the trace records). All kv lengths are set
    to ``max_seqlen_k`` since trace records the per-call max only.

    ``cap_blocks`` caps ``num_blocks`` (the paged KV cache size) so that
    sweeps over very large captured shapes do not exhaust HBM. The cap
    must remain >= max_blocks_per_seq * num_seqs to keep the block_table
    indices in range.
    """
    torch.manual_seed(seed)

    q_dtype = _torch_dtype(shape.q_dtype)
    k_dtype = _torch_dtype(shape.k_dtype)
    v_dtype = _torch_dtype(shape.v_dtype)
    out_dtype = _torch_dtype(shape.out_dtype)

    query_lens = _distribute_lens(shape.total_q, shape.num_seqs, shape.max_seqlen_q)
    kv_lens_list = [shape.max_seqlen_k] * shape.num_seqs

    num_blocks = shape.num_blocks
    min_blocks = max(1, shape.max_blocks_per_seq)
    if cap_blocks is not None:
        num_blocks = min(num_blocks, max(cap_blocks, min_blocks))

    # ``torch.randn`` has no FP8 kernel; for fp8 caches generate in bf16 and
    # cast. (FP8 e4m3 has ~2 mantissa bits, so the scale of randn values is
    # fine for a perf/correctness smoke -- production uses calibrated scales.)
    def _randn(*sizes, dtype):
        if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2):
            return torch.randn(*sizes, dtype=torch.bfloat16, device=device).to(dtype)
        return torch.randn(*sizes, dtype=dtype, device=device)

    query = _randn(
        shape.total_q,
        shape.num_query_heads,
        shape.head_size,
        dtype=q_dtype,
    )
    key_cache = _randn(
        num_blocks,
        shape.block_size,
        shape.num_kv_heads,
        shape.head_size,
        dtype=k_dtype,
    )
    value_cache = _randn(
        num_blocks,
        shape.block_size,
        shape.num_kv_heads,
        shape.head_size,
        dtype=v_dtype,
    )
    output = torch.empty(
        shape.total_q,
        shape.num_query_heads,
        shape.head_size,
        dtype=out_dtype,
        device=device,
    )

    cu_seqlens_q = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0,
        num_blocks,
        (shape.num_seqs, shape.max_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    sinks = (
        torch.randn(shape.num_query_heads, dtype=torch.bfloat16, device=device)
        if shape.has_sinks
        else None
    )
    alibi_slopes = (
        -torch.linspace(
            0.05, 0.5, shape.num_query_heads, dtype=torch.float32, device=device
        )
        if shape.has_alibi
        else None
    )

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "output": output,
        "cu_seqlens_q": cu_seqlens_q,
        "kv_lens": kv_lens,
        "block_tables": block_tables,
        "sinks": sinks,
        "alibi_slopes": alibi_slopes,
        "query_lens": query_lens,
        "kv_lens_list": kv_lens_list,
        "scale": shape.softmax_scale,
        "max_query_len": shape.max_seqlen_q,
        "max_kv_len": shape.max_seqlen_k,
        "num_blocks": num_blocks,
    }


def attention_flops(
    shape: UAShape, query_lens: list[int], kv_lens_list: list[int]
) -> int:
    """Causal-attention FLOPs: 4 * sum_i(q_i * kv_i_effective) * H * d.

    For each sequence ``i``, the causal mask makes the average effective
    KV length per query equal to ``kv_len - q_len + (q_len + 1) / 2``
    (assuming a contiguous query suffix of size ``q_len`` at the end of
    the kv stream, matching aiter/Triton's convention). Sum that across
    sequences and multiply by ``4 * num_query_heads * head_size`` (QK and
    PV are each 2 flop / mac).
    """
    qd = shape.num_query_heads * shape.head_size
    macs = 0
    for q_len, kv_len in zip(query_lens, kv_lens_list):
        if q_len <= 0 or kv_len <= 0:
            continue
        if shape.window_size[0] >= 0:
            win = shape.window_size[0] + 1
            avg_eff = min(kv_len, (win + 1) // 2 + max(0, kv_len - win))
            # Simplified: treat each query as attending to <= win keys.
            avg_eff = min(win, kv_len)
        else:
            avg_eff = kv_len - q_len + (q_len + 1) / 2
            if avg_eff < 0:
                avg_eff = kv_len / 2
        macs += int(q_len * avg_eff)
    return 4 * macs * qd
