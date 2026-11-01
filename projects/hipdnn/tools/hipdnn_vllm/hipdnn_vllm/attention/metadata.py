# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Per-batch information the HIPDNN backend attaches to vLLM's metadata.

Classification uses host-side scalars vLLM already has (``num_reqs``,
``num_actual_tokens``, ``max_query_len``, ``max_seq_len``) and never reads
device memory, so it is capture-invariant: under HIP-graph capture it depends
only on the padded batch shape.

Encoder-only batches (BERT-style embedders: vLLM's ``EncoderOnlyAttention``
builds their metadata with ``causal=False`` and an ``EncoderOnlyAttentionSpec``)
are their own class. For them the info also carries the host-side request
layout P1e needs, ``encoder_groups``: runs of consecutive requests of equal
length, each ``(first_token, num_requests, length)``, from vLLM's
``query_start_loc_cpu``.

Decoder batches also carry :class:`PagedViews`, the device tensors the P2a/P2b
graphs (paged K/V, ragged Q) bind. Every view is sized for the per-process
bound ``B = max_num_seqs``. A plan built for a smaller batch ``B' <= B`` binds
the first ``B'`` rows of the same views (:meth:`PagedViews.prefix`), so every
plan size reads the same persistent buffers:

* ``page_table`` ``[B, M]``: vLLM's block table, zero-copy. CommonAttentionMetadata
  slices the model runner's persistent ``[max_num_reqs, M]`` buffer to the
  padded batch; the view is widened back to ``B`` rows over the same storage.
* ``seq_len_kv`` ``[B]``: vLLM's ``seq_lens``, widened the same way, zero-copy.
* ``ragged_offset`` ``[B + 1]``: ``query_start_loc`` copied on device into a
  builder-owned buffer whose tail repeats the last offset, so rows past the
  padded batch have ``q_len = 0``. (The runner's own tail is only refreshed on
  real steps; capture dummy runs leave stale, possibly larger, offsets there,
  which would make phantom rows write past the output.)
* ``seq_len_q`` ``[B]``: ``ragged_offset[1:] - ragged_offset[:-1]`` on device.

Rows with ``q_len = 0`` own no query tokens, so the engines never read their
``seq_len_kv`` or page-table row and never write output for them. That is why
the widened views may expose stale tails. When a source tensor's storage does
not reach ``B`` rows (not one of the runner's persistent buffers), it is copied
into a builder-owned buffer instead. Nothing here syncs with the host, and all
of it reads and writes persistent buffers, so FULL CUDA-graph replays see each
step's values.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import Any, Sequence

__all__ = [
    "BatchClass",
    "CaptureSize",
    "HipdnnBatchInfo",
    "PagedBounds",
    "PagedBuffers",
    "PagedViews",
    "classify",
    "encoder_groups",
]


class BatchClass(str, enum.Enum):
    UNIFORM_DECODE = "uniform_decode"  # every request has q_len == 1 (or k+1 for spec decode)
    PREFILL_NO_CONTEXT = "prefill_no_context"  # one request, whole sequence is the query
    MIXED = "mixed"  # anything else: chunked prefill, prefill+decode mixes
    ENCODER_ONLY = "encoder_only"  # bidirectional, no KV cache; every request is whole


@dataclass(frozen=True)
class PagedBounds:
    """Per-process upper bounds of every P2a/P2b plan."""

    max_num_seqs: int  # B
    max_decode_query_len: int  # P2a max_seq_len_q: 1, or k+1 with spec decode
    max_query_len: int  # P2b max_seq_len_q: min(max_num_batched_tokens, max_model_len)
    max_model_len: int  # rounded up to the block size for max_seq_len_kv


# One FULL CUDA graph vLLM captures: (uniform decode, padded num_reqs, max_seq_len_q).
CaptureSize = tuple[bool, int, int]


@dataclass(frozen=True)
class PagedViews:
    bounds: PagedBounds
    ragged_offset: Any  # [B + 1] int32
    seq_len_q: Any  # [B] int32
    seq_len_kv: Any  # [B] int32
    page_table: Any  # [B, M] int32
    # Rows this step's batch occupies (vLLM's num_reqs, padded): a plan's B must reach it.
    rows: int = 0
    # The FULL graphs vLLM captures, whose plans are built before any capture.
    capture_sizes: tuple[CaptureSize, ...] = ()
    # vLLM's V2 model runner: under FULL capture, this batch's num_reqs and
    # max_query_len are the graph's replay bounds (its descriptor has no num_reqs).
    capture_from_metadata: bool = False

    def prefix(self, batch: int) -> "PagedViews":
        """The first ``batch`` rows of every view: zero-copy, same storage."""
        if batch == int(self.seq_len_q.shape[0]):
            return self
        return replace(
            self,
            ragged_offset=self.ragged_offset[: batch + 1],
            seq_len_q=self.seq_len_q[:batch],
            seq_len_kv=self.seq_len_kv[:batch],
            page_table=self.page_table[:batch],
        )


def _widen_rows(t: Any, rows: int) -> Any:
    """``t`` (rows ``<= rows``) as a ``rows``-row view over its own storage, or
    ``None`` when the storage does not reach that far."""
    if t.shape[0] == rows:
        return t
    last = t.storage_offset() + (rows - 1) * t.stride(0)
    last += sum((n - 1) * s for n, s in zip(t.shape[1:], t.stride()[1:]))
    if (last + 1) * t.element_size() > t.untyped_storage().nbytes():
        return None
    return t.as_strided((rows, *t.shape[1:]), t.stride())


class PagedBuffers:
    """Builder-owned persistent device buffers behind :class:`PagedViews`."""

    def __init__(self, bounds: PagedBounds, device: Any):
        import torch

        b = bounds.max_num_seqs
        self.bounds = bounds
        self._device = device
        self.ragged_offset = torch.zeros(b + 1, dtype=torch.int32, device=device)
        self.seq_len_q = torch.zeros(b, dtype=torch.int32, device=device)
        # Set by the metadata builder once vLLM has fixed its capture sizes.
        self.capture_sizes: tuple[CaptureSize, ...] = ()
        self.capture_from_metadata = False
        # Allocated only if a source cannot be widened in place.
        self._seq_len_kv: Any = None
        self._page_table: Any = None

    def _own(self, attr: str, src: Any, rows: int, zero_tail: bool) -> Any:
        import torch

        buf = getattr(self, attr)
        if buf is None or buf.shape[1:] != src.shape[1:]:
            buf = torch.zeros((rows, *src.shape[1:]), dtype=src.dtype, device=self._device)
            setattr(self, attr, buf)
        n = src.shape[0]
        buf[:n].copy_(src, non_blocking=True)
        if zero_tail and n < rows:
            buf[n:].zero_()
        return buf

    def refresh(self, query_start_loc: Any, seq_lens: Any, block_table: Any) -> PagedViews | None:
        """Bind this step's batch. Device ops only (capture-safe); ``None`` when
        the batch exceeds the bound or is malformed for the paged graph."""
        import torch

        b = self.bounds.max_num_seqs
        n = int(seq_lens.shape[0])
        if (
            block_table is None
            or n > b
            or query_start_loc.shape[0] != n + 1
            or block_table.shape[0] != n
            or block_table.dtype != torch.int32
            or seq_lens.dtype != torch.int32
            or query_start_loc.dtype != torch.int32
        ):
            return None
        off = self.ragged_offset
        off[: n + 1].copy_(query_start_loc, non_blocking=True)
        if n < b:
            off[n + 1 :].copy_(query_start_loc[n:].expand(b - n), non_blocking=True)
        torch.sub(off[1:], off[:-1], out=self.seq_len_q)
        seq_len_kv = _widen_rows(seq_lens, b)
        if seq_len_kv is None:
            seq_len_kv = self._own("_seq_len_kv", seq_lens, b, zero_tail=True)
        page_table = _widen_rows(block_table, b)
        if page_table is None:
            # Rows past the batch have q_len 0 and are never read.
            page_table = self._own("_page_table", block_table, b, zero_tail=False)
        return PagedViews(
            self.bounds, off, self.seq_len_q, seq_len_kv, page_table, n, self.capture_sizes, self.capture_from_metadata
        )


@dataclass(frozen=True)
class HipdnnBatchInfo:
    batch_class: BatchClass
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    # Zero-copy device views from CommonAttentionMetadata, for the hipDNN paths.
    query_start_loc: Any
    seq_lens: Any
    block_table: Any
    # ENCODER_ONLY only: ((first_token, num_requests, length), ...).
    encoder_groups: tuple[tuple[int, int, int], ...] = ()
    # Decoder batches: the P2a/P2b bindings, or None when the batch cannot use them.
    paged: PagedViews | None = None


def classify(
    num_reqs: int,
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    max_decode_query_len: int,
    encoder_only: bool = False,
) -> BatchClass:
    """Classify a batch from host scalars only.

    ``max_decode_query_len`` is 1 without speculative decoding and ``k + 1``
    with ``k`` speculative tokens. ``encoder_only`` is a property of the
    metadata builder (its KV-cache spec / ``causal=False``), not of the batch.
    """
    if encoder_only:
        return BatchClass.ENCODER_ONLY
    if num_reqs <= 0:
        return BatchClass.MIXED
    if max_query_len <= max_decode_query_len and num_actual_tokens == num_reqs * max_query_len:
        return BatchClass.UNIFORM_DECODE
    # With one request the query covers the whole sequence exactly when there
    # is no cached context. With several requests the host scalars cannot
    # prove that, so they stay MIXED.
    if num_reqs == 1 and max_query_len == max_seq_len:
        return BatchClass.PREFILL_NO_CONTEXT
    return BatchClass.MIXED


def encoder_groups(starts: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Group requests from cumulative token offsets ``starts`` (``[num_reqs+1]``)
    into runs of consecutive equal-length requests: one dense graph call each.
    Empty requests are skipped."""
    groups: list[list[int]] = []
    for i in range(len(starts) - 1):
        first, length = int(starts[i]), int(starts[i + 1]) - int(starts[i])
        if length <= 0:
            continue
        last = groups[-1] if groups else None
        if last is not None and last[2] == length and last[0] + last[1] * length == first:
            last[1] += 1
        else:
            groups.append([first, 1, length])
    return tuple((g[0], g[1], g[2]) for g in groups)


def build_info(
    common_attn_metadata: Any,
    max_decode_query_len: int,
    encoder_only: bool = False,
    paged: PagedBuffers | None = None,
) -> HipdnnBatchInfo:
    cm = common_attn_metadata
    encoder_only = encoder_only or cm.causal is False
    groups: tuple[tuple[int, int, int], ...] = ()
    views: PagedViews | None = None
    if encoder_only:
        groups = encoder_groups(cm.query_start_loc_cpu[: cm.num_reqs + 1].tolist())
    elif paged is not None:
        views = paged.refresh(cm.query_start_loc, cm.seq_lens, cm.block_table_tensor)
    return HipdnnBatchInfo(
        batch_class=classify(
            cm.num_reqs,
            cm.num_actual_tokens,
            cm.max_query_len,
            cm.max_seq_len,
            max_decode_query_len,
            encoder_only,
        ),
        num_reqs=cm.num_reqs,
        num_actual_tokens=cm.num_actual_tokens,
        max_query_len=cm.max_query_len,
        max_seq_len=cm.max_seq_len,
        query_start_loc=cm.query_start_loc,
        seq_lens=cm.seq_lens,
        block_table=cm.block_table_tensor,
        encoder_groups=groups,
        paged=views,
    )


