# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Paths P2a (paged uniform decode) and P2b (ragged + paged).

Both paths run one hipDNN SDPA graph per layer call over the TRITON_ATTN KV
cache, with no data movement:

* Q / O: vLLM's packed ``[T, Hq, D]`` query and output, ragged through ONE
  offsets tensor (``PagedViews.ragged_offset``, vLLM's ``query_start_loc``)
  shared by Q and O, in token units, plus ``seq_len_q``.
* K / V: the cache's logical ``[num_blocks, Hkv, block_size, 2D]`` split along
  the last dim (``kv_cache.transpose(1, 2).split(D, -1)`` transposed back): two
  strided containers ``[num_blocks, Hkv, block_size, D]`` with the same strides.
* One page table (vLLM's block table, one uid for page_table_k and
  page_table_v), ``seq_len_kv`` (vLLM's ``seq_lens``) and the build-time bound
  ``max_seq_len_kv`` = ``max_model_len`` rounded up to the block size.
* Bottom-right causal, with vLLM's sliding window ``(W-1, 0)`` as left bound W-1.
* Attention sinks (gpt-oss): the layer's ``sinks`` ``[Hq]`` bound as hipDNN's
  ``sink_token`` ``[1, Hq, 1, 1]`` (see ``HipdnnAttentionImpl._sink``). A sink
  graph is a different plan from the sink-free one; whether an engine takes it
  is decided by the probe like every other attribute.

Plan sizes. A plan is built for a batch ``B`` and a query bound
``max_seq_len_q``; every per-step quantity (offsets, lengths, block ids) is
read by the engine from device tensors, and rows past the live batch have
``q_len = 0`` (see :mod:`.metadata`). The engines size their launch grids from
``B`` and ``max_seq_len_q`` (query tiles of every sequence and head), and the
work items past a sequence's real rows only exit, so a plan sized for the
per-process upper bounds (``B = max_num_seqs``; ``max_seq_len_q`` = the decode
query length for P2a, ``min(max_num_batched_tokens, max_model_len)`` for P2b)
launches mostly empty work for a small batch. Each call therefore runs a plan
sized from capture-invariant or bucketed host quantities:

* Under CUDA-graph capture, the size the graph is replayed within. vLLM's V1
  model runner names it in the forward context's batch descriptor: ``B`` =
  its padded ``num_reqs``; ``max_seq_len_q`` = the decode query length for a
  uniform-decode graph, else its padded ``num_tokens``. The V2 model runner
  passes no descriptor for FULL graphs (``set_forward_context`` then fills in
  one with no ``num_reqs``); it builds the capture batch's metadata from the
  graph's own bounds instead (``num_reqs``, and the promised
  ``max_query_len``), so those are the size. Every batch the graph is
  replayed for fits it.
* Eager: ``B`` = the next power of two >= ``num_reqs`` and, for P2b,
  ``max_seq_len_q`` = the next power of two >= ``max_query_len``, each capped
  at the upper bound. P2a keeps the decode query length.

A plan of ``B' < max_num_seqs`` binds the first ``B'`` rows of the persistent
views (:meth:`.metadata.PagedViews.prefix`, zero-copy). ``max_seq_len_kv``
stays the upper bound: the engines read each sequence's key count from
``seq_len_kv`` and their grids do not depend on the bound.

The plans per layer signature are bounded: one upper-bound plan per path, one
per FULL graph vLLM captures, at most ``log2(max_num_seqs) + 2`` eager P2a
sizes and ``(log2(max_num_seqs) + 2) * (log2(max_query_len) + 2)`` eager P2b
sizes. The coverage report lists the sizes built (``attention_plans``).

Plans are built only on eager calls, never under capture. At a layer's first
eager call for a KV cache (vLLM first profiles with a small cache, then
allocates the real one, so this runs once per cache), :meth:`PagedPath.prepare`
probes both paths at the upper bounds and builds the plan of every FULL graph
vLLM will capture. Each new plan runs once on an empty batch (all ``q_len = 0``,
nothing read or written) so engines finish their lazy per-device setup before
any capture. An eager call builds its bucket's plan on first use. Under
capture, a size with no built plan uses the smallest built plan that covers it
(at worst the upper bound). The probe results are fixed per cache, which keeps
the route a function of capture-invariant inputs.
"""

from __future__ import annotations

from typing import Any

import torch

from .. import stats
from ..declines import Decline, Stage, parse_op_error, parse_reason
from .metadata import PagedBounds, PagedViews
from .routing import P2A, P2B, LayerStatic

__all__ = [
    "PagedPath",
    "PlanSize",
    "bound_size",
    "capture_size",
    "eager_size",
    "graph_kwargs",
    "next_pow2",
    "split_kv_cache",
]

# (B, max_seq_len_q) of one P2a / P2b plan.
PlanSize = tuple[int, int]


def split_kv_cache(kv_cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V containers ``[num_blocks, Hkv, block_size, D]`` over the
    TRITON_ATTN cache (logical ``[num_blocks, Hkv, block_size, 2D]``): same
    strides, V offset by D elements."""
    return kv_cache[..., :head_size], kv_cache[..., head_size : 2 * head_size]


def next_pow2(n: int) -> int:
    """The smallest power of two >= ``n`` (1 for ``n <= 1``)."""
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def bound_size(path: str, bounds: PagedBounds) -> PlanSize:
    """The per-process upper bound of ``path``: it serves every batch."""
    return bounds.max_num_seqs, (bounds.max_decode_query_len if path == P2A else bounds.max_query_len)


def eager_size(path: str, bounds: PagedBounds, num_reqs: int, max_query_len: int) -> PlanSize:
    """The power-of-two bucket an eager batch of ``num_reqs`` requests whose
    longest query is ``max_query_len`` tokens runs, capped at the bounds."""
    b = min(next_pow2(num_reqs), bounds.max_num_seqs)
    if path == P2A:
        return b, bounds.max_decode_query_len
    return b, min(next_pow2(max_query_len), bounds.max_query_len)


def capture_size(
    path: str,
    bounds: PagedBounds,
    descriptor: Any,
    num_reqs: int,
    max_query_len: int,
    from_metadata: bool = False,
) -> PlanSize:
    """The size a CUDA graph being captured needs: it must hold every batch
    the graph is replayed for. ``num_reqs`` / ``max_query_len`` are the capture
    batch's own (padded) host scalars.

    * V1 model runner: ``descriptor`` is the graph's ``BatchDescriptor``. Its
      capture batch spreads the tokens one per request, so only the
      descriptor bounds the replayed query lengths.
    * V2 model runner (``from_metadata``), FULL graphs: no descriptor, or the
      ``BatchDescriptor(num_tokens)`` that ``set_forward_context`` fills in
      (no ``num_reqs``); the capture batch's ``num_reqs`` and
      ``max_query_len`` are the graph's replay bounds.
    * Neither: the upper bound."""
    q = bounds.max_decode_query_len
    if descriptor is not None and not getattr(descriptor, "num_reqs", None):
        descriptor = None  # bounds no requests
    if from_metadata and descriptor is None:
        b = min(num_reqs, bounds.max_num_seqs)
        return b, q if path == P2A else min(max(max_query_len, 1), bounds.max_query_len)
    if descriptor is None:
        return bound_size(path, bounds)
    b = min(max(int(descriptor.num_reqs), num_reqs), bounds.max_num_seqs)
    if path == P2A or descriptor.uniform:
        return b, q
    return b, min(max(int(descriptor.num_tokens), max_query_len), bounds.max_query_len)


def graph_kwargs(
    path: str,
    p: PagedViews,
    block_size: int,
    left_bound: int,
    size: PlanSize | None = None,
    sink: torch.Tensor | None = None,
) -> dict[str, Any]:
    """The ``sdpa_fwd`` arguments of the P2a / P2b graph of ``size`` (default:
    the upper bound) for these bindings."""
    b, max_q = size if size is not None else bound_size(path, p.bounds)
    v = p.prefix(b)
    max_kv = min(-(-p.bounds.max_model_len // block_size) * block_size, int(p.page_table.shape[1]) * block_size)
    return dict(
        causal=True,
        bottom_right=True,
        left_bound=left_bound,
        ragged_offset_q=v.ragged_offset,
        seq_len_q=v.seq_len_q,
        max_seq_len_q=max_q,
        page_table=v.page_table,
        seq_len_kv=v.seq_len_kv,
        max_seq_len_kv=max_kv,
        sink=sink,
    )


# Empty-batch length tensors for the warm-up launch, per (device, B).
_EMPTY: dict[tuple[Any, int], tuple[torch.Tensor, torch.Tensor]] = {}
# Plans already warmed in this process: (layer, KV container, path, size). Layers
# with the same static attributes and cache layout share one plan.
_WARMED: set[tuple[Any, ...]] = set()


def _empty_batch(device: Any, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, b)
    t = _EMPTY.get(key)
    if t is None:
        t = (
            torch.zeros(b + 1, dtype=torch.int32, device=device),  # ragged offsets
            torch.zeros(b, dtype=torch.int32, device=device),  # seq_len_q / seq_len_kv
        )
        _EMPTY[key] = t
    return t


def _container_key(kv_cache: Any) -> Any:
    """What a plan's K/V description depends on: the cache's layout."""
    if kv_cache is None:
        return None
    return (tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype, kv_cache.device)


class _Plans:
    """The plans one layer has for one KV cache."""

    def __init__(self, container: Any):
        self.container = container
        # Upper-bound probe per path: None = served (engine name in engines), else the decline.
        self.status: dict[str, Decline | None] = {}
        self.engines: dict[str, str] = {}
        self.built: dict[str, set[PlanSize]] = {P2A: set(), P2B: set()}
        self.declined: set[tuple[str, int, int]] = set()

    def cover(self, path: str, b: int, max_q: int) -> PlanSize | None:
        """The smallest built plan of ``path`` that holds ``(b, max_q)``."""
        best = None
        for size in self.built[path]:
            if size[0] < b or size[1] < max_q:
                continue
            if best is None or (size[0] * size[1], size) < (best[0] * best[1], best):
                best = size
        return best


class PagedPath:
    def __init__(self, layer: LayerStatic):
        self._layer = layer
        # vLLM's decoder window is (W-1, 0), (-1, -1) when unset.
        self._left = int(layer.sliding_window[0])
        self._plans: _Plans | None = None
        # The sink_token tensor bound by every call ([1, Hq, 1, 1] or None),
        # fixed at prepare so the plans and every launch use one address.
        self._sink: torch.Tensor | None = None
        self._C: Any = None

    def _c(self) -> Any:
        if self._C is None:
            from .. import _bootstrap

            _bootstrap.load()
            from .. import _C

            self._C = _C
        return self._C

    @property
    def prepared(self) -> bool:
        return self._plans is not None

    def ready(self, kv_cache: Any) -> bool:
        """Whether :meth:`prepare` ran for this KV cache's layout."""
        return self._plans is not None and self._plans.container == _container_key(kv_cache)

    def status(self, path: str) -> Decline | None:
        return self._plans.status[path]

    def engine(self, path: str) -> str:
        return self._plans.engines.get(path, "") if self._plans is not None else ""

    def plan_sizes(self, path: str) -> list[PlanSize]:
        """The sizes this layer has built plans for, for the current cache."""
        return sorted(self._plans.built[path]) if self._plans is not None else []

    # ------------------------------------------------------------ binding --

    def _tensors(self, query, kv_cache, output, info):
        layer = self._layer
        n = info.num_actual_tokens
        d = layer.head_size
        q = query[:n].view(n, layer.num_heads, d)
        o = output[:n].view(n, layer.num_heads, d)
        k, v = split_kv_cache(kv_cache, d)
        return q, k, v, o

    def _kwargs(self, path: str, p: PagedViews, kv_cache: torch.Tensor, size: PlanSize) -> dict[str, Any]:
        return graph_kwargs(path, p, int(kv_cache.shape[2]), self._left, size, self._sink)

    def _shape_decline(self, kv_cache: torch.Tensor) -> Decline | None:
        layer = self._layer
        if kv_cache is None or kv_cache.dim() != 4:
            return Decline.feature("kv_layout")
        if kv_cache.shape[1] != layer.num_kv_heads or kv_cache.shape[3] != 2 * layer.head_size:
            # Not the plain TRITON_ATTN (num_blocks, Hkv, block_size, 2D) cache.
            return Decline.feature("kv_layout")
        return None

    def _build(self, plans: _Plans, path: str, size: PlanSize, origin: str, tensors, p, kv_cache) -> Decline | None:
        """Build (or fetch) the plan of ``size`` and warm it with an empty
        batch. Eager only. Returns the decline when no engine takes it."""
        q, k, v, o = tensors
        kw = self._kwargs(path, p, kv_cache, size)
        reason, detail = self._c().sdpa_fwd_probe(q, k, v, o, self._layer.scale, **kw)
        if reason:
            d = parse_reason(reason)
            plans.declined.add((path, *size))
            return Decline(d.stage, d.detail, detail)
        warm = (self._layer, plans.container, path, size)
        if warm not in _WARMED:
            offsets, lengths = _empty_batch(q.device, size[0])
            kw.update(ragged_offset_q=offsets, seq_len_q=lengths, seq_len_kv=lengths)
            try:
                torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, o, self._layer.scale, **kw)
            except RuntimeError as exc:
                parsed = parse_op_error(exc)
                if parsed is None:
                    raise
                plans.declined.add((path, *size))
                return parsed[1]
            _WARMED.add(warm)
        plans.built[path].add(size)
        plans.engines.setdefault(path, detail)
        stats.note_attention_plan(path, size[0], size[1], origin)
        return None

    # ------------------------------------------------------------- warm-up --

    def prepare(self, query, kv_cache, output, info, sink: torch.Tensor | None = None) -> None:
        """Probe P2a and P2b at the upper bounds for this layer's signature and
        KV cache, and build the plans of every FULL graph vLLM captures. Eager
        only; runs once per layer and KV cache. ``sink`` is the layer's
        ``sink_token`` view, bound by this and every later call."""
        self._sink = sink
        p = info.paged
        plans = _Plans(_container_key(kv_cache))
        shape = self._shape_decline(kv_cache)
        if shape is not None:
            plans.status = {P2A: shape, P2B: shape}
            self._plans = plans
            return
        tensors = self._tensors(query, kv_cache, output, info)
        for path in (P2A, P2B):
            size = bound_size(path, p.bounds)
            plans.status[path] = self._build(plans, path, size, "bound", tensors, p, kv_cache)
        for uniform, b, max_q in p.capture_sizes:
            # A uniform-decode graph holds P2a, or P2b when only P2b is served.
            path = P2A if uniform and plans.status[P2A] is None else P2B
            size = (b, max_q)
            if plans.status[path] is not None or size in plans.built[path] or (path, *size) in plans.declined:
                continue
            self._build(plans, path, size, "capture", tensors, p, kv_cache)
        self._plans = plans

    # ----------------------------------------------------------------- run --

    def run(
        self, path: str, query, kv_cache, output, info, capturing: bool = False, descriptor: Any = None
    ) -> Decline | None:
        """Write the batch's attention into ``output``. Returns the decline
        (nothing written) when this layer call cannot take ``path``.
        ``descriptor`` is vLLM's batch descriptor of the graph being captured."""
        p = info.paged
        if p is None:
            return Decline.feature("paged_metadata")
        plans = self._plans
        status = plans.status.get(path) if plans is not None else Decline(Stage.CAPTURE_INVARIANT, "not_prepared")
        if status is not None:
            return status
        tensors = self._tensors(query, kv_cache, output, info)
        num_reqs = max(int(info.num_reqs), p.rows)
        if capturing:
            need = capture_size(
                path, p.bounds, descriptor, num_reqs, int(info.max_query_len), p.capture_from_metadata
            )
            size = plans.cover(path, *need)
        else:
            want = eager_size(path, p.bounds, num_reqs, int(info.max_query_len))
            if want not in plans.built[path] and (path, *want) not in plans.declined:
                self._build(plans, path, want, "eager", tensors, p, kv_cache)
            size = want if want in plans.built[path] else plans.cover(path, *want)
        if size is None:  # the upper bound is built whenever the path is served
            return Decline(Stage.CAPTURE_INVARIANT, "no_plan")
        q, k, v, o = tensors
        torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, o, self._layer.scale, **self._kwargs(path, p, kv_cache, size))
        return None
