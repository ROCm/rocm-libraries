# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``HIPDNN`` attention backend.

The backend subclasses TRITON_ATTN and inherits its KV-cache layout, block
sizes, dtypes, head sizes, feature flags, ``forward_includes_kv_cache_update =
False`` and its KV write (``do_kv_cache_update``). Platform validation therefore
passes whenever TRITON_ATTN's would, and the inherited ``forward`` is the
fallback (path P0), reading the same cache with no data movement.

Each call is routed by :mod:`.routing` and counted in the coverage report.
Decoder attention runs through paths P2a/P2b (:mod:`.paged`): one hipDNN SDPA
launch per layer call over the paged cache, from plans sized per batch class:
the exact size of each CUDA graph vLLM captures, and power-of-two buckets for
eager batches, all within the process's scheduler bounds.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.attention.backend import AttentionCGSupport, max_decode_query_len
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

from .. import stats
from ..config import load_config
from ..declines import Decline, DeclineError, Stage, parse_op_error
from .encoder_only import EncoderOnlyPath
from .metadata import BatchClass, CaptureSize, PagedBounds, PagedBuffers, build_info
from .paged import PagedPath
from .routing import P1E, P2A, P2B, Route, layer_static_from_impl, select

__all__ = [
    "HipdnnAttentionBackend",
    "HipdnnAttentionImpl",
    "HipdnnAttentionMetadataBuilder",
]

logger = logging.getLogger("hipdnn_vllm")


def uses_v2_model_runner(vllm_config) -> bool:
    """Whether vLLM runs its V2 GPU model runner (``vllm/v1/worker/gpu``)."""
    return bool(getattr(vllm_config, "use_v2_model_runner", False))


def capture_sizes(vllm_config, bounds: PagedBounds) -> tuple[CaptureSize, ...]:
    """The FULL CUDA graphs vLLM captures, as ``(uniform, num_reqs, max_seq_len_q)``.

    A mixed FULL graph of ``x`` tokens holds up to ``min(x, max_num_seqs)``
    requests of up to ``x`` tokens; a uniform-decode FULL graph holds requests
    of ``q`` tokens, ``q`` = the decode query length. PIECEWISE graphs run
    attention outside the graph and need no plan. Mirrors, per model runner:

    * V1: ``CudagraphDispatcher.initialize_cudagraph_keys`` and
      ``_create_padded_batch_descriptor`` (vllm/v1/cudagraph_dispatcher.py):
      uniform graphs of the capture sizes ``x`` that ``q`` divides, ``x // q``
      requests.
    * V2: ``CudaGraphManager._init_candidates``
      (vllm/v1/worker/gpu/cudagraph_utils.py): uniform graphs of every capture
      size rounded up to a multiple of ``q``.
    """
    from vllm.config import CUDAGraphMode

    cc = vllm_config.compilation_config
    mode = cc.cudagraph_mode
    if mode is None or mode == CUDAGraphMode.NONE:
        return ()
    sizes = [int(x) for x in (cc.cudagraph_capture_sizes or ())]
    if not sizes:
        return ()
    max_b, q = bounds.max_num_seqs, bounds.max_decode_query_len
    out: set[CaptureSize] = set()
    if mode.mixed_mode() == CUDAGraphMode.FULL:
        out.update((False, min(x, max_b), min(x, bounds.max_query_len)) for x in sizes)
    if mode.decode_mode() == CUDAGraphMode.FULL and mode.separate_routine():
        if uses_v2_model_runner(vllm_config):
            max_capture = int(cc.max_cudagraph_capture_size or max(sizes))
            rounded = {-(-x // q) * q for x in sizes}
            out.update((True, x // q, q) for x in rounded if x <= q * max_b and x <= max_capture)
        else:
            out.update((True, min(x // q, max_b), q) for x in sizes if q <= x <= q * max_b and x % q == 0)
    return tuple(sorted(out))


class HipdnnAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """TRITON_ATTN metadata plus ``metadata.hipdnn`` (:class:`HipdnnBatchInfo`).

    * P2a/P2b are single launches with no workspace, no host sync and no
      allocation. Their plans are bounded by ``max_num_seqs`` /
      ``max_num_batched_tokens`` / ``max_model_len`` and read lengths, offsets
      and block ids from the persistent buffers this builder refreshes in
      :meth:`build` with device ops only. A graph holds the plan sized for the
      batches it is replayed for, built before any capture.
    * Under capture the route follows the graph, not the dummy batch: P2a only
      for uniform-decode graphs (replayed only for uniform decode batches of
      that query length); P2b, which takes any batch, for every other FULL
      graph. vLLM's V1 model runner names the graph in the forward context's
      batch descriptor; the V2 runner passes none for FULL graphs (the forward
      context then holds one with no request count) and builds the capture
      batch's metadata from the graph's bounds instead.
    * Plans are probed, built for every FULL graph vLLM captures, and warmed at
      a layer's first eager call for each KV cache; a layer that has not run
      eagerly on the cache keeps TRITON_ATTN in the graph (P0, capture-safe),
      as does every pre-filtered feature and P1e.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        self._max_decode_query_len = max_decode_query_len(vllm_config)
        self._encoder_only = isinstance(kv_cache_spec, EncoderOnlyAttentionSpec)
        self._vllm_config = vllm_config
        self._capture_sizes_known = False
        self._paged: PagedBuffers | None = None
        if not self._encoder_only and load_config().op_enabled("attention"):
            sched = vllm_config.scheduler_config
            max_model_len = int(vllm_config.model_config.max_model_len)
            bounds = PagedBounds(
                max_num_seqs=int(sched.max_num_seqs),
                max_decode_query_len=self._max_decode_query_len,
                max_query_len=min(int(sched.max_num_batched_tokens), max_model_len),
                max_model_len=max_model_len,
            )
            self._paged = PagedBuffers(bounds, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        if self._paged is not None and not self._capture_sizes_known:
            # vLLM resolves the CUDA-graph mode and capture sizes after it
            # creates the builders and before the first batch is built.
            self._paged.capture_sizes = capture_sizes(self._vllm_config, self._paged.bounds)
            self._paged.capture_from_metadata = uses_v2_model_runner(self._vllm_config)
            self._capture_sizes_known = True
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.hipdnn = build_info(
            common_attn_metadata, self._max_decode_query_len, self._encoder_only, self._paged
        )
        return metadata

    def _refresh(self, metadata) -> None:
        if self._paged is not None and metadata.hipdnn.paged is not None:
            self._paged.refresh(metadata.query_start_loc, metadata.seq_lens, metadata.block_table)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # TRITON_ATTN sets seq_lens to 1 after build(); rebind so a copied (not
        # widened) seq_len_kv sees it as well.
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        self._refresh(metadata)
        return metadata

    def update_draft_decode_metadata(self, metadata) -> None:
        # A speculator may record this inside its draft graph: device ops only.
        super().update_draft_decode_metadata(metadata)
        self._refresh(metadata)


def _capture_descriptor():
    """vLLM's batch descriptor of the graph being captured, or ``None`` when it
    does not bound the graph's requests. The V2 model runner captures FULL
    graphs with no descriptor, and ``set_forward_context``
    (vllm/forward_context.py) then fills in ``BatchDescriptor(num_tokens)``,
    which carries no ``num_reqs``; neither does a PIECEWISE descriptor."""
    if not is_forward_context_available():
        return None
    descriptor = get_forward_context().batch_descriptor
    if descriptor is None or getattr(descriptor, "num_reqs", None) is None:
        return None
    return descriptor


def _captured_class(batch_class: BatchClass, from_metadata: bool) -> BatchClass:
    """The batch class a graph being captured is replayed for: uniform decode
    only when vLLM captures a uniform-decode graph. The V2 model runner
    (``from_metadata``) captures FULL graphs with no request-bounding
    descriptor and builds the capture batch's metadata from the graph's
    bounds, so its class is the graph's; the V1 runner's capture batch looks
    uniform for small mixed graphs too, so only its descriptor tells."""
    if batch_class is BatchClass.UNIFORM_DECODE:
        descriptor = _capture_descriptor()
        if descriptor is None:
            return BatchClass.UNIFORM_DECODE if from_metadata else BatchClass.MIXED
        if descriptor.uniform:
            return BatchClass.UNIFORM_DECODE
    return BatchClass.MIXED


class HipdnnAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = load_config()
        self._attention_enabled = config.op_enabled("attention")
        self._strict = config.strict
        self._debug_route = config.attn_debug_route
        self._layer_static = layer_static_from_impl(self)
        # Routes are a function of (layer static attributes, batch class,
        # capturing) only.
        self._routes: dict[tuple[BatchClass | None, bool], Route] = {}
        self._paths = {P1E: EncoderOnlyPath(self._layer_static)}
        self._paged = PagedPath(self._layer_static)
        # hipDNN sink_token views of self.sinks, per Q dtype (see _sink).
        self._sinks_hipdnn: dict[torch.dtype, torch.Tensor] = {}

    def _sink(self, dtype: torch.dtype) -> torch.Tensor | None:
        """The layer's attention sinks as hipDNN's ``sink_token``: vLLM's
        ``sinks`` ``[Hq]`` (a Parameter in the model dtype) viewed as
        ``[1, Hq, 1, 1]`` with strides ``[Hq, 1, 1, 1]``, in the Q dtype the
        rocKE engines load it in. None for a layer without sinks.

        Zero-copy when ``sinks`` is already a contiguous tensor of the Q dtype;
        otherwise it is converted once (at the first eager call, after the
        weights are loaded) and the copy is kept here, so the address bound by
        a captured graph stays valid for every replay."""
        sinks = self.sinks
        if sinks is None:
            return None
        view = self._sinks_hipdnn.get(dtype)
        if view is None:
            src = sinks.detach()
            if src.dtype != dtype or not src.is_contiguous():
                src = src.to(dtype).contiguous()
            view = src.view(1, self._layer_static.num_heads, 1, 1)
            self._sinks_hipdnn[dtype] = view
        return view

    def _route(self, batch_class: BatchClass | None, capturing: bool) -> Route:
        route = self._routes.get((batch_class, capturing))
        if route is None:
            route = select(self._layer_static, batch_class, self._attention_enabled, capturing)
            self._routes[(batch_class, capturing)] = route
            if self._debug_route:
                logger.info(
                    "hipdnn_vllm attention route: layer=%s batch=%s capturing=%s path=%s reason=%s",
                    self._layer_static,
                    batch_class.value if batch_class else None,
                    capturing,
                    route.path,
                    route.reason,
                )
        return route

    def _fallback(self, op: str, decline: Decline, *args):
        stats.record(op, stats.ROUTE_FALLBACK, decline.reason)
        if self._strict and self._attention_enabled:
            raise DeclineError(op, decline)
        return super().forward(*args)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        args = (
            layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale, output_block_scale,
        )
        if attn_metadata is None:
            # Profiling run: no batch, nothing to route or count.
            return super().forward(*args)
        info = getattr(attn_metadata, "hipdnn", None)
        batch_class = info.batch_class if info is not None else None
        capturing = torch.cuda.is_current_stream_capturing()
        route_class = batch_class
        if capturing and batch_class is not None and batch_class is not BatchClass.ENCODER_ONLY:
            route_class = _captured_class(batch_class, info.paged is not None and info.paged.capture_from_metadata)
        route = self._route(route_class, capturing)
        op = f"attention.{batch_class.value}" if batch_class else "attention"
        if route.path is None:
            return self._fallback(op, route.decline, *args)
        if output_scale is not None or output_block_scale is not None:
            return self._fallback(op, Decline.feature("output_quant"), *args)
        try:
            if route.path in (P2A, P2B):
                decline = self._run_paged(route.path, query, kv_cache, attn_metadata, output, capturing)
            else:
                sink = self._sink(query.dtype)
                decline = self._paths[route.path].run(query, key, value, output, info, sink)
        except RuntimeError as exc:
            parsed = parse_op_error(exc)
            if parsed is None:
                raise
            decline = parsed[1]  # launch failure after a successful build
        if decline is not None:
            return self._fallback(op, decline, *args)
        stats.record(op, stats.ROUTE_HIPDNN)
        return output

    def _run_paged(self, path, query, kv_cache, attn_metadata, output, capturing) -> Decline | None:
        info = attn_metadata.hipdnn
        # Batch-level TRITON_ATTN features the paged graph has no mapping for.
        if getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None:
            return Decline.feature("mm_prefix")
        if getattr(attn_metadata, "rswa_window", None) is not None:
            return Decline.feature("rswa")
        if info.paged is None:
            return Decline.feature("paged_metadata")
        paged = self._paged
        if not paged.ready(kv_cache):
            if capturing:
                # No plan build under capture: this graph keeps TRITON_ATTN.
                return Decline(Stage.CAPTURE_INVARIANT, "not_prepared")
            # First eager call for this KV cache (vLLM profiles with a small
            # cache before allocating the real one).
            paged.prepare(query, kv_cache, output, info, self._sink(query.dtype))
            if self._debug_route:
                logger.info(
                    "hipdnn_vllm paged attention: layer=%s p2a=%s p2b=%s plans p2a=%s p2b=%s",
                    self._layer_static,
                    paged.status(P2A).reason if paged.status(P2A) else paged.engine(P2A),
                    paged.status(P2B).reason if paged.status(P2B) else paged.engine(P2B),
                    paged.plan_sizes(P2A),
                    paged.plan_sizes(P2B),
                )
        if path == P2A and paged.status(P2A) is not None and paged.status(P2B) is None:
            path = P2B  # P2b serves every decoder batch, uniform decode included
        descriptor = _capture_descriptor() if capturing else None
        return paged.run(path, query, kv_cache, output, info, capturing, descriptor)


class HipdnnAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "HIPDNN"

    @staticmethod
    def get_impl_cls() -> type[HipdnnAttentionImpl]:
        return HipdnnAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[HipdnnAttentionMetadataBuilder]:
        return HipdnnAttentionMetadataBuilder
