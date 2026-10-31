# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Attention route selection.

The route depends only on capture-invariant inputs: the layer's static
attributes, the batch class and whether the stream is capturing, never on
tensor contents. Under capture the caller passes the class of the graph being
captured (see ``HipdnnAttentionImpl``), not the dummy batch's.

Paths:

* ``P1E`` (encoder-only): dense non-causal hipDNN SDPA per run of equal-length
  requests, eager only. Whether an engine accepts the signature is probed per
  call; a decline falls back with the build-stage reason.
* ``P2A`` (paged uniform decode, q_len 1 or k+1): ragged Q over the paged
  TRITON_ATTN cache, ``max_seq_len_q`` = the decode query length.
* ``P2B`` (ragged + paged: mixed, chunked prefill, prefill): the same graph with
  ``max_seq_len_q`` up to the scheduler's token budget. It serves every
  decoder batch, so a uniform-decode batch whose P2A plan is declined uses it.
  Both paths size each plan per batch (:mod:`.paged`).

Whether an engine accepts a P2 signature is probed once per layer signature and
KV cache at the first eager call (:mod:`.paged`); the impl applies that result.
Attention sinks are part of that signature (the graph's ``sink_token``), so the
engine decides; a decline falls back with the build-stage reason.

Fallback reasons, most specific first:

1. ``DISABLED`` when ``attention`` is not in ``HIPDNN_VLLM_OPS``.
2. ``UNSUPPORTED_FEATURE:<feature>`` for layer features with no hipDNN mapping
   (softcap, alibi, fp8_kv, chunk_lookback, encoder_decoder). No build
   is attempted.
3. ``CAPTURE_INVARIANT`` for encoder-only attention under HIP-graph capture:
   P1e is per-request and eager only, so a captured graph keeps TRITON_ATTN.
4. ``UNSUPPORTED_FEATURE:non_causal`` for a decoder layer handed non-causal
   (encoder-only) metadata, and ``UNSUPPORTED_FEATURE:metadata`` when the
   metadata carries no HIPDNN batch info.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..declines import Decline, Stage
from .metadata import BatchClass

__all__ = ["LayerStatic", "P1E", "P2A", "P2B", "Route", "select", "layer_static_from_impl"]

# Path identifiers carried by Route.path.
P1E = "p1e"
P2A = "p2a"
P2B = "p2b"

# vLLM AttentionType values (vllm/v1/attention/backend.py). Kept as strings so
# routing stays importable and testable without vLLM.
DECODER = "decoder"
ENCODER = "encoder"
ENCODER_ONLY = "encoder_only"
ENCODER_DECODER = "encoder_decoder"


@dataclass(frozen=True)
class LayerStatic:
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    sliding_window: tuple[int, int]
    logits_soft_cap: float
    has_alibi: bool
    has_sinks: bool
    kv_cache_dtype: str
    attn_type: str
    # Chunked local attention with lookback (-1 = off). Only the inherited
    # Triton kernel implements it; hipDNN has no such mask.
    chunk_lookback: int = -1


@dataclass(frozen=True)
class Route:
    path: Any  # a path id (P1E), or None for fallback
    decline: Decline | None

    @property
    def reason(self) -> str:
        return self.decline.reason if self.decline else ""


def _attn_type_name(attn_type: Any) -> str:
    return str(getattr(attn_type, "value", attn_type)).lower()


def layer_static_from_impl(impl: Any) -> LayerStatic:
    """Capture a TritonAttentionImpl's static attributes after its __init__."""
    return LayerStatic(
        num_heads=int(impl.num_heads),
        num_kv_heads=int(impl.num_kv_heads),
        head_size=int(impl.head_size),
        scale=float(impl.scale),
        sliding_window=tuple(impl.sliding_window),
        logits_soft_cap=float(impl.logits_soft_cap or 0.0),
        has_alibi=impl.alibi_slopes is not None,
        has_sinks=impl.sinks is not None,
        kv_cache_dtype=str(impl.kv_cache_dtype),
        attn_type=_attn_type_name(impl.attn_type),
        chunk_lookback=int(getattr(impl, "chunk_lookback", -1)),
    )


def prefilter(layer: LayerStatic) -> Decline | None:
    """Layer features hipDNN cannot express today; no build is attempted."""
    if layer.attn_type == ENCODER_DECODER or layer.attn_type == ENCODER:
        return Decline.feature("encoder_decoder")
    if layer.chunk_lookback > -1:
        return Decline.feature("chunk_lookback")
    if layer.logits_soft_cap > 0:
        return Decline.feature("softcap")
    if layer.has_alibi:
        return Decline.feature("alibi")
    if layer.kv_cache_dtype not in ("auto", "bfloat16", "float16", "half"):
        return Decline.feature("fp8_kv" if layer.kv_cache_dtype.startswith("fp8") else "quant_kv")
    return None


def select(
    layer: LayerStatic,
    batch_class: BatchClass | None,
    attention_enabled: bool,
    capturing: bool = False,
) -> Route:
    if not attention_enabled:
        return Route(None, Decline(Stage.DISABLED))
    pre = prefilter(layer)
    if pre is not None:
        return Route(None, pre)
    if layer.attn_type == ENCODER_ONLY:
        if batch_class is not BatchClass.ENCODER_ONLY:
            # The metadata did not come from an encoder-only builder.
            return Route(None, Decline.feature("encoder_only_metadata"))
        if capturing:
            return Route(None, Decline(Stage.CAPTURE_INVARIANT, "p1e_eager_only"))
        return Route(P1E, None)
    if batch_class is None:
        return Route(None, Decline.feature("metadata"))
    if batch_class is BatchClass.ENCODER_ONLY:
        return Route(None, Decline.feature("non_causal"))
    if batch_class is BatchClass.UNIFORM_DECODE:
        return Route(P2A, None)
    return Route(P2B, None)
