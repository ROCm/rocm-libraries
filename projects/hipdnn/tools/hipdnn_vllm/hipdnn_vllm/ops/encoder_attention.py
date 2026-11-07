# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""vLLM ``MMEncoderAttention`` (ViT attention) on hipDNN.

:class:`HipdnnMMEncoderAttention` replaces vLLM's layer through
``CustomOp.register_oot``. On ROCm, vLLM dispatches ``forward_hip``, which
calls ``torch.ops.hipdnn_vllm.mm_encoder_attention``. That op is opaque to
torch.compile. Its eager body runs ``torch.ops.hipdnn_vllm.encoder_attention``
(dense non-causal hipDNN SDPA graphs, one batched graph when every image in
``cu_seqlens`` has the same length, else one launch per image with plans keyed
by length). If hipDNN declines, the body runs the parent's ``forward_hip``
instead. Fallback is decided inside the op because the route depends on the
image lengths in ``cu_seqlens``, which a traced graph cannot branch on.

The opaque op cannot take the layer module, so it finds a layer by a signature
string: the static attributes that the parent's non-FP8 ``forward_hip``
depends on. Every layer with the same signature computes the same function,
so any one of them can serve the fallback. FP8 layers carry per-layer scale
buffers, so they never enter the op: ``forward_hip`` sends them to the parent
directly (``UNSUPPORTED_FEATURE:fp8``).

``maybe_compute_seq_lens`` and ``maybe_recompute_cu_seqlens`` are inherited
on purpose. The op takes cu_seqlens as token offsets, which is exactly what
the parent returns for every non-FlashInfer ViT backend (FlashInfer does not
exist on ROCm). The fallback also receives the parent's own layout unchanged.
"""

import logging
import weakref
from typing import Optional

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention

from .. import _bootstrap, stats
from ..config import load_config
from ..declines import Decline, DeclineError, parse_op_error

__all__ = ["OP", "HipdnnMMEncoderAttention", "encoder_attention", "register_oot"]

OP = "encoder_attention"
_OOT_NAME = MMEncoderAttention.__name__

logger = logging.getLogger("hipdnn_vllm")

# signature -> a live layer with that signature (fallback target of the opaque op).
_LAYERS: "weakref.WeakValueDictionary[str, HipdnnMMEncoderAttention]" = weakref.WeakValueDictionary()


def encoder_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """``torch.ops.hipdnn_vllm.encoder_attention``: q/k/v are ``[B, S, H, D]``.
    Raises the op's decline error when hipDNN does not run it."""
    _bootstrap.load()
    return torch.ops.hipdnn_vllm.encoder_attention(q, k, v, cu_seqlens, float(scale))


@torch.library.custom_op("hipdnn_vllm::mm_encoder_attention", mutates_args=())
def _mm_encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
    signature: str,
) -> torch.Tensor:
    layer = _LAYERS.get(signature)
    if layer is None:
        raise RuntimeError(f"hipdnn_vllm: no live MMEncoderAttention layer with signature {signature!r}")
    return layer._forward_routed(query, key, value, cu_seqlens, max_seqlen, sequence_lengths)


@_mm_encoder_attention.register_fake
def _mm_encoder_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    max_seqlen: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
    signature: str,
) -> torch.Tensor:
    # The parent keeps a 3-D [B, S, H*D] input 3-D and a 4-D input 4-D.
    if query.dim() == 4:
        return query.new_empty((*query.shape[:3], value.shape[-1]))
    return query.new_empty(query.shape)


class HipdnnMMEncoderAttention(MMEncoderAttention):
    """MMEncoderAttention whose ROCm forward runs on hipDNN, falling back to the parent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hipdnn_strict = load_config().strict
        self._hipdnn_signature = "|".join(
            str(part)
            for part in (
                self.attn_backend.name,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                float(self.scale).hex(),
                self.dtype,
                self._fa_version,
            )
        )
        if not self.fp8_enabled:
            _LAYERS[self._hipdnn_signature] = self

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fp8_enabled:
            decline = Decline.feature("fp8")
            if self._hipdnn_strict:
                raise DeclineError(OP, decline)
            if not torch.compiler.is_compiling():
                stats.record(OP, stats.ROUTE_FALLBACK, decline.reason)
            return super().forward_hip(query, key, value, cu_seqlens, max_seqlen, sequence_lengths)
        return torch.ops.hipdnn_vllm.mm_encoder_attention(
            query, key, value, cu_seqlens, max_seqlen, sequence_lengths, self._hipdnn_signature
        )

    def _forward_routed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[torch.Tensor],
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Eager body of the opaque op: hipDNN, or the parent on a decline.
        The decline was already counted by ``_C``."""
        bsz, q_len = query.shape[:2]
        is_reshaped = query.dim() != 4
        q, k, v = self.view_qkv_to_4d(query, key, value, bsz, q_len, key.shape[1])
        try:
            out = encoder_attention(q, k, v, cu_seqlens, self.scale)
        except RuntimeError as exc:
            parsed = parse_op_error(exc)
            if parsed is None:
                raise
            if self._hipdnn_strict:
                raise DeclineError(*parsed) from exc
            return MMEncoderAttention.forward_hip(
                self, query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )
        return out.reshape(bsz, q_len, -1) if is_reshaped else out


def register_oot() -> None:
    """Install :class:`HipdnnMMEncoderAttention` for ``MMEncoderAttention``. Idempotent."""
    existing = op_registry_oot.get(_OOT_NAME)
    if existing is HipdnnMMEncoderAttention:
        return
    if existing is not None:
        logger.warning(
            "hipdnn_vllm: %s already has out-of-tree class %s; not overriding it",
            _OOT_NAME,
            existing.__qualname__,
        )
        return
    CustomOp.register_oot(name=_OOT_NAME)(HipdnnMMEncoderAttention)
