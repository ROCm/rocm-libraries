# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Path P1e: encoder-only attention through hipDNN SDPA.

vLLM hands an encoder-only layer packed ``query`` ``[T, Hq, D]``, ``key`` /
``value`` ``[T, Hkv, D]`` (usually strided views of the fused QKV projection)
and an ``output`` ``[T, Hq, Dv]``. Every request is whole and bidirectional.
For each run of consecutive equal-length requests (``HipdnnBatchInfo.
encoder_groups``) this path describes the run as a dense non-causal graph over
zero-copy BSHD views permuted to BHSD (``[n, H, L, D]``, token stride and head
stride unchanged, batch stride ``L * token_stride``) and writes the result
straight into vLLM's output buffer.

All runs are probed before any runs execute, so a decline leaves nothing
half-written and the caller falls back for the whole layer call. Probing hits
the plan cache after the first call per signature; declined signatures are
cached too.
"""

from __future__ import annotations

from typing import Any

import torch

from ..declines import Decline, parse_reason
from .metadata import HipdnnBatchInfo
from .routing import LayerStatic

__all__ = ["EncoderOnlyPath"]


class EncoderOnlyPath:
    def __init__(self, layer: LayerStatic):
        self._layer = layer
        # vLLM's encoder sliding window is (W-1, W-1), (-1, -1) when unset: the
        # hipDNN diagonal-band bounds as is.
        self._left, self._right = (int(b) for b in layer.sliding_window)
        self._C: Any = None

    def _c(self) -> Any:
        if self._C is None:
            from .. import _bootstrap

            _bootstrap.load()
            from .. import _C

            self._C = _C
        return self._C

    def run(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        info: HipdnnBatchInfo,
        sink: torch.Tensor | None = None,
    ) -> Decline | None:
        """Write attention for every request into ``output``. Returns the
        decline (nothing written) when hipDNN cannot serve the batch. ``sink``
        is the layer's ``sink_token`` ``[1, Hq, 1, 1]`` (or None), bound by
        every run's graph."""
        layer = self._layer
        n_tok = info.num_actual_tokens
        hq, hkv = layer.num_heads, layer.num_kv_heads
        q = query[:n_tok].view(n_tok, hq, -1)
        k = key[:n_tok].view(n_tok, hkv, -1)
        v = value[:n_tok].view(n_tok, hkv, -1)
        o = output[:n_tok].view(n_tok, hq, -1)
        C = self._c()
        calls = []
        for first, count, length in info.encoder_groups:
            end = first + count * length
            views = tuple(
                t[first:end].view(count, length, t.shape[1], t.shape[2]).transpose(1, 2)
                for t in (q, k, v, o)
            )
            reason, detail = C.sdpa_fwd_probe(
                *views,
                layer.scale,
                causal=False,
                left_bound=self._left,
                right_bound=self._right,
                sink=sink,
            )
            if reason:
                d = parse_reason(reason)
                return Decline(d.stage, d.detail, detail)
            calls.append(views)
        for qv, kv, vv, ov in calls:
            torch.ops.hipdnn_vllm.sdpa_fwd(
                qv,
                kv,
                vv,
                ov,
                layer.scale,
                causal=False,
                left_bound=self._left,
                right_bound=self._right,
                sink=sink,
            )
        return None
