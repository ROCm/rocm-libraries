# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``register_fake`` (meta) implementations for every ``torch.ops.hipdnn_vllm`` op,
so the ops trace under torch.compile / Dynamo. Imported by the bootstrap right
after ``_C`` registers the ops."""

from __future__ import annotations

import torch


@torch.library.register_fake("hipdnn_vllm::rms_norm")
def _rms_norm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.register_fake("hipdnn_vllm::rms_norm_rows")
def _rms_norm_rows_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float, strict: bool
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.register_fake("hipdnn_vllm::fused_add_rms_norm_rows")
def _fused_add_rms_norm_rows_fake(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, strict: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=x.dtype, device=x.device),
        torch.empty(x.shape, dtype=x.dtype, device=x.device),
    )


@torch.library.register_fake("hipdnn_vllm::linear")
def _linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


@torch.library.register_fake("hipdnn_vllm::sdpa_fwd")
def _sdpa_fwd_fake(
    q, k, v, out, scale, causal=False, bottom_right=True, left_bound=-1, right_bound=-1,
    ragged_offset_q=None, seq_len_q=None, max_seq_len_q=0, ragged_offset_kv=None,
    page_table=None, seq_len_kv=None, max_seq_len_kv=0, sink=None,
) -> None:
    # Writes `out` in place; nothing is returned.
    return None


@torch.library.register_fake("hipdnn_vllm::encoder_attention")
def _encoder_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]))


@torch.library.register_fake("hipdnn_vllm::layer_norm")
def _layer_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    strict: bool,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@torch.library.register_fake("hipdnn_vllm::conv_fwd")
def _conv_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
    strict: bool,
) -> torch.Tensor:
    out = [x.shape[0], weight.shape[0]]
    for i, (s, p, d) in enumerate(zip(stride, padding, dilation)):
        k = weight.shape[i + 2]
        out.append((x.shape[i + 2] + 2 * p - d * (k - 1) - 1) // s + 1)
    return torch.empty(out, dtype=x.dtype, device=x.device)
