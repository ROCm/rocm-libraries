# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent CPU FP32 decode references from rounded logical inputs."""

import math


def _reference(
    torch, q, k, v, lengths, window, *, features=None, table=None, emit_lse=False
):
    """Evaluate logical CPU rows independently of production page addressing.

    Keep softmax for ordinary/feature outputs and logsumexp-based weights for
    LSE outputs to preserve the original FP32 rounding paths.
    """
    output = torch.zeros_like(q, dtype=torch.float32)
    lse = (
        torch.full((*q.shape[:2], 1), float("-inf"), dtype=torch.float32)
        if emit_lse
        else None
    )
    ratio = q.shape[1] // k.shape[1 if table is None else 2]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        begin = max(0, length - window) if window else 0
        if table is None:
            keys = k[batch, :, begin:length].permute(1, 0, 2)
            values = v[batch, :, begin:length].permute(1, 0, 2)
        else:
            # The oracle receives the saved valid table, never live poisoned metadata.
            ids = table[batch].long()
            keys = k[ids].reshape(-1, k.shape[2], q.shape[2])[begin:length].float()
            values = v[ids].reshape(-1, v.shape[2], q.shape[2])[begin:length].float()
        for head in range(q.shape[1]):
            scores = (
                keys[:, head // ratio].float() @ q[batch, head].float()
            ) / math.sqrt(q.shape[2])
            if features is not None:
                if features["softcap"]:
                    scores = features["softcap"] * torch.tanh(
                        scores / features["softcap"]
                    )
                if features["alibi_slopes"] is not None:
                    positions = torch.arange(begin, length) - (length - 1)
                    scores += features["alibi_slopes"][head] * positions
                if features["qq_bias"] is not None:
                    # Decode has one query: bias only the last KV position.
                    scores[-1] += features["qq_bias"][0, 0]
            if features is not None and features["sinks"] is not None:
                weights = torch.cat(
                    (scores, features["sinks"][head].float().reshape(1))
                ).softmax(0)[:-1]
            elif emit_lse:
                normalizer = torch.logsumexp(scores, 0)
                weights = torch.exp(scores - normalizer)
                lse[batch, head, 0] = normalizer
            else:
                weights = scores.softmax(0)
            output[batch, head] = weights @ values[:, head // ratio].float()
    return (output, lse) if emit_lse else output


def _feature_reference(torch, q, k, v, lengths, window, features):
    """CPU FP32 SDPA including a zero-valued sink in the normalization."""
    return _reference(torch, q, k, v, lengths, window, features=features)


def reference(torch, q, k, v, table, lengths, page, window):
    """Gather only from a known-valid table; return output and logsumexp."""
    return _reference(torch, q, k, v, lengths, window, table=table, emit_lse=True)
