# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent CPU FP32 decode references from rounded logical inputs."""

import math


def _reference(torch, q, k, v, lengths, window):
    output = torch.zeros_like(q, dtype=torch.float32)
    ratio = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        begin = max(0, length - window) if window else 0
        for head in range(q.shape[1]):
            keys = k[batch, head // ratio, begin:length].float()
            values = v[batch, head // ratio, begin:length].float()
            scores = (keys @ q[batch, head].float()) / math.sqrt(q.shape[2])
            output[batch, head] = scores.softmax(0) @ values
    return output


def _feature_reference(torch, q, k, v, lengths, window, features):
    """CPU FP32 SDPA from rounded logical inputs, including a zero-valued sink."""
    output = torch.zeros_like(q, dtype=torch.float32)
    ratio = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        begin = max(0, length - window) if window else 0
        for head in range(q.shape[1]):
            scores = (
                k[batch, head // ratio, begin:length].float() @ q[batch, head].float()
            ) / math.sqrt(q.shape[2])
            if features["softcap"]:
                scores = features["softcap"] * torch.tanh(scores / features["softcap"])
            if features["alibi_slopes"] is not None:
                positions = torch.arange(begin, length) - (length - 1)
                scores += features["alibi_slopes"][head] * positions
            if features["qq_bias"] is not None:
                # Decode has one query: only the last KV position is query-to-query.
                scores[-1] += features["qq_bias"][0, 0]
            if features["sinks"] is not None:
                weights = torch.cat(
                    (scores, features["sinks"][head].float().reshape(1))
                ).softmax(0)[:-1]
            else:
                weights = scores.softmax(0)
            output[batch, head] = (
                weights @ v[batch, head // ratio, begin:length].float()
            )
    return output


def reference(torch, q, k, v, table, lengths, page, window):
    """Independent CPU FP32 softmax from the rounded input tensors."""
    batch, heads, dim = q.shape
    ratio = heads // k.shape[2]
    output = torch.zeros((batch, heads, dim), dtype=torch.float32)
    lse = torch.full((batch, heads, 1), float("-inf"), dtype=torch.float32)
    for seq, length in enumerate(lengths):
        begin = max(0, length - window) if window else 0
        if begin == length:
            continue
        keys = k[table[seq].long()].reshape(-1, k.shape[2], dim)[begin:length].float()
        values = v[table[seq].long()].reshape(-1, v.shape[2], dim)[begin:length].float()
        for head in range(heads):
            scores = (keys[:, head // ratio] @ q[seq, head].float()) / math.sqrt(dim)
            normalizer = torch.logsumexp(scores, 0)
            output[seq, head] = (
                torch.exp(scores - normalizer) @ values[:, head // ratio]
            )
            lse[seq, head, 0] = normalizer
    return output, lse
