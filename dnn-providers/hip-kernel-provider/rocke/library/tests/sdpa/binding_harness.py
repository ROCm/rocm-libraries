# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map ``cases.py`` harness tensors onto ``sdpa_unified_bind``'s tensor contract.

Shared by the CPU data-movement test and the GPU numeric test. Torch is
imported lazily.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bind_tensors", "outputs"]


def bind_tensors(case, dev, req):
    """The binding's ``tensors`` for ``case``, from ``helpers.to_torch`` output.

    A ragged operand is its packed ``[T, H, D]`` buffer with token offsets
    (``request_from_case`` sets ``ragged_offset_multiplier = H * D``). Stats
    are allocated in the request's ``(B, H_q, S_q, 1)`` shape.
    """
    import torch

    tensors = {n: dev[n] for n in ("q", "k", "v", "o")}
    for key, src in (
        ("seq_len_q", "seq_len_q"),
        ("seq_len_kv", "seq_len_kv"),
        ("ragged_offset_q", "q_offsets"),
        ("ragged_offset_k", "kv_offsets"),
    ):
        if dev[src] is not None:
            tensors[key] = dev[src]
    if req.stats is not None:
        tensors["stats"] = torch.zeros(
            req.stats.dims, dtype=torch.float32, device=dev["q"].device
        )
    return tensors


def outputs(case, tensors, req):
    """O and LSE as host arrays shaped like the case oracle's."""
    o = tensors["o"].double().cpu().numpy()
    if req.stats is None:
        return o, None
    stats = tensors["stats"][..., 0].double().cpu().numpy()
    if case.length_mode != "ragged":
        return o, stats
    q_lens, _ = case.lens()
    return o, np.concatenate([stats[b, :, :n].T for b, n in enumerate(q_lens)])
