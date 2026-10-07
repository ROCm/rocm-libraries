# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side adapter for hipDNN per-sequence padding lengths (SEQ_LEN_Q/KV).

hipDNN supplies padding lengths as int32 tensors of shape ``[B, 1, 1, 1]`` --
one valid (unpadded) count per batch item. The SDPA stride shell
(``fmha_sdpa_fwd.py``, SH-01) reads flat per-batch length arrays. This module is
the CPU-side validation + translation between the two, run before the kernel
launches; it emits no kernel IR.

The shell's exact kernarg contract is not yet frozen (ST4). The returned
per-batch ``int32`` arrays are the stable part of the interface; the call
signature may be adjusted when the shell lands.
"""

from __future__ import annotations

import numpy as np


def _validate_lengths(lengths, *, batch, s_max, name):
    arr = np.asarray(lengths)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name}: lengths must be an integer dtype, got {arr.dtype}")
    if arr.size != batch:
        raise ValueError(f"{name}: expected {batch} per-batch lengths, got {arr.size}")
    flat = arr.reshape(batch).astype(np.int32, copy=False)
    if flat.min(initial=0) < 0 or flat.max(initial=0) > s_max:
        raise ValueError(
            f"{name}: every length must satisfy 0 <= len <= {s_max}; "
            f"got min={int(flat.min())}, max={int(flat.max())}"
        )
    return np.ascontiguousarray(flat, dtype=np.int32)


def sdpa_seqlens(seqlen_q, seqlen_kv, *, batch, s_q_max, s_kv_max):
    """Validate hipDNN SEQ_LEN_Q / SEQ_LEN_KV and return flat per-batch arrays.

    ``seqlen_q`` / ``seqlen_kv`` are array-likes of ``batch`` integer counts
    (hipDNN supplies shape ``[B, 1, 1, 1]``); a count is the valid (unpadded)
    length for that batch item. Returns ``(len_q, len_kv)`` as contiguous
    ``int32`` arrays of shape ``(batch,)`` for the shell to read.

    Raises ``ValueError`` if a tensor is not integer-typed, does not hold
    exactly ``batch`` counts, or carries a length outside ``[0, S_max]``.
    """
    len_q = _validate_lengths(seqlen_q, batch=batch, s_max=s_q_max, name="SEQ_LEN_Q")
    len_kv = _validate_lengths(
        seqlen_kv, batch=batch, s_max=s_kv_max, name="SEQ_LEN_KV"
    )
    return len_q, len_kv
