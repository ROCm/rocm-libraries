# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side adapter for hipDNN per-sequence length tensors (SEQ_LEN_Q/KV).

hipDNN supplies per-sequence lengths as int32 tensors of shape ``[B, 1, 1, 1]``
-- one valid count per batch item. The same counts drive two layouts, and this
module is the CPU-side validation + translation for both, run before the kernel
launches; it emits no kernel IR:

* padded dense tensors (ticket 10) -- ``sdpa_seqlens`` returns the flat
  per-batch lengths the stride shell (``fmha_sdpa_fwd.py``, SH-01) masks
  against;
* ragged / THD packing (ticket 12) -- ``sdpa_cu_seqlens`` returns the
  cumulative ``cu_seqlens`` arrays the varlen kernels (e.g. ``fmha_varlen``)
  address with.

The shell's exact kernarg contract is not yet frozen (ST4). The returned
``int32`` arrays are the stable part of the interface; the call signatures may
be adjusted when the shell lands.
"""

from __future__ import annotations

import numpy as np


def _validate_lengths(lengths, *, batch, name, s_max=None):
    arr = np.asarray(lengths)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name}: lengths must be an integer dtype, got {arr.dtype}")
    if arr.size != batch:
        raise ValueError(f"{name}: expected {batch} per-batch lengths, got {arr.size}")
    flat = arr.reshape(batch).astype(np.int32, copy=False)
    if flat.min(initial=0) < 0:
        raise ValueError(
            f"{name}: every length must be >= 0; got min={int(flat.min())}"
        )
    if s_max is not None and flat.max(initial=0) > s_max:
        raise ValueError(
            f"{name}: every length must satisfy 0 <= len <= {s_max}; "
            f"got max={int(flat.max())}"
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


def _cumulative(lengths):
    # Exclusive prefix sum with a leading 0: cu[0] = 0, cu[-1] = total tokens.
    cu = np.zeros(lengths.size + 1, dtype=np.int32)
    np.cumsum(lengths, out=cu[1:])
    return cu


def sdpa_cu_seqlens(seqlen_q, seqlen_kv, *, batch):
    """Validate ragged per-sequence lengths and return cumulative ``cu_seqlens``.

    ``seqlen_q`` / ``seqlen_kv`` are array-likes of ``batch`` integer counts --
    hipDNN's ragged per-sequence lengths. Ragged tokens are packed with no
    padding, so there is no per-sequence upper bound. Returns
    ``(cu_seqlens_q, cu_seqlens_kv)`` as contiguous ``int32`` arrays of shape
    ``(batch + 1,)``: exclusive prefix sums with a leading 0, whose last element
    is the packed token total -- the form ``fmha_varlen`` addresses with.

    Raises ``ValueError`` if a tensor is not integer-typed, does not hold
    exactly ``batch`` counts, or carries a negative length.
    """
    len_q = _validate_lengths(seqlen_q, batch=batch, name="SEQ_LEN_Q")
    len_kv = _validate_lengths(seqlen_kv, batch=batch, name="SEQ_LEN_KV")
    return _cumulative(len_q), _cumulative(len_kv)
