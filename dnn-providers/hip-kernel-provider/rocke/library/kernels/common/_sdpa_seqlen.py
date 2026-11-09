# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side adapter for hipDNN per-sequence length tensors (SEQ_LEN_Q/KV).

hipDNN supplies per-sequence lengths as int32 tensors of shape ``[B, 1, 1, 1]``
-- one valid count per batch item. The same counts drive two layouts, and this
module is the CPU-side validation + translation for both, run before the kernel
launches; it emits no kernel IR:

* padded dense tensors -- ``sdpa_seqlens`` returns the flat per-batch lengths the
  planned SDPA forward shell masks against;
* ragged / THD packing -- ``sdpa_cu_seqlens`` returns the cumulative
  ``cu_seqlens`` arrays the varlen kernels (e.g. ``fmha_varlen``) address with.

Scope of the ragged adapter: it assumes a FULLY PACKED layout (sequence ``b + 1``
starts exactly where ``b`` ends, with no per-sequence trailing padding), so the
offsets are the prefix sums of the lengths. hipDNN's general ragged contract
(RFC 0014) also allows a caller-supplied ragged_offset with trailing padding,
where the offset is NOT the length prefix sum; supporting that offset tensor is
deferred, and a caller that needs trailing padding must not use this adapter.
``sdpa_cu_seqlens`` enforces the packed assumption implicitly -- it only accepts
the lengths and derives the offsets from them.

The returned ``int32`` arrays are the stable part of the interface; the call
signatures may be adjusted when the shell lands.
"""

from __future__ import annotations

import numpy as np

_I32_MAX = int(np.iinfo(np.int32).max)


def _require_pairing(seqlen_q, seqlen_kv):
    """Q and KV lengths must be both set or both unset (the cu_seqlens pairing
    rule cuDNN frontend applies). A one-sided call is a caller error, not a
    silently valid half-masked launch, so reject it up front with a clear
    message instead of failing by accident later in the dtype check."""
    if (seqlen_q is None) != (seqlen_kv is None):
        raise ValueError("SEQ_LEN_Q and SEQ_LEN_KV must both be set or both unset")


def _validate_lengths(lengths, *, batch, name, s_max=None):
    arr = np.asarray(lengths)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name}: lengths must be an integer dtype, got {arr.dtype}")
    if arr.size != batch:
        raise ValueError(f"{name}: expected {batch} per-batch lengths, got {arr.size}")
    # Range-check on a widened int64 copy BEFORE narrowing to int32. Narrowing
    # first would wrap an out-of-range value (e.g. 2**32 + 3 -> 3) and then let
    # the wrapped value pass every check.
    wide = arr.reshape(batch).astype(np.int64)
    if wide.min(initial=0) < 0:
        raise ValueError(
            f"{name}: every length must be >= 0; got min={int(wide.min())}"
        )
    hi = s_max if s_max is not None else _I32_MAX
    if wide.max(initial=0) > hi:
        bound = (
            f"0 <= len <= {s_max}"
            if s_max is not None
            else f"len <= {_I32_MAX} (int32)"
        )
        raise ValueError(
            f"{name}: every length must satisfy {bound}; got max={int(wide.max())}"
        )
    return np.ascontiguousarray(wide, dtype=np.int32)


def sdpa_seqlens(seqlen_q=None, seqlen_kv=None, *, batch, s_q_max, s_kv_max):
    """Validate hipDNN SEQ_LEN_Q / SEQ_LEN_KV and return flat per-batch arrays.

    ``seqlen_q`` / ``seqlen_kv`` are array-likes of ``batch`` integer counts
    (hipDNN supplies shape ``[B, 1, 1, 1]``); a count is the valid (unpadded)
    length for that batch item. Both must be set or both unset: unset (the
    default) means no padding and returns ``(None, None)``. When set, returns
    ``(len_q, len_kv)`` as contiguous ``int32`` arrays of shape ``(batch,)`` for
    the shell to read.

    Raises ``ValueError`` if only one of the pair is set, if a tensor is not
    integer-typed, does not hold exactly ``batch`` counts, or carries a length
    outside ``[0, S_max]``.
    """
    _require_pairing(seqlen_q, seqlen_kv)
    if seqlen_q is None:
        return None, None
    len_q = _validate_lengths(seqlen_q, batch=batch, s_max=s_q_max, name="SEQ_LEN_Q")
    len_kv = _validate_lengths(
        seqlen_kv, batch=batch, s_max=s_kv_max, name="SEQ_LEN_KV"
    )
    return len_q, len_kv


def _cumulative(lengths, *, name):
    # Exclusive prefix sum with a leading 0: cu[0] = 0, cu[-1] = total tokens.
    # Accumulate in int64: the ragged path has no per-sequence upper bound, so an
    # int32 running sum could wrap negative on a large packed total with no error.
    cu = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths.astype(np.int64), out=cu[1:])
    total = int(cu[-1])
    if total > _I32_MAX:
        # The kernel addresses a token as an int32 element offset (total * H * D),
        # so even this token-count bound is optimistic; the shell tightens it to
        # the real element offset once it knows H and D.
        raise ValueError(
            f"{name}: packed token total {total} exceeds int32 ({_I32_MAX})"
        )
    return np.ascontiguousarray(cu, dtype=np.int32)


def sdpa_cu_seqlens(seqlen_q=None, seqlen_kv=None, *, batch):
    """Validate PACKED ragged per-sequence lengths and return cumulative
    ``cu_seqlens``.

    ``seqlen_q`` / ``seqlen_kv`` are array-likes of ``batch`` integer counts --
    hipDNN's ragged per-sequence lengths. This adapter assumes a fully packed
    layout (no trailing padding, see the module docstring), so the offsets are
    the length prefix sums. Both lengths must be set or both unset: unset (the
    default) returns ``(None, None)``. When set, returns
    ``(cu_seqlens_q, cu_seqlens_kv)`` as contiguous ``int32`` arrays of shape
    ``(batch + 1,)``: exclusive prefix sums with a leading 0, whose last element
    is the packed token total -- the form ``fmha_varlen`` addresses with.

    Raises ``ValueError`` if only one of the pair is set, if a tensor is not
    integer-typed, does not hold exactly ``batch`` counts, carries a negative or
    above-int32 length, or packs a token total that overflows int32.
    """
    _require_pairing(seqlen_q, seqlen_kv)
    if seqlen_q is None:
        return None, None
    len_q = _validate_lengths(seqlen_q, batch=batch, name="SEQ_LEN_Q")
    len_kv = _validate_lengths(seqlen_kv, batch=batch, name="SEQ_LEN_KV")
    return _cumulative(len_q, name="SEQ_LEN_Q"), _cumulative(len_kv, name="SEQ_LEN_KV")
