# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Specification + host reference for the SDPA padding mask (ticket 10).

Given hipDNN's per-sequence valid lengths (SEQ_LEN_Q / SEQ_LEN_KV, normalized
by :mod:`kernels.common._sdpa_seqlen`), this says which key columns contribute
to each query's softmax and which query rows carry a real output. It is the
mask that sits ON TOP of the arbitrary-length buffer clipping (ticket 8):
clipping makes a partial tile safe to *read*; this makes the padded positions
not *count*. A key read as 0 by a clipped load still scores 0 and would take
softmax weight, so the mask below is a separate, still-required step.

Two pieces, matching where each lands in the shell (``fmha_sdpa_fwd.py``, ST4):

1. KV score mask -- the shell's ``extra_mask_predicate(b, k_idx)`` hook. Keep
   key column ``k`` iff ``k < len_kv``; masked columns are driven to ``neg_inf``
   by the hook's existing ``b.select(keep, score, neg_inf)``. The IR emitter is
   a one-liner mirroring :func:`rocke.helpers.attention.causal_mask` --
   ``b.cmp_lt(key_pos, len_kv)`` -- added to the shell with its byte-identity
   C++ mirror when the shell lands. Until then this module is the host-side
   reference that emitter must match.

2. Query-row output validity -- the epilogue. A row with ``q >= len_q`` is
   padding and is written as ``O = 0`` (LSE ``-inf``). A row left with no valid
   key (e.g. ``len_kv == 0``) is fully masked and reaches ``O = 0`` through the
   epilogue's existing ``l == 0`` guard; no extra kernel logic is needed for it.

Causal / sliding-window masking is NOT here -- the shell already applies it
through its own ``mask_mode``; padding composes with it. These predicates are
pure Python so the padding semantics are locked and cross-checked against
``dense_attention_reference`` on the host, with no GPU.
"""

from __future__ import annotations


def keep_key(key_pos, len_kv):
    """KV padding score mask: keep key column ``key_pos`` iff it is valid.

    ``key_pos`` is the key column index and ``len_kv`` the sequence's valid key
    count; accepts scalars or numpy arrays. Emitter form when wired into the
    shell's ``extra_mask_predicate`` hook: ``b.cmp_lt(key_pos, len_kv)``.
    """
    return key_pos < len_kv


def row_is_valid(query_pos, len_q):
    """Query-row output validity: a row at or past ``len_q`` is padding.

    Returns ``False`` for padded rows, which the epilogue writes as ``O = 0``
    (LSE ``-inf``). Accepts scalars or numpy arrays.
    """
    return query_pos < len_q
