# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Case matrix and fixture scope of the coordinate-codeword placement probe (P10).

The probe launches a dense kernel on zero Q, so the softmax row of query ``q`` is
uniform over its legal keys, and on a V whose value at ``(batch, KV head, key,
dimension)`` encodes that coordinate. Every stored output then names the
coordinate it belongs to, and a guarded output allocation shows whether anything
was written outside the logical tensor. There is intentionally no attention
implementation here: each expectation is a mean of codeword values over a
legal-key interval.

This module holds what does not depend on dispatch: the cases and the bounds
inside which the codeword is exact.
"""

from __future__ import annotations

from dataclasses import replace

from ._analytical import UNIFORM_QUICK_CASES, UniformCase, legal_interval

# The codeword is ``20 + (b + 1) / 4`` in lane 0, ``26 + (h + 1) / 4`` in lane 1,
# ``-k / 2`` in lane 2, a key-parity value in lane 3 and ``(d + 1) / 16`` in the
# lanes from 4 on. Inside these bounds every entry is both an FP16 and a BF16
# value, and the lanes take disjoint value ranges. BF16's eight significant bits
# are what stops lane 2 at key 255.
P10_MAX_BATCH = 2
P10_MAX_KV_HEADS = 4
P10_MAX_SEQLEN_K = 256
P10_MAX_HEAD_DIM = 64
P10_COORDINATE_LANES = 4

# The uniform-probability lane's two specifications, at batch 1 and at batch 2.
# Batch placement is observable only at batch 2. These are grouped-query cases:
# with zero Q the query heads of one KV group store identical values, so they
# expose KV-head routing and the group map but not which head of a group an
# output was written to.
P10_GROUPED_CASES = tuple(
    replace(case, batch=batch) for case in UNIFORM_QUICK_CASES for batch in (1, 2)
)
# One query head per KV head makes every output head distinct. These cases own
# output-head placement.
P10_UNGROUPED_CASES = tuple(
    replace(
        case,
        batch=P10_MAX_BATCH,
        nhead_q=P10_MAX_KV_HEADS,
        nhead_k=P10_MAX_KV_HEADS,
    )
    for case in UNIFORM_QUICK_CASES
)
P10_QUICK_CASES = P10_GROUPED_CASES + P10_UNGROUPED_CASES


def require_p10_scope(case: UniformCase) -> None:
    """Refuse a case on which the codeword expectations are not defined.

    Two things put a case outside the scope. Past the bounds above a codeword
    entry would round on encoding, so the kernel would not read the values the
    expectations assume. And a query row without a legal key has no mean to
    expect; a window whose queries run past the keys produces such rows.
    """
    upper_bounds = (
        ("batch", case.batch, P10_MAX_BATCH),
        ("nhead_k", case.nhead_k, P10_MAX_KV_HEADS),
        ("seqlen_k", case.seqlen_k, P10_MAX_SEQLEN_K),
        ("head_dim", case.head_dim, P10_MAX_HEAD_DIM),
    )
    for name, value, limit in upper_bounds:
        if value > limit:
            raise ValueError(
                f"case {case.case_id} is outside the codeword scope - "
                f"{name}={value}, expected at most {limit}"
            )
    if case.head_dim < P10_COORDINATE_LANES:
        raise ValueError(
            f"case {case.case_id} is outside the codeword scope - "
            f"head_dim={case.head_dim}, expected at least {P10_COORDINATE_LANES} "
            f"for the coordinate lanes"
        )
    for query in range(case.seqlen_q):
        first, last = legal_interval(case, query)
        if last < first:
            raise ValueError(
                f"case {case.case_id} is outside the codeword scope - "
                f"query row {query} has no legal key, expected every row to have one"
            )
