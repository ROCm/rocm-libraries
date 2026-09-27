# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MMA soundness (gate 2) -- PURE calc, no matplotlib. Orthogonal to the addressing round-trip.

Every recorded MMA's CONSUMED operand encodings must be a sound, K-aligned pair against the canonical
machine (``operand_soundness`` + ``diagnose_k_match``, bundled as ``mma_pair_compatible``).
"""

from __future__ import annotations

from typing import Any


class MmaSoundnessError(RuntimeError):
    """Raised when a recorded MMA's consumed operands are not a sound, K-aligned pair (the sound MAC)."""


def verify_mma_soundness(pipeline: Any) -> int:
    """Gate 2: every recorded MMA's CONSUMED operand encodings must be sound MMA operands against the
    canonical machine (``operand_soundness``) AND share a K-distribution (``diagnose_k_match``) -- bundled
    as ``mma_pair_compatible``. This is the operand-correctness the addressing round-trip is BLIND to: a
    scrambled/duplicated operand K passes the address gate but is caught here. ``a_canon``/``b_canon`` are
    the trusted canonical layouts from the MMA definition; ``a_enc``/``b_enc`` are the kernel's own
    (interleaved) operands. Returns the count verified; raises on the first unsound MMA. Correctness SOT:
    ``docs/mma_is_machinery.md`` (the three-condition sound MAC).
    """
    from ..transforms import mma_pair_compatible

    verified = 0
    for op in pipeline.ops:
        if op.kind != "mma":
            continue
        d = mma_pair_compatible(op.a_enc, op.b_enc, a_canon=op.a_canon, b_canon=op.b_canon)
        if d.severity == "error":
            raise MmaSoundnessError(f"MMA op seq {op.seq}: {d.message}")
        verified += 1
    return verified
