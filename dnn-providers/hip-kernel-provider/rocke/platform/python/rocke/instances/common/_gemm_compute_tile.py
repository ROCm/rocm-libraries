# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-warp MFMA fragment grid for a staged UniversalGemm K tile."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ...core.ir import IRBuilder, Value


def accumulate_mfma_tile(
    b: IRBuilder,
    *,
    k_atoms: int,
    m_atoms: int,
    n_atoms: int,
    accs: Sequence[Value],
    read_a: Callable[[int, int], Value],
    read_b: Callable[[int, int], Value],
    mma: Callable[[Value, Value, Value], Value],
    prioritize_clusters: bool,
) -> list[Value]:
    """Accumulate one staged K tile using fragment readers bound to its slot.

    The readers own lane and LDS addressing; this routine owns reuse across
    atom rows/columns and MMA order. Priority hints do not publish LDS data.
    """
    out = list(accs)
    for kk in range(k_atoms):
        a_rows = [read_a(kk, mi) for mi in range(m_atoms)]
        b_cols = [read_b(kk, ni) for ni in range(n_atoms)]
        for mi, a in enumerate(a_rows):
            if prioritize_clusters:
                b.s_setprio(1)
            for ni, bb in enumerate(b_cols):
                flat = mi * n_atoms + ni
                out[flat] = mma(a, bb, out[flat])
            if prioritize_clusters:
                b.s_setprio(0)
                b.sched_barrier(0)
    return out
