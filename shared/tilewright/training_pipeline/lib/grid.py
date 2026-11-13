# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The 96-cell categorical grid over GEMM problem space; one model per cell.

Tiers (inclusive upper bounds, identical to the engine's routing):
    M, N : Tiny (<=32)  Small (<=128)  Mid (<=512)  Large (>512)
    K    : TinyK (<=32) MidK (<=512)   LargeK (>512)
    B    : Bnone (B==1) Bany (B>1)

Cell label format:  '<Mtier>|<Ntier>|<Ktier>|<Btier>'
"""
from __future__ import annotations

from typing import List


def m_tier(M: int) -> str:
    if M <= 32:
        return "Tiny"
    if M <= 128:
        return "Small"
    if M <= 512:
        return "Mid"
    return "Large"


def n_tier(N: int) -> str:
    if N <= 32:
        return "Tiny"
    if N <= 128:
        return "Small"
    if N <= 512:
        return "Mid"
    return "Large"


def k_tier(K: int) -> str:
    if K <= 32:
        return "TinyK"
    if K <= 512:
        return "MidK"
    return "LargeK"


def b_tier(B: int) -> str:
    return "Bnone" if B == 1 else "Bany"


def cell_key(M: int, N: int, K: int, B: int) -> str:
    return f"{m_tier(M)}|{n_tier(N)}|{k_tier(K)}|{b_tier(B)}"


M_TIERS = ("Tiny", "Small", "Mid", "Large")
K_TIERS_ALL = ("TinyK", "MidK", "LargeK")
B_TIERS = ("Bnone", "Bany")


def all_cell_labels() -> List[str]:
    return [
        f"{m}|{n}|{k}|{b}"
        for m in M_TIERS
        for n in M_TIERS
        for k in K_TIERS_ALL
        for b in B_TIERS
    ]


# Sampling bounds for shape synthesis. The tier classifiers above are
# unbounded above; these ranges are only where stage01 samples, and they have
# to cover the operating range of real workloads or the models extrapolate on
# real GEMMs they never saw. The feasibility filter in lib/shapes.py keeps the
# corners of these boxes benchable.
TIER_RANGES_MN = {
    "Tiny": (1, 32),
    "Small": (33, 128),
    "Mid": (129, 512),
    "Large": (513, 524288),
}
TIER_RANGES_K = {
    "TinyK": (1, 32),
    "MidK": (33, 512),
    "LargeK": (513, 131072),
}
# Batch counts for the Bany tier. Large batch counts occur with small
# matrices, so synthesis draws B from the values that keep each sampled
# (M, N, K) benchable rather than uniformly from this list.
BANY_VALUES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
