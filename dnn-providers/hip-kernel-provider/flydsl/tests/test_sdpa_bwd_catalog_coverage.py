# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Every head dim the dK/dV and dQ tables serve together has a prep row.

The split engine runs a prep, a dK/dV and a dQ object per graph and declines a graph any of the three cannot serve
(``servesHeadDims`` in ``FlydslSdpaBwdSplitNative.cpp``): an exact row serves dims equal to its tile, a padded one the
dims above its floor and at most its tile, and prep reads the V width only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _instances as instances  # noqa: E402
from generators._sdpa_bwd_spec import GFX950, rung_below  # noqa: E402

GFX11 = instances.SDPA_BWD_GFX11_TARGET
MAX_HEAD_DIM = 512
TABLES = {
    GFX950: instances.SDPA_BWD_TABLES,
    GFX11: instances.SDPA_BWD_GFX11_TABLES,
}


def served(row: tuple, arch: str) -> range:
    """The head dims a table row serves on both sides of the pair (columns: dtype, tile, cls, dform, dropout, padded, floor)."""
    dtype, tile, _, _, _, padded, floor = row[:7]
    if not padded:
        return range(tile, tile + 1)
    low = {"g1": 0, "tail": rung_below(tile, dtype, arch)}.get(floor)
    return range((int(floor) if low is None else low) + 1, tile + 1)


@pytest.mark.parametrize("arch", [GFX950, GFX11])
def test_every_head_dim_the_gradient_tables_serve_together_has_a_prep_row(
    arch: str,
) -> None:
    tables = TABLES[arch]
    gaps = []
    for dtype in sorted({row[0] for row in tables["dkdv"]}):
        prep = {
            v for row in tables["prep"] if row[0] == dtype for v in served(row, arch)
        }
        for cls in sorted({row[2] for row in tables["dkdv"] if row[0] == dtype}):
            ranges = {
                kernel: [
                    served(row, arch)
                    for row in tables[kernel]
                    if row[0] == dtype and row[2] == cls
                ]
                for kernel in ("dkdv", "dq")
            }
            for v in range(1, MAX_HEAD_DIM + 1):
                # The QK dims a graph of V width v can have, for each kernel: the dims of a row that serves v.
                qk = {
                    kernel: {d for r in rows if v in r for d in r}
                    for kernel, rows in ranges.items()
                }
                if qk["dkdv"] & qk["dq"] and v not in prep:
                    gaps.append((dtype, cls, v))
    by_group: dict[tuple[str, str], list[int]] = {}
    for dtype, cls, v in gaps:
        by_group.setdefault((dtype, cls), []).append(v)
    assert not by_group, {
        group: f"V {dims[0]}..{dims[-1]} ({len(dims)} dims)"
        for group, dims in by_group.items()
    }
