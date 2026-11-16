# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The gfx11 and gfx12 default plan: a smaller ladder rung outranks a larger one.

Within one (op, dtype, class, dropout) the priority of a ``gfx11-generic`` or ``gfx12-generic`` backward row falls strictly with its
tile, so the heuristic's default order takes the smallest rung that covers the problem and never falls back to the kernel-id
tie-break. gfx12-generic's wide rungs (320, 384, 448, 512) extend the same formula. Pure Python.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators._instances import instances_for  # noqa: E402


@pytest.mark.parametrize("kind", ["prep", "dkdv", "dq"])
def test_priority_falls_strictly_with_tile(kind: str) -> None:
    groups = defaultdict(dict)
    for row in instances_for(f"sdpa_bwd_{kind}", "gfx11-generic"):
        knobs = row.knobs
        key = (
            row.dtype,
            knobs["causal"],
            knobs["has_bias"],
            knobs["has_dropout"],
            knobs.get("bias_f32"),
            knobs.get("u2"),
            knobs.get("store_db"),
        )
        tile = knobs["head_dim"]
        assert tile not in groups[key], f"duplicate tile {tile} in {key}"
        groups[key][tile] = row.priority
    assert groups
    for key, by_tile in groups.items():
        ordered = [by_tile[tile] for tile in sorted(by_tile)]
        assert all(a > b for a, b in zip(ordered, ordered[1:])), (key, by_tile)


def test_the_priority_formula_extends_to_the_gfx12_wide_rungs() -> None:
    from generators import _instances as inst  # noqa: PLC0415
    from generators._sdpa_bwd_spec import (
        GFX12_FP32_LADDER,
        GFX12_LADDER,
    )  # noqa: PLC0415

    # The instance module spells the ladders itself (it imports no generator module); the Spec module's are the same.
    assert inst.SDPA_BWD_GFX12_LADDER == GFX12_LADDER
    assert inst.SDPA_BWD_GFX12_FP32_LADDER == GFX12_FP32_LADDER
    assert inst.PRIORITY_SPECIALIZED == 100


@pytest.mark.parametrize("dtype", ["f16", "bf16", "fp32"])
@pytest.mark.parametrize("bias_store", [(0, 0), (1, 0), (1, 1)])
def test_a_gfx12_row_of_each_rung_ranks_below_every_smaller_rung(
    dtype, bias_store
) -> None:
    """A synthetic gfx12 table with one row per rung of the arch's ladder for the dtype: priorities fall strictly with the tile, the wide
    rungs at 93 / 85 / 77 / 69 (plus one for a bias dQ row that stores no dB)."""
    from generators import _instances as inst  # noqa: PLC0415

    has_bias, no_db = bias_store
    ladder = (
        inst.SDPA_BWD_GFX12_FP32_LADDER
        if dtype == "fp32"
        else inst.SDPA_BWD_GFX12_LADDER
    )
    cls = "bias" if has_bias else "plain"
    names = inst.SDPA_BWD_GFX11_KNOBS["dq"][
        :-2
    ]  # the trailing feature columns (bias_f32, u2) may be left out of a row
    perf = tuple(
        (
            0
            if (n == "store_db" and no_db)
            else "auto" if n in ("bias_stage", "kt_lds_layout") else -1
        )
        for n in names
    )
    priorities = {}
    for tile in ladder:
        row = (dtype, tile, cls, "bk", 0, 1, "g1", 0, 1, *perf)
        priorities[tile] = inst._sdpa_bwd_rdna_row("dq", "gfx12-generic", row).priority
    for tile, priority in priorities.items():
        assert priority == 100 + 1 + 2 * ((256 - tile) // 16) + (
            has_bias and no_db
        ), tile
    ordered = [priorities[t] for t in sorted(priorities)]
    assert all(a > b for a, b in zip(ordered, ordered[1:]))
    if not has_bias:
        for tile, priority in {320: 93, 384: 85, 448: 77, 512: 69}.items():
            if tile in priorities:
                assert priorities[tile] == priority


@pytest.mark.parametrize(
    "dtype,tile,floor",
    [
        ("f16", 320, "256"),
        ("f16", 384, "320"),
        ("f16", 448, "384"),
        ("f16", 512, "448"),
        ("bf16", 512, "448"),
        ("fp32", 512, "256"),
        ("f16", 256, "224"),
        ("fp32", 256, "224"),
    ],
)
def test_a_gfx12_prep_floor_is_the_rung_below_on_its_dtype_ladder(
    dtype, tile, floor
) -> None:
    """The prep floor comes from the (arch, dtype) ladder: 16-bit 512 sits above 448, fp32 512 above 256; gfx11 prep floors are unchanged."""
    from generators import _instances as inst  # noqa: PLC0415

    row = (dtype, tile, "plain", "bk", 0, 1, "tail", 0, 1)
    assert inst._sdpa_bwd_rdna_row("prep", "gfx12-generic", row).knobs["floor"] == floor
    if tile <= 256:
        assert (
            inst._sdpa_bwd_rdna_row("prep", "gfx11-generic", row).knobs["floor"]
            == floor
        )
