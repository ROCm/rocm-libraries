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

pytestmark = pytest.mark.quick


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
    rungs at 85 / 69 / 53 / 37 (plus one for a bias dQ row that stores no dB)."""
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
        assert priority == 100 + 1 + 4 * ((256 - tile) // 16) + (
            has_bias and no_db
        ), tile
    ordered = [priorities[t] for t in sorted(priorities)]
    assert all(a > b for a, b in zip(ordered, ordered[1:]))
    if not has_bias:
        for tile, priority in {320: 85, 384: 69, 448: 53, 512: 37}.items():
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


def _dq_perf(inst, no_db: bool) -> tuple:
    names = inst.SDPA_BWD_GFX11_KNOBS["dq"][:-2]
    return tuple(
        (
            0
            if (n == "store_db" and no_db)
            else "auto" if n in ("bias_stage", "kt_lds_layout") else -1
        )
        for n in names
    )


@pytest.mark.parametrize("kind", ["dkdv", "dq"])
@pytest.mark.parametrize("cls,no_db", [("plain", 0), ("bias", 0), ("bias", 1)])
def test_a_gfx12_exact_row_outranks_its_padded_twin_and_ranks_below_the_next_smaller_rung(
    kind, cls, no_db
) -> None:
    """Exact (``padded`` 0) above padded at one tile; the padded row of a rung is above the next rung's exact row; a bias dQ
    padded ``store_db`` 0 row ranks below the exact dB-storing row of its rung."""
    from generators import _instances as inst  # noqa: PLC0415

    names = inst.SDPA_BWD_GFX11_KNOBS[kind][:-2]
    perf = _dq_perf(inst, bool(no_db)) if kind == "dq" else tuple(-1 for _ in names)

    def prio(tile: int, padded: int, store_no_db: int = 0) -> int:
        p = _dq_perf(inst, bool(store_no_db)) if kind == "dq" else perf
        row = ("f16", tile, cls, "bk", 0, padded, "tail", 0, 1, *p)
        return inst._sdpa_bwd_rdna_row(kind, "gfx12-generic", row).priority

    for tile in (48, 64, 80, 96, 128, 256):
        assert prio(tile, 0, no_db) > prio(tile, 1, no_db)
        assert prio(tile, 1, no_db) < prio(tile - 16 if tile != 96 else 80, 0, no_db)
    if kind == "dq" and cls == "bias":
        # the gfx12 formula: exact dB-storing (+2) over padded store_db 0 (+1)
        assert prio(128, 0, 0) > prio(128, 1, 1)
    # gfx11 keeps the padded-agnostic formula
    row = ("f16", 128, cls, "bk", 0, 0, "tail", 0, 1, *perf)
    assert inst._sdpa_bwd_rdna_row(kind, "gfx11-generic", row).priority == (
        100
        + 1
        + 2 * ((256 - 128) // 16)
        + (kind == "dq" and cls == "bias" and bool(no_db))
    )


@pytest.mark.parametrize("kind", ["prep", "dkdv", "dq"])
def test_gfx12_priorities_are_strictly_unique_per_class(kind: str) -> None:
    """Within a gfx12 (op, dtype, class, dropout, knobs, store_db), no two rows share a priority."""
    groups = defaultdict(dict)
    for row in instances_for(f"sdpa_bwd_{kind}", "gfx12-generic"):
        k = row.knobs
        key = (
            row.dtype,
            k["causal"],
            k["has_bias"],
            k["has_dropout"],
            k.get("bias_f32"),
            k.get("u2"),
            k.get("store_db"),
            k["head_dim_v"] == k["head_dim"],
        )
        slot = (k["head_dim"], k["padded"], k["floor"])
        assert slot not in groups[key], f"duplicate {slot} in {key}"
        assert row.priority not in groups[key].values(), (key, slot, row.priority)
        groups[key][slot] = row.priority
    assert groups
