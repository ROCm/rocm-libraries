# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The CDNA-9 backward row set (``tools/sdpa_bwd_rowset.py``): the row counts, and the knob overlay of an arch.

Pure Python.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import sdpa_bwd_rowset as rowset  # noqa: E402
from generators._sdpa_bwd_spec import Spec  # noqa: E402


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942"])
def test_the_row_set_has_the_documented_counts(arch: str) -> None:
    """16-bit 288 + 24 prep + 2 reduce, fp32 104 on the 8-rung ladder + 13 at hd 512, the 72 feature rows (``bias_f32``/``u2``) and the fp32 reduce: 504,
    every key canonical and distinct."""
    specs = rowset.rowset(arch, rowset.load_overlay(None))
    assert len(specs) == 504
    assert len({s.key() for s in specs}) == 504
    assert all(Spec.parse(s.key()) == s for s in specs)
    by = collections.Counter((s.kernel, s.dtype == "fp32") for s in specs)
    assert by == {
        ("prep", False): 24,
        ("dkdv", False): 144 + 40,
        ("dq", False): 144 + 32,
        ("dbias_reduce", False): 2,
        ("dbias_reduce", True): 1,
        ("prep", True): 9,
        ("dkdv", True): 54,
        ("dq", True): 54,
    }
    assert sum(s.dtype == "fp32" and s.tile == 512 for s in specs) == 13


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942"])
def test_a_feature_row_is_gfx11s_feature_pins_on_the_perf_pins_of_the_archs_sibling(
    arch: str,
) -> None:
    from generators._instances import SDPA_BWD_GFX11_TARGET  # noqa: PLC0415
    from generators._sdpa_bwd_rdna_build import specs_for  # noqa: PLC0415

    overlay_path = FLYDSL / "tools" / f"sdpa_bwd_rowset_knobs_{arch}.csv"
    specs = rowset.rowset(arch, rowset.load_overlay(overlay_path))
    plain = {s.functional: s for s in specs if not (s.bias_f32 or s.u2)}
    feature = [s for s in specs if s.bias_f32 or s.u2]
    gfx11 = [
        s
        for kernel in ("dkdv", "dq")
        for s in specs_for(kernel, SDPA_BWD_GFX11_TARGET)
        if s.bias_f32 or s.u2
    ]
    assert len(feature) == len(gfx11) == 72
    assert collections.Counter(
        (s.kernel, "bias_f32" if s.bias_f32 else "u2") for s in feature
    ) == {
        ("dkdv", "bias_f32"): 8,
        ("dkdv", "u2"): 32,
        ("dq", "bias_f32"): 16,
        ("dq", "u2"): 16,
    }
    assert all(s.dtype != "fp32" and s.tile in (64, 128) and s.padded for s in feature)
    for row, theirs in zip(feature, gfx11, strict=True):
        assert row.functional[1:10] == (
            theirs.dtype,
            theirs.tile,
            theirs.cls,
            theirs.dform,
            theirs.dropout,
            theirs.padded,
            theirs.floor,
            theirs.hdv,
            theirs.hints,
        )
        assert (row.bias_f32, row.u2, row.store_db) == (
            theirs.bias_f32,
            theirs.u2,
            theirs.store_db,
        )
        sibling = plain[(row.kernel, *row.functional[1:])]
        perf = lambda s: {
            k: v for k, v in s.knobs.items() if k not in ("bias_f32", "u2", "store_db")
        }  # noqa: E731
        assert perf(row) == perf(sibling), row.key()


def test_a_feature_row_without_a_sibling_is_refused() -> None:
    with pytest.raises(ValueError, match="no sibling"):
        rowset.feature_rows("gfx90a", [])


def test_the_knob_overlay_spells_only_the_cells_it_names() -> None:
    overlay = rowset.load_overlay(FLYDSL / "tools" / "sdpa_bwd_rowset_knobs_gfx90a.csv")
    specs = {s.key(): s for s in rowset.rowset("gfx90a", overlay)}
    assert len(specs) == 504
    pinned = [s for s in specs.values() if s.grid_order]
    assert pinned and all(s.kernel == "dkdv" and s.dtype != "fp32" for s in pinned)
    assert not any(s.grid_order for s in specs.values() if s.tile == 512)


def test_the_gfx90a_overlay_spells_vec_tail_on_every_prep_row_and_nothing_else_prep() -> (
    None
):
    overlay = rowset.load_overlay(FLYDSL / "tools" / "sdpa_bwd_rowset_knobs_gfx90a.csv")
    specs = rowset.rowset("gfx90a", overlay)
    prep = [s for s in specs if s.kernel == "prep"]
    assert len(prep) == 33 and all(s.vec_tail == 1 and s.padded for s in prep)
    assert not any(s.vec_tail for s in specs if s.kernel != "prep")
    assert not any(
        s.vec_tail for s in rowset.rowset("gfx942", rowset.load_overlay(None))
    )
