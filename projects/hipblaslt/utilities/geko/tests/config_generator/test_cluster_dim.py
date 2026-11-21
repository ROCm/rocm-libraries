# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 ClusterDim selection: hardware shapes, grid fit, traffic, and GA weights."""

from __future__ import annotations

import numpy as np
import pytest

from geko.config_generator.config_sections_generator import (
    DUCTILE_COST_SCALE,
    _cluster_variant_cost_offsets,
)
from geko.config_generator.constants import HARDWARE_MAP
from geko.config_generator.fork_params.hw_profiles.gfx1250.cluster_dim import (
    hardware_cluster_dims,
    select_cluster_dims,
    tile_grid,
    wgps_per_shader_engine,
)
from geko.config_generator.shared_utils import ForkParameter

SHAPES_16_WGPS = [
    (1, 1),
    (1, 2), (2, 1),
    (1, 4), (2, 2), (4, 1),
    (1, 8), (2, 4), (4, 2), (8, 1),
    (2, 8), (4, 4), (8, 2),
]


@pytest.mark.parametrize("arch", ["gfx1250", "gfx1250_96cu", "gfx1250_192cu", "gfx1250-strict_96cu"])
def test_every_gfx1250_key_has_16_wgp_shader_engines(arch: str) -> None:
    hw = HARDWARE_MAP[arch]
    assert wgps_per_shader_engine(hw["CUs"], hw["XCC"]) == 16


def test_wgps_per_shader_engine_rejects_an_uneven_split() -> None:
    with pytest.raises(ValueError, match="shader engines"):
        wgps_per_shader_engine(100, 3)


def test_hardware_shapes_divide_a_16_wgp_engine() -> None:
    """Sizes 1, 2, 4, 8 and 16, without the 16x1 / 1x16 shapes Tensile rejects."""
    assert hardware_cluster_dims(16) == SHAPES_16_WGPS


def test_hardware_shapes_follow_the_engine_size() -> None:
    assert hardware_cluster_dims(8) == SHAPES_16_WGPS[:10]


def test_tile_grid_rounds_up() -> None:
    assert tile_grid(4096, 4096, (256, 352)) == (16, 12)


@pytest.mark.parametrize(
    "grid,macro_tile,expected",
    [
        # 4096 x 4096 at MT 256x352. N has 12 tiles, so Ck is 1, 2 or 4, and the
        # wider MT1 makes the shapes with more peers along M fetch less.
        ((16, 12), (256, 352), [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4)]),
        # 4096 x 4096 at MT 512x176: the mirror case, with 24 tiles along N.
        ((8, 24), (512, 176), [(1, 1), (1, 2), (1, 4), (2, 4), (2, 8)]),
        # Square tile on a square grid: both orientations tie at sizes 2 and 8.
        ((16, 16), (256, 256), [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4)]),
        # One tile along M leaves only clusters along N.
        ((1, 16), (256, 256), [(1, 1), (1, 2), (1, 4), (1, 8)]),
    ],
)
def test_select_keeps_fitting_least_traffic_shapes_per_size(grid, macro_tile, expected) -> None:
    assert select_cluster_dims(SHAPES_16_WGPS, grid, macro_tile) == expected


def test_select_falls_back_to_all_candidates_when_none_fits() -> None:
    assert select_cluster_dims([[2, 2], [4, 4]], (3, 5), (256, 256)) == [(2, 2), (4, 4)]


def _entry(mi: list, shape=None) -> dict:
    entry = {"MatrixInstruction": ForkParameter(name="MatrixInstruction", values=mi)}
    if shape is not None:
        entry["ClusterDim"] = ForkParameter(name="ClusterDim", values=[list(shape)])
    return entry


def test_cluster_variants_share_their_mi_sampling_mass() -> None:
    """At equal cost, an MI with three cluster shapes is drawn as often as one with one shape."""
    mi_a = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    mi_b = [16, 16, 32, 1, 1, 4, 16, 4, 1]
    entries = [_entry(mi_a, (1, 1)), _entry(mi_a, (2, 1)), _entry(mi_a, (2, 2)), _entry(mi_b, (1, 1))]
    offsets = _cluster_variant_cost_offsets(entries)
    weights = np.exp(-DUCTILE_COST_SCALE * offsets)
    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights[0] == pytest.approx(weights[1]) == pytest.approx(weights[2])


def test_entries_without_cluster_dim_keep_their_cost() -> None:
    mi = [16, 16, 32, 1, 1, 8, 8, 2, 2]
    assert _cluster_variant_cost_offsets([_entry(mi), _entry(mi)]).tolist() == [0.0, 0.0]


def _variant(entry: dict, lsi: int, tdm: int) -> dict:
    return {
        **entry,
        "LDSSegmentInterleave": ForkParameter(name="LDSSegmentInterleave", values=[lsi]),
        "TDMFuse": ForkParameter(name="TDMFuse", values=[tdm]),
    }


def test_lds_segment_interleave_variants_share_their_mi_sampling_mass() -> None:
    """The (LDSSegmentInterleave, TDMFuse) entries of one MI, each repeated per cluster
    shape, are drawn as often as an MI with a single entry."""
    mi_a = [32, 16, 128, 1, 1, 8, 4, 2, 2]
    mi_b = [16, 16, 128, 1, 1, 4, 4, 2, 4]
    pairs_a = [(0, 0), (0, 1), (1, 0)]
    entries = [_variant(_entry(mi_a, shape), *pair) for pair in pairs_a for shape in ((1, 1), (4, 4))]
    entries += [_variant(_entry(mi_b, (1, 1)), 0, 1)]
    weights = np.exp(-DUCTILE_COST_SCALE * _cluster_variant_cost_offsets(entries))
    assert weights[:6].sum() == pytest.approx(weights[6])
    assert len(set(np.round(weights[:6], 6))) == 1
    # Without ClusterDim the pairs still split one MI's mass, and an MI with fewer
    # pairs (one that cannot interleave) keeps the same total.
    no_cd = [_variant(_entry(mi_a), *pair) for pair in pairs_a]
    no_cd += [_variant(_entry(mi_b), 0, 0), _variant(_entry(mi_b), 0, 1)]
    weights = np.exp(-DUCTILE_COST_SCALE * _cluster_variant_cost_offsets(no_cd))
    assert weights[:3].sum() == pytest.approx(weights[3:].sum()) == pytest.approx(1.0)


def test_temporal_hint_variants_share_their_mi_sampling_mass() -> None:
    """TemporalHintA/B pairs coupled per TDMFuse split one MI's mass like the other variants."""
    mi_a = [32, 16, 128, 1, 1, 8, 4, 2, 2]
    mi_b = [16, 16, 128, 1, 1, 4, 4, 2, 4]

    def th(entry: dict, a: int, b: int) -> dict:
        return {
            **entry,
            "TemporalHintA": ForkParameter(name="TemporalHintA", values=[a]),
            "TemporalHintB": ForkParameter(name="TemporalHintB", values=[b]),
        }

    entries = [th(_variant(_entry(mi_a), 0, 0), x, x) for x in (0, 1)]
    entries += [th(_variant(_entry(mi_a), 0, 1), x, y) for x in (0, 1) for y in (0, 1)]
    entries += [_variant(_entry(mi_b), 0, 1)]
    weights = np.exp(-DUCTILE_COST_SCALE * _cluster_variant_cost_offsets(entries))
    assert weights[:6].sum() == pytest.approx(weights[6]) == pytest.approx(1.0)
    assert len(set(np.round(weights[:6], 6))) == 1

def test_source_swap_entries_keep_their_own_share() -> None:
    """SourceSwap is not a variant field: each SourceSwap entry of an MI keeps the mass
    of a separate MI, as without the LDSSegmentInterleave / TDMFuse split."""
    mi = [16, 16, 128, 1, 1, 8, 8, 2, 2]

    def ss(entry: dict, value: bool) -> dict:
        return {**entry, "SourceSwap": ForkParameter(name="SourceSwap", values=[value])}

    entries = [_variant(ss(_entry(mi, (1, 1)), swap), *pair) for swap in (False, True) for pair in ((0, 0), (1, 0))]
    weights = np.exp(-DUCTILE_COST_SCALE * _cluster_variant_cost_offsets(entries))
    assert weights[:2].sum() == pytest.approx(weights[2:].sum()) == pytest.approx(1.0)


def test_source_swap_entries_keep_an_equal_share_per_value() -> None:
    """SourceSwap-split entries are not merged into one MI variant set: each SourceSwap
    value of an MI gets the same mass even when its tile keeps a different number of
    ClusterDim shapes, and MIs that are split alike keep their relative mass."""
    mi_a = [32, 16, 128, 1, 1, 8, 4, 2, 2]
    mi_b = [16, 16, 32, 1, 1, 4, 4, 2, 4]

    def ss(entry: dict, value: bool) -> dict:
        return {**entry, "SourceSwap": ForkParameter(name="SourceSwap", values=[value])}

    entries = [
        ss(_entry(mi_a, (1, 1)), False), ss(_entry(mi_a, (2, 8)), False),
        ss(_entry(mi_a, (1, 1)), True), ss(_entry(mi_a, (2, 2)), True),
        ss(_entry(mi_a, (4, 4)), True),
        ss(_entry(mi_b, (1, 1)), False), ss(_entry(mi_b, (1, 1)), True),
    ]
    weights = np.exp(-DUCTILE_COST_SCALE * _cluster_variant_cost_offsets(entries))
    assert weights[0:2].sum() == pytest.approx(weights[2:5].sum())
    assert weights[0:2].sum() == pytest.approx(weights[5]) == pytest.approx(weights[6])
