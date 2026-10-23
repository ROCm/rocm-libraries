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
