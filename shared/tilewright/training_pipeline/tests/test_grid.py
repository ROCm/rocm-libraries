# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from lib import grid


@pytest.mark.parametrize(
    "v, tier",
    [(1, "Tiny"), (32, "Tiny"), (33, "Small"), (128, "Small"), (129, "Mid")]
    + [(512, "Mid"), (513, "Large"), (10**6, "Large")],
)
def test_mn_tiers(v, tier):
    assert grid.m_tier(v) == tier
    assert grid.n_tier(v) == tier


@pytest.mark.parametrize(
    "v, tier",
    [(1, "TinyK"), (32, "TinyK"), (33, "MidK"), (512, "MidK"), (513, "LargeK")],
)
def test_k_tiers(v, tier):
    assert grid.k_tier(v) == tier


def test_b_tier_and_cell_key():
    assert grid.b_tier(1) == "Bnone"
    assert grid.b_tier(2) == "Bany"
    assert grid.cell_key(16, 200, 4096, 8) == "Tiny|Mid|LargeK|Bany"


def test_all_cell_labels_cover_the_grid():
    labels = grid.all_cell_labels()
    assert len(labels) == 96 == len(set(labels))
    assert grid.cell_key(1, 1, 1, 1) in labels


def test_sampling_ranges_stay_inside_their_tiers():
    for tier, (lo, hi) in grid.TIER_RANGES_MN.items():
        assert grid.m_tier(lo) == tier == grid.m_tier(hi)
    for tier, (lo, hi) in grid.TIER_RANGES_K.items():
        assert grid.k_tier(lo) == tier == grid.k_tier(hi)
    assert all(grid.b_tier(b) == "Bany" for b in grid.BANY_VALUES)
