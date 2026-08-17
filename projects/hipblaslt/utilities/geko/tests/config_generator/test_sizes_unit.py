# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from geko.config_generator import sizes as gsizes


def test_get_sizes_size_option_zero_and_dedup() -> None:
    cfg = {"SIZE_OPTION": 0, "Sizes": [[16, 16, 1, 16], [16, 16, 1, 16], [32, 32, 1, 32]]}
    out = gsizes.get_sizes(cfg)
    assert out == [[16, 16, 1, 16], [32, 32, 1, 32]]


def test_get_sizes_option_zero_requires_sizes() -> None:
    with pytest.raises(ValueError, match="'Sizes' field"):
        gsizes.get_sizes({"SIZE_OPTION": 0})


def test_get_sizes_option_one_default_density_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = {"SIZE_OPTION": 1, "CUs": 256, "MACROTILE_OPT": False, "GA": False}
    monkeypatch.setattr(gsizes, "_generate_grid_sizes", lambda _c: [[[16, 16, 1, 16], [32, 32, 1, 32]]])
    out = gsizes.get_sizes(cfg)
    assert cfg["GRID_DENSITY"] == 4
    assert out == [[16, 16, 1, 16], [32, 32, 1, 32]]


def test_get_sizes_option_one_macro_ga_not_supported() -> None:
    cfg = {"SIZE_OPTION": 1, "CUs": 256, "GRID_DENSITY": 2, "MACROTILE_OPT": True, "GA": True}
    with pytest.raises(NotImplementedError, match="Grid mode"):
        gsizes.get_sizes(cfg)


def test_get_sizes_option_two_and_invalid_option() -> None:
    with pytest.raises(ValueError, match="SIZE_OPTION 2"):
        gsizes.get_sizes({"SIZE_OPTION": 2})
    with pytest.raises(ValueError, match="Incorrect SIZE_OPTION"):
        gsizes.get_sizes({"SIZE_OPTION": 99})


def test_factorize_and_boundary_and_points() -> None:
    pairs = gsizes._factorize(12)
    assert (2, 6.0) in pairs
    assert (3, 4.0) in pairs

    boundary = gsizes._generate_boundary(4, 2, 2, 16, 16)
    assert len(boundary) >= 2

    pts = gsizes._generate_points(boundary, density=1)
    assert (16, 16) in pts


def test_find_first_point_above_and_is_left() -> None:
    boundary = [(16, 16), (32, 32), (64, 64)]
    assert gsizes._find_first_point_above(20, tuple([16, 32, 64])) == 1
    assert gsizes._find_first_point_above(99, tuple([16, 32, 64])) == -1
    assert gsizes._is_left(16, 16, boundary) is True
    assert gsizes._is_left(128, 70, boundary) is False


def test_remove_points_and_generate_sizes_from_points() -> None:
    boundary = [(16, 16), (32, 32), (64, 64)]
    pts = [(16, 16), (32, 32), (128, 128)]
    valid = gsizes._remove_points_outside_boundary(pts, boundary)
    assert (16, 16) in valid

    total, sizes = gsizes._generate_sizes_from_points([(16, 16), (2048, 2048)], [256, 8192])
    assert total >= 2
    # For k>4096 with both dims >1024, the point is skipped.
    assert [2048, 2048, 1, 8192] not in sizes[1]


def test_generate_grid_sizes_density_assert() -> None:
    with pytest.raises(AssertionError, match="Grid density range"):
        gsizes._generate_grid_sizes({"CUs": 256, "GRID_DENSITY": 0})
