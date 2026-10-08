# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU behavior checks for the rocKE-only KDA decode benchmark."""

from __future__ import annotations

import pytest

from benchmarks.gfx950.gdn import benchmark_kda_decode as bench


def test_parse_variants_accepts_known_names_and_deduplicates():
    assert bench.parse_variants("fused,precomputed,fused,simple") == (
        "fused",
        "precomputed",
        "simple",
    )


def test_parse_variants_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown variant"):
        bench.parse_variants("fused,aiter")
    with pytest.raises(ValueError, match="at least one"):
        bench.parse_variants("")


def test_production_variants_derive_from_the_dispatched_tile():
    variants, result = bench.production_variants(
        batch=8,
        num_k_heads=32,
        num_v_heads=32,
        head_dim=128,
        names=("fused", "precomputed", "simple"),
    )

    assert result.candidate.spec_id == "kda_w512"
    fused = variants["fused"]
    raw = variants["precomputed"]
    simple = variants["simple"]
    tile = (fused.num_warps, fused.warp_threads_k, fused.blocks_per_v_dim)
    assert tile == (1, 16, 4)
    assert (raw.num_warps, raw.warp_threads_k, raw.blocks_per_v_dim) == tile
    assert raw.fuse_gate is False
    assert simple.simple is True
    assert simple.gate_kind == "kda"


def test_compare_selected_to_sweep_reports_best_and_ratio():
    rows = [
        (4.0, (2, 16, 1), 1e-3),
        (5.0, (1, 16, 4), 1e-3),
        (6.0, (4, 16, 4), 1e-3),
    ]
    result = bench.compare_selected_to_sweep((1, 16, 4), rows)

    assert result.selected_us == 5.0
    assert result.best_tile == (2, 16, 1)
    assert result.best_us == 4.0
    assert result.selected_over_best == 1.25


def test_compare_selected_to_sweep_fails_when_selected_missing_or_rows_empty():
    with pytest.raises(ValueError, match="no correct, timeable tile"):
        bench.compare_selected_to_sweep((1, 16, 4), [])
    with pytest.raises(ValueError, match="selected tile"):
        bench.compare_selected_to_sweep((1, 16, 4), [(4.0, (2, 16, 1), 1e-3)])


def test_failure_accumulator_is_fail_closed():
    failures = bench.Failures()
    assert failures.exit_code == 0
    failures.add("device timing failed")
    assert failures.exit_code == 1
    assert failures.messages == ["device timing failed"]
