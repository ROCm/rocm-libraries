# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU control-flow tests for the GDN/KDA decode tuner."""

from __future__ import annotations

import dataclasses as dc
from itertools import product
from types import SimpleNamespace

from builders.gfx950.gdn import tune
from dispatch.gdn.gfx950 import BLOCKS_PER_V_DIM, NUM_WARPS, WARP_THREADS_K
from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec


def test_legal_configs_reuses_registry_tile_space():
    assert tune.NUM_WARPS is NUM_WARPS
    assert tune.WARP_THREADS_K is WARP_THREADS_K
    assert tune.BLOCKS_PER_V_DIM is BLOCKS_PER_V_DIM

    base = dc.replace(GdnDecodeSpec(), gate_kind="kda", num_k_heads=16, num_v_heads=32)
    expected = [
        tile
        for tile in product(NUM_WARPS, WARP_THREADS_K, BLOCKS_PER_V_DIM)
        if is_valid_spec(
            dc.replace(
                base,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
            ),
            arch=tune.ARCH,
        )[0]
    ]

    assert tune.legal_configs(base) == expected


def test_sweep_registry_batch_returns_empty_without_registry_results():
    assert tune.sweep_registry_batch(1, ()) == []


def test_main_fails_when_any_requested_registry_cell_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)

    def fake_results(request):
        return () if request.num_k_heads == 16 and request.batch == 2 else (object(),)

    monkeypatch.setattr(tune, "dispatch_gdn_decode_bases", fake_results)
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [] if not results else [(1.0, (1, 8, 1), "test", 0.0)],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="test")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune.py",
            "--geometries",
            "16/32,8/16",
            "--batches",
            "1,2",
            "--top",
            "1",
        ],
    )

    assert tune.main() == 1
    assert (
        "batch 2: no candidate was both correct and timeable" in capsys.readouterr().out
    )


def test_main_reports_dispatcher_default_outside_top_rows(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode_bases",
        lambda request: (object(), object(), object()),
    )
    monkeypatch.setattr(
        tune,
        "sweep_registry_batch",
        lambda batch, results: [
            (5.0, (4, 16, 8), "fast", 0.0),
            (6.1, (2, 16, 8), "default", 0.0),
            (6.4, (1, 8, 1), "other", 0.0),
        ],
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="default")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["tune.py", "--batches", "1", "--geometries", "16/32", "--top", "1"],
    )

    assert tune.main() == 0
    output = capsys.readouterr().out
    assert (
        "auto tile (base configuration): 6.100us  default tile=(2, 16, 8) rank=2/3"
        in output
    )
    assert "fastest legal candidate: 5.000us  fast tile=(4, 16, 8)" in output
    assert "auto tile base / fastest = 1.220x" in output
    assert "consider auto tile (4, 16, 8)" in output


def test_main_reports_missing_gdn_default_and_continues(monkeypatch, capsys):
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    monkeypatch.setattr(
        tune, "dispatch_gdn_decode_bases", lambda request: (object(), object())
    )
    # Batch 1: the default failed correctness, so it never reached the rows.
    rows = {
        1: [(5.0, (4, 16, 8), "fast", 0.0)],
        2: [(5.0, (2, 16, 8), "default", 0.0)],
    }
    monkeypatch.setattr(
        tune, "sweep_registry_batch", lambda batch, results: rows[batch]
    )
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(candidate=SimpleNamespace(spec_id="default")),
    )
    monkeypatch.setattr(
        "sys.argv", ["tune.py", "--batches", "1,2", "--geometries", "16/32"]
    )

    assert tune.main() == 1
    output = capsys.readouterr().out
    assert "auto tile 'default' is NOT in the correct-and-timeable set" in output
    assert "=== Hk16/Hv32 batch 2" in output
    assert "manual review: retain the auto tile" in output


def test_report_gdn_dispatcher_default_keeps_fastest_default(capsys):
    tune.report_gdn_dispatcher_default([(5.0, (2, 16, 8), "default", 0.0)], "default")

    assert capsys.readouterr().out.splitlines() == [
        "  auto tile (base configuration): 5.000us  default tile=(2, 16, 8) rank=1/1",
        "  fastest legal candidate: 5.000us  default tile=(2, 16, 8)",
        "  auto tile base / fastest = 1.000x",
        "  manual review: retain the auto tile",
    ]


def test_main_flags_kda_cells_where_the_auto_tile_was_not_measured(monkeypatch, capsys):
    """A KDA cell whose correct, timed rows miss the auto tile says so, and
    the run continues with the other cells."""
    shipped = (4, 16, 4)
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    monkeypatch.setattr(
        tune,
        "dispatch_gdn_decode",
        lambda request: SimpleNamespace(
            spec=SimpleNamespace(
                kernel_spec=SimpleNamespace(
                    num_warps=shipped[0],
                    warp_threads_k=shipped[1],
                    blocks_per_v_dim=shipped[2],
                    num_v_heads=request.num_v_heads,
                )
            )
        ),
    )
    monkeypatch.setattr(tune, "legal_configs", lambda base: [])

    # Hv=32 cells win with the shipped tile; Hv=16 cells win with another tile
    # and never measure the shipped one.
    def fake_sweep(base, batch, configs):
        if base.num_v_heads == 32:
            return [(1.0, shipped, 0.0)]
        return [(1.0, (1, 16, 4), 0.0)]

    monkeypatch.setattr(tune, "sweep_batch", fake_sweep)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune.py",
            "--gate-kind",
            "kda",
            "--geometries",
            "16/32,8/16",
            "--batches",
            "4,8",
        ],
    )

    assert tune.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert sum("auto tile (4, 16, 4) is NOT in the" in line for line in lines) == 2


def test_main_sweeps_kda_tiles_at_their_base_configuration(monkeypatch):
    """The KDA study starts from the dispatched tile with every knob off, not
    from ``auto``'s knob stack, so every tile is timed at its base."""
    monkeypatch.setattr(tune, "device_is_visible", lambda: True)
    bases = []
    monkeypatch.setattr(tune, "legal_configs", lambda base: bases.append(base) or [])
    monkeypatch.setattr(tune, "sweep_batch", lambda base, batch, configs: [])
    monkeypatch.setattr(
        "sys.argv", ["tune.py", "--gate-kind", "kda", "--batches", "16"]
    )

    tune.main()
    (base,) = bases
    assert base == dc.replace(
        GdnDecodeSpec(),
        gate_kind="kda",
        num_k_heads=16,
        num_v_heads=32,
        num_warps=1,
        warp_threads_k=16,
        blocks_per_v_dim=8,
    )
