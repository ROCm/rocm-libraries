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

    monkeypatch.setattr(tune, "dispatch_gdn_decode_all", fake_results)
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
