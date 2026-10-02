# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU control-flow tests for the GDN/KDA decode tuner."""

from __future__ import annotations

from builders.gfx950.gdn import tune


def test_complete_sweep_returns_success(capsys):
    assert tune.report_missing_cells([]) == 0
    assert capsys.readouterr().err == ""


def test_incomplete_sweep_returns_failure_and_names_every_cell(capsys):
    missing = [(32, 32, 8), (16, 32, 4)]

    assert tune.report_missing_cells(missing) == 1
    err = capsys.readouterr().err
    assert "incomplete sweep" in err
    assert "Hk=32 Hv=32 batch=8" in err
    assert "Hk=16 Hv=32 batch=4" in err


def test_main_fails_when_any_requested_cell_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune, "legal_configs", lambda spec: [(1, 8, 1)])

    def fake_sweep(base, batch, configs):
        if base.num_k_heads == 16 and batch == 2:
            return []
        return [(1.0, (1, 8, 1), 0.0)]

    monkeypatch.setattr(tune, "sweep_batch", fake_sweep)
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
    err = capsys.readouterr().err
    assert "Hk=16 Hv=32 batch=2" in err
