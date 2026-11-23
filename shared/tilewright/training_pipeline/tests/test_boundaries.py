# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import pytest
import torch

import run_orchestrator as ro
import stage05_train as s5
from lib import boundaries as bd

from test_stage05_train import _cell_kwargs, _main, write_round

MID = "Mid|Large|MidK|Bnone"
LARGE = "Large|Large|MidK|Bnone"


def test_cell_box_has_inner_sides_only():
    assert bd.cell_box("Tiny|Large|TinyK|Bany") == {
        "M": (None, 32.5),
        "N": (512.5, None),
        "K": (None, 32.5),
    }
    assert bd.cell_box("Mid|Small|MidK|Bnone") == {
        "M": (128.5, 512.5),
        "N": (32.5, 128.5),
        "K": (32.5, 512.5),
    }
    assert bd.cell_box("Large|Large|LargeK|Bnone#M<=4096#K>2048") == {
        "M": (512.5, 4096.5),
        "N": (512.5, None),
        "K": (2048.5, None),
    }


def test_distances_are_symmetric_across_a_threshold():
    lo, hi = bd.cell_box(MID), bd.cell_box(LARGE)
    assert bd.boundary_distance(lo, 512, 4096, 256) == pytest.approx(
        bd.boundary_distance(hi, 513, 4096, 256), abs=1e-5
    )
    assert bd.boundary_distance(lo, 256, 4096, 256) == pytest.approx(1.0, abs=0.01)
    assert bd.octaves_outside(lo, 256, 4096, 256) == 0.0
    assert bd.octaves_outside(lo, 1025, 4096, 256) == pytest.approx(1.0, abs=0.01)
    assert bd.octaves_outside(lo, 600, 4096, 2048) == pytest.approx(2.0, abs=0.01)
    assert [bd.distance_bucket(d) for d in (0.1, 0.25, 0.7, 1.0)] == [
        "<0.25",
        "0.25-0.5",
        "0.5-1",
        ">=1",
    ]


def _g(m, n, k, b=1):
    return {"m": m, "n": n, "k": k, "batch_count": b}


def test_margin_members_cross_shared_sides_within_the_margin():
    pool = {
        MID: [_g(400, 4096, 256)],
        LARGE: [_g(600, 4096, 256), _g(2000, 4096, 256), _g(600, 4096, 256, 4)],
        "Large|Large|LargeK|Bnone": [_g(600, 4096, 600), _g(600, 4096, 4000)],
        MID + "#N<=2048": [_g(400, 1800, 256)],
    }
    got = bd.margin_members(MID, pool, 0.5)
    assert got == [pool["Large|Large|LargeK|Bnone"][0], pool[LARGE][0]]
    assert bd.margin_members(MID, pool, 0) == []
    assert (
        bd.margin_members(MID + "#N>2048", pool, 0.5)[-1] == pool[MID + "#N<=2048"][0]
    )


def _key(g):
    return (g["m"], g["n"], g["k"], g["batch_count"])


def _shapes(seed, n, m_range):
    rng = random.Random(seed)
    return [
        (rng.randint(*m_range), rng.randint(600, 4000), rng.randint(2, 16) * 32, 1)
        for _ in range(n)
    ]


def test_margin_training_never_sees_held_out_gemms(tmp_path, monkeypatch):
    """The margin pool is the training GEMMs stage05 loads: the gate slice and
    a hold-out split stay out, and so does every GEMM a neighbouring cell
    holds out for its own validation."""
    rd = tmp_path / "round_1"
    train = _shapes(1, 20, (300, 512)) + _shapes(2, 20, (513, 1000))
    gate = _shapes(3, 6, (400, 512)) + _shapes(4, 6, (513, 700))
    holdout = _shapes(5, 6, (400, 512)) + _shapes(6, 6, (513, 700))
    train_dir = write_round(rd / "stage04b" / "train", train)
    write_round(rd / "stage04b" / "gate", gate)
    write_round(tmp_path / "split" / "holdout", holdout)
    assert ro.Pipeline.training_data_dir(rd) == train_dir
    assert ro.Pipeline.gate_dir(rd) == rd / "stage04b" / "gate"

    seen = {}
    real = s5.train_cell

    def spy(**kw):
        seen[kw["cell"]] = kw
        return real(**kw)

    monkeypatch.setattr(s5, "train_cell", spy)
    extra = ["--skip-deployed-eval", "--validation-frac", "0.25"]
    extra += ["--boundary-margin", "1", "--boundary-weight", "0.3"]
    assert _main(monkeypatch, tmp_path, [train_dir], tmp_path / "out", *extra) == 0
    assert set(seen) == {MID, LARGE}
    held_out = {tuple(s) for s in gate + holdout}
    for cell, kw in seen.items():
        other = seen[LARGE if cell == MID else MID]
        other_train, other_val = s5.split_validation(other["gemms"], 0.25, 5)
        margin = {_key(g) for g in kw["margin_gemms"]}
        assert margin and kw["margin_weight"] == 0.3
        assert margin <= {_key(g) for g in other_train}
        assert not margin & {_key(g) for g in other_val}
        assert not margin & {_key(g) for g in kw["gemms"]}
        assert not margin & held_out


def test_margin_gemms_join_the_loss_but_not_the_measurements(tmp_path):
    data = write_round(tmp_path / "d", _shapes(7, 24, (300, 512)))
    margin_data = write_round(tmp_path / "m", _shapes(8, 12, (513, 900)))
    (gemms,) = s5.load_enriched_chunks(str(data), True)[0].values()
    (margin,) = s5.load_enriched_chunks(str(margin_data), True)[0].values()
    kw = _cell_kwargs(cell=MID, use_teacher=True)
    plain = s5.train_cell(gemms=gemms, selector=None, **kw)
    r = s5.train_cell(
        gemms=gemms, selector=None, margin_gemms=margin, margin_weight=0.3, **kw
    )
    assert plain["n_margin_gemms"] == 0 and r["n_margin_gemms"] == len(margin)
    assert r["n_train_gemms"] == plain["n_train_gemms"]
    assert r["n_pairs"] == plain["n_pairs"]
    assert {m["n_eval_sample"] for m in r["epoch_metrics"]} == {r["n_train_gemms"]}
    assert r["steps"] > plain["steps"]


def test_weighted_listwise_loss_is_a_weighted_mean():
    scores = torch.tensor([1.0, 0.0, 0.5, 2.0, 0.0])
    gidx = torch.tensor([0, 0, 1, 1, 1])
    targets = torch.tensor([0.9, 0.1, 0.2, 0.7, 0.1])
    per = [
        s5.listwise_loss(scores[:2], gidx[:2], 1, targets[:2]),
        s5.listwise_loss(scores[2:], gidx[2:] - 1, 1, targets[2:]),
    ]
    w = torch.tensor([1.0, 0.25])
    got = s5.listwise_loss(scores, gidx, 2, targets, weights=w)
    assert float(got) == pytest.approx(float((per[0] + 0.25 * per[1]) / 1.25))
    same = s5.listwise_loss(scores, gidx, 2, targets, weights=torch.ones(2))
    assert float(same) == pytest.approx(
        float(s5.listwise_loss(scores, gidx, 2, targets))
    )


def test_boundary_option_validation():
    ok = ro.validate_config({"train": {"boundary_margin": 0.5}})[0]
    assert not [e for e in ok if e.startswith("train")]
    for bad in (
        {"boundary_margin": -1},
        {"boundary_weight": 0},
        {"boundary_weight": 1.5},
    ):
        errors = ro.validate_config({"train": bad})[0]
        assert any(f"train.{next(iter(bad))}" in e for e in errors), bad
