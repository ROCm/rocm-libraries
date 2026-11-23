# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import msgpack
import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage03c_bench_contexts as s03c
from lib import contexts as cx
from lib import pool_libdir as pl
from lib.dat import read_tensile_logic
from test_run_orchestrator import (
    STEM,
    _pipeline,
    arg,
    calls,
    defined_options,
    env,
    make_config,
    run,
    stubs,
)

PIPELINE_DIR = Path(__file__).resolve().parent.parent

CTX_LIST = [
    {"cu_budget": 0},
    {"cu_budget": 128, "cotenant": True},
    {"cu_budget": 64, "cotenant": True, "fraction": 0.5},
]


def _cfg(**sec):
    return {"contexts": {"list": list(CTX_LIST), **sec}}


def _shares(cfg, n_cu=None):
    """The contexts of `cfg` (`cx.contexts_from_config`) but the launch-keyed
    workspace contexts of rule version 3: its CU shares and holdouts."""
    return [c for c in cx.contexts_from_config(cfg, n_cu) if not c.launch_keyed]


# ── config ───────────────────────────────────────────────────────────────────


def test_contexts_from_config_skips_the_exclusive_entry():
    ctxs = _shares(_cfg(holdout=[{"cu_budget": 96}]))
    assert [c.label for c in ctxs] == ["cu128_cot", "cu64_cot", "cu96"]
    assert [c.holdout for c in ctxs] == [False, False, True]
    assert ctxs[1].fraction == 0.5 and ctxs[0].cotenant_cus(256) == 128
    assert ctxs[2].cotenant_cus(256) == 0
    assert cx.BenchContext(schedule="dynamic").label == "allcu_dynamic"
    assert cx.BenchContext(cu_budget=8, name="small").label == "small"
    assert cx.trained_fractions(ctxs, 256) == [0.25, 0.5, 1.0]


@pytest.mark.parametrize(
    "sec, n_cu, needle",
    [
        ({"list": [{"cu_budget": 0, "cotenant": True}]}, None, "cotenant needs"),
        ({"holdout": [{"cu_budget": 0}]}, None, "cannot be held out"),
        ({"list": [{"cu_budget": 0, "fraction": 0.5}]}, None, "fraction must be 1"),
        ({"list": [{"cu_budget": 256}]}, 256, "must be below"),
        (
            {"list": [{"cu_budget": 64}], "holdout": [{"cu_budget": 64}]},
            None,
            "also contexts.list[0]",
        ),
        (
            {"list": [{"cu_budget": 64, "name": "a"}, {"cu_budget": 32, "name": "a"}]},
            None,
            "name 'a'",
        ),
    ],
)
def test_context_errors(sec, n_cu, needle):
    errors = cx.context_errors({"contexts": sec}, n_cu)
    assert any(needle in e for e in errors), errors


@pytest.mark.parametrize(
    "fraction, n_cu, floor, budget",
    [
        (0.75, 256, None, 192),
        (0.5, 256, None, 128),
        (0.25, 256, None, 64),
        (0.875, 256, None, 224),
        (0.375, 256, None, 96),
        (0.75, 80, None, 64),
        (0.25, 80, None, 32),
        (0.01, 256, None, 32),
        (0.99, 256, None, 224),
        (0.25, 304, None, 64),
        (0.25, 304, 0.25, 96),
        (0.25, 304, 0.9, 288),
        (0.5, 33, None, 32),
    ],
)
def test_fraction_budget_is_a_multiple_of_the_budget_granule(
    fraction, n_cu, floor, budget
):
    got = cx.fraction_budget(fraction, n_cu, floor)
    assert got == budget and got % cx.CU_BUDGET_GRANULE == 0 and 0 < got < n_cu


def test_a_device_without_a_granule_below_its_cus_has_no_budget():
    with pytest.raises(ValueError, match="multiple of 32"):
        cx.fraction_budget(0.5, 32)
    with pytest.raises(ValueError, match="multiple of 64"):
        cx.fraction_budget(0.5, 64, granule=64)
    with pytest.raises(ValueError, match="must be positive"):
        cx.fraction_budget(0.5, 256, granule=0)


EIGHTHS = [
    {"cu_budget": 0},
    {"cu_fraction": 0.75, "cotenant": True, "fraction": 0.1},
    {"cu_fraction": 0.5, "cotenant": True, "fraction": 0.1},
    {"cu_fraction": 0.25, "cotenant": True, "fraction": 0.1},
    {"cu_fraction": 0.125, "cotenant": True, "fraction": 0.1},
]


def _budgets(cfg, n_cu):
    return [c.cu_budget for c in _shares(cfg, n_cu)]


def test_shares_from_an_eighth_resolve_to_whole_granules_from_one_granule_up():
    """Three quarters, half, a quarter and an eighth of the device's CUs are
    whole multiples of the budget granule, one granule the smallest: an
    eighth exactly on a device of eight granules, one granule where an
    eighth is less. However small a share, its budget is never below one
    granule, and it is an SM target hipBLASLt takes as it is."""
    granule = cx.CU_BUDGET_GRANULE
    assert granule == 32
    cfg = {"contexts": {"list": EIGHTHS}}
    assert cx.budget_granule(cfg) == granule == cx.budget_granule({})
    assert _budgets(cfg, 256) == [192, 128, 64, 32]
    assert _budgets(cfg, 192) == [160, 96, 64, 32]
    out, lines = cx.resolve_fractions(cfg, 192)
    assert lines[-1] == "contexts.list[4]: cu_fraction 0.125 is 32 of 192 CUs"
    assert cx.context_errors(cfg, 256) == [] == cx.context_errors(out, 192)
    for n_cu in range(granule + 1, 16 * granule + 1):
        for share in (0.001, 0.01, 0.05, 0.1, 0.125, 0.2, 0.25):
            budget = cx.fraction_budget(share, n_cu)
            assert budget % granule == 0 and granule <= budget < n_cu, (n_cu, share)
        lowest = min(_budgets(cfg, n_cu))
        assert lowest == cx.fraction_budget(0.125, n_cu), n_cu
        assert lowest == granule or n_cu >= 12 * granule, n_cu


def test_the_budgets_keep_their_granule_though_the_sm_target_is_exact():
    """The budget granule is the budgets' own: hipBLASLt's launch rule takes
    the SM target as it is (a granule of one CU), so launches follow every
    budget exactly, and the shares still resolve to the same whole granules,
    the eighth's to one granule on a device of 192 CUs."""
    from lib import hipblaslt_launch as hl

    cfg = {"contexts": {"list": EIGHTHS}}
    assert hl.SM_TARGET_GRANULE == 1
    assert _budgets(cfg, 256) == [192, 128, 64, 32]
    assert _budgets(cfg, 192) == [160, 96, 64, 32]
    assert cx.fraction_budget(0.125, 192) == 32 == cx.fraction_budget(0.01, 256)
    assert cx.context_errors(cfg, 192) == []


def test_a_config_granule_sets_the_budgets_and_must_be_one_cu_or_more():
    """`contexts.granule` replaces the default granule for every budget a
    share resolves to, and for the check of a budget given beside its
    share; any granule of one CU or more sizes the kernels for exactly the
    CUs of each budget, and one of none is the only error reported then."""
    coarse = {"contexts": {"list": EIGHTHS, "granule": 64}}
    assert cx.budget_granule(coarse) == 64
    assert _budgets(coarse, 256) == [192, 128, 64]
    out, lines = cx.resolve_fractions(coarse, 256)
    assert lines[-1] == (
        "contexts.list[4]: cu_fraction 0.125 is 64 of 256 CUs, the budget of "
        "contexts.list[3]; left out"
    )
    assert _budgets(coarse, 192) == [128, 64]
    _, (*_, line) = cx.resolve_fractions(coarse, 64)
    assert line == (
        "contexts.list[4]: cu_fraction 0.125 has no budget on a device of 64 CUs "
        "(a budget is a multiple of 64 CUs below its CU count); left out"
    )
    assert cx.context_errors(coarse, 256) == [] == cx.context_errors(out, 256)
    given = {"list": [{"cu_fraction": 0.125, "cu_budget": 64}]}
    assert cx.context_errors({"contexts": dict(given, granule=64)}, 256) == []
    (err,) = cx.context_errors({"contexts": given}, 256)
    assert err.endswith("is not cu_fraction 0.125 of the device's 256 CUs (32)")
    for fine, budgets in ((16, [192, 128, 64, 32]), (48, [192, 144, 48])):
        cfg = {"contexts": {"list": EIGHTHS, "granule": fine}}
        assert cx.context_errors(cfg, 256) == [] == cx.context_errors(cfg)
        assert _budgets(cfg, 256) == budgets
    with pytest.raises(ValueError, match="must be positive"):
        cx.resolve_fractions({"contexts": {"list": EIGHTHS, "granule": 0}}, 256)
    for bad in (0, -32):
        cfg = {"contexts": {"list": EIGHTHS, "granule": bad}}
        want = [f"contexts.granule {bad}: must be at least one CU"]
        assert cx.granule_errors(cfg) == want == cx.context_errors(cfg)
    for fine in (1, 32, 96):
        assert cx.granule_errors({"contexts": {"granule": fine}}) == []


FRACTIONS = {
    "list": [
        {"cu_budget": 0},
        {"cu_fraction": 0.75, "cotenant": True, "fraction": 0.15},
        {"cu_fraction": 0.5, "cotenant": True, "fraction": 0.15},
        {"cu_fraction": 0.25, "cotenant": True, "fraction": 0.15},
    ],
    "holdout": [
        {"cu_fraction": 0.875, "cotenant": True, "fraction": 0.03},
        {"cu_fraction": 0.375, "cotenant": True, "fraction": 0.03},
    ],
}


def test_fractions_resolve_for_the_devices_cus():
    cfg = {"contexts": FRACTIONS, "train": {"min_cu_fraction": 0.25}}
    assert cx.has_unresolved(cfg)
    with pytest.raises(ValueError, match="needs the device's CU count"):
        cx.contexts_from_config(cfg)
    out, lines = cx.resolve_fractions(cfg, 256)
    assert not cx.has_unresolved(out) and cfg["contexts"] == FRACTIONS
    assert [m.get("cu_budget") for m in out["contexts"]["list"]] == [0, 192, 128, 64]
    assert [m["cu_budget"] for m in out["contexts"]["holdout"]] == [224, 96]
    assert out["contexts"]["list"][1]["cu_fraction"] == 0.75
    assert lines[0] == "contexts.list[1]: cu_fraction 0.75 is 192 of 256 CUs"
    ctxs = _shares(out)
    assert [c.label for c in ctxs] == [
        "cu192_cot",
        "cu128_cot",
        "cu64_cot",
        "cu224_cot",
        "cu96_cot",
    ]
    assert _shares(cfg, 256) == ctxs
    assert cx.left_out(cfg, out) == []
    assert cx.resolve_fractions(out, 128) == (out, [])
    assert cx.context_errors(out, 256) == []


def test_fractions_that_meet_on_a_small_device_keep_the_first():
    cfg = {"contexts": FRACTIONS, "train": {"min_cu_fraction": 0.25}}
    out, lines = cx.resolve_fractions(cfg, 128)
    assert [m.get("cu_budget") for m in out["contexts"]["list"]] == [0, 96, 64, 32]
    assert out["contexts"]["holdout"] == []
    assert lines[3:] == [
        "contexts.holdout[0]: cu_fraction 0.875 is 96 of 128 CUs, the budget of "
        "contexts.list[1]; left out",
        "contexts.holdout[1]: cu_fraction 0.375 is 64 of 128 CUs, the budget of "
        "contexts.list[2]; left out",
    ]
    assert cx.context_errors(cfg, 128) == [] and cx.context_errors(out, 128) == []
    gone = cx.left_out(cfg, out)
    assert gone == [("holdout", i, FRACTIONS["holdout"][i]) for i in (0, 1)]
    tiny, _ = cx.resolve_fractions(cfg, 64)
    assert [c.cu_budget for c in _shares(tiny)] == [32]
    assert [(key, i) for key, i, _ in cx.left_out(cfg, tiny)] == [
        ("list", 2),
        ("list", 3),
        ("holdout", 0),
        ("holdout", 1),
    ]
    fixed = {"contexts": {"list": [{"cu_budget": 64, "cotenant": True}]}}
    fixed["contexts"]["holdout"] = [{"cu_fraction": 0.5, "cotenant": True}]
    out, (line,) = cx.resolve_fractions(fixed, 128)
    assert out["contexts"]["holdout"] == [] and line.endswith(
        "the budget of contexts.list[0]; left out"
    )
    assert [(key, i) for key, i, _ in cx.left_out(fixed, out)] == [("holdout", 0)]


def test_context_errors_of_fractions():
    dup = {"list": [{"cu_fraction": 0.5}, {"cu_fraction": 0.5, "name": "x"}]}
    (err,) = cx.context_errors({"contexts": dup})
    assert "contexts.list[1]: cu_fraction 0.5 / schedule default is also" in err
    named = {"list": [{"cu_fraction": 0.5, "name": "a"}, {"cu_budget": 8, "name": "a"}]}
    assert any("name 'a'" in e for e in cx.context_errors({"contexts": named}))
    small = {"contexts": {"list": [{"cu_fraction": 0.5}]}}
    assert cx.context_errors(small, 16) == []
    out, (line,) = cx.resolve_fractions(small, 16)
    assert out["contexts"]["list"] == [] and line == (
        "contexts.list[0]: cu_fraction 0.5 has no budget on a device of 16 CUs "
        "(a budget is a multiple of 32 CUs below its CU count); left out"
    )
    assert cx.left_out(small, out) == [("list", 0, {"cu_fraction": 0.5})]
    both = {"list": [{"cu_fraction": 0.5, "cu_budget": 96}]}
    (err,) = cx.context_errors({"contexts": both}, 256)
    assert err == (
        "contexts.list[0]: cu_budget 96 is not cu_fraction 0.5 of the device's "
        "256 CUs (128)"
    )
    assert cx.context_errors({"contexts": both}) == []
    assert cx.context_errors({"contexts": both}, 192) == []
    held = {"holdout": [{"cu_fraction": 0.5}], "list": [{"cu_fraction": 0.5}]}
    assert cx.context_errors({"contexts": held}, 256) == []


def test_trained_entries_and_reduced_cus():
    cfg = {"contexts": dict(FRACTIONS)}
    cfg["contexts"]["list"] = FRACTIONS["list"] + [{"schedule": "dynamic"}]
    trained = cx.trained_entries(cfg)
    assert trained == FRACTIONS["list"][1:] + [{"schedule": "dynamic"}]
    assert [cx.reduces_cus(m) for m in trained] == [True, True, True, False]
    assert cx.reduces_cus({"cu_budget": 64})


SHARES = {
    "list": [
        {"cu_budget": 0},
        {"cu_fraction": 0.75, "cotenant": True, "fraction": 0.1},
        {"cu_fraction": 0.5, "cotenant": True, "fraction": 0.1},
        {"cu_fraction": 0.25, "cotenant": True, "fraction": 0.1},
        {"cu_fraction": 0.125, "cotenant": True, "fraction": 0.1, "enabled": False},
    ],
    "holdout": [{"cu_fraction": 0.375, "cotenant": True, "enabled": False}],
}


def test_an_entry_turned_off_is_left_out_everywhere():
    cfg = {"contexts": SHARES}
    assert [(k, i) for k, i, _ in cx.off_entries(cfg)] == [("list", 4), ("holdout", 0)]
    assert cx.trained_entries(cfg) == SHARES["list"][1:4]
    assert cx.holdout_entries(cfg) == [] and cx.has_unresolved(cfg)
    assert _budgets(cfg, 256) == [192, 128, 64]
    out, lines = cx.resolve_fractions(cfg, 256)
    assert out["contexts"]["holdout"] == [] and cfg["contexts"] == SHARES
    assert [m.get("cu_fraction") for m in out["contexts"]["list"]] == [
        None,
        0.75,
        0.5,
        0.25,
    ]
    assert [m["cu_budget"] for m in out["contexts"]["list"]] == [0, 192, 128, 64]
    assert lines == [
        "contexts.list[1]: cu_fraction 0.75 is 192 of 256 CUs",
        "contexts.list[2]: cu_fraction 0.5 is 128 of 256 CUs",
        "contexts.list[3]: cu_fraction 0.25 is 64 of 256 CUs",
        "contexts.list[4]: cu_fraction 0.125 is off (enabled: false)",
        "contexts.holdout[0]: cu_fraction 0.375 is off (enabled: false)",
    ]
    assert cx.left_out(cfg, out) == [] and cx.context_errors(cfg, 256) == []
    explicit = {"contexts": {"list": [{"cu_budget": 64}, dict(SHARES["list"][4])]}}
    assert not cx.has_unresolved(explicit)
    with pytest.raises(ValueError, match="needs the device's CU count"):
        cx.resolve_fractions(cfg, None)
    out, (line,) = cx.resolve_fractions(explicit, None)
    assert out["contexts"]["list"] == [{"cu_budget": 64}]
    assert line == "contexts.list[1]: cu_fraction 0.125 is off (enabled: false)"
    assert [c.label for c in _shares(explicit)] == ["cu64"]
    budget = {"cu_budget": 32, "schedule": "dynamic", "enabled": False}
    assert cx.describe(budget) == "cu_budget 32 / schedule dynamic"


def test_turned_on_the_eighth_resolves_on_devices_of_eight_granules_up():
    on = {"contexts": {"list": [dict(m, enabled=True) for m in SHARES["list"]]}}
    assert _budgets(on, 256) == [192, 128, 64, 32]
    out, lines = cx.resolve_fractions(on, 128)
    assert [m["cu_budget"] for m in out["contexts"]["list"]] == [0, 96, 64, 32]
    assert lines[-1] == (
        "contexts.list[4]: cu_fraction 0.125 is 32 of 128 CUs, the budget of "
        "contexts.list[3]; left out"
    )


def test_a_device_too_small_for_every_share_keeps_the_exclusive_context():
    cfg = {"contexts": SHARES}
    out, lines = cx.resolve_fractions(cfg, 32)
    assert out["contexts"] == {"list": [{"cu_budget": 0}], "holdout": []}
    assert [ln.split(":")[0] for ln in lines if ln.endswith("left out")] == [
        "contexts.list[1]",
        "contexts.list[2]",
        "contexts.list[3]",
    ]
    assert cx.context_errors(cfg, 32) == [] and _shares(cfg, 32) == []
    assert [(k, i) for k, i, _ in cx.left_out(cfg, out)] == [
        ("list", 1),
        ("list", 2),
        ("list", 3),
    ]
    tiny = {"contexts": {"list": [{"cu_budget": 64, "cotenant": True}]}}
    (err,) = cx.context_errors(tiny, 32)
    assert "cu_budget 64 must be below the device's 32 CUs" in err


def test_a_share_below_the_cu_floor_and_an_exclusive_entry_turned_off_are_errors():
    floored = {"contexts": SHARES, "train": {"min_cu_fraction": 0.25}}
    assert cx.context_errors(floored) == []
    on = json.loads(json.dumps(floored))
    on["contexts"]["list"][4]["enabled"] = True
    (err,) = cx.context_errors(on)
    assert err.startswith(
        "contexts.list[4]: cu_fraction 0.125 is below train.min_cu_fraction 0.25, "
        "the smallest share the models may declare"
    )
    on["contexts"]["holdout"][0].update(enabled=True, cu_fraction=0.1)
    assert cx.context_errors(on) == [err]
    off = {"contexts": {"list": [{"cu_budget": 0, "enabled": False}]}}
    (err,) = cx.context_errors(off)
    assert err == (
        "contexts.list[0]: the exclusive context cannot be turned off (it is the "
        "round's own bench)"
    )


def test_orchestrator_validates_contexts(tmp_path):
    base, _ = ro.load_config(make_config(tmp_path))
    good = dict(base, contexts=_cfg()["contexts"])
    good["train"] = dict(base["train"], catalog="v3")
    assert ro.validate_config(good)[0] == []

    def errors(**patch):
        cfg = json.loads(json.dumps(good))
        for dotted, value in patch.items():
            cur = cfg
            *head, last = dotted.split(".")
            for p in head:
                cur = cur.setdefault(p, {})
            cur[last] = value
        return ro.validate_config(cfg)[0]

    assert any("need train.catalog v3" in e for e in errors(**{"train.catalog": "v2"}))
    assert any("unknown key" in e for e in errors(**{"contexts.list": [{"cus": 3}]}))
    assert any(
        "fraction" in e for e in errors(**{"contexts.list": [{"fraction": 0.0}]})
    )
    sched = errors(
        **{
            "contexts.list": [{"cu_budget": 64, "schedule": "dynamic"}],
            "train.schedules": ["default"],
        }
    )
    assert any("leaves out the benched schedule(s) ['dynamic']" in e for e in sched)
    low = errors(**{"hardware.n_cu": 256, "train.min_cu_fraction": 0.5})
    assert any("does not cover the trained budget fraction 0.25" in e for e in low)
    assert errors(**{"hardware.n_cu": 256, "train.min_cu_fraction": 0.25}) == []
    assert any(
        "needs planner.enabled" in e for e in errors(**{"contexts.replan": True})
    )
    held = dict(base, contexts={"holdout": [{"cu_budget": 96}]})
    assert ro.validate_config(held)[0] == []


def test_a_run_resolves_fractional_contexts_when_it_starts(
    tmp_path, stubs, env, capsys
):
    path = _context_config(
        tmp_path,
        **{
            "contexts.list": FRACTIONS["list"],
            "contexts.holdout": FRACTIONS["holdout"],
            "train.min_cu_fraction": 0.25,
            # Three rounds with no retrain before a split warn: round 1 has
            # no improvement baseline.
            "full.total_rounds": 4,
        },
    )
    cfg, _ = ro.load_config(path)
    assert ro.validate_config(cfg) == ([], [])
    capsys.readouterr()
    assert run(stubs, path, tmp_path / "unknown", "--mode", "initial") == 2
    assert "a cu_fraction needs the CU count of bench device 0" in (
        capsys.readouterr().err
    )
    assert not calls(env)

    path = _context_config(
        tmp_path,
        **{
            "contexts.list": FRACTIONS["list"],
            "contexts.holdout": FRACTIONS["holdout"],
            "train.min_cu_fraction": 0.25,
            "hardware.n_cu": 128,
        },
    )
    run_dir = tmp_path / "run"
    assert run(stubs, path, run_dir, "--mode", "initial") == 0
    out = capsys.readouterr()
    assert "contexts.holdout[0]: cu_fraction 0.875 is 96 of 128 CUs" in out.err
    assert "contexts.list[1]: cu_fraction 0.75 is 96 of 128 CUs" in out.out
    assert "contexts.holdout: every entry is left out on this device" in out.err
    log = calls(env)
    (snap,) = {Path(arg(c, "--config-yaml")) for c in log if arg(c, "--config-yaml")}
    text = snap.read_text()
    snapped = yaml.safe_load(text)["contexts"]
    assert [m.get("cu_budget") for m in snapped["list"]] == [0, 96, 64, 32]
    assert [m.get("cu_fraction") for m in snapped["list"]] == [None, 0.75, 0.5, 0.25]
    assert snapped["holdout"] == []
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert "Resolved at the start of this invocation: contexts: CU shares" in head
    assert "contexts.list[1]: cu_fraction 0.75 is 96 of 128 CUs;" in head
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    budgets = [s04[i + 1] for i, a in enumerate(s04) if a == "--context-log-dir"]
    assert budgets == ["96", "64", "32"]
    assert ro.validate_config(yaml.safe_load(text))[0] == []


def test_a_run_benches_the_shares_it_turns_on_that_fit_its_device(
    tmp_path, stubs, env, capsys
):
    """The shares the config turns on that the device can give are benched;
    those turned off are named and dropped from the snapshot; on a device
    with no budget below its CU count the run trains the exclusive context
    alone and says so."""

    def start(n_cu, run_dir):
        path = _context_config(
            tmp_path,
            **{
                "contexts.list": SHARES["list"],
                "contexts.holdout": SHARES["holdout"],
                "hardware.n_cu": n_cu,
            },
        )
        env.write_text("")
        capsys.readouterr()
        assert run(stubs, path, run_dir, "--mode", "initial") == 0
        log = calls(env)
        (snap,) = {
            Path(arg(c, "--config-yaml")) for c in log if arg(c, "--config-yaml")
        }
        return log, capsys.readouterr(), yaml.safe_load(snap.read_text())["contexts"]

    log, out, snapped = start(256, tmp_path / "whole")
    assert "contexts: CU shares of the 256 CUs of bench device 0 (hardware.n_cu)" in (
        out.out
    )
    assert "contexts.list[4]: cu_fraction 0.125 is off (enabled: false)" in out.out
    assert "contexts.holdout[0]: cu_fraction 0.375 is off" in out.out
    assert "left out" not in out.err
    assert [m.get("cu_budget") for m in snapped["list"]] == [0, 192, 128, 64]
    assert snapped["holdout"] == [] and "enabled" not in str(snapped)
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    budgets = [s04[i + 1] for i, a in enumerate(s04) if a == "--context-log-dir"]
    assert budgets == ["192", "128", "64"]

    log, out, snapped = start(32, tmp_path / "small")
    assert "contexts.list[1]: cu_fraction 0.75 has no budget on a device of 32 CUs" in (
        out.err
    )
    assert "contexts.list: every context of fewer CUs is left out on this device" in (
        out.err
    )
    assert "contexts.holdout: every entry is left out" not in out.err
    assert snapped == dict(snapped, list=[{"cu_budget": 0}], holdout=[])
    assert "stage03c" not in [c["stage"] for c in log]
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    assert "--context-log-dir" not in s04


def test_a_snapshot_names_the_entries_turned_off_as_read_from_no_device(tmp_path):
    """With no share to resolve, the snapshot's resolved lines are the
    entries the config turns off; it says no device value was read."""
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["contexts"] = {
        "list": [
            {"cu_budget": 0},
            {"cu_budget": 64, "cotenant": True, "fraction": 0.1, "enabled": False},
        ]
    }
    out, lines = ro.resolve_contexts(cfg, 0)
    assert lines == ["contexts.list[1]: cu_budget 64 is off (enabled: false)"]
    snap = ro.write_snapshot(tmp_path / "run", out, "", "initial", None, lines)
    text = snap.read_text()
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert (
        "Resolved at the start of this invocation: contexts.list[1]: cu_budget "
        "64 is off (enabled: false)" in head
    )
    assert "bench device" not in head
    assert yaml.safe_load(text)["contexts"]["list"] == [{"cu_budget": 0}]


def test_a_cu_floor_without_a_trained_reduced_context_warns(tmp_path):
    base, _ = ro.load_config(make_config(tmp_path))
    base["train"] = dict(base["train"], catalog="v3", min_cu_fraction=0.25)

    def warnings(contexts):
        cfg = dict(base, contexts=contexts)
        errors, warns = ro.validate_config(cfg)
        assert errors == []
        return [w for w in warns if w.startswith("train.min_cu_fraction")]

    (warn,) = warnings({})
    assert warn.startswith("train.min_cu_fraction 0.25: no trained context")
    assert len(warnings({"holdout": [{"cu_budget": 96, "cotenant": True}]})) == 1
    sched = {"list": [{"cu_budget": 0}, {"schedule": "dynamic"}]}
    assert len(warnings(dict(sched, holdout=[{"cu_fraction": 0.5}]))) == 1
    assert warnings({"list": [{"cu_budget": 128, "cotenant": True}]}) == []
    assert warnings({"list": [{"cu_fraction": 0.5, "cotenant": True}]}) == []
    base["train"]["min_cu_fraction"] = None
    assert warnings({}) == []


def test_preflight_needs_the_cotenant_kernel(tmp_path, monkeypatch):
    clients = tmp_path / "clients"
    clients.mkdir()
    launcher = clients / "hipblaslt-cotenant"

    def errors():
        """The errors with the launcher's path, which holds any text the
        test's base directory holds, as LAUNCHER."""
        return [
            e.replace(str(launcher), "LAUNCHER") for e in ro._cotenant_errors(launcher)
        ]

    assert errors() == ["hipblaslt-cotenant not found: LAUNCHER"]
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    monkeypatch.delenv("HIPCC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    monkeypatch.setattr(ro, "rocm_bin", lambda: str(tmp_path / "nothing"))
    fix = (
        "; build the bench's hipblaslt-cotenant-kernel target or set "
        "contexts.cotenant_binary"
    )
    assert errors() == [
        "contexts: LAUNCHER has no executable hipblaslt-cotenant-kernel beside it "
        "nor hipblaslt-cotenant-kernel.hip and no hipcc to build it" + fix
    ]
    source = clients / "hipblaslt-cotenant-kernel.hip"
    source.write_text("")
    assert errors() == [
        "contexts: LAUNCHER has no executable hipblaslt-cotenant-kernel beside it "
        "and no hipcc to build it" + fix
    ]
    monkeypatch.setenv("HIPCC", "/opt/hipcc")
    assert ro._cotenant_errors(launcher) == []
    source.unlink()
    assert errors() == [
        "contexts: LAUNCHER has no executable hipblaslt-cotenant-kernel beside it "
        "nor hipblaslt-cotenant-kernel.hip" + fix
    ]
    kernel = clients / "hipblaslt-cotenant-kernel"
    kernel.write_text("#!/bin/sh\n")
    kernel.chmod(0o755)
    assert ro._cotenant_errors(launcher) == []


def test_held_out_budgets_between_trained_ones_launch_as_their_own(capsys):
    """hipBLASLt takes every budget as its SM target as it is, so a held-out
    budget between trained ones holds out a launch of its own: preflight
    has nothing to warn about it, whatever the schedule."""

    def preflight(holdout):
        cfg = {
            "runtime": {"devices": [0]},
            "hardware": {"n_cu": 112},
            "contexts": {
                "list": [{"cu_budget": 0}, {"cu_budget": 64}, {"cu_budget": 32}],
                "holdout": holdout,
            },
        }
        assert ro._context_preflight(cfg, None) == []
        return capsys.readouterr().err

    for holdout in (
        [{"cu_budget": 84}, {"cu_budget": 40}],
        [{"cu_budget": 84}, {"cu_budget": 40, "schedule": "dynamic"}],
    ):
        assert "held-out" not in preflight(holdout)


# ── shape selection ──────────────────────────────────────────────────────────


def test_select_shapes_by_fraction_per_cell():
    shapes = [(m, n, 256, 1) for m in (16, 32, 4096, 8192) for n in (64, 8192)] * 2
    full = cx.select_shapes(shapes, cx.BenchContext(cu_budget=64), 1)
    assert full == list(dict.fromkeys(shapes))
    half = cx.BenchContext(cu_budget=64, fraction=0.5)
    a = cx.select_shapes(shapes, half, 1)
    assert set(a) == set(cx.select_shapes(list(reversed(shapes)), half, 1))
    assert a == cx.select_shapes(shapes, half, 1)
    assert [s for s in full if s in a] == a
    from lib.planner import shape_cell

    cells = {}
    for s in full:
        cells.setdefault(shape_cell(s), []).append(s)
    for members in cells.values():
        picked = [s for s in members if s in a]
        assert len(picked) == max(1, -(-len(members) // 2))
    other = cx.BenchContext(cu_budget=32, fraction=0.5)
    assert len(cx.select_shapes(shapes, other, 1)) == len(a)


def test_a_context_benches_the_shapes_targeted_at_its_tails(tmp_path):
    from lib.planner import shape_cell

    cell = [(4096 + i, 8192, 256, 1) for i in range(20)]
    other = [(16 + i, 64, 256, 1) for i in range(10)]
    shapes = cell + other
    ctx = cx.BenchContext(cu_budget=128, cotenant=True, fraction=0.1)
    plain = cx.select_shapes(shapes, ctx, 3)
    assert cx.select_shapes(shapes, ctx, 3, ()) == plain
    forced = {cell[5], cell[6], cell[7], (1, 1, 1, 1)}
    got = cx.select_shapes(shapes, ctx, 3, forced)
    assert forced - {(1, 1, 1, 1)} <= set(got) and (1, 1, 1, 1) not in got
    # the forced shapes take the cell's quota first, and the rest is drawn
    assert sum(1 for s in got if shape_cell(s) == shape_cell(cell[0])) == 3
    assert [s for s in got if s in other] == [s for s in plain if s in other]
    doc = {
        "leaves": {
            "L": [
                {"tail": {"cu_budget": 128, "schedule": "default"}, "shapes": [
                    list(cell[5]), list(cell[6])]},
                {"tail": {"cu_budget": 0, "schedule": "default"}, "shapes": [
                    list(cell[7])]},
                {"tail": {"cu_budget": 64, "schedule": "default"}, "shapes": [
                    list(cell[8])]},
            ]
        }
    }  # fmt: skip
    path = tmp_path / "targeted_shapes.json"
    path.write_text(json.dumps(doc))
    assert cx.targeted_for(path, ctx) == {cell[5], cell[6]}
    assert cx.targeted_for(path, cx.BenchContext(cu_budget=64)) == {cell[8]}
    assert cx.targeted_for(path, cx.BenchContext()) == set()
    assert cx.targeted_for(None, ctx) == set()
    assert cx.targeted_for(tmp_path / "missing.json", ctx) == set()
    # A seeded anchor's shapes go to the contexts its bench_contexts name,
    # whatever its own budget: both trained neighbours of an untrained one.
    doc["leaves"]["L"] += [
        {"anchor": {"cu_budget": "48", "bench_contexts": ["cu32_cot", "cu64_cot"]},
         "shapes": [list(cell[9])]},
        {"anchor": {"cu_budget": "0", "bench_contexts": []},
         "shapes": [list(cell[3])]},
    ]  # fmt: skip
    path.write_text(json.dumps(doc))
    cu64 = cx.BenchContext(cu_budget=64, cotenant=True, fraction=0.1)
    cu32 = cx.BenchContext(cu_budget=32, cotenant=True, fraction=0.1)
    assert (cu64.label, cu32.label) == ("cu64_cot", "cu32_cot")
    assert cx.targeted_for(path, cu64) == {cell[8], cell[9]}
    assert cx.targeted_for(path, cu32) == {cell[9]}
    assert cx.targeted_for(path, ctx) == {cell[5], cell[6]}
    assert cx.targeted_for(path, cx.BenchContext()) == set()


# ── bench wrapper ────────────────────────────────────────────────────────────


def _spec(tmp_path, ctx, **kw):
    return cx.bench_spec(
        ctx, 256, "/b/hipblaslt-bench", "/b/hipblaslt-cotenant", str(tmp_path), **kw
    )


def test_bench_command_with_a_cotenant(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin")
    ctx = cx.BenchContext(cu_budget=64, schedule="dynamic", cotenant=True)
    spec = _spec(tmp_path, ctx, library_path="/lib", path_append=["/rocm/bin"])
    cmd, env = cx.bench_command(spec, ["--device", "3", "--yaml", "f.yaml"], "s1")
    assert cmd == [
        "/b/hipblaslt-cotenant",
        "--cus",
        "192",
        "--device",
        "3",
        "--log",
        str(tmp_path / "cotenant_192_s1.log"),
        "--",
        "/b/hipblaslt-bench",
        "--device",
        "0",
        "--yaml",
        "f.yaml",
        "--sm_count_target",
        "64",
        "--hybrid_assignment_policy",
        "DynamicWorkQueue",
    ]
    assert env == {
        "HIPBLASLT_TENSILE_LIBPATH": "/lib",
        "PATH": os.pathsep.join(["/usr/bin", "/rocm/bin"]),
    }
    assert "LD_LIBRARY_PATH" not in env


def test_bench_command_without_a_cotenant(tmp_path):
    spec = _spec(tmp_path, cx.BenchContext(cu_budget=96))
    assert spec["cotenant"] is None and spec["hybrid_assignment_policy"] is None
    cmd, env = cx.bench_command(spec, ["--device=2", "--yaml", "f.yaml"], "s")
    assert cmd == [
        "/b/hipblaslt-bench",
        "--device",
        "2",
        "--yaml",
        "f.yaml",
        "--sm_count_target",
        "96",
    ]
    assert env == {}
    auto = _spec(tmp_path, cx.BenchContext(schedule="auto"))
    cmd, _ = cx.bench_command(auto, ["--yaml", "f.yaml"], "s")
    assert cmd == ["/b/hipblaslt-bench", "--yaml", "f.yaml"] + [
        "--hybrid_assignment_policy",
        "Auto",
    ]


def test_bench_spec_rejects_impossible_contexts(tmp_path):
    with pytest.raises(ValueError, match="needs hipblaslt-cotenant"):
        cx.bench_spec(cx.BenchContext(64, cotenant=True), 256, "b", None, "l")


def test_wrapper_runs_the_bench_through_the_cotenant(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    build = fh.write_fake_build(tmp_path / "build", "gfx950", bf.LIBRARY_STEM)
    echo = tmp_path / "echo-bench"
    echo.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'lib': "
        "os.environ.get('HIPBLASLT_TENSILE_LIBPATH'), "
        "'path': os.environ['PATH']}))\n"
    )
    echo.chmod(0o755)
    ctx = cx.BenchContext(cu_budget=48, cotenant=True)
    spec = cx.bench_spec(
        ctx,
        64,
        str(echo),
        str(build / "clients" / "hipblaslt-cotenant"),
        str(tmp_path / "cot"),
        library_path="/pool",
        path_append=["/rocm/bin"],
    )
    wrapper = cx.write_wrapper(
        tmp_path / "ctx", spec, sys.executable, PIPELINE_DIR / "lib/context_bench.py"
    )
    assert json.loads((tmp_path / "ctx" / cx.SPEC_FILE).read_text()) == spec
    proc = subprocess.run(
        [str(wrapper), "--device", "1", "--yaml", "x.yaml"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout.splitlines()
    assert out[0] == "launching cotenant on 16 CUs"
    got = json.loads(out[-1])
    assert got["argv"] == [
        "--device",
        "0",
        "--yaml",
        "x.yaml",
        "--sm_count_target",
        "48",
    ]
    assert got["lib"] == "/pool" and got["path"].endswith(os.pathsep + "/rocm/bin")
    (log,) = (tmp_path / "cot").glob("cotenant_16_*.log")
    assert log.read_text() == "cus 16 device 1\n"


def test_the_bench_a_wrapper_runs_is_found_behind_it(tmp_path):
    spec = _spec(tmp_path, cx.BenchContext(cu_budget=64, cotenant=True))
    wrapper = cx.write_wrapper(tmp_path / "ctx", spec, sys.executable, Path("x.py"))
    assert cx.wrapped_bench(wrapper) == Path("/b/hipblaslt-bench")
    plain = tmp_path / "hipblaslt-bench"
    assert cx.wrapped_bench(plain) == plain
    (tmp_path / "ctx" / cx.SPEC_FILE).write_text("not json")
    assert cx.wrapped_bench(wrapper) == wrapper


def test_cotenant_device_resolves_the_stage_visible_devices():
    assert cx.cotenant_device("1", {}) == "1"
    assert cx.cotenant_device("1", {"HIP_VISIBLE_DEVICES": "6, 3"}) == "3"
    assert cx.cotenant_device("0", {"CUDA_VISIBLE_DEVICES": "2"}) == "2"
    assert cx.cotenant_device("0", {"ROCR_VISIBLE_DEVICES": "4"}) == "0"
    for device, visible in (("1", "3"), ("0", "GPU-ab,1")):
        with pytest.raises(ValueError, match="would run elsewhere"):
            cx.cotenant_device(device, {"HIP_VISIBLE_DEVICES": visible})


def test_wrapper_keeps_the_cotenant_on_the_stage_devices(tmp_path):
    build = fh.write_fake_build(tmp_path / "build", "gfx950", bf.LIBRARY_STEM)
    echo = tmp_path / "echo-bench"
    echo.write_text(
        f"#!{sys.executable}\nimport os\n"
        "print(os.environ.get('HIP_VISIBLE_DEVICES'))\n"
    )
    echo.chmod(0o755)
    spec = cx.bench_spec(
        cx.BenchContext(cu_budget=48, cotenant=True),
        64,
        str(echo),
        str(build / "clients" / "hipblaslt-cotenant"),
        str(tmp_path / "cot"),
    )
    wrapper = cx.write_wrapper(
        tmp_path / "ctx", spec, sys.executable, PIPELINE_DIR / "lib/context_bench.py"
    )
    env = {k: v for k, v in os.environ.items() if not k.endswith("VISIBLE_DEVICES")}

    def run(device, visible):
        return subprocess.run(
            [str(wrapper), "--device", device, "--yaml", "x.yaml"],
            capture_output=True,
            text=True,
            env={**env, "HIP_VISIBLE_DEVICES": visible},
        )

    proc = run("1", "6,3")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "3"
    (log,) = (tmp_path / "cot").glob("cotenant_16_*.log")
    assert log.read_text() == "cus 16 device 3\n"
    proc = run("1", "3")
    assert proc.returncode == 2 and "would run elsewhere" in proc.stderr


# ── pool-only library directory ──────────────────────────────────────────────


def _write_logic(path, n_rows):
    doc = {
        "solutions": [{"index": i} for i in range(n_rows)],
        "library": {"type": "Matching", "rows": [[i] for i in range(n_rows)]},
    }
    raw = msgpack.packb(doc, use_bin_type=True)
    path.write_bytes(zlib.compress(raw) if path.name.endswith(".zlib") else raw)


def test_pool_libdir_empties_the_other_libraries_of_the_type(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pool = "TensileLibrary_BB_BB_HA_Bias_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    bank = pool.replace("_ID75a0", "_CU256_ID75a0")
    other_type = "TensileLibrary_HH_HH_HA_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    for name in (f"{pool}.dat", f"{bank}.dat.zlib", f"{other_type}.dat"):
        _write_logic(src / name, 3)
    (src / f"{bank}.co").write_bytes(b"code")
    assert pl.split_stem(pool) == (pool[: -len("_ID75a0_gfx950")], "_ID75a0_gfx950")
    assert pl.split_stem("TensileLibrary_X_gfx1250") == ("TensileLibrary_X", "_gfx1250")
    assert pl.split_stem("TensileLibrary_X_gfx1250-strict") == (
        "TensileLibrary_X",
        "_gfx1250-strict",
    )
    with pytest.raises(ValueError):
        pl.split_stem("TensileLibrary_noarch")
    dst = tmp_path / "dst"
    summary = pl.build_pool_libdir(src, [pool], dst)
    assert summary["emptied"] == [f"{bank}.dat.zlib"]
    assert (dst / f"{pool}.dat").is_symlink() and (dst / f"{bank}.co").is_symlink()
    assert (dst / f"{other_type}.dat").is_symlink()
    emptied = read_tensile_logic(dst / f"{bank}.dat.zlib")
    assert emptied["solutions"] == [] and emptied["library"]["rows"] == []
    assert len(read_tensile_logic(dst / f"{pool}.dat")["solutions"]) == 3
    assert pl.build_pool_libdir(src, [pool, bank], dst)["emptied"] == []
    assert (dst / f"{bank}.dat.zlib").is_symlink()
    with pytest.raises(FileNotFoundError):
        pl.build_pool_libdir(src, [pool.replace("BB_BB", "SS_SS")], dst)


def test_serving_libdir_serves_the_staged_model_with_the_librarys_kernels(tmp_path):
    import hashlib

    src = tmp_path / "src"
    src.mkdir()
    pool = "TensileLibrary_BB_BB_HA_Bias_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    other = "TensileLibrary_HH_HH_HA_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    # The index rows are keyed by the stem without its chip-ID part.
    pool_key = pool.replace("_ID75a0", "")
    other_key = other.replace("_ID75a0", "")
    _write_logic(src / f"{pool}.dat", 3)
    (src / f"{pool}.co").write_bytes(b"code")
    (src / "old.tilewright.bin").write_bytes(b"shipped model")
    (src / "h.tilewright.bin").write_bytes(b"other model")
    (src / "tilewright_index").write_text(
        f"# shipped\n{pool_key}\told.tilewright.bin\n"
        f"{other_key} h.tilewright.bin # kept\n"
        f"TensileLibrary_X_gfx950\tnew.tilewright.bin\n"
        f"{pool}\tlegacy.tilewright.bin\n"
    )
    model = tmp_path / "staged.bin"
    model.write_bytes(b"MLREC_v3 staged")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    dst = tmp_path / "validate" / "stage07_lib" / "gfx950"
    summary = pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model, sha)
    assert summary["other_stems"] == [other_key]
    assert summary["left_out_stems"] == ["TensileLibrary_X_gfx950"]
    assert summary["index_key"] == pool_key
    assert (dst / f"{pool}.dat").is_symlink() and (dst / f"{pool}.co").is_symlink()
    assert (dst / "old.tilewright.bin").is_symlink()
    assert not (dst / "new.tilewright.bin").is_symlink()
    assert (dst / "new.tilewright.bin").read_bytes() == b"MLREC_v3 staged"
    assert pl.index_entries((dst / "tilewright_index").read_text()) == [
        (other_key, "h.tilewright.bin"),
        (pool_key, "new.tilewright.bin"),
    ]
    assert (src / "tilewright_index").read_text().startswith("# shipped")
    # Rows for the devices of one PCI chip ID: the stem's give way to the one
    # row that serves the staged model on every device; other stems' stay,
    # and so does another stem's malformed row, for which the engine loads
    # no model.
    (src / "tilewright_index").write_text(
        f"{pool_key}\tchip.tilewright.bin\tchip=0x1a2b\n{pool_key}\told.tilewright.bin\n"
        f"{other_key}\th1.tilewright.bin\tchip=0x00ff\n{other_key}\th.tilewright.bin\n"
        f"{other_key} bad.bin chip=0xzz # typo\n{pool_key} worse.bin chip=0x12 more\n"
    )
    summary = pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model, sha)
    assert summary["other_stems"] == [other_key] and summary["left_out_stems"] == []
    assert summary["malformed_rows"] == [f"{other_key} bad.bin chip=0xzz # typo"]
    assert (dst / "tilewright_index").read_text() == (
        f"{other_key}\th.tilewright.bin\n{other_key}\th1.tilewright.bin\tchip=0x00ff\n"
        f"{other_key} bad.bin chip=0xzz # typo\n{pool_key}\tnew.tilewright.bin\n"
    )
    (dst / "leftover").write_text("from the last attempt")
    pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model)
    assert not (dst / "leftover").exists()
    with pytest.raises(ValueError, match="does not hash to"):
        pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model, "0" * 64)
    with pytest.raises(FileNotFoundError):
        pl.build_serving_libdir(src, dst, other, "new.tilewright.bin", model)
    with pytest.raises(ValueError, match="plain file name"):
        pl.build_serving_libdir(src, dst, pool, "../new.bin", model)
    tw = pytest.importorskip("tilewright")
    with pytest.raises(ValueError, match=f"malformed row for {other_key}"):
        tw.load_model_by_index(other_key, str(dst))
    with pytest.raises(ValueError, match="new.tilewright.bin"):
        tw.load_model_by_index(pool_key, str(dst), 0x12)


def test_serving_libdir_serves_the_staged_sidecars_in_place_of_the_models(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pool = "TensileLibrary_BB_BB_HA_Bias_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    _write_logic(src / f"{pool}.dat", 3)
    # The sidecars are named after the model file the staged model replaces.
    (src / "new.tilewright.blocklist").write_text("shipped blocklist\n")
    (src / "new.tilewright.guards").write_text("shipped guards\n")
    model = tmp_path / "staged.bin"
    model.write_bytes(b"MLREC_v3 staged")
    side = tmp_path / "staged.blocklist"
    side.write_text("Cijk_K\n")
    guards = tmp_path / "staged.guards"
    guards.write_text("Cijk_K\tM\t4\n")
    dst = tmp_path / "dst"
    summary = pl.build_serving_libdir(
        src, dst, pool, "new.tilewright.bin", model, blocklist=side, guards=guards
    )
    for name, text in (("blocklist", "Cijk_K\n"), ("guards", "Cijk_K\tM\t4\n")):
        f = dst / f"new.tilewright.{name}"
        assert not f.is_symlink() and f.read_text() == text
    assert summary["guards"] == str(guards) and summary["guards_sha256"]
    # Without staged guards the model has none, whatever the source holds.
    summary = pl.build_serving_libdir(
        src, dst, pool, "new.tilewright.bin", model, blocklist=side
    )
    assert not (dst / "new.tilewright.guards").exists()
    assert summary["guards"] is None and summary["guards_sha256"] is None


# ── orchestrator wiring ──────────────────────────────────────────────────────


def _context_config(tmp_path, **extra):
    """A config of CTX_LIST and a held-out budget, without the launch-keyed
    workspace contexts of rule version 3 unless `extra` asks for them."""
    return make_config(
        tmp_path,
        **{
            "train.catalog": "v3",
            "contexts.list": list(CTX_LIST),
            "contexts.holdout": [{"cu_budget": 96, "cotenant": True}],
            "contexts.workspaces": [],
            **extra,
        },
    )


def test_context_round_wiring(tmp_path, stubs, env):
    cfg = _context_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    rd0 = run_dir / "round_0"
    stages0 = [c["stage"] for c in log if "round_0" in (arg(c, "--out-dir") or "")]
    assert stages0[:4] == ["stage01", "stage03", "stage03c", "stage04"]
    s03c = [c for c in log if c["stage"] == "stage03c"][0]
    assert arg(s03c, "--out-dir") == str(rd0 / "stage03c")
    assert arg(s03c, "--shapes-yaml") == str(rd0 / "stage01" / "shapes.yaml")
    cmds = json.loads(arg(s03c, "--commands"))
    assert sorted(cmds) == ["stage03", "stage04"]
    s03 = {"argv": cmds["stage03"]}
    assert arg(s03, "--in-dir") == "{ctx_dir}"
    assert arg(s03, "--out-dir") == "{ctx_dir}/stage03"
    assert arg({"argv": cmds["stage04"]}, "--out-dir") == "{ctx_dir}/stage04"
    assert "--log-dir" not in cmds["stage04"]
    assert arg(s03c, "--path-append").endswith("bin")
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    i = s04.index("--context-log-dir")
    assert s04[i : i + 4] == [
        "--context-log-dir",
        "128",
        "default",
        str(rd0 / "stage03c" / "cu128_cot" / "stage03" / "logs"),
    ]
    assert s04.count("--context-log-dir") == 2
    s04b = [c for c in log if c["stage"] == "stage04b"][0]
    assert arg(s04b, "--context-holdout-dir") == str(
        run_dir / "round_1" / "stage03c" / "holdout" / "stage04"
    )
    single = tmp_path / "single"
    assert run(stubs, cfg, single, "--mode", "initial") == 0
    env.write_text("")
    assert run(stubs, cfg, single, "--mode", "initial", "--from-stage", "3.5") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03c", "stage04"]


def test_rebench_runs_before_the_context_benches(tmp_path, stubs, env):
    cfg = _context_config(tmp_path, **{"bench.rebench": {"enabled": True}})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    rd0 = run_dir / "round_0"
    log = [c for c in calls(env) if "round_0" in (arg(c, "--out-dir") or "")]
    assert [c["stage"] for c in log][:5] == [
        "stage01",
        "stage03",
        "stage03b",
        "stage03c",
        "stage04",
    ]
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(rd0 / "stage03b" / "logs")
    assert s04["argv"].count("--context-log-dir") == 2
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.2") == 0
    assert [c["stage"] for c in calls(env)][:3] == ["stage03b", "stage03c", "stage04"]
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.5") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03c", "stage04"]


def test_without_contexts_there_is_no_context_stage(tmp_path, stubs, env):
    cfg = make_config(tmp_path, **{"contexts.workspaces": []})
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    log = calls(env)
    assert "stage03c" not in [c["stage"] for c in log]
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    assert "--context-log-dir" not in s04


def test_pool_only_library_reaches_stage03(tmp_path):
    rd = tmp_path / "run" / "round_0"
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage03(rd)
    assert "--pool-only-library" not in plain
    cfg = make_config(tmp_path, **{"bench.pool_only_library": True})
    args = _pipeline(tmp_path, cfg).args_stage03(rd)
    assert arg({"argv": args}, "--pool-only-library") == STEM
    assert arg({"argv": args}, "--library-dir").endswith("gfx950")


def test_replanned_contexts_reach_stage02_and_stage03c(tmp_path):
    cfg = _context_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.model_file": str(tmp_path / "m.pt"),
            "contexts.replan": True,
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    wanted = json.loads(arg({"argv": p.args_stage02(rd, step)}, "--contexts"))
    assert [c["name"] for c in wanted] == ["cu128_cot", "cu64_cot", "cu96_cot"]
    assert [c["fraction"] for c in wanted] == [c.fraction for c in p.contexts]
    # Each re-plan ranks under the summation order and workspace it benches in.
    assert [(c["uniform_summation_order"], c["workspace_bytes"]) for c in wanted] == [
        (c.uniform_summation_order, c.workspace_bytes) for c in p.contexts
    ]
    args = p.args_stage03c(rd)
    assert arg({"argv": args}, "--context-shapes-dir") == str(
        rd / "stage02" / "contexts"
    )
    replan = rd / "stage02" / "contexts" / "cu64_cot" / "plan.json"
    replan.parent.mkdir(parents=True)
    replan.write_text("{}")
    s04 = p.args_stage04(rd)
    i = s04.index("--context-plan")
    assert s04[i : i + 4] == ["--context-plan", "64", "default", str(replan)]


def test_context_stage_cli_contract(tmp_path, pipeline_dir):
    cfg = _context_config(
        tmp_path,
        **{
            "bench.pool_only_library": True,
            "planner.enabled": True,
            "planner.model_file": str(tmp_path / "m.pt"),
            "contexts.replan": True,
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_1"
    step = ro.RoundStep(0, "initial", [], None, False)
    active = ro.RoundStep(1, "active", [rd.parent / "round_0"], None, True)
    s03c = p.args_stage03c(rd)
    cmds = json.loads(arg({"argv": s03c}, "--commands"))
    built = {
        "stage02": p.args_stage02(rd, step),
        "stage03": p.args_stage03(rd) + cmds["stage03"][1:],
        "stage03c": s03c,
        "stage04": p.args_stage04(rd) + cmds["stage04"][1:],
        "stage04b": p.args_stage04b(rd, active, rd.parent / "round_0" / "stage05"),
    }
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


# ── stage03c on the fake bench ───────────────────────────────────────────────


def test_stage03c_benches_each_context(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    """stage03c with the fake bench and cotenant, then stage04 with the
    context log directories: every trained context's rows carry its
    columns, a narrower budget changes winners, the held-out context is
    enriched on its own."""
    arch = "gfx950"
    build = fh.write_fake_build(tmp_path / "build", arch, bf.LIBRARY_STEM)
    libdir = build / "Tensile" / "library" / arch
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(libdir))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    shapes = [(4096, 4096, 1024, 1), (2048, 8192, 512, 1), (8192, 1024, 2048, 1)]
    shapes += [(64, 128, 256, 1), (1024, 1024, 4096, 1)]
    yaml_path = tmp_path / "shapes.yaml"
    yaml_path.write_text(
        "".join(
            f"- {{M: {m}, N: {n}, K: {k}, batch_count: {b}, transA: T, transB: N, "
            f"a_type: f8_r, b_type: f8_r, c_type: bf16_r, d_type: bf16_r, "
            f"compute_type: c_f32_r, requested_solution_num: -1, "
            f"skip_slow_solution_ratio: 0, iters: 1, cold_iters: 0}}\n"
            for m, n, k, b in shapes
        )
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        json.dumps(
            {
                "arch": arch,
                "runtime": {"build_dir": str(build)},
                "hipblaslt": {"library_stem": bf.LIBRARY_STEM},
                "hardware": {"n_cu": 64, "lds_bytes": 65536, "l2_bytes": 4194304},
                "contexts": {
                    "list": [
                        {"cu_budget": 0},
                        {"cu_budget": 16, "cotenant": True},
                        {"cu_budget": 32, "fraction": 0.5},
                    ],
                    "holdout": [{"cu_budget": 8, "cotenant": True}],
                    "seed": 3,
                    "workspaces": [],
                },
            }
        )
    )
    s03 = [
        str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
        "--in-dir",
        "{ctx_dir}",
        "--out-dir",
        "{ctx_dir}/stage03",
        "--bench-binary",
        str(build / "clients" / "hipblaslt-bench"),
        "--devices",
        "0",
        "--blocks-per-gpu",
        "1",
        "--stall-timeout-s",
        "120",
        "--startup-grace-s",
        "120",
    ]
    s04 = [
        str(PIPELINE_DIR / "stages" / "stage04_convert_to_enriched_dataset.py"),
        "--out-dir",
        "{ctx_dir}/stage04",
        "--dat-dir",
        str(libdir),
        "--library-stem",
        bf.LIBRARY_STEM,
        "--quiet",
    ]
    out = tmp_path / "stage03c"

    def stage03c(out, cmds):
        return s03c.main(
            [
                "--shapes-yaml",
                str(yaml_path),
                "--out-dir",
                str(out),
                "--config-yaml",
                str(cfg),
                "--bench-binary",
                str(build / "clients" / "hipblaslt-bench"),
                "--commands",
                json.dumps(cmds),
                "--pool-only-library",
                bf.LIBRARY_STEM,
                "--library-dir",
                str(libdir),
                "--path-append",
                str(tmp_path / "rocm" / "bin"),
                "--quiet",
            ]
        )

    stale = tmp_path / "stale"
    assert stage03c(stale, {"stage02": s03, "stage03": s03, "stage04": s04}) == 1
    assert not stale.exists()
    rc = stage03c(out, {"stage03": s03, "stage04": s04})
    summary = json.loads((out / "contexts.json").read_text())
    assert rc == 0, summary
    assert all(sorted(c["elapsed_s"]) == ["stage03"] for c in summary["contexts"])
    by_label = {r["name"]: r for r in summary["contexts"]}
    assert set(by_label) == {"cu16_cot", "cu32", "cu8_cot"}
    assert by_label["cu16_cot"]["n_shapes"] == len(shapes)
    assert by_label["cu32"]["n_shapes"] < len(shapes)
    assert by_label["cu16_cot"]["cotenant_cus"] == 48
    assert list((out / "cu16_cot" / "cotenant_logs").glob("cotenant_48_*.log"))
    assert not (out / "cu32" / "cotenant_logs").exists() or not list(
        (out / "cu32" / "cotenant_logs").glob("*.log")
    )
    spec = json.loads((out / "cu16_cot" / cx.SPEC_FILE).read_text())
    assert spec["library_path"] == str(out / "libdir_pool")

    rows = _enrich(
        tmp_path / "stage04",
        libdir,
        [
            ("--log-dir", None, None, None),
            ("--context-log-dir", "16", "default", by_label["cu16_cot"]["log_dir"]),
            ("--context-log-dir", "32", "default", by_label["cu32"]["log_dir"]),
        ],
        yaml_path,
        build,
    )
    by_ctx = {}
    for r in rows:
        key = (r.get("cu_budget") or "0", r.get("schedule") or "default")
        by_ctx.setdefault(key, set()).add((r["m"], r["n"], r["k"]))
    assert set(by_ctx) == {("0", "default"), ("16", "default"), ("32", "default")}
    assert len(by_ctx[("0", "default")]) == len(by_ctx[("16", "default")]) == 5
    assert len(by_ctx[("32", "default")]) == by_label["cu32"]["n_shapes"]
    summary04 = json.loads((tmp_path / "stage04" / "enrich_summary.json").read_text())
    assert summary04["gemms_by_context"]["16/default"] == 5
    win = {}
    for r in rows:
        key = (r.get("cu_budget") or "0", r["m"], r["n"], r["k"])
        if key not in win or float(r["us"]) < win[key][0]:
            win[key] = (float(r["us"]), r["sol_idx_global"])
    assert any(
        win[("0", m, n, k)][1] != win[("16", m, n, k)][1]
        for (b, m, n, k) in win
        if b == "16"
    )
    held = _csv_rows(out / "holdout" / "stage04")
    assert {r["cu_budget"] for r in held} == {"8"}


EXIT_AS = """
import os, signal, sys
how = sys.argv[1]
if how.startswith("SIG"):
    os.kill(os.getpid(), getattr(signal, how))
sys.exit(int(how))
"""


@pytest.mark.parametrize(
    "stage03, stage04, rc",
    [
        ("2", "0", 2),
        ("SIGKILL", "0", 2),
        ("1", "0", 1),
        ("SIGUSR1", "0", 1),
        ("0", "2", 1),
        ("0", "SIGTERM", 2),
    ],
)
def test_stage03c_exits_2_only_for_what_a_retry_may_cure(
    tmp_path, stage03, stage04, rc
):
    """A context's bench exit 2 or a sub-stage killed from outside: another
    attempt may cure it; any other failure (the held-out enrichment finding
    no rows included) is not GPU trouble."""
    shapes = tmp_path / "shapes.yaml"
    shapes.write_text(
        "- {M: 4096, N: 4096, K: 1024, batch_count: 1, transA: T, transB: N, "
        "a_type: f8_r, b_type: f8_r, c_type: bf16_r, d_type: bf16_r, "
        "compute_type: c_f32_r, requested_solution_num: -1}\n"
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        json.dumps(
            {
                "hardware": {"n_cu": 64, "lds_bytes": 65536, "l2_bytes": 4194304},
                "contexts": {
                    "list": [{"cu_budget": 32}],
                    "holdout": [{"cu_budget": 16}],
                    "workspaces": [],
                },
            }
        )
    )
    script = tmp_path / "exit_as.py"
    script.write_text(EXIT_AS)
    cmds = {"stage03": [str(script), stage03], "stage04": [str(script), stage04]}
    argv = ["--shapes-yaml", str(shapes), "--out-dir", str(tmp_path / "out")]
    argv += ["--config-yaml", str(cfg), "--bench-binary", str(tmp_path / "bench")]
    argv += ["--commands", json.dumps(cmds), "--quiet"]
    assert s03c.main(argv) == rc


def _csv_rows(directory):
    import csv

    rows = []
    for path in sorted(Path(directory).glob("chunk_*.csv")):
        with path.open() as f:
            rows += list(csv.DictReader(f))
    return rows


def _enrich(out, libdir, sources, yaml_path, build):
    """Run the exclusive bench of `yaml_path` through stage03 and stage04
    with the given context log directories."""
    s03_out = out.parent / "exclusive"
    s03_in = out.parent / "exclusive_in"
    s03_in.mkdir()
    (s03_in / "shapes.yaml").write_text(yaml_path.read_text())
    subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
            "--in-dir",
            str(s03_in),
            "--out-dir",
            str(s03_out),
            "--bench-binary",
            str(build / "clients" / "hipblaslt-bench"),
            "--devices",
            "0",
            "--blocks-per-gpu",
            "1",
            "--stall-timeout-s",
            "120",
            "--startup-grace-s",
            "120",
        ],
        check=True,
        capture_output=True,
    )
    args = [
        sys.executable,
        str(PIPELINE_DIR / "stages" / "stage04_convert_to_enriched_dataset.py"),
        "--out-dir",
        str(out),
        "--dat-dir",
        str(libdir),
        "--library-stem",
        bf.LIBRARY_STEM,
        "--quiet",
    ]
    for flag, budget, sched, d in sources:
        if flag == "--log-dir":
            args += [flag, str(s03_out / "logs")]
        else:
            args += [flag, budget, sched, d]
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return _csv_rows(out)


# ── summation order and workspace (launch rule version 3) ────────────────────


@pytest.fixture
def rule_v3(monkeypatch):
    from lib import hipblaslt_launch as hl

    monkeypatch.setattr(hl, "RULE_VERSION", 3)


@pytest.fixture
def rule_v2(monkeypatch):
    from lib import hipblaslt_launch as hl

    monkeypatch.setattr(hl, "RULE_VERSION", 2)


def test_a_context_names_its_summation_order_and_workspace_in_its_label():
    assert cx.BenchContext().label == "allcu" and cx.BenchContext().is_exclusive()
    off = cx.BenchContext(uniform_summation_order=False)
    assert off.label == "allcu_usooff" and not off.is_exclusive()
    assert cx.BenchContext(cu_budget=64, cotenant=True, workspace_bytes=0).label == (
        "cu64_cot_ws0m"
    )
    assert not cx.BenchContext(workspace_bytes=0).is_exclusive()
    assert cx.BenchContext(workspace_bytes=3 << 20).label == "allcu_ws3m"
    assert cx.BenchContext(workspace_bytes=1000).label == "allcu_ws1000b"
    assert cx.FULL_WORKSPACE_BYTES == cx.MAX_WORKSPACE_BYTES == (1 << 32) - 1


@pytest.mark.parametrize(
    "value, want",
    [
        (None, True),
        (True, True),
        (False, False),
        ("on", True),
        ("OFF", False),
        (1, True),
        (0, False),
    ],
)
def test_a_summation_order_is_on_or_off(value, want):
    m = {"cu_budget": 64, "uniform_summation_order": value}
    assert cx.BenchContext.from_mapping(m).uniform_summation_order is want


def test_a_config_spells_the_summation_order_on_and_off():
    m = yaml.safe_load(
        "{cu_budget: 64, uniform_summation_order: off, workspace_bytes: 0}"
    )
    ctx = cx.BenchContext.from_mapping(m)
    assert (ctx.uniform_summation_order, ctx.workspace_bytes) == (False, 0)
    assert (
        cx.describe(m)
        == "cu_budget 64 / uniform_summation_order off / workspace_bytes 0"
    )


@pytest.mark.parametrize(
    "entry, needle",
    [
        ({"cu_budget": 64, "uniform_summation_order": "sometimes"}, "give on or off"),
        ({"cu_budget": 64, "uniform_summation_order": 2}, "give on or off"),
        ({"cu_budget": 64, "workspace_bytes": -1}, "from 0 to 4294967295"),
        ({"cu_budget": 64, "workspace_bytes": 1 << 32}, "from 0 to 4294967295"),
        ({"cu_budget": 64, "workspace_bytes": "big"}, "whole number of bytes"),
        ({"cu_budget": 64, "workspace_bytes": True}, "whole number of bytes"),
        ({"cu_fraction": 0.5, "workspace_bytes": 0.5}, "whole number of bytes"),
    ],
)
def test_context_errors_name_a_bad_summation_order_or_workspace(entry, needle):
    errors = cx.context_errors({"contexts": {"list": [entry]}}, 256)
    assert len(errors) == 1, errors
    assert errors[0].startswith("contexts.list[0]: ") and needle in errors[0]
    if "cu_budget" in entry:
        with pytest.raises(ValueError):
            cx.BenchContext.from_mapping(entry)


def test_the_config_schema_takes_the_summation_order_and_workspace(tmp_path):
    entry = {"cu_budget": 0, "uniform_summation_order": False, "workspace_bytes": 0}
    good = {
        "train.catalog": "v3",
        "contexts.list": [{"cu_budget": 0}, entry],
        "contexts.workspaces": [0],
    }
    base, _ = ro.load_config(make_config(tmp_path, **good))

    def errors(**kw):
        cfg = json.loads(json.dumps(base))
        for dotted, value in kw.items():
            *path, last = dotted.split(".")
            cur = cfg
            for p in path:
                cur = cur[p]
            cur[last] = value
        return ro.validate_config(cfg)[0]

    assert errors() == []
    for bad, needle in (
        (
            {"cu_budget": 0, "workspace_bytes": 1 << 32},
            "contexts.list[1].workspace_bytes",
        ),
        (
            {"cu_budget": 0, "uniform_summation_order": "sometimes"},
            "uniform_summation_order",
        ),
        ({"cu_budget": 0, "launch_keyed": 1}, "launch_keyed"),
    ):
        got = errors(**{"contexts.list": [{"cu_budget": 0}, bad]})
        assert any(needle in e for e in got), (bad, got)
    got = errors(**{"contexts.workspaces": [-1]})
    assert any("contexts.workspaces[0]" in e for e in got), got


def test_contexts_that_differ_in_order_or_workspace_are_distinct():
    sec = {
        "list": [
            {"cu_budget": 64},
            {"cu_budget": 64, "workspace_bytes": 0},
            {"cu_budget": 64, "uniform_summation_order": "off"},
        ]
    }
    assert cx.context_errors({"contexts": sec}) == []
    labels = [c.label for c in _shares({"contexts": sec})]
    assert labels == ["cu64", "cu64_ws0m", "cu64_usooff"]
    sec["list"].append({"cu_budget": 64, "workspace_bytes": 0})
    errors = cx.context_errors({"contexts": sec})
    assert errors and all(e.startswith("contexts.list[3]: ") for e in errors)
    assert any(
        "also contexts.list[1]" in e and "summation order and workspace" in e
        for e in errors
    )


def test_fractions_that_differ_in_workspace_resolve_to_two_contexts():
    cfg = {
        "contexts": {
            "list": [{"cu_fraction": 0.5}, {"cu_fraction": 0.5, "workspace_bytes": 0}]
        }
    }
    out, _ = cx.resolve_fractions(cfg, 256)
    assert [m["cu_budget"] for m in out["contexts"]["list"]] == [128, 128]
    assert cx.left_out(cfg, out) == [] and cx.context_errors(cfg, 256) == []
    same = {"contexts": {"list": [{"cu_fraction": 0.5}, {"cu_fraction": 0.5}]}}
    out, said = cx.resolve_fractions(same, 256)
    assert len(out["contexts"]["list"]) == 1 and "left out" in said[1]


def test_line_knobs_carry_the_context_from_rule_version_three_on():
    ctx = cx.BenchContext(uniform_summation_order=False, workspace_bytes=0)
    assert cx.line_knobs(ctx, 1) == cx.line_knobs(ctx, 2) == {}
    assert cx.line_knobs(ctx, 3) == {
        "uniform_summation_order": 0,
        "user_allocated_workspace": 0,
        "workspace_per_process": "true",
    }
    full = cx.line_knobs(cx.BenchContext(), 3)
    assert (full["uniform_summation_order"], full["user_allocated_workspace"]) == (
        1,
        cx.FULL_WORKSPACE_BYTES,
    )


def test_no_workspace_context_before_rule_version_three(rule_v2):
    assert cx.workspace_contexts({"contexts": {}}) == []
    assert [c.label for c in cx.contexts_from_config(_cfg())] == [
        "cu128_cot",
        "cu64_cot",
    ]


def test_rule_version_three_adds_a_launch_keyed_context_per_workspace(rule_v3):
    ctxs = cx.contexts_from_config(_cfg(holdout=[{"cu_budget": 96}]))
    assert [c.label for c in ctxs] == ["cu128_cot", "cu64_cot", "allcu_ws0m", "cu96"]
    ws = ctxs[2]
    assert ws.launch_keyed and ws.workspace_bytes == 0 and ws.cu_budget == 0
    assert ws.uniform_summation_order and ws.fraction == 1.0 and not ws.holdout
    two = cx.contexts_from_config(_cfg(workspaces=[0, 32 << 20]))
    assert [c.label for c in two if c.launch_keyed] == ["allcu_ws0m", "allcu_ws32m"]
    assert not [
        c for c in cx.contexts_from_config(_cfg(workspaces=[])) if c.launch_keyed
    ]
    given = _cfg()
    given["contexts"]["list"].append(
        {"cu_budget": 0, "workspace_bytes": 0, "name": "nows"}
    )
    assert [
        c.label for c in cx.contexts_from_config(given) if c.workspace_bytes == 0
    ] == ["nows"]
    assert cx.context_errors(_cfg(workspaces=[0, 32 << 20])) == []


def test_a_configured_full_workspace_is_the_rounds_own(rule_v3):
    """`contexts.full_workspace_bytes` sets the workspace of the round's own
    context: an entry that names it or none is the exclusive context, a
    label names any other, and the launch pass and the lines use it."""
    from lib import bench_yaml

    small = 1 << 30
    cfg = _cfg(full_workspace_bytes=small)
    launch = cx.round_launch(cfg)
    assert launch == (True, small)
    assert cx.full_workspace(_cfg()) == cx.FULL_WORKSPACE_BYTES
    own = cx.round_context(cfg)
    assert own.is_exclusive() and own.label == "allcu" and own.workspace_bytes == small
    entry = {"cu_budget": 0, "workspace_bytes": small}
    assert cx.BenchContext.from_mapping(entry, launch=launch).is_exclusive()
    assert cx.describe(entry, launch) == "cu_budget 0"
    full = {"cu_budget": 64, "workspace_bytes": cx.FULL_WORKSPACE_BYTES}
    label = cx.BenchContext.from_mapping(full, launch=launch).label
    assert label == f"cu64_ws{cx.FULL_WORKSPACE_BYTES}b"
    assert (
        cx.BenchContext.from_mapping({"cu_budget": 64}, launch=launch).label == "cu64"
    )
    (ws,) = [c for c in cx.contexts_from_config(cfg) if c.launch_keyed]
    assert (ws.label, ws.workspace_bytes, ws.round_workspace) == (
        "allcu_ws0m",
        0,
        small,
    )
    seen = []
    shapes = [(256, 256, 8192, 1)]
    cx.launch_changing_shapes(
        shapes, ws, [{"name": "b"}], _splitting_launch(seen), 64, ""
    )
    assert {c.workspace_bytes for *_, c, _ in seen} == {0, small}
    knobs = bench_yaml.context_knobs(cfg)
    assert (knobs["user_allocated_workspace"], knobs["uniform_summation_order"]) == (
        small,
        1,
    )
    assert cx.context_errors(cfg) == []
    (err,) = cx.workspace_errors(_cfg(full_workspace_bytes=small, workspaces=[small]))
    assert err.endswith("the full workspace is the round's own bench")
    (err,) = cx.context_errors(_cfg(full_workspace_bytes=0))
    assert err.startswith("contexts.full_workspace_bytes: ")
    with pytest.raises(ValueError, match="full workspace holds partial tiles"):
        cx.contexts_from_config(_cfg(full_workspace_bytes=0))


@pytest.mark.parametrize(
    "arch, order",
    [
        ("gfx950", True),
        ("gfx942", True),
        ("gfx1100", False),
        ("gfx1151", False),
        ("gfx1201:xnack-", False),
        (None, True),
    ],
)
def test_the_rounds_own_order_is_the_handle_default_of_its_arch(rule_v3, arch, order):
    """hipBLASLt's handle turns the order off on RDNA, so the round's own
    context has it off there unless the config sets it: an entry that names
    no order takes the round's, and one that names the other order is no
    exclusive context."""
    cfg = dict(_cfg(), arch=arch)
    assert cx.round_order(cfg) is order
    own = cx.round_context(cfg)
    assert own.uniform_summation_order is order and own.is_exclusive()
    launch = cx.round_launch(cfg)
    assert (
        cx.BenchContext.from_mapping({"cu_budget": 64}, launch=launch).label == "cu64"
    )
    other = {"cu_budget": 0, "uniform_summation_order": not order}
    assert not cx.BenchContext.from_mapping(other, launch=launch).is_exclusive()
    (ws,) = [c for c in cx.contexts_from_config(cfg) if c.launch_keyed]
    assert ws.uniform_summation_order is order
    flipped = dict(cfg, contexts=dict(cfg["contexts"]))
    flipped["contexts"][cx.ROUND_ORDER_KEY] = not order
    assert cx.round_order(flipped) is (not order)
    assert cx.context_errors(cfg) == [] and cx.context_errors(flipped) == []


def test_an_override_of_the_rounds_order_is_on_or_off():
    assert cx.round_order(_cfg(round_uniform_summation_order="off")) is False
    (err,) = cx.context_errors(_cfg(round_uniform_summation_order="maybe"))
    assert err.startswith("contexts.round_uniform_summation_order: ")


@pytest.mark.parametrize(
    "sizes, needle",
    [
        ([(1 << 32) - 1], "the round's own bench"),
        ([None], "the round's own bench"),
        ([0, 0], "listed before"),
        ([-1], "from 0 to"),
        ("0", "give a list"),
    ],
)
def test_context_errors_name_a_bad_workspace_list(sizes, needle, rule_v3):
    errors = cx.context_errors(_cfg(workspaces=sizes))
    assert len(errors) == 1, errors
    assert errors[0].startswith("contexts.workspaces") and needle in errors[0]
    assert not [
        c for c in cx.contexts_from_config(_cfg(workspaces=sizes)) if c.launch_keyed
    ]


def test_bench_spec_records_the_summation_order_and_workspace():
    ctx = cx.BenchContext(workspace_bytes=0, uniform_summation_order=False)
    spec = cx.bench_spec(ctx, 256, "b", None, "l")
    assert (spec["uniform_summation_order"], spec["workspace_bytes"]) == (0, 0)
    assert spec["context"]["name"] == "allcu_usooff_ws0m"
    assert spec["context"]["launch_keyed"] is False
    assert spec["context"]["uniform_summation_order"] is False


def _splitting_launch(seen):
    """A launch in which kernel b splits the tiles of deep shapes, which
    needs a workspace, and every other launch is the same."""

    def launch(problem, kernel, attributes, context, n_cu):
        seen.append((problem, kernel["name"], attributes, context, n_cu))
        split = kernel["name"] == "b" and problem["k"] >= 4096
        if not split:
            return ("data-parallel",)
        return ("tree",) if context.workspace_bytes == 0 else ("parallel",)

    return launch


def test_launch_changing_shapes_keeps_the_lines_a_workspace_changes():
    seen = []
    kernels = [{"name": "a", "attributes": {"x": 1}}, {"name": "b"}]
    shapes = [(256, 256, 1024, 1), (256, 256, 8192, 1), (128, 128, 4096, 2)]
    ctx = cx.BenchContext(
        uniform_summation_order=False, workspace_bytes=0, launch_keyed=True
    )
    out = cx.launch_changing_shapes(
        shapes, ctx, kernels, _splitting_launch(seen), 64, "bfloat16"
    )
    assert out == [(256, 256, 8192, 1), (128, 128, 4096, 2)]
    assert {
        (c.workspace_bytes, c.uniform_summation_order) for _, _, _, c, _ in seen
    } == {
        (0, False),
        (cx.FULL_WORKSPACE_BYTES, False),
    }
    assert {n for *_, n in seen} == {64}
    assert all(p["mi_dtype"] == "bfloat16" for p, *_ in seen)
    assert (seen[0][0]["m"], seen[0][0]["batch"]) == (256, 1)
    assert any(name == "a" and attrs == {"x": 1} for _, name, attrs, _, _ in seen)
    assert (
        cx.launch_changing_shapes(
            shapes, ctx, [kernels[0]], _splitting_launch([]), 64, ""
        )
        == []
    )


def _context_bench_args(tmp_path):
    return argparse.Namespace(
        out_dir=tmp_path / "out",
        context_shapes_dir=None,
        targeted_shapes=None,
        bench_binary=tmp_path / "hipblaslt-bench",
        path_append=[],
        quiet=True,
    )


_SHAPES = [(256, 256, 1024, 1), (256, 256, 8192, 1)]
_LINES = {
    s: f"- {{function: matmul, M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}, "
    f"uniform_summation_order: -1}}\n"
    for s in _SHAPES
}
_NO_OP = {"stage03": ["-c", "pass"]}


def test_stage03c_leaves_the_lines_alone_before_rule_version_three(tmp_path, rule_v2):
    args = _context_bench_args(tmp_path)
    ctx = cx.BenchContext(cu_budget=32)
    rec = s03c._bench_context(
        ctx, args, {}, _NO_OP, _LINES, _SHAPES, 1, 64, tmp_path, None
    )
    assert rec["rc"] == 0 and rec["n_shapes"] == 2 and "n_shapes_unchanged" not in rec
    assert (args.out_dir / "cu32" / "shapes.yaml").read_text() == "".join(
        _LINES.values()
    )


def test_stage03c_writes_each_context_with_its_order_and_workspace(tmp_path, rule_v3):
    args = _context_bench_args(tmp_path)
    ctx = cx.BenchContext(cu_budget=32, uniform_summation_order=False)
    rec = s03c._bench_context(
        ctx, args, {}, _NO_OP, _LINES, _SHAPES, 1, 64, tmp_path, None
    )
    assert rec["rc"] == 0 and rec["n_shapes"] == 2
    assert (rec["uniform_summation_order"], rec["workspace_bytes"]) == (
        False,
        cx.FULL_WORKSPACE_BYTES,
    )
    lines = (args.out_dir / "cu32_usooff" / "shapes.yaml").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "uniform_summation_order: 0" in line and "-1" not in line
        assert f"user_allocated_workspace: {cx.FULL_WORKSPACE_BYTES}" in line
        assert line.endswith("workspace_per_process: true}")
    spec = json.loads((args.out_dir / "cu32_usooff" / cx.SPEC_FILE).read_text())
    assert (spec["uniform_summation_order"], spec["sm_count_target"]) == (0, 32)


def test_stage03c_benches_a_launch_keyed_context_where_a_launch_changes(
    tmp_path, rule_v3
):
    args = _context_bench_args(tmp_path)
    ws = cx.BenchContext(workspace_bytes=0, launch_keyed=True)
    launches = {
        "kernels": [{"name": "b"}],
        "launch": _splitting_launch([]),
        "mi_dtype": "",
    }
    rec = s03c._bench_context(
        ws, args, {}, _NO_OP, _LINES, _SHAPES, 1, 64, tmp_path, None, launches
    )
    assert (rec["n_shapes"], rec["n_shapes_unchanged"]) == (1, 1)
    (line,) = (args.out_dir / "allcu_ws0m" / "shapes.yaml").read_text().splitlines()
    assert "K: 8192" in line and "user_allocated_workspace: 0," in line
    assert "uniform_summation_order: 1" in line


# ── co-tenant contexts: no clustered kernel ──────────────────────────────────


class _Pool:
    def __init__(self, kernels):
        self.kernels = kernels


def test_a_cotenant_context_splits_the_pool_by_its_clusters(tmp_path, monkeypatch):
    """A kernel whose ClusterDim spans more than one workgroup is clustered;
    the pool is the config's, from the build's library of its arch
    (`runtime.build_dir`, else the release build of the checkout) unless a
    library directory is given."""
    from lib import library_pool

    calls = []

    def pool(cfg, dat_dir, stem):
        calls.append((Path(dat_dir), stem))
        return _Pool(
            [
                {"sol_idx_global": 9, "attributes": {"cluster_dim_x": 2}},
                {"sol_idx_global": 3, "attributes": {}},
                {
                    "sol_idx_global": 8,
                    "attributes": {
                        "cluster_dim_x": 1,
                        "cluster_dim_y": 1,
                        "cluster_dim_z": 1,
                    },
                },
                {
                    "sol_idx_global": 4,
                    "attributes": {"cluster_dim_x": 4, "cluster_dim_y": 4},
                },
                {"sol_idx_global": 6, "attributes": None},
                {"sol_idx_global": 5, "attributes": {"cluster_dim_z": 2}},
            ]
        )

    monkeypatch.setattr(library_pool, "config_pool", pool)
    cfg = {
        "arch": "gfx1250:xnack-",
        "runtime": {"build_dir": str(tmp_path / "build")},
        "hipblaslt": {"library_stem": "TensileLibrary_X_gfx1250"},
    }
    assert cx.cotenant_kernels(cfg) == ({4, 5, 9}, [3, 6, 8])
    assert cx.cotenant_kernels(cfg, tmp_path / "lib") == ({4, 5, 9}, [3, 6, 8])
    released = dict(cfg, runtime={"rocm_libraries_root": str(tmp_path / "src")})
    assert cx.cotenant_kernels(released) == ({4, 5, 9}, [3, 6, 8])
    library = tmp_path / "build" / "Tensile" / "library" / "gfx1250"
    release = tmp_path / "src" / "projects" / "hipblaslt" / "build" / "release"
    assert calls == [
        (library, "TensileLibrary_X_gfx1250"),
        (tmp_path / "lib", "TensileLibrary_X_gfx1250"),
        (release / "Tensile" / "library" / "gfx1250", "TensileLibrary_X_gfx1250"),
    ]


def test_a_cotenant_line_benches_no_clustered_kernel():
    """A line that lists kernels loses the clustered ones, which can leave
    its list empty; one that benches the whole library lists the pool's
    unclustered kernels instead."""
    from lib.bench_yaml import line_knob, line_solution_list

    listed = (
        "- {function: matmul, M: 256, N: 256, K: 64, batch_count: 1, "
        "solution_index_list: [7, 9, 11]}\n"
    )
    assert line_solution_list(cx.cotenant_line(listed, {9}, [3, 7, 11])) == [7, 11]
    assert cx.cotenant_line(listed, set(), [3]) == listed
    only = listed.replace("[7, 9, 11]", "[9]")
    assert line_solution_list(cx.cotenant_line(only, {9}, [3])) == []
    whole = (
        "- {function: matmul, M: 256, N: 256, K: 64, batch_count: 1, "
        "algo_method: all, requested_solution_num: 50}\n"
    )
    out = cx.cotenant_line(whole, {9}, [3, 7, 11])
    assert line_solution_list(out) == [3, 7, 11] and out.endswith("}\n")
    assert line_knob(out, "algo_method") is None
    assert line_knob(out, "requested_solution_num") == "-1"


def test_stage03c_benches_no_clustered_kernel_next_to_a_cotenant(
    tmp_path, rule_v3, monkeypatch
):
    """A co-tenant context's lines leave out the pool's clustered kernels and
    a line left with none; a line that benches the whole library lists the
    unclustered ones. A context without a co-tenant keeps every kernel."""
    from lib.bench_yaml import line_solution_list

    shapes = [(256, 256, 1024, 1), (256, 256, 8192, 1), (128, 128, 4096, 2)]
    listed = {shapes[0]: "[7, 9]", shapes[1]: "[9]"}
    lines = {
        s: f"- {{function: matmul, M: {s[0]}, N: {s[1]}, K: {s[2]}, "
        f"batch_count: {s[3]}, "
        + (f"solution_index_list: {listed[s]}, " if s in listed else "")
        + "uniform_summation_order: -1}\n"
        for s in shapes
    }
    monkeypatch.setattr(s03c, "cotenant_kernels", lambda cfg: ({9}, [3, 7]))
    args = _context_bench_args(tmp_path)
    cot = cx.BenchContext(cu_budget=32, cotenant=True)
    rec = s03c._bench_context(
        cot, args, {}, _NO_OP, lines, shapes, 1, 64, tmp_path, None
    )
    assert rec["rc"] == 0 and rec["n_shapes"] == 2
    written = (args.out_dir / "cu32_cot" / "shapes.yaml").read_text().splitlines()
    assert [line_solution_list(x) for x in written] == [[7], [3, 7]]
    assert all("K: 8192" not in x for x in written)
    alone = cx.BenchContext(cu_budget=32)
    rec = s03c._bench_context(
        alone, args, {}, _NO_OP, lines, shapes, 1, 64, tmp_path, None
    )
    assert rec["n_shapes"] == 3
    kept = (args.out_dir / "cu32" / "shapes.yaml").read_text().splitlines()
    assert [line_solution_list(x) for x in kept] == [[7, 9], [9], None]


def test_the_audit_benches_no_clustered_kernel_in_a_cotenant_unit(
    tmp_path, monkeypatch
):
    """A co-tenant unit's lines leave out the pool's clustered kernels, the
    split taken once from the run's library, and then the blocked kernels; a
    line left with none is not benched. An exclusive unit keeps every kernel
    the blocklist leaves."""
    import types

    from lib.bench_yaml import line_solution_list

    a = ro.AuditRun.__new__(ro.AuditRun)
    a._blocked = {11}
    a._guarded = {}
    a.shapes_this_run = 0
    a.p = types.SimpleNamespace(
        cfg={"arch": "gfx1250"},
        paths=types.SimpleNamespace(library_dir=tmp_path / "lib"),
    )
    splits = []

    def split(cfg, library_dir):
        splits.append((cfg, library_dir))
        return {9}, [3, 7, 11]

    monkeypatch.setattr(ro.ctxlib, "cotenant_kernels", split)
    benched = {}
    monkeypatch.setattr(a, "wrapper", lambda label, unit: label, raising=False)
    monkeypatch.setattr(
        a,
        "bench_part",
        lambda part, lines, w: benched.setdefault(w, []).extend(lines) or 0,
        raising=False,
    )
    monkeypatch.setattr(a, "write_progress", lambda *x: None, raising=False)
    shapes = [(64, 64, 64, 1), (128, 64, 64, 1), (256, 64, 64, 1)]
    listed = {64: "solution_index_list: [7, 9, 11]", 128: "solution_index_list: [9]"}

    def line_of(label, i, s):
        knob = listed.get(s[0], "algo_method: all")
        return (
            f"- {{M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}, {knob}, "
            f"requested_solution_num: 8}}\n"
        )

    cot = {"cu_budget": 192, "cotenant": True}
    units = {
        "exclusive": {"parts": [shapes]},
        "cu192_cot": {"context": cot, "parts": [shapes]},
        "cu128_cot": {"context": dict(cot, cu_budget=128), "parts": [shapes]},
    }
    assert a.bench_units(tmp_path, units, line_of, "test") == 0
    assert [line_solution_list(x) for x in benched["exclusive"]] == [[7, 9], [9], None]
    for label in ("cu192_cot", "cu128_cot"):
        assert [line_solution_list(x) for x in benched[label]] == [[7], [3, 7]]
    assert splits == [(a.p.cfg, tmp_path / "lib")]


def _stub_audit(tmp_path, monkeypatch, clustered, plain):
    """An AuditRun of `tmp_path` whose pool splits into `clustered` and
    `plain` for the co-tenant contexts (lib/contexts.py cotenant_kernels),
    without a blocklist or guards, its benches, wrappers and progress
    stubbed: (the run, {unit: its bench lines}, the library directories the
    split was read from)."""
    import types

    a = ro.AuditRun.__new__(ro.AuditRun)
    a.dir = tmp_path
    a._blocked, a._guarded = set(), {}
    a.shapes_this_run = 0
    a.p = types.SimpleNamespace(
        cfg={"arch": "gfx1250"},
        paths=types.SimpleNamespace(library_dir=tmp_path / "lib"),
        run_stage=lambda *args: 0,
    )
    splits = []

    def split(cfg, library_dir):
        splits.append(library_dir)
        return set(clustered), list(plain)

    monkeypatch.setattr(ro.ctxlib, "cotenant_kernels", split)
    benched = {}
    monkeypatch.setattr(a, "wrapper", lambda label, unit: label, raising=False)
    monkeypatch.setattr(
        a,
        "bench_part",
        lambda part, lines, w: benched.setdefault(w, []).extend(lines) or 0,
        raising=False,
    )
    monkeypatch.setattr(a, "write_progress", lambda *x: None, raising=False)
    return a, benched, splits


def _cotenant_units(shapes):
    return {
        "exclusive": {"context": None, "parts": [shapes]},
        "allcu_ws0m": {
            "context": {"cu_budget": 0, "workspace_bytes": 0},
            "parts": [shapes],
        },
        "cu192_cot": {
            "context": {"cu_budget": 192, "cotenant": True},
            "parts": [shapes],
        },
    }


def _write_part(root, label, shapes, listed):
    part = ro.AuditRun.part_dir(root, label, 0)
    part.mkdir(parents=True, exist_ok=True)
    (part / "shapes.yaml").write_text(
        "".join(
            f"- {{M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}, "
            f"solution_index_list: [{', '.join(map(str, listed))}]}}\n"
            for s in shapes
        )
    )


def test_a_generation_counts_the_clustered_kernels_its_cotenant_units_leave_out(
    tmp_path, monkeypatch
):
    """A generation records a kernel when every line of every unit lists it,
    but a co-tenant unit need not list a clustered kernel, which no
    co-tenant context benches. A kernel a co-tenant line leaves out
    otherwise still counts as not benched."""
    a, _, splits = _stub_audit(tmp_path, monkeypatch, {9, 13}, [7, 11])
    shapes = [(64, 64, 64, 1), (128, 64, 64, 1)]
    units = _cotenant_units(shapes)
    for label in ("exclusive", "allcu_ws0m"):
        _write_part(tmp_path, label, shapes, [7, 9, 11])
    _write_part(tmp_path, "cu192_cot", shapes, [7, 11])
    planned = {"7": "d7", "9": "d9", "11": "d11", "13": "d13"}
    assert a.listed_everywhere(tmp_path, units, planned) == {
        "7": "d7",
        "9": "d9",
        "11": "d11",
    }
    assert splits == [tmp_path / "lib"]
    _write_part(tmp_path, "cu192_cot", shapes, [7])
    assert a.listed_everywhere(tmp_path, units, planned) == {"7": "d7", "9": "d9"}


def test_a_cotenant_part_left_with_no_line_stops_the_audit(
    tmp_path, monkeypatch, capsys
):
    """A co-tenant part whose lines list only clustered or blocked kernels is
    not benched as an empty part: the audit stops and says why."""
    a, benched, _ = _stub_audit(tmp_path, monkeypatch, {9}, [3])
    a._blocked = {3}
    shapes = [(64, 64, 64, 1), (128, 64, 64, 1)]
    units = {"cu192_cot": _cotenant_units(shapes)["cu192_cot"]}

    def line_of(label, i, s):
        listed = "[9]" if s[0] == 64 else "[3, 9]"
        return (
            f"- {{M: {s[0]}, N: {s[1]}, K: {s[2]}, batch_count: {s[3]}, "
            f"solution_index_list: {listed}}}\n"
        )

    assert a.bench_units(tmp_path, units, line_of, "increment 1") == 1
    assert benched == {}
    assert not a.part_dir(tmp_path, "cu192_cot", 0).exists()
    err = capsys.readouterr().err
    assert "no line of cu192_cot lists a kernel a co-tenant context benches" in err


def test_an_audit_increment_of_clustered_kernels_leaves_out_the_cotenant_units(
    tmp_path, monkeypatch
):
    """With the co-tenant units listing no clustered kernel, generation 0
    still records every kernel it benched, so the audit plans no increment.
    A kernel the pool gains, clustered, gets an increment that benches it in
    the units without a co-tenant only, records it and leaves nothing
    pending."""
    from lib.bench_yaml import line_solution_list

    a, benched, _ = _stub_audit(tmp_path, monkeypatch, {9, 12}, [1, 2])
    kernels = {1: "d1", 2: "d2", 9: "d9"}
    for name in ("current_kernels", "required_kernels"):
        monkeypatch.setattr(a, name, lambda: dict(kernels), raising=False)
    monkeypatch.setattr(a, "current_identity", lambda: None, raising=False)
    staged = []
    monkeypatch.setattr(
        a,
        "args_stage04",
        lambda root, units: staged.append(sorted(units)) or [],
        raising=False,
    )
    shapes = [(64, 64, 64, 1), (128, 64, 64, 1)]
    units = _cotenant_units(shapes)
    for label in ("exclusive", "allcu_ws0m"):
        _write_part(tmp_path, label, shapes, [1, 2, 9])
    _write_part(tmp_path, "cu192_cot", shapes, [1, 2])
    planned = {"1": "d1", "2": "d2", "9": "d9"}
    m = {
        "generations": [
            {"index": 0, "kernels": a.listed_everywhere(tmp_path, units, planned)}
        ]
    }
    assert m["generations"][0]["kernels"] == planned
    assert a.extend(m, units) == 0
    assert not (tmp_path / "increments").exists() and benched == {}

    kernels[12] = "d12"
    assert a.extend(m, units) == 0
    assert sorted(benched) == ["allcu_ws0m", "exclusive"]
    assert all(
        line_solution_list(ln) == [12] for lines in benched.values() for ln in lines
    )
    assert staged == [["allcu_ws0m", "exclusive"]]
    assert [g["kernels"] for g in m["generations"]] == [planned, {"12": "d12"}]
    assert m["status"] == "complete"


_LISTED = (
    "- {function: matmul, M: 256, N: 256, K: 8192, batch_count: 1, "
    "solution_index_list: [7, 9], uniform_summation_order: 1}\n"
)


def _typed(k, a_type="bf16_r", alpha=1, beta=0, listed="solution_index_list: [7, 9], "):
    return (
        f"- {{function: matmul, M: 256, N: 256, K: {k}, batch_count: 1, "
        f"a_type: {a_type}, b_type: {a_type}, alpha: {alpha}, beta: {beta}, "
        f"{listed}uniform_summation_order: 1}}\n"
    )


def test_an_integer_exact_canary_takes_only_lines_the_bench_checks_exactly():
    """The bench compares an integer-exact 16-bit line with its host
    reference up to K 512 (256 with alpha 2 and beta 0 or -2), never on
    gfx11; other A types at any K. A line without a_type has the bench's
    default A, f16."""
    checkable = cx.integer_exact_checkable
    assert checkable(_typed(512), "gfx950") and not checkable(_typed(513), "gfx950")
    assert checkable(_typed(512, "f16_r"), "gfx950")
    assert checkable(_typed(256, alpha=2), "gfx950")
    assert not checkable(_typed(257, alpha=2), "gfx950")
    assert not checkable(_typed(257, alpha=2, beta=-2), "gfx950")
    assert checkable(_typed(257, alpha=2, beta=1), "gfx950")
    assert checkable(_typed(1 << 16, "f32_r"), "gfx950")
    assert not checkable(_typed(64), "gfx1151")
    assert checkable(_typed(64, "f32_r"), "gfx1151")
    untyped = "- {function: matmul, M: 64, N: 64, K: 1024, batch_count: 1}\n"
    assert not checkable(untyped, "gfx950")
    assert not checkable("- {function: matmul, a_type: f32_r}\n", "gfx950")


def test_the_canaries_of_a_workspace_context_sample_its_lines_and_kernels():
    """An integer-exact canary runs one kernel of a line the bench checks
    exactly: one the line lists, else one of the pool's; the verification
    lines are the context's own; the draws follow the seed."""
    from lib.bench_yaml import line_solution_list

    settings = dict(cx.CANARY_DEFAULTS, fast_check_samples=3, verify_lines=1)
    deep = _typed(8192)
    lines = [_typed(512), deep, _typed(128, listed="")]
    exact, verify = cx.canary_lines(
        lines, [100, 101], settings, 5, "allcu_ws0m", "gfx950"
    )
    assert (exact, verify) == cx.canary_lines(
        lines, [100, 101], settings, 5, "allcu_ws0m", "gfx950"
    )
    assert len(exact) == 3 and len(verify) == 1 and verify[0] in lines
    for line in exact:
        assert "initialization: integer_exact" in line and "fast_check" not in line
        assert "skip_slow_solution_ratio: 0" in line and "K: 8192" not in line
        (kernel,) = line_solution_list(line)
        assert kernel in ((7, 9) if "K: 512" in line else (100, 101))
    pairs = cx.checkable_pairs(lines, [100, 101], "gfx950")
    assert pairs == [(0, 7), (0, 9), (2, 100), (2, 101)]
    assert cx.canary_lines([deep], [100], settings, 5, "x", "gfx950") == ([], [deep])
    none = dict(settings, fast_check_samples=0, verify_lines=0)
    assert cx.canary_lines(lines, [100], none, 5, "x", "gfx950") == ([], [])
    assert cx.canary_lines([], [100], settings, 5, "x", "gfx950") == ([], [])


def _bench_output(*rows, compared=True):
    """hipblaslt-bench's output of one problem per (M, N, K, solution) of
    `rows`, each result printed with the comparison of its D with the host
    reference (`norm_error`) or, without `compared`, without it."""
    head = "transA,transB,batch_count,m,n,k,us" + (
        ",CPU-us,norm_error" if compared else ""
    )
    out = ["hipBLASLt version: 100000"]
    for m, n, k, sol in rows:
        values = f"T,N,1,{m},{n},{k},2.5" + (",9.5,0" if compared else "")
        out += [
            "Is supported 1 / Total solutions: 1",
            f"[0]:{head}",
            f"    {values}",
            f"    --Solution index: {sol}",
        ]
    return "\n".join(out) + "\n"


def test_the_verdict_of_a_context_canary_counts_the_checks_performed():
    exact = [_typed(512, listed="solution_index_list: [100], ")]
    verify = [_typed(4096)]
    log = _bench_output(
        (256, 256, 512, 100), (256, 256, 4096, 7), (256, 256, 4096, 9), (64, 64, 64, 5)
    )
    verdict = cx.canary_verdict([log], exact, verify)
    assert (verdict["exact_checks"], verdict["verify_checks"]) == (1, 2)
    assert (verdict["checks"], verdict["status"]) == (3, "verified")
    assert cx.canary_refusal(verdict) is None
    other_kernel = _bench_output((256, 256, 512, 101))
    assert cx.canary_verdict([other_kernel], exact, verify)["checks"] == 0
    uncompared = _bench_output((256, 256, 512, 100), compared=False)
    assert cx.canary_verdict([uncompared], exact, verify)["status"] == "not verified"
    skipped = (
        "Skipping integer_exact: 16-bit format with K=970 > 512 "
        "(exact representability limit)\n"
    )
    verdict = cx.canary_verdict([skipped], exact, [])
    assert (verdict["unsupported"], verdict["checks"]) == (1, 0)
    record = dict(verdict, exact_lines=1, checkable_pairs=1, verify_lines=0)
    assert verdict["status"] == "not verified"
    assert "performed no check" in cx.canary_refusal(record)
    assert cx.canary_verdict([log], exact, verify, counted=[])["checks"] == 0


def test_a_failed_canary_check_is_a_failure_whatever_else_was_checked():
    ok = _bench_output((256, 256, 4096, 7))
    failed = (
        "post_bench_verify_iters failed for solution 7: 1 of 3 relaunches gave a "
        "different D than the first launch\n"
        "hipblaslt-bench: 1 --verify check(s) failed; exiting with code 2 "
        "(--verify_strict)\n"
    )
    unsupported = "fast_check does not support timing runs\n"
    verdict = cx.canary_verdict([ok, failed, unsupported], [], [_typed(4096)])
    assert (verdict["failed"], verdict["unsupported"], verdict["checks"]) == (2, 1, 1)
    assert verdict["examples"][0].startswith("post_bench_verify_iters failed for")
    assert verdict["status"] == "failed"
    assert "canary check(s) failed" in cx.canary_refusal(verdict)
    assert cx.canary_verdict([]) == {
        "failed": 0,
        "examples": [],
        "unsupported": 0,
        "unsupported_examples": [],
        "exact_checks": 0,
        "verify_checks": 0,
        "checks": 0,
        "status": "not verified",
    }


def test_a_wrapper_passes_the_benches_its_own_arguments(tmp_path):
    spec = cx.bench_spec(cx.BenchContext(workspace_bytes=0), 64, "bench", None, "l")
    spec["bench_args"] = ["--post_bench_verify_iters", "3", "--verify_strict"]
    cmd, _ = cx.bench_command(spec, ["--device", "0", "--yaml", "y"], "s", {})
    assert cmd[-3:] == ["--post_bench_verify_iters", "3", "--verify_strict"]
    assert cmd[:5] == ["bench", "--device", "0", "--yaml", "y"]


_F32_LINES = {
    s: line.replace("batch_count: 1,", "batch_count: 1, a_type: f32_r,")
    for s, line in _LINES.items()
}


def _stage03c_canaries(tmp_path, log_text, cfg):
    """stage03c's canaries of the launch-keyed W=0 context of `_F32_LINES`,
    benched by a stage03 that writes `log_text` as its one block log."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = _context_bench_args(tmp_path)
    ws = cx.BenchContext(workspace_bytes=0, launch_keyed=True)
    launches = {
        "kernels": [{"name": "b", "sol_idx_global": 100}],
        "launch": _splitting_launch([]),
        "mi_dtype": "",
    }
    rec = s03c._bench_context(
        ws, args, {}, _NO_OP, _F32_LINES, _SHAPES, 1, 64, tmp_path, None, launches
    )
    src = tmp_path / "bench_output.log"
    src.write_text(log_text)
    write_log = (
        "import pathlib, shutil, sys; d = pathlib.Path(sys.argv[1]) / 'stage03' / "
        f"'logs'; shutil.copy({str(src)!r}, d / 'block_0000_lines1-2_gpu0.log')"
    )
    return s03c._canary_context(
        ws,
        args,
        cfg,
        {"stage03": ["-c", write_log, "{ctx_dir}"]},
        rec,
        64,
        tmp_path,
        None,
        launches,
    )


def test_stage03c_reads_the_canaries_of_a_workspace_context(tmp_path, rule_v3):
    cfg = {"contexts": {"canaries": {"enabled": True, "verify_iters": 4}}}
    failed = (
        "post_bench_verify_iters failed for solution 100: 1 of 3 relaunches gave "
        "a different D\n"
    )
    out = _stage03c_canaries(tmp_path / "a", failed, cfg)
    assert (out["failed"], out["rc"], out["verify_iters"]) == (1, 0, 4)
    assert (out["exact_lines"], out["checkable_pairs"], out["verify_lines"]) == (
        1,
        1,
        1,
    )
    assert out["status"] == "failed" and s03c.canary_refusal(out)
    cdir = Path(out["dir"])
    spec = json.loads((cdir / cx.SPEC_FILE).read_text())
    assert spec["bench_args"] == ["--post_bench_verify_iters", "4", "--verify_strict"]
    exact, verify = (cdir / "shapes.yaml").read_text().splitlines()
    assert "initialization: integer_exact" in exact and "fast_check" not in exact
    assert "solution_index_list: [100]" in exact
    assert "user_allocated_workspace: 0" in verify and "integer_exact" not in verify
    checked = _bench_output((256, 256, 8192, 100), (256, 256, 8192, 100))
    out = _stage03c_canaries(tmp_path / "b", checked, cfg)
    assert (out["exact_checks"], out["verify_checks"], out["status"]) == (
        1,
        1,
        "verified",
    )
    assert s03c.canary_refusal(out) is None
    out = _stage03c_canaries(tmp_path / "c", "", cfg)
    assert (out["rc"], out["checks"], out["status"]) == (0, 0, "not verified")
    assert "not verified" in s03c.canary_refusal(out)


def test_a_context_whose_canaries_draw_no_line_is_not_verified(tmp_path, rule_v3):
    """With no line to check, nothing runs, and the context is not verified."""
    off = {"fast_check_samples": 0, "verify_lines": 0}
    cfg = {"contexts": {"canaries": dict(enabled=True, **off)}}
    out = _stage03c_canaries(tmp_path, "", cfg)
    assert (out["rc"], out["exact_lines"], out["verify_lines"]) == (None, 0, 0)
    assert out["status"] == "not verified" and s03c.canary_refusal(out)
    assert not (Path(out["dir"]) / "shapes.yaml").exists()


def test_stage03c_needs_the_library_of_a_launch_keyed_context(tmp_path):
    args = argparse.Namespace(launch_library_dir=None, library_stem=None)
    with pytest.raises(ValueError, match="needs --launch-library-dir"):
        s03c.launch_pass(args, {"hipblaslt": {}}, None)


def test_launch_keyed_contexts_reach_stage03c_but_not_the_planner(tmp_path, rule_v3):
    cfg = _context_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.model_file": str(tmp_path / "m.pt"),
            "contexts.replan": True,
            "contexts.workspaces": [0],
        },
    )
    p = _pipeline(tmp_path, cfg)
    assert [c.label for c in p.contexts if c.launch_keyed] == ["allcu_ws0m"]
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    wanted = json.loads(arg({"argv": p.args_stage02(rd, step)}, "--contexts"))
    assert "allcu_ws0m" not in [c["name"] for c in wanted]
    args = p.args_stage03c(rd)
    assert arg({"argv": args}, "--launch-library-dir") == str(p.paths.library_dir)
    assert arg({"argv": args}, "--library-stem") == p.stem
    s04 = p.context_log_args(rd)
    logs = str(p.context_dir(rd, p.contexts[-2]) / "stage03" / "logs")
    i = s04.index(logs)
    assert s04[i - 3 : i] == ["--context-log-dir", "0", "default"]
