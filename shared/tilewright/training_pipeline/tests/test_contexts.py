# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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


# ── config ───────────────────────────────────────────────────────────────────


def test_contexts_from_config_skips_the_exclusive_entry():
    ctxs = cx.contexts_from_config(_cfg(holdout=[{"cu_budget": 96}]))
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
    return [c.cu_budget for c in cx.contexts_from_config(cfg, n_cu)]


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
    ctxs = cx.contexts_from_config(out)
    assert [c.label for c in ctxs] == [
        "cu192_cot",
        "cu128_cot",
        "cu64_cot",
        "cu224_cot",
        "cu96_cot",
    ]
    assert cx.contexts_from_config(cfg, 256) == ctxs
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
    assert [c.cu_budget for c in cx.contexts_from_config(tiny)] == [32]
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
    assert [c.cu_budget for c in cx.contexts_from_config(cfg, 256)] == [192, 128, 64]
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
    assert [c.label for c in cx.contexts_from_config(explicit)] == ["cu64"]
    budget = {"cu_budget": 32, "schedule": "dynamic", "enabled": False}
    assert cx.describe(budget) == "cu_budget 32 / schedule dynamic"


def test_turned_on_the_eighth_resolves_on_devices_of_eight_granules_up():
    on = {"contexts": {"list": [dict(m, enabled=True) for m in SHARES["list"]]}}
    budgets = [c.cu_budget for c in cx.contexts_from_config(on, 256)]
    assert budgets == [192, 128, 64, 32]
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
    assert cx.context_errors(cfg, 32) == [] and cx.contexts_from_config(cfg, 32) == []
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
    _write_logic(src / f"{pool}.dat", 3)
    (src / f"{pool}.co").write_bytes(b"code")
    (src / "old.tilewright.bin").write_bytes(b"shipped model")
    (src / "h.tilewright.bin").write_bytes(b"other model")
    (src / "tilewright_index").write_text(
        f"# shipped\n{pool}\told.tilewright.bin\n{other} h.tilewright.bin # kept\n"
        f"TensileLibrary_X_gfx950\tnew.tilewright.bin\n"
    )
    model = tmp_path / "staged.bin"
    model.write_bytes(b"MLREC_v3 staged")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    dst = tmp_path / "validate" / "stage07_lib" / "gfx950"
    summary = pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model, sha)
    assert summary["other_stems"] == [other]
    assert summary["left_out_stems"] == ["TensileLibrary_X_gfx950"]
    assert (dst / f"{pool}.dat").is_symlink() and (dst / f"{pool}.co").is_symlink()
    assert (dst / "old.tilewright.bin").is_symlink()
    assert not (dst / "new.tilewright.bin").is_symlink()
    assert (dst / "new.tilewright.bin").read_bytes() == b"MLREC_v3 staged"
    assert pl.index_entries((dst / "tilewright_index").read_text()) == [
        (other, "h.tilewright.bin"),
        (pool, "new.tilewright.bin"),
    ]
    assert (src / "tilewright_index").read_text().startswith("# shipped")
    # Rows for the devices of one PCI chip ID: the stem's give way to the one
    # row that serves the staged model on every device; other stems' stay,
    # and so does another stem's malformed row, for which the engine loads
    # no model.
    (src / "tilewright_index").write_text(
        f"{pool}\tchip.tilewright.bin\tchip=0x1a2b\n{pool}\told.tilewright.bin\n"
        f"{other}\th1.tilewright.bin\tchip=0x00ff\n{other}\th.tilewright.bin\n"
        f"{other} bad.bin chip=0xzz # typo\n{pool} worse.bin chip=0x12 more\n"
    )
    summary = pl.build_serving_libdir(src, dst, pool, "new.tilewright.bin", model, sha)
    assert summary["other_stems"] == [other] and summary["left_out_stems"] == []
    assert summary["malformed_rows"] == [f"{other} bad.bin chip=0xzz # typo"]
    assert (dst / "tilewright_index").read_text() == (
        f"{other}\th.tilewright.bin\n{other}\th1.tilewright.bin\tchip=0x00ff\n"
        f"{other} bad.bin chip=0xzz # typo\n{pool}\tnew.tilewright.bin\n"
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
    with pytest.raises(ValueError, match=f"malformed row for {other}"):
        tw.load_model_by_index(other, str(dst))
    with pytest.raises(ValueError, match="new.tilewright.bin"):
        tw.load_model_by_index(pool, str(dst), 0x12)


def test_serving_libdir_serves_the_staged_sidecars_in_place_of_the_stems(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pool = "TensileLibrary_BB_BB_HA_Bias_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    _write_logic(src / f"{pool}.dat", 3)
    (src / f"{pool}.tilewright.blocklist").write_text("shipped blocklist\n")
    (src / f"{pool}.tilewright.guards").write_text("shipped guards\n")
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
        f = dst / f"{pool}.tilewright.{name}"
        assert not f.is_symlink() and f.read_text() == text
    assert summary["guards"] == str(guards) and summary["guards_sha256"]
    # Without staged guards the stem has none, whatever the source holds.
    summary = pl.build_serving_libdir(
        src, dst, pool, "new.tilewright.bin", model, blocklist=side
    )
    assert not (dst / f"{pool}.tilewright.guards").exists()
    assert summary["guards"] is None and summary["guards_sha256"] is None


# ── orchestrator wiring ──────────────────────────────────────────────────────


def _context_config(tmp_path, **extra):
    return make_config(
        tmp_path,
        **{
            "train.catalog": "v3",
            "contexts.list": list(CTX_LIST),
            "contexts.holdout": [{"cu_budget": 96, "cotenant": True}],
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
    assert run(stubs, make_config(tmp_path), tmp_path / "run", "--mode", "initial") == 0
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
                "hardware": {"n_cu": 64, "lds_bytes": 65536, "l2_bytes": 4194304},
                "contexts": {
                    "list": [
                        {"cu_budget": 0},
                        {"cu_budget": 16, "cotenant": True},
                        {"cu_budget": 32, "fraction": 0.5},
                    ],
                    "holdout": [{"cu_budget": 8, "cotenant": True}],
                    "seed": 3,
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
