# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the pure-Python parts of the skinny GEMM sweep: no GPU, no bindings."""

import argparse
import collections
import concurrent.futures
import contextlib
import functools
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import threading
import time
import types
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pytest
import yaml

import analyze_skinny_gemm_sweep as analyze
import bench_skinny_gemm as bench
import select_skinny_gemm_from_sweep as from_sweep
import select_skinny_gemm_kernels as kernels_tool
import skinny_gemm_selection as selection
import skinny_gemm_support as support
import sweep_skinny_gemm as sweep
import sweep_skinny_gemm_results as results_module
import sweep_skinny_gemm_config as config
import sweep_skinny_gemm_inputs as inputs_module
import sweep_skinny_gemm_smi as smi
import sweep_skinny_gemm_worker as worker

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(text: str) -> config.SweepInput:
    return config.parse_input(yaml.safe_load(text))


def test_range_expansion():
    assert config.expand_range("r", {"min": 1, "max": 8, "step": 3}) == [1, 4, 7]
    assert config.expand_range("r", {"min": 1, "max": 16, "mul": 2}) == [1, 2, 4, 8, 16]
    with pytest.raises(config.SweepConfigError, match="exactly one"):
        config.expand_range("r", {"min": 1, "max": 8})
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        config.expand_range("r", {"min": 1, "max": 8, "stp": 1})


def test_unknown_keys_and_types_are_errors():
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweeep: {}}]")
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {bn: 1}}]")
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("settings: {iter: 3}\nproblems: [{name: a, m: 1, n: 2, k: 3}]")
    with pytest.raises(config.SweepConfigError, match="expected true or false"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {grid3d: 1}}]")
    with pytest.raises(config.SweepConfigError, match="expected an integer"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {bm: true}}]")


def test_shapes_transpose_and_spk():
    text = (
        "problems: [{name: a, m: [1, 2], n: 8, k: 512, transpose: true, "
        "spk: [k, 256], useBeta: false}]"
    )
    cases = parse(text).cases
    assert {(c.m, c.n) for c in cases} == {(1, 8), (2, 8), (8, 1), (8, 2)}
    # "k" is no split-K, and the case prints it as such.
    assert {c.spk for c in cases if (c.m, c.n) == (1, 8)} == {256, None}
    assert {"spk=k" in c.label for c in cases if c.spk is None} == {True}
    assert {"spk=256" in c.label for c in cases if c.spk == 256} == {True}


def test_ld_defaults_and_validation():
    case = parse("problems: [{name: a, m: 5, n: 8, k: 4100}]").cases[0]
    assert (case.ld_a, case.ld_d, case.ld_c) == (4100, 10, 10)
    with pytest.raises(config.SweepConfigError, match="multiple of 4"):
        parse("problems: [{name: a, m: 5, n: 8, k: 64, ldA: 66}]")


def test_sweep_list_merge_and_dedupe():
    text = """
defaults:
  sweep:
    - {bm: [1, 2], grid3d: true}
    - {bm: 1, grid3d: false, group: [1, 2], numXcds: 1, blocks: [4, 100000]}
problems:
  - {name: a, m: 4, n: 4096, k: 1024, sweep: {tn: [1, 2]}}
"""
    case = parse(text).cases[0]
    assert len(case.sweeps) == 2
    expanded = config.expand_case(case)
    # 2 bm * 2 tn on the 3-D grid; persistent: 2 groups * 2 tn * 2 capped blocks.
    assert len(expanded.problems) == 4 + 8
    persistent = [p for p in expanded.problems if not p.grid3d]
    tiles = {p.tn: support.countTiles(4, 4096, 1024, 1024, 1, p.bn) for p in persistent}
    assert all(p.blocks <= tiles[p.tn] for p in persistent)


def test_grid3d_forces_plain_order_and_duplicates_collapse():
    text = """
problems:
  - {name: a, m: 4, n: 4096, k: 1024,
     sweep: {lanesK: 64, group: [1, 2, 4], numXcds: [1, 8], grid3d: true}}
"""
    expanded = config.expand_case(parse(text).cases[0])
    assert len(expanded.problems) == 1
    assert expanded.problems[0].group == expanded.problems[0].xcds == 1


def test_constraint_failures_are_recorded():
    text = "problems: [{name: a, m: 4, n: 4096, k: 1024, sweep: {lanesK: [32, 64]}}]"
    expanded = config.expand_case(parse(text).cases[0])
    assert [f.stage for f in expanded.failures] == ["constraint"]
    assert expanded.swept == 2


def test_derived_knobs_in_record():
    text = (
        "problems: [{name: a, m: 1, n: 4096, k: 4100, "
        "sweep: {lanesK: 16, upScales: 1}}]"
    )
    record = config.config_record(config.expand_case(parse(text).cases[0]).problems[0])
    assert record["upIters"] == 5


def test_default_inputs_parse():
    assert config.load_input(os.path.join(HERE, "sweep_skinny_gemm_shapes.yaml")).cases
    assert config.load_input(os.path.join(HERE, "sweep_skinny_gemm_smoke.yaml")).cases


def test_parse_accepts_zero_refine_settings():
    text = (
        "settings: {refine_top: 0, refine_iters: 0, top: 0}\n"
        "problems: [{name: a, m: 1, n: 512, k: 1024}]"
    )
    assert parse(text).cases


@pytest.mark.parametrize(
    "setting", ["iters", "verify_full_max_elems", "warmup", "seed", "top"]
)
def test_parse_rejects_negative_settings(setting):
    text = f"settings: {{{setting}: -1}}\nproblems: [{SHAPES}]"
    with pytest.raises(config.SweepConfigError, match=f"settings.{setting}"):
        parse(text)


def test_parse_requires_positive_iters():
    with pytest.raises(config.SweepConfigError, match="settings.iters: must be at"):
        parse(f"settings: {{iters: 0}}\nproblems: [{SHAPES}]")


def settings_with(**overrides) -> dict:
    return {**config.SETTING_DEFAULTS, **overrides}


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"iters": 0}, r"settings.iters: must be at least 1, got 0"),
        ({"verify_full_max_elems": 0}, "settings.verify_full_max_elems"),
        ({"warmup": -1}, "settings.warmup"),
        ({"seed": -1}, "settings.seed"),
        ({"refine_iters": 0}, "settings.refine_iters: must be at least 1"),
        ({"top": 0}, "settings.top: must be at least 1"),
        ({"refine_top": 0}, "settings.refine_top: must be at least 1"),
        ({"refine_top": 2, "top": 5}, "settings.refine_top: must be at least top"),
        (
            {"refine_top": 0, "refine_iters": 0, "top": 0},
            "--one-shot",
        ),
    ],
)
def test_check_settings_rejects(overrides, match):
    with pytest.raises(config.SweepConfigError, match=match):
        config.check_settings(settings_with(**overrides), one_shot=False)


def test_check_settings_one_shot_ignores_refine_settings():
    zero = settings_with(refine_top=0, refine_iters=0, top=0)
    config.check_settings(zero, one_shot=True)
    config.check_settings(settings_with(), one_shot=False)
    with pytest.raises(config.SweepConfigError, match="settings.iters"):
        config.check_settings(settings_with(iters=0), one_shot=True)


def resolve(*args: str) -> dict:
    options = sweep.make_parser().parse_args(list(args))
    return sweep.resolve_settings(dict(config.SETTING_DEFAULTS), options)


@pytest.mark.parametrize(
    "args, match",
    [
        (["--iters", "0"], "settings.iters"),
        (["--refine-iters", "0"], "settings.refine_iters"),
        (["--top", "0"], "settings.top"),
        (["--refine-top", "1"], "settings.refine_top: must be at least top"),
        (["--build-jobs", "0"], "--build-jobs: must be at least 1, got 0"),
        (["--max-busy", "-1"], "--max-busy"),
        (["--max-busy", "101"], "--max-busy"),
        (["--max-busy", "nan"], "--max-busy"),
        (["--max-busy", "inf"], "--max-busy"),
    ],
)
def test_resolve_settings_rejects_overrides(args, match):
    with pytest.raises(config.SweepConfigError, match=match):
        resolve(*args)


def test_resolve_settings_accepts_valid_overrides():
    assert resolve("--one-shot", "--top", "0")["top"] == 0
    assert resolve("--top", "2", "--build-jobs", "1")["top"] == 2


def test_list_runs_without_arguments(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["sweep_skinny_gemm.py", "--list"])
    assert sweep.main() == 0
    assert "total: shapes=" in capsys.readouterr().out


def test_list_on_smoke_input(monkeypatch, capsys):
    smoke = os.path.join(HERE, "sweep_skinny_gemm_smoke.yaml")
    monkeypatch.setattr("sys.argv", ["sweep_skinny_gemm.py", smoke, "--list"])
    assert sweep.main() == 0
    assert "total: shapes=" in capsys.readouterr().out


def test_default_input_needs_one_shot_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv", ["sweep_skinny_gemm.py", "--dry-run", "--work-dir", str(tmp_path)]
    )
    assert sweep.main() == 2
    assert not os.listdir(tmp_path)


MFMA_SWEEP = {
    "mfma": True,
    "bm": 16,
    "tn": 2,
    "lanesK": 64,
    "waves": 1,
    "prefetch": 2,
    "upScales": 0,
}


def mfma_case(k: int = 1024, spk="k", **sweep) -> config.Case:
    """A case of a 4 x 4096 problem whose sweep is MFMA_SWEEP with the changes."""
    knobs = {**MFMA_SWEEP, **sweep}
    entry = {
        "name": "a",
        "m": 4,
        "n": 4096,
        "k": k,
        "spk": spk,
        "useBeta": False,
        "sweep": knobs,
    }
    return parse(yaml.safe_dump({"problems": [entry]})).cases[0]


def test_default_sweep_is_not_mfma():
    assert config.DEFAULT_SWEEP["mfma"] == [False]
    text = "problems: [{name: a, m: 4, n: 4096, k: 1024, useBeta: false}]"
    expanded = config.expand_case(parse(text).cases[0])
    assert expanded.problems
    assert all(not config.config_record(p)["mfma"] for p in expanded.problems)
    assert not any("MFMA" in support.gemmKernelName(p) for p in expanded.problems)


def test_mfma_sweep_expands_to_the_valid_product():
    case = mfma_case(bm=[16, 32], tn=[1, 2], waves=[1, 2], prefetch=[1, 2])
    expanded = config.expand_case(case)
    assert expanded.swept == 16 and not expanded.failures
    assert len(expanded.problems) == 16
    for p in expanded.problems:
        assert p.mfma and support.gemmKernelName(p).endswith("_MFMA1")
        assert p.bn == 16 * p.tn and p.ki == 256
        record = config.config_record(p)
        assert record["mfma"] is True
        assert record["upIters"] == 0


@pytest.mark.parametrize(
    "KNOBS, K, REASON",
    [
        ({"lanesK": 16}, 1024, "lanes_k=64"),
        ({"upScales": 1}, 1024, "up_scales=0"),
        ({"prefetch": 0}, 1024, "pf >= 1"),
    ],
)
def test_mfma_invalid_combinations_are_constraint_failures(KNOBS, K, REASON):
    expanded = config.expand_case(mfma_case(k=K, **KNOBS))
    assert not expanded.problems and expanded.swept == 1
    assert [f.stage for f in expanded.failures] == ["constraint"]
    assert REASON in expanded.failures[0].reason
    assert expanded.failures[0].config["mfma"] is True


@pytest.mark.parametrize("BM", [[1], [1, 2, 4, 8], [8, 16], [32, 1], [16, 32]])
def test_mfma_bm_is_the_16_and_32_of_the_sweep_else_both(BM):
    expanded = config.expand_case(mfma_case(bm=BM))
    expected = sorted({16, 32} & set(BM)) or [16, 32]
    assert not expanded.failures
    assert sorted(p.bm for p in expanded.problems) == expected


def test_mfma_default_bm_sweeps_both_block_sizes():
    text = (
        "problems: [{name: a, m: 4, n: 4096, k: 1024, useBeta: false, "
        "sweep: {mfma: true}}]"
    )
    expanded = config.expand_case(parse(text).cases[0])
    assert not expanded.failures
    assert sorted(p.bm for p in expanded.problems) == [16, 32]


def test_mixed_sweep_keeps_the_valu_bm_list_for_valu_only():
    case = mfma_case(mfma=[False, True], bm=[1, 2, 16], lanesK=64, tn=1, waves=1)
    problems = config.expand_case(case).problems
    assert sorted(p.bm for p in problems if p.mfma) == [16]
    assert sorted(p.bm for p in problems if not p.mfma) == [1, 2, 16]


def test_mixed_valu_and_mfma_sweep_dedupes_by_kernel_name():
    case = mfma_case(mfma=[False, True, True], bm=16)
    expanded = config.expand_case(case)
    names = [support.gemmKernelName(p) for p in expanded.problems]
    assert len(set(names)) == len(names)
    mfma = [p for p in expanded.problems if p.mfma]
    assert len(mfma) == 1 and names[expanded.problems.index(mfma[0])].endswith("_MFMA1")
    assert expanded.swept == 2


def test_persistent_mfma_blocks_are_capped_with_the_mfma_block_width():
    case = mfma_case(
        k=1024, spk=512, grid3d=False, blocks=100000, tn=[1, 2], bm=[16, 32]
    )
    expanded = config.expand_case(case)
    assert len(expanded.problems) == 4
    for p in expanded.problems:
        tiles = (
            support.ceilDiv(4096, 16 * p.tn)
            * support.ceilDiv(4, p.bm)
            * support.ceilDiv(1024, 512)
        )
        assert p.mfma and not p.grid3d and p.blocks == tiles


def test_default_sweep_is_not_lds():
    assert config.DEFAULT_SWEEP["lds"] == [False]
    expanded = config.expand_case(mfma_case())
    assert [p.lds for p in expanded.problems] == [False]
    assert config.config_record(expanded.problems[0])["lds"] is False
    assert not support.gemmKernelName(expanded.problems[0]).endswith("_LDS1")


def test_lds_sweep_expands_with_the_lds_kernel_names():
    case = mfma_case(bm=[16, 32], tn=[1, 2], prefetch=[2, 3], lds=[False, True])
    expanded = config.expand_case(case)
    assert expanded.swept == 16 and not expanded.failures
    lds = [p for p in expanded.problems if p.lds]
    assert len(lds) == 8
    for p in lds:
        assert support.gemmKernelName(p).endswith("_MFMA1_LDS1")
        assert config.config_record(p)["lds"] is True
        assert p.bn == 16 * p.tn and p.ki == 256
    names = [support.gemmKernelName(p) for p in expanded.problems]
    assert len(set(names)) == 16


def test_lds_without_mfma_is_a_constraint_failure():
    case = mfma_case(mfma=False, bm=1, lds=True, lanesK=64, tn=1, waves=1)
    expanded = config.expand_case(case)
    assert not expanded.problems
    assert [f.stage for f in expanded.failures] == ["constraint"]
    assert "lds needs mfma" in expanded.failures[0].reason
    assert expanded.failures[0].config["lds"] is True


def test_mixed_mfma_lds_sweep_records_the_invalid_combination_only():
    case = mfma_case(mfma=[False, True], lds=[False, True], bm=16)
    expanded = config.expand_case(case)
    assert expanded.swept == 4 and len(expanded.failures) == 1
    assert len(expanded.problems) == 3
    assert not any(p.lds and not p.mfma for p in expanded.problems)


def test_lds_values_must_be_booleans_not_ranges():
    prefix = "problems: [{name: a, m: 1, n: 2, k: 3, sweep: "
    with pytest.raises(config.SweepConfigError, match="expected true or false"):
        parse(prefix + "{lds: [true, 1]}}]")
    with pytest.raises(config.SweepConfigError, match="lds takes booleans"):
        parse(prefix + "{lds: {min: 0, max: 1, step: 1}}}]")


def test_knob_sweep_input_with_mfma_and_lds_parses():
    text = (
        "problems: [{name: a, m: 5120, n: [8, 32], k: [8192, 8292], useBeta: false, "
        "sweep: {bm: [1, 4], tn: {min: 1, max: 4, step: 1}, waves: [1, 2, 4], "
        "prefetch: {min: 2, max: 8, step: 1}, mfma: true, lds: true, "
        "lanesK: 64, upScales: 0}}]"
    )
    cases = parse(text).cases
    assert len(cases) == 4
    expanded = config.expand_case(cases[0])
    assert expanded.swept == 2 * 4 * 3 * 7 and not expanded.failures
    assert all(p.mfma and p.lds for p in expanded.problems)


def test_mfma_values_must_be_booleans_not_ranges():
    prefix = "problems: [{name: a, m: 1, n: 2, k: 3, sweep: "
    with pytest.raises(config.SweepConfigError, match="expected true or false"):
        parse(prefix + "{mfma: [true, 1]}}]")
    with pytest.raises(config.SweepConfigError, match="mfma takes booleans"):
        parse(prefix + "{mfma: {min: 0, max: 1, step: 1}}}]")


class FakeAmdSmi:
    """A stand-in for the amdsmi module that logs the set requests and keeps a
    performance level; with fail every set request raises."""

    class AmdSmiClkType:
        GFX = "gfx"
        MEM = "mem"

    def __init__(self, fail: bool, level: str = "AUTO"):
        self.fail = fail
        self.level = level
        self.calls: list[tuple] = []
        self.AmdSmiDevPerfLevel = {"AUTO": "auto", "HIGH": "high", "MANUAL": "manual"}
        # Levels whose set request raises.
        self.fail_levels: set[str] = set()
        # The determinism mode is accepted but does not change the level.
        self.stuck_determinism = False
        # The determinism mode is applied, then the process is interrupted.
        self.interrupt = False
        self.unreadable = False
        self.activity: dict[str, list] = {}
        self.processes: dict[str, list] = {}
        self.metrics: dict = {}
        self.bdf: dict[str, str] = {}

    def amdsmi_get_gpu_perf_level(self, handle):
        if self.unreadable:
            raise RuntimeError("no perf level")
        return f"AMDSMI_DEV_PERF_LEVEL_{self.level}"

    def amdsmi_get_clock_info(self, handle, kind):
        return {"clk": 100, "max_clk": 2400}

    def amdsmi_get_gpu_activity(self, handle):
        return {"gfx_activity": self.activity.get(handle, [0]).pop(0)}

    def amdsmi_get_gpu_process_list(self, handle):
        return self.processes.get(handle, [])

    def amdsmi_get_gpu_metrics_info(self, handle):
        return self.metrics

    def amdsmi_get_gpu_device_bdf(self, handle):
        return self.bdf[handle]

    def amdsmi_set_gpu_perf_determinism_mode(self, handle, sclk):
        self.record("determinism", handle, sclk)
        if not self.stuck_determinism:
            self.level = "DETERMINISM"
        if self.interrupt:
            raise KeyboardInterrupt

    def amdsmi_set_gpu_perf_level(self, handle, level):
        self.record("level", handle, level)
        if level in self.fail_levels:
            raise PermissionError("level refused")
        self.level = level.upper()

    def record(self, *call):
        self.calls.append(call)
        if self.fail:
            raise PermissionError("root is required")


def install_fake_smi(monkeypatch, fail: bool, level: str = "AUTO") -> FakeAmdSmi:
    fake = FakeAmdSmi(fail, level)
    monkeypatch.setattr(
        smi,
        "_smi",
        types.SimpleNamespace(
            module=fake, pid=os.getpid(), handles=["h0", "h1"], handle=lambda g: f"h{g}"
        ),
    )
    # The existing tests lock in-process, as root does.
    monkeypatch.setattr(smi, "make_sudo_session", lambda: None)
    return fake


def test_stabilizer_restores_only_successful_locks(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    with smi.GpuStabilizer([1], require_stable=True) as stabilizer:
        assert stabilizer.summary()["status"] == "locked"
    assert fake.calls == [("determinism", "h1", 2400), ("level", "h1", "auto")]


def test_stabilizer_not_required_never_sets(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=True)
    with smi.GpuStabilizer([1], require_stable=False) as stabilizer:
        assert stabilizer.summary()["status"] == "unavailable"
    assert not fake.calls


def test_stabilizer_failed_lock_aborts(monkeypatch):
    install_fake_smi(monkeypatch, fail=True)
    with pytest.raises(smi.SweepAbort):
        with smi.GpuStabilizer([1], require_stable=True):
            pass


def test_stabilizer_registers_atexit_after_the_smi_initialization(monkeypatch):
    install_fake_smi(monkeypatch, fail=False)
    events = []
    get_smi = smi.get_smi
    monkeypatch.setattr(smi, "get_smi", lambda: events.append("init") or get_smi())
    monkeypatch.setattr(smi.atexit, "register", lambda f: events.append("register"))
    with smi.GpuStabilizer([1], require_stable=True):
        pass
    assert events[:2] == ["init", "register"]


def test_restore_in_another_process_does_nothing(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    stabilizer = smi.GpuStabilizer([1], require_stable=True)
    with stabilizer:
        stabilizer.pid = os.getpid() + 1
        calls = len(fake.calls)
        stabilizer.restore()
        assert len(fake.calls) == calls
        stabilizer.pid = os.getpid()
    assert fake.calls[-1] == ("level", "h1", "auto")


def test_unreadable_prior_level_is_never_locked(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    fake.unreadable = True
    stabilizer = smi.GpuStabilizer([1], require_stable=True)
    with pytest.raises(smi.SweepAbort, match=r"cannot lock the clocks of GPU\(s\) \[1\]"):
        with stabilizer:
            pass
    assert not fake.calls
    assert stabilizer.summary()["reason"] == "cannot read the prior performance level"


def test_interrupt_right_after_a_set_is_still_restored(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False, level="MANUAL")
    fake.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        with smi.GpuStabilizer([1], require_stable=True):
            pass
    assert fake.level == "MANUAL"
    assert fake.calls[-1] == ("level", "h1", "manual")


def test_read_back_mismatch_tries_the_next_mode_and_undoes_the_set(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False, level="MANUAL")
    fake.stuck_determinism = True
    with smi.GpuStabilizer([1], require_stable=True) as stabilizer:
        record = stabilizer.summary()["perGpu"][1]
        assert record["mode"] == "perflevel-high"
        assert "reads back MANUAL, not DETERMINISM" in record["reason"]
    assert fake.calls == [
        ("determinism", "h1", 2400),
        ("level", "h1", "manual"),
        ("level", "h1", "high"),
        ("level", "h1", "manual"),
    ]
    assert fake.level == "MANUAL"


@pytest.mark.parametrize("prior", ["AUTO", "MANUAL"])
def test_determinism_restore_sets_the_prior_level(monkeypatch, prior):
    fake = install_fake_smi(monkeypatch, fail=False, level=prior)
    with smi.GpuStabilizer([0], require_stable=True):
        assert fake.level == "DETERMINISM"
    assert fake.calls[-1] == ("level", "h0", prior.lower())
    assert fake.level == prior


def test_failed_restore_resets_to_auto_and_is_retried(monkeypatch, capsys):
    fake = install_fake_smi(monkeypatch, fail=False, level="MANUAL")
    stabilizer = smi.GpuStabilizer([0], require_stable=True)
    stabilizer.__enter__()
    fake.fail_levels = {"manual"}
    stabilizer.restore()
    assert fake.level == "AUTO"
    assert not stabilizer.locks[0].restored
    assert "GPU 0: restoring MANUAL failed" in capsys.readouterr().err
    fake.fail_levels = set()
    stabilizer.restore()
    assert stabilizer.locks[0].restored and fake.level == "MANUAL"
    calls = len(fake.calls)
    stabilizer.restore()
    assert len(fake.calls) == calls
    stabilizer.close()


def test_failed_restore_and_reset_name_the_manual_command(monkeypatch, capsys):
    fake = install_fake_smi(monkeypatch, fail=False, level="MANUAL")
    stabilizer = smi.GpuStabilizer([0], require_stable=True)
    stabilizer.__enter__()
    fake.fail_levels = {"manual", "auto"}
    stabilizer.restore()
    assert "amd-smi reset -g 0 --clocks" in capsys.readouterr().err
    fake.fail_levels = set()
    stabilizer.close()


AMD_SMI = "/opt/rocm/bin/amd-smi"
BDF1 = "0000:95:00.0"


class FakeSudoRunner:
    """A stand-in for subprocess.run that plays sudo and amd-smi: it records every
    call and applies the sets to the FakeAmdSmi."""

    def __init__(self, fake: FakeAmdSmi):
        self.fake = fake
        self.calls: list[tuple[list[str], dict]] = []
        # The result of sudo -n -v.
        self.cached = True
        # The result of sudo -v and whether it leaves credentials behind.
        self.prompt_rc = 0
        self.prompt_raises: BaseException | None = None
        self.keeps_credentials = True
        # The exit code of amd-smi.
        self.set_rc = 0

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if argv == ["sudo", "-n", "-v"]:
            return self.done(argv, 0 if self.cached else 1)
        if argv == ["sudo", "-v"]:
            return self.prompt(argv)
        return self.amd_smi(argv)

    def done(self, argv, rc, stderr=""):
        return subprocess.CompletedProcess(argv, rc, stdout="", stderr=stderr)

    def prompt(self, argv):
        if self.prompt_raises is not None:
            raise self.prompt_raises
        if self.prompt_rc == 0 and self.keeps_credentials:
            self.cached = True
        return self.done(argv, self.prompt_rc)

    def amd_smi(self, argv):
        assert argv[:3] == ["sudo", "-n", AMD_SMI]
        if self.set_rc != 0:
            return self.done(argv, self.set_rc, "Permission denied\n\n")
        if argv[3] == "set" and argv[6] == "-l":
            self.fake.level = argv[7]
        if argv[3] == "set" and argv[6] == "-d":
            self.fake.level = "DETERMINISM"
        return self.done(argv, 0)

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _ in self.calls]


def install_sudo(monkeypatch, fake, runner, has_tty=True, interval=3600):
    """Makes the stabilizer lock through the fake sudo runner; the atexit
    registrations are recorded in fake.registered instead of made."""
    fake.bdf = {"h0": "0000:85:00.0", "h1": BDF1}
    session = smi.SudoSession(
        AMD_SMI, runner=runner, has_tty=lambda: has_tty, interval=interval
    )
    monkeypatch.setattr(smi, "make_sudo_session", lambda: session)
    fake.registered = []
    monkeypatch.setattr(smi.atexit, "register", fake.registered.append)
    monkeypatch.setattr(smi.atexit, "unregister", lambda f: None)
    # Nothing but the fake runner may run a process.
    monkeypatch.setattr(smi.subprocess, "run", pytest.fail)
    return session


def sudo_setup(monkeypatch, cached=True, **options):
    fake = install_fake_smi(monkeypatch, fail=False)
    runner = FakeSudoRunner(fake)
    runner.cached = cached
    session = install_sudo(monkeypatch, fake, runner, **options)
    return fake, runner, session


def set_argv(*args):
    return ["sudo", "-n", AMD_SMI, "set", "-g", BDF1, *args]


def test_sudo_prompts_then_locks_and_restores_through_sudo(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)
    with smi.GpuStabilizer([1], require_stable=True) as stabilizer:
        assert stabilizer.summary()["status"] == "locked"
    argvs = runner.argvs()
    assert argvs[:3] == [["sudo", "-n", "-v"], ["sudo", "-v"], ["sudo", "-n", "-v"]]
    assert runner.calls[1][1] == {}
    sets = [argv for argv in argvs if argv[:3] == ["sudo", "-n", AMD_SMI]]
    assert sets == [set_argv("-d", "2400"), set_argv("-l", "AUTO")]
    assert not fake.calls


def test_no_password_is_ever_passed(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)
    with smi.GpuStabilizer([1], require_stable=True):
        pass
    with pytest.raises(smi.SweepAbort):
        with smi.GpuStabilizer([1], require_stable=True):
            runner.cached = False
    assert runner.calls
    for argv, kwargs in runner.calls:
        assert "-S" not in argv
        assert "input" not in kwargs
        assert kwargs.get("stdin") in (None, subprocess.DEVNULL)
        assert argv == ["sudo", "-v"] or argv[1] == "-n"


def test_root_sets_in_process_without_sudo(monkeypatch):
    fake = FakeAmdSmi(fail=False)
    monkeypatch.setattr(
        smi,
        "_smi",
        types.SimpleNamespace(
            module=fake, pid=os.getpid(), handles=["h0", "h1"], handle=lambda g: f"h{g}"
        ),
    )
    monkeypatch.setattr(smi.os, "geteuid", lambda: 0)
    monkeypatch.setattr(smi.subprocess, "run", pytest.fail)
    assert smi.make_sudo_session() is None
    with smi.GpuStabilizer([1], require_stable=True):
        pass
    assert fake.calls == [("determinism", "h1", 2400), ("level", "h1", "auto")]


def test_not_root_makes_a_session_with_an_absolute_amd_smi(monkeypatch):
    monkeypatch.setattr(smi.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(smi, "find_amd_smi", lambda: AMD_SMI)
    session = smi.make_sudo_session()
    assert isinstance(session, smi.SudoSession)
    assert os.path.isabs(session.amd_smi)


def test_refused_password_aborts_before_anything_is_locked(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)
    runner.prompt_rc = 1
    sigterm = signal.getsignal(signal.SIGTERM)
    with pytest.raises(smi.SweepAbort) as info:
        with smi.GpuStabilizer([1], require_stable=True):
            pass
    message = str(info.value)
    assert "sudo -v" in message and "! sudo -v" in message
    assert "nothing was locked" in message
    assert all(argv[:2] in (["sudo", "-n"], ["sudo", "-v"]) for argv in runner.argvs())
    assert not any(argv[2:3] == [AMD_SMI] for argv in runner.argvs())
    assert not fake.calls
    assert fake.registered == []
    assert signal.getsignal(signal.SIGTERM) is sigterm


def test_no_terminal_aborts_without_prompting(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False, has_tty=False)
    with pytest.raises(smi.SweepAbort, match="no terminal"):
        with smi.GpuStabilizer([1], require_stable=True):
            pass
    assert ["sudo", "-v"] not in runner.argvs()


def test_cancelled_prompt_aborts(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)
    runner.prompt_raises = KeyboardInterrupt()
    with pytest.raises(smi.SweepAbort, match="cancelled"):
        with smi.GpuStabilizer([1], require_stable=True):
            pass


def test_credentials_that_are_not_kept_abort(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)
    runner.keeps_credentials = False
    with pytest.raises(smi.SweepAbort, match="timestamp_timeout"):
        with smi.GpuStabilizer([1], require_stable=True):
            pass
    assert not fake.calls


def test_cached_credentials_are_not_prompted_for(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=True)
    with smi.GpuStabilizer([1], require_stable=True):
        pass
    assert ["sudo", "-v"] not in runner.argvs()


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_heartbeat_refreshes_until_stopped(monkeypatch):
    runner = FakeSudoRunner(FakeAmdSmi(fail=False))
    session = smi.SudoSession(AMD_SMI, runner=runner, interval=0.01)
    session.start_heartbeat()
    assert wait_for(lambda: len(runner.calls) >= 2)
    session.stop_heartbeat()
    assert not session.thread.is_alive()
    assert all(argv == ["sudo", "-n", "-v"] for argv in runner.argvs())
    count = len(runner.calls)
    time.sleep(0.05)
    assert len(runner.calls) == count


def test_heartbeat_warns_once_and_exits_when_the_credentials_expire(capsys):
    runner = FakeSudoRunner(FakeAmdSmi(fail=False))
    runner.cached = False
    session = smi.SudoSession(AMD_SMI, runner=runner, interval=0.01)
    session.start_heartbeat()
    session.thread.join(5)
    assert not session.thread.is_alive()
    assert capsys.readouterr().err.count("the sudo credentials expired") == 1


def test_expired_credentials_at_restore_print_the_commands_and_fail(
    monkeypatch, capsys
):
    fake, runner, _ = sudo_setup(monkeypatch)
    stabilizer = smi.GpuStabilizer([1], require_stable=True)
    with pytest.raises(smi.SweepAbort, match="still locked"):
        with stabilizer:
            runner.cached = False
    err = capsys.readouterr().err
    assert f"run: sudo {AMD_SMI} set -g {BDF1} -l AUTO" in err
    assert runner.argvs().count(["sudo", "-v"]) == 0
    assert set_argv("-l", "AUTO") not in runner.argvs()
    assert not stabilizer.locks[1].restored


def test_expired_credentials_do_not_mask_an_exception(monkeypatch, capsys):
    fake, runner, _ = sudo_setup(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        with smi.GpuStabilizer([1], require_stable=True):
            runner.cached = False
            raise RuntimeError("boom")
    assert f"run: sudo {AMD_SMI} set -g {BDF1} -l AUTO" in capsys.readouterr().err


def test_sudo_is_not_used_by_another_process():
    runner = FakeSudoRunner(FakeAmdSmi(fail=False))
    session = smi.SudoSession(AMD_SMI, runner=runner)
    session.pid = os.getpid() + 1
    assert not session.run_amd_smi("set", "-g", BDF1, "-l", "AUTO").ok
    assert not runner.calls


def test_failed_sudo_restore_names_the_reset_command_and_fails(monkeypatch, capsys):
    fake, runner, _ = sudo_setup(monkeypatch)
    stabilizer = smi.GpuStabilizer([1], require_stable=True)
    with pytest.raises(smi.SweepAbort, match="still locked"):
        with stabilizer:
            runner.set_rc = 1
    err = capsys.readouterr().err
    assert f"sudo {AMD_SMI} reset -g {BDF1} --clocks" in err
    assert "exited with 1: Permission denied" in err
    assert not stabilizer.locks[1].restored


def test_smi_failure_aborts_before_the_password_prompt(monkeypatch):
    fake, runner, _ = sudo_setup(monkeypatch, cached=False)

    def fail():
        raise smi.SweepAbort("cannot initialize amd-smi")

    monkeypatch.setattr(smi, "get_smi", fail)
    threads = threading.active_count()
    with pytest.raises(smi.SweepAbort, match="cannot initialize"):
        with smi.GpuStabilizer([1], require_stable=True):
            pass
    assert not runner.calls
    assert threading.active_count() == threads


def test_manual_restore_without_a_pci_address_does_not_guess(monkeypatch, capsys):
    fake, runner, session = sudo_setup(monkeypatch, cached=False)
    stabilizer = smi.GpuStabilizer([1], require_stable=True)
    stabilizer.pid = os.getpid()
    stabilizer.sudo = session
    stabilizer.locks[1] = smi.GpuLock(1, "AUTO", mode="perfdeterminism")
    stabilizer.restore()
    err = capsys.readouterr().err
    assert "the PCI address of" in err and "is unknown" in err
    assert "sudo " + AMD_SMI not in err
    assert not any(argv[2:3] == [AMD_SMI] for argv in runner.argvs())


def test_lock_records_the_pci_address_in_sudo_mode(monkeypatch):
    sudo_setup(monkeypatch)
    with smi.GpuStabilizer([1], require_stable=True) as stabilizer:
        assert stabilizer.locks[1].bdf == BDF1


def test_sudo_is_not_made_without_the_flag(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)

    def fail():
        raise AssertionError("no sudo without the flag")

    monkeypatch.setattr(smi, "make_sudo_session", fail)
    with smi.GpuStabilizer([1], require_stable=False):
        pass
    assert not fake.calls


def test_find_amd_smi_searches_rocm_path_then_aborts(monkeypatch, tmp_path):
    monkeypatch.setattr(smi.shutil, "which", lambda name: None)
    binary = tmp_path / "bin" / "amd-smi"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert smi.find_amd_smi() == str(binary)
    binary.chmod(0o644)
    monkeypatch.setattr(smi.os, "access", lambda path, mode: False)
    with pytest.raises(smi.SweepAbort, match="cannot find the amd-smi"):
        smi.find_amd_smi()


def test_terminal_available_never_raises():
    assert isinstance(smi.terminal_available(), bool)


def test_use_is_averaged_and_not_a_number_is_skipped(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    fake.activity = {"h0": [10, "N/A", 20, "N/A", 30], "h1": ["N/A"] * 5}
    sleeps = []
    use = smi.read_use(sleep=sleeps.append)
    assert use == {0: 20.0, 1: None}
    assert sleeps == [smi.BUSY_INTERVAL_S] * (smi.BUSY_SAMPLES - 1)


def test_process_counts_exclude_this_process(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    fake.processes = {"h1": [{"pid": os.getpid()}, {"pid": 1}, {"pid": 2}]}
    assert smi.read_process_counts() == {0: 0, 1: 2}


def patch_busyness(monkeypatch, use, processes=None):
    monkeypatch.setattr(smi, "read_use", lambda: use)
    monkeypatch.setattr(smi, "read_process_counts", lambda: processes or {})


def test_unknown_use_is_busy(monkeypatch):
    patch_busyness(monkeypatch, {5: 0.0, 6: None, 7: 1.0})
    assert smi.select_gpus("auto", 5.0, False) == [5, 7]
    with pytest.raises(smi.SweepAbort, match="GPU 6 has an unknown use.*--allow-busy"):
        smi.select_gpus("5,6", 5.0, False)
    assert smi.select_gpus("5,6", 5.0, True) == [5, 6]


def test_gpu_with_processes_is_busy(monkeypatch):
    patch_busyness(monkeypatch, {5: 0.0, 6: 0.0}, {5: 2})
    assert smi.select_gpus("auto", 5.0, False) == [6]
    with pytest.raises(smi.SweepAbort, match="GPU 5 has 2 processes"):
        smi.select_gpus("5", 5.0, False)


def test_no_idle_gpu_names_every_reason(monkeypatch):
    patch_busyness(monkeypatch, {5: 90.0, 6: None})
    with pytest.raises(smi.SweepAbort, match="no GPU is idle: GPU 5 is 90.0% busy"):
        smi.select_gpus("auto", 5.0, False)


def test_metrics_sample_takes_the_throttle_status_that_is_a_number(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    fake.metrics = {"throttle_status": "N/A", "indep_throttle_status": 3}
    assert smi.sample_metrics(1) == {"sclkMHz": 100, "mclkMHz": 100, "throttleStatus": 3}
    fake.metrics = {"throttle_status": "N/A", "indep_throttle_status": "N/A"}
    assert smi.sample_metrics(1) == {"sclkMHz": 100, "mclkMHz": 100}


def test_metrics_sample_never_raises(monkeypatch):
    install_fake_smi(monkeypatch, fail=False)
    monkeypatch.setattr(smi, "read_clock", lambda gpu, kind: 1 / 0)
    assert smi.sample_metrics(1) == {}


def test_clock_sampler_summarizes_the_samples():
    items = [
        {"sclkMHz": 2000, "throttleStatus": 0},
        {"sclkMHz": 1000, "throttleStatus": 4},
        {"sclkMHz": 1500},
    ]

    def sampler(gpu):
        item = items.pop(0)
        if not items:
            clock.stopped.set()
        return item

    clock = smi.ClockSampler(5, sampler=sampler, interval=0.001)
    with clock:
        clock.thread.join(10)
    assert clock.summary() == {
        "samples": 3,
        "sclkMinMHz": 1000,
        "sclkMeanMHz": 1500.0,
        "sclkMaxMHz": 2000,
        "throttledSamples": 1,
    }


def test_clock_sampler_takes_a_sample_for_a_short_body():
    clock = smi.ClockSampler(5, sampler=lambda gpu: {"sclkMHz": 7}, interval=60)
    with clock:
        pass
    assert clock.summary()["samples"] == 1
    assert smi.summarize_samples([]) == {"samples": 0, "throttledSamples": 0}


@pytest.mark.parametrize(
    "clocks, throttled",
    [
        ({}, False),
        ({"during": {"throttledSamples": 0}, "start": {"throttleStatus": 0}}, False),
        ({"during": {"throttledSamples": 2}}, True),
        ({"during": {"throttledSamples": 0}, "end": {"throttleStatus": 8}}, True),
        ({"start": {"throttleStatus": 1}}, True),
    ],
)
def test_case_record_marks_throttled_cases(tmp_path, capsys, clocks, throttled):
    instance, worker_ = loop_sweep(tmp_path, ["c0"])
    instance.on_message(worker_, case_message(0, clocks=clocks))
    assert instance.writer.records[0].get("throttled", False) is throttled
    assert instance.meta["casesThrottled"] == int(throttled)
    assert ("c0: a throttler was active" in capsys.readouterr().err) is throttled


@pytest.mark.parametrize("one_shot", [False, True])
def test_time_stage_records_the_clocks_around_and_during_the_timing(monkeypatch, one_shot):
    monkeypatch.setattr(worker, "sample_metrics", lambda gpu: {"sclkMHz": 1})
    monkeypatch.setattr(worker, "screen", lambda *args: {0: "screened"})
    monkeypatch.setattr(worker, "refine", lambda *args: {0: "refined"})
    monkeypatch.setattr(
        worker, "ClockSampler", lambda gpu: smi.ClockSampler(gpu, lambda g: {"sclkMHz": 5})
    )
    job = types.SimpleNamespace(one_shot=one_shot)
    outcome = {"phases": {}}
    screening, refined = worker.time_stage(
        job, 3, None, types.SimpleNamespace(ramp=lambda ms: None), outcome, None
    )
    assert screening == {0: "screened"}
    assert refined == (None if one_shot else {0: "refined"})
    clocks = outcome["clocks"]
    assert clocks["start"] == clocks["end"] == {"sclkMHz": 1}
    assert clocks["during"]["sclkMaxMHz"] == 5 and clocks["during"]["throttledSamples"] == 0
    assert outcome["timed"] == 1 and (outcome["phases"]["refinementS"] == 0.0) is one_shot


def test_pci_addresses_compare_domain_bus_and_device_only():
    assert smi.format_pci_address(0, 0x95, 0) == "0000:95:00.0"
    assert smi.parse_pci_address("0000:95:00.1") == smi.parse_pci_address("0000:95:00.0")
    assert smi.parse_pci_address("N/A") is None and smi.parse_pci_address(None) is None


def install_mapped_smi(monkeypatch, mapping: dict[int, str]) -> FakeAmdSmi:
    """An amd-smi whose indices are not the HIP ordinals: handle h2 is the device
    0000:85:00.0 and h1 is 0000:e5:00.0."""
    fake = FakeAmdSmi(fail=False)
    fake.activity = {f"h{i}": [0] * 20 for i in range(3)}
    real = smi.AmdSmi.__new__(smi.AmdSmi)
    real.module = fake
    real.handles = ["h0", "h1", "h2"]
    real.by_address = {(0, 0xE5, 0): "h1", (0, 0x85, 0): "h2"}
    monkeypatch.setattr(smi, "_smi", real)
    monkeypatch.setattr(smi, "_hip_ordinals", None)
    monkeypatch.setattr(smi.time, "sleep", lambda seconds: None)
    smi.use_hip_ordinals(mapping)
    real.pid = os.getpid()
    return fake


def test_ordinals_resolve_to_the_handle_with_the_same_pci_address(monkeypatch):
    install_mapped_smi(monkeypatch, {5: "0000:85:00.0", 6: "0000:E5:00.0"})
    handle = smi.get_smi().handle
    assert handle(5) == "h2" and handle(6) == "h1"


def test_ordinal_missing_from_amd_smi_is_reported(monkeypatch):
    install_mapped_smi(monkeypatch, {4: "0000:aa:00.0"})
    handle = smi.get_smi().handle
    with pytest.raises(smi.SweepAbort, match=r"HIP device 4 \(0000:aa:00.0\) is not reported"):
        handle(4)
    with pytest.raises(smi.SweepAbort, match="HIP device 9 .unknown address."):
        handle(9)


def test_handles_are_keyed_by_the_pci_address(monkeypatch):
    fake = FakeAmdSmi(fail=False)
    fake.bdf = {"a": "0000:95:00.0", "c": "0000:E5:00.1"}
    real = smi.AmdSmi.__new__(smi.AmdSmi)
    real.module, real.handles = fake, ["a", "b", "c"]
    assert real.handles_by_address() == {(0, 0x95, 0): "a", (0, 0xE5, 0): "c"}


def test_auto_selects_idle_hip_ordinals(monkeypatch):
    fake = install_mapped_smi(
        monkeypatch, {5: "0000:85:00.0", 6: "0000:e5:00.0", 7: "0000:ff:00.0"}
    )
    fake.activity["h1"] = [90] * 20
    assert smi.select_gpus("auto", 5.0, False) == [5]
    with pytest.raises(
        smi.SweepAbort, match=r"HIP device 6 \(0000:e5:00.0\) is 90.0% busy"
    ):
        smi.select_gpus("5,6", 5.0, False)
    with pytest.raises(smi.SweepAbort, match="HIP device 7 .0000:ff:00.0. has an unknown use"):
        smi.select_gpus("5,7", 5.0, False)
    assert smi.select_gpus("5", 5.0, False) == [5]


def test_mapped_amd_smi_calls_use_the_device_of_the_ordinal(monkeypatch):
    fake = install_mapped_smi(monkeypatch, {5: "0000:85:00.0"})
    assert smi.sample_metrics(5) == {"sclkMHz": 100, "mclkMHz": 100}
    assert smi.set_perf_level(5, "high").ok
    assert fake.calls == [("level", "h2", "high")]


def test_probe_failure_keeps_the_amd_smi_numbering(monkeypatch, capsys):
    monkeypatch.setattr(smi, "_hip_ordinals", None)

    def probe():
        raise TimeoutError("no answer")

    monkeypatch.setattr(smi, "probe_hip_ordinals", probe)
    smi.map_hip_ordinals()
    assert smi._hip_ordinals is None
    assert "warning: cannot probe the HIP devices (TimeoutError: no answer)" in (
        capsys.readouterr().err
    )


def test_probe_returns_the_answer_of_the_child():
    answer = smi.probe_hip_ordinals(timeout=60, function=functools.partial(dict, a=1))
    assert answer == {"a": 1}


def test_probe_reports_the_failure_of_the_child():
    with pytest.raises(RuntimeError, match="ValueError: invalid literal"):
        smi.probe_hip_ordinals(timeout=60, function=functools.partial(int, "x"))


def test_probe_timeout_terminates_the_child():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no answer within 0.5 s"):
        smi.probe_hip_ordinals(timeout=0.5, function=functools.partial(time.sleep, 60))
    assert time.monotonic() - started < 10


def test_probe_success_maps_the_ordinals(monkeypatch):
    monkeypatch.setattr(smi, "_hip_ordinals", None)
    monkeypatch.setattr(smi, "probe_hip_ordinals", lambda: {0: "0000:75:00.0"})
    smi.map_hip_ordinals()
    assert smi._hip_ordinals == {0: "0000:75:00.0"}
    assert smi.describe(0) == "HIP device 0 (0000:75:00.0)"


def test_hip_pci_addresses_drop_the_masks_and_list_every_ordinal(monkeypatch):
    for name in worker.VISIBILITY_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(worker.support, "getDeviceCount", lambda: 3)
    monkeypatch.setattr(worker, "pci_address_of", lambda device: f"0000:{device:02x}:00.0")
    assert worker.hip_pci_addresses() == {
        0: "0000:00:00.0",
        1: "0000:01:00.0",
        2: "0000:02:00.0",
    }
    assert not any(name in os.environ for name in worker.VISIBILITY_VARIABLES)


def test_gpu_worker_maps_its_id_to_its_own_pci_address(monkeypatch):
    calls = []
    patch_worker_process(monkeypatch, None)
    monkeypatch.setattr(worker, "use_hip_ordinals", calls.append)
    worker.gpu_worker(2, types.SimpleNamespace(send=lambda m: None, recv=lambda: None))
    assert calls == [{2: "0000:75:00.0"}]


def test_bdf_mismatch_aborts(monkeypatch):
    monkeypatch.setattr(smi, "device_bdf", lambda gpu: "0000:95:00.0")
    smi.check_same_device(5, "0000:95:00.0")
    with pytest.raises(
        smi.SweepAbort,
        match="GPU 5: amd-smi device 0000:95:00.0 is not HIP device 5 .0000:75:00.0.; "
        "the ids are not the same numbering",
    ):
        smi.check_same_device(5, "0000:75:00.0")


@pytest.mark.parametrize("smi_address, hip_address", [(None, "0000:75:00.0"), ("0000:95:00.0", None)])
def test_unavailable_bdf_warns_and_continues(monkeypatch, capsys, smi_address, hip_address):
    monkeypatch.setattr(smi, "device_bdf", lambda gpu: smi_address)
    smi.check_same_device(5, hip_address)
    assert "warning: GPU 5: cannot compare" in capsys.readouterr().err


def test_device_bdf_is_none_when_it_cannot_be_read(monkeypatch):
    fake = install_fake_smi(monkeypatch, fail=False)
    fake.bdf = {"h0": "0000:95:00.0"}
    assert smi.device_bdf(0) == "0000:95:00.0"
    assert smi.device_bdf(1) is None


def test_ready_message_with_another_device_aborts_the_sweep(tmp_path, monkeypatch):
    instance, worker_ = loop_sweep(tmp_path, ["c0"])
    worker_.ready = False
    monkeypatch.setattr(smi, "device_bdf", lambda gpu: "0000:95:00.0")
    ready = {"name": "n", "timing": "t", "pci": "0000:75:00.0"}
    with pytest.raises(smi.SweepAbort, match="not the same numbering"):
        instance.on_message(worker_, ("ready", 0, ready))
    assert not worker_.ready


def small_problem(spk: int | None, use_beta: bool, beta: float) -> support.Problem:
    case = parse(
        f"""
problems:
  - name: a
    m: 5
    n: 300
    k: 1000
    ldD: 16
    ldC: 16
    spk: {"k" if spk is None else spk}
    useBeta: {str(use_beta).lower()}
    alpha: 0.75
    beta: {beta}
    sweep: {{bm: 2, tn: 1, lanesK: 16, waves: 1, prefetch: 1, upScales: 0}}
"""
    ).cases[0]
    return config.expand_case(case).problems[0]


def make_verifier(p: support.Problem, full: bool) -> inputs_module.Verifier:
    cache = inputs_module.OperandCache()
    return inputs_module.Verifier(inputs_module.make_case_inputs(p, cache), full)


def d_bits_of(p, expected: np.ndarray) -> np.ndarray:
    """D bits holding the bf16 rounding of expected [m, n] in the layout of D."""
    bits, _ = support.makeOutputs(p.m, p.n, p.ldD, 1)
    index = np.arange(p.n)[:, None] * p.ldD + np.arange(p.m)
    bits[index] = support.roundToBf16(expected.T)
    return bits


def split_workspace(p, verifier) -> np.ndarray:
    """The workspace [splits, n, m] of exact per-split partial products."""
    a = verifier.inputs.a.dequant(slice(None)).astype(np.float64)
    b = verifier.inputs.b.dequant(slice(None)).astype(np.float64)
    ws = np.empty((p.splits, p.n, p.m), dtype=np.float32)
    for split in range(p.splits):
        low, high = split * p.spk, min(p.k, (split + 1) * p.spk)
        ws[split] = (a[:, low:high] @ b[:, low:high].T).T
    return ws.ravel()


def test_dense_product_matches_support_reference():
    a = inputs_module.generate_operand(3, 4100, 4100, 1, 0)
    b = inputs_module.generate_operand(300, 4100, 4104, 2, 1)
    codes_a, codes_b = a.codes().T, b.codes().T
    expected = support.referenceProduct(
        support.dequant(codes_a, a.scales), support.dequant(codes_b, b.scales)
    )
    np.testing.assert_allclose(
        inputs_module.dense_product(a, b), expected, rtol=1e-4, atol=1e-2
    )
    assert (a.raw.size, b.swizzled.size) == (
        support.operandExtent(3, 4100, 4100),
        support.scaleExtent(300, 4100),
    )


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("use_beta, beta", [(False, 0.0), (True, 0.5)])
def test_verifier_single_split(full, use_beta, beta):
    p = small_problem(None, use_beta, beta)
    verifier = make_verifier(p, full)
    dense = make_verifier(p, True)
    expected = support.singleSplitD(p, dense.support_inputs())
    d_bits = d_bits_of(p, expected)
    verifier.check_gemm(p, d_bits, np.zeros(1, dtype=np.float32))
    count = len(verifier.sample(p)[0])
    assert verifier.label(p) == ("full" if full else f"sampled({count})")
    rows, cols = verifier.sample(p)
    corrupt = d_bits.copy()
    corrupt[cols[3] * p.ldD + rows[3]] ^= 0x4000
    with pytest.raises(AssertionError):
        verifier.check_gemm(p, corrupt, np.zeros(1, dtype=np.float32))
    padded = d_bits.copy()
    padded[p.m] = 0
    with pytest.raises(AssertionError, match="padding"):
        verifier.check_gemm(p, padded, np.zeros(1, dtype=np.float32))


def widest_two_ulp_position(p, verifier, d_bits, atol, d_rtol) -> int:
    """The D index of the checked element whose +2 ulp error is rejected by
    d_rtol and accepted by the default, with the largest margin over atol."""
    rows, cols = verifier.sample(p)
    positions = cols * p.ldD + rows
    values = np.abs(support.bf16ToFloat32(d_bits[positions])).astype(np.float64)
    values[values == 0] = np.nan
    ulps = 2.0 ** (np.floor(np.log2(values)) - 7)
    error = 2 * ulps
    low_margin = error - d_rtol * values - atol
    high_margin = 1e-2 * values - error - atol
    margin = np.nan_to_num(np.minimum(low_margin, high_margin), nan=-1.0)
    assert margin.max() > 0, "no element separates the two tolerances"
    return int(positions[np.argmax(margin)])


@pytest.mark.parametrize("full", [True, False])
def test_verifier_d_rtol_is_configurable(full):
    import select_skinny_gemm_from_sweep as selector

    p = small_problem(None, True, 0.5)
    inputs = inputs_module.make_case_inputs(p, inputs_module.OperandCache())
    strict = inputs_module.Verifier(inputs, full, d_rtol=selector.BF16_EPS)
    default = inputs_module.Verifier(inputs, full)
    expected = support.singleSplitD(p, make_verifier(p, True).support_inputs())
    d_bits = d_bits_of(p, expected)
    no_workspace = np.zeros(1, dtype=np.float32)
    strict.check_gemm(p, d_bits, no_workspace)

    atol = 1e-4 * max(1.0, float(np.abs(expected).max()))
    position = widest_two_ulp_position(p, strict, d_bits, atol, selector.BF16_EPS)
    one_ulp = d_bits.copy()
    one_ulp[position] += 1
    strict.check_gemm(p, one_ulp, no_workspace)
    two_ulps = d_bits.copy()
    two_ulps[position] += 2
    with pytest.raises(AssertionError, match="D mismatch"):
        strict.check_gemm(p, two_ulps, no_workspace)
    default.check_gemm(p, two_ulps, no_workspace)


@pytest.mark.parametrize("full", [True, False])
def test_verifier_split_k_workspace_and_reduce(full):
    p = small_problem(256, True, 0.5)
    assert p.splits == 4
    verifier = make_verifier(p, full)
    ws = split_workspace(p, verifier)
    d_unused = np.zeros(1, dtype=np.uint16)
    verifier.check_gemm(p, d_unused, ws)
    rows, cols = verifier.sample(p)
    corrupt = ws.copy()
    # The last split, which holds the tail: only its own slice is wrong.
    corrupt[3 * p.m * p.n + cols[5] * p.m + rows[5]] += 100.0
    with pytest.raises(AssertionError, match=None if full else "split 3"):
        verifier.check_gemm(p, d_unused, corrupt)
    expected = support.expectedD(p, make_verifier(p, True).support_inputs())
    d_bits = d_bits_of(p, expected)
    verifier.check_reduced(p, d_bits)
    d_bits[cols[7] * p.ldD + rows[7]] ^= 0x4000
    with pytest.raises(AssertionError):
        verifier.check_reduced(p, d_bits)


def test_sample_positions_cover_ends_and_tile_boundaries():
    p = small_problem(None, False, 0.0)
    rows, cols = inputs_module.sample_positions(p)
    assert set(rows.tolist()) == set(range(p.m))
    assert {0, p.n - 1, p.bn - 1, p.bn} <= set(cols.tolist())
    assert (rows < p.m).all() and (cols < p.n).all()


def test_operand_generation_is_deterministic_and_cached():
    cache = inputs_module.OperandCache()
    first = cache.get(inputs_module.ROLE_A, 70, 520, 524)
    assert cache.get(inputs_module.ROLE_A, 70, 520, 524) is first
    again = inputs_module.generate_operand(
        70, 520, 524, inputs_module.INPUT_SEED, inputs_module.ROLE_A
    )
    assert (first.raw == again.raw).all() and (first.swizzled == again.swizzled).all()
    assert not np.isnan(support.decodeE4M3FN(first.codes())).any()


def test_rank_all_screened_lists_every_configuration():
    timings = {
        0: worker.Timing([3000], []),
        1: worker.Timing([1000], []),
        2: worker.Timing([2000], []),
    }
    outcome: dict = {"ranked": []}
    verified = {0: "full", 1: "sampled(4)", 2: "no"}
    worker.rank_all_screened(timings, outcome, verified)
    assert [entry["index"] for entry in outcome["ranked"]] == [1, 2, 0]
    assert outcome["ranked"][0]["verified"] == "sampled(4)"
    assert outcome["ranked"][0]["timeUs"] == outcome["ranked"][0]["screeningUs"]


def test_results_msgpack_roundtrip(tmp_path):
    import sweep_skinny_gemm_results as results_io

    document = {
        "meta": {"tool": "test", "oneShot": True},
        "results": [{"problem": {"name": "a"}, "top": []}],
    }
    path = tmp_path / "out.msgpack"
    results_io.write_results(str(path), document, results_io.FORMAT_MSGPACK)
    loaded = results_io.read_results(str(path))
    assert loaded["meta"]["oneShot"] is True
    assert loaded["results"][0]["problem"]["name"] == "a"


def test_wrong_result_is_dropped_and_next_promoted(monkeypatch):
    p = config.expand_case(
        parse(
            "problems: [{name: a, m: 4, n: 64, k: 256, sweep: {tn: [1, 2, 4]}}]"
        ).cases[0]
    ).problems
    job = worker.Job(
        0,
        [(support.gemmKernelName(q), q) for q in p],
        "",
        {"top": 2, "verify_full_max_elems": 10**9},
        True,
    )

    def check(job, buffers, problem, name, verifier):
        assert problem.tn != 1, "D mismatch"

    monkeypatch.setattr(worker, "check_result", check)
    buffers = types.SimpleNamespace(inputs=None)
    timings = {i: worker.Timing([1000 * (i + 1)], []) for i in range(3)}
    outcome = {"ranked": [], "failures": []}
    worker.pick_winners(job, buffers, timings, timings, outcome)
    assert [w["index"] for w in outcome["ranked"]] == [1, 2]
    assert outcome["failures"][0]["status"] == "wrong result"
    assert all(w["verified"] == "full" for w in outcome["ranked"])
    job.verify = False
    outcome = {"ranked": [], "failures": []}
    worker.pick_winners(job, buffers, timings, timings, outcome)
    assert {w["verified"] for w in outcome["ranked"]} == {"no"}


def test_a_and_b_differ_when_shape_and_ld_are_equal():
    cache = inputs_module.OperandCache()
    a = cache.get(inputs_module.ROLE_A, 64, 512, 512)
    b = cache.get(inputs_module.ROLE_B, 64, 512, 512)
    assert a is not b
    assert (a.raw != b.raw).any() and (a.scales != b.scales).any()
    again = inputs_module.OperandCache().get(inputs_module.ROLE_B, 64, 512, 512)
    assert (again.raw == b.raw).all() and (again.swizzled == b.swizzled).all()
    square = config.expand_case(
        parse("problems: [{name: a, m: 64, n: 64, k: 512}]").cases[0]
    ).problems[0]
    assert square.m == square.n and square.ldA == square.ldB
    inputs = inputs_module.make_case_inputs(square, cache)
    assert (inputs.a.raw != inputs.b.raw).any()


def alpha_beta(text: str, seed: int | None = None) -> dict:
    """(alpha, beta) of every case, by problem and beta mode."""
    cases = config.parse_input(yaml.safe_load(text), seed).cases
    return {
        (c.name, c.m, c.n, c.k, c.spk, c.use_beta, c.beta_mode): (c.alpha, c.beta)
        for c in cases
    }


SHAPES = "{name: a, m: [1, 2, 4], n: [64, 128], k: 512, transpose: true, useBeta: true}"


def test_random_alpha_beta_are_deterministic_and_independent_of_order():
    forward = alpha_beta(f"problems: [{SHAPES}]")
    assert forward == alpha_beta(f"problems: [{SHAPES}]")
    assert alpha_beta(f"problems: [{SHAPES}]", seed=7) != forward
    # Entries listed in another order, alone or with other entries around.
    extra = "{name: b, m: 3, n: 64, k: 512}"
    shuffled = alpha_beta(f"problems: [{extra}, {SHAPES}]")
    assert {key: shuffled[key] for key in forward} == forward
    alone = "problems: [{name: a, m: 4, n: 128, k: 512, useBeta: true}]"
    for key, value in alpha_beta(alone).items():
        assert forward[key] == value


def test_random_alpha_beta_ranges():
    text = (
        "problems: [{name: a, m: [1, 2, 3, 4, 5, 6, 7, 8], n: [64, 96, 128, 160], "
        "k: 512, useBeta: true}]"
    )
    values = alpha_beta(text, seed=1)
    alphas = [a for a, _ in values.values()]
    betas = [b for (*_, mode), (_, b) in values.items() if mode == "random"]
    assert all(0.25 <= a <= 2.0 and abs(a - 1) >= 0.05 for a in alphas)
    assert all(0.25 <= abs(b) <= 2.0 and abs(abs(b) - 1) >= 0.05 for b in betas)
    assert any(b < 0 for b in betas) and any(b > 0 for b in betas)


def test_default_covers_both_use_beta_and_both_beta_modes():
    values = alpha_beta("problems: [{name: a, m: 4, n: 64, k: 512}]")
    # The kernel that never reads C runs once; the useBeta kernel in two modes.
    assert sorted((key[5], key[6]) for key in values) == [
        (False, "none"),
        (True, "random"),
        (True, "zero"),
    ]
    zero = values[("a", 4, 64, 512, None, True, "zero")]
    random = values[("a", 4, 64, 512, None, True, "random")]
    assert zero[1] == 0.0 and random[1] not in (0.0, 1.0)
    # The modes of a problem share alpha, which is drawn once per problem.
    assert zero[0] == random[0]
    assert values[("a", 4, 64, 512, None, False, "none")][1] == 0.0


@pytest.mark.parametrize("SPK", ["", ", spk: k", ", spk: [k]"])
def test_default_spk_never_builds_a_split_k_problem(SPK):
    cases = parse(
        f"problems: [{{name: a, m: 4, n: 4096, k: 8192{SPK}, "
        "sweep: {mfma: [false, true], grid3d: [true, false]}}]"
    ).cases
    assert cases and all(c.spk is None for c in cases)
    for case in cases:
        problems = config.expand_case(case).problems
        assert problems
        for problem in problems:
            assert not problem.use_split_k and problem.splits == 1
            assert config.config_record(problem)["spk"] is None
            assert "_USE_SPLIT_K1" not in support.gemmKernelName(problem)


@pytest.mark.parametrize("SPK", [4096, 8192, 16384])
def test_an_int_spk_builds_the_split_k_variant_even_for_one_split(SPK):
    case = parse(
        f"problems: [{{name: a, m: 4, n: 4096, k: 8192, spk: {SPK}, useBeta: false}}]"
    ).cases[0]
    assert case.spk == SPK
    for problem in config.expand_case(case).problems:
        assert problem.use_split_k and problem.spk == SPK
        assert config.config_record(problem)["spk"] == SPK
        assert support.gemmKernelName(problem).endswith("_USE_SPLIT_K1")


def test_split_k_makes_beta_matter_without_use_beta():
    text = "problems: [{name: a, m: 4, n: 64, k: 512, spk: 256, useBeta: false}]"
    values = alpha_beta(text)
    assert sorted(key[6] for key in values) == ["random", "zero"]


def test_beta_modes_override_and_pinned_beta():
    only_zero = "{name: a, m: 4, n: 64, k: 512, useBeta: true, betaModes: [zero]}"
    assert [k[6] for k in alpha_beta(f"problems: [{only_zero}]")] == ["zero"]
    pinned = "{name: a, m: 4, n: 64, k: 512, alpha: 0.5, beta: 0.5}"
    values = alpha_beta(f"problems: [{pinned}]")
    assert values[("a", 4, 64, 512, None, True, "pinned")] == (0.5, 0.5)
    assert values[("a", 4, 64, 512, None, False, "pinned")] == (0.5, 0.5)
    assert len(values) == 2
    zero = "{name: a, m: 4, n: 64, k: 512, useBeta: true, beta: 0.0}"
    zeros = alpha_beta(f"problems: [{zero}]")
    assert list(zeros) == [("a", 4, 64, 512, None, True, "zero")]
    assert list(zeros.values())[0][1] == 0.0
    ranged = (
        "{name: a, m: 4, n: 64, k: 512, useBeta: true, "
        "beta: {random: {min: 3, max: 4}}}"
    )
    values = alpha_beta(f"problems: [{ranged}]")
    assert values[("a", 4, 64, 512, None, True, "zero")][1] == 0.0
    assert 3 <= abs(values[("a", 4, 64, 512, None, True, "random")][1]) <= 4


def test_alpha_beta_spec_errors():
    base = "problems: [{name: a, m: 4, n: 64, k: 512, beta: %s}]"
    bads = ("{rnd: 1}", "{random: {min: 0, max: 1}}", "{random: {min: 2, max: 1}}")
    for bad in (*bads, "fixed"):
        with pytest.raises(config.SweepConfigError):
            config.parse_input(yaml.safe_load(base % bad))


def test_resolved_alpha_beta_are_in_the_problem_record():
    case = config.parse_input(yaml.safe_load(f"problems: [{SHAPES}]")).cases[0]
    record = case.record()
    assert (record["alpha"], record["beta"]) == (case.alpha, case.beta)
    assert record["betaMode"] == case.beta_mode == "zero"
    assert case.alpha != 1.0 and case.beta == 0.0
    random = config.parse_input(yaml.safe_load(f"problems: [{SHAPES}]")).cases[1]
    assert random.beta_mode == "random" and random.beta != 0.0


def test_beta_zero_poisons_c():
    cache = inputs_module.OperandCache()
    p = small_problem(None, True, 0.0)
    c = support.bf16ToFloat32(inputs_module.make_case_inputs(p, cache).c_bits)
    index = np.arange(p.n)[:, None] * p.ldC + np.arange(p.m)
    assert np.isnan(c[index]).all()


def selector_document(meta, results):
    return {"meta": meta, "results": results}


def write_selector_results(tmp_path, meta, results):
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump(selector_document(meta, results)))
    return str(path)


def selector_problem():
    return {
        "name": "p",
        "m": 1,
        "n": 1,
        "k": 1,
        "spk": 1,
        "useBeta": True,
        "betaMode": "one",
    }


COMPLETE_META = {"oneShot": True, "casesDone": 1, "casesTotal": 1}


def load(results, meta=COMPLETE_META, only=None, limit=None, **flags):
    flags = {"require_one_shot": True, "use_beta_only": True} | flags
    return selection.load_cases(selector_document(meta, results), only, limit, **flags)


def sweep_config_record(use_beta=True, **knobs):
    return {
        "bm": 1, "tn": 1, "lanesK": 16, "waves": 1, "prefetch": 1, "upIters": 0,
        "group": 1, "numXcds": 1, "grid3d": True, "blocks": None, "spk": None,
        "useBeta": use_beta, **knobs,
    }


def top_item(rank, time_us, kernel=None, use_beta=True, verified=None, **knobs):
    item = {
        "rank": rank,
        "timeUs": time_us,
        "kernel": kernel or f"kernel{rank}_USE_BETA1",
        "config": sweep_config_record(use_beta, **knobs),
    }
    if verified is not None:
        item["verified"] = verified
    return item


def selector_entry(top, **problem):
    return {"problem": selector_problem() | problem, "top": top}


def selector_target_and_config(spk):
    target = {
        "name": "p", "m": 5, "n": 300, "k": 2048, "ldA": 2048, "ldB": 2048,
        "ldD": 10, "ldC": 10, "alpha": 1.0, "beta": 0.0, "useBeta": False,
    }
    cfg = {
        "bm": 2, "tn": 1, "lanesK": 16, "waves": 1, "prefetch": 1, "upIters": 0,
        "group": 1, "numXcds": 1, "grid3d": True, "blocks": None, "spk": spk,
    }
    return target, cfg


def test_selector_validation_problem_rebuilds_the_recorded_variant():
    case = selection.SweepCase(0, selector_problem() | {
        "m": 5, "n": 300, "k": 2048, "ldA": 2048, "ldB": 2048, "ldD": 10,
        "ldC": 10, "alpha": 1.0, "beta": 0.0, "useBeta": False,
    }, 1.0, [])
    _, cfg = selector_target_and_config(1024)
    for spk, suffix, split_k in ((1024, "_USE_SPLIT_K1", True), (2048, "", False)):
        cfg = {**cfg, "spk": spk, "upIters": 0}
        name = "gemm_BM2_TN1_LANES_K16_WAVES1_UP_ITERS0_PF1_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0"
        cfg["kernel"] = name + suffix
        assert from_sweep.validation_problem(case, cfg).use_split_k == split_k
    # A result file of the old format records spk = k and no suffix.
    cfg["kernel"] = name
    assert not from_sweep.validation_problem(case, {**cfg, "spk": 2048}).use_split_k
    with pytest.raises(ValueError, match="does not match selected kernel"):
        from_sweep.validation_problem(case, {**cfg, "kernel": name + "X"})


def test_selector_retargets_to_the_kernel_without_split_k_when_it_cannot_split():
    target, cfg = selector_target_and_config(None)
    problem, degraded = kernels_tool.build_problem(target, cfg, 2048)
    assert not problem.use_split_k and not degraded
    # A split that the source did not make, or one that the target cannot keep.
    for spk, source_k in ((2048, 2048), (4096, 2048), (300, 4096)):
        _, cfg = selector_target_and_config(spk)
        problem, degraded = kernels_tool.build_problem(target, cfg, source_k)
        assert not problem.use_split_k
        assert degraded == (spk < source_k)
    _, cfg = selector_target_and_config(1024)
    problem, degraded = kernels_tool.build_problem(target, cfg, 4096)
    assert problem.use_split_k and problem.spk == 1024 and not degraded


def test_selector_validation_settings_floor():
    def settings_document(iters, warmup):
        settings = {"iters": iters, "warmup": warmup, "verify_full_max_elems": 100}
        return selector_document({**COMPLETE_META, "settings": settings}, [])

    small = settings_document(5, 2)
    assert from_sweep.validation_settings(small, None, None) == (30, 10, 100)
    assert from_sweep.validation_settings(small, 3, 0) == (3, 0, 100)
    large = settings_document(50, 20)
    assert from_sweep.validation_settings(large, None, None) == (50, 20, 100)
    missing = selector_document(COMPLETE_META, [])
    with pytest.raises(ValueError, match="settings.iters"):
        from_sweep.validation_settings(missing, None, None)


def selector_case():
    return selection.SweepCase(0, selector_problem(), None, [])


def test_correctness_errors_reports_assertion_as_value_error():
    case = selector_case()
    original = AssertionError("D mismatch")
    with pytest.raises(ValueError) as info:
        with from_sweep.correctness_errors(case, "kernelX"):
            raise original
    assert case.text() in str(info.value)
    assert "kernelX" in str(info.value)
    assert info.value.__cause__ is original


def test_correctness_errors_passes_other_exceptions_through():
    case = selector_case()
    with pytest.raises(KeyError):
        with from_sweep.correctness_errors(case, "kernelX"):
            raise KeyError("x")
    with from_sweep.correctness_errors(case, "kernelX"):
        pass


# The shared selection module: loading.


def test_read_document_checks_the_shape(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="not a sweep results document"):
        selection.read_document(str(path))
    path.write_text("meta: 1\nresults: []\n")
    with pytest.raises(ValueError, match="meta is not a mapping"):
        selection.read_document(str(path))
    path.write_text("results: []\n")
    with pytest.raises(ValueError, match="meta is not a mapping"):
        selection.read_document(str(path))
    good = write_selector_results(tmp_path, COMPLETE_META, [])
    assert selection.read_document(good)["results"] == []


def test_load_cases_requires_one_shot_only_when_asked():
    meta = {"casesDone": 1, "casesTotal": 1}
    results = [selector_entry([top_item(1, 2.0)])]
    with pytest.raises(ValueError, match="--one-shot"):
        load(results, meta)
    assert len(load(results, meta, require_one_shot=False)) == 1
    with pytest.raises(ValueError, match="--one-shot"):
        load(results, {**meta, "oneShot": False})


@pytest.mark.parametrize("flag", ["interrupted", "aborted"])
@pytest.mark.parametrize("require_one_shot", [True, False])
def test_load_cases_rejects_an_interrupted_or_aborted_sweep(flag, require_one_shot):
    with pytest.raises(ValueError, match="interrupted or aborted"):
        load([], {**COMPLETE_META, flag: True}, require_one_shot=require_one_shot)


@pytest.mark.parametrize(
    "meta",
    [
        {"oneShot": True},
        {"oneShot": True, "casesDone": 1},
        {"oneShot": True, "casesDone": 1, "casesTotal": 2},
    ],
)
def test_load_cases_rejects_incomplete_results(meta):
    with pytest.raises(ValueError, match=r"results are incomplete \(.* of .* cases"):
        load([], meta)


def test_load_cases_names_the_malformed_entry():
    good = selector_entry([top_item(1, 2.0)])
    with pytest.raises(ValueError, match=r"results\[1\]: malformed entry"):
        load([good, {"top": []}])
    with pytest.raises(ValueError, match=r"results\[1\]: malformed entry"):
        load([good, "text"])
    bad_item = {"rank": 1, "config": sweep_config_record()}
    with pytest.raises(ValueError, match=r"results\[0\]: malformed entry.*timeUs"):
        load([selector_entry([bad_item])])
    problem = selector_problem()
    del problem["useBeta"]
    with pytest.raises(ValueError, match=r"results\[0\]: malformed entry"):
        load([{"problem": problem, "top": []}])


@pytest.mark.parametrize("top", [[1], "abc", [{"rank": 1, "timeUs": 1.0, "config": None}]])
@pytest.mark.parametrize("use_beta_only", [True, False])
def test_load_cases_names_a_malformed_top(top, use_beta_only):
    results = [selector_entry([top_item(1, 1.0)]), {"problem": selector_problem(), "top": top}]
    with pytest.raises(ValueError, match=r"results\[1\]: malformed entry"):
        load(results, use_beta_only=use_beta_only)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1.5, "1.0", True, None])
def test_load_cases_rejects_a_bad_time(bad):
    results = [selector_entry([top_item(1, bad)], name="named")]
    with pytest.raises(ValueError, match=r"results\[0\] \(named\): timeUs"):
        load(results)


def test_load_cases_drops_unverified_and_failed_timings():
    top = [
        top_item(1, 1.0, verified="failed"),
        top_item(2, 2.0, verified="no"),
        top_item(3, 3.0, verified="full"),
        top_item(4, 4.0, verified="sampled(5)"),
        top_item(5, 5.0),
    ]
    (case,) = load([selector_entry(top)])
    assert [item["rank"] for item in case.top] == [3, 4, 5]
    assert case.best_us == 3.0 and case.reason is None


def test_load_cases_keeps_a_case_without_timings_with_the_reason():
    results = [
        selector_entry([]),
        selector_entry([top_item(1, 1.0, verified="no")]),
        {"problem": selector_problem(), "top": None},
    ]
    for case in load(results):
        assert case.best_us is None and case.top == []
        assert case.reason == selection.REASON_NO_TIMINGS_USE_BETA
    (case,) = load(results[:1], use_beta_only=False)
    assert case.reason == selection.REASON_NO_TIMINGS
    assert selection.REASON_NO_TIMINGS == "no verified configurations in the sweep top"
    assert selection.REASON_NO_TIMINGS_USE_BETA == (
        "no verified USE_BETA1 configurations in the sweep top"
    )
    assert load(results)[1].index == 1


def test_load_cases_strips_the_launch_lists_and_sorts_by_time_then_rank():
    slow = {**top_item(1, 3.0), "gemmNs": [3000], "reduceNs": [1]}
    tie_b = {**top_item(3, 2.0), "screeningGemmNs": [1], "screeningReduceNs": [1]}
    tie_a = top_item(2, 2.0)
    (case,) = load([selector_entry([slow, tie_b, tie_a])])
    assert [item["rank"] for item in case.top] == [2, 3, 1]
    assert case.best_us == 2.0
    assert not any(key in item for item in case.top for key in results_module.LAUNCH_LIST_KEYS)


def test_load_cases_use_beta_only_filters_problems_and_entries():
    results = [
        selector_entry([top_item(1, 1.0, use_beta=False), top_item(2, 2.0)]),
        selector_entry([top_item(1, 1.0)], useBeta=False, name="nobeta"),
    ]
    (case,) = load(results)
    assert [item["rank"] for item in case.top] == [2]
    both = load(results, use_beta_only=False)
    assert [len(c.top) for c in both] == [2, 1]


def test_load_cases_only_and_limit_number_the_kept_cases():
    results = [selector_entry([top_item(1, 1.0)], name=n) for n in ("ab", "b", "bc", "x")]
    cases = load(results, only="b", limit=2)
    assert [(c.index, c.problem["name"]) for c in cases] == [(0, "ab"), (1, "b")]
    assert load(results, limit=0) == []


def bench_args(*argv):
    return bench.makeParser().parse_args(list(argv))


@pytest.mark.parametrize("spk", [None, 256])
def test_bench_config_round_trips_a_selection_config(spk):
    identity = selection.identity_of(sweep_config_record(spk=spk))
    text = yaml.safe_dump(selection.identity_record(identity))
    given = ["--spk", "256"] if spk else []
    overrides = bench.knobOverrides(bench_args("--config", text, *given))
    assert overrides["bm"] == 1 and overrides["use_beta"] is True
    assert overrides.get("spk") == spk
    knobs = {name: overrides.get(name) for name in support.KNOB_OVERRIDE_NAMES}
    knobs["mfma"] = bool(knobs["mfma"])
    problem = support.makeProblem(
        64, 64, 512, padding=(0, 0, 0), use_beta=overrides["use_beta"], **knobs
    )
    assert problem.use_split_k == (spk is not None)


def test_bench_config_split_k_needs_the_matching_spk():
    split = yaml.safe_dump(selection.identity_record(
        selection.identity_of(sweep_config_record(spk=256))))
    plain = yaml.safe_dump(selection.identity_record(
        selection.identity_of(sweep_config_record(spk=None))))
    with pytest.raises(SystemExit, match="use_split_k true needs --spk"):
        bench.knobOverrides(bench_args("--config", split))
    with pytest.raises(SystemExit, match="without split-K; drop --spk"):
        bench.knobOverrides(bench_args("--config", plain, "--spk", "256"))
    # A mapping without the key keeps the old behaviour.
    assert bench.knobOverrides(bench_args("--config", "{bm: 2}", "--spk", "256"))["spk"] == 256


def test_selector_loaders_accept_entries_with_and_without_launch_lists():
    with_lists = {
        **top_item(1, 2.0, kernel="k0"),
        "gemmNs": [2000, 2000], "reduceNs": [1, 2], "screeningGemmNs": [1],
        "screeningReduceNs": [1],
    }
    without = top_item(2, 3.0, kernel="k1")
    for flags in (
        {"require_one_shot": True, "use_beta_only": True},
        {"require_one_shot": False, "use_beta_only": False},
    ):
        (case,) = load([selector_entry([with_lists, without])], **flags)
        assert [t["kernel"] for t in case.top] == ["k0", "k1"]
        assert not any(key in t for t in case.top for key in results_module.LAUNCH_LIST_KEYS)


@pytest.mark.parametrize("parse", [selection.non_negative_finite_float, selection.positive_finite_float])
@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "-1"])
def test_selection_float_types_reject_non_finite_and_negative(parse, text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse(text)


def test_selection_argument_types_bounds():
    assert selection.non_negative_finite_float("0") == 0.0
    assert selection.positive_finite_float("0.5") == 0.5
    with pytest.raises(argparse.ArgumentTypeError):
        selection.positive_finite_float("0")
    assert selection.positive_int("1") == 1 and selection.non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        selection.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        selection.non_negative_int("-1")


# The shared selection module: identities.


def test_identity_of_tells_the_split_k_kernel_from_the_one_without():
    plain = selection.identity_of(sweep_config_record(spk=None))
    split = selection.identity_of(sweep_config_record(spk=256))
    other_split = selection.identity_of(sweep_config_record(spk=512))
    assert not plain.use_split_k and split.use_split_k
    assert plain != split
    # The split size is re-derived per target: it is not part of the identity.
    assert split == other_split
    assert selection.identity_of(sweep_config_record(use_beta=False)).use_beta is False
    assert selection.identity_of(sweep_config_record()).use_beta is True
    assert selection.identity_of({k: v for k, v in sweep_config_record().items() if k != "useBeta"}).use_beta is False


def test_identity_tie_break_and_record():
    identity = selection.identity_of(sweep_config_record(grid3d=False, blocks=None, spk=64))
    key = selection.tie_break_key(identity)
    assert key[identity._fields.index("blocks")] == -1
    assert key[identity._fields.index("use_split_k")] == 1
    record = selection.identity_record(identity)
    assert list(record) == list(identity._fields) and record["grid3d"] is False


def fake_entry(time_us, rank=1):
    return types.SimpleNamespace(time_us=time_us, rank=rank)


def test_assign_disjoint_prefers_the_fastest_library_qualifier():
    a, b, c = (selection.identity_of(sweep_config_record(bm=bm)) for bm in (1, 2, 3))
    qualified = {
        0: {a: fake_entry(1.0), b: fake_entry(1.1)},
        1: {a: fake_entry(1.0), b: fake_entry(1.1)},
        2: {c: fake_entry(1.0)},
        3: {a: fake_entry(1.0), b: fake_entry(1.0)},
        4: {},
    }
    covers = selection.assign_disjoint(qualified, {b, a})
    assert {i: identity for i, (identity, _) in covers.items()} == {0: a, 1: a, 3: a}
    covers = selection.assign_disjoint(qualified, {b})
    assert {i: identity for i, (identity, _) in covers.items()} == {0: b, 1: b, 3: b}
    # Equal times: the identity with the least tie break key.
    assert selection.fastest_winner(qualified[3])[0] == a


# The library selection of select_skinny_gemm_from_sweep.


def knob_case_results(rows):
    """One result per row; a row is a list of (bm, time) pairs, ranked by time."""
    results = []
    for index, row in enumerate(rows):
        top = [
            top_item(rank, time_us, kernel=f"bm{bm}_USE_BETA1", bm=bm)
            for rank, (bm, time_us) in enumerate(row, 1)
        ]
        results.append(selector_entry(top, name=f"p{index}"))
    return results


SELECTION_ROWS = [
    [(1, 1.0), (2, 1.05), (3, 1.5)],
    [(2, 1.0), (1, 1.08), (4, 1.02)],
    [(3, 1.0), (4, 1.03), (1, 1.04)],
    [(4, 1.0), (3, 1.09), (2, 1.2)],
    [(1, 2.0), (4, 2.1), (3, 2.15)],
    [(2, 3.0), (3, 3.1), (4, 3.2)],
]


def run_selection_on(results, max_configs=2, tolerance=0.1):
    meta = {**COMPLETE_META, "casesTotal": 1, "casesDone": 1}
    cases = load(results, meta)
    result = from_sweep.select_library(cases, tolerance, max_configs, 0)
    named = {cases[i].problem["name"]: cover.identity for i, cover in result.covers.items()}
    return result.library, named


def test_select_library_is_deterministic_under_shuffling():
    expected = run_selection_on(knob_case_results(SELECTION_ROWS))
    assert expected[1]
    for seed in range(8):
        rng = random.Random(seed)
        results = knob_case_results(SELECTION_ROWS)
        for result in results:
            rng.shuffle(result["top"])
        rng.shuffle(results)
        assert run_selection_on(results) == expected


def test_select_library_breaks_ties_with_the_tie_break_key():
    for order in ([(2, 1.0), (1, 1.0)], [(1, 1.0), (2, 1.0)]):
        results = [selector_entry(
            [top_item(1, time_us, bm=bm) for bm, time_us in order], name="p")]
        library, named = run_selection_on(results, max_configs=1)
        assert named["p"].bm == 1 and [i.bm for i in library] == [1]


def test_select_library_tolerance_edge():
    results = [selector_entry([top_item(1, 10.0, bm=1), top_item(2, 15.0, bm=2),
                               top_item(3, 15.000001, bm=3)])]
    (case,) = load(results)
    qualified = from_sweep.qualified_per_case(case, 0.5)
    assert sorted(identity.bm for identity in qualified) == [1, 2]
    assert 15.0 == qualified[selection.identity_of(sweep_config_record(bm=2))].time_us


def test_select_library_respects_max_configs():
    rows = [[(bm, 1.0)] for bm in range(1, 6)]
    for max_configs in (1, 2, 3):
        library, named = run_selection_on(knob_case_results(rows), max_configs)
        assert len(library) == max_configs == len(named)


def test_select_library_leaves_cases_without_timings_uncovered():
    cases = load([selector_entry([]), selector_entry([top_item(1, 1.0)], name="q")])
    result = from_sweep.select_library(cases, 0.1, 1, 0)
    assert list(result.covers) == [1]
    record = from_sweep.uncovered_record(cases[0], {}, set(result.library), 0.1)
    assert record["reason"] == selection.REASON_NO_TIMINGS_USE_BETA
    assert record["bestUs"] is None


def test_uncovered_record_reports_the_library_time_beyond_the_tolerance():
    cases = load(knob_case_results([[(1, 1.0), (2, 3.0)]]))
    slow = selection.identity_of(sweep_config_record(bm=2))
    qualified = from_sweep.qualified_per_case(cases[0], 0.1)
    record = from_sweep.uncovered_record(cases[0], qualified, {slow}, 0.1)
    assert record["reason"] == "sweep best not in library; no library config within tolerance"
    assert record["bestInLibrary"]["timeUs"] == 3.0
    assert record["bestInLibrary"]["ratio"] == 3.0
    assert record["bestInLibrary"]["withinTolerance"] is False
    assert record["bestInLibrary"]["config"]["bm"] == 2
    assert record["sweepBest"]["timeUs"] == 1.0


def test_uncovered_record_without_a_library_entry_has_no_library_time():
    cases = load(knob_case_results([[(1, 1.0)]]))
    qualified = from_sweep.qualified_per_case(cases[0], 0.1)
    record = from_sweep.uncovered_record(cases[0], qualified, set(), 0.1)
    assert record["bestInLibrary"] is None


def test_representatives_keep_the_best_ranked_config():
    cases = load(knob_case_results([[(1, 1.0)], [(1, 1.0)]]))
    cases[1].top[0]["rank"] = 0
    qualified = from_sweep.build_qualified(cases, 0.1)
    (identity,) = qualified[0]
    reps = from_sweep.representatives(qualified, {identity})
    assert reps == {identity: cases[1].top[0]["config"]}
    assert not any("_rank" in config for config in reps.values())


def fake_selection(entries_by_case):
    """A selection whose library is one identity covering the given kernels."""
    identity = selection.identity_of(sweep_config_record())
    covers = {
        index: from_sweep.Cover(
            identity, from_sweep.QualifiedEntry(1.0, sweep_config_record(), kernel, 1)
        )
        for index, kernel in entries_by_case.items()
    }
    return from_sweep.Selection([identity], collections.Counter(), covers, {}, {})


def test_check_library_kernels():
    from_sweep.check_library_kernels(fake_selection({0: "a_USE_BETA1", 1: "a_USE_BETA1"}))
    with pytest.raises(ValueError, match="has 2 kernels, expected 1"):
        from_sweep.check_library_kernels(fake_selection({0: "a_USE_BETA1", 1: "b_USE_BETA1"}))
    with pytest.raises(ValueError, match="not USE_BETA1"):
        from_sweep.check_library_kernels(fake_selection({0: "a_USE_BETA0"}))


def selector_main(monkeypatch, tmp_path, results, meta=None):
    meta = meta or {
        **COMPLETE_META,
        "casesTotal": 1,
        "settings": {"iters": 5, "warmup": 1, "verify_full_max_elems": 10},
    }
    path = write_selector_results(tmp_path, meta, results)
    argv = ["select", path, "--kernels", str(tmp_path / "k"), "-o", str(tmp_path / "s.yaml"),
            "--max-configs", "1"]
    monkeypatch.setattr("sys.argv", argv)
    return from_sweep.main()


def test_selector_fails_before_emission_when_a_config_has_two_kernels(monkeypatch, tmp_path, capsys):
    def emit(*args):
        raise AssertionError("emission must not run")

    monkeypatch.setattr(from_sweep, "emit_kernels", emit)
    results = [
        selector_entry([top_item(1, 1.0, kernel="one_USE_BETA1")], name="p0"),
        selector_entry([top_item(1, 1.0, kernel="two_USE_BETA1")], name="p1"),
    ]
    assert selector_main(monkeypatch, tmp_path, results) == 1
    assert "has 2 kernels, expected 1" in capsys.readouterr().err
    assert not (tmp_path / "k").exists() and not (tmp_path / "s.yaml").exists()


def test_selector_fails_before_emission_for_a_use_beta0_kernel(monkeypatch, tmp_path, capsys):
    results = [selector_entry([top_item(1, 1.0, kernel="one_USE_BETA0")])]
    assert selector_main(monkeypatch, tmp_path, results) == 1
    assert "not USE_BETA1" in capsys.readouterr().err


def test_selector_errors_when_no_case_has_verified_timings(monkeypatch, tmp_path, capsys):
    results = [selector_entry([top_item(1, 1.0, verified="no")])]
    assert selector_main(monkeypatch, tmp_path, results) == 2
    assert "no case has verified timings (was the sweep run with --no-verify?)" in (
        capsys.readouterr().err
    )


def test_selector_reports_a_results_error(monkeypatch, tmp_path, capsys):
    assert selector_main(monkeypatch, tmp_path, [], {"oneShot": True}) == 2
    assert "error: results are incomplete" in capsys.readouterr().err


# The greedy selection of select_skinny_gemm_kernels.


def kernels_selector(cases, tolerance=0.1, max_configs=None):
    return kernels_tool.Selector(
        cases, tolerance, 1, 0, 0.0, False, None, None, 1, max_configs
    )


def kernels_case(index, best_us, top_bms, reason=None):
    top = [top_item(rank, best_us * (1 + 0.01 * rank), bm=bm) for rank, bm in enumerate(top_bms, 1)]
    return selection.SweepCase(
        index, selector_problem() | {"name": f"p{index}"}, best_us if top else None, top, reason
    )


def bm_identity(bm):
    return selection.identity_of(sweep_config_record(bm=bm))


def test_drop_dominated_needs_the_measured_config_to_cover_the_case():
    case = kernels_case(0, 1.0, [1, 2, 3])
    selector = kernels_selector([case])
    selector.candidates[0][bm_identity(3)] = selector.candidates[0][bm_identity(3)]._replace(time_us=5.0)
    # Beyond the tolerance of the best: nothing is dropped, though 2.0 is within
    # the tolerance of the slowest candidate.
    selector.drop_dominated(0, bm_identity(9), 2.0)
    assert len(selector.candidates[0]) == 3
    # Within the tolerance: the candidates it is within tolerance of go.
    selector.protected[0].add(bm_identity(2))
    selector.drop_dominated(0, bm_identity(9), 1.05)
    assert set(selector.candidates[0]) == {bm_identity(2)}


def test_cases_without_timings_are_never_tallied_or_searched():
    cases = [kernels_case(0, 0.0, [], "no timings"), kernels_case(1, 1.0, [1])]
    selector = kernels_selector(cases)
    assert selector.uncovered() == [1]
    entries = kernels_tool.tally(cases, selector.uncovered(), selector.candidates, selector.tried)
    assert [e.frequency for e in entries] == [1]


def patch_rounds(monkeypatch):
    rounds = []

    def run_round(self, identity, case_indices):
        rounds.append((identity, case_indices))
        for i in case_indices:
            self.tried[i].add(identity)
        return 0

    monkeypatch.setattr(kernels_tool.Selector, "run_round", run_round)
    return rounds


def test_stop_reason_all_cases_covered(monkeypatch):
    rounds = patch_rounds(monkeypatch)
    result = kernels_selector([kernels_case(0, 0.0, [], "none")]).run()
    assert result.stop_reason == kernels_tool.REASON_ALL_COVERED == "all cases covered"
    assert rounds == []


def test_stop_reason_exhausted_and_round_limit(monkeypatch):
    patch_rounds(monkeypatch)
    cases = [kernels_case(0, 1.0, [1, 2])]
    result = kernels_selector(cases).run()
    assert result.stop_reason == kernels_tool.REASON_EXHAUSTED == "no untried candidates left"
    monkeypatch.setattr(kernels_tool.Selector, "round_limit", lambda self: 0)
    result = kernels_selector(cases).run()
    assert result.stop_reason == kernels_tool.REASON_ROUND_LIMIT == "round limit reached"


def planned_cover(selector, index, bm, time_us):
    """Records a within-tolerance measurement of bm for case index."""
    target, cfg = selector_target_and_config(None)
    cfg["bm"] = bm
    problem, _ = kernels_tool.build_problem(target, cfg, 2048)
    plan = kernels_tool.Plan(index, problem, False, None)
    selector.measured_best[index] = (bm_identity(bm), time_us)
    selector.record_within(index, bm_identity(bm), plan, time_us)


def test_uncovered_reasons_tell_the_cap_from_the_search_end(monkeypatch):
    cases = [
        kernels_case(0, 0.0, [], "loader reason"),
        kernels_case(1, 1.0, [1]),
        kernels_case(2, 1.0, [1]),
        kernels_case(3, 1.0, [2]),
        kernels_case(4, 1.0, [2]),
    ]
    selector = kernels_selector(cases, max_configs=1)
    planned_cover(selector, 1, 1, 1.0)
    planned_cover(selector, 2, 1, 1.0)
    planned_cover(selector, 3, 2, 1.0)
    # Measured, but beyond the tolerance.
    selector.measured_best[4] = (bm_identity(2), 3.0)
    selector.rebuild_assignment()
    result = selector.result(kernels_tool.REASON_EXHAUSTED)
    assert set(result.covers) == {1, 2}
    reasons = {c.index: kernels_tool.uncovered_reason(c, result) for c in cases[:1] + cases[3:]}
    assert reasons == {
        0: "loader reason",
        3: kernels_tool.REASON_CAP,
        4: kernels_tool.REASON_EXHAUSTED,
    }
    assert kernels_tool.REASON_CAP == (
        "within tolerance, but its configurations were cut by "
        "--max-configs/--min-config-cases"
    )
    unmeasured = kernels_case(5, 1.0, [1])
    assert kernels_tool.uncovered_reason(unmeasured, result) == kernels_tool.REASON_UNMEASURED
    record = kernels_tool.uncovered_record(cases[0], result, 0.1)
    assert record["reason"] == "loader reason" and record["bestUs"] is None
    assert record["measuredBest"] is None
    record = kernels_tool.uncovered_record(cases[4], result, 0.1)
    assert record["measuredBest"]["withinTolerance"] is False


def test_kernels_selector_argument_types():
    parser = kernels_tool.make_parser()
    base = ["results.yaml", "-o", "out"]
    assert parser.parse_args(base).max_configs is None
    for flag in ("--tolerance=nan", "--tolerance=-1", "--max-configs=0", "--cache-gb=0",
                 "--cache-gb=inf", "--warmup-ms=nan", "--jobs=0", "--limit-cases=-1"):
        with pytest.raises(SystemExit):
            parser.parse_args(base + [flag])
    args = parser.parse_args(base + ["--limit-cases=0", "--warmup-ms=0", "--tolerance=0"])
    assert (args.limit_cases, args.warmup_ms, args.tolerance) == (0, 0.0, 0.0)


def test_selection_modules_do_not_initialize_hip():
    code = (
        "import os; os.environ.pop('HIP_VISIBLE_DEVICES', None)\n"
        "import select_skinny_gemm_from_sweep, select_skinny_gemm_kernels\n"
        "import skinny_gemm_selection\n"
        "import glob\n"
        "fds = [os.path.realpath(p) for p in glob.glob('/proc/self/fd/*')]\n"
        "assert not any('kfd' in path for path in fds), fds\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=HERE)


ROUND_TRIP_INPUT = """\
settings: {iters: 10, warmup: 2, refine_top: 8, refine_iters: 30, top: 5}
defaults: {useBeta: false}
problems:
  - name: %s
    m: 1
    n: 512
    k: 4096
    sweep:
      - bm: [1, 2]
        tn: 1
        lanesK: 64
        waves: 4
        prefetch: 2
        upScales: 0
        grid3d: true
"""


def temporaries(directory):
    return [name for name in os.listdir(directory) if name.endswith(".tmp")]


def make_failure_group(problem, config_rec):
    failure = {
        "config": config_rec,
        "status": "failed to build",
        "stage": "assemble",
        "reason": "r",
    }
    return {"problem": problem, "failures": [failure]}


def test_results_round_trip_and_atomicity(tmp_path):
    path = str(tmp_path / "r.yaml")
    document = {"meta": {"casesDone": 1}, "results": [{"a": [1, 2], "b": "x"}]}
    results_module.write_results(path, document, results_module.FORMAT_YAML)
    assert temporaries(tmp_path) == []
    loaded = results_module.read_results(path)
    assert loaded["results"] == document["results"]
    assert loaded["meta"]["casesDone"] == 1
    with open(path, "w") as stream:
        stream.write("meta: {a: [1, 2\nresults: ]]")
    with pytest.raises(yaml.YAMLError):
        results_module.read_results(path)


def test_read_failures_edge_cases(tmp_path):
    path = str(tmp_path / "r.yaml")
    assert results_module.read_failures(path) == []
    failed = results_module.failed_path(path)
    assert failed == str(tmp_path / "r.failed.yaml")
    open(failed, "w").close()
    assert results_module.read_failures(path) == []
    with open(failed, "w") as stream:
        for name in ("a", "b"):
            results_module.dump_block([make_failure_group({"name": name}, {})], stream)
    groups = results_module.read_failures(path)
    assert [g["problem"]["name"] for g in groups] == ["a", "b"]
    with open(failed, "w") as stream:
        stream.write("failures: []\n")
    with pytest.raises(ValueError, match="list of failure groups"):
        results_module.read_failures(path)


def test_result_writer_appends_groups_and_throttles(tmp_path, monkeypatch):
    path = str(tmp_path / "r.yaml")
    calls = []
    monkeypatch.setattr(sweep, "write_results", lambda *args: calls.append(args))
    writer = sweep.ResultWriter(path, {}, "yaml", interval_s=3600)
    failures = [{"config": {"bm": 1}, "status": "failed to build"}]
    writer.add(0, {"problem": {"name": "a"}}, failures)
    writer.add(1, {"problem": {"name": "b"}}, failures)
    assert len(calls) == 1
    writer.write(force=True)
    assert len(calls) == 2
    assert len(calls[1][1]["results"]) == 2
    groups = results_module.read_failures(path)
    assert [g["problem"]["name"] for g in groups] == ["a", "b"]
    assert all(g["failures"] == failures for g in groups)


def write_round_trip_case(tmp_path, name):
    input_path = tmp_path / "input.yaml"
    input_path.write_text(ROUND_TRIP_INPUT % name)
    sweep_input = config.load_input(str(input_path), 7)
    expanded = config.expand_case(sweep_input.cases[0])
    records = [config.config_record(p) for p in expanded.problems]
    assert len(records) == 2
    problem = sweep_input.cases[0].record()
    meta = {"inputFile": str(input_path), "settings": {"seed": 7}}
    top = [{"rank": 1, "config": records[0]}]
    path = str(tmp_path / "r.yaml")
    document = {"meta": meta, "results": [{"problem": problem, "top": top}]}
    results_module.write_results(path, document, results_module.FORMAT_YAML)
    return path, problem, records


def analyzed(tmp_path, group_name):
    path, problem, records = write_round_trip_case(tmp_path, "case")
    failed_problem = {**problem, "name": group_name}
    group = make_failure_group(failed_problem, records[1])
    with open(results_module.failed_path(path), "w") as stream:
        results_module.dump_block([group], stream)
    meta, _, _ = analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    return meta


def test_analyzer_excludes_build_failure_of_the_same_case(tmp_path):
    meta = analyzed(tmp_path, "case")
    assert meta["buildFailuresExcluded"] == 1
    assert meta["modes"]["grid"]["candidates"] == 1
    assert meta["modes"]["grid"]["tops"] == 1


def test_analyzer_keeps_build_failure_of_another_case(tmp_path):
    meta = analyzed(tmp_path, "other")
    assert meta["buildFailuresExcluded"] == 0
    assert meta["modes"]["grid"]["candidates"] == 2


def test_analyzer_without_failed_file(tmp_path):
    path, _, _ = write_round_trip_case(tmp_path, "case")
    meta, _, _ = analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    assert meta["buildFailuresExcluded"] == 0
    assert meta["modes"]["grid"]["tops"] == 1


TWO_PROBLEM_INPUT = """\
settings: {iters: 10, warmup: 2, refine_top: 8, refine_iters: 30, top: 5}
defaults: {useBeta: false}
problems:
%s
"""
PROBLEM_TEXT = """\
  - name: %s
    m: 1
    n: 512
    k: 4096
    sweep:
      - bm: [1, 2]
        tn: 1
        lanesK: 64
        waves: 4
        prefetch: 2
        upScales: 0
        grid3d: true
"""


def write_two_problem_case(tmp_path, meta_extra=None, results_names=("a",)):
    """An input of the problems a and b, results for results_names only."""
    input_path = tmp_path / "input.yaml"
    input_path.write_text(TWO_PROBLEM_INPUT % "".join(PROBLEM_TEXT % n for n in "ab"))
    sweep_input = config.load_input(str(input_path), 7)
    meta = {"inputFile": str(input_path), "settings": {"seed": 7}, **(meta_extra or {})}
    results = []
    for case in sweep_input.cases:
        if case.name not in results_names:
            continue
        record = config.config_record(config.expand_case(case).problems[0])
        results.append({"problem": case.record(), "top": [{"rank": 1, "config": record}]})
    path = str(tmp_path / "r.yaml")
    results_module.write_results(path, {"meta": meta, "results": results}, results_module.FORMAT_YAML)
    return path


def test_analyzer_skips_cases_without_a_results_record(tmp_path, capsys):
    path = write_two_problem_case(tmp_path)
    meta, _, _ = analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    assert meta["casesMissing"] == 1 and meta["casesAnalyzed"] == 1
    # The skipped case is not counted as all-non-top: its two configs are absent.
    assert meta["modes"]["grid"] == {"candidates": 2, "tops": 1, "topRate": 0.5}
    analyze.print_summary({"meta": meta, "modes": {"grid": {}, "persistent": {}}})
    assert "Skipped 1 cases without a results record" in capsys.readouterr().out


def test_analyzer_does_not_print_missing_cases_when_there_are_none(tmp_path, capsys):
    path = write_two_problem_case(tmp_path, results_names=("a", "b"))
    meta, _, _ = analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    assert meta["casesMissing"] == 0
    analyze.print_summary({"meta": meta, "modes": {"grid": {}, "persistent": {}}})
    assert "Skipped" not in capsys.readouterr().out


def test_analyzer_refuses_one_shot_results(tmp_path, capsys):
    path = write_two_problem_case(tmp_path, {"oneShot": True})
    with pytest.raises(ValueError, match="top membership carries no signal"):
        analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    assert analyze.main([path, "--no-progress"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: one-shot results list every screened configuration")
    assert "analyze a refine+top sweep" in err


def test_analyzer_ignores_failed_top_entries(tmp_path):
    path, problem, records = write_round_trip_case(tmp_path, "case")
    top = [
        {"rank": 1, "config": records[0], "verified": "failed"},
        {"rank": 2, "config": records[1], "verified": "full"},
    ]
    document = results_module.read_results(path)
    document["results"][0]["top"] = top
    results_module.write_results(path, document, results_module.FORMAT_YAML)
    assert analyze.top_identities_by_case(document["results"]) == {
        analyze.case_key(problem): {analyze.identity_of(records[1])}
    }


def knob_rows(ms, knob, rows):
    for value, candidates, tops, phi, p_value in rows:
        ms.value_stats[(knob, value)] = analyze.KnobValueStats(
            knob, value, candidates=candidates, tops=tops, phi=phi, p_value_vs_rest=p_value
        )


def prune_bm(ms, swept, policy="frequency", **overrides):
    params = dict(
        mode="grid", policy=policy, min_trials=0, max_phi=0.01, p_threshold=0.05,
        swept_values={"bm": swept}, keep_at_least=1, min_top_slots=1,
        min_top_share=0.0, min_top_relative=0.0,
    ) | overrides
    return analyze.prune_mode(ms, **params)


def test_frequency_prune_drops_a_value_on_weak_phi_alone():
    ms = analyze.ModeState(global_tops=100, global_candidates=1000)
    # The mode baseline top rate is 0.1. Every value passes the frequency
    # thresholds; only the second has a weak phi, a weak p-value and a low rate.
    knob_rows(ms, "bm", [
        (1, 400, 60, 0.3, 0.0),
        (2, 400, 30, 0.0, 0.5),
        (3, 200, 40, 0.3, 0.0),
    ])
    assert prune_bm(ms, {1, 2, 3}) == {"bm": [2]}
    # The same row with a strong phi stays.
    ms.value_stats[("bm", 2)].phi = 0.5
    assert prune_bm(ms, {1, 2, 3}) == {}


def test_prune_ranking_is_deterministic_with_ties():
    results = set()
    for swept in ({8, 2}, {2, 8}, set([8, 2, 4])):
        ms = analyze.ModeState(global_tops=10, global_candidates=100)
        knob_rows(ms, "bm", [(2, 10, 0, 0.0, 1.0), (8, 10, 0, 0.0, 1.0), (4, 10, 5, 0.3, 0.0)])
        results.add(repr(prune_bm(ms, swept, policy="never-top")))
    assert results == {repr({"bm": [8]}), repr({"bm": [2, 8]})}
    # With 4 not swept, the tie between 2 and 8 keeps 2 (the least str).
    ms = analyze.ModeState(global_tops=10, global_candidates=100)
    knob_rows(ms, "bm", [(2, 10, 0, 0.0, 1.0), (8, 10, 0, 0.0, 1.0)])
    assert prune_bm(ms, {8, 2}, policy="never-top") == {"bm": [8]}
    assert [r.value for r in analyze.rows_for_swept_knob(ms, "bm", {8, 2, 16})] == [16, 2, 8]


def sweep_input_raw():
    return {
        "defaults": {"sweep": [
            {"grid3d": True, "bm": [1, 2], "tn": 1},
            {"grid3d": False, "bm": [1, 2, 3], "group": 4, "blocks": [64, 256]},
        ]},
        "problems": [],
    }


def test_apply_prune_never_empties_a_list_or_removes_a_scalar(capsys):
    prune = {"grid": {"bm": [1, 2], "tn": [1]}, "persistent": {"bm": [1], "group": [4]}}
    out = analyze.apply_prune_to_input(sweep_input_raw(), prune)
    grid, persistent = out["defaults"]["sweep"]
    assert grid["bm"] == [1, 2] and grid["tn"] == 1
    assert persistent["bm"] == [2, 3] and persistent["group"] == 4
    err = capsys.readouterr().err
    assert "keeping bm: [1, 2] of the grid sweep mapping" in err
    assert "keeping tn: 1 of the grid sweep mapping" in err
    assert "keeping group: 4 of the persistent sweep mapping" in err
    assert "persistent sweep mapping: every value" not in err
    # The input is not modified.
    assert sweep_input_raw()["defaults"]["sweep"][0]["bm"] == [1, 2]


def test_requested_blocks_maps_a_capped_count_to_the_smallest_larger_swept_value():
    swept = {64, 256, 1024}
    assert analyze.requested_blocks(64, swept) == 64
    assert analyze.requested_blocks(1024, swept) == 1024
    assert analyze.requested_blocks(100, swept) == 256
    assert analyze.requested_blocks(257, swept) == 1024
    assert analyze.requested_blocks(5000, swept) == 5000
    assert analyze.requested_blocks(3, set()) == 3
    knobs = {"blocks": 100, "bm": 1}
    assert analyze.attribute_blocks(knobs, swept)["blocks"] == 256
    assert analyze.attribute_blocks({"blocks": None}, swept) == {"blocks": None}


def test_analyzer_attributes_a_capped_block_count_to_its_swept_value(tmp_path):
    input_path = tmp_path / "input.yaml"
    input_path.write_text(
        "problems:\n"
        "  - {name: p, m: 1, n: 64, k: 256, useBeta: false, sweep: [{bm: 1, tn: 1, "
        "lanesK: 64, waves: 4, prefetch: 2, upScales: 0, grid3d: false, group: 1, "
        "numXcds: 1, blocks: [1, 1000000, 2000000]}]}\n"
    )
    sweep_input = config.load_input(str(input_path), 7)
    problems = config.expand_case(sweep_input.cases[0]).problems
    recorded = sorted(p.blocks for p in problems)
    assert recorded[-1] < 1000000
    meta = {"inputFile": str(input_path), "settings": {"seed": 7}}
    results = [{"problem": sweep_input.cases[0].record(), "top": []}]
    path = str(tmp_path / "r.yaml")
    results_module.write_results(path, {"meta": meta, "results": results}, results_module.FORMAT_YAML)
    _, state, _ = analyze.analyze(path, None, only=None, limit_cases=None, progress=False)
    values = {value for knob, value in state.mode("persistent").value_stats if knob == "blocks"}
    assert values == {1, 1000000}


def test_failure_and_valid_configs_have_the_same_up_scales_identity():
    for up_scales in (0, 1):
        text = ROUND_TRIP_INPUT.replace("upScales: 0", f"upScales: {up_scales}") % "case"
        case = config.parse_input(yaml.safe_load(text)).cases[0]
        for problem in config.expand_case(case).problems:
            valid = config.config_record(problem)
            assert valid["upScales"] == up_scales
            raw = config.RawConfig(
                problem.bm, problem.tn, problem.lanes_k, problem.waves, problem.pf,
                problem.up_scales, True, problem.group, problem.xcds, None,
            )
            failure = config.raw_record(case, raw)
            assert analyze.identity_of(failure) == analyze.identity_of(valid)


def test_analyzer_identity_of_an_old_file_uses_up_iters():
    old = {"bm": 1, "tn": 1, "lanesK": 64, "waves": 4, "prefetch": 2, "group": 1,
           "numXcds": 1, "grid3d": True, "blocks": None}
    assert analyze.identity_of({**old, "upIters": 3}).up_scales == 1
    assert analyze.identity_of({**old, "upIters": 0}).up_scales == 0
    assert analyze.identity_of({**old, "upIters": 3, "upScales": 0}).up_scales == 0


def test_analyzer_main_writes_the_report_and_the_trimmed_input(tmp_path, capsys):
    path, _, _ = write_round_trip_case(tmp_path, "case")
    report_path, trim_path = tmp_path / "report.yaml", tmp_path / "trim.yaml"
    argv = [path, "-o", str(report_path), "--trim-yaml", str(trim_path), "--no-progress",
            "--min-trials-grid", "0", "--keep-at-least", "1", "--min-top-slots-grid", "1"]
    assert analyze.main(argv) == 0
    out = capsys.readouterr().out
    assert "Analyzed 1 cases" in out and "Recommended prune" in out
    report = yaml.safe_load(report_path.read_text())
    assert report["meta"]["casesMissing"] == 0
    assert report["meta"]["prunePolicy"]["frequencyUsesPhi"] is True
    assert report["modes"]["grid"]["recommendedPrune"] == {"bm": [2]}
    trimmed = yaml.safe_load(trim_path.read_text())
    assert trimmed["problems"][0]["sweep"][0]["bm"] == [1]


def test_config_record_has_up_scales():
    p = config.expand_case(parse(f"problems: [{SHAPES}]").cases[0]).problems[0]
    record = config.config_record(p)
    assert record["upScales"] == p.up_scales and "upIters" in record


def test_analyzer_main_rejects_bad_option_values(tmp_path, capsys):
    path, _, _ = write_round_trip_case(tmp_path, "case")
    for flag in ("--keep-at-least=-1", "--max-phi=nan", "--p-threshold=1.5",
                 "--min-top-share=inf", "--min-trials=-3", "--limit-cases=-1"):
        with pytest.raises(SystemExit):
            analyze.main([path, flag])
    capsys.readouterr()


def test_analyzer_main_reports_a_malformed_results_file(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    assert analyze.main([str(path), "--no-progress"]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert analyze.main([str(tmp_path / "missing.yaml")]) == 2


def test_remove_stale_temporaries(tmp_path):
    with open("/proc/sys/kernel/pid_max") as stream:
        dead = int(stream.read()) + 1
    mine = os.getpid()
    names = [f"x.yaml.{dead}.tmp", f"x.yaml.{mine}.tmp", "x.yaml", f"y.{dead}.tmp"]
    for name in names:
        (tmp_path / name).write_text("")
    removed = results_module.remove_stale_temporaries(str(tmp_path), "x.yaml.")
    assert removed == 1
    assert sorted(os.listdir(tmp_path)) == sorted(names[1:])
    assert results_module.remove_stale_temporaries(str(tmp_path)) == 1
    assert f"y.{dead}.tmp" not in os.listdir(tmp_path)


class FakeStabilizer:
    def summary(self):
        return {"status": "not applicable"}


class FakeWriter:
    def __init__(self):
        self.added = []
        self.records = {}
        self.writes = 0

    def add(self, index, record, failures):
        self.added.append(index)
        self.records[index] = record

    def write(self, force):
        self.writes += 1


def make_sweep(tmp_path, expanded=(), gpus=()):
    options = types.SimpleNamespace(
        work_dir=str(tmp_path),
        input="in.yaml",
        no_verify=False,
        one_shot=False,
        output_format="yaml",
        build_jobs=1,
    )
    settings = {"verify_full_max_elems": 10}
    instance = sweep.Sweep(options, settings, list(expanded))
    instance.meta = instance.make_meta(list(gpus), FakeStabilizer())
    instance.writer = FakeWriter()
    return instance


def test_sweep_meta_has_error_fields_and_counts_errors(tmp_path):
    instance = make_sweep(tmp_path)
    assert instance.meta["casesErrored"] == 0
    assert instance.meta["interrupted"] is False
    assert instance.meta["aborted"] is None
    case = types.SimpleNamespace(label="c1", record=lambda: {"name": "c1"})
    expanded = types.SimpleNamespace(
        case=case, failures=[], swept=0, problems=[]
    )
    state = sweep.make_state(0, expanded)
    outcome = {"error": "boom", "ranked": [], "failures": [], "timed": 0}
    instance.finish_case(state, {**outcome, "gpu": 0, "seconds": 1})
    assert instance.meta["casesErrored"] == 1
    assert instance.errored_labels == ["c1"]
    assert instance.writer.added == [0]


def sweep_text(knobs: str) -> str:
    return f"problems: [{{name: a, m: 1, n: 4096, k: 4096, sweep: {{{knobs}}}}}]"


@pytest.mark.parametrize(
    "knobs, knob, match",
    [
        ("bm: 0", "bm", "must be at least 1, got 0"),
        ("lanesK: 0", "lanesK", "must be at least 1"),
        ("tn: 0", "tn", "must be at least 1"),
        ("waves: 0", "waves", "must be at least 1"),
        ("upScales: 2", "upScales", "must be 0 or 1"),
        ("prefetch: -1", "prefetch", "must be at least 0"),
        ("bm: {min: 0, max: 4, mul: 2}", "bm", "must be at least 1"),
        ("bm: {min: 4, max: 1, step: 1}", "bm", "no values"),
    ],
)
def test_invalid_knob_values_are_config_errors(knobs, knob, match):
    pattern = f"problems.0..sweep.{knob}.*{match}"
    with pytest.raises(config.SweepConfigError, match=pattern):
        parse(sweep_text(knobs))


def test_zero_prefetch_is_accepted():
    assert parse(sweep_text("prefetch: 0")).cases


@pytest.mark.parametrize("knobs", ["lanesK: 128", "lanesK: 128, grid3d: false"])
def test_oversized_lanes_are_constraint_failures(knobs):
    expanded = config.expand_case(parse(sweep_text(knobs)).cases[0])
    assert expanded.problems == []
    assert [f.stage for f in expanded.failures] == ["constraint"]
    reason = expanded.failures[0].reason
    if "grid3d" in knobs:
        assert reason == "lanes_k must be in (4, 8, 16, 64)"
        return
    assert reason == "invalid knob combination: division by zero"


def test_ordinary_constraint_failure_keeps_its_reason():
    text = "problems: [{name: a, m: 4, n: 4096, k: 1024, sweep: {lanesK: [32, 64]}}]"
    expanded = config.expand_case(parse(text).cases[0])
    assert len(expanded.failures) == 1
    assert not expanded.failures[0].reason.startswith("invalid knob combination")


@pytest.mark.parametrize(
    "ld, match",
    [
        ("ldA: 4092", "ldA: 4092 bytes is below alignUp.k, 4. = 4096"),
        ("ldB: 4092", "ldB: 4092 bytes is below"),
        ("ldD: 6", "ldD: 6 bytes is below 2 \\* m = 8 for shape m=4 n=64"),
        ("ldC: 6", "ldC: 6 bytes is below"),
    ],
)
def test_explicit_ld_must_hold_every_shape(ld, match):
    text = f"problems: [{{name: a, m: [1, 4], n: 64, k: 4096, {ld}}}]"
    with pytest.raises(config.SweepConfigError, match=match):
        parse(text)


def test_explicit_ld_must_hold_the_transposed_shape():
    text = "problems: [{name: a, m: 1, n: 64, k: 64, transpose: true, ldD: 2}]"
    with pytest.raises(config.SweepConfigError, match="ldD: 2 bytes.*m=64 n=1"):
        parse(text)


def test_valid_explicit_ld_is_accepted():
    text = "problems: [{name: a, m: 4, n: 64, k: 4096, ldA: 5000, ldD: 8, ldC: 16}]"
    case = parse(text).cases[0]
    assert (case.ld_a, case.ld_d, case.ld_c) == (5000, 8, 16)


@pytest.mark.parametrize("scalar", ["alpha: .nan", "beta: .inf", "alpha: -.inf"])
def test_scalars_must_be_finite(scalar):
    text = f"problems: [{{name: a, m: 1, n: 64, k: 64, {scalar}}}]"
    with pytest.raises(config.SweepConfigError, match="finite number"):
        parse(text)


@pytest.mark.parametrize(
    "spec, match",
    [
        ("{min: 1.5, max: 1.5}", "give a number to pin"),
        ("{min: 0.98, max: 1.02}", "within 0.05 of 1"),
        ("{min: 2, max: 1.5}", "min must be below max"),
        ("{min: 0, max: 1.5}", "min must be positive"),
    ],
)
def test_random_range_errors(spec, match):
    text = f"problems: [{{name: a, m: 1, n: 64, k: 64, alpha: {{random: {spec}}}}}]"
    with pytest.raises(config.SweepConfigError, match=match):
        parse(text)


def test_pinned_scalar_may_be_trivial():
    text = "problems: [{name: a, m: 1, n: 64, k: 64, alpha: 1.0, beta: -3}]"
    assert parse(text).cases


# Milestone 3: build pool, GPU pool and worker robustness.

DEAD_PID_OFFSET = 1


def dead_pid() -> int:
    with open("/proc/sys/kernel/pid_max") as stream:
        return int(stream.read()) + DEAD_PID_OFFSET


def test_build_kernel_emit_failure(monkeypatch, tmp_path):
    def fail(problem):
        raise ValueError("bad knobs\nsecond line")

    monkeypatch.setattr(support, "gemmAsm", fail)
    result = sweep.build_kernel("k", object(), str(tmp_path))
    assert (result.ok, result.stage) == (False, "emit")
    assert result.message == "bad knobs"
    assert os.listdir(tmp_path) == []


def test_build_kernel_assemble_failure(monkeypatch, tmp_path):
    def fail(asm):
        raise RuntimeError("assembler failed")

    monkeypatch.setattr(support, "gemmAsm", lambda problem: "asm")
    monkeypatch.setattr(support, "assemble", fail)
    result = sweep.build_kernel("k", object(), str(tmp_path))
    assert (result.ok, result.stage) == (False, "assemble")
    assert os.listdir(tmp_path) == []


def test_build_kernel_success_leaves_only_the_hsaco(monkeypatch, tmp_path):
    seen = []

    def assemble(asm):
        seen.extend(os.listdir(tmp_path))
        return b"hsaco"

    monkeypatch.setattr(support, "reduceAsm", lambda: "asm")
    monkeypatch.setattr(support, "assemble", assemble)
    result = sweep.build_kernel("k", None, str(tmp_path))
    assert (result.ok, result.message) == (True, "built")
    assert (tmp_path / "k.hsaco").read_bytes() == b"hsaco"
    assert os.listdir(tmp_path) == ["k.hsaco"]
    assert seen == [f"k.hsaco.{os.getpid()}.building"]


def test_build_kernel_reuses_an_existing_file(tmp_path):
    (tmp_path / "k.hsaco").write_bytes(b"old")
    result = sweep.build_kernel("k", object(), str(tmp_path))
    assert (result.ok, result.message) == (True, "cached")
    assert os.listdir(tmp_path) == ["k.hsaco"]


class FakeExecutor:
    """Hands out futures that the test resolves."""

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.calls = []
        self.closed = False

    def submit(self, function, name, problem, hsaco_dir):
        future = concurrent.futures.Future()
        self.calls.append((name, future))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.closed = True


class FakeFactory:
    def __init__(self):
        self.executors = []

    def __call__(self, max_workers):
        self.executors.append(FakeExecutor(max_workers))
        return self.executors[-1]


def built(name: str) -> sweep.BuildResult:
    return sweep.BuildResult(name, True, "", "built")


def make_stage(tmp_path):
    factory = FakeFactory()
    return sweep.BuildStage(str(tmp_path), 2, factory), factory


def test_build_stage_returns_finished_builds(tmp_path):
    stage, factory = make_stage(tmp_path)
    for name in "ab":
        stage.submit(name, None)
    stage.submit("a", None)
    assert [name for name, _ in factory.executors[0].calls] == ["a", "b"]
    assert stage.poll(0) == []
    (_, first), (_, second) = factory.executors[0].calls
    second.set_result(built("b"))
    first.set_exception(RuntimeError("boom\nmore"))
    names = {result.name: result for result in stage.poll(1)}
    assert names["b"].ok
    assert names["a"].message == "build process failed: boom"
    assert not names["a"].ok and stage.results.keys() == {"a", "b"}
    assert stage.poll(0) == []


def test_build_stage_retries_the_suspect_alone(tmp_path):
    stage, factory = make_stage(tmp_path)
    for name in "abc":
        stage.submit(name, None)
    (tmp_path / f"b.hsaco.{dead_pid()}.building").write_text("")
    (_, a), (_, b), (_, c) = factory.executors[0].calls
    a.set_result(built("a"))
    b.set_exception(BrokenProcessPool())
    c.set_exception(BrokenProcessPool())
    assert [result.name for result in stage.poll(0)] == ["a"]
    first, second, isolated = factory.executors
    assert first.closed and stage.restarts == 1
    assert [name for name, _ in second.calls] == ["c"]
    assert isolated.max_workers == 1
    assert [name for name, _ in isolated.calls] == ["b"]
    assert os.listdir(tmp_path) == []
    second.calls[0][1].set_result(built("c"))
    isolated.calls[0][1].set_exception(BrokenProcessPool())
    results = {result.name: result for result in stage.poll(1)}
    assert results["c"].ok
    assert (results["b"].ok, results["b"].stage) == (False, "emit")
    assert "crashed (retried alone)" in results["b"].message


def test_build_stage_suspect_that_builds_alone_is_recorded(tmp_path):
    stage, factory = make_stage(tmp_path)
    stage.submit("a", None)
    (tmp_path / f"a.hsaco.{dead_pid()}.building").write_text("")
    factory.executors[0].calls[0][1].set_exception(BrokenProcessPool())
    assert stage.poll(0) == []
    factory.executors[2].calls[0][1].set_result(built("a"))
    assert [result.ok for result in stage.poll(1)] == [True]
    stage.shutdown()
    assert all(executor.closed for executor in factory.executors)


def test_build_stage_aborts_after_too_many_crashes(tmp_path):
    stage, factory = make_stage(tmp_path)
    stage.submit("a", None)
    for _ in range(sweep.MAX_POOL_RESTARTS):
        factory.executors[-1].calls[-1][1].set_exception(BrokenProcessPool())
        assert stage.poll(0) == []
    factory.executors[-1].calls[-1][1].set_exception(BrokenProcessPool())
    with pytest.raises(sweep.SweepAbort, match="build pool crashed 4 times"):
        stage.poll(0)


def test_crashed_kernels_ignores_live_markers(tmp_path):
    names = [f"a.hsaco.{dead_pid()}.building", f"b.hsaco.{os.getpid()}.building"]
    for name in names:
        (tmp_path / name).write_text("")
    assert sweep.crashed_kernels(str(tmp_path)) == ["a"]
    assert os.listdir(tmp_path) == [names[1]]


def test_remove_stale_temporaries_by_suffix(tmp_path):
    dead = dead_pid()
    names = [f"k.hsaco.{dead}.building", f"k.hsaco.{dead}.tmp"]
    for name in names:
        (tmp_path / name).write_text("")
    removed = results_module.remove_stale_temporaries(
        str(tmp_path), suffix=".building"
    )
    assert removed == 1
    assert os.listdir(tmp_path) == [names[1]]


def make_expanded(label: str, problems=()):
    case = types.SimpleNamespace(label=label, record=lambda: {"name": label})
    return types.SimpleNamespace(
        case=case, failures=[], swept=len(problems), problems=list(problems)
    )


def test_case_built_blames_split_k_when_the_reduce_kernel_failed(tmp_path):
    text = "problems: [{name: a, m: 4, n: 64, k: 512, spk: [k, 256], useBeta: false}]"
    cases = parse(text).cases
    plain, split = (config.expand_case(case).problems[0] for case in cases[:2])
    assert plain.splits == 1 and split.splits > 1
    instance = make_sweep(tmp_path, [make_expanded("c", [plain, split])])
    instance.pool = FakePool([])
    reduce = sweep.BuildResult(support.REDUCE_KERNEL_NAME, False, "emit", "boom")
    names = [support.gemmKernelName(p) for p in (plain, split)]
    results = {name: built(name) for name in names}
    results[support.REDUCE_KERNEL_NAME] = reduce
    instance.builds = types.SimpleNamespace(results=results)
    state = instance.states[0]
    instance.case_built(state)
    assert [name for name, _ in state.entries] == [names[0]]
    (failure,) = state.build_failures
    assert failure.reason == f"{support.REDUCE_KERNEL_NAME}: boom"
    assert [job.entries for job in instance.ready_jobs] == [state.entries]


class NoBuilds:
    def poll(self, timeout):
        return []

    def shutdown(self):
        pass


class FakePool:
    """Scripted pool: each poll and lost call pops the next list."""

    def __init__(self, workers, messages=(), lost=()):
        self.workers = workers
        self.messages = list(messages)
        self.lost_script = list(lost)
        self.restarted = []
        self.stopped = False
        self.calls = 0

    def idle(self):
        return [w for w in self.workers if w.ready and w.job is None]

    def send(self, worker, job):
        worker.job = job
        worker.dispatched = time.monotonic()

    def poll(self, timeout):
        self.calls += 1
        assert self.calls < 20, "the loop does not terminate"
        return self.messages.pop(0) if self.messages else []

    def lost(self):
        return self.lost_script.pop(0) if self.lost_script else []

    def restart(self, worker):
        self.restarted.append(worker.gpu)
        worker.job = None

    def stop(self):
        self.stopped = True


def loop_sweep(tmp_path, labels, messages=(), lost=()):
    expanded = [make_expanded(label) for label in labels]
    instance = make_sweep(tmp_path, expanded, gpus=[0])
    worker_ = sweep.GpuWorker(0, None, None, ready=True)
    instance.pool = FakePool([worker_], messages, lost)
    instance.builds = NoBuilds()
    for state in instance.states:
        instance.ready_jobs.append(worker.Job(state.index, [], "", {}, True))
    return instance, worker_


def case_message(index, **extra):
    outcome = {
        "index": index,
        "gpu": 0,
        "ranked": [],
        "failures": [],
        "timed": 0,
        "clocks": {},
        "phases": {},
        "seconds": 1.0,
        **extra,
    }
    return ("case", 0, outcome)


def test_lost_worker_loses_its_case_and_the_sweep_continues(tmp_path):
    instance, worker_ = loop_sweep(tmp_path, ["c0", "c1"])
    instance.pool.lost_script = [[(worker_, "worker died (exit code -9)")], []]
    instance.pool.messages = [[], [(worker_, case_message(1))]]
    instance.loop()
    assert instance.writer.added == [0, 1]
    assert instance.writer.records[0]["error"] == "worker died (exit code -9)"
    assert "error" not in instance.writer.records[1]
    assert instance.pool.restarted == [0]
    assert worker_.deaths == 1
    assert instance.lost_cases[0] == ["c0"]
    assert instance.meta["casesErrored"] == 1


def test_three_deaths_on_one_gpu_abort_the_sweep(tmp_path):
    reason = "timeout: no progress for 600 s"
    instance, worker_ = loop_sweep(tmp_path, ["c0", "c1", "c2", "c3"])
    instance.pool.lost_script = [[(worker_, reason)]] * 3
    assert instance.run() == 1
    aborted = instance.meta["aborted"]
    assert "GPU 0" in aborted and reason in aborted
    assert "cases lost on it: c0; c1; c2" in aborted
    assert "cases not completed: c3" in aborted
    assert instance.meta["casesNotCompleted"] == 1
    assert instance.pool.stopped and instance.writer.writes == 1


def test_fault_restarts_the_worker_without_counting_a_death(tmp_path):
    instance, worker_ = loop_sweep(tmp_path, ["c0"])
    message = case_message(0, error="device fault: boom", fault=True)
    instance.pool.messages = [[(worker_, message)]]
    instance.loop()
    assert instance.pool.restarted == [0]
    assert worker_.deaths == 0
    assert instance.writer.records[0]["error"] == "device fault: boom"


def test_fatal_message_is_handled_as_a_lost_worker(tmp_path):
    instance, worker_ = loop_sweep(tmp_path, ["c0", "c1"])
    traceback_text = "Traceback:\n  File x\nRuntimeError: no device\n\n"
    instance.pool.messages = [[(worker_, ("fatal", 0, traceback_text))], []]
    instance.pool.messages.append([(worker_, case_message(1))])
    instance.loop()
    error = instance.writer.records[0]["error"]
    assert error == "worker failed: RuntimeError: no device"
    assert instance.pool.restarted == [0]


def fake_process(alive=True, exitcode=None):
    return types.SimpleNamespace(is_alive=lambda: alive, exitcode=exitcode)


def test_gpu_pool_lost_reports_deaths_and_hangs():
    pool = sweep.GpuPool.__new__(sweep.GpuPool)
    pool.hang_timeout_s = 10.0
    old = time.monotonic() - 100
    job = object()
    hung = sweep.GpuWorker(0, fake_process(), None, ready=True, job=job, last_seen=old)
    dead = sweep.GpuWorker(1, fake_process(False, -11), None, ready=True)
    idle = sweep.GpuWorker(2, fake_process(), None, ready=True, last_seen=old)
    starting = sweep.GpuWorker(3, fake_process(), None, last_seen=old)
    busy = sweep.GpuWorker(4, fake_process(), None, ready=True, job=job)
    broken = sweep.GpuWorker(5, fake_process(), None, ready=True, broken=True)
    pool.workers = [hung, dead, idle, starting, busy, broken]
    reasons = {w.gpu: reason for w, reason in pool.lost()}
    assert reasons.keys() == {0, 1, 3, 5}
    assert reasons[0] == reasons[3] == "timeout: no progress for 10 s"
    assert reasons[1] == "worker died (exit code -11)"
    assert reasons[5] == "worker connection lost"


def test_gpu_pool_idle_skips_busy_and_broken_workers():
    pool = sweep.GpuPool.__new__(sweep.GpuPool)
    free = sweep.GpuWorker(0, None, None, ready=True)
    pool.workers = [
        free,
        sweep.GpuWorker(1, None, None),
        sweep.GpuWorker(2, None, None, ready=True, job=object()),
        sweep.GpuWorker(3, None, None, ready=True, broken=True),
    ]
    assert pool.idle() == [free]


class FakeFlush:
    def ramp(self, milliseconds):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False


def patch_worker_process(monkeypatch, run_job):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    # The test process keeps its Ctrl-C handling.
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "OperandCache", lambda: None)
    monkeypatch.setattr(worker, "FastLlcFlush", FakeFlush)
    monkeypatch.setattr(worker, "device_name", lambda: "fake gpu")
    monkeypatch.setattr(worker, "device_pci_address", lambda: "0000:75:00.0")
    monkeypatch.setattr(worker.bench, "timingMethod", lambda: "fake timing")
    monkeypatch.setattr(worker, "run_job", run_job)


READY = {"name": "fake gpu", "timing": "fake timing", "pci": "0000:75:00.0"}


def drain(conn) -> list:
    messages = []
    while conn.poll(0):
        messages.append(conn.recv())
    return messages


def test_gpu_worker_serves_jobs_until_none(monkeypatch):
    def run_job(job, gpu, flush, cache, heartbeat):
        heartbeat()
        return {"index": job.index, "gpu": gpu}

    patch_worker_process(monkeypatch, run_job)
    parent, child = multiprocessing.Pipe()
    parent.send(worker.Job(7, [], "", {}, True))
    parent.send(None)
    worker.gpu_worker(2, child)
    assert drain(parent) == [
        ("ready", 2, READY),
        ("case", 2, {"index": 7, "gpu": 2}),
    ]
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"


def test_gpu_worker_exits_after_a_fault_outcome(monkeypatch):
    def run_job(job, gpu, flush, cache, heartbeat):
        return {"index": job.index, "gpu": gpu, "fault": True}

    patch_worker_process(monkeypatch, run_job)
    parent, child = multiprocessing.Pipe()
    parent.send(worker.Job(0, [], "", {}, True))
    parent.send(worker.Job(1, [], "", {}, True))
    worker.gpu_worker(2, child)
    kinds = [message[0] for message in drain(parent)]
    assert kinds == ["ready", "case"]
    # The second job was never received.
    assert child.poll(0)


def test_gpu_worker_returns_when_the_parent_is_gone(monkeypatch):
    class OrphanConn:
        sent = []

        def send(self, message):
            self.sent.append(message)

        def recv(self):
            raise EOFError

    patch_worker_process(monkeypatch, None)
    conn = OrphanConn()
    worker.gpu_worker(2, conn)
    assert conn.sent == [("ready", 2, READY)]


def test_gpu_worker_reports_a_fatal_error(monkeypatch):
    patch_worker_process(monkeypatch, None)
    monkeypatch.setattr(worker, "device_name", lambda: 1 / 0)
    parent, child = multiprocessing.Pipe()
    with pytest.raises(ZeroDivisionError):
        worker.gpu_worker(2, child)
    (message,) = drain(parent)
    assert message[0] == "fatal" and "ZeroDivisionError" in message[2]


def run_failing_job(monkeypatch, error, healthy):
    def time_case(*args):
        raise error

    monkeypatch.setattr(worker, "time_case", time_case)
    monkeypatch.setattr(worker, "device_healthy", lambda: healthy)
    return worker.run_job(worker.Job(3, [], "", {}, True), 0, None, None)


def test_run_job_reports_a_failing_case(monkeypatch):
    outcome = run_failing_job(monkeypatch, RuntimeError("boom\nmore"), True)
    assert outcome["error"] == "RuntimeError: boom"
    assert outcome["fault"] is False
    assert outcome["index"] == 3 and outcome["ranked"] == []


def test_run_job_flags_a_fault_when_the_device_is_unhealthy(monkeypatch):
    outcome = run_failing_job(monkeypatch, RuntimeError("boom"), False)
    assert outcome["fault"] is True


def test_run_job_reports_a_device_fault(monkeypatch):
    outcome = run_failing_job(monkeypatch, worker.DeviceFault("hang"), True)
    assert outcome["error"] == "device fault: hang"
    assert outcome["fault"] is True


def test_device_healthy_never_raises(monkeypatch):
    def fail():
        raise ImportError("no hip")

    monkeypatch.setattr(support, "_hip", fail)
    assert worker.device_healthy() is False


class FakeCandidate:
    def __init__(self, p, kernel, buffers):
        self.problem = p

    def prime(self, flush):
        pass

    def warm(self, count, flush):
        pass

    def time(self, count, flush):
        if self.problem == "bad":
            raise TypeError("bad launch")
        return worker.Timing([1000], [])


def screen_job():
    entries = [("k0", "ok"), ("k1", "bad"), ("k2", "ok")]
    settings = {"warmup": 1, "iters": 2}
    return worker.Job(0, entries, "", settings, True)


def patch_screen(monkeypatch, healthy):
    monkeypatch.setattr(worker, "load_kernel", lambda d, n: contextlib.nullcontext(n))
    monkeypatch.setattr(worker, "Candidate", FakeCandidate)
    monkeypatch.setattr(worker, "device_healthy", lambda: healthy)


def test_screen_records_any_exception_as_a_failure(monkeypatch):
    patch_screen(monkeypatch, True)
    outcome = {"failures": []}
    beats = []
    timings = worker.screen(screen_job(), None, None, outcome, lambda: beats.append(1))
    assert sorted(timings) == [0, 2]
    (failure,) = outcome["failures"]
    assert (failure["index"], failure["status"]) == (1, "failed to run")
    assert failure["reason"] == "bad launch"
    assert len(beats) == 3


def test_screen_raises_a_device_fault_after_recording_the_failure(monkeypatch):
    patch_screen(monkeypatch, False)
    outcome = {"failures": []}
    with pytest.raises(worker.DeviceFault, match="bad launch"):
        worker.screen(screen_job(), None, None, outcome)
    assert [f["index"] for f in outcome["failures"]] == [1]


def test_verification_keeps_wrong_result_apart_from_failed_to_run(monkeypatch):
    errors = {"k1": AssertionError("D mismatch"), "k2": TypeError("bad launch")}

    def check(job, buffers, p, name, verifier):
        if name in errors:
            raise errors[name]

    class FakeVerifier:
        def __init__(self, inputs, full):
            pass

        def label(self, p):
            return "full"

    monkeypatch.setattr(worker, "check_result", check)
    monkeypatch.setattr(worker, "Verifier", FakeVerifier)
    monkeypatch.setattr(worker, "device_healthy", lambda: True)
    p = types.SimpleNamespace(m=1, n=1, k=1)
    entries = [(name, p) for name in ("k0", "k1", "k2")]
    settings = {"verify_full_max_elems": 10}
    job = worker.Job(0, entries, "", settings, True)
    screening = {i: worker.Timing([1], []) for i in range(3)}
    outcome = {"failures": []}
    buffers = types.SimpleNamespace(inputs=None)
    labels = worker.verify_screened(job, buffers, screening, outcome)
    assert labels == {0: "full", 1: "failed", 2: "failed"}
    statuses = [f["status"] for f in outcome["failures"]]
    assert statuses == ["wrong result", "failed to run"]
    monkeypatch.setattr(worker, "device_healthy", lambda: False)
    with pytest.raises(worker.DeviceFault):
        worker.verify_screened(job, buffers, screening, {"failures": []})


def test_heartbeat_is_throttled(monkeypatch):
    sent = []
    conn = types.SimpleNamespace(send=sent.append)
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    beat = worker.Heartbeat(conn, 4, 9)
    beat()
    assert sent == []
    clock[0] += worker.PROGRESS_INTERVAL_S
    beat()
    beat()
    assert sent == [("progress", 4, 9)]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_resolve_settings_rejects_hang_timeouts(value):
    with pytest.raises(config.SweepConfigError, match="--hang-timeout"):
        resolve("--hang-timeout", value)


def test_hang_timeout_defaults_and_accepts_fractions():
    options = sweep.make_parser().parse_args(["--hang-timeout", "0.5"])
    assert options.hang_timeout == 0.5
    assert sweep.make_parser().parse_args([]).hang_timeout == sweep.HANG_TIMEOUT_S


def test_gpu_lists_have_unique_ids(monkeypatch):
    assert smi.parse_gpu_list("5,5,6") == [5, 6]
    patch_busyness(monkeypatch, {5: 0.0, 6: 0.0, 7: 0.0})
    assert smi.select_gpus("auto", 5.0, False) == [5, 6, 7]
    assert smi.select_gpus("7,5,7,5", 5.0, False) == [7, 5]


def test_pid_alive_treats_a_zombie_as_dead():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    try:
        deadline = time.monotonic() + 5
        while results_module.process_state(pid) != "Z":
            assert time.monotonic() < deadline, "the child did not exit"
            time.sleep(0.01)
        assert os.kill(pid, 0) is None
        assert not results_module.pid_alive(pid)
    finally:
        os.waitpid(pid, 0)
    assert not results_module.pid_alive(pid)
    assert results_module.pid_alive(os.getpid())


def segfaulting_build(name, problem, hsaco_dir):
    """Stands in for build_kernel in a real pool process."""
    if name != "bad":
        return built(name)
    marker = os.path.join(hsaco_dir, f"{name}.hsaco.{os.getpid()}.building")
    with open(marker, "w"):
        pass
    os.kill(os.getpid(), signal.SIGSEGV)


class RoutedExecutor:
    """A real spawn pool that runs segfaulting_build instead of build_kernel."""

    def __init__(self, max_workers):
        self.pool = sweep.make_build_executor(max_workers)

    def submit(self, function, name, problem, hsaco_dir):
        return self.pool.submit(segfaulting_build, name, problem, hsaco_dir)

    def shutdown(self, wait=True, cancel_futures=False):
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


def test_build_stage_blames_the_kernel_that_segfaults(tmp_path):
    stage = sweep.BuildStage(str(tmp_path), 1, RoutedExecutor)
    try:
        for name in ("good", "bad"):
            stage.submit(name, None)
        deadline = time.monotonic() + 60
        while len(stage.results) < 2:
            assert time.monotonic() < deadline, "the build stage did not finish"
            stage.poll(0.2)
    finally:
        stage.shutdown()
    assert stage.results["good"].ok
    assert "crashed (retried alone)" in stage.results["bad"].message
    assert os.listdir(tmp_path) == []


# Timing methodology: flush before every launch, events around the kernel only.


class FakeHandle:
    """A hip handle with the pointer accessor the launch code uses."""

    counter = 100
    names: dict = {}

    def __init__(self, name):
        FakeHandle.counter += 1
        self.name = name
        self.address = FakeHandle.counter
        FakeHandle.names[self.address] = name

    def as_c_void_p(self):
        return types.SimpleNamespace(value=self.address)


class FakeHip:
    """Records the calls of the timing code in log, as (kind, ...) tuples."""

    class hipError_t:
        hipSuccess = 0

    class hipMemcpyKind:
        hipMemcpyHostToDevice = 1

    def __init__(self, log):
        self.log = log
        self.events = {}
        self.fail_destroy = set()
        self.fail_create_at = None
        self.created = 0
        # Elapsed times in ms handed out in order; empty gives 3 us.
        self.elapsed = []
        self.malloc_error = 0

    def hipEventCreate(self):
        self.created += 1
        if self.created == self.fail_create_at:
            return (1, None)
        event = FakeHandle(f"event{self.created}")
        self.events[event.address] = event.name
        return (0, event)

    def hipEventDestroy(self, event):
        self.log.append(("destroy", event.name))
        return 1 if event.name in self.fail_destroy else 0

    def hipEventRecord(self, event, stream):
        self.log.append(("record", event.name))
        return 0

    def hipEventSynchronize(self, event):
        return 0

    def hipEventElapsedTime(self, start, stop):
        return (0, self.elapsed.pop(0) if self.elapsed else 0.003)

    def hipMalloc(self, nbytes):
        return (self.malloc_error, None if self.malloc_error else FakeHandle("buffer"))

    def hipDeviceSynchronize(self):
        self.log.append(("device_sync",))
        return 0

    def hipModuleLaunchKernel(self, function, *dims, **options):
        self.log.append(("launch", function.name))
        return 0


class FakeTimingFlush:
    def __init__(self, log):
        self.log = log

    def flush(self):
        self.log.append(("flush",))


def patch_timing(monkeypatch, ext):
    """A fake hip, with a fake ext launch if ext, else none; returns the log."""
    log: list = []
    hip = FakeHip(log)
    monkeypatch.setattr(support, "_hip", lambda: hip)
    monkeypatch.setattr(bench, "packArgs", lambda args: FakeHandle("packed"))

    def ext_launch(*args):
        # kernel, 3 global sizes, 3 block sizes, shared, stream, params, extra,
        # start, stop, flags.
        start, stop = hip.events[args[11]], hip.events[args[12]]
        log.append(("ext", start, stop, args[1:7], FakeHandle.names[args[0]]))
        return 0

    function = ext_launch if ext else None
    monkeypatch.setattr(bench, "extLaunchFunction", lambda: function)
    return log, hip


def gemm_kernel():
    return types.SimpleNamespace(function=FakeHandle("gemm"))


def launches_only(log):
    """The flush, launch and event-record entries; an ext launch keeps its two
    events."""
    kept = []
    for entry in log:
        if entry[0] == "ext":
            kept.append(entry[:3])
        elif entry[0] in ("flush", "launch", "record"):
            kept.append(entry)
    return kept


def test_time_launches_flushes_before_every_ext_launch(monkeypatch):
    log, _ = patch_timing(monkeypatch, True)
    times = bench.timeLaunches(
        gemm_kernel(), (2, 1, 1), (64, 1, 1), (), 4, FakeTimingFlush(log)
    )
    assert times == [3000] * 4
    assert launches_only(log) == [("flush",), ("ext", "event1", "event2")] * 4
    # The global sizes of the ext launch are in threads.
    assert [e[3] for e in log if e[0] == "ext"][0] == (128, 1, 1, 64, 1, 1)
    assert not [e for e in log if e[0] in ("record", "launch")]


def test_time_launches_fallback_records_events_around_the_launch_only(monkeypatch):
    log, _ = patch_timing(monkeypatch, False)
    bench.timeLaunches(gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 4, FakeTimingFlush(log))
    one = [("flush",), ("record", "event1"), ("launch", "gemm"), ("record", "event2")]
    assert launches_only(log) == one * 4


def test_time_launches_events_method_forces_the_fallback(monkeypatch):
    log, _ = patch_timing(monkeypatch, True)
    bench.timeLaunches(gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 1, None, "events")
    assert launches_only(log) == [
        ("record", "event1"),
        ("launch", "gemm"),
        ("record", "event2"),
    ]
    assert bench.timingMethod("events") == bench.EVENT_TIMING
    assert bench.timingMethod() == bench.EXT_TIMING


def test_time_launches_without_ext_names_the_fallback(monkeypatch):
    patch_timing(monkeypatch, False)
    assert bench.timingMethod() == bench.EVENT_TIMING


def test_time_launches_destroys_both_events_when_one_destroy_fails(monkeypatch):
    log, hip = patch_timing(monkeypatch, True)
    hip.fail_destroy = {"event2"}
    with pytest.raises(RuntimeError, match="hip call failed"):
        bench.timeLaunches(gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 1, None)
    assert [e for e in log if e[0] == "destroy"] == [
        ("destroy", "event2"),
        ("destroy", "event1"),
    ]


def test_time_launches_does_not_leak_when_the_second_event_fails(monkeypatch):
    log, hip = patch_timing(monkeypatch, True)
    hip.fail_create_at = 2
    with pytest.raises(RuntimeError):
        bench.timeLaunches(gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 1, None)
    assert [e for e in log if e[0] == "destroy"] == [("destroy", "event1")]


def test_ext_launch_error_is_reported(monkeypatch):
    patch_timing(monkeypatch, True)
    monkeypatch.setattr(bench, "extLaunchFunction", lambda: lambda *args: 719)
    with pytest.raises(RuntimeError, match="hipExtModuleLaunchKernel failed: error 719"):
        bench.timeLaunches(gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 1, None)


def test_warm_launches_are_flush_launch_pairs_without_events(monkeypatch):
    log, _ = patch_timing(monkeypatch, True)
    bench.warmLaunches(
        gemm_kernel(), (1, 1, 1), (64, 1, 1), (), 3, FakeTimingFlush(log)
    )
    kinds = [e for e in log if e[0] != "device_sync"]
    assert kinds == [("flush",), ("launch", "gemm")] * 3
    assert [e for e in log if e[0] == "device_sync"] == [("device_sync",)] * 3


def reduce_step():
    return types.SimpleNamespace(
        kernel=types.SimpleNamespace(function=FakeHandle("reduce")),
        grid=(1, 1, 1),
        block=(64, 1, 1),
        args=(),
    )


def make_candidate(monkeypatch, with_reduce):
    monkeypatch.setattr(support, "gemmBlock", lambda p: (64, 1, 1))
    buffers = types.SimpleNamespace(
        gemm_args=lambda p: (), reduce=reduce_step() if with_reduce else None
    )
    problem = types.SimpleNamespace(grid=(1, 1, 1))
    return worker.Candidate(problem, gemm_kernel(), buffers)


def test_candidate_times_flushed_gemm_pairs_then_warm_reduce(monkeypatch):
    log, _ = patch_timing(monkeypatch, False)
    candidate = make_candidate(monkeypatch, True)
    timing = candidate.time(3, FakeTimingFlush(log))
    sequence = [e[1] if e[0] == "launch" else e[0] for e in log if e[0] in ("flush", "launch")]
    assert sequence == ["flush", "gemm"] * 3 + ["reduce"] * 3
    assert len(timing.gemm_ns) == len(timing.reduce_ns) == 3


def test_candidate_warm_and_prime_are_untimed(monkeypatch):
    log, _ = patch_timing(monkeypatch, True)
    candidate = make_candidate(monkeypatch, True)
    candidate.prime(FakeTimingFlush(log))
    sequence = [e[1] if e[0] == "launch" else e[0] for e in log if e[0] in ("flush", "launch", "ext", "record")]
    count = bench.POST_LOAD_LAUNCHES
    assert sequence == ["flush", "gemm"] * count + ["reduce"] * count


def test_screening_issues_post_load_warmup_and_timed_flushes(monkeypatch):
    log, _ = patch_timing(monkeypatch, True)
    candidate = make_candidate(monkeypatch, False)
    monkeypatch.setattr(worker, "load_kernel", lambda d, n: contextlib.nullcontext(None))
    monkeypatch.setattr(worker, "Candidate", lambda p, kernel, buffers: candidate)
    settings = {"warmup": 3, "iters": 4}
    job = worker.Job(0, [("k0", None)], "", settings, True)
    timings = worker.screen(job, None, FakeTimingFlush(log), {"failures": []})
    flushes = [e for e in log if e[0] == "flush"]
    launches = [e for e in log if e[0] in ("launch", "ext")]
    untimed = 2 + 3
    assert len(flushes) == len(launches) == untimed + 4
    assert len([e for e in launches if e[0] == "ext"]) == 4
    assert len(timings[0].gemm_ns) == 4


class RotationCandidate:
    order: list = []

    def __init__(self, p, kernel, buffers):
        self.index = p

    def prime(self, flush):
        pass

    def time(self, count, flush):
        RotationCandidate.order.append(self.index)
        return worker.Timing([1000] * count, [])


def test_refine_rotates_the_candidate_order_every_round(monkeypatch):
    monkeypatch.setattr(worker, "load_kernel", lambda d, n: contextlib.nullcontext(n))
    monkeypatch.setattr(worker, "Candidate", RotationCandidate)
    RotationCandidate.order = []
    entries = [(f"k{i}", i) for i in range(3)]
    settings = {"refine_top": 3, "refine_iters": 4 * worker.ROUND_LAUNCHES}
    job = worker.Job(0, entries, "", settings, False)
    screening = {i: worker.Timing([1000 * (i + 1)], []) for i in range(3)}
    refined = worker.refine(job, None, None, screening, {"failures": []})
    assert RotationCandidate.order == [0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2]
    assert all(len(t.gemm_ns) == 4 * worker.ROUND_LAUNCHES for t in refined.values())


def sample_problem(m, n, k):
    knobs = {name: None for name in support.KNOB_OVERRIDE_NAMES}
    return support.makeProblem(m, n, k, padding=(0, 0, 0), **knobs)


def test_percentile_uses_the_nearest_rank():
    values = list(range(1, 11))
    assert bench.percentile(values, 0.9) == 9
    assert bench.percentile([5], 0.9) == 5
    assert bench.percentile([3, 1, 2], 0.9) == 3


def test_format_sweep_line_reports_means_with_min_and_p90_columns():
    p = sample_problem(4, 4096, 4096)
    line = bench.formatSweepLine(p, [1000, 2000, 6000], None)
    assert "gemm_us=3.00 gemm_min_us=1.00 gemm_p90_us=6.00" in line
    assert "total_us" not in line
    line = bench.formatSweepLine(p, [1000, 2000, 6000], [500, 500, 1000])
    assert "reduce_us=0.67" in line and "total_us=3.67" in line


def test_timing_statistics_rank_on_the_mean():
    timing = worker.Timing([1000, 2000, 6000], [500, 500, 1000])
    assert timing.totals_ns == [1500, 2500, 7000]
    assert timing.total_ns == pytest.approx(11000 / 3)
    assert (timing.min_ns, timing.p90_ns) == (1500, 7000)


def test_winner_entry_has_min_p90_and_launch_lists_but_one_shot_has_no_stats():
    refined = worker.Timing([1000, 2000, 6000], [500, 500, 1000])
    screened = worker.Timing([4000, 5000], [500, 500])
    entry = worker.winner_entry(3, refined, screened, "full")
    assert (entry["minUs"], entry["p90Us"]) == (1.5, 7.0)
    assert entry["gemmNs"] == [1000, 2000, 6000]
    assert entry["reduceNs"] == [500, 500, 1000]
    assert entry["screeningGemmNs"] == [4000, 5000]
    assert entry["screeningReduceNs"] == [500, 500]
    one_shot = worker.screening_only_entry(3, refined, "full")
    assert "minUs" not in one_shot and "p90Us" not in one_shot
    assert one_shot["gemmNs"] == [1000, 2000, 6000]
    assert "screeningGemmNs" not in one_shot
    plain = worker.screening_only_entry(3, worker.Timing([1000], []), "no")
    assert "reduceNs" not in plain


def mean_of_lists_us(entry):
    reduce = entry.get("reduceNs", [0] * len(entry["gemmNs"]))
    totals = [g + r for g, r in zip(entry["gemmNs"], reduce)]
    return round(sum(totals) / len(totals) / 1000, 3)


@pytest.mark.parametrize("fmt", ["yaml", "msgpack"])
def test_launch_lists_round_trip_and_time_is_their_mean(tmp_path, fmt):
    refined = worker.Timing([1000, 2000, 6000], [500, 500, 1000])
    screened = worker.Timing([4000, 5000, 4500], [500, 500, 500])
    top = [
        {"rank": 1, "config": {}, **worker.winner_entry(0, refined, screened, "no")},
        {"rank": 2, "config": {}, **worker.screening_only_entry(1, refined, "no")},
    ]
    path = str(tmp_path / f"r.{fmt}")
    document = {"meta": {}, "results": [{"problem": {"name": "a"}, "top": top}]}
    results_module.write_results(path, document, fmt)
    loaded = results_module.read_results(path)["results"][0]["top"]
    assert loaded == top
    for entry in loaded:
        assert entry["timeUs"] == mean_of_lists_us(entry)
        assert all(isinstance(v, int) for v in entry["gemmNs"])


def test_selector_launch_settings_default_to_the_sweep_settings():
    import argparse

    import select_skinny_gemm_kernels as kernels

    def resolved(iters, warmup, meta):
        args = argparse.Namespace(iters=iters, warmup=warmup)
        kernels.resolve_launch_settings(args, meta)
        return args.iters, args.warmup

    meta = {"settings": {"iters": 50, "warmup": 7}}
    assert resolved(None, None, meta) == (50, 7)
    assert resolved(10, 0, meta) == (10, 0)
    assert resolved(None, None, {}) == (kernels.DEFAULT_ITERS, kernels.DEFAULT_WARMUP)
    assert resolved(None, None, {"settings": {"iters": 9, "warmup": 0}}) == (9, 0)


def test_selector_mean_us_is_the_mean_of_the_per_iteration_totals():
    import select_skinny_gemm_kernels as kernels

    assert kernels.mean_us([1000, 2000, 6000], None) == pytest.approx(3.0)
    assert kernels.mean_us([1000, 2000, 6000], [500, 500, 1000]) == pytest.approx(11 / 3)


def test_result_writer_interval_grows_with_the_rewrite_duration(tmp_path, monkeypatch):
    now = [0.0]
    writes = []

    def slow_write(path, document, fmt):
        writes.append(now[0])
        now[0] += 20.0

    monkeypatch.setattr(sweep.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sweep, "write_results", slow_write)
    writer = sweep.ResultWriter(str(tmp_path / "r.yaml"), {}, "yaml", interval_s=30)
    writer.add(0, {"problem": {"name": "a"}}, [])
    assert writes == [0.0]
    # 10 times the 20 s rewrite is 200 s, longer than the 30 s interval.
    now[0] = 100.0
    writer.add(1, {"problem": {"name": "b"}}, [])
    assert writes == [0.0]
    now[0] = 230.0
    writer.add(2, {"problem": {"name": "c"}}, [])
    assert writes == [0.0, 230.0]
    now[0] = 231.0
    writer.write(force=True)
    assert len(writes) == 3


def test_timing_meta_names_the_method_and_the_reduce_policy():
    meta = sweep.timing_meta("m")
    assert meta["method"] == "m" and meta["flushPerLaunch"] is True
    assert meta["flushMiB"] == bench.FLUSH_MB == 512
    assert meta["statistic"] == "mean of the timed launches"
    assert "warm" in meta["reduce"]


def test_axis_indices_tile_samples_grow_with_the_tile_count_and_are_capped():
    extent_small, extent_large, tile = 4096, 4096 * 64, 16

    def count(extent):
        rng = np.random.default_rng(1)
        return len(inputs_module.axis_indices(extent, tile, rng))

    assert count(extent_large) > count(extent_small) > 100
    rng = np.random.default_rng(1)
    indices = inputs_module.axis_indices(extent_large, tile, rng)
    # The cap bounds the three samples of each random boundary.
    assert len(indices) <= 3 * inputs_module.SAMPLE_TILE_MAX + 400


def test_axis_indices_include_the_neighbourhood_of_random_boundaries():
    tile, extent = 16, 16 * 800
    indices = set(inputs_module.axis_indices(extent, tile, np.random.default_rng(3)).tolist())
    mid = [b for b in range(1, 800) if all(
        v in indices for v in (b * tile - 1, b * tile, b * tile + tile // 2))]
    assert len(mid) >= 100


class FakeOperand:
    def __init__(self, nbytes):
        self.nbytes = nbytes


def test_operand_cache_never_evicts_the_kept_operand(monkeypatch):
    monkeypatch.setattr(
        inputs_module, "generate_operand", lambda cols, k, ld, seed, role: FakeOperand(100)
    )
    cache = inputs_module.OperandCache(limit=150)
    key_a = inputs_module.operand_key(inputs_module.ROLE_A, 1, 1, 1)
    cache.get(inputs_module.ROLE_A, 1, 1, 1)
    # Without keep the older A would be evicted to fit the limit.
    cache.get(inputs_module.ROLE_B, 2, 1, 1, keep=(key_a,))
    assert key_a in cache.operands and len(cache.operands) == 2
    cache.get(inputs_module.ROLE_B, 3, 1, 1)
    assert key_a not in cache.operands
    assert len(cache.operands) == 1


def test_make_case_inputs_keeps_a_while_fetching_b(monkeypatch):
    monkeypatch.setattr(
        inputs_module, "generate_operand", lambda cols, k, ld, seed, role: FakeOperand(100)
    )
    cache = inputs_module.OperandCache(limit=50)
    p = sample_problem(4, 4096, 512)
    inputs = inputs_module.make_case_inputs(p, cache)
    assert len(cache.operands) == 2
    assert inputs.a is cache.operands[inputs_module.operand_key(inputs_module.ROLE_A, p.m, p.k, p.ldA)]


class RaisingHip:
    """A hip whose unload and free fail."""

    class hipError_t:
        hipSuccess = 0

    class hipMemcpyKind:
        hipMemcpyHostToDevice = 1

    def hipModuleLoadData(self, data):
        return (0, FakeHandle("module"))

    def hipModuleGetFunction(self, module, name):
        return (0, FakeHandle("function"))

    def hipModuleUnload(self, module):
        return 1

    def hipMalloc(self, nbytes):
        return (0, FakeHandle("buffer"))

    def hipMemcpy(self, *args):
        return 0

    def hipFree(self, buffer):
        return 1


def test_exit_does_not_mask_the_exception_in_flight(monkeypatch):
    monkeypatch.setattr(support, "_hip", lambda: RaisingHip())
    with pytest.raises(ValueError, match="body"):
        with support.LoadedKernel(b"", "k"):
            raise ValueError("body")
    with pytest.raises(ValueError, match="body"):
        with support.DeviceBuffer(np.zeros(4, dtype=np.uint8)):
            raise ValueError("body")
    # Without an exception in flight the close error is reported.
    with pytest.raises(RuntimeError, match="hip call failed"):
        with support.LoadedKernel(b"", "k"):
            pass
    with pytest.raises(RuntimeError, match="hip call failed"):
        with support.DeviceBuffer(np.zeros(4, dtype=np.uint8)):
            pass


def test_case_buffers_exit_does_not_mask_the_exception_in_flight():
    class Stack:
        def close(self):
            raise RuntimeError("close failed")

    buffers = worker.CaseBuffers.__new__(worker.CaseBuffers)
    buffers.stack = Stack()
    with pytest.raises(ValueError, match="body"):
        with buffers:
            raise ValueError("body")
    with pytest.raises(RuntimeError, match="close failed"):
        with buffers:
            pass


def test_flush_close_frees_the_kernel_even_if_the_free_raises(monkeypatch):
    import sweep_skinny_gemm_flush as flush_module

    closed = []
    flush = flush_module.FastLlcFlush.__new__(flush_module.FastLlcFlush)
    flush.buffer = FakeHandle("buffer")
    flush.kernel = types.SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(support, "_hip", lambda: RaisingHip())
    with pytest.raises(RuntimeError):
        flush.close()
    assert closed == [True] and flush.buffer is None


def test_flush_ramp_flushes_until_the_time_has_passed(monkeypatch):
    import sweep_skinny_gemm_flush as flush_module

    now = [0.0]
    flushes = []
    monkeypatch.setattr(flush_module.time, "monotonic", lambda: now[0])
    flush = flush_module.FastLlcFlush.__new__(flush_module.FastLlcFlush)
    flush.flush = lambda: (flushes.append(1), now.__setitem__(0, now[0] + 0.01))
    flush.ramp(50)
    assert len(flushes) == 5


# Directive: a flush before every GEMM launch, warm or timed; events only on the
# timed launches; the mean of the timed launches only.


def kernel_sequence(log):
    """The flushes and kernel launches in order; timed launches (the ones with
    events) are prefixed with timed."""
    sequence = []
    for entry in log:
        if entry[0] == "flush":
            sequence.append("flush")
        elif entry[0] == "launch":
            sequence.append(f"warm {entry[1]}")
        elif entry[0] == "ext":
            sequence.append(f"timed {entry[4]}")
    return sequence


def patch_selector_support(monkeypatch):
    kernels = {"gemm": gemm_kernel(), "reduce": reduce_step().kernel}
    monkeypatch.setattr(support, "gemmKernel", lambda p, version, preload: kernels["gemm"])
    monkeypatch.setattr(support, "reduceKernel", lambda version, preload: kernels["reduce"])
    monkeypatch.setattr(support, "gemmArgs", lambda *args: ())
    monkeypatch.setattr(support, "gemmBlock", lambda p: (64, 1, 1))
    monkeypatch.setattr(support, "reduceArgs", lambda *args: ())
    monkeypatch.setattr(support, "reduceLaunchDims", lambda m, n: ((1, 1, 1), (64, 1, 1)))
    return kernels


def selector_problem_fake(splits):
    return types.SimpleNamespace(
        grid=(1, 1, 1), splits=splits, m=4, n=4096, ldD=4, ldC=4, alpha=1.0, beta=0.0
    )


WARM, ITERS = 2, 3
FAKE_DEVICE = types.SimpleNamespace(d=None, a=None, b=None, scl_a=None, scl_b=None, ws=None, c=None)
GEMM_PATTERN = ["flush", "warm gemm"] * WARM + ["flush", "timed gemm"] * ITERS


def test_selector_warm_up_with_a_count_and_no_wall_time(monkeypatch):
    import select_skinny_gemm_kernels as kernels

    log, _ = patch_timing(monkeypatch, True)
    patch_selector_support(monkeypatch)
    kernel, flush = gemm_kernel(), FakeTimingFlush(log)
    launch = lambda n: bench.warmLaunches(kernel, (1, 1, 1), (64, 1, 1), (), n, flush)
    kernels.warm_up(launch, 3, 0.0)
    assert kernel_sequence(log) == ["flush", "warm gemm"] * 3
    log.clear()
    kernels.warm_up(launch, 0, 0.0)
    assert log == []


def measure_job(splits):
    problem = selector_problem_fake(splits)
    return types.SimpleNamespace(
        problem=problem, iters=ITERS, warmup=WARM, warmup_ms=0.0, verify=False
    )


def test_selector_measure_us_flushes_every_gemm_launch_and_averages_timed_only(monkeypatch):
    import select_skinny_gemm_kernels as kernels

    log, hip = patch_timing(monkeypatch, True)
    patch_selector_support(monkeypatch)
    hip.elapsed = [0.001, 0.002, 0.006]
    result = kernels.measure_us(measure_job(1), None, FAKE_DEVICE, FakeTimingFlush(log))
    assert kernel_sequence(log) == GEMM_PATTERN
    assert result == pytest.approx(3.0)


def test_selector_measure_us_split_k_reduce_is_warm_and_unflushed(monkeypatch):
    import select_skinny_gemm_kernels as kernels

    log, hip = patch_timing(monkeypatch, True)
    patch_selector_support(monkeypatch)
    hip.elapsed = [0.001, 0.002, 0.006, 0.0005, 0.0005, 0.001]
    result = kernels.measure_us(measure_job(2), None, FAKE_DEVICE, FakeTimingFlush(log))
    expected = GEMM_PATTERN + ["warm reduce"] * WARM + ["timed reduce"] * ITERS
    assert kernel_sequence(log) == expected
    assert result == pytest.approx((1.5 + 2.5 + 7.0) / 3)


@pytest.mark.parametrize("splits", [1, 2])
def test_validate_selection_time_problem_has_the_same_pattern(monkeypatch, splits):
    import select_skinny_gemm_from_sweep as selector

    log, hip = patch_timing(monkeypatch, True)
    patch_selector_support(monkeypatch)
    hip.elapsed = [0.001, 0.002, 0.006] + ([0.0005, 0.0005, 0.001] if splits > 1 else [])
    reducer = reduce_step().kernel
    result = selector.time_problem(
        selector_problem_fake(splits),
        gemm_kernel(),
        lambda: reducer,
        (None,) * 7,
        FakeTimingFlush(log),
        ITERS,
        WARM,
    )
    expected = list(GEMM_PATTERN)
    if splits > 1:
        expected += ["warm reduce"] * WARM + ["timed reduce"] * ITERS
    assert kernel_sequence(log) == expected
    assert result == pytest.approx(3.0 if splits == 1 else (1.5 + 2.5 + 7.0) / 3)


# The flush buffer allocation.


class FakeLoadedKernel:
    closed: list = []

    def __init__(self, hsaco, name):
        pass

    def close(self):
        FakeLoadedKernel.closed.append(True)


def patch_failing_flush_init(monkeypatch, module):
    log, hip = patch_timing(monkeypatch, True)
    hip.malloc_error = 2
    FakeLoadedKernel.closed = []
    monkeypatch.setattr(module, "LoadedKernel", FakeLoadedKernel)
    monkeypatch.setattr(
        support, "getDeviceProperties", lambda: types.SimpleNamespace(multiProcessorCount=4)
    )
    fake_jit = types.SimpleNamespace(assembleToHSACO=lambda asm, target: b"")
    monkeypatch.setitem(__import__("sys").modules, "tensilelite_jit", fake_jit)


def test_allocate_flush_buffer_names_the_size_on_failure(monkeypatch):
    _, hip = patch_timing(monkeypatch, True)
    hip.malloc_error = 2
    with pytest.raises(RuntimeError, match="cannot allocate the 512 MiB"):
        bench.allocateFlushBuffer(512 * 2**20)


def test_llc_flush_init_closes_the_kernel_when_the_allocation_fails(monkeypatch):
    patch_failing_flush_init(monkeypatch, bench)
    with pytest.raises(RuntimeError, match="cannot allocate the 512 MiB"):
        bench.LlcFlush()
    assert FakeLoadedKernel.closed == [True]


def test_fast_llc_flush_init_closes_the_kernel_when_the_allocation_fails(monkeypatch):
    import sweep_skinny_gemm_flush as flush_module

    patch_failing_flush_init(monkeypatch, flush_module)
    with pytest.raises(RuntimeError, match="cannot allocate the 512 MiB"):
        flush_module.FastLlcFlush()
    assert FakeLoadedKernel.closed == [True]


def test_ext_loader_tries_the_bare_soname_first(monkeypatch):
    tried = []

    def cdll(path):
        tried.append(path)
        raise OSError(path)

    monkeypatch.setattr(bench.ctypes, "CDLL", cdll)
    monkeypatch.setenv("ROCM_PATH", "/rocm")
    assert bench.loadExtLaunch() is None
    assert tried == ["libamdhip64.so", "/rocm/lib/libamdhip64.so"]


class RecordingBuilds:
    def __init__(self):
        self.names = []
        self.names_done = []
        self.results = {}

    def submit(self, name, problem):
        self.names.append(name)
        self.results[name] = built(name)

    def poll(self, timeout):
        """Every submitted kernel finishes at the first poll after its submit."""
        done, self.names_done = self.names[len(self.names_done) :], self.names
        return [built(name) for name in done]

    def shutdown(self):
        pass


def resume_input(tmp_path, names) -> str:
    entries = [
        {"name": name, "m": 4, "n": 4096, "k": 1024, "spk": "k", "useBeta": False}
        for name in names
    ]
    path = tmp_path / "in.yaml"
    path.write_text(yaml.safe_dump({"problems": entries, "defaults": {"sweep": MFMA_SWEEP}}))
    return str(path)


def resume_options(tmp_path, names, *args):
    arguments = [
        resume_input(tmp_path, names),
        "--resume",
        "-o",
        str(tmp_path / ("r.msgpack" if "--msgpack" in args else "r.yaml")),
        "--work-dir",
        str(tmp_path),
        *[a for a in args if a != "--msgpack"],
    ]
    options = sweep.make_parser().parse_args(arguments)
    options.output_format = results_module.output_format(options.output, None)
    return options


def resume_cases(options):
    sweep_input = config.load_input(options.input, options.seed)
    expanded = [config.expand_case(case) for case in sweep_input.cases]
    return sweep.resolve_settings(sweep_input.settings, options), expanded


def write_previous(tmp_path, options, names, errored=(), mutate=None):
    """A previous output with a record and a failed group per named case."""
    previous = resume_options(tmp_path, names)
    settings, expanded = resume_cases(previous)
    meta = {
        "oneShot": options.one_shot,
        "settings": sweep.meta_settings(settings, options),
        "timing": {"method": "m"},
        "gpuName": "gpu",
        "gpus": [],
    }
    if mutate:
        mutate(meta)
    records = []
    for item in expanded:
        record = {"problem": item.case.record(), "marker": item.case.name}
        if item.case.name in errored:
            record["error"] = "boom"
        records.append(record)
        group = make_failure_group(record["problem"], {"bm": 1})
        with open(results_module.failed_path(options.output), "a") as stream:
            results_module.dump_block([group], stream)
    document = {"meta": meta, "results": records}
    results_module.write_results(options.output, document, options.output_format)


def resumed_sweep(tmp_path, names, *args):
    options = resume_options(tmp_path, names, *args)
    settings, expanded = resume_cases(options)
    resumed = sweep.load_resumed(options, settings, expanded)
    instance = sweep.Sweep(options, settings, expanded, resumed)
    instance.meta = instance.make_meta([0], FakeStabilizer())
    instance.writer = sweep.ResultWriter(
        options.output, instance.meta, options.output_format, resumed=resumed
    )
    instance.builds = RecordingBuilds()
    instance.pool = FakePool([sweep.GpuWorker(0, None, None, ready=True)])
    return instance, resumed


def finish_job_cases(instance):
    for job in list(instance.ready_jobs):
        outcome = case_message(job.index)[2]
        instance.finish_case(instance.states[job.index], outcome)


@pytest.mark.parametrize("fmt", [[], ["--msgpack"]])
def test_resume_reruns_only_errored_and_new_cases(tmp_path, fmt):
    options = resume_options(tmp_path, "abx", *fmt)
    write_previous(tmp_path, options, "abx", errored={"b"})
    before = results_module.read_results(options.output)
    instance, resumed = resumed_sweep(tmp_path, "abc", *fmt)
    assert sorted(resumed.records) == [0]
    assert (resumed.dropped, resumed.rerun) == (1, 2)
    instance.submit_builds()
    assert len(instance.builds.names) == 2
    for name in instance.builds.names:
        instance.on_build(built(name))
    assert [job.index for job in instance.ready_jobs] == [1, 2]
    assert instance.remaining == 2 and instance.finished == {0}
    assert instance.meta["resumedCases"] == 1 and instance.meta["casesDone"] == 1
    finish_job_cases(instance)
    instance.writer.write(force=True)
    after = results_module.read_results(options.output)
    assert after["results"][0] == before["results"][0]
    assert [r["problem"]["name"] for r in after["results"]] == ["a", "b", "c"]
    assert not any("error" in r for r in after["results"])
    assert after["meta"]["resumedCases"] == 1
    assert instance.meta["casesDone"] == 3


def test_resume_failed_file_keeps_only_the_reused_cases(tmp_path):
    options = resume_options(tmp_path, "abx")
    write_previous(tmp_path, options, "abx", errored={"b"})
    instance, _ = resumed_sweep(tmp_path, "abc")
    groups = results_module.read_failures(options.output)
    assert [g["problem"]["name"] for g in groups] == ["a"]
    failure = [{"config": {"bm": 2}, "status": "failed to build"}]
    instance.writer.add(2, {"problem": {"name": "c"}}, failure)
    groups = results_module.read_failures(options.output)
    assert [g["problem"]["name"] for g in groups] == ["a", "c"]
    assert temporaries(tmp_path) == []


def test_resume_of_everything_builds_and_times_nothing(tmp_path):
    options = resume_options(tmp_path, "ab")
    write_previous(tmp_path, options, "ab")
    instance, resumed = resumed_sweep(tmp_path, "ab")
    instance.submit_builds()
    assert instance.builds.names == [] and not instance.ready_jobs
    assert instance.remaining == 0
    instance.loop()
    assert instance.meta["timing"] == {"method": "m"}
    assert instance.meta["gpuName"] == "gpu"
    assert instance.meta["casesDone"] == instance.meta["casesTotal"] == 2


def test_resume_rejects_different_settings(tmp_path):
    options = resume_options(tmp_path, "ab")
    write_previous(tmp_path, options, "ab")
    with pytest.raises(config.SweepConfigError, match=r"iters \(previous 20, current 7\)"):
        resumed_sweep(tmp_path, "ab", "--iters", "7")
    with pytest.raises(config.SweepConfigError, match=r"verify \(previous True, current False"):
        resumed_sweep(tmp_path, "ab", "--no-verify")


def test_resume_rejects_a_different_one_shot_mode(tmp_path):
    options = resume_options(tmp_path, "ab", "--one-shot")
    write_previous(tmp_path, options, "ab")
    with pytest.raises(config.SweepConfigError, match="oneShot True, this run False"):
        resumed_sweep(tmp_path, "ab")


def test_resume_needs_an_existing_output(tmp_path):
    with pytest.raises(config.SweepConfigError, match="--resume: no results at .*r.yaml"):
        resumed_sweep(tmp_path, "ab")


def test_resume_rejects_a_file_that_is_not_a_results_document(tmp_path):
    (tmp_path / "r.yaml").write_text("- 1\n")
    with pytest.raises(config.SweepConfigError, match="not a sweep results file"):
        resumed_sweep(tmp_path, "ab")


def test_resume_is_off_by_default():
    assert sweep.make_parser().parse_args([]).resume is False


def test_resume_builds_and_times_a_kernel_shared_with_a_reused_case_once(tmp_path):
    options = resume_options(tmp_path, "a")
    write_previous(tmp_path, options, "a")
    instance, _ = resumed_sweep(tmp_path, "ab")
    worker_ = instance.pool.workers[0]
    instance.pool.messages = [[(worker_, case_message(1))]]
    instance.submit_builds()
    instance.loop()
    assert len(instance.builds.names) == 1
    assert instance.remaining == 0 and instance.finished == {0, 1}
    assert [r["problem"]["name"] for r in instance.writer.records.values()] == ["a", "b"]
    assert instance.meta["casesDone"] == 2 and not instance.ready_jobs


def test_abort_of_a_resumed_run_keeps_the_reused_records(tmp_path):
    options = resume_options(tmp_path, "a")
    write_previous(tmp_path, options, "a")
    before = results_module.read_results(options.output)["results"][0]
    instance, _ = resumed_sweep(tmp_path, "abcd")
    worker_ = instance.pool.workers[0]
    instance.pool.lost_script = [[(worker_, "worker died (exit code -9)")]] * 3
    instance.submit_builds()
    assert instance.run() == 1
    results = results_module.read_results(options.output)["results"]
    assert results[0] == before
    assert instance.meta["aborted"]
    assert instance.meta["resumedCases"] == 1


def test_stale_temporaries_of_the_failed_file_are_removed_at_start(tmp_path):
    instance = make_sweep(tmp_path)
    instance.options.output = str(tmp_path / "r.yaml")
    os.makedirs(instance.hsaco_dir)
    names = [f"r.yaml.{dead_pid()}.tmp", f"r.failed.yaml.{dead_pid()}.tmp"]
    for name in names:
        (tmp_path / name).write_text("")
    instance.remove_stale_files()
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]
