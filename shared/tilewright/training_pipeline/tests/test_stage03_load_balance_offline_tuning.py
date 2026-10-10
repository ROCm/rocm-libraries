# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import io
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest
import yaml

import gpu_fakes as gf
import stage03_load_balance_offline_tuning as s03
from bench_fakes import (
    BANNER,
    LIBRARY_STEM,
    Gemm,
    bench_line,
    gentest_entries,
    kernel_name,
    measured_lines,
    preamble,
    solution,
    winner,
    write_fake_bench,
    write_logic,
)
from lib import fingerprint, kfd
from lib import kernel_identity as ki
from lib.bench_log import CANARY_FLAG_MARK, LogProbe, iter_problems, parse_log_file
from lib.bench_yaml import (
    CANARY_DEFAULTS,
    ENTRY_NAME_KEY,
    adaptive_line,
    canary_line,
    canary_needs_kernel,
    drop_line_knob,
    least_work_line,
    line_has_prescreen,
    line_solution_list,
    name_entry,
    read_bench_lines,
    shape_of_line,
)
from lib.dat import kernel_dat_info

GEMMS = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(10)]
PIPELINE_DIR = Path(__file__).resolve().parent.parent


def _key(g):
    return f"{g.m}x{g.n}x{g.k}x{g.batch}"


def _problem(g, sol=1):
    return (
        preamble(2)
        + measured_lines(0, g, sol, 5.0)
        + measured_lines(1, g, sol + 1, 4.0)
        + winner(1, g, sol + 1, 4.0)
    )


@pytest.fixture
def fake_bench(tmp_path, monkeypatch):
    bench = write_fake_bench(tmp_path)
    state = tmp_path / "state"
    state.mkdir()

    def with_plan(**plan):
        plan.setdefault("state_dir", str(state))
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan))
        monkeypatch.setenv("FAKE_BENCH_PLAN", str(path))
        return bench

    return with_plan


# ── warm-up ──────────────────────────────────────────────────────────────────


def test_render_warmup_line():
    line = (
        "- {function: matmul, M: 5, iters: 50, cold_iters: 50, adaptive: true, "
        "measure_time: 2.0, rotating: 16, skip_slow_solution_ratio: 0.5}\n"
    )
    out = s03.render_warmup_line(line)
    assert out.startswith("- {function: matmul, M: 5, iters: 1, cold_iters: 1,")
    for forced in (
        "rotating: 0",
        "skip_slow_solution_ratio: 0",
        "requested_solution_num: -1",
        "print_kernel_info: 0",
    ):
        assert forced in out
    assert "adaptive" not in out and "measure_time" not in out
    assert s03.render_warmup_line("- not a mapping") is None
    assert s03.render_warmup_line("- {a 1}") is None
    prescreen = line.replace("}", ", skip_screen_prepass: true}")
    assert s03.render_warmup_line(prescreen) is None


def test_stream_split_routes_the_first_problem_to_the_warmup_log():
    first_no_solution = preamble(2)[:1] + ["error: NO solution found! at x"]
    g1, g2 = GEMMS[:2]
    src = io.StringIO(
        "\n".join(BANNER + first_no_solution + _problem(g1) + _problem(g2)) + "\n"
    )
    warm, block = io.StringIO(), io.StringIO()
    assert s03._stream_split(src, warm, block) >= 0.0
    warm_lines = warm.getvalue().splitlines()
    block_lines = block.getvalue().splitlines()
    assert warm_lines == BANNER + first_no_solution
    assert block_lines == BANNER + _problem(g1) + _problem(g2)


# ── log probe ────────────────────────────────────────────────────────────────


def test_log_probe_progress_and_fatal(tmp_path):
    log = tmp_path / "block.log"
    log.write_text("old sub-run\nWinner: \n")
    probe = LogProbe(log, log.stat().st_size)
    assert not probe.progressed()
    with log.open("a") as f:
        f.write("\n".join(BANNER + preamble(1) + measured_lines(0, GEMMS[0], 1, 2.0)))
    assert not probe.progressed()
    with log.open("a") as f:
        f.write("\n" + preamble(1)[0] + "\nHSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION")
    assert probe.progressed() and probe.fatal_seen()


def test_log_probe_tells_when_the_library_is_loaded(tmp_path):
    log = tmp_path / "canary.log"
    rotating, supported = preamble(1)
    log.write_text("\n".join(BANNER + [rotating]) + "\n")
    probe = LogProbe(log, 0)
    assert not probe.loaded()
    with log.open("a") as f:
        f.write(supported + "\n")
    assert probe.loaded() and not probe.progressed()


# ── crash recovery within a block ────────────────────────────────────────────


def _block(gemms, **knobs):
    return s03._build_blocks([bench_line(g, **knobs) for g in gemms], len(gemms))[0]


def _run_block(block, bench, log, **kw):
    s03._run_block_with_recovery(block, 0, bench, log, stall_s=600.0, **kw)


def test_a_live_stage_keeps_its_work_dir_and_a_gone_one_loses_it(tmp_path):
    """stage03 keeps its bench yamls in tmp/<pid>: a stale attempt still
    running into the same output directory and a new one never remove each
    other's, and what a process that is gone left is cleared."""
    import os

    tmp = tmp_path / "out" / s03.TMP_DIR
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for pid in (gone.pid, live.pid):
            (tmp / str(pid)).mkdir(parents=True)
            (tmp / str(pid) / "block_0000.yaml").write_text("- {M: 1}\n")
        (tmp / "keep").mkdir()
        mine = tmp / str(os.getpid())
        mine.mkdir()
        (mine / "block_0001.yaml").write_text("- {M: 2}\n")
        s03._make_work_dir(mine)
        assert mine.is_dir() and not list(mine.iterdir())
        assert (tmp / str(live.pid) / "block_0000.yaml").is_file()
        assert not (tmp / str(gone.pid)).exists()
        assert (tmp / "keep").is_dir()
    finally:
        live.kill()
        live.wait()


def test_crashing_shape_is_skipped_and_the_block_resumes(tmp_path, fake_bench):
    bench = fake_bench(crash=[_key(GEMMS[2])])
    block = _block(GEMMS[:5])
    log = tmp_path / "logs" / "block.log"
    _run_block(block, bench, log)
    assert block.skipped == {2: 134}
    assert (block.next_offset, block.abandoned, block.n_benched) == (5, False, 4)
    text = log.read_text()
    assert "SKIPPED CRASHED GEMM at block-offset 2 (yaml line 3)" in text
    assert "transA=T transB=N (rc=134)" in text
    probs = parse_log_file(log)
    assert sum(p.complete for p in probs) == 4
    assert [p.shape()[:3] for p in probs if not p.complete] == [
        (GEMMS[2].m, GEMMS[2].n, GEMMS[2].k)
    ]


def test_single_candidate_problems_are_counted_without_winner_lines(
    tmp_path, fake_bench
):
    bench = fake_bench(n_candidates=1, crash=[_key(GEMMS[3])])
    block = _block(GEMMS[:5])
    _run_block(block, bench, tmp_path / "block.log")
    assert block.skipped == {3: 134}
    assert block.next_offset == 5


def test_teardown_crash_counts_as_success(tmp_path, fake_bench):
    bench = fake_bench(teardown_crash=True)
    block = _block(GEMMS[:4])
    _run_block(block, bench, tmp_path / "block.log")
    assert (block.skipped, block.next_offset, block.rc) == ({}, 4, 0)


def test_block_is_abandoned_after_repeated_zero_progress(tmp_path, fake_bench):
    bench = fake_bench(crash_silent=[_key(g) for g in GEMMS[:3]])
    block = _block(GEMMS[:5])
    log = tmp_path / "block.log"
    _run_block(block, bench, log)
    assert sorted(block.skipped) == [0, 1, 2]
    assert block.abandoned and block.unattempted == [3, 4]
    assert "ABORT: 3 consecutive sub-runs finished no GEMM; 2 GEMMs" in log.read_text()


def test_warmup_output_is_kept_out_of_the_block_log(tmp_path, fake_bench):
    bench = fake_bench()
    block = _block(GEMMS[:3])
    log, warm_log = tmp_path / "block.log", tmp_path / "warmup.log"
    calls = []
    _run_block(
        block,
        bench,
        log,
        warmup=True,
        warmup_log_path=warm_log,
        on_warmup=lambda n, s: calls.append(n),
    )
    assert block.next_offset == 3 and not block.skipped
    assert [p.shape()[:3] for p in parse_log_file(log)] == [
        (g.m, g.n, g.k) for g in GEMMS[:3]
    ]
    assert len(parse_log_file(warm_log)) == 1
    assert calls == [1]


# ── whole stage ──────────────────────────────────────────────────────────────


def _identity_args(tmp_path):
    """--identity-dir and --identity-stem of a library file holding the
    kernels the fake bench times (solution indices 100 to 102, named as it
    prints them)."""
    d = tmp_path / "library"
    if not d.is_dir():
        d.mkdir()
        write_logic(d / f"{LIBRARY_STEM}.dat", [solution(100 + i, i) for i in range(3)])
    return ["--identity-dir", d, "--identity-stem", LIBRARY_STEM]


def _main(monkeypatch, tmp_path, bench, *extra, gemms=GEMMS, in_dir=None, canary=False):
    """stage03 over `gemms` (or `in_dir`) on devices 0 and 1 without warm-up;
    with `canary`, under the canary without back-off, its kernel from a
    library file (the fake's lines are block-scaled)."""
    if in_dir is None:
        in_dir = tmp_path / "in"
        in_dir.mkdir(exist_ok=True)
        (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in gemms))
    out = tmp_path / "stage03"
    argv = ["stage03", "--in-dir", in_dir, "--out-dir", out, "--bench-binary", bench]
    argv += ["--devices", "0,1", "--no-warmup-pass", "--quiet"]
    if canary:
        argv += ["--canary", "--canary-backoff-s", 0, *_identity_args(tmp_path)]
    else:
        argv += ["--no-canary"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv + list(extra)])
    rc = s03.main()
    summary = json.loads((out / "stage03_summary.json").read_text())
    return rc, out, summary


ON_OFF = pytest.mark.parametrize("canary", [False, True], ids=["plain", "canary"])


@ON_OFF
def test_stage_retries_skips_and_reports_lost_shapes(
    tmp_path, monkeypatch, fake_bench, canary
):
    bench = fake_bench(crash_once=[_key(GEMMS[3])], crash=[_key(GEMMS[7])])
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, canary=canary
    )
    assert rc == 0
    assert (summary["n_blocks"], summary["block_size"]) == (4, 3)
    assert (summary["skipped"], summary["retried"], summary["recovered"]) == (2, 2, 1)
    assert (summary["lost"], summary["crash_blocks"]) == (1, [])
    assert (out / "skipped_gemms.txt").read_text().splitlines() == [
        "yaml_line,M,N,K,B,rc,reason",
        f"8,{GEMMS[7].m},{GEMMS[7].n},{GEMMS[7].k},1,134,crashed",
    ]
    logs = sorted(p.name for p in (out / "logs").iterdir())
    assert len([n for n in logs if n.startswith("block_0")]) == 4
    assert "block_retry_gpu0.log" in logs
    retry = parse_log_file(out / "logs" / "block_retry_gpu0.log")
    assert [p.shape()[:3] for p in retry if p.complete] == [
        (GEMMS[3].m, GEMMS[3].n, GEMMS[3].k)
    ]

    (out / "logs" / "block_0042_lines1-1_gpu9.log").write_text("stale\n")
    (out / "logs" / "warmup_0042_gpu9.log").write_text("stale\n")
    (out / "logs" / "canary_0042_gpu9.log").write_text("stale\n")
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, canary=canary
    )
    assert rc == 0 and (summary["skipped"], summary["lost"]) == (1, 1)
    logs = sorted(p.name for p in (out / "logs").iterdir())
    assert not [n for n in logs if "0042" in n]
    assert len([n for n in logs if n.startswith("block_")]) == 5
    assert len([n for n in logs if n.startswith("canary_")]) == (5 if canary else 0)
    assert summary["misaligned"] == []
    if canary:
        assert summary["canary"]["off"] == {} and not summary["canary"]["disabled"]


# The least-work line, which the canary times, comes last: the shapes that
# crash below are not the canary's.
DESC = GEMMS[::-1]


@ON_OFF
def test_retry_recovers_an_abandoned_blocks_remaining_shapes(
    tmp_path, monkeypatch, fake_bench, canary
):
    bench = fake_bench(crash_silent_once=[_key(g) for g in DESC[:3]])
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 1, gemms=DESC, canary=canary
    )
    assert rc == 0
    assert (summary["skipped"], summary["unattempted"]) == (3, 2)
    assert (summary["recovered"], summary["lost"], summary["crash_blocks"]) == (
        5,
        0,
        [],
    )
    assert not (out / "skipped_gemms.txt").exists()
    if canary:
        assert summary["canary"]["off"] == {} and summary["canary"]["rebenched"] == []


@ON_OFF
def test_unrecovered_abandoned_block_fails_the_tolerance_gate(
    tmp_path, monkeypatch, fake_bench, canary
):
    bench = fake_bench(crash_silent=[_key(g) for g in DESC[:3]])
    extra = ("--blocks-per-gpu", 1, "--no-retry-skipped")
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, *extra, gemms=DESC, canary=canary
    )
    assert rc == 2
    assert summary["crash_blocks"] == ["block_0000_lines1-5"]
    assert summary["crash_block_tolerance"] == 0
    reasons = [
        ln.rsplit(",", 1)[1]
        for ln in (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    ]
    assert reasons == ["crashed"] * 3 + ["unattempted"] * 2
    rc, _, _ = _main(
        monkeypatch,
        tmp_path,
        bench,
        *extra,
        "--max-crash-blocks",
        1,
        gemms=DESC,
        canary=canary,
    )
    assert rc == 0


def test_bench_yamls_stay_in_the_stage_directory(tmp_path, monkeypatch, fake_bench):
    system_tmp = tmp_path / "system_tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    pids = tmp_path / "pids"
    pids.mkdir()
    bench = fake_bench(crash=[_key(GEMMS[2])], pid_dir=str(pids))
    rc, out, _ = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 1)
    assert rc == 0
    records = [json.loads(p.read_text()) for p in pids.iterdir()]
    assert len(records) >= 3
    for rec in records:
        assert rec["yaml"].startswith(str(out / "tmp") + "/")
    assert list(system_tmp.iterdir()) == []
    assert not (out / "tmp").exists()


def _shape(g):
    return (g.m, g.n, g.k, g.batch)


def _entry_name(line):
    import re

    m = re.search(rf"(?<![\w]){ENTRY_NAME_KEY}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else None


def _unnamed(line):
    return drop_line_knob(line, ENTRY_NAME_KEY)


def test_stage_times_every_line_adaptively(tmp_path, monkeypatch, fake_bench):
    yamls = tmp_path / "yamls"
    yamls.mkdir()
    bench = fake_bench(yaml_dir=str(yamls))
    in_dir = tmp_path / "stage02"
    in_dir.mkdir()
    fixed = [bench_line(g, skip_slow_solution_ratio=0.25) for g in GEMMS[:3]]
    adaptive = [adaptive_line(bench_line(g), {}) for g in GEMMS[3:5]]
    (in_dir / "shapes.yaml").write_text("".join(fixed + adaptive))
    (in_dir / "shapes_probed.yaml").write_text("".join(map(bench_line, GEMMS[5:])))
    cfg = {"bench": {"adaptive": {"measure_time": 4.0}}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--config-yaml", cfg_path, in_dir=in_dir
    )
    assert rc == 0 and summary["in_yaml"] == str(in_dir / "shapes.yaml")
    benched = [ln for p in sorted(yamls.iterdir()) for ln in read_bench_lines(p)]
    assert sorted(_entry_name(ln) for ln in benched) == [f"line{i}" for i in "12345"]
    by_shape = {shape_of_line(ln): _unnamed(ln) for ln in benched}
    assert sorted(by_shape) == sorted(map(_shape, GEMMS[:5]))
    for g, line in zip(GEMMS[:3], fixed):
        timed = by_shape[_shape(g)]
        assert timed == adaptive_line(line, cfg)
        assert (
            "measure_time: 4.0" in timed and "skip_slow_solution_ratio: 0.25" in timed
        )
    for g, line in zip(GEMMS[3:5], adaptive):
        assert by_shape[_shape(g)] == line

    stale = tmp_path / "probe_era" / "stage02"
    stale.mkdir(parents=True)
    (stale / "shapes_probed.yaml").write_text("".join(map(bench_line, GEMMS)))
    argv = ["stage03", "--in-dir", stale, "--out-dir", tmp_path / "o2"]
    argv += ["--bench-binary", bench, "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    assert s03.main() == 1


# ── entries and the output they print ────────────────────────────────────────


def test_the_fake_bench_runs_the_first_of_equal_entries_alone():
    a, b = bench_line(GEMMS[0]), bench_line(GEMMS[1])
    fields = a.strip()[3:-1].split(", ")
    reordered = "- {" + ", ".join(reversed(fields)) + "}\n"
    assert gentest_entries(a + reordered + b + a) == [a.rstrip(), b.rstrip()]
    named = [name_entry(a, f"line{i}") for i in (1, 2)]
    assert gentest_entries("".join(named)) == [ln.rstrip() for ln in named]


def test_equal_lines_of_the_yaml_are_each_benched(tmp_path, monkeypatch, fake_bench):
    """Equal lines (audit v2 times its canary that way) are named apart, so
    the bench runs every one: the block log holds a problem for each."""
    yamls = tmp_path / "yamls"
    yamls.mkdir()
    bench = fake_bench(yaml_dir=str(yamls))
    gemms = [GEMMS[0]] * 3 + [GEMMS[1]]
    rc, out, summary = _main(
        monkeypatch,
        tmp_path,
        bench,
        "--devices",
        "0",
        "--blocks-per-gpu",
        1,
        gemms=gemms,
    )
    assert rc == 0 and summary["benched"] == 4 and summary["misaligned"] == []
    (log,) = (out / "logs").glob("block_*.log")
    assert [p.shape() for p in parse_log_file(log)] == [_shape(g) for g in gemms]
    (copy,) = yamls.iterdir()
    names = [_entry_name(ln) for ln in read_bench_lines(copy)]
    assert names == ["line1", "line2", "line3", "line4"]


def test_misalignment_compares_each_problem_with_its_entry():
    probs = list(iter_problems(_problem(GEMMS[0]) + _problem(GEMMS[1])))
    shapes = [_shape(g) for g in GEMMS[:3]]
    assert s03._misalignment(probs, shapes, False) is None
    fewer = "the process printed 2 problem(s) for 3 entr(ies)"
    assert s03._misalignment(probs, shapes, True) == fewer
    more = "the process printed 2 problem(s) for 1 entr(ies)"
    assert s03._misalignment(probs, shapes[:1], False) == more
    swapped = s03._misalignment(probs, [shapes[1], shapes[0]], True)
    assert swapped == "problem 1 of the process is 64x32x128x1, its entry 72x36x144x1"
    none_found = list(iter_problems(preamble(1)[:1] + ["error: NO solution found!"]))
    assert s03._misalignment(none_found, shapes[2:], True) is None


def test_output_that_does_not_line_up_with_the_entries_fails_the_stage(
    tmp_path, monkeypatch, fake_bench
):
    """An entry the bench gives up before its problem's first line leaves the
    process's output a problem short: the block log and the summary say so,
    and the stage fails instead of reporting every shape benched."""
    bench = fake_bench(silent_text="name: line5}")
    extra = ("--devices", "0", "--blocks-per-gpu", 1)
    rc, out, summary = _main(monkeypatch, tmp_path, bench, *extra, gemms=GEMMS[:5])
    assert rc == 2
    why = "the process printed 4 problem(s) for 5 entr(ies)"
    assert summary["misaligned"] == [f"block_0000_lines1-5: {why}"]
    assert "printed output that does not line up" in summary["problems"][0]
    (log,) = (out / "logs").glob("block_*.log")
    assert f"*** MISALIGNED: {why} ***" in log.read_text()


def test_retry_pass_is_skipped_above_the_cap(tmp_path, monkeypatch, fake_bench):
    bench = fake_bench(crash=[_key(GEMMS[1]), _key(GEMMS[6])])
    rc, _, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, "--max-retry-gemms", 1
    )
    assert rc == 0
    assert (summary["retried"], summary["lost"]) == (0, 2)


def _env_recording_bench(tmp_path, names):
    """A fake bench that writes the values of `names` in its environment to
    `<tmp_path>/env/<pid>.json`, then runs bench_fakes."""
    record = tmp_path / "env"
    record.mkdir()
    path = tmp_path / "env-bench"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        f"names = {list(names)!r}\n"
        f"out = os.path.join({str(record)!r}, str(os.getpid()) + '.json')\n"
        "with open(out, 'w') as f:\n"
        "    json.dump({n: os.environ.get(n) for n in names}, f)\n"
        "from bench_fakes import fake_bench_main\n"
        "sys.exit(fake_bench_main(sys.argv[1:]))\n"
    )
    path.chmod(0o755)
    return path, record


def test_every_bench_gets_the_runtime_env_and_platform_records_it(
    tmp_path, monkeypatch, fake_bench
):
    from lib.arch import STRICT_SWITCH

    fake_bench()
    for name in (STRICT_SWITCH, "TW_RUNTIME_KNOB"):
        monkeypatch.delenv(name, raising=False)
    bench, record = _env_recording_bench(tmp_path, [STRICT_SWITCH, "TW_RUNTIME_KNOB"])
    cfg = {"runtime": {"env": {STRICT_SWITCH: "0", "TW_RUNTIME_KNOB": 7}}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc, out, _ = _main(monkeypatch, tmp_path, bench, "--config-yaml", cfg_path)
    assert rc == 0
    seen = [json.loads(p.read_text()) for p in record.iterdir()]
    assert seen and all(s == {STRICT_SWITCH: "0", "TW_RUNTIME_KNOB": "7"} for s in seen)
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert platform["environment"] == {STRICT_SWITCH: "0", "TW_RUNTIME_KNOB": "7"}

    for p in record.iterdir():
        p.unlink()
    rc, out, _ = _main(monkeypatch, tmp_path, bench)
    assert rc == 0
    seen = [json.loads(p.read_text()) for p in record.iterdir()]
    assert seen and all(s == dict.fromkeys(s) for s in seen)
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert platform["environment"] == {STRICT_SWITCH: None}


def test_benches_get_launch_overrides_from_the_runtime_env_alone(
    tmp_path, monkeypatch, fake_bench
):
    """A launch override of the stage's environment does not reach a bench;
    one the config's runtime.env sets does, and platform.json records
    both."""
    fake_bench()
    monkeypatch.setenv("TENSILE_FIXED_WGM", "4")
    monkeypatch.setenv("TENSILE_STREAMK5_FORCE_MODE", "1")
    names = ["TENSILE_FIXED_WGM", "TENSILE_STREAMK5_FORCE_MODE", "TENSILE_DB"]
    bench, record = _env_recording_bench(tmp_path, names)
    cfg = {"runtime": {"env": {"TENSILE_STREAMK5_FORCE_MODE": 0, "TENSILE_DB": 2}}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc, out, _ = _main(monkeypatch, tmp_path, bench, "--config-yaml", cfg_path)
    assert rc == 0
    seen = [json.loads(p.read_text()) for p in record.iterdir()]
    assert seen and all(
        s
        == {
            "TENSILE_FIXED_WGM": None,
            "TENSILE_STREAMK5_FORCE_MODE": "0",
            "TENSILE_DB": "2",
        }
        for s in seen
    )
    launch = json.loads((out / fingerprint.PLATFORM_FILE).read_text())["launch"]
    assert launch["overrides"] == {"TENSILE_STREAMK5_FORCE_MODE": "0"}
    assert launch["stripped"] == ["TENSILE_FIXED_WGM"]
    assert launch["heuristic"] is None and launch["workspace_bytes"] is None

    for p in record.iterdir():
        p.unlink()
    rc, out, _ = _main(monkeypatch, tmp_path, bench)
    assert rc == 0
    seen = [json.loads(p.read_text()) for p in record.iterdir()]
    assert seen and all(s == dict.fromkeys(names) for s in seen)
    launch = json.loads((out / fingerprint.PLATFORM_FILE).read_text())["launch"]
    assert launch["overrides"] == {}
    assert launch["stripped"] == ["TENSILE_FIXED_WGM", "TENSILE_STREAMK5_FORCE_MODE"]


# ── GPUs: leases, platform, quarantine ───────────────────────────────────────


@pytest.fixture
def gpus(tmp_path, monkeypatch):
    """A fake sysfs with GPUs 0..2 (gpu_ids 1001..1003) and the stage's
    devices 0 and 1 visible as GPUs 0 and 1."""
    root = gf.write_sysfs(tmp_path / "sys")
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(root))
    for var in kfd.HIP_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(kfd.ROCR_VAR, "0,1")
    return root


def _kfd(root, **kw):
    return {"root": str(root), "gpu_ids": {"0": 1001, "1": 1002}, **kw}


@ON_OFF
def test_a_block_whose_gpu_another_process_uses_is_quarantined_and_benched_again(
    tmp_path, monkeypatch, fake_bench, gpus, canary
):
    with gf.outside_process() as pid:
        bench = fake_bench(kfd=_kfd(gpus, foreign={"1": pid}, hold_s=0.2))
        rc, out, summary = _main(
            monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, canary=canary
        )
    assert rc == 0 and summary["problems"] == [] and summary["lost"] == 0
    (q,) = summary["quarantined"]
    assert (q["device"], q["attempt"], q["rule"]) == (1, 1, "foreign-process")
    assert (q["foreign"][0]["pid"], q["foreign"][0]["comm"]) == (pid, "sleep")
    assert q["gpu"]["gpu_id"] == 1002 and q["foreign"][0]["first_seen"]
    qdir = Path(q["dir"])
    assert qdir.parent == out / "quarantine"
    assert qdir.name == f"{q['block']}_gpu1_q1"
    assert json.loads((qdir / "quarantine.json").read_text())["foreign"] == q["foreign"]
    moved = (qdir / f"{q['block']}_gpu1.log").read_text()
    assert f"*** QUARANTINED [foreign-process]: pid {pid} ('sleep'" in moved
    (retired,) = summary["retired_devices"]
    assert retired["device"] == 1 and "still on it" in retired["why"]
    logs = sorted(p.name for p in (out / "logs").glob("block_*.log"))
    assert len(logs) == 4 and all(n.endswith("_gpu0.log") for n in logs)
    again = next(b for b in summary["blocks"] if b["name"] == q["block"])
    assert (again["device"], again["gpu"], again["gpu_id"]) == (0, 0, 1001)
    assert (again["quarantines"], again["perf_level"], again["power_cap_w"]) == (
        1,
        "auto",
        100.0,
    )
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert [g["gpu_id"] for g in platform["gpus"]] == [1001, 1002]
    if canary:
        assert f"{s03.CANARY_LOG_PREFIX}{q['block'][6:10]}_gpu1.log" in q["files"]
        assert set(summary["canary"]["references"]) == {"0"}
        assert not summary["canary"]["off"] and not summary["canary"]["disabled"]


@ON_OFF
def test_a_block_quarantined_more_often_than_allowed_fails_the_stage(
    tmp_path, monkeypatch, fake_bench, gpus, canary
):
    with gf.outside_process() as pid:
        bench = fake_bench(kfd=_kfd(gpus, foreign={"1": pid}))
        rc, out, summary = _main(
            monkeypatch,
            tmp_path,
            bench,
            "--blocks-per-gpu",
            2,
            "--max-quarantines",
            0,
            canary=canary,
        )
    assert rc == 2
    (name,) = summary["quarantine_failed"]
    assert "quarantined more than --max-quarantines 0 time(s)" in summary["problems"][0]
    reasons = [
        ln.rsplit(",", 1)[1]
        for ln in (out / "skipped_gemms.txt").read_text().splitlines()[1:]
    ]
    assert reasons and set(reasons) == {"quarantined"}
    assert not list((out / "logs").glob(f"{name}_*.log"))


def test_a_gpu_busy_before_its_first_block_takes_no_block(
    tmp_path, monkeypatch, fake_bench, gpus
):
    with gf.outside_process() as pid:
        gf.set_memory(gpus, pid, {1002: 8192})
        bench = fake_bench(kfd=_kfd(gpus))
        rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2)
    assert rc == 0 and summary["quarantined"] == []
    (retired,) = summary["retired_devices"]
    assert retired["device"] == 1 and f"pid {pid} ('sleep'" in retired["why"]
    assert all(b["device"] == 0 for b in summary["blocks"])


def test_a_bench_on_a_gpu_it_was_not_given_fails_the_stage(
    tmp_path, monkeypatch, fake_bench, gpus
):
    bench = fake_bench(kfd=_kfd(gpus, gpu_ids={"0": 1003, "1": 1002}, hold_s=1.0))
    rc, out, summary = _main(
        monkeypatch,
        tmp_path,
        bench,
        "--blocks-per-gpu",
        1,
        "--stall-timeout-s",
        2,
        gemms=GEMMS[:4],
    )
    assert rc == 2
    assert "the GPU numbering is wrong" in summary["problems"][0]
    assert {q["rule"] for q in summary["quarantined"]} == {"gpu-numbering"}
    assert [r["device"] for r in summary["retired_devices"]] == [0]
    assert summary["lost"] == 0


def test_every_bench_sees_the_gpu_of_its_device_alone(
    tmp_path, monkeypatch, fake_bench, gpus
):
    monkeypatch.setenv(kfd.ROCR_VAR, "2,0")
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    ids = {"2": 1003, "0": 1001}
    bench = fake_bench(pid_dir=str(pid_dir), kfd=_kfd(gpus, gpu_ids=ids))
    rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2)
    assert rc == 0 and summary["quarantined"] == [] and summary["lost"] == 0
    runs = [json.loads(p.read_text()) for p in pid_dir.iterdir()]
    assert {(r["device"], r["rocr"]) for r in runs} == {("0", "2"), ("0", "0")}
    by_device = {(b["device"], b["gpu"], b["gpu_id"]) for b in summary["blocks"]}
    assert by_device == {(0, 2, 1003), (1, 0, 1001)}
    names = {p.name.rsplit("_", 1)[1] for p in (out / "logs").glob("block_*.log")}
    assert names == {"gpu0.log", "gpu1.log"}


@pytest.fixture
def quick_recheck(monkeypatch):
    monkeypatch.setattr(kfd, "UNATTRIBUTED_CONFIRM_S", 0.2)
    monkeypatch.setattr(kfd, "UNATTRIBUTED_STEP_S", 0.05)
    monkeypatch.setattr(kfd, "UNATTRIBUTED_SETTLE_S_PER_GIB", 0.0)
    monkeypatch.setattr(kfd, "RECHECK_S", 0.1)
    monkeypatch.setattr(kfd, "READMIT_CLEAN_S", 0.5)


def _clear_card_when(root, gpu, ready, after_s=0.0):
    def run():
        assert gf.wait_for(ready)
        time.sleep(after_s)
        gf.set_card_used(root, gpu, gf.IDLE_CARD_BYTES)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_unattributed_vram_quarantines_the_block_until_the_gpu_is_clean(
    tmp_path, monkeypatch, fake_bench, gpus, quick_recheck
):
    bench = fake_bench(kfd=_kfd(gpus, hidden={"1": 3 << 30}, hold_s=0.4))
    qdir = tmp_path / "stage03" / "quarantine"
    quarantined = lambda: any(qdir.glob("*/quarantine.json"))  # noqa: E731
    clear = _clear_card_when(gpus, gf.GPUS[1], quarantined, after_s=0.5)
    rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 5)
    clear.join()
    assert rc == 0 and summary["problems"] == [] and summary["lost"] == 0
    (q,) = summary["quarantined"]
    assert (q["device"], q["rule"], q["foreign"]) == (1, "unattributed-memory", [])
    assert q["unattributed"]["vram_bytes"] >= 3 << 30
    assert "of VRAM no listed process holds" in q["reason"]
    assert summary["retired_devices"] == []
    (spell,) = summary["returned_devices"]
    assert spell["device"] == 1 and "of VRAM no listed process holds" in spell["why"]
    assert sum(b["device"] == 1 for b in summary["blocks"]) >= 1


@pytest.mark.parametrize("per_gib", [0.1, 0.0])
def test_memory_a_bench_leaves_on_exit_counts_only_past_its_settle_window(
    tmp_path, monkeypatch, fake_bench, gpus, quick_recheck, per_gib
):
    """Every bench holds most of its card, which shows that memory in use for
    a while after the bench exits; a GPU benches another block after it. A
    window the memory scales outlasts that: no block is quarantined and no
    GPU retired. A window shorter than the release quarantines the block."""
    monkeypatch.setattr(kfd, "UNATTRIBUTED_SETTLE_S_PER_GIB", per_gib)
    held = {"0": 64 << 30, "1": 64 << 30}
    bench = fake_bench(kfd=_kfd(gpus, vram=held, linger_s=1.0))
    rc, out, summary = _main(
        monkeypatch,
        tmp_path,
        bench,
        "--blocks-per-gpu",
        2,
        "--max-quarantines",
        0,
        gemms=GEMMS[:4],
    )
    if per_gib:
        assert rc == 0 and summary["quarantined"] == [] and summary["lost"] == 0
        assert summary["retired_devices"] == summary["returned_devices"] == []
        per_gpu = [sum(b["device"] == d for b in summary["blocks"]) for d in (0, 1)]
        assert sum(per_gpu) == 4 and max(per_gpu) >= 2
    else:
        assert rc == 2
        assert {q["rule"] for q in summary["quarantined"]} == {"unattributed-memory"}


@pytest.mark.parametrize("per_gib", [0.1, 0.0])
def test_memory_on_a_gpu_when_its_block_is_due_is_waited_out_in_its_window(
    tmp_path, monkeypatch, fake_bench, gpus, quick_recheck, per_gib
):
    """Memory no process holds is in GPU 0's use when the stage starts, as
    an earlier bench leaves it, and goes a second later. A window the memory
    scales outlasts that: the check before the first block waits for it and
    the GPU takes its blocks. A window shorter than the release retires the
    GPU until it has been clean for a while."""
    monkeypatch.setattr(kfd, "UNATTRIBUTED_SETTLE_S_PER_GIB", per_gib)
    gf.set_card_used(gpus, gf.GPUS[0], gf.IDLE_CARD_BYTES + (64 << 30))
    started = tmp_path / "stage03" / fingerprint.PLATFORM_FILE
    clear = _clear_card_when(gpus, gf.GPUS[0], started.exists, after_s=1.0)
    rc, out, summary = _main(
        monkeypatch,
        tmp_path,
        fake_bench(kfd=_kfd(gpus)),
        "--devices",
        "0",
        "--blocks-per-gpu",
        2,
        gemms=GEMMS[:2],
    )
    clear.join()
    assert rc == 0 and summary["lost"] == 0 and summary["quarantined"] == []
    assert summary["retired_devices"] == []
    returned = [r["device"] for r in summary["returned_devices"]]
    assert returned == ([] if per_gib else [0])
    assert [b["device"] for b in summary["blocks"]] == [0, 0]


def test_with_every_gpu_retired_the_stage_waits_a_bounded_time(
    tmp_path, monkeypatch, fake_bench, gpus, quick_recheck
):
    monkeypatch.setattr(kfd, "ALL_RETIRED_WAIT_S", 0.5)
    for g in gf.GPUS[:2]:
        gf.set_card_used(gpus, g, 2 << 30)
    rc, out, summary = _main(monkeypatch, tmp_path, fake_bench(kfd=_kfd(gpus)))
    assert rc == 2 and summary["benched"] == 0
    assert [r["device"] for r in summary["retired_devices"]] == [0, 1]
    assert "every GPU stopped taking blocks" in summary["problems"][-1]

    monkeypatch.setattr(kfd, "ALL_RETIRED_WAIT_S", 60.0)
    t0 = time.monotonic()
    clear = _clear_card_when(gpus, gf.GPUS[1], lambda: time.monotonic() - t0 > 0.5)
    rc, out, summary = _main(monkeypatch, tmp_path, fake_bench(kfd=_kfd(gpus)))
    clear.join()
    assert rc == 0 and summary["lost"] == 0
    assert [r["device"] for r in summary["retired_devices"]] == [0]
    assert [r["device"] for r in summary["returned_devices"]] == [1]
    assert {b["device"] for b in summary["blocks"]} == {1}


HOLD = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {root!r})
    from lib import gpu_lease
    with gpu_lease.hold_gpus([1], "another run", run_dir="/runs/other"):
        print("held", flush=True)
        time.sleep(60)
    """
)


def test_a_leased_gpu_or_gpus_that_disagree_refuse_the_stage(
    tmp_path, monkeypatch, fake_bench, gpus, capsys
):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in GEMMS))
    out = tmp_path / "stage03"
    argv = ["stage03", "--in-dir", in_dir, "--out-dir", out, "--bench-binary"]
    argv += [fake_bench(), "--devices", "0,1", "--no-warmup-pass", "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD.format(root=str(PIPELINE_DIR))],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert s03.main() == 2
    finally:
        holder.kill()
        holder.wait()
    err = capsys.readouterr().err
    assert f"GPU 1 is leased by pid {holder.pid}" in err
    assert "for 'another run' in /runs/other" in err
    assert not (out / "logs").exists()

    (gpus / "class/drm/renderD136/device/power_dpm_force_performance_level").write_text(
        "manual\n"
    )
    assert s03.main() == 2
    assert "disagree on the performance level" in capsys.readouterr().err
    platform = json.loads((out / fingerprint.PLATFORM_FILE).read_text())
    assert platform["problems"] and not list((out / "logs").glob("block_*.log"))


# ── canary ───────────────────────────────────────────────────────────────────

LEAST = GEMMS[0]
PLAIN = {"scaleA": 0, "scaleB": 0}
# The canary kernel of the fake's block-scaled lines: the first kernel of the
# library file `_identity_args` writes.
CANARY_LABEL = ki.signature_label(kernel_dat_info(solution(100, 0)))


def test_the_canary_times_one_kernel_of_the_least_work_line():
    big = bench_line(Gemm(256, 256, 256), **PLAIN)
    tied = bench_line(Gemm(32, 64, 128), **PLAIN)
    least = bench_line(Gemm(64, 32, 128), skip_slow_solution_ratio=0.45, **PLAIN)
    assert least_work_line([big, least, tied]) == least
    assert least_work_line(["- {function: matmul}\n"]) is None
    line = canary_line(least)
    assert shape_of_line(line) == (64, 32, 128, 1)
    for knob in ("algo_method: 1", "requested_solution_num: 1", "print_kernel_info: 1"):
        assert knob in line
    assert "skip_slow_solution_ratio: 0" in line and "0.45" not in line
    assert line_solution_list(line) is None
    planned = bench_line(
        Gemm(64, 32, 128), solution_index_list="[5, 7]", skip_screen_prepass="true"
    )
    line = canary_line(planned, kernel=9)
    assert line_solution_list(line) == [5] and "algo_method" not in line
    assert not line_has_prescreen(line)
    mx = bench_line(Gemm(64, 32, 128), scaleA=3, algo_method=0)
    assert canary_needs_kernel(mx) and not canary_needs_kernel(planned)
    assert not canary_needs_kernel(least)
    assert canary_line(mx) is None
    line = canary_line(mx, kernel=9)
    assert line_solution_list(line) == [9] and "algo_method" not in line


def test_stream_split_puts_the_canary_around_the_shapes():
    g1, g2 = GEMMS[:2]
    one = preamble(1) + measured_lines(0, LEAST, 7, 3.0)
    src = io.StringIO("\n".join(BANNER + _problem(g1) + one + one + _problem(g2) + one))
    warm, block, canary = io.StringIO(), io.StringIO(), io.StringIO()
    s03._stream_split(src, warm, block, canary_f=canary, canaries=2, body=1)
    assert warm.getvalue().splitlines() == BANNER + _problem(g1)
    assert block.getvalue().splitlines() == BANNER + _problem(g2)
    assert canary.getvalue().splitlines() == one * 3
    src.seek(0)
    block, canary = io.StringIO(), io.StringIO()
    s03._stream_split(src, None, block, warmups=0, canary_f=canary, canaries=1, body=2)
    assert block.getvalue().splitlines() == BANNER + one * 2
    assert canary.getvalue().splitlines() == _problem(g1) + _problem(g2) + one


def test_a_reading_drifts_when_it_is_over_its_gpus_median_by_the_tolerance():
    c = s03.Canary(bench_line(LEAST), timings=3, tolerance=0.1)
    assert c.judge(0, [10.0, 10.2]) == (pytest.approx(10.1), [])
    assert c.judge(0, [10.0, 12.0]) == (pytest.approx(10.1), [12.0])
    assert c.judge(1, [30.0, 31.0])[1] == []
    assert c.judge(0, [8.0])[1] == []
    s = c.summary()
    assert s["references"] == {"0": 10.0, "1": 30.5}
    assert s["n_readings"] == {"0": 5, "1": 2}
    assert (c.reference(0), c.reference(2)) == (10.0, None)
    assert c.over(10.0, [10.9, 11.2]) == [11.2]


def test_diagnostics_name_the_canary_kernel_by_its_signature():
    """With the library files the bench loaded, by the signature of the
    kernel its index names when that kernel has the printed name; else by
    what the name gives, never by the index."""
    (prob,) = iter_problems(preamble(1) + measured_lines(0, LEAST, 100, 3.0))
    row = prob.tested[0]
    by_name = f"MT128x128x64 (name {ki.name_hash(kernel_name((128, 128, 64)))})"
    assert s03.kernel_label(row) == by_name
    library = [(LIBRARY_STEM, [kernel_dat_info(solution(100, 0))], None)]
    assert s03.kernel_label(row, ki.identity_of_files(library)) == CANARY_LABEL
    other = [(LIBRARY_STEM, [kernel_dat_info(solution(100, 0, mt=(64, 64, 64)))], None)]
    assert s03.kernel_label(row, ki.identity_of_files(other)) == by_name


def test_the_canary_follows_the_config_unless_the_command_line_says(tmp_path):
    lines = [bench_line(g, **PLAIN) for g in GEMMS]

    def resolve(cfg, lines=lines, identity=None, **cli):
        unset = ("canary", "canary_tolerance", "canary_timings", "canary_backoff_s")
        ns = argparse.Namespace(**{**dict.fromkeys(unset), **cli})
        return s03._resolve_canary(ns, cfg, lines, identity)

    assert CANARY_DEFAULTS["enabled"] is False
    assert resolve({}) == (None, "turned off")
    c, _ = resolve({}, canary=True)
    assert (c.tolerance, c.timings, c.backoff_s) == (
        CANARY_DEFAULTS["tolerance"],
        CANARY_DEFAULTS["timings"],
        CANARY_DEFAULTS["backoff_s"],
    )
    assert c.shape == _shape(LEAST)
    on = {"enabled": True, "tolerance": 0.2, "timings": 5, "backoff_s": 7}
    cfg = {"bench": {"canary": on}}
    c, _ = resolve(cfg)
    assert (c.tolerance, c.timings, c.backoff_s) == (0.2, 5, 7.0)
    c, _ = resolve(cfg, canary_tolerance=0.3, canary_backoff_s=0.0)
    assert (c.tolerance, c.backoff_s) == (0.3, 0.0)
    assert resolve(cfg, canary=False) == (None, "turned off")
    with pytest.raises(ValueError, match="tolerance above 0"):
        resolve(cfg, canary_tolerance=0.0)
    with pytest.raises(ValueError, match="back-off of at least 0 s"):
        resolve(cfg, canary_backoff_s=-1.0)
    mx = [bench_line(g) for g in GEMMS]
    c, why = resolve(cfg, lines=mx)
    assert c is None and "--identity-dir" in why
    d = _identity_args(tmp_path)[1]
    identity = s03.library_identity(d, [LIBRARY_STEM])
    c, _ = resolve(cfg, lines=mx, identity=identity)
    assert line_solution_list(c.line) == [100] and c.identity is identity


def _canary_logs(out):
    return sorted((out / "logs").glob(f"{s03.CANARY_LOG_PREFIX}*.log"))


def test_every_bench_process_times_the_canary_around_the_shapes(
    tmp_path, monkeypatch, fake_bench
):
    """The fake runs only the entries hipblaslt-bench runs. Every process
    times its canary entries, each named apart, before its shapes and after
    them; the canary logs hold only the canary's problems and the block
    logs every shape's."""
    yamls = tmp_path / "yamls"
    yamls.mkdir()
    bench = fake_bench(yaml_dir=str(yamls))
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, canary=True
    )
    assert rc == 0 and summary["lost"] == 0 and summary["misaligned"] == []
    c = summary["canary"]
    assert c["shape"] == list(_shape(LEAST)) and line_solution_list(c["line"]) == [100]
    assert (c["rebenched"], c["flagged"], c["flagged_at_end"]) == ([], [], [])
    assert (c["off"], c["disabled"], c["misaligned"]) == ({}, None, [])
    assert set(c["references"]) == {"0", "1"}
    assert c["kernels"] == {"0": CANARY_LABEL, "1": CANARY_LABEL}
    assert list(summary[ki.RECORD_KEY]["kernel_ids"][LIBRARY_STEM]) == ["100"]
    timings = CANARY_DEFAULTS["timings"]
    head = [f"canary_start{i}" for i in range(1, timings + 1)]
    tail = [f"canary_end{i}" for i in range(1, timings + 1)]
    shapes = []
    for y in yamls.iterdir():
        lines = read_bench_lines(y)
        names = [_entry_name(ln) for ln in lines]
        assert names[:timings] == head and names[-timings:] == tail
        assert len(gentest_entries("".join(lines))) == len(lines)
        canaries = lines[:timings] + lines[-timings:]
        assert {_unnamed(ln) for ln in canaries} == {c["line"] + "\n"}
        shapes += [shape_of_line(ln) for ln in lines[timings:-timings]]
    assert sorted(shapes) == sorted(_shape(g) for g in GEMMS)
    for b in summary["blocks"]:
        (r,) = b["canary"]["readings"]
        assert len(r["start_timings"]) == len(r["end_timings"]) == timings
        assert r["start"] == r["end"] == c["references"][str(b["device"])]
        assert (b["canary"]["drifted"], b["canary"]["rebenched"]) == ([], 0)
    assert len(_canary_logs(out)) == summary["n_blocks"]
    for log in _canary_logs(out):
        probs = parse_log_file(log)
        assert len(probs) == 2 * timings
        assert {(p.shape(), len(p.tested)) for p in probs} == {(_shape(LEAST), 1)}
    benched = [
        p.shape()
        for log in (out / "logs").glob("block_*.log")
        for p in parse_log_file(log)
    ]
    assert sorted(benched) == sorted(_shape(g) for g in GEMMS)


def test_canary_output_that_does_not_line_up_is_set_aside_and_benched_again(
    tmp_path, monkeypatch, fake_bench
):
    """Unnamed, the equal canary entries run once, before the shapes: by
    position two of the block's shapes would go to the canary log and the
    canary would read none after them. The stage sees the process's output
    not line up with its entries, moves it to misaligned/, turns the canary
    off on every GPU and benches the shapes again without it."""
    monkeypatch.setattr(s03, "name_entry", lambda line, name: line.rstrip() + "\n")
    rc, out, summary = _main(
        monkeypatch,
        tmp_path,
        fake_bench(),
        "--devices",
        "0",
        "--blocks-per-gpu",
        2,
        canary=True,
    )
    assert rc == 0 and (summary["skipped"], summary["lost"]) == (0, 0)
    assert summary["benched"] == len(GEMMS) and summary["misaligned"] == []
    c = summary["canary"]
    assert "does not line up with its entries" in c["disabled"]
    (rec,) = c["misaligned"]
    name = "block_0000_lines1-5"
    assert rec["block"] == name and rec["subrun"] == 1
    assert sorted(rec["files"]) == [f"{name}_gpu0.log", "canary_0000_gpu0.log"]
    d = Path(rec["dir"])
    assert d == out / s03.MISALIGNED_DIR / f"{name}_gpu0_s1"
    assert (
        json.loads((d / s03.MISALIGNED_RECORD).read_text())["reason"] == rec["reason"]
    )
    assert [p.shape() for p in parse_log_file(d / f"{name}_gpu0.log")] == [
        _shape(g) for g in GEMMS[2:5]
    ]
    log = (out / "logs" / f"{name}_gpu0.log").read_text()
    assert "*** CANARY MISALIGNED" in log
    benched = [
        p.shape()
        for log in sorted((out / "logs").glob("block_*.log"))
        for p in parse_log_file(log)
    ]
    assert benched == [_shape(g) for g in GEMMS]


def _drift_run(monkeypatch, tmp_path, fake_bench, processes, *extra):
    """Ten blocks of one GEMM on one GPU; the processes `processes` of the
    fifth block's GEMM run three times slower, as on a loaded host."""
    slow = [{"shape": _key(GEMMS[4]), "factor": 3.0, "processes": processes}]
    bench = fake_bench(slow=slow)
    extra = ("--devices", "0", "--blocks-per-gpu", 10, *extra)
    return _main(monkeypatch, tmp_path, bench, *extra, canary=True)


def test_a_block_whose_canary_drifts_is_benched_again(
    tmp_path, monkeypatch, fake_bench
):
    rc, out, summary = _drift_run(monkeypatch, tmp_path, fake_bench, [0])
    name = "block_0004_lines5-5"
    assert rc == 0 and summary["lost"] == 0
    c = summary["canary"]
    assert (c["rebenched"], c["flagged"], c["flagged_at_end"]) == ([name], [], [])
    ref = c["references"]["0"]
    (d,) = (out / s03.DRIFT_DIR).iterdir()
    rec = json.loads((d / s03.DRIFT_RECORD).read_text())
    assert d.name == f"{name}_gpu0_d1" and rec["block"] == name
    assert sorted(rec["files"]) == [f"{name}_gpu0.log", "canary_0004_gpu0.log"]
    assert rec["drifted"] == [pytest.approx(3 * ref)] * 2
    assert rec["tolerance"] == CANARY_DEFAULTS["tolerance"] and rec["gpu_id"] is None
    assert rec["kernel"] == CANARY_LABEL
    assert "CANARY DRIFT" in (d / f"{name}_gpu0.log").read_text()
    b = next(b for b in summary["blocks"] if b["name"] == name)
    assert (b["canary"]["rebenched"], b["canary"]["flagged"]) == (1, False)
    assert b["canary"]["drifted"] == []
    (prob,) = parse_log_file(out / "logs" / f"{name}_gpu0.log")
    assert min(float(t.fields["us"]) for t in prob.tested) == pytest.approx(10.0)


def test_a_block_that_drifts_again_is_kept_and_flagged(
    tmp_path, monkeypatch, fake_bench
):
    rc, out, summary = _drift_run(monkeypatch, tmp_path, fake_bench, [0, 1])
    name = "block_0004_lines5-5"
    assert rc == 0 and summary["lost"] == 0 and summary["problems"] == []
    assert (summary["canary"]["rebenched"], summary["canary"]["flagged"]) == (
        [name],
        [name],
    )
    b = next(b for b in summary["blocks"] if b["name"] == name)
    assert (b["canary"]["rebenched"], b["canary"]["flagged"]) == (1, True)
    log = out / "logs" / f"{name}_gpu0.log"
    assert f"{CANARY_FLAG_MARK}: canary " in log.read_text()
    (prob,) = parse_log_file(log)
    assert min(float(t.fields["us"]) for t in prob.tested) == pytest.approx(30.0)


def test_a_block_whose_canary_is_slow_from_the_start_is_flagged_at_the_end(
    tmp_path, monkeypatch, fake_bench
):
    """The first block on the GPU runs three times slower, its canary too:
    judged against its own readings it passes, and the stage's end judges
    it again against the GPU's final reference and flags it."""
    slow = [{"shape": _key(LEAST), "factor": 3.0, "processes": [0]}]
    extra = ("--devices", "0", "--blocks-per-gpu", 10)
    rc, out, summary = _main(
        monkeypatch, tmp_path, fake_bench(slow=slow), *extra, canary=True
    )
    name = "block_0000_lines1-1"
    c = summary["canary"]
    assert rc == 0 and (c["rebenched"], c["flagged"], c["flagged_at_end"]) == (
        [],
        [],
        [name],
    )
    b = next(b for b in summary["blocks"] if b["name"] == name)
    assert b["canary"]["drifted"] == [] and b["canary"]["flagged_at_end"]
    assert (
        b["canary"]["drifted_at_end"] == [pytest.approx(3 * c["references"]["0"])] * 2
    )
    log = (out / "logs" / f"{name}_gpu0.log").read_text()
    assert f"{CANARY_FLAG_MARK}: canary " in log and "stage's end" in log
    others = [b for b in summary["blocks"] if b["name"] != name]
    assert not any(b["canary"]["flagged"] for b in others)


def test_a_block_put_back_with_a_delay_is_taken_once_it_is_due():
    blocks = s03._build_blocks([bench_line(g) for g in GEMMS[:3]], 1)
    now = [0.0]
    work = s03._Work(blocks, max_quarantines=2, clock=lambda: now[0])
    first = work.take(lambda: False)
    work.requeue(first, delay_s=5.0)
    assert work.take(lambda: False) is blocks[1]
    assert work.take(lambda: False) is blocks[2]

    def tick():
        now[0] += 2.5
        return False

    assert work.take(tick) is first and now[0] >= 5.0


def test_a_retry_pass_whose_canary_drifts_is_benched_again(
    tmp_path, monkeypatch, fake_bench
):
    """A crash leaves a shape to the retry pass, whose process runs three
    times slower: its canary drifts, its logs move to drift/, and the retry
    pass benches it once more."""
    slow = [{"shape": _key(GEMMS[3]), "factor": 3.0, "processes": [1]}]
    bench = fake_bench(crash_once=[_key(GEMMS[3])], slow=slow)
    extra = ("--devices", "0", "--blocks-per-gpu", 2)
    rc, out, summary = _main(monkeypatch, tmp_path, bench, *extra, canary=True)
    assert rc == 0 and (summary["recovered"], summary["lost"]) == (1, 0)
    assert summary["canary"]["rebenched"] == ["block_retry"]
    (d,) = (out / s03.DRIFT_DIR).iterdir()
    rec = json.loads((d / s03.DRIFT_RECORD).read_text())
    assert d.name == "block_retry_gpu0_d1" and rec["block"] == "block_retry"
    assert sorted(rec["files"]) == ["block_retry_gpu0.log", "canary_retry_gpu0.log"]
    retry = next(b for b in summary["blocks"] if b["name"] == "block_retry")
    assert (retry["canary"]["rebenched"], retry["canary"]["flagged"]) == (1, False)
    (prob,) = parse_log_file(out / "logs" / "block_retry_gpu0.log")
    assert prob.shape() == _shape(GEMMS[3])
    assert min(float(t.fields["us"]) for t in prob.tested) == pytest.approx(10.0)


def test_the_retry_pass_waits_out_the_back_off_before_benching_again(
    tmp_path, fake_bench
):
    """The canary of the retry pass's process, three times slower, drifts
    against the readings the GPU's blocks took: the pass sleeps the
    back-off before it benches the shape again."""
    bench = fake_bench(
        slow=[{"shape": _key(GEMMS[1]), "factor": 3.0, "processes": [0]}]
    )
    block = _block(GEMMS[:2], **PLAIN)
    block.next_offset, block.skipped = 2, {1: 134}
    line = canary_line(bench_line(LEAST, **PLAIN))
    canary = s03.Canary(line, timings=3, tolerance=0.1, backoff_s=7.0)
    # The fake times a one-kernel problem of the default 10 us base at rank 0,
    # which it makes a quarter slower.
    canary.judge(0, [12.5] * 8)
    sleeps = []
    retried, recovered, retry = s03._retry_pass(
        [block],
        [0],
        bench,
        tmp_path / "logs",
        600.0,
        600.0,
        canary=canary,
        sleep=sleeps.append,
    )
    assert (retried, recovered, sleeps) == (1, 1, [7.0])
    assert canary.rebenched == ["block_retry"] and not retry.canary_flagged


@pytest.mark.parametrize("warmup", ["--no-warmup-pass", "--warmup-pass"])
def test_a_canary_that_fails_the_bench_is_turned_off_without_costing_a_shape(
    tmp_path, monkeypatch, fake_bench, warmup
):
    """The canary crashes the first process of a block after it printed its
    first lines, with or without a warm-up shape before it: the canary goes
    off on that GPU, a new process benches the shapes, and none is
    skipped."""
    bench = fake_bench(crash_text="name: canary_start1}")
    extra = (warmup, "--devices", "0", "--blocks-per-gpu", 2)
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, *extra, gemms=GEMMS[:4], canary=True
    )
    assert rc == 0 and (summary["skipped"], summary["lost"]) == (0, 0)
    assert "failed in the canary before" in summary["canary"]["off"]["0"]
    first = (out / "logs" / "block_0000_lines1-2_gpu0.log").read_text()
    assert "CANARY FAILED" in first and "SKIPPED" not in first
    benched = [
        p.shape() for p in parse_log_file(out / "logs" / "block_0001_lines3-4_gpu0.log")
    ]
    assert benched == [_shape(g) for g in GEMMS[2:4]]


def test_a_process_that_dies_before_the_canary_prints_is_not_blamed_on_it(
    tmp_path, monkeypatch, fake_bench
):
    """A process dies before it prints anything, its canary entry first and
    no warm-up before it: nothing shows the canary did it, so the canary
    stays on and the block's first shape is charged, as for any process
    that dies silently; the retry pass benches that shape."""
    bench = fake_bench(crash_silent_once=[_key(LEAST)])
    extra = ("--devices", "0", "--blocks-per-gpu", 2)
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, *extra, gemms=GEMMS[:4], canary=True
    )
    assert rc == 0
    assert (summary["skipped"], summary["recovered"], summary["lost"]) == (1, 1, 0)
    assert summary["canary"]["off"] == {} and summary["canary"]["disabled"] is None
    first = (out / "logs" / "block_0000_lines1-2_gpu0.log").read_text()
    assert "SKIPPED CRASHED GEMM at block-offset 0" in first
    assert "CANARY FAILED" not in first


@pytest.mark.parametrize(
    "crash, blamed", [("crash_silent_text", False), ("crash_text", True)]
)
def test_a_process_that_fails_after_the_shapes_keeps_their_data(
    tmp_path, monkeypatch, fake_bench, crash, blamed
):
    """Every process dies at its first canary entry after the block's
    shapes: before printing anything there (nothing shows the canary did it,
    so it stays on) or in its first lines (it goes off on the GPU). Either
    way every shape keeps its data."""
    bench = fake_bench(**{crash: "name: canary_end1}"})
    extra = ("--devices", "0", "--blocks-per-gpu", 2)
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, *extra, gemms=GEMMS[:4], canary=True
    )
    assert rc == 0 and (summary["benched"], summary["skipped"], summary["lost"]) == (
        4,
        0,
        0,
    )
    off = summary["canary"]["off"]
    if blamed:
        assert "failed in the canary after the block's shapes" in off["0"]
    else:
        assert off == {}
        for b in summary["blocks"]:
            assert [r["end"] for r in b["canary"]["readings"]] == [None]
