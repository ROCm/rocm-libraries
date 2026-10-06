# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import io
import json
import sys
import tempfile

import pytest

import stage03_load_balance_offline_tuning as s03
from bench_fakes import (
    BANNER,
    Gemm,
    bench_line,
    measured_lines,
    preamble,
    winner,
    write_fake_bench,
)
from lib.bench_log import LogProbe, parse_log_file

GEMMS = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(10)]


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


# ── crash recovery within a block ────────────────────────────────────────────


def _block(gemms, **knobs):
    return s03._build_blocks([bench_line(g, **knobs) for g in gemms], len(gemms))[0]


def _run_block(block, bench, log, **kw):
    s03._run_block_with_recovery(block, 0, bench, log, stall_s=600.0, **kw)


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


def _main(monkeypatch, tmp_path, bench, *extra, gemms=GEMMS):
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in gemms))
    out = tmp_path / "stage03"
    argv = ["stage03", "--in-dir", in_dir, "--out-dir", out, "--bench-binary", bench]
    argv += ["--devices", "0,1", "--no-warmup-pass", "--quiet", *extra]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    rc = s03.main()
    summary = json.loads((out / "stage03_summary.json").read_text())
    return rc, out, summary


def test_stage_retries_skips_and_reports_lost_shapes(tmp_path, monkeypatch, fake_bench):
    bench = fake_bench(crash_once=[_key(GEMMS[3])], crash=[_key(GEMMS[7])])
    rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2)
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
    rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2)
    assert rc == 0 and (summary["skipped"], summary["lost"]) == (1, 1)
    logs = sorted(p.name for p in (out / "logs").iterdir())
    assert not [n for n in logs if "0042" in n]
    assert len([n for n in logs if n.startswith("block_")]) == 5


def test_retry_recovers_an_abandoned_blocks_remaining_shapes(
    tmp_path, monkeypatch, fake_bench
):
    bench = fake_bench(crash_silent_once=[_key(g) for g in GEMMS[:3]])
    rc, out, summary = _main(monkeypatch, tmp_path, bench, "--blocks-per-gpu", 1)
    assert rc == 0
    assert (summary["skipped"], summary["unattempted"]) == (3, 2)
    assert (summary["recovered"], summary["lost"], summary["crash_blocks"]) == (
        5,
        0,
        [],
    )
    assert not (out / "skipped_gemms.txt").exists()


def test_unrecovered_abandoned_block_fails_the_tolerance_gate(
    tmp_path, monkeypatch, fake_bench
):
    bench = fake_bench(crash_silent=[_key(g) for g in GEMMS[:3]])
    rc, out, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 1, "--no-retry-skipped"
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
        "--blocks-per-gpu",
        1,
        "--no-retry-skipped",
        "--max-crash-blocks",
        1,
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


def test_retry_pass_is_skipped_above_the_cap(tmp_path, monkeypatch, fake_bench):
    bench = fake_bench(crash=[_key(GEMMS[1]), _key(GEMMS[6])])
    rc, _, summary = _main(
        monkeypatch, tmp_path, bench, "--blocks-per-gpu", 2, "--max-retry-gemms", 1
    )
    assert rc == 0
    assert (summary["retried"], summary["lost"]) == (0, 2)
