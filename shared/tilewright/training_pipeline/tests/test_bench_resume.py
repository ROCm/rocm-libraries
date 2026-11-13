"""stage03's cross-run resume (lib/bench_resume.py): a stage that an
interruption or an AC cycle cut off mid-shape goes on after its last
finished shape, keeps what it measured and benches the cut-off shape again;
its canary holds each resumed GPU to its baseline from before the
interruption (one that moved starts a new segment, its interrupted block
benched again from its start); a resume holds one GPU model; a stage that
ended is not continued; stage04 stamps each row with its bench process."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import stage03_load_balance_offline_tuning as s03
import test_stage03_load_balance_offline_tuning as t3
from bench_fakes import LIBRARY_STEM, Gemm, bench_line
from lib import bench_resume
from lib.bench_log import parse_log_file, parse_process_header
from test_stage03_load_balance_offline_tuning import fake_bench  # noqa: F401
from test_stage04_convert_to_enriched_dataset import (  # noqa: F401
    GC,
    _gemm,
    _rows,
    dataset,
)
from test_stage04_convert_to_enriched_dataset import _run as run_stage04

GEMMS = t3.GEMMS
SHAPES = [(g.m, g.n, g.k) for g in GEMMS]
ONE_BLOCK = ("--devices", "0", "--blocks-per-gpu", 1)
TWO_BLOCKS = ("--devices", "0", "--blocks-per-gpu", 2)
AC_CYCLE = "after-the-ac-cycle"


@pytest.fixture
def bench(fake_bench, tmp_path):  # noqa: F811
    """`fake_bench` with every bench process recorded in `pids/` and its yaml
    copied to `yamls/`; the test's end stops any process still running."""
    for name in ("pids", "yamls"):
        (tmp_path / name).mkdir()

    def with_plan(**plan):
        return fake_bench(
            pid_dir=str(tmp_path / "pids"), yaml_dir=str(tmp_path / "yamls"), **plan
        )

    yield with_plan
    _stop_benches(tmp_path)


def _stop_benches(tmp_path):
    """SIGKILL the process group of every fake bench of the test still
    running, as an AC cycle stops them."""
    for p in (tmp_path / "pids").glob("[0-9]*"):
        try:
            pgid = int(json.loads(p.read_text())["pgid"])
            cmd = Path(f"/proc/{pgid}/cmdline").read_bytes().decode(errors="replace")
        except (OSError, ValueError, KeyError):
            continue
        if str(tmp_path) in cmd:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass


def _argv(tmp_path, bench, *extra, canary=False, gemms=GEMMS):
    """The arguments `t3._main` gives stage03, `extra` after them."""
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in gemms))
    out = tmp_path / "stage03"
    argv = ["--in-dir", in_dir, "--out-dir", out, "--bench-binary", bench]
    argv += ["--devices", "0,1", "--no-warmup-pass", "--quiet", "--no-health-canary"]
    argv += ["--no-warmup-line"]
    if canary:
        argv += ["--canary", "--canary-backoff-s", 0, *t3._identity_args(tmp_path)]
    else:
        argv += ["--no-canary"]
    return [str(a) for a in argv + list(extra)], out


def _killed_mid_shape(tmp_path, bench, g, *extra, canary=False):
    """stage03 in a process of its own, killed with every bench process once
    a bench hangs in the shape of `g` (its hang_once marker): what an
    interruption or an AC cycle in the middle of a shape leaves."""
    argv, out = _argv(tmp_path, bench, *extra, canary=canary)
    marker = tmp_path / "state" / f"hang_once_{t3._key(g)}"
    proc = subprocess.Popen(
        [sys.executable, s03.__file__, *argv],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        while not marker.exists():
            assert proc.poll() is None, "stage03 ended before the bench hung"
            assert (
                time.time() < deadline
            ), "the bench never reached the shape it hangs in"
            time.sleep(0.1)
        time.sleep(0.5)
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        _stop_benches(tmp_path)
    return out


def _stopped_mid_shape(tmp_path, bench, g, *extra, sig=signal.SIGTERM, gemms=GEMMS):
    """stage03 over `gemms` in a process of its own, sent `sig` once a bench
    hangs in the shape of `g` (its hang_once marker), as the orchestrator
    stops a stage: the stage stops its benches itself. Returns its output
    directory and exit code."""
    argv, out = _argv(tmp_path, bench, *extra, gemms=gemms)
    marker = tmp_path / "state" / f"hang_once_{t3._key(g)}"
    proc = subprocess.Popen(
        [sys.executable, s03.__file__, *argv],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        while not marker.exists():
            assert proc.poll() is None, "stage03 ended before the bench hung"
            assert (
                time.time() < deadline
            ), "the bench never reached the shape it hangs in"
            time.sleep(0.1)
        time.sleep(0.5)
        proc.send_signal(sig)
        rc = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        _stop_benches(tmp_path)
    return out, rc


def _unit(out, name):
    return json.loads((out / bench_resume.UNITS / f"{name}.json").read_text())


def _resume(
    monkeypatch, tmp_path, bench, *extra, canary=False, boot=AC_CYCLE, gemms=GEMMS
):
    """stage03 started again after the interruption, on a new boot."""
    monkeypatch.setenv(bench_resume.BOOT_ID_ENV, boot)
    return t3._main(monkeypatch, tmp_path, bench, *extra, canary=canary, gemms=gemms)


def _yamls(tmp_path):
    """{bench pid: the shapes its yaml lists}, the canary's entries (which
    list the kernel they time) left out."""
    out = {}
    for p in (tmp_path / "yamls").glob("*.yaml"):
        entries = yaml.safe_load(p.read_text())
        out[int(p.stem)] = [
            (d["M"], d["N"], d["K"]) for d in entries if "solution_index_list" not in d
        ]
    return out


def _log(out, block):
    """The one log of block number `block` in `out/logs`."""
    (log,) = (out / "logs").glob(f"block_{block:04d}_*.log")
    return log


def _headers(log):
    """[(line number, PROCESS header fields)] of a block log."""
    out = []
    for n, line in enumerate(log.read_text().splitlines(), 1):
        h = parse_process_header(line)
        if h is not None:
            out.append((n, h))
    return out


def _complete(log):
    return [p.shape()[:3] for p in parse_log_file(log) if p.complete]


def _journal(out):
    return json.loads((out / bench_resume.JOURNAL).read_text())


def test_a_stage_killed_mid_shape_goes_on_after_its_last_finished_shape(
    tmp_path, monkeypatch, bench
):
    b = bench(hang_once=[t3._key(GEMMS[2])])
    out = _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK)
    journal = _journal(out)
    assert journal["end"] is None and len(journal["runs"]) == 1
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert rc == 0 and summary["lost"] == 0
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert later == [SHAPES[2:]]
    log = _log(out, 0)
    (first, h1), (second, h2) = _headers(log)
    assert (h1["run"], h1["segment"], h1["canary"]) == ("1", "1", "off")
    assert (h2["run"], h2["segment"], h2["boot_id"]) == ("2", "1", AC_CYCLE)
    probs = parse_log_file(log)
    (cut,) = [p for p in probs if not p.complete]
    assert cut.shape() is None and first < cut.line_no < second
    assert [
        p.shape()[:3] for p in probs if p.complete and p.line_no < second
    ] == SHAPES[:2]
    assert [p.shape()[:3] for p in probs if p.line_no > second] == SHAPES[2:]
    journal = _journal(out)
    assert journal["end"]["rc"] == 0 and len(journal["runs"]) == 2
    assert journal["runs"][1]["boot_id"] == AC_CYCLE


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_stage_stopped_mid_shape_leaves_the_shape_to_the_resume(
    tmp_path, monkeypatch, bench, sig
):
    """The stage's own stop ends its benches: the line one was in is no
    crash, and the block stays unfinished for the run that resumes."""
    b = bench(hang_once=[t3._key(GEMMS[2])])
    out, rc = _stopped_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK, sig=sig)
    assert rc == 128 + sig and _journal(out)["end"] is None
    text = _log(out, 0).read_text()
    assert "SKIPPED CRASHED" not in text and "STOPPED at block-offset 2" in text
    unit = _unit(out, "block_0000_lines1-10")
    assert (unit["next_offset"], unit["skipped"]) == (2, {})
    assert not unit["done"] and not unit["abandoned"] and unit["stopped"]
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert (summary["lost"], summary["skipped"], summary["retried"]) == (0, 0, 0)
    assert rc == 0
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert later == [SHAPES[2:]]
    assert _complete(_log(out, 0)) == SHAPES


def test_a_stop_leaving_more_lines_than_the_retry_pass_takes_loses_none(
    tmp_path, monkeypatch, bench
):
    """A stop early in a long block leaves more lines than --max-retry-gemms,
    which a retry pass would refuse: the resumed block benches them."""
    b = bench(hang_once=[t3._key(MANY[3])])
    extra = (*ONE_BLOCK, "--max-retry-gemms", 2)
    out, rc = _stopped_mid_shape(tmp_path, b, MANY[3], *extra, gemms=MANY)
    assert rc == 128 + signal.SIGTERM
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *extra, gemms=MANY)
    assert (summary["lost"], summary["crash_blocks"]) == (0, [])
    assert rc == 0
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert later == [[(g.m, g.n, g.k) for g in MANY[3:]]]


def test_a_crash_before_the_stop_still_counts_as_a_crash(tmp_path, monkeypatch, bench):
    """A bench that crashed on its own skips its line, as any crash does,
    also when the stage stops later in the block; the resumed block goes on
    after the line the stop cut off, and the retry pass benches the crash."""
    b = bench(crash_silent=[t3._key(GEMMS[1])], hang_once=[t3._key(GEMMS[4])])
    out, rc = _stopped_mid_shape(tmp_path, b, GEMMS[4], *ONE_BLOCK)
    assert rc == 128 + signal.SIGTERM
    text = _log(out, 0).read_text()
    assert "SKIPPED CRASHED GEMM at block-offset 1" in text
    assert "SKIPPED CRASHED GEMM at block-offset 4" not in text
    assert "STOPPED at block-offset 4" in text
    unit = _unit(out, "block_0000_lines1-10")
    assert (unit["next_offset"], unit["skipped"]) == (4, {"1": 134})
    assert unit["stopped"]
    b = bench()
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert rc == 0 and summary["lost"] == 0
    assert (summary["skipped"], summary["recovered"]) == (1, 1)
    later = [s for pid, s in sorted(_yamls(tmp_path).items()) if pid not in before]
    assert later == [SHAPES[4:], [SHAPES[1]]]


def test_an_ac_cycle_whose_canary_holds_goes_on_with_the_baseline(
    tmp_path, monkeypatch, bench
):
    b = bench(hang_once=[t3._key(GEMMS[-2])])
    _killed_mid_shape(tmp_path, b, GEMMS[-2], *TWO_BLOCKS, canary=True)
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *TWO_BLOCKS, canary=True)
    assert rc == 0 and summary["lost"] == 0
    assert summary["canary"]["rebenched"] == [] and not (out / s03.DRIFT_DIR).exists()
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert [s[0] for s in later] == [SHAPES[-2]]
    (verdict,) = summary["canary"]["boundaries"]
    assert (
        verdict["held"] and verdict["segment"] == 1 and verdict["gpus"] == ["device0"]
    )
    run2 = _journal(out)["runs"][1]["devices"]["0"]
    assert run2["segment"] == 1 and run2["canary"]["held"]
    for block in (0, 1):
        assert {h["segment"] for _, h in _headers(_log(out, block))} == {"1"}
    assert {h["run"] for _, h in _headers(_log(out, 1))} == {"1", "2"}
    assert sorted(_complete(_log(out, 0)) + _complete(_log(out, 1))) == sorted(SHAPES)


def test_an_ac_cycle_whose_canary_moved_benches_the_interrupted_block_again(
    tmp_path, monkeypatch, bench
):
    last = GEMMS[-2]
    slow = [{"shape": t3._key(last), "factor": 3.0, "processes": [1]}]
    b = bench(hang_once=[t3._key(last)], slow=slow)
    _killed_mid_shape(tmp_path, b, last, *TWO_BLOCKS, canary=True)
    before = _yamls(tmp_path)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *TWO_BLOCKS, canary=True)
    assert rc == 0 and summary["lost"] == 0
    name = "block_0001_lines6-10"
    assert summary["canary"]["rebenched"] == [name]
    (verdict,) = summary["canary"]["boundaries"]
    assert not verdict["held"] and verdict["segment"] == 1
    (d,) = (out / s03.DRIFT_DIR).iterdir()
    assert d.name == f"{name}_gpu0_b2"
    rec = json.loads((d / s03.DRIFT_RECORD).read_text())
    assert rec["block"] == name and rec["segment"] == 2
    assert rec["boundary"]["reference"] == verdict["reference"]
    assert {r["run"] for r in rec["readings"]} == {2}
    assert "CANARY BOUNDARY" in (d / f"{name}_gpu0.log").read_text()
    later = [s for pid, s in sorted(_yamls(tmp_path).items()) if pid not in before]
    assert later == [SHAPES[8:], SHAPES[5:]]
    run2 = _journal(out)["runs"][1]["devices"]["0"]
    assert (run2["first_segment"], run2["segment"]) == (1, 2)
    assert not run2["canary"]["held"]
    assert {h["segment"] for _, h in _headers(_log(out, 0))} == {"1"}
    assert {(h["run"], h["segment"]) for _, h in _headers(_log(out, 1))} == {("2", "2")}
    assert _complete(_log(out, 1)) == SHAPES[5:]


def test_a_resume_on_another_gpu_of_the_model_goes_on_there(
    tmp_path, monkeypatch, bench
):
    b = bench(hang_once=[t3._key(GEMMS[-2])])
    _killed_mid_shape(tmp_path, b, GEMMS[-2], *TWO_BLOCKS, canary=True)
    before = set(_yamls(tmp_path))
    rc, out, summary = _resume(
        monkeypatch, tmp_path, b, "--devices", "1", "--blocks-per-gpu", 2, canary=True
    )
    assert rc == 0 and summary["lost"] == 0
    logs = sorted(p.name for p in (out / "logs").glob("block_*.log"))
    assert logs == ["block_0000_lines1-5_gpu0.log", "block_0001_lines6-10_gpu1.log"]
    (verdict,) = summary["canary"]["boundaries"]
    assert (
        verdict["held"]
        and verdict["gpus"] == ["device0"]
        and verdict["gpu"] == "device1"
    )
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert [s[0] for s in later] == [SHAPES[-2]]
    assert _complete(_log(out, 1)) == SHAPES[5:]
    devices = [set(r["devices"]) for r in _journal(out)["runs"]]
    assert devices == [{"0"}, {"1"}]
    assert {h["segment"] for _, h in _headers(_log(out, 1))} == {"1"}


def test_a_resume_on_another_gpu_model_is_refused_and_keeps_the_data(
    tmp_path, monkeypatch, bench
):
    b = bench(hang_once=[t3._key(GEMMS[2])])
    out = _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK)
    logs = {p.name: p.read_bytes() for p in (out / "logs").glob("block_*.log")}
    model = {"gfx_target_version": 1, "device_id": 2, "cu_count": 3}
    monkeypatch.setattr(s03.bench_resume, "gpu_model", lambda gpu: model)
    argv, _ = _argv(tmp_path, b, *ONE_BLOCK)
    monkeypatch.setattr(sys, "argv", ["stage03", *argv])
    assert s03.main() == 2
    assert {p.name: p.read_bytes() for p in (out / "logs").glob("block_*.log")} == logs
    journal = _journal(out)
    assert journal["end"] is None and len(journal["runs"]) == 1


def test_a_stage_that_passed_is_kept_with_keep_passed_and_benched_anew_without(
    tmp_path, monkeypatch, bench
):
    b = bench()
    rc, out, _ = t3._main(monkeypatch, tmp_path, b, *ONE_BLOCK, "--keep-passed")
    assert rc == 0
    log = _log(out, 0).read_bytes()
    n = len(_yamls(tmp_path))
    rc, out, _ = t3._main(monkeypatch, tmp_path, b, *ONE_BLOCK, "--keep-passed")
    assert rc == 0 and len(_yamls(tmp_path)) == n
    assert _log(out, 0).read_bytes() == log and len(_journal(out)["runs"]) == 1
    rc, out, summary = t3._main(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert rc == 0 and summary["lost"] == 0 and len(_yamls(tmp_path)) == n + 1
    (aside,) = (out / bench_resume.SUPERSEDED).iterdir()
    assert (aside / "logs" / _log(out, 0).name).read_bytes() == log
    assert len(_journal(out)["runs"]) == 1


def test_a_stage_that_failed_goes_on_and_benches_again_only_what_failed(
    tmp_path, monkeypatch, bench
):
    """A run that failed for a crash block keeps what it measured: the
    block goes on after its last finished shape and its crashed shapes go
    to the retry pass, with the crash-block count started anew."""
    b = bench(crash_silent=[t3._key(g) for g in GEMMS[:3]])
    extra = (*ONE_BLOCK, "--max-crash-blocks", 0, "--keep-passed")
    rc, out, summary = t3._main(monkeypatch, tmp_path, b, *extra, "--no-retry-skipped")
    assert rc == 2 and summary["crash_blocks"] == ["block_0000_lines1-10"]
    assert _journal(out)["end"]["rc"] == 2
    log = _log(out, 0).read_bytes()
    b = bench()
    n = len(_yamls(tmp_path))
    rc, out, summary = t3._main(monkeypatch, tmp_path, b, *extra)
    assert rc == 0 and summary["lost"] == 0 and summary["crash_blocks"] == []
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == [
        SHAPES[3:],
        SHAPES[:3],
    ]
    assert _log(out, 0).read_bytes().startswith(log)
    record = summary["after_failure"]
    assert record["rc"] == 2 and record["crash_count_reset"]
    assert record["crash_blocks"] == record["reopened"] == ["block_0000_lines1-10"]
    assert _journal(out)["runs"][1]["after_failure"] == record


MANY = [Gemm(64 + 8 * i, 32 + 4 * i, 128 + 16 * i) for i in range(16)]


def test_after_a_failure_the_blocks_after_the_last_passed_canary_are_benched_again(
    tmp_path, monkeypatch, bench
):
    """A run fails for a crash block whose canary passed, after which a
    block drifts twice (kept, flagged): the resume keeps the blocks up to
    the crash block, which goes on, and benches the flagged block again
    from its start."""
    crashing, flagged = MANY[8:12], MANY[12:16]
    slow = [{"shape": t3._key(flagged[0]), "factor": 3.0, "processes": [0, 1]}]
    extra = ("--devices", "0", "--blocks-per-gpu", 4, "--max-crash-blocks", 0)
    b = bench(slow=slow, crash_silent=[t3._key(g) for g in crashing[:3]])
    rc, out, summary = t3._main(
        monkeypatch, tmp_path, b, *extra, gemms=MANY, canary=True
    )
    assert rc == 2 and summary["crash_blocks"] == ["block_0002_lines9-12"]
    assert summary["canary"]["flagged"] == ["block_0003_lines13-16"]
    b = bench(slow=slow)
    n = len(_yamls(tmp_path))
    rc, out, summary = t3._main(
        monkeypatch, tmp_path, b, *extra, gemms=MANY, canary=True
    )
    assert rc == 0 and summary["lost"] == 0
    record = summary["after_failure"]
    assert record["after_last_passed_canary"] == {"gpu0": ["block_0003_lines13-16"]}
    assert record["crash_blocks"] == ["block_0002_lines9-12"]
    assert sorted(record["reopened"]) == [
        "block_0002_lines9-12",
        "block_0003_lines13-16",
    ]
    shapes = [
        [(g.m, g.n, g.k) for g in gs] for gs in (crashing[3:], flagged, crashing[:3])
    ]
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == shapes
    aside = out / s03.DRIFT_DIR / "block_0003_lines13-16_f2"
    assert (
        "canary passed" in json.loads((aside / s03.DRIFT_RECORD).read_text())["reason"]
    )
    (verdict,) = summary["canary"]["boundaries"]
    assert verdict["held"]


def test_an_ac_cycle_after_which_the_gpu_reads_faster_starts_a_new_segment(
    tmp_path, monkeypatch, bench
):
    """Across an interruption the canary holds a GPU to its baseline on both
    sides: one reading faster beyond the tolerance starts a new segment, its
    interrupted block benched again from its start."""
    last = GEMMS[-2]
    slow = [{"shape": t3._key(GEMMS[0]), "factor": 3.0, "processes": [0]}]
    b = bench(hang_once=[t3._key(last)], slow=slow)
    _killed_mid_shape(tmp_path, b, last, *TWO_BLOCKS, canary=True)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *TWO_BLOCKS, canary=True)
    assert rc == 0 and summary["lost"] == 0
    (verdict,) = summary["canary"]["boundaries"]
    assert not verdict["held"] and verdict["direction"] == "faster"
    (d,) = (out / s03.DRIFT_DIR).iterdir()
    assert d.name == "block_0001_lines6-10_gpu0_b2"
    assert "faster by more than" in (d / "block_0001_lines6-10_gpu0.log").read_text()
    assert _journal(out)["runs"][1]["devices"]["0"]["segment"] == 2


def _stage_code_changed(monkeypatch):
    """This stage's code reads as changed since the run it continues."""
    real = bench_resume.file_digest
    stage = Path(s03.__file__).name
    monkeypatch.setattr(
        s03.bench_resume,
        "file_digest",
        lambda p: "f" * 64 if Path(p).name == stage else real(p),
    )


def test_a_change_of_the_stage_code_starts_anew(tmp_path, monkeypatch, bench):
    b = bench(hang_once=[t3._key(GEMMS[2])])
    _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK)
    _stage_code_changed(monkeypatch)
    n = len(_yamls(tmp_path))
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert rc == 0 and summary["lost"] == 0
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == [SHAPES]
    (aside,) = (out / bench_resume.SUPERSEDED).iterdir()
    assert list((aside / "logs").glob("block_*.log"))


@pytest.mark.parametrize("legacy", [False, True], ids=["recorded", "before_code_field"])
def test_the_operator_lets_a_run_go_on_across_a_code_change(
    tmp_path, monkeypatch, bench, legacy
):
    """--resume-across-code-change with a note continues the run after its
    last finished shape; the journal's run records the change and the note
    and each PROCESS header the code. A journal from before the code was
    recorded apart (`legacy`) is held to its block layout."""
    b = bench(hang_once=[t3._key(GEMMS[2])])
    out = _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK)
    path = out / bench_resume.JOURNAL
    journal = json.loads(path.read_text())
    if legacy:
        del journal["code"]
        path.write_text(json.dumps(journal))
    _stage_code_changed(monkeypatch)
    n = len(_yamls(tmp_path))
    note = "a log line of the stage changed"
    extra = (*ONE_BLOCK, "--resume-across-code-change", "--note", note)
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *extra)
    assert rc == 0 and summary["lost"] == 0
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == [SHAPES[2:]]
    change = _journal(out)["runs"][1]["code_change"]
    assert change == {"from": journal.get("code"), "to": "f" * 64, "note": note}
    codes = [h.get("code") for _, h in _headers(_log(out, 0))]
    assert codes[-1] == "f" * 12 and codes[0] != codes[-1]


def test_a_resume_reads_the_config_by_its_content(tmp_path, monkeypatch, bench):
    """The orchestrator snapshots the config anew for every invocation: a
    resume over the same content at another path continues the run."""
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("bench: {}\n")
    second.write_text("bench: {}\n# another invocation's snapshot\n")
    b = bench(hang_once=[t3._key(GEMMS[2])])
    _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK, "--config-yaml", first)
    n = len(_yamls(tmp_path))
    rc, out, summary = _resume(
        monkeypatch, tmp_path, b, *ONE_BLOCK, "--config-yaml", second
    )
    assert rc == 0 and summary["lost"] == 0
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == [SHAPES[2:]]


def test_a_stage_cut_off_over_other_lines_starts_anew(tmp_path, monkeypatch, bench):
    b = bench(hang_once=[t3._key(GEMMS[2])])
    out = _killed_mid_shape(tmp_path, b, GEMMS[2], *ONE_BLOCK)
    n = len(_yamls(tmp_path))
    rc, out, summary = t3._main(monkeypatch, tmp_path, b, *ONE_BLOCK, gemms=GEMMS[:9])
    assert rc == 0 and summary["lost"] == 0
    assert [s for pid, s in sorted(_yamls(tmp_path).items())][n:] == [SHAPES[:9]]
    (aside,) = (out / bench_resume.SUPERSEDED).iterdir()
    assert list((aside / "logs").glob("block_*.log"))
    assert (aside / bench_resume.JOURNAL).is_file()


def test_a_retry_pass_cut_off_goes_on_after_its_last_finished_shape(
    tmp_path, monkeypatch, bench
):
    k1, k3 = t3._key(GEMMS[1]), t3._key(GEMMS[3])
    b = bench(crash_silent_once=[k1, k3], hang_once=[k3])
    out = _killed_mid_shape(tmp_path, b, GEMMS[3], *ONE_BLOCK)
    assert _journal(out)["end"] is None
    before = _yamls(tmp_path)
    assert before[max(before)] == [SHAPES[1], SHAPES[3]]
    rc, out, summary = _resume(monkeypatch, tmp_path, b, *ONE_BLOCK)
    assert rc == 0 and summary["lost"] == 0 and summary["recovered"] == 2
    later = [s for pid, s in _yamls(tmp_path).items() if pid not in before]
    assert later == [[SHAPES[3]]]
    retry = out / "logs" / "block_retry_gpu0.log"
    assert _complete(retry) == [SHAPES[1], SHAPES[3]]
    assert [h["run"] for _, h in _headers(retry)] == ["1", "2"]


def test_stage04_stamps_each_row_with_its_bench_process(
    dataset, monkeypatch
):  # noqa: F811
    """A row names the GPU, boot, run and segment of the process that benched
    it and the canary's state, from its log's PROCESS header; a log from
    before the headers gives none."""
    log = dataset / "logs" / "block_retry_gpu0.log"
    head = (
        "*** PROCESS gpu_uuid=GPU-aa gpu_bdf=0000:01:00.0 boot_id=bb run=2 "
        "segment=3 canary=on\n"
    )
    log.write_text(head + log.read_text())
    assert run_stage04(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    rows = _rows(dataset / "out")
    fields = ("gpu_uuid", "gpu_bdf", "boot_id", "run", "segment", "canary_state")
    stamped = {tuple(r[f] for f in fields) for r in rows[_gemm(GC)]}
    assert stamped == {("GPU-aa", "0000:01:00.0", "bb", "2", "3", "judged")}
    others = {r["boot_id"] for g, rs in rows.items() if g != _gemm(GC) for r in rs}
    assert others <= {"", None}
