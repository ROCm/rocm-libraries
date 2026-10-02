# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import json

import pytest

from Tensile import JitDebug


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value, categories", [
    ("timing", {"timing"}), (" Progress ", {"progress"}), ("all", {"timing", "progress"}),
    ("timing,all", {"timing", "progress"}), (",timing,,", {"timing"}),
    ("progress,timing", {"timing", "progress"}),
])
def test_categories_are_names_or_all(value, categories):
    assert JitDebug.parseCategories(value) == categories


@pytest.mark.parametrize("value", ["0", "1", "bogus", "timing,bogus", "", " , "])
def test_numbers_and_unknown_names_are_rejected(value):
    with pytest.raises(ValueError):
        JitDebug.parseCategories(value)


def parse(argv):
    parser = argparse.ArgumentParser()
    JitDebug.addArguments(parser)
    args = vars(parser.parse_args(argv))
    return JitDebug.fromArguments(parser, args, module="m", mode="explicit"), args


def test_cli_options_are_private_and_removed_from_the_arguments(tmp_path, capsys):
    recorder, args = parse([])
    assert recorder is JitDebug.NULL and args == {}
    recorder, args = parse(["--debug", "all", "--debug-dir", str(tmp_path / "d")])
    assert recorder.timing and recorder.progress and args == {}
    assert (tmp_path / "d").is_dir()
    parser = argparse.ArgumentParser()
    JitDebug.addArguments(parser)
    assert "--debug" not in parser.format_help()
    with pytest.raises(SystemExit):
        parse(["--debug-dir", str(tmp_path)])
    assert "--debug-dir requires --debug" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse(["--debug", "1"])
    assert "unknown category '1'" in capsys.readouterr().err


def test_null_recorder_reads_no_clock(monkeypatch):
    def forbidden():
        raise AssertionError("a clock was read")

    for name in ("perf_counter_ns", "monotonic_ns", "process_time_ns"):
        monkeypatch.setattr(JitDebug.time, name, forbidden)
    null = JitDebug.NULL
    with null.span("setup") as span:
        pass
    assert span == {}
    null.event("stage")
    null.request(requested=1)
    null.candidate(id=1)
    null.bundle(0)
    null.published(1)
    null.finish("ok")


def events(directory):
    return [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]


def test_progress_events_are_versioned_ordered_and_flushed(tmp_path):
    recorder = JitDebug.Recorder({"progress"}, tmp_path, module="m", mode="prediction")
    recorder.request(requested=2, candidates=3)
    assert len(events(tmp_path)) == 1
    recorder.bundle(0)
    with recorder.span("select"):
        recorder.candidate(index=0, of=3, id=7, outcome="rejected", reason="tensile", ns=5)
    recorder.published(1)
    recorder.finish("ok")
    lines = events(tmp_path)
    assert [line["seq"] for line in lines] == list(range(1, 6))
    assert all(line["v"] == 1 and line["pid"] > 0 and line["mono_ns"] > 0 for line in lines)
    assert [line["kind"] for line in lines] == ["request", "stage", "candidate", "stage", "done"]
    assert lines[0] == {**lines[0], "module": "m", "mode": "prediction", "requested": 2,
                        "candidates": 3}
    assert lines[1] == {**lines[1], "stage": "select", "phase": "start", "rank": 0}
    assert lines[3]["phase"] == "end" and lines[3]["status"] == "ok" and "ns" not in lines[3]
    assert "ns" not in lines[2] and lines[2]["reason"] == "tensile" and lines[2]["rank"] == 0
    assert lines[4] == {**lines[4], "status": "ok", "bundles_published": 1, "requested": 2}
    assert "error_type" not in lines[4]
    assert not (tmp_path / "timing.json").exists()


class Rejected(Exception):
    pass


def test_timing_spans_nest_and_record_their_status(tmp_path):
    recorder = JitDebug.Recorder({"timing"}, tmp_path, module="m", mode="explicit")
    with recorder.span("select", rejects=(Rejected,)):
        with pytest.raises(Rejected):
            with recorder.span("derive", stage=False, rejects=(Rejected,), candidate=3):
                raise Rejected()
        recorder.candidate(index=0, of=1, id=3, outcome="rejected", reason="tensile", ns=9)
    with pytest.raises(OSError):
        with recorder.span("publish"):
            raise OSError("full")
    recorder.finish("failed", "OSError")
    assert not (tmp_path / "events.jsonl").exists()
    assert [path.name for path in tmp_path.iterdir()] == ["timing.json"]
    timing = json.loads((tmp_path / "timing.json").read_text())
    assert timing | {"v": 1, "producer": "python", "module": "m", "mode": "explicit",
                     "categories": ["timing"], "clock": "perf_counter_ns", "status": "failed",
                     "error_type": "OSError", "bundles_published": 0, "cpu_threads": 1} == timing
    assert timing["total_ns"] >= timing["totals"]["select"] >= timing["totals"]["derive"] > 0
    assert [(s["id"], s["parent"], s["name"], s["status"]) for s in timing["spans"]] == [
        (1, None, "select", "ok"), (2, 1, "derive", "rejected"), (3, None, "publish", "failed")]
    assert timing["spans"][1]["candidate"] == 3 and timing["spans"][1]["rank"] is None
    assert timing["candidates"] == [
        {"index": 0, "of": 1, "id": 3, "outcome": "rejected", "reason": "tensile", "ns": 9}]
    assert timing["cpu_ns"] >= 0 and timing["children_cpu_ns"] >= 0


def test_without_a_directory_output_goes_to_stderr(tmp_path, capsys):
    recorder = JitDebug.Recorder({"timing", "progress"}, None, module="m", mode="explicit")
    with recorder.span("setup"):
        print("generator output")
    recorder.finish("ok")
    captured = capsys.readouterr()
    assert captured.out == "generator output\n"
    lines = captured.err.splitlines()
    assert lines[0] == "progress: setup start"
    assert lines[1].startswith("progress: setup end ok ") and lines[1].endswith(" ms")
    assert lines[2].startswith("progress: done module=m mode=explicit status=ok")
    assert lines[3].startswith("timing: m ok: total ") and lines[4].startswith("timing:   setup")
    assert not list(tmp_path.iterdir())


def test_recording_failures_never_fail_the_command(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("")
    recorder = JitDebug.Recorder({"timing", "progress"}, blocked / "d", module="m", mode="explicit")
    assert not recorder.timing and not recorder.progress
    with recorder.span("setup"):
        pass
    recorder.finish("ok")
    removed = tmp_path / "removed"
    recorder = JitDebug.Recorder({"timing", "progress"}, removed, module="m", mode="explicit")
    for path in removed.iterdir():
        path.unlink()
    removed.rmdir()
    recorder.event("stage", stage="setup", phase="start")
    recorder.finish("ok")
    assert not removed.exists()
