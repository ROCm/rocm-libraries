# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host decisions of the per-GPU conv sweep runner, with the devices stubbed.

What needs GPUs is the sweeping; what needs testing is how the work is split,
how a freed card picks up the next shard, and whether the two engines' rows
actually join -- all host-side, so ``Popen`` is replaced by a fake and no
benchmark is run.

The join is the load-bearing assumption of the whole table: the two sweeps
describe a shape with different types, and only
:func:`benchmarks.common.early_stop.case_key` makes them meet. If that ever
stopped agreeing, the merge would silently report every case twice, once per
engine, instead of once with a winner.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from benchmarks.common import case_results, run_conv_shards
from benchmarks.common.early_stop import case_key
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_direct_grouped import DirectConvProblem


# --------------------------------------------------------------------------- split
def test_every_case_lands_in_exactly_one_shard():
    cases = [f"convbfp16 -n {i}" for i in range(23)]
    dealt = run_conv_shards.split_cases(cases, 4)
    assert len(dealt) == 4
    flat = [case for shard in dealt for case in shard]
    assert sorted(flat) == sorted(cases)
    assert len(flat) == len(set(flat))


def test_shards_are_balanced_in_count():
    dealt = run_conv_shards.split_cases([str(i) for i in range(100)], 8)
    sizes = {len(shard) for shard in dealt}
    assert max(sizes) - min(sizes) <= 1


def test_round_robin_spreads_a_grouped_file():
    """A case file is grouped by model and resolution, so a contiguous split
    would hand one shard every expensive shape. Dealing round-robin is what
    keeps the cost mixed, and that is the reason this split was chosen."""
    cheap, pricey = ["cheap"] * 50, ["pricey"] * 50
    dealt = run_conv_shards.split_cases(cheap + pricey, 4)
    for shard in dealt:
        assert shard.count("pricey") == pytest.approx(len(shard) / 2, abs=1)


def test_more_shards_than_cases_drops_the_empty_ones():
    """A shard with no case would read the whole kernel cache for nothing."""
    dealt = run_conv_shards.split_cases(["a", "b"], 8)
    assert dealt == [["a"], ["b"]]


def test_comments_and_blanks_are_not_cases(tmp_path):
    f = tmp_path / "cases.txt"
    f.write_text("# header\n\nconvbfp16 -n 1\n\n# mid\nconvbfp16 -n 2\n")
    assert run_conv_shards.read_cases(f) == ["convbfp16 -n 1", "convbfp16 -n 2"]


# --------------------------------------------------------------------------- devices
@pytest.mark.parametrize(
    "spec,env,expected",
    [
        ("4", {}, ["0", "1", "2", "3"]),
        ("2,5", {}, ["2", "5"]),
        ("auto", {"HIP_VISIBLE_DEVICES": "3,4"}, ["3", "4"]),
        ("auto", {"ROCR_VISIBLE_DEVICES": "0,1,2"}, ["0", "1", "2"]),
        ("auto", {}, ["0"]),
    ],
)
def test_resolve_devices(spec, env, expected):
    assert run_conv_shards.resolve_devices(spec, env) == expected


def test_resolve_devices_rejects_nothing_to_run_on():
    with pytest.raises(ValueError):
        run_conv_shards.resolve_devices("0")


# --------------------------------------------------------------------------- queue
class _FakeProc:
    """A process that finishes after ``ticks`` polls with ``rc``."""

    def __init__(self, ticks: int, rc: int = 0):
        self.ticks = ticks
        self.rc = rc

    def poll(self):
        if self.ticks > 0:
            self.ticks -= 1
            return None
        return self.rc


def _plan(tmp_path, n):
    return [(i, run_conv_shards.shard_paths(tmp_path, i, 2)) for i in range(n)]


def test_a_freed_card_takes_the_next_shard(tmp_path):
    """The point of dealing more shards than cards: a long shard holds up the
    card running it, not a fixed quarter of the work."""
    starts = []
    durations = {0: 6, 1: 1, 2: 1, 3: 1, 4: 1}

    def popen(cmd, **_kw):
        index = int(cmd[-1])
        starts.append(index)
        return _FakeProc(durations[index])

    done = run_shards_with(tmp_path, 5, popen, starts)
    assert starts == [0, 1, 2, 3, 4], "shards are dealt in order"
    card = {index: device for index, device, _rc, _s in done}
    # Shard 0 holds its card for the whole run; every other shard goes to the
    # other one, one after the next, instead of waiting for a fixed slot.
    assert len(card) == 5
    assert {card[i] for i in (1, 2, 3, 4)} == {"0", "1"} - {card[0]}
    finished = [index for index, _d, _rc, _s in done]
    assert finished.index(0) >= 3, "the slow shard outlasts the quick ones"


def run_shards_with(tmp_path, n, popen, _starts):
    return run_conv_shards.run_shards(
        _plan(tmp_path, n),
        devices=["0", "1"],
        command=lambda index, paths: ["python", str(index)],
        popen=popen,
        poll_seconds=0,
    )


def test_every_shard_runs_on_one_card_at_a_time(tmp_path):
    live = {"now": 0, "max": 0}

    def popen(cmd, **_kw):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])

        class P:
            def poll(_self):
                live["now"] -= 1
                return 0

        return P()

    run_shards_with(tmp_path, 8, popen, [])
    assert live["max"] <= 2


def test_a_failing_shard_does_not_stop_the_others(tmp_path):
    def popen(cmd, **_kw):
        return _FakeProc(0, rc=1 if cmd[-1] == "2" else 0)

    done = run_shards_with(tmp_path, 4, popen, [])
    assert len(done) == 4
    assert {index: rc for index, _d, rc, _s in done} == {0: 0, 1: 0, 2: 1, 3: 0}


def test_the_shard_command_carries_the_user_arguments(tmp_path):
    paths = run_conv_shards.shard_paths(tmp_path, 3, 2)
    cmd = run_conv_shards.shard_command(
        tmp_path / "bench.py", ["--run-from-cache", "/c", "--fast-rank"], paths
    )
    assert "--fast-rank" in cmd
    assert cmd[cmd.index("--miopen-file") + 1] == str(paths["cases"])
    assert cmd[cmd.index("--results") + 1] == str(paths["rows"])


@pytest.mark.parametrize("flag", ["--miopen-file", "--results"])
def test_the_runners_own_flags_are_refused_in_the_forwarded_args(
    tmp_path, flag, capsys
):
    f = tmp_path / "cases.txt"
    f.write_text("convbfp16 -n 1\n")
    rc = run_conv_shards.main(
        ["--miopen-file", str(f), "--devices", "1", "--", flag, "x"]
    )
    assert rc == 2
    assert flag in capsys.readouterr().err


def test_a_jit_run_is_refused_before_any_process_starts(tmp_path, capsys):
    """Without a cache every card would build the same kernels again, and no
    rows would be written -- the runner says so instead of starting N
    processes to watch them all exit 2."""
    f = tmp_path / "cases.txt"
    f.write_text("convbfp16 -n 1\n")
    rc = run_conv_shards.main(["--miopen-file", str(f), "--devices", "2"])
    assert rc == 2
    assert "--run-from-cache" in capsys.readouterr().err
    assert not list(tmp_path.glob("shard-*"))


# --------------------------------------------------------------------------- merge
def _row(engine, key, ms, kernel, tflops=100.0, runners_up=None, timing=None):
    row = {
        "key": key,
        "engine": engine,
        "case": 1,
        "shape": "shape",
        "dtype": "bf16",
        "direction": "fwd",
        "tflops": tflops,
        "ms": ms,
        "gbps": 1.0,
        "kernel": kernel,
        "candidates": 7,
    }
    if runners_up:
        row["runners_up"] = list(runners_up)
    if timing is not None:
        row["timing"] = timing
    return row


def test_the_two_engines_join_on_one_row():
    rows = case_results.merge(
        [_row("direct", "k", 0.5, "d-kernel"), _row("implicit", "k", 1.0, "i-kernel")]
    )
    assert len(rows) == 1
    assert rows[0]["direct_kernel"] == "d-kernel"
    assert rows[0]["implicit_kernel"] == "i-kernel"
    assert rows[0]["winner"] == "direct"
    assert rows[0]["speedup"] == 2.0


def test_a_case_only_one_engine_reached_still_gets_a_row():
    """A shape the direct kernels do not cover is the interesting case, not an
    error -- the table has to say so rather than drop it."""
    rows = case_results.merge([_row("implicit", "k", 1.0, "i-kernel")])
    assert len(rows) == 1
    assert rows[0]["direct_ms"] is None
    assert rows[0]["winner"] is None and rows[0]["speedup"] is None


def test_the_same_case_measured_twice_keeps_the_faster():
    """Shards do not overlap, but a re-run after a killed shard merges with
    the rows it left, so the merge has to be idempotent."""
    rows = case_results.merge(
        [_row("direct", "k", 0.9, "slow"), _row("direct", "k", 0.4, "fast")]
    )
    assert rows[0]["direct_kernel"] == "fast"
    assert (
        case_results.merge(
            [_row("direct", "k", 0.4, "fast"), _row("direct", "k", 0.9, "slow")]
        )
        == rows
    )


def test_the_runner_up_kernels_get_a_column_each():
    """Names only -- what a near-tie costs you is which instance won, not a
    second TFLOPS column that repeats the first."""
    rows = case_results.merge(
        [_row("direct", "k", 0.5, "best", runners_up=["second", "third"])]
    )
    assert rows[0]["direct_kernel"] == "best"
    assert rows[0]["direct_kernel_2"] == "second"
    assert rows[0]["direct_kernel_3"] == "third"


def test_a_shorter_ranking_leaves_the_rest_of_the_columns_empty():
    rows = case_results.merge([_row("implicit", "k", 1.0, "best", runners_up=["x"])])
    assert rows[0]["implicit_kernel_2"] == "x"
    assert rows[0]["implicit_kernel_3"] is None


def test_the_faster_row_brings_its_whole_ranking():
    """A ranking means something only within the run that measured it: the
    slower row goes whole, rather than leaving its runner-ups behind to be read
    as the faster run's."""
    rows = case_results.merge(
        [
            _row("direct", "k", 0.9, "slow", runners_up=["slow-2", "slow-3"]),
            _row("direct", "k", 0.4, "fast", runners_up=["fast-2"]),
        ]
    )
    assert rows[0]["direct_kernel"] == "fast"
    assert rows[0]["direct_kernel_2"] == "fast-2"
    assert rows[0]["direct_kernel_3"] is None


def test_merge_is_ordered_by_key():
    rows = case_results.merge([_row("direct", k, 1.0, "x") for k in ("c", "a", "b")])
    assert [r["key"] for r in rows] == ["a", "b", "c"]


def test_a_partial_last_line_is_skipped_not_fatal(tmp_path):
    """Rows are flushed per case, so a killed shard can leave half a line."""
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps(_row("direct", "k", 1.0, "x")) + '\n{"key": "trunc')
    assert [r["key"] for r in case_results.read_rows(f)] == ["k"]


def test_write_leaves_a_csv_beside_the_jsonl(tmp_path):
    rows = case_results.merge(
        [_row("direct", "k", 0.5, "d"), _row("implicit", "k", 1.0, "i")]
    )
    out = tmp_path / "results.jsonl"
    csv_path = case_results.write(rows, out)
    assert csv_path.is_file() and csv_path != out
    header, body = csv_path.read_text().splitlines()[:2]
    assert header.split(",") == list(case_results.CSV_FIELDS)
    assert "d" in body and "i" in body


def test_the_csv_is_narrower_than_the_json_it_is_written_from(tmp_path):
    """The CSV is the reading view; the JSON keeps the fields it leaves out, so
    nothing is lost by the narrowing."""
    rows = case_results.merge(
        [_row("direct", "k", 0.5, "d"), _row("implicit", "k", 1.0, "i")]
    )
    out = tmp_path / "results.jsonl"
    csv_path = case_results.write(rows, out)
    assert set(case_results.CSV_FIELDS) < set(case_results.MERGED_FIELDS)
    assert "tflops" not in csv_path.read_text().splitlines()[0]
    assert json.loads(out.read_text().splitlines()[0])["direct_tflops"] == 100.0


def test_the_winner_columns_repeat_the_winning_engines_row(tmp_path):
    rows = case_results.merge(
        [_row("direct", "k", 0.5, "d"), _row("implicit", "k", 1.0, "i")]
    )
    assert rows[0]["winner_ms"] == 0.5 and rows[0]["winner_kernel"] == "d"


def test_a_one_sided_case_names_no_winner():
    """One time is not a comparison: the engine that did reach the case is
    already on the row under its own name."""
    rows = case_results.merge([_row("implicit", "k", 1.0, "i")])
    assert rows[0]["winner_ms"] is None and rows[0]["winner_kernel"] is None


# ------------------------------------------------------------------ miopen cmd
_MIOPEN_LINE = (
    "convbfp16 -F 1 -n 4 -c 64 -k 128 -H 320 -W 320 -y 3 -x 3 -p 1 -q 1 "
    "-u 2 -v 2 -l 1 -j 1 -g 1 -m conv -I NHWC -O NHWC -f NHWC"
)


def _key_of(line, direction="fwd"):
    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

    problem, dtype, _forw = parse_miopen_cmd(line)
    return case_key(problem, dtype, direction)


def test_the_case_file_puts_its_own_line_on_the_row(tmp_path):
    """The sweeps key their rows by geometry, so the line the user wrote is not
    in them -- the merge reads it back off the input the run was given."""
    f = tmp_path / "cases.txt"
    f.write_text(f"# a comment\n\n{_MIOPEN_LINE}\n")
    key = _key_of(_MIOPEN_LINE)
    rows = case_results.merge(
        [_row("direct", key, 0.5, "d")],
        case_results.miopen_cmds_by_key(miopen_file=f),
    )
    assert rows[0]["miopen_cmd"] == _MIOPEN_LINE


def test_a_case_with_no_command_behind_it_gets_one_written_from_its_key():
    """A run given shape flags rather than a command still has to leave a row
    you can re-run."""
    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

    key = _key_of(_MIOPEN_LINE)
    rows = case_results.merge([_row("direct", key, 0.5, "d")])
    cmd = rows[0]["miopen_cmd"]
    assert cmd != _MIOPEN_LINE and "MIOpenDriver convbfp16" in cmd
    # It describes the same case: round-tripping it lands back on the key.
    problem, dtype, forw = parse_miopen_cmd(cmd)
    assert forw == 1 and case_key(problem, dtype, "fwd") == key


def test_a_key_the_case_file_does_not_cover_falls_back_to_a_written_command(tmp_path):
    f = tmp_path / "cases.txt"
    f.write_text(_MIOPEN_LINE + "\n")
    rows = case_results.merge(
        [_row("direct", _key_of(_MIOPEN_LINE, "wgrad"), 0.5, "d")],
        case_results.miopen_cmds_by_key(miopen_file=f),
    )
    assert rows[0]["miopen_cmd"].endswith("-F 4 -t 1 -in_layout=NHWC")


def test_a_case_file_line_that_does_not_parse_is_skipped_not_fatal(tmp_path):
    """Writing the results of a finished sweep must not fail over the label on
    a row."""
    f = tmp_path / "cases.txt"
    f.write_text(f"not a MIOpenDriver command\n{_MIOPEN_LINE}\n")
    assert list(case_results.miopen_cmds_by_key(miopen_file=f)) == [
        _key_of(_MIOPEN_LINE)
    ]


def test_a_key_that_does_not_parse_is_left_as_it_is():
    assert case_results.miopen_cmd_from_key("nonsense") == "nonsense"


def test_a_writer_without_a_path_writes_nothing(tmp_path):
    writer = case_results.ResultsWriter(None, "direct")
    writer.row(
        key="k",
        case=1,
        shape="s",
        dtype="bf16",
        direction="fwd",
        tflops=1.0,
        ms=1.0,
        gbps=1.0,
        kernel="x",
        candidates=1,
    )
    writer.close()
    assert writer.n_rows == 0
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- end to end
_STUB = """\
import argparse, json, sys
ap = argparse.ArgumentParser()
ap.add_argument("--miopen-file")
ap.add_argument("--results")
ap.add_argument("--run-from-cache")
args = ap.parse_args()
# --script compare's real shards write already-merged rows (both engines,
# winner, speedup) through case_results._write_results, not one row per
# engine -- this stub mimics that so the runner's own merge of merged rows
# is what the test exercises.
with open(args.results, "w") as out:
    for line in open(args.miopen_file):
        line = line.strip()
        if not line:
            continue
        n = int(line.split()[-1])
        direct_ms, implicit_ms = 1.0 + n, 2.0 + n
        out.write(json.dumps({
            "key": f"fwd/bf16/case{n}", "miopen_cmd": None, "shape": f"s{n}",
            "dtype": "bf16", "direction": "fwd",
            "direct_tflops": 100.0, "direct_ms": direct_ms,
            "direct_kernel": f"direct-kernel-{n}",
            "direct_kernel_2": None, "direct_kernel_3": None,
            "implicit_tflops": 100.0, "implicit_ms": implicit_ms,
            "implicit_kernel": f"implicit-kernel-{n}",
            "implicit_kernel_2": None, "implicit_kernel_3": None,
            "winner": "direct", "winner_ms": direct_ms,
            "winner_kernel": f"direct-kernel-{n}",
            "speedup": round(implicit_ms / direct_ms, 3), "timing": "staged",
        }) + "\\n")
print("stub done")
"""


def test_a_whole_run_splits_schedules_and_merges(tmp_path, monkeypatch, capsys):
    """main() end to end against a stub benchmark: the only thing missing is
    the GPU, and the runner never touches one."""
    stub = tmp_path / "stub_bench.py"
    stub.write_text(_STUB)
    monkeypatch.setattr(run_conv_shards, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run_conv_shards, "SCRIPTS", {"compare": stub.name})
    monkeypatch.setattr(run_conv_shards, "POLL_SECONDS", 0)

    cases = tmp_path / "cases.txt"
    cases.write_text("".join(f"convbfp16 -n {i}\n" for i in range(10)))
    logs = tmp_path / "logs"

    rc = run_conv_shards.main(
        [
            "--miopen-file",
            str(cases),
            "--devices",
            "2",
            "--shards",
            "4",
            "--log-dir",
            str(logs),
            "--",
            "--run-from-cache",
            "/nonexistent",
        ]
    )
    assert rc == 0, capsys.readouterr()

    # Each shard's input is kept, and together they are the original file.
    shard_cases = sorted(logs.glob("shard-*.cases"))
    assert len(shard_cases) == 4
    dealt = [ln for f in shard_cases for ln in f.read_text().split() if ln.isdigit()]
    assert sorted(int(n) for n in dealt) == list(range(10))

    rows = case_results.read_rows(logs / "results.jsonl")
    assert len(rows) == 10
    assert (logs / "results.jsonl.csv").is_file()
    # Direct was given the lower time for every case, so it wins all of them.
    assert all(r["winner"] == "direct" for r in rows)
    assert {r["key"] for r in rows} == {f"fwd/bf16/case{i}" for i in range(10)}
    assert "10 case(s)" in capsys.readouterr().out


def test_a_run_whose_shards_measured_nothing_is_a_failure(tmp_path, monkeypatch):
    """Eight cards and an empty table means the run did not do what was asked;
    exiting 0 would hide that in a nightly job."""
    stub = tmp_path / "stub_bench.py"
    stub.write_text("import sys; open(sys.argv[sys.argv.index('--results')+1],'w')")
    monkeypatch.setattr(run_conv_shards, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run_conv_shards, "SCRIPTS", {"compare": stub.name})
    monkeypatch.setattr(run_conv_shards, "POLL_SECONDS", 0)
    cases = tmp_path / "cases.txt"
    cases.write_text("convbfp16 -n 1\n")
    rc = run_conv_shards.main(
        [
            "--miopen-file",
            str(cases),
            "--devices",
            "1",
            "--log-dir",
            str(tmp_path / "logs"),
            "--",
            "--run-from-cache",
            "/c",
        ]
    )
    assert rc != 0


# --------------------------------------------------------------------------- resume
def test_only_this_engines_rows_are_skipped():
    """A run killed between the direct sweep and the implicit one has the case
    on one side only; resuming must finish the other, not skip it."""
    rows = [_row("direct", "k", 1.0, "d"), _row("implicit", "other", 1.0, "i")]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "rows.jsonl"
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert case_results.measured_keys(f, "direct") == {"k"}
        assert case_results.measured_keys(f, "implicit") == {"other"}
    assert case_results.measured_keys(None, "direct") == set()


def test_resume_after_the_separator_is_caught_here(tmp_path, capsys):
    """argparse.REMAINDER swallows anything after --, so a misplaced runner
    flag would reach the child and be reported by the wrong script."""
    f = tmp_path / "cases.txt"
    f.write_text("convbfp16 -n 1\n")
    rc = run_conv_shards.main(
        [
            "--miopen-file",
            str(f),
            "--devices",
            "1",
            "--",
            "--run-from-cache",
            "/c",
            "--resume",
        ]
    )
    assert rc == 2
    assert "goes before --" in capsys.readouterr().err


def test_resume_carries_the_earlier_rows_and_skips_their_cases(
    tmp_path, monkeypatch, capsys
):
    """Second pass: the stub records which cases it was actually given, and
    the merged table still holds the first pass's rows."""
    stub = tmp_path / "stub_bench.py"
    stub.write_text(_RESUME_STUB)
    monkeypatch.setattr(run_conv_shards, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run_conv_shards, "SCRIPTS", {"compare": stub.name})
    monkeypatch.setattr(run_conv_shards, "POLL_SECONDS", 0)
    cases = tmp_path / "cases.txt"
    cases.write_text("".join(f"convbfp16 -n {i}\n" for i in range(6)))
    logs = tmp_path / "logs"
    runner = [
        "--miopen-file",
        str(cases),
        "--devices",
        "1",
        "--shards",
        "2",
        "--log-dir",
        str(logs),
    ]
    forwarded = ["--", "--run-from-cache", "/c"]
    argv = runner + forwarded

    # First pass: the stub records cases 0-2 and then "dies" on the rest.
    monkeypatch.setenv("STUB_LIMIT", "2")
    run_conv_shards.main(argv)
    first = {r["key"] for r in case_results.read_rows(logs / "results.jsonl")}
    assert first and len(first) < 6

    # Second pass with --resume: the earlier cases are handed to the sweep as
    # already measured, and only the rest are run.
    monkeypatch.delenv("STUB_LIMIT")
    (logs / "ran.json").unlink()  # the stub appends; this pass's cases only
    capsys.readouterr()
    rc = run_conv_shards.main(runner + ["--resume"] + forwarded)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "row(s) carried" in out

    rows = case_results.read_rows(logs / "results.jsonl")
    assert {r["key"] for r in rows} == {f"fwd/bf16/case{i}" for i in range(6)}
    # Nothing measured twice, and the first pass's rows survived.
    assert len(rows) == 6
    ran = json.loads((logs / "ran.json").read_text())
    assert not (set(ran) & first), f"re-measured {set(ran) & first}"


_RESUME_STUB = """\
import argparse, json, os, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("--miopen-file")
ap.add_argument("--results")
ap.add_argument("--skip-measured")
ap.add_argument("--run-from-cache")
args = ap.parse_args()
done = set()
if args.skip_measured:
    for line in open(args.skip_measured):
        done.add(json.loads(line)["key"])
limit = int(os.environ.get("STUB_LIMIT", "999"))
ran = []
rows = []
for n_done, line in enumerate(open(args.miopen_file)):
    line = line.strip()
    if not line:
        continue
    n = int(line.split()[-1])
    key = f"fwd/bf16/case{n}"
    if key in done:
        continue
    if n_done >= limit:
        break
    ran.append(key)
    rows.append({
        "key": key, "miopen_cmd": None, "shape": f"s{n}", "dtype": "bf16",
        "direction": "fwd", "direct_tflops": 1.0, "direct_ms": 1.0,
        "direct_kernel": f"direct-{n}", "direct_kernel_2": None,
        "direct_kernel_3": None, "implicit_tflops": 1.0, "implicit_ms": 2.0,
        "implicit_kernel": f"implicit-{n}", "implicit_kernel_2": None,
        "implicit_kernel_3": None, "winner": "direct", "winner_ms": 1.0,
        "winner_kernel": f"direct-{n}", "speedup": 2.0, "timing": "staged",
    })
with open(args.results, "w") as out:
    for row in rows:
        out.write(json.dumps(row) + "\\n")
log = pathlib.Path(args.results).parent / "ran.json"
prev = json.loads(log.read_text()) if log.is_file() else []
log.write_text(json.dumps(prev + ran))
"""


# --------------------------------------------------------------------------- progress
def _rows_file(path, keys, engines=("direct", "implicit")):
    path.write_text(
        "".join(
            json.dumps(_row(engine, key, 1.0, f"{engine}-{key}")) + "\n"
            for key in keys
            for engine in engines
        )
    )
    return path


def test_a_case_both_engines_measured_counts_once(tmp_path):
    """The row files hold one row per engine; progress is in cases."""
    a = _rows_file(tmp_path / "shard-0.jsonl", ["k1", "k2"])
    b = _rows_file(tmp_path / "shard-1.jsonl", ["k3"], engines=("implicit",))

    assert run_conv_shards.measured_keys([a, b]) == {"k1", "k2", "k3"}


def test_measured_keys_ignores_a_shard_that_has_written_nothing(tmp_path):
    """Every shard's row file is read on each tick, including the ones whose
    card has not reached its first case yet."""
    a = _rows_file(tmp_path / "shard-0.jsonl", ["k1"])

    assert run_conv_shards.measured_keys([a, tmp_path / "absent.jsonl"]) == {"k1"}


def test_the_progress_line_reports_the_run(tmp_path):
    line = run_conv_shards.progress_line(
        measured=142,
        total=4218,
        baseline=0,
        shards_done=7,
        shards=22,
        busy=8,
        devices=8,
        elapsed=12 * 60,
    )
    assert "142/4218 cases (3%)" in line
    assert "7/22 shards" in line and "8/8 busy" in line
    # 142 cases in 12 min leaves 4076 -- hours, so the ETA says hours.
    assert "ETA 5.7 h" in line


def test_a_resumed_run_does_not_extrapolate_from_carried_cases(tmp_path):
    """4000 cases carried in by --resume took this run no time. Dividing the
    elapsed minute by them would promise the remaining 218 in seconds."""
    line = run_conv_shards.progress_line(
        measured=4010,
        total=4218,
        baseline=4000,
        shards_done=0,
        shards=22,
        busy=8,
        devices=8,
        elapsed=60,
    )
    # 10 cases in a minute, 208 to go: ~21 min, not the ~3 s the carried rows
    # would imply.
    assert "ETA 21 min" in line


def test_the_progress_line_omits_an_eta_it_cannot_compute(tmp_path):
    """Nothing measured yet: no rate, so no guess."""
    line = run_conv_shards.progress_line(
        measured=0,
        total=4218,
        baseline=0,
        shards_done=0,
        shards=22,
        busy=8,
        devices=8,
        elapsed=30,
    )
    assert "0/4218 cases (0%)" in line
    assert "ETA" not in line


def test_progress_is_reported_while_the_shards_run(tmp_path):
    """The point of the line: the console says something between one shard
    finishing and the next, which on a long sweep is a quarter of an hour."""
    ticks = []

    def popen(cmd, **_kw):
        return _FakeProc(3)

    run_conv_shards.run_shards(
        _plan(tmp_path, 4),
        devices=["0", "1"],
        command=lambda index, paths: ["python", str(index)],
        popen=popen,
        poll_seconds=0,
        on_progress=lambda shards_done, busy: ticks.append((shards_done, busy)),
        progress_seconds=0,
    )

    assert ticks, "no progress was reported"
    # Reported while cards were still working, not only at the end.
    assert any(busy > 0 for _done, busy in ticks)
    assert ticks[-1][0] == 4, "the last tick sees every shard finished"


# --------------------------------------------------------------------------- compare
def test_compare_merges_one_engine_when_the_other_was_skipped(tmp_path, capsys):
    """--skip-direct leaves no direct rows; the table is then one-sided rather
    than missing, and the driver must not trip over the absent file."""
    from benchmarks.common import benchmark_conv_compare as compare

    implicit = tmp_path / "implicit.jsonl"
    implicit.write_text(json.dumps(_row("implicit", "k", 1.0, "i-kernel")) + "\n")
    out = tmp_path / "merged.jsonl"

    compare._write_results(str(out), str(tmp_path / "absent.jsonl"), str(implicit))

    rows = case_results.read_rows(out)
    assert len(rows) == 1 and rows[0]["implicit_kernel"] == "i-kernel"
    assert "results:" in capsys.readouterr().out


def test_compare_says_so_when_there_is_nothing_to_write(tmp_path, capsys):
    from benchmarks.common import benchmark_conv_compare as compare

    out = tmp_path / "merged.jsonl"
    compare._write_results(str(out), str(tmp_path / "a"), str(tmp_path / "b"))
    assert not out.exists()
    assert "no per-case results" in capsys.readouterr().err


def _compare_argv(cache) -> list:
    """The smallest argv that gets ``main`` to the sweep it will stub out."""
    return ["x", "--run-from-cache", str(cache), "--skip-direct"]


def test_compare_fails_when_a_sweep_dies(tmp_path, monkeypatch, capsys):
    """A sweep that crashed on every case prints no ``Best:`` line -- but then
    neither does a shape none of the cached kernels support. The exit status is
    the only thing that separates them, and ``run_conv_shards.py`` reports it
    per shard."""
    from benchmarks.common import benchmark_conv_compare as compare

    monkeypatch.setattr(compare, "_run_script", lambda *a, **k: (None, "", "boom", 1))
    monkeypatch.setattr(sys, "argv", _compare_argv(tmp_path))

    assert compare.main() == 1
    assert "exited 1" in capsys.readouterr().err


def test_compare_turns_a_killed_sweep_into_a_usable_status(tmp_path, monkeypatch):
    """A child killed by a signal reports a negative status. Returned verbatim
    it would be truncated to an arbitrary byte -- zero among them, which is the
    one value that must not come out of a failed run."""
    from benchmarks.common import benchmark_conv_compare as compare

    monkeypatch.setattr(compare, "_run_script", lambda *a, **k: (None, "", "", -9))
    monkeypatch.setattr(sys, "argv", _compare_argv(tmp_path))

    assert compare.main() == 1


def test_compare_succeeds_when_the_sweep_does(tmp_path, monkeypatch):
    """The other half of the status: a healthy sweep still exits zero, so the
    shard runner does not start reporting every shard as failed."""
    from benchmarks.common import benchmark_conv_compare as compare

    monkeypatch.setattr(compare, "_run_script", lambda *a, **k: (1.0, "kern", "", 0))
    monkeypatch.setattr(sys, "argv", _compare_argv(tmp_path))

    assert compare.main() == 0


def test_the_comparison_driver_needs_no_torch():
    """conv_compare is a launcher: it has to run where torch is not installed
    (the benchmark scripts it spawns import it, it does not)."""
    import subprocess

    probe = (
        "import sys; sys.argv=['x'];"
        "import benchmarks.common.benchmark_conv_compare;"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------- the key
def test_the_two_problem_types_meet_on_one_key():
    """The assumption the merged table rests on: the same convolution, read by
    the direct sweep and by the implicit-GEMM sweep, keys the same.

    Their ``short()`` strings do not even carry the same fields, so this is the
    only thing that can join them -- and it is the key ``--early-stop-seed``
    already uses to carry a direct time into the implicit sweep.
    """
    direct = DirectConvProblem(
        N=16, H=28, W=28, groups=4, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
    )
    implicit = ConvProblem(
        N=16, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4
    )
    assert direct.short() != implicit.short()
    assert case_key(direct, "bf16", "fwd") == case_key(implicit, "bf16", "fwd")


def test_a_different_shape_keys_differently():
    a = DirectConvProblem(N=16, H=28, W=28, groups=4, cpg=16, kpg=16)
    b = DirectConvProblem(N=16, H=28, W=28, groups=4, cpg=16, kpg=16, stride=2)
    assert case_key(a, "bf16", "fwd") != case_key(b, "bf16", "fwd")
    assert case_key(a, "bf16", "fwd") != case_key(a, "fp16", "fwd")
    assert case_key(a, "bf16", "fwd") != case_key(a, "bf16", "dgrad")


# --------------------------------------------------------------------------- timing
def test_a_row_without_timing_is_a_block_row():
    """Rows written before the field existed were all timed back to back."""
    assert case_results.row_timing(_row("direct", "k", 1.0, "d")) == "block"
    assert case_results.row_timing(_row("direct", "k", 1.0, "d", timing="staged")) == (
        "staged"
    )


def test_the_writer_records_how_the_row_was_timed(tmp_path):
    f = tmp_path / "rows.jsonl"
    with case_results.ResultsWriter(str(f), "implicit", "staged") as writer:
        writer.row(
            key="k",
            case=1,
            shape="s",
            dtype="bf16",
            direction="fwd",
            tflops=1.0,
            ms=1.0,
            gbps=1.0,
            kernel="x",
            candidates=1,
        )
    assert [r["timing"] for r in case_results.read_rows(f)] == ["staged"]
    with pytest.raises(ValueError, match="timing"):
        case_results.ResultsWriter(None, "direct", "events")


def test_resume_skips_only_cases_timed_the_same_way(tmp_path):
    """A case measured back to back is not measured for a staged run."""
    f = tmp_path / "rows.jsonl"
    rows = [
        _row("direct", "old", 1.0, "d"),
        _row("direct", "new", 1.0, "d", timing="staged"),
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert case_results.measured_keys(f, "direct") == {"old"}
    assert case_results.measured_keys(f, "direct", "staged") == {"new"}


def test_a_merge_takes_one_timing():
    """The faster-row rule must not pick the cheaper protocol over the faster
    kernel: a staged row of a case is never replaced by its block row."""
    rows = [
        _row("direct", "k", 1.0, "block-kernel"),
        _row("direct", "k", 1.2, "staged-kernel", timing="staged"),
    ]
    merged = case_results.merge(rows, timing="staged")
    assert merged[0]["direct_kernel"] == "staged-kernel"
    assert merged[0]["timing"] == "staged"
    with pytest.raises(ValueError, match="different timings"):
        case_results.merge(rows)


def test_a_merge_of_one_timing_needs_no_argument():
    merged = case_results.merge([_row("direct", "k", 1.0, "d", timing="staged")])
    assert merged[0]["timing"] == "staged"
    assert case_results.merge([_row("direct", "k", 1.0, "d")])[0]["timing"] == "block"


@pytest.mark.parametrize(
    "forwarded, timing",
    [
        ([], "staged"),
        (["--fast-rank"], "staged"),
        (["--timing", "block"], "block"),
        (["--fast-rank", "--timing=block"], "block"),
        (["--timing", "block", "--timing", "staged"], "staged"),
    ],
)
def test_the_runner_reads_the_timing_it_forwards(forwarded, timing):
    assert run_conv_shards.forwarded_timing(forwarded) == timing
