# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host tests of the gfx950 GDN/KDA decode table sweep (no GPU).

The GPU stages run against a fake driver whose timers return chosen samples,
so selection, ordering and aggregation are checked exactly.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from types import SimpleNamespace

import pytest

from ._gdn_gate_support import BAD_VERDICTS
from benchmarks.gfx950.gdn import decode_table_sweep as sweep
from dispatch.gdn import dispatch_gdn_decode
from dispatch.gdn.gate_rules import Gate
from rocke.dispatch.core import PinRefused

SHAPE = sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, 1)

# The run settings of a stage-1 file: its walk and its cold-memory rotation.
_WALK = dict(sweep_level="production", tuning_sample=0, seed=0, bases_only=False)
_META = dict(_WALK, rotate_mb=1024, cold_target_mb=1024, sha="abc", device="gfx950")


def _parse(*argv):
    ap = sweep.build_parser()
    args = ap.parse_args(list(argv))
    sweep.check_args(ap, args)
    return args


def _refused(capsys, flag: str, *argv) -> None:
    """``argv`` is refused as a usage error (exit 2) whose message names ``flag``."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as err:
        _parse(*argv)
    assert err.value.code == 2, argv
    assert flag in capsys.readouterr().err, argv


# ---------------------------------------------------------------- CLI
def test_cli_defaults_and_value_parsers(capsys):
    args = _parse()
    assert (args.rounds, args.s2_rounds, args.runs, args.top_k) == (5, 30, 3, 8)
    assert args.rotate_mb is None  # resolved to the driver's ROTATE_MB at run time
    assert args.sweep_level == "production" and args.shard == (0, 1)
    args = _parse(
        "--heads",
        "16/32,4/8",
        "--batches",
        "1,8",
        "--shard",
        "2/4",
        "--stage1-only",
        "--tuning-ids",
        "a,b",
    )
    assert args.heads == ((16, 32), (4, 8))
    assert args.batches == (1, 8)
    assert args.shard == (2, 4)
    assert args.tuning_ids == ("a", "b")
    for bad in (("--shard", "4/4"), ("--heads", "16"), ("--batches", ",")):
        _refused(capsys, bad[0], *bad)


def test_cli_refuses_a_partial_stage_two(capsys):
    """A window sees part of each shape, so ranking it for stage 2 is refused."""
    for window in (("--shard", "1/2"), ("--offset", "3"), ("--limit", "5")):
        _refused(capsys, "--stage1-only", *window)
        _parse(*window, "--stage1-only")
        _parse(*window, "--list-only")
    _refused(capsys, "--stage1-only", "--stage1-json", "x.json", "--stage1-only")
    for walk in (("--bases-only",), ("--tuning-ids", "a"), ("--sweep-level", "full")):
        _refused(capsys, walk[0], "--stage1-json", "x.json", *walk)
    _refused(capsys, "--runs", "--runs", "0")


def test_bases_only_is_refused_at_the_full_sweep_level(capsys):
    """A sampled ``full`` walk draws random knob sets, so the base filter keeps
    only the bases the sample happens to contain, not each tile's: a run would
    time that remnant and ``auto`` and exit 0. It is refused."""
    full = ("--sweep-level", "full", "--tuning-sample", "3")
    walk = sweep.options(sweep_level="full", tuning_sample=3, bases_only=True)
    _, kept = sweep.shape_configs(SHAPE, walk)
    _, bases = sweep.shape_configs(SHAPE, sweep.options(bases_only=True))
    assert len(kept) < len(bases)
    _refused(capsys, "--bases-only", "--bases-only", *full)
    _refused(capsys, "--bases-only", "--list-only", "--bases-only", *full)
    with pytest.raises(SystemExit) as err:
        sweep.main(["--list-only", "--bases-only", *full])
    assert err.value.code == 2
    assert "--bases-only" in capsys.readouterr().err
    # Where the filter means something it stays accepted.
    assert _parse("--bases-only").bases_only
    assert not _parse(*full).bases_only


@pytest.mark.parametrize(
    "text", ["", ",", "1,,2", "1,", "a", "1,x", "0", "-3,1", "1,0"]
)
def test_the_sweep_refuses_bad_batches_as_a_usage_error(text, capsys):
    # The parser itself is tested in tests/dispatch/gdn/test_cli_args.py.
    _refused(capsys, "--batches", "--batches", text)


def test_cli_grid_and_shape_file_filters(tmp_path):
    args = _parse(
        "--gate-kind",
        "kda",
        "--mode",
        "fused",
        "--state-dtype",
        "f32",
        "--heads",
        "8/8,32/32",
        "--batches",
        "1,16",
    )
    shapes = sweep.select_shapes(args)
    assert [(s.num_k_heads, s.batch) for s in shapes] == [
        (8, 1),
        (8, 16),
        (32, 1),
        (32, 16),
    ]
    assert {s.case for s in shapes} == {"kda_fused_f32"}

    path = tmp_path / "shapes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "gate_kind": "gdn",
                    "mode": "plain",
                    "state_dtype": "f16",
                    "heads": ["4/8", "16/32"],
                    "batches": [1, 2],
                },
                {
                    "gate_kind": "kda",
                    "mode": "plain",
                    "state_dtype": "bf16",
                    "num_k_heads": 8,
                    "num_v_heads": 8,
                    "batch": 4,
                    "head_dim": 64,
                },
            ]
        )
    )
    every = sweep.select_shapes(_parse("--shapes", str(path)))
    assert len(every) == 5
    # f16 state reads the bf16 AUTO_DEFAULTS row.
    assert every[0].case == "gdn_plain_bf16" and every[0].state_dtype == "f16"
    narrowed = sweep.select_shapes(
        _parse("--shapes", str(path), "--heads", "16/32", "--batches", "2")
    )
    assert narrowed == [sweep.Shape("gdn", "plain", "f16", 16, 32, 128, 2)]
    assert sweep.select_shapes(_parse("--shapes", str(path), "--head-dim", "64")) == [
        sweep.Shape("kda", "plain", "bf16", 8, 8, 64, 4)
    ]
    path.write_text(
        json.dumps(
            [
                {
                    "gate_kind": "gdn",
                    "mode": "plain",
                    "state_dtype": "bf16",
                    "batch": 1,
                    "heds": ["1/1"],
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="unknown --shapes keys"):
        sweep.select_shapes(_parse("--shapes", str(path)))
    with pytest.raises(ValueError, match="mode"):
        sweep.Shape("gdn", "conv", "bf16", 16, 32, 128, 1)


def test_shipped_shape_grid_covers_every_case_and_dispatches():
    path = os.path.join(os.path.dirname(sweep.__file__), "decode_shapes.json")
    shapes = sweep.load_shapes(path)
    by_case: dict = {}
    for shape in shapes:
        by_case.setdefault(shape.case, []).append(shape)
    assert sorted(by_case) == sorted(
        f"{g}_{m}_{s}"
        for g in ("gdn", "kda")
        for m in ("plain", "fused")
        for s in ("bf16", "f32")
    )
    assert all(len(v) == 21 for v in by_case.values())
    for shape in shapes:
        assert dispatch_gdn_decode(shape.request()).spec.tuning_id


# ---------------------------------------------------------------- walk
def test_window_shards_are_disjoint_and_cover_the_window():
    for count, offset, limit, n in itertools.product(
        (0, 1, 7, 54), (0, 3), (0, 5), (1, 2, 3, 8)
    ):
        expected = list(
            range(offset, count if not limit else min(count, offset + limit))
        )
        shards = [sweep.window(count, (i, n), offset, limit) for i in range(n)]
        flat = [i for s in shards for i in s]
        assert sorted(flat) == expected
        assert len(flat) == len(set(flat))


def test_shape_configs_always_carry_auto_once():
    args = sweep.options(bases_only=True)
    auto, configs = sweep.shape_configs(SHAPE, args)
    ids = [r.spec.tuning_id for r in configs]
    assert all(not r.spec.knobs for r in configs[:-1])
    # auto runs a knob stack, so --bases-only appends it, once, at the end.
    assert ids[-1] == auto.spec.tuning_id and ids.count(auto.spec.tuning_id) == 1
    assert len(configs) == 54 + 1
    full_auto, full = sweep.shape_configs(SHAPE, sweep.options())
    assert [r.spec.tuning_id for r in full].count(full_auto.spec.tuning_id) == 1
    picked = sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(ids[3], ids[5])))[1]
    assert [r.spec.tuning_id for r in picked] == [ids[3], ids[5], auto.spec.tuning_id]
    with pytest.raises(TypeError, match="unknown sweep options"):
        sweep.options(rotate=0)


def test_pinned_result_replays_a_recorded_row_and_refuses_another():
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    result = configs[7]
    row = sweep.config_row(SHAPE, result, 7, auto.spec.tuning_id, {})
    row = json.loads(json.dumps(row))  # as read back from a file
    assert sweep.pinned_result(SHAPE, row).spec == result.spec
    other = sweep.Shape("gdn", "plain", "bf16", 16, 32, 64, 1)
    with pytest.raises(PinRefused):
        sweep.pinned_result(other, row)


# ---------------------------------------------------------------- stage 2
def test_interleaved_order_alternates_within_and_across_runs():
    seen = []
    samplers = [lambda i=i: seen.append(i) or float(i) for i in range(4)]
    samples = sweep.interleaved_samples(samplers, rounds=3, run=0)
    assert seen == [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]
    assert samples == [[0.0] * 3, [1.0] * 3, [2.0] * 3, [3.0] * 3]
    assert sweep.run_order(4, 1) == [3, 2, 1, 0]
    assert sweep.run_order(5, 2) == [2, 3, 4, 0, 1]
    assert sweep.run_order(4, 3) == [0, 1, 2, 3]
    seen.clear()
    sweep.interleaved_samples(samplers, rounds=2, run=2)
    assert seen == [2, 3, 0, 1, 1, 0, 3, 2]


def _row(index, us, status="ok", auto=False):
    return {
        "index": index,
        "us": us,
        "status": status,
        "auto": auto,
        "tuning_id": f"c{index}",
    }


def test_stage2_picks_top_k_then_auto_whatever_its_status():
    rows = [
        _row(0, 5.0),
        _row(1, 3.0),
        _row(2, 3.0),
        _row(3, 1.0, "fail"),
        _row(4, 9.0, auto=True),
        _row(5, 4.0),
    ]
    assert [r["index"] for r in sweep.stage2_picks(rows, 3)] == [1, 2, 5, 4]
    assert [r["index"] for r in sweep.stage2_picks(rows, 6)] == [1, 2, 5, 0, 4]
    rows[4]["status"] = "fail"
    assert [r["index"] for r in sweep.stage2_picks(rows, 1)] == [1, 4]


class _FakeCold:
    def __init__(self, samples, copies):
        self._samples = samples
        self.copies = copies
        self.closed = False

    def replay_us(self):
        return next(self._samples)

    def close(self):
        self.closed = True


class _CaptureUnavailable(RuntimeError):
    pass


class _FakeDriver:
    """The builder driver's sweep surface, with chosen timer samples. It
    checks that the gate sees disjoint read / write slots and timing the
    in-place ones. Its gate returns the real (torch-free) ``Gate``, so every
    test that gates through it uses the one pass test: ``failing`` ids fail
    with a large output error, ``verdicts`` maps an id to the ``Gate`` to
    return."""

    CaptureUnavailable = _CaptureUnavailable

    def __init__(self, samples, *, failing=(), raising=(), no_graph=(), verdicts=None):
        # tuning_id -> samples, consumed in order across every capture
        self.samples = {tid: iter(values) for tid, values in samples.items()}
        self.failing, self.raising, self.no_graph = failing, raising, no_graph
        self.verdicts = verdicts or {}
        self.ids = {}
        self.colds = []
        self.value_sets_made = 0
        self.seeds = []  # the seed of every make_inputs call

    def name(self, spec):
        return self.ids[spec]

    def make_inputs(self, spec, batch, seed, disjoint_writes):
        self.seeds.append(seed)
        return {"disjoint": disjoint_writes}

    def reference(self, spec, inp):
        assert inp["disjoint"], "the reference must score the gate's inputs"
        return None

    def touched_bytes(self, spec, inp, batch):
        assert not inp["disjoint"], "the rotation is sized on the timed inputs"
        return 2**20

    def launcher_for(self, spec, arch):
        if self.name(spec) in self.raising:
            raise RuntimeError("compile failed")
        return spec

    def run_values(self, spec, inp, launcher, batch):
        assert inp["disjoint"], "the gate must run on disjoint read/write slots"
        return {}

    def gate(self, spec, inp, values, ref):
        assert inp["disjoint"]
        name = self.name(spec)
        if name in self.verdicts:
            return self.verdicts[name]
        return Gate(1.0 if name in self.failing else 0.0, 0.0, 0.0, False)

    def cold_value_sets(self, spec, inp, batch, rotate_mb):
        assert not inp["disjoint"], "timing runs on the in-place slots"
        self.value_sets_made += 1
        return [object()] * 4

    def capture_cold(self, spec, batch, sets, launcher):
        tid = self.name(spec)
        if tid in self.no_graph:
            raise _CaptureUnavailable("RuntimeError: operation not permitted")
        cold = _FakeCold(self.samples[tid], len(sets))
        self.colds.append(cold)
        return cold

    def drain(self):
        pass


def _fake(monkeypatch, configs, samples, **kw):
    driver = _FakeDriver(samples, **kw)
    driver.ids = {r.spec.kernel_spec: r.spec.tuning_id for r in configs}
    monkeypatch.setattr(sweep, "_driver", lambda: driver)
    return driver


def test_stage1_records_the_median_and_every_failure(monkeypatch):
    args = sweep.options(bases_only=True, rotate_mb=1024, rounds=5)
    auto, configs = sweep.shape_configs(SHAPE, args)
    ids = [r.spec.tuning_id for r in configs]
    driver = _fake(
        monkeypatch,
        configs,
        {ids[0]: [5.0, 1.0, 100.0, 3.0, 4.0], ids[-1]: [2.0] * 5},
        failing={ids[1]},
        raising={ids[2]},
        no_graph={ids[3]},
    )
    rows = sweep.stage1_shape(
        SHAPE, auto, configs, [0, 1, 2, 3, len(ids) - 1], args, {"rotate_mb": 1024}
    )
    status = {r["tuning_id"]: r["status"] for r in rows}
    assert status == {
        ids[0]: "ok",
        ids[1]: "fail",
        ids[2]: "error",
        ids[3]: "no_graph",
        ids[-1]: "ok",
    }
    first = rows[0]
    assert first["us"] == 4.0 and first["runs"] == [4.0] and first["copies"] == 4
    kernel = configs[0].spec.kernel_spec
    assert first["stage"] == 1
    assert first["tile"] == [
        kernel.num_warps,
        kernel.warp_threads_k,
        kernel.blocks_per_v_dim,
    ]
    assert rows[-1]["auto"] and not first["auto"]
    assert "compile failed" in rows[2]["reason"]
    assert "operation not permitted" in rows[3]["reason"]  # the capture's cause
    assert rows[1]["us"] is None  # a gate failure is never timed
    assert all(c.closed for c in driver.colds)
    assert driver.value_sets_made == 1  # one set of buffers per shape


@pytest.mark.parametrize("verdict", BAD_VERDICTS.values(), ids=BAD_VERDICTS.keys())
def test_a_launch_the_gate_fails_is_never_timed_or_ranked(monkeypatch, verdict):
    """The sweep scores a configuration with the driver's ``Gate`` (the real
    one: it needs no torch) and must neither time nor rank one that fails,
    however fast it would have been, and the run must fail."""
    args = sweep.options(bases_only=True, rotate_mb=1024, rounds=3)
    auto, configs = sweep.shape_configs(SHAPE, args)
    ids = [r.spec.tuning_id for r in configs]
    driver = _fake(
        monkeypatch,
        configs,
        {ids[0]: [5.0] * 3, ids[1]: [0.1] * 3},  # the failing one is the fastest
        verdicts={ids[1]: verdict},
    )
    rows = sweep.stage1_shape(SHAPE, auto, configs, [0, 1], args, {"rotate_mb": 1024})
    good, bad = rows
    assert (good["status"], bad["status"]) == ("ok", "fail")
    assert bad["us"] is None and not bad["auto"] and not good["auto"]
    assert len(driver.colds) == 1  # the failing configuration was never captured
    assert bad["spill"] == verdict.spill
    for name in ("out_err", "state_err", "conv_err"):
        got, want = bad[name], getattr(verdict, name)
        assert got == want or (math.isnan(got) and math.isnan(want))
    assert [r["tuning_id"] for r in sweep.stage2_picks(rows, 2)] == [ids[0]]
    assert sweep.fastest_timed(rows)["tuning_id"] == ids[0]
    assert sweep.summarize(rows, require_auto=False) == 1
    assert sweep.summarize([good], require_auto=False) == 0


def test_stage2_retimes_interleaved_and_reports_the_median_of_run_medians(monkeypatch):
    args = sweep.options(rotate_mb=1024, s2_rounds=3, runs=3)
    auto, configs = sweep.shape_configs(SHAPE, args)
    a, b = configs[0], next(
        r for r in configs if r.spec.tuning_id == auto.spec.tuning_id
    )
    # Per run, 3 rounds each; run medians a: 2, 7, 4 -> 4; auto: 6, 1, 3 -> 3.
    samples = {
        a.spec.tuning_id: [1, 2, 9, 7, 7, 0, 4, 4, 5],
        b.spec.tuning_id: [6, 6, 8, 1, 2, 1, 3, 0, 3],
    }
    driver = _fake(monkeypatch, configs, samples)
    rows = sweep.stage2_shape(SHAPE, auto, [(0, a), (5, b)], args, {"rotate_mb": 1024})
    assert [r["runs"] for r in rows] == [[2, 7, 4], [6, 1, 3]]
    assert [r["us"] for r in rows] == [4, 3]
    assert [r["stage"] for r in rows] == [2, 2] and rows[1]["auto"]
    assert driver.value_sets_made == 3  # fresh buffers every run
    assert len(driver.colds) == 6 and all(c.closed for c in driver.colds)


# ---------------------------------------------------------------- files
def _rows():
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    meta = dict(_META, sha="abc", device="gfx950", arch="gfx950")
    out = []
    for stage, index, us in ((1, 0, 3.0), (1, 1, 2.0), (2, 0, 2.5), (2, 1, 2.6)):
        row = sweep.config_row(
            SHAPE,
            configs[index],
            index,
            configs[0].spec.tuning_id,
            dict(meta, stage=stage),
        )
        row.update(status="ok", us=us, runs=[us])
        out.append(row)
    return out


def test_rows_round_trip_through_json_and_refuse_a_foreign_file(tmp_path):
    rows = _rows()
    path = tmp_path / "rows.json"
    sweep.write_rows(str(path), rows)
    assert sweep.load_rows([str(tmp_path / "*.json")]) == rows
    assert {sweep.shape_of(r) for r in rows} == {SHAPE}
    for key in (
        "tuning_id",
        "tile",
        "knobs",
        "us",
        "runs",
        "status",
        "sha",
        "arch",
        "out_err",
        "spill",
        *sweep.SHAPE_FIELDS,
    ):
        assert key in rows[0]
    (tmp_path / "other.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="not a"):
        sweep.load_rows([str(tmp_path / "other.json")])


def test_summarize_prefers_stage_two_and_fails_closed(capsys):
    rows = _rows()
    assert sweep.summarize(rows) == 0
    out = capsys.readouterr().out
    # Stage 2 decides: config 0 (auto here) is fastest there, unlike stage 1.
    line = next(l for l in out.splitlines() if l.startswith(SHAPE.label))
    assert rows[2]["tuning_id"] in line and "1.000" in line and " 2 " in line
    rows[3]["status"] = "fail"
    rows[3]["reason"] = "fp32 gate"
    assert sweep.summarize(rows) == 1
    for row in rows:
        row["auto"] = False
    capsys.readouterr()
    assert sweep.summarize(rows[:2]) == 1
    assert "auto configuration not timed" in capsys.readouterr().out
    # A windowed stage-1 run times auto in one window only.
    assert sweep.summarize(rows[:2], require_auto=False) == 0
    rows[0]["rotate_mb"] = 0
    sweep.summarize(rows)
    assert "WARNING" in capsys.readouterr().out


def test_stage2_from_files_splits_shapes_across_shards(monkeypatch, tmp_path):
    shapes = [sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, b) for b in (1, 2, 3)]
    rows = []
    for shape in shapes:
        auto, configs = sweep.shape_configs(shape, sweep.options())
        for i in (0, 1):
            row = sweep.config_row(
                shape,
                configs[i],
                i,
                auto.spec.tuning_id,
                dict(_META, stage=1),
            )
            row.update(status="ok", us=float(i + 1))
            rows.append(row)
    sweep.write_rows(str(tmp_path / "s1.json"), rows)
    seen = {}

    def fake_stage2(shape, auto, picks, args, meta):
        seen[shape] = [r.spec.tuning_id for _, r in picks]
        return []

    monkeypatch.setattr(sweep, "stage2_shape", fake_stage2)
    covered = []
    for i in range(2):
        seen.clear()
        args = sweep.options(
            stage1_json=[str(tmp_path / "s1.json")], shard=(i, 2), top_k=1
        )
        sweep.retime(args, {})
        covered.extend(seen)
        for shape, ids in seen.items():
            auto = dispatch_gdn_decode(shape.request()).spec.tuning_id
            assert ids[0] == rows[0]["tuning_id"] and ids[-1] == auto
    assert sorted(covered) == shapes


# ---------------------------------------------------------------- fail closed
def test_cli_refuses_flags_a_mode_would_accept_and_ignore(capsys):
    # --summarize reads saved rows only: no filter, output or walk/timing flag.
    for extra in (
        ("--gate-kind", "kda"),
        ("--batches", "16"),
        ("--output-json", "x.json"),
        ("--bases-only",),
        ("--top-k", "2"),
        ("--rotate-mb", "0"),
    ):
        _refused(capsys, extra[0], "--summarize", "x.json", *extra)
    _parse("--summarize", "x.json", "y.json")
    # --stage1-only runs no stage 2.
    for extra in (("--top-k", "2"), ("--s2-rounds", "4"), ("--runs", "1")):
        _refused(capsys, extra[0], "--stage1-only", *extra)
    _parse("--stage1-only", "--rounds", "3")
    _refused(capsys, "--list-only", "--stage1-json", "x.json", "--list-only")
    # Out-of-domain values are refused, never replaced by a default.
    for bad in (("--head-dim", "0"), ("--head-dim", "-8"), ("--rotate-mb", "-1024")):
        _refused(capsys, bad[0], *bad)


def test_shape_entries_refuse_conflicting_or_empty_keys():
    base = {"gate_kind": "gdn", "mode": "plain", "state_dtype": "bf16"}
    for bad, match in (
        (
            dict(heads=["16/32"], num_k_heads=4, num_v_heads=8, batches=[1]),
            "both heads",
        ),
        (dict(heads=["16/32"], batches=[1], batch=64), "both batches"),
        (dict(heads=["16/32"], batches=[]), "empty batches"),
        (dict(heads=[], batches=[1]), "empty heads"),
    ):
        with pytest.raises(ValueError, match=match):
            sweep.expand_entry(dict(base, **bad))


def test_tuning_ids_the_walk_does_not_list_are_an_error(capsys):
    production = sweep.shape_configs(SHAPE, sweep.options())[1]
    listed = {r.spec.tuning_id for r in production}
    real = production[3].spec.tuning_id
    with pytest.raises(ValueError, match="typo_id") as err:
        sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(real, "typo_id")))
    assert "--sweep-level production" in str(err.value) and real not in str(err.value)
    # A full-level id is not in the production walk: refused there, found at full.
    full = sweep.shape_configs(SHAPE, sweep.options(sweep_level="full"))[1]
    extra = next(r.spec.tuning_id for r in full if r.spec.tuning_id not in listed)
    with pytest.raises(ValueError, match="production"):
        sweep.shape_configs(SHAPE, sweep.options(tuning_ids=(extra,)))
    found = sweep.shape_configs(
        SHAPE, sweep.options(sweep_level="full", tuning_ids=(extra,))
    )[1]
    assert found[0].spec.tuning_id == extra
    # The command line exits non-zero and names the id.
    assert sweep.main(["--batches", "1", "--list-only", "--tuning-ids", "typo_id"]) == 1
    assert "typo_id" in capsys.readouterr().out


def test_a_stage_one_failure_fails_the_run_after_stage_two(monkeypatch, tmp_path):
    """A configuration that fails the gate in stage 1 is never re-timed, so the
    stage-2 rows alone are clean: the summary must still fail, in one process
    and from saved files (``--summarize s1 s2``)."""
    ids = [r.spec.tuning_id for r in sweep.shape_configs(SHAPE, sweep.options())[1]]
    args = sweep.options(
        tuning_ids=(ids[0], ids[1]), rotate_mb=1024, rounds=1, top_k=1, runs=1
    )
    auto, configs = sweep.shape_configs(SHAPE, args)
    _fake(
        monkeypatch,
        configs,
        {r.spec.tuning_id: [1.0] * 64 for r in configs},
        failing={ids[1]},
    )
    rows = sweep.sweep(args, [SHAPE], dict(_META))
    stage1 = [r for r in rows if r["stage"] == 1]
    stage2 = [r for r in rows if r["stage"] == 2]
    assert {r["status"] for r in stage2} == {"ok"} and ids[1] not in {
        r["tuning_id"] for r in stage2
    }
    assert sweep.summarize(stage2) == 0
    assert sweep.summarize(rows) == 1
    s1, s2 = str(tmp_path / "s1.json"), str(tmp_path / "s2.json")
    sweep.write_rows(s1, stage1)
    sweep.write_rows(s2, stage2)
    assert sweep.main(["--summarize", s1, s2]) == 1


def test_stage2_from_files_fails_on_a_shape_with_no_timed_row(
    monkeypatch, tmp_path, capsys
):
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    rows = []
    for i in (0, 1):
        row = sweep.config_row(
            SHAPE, configs[i], i, auto.spec.tuning_id, dict(_META, stage=1)
        )
        row.update(status="fail", reason="fp32 gate")
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)

    def no_stage2(*args):
        raise AssertionError("a shape with no timed row was re-timed")

    monkeypatch.setattr(sweep, "stage2_shape", no_stage2)
    out = sweep.retime(sweep.options(stage1_json=[path]), {"rotate_mb": 1024})
    assert [(r["stage"], r["status"]) for r in out] == [(2, "error")]
    assert "no timed stage-1 row" in out[0]["reason"]
    assert sweep.summarize(out) == 1
    # Filters that leave no recorded shape: an error, not an empty success.
    with pytest.raises(ValueError, match="no recorded stage-1 shape"):
        sweep.retime(sweep.options(stage1_json=[path], gate_kind="kda"), {})
    with pytest.raises(SystemExit) as err:
        sweep.main(["--stage1-json", path, "--gate-kind", "kda"])
    assert err.value.code == 2
    assert "no recorded stage-1 shape" in capsys.readouterr().err


def test_the_gate_rejects_a_kernel_that_reads_its_write_slot(monkeypatch):
    """The sweep gates on disjoint read / write slots. A kernel that reads its
    state from the write slot computes the right answer when the slots are the
    same (the in-place inputs timing uses), so only disjoint slots catch it."""
    torch = pytest.importorskip("torch")
    from builders.gfx950.gdn import gdn_decode as g

    def reads_write_slot(spec, inp, launcher, batch):
        out, state, _ = g.reference(spec, dict(inp, read_indices=inp["write_indices"]))
        values = {"out": out.to(torch.bfloat16), "state": inp["state"].clone()}
        values["state"][inp["write_indices"].long()] = state.to(values["state"].dtype)
        return values

    def not_timed(*args):
        raise AssertionError("a configuration that failed the gate was timed")

    driver = SimpleNamespace(
        make_inputs=lambda spec, batch, seed, disjoint_writes: g.make_inputs(
            spec, batch, seed=seed, device="cpu", disjoint_writes=disjoint_writes
        ),
        reference=g.reference,
        gate=g.gate,
        touched_bytes=g.touched_bytes,
        launcher_for=lambda spec, arch: object(),
        run_values=reads_write_slot,
        cold_value_sets=not_timed,
        capture_cold=not_timed,
        CaptureUnavailable=g.CaptureUnavailable,
        drain=lambda: None,
    )
    monkeypatch.setattr(sweep, "_driver", lambda: driver)
    auto = dispatch_gdn_decode(SHAPE.request())
    (row,) = sweep.stage1_shape(SHAPE, auto, [auto], [0], sweep.options(), {})
    assert row["status"] == "fail", row
    # The blind spot this closes: on in-place slots the same kernel passes.
    spec = auto.spec.kernel_spec
    inp = g.make_inputs(spec, 1, device="cpu", disjoint_writes=False)
    values = reads_write_slot(spec, inp, None, 1)
    assert g.gate(spec, inp, values, g.reference(spec, inp)).passed


# ---------------------------------------------------------------- flag checks
def test_empty_tuning_ids_are_refused_not_read_as_no_filter(capsys):
    """An empty tuple means "no filter", so an empty list or entry used to run
    the whole walk."""
    for bad in ("", ",", "a,,b", " ", "a,"):
        _refused(capsys, "--tuning-ids", "--tuning-ids", bad)
    assert _parse().tuning_ids == ()
    assert _parse("--tuning-ids", "a, b").tuning_ids == ("a", "b")
    with pytest.raises(SystemExit) as err:
        sweep.main(["--batches", "1", "--list-only", "--tuning-ids", ""])
    assert err.value.code == 2 and "--tuning-ids" in capsys.readouterr().err


def test_tuning_sample_needs_the_full_sweep_level(capsys):
    for bad in (
        ("--tuning-sample", "5"),  # ignored at production: refused
        ("--sweep-level", "full"),  # the whole knob space
        ("--sweep-level", "full", "--tuning-sample", "0"),
        ("--sweep-level", "full", "--tuning-sample", "-1"),
    ):
        _refused(capsys, "--tuning-sample", *bad)
    args = _parse("--sweep-level", "full", "--tuning-sample", "5")
    assert (args.sweep_level, args.tuning_sample) == ("full", 5)
    assert _parse().tuning_sample == 0


def test_list_only_has_no_rows_for_an_output_file(tmp_path, capsys):
    out = str(tmp_path / "x.json")
    with pytest.raises(SystemExit) as err:
        sweep.main(["--list-only", "--output-json", out])
    assert err.value.code == 2 and not os.path.exists(out)
    assert "--output-json" in capsys.readouterr().err
    _parse("--list-only")


def test_stage1_json_refuses_rounds_too(capsys):
    """Stage 2 from files reads --s2-rounds, not the stage-1 --rounds."""
    _refused(capsys, "--rounds", "--stage1-json", "x.json", "--rounds", "3")
    _parse("--stage1-json", "x.json", "--s2-rounds", "4")


def test_stage1_json_refuses_seed_it_would_ignore(capsys):
    """Stage 2 takes the walk (seed included) from the stage-1 rows, so a
    ``--seed`` given beside ``--stage1-json`` would be a mislabel."""
    _refused(capsys, "--seed", "--stage1-json", "x.json", "--seed", "3")
    _parse("--stage1-json", "x.json", "--s2-rounds", "4")


def test_unreadable_or_non_schema_inputs_are_usage_errors(tmp_path, capsys):
    """A missing file, invalid JSON and a foreign JSON file exit 2 naming the
    path, for every input the CLI reads (--summarize, --stage1-json, --shapes),
    never a traceback."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"schema": "other", "rows": []}')
    shapes_not_list = tmp_path / "shapes.json"
    shapes_not_list.write_text("{}")
    missing = tmp_path / "missing.json"
    cases = [
        (flag, path)
        for flag in ("--summarize", "--stage1-json")
        for path in (missing, bad_json, foreign)
    ] + [("--shapes", p) for p in (missing, bad_json, shapes_not_list)]
    for flag, path in cases:
        argv = [flag, str(path)] + (["--list-only"] if flag == "--shapes" else [])
        capsys.readouterr()
        with pytest.raises(SystemExit) as err:
            sweep.main(argv)
        assert err.value.code == 2, (flag, path)
        assert str(path) in capsys.readouterr().err, (flag, path)


@pytest.mark.parametrize("flag", ("--summarize", "--stage1-json", "--shapes"))
def test_a_missing_path_with_backslashes_is_named_as_typed(flag, tmp_path, capsys):
    """``str(OSError)`` shows the path through ``repr``, which doubles each
    backslash of a Windows path; the usage error must print the path as given.
    A backslash is an ordinary file-name character on POSIX, so this fails
    there too when the message is built from ``str(OSError)``."""
    missing = str(tmp_path / "no\\such\\file.json")
    argv = [flag, missing] + (["--list-only"] if flag == "--shapes" else [])
    with pytest.raises(SystemExit) as err:
        sweep.main(argv)
    assert err.value.code == 2
    assert missing in capsys.readouterr().err


def test_a_pin_that_no_shape_resolves_is_refused_with_its_own_error():
    """Pinned to a head dim the tile's tuning does not read: the dispatcher
    refuses the pin before the re-dispatch check runs."""
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    row = json.loads(
        json.dumps(sweep.config_row(SHAPE, configs[7], 7, auto.spec.tuning_id, {}))
    )
    other = sweep.Shape("gdn", "plain", "bf16", 16, 32, 64, 1)
    with pytest.raises(PinRefused):
        sweep.pinned_result(other, row)


def test_a_pin_that_re_dispatches_as_another_configuration_is_refused(monkeypatch):
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    row = json.loads(
        json.dumps(sweep.config_row(SHAPE, configs[7], 7, auto.spec.tuning_id, {}))
    )
    assert configs[8].spec.tuning_id != row["tuning_id"]
    monkeypatch.setattr(sweep, "dispatch_gdn_decode", lambda request: configs[8])
    with pytest.raises(ValueError, match="re-dispatched as"):
        sweep.pinned_result(SHAPE, row)


def test_a_stage_two_gate_failure_fails_the_run(monkeypatch):
    """Stage 2 gates each configuration again on its own inputs: a failure
    there is recorded, never timed, and fails the summary. ``auto`` is timed
    and the other arm passes, so exit 1 is owed to the failed row alone."""
    auto, configs = sweep.shape_configs(SHAPE, sweep.options(bases_only=True))
    a, b = configs[0], configs[1]
    args = sweep.options(rotate_mb=1024, s2_rounds=2, runs=1)
    _fake(
        monkeypatch,
        configs + [auto],
        {r.spec.tuning_id: [1.0] * 8 for r in configs + [auto]},
        failing={b.spec.tuning_id},
    )
    picks = [(0, a), (1, b), (-1, auto)]
    rows = sweep.stage2_shape(SHAPE, auto, picks, args, dict(_META, stage=2))
    status = {r["tuning_id"]: r["status"] for r in rows}
    assert status[b.spec.tuning_id] == "fail" and status[a.spec.tuning_id] == "ok"
    assert status[auto.spec.tuning_id] == "ok"
    assert [r for r in rows if r["status"] == "fail"][0]["us"] is None
    assert sweep.summarize(rows) == 1
    # Without the failed row the same rows pass: the failure is the cause.
    assert sweep.summarize([r for r in rows if r["status"] == "ok"]) == 0


def test_a_stage_two_gate_failure_in_a_run_from_files_fails_the_run(
    monkeypatch, tmp_path
):
    auto, configs = sweep.shape_configs(SHAPE, sweep.options(bases_only=True))
    ids = [configs[0].spec.tuning_id, configs[1].spec.tuning_id]
    rows = []
    for i in (0, 1):
        row = sweep.config_row(
            SHAPE, configs[i], i, auto.spec.tuning_id, dict(_META, stage=1)
        )
        row.update(status="ok", us=float(i + 1))
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)
    _fake(
        monkeypatch,
        configs + [auto],
        {r.spec.tuning_id: [1.0] * 64 for r in configs},
        failing={ids[1]},
    )
    args = sweep.options(
        stage1_json=[path], top_k=2, s2_rounds=2, runs=1, rotate_mb=1024
    )
    out = sweep.retime(args, dict(_META))
    assert {r["tuning_id"]: r["status"] for r in out}[ids[1]] == "fail"
    assert sweep.summarize(out) == 1


def test_a_run_that_times_nothing_fails_unless_another_shard_covers_it():
    """``--offset`` past every shape's configurations leaves nothing for any
    shard: a silent no-op, so a failure. One ``--shard I/N`` that happens to own
    no index of a small window is fine when another shard owns it."""
    count = len(sweep.shape_configs(SHAPE, sweep.options())[1])
    past = sweep.options(offset=count)
    assert sweep.sweep(past, [SHAPE], {}) == []
    assert sweep.timed_nothing(past, [SHAPE])
    assert sweep.timed_nothing(sweep.options(offset=count, shard=(1, 4)), [SHAPE])
    # Window [0, 1): index 0 belongs to shard 0; shard 1 times nothing, legitimately.
    other = sweep.options(limit=1, shard=(1, 2), stage1_only=True)
    assert sweep.sweep(other, [SHAPE], {}) == []
    assert not sweep.timed_nothing(other, [SHAPE])
    # One shape with configurations in the window is enough.
    small = sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, 2)
    assert not sweep.timed_nothing(sweep.options(offset=count - 1), [SHAPE, small])


def test_stage2_from_files_records_the_stage_one_walk(monkeypatch, tmp_path, capsys):
    """The stage-2 rows' own arguments are forced to defaults under
    ``--stage1-json``: they must carry the walk of the rows they re-time."""
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    walk = dict(sweep_level="full", tuning_sample=3, seed=5, bases_only=True)
    rows = []
    for i in (0, 1):
        row = sweep.config_row(
            SHAPE, configs[i], i, auto.spec.tuning_id, dict(_META, stage=1, **walk)
        )
        row.update(status="ok", us=float(i + 1))
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)
    seen = []
    monkeypatch.setattr(
        sweep,
        "stage2_shape",
        lambda shape, auto, picks, args, meta: seen.append(meta) or [],
    )
    sweep.retime(
        sweep.options(stage1_json=[path]), dict(_META, sweep_level="production")
    )
    assert [{k: m[k] for k in walk} for m in seen] == [walk]
    # Rows of two walks cannot be re-timed as one run: refused up front.
    rows[1]["seed"] = 6
    sweep.write_rows(path, rows)
    with pytest.raises(ValueError, match="2 walks"):
        sweep.retime(sweep.options(stage1_json=[path]), dict(_META))
    with pytest.raises(SystemExit) as err:
        sweep.main(["--stage1-json", path])
    assert err.value.code == 2 and "2 walks" in capsys.readouterr().err
    # A row that never recorded a walk is refused where rows are read
    # (``load_rows``), before ``stage1_walk`` sees it.
    del rows[1]["seed"]
    sweep.write_rows(path, rows)
    with pytest.raises(ValueError, match="row 1 lacks seed"):
        sweep.load_rows([path])
    with pytest.raises(ValueError, match="row 1 lacks seed"):
        sweep.retime(sweep.options(stage1_json=[path]), dict(_META))


def test_run_meta_records_walk_toolchain_and_a_dirty_tree(monkeypatch):
    toolchain = dict(rocm_version="7.1", llvm_version="21.0.0", torch_version="2.9")
    monkeypatch.setattr(sweep, "_toolchain", lambda: toolchain)
    args = sweep.options(bases_only=True, rotate_mb=8, seed=3)
    answers = {("rev-parse", "HEAD"): "abc", ("status", "--porcelain", "-uno"): ""}
    monkeypatch.setattr(sweep, "_git", lambda *a: answers[a])
    meta = sweep.run_meta(args, "gfx950", 1024)
    assert meta["sha"] == "abc"
    assert (meta["bases_only"], meta["seed"], meta["rotate_mb"]) == (True, 3, 8)
    assert meta["cold_target_mb"] == 1024
    assert {k: meta[k] for k in toolchain} == toolchain
    answers[("status", "--porcelain", "-uno")] = " M gdn_decode.py"
    assert sweep.run_meta(args, "gfx950", 1024)["sha"] == "abc-dirty"
    # No git, no repo: unknown, never an error.
    monkeypatch.setattr(sweep, "_git", lambda *a: None)
    assert sweep.run_meta(args, "gfx950", 1024)["sha"] == "unknown"
    assert (
        sweep.run_meta(sweep.options(source_sha="pin"), "gfx950", 1024)["sha"] == "pin"
    )


def test_run_meta_records_the_resolved_backend_and_llvm_flavor(monkeypatch):
    """The backend and LLVM flavor a compile resolves are recorded beside the
    versions: the accessors ``compile_kernel`` uses, not environment reads."""
    from rocke.core import backend, lower_llvm

    monkeypatch.setattr(backend, "resolve_backend", lambda backend=None: "cpp")
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", lambda: "llvm22")
    tool = sweep._toolchain()
    assert (tool["backend"], tool["llvm_flavor"]) == ("cpp", "llvm22")
    monkeypatch.setattr(sweep, "_git", lambda *a: "abc" if a[0] == "rev-parse" else "")
    meta = sweep.run_meta(sweep.options(), "gfx950", 1024)
    assert (meta["backend"], meta["llvm_flavor"]) == ("cpp", "llvm22")

    def refuse(*a, **k):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(backend, "resolve_backend", refuse)
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", refuse)
    tool = sweep._toolchain()
    assert (tool["backend"], tool["llvm_flavor"]) == ("unknown", "unknown")


@pytest.mark.parametrize(
    "walk",
    [
        dict(sweep_level="full", tuning_sample=2, seed=5, bases_only=True),
        dict(sweep_level="production", tuning_sample=0, seed=0, bases_only=False),
    ],
    ids=["seed5", "seed0"],
)
def test_stage2_from_files_builds_its_inputs_with_the_recorded_seed(
    monkeypatch, tmp_path, walk
):
    """``--seed`` is refused beside ``--stage1-json``, so stage 2's own seed
    is the default; the gate and timing inputs must use the seed the stage-1
    rows record, and the stage-2 rows must say the same."""
    auto, configs = sweep.shape_configs(SHAPE, sweep.options(bases_only=True))
    rows = []
    for i in (0, 1):
        row = sweep.config_row(
            SHAPE, configs[i], i, auto.spec.tuning_id, dict(_META, stage=1, **walk)
        )
        row.update(status="ok", us=float(i + 1))
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)
    driver = _fake(
        monkeypatch,
        configs + [auto],
        {r.spec.tuning_id: [1.0] * 64 for r in configs},
    )
    args = sweep.options(
        stage1_json=[path], top_k=2, s2_rounds=2, runs=1, rotate_mb=1024
    )
    assert args.seed == 0  # the command line's own seed is never the recorded one
    out = sweep.retime(args, dict(_META))
    assert out and {r["seed"] for r in out} == {walk["seed"]}
    assert {k: r[k] for r in out for k in walk} == walk
    assert driver.seeds and set(driver.seeds) == {walk["seed"]}


def _file_of(tmp_path, rows, name="rows.json"):
    path = str(tmp_path / name)
    sweep.write_rows(path, rows)
    return path


def _run_refused(capsys, argv, path, *words):
    """``sweep.main(argv)`` exits 2 with a message naming ``path`` and every
    word of ``words``; never a traceback."""
    capsys.readouterr()
    with pytest.raises(SystemExit) as err:
        sweep.main(argv)
    assert err.value.code == 2, argv
    message = capsys.readouterr().err
    assert path in message and all(w in message for w in words), message


# Every key a reader of a saved row reads unconditionally, listed by hand (not
# from ROW_KEYS) so a test can fail when ROW_KEYS is incomplete.
_READ_KEYS = (
    *sweep.SHAPE_FIELDS,
    "stage",
    "index",
    "spec_id",
    "tile",
    "status",
    "auto",
    "us",
    "reason",
    "tuning_id",
    "knobs",
    "rotate_mb",
    "sha",
    "device",
    "sweep_level",
    "tuning_sample",
    "seed",
    "bases_only",
)


def test_rows_that_lack_a_required_key_are_usage_errors(tmp_path, capsys):
    """A schema-valid file whose row lacks a key every reader needs exits 2
    naming the file, the row and the key, for --summarize and --stage1-json
    alike."""
    rows = _rows()
    del rows[2]["stage"]
    path = _file_of(tmp_path, rows)
    for flag in ("--summarize", "--stage1-json"):
        _run_refused(capsys, [flag, path], path, "row 2", "stage")
    with pytest.raises(ValueError, match="row 2 lacks stage"):
        sweep.load_rows([path])
    # Every key the readers read, spelled out here rather than derived from
    # ROW_KEYS, so the test fails when ROW_KEYS misses one. Each is refused
    # by both file readers, naming the row and the key (the report's entry
    # point is checked in test_gdn_decode_defaults_report.py).
    assert set(_READ_KEYS) <= set(sweep.ROW_KEYS)
    for key in _READ_KEYS:
        broken = _rows()
        del broken[0][key]
        path = _file_of(tmp_path, broken, f"{key}.json")
        with pytest.raises(ValueError, match=f"row 0 lacks {key}"):
            sweep.load_rows([path])
        for flag in ("--summarize", "--stage1-json"):
            _run_refused(capsys, [flag, path], path, "row 0", key)
    path = _file_of(tmp_path, ["not a row"], "scalar.json")
    _run_refused(capsys, ["--summarize", path], path, "row 0")


@pytest.mark.parametrize(
    "field,value",
    [
        ("gate_kind", "x"),
        ("mode", "conv"),
        ("state_dtype", "f64"),
        ("batch", 0),
        ("num_k_heads", -1),
        ("head_dim", "128"),
        ("sweep_level", "x"),
        ("tuning_sample", "3"),
        ("seed", 1.5),
        ("bases_only", 1),
        ("rotate_mb", "big"),
        ("cold_target_mb", "big"),
    ],
)
def test_rows_with_a_bad_value_are_usage_errors(tmp_path, capsys, field, value):
    """A row whose keys are all there but whose shape or walk fields are
    invalid is refused when it is read, naming the file, row and field, for
    --summarize and --stage1-json alike (not a traceback in a later step)."""
    rows = _rows()
    rows[1][field] = value
    path = _file_of(tmp_path, rows)
    with pytest.raises(ValueError, match=f"row 1: {field}"):
        sweep.load_rows([path])
    for flag in ("--summarize", "--stage1-json"):
        _run_refused(capsys, [flag, path], path, "row 1", field)


# Each value below is one the tool never writes. ``ok`` rows carry a finite
# ``us`` > 0; a bad ``us`` is placed on an ok row (``_rows()`` rows are ok).
_BAD_VALUES = [
    ("stage", 3),
    ("stage", 0),
    ("stage", None),
    ("stage", "x"),
    ("stage", True),
    ("status", None),
    ("status", 1),
    ("status", ["ok"]),
    ("status", "weird"),
    ("status", "OK"),
    ("us", "3.0"),
    ("us", None),
    ("us", [1.0]),
    ("us", 0),
    ("us", -1.0),
    ("us", float("nan")),
    ("us", float("inf")),
    ("us", True),
    ("auto", "yes"),
    ("auto", 1),
    ("auto", None),
    ("tuning_id", None),
    ("tuning_id", ["a"]),
    ("tuning_id", 5),
    ("tuning_id", ""),
    ("knobs", None),
    ("knobs", 1),
    ("knobs", []),
    ("knobs", "x"),
    ("index", "0"),
    ("index", None),
    ("index", 1.5),
    ("index", -2),
    ("index", -1),  # a stage-1 ok row has a position in the walk
    ("index", True),
]


@pytest.mark.parametrize("field,value", _BAD_VALUES)
def test_rows_with_an_unwritable_value_are_usage_errors_in_every_reader(
    tmp_path, capsys, field, value
):
    """``--summarize``, ``--stage1-json`` and the report refuse a row holding a
    value the tool never writes with exit 2 and ``row N: field``; never a
    traceback, never a silently dropped row."""
    from benchmarks.gfx950.gdn import decode_defaults_report as report

    rows = _rows()
    rows[1][field] = value
    path = _file_of(tmp_path, rows)
    with pytest.raises(ValueError, match=f"row 1: {field}"):
        sweep.load_rows([path])
    for flag in ("--summarize", "--stage1-json"):
        _run_refused(capsys, [flag, path], path, f"row 1: {field}")
    capsys.readouterr()
    with pytest.raises(SystemExit) as err:
        report.main([path])
    assert err.value.code == 2
    assert f"row 1: {field}" in capsys.readouterr().err


@pytest.mark.parametrize("us", [1.0, "x", [], 0, -3])
def test_a_row_that_is_not_ok_has_no_time(tmp_path, capsys, us):
    rows = _rows()
    rows[1].update(status="fail", us=us, reason="gate")
    path = _file_of(tmp_path, rows)
    with pytest.raises(ValueError, match="row 1: us must be null on a fail row"):
        sweep.load_rows([path])
    _run_refused(capsys, ["--summarize", path], path, "row 1: us")


def test_every_row_shape_the_tool_writes_passes_the_row_check(tmp_path):
    """The rows the writers produce are all accepted: a timed row, the
    stage-2 ``auto`` row stage 1 did not list (``index`` -1), each status, and
    the row of a shape with nothing to time, at either stage."""
    auto, configs = sweep.shape_configs(SHAPE, sweep.options())
    # The run settings exactly as the tool writes them (``run_meta``).
    meta = sweep.run_meta(
        sweep.options(source_sha="abc", rotate_mb=1024), "gfx950", 1024
    )
    rows = _rows()
    stage2_auto = sweep.config_row(
        SHAPE, auto, -1, auto.spec.tuning_id, dict(meta, stage=2)
    )
    stage2_auto.update(status="ok", us=2.0, runs=[2.0])
    rows.append(stage2_auto)
    for status in ("fail", "error", "no_graph"):
        row = sweep.config_row(
            SHAPE, configs[2], 2, auto.spec.tuning_id, dict(meta, stage=1)
        )
        row.update(status=status, reason="x")
        rows.append(row)
    for stage in (1, 2):
        rows.append(
            sweep.shape_error_row(SHAPE, ValueError("refused"), meta, stage=stage)
        )
    path = _file_of(tmp_path, json.loads(json.dumps(rows)))
    assert len(sweep.load_rows([path])) == len(rows)


def test_the_walk_fields_are_one_list_every_copy_derives_from():
    """``run_meta`` builds its walk part from ``WALK_FIELDS``, and each row
    carries every one of them (``ROW_KEYS``), so a walk field added to the one
    list reaches the rows, the readers and the refused flags together."""
    meta = sweep.run_meta(sweep.options(source_sha="abc"), "gfx950", 1024)
    assert set(sweep.WALK_FIELDS) <= set(meta)
    assert set(sweep.WALK_FIELDS) <= set(sweep.ROW_KEYS)
    assert sweep.walk_of(meta) == tuple(meta[k] for k in sweep.WALK_FIELDS)


def test_every_row_key_has_exactly_one_validator():
    assert set(sweep.ROW_CHECKS) == set(sweep.ROW_KEYS)
    assert set(sweep.OPTIONAL_ROW_CHECKS) == {"cold_target_mb"}
    assert not set(sweep.OPTIONAL_ROW_CHECKS) & set(sweep.ROW_KEYS)


@pytest.mark.parametrize(
    "rotate_mb,cold_target_mb",
    [(0, -1), (0, 0), (float("inf"), 1024), (1024, float("inf")), (10**400, 1024)],
)
def test_a_rotation_or_target_that_cannot_mean_cold_is_refused(
    tmp_path, rotate_mb, cold_target_mb
):
    """``is_cold`` would call ``(0, -1)`` and ``(inf, 1024)`` cold, so the
    report would print a proposal from them: the row is refused instead."""
    rows = _rows()
    rows[0].update(rotate_mb=rotate_mb, cold_target_mb=cold_target_mb)
    with pytest.raises(ValueError, match="row 0: (rotate_mb|cold_target_mb)"):
        sweep.load_rows([_file_of(tmp_path, rows)])


# One value of every kind a hand-edited or truncated file can hold.
_FUZZ_VALUES = [
    None,
    True,
    False,
    0,
    -1,
    1.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    "",
    "x",
    [],
    [1],
    {},
    10**400,
]


@pytest.mark.parametrize("field", [*sweep.ROW_KEYS, "cold_target_mb"])
def test_no_value_of_any_row_field_is_a_traceback_in_any_reader(
    tmp_path, capsys, field
):
    """Every row field x every kind of bad value: ``--summarize``, the
    ``--stage1-json`` row loading and the report either exit 2 (a ``ValueError``
    naming the row, for the loading) because ``load_rows`` refuses the row, or
    load it and never raise; no other exception, ever."""
    from benchmarks.gfx950.gdn import decode_defaults_report as report

    refused = 0
    base = json.dumps(_rows())
    for n, value in enumerate(_FUZZ_VALUES):
        rows = json.loads(base)
        rows[1][field] = value
        path = _file_of(tmp_path, rows, f"{n}.json")
        try:
            sweep.load_rows([path])
            loads = True
        except ValueError as exc:
            loads = False
            refused += 1
            assert "row 1" in str(exc), (field, value)
        where = (field, value)

        def exit_code(call):
            capsys.readouterr()
            try:
                call()
            except SystemExit as exc:
                return exc.code
            return None

        for call in (
            lambda: sweep.main(["--summarize", path]),
            lambda: report.main([path]),
            lambda: report.main([path, "--stage", "2"]),
        ):
            assert exit_code(call) == (None if loads else 2), where
        try:
            sweep.stage1_shapes(sweep.options(stage1_json=[path]))
            assert loads, where
        except ValueError as exc:
            if not loads:
                assert "row 1" in str(exc), where
    assert refused, f"no bad value of {field} was refused"


def test_stage1_walk_refuses_rows_of_two_walks_directly():
    """``stage1_walk`` owns the mixed-walk refusal; a row without the walk
    fields never reaches it (``load_rows`` refuses that first)."""
    a = dict(_WALK)
    assert sweep.stage1_walk([a, dict(a)]) == _WALK
    with pytest.raises(ValueError, match="2 walks"):
        sweep.stage1_walk([a, dict(a, seed=1)])


def test_a_shape_entry_that_lacks_a_required_key_is_a_usage_error(tmp_path, capsys):
    full = {"gate_kind": "gdn", "mode": "plain", "state_dtype": "bf16", "heads": ["16/32"], "batch": 1}  # fmt: skip
    for key in ("gate_kind", "mode", "state_dtype", "heads", "batch"):
        entry = {k: v for k, v in full.items() if k != key}
        path = str(tmp_path / f"no_{key}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([full, entry], fh)
        _run_refused(capsys, ["--shapes", path, "--list-only"], path, "entry 1", key)
    # num_k_heads without num_v_heads, and neither heads form, lack a key too.
    for entry in (
        {k: v for k, v in full.items() if k != "heads"} | {"num_k_heads": 16},
        {k: v for k, v in full.items() if k != "heads"},
    ):
        with pytest.raises(ValueError, match="lacks num_"):
            sweep.expand_entry(entry)


@pytest.mark.parametrize(
    "bad",
    [
        dict(heads=["16-32"], batches=[1]),
        dict(heads=[""], batches=[1]),
        dict(heads=["16/32", " "], batches=[1]),
        dict(heads=5, batches=[1]),
        dict(heads="16/32", batches=[1]),
        dict(heads=[16], batches=[1]),
        dict(heads=["a/b"], batches=[1]),
        dict(heads=["16/32"], batches=5),
        dict(heads=["16/32"], batches="1,2"),
        dict(heads=["16/32"], batches=["a"]),
        dict(heads=["16/32"], batches=[0]),
        dict(heads=["16/32"], batches=[True]),
        dict(heads=["16/32"], batch="5"),
        dict(heads=["16/32"], batches=[1], mode=["fused"]),
        dict(heads=["16/32"], batches=[1], head_dim="128"),
    ],
    ids=lambda b: ",".join(f"{k}={v!r}" for k, v in b.items()),
)
def test_a_malformed_shape_entry_is_a_usage_error_naming_its_entry(
    tmp_path, capsys, bad
):
    """Whatever is wrong inside an entry, ``expand_entry`` raises a
    ``ValueError`` (never an ``ArgumentTypeError`` or ``TypeError``), so
    ``--shapes`` exits 2 naming the file and the entry."""
    good = {"gate_kind": "gdn", "mode": "plain", "state_dtype": "bf16", "heads": ["16/32"], "batch": 1}  # fmt: skip
    entry = {"gate_kind": "gdn", "mode": "plain", "state_dtype": "bf16", **bad}
    with pytest.raises(ValueError):
        sweep.expand_entry(entry)
    path = str(tmp_path / "shapes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([good, entry], fh)
    _run_refused(capsys, ["--shapes", path, "--list-only"], path, "entry 1")


def test_toolchain_capture_never_raises_and_always_names_its_fields():
    assert set(sweep._toolchain()) == {
        "rocm_version",
        "llvm_version",
        "torch_version",
        "backend",
        "llvm_flavor",
    }


def test_rotation_below_the_cold_target_is_not_cold(capsys):
    cold = dict(rotate_mb=1024, cold_target_mb=1024)
    assert sweep.is_cold(cold) and sweep.is_cold(dict(cold, rotate_mb=2048))
    assert not sweep.is_cold(dict(cold, rotate_mb=8))
    assert not sweep.is_cold(dict(cold, rotate_mb=0))
    assert not sweep.is_cold({"rotate_mb": 1024})  # no recorded target
    rows = _rows()
    rows[0]["rotate_mb"] = 8
    sweep.summarize(rows)
    assert "not timed in cold memory" in capsys.readouterr().out


# ---------------------------------------------------------------- shards, windows
_LIST_ONLY_IGNORED = (
    ("--output-json", "x.json"),
    ("--rotate-mb", "8"),
    ("--rounds", "3"),
    ("--s2-rounds", "4"),
    ("--runs", "1"),
    ("--top-k", "2"),
    ("--stage1-only",),
    ("--source-sha", "abc"),
)


@pytest.mark.parametrize("extra", _LIST_ONLY_IGNORED, ids=lambda e: e[0])
def test_list_only_refuses_every_flag_it_would_ignore(extra, capsys):
    _refused(capsys, extra[0], "--list-only", *extra)
    # The same flag is accepted where it is read.
    if extra[0] not in ("--output-json",):
        _parse(*extra)
    # A walk flag a listing does read stays accepted.
    _parse("--list-only", "--shard", "1/2", "--bases-only", "--seed", "3")


def test_summarize_of_no_rows_fails(tmp_path, capsys):
    assert sweep.summarize([]) == 1
    assert "no rows" in capsys.readouterr().out
    path = str(tmp_path / "empty.json")
    sweep.write_rows(path, [])
    assert sweep.main(["--summarize", path]) == 1


def test_a_stage2_shard_past_the_recorded_shapes_owns_nothing_and_succeeds(
    monkeypatch, tmp_path, capsys
):
    shapes = [sweep.Shape("gdn", "plain", "bf16", 16, 32, 128, b) for b in (1, 2, 3)]
    rows = []
    for shape in shapes:
        auto, configs = sweep.shape_configs(shape, sweep.options())
        row = sweep.config_row(
            shape, configs[0], 0, auto.spec.tuning_id, dict(_META, stage=1)
        )
        row.update(status="ok", us=1.0)
        rows.append(row)
    path = str(tmp_path / "s1.json")
    sweep.write_rows(path, rows)
    args = sweep.options(stage1_json=[path], shard=(5, 8))
    assert sweep.retime(args, {}) == []
    assert sweep.finish(args, [], [], recorded=len(shapes)) == 0
    assert "shard 5/8 owns no shape (3 shapes recorded)" in capsys.readouterr().out


def test_a_stage1_shard_that_owns_nothing_succeeds_but_a_no_op_window_fails(capsys):
    other = sweep.options(limit=1, shard=(1, 2), stage1_only=True)
    assert sweep.finish(other, [], [SHAPE], 0) == 0
    assert "owns no configuration" in capsys.readouterr().out
    count = len(sweep.shape_configs(SHAPE, sweep.options())[1])
    assert sweep.finish(sweep.options(offset=count), [], [SHAPE], 0) == 1
    assert "nothing was timed" in capsys.readouterr().err
