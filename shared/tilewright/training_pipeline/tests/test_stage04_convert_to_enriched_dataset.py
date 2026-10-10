# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
import json
import sys
from collections import defaultdict

import pytest

import stage04_convert_to_enriched_dataset as s04
from bench_fakes import (
    BANNER,
    LIBRARY_STEM,
    OTHER_STEM,
    Gemm,
    measured_lines,
    preamble,
    skip,
    solution,
    winner,
    write_logic,
)
from lib.bench_log import CANARY_FLAG_MARK
from lib.dat import KERNELS_FILE, KernelIndex, kernel_dat_info, read_kernel_attributes

GA = Gemm(256, 512, 1024)
GB = Gemm(64, 32, 4096, 4, "N", "T")
GC = Gemm(128, 128, 128)
GE = Gemm(512, 512, 512)
GF = Gemm(1024, 256, 2048)
GW = Gemm(7, 7, 7)

MT = {
    10: (128, 128, 64),
    11: (256, 128, 64),
    12: (64, 64, 64),
    13: (32, 32, 128),
    14: (16, 64, 128),
    15: (64, 32, 32),
    16: (128, 64, 128),
    17: (128, 64, 128),
    18: (256, 256, 64),
}


def _library_solutions():
    sols = []
    for local, (index, mt) in enumerate(sorted(MT.items())):
        mi = (0, 0, 0, 0) if index == 15 else (16, 16, 128, 1)
        sols.append(solution(index, local, mt, mi=mi, nta=4 if index == 13 else 0))
    return sols


def _log_lines():
    lines = list(BANNER)
    # A: shifted columns (splitK/wgm) and adaptive statistics; skip rows in
    # scientific and fixed notation.
    kw = dict(split_k=True, adaptive=True)
    lines += preamble(4, adaptive=True)
    lines += measured_lines(0, GA, 10, 20.0, MT[10], samples=40, cv=0.02, **kw)
    lines += measured_lines(1, GA, 11, 12.5, MT[11], samples=60, cv=0.01, **kw)
    lines += [skip(2, 12, "1.2e+02", best="12"), skip(3, 13, "456.79", best="12.50")]
    lines += winner(1, GA, 11, 12.5, mt=MT[11], **kw)
    # B: a single candidate, so no winner line.
    lines += preamble(1) + measured_lines(0, GB, 14, 3.0, MT[14])
    # C: cut short by a crash.
    lines += preamble(4) + measured_lines(0, GC, 10, 9.0, MT[10])
    lines += ["", "*** SKIPPED CRASHED GEMM at block-offset 2 (yaml line 3): ***"]
    lines += BANNER
    # D: no solution.
    lines += preamble(2)[:1] + ["error: NO solution found! at testing_matmul.hpp:1"]
    # E: Dot2 kernel, a solution outside the library, an unparsable skip line.
    lines += preamble(4) + measured_lines(0, GE, 15, 9.0, MT[15])
    lines += measured_lines(1, GE, 90, 5.0, (512, 512, 64))
    lines += ["Skip solution: 2 (garbled)", skip(3, 16, "50")]
    lines += winner(1, GE, 90, 5.0, mt=(512, 512, 64))
    # F: two kernels with identical parameters.
    lines += preamble(3) + measured_lines(0, GF, 16, 8.0, MT[16])
    lines += measured_lines(1, GF, 17, 6.0, MT[17])
    lines += measured_lines(2, GF, 18, 7.0, MT[18])
    lines += winner(1, GF, 17, 6.0, mt=MT[17])
    return lines


@pytest.fixture
def dataset(tmp_path):
    dat_dir = tmp_path / "library"
    dat_dir.mkdir()
    write_logic(dat_dir / (LIBRARY_STEM + ".dat.zlib"), _library_solutions())
    write_logic(dat_dir / (OTHER_STEM + ".dat"), [solution(90, 0, (512, 512, 64))])
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "block_0000_lines1-6_gpu0.log").write_text("\n".join(_log_lines()) + "\n")
    retry = BANNER + preamble(2) + measured_lines(0, GC, 10, 4.0, MT[10])
    retry += measured_lines(1, GC, 11, 5.0, MT[11]) + winner(0, GC, 10, 4.0, mt=MT[10])
    (logs / "block_retry_gpu0.log").write_text("\n".join(retry) + "\n")
    warm = BANNER + preamble(1) + measured_lines(0, GW, 10, 1.0, MT[10])
    (logs / "warmup_0000_gpu0.log").write_text("\n".join(warm) + "\n")
    return tmp_path


def _run(monkeypatch, root, *extra):
    argv = ["stage04", "--log-dir", root / "logs", "--out-dir", root / "out"]
    argv += ["--dat-dir", root / "library", "--quiet", *extra]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    return s04.main()


def _rows(out_dir):
    rows, headers = [], set()
    for path in sorted(out_dir.glob("chunk_*.csv")):
        with path.open() as f:
            reader = csv.DictReader(f)
            headers.add(tuple(reader.fieldnames))
            rows += list(reader)
    assert headers == {tuple(s04.OUT_FIELDS)}
    by_gemm = defaultdict(list)
    for r in rows:
        by_gemm[(int(r["m"]), int(r["n"]), int(r["k"]), int(r["batch_count"]))].append(
            r
        )
    return by_gemm


def _gemm(g):
    return (g.m, g.n, g.k, g.batch)


def test_enriched_rows(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    by_gemm = _rows(dataset / "out")
    assert set(by_gemm) == {_gemm(g) for g in (GA, GB, GC, GE, GF)}

    a = {int(r["rank"]): r for r in by_gemm[_gemm(GA)]}
    assert sorted(a) == [0, 1, 2, 3]
    assert a[0]["us"] == "20" and a[1]["us"] == "12.5"
    assert a[0]["rotating_buffer"] == "16" and a[0]["transA"] == "T"
    assert (a[0]["samples"], a[0]["cv"], a[0]["status"]) == ("40", "0.02", "converged")
    assert (a[0]["is_origami_pick"], a[1]["is_origami_pick"]) == ("1", "0")
    assert [a[r]["is_winner"] for r in range(4)] == ["False", "True", "False", "False"]
    assert (a[2]["is_skip"], a[2]["us"], a[2]["sol_idx_global"]) == (
        "True",
        "120.0",
        "12",
    )
    assert (a[3]["us"], a[3]["cache_hints_a"], a[3]["samples"]) == ("456.79", "4", "")
    assert a[2]["m"] == "256" and a[2]["compute_type"] == "f32_r"
    assert a[2]["hipblaslt-Gflops"] == ""
    assert (a[1]["mt_m"], a[1]["sol_idx_local"], a[1]["mi_k"]) == ("256", "1", "128")

    (b,) = by_gemm[_gemm(GB)]
    assert (b["is_winner"], b["is_origami_pick"], b["transA"]) == ("True", "1", "N")

    c = {int(r["rank"]): r for r in by_gemm[_gemm(GC)]}
    assert sorted(c) == [0, 1] and c[0]["us"] == "4" and c[0]["is_winner"] == "True"

    e = {int(r["rank"]): r for r in by_gemm[_gemm(GE)]}
    assert sorted(e) == [0, 3]
    assert (e[0]["mi_m"], e[0]["mi_n"], e[0]["mi_k"]) == ("1", "1", "64")
    assert e[0]["is_winner"] == "True" and e[3]["is_skip"] == "True"

    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    counters = summary["counters"]
    assert summary["unparsed_skip_lines"] == 1
    assert summary["unparsed_skip_examples"] == ["Skip solution: 2 (garbled)"]
    assert (summary["gemms"], summary["library_files"]) == (
        5,
        [LIBRARY_STEM + ".dat.zlib"],
    )
    assert summary["invalid_library_files"] == []
    assert summary["missing_size_mapping_keys"] == {}
    assert counters["problems_incomplete"] == 1
    assert counters["problems_no_solution"] == 1
    assert counters["tested_outside_library"] == 1
    assert counters["problems_kept"] == 5
    assert "rows_collapsed" not in counters


def test_execution_context_columns(dataset, monkeypatch):
    assert _run(monkeypatch, dataset) == 0
    rows = [r for g in _rows(dataset / "out").values() for r in g]
    assert {(r["cu_budget"], r["schedule"]) for r in rows} == {("0", "default")}
    args = ("--cu-budget", "64", "--schedule", "dynamic")
    assert _run(monkeypatch, dataset, *args) == 0
    rows = [r for g in _rows(dataset / "out").values() for r in g]
    assert {(r["cu_budget"], r["schedule"]) for r in rows} == {("64", "dynamic")}
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["context"] == {"cu_budget": 64, "schedule": "dynamic"}
    with pytest.raises(SystemExit):
        _run(monkeypatch, dataset, "--cu-budget", "-1")


def test_kernels_with_identical_parameters_keep_their_own_rows(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    f = {int(r["rank"]): r for r in _rows(dataset / "out")[_gemm(GF)]}
    assert sorted(f) == [0, 1, 2]
    assert [f[r]["sol_idx_global"] for r in range(3)] == ["16", "17", "18"]
    assert [f[r]["us"] for r in range(3)] == ["8", "6", "7"]
    assert [f[r]["is_origami_pick"] for r in range(3)] == ["1", "0", "0"]
    assert [f[r]["is_winner"] for r in range(3)] == ["False", "True", "False"]
    assert all(f[0][c] == f[1][c] for c in s04.CONFIG_FIELDS)


def test_kernel_attributes_of_every_written_solution(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    out = dataset / "out"
    written = {int(r["sol_idx_global"]) for rs in _rows(out).values() for r in rs}
    got = read_kernel_attributes(out / KERNELS_FILE)
    assert set(got) == written == set(range(10, 19))
    by_index = {s["index"]: s for s in _library_solutions()}
    for sid, attributes in got.items():
        assert attributes == kernel_dat_info(by_index[sid])["attributes"]
    summary = json.loads((out / "enrich_summary.json").read_text())
    assert summary["n_kernels"] == 9


def test_library_problems_are_reported(dataset, monkeypatch):
    sols = _library_solutions()
    del sols[0]["sizeMapping"]["WaveGroup"]
    write_logic(dataset / "library" / (LIBRARY_STEM + ".dat.zlib"), sols)
    write_logic(
        dataset / "library" / (OTHER_STEM + ".dat"),
        [solution(90, 0, (512, 512, 64), streamK=1)],
    )
    assert _run(monkeypatch, dataset) == 0
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["missing_size_mapping_keys"] == {"WaveGroup": 1}
    assert summary["library_files"] == [LIBRARY_STEM + ".dat.zlib"]
    assert summary["invalid_library_files"] == [
        f"{OTHER_STEM}.dat: solution 90: unsupported legacy streamK mode 1"
    ]
    assert _run(monkeypatch, dataset, "--library-stem", OTHER_STEM) == 1


def test_without_library_stem_every_library_is_loaded(dataset, monkeypatch):
    assert _run(monkeypatch, dataset) == 0
    e = {int(r["rank"]): r for r in _rows(dataset / "out")[_gemm(GE)]}
    assert sorted(e) == [0, 1, 3]
    assert e[1]["is_winner"] == "True" and e[1]["sol_idx_global"] == "90"


def test_chunking_and_stale_chunks(dataset, monkeypatch):
    out = dataset / "out"
    out.mkdir()
    (out / "chunk_0099.csv").write_text("stale\n")
    assert _run(monkeypatch, dataset, "--rows-per-chunk", 3) == 0
    chunks = sorted(p.name for p in out.glob("chunk_*.csv"))
    assert chunks == [f"chunk_{i:04d}.csv" for i in range(len(chunks))]
    assert len(chunks) == 5
    assert sum(len(v) for v in _rows(out).values()) == 13


def test_max_gemms_limits_the_output(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--max-gemms", 2) == 0
    assert len(_rows(dataset / "out")) == 2


def test_bad_inputs(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--library-stem", "TensileLibrary_missing") == 1
    assert (
        _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM, "--scale-mode", 1)
        == 1
    )
    empty = dataset / "logs_empty"
    empty.mkdir()
    (empty / "block_0000_lines1-1_gpu0.log").write_text(
        "\n".join(BANNER + preamble(1)[:1] + ["NO solution found!"]) + "\n"
    )
    argv = ["stage04", "--log-dir", empty, "--out-dir", dataset / "o2"]
    argv += ["--dat-dir", dataset / "library", "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    assert s04.main() == 2


def _config(path, **changes):
    """A run config whose hipblaslt section is the logs' F8 MX problem type
    with `changes`."""
    import yaml

    hl = {
        "a_type": "f8_r",
        "b_type": "f8_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "scale_type": "f32_r",
        "compute_type": "c_f32_r",
        "scaleA": 3,
        "scaleB": 3,
        "transA": "T",
        "transB": "N",
        "library_stem": LIBRARY_STEM,
    }
    hl.update(changes)
    path.write_text(yaml.safe_dump({"hipblaslt": hl}))
    return path


def test_rows_of_another_problem_type_stop_the_stage(dataset, monkeypatch, capsys):
    def run(**changes):
        cfg = _config(dataset / "config.yaml", **changes)
        stem = ["--library-stem", LIBRARY_STEM]
        rc = _run(monkeypatch, dataset, *stem, "--config-yaml", cfg)
        return rc, capsys.readouterr().err

    assert run()[0] == 0
    assert len(_rows(dataset / "out")) == 5
    rc, err = run(a_type="bf16_r", b_type="bf16_r", scaleA=0, scaleB=0)
    assert rc == 1
    assert "block_0000_lines1-6_gpu0.log: problem (256, 512, 1024, 1)" in err
    assert (
        "ran a_type f8_r (the config's: bf16_r), b_type f8_r (the config's: "
        "bf16_r), scaleA 3 (the config's: 0), scaleB 3 (the config's: 0): the "
        "bench ran another problem type than the config's"
    ) in err
    rc, err = run(scaleB=1)
    assert rc == 1 and "scaleB 3 (the config's: 1)" in err and "scaleA" not in err
    rc, err = run(compute_type="c_xf32_r")
    assert rc == 1 and "compute_type f32_r (the config's: xf32_r)" in err
    assert run(compute_type="c_f32_pedantic_r")[0] == 0
    (dataset / "bad.yaml").write_text("runtime: {}\n")
    rc = _run(monkeypatch, dataset, "--config-yaml", dataset / "bad.yaml", "--quiet")
    assert rc == 1 and "has no hipblaslt section" in capsys.readouterr().err


def test_scale_modes_printed_by_name_are_compared_by_mode(dataset):
    from lib.bench_log import parse_log_file
    from lib.bench_yaml import printed_types

    hl = {"a_type": "f8_r", "b_type": "f8_r", "c_type": "bf16_r", "d_type": "bf16_r"}
    expect = printed_types(dict(hl, compute_type="c_f32_r", scaleA=3, scaleB=3))
    enricher = s04.Enricher(KernelIndex(by_index={}), quiet=True)
    enricher.expect = expect
    named = "\n".join(_log_lines()).replace(",f32_r,3,3,", ",f32_r,Block_32_UE8M0,3,")
    log = dataset / "named.log"
    log.write_text(named + "\n")
    for prob in parse_log_file(log):
        enricher.check_types(prob)
    enricher.expect = dict(expect, scaleA=1)
    with pytest.raises(s04.TypeMismatch, match="scaleA Block_32_UE8M0"):
        for prob in parse_log_file(log):
            enricher.check_types(prob)


def test_kernel_name_mismatches_become_fatal():
    info = kernel_dat_info(solution(10, 0, MT[10]))
    enricher = s04.Enricher(KernelIndex(by_index={10: info}), quiet=True)
    assert enricher.name_matches(10, info["kernel_name"])
    assert enricher.name_matches(10, "") and enricher.name_matches(99, "Cijk_X")
    assert enricher.name_mismatches == 0
    same_tile = info["kernel_name"].replace("_SN_TEST", "_SN_OTHER")
    assert not any(enricher.name_matches(10, same_tile) for _ in range(10))
    assert enricher.name_mismatches == 10
    with pytest.raises(RuntimeError, match="disagree.*build changed"):
        enricher.name_matches(10, "Cijk_X_MT256x128x64_MI16_Y")


# ── planned runs ─────────────────────────────────────────────────────────────


def _plan(path, entries):
    shapes = [
        {"m": g.m, "n": g.n, "k": g.k, "batch": g.batch, "kernels": kernels}
        for g, kernels in entries
    ]
    path.write_text(json.dumps({"version": 1, "shapes": shapes}))
    return path


def test_rows_without_plan_count_as_planned(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    for rows in _rows(dataset / "out").values():
        assert {(r["is_planned"], r["n_planned"]) for r in rows} == {("1", "")}
    assert s04.OUT_FIELDS[-2:] == ["is_planned", "n_planned"]


def test_planned_runs_record_list_membership(dataset, monkeypatch):
    plan = _plan(
        dataset / "plan.json",
        [(GA, [11, 12, 10]), (GF, None), (GE, [15, 16, 90, 77])],
    )
    assert (
        _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM, "--plan", plan) == 0
    )
    by_gemm = _rows(dataset / "out")
    a = {int(r["rank"]): r for r in by_gemm[_gemm(GA)]}
    assert [a[r]["is_planned"] for r in range(4)] == ["1", "1", "1", "0"]
    assert {a[r]["n_planned"] for r in range(4)} == {"3"}
    assert {r["is_origami_pick"] for r in a.values()} == {"0"}
    assert [a[r]["is_winner"] for r in range(4)] == ["False", "True", "False", "False"]
    f = by_gemm[_gemm(GF)]
    assert {(r["is_planned"], r["n_planned"], r["is_origami_pick"]) for r in f} == {
        ("1", "", "0")
    }
    e = {int(r["rank"]): r for r in by_gemm[_gemm(GE)]}
    assert (e[0]["is_planned"], e[3]["is_planned"], e[0]["n_planned"]) == (
        "1",
        "1",
        "4",
    )
    (b,) = by_gemm[_gemm(GB)]
    assert (b["is_planned"], b["n_planned"], b["is_origami_pick"]) == ("1", "", "1")
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["counters"]["rows_unplanned"] == 1
    assert summary["counters"]["gemms_in_plan"] == 3
    assert summary["plan"] == str(plan)


def test_the_sacrificial_copy_never_becomes_a_row(dataset, monkeypatch):
    lines = list(BANNER) + preamble(4)
    lines += [skip(0, 12, "95.00", best="9.00")]
    lines += measured_lines(1, GE, 12, 9.0, MT[12])
    lines += measured_lines(2, GE, 10, 11.0, MT[10])
    lines += measured_lines(3, GE, 11, 10.0, MT[11])
    lines += winner(1, GE, 12, 9.0, mt=MT[12])
    logs = dataset / "logs"
    for p in logs.glob("*.log"):
        p.unlink()
    (logs / "block_0000_lines1-1_gpu0.log").write_text("\n".join(lines) + "\n")
    plan = _plan(dataset / "plan.json", [(GE, [12, 10, 11])])
    assert (
        _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM, "--plan", plan) == 0
    )
    rows = {int(r["rank"]): r for r in _rows(dataset / "out")[_gemm(GE)]}
    assert sorted(rows) == [1, 2, 3]
    assert rows[1]["sol_idx_global"] == "12" and rows[1]["is_skip"] == "False"
    assert rows[1]["is_winner"] == "True" and rows[1]["us"] == "9"
    assert {r["n_planned"] for r in rows.values()} == {"3"}
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["counters"]["sacrificial_rows"] == 1
    assert summary["counters"]["rows_tested"] == 3


def test_several_log_dirs_and_extra_library(dataset, monkeypatch):
    seed_logs = dataset / "seed_logs"
    seed_logs.mkdir()
    g = Gemm(96, 96, 96)
    lines = BANNER + preamble(2) + measured_lines(0, g, 90, 2.0, (512, 512, 64))
    lines += measured_lines(1, g, 12, 3.0, MT[12]) + winner(0, g, 90, 2.0)
    (seed_logs / "block_0000_lines1-1_gpu0.log").write_text("\n".join(lines) + "\n")
    rc = _run(
        monkeypatch,
        dataset,
        "--library-stem",
        LIBRARY_STEM,
        "--log-dir",
        seed_logs,
        "--extra-library-stem",
        OTHER_STEM,
    )
    assert rc == 0
    by_gemm = _rows(dataset / "out")
    assert _gemm(g) in by_gemm and _gemm(GA) in by_gemm
    rows = {r["sol_idx_global"]: r for r in by_gemm[_gemm(g)]}
    assert set(rows) == {"90", "12"} and rows["90"]["is_winner"] == "True"
    e = {int(r["rank"]): r for r in by_gemm[_gemm(GE)]}
    assert sorted(e) == [0, 1, 3]
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["log_files"] == 3
    assert summary["extra_library_stems"] == [OTHER_STEM]


def test_the_launch_records_of_the_log_dirs_are_summed_up(dataset, monkeypatch, capsys):
    """enrich_summary.json carries what the benches of every log directory
    launched the kernels with, and two launch heuristics are warned about."""
    a = {"overrides": {}, "heuristic": {"digest": "sha256:" + "a" * 64}}
    (dataset / "platform.json").write_text(
        json.dumps({"launch": {**a, "workspace_bytes": 8}})
    )
    seed = dataset / "seed"
    (seed / "logs").mkdir(parents=True)
    (seed / "logs" / "block_0000_lines1-1_gpu0.log").write_text(
        (dataset / "logs" / "block_retry_gpu0.log").read_text()
    )
    rc = _run(monkeypatch, dataset)
    assert rc == 0
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["launch"] == {
        "heuristics": ["sha256:" + "a" * 64],
        "overrides": {},
        "workspace_bytes": [8],
        "unknown": 0,
    }
    capsys.readouterr()
    merged = {
        "heuristics": ["sha256:" + "b" * 64],
        "overrides": {"TENSILE_PRECISE_SM_TARGET": ["1"]},
        "workspace_bytes": [8],
        "unknown": 0,
    }
    (seed / "rebench_summary.json").write_text(json.dumps({"launch": merged}))
    assert _run(monkeypatch, dataset, "--log-dir", seed / "logs") == 0
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["launch"]["heuristics"] == [
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    ]
    assert summary["launch"]["overrides"] == {"TENSILE_PRECISE_SM_TARGET": ["1"]}
    err = capsys.readouterr().err
    assert "2 launch heuristics" in err and "b" * 16 in err
    (dataset / "platform.json").unlink()
    assert _run(monkeypatch, dataset) == 0
    summary = json.loads((dataset / "out" / "enrich_summary.json").read_text())
    assert summary["launch"]["heuristics"] == [] and summary["launch"]["unknown"] == 1


def test_rows_of_a_log_its_canary_flagged_are_marked(dataset, monkeypatch):
    """stage03 kept the retry pass's data although its canary read the host
    as loaded: every row of that log, and of no other, is marked."""
    with (dataset / "logs" / "block_retry_gpu0.log").open("a") as f:
        f.write(f"\n{CANARY_FLAG_MARK}: canary 30 us on gpu0, again; kept ***\n")
    assert _run(monkeypatch, dataset, "--library-stem", LIBRARY_STEM) == 0
    flags = {
        g: {r["canary_flagged"] for r in rows}
        for g, rows in _rows(dataset / "out").items()
    }
    assert flags == {
        **{_gemm(g): {"0"} for g in (GA, GB, GE, GF)},
        _gemm(GC): {"1"},
    }
    counters = json.loads((dataset / "out" / "enrich_summary.json").read_text())[
        "counters"
    ]
    assert (counters["logs_canary_flagged"], counters["problems_canary_flagged"]) == (
        1,
        1,
    )


def test_extra_library_needs_a_stem(dataset, monkeypatch):
    assert _run(monkeypatch, dataset, "--extra-library-stem", OTHER_STEM) == 1


def test_unreadable_plan_is_an_error(dataset, monkeypatch):
    (dataset / "plan.json").write_text("{not json")
    rc = _run(
        monkeypatch,
        dataset,
        "--library-stem",
        LIBRARY_STEM,
        "--plan",
        dataset / "plan.json",
    )
    assert rc == 1
