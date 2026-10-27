# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import msgpack
import pytest
import yaml

import bench_fakes as bf
from test_stage05_train import (
    CELLS3,
    HARDWARE,
    PIPELINE_DIR,
    kernel_pool,
    latency_us,
    library_solutions,
    run_stage,
    shapes_for,
    train,
    write_config,
    write_round,
)

pytest.importorskip("tilewright")

import stage08_optional_validate_selection_efficiency as s08  # noqa: E402
from lib.bench_yaml import read_bench_lines  # noqa: E402

STEM = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
FIXED_ITERS = re.compile(r"(?<![\w])(cold_)?iters:")
HIPBLASLT = {
    "a_type": "bf16_r",
    "b_type": "bf16_r",
    "c_type": "bf16_r",
    "d_type": "bf16_r",
    "scale_type": "f32_r",
    "compute_type": "c_f32_r",
    "transA": "T",
    "transB": "N",
    "library_stem": STEM,
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s08")
    write_round(tmp / "round_0" / "stage04", shapes_for(20, 24, CELLS3[:2]))
    train(tmp, [tmp / "round_0" / "stage04"], tmp / "round_0" / "stage05")
    write_round(tmp / "heldout", shapes_for(21, 15, CELLS3))
    write_config(tmp / "config.yaml")
    return tmp


def stage08(tmp, out, *extra):
    run_stage(
        "stage08_optional_validate_selection_efficiency.py",
        "--train-dir",
        tmp / "round_0" / "stage05",
        "--arch",
        "gfx950",
        "--config-yaml",
        tmp / "config.yaml",
        "--weight-dtype",
        "int8",
        "--out-dir",
        out,
        "--dataset",
        f"heldout={tmp / 'heldout'}",
        "--bootstrap",
        50,
        "--quiet",
        *extra,
    )
    return json.loads((out / "summary.json").read_text())["datasets"]


def test_every_gemm_is_scored_as_deployed(trained):
    out = trained / "v1"
    (ds,) = stage08(trained, out)
    assert ds["rc"] == 0
    s = ds["summary"]
    assert s["weight_dtype"] == "int8" and s["n_routing_mismatch"] == 0
    overall = s["overall"]
    assert overall["n_gemms"] == overall["n_evaluated"] == 45
    assert overall["paired"]["n"] == 45
    assert ds["model_sel_eff"] == pytest.approx(overall["sel_eff"])
    assert ds["n_evaluated"] == 45 and ds["n_miss"] == overall["n_miss"]
    assert ds["paired_n"] == 45 and s["pick_measurement"] == "warmup"
    assert ds["paired_model_sel_eff"] == pytest.approx(
        overall["paired"]["model_sel_eff"]
    )
    assert "model_geomean_sel_eff" not in ds
    per_cell = json.loads((out / "heldout" / "per_cell_sel_eff.json").read_text())
    rows = {r["cell"]: r for r in per_cell["per_cell"]}
    unmodeled = rows[CELLS3[2]]
    assert unmodeled["model_cell"] is None and unmodeled["n_origami_fallback"] == 15
    assert unmodeled["model_sel_eff"] == pytest.approx(unmodeled["origami_sel_eff"])
    assert rows[CELLS3[0]]["model_cell"] == CELLS3[0]
    with (out / "heldout" / "per_gemm.csv").open() as f:
        per_gemm = list(csv.DictReader(f))
    assert len(per_gemm) == 45
    assert {r["served_by"] for r in per_gemm} >= {"model", "origami_fallback"}
    assert sum(1 for r in per_gemm if r["miss"]) == ds["n_miss"]
    for r in rows.values():
        assert r["n_miss"] <= r["n_evaluated"]
        assert r["paired_model_sel_eff"] == pytest.approx(r["paired"]["model_sel_eff"])


def test_library_stem_restricts_the_pool(trained, tmp_path):
    keep = kernel_pool()[::2]
    sols = [
        {
            "index": k["sol_idx_global"],
            "libraryLogicIndex": k["sol_idx_global"] - 100,
            "name": f"kernel{k['sol_idx_global']}",
            "sizeMapping": {
                "macroTile": [k["mt_m"], k["mt_n"], 1],
                "depthU": k["mt_k"],
                "matrixInstruction": [k["mi_m"], k["mi_n"], k["mi_k"], 1],
                "CUOccupancy": k["occupancy"],
                "nonTemporalA": k["cache_hints_a"],
                "nonTemporalB": k["cache_hints_b"],
                "grvwA": k["grvw_a"],
                "grvwB": k["grvw_b"],
                "gwvwD": k["gwvw_d"],
            },
        }
        for k in keep
    ]
    (tmp_path / f"{STEM}.dat").write_bytes(msgpack.packb({"solutions": sols}))
    (ds,) = stage08(
        trained,
        trained / "v2",
        "--library-dir",
        tmp_path,
        "--library-stem",
        STEM,
    )
    s = ds["summary"]
    assert s["library_stem"] == STEM and s["n_out_of_library_rows"] > 0
    with (trained / "v2" / "heldout" / "per_gemm.csv").open() as f:
        rows = list(csv.DictReader(f))
    picks = {int(r["pick_sol_idx"]) for r in rows if r["pick_sol_idx"]}
    assert picks <= {k["sol_idx_global"] for k in keep}
    sid_at = [k["sol_idx_global"] for k in keep]
    for r in rows:
        if r["pick_index"]:
            assert sid_at[int(r["pick_index"])] == int(r["pick_sol_idx"])
        if r["origami_index"]:
            assert sid_at[int(r["origami_index"])] == int(r["origami_sol_idx"])
        if r["served_by"] == "origami_fallback":
            assert r["pick_index"] == r["origami_index"]
    assert {r["served_by"] for r in rows if r["pick_index"]} >= {"model"}


def test_bad_dataset_fails_the_stage(trained, tmp_path):
    (tmp_path / "ds.yaml").write_text("- {M: 64, N: 64, K: 64, batch_count: 1}\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(
                PIPELINE_DIR
                / "stages"
                / "stage08_optional_validate_selection_efficiency.py"
            ),
            "--train-dir",
            str(trained / "round_0" / "stage05"),
            "--arch",
            "gfx950",
            "--config-yaml",
            str(trained / "config.yaml"),
            "--out-dir",
            str(tmp_path / "out"),
            "--dataset",
            f"y={tmp_path / 'ds.yaml'}",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 1
    (ds,) = json.loads((tmp_path / "out" / "summary.json").read_text())["datasets"]
    assert ds["rc"] == 1 and "bench-binary" in ds["reason"]


def test_a_routing_mismatch_fails_the_dataset(trained, tmp_path, monkeypatch):
    from lib import evaluate as ev

    real = ev.evaluate_model

    def skewed(*a, **kw):
        out = real(*a, **kw)
        out.n_routing_mismatch = 2
        return out

    monkeypatch.setattr(ev, "evaluate_model", skewed)
    out = tmp_path / "out"
    argv = ["stage08", "--train-dir", trained / "round_0" / "stage05"]
    argv += ["--arch", "gfx950", "--config-yaml", trained / "config.yaml"]
    argv += ["--out-dir", out, "--dataset", f"heldout={trained / 'heldout'}"]
    argv += ["--bootstrap", 0, "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    assert s08.main() == 1
    (ds,) = json.loads((out / "summary.json").read_text())["datasets"]
    assert ds["rc"] == 1 and "2 of 45 GEMMs" in ds["reason"]
    assert ds["summary"]["n_routing_mismatch"] == 2
    assert (out / "heldout" / "per_gemm.csv").is_file()


def test_dataset_library_dirs_use_the_build_dir_and_arch_base():
    got = s08.dataset_library_dirs(["a=/b/build", "c:/d"], "gfx1250-strict:xnack+")
    assert got == {
        "a": Path("/b/build/Tensile/library/gfx1250-strict"),
        "c": Path("/d/Tensile/library/gfx1250-strict"),
    }
    with pytest.raises(ValueError):
        s08.dataset_library_dirs(["nodelimiter"], "gfx950")


@pytest.mark.parametrize("probe", [None, {"skip_ratio_mode": "sigmoid"}])
def test_yaml_dataset_is_benched_by_stage03_alone(tmp_path, monkeypatch, probe):
    cfg = {"hipblaslt": HIPBLASLT, "bench": {"adaptive": {"measure_time": 2.0}}}
    if probe is not None:
        cfg.update(probe=probe, bench={"adaptive": {"enabled": False}})
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    dataset = tmp_path / "ds.yaml"
    dataset.write_text("- {M: 64, N: 64, K: 64, batch_count: 1}\n")
    bench = tmp_path / "hipblaslt-bench"
    bench.write_text("")
    calls = []
    monkeypatch.setattr(
        s08, "_run_subprocess", lambda cmd, log, quiet=False: calls.append(cmd) or 0
    )
    args = argparse.Namespace(
        bench_binary=bench,
        config_yaml=config,
        devices="0",
        blocks_per_gpu=1,
        quiet=True,
    )
    _, error = s08._bench_yaml("ds", dataset, tmp_path / "out", args)
    assert error is None
    assert [Path(cmd[2]).name for cmd in calls] == [
        "stage03_load_balance_offline_tuning.py"
    ]
    (cmd,) = calls
    assert cmd[cmd.index("--in-dir") + 1] == str(tmp_path / "out" / "bench_input")
    assert cmd[cmd.index("--config-yaml") + 1] == str(config)
    (line,) = read_bench_lines(tmp_path / "out" / "bench_input" / "shapes.yaml")
    assert "adaptive: true" in line and not FIXED_ITERS.search(line)
    assert ("measure_time: 2.0" in line) == (probe is None)
    assert s08._bench_input_yamls(tmp_path / "out") == [
        tmp_path / "out" / "bench_input" / "shapes.yaml"
    ]


def test_yaml_entries_become_training_bench_lines(tmp_path, monkeypatch):
    from lib import bench_yaml
    from lib.shapes import Shape

    entries = [
        {"M": 64, "N": 128, "K": 256, "batch_count": 1, "a_type": "bf16_r"},
        {"M": 64, "N": 128, "K": 256, "transA": "T", "iters": 2},
        {"M": 64, "N": 128, "K": 256, "transA": "N"},
        {"M": 8, "N": 8, "K": 8, "batch_count": 4, "compute_type": "f32_r"},
        {"M": 8, "N": 8, "K": 8, "batch_count": 4, "compute_type": "f16_r"},
        {"N": 8, "K": 8},
        "not a mapping",
    ]
    shapes, counts = s08.dataset_shapes(entries, HIPBLASLT)
    assert shapes == [Shape(64, 128, 256, 1), Shape(8, 8, 8, 4)]
    assert counts == {
        "n_entries": 6,
        "n_other_type": 2,
        "n_repeated": 1,
        "n_unreadable": 1,
    }
    cfg = {
        "hipblaslt": HIPBLASLT,
        "bench": {
            "skip_slow_solution_ratio": 0.5,
            "rotating_mb": 512,
            "rotating_target_blocks": 16,
            "adaptive": {"measure_time": 1.5},
        },
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    dataset = tmp_path / "ds.yaml"
    dataset.write_text(yaml.safe_dump(entries))
    bench = tmp_path / "hipblaslt-bench"
    bench.write_text("")
    monkeypatch.setattr(s08, "_run_subprocess", lambda cmd, log, quiet=False: 0)
    args = argparse.Namespace(
        bench_binary=bench,
        config_yaml=config,
        devices="0",
        blocks_per_gpu=1,
        quiet=True,
    )
    logs, error = s08._bench_yaml("ds", dataset, tmp_path / "out", args)
    assert error is None and logs == tmp_path / "out" / "bench" / "logs"
    lines = (tmp_path / "out" / "bench_input" / "shapes.yaml").read_text().splitlines()
    assert lines == [bench_yaml.render_bench_line(s, cfg) for s in shapes]
    for line in lines:
        assert "flush: true" in line and "adaptive: true" in line
        assert "skip_slow_solution_ratio: 0.5" in line
        assert not FIXED_ITERS.search(line)
        assert int(re.search(r"rotating: (\d+)", line).group(1)) >= 1
    counts_file = json.loads((tmp_path / "out" / "bench_input.json").read_text())
    assert counts_file["n_benched"] == 2
    config.write_text(yaml.safe_dump({"bench": cfg["bench"]}))
    _, error = s08._bench_yaml("ds", dataset, tmp_path / "o2", args)
    assert "hipblaslt section" in error
    config.write_text(yaml.safe_dump(cfg))
    dataset.write_text(yaml.safe_dump([entries[2]]))
    _, error = s08._bench_yaml("ds", dataset, tmp_path / "o3", args)
    assert error.startswith("no problem of the config's data types and layout")


def test_the_external_reference_prints_only_on_request(trained):
    out = trained / "printed"
    common = [
        "--train-dir",
        trained / "round_0" / "stage05",
        "--arch",
        "gfx950",
        "--config-yaml",
        trained / "config.yaml",
        "--dataset",
        f"heldout={trained / 'heldout'}",
        "--bootstrap",
        0,
    ]
    plain = run_stage(
        "stage08_optional_validate_selection_efficiency.py",
        *common,
        "--out-dir",
        out / "plain",
    )
    assert "origami" not in plain.stdout.lower() and "misses" in plain.stdout
    ref = run_stage(
        "stage08_optional_validate_selection_efficiency.py",
        *common,
        "--out-dir",
        out / "ref",
        "--origami-reference",
    )
    assert "[reference] Origami on the same 45 GEMMs" in ref.stdout


def test_a_dataset_of_another_build_is_read_at_its_kernels(trained, tmp_path):
    """The held-out rows, recorded as enriched with build A, evaluated with
    build B, which numbers every kernel otherwise: the same kernels are
    picked and charged, named by B's solution indices, kernel ids and
    signatures in per_gemm.csv."""
    import shutil

    from lib import kernel_identity as ki
    from test_kernel_identity import pool_library, record_rows

    ident_a = pool_library(tmp_path / "a", kernel_pool(), stem=STEM)
    ident_b = pool_library(tmp_path / "b", kernel_pool(), offset=1000, stem=STEM)
    data = record_rows(
        shutil.copytree(trained / "heldout", tmp_path / "data"), ident_a, STEM
    )

    def evaluate(lib, dataset, out):
        run_stage(
            "stage08_optional_validate_selection_efficiency.py",
            "--train-dir", trained / "round_0" / "stage05", "--arch", "gfx950",
            "--config-yaml", trained / "config.yaml", "--weight-dtype", "int8",
            "--out-dir", out, "--dataset", f"heldout={dataset}", "--bootstrap", 50,
            "--library-dir", lib, "--library-stem", STEM, "--quiet",
        )  # fmt: skip
        (ds,) = json.loads((out / "summary.json").read_text())["datasets"]
        with (out / "heldout" / "per_gemm.csv").open() as f:
            return ds, list(csv.DictReader(f))

    def kernels(rows, ident):
        out = {}
        for r in rows:
            for what in ("pick", "origami"):
                if r[f"{what}_sol_idx"]:
                    kid = ident.ids[int(r[f"{what}_sol_idx"])]
                    assert r[f"{what}_kernel_id"] == kid
                    assert r[f"{what}_signature"] == ki.label_of_id(kid)
            key = (r["m"], r["n"], r["k"], r["batch"])
            out[key] = (r["pick_kernel_id"], r["origami_kernel_id"], r["sel_eff"])
        return out

    ds_a, rows_a = evaluate(tmp_path / "a", trained / "heldout", tmp_path / "out_a")
    ds_b, rows_b = evaluate(tmp_path / "b", data, tmp_path / "out_b")
    assert ds_a["rc"] == ds_b["rc"] == 0
    s = ds_b["summary"]
    assert s["library_digest"] == ident_b.digest
    assert s["n_rows_kernel_not_in_library"] == s["n_out_of_library_rows"] == 0
    assert kernels(rows_b, ident_b) == kernels(rows_a, ident_a)
    assert all(int(r["pick_sol_idx"]) >= 1000 for r in rows_b if r["pick_sol_idx"])


# ── the bank ─────────────────────────────────────────────────────────────────

BANK_STEM = STEM.replace("_Dijk_", "_Dijk_CU256_")


def write_bank_library(d, table, bank):
    """A library dir whose STEM has the kernels `table` (kernel_pool
    entries) in its Prediction table and whose BANK_STEM, a file of the same
    problem type, has `bank` in an Equality row."""
    d.mkdir(parents=True, exist_ok=True)
    bf.write_logic(
        d / f"{STEM}.dat",
        library_solutions(table),
        table=[k["sol_idx_global"] for k in table],
    )
    sols = library_solutions(bank)
    rows = [
        {
            "predicate": {"type": "EqualityMatching"},
            "library": {
                "type": "Matching",
                "table": [{"key": [1, 1, 1, 1], "index": s["index"]} for s in sols],
            },
        }
    ]
    doc = {"solutions": sols, "library": {"type": "Problem", "rows": rows}}
    (d / f"{BANK_STEM}.dat").write_bytes(msgpack.packb(doc, use_bin_type=True))
    return d


def write_bench_logs(d, shapes, kernels):
    """A stage03 block log of `shapes`, each benched over every kernel of
    `kernels` in list order at test_stage05_train's latencies."""
    lines = list(bf.BANNER)
    for shape in shapes:
        g = bf.Gemm(*shape)
        lines += bf.preamble(len(kernels))
        timed = []
        for rank, k in enumerate(kernels):
            us = latency_us(shape, k)
            mt = (k["mt_m"], k["mt_n"], k["mt_k"])
            sid = k["sol_idx_global"]
            lines += bf.measured_lines(rank, g, sid, us, mt, types=bf.BF16_TYPES)
            timed.append((us, rank, sid, mt))
        us, rank, sid, mt = min(timed)
        lines += bf.winner(rank, g, sid, us, mt=mt, types=bf.BF16_TYPES)
    d.mkdir(parents=True)
    (d / f"block_0000_lines1-{len(shapes)}_gpu0.log").write_text("\n".join(lines))
    return d


@pytest.fixture(scope="module")
def bank_world(trained, tmp_path_factory):
    """Two kernels of the pool in the library's table, the others in its
    bank, a config whose planner ranks both, and bench logs of GEMMs of the
    trained cells over every kernel."""
    tmp = tmp_path_factory.mktemp("s08bank")
    pool = kernel_pool()
    table, bank = pool[:2], pool[2:]
    lib = write_bank_library(tmp / "lib", table, bank)
    shapes = shapes_for(31, 8, CELLS3[:2])
    logs = write_bench_logs(tmp / "logs", shapes, table + bank)
    cfg = {
        "config_id": "unit_bbs_tn",
        "arch": "gfx950",
        "hardware": dict(HARDWARE, pci_chip_id=0x75A0),
        "planner": {"enabled": True, "bank_library_stems": [BANK_STEM]},
    }
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))
    return {
        "tmp": tmp,
        "lib": lib,
        "logs": logs,
        "shapes": shapes,
        "bank": {k["sol_idx_global"]: k for k in bank},
        "pool": pool,
    }


def bank_stage08(trained, world, out, dataset, rc=0):
    run_stage(
        "stage08_optional_validate_selection_efficiency.py",
        "--train-dir", trained / "round_0" / "stage05", "--arch", "gfx950",
        "--config-yaml", world["tmp"] / "config.yaml", "--weight-dtype", "int8",
        "--library-dir", world["lib"], "--library-stem", STEM,
        "--out-dir", out, "--dataset", f"bank={dataset}", "--bootstrap", 0,
        "--quiet", expect_rc=rc,
    )  # fmt: skip
    (ds,) = json.loads((out / "summary.json").read_text())["datasets"]
    return ds


def test_bench_logs_keep_the_bank_when_a_bank_kernel_wins_or_is_picked(
    trained, bank_world
):
    """The pool ranks the table and the bank, so the bank's rows must reach
    the evaluation: a GEMM a bank kernel wins is scored against that kernel,
    and a bank kernel the model picks is charged its own latency instead of
    being passed over for a table kernel."""
    out = bank_world["tmp"] / "out"
    ds = bank_stage08(trained, bank_world, out, bank_world["logs"])
    assert ds["rc"] == 0
    enriched = json.loads(
        (out / "bank" / "stage04_enriched" / "enrich_summary.json").read_text()
    )
    assert enriched["extra_library_stems"] == [BANK_STEM]
    assert ds["summary"]["bank_library_stems"] == [BANK_STEM]
    assert ds["summary"]["n_out_of_library_rows"] == 0
    bank, pool = bank_world["bank"], bank_world["pool"]
    with (out / "bank" / "per_gemm.csv").open() as f:
        rows = {(int(r["m"]), int(r["n"]), int(r["k"])): r for r in csv.DictReader(f)}
    assert len(rows) == len(bank_world["shapes"])
    bank_wins = bank_picks = 0
    for shape in bank_world["shapes"]:
        r = rows[shape[:3]]
        us = {k["sol_idx_global"]: float(f"{latency_us(shape, k):g}") for k in pool}
        best = min(us, key=us.get)
        bank_wins += best in bank
        assert float(r["winner_us"]) == pytest.approx(us[best])
        assert r["served_by"] == "model"
        pick = int(r["pick_sol_idx"])
        assert float(r["pick_us"]) == pytest.approx(us[pick])
        bank_picks += pick in bank
    assert bank_wins and bank_picks


def test_rows_enriched_without_the_bank_fail_the_dataset(trained, bank_world):
    rows = bank_world["tmp"] / "table_only"
    run_stage(
        "stage04_convert_to_enriched_dataset.py",
        "--log-dir", bank_world["logs"], "--out-dir", rows,
        "--dat-dir", bank_world["lib"], "--library-stem", STEM, "--quiet",
    )  # fmt: skip
    ds = bank_stage08(trained, bank_world, bank_world["tmp"] / "out2", rows, rc=1)
    assert ds["rc"] == 1
    assert "without the rows of the bank's library file(s)" in ds["reason"]
    assert BANK_STEM in ds["reason"]
