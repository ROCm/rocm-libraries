# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two rounds of the real pipeline on CPU: the fake hipblaslt-bench times a
fake kernel library with a synthetic cost model, the real stages train,
evaluate, deploy into a fake checkout (with a fake `cmake` doing the
co-location build) and validate."""

import csv
import json
import os
import re
import sys
from collections import defaultdict

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage04_convert_to_enriched_dataset as s04
from lib import dat
from lib.arch import STRICT_SWITCH


ARCH = "gfx1250-strict"

FAKE_CMAKE = """
import shutil, sys
from pathlib import Path
build = Path(sys.argv[sys.argv.index("--build") + 1])
if sys.argv[sys.argv.index("--target") + 1] != "hipblaslt-tilewright-models":
    sys.exit(2)
weights = Path({weights!r})
for d in weights.iterdir():
    lib = build / "Tensile" / "library" / d.name
    if lib.is_dir():
        for f in (d / d.name).iterdir():
            shutil.copy(f, lib / f.name)
"""


@pytest.mark.parametrize(
    "twins, contexts",
    [(False, False), (True, False), (False, True)],
    ids=["unique", "duplicate_kernels", "contexts"],
)
def test_two_rounds_on_cpu(tmp_path, monkeypatch, twins, contexts):
    pytest.importorskip("tilewright")
    fh.unhide_devices(monkeypatch)
    kernels = fh.kernels_grid(twins=twins)
    bench_root = tmp_path / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ARCH, bf.LIBRARY_STEM, kernels)
    deploy_root = tmp_path / "deploy"
    (deploy_root / "shared" / "tilewright").mkdir(parents=True)
    deploy_build = deploy_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(deploy_build, ARCH, bf.LIBRARY_STEM, kernels)
    weights = deploy_root / "shared" / "tilewright" / "weights" / "hipblaslt"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cmake").write_text(
        f"#!{sys.executable}\n" + FAKE_CMAKE.format(weights=str(weights))
    )
    (bindir / "cmake").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(build / "Tensile" / "library" / ARCH))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    n_cu, lds, l2 = (int(v) for v in fh.FAKE_HW.split(","))
    cfg = {
        "config_id": "e2e",
        "arch": ARCH,
        "hipblaslt": {
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
            "library_stem": bf.LIBRARY_STEM,
        },
        "hardware": {"n_cu": n_cu, "lds_bytes": lds, "l2_bytes": l2},
        "runtime": {"devices": [0], "rocm_libraries_root": str(bench_root)},
        "seed": {
            "target_per_cell": 12,
            "shapes_per_cell": 12,
            "seed": 3,
            "n_strata": 1,
            "exclude": [
                {"max_m": 512},
                {"max_n": 512},
                {"max_k": 512},
                {"min_batch": 2},
            ],
        },
        "bench": {
            "blocks_per_gpu": 1,
            "startup_grace_s": 120,
            "stall_timeout_s": 120,
            "device_memory_gib": 64,
            "skip_slow_solution_ratio": 0.5,
            "adaptive": {"measure_time": 1.0},
        },
        "train": {"epochs": 3, "workers": 1, "min_cell_gemms": 6, "smart_k": 4},
        "validate": {
            "min_cell_gemms": 3,
            "max_split_depth": 0,
            "sel_eff_threshold": 0.999,
        },
        "full": {"total_rounds": 2},
        "deploy": {
            "weight_dtype": "int8",
            "rocm_libraries_to_deploy": str(deploy_root),
            "targets": [ARCH],
            "apply_patch": True,
            "build_after_apply": True,
        },
        "stage07": {
            "parity_sample_trained": 6,
            "timing_request_sizes": [1],
            "timing_repetitions": 2,
        },
    }
    if contexts:
        cfg["hardware"]["n_cu"] = CONTEXT_CUS
        monkeypatch.setenv("FAKE_HW", f"{CONTEXT_CUS},{lds},{l2}")
        cfg["train"]["catalog"] = "v3"
        cfg["bench"]["pool_only_library"] = True
        cfg["runtime"]["env"] = {STRICT_SWITCH: "0"}
        cfg["contexts"] = {
            "list": [{"cu_budget": 0}, {"cu_fraction": 0.5, "cotenant": True}],
            "holdout": [
                {"cu_fraction": 0.25, "cotenant": True, "fraction": 0.5},
                {"cu_fraction": 0.6, "cotenant": True, "fraction": 0.5},
            ],
        }
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and warnings == []
    cfg_path = tmp_path / "e2e.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "run"
    rc = ro.main(
        ["--config", str(cfg_path), "--mode", "full", "--run-dir", str(run_dir)]
    )
    assert rc == 0

    for r in (0, 1):
        assert (run_dir / f"round_{r}" / ro.ROUND_MARKER).is_file()
    assert (run_dir / "round_1" / "stage04b" / "decisions.json").is_file()
    listed = (run_dir / "round_1" / "stage04b" / "retrain_cells.txt").read_text()
    s05 = json.loads((run_dir / "round_1" / "stage05" / ro.STAGE_MARKER).read_text())
    assert s05["cmd"][s05["cmd"].index("--only-cells") + 1] == listed
    rec = json.loads(
        (run_dir / "round_1" / "stage06" / "deploy_record.json").read_text()
    )
    assert rec["applied"] and rec["built"] and rec["deployed"]
    gate = rec["gate"]
    assert gate["n_evaluated"] == gate["n_gemms"] > 0 and gate["passed"]
    assert gate["csv_dirs"] == [str(run_dir / "round_1" / "stage04b" / "gate")]
    breakdown = json.loads((run_dir / "held_out_sel_eff_breakdown.json").read_text())
    assert breakdown["source"] == "deploy_gate"
    assert breakdown["held_out_sel_eff"] == pytest.approx(gate["sel_eff"])
    t = rec["targets"][0]
    index = (weights / ARCH / ARCH / "tilewright_index").read_text()
    assert f"{bf.LIBRARY_STEM}\t{t['weights_file']}" in index
    colocated = deploy_build / "Tensile" / "library" / ARCH / t["weights_file"]
    assert (
        colocated.read_bytes()
        == (weights / ARCH / ARCH / t["weights_file"]).read_bytes()
    )

    s7 = json.loads((run_dir / "validate" / "stage07" / "summary.json").read_text())
    assert s7["ok"], s7["failures"]
    parity = next(iter(s7["per_yaml"].values()))["parity"]
    assert parity["match_rate"] == 1.0 and parity["diag_ok"]
    assert parity["library_pool_size"] == len(kernels)
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "held_out_sel_eff_breakdown.json").is_file()

    if contexts:
        _check_contexts(run_dir)
        return

    enriched = run_dir / "round_0" / "stage04"
    listed = dat.read_kernel_attributes(enriched / dat.KERNELS_FILE)
    assert set(listed) == {k["index"] for k in kernels}
    for k in kernels:
        info = dat.kernel_dat_info(k)
        assert listed[k["index"]] == info["attributes"]
    rows = defaultdict(dict)
    for path in sorted(enriched.glob("chunk_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                rows[(r["m"], r["n"], r["k"], r["batch_count"])][
                    int(r["sol_idx_global"])
                ] = r
    assert all(len(by_sid) == len(kernels) for by_sid in rows.values())
    if twins:
        n = len(kernels) // 2
        for by_sid in rows.values():
            for i in range(n):
                base, twin = by_sid[i], by_sid[i + n]
                assert [base[f] for f in s04.CONFIG_FIELDS] == [
                    twin[f] for f in s04.CONFIG_FIELDS
                ]
                assert float(twin["us"]) > float(base["us"])


# Made up, like every device value of these fake runs.
CONTEXT_CUS = 128


def _check_contexts(run_dir):
    """The run resolves the contexts' fractions for the device's
    CONTEXT_CUS: every shape is enriched exclusively and under the trained
    half of the CUs, the model declares that budget, stage04b reports the
    held-out quarter, and the held-out entry that lands on the trained
    budget is left out, which the snapshot and the report say. Every bench
    gets runtime.env, and stage04 checks the benched problem type."""
    from test_stage05_train import load_bundle

    rd0 = run_dir / "round_0"
    gemms = defaultdict(set)
    for path in sorted((rd0 / "stage04").glob("chunk_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                gemms[r.get("cu_budget") or "0"].add((r["m"], r["n"], r["k"]))
    assert set(gemms) == {"0", "64"} and gemms["0"] == gemms["64"]
    summary = json.loads((rd0 / "stage03c" / "contexts.json").read_text())
    assert [c["name"] for c in summary["contexts"]] == ["cu64_cot", "cu32_cot"]
    assert summary["library_path"].endswith("libdir_pool")
    bundle = load_bundle(rd0 / "stage05" / "models.pt")
    assert bundle["catalog"] == "v3"
    assert bundle["contexts"]["min_cu_fraction"] == 64 / CONTEXT_CUS
    metrics = json.loads((rd0 / "stage05" / "metrics.json").read_text())
    assert metrics["load_stats"]["gemms_non_exclusive"] == len(gemms["64"])
    held = json.loads((run_dir / "round_1" / "stage04b" / "decisions.json").read_text())
    report = held["aggregate"]["context_holdout"]["32/default"]
    assert 0 < report["n_gemms"] < len(gemms["0"]) + len(gemms["64"])

    (snap,) = [
        p
        for p in (run_dir / "configs").glob("*.yaml")
        if not p.name.endswith(".source.yaml")
    ]
    text = snap.read_text()
    resolved = yaml.safe_load(text)["contexts"]
    assert [(m.get("cu_fraction"), m["cu_budget"]) for m in resolved["list"]] == [
        (None, 0),
        (0.5, 64),
    ]
    assert [(m["cu_fraction"], m["cu_budget"]) for m in resolved["holdout"]] == [
        (0.25, 32)
    ]
    head = " ".join(ln[2:] for ln in text.splitlines() if ln.startswith("# "))
    assert (
        "contexts.holdout[1]: cu_fraction 0.6 is 64 of 128 CUs, the budget of "
        "contexts.list[1]; left out" in head
    )
    for owner in (
        rd0 / "stage03",
        rd0 / "stage03c" / "cu64_cot" / "stage03",
        rd0 / "stage03c" / "cu32_cot" / "stage03",
        run_dir / "validate" / "stage07",
    ):
        platform = json.loads((owner / "platform.json").read_text())
        assert platform["environment"] == {STRICT_SWITCH: "0"}, owner
    assert (
        "problem type   : {'a_type': 'f8_r', 'b_type': 'f8_r', 'c_type': 'bf16_r'"
        in (rd0 / "stage04.log").read_text()
    )
    page = (run_dir / "report.html").read_text()
    assert "contexts.holdout[1] (cu_fraction 0.6)" in page
    assert "cu64 (cu64_cot)" in page and "cu32 (cu32_cot held out)" in page


BANK_STEM = bf.LIBRARY_STEM.replace(f"_{ARCH}", f"_Bank_{ARCH}")


def test_one_command_runs_and_validates_a_bank_model(tmp_path, monkeypatch):
    """A v3 run with a planner and a bank of kernels, validated by stage 7
    through a library directory of the bench library served by another
    build, and by stage 8; with the Origami reference on, stage 7 pairs the
    runtime's picks with and without tilewright on a sample of the audit set,
    which fills the report's held-out GEMMs of the training distribution
    although no planned bench timed Origami's pick; the same command again
    runs nothing, and a stage-7 build that does not serve the bank stops it
    before any stage."""
    pytest.importorskip("tilewright")
    fh.unhide_devices(monkeypatch)
    kernels = fh.kernels_grid()
    bank = [
        bf.solution(100 + i, 100 + i, mt=mt, mi=mi)
        for i, (mt, mi) in enumerate(
            (mt, mi)
            for mt in ((64, 128, 64), (128, 64, 64))
            for mi in ((16, 16, 128, 1), (32, 32, 64, 1))
        )
    ]
    bench_root = tmp_path / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ARCH, bf.LIBRARY_STEM, kernels)
    lib = build / "Tensile" / "library" / ARCH
    fh.write_bank_library(lib, BANK_STEM, bank)
    (lib / "tilewright_index").write_text(f"{bf.OTHER_STEM}\tother.tilewright.bin\n")
    served = fh.write_fake_build(tmp_path / "v3build", ARCH, bf.LIBRARY_STEM, kernels)
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(lib))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    monkeypatch.setenv("FAKE_BANK_STEMS", BANK_STEM)
    dataset = tmp_path / "val.yaml"
    dataset.write_text(
        "".join(
            bf.bench_line(bf.Gemm(m, n, k), compute_type="c_f32_r")
            for m, n, k in ((1024, 2048, 4096), (4096, 4096, 1024), (2048, 1024, 768))
        )
    )
    n_cu, lds, l2 = (int(v) for v in fh.FAKE_HW.split(","))
    cfg = {
        "config_id": "onecmd",
        "arch": ARCH,
        "hipblaslt": {
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
            "library_stem": bf.LIBRARY_STEM,
        },
        "hardware": {"n_cu": n_cu, "lds_bytes": lds, "l2_bytes": l2},
        "runtime": {"devices": [0], "rocm_libraries_root": str(bench_root)},
        "seed": {
            "target_per_cell": 12,
            "shapes_per_cell": 12,
            "seed": 3,
            "exclude": [
                {"max_m": 512},
                {"max_n": 512},
                {"max_k": 512},
                {"min_batch": 2},
            ],
        },
        "bench": {
            "blocks_per_gpu": 1,
            "startup_grace_s": 120,
            "stall_timeout_s": 120,
            "device_memory_gib": 64,
            "skip_slow_solution_ratio": 0.5,
            "adaptive": {"measure_time": 1.0},
        },
        "planner": {
            "enabled": True,
            "mode": "planned",
            "seed_per_cell": 2,
            "neighbors_k": 2,
            "bank_library_stems": [BANK_STEM],
            "top_k": 4,
            "families_top_f": 2,
            "explore_random_k": 2,
            "min_coverage": 1,
        },
        "train": {
            "epochs": 3,
            "workers": 1,
            "min_cell_gemms": 6,
            "min_base_cell_gemms": 4,
            "catalog": "v3",
            "whitelist": "none",
            "cascade_k": 4,
            "served_candidates": True,
            "smart_k": 0,
        },
        "validate": {
            "min_cell_gemms": 3,
            "max_split_depth": 0,
            "sel_eff_threshold": 0.999,
        },
        "full": {"total_rounds": 2},
        "deploy": {"weight_dtype": "int8", "targets": [ARCH]},
        "stage07": {
            "build_dir": str(served),
            "parity_sample_trained": 6,
            "parity_sample_audit": 6,
            "timing_request_sizes": [1],
            "timing_repetitions": 2,
        },
        "stage08": {"datasets": [{"name": "val", "path": str(dataset)}]},
        "audit": {
            "dir": str(tmp_path / "audit"),
            "shapes_per_cell": 2,
            "shapes_per_context": 1,
            "seed": 11,
            # the fewest the per-cell gate's cell tests take at gate.alpha 0.05
            "bootstrap": 40,
        },
        "report": {"origami_reference": True},
    }
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and warnings == []
    cfg_path = tmp_path / "onecmd.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    audit = ["--config", str(cfg_path), "--mode", "audit"]
    assert ro.main(audit + ["--run-dir", str(tmp_path / "audit")]) == 0
    run_dir = tmp_path / "run"
    argv = ["--config", str(cfg_path), "--mode", "full", "--run-dir", str(run_dir)]
    assert ro.main(argv) == 0

    enriched = dat.read_kernel_attributes(
        run_dir / "round_0" / "stage04" / dat.KERNELS_FILE
    )
    assert {s["index"] for s in bank} <= set(enriched)
    libdir = run_dir / "validate" / "stage07_lib" / ARCH
    rec = json.loads(
        (run_dir / "round_1" / "stage06" / "deploy_record.json").read_text()
    )
    t = rec["targets"][0]
    assert (libdir / "tilewright_index").read_text() == (
        f"{bf.OTHER_STEM}\tother.tilewright.bin\n{bf.LIBRARY_STEM}\t{t['weights_file']}\n"
    )
    assert not list((build / "Tensile" / "library" / ARCH).glob("*.tilewright.bin"))
    s7 = json.loads((run_dir / "validate" / "stage07" / "summary.json").read_text())
    assert s7["ok"], s7["failures"]
    assert s7["model"] == str(libdir / t["weights_file"])
    assert s7["bank_served"] and s7["origami_reference"]
    pool = len(kernels) + len(bank)
    for y in s7["per_yaml"].values():
        assert y["parity"]["runtime_pool_sizes"] == [pool]
        assert y["parity"]["match_rate"] == 1.0
    held = s7["per_yaml"]["stage07_audit_6"]
    assert held["held_out"] and not s7["per_yaml"]["stage07_trained_6"]["held_out"]
    paired = held["origami_reference"]["paired"]
    assert paired["n"] > 0 and 0 < paired["origami_sel_eff"] <= 1
    decisions = json.loads(
        (run_dir / "round_1" / "stage04b" / "decisions.json").read_text()
    )
    assert not any(c.get("origami_n_eval") for c in decisions["per_cell"])
    page = (run_dir / "report.html").read_text()
    row = re.search(
        r"held-out GEMMs of the training distribution: the audit set's "
        r"stage07_audit_6 \(stage 7, hipBLASLt's picks\)</td><td>(\d+)</td>"
        r"<td>([\d.]+)</td><td>([\d.]+)</td>",
        page,
    )
    assert row and int(row.group(1)) == paired["n"]
    assert float(row.group(3)) == pytest.approx(
        paired["origami_sel_eff"] * 100, abs=0.01
    )
    status = json.loads((run_dir / ro.RUN_STATUS).read_text())
    assert status["complete"] and status["invocation"]["exit_status"] == 0
    assert {k: v["status"] for k, v in status["validation"].items()} == {
        "stage07": "passed",
        "stage08": "passed",
    }

    assert ro.main(argv) == 0
    timing = sorted((run_dir / "timing").glob("*_full*.json"))[-1]
    assert json.loads(timing.read_text())["rows"] == []

    host = served / "tensilelite" / "libtensilelite-host.so.1.0"
    host.write_bytes(b"MLREC_v2 MLREC_v3")
    assert ro.main(argv) == 2
    assert (
        json.loads((run_dir / ro.RUN_STATUS).read_text())["invocation"]["exit_status"]
        == 2
    )
