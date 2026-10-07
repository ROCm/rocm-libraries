# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two rounds of the real pipeline on CPU: the fake hipblaslt-bench times a
fake kernel library with a synthetic cost model, the real stages train,
evaluate, deploy into a fake checkout (with a fake `cmake` doing the
co-location build) and validate."""

import csv
import json
import os
import sys
from collections import defaultdict

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage04_convert_to_enriched_dataset as s04
from lib import dat


ARCH = "gfx1250v0"

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
            "adaptive": {"enabled": True, "measure_time": 1.0},
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
        cfg["train"]["catalog"] = "v3"
        cfg["bench"]["pool_only_library"] = True
        cfg["contexts"] = {
            "list": [{"cu_budget": 0}, {"cu_budget": 16, "cotenant": True}],
            "holdout": [{"cu_budget": 32, "cotenant": True, "fraction": 0.5}],
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


def _check_contexts(run_dir):
    """Every shape is enriched exclusively and under 16 CUs, the model
    declares the trained budget, stage04b reports the held-out one."""
    from test_stage05_train import load_bundle

    rd0 = run_dir / "round_0"
    gemms = defaultdict(set)
    for path in sorted((rd0 / "stage04").glob("chunk_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                gemms[r.get("cu_budget") or "0"].add((r["m"], r["n"], r["k"]))
    assert set(gemms) == {"0", "16"} and gemms["0"] == gemms["16"]
    summary = json.loads((rd0 / "stage03c" / "contexts.json").read_text())
    assert [c["name"] for c in summary["contexts"]] == ["cu16_cot", "cu32_cot"]
    assert summary["library_path"].endswith("libdir_pool")
    bundle = load_bundle(rd0 / "stage05" / "models.pt")
    assert bundle["catalog"] == "v3"
    assert bundle["contexts"]["min_cu_fraction"] == 16 / 64
    metrics = json.loads((rd0 / "stage05" / "metrics.json").read_text())
    assert metrics["load_stats"]["gemms_non_exclusive"] == len(gemms["16"])
    held = json.loads((run_dir / "round_1" / "stage04b" / "decisions.json").read_text())
    report = held["aggregate"]["context_holdout"]["32/default"]
    assert 0 < report["n_gemms"] < len(gemms["0"]) + len(gemms["16"])
