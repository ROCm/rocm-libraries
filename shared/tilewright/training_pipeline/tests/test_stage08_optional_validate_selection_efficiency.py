# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import msgpack
import pytest
import yaml

from test_stage05_train import (
    CELLS3,
    PIPELINE_DIR,
    kernel_pool,
    run_stage,
    shapes_for,
    train,
    write_config,
    write_round,
)

pytest.importorskip("tilewright")

import stage08_optional_validate_selection_efficiency as s08  # noqa: E402

STEM = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"


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
    got = s08.dataset_library_dirs(["a=/b/build", "c:/d"], "gfx1250v0:xnack+")
    assert got == {
        "a": Path("/b/build/Tensile/library/gfx1250v0"),
        "c": Path("/d/Tensile/library/gfx1250v0"),
    }
    with pytest.raises(ValueError):
        s08.dataset_library_dirs(["nodelimiter"], "gfx950")


@pytest.mark.parametrize("adaptive, probed", [(False, True), (True, False)])
def test_yaml_dataset_probes_only_without_adaptive_timing(
    tmp_path, monkeypatch, adaptive, probed
):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "probe": {"skip_ratio_mode": "sigmoid"},
                "bench": {"adaptive": {"enabled": adaptive}},
            }
        )
    )
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
    scripts = [Path(cmd[2]).name for cmd in calls]
    assert ("stage02_probe.py" in scripts) == probed
    assert scripts[-1] == "stage03_load_balance_offline_tuning.py"
