# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage05 on synthetic enriched data (also used by the stage04b / stage08
tests): a kernel pool, a deterministic latency model, and chunk_*.csv files
with the stage04 column schema."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bench_fakes as bf
import stage05_train as s5
from lib import dat, feasibility, grid, hipblaslt_launch
from lib import features as fs
from lib import hardware as hwlib

PIPELINE_DIR = Path(__file__).resolve().parent.parent
HARDWARE = {"n_cu": 100, "lds_bytes": 65536, "l2_bytes": 4194304}

COLUMNS = (
    "is_winner,is_skip,is_origami_pick,rank,sol_idx_global,sol_idx_local,transA,"
    "transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,"
    "ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,"
    "scaleB,rotating_buffer,flush,use_gpu_timer,hipblaslt-Gflops,hipblaslt-GB/s,"
    "us,samples,cv,rel_iqr,status,mt_m,mt_n,mt_k,mi_m,mi_n,mi_k,occupancy,"
    "cache_hints_a,cache_hints_b,grvw_a,grvw_b,gwvw_d"
).split(",")
CONFIG = (
    "mt_m mt_n mt_k mi_m mi_n mi_k occupancy cache_hints_a cache_hints_b "
    "grvw_a grvw_b gwvw_d"
).split()

CELL_BOXES = {
    "Large|Large|LargeK|Bnone": ((600, 6000), (600, 6000), (600, 6000), (1, 1)),
    "Mid|Large|MidK|Bnone": ((130, 512), (600, 8000), (64, 512), (1, 1)),
    "Small|Small|MidK|Bnone": ((33, 128), (33, 128), (64, 512), (1, 1)),
    "Tiny|Mid|MidK|Bany": ((8, 32), (130, 512), (64, 512), (2, 8)),
}


def kernel_pool():
    out = []
    tiles = [(32, 32), (64, 64), (64, 128), (128, 64), (128, 128), (256, 128)]
    for mt_m, mt_n in tiles + [(128, 256), (256, 256)]:
        for mt_k in (64, 128):
            mi = (16, 16, 32) if mt_m <= 64 else (32, 32, 16)
            out.append((mt_m, mt_n, mt_k, *mi, 1, 0, 0, 8, 8, 4))
    out += [(32, 128, 64, 16, 16, 32, 2, 0, 4, 8, 8, 4)]
    out += [(64, 256, 64, 16, 16, 32, 2, 0, 4, 8, 8, 4)]
    return [dict(zip(CONFIG, k), sol_idx_global=100 + i) for i, k in enumerate(out)]


def _u(*key):
    h = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64)


def latency_us(shape, kern):
    m, n, k, b = shape
    tiles = math.ceil(m / kern["mt_m"]) * math.ceil(n / kern["mt_n"]) * b
    k_iters = math.ceil(k / kern["mt_k"])
    area = kern["mt_m"] * kern["mt_n"]
    tile_t = (
        area * k_iters * kern["mt_k"] / ((0.3 + 0.7 * min(1.0, area / 32768)) * 4e5)
    )
    nt = 0.85 if kern["cache_hints_b"] == 4 and m <= 2 * kern["mt_m"] else 1.0
    noise = 1.0 + 0.04 * (_u("noise", shape, kern["sol_idx_global"]) - 0.5)
    return (2.0 + math.ceil(tiles / 100) * tile_t * nt) * noise


def gemm_rows(shape, pool):
    m, n, k, b = shape
    lat = {kk["sol_idx_global"]: latency_us(shape, kk) for kk in pool}
    order = sorted(
        pool,
        key=lambda kk: lat[kk["sol_idx_global"]]
        * (0.7 + 0.6 * _u("origami", shape, kk["sol_idx_global"])),
    )
    rows = []
    for rank, kk in enumerate(order):
        skip = rank > 0 and _u("skip", shape, kk["sol_idx_global"]) < 0.15
        row = dict.fromkeys(COLUMNS, "")
        row.update(
            is_winner="False",
            is_skip="True" if skip else "False",
            is_origami_pick=1 if rank == 0 else 0,
            rank=rank,
            sol_idx_global=kk["sol_idx_global"],
            sol_idx_local=kk["sol_idx_global"] - 100,
            transA="T",
            transB="N",
            batch_count=b,
            m=m,
            n=n,
            k=k,
            a_type="bf16_r",
            b_type="bf16_r",
            c_type="bf16_r",
            d_type="bf16_r",
            compute_type="f32_r",
            us=f"{lat[kk['sol_idx_global']] * (1.3 if skip else 1.0):.4f}",
        )
        row.update({f: kk[f] for f in CONFIG})
        rows.append(row)
    tested = [r for r in rows if r["is_skip"] == "False"]
    min(tested, key=lambda r: float(r["us"]))["is_winner"] = "True"
    return rows


def shapes_for(seed, per_cell, cells=None):
    rng = random.Random(seed)
    out = []
    for label in cells or CELL_BOXES:
        (m0, m1), (n0, n1), (k0, k1), (b0, b1) = CELL_BOXES[label]
        for _ in range(per_cell):
            k = max(32, rng.randint(k0, k1) // 32 * 32)
            out.append(
                (rng.randint(m0, m1), rng.randint(n0, n1), k, rng.randint(b0, b1))
            )
    return out


def write_round(out_dir, shapes, pool=None, rows_per_chunk=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for s in shapes for r in gemm_rows(s, pool or kernel_pool())]
    for i in range(0, len(rows), rows_per_chunk):
        path = out_dir / f"chunk_{i // rows_per_chunk:04d}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows[i : i + rows_per_chunk])
    return out_dir


def write_config(path):
    path.write_text(
        "config_id: unit_bbs_tn\narch: gfx950\nhardware:\n"
        + "".join(f"  {k}: {v}\n" for k, v in HARDWARE.items())
    )
    return path


def have_engine():
    return importlib.util.find_spec("tilewright") is not None


def run_stage(script, *args, env=None, expect_rc=0):
    proc = subprocess.run(
        [sys.executable, str(PIPELINE_DIR / "stages" / script), *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", **(env or {})},
        timeout=900,
    )
    assert proc.returncode == expect_rc, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc


def train_args(tmp, data_dirs, out_dir, *extra):
    args = []
    for d in data_dirs:
        args += ["--enriched-csv-dir", d]
    args += [
        "--output-dir",
        out_dir,
        "--arch",
        "gfx950",
        "--config-yaml",
        write_config(tmp / "config.yaml"),
        "--epochs",
        6,
        "--n-estimators",
        8,
        "--min-cell-gemms",
        12,
        "--smart-k",
        3,
        "--seed",
        5,
        "--quiet",
        *extra,
    ]
    if not have_engine() and "--skip-deployed-eval" not in args:
        args.append("--skip-deployed-eval")
    return [str(a) for a in args]


def train(tmp, data_dirs, out_dir, *extra, expect_rc=0):
    return run_stage(
        "stage05_train.py",
        *train_args(tmp, data_dirs, out_dir, *extra),
        expect_rc=expect_rc,
    )


def load_bundle(path):
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=True)


CELLS3 = ["Large|Large|LargeK|Bnone", "Mid|Large|MidK|Bnone", "Small|Small|MidK|Bnone"]


def test_end_to_end(tmp_path):
    data = write_round(tmp_path / "round_0" / "stage04", shapes_for(0, 24, CELLS3))
    out = tmp_path / "round_0" / "stage05"
    train(tmp_path, [data], out, "--weight-dtype", "int4")
    bundle = load_bundle(out / "models.pt")
    assert sorted(bundle["models"]) == sorted(CELLS3)
    assert bundle["q_names"] == fs.query_feature_names()
    nta = fs.item_feature_names().index("nta_norm")
    for entry in bundle["models"].values():
        assert entry["smart_k_signatures"] and len(entry["smart_k_signatures"][0]) == 8
        assert "q_proj.0.weight" in entry["state_dict"]
        assert entry["state_dict"]["temperature"].shape == (1,)
        assert "log_temperature" not in entry["state_dict"]
        assert entry["feature_norms"]["i_std"][nta] == 0.0

    cells = {
        c["label"]: c for c in json.loads((out / "cells.json").read_text())["cells"]
    }
    for label, c in cells.items():
        assert c["seed"] == s5.cell_seed(5, label)
        assert c["n_val_gemms"] > 0 and c["selection_set"] == "validation"
        assert c["n_train_gemms"] + c["n_val_gemms"] == c["n_gemms"] == 24
        assert c["origami"]["n_eval"] == c["n_train_gemms"]
        assert 1 <= c["best_step"] <= c["steps"] and c["best_phase"] == "main"
        assert c["selection_metric"] == (
            "deployed_validation" if have_engine() else "validation"
        )
        if have_engine():
            # The kept checkpoint was measured exactly as it is deployed.
            assert c["selection_deployed_val_sel_eff"] == c["deployed_val"]["sel_eff"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["environment"]["artifact_knobs"]) == set(s5.ARTIFACT_ENV_KNOBS)
    assert metrics["hardware"] == {**HARDWARE, "num_xcd": None}
    assert metrics["weight_dtype"] == "int4"
    assert metrics["arch_constants"]["parallel_mi_cu"] == 4.0
    with (out / "training_log.csv").open() as f:
        header = next(csv.reader(f))
    assert "val_sel_eff" in header and "n_val_gemms" in header

    if not have_engine():
        return
    import tilewright as tw

    from lib import evaluate as ev
    from lib import hardware as hwlib
    from lib import mlrec

    for c in cells.values():
        assert c["deployed"]["n_eval"] == c["n_train_gemms"]
        assert c["deployed_val"]["n_eval"] == c["n_val_gemms"]
    assert metrics["validation"]["n_gemms"] == sum(
        c["n_val_gemms"] for c in cells.values()
    )
    data_v2 = mlrec.write_model(
        bundle, None, "gfx950", hwlib.arch_constants("gfx950"), "int4"
    )
    model = tw.load_model_from_memory(data_v2)
    assert tw.describe(model).n_cells == 3
    gemms = [
        g for gs in s5.load_enriched_chunks(str(data), True)[0].values() for g in gs
    ]
    result = ev.evaluate_model(
        data_v2, gemms, hardware=hwlib.DeviceHardware(**HARDWARE)
    )
    assert result.summary()["n_model_served"] > 0


def test_the_model_options_are_recorded_as_the_command_resolves_them(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(4, 14, CELLS3[:1]))
    out = tmp_path / "stage05"
    flagless = train_args(tmp_path, [data], out, "--boundary-margin", "0.5")
    args = flagless + ["--repeat-aggregation", "median"]
    run_stage("stage05_train.py", *args)
    record = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
    assert record == s5.command_model_options(args)
    assert record["repeat_aggregation"] == "median"
    assert s5.command_model_options(flagless)["repeat_aggregation"] == "min"
    assert (record["catalog"], record["cascade_k"], record["whitelist"]) == (
        "v2",
        0,
        "signature",
    )
    assert record["tie_tolerance"] == s5.DEFAULT_TIE_TOLERANCE
    assert record["attribute_spec"] == [] and record["boundary_margin"] == 0.5
    every = set(vars(s5.arg_parser().parse_args(args)))
    assert set(record) - set(s5.PREDATING_RECORDS) | s5.NOT_MODEL_SHAPING == every
    assert record["identity_rule"] == "tdm_th_v1"
    assert not s5.NOT_MODEL_SHAPING & set(record)


def test_seed_and_validation_split():
    assert s5.cell_seed(1, "a") == s5.cell_seed(1, "a") != s5.cell_seed(1, "b")
    assert s5.cell_seed(1, "a") != s5.cell_seed(2, "a")
    gemms = [
        dict(
            m=m,
            n=64,
            k=64,
            batch_count=1,
            transA="T",
            transB="N",
            a_type="bf16_r",
            b_type="bf16_r",
            c_type="bf16_r",
            d_type="bf16_r",
            compute_type="f32_r",
        )
        for m in range(1, 401)
    ]
    train_g, val_g = s5.split_validation(gemms, 0.25, 9)
    assert 50 < len(val_g) < 150 and len(train_g) + len(val_g) == 400
    more_train, more_val = s5.split_validation(gemms[:200], 0.25, 9)
    assert {id(g) for g in more_val} == {id(g) for g in val_g if g["m"] <= 200}
    assert s5.split_validation(gemms, 0.0, 9) == (gemms, [])
    assert s5.split_validation(gemms[:1], 0.99, 9)[1] == []


def test_label_seeds_make_cells_independent_of_the_run(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(1, 20, CELLS3))
    full, alone, par = tmp_path / "full", tmp_path / "alone", tmp_path / "par"
    train(tmp_path, [data], full, "--skip-deployed-eval")
    train(tmp_path, [data], alone, "--skip-deployed-eval", "--only-cells", CELLS3[2])
    proc = train(
        tmp_path,
        [data],
        par,
        "--skip-deployed-eval",
        "--train-workers",
        2,
        "--train-threads-per-worker",
        1,
    )
    assert "retrying serially" not in proc.stdout + proc.stderr
    a = load_bundle(full / "models.pt")["models"]
    b = load_bundle(alone / "models.pt")["models"]
    c = load_bundle(par / "models.pt")["models"]
    assert list(b) == [CELLS3[2]]
    for other in (b, c):
        for label, entry in other.items():
            for name, t in entry["state_dict"].items():
                assert t.equal(a[label]["state_dict"][name]), (label, name)
    serial_log = (full / "training_log.csv").read_text()
    cells = json.loads((full / "cells.json").read_text())["cells"]
    # Short schedules measure every step.
    assert len(serial_log.splitlines()) == 1 + sum(c["steps"] for c in cells)
    assert (par / "training_log.csv").read_text() == serial_log


def test_retrain_routes_split_children_and_carries_the_rest(tmp_path):
    r0, r1 = tmp_path / "round_0", tmp_path / "round_1"
    d0 = write_round(r0 / "stage04", shapes_for(2, 30, CELLS3[:2]))
    train(tmp_path, [d0], r0 / "stage05", "--skip-deployed-eval")
    from lib import subcells as sc

    rule = sc.build_split_rule(CELLS3[0], "M", 2500)
    sc.write_splits_json(r1 / "stage04b" / "splits.json", 1, [rule])
    d1 = write_round(r1 / "stage04", shapes_for(3, 30, CELLS3[:1]))
    train(
        tmp_path,
        [d0, d1],
        r1 / "stage05",
        "--skip-deployed-eval",
        "--prior-round-dir",
        r0,
        "--current-round-dir",
        r1,
        "--only-cells",
        f"{rule.lo_label},{rule.hi_label}",
    )
    bundle = load_bundle(r1 / "stage05" / "models.pt")
    assert sorted(bundle["models"]) == sorted([rule.lo_label, rule.hi_label, CELLS3[1]])
    assert bundle["n_models_carried_forward"] == 1
    assert CELLS3[0] not in bundle["models"] and bundle["fallback_parents"] == []
    cells_json = json.loads((r1 / "stage05" / "cells.json").read_text())
    assert cells_json["model_labels"] == sorted(bundle["models"])


def test_a_split_child_too_small_to_train_is_served_by_the_parent(tmp_path):
    from lib import subcells as sc

    r0, r1 = tmp_path / "round_0", tmp_path / "round_1"
    d0 = write_round(r0 / "stage04", shapes_for(2, 30, CELLS3[:2]))
    train(tmp_path, [d0], r0 / "stage05", "--skip-deployed-eval")
    rule = sc.build_split_rule(CELLS3[0], "M", 5800)
    sc.write_splits_json(r1 / "stage04b" / "splits.json", 1, [rule])
    d1 = write_round(r1 / "stage04", shapes_for(3, 30, CELLS3[:1]))
    proc = train(
        tmp_path,
        [d0, d1],
        r1 / "stage05",
        "--skip-deployed-eval",
        "--prior-round-dir",
        r0,
        "--current-round-dir",
        r1,
        "--only-cells",
        f"{rule.lo_label},{rule.hi_label}",
        expect_rc=s5.EXIT_CELLS_NOT_TRAINED,
    )
    assert f"{rule.hi_label}: requested but not trained" in proc.stderr
    out = r1 / "stage05"
    bundle = load_bundle(out / "models.pt")
    models = bundle["models"]
    assert sorted(models) == sorted([CELLS3[0], rule.lo_label, CELLS3[1]])
    assert bundle["fallback_parents"] == [CELLS3[0]]
    metrics = json.loads((out / "metrics.json").read_text())
    assert [r["cell"] for r in metrics["cells_not_trained"]] == [rule.hi_label]
    assert metrics["cells_not_trained"][0]["reason"].startswith("too few gemms")
    assert metrics["cells_failed"] == [] and metrics["fallback_parents"] == [CELLS3[0]]
    cells_json = json.loads((out / "cells.json").read_text())
    assert cells_json["model_labels"] == sorted(models)
    assert [c["label"] for c in cells_json["cells"]] == [rule.lo_label]

    tree = sc.split_tree_from_labels(models)
    assert sc.route(5900, 1000, 1000, 1, tree, models) == (rule.hi_label, CELLS3[0])
    assert sc.route(5000, 1000, 1000, 1, tree, models)[1] == rule.lo_label
    if not have_engine():
        return
    import tilewright as tw

    from lib import evaluate as ev
    from lib import hardware as hwlib
    from lib import mlrec

    data = mlrec.write_model(
        bundle, None, "gfx950", hwlib.arch_constants("gfx950"), "bf16"
    )
    model = tw.load_model_from_memory(data)
    for m, served in ((5900, CELLS3[0]), (5000, rule.lo_label)):
        prob = ev.make_problem(
            tw,
            fs.problem_kwargs_from_row(
                dict(
                    m=m,
                    n=1000,
                    k=1000,
                    batch_count=1,
                    transA="T",
                    transB="N",
                    a_type="bf16_r",
                    b_type="bf16_r",
                    c_type="bf16_r",
                    d_type="bf16_r",
                    compute_type="f32_r",
                )
            ),
        )
        assert ev.model_cell_label(tw, model, prob) == served


def test_carry_forward_keeps_the_nearest_parent_only():
    from lib import subcells as sc

    a = sc.build_split_rule(CELLS3[0], "M", 3000)
    b = sc.build_split_rule(a.hi_label, "N", 2000)
    tree = {a.cell: a, b.cell: b}
    prior = {CELLS3[0]: "p", a.hi_label: "hi", CELLS3[1]: "other"}
    carried, parents = s5.carry_forward(
        prior, {a.lo_label: "new", b.lo_label: "new"}, tree
    )
    assert carried == {a.hi_label: "hi", CELLS3[1]: "other"} and parents == [a.hi_label]
    full = {a.lo_label: "n", b.lo_label: "n", b.hi_label: "n"}
    assert s5.carry_forward(prior, full, tree) == ({CELLS3[1]: "other"}, [])
    carried, parents = s5.carry_forward({CELLS3[0]: "p"}, {b.lo_label: "n"}, tree)
    assert carried == {CELLS3[0]: "p"} and parents == [CELLS3[0]]


def _main(monkeypatch, tmp, data_dirs, out_dir, *extra):
    argv = ["stage05_train.py", *train_args(tmp, data_dirs, out_dir, *extra)]
    monkeypatch.setattr(sys, "argv", argv)
    return s5.main()


def test_a_failed_cell_fails_the_stage_after_writing_outputs(tmp_path, monkeypatch):
    data = write_round(tmp_path / "stage04", shapes_for(7, 16, CELLS3[1:]))
    real = s5.train_cell

    def flaky(**kw):
        if kw["cell"] == CELLS3[1]:
            raise RuntimeError("simulated training failure")
        return real(**kw)

    monkeypatch.setattr(s5, "train_cell", flaky)
    out = tmp_path / "out"
    rc = _main(monkeypatch, tmp_path, [data], out, "--skip-deployed-eval")
    assert rc == s5.EXIT_CELLS_NOT_TRAINED
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["cells_failed"] == [CELLS3[1]]
    assert "simulated training failure" in metrics["failures"][CELLS3[1]]
    assert list(load_bundle(out / "models.pt")["models"]) == [CELLS3[2]]
    with (out / "training_log.csv").open() as f:
        assert {row["cell"] for row in csv.DictReader(f)} == {CELLS3[2]}


def test_a_routing_mismatch_fails_the_cell(tmp_path, monkeypatch):
    if not have_engine():
        return
    from lib import evaluate as ev

    data = write_round(tmp_path / "stage04", shapes_for(8, 16, CELLS3[2:]))
    real = ev.evaluate_model

    def skewed(*a, **kw):
        out = real(*a, **kw)
        out.n_routing_mismatch = 1
        return out

    monkeypatch.setattr(ev, "evaluate_model", skewed)
    out = tmp_path / "out"
    assert _main(monkeypatch, tmp_path, [data], out) == s5.EXIT_CELLS_NOT_TRAINED
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["cells_failed"] == [CELLS3[2]]
    assert "RoutingMismatchError" in metrics["failures"][CELLS3[2]]


def test_gemm_counts_match_the_loader(tmp_path):
    from lib import subcells as sc

    a = write_round(tmp_path / "a", shapes_for(6, 14, CELLS3))
    with (a / "chunk_0000.csv").open() as f:
        row = next(csv.DictReader(f))
    b = tmp_path / "b"
    b.mkdir()
    with (b / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(row)
        w.writerow(dict(row, m="9999", us="nan"))
        w.writerow(dict(row, m="9998", mt_m="0"))
        w.writerow(dict(row, m="9997", sol_idx_global="", rank="-1"))
    rule = sc.build_split_rule(CELLS3[0], "M", 2500)
    tree = {rule.cell: rule}
    cells, stats = s5.load_enriched_chunks([a, b], True, tree)
    assert stats["bad_us"] == stats["bad_kernel_params"] == stats["skip_no_rank"] == 1
    shapes = s5.gemm_shapes_by_leaf([a, b, tmp_path / "missing"], tree)
    assert {k: len(v) for k, v in shapes.items()} == {
        k: len(v) for k, v in cells.items()
    }
    assert {rule.lo_label, rule.hi_label} <= set(shapes)
    assert sorted(shapes[rule.lo_label]) == sorted(
        (g["m"], g["n"], g["k"], g["batch_count"]) for g in cells[rule.lo_label]
    )


def test_requires_an_arch_with_constants(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(4, 2, CELLS3[:1]))
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage05_train.py"),
            "--enriched-csv-dir",
            str(data),
            "--output-dir",
            str(tmp_path / "o"),
            "--arch",
            "gfx90a",
            "--config-yaml",
            str(write_config(tmp_path / "c.yaml")),
            "--skip-deployed-eval",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0 and "no model constants" in proc.stderr


FAKE_V2_ENGINE = """
class _Info:
    def __init__(self, data):
        self.feature_catalog_hash = "e7fe4b524851e895"
        self.n_cells, self.n_splits = 1, 0


def load_model_from_memory(data):
    if bytes(data[:8]) != b"MLREC_v2":
        raise RuntimeError("<memory>: not a tilewright model file (bad magic)")
    return _Info(data)


def describe(model):
    return model
"""


def test_an_engine_that_cannot_load_the_trained_format_stops_at_once(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(4, 16, CELLS3[:1]))
    fake = tmp_path / "fake" / "tilewright"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_V2_ENGINE)
    out = tmp_path / "out"
    args = train_args(tmp_path, [data], out, "--catalog", "v3")
    args = [a for a in args if a != "--skip-deployed-eval"]
    path = os.pathsep.join([str(fake.parent), os.environ.get("PYTHONPATH", "")])
    proc = run_stage("stage05_train.py", *args, env={"PYTHONPATH": path}, expect_rc=1)
    err = proc.stdout + proc.stderr
    assert f"the tilewright module {fake / '__init__.py'} cannot serve" in err
    assert "MLREC_v3, catalog v3 b8b9f4a518add96f, bf16 weights" in err
    assert "bad magic" in err and "Traceback" not in err
    assert not (out / "models.pt").exists() and not (out / "metrics.json").exists()


@pytest.mark.parametrize("catalog", ["v2", "v3"])
def test_the_probe_model_reads_back_as_written(catalog):
    tw = pytest.importorskip("tilewright")
    import types

    args = types.SimpleNamespace(
        catalog=catalog,
        attribute_spec=None,
        arch="gfx950",
        library_stem=None,
        whitelist=None,
        cascade_k=None,
        non_temporal_rule=True,
        tie_tolerance=0.01,
        repeat_aggregation="median",
    )
    cat = s5.resolve_options(args, {})
    serving = s5.serving_options(args)
    data = s5.probe_model(
        cat, serving, catalog == "v3", "gfx950", hwlib.arch_constants("gfx950"), "int8"
    )
    assert data[:8] == f"MLREC_{catalog}".encode()
    assert s5.engine_round_trip_errors(data, tw) == []

    class Shifted:
        def __init__(self, info, **changes):
            self.info, self.changes = info, changes

        def __getattr__(self, name):
            return self.changes.get(name, getattr(self.info, name))

    wrong = types.SimpleNamespace(
        load_model_from_memory=tw.load_model_from_memory,
        describe=lambda m: Shifted(tw.describe(m), n_cells=2, query_dim=1),
    )
    assert s5.engine_round_trip_errors(data, wrong) == [
        "n_cells reads back as 2, written 1",
        f"query_dim reads back as 1, written {len(cat.names()[0])}",
    ]


# ── kernel identity ───────────────────────────────────────────────────────────


def twin_pool():
    """kernel_pool() plus a second solution with the parameters of each of
    its first six kernels."""
    base = kernel_pool()
    return base + [dict(k, sol_idx_global=500 + i) for i, k in enumerate(base[:6])]


def library_solutions(pool):
    """Logic-file solutions of `pool` entries, in the given order."""
    return [
        bf.solution(
            k["sol_idx_global"],
            local,
            (k["mt_m"], k["mt_n"], k["mt_k"]),
            (k["mi_m"], k["mi_n"], k["mi_k"], 1),
            nta=k["cache_hints_a"],
            ntb=k["cache_hints_b"],
            occupancy=k["occupancy"],
            grvw=(k["grvw_a"], k["grvw_b"]),
            gwvw=k["gwvw_d"],
        )
        for local, k in enumerate(pool)
    ]


def test_solutions_with_identical_parameters_stay_separate_candidates(tmp_path):
    pool = twin_pool()
    data = write_round(tmp_path / "stage04", shapes_for(9, 4, CELLS3[:1]), pool)
    base_attrs = dat.kernel_dat_info(bf.solution(0, 0))["attributes"]
    listed = {
        k["sol_idx_global"]: {
            "attributes": dict(base_attrs, workgroup_mapping=k["sol_idx_global"] % 7)
        }
        for k in pool[:-1]
    }
    dat.write_kernel_attributes(data / dat.KERNELS_FILE, listed, bf.LIBRARY_STEM)
    measured = {}
    for path in sorted(data.glob("chunk_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                key = (int(r["m"]), int(r["n"]), int(r["k"]), int(r["sol_idx_global"]))
                measured[key] = float(r["us"])

    cells, _stats = s5.load_enriched_chunks(str(data), True)
    (gemms,) = cells.values()
    assert len(gemms) == 4
    for g in gemms:
        by_sid = {c["sol_idx_global"]: c for c in g["candidates"]}
        assert len(by_sid) == len(g["candidates"]) == len(pool)
        for sid, c in by_sid.items():
            assert c["us"] == measured[(g["m"], g["n"], g["k"], sid)]
            if sid in listed:
                assert c["attributes"]["workgroup_mapping"] == sid % 7
            else:
                assert "attributes" not in c
        for i in range(6):
            base, twin = by_sid[pool[i]["sol_idx_global"]], by_sid[500 + i]
            assert fs.config_kwargs_from_row(base) == fs.config_kwargs_from_row(twin)
            assert base["us"] != twin["us"]
        tested = [c["us"] for c in g["candidates"] if not c["is_skip"]]
        assert g["winner_us"] == min(tested)

    names = (fs.query_feature_names(), fs.item_feature_names())
    names += (fs.interaction_feature_names(),)
    hw, consts = hwlib.DeviceHardware(**HARDWARE), hwlib.arch_constants("gfx950")
    _q, ivec, xvec, gidx, us, _win, _rank = s5.build_cell_tensors(
        gemms[:1], *names, hw, consts
    )
    sids = [c["sol_idx_global"] for c in gemms[0]["candidates"]]
    assert len(us) == len(sids) and (gidx == 0).all()
    for i in range(6):
        a, b = sids.index(pool[i]["sol_idx_global"]), sids.index(500 + i)
        assert (ivec[a] == ivec[b]).all() and (xvec[a] == xvec[b]).all()
        assert us[a] != us[b]


def test_a_signature_costs_its_fastest_solution():
    kern = kernel_pool()

    def cand(k, sid, us):
        return dict(k, sol_idx_global=sid, us=us)

    g = {
        "winner_us": 3.0,
        "candidates": [
            cand(kern[0], 1, 5.0),
            cand(kern[0], 2, 3.0),
            cand(kern[1], 3, 4.0),
        ],
    }
    sig_a, sig_b = (tuple(dat.sig_from_row(k)) for k in kern[:2])
    assert s5._select_smart_k_oracle([g], k=1) == [sig_a]
    assert s5._select_smart_k_oracle([g], k=2) == [sig_a, sig_b]


def test_the_student_aware_cache_costs_a_signature_its_fastest_solution(tmp_path):
    import torch

    pool = twin_pool()
    data = write_round(tmp_path / "stage04", shapes_for(11, 6, CELLS3[:1]), pool)
    (gemms,) = s5.load_enriched_chunks(str(data), True)[0].values()
    names = (fs.query_feature_names(), fs.item_feature_names())
    names += (fs.interaction_feature_names(),)
    hw, consts = hwlib.DeviceHardware(**HARDWARE), hwlib.arch_constants("gfx950")
    q, i, x, _gidx, _us, _win, _rank = s5.build_cell_tensors(gemms, *names, hw, consts)
    norms = s5.whiten(q, i, x)[3]
    torch.manual_seed(0)
    student = s5.GenericTwoTower(len(names[0]), len(names[1]), len(names[2]), 8, 16, 8)
    cache, winner_us = s5._student_score_cache(
        gemms, student, *names, hw, consts, norms, "cpu", {}
    )
    assert sorted(winner_us) == list(range(len(gemms)))
    n_checked = 0
    for gi, g in enumerate(gemms):
        prob = fs.problem_kwargs_from_row(g)
        fastest = {}
        for c in g["candidates"]:
            if not feasibility.passes_gates(
                prob,
                fs.config_kwargs_from_row(c),
                lds_bytes=hw.lds_bytes,
                nt_a_available=False,
                nt_b_available=False,
            ):
                continue
            sig = tuple(dat.sig_from_row(c))
            fastest[sig] = min(fastest.get(sig, math.inf), c["us"])
        assert {s: cache[s][gi][0] for s in fastest} == fastest
        n_checked += sum(1 for k in pool[:6] if tuple(dat.sig_from_row(k)) in fastest)
    assert n_checked > 0


class _FixedScores:
    def __init__(self, scores):
        import torch

        self.scores = torch.tensor(scores)

    def eval(self):
        pass

    def train(self):
        pass

    def score_pairs(self, q, i, x):
        return self.scores


def test_training_picks_break_score_ties_by_pair_order():
    import torch

    gidx = torch.tensor([0, 0, 0, 1, 1])
    us = torch.tensor([6.0, 4.0, 10.0, 2.0, 3.0], dtype=torch.float64)
    win = torch.tensor([4.0, 4.0, 4.0, 2.0, 2.0], dtype=torch.float64)
    q, i, x = torch.zeros(2, 1), torch.zeros(5, 1), torch.zeros(5, 1)
    out = s5.evaluate_sel_eff(
        _FixedScores([1.0, 1.0, 0.5, 0.2, 0.2]),
        q,
        i,
        x,
        gidx,
        us,
        win,
        sample_set_mask=torch.ones(2, dtype=torch.bool),
        device="cpu",
    )
    assert out["n_eval"] == 2
    assert out["sel_eff"] == pytest.approx(math.sqrt(4.0 / 6.0))
    assert out["pick_us_geomean"] == pytest.approx(math.sqrt(6.0 * 2.0))


def test_training_sees_each_gemms_candidates_in_pool_order(tmp_path, monkeypatch):
    pool = kernel_pool()
    data = write_round(tmp_path / "stage04", shapes_for(10, 14, CELLS3[2:]), pool)
    order = pool[::-1]
    lib = tmp_path / "library"
    lib.mkdir()
    bf.write_logic(lib / f"{bf.LIBRARY_STEM}.dat", library_solutions(order))
    position = {k["sol_idx_global"]: p for p, k in enumerate(order)}
    seen = []
    real = s5.train_cell

    def spy(**kw):
        seen.extend([c["sol_idx_global"] for c in g["candidates"]] for g in kw["gemms"])
        return real(**kw)

    monkeypatch.setattr(s5, "train_cell", spy)
    extra = ["--skip-deployed-eval", "--library-dir", lib]
    extra += ["--library-stem", bf.LIBRARY_STEM]
    assert _main(monkeypatch, tmp_path, [data], tmp_path / "out", *extra) == 0
    assert len(seen) == 14
    for sids in seen:
        assert sids == sorted(sids, key=position.__getitem__)
        assert sids != sorted(sids)


def test_a_pool_of_tdm_loaded_operands_refuses_what_reads_their_hints(
    tmp_path, monkeypatch, capsys
):
    """With a kernel whose operands the tensor data mover loads in its
    pool, stage05 stops before reading any GEMM when the model would serve
    the non-temporal rules (v2's default) or read such an operand's hint as
    an attribute; the same pool trains without them."""
    from test_kernel_identity import tdm_solution

    pool = kernel_pool()
    data = write_round(tmp_path / "stage04", shapes_for(10, 14, CELLS3[2:]), pool)
    lib = tmp_path / "library"
    lib.mkdir()
    sols = library_solutions(pool)
    bf.write_logic(
        lib / f"{bf.LIBRARY_STEM}.dat", sols + [tdm_solution(900, len(sols))]
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"name": "temporal_hint_a"}]))
    extra = ["--skip-deployed-eval", "--library-dir", lib]
    extra += ["--library-stem", bf.LIBRARY_STEM]
    out = tmp_path / "out"
    for argv, refused in (
        ((), "--non-temporal-rule (v2: without --no-non-temporal-rule): it reads"),
        (("--catalog", "v3", "--non-temporal-rule"), "--non-temporal-rule"),
        (("--catalog", "v3", "--attribute-spec", spec), "the attribute spec: it"),
    ):
        with pytest.raises(SystemExit) as e:
            _main(monkeypatch, tmp_path, [data], out, *extra, *argv)
        assert e.value.code == 1
        err = capsys.readouterr().err
        assert f"[identity] {refused}" in err
        assert "the tensor data mover loads operand A/B" in err
        assert not out.exists() or not (out / "models.pt").exists()
    rc = _main(monkeypatch, tmp_path, [data], out, *extra, "--no-non-temporal-rule")
    assert rc == 0
    options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
    assert options["identity_rule"] == "tdm_th_v1"
    assert load_bundle(out / "models.pt")["identity_rule"] == "tdm_th_v1"


def test_records_from_before_the_identity_rule_name_its_first_version():
    """A model trained before the identity rule was recorded counts as
    trained under tdm_th_v1, whose keys every pool it could have read
    gives, so the rule alone retrains no carried cell."""
    args = s5.arg_parser().parse_args(
        ["--enriched-csv-dir", "d", "--output-dir", "o", "--arch", "gfx950"]
    )
    current = s5.model_options(args, s5.resolve_options(args, {}))
    assert current["identity_rule"] == "tdm_th_v1"
    old = {k: v for k, v in current.items() if k != "identity_rule"}
    assert s5.changed_model_options(old, current) == {}
    assert s5.changed_model_options({**old, "identity_rule": "x"}, current) == {
        "identity_rule": ("x", "tdm_th_v1")
    }


def test_training_records_the_member_order_of_its_pool(tmp_path, monkeypatch):
    """With a library, stage05 writes the order of the members of every
    candidate key of its pool (twins here) for stage06 to compare."""
    from lib import evaluate as ev
    from test_kernel_identity import pool_library

    pool = twin_pool()
    data = write_round(tmp_path / "stage04", shapes_for(10, 14, CELLS3[2:]), pool)
    lib = tmp_path / "library"
    identity = pool_library(lib, pool)
    extra = ["--skip-deployed-eval", "--library-dir", lib]
    extra += ["--library-stem", bf.LIBRARY_STEM]
    assert _main(monkeypatch, tmp_path, [data], tmp_path / "out", *extra) == 0
    rec = json.loads((tmp_path / "out" / s5.MEMBER_ORDER_FILE).read_text())
    assert rec["library_digest"] == identity.digest and len(rec["keys"]) == 6
    library = ev.library_pool(lib, bf.LIBRARY_STEM)
    assert rec == s5.member_order(library, None, False, False, "gfx950", identity)
    out = tmp_path / "out"
    assert _main(monkeypatch, tmp_path, [data], out, "--skip-deployed-eval") == 0
    assert not (out / s5.MEMBER_ORDER_FILE).exists()


# ── whitening ─────────────────────────────────────────────────────────────────


def test_constant_item_features_whiten_to_zero(monkeypatch):
    rng = np.random.default_rng(0)
    n = 64
    alt = np.arange(n) % 2 == 1
    q = rng.normal(size=(n, 3)).astype(np.float32)
    q[:, 1] = 0.25
    i = rng.normal(size=(n, 4)).astype(np.float32)
    i[:, 0] = 1.0 / 9.0
    i[:, 2] = np.where(alt, 0.5, 0.5 + 1.8e-3)
    i[:, 3] = np.where(alt, 0.5, 0.5 + 2.2e-3)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    x[:, 0] = 3.0
    qw, iw, xw, norms = s5.whiten(q, i, x)
    sd = np.asarray(norms["i_std"], dtype=np.float32)
    assert sd[0] < 1e-6 and 0.8e-3 < sd[2] < 1e-3 < sd[3] < 1.2e-3
    dev = np.abs(i - np.asarray(norms["i_mean"], dtype=np.float32))
    zero = s5.whitens_to_zero("i", sd, dev)
    assert zero.shape == i.shape
    # Training values lie within 2 std of their own statistics.
    assert not zero.any()
    assert not s5.whitens_to_zero("q", norms["q_std"], q).any()
    assert not s5.whitens_to_zero("x", norms["x_std"], x).any()
    np.testing.assert_allclose(iw[:, 0], 0.0, atol=1e-6)
    assert (iw[:, 3] != 0).all()
    np.testing.assert_allclose(np.abs(iw[:, 2]), 1.0, rtol=1e-3)
    np.testing.assert_allclose(
        iw[:, 1], (i[:, 1] - i[:, 1].mean()) / i[:, 1].std(), rtol=1e-4
    )
    assert norms["q_std"][1] == 0.0 and (qw[:, 1] == 0).all()
    assert (xw[:, 0] == 0).all()

    pq, pi, px = s5.apply_whiten(
        np.asarray([[0.0, 0.75, 0.0]], dtype=np.float32),
        np.asarray([[2.0 / 9.0, 0.0, 0.7, 0.5]], dtype=np.float32),
        np.asarray([[5.0, 0.0]], dtype=np.float32),
        norms,
    )
    assert pi[0, 0] == 0 and pi[0, 2] == 0 and pi[0, 3] != 0
    assert pq[0, 1] == pytest.approx(0.5) and px[0, 0] == pytest.approx(2.0)

    monkeypatch.setattr(s5, "CONSTANT_FEATURE_STD", 0.0)
    assert not s5.whitens_to_zero("i", sd, dev).any()
    pi = s5.apply_whiten(q[:1], np.asarray([[2.0 / 9.0, 0, 0.5, 0.5]]), x[:1], norms)[1]
    assert pi[0, 0] == pytest.approx(1.0 / 9.0)


def test_training_whitening_matches_the_engine():
    if not have_engine():
        return
    import tilewright as tw
    import torch

    from lib import evaluate as ev
    from lib import mlrec

    if not hasattr(tw, "supports"):
        pytest.skip("the installed tilewright module predates the constant rule")
    names = (fs.query_feature_names(), fs.item_feature_names())
    names += (fs.interaction_feature_names(),)
    hw, consts = hwlib.DeviceHardware(**HARDWARE), hwlib.arch_constants("gfx950")
    pool = kernel_pool()[:16]
    rows = [gemm_rows((m, 2048, 4096, 1), pool[:1])[0] for m in range(1024, 4096, 384)]
    feats = [
        fs.feature_vectors(
            fs.problem_kwargs_from_row(r),
            fs.config_kwargs_from_row(k),
            hardware=hw,
            constants=consts,
        )
        for r in rows
        for k in pool
    ]
    q, i, x = (np.asarray([f[t] for f in feats], dtype=np.float32) for t in range(3))
    norms = s5.whiten(q, i, x)[3]
    constant = s5.whitens_to_zero("i", norms["i_std"], np.ones(len(names[1])))
    assert constant[names[1].index("occupancy_norm")] and not constant.all()
    torch.manual_seed(1)
    student = s5.GenericTwoTower(len(names[0]), len(names[1]), len(names[2]), 8, 16, 8)
    student.eval()
    label = "Large|Large|LargeK|Bnone"
    entry = {
        "state_dict": {k: v.detach() for k, v in student.state_dict().items()},
        "feature_norms": norms,
        "embed_dim": 8,
        "hidden_dim": 16,
        "inter_hidden": 8,
        "smart_k_signatures": [],
    }
    bundle = dict(zip(("q_names", "i_names", "x_names"), names), models={label: entry})
    model = tw.load_model_from_memory(
        mlrec.write_model(bundle, None, "gfx950", consts, "fp32")
    )
    probe = [pool[3], dict(pool[3], occupancy=2), dict(pool[3], occupancy=3), pool[9]]
    row = dict(rows[2])
    prob = fs.problem_kwargs_from_row(row)
    ranked = ev.candidate_set(tw, model, probe).rank(
        ev.make_problem(tw, prob), ev.make_hardware(tw, hw), len(probe)
    )
    assert all(r.scored for r in ranked)
    engine = {r.config_index: r.score for r in ranked}
    assert engine[0] == engine[1] == engine[2] != engine[3]
    for j, k in enumerate(probe):
        qv, iv, xv = fs.feature_vectors(
            prob, fs.config_kwargs_from_row(k), hardware=hw, constants=consts
        )
        qa, ia, xa = s5.apply_whiten(
            np.asarray([qv], dtype=np.float32),
            np.asarray([iv], dtype=np.float32),
            np.asarray([xv], dtype=np.float32),
            norms,
        )
        with torch.no_grad():
            py = student.score_pairs(
                torch.from_numpy(qa), torch.from_numpy(ia), torch.from_numpy(xa)
            )
        assert float(py[0]) == pytest.approx(engine[j], rel=1e-4, abs=1e-4)


def write_attributes(data, pool):
    base = dat.kernel_dat_info(bf.solution(0, 0))["attributes"]
    listed = {
        k["sol_idx_global"]: {
            "attributes": dict(
                base,
                workgroup_mapping=k["sol_idx_global"] % 5,
                tile_processing_strategy=k["sol_idx_global"] % 3,
                work_assignment=(
                    0
                    if k["sol_idx_global"] % 3 == 1
                    else (k["sol_idx_global"] // 3) % 3
                ),
            )
        }
        for k in pool
    }
    dat.write_kernel_attributes(data / dat.KERNELS_FILE, listed, bf.LIBRARY_STEM)


def write_context_round(out_dir, shapes, cu_budget, schedule, pool=None):
    out_dir = write_round(out_dir, shapes, pool)
    for path in sorted(out_dir.glob("chunk_*.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(COLUMNS) + ["cu_budget", "schedule"])
            w.writeheader()
            w.writerows(dict(r, cu_budget=cu_budget, schedule=schedule) for r in rows)
    return out_dir


def test_v3_training_smoke(tmp_path):
    pool = kernel_pool()
    data = write_round(tmp_path / "r0" / "stage04", shapes_for(0, 20, CELLS3[:2]))
    write_attributes(data, pool)
    shapes = shapes_for(1, 16, CELLS3[:2])
    ctx = write_context_round(tmp_path / "ctx" / "stage04", shapes, 50, "dynamic")
    write_attributes(ctx, pool)
    out = tmp_path / "v3" / "stage05"
    train(tmp_path, [data, ctx], out, "--catalog", "v3", "--whitelist", "view-key")
    bundle = load_bundle(out / "models.pt")
    assert bundle["catalog"] == "v3"
    spec = fs.parse_attribute_spec(bundle["attribute_spec"])
    assert spec == fs.DEFAULT_ATTRIBUTE_SPEC
    assert bundle["q_names"] == fs.v3_query_feature_names()
    assert bundle["i_names"] == fs.v3_item_feature_names(spec)
    assert bundle["contexts"] == {
        "min_cu_fraction": 0.5,
        "max_cu_fraction": 1.0,
        "schedules": ["default", "dynamic"],
    }
    assert bundle["clip"] is True
    assert bundle["serving"] == {
        "small_batched_rule": False,
        "dot2_rule": False,
        "non_temporal_rule": False,
        "non_temporal_family_rule": False,
        "whitelist_tier": True,
        "launch_keys": False,
        "trained_key_gate": True,
        "cascade_keys": 32,
    }
    rows = s5.load_enriched_chunks([str(data), str(ctx)], True)[0]
    trained = s5.trained_view_keys(
        [g for gs in rows.values() for g in gs], fs.Catalog("v3")
    )
    assert bundle["trained_view_keys"] == sorted(trained) and trained
    assert sorted(bundle["models"]) == sorted(CELLS3[:2])
    for entry in bundle["models"].values():
        assert entry["smart_k_signatures"] == [] and entry["view_key_hashes"]
        b = entry["feature_bounds"]
        assert len(b["i_lo"]) == len(bundle["i_names"])
        assert all(lo <= hi for lo, hi in zip(b["x_lo"], b["x_hi"]))
    cells = json.loads((out / "cells.json").read_text())
    assert cells["catalog"] == "v3" and len(cells["uncovered_base_cells"]) == 94
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["feature_names_hash"] == fs.v3_catalog_hash()
    assert metrics["load_stats"]["gemms_non_exclusive"] == len(set(shapes))

    # Provenance: the same in models.pt and metrics.json, over the chunks of
    # both folders in training order.
    from lib import hipblaslt_launch
    from lib import provenance as prov

    recorded = bundle["provenance"]
    assert metrics["provenance"] == recorded
    assert list(recorded) == [
        "config_id",
        "training_data_hash",
        "source_revision",
        "date",
        "launch_model",
    ]
    assert recorded["config_id"] == "unit_bbs_tn"
    chunks = [p for d in (data, ctx) for p in sorted(Path(d).glob("chunk_*.csv"))]
    assert recorded["training_data_hash"] == prov.training_data_hash(chunks)
    assert recorded["source_revision"] == prov.source_revision()
    assert recorded["launch_model"] == hipblaslt_launch.HipblasltLaunch("gfx950").id

    if not have_engine():
        return
    import tilewright as tw

    from lib import evaluate as ev
    from lib import hardware as hwlib
    from lib import mlrec

    for c in cells["cells"]:
        assert c["deployed"]["n_model_served"] == c["deployed"]["n_eval"] > 0
    data_v3 = mlrec.write_model(
        bundle, None, "gfx950", hwlib.arch_constants("gfx950"), "bf16"
    )
    info = tw.describe(tw.load_model_from_memory(data_v3))
    assert info.format_version == 3 and info.uncovered_base_labels == 94
    assert info.provenance.launch_model == recorded["launch_model"]
    assert info.provenance.training_data_hash == recorded["training_data_hash"]
    gemms = [
        g
        for gs in s5.load_enriched_chunks([str(data), str(ctx)], True)[0].values()
        for g in gs
    ]
    hw = hwlib.DeviceHardware(**HARDWARE)
    result = ev.evaluate_model(data_v3, gemms, hardware=hw)
    assert result.n_routing_mismatch == 0
    assert result.summary()["n_model_served"] == len(gemms)
    other = [dict(g, m=8, n=200, batch_count=4) for g in gemms[:3]]
    assert ev.evaluate_model(data_v3, other, hardware=hw).n_routing_mismatch == 0


def _planned_rounds(tmp_path, n_per_cell=16):
    """Two v3 rounds of one cell benched from two halves of the pool, and a
    plan listing each shape's half: every GEMM has pool kernels off its
    list that the other round measured."""
    pool = kernel_pool()
    halves = (pool[: len(pool) // 2], pool[len(pool) // 2 :])
    plan = {"shapes": []}
    dirs = []
    for r, half in enumerate(halves):
        shapes = shapes_for(10 + r, n_per_cell, CELLS3[:1])
        d = write_round(tmp_path / f"r{r}" / "stage04", shapes, half)
        write_attributes(d, pool)
        dirs.append(d)
        listed = [k["sol_idx_global"] for k in half]
        plan["shapes"] += [
            {"m": m, "n": n, "k": k, "batch": b, "kernels": listed}
            for m, n, k, b in shapes
        ]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return dirs, path


PLANNED_V3 = ("--catalog", "v3", "--min-cell-gemms", 8, "--min-base-cell-gemms", 8)


def test_planned_training_counts_off_list_picks(tmp_path):
    dirs, plan = _planned_rounds(tmp_path)
    runs = {}
    for name, extra in (
        ("skip", ()),
        ("plan", ("--plan", plan)),
        ("charged", ("--plan", plan, "--off-list-sel-eff", 1)),
    ):
        out = tmp_path / name / "stage05"
        train(tmp_path, dirs, out, *PLANNED_V3, *extra)
        runs[name] = json.loads((out / "metrics.json").read_text())
    assert runs["skip"]["load_stats"]["gemms_listed"] == 0
    assert runs["plan"]["load_stats"]["gemms_listed"] == 32
    if have_engine():
        val = {k: m["validation"] for k, m in runs.items()}
        off = {k: v["n_miss_by_reason"]["not_benched"] for k, v in val.items()}
        assert off["skip"] == 0 and off["plan"] > 0
        assert val["charged"]["sel_eff"] > val["plan"]["sel_eff"]


def test_auto_off_list_charge_comes_from_the_rebench_record(tmp_path):
    dirs, plan = _planned_rounds(tmp_path)
    rows = [
        {"cu_budget": 0, "schedule": "default", "reason": "not_benched"}
        | {"status": "measured", "ratio": r}
        for r in (0.5, 0.8, 1.25)
    ]
    record = tmp_path / "rebench_misses.json"
    record.write_text(json.dumps({"held_out": {"gemms": rows}}))
    auto = ("--plan", plan, "--off-list-sel-eff", "auto", "--off-list-min-misses", 3)
    runs, policy = {}, {}
    for name, extra in (
        ("measured", ("--off-list-rebench", record)),
        ("missing", ("--off-list-rebench", tmp_path / "none.json")),
        ("short", ("--off-list-rebench", record, "--off-list-fallback", 0.7)),
        ("fixed", ("--plan", plan, "--off-list-sel-eff", 0.4 ** (1 / 3))),
    ):
        out = tmp_path / name / "stage05"
        args = extra if name == "fixed" else (*auto, *extra)
        if name == "short":
            args = (*args, "--off-list-min-misses", 4)
        train(tmp_path, dirs, out, *PLANNED_V3, *args)
        runs[name] = json.loads((out / "metrics.json").read_text())
        options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
        policy[name] = options["off_list_sel_eff"]
    assert [policy[k] for k in ("measured", "missing", "short")] == ["auto"] * 3
    assert policy["fixed"] == pytest.approx(0.4 ** (1 / 3))
    got = runs["measured"]["off_list_charge"]
    assert got["source"] == "rebench" and got["n"] == 3
    assert got["sel_eff"] == pytest.approx(0.4 ** (1 / 3))
    assert got["rebench_file"] == str(record)
    assert runs["missing"]["off_list_charge"]["source"] == "fallback"
    assert runs["missing"]["off_list_charge"]["sel_eff"] is None
    assert runs["short"]["off_list_charge"]["sel_eff"] == 0.7
    assert runs["fixed"]["off_list_charge"] == {
        "source": "configured",
        "sel_eff": pytest.approx(0.4 ** (1 / 3)),
    }
    if have_engine():
        val = {k: m["validation"] for k, m in runs.items()}
        assert val["measured"]["n_miss_by_reason"]["not_benched"] > 0
        assert val["measured"]["sel_eff"] == pytest.approx(val["fixed"]["sel_eff"])
        assert val["missing"]["sel_eff"] < val["measured"]["sel_eff"]


def test_pseudo_negatives_join_the_loss_of_planned_gemms(tmp_path):
    dirs, plan = _planned_rounds(tmp_path)
    v3 = PLANNED_V3
    out = tmp_path / "plain" / "stage05"
    train(tmp_path, dirs, out, *v3, "--plan", plan)
    cells = json.loads((out / "cells.json").read_text())["cells"]
    assert [c["n_unmeasured_pairs"] for c in cells] == [0]

    out = tmp_path / "pseudo" / "stage05"
    train(tmp_path, dirs, out, *v3, "--plan", plan, "--unmeasured-samples", 3)
    (cell,) = json.loads((out / "cells.json").read_text())["cells"]
    assert 0 < cell["n_unmeasured_pairs"] <= 3 * cell["n_train_gemms"]
    if have_engine():
        assert cell["deployed_val"]["n_eval"] == cell["n_val_gemms"]


def test_stage1_recall_loss_moves_only_the_embedding_term_toward_the_full_ranking():
    torch = pytest.importorskip("torch")
    gidx = torch.tensor([0, 0, 0, 1, 1])
    scores = torch.tensor([3.0, 0.0, -1.0, 0.0, 2.0], requires_grad=True)
    agree = torch.tensor([2.0, 0.0, -1.0, -1.0, 1.0])
    disagree = torch.tensor([-1.0, 0.0, 2.0, 1.0, -1.0])
    assert float(s5.stage1_recall_loss(agree, scores, gidx, 2)) < float(
        s5.stage1_recall_loss(disagree, scores, gidx, 2)
    )
    dot = disagree.clone().requires_grad_(True)
    s5.stage1_recall_loss(dot, scores, gidx, 2).backward()
    assert scores.grad is None and dot.grad is not None
    assert float(dot.grad[2]) > 0 > float(dot.grad[0])
    w = torch.tensor([1.0, 0.0])
    assert float(
        s5.stage1_recall_loss(disagree, scores, gidx, 2, weights=w)
    ) == pytest.approx(
        float(s5.stage1_recall_loss(disagree[:3], scores[:3], gidx[:3], 1))
    )


def _option(key):
    """The stage05 option of train key `key`, as the orchestrator passes it
    (run_orchestrator.STAGE05_OPTION_KEYS)."""
    return "--" + key.replace("_", "-")


def test_stage1_recall_weight_changes_only_the_trained_weights(tmp_path):
    dirs, plan = _planned_rounds(tmp_path)
    torch = pytest.importorskip("torch")
    states = {}
    for w in (None, "0", "1"):
        out = tmp_path / f"w{w}" / "stage05"
        extra = () if w is None else (_option("stage1_recall_weight"), w)
        train(tmp_path, dirs, out, *PLANNED_V3, "--plan", plan, *extra)
        bundle = torch.load(str(out / "models.pt"), weights_only=True)
        (entry,) = bundle["models"].values()
        states[w] = entry["state_dict"]
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)  # noqa: E731
    assert same(states[None], states["0"])
    assert not same(states["0"], states["1"])


def test_global_deployed_numbers_weight_cells_by_their_evaluated_gemms():
    cells = [
        {"n_val_gemms": 100, "deployed_val": {"sel_eff": 0.5, "n_eval": 10}},
        {"n_val_gemms": 10, "deployed_val": {"sel_eff": 1.0, "n_eval": 10}},
        {"n_val_gemms": 5, "deployed_val": None},
        {"n_val_gemms": 5, "deployed_val": {"sel_eff": float("nan"), "n_eval": 0}},
    ]
    assert s5.pooled_mean(cells, "deployed_val", "sel_eff") == pytest.approx(
        math.sqrt(0.5)
    )


def test_unmeasured_kernels_are_feasible_and_off_the_list():
    pool = kernel_pool()
    g = {
        "m": 64,
        "n": 4096,
        "k": 1024,
        "batch_count": 1,
        "transA": "T",
        "transB": "N",
        "a_type": "bf16_r",
        "b_type": "bf16_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "compute_type": "f32_r",
        "candidates": [dict(pool[0], us=1.0)],
        "listed": frozenset({pool[0]["sol_idx_global"], pool[1]["sol_idx_global"]}),
    }
    kw = dict(
        lds_bytes=65536, nt_flags=(False, True), rules={"non_temporal_rule": True}
    )
    got = s5.unmeasured_kernels(g, pool, 100, random.Random(0), **kw)
    ids = {k["sol_idx_global"] for k in got}
    assert ids and not ids & g["listed"]
    prob = fs.problem_kwargs_from_row(g)
    for k in pool:
        ok = feasibility.passes_gates(
            prob,
            fs.config_kwargs_from_row(k),
            lds_bytes=65536,
            nt_a_available=False,
            nt_b_available=True,
            non_temporal_rule=True,
        )
        assert (k["sol_idx_global"] in ids) == (
            ok and k["sol_idx_global"] not in g["listed"]
        )
    assert len(s5.unmeasured_kernels(g, pool, 2, random.Random(0), **kw)) == 2
    assert (
        s5.unmeasured_kernels(dict(g, listed=None), pool, 5, random.Random(0), **kw)
        == []
    )


def test_unsupported_kernels_draw_keys_no_training_gemm_measured():
    pool, _ = _served_pool()
    key_of = s5.served_key_of(fs.Catalog("v3"), False)
    measured = dict(GEMM_KEY, candidates=[dict(pool[1], us=1.0, is_skip=False)])
    supported = s5.supported_keys([measured], key_of)
    assert supported == {key_of(pool[0])}
    cands = s5.unsupported_candidates(pool, supported, key_of)
    # One entry per unsupported key, its first pool kernel and its kernels.
    assert [(k["sol_idx_global"], sorted(s)) for k, s in cands] == [
        (20, [20]),
        (30, [30, 31]),
    ]
    kw = dict(
        lds_bytes=65536, nt_flags=(False, False), rules={"non_temporal_rule": True}
    )
    g = _served_gemm([(11, 1.0, False)], listed={10, 11})
    got = s5.unsupported_kernels(g, cands, 10, random.Random(0), **kw)
    assert sorted(k["sol_idx_global"] for k in got) == [20, 30]
    listed = _served_gemm([(11, 1.0, False)], listed={10, 11, 31})
    got = s5.unsupported_kernels(listed, cands, 10, random.Random(0), **kw)
    assert [k["sol_idx_global"] for k in got] == [20]
    assert len(s5.unsupported_kernels(g, cands, 1, random.Random(0), **kw)) == 1
    unplanned = dict(g, listed=None)
    assert s5.unsupported_kernels(unplanned, cands, 5, random.Random(0), **kw) == []
    # Both unsupported keys need more LDS than this.
    small_lds = dict(kw, lds_bytes=16384)
    assert s5.unsupported_kernels(g, cands, 10, random.Random(0), **small_lds) == []


def test_targeted_draws_admit_the_members_the_family_rule_keeps():
    # A batched problem thin in M with K off the DepthU alignment: the
    # non-temporal rule rejects every hinted kernel, and the family rule
    # keeps the lowest hints of a family without a hint-free member.
    base = dict(zip(CONFIG, (16, 64, 64, 16, 16, 32, 1, 0, 0, 8, 8, 4)))
    pool = [
        dict(base, sol_idx_global=1),
        dict(base, mt_n=128, cache_hints_a=1, cache_hints_b=1, sol_idx_global=2),
        dict(base, mt_n=128, cache_hints_a=1, cache_hints_b=3, sol_idx_global=3),
    ]
    g = dict(GEMM_KEY, m=8, n=4096, k=48, batch_count=16, listed=frozenset({1}))
    g["candidates"] = [dict(pool[0], us=1.0, is_skip=False)]
    key_of = s5.served_key_of(fs.Catalog("v3"), False)
    cands = s5.unsupported_candidates(pool, s5.supported_keys([g], key_of), key_of)
    rules = feasibility.soft_rules(
        {"non_temporal_rule": True, "non_temporal_family_rule": True}, 3
    )
    kw = dict(lds_bytes=65536, nt_flags=(False, False), rules=rules)
    families = feasibility.non_temporal_families(pool)
    got = s5.unsupported_kernels(
        g, cands, 5, random.Random(0), **kw, nt_families=families
    )
    assert [k["sol_idx_global"] for k in got] == [2]
    plain = feasibility.soft_rules({"non_temporal_rule": True}, 3)
    kw = dict(kw, rules=plain)
    assert s5.unsupported_kernels(g, cands, 5, random.Random(0), **kw) == []


def test_targeted_pseudo_negatives_go_only_where_the_cell_lacks_keys(tmp_path):
    pool = kernel_pool()
    halves = (pool[: len(pool) // 2], pool[len(pool) // 2 :])
    runs = {}
    for name in ("all", "split"):
        dirs, entries = [], []
        for c, cell in enumerate(CELLS3[:2]):
            part = pool if name == "all" else halves[c]
            shapes = shapes_for(10 + c, 16, [cell])
            d = write_round(tmp_path / name / f"c{c}" / "stage04", shapes, part)
            write_attributes(d, pool)
            dirs.append(d)
            listed = [k["sol_idx_global"] for k in part]
            entries += [
                {"m": m, "n": n, "k": k, "batch": b, "kernels": listed}
                for m, n, k, b in shapes
            ]
        path = tmp_path / name / "plan.json"
        path.write_text(json.dumps({"shapes": entries}))
        out = tmp_path / name / "stage05"
        train(
            tmp_path,
            dirs,
            out,
            *PLANNED_V3,
            "--plan",
            path,
            "--pseudo-negatives",
            "targeted",
            "--unmeasured-samples",
            3,
        )
        runs[name] = json.loads((out / "cells.json").read_text())["cells"]
    assert [c["n_unmeasured_pairs"] for c in runs["all"]] == [0, 0]
    for cell in runs["split"]:
        assert 0 < cell["n_unmeasured_pairs"] <= 3 * cell["n_train_gemms"]
    options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
    assert options["pseudo_negatives"] == "targeted"
    assert (options["unmeasured_margin"], options["unmeasured_weight"]) == (1.0, 0.1)


def test_an_unnamed_draw_trains_as_random(tmp_path):
    torch = pytest.importorskip("torch")
    dirs, plan = _planned_rounds(tmp_path)
    states = {}
    for name, extra in (("default", ()), ("random", ("--pseudo-negatives", "random"))):
        out = tmp_path / name / "stage05"
        train(
            tmp_path,
            dirs,
            out,
            *PLANNED_V3,
            "--plan",
            plan,
            "--unmeasured-samples",
            3,
            *extra,
        )
        bundle = torch.load(str(out / "models.pt"), weights_only=True)
        (entry,) = bundle["models"].values()
        states[name] = entry["state_dict"]
        options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
        # The sample count implies the draw, so the record leaves it out.
        assert options["pseudo_negatives"] is None
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["args"]["pseudo_negatives"] == "random"
    a, b = states["default"], states["random"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def _resolved(*argv):
    args = s5.arg_parser().parse_args(
        ["--enriched-csv-dir", "d", "--output-dir", "o", "--arch", "gfx950", *argv]
    )
    s5.resolve_pseudo_negatives(args)
    return (
        args.pseudo_negatives,
        args.unmeasured_samples,
        args.unmeasured_margin,
        args.unmeasured_weight,
    )


def test_each_pseudo_negative_draw_has_its_defaults():
    assert _resolved() == ("none", 0, 0.05, 0.3)
    assert _resolved("--unmeasured-samples", "128") == ("random", 128, 0.05, 0.3)
    assert _resolved("--pseudo-negatives", "targeted") == ("targeted", 16, 1.0, 0.1)
    assert _resolved(
        "--pseudo-negatives", "targeted", "--unmeasured-margin", "0.5"
    ) == ("targeted", 16, 0.5, 0.1)
    assert _resolved("--pseudo-negatives", "random", "--unmeasured-samples", "4")[
        :2
    ] == ("random", 4)


def test_the_recorded_draw_is_the_one_the_sample_count_does_not_imply():
    def recorded(*argv):
        args = s5.arg_parser().parse_args(
            ["--enriched-csv-dir", "d", "--output-dir", "o", "--arch", "gfx950", *argv]
        )
        return s5.model_options(args, s5.resolve_options(args, {}))

    old = {"unmeasured_samples": 128}
    for argv in (
        ("--unmeasured-samples", "128"),
        ("--pseudo-negatives", "random", "--unmeasured-samples", "128"),
    ):
        current = recorded(*argv)
        assert current["pseudo_negatives"] is None
        assert "pseudo_negatives" not in s5.changed_model_options(old, current)
    targeted = recorded("--pseudo-negatives", "targeted", "--catalog", "v3")
    assert targeted["pseudo_negatives"] == "targeted"
    changes = s5.changed_model_options(old, targeted)
    assert changes["pseudo_negatives"] == (None, "targeted")
    assert changes["unmeasured_samples"] == (128, 16)


def test_pseudo_negative_options_that_contradict_stop_the_stage(tmp_path):
    dirs, plan = _planned_rounds(tmp_path)
    for i, extra in enumerate(
        (
            ("--pseudo-negatives", "none", "--unmeasured-samples", 3),
            ("--pseudo-negatives", "random"),
            ("--pseudo-negatives", "targeted", "--unmeasured-samples", 0),
        )
    ):
        out = tmp_path / f"bad{i}" / "stage05"
        proc = train(
            tmp_path, dirs, out, *PLANNED_V3, "--plan", plan, *extra, expect_rc=1
        )
        assert "--pseudo-negatives" in proc.stdout + proc.stderr
    proc = train(
        tmp_path,
        dirs,
        tmp_path / "v2" / "stage05",
        "--pseudo-negatives",
        "targeted",
        expect_rc=1,
    )
    assert "targeted needs --catalog v3" in proc.stdout + proc.stderr


def _served_pool():
    base = dict(zip(CONFIG, (64, 64, 64, 16, 16, 32, 1, 0, 0, 8, 8, 4)))
    other = dict(base, mt_n=128)
    third = dict(base, mt_m=128)
    pool = [
        dict(base, sol_idx_global=10),
        dict(base, sol_idx_global=11, grvw_a=4),
        dict(other, sol_idx_global=20),
        dict(third, sol_idx_global=30),
        dict(third, sol_idx_global=31, gwvw_d=2),
    ]
    return pool, {s5.ev.pool_key(GEMM_KEY): pool}


GEMM_KEY = {
    "m": 512,
    "n": 512,
    "k": 512,
    "batch_count": 1,
    "transA": "T",
    "transB": "N",
    "a_type": "bf16_r",
    "b_type": "bf16_r",
    "c_type": "bf16_r",
    "d_type": "bf16_r",
    "compute_type": "f32_r",
}


def _served_gemm(rows, listed=None):
    g = dict(
        GEMM_KEY,
        candidates=[dict(sol_idx_global=s, us=u, is_skip=k) for s, u, k in rows],
    )
    g["winner_us"] = min(c["us"] for c in g["candidates"] if not c["is_skip"])
    if listed is not None:
        g["listed"] = frozenset(listed)
    return g


def test_served_candidates_keep_the_kernel_the_runtime_runs_per_key():
    pool, pools = _served_pool()
    key_of = s5.served_key_of(fs.Catalog("v3"), False)
    rows = [(11, 1.0, False), (20, 2.0, False), (31, 1.5, False), (99, 3.0, False)]
    planned = _served_gemm(rows, listed={10, 11, 20, 31})
    exhaustive = _served_gemm(rows)
    stats = s5.served_candidates([planned, exhaustive], pools, key_of)
    # 10 was benched without a row, so 11 serves its key; 30 is off the plan,
    # so its key was not measured as served; 99 is outside the pool.
    assert [c["sol_idx_global"] for c in planned["candidates"]] == [11, 20, 99]
    assert [c["sol_idx_global"] for c in exhaustive["candidates"]] == [11, 20, 31, 99]
    assert planned["target_winner_us"] == 1.0 and s5.target_winner_us(planned) == 1.0
    assert stats["keys_unmeasured"] == 1 and stats["candidates_in"] == 8
    assert stats["candidates_kept"] == 7

    fast_second = _served_gemm([(10, 2.0, False), (11, 1.0, False)])
    s5.served_candidates([fast_second], pools, key_of)
    assert [c["sol_idx_global"] for c in fast_second["candidates"]] == [10]
    assert fast_second["winner_us"] == 1.0 and fast_second["target_winner_us"] == 2.0

    nothing = _served_gemm([(31, 1.0, False)], listed={31})
    stats = s5.served_candidates([nothing], pools, key_of)
    assert stats["gemms_without_served"] == 1 and len(nothing["candidates"]) == 1
    assert s5.target_winner_us({"winner_us": 3.0}) == 3.0


def test_served_candidates_split_launch_classes_only_with_launch_keys():
    pool, pools = _served_pool()
    pool[1]["attributes"] = {"tile_processing_strategy": 2, "work_assignment": 1}
    catalog = fs.Catalog(
        "v3", fs.DEFAULT_ATTRIBUTE_SPEC, hipblaslt_launch.HipblasltLaunch("gfx950")
    )
    rows = [(10, 2.0, False), (11, 1.0, False)]
    view = _served_gemm(rows)
    s5.served_candidates([view], pools, s5.served_key_of(catalog, False))
    assert [c["sol_idx_global"] for c in view["candidates"]] == [10]
    launch = _served_gemm(rows)
    s5.served_candidates([launch], pools, s5.served_key_of(catalog, True))
    assert [c["sol_idx_global"] for c in launch["candidates"]] == [10, 11]


def test_served_candidates_reach_training(tmp_path):
    pool = kernel_pool()
    pool += [dict(k, sol_idx_global=200 + i, grvw_a=4) for i, k in enumerate(pool[:4])]
    shapes = shapes_for(0, 14, CELLS3[:1])
    data = write_round(tmp_path / "r0" / "stage04", shapes, pool)
    out = tmp_path / "round_0" / "stage05"
    train(
        tmp_path,
        [data],
        out,
        "--catalog",
        "v3",
        "--served-candidates",
        "--skip-deployed-eval",
    )
    metrics = json.loads((out / "metrics.json").read_text())
    served = metrics["load_stats"]["served_candidates"]
    assert served["candidates_in"] - served["candidates_kept"] == 4 * len(set(shapes))
    options = json.loads((out / "model_options.json").read_text())
    assert options["served_candidates"] is True
    train(
        tmp_path,
        [data],
        tmp_path / "v2",
        "--served-candidates",
        "--skip-deployed-eval",
        expect_rc=1,
    )


def test_unmeasured_pairs_join_each_gemm_after_its_measured_pairs():
    q = np.zeros((2, 1), np.float32)
    i = np.arange(3, dtype=np.float32).reshape(3, 1)
    x = np.zeros((3, 1), np.float32)
    gidx = np.array([0, 0, 1])
    us = np.array([1.0, 2.0, 4.0])
    win = np.array([1.0, 1.0, 4.0])
    gemms = [{"winner_us": 1.0}, {"winner_us": 4.0}]
    kernels = [[{"k": 0}], [{"k": 1}, {"k": 2}]]

    class Catalog:
        def vectors(self, *a, **kw):
            return [0.0], [9.0], [0.0]

    import unittest.mock as um

    with um.patch.object(
        s5.fs, "problem_kwargs_from_row", lambda g: {}
    ), um.patch.object(s5.fs, "config_kwargs_from_row", lambda k: {}), um.patch.object(
        s5.ev, "context_of", lambda g: None
    ):
        arrays, measured, pseudo = s5.with_unmeasured_pairs(
            (q, i, x, gidx, us, win),
            gemms,
            kernels,
            0.1,
            q_names=[],
            i_names=[],
            x_names=[],
            hardware=None,
            constants=None,
            catalog=Catalog(),
        )
    assert arrays[3].tolist() == [0, 0, 0, 1, 1, 1]
    assert pseudo.tolist() == [False, False, True, False, True, True]
    assert measured.tolist() == [0, 1, 3]
    assert arrays[1][:, 0].tolist() == [0.0, 1.0, 9.0, 2.0, 9.0, 9.0]
    assert arrays[4].tolist() == pytest.approx([1.0, 2.0, 1.1, 4.0, 4.4, 4.4])
    assert arrays[5].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]


POOLED = (
    "--recipe",
    "pooled_ft",
    "--pooled-epochs",
    3,
    "--pooled-n-estimators",
    8,
    "--fine-tune-epochs",
    3,
)
WEIGHTS = ("q_proj", "i_proj", "inter_mlp")


def _v3_round(tmp_path, n_per_cell=24, cells=CELLS3):
    pool = kernel_pool()
    data = write_round(tmp_path / "r0" / "stage04", shapes_for(0, n_per_cell, cells))
    write_attributes(data, pool)
    return data


def _same_weights(a, b):
    import torch

    return all(torch.equal(a[k], b[k]) for k in a if k.startswith(WEIGHTS))


def test_the_pooled_recipe_fine_tunes_one_student_in_every_cell(tmp_path):
    data = _v3_round(tmp_path)
    out = tmp_path / "stage05"
    train(tmp_path, [data], out, "--catalog", "v3", *POOLED)
    bundle = load_bundle(out / "models.pt")
    assert bundle["recipe"] == "pooled_ft"
    pooled = bundle["pooled"]["entry"]
    widths = (pooled["embed_dim"], pooled["hidden_dim"], pooled["inter_hidden"])
    assert widths == s5.POOLED_CAPACITY
    cells = {
        c["label"]: c for c in json.loads((out / "cells.json").read_text())["cells"]
    }
    assert sorted(bundle["models"]) == sorted(cells) == sorted(CELLS3)
    for label, entry in bundle["models"].items():
        # Every cell keeps the pooled student's features and widths.
        for key in ("feature_norms", "feature_bounds", "embed_dim", "hidden_dim"):
            assert entry[key] == pooled[key], key
        c = cells[label]
        assert c["best_phase"] in ("pooled", "fine_tune")
        assert 0 < c["pooled_val_sel_eff"] <= 1
        kept = _same_weights(entry["state_dict"], pooled["state_dict"])
        assert kept == (c["best_phase"] == "pooled") == (c["best_epoch"] == -1)
        assert c["steps"] == 3 * math.ceil(c["n_train_gemms"] / 16)
    with (out / "training_log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["cell"] for r in rows} == {s5.POOLED_LABEL, *CELLS3}
    assert {r["phase"] for r in rows} == {"pooled", "fine_tune"}
    options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
    assert options["recipe"] == "pooled_ft"
    metrics = json.loads((out / "metrics.json").read_text())
    student = metrics["pooled"]["student"]
    assert metrics["recipe"] == "pooled_ft" and student["n_cells"] == 3
    assert student["n_val_gemms"] == sum(c["n_val_gemms"] for c in cells.values())
    assert metrics["pooled"]["teacher"]["n_estimators"] == 8
    # 0 threads are every CPU the stage may use, recorded as the count.
    cpus = len(os.sched_getaffinity(0))
    assert metrics["pooled"]["threads"] == metrics["pooled"]["teacher"]["threads"]
    assert metrics["pooled"]["threads"] == cpus
    tune = metrics["pooled"]["fine_tune"]
    assert (tune["workers"], tune["threads_per_worker"]) == (1, max(2, cpus))
    if have_engine():
        for c in cells.values():
            assert c["deployed_val"]["n_eval"] == c["n_val_gemms"]


def test_without_fine_tuning_every_cell_serves_the_pooled_student(tmp_path):
    data = _v3_round(tmp_path, cells=CELLS3[:2])
    out = tmp_path / "stage05"
    train(tmp_path, [data], out, "--catalog", "v3", *POOLED, "--fine-tune-epochs", 0)
    bundle = load_bundle(out / "models.pt")
    pooled = bundle["pooled"]["entry"]["state_dict"]
    for entry in bundle["models"].values():
        assert _same_weights(entry["state_dict"], pooled)
        assert entry["best_epoch"] == -1


def test_cells_left_out_of_training_still_teach_the_pooled_student(tmp_path):
    """--skip-cells, like --only-cells, chooses the cells to train: the
    pooled phase learns from every loaded cell."""
    data = _v3_round(tmp_path)
    records = {}
    for name, pick in (
        ("skip", ("--skip-cells", CELLS3[2])),
        ("only", ("--only-cells", ",".join(CELLS3[:2]))),
    ):
        out = tmp_path / name
        train(tmp_path, [data], out, "--catalog", "v3", *POOLED, *pick)
        assert sorted(load_bundle(out / "models.pt")["models"]) == sorted(CELLS3[:2])
        cells = json.loads((out / "cells.json").read_text())["cells"]
        assert sorted(c["label"] for c in cells) == sorted(CELLS3[:2])
        records[name] = json.loads((out / "metrics.json").read_text())["pooled"]
    same = ("n_cells", "n_train_gemms", "n_pairs", "n_val_gemms", "best_val_sel_eff")
    skip, only = ({k: records[n]["student"][k] for k in same} for n in ("skip", "only"))
    assert skip == only and skip["n_cells"] == 3


def test_the_pooled_recipe_chooses_whitelists_with_the_fine_tuned_weights(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(2, 20, CELLS3[:2]))
    out = tmp_path / "stage05"
    train(tmp_path, [data], out, *POOLED, "--skip-deployed-eval")
    bundle = load_bundle(out / "models.pt")
    cells = {
        c["label"]: c for c in json.loads((out / "cells.json").read_text())["cells"]
    }
    for label, entry in bundle["models"].items():
        assert 0 < len(entry["smart_k_signatures"]) <= 3
        assert cells[label]["smart_k_metrics"]["mode"] != "none"


def test_the_pooled_recipe_trains_alike_serially_and_in_parallel(tmp_path):
    torch = pytest.importorskip("torch")
    data = _v3_round(tmp_path)
    threads = (
        "--train-threads-per-worker",
        1,
        "--pooled-threads",
        2,
        "--pooled-teacher-threads",
        3,
    )
    states = {}
    for name, extra, workers in (
        ("serial", (), 1),
        ("parallel", ("--train-workers", 2), 2),
    ):
        out = tmp_path / name
        proc = train(
            tmp_path, [data], out, "--catalog", "v3", *POOLED, *threads, *extra
        )
        assert "retrying serially" not in proc.stdout + proc.stderr
        states[name] = load_bundle(out / "models.pt")
        record = json.loads((out / "metrics.json").read_text())["pooled"]
        assert (record["threads"], record["teacher"]["threads"]) == (2, 3)
        tune = record["fine_tune"]
        assert (tune["workers"], tune["threads_per_worker"]) == (workers, 1)
        assert states[name]["pooled"]["fine_tune"] == tune
    a, b = states["serial"], states["parallel"]
    for key in a["pooled"]["entry"]["state_dict"]:
        assert torch.equal(
            a["pooled"]["entry"]["state_dict"][key],
            b["pooled"]["entry"]["state_dict"][key],
        )
    for label, entry in a["models"].items():
        for key, t in entry["state_dict"].items():
            assert t.equal(b["models"][label]["state_dict"][key]), (label, key)


def test_the_pooled_recipe_trains_both_phases_with_pseudo_negatives(tmp_path):
    pool = kernel_pool()
    halves = (pool[: len(pool) // 2], pool[len(pool) // 2 :])
    dirs, entries = [], []
    for c, cell in enumerate(CELLS3[:2]):
        shapes = shapes_for(10 + c, 16, [cell])
        d = write_round(tmp_path / f"c{c}" / "stage04", shapes, halves[c])
        write_attributes(d, pool)
        dirs.append(d)
        listed = [k["sol_idx_global"] for k in halves[c]]
        entries += [
            {"m": m, "n": n, "k": k, "batch": b, "kernels": listed}
            for m, n, k, b in shapes
        ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"shapes": entries}))
    out = tmp_path / "stage05"
    pseudo = ("--pseudo-negatives", "targeted", "--unmeasured-samples", 3)
    train(tmp_path, dirs, out, *PLANNED_V3, "--plan", plan, *POOLED, *pseudo)
    cells = json.loads((out / "cells.json").read_text())["cells"]
    for cell in cells:
        assert 0 < cell["n_unmeasured_pairs"] <= 3 * cell["n_train_gemms"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["pooled"]["student"]["n_unmeasured_pairs"] == sum(
        c["n_unmeasured_pairs"] for c in cells
    )


def test_the_pooled_recipe_refuses_what_it_cannot_honour(tmp_path):
    data = _v3_round(tmp_path, n_per_cell=14, cells=CELLS3[:1])
    for i, extra in enumerate(
        (
            ("--validation-frac", 0),
            ("--boundary-margin", 0.5),
            ("--qat-frac", 0.1, "--weight-dtype", "int8"),
        )
    ):
        out = tmp_path / f"bad{i}"
        proc = train(
            tmp_path, [data], out, "--catalog", "v3", *POOLED, *extra, expect_rc=1
        )
        assert "--recipe pooled_ft" in proc.stdout + proc.stderr
        assert not (out / "models.pt").exists()


def test_cell_pairs_put_pseudo_negatives_back_where_they_were_drawn():
    pairs = s5.CellPairs(
        "c",
        2,
        (
            np.zeros((2, 1), np.float32),
            np.array([[0.0], [1.0], [2.0]], np.float32),
            np.array([[5.0], [6.0], [7.0]], np.float32),
            np.array([0, 0, 1]),
            np.array([1.0, 2.0, 4.0]),
            np.array([1.0, 1.0, 4.0]),
        ),
        None,
        (
            np.array([[9.0], [8.0], [7.5]], np.float32),
            np.array([[0.5], [0.6], [0.7]], np.float32),
            np.array([0, 1, 1]),
            np.array([1.1, 4.4, 4.4]),
            np.array([1.0, 4.0, 4.0]),
            np.array([0, 1, 3]),
            np.array([False, False, True, False, True, True]),
        ),
    )
    assert pairs.n_pseudo == 3
    (q, i, x, g, us, win), measured, mask = pairs.fit_arrays()
    assert i[:, 0].tolist() == [0.0, 1.0, 9.0, 2.0, 8.0, 7.5]
    assert x[:, 0].tolist() == pytest.approx([5.0, 6.0, 0.5, 7.0, 0.6, 0.7])
    assert g.tolist() == [0, 0, 0, 1, 1, 1]
    assert us.tolist() == pytest.approx([1.0, 2.0, 1.1, 4.0, 4.4, 4.4])
    assert win.tolist() == [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    plain = s5.CellPairs("c", 2, pairs.train, None)
    arrays, measured, mask = plain.fit_arrays()
    assert arrays is plain.train and not mask.any() and measured.tolist() == [0, 1, 2]


def test_concatenated_cells_keep_each_gemm_its_own_pairs():
    a = (np.zeros((2, 1)), *(np.zeros((3, 1)),) * 2, np.array([0, 1, 1]))
    b = (np.ones((1, 1)), *(np.ones((2, 1)),) * 2, np.array([0, 0]))
    a += (np.arange(3.0), np.arange(3.0))
    b += (np.arange(2.0), np.arange(2.0))
    q, i, x, g, us, win = s5.concat_pairs([a, b])
    assert q.shape == (3, 1) and g.tolist() == [0, 1, 1, 2, 2]
    assert us.tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_the_recipe_shapes_the_model_and_its_schedule_does_not():
    schedule = {
        "pooled_epochs",
        "pooled_batch_gemms",
        "pooled_lr",
        "pooled_eval_points",
        "pooled_n_estimators",
        "pooled_num_leaves",
        "pooled_min_data_in_leaf",
        "fine_tune_epochs",
        "fine_tune_lr",
        "pooled_seed",
        "pooled_threads",
    }
    assert schedule <= s5.NOT_MODEL_SHAPING and "recipe" not in s5.NOT_MODEL_SHAPING
    assert s5.changed_model_options({}, {"recipe": "per_leaf"}) == {}
    assert s5.changed_model_options({}, {"recipe": "pooled_ft"}) == {
        "recipe": ("per_leaf", "pooled_ft")
    }


def test_v2_training_drops_non_exclusive_gemms(tmp_path):
    data = write_round(tmp_path / "r0" / "stage04", shapes_for(0, 14, CELLS3[:1]))
    shapes = shapes_for(1, 6, CELLS3[:1])
    ctx = write_context_round(tmp_path / "ctx" / "stage04", shapes, 50, "default")
    out = tmp_path / "round_0" / "stage05"
    train(tmp_path, [data, ctx], out, "--skip-deployed-eval")
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["load_stats"]["gemms_dropped_non_exclusive"] == len(set(shapes))
    assert metrics["contexts"] is None
    bundle = load_bundle(out / "models.pt")
    assert "catalog" not in bundle and bundle["serving"]["cascade_keys"] == 0

    v3 = tmp_path / "round_1" / "stage05"
    proc = train(
        tmp_path, [data], v3, "--catalog", "v3", "--skip-deployed-eval",
        "--prior-round-dir", tmp_path / "round_0", expect_rc=1,
    )  # fmt: skip
    assert "cannot be carried forward" in proc.stdout + proc.stderr
    train(tmp_path, [data], v3, "--whitelist", "view-key", expect_rc=1)
    train(tmp_path, [data], v3, "--dot2-rule", expect_rc=1)
    train(tmp_path, [data], v3, "--launch-keys", expect_rc=1)
    train(tmp_path, [data], v3, "--non-temporal-family-rule", expect_rc=1)
    proc = train(
        tmp_path, [data], v3, "--catalog", "v3", "--non-temporal-family-rule",
        "--skip-deployed-eval", expect_rc=1,
    )  # fmt: skip
    assert "needs --non-temporal-rule" in proc.stdout + proc.stderr


def test_v3_bundles_carry_the_family_rule_and_every_trained_key(tmp_path):
    pool = kernel_pool()
    rule = ("--catalog", "v3", "--non-temporal-rule", "--non-temporal-family-rule")
    data = write_round(tmp_path / "round_0" / "stage04", shapes_for(0, 20, CELLS3[:1]))
    write_attributes(data, pool)
    train(tmp_path, [data], tmp_path / "round_0" / "stage05", *rule)
    first = load_bundle(tmp_path / "round_0" / "stage05" / "models.pt")
    assert first["serving"]["non_temporal_family_rule"]
    assert first["serving"]["trained_key_gate"] and first["trained_view_keys"]
    more = write_round(
        tmp_path / "round_1" / "stage04", shapes_for(1, 20, CELLS3[1:2]), pool=pool[:6]
    )
    write_attributes(more, pool)
    train(tmp_path, [more], tmp_path / "round_1" / "stage05", *rule,
          "--prior-round-dir", tmp_path / "round_0")  # fmt: skip
    second = load_bundle(tmp_path / "round_1" / "stage05" / "models.pt")
    assert set(first["trained_view_keys"]) <= set(second["trained_view_keys"])
    assert len(second["trained_view_keys"]) == len(set(second["trained_view_keys"]))


def test_serving_options_of_the_family_rule_and_the_trained_key_gate():
    from lib import mlrec

    plain = s5.serving_options(argparse.Namespace(catalog="v3"))
    assert plain["trained_key_gate"] is True
    assert plain["non_temporal_family_rule"] is False
    off = s5.serving_options(argparse.Namespace(catalog="v3", no_trained_key_gate=True))
    assert dict(plain, trained_key_gate=False) == off
    family = s5.serving_options(
        argparse.Namespace(
            catalog="v3", non_temporal_rule=True, non_temporal_family_rule=True
        )
    )
    flags = mlrec.v3_serving_flags(family)
    assert (
        flags & mlrec.V3_NON_TEMPORAL_FAMILY_RULE and flags & mlrec.V3_TRAINED_KEY_GATE
    )
    assert "trained_key_gate" not in s5.serving_options(
        argparse.Namespace(catalog="v2")
    )


def test_trained_view_keys_are_the_hashes_of_measured_view_keys():
    spec = (fs.AttributeFeature("param_a"),)
    row = dict(mt_m=64, mt_n=64, mt_k=32, mi_m=16, mi_n=16, mi_k=32, cache_hints_a=0,
               cache_hints_b=4, occupancy=2)  # fmt: skip
    gemms = [
        {"candidates": [dict(row, attributes={"param_a": 3}), dict(row, occupancy=1)]},
        {"candidates": [dict(row, attributes={"param_a": 3, "other": 1})]},
    ]
    got = s5.trained_view_keys(gemms, fs.Catalog("v3", spec))
    cfg = fs.config_kwargs_from_row(row)
    assert got == {
        fs.view_key_hash(fs.v3_view_key(cfg, {"param_a": 3}, spec)),
        fs.view_key_hash(fs.v3_view_key(cfg, {}, spec)),
    }


def test_launch_keys_reach_the_model_file():
    from lib import mlrec

    plain = s5.serving_options(argparse.Namespace(catalog="v3"))
    keyed = s5.serving_options(argparse.Namespace(catalog="v3", launch_keys=True))
    assert plain["launch_keys"] is False and keyed["launch_keys"] is True
    assert dict(plain, launch_keys=True) == keyed
    assert mlrec.v3_serving_flags(plain) & mlrec.V3_LAUNCH_KEYS == 0
    assert mlrec.v3_serving_flags(keyed) & mlrec.V3_LAUNCH_KEYS
    assert "launch_keys" not in s5.serving_options(argparse.Namespace(catalog="v2"))


def test_declared_contexts_and_catalog_selection(tmp_path):
    gemms = [{}, {"cu_budget": 25}, {"cu_budget": 300, "schedule": "auto"}]
    assert s5.declared_contexts(gemms, 100) == {
        "min_cu_fraction": 0.25,
        "max_cu_fraction": 1.0,
        "schedules": ["default", "auto"],
    }
    assert s5.declared_contexts([{}], 100, 0.5, "dynamic")["schedules"] == ["dynamic"]
    with pytest.raises(ValueError):
        s5.declared_contexts([{}], 100, 0.0)
    with pytest.raises(ValueError):
        s5.declared_contexts([{}], 100, None, "fast")

    import argparse

    def ns(**kw):
        return argparse.Namespace(
            **{"catalog": None, "attribute_spec": None, "arch": "gfx950", **kw}
        )

    assert s5.training_catalog(ns(), {}) == fs.V2_CATALOG
    cfg = {"train": {"catalog": "v3", "attribute_spec": [{"name": "param_a"}]}}
    assert s5.training_catalog(ns(), cfg).spec == (fs.AttributeFeature("param_a"),)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("- {name: wave_group_0, transform: log2p1}\n")
    cat = s5.training_catalog(ns(catalog="v3", attribute_spec=spec_file), {})
    assert cat.spec[0].transform == "log2p1"
    assert s5.training_catalog(ns(catalog="v3"), {}).spec == fs.DEFAULT_ATTRIBUTE_SPEC
    launch = s5.training_catalog(ns(catalog="v3"), {}).launch
    assert isinstance(launch, hipblaslt_launch.HipblasltLaunch)
    assert launch.arch == "gfx950"
    stem = "TensileLibrary_F8F8_BF8_HA_Contraction_l_Alik_Bljk_Cijk_Dijk"
    strict = ns(
        catalog="v3", arch="gfx1250-strict", library_stem=stem + "_gfx1250-strict"
    )
    assert s5.training_catalog(strict, {}).launch.arch == "gfx1250"
    assert s5.training_catalog(strict, {}).launch.num_xcd is None
    from_cfg = {"hipblaslt": {"library_stem": stem + "_gfx942"}}
    assert s5.training_catalog(ns(catalog="v3"), from_cfg).launch.arch == "gfx942"
    from_cfg["hardware"] = {"num_xcd": 2}
    assert s5.training_catalog(ns(catalog="v3"), from_cfg).launch.num_xcd == 2
    with pytest.raises(ValueError):
        s5.training_catalog(ns(catalog="v2", attribute_spec=spec_file), {})
    assert s5.uncovered_base_cells({"Large|Large|LargeK|Bnone#M<=1000": {}}) == [
        c for c in grid.all_cell_labels() if c != "Large|Large|LargeK|Bnone"
    ]


# ── training schedule, targets and checkpoints ───────────────────────────────


def test_learning_rate_schedules():
    lr = [s5.learning_rate(s, 100, 1.0, warmup_frac=0.1) for s in range(100)]
    assert lr[0] == pytest.approx(1 / 11) and lr[9] == pytest.approx(10 / 11)
    assert lr[10] == pytest.approx(1.0) and lr[-1] == pytest.approx(0.02)
    assert all(a >= b for a, b in zip(lr[10:], lr[11:]))
    step = [
        s5.learning_rate(s, 101, 1.0, schedule="step", warmup_frac=0)
        for s in range(101)
    ]
    assert step[59] == 1.0 and step[60] == pytest.approx(0.1)
    assert step[85] == pytest.approx(0.01) and step[-1] == pytest.approx(0.01)
    assert s5.learning_rate(50, 100, 3.0, schedule="constant") == 3.0
    assert s5.learning_rate(0, 1, 2.0) == 2.0
    with pytest.raises(ValueError):
        s5.learning_rate(50, 100, 1.0, schedule="linear")


def test_near_ties_share_the_top_target_and_relevance():
    us = np.asarray([10.0, 10.05, 10.2, 11.0, 30.0, 5.0, 5.2])
    win = np.asarray([10.0] * 5 + [5.0] * 2)
    gidx = np.asarray([0] * 5 + [1] * 2)
    tol = math.log1p(0.01)
    t = s5.per_gemm_softmax_targets_np(us, win, gidx, alpha=12.0, target_tolerance=tol)
    assert t[0] == t[1] > t[2] > t[3] > t[4] and t[5] > t[6]
    assert t[:5].sum() == pytest.approx(1.0) and t[5:].sum() == pytest.approx(1.0)
    ref = np.exp(-12.0 * np.maximum(np.log(us / win) - tol, 0.0))
    for g in (0, 1):
        np.testing.assert_allclose(t[gidx == g], ref[gidx == g] / ref[gidx == g].sum())
    rel = s5.relevance_labels(us, win, 0.01)
    assert list(rel) == [6, 6, 5, 4, 0, 6, 5]
    old_bins = 5 - np.searchsorted(
        [0.05, 0.10, 0.20, 0.40, 0.80], np.log(us / win), "right"
    )
    assert list(s5.relevance_labels(us, win, math.expm1(0.05))) == list(old_bins)
    assert s5.config_tie_tolerance({"stage08": {"tie_tolerance": 0.016}}) == 0.016
    assert s5.config_tie_tolerance({}) == s5.DEFAULT_TIE_TOLERANCE


def test_fake_quantization_rounds_like_the_model_file():
    import torch

    from lib import mlrec

    w = torch.randn(7, 5, generator=torch.Generator().manual_seed(3)) * 0.3
    for dtype in ("int8", "int4", "bf16"):
        code = mlrec.WEIGHT_DTYPES[dtype]
        stored = mlrec.decode_weight(
            mlrec.encode_weight(w.numpy(), code), w.numel(), code
        )
        q = s5.fake_quant_weight(w, dtype)
        np.testing.assert_allclose(q.numpy().reshape(-1), stored, rtol=1e-6, atol=1e-7)
    w.requires_grad_(True)
    s5.fake_quant_weight(w, "int4").sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))
    assert s5.fake_quant_weight(w, "fp32") is w


def test_temperature_is_log_parameterized_and_exported_as_the_engine_reads_it():
    import torch

    torch.manual_seed(0)
    s = s5.GenericTwoTower(4, 3, 2, 8, 16, 8)
    with torch.no_grad():
        s.log_temperature.fill_(math.log(0.5))
    q, i, x = torch.randn(6, 4), torch.randn(6, 3), torch.randn(6, 2)
    state = s.export_state()
    assert "log_temperature" not in state
    assert float(state["temperature"][0]) == pytest.approx(0.5)
    other = s5.GenericTwoTower(4, 3, 2, 8, 16, 8)
    other.load_exported(state)
    assert torch.equal(other.score_pairs(q, i, x), s.score_pairs(q, i, x))
    for stored, used in ((-0.5, 0.5), (0.01, s5.MIN_TEMPERATURE)):
        other.load_exported(dict(state, temperature=torch.tensor([stored])))
        assert float(other.temperature().detach()) == pytest.approx(used)
    old = dict(state)
    del old["temperature"]
    other.load_exported(old)
    assert float(other.temperature().detach()) == pytest.approx(1.0)
    grouped = s.score_grouped(
        q[:2], torch.tensor([0, 0, 1, 1, 0, 1]), i, torch.arange(6), x
    )
    rows = s.score_pairs(q[[0, 0, 1, 1, 0, 1]], i, x)
    torch.testing.assert_close(grouped, rows)
    fixed = s5.GenericTwoTower(4, 3, 2, 8, 16, 8, learn_temperature=False)
    assert not fixed.log_temperature.requires_grad


def _reference_greedy(sig_to_data, winner_us, k):
    """The pure-Python whitelist greedy `_greedy_table` replaces."""
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    chosen, best, pick = [], {gi: -math.inf for gi in winner_us}, {}
    pick = {gi: math.inf for gi in winner_us}

    def take(sig):
        chosen.append(sig)
        for gi, (us, score) in sig_to_data[sig].items():
            if score > best[gi]:
                best[gi], pick[gi] = score, us

    while len(chosen) < k and any(p == math.inf for p in pick.values()):
        top, top_cover, top_gain = None, 0, -math.inf
        for sig, data in sig_to_data.items():
            if sig in chosen:
                continue
            cover, gain = 0, 0.0
            for gi in winner_us:
                c = data.get(gi)
                if c is None:
                    continue
                if pick[gi] == math.inf:
                    cover += 1
                    gain += log_win[gi] - math.log(c[0])
                elif c[1] > best[gi]:
                    gain += math.log(pick[gi]) - math.log(c[0])
            if cover and (
                cover > top_cover or (cover == top_cover and gain > top_gain)
            ):
                top, top_cover, top_gain = sig, cover, gain
        if top is None:
            break
        take(top)
    while len(chosen) < k:
        top, top_obj = None, -math.inf
        for sig, data in sig_to_data.items():
            if sig in chosen:
                continue
            total, count = 0.0, 0
            for gi in winner_us:
                c = data.get(gi)
                new = c[0] if c is not None and c[1] > best[gi] else pick[gi]
                if new == math.inf:
                    continue
                total += log_win[gi] - math.log(new)
                count += 1
            if count and total / count > top_obj:
                top, top_obj = sig, total / count
        if top is None:
            break
        take(top)
    return chosen


def test_vectorized_whitelist_greedy_matches_the_reference():
    rng = random.Random(4)
    for trial in range(30):
        n_sig, n_g = rng.randint(1, 12), rng.randint(1, 15)
        winner = {gi: rng.uniform(1, 10) for gi in range(n_g) if rng.random() < 0.9}
        cache = {}
        for s in range(n_sig):
            data = {
                gi: (w * rng.uniform(1.0, 1.6), rng.gauss(0, 1))
                for gi, w in winner.items()
                if rng.random() < 0.5
            }
            if data:
                cache[(trial, s)] = data
        table = s5.ScoreTable.from_cache(cache, winner)
        for k in (1, 3, 8, 20):
            got = [table.keys[j] for j in s5._greedy_table(table, k)]
            assert got == _reference_greedy(cache, winner, k), (trial, k)


def test_the_key_index_matches_the_student_score_cache(tmp_path):
    import torch

    pool = twin_pool()
    data = write_round(tmp_path / "stage04", shapes_for(12, 8, CELLS3[:2]), pool)
    gemms = [
        g for gs in s5.load_enriched_chunks(str(data), True)[0].values() for g in gs
    ]
    names = (fs.query_feature_names(), fs.item_feature_names())
    names += (fs.interaction_feature_names(),)
    hw, consts = hwlib.DeviceHardware(**HARDWARE), hwlib.arch_constants("gfx950")
    q, i, x, gidx, us, win, _rank = s5.build_cell_tensors(gemms, *names, hw, consts)
    qw, iw, xw, norms = s5.whiten(q, i, x)
    torch.manual_seed(2)
    student = s5.GenericTwoTower(len(names[0]), len(names[1]), len(names[2]), 8, 16, 8)
    nt = {k: (False, True) for k in {s5.ev.pool_key(g) for g in gemms}}
    rules = {"non_temporal_rule": True}
    ref = s5.ScoreTable.from_cache(
        *s5._student_score_cache(
            gemms, student, *names, hw, consts, norms, "cpu", nt, rules=rules
        )
    )
    pairs = s5._PairSet((qw, iw, xw, gidx, us, win), "cpu")
    student.eval()
    with torch.no_grad():
        scores = pairs.scores(student).numpy()
    index = s5.KeyIndex(gemms, lds_bytes=hw.lds_bytes, nt_flags=nt, rules=rules)
    table = index.table(scores)
    assert table.keys == ref.keys and table.gemms == ref.gemms
    np.testing.assert_array_equal(table.us, ref.us)
    np.testing.assert_allclose(table.score, ref.score, rtol=1e-5, atol=1e-6)
    sk = s5.select_smart_k_signatures(gemms, 4, table=table, k_max=6, k_min=2)
    assert (
        sk["sigs"]
        == s5.select_smart_k_signatures(gemms, 4, table=ref, k_max=6, k_min=2)["sigs"]
    )
    assert sk["metrics"]["mode"] == "student_aware" and 2 <= len(sk["sigs"]) <= 6


def _cell_kwargs(**over):
    names = (fs.query_feature_names(), fs.item_feature_names())
    names += (fs.interaction_feature_names(),)
    kw = dict(
        cell=CELLS3[0],
        q_names=names[0],
        i_names=names[1],
        x_names=names[2],
        hardware=hwlib.DeviceHardware(**HARDWARE),
        constants=hwlib.arch_constants("gfx950"),
        epochs=4,
        lr=2e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        lam=0.5,
        T=2.0,
        alpha=12.0,
        tie_tolerance=0.01,
        n_estimators=8,
        num_leaves=7,
        lgb_lr=0.1,
        seed=3,
        device="cpu",
        eval_sample_frac=1.0,
        use_teacher=False,
        quiet=True,
        training_log_writer=None,
        validation_frac=0.25,
        split_seed=5,
        batch_gemms=4,
        eval_points=8,
    )
    kw.update(over)
    return kw


def test_the_best_deployed_validation_checkpoint_is_kept(tmp_path, monkeypatch):
    data = write_round(tmp_path / "stage04", shapes_for(13, 24, CELLS3[:1]))
    (gemms,) = s5.load_enriched_chunks(str(data), True)[0].values()
    measured = iter([0.90, 0.95, 0.93, 0.95, 0.91] + [0.5] * 100)
    monkeypatch.setattr(
        s5.ShippedSelector, "measures_validation", property(lambda s: True)
    )
    monkeypatch.setattr(
        s5.ShippedSelector, "validation_sel_eff", lambda self, entry: next(measured)
    )
    selector = {"engine": False, "whitelist": {"mode": "signature", "k": 3}}
    r = s5.train_cell(gemms=gemms, selector=selector, **_cell_kwargs())
    steps = [m["step"] for m in r["epoch_metrics"]]
    seen = [m["val_deployed_sel_eff"] for m in r["epoch_metrics"]]
    assert seen[:5] == [0.90, 0.95, 0.93, 0.95, 0.91]
    tied = [m for m in r["epoch_metrics"] if m["val_deployed_sel_eff"] == 0.95]
    kept = min(tied, key=lambda m: m["val_loss"])
    assert (
        r["best_step"] == kept["step"] and r["selection_deployed_val_sel_eff"] == 0.95
    )
    assert r["selection_metric"] == "deployed_validation"
    assert r["steps"] == steps[-1] and len(steps) == 8 and not r["stopped_early"]
    assert r["smart_k_signatures"] and r["smart_k_metrics"]["mode"] == "student_aware"

    measured = iter([0.9, 0.8, 0.8, 0.8] + [0.99] * 100)
    r = s5.train_cell(gemms=gemms, selector=selector, **_cell_kwargs(patience=3))
    assert r["stopped_early"] and len(r["epoch_metrics"]) == 4
    assert r["best_step"] == r["epoch_metrics"][0]["step"]


def test_deployed_numbers_charge_picks_by_the_configured_rule(monkeypatch):
    from lib import evaluate as ev

    class Stop(Exception):
        pass

    seen = []

    def evaluate_bundle(bundle, gemms, **kw):
        seen.append(kw.get("pick_measurement"))
        raise Stop

    monkeypatch.setattr(s5.ev, "evaluate_bundle", evaluate_bundle)
    monkeypatch.setattr(s5, "cell_bundle", lambda *a, **k: {})
    monkeypatch.setattr(s5, "model_entry", lambda r: r)
    selector = SimpleNamespace(
        label="c",
        names=([], [], []),
        spec={"arch": "gfx950", "weight_dtype": "int4", "pick_measurement": "timed"},
        val_g=[],
        constants=None,
        hardware=None,
        tw=None,
    )
    with pytest.raises(Stop):
        s5.ShippedSelector.validation_sel_eff(selector, {})
    for rule in ev.PICK_MEASUREMENTS:
        with pytest.raises(Stop):
            s5.deployed_eval_cell(
                "c",
                {},
                [],
                [],
                names=([], [], []),
                arch="gfx950",
                constants=None,
                hardware=None,
                weight_dtype="int4",
                pools={},
                tw=None,
                pick_measurement=rule,
            )
    assert seen == ["timed", *ev.PICK_MEASUREMENTS]


def test_quantization_aware_phase_continues_from_the_kept_checkpoint(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(14, 20, CELLS3[:1]))
    (gemms,) = s5.load_enriched_chunks(str(data), True)[0].values()
    kw = _cell_kwargs(weight_dtype="int4", qat_frac=0.5, eval_points=4)
    r = s5.train_cell(gemms=gemms, selector=None, **kw)
    phases = [m["phase"] for m in r["epoch_metrics"]]
    assert phases.count("main") == 4 and phases.count("qat") >= 1
    assert r["steps"] > max(
        m["step"] for m in r["epoch_metrics"] if m["phase"] == "main"
    )
    assert r["selection_metric"] == "validation"
    assert "smart_k_signatures" not in r
    full = s5.train_cell(gemms=gemms, selector=None, **_cell_kwargs(batch_gemms=0))
    assert full["steps"] == 4 and len(full["epoch_metrics"]) == 4


def test_bench_launch_merges_the_launch_summaries_of_the_enriched_dirs(tmp_path):
    from lib import launch_settings as ls

    a, b = tmp_path / "round_0" / "stage04", tmp_path / "round_1" / "stage04"
    for d in (a, b):
        d.mkdir(parents=True)
    rec = {"overrides": {}, "heuristic": {"digest": "sha256:aa"}, "workspace_bytes": 8}
    (a / "enrich_summary.json").write_text(
        json.dumps({"gemms": 3, "launch": ls.summarize([rec])})
    )
    (b / "enrich_summary.json").write_text(json.dumps({"gemms": 2}))
    got = s5.bench_launch([a, b, tmp_path / "missing"])
    assert got == {
        "heuristics": ["sha256:aa"],
        "overrides": {},
        "workspace_bytes": [8],
        "unknown": 2,
    }
