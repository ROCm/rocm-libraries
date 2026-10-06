# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage05 on synthetic enriched data (also used by the stage04b / stage08
tests): a kernel pool, a deterministic latency model, and chunk_*.csv files
with the stage04 column schema."""

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

import numpy as np
import pytest

import bench_fakes as bf
import stage05_train as s5
from lib import dat, feasibility
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
        "arch: gfx950\nhardware:\n"
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
        assert entry["feature_norms"]["i_std"][nta] == 0.0

    cells = {
        c["label"]: c for c in json.loads((out / "cells.json").read_text())["cells"]
    }
    for label, c in cells.items():
        assert c["seed"] == s5.cell_seed(5, label)
        assert c["n_val_gemms"] > 0 and c["selection_set"] == "validation"
        assert c["n_train_gemms"] + c["n_val_gemms"] == c["n_gemms"] == 24
        assert c["origami"]["n_eval"] == c["n_train_gemms"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["environment"]["artifact_knobs"]) == set(s5.ARTIFACT_ENV_KNOBS)
    assert metrics["hardware"] == HARDWARE and metrics["weight_dtype"] == "int4"
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
    train(
        tmp_path,
        [data],
        par,
        "--skip-deployed-eval",
        "--train-workers",
        2,
        "--train-threads-per-worker",
        1,
    )
    a = load_bundle(full / "models.pt")["models"]
    b = load_bundle(alone / "models.pt")["models"]
    c = load_bundle(par / "models.pt")["models"]
    assert list(b) == [CELLS3[2]]
    for other in (b, c):
        for label, entry in other.items():
            for name, t in entry["state_dict"].items():
                assert t.equal(a[label]["state_dict"][name]), (label, name)
    serial_log = (full / "training_log.csv").read_text()
    assert len(serial_log.splitlines()) == 1 + 6 * len(CELLS3)
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
