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

import stage05_train as s5
from lib import features as fs

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
    for entry in bundle["models"].values():
        assert entry["smart_k_signatures"] and len(entry["smart_k_signatures"][0]) == 8
        assert "q_proj.0.weight" in entry["state_dict"]

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
