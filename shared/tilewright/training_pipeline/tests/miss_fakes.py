# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GEMMs whose deployed picks miss, on the fake hipBLASLt build.

The fake library holds `fake_hipblaslt.kernels_grid()`; a GEMM's rows time
every kernel with the fake bench's cost model, so a re-bench of a pick
measures exactly the latency the rows would have given it. A test bundle
whitelists only PICK, so the model always picks it, and the rows make that
pick a miss: a skip row with a warm-up WARMUP_FACTOR times its latency
("screened"), or no row with the GEMM benched from a list without it
("not_benched")."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

import fake_hipblaslt as fh
from test_stage05_train import COLUMNS

ARCH = "gfx950"
STEM = "TensileLibrary_test_gfx950"
PICK = 2
PICK_SIGNATURE = [128, 128, 64, 16, 16, 128, 0, 0]
N_CU = 64
WARMUP_FACTOR = 5.0
CONTEXT_COLUMNS = list(COLUMNS) + ["cu_budget", "schedule"]
Shape = Tuple[int, int, int, int]


def build(tmp: Path) -> Path:
    return fh.write_fake_build(Path(tmp) / "build", ARCH, STEM)


def library_dir(root: Path) -> Path:
    return Path(root) / "Tensile" / "library" / ARCH


def bench_binary(root: Path) -> Path:
    return Path(root) / "clients" / "hipblaslt-bench"


def env(root: Path) -> Dict[str, str]:
    return {"FAKE_LIBRARY_DIR": str(library_dir(root)), "FAKE_LIBRARY_STEM": STEM}


def config_dict(**sections: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "config_id": "miss_rebench_test",
        "arch": ARCH,
        "hipblaslt": {
            "transA": "T",
            "transB": "N",
            "a_type": "bf16_r",
            "b_type": "bf16_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "library_stem": STEM,
        },
        "hardware": {"n_cu": N_CU, "lds_bytes": 65536, "l2_bytes": 4194304},
        "deploy": {"weight_dtype": "bf16"},
    }
    cfg.update(sections)
    return cfg


def config(path: Path, **sections: Any) -> Path:
    Path(path).write_text(yaml.safe_dump(config_dict(**sections)))
    return Path(path)


def times(shape: Shape, width: int = N_CU) -> Dict[int, float]:
    """{solution index: the fake bench's latency} of every library kernel."""
    return {
        int(s["index"]): fh.kernel_time_us(*shape, s["sizeMapping"], width)
        for s in fh.kernels_grid()
    }


def true_sel_eff(shape: Shape, width: int = N_CU, pick: int = PICK) -> float:
    t = times(shape, width)
    return min(t.values()) / t[pick]


def near_pick_shapes(n: int, floor: float, width: int = N_CU) -> List[Shape]:
    """`n` shapes of one cell on which PICK is within `floor` of the
    fastest kernel but not the fastest."""
    out = []
    for m in range(640, 4096, 64):
        for nn in (768, 1024, 1536, 2048):
            s = (m, nn, 1024, 1)
            se = true_sel_eff(s, width)
            if floor <= se < 1.0:
                out.append(s)
            if len(out) == n:
                return out
    raise AssertionError("not enough shapes")


def rows(
    shape: Shape,
    *,
    miss: str = "screened",
    width: int = N_CU,
    cu_budget: Optional[int] = None,
    schedule: str = "default",
    pick: int = PICK,
) -> List[Dict[str, Any]]:
    """Enriched rows of one GEMM: every kernel timed, `pick` a skip row with
    an inflated warm-up ("screened"), absent ("not_benched") or timed
    ("none")."""
    from lib.dat import kernel_dat_info

    t = times(shape, width)
    if miss == "not_benched":
        t.pop(pick)
    kernels = {int(s["index"]): kernel_dat_info(s) for s in fh.kernels_grid()}
    order = sorted(t, key=lambda s: (t[s], s))
    out = []
    for rank, sid in enumerate(order):
        skip = miss == "screened" and sid == pick
        row = dict.fromkeys(CONTEXT_COLUMNS, "")
        m, n, k, b = shape
        row.update(
            is_winner="True" if rank == 0 else "False",
            is_skip="True" if skip else "False",
            is_origami_pick=1 if rank == len(order) - 1 else 0,
            rank=rank,
            sol_idx_global=sid,
            sol_idx_local=sid,
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
            us=f"{t[sid] * (WARMUP_FACTOR if skip else 1.0):.4f}",
        )
        if cu_budget is not None:
            row.update(cu_budget=cu_budget, schedule=schedule)
        row.update({f: kernels[sid][f] for f in CONFIG_FIELDS})
        out.append(row)
    return out


CONFIG_FIELDS = (
    "mt_m mt_n mt_k mi_m mi_n mi_k occupancy cache_hints_a cache_hints_b "
    "grvw_a grvw_b gwvw_d"
).split()


def write_chunks(out_dir: Path, all_rows: Iterable[Mapping[str, Any]]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONTEXT_COLUMNS)
        w.writeheader()
        w.writerows(all_rows)
    return out_dir


def write_plan(path: Path, shapes: Sequence[Shape], exclude: int = PICK) -> Path:
    """A plan that benched `shapes` from a list of every kernel but
    `exclude`."""
    kernels = [s for s in times(shapes[0]) if s != exclude]
    doc = {
        "shapes": [
            {"m": m, "n": n, "k": k, "batch": b, "kernels": kernels}
            for m, n, k, b in shapes
        ]
    }
    Path(path).write_text(json.dumps(doc))
    return Path(path)


def gemm(shape: Shape, **kw: Any) -> Dict[str, Any]:
    """The GEMM dict `rows` describes (stage05's loader format, enough for
    lib/evaluate's measurements and lib/miss_rebench)."""
    rs = rows(shape, **kw)
    m, n, k, b = shape
    g: Dict[str, Any] = {
        "m": m,
        "n": n,
        "k": k,
        "batch_count": b,
        "transA": "T",
        "transB": "N",
        "candidates": [
            {
                "sol_idx_global": int(r["sol_idx_global"]),
                "us": float(r["us"]),
                "is_skip": r["is_skip"] == "True",
            }
            for r in rs
        ],
    }
    if kw.get("cu_budget") is not None:
        g["cu_budget"] = kw["cu_budget"]
        g["schedule"] = kw.get("schedule", "default")
    timed = [c["us"] for c in g["candidates"] if not c["is_skip"]]
    g["winner_us"] = min(timed)
    return g


def bundle(train_dir: Path, shapes: Sequence[Shape], seed: int = 0) -> Path:
    """A random-weight bundle with a cell for every shape's cell, whitelisting
    only PICK."""
    from conftest import write_bundle
    from lib.grid import cell_key

    labels = sorted({cell_key(*s) for s in shapes})
    return write_bundle(
        Path(train_dir),
        labels,
        seed=seed,
        signatures={lb: [PICK_SIGNATURE] for lb in labels},
    )
