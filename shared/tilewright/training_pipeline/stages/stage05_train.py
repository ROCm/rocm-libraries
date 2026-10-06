#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage05_train -- per-cell distillation trainer of the kernel recommender.

Input : one or more folders of enriched chunk_*.csv from stage04. Repeating
        --enriched-csv-dir trains on their union (cross-round augmentation).
Output: <out>/models.pt         per-cell student weights, feature statistics
                                and kernel whitelists (what stage06 deploys)
        <out>/cells.json        per-cell metadata and measurements
        <out>/metrics.json      global measurements, arguments, environment
        <out>/training_log.csv  per-epoch loss and selection efficiency

Every GEMM routes to a leaf of the 96-cell grid plus the cumulative split
tree; each leaf with at least --min-cell-gemms GEMMs is trained:

  1. Validation split: a GEMM is a validation GEMM when a hash of (--seed,
     GEMM identity) falls below --validation-frac, so it keeps its side
     across rounds and cells. With no validation GEMMs (or
     --validation-frac 0) training and epoch selection use every GEMM.
  2. LightGBM LambdaRank teacher on the training GEMMs.
  3. GenericTwoTower student, full batch for --epochs:
         loss = lam * KL(teacher_T || student_T) * T^2 + (1 - lam) * soft_CE
     The epoch with the best validation selection efficiency is kept.
  4. Kernel whitelist ("smart_K") chosen greedily with the student in the
     loop on the training GEMMs, through the engine's LDS gate and
     feasibility rule with non-temporal availability of the library pool.
  5. Deployed evaluation through the engine (lib/evaluate.py) at
     --weight-dtype on training and validation GEMMs; the Origami baseline is
     the bench's Origami pick (`is_origami_pick`).

Selection efficiency (sel_eff) = winner_us / pick_us, geometric mean over
GEMMs; 1.0 picks the fastest benched kernel every time. The validation GEMMs
also choose the epoch and the whitelist fallback, so the validation numbers
here are model-selection scores; stage04b measures on held-out shapes.

Every measured solution of a GEMM is one candidate (one training pair) with
its own latency and target. Solutions with identical kernel parameters get
identical features and scores, and the engine ranks the first of them in
pool order first; each GEMM's candidates are kept in pool order so that the
training-time picks break such ties the same way. Whitelists stay per kernel
signature.

Cells this round does not train keep their model from the latest
--prior-round-dir. A split parent's model stays in the bundle while a leaf
under it has no model of its own (`fallback_parents`): the engine serves such
a leaf with its nearest trained ancestor.

Exit status: 0; 3 after every output is written when a cell failed (training
or deployed evaluation) or an --only-cells label was not trained (too few or
no GEMMs); 1 on setup errors.

    python3 stage05_train.py --arch gfx950 --config-yaml <config> \\
        --enriched-csv-dir <run>/round_0/stage04 \\
        --output-dir <run>/round_0/stage05
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import lightgbm as lgb

    _LGB_OK = True
except ImportError:
    _LGB_OK = False
    lgb = None  # type: ignore

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(THIS_DIR)
if PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, PIPELINE_ROOT)

from lib import evaluate as ev  # noqa: E402
from lib import feasibility  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import grid  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib.dat import KERNELS_FILE, read_kernel_attributes, sig_from_row  # noqa: E402

EXIT_CELLS_NOT_TRAINED = 3

# Environment knobs that change the trained artifact; metrics.json records
# their values.
ARTIFACT_ENV_KNOBS: Tuple[str, ...] = (
    "ML_CAPACITY_MODE",
    "ML_FORCE_CAPACITY",
    "ML_LARGE_CELL_CAPACITY",
    "ML_DROPOUT",
    "ML_SMARTK_RUNTIME_WEIGHT",
)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [jsonable(v) for v in sorted(obj)]
    if isinstance(obj, np.ndarray):
        return None
    return obj


def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "    n/a"
    return f"{v * 100:6.2f}%"


def cell_seed(seed: int, label: str) -> int:
    """Per-cell seed derived from the cell label, independent of which other
    cells are trained and of their order."""
    h = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def _split_unit(seed: int, gemm: Mapping[str, Any]) -> float:
    key = "|".join(str(v) for v in ev.gemm_key(gemm))
    h = hashlib.sha256(f"{int(seed)}|{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64)


def split_validation(
    gemms: Sequence[Dict[str, Any]], frac: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(training, validation) GEMMs. Both sides must be non-empty, otherwise
    every GEMM is a training GEMM."""
    if frac <= 0:
        return list(gemms), []
    val = [g for g in gemms if _split_unit(seed, g) < frac]
    train = [g for g in gemms if _split_unit(seed, g) >= frac]
    if not val or not train:
        return list(gemms), []
    return train, val


# ── data loading ──────────────────────────────────────────────────────────────


def _flag(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def _chunk_files(folder) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(folders, (folder, chunk file name) pairs) of one folder or a list."""
    folders: List[str] = (
        [folder] if isinstance(folder, (str, os.PathLike)) else list(folder)
    )
    chunk_files: List[Tuple[str, str]] = []
    for fd in folders:
        fd = str(fd)
        if not os.path.isdir(fd):
            continue
        for p in sorted(os.listdir(fd)):
            if p.startswith("chunk_") and p.endswith(".csv"):
                chunk_files.append((fd, p))
    return folders, chunk_files


def _solution_id(row: Mapping[str, Any]) -> Optional[int]:
    """`sol_idx_global`, else the synthetic id -(rank + 1), else None."""
    rank_val = int(row.get("rank", -1) or -1)
    sig = row.get("sol_idx_global", "")
    try:
        sig_int = int(sig) if sig not in ("", None) else None
    except (TypeError, ValueError):
        sig_int = None
    if sig_int is None:
        if rank_val < 0:
            return None
        sig_int = -(rank_val + 1)
    return sig_int


def _row_rejection(row: Mapping[str, Any]) -> Optional[str]:
    """The load counter of a row that does not become a candidate, or None."""
    try:
        us = float(row["us"])
    except (KeyError, TypeError, ValueError):
        return "bad_us"
    if us <= 0 or math.isnan(us):
        return "bad_us"
    try:
        mt_m_check = int(row.get("mt_m", 0) or 0)
        mi_m_check = int(row.get("mi_m", 0) or 0)
    except (TypeError, ValueError):
        mt_m_check, mi_m_check = 0, 0
    if mt_m_check == 0 or mi_m_check == 0:
        return "bad_kernel_params"
    if _solution_id(row) is None:
        return "skip_no_rank"
    return None


def _gemm_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(row["m"]),
        int(row["n"]),
        int(row["k"]),
        int(row["batch_count"]),
        row["transA"],
        row["transB"],
        row["a_type"],
        row["b_type"],
        row["c_type"],
        row["d_type"],
        row["compute_type"],
    )


def _leaf_of(
    m: int, n: int, k: int, b: int, split_tree: Optional[Dict[str, Any]]
) -> str:
    base = grid.cell_key(m, n, k, b)
    return sc.assign_subcell(base, m, n, k, b, split_tree) if split_tree else base


def gemm_shapes_by_leaf(
    folder, split_tree: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """{leaf: [(m, n, k, batch)]} of the GEMMs `load_enriched_chunks` builds
    from the same folders and split tree, without reading candidates. Missing
    folders or chunk files count nothing."""
    _folders, chunk_files = _chunk_files(folder)
    keys = set()
    for fd, fname in chunk_files:
        with open(os.path.join(fd, fname)) as f:
            for row in csv.DictReader(f):
                if _row_rejection(row) is None:
                    keys.add(_gemm_key(row))
    out: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for key in sorted(keys):
        m, n, k, b = (int(v) for v in key[:4])
        out[_leaf_of(m, n, k, b, split_tree)].append((m, n, k, b))
    return dict(out)


def load_enriched_chunks(
    folder, quiet: bool, split_tree: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, List[dict]], dict]:
    """Read chunk_*.csv files of one folder or a list of folders (their union)
    and group the rows by GEMM, then by leaf cell (`split_tree` routes base
    cells to sub-cells).

    Tested and skip rows both become candidates, one per solution: solutions
    with identical kernel parameters stay separate candidates. A row without
    a solution index gets the synthetic id -(rank + 1). Rows of one solution
    index of a GEMM (several rounds, a retried problem) keep the fastest; the
    GEMM's winner is its fastest tested solution (skip-row latencies come
    from one warm-up run). `is_origami_pick` is None when the CSV has no such
    column. A candidate carries the kernel's `attributes` when a folder's
    kernels.json lists its solution index.

    Returns ({cell: [gemm dict]}, counters)."""
    t0 = time.time()
    folders, chunk_files = _chunk_files(folder)
    if not chunk_files:
        raise SystemExit(f"no chunk_*.csv in any of {folders}")

    stats: Dict[str, int] = defaultdict(int)
    stats["n_source_dirs"] = len(folders)
    attributes: Dict[int, Dict[str, int]] = {}
    for fd in folders:
        path = os.path.join(str(fd), KERNELS_FILE)
        if not os.path.isfile(path):
            continue
        try:
            listed = read_kernel_attributes(path)
        except ValueError as e:
            ui.warn("load", f"{e}; its kernels carry no attributes")
            continue
        for sid, attrs in listed.items():
            if attributes.setdefault(sid, attrs) != attrs:
                stats["kernel_attribute_conflicts"] += 1
    if stats["kernel_attribute_conflicts"]:
        ui.warn(
            "load",
            f"{stats['kernel_attribute_conflicts']} solution(s) have different "
            f"attributes in different {KERNELS_FILE} files; the first is used",
        )
    gemms_by_key: Dict[Tuple, dict] = {}
    for fi, (fd, fname) in enumerate(chunk_files, 1):
        with open(os.path.join(fd, fname)) as f:
            reader = csv.DictReader(f)
            has_origami_col = "is_origami_pick" in (reader.fieldnames or [])
            for row in reader:
                stats["rows_total"] += 1
                rejected = _row_rejection(row)
                if rejected is not None:
                    stats[rejected] += 1
                    continue
                is_skip = row.get("is_skip", "False") == "True"
                us = float(row["us"])
                rank_val = int(row.get("rank", -1) or -1)
                sig_int = _solution_id(row)
                if is_skip:
                    stats["skip_rows_used"] += 1
                else:
                    stats["tested_rows"] += 1
                key = _gemm_key(row)
                g = gemms_by_key.get(key)
                if g is None:
                    g = {
                        "m": key[0],
                        "n": key[1],
                        "k": key[2],
                        "batch_count": key[3],
                        "transA": key[4],
                        "transB": key[5],
                        "a_type": key[6],
                        "b_type": key[7],
                        "c_type": key[8],
                        "d_type": key[9],
                        "compute_type": key[10],
                        "candidates": [],
                    }
                    gemms_by_key[key] = g
                try:
                    sol_idx_local_val = (
                        int(row["sol_idx_local"])
                        if row.get("sol_idx_local") not in ("", None)
                        else -1
                    )
                except (TypeError, ValueError):
                    sol_idx_local_val = -1
                cand = {
                    "sol_idx_global": sig_int,
                    "sol_idx_local": sol_idx_local_val,
                    "us": us,
                    "rank": rank_val,
                    "is_winner": row.get("is_winner", "False") == "True",
                    "is_skip": is_skip,
                    "is_origami_pick": (
                        _flag(row.get("is_origami_pick", ""))
                        if has_origami_col
                        else None
                    ),
                }
                cand.update(fs.config_kwargs_from_row(row))
                if sig_int in attributes:
                    cand["attributes"] = attributes[sig_int]
                g["candidates"].append(cand)
        if not quiet and fi % 200 == 0:
            print(
                f"  [load] {fi:>4d}/{len(chunk_files)}  "
                f"rows={ui.fmt_int(stats['rows_total']):>11s}  "
                f"gemms={ui.fmt_int(len(gemms_by_key)):>8s}  "
                f"({ui.fmt_dur(time.time() - t0)})",
                flush=True,
            )

    if not quiet:
        ui.ok(
            "load",
            f"chunks: {len(chunk_files)} files  "
            f"rows={ui.fmt_int(stats['rows_total'])}  "
            f"tested={ui.fmt_int(stats['tested_rows'])}  "
            f"skip-used={ui.fmt_int(stats['skip_rows_used'])}  "
            f"bad_us={ui.fmt_int(stats['bad_us'])}  "
            f"bad_kp={ui.fmt_int(stats['bad_kernel_params'])}  "
            f"({ui.fmt_dur(time.time() - t0)})",
        )

    for g in gemms_by_key.values():
        by_idx: Dict[int, dict] = {}
        for c in g["candidates"]:
            sid = c["sol_idx_global"]
            prev = by_idx.get(sid)
            if prev is None or c["us"] < prev["us"]:
                if prev is not None and prev["is_origami_pick"]:
                    c["is_origami_pick"] = True
                by_idx[sid] = c
            elif c["is_origami_pick"]:
                prev["is_origami_pick"] = True
        g["candidates"] = list(by_idx.values())
        tested_us = [c["us"] for c in g["candidates"] if not c.get("is_skip", False)]
        g["winner_us"] = (
            min(tested_us) if tested_us else min(c["us"] for c in g["candidates"])
        )
        for r, c in enumerate(sorted(g["candidates"], key=lambda x: x["us"])):
            c["rank_us"] = r

    cells: Dict[str, List[dict]] = defaultdict(list)
    for g in gemms_by_key.values():
        label = _leaf_of(
            int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"]), split_tree
        )
        g["cell"] = label
        cells[label].append(g)

    stats["gemms"] = len(gemms_by_key)
    stats["cells_populated"] = len(cells)
    if not quiet:
        ui.ok(
            "load",
            f"GEMMs: {ui.fmt_int(stats['gemms'])}  "
            f"populated cells: {stats['cells_populated']}",
        )
    return cells, dict(stats)


# ── student model ─────────────────────────────────────────────────────────────

if _TORCH_OK:

    class GenericTwoTower(nn.Module):
        """score = dot(q_proj(query), i_proj(item)) / max(|T|, 0.1)
        + inter_mlp(interaction). Tensor names (q_proj.0/2/4, i_proj.0/2,
        inter_mlp.0/2, temperature) are the model file's weight order, so
        dropout is functional and never a module."""

        def __init__(
            self,
            q_dim: int,
            i_dim: int,
            x_dim: int,
            embed_dim: int = 32,
            hidden_dim: int = 64,
            inter_hidden: int = 16,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            def mlp(d_in: int, hid: int, d_out: int, depth: int = 3) -> nn.Sequential:
                layers: List[nn.Module] = []
                d = d_in
                for _ in range(depth - 1):
                    layers += [nn.Linear(d, hid), nn.ReLU()]
                    d = hid
                layers.append(nn.Linear(d, d_out))
                return nn.Sequential(*layers)

            self.q_proj = mlp(q_dim, hidden_dim, embed_dim, depth=3)
            self.i_proj = mlp(i_dim, hidden_dim, embed_dim, depth=2)
            self.inter_mlp = nn.Sequential(
                nn.Linear(x_dim, inter_hidden), nn.ReLU(), nn.Linear(inter_hidden, 1)
            )
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.dropout_p = float(dropout)

        def score_pairs(
            self,
            q_per_pair: torch.Tensor,
            i_per_pair: torch.Tensor,
            x_per_pair: torch.Tensor,
        ) -> torch.Tensor:
            """Inputs broadcast to (N_pairs, dim); returns (N_pairs,) scores."""
            eg = self.q_proj(q_per_pair)
            et = self.i_proj(i_per_pair)
            xin = x_per_pair
            p = getattr(self, "dropout_p", 0.0)
            if p > 0 and self.training:
                import torch.nn.functional as _F

                eg = _F.dropout(eg, p=p, training=True)
                et = _F.dropout(et, p=p, training=True)
                xin = _F.dropout(xin, p=p, training=True)
            T = self.temperature.abs().clamp(min=0.1)
            dot = (eg * et).sum(dim=-1) / T
            ix = self.inter_mlp(xin).squeeze(-1)
            return dot + ix


def student_from_entry(
    entry: Mapping[str, Any], q_dim: int, i_dim: int, x_dim: int, device: str
) -> Any:
    s = GenericTwoTower(
        q_dim=q_dim,
        i_dim=i_dim,
        x_dim=x_dim,
        embed_dim=int(entry["embed_dim"]),
        hidden_dim=int(entry["hidden_dim"]),
        inter_hidden=int(entry["inter_hidden"]),
    ).to(device)
    s.load_state_dict(entry["state_dict"])
    s.eval()
    return s


# ── cell-level feature tensors ────────────────────────────────────────────────


def build_cell_tensors(
    gemms: List[dict],
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
) -> Tuple[Any, ...]:
    """(q_per_gemm, i_per_pair, x_per_pair, gemm_idx_for_pair, us_per_pair,
    winner_us_per_pair, rank_us_per_pair) of a cell's GEMMs."""
    q_per_gemm: List[List[float]] = []
    i_per_pair: List[List[float]] = []
    x_per_pair: List[List[float]] = []
    gemm_idx_for_pair: List[int] = []
    us_per_pair: List[float] = []
    winner_us_per_pair: List[float] = []
    rank_us_per_pair: List[int] = []
    for gi, g in enumerate(gemms):
        prob = fs.problem_kwargs_from_row(g)
        q, _, _ = fs.feature_vectors(
            prob,
            fs.config_kwargs_from_row(g["candidates"][0]),
            hardware=hardware,
            constants=constants,
        )
        q_per_gemm.append(q)
        for c in g["candidates"]:
            _, i_vec, x_vec = fs.feature_vectors(
                prob,
                fs.config_kwargs_from_row(c),
                hardware=hardware,
                constants=constants,
            )
            i_per_pair.append(i_vec)
            x_per_pair.append(x_vec)
            gemm_idx_for_pair.append(gi)
            us_per_pair.append(c["us"])
            winner_us_per_pair.append(g["winner_us"])
            rank_us_per_pair.append(c["rank_us"])
    return (
        np.asarray(q_per_gemm, dtype=np.float32),
        np.asarray(i_per_pair, dtype=np.float32),
        np.asarray(x_per_pair, dtype=np.float32),
        np.asarray(gemm_idx_for_pair, dtype=np.int64),
        np.asarray(us_per_pair, dtype=np.float64),
        np.asarray(winner_us_per_pair, dtype=np.float64),
        np.asarray(rank_us_per_pair, dtype=np.int64),
    )


# Must match the engine's whitening of a model's stored statistics, which
# inference applies to the same features.
CONSTANT_FEATURE_STD = 1e-3


def whitens_to_zero(
    tower: str, std: Sequence[float], deviation: np.ndarray
) -> np.ndarray:
    """The constant-feature rule, identical to the engine's (float32): mask,
    shaped like `deviation` (|x - mean| of each value), of the values of one
    tower ("q" query, "i" item, "x" interaction) that whiten to 0. A value of
    an item feature whose stored std is below CONSTANT_FEATURE_STD whitens to
    0 when it lies more than 2 std from the stored mean."""
    sd = np.asarray(std, dtype=np.float32)
    dev = np.asarray(deviation, dtype=np.float32)
    if tower != "i":
        return np.zeros(np.shape(dev), dtype=bool)
    return (sd < np.float32(CONSTANT_FEATURE_STD)) & (dev > np.float32(2.0) * sd)


def whiten_features(
    a: np.ndarray, mean: Sequence[float], std: Sequence[float], tower: str
) -> np.ndarray:
    """Whitened feature columns of one tower, in float32 from the stored
    statistics: (x - mean) / std, dividing by 1 when the std is below 1e-6,
    and 0 where `whitens_to_zero`."""
    mu = np.asarray(mean, dtype=np.float32)
    sd = np.asarray(std, dtype=np.float32)
    d = np.asarray(a, dtype=np.float32) - mu
    out = d / np.where(sd < np.float32(1e-6), np.float32(1.0), sd)
    zero = whitens_to_zero(tower, sd, np.abs(d))
    return np.where(zero, np.float32(0.0), out).astype(np.float32)


def whiten(
    qvec: np.ndarray, ivec: np.ndarray, xvec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Whitened training features and the per-feature (mean, std) statistics
    the model file stores (`whiten_features` with the training data's own
    statistics; the std as measured, so that the engine sees which features
    were constant)."""
    norms: Dict[str, List[float]] = {}
    out = []
    for tower, a in (("q", qvec), ("i", ivec), ("x", xvec)):
        a32 = np.asarray(a, dtype=np.float32)
        mu, sd = a32.mean(axis=0), a32.std(axis=0)
        norms[f"{tower}_mean"], norms[f"{tower}_std"] = mu.tolist(), sd.tolist()
        out.append(whiten_features(a32, mu, sd, tower))
    return out[0], out[1], out[2], norms


def apply_whiten(
    q: np.ndarray, i: np.ndarray, x: np.ndarray, norms: Dict[str, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten with stored statistics (`whiten_features`)."""
    return (
        whiten_features(q, norms["q_mean"], norms["q_std"], "q"),
        whiten_features(i, norms["i_mean"], norms["i_std"], "i"),
        whiten_features(x, norms["x_mean"], norms["x_std"], "x"),
    )


# ── LightGBM teacher ──────────────────────────────────────────────────────────


def fit_lgb_teacher(
    qvec: np.ndarray,
    ivec: np.ndarray,
    xvec: np.ndarray,
    gemm_idx: np.ndarray,
    us: np.ndarray,
    winner_us: np.ndarray,
    n_estimators: int = 300,
    num_leaves: int = 31,
    lr: float = 0.05,
    seed: int = 0,
    quiet: bool = False,
) -> Any:
    """LambdaRank teacher; one ranking group per GEMM, relevance 5 (winner)
    .. 0 from log-spaced buckets of log(us / winner_us)."""
    if not _LGB_OK:
        raise SystemExit("lightgbm not installed; pip install -r requirements.txt")
    X = np.concatenate([qvec[gemm_idx], ivec, xvec], axis=1)
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    bins = np.array([0.05, 0.10, 0.20, 0.40, 0.80])
    rel = 5 - np.searchsorted(bins, log_ratio, side="right")
    rel = np.clip(rel, 0, 5).astype(np.int32)
    # LightGBM's `group=` needs rows sorted by group.
    order = np.argsort(gemm_idx, kind="stable")
    Xs = X[order]
    rs = rel[order]
    _, group_sizes = np.unique(gemm_idx[order], return_counts=True)
    train_set = lgb.Dataset(Xs, label=rs, group=group_sizes, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": int(num_leaves),
        "learning_rate": float(lr),
        "verbose": -1 if quiet else 0,
        "force_col_wise": True,
        # The teacher must not depend on the thread count, which differs
        # between serial and parallel training.
        "deterministic": True,
        "seed": int(seed),
        # Same thread budget as torch; unset, LightGBM takes every core and
        # oversubscribes the parallel training pool.
        "num_threads": max(1, torch.get_num_threads()),
    }
    callbacks = [lgb.log_evaluation(period=0)] if quiet else None
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(n_estimators),
        callbacks=callbacks,
    )
    teacher_scores = booster.predict(X)
    return booster, teacher_scores.astype(np.float32)


# ── per-GEMM softmax helpers ──────────────────────────────────────────────────


def per_gemm_softmax_targets_np(
    us: np.ndarray,
    winner_us: np.ndarray,
    gemm_idx: np.ndarray,
    alpha: float,
    target_tolerance: float = 0.05,
) -> np.ndarray:
    """target_j ~ exp(-alpha * max(0, log(us / winner_us) - tol)), normalized
    per GEMM."""
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    adj = np.maximum(log_ratio - target_tolerance, 0.0)
    logits = -alpha * adj
    out = np.zeros_like(logits, dtype=np.float64)
    for gid in np.unique(gemm_idx):
        mask = gemm_idx == gid
        lg = logits[mask]
        lg = lg - lg.max()
        e = np.exp(lg)
        out[mask] = e / np.maximum(e.sum(), 1e-12)
    return out


def per_gemm_softmax_T(
    scores: torch.Tensor, gemm_idx: torch.Tensor, T: float, n_groups: int
) -> torch.Tensor:
    """Per-group softmax with temperature T, shape (T_total,)."""
    s = scores / T
    group_max = torch.full((n_groups,), float("-inf"), device=s.device, dtype=s.dtype)
    group_max.scatter_reduce_(0, gemm_idx, s, reduce="amax", include_self=False)
    e = torch.exp(s - group_max.index_select(0, gemm_idx))
    group_sum = torch.zeros(n_groups, device=s.device, dtype=s.dtype)
    group_sum.scatter_add_(0, gemm_idx, e)
    return e / group_sum.index_select(0, gemm_idx).clamp(min=1e-12)


def per_gemm_log_softmax(
    scores: torch.Tensor, gemm_idx: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Per-group log_softmax, shape (T_total,)."""
    group_max = torch.full(
        (n_groups,), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    group_max.scatter_reduce_(0, gemm_idx, scores, reduce="amax", include_self=False)
    s_centered = scores - group_max.index_select(0, gemm_idx)
    e = torch.exp(s_centered)
    group_sum = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
    group_sum.scatter_add_(0, gemm_idx, e)
    log_norm = group_sum.clamp(min=1e-12).log()
    return s_centered - log_norm.index_select(0, gemm_idx)


# ── training-time selection efficiency ────────────────────────────────────────


def evaluate_sel_eff(
    student: Any,
    q_t: torch.Tensor,
    i_t: torch.Tensor,
    x_t: torch.Tensor,
    gemm_idx_t: torch.Tensor,
    us_t: torch.Tensor,
    winner_us_t: torch.Tensor,
    sample_set_mask: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    """Geomean winner_us / pick_us of the student's top-1 over each GEMM's
    benched candidates, on the GEMMs selected by `sample_set_mask`. Ties go
    to the lowest pair index: the first in pool order, as in the engine,
    when each GEMM's candidates are in pool order."""
    n_gemms = q_t.shape[0]
    n_pairs = gemm_idx_t.shape[0]
    pair_in_sample = sample_set_mask.index_select(0, gemm_idx_t)

    student.eval()
    with torch.no_grad():
        q_per_pair = q_t.index_select(0, gemm_idx_t)
        scores = student.score_pairs(q_per_pair, i_t, x_t)
        scores_masked = scores.clone()
        scores_masked[~pair_in_sample] = float("-inf")
        group_max = torch.full(
            (n_gemms,), float("-inf"), device=device, dtype=scores_masked.dtype
        )
        group_max.scatter_reduce_(
            0, gemm_idx_t, scores_masked, reduce="amax", include_self=False
        )
        is_pick = (
            scores_masked == group_max.index_select(0, gemm_idx_t)
        ) & pair_in_sample
        sentinel = n_pairs
        pair_indices = torch.arange(n_pairs, device=device, dtype=torch.long)
        masked_pair_idx = torch.where(
            is_pick, pair_indices, torch.full_like(pair_indices, sentinel)
        )
        winner_pair_idx = torch.full(
            (n_gemms,), sentinel, device=device, dtype=torch.long
        )
        winner_pair_idx.scatter_reduce_(
            0, gemm_idx_t, masked_pair_idx, reduce="amin", include_self=False
        )
        winner_us_per_gemm = torch.full(
            (n_gemms,), float("nan"), device=device, dtype=winner_us_t.dtype
        )
        winner_us_per_gemm.scatter_reduce_(
            0, gemm_idx_t, winner_us_t, reduce="amin", include_self=False
        )
        has_pick = (
            (winner_pair_idx < sentinel)
            & sample_set_mask
            & ~torch.isnan(winner_us_per_gemm)
        )
        pick_us_all = us_t[winner_pair_idx.clamp(max=n_pairs - 1)]
        pick_us = pick_us_all[has_pick]
        winner_us_g = winner_us_per_gemm[has_pick]
        log_pick = torch.log(pick_us.clamp(min=1e-9))
        log_win = torch.log(winner_us_g.clamp(min=1e-9))
        log_ratios = log_pick - log_win
    student.train()
    n = int(log_ratios.numel())
    if n == 0:
        return {
            "sel_eff": 0.0,
            "pick_us_geomean": float("nan"),
            "winner_us_geomean": float("nan"),
            "n_eval": 0,
        }
    geomean_ratio = float(torch.exp(log_ratios.mean()).item())
    return {
        "sel_eff": 1.0 / max(geomean_ratio, 1e-9),
        "pick_us_geomean": float(torch.exp(log_pick.mean()).item()),
        "winner_us_geomean": float(torch.exp(log_win.mean()).item()),
        "n_eval": n,
    }


# ── kernel whitelist (smart_K) ────────────────────────────────────────────────


def _select_smart_k_oracle(
    gemms: List[Dict[str, Any]], k: int = 10
) -> List[Tuple[int, ...]]:
    """Greedy max-coverage that assumes the oracle-best kernel of the list is
    picked: maximizes mean(log(winner_us / min_us_in_list)), a signature's
    latency for a GEMM being the minimum over its solutions. An upper bound
    of any list; the deployed list uses the student-aware variant."""
    kernel_best_us: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(dict)
    winner_us: Dict[int, float] = {}
    for gi, g in enumerate(gemms):
        w = float(g.get("winner_us", 0) or 0)
        if w <= 0:
            continue
        winner_us[gi] = w
        for c in g.get("candidates", []):
            us = float(c.get("us", 0) or 0)
            if us <= 0:
                continue
            sig = tuple(int(v) for v in sig_from_row(c))
            prev = kernel_best_us[sig].get(gi)
            if prev is None or us < prev:
                kernel_best_us[sig][gi] = us
    if not winner_us or not kernel_best_us:
        return []
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    chosen: List[Tuple[int, ...]] = []
    chosen_set: set = set()
    best_us_per_gemm: Dict[int, float] = {gi: float("inf") for gi in winner_us}
    for _step in range(int(k)):
        best_sig: Optional[Tuple[int, ...]] = None
        best_score = -float("inf")
        for sig, us_by_g in kernel_best_us.items():
            if sig in chosen_set:
                continue
            total_log = 0.0
            count = 0
            for gi in winner_us:
                cur = best_us_per_gemm[gi]
                cand_us = us_by_g.get(gi)
                if cand_us is not None and cand_us < cur:
                    cur = cand_us
                if cur == float("inf"):
                    continue
                total_log += log_win[gi] - math.log(cur)
                count += 1
            if count == 0:
                continue
            avg_log = total_log / count
            if avg_log > best_score:
                best_score = avg_log
                best_sig = sig
        if best_sig is None:
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, us in kernel_best_us[best_sig].items():
            if us < best_us_per_gemm[gi]:
                best_us_per_gemm[gi] = us
    return chosen


def _student_score_cache(
    gemms: List[Dict[str, Any]],
    student: Any,
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    norms: Dict[str, Any],
    device: str,
    nt_flags: Mapping[Tuple[str, ...], Tuple[bool, bool]],
) -> Tuple[Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]], Dict[int, float]]:
    """({sig: {gi: (us, student score)}}, {gi: winner_us}) over the
    (kernel, GEMM) pairs that pass the engine's LDS gate and feasibility rule
    (`nt_flags`: per pool key, non-temporal availability of the library).
    A signature's entry for a GEMM is its fastest passing solution (the first
    in candidate order among equally fast ones) with that solution's score."""
    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]] = defaultdict(
        dict
    )
    winner_us: Dict[int, float] = {}
    student.eval()
    with torch.no_grad():
        for gi, g in enumerate(gemms):
            w = float(g.get("winner_us", 0) or 0)
            if w <= 0:
                continue
            prob = fs.problem_kwargs_from_row(g)
            nt_a, nt_b = nt_flags.get(ev.pool_key(g), (False, False))
            us_list: List[float] = []
            sig_list: List[Tuple[int, ...]] = []
            i_rows: List[List[float]] = []
            x_rows: List[List[float]] = []
            q_vec: Optional[List[float]] = None
            for c in g.get("candidates", []):
                us = float(c.get("us", 0) or 0)
                if us <= 0:
                    continue
                cfg = fs.config_kwargs_from_row(c)
                if not feasibility.passes_gates(
                    prob,
                    cfg,
                    lds_bytes=hardware.lds_bytes,
                    nt_a_available=nt_a,
                    nt_b_available=nt_b,
                ):
                    continue
                q, ivec, xvec = fs.feature_vectors(
                    prob, cfg, hardware=hardware, constants=constants
                )
                if q_vec is None:
                    q_vec = q
                i_rows.append(ivec)
                x_rows.append(xvec)
                us_list.append(us)
                sig_list.append(tuple(int(v) for v in sig_from_row(c)))
            if not us_list or q_vec is None:
                continue
            winner_us[gi] = w
            n = len(us_list)
            qa, ia, xa = apply_whiten(
                np.asarray([q_vec] * n, dtype=np.float32),
                np.asarray(i_rows, dtype=np.float32),
                np.asarray(x_rows, dtype=np.float32),
                norms,
            )
            scores = (
                student.score_pairs(
                    torch.tensor(qa, dtype=torch.float32, device=device),
                    torch.tensor(ia, dtype=torch.float32, device=device),
                    torch.tensor(xa, dtype=torch.float32, device=device),
                )
                .cpu()
                .numpy()
            )
            for j in range(n):
                prev = sig_to_data[sig_list[j]].get(gi)
                if prev is None or us_list[j] < prev[0]:
                    sig_to_data[sig_list[j]][gi] = (us_list[j], float(scores[j]))
    return sig_to_data, winner_us


def _greedy_student_aware(
    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]],
    winner_us: Dict[int, float],
    k: int,
) -> List[Tuple[int, ...]]:
    """Two-stage greedy list builder.

    Stage 1 (coverage): until every GEMM has a feasible signature in the
    list, add the signature covering the most uncovered GEMMs (ties by regret
    gain), so no training-distribution GEMM is left without a whitelisted
    candidate. Stage 2 (regret): fill the remaining slots with the signature
    that most improves mean log(winner_us / student_pick_us)."""
    if not winner_us or not sig_to_data:
        return []
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    chosen: List[Tuple[int, ...]] = []
    chosen_set: set = set()
    best_score_per_gemm: Dict[int, float] = {gi: -float("inf") for gi in winner_us}
    pick_us_per_gemm: Dict[int, float] = {gi: float("inf") for gi in winner_us}

    while len(chosen) < int(k):
        n_uncovered = sum(1 for gi in winner_us if pick_us_per_gemm[gi] == float("inf"))
        if n_uncovered == 0:
            break
        best_sig: Optional[Tuple[int, ...]] = None
        best_cover = 0
        best_regret_tiebreak = -float("inf")
        for sig, gi_data in sig_to_data.items():
            if sig in chosen_set:
                continue
            new_cover = 0
            regret_delta = 0.0
            for gi in winner_us:
                cur_pick_us = pick_us_per_gemm[gi]
                cand = gi_data.get(gi)
                if cand is None:
                    continue
                cand_us, cand_score = cand
                if cur_pick_us == float("inf"):
                    new_cover += 1
                    regret_delta += log_win[gi] - math.log(cand_us)
                elif cand_score > best_score_per_gemm[gi]:
                    regret_delta += math.log(cur_pick_us) - math.log(cand_us)
            if new_cover == 0:
                continue
            if new_cover > best_cover or (
                new_cover == best_cover and regret_delta > best_regret_tiebreak
            ):
                best_cover = new_cover
                best_regret_tiebreak = regret_delta
                best_sig = sig
        if best_sig is None:
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, (us, score) in sig_to_data[best_sig].items():
            if score > best_score_per_gemm[gi]:
                best_score_per_gemm[gi] = score
                pick_us_per_gemm[gi] = us

    while len(chosen) < int(k):
        best_sig = None
        best_obj = -float("inf")
        for sig, gi_data in sig_to_data.items():
            if sig in chosen_set:
                continue
            total_log = 0.0
            count = 0
            for gi in winner_us:
                cur_score = best_score_per_gemm[gi]
                cur_pick_us = pick_us_per_gemm[gi]
                cand = gi_data.get(gi)
                if cand is not None:
                    cand_us, cand_score = cand
                    new_pick_us = cand_us if cand_score > cur_score else cur_pick_us
                else:
                    new_pick_us = cur_pick_us
                if new_pick_us == float("inf"):
                    continue
                total_log += log_win[gi] - math.log(new_pick_us)
                count += 1
            if count == 0:
                continue
            avg = total_log / count
            if avg > best_obj:
                best_obj = avg
                best_sig = sig
        if best_sig is None:
            break
        chosen.append(best_sig)
        chosen_set.add(best_sig)
        for gi, (us, score) in sig_to_data[best_sig].items():
            if score > best_score_per_gemm[gi]:
                best_score_per_gemm[gi] = score
                pick_us_per_gemm[gi] = us
    return chosen


# With ML_SMARTK_RUNTIME_WEIGHT the knee weights each GEMM's log ratio by its
# winner_us, so dropping a signature that slows long-running GEMMs costs more.
_SMARTK_RT_WEIGHT = os.environ.get("ML_SMARTK_RUNTIME_WEIGHT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _evaluate_kset_student_aware(
    sig_list: List[Tuple[int, ...]],
    sig_to_data: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]],
    winner_us: Dict[int, float],
) -> Dict[str, Any]:
    """sel_eff when the student picks from each prefix of `sig_list`: the
    regret curve [(k_size, sel_eff, n_covered)]."""
    if not sig_list or not winner_us:
        return {"sel_eff": float("nan"), "n_eval": 0, "regret_curve": []}
    prefixes: List[set] = []
    cur: set = set()
    for s in sig_list:
        cur = cur | {s}
        prefixes.append(set(cur))
    log_win = {gi: math.log(w) for gi, w in winner_us.items()}
    regret_curve: List[Tuple[int, float, int]] = []
    for k_size, sk_set in enumerate(prefixes, start=1):
        total_log = 0.0
        wsum = 0.0
        count = 0
        for gi, w_us in winner_us.items():
            best_score = -float("inf")
            best_us = float("inf")
            for sig in sk_set:
                cand = sig_to_data.get(sig, {}).get(gi)
                if cand is None:
                    continue
                u, s = cand
                if s > best_score:
                    best_score = s
                    best_us = u
            if best_us == float("inf"):
                continue
            w = float(w_us) if _SMARTK_RT_WEIGHT else 1.0
            total_log += w * (log_win[gi] - math.log(best_us))
            wsum += w
            count += 1
        if count == 0:
            regret_curve.append((k_size, float("nan"), 0))
            continue
        sel_eff = math.exp(total_log / wsum) if wsum else float("nan")
        regret_curve.append((k_size, sel_eff, count))
    final = regret_curve[-1] if regret_curve else (0, float("nan"), 0)
    return {"sel_eff": final[1], "n_eval": final[2], "regret_curve": regret_curve}


def _adaptive_knee_k(
    regret_curve: List[Tuple[int, float, int]], k_min: int, saturation_eps: float
) -> Tuple[int, Dict[str, Any]]:
    """Smallest prefix that keeps the full list's GEMM coverage and is within
    `saturation_eps` of the best full-coverage sel_eff, clamped to >= k_min.
    (A shorter prefix that covers fewer GEMMs can score higher by dropping
    the hard ones, so only full-coverage prefixes set the target.)"""
    if not regret_curve:
        return max(1, int(k_min)), {
            "reason": "empty_curve",
            "k_full_curve": 0,
            "k_star": max(1, int(k_min)),
            "best_sel_eff": float("nan"),
            "sel_eff_at_k_star": None,
            "full_coverage": 0,
            "saturation_eps": saturation_eps,
            "k_min": int(k_min),
        }
    k_full = regret_curve[-1][0]
    full_cover = max((c for (_ks, _se, c) in regret_curve), default=0)
    full_sels = [se for (_ks, se, c) in regret_curve if c >= full_cover and se == se]
    best_full_sel = max(full_sels) if full_sels else float("nan")
    target = best_full_sel - saturation_eps if best_full_sel == best_full_sel else None
    k_star = k_full
    for k_size, se, cov in regret_curve:
        if cov >= full_cover and ((target is None) or (se == se and se >= target)):
            k_star = k_size
            break
    k_star = min(max(int(k_min), int(k_star)), k_full)
    info = {
        "k_full_curve": k_full,
        "k_star": k_star,
        "best_sel_eff": best_full_sel,
        "sel_eff_at_k_star": next(
            (se for (ks, se, _c) in regret_curve if ks == k_star), None
        ),
        "full_coverage": full_cover,
        "saturation_eps": saturation_eps,
        "k_min": int(k_min),
    }
    return k_star, info


def _seed_oracle_winners(
    gemms: List[Dict[str, Any]], sigs: List[Tuple[int, ...]], n_seed: int, cap: int
) -> List[Tuple[int, ...]]:
    """Prepend the first `n_seed` oracle max-coverage signatures missing from
    `sigs` (the kernels winning most GEMMs), truncating to `cap`."""
    if n_seed <= 0:
        return sigs
    seeds = _select_smart_k_oracle(gemms, k=int(n_seed))
    have = set(sigs)
    add = [s for s in seeds if s not in have]
    if not add:
        return sigs
    merged = add + list(sigs)
    if cap and len(merged) > int(cap):
        merged = merged[: int(cap)]
    return merged


def select_smart_k_signatures(
    gemms: List[Dict[str, Any]],
    k: int = 10,
    *,
    student: Optional[Any] = None,
    q_names: Optional[List[str]] = None,
    i_names: Optional[List[str]] = None,
    x_names: Optional[List[str]] = None,
    hardware: Optional[hwlib.DeviceHardware] = None,
    constants: Optional[hwlib.ArchConstants] = None,
    norms: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    nt_flags: Optional[Mapping[Tuple[str, ...], Tuple[bool, bool]]] = None,
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    saturation_eps: float = 0.005,
    oracle_seed: int = 0,
) -> Dict[str, Any]:
    """Kernel whitelist of one cell: {"sigs": [...], "metrics": {...}}.

    With a student, the greedy list is built against the student's actual
    picks (`_greedy_student_aware`); without one it is the oracle list. With
    `k_max` the list is grown to k_max and cut at the regret-curve knee in
    [k_min, k_max]; otherwise it has `k` entries. `oracle_seed` > 0 forces
    the cell's dominant oracle winners into the list."""
    if student is None or norms is None or q_names is None:
        return {
            "sigs": _select_smart_k_oracle(gemms, k=k),
            "metrics": {"mode": "oracle"},
        }
    assert hardware is not None and constants is not None
    sig_to_data, winner_us = _student_score_cache(
        gemms,
        student,
        q_names,
        i_names or [],
        x_names or [],
        hardware,
        constants,
        norms,
        device,
        nt_flags or {},
    )
    if not sig_to_data:
        return {
            "sigs": _select_smart_k_oracle(gemms, k=k),
            "metrics": {"mode": "oracle_fallback", "reason": "empty_cache"},
        }
    adaptive = k_max is not None
    k_build = int(k_max) if adaptive else int(k)
    sigs_full = _greedy_student_aware(sig_to_data, winner_us, k_build)
    knee_info: Optional[Dict[str, Any]] = None
    if adaptive:
        full_curve = _evaluate_kset_student_aware(sigs_full, sig_to_data, winner_us)[
            "regret_curve"
        ]
        k_star, knee_info = _adaptive_knee_k(
            full_curve, int(k_min) if k_min is not None else 1, saturation_eps
        )
        sigs_student = sigs_full[:k_star]
    else:
        sigs_student = sigs_full
    if oracle_seed and int(oracle_seed) > 0:
        sigs_student = _seed_oracle_winners(
            gemms, sigs_student, int(oracle_seed), k_build
        )
    sigs_oracle = _select_smart_k_oracle(gemms, k=len(sigs_student) or k)
    student_eval = _evaluate_kset_student_aware(sigs_student, sig_to_data, winner_us)
    oracle_eval = _evaluate_kset_student_aware(sigs_oracle, sig_to_data, winner_us)
    return {
        "sigs": sigs_student,
        "metrics": {
            "mode": "student_aware",
            "adaptive_k": bool(adaptive),
            "k_chosen": len(sigs_student),
            "k_build": k_build,
            "knee": knee_info,
            "n_sigs_student": len(sigs_student),
            "n_sigs_oracle": len(sigs_oracle),
            "student_aware_sel_eff": student_eval["sel_eff"],
            "oracle_sel_eff_under_student": oracle_eval["sel_eff"],
            "n_eval_gemms": student_eval["n_eval"],
            "student_aware_regret_curve": student_eval["regret_curve"],
            "oracle_regret_curve": oracle_eval["regret_curve"],
        },
    }


# ── training ──────────────────────────────────────────────────────────────────


def _train_worker_init(threads_per_worker: int) -> None:
    """Process-pool initializer: cap every thread pool of a training worker
    so workers x threads stays within the core count."""
    t = max(1, int(threads_per_worker))
    for _v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_v] = str(t)
    try:
        torch.set_num_threads(t)
    except Exception:
        pass


def _capacity(
    cell: str,
    n_gemms: int,
    us: np.ndarray,
    winner_us: np.ndarray,
    ivec_raw: np.ndarray,
) -> Tuple[int, int, int, float]:
    """(embed, hidden, inter_hidden, dropout) of a cell's student.

    ML_CAPACITY_MODE=difficulty (default) sizes by the number of distinct
    (near-)winning kernels of the cell; =n_gemms by its GEMM count.
    ML_FORCE_CAPACITY / ML_LARGE_CELL_CAPACITY ("ed,hd,ih") override every
    cell / the cells with a Large M/N or LargeK tier; ML_DROPOUT applies to
    those overridden or large cells."""
    mode = os.environ.get("ML_CAPACITY_MODE", "difficulty").strip().lower()
    if mode == "n_gemms":
        if n_gemms < 150:
            ed, hd, ih = 32, 96, 48
        elif n_gemms < 600:
            ed, hd, ih = 64, 160, 96
        elif n_gemms < 2000:
            ed, hd, ih = 80, 192, 112
        else:
            ed, hd, ih = 96, 256, 128
    else:
        wmask = us <= (winner_us * 1.001)
        wkeys = {tuple(np.round(ivec_raw[j], 3)) for j in np.nonzero(wmask)[0]}
        n_distinct_winners = max(1, len(wkeys))
        if n_gemms < 60 or n_distinct_winners <= 2:
            ed, hd, ih = 16, 48, 24
        elif n_distinct_winners <= 6:
            ed, hd, ih = 24, 80, 40
        elif n_distinct_winners <= 14:
            ed, hd, ih = 40, 112, 56
        elif n_distinct_winners <= 28:
            ed, hd, ih = 64, 160, 96
        else:
            ed, hd, ih = 80, 192, 112
    force_cap = os.environ.get("ML_FORCE_CAPACITY")
    if force_cap:
        try:
            ed, hd, ih = (int(v) for v in force_cap.split(","))
        except ValueError:
            pass
    base = sc.base_cell(cell).split("|")
    is_large = len(base) >= 3 and (
        base[0] == "Large" or base[1] == "Large" or base[2] == "LargeK"
    )
    large_cap = os.environ.get("ML_LARGE_CELL_CAPACITY")
    if large_cap and is_large:
        try:
            ed, hd, ih = (int(v) for v in large_cap.split(","))
        except ValueError:
            pass
    try:
        dropout = float(os.environ.get("ML_DROPOUT", "0") or 0)
    except ValueError:
        dropout = 0.0
    cell_dropout = (
        dropout if (dropout > 0 and (is_large or large_cap or force_cap)) else 0.0
    )
    return ed, hd, ih, cell_dropout


TRAINING_LOG_FIELDS: Tuple[str, ...] = (
    "cell",
    "epoch",
    "train_loss",
    "sel_eff",
    "val_sel_eff",
    "pick_us_geomean",
    "n_eval_sample",
    "n_gemms",
    "n_val_gemms",
)


def training_log_row(
    cell: str, m: Mapping[str, Any], n_gemms: int, n_val_gemms: int
) -> Dict[str, Any]:
    """training_log.csv row of one `epoch_metrics` entry."""
    return {
        "cell": cell,
        "epoch": m["epoch"],
        "train_loss": f"{m['train_loss']:.6f}",
        "sel_eff": f"{m['sel_eff']:.6f}",
        "val_sel_eff": "" if m["val_sel_eff"] is None else f"{m['val_sel_eff']:.6f}",
        "pick_us_geomean": f"{m['pick_us_geomean']:.4f}",
        "n_eval_sample": m["n_eval_sample"],
        "n_gemms": n_gemms,
        "n_val_gemms": n_val_gemms,
    }


def _tensors(arrays: Tuple[Any, ...], device: str) -> Tuple[Any, ...]:
    q, i, x, gidx, us, win = arrays
    return (
        torch.tensor(q, dtype=torch.float32, device=device),
        torch.tensor(i, dtype=torch.float32, device=device),
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(gidx, dtype=torch.long, device=device),
        torch.tensor(us, dtype=torch.float64, device=device),
        torch.tensor(win, dtype=torch.float64, device=device),
    )


def train_cell(
    *,
    cell: str,
    gemms: List[dict],
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    lam: float,
    T: float,
    alpha: float,
    target_tolerance: float,
    n_estimators: int,
    num_leaves: int,
    lgb_lr: float,
    seed: int,
    device: str,
    eval_sample_frac: float,
    use_teacher: bool,
    quiet: bool,
    training_log_writer: Any,
    validation_frac: float = 0.0,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Train one cell from freshly initialized weights (seeded by `seed`).

    The student trains on the cell's training GEMMs; the epoch kept is the
    one with the best sel_eff on the validation GEMMs, or on an
    `eval_sample_frac` sample of the training GEMMs when the cell has no
    validation GEMMs. Returns the models.pt entry plus measurements."""
    t0 = time.time()
    torch.manual_seed(int(seed))
    train_g, val_g = split_validation(gemms, validation_frac, split_seed)
    n_gemms = len(train_g)

    (qvec_raw, ivec_raw, xvec_raw, gemm_idx, us, winner_us, _rank_us) = (
        build_cell_tensors(train_g, q_names, i_names, x_names, hardware, constants)
    )
    qvec, ivec, xvec, norms = whiten(qvec_raw, ivec_raw, xvec_raw)

    teacher_scores_t = None
    if use_teacher:
        _booster, teacher_scores = fit_lgb_teacher(
            qvec_raw,
            ivec_raw,
            xvec_raw,
            gemm_idx,
            us,
            winner_us,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            lr=lgb_lr,
            seed=seed,
            quiet=quiet,
        )
        teacher_scores_t = torch.tensor(
            teacher_scores, dtype=torch.float32, device=device
        )

    soft_targets_t = torch.tensor(
        per_gemm_softmax_targets_np(
            us, winner_us, gemm_idx, alpha=alpha, target_tolerance=target_tolerance
        ),
        dtype=torch.float32,
        device=device,
    )
    q_t, i_t, x_t, gemm_idx_t, us_t, winner_us_t = _tensors(
        (qvec, ivec, xvec, gemm_idx, us, winner_us), device
    )

    rng_eval = np.random.default_rng(seed)
    n_eval_sample = max(1, int(round(n_gemms * eval_sample_frac)))
    eval_idx_np = np.sort(rng_eval.choice(n_gemms, size=n_eval_sample, replace=False))
    eval_sample_mask = torch.zeros(n_gemms, dtype=torch.bool, device=device)
    eval_sample_mask[torch.tensor(eval_idx_np, dtype=torch.long, device=device)] = True

    val_t: Optional[Tuple[Any, ...]] = None
    val_mask = None
    if val_g:
        (vq_raw, vi_raw, vx_raw, v_gidx, v_us, v_win, _v_rank) = build_cell_tensors(
            val_g, q_names, i_names, x_names, hardware, constants
        )
        vq, vi, vx = apply_whiten(vq_raw, vi_raw, vx_raw, norms)
        val_t = _tensors((vq, vi, vx, v_gidx, v_us, v_win), device)
        val_mask = torch.ones(len(val_g), dtype=torch.bool, device=device)

    ed, hd, ih, cell_dropout = _capacity(cell, n_gemms, us, winner_us, ivec_raw)
    student = GenericTwoTower(
        q_dim=len(q_names),
        i_dim=len(i_names),
        x_dim=len(x_names),
        embed_dim=ed,
        hidden_dim=hd,
        inter_hidden=ih,
        dropout=cell_dropout,
    )
    student.to(device)
    student.train()
    optim = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    best_sel_eff = -1.0
    best: Dict[str, Any] = {}
    best_epoch = -1
    best_state: Optional[Dict[str, Any]] = None
    per_epoch_metrics: List[Dict[str, Any]] = []
    n_groups = int(n_gemms)
    for ep in range(epochs):
        optim.zero_grad()
        scores = student.score_pairs(q_t.index_select(0, gemm_idx_t), i_t, x_t)
        log_s = per_gemm_log_softmax(scores, gemm_idx_t, n_groups)
        ce = -(soft_targets_t * log_s).sum() / float(n_gemms)
        if use_teacher and teacher_scores_t is not None:
            t_soft = per_gemm_softmax_T(
                teacher_scores_t, gemm_idx_t, T=T, n_groups=n_groups
            )
            s_soft = per_gemm_softmax_T(scores, gemm_idx_t, T=T, n_groups=n_groups)
            kl_pair = t_soft * (
                torch.log(t_soft.clamp(min=1e-12)) - torch.log(s_soft.clamp(min=1e-12))
            )
            kl = kl_pair.sum() / float(n_gemms) * (T**2)
            loss = lam * kl + (1.0 - lam) * ce
        else:
            loss = ce
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optim.step()
        train_loss = float(loss.item())

        tr = evaluate_sel_eff(
            student,
            q_t,
            i_t,
            x_t,
            gemm_idx_t,
            us_t,
            winner_us_t,
            sample_set_mask=eval_sample_mask,
            device=device,
        )
        va = (
            evaluate_sel_eff(student, *val_t, sample_set_mask=val_mask, device=device)
            if val_t is not None
            else None
        )
        sel = va if va is not None else tr
        per_epoch_metrics.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "sel_eff": tr["sel_eff"],
                "val_sel_eff": va["sel_eff"] if va is not None else None,
                "pick_us_geomean": tr["pick_us_geomean"],
                "n_eval_sample": tr["n_eval"],
            }
        )
        if training_log_writer is not None:
            training_log_writer.writerow(
                training_log_row(cell, per_epoch_metrics[-1], n_gemms, len(val_g))
            )
        if sel["sel_eff"] > best_sel_eff:
            best_sel_eff = sel["sel_eff"]
            best = {"train": tr, "val": va, "sel": sel}
            best_epoch = ep
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
        if not quiet and (ep < 3 or ep % 10 == 0 or ep == epochs - 1):
            vs = "" if va is None else f"  val={va['sel_eff'] * 100:.2f}%"
            print(
                f"  {ui.C.GREY}[{cell}]{ui.C.ENDC} ep {ep:>3d}/{epochs - 1}  "
                f"train_loss={train_loss:7.4f}  "
                f"sel_eff={tr['sel_eff'] * 100:.2f}%{vs}  "
                f"({tr['n_eval']} eval)",
                flush=True,
            )

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    sel = best.get("sel") or {}
    return {
        "cell": cell,
        "n_gemms": len(gemms),
        "n_train_gemms": n_gemms,
        "n_val_gemms": len(val_g),
        "selection_set": "validation" if val_g else "training",
        "n_pairs": int(ivec_raw.shape[0]),
        "q_dim": len(q_names),
        "i_dim": len(i_names),
        "x_dim": len(x_names),
        "embed_dim": ed,
        "hidden_dim": hd,
        "inter_hidden": ih,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "best_sel_eff": best_sel_eff,
        "best_pick_us_geomean": sel.get("pick_us_geomean", float("nan")),
        "winner_us_geomean": sel.get("winner_us_geomean", float("nan")),
        "train_sel_eff": (best.get("train") or {}).get("sel_eff"),
        "val_sel_eff": (best.get("val") or {}).get("sel_eff"),
        "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "feature_norms": norms,
        "epoch_metrics": per_epoch_metrics,
        "elapsed_s": time.time() - t0,
    }


# ── deployed evaluation ───────────────────────────────────────────────────────


def model_entry(r: Mapping[str, Any]) -> Dict[str, Any]:
    """models.pt entry of a train_cell result."""
    return {
        "state_dict": r["state_dict"],
        "feature_norms": r["feature_norms"],
        "embed_dim": r["embed_dim"],
        "hidden_dim": r["hidden_dim"],
        "inter_hidden": r["inter_hidden"],
        "best_epoch": r["best_epoch"],
        "smart_k_signatures": r.get("smart_k_signatures") or [],
    }


def _brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sel_eff": summary["sel_eff"],
        "pick_us_geomean": summary["pick_us_geomean"],
        "winner_us_geomean": summary["winner_us_geomean"],
        "n_eval": summary["n_evaluated"],
        "n_model_served": summary["n_model_served"],
        "n_origami_fallback": summary["n_origami_fallback"],
    }


def _origami_brief(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sel_eff": summary["origami_sel_eff"],
        "pick_us_geomean": summary["origami_pick_us_geomean"],
        "n_eval": summary["n_origami"],
    }


def deployed_eval_cell(
    label: str,
    r: Mapping[str, Any],
    train_g: List[dict],
    val_g: List[dict],
    *,
    names: Tuple[List[str], List[str], List[str]],
    arch: str,
    constants: hwlib.ArchConstants,
    hardware: hwlib.DeviceHardware,
    weight_dtype: str,
    pools: Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]],
    tw: Any,
) -> Tuple[Dict[str, Any], List[ev.GemmResult]]:
    """Deployed measurements of one cell (a one-cell model routes the cell's
    GEMMs exactly like the full bundle) and its validation GemmResults."""
    bundle = {
        "q_names": names[0],
        "i_names": names[1],
        "x_names": names[2],
        "models": {label: model_entry(r)},
    }
    out = ev.evaluate_bundle(
        bundle,
        list(train_g) + list(val_g),
        arch=arch,
        constants=constants,
        hardware=hardware,
        weight_dtype=weight_dtype,
        pools=pools,
        tw=tw,
    )
    ev.require_consistent_routing(out, f"{label}: deployed evaluation")
    res_train, res_val = out.results[: len(train_g)], out.results[len(train_g) :]
    s_train = ev.summarize(res_train)
    meas = {
        "deployed": _brief(s_train),
        "origami": _origami_brief(s_train),
        "per_gemm_log_ratio": np.asarray(
            [x.log_ratio for x in res_train], dtype=np.float64
        ),
    }
    if res_val:
        s_val = ev.summarize(res_val)
        meas["deployed_val"] = _brief(s_val)
        meas["origami_val"] = _origami_brief(s_val)
    return meas, res_val


# ── main ──────────────────────────────────────────────────────────────────────


def _load_prior_models(prior_round_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Models of the latest prior round's bundle; SystemExit when it exists
    but cannot be read, since the cells this round does not train would be
    lost."""
    if not prior_round_dirs:
        return {}
    path = Path(prior_round_dirs[-1]) / "stage05" / "models.pt"
    if not path.exists():
        return {}
    try:
        prior = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:
        ui.err("carry", f"cannot load the prior bundle {path}: {e}")
        raise SystemExit(1)
    return dict(prior.get("models") or {})


def carry_forward(
    prior: Mapping[str, Any],
    trained: Mapping[str, Any],
    split_tree: Mapping[str, sc.SplitRule],
) -> Tuple[Dict[str, Any], List[str]]:
    """(prior models the bundle keeps next to `trained`, the split parents
    among them).

    A prior model of a leaf of `split_tree` that this round did not train is
    kept. A prior model of a split parent is kept while a leaf under it would
    otherwise be served by no model or by one above the parent, so the
    parent's model goes on serving the leaves its children do not cover."""
    models = dict(trained)
    carried: Dict[str, Any] = {}
    for label, entry in prior.items():
        if label not in split_tree and label not in models:
            models[label] = carried[label] = entry
    fallback: List[str] = []
    parents = [p for p in prior if p in split_tree and p not in models]
    for parent in sorted(parents, key=lambda p: (-sc.split_depth(p), p)):
        for leaf in sc.leaves_under(parent, dict(split_tree)):
            served = sc.resolve_model_cell(leaf, models)
            if served is None or not served.startswith(parent + "#"):
                models[parent] = carried[parent] = prior[parent]
                fallback.append(parent)
                break
    return carried, sorted(fallback)


def wlog_mean(pairs: List[Tuple[Any, int]]) -> float:
    ls = ws = 0.0
    for v, w in pairs:
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        if v <= 0 or w <= 0:
            continue
        ls += math.log(v) * w
        ws += w
    return math.exp(ls / ws) if ws > 0 else float("nan")


def _environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "artifact_knobs": {k: os.environ.get(k) for k in ARTIFACT_ENV_KNOBS},
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if _TORCH_OK:
        env["torch"] = torch.__version__
        env["torch_num_threads"] = torch.get_num_threads()
    if _LGB_OK:
        env["lightgbm"] = lgb.__version__
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage05: per-cell distillation trainer (LightGBM teacher "
        "+ GenericTwoTower student)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--enriched-csv-dir",
        required=True,
        action="append",
        help="folder of chunk_*.csv; repeat to train on the union of rounds",
    )
    ap.add_argument(
        "--prior-round-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable `<run>/round_<i>/` dir, oldest first: their "
        "stage04b/splits.json build the split tree and the last one's "
        "models.pt provides the cells this round does not retrain",
    )
    ap.add_argument(
        "--current-round-dir",
        type=Path,
        default=None,
        help="this round's dir; its stage04b/splits.json joins the split tree "
        "so the children of this round's splits receive their GEMMs",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arch", required=True, help="config arch (e.g. gfx950)")
    ap.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="run config; its `hardware:` block (n_cu, lds_bytes, l2_bytes) "
        "sets the device values of the features",
    )
    ap.add_argument(
        "--hardware-device",
        type=int,
        default=0,
        help="device whose KFD topology supplies hardware values missing "
        "from the config",
    )
    ap.add_argument(
        "--weight-dtype",
        choices=sorted(ev.mlrec.WEIGHT_DTYPES),
        default="bf16",
        help="weight dtype of the deployed model; the deployed evaluation " "uses it",
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help="Tensile library dir holding <library-stem>.dat[.zlib]; with "
        "--library-stem that library is the candidate pool (default: the "
        "kernels measured in the training data)",
    )
    ap.add_argument("--library-stem", default=None)
    ap.add_argument(
        "--skip-deployed-eval",
        action="store_true",
        help="train without the tilewright module (no deployed measurements, "
        "no --smartk-fallback-delta)",
    )
    ap.add_argument(
        "--smart-k",
        type=int,
        default=10,
        help="whitelist size per cell (signatures the engine scores first)",
    )
    ap.add_argument(
        "--smart-k-max",
        type=int,
        default=None,
        help="grow each whitelist to this size and cut it at the regret-curve "
        "knee in [--smart-k-min, --smart-k-max] instead of using --smart-k",
    )
    ap.add_argument("--smart-k-min", type=int, default=4)
    ap.add_argument(
        "--smart-k-saturation-eps",
        type=float,
        default=0.005,
        help="knee: smallest list within this sel_eff of the best that keeps "
        "full GEMM coverage",
    )
    ap.add_argument(
        "--smartk-fallback-delta",
        type=float,
        default=0.0,
        help="clear a cell's whitelist (the engine then ranks every feasible "
        "kernel) when deployed sel_eff minus the student's full-ranking "
        "sel_eff is <= this negative value; 0 disables",
    )
    ap.add_argument(
        "--oracle-seed",
        type=int,
        default=0,
        help="force this many oracle max-coverage signatures into each " "whitelist",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="distillation weight: loss = lam*KL + (1-lam)*soft_CE",
    )
    ap.add_argument("--T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument(
        "--alpha", type=float, default=12.0, help="soft-CE concentration on the winner"
    )
    ap.add_argument(
        "--target-tolerance",
        type=float,
        default=0.05,
        help="log-ratio within which a candidate counts as the winner in soft-CE",
    )
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--lgb-lr", type=float, default=0.05)
    ap.add_argument(
        "--no-teacher",
        action="store_true",
        help="no LightGBM teacher; soft-CE only",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="torch device")
    ap.add_argument(
        "--train-workers",
        type=int,
        default=1,
        help="cells trained concurrently in a process pool",
    )
    ap.add_argument(
        "--train-threads-per-worker",
        type=int,
        default=0,
        help="torch/BLAS/LightGBM threads per worker (0: cpu_count // workers)",
    )
    ap.add_argument(
        "--validation-frac",
        type=float,
        default=0.2,
        help="fraction of each cell's GEMMs held out to choose the best epoch "
        "and to measure the trained model (0: none)",
    )
    ap.add_argument(
        "--eval-sample-frac",
        type=float,
        default=1.0,
        help="fraction of the training GEMMs whose sel_eff is logged per "
        "epoch (and selects the epoch when a cell has no validation GEMMs)",
    )
    ap.add_argument(
        "--only-cells",
        default=None,
        help="comma-separated labels to train; one that cannot be trained "
        f"makes the stage exit {EXIT_CELLS_NOT_TRAINED}",
    )
    ap.add_argument("--skip-cells", default=None, help="comma-separated labels")
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=50,
        help="cells with fewer GEMMs are not trained",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not _TORCH_OK:
        ui.err("err", "torch is not importable; pip install -r requirements.txt")
        raise SystemExit(1)
    if not _LGB_OK and not args.no_teacher:
        ui.err("err", "lightgbm is not importable; install it or pass --no-teacher")
        raise SystemExit(1)

    cfg: Dict[str, Any] = {}
    if args.config_yaml is not None:
        with args.config_yaml.open() as f:
            cfg = yaml.safe_load(f) or {}
    try:
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
        constants = hwlib.arch_constants(args.arch)
    except (ValueError, RuntimeError, OSError) as e:
        ui.err("hardware", str(e))
        raise SystemExit(1)
    tw = None
    if not args.skip_deployed_eval:
        try:
            tw = ev.tilewright_module()
        except RuntimeError as e:
            ui.err("engine", f"{e} (or pass --skip-deployed-eval)")
            raise SystemExit(1)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    q_names = fs.query_feature_names()
    i_names = fs.item_feature_names()
    x_names = fs.interaction_feature_names()
    names = (q_names, i_names, x_names)

    if not args.quiet:
        ui.banner("stage05  per-cell distillation trainer", ui.C.HEAD)
        ui.info("cfg", f"enriched dirs   : {args.enriched_csv_dir}")
        ui.info("cfg", f"output_dir      : {args.output_dir}")
        ui.info("cfg", f"arch / device   : {args.arch} / {args.device}")
        ui.info("cfg", f"hardware        : {asdict(hardware)}")
        ui.info("cfg", f"weight dtype    : {args.weight_dtype}")
        ui.info("cfg", f"epochs / seed   : {args.epochs} / {args.seed}")
        ui.info("cfg", f"validation frac : {args.validation_frac}")
        ui.info(
            "cfg",
            f"teacher         : {'off' if args.no_teacher else 'LightGBM'}  "
            f"lam/T/alpha={args.lam}/{args.T}/{args.alpha}",
        )
        ui.info(
            "feat",
            f"feature dims Q={len(q_names)} I={len(i_names)} X={len(x_names)}  "
            f"hash={fs.feature_names_hash()}",
        )

    prior_models = _load_prior_models(args.prior_round_dir or [])
    split_tree = sc.load_cumulative_split_tree(args.prior_round_dir or [])
    if args.current_round_dir is not None:
        split_tree.update(sc.load_cumulative_split_tree([args.current_round_dir]))
    if split_tree and not args.quiet:
        ui.ok("splits", f"split tree: {len(split_tree)} split cells")
        for line in sc.format_split_tree(split_tree).splitlines():
            print(f"        {line}", flush=True)

    cells, load_stats = load_enriched_chunks(
        args.enriched_csv_dir, args.quiet, split_tree
    )
    all_gemms = [g for gs in cells.values() for g in gs]
    if (args.library_dir is None) != (args.library_stem is None):
        ui.err("pool", "--library-dir and --library-stem go together")
        raise SystemExit(1)
    library = (
        ev.library_pool(args.library_dir, args.library_stem)
        if args.library_stem
        else None
    )
    pools = ev.pools_for(all_gemms, library)
    ev.order_candidates_by_pool(all_gemms, pools)
    nt_flags = {k: feasibility.non_temporal_availability(v) for k, v in pools.items()}
    if not args.quiet:
        for k, v in pools.items():
            ui.info(
                "pool",
                f"{'/'.join(k)}: {len(v)} kernels  "
                f"non-temporal A/B available: {nt_flags[k]}",
            )

    def _labels(s: Optional[str]) -> Optional[set]:
        return {c.strip() for c in s.split(",") if c.strip()} if s else None

    only = _labels(args.only_cells)
    skip = _labels(args.skip_cells) or set()
    base_set = set(grid.all_cell_labels())
    cell_order = [c for c in grid.all_cell_labels() if c in cells] + sorted(
        (c for c in cells if c not in base_set), key=lambda c: (sc.split_depth(c), c)
    )
    cells_to_train: List[str] = []
    cells_skipped: List[Tuple[str, str]] = []
    for label in cell_order:
        if only is not None and label not in only:
            cells_skipped.append((label, "not-in-only"))
        elif label in skip:
            cells_skipped.append((label, "in-skip"))
        elif len(cells[label]) < args.min_cell_gemms:
            cells_skipped.append((label, f"too few gemms ({len(cells[label])})"))
        else:
            cells_to_train.append(label)
    cells_unpopulated = [
        lbl
        for lbl in grid.all_cell_labels()
        if lbl not in cells and not any(c.startswith(lbl + "#") for c in cells)
    ]
    if not args.quiet:
        ui.ok(
            "cells",
            f"populated={len(cells)}  to_train={len(cells_to_train)}  "
            f"skipped={len(cells_skipped)}  unpopulated base cells="
            f"{len(cells_unpopulated)}  (min_cell_gemms={args.min_cell_gemms})",
        )
        for label, why in cells_skipped[:6]:
            ui.warn("cells", f"  skip {label}: {why}")
        if len(cells_skipped) > 6:
            ui.info("cells", f"  ... ({len(cells_skipped) - 6} more skipped)")
        if cells_unpopulated:
            ui.warn(
                "cells",
                f"no data for {len(cells_unpopulated)} base cells; the runtime "
                f"ranks them with Origami: {', '.join(cells_unpopulated)}",
            )

    train_kwargs = dict(
        q_names=q_names,
        i_names=i_names,
        x_names=x_names,
        hardware=hardware,
        constants=constants,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        lam=args.lam,
        T=args.T,
        alpha=args.alpha,
        target_tolerance=args.target_tolerance,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        lgb_lr=args.lgb_lr,
        device=args.device,
        eval_sample_frac=args.eval_sample_frac,
        use_teacher=not args.no_teacher,
        validation_frac=args.validation_frac,
        split_seed=args.seed,
    )

    log_path = os.path.join(args.output_dir, "training_log.csv")
    all_results: List[Dict[str, Any]] = []
    cells_failed: List[str] = []
    failures: Dict[str, str] = {}
    val_results: List[ev.GemmResult] = []
    with open(log_path, "w", newline="") as logf:
        log_writer = csv.DictWriter(logf, fieldnames=list(TRAINING_LOG_FIELDS))
        log_writer.writeheader()

        train_results: Dict[str, Dict[str, Any]] = {}
        n_workers = min(int(args.train_workers or 1), len(cells_to_train))
        if n_workers > 1:
            import concurrent.futures as _cf
            import multiprocessing as _mp

            tpw = int(args.train_threads_per_worker or 0)
            if tpw <= 0:
                tpw = max(2, (os.cpu_count() or 8) // n_workers)
            for _v in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ[_v] = str(tpw)
            order = sorted(cells_to_train, key=lambda c: len(cells[c]), reverse=True)
            if not args.quiet:
                ui.info(
                    "parallel",
                    f"training {len(order)} cells on {n_workers} workers x "
                    f"{tpw} threads",
                )
            t_par = time.time()
            with _cf.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=_mp.get_context("spawn"),
                initializer=_train_worker_init,
                initargs=(tpw,),
            ) as ex:
                futures = {
                    ex.submit(
                        train_cell,
                        cell=label,
                        gemms=cells[label],
                        seed=cell_seed(args.seed, label),
                        quiet=True,
                        training_log_writer=None,
                        **train_kwargs,
                    ): label
                    for label in order
                }
                for nd, fut in enumerate(_cf.as_completed(futures), 1):
                    label = futures[fut]
                    try:
                        train_results[label] = fut.result()
                    except Exception as e:
                        ui.warn(
                            "parallel", f"{label}: failed ({e!r}); retrying serially"
                        )
                    if not args.quiet and (nd % 10 == 0 or nd == len(order)):
                        ui.info(
                            "parallel",
                            f"  {nd}/{len(order)} cells trained "
                            f"({ui.fmt_dur(time.time() - t_par)})",
                        )

        for ci, label in enumerate(cells_to_train, 1):
            gemms = cells[label]
            if not args.quiet:
                print(
                    f"\n{ui.C.BLUE}[cell {ci:>2d}/{len(cells_to_train)}]{ui.C.ENDC} "
                    f"{ui.C.BOLD}{label}{ui.C.ENDC}  n_gemms={ui.fmt_int(len(gemms))}",
                    flush=True,
                )
            try:
                r = train_results.get(label)
                if r is None:
                    r = train_cell(
                        cell=label,
                        gemms=gemms,
                        seed=cell_seed(args.seed, label),
                        quiet=args.quiet,
                        training_log_writer=log_writer,
                        **train_kwargs,
                    )
                else:
                    log_writer.writerows(
                        training_log_row(label, m, r["n_train_gemms"], r["n_val_gemms"])
                        for m in r["epoch_metrics"]
                    )
                logf.flush()
                train_g, val_g = split_validation(
                    gemms, args.validation_frac, args.seed
                )
                student = student_from_entry(
                    r, len(q_names), len(i_names), len(x_names), args.device
                )
                sk = select_smart_k_signatures(
                    train_g,
                    k=int(args.smart_k),
                    student=student,
                    q_names=q_names,
                    i_names=i_names,
                    x_names=x_names,
                    hardware=hardware,
                    constants=constants,
                    norms=r["feature_norms"],
                    device=args.device,
                    nt_flags=nt_flags,
                    k_min=(
                        int(args.smart_k_min) if args.smart_k_max is not None else None
                    ),
                    k_max=(
                        int(args.smart_k_max) if args.smart_k_max is not None else None
                    ),
                    saturation_eps=float(args.smart_k_saturation_eps),
                    oracle_seed=int(args.oracle_seed or 0),
                )
                r["smart_k_signatures"] = sk["sigs"]
                r["smart_k_metrics"] = sk["metrics"]
                if not args.quiet:
                    m = sk["metrics"]
                    extra = (
                        f"student_aware_sel_eff={m['student_aware_sel_eff']:.4f}  "
                        f"oracle_under_student={m['oracle_sel_eff_under_student']:.4f}"
                        if m.get("mode") == "student_aware"
                        else f"mode={m.get('mode')}"
                    )
                    ui.info("smart_k", f"{label}  K={len(sk['sigs'])}  {extra}")
                r["origami"] = ev.origami_summary(train_g)
                if val_g:
                    r["origami_val"] = ev.origami_summary(val_g)
                cell_val: List[ev.GemmResult] = []
                eval_kw = dict(
                    names=names,
                    arch=args.arch,
                    constants=constants,
                    hardware=hardware,
                    weight_dtype=args.weight_dtype,
                    pools=pools,
                    tw=tw,
                )
                if tw is not None:
                    meas, cell_val = deployed_eval_cell(
                        label, r, train_g, val_g, **eval_kw
                    )
                    r.update(meas)
                if tw is not None and "deployed" in r:
                    fb = float(args.smartk_fallback_delta or 0.0)
                    dep_key = "deployed_val" if "deployed_val" in r else "deployed"
                    model_se = (
                        r.get("val_sel_eff")
                        if dep_key == "deployed_val"
                        else r.get("train_sel_eff")
                    )
                    dep_se = r[dep_key]["sel_eff"]
                    if (
                        fb < 0
                        and model_se
                        and math.isfinite(dep_se)
                        and dep_se - model_se <= fb
                        and r["smart_k_signatures"]
                    ):
                        n_dropped = len(r["smart_k_signatures"])
                        r["smart_k_signatures"] = []
                        meas, cell_val = deployed_eval_cell(
                            label, r, train_g, val_g, **eval_kw
                        )
                        r.update(meas)
                        r["smart_k_metrics"]["quality_fallback"] = {
                            "fired": True,
                            "measured_on": dep_key,
                            "k_delta_before": dep_se - model_se,
                            "sel_eff_before": dep_se,
                            "sel_eff_after": r[dep_key]["sel_eff"],
                            "n_sigs_dropped": n_dropped,
                        }
                        if not args.quiet:
                            ui.info(
                                "smart_k",
                                f"{label}  whitelist cleared (deployed "
                                f"{fmt_pct(dep_se)} vs model {fmt_pct(model_se)})",
                            )
                val_results.extend(cell_val)
                all_results.append(r)
                if not args.quiet:
                    dep = r.get("deployed") or {}
                    dval = r.get("deployed_val") or {}
                    ori = r.get("origami_val") or r.get("origami") or {}
                    ui.ok(
                        "cell",
                        f"{label}  best_ep={r['best_epoch']}  "
                        f"model={fmt_pct(r['best_sel_eff'])} ({r['selection_set']})  "
                        f"deployed={fmt_pct(dep.get('sel_eff'))}  "
                        f"deployed_val={fmt_pct(dval.get('sel_eff'))}  "
                        f"origami={fmt_pct(ori.get('sel_eff'))}  "
                        f"({ui.fmt_dur(r['elapsed_s'])})",
                    )
            except Exception as e:
                import traceback

                ui.err("cell", f"{label} FAILED: {e}")
                ui.err("cell", traceback.format_exc().rstrip())
                cells_failed.append(label)
                failures[label] = f"{type(e).__name__}: {e}"

    cells_not_trained: List[Dict[str, str]] = []
    if only is not None:
        trained_labels = {r["cell"] for r in all_results}
        skip_reason = dict(cells_skipped)
        for label in sorted(only - trained_labels - set(cells_failed) - skip):
            cells_not_trained.append(
                {
                    "cell": label,
                    "reason": skip_reason.get(label, "no GEMMs route to this cell"),
                }
            )

    if not args.quiet:
        ui.banner("Aggregate + save", ui.C.BLUE)
        w = min(max(max((len(r["cell"]) for r in all_results), default=24), 24), 80)
        hdr = (
            f"  {'cell':<{w}}  {'n_gemms':>8}  {'model%':>7}  {'deploy%':>7}  "
            f"{'dep_val%':>8}  {'orig_val%':>9}  {'K':>3}"
        )
        print(hdr, flush=True)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        for r in sorted(
            all_results,
            key=lambda r: (r.get("deployed_val") or r.get("deployed") or {}).get(
                "sel_eff", float("nan")
            ),
        ):
            print(
                f"  {r['cell']:<{w}}  {ui.fmt_int(r['n_gemms']):>8}  "
                f"{fmt_pct(r['best_sel_eff'])}  "
                f"{fmt_pct((r.get('deployed') or {}).get('sel_eff'))}  "
                f" {fmt_pct((r.get('deployed_val') or {}).get('sel_eff'))}  "
                f"  {fmt_pct((r.get('origami_val') or {}).get('sel_eff'))}  "
                f"{len(r.get('smart_k_signatures') or []):>3}",
                flush=True,
            )

    def _g(key: str, field: str, weight: str = "n_gemms") -> float:
        return wlog_mean(
            [
                ((r.get(key) or {}).get(field), int(r.get(weight, 0) or 0))
                for r in all_results
            ]
        )

    validation_summary = (
        ev.summarize(val_results, bootstrap=1000) if val_results else None
    )
    global_m = {
        "global_sel_eff": wlog_mean(
            [(r.get("best_sel_eff"), r["n_gemms"]) for r in all_results]
        ),
        "global_model_pick_us_geomean": wlog_mean(
            [(r.get("best_pick_us_geomean"), r["n_gemms"]) for r in all_results]
        ),
        "global_winner_us_geomean": wlog_mean(
            [(r.get("winner_us_geomean"), r["n_gemms"]) for r in all_results]
        ),
        "global_origami_sel_eff": _g("origami", "sel_eff"),
        "global_origami_pick_us_geomean": _g("origami", "pick_us_geomean"),
        "global_deployed_sel_eff": _g("deployed", "sel_eff"),
        "global_deployed_pick_us_geomean": _g("deployed", "pick_us_geomean"),
        "global_val_sel_eff": wlog_mean(
            [(r.get("val_sel_eff"), r.get("n_val_gemms", 0)) for r in all_results]
        ),
        "global_deployed_val_sel_eff": _g("deployed_val", "sel_eff", "n_val_gemms"),
        "global_origami_val_sel_eff": _g("origami_val", "sel_eff", "n_val_gemms"),
    }

    def _cell_entry(r: Dict[str, Any]) -> Dict[str, Any]:
        def _scalars(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not isinstance(d, dict):
                return d
            return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

        entry = {
            "label": r["cell"],
            "n_gemms": r["n_gemms"],
            "n_train_gemms": r["n_train_gemms"],
            "n_val_gemms": r["n_val_gemms"],
            "selection_set": r["selection_set"],
            "n_pairs": r["n_pairs"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_sel_eff": r["best_sel_eff"],
            "train_sel_eff": r.get("train_sel_eff"),
            "val_sel_eff": r.get("val_sel_eff"),
            "best_pick_us_geomean": r.get("best_pick_us_geomean"),
            "winner_us_geomean": r.get("winner_us_geomean"),
            "embed_dim": r["embed_dim"],
            "hidden_dim": r["hidden_dim"],
            "inter_hidden": r["inter_hidden"],
            "elapsed_s": r["elapsed_s"],
            "origami": _scalars(r.get("origami")),
            "deployed": _scalars(r.get("deployed")),
            "origami_val": _scalars(r.get("origami_val")),
            "deployed_val": _scalars(r.get("deployed_val")),
            "smart_k_signatures": [
                [int(v) for v in s] for s in r.get("smart_k_signatures") or []
            ],
            "smart_k_metrics": r.get("smart_k_metrics"),
        }
        if sc.is_subcell(r["cell"]):
            entry["base_cell"] = sc.base_cell(r["cell"])
            entry["split_depth"] = sc.split_depth(r["cell"])
        return entry

    new_models: Dict[str, Any] = {r["cell"]: model_entry(r) for r in all_results}
    carried, fallback_parents = carry_forward(prior_models, new_models, split_tree)
    new_models.update(carried)
    if carried and not args.quiet:
        ui.ok("carry", f"carried {len(carried)} models forward from the prior round")
    if fallback_parents and not args.quiet:
        ui.info(
            "carry",
            f"kept split parents for leaves without a model: "
            f"{', '.join(fallback_parents)}",
        )

    cells_json = {
        "schema_version": "pipeline-v3-2026-10-03",
        "feature_version": fs.FEATURES_VERSION,
        "feature_names_hash": fs.feature_names_hash(),
        "q_names": q_names,
        "i_names": i_names,
        "x_names": x_names,
        "n_cells_populated": len(cells),
        "n_cells_trained": len(all_results),
        "n_subcells_trained": sum(1 for r in all_results if sc.is_subcell(r["cell"])),
        "model_labels": sorted(new_models),
        "fallback_parents": fallback_parents,
        "cells": [_cell_entry(r) for r in all_results],
    }
    with open(os.path.join(args.output_dir, "cells.json"), "w") as f:
        json.dump(jsonable(cells_json), f, indent=2)

    torch.save(
        {
            "schema_version": cells_json["schema_version"],
            "feature_version": fs.FEATURES_VERSION,
            "q_dim": len(q_names),
            "i_dim": len(i_names),
            "x_dim": len(x_names),
            "q_names": q_names,
            "i_names": i_names,
            "x_names": x_names,
            "models": new_models,
            "n_models_trained_this_round": len(all_results),
            "n_models_carried_forward": len(carried),
            "fallback_parents": fallback_parents,
        },
        os.path.join(args.output_dir, "models.pt"),
    )

    def _per_cell(get) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in all_results:
            try:
                out[r["cell"]] = get(r)
            except (KeyError, TypeError):
                out[r["cell"]] = None
        return out

    metrics = {
        **global_m,
        "n_cells_trained": len(all_results),
        "n_cells_failed": len(cells_failed),
        "cells_failed": cells_failed,
        "failures": failures,
        "cells_not_trained": cells_not_trained,
        "fallback_parents": fallback_parents,
        "n_gemms_trained": sum(r["n_gemms"] for r in all_results),
        "validation": validation_summary,
        "load_stats": load_stats,
        "args": jsonable(vars(args)),
        "hardware": asdict(hardware),
        "arch_constants": {
            "parallel_mi_cu": constants.parallel_mi_cu,
            "bw": list(constants.bw),
            "mi_default": constants.mi_default,
            "n_mi_entries": len(constants.mi_table),
        },
        "weight_dtype": args.weight_dtype,
        "environment": _environment(),
        "feature_version": fs.FEATURES_VERSION,
        "feature_names_hash": fs.feature_names_hash(),
        "elapsed_total_s": time.time() - t0,
        "per_cell_sel_eff": _per_cell(lambda r: r["best_sel_eff"]),
        "per_cell_val_sel_eff": _per_cell(lambda r: r.get("val_sel_eff")),
        "per_cell_origami_sel_eff": _per_cell(lambda r: r["origami"]["sel_eff"]),
        "per_cell_deployed_sel_eff": _per_cell(lambda r: r["deployed"]["sel_eff"]),
        "per_cell_deployed_val_sel_eff": _per_cell(
            lambda r: r["deployed_val"]["sel_eff"]
        ),
        "per_cell_winner_us_geomean": _per_cell(lambda r: r.get("winner_us_geomean")),
        "smart_k": args.smart_k,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(jsonable(metrics), f, indent=2)

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        for key in (
            "global_sel_eff",
            "global_val_sel_eff",
            "global_deployed_sel_eff",
            "global_deployed_val_sel_eff",
            "global_origami_val_sel_eff",
        ):
            print(
                f"  {ui.C.BOLD}{key:32s}:{ui.C.ENDC} {fmt_pct(metrics[key])}",
                flush=True,
            )
        if validation_summary:
            p = validation_summary["paired"]
            print(
                f"  {ui.C.BOLD}{'validation vs origami (paired)':32s}:{ui.C.ENDC} "
                f"n={p['n']}  wins/ties/losses={p['wins']}/{p['ties']}/"
                f"{p['losses']}",
                flush=True,
            )
        print(
            f"  {ui.C.BOLD}{'n_cells_trained':32s}:{ui.C.ENDC} {len(all_results)}",
            flush=True,
        )
        if cells_failed:
            print(
                f"  {ui.C.RED}{'n_cells_failed':32s}:{ui.C.ENDC} {len(cells_failed)}  "
                f"({', '.join(cells_failed[:5])}"
                f"{'...' if len(cells_failed) > 5 else ''})",
                flush=True,
            )
        print(
            f"  {ui.C.BOLD}{'elapsed':32s}:{ui.C.ENDC} "
            f"{ui.fmt_dur(metrics['elapsed_total_s'])}",
            flush=True,
        )
        print(
            f"  {ui.C.BOLD}{'outputs':32s}:{ui.C.ENDC} {args.output_dir}/ "
            f"(cells.json, models.pt, training_log.csv, metrics.json)",
            flush=True,
        )
    if cells_failed or cells_not_trained:
        for label in cells_failed:
            ui.err("done", f"{label}: failed ({failures[label]})")
        for row in cells_not_trained:
            ui.err(
                "done", f"{row['cell']}: requested but not trained ({row['reason']})"
            )
        ui.err(
            "done",
            f"{len(cells_failed)} cell(s) failed and {len(cells_not_trained)} "
            f"requested cell(s) were not trained; outputs are written "
            f"(metrics.json: cells_failed, cells_not_trained); "
            f"exit {EXIT_CELLS_NOT_TRAINED}",
        )
        return EXIT_CELLS_NOT_TRAINED
    return 0


if __name__ == "__main__":
    sys.exit(main())
