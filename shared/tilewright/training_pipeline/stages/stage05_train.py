#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage05_train -- per-cell distillation trainer of the kernel recommender.

Input : one or more folders of enriched chunk_*.csv from stage04. Repeating
        --enriched-csv-dir trains on their union (cross-round augmentation).
Output: <out>/models.pt         per-cell student weights, feature statistics,
                                kernel whitelists and provenance (what stage06
                                deploys; lib/provenance.py); for v3 also the
                                view keys with training rows and whether the
                                pool held the bank; the training recipe and,
                                for pooled_ft, its pooled student
        <out>/cells.json        per-cell metadata and measurements
        <out>/metrics.json      global measurements, arguments, environment,
                                provenance, how the benches of the
                                enriched folders launched the kernels
                                (`bench_launch`, lib/launch_settings.py) and
                                the blocklists that filtered the labels
                                (`blocklist`, `labels_blocklist`); models.pt
                                keeps both too
        <out>/model_options.json
                                the model-shaping options the cells it trains
                                are trained with and the kernel identity rule
                                (`model_options`)
        <out>/member_order.json with --library-stem: the order of the members
                                of every candidate key of the pool the
                                runtime serves from (`member_order`), which
                                stage06 compares with the library it deploys
                                for
        <out>/whitelist_tier.json
                                with --whitelist-tier-k: the bundle's
                                whitelist tier, per cell its keys and what
                                they were chosen on (lib/whitelist_tier.py)
        <out>/training_log.csv  loss and selection efficiency per measured
                                checkpoint

Every GEMM routes to a leaf of the 96-cell grid plus the cumulative split
tree; each leaf with at least --min-cell-gemms GEMMs is trained:

  1. Validation split: a GEMM is a validation GEMM when a hash of (--seed,
     GEMM identity) falls below --validation-frac, so it keeps its side
     across rounds and cells. With no validation GEMMs (or
     --validation-frac 0) training and checkpoint selection use every GEMM.
  2. LightGBM LambdaRank teacher on the training GEMMs.
  3. GenericTwoTower student, in mini-batches of --batch-gemms GEMMs for
     --epochs passes, with a warmup and cosine (or step) learning rate:
         loss = lam * KL(teacher_T || student_T) * T^2 + (1 - lam) * soft_CE
     per GEMM over all its candidates. The targets of soft_CE and the
     teacher's top relevance grade treat every candidate within
     --tie-tolerance of the GEMM's winner as a winner.
  4. --eval-points checkpoints are measured as deployed: the kernel
     whitelist ("smart_K") is chosen greedily with the student in the loop
     on the training GEMMs, through the engine's LDS gate and feasibility
     rule with non-temporal availability of the library pool, and the
     engine ranks the validation GEMMs at --weight-dtype under the model's
     serving options (lib/evaluate.py). The best checkpoint is kept
     (--patience stops early); --qat-frac continues from it with the
     weights rounded as --weight-dtype stores them. With --plan (and
     --context-plan) a GEMM benched from a kernel list knows its list, and
     a pick off it is a miss charged by --pick-measurement, as stage04b and
     the deploy gate charge it; without, the evaluation passes over every
     ranked kernel without a row, so it scores the first measured one.
  5. Deployed evaluation through the engine at --weight-dtype on training
     and validation GEMMs; the Origami baseline is the bench's Origami pick
     (`is_origami_pick`).

That is --recipe per_leaf. --recipe pooled_ft replaces steps 2 to 4
(`run_pooled_ft`): one LightGBM teacher and one student learn the training
GEMMs of every cell together, the student's checkpoints measured on every
cell's validation GEMMs, and each cell to train fine-tunes that student on
its own GEMMs, keeping it unless an epoch beats it on the cell's validation
GEMMs.

With --whitelist-tier-k K (v3) every cell of the finished bundle, carried
cells included, then gets a whitelist tier of K view keys chosen on the
training side of the enriched data with the deployed model's scores
(`whitelist_tier`, lib/whitelist_tier.py); the cascade serves problems none
of a cell's tier keys suits. The weights are those trained without it.

Selection efficiency (sel_eff) = winner_us / pick_us, geometric mean over
GEMMs; 1.0 picks the fastest benched kernel every time. The validation GEMMs
also choose the checkpoint and the whitelist fallback, so the validation
numbers here are model-selection scores; stage04b measures on held-out
shapes.

Every measured solution of a GEMM is one candidate (one training pair) with
its own latency and target. Solutions with identical kernel parameters get
identical features and scores, and the engine ranks the first of them in
pool order first; each GEMM's candidates are kept in pool order so that the
training-time picks break such ties the same way. With --served-candidates
(v3) a GEMM keeps one candidate per key instead, the solution the runtime
runs for it (`served_candidates`). Whitelists stay per kernel signature.
With --library-stem, a pool of kernels whose operand the tensor data mover
loads, whose temporal hint does not reach the code, stops the stage with 1
before it reads any GEMM when the model would serve the non-temporal rules
or read such a hint as an attribute (`lib.kernel_identity.tdm_conflicts`).

Cells this round does not train keep their model from the latest
--prior-round-dir. A split parent's model stays in the bundle while a leaf
under it has no model of its own (`fallback_parents`): the engine serves such
a leaf with its nearest trained ancestor.

Before it reads any GEMM, stage05 packs an untrained one-cell model of the
format it trains (catalog, attribute spec, serving options, clip and
--weight-dtype) and loads it with the tilewright module: a module that
cannot load it, or reads it back otherwise than written, stops the stage at
once, naming the module and the format it was asked for.

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
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

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

from lib import audit as auditlib  # noqa: E402
from lib import bank_pool  # noqa: E402
from lib import boundaries  # noqa: E402
from lib import correctness as cx  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import feasibility  # noqa: E402
from lib import features as fs  # noqa: E402
from lib import grid  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import hipblaslt_launch  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import launch_settings  # noqa: E402
from lib import miss_rebench as mr  # noqa: E402
from lib import procs  # noqa: E402
from lib import provenance as prov  # noqa: E402
from lib import repeats  # noqa: E402
from lib import run_analysis as ra  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib import whitelist_tier as wt  # noqa: E402
from lib.dat import KERNELS_FILE, read_kernel_attributes, sig_from_row  # noqa: E402
from lib.fs import write_json  # noqa: E402

EXIT_CELLS_NOT_TRAINED = 3

DEFAULT_BATCH_GEMMS = 16
DEFAULT_LR = 2e-3
DEFAULT_EVAL_POINTS = 40
DEFAULT_PATIENCE = 0
DEFAULT_STUDENT_TEMPERATURE = "learned"
DEFAULT_QAT_FRAC = 0.0
# Equal to the tie tolerance lib/evaluate.py summarizes with by default, so
# that training and evaluation agree on what a tie is.
DEFAULT_TIE_TOLERANCE = 0.01

PSEUDO_NEGATIVES = ("none", "random", "targeted")
# --unmeasured-samples, --unmeasured-margin and --unmeasured-weight of each
# --pseudo-negatives draw when not given.
_UNTARGETED = {
    "unmeasured_samples": 0,
    "unmeasured_margin": 0.05,
    "unmeasured_weight": 0.3,
}
PSEUDO_NEGATIVE_DEFAULTS: Dict[str, Dict[str, Union[int, float]]] = {
    "none": _UNTARGETED,
    "random": _UNTARGETED,
    "targeted": {
        "unmeasured_samples": 16,
        "unmeasured_margin": 1.0,
        "unmeasured_weight": 0.1,
    },
}

# Environment knobs that change the trained artifact; metrics.json records
# their values.
ARTIFACT_ENV_KNOBS: Tuple[str, ...] = (
    "ML_CAPACITY_MODE",
    "ML_FORCE_CAPACITY",
    "ML_LARGE_CELL_CAPACITY",
    "ML_DROPOUT",
    "ML_SMARTK_RUNTIME_WEIGHT",
)

MODEL_OPTIONS_FILE = "model_options.json"
MEMBER_ORDER_FILE = ki.MEMBER_ORDER_FILE
# What the serving options `lib.kernel_identity.tdm_conflicts` names are
# called on this command line (the non-temporal rule is v2's default).
TDM_CONFLICT_OPTIONS = {
    "non_temporal_rule": "--non-temporal-rule (v2: without --no-non-temporal-rule)",
    "non_temporal_family_rule": "--non-temporal-family-rule",
    "attribute_spec": "the attribute spec",
}
# Arguments that do not shape what a cell's model is trained for or how the
# bundle serves it; every other argument does (`model_options`).
NOT_MODEL_SHAPING = frozenset(
    {
        # the data, its validation split, and the library and device it was
        # measured on
        "enriched_csv_dir",
        "prior_round_dir",
        "current_round_dir",
        "plan",
        "context_plan",
        "off_list_rebench",
        "config_yaml",
        "validation_frac",
        "arch",
        "hardware_device",
        "library_dir",
        "library_stem",
        # which cells train
        "only_cells",
        "skip_cells",
        "min_cell_gemms",
        "min_base_cell_gemms",
        # how the work runs
        "output_dir",
        "device",
        "train_workers",
        "train_threads_per_worker",
        "seed",
        "quiet",
        # the optimizer's budget and schedule: how closely a cell fits its
        # objective, not which objective
        "epochs",
        "batch_gemms",
        "lr",
        "lr_schedule",
        "warmup_frac",
        "lr_min_frac",
        "eval_points",
        "eval_sample_frac",
        "patience",
        "weight_decay",
        "grad_clip_norm",
        "qat_lr_frac",
        "n_estimators",
        "num_leaves",
        "lgb_lr",
        "pooled_epochs",
        "pooled_batch_gemms",
        "pooled_lr",
        "pooled_eval_points",
        "pooled_n_estimators",
        "pooled_num_leaves",
        "pooled_min_data_in_leaf",
        "fine_tune_epochs",
        "fine_tune_lr",
        # how the pooled recipe's work runs
        "pooled_seed",
        "pooled_threads",
        "pooled_teacher_threads",
    }
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


def config_tie_tolerance(cfg: Mapping[str, Any]) -> float:
    """The tie tolerance evaluations of the run use (the config's
    `stage08.tie_tolerance`), else DEFAULT_TIE_TOLERANCE."""
    stage08 = cfg.get("stage08") if isinstance(cfg, Mapping) else None
    value = (
        (stage08 or {}).get("tie_tolerance") if isinstance(stage08, Mapping) else None
    )
    return DEFAULT_TIE_TOLERANCE if value is None else float(value)


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


def labels_blocklist(
    blocklist: Optional[cx.Blocklist], folders: Sequence[os.PathLike]
) -> Dict[str, Any]:
    """The blocklists that filtered the training labels: the one this stage
    applied (`lib.correctness.applied_record`) and, as `enriched`, the
    digests of those the stage04 of the enriched `folders` applied (null for
    one that blocked nothing), a folder enriched before stage04 recorded it
    counted in `unknown`."""
    enriched: set = set()
    unknown = 0
    for fd in folders:
        try:
            doc = json.loads((Path(fd) / "enrich_summary.json").read_text())
        except (OSError, ValueError):
            doc = None
        rec = doc.get("blocklist") if isinstance(doc, dict) else None
        if isinstance(rec, dict):
            enriched.add(rec.get("digest"))
        else:
            unknown += 1
    return {
        **cx.applied_record(blocklist),
        "enriched": sorted(enriched, key=lambda d: (d is not None, d or "")),
        "unknown": unknown,
    }


def bench_launch(folders: Sequence[os.PathLike]) -> Dict[str, Any]:
    """How the benches of the enriched `folders` launched the kernels: the
    launch summaries their stage04 recorded, merged
    (lib/launch_settings.py)."""
    found: List[Optional[Mapping[str, Any]]] = []
    for fd in folders:
        try:
            doc = json.loads((Path(fd) / "enrich_summary.json").read_text())
        except (OSError, ValueError):
            doc = None
        rec = doc.get("launch") if isinstance(doc, dict) else None
        found.append(rec if isinstance(rec, dict) else None)
    return launch_settings.summarize(found)


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


def _float_or_none(text: Any) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _row_context(row: Mapping[str, Any]) -> fs.ExecutionContext:
    """Execution context of an enriched row; rows of CSVs without the
    context columns are exclusive."""
    return ev.context_of(row)


def _gemm_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    """The GEMM a row measures: problem identity, plus its execution context
    when that is not exclusive."""
    ctx = _row_context(row)
    extra = () if ctx.is_exclusive() else (ctx.cu_budget, ctx.schedule)
    return _problem_key(row) + extra


def _problem_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
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
                    keys.add(_problem_key(row))
    out: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for key in sorted(keys):
        m, n, k, b = (int(v) for v in key[:4])
        out[_leaf_of(m, n, k, b, split_tree)].append((m, n, k, b))
    return dict(out)


def load_enriched_chunks(
    folder,
    quiet: bool,
    split_tree: Optional[Dict[str, Any]] = None,
    identity: Optional[ki.LibraryIdentity] = None,
    aggregation: str = repeats.DEFAULT_AGGREGATION,
    blocklist: Optional[cx.Blocklist] = None,
) -> Tuple[Dict[str, List[dict]], dict]:
    """Read chunk_*.csv files of one folder or a list of folders (their union)
    and group the rows by GEMM, then by leaf cell (`split_tree` routes base
    cells to sub-cells).

    Tested and skip rows both become candidates, one per solution: solutions
    with identical kernel parameters stay separate candidates. A row without
    a solution index gets the synthetic id -(rank + 1). Rows of one solution
    index of a GEMM (several rounds, a retried problem) merge into one by
    `aggregation` (lib/repeats.py: by default the median of its sustained
    rows if it has any (stage03d), else of its tested rows; a GEMM with
    sustained rows scales its other kernels to them); the GEMM's winner is
    its fastest tested solution (skip-row latencies come from one warm-up
    run). A tested candidate keeps its
    reading's `cv` and `status` (lib/bench_log.py adaptive statistics).
    `is_origami_pick` is None when the CSV has no such column. A candidate
    carries the kernel's `attributes` when a folder's kernels.json lists its
    solution index. Rows of one problem under different execution contexts
    (`cu_budget`, `schedule` columns) are different GEMMs; a GEMM under a
    non-exclusive context carries them.
    With `identity` (the library the GEMMs are joined with), a folder whose
    kernels.json records kernel ids of another hipBLASLt build is read with
    its solution indices mapped to the library's, and a folder without a
    record is checked against it by engine config and attributes
    (`ki.folder_remap`); rows of a kernel the library does not hold under
    one index, or that differs from the library's kernel at its index, are
    left out (`rows_kernel_not_in_library`).
    Rows of a kernel `blocklist` names (lib/correctness.py), by its kernel id
    in `identity`'s library or, without `identity`, in its folder's record,
    are left out (`rows_blocked`), as stage04 leaves them out of folders it
    enriched with that blocklist.

    Returns ({cell: [gemm dict]}, counters)."""
    repeats.check(aggregation)
    t0 = time.time()
    folders, chunk_files = _chunk_files(folder)
    if not chunk_files:
        raise SystemExit(f"no chunk_*.csv in any of {folders}")

    stats: Dict[str, int] = defaultdict(int)
    stats["n_source_dirs"] = len(folders)
    attributes: Dict[int, Dict[str, int]] = {}
    remaps: Dict[str, ki.Remap] = {}
    blocked: Dict[str, Set[int]] = {}
    in_library = (
        blocklist.blocked_indices(identity.ids)
        if blocklist and identity is not None
        else set()
    )
    for fd in folders:
        doc = None
        if identity is not None or blocklist:
            try:
                doc = ev.kernels_json(fd)
            except ValueError as e:
                ui.warn("load", f"{e}; its kernels are checked by their rows alone")
        try:
            remaps[str(fd)] = remap = ki.folder_remap(doc, identity, str(fd))
        except ValueError as e:
            raise SystemExit(f"cannot read the kernel identities of {fd}: {e}")
        if blocklist and identity is None:
            try:
                ids = ki.recorded_ids(ki.read_record(doc)) or {}
            except ValueError:
                ids = {}
            blocked[str(fd)] = blocklist.blocked_indices(ids)
        else:
            blocked[str(fd)] = in_library
        path = os.path.join(str(fd), KERNELS_FILE)
        if not os.path.isfile(path):
            continue
        try:
            listed = read_kernel_attributes(path)
        except ValueError as e:
            ui.warn("load", f"{e}; its kernels carry no attributes")
            continue
        for old, attrs in listed.items():
            sid = remap.index(old)
            if sid is not None and attributes.setdefault(sid, attrs) != attrs:
                stats["kernel_attribute_conflicts"] += 1
    if stats["kernel_attribute_conflicts"]:
        ui.warn(
            "load",
            f"{stats['kernel_attribute_conflicts']} solution(s) have different "
            f"attributes in different {KERNELS_FILE} files; the first is used",
        )
    gemms_by_key: Dict[Tuple, dict] = {}
    for fi, (fd, fname) in enumerate(chunk_files, 1):
        remap = remaps[str(fd)]
        out_of_pool = blocked[str(fd)]
        with open(os.path.join(fd, fname)) as f:
            reader = csv.DictReader(f)
            has_origami_col = "is_origami_pick" in (reader.fieldnames or [])
            for row in reader:
                stats["rows_total"] += 1
                rejected = _row_rejection(row)
                if rejected is not None:
                    stats[rejected] += 1
                    continue
                sig_int = remap.row_index(_solution_id(row), row)
                if sig_int is None:
                    stats["rows_kernel_not_in_library"] += 1
                    continue
                if sig_int in out_of_pool:
                    stats["rows_blocked"] += 1
                    continue
                is_skip = row.get("is_skip", "False") == "True"
                us = float(row["us"])
                rank_val = int(row.get("rank", -1) or -1)
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
                    if len(key) > 11:
                        g["cu_budget"], g["schedule"] = key[11], key[12]
                        stats["gemms_non_exclusive"] += 1
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
                if not is_skip and row.get("status"):
                    cand["status"] = sys.intern(row["status"])
                    cand["cv"] = _float_or_none(row.get("cv"))
                if row.get("is_sustained") == "True":
                    cand["sustained"] = True
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

    for remap in remaps.values():
        remap.report()
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
        n_rows = len(g["candidates"])
        g["candidates"] = repeats.merge_candidates(g["candidates"], aggregation)
        if len(g["candidates"]) < n_rows:
            stats["rows_merged"] += n_rows - len(g["candidates"])
            stats["kernels_repeated"] += sum(
                1 for c in g["candidates"] if c.get("n_repeats", 1) > 1
            )
        if any(c.get("sustained") for c in g["candidates"]):
            stats["gemms_sustained"] += 1
            g["sustained_ratio"] = repeats.sustained_ratio(g["candidates"])
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


def member_order(
    library: Sequence[Mapping[str, Any]],
    catalog: Optional[fs.Catalog],
    v3: bool,
    launch_keys: bool,
    arch: Optional[str],
    identity: Optional[ki.LibraryIdentity],
) -> Dict[str, Any]:
    """The library digest and, per candidate key with several members (the
    model's grouping: `lib.audit.candidate_keys`), a digest of its members'
    kernel ids in pool order (`ki.member_orders`). The runtime serves a
    key's first member whose predicates pass, the member served candidates
    train on, so a build that orders them otherwise serves other kernels."""
    header = {
        "version": 3 if v3 else 2,
        "attribute_spec": catalog.spec if v3 and catalog is not None else (),
        "serving": {"launch_keys": bool(launch_keys)},
        "arch": arch,
    }
    key_of, _ = auditlib.candidate_keys(library, header)
    return {
        "library_digest": identity.digest if identity is not None else None,
        "keys": ki.member_orders(library, lambda k: key_of[int(k["sol_idx_global"])]),
    }


def target_winner_us(gemm: Mapping[str, Any]) -> float:
    """The latency a GEMM's training targets are relative to: the fastest
    candidate it keeps (`served_candidates`), else its winner."""
    return float(gemm.get("target_winner_us", gemm["winner_us"]))


def served_key_of(
    catalog: fs.Catalog, launch_keys: bool
) -> Callable[[Mapping[str, Any]], Tuple[int, ...]]:
    """The candidate key the engine groups a kernel by for a v3 model: its
    view key, plus its launch key when the model serves launch keys."""
    launch = catalog.launch if launch_keys else None

    def key_of(kernel: Mapping[str, Any]) -> Tuple[int, ...]:
        return fs.v3_candidate_key(
            fs.config_kwargs_from_row(kernel),
            kernel.get("attributes"),
            catalog.spec,
            launch,
        )

    return key_of


def served_candidates(
    gemms: Sequence[Dict[str, Any]],
    pools: Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]],
    key_of: Callable[[Mapping[str, Any]], Tuple[int, ...]],
) -> Dict[str, int]:
    """Keep, of each GEMM's candidates, the kernel the runtime runs for each
    candidate key: the key's first pool kernel with a row, passing over pool
    kernels the GEMM was benched with but has no row of (their predicates
    fail). A key whose first such kernel is off the GEMM's planned list was
    not measured as served, and none of its candidates is kept. Candidates
    outside the pool are kept. Every GEMM gets `target_winner_us`, the
    fastest tested latency it keeps (else its fastest kept latency); a GEMM
    keeping no candidate keeps them all. Returns counters."""
    members: Dict[Tuple[str, ...], Dict[Tuple[int, ...], List[int]]] = {}
    keys: Dict[Tuple[str, ...], Dict[int, Tuple[int, ...]]] = {}
    for pk, pool in pools.items():
        by_key: Dict[Tuple[int, ...], List[int]] = {}
        key_by_sid: Dict[int, Tuple[int, ...]] = {}
        for k in pool:
            ck = key_of(k)
            sid = int(k["sol_idx_global"])
            by_key.setdefault(ck, []).append(sid)
            key_by_sid[sid] = ck
        members[pk], keys[pk] = by_key, key_by_sid
    stats: Dict[str, int] = defaultdict(int)
    for g in gemms:
        pk = ev.pool_key(g)
        key_by_sid = keys.get(pk, {})
        by_key = members.get(pk, {})
        listed = g.get("listed")
        by_sid = {int(c["sol_idx_global"]): c for c in g["candidates"]}
        kept: List[Dict[str, Any]] = []
        done: set = set()
        for c in g["candidates"]:
            sid = int(c["sol_idx_global"])
            ck = key_by_sid.get(sid)
            if ck is None:
                kept.append(c)
                continue
            if ck in done:
                continue
            done.add(ck)
            for member in by_key[ck]:
                if member in by_sid:
                    kept.append(by_sid[member])
                    break
                if listed is not None and member not in listed:
                    stats["keys_unmeasured"] += 1
                    break
        stats["candidates_in"] += len(g["candidates"])
        if not kept:
            stats["gemms_without_served"] += 1
            kept = list(g["candidates"])
        stats["candidates_kept"] += len(kept)
        g["candidates"] = kept
        tested = [c["us"] for c in kept if not c.get("is_skip")]
        g["target_winner_us"] = min(tested) if tested else min(c["us"] for c in kept)
    return dict(stats)


# ── student model ─────────────────────────────────────────────────────────────

# The engine divides the embedding term by max(|T|, MIN_TEMPERATURE).
MIN_TEMPERATURE = 0.1
QUANTIZED_WEIGHT_LEVELS = {"int8": 127.0, "int4": 7.0}

if _TORCH_OK:
    import torch.nn.functional as F

    def fake_quant_weight(w: torch.Tensor, weight_dtype: str) -> torch.Tensor:
        """`w` rounded the way the model file stores a weight matrix at
        `weight_dtype` (lib/mlrec.py: one symmetric scale per tensor for
        int8 / int4, round to nearest even for bf16), with the gradient of
        `w` itself (straight-through)."""
        levels = QUANTIZED_WEIGHT_LEVELS.get(weight_dtype)
        wd = w.detach()
        if levels is not None:
            amax = wd.abs().max()
            scale = torch.where(amax > 0, amax / levels, torch.ones_like(amax))
            q = torch.clamp(torch.round(wd / scale), -levels, levels) * scale
        elif weight_dtype == "bf16":
            q = wd.to(torch.bfloat16).to(w.dtype)
        else:
            return w
        return q + (w - wd)

    class GenericTwoTower(nn.Module):
        """score = dot(q_proj(query), i_proj(item)) / max(T, 0.1)
        + inter_mlp(interaction). Tensor names (q_proj.0/2/4, i_proj.0/2,
        inter_mlp.0/2) are the model file's weight order, so dropout is
        functional and never a module. T = exp(log_temperature), learned
        unless `learn_temperature` is false; `export_state` and
        `load_exported` convert to and from the `temperature` entry the
        model file stores. With `fake_quant` set to a weight dtype the
        weight matrices are rounded as the model file stores them
        (`fake_quant_weight`)."""

        def __init__(
            self,
            q_dim: int,
            i_dim: int,
            x_dim: int,
            embed_dim: int = 32,
            hidden_dim: int = 64,
            inter_hidden: int = 16,
            dropout: float = 0.0,
            learn_temperature: bool = True,
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
            self.log_temperature = nn.Parameter(
                torch.zeros(()), requires_grad=bool(learn_temperature)
            )
            self.dropout_p = float(dropout)
            self.fake_quant: Optional[str] = None

        def temperature(self) -> torch.Tensor:
            return self.log_temperature.exp().clamp(min=MIN_TEMPERATURE)

        def _run(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    w = layer.weight
                    if self.fake_quant is not None:
                        w = fake_quant_weight(w, self.fake_quant)
                    x = F.linear(x, w, layer.bias)
                else:
                    x = layer(x)
            return x

        def score_grouped(
            self,
            q_rows: torch.Tensor,
            q_index: torch.Tensor,
            i_rows: torch.Tensor,
            i_index: torch.Tensor,
            x_per_pair: torch.Tensor,
        ) -> torch.Tensor:
            """(N_pairs,) scores of pairs whose query and item features are
            rows `q_index` of `q_rows` and `i_index` of `i_rows`: each tower
            runs once per distinct row."""
            dot, ix = self.score_parts_grouped(
                q_rows, q_index, i_rows, i_index, x_per_pair
            )
            return dot + ix

        def score_parts_grouped(
            self,
            q_rows: torch.Tensor,
            q_index: torch.Tensor,
            i_rows: torch.Tensor,
            i_index: torch.Tensor,
            x_per_pair: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """`score_grouped` as (embedding term, interaction term): the
            cascade's first stage ranks by the embedding term alone."""
            eg = self._run(self.q_proj, q_rows).index_select(0, q_index)
            et = self._run(self.i_proj, i_rows).index_select(0, i_index)
            xin = x_per_pair
            p = self.dropout_p
            if p > 0 and self.training:
                eg = F.dropout(eg, p=p, training=True)
                et = F.dropout(et, p=p, training=True)
                xin = F.dropout(xin, p=p, training=True)
            dot = (eg * et).sum(dim=-1) / self.temperature()
            ix = self._run(self.inter_mlp, xin).squeeze(-1)
            return dot, ix

        def score_pairs(
            self,
            q_per_pair: torch.Tensor,
            i_per_pair: torch.Tensor,
            x_per_pair: torch.Tensor,
        ) -> torch.Tensor:
            """Inputs broadcast to (N_pairs, dim); returns (N_pairs,) scores."""
            idx = torch.arange(q_per_pair.shape[0], device=q_per_pair.device)
            return self.score_grouped(q_per_pair, idx, i_per_pair, idx, x_per_pair)

        def export_state(self) -> Dict[str, torch.Tensor]:
            """The weights as a models.pt entry and the model file hold
            them: the WEIGHT_ORDER tensors and `temperature`."""
            sd = {
                k: v.detach().cpu().clone()
                for k, v in self.state_dict().items()
                if k != "log_temperature"
            }
            sd["temperature"] = self.temperature().detach().cpu().reshape(1).clone()
            return sd

        def load_exported(self, state: Mapping[str, Any]) -> None:
            """Load an `export_state` dict; a missing `temperature` means 1,
            and any value is taken as the engine takes it, max(|T|, 0.1)."""
            sd = {k: v for k, v in state.items() if k != "temperature"}
            t = state.get("temperature")
            t_val = 1.0 if t is None else float(torch.as_tensor(t).reshape(-1)[0])
            t_val = max(abs(t_val), MIN_TEMPERATURE)
            sd["log_temperature"] = torch.tensor(math.log(t_val))
            self.load_state_dict(sd)


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
    s.load_exported(entry["state_dict"])
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
    catalog: fs.Catalog = fs.V2_CATALOG,
) -> Tuple[Any, ...]:
    """(q_per_gemm, i_per_pair, x_per_pair, gemm_idx_for_pair, us_per_pair,
    winner_us_per_pair, rank_us_per_pair) of a cell's GEMMs, under each
    GEMM's execution context."""
    q_per_gemm: List[List[float]] = []
    i_per_pair: List[List[float]] = []
    x_per_pair: List[List[float]] = []
    gemm_idx_for_pair: List[int] = []
    us_per_pair: List[float] = []
    winner_us_per_pair: List[float] = []
    rank_us_per_pair: List[int] = []
    for gi, g in enumerate(gemms):
        prob = fs.problem_kwargs_from_row(g)
        context = ev.context_of(g)
        q, _, _ = catalog.vectors(
            prob,
            fs.config_kwargs_from_row(g["candidates"][0]),
            hardware=hardware,
            constants=constants,
            context=context,
        )
        q_per_gemm.append(q)
        for c in g["candidates"]:
            _, i_vec, x_vec = catalog.vectors(
                prob,
                fs.config_kwargs_from_row(c),
                c.get("attributes"),
                hardware=hardware,
                constants=constants,
                context=context,
            )
            i_per_pair.append(i_vec)
            x_per_pair.append(x_vec)
            gemm_idx_for_pair.append(gi)
            us_per_pair.append(c["us"])
            winner_us_per_pair.append(target_winner_us(g))
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


def feature_bounds(
    q: np.ndarray, i: np.ndarray, x: np.ndarray
) -> Dict[str, List[float]]:
    """Per-feature [min, max] of raw training features: the clip bounds an
    MLREC_v3 model applies before whitening."""
    out: Dict[str, List[float]] = {}
    for tower, a in (("q", q), ("i", i), ("x", x)):
        a32 = np.asarray(a, dtype=np.float32)
        out[f"{tower}_lo"] = a32.min(axis=0).tolist()
        out[f"{tower}_hi"] = a32.max(axis=0).tolist()
    return out


def clip_features(
    q: np.ndarray,
    i: np.ndarray,
    x: np.ndarray,
    bounds: Optional[Mapping[str, Sequence[float]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw features clamped to `bounds` as the engine clamps them (float32
    min(max(v, lo), hi)); unchanged without bounds."""
    if not bounds:
        return q, i, x
    out = []
    for tower, a in (("q", q), ("i", i), ("x", x)):
        lo = np.asarray(bounds[f"{tower}_lo"], dtype=np.float32)
        hi = np.asarray(bounds[f"{tower}_hi"], dtype=np.float32)
        out.append(np.minimum(np.maximum(np.asarray(a, dtype=np.float32), lo), hi))
    return out[0], out[1], out[2]


def apply_whiten(
    q: np.ndarray,
    i: np.ndarray,
    x: np.ndarray,
    norms: Dict[str, List[float]],
    bounds: Optional[Mapping[str, Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten with stored statistics (`whiten_features`), after clamping to
    `bounds` when given."""
    q, i, x = clip_features(q, i, x, bounds)
    return (
        whiten_features(q, norms["q_mean"], norms["q_std"], "q"),
        whiten_features(i, norms["i_mean"], norms["i_std"], "i"),
        whiten_features(x, norms["x_mean"], norms["x_std"], "x"),
    )


# ── LightGBM teacher ──────────────────────────────────────────────────────────

RELEVANCE_LOG_BINS = (0.05, 0.10, 0.20, 0.40, 0.80)


def relevance_labels(
    us: np.ndarray, winner_us: np.ndarray, tie_tolerance: float
) -> np.ndarray:
    """LambdaRank relevance per pair: the top grade for every candidate
    within `tie_tolerance` (relative) of its GEMM's winner, then one grade
    per log-spaced bucket of log(us / winner_us) (RELEVANCE_LOG_BINS beyond
    the tolerance), 0 beyond the last."""
    tol = math.log1p(max(float(tie_tolerance), 0.0))
    bins = np.asarray([tol] + [b for b in RELEVANCE_LOG_BINS if b > tol * 1.000001])
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    rel = len(bins) - np.searchsorted(bins, log_ratio, side="right")
    return np.clip(rel, 0, len(bins)).astype(np.int32)


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
    tie_tolerance: float = math.expm1(RELEVANCE_LOG_BINS[0]),
    min_data_in_leaf: Optional[int] = None,
    num_threads: Optional[int] = None,
) -> Any:
    """LambdaRank teacher; one ranking group per GEMM, graded by
    `relevance_labels`. `min_data_in_leaf` None keeps LightGBM's default;
    `num_threads` None takes torch's thread count."""
    if not _LGB_OK:
        raise SystemExit("lightgbm not installed; pip install -r requirements.txt")
    X = np.concatenate([qvec[gemm_idx], ivec, xvec], axis=1)
    rel = relevance_labels(us, winner_us, tie_tolerance)
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
        # The same data, seed and thread count give the same trees; another
        # thread count can give other trees.
        "deterministic": True,
        "seed": int(seed),
        # Same thread budget as torch; unset, LightGBM takes every core and
        # oversubscribes the parallel training pool.
        "num_threads": max(1, int(num_threads or torch.get_num_threads())),
    }
    if min_data_in_leaf is not None:
        params["min_data_in_leaf"] = int(min_data_in_leaf)
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
    per GEMM: every candidate within `target_tolerance` (log ratio) of the
    winner gets the same, largest share."""
    log_ratio = np.log(np.maximum(us, 1e-9) / np.maximum(winner_us, 1e-9))
    adj = np.maximum(log_ratio - target_tolerance, 0.0)
    logits = -alpha * adj
    gidx = np.asarray(gemm_idx, dtype=np.int64)
    n = int(gidx.max()) + 1 if gidx.size else 0
    top = np.full(n, -np.inf)
    np.maximum.at(top, gidx, logits)
    e = np.exp(logits - top[gidx])
    total = np.bincount(gidx, weights=e, minlength=n)
    return e / np.maximum(total[gidx], 1e-12)


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
    benched candidates, on the GEMMs selected by `sample_set_mask`
    (`sel_eff_of_scores`)."""
    student.eval()
    with torch.no_grad():
        q_per_pair = q_t.index_select(0, gemm_idx_t)
        scores = student.score_pairs(q_per_pair, i_t, x_t)
    student.train()
    return sel_eff_of_scores(
        scores, gemm_idx_t, us_t, winner_us_t, sample_set_mask, q_t.shape[0]
    )


def sel_eff_of_scores(
    scores: torch.Tensor,
    gemm_idx_t: torch.Tensor,
    us_t: torch.Tensor,
    winner_us_t: torch.Tensor,
    sample_set_mask: torch.Tensor,
    n_gemms: int,
) -> Dict[str, float]:
    """Geomean winner_us / pick_us of the top-scored candidate of each GEMM
    selected by `sample_set_mask`. Ties go to the lowest pair index: the
    first in pool order, as in the engine, when each GEMM's candidates are
    in pool order."""
    device = scores.device
    n_pairs = gemm_idx_t.shape[0]
    pair_in_sample = sample_set_mask.index_select(0, gemm_idx_t)
    with torch.no_grad():
        scores_masked = scores.detach().clone()
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
    gemms: List[Dict[str, Any]], k: int = 10, key_of: Optional[Any] = None
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
            sig = key_of(c) if key_of else tuple(int(v) for v in sig_from_row(c))
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
    non_temporal_rule: bool = True,
    *,
    catalog: fs.Catalog = fs.V2_CATALOG,
    rules: Optional[Mapping[str, bool]] = None,
    bounds: Optional[Mapping[str, Sequence[float]]] = None,
    key_of: Optional[Any] = None,
    nt_families: Optional[Mapping[Tuple[str, ...], Any]] = None,
) -> Tuple[Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]], Dict[int, float]]:
    """({sig: {gi: (us, student score)}}, {gi: winner_us}) over the
    (kernel, GEMM) pairs that pass the engine's LDS gate and feasibility rule
    (`nt_flags`: per pool key, non-temporal availability of the library, and
    `nt_families` its kernel families; `rules`: the soft-rule keywords of
    `feasibility.passes_gates`, default v2's with `non_temporal_rule`). A
    signature's entry for a GEMM is its
    fastest passing solution (the first in candidate order among equally fast
    ones) with that solution's score. `key_of(candidate)` replaces the
    signature as the whitelist key (MLREC_v3 view keys)."""
    if rules is None:
        rules = {"non_temporal_rule": non_temporal_rule}
    if key_of is None:
        key_of = lambda c: tuple(int(v) for v in sig_from_row(c))  # noqa: E731
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
            context = ev.context_of(g)
            nt_a, nt_b = nt_flags.get(ev.pool_key(g), (False, False))
            families = (nt_families or {}).get(ev.pool_key(g))
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
                    nt_families=families,
                    **rules,
                ):
                    continue
                q, ivec, xvec = catalog.vectors(
                    prob,
                    cfg,
                    c.get("attributes"),
                    hardware=hardware,
                    constants=constants,
                    context=context,
                )
                if q_vec is None:
                    q_vec = q
                i_rows.append(ivec)
                x_rows.append(xvec)
                us_list.append(us)
                sig_list.append(key_of(c))
            if not us_list or q_vec is None:
                continue
            winner_us[gi] = w
            n = len(us_list)
            qa, ia, xa = apply_whiten(
                np.asarray([q_vec] * n, dtype=np.float32),
                np.asarray(i_rows, dtype=np.float32),
                np.asarray(x_rows, dtype=np.float32),
                norms,
                bounds,
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


class ScoreTable:
    """Whitelist keys x evaluated GEMMs: the latency (`us`, inf where the
    key has no feasible measured solution for the GEMM) and student score
    (`score`, -inf there) of each key's fastest feasible solution, with the
    GEMMs' log winner latencies. Keys keep their first-seen order, GEMMs
    their order in the cell."""

    def __init__(
        self,
        keys: List[Tuple[int, ...]],
        gemms: List[int],
        us: np.ndarray,
        score: np.ndarray,
        winner_us: np.ndarray,
    ) -> None:
        self.keys = list(keys)
        self.gemms = list(gemms)
        self.us = np.asarray(us, dtype=np.float64).reshape(len(keys), len(gemms))
        self.score = np.asarray(score, dtype=np.float64).reshape(self.us.shape)
        self.has = np.isfinite(self.us)
        self.winner_us = np.asarray(winner_us, dtype=np.float64)
        self.log_win = np.log(self.winner_us)
        self.log_us = np.log(np.where(self.has, self.us, 1.0))
        self.row = {key: j for j, key in enumerate(self.keys)}

    @classmethod
    def from_cache(
        cls,
        sig_to_data: Mapping[Tuple[int, ...], Mapping[int, Tuple[float, float]]],
        winner_us: Mapping[int, float],
    ) -> "ScoreTable":
        """Table of a `_student_score_cache` result."""
        keys, gemms = list(sig_to_data), list(winner_us)
        col = {gi: c for c, gi in enumerate(gemms)}
        us = np.full((len(keys), len(gemms)), np.inf)
        score = np.full((len(keys), len(gemms)), -np.inf)
        for j, key in enumerate(keys):
            for gi, (u, s) in sig_to_data[key].items():
                us[j, col[gi]], score[j, col[gi]] = u, s
        return cls(keys, gemms, us, score, [winner_us[gi] for gi in gemms])

    @property
    def empty(self) -> bool:
        return not self.keys or not self.gemms


def _greedy_table(table: ScoreTable, k: int) -> List[int]:
    """Two-stage greedy list builder over `table` (rows of the chosen keys).

    Stage 1 (coverage): until every GEMM has a feasible key in the list, add
    the key covering the most uncovered GEMMs (ties by regret gain), so no
    training-distribution GEMM is left without a whitelisted candidate.
    Stage 2 (regret): fill the remaining slots with the key that most
    improves mean log(winner_us / student_pick_us). The student picks the
    best-scored listed key of a GEMM (the earlier listed on equal scores);
    ties between keys go to the first in table order."""
    if table.empty:
        return []
    n_keys, n_gemms = table.us.shape
    has, score, log_us, log_win = table.has, table.score, table.log_us, table.log_win
    chosen: List[int] = []
    free = np.ones(n_keys, dtype=bool)
    best = np.full(n_gemms, -np.inf)
    log_pick = np.zeros(n_gemms)
    covered = np.zeros(n_gemms, dtype=bool)

    def take(j: int) -> None:
        upd = has[j] & (score[j] > best)
        best[upd] = score[j][upd]
        log_pick[upd] = log_us[j][upd]
        covered[upd] = True
        free[j] = False
        chosen.append(j)

    while len(chosen) < int(k) and not covered.all():
        new = has & ~covered
        better = has & covered & (score > best)
        gain = np.where(new, log_win - log_us, 0.0) + np.where(
            better, log_pick - log_us, 0.0
        )
        n_new = new.sum(axis=1)
        ok = free & (n_new > 0)
        if not ok.any():
            break
        top = ok & (n_new == n_new[ok].max())
        take(int(np.argmax(np.where(top, gain.sum(axis=1), -np.inf))))

    while len(chosen) < int(k):
        better = has & (score > best)
        valid = better | covered
        gain = np.where(valid, log_win - np.where(better, log_us, log_pick), 0.0)
        count = valid.sum(axis=1)
        ok = free & (count > 0)
        if not ok.any():
            break
        avg = gain.sum(axis=1) / np.maximum(count, 1)
        take(int(np.argmax(np.where(ok, avg, -np.inf))))
    return chosen


# With ML_SMARTK_RUNTIME_WEIGHT the knee weights each GEMM's log ratio by its
# winner_us, so dropping a signature that slows long-running GEMMs costs more.
_SMARTK_RT_WEIGHT = os.environ.get("ML_SMARTK_RUNTIME_WEIGHT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _evaluate_keys(
    table: ScoreTable, keys: Sequence[Tuple[int, ...]]
) -> Dict[str, Any]:
    """sel_eff when the student picks from each prefix of `keys` (keys
    outside `table` add nothing): the regret curve [(k_size, sel_eff,
    n_covered)] over the GEMMs the prefix covers."""
    if not keys or table.empty:
        return {"sel_eff": float("nan"), "n_eval": 0, "regret_curve": []}
    n_gemms = len(table.gemms)
    best = np.full(n_gemms, -np.inf)
    log_pick = np.zeros(n_gemms)
    covered = np.zeros(n_gemms, dtype=bool)
    weight = table.winner_us if _SMARTK_RT_WEIGHT else np.ones(n_gemms)
    regret_curve: List[Tuple[int, float, int]] = []
    seen: set = set()
    for k_size, key in enumerate(keys, start=1):
        j = table.row.get(key)
        if j is not None and key not in seen:
            seen.add(key)
            upd = table.has[j] & (table.score[j] > best)
            best[upd] = table.score[j][upd]
            log_pick[upd] = table.log_us[j][upd]
            covered[upd] = True
        n = int(covered.sum())
        if n == 0:
            regret_curve.append((k_size, float("nan"), 0))
            continue
        wsum = float(weight[covered].sum())
        total = float((weight * (table.log_win - log_pick))[covered].sum())
        sel_eff = math.exp(total / wsum) if wsum else float("nan")
        regret_curve.append((k_size, sel_eff, n))
    final = regret_curve[-1]
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
    sigs: List[Tuple[int, ...]],
    seeds: Sequence[Tuple[int, ...]],
    cap: int,
) -> List[Tuple[int, ...]]:
    """Prepend the oracle max-coverage signatures `seeds` (the kernels
    winning most GEMMs) missing from `sigs`, truncating to `cap`."""
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
    non_temporal_rule: bool = True,
    catalog: fs.Catalog = fs.V2_CATALOG,
    rules: Optional[Mapping[str, bool]] = None,
    bounds: Optional[Mapping[str, Sequence[float]]] = None,
    key_of: Optional[Any] = None,
    table: Optional[ScoreTable] = None,
    oracle_seeds: Optional[List[Tuple[int, ...]]] = None,
    with_metrics: bool = True,
    nt_families: Optional[Mapping[Tuple[str, ...], Any]] = None,
) -> Dict[str, Any]:
    """Kernel whitelist of one cell: {"sigs": [...], "metrics": {...}}.

    With a student (or its `table`, which `_student_score_cache` would
    otherwise build), the greedy list is built against the student's actual
    picks (`_greedy_table`); without one it is the oracle list. With `k_max`
    the list is grown to k_max and cut at the regret-curve knee in [k_min,
    k_max]; otherwise it has `k` entries. `oracle_seed` > 0 forces the
    cell's dominant oracle winners (`oracle_seeds` when precomputed) into
    the list. Without `with_metrics` the metrics hold only the list's
    construction (no oracle comparison)."""
    if table is None:
        if student is None or norms is None or q_names is None:
            return {
                "sigs": _select_smart_k_oracle(gemms, k=k, key_of=key_of),
                "metrics": {"mode": "oracle"},
            }
        assert hardware is not None and constants is not None
        table = ScoreTable.from_cache(
            *_student_score_cache(
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
                non_temporal_rule,
                catalog=catalog,
                rules=rules,
                bounds=bounds,
                key_of=key_of,
                nt_families=nt_families,
            )
        )
    if table.empty:
        return {
            "sigs": _select_smart_k_oracle(gemms, k=k, key_of=key_of),
            "metrics": {"mode": "oracle_fallback", "reason": "empty_cache"},
        }
    adaptive = k_max is not None
    k_build = int(k_max) if adaptive else int(k)
    sigs_full = [table.keys[j] for j in _greedy_table(table, k_build)]
    knee_info: Optional[Dict[str, Any]] = None
    if adaptive:
        full_curve = _evaluate_keys(table, sigs_full)["regret_curve"]
        k_star, knee_info = _adaptive_knee_k(
            full_curve, int(k_min) if k_min is not None else 1, saturation_eps
        )
        sigs_student = sigs_full[:k_star]
    else:
        sigs_student = sigs_full
    if oracle_seed and int(oracle_seed) > 0:
        if oracle_seeds is None:
            oracle_seeds = _select_smart_k_oracle(
                gemms, k=int(oracle_seed), key_of=key_of
            )
        sigs_student = _seed_oracle_winners(sigs_student, oracle_seeds, k_build)
    metrics: Dict[str, Any] = {
        "mode": "student_aware",
        "adaptive_k": bool(adaptive),
        "k_chosen": len(sigs_student),
        "k_build": k_build,
        "knee": knee_info,
        "n_sigs_student": len(sigs_student),
    }
    if not with_metrics:
        return {"sigs": sigs_student, "metrics": metrics}
    sigs_oracle = _select_smart_k_oracle(gemms, k=len(sigs_student) or k, key_of=key_of)
    student_eval = _evaluate_keys(table, sigs_student)
    oracle_eval = _evaluate_keys(table, sigs_oracle)
    metrics.update(
        {
            "n_sigs_oracle": len(sigs_oracle),
            "student_aware_sel_eff": student_eval["sel_eff"],
            "oracle_sel_eff_under_student": oracle_eval["sel_eff"],
            "n_eval_gemms": student_eval["n_eval"],
            "student_aware_regret_curve": student_eval["regret_curve"],
            "oracle_regret_curve": oracle_eval["regret_curve"],
        }
    )
    return {"sigs": sigs_student, "metrics": metrics}


# ── training ──────────────────────────────────────────────────────────────────


def _usable_cpus() -> int:
    """CPUs this process may run on (its affinity mask, not the whole host)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 8


def _train_worker_init(threads_per_worker: int, stage_pid: int) -> None:
    """Process-pool initializer: end the training worker with the stage
    process `stage_pid` (also when that one is killed), and cap every thread
    pool of the worker so workers x threads stays within the core count."""
    procs.die_with(stage_pid)
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


def worker_threads(train_workers: int, threads_per_worker: int) -> int:
    """Threads per training worker: --train-threads-per-worker, else the
    usable CPUs shared among the workers."""
    if threads_per_worker > 0:
        return int(threads_per_worker)
    return max(2, _usable_cpus() // max(1, int(train_workers)))


def run_in_workers(
    fn: Callable[..., Any],
    jobs: Mapping[str, Mapping[str, Any]],
    order: Sequence[str],
    n_workers: int,
    threads_per_worker: int,
    *,
    quiet: bool,
    done: str,
) -> Dict[str, Any]:
    """{cell: fn(**jobs[cell])} of the cells in `order`, on `n_workers`
    spawned processes of `threads_per_worker` threads each that end with
    this one. A cell whose job raises is left out with a warning, for the
    caller to run again in this process; progress is reported every ten
    cells, as cells `done`."""
    import concurrent.futures as cf
    import multiprocessing as mp

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(threads_per_worker)
    out: Dict[str, Any] = {}
    t0 = time.time()
    with cf.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_train_worker_init,
        initargs=(threads_per_worker, os.getpid()),
    ) as ex:
        futures = {ex.submit(fn, **jobs[label]): label for label in order}
        for nd, fut in enumerate(cf.as_completed(futures), 1):
            label = futures[fut]
            try:
                out[label] = fut.result()
            except Exception as e:
                ui.warn("parallel", f"{label}: failed ({e!r}); retrying serially")
            if not quiet and (nd % 10 == 0 or nd == len(order)):
                ui.info(
                    "parallel",
                    f"  {nd}/{len(order)} cells {done} "
                    f"({ui.fmt_dur(time.time() - t0)})",
                )
    return out


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
    "phase",
    "epoch",
    "step",
    "lr",
    "train_loss",
    "sel_eff",
    "val_sel_eff",
    "val_deployed_sel_eff",
    "val_loss",
    "pick_us_geomean",
    "n_eval_sample",
    "n_gemms",
    "n_val_gemms",
)


def _fmt(v: Optional[float], spec: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return format(v, spec)


def training_log_row(
    cell: str, m: Mapping[str, Any], n_gemms: int, n_val_gemms: int
) -> Dict[str, Any]:
    """training_log.csv row of one `epoch_metrics` entry (one evaluation
    point)."""
    return {
        "cell": cell,
        "phase": m["phase"],
        "epoch": m["epoch"],
        "step": m["step"],
        "lr": _fmt(m["lr"], ".3e"),
        "train_loss": _fmt(m["train_loss"], ".6f"),
        "sel_eff": _fmt(m["sel_eff"], ".6f"),
        "val_sel_eff": _fmt(m["val_sel_eff"], ".6f"),
        "val_deployed_sel_eff": _fmt(m["val_deployed_sel_eff"], ".6f"),
        "val_loss": _fmt(m["val_loss"], ".6f"),
        "pick_us_geomean": _fmt(m["pick_us_geomean"], ".4f"),
        "n_eval_sample": m["n_eval_sample"],
        "n_gemms": n_gemms,
        "n_val_gemms": n_val_gemms,
    }


LR_SCHEDULES: Tuple[str, ...] = ("cosine", "step", "constant")
STEP_DECAY_AT: Tuple[float, ...] = (0.6, 0.85)


def learning_rate(
    step: int,
    total: int,
    peak: float,
    *,
    schedule: str = "cosine",
    warmup_frac: float = 0.05,
    min_frac: float = 0.02,
) -> float:
    """Rate of optimizer step `step` (0-based) of `total`: a linear warmup
    over the first `warmup_frac` of the steps, then `schedule`: cosine decay
    to `min_frac` x `peak`, tenfold drops at STEP_DECAY_AT, or constant."""
    total = max(1, int(total))
    warm = min(total - 1, int(round(max(0.0, warmup_frac) * total)))
    if step < warm:
        return peak * (step + 1) / (warm + 1)
    if schedule == "constant":
        return peak
    t = min(1.0, (step - warm) / max(1, total - 1 - warm))
    if schedule == "step":
        return peak * 0.1 ** sum(1 for f in STEP_DECAY_AT if t >= f)
    if schedule != "cosine":
        raise ValueError(f"unknown learning-rate schedule {schedule!r}")
    lo = peak * min_frac
    return lo + (peak - lo) * 0.5 * (1.0 + math.cos(math.pi * t))


class _PairSet:
    """The candidate pairs of a list of GEMMs as tensors: query rows per
    GEMM, item rows once per distinct item vector, interaction rows,
    latencies and winners per pair. Each GEMM's pairs are contiguous, in
    candidate order (`build_cell_tensors`)."""

    def __init__(self, arrays: Tuple[Any, ...], device: str) -> None:
        q, i, x, gidx, us, win = arrays
        gidx = np.asarray(gidx, dtype=np.int64)
        if gidx.size and np.any(np.diff(gidx) < 0):
            raise ValueError("pairs must be grouped by GEMM")
        i_rows, i_index = np.unique(
            np.asarray(i, dtype=np.float32), axis=0, return_inverse=True
        )
        self.n_gemms = int(np.asarray(q).shape[0])
        self.counts = np.bincount(gidx, minlength=self.n_gemms)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(
            np.int64
        )
        self.q = torch.tensor(q, dtype=torch.float32, device=device)
        self.i = torch.tensor(i_rows, dtype=torch.float32, device=device)
        self.i_index = torch.tensor(
            i_index.reshape(-1), dtype=torch.long, device=device
        )
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.gidx = torch.tensor(gidx, dtype=torch.long, device=device)
        self.us = torch.tensor(us, dtype=torch.float64, device=device)
        self.win = torch.tensor(win, dtype=torch.float64, device=device)
        self.device = device

    def scores(self, student: Any) -> torch.Tensor:
        return student.score_grouped(self.q, self.gidx, self.i, self.i_index, self.x)

    def batch(self, gemms: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """(pair indices, batch-local GEMM index per pair) of `gemms`."""
        idx = np.concatenate(
            [np.arange(self.starts[g], self.starts[g] + self.counts[g]) for g in gemms]
        )
        local = np.repeat(np.arange(len(gemms)), self.counts[gemms])
        return (
            torch.from_numpy(idx).to(self.device),
            torch.from_numpy(local).to(self.device),
        )

    def batch_scores(
        self, student: Any, gemms: np.ndarray, idx: torch.Tensor, local: torch.Tensor
    ) -> torch.Tensor:
        dot, ix = self.batch_score_parts(student, gemms, idx, local)
        return dot + ix

    def batch_score_parts(
        self, student: Any, gemms: np.ndarray, idx: torch.Tensor, local: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sel = torch.from_numpy(np.asarray(gemms, dtype=np.int64)).to(self.device)
        return student.score_parts_grouped(
            self.q.index_select(0, sel),
            local,
            self.i,
            self.i_index.index_select(0, idx),
            self.x.index_select(0, idx),
        )


def unmeasured_kernels(
    g: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    n: int,
    rng: random.Random,
    *,
    lds_bytes: int,
    nt_flags: Tuple[bool, bool],
    rules: Mapping[str, bool],
    nt_families: Optional[Mapping[Tuple[int, ...], Any]] = None,
) -> List[Mapping[str, Any]]:
    """Up to `n` pool kernels at random that the GEMM's plan left off its
    list (`listed`) and that pass the engine's LDS gate and feasibility rule
    under the model's serving rules (`nt_families`: the pool's kernel
    families): kernels the deployed model can pick that nothing measured for
    it. None for a GEMM benched without a list, where a kernel without a row
    is one the runtime skips."""
    listed = g.get("listed")
    if listed is None or n <= 0:
        return []
    measured = {int(c["sol_idx_global"]) for c in g["candidates"]}
    prob = fs.problem_kwargs_from_row(g)
    order = list(range(len(pool)))
    rng.shuffle(order)
    out: List[Mapping[str, Any]] = []
    for j in order:
        k = pool[j]
        sid = int(k["sol_idx_global"])
        if sid in listed or sid in measured:
            continue
        if feasibility.passes_gates(
            prob,
            fs.config_kwargs_from_row(k),
            lds_bytes=lds_bytes,
            nt_a_available=nt_flags[0],
            nt_b_available=nt_flags[1],
            nt_families=nt_families,
            **rules,
        ):
            out.append(k)
            if len(out) >= n:
                break
    return out


def supported_keys(
    gemms: Iterable[Mapping[str, Any]],
    key_of: Callable[[Mapping[str, Any]], Tuple[int, ...]],
) -> set:
    """Candidate keys some GEMM of `gemms` has a row of."""
    return {key_of(c) for g in gemms for c in g["candidates"]}


def unsupported_candidates(
    pool: Sequence[Mapping[str, Any]],
    supported: set,
    key_of: Callable[[Mapping[str, Any]], Tuple[int, ...]],
) -> List[Tuple[Mapping[str, Any], frozenset]]:
    """The candidate keys of `pool` outside `supported`, in pool order, each
    as (its first pool kernel, the one the engine scores the key by, and the
    solution indices of its kernels)."""
    first: Dict[Tuple[int, ...], Mapping[str, Any]] = {}
    members: Dict[Tuple[int, ...], List[int]] = {}
    for k in pool:
        key = key_of(k)
        if key in supported:
            continue
        first.setdefault(key, k)
        members.setdefault(key, []).append(int(k["sol_idx_global"]))
    return [(first[key], frozenset(members[key])) for key in first]


def unsupported_kernels(
    g: Mapping[str, Any],
    candidates: Sequence[Tuple[Mapping[str, Any], frozenset]],
    n: int,
    rng: random.Random,
    *,
    lds_bytes: int,
    nt_flags: Tuple[bool, bool],
    rules: Mapping[str, bool],
    nt_families: Optional[Mapping[Tuple[int, ...], Any]] = None,
) -> List[Mapping[str, Any]]:
    """Up to `n` of `candidates` (`unsupported_candidates`: the candidate
    keys no training GEMM of the cell measured) at random that the serving
    rules admit for this GEMM, each as its first pool kernel. A key with a
    kernel on the GEMM's list is left out (a listed kernel without a row is
    one the runtime skips). None for a GEMM benched without a list."""
    listed = g.get("listed")
    if listed is None or n <= 0:
        return []
    keys = [k for k, sids in candidates if not sids & listed]
    rng.shuffle(keys)
    prob = fs.problem_kwargs_from_row(g)
    out: List[Mapping[str, Any]] = []
    for k in keys:
        if feasibility.passes_gates(
            prob,
            fs.config_kwargs_from_row(k),
            lds_bytes=lds_bytes,
            nt_a_available=nt_flags[0],
            nt_b_available=nt_flags[1],
            nt_families=nt_families,
            **rules,
        ):
            out.append(k)
            if len(out) >= n:
                break
    return out


def draw_pseudo_negatives(
    gemms: Sequence[Mapping[str, Any]],
    own: Sequence[Mapping[str, Any]],
    mode: str,
    n: int,
    seed: int,
    selector: Optional[Mapping[str, Any]],
    catalog: fs.Catalog,
    lds_bytes: int,
) -> List[List[Mapping[str, Any]]]:
    """The pseudo-negative kernels of each of `gemms` (a cell's GEMMs; `own`,
    its training GEMMs, decide which keys the cell measured): `random`, up
    to `n` feasible pool kernels off a GEMM's list (`unmeasured_kernels`);
    `targeted`, up to `n` keys no GEMM of `own` has a row of
    (`unsupported_kernels`). One generator seeded by `seed` draws for every
    GEMM in turn. The selector spec supplies the pools and serving rules."""
    pools = (selector or {}).get("pools") or {}
    wl_spec = (selector or {}).get("whitelist") or {}
    nt_flags = wl_spec.get("nt_flags") or {}
    rules = wl_spec.get("rules") or {}
    nt_families = wl_spec.get("nt_families") or {}
    rng = random.Random(int(seed) ^ 0x5EED)
    if mode == "targeted":
        serving = (selector or {}).get("serving") or {}
        key_of = served_key_of(catalog, bool(serving.get("launch_keys")))
        supported = supported_keys(own, key_of)
        candidates = {
            pk: unsupported_candidates(pools[pk], supported, key_of)
            for pk in {ev.pool_key(g) for g in gemms}
            if pk in pools
        }
        return [
            unsupported_kernels(
                g,
                candidates.get(ev.pool_key(g), ()),
                int(n),
                rng,
                lds_bytes=lds_bytes,
                nt_flags=tuple(nt_flags.get(ev.pool_key(g), (False, False))),
                rules=rules,
                nt_families=nt_families.get(ev.pool_key(g)),
            )
            for g in gemms
        ]
    return [
        unmeasured_kernels(
            g,
            pools.get(ev.pool_key(g), ()),
            int(n),
            rng,
            lds_bytes=lds_bytes,
            nt_flags=tuple(nt_flags.get(ev.pool_key(g), (False, False))),
            rules=rules,
            nt_families=nt_families.get(ev.pool_key(g)),
        )
        for g in gemms
    ]


def with_unmeasured_pairs(
    arrays: Tuple[np.ndarray, ...],
    gemms: Sequence[Mapping[str, Any]],
    kernels: Sequence[Sequence[Mapping[str, Any]]],
    margin: float,
    *,
    q_names: List[str],
    i_names: List[str],
    x_names: List[str],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    catalog: fs.Catalog,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """`build_cell_tensors` arrays (raw q, i, x, gemm index, us, winner) with
    each GEMM's `kernels` added after its measured pairs at the pseudo
    latency winner x (1 + `margin`); returns (arrays, measured: the new
    position of each measured pair, pseudo: mask of the added pairs)."""
    q, i, x, gidx, us, win = arrays
    pi: List[List[float]] = []
    px: List[List[float]] = []
    pg: List[int] = []
    for gi, (g, ks) in enumerate(zip(gemms, kernels)):
        if not ks:
            continue
        prob = fs.problem_kwargs_from_row(g)
        context = ev.context_of(g)
        for k in ks:
            _, i_vec, x_vec = catalog.vectors(
                prob,
                fs.config_kwargs_from_row(k),
                k.get("attributes"),
                hardware=hardware,
                constants=constants,
                context=context,
            )
            pi.append(i_vec)
            px.append(x_vec)
            pg.append(gi)
    n_meas = len(gidx)
    if not pg:
        return arrays, np.arange(n_meas), np.zeros(n_meas, dtype=bool)
    p_gidx = np.asarray(pg, dtype=np.int64)
    p_win = np.asarray([target_winner_us(gemms[j]) for j in pg], dtype=np.float64)
    all_g = np.concatenate([gidx, p_gidx])
    order = np.argsort(all_g, kind="stable")
    pseudo = np.concatenate([np.zeros(n_meas, bool), np.ones(len(pg), bool)])[order]
    measured = np.empty(n_meas, dtype=np.int64)
    measured[order[~pseudo]] = np.flatnonzero(~pseudo)
    out = (
        q,
        np.concatenate([i, np.asarray(pi, dtype=np.float32)])[order],
        np.concatenate([x, np.asarray(px, dtype=np.float32)])[order],
        all_g[order],
        np.concatenate([us, p_win * (1.0 + float(margin))])[order],
        np.concatenate([win, p_win])[order],
    )
    return out, measured, pseudo


def augmented_fit_inputs(
    q_raw: np.ndarray,
    q_white: np.ndarray,
    aug: Tuple[np.ndarray, ...],
    measured_pos: np.ndarray,
    pseudo: np.ndarray,
    norms: Dict[str, List[float]],
    bounds: Optional[Mapping[str, Sequence[float]]],
    teacher_p: Optional[torch.Tensor],
    *,
    alpha: float,
    tol: float,
    weight: float,
    device: str,
) -> Tuple[_PairSet, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(pairs, targets, teacher probabilities, their logs) of a fit set
    with pseudo-negatives (`with_unmeasured_pairs` arrays `aug`): its rows
    whitened with the measured pairs' statistics and bounds, soft-CE
    targets over every pair with the pseudo pairs' scaled by `weight`, and
    the teacher's probabilities of the measured pairs at their new
    positions, 0 for the pseudo ones."""
    _, ai, ax = apply_whiten(q_raw, aug[1], aug[2], norms, bounds)
    pairs = _PairSet((q_white, ai, ax, aug[3], aug[4], aug[5]), device)
    t = per_gemm_softmax_targets_np(
        aug[4], aug[5], aug[3], alpha=alpha, target_tolerance=tol
    )
    t[pseudo] *= float(weight)
    targets = torch.tensor(t, dtype=torch.float32, device=device)
    if teacher_p is None:
        return pairs, targets, None, None
    pos = torch.as_tensor(measured_pos, dtype=torch.long, device=device)
    tp = torch.zeros(len(pseudo), device=device)
    tp[pos] = teacher_p
    return pairs, targets, tp, torch.log(tp.clamp(min=1e-12))


def listwise_loss(
    scores: torch.Tensor,
    gemm_idx: torch.Tensor,
    n_gemms: int,
    targets: torch.Tensor,
    teacher_p: Optional[torch.Tensor] = None,
    teacher_logp: Optional[torch.Tensor] = None,
    lam: float = 0.5,
    T: float = 2.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean over GEMMs of the soft cross-entropy against `targets` and, with
    a teacher, lam * KL(teacher_T || student_T) * T^2 + (1 - lam) * soft_CE;
    every quantity is a per-GEMM softmax over the GEMM's pairs. `weights`
    (one per GEMM) makes it a weighted mean."""
    log_s = per_gemm_log_softmax(scores, gemm_idx, n_gemms)
    if weights is None:
        pair_w, total = 1.0, float(n_gemms)
    else:
        pair_w, total = weights.index_select(0, gemm_idx), weights.sum()
    ce = -(pair_w * targets * log_s).sum() / total
    if teacher_p is None or teacher_logp is None:
        return ce
    log_s_t = per_gemm_log_softmax(scores / T, gemm_idx, n_gemms)
    kl = (pair_w * teacher_p * (teacher_logp - log_s_t)).sum() / total * (T**2)
    return lam * kl + (1.0 - lam) * ce


def stage1_recall_loss(
    dot: torch.Tensor,
    scores: torch.Tensor,
    gemm_idx: torch.Tensor,
    n_gemms: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean over GEMMs of the soft cross-entropy of the embedding term's
    per-GEMM softmax against the full scores' (held fixed): it moves the
    cascade's first-stage ranking toward the full ranking at its top, and
    leaves the full scores to the main loss."""
    target = per_gemm_log_softmax(scores.detach(), gemm_idx, n_gemms).exp()
    log_d = per_gemm_log_softmax(dot, gemm_idx, n_gemms)
    if weights is None:
        pair_w, total = 1.0, float(n_gemms)
    else:
        pair_w, total = weights.index_select(0, gemm_idx), weights.sum()
    return -(pair_w * target * log_d).sum() / total


class KeyIndex:
    """The pair that stands for each whitelist key in each GEMM of a list,
    as `_student_score_cache` accounts for it: the key's fastest solution
    passing the engine's LDS gate and feasibility rule (the first in
    candidate order among equally fast ones). Pairs are numbered in
    candidate order over the GEMMs (`build_cell_tensors`); `table` turns the
    pairs' scores into a ScoreTable."""

    def __init__(
        self,
        gemms: Sequence[Mapping[str, Any]],
        *,
        lds_bytes: int,
        nt_flags: Mapping[Tuple[str, ...], Tuple[bool, bool]],
        rules: Mapping[str, bool],
        key_of: Optional[Any] = None,
        nt_families: Optional[Mapping[Tuple[str, ...], Any]] = None,
    ) -> None:
        if key_of is None:
            key_of = lambda c: tuple(int(v) for v in sig_from_row(c))  # noqa: E731
        rows: Dict[Tuple[int, ...], int] = {}
        cols: Dict[int, int] = {}
        best: Dict[Tuple[int, int], Tuple[float, int]] = {}
        p = -1
        for gi, g in enumerate(gemms):
            w = float(g.get("winner_us", 0) or 0)
            prob = fs.problem_kwargs_from_row(g)
            nt_a, nt_b = nt_flags.get(ev.pool_key(g), (False, False))
            families = (nt_families or {}).get(ev.pool_key(g))
            for c in g.get("candidates", []):
                p += 1
                us = float(c.get("us", 0) or 0)
                if w <= 0 or us <= 0:
                    continue
                if not feasibility.passes_gates(
                    prob,
                    fs.config_kwargs_from_row(c),
                    lds_bytes=lds_bytes,
                    nt_a_available=nt_a,
                    nt_b_available=nt_b,
                    nt_families=families,
                    **rules,
                ):
                    continue
                cell = (
                    rows.setdefault(key_of(c), len(rows)),
                    cols.setdefault(gi, len(cols)),
                )
                prev = best.get(cell)
                if prev is None or us < prev[0]:
                    best[cell] = (us, p)
        self.keys = list(rows)
        self.gemms = list(cols)
        self.us = np.full((len(rows), len(cols)), np.inf)
        self.pair = np.full((len(rows), len(cols)), -1, dtype=np.int64)
        for (j, c), (us, pi) in best.items():
            self.us[j, c], self.pair[j, c] = us, pi
        self.winner_us = np.asarray(
            [float(gemms[gi]["winner_us"]) for gi in self.gemms], dtype=np.float64
        )

    def table(self, pair_scores: np.ndarray) -> ScoreTable:
        scores = np.asarray(pair_scores, dtype=np.float64).reshape(-1)
        picked = scores[np.maximum(self.pair, 0)] if scores.size else self.us * 0
        score = np.where(self.pair >= 0, picked, -np.inf)
        return ScoreTable(self.keys, self.gemms, self.us, score, self.winner_us)


def whitelist_key(mode: str, catalog: fs.Catalog) -> Optional[Any]:
    """Whitelist key of a candidate for `mode`: None (the kernel signature)
    unless "view-key", the catalog's v3 view key."""
    if mode != "view-key":
        return None

    def key_of(c: Mapping[str, Any]) -> Tuple[int, ...]:
        return fs.v3_view_key(
            fs.config_kwargs_from_row(c), c.get("attributes"), catalog.spec
        )

    return key_of


def cell_bundle(
    label: str,
    entry: Mapping[str, Any],
    names: Tuple[List[str], List[str], List[str]],
    serving: Optional[Mapping[str, Any]],
    bundle_meta: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """A bundle holding one cell's models.pt entry: it routes the cell's
    GEMMs exactly like the full bundle, and an MLREC_v3 one serves every
    other base cell from it."""
    return {
        **dict(bundle_meta or {}),
        "q_names": names[0],
        "i_names": names[1],
        "x_names": names[2],
        "models": {label: dict(entry)},
        "serving": dict(serving or {}),
    }


class ShippedSelector:
    """How the deployed model ranks one cell: its whitelist, chosen with the
    student in the loop on the training GEMMs (`spec["whitelist"]`: mode
    "signature", "view-key" or "none", and the k, k_min, k_max,
    saturation_eps, oracle_seed, nt_flags and rules of
    `select_smart_k_signatures`), and, when `spec["engine"]` is set and the
    cell has validation GEMMs, the engine's picks for those under the
    model's serving options at `spec["weight_dtype"]` (lib/evaluate.py), as
    `deployed_eval_cell` measures them."""

    def __init__(
        self,
        label: str,
        spec: Mapping[str, Any],
        train_g: List[dict],
        val_g: List[dict],
        *,
        names: Tuple[List[str], List[str], List[str]],
        hardware: hwlib.DeviceHardware,
        constants: hwlib.ArchConstants,
        catalog: fs.Catalog,
    ) -> None:
        self.label, self.spec = label, spec
        self.wl: Mapping[str, Any] = spec.get("whitelist") or {"mode": "none"}
        mode = str(self.wl.get("mode", "none"))
        k_build = self.wl.get("k_max")
        if k_build is None:
            k_build = self.wl.get("k", 0)
        self.mode = mode if int(k_build or 0) > 0 else "none"
        self.train_g, self.val_g = train_g, val_g
        self.names, self.hardware, self.constants = names, hardware, constants
        self.key_of = whitelist_key(self.mode, catalog)
        self.index: Optional[KeyIndex] = None
        self.oracle_seeds: Optional[List[Tuple[int, ...]]] = None
        if self.mode != "none":
            self.index = KeyIndex(
                train_g,
                lds_bytes=hardware.lds_bytes,
                nt_flags=self.wl.get("nt_flags") or {},
                rules=self.wl.get("rules") or {},
                key_of=self.key_of,
                nt_families=self.wl.get("nt_families"),
            )
            n_seed = int(self.wl.get("oracle_seed") or 0)
            if n_seed > 0:
                self.oracle_seeds = _select_smart_k_oracle(
                    train_g, k=n_seed, key_of=self.key_of
                )
        self.tw = ev.tilewright_module() if spec.get("engine") and val_g else None

    @property
    def measures_validation(self) -> bool:
        return self.tw is not None

    def whitelist(
        self, pair_scores: np.ndarray, with_metrics: bool = False
    ) -> Dict[str, Any]:
        """{"sigs", "view_key_hashes", "metrics"} of the whitelist the
        student with these training-pair scores gets."""
        if self.index is None:
            return {"sigs": [], "view_key_hashes": [], "metrics": {"mode": "none"}}
        wl = self.wl
        sk = select_smart_k_signatures(
            self.train_g,
            int(wl.get("k", 10)),
            k_min=wl.get("k_min"),
            k_max=wl.get("k_max"),
            saturation_eps=float(wl.get("saturation_eps", 0.005)),
            oracle_seed=int(wl.get("oracle_seed") or 0),
            key_of=self.key_of,
            table=self.index.table(pair_scores),
            oracle_seeds=self.oracle_seeds,
            with_metrics=with_metrics,
        )
        if self.mode == "view-key":
            hashes = [fs.view_key_hash(k) for k in sk["sigs"]]
            return {"sigs": [], "view_key_hashes": hashes, "metrics": sk["metrics"]}
        return {"sigs": sk["sigs"], "view_key_hashes": [], "metrics": sk["metrics"]}

    def validation_sel_eff(self, entry: Mapping[str, Any]) -> float:
        bundle = cell_bundle(
            self.label,
            entry,
            self.names,
            self.spec.get("serving"),
            self.spec.get("bundle_meta"),
        )
        out = ev.evaluate_bundle(
            bundle,
            self.val_g,
            arch=self.spec["arch"],
            constants=self.constants,
            hardware=self.hardware,
            weight_dtype=self.spec["weight_dtype"],
            pools=self.spec.get("pools"),
            tw=self.tw,
            pick_measurement=self.spec.get(
                "pick_measurement", ev.DEFAULT_PICK_MEASUREMENT
            ),
            off_list_sel_eff=self.spec.get("off_list_sel_eff"),
        )
        ev.require_consistent_routing(out, f"{self.label}: early stopping")
        return float(ev.summarize(out.results)["sel_eff"])


def distill_step(
    student: Any,
    optim: Any,
    rate: float,
    pairs: _PairSet,
    gemms: np.ndarray,
    targets: torch.Tensor,
    teacher_p: Optional[torch.Tensor],
    teacher_logp: Optional[torch.Tensor],
    *,
    lam: float,
    T: float,
    stage1_recall_weight: float,
    grad_clip_norm: float,
    weights: Optional[torch.Tensor] = None,
) -> float:
    """One optimizer step at learning rate `rate` on the loss of `gemms`
    (indices into `pairs`): `listwise_loss` against the targets and the
    teacher's probabilities, plus `stage1_recall_weight` times
    `stage1_recall_loss`, gradients clipped to `grad_clip_norm` (0: not
    clipped). Returns the loss."""
    for group in optim.param_groups:
        group["lr"] = rate
    idx, local = pairs.batch(gemms)
    dot, ix = pairs.batch_score_parts(student, gemms, idx, local)
    scores = dot + ix
    loss = listwise_loss(
        scores,
        local,
        len(gemms),
        targets.index_select(0, idx),
        None if teacher_p is None else teacher_p.index_select(0, idx),
        None if teacher_logp is None else teacher_logp.index_select(0, idx),
        lam=lam,
        T=T,
        weights=weights,
    )
    if stage1_recall_weight > 0:
        loss = loss + float(stage1_recall_weight) * stage1_recall_loss(
            dot, scores, local, len(gemms), weights=weights
        )
    optim.zero_grad()
    loss.backward()
    if grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
    optim.step()
    return float(loss.item())


def _optimizer(student: Any, lr: float, weight_decay: float) -> Any:
    """AdamW with weight decay on the weight matrices only."""
    decay = [p for p in student.parameters() if p.requires_grad and p.ndim >= 2]
    rest = [p for p in student.parameters() if p.requires_grad and p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": rest, "weight_decay": 0.0},
        ],
        lr=lr,
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
    tie_tolerance: float,
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
    catalog: fs.Catalog = fs.V2_CATALOG,
    clip: bool = False,
    batch_gemms: int = 16,
    lr_schedule: str = "cosine",
    warmup_frac: float = 0.05,
    lr_min_frac: float = 0.02,
    eval_points: int = 40,
    patience: int = 0,
    learn_temperature: bool = True,
    weight_dtype: str = "fp32",
    qat_frac: float = 0.0,
    qat_lr_frac: float = 0.1,
    selector: Optional[Mapping[str, Any]] = None,
    margin_gemms: Sequence[dict] = (),
    margin_weight: float = 1.0,
    unmeasured_samples: int = 0,
    unmeasured_margin: float = 0.05,
    unmeasured_weight: float = 0.3,
    pseudo_negatives: str = "random",
    stage1_recall_weight: float = 0.0,
) -> Dict[str, Any]:
    """Train one cell from freshly initialized weights (seeded by `seed`).
    With `clip` the entry carries the training features' bounds
    (`feature_bounds`), which validation features are clamped to.

    The student trains on the cell's training GEMMs in mini-batches of
    `batch_gemms` GEMMs (0: all), the loss of each GEMM taken over all its
    candidates (`listwise_loss`), for `epochs` passes, with the learning rate
    of `learning_rate`. `eval_points` times over the schedule the checkpoint
    is measured: with `selector` (a ShippedSelector spec) its whitelist is
    chosen, and when the selector measures the validation GEMMs, the kept
    checkpoint is the one with the best deployed validation sel_eff; else
    the student's own validation sel_eff, else (no validation GEMMs) its
    sel_eff on an `eval_sample_frac` sample of the training GEMMs. Equal
    measurements keep the checkpoint with the lower validation loss.
    Training stops after `patience` (> 0) measurements without a better one.
    Every measurement uses the weights as `weight_dtype` stores them. With
    `qat_frac` > 0 and an int8 / int4 `weight_dtype`, training continues
    from the kept checkpoint for `qat_frac` of the schedule's steps with the
    weights rounded as stored (`fake_quant_weight`), the learning rate
    falling from `qat_lr_frac` x `lr`, and keeps a better checkpoint.

    `margin_gemms` (training GEMMs of other cells, `lib/boundaries.py`) join
    the student's loss with weight `margin_weight`; the teacher, the
    whitelist and every measurement use the cell's own GEMMs only.

    With `unmeasured_samples` > 0 (and a selector's pools) every training
    GEMM benched from a plan gets up to that many kernels in its loss as
    pseudo-negatives (`draw_pseudo_negatives`): with `pseudo_negatives`
    "random", feasible pool kernels off its list; "targeted", keys no
    training GEMM of the cell measured. Their soft-CE target is the one of
    a kernel `unmeasured_margin` slower than the GEMM's winner, scaled by
    `unmeasured_weight`, and the teacher gives them none. They change only
    the loss: the feature statistics, the teacher, the whitelist and every
    measurement use the measured pairs.

    With `stage1_recall_weight` > 0 the loss adds that times
    `stage1_recall_loss`, which trains the embedding term alone to rank each
    GEMM's pairs as the full score does: the cascade's first stage keeps
    the keys the full score would pick.

    Returns the models.pt entry (weights and whitelist) plus measurements."""
    t0 = time.time()
    torch.manual_seed(int(seed))
    train_g, val_g = split_validation(gemms, validation_frac, split_seed)
    n_gemms = len(train_g)
    fit_g = train_g + list(margin_gemms)
    n_fit = len(fit_g)

    (qvec_raw, ivec_raw, xvec_raw, gemm_idx, us, winner_us, _rank_us) = (
        build_cell_tensors(
            fit_g, q_names, i_names, x_names, hardware, constants, catalog
        )
    )
    n_own_pairs = int(np.count_nonzero(np.asarray(gemm_idx) < n_gemms))
    own = slice(0, n_own_pairs)
    qvec, ivec, xvec, norms = whiten(qvec_raw, ivec_raw, xvec_raw)
    bounds = feature_bounds(qvec_raw, ivec_raw, xvec_raw) if clip else None
    tr_set = _PairSet((qvec, ivec, xvec, gemm_idx, us, winner_us), device)
    gemm_w = None
    if n_fit > n_gemms:
        gemm_w = torch.ones(n_fit, dtype=torch.float32, device=device)
        gemm_w[n_gemms:] = float(margin_weight)

    teacher_p = teacher_logp = None
    if use_teacher:
        booster, teacher_scores = fit_lgb_teacher(
            qvec_raw[:n_gemms],
            ivec_raw[own],
            xvec_raw[own],
            gemm_idx[own],
            us[own],
            winner_us[own],
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            lr=lgb_lr,
            seed=seed,
            quiet=quiet,
            tie_tolerance=tie_tolerance,
        )
        if n_fit > n_gemms:
            teacher_scores = booster.predict(
                np.concatenate([qvec_raw[gemm_idx], ivec_raw, xvec_raw], axis=1)
            ).astype(np.float32)
        teacher_p = per_gemm_softmax_T(
            torch.tensor(teacher_scores, dtype=torch.float32, device=device),
            tr_set.gidx,
            T=T,
            n_groups=n_fit,
        )
        teacher_logp = torch.log(teacher_p.clamp(min=1e-12))

    tol = math.log1p(max(float(tie_tolerance), 0.0))
    targets = torch.tensor(
        per_gemm_softmax_targets_np(
            us, winner_us, gemm_idx, alpha=alpha, target_tolerance=tol
        ),
        dtype=torch.float32,
        device=device,
    )
    fit_set, fit_targets = tr_set, targets
    fit_teacher_p, fit_teacher_logp = teacher_p, teacher_logp
    n_unmeasured = 0
    if unmeasured_samples > 0 and (selector or {}).get("pools"):
        picked = draw_pseudo_negatives(
            fit_g,
            train_g,
            pseudo_negatives,
            unmeasured_samples,
            seed,
            selector,
            catalog,
            hardware.lds_bytes,
        )
        raw = (qvec_raw, ivec_raw, xvec_raw, gemm_idx, us, winner_us)
        aug, measured_pos, pseudo = with_unmeasured_pairs(
            raw,
            fit_g,
            picked,
            unmeasured_margin,
            q_names=q_names,
            i_names=i_names,
            x_names=x_names,
            hardware=hardware,
            constants=constants,
            catalog=catalog,
        )
        n_unmeasured = int(pseudo.sum())
        if n_unmeasured:
            fit_set, fit_targets, fit_teacher_p, fit_teacher_logp = (
                augmented_fit_inputs(
                    qvec_raw,
                    qvec,
                    aug,
                    measured_pos,
                    pseudo,
                    norms,
                    bounds,
                    teacher_p,
                    alpha=alpha,
                    tol=tol,
                    weight=unmeasured_weight,
                    device=device,
                )
            )

    rng_eval = np.random.default_rng(seed)
    n_eval_sample = max(1, int(round(n_gemms * eval_sample_frac)))
    eval_idx_np = np.sort(rng_eval.choice(n_gemms, size=n_eval_sample, replace=False))
    eval_sample_mask = torch.zeros(n_fit, dtype=torch.bool, device=device)
    eval_sample_mask[torch.tensor(eval_idx_np, dtype=torch.long, device=device)] = True

    va_set: Optional[_PairSet] = None
    va_targets = None
    if val_g:
        (vq_raw, vi_raw, vx_raw, v_gidx, v_us, v_win, _v_rank) = build_cell_tensors(
            val_g, q_names, i_names, x_names, hardware, constants, catalog
        )
        vq, vi, vx = apply_whiten(vq_raw, vi_raw, vx_raw, norms, bounds)
        va_set = _PairSet((vq, vi, vx, v_gidx, v_us, v_win), device)
        va_targets = torch.tensor(
            per_gemm_softmax_targets_np(
                v_us, v_win, v_gidx, alpha=alpha, target_tolerance=tol
            ),
            dtype=torch.float32,
            device=device,
        )
    val_mask = torch.ones(len(val_g), dtype=torch.bool, device=device)
    judge = (
        ShippedSelector(
            cell,
            selector,
            train_g,
            val_g,
            names=(q_names, i_names, x_names),
            hardware=hardware,
            constants=constants,
            catalog=catalog,
        )
        if selector is not None
        else None
    )
    shipped = judge is not None and judge.measures_validation

    ed, hd, ih, cell_dropout = _capacity(
        cell, n_gemms, us[own], winner_us[own], ivec_raw[own]
    )
    student = GenericTwoTower(
        q_dim=len(q_names),
        i_dim=len(i_names),
        x_dim=len(x_names),
        embed_dim=ed,
        hidden_dim=hd,
        inter_hidden=ih,
        dropout=cell_dropout,
        learn_temperature=learn_temperature,
    )
    student.to(device)
    student.train()

    def entry_of(state: Mapping[str, Any], wl: Mapping[str, Any]) -> Dict[str, Any]:
        entry = {
            "state_dict": state,
            "feature_norms": norms,
            "embed_dim": ed,
            "hidden_dim": hd,
            "inter_hidden": ih,
            "smart_k_signatures": wl.get("sigs") or [],
            "view_key_hashes": wl.get("view_key_hashes") or [],
        }
        if bounds:
            entry["feature_bounds"] = bounds
        return entry

    def stored_scores(pairs: _PairSet) -> torch.Tensor:
        """Scores of the current weights as the model file stores them."""
        student.eval()
        saved = student.fake_quant
        student.fake_quant = weight_dtype
        try:
            with torch.no_grad():
                return pairs.scores(student)
        finally:
            student.fake_quant = saved
            student.train()

    def measure() -> Dict[str, Any]:
        tr_scores = stored_scores(tr_set)
        tr = sel_eff_of_scores(
            tr_scores, tr_set.gidx, tr_set.us, tr_set.win, eval_sample_mask, n_fit
        )
        va = va_loss = None
        if va_set is not None:
            va_scores = stored_scores(va_set)
            va = sel_eff_of_scores(
                va_scores, va_set.gidx, va_set.us, va_set.win, val_mask, len(val_g)
            )
            va_loss = float(
                listwise_loss(va_scores, va_set.gidx, len(val_g), va_targets)
            )
        out: Dict[str, Any] = {"train": tr, "val": va, "val_loss": va_loss}
        out["state"] = student.export_state()
        out["deployed_val"] = None
        if shipped:
            assert judge is not None
            wl = judge.whitelist(tr_scores[own].cpu().numpy())
            out["deployed_val"] = judge.validation_sel_eff(entry_of(out["state"], wl))
        primary = out["deployed_val"]
        if primary is None:
            primary = (va or tr)["sel_eff"]
        if not math.isfinite(primary):
            primary = -math.inf
        out["key"] = (primary, -va_loss if va_loss is not None else 0.0)
        return out

    per_epoch_metrics: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    batch = n_fit if batch_gemms <= 0 else min(int(batch_gemms), n_fit)
    steps_per_epoch = math.ceil(n_fit / batch)
    rng_batch = np.random.default_rng(int(seed) + 1)
    stats = {"steps": 0, "stopped_early": False}

    def run_phase(phase: str, n_steps: int, peak: float, warmup: float) -> None:
        """`n_steps` optimizer steps of mini-batch epochs, measured
        `eval_points` times (at least once at the end); keeps the best
        checkpoint in `best`."""
        nonlocal best
        optim = _optimizer(student, peak, weight_decay)
        eval_every = max(1, math.ceil(n_steps / max(1, int(eval_points))))
        since_best, losses, step = 0, [], 0
        while step < n_steps:
            order = rng_batch.permutation(n_fit)
            for b0 in range(0, n_fit, batch):
                if step >= n_steps:
                    break
                sel = np.sort(order[b0 : b0 + batch])
                rate = learning_rate(
                    step,
                    n_steps,
                    peak,
                    schedule=lr_schedule,
                    warmup_frac=warmup,
                    min_frac=lr_min_frac,
                )
                losses.append(
                    distill_step(
                        student,
                        optim,
                        rate,
                        fit_set,
                        sel,
                        fit_targets,
                        fit_teacher_p,
                        fit_teacher_logp,
                        lam=lam,
                        T=T,
                        stage1_recall_weight=stage1_recall_weight,
                        grad_clip_norm=grad_clip_norm,
                        weights=None if gemm_w is None else gemm_w[sel],
                    )
                )
                step += 1
                stats["steps"] += 1
                if step % eval_every and step != n_steps:
                    continue
                m = measure()
                tr, va = m["train"], m["val"]
                rec = {
                    "phase": phase,
                    "epoch": (stats["steps"] - 1) // steps_per_epoch,
                    "step": stats["steps"],
                    "lr": rate,
                    "train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "sel_eff": tr["sel_eff"],
                    "val_sel_eff": va["sel_eff"] if va is not None else None,
                    "val_deployed_sel_eff": m["deployed_val"],
                    "val_loss": m["val_loss"],
                    "pick_us_geomean": tr["pick_us_geomean"],
                    "n_eval_sample": tr["n_eval"],
                }
                losses = []
                per_epoch_metrics.append(rec)
                if training_log_writer is not None:
                    training_log_writer.writerow(
                        training_log_row(cell, rec, n_gemms, len(val_g))
                    )
                if best is None or m["key"] > best["key"]:
                    best = dict(m, rec=rec)
                    since_best = 0
                else:
                    since_best += 1
                if not quiet:
                    vs = "" if va is None else f"  val={va['sel_eff'] * 100:.2f}%"
                    if m["deployed_val"] is not None:
                        vs += f"  val_deployed={m['deployed_val'] * 100:.2f}%"
                    print(
                        f"  {ui.C.GREY}[{cell}]{ui.C.ENDC} {phase} step "
                        f"{stats['steps']:>5d}  lr={rate:.2e}  "
                        f"loss={rec['train_loss']:7.4f}  "
                        f"sel_eff={tr['sel_eff'] * 100:.2f}%{vs}",
                        flush=True,
                    )
                if patience > 0 and since_best >= patience:
                    stats["stopped_early"] = stats["stopped_early"] or phase == "main"
                    return

    run_phase("main", max(1, int(epochs)) * steps_per_epoch, lr, warmup_frac)
    qat = weight_dtype in QUANTIZED_WEIGHT_LEVELS and qat_frac > 0
    if qat:
        assert best is not None
        student.load_exported(best["state"])
        student.fake_quant = weight_dtype
        n_qat = max(1, int(round(qat_frac * max(1, int(epochs)) * steps_per_epoch)))
        run_phase("qat", n_qat, lr * qat_lr_frac, 0.0)
        student.fake_quant = None

    assert best is not None
    student.load_exported(best["state"])
    wl = None
    if judge is not None:
        wl = judge.whitelist(stored_scores(tr_set)[own].cpu().numpy(), True)
    student.eval()
    sel = best["val"] if best["val"] is not None else best["train"]
    rec = best["rec"]
    out: Dict[str, Any] = {
        "cell": cell,
        "n_gemms": len(gemms),
        "n_train_gemms": n_gemms,
        "n_val_gemms": len(val_g),
        "n_margin_gemms": n_fit - n_gemms,
        "selection_set": "validation" if val_g else "training",
        "selection_metric": (
            "deployed_validation"
            if shipped
            else ("validation" if val_g else "training")
        ),
        "n_pairs": n_own_pairs,
        "n_unmeasured_pairs": n_unmeasured,
        "q_dim": len(q_names),
        "i_dim": len(i_names),
        "x_dim": len(x_names),
        "embed_dim": ed,
        "hidden_dim": hd,
        "inter_hidden": ih,
        "seed": int(seed),
        "best_epoch": rec["epoch"],
        "best_step": rec["step"],
        "best_phase": rec["phase"],
        "steps": stats["steps"],
        "stopped_early": stats["stopped_early"],
        "temperature": float(best["state"]["temperature"][0]),
        "best_sel_eff": sel["sel_eff"],
        "best_pick_us_geomean": sel.get("pick_us_geomean", float("nan")),
        "winner_us_geomean": sel.get("winner_us_geomean", float("nan")),
        "train_sel_eff": best["train"]["sel_eff"],
        "val_sel_eff": (best["val"] or {}).get("sel_eff"),
        "selection_deployed_val_sel_eff": best["deployed_val"],
        "state_dict": best["state"],
        "feature_norms": norms,
        "feature_bounds": bounds,
        "epoch_metrics": per_epoch_metrics,
        "elapsed_s": time.time() - t0,
    }
    if wl is not None:
        out["smart_k_signatures"] = wl["sigs"]
        out["view_key_hashes"] = wl["view_key_hashes"]
        out["smart_k_metrics"] = wl["metrics"]
    return out


# ── pooled recipe ─────────────────────────────────────────────────────────────

RECIPES = ("per_leaf", "pooled_ft")
# The pooled recipe's student widths (embedding, hidden, interaction), the
# widest per-leaf capacity: every cell fine-tunes the one pooled student.
POOLED_CAPACITY = (80, 192, 112)
# training_log.csv's cell of the pooled student's checkpoints.
POOLED_LABEL = "pooled"


@dataclass(frozen=True)
class DistillSpec:
    """The loss, schedule shape and stored-weight settings the pooled
    recipe's phases share (the stage05 options of the same names)."""

    lam: float
    T: float
    alpha: float
    tie_tolerance: float
    stage1_recall_weight: float
    weight_decay: float
    grad_clip_norm: float
    lr_schedule: str
    warmup_frac: float
    lr_min_frac: float
    learn_temperature: bool
    weight_dtype: str
    unmeasured_weight: float
    device: str

    @property
    def tol(self) -> float:
        return math.log1p(max(float(self.tie_tolerance), 0.0))


@dataclass
class CellPairs:
    """A cell's pairs as the pooled recipe trains them: the
    `build_cell_tensors` arrays (q per GEMM; i, x, GEMM index, us and target
    winner per pair) of its training GEMMs and of its validation GEMMs (None
    without any), and its pseudo-negatives (`with_unmeasured_pairs`): the
    added pairs' i, x, GEMM index, us and winner, then the new position of
    each measured pair and the mask of added pairs (None without any)."""

    cell: str
    n_gemms: int
    train: Tuple[np.ndarray, ...]
    val: Optional[Tuple[np.ndarray, ...]]
    pseudo: Optional[Tuple[np.ndarray, ...]] = None

    @property
    def n_pseudo(self) -> int:
        return 0 if self.pseudo is None else int(self.pseudo[-1].sum())

    def fit_arrays(self) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """(the training arrays with the pseudo-negatives in place, as
        `with_unmeasured_pairs` returns them, the measured pairs' positions,
        the mask of pseudo pairs)."""
        n = len(self.train[3])
        if self.pseudo is None:
            return self.train, np.arange(n), np.zeros(n, dtype=bool)
        *added, measured_pos, mask = self.pseudo
        out = [self.train[0]]
        for measured, extra in zip(self.train[1:], added):
            a = np.empty((len(mask),) + np.shape(measured)[1:], dtype=measured.dtype)
            a[measured_pos] = measured
            a[mask] = extra
            out.append(a)
        return tuple(out), measured_pos, mask


def cell_pairs(
    *,
    cell: str,
    gemms: List[dict],
    names: Tuple[List[str], List[str], List[str]],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    catalog: fs.Catalog,
    validation_frac: float,
    split_seed: int,
    seed: int,
    selector: Optional[Mapping[str, Any]],
    unmeasured_samples: int,
    unmeasured_margin: float,
    pseudo_negatives: str,
) -> CellPairs:
    """A cell's CellPairs: `split_validation`'s sides and, with
    `unmeasured_samples` > 0, the pseudo-negatives `train_cell` adds for the
    cell, drawn with its `seed`."""
    train_g, val_g = split_validation(gemms, validation_frac, split_seed)

    def tensors(gs: List[dict]) -> Tuple[np.ndarray, ...]:
        return build_cell_tensors(gs, *names, hardware, constants, catalog)[:6]

    train = tensors(train_g)
    pseudo = None
    if unmeasured_samples > 0 and (selector or {}).get("pools"):
        picked = draw_pseudo_negatives(
            train_g,
            train_g,
            pseudo_negatives,
            unmeasured_samples,
            seed,
            selector,
            catalog,
            hardware.lds_bytes,
        )
        aug, measured_pos, mask = with_unmeasured_pairs(
            train,
            train_g,
            picked,
            unmeasured_margin,
            q_names=names[0],
            i_names=names[1],
            x_names=names[2],
            hardware=hardware,
            constants=constants,
            catalog=catalog,
        )
        if mask.any():
            pseudo = (*(a[mask] for a in aug[1:]), measured_pos, mask)
    return CellPairs(cell, len(gemms), train, tensors(val_g) if val_g else None, pseudo)


def concat_pairs(arrays: Sequence[Tuple[np.ndarray, ...]]) -> Tuple[np.ndarray, ...]:
    """`build_cell_tensors` arrays of several cells as one set: query rows
    stacked, each cell's GEMM indices shifted past the GEMMs before it."""
    offsets = np.cumsum([0] + [len(a[0]) for a in arrays])[:-1]
    return (
        np.concatenate([a[0] for a in arrays]),
        np.concatenate([a[1] for a in arrays]),
        np.concatenate([a[2] for a in arrays]),
        np.concatenate([a[3] + o for a, o in zip(arrays, offsets)]),
        np.concatenate([a[4] for a in arrays]),
        np.concatenate([a[5] for a in arrays]),
    )


def stored_sel_eff(student: Any, pairs: _PairSet, weight_dtype: str) -> Dict[str, Any]:
    """`sel_eff_of_scores` of every GEMM of `pairs` under the student's
    weights as the model file stores them at `weight_dtype`."""
    student.eval()
    saved = student.fake_quant
    student.fake_quant = weight_dtype
    try:
        with torch.no_grad():
            scores = pairs.scores(student)
    finally:
        student.fake_quant = saved
        student.train()
    every = torch.ones(pairs.n_gemms, dtype=torch.bool, device=scores.device)
    return sel_eff_of_scores(
        scores, pairs.gidx, pairs.us, pairs.win, every, pairs.n_gemms
    )


def _teacher_probabilities(
    scores: Optional[np.ndarray], pairs: _PairSet, T: float
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if scores is None:
        return None, None
    p = per_gemm_softmax_T(
        torch.tensor(scores, dtype=torch.float32, device=pairs.device),
        pairs.gidx,
        T=T,
        n_groups=pairs.n_gemms,
    )
    return p, torch.log(p.clamp(min=1e-12))


def _fit_inputs(
    pairs: _PairSet,
    raw: Tuple[np.ndarray, ...],
    q_white: np.ndarray,
    augmented: Optional[Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]],
    teacher_scores: Optional[np.ndarray],
    norms: Dict[str, List[float]],
    bounds: Optional[Mapping[str, Sequence[float]]],
    spec: DistillSpec,
) -> Tuple[_PairSet, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """(pairs, targets, teacher probabilities, their logs) the loss trains
    on: the measured `pairs` (raw arrays `raw`), or with pseudo-negatives
    (`augmented`, as `CellPairs.fit_arrays` returns them) those pairs
    augmented (`augmented_fit_inputs`)."""
    tp, tlogp = _teacher_probabilities(teacher_scores, pairs, spec.T)
    if augmented is not None:
        aug, measured_pos, pseudo = augmented
        return augmented_fit_inputs(
            raw[0],
            q_white,
            aug,
            measured_pos,
            pseudo,
            norms,
            bounds,
            tp,
            alpha=spec.alpha,
            tol=spec.tol,
            weight=spec.unmeasured_weight,
            device=pairs.device,
        )
    targets = per_gemm_softmax_targets_np(
        raw[4], raw[5], raw[3], alpha=spec.alpha, target_tolerance=spec.tol
    )
    return (
        pairs,
        torch.tensor(targets, dtype=torch.float32, device=pairs.device),
        tp,
        tlogp,
    )


def train_pooled_student(
    parts: Sequence[CellPairs],
    teacher_scores: Optional[np.ndarray],
    spec: DistillSpec,
    *,
    names: Tuple[List[str], List[str], List[str]],
    clip: bool,
    epochs: int,
    batch_gemms: int,
    lr: float,
    eval_points: int,
    seed: int,
    capacity: Tuple[int, int, int] = POOLED_CAPACITY,
    on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """The pooled student: one GenericTwoTower trained on the training pairs
    of every cell of `parts` together (their feature statistics and bounds,
    `teacher_scores` of the measured pairs in that order, and the cells'
    pseudo-negatives) in mini-batches of `batch_gemms` GEMMs for `epochs`
    passes, measured on every cell's validation GEMMs together
    (`stored_sel_eff`) every 1/`eval_points` of the schedule and at its end,
    keeping the best checkpoint (the first of equal ones). `seed` seeds the
    weights and, plus one, the batch order. Returns its models.pt entry
    (`entry`) and measurements."""
    t0 = time.time()
    device = spec.device
    train = concat_pairs([p.train for p in parts])
    val_parts = [p.val for p in parts if p.val is not None]
    if not val_parts:
        raise ValueError("the pooled student needs validation GEMMs")
    val = concat_pairs(val_parts)
    qv, iv, xv, norms = whiten(train[0], train[1], train[2])
    bounds = feature_bounds(train[0], train[1], train[2]) if clip else None
    vq, vi, vx = apply_whiten(val[0], val[1], val[2], norms, bounds)
    tr_set = _PairSet((qv, iv, xv, *train[3:]), device)
    va_set = _PairSet((vq, vi, vx, *val[3:]), device)
    n_unmeasured = sum(p.n_pseudo for p in parts)
    augmented = None
    if n_unmeasured:
        fits = [p.fit_arrays() for p in parts]
        offsets = np.cumsum([0] + [len(f[2]) for f in fits])[:-1]
        augmented = (
            concat_pairs([f[0] for f in fits]),
            np.concatenate([f[1] + o for f, o in zip(fits, offsets)]),
            np.concatenate([f[2] for f in fits]),
        )
        del fits
    fit_set, targets, tp, tlogp = _fit_inputs(
        tr_set, train, qv, augmented, teacher_scores, norms, bounds, spec
    )
    del augmented, qv, iv, xv, vq, vi, vx
    ed, hd, ih = (int(v) for v in capacity)
    torch.manual_seed(int(seed))
    student = GenericTwoTower(
        q_dim=len(names[0]),
        i_dim=len(names[1]),
        x_dim=len(names[2]),
        embed_dim=ed,
        hidden_dim=hd,
        inter_hidden=ih,
        learn_temperature=spec.learn_temperature,
    )
    student.to(device)
    optim = _optimizer(student, lr, spec.weight_decay)
    n = fit_set.n_gemms
    batch = n if batch_gemms <= 0 else min(int(batch_gemms), n)
    steps_per_epoch = math.ceil(n / batch)
    total = max(1, int(epochs)) * steps_per_epoch
    every = max(1, total // max(1, int(eval_points)))
    rng = np.random.default_rng(int(seed) + 1)
    best: Optional[Dict[str, Any]] = None
    losses: List[float] = []
    step = 0
    while step < total:
        order = rng.permutation(n)
        for b0 in range(0, n, batch):
            if step >= total:
                break
            sel = np.sort(order[b0 : b0 + batch])
            rate = learning_rate(
                step,
                total,
                lr,
                schedule=spec.lr_schedule,
                warmup_frac=spec.warmup_frac,
                min_frac=spec.lr_min_frac,
            )
            losses.append(
                distill_step(
                    student,
                    optim,
                    rate,
                    fit_set,
                    sel,
                    targets,
                    tp,
                    tlogp,
                    lam=spec.lam,
                    T=spec.T,
                    stage1_recall_weight=spec.stage1_recall_weight,
                    grad_clip_norm=spec.grad_clip_norm,
                )
            )
            step += 1
            if step % every and step != total:
                continue
            va = stored_sel_eff(student, va_set, spec.weight_dtype)
            rec = {
                "phase": POOLED_LABEL,
                "epoch": (step - 1) // steps_per_epoch,
                "step": step,
                "lr": rate,
                "train_loss": float(np.mean(losses)),
                "sel_eff": None,
                "val_sel_eff": va["sel_eff"],
                "val_deployed_sel_eff": None,
                "val_loss": None,
                "pick_us_geomean": va["pick_us_geomean"],
                "n_eval_sample": va["n_eval"],
            }
            losses = []
            if on_checkpoint is not None:
                on_checkpoint(rec)
            if best is None or va["sel_eff"] > best["val"]["sel_eff"]:
                best = {"val": va, "rec": rec, "state": student.export_state()}
    assert best is not None
    entry = {
        "state_dict": best["state"],
        "feature_norms": norms,
        "embed_dim": ed,
        "hidden_dim": hd,
        "inter_hidden": ih,
        "best_epoch": best["rec"]["epoch"],
        "smart_k_signatures": [],
    }
    if bounds:
        entry["feature_bounds"] = bounds
    return {
        "entry": entry,
        "n_cells": len(parts),
        "n_train_gemms": n,
        "n_pairs": int(len(train[3])),
        "n_unmeasured_pairs": n_unmeasured,
        "n_val_gemms": va_set.n_gemms,
        "embed_dim": ed,
        "hidden_dim": hd,
        "inter_hidden": ih,
        "steps": total,
        "best_step": best["rec"]["step"],
        "best_val_sel_eff": best["val"]["sel_eff"],
        "elapsed_s": time.time() - t0,
    }


def fine_tune_cell(
    *,
    cell: str,
    pairs: CellPairs,
    init: Mapping[str, Any],
    teacher_scores: Optional[np.ndarray],
    spec: DistillSpec,
    names: Tuple[List[str], List[str], List[str]],
    epochs: int,
    lr: float,
    batch_gemms: int,
    seed: int,
    cell_seed: int,
) -> Dict[str, Any]:
    """One cell's model under the pooled recipe: the pooled student `init`
    (its weights, feature statistics and bounds) trained further on the
    cell's training pairs (`teacher_scores` of the measured ones, and its
    pseudo-negatives) for `epochs` passes of `batch_gemms`-GEMM batches at a
    peak rate `lr`, measured on the cell's validation GEMMs after each pass
    (`stored_sel_eff`). The kept weights are the best measured ones,
    `init`'s unless a pass beats them; a cell without validation GEMMs keeps
    `init`. `seed` seeds the weights' initialization before `init` replaces
    them and the batch order; `cell_seed`, the seed its pseudo-negatives
    were drawn with, is recorded. Returns what `train_cell` returns."""
    t0 = time.time()
    device = spec.device
    torch.manual_seed(int(seed))
    student = GenericTwoTower(
        q_dim=len(names[0]),
        i_dim=len(names[1]),
        x_dim=len(names[2]),
        embed_dim=int(init["embed_dim"]),
        hidden_dim=int(init["hidden_dim"]),
        inter_hidden=int(init["inter_hidden"]),
        learn_temperature=spec.learn_temperature,
    ).to(device)
    student.load_exported(init["state_dict"])
    student.eval()
    norms, bounds = init["feature_norms"], init.get("feature_bounds")
    q, i, x = pairs.train[:3]
    qv, iv, xv = apply_whiten(q, i, x, norms, bounds)
    tr_set = _PairSet((qv, iv, xv, *pairs.train[3:]), device)
    va_set = None
    if pairs.val is not None:
        vq, vi, vx = apply_whiten(*pairs.val[:3], norms, bounds)
        va_set = _PairSet((vq, vi, vx, *pairs.val[3:]), device)
    history: List[Dict[str, Any]] = []
    steps = 0

    def record(epoch: int, rate: Optional[float], loss: Optional[float]) -> Dict:
        assert va_set is not None
        va = stored_sel_eff(student, va_set, spec.weight_dtype)
        history.append(
            {
                "phase": "fine_tune",
                "epoch": epoch,
                "step": steps,
                "lr": rate,
                "train_loss": loss,
                "sel_eff": None,
                "val_sel_eff": va["sel_eff"],
                "val_deployed_sel_eff": None,
                "val_loss": None,
                "pick_us_geomean": va["pick_us_geomean"],
                "n_eval_sample": va["n_eval"],
            }
        )
        return {"val": va, "rec": history[-1], "state": student.export_state()}

    best: Optional[Dict[str, Any]] = None
    pooled_val = None
    if va_set is not None:
        best = record(-1, None, None)
        pooled_val = best["val"]["sel_eff"]
        fit_set, targets, tp, tlogp = _fit_inputs(
            tr_set,
            pairs.train,
            qv,
            pairs.fit_arrays() if pairs.n_pseudo else None,
            teacher_scores,
            norms,
            bounds,
            spec,
        )
        optim = _optimizer(student, lr, spec.weight_decay)
        n = fit_set.n_gemms
        batch = n if batch_gemms <= 0 else min(int(batch_gemms), n)
        total = max(0, int(epochs)) * math.ceil(n / batch)
        rng = np.random.default_rng(int(seed))
        student.train()
        for epoch in range(max(0, int(epochs))):
            order = rng.permutation(n)
            losses = []
            for b0 in range(0, n, batch):
                sel = np.sort(order[b0 : b0 + batch])
                rate = learning_rate(
                    steps,
                    total,
                    lr,
                    schedule=spec.lr_schedule,
                    warmup_frac=spec.warmup_frac,
                    min_frac=spec.lr_min_frac,
                )
                losses.append(
                    distill_step(
                        student,
                        optim,
                        rate,
                        fit_set,
                        sel,
                        targets,
                        tp,
                        tlogp,
                        lam=spec.lam,
                        T=spec.T,
                        stage1_recall_weight=spec.stage1_recall_weight,
                        grad_clip_norm=spec.grad_clip_norm,
                    )
                )
                steps += 1
            m = record(epoch, rate, float(np.mean(losses)))
            if m["val"]["sel_eff"] > best["val"]["sel_eff"]:
                best = m
        student.load_exported(best["state"])
    state = dict(init["state_dict"]) if best is None else best["state"]
    train_eval = stored_sel_eff(student, tr_set, spec.weight_dtype)
    sel = train_eval if best is None else best["val"]
    rec = (
        {"epoch": -1, "step": 0, "phase": POOLED_LABEL} if best is None else best["rec"]
    )
    return {
        "cell": cell,
        "recipe": "pooled_ft",
        "n_gemms": pairs.n_gemms,
        "n_train_gemms": tr_set.n_gemms,
        "n_val_gemms": 0 if va_set is None else va_set.n_gemms,
        "n_margin_gemms": 0,
        "selection_set": "training" if va_set is None else "validation",
        "selection_metric": "training" if va_set is None else "validation",
        "n_pairs": int(len(pairs.train[3])),
        "n_unmeasured_pairs": pairs.n_pseudo,
        "q_dim": len(names[0]),
        "i_dim": len(names[1]),
        "x_dim": len(names[2]),
        "embed_dim": int(init["embed_dim"]),
        "hidden_dim": int(init["hidden_dim"]),
        "inter_hidden": int(init["inter_hidden"]),
        "seed": int(cell_seed),
        "best_epoch": rec["epoch"],
        "best_step": rec["step"],
        "best_phase": rec["phase"] if rec["epoch"] >= 0 else POOLED_LABEL,
        "steps": steps,
        "stopped_early": False,
        "temperature": float(state["temperature"][0]),
        "best_sel_eff": sel["sel_eff"],
        "best_pick_us_geomean": sel.get("pick_us_geomean", float("nan")),
        "winner_us_geomean": sel.get("winner_us_geomean", float("nan")),
        "train_sel_eff": train_eval["sel_eff"],
        "val_sel_eff": None if best is None else best["val"]["sel_eff"],
        "pooled_val_sel_eff": pooled_val,
        "selection_deployed_val_sel_eff": None,
        "state_dict": state,
        "feature_norms": norms,
        "feature_bounds": bounds,
        "epoch_metrics": history,
        "elapsed_s": time.time() - t0,
        "smart_k_signatures": [],
        "view_key_hashes": [],
        "smart_k_metrics": {"mode": "none"},
    }


def pooled_capacity() -> Tuple[int, int, int]:
    """The pooled student's widths: ML_FORCE_CAPACITY ("ed,hd,ih") when set,
    as for per-leaf students, else POOLED_CAPACITY."""
    force = os.environ.get("ML_FORCE_CAPACITY")
    if force:
        try:
            ed, hd, ih = (int(v) for v in force.split(","))
            return ed, hd, ih
        except ValueError:
            pass
    return POOLED_CAPACITY


def distill_spec(args: argparse.Namespace) -> DistillSpec:
    return DistillSpec(
        lam=args.lam,
        T=args.T,
        alpha=args.alpha,
        tie_tolerance=args.tie_tolerance,
        stage1_recall_weight=args.stage1_recall_weight,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        lr_min_frac=args.lr_min_frac,
        learn_temperature=args.student_temperature == "learned",
        weight_dtype=args.weight_dtype,
        unmeasured_weight=args.unmeasured_weight,
        device=args.device,
    )


def pooled_teacher_scores(
    parts: Sequence[CellPairs], args: argparse.Namespace, threads: int
) -> np.ndarray:
    """The pooled LightGBM teacher's scores of the measured training pairs
    of `parts`, in their order: one model over all of them."""
    q, i, x, gidx, us, win = concat_pairs([p.train for p in parts])
    _, scores = fit_lgb_teacher(
        q,
        i,
        x,
        gidx,
        us,
        win,
        n_estimators=args.pooled_n_estimators,
        num_leaves=args.pooled_num_leaves,
        lr=args.lgb_lr,
        seed=args.pooled_seed,
        quiet=args.quiet,
        tie_tolerance=args.tie_tolerance,
        min_data_in_leaf=args.pooled_min_data_in_leaf,
        num_threads=threads,
    )
    return scores


def run_pooled_ft(
    cells: Mapping[str, List[dict]],
    to_train: Sequence[str],
    order: Sequence[str],
    args: argparse.Namespace,
    *,
    names: Tuple[List[str], List[str], List[str]],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    catalog: fs.Catalog,
    selector: Mapping[str, Any],
    clip: bool,
    n_workers: int,
    threads_per_worker: int,
    log_writer: Any,
) -> Dict[str, Any]:
    """The pooled recipe up to its fine-tuned cells: every cell's pairs
    (`cell_pairs`, all of `cells` in label order), the pooled teacher's
    scores of them (on --pooled-teacher-threads threads) and the pooled
    student (`train_pooled_student`, on --pooled-threads threads), then the
    cells of `to_train` fine-tuned from it (`fine_tune_cell`) on
    `threads_per_worker` threads each, in `order` when more than one worker
    trains.
    Returns {"entry": the pooled student's models.pt entry, "record": what
    metrics.json and models.pt keep of the pooled phase (the thread counts
    of each part among it), "results": {cell:
    fine-tune result}, "jobs": {cell: fine_tune_cell arguments, for a cell
    to train again here}, "parts": {cell: CellPairs}}."""
    t0 = time.time()
    labels = sorted(cells)
    prep = {
        label: dict(
            cell=label,
            gemms=cells[label],
            names=names,
            hardware=hardware,
            constants=constants,
            catalog=catalog,
            validation_frac=args.validation_frac,
            split_seed=args.seed,
            seed=cell_seed(args.seed, label),
            selector=selector,
            unmeasured_samples=args.unmeasured_samples,
            unmeasured_margin=args.unmeasured_margin,
            pseudo_negatives=args.pseudo_negatives,
        )
        for label in labels
    }
    parts: Dict[str, CellPairs] = {}
    if n_workers > 1:
        largest = sorted(labels, key=lambda c: len(cells[c]), reverse=True)
        parts = run_in_workers(
            cell_pairs,
            prep,
            largest,
            n_workers,
            threads_per_worker,
            quiet=args.quiet,
            done="prepared",
        )
    for label in labels:
        if label not in parts:
            parts[label] = cell_pairs(**prep[label])
    ordered = [parts[label] for label in labels]
    threads = int(args.pooled_threads) or _usable_cpus()
    teacher_threads = int(args.pooled_teacher_threads) or threads
    torch.set_num_threads(threads)
    spec = distill_spec(args)
    t_teacher = time.time()
    teacher = (
        None
        if args.no_teacher
        else pooled_teacher_scores(ordered, args, teacher_threads)
    )
    teacher_s = time.time() - t_teacher
    if not args.quiet:
        ui.ok(
            "pooled",
            f"{len(ordered)} cells' pairs; teacher "
            f"{'off' if teacher is None else f'scored in {ui.fmt_dur(teacher_s)}'} "
            f"on {teacher_threads} threads",
        )
    n_train = sum(len(p.train[0]) for p in ordered)
    n_val = sum(0 if p.val is None else len(p.val[0]) for p in ordered)

    def on_checkpoint(rec: Dict[str, Any]) -> None:
        log_writer.writerow(training_log_row(POOLED_LABEL, rec, n_train, n_val))
        if not args.quiet:
            print(
                f"  {ui.C.GREY}[{POOLED_LABEL}]{ui.C.ENDC} step {rec['step']:>6d}  "
                f"lr={rec['lr']:.2e}  loss={rec['train_loss']:7.4f}  "
                f"val={rec['val_sel_eff'] * 100:.2f}%",
                flush=True,
            )

    student = train_pooled_student(
        ordered,
        teacher,
        spec,
        names=names,
        clip=clip,
        epochs=args.pooled_epochs,
        batch_gemms=args.pooled_batch_gemms,
        lr=args.pooled_lr,
        eval_points=args.pooled_eval_points,
        seed=args.pooled_seed,
        capacity=pooled_capacity(),
        on_checkpoint=on_checkpoint,
    )
    if not args.quiet:
        ui.ok(
            "pooled",
            f"student {student['embed_dim']}/{student['hidden_dim']}/"
            f"{student['inter_hidden']}: best step {student['best_step']}/"
            f"{student['steps']}, val {fmt_pct(student['best_val_sel_eff'])} "
            f"({ui.fmt_dur(student['elapsed_s'])})",
        )
    ends = np.cumsum([0] + [len(p.train[3]) for p in ordered])
    scores = {
        p.cell: None if teacher is None else teacher[a:b]
        for p, a, b in zip(ordered, ends[:-1], ends[1:])
    }
    jobs = {
        label: dict(
            cell=label,
            pairs=parts[label],
            init=student["entry"],
            teacher_scores=scores[label],
            spec=spec,
            names=names,
            epochs=args.fine_tune_epochs,
            lr=args.fine_tune_lr,
            batch_gemms=args.batch_gemms,
            seed=args.pooled_seed,
            cell_seed=cell_seed(args.seed, label),
        )
        for label in to_train
    }
    # Fine-tuning here (one worker, or a retry) runs on as many threads as
    # in a worker, so that its result does not depend on where it ran.
    torch.set_num_threads(threads_per_worker)
    results: Dict[str, Any] = {}
    if n_workers > 1:
        results = run_in_workers(
            fine_tune_cell,
            jobs,
            [c for c in order if c in jobs],
            n_workers,
            threads_per_worker,
            quiet=args.quiet,
            done="fine-tuned",
        )
    record = {
        "student": {k: v for k, v in student.items() if k != "entry"},
        "teacher": (
            None
            if teacher is None
            else {
                "n_estimators": args.pooled_n_estimators,
                "num_leaves": args.pooled_num_leaves,
                "min_data_in_leaf": args.pooled_min_data_in_leaf,
                "lr": args.lgb_lr,
                "threads": teacher_threads,
                "elapsed_s": teacher_s,
            }
        ),
        "fine_tune": {
            "epochs": args.fine_tune_epochs,
            "lr": args.fine_tune_lr,
            "batch_gemms": args.batch_gemms,
            "workers": n_workers,
            "threads_per_worker": threads_per_worker,
        },
        "seed": args.pooled_seed,
        "threads": threads,
        "elapsed_s": time.time() - t0,
    }
    return {
        "entry": student["entry"],
        "record": record,
        "results": results,
        "jobs": jobs,
        "parts": parts,
    }


def pooled_whitelist(
    label: str,
    r: Mapping[str, Any],
    pairs: CellPairs,
    train_g: List[dict],
    val_g: List[dict],
    selector: Mapping[str, Any],
    *,
    names: Tuple[List[str], List[str], List[str]],
    hardware: hwlib.DeviceHardware,
    constants: hwlib.ArchConstants,
    catalog: fs.Catalog,
) -> Dict[str, Any]:
    """The whitelist of a fine-tuned cell (`ShippedSelector`, with the
    scores of its training pairs under its kept weights as stored), as the
    keys of a `train_cell` result; {} when the selector chooses none."""
    judge = ShippedSelector(
        label,
        selector,
        train_g,
        val_g,
        names=names,
        hardware=hardware,
        constants=constants,
        catalog=catalog,
    )
    if judge.index is None:
        return {}
    student = student_from_entry(model_entry(r), *(len(n) for n in names), "cpu")
    q, i, x = apply_whiten(
        *pairs.train[:3], r["feature_norms"], r.get("feature_bounds")
    )
    student.fake_quant = str(selector["weight_dtype"])
    with torch.no_grad():
        scores = _PairSet((q, i, x, *pairs.train[3:]), "cpu").scores(student)
    wl = judge.whitelist(scores.numpy(), True)
    return {
        "smart_k_signatures": wl["sigs"],
        "view_key_hashes": wl["view_key_hashes"],
        "smart_k_metrics": wl["metrics"],
    }


# ── deployed evaluation ───────────────────────────────────────────────────────


def model_entry(r: Mapping[str, Any]) -> Dict[str, Any]:
    """models.pt entry of a train_cell result."""
    entry = {
        "state_dict": r["state_dict"],
        "feature_norms": r["feature_norms"],
        "embed_dim": r["embed_dim"],
        "hidden_dim": r["hidden_dim"],
        "inter_hidden": r["inter_hidden"],
        "best_epoch": r["best_epoch"],
        "smart_k_signatures": r.get("smart_k_signatures") or [],
    }
    if r.get("feature_bounds"):
        entry["feature_bounds"] = r["feature_bounds"]
    if r.get("view_key_hashes"):
        entry["view_key_hashes"] = list(r["view_key_hashes"])
    return entry


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


PROBE_WIDTH = 4


def probe_model(
    catalog: fs.Catalog,
    serving: Mapping[str, Any],
    clip: bool,
    arch: str,
    constants: hwlib.ArchConstants,
    weight_dtype: str,
) -> bytes:
    """Model file bytes of an untrained one-cell bundle in the format a run
    with these options deploys: its catalog and attribute spec, serving
    options, clip bounds and weight dtype."""
    names = catalog.names()
    student = GenericTwoTower(
        *(len(n) for n in names),
        embed_dim=PROBE_WIDTH,
        hidden_dim=PROBE_WIDTH,
        inter_hidden=PROBE_WIDTH,
    )

    def per_feature(a: str, va: float, b: str, vb: float) -> Dict[str, List[float]]:
        return {
            f"{t}_{k}": [v] * len(n)
            for t, n in zip("qix", names)
            for k, v in ((a, va), (b, vb))
        }

    entry: Dict[str, Any] = {
        "state_dict": student.export_state(),
        "feature_norms": per_feature("mean", 0.0, "std", 1.0),
        "embed_dim": PROBE_WIDTH,
        "hidden_dim": PROBE_WIDTH,
        "inter_hidden": PROBE_WIDTH,
        "smart_k_signatures": [],
    }
    meta: Dict[str, Any] = {}
    if catalog.version == 3:
        meta = {
            "catalog": "v3",
            "attribute_spec": [a.to_dict() for a in catalog.spec],
            "clip": clip,
        }
        if serving.get("trained_key_gate"):
            meta["trained_view_keys"] = [1]
        if clip:
            entry["feature_bounds"] = per_feature("lo", -1.0, "hi", 1.0)
    bundle = cell_bundle(grid.all_cell_labels()[0], entry, names, serving, meta)
    return ev.mlrec.write_model(bundle, None, arch, constants, weight_dtype)


def engine_round_trip_errors(data: bytes, tw: Any) -> List[str]:
    """What keeps the engine from reading model file `data` back as written:
    a load error, or each header field it reports otherwise (fields an older
    module does not report are not compared)."""
    try:
        model = ev.load_model_bytes(data, tw)
        info = tw.describe(model)
    except Exception as e:  # noqa: BLE001
        return [str(e)]
    parsed = ev.mlrec.read_model(data)
    want: Dict[str, Any] = {
        "format_version": parsed["version"],
        "feature_catalog_hash": parsed["feature_catalog_hash"],
        "n_cells": len(parsed["cells"]),
        "n_splits": len(parsed["splits"]),
        "query_dim": parsed["dims"][0],
        "item_dim": parsed["dims"][1],
        "inter_dim": parsed["dims"][2],
    }
    if parsed["version"] == 3:
        want.update(parsed["serving"], clip=parsed["clip"])
    out = []
    for key, value in want.items():
        if not hasattr(info, key):
            continue
        got = getattr(info, key)
        if (str(got) if isinstance(value, str) else type(value)(got)) != value:
            out.append(f"{key} reads back as {got!r}, written {value!r}")
    return out


def serving_options(args: argparse.Namespace) -> Dict[str, Any]:
    """The serving options written into the model file header."""
    if getattr(args, "catalog", "v2") == "v3":
        return {
            "small_batched_rule": bool(getattr(args, "small_batched_rule", False)),
            "dot2_rule": bool(getattr(args, "dot2_rule", False)),
            "non_temporal_rule": bool(getattr(args, "non_temporal_rule", False)),
            "non_temporal_family_rule": bool(
                getattr(args, "non_temporal_family_rule", False)
            ),
            "whitelist_tier": getattr(args, "whitelist", "none") != "none",
            "launch_keys": bool(getattr(args, "launch_keys", False)),
            "trained_key_gate": not getattr(args, "no_trained_key_gate", False),
            "cascade_keys": int(getattr(args, "cascade_k", 32) or 0),
        }
    return {
        "non_temporal_rule": not getattr(args, "no_non_temporal_rule", False),
        "cascade_keys": int(getattr(args, "cascade_k", 0) or 0),
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
    serving: Optional[Mapping[str, Any]] = None,
    bundle_meta: Optional[Mapping[str, Any]] = None,
    pick_measurement: str = ev.DEFAULT_PICK_MEASUREMENT,
    off_list_sel_eff: Union[None, float, ev.OffListCharge] = None,
) -> Tuple[Dict[str, Any], List[ev.GemmResult]]:
    """Deployed measurements of one cell (`cell_bundle`) and its validation
    GemmResults, picks charged by `pick_measurement` and `off_list_sel_eff`
    (lib/evaluate.py).
    `bundle_meta`: the bundle's catalog keys (`catalog`, `attribute_spec`,
    `contexts`, `clip`)."""
    bundle = cell_bundle(label, model_entry(r), names, serving, bundle_meta)
    out = ev.evaluate_bundle(
        bundle,
        list(train_g) + list(val_g),
        arch=arch,
        constants=constants,
        hardware=hardware,
        weight_dtype=weight_dtype,
        pools=pools,
        tw=tw,
        pick_measurement=pick_measurement,
        off_list_sel_eff=off_list_sel_eff,
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


# ── whitelist tier ────────────────────────────────────────────────────────────


def whitelist_tier_errors(
    args: argparse.Namespace, v3: bool, engine: bool
) -> List[str]:
    """Why --whitelist-tier-k cannot give the bundle a tier with these
    options; none when it is 0."""
    if int(getattr(args, "whitelist_tier_k", 0) or 0) <= 0:
        return []
    return [
        text
        for text, wrong in (
            ("needs --catalog v3", not v3),
            (
                f"needs --whitelist none, not {args.whitelist}: the tier "
                f"replaces the smart_K whitelist",
                args.whitelist != "none",
            ),
            (
                "lists view keys, and --launch-keys scores each launch class "
                "of a view key on its own",
                bool(getattr(args, "launch_keys", False)),
            ),
            (
                "needs the pool the runtime ranks: --library-dir and --library-stem",
                args.library_dir is None or args.library_stem is None,
            ),
            ("scores with the engine: drop --skip-deployed-eval", not engine),
        )
        if wrong
    ]


def whitelist_tier(
    args: argparse.Namespace,
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    catalog: fs.Catalog,
    hardware: hwlib.DeviceHardware,
    identity: Optional[ki.LibraryIdentity],
    blocklist: Optional[cx.Blocklist],
    split_tree: Mapping[str, sc.SplitRule],
) -> wt.Tier:
    """The whitelist tier of `bundle` with --whitelist-tier-k keys per cell
    (lib/whitelist_tier.py): chosen on the training side of the GEMMs of
    --enriched-csv-dir, read again as this stage read them (the library's
    identity, --repeat-aggregation, the blocklist) and split as it split
    them, over the pool of --library-dir / --library-stem, with the scores
    of the bundle at --weight-dtype, in --train-workers processes."""
    this = sys.modules[__name__]
    workers = max(1, int(args.train_workers or 1))
    bench = ra.load_bench(
        [Path(d) for d in args.enriched_csv_dir],
        jobs=workers,
        identity=identity,
        aggregation=args.repeat_aggregation,
        blocklist=blocklist,
    )
    gemms = wt.training_gemms(
        bench.gemms, split_tree, args.validation_frac, args.seed, s5=this
    )
    u = wt.universe(
        cfg, args.library_dir, args.library_stem, args.hardware_device, catalog.spec
    )
    return wt.build(
        bundle,
        gemms,
        k=args.whitelist_tier_k,
        u=u,
        hardware=hardware,
        arch=args.arch,
        weight_dtype=args.weight_dtype,
        workers=workers,
        s5=this,
        say=None if args.quiet else (lambda m: ui.info("tier", m)),
    )


# ── main ──────────────────────────────────────────────────────────────────────


def training_catalog(args: argparse.Namespace, cfg: Mapping[str, Any]) -> fs.Catalog:
    """The catalog to train: --catalog, else the config's `train.catalog`,
    else v2; a v3 catalog's spec from --attribute-spec, else the config's
    `train.attribute_spec`, else DEFAULT_ATTRIBUTE_SPEC; its launch plans are
    hipBLASLt's, the launch the measurements ran with, for the library of
    --library-stem, else of the config's `hipblaslt.library_stem`, else for
    --arch, on a device of the XCD count of --hardware-device
    (`hwlib.device_num_xcd`)."""
    train_cfg = (cfg.get("train") or {}) if isinstance(cfg, Mapping) else {}
    name = args.catalog or str(train_cfg.get("catalog") or "v2")
    args.catalog = name
    if name != "v3":
        if args.attribute_spec is not None:
            raise ValueError("--attribute-spec needs --catalog v3")
        return fs.Catalog(name)
    if args.attribute_spec is not None:
        raw = yaml.safe_load(args.attribute_spec.read_text())
    else:
        raw = train_cfg.get("attribute_spec")
    spec = fs.DEFAULT_ATTRIBUTE_SPEC if raw is None else fs.parse_attribute_spec(raw)
    stem = getattr(args, "library_stem", None) or (
        (cfg.get("hipblaslt") or {}).get("library_stem")
        if isinstance(cfg, Mapping)
        else None
    )
    num_xcd = hwlib.device_num_xcd(cfg, int(getattr(args, "hardware_device", 0) or 0))
    launch = (
        hipblaslt_launch.HipblasltLaunch.for_library(str(stem), num_xcd)
        if stem
        else hipblaslt_launch.HipblasltLaunch(str(args.arch), num_xcd)
    )
    return fs.Catalog("v3", spec, launch)


def attach_plans(
    gemms: Sequence[Dict[str, Any]],
    plans: Sequence[Path],
    context_plans: Sequence[Sequence[str]] = (),
    identity: Optional[ki.LibraryIdentity] = None,
) -> int:
    """Read --plan / --context-plan and set every GEMM's planned list
    (`ev.attach_plans`, its indices those of `identity`'s library); returns
    the number of GEMMs listed."""
    by_context: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for budget, schedule, path in context_plans:
        key = (int(budget), str(schedule).lower())
        by_context.setdefault(key, []).append(ev.read_plan(path))
    docs = [ev.read_plan(p) for p in plans]
    return ev.attach_plans(gemms, docs, by_context, identity=identity)


def off_list_option(value: str) -> Union[str, float]:
    """--off-list-sel-eff: a number, or auto."""
    return "auto" if value.strip().lower() == "auto" else float(value)


def resolve_off_list_charge(
    args: argparse.Namespace,
) -> Tuple[Union[None, float, ev.OffListCharge], Optional[Dict[str, Any]]]:
    """(the charge of an off-list pick for `ev.evaluate_model`, the report
    metrics.json keeps) of --off-list-sel-eff; with auto, from
    --off-list-rebench when it exists, else --off-list-fallback."""
    value = args.off_list_sel_eff
    if value is None:
        return None, None
    if value != "auto":
        return float(value), {"source": "configured", "sel_eff": float(value)}
    path = args.off_list_rebench
    records = None
    if path is not None and Path(path).is_file():
        records = json.loads(Path(path).read_text())
    charge, report = mr.off_list_charge(
        records, min_misses=args.off_list_min_misses, fallback=args.off_list_fallback
    )
    report["rebench_file"] = None if records is None else str(path)
    return charge, report


def describe_off_list_charge(report: Mapping[str, Any]) -> str:
    se = report.get("sel_eff")
    what = "the slowest latency measured" if se is None else f"{se * 100:.2f}%"
    if report["source"] == "configured":
        return f"off-list picks charged {what}"
    if report["source"] == "fallback":
        return (
            f"off-list picks charged {what}: {report['n']} measured off-list "
            f"misses, fewer than {report['min_misses']}"
        )
    ctx = ", ".join(
        f"{k} {v * 100:.2f}% (n={report['n_by_context'][k]})"
        for k, v in report["by_context"].items()
    )
    return (
        f"off-list picks charged {what}, the geomean of {report['n']} re-benched "
        f"off-list misses" + (f"; by context: {ctx}" if ctx else "")
    )


def drop_non_exclusive(
    cells: Mapping[str, List[dict]], n_cu: int
) -> Tuple[Dict[str, List[dict]], int]:
    """Cells without the GEMMs measured under a non-exclusive context, and
    their number."""
    out: Dict[str, List[dict]] = {}
    n = 0
    for label, gemms in cells.items():
        kept = [g for g in gemms if ev.context_of(g).is_exclusive(n_cu)]
        n += len(gemms) - len(kept)
        if kept:
            out[label] = kept
    return out, n


def declared_contexts(
    gemms: Sequence[Mapping[str, Any]],
    n_cu: int,
    min_cu_fraction: Optional[float] = None,
    schedules: Optional[str] = None,
) -> Dict[str, Any]:
    """The contexts an MLREC_v3 model declares: usable-CU fractions from
    `min_cu_fraction` (default: the smallest in `gemms`) to 1, and the
    `schedules` (default: default plus those in `gemms`)."""
    seen = [ev.context_of(g) for g in gemms]
    if min_cu_fraction is None:
        min_cu_fraction = min(
            (c.effective_cus(n_cu) / float(n_cu) for c in seen), default=1.0
        )
    lo = float(min_cu_fraction)
    if not 0.0 < lo <= 1.0:
        raise ValueError(f"min CU fraction {lo} must be in (0, 1]")
    if schedules:
        names = {x.strip().lower() for x in schedules.split(",") if x.strip()}
        unknown = names - set(fs.SCHEDULES)
        if unknown:
            raise ValueError(f"unknown schedules {sorted(unknown)}")
    else:
        names = {"default"} | {c.schedule for c in seen}
    return {
        "min_cu_fraction": lo,
        "max_cu_fraction": 1.0,
        "schedules": [x for x in fs.SCHEDULES if x in names],
    }


def uncovered_base_cells(models: Mapping[str, Any]) -> List[str]:
    """Base cells no model label is or lies under."""
    return [
        b
        for b in grid.all_cell_labels()
        if not any(m == b or m.startswith(b + "#") for m in models)
    ]


def _load_prior_models(
    prior_round_dirs: Sequence[Path], catalog: fs.Catalog = fs.V2_CATALOG
) -> Tuple[Dict[str, Any], List[int]]:
    """Models of the latest prior round's bundle, without its whitelist tier
    (a bundle with --whitelist-tier-k gives every cell a new one), and the
    view keys it records as trained (`trained_view_keys`); SystemExit when
    it exists but cannot be read, or was trained with another catalog or
    attribute spec, since the cells this round does not train would be
    lost."""
    if not prior_round_dirs:
        return {}, []
    path = Path(prior_round_dirs[-1]) / "stage05" / "models.pt"
    if not path.exists():
        return {}, []
    try:
        prior = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:
        ui.err("carry", f"cannot load the prior bundle {path}: {e}")
        raise SystemExit(1)
    prior_catalog = str(prior.get("catalog") or "v2")
    try:
        prior_spec = fs.parse_attribute_spec(prior.get("attribute_spec") or [])
    except ValueError as e:
        ui.err("carry", f"the prior bundle {path} has an invalid attribute spec: {e}")
        raise SystemExit(1)
    if prior_catalog != catalog.name or prior_spec != catalog.spec:
        ui.err(
            "carry",
            f"the prior bundle {path} uses catalog {prior_catalog} with "
            f"{len(prior_spec)} attributes; this round trains {catalog.name} with "
            f"{len(catalog.spec)}: its cells cannot be carried forward",
        )
        raise SystemExit(1)
    trained = [int(h) for h in prior.get("trained_view_keys") or []]
    return dict(wt.strip(prior).get("models") or {}), trained


def trained_view_keys(gemms: Iterable[Mapping[str, Any]], catalog: fs.Catalog) -> set:
    """The `features.view_key_hash` of the v3 view key of every kernel with a
    row in `gemms`: what an MLREC_v3 model's trained-key gate lists."""
    keys = {
        fs.v3_view_key(fs.config_kwargs_from_row(c), c.get("attributes"), catalog.spec)
        for g in gemms
        for c in g.get("candidates", ())
    }
    return {fs.view_key_hash(k) for k in keys}


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


def pooled_mean(cells: Iterable[Mapping[str, Any]], key: str, field: str) -> float:
    """Geomean of the cells' `[key][field]` weighted by each one's `n_eval`:
    a cell's summary covers only the GEMMs it evaluated, so this is the
    mean over the pooled evaluated GEMMs."""
    return wlog_mean(
        [
            ((c.get(key) or {}).get(field), int((c.get(key) or {}).get("n_eval", 0)))
            for c in cells
        ]
    )


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


class _RaisingParser(argparse.ArgumentParser):
    """Raises ValueError where ArgumentParser prints its usage and exits."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def arg_parser(
    parser_class: Type[argparse.ArgumentParser] = argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    ap = parser_class(
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
        "--pick-measurement",
        choices=ev.PICK_MEASUREMENTS,
        default="warmup",
        help="how the deployed evaluation and early stopping charge a pick: "
        "warmup: a skip row counts with its warm-up latency and a kernel off the "
        "plan's list is a miss; timed: both are misses; any: a skip row counts "
        "and an off-list kernel is passed over (lib/evaluate.py)",
    )
    ap.add_argument(
        "--plan",
        type=Path,
        action="append",
        default=None,
        help="repeatable stage02 plan.json (rounds and seed benches) of the "
        "training data: the deployed evaluation and early stopping count a "
        "pick off its GEMM's planned list as a miss instead of passing it over",
    )
    ap.add_argument(
        "--off-list-sel-eff",
        type=off_list_option,
        default=None,
        help="charge a pick off its GEMM's planned list the winner's latency "
        "over this selection efficiency, in (0, 1], instead of the slowest "
        "latency the GEMM measured; auto: the efficiency --off-list-rebench "
        "measured",
    )
    ap.add_argument(
        "--off-list-rebench",
        type=Path,
        default=None,
        help="with --off-list-sel-eff auto: stage04b's rebench_misses.json, "
        "whose re-benched off-list misses give the charge (lib/miss_rebench.py "
        "off_list_charge)",
    )
    ap.add_argument(
        "--off-list-min-misses",
        type=int,
        default=30,
        help="auto: measured off-list misses needed overall, and per context "
        "for a context's own charge",
    )
    ap.add_argument(
        "--off-list-fallback",
        type=float,
        default=None,
        help="auto: the charge without enough measured off-list misses, in "
        "(0, 1] (default: the slowest latency measured)",
    )
    ap.add_argument(
        "--context-plan",
        nargs=3,
        action="append",
        default=None,
        metavar=("CU_BUDGET", "SCHEDULE", "PLAN"),
        help="repeatable plan.json a context's GEMMs were re-planned with "
        "(stage02/contexts/<label>/plan.json); they take their lists from it",
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
        "--cascade-k",
        type=int,
        default=None,
        help="serve cells without a whitelist in two stages: every feasible kernel "
        "by the embedding term, the best this many by the full score (0: all; "
        "default 0 for v2, 32 for v3)",
    )
    ap.add_argument(
        "--catalog",
        choices=("v2", "v3"),
        default=None,
        help="feature catalog and model file format (MLREC_v2 / MLREC_v3; "
        "default: the config's train.catalog, else v2)",
    )
    ap.add_argument(
        "--attribute-spec",
        type=Path,
        default=None,
        help="v3: JSON or YAML item-attribute spec (list of {name, default, "
        "transform, values}); default: the config's train.attribute_spec, else "
        "lib.features.DEFAULT_ATTRIBUTE_SPEC",
    )
    ap.add_argument(
        "--whitelist",
        choices=("signature", "view-key", "none"),
        default=None,
        help="kernel whitelist per cell, keyed by signature or (v3) view key; "
        "none serves every feasible kernel through the cascade (default: "
        "signature for v2, none for v3)",
    )
    ap.add_argument(
        "--whitelist-tier-k",
        type=int,
        default=0,
        help="v3, with --whitelist none: after training, give every cell of "
        "the bundle a whitelist tier of this many view keys, chosen on the "
        "training GEMMs with the deployed model's scores over the pool of "
        "--library-dir / --library-stem; the cascade serves problems none of "
        "them suits (lib/whitelist_tier.py; 0: no tier)",
    )
    ap.add_argument(
        "--no-clip-features",
        action="store_true",
        help="v3: do not store the training feature ranges the engine clamps to",
    )
    ap.add_argument(
        "--small-batched-rule",
        action="store_true",
        help="v3: serve with the soft small-batched one-tile rule",
    )
    ap.add_argument(
        "--dot2-rule",
        action="store_true",
        help="v3: serve with the soft Dot2 M < 3 rule",
    )
    ap.add_argument(
        "--non-temporal-rule",
        action="store_true",
        help="v3: serve with the soft non-temporal rule",
    )
    ap.add_argument(
        "--non-temporal-family-rule",
        action="store_true",
        help="v3, with --non-temporal-rule: in the family-wise rule's region "
        "(lib/feasibility.py in_family_region), a kernel family the rule "
        "rejects whole keeps one member",
    )
    ap.add_argument(
        "--no-trained-key-gate",
        action="store_true",
        help="v3: serve without the trained-key gate, under which the model "
        "scores only kernels whose view key has training rows unless none of "
        "them is feasible",
    )
    ap.add_argument(
        "--served-candidates",
        action="store_true",
        help="v3: train on one candidate per key, the kernel the runtime runs "
        "for it (the key's first pool kernel with a row), instead of every "
        "measured kernel",
    )
    ap.add_argument(
        "--launch-keys",
        action="store_true",
        help="v3: serve with launch keys, so that the engine scores each launch "
        "class of a view key on its own",
    )
    ap.add_argument(
        "--min-cu-fraction",
        type=float,
        default=None,
        help="v3: smallest usable-CU fraction the model declares (default: the "
        "smallest in the training data)",
    )
    ap.add_argument(
        "--schedules",
        default=None,
        help="v3: comma-separated schedules the model declares (default: "
        "default plus those in the training data)",
    )
    ap.add_argument(
        "--min-base-cell-gemms",
        type=int,
        default=10,
        help="v3: base cells with at least this many GEMMs are trained even "
        "below --min-cell-gemms, so the bundle covers as many base cells as "
        "the data allows",
    )
    ap.add_argument(
        "--no-non-temporal-rule",
        action="store_true",
        help="choose whitelists without the non-temporal feasibility rule "
        "(for models served without it)",
    )
    ap.add_argument(
        "--oracle-seed",
        type=int,
        default=0,
        help="force this many oracle max-coverage signatures into each " "whitelist",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="passes over each cell's training GEMMs",
    )
    ap.add_argument(
        "--batch-gemms",
        type=int,
        default=DEFAULT_BATCH_GEMMS,
        help="GEMMs per optimizer step, each with all its candidates (0: all)",
    )
    ap.add_argument("--lr", type=float, default=DEFAULT_LR, help="peak learning rate")
    ap.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="cosine")
    ap.add_argument(
        "--warmup-frac",
        type=float,
        default=0.05,
        help="fraction of the steps with a linear learning-rate warmup",
    )
    ap.add_argument(
        "--lr-min-frac",
        type=float,
        default=0.02,
        help="cosine schedule: final learning rate as a fraction of --lr",
    )
    ap.add_argument(
        "--eval-points",
        type=int,
        default=DEFAULT_EVAL_POINTS,
        help="checkpoints measured over the schedule to keep the best",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="stop after this many measured checkpoints without a better one "
        "(0: train the whole schedule)",
    )
    ap.add_argument(
        "--student-temperature",
        choices=("learned", "fixed"),
        default=DEFAULT_STUDENT_TEMPERATURE,
        help="learn the embedding-term temperature (log-parameterized) or "
        "keep it at 1",
    )
    ap.add_argument(
        "--qat-frac",
        type=float,
        default=DEFAULT_QAT_FRAC,
        help="int8 / int4 weights: continue from the kept checkpoint for this "
        "fraction of the schedule's steps with the weights rounded as stored "
        "(0: off)",
    )
    ap.add_argument(
        "--qat-lr-frac",
        type=float,
        default=0.1,
        help="peak learning rate of the quantization-aware phase, as a "
        "fraction of --lr",
    )
    ap.add_argument(
        "--recipe",
        choices=RECIPES,
        default="per_leaf",
        help="per_leaf: every cell's student trains from fresh weights on the "
        "cell's GEMMs, distilled from a LightGBM teacher of its own; "
        "pooled_ft: one student trains on every cell's GEMMs, distilled from "
        "one LightGBM teacher over all of them, and every cell fine-tunes it on "
        "its own GEMMs",
    )
    ap.add_argument(
        "--pooled-epochs",
        type=int,
        default=12,
        help="pooled_ft: passes of the pooled student over every cell's "
        "training GEMMs",
    )
    ap.add_argument(
        "--pooled-batch-gemms",
        type=int,
        default=64,
        help="pooled_ft: GEMMs per optimizer step of the pooled student (0: all)",
    )
    ap.add_argument(
        "--pooled-lr",
        type=float,
        default=2e-3,
        help="pooled_ft: peak learning rate of the pooled student",
    )
    ap.add_argument(
        "--pooled-eval-points",
        type=int,
        default=24,
        help="pooled_ft: checkpoints of the pooled student measured on every "
        "cell's validation GEMMs to keep the best",
    )
    ap.add_argument(
        "--pooled-n-estimators",
        type=int,
        default=1000,
        help="pooled_ft: trees of the pooled LightGBM teacher",
    )
    ap.add_argument(
        "--pooled-num-leaves",
        type=int,
        default=255,
        help="pooled_ft: leaves per tree of the pooled teacher",
    )
    ap.add_argument(
        "--pooled-min-data-in-leaf",
        type=int,
        default=50,
        help="pooled_ft: fewest pairs per leaf of the pooled teacher",
    )
    ap.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=12,
        help="pooled_ft: passes of each cell's fine-tuning over its training "
        "GEMMs, in --batch-gemms batches (0: every cell keeps the pooled student)",
    )
    ap.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-3,
        help="pooled_ft: peak learning rate of the fine-tuning",
    )
    ap.add_argument(
        "--pooled-seed",
        type=int,
        default=0,
        help="pooled_ft: seed of the pooled teacher, of the pooled student's "
        "weights and batches and of the fine-tuning batches (--seed still "
        "splits the GEMMs and draws the pseudo-negatives)",
    )
    ap.add_argument(
        "--pooled-threads",
        type=int,
        default=0,
        help="pooled_ft: threads of the pooled student (0: every CPU the stage "
        "may use); the fine-tuning of a cell takes --train-threads-per-worker",
    )
    ap.add_argument(
        "--pooled-teacher-threads",
        type=int,
        default=0,
        help="pooled_ft: threads of the pooled LightGBM teacher, whose model "
        "depends on them (0: --pooled-threads)",
    )
    ap.add_argument(
        "--boundary-margin",
        type=float,
        default=0.0,
        help="octaves outside each cell's box on its inner sides from which "
        "neighbouring cells' training GEMMs join its loss (0: off)",
    )
    ap.add_argument(
        "--boundary-weight",
        type=float,
        default=0.5,
        help="loss weight of those margin GEMMs, in (0, 1]",
    )
    ap.add_argument(
        "--pseudo-negatives",
        choices=PSEUDO_NEGATIVES,
        default=None,
        help="kernels added to each planned training GEMM's loss as clearly "
        "slower than its winner (needs --plan and the library): none; random, "
        "feasible pool kernels off its list; targeted (v3), keys no training "
        "GEMM of the cell measured (default: random with --unmeasured-samples "
        "above 0, else none)",
    )
    ap.add_argument(
        "--unmeasured-samples",
        type=int,
        default=None,
        help="pseudo-negatives per planned training GEMM (default: 16 for "
        "targeted, else 0, which random does not take)",
    )
    ap.add_argument(
        "--unmeasured-margin",
        type=float,
        default=None,
        help="their pseudo latency: the GEMM's winner x (1 + margin) (default: "
        "1.0 for targeted, else 0.05)",
    )
    ap.add_argument(
        "--unmeasured-weight",
        type=float,
        default=None,
        help="scale of their soft-CE target, in (0, 1] (default: 0.1 for "
        "targeted, else 0.3)",
    )
    ap.add_argument(
        "--stage1-recall-weight",
        type=float,
        default=0.0,
        help="weight of a loss training the embedding term alone to rank as "
        "the full score does, for the cascade's first stage (0: off)",
    )
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
        "--repeat-aggregation",
        choices=repeats.AGGREGATIONS,
        default=None,
        help="how several rows of one kernel of one GEMM merge into its label "
        "(lib/repeats.py; default: the config's train.repeat_aggregation, else "
        f"{repeats.DEFAULT_AGGREGATION})",
    )
    ap.add_argument(
        "--tie-tolerance",
        type=float,
        default=None,
        help="relative latency difference within which a candidate ties with "
        "its GEMM's winner in the training targets (default: the config's "
        f"stage08.tie_tolerance, else {DEFAULT_TIE_TOLERANCE})",
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
        help="torch/BLAS/LightGBM threads per worker (0: usable CPUs // workers)",
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
    ap.add_argument(
        "--skip-cells",
        default=None,
        help="comma-separated labels not to train; like the cells --only-cells "
        "leaves out, their GEMMs still reach the pooled phase of --recipe "
        "pooled_ft",
    )
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=50,
        help="cells with fewer GEMMs are not trained",
    )
    ap.add_argument("--quiet", action="store_true")
    return ap


def run_config(path: Optional[Path]) -> Dict[str, Any]:
    """The --config-yaml mapping, {} without one."""
    if path is None:
        return {}
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def default_pseudo_negatives(unmeasured_samples: Optional[int]) -> str:
    """The --pseudo-negatives draw of an invocation that does not name one:
    random when --unmeasured-samples asks for some, as before the option
    existed, else none."""
    return "random" if int(unmeasured_samples or 0) > 0 else "none"


def resolve_pseudo_negatives(args: argparse.Namespace) -> None:
    """Fill in --pseudo-negatives (`default_pseudo_negatives`) and the
    --unmeasured-* options its draw leaves unset
    (PSEUDO_NEGATIVE_DEFAULTS)."""
    if getattr(args, "pseudo_negatives", None) is None:
        samples = getattr(args, "unmeasured_samples", None)
        args.pseudo_negatives = default_pseudo_negatives(samples)
    for key, value in PSEUDO_NEGATIVE_DEFAULTS[args.pseudo_negatives].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def resolve_options(args: argparse.Namespace, cfg: Mapping[str, Any]) -> fs.Catalog:
    """Fill in the options whose defaults come from the config, the catalog
    or another option (--tie-tolerance, --repeat-aggregation, --catalog,
    --cascade-k, --whitelist, the pseudo-negatives' options) and return the
    catalog to train (`training_catalog`)."""
    if args.tie_tolerance is None:
        args.tie_tolerance = config_tie_tolerance(cfg)
    if args.repeat_aggregation is None:
        args.repeat_aggregation = repeats.config_aggregation(cfg)
    resolve_pseudo_negatives(args)
    catalog = training_catalog(args, cfg)
    v3 = catalog.version == 3
    if args.cascade_k is None:
        args.cascade_k = 32 if v3 else 0
    if args.whitelist is None:
        args.whitelist = "none" if v3 else "signature"
    return catalog


def recorded_options(args: argparse.Namespace) -> Dict[str, Any]:
    """The arguments of an invocation as records name them: the
    pseudo-negatives' options resolved (`resolve_pseudo_negatives`) and the
    draw None where --unmeasured-samples implies it
    (`default_pseudo_negatives`), so that a record from before the draws
    compares equal while its training is the same."""
    resolved = argparse.Namespace(**vars(args))
    resolve_pseudo_negatives(resolved)
    opts = vars(resolved)
    if opts["pseudo_negatives"] == default_pseudo_negatives(opts["unmeasured_samples"]):
        opts["pseudo_negatives"] = None
    return opts


def model_options(args: argparse.Namespace, catalog: fs.Catalog) -> Dict[str, Any]:
    """The model-shaping options of an invocation `resolve_options` resolved:
    every argument but NOT_MODEL_SHAPING, as records name them
    (`recorded_options`), with the catalog's attribute spec for
    --attribute-spec and the kernel identity rule its keys follow
    (`identity_rule`, lib/kernel_identity.py). A policy stays as given:
    --off-list-sel-eff auto, not the charge it measures; an unset
    --min-cu-fraction, not the fraction the data declares."""
    opts = {
        k: v for k, v in recorded_options(args).items() if k not in NOT_MODEL_SHAPING
    }
    opts["attribute_spec"] = [a.to_dict() for a in catalog.spec]
    opts["identity_rule"] = ki.IDENTITY_RULE
    return jsonable(opts)


# The values model-shaping options had before they were options: a record
# or command without one was trained with this value, not with its default.
PREDATING_OPTIONS: Dict[str, Any] = {
    "repeat_aggregation": repeats.PREDATING_AGGREGATION,
}
# The same for what `model_options` records beside the arguments. No pool
# trained before the identity rule was recorded held a kernel with a
# temporal hint on an operand the tensor data mover loads, so their keys
# are those of the rule `tdm_th_v1`.
PREDATING_RECORDS: Dict[str, Any] = {
    "identity_rule": "tdm_th_v1",
}


def command_model_options(argv: Sequence[str]) -> Dict[str, Any]:
    """`model_options` of a stage05 command line (the arguments after the
    script), resolved with the --config-yaml it names; arguments this stage05
    no longer defines are ignored, and a PREDATING_OPTIONS option the
    command does not give takes its value from before it was an option.
    ValueError, OSError or yaml.YAMLError when it does not parse or
    resolve."""
    args, _unknown = arg_parser(_RaisingParser).parse_known_args(list(argv))
    for key, value in PREDATING_OPTIONS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return model_options(args, resolve_options(args, run_config(args.config_yaml)))


def changed_model_options(
    trained: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Tuple[Any, Any]]:
    """{option: (trained with, current)} of the model-shaping options of
    `current` (`model_options`) that `trained` gives another value; an
    option `trained` does not name predates the option, so it had its
    PREDATING_OPTIONS (PREDATING_RECORDS) value, else its default."""
    ap = arg_parser()
    out: Dict[str, Tuple[Any, Any]] = {}
    for key, value in current.items():
        if key in trained:
            old = trained[key]
        elif key in PREDATING_OPTIONS:
            old = PREDATING_OPTIONS[key]
        elif key in PREDATING_RECORDS:
            old = PREDATING_RECORDS[key]
        else:
            old = jsonable(ap.get_default(key))
        if old != value:
            out[key] = (old, value)
    return out


# The arguments that name files or directories. Only these are checked
# against the audit set: other values, such as a cell list, can be longer than
# a file name may be.
PATH_ARGUMENTS = (
    "enriched_csv_dir",
    "prior_round_dir",
    "current_round_dir",
    "output_dir",
    "config_yaml",
    "plan",
    "off_list_rebench",
    "context_plan",
    "library_dir",
    "attribute_spec",
)


def main() -> int:
    args = arg_parser().parse_args()
    paths = auditlib.existing_paths(getattr(args, name) for name in PATH_ARGUMENTS)
    leaks = auditlib.audit_inputs(paths)
    if leaks:
        for leak in leaks:
            ui.err("audit", leak)
        raise SystemExit(2)

    if not _TORCH_OK:
        ui.err("err", "torch is not importable; pip install -r requirements.txt")
        raise SystemExit(1)
    if not _LGB_OK and not args.no_teacher:
        ui.err("err", "lightgbm is not importable; install it or pass --no-teacher")
        raise SystemExit(1)

    cfg = run_config(args.config_yaml)
    try:
        hardware = hwlib.device_hardware(cfg, args.hardware_device)
        constants = hwlib.arch_constants(args.arch)
    except (ValueError, RuntimeError, OSError) as e:
        ui.err("hardware", str(e))
        raise SystemExit(1)
    try:
        catalog = resolve_options(args, cfg)
    except (ValueError, OSError) as e:
        ui.err("catalog", str(e))
        raise SystemExit(1)
    bad = [
        f"--{name.replace('_', '-')} {getattr(args, name)}"
        for name, ok in (
            ("tie_tolerance", lambda v: 0 <= v < 1),
            ("batch_gemms", lambda v: v >= 0),
            ("eval_points", lambda v: v >= 1),
            ("patience", lambda v: v >= 0),
            ("warmup_frac", lambda v: 0 <= v < 1),
            ("lr_min_frac", lambda v: 0 <= v <= 1),
            ("qat_frac", lambda v: v >= 0),
            ("qat_lr_frac", lambda v: v > 0),
            ("boundary_margin", lambda v: v >= 0),
            ("boundary_weight", lambda v: 0 < v <= 1),
            ("off_list_sel_eff", lambda v: v is None or v == "auto" or 0 < v <= 1),
            ("off_list_min_misses", lambda v: v >= 1),
            ("off_list_fallback", lambda v: v is None or 0 < v <= 1),
            ("unmeasured_samples", lambda v: v >= 0),
            ("unmeasured_margin", lambda v: v >= 0),
            ("unmeasured_weight", lambda v: 0 < v <= 1),
            ("stage1_recall_weight", lambda v: v >= 0),
            ("pooled_epochs", lambda v: v >= 1),
            ("pooled_batch_gemms", lambda v: v >= 0),
            ("pooled_lr", lambda v: v > 0),
            ("pooled_eval_points", lambda v: v >= 1),
            ("pooled_n_estimators", lambda v: v >= 1),
            ("pooled_num_leaves", lambda v: v >= 2),
            ("pooled_min_data_in_leaf", lambda v: v >= 1),
            ("pooled_threads", lambda v: v >= 0),
            ("pooled_teacher_threads", lambda v: v >= 0),
            ("fine_tune_epochs", lambda v: v >= 0),
            ("fine_tune_lr", lambda v: v > 0),
            ("whitelist_tier_k", lambda v: v >= 0),
        )
        if not ok(getattr(args, name))
    ]
    if bad:
        ui.err("args", f"out of range: {', '.join(bad)}")
        raise SystemExit(1)
    if (args.pseudo_negatives == "none") != (args.unmeasured_samples == 0):
        ui.err(
            "args",
            f"--pseudo-negatives {args.pseudo_negatives} with --unmeasured-samples "
            f"{args.unmeasured_samples}: none draws no pseudo-negatives, random "
            f"and targeted draw at least one per GEMM",
        )
        raise SystemExit(1)
    if args.recipe == "pooled_ft":
        unsupported = [
            text
            for text, wrong in (
                (
                    "--validation-frac 0: the pooled student and the fine-tuning "
                    "keep their checkpoints by validation GEMMs",
                    args.validation_frac <= 0,
                ),
                (
                    "--boundary-margin: the pooled student already trains on "
                    "every cell's GEMMs",
                    args.boundary_margin > 0,
                ),
                ("--qat-frac: per_leaf only", args.qat_frac > 0),
            )
            if wrong
        ]
        if unsupported:
            ui.err("recipe", f"--recipe pooled_ft: {'; '.join(unsupported)}")
            raise SystemExit(1)
    tw = None
    if not args.skip_deployed_eval:
        try:
            tw = ev.tilewright_module()
        except RuntimeError as e:
            ui.err("engine", f"{e} (or pass --skip-deployed-eval)")
            raise SystemExit(1)

    v3 = catalog.version == 3
    if args.whitelist == "view-key" and not v3:
        ui.err("catalog", "--whitelist view-key needs --catalog v3")
        raise SystemExit(1)
    if args.pseudo_negatives == "targeted" and not v3:
        ui.err("catalog", "--pseudo-negatives targeted needs --catalog v3")
        raise SystemExit(1)
    v3_only = (
        "small_batched_rule",
        "dot2_rule",
        "non_temporal_rule",
        "non_temporal_family_rule",
        "launch_keys",
    )
    if not v3 and (
        any(getattr(args, a) for a in v3_only)
        or args.min_cu_fraction is not None
        or args.schedules
    ):
        ui.err("catalog", "soft-rule, launch-key and context options need --catalog v3")
        raise SystemExit(1)
    if v3 and args.no_non_temporal_rule:
        ui.err("catalog", "v3 serves without the non-temporal rule unless "
               "--non-temporal-rule is given")  # fmt: skip
        raise SystemExit(1)
    if args.non_temporal_family_rule and not args.non_temporal_rule:
        ui.err("catalog", "--non-temporal-family-rule refines the non-temporal "
               "rule; it needs --non-temporal-rule")  # fmt: skip
        raise SystemExit(1)
    tier_errors = whitelist_tier_errors(args, v3, tw is not None)
    if tier_errors:
        ui.err("tier", f"--whitelist-tier-k {args.whitelist_tier_k}: "
               f"{'; '.join(tier_errors)}")  # fmt: skip
        raise SystemExit(1)
    serving = serving_options(args)
    rules = feasibility.soft_rules(serving, catalog.version)
    clip = v3 and not args.no_clip_features
    if tw is not None:
        probe_serving = dict(serving)
        if v3 and bank_pool.bank_source(cfg).enabled:
            probe_serving["bank"] = True
        if args.whitelist_tier_k > 0:
            probe_serving["whitelist_tier"] = True
        probe = probe_model(
            catalog, probe_serving, clip, args.arch, constants, args.weight_dtype
        )
        errors = engine_round_trip_errors(probe, tw)
        if errors:
            ui.err(
                "engine",
                f"the tilewright module {getattr(tw, '__file__', tw)} cannot "
                f"serve the models this run trains ({probe[:8].decode()}, "
                f"catalog {catalog.name} {catalog.hash()}, {args.weight_dtype} "
                f"weights): {'; '.join(errors)}. Put a build of this "
                f"checkout's engine first on PYTHONPATH (its python/package "
                f"directory) or reinstall the module from it.",
            )
            raise SystemExit(1)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    q_names, i_names, x_names = catalog.names()
    names = (q_names, i_names, x_names)
    launch_summary = bench_launch(_chunk_files(args.enriched_csv_dir)[0])
    if len(launch_summary["heuristics"]) > 1:
        ui.warn(
            "launch",
            f"the training GEMMs were benched with "
            f"{len(launch_summary['heuristics'])} launch heuristics "
            f"({launch_settings.describe_digests(launch_summary)}): the "
            f"launch-rule sources of their builds differ, so their labels may "
            f"measure different launches",
        )

    if not args.quiet:
        ui.banner("stage05  per-cell distillation trainer", ui.C.HEAD)
        ui.info("cfg", f"enriched dirs   : {args.enriched_csv_dir}")
        ui.info("cfg", f"output_dir      : {args.output_dir}")
        ui.info("cfg", f"arch / device   : {args.arch} / {args.device}")
        ui.info("cfg", f"hardware        : {asdict(hardware)}")
        ui.info("cfg", f"weight dtype    : {args.weight_dtype}")
        ui.info("cfg", f"epochs / seed   : {args.epochs} / {args.seed}")
        ui.info(
            "cfg",
            f"schedule        : {args.batch_gemms} GEMMs/step  lr={args.lr} "
            f"{args.lr_schedule}  {args.eval_points} checkpoints  "
            f"patience={args.patience}  qat_frac={args.qat_frac}  "
            f"temperature={args.student_temperature}",
        )
        ui.info("cfg", f"validation frac : {args.validation_frac}")
        ui.info(
            "cfg",
            f"teacher         : {'off' if args.no_teacher else 'LightGBM'}  "
            f"lam/T/alpha={args.lam}/{args.T}/{args.alpha}  "
            f"tie_tolerance={args.tie_tolerance}",
        )
        ui.info("cfg", f"repeats         : {args.repeat_aggregation}")
        ui.info("cfg", f"bench launch    : {launch_settings.describe(launch_summary)}")
        ui.info(
            "feat",
            f"catalog {catalog.name}  feature dims Q={len(q_names)} "
            f"I={len(i_names)} X={len(x_names)}  hash={catalog.hash()}",
        )
        ui.info("serve", f"serving options : {serving}  whitelist={args.whitelist}")

    prior_models, prior_trained = _load_prior_models(
        args.prior_round_dir or [], catalog
    )
    split_tree = sc.load_cumulative_split_tree(args.prior_round_dir or [])
    if args.current_round_dir is not None:
        split_tree.update(sc.load_cumulative_split_tree([args.current_round_dir]))
    if split_tree and not args.quiet:
        ui.ok("splits", f"split tree: {len(split_tree)} split cells")
        for line in sc.format_split_tree(split_tree).splitlines():
            print(f"        {line}", flush=True)

    if (args.library_dir is None) != (args.library_stem is None):
        ui.err("pool", "--library-dir and --library-stem go together")
        raise SystemExit(1)
    identity = (
        ev.library_identity(
            args.library_dir, args.library_stem, cfg, args.hardware_device
        )
        if args.library_stem
        else None
    )
    try:
        blocklist = cx.config_blocklist(cfg)
    except ValueError as e:
        ui.err("blocklist", str(e))
        raise SystemExit(1)
    lib_pool = None
    if args.library_stem:
        lib_pool = bank_pool.config_pool(
            cfg, args.library_dir, args.library_stem, args.hardware_device
        )
        conflicts = ki.tdm_conflicts(
            lib_pool.kernels, rules, [a.name for a in catalog.spec]
        )
        for key, why in conflicts.items():
            ui.err("identity", f"{TDM_CONFLICT_OPTIONS[key]}: {why}")
        if conflicts:
            raise SystemExit(1)
    cells, load_stats = load_enriched_chunks(
        args.enriched_csv_dir,
        args.quiet,
        split_tree,
        identity,
        aggregation=args.repeat_aggregation,
        blocklist=blocklist,
    )
    if load_stats.get("rows_blocked") and not args.quiet:
        ui.info(
            "blocklist",
            f"left out {load_stats['rows_blocked']:,} row(s) of kernels that "
            f"compute wrong results (lib/correctness.py)",
        )
    label_blocklist = labels_blocklist(
        blocklist, _chunk_files(args.enriched_csv_dir)[0]
    )
    if not v3:
        cells, n_dropped = drop_non_exclusive(cells, hardware.n_cu)
        load_stats["gemms_dropped_non_exclusive"] = n_dropped
        if n_dropped:
            ui.warn(
                "context",
                f"dropped {n_dropped} GEMM(s) measured under a CU budget or "
                f"non-default schedule: v2 models serve only exclusive use",
            )
    all_gemms = [g for gs in cells.values() for g in gs]
    try:
        load_stats["gemms_listed"] = attach_plans(
            all_gemms, args.plan or [], args.context_plan or [], identity
        )
    except (OSError, ValueError, KeyError) as e:
        ui.err("plan", f"cannot read the plans: {e}")
        raise SystemExit(1)
    if (args.plan or args.context_plan) and not args.quiet:
        ui.ok(
            "plan",
            f"{load_stats['gemms_listed']:,} of {len(all_gemms):,} GEMMs were "
            f"benched from a kernel list; picks off it are misses",
        )
    try:
        off_list, off_list_report = resolve_off_list_charge(args)
    except (OSError, ValueError) as e:
        ui.err("off-list", f"cannot read --off-list-rebench: {e}")
        raise SystemExit(1)
    if off_list_report and not args.quiet:
        ui.ok("off-list", describe_off_list_charge(off_list_report))
    try:
        contexts = (
            declared_contexts(all_gemms, hardware.n_cu, args.min_cu_fraction,
                              args.schedules)  # fmt: skip
            if v3
            else None
        )
    except ValueError as e:
        ui.err("context", str(e))
        raise SystemExit(1)
    bundle_meta: Dict[str, Any] = (
        {
            "catalog": "v3",
            "attribute_spec": [a.to_dict() for a in catalog.spec],
            "contexts": contexts,
            "clip": clip,
        }
        if v3
        else {}
    )
    if v3 and not args.quiet:
        ui.info("context", f"declared contexts: {contexts}")
    if v3:
        trained_keys = trained_view_keys(all_gemms, catalog) | set(prior_trained)
        bundle_meta["trained_view_keys"] = sorted(trained_keys)
        if not args.quiet:
            ui.info(
                "gate",
                f"{len(trained_keys):,} view keys with training rows; trained-key "
                f"gate {'on' if serving['trained_key_gate'] else 'off'}",
            )
    library = None
    if lib_pool is not None:
        library = ev.pool_from_kernels(lib_pool.kernels)
        if v3:
            serving["bank"] = bool(lib_pool.bank.kernels)
    pools = ev.pools_for(all_gemms, library)
    ev.order_candidates_by_pool(all_gemms, pools)
    if args.served_candidates:
        if not v3:
            ui.err("catalog", "--served-candidates needs --catalog v3")
            raise SystemExit(1)
        load_stats["served_candidates"] = served_candidates(
            all_gemms, pools, served_key_of(catalog, args.launch_keys)
        )
        if not args.quiet:
            st = load_stats["served_candidates"]
            ui.ok(
                "served",
                f"kept {st.get('candidates_kept', 0):,} of "
                f"{st.get('candidates_in', 0):,} candidates, one per key as "
                f"the runtime serves it; {st.get('keys_unmeasured', 0):,} keys "
                f"unmeasured as served",
            )
    nt_flags = {k: feasibility.non_temporal_availability(v) for k, v in pools.items()}
    nt_families = (
        {k: feasibility.non_temporal_families(v) for k, v in pools.items()}
        if rules.get("non_temporal_family_rule")
        else None
    )
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
        elif len(cells[label]) < args.min_cell_gemms and not (
            v3 and label in base_set and len(cells[label]) >= args.min_base_cell_gemms
        ):
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
            served = (
                "the model scores them with its nearest trained base cell"
                if v3
                else "the runtime ranks them with Origami"
            )
            ui.warn(
                "cells",
                f"no data for {len(cells_unpopulated)} base cells; {served}: "
                f"{', '.join(cells_unpopulated)}",
            )

    selector = {
        "engine": tw is not None,
        "arch": args.arch,
        "weight_dtype": args.weight_dtype,
        "pick_measurement": args.pick_measurement,
        "off_list_sel_eff": off_list,
        "pools": pools,
        "serving": serving,
        "bundle_meta": bundle_meta,
        "whitelist": {
            "mode": args.whitelist,
            "k": int(args.smart_k),
            "k_min": int(args.smart_k_min) if args.smart_k_max is not None else None,
            "k_max": int(args.smart_k_max) if args.smart_k_max is not None else None,
            "saturation_eps": float(args.smart_k_saturation_eps),
            "oracle_seed": int(args.oracle_seed or 0),
            "nt_flags": nt_flags,
            "nt_families": nt_families,
            "rules": rules,
        },
    }
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
        tie_tolerance=args.tie_tolerance,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        lgb_lr=args.lgb_lr,
        device=args.device,
        eval_sample_frac=args.eval_sample_frac,
        use_teacher=not args.no_teacher,
        validation_frac=args.validation_frac,
        split_seed=args.seed,
        catalog=catalog,
        clip=clip,
        batch_gemms=args.batch_gemms,
        lr_schedule=args.lr_schedule,
        warmup_frac=args.warmup_frac,
        lr_min_frac=args.lr_min_frac,
        eval_points=args.eval_points,
        patience=args.patience,
        learn_temperature=args.student_temperature == "learned",
        weight_dtype=args.weight_dtype,
        qat_frac=args.qat_frac,
        qat_lr_frac=args.qat_lr_frac,
        selector=selector,
        margin_weight=args.boundary_weight,
        unmeasured_samples=args.unmeasured_samples,
        unmeasured_margin=args.unmeasured_margin,
        unmeasured_weight=args.unmeasured_weight,
        pseudo_negatives=args.pseudo_negatives,
        stage1_recall_weight=args.stage1_recall_weight,
    )
    margins = boundaries.margin_pools(
        cells,
        cells_to_train,
        args.boundary_margin,
        lambda gs: split_validation(gs, args.validation_frac, args.seed)[0],
    )
    if args.boundary_margin > 0 and not args.quiet:
        ui.info(
            "margin",
            f"{sum(map(len, margins.values()))} margin GEMMs within "
            f"{args.boundary_margin:g} octaves, weight {args.boundary_weight:g}",
        )

    pooled_ft = args.recipe == "pooled_ft"
    pooled: Optional[Dict[str, Any]] = None
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
        tpw = worker_threads(n_workers, int(args.train_threads_per_worker or 0))
        order = sorted(cells_to_train, key=lambda c: len(cells[c]), reverse=True)
        if n_workers > 1 and not args.quiet:
            ui.info(
                "parallel",
                f"training {len(order)} cells on {n_workers} workers x "
                f"{tpw} threads",
            )
        if pooled_ft and cells_to_train:
            try:
                pooled = run_pooled_ft(
                    cells,
                    cells_to_train,
                    order,
                    args,
                    names=names,
                    hardware=hardware,
                    constants=constants,
                    catalog=catalog,
                    selector=selector,
                    clip=clip,
                    n_workers=n_workers,
                    threads_per_worker=tpw,
                    log_writer=log_writer,
                )
            except ValueError as e:
                ui.err("pooled", str(e))
                raise SystemExit(1)
            train_results = pooled["results"]
            logf.flush()
        elif n_workers > 1:
            jobs = {
                label: dict(
                    cell=label,
                    gemms=cells[label],
                    margin_gemms=margins[label],
                    seed=cell_seed(args.seed, label),
                    quiet=True,
                    training_log_writer=None,
                    **train_kwargs,
                )
                for label in order
            }
            train_results = run_in_workers(
                train_cell,
                jobs,
                order,
                n_workers,
                tpw,
                quiet=args.quiet,
                done="trained",
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
                if r is None and pooled is not None:
                    r = fine_tune_cell(**pooled["jobs"][label])
                if r is None:
                    r = train_cell(
                        cell=label,
                        gemms=gemms,
                        margin_gemms=margins[label],
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
                if pooled is not None:
                    r.update(
                        pooled_whitelist(
                            label,
                            r,
                            pooled["parts"][label],
                            train_g,
                            val_g,
                            selector,
                            names=names,
                            hardware=hardware,
                            constants=constants,
                            catalog=catalog,
                        )
                    )
                if not args.quiet:
                    m = r["smart_k_metrics"]
                    extra = (
                        f"student_aware_sel_eff={m['student_aware_sel_eff']:.4f}  "
                        f"oracle_under_student={m['oracle_sel_eff_under_student']:.4f}"
                        if m.get("mode") == "student_aware"
                        else f"mode={m.get('mode')}"
                    )
                    n_keys = len(r["smart_k_signatures"]) + len(r["view_key_hashes"])
                    ui.info("smart_k", f"{label}  K={n_keys}  {extra}")
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
                    serving=serving,
                    bundle_meta=bundle_meta,
                    pick_measurement=args.pick_measurement,
                    off_list_sel_eff=off_list,
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
                        and (r["smart_k_signatures"] or r["view_key_hashes"])
                    ):
                        n_dropped = len(r["smart_k_signatures"])
                        n_dropped += len(r["view_key_hashes"])
                        r["smart_k_signatures"] = []
                        r["view_key_hashes"] = []
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
                        f"{label}  best_step={r['best_step']}/{r['steps']} "
                        f"({r['best_phase']})  "
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

    def _g(key: str, field: str) -> float:
        return pooled_mean(all_results, key, field)

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
        "global_deployed_val_sel_eff": _g("deployed_val", "sel_eff"),
        "global_origami_val_sel_eff": _g("origami_val", "sel_eff"),
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
            "n_margin_gemms": r.get("n_margin_gemms", 0),
            "n_unmeasured_pairs": r.get("n_unmeasured_pairs", 0),
            "selection_set": r["selection_set"],
            "n_pairs": r["n_pairs"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "best_step": r["best_step"],
            "best_phase": r["best_phase"],
            "steps": r["steps"],
            "stopped_early": r["stopped_early"],
            "temperature": r["temperature"],
            "selection_metric": r["selection_metric"],
            "selection_deployed_val_sel_eff": r["selection_deployed_val_sel_eff"],
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
            "n_view_key_hashes": len(r.get("view_key_hashes") or []),
        }
        if sc.is_subcell(r["cell"]):
            entry["base_cell"] = sc.base_cell(r["cell"])
            entry["split_depth"] = sc.split_depth(r["cell"])
        if pooled is not None:
            entry["pooled_val_sel_eff"] = r.get("pooled_val_sel_eff")
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

    uncovered = uncovered_base_cells(new_models)
    if uncovered and not args.quiet:
        served = (
            "scored through their nearest trained base cell"
            if v3
            else "ranked by Origami at runtime"
        )
        ui.warn(
            "coverage",
            f"{len(uncovered)} of {len(base_set)} base cells have no model; "
            f"they are {served}",
        )

    cells_json = {
        "schema_version": "pipeline-v3-2026-10-03",
        "feature_version": fs.FEATURES_VERSION,
        "feature_names_hash": catalog.hash(),
        "catalog": catalog.name,
        "recipe": args.recipe,
        "uncovered_base_cells": uncovered,
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

    provenance = prov.training_provenance(
        cfg,
        catalog,
        [os.path.join(fd, f) for fd, f in _chunk_files(args.enriched_csv_dir)[1]],
    )
    bundle: Dict[str, Any] = {
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
        "serving": serving,
        **bundle_meta,
        "provenance": provenance,
        "identity_rule": ki.IDENTITY_RULE,
        "recipe": args.recipe,
        **(
            {}
            if pooled is None
            else {"pooled": {"entry": pooled["entry"], **pooled["record"]}}
        ),
        "bench_launch": launch_summary,
        "blocklist": label_blocklist,
    }
    tier: Optional[wt.Tier] = None
    tier_summary: Optional[Dict[str, Any]] = None
    if args.whitelist_tier_k > 0 and new_models:
        if not args.quiet:
            ui.banner("Whitelist tier", ui.C.BLUE)
        t_tier = time.time()
        try:
            tier = whitelist_tier(
                args,
                bundle,
                cfg,
                catalog=catalog,
                hardware=hardware,
                identity=identity,
                blocklist=blocklist,
                split_tree=split_tree,
            )
        except (ValueError, RuntimeError, OSError) as e:
            ui.err("tier", f"cannot build the whitelist tier: {e}")
            raise SystemExit(1)
        bundle = wt.apply(bundle, tier)
        tier_summary = {**tier.record, "elapsed_s": time.time() - t_tier}
        for entry in cells_json["cells"]:
            entry["n_whitelist_tier_keys"] = len(tier.keys.get(entry["label"], []))
        if not args.quiet:
            bare = sorted(c for c in bundle["models"] if not tier.keys.get(c))
            ui.ok(
                "tier",
                f"{wt.record_text(tier.record)}, chosen on "
                f"{tier.record['n_gemms']:,} training GEMMs "
                f"({ui.fmt_dur(tier_summary['elapsed_s'])})",
            )
            if bare:
                ui.warn(
                    "tier",
                    f"{len(bare)} cells without a training GEMM keep no tier "
                    f"and serve through the cascade: {', '.join(bare[:6])}"
                    + (" ..." if len(bare) > 6 else ""),
                )
    with open(os.path.join(args.output_dir, "cells.json"), "w") as f:
        json.dump(jsonable(cells_json), f, indent=2)
    torch.save(bundle, os.path.join(args.output_dir, "models.pt"))
    tier_path = Path(args.output_dir) / wt.TIER_FILE
    if tier is not None:
        wt.write_doc(tier_path, tier)
    else:
        tier_path.unlink(missing_ok=True)
    write_json(Path(args.output_dir) / MODEL_OPTIONS_FILE, model_options(args, catalog))
    if library is not None:
        write_json(
            Path(args.output_dir) / MEMBER_ORDER_FILE,
            member_order(library, catalog, v3, args.launch_keys, args.arch, identity),
        )
    else:
        (Path(args.output_dir) / MEMBER_ORDER_FILE).unlink(missing_ok=True)

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
        "off_list_charge": off_list_report,
        "args": jsonable(vars(args)),
        "hardware": asdict(hardware),
        "arch_constants": {
            "parallel_mi_cu": constants.parallel_mi_cu,
            "bw": list(constants.bw),
            "mi_default": constants.mi_default,
            "n_mi_entries": len(constants.mi_table),
        },
        "weight_dtype": args.weight_dtype,
        "recipe": args.recipe,
        "pooled": None if pooled is None else pooled["record"],
        "environment": _environment(),
        "feature_version": fs.FEATURES_VERSION,
        "feature_names_hash": catalog.hash(),
        "catalog": catalog.name,
        "serving": bundle["serving"],
        "whitelist_tier": tier_summary,
        "contexts": contexts,
        "uncovered_base_cells": uncovered,
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
        "provenance": provenance,
        "identity_rule": ki.IDENTITY_RULE,
        "bench_launch": launch_summary,
        "blocklist": label_blocklist,
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
            f"(cells.json, models.pt, model_options.json, training_log.csv, "
            f"metrics.json{', member_order.json' if library is not None else ''}"
            f"{', ' + wt.TIER_FILE if tier is not None else ''})",
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
