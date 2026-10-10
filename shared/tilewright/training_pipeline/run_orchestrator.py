# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the tilewright training pipeline for one config.

Modes:

    initial   round_0 only: seed shapes for every grid cell, bench, train.
    active    one more round on top of the complete rounds already in
              --run-dir (round_k, k = number of complete rounds). An
              incomplete round_k is resumed.
    full      round_0 plus active rounds up to `full.total_rounds`. Complete
              rounds are skipped, so re-running the same command resumes.
              With audit.dir the run builds the audit set when it is not
              complete (audit.build, `AuditBuild`) and measures its bundles
              under audit protocol v2 and the sustained composite
              (audit.v2, audit.sustained); full.final_retrain retrains every
              cell on every round's data at the end and ships it when the
              audit set finds it not worse under the official protocol
              (audit.official; `FinalRetrain`).
    validate  stages 7 and 8 against an existing --run-dir; no training.
    enhance   --enhance-rounds active rounds (default 1) on top of the
              complete rounds, restricted to chosen cells: --cells names
              them, --weakest N adds the N cells of lowest training-side
              selection efficiency below the target (--rank-by gain: of
              largest GEMM share x gap), --below T every cell below T.
              stage01 draws shapes for those leaves only (also leaves
              without a model), stages 3-4 bench and enrich them, stage04b
              measures the current model on them, stage05 retrains them
              (--only-cells; split children replace a split leaf), and the
              last enhance round deploys like a final active round. The
              evidence is training-side only (stage04b held-out rows,
              stage05 validation and in-sample scores; lib/cell_evidence.py),
              against the fastest measured kernel and an absolute target;
              validation datasets never select cells. round_k/enhance.json
              records the selection; an incomplete enhance round resumes
              with it. --dry-run prints the selection and the stage
              commands and writes nothing.
    audit     build, resume or extend the audit set in --run-dir (its own
              directory, lib/audit.py): shapes from stage01 with
              audit.seed, none of the --source-run-dir rounds' shapes,
              benched over every pool and bank kernel in every context in
              resumable parts; a later pool change benches only the
              changed kernels. A training config points at it with
              audit.dir: rounds exclude its shapes, stage05 refuses it,
              stage04b and stage06 report the models on it (audit.json).

Deploy:

    An active round measures the bundle it trained on its gate GEMMs (a
    deploy.gate_frac slice of its new held-out GEMMs that it did not train
    on) in stage06, and deploys it when it reaches deploy.target_sel_eff.
    The final round (deploy.on_final_round) also deploys a bundle below the
    target unless it is worse than the bundle deployed before on the same
    GEMMs. --deploy deploys every round. With audit.dir, once the run ships
    a bundle, stage06 also scores that bundle on the audit set and keeps
    the round's out when a cell or a tail pick regresses against it (the
    gate section, `lib.audit.cell_gate`; --deploy only records it), as the
    final retrain's gate does.

Resume:

    --from-round R   full: re-run round R and every later round (rounds
                     before R must be complete). validate: take round R's
                     trained models instead of the deployed bundle's (the
                     latest round's when nothing was deployed).
    --from-stage N   re-run the target round from logical stage N
                     (1, 2, 3, 3.2, 3.5, 3.7, 4, 4.5, 5, 6; validate mode: 7
                     or 8), or from the stage of that name (stage02,
                     stage03c, ...). Earlier stages must have completed.
                     Without it a resumed round restarts at its first stage
                     that did not complete. 1.5 and stage01b, the planner's
                     names before it became stage 2, still name it, with a
                     deprecation note.

A stage that is re-run gets a fresh output directory: the previous attempt's
directory and log move to `<round>/stale/`. A completed stage writes
`.stage_complete.json` into its directory, a completed round writes
`.round_complete.json`. A run or audit directory written before the planner
became stage 2 keeps it in `stage01b/` (lib/run_layout.py): its readers and
resumes, and the rounds a resume adds, go on with that name.

Each invocation writes its expanded config to `<run-dir>/configs/` and passes
that snapshot to every stage, so later edits of the config file cannot leak
into a running or resumed pipeline. An invocation holds an exclusive lock on
`<run-dir>/.lock`; a second one on the same run directory exits with 2.
A stage that benches runs under the lease of its GPUs (lib/gpu_lease.py):
a GPU another process tree leases is left to it, naming the holder, and
only when it leases every GPU of the stage does the attempt wait
`runtime.gpu_lease_wait_s` and then fail with 2 before it starts. Devices
whose GPU `runtime.exclude_gpus` names leave the run when it starts, and no
stage on GPUs starts while the run holds its GPUs for being unlike they were
when it started (lib/gpu_health.py, `bench.health`): a hold of every GPU
holds every such stage, the hold of single GPUs a stage on one of them, or,
for a stage whose benches run through stage 3 (which keeps a held GPU out
of its blocks), one on nothing but held GPUs.

A stage attempt that fails for a transient reason (a busy GPU lease, a bench
stage's exit 2, a death by SIGHUP, SIGINT, SIGKILL or SIGTERM this
orchestrator did not send) is retried `runtime.stage_retries` times after a
growing wait (`runtime.stage_retry_backoff_s`), each attempt in a fresh
directory.

Pin: `<run-dir>/pin.json` records the pipeline code, the engine module and
the pinned config keys the run is produced with (lib/pinning.py), and every
stage marker the pin its stage ran with. An invocation that differs exits
with 2 before any stage runs, naming the change, unless
--allow-code-change accepts it; a change of the code or the engine module
while an invocation runs stops it before its next stage. An audit directory
keeps only the config keys its benches read, and `--mode validate` keeps
nothing: it names a difference and goes on (`pin_parts`).

Stages run in their own process group. SIGINT, SIGTERM or SIGHUP stops the
running stage (SIGINT to its group, SIGKILL after 30 s) and exits with
128 + the signal number; the same command resumes the round. A stage gets
SIGTERM when the orchestrator dies, also by SIGKILL, and carries an attempt
token that every process it starts inherits (lib/attempts.py);
`<run-dir>/.running.json` names the running attempt, and the next invocation
stops whatever an orchestrator that died left running before it moves any
output aside.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import itertools
import json
import math
import mmap
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lib import arch as archlib  # noqa: E402
from lib import attempts  # noqa: E402
from lib import audit as auditlib  # noqa: E402
from lib import audit_v2 as v2lib  # noqa: E402
from lib import bank_pool  # noqa: E402
from lib import bench_env as benv  # noqa: E402
from lib import cell_evidence as ce  # noqa: E402
from lib import contexts as ctxlib  # noqa: E402
from lib import correctness as cxlib  # noqa: E402
from lib import correctness_gate as cgate  # noqa: E402
from lib import device_values as dvlib  # noqa: E402
from lib import fingerprint  # noqa: E402
from lib import gpu_health  # noqa: E402
from lib import gpu_lease  # noqa: E402
from lib import hazards as hazlib  # noqa: E402
from lib import hipblaslt_launch as hllib  # noqa: E402
from lib import host_load  # noqa: E402
from lib import kernel_identity as ki  # noqa: E402
from lib import kfd  # noqa: E402
from lib import library_pool as lp  # noqa: E402
from lib import pinning  # noqa: E402
from lib import procs  # noqa: E402
from lib import rebench as rebenchlib  # noqa: E402
from lib import repeats  # noqa: E402
from lib import run_bundles as rb  # noqa: E402
from lib import run_layout  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import sustained as suslib  # noqa: E402
from lib import ui  # noqa: E402
from lib import weak_spots  # noqa: E402
from lib.arch import build_alias_of  # noqa: E402
from lib.bench_env import TILEWRIGHT_BANK_VAR  # noqa: E402
from lib.bench_yaml import (  # noqa: E402
    ADAPTIVE_DEFAULTS,
    ADAPTIVE_LONG_DEFAULTS,
    CANARY_DEFAULTS,
    DEFAULT_ORDER_SEED,
    KERNEL_ORDERS,
    LONG_KNOB_KEYS,
    WARMUP_LINE_DEFAULTS,
    roofline_us,
)
from lib.dat import (  # noqa: E402
    is_tensile_contraction_logic,
    load_library_kernels,
    logic_stem,
    mx_block_size_for_scale_mode,
)
from lib.grid import all_cell_labels  # noqa: E402
from lib.planner import MAX_BENCH_LIST, RECALL_MIN_CEILING, PlanParams  # noqa: E402
from lib.stage05_split import stage05_module  # noqa: E402

RUNS = THIS_DIR / "runs"
STAGE_MARKER = ".stage_complete.json"
ROUND_MARKER = ".round_complete.json"
RUN_STATUS = "run_status.json"
# Stage-marker field of a validation stage: what its pass vouches for
# (`Pipeline.validation_key`).
VALIDATED = "validated"
ENHANCE_RECORD = "enhance.json"
LOCK_NAME = ".lock"
FEATURE_DIMS = (55, 12, 37)
# stage05: outputs written, but a cell failed or a requested cell was not
# trained.
STAGE05_CELLS_NOT_TRAINED = 3
CARRIED_CELLS = "cells_carried.json"
# The arguments and pin of a stage05 that exited 3 with its outputs written,
# so that a later invocation with train.on_cell_failure: carry goes on with
# that bundle instead of training it again.
UNTRAINED_BUNDLE = ".untrained_bundle.json"
FINAL_RETRAIN_DIR = rb.FINAL_RETRAIN_DIR
AUDIT_RESULTS_DIR = rb.AUDIT_RESULTS_DIR
WEAK_SPOTS_DIR = "weak_spots"
AUDIT_EVAL_SCRIPT = THIS_DIR / "scripts" / "audit_eval.py"
AUDIT_V2_SCRIPT = THIS_DIR / "scripts" / "audit_v2.py"


class ConfigError(Exception):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class Interrupted(KeyboardInterrupt):
    """SIGTERM or SIGHUP, handled like SIGINT."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)


# ── config schema ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spec:
    kind: str
    required: bool = False
    default: Any = None
    nullable: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    pattern: Optional[str] = None
    item: Optional["Spec"] = None
    min_len: int = 0
    children: Optional[Mapping[str, "Spec"]] = None


def _int(**kw: Any) -> Spec:
    return Spec("int", **kw)


def _num(**kw: Any) -> Spec:
    return Spec("num", **kw)


def _bool(**kw: Any) -> Spec:
    return Spec("bool", **kw)


def _str(**kw: Any) -> Spec:
    return Spec("str", **kw)


def _list(item: Spec, **kw: Any) -> Spec:
    return Spec("list", item=item, **kw)


def _section(children: Mapping[str, Spec], **kw: Any) -> Spec:
    return Spec("section", children=children, **kw)


_DTYPE = r"^[a-z0-9_]+$"
_PATH_NAME = r"^[A-Za-z0-9_.+:-]+$"
_OPTION_NAME = r"^[A-Za-z][A-Za-z0-9_]*$"
AUDIT_BUILDS = ("before", "parallel", "off")
AUDIT_V2_MODES = ("rounds", "final", "off")
FINAL_GATES = ("not-worse", "none")
_PLAN_DEFAULTS = PlanParams()
_QUIET = host_load.DEFAULTS
_HEALTH = gpu_health.DEFAULTS
_HEALTH_CANARY = gpu_health.CANARY_DEFAULTS
_ADAPTIVE = ADAPTIVE_DEFAULTS
_ADAPTIVE_LONG = ADAPTIVE_LONG_DEFAULTS
_CANARY = CANARY_DEFAULTS
_WARMUP_LINE = WARMUP_LINE_DEFAULTS
_REREAD = rebenchlib.REREAD_DEFAULTS
_CONTEXT = _section(
    {
        "cu_budget": _int(default=0, minimum=0),
        "cu_fraction": _num(
            minimum=0, maximum=1, exclusive_minimum=True, exclusive_maximum=True
        ),
        "schedule": _str(default="default", choices=ctxlib.SCHEDULES),
        "cotenant": _bool(default=False),
        "fraction": _num(default=1.0, minimum=0, maximum=1, exclusive_minimum=True),
        "name": _str(pattern=r"^[A-Za-z0-9_.-]+$"),
    }
)
EXCLUDE_KEYS = (
    "min_m",
    "max_m",
    "min_n",
    "max_n",
    "min_k",
    "max_k",
    "min_batch",
    "max_batch",
    "min_mn",
    "max_mn",
)

SCHEMA = _section(
    {
        "config_id": _str(required=True, pattern=r"^[A-Za-z0-9_.-]+$"),
        "arch": _str(required=True, pattern=r"^gfx[0-9a-z]+(-[a-z]+)?(:[a-z0-9+-]+)*$"),
        "hipblaslt": _section(
            {
                "a_type": _str(required=True, pattern=_DTYPE),
                "b_type": _str(required=True, pattern=_DTYPE),
                "c_type": _str(required=True, pattern=_DTYPE),
                "d_type": _str(required=True, pattern=_DTYPE),
                "scale_type": _str(required=True, pattern=_DTYPE),
                "compute_type": _str(required=True, pattern=_DTYPE),
                "scaleA": _int(minimum=0),
                "scaleB": _int(minimum=0),
                "transA": _str(required=True, choices=("T", "N")),
                "transB": _str(required=True, choices=("T", "N")),
                "library_stem": _str(
                    required=True, pattern=r"^TensileLibrary_[A-Za-z0-9_.-]+$"
                ),
            },
            required=True,
        ),
        "library": _section(
            {
                "kind": Spec("names", choices=lp.KINDS),
                "bok_rows": Spec("names", choices=tuple(lp.ROW_PREDICATES)),
            }
        ),
        "hardware": _section(
            {
                "n_cu": _int(minimum=1),
                "lds_bytes": _int(minimum=1),
                "l2_bytes": _int(minimum=1),
                "pci_chip_id": _int(minimum=0),
                "num_xcd": _int(minimum=1),
            }
        ),
        "runtime": _section(
            {
                "devices": _list(_int(minimum=0), required=True, min_len=1),
                "min_free_gb": _num(nullable=True, minimum=0),
                "rocm_libraries_root": _str(required=True),
                "env": Spec("env"),
                "build_dir": _str(),
                "gpu_lease_wait_s": _num(default=3600.0, minimum=0),
                "exclude_gpus": _list(Spec("gpu"), default=[]),
                "stage_retries": _int(default=1, minimum=0),
                "stage_retry_backoff_s": _num(default=300.0, minimum=0),
            },
            required=True,
        ),
        "seed": _section(
            {
                "target_per_cell": _int(default=1600, minimum=1),
                "shapes_per_cell": _int(default=150, minimum=1),
                "seed": _int(default=42),
                "n_strata": _int(default=1, minimum=1),
                "exclude": _list(
                    _section({key: _int(minimum=0) for key in EXCLUDE_KEYS})
                ),
            }
        ),
        "bench": _section(
            {
                "blocks_per_gpu": _int(default=20, minimum=1),
                "stall_timeout_s": _num(default=600.0, minimum=0),
                "startup_grace_s": _num(default=900.0, minimum=0),
                "quiet_host": _section(
                    {
                        "enabled": _bool(default=_QUIET["enabled"]),
                        "every_s": _num(
                            default=_QUIET["every_s"], minimum=1.0, maximum=300.0
                        ),
                        "quiet_busy": _num(
                            default=_QUIET["quiet_busy"], minimum=0, maximum=1
                        ),
                        "quiet_core_busy": _num(
                            default=_QUIET["quiet_core_busy"], minimum=0, maximum=1
                        ),
                        "hot_busy": _num(
                            default=_QUIET["hot_busy"], minimum=0, maximum=1
                        ),
                        "abort_samples": _int(
                            default=_QUIET["abort_samples"], minimum=1
                        ),
                        "max_wait_s": _num(default=_QUIET["max_wait_s"], minimum=0),
                    }
                ),
                "health": _section(
                    {
                        "enabled": _bool(default=_HEALTH["enabled"]),
                        "poll_s": _num(
                            default=_HEALTH["poll_s"], minimum=1.0, maximum=3600.0
                        ),
                        "max_wait_s": _num(default=_HEALTH["max_wait_s"], minimum=0),
                        "ttm": _bool(default=_HEALTH["ttm"]),
                        "canary": _section(
                            {
                                "enabled": _bool(default=_HEALTH_CANARY["enabled"]),
                                "lines": _int(
                                    default=_HEALTH_CANARY["lines"],
                                    minimum=1,
                                    maximum=16,
                                ),
                                "timings": _int(
                                    default=_HEALTH_CANARY["timings"],
                                    minimum=1,
                                    maximum=32,
                                ),
                                "tolerance": _num(
                                    default=_HEALTH_CANARY["tolerance"],
                                    minimum=0,
                                    maximum=1,
                                    exclusive_minimum=True,
                                ),
                                "retry_s": _num(
                                    default=_HEALTH_CANARY["retry_s"], minimum=0
                                ),
                            }
                        ),
                    }
                ),
                "warmup_pass": _bool(default=True),
                "warmup_line": _section(
                    {
                        "enabled": _bool(default=_WARMUP_LINE["enabled"]),
                        "kernels": _int(
                            default=_WARMUP_LINE["kernels"], minimum=1, maximum=64
                        ),
                        "size": _int(default=_WARMUP_LINE["size"], minimum=1),
                    }
                ),
                "skip_slow_solution_ratio": _num(default=0.0, minimum=0, maximum=1),
                "kernel_order": _str(default="shuffled", choices=KERNEL_ORDERS),
                "order_seed": _int(default=DEFAULT_ORDER_SEED),
                "sacrificial_kernel": _bool(default=True),
                "pool_only_library": _bool(default=False),
                "rotating_mb": _num(minimum=0),
                "rotating_target_blocks": _int(minimum=1),
                "device_memory_gib": _num(minimum=0, exclusive_minimum=True),
                "sustained": _section(
                    {
                        "enabled": _bool(default=False),
                        "min_us": _num(
                            default=suslib.DEFAULTS["min_us"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        "compute_bound_tflops": _num(
                            default=suslib.DEFAULTS["compute_bound_tflops"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        "compute_bound_min_us": _num(
                            default=suslib.DEFAULTS["compute_bound_min_us"], minimum=0
                        ),
                        "top_k": _int(default=suslib.DEFAULTS["top_k"], minimum=1),
                        "tolerance": _num(
                            default=suslib.DEFAULTS["tolerance"], minimum=0, maximum=1
                        ),
                        "max_kernels": _int(
                            default=suslib.DEFAULTS["max_kernels"], minimum=2
                        ),
                        "runs": _int(default=suslib.DEFAULTS["runs"], minimum=1),
                        "warmup_time": _num(
                            default=suslib.DEFAULTS["warmup_time"], minimum=0
                        ),
                        "measure_time": _num(
                            default=suslib.DEFAULTS["measure_time"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        "max_measure_time": _num(
                            default=suslib.DEFAULTS["max_measure_time"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        "gemms_per_block": _int(
                            default=suslib.DEFAULTS["gemms_per_block"], minimum=1
                        ),
                    }
                ),
                "canary": _section(
                    {
                        "enabled": _bool(default=_CANARY["enabled"]),
                        "tolerance": _num(
                            default=_CANARY["tolerance"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        "timings": _int(default=_CANARY["timings"], minimum=1),
                        "backoff_s": _num(default=_CANARY["backoff_s"], minimum=0),
                    }
                ),
                "rebench": _section(
                    {
                        "enabled": _bool(default=False),
                        "tolerance": _num(
                            default=_REREAD["tolerance"],
                            minimum=0,
                            maximum=1,
                            exclusive_minimum=True,
                        ),
                        "tolerance_us": _num(
                            default=_REREAD["tolerance_us"], minimum=0
                        ),
                        "max_kernels": _int(default=_REREAD["max_kernels"], minimum=2),
                        "runs": _int(default=_REREAD["runs"], minimum=0),
                        "fast_below_us": _num(
                            default=_REREAD["fast_below_us"], minimum=0
                        ),
                        "fast_runs": _int(default=_REREAD["fast_runs"], minimum=0),
                        "slow_from_us": _num(
                            default=_REREAD["slow_from_us"], minimum=0
                        ),
                        "slow_runs": _int(default=_REREAD["slow_runs"], minimum=0),
                        "slow_min_iters": _int(
                            default=_REREAD["slow_min_iters"], minimum=0
                        ),
                        "inline_copies": _int(
                            default=_REREAD["inline_copies"], minimum=1
                        ),
                        "with_original": _bool(default=_REREAD["with_original"]),
                        "align_to_original": _bool(
                            default=_REREAD["align_to_original"]
                        ),
                        "sacrificial_kernel": _bool(
                            default=_REREAD["sacrificial_kernel"]
                        ),
                        "gemm_order": _str(
                            default=_REREAD["gemm_order"],
                            choices=rebenchlib.GEMM_ORDERS,
                        ),
                    }
                ),
                "adaptive_long": _section(
                    {
                        "enabled": _bool(default=_ADAPTIVE_LONG["enabled"]),
                        "min_roofline_us": _num(
                            default=_ADAPTIVE_LONG["min_roofline_us"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                        **{
                            k: _num(default=_ADAPTIVE_LONG[k], minimum=0)
                            for k in LONG_KNOB_KEYS
                        },
                        "peak_tflops": _num(
                            nullable=True, minimum=0, exclusive_minimum=True
                        ),
                        "bandwidth_gbs": _num(
                            nullable=True, minimum=0, exclusive_minimum=True
                        ),
                    }
                ),
                "adaptive": _section(
                    {
                        "warmup_time": _num(
                            default=_ADAPTIVE["warmup_time"], minimum=0
                        ),
                        "sample_time": _num(
                            default=_ADAPTIVE["sample_time"], minimum=0
                        ),
                        "measure_time": _num(
                            default=_ADAPTIVE["measure_time"], minimum=0
                        ),
                        "max_measure_time": _num(
                            default=_ADAPTIVE["max_measure_time"], minimum=0
                        ),
                        "min_iters": _int(default=_ADAPTIVE["min_iters"], minimum=0),
                        "max_iters": _int(default=_ADAPTIVE["max_iters"], minimum=0),
                        "noise_threshold": _num(
                            default=_ADAPTIVE["noise_threshold"], minimum=0
                        ),
                        "stability_threshold": _num(
                            default=_ADAPTIVE["stability_threshold"], minimum=0
                        ),
                        "stability_window": _int(
                            default=_ADAPTIVE["stability_window"], minimum=0
                        ),
                        "stability_interval": _int(
                            default=_ADAPTIVE["stability_interval"], minimum=0
                        ),
                    }
                ),
            }
        ),
        "planner": _section(
            {
                "enabled": _bool(default=False),
                "mode": _str(
                    default="exhaustive", choices=("exhaustive", "planned", "auto")
                ),
                "exhaustive_max_kernels": _int(default=512, minimum=0),
                "seed_per_cell": _int(default=8, minimum=1),
                "ranker": _str(default="neighbors", choices=("neighbors", "stage05")),
                "neighbors_k": _int(default=4, minimum=1),
                "model_file": _str(nullable=True),
                "bank_library_stems": _list(
                    _str(pattern=r"^TensileLibrary_[A-Za-z0-9_.-]+$"), default=[]
                ),
                "bank": _str(default="none", choices=bank_pool.BANK_MODES),
                "rules": _str(default="config", choices=("config", "model")),
                "top_k": _int(default=_PLAN_DEFAULTS.top_k, minimum=0),
                "top_k_round0": _int(nullable=True, minimum=0),
                "recall_min_ceiling": _num(
                    default=RECALL_MIN_CEILING, minimum=0, maximum=1
                ),
                "families_top_f": _int(
                    default=_PLAN_DEFAULTS.families_top_f, minimum=0
                ),
                "family_variants": _int(
                    default=_PLAN_DEFAULTS.family_variants, minimum=0
                ),
                "explore_random_k": _int(
                    default=_PLAN_DEFAULTS.explore_random_k, minimum=0
                ),
                "min_coverage": _int(default=_PLAN_DEFAULTS.min_coverage, minimum=0),
                "max_list": _int(
                    default=_PLAN_DEFAULTS.max_list, minimum=1, maximum=MAX_BENCH_LIST
                ),
                "seed": _int(default=_PLAN_DEFAULTS.seed),
            }
        ),
        "contexts": _section(
            {
                "list": _list(_CONTEXT, default=[]),
                "holdout": _list(_CONTEXT, default=[]),
                "seed": _int(default=42),
                "cotenant_binary": _str(nullable=True),
                "path_append": _list(_str(), nullable=True),
                "replan": _bool(default=False),
            }
        ),
        "train": _section(
            {
                "workers": _int(default=1, minimum=1),
                "threads_per_worker": _int(default=0, minimum=0),
                "epochs": _int(default=200, minimum=1),
                "device": _str(default="cpu"),
                "eval_sample_frac": _num(
                    default=1.0, minimum=0, maximum=1, exclusive_minimum=True
                ),
                "smart_k": _int(default=10, minimum=0),
                "smart_k_max": _int(nullable=True, minimum=1),
                "smart_k_min": _int(default=4, minimum=1),
                "smart_k_saturation_eps": _num(default=0.005, minimum=0),
                "non_temporal_rule": _bool(nullable=True),
                "non_temporal_family_rule": _bool(default=False),
                "trained_key_gate": _bool(default=True),
                "cascade_k": _int(nullable=True, minimum=0, maximum=65535),
                "catalog": _str(default="v2", choices=("v2", "v3")),
                "attribute_spec": _list(
                    _section(
                        {
                            "name": _str(required=True),
                            "default": _int(),
                            "transform": _str(
                                choices=("identity", "log2p1", "nonzero", "onehot")
                            ),
                            "values": _list(_int()),
                        }
                    ),
                    nullable=True,
                ),
                "whitelist": _str(
                    nullable=True, choices=("signature", "view-key", "none")
                ),
                "whitelist_tier_k": _int(default=0, minimum=0),
                "clip_features": _bool(default=True),
                "small_batched_rule": _bool(default=False),
                "launch_keys": _bool(default=False),
                "served_candidates": _bool(default=False),
                "dot2_rule": _bool(default=False),
                "min_cu_fraction": _num(
                    nullable=True, minimum=0, maximum=1, exclusive_minimum=True
                ),
                "schedules": _list(
                    _str(choices=("default", "dynamic", "auto")), nullable=True
                ),
                "min_base_cell_gemms": _int(default=10, minimum=1),
                "min_cell_gemms": _int(default=50, minimum=1),
                "validation_frac": _num(minimum=0, maximum=1),
                "batch_gemms": _int(minimum=0),
                "lr": _num(minimum=0, exclusive_minimum=True),
                "lr_schedule": _str(choices=("cosine", "step", "constant")),
                "warmup_frac": _num(minimum=0, maximum=1, exclusive_maximum=True),
                "lr_min_frac": _num(minimum=0, maximum=1),
                "eval_points": _int(minimum=1),
                "patience": _int(minimum=0),
                "student_temperature": _str(choices=("learned", "fixed")),
                "qat_frac": _num(minimum=0),
                "qat_lr_frac": _num(minimum=0, exclusive_minimum=True),
                "boundary_margin": _num(minimum=0),
                "boundary_weight": _num(minimum=0, maximum=1, exclusive_minimum=True),
                "off_list_sel_eff": Spec("efficiency_or_auto"),
                "off_list_fallback": _num(minimum=0, maximum=1, exclusive_minimum=True),
                "off_list_min_misses": _int(default=30, minimum=1),
                "pseudo_negatives": _str(choices=("none", "random", "targeted")),
                "unmeasured_samples": _int(minimum=0),
                "unmeasured_margin": _num(minimum=0),
                "unmeasured_weight": _num(minimum=0, maximum=1, exclusive_minimum=True),
                "stage1_recall_weight": _num(minimum=0),
                "recipe": _str(choices=("per_leaf", "pooled_ft")),
                "pooled_epochs": _int(minimum=1),
                "pooled_batch_gemms": _int(minimum=0),
                "pooled_lr": _num(minimum=0, exclusive_minimum=True),
                "pooled_eval_points": _int(minimum=1),
                "pooled_threads": _int(minimum=0),
                "pooled_teacher_threads": _int(minimum=0),
                "fine_tune_epochs": _int(minimum=0),
                "fine_tune_lr": _num(minimum=0, exclusive_minimum=True),
                "repeat_aggregation": _str(
                    default=repeats.DEFAULT_AGGREGATION, choices=repeats.AGGREGATIONS
                ),
                "on_cell_failure": _str(default="stop", choices=("stop", "carry")),
                "carry_max_fraction": _num(default=0.25, minimum=0, maximum=1),
            }
        ),
        "validate": _section(
            {
                "sel_eff_threshold": Spec(
                    "threshold", default=0.95, exclusive_minimum=True
                ),
                "split_floor": Spec("threshold", default=0.85),
                "min_cell_gemms": _int(default=20, minimum=1),
                "split_min_improvement": _num(default=0.02),
                "split_after_attempts": _int(default=2, minimum=0),
                "max_split_depth": _int(default=6, minimum=0),
                "structural_split_min_octaves": _num(default=0.0, minimum=0),
                "structural_split_max_depth": _int(default=-1),
                "pick_measurement": _str(
                    default="warmup", choices=("warmup", "timed", "any")
                ),
                "rebench_misses": _bool(default=True),
                "rebench_runs": _int(default=3, minimum=1),
            }
        ),
        "full": _section(
            {
                "total_rounds": _int(default=3, minimum=1),
                "final_retrain": _section(
                    {
                        "enabled": _bool(default=False),
                        "gate": _str(default="not-worse", choices=FINAL_GATES),
                        "recipe": Spec("mapping", nullable=True),
                    }
                ),
            }
        ),
        "audit": _section(
            {
                "dir": _str(nullable=True),
                "build": _str(default="before", choices=AUDIT_BUILDS),
                "devices": _list(_int(minimum=0), nullable=True, min_len=1),
                "v2": _str(default="rounds", choices=AUDIT_V2_MODES),
                "v2_set_tolerance": _num(
                    default=v2lib.DEFAULTS["set_tolerance"], minimum=0
                ),
                "v2_set_tolerance_us": _num(
                    default=v2lib.DEFAULTS["set_tolerance_us"], minimum=0
                ),
                "v2_set_max": _int(default=v2lib.DEFAULTS["set_max"], minimum=1),
                "sustained": _bool(default=auditlib.DEFAULTS["sustained"]),
                "official": _str(
                    default=auditlib.OFFICIAL_DEFAULT,
                    choices=auditlib.OFFICIAL_CHOICES,
                ),
                "shapes_per_cell": _int(
                    default=auditlib.DEFAULTS["shapes_per_cell"], minimum=1
                ),
                "shapes_per_context": _int(
                    default=auditlib.DEFAULTS["shapes_per_context"], minimum=1
                ),
                "seed": _int(default=auditlib.DEFAULTS["seed"]),
                "target": _num(
                    default=auditlib.DEFAULTS["target"],
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                ),
                "bootstrap": _int(default=auditlib.DEFAULTS["bootstrap"], minimum=0),
                "part_shapes": _int(
                    default=auditlib.DEFAULTS["part_shapes"], minimum=1
                ),
            }
        ),
        "deploy": _section(
            {
                "target_sel_eff": _num(
                    default=0.95,
                    nullable=True,
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                ),
                "on_final_round": _bool(default=True),
                "stop_when_target_reached": _bool(default=False),
                "gate_frac": _num(default=0.2, minimum=0, maximum=0.5),
                "weight_dtype": _str(
                    default="bf16", choices=("fp32", "bf16", "int8", "int4")
                ),
                "rocm_libraries_to_deploy": _str(nullable=True),
                "build_dir": _str(),
                "targets": _list(Spec("target"), min_len=1),
                "apply_patch": _bool(default=False),
                "build_after_apply": _bool(default=False),
                "build_jobs": _int(nullable=True, minimum=1),
            }
        ),
        "stage07": _section(
            {
                "build_dir": _str(nullable=True),
                "cpus": _str(nullable=True),
                "parity_sample_trained": _int(default=0, minimum=0),
                "parity_sample_audit": _int(default=0, minimum=0),
                "bench_yamls": _list(_str(), default=[]),
                "timing_request_sizes": _list(_int(minimum=1), default=[1], min_len=1),
                "timing_repetitions": _int(default=3, minimum=1),
                "parity_min_match_rate": _num(default=1.0, minimum=0, maximum=1),
                "parity_top_k": _int(default=8, minimum=0),
                "bootstrap": _int(default=1000, minimum=0),
                "production_route": _bool(default=True),
            }
        ),
        "stage08": _section(
            {
                "datasets": _list(
                    _section(
                        {
                            "name": _str(required=True, pattern=r"^[A-Za-z0-9_.-]+$"),
                            "path": _str(required=True),
                            "rocm_libraries_used": _str(),
                        }
                    ),
                    default=[],
                ),
                "tie_tolerance": _num(minimum=0),
                "bootstrap": _int(minimum=0),
            }
        ),
        "gate": _section(
            {
                "per_cell": _bool(default=True),
                "cell_tolerance": _num(
                    default=auditlib.CELL_GATE_DEFAULTS["cell_tolerance"],
                    minimum=0,
                    maximum=1,
                    exclusive_maximum=True,
                ),
                "cell_floor": Spec(
                    "threshold", default=auditlib.CELL_GATE_DEFAULTS["cell_floor"]
                ),
                "alpha": _num(
                    default=auditlib.CELL_GATE_DEFAULTS["alpha"],
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                    exclusive_maximum=True,
                ),
                "tail_floor": Spec(
                    "threshold", default=auditlib.CELL_GATE_DEFAULTS["tail_floor"]
                ),
                "tail_collapse": _num(
                    default=auditlib.CELL_GATE_DEFAULTS["tail_collapse"],
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                ),
                "tail_budget": _int(
                    default=auditlib.CELL_GATE_DEFAULTS["tail_budget"], minimum=0
                ),
                "groups": _str(
                    default=auditlib.CELL_GATE_DEFAULTS["groups"],
                    choices=auditlib.CELL_GATE_GROUPS,
                ),
            }
        ),
        "report": _section(
            {
                "origami_reference": _bool(default=False),
                "timeout_s": _num(default=3600.0, minimum=0),
                "pool_gap_fraction": _num(
                    default=weak_spots.POOL_GAP_FRACTION,
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                ),
            }
        ),
        "correctness": _section(
            {
                "enabled": _bool(default=False),
                "lists": _list(_str(), default=[]),
                "repetitions": _int(
                    default=cgate.DEFAULTS["repetitions"],
                    minimum=cgate.MIN_REPETITIONS,
                ),
                "trigger_repetitions": _int(
                    default=cgate.DEFAULTS["trigger_repetitions"],
                    minimum=cgate.MIN_REPETITIONS,
                ),
                "first_launch": _str(
                    default=cgate.DEFAULTS["first_launch"],
                    choices=cgate.FIRST_LAUNCH_POLICIES,
                ),
                "timeout_s": _num(
                    default=cgate.DEFAULTS["timeout_s"],
                    minimum=0,
                    exclusive_minimum=True,
                ),
                "devices": _list(_int(minimum=0), nullable=True, min_len=1),
                "cache_dir": _str(nullable=True),
                "harness": _str(nullable=True),
                "compiler": _str(nullable=True),
                "blocklist": _str(nullable=True),
            }
        ),
        "hazards": _section(
            {
                "enabled": _bool(default=hazlib.DEFAULTS["enabled"]),
                "timeout_s": _num(
                    default=hazlib.DEFAULTS["timeout_s"],
                    minimum=0,
                    exclusive_minimum=True,
                ),
                "runs": _int(default=hazlib.DEFAULTS["runs"], minimum=1),
                "devices": _list(_int(minimum=0), nullable=True, min_len=1),
                "cache_dir": _str(nullable=True),
            }
        ),
    }
)

# train.* keys passed to stage05 as --<key with dashes> only when set, so an
# unset key keeps stage05's own default.
STAGE05_OPTION_KEYS = (
    "validation_frac",
    "batch_gemms",
    "lr",
    "lr_schedule",
    "warmup_frac",
    "lr_min_frac",
    "eval_points",
    "patience",
    "student_temperature",
    "qat_frac",
    "qat_lr_frac",
    "boundary_margin",
    "boundary_weight",
    "stage1_recall_weight",
    "pseudo_negatives",
    "unmeasured_samples",
    "unmeasured_margin",
    "unmeasured_weight",
    "recipe",
    "pooled_epochs",
    "pooled_batch_gemms",
    "pooled_lr",
    "pooled_eval_points",
    "pooled_threads",
    "pooled_teacher_threads",
    "fine_tune_epochs",
    "fine_tune_lr",
)


def _type_name(v: Any) -> str:
    return "null" if v is None else type(v).__name__


def _check_value(spec: Spec, value: Any, where: str, errors: List[str]) -> None:
    if value is None:
        if not spec.nullable and spec.kind != "section":
            errors.append(f"{where}: must not be null")
        return
    kind = spec.kind
    if kind == "section":
        if not isinstance(value, dict):
            errors.append(f"{where}: expected a mapping, got {_type_name(value)}")
            return
        assert spec.children is not None
        for key in value:
            if key not in spec.children:
                errors.append(
                    f"{where}.{key}: unknown key" if where else f"{key}: unknown key"
                )
        for key, child in spec.children.items():
            path = f"{where}.{key}" if where else key
            if key in value:
                _check_value(child, value[key], path, errors)
            elif child.required:
                errors.append(f"{path}: required key is missing")
        return
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{where}: expected an integer, got {value!r}")
            return
    elif kind == "num":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}: expected a number, got {value!r}")
            return
        if not math.isfinite(float(value)):
            errors.append(f"{where}: must be finite, got {value!r}")
            return
    elif kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{where}: expected true or false, got {value!r}")
        return
    elif kind == "str":
        if not isinstance(value, str) or not value:
            errors.append(f"{where}: expected a non-empty string, got {value!r}")
            return
        if spec.pattern and not re.match(spec.pattern, value):
            errors.append(f"{where}: {value!r} does not match {spec.pattern}")
    elif kind == "threshold":
        span = "(0, 1]" if spec.exclusive_minimum else "[0, 1] (0 disables)"
        if isinstance(value, str) and value.strip().lower() == "origami":
            errors.append(
                f"{where}: 'origami' is not accepted; thresholds are absolute "
                f"selection efficiencies, a number in {span}"
            )
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}: expected a number in {span}, got {value!r}")
            return
        v = float(value)
        low_ok = v > 0.0 if spec.exclusive_minimum else v >= 0.0
        if not (math.isfinite(v) and low_ok and v <= 1.0):
            errors.append(f"{where}: expected a number in {span}, got {value!r}")
        return
    elif kind == "efficiency_or_auto":
        if value == "auto":
            return
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (math.isfinite(float(value)) and 0.0 < float(value) <= 1.0)
        ):
            errors.append(
                f"{where}: expected a number in (0, 1] or 'auto', got {value!r}"
            )
        return
    elif kind == "mapping":
        if not isinstance(value, dict):
            errors.append(f"{where}: expected a mapping, got {_type_name(value)}")
            return
        for key, item in value.items():
            if not isinstance(key, str) or not re.match(_OPTION_NAME, key):
                errors.append(f"{where}: {key!r} is not an option name")
            elif not isinstance(item, (bool, int, float, str, list, type(None))) or (
                isinstance(item, list)
                and not all(isinstance(v, (bool, int, float, str)) for v in item)
            ):
                errors.append(
                    f"{where}.{key}: expected a scalar or a list of scalars, got "
                    f"{_type_name(item)}"
                )
        return
    elif kind == "target":
        if isinstance(value, str):
            if not re.match(_PATH_NAME, value):
                errors.append(f"{where}: invalid target directory name {value!r}")
            return
        if isinstance(value, dict):
            unknown = sorted(set(value) - {"dir", "library_stem"})
            if unknown:
                errors.append(f"{where}: unknown keys {unknown}")
            d = value.get("dir")
            if not isinstance(d, str) or not re.match(_PATH_NAME, d):
                errors.append(f"{where}.dir: expected a directory name")
            stem = value.get("library_stem")
            if stem is not None and not (
                isinstance(stem, str) and stem.startswith("TensileLibrary_")
            ):
                errors.append(f"{where}.library_stem: expected a TensileLibrary_ stem")
            return
        errors.append(f"{where}: expected a directory name or {{dir, library_stem}}")
        return
    elif kind == "env":
        errors.extend(benv.runtime_env_errors(value, where))
        return
    elif kind == "gpu":
        if isinstance(value, bool) or not (
            (isinstance(value, int) and value >= 0)
            or (isinstance(value, str) and value.strip())
        ):
            errors.append(
                f"{where}: expected a GPU index (the one ROCR_VISIBLE_DEVICES "
                f"takes), a PCI address or a GPU-<hex> id, got {value!r}"
            )
        return
    elif kind == "names":
        items = [value] if isinstance(value, str) else value
        assert spec.choices is not None
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(v, str) or v not in spec.choices for v in items)
            or len(set(items)) != len(items)
        ):
            errors.append(
                f"{where}: expected one of {list(spec.choices)} or a list of "
                f"distinct ones, got {value!r}"
            )
        return
    elif kind == "list":
        if not isinstance(value, list):
            errors.append(f"{where}: expected a list, got {_type_name(value)}")
            return
        if len(value) < spec.min_len:
            errors.append(f"{where}: needs at least {spec.min_len} entries")
        assert spec.item is not None
        for i, item in enumerate(value):
            _check_value(spec.item, item, f"{where}[{i}]", errors)
        return
    if spec.choices is not None and value not in spec.choices:
        errors.append(f"{where}: {value!r} is not one of {list(spec.choices)}")
    if spec.minimum is not None and isinstance(value, (int, float)):
        if value < spec.minimum or (spec.exclusive_minimum and value == spec.minimum):
            op = ">" if spec.exclusive_minimum else ">="
            errors.append(f"{where}: must be {op} {spec.minimum:g}, got {value!r}")
    if spec.maximum is not None and isinstance(value, (int, float)):
        if value > spec.maximum or (spec.exclusive_maximum and value == spec.maximum):
            op = "<" if spec.exclusive_maximum else "<="
            errors.append(f"{where}: must be {op} {spec.maximum:g}, got {value!r}")


def _spec_at(dotted: str) -> Spec:
    spec = SCHEMA
    for part in dotted.split("."):
        assert spec.children is not None and part in spec.children, dotted
        spec = spec.children[part]
    return spec


def cfg_get(cfg: Mapping[str, Any], dotted: str) -> Any:
    """Config value at `dotted`, or the schema default when absent."""
    spec = _spec_at(dotted)
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return spec.default
        cur = cur[part]
    return cur


def arch_base(arch: str) -> str:
    return str(arch).split(":", 1)[0]


def compiler_target(arch: str) -> str:
    """Name the Tensile library files of the subtree `arch` carry: a build
    alias ships under its base target's name (the files of hipBLASLt's
    gfx1250v0 subtree end in _gfx1250), any other subtree under its own."""
    base = arch_base(arch)
    m = re.match(r"^(gfx[0-9a-f]+?)v\d+$", base)
    return m.group(1) if m else base


def deploy_targets(cfg: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(weights subdirectory name, index stem) for every deploy target."""
    arch = str(cfg["arch"])
    stem = str(cfg["hipblaslt"]["library_stem"])
    own_suffix = "_" + compiler_target(arch)
    out: List[Tuple[str, str]] = []
    for entry in cfg_get(cfg, "deploy.targets") or [arch_base(arch)]:
        if isinstance(entry, dict):
            d = str(entry["dir"])
            explicit = entry.get("library_stem")
        else:
            d, explicit = str(entry), None
        if explicit:
            out.append((d, str(explicit)))
            continue
        want = compiler_target(d)
        if want == compiler_target(arch):
            out.append((d, stem))
        elif stem.endswith(own_suffix):
            out.append((d, stem[: -len(own_suffix)] + "_" + want))
        else:
            raise ConfigError(
                [
                    f"deploy.targets: cannot derive the library stem for {d!r} "
                    f"from {stem!r}; give {{dir: {d}, library_stem: ...}}"
                ]
            )
    return out


def stage07_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(cfg_get(cfg, "stage07.bench_yamls")) or any(
        cfg_get(cfg, f"stage07.{key}") > 0
        for key in ("parity_sample_trained", "parity_sample_audit")
    )


def stage08_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(cfg_get(cfg, "stage08.datasets"))


VALIDATION_STAGES = (
    ("stage07", 7.0, stage07_enabled),
    ("stage08", 8.0, stage08_enabled),
)


def _split_rounds_possible(cfg: Mapping[str, Any]) -> bool:
    return (
        cfg_get(cfg, "validate.max_split_depth") > 0
        and cfg_get(cfg, "full.total_rounds") >= 3
    )


def _structural_splits_possible(cfg: Mapping[str, Any]) -> bool:
    """Whether stage04b's structural width trigger, which splits a leaf
    whatever its retrain attempts, can split some base cell: one whose
    widest splittable axis spans validate.structural_split_min_octaves,
    above the trigger's depth limit."""
    octaves = float(cfg_get(cfg, "validate.structural_split_min_octaves") or 0)
    max_depth = int(cfg_get(cfg, "validate.max_split_depth"))
    depth = int(cfg_get(cfg, "validate.structural_split_max_depth"))
    if depth < 0:
        depth = max_depth
    if octaves <= 0 or min(depth, max_depth) <= 0:
        return False
    return any(
        max(sc.cell_axis_octaves(c).values(), default=0.0) >= octaves
        for c in all_cell_labels()
    )


REMOVED_KEYS = {
    "probe": (
        "the probe stage was removed and adaptive timing is always on: "
        "hipblaslt-bench times every kernel adaptively (bench.adaptive); "
        "delete this section"
    ),
    "bench.adaptive.enabled": "adaptive timing is always on; delete this key",
    "bench.rebench.iters_factor": (
        "re-bench lines keep the adaptive timing of their bench line; delete "
        "this key"
    ),
    "planner.probe_kernels": (
        "plans name no probe kernels: the probe stage was removed; delete this key"
    ),
}


def _without_removed_keys(cfg: Any, errors: List[str]) -> Any:
    """`cfg` without the REMOVED_KEYS it sets, with an error for each."""
    if not isinstance(cfg, dict):
        return cfg
    out = copy.deepcopy(cfg)
    for dotted, why in REMOVED_KEYS.items():
        *parents, leaf = dotted.split(".")
        node: Any = out
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and leaf in node:
            del node[leaf]
            errors.append(f"{dotted}: {why}")
    return out


def _adaptive_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    """hipblaslt-bench's rules for an adaptive timing config; it times no
    kernel of a line that breaks them. They hold for the fields of every
    line: `bench.adaptive`, with `bench.adaptive_long`'s knobs over them on
    a long line and the slow band's iterations on a near-winner re-read."""
    base = {k: cfg_get(cfg, f"bench.adaptive.{k}") for k in ADAPTIVE_DEFAULTS}
    variants = [("bench.adaptive", base)]
    if cfg_get(cfg, "bench.adaptive_long.enabled"):
        long_knobs = {
            k: cfg_get(cfg, f"bench.adaptive_long.{k}") for k in LONG_KNOB_KEYS
        }
        variants.append(
            ("bench.adaptive with bench.adaptive_long", {**base, **long_knobs})
        )
    for where, a in variants:
        if a["max_iters"] > 0 and a["min_iters"] > a["max_iters"]:
            errors.append(f"{where}: min_iters exceeds max_iters")
        if a["max_measure_time"] > 0 and a["measure_time"] > a["max_measure_time"]:
            errors.append(f"{where}: measure_time exceeds max_measure_time")
        if a["max_measure_time"] <= 0 and a["max_iters"] <= 0:
            errors.append(
                f"{where}: max_measure_time or max_iters must be above 0 (the "
                f"ceiling of a measurement)"
            )
    if base["stability_threshold"] > 0 and (
        base["stability_window"] < 2 or base["stability_interval"] < 1
    ):
        errors.append(
            "bench.adaptive.stability_threshold above 0 needs stability_window "
            ">= 2 and stability_interval >= 1"
        )
    slow_iters = cfg_get(cfg, "bench.rebench.slow_min_iters")
    if base["max_iters"] > 0 and slow_iters > base["max_iters"]:
        errors.append("bench.rebench.slow_min_iters exceeds bench.adaptive.max_iters")
    if cfg_get(cfg, "bench.rebench.fast_below_us") > cfg_get(
        cfg, "bench.rebench.slow_from_us"
    ):
        errors.append("bench.rebench.fast_below_us exceeds slow_from_us")
    copies = cfg_get(cfg, "bench.rebench.inline_copies")
    if copies * max(2, cfg_get(cfg, "bench.rebench.max_kernels")) > MAX_BENCH_LIST:
        errors.append(
            f"bench.rebench.inline_copies x max_kernels lists more kernels than a "
            f"bench line takes ({MAX_BENCH_LIST})"
        )
    if cfg_get(cfg, "bench.rebench.align_to_original") and not cfg_get(
        cfg, "bench.rebench.with_original"
    ):
        errors.append(
            "bench.rebench.align_to_original shifts the re-reads to the original "
            "timings they merge with: it needs with_original"
        )


# train keys of the per-leaf schedule, which train.recipe pooled_ft does not
# read: they reach stage05 only when set, but for epochs (its default).
PER_LEAF_SCHEDULE_KEYS = ("epochs", "lr", "eval_points", "patience")


def _train_config_checks(
    cfg: Mapping[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """The train section's options against each other, as stage05 would
    refuse or ignore them."""
    if cfg_get(cfg, "train.launch_keys") and cfg_get(cfg, "train.catalog") != "v3":
        errors.append("train.launch_keys: launch keys need train.catalog v3")
    if cfg_get(cfg, "train.non_temporal_family_rule") and not (
        cfg_get(cfg, "train.catalog") == "v3"
        and cfg_get(cfg, "train.non_temporal_rule")
    ):
        errors.append(
            "train.non_temporal_family_rule: the family-wise rule refines the "
            "non-temporal rule of a v3 model; it needs train.catalog v3 and "
            "train.non_temporal_rule: true"
        )
    if (
        cfg_get(cfg, "train.served_candidates")
        and cfg_get(cfg, "train.catalog") != "v3"
    ):
        errors.append(
            "train.served_candidates: served candidates need train.catalog v3"
        )
    _whitelist_tier_config_checks(cfg, errors)
    _pseudo_negative_config_checks(cfg, errors)
    if cfg_get(cfg, "train.recipe") == "pooled_ft":
        for key, wrong, why in (
            (
                "validation_frac",
                cfg_get(cfg, "train.validation_frac") == 0,
                "the pooled student and the fine-tuning keep their checkpoints by "
                "validation GEMMs",
            ),
            (
                "boundary_margin",
                (cfg_get(cfg, "train.boundary_margin") or 0) > 0,
                "the pooled student already trains on every cell's GEMMs",
            ),
            ("qat_frac", (cfg_get(cfg, "train.qat_frac") or 0) > 0, "per_leaf only"),
        ):
            if wrong:
                errors.append(f"train.{key} with train.recipe pooled_ft: {why}")
        train = cfg.get("train") or {}
        ignored = [
            k
            for k in PER_LEAF_SCHEDULE_KEYS
            if k in train
            and (k != "epochs" or train[k] != _spec_at(f"train.{k}").default)
        ]
        if ignored:
            warnings.append(
                f"train.{', train.'.join(ignored)}: train.recipe pooled_ft does "
                f"not read the per-leaf schedule (its own: pooled_epochs, "
                f"pooled_lr, pooled_eval_points, fine_tune_epochs, fine_tune_lr; "
                f"it stops no phase early)"
            )
    smax = cfg_get(cfg, "train.smart_k_max")
    if smax is not None and cfg_get(cfg, "train.smart_k_min") > smax:
        errors.append("train.smart_k_min > train.smart_k_max")


def validate_config(
    cfg: Any, mode: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for a loaded, env-expanded config run in --mode
    `mode` (None: any mode); what only stage 7 needs is a warning in --mode
    audit, which runs no stage 7."""
    errors: List[str] = []
    warnings: List[str] = []
    _check_value(SCHEMA, _without_removed_keys(cfg, errors), "", errors)
    if errors:
        return errors, warnings

    devices = cfg["runtime"]["devices"]
    if len(set(devices)) != len(devices):
        errors.append("runtime.devices: duplicate device indices")

    arch = cfg["arch"]
    stem = cfg["hipblaslt"]["library_stem"]
    tuned = build_alias_of(arch)
    if tuned:
        errors.append(
            f"arch: {arch!r} is hipBLASLt's build alias for a library subtree "
            f"({tuned}'s logic built for {compiler_target(arch)}), not an "
            f"architecture; set arch: {tuned} with a hipblaslt.library_stem "
            f"ending in _{tuned}, and list {arch_base(arch)} in deploy.targets "
            f"to ship the model to that subtree"
        )
    if not stem.endswith("_" + compiler_target(arch)):
        warnings.append(
            f"hipblaslt.library_stem does not end in _{compiler_target(arch)}; "
            f"Tensile names {arch} libraries with that suffix"
        )
    try:
        targets = deploy_targets(cfg)
    except ConfigError as e:
        errors.extend(e.errors)
        targets = []
    dirs = [d for d, _ in targets]
    if len(set(dirs)) != len(dirs):
        errors.append("deploy.targets: duplicate target directories")
    _block_scale_config_checks(cfg, errors)
    known = hllib.QUEUE_COUNTS.get(archlib.canonical_arch(arch))
    num_xcd = cfg_get(cfg, "hardware.num_xcd")
    if num_xcd is not None and known is not None and num_xcd != known:
        errors.append(
            f"hardware.num_xcd {num_xcd}: hipBLASLt's dynamic-queue kernels of "
            f"{arch} keep {known} queues whatever the device; leave it unset"
        )
    _adaptive_config_checks(cfg, errors)
    few = rebenchlib.align_copies_warning(
        cfg_get(cfg, "bench.rebench.align_to_original"),
        cfg_get(cfg, "bench.rebench.inline_copies"),
    )
    if few:
        warnings.append(f"bench.rebench.{few}")
    _guard_config_checks(cfg, errors, warnings)
    errors += suslib.config_errors(cfg)
    _correctness_config_checks(cfg, errors, mode)
    _cell_gate_config_checks(cfg, errors)
    overrides = [k for k in benv.runtime_env(cfg) if benv.is_launch_override(k)]
    if overrides:
        warnings.append(
            f"runtime.env sets the launch overrides {', '.join(sorted(overrides))}: "
            f"every bench times its kernels launched under them, as platform.json "
            f"and the deploy record say"
        )
    forced = sorted(k for k in overrides if k in benv.HYBRID_FORCE_VARS)
    if forced:
        warnings.append(
            f"runtime.env sets {', '.join(forced)}: it forces the hybrid schedule "
            f"of every bench, over the one TensileLite picks and the one each "
            f"execution context asks for (--hybrid_assignment_policy), so the "
            f"labels may measure another schedule than the runtime makes"
        )

    library_errors, library_warnings = lp.config_problems(cfg)
    errors += library_errors
    warnings += library_warnings
    planner = cfg.get("planner")
    if isinstance(planner, dict) and not planner.get("enabled"):
        ignored = sorted(k for k in planner if k != "enabled")
        if ignored:
            warnings.append(
                f"planner: {', '.join(ignored)} ignored (planner.enabled is false)"
            )
    elif isinstance(planner, dict) and planner.get("top_k_round0") is not None:
        if planner.get("model_file"):
            warnings.append(
                "planner.top_k_round0 ignored: planner.model_file ranks every "
                "round, round 0 included"
            )
        elif cfg_get(cfg, "planner.mode") == "exhaustive":
            warnings.append(
                "planner.top_k_round0 ignored: planner.mode exhaustive benches "
                "every kernel"
            )

    target = cfg_get(cfg, "seed.target_per_cell")
    per_leaf = max(
        cfg_get(cfg, "seed.shapes_per_cell"), cfg_get(cfg, "train.min_cell_gemms")
    )
    train_min = cfg_get(cfg, "train.min_cell_gemms")
    val_min = cfg_get(cfg, "validate.min_cell_gemms")
    if train_min > target:
        errors.append(
            f"train.min_cell_gemms ({train_min}) > seed.target_per_cell "
            f"({target}): round_0 would train no cell"
        )
    elif train_min == target:
        warnings.append(
            f"seed.target_per_cell equals train.min_cell_gemms ({target}): a cell "
            f"trains in round_0 only when every one of its shapes benches; give "
            f"seed.target_per_cell some slack"
        )
    if val_min > per_leaf:
        errors.append(
            f"validate.min_cell_gemms ({val_min}) exceeds the new GEMMs a leaf "
            f"gets per round ({per_leaf} = max(seed.shapes_per_cell, "
            f"train.min_cell_gemms)): no cell could be evaluated"
        )
    if _split_rounds_possible(cfg):
        if per_leaf < 2 * val_min:
            errors.append(
                f"split budget infeasible: a split needs validate.min_cell_gemms "
                f"({val_min}) new GEMMs on each side, but a leaf gets {per_leaf} "
                f"per round; raise seed.shapes_per_cell to >= {2 * val_min}, "
                f"lower validate.min_cell_gemms, or set validate.max_split_depth: 0"
            )
        last_split_round = cfg_get(cfg, "full.total_rounds") - 2
        if cfg_get(cfg, "validate.split_after_attempts") > last_split_round - 1:
            if _structural_splits_possible(cfg):
                warnings.append(
                    "validate.split_after_attempts leaves no round in which a "
                    "cell splits for its selection efficiency (such splits need "
                    "that many prior retrains and are disabled in the final "
                    "round); only the structural width trigger "
                    "(validate.structural_split_min_octaves) splits cells"
                )
            else:
                warnings.append(
                    "validate.split_after_attempts leaves no round in which a "
                    "split can happen (splits need that many prior retrains and "
                    "are disabled in the final round)"
                )
    elif cfg_get(cfg, "validate.max_split_depth") > 0:
        warnings.append(
            "validate.max_split_depth > 0 but full.total_rounds < 3: the final "
            "round never splits, so no split can happen"
        )
    _context_config_checks(cfg, errors, warnings)
    _audit_config_checks(cfg, errors)
    _final_retrain_config_checks(cfg, errors, warnings)
    _train_config_checks(cfg, errors, warnings)

    deploy_root = cfg_get(cfg, "deploy.rocm_libraries_to_deploy")
    apply_patch = cfg_get(cfg, "deploy.apply_patch")
    build_after = cfg_get(cfg, "deploy.build_after_apply")
    if apply_patch and not deploy_root:
        errors.append("deploy.apply_patch requires deploy.rocm_libraries_to_deploy")
    if build_after and not apply_patch:
        warnings.append("deploy.build_after_apply is ignored without apply_patch")
    if (
        cfg_get(cfg, "deploy.gate_frac") == 0
        and cfg_get(cfg, "deploy.target_sel_eff") is not None
        and cfg_get(cfg, "full.total_rounds") > 1
    ):
        warnings.append(
            "deploy.gate_frac is 0: no round can measure the bundle it trains, so "
            "deploy.target_sel_eff never deploys one; only the final round "
            "(deploy.on_final_round) or --deploy deploys, unmeasured"
        )
    stage07_errors: List[str] = []
    _stage07_config_checks(cfg, stage07_errors, warnings)
    if mode == "audit":
        warnings += [f"{e} (--mode audit runs no stage 7)" for e in stage07_errors]
    else:
        errors += stage07_errors
    return errors, warnings


def _guard_config_checks(
    cfg: Mapping[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """bench.quiet_host must not start a bench part under load it would abort
    it for; a stall timeout of 0 lets a hung line hold its GPU for good."""
    quiet, hot = cfg_get(cfg, "bench.quiet_host.quiet_busy"), cfg_get(
        cfg, "bench.quiet_host.hot_busy"
    )
    if cfg_get(cfg, "bench.quiet_host.enabled") and quiet > hot:
        errors.append(
            f"bench.quiet_host: quiet_busy {quiet:g} above hot_busy {hot:g} starts "
            f"a part under load that aborts it"
        )
    if cfg_get(cfg, "bench.stall_timeout_s") == 0:
        warnings.append(
            "bench.stall_timeout_s is 0: a line that hangs holds its GPU until "
            "someone stops the run"
        )


def exclude_gpus(
    cfg: Dict[str, Any], audit_devices: Optional[List[int]] = None
) -> Tuple[Dict[str, Any], Optional[List[int]], List[str], List[str]]:
    """`cfg` (and the audit's --devices, `audit_devices`) without the devices
    whose GPU runtime.exclude_gpus names, in runtime.devices, audit.devices,
    correctness.devices and hazards.devices; (config, audit devices, notes,
    errors)."""
    refs = cfg_get(cfg, "runtime.exclude_gpus") or []
    if not refs:
        return cfg, audit_devices, [], []
    excluded, unknown = kfd.gpu_refs(refs)
    notes = [f"runtime.exclude_gpus: {r} names no GPU here" for r in unknown]
    errors: List[str] = []
    out = copy.deepcopy(cfg)
    for dotted in (
        "runtime.devices",
        "audit.devices",
        "correctness.devices",
        "hazards.devices",
    ):
        listed = cfg_get(out, dotted)
        if not listed:
            continue
        kept, dropped = kfd.without_gpus(listed, excluded)
        if not dropped:
            continue
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            node = node[part]
        node[leaf] = kept
        notes.append(
            f"{dotted}: devices {dropped} left out (runtime.exclude_gpus names "
            f"their GPUs)"
        )
        if not kept:
            errors.append(f"{dotted}: every device is excluded by runtime.exclude_gpus")
    if audit_devices:
        kept, dropped = kfd.without_gpus(audit_devices, excluded)
        if dropped:
            notes.append(
                f"--devices: devices {dropped} left out (runtime.exclude_gpus)"
            )
            if not kept:
                errors.append(
                    "--devices: every device is excluded by runtime.exclude_gpus"
                )
        audit_devices = kept
    return out, audit_devices, notes, errors


# hipBLASLt data types whose libraries all take block scales (MX).
BLOCK_SCALED_TYPES = ("f4_r", "f6_r", "bf6_r")


def _block_scale_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    """An operand of a type of BLOCK_SCALED_TYPES needs its block format in
    scaleA / scaleB: without one a bench line asks for an unscaled GEMM no
    library holds, and shapes keep K off the block multiples."""
    hl = cfg["hipblaslt"]
    for operand in ("A", "B"):
        dtype = hl.get(f"{operand.lower()}_type")
        mode = hl.get(f"scale{operand}")
        if dtype in BLOCK_SCALED_TYPES and not mx_block_size_for_scale_mode(mode):
            errors.append(
                f"hipblaslt.scale{operand}: {dtype} operands are block-scaled; set "
                f"scale{operand} to the library's block format (3: Block_32_UE8M0), "
                f"not {'nothing' if mode is None else mode}"
            )


def _whitelist_tier_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    """train.whitelist_tier_k against the catalog, the smart_K whitelist and
    launch keys of the train section of `cfg`."""
    k = cfg_get(cfg, "train.whitelist_tier_k")
    if not k:
        return
    whitelist = cfg_get(cfg, "train.whitelist")
    for wrong, why in (
        (
            cfg_get(cfg, "train.catalog") != "v3",
            "the tier lists view keys of a v3 model; it needs train.catalog v3",
        ),
        (
            whitelist not in (None, "none"),
            f"the tier replaces the smart_K whitelist (train.whitelist "
            f"{whitelist}); set train.whitelist none or leave it unset",
        ),
        (
            bool(cfg_get(cfg, "train.launch_keys")),
            "the tier lists view keys, and train.launch_keys scores each launch "
            "class of a view key on its own",
        ),
    ):
        if wrong:
            errors.append(f"train.whitelist_tier_k {k}: {why}")


def _pseudo_negative_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    """train.pseudo_negatives against the catalog and the sample count
    stage05 takes (its own default where train.unmeasured_samples is
    unset)."""
    mode = cfg_get(cfg, "train.pseudo_negatives")
    samples = cfg_get(cfg, "train.unmeasured_samples")
    if mode == "targeted" and cfg_get(cfg, "train.catalog") != "v3":
        errors.append(
            "train.pseudo_negatives: targeted draws keys of a v3 model; it needs "
            "train.catalog v3"
        )
    if mode == "none" and samples:
        errors.append(
            f"train.pseudo_negatives none draws none; train.unmeasured_samples "
            f"{samples} asks for some"
        )
    if mode in ("random", "targeted") and samples == 0:
        errors.append(
            f"train.pseudo_negatives {mode} draws at least one per GEMM; "
            f"train.unmeasured_samples is 0"
        )
    if mode == "random" and samples is None:
        errors.append(
            "train.pseudo_negatives random draws train.unmeasured_samples "
            "pseudo-negatives per GEMM; set it"
        )


def _stage07_config_checks(
    cfg: Mapping[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """stage07 needs a build that serves the model the run deploys."""
    cpus = cfg_get(cfg, "stage07.cpus")
    if cpus is not None and not fingerprint.parse_cpulist(str(cpus)):
        errors.append(f"stage07.cpus: {cpus!r} is not a CPU list such as 4,5 or 8-11")
    if (
        cfg_get(cfg, "stage07.parity_sample_audit") > 0
        and auditlib.config_dir(cfg) is None
    ):
        errors.append(
            "stage07.parity_sample_audit samples the audit set's GEMMs: set audit.dir"
        )
    if not stage07_enabled(cfg) or cfg_get(cfg, "stage07.build_dir"):
        return
    if not cfg_get(cfg, "deploy.rocm_libraries_to_deploy"):
        errors.append(
            "stage07 needs a hipBLASLt build that serves the model this run "
            "deploys: set stage07.build_dir (a build whose TensileLite serves "
            "the trained format; stage 7 serves the staged model from a library "
            "directory beside the bench library) or deploy.rocm_libraries_to_deploy "
            "with deploy.apply_patch and deploy.build_after_apply"
        )
    elif not (
        cfg_get(cfg, "deploy.apply_patch") and cfg_get(cfg, "deploy.build_after_apply")
    ):
        warnings.append(
            "stage07 validates the model this run deploys and the bench loads "
            "it from the build tree: without stage07.build_dir, or "
            "deploy.apply_patch and deploy.build_after_apply, it can only run "
            "after a manual deploy and model build (hipblaslt-tilewright-models)"
        )


# ── config loading ──────────────────────────────────────────────────────────


_UNEXPANDED_VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def expand_config_env(node: Any, missing: set) -> Any:
    """Expand ${VAR}, $VAR and a leading ~ in every string; record names
    that stay unexpanded in `missing`."""
    if isinstance(node, dict):
        return {k: expand_config_env(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_config_env(v, missing) for v in node]
    if isinstance(node, str) and ("$" in node or node.startswith("~")):
        names = set(_UNEXPANDED_VAR_RE.findall(node))
        missing.update(n for n in names if os.environ.get(n, "") == "")
        out = os.path.expanduser(os.path.expandvars(node))
        if "${" in out and not missing:
            missing.add(f"(unsupported syntax in {node!r})")
        return out
    return node


def load_config(path: Path) -> Tuple[Dict[str, Any], str]:
    """(expanded config, raw text). Raises ConfigError."""
    if not path.is_file():
        raise ConfigError([f"config not found: {path}"])
    raw = path.read_text()
    try:
        cfg = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError([f"{path}: invalid yaml: {e}"])
    if not isinstance(cfg, dict):
        raise ConfigError([f"{path}: top level must be a mapping"])
    missing: set = set()
    cfg = expand_config_env(cfg, missing)
    if missing:
        raise ConfigError(
            [
                f"{path} references unset or empty environment variable(s): "
                f"{', '.join(sorted(missing))}"
            ]
        )
    return cfg, raw


# Every key that sets how a bench times kernels; a snapshot holds each one.
BENCH_TIMING_KEYS = (
    *(f"bench.adaptive.{k}" for k in ADAPTIVE_DEFAULTS),
    "bench.skip_slow_solution_ratio",
    "bench.kernel_order",
    "bench.order_seed",
    "bench.sacrificial_kernel",
    "bench.warmup_pass",
    *(f"bench.warmup_line.{k}" for k in WARMUP_LINE_DEFAULTS),
    *(f"bench.canary.{k}" for k in CANARY_DEFAULTS),
    "bench.rebench.enabled",
    *(f"bench.rebench.{k}" for k in rebenchlib.REREAD_DEFAULTS),
    *(f"bench.adaptive_long.{k}" for k in ADAPTIVE_LONG_DEFAULTS),
    *(f"bench.sustained.{k}" for k in suslib.DEFAULTS),
)


def resolve_bench_timing(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """`cfg` with every BENCH_TIMING_KEYS key set to the value the stages
    apply, its default when the config leaves it out."""
    out = copy.deepcopy(dict(cfg))
    for dotted in BENCH_TIMING_KEYS:
        *parents, leaf = dotted.split(".")
        node = out
        for part in parents:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[leaf] = cfg_get(cfg, dotted)
    return out


def bench_timing_text(cfg: Mapping[str, Any]) -> str:
    """How the benches of `cfg` time kernels, in one line."""
    adaptive = ", ".join(
        f"{k} {cfg_get(cfg, 'bench.adaptive.' + k)}" for k in ADAPTIVE_DEFAULTS
    )
    if cfg_get(cfg, "bench.adaptive_long.enabled"):
        long = {
            k: cfg_get(cfg, f"bench.adaptive_long.{k}") for k in ADAPTIVE_LONG_DEFAULTS
        }
        adaptive += (
            f"; on lines whose roofline is at least {long['min_roofline_us']:g} us "
            f"(peak_tflops {long['peak_tflops']}, bandwidth_gbs "
            f"{long['bandwidth_gbs']}) "
            + ", ".join(f"{k} {long[k]}" for k in LONG_KNOB_KEYS)
        )
    ratio = float(cfg_get(cfg, "bench.skip_slow_solution_ratio") or 0.0)
    if ratio <= 0:
        screen = "no skip screen (every listed kernel is timed)"
    elif cfg_get(cfg, "planner.enabled"):
        screen = (
            f"skip_slow_solution_ratio {ratio:g} against every listed kernel's "
            f"warm-up (skip_screen_prepass on the planner's lines)"
        )
        if cfg_get(cfg, "bench.sacrificial_kernel"):
            screen += ", behind a sacrificial copy of the first listed kernel"
    else:
        screen = (
            f"skip_slow_solution_ratio {ratio:g} against the fastest warm-up "
            f"so far (no screen prepass)"
        )
    if not cfg_get(cfg, "planner.enabled"):
        on = "on" if cfg_get(cfg, "bench.warmup_pass") else "off"
        screen += f"; warmup_pass {on}"
    if cfg_get(cfg, "bench.warmup_line.enabled"):
        size = cfg_get(cfg, "bench.warmup_line.size")
        screen += (
            f"; every bench process starts with a discarded warm-up line of "
            f"{cfg_get(cfg, 'bench.warmup_line.kernels')} kernel(s) on "
            f"{size}x{size}x{size}"
        )
    else:
        screen += "; no warm-up line"
    if cfg_get(cfg, "bench.kernel_order") == "shuffled":
        order = (
            f"kernel lists in a per-GEMM shuffle (order_seed "
            f"{cfg_get(cfg, 'bench.order_seed')})"
        )
    else:
        order = "kernel lists in the order listed"
    if cfg_get(cfg, "bench.rebench.enabled"):
        rr = {k: cfg_get(cfg, f"bench.rebench.{k}") for k in rebenchlib.REREAD_DEFAULTS}
        slow_window = (
            f" in windows of at least {rr['slow_min_iters']} iterations"
            if rr["slow_min_iters"]
            else ""
        )
        floor = (
            f" or {rr['tolerance_us']:g} us, whichever is wider,"
            if rr["tolerance_us"] > 0
            else ""
        )
        copies = (
            f" listing every candidate {rr['inline_copies']}x"
            if rr["inline_copies"] > 1
            else ""
        )
        rebench = (
            f"near-winner re-reads: the candidates within {rr['tolerance']:g}"
            f"{floor} of the winner, at most {rr['max_kernels']}, read "
            f"{rr['fast_runs']}x below {rr['fast_below_us']:g} us, "
            f"{rr['slow_runs']}x from {rr['slow_from_us']:g} us{slow_window}, "
            f"{rr['runs']}x in between, each in a bench of its own{copies}, no "
            f"screen, "
            + ("a sacrificial first kernel, " if rr["sacrificial_kernel"] else "")
            + f"GEMMs {rr['gemm_order']}, labelled by the median of "
            + (
                "the original and the re-reads"
                if rr["with_original"]
                else "the re-reads"
            )
            + (
                ", each run's re-reads shifted by their median difference from "
                "the original"
                if rr["with_original"] and rr["align_to_original"]
                else ""
            )
        )
    else:
        rebench = "no near-winner re-reads"
    if cfg_get(cfg, "bench.sustained.enabled"):
        sus = {k: cfg_get(cfg, f"bench.sustained.{k}") for k in suslib.DEFAULTS}
        rebench += (
            f"; sustained finalists of power-capped GEMMs (winner >= "
            f"{sus['min_us']:g} us, or >= {sus['compute_bound_tflops']:g} TFLOP/s "
            f"and >= {sus['compute_bound_min_us']:g} us): the {sus['top_k']} "
            f"fastest and those within {sus['tolerance']:g} of the winner, at "
            f"most {sus['max_kernels']}, {sus['runs']} runs, warmup_time "
            f"{sus['warmup_time']:g}, measure_time {sus['measure_time']:g}, "
            f"max_measure_time {sus['max_measure_time']:g}"
        )
    else:
        rebench += "; no sustained finalists"
    if cfg_get(cfg, "bench.canary.enabled"):
        canary = (
            f"canary timed {cfg_get(cfg, 'bench.canary.timings')}x before and "
            f"after every bench process's shapes, a block benched again once, "
            f"after {cfg_get(cfg, 'bench.canary.backoff_s'):g} s, when a reading "
            f"is over {cfg_get(cfg, 'bench.canary.tolerance'):g} above its GPU's "
            f"median"
        )
    else:
        canary = "no canary"
    return (
        f"adaptive timing (always on) with {adaptive}; {screen}; {order}; "
        f"{rebench}; {canary}"
    )


def guard_text(cfg: Mapping[str, Any]) -> str:
    """How the stages of `cfg` guard their benches, in one line."""
    quiet = host_load.Settings.of(cfg)
    health = gpu_health.Settings.of(cfg)
    parts = [
        quiet.describe() if quiet.enabled else "no quiet-host gate",
        (
            f"GPU health checked against the run's start every {health.poll_s:g} s"
            + (", with the host's ttm workers" if health.ttm else "")
            + ", a failed check holding the GPUs that fail it (every GPU when "
            "the host fails it or every GPU checked does)"
            + (
                f" (giving up after {health.max_wait_s:g} s)"
                if health.max_wait_s > 0
                else ""
            )
            if health.enabled
            else "no GPU health checks"
        ),
        (
            f"a canary of {health.canary.lines} reference line(s) x "
            f"{health.canary.timings} per GPU and session, tolerance "
            f"{health.canary.tolerance:g}"
            if health.canary_on
            else "no health canary"
        ),
        f"a line silent for {cfg_get(cfg, 'bench.stall_timeout_s'):g} s is killed "
        f"and quarantined in the run for its protocol",
    ]
    refs = cfg_get(cfg, "runtime.exclude_gpus") or []
    if refs:
        parts.append("excluded GPUs " + ", ".join(str(r) for r in refs))
    return "; ".join(parts)


# Keys an invocation may change without --allow-code-change: where and how
# patiently the run executes, how far it goes, what it does on a failed cell
# and what it validates and reports after training, below which seed recall
# stage02 warns, how it builds and
# measures with the audit set and the protocol of its official number, the
# final retrain, which records its own options and trains again when they
# change (`FinalRetrain`), and the per-cell gate; a final retrain's decision
# records its gate settings and official protocol and is made again under
# new ones. None changes a GEMM's rows or a cell a round trained, except the
# warm-up line every bench process starts with: it benches no line of the
# run and only keeps a process's first read of each library file out of its
# screen, every stage's platform.json records it, and a run or audit
# directory benched before it existed resumes with it on.
UNPINNED_KEYS = (
    "runtime.devices",
    "runtime.exclude_gpus",
    "runtime.gpu_lease_wait_s",
    "runtime.stage_retries",
    "runtime.stage_retry_backoff_s",
    "runtime.min_free_gb",
    "bench.stall_timeout_s",
    "bench.startup_grace_s",
    "bench.quiet_host",
    "bench.health",
    "bench.warmup_line",
    "planner.recall_min_ceiling",
    "train.workers",
    "train.threads_per_worker",
    "train.pooled_threads",
    "train.pooled_teacher_threads",
    "train.on_cell_failure",
    "train.carry_max_fraction",
    "full.total_rounds",
    "full.final_retrain",
    "audit.build",
    "audit.devices",
    "audit.v2",
    "audit.v2_set_tolerance",
    "audit.v2_set_tolerance_us",
    "audit.v2_set_max",
    "audit.sustained",
    "audit.official",
    "deploy.rocm_libraries_to_deploy",
    "deploy.build_dir",
    "deploy.apply_patch",
    "deploy.build_after_apply",
    "deploy.build_jobs",
    "stage07",
    "stage08",
    "gate",
    "report",
    "correctness.timeout_s",
    "correctness.devices",
    "correctness.cache_dir",
    "correctness.blocklist",
    "hazards.timeout_s",
    "hazards.devices",
    "hazards.cache_dir",
)
# Also unpinned in an audit directory: the keys of the training configs its
# benches never read, so that the runs sharing the set may differ in them,
# and the hazard canary's, whose quarantine only leaves kernels out of the
# pool the set's completeness is checked against (`config_pool_digests`).
AUDIT_UNPINNED_KEYS = (
    "config_id",
    "seed.seed",
    "seed.target_per_cell",
    "seed.shapes_per_cell",
    "planner.exhaustive_max_kernels",
    "planner.seed_per_cell",
    "planner.neighbors_k",
    "planner.rules",
    "planner.top_k",
    "planner.top_k_round0",
    "planner.families_top_f",
    "planner.family_variants",
    "planner.explore_random_k",
    "planner.min_coverage",
    "planner.max_list",
    "planner.seed",
    "train",
    "validate",
    "full",
    "deploy",
    "audit.dir",
    "audit.target",
    "audit.bootstrap",
    "correctness",
    "hazards",
)


def pin_parts(mode: str) -> Tuple[str, ...]:
    """The parts of the pin (lib/pinning.py) an invocation in `mode` must
    keep: a training run all of them; an audit directory, which runs of
    other code share and extend, the config keys its benches read; a
    validation, which produces no data or model of the run, none."""
    if mode == "validate":
        return ()
    if mode == "audit":
        return ("config",)
    return pinning.PARTS


# Sections and keys pinned only when the config sets them: a config without
# one names what it would with the keys that came before it (the library
# section: the planner's legacy bank keys; planner.top_k_round0:
# planner.top_k), so its runs keep their pins.
PINNED_WHEN_SET = ("library", "planner.top_k_round0")
# Keys pinned only at a value other than the one at which they change
# nothing, which is what a pin written before they existed means, so that
# the runs and audit directories produced before them keep their pins.
PINNED_UNLESS_NEUTRAL: Dict[str, Any] = {
    f"bench.rebench.{k}": v for k, v in rebenchlib.NEUTRAL_REREADS.items()
}


def _config_sets(cfg: Mapping[str, Any], dotted: str) -> bool:
    """Whether `cfg` itself sets `dotted`: to a value other than null, and a
    section to a non-empty one."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or cur.get(part) is None:
            return False
        cur = cur[part]
    return bool(cur) if isinstance(cur, Mapping) else True


def pinned_config(cfg: Mapping[str, Any], audit: bool = False) -> Dict[str, Any]:
    """{dotted key: value} of every schema key of `cfg` at the value the
    stages apply (its default when absent), without UNPINNED_KEYS (and
    AUDIT_UNPINNED_KEYS for an audit directory), the sections and keys of
    PINNED_WHEN_SET it does not set and the keys of PINNED_UNLESS_NEUTRAL at
    their neutral values, and the digests of the files `correctness.lists`
    names (`correctness.lists_sha256`), whose contents decide which kernels
    leave the pool."""
    skip = UNPINNED_KEYS + (AUDIT_UNPINNED_KEYS if audit else ())
    skip += tuple(s for s in PINNED_WHEN_SET if not _config_sets(cfg, s))
    out: Dict[str, Any] = {}

    def walk(spec: Spec, prefix: str) -> None:
        assert spec.children is not None
        for key, child in spec.children.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if dotted in skip:
                continue
            if child.kind == "section":
                walk(child, dotted)
                continue
            value = cfg_get(cfg, dotted)
            if (
                dotted in PINNED_UNLESS_NEUTRAL
                and value == PINNED_UNLESS_NEUTRAL[dotted]
            ):
                continue
            out[dotted] = copy.deepcopy(value)

    walk(SCHEMA, "")
    if "correctness.lists" in out and out["correctness.lists"]:
        out["correctness.lists_sha256"] = cxlib.list_digests(out["correctness.lists"])
    return out


class _FlowListDumper(yaml.SafeDumper):
    pass


def _scalar_list_repr(dumper: yaml.SafeDumper, data: list) -> Any:
    flow = all(isinstance(x, (int, float, bool, str)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_FlowListDumper.add_representer(list, _scalar_list_repr)


def write_snapshot(
    run_dir: Path,
    cfg: Mapping[str, Any],
    raw: str,
    mode: str,
    pin: Optional[Mapping[str, Any]] = None,
    resolved: Sequence[str] = (),
) -> Path:
    """Write `<run_dir>/configs/<ts>_<mode>.yaml` (expanded, with the bench
    timing resolved and summed up in its first lines, followed by the
    invocation's `pin` and the `resolved` lines: what the invocation read
    from the device) next to the raw source text; never reuses an existing
    name."""
    out_dir = run_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1000):
        name = f"{stamp}_{mode}" + (f"_{n}" if n else "")
        snap = out_dir / f"{name}.yaml"
        try:
            fd = os.open(snap, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        head = textwrap.wrap(
            "Bench timing of this invocation, resolved below: "
            + bench_timing_text(cfg),
            width=77,
            break_on_hyphens=False,
        )
        if pin is not None:
            head += textwrap.wrap(
                "Produced with: " + pinning.describe(pin),
                width=77,
                break_on_hyphens=False,
                break_long_words=False,
            )
        if benv.runtime_env_text(cfg):
            head += textwrap.wrap(
                "Every stage, bench and cotenant of this invocation runs with "
                "runtime.env: " + benv.runtime_env_text(cfg),
                width=77,
                break_on_hyphens=False,
            )
        if resolved:
            head += textwrap.wrap(
                "Resolved from the bench device at the start of this "
                "invocation: " + "; ".join(resolved),
                width=77,
                break_on_hyphens=False,
            )
        with os.fdopen(fd, "w") as f:
            f.write("".join(f"# {line}\n" for line in head))
            yaml.dump(
                resolve_bench_timing(cfg), f, Dumper=_FlowListDumper, sort_keys=False
            )
        (out_dir / f"{name}.source.yaml").write_text(raw)
        return snap
    raise RuntimeError(f"could not allocate a config snapshot name in {out_dir}")


# ── paths and preflight ─────────────────────────────────────────────────────


@dataclass
class Paths:
    bench_root: Path
    build_dir: Path
    bench_binary: Path
    library_dir: Path
    deploy_root: Optional[Path]
    deploy_build_dir: Optional[Path]

    @staticmethod
    def library_file(library_dir: Path, stem: str) -> Optional[Path]:
        for suffix in (".dat", ".dat.zlib", ".yaml"):
            p = library_dir / f"{stem}{suffix}"
            if p.is_file():
                return p
        return None


def _hipblaslt_build(root: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    return root / "projects" / "hipblaslt" / "build" / "release"


def derive_paths(cfg: Mapping[str, Any]) -> Paths:
    root = Path(cfg["runtime"]["rocm_libraries_root"])
    build = _hipblaslt_build(root, cfg_get(cfg, "runtime.build_dir"))
    deploy_root_s = cfg_get(cfg, "deploy.rocm_libraries_to_deploy")
    deploy_root = Path(deploy_root_s) if deploy_root_s else None
    deploy_build = (
        _hipblaslt_build(deploy_root, cfg_get(cfg, "deploy.build_dir"))
        if deploy_root
        else None
    )
    return Paths(
        bench_root=root,
        build_dir=build,
        bench_binary=build / "clients" / "hipblaslt-bench",
        library_dir=build / "Tensile" / "library" / arch_base(cfg["arch"]),
        deploy_root=deploy_root,
        deploy_build_dir=deploy_build,
    )


def _library_errors(library_dir: Path, stem: str, what: str) -> List[str]:
    if not library_dir.is_dir():
        return [
            f"{what} library directory is missing: {library_dir}. Build "
            f"hipBLASLt with {library_dir.name} among its GPU targets first."
        ]
    if Paths.library_file(library_dir, stem) is not None:
        return []
    key = bank_pool.problem_type_key(stem)
    kin = sorted(
        {
            logic_stem(p.name)
            for p in library_dir.glob("TensileLibrary_*Contraction*")
            if is_tensile_contraction_logic(p.name)
            and bank_pool.problem_type_key(logic_stem(p.name)) == key
        }
    )
    if kin:
        return [
            f"the {what} library {stem} has no .dat/.dat.zlib in {library_dir}; "
            f"the libraries of its problem type there: {', '.join(kin[:8])}"
        ]
    return [
        f"the {what} library {stem} is missing: {library_dir} holds no library "
        f"of its problem type at all, so this hipBLASLt build has none to bench "
        f"or serve for {library_dir.name}; build hipBLASLt with a library of "
        f"that type (an Origami library or a bank of kernels) first"
    ]


def _executable_errors(path: Path, what: str) -> List[str]:
    if not path.is_file():
        return [f"{what} not found: {path}"]
    if not os.access(path, os.X_OK):
        return [f"{what} is not executable: {path}"]
    return []


def _tilewright_errors(catalog: str = "v2") -> List[str]:
    try:
        from lib import evaluate

        tilewright = evaluate.tilewright_module()
    except Exception as e:
        return [
            f"the tilewright Python module does not import ({e!r}); install it "
            f"with `pip install shared/tilewright/python`"
        ]
    try:
        from lib import features
    except Exception as e:
        return [f"lib.features does not import: {e!r}"]
    have = getattr(tilewright, "feature_catalog_hash", None)
    if callable(have) and have() != features.feature_names_hash():
        return [
            f"feature catalog hash mismatch: tilewright module "
            f"{have()} vs lib/features.py {features.feature_names_hash()}"
        ]
    if catalog == "v3":
        hashes = getattr(tilewright, "feature_catalog_hashes", None)
        listed = list(hashes()) if callable(hashes) else []
        if features.v3_catalog_hash() not in listed:
            return [
                f"train.catalog v3: the tilewright module does not compute the "
                f"v3 catalog {features.v3_catalog_hash()} (it has {listed}); "
                f"reinstall it with `pip install shared/tilewright/python`"
            ]
    try:
        current = evaluate.module_takes_attributes(tilewright)
    except Exception as e:
        return [f"the tilewright module cannot build kernel configs: {e!r}"]
    if not current:
        return [
            "the tilewright module is older than this checkout's engine (its "
            "Config takes no kernel attributes); reinstall it with "
            "`pip install shared/tilewright/python`"
        ]
    return []


PLANNER_BENCH_KEYS = ("solution_index_list", "skip_screen_prepass")


def _context_config_checks(
    cfg: Mapping[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """contexts: rules beyond the schema; the device CU count is known here
    only from `hardware.n_cu` (preflight checks the rest, and a run resolves
    every cu_fraction when it starts)."""
    n_cu = cfg_get(cfg, "hardware.n_cu")
    errors += ctxlib.context_errors(cfg, n_cu)
    trained = ctxlib.trained_entries(cfg)
    if trained and cfg_get(cfg, "train.catalog") != "v3":
        errors.append(
            "contexts.list: non-exclusive contexts need train.catalog v3 "
            "(v2 models serve only exclusive use and stage05 drops their GEMMs)"
        )
    schedules = cfg_get(cfg, "train.schedules")
    if schedules:
        benched = {str(m.get("schedule") or "default") for m in trained}
        missing = sorted(benched - set(schedules))
        if missing:
            errors.append(
                f"train.schedules {schedules} leaves out the benched "
                f"schedule(s) {missing}"
            )
    if n_cu is not None:
        errors += _min_cu_fraction_errors(cfg, int(n_cu))
    lo = cfg_get(cfg, "train.min_cu_fraction")
    if lo is not None and lo < 1 and not any(ctxlib.reduces_cus(m) for m in trained):
        warnings.append(
            f"train.min_cu_fraction {lo}: no trained context (contexts.list) "
            f"runs on fewer CUs than the device has, so the models would declare "
            f"budgets down to {lo} of the CUs without data from any; train a "
            f"reduced-CU context or leave train.min_cu_fraction unset"
        )
    if cfg_get(cfg, "contexts.replan") and not cfg_get(cfg, "planner.enabled"):
        errors.append("contexts.replan needs planner.enabled")


def _audit_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    st = auditlib.settings(cfg)
    if st["shapes_per_context"] > st["shapes_per_cell"]:
        errors.append(
            f"audit.shapes_per_context ({st['shapes_per_context']}) exceeds "
            f"audit.shapes_per_cell ({st['shapes_per_cell']})"
        )
    if st["seed"] == cfg_get(cfg, "seed.seed"):
        errors.append(
            "audit.seed equals seed.seed: the audit set must be drawn with a "
            "seed of its own"
        )
    devices = cfg_get(cfg, "audit.devices") or []
    if len(set(devices)) != len(devices):
        errors.append("audit.devices: duplicate device indices")
    if cfg_get(cfg, "audit.official") == "sustained" and not cfg_get(
        cfg, "audit.sustained"
    ):
        errors.append(
            "audit.official sustained takes the official exclusive number from "
            "the sustained composite, which audit.sustained false never "
            "measures: set audit.official v2, or audit.sustained true"
        )
    if auditlib.config_dir(cfg) is None or cfg_get(cfg, "audit.build") != "parallel":
        return
    shared = sorted(set(devices) & set(cfg["runtime"]["devices"]))
    if not devices:
        errors.append(
            "audit.build parallel benches the audit set beside the rounds: list "
            "its devices in audit.devices, none of them in runtime.devices"
        )
    elif shared:
        errors.append(
            f"audit.devices {shared} are runtime.devices too: audit.build "
            f"parallel benches the audit set beside the rounds on devices of "
            f"its own"
        )


# train.* keys a final-retrain recipe cannot set: stage05 reads them from the
# config file, and the cells it carries forward must have been trained with
# them.
RECIPE_FIXED_TRAIN_KEYS = ("catalog", "attribute_spec")
# Recipe keys (train keys and stage05 options) of what stage06 deploys every
# bundle with as the run's config sets it (its `deployed_serving`): the final
# retrain keeps the run's, so that its gate scores the bytes stage06 deploys.
RECIPE_DEPLOYED_KEYS = {
    "trained_key_gate": "train.trained_key_gate",
    "no_trained_key_gate": "train.trained_key_gate",
}
# stage05 options the orchestrator derives from the run (its rounds, library
# and device) and from the final retrain itself (every cell).
RECIPE_RESERVED_OPTIONS = frozenset(
    {
        "enriched_csv_dir",
        "prior_round_dir",
        "current_round_dir",
        "output_dir",
        "config_yaml",
        "arch",
        "hardware_device",
        "library_dir",
        "library_stem",
        "weight_dtype",
        "plan",
        "context_plan",
        "off_list_rebench",
        "only_cells",
        "skip_cells",
        "catalog",
        "attribute_spec",
        "help",
    }
)
# stage05 options of the final retrain that only place its inputs, outputs
# and work: a change of them alone does not train it again.
FINAL_RETRAIN_PLACEMENT = frozenset(
    {
        "enriched_csv_dir",
        "prior_round_dir",
        "current_round_dir",
        "output_dir",
        "config_yaml",
        "hardware_device",
        "library_dir",
        "plan",
        "context_plan",
        "off_list_rebench",
        "device",
        "train_workers",
        "train_threads_per_worker",
        "pooled_threads",
        "pooled_teacher_threads",
        "quiet",
    }
)


def _stage05_option_actions() -> Dict[str, argparse.Action]:
    """{dest: action} of stage05's options."""
    return {
        a.dest: a for a in stage05_module().arg_parser()._actions if a.option_strings
    }


def _recipe_argv(action: argparse.Action, value: Any) -> List[str]:
    flag = next(s for s in action.option_strings if s.startswith("--"))
    if action.nargs == 0:
        return [flag] if value else []
    out: List[str] = []
    for v in value if isinstance(value, list) else [value]:
        if v is not None:
            out += [flag, str(v).lower() if isinstance(v, bool) else str(v)]
    return out


def final_retrain_recipe(
    cfg: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """(train.* overrides, further stage05 arguments, errors) of
    `full.final_retrain.recipe`: a key of the train section overrides that
    key for the final retrain only; any other key is a stage05 option
    (`--key-with-dashes`; true gives a flag, false or null leaves it out, a
    list repeats it), such as the options of a training recipe that stage05
    gains."""
    recipe = dict(cfg_get(cfg, "full.final_retrain.recipe") or {})
    train_spec = _spec_at("train")
    assert train_spec.children is not None
    overrides: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    errors: List[str] = []
    for key, value in recipe.items():
        where = f"full.final_retrain.recipe.{key}"
        if key in RECIPE_FIXED_TRAIN_KEYS:
            errors.append(
                f"{where}: the final retrain keeps train.{key} (stage05 reads "
                f"it from the config, and carried cells were trained with it)"
            )
        elif key in RECIPE_DEPLOYED_KEYS:
            errors.append(
                f"{where}: the final retrain keeps {RECIPE_DEPLOYED_KEYS[key]} "
                f"(stage06 deploys every bundle with the run's, and the final "
                f"retrain's gate scores the bundle stage06 deploys)"
            )
        elif key in train_spec.children:
            n = len(errors)
            _check_value(train_spec.children[key], value, where, errors)
            if len(errors) == n:
                overrides[key] = value
        elif key in RECIPE_RESERVED_OPTIONS:
            errors.append(f"{where}: the orchestrator sets this stage05 option")
        else:
            extra[key] = value
    if not extra:
        return overrides, [], errors
    actions = _stage05_option_actions()
    argv: List[str] = []
    for key, value in extra.items():
        where = f"full.final_retrain.recipe.{key}"
        action = actions.get(key)
        if action is None:
            errors.append(
                f"{where}: neither a train key nor an option of stage05 "
                f"(--{key.replace('_', '-')})"
            )
            continue
        if action.nargs == 0 and not isinstance(value, (bool, type(None))):
            errors.append(f"{where}: a flag of stage05 takes true or false")
            continue
        if isinstance(value, list) and not isinstance(action, argparse._AppendAction):
            errors.append(f"{where}: stage05 takes one value for this option")
            continue
        argv += _recipe_argv(action, value)
    if argv and not errors:
        s5 = stage05_module()
        base = ["--enriched-csv-dir", "-", "--output-dir", "-", "--arch", "-"]
        try:
            s5.arg_parser(s5._RaisingParser).parse_known_args(base + argv)
        except ValueError as e:
            errors.append(f"full.final_retrain.recipe: stage05 refuses it: {e}")
    return overrides, argv, errors


def _final_retrain_config_checks(
    cfg: Mapping[str, Any], errors: List[str], warnings: List[str]
) -> None:
    """The recipe, and the train section the final retrain trains with (the
    run's with the recipe's overrides) under the train section's checks,
    those of the run's own section left to it."""
    overrides, _, recipe_errors = final_retrain_recipe(cfg)
    errors += recipe_errors
    if overrides and not recipe_errors:
        base_errors: List[str] = []
        base_warnings: List[str] = []
        _train_config_checks(cfg, base_errors, base_warnings)
        merged = {**cfg, "train": {**dict(cfg.get("train") or {}), **overrides}}
        final_errors: List[str] = []
        final_warnings: List[str] = []
        _train_config_checks(merged, final_errors, final_warnings)
        where = "full.final_retrain.recipe"
        errors += [f"{where}: {e}" for e in final_errors if e not in base_errors]
        warnings += [f"{where}: {w}" for w in final_warnings if w not in base_warnings]
    if not cfg_get(cfg, "full.final_retrain.enabled"):
        return
    if cfg_get(cfg, "full.final_retrain.gate") == "none":
        warnings.append(
            "full.final_retrain.gate is none: the final retrain replaces the "
            "shipped bundle without a comparison on held-out GEMMs"
        )
    elif auditlib.config_dir(cfg) is None:
        errors.append(
            "full.final_retrain.gate not-worse compares the final retrain with "
            "the shipped bundle on the audit set: set audit.dir, or gate: none"
        )


def _min_cu_fraction_errors(cfg: Mapping[str, Any], n_cu: int) -> List[str]:
    lo = cfg_get(cfg, "train.min_cu_fraction")
    if lo is None:
        return []
    try:
        contexts = ctxlib.contexts_from_config(cfg, n_cu)
    except ValueError:
        return []
    covered = ctxlib.trained_fractions(contexts, n_cu)
    if covered[0] < float(lo):
        return [
            f"train.min_cu_fraction {lo} does not cover the trained budget "
            f"fraction {covered[0]:.4g} ({n_cu} CUs); lower it or leave it unset"
        ]
    return []


def _context_preflight(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    if not (ctxlib.trained_entries(cfg) or cfg_get(cfg, "contexts.holdout")):
        return []
    from lib import hardware as hwlib

    try:
        n_cu = hwlib.device_n_cu(cfg, int(cfg["runtime"]["devices"][0]))
    except (OSError, ValueError, RuntimeError) as e:
        return [f"contexts: cannot read the device CU count: {e}"]
    errs = ctxlib.context_errors(cfg, n_cu) + _min_cu_fraction_errors(cfg, n_cu)
    if errs:
        return errs
    contexts = ctxlib.contexts_from_config(cfg, n_cu)
    if not contexts:
        return []
    if any(c.cotenant for c in contexts):
        errs += _cotenant_errors(context_cotenant(cfg, paths))
    lo = ctxlib.trained_fractions(contexts, n_cu)[0]
    for c in contexts:
        if c.holdout and 0 < c.cu_budget < n_cu and c.cu_budget / n_cu < lo:
            ui.warn(
                "contexts",
                f"held-out context {c.label} is below the smallest trained "
                f"budget fraction {lo:.3g}: the model does not declare it",
            )
    return errs


def resolve_contexts(
    cfg: Dict[str, Any], device: int
) -> Tuple[Dict[str, Any], List[str]]:
    """(`cfg` with every contexts entry that gives a cu_fraction resolved
    for the CU count of bench device `device` (lib/contexts.py
    `resolve_fractions`), one line per entry saying what it became).
    ConfigError when the CU count cannot be read or the resolved contexts
    break a rule."""
    if not ctxlib.has_unresolved(cfg):
        return cfg, []
    from lib import hardware as hwlib

    try:
        n_cu = hwlib.device_n_cu(cfg, device)
    except (OSError, ValueError, RuntimeError) as e:
        raise ConfigError(
            [
                f"contexts: a cu_fraction needs the CU count of bench device "
                f"{device}, which cannot be read ({e})"
            ]
        )
    try:
        out, lines = ctxlib.resolve_fractions(cfg, n_cu)
    except ValueError as e:
        raise ConfigError([f"contexts: {e}"])
    errors = ctxlib.context_errors(out, n_cu) + _min_cu_fraction_errors(out, n_cu)
    if errors:
        raise ConfigError(errors)
    return out, lines


def resolve_num_xcd(
    cfg: Dict[str, Any], device: int
) -> Tuple[Dict[str, Any], List[str]]:
    """(`cfg` with `hardware.num_xcd` read from bench device `device`'s KFD
    node when `arch` keeps no fixed queue count (lib/hipblaslt_launch.py)
    and the config gives none, and a line saying so); `cfg` itself when the
    count is not needed or cannot be read, which preflight judges."""
    if hllib.queue_count(cfg["arch"]) is not None:
        return cfg, []
    if cfg_get(cfg, "hardware.num_xcd") is not None:
        return cfg, []
    from lib import hardware as hwlib

    num_xcd = hwlib.device_num_xcd(cfg, device)
    if num_xcd is None:
        return cfg, []
    out = copy.deepcopy(cfg)
    out["hardware"] = {**dict(out.get("hardware") or {}), "num_xcd": int(num_xcd)}
    return out, [
        f"hardware.num_xcd {num_xcd}, the XCCs of bench device {device}'s KFD node "
        f"in its full-device partition"
    ]


def resolve_roofline(
    cfg: Dict[str, Any], device: int
) -> Tuple[Dict[str, Any], List[str]]:
    """(`cfg` with the roofline rates of `bench.adaptive_long` the config
    leaves out read from bench device `device` (lib/hardware.py
    `roofline_rates`), a line per rate read or left unknown); `cfg` itself
    when the block is off or gives both. ConfigError when no term of the
    bound is known: neither rate, or no peak and a data type whose size the
    memory term cannot count (lib/bench_yaml.py `roofline_us`)."""
    keys = ("peak_tflops", "bandwidth_gbs")
    given = {k: cfg_get(cfg, f"bench.adaptive_long.{k}") for k in keys}
    if not cfg_get(cfg, "bench.adaptive_long.enabled") or None not in given.values():
        return cfg, []
    from lib import hardware as hwlib

    rates = hwlib.roofline_rates(cfg, device)
    read = {
        "peak_tflops": (rates.peak_tflops, rates.peak_note),
        "bandwidth_gbs": (rates.bandwidth_gbs, rates.bandwidth_note),
    }
    block = dict(cfg_get(cfg, "bench.adaptive_long") or {})
    lines = []
    for key in keys:
        value, note = read[key]
        if given[key] is not None:
            continue
        if value is None:
            lines.append(
                f"bench.adaptive_long.{key} unknown ({note}): its roofline term "
                f"is left out"
            )
            continue
        block[key] = round(float(value), 3)
        lines.append(
            f"bench.adaptive_long.{key} {block[key]:g} (bench device {device}: "
            f"{note})"
        )
    if all(block.get(k) is None for k in keys):
        raise ConfigError(
            [
                f"bench.adaptive_long: neither peak_tflops nor bandwidth_gbs is set "
                f"and bench device {device} gives neither ({rates.peak_note}; "
                f"{rates.bandwidth_note}); set them in the run config"
            ]
        )
    hl = cfg.get("hipblaslt") or {}
    if block.get("peak_tflops") is None and (
        roofline_us((1, 1, 1, 1), hl, None, block.get("bandwidth_gbs")) is None
    ):
        raise ConfigError(
            [
                f"bench.adaptive_long: bench device {device} gives no peak "
                f"({rates.peak_note}), and the memory term cannot count the bytes "
                f"of a hipblaslt data type of unknown size (lib/shapes.py "
                f"BYTES_PER_ELEMENT); set bench.adaptive_long.peak_tflops"
            ]
        )
    out = copy.deepcopy(cfg)
    out["bench"] = {**dict(out.get("bench") or {}), "adaptive_long": block}
    return out, lines


# The correctness gate's directory in a run directory: the output of the last
# gate that succeeded in gate/ (a gate writes into gate.new first), the
# blocklist each invocation applies in blocklists/, its default verdict cache
# in cache/.
CORRECTNESS_DIR = "correctness"


def correctness_gate_dir(run_dir: Path) -> Path:
    return run_dir / CORRECTNESS_DIR / "gate"


def correctness_attempt_dir(run_dir: Path) -> Path:
    return run_dir / CORRECTNESS_DIR / "gate.new"


def invocation_blocklist(run_dir: Path, invocation: str) -> Path:
    return run_dir / CORRECTNESS_DIR / "blocklists" / f"{invocation}.json"


def hazard_canary_dir(run_dir: Path) -> Path:
    return run_dir / hazlib.HAZARDS_DIR / "canary"


def hazard_canary_attempt_dir(run_dir: Path) -> Path:
    return run_dir / hazlib.HAZARDS_DIR / "canary.new"


def canary_needed(cfg: Mapping[str, Any], paths: Paths) -> bool:
    """Whether the invocation runs the hazard canary: `hazards.enabled` and
    a kernel of a hazard class in the pool that no list of
    `correctness.lists` names (lib/hazards.py). A pool that needs the bench
    device to be read (its bank) and cannot be is read from its files; a
    library that cannot be read at all runs no canary, and preflight names
    what is wrong with it."""
    if not cfg_get(cfg, "hazards.enabled"):
        return False
    stem = str(cfg["hipblaslt"]["library_stem"])
    try:
        listed = cxlib.lists_blocklist(cfg)
    except ValueError:
        listed = cxlib.Blocklist()
    try:
        kernels = bank_pool.config_pool(
            cfg,
            paths.library_dir,
            stem,
            int(cfg["runtime"]["devices"][0]),
            apply_blocklist=False,
        ).kernels
    except (OSError, ValueError, RuntimeError, KeyError):
        kernels = []
        for s in [stem, *(cfg_get(cfg, "planner.bank_library_stems") or [])]:
            try:
                kernels += load_library_kernels(paths.library_dir, s)
            except (OSError, ValueError):
                continue
    return any(hazlib.hazard_classes(k) and not listed.blocks(k) for k in kernels)


def resolve_correctness(
    cfg: Dict[str, Any],
    run_dir: Path,
    invocation: str,
    canary: Optional[bool] = None,
) -> Dict[str, Any]:
    """`cfg` with the files of `correctness.lists` that are not shipped
    lists as absolute paths; when the gate runs (`correctness.enabled`),
    `correctness.cache_dir` (default `<run_dir>/correctness/cache`)
    absolute; when the gate or the hazard canary runs (`canary`, default
    `hazards.enabled`; `canary_needed`), `correctness.blocklist` naming the
    blocklist of the invocation `invocation` (`invocation_blocklist`: the
    gate's copy, else the canary's quarantine: `Pipeline.correctness_gate`),
    which every stage then applies (lib/correctness.py), and else none (a
    snapshot's canary blocklist of an earlier invocation, the one a config
    may name with the gate off, `_correctness_config_checks`, is dropped);
    and with the canary on `hazards.cache_dir` (default
    `<run_dir>/hazards/cache`) absolute. An audit directory's config has no
    correctness section of its own (`lib.audit.bench_config`): it gets the
    blocklist alone."""
    sec = cfg.get("correctness")
    enabled = bool(cfg_get(cfg, "hazards.enabled"))
    canary = enabled if canary is None else bool(canary) and enabled
    if not isinstance(sec, dict) and not enabled:
        return cfg
    out = copy.deepcopy(cfg)
    if isinstance(sec, dict):
        sec = dict(sec)
        sec["lists"] = [
            str(e) if cxlib.shipped_list(e) else str(Path(str(e)).resolve())
            for e in sec.get("lists") or []
        ]
        if sec.get("enabled"):
            cache = sec.get("cache_dir") or run_dir / CORRECTNESS_DIR / "cache"
            sec["cache_dir"] = str(Path(cache).resolve())
    else:
        sec = {}
    if sec.get("enabled") or canary:
        sec["blocklist"] = str(invocation_blocklist(run_dir, invocation))
    else:
        sec.pop("blocklist", None)
    out["correctness"] = sec
    if enabled:
        hz = dict(out.get("hazards") or {})
        store = hz.get("cache_dir") or run_dir / hazlib.HAZARDS_DIR / "cache"
        hz["cache_dir"] = str(Path(store).resolve())
        out["hazards"] = hz
    return out


def _cell_gate_config_checks(cfg: Mapping[str, Any], errors: List[str]) -> None:
    """With the per-cell gate on and an audit set, audit.bootstrap must give
    its cell tests enough sign flips to reach gate.alpha."""
    settings = auditlib.cell_gate_settings(cfg)
    if settings is None or auditlib.config_dir(cfg) is None:
        return
    need = auditlib.min_flips(settings["alpha"])
    have = int(auditlib.settings(cfg)["bootstrap"])
    if have < need:
        errors.append(
            f"audit.bootstrap {have}: the per-cell gate's cell tests take it as "
            f"their sign flips and cannot reach gate.alpha {settings['alpha']:g} "
            f"with fewer than {need}; raise it or set gate.per_cell: false"
        )


def _canary_blocklist(cfg: Mapping[str, Any], path: Any) -> bool:
    """Whether the blocklist file `path` holds the hazard canary's entries
    alone, as the orchestrator writes an invocation's when the canary runs
    without the gate (`Pipeline.correctness_gate`), and the canary is on: its
    store holds what those entries say, and the invocation names its own."""
    if not cfg_get(cfg, "hazards.enabled"):
        return False
    try:
        listed = cxlib.read_list(path)
    except ValueError:
        return False
    return all(e.source == hazlib.SOURCE for e in listed.entries)


def _correctness_config_checks(
    cfg: Mapping[str, Any], errors: List[str], mode: Optional[str] = None
) -> None:
    """The lists must be readable, the harness must run the problem type of
    a run whose gate is on, and only the orchestrator names the gate's
    blocklist: one a config names with the gate off (a run's snapshot with
    `enabled` turned off) would be applied, or fail every stage once gone,
    with no gate to answer for it. Let through are the canary's blocklist of
    an invocation without the gate (`_canary_blocklist`, which the next
    invocation replaces with its own) and any blocklist in --mode audit,
    whose config has no correctness section (`lib.audit.bench_config`)."""
    for entry in cfg_get(cfg, "correctness.lists") or []:
        try:
            cxlib.read_list(cxlib.list_path(entry))
        except ValueError as e:
            errors.append(f"correctness.lists: {e}")
    if not cfg_get(cfg, "correctness.enabled"):
        named = cfg_get(cfg, "correctness.blocklist")
        if named and mode != "audit" and not _canary_blocklist(cfg, named):
            errors.append(
                "correctness.blocklist: the orchestrator names the gate's "
                "blocklist of each invocation when correctness.enabled; with "
                "the gate off, give a blocklist file in correctness.lists"
            )
        return
    try:
        cgate.harness_problem(cfg)
    except cgate.GateError as e:
        errors.append(f"correctness.enabled: {e}; give correctness.lists instead")
    harness = cfg_get(cfg, "correctness.harness")
    if harness and not Path(harness).is_file():
        errors.append(f"correctness.harness: {harness} does not exist")
    devices = cfg_get(cfg, "correctness.devices") or []
    if len(set(devices)) != len(devices):
        errors.append("correctness.devices: duplicate device indices")


def _queue_count_errors(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """A v3 run whose pool holds a kernel that can launch with a dynamic
    queue (lib/hipblaslt_launch.py `takes_dynamic_queue`) under the run's
    schedules and budgets needs the queue count of its arch: QUEUE_COUNTS',
    else the device's XCD count (`hardware.num_xcd`, which the run reads
    from a device that runs as one partition when it starts). The pool is
    read whole, blocked kernels too: this runs before the correctness gate
    writes the blocklist the config names."""
    arch = str(cfg["arch"])
    stem = str(cfg["hipblaslt"]["library_stem"])
    if cfg_get(cfg, "train.catalog") != "v3":
        return []
    if hllib.queue_count(arch, cfg_get(cfg, "hardware.num_xcd")) is not None:
        return []
    if Paths.library_file(paths.library_dir, stem) is None:
        return []
    device = int(cfg["runtime"]["devices"][0])
    try:
        pool = bank_pool.config_pool(
            cfg, paths.library_dir, stem, device, apply_blocklist=False
        ).kernels
    except (OSError, ValueError, RuntimeError) as e:
        return [f"cannot read the kernel pool of {stem}: {e}"]
    contexts = (
        ctxlib.contexts_from_config(cfg) if not ctxlib.has_unresolved(cfg) else []
    )
    schedules = {"default", *(c.schedule for c in contexts)}
    schedules |= set(cfg_get(cfg, "train.schedules") or [])
    reduced = any(c.cu_budget > 0 for c in contexts)
    dynamic = [
        k
        for k in pool
        if hllib.takes_dynamic_queue(k, k.get("attributes"), arch, schedules, reduced)
    ]
    if not dynamic:
        return []
    return [
        f"{len(dynamic)} of the {len(pool)} kernels of the pool (e.g. "
        f"{ki.kernel_id(dynamic[0])}) can launch with a dynamic queue, whose "
        f"flag region depends on the device's XCD count, unknown for {arch}: "
        f"set hardware.num_xcd in the run config, or run on bench devices "
        f"that run as one partition ({kfd.FULL_DEVICE_PARTITION}), whose KFD "
        f"topology gives it"
    ]


TDM_CONFLICT_KEYS = {
    "non_temporal_rule": "train.non_temporal_rule",
    "non_temporal_family_rule": "train.non_temporal_family_rule",
    "attribute_spec": "train.attribute_spec",
}


def _tdm_conflict_texts(
    cfg: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> List[str]:
    v3 = cfg_get(cfg, "train.catalog") == "v3"
    nt = cfg_get(cfg, "train.non_temporal_rule")
    rules = {
        "non_temporal_rule": bool(nt) if v3 else nt is not False,
        "non_temporal_family_rule": bool(
            cfg_get(cfg, "train.non_temporal_family_rule")
        ),
    }
    names = [str(a.get("name")) for a in cfg_get(cfg, "train.attribute_spec") or []]
    return [
        f"{TDM_CONFLICT_KEYS[key]}: {why}; "
        + ("leave those attributes out" if key == "attribute_spec" else "set it false")
        for key, why in ki.tdm_conflicts(pool, rules, names if v3 else ()).items()
    ]


def _tdm_identity_errors(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """The serving options the run's models, the final retrain's included,
    cannot have where the tensor data mover loads an operand of the pool's
    kernels (`lib.kernel_identity.tdm_conflicts`): the non-temporal rules,
    on by default in v2, and attributes holding such an operand's temporal
    hint. The pool is read whole, blocked kernels too."""
    stem = str(cfg["hipblaslt"]["library_stem"])
    if Paths.library_file(paths.library_dir, stem) is None:
        return []
    try:
        pool = bank_pool.config_pool(
            cfg,
            paths.library_dir,
            stem,
            int(cfg["runtime"]["devices"][0]),
            apply_blocklist=False,
        ).kernels
    except (OSError, ValueError, RuntimeError) as e:
        return [f"cannot read the kernel pool of {stem}: {e}"]
    errs = _tdm_conflict_texts(cfg, pool)
    overrides, _, recipe_errors = final_retrain_recipe(cfg)
    if overrides and not recipe_errors:
        merged = {**cfg, "train": {**dict(cfg.get("train") or {}), **overrides}}
        errs += [
            f"full.final_retrain.recipe: {e}"
            for e in _tdm_conflict_texts(merged, pool)
            if e not in errs
        ]
    return errs


def _cotenant_errors(launcher: Path) -> List[str]:
    """hipblaslt-cotenant runs the busy kernel beside it, or builds it with
    hipcc from the source beside it; without either it fails on its first
    context bench."""
    errs = _executable_errors(launcher, "hipblaslt-cotenant")
    if errs:
        return errs
    kernel = launcher.parent / "hipblaslt-cotenant-kernel"
    if kernel.is_file() and os.access(kernel, os.X_OK):
        return []
    source = launcher.parent / "hipblaslt-cotenant-kernel.hip"
    search = os.pathsep.join([os.environ.get("PATH", ""), rocm_bin()])
    hipcc = os.environ.get("HIPCC") or shutil.which("hipcc", path=search)
    if source.is_file() and hipcc:
        return []
    return [
        f"contexts: {launcher} has no executable {kernel.name} beside it"
        + ("" if source.is_file() else f" nor {source.name}")
        + ("" if hipcc else " and no hipcc to build it")
        + "; build the bench's hipblaslt-cotenant-kernel target or set "
        "contexts.cotenant_binary"
    ]


def context_cotenant(cfg: Mapping[str, Any], paths: Paths) -> Path:
    explicit = cfg_get(cfg, "contexts.cotenant_binary")
    return (
        Path(explicit) if explicit else paths.bench_binary.parent / "hipblaslt-cotenant"
    )


def _planner_bench_errors(bench_binary: Path) -> List[str]:
    """The bench's yaml schema (hipblaslt_common.yaml next to it) must define
    the keys planned bench lines use."""
    common = bench_binary.parent / "hipblaslt_common.yaml"
    try:
        text = common.read_text()
    except OSError:
        return [f"planner.enabled: cannot read the bench's yaml schema {common}"]
    missing = [k for k in PLANNER_BENCH_KEYS if f"{k}:" not in text]
    if missing:
        return [
            f"planner.enabled: hipblaslt-bench at {bench_binary} does not "
            f"support {missing}; rebuild it from a hipBLASLt with those options"
        ]
    m = re.search(r"solution_index_list:\s*c_int32\*(\d+)", text)
    if m is None or int(m.group(1)) < MAX_BENCH_LIST:
        return [
            f"planner.enabled: hipblaslt-bench at {bench_binary} takes "
            f"{m.group(1) if m else 'an unknown number of'} kernels per "
            f"solution_index_list, fewer than {MAX_BENCH_LIST}"
        ]
    return []


def _pool_errors(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """The kernel pool the config names (lib/library_pool.py) must resolve
    on the first device: with `library.kind`, origami needs the library's
    Prediction table and bok alone a kernel the device accepts; of the
    legacy keys, `planner.bank: auto` must resolve its bank."""
    stem = cfg["hipblaslt"]["library_stem"]
    device = int(cfg["runtime"]["devices"][0])
    try:
        source = lp.pool_source(cfg)
    except ValueError as e:
        return [str(e)]
    if source.legacy:
        if not source.reads_bok or source.stems is not None:
            return []
        try:
            bank_pool.bank_library_stems(cfg, paths.library_dir, stem, device)
        except (OSError, ValueError, RuntimeError) as e:
            return [f"planner.bank: auto: {e}"]
        return []
    try:
        pool = lp.config_pool(
            cfg, paths.library_dir, stem, device, apply_blocklist=False
        )
    except (OSError, ValueError, RuntimeError) as e:
        return [f"{source.text()}: {e}"]
    if source.kinds == ("bok",) and not pool.kernels:
        return [
            f"{source.text()}: no file of the problem type of {stem} in "
            f"{paths.library_dir} holds a kernel of those rows that device "
            f"{device} accepts"
        ]
    return []


def _device_settings_errors(
    cfg: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None
) -> List[str]:
    """The devices of runtime.devices must agree on every setting the GPUs
    of a stage must share (lib/fingerprint.py `MUST_MATCH`), among them the
    CU count, compute partition mode and XCCs: a run reads its device
    values, such as the CU count its cu_fraction budgets resolve against
    and the XCD count, from its first device alone. A device the
    environment (`environ`, default this process's) hides or the KFD
    topology does not list is not compared, nor is a value sysfs does not
    give."""
    from lib import fingerprint

    env = os.environ if environ is None else environ
    by_index = {g.index: g for g in kfd.gpus()}
    rows = []
    for d in cfg["runtime"]["devices"]:
        try:
            gpu = kfd.physical_gpu(int(d), env)
        except kfd.GpuMapError:
            continue
        if gpu in by_index:
            rows.append(fingerprint.gpu_settings(int(d), gpu, by_index[gpu]))
    return [
        f"runtime.devices: {m}"
        for m in fingerprint.mismatches(rows, "the run's devices")
    ]


def _bench_device_errors(
    cfg: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None
) -> List[str]:
    """Every device of runtime.devices must load the library subtree `arch`
    names. For an arch whose subtree depends on the device's ASIC revision
    and the runtime's mode (lib/arch.py), that is read from the KFD node of
    each of the run's devices and the environment its benches get
    (`environ`, default this process's, with runtime.env over it); for any
    other arch nothing is read. A device the environment hides is not
    checked."""
    arch = arch_base(cfg["arch"])
    if not archlib.mode_dependent(arch):
        return []
    env = {**(os.environ if environ is None else environ), **benv.runtime_env(cfg)}
    value = env.get(archlib.STRICT_SWITCH)
    switch = archlib.STRICT_SWITCH + (f"={value}" if value else " unset")
    by_index = {g.index: g for g in kfd.gpus()}
    errs: List[str] = []
    for d in cfg["runtime"]["devices"]:
        try:
            gpu = kfd.physical_gpu(int(d), env)
        except kfd.GpuMapError as e:
            errs.append(f"runtime.devices: {e}")
            continue
        if gpu is None:
            continue
        where = f"runtime.devices: device {d} (GPU {gpu})"
        target, revision = (
            kfd.gpu_isa(by_index[gpu]) if gpu in by_index else (None, None)
        )
        if target is None:
            errs.append(
                f"{where} has no gfx_target_version in the KFD topology: cannot "
                f"tell which library subtree it loads"
            )
            continue
        reported = archlib.reported_arch(target, revision, env)
        loaded = archlib.library_subtree(reported, revision)
        if loaded == arch:
            continue
        mode = (
            f" (ASIC revision {'unknown' if revision is None else revision}, "
            f"{switch})"
            if archlib.mode_dependent(target)
            else ""
        )
        hint = ""
        if (
            archlib.library_subtree(
                archlib.reported_arch(target, revision, {archlib.STRICT_SWITCH: "0"}),
                revision,
            )
            == arch
        ):
            hint = f' (set runtime.env: {{{archlib.STRICT_SWITCH}: "0"}})'
        errs.append(
            f"{where} reports {reported}{mode}, so hipBLASLt loads "
            f"library/{loaded}/ for it, not library/{arch}/: "
            f"{archlib.device_requirement(arch)}{hint}"
        )
    return errs


def _device_value_errors(
    cfg: Mapping[str, Any],
    run_dir: Path,
    record: Optional[Path] = None,
    mode: Optional[str] = None,
) -> List[str]:
    """The first device's KFD and HIP values against each other and the
    run's (lib/device_values.py), recorded in `record` (default
    `<run_dir>/platform.json`; an invocation's is its snapshot's
    `.platform.json`). A difference refuses a run before its first stage
    completes; one that already holds data (its pin.json) was made with the
    values it uses, which only a new run can change, and an audit directory
    serves runs that check their own device: there it is a warning, unless
    the device does not answer."""
    c = dvlib.check(
        cfg,
        run_dir / "tools",
        float(cfg_get(cfg, "runtime.gpu_lease_wait_s")),
        run_dir,
    )
    for w in c.warnings:
        ui.warn("device", w)
    if c.skipped:
        ui.info("device", f"device values not compared: {c.skipped}")
    dvlib.write_platform(record or run_dir / fingerprint.PLATFORM_FILE, c)
    held = mode == "audit" or pinning.read_run_pin(run_dir) is not None
    if c.errors and held and not c.no_answer:
        why = "an audit set" if mode == "audit" else "a run that already holds data"
        for e in c.errors:
            ui.warn("device", f"{e} (not refused for {why})")
        return []
    return c.errors


def _bench_preflight(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """The bench, its library and devices, contexts, planner and kernel pool
    of a config whose stages bench."""
    stem = cfg["hipblaslt"]["library_stem"]
    library = _library_errors(paths.library_dir, stem, "bench")
    errs = _executable_errors(paths.bench_binary, "hipblaslt-bench") + library
    errs += _bench_device_errors(cfg)
    errs += _device_settings_errors(cfg)
    errs += _context_preflight(cfg, paths)
    if cfg_get(cfg, "planner.enabled"):
        errs += _planner_bench_errors(paths.bench_binary)
        for bank in cfg_get(cfg, "planner.bank_library_stems") or []:
            errs += _library_errors(paths.library_dir, bank, "planner bank")
    if not library:
        errs += _pool_errors(cfg, paths)
    return errs


def audit_devices(cfg: Mapping[str, Any]) -> List[int]:
    """The devices the run's audit set benches on: audit.devices, else
    runtime.devices."""
    return [int(d) for d in cfg_get(cfg, "audit.devices") or cfg["runtime"]["devices"]]


def config_pool_digests(cfg: Mapping[str, Any], paths: Paths) -> Dict[int, str]:
    """{solution: kernel digest} (lib/audit.py) of the kernel pool the
    config's library and bank give now, blocked kernels too: the audit set
    benches every kernel (`lib.audit.bench_config`) but those the hazard
    canary keeps out, which never run among others (lib/hazards.py
    `quarantined_names`), and its checks run before the correctness gate
    writes the invocation's blocklist; OSError, ValueError, RuntimeError or
    KeyError when it cannot be read."""
    pool = bank_pool.config_pool(
        cfg,
        paths.library_dir,
        cfg["hipblaslt"]["library_stem"],
        int(cfg["runtime"]["devices"][0]),
        apply_blocklist=False,
    )
    return auditlib.pool_digests(hazlib.without_quarantined(cfg, pool.kernels))


def audit_status(cfg: Mapping[str, Any], paths: Paths) -> Tuple[str, Optional[str]]:
    """`lib.audit.status` of the config's audit set for the kernel pool its
    library gives now; ("error", why) when that pool cannot be read."""
    try:
        current = config_pool_digests(cfg, paths)
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        return "error", f"cannot read the kernel pool in {paths.library_dir}: {e}"
    return auditlib.status(auditlib.config_dir(cfg), current)


def audit_build_refusal(audit_dir: Path, current: Mapping[int, str]) -> Optional[str]:
    """Why `--mode audit` refuses to go on building the audit set in
    `audit_dir` with the kernel pool `current` ({solution: kernel digest}):
    its first generation, or the increment it would bench next, was planned
    with another hipBLASLt build, whose solution indices name other kernels
    now (`AuditRun.run`, `AuditRun.extend`); None when it does not."""
    m = auditlib.manifest(audit_dir)
    if m is None:
        return None
    if not m.get("generations"):
        planned = m.get("planned_kernels") or {}
        if not stage_complete(run_layout.plan_dir(audit_dir)):
            return None
        if planned and changed_kernels(planned, current):
            return "its shapes were planned with another hipBLASLt build"
        return None
    if not auditlib.unbenched(m, current):
        return None
    k = len(m["generations"])
    path = Path(audit_dir) / "increments" / f"inc_{k:03d}" / "increment.json"
    rec = _read_json(path)
    if isinstance(rec, dict) and changed_kernels(rec.get("kernels") or {}, current):
        return f"its increment {k} ({path}) was planned with another hipBLASLt build"
    return None


def audit_location_errors(audit_dir: Path) -> List[str]:
    """An audit directory never holds training rounds nor lies inside a
    training run."""
    d = Path(audit_dir).resolve()
    errs: List[str] = []
    if round_dirs(d):
        errs.append(f"{d} holds training rounds; give the audit its own directory")
    for parent in d.parents:
        if (
            (parent / "configs").is_dir()
            and (parent / "orchestrator.log").is_file()
            and not (parent / auditlib.MANIFEST).is_file()
        ):
            errs.append(f"{d} lies inside the training run {parent}")
            break
    return errs


def audit_binding(
    train_cfg: Mapping[str, Any], audit_cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """What an audit directory is bound to (`lib.audit.BOUND_KEYS`, arch,
    library stem and contexts), for the audit config `audit_cfg`
    (`lib.audit.bench_config`) of `train_cfg`."""
    st = auditlib.settings(train_cfg)
    return {
        **{k: st[k] for k in auditlib.BOUND_KEYS},
        "arch": str(audit_cfg["arch"]),
        "library_stem": str(audit_cfg["hipblaslt"]["library_stem"]),
        "contexts": [c.to_dict() for c in ctxlib.contexts_from_config(audit_cfg)],
    }


def audit_binding_errors(
    audit_dir: Path, want: Mapping[str, Any], m: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Errors when the manifest `m` (default the one in `audit_dir`) was
    built with another binding than `want` (`audit_binding`)."""
    if m is None:
        m = auditlib.manifest(audit_dir)
        if m is None:
            return []
    if m.get("schema") != auditlib.SCHEMA_VERSION:
        return [f"{Path(audit_dir) / auditlib.MANIFEST} is not an audit manifest"]
    have = dict(m.get("binding") or {})
    return [
        f"{audit_dir} was built with {k}={have.get(k)!r}; this config has "
        f"{want[k]!r} (use another --run-dir for other audit settings)"
        for k in want
        if have.get(k) != want[k]
    ]


def _audit_build_errors(
    cfg: Mapping[str, Any], paths: Paths, known: Sequence[str] = ()
) -> List[str]:
    """What keeps a training run from building the audit set it is
    incomplete for (audit.build before or parallel): the audit directory's
    place, the settings it was built with, a plan of another hipBLASLt
    build (`audit_build_refusal`), another builder holding it, and the
    bench preflight of its config (`lib.audit.bench_config`); errors
    already in `known` are not repeated. With audit.build off, a final
    retrain gated on the audit set waits for it at the end of the run: a
    warning."""
    d = auditlib.config_dir(cfg)
    if d is None:
        return []
    st, why = audit_status(cfg, paths)
    if st == "ok" or (st == "error" and known):
        return []
    if st == "error":
        return [f"audit.dir: {why}"]
    if cfg_get(cfg, "audit.build") == "off":
        if cfg_get(cfg, "full.final_retrain.enabled") and (
            cfg_get(cfg, "full.final_retrain.gate") == "not-worse"
        ):
            ui.warn(
                "preflight",
                f"the audit set {d} is not complete ({why}) and audit.build is "
                f"off: full.final_retrain.gate not-worse decides once --mode "
                f"audit completes it, and the run exits 1 until then",
            )
        return []
    acfg = auditlib.bench_config(cfg, audit_devices(cfg))
    errs = audit_location_errors(d)
    errs += audit_binding_errors(d, audit_binding(cfg, acfg))
    try:
        refusal = audit_build_refusal(d, config_pool_digests(cfg, paths))
    except (OSError, ValueError, RuntimeError, KeyError):
        refusal = None
    if refusal:
        errs.append(
            f"the audit set {d} cannot be built further with this hipBLASLt "
            f"build: {refusal}. Bench it with that build, point audit.dir at a "
            f"new directory, or set audit.build: off"
        )
    if d.is_dir():
        fd = lock_run_dir(d)
        if fd is None:
            errs.append(
                f"another process is building the audit set {d} (it holds "
                f"{d / LOCK_NAME}); wait for it, or set audit.build: off"
            )
        else:
            os.close(fd)
    errs += [
        f"audit set: {e}"
        for e in _bench_preflight(acfg, derive_paths(acfg))
        if e not in known
    ]
    return errs


@dataclass(frozen=True)
class Stage07Build:
    """The hipBLASLt build stage 7 runs: with `library` (stage07.build_dir)
    it serves the staged model from a library directory beside the bench
    library, else the model co-located in the deploy build's library."""

    build_dir: Path
    library: bool

    @property
    def bench_binary(self) -> Path:
        return self.build_dir / "clients" / "hipblaslt-bench"

    @property
    def what(self) -> str:
        return "stage07.build_dir" if self.library else "the deploy build"


def stage07_build(cfg: Mapping[str, Any], paths: Paths) -> Optional[Stage07Build]:
    explicit = cfg_get(cfg, "stage07.build_dir")
    if explicit:
        return Stage07Build(Path(str(explicit)), True)
    if paths.deploy_build_dir is not None:
        return Stage07Build(paths.deploy_build_dir, False)
    return None


# Files of a hipBLASLt build that hold TensileLite's host code: its shared
# TensileLite library, else hipBLASLt's own library (a static TensileLite).
TENSILELITE_LIBRARIES = (
    "tensilelite/libtensilelite-host.so*",
    "library/libhipblaslt.so*",
)


def tensilelite_libraries(build_dir: Path) -> List[Path]:
    for pattern in TENSILELITE_LIBRARIES:
        found = {p.resolve() for p in Path(build_dir).glob(pattern) if p.is_file()}
        if found:
            return sorted(found)
    return []


def build_libraries(build_dir: Path) -> List[List[Any]]:
    """[name, size, mtime] of the libraries a build's bench loads (its
    TensileLite and hipBLASLt), which change when the build is rebuilt."""
    out = []
    for pattern in TENSILELITE_LIBRARIES:
        for p in sorted({p.resolve() for p in Path(build_dir).glob(pattern)}):
            if p.is_file():
                st = p.stat()
                out.append([p.name, st.st_size, st.st_mtime_ns])
    return out


def _holds(path: Path, marker: bytes) -> bool:
    try:
        with path.open("rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as m:
            return m.find(marker) >= 0
    except (OSError, ValueError):
        return False


def stage07_capabilities(
    cfg: Mapping[str, Any], blocks_pool: bool = False
) -> Dict[str, str]:
    """{string a stage-7 TensileLite must hold: what it lacks without it}:
    the loader of the format train.catalog trains, with a bank the setting
    that serves it and, when the run leaves kernels of the pool out
    (`blocks_pool`), the reader of the blocklist sidecar stage06 ships."""
    catalog = str(cfg_get(cfg, "train.catalog") or "v2")
    out = {
        f"MLREC_{catalog}": f"cannot load MLREC_{catalog} models (train.catalog "
        f"{catalog})"
    }
    if bank_pool.bank_source(cfg).enabled:
        out[TILEWRIGHT_BANK_VAR] = (
            "does not serve a bank of kernels: the runtime would rank the "
            "Prediction table alone while the model ranks the bank too"
        )
    if blocks_pool:
        out[cxlib.SIDECAR_SUFFIX] = (
            "does not read the blocklist sidecar of a model: the runtime would "
            "rank the kernels the run leaves out of the pool"
        )
    return out


def blocklist_reaches_pool(cfg: Mapping[str, Any], paths: Paths) -> bool:
    """Whether the runtime has to leave kernels of the pool out for the run:
    with the correctness gate on (it runs after the preflight and can block
    any kernel), with the hazard canary to run (`canary_needed`: it runs
    after the preflight too and can quarantine the pool's kernels of the
    hazard classes), or when the config's lists name a kernel of the pool
    (or the pool cannot be read to tell)."""
    if cfg_get(cfg, "correctness.enabled"):
        return True
    try:
        listed = cxlib.lists_blocklist(cfg)
    except ValueError:
        return True
    if canary_needed(cfg, paths):
        return True
    if not listed:
        return False
    try:
        pool = bank_pool.config_pool(
            cfg,
            paths.library_dir,
            str(cfg["hipblaslt"]["library_stem"]),
            int(cfg["runtime"]["devices"][0]),
            apply_blocklist=False,
        )
    except (OSError, ValueError, RuntimeError, KeyError):
        return True
    return any(listed.blocks(k) for k in pool.kernels)


def stage07_build_errors(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """The build stage 7 runs: its bench, the library the deploy build
    serves the model from, and its TensileLite's capabilities
    (`stage07_capabilities`, `blocklist_reaches_pool`)."""
    build = stage07_build(cfg, paths)
    if build is None:
        return [
            "stage07 has no hipBLASLt build to run: set stage07.build_dir or "
            "deploy.rocm_libraries_to_deploy"
        ]
    errs = _executable_errors(build.bench_binary, f"{build.what} hipblaslt-bench")
    cpus = fingerprint.parse_cpulist(str(cfg_get(cfg, "stage07.cpus") or "")) or []
    absent = [c for c in cpus if c >= (os.cpu_count() or 0)]
    if absent:
        errs.append(
            f"stage07.cpus names CPUs this host does not have: {absent} "
            f"({os.cpu_count()} CPUs)"
        )
    if not build.library:
        errs += _library_errors(
            build.build_dir / "Tensile" / "library" / arch_base(cfg["arch"]),
            cfg["hipblaslt"]["library_stem"],
            "deploy",
        )
    libs = tensilelite_libraries(build.build_dir)
    if not libs:
        return errs + [
            f"{build.what} {build.build_dir} holds no TensileLite library "
            f"({' or '.join(TENSILELITE_LIBRARIES)})"
        ]
    wanted = stage07_capabilities(cfg, blocklist_reaches_pool(cfg, paths))
    for marker, lacks in wanted.items():
        if not any(_holds(lib, marker.encode()) for lib in libs):
            errs.append(
                f"{build.what} {build.build_dir}: its TensileLite "
                f"({', '.join(lib.name for lib in libs)}) {lacks}; no {marker!r} "
                f"in it"
            )
    return errs


DATASET_TYPE_FIELDS = (
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "transA",
    "transB",
    "scaleA",
    "scaleB",
)


def _type_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if name == "compute_type" and text.lower().startswith("c_"):
        text = text[2:]
    return text


# Problems of a stage08 dataset preflight reads at most to find one of the
# config's data types and layout.
DATASET_SCAN = 1000


def _yaml_problems(path: Path) -> Iterator[Mapping[str, Any]]:
    """The entries of a bench yaml: its `- {...}` lines one at a time, as
    benches are written, or the whole file when it has none."""
    flow = False
    with path.open() as f:
        for line in f:
            text = line.strip()
            if text.startswith("- {"):
                flow = True
                entry = yaml.safe_load(text[2:])
                if isinstance(entry, dict):
                    yield entry
    if not flow:
        entries = yaml.safe_load(path.read_text()) or []
        for entry in [entries] if isinstance(entries, dict) else entries:
            if isinstance(entry, dict):
                yield entry


def _csv_rows(chunks: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    for chunk in chunks:
        with chunk.open(newline="") as f:
            yield from csv.DictReader(f)


def _log_rows(logs: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    """The first measured row of every problem of the logs, in file order."""
    from lib.bench_log import iter_problems

    for log in logs:
        with log.open(errors="replace") as f:
            for prob in iter_problems(f):
                if prob.tested:
                    yield prob.tested[0].fields


def _dataset_problems(path: Path) -> Tuple[str, Iterator[Mapping[str, Any]]]:
    """(kind, its problems, read as they are consumed) of a stage08 dataset:
    a bench yaml's entries, the rows of an enriched directory's chunk_*.csv
    files or the problems of a log directory's block_*.log files, file
    after file; kind `missing` or `unknown` without problems otherwise."""
    if not path.exists():
        return "missing", iter(())
    if path.is_file():
        if path.suffix.lower() not in (".yaml", ".yml"):
            return "unknown", iter(())
        return "yaml", _yaml_problems(path)
    chunks = sorted(path.glob("chunk_*.csv"))
    if chunks:
        return "enriched", _csv_rows(chunks)
    logs = sorted(path.glob("block_*.log"))
    if logs:
        return "logs", _log_rows(logs)
    return "unknown", iter(())


def _dataset_errors(cfg: Mapping[str, Any]) -> List[str]:
    """Every stage08 dataset exists, as a bench yaml or a directory of
    block_*.log or chunk_*.csv files, and one of its first DATASET_SCAN
    problems has the config's data types and layout."""
    want = {
        f: _type_text(f, cfg["hipblaslt"][f])
        for f in DATASET_TYPE_FIELDS
        if cfg["hipblaslt"].get(f) is not None
    }

    def matches(r: Mapping[str, Any]) -> bool:
        return all(
            r.get(f) in (None, "") or _type_text(f, r[f]) == v for f, v in want.items()
        )

    errs: List[str] = []
    for ds in cfg_get(cfg, "stage08.datasets"):
        where = f"stage08.datasets {ds['name']}"
        path = Path(str(ds["path"]))
        first: Optional[Mapping[str, Any]] = None
        found = False
        try:
            kind, problems = _dataset_problems(path)
            for r in itertools.islice(problems, DATASET_SCAN):
                first = r if first is None else first
                if matches(r):
                    found = True
                    break
        except (OSError, ValueError, yaml.YAMLError, csv.Error) as e:
            errs.append(f"{where}: cannot read {path}: {e}")
            continue
        if kind == "missing":
            errs.append(f"{where}: {path} does not exist")
        elif kind == "unknown":
            errs.append(
                f"{where}: {path} is neither a bench yaml nor a directory of "
                f"block_*.log or chunk_*.csv files"
            )
        elif not found:
            seen = ", ".join(
                f"{f}={first[f]}"
                for f, v in want.items()
                if first is not None
                and first.get(f) not in (None, "")
                and _type_text(f, first[f]) != v
            )
            errs.append(
                f"{where}: none of the first {DATASET_SCAN} problems of {path} has "
                f"the config's data types and layout "
                f"({', '.join(f'{f}={v}' for f, v in want.items())})"
                + (f"; its first one has {seen}" if seen else "")
            )
    return errs


def _audit_preflight(cfg: Mapping[str, Any], paths: Paths) -> List[str]:
    """With audit.build off, audit.dir names an audit set: an error without
    one, since the run builds none. With before or parallel the run builds
    or extends the set itself (`AuditBuild`; `_audit_build_errors` checks
    that it can). A set that is still benched or misses kernels of the pool
    is reported when the run starts (`AuditBuild.prepare`)."""
    d = auditlib.config_dir(cfg)
    if d is None or cfg_get(cfg, "audit.build") != "off":
        return []
    if auditlib.manifest(d) is None:
        return [
            f"audit.dir {d} holds no audit set ({auditlib.MANIFEST}) and "
            f"audit.build is off: build it first (--mode audit --run-dir {d} "
            f"with this config), let the run build it (audit.build: before or "
            f"parallel), or unset audit.dir"
        ]
    return []


def round_bytes(rd: Path) -> int:
    """Bytes of the files under `rd` (links not followed)."""
    total = 0
    for root, _dirs, files in os.walk(rd):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def round_size(rd: Path) -> int:
    """Bytes of a complete round: what its round marker records, else
    measured."""
    recorded = (_read_json(rd / ROUND_MARKER) or {}).get("bytes")
    if isinstance(recorded, int) and not isinstance(recorded, bool):
        return recorded
    return round_bytes(rd)


def round_estimate(run_dir: Path) -> Optional[Tuple[Path, int, bool]]:
    """(round, its size, whether it is an active round) the next round's size
    is judged by: the largest complete active round, else the latest
    complete round, an initial one that also holds the seed benches; None
    before a round completed."""
    done = complete_prefix(run_dir)
    active = [
        rd
        for rd in done
        if (_read_json(rd / ROUND_MARKER) or {}).get("mode") == "active"
    ]
    if active:
        sizes = [(round_size(rd), rd) for rd in active]
        size, rd = max(sizes, key=lambda t: t[0])
        return rd, size, True
    if done:
        return done[-1], round_size(done[-1]), False
    return None


def disk_problems(
    run_dir: Path, rounds_left: int, min_free_gb: Optional[float] = None
) -> Tuple[List[str], List[str]]:
    """(errors, warnings) of the free space of the run directory's file
    system for `rounds_left` more rounds. With `min_free_gb`
    (runtime.min_free_gb; 0: no check) an error below that much; else judged
    by `round_estimate`: an error when an active round's size does not fit,
    a warning when an initial round's does not (it also holds the seed
    benches) or when `rounds_left` such rounds do not. Nothing before a round
    completed."""
    if rounds_left < 1 or min_free_gb == 0:
        return [], []

    def where(free: int) -> str:
        return f"{free / 1e9:.1f} GB free on the file system of {run_dir}"

    if min_free_gb is not None:
        free = shutil.disk_usage(run_dir).free
        if free < float(min_free_gb) * 1e9:
            return [
                f"{where(free)}, runtime.min_free_gb asks for {float(min_free_gb):g} "
                f"GB before each round: free space, then run the same command"
            ], []
        return [], []
    est = round_estimate(run_dir)
    if est is None:
        return [], []
    rd, size, active = est
    free = shutil.disk_usage(run_dir).free
    setting = "runtime.min_free_gb sets the space a round needs; 0: no check"
    text = (
        f"{where(free)}; its "
        f"{'largest complete active' if active else 'latest complete'} round, "
        f"{rd.name}, takes {size / 1e9:.1f} GB"
    )
    if free < size and active:
        return [
            f"{text}: free space before the next round, then run the same "
            f"command ({setting})"
        ], []
    if free < size:
        return [], [
            f"{text}, more than an active round when it holds the seed benches, so "
            f"the next round may not fit ({setting})"
        ]
    if free < size * rounds_left:
        return [], [f"{text}, so {rounds_left} more rounds may not fit"]
    return [], []


def preflight(
    cfg: Mapping[str, Any],
    paths: Paths,
    mode: str,
    run_dir: Optional[Path] = None,
    record: Optional[Path] = None,
) -> List[str]:
    """What keeps the invocation in `mode` from starting. With `run_dir`, a
    mode that benches also compares the first device's KFD and HIP values
    (`_device_value_errors`), recorded in `record`."""
    stem = cfg["hipblaslt"]["library_stem"]
    errs: List[str] = _queue_count_errors(cfg, paths)
    if mode != "validate":
        errs += _bench_preflight(cfg, paths)
        if run_dir is not None:
            errs += _device_value_errors(cfg, run_dir, record, mode)
        if mode != "audit":
            errs += _audit_build_errors(cfg, paths, errs)
            errs += _tdm_identity_errors(cfg, paths)
        if cfg_get(cfg, "deploy.apply_patch"):
            root = paths.deploy_root
            if root is None or not (root / "shared" / "tilewright").is_dir():
                errs.append(
                    f"deploy.rocm_libraries_to_deploy is not a rocm-libraries "
                    f"checkout with shared/tilewright: {root}"
                )
            if cfg_get(cfg, "deploy.build_after_apply"):
                bd = paths.deploy_build_dir
                if bd is None or not (bd / "CMakeCache.txt").is_file():
                    errs.append(f"deploy build directory is not configured: {bd}")
    else:
        rebench = stage08_enabled(cfg) and bool(cfg_get(cfg, "validate.rebench_misses"))
        if stage08_enabled(cfg):
            errs += _library_errors(paths.library_dir, stem, "bench")
            if rebench:
                errs += _executable_errors(paths.bench_binary, "hipblaslt-bench")
        if stage07_enabled(cfg) or rebench:
            errs += _bench_device_errors(cfg)
            errs += _device_settings_errors(cfg)
            if run_dir is not None:
                errs += _device_value_errors(cfg, run_dir, record, mode)
    if mode != "audit":
        if stage07_enabled(cfg):
            errs += stage07_build_errors(cfg, paths)
        errs += _dataset_errors(cfg)
    if mode == "full":
        errs += _audit_preflight(cfg, paths)
        if run_dir is not None:
            left = int(cfg_get(cfg, "full.total_rounds")) - len(
                complete_prefix(run_dir)
            )
            disk_errs, disk_warns = disk_problems(
                run_dir, left, cfg_get(cfg, "runtime.min_free_gb")
            )
            errs += disk_errs
            for w in disk_warns:
                ui.warn("disk", w)
    errs += _tilewright_errors(str(cfg_get(cfg, "train.catalog") or "v2"))
    return list(dict.fromkeys(errs))


def print_pool(cfg: Mapping[str, Any], paths: Paths) -> None:
    """Print what the kernel pool the config names is made of on its first
    device (`lib.library_pool.describe`), blocked kernels included: the
    correctness gate, which can block any of them, runs after this."""
    stem = cfg["hipblaslt"]["library_stem"]
    try:
        pool = lp.config_pool(
            cfg,
            paths.library_dir,
            stem,
            int(cfg["runtime"]["devices"][0]),
            apply_blocklist=False,
        )
    except (OSError, ValueError, RuntimeError) as e:
        ui.warn("pool", f"cannot read the kernel pool of {stem}: {e}")
        return
    for level, line in lp.describe(pool):
        (ui.warn if level == "warn" else ui.info)("pool", line)


# ── subprocess + logging ────────────────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_ORIG_STDOUT: Optional[Any] = None


class _Tee:
    """Writes to the terminal and, without ANSI escapes, to a log file."""

    def __init__(self, terminal: Any, logfile: Any) -> None:
        self.terminal = terminal
        self.logfile = logfile

    def write(self, data: str) -> int:
        try:
            self.terminal.write(data)
        except (OSError, ValueError):
            pass
        try:
            self.logfile.write(_ANSI_RE.sub("", data))
        except ValueError:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except (OSError, ValueError):
            pass
        try:
            self.logfile.flush()
        except ValueError:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def _signal_stage(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop_stage(proc: subprocess.Popen, grace_s: float = 30.0) -> None:
    """SIGINT the stage's process group, SIGKILL it after `grace_s`."""
    _signal_stage(proc, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _signal_stage(proc, signal.SIGKILL)
        proc.wait()


def stop_leftovers(token: str, label: str) -> None:
    """Stop the processes of an ended stage attempt that still run (benches
    of a stage killed before it could stop them, for one)."""
    found, left = attempts.stop(token)
    if found:
        ui.warn("stage", f"stopped {len(found)} process(es) {label} left running")
    if left:
        ui.err("stage", f"processes {left} of {label} do not stop")


# Seconds the output of a stage that ended may still take to drain.
COPY_JOIN_S = 10.0


def _copy_lines(stream: Any, term: Any, logf: Any) -> None:
    """Copy a stage's output to the terminal and its log until EOF."""
    try:
        for line in stream:
            try:
                term.write(line)
                term.flush()
            except (OSError, ValueError):
                pass
            logf.write(line)
    except (OSError, ValueError):
        pass


def run_logged(
    cmd: List[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    *,
    running: Optional[attempts.Running] = None,
    label: str = "",
    pass_fds: Sequence[int] = (),
) -> int:
    """Run `cmd` in its own process group, copying its output to `log_path`
    and the terminal (not to orchestrator.log). Returns the exit code
    (negative for a signal) when it exits. An interrupt stops the stage
    (`stop_stage`) and is re-raised.

    The stage gets SIGTERM when this process dies, even by SIGKILL
    (`procs.die_with_parent`), and a fresh attempt token in its environment
    (lib/attempts.py), which `running`, the run directory's record, names
    while it runs. When it ends, every process still carrying the token is
    stopped. `pass_fds` stay open in the stage: GPU lease descriptors, so
    that a lease lasts as long as the stage."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = label or log_path.stem
    ui.info("run", " ".join(str(c) for c in cmd))
    t0 = time.time()
    term = _ORIG_STDOUT if _ORIG_STDOUT is not None else sys.stdout
    token = attempts.new_token()
    with log_path.open("w", buffering=1, errors="replace") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            errors="replace",
            env=attempts.environ_with(token, env),
            start_new_session=True,
            preexec_fn=procs.die_with_parent(),
            pass_fds=tuple(pass_fds),
        )
        # A process the stage leaves behind can hold the pipe open after the
        # stage exits, so the stage's end is its exit, not the pipe's EOF.
        copier = threading.Thread(
            target=_copy_lines, args=(proc.stdout, term, logf), daemon=True
        )
        copier.start()
        try:
            if running is not None:
                running.add(token, pid=proc.pid, cmd=cmd, label=label)
            rc = proc.wait()
        except KeyboardInterrupt:
            stop_stage(proc)
            raise
        finally:
            if proc.poll() is None:
                stop_stage(proc)
            stop_leftovers(token, label)
            copier.join(timeout=COPY_JOIN_S)
            if running is not None:
                running.remove(token)
    el = time.time() - t0
    if rc == 0:
        ui.ok("run", f"completed in {ui.fmt_dur(el)}")
    else:
        ui.err("run", f"FAILED (rc={rc}) after {ui.fmt_dur(el)}; see {log_path}")
    return rc


@dataclass
class Started:
    """A command `start_logged` started: its process and attempt token."""

    proc: subprocess.Popen
    token: str
    label: str
    running: Optional[attempts.Running] = None

    def end(self) -> None:
        """After the command ended: stop what still carries its token and
        drop its entry in the run directory's record."""
        stop_leftovers(self.token, self.label)
        if self.running is not None:
            self.running.remove(self.token)


def start_logged(
    cmd: List[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    *,
    running: Optional[attempts.Running] = None,
    label: str = "",
) -> Started:
    """Start `cmd` in its own process group with its output appended to
    `log_path` (not to the terminal), for the caller to wait for or stop
    (`stop_stage`) and then `end`. Like a stage of `run_logged` it gets
    SIGTERM when this process dies, even by SIGKILL, and a fresh attempt
    token in its environment, which `running` names while it runs, so that
    the next invocation stops it when this one died without doing so."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = label or log_path.stem
    ui.info("run", " ".join(str(c) for c in cmd) + f"  (output: {log_path})")
    token = attempts.new_token()
    with log_path.open("a") as out:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=attempts.environ_with(token, env),
            start_new_session=True,
            preexec_fn=procs.die_with_parent(),
        )
    if running is not None:
        running.add(token, pid=proc.pid, cmd=cmd, label=label)
    return Started(proc, token, label, running)


def exit_code(rc: int) -> int:
    """Shell-style status for a child's return code (signal N -> 128+N)."""
    return 128 - rc if rc < 0 else rc


class Timing:
    """Per-(round, stage) wall-clock ledger of one invocation."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.t0 = time.time()

    def add(
        self,
        round_idx: Optional[int],
        stage: str,
        rc: int,
        elapsed: float,
        attempt: int = 1,
    ) -> None:
        row = {"round": round_idx, "stage": stage, "rc": rc, "elapsed_s": elapsed}
        if attempt > 1:
            row["attempt"] = attempt
        self.rows.append(row)

    def total(self) -> float:
        return time.time() - self.t0

    def print_summary(self) -> None:
        if not self.rows:
            return
        ui.banner("timing summary", ui.C.HEAD)
        width = max(12, *(len(r["stage"]) for r in self.rows))
        current: Any = object()
        for r in self.rows:
            if r["round"] != current:
                current = r["round"]
                ui.info("timing", "run" if current is None else f"round_{current}")
            tag = "ok " if r["rc"] == 0 else f"rc={r['rc']}"
            if r.get("attempt"):
                tag += f", attempt {r['attempt']}"
            print(
                f"   {r['stage']:<{width}}  {ui.fmt_dur(r['elapsed_s']):>8}  [{tag}]",
                flush=True,
            )
        ui.ok("timing", f"wall-clock total: {ui.fmt_dur(self.total())}")

    def write(self, run_dir: Path, mode: str) -> None:
        out = run_dir / "timing"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"{stamp}_{mode}.json"
        n = 1
        while path.exists():
            path = out / f"{stamp}_{mode}_{n}.json"
            n += 1
        path.write_text(
            json.dumps(
                {"mode": mode, "total_s": self.total(), "rows": self.rows}, indent=2
            )
            + "\n"
        )


# ── stage directories ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageDef:
    number: float
    key: str
    script: str
    title: str


STAGE_DEFS: Dict[str, StageDef] = {
    s.key: s
    for s in (
        StageDef(0.0, "stage00a", "stage00a_hazard_canary.py", "hazard canary"),
        StageDef(0.0, "stage00", "stage00_correctness_gate.py", "correctness gate"),
        StageDef(1.0, "stage01", "stage01_generate_OOB_shapes.py", "generate shapes"),
        StageDef(2.0, "stage02", "stage02_plan_benchmarks.py", "plan benchmarks"),
        StageDef(3.0, "stage03", "stage03_load_balance_offline_tuning.py", "bench"),
        StageDef(
            3.2, "stage03b", "stage03b_rebench_near_ties.py", "re-read near-winners"
        ),
        StageDef(3.5, "stage03c", "stage03c_bench_contexts.py", "bench contexts"),
        StageDef(
            3.7,
            "stage03d",
            "stage03d_sustained_finalists.py",
            "sustained finalists",
        ),
        StageDef(4.0, "stage04", "stage04_convert_to_enriched_dataset.py", "enrich"),
        StageDef(
            4.5,
            "stage04b",
            "stage04b_iterate_active_learning.py",
            "validate prior model",
        ),
        StageDef(5.0, "stage05", "stage05_train.py", "train"),
        StageDef(6.0, "stage06", "stage06_deploy_weights.py", "deploy"),
        StageDef(
            7.0,
            "stage07",
            "stage07_optional_validate_selection_time.py",
            "selection time + parity",
        ),
        StageDef(
            8.0,
            "stage08",
            "stage08_optional_validate_selection_efficiency.py",
            "selection efficiency",
        ),
    )
}
BENCH_STAGES = ("stage03", "stage03b", "stage03c", "stage03d")
# Stages whose benches run through stage 3, which keeps a GPU the run holds
# out of its blocks and benches on the others (lib/gpu_health.py `holds`).
GUARDED_STAGES = ("stage02", *BENCH_STAGES, "stage04b", "stage06", "stage08")
# Stages whose exit status 2 reports trouble with the GPUs they bench on.
RETRIED_BENCH_STAGES = ("stage02", *BENCH_STAGES, "stage07")
TRAINING_STAGE_NUMBERS = (1.0, 2.0, 3.0, 3.2, 3.5, 3.7, 4.0, 4.5, 5.0, 6.0)
# The planner's number before it became stage 2 (its name was
# run_layout.LEGACY_PLAN_STAGE); --from-stage takes both, with a note.
OLD_PLANNER_NUMBER = 1.5


def parse_stage(text: str) -> Tuple[float, Optional[str]]:
    """The logical stage number --from-stage `text` names: a number, or the
    name of a stage (a key of STAGE_DEFS such as stage03c); with a
    deprecation note when it names the planner as it was named before it
    became stage 2. ValueError when it names neither."""
    try:
        number = float(text)
    except ValueError:
        name = text.strip()
        if name in STAGE_DEFS:
            return STAGE_DEFS[name].number, None
        if name != run_layout.LEGACY_PLAN_STAGE:
            raise ValueError(
                f"--from-stage {text}: neither a stage number nor a stage name "
                f"such as stage02"
            ) from None
        number = OLD_PLANNER_NUMBER
    if number != OLD_PLANNER_NUMBER:
        return number, None
    planner = STAGE_DEFS["stage02"]
    return planner.number, (
        f"--from-stage {text} is deprecated: the planner is stage "
        f"{planner.number:g} ({planner.key}) now; use --from-stage "
        f"{planner.number:g}"
    )


def stage_complete(stage_dir: Path) -> bool:
    return (stage_dir / STAGE_MARKER).is_file()


def mark_stage_complete(
    stage_dir: Path,
    cmd: Optional[List[str]],
    note: str = "",
    extra: Optional[Mapping[str, Any]] = None,
    pin: Optional[Mapping[str, Any]] = None,
    attempt: int = 1,
) -> None:
    """Write the stage's completion marker; `extra` adds fields to it, and
    `pin` (lib/pinning.py) is what the stage ran with."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    rec: Dict[str, Any] = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "cmd": [str(c) for c in cmd] if cmd else None,
    }
    if note:
        rec["note"] = note
    rec.update(extra or {})
    if attempt > 1:
        rec["attempt"] = attempt
    if pin is not None:
        rec["pin"] = pinning.summary(pin)
    (stage_dir / STAGE_MARKER).write_text(json.dumps(rec, indent=2) + "\n")


def stage03c_token() -> str:
    """Stands for a context's directory in stage03c's --commands."""
    return "{ctx_dir}"


def rocm_bin() -> str:
    """Where hipblaslt-cotenant finds rocminfo."""
    return str(Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "bin")


def _strip_pool_only(args: List[str]) -> List[str]:
    """`args` without the stage03 options of bench.pool_only_library (the
    context wrapper sets the library directory itself)."""
    out: List[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("--pool-only-library", "--library-dir"):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _replace_arg(args: List[str], flag: str, value: str) -> List[str]:
    """`args` with the value after every `flag` replaced, or `flag value`
    appended when absent."""
    out = list(args)
    hits = [i for i, a in enumerate(out[:-1]) if a == flag]
    if not hits:
        return out + [flag, value]
    for i in hits:
        out[i + 1] = value
    return out


def move_aside(owner_dir: Path, *paths: Path) -> None:
    """Move stale stage outputs of a previous attempt into
    `<owner_dir>/stale/<stamp>/`."""
    present = [p for p in paths if p.exists() or p.is_symlink()]
    if not present:
        return
    owner = owner_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = owner_dir / "stale" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for p in present:
        if p.resolve().parent != owner:
            raise RuntimeError(f"refusing to move {p}: not inside {owner_dir}")
        shutil.move(str(p), str(dest / p.name))
    ui.warn("stale", f"moved previous attempt's {[p.name for p in present]} to {dest}")


# ── run-dir discovery ───────────────────────────────────────────────────────


def round_index(path: Path) -> Optional[int]:
    m = re.fullmatch(r"round_(\d+)", path.name)
    return int(m.group(1)) if m else None


def round_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.is_dir():
        return []
    out = [p for p in run_dir.iterdir() if p.is_dir() and round_index(p) is not None]
    return sorted(out, key=lambda p: round_index(p) or 0)


def round_complete(rd: Path) -> bool:
    return (rd / ROUND_MARKER).is_file()


def models_ready(rd: Path) -> bool:
    return (rd / "stage05" / "models.pt").is_file() and stage_complete(rd / "stage05")


def complete_prefix(run_dir: Path) -> List[Path]:
    """round_0..round_{k-1}, the longest run of complete rounds from 0."""
    out: List[Path] = []
    by_idx = {round_index(p): p for p in round_dirs(run_dir)}
    k = 0
    while k in by_idx and round_complete(by_idx[k]):
        out.append(by_idx[k])
        k += 1
    return out


def latest_models_dir(rounds: Iterable[Path]) -> Optional[Path]:
    """stage05/ of the latest of `rounds` (bundle directories, in the order
    the run deploys them: `lib.run_bundles.bundle_key`) whose stage05
    completed with a models.pt."""
    for rd in sorted(rounds, key=rb.bundle_key, reverse=True):
        if models_ready(rd):
            return rd / "stage05"
    return None


def latest_deploy_record(
    rounds: Iterable[Path], deployed_only: bool = True
) -> Optional[Path]:
    """deploy_record.json of the latest of `rounds` (bundle directories, in
    the order the run deploys them: `lib.run_bundles.bundle_key`) whose
    stage06 completed; with `deployed_only`, of the latest one the deploy
    gate let through."""
    for rd in sorted(rounds, key=rb.bundle_key, reverse=True):
        p = rd / "stage06" / "deploy_record.json"
        if not (p.is_file() and stage_complete(rd / "stage06")):
            continue
        if deployed_only and (_read_json(p) or {}).get("deployed", True) is False:
            continue
        return p
    return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def held_out_sel_eff(decisions_path: Path) -> float:
    """n_eval-weighted geomean of `sel_eff_new` in one decisions.json."""
    d = _read_json(decisions_path)
    log_sum = w_sum = 0.0
    for c in (d or {}).get("per_cell") or []:
        if not isinstance(c, dict):
            continue
        se, n = c.get("sel_eff_new"), c.get("n_eval") or 0
        if isinstance(se, (int, float)) and se > 0 and n > 0:
            log_sum += math.log(float(se)) * float(n)
            w_sum += float(n)
    return math.exp(log_sum / w_sum) if w_sum > 0 else float("nan")


def shipped_round(rounds: Iterable[Path]) -> Optional[Path]:
    """The bundle directory of `rounds` (a round, or the final retrain)
    whose bundle this run deployed last."""
    rec = latest_deploy_record(rounds)
    return rec.parent.parent if rec is not None else None


def carried_cells(
    priors: Sequence[Path], retrained: Iterable[str], tree: Mapping[str, Any]
) -> List[str]:
    """The cells of the latest bundle of `priors` that a round carries
    forward: those it does not retrain (`retrained`) and its split tree
    `tree` (with the round's own splits) does not split."""
    if not priors:
        return []
    skip = set(retrained) | set(tree)
    return [c for c in ce.model_labels(priors[-1].parent, priors) if c not in skip]


def trained_model_options(rd: Path) -> Optional[Dict[str, Any]]:
    """The model-shaping options round `rd`'s stage05 trained with: its
    model_options.json, else those of the stage05 command its stage marker
    records; None when neither tells."""
    s5 = stage05_module()
    recorded = _read_json(rd / "stage05" / s5.MODEL_OPTIONS_FILE)
    if isinstance(recorded, dict):
        return recorded
    cmd = (_read_json(rd / "stage05" / STAGE_MARKER) or {}).get("cmd") or []
    script = STAGE_DEFS["stage05"].script
    at = [i for i, a in enumerate(cmd) if Path(str(a)).name == script]
    if not at:
        return None
    try:
        return s5.command_model_options([str(a) for a in cmd[at[0] + 1 :]])
    except (ValueError, OSError, yaml.YAMLError):
        return None


def option_changes(
    priors: Sequence[Path],
    cells: Iterable[str],
    options: Mapping[str, Any],
    final_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Tuple[Any, Any]]]]:
    """{cell: {option: (trained with, `options`)}} of the `cells` whose
    model was trained with other model-shaping options (stage05
    `model_options`); {cell: None} when those options are unknown. Where a
    cell was trained is `lib.run_bundles.cell_origins` over `priors` (bundle
    directories in the order the run deploys them): the latest of them that
    trained it, or the bundle the record of the one that carried it names,
    whose options are read there while it is in `priors` with that
    training, else taken from the record. A cell a final retrain trained is
    compared with `final_options` instead, the options a final retrain
    trains with now (`Pipeline.final_retrain_model_options`), when they are
    given."""
    s5 = stage05_module()
    origins = rb.cell_origins(priors)
    read: Dict[Path, Optional[Dict[str, Any]]] = {}
    out: Dict[str, Optional[Dict[str, Tuple[Any, Any]]]] = {}
    for cell in cells:
        o = origins.get(cell)
        trained = None
        if o is not None and o["bundle"] is not None:
            if o["bundle"] not in read:
                read[o["bundle"]] = trained_model_options(o["bundle"])
            trained = read[o["bundle"]]
        elif o is not None:
            trained = o["model_options"]
        final = o is not None and round_index(Path(o["trained_in"])) is None
        want = final_options if final and final_options is not None else options
        diff = None if trained is None else s5.changed_model_options(trained, want)
        if diff is None or diff:
            out[cell] = diff
    return out


def record_cell_origins(s5: Path, lineage: Sequence[Path]) -> None:
    """`stage05/cell_origins.json` of the bundle stage05 wrote into `s5`:
    for each cell of its model that it did not train but carried from the
    bundles before it (`lineage`, in the order the run deploys them), the
    bundle and the stage05 completion that trained it
    (`lib.run_bundles.cell_origins`), with the cell's cells.json entry
    there and the model-shaping options it trained with
    (`trained_model_options`), so that the bundle still tells them once
    that bundle has left the run or trained again (a final retrain
    replaced, a round run again). `source` names the latest bundle of
    `lineage`."""
    cj = _read_json(s5 / "cells.json")
    if not isinstance(cj, dict) or not lineage:
        return
    labels = cj.get("model_labels")
    if not isinstance(labels, list):
        labels = [c.get("label") for c in cj.get("cells") or [] if isinstance(c, dict)]
    trained = {str(c["label"]) for c in rb.trained_entries(s5.parent)}
    origins = rb.cell_origins(lineage)
    table: Dict[str, Any] = {}
    cells: Dict[str, Any] = {}
    for label in (str(x) for x in labels if x):
        o = origins.get(label)
        if label in trained or o is None:
            continue
        key = f"{o['trained_in']}@{o['finished']}"
        if key not in table:
            table[key] = {
                "bundle": o["trained_in"],
                "finished": o["finished"],
                "model_options": (
                    trained_model_options(o["bundle"])
                    if o["bundle"] is not None
                    else o["model_options"]
                ),
            }
        cells[label] = {"origin": key, "entry": o["entry"]}
    from lib.fs import atomic_write_text

    rec = {"source": Path(lineage[-1]).name, "origins": table, "cells": cells}
    atomic_write_text(
        s5 / rb.CELL_ORIGINS, json.dumps(_json_safe(rec), indent=2) + "\n"
    )


def _option_text(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def describe_option_changes(
    changes: Mapping[str, Optional[Mapping[str, Tuple[Any, Any]]]],
) -> str:
    """`option trained-with -> now, ...` over the cells of `changes`
    (`option_changes`), every value an option was trained with that differs
    from now, and how many cells were trained with unknown options."""
    was: Dict[str, set] = {}
    now: Dict[str, Any] = {}
    unknown = 0
    for diff in changes.values():
        if diff is None:
            unknown += 1
            continue
        for key, (old, new) in diff.items():
            was.setdefault(key, set()).add(_option_text(old))
            now[key] = new
    parts = [
        f"{key} {'/'.join(sorted(was[key]))} -> {_option_text(now[key])}"
        for key in sorted(was)
    ]
    if unknown:
        parts.append(f"{unknown} trained with unknown options")
    return ", ".join(parts)


def _sha256_file(path: Path) -> Optional[str]:
    import hashlib

    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _f(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return float(x)
    return None


def _stage04b_cells(rows: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in rows:
        if not isinstance(c, dict) or not c.get("cell"):
            continue
        se, n = _f(c.get("sel_eff_new")), int(c.get("n_eval") or 0)
        if se is None or se <= 0 or n <= 0:
            continue
        out[str(c["cell"])] = {
            "sel_eff_new": se,
            "n_eval": n,
            "n_miss": int(c.get("n_miss") or 0),
            "paired_sel_eff_new": _f(c.get("paired_sel_eff_new")),
            "origami_sel_eff_new": _f(c.get("origami_sel_eff_new")),
            "origami_n_eval": int(c.get("origami_n_eval") or 0),
            "winner_us_geomean": _f(c.get("winner_us_geomean")),
            "pick_us_geomean": _f(c.get("pick_us_geomean")),
            "origami_pick_us_geomean": _f(c.get("origami_pick_us_geomean")),
        }
    return out


def _paired(per_cell: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The model and Origami on the same GEMMs: per cell both numbers are
    over its `origami_n_eval` GEMMs."""

    def geo(key: str) -> float:
        ls = ws = 0.0
        for c in per_cell.values():
            v, n = _f(c.get(key)), int(c.get("origami_n_eval") or 0)
            if v is not None and v > 0 and n > 0 and _f(c.get("paired_sel_eff_new")):
                ls += math.log(v) * n
                ws += n
        return math.exp(ls / ws) if ws > 0 else float("nan")

    n = sum(
        int(c.get("origami_n_eval") or 0)
        for c in per_cell.values()
        if _f(c.get("paired_sel_eff_new")) and _f(c.get("origami_sel_eff_new"))
    )
    if not n:
        return None
    return {
        "n": n,
        "model_sel_eff": geo("paired_sel_eff_new"),
        "origami_sel_eff": geo("origami_sel_eff_new"),
    }


def shipped_bundle_numbers(run_dir: Path) -> Dict[str, Any]:
    """The bundle this run deployed last (a round's, or its final
    retrain's: lib/run_bundles.py), measured on GEMMs it never trained on
    (`bundle_numbers`, under `shipped`). `final_bundle` describes a later
    stage06 whose gate kept its bundle out; `final_retrain` the final
    retrain's decision and whether it ships (`final_retrain_numbers`)."""
    bundles = rb.bundles(run_dir, round_dirs(run_dir))
    last = latest_deploy_record(bundles, deployed_only=False)
    rec_path = latest_deploy_record(bundles)
    out = bundle_numbers(run_dir, rec_path)
    out["shipped"] = out.pop("bundle")
    out["final_bundle"] = None
    if last is not None and last != rec_path:
        rec = _read_json(last) or {}
        out["final_bundle"] = {
            "round": last.parent.parent.name,
            "train_dir": rec.get("train_dir"),
            "deployed": False,
            "reason": rec.get("not_deployed"),
            "gate_sel_eff": _f((rec.get("gate") or {}).get("sel_eff")),
        }
    out["final_retrain"] = final_retrain_numbers(run_dir, rec_path)
    return out


def final_retrain_numbers(
    run_dir: Path, shipped: Optional[Path]
) -> Optional[Dict[str, Any]]:
    """The final retrain's decision (final_retrain/decision.json) and
    whether it ships, its stage06 deploy record being `shipped`, the record
    of the bundle the run deployed last; `not_shipped` says why it does
    not, `rounds_after` names the complete rounds it did not train on. None
    without a decision."""
    d = Path(run_dir) / FINAL_RETRAIN_DIR
    dec = _read_json(d / FinalRetrain.DECISION)
    if not isinstance(dec, dict):
        return None
    final = rb.final_retrain(run_dir)
    ships = final is not None and shipped is not None and shipped.parent.parent == final
    trained = rb.trained_rounds(d)
    why = None
    if ships:
        pass
    elif not dec.get("accepted"):
        why = dec.get("reason")
    elif not stage_complete(d / "stage06"):
        why = "it passed its gate, and its stage06 has not deployed it"
    elif final is None:
        why = (
            f"{', '.join(rb.changed_rounds(run_dir, trained)) or 'its rounds'} "
            f"changed since it trained on them"
        )
    else:
        assert shipped is not None
        why = f"{shipped.parent.parent.name}'s bundle deployed after it"
    return {
        **{
            k: dec.get(k)
            for k in (
                "accepted",
                "reason",
                "protocol",
                "official",
                "gate",
                "selected_by_audit",
                "selection_note",
            )
        },
        "ships": ships,
        "not_shipped": why,
        "deploy_record": (
            str(d / "stage06" / "deploy_record.json") if final is not None else None
        ),
        "trained_on": list(trained),
        "rounds_after": [
            rd.name for rd in complete_prefix(run_dir) if rd.name not in trained
        ],
        "baseline": dec.get("baseline"),
        "final": dec.get("final"),
        "candidate": candidate_numbers(run_dir),
    }


def candidate_numbers(run_dir: Path) -> Optional[Dict[str, Any]]:
    """A new final retrain in final_retrain_next/ that has not replaced the
    one in final_retrain/ (`FinalRetrain.prepare`): its state
    (`final_retrain_state`) and decision; None without one."""
    d = Path(run_dir) / rb.FINAL_RETRAIN_NEXT_DIR
    if not d.is_dir():
        return None
    dec = _read_json(d / FinalRetrain.DECISION)
    dec = dec if isinstance(dec, dict) else {}
    return {
        **final_retrain_state(run_dir, d),
        **{
            k: dec.get(k)
            for k in (
                "accepted",
                "protocol",
                "official",
                "selected_by_audit",
                "selection_note",
                "baseline",
                "final",
            )
        },
        "deployed": stage_complete(d / "stage06"),
    }


def bundle_numbers(run_dir: Path, rec_path: Optional[Path]) -> Dict[str, Any]:
    """The bundle of the stage06 deploy record `rec_path` (deployed or not),
    measured on GEMMs it never trained on: its deploy gate
    (round_R/stage06/gate.json) and every stage04b that evaluated the same
    models.pt (the next round's, or this round's when it carried the bundle
    forward); for the final retrain, which trained on every round's GEMMs,
    its audit record (`final_audit_measurement`). The headline is the gate,
    else the latest of those measurements; `bundle` is None without a
    record."""
    rounds = round_dirs(run_dir)
    out: Dict[str, Any] = {
        "source": None,
        "bundle": None,
        "held_out_sel_eff": float("nan"),
        "n_gemms": 0,
        "n_cells": 0,
        "n_miss": 0,
        "paired_origami": None,
        "per_cell": {},
        "measurements": [],
    }
    if rec_path is None:
        return out
    rec = _read_json(rec_path) or {}
    rd = rec_path.parent.parent
    train_dir = Path(str(rec.get("train_dir") or rd / "stage05"))
    sha = _sha256_file(train_dir / "models.pt")
    out["bundle"] = {
        "round": rd.name,
        "train_dir": str(train_dir),
        "deploy_record": str(rec_path),
        "models_sha256": sha,
        "model_file_sha256": rec.get("sha256"),
    }
    gate = _read_json(rd / "stage06" / "gate.json")
    if isinstance(gate, dict) and gate.get("n_evaluated"):
        cells = gate.get("per_cell") if isinstance(gate.get("per_cell"), dict) else {}
        out["measurements"].append(
            {
                "source": "deploy_gate",
                "round": rd.name,
                "sel_eff": _f(gate.get("sel_eff")),
                "n_eval": int(gate.get("n_evaluated") or 0),
                "n_miss": int(gate.get("n_miss") or 0),
                "pick_measurement": gate.get("pick_measurement"),
                "per_cell": dict(cells),
            }
        )
    hashes: Dict[str, Optional[str]] = {}
    for later in rounds:
        dec = _read_json(later / "stage04b" / "decisions.json")
        if not isinstance(dec, dict) or not dec.get("models_dir"):
            continue
        mdir = Path(str(dec["models_dir"]))
        same = mdir.resolve() == train_dir.resolve()
        if not same and sha is not None:
            key = str(mdir.resolve())
            if key not in hashes:
                hashes[key] = _sha256_file(mdir / "models.pt")
            same = hashes[key] == sha
        if not same:
            continue
        agg = dec.get("aggregate") if isinstance(dec.get("aggregate"), dict) else {}
        cells = _stage04b_cells(dec.get("per_cell") or [])
        out["measurements"].append(
            {
                "source": "stage04b",
                "round": later.name,
                "sel_eff": _f(agg.get("global_sel_eff_new")),
                "n_eval": int(agg.get("n_gemms_evaluated") or 0),
                "n_miss": int(agg.get("n_miss_total") or 0),
                "pick_measurement": agg.get("pick_measurement"),
                "per_cell": cells,
            }
        )
    if round_index(rd) is None:
        audited = final_audit_measurement(rd)
        if audited is not None:
            out["measurements"].append(audited)
    usable = [m for m in out["measurements"] if m["sel_eff"] is not None]
    if not usable:
        return out
    head = next((m for m in usable if m["source"] == "deploy_gate"), usable[-1])
    out.update(
        source=head["source"],
        held_out_sel_eff=head["sel_eff"],
        n_gemms=head["n_eval"],
        n_miss=head["n_miss"],
        n_cells=len(head["per_cell"]),
        per_cell=head["per_cell"],
        paired_origami=_paired(head["per_cell"]),
    )
    return out


def _audit_rank(rec: Any) -> Tuple[bool, bool, bool, bool, bool]:
    """How complete an audit record is: measured, complete, then its v2
    complete, its sustained composite complete, its v2 partial."""
    if not isinstance(rec, dict):
        return (False, False, False, False, False)
    v2 = auditlib.protocol_record(rec, "v2").get("status")
    return (
        auditlib.is_measured(rec),
        rec.get("status") == "ok",
        v2 == "ok",
        auditlib.protocol_record(rec, "sustained").get("status") == "ok",
        v2 == "partial",
    )


def final_audit_record(final_dir: Path) -> Optional[Dict[str, Any]]:
    """The most complete audit record of the final retrain in `final_dir`
    (`_audit_rank`): its stage06 audit.json, the one the end of a later run
    scored again for the same model file (audit_results/final_retrain.json),
    else the one its gate scored; None without one."""
    own = _read_json(final_dir / "stage06" / auditlib.RESULT_FILE)
    later = _read_json(final_dir.parent / AUDIT_RESULTS_DIR / f"{final_dir.name}.json")
    found = [own] if isinstance(own, dict) else []
    if isinstance(later, dict) and (
        not found or later.get("model_sha256") == own.get("model_sha256")
    ):
        found.append(later)
    if not any(auditlib.is_measured(r) for r in found):
        doc = _read_json(final_dir / FinalRetrain.AUDIT)
        models = doc.get("models") if isinstance(doc, dict) else None
        gate = models[-1] if isinstance(models, list) and models else doc
        if isinstance(gate, dict):
            found.append(gate)
    return max(found, key=_audit_rank) if found else None


def final_audit_measurement(final_dir: Path) -> Optional[Dict[str, Any]]:
    """The final retrain in `final_dir` on GEMMs it never trained on: the
    exclusive group of its audit record (`final_audit_record`), under the
    official protocol it was scored with when that is complete for it, else
    protocol v2 when that is, its cells being base cells, with v2's numbers
    alongside (`v2`) under the sustained composite; None when the audit set
    did not measure it."""
    rec = final_audit_record(final_dir)
    if not (isinstance(rec, dict) and auditlib.is_measured(rec)):
        return None
    view, proto = auditlib.view(rec, rec.get("official"))
    g = auditlib.evaluated_group(view, auditlib.EXCLUSIVE)
    if g is None:
        return None
    alongside = None
    if proto == "sustained":
        g2 = auditlib.evaluated_group(
            auditlib.protocol_record(rec, "v2"), auditlib.EXCLUSIVE
        )
        if g2 is not None:
            alongside = {
                "cell_mean": _f(g2.get("cell_mean")),
                "cell_mean_ci": g2.get("cell_mean_ci"),
                "n_cells_below_target": g2.get("n_cells_below_target"),
            }
    cells = {
        str(c): {
            "sel_eff_new": _f(v.get("sel_eff")),
            "n_eval": int(v.get("n") or 0),
            "n_miss": int(v.get("n_miss") or 0),
        }
        for c, v in (g.get("per_cell") or {}).items()
        if isinstance(v, dict) and _f(v.get("sel_eff"))
    }
    return {
        "source": "audit",
        "round": final_dir.name,
        "sel_eff": _f(g.get("gemm_geomean")),
        "n_eval": int(g.get("n_gemms") or 0),
        "n_miss": sum(c["n_miss"] for c in cells.values()),
        "pick_measurement": rec.get("pick_measurement"),
        "protocol": proto,
        "cell_mean": _f(g.get("cell_mean")),
        "cell_mean_ci": g.get("cell_mean_ci"),
        "n_cells_below_target": g.get("n_cells_below_target"),
        "per_cell": cells,
        "v2": alongside,
    }


def _ci_text(ci: Any) -> str:
    if isinstance(ci, list) and len(ci) == 2 and all(_f(v) is not None for v in ci):
        return f" [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"
    return ""


def print_shipped(numbers: Mapping[str, Any], origami_reference: bool) -> None:
    final = numbers.get("final_bundle")
    shipped = numbers.get("shipped")
    retrain = numbers.get("final_retrain") or {}
    name = shipped["round"] if shipped is not None else None
    unmeasured = "no deploy gate GEMMs"
    if shipped is not None and round_index(Path(str(name))) is None:
        n = len(retrain.get("trained_on") or [])
        name = f"{name} (every cell retrained on the data of {n} rounds)"
        unmeasured = "the audit set did not measure it"
    head = next(
        (
            m
            for m in numbers.get("measurements") or []
            if m["source"] == numbers.get("source")
        ),
        None,
    )
    if shipped is None:
        ui.warn("done", "this run deployed no bundle")
    elif numbers.get("source") is None:
        ui.ok(
            "done",
            f"shipped bundle {name}: not measured on GEMMs it did not train on "
            f"({unmeasured})",
        )
    elif numbers["source"] == "audit" and head is not None:
        cm = head.get("cell_mean")
        ui.ok(
            "done",
            f"shipped bundle {name}: "
            + (
                f"audit cell mean {cm * 100:.2f}%{_ci_text(head.get('cell_mean_ci'))}, "
                if cm is not None
                else ""
            )
            + f"GEMM geomean {numbers['held_out_sel_eff'] * 100:.2f}% on "
            f"{numbers['n_gemms']} audit GEMMs it never trained on over "
            f"{numbers['n_cells']} cells ({numbers['n_miss']} misses; protocol "
            f"{head.get('protocol')})"
            + (
                f"; v2 alongside {head['v2']['cell_mean'] * 100:.2f}%"
                f"{_ci_text(head['v2'].get('cell_mean_ci'))}"
                if (head.get("v2") or {}).get("cell_mean") is not None
                else ""
            ),
        )
    else:
        ui.ok(
            "done",
            f"shipped bundle {name}: "
            f"{numbers['held_out_sel_eff'] * 100:.2f}% on {numbers['n_gemms']} "
            f"held-out GEMMs over {numbers['n_cells']} cells "
            f"({numbers['n_miss']} misses; {numbers['source']})",
        )
    if shipped is not None and numbers.get("source") is not None:
        for m in numbers.get("measurements") or []:
            if m["source"] == numbers["source"] and m["round"] == shipped["round"]:
                continue
            if m["sel_eff"] is not None:
                ui.ok(
                    "done",
                    f"  also measured by {m['round']} {m['source']}: "
                    f"{m['sel_eff'] * 100:.2f}% on {m['n_eval']} GEMMs",
                )
        p = numbers.get("paired_origami")
        if origami_reference and p:
            ui.info(
                "done",
                f"[reference] Origami on the same {p['n']} GEMMs: "
                f"{p['origami_sel_eff'] * 100:.2f}% (shipped bundle "
                f"{p['model_sel_eff'] * 100:.2f}%)",
            )
    if final is not None:
        gse = final.get("gate_sel_eff")
        ui.warn(
            "done",
            f"the bundle of {final['round']} was not deployed"
            + (f" ({gse * 100:.2f}% on its gate GEMMs)" if gse is not None else "")
            + (f": {final['reason']}" if final.get("reason") else ""),
        )
    if not retrain:
        return
    cand = retrain.get("candidate")
    if retrain.get("ships"):
        ui.ok("done", f"the final retrain ships: {retrain.get('reason')}")
        if retrain.get("selection_note"):
            ui.info("done", f"  {retrain['selection_note']}")
        after = retrain.get("rounds_after") or []
        if after and not cand:
            ui.warn(
                "done",
                f"the final retrain did not train on {', '.join(after)}, which "
                f"completed after it: the next --mode full trains it again",
            )
    else:
        ui.warn(
            "done", f"the final retrain does not ship: {retrain.get('not_shipped')}"
        )
        chosen = (retrain.get("baseline") or {}).get("round")
        if (
            retrain.get("selection_note")
            and retrain.get("accepted") is False
            and shipped is not None
            and shipped["round"] == chosen
        ):
            ui.info("done", f"  {retrain['selection_note']}")
    if cand:
        print_candidate(cand, shipped)


def print_candidate(
    cand: Mapping[str, Any], shipped: Optional[Mapping[str, Any]]
) -> None:
    """The run summary's line on a new final retrain that has not replaced
    the one in final_retrain/ (`candidate_numbers`)."""
    rounds = ", ".join(cand.get("trained_on") or []) or "no round yet"
    who = f"a new final retrain (trained on {rounds}; {cand.get('dir')})"
    status = cand.get("status")
    if status == "rejected":
        ui.warn("done", f"{who} does not ship: {cand.get('reason')}")
        chosen = (cand.get("baseline") or {}).get("round")
        if cand.get("selection_note") and (shipped or {}).get("round") == chosen:
            ui.info("done", f"  {cand['selection_note']}")
    elif status == "undecided":
        ui.warn(
            "done",
            f"{who} is not decided yet: its gate waits for the audit set, and "
            f"the same command decides it",
        )
    elif status == "accepted" and cand.get("deployed"):
        ui.warn(
            "done",
            f"{who} passed its gate and stage06 deployed it: the next --mode "
            f"full puts it in the place of the final retrain it replaces",
        )
    elif status == "accepted":
        ui.warn(
            "done",
            f"{who} passed its gate and stage06 has not deployed it: the same "
            f"command deploys it",
        )
    else:
        ui.warn(
            "done",
            f"{who} is pending ({cand.get('reason') or 'its training did not finish'})"
            f": the next --mode full trains it",
        )


def untrained_cell_lines(run_dir: Path) -> List[str]:
    """One line per round that went on with cells its stage05 did not train
    (train.on_cell_failure: carry; stage05/cells_carried.json)."""
    out = []
    for rd in round_dirs(run_dir):
        path = rd / "stage05" / CARRIED_CELLS
        rec = _read_json(path)
        cells = [str(c.get("cell")) for c in (rec or {}).get("cells") or []]
        if cells:
            out.append(
                f"{rd.name} went on without training {len(cells)} cell"
                f"{'s' if len(cells) != 1 else ''}: {', '.join(cells)} ({path})"
            )
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _plain(obj: Any) -> Any:
    """`obj` as it reads back from JSON."""
    return json.loads(json.dumps(_json_safe(obj)))


def validated_by(stage_dir: Path) -> Optional[Any]:
    """What the pass of the validation stage in `stage_dir` vouches for (its
    marker's VALIDATED field); None without a pass."""
    rec = _read_json(stage_dir / STAGE_MARKER)
    return rec.get(VALIDATED) if isinstance(rec, dict) else None


def validation_covers(passed: Any, want: Mapping[str, Any]) -> bool:
    """A pass that vouches for `passed` stands for `want`: the same model and
    settings, where a pass that also measured the Origami reference stands
    for one without it."""
    if not isinstance(passed, dict):
        return False
    if passed.get("origami_reference") and not want.get("origami_reference"):
        passed = {**passed, "origami_reference": want.get("origami_reference")}
    return passed == dict(want)


# ── pipeline ────────────────────────────────────────────────────────────────


@dataclass
class RoundStep:
    index: int
    mode: str
    priors: List[Path]
    from_stage: Optional[float]
    is_final: bool
    targets: Optional[List[str]] = None
    shapes_per_cell: Optional[int] = None
    selection: Optional[Dict[str, Any]] = None


@dataclass
class RoundResult:
    rc: int
    held_out_sel_eff: float = float("nan")
    gate_sel_eff: float = float("nan")
    deployed: bool = False
    target_reached: bool = False


@dataclass
class GatePlan:
    """The stage06 deploy gate of one round (stage06 `Gate`)."""

    csv_dir: Path
    target: Optional[float] = None
    fallback: str = "none"
    baseline_train_dir: Optional[Path] = None


@dataclass
class CellGatePlan:
    """The per-cell gate of a round's stage06 (stage06 `CellGate`): the
    directory of the bundle the run ships, the config's gate settings, and
    whether it only records (--deploy)."""

    baseline_dir: Path
    settings: Dict[str, Any]
    record_only: bool = False


class Pipeline:
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        snapshot: Path,
        run_dir: Path,
        paths: Paths,
        stages_dir: Path,
        timing: Timing,
        force_deploy: bool,
        origami_reference: bool = False,
        pin: Optional[Dict[str, Any]] = None,
        pin_parts: Sequence[str] = pinning.PARTS,
        invocation: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.snapshot = snapshot
        self.run_dir = run_dir
        self.invocation = invocation or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.guard_dir = Path(run_dir) / gpu_health.GUARD_DIR
        self.run_root = host_load.run_root() or os.getpid()
        self.paths = paths
        self.stages_dir = stages_dir
        self.timing = timing
        self.force_deploy = force_deploy
        self.origami_reference = origami_reference
        self.pin = pin
        self.pin_parts = tuple(pin_parts)
        self.running = attempts.Running(run_dir)
        self.layout = run_layout.of(run_dir)
        self.arch = str(cfg["arch"])
        self.stem = str(cfg["hipblaslt"]["library_stem"])
        self.devices = [int(d) for d in cfg["runtime"]["devices"]]
        self.weight_dtype = str(cfg_get(cfg, "deploy.weight_dtype"))
        self.planner = bool(cfg_get(cfg, "planner.enabled"))
        self.contexts = ctxlib.contexts_from_config(cfg)
        self.rebench = bool(cfg_get(cfg, "bench.rebench.enabled"))
        self.sustained = bool(cfg_get(cfg, "bench.sustained.enabled"))
        self._bank_stems: Optional[List[str]] = None
        self.validation: Dict[str, Dict[str, Any]] = {}
        self.blocklist: Optional[Dict[str, Any]] = None
        self.canary = bool(cfg_get(cfg, "hazards.enabled"))
        self.quarantine: Optional[Path] = None

    def get(self, dotted: str) -> Any:
        return cfg_get(self.cfg, dotted)

    def stage_env(self) -> Dict[str, str]:
        """The environment of every stage process: this one's with
        runtime.env over it, which the stages' benches inherit, the run's
        guard directory and this invocation's session (lib/gpu_health.py),
        which the benches do not get, and the process whose tree is the run
        (lib/host_load.py `run_root`: this one, or the orchestrator that
        started it, as for a parallel audit build)."""
        return {
            **os.environ,
            **benv.runtime_env(self.cfg),
            gpu_health.GUARD_DIR_VAR: str(self.guard_dir),
            gpu_health.SESSION_VAR: self.invocation,
            host_load.RUN_ROOT_VAR: str(self.run_root),
        }

    def wait_gpu_health(
        self, devices: Sequence[int], label: str, per_gpu: bool = False
    ) -> bool:
        """Wait, while the run's hold holds the GPUs of `devices`, checking
        them (and those the hold names) against the run's baseline
        (lib/gpu_health.py), before a stage on them starts: a hold of every
        GPU of the run holds any stage, a hold of single GPUs a stage on
        one of them, or, for a stage that keeps a held GPU out of its work
        itself (`per_gpu`: stage 3's blocks), only one on nothing but held
        GPUs. False when bench.health.max_wait_s passes."""
        settings = gpu_health.Settings.of(self.cfg)
        if not settings.enabled or not devices:
            return True
        try:
            phys = kfd.bench_gpus(devices)
        except kfd.GpuMapError:
            return True
        topology = {g.index: g for g in kfd.gpus()}
        gpus = [topology[g] for g in phys if g in topology]
        keys = [gpu_health.gpu_key(g) for g in gpus]

        def blocks(hold: Mapping[str, Any]) -> bool:
            held = [k for k in keys if gpu_health.holds(hold, k)]
            if not keys or gpu_health.run_wide(hold):
                return True
            return len(held) == len(keys) if per_gpu else bool(held)

        guard = gpu_health.Guard(
            self.guard_dir,
            self.invocation,
            by=f"orchestrator pid {os.getpid()} before {label}",
        )
        return gpu_health.wait_healthy(
            guard, gpus, settings, report=lambda m: ui.warn("health", m), blocks=blocks
        )

    # ── stage plumbing ──

    def stage_name(self, key: str) -> str:
        """The name of the stage `key`'s directory, log and timing rows in
        this run (lib/run_layout.py)."""
        return self.layout.dir_name(key)

    def plan_dir(self, owner: Path) -> Path:
        """The planner's directory of a round, or of the audit directory
        this pipeline builds."""
        return Path(owner) / self.stage_name("stage02")

    def health_devices(self, key: str, leased: Sequence[int]) -> List[int]:
        """The devices whose GPUs are checked before stage `key` starts: the
        ones it leased (`stage_devices` that no other process tree holds),
        and for the correctness gate and the hazard canary, which lease
        their own, the ones they are given."""
        if key == "stage00":
            return [int(d) for d in (self.get("correctness.devices") or self.devices)]
        if key == "stage00a":
            return [int(d) for d in (self.get("hazards.devices") or self.devices[:1])]
        return list(leased)

    def stage_devices(self, key: str, args: List[str]) -> List[int]:
        """The devices a stage benches on with `args`: none for a stage
        that does not bench."""
        if key in BENCH_STAGES:
            return list(self.devices)
        if key == "stage02" and "--seed-commands" in args:
            return list(self.devices)
        if key in ("stage04b", "stage06", "stage08") and "--rebench-misses" in args:
            return list(self.devices)
        if key == "stage07":
            return self.devices[:1]
        return []

    def stage_cmd(self, key: str, args: List[str]) -> List[str]:
        script = self.stages_dir / STAGE_DEFS[key].script
        return [sys.executable, "-u", str(script), *args]

    def mark_complete(
        self,
        stage_dir: Path,
        cmd: Optional[List[str]],
        note: str = "",
        attempt: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """The stage's completion marker with the fields of `extra`, the
        gate's blocklist the invocation applies (`blocklist`) and this
        invocation's pin, which becomes the run's (`<run dir>/pin.json`) when
        it has none yet, unless the invocation keeps no part of it
        (validation)."""
        if self.blocklist is not None:
            extra = {**(extra or {}), "blocklist": self.blocklist}
        mark_stage_complete(
            stage_dir, cmd, note, extra=extra, pin=self.pin, attempt=attempt
        )
        if (
            self.pin is not None
            and self.pin_parts
            and pinning.read_run_pin(self.run_dir) is None
        ):
            first = os.path.relpath(stage_dir, self.run_dir)
            pinning.write_run_pin(self.run_dir, self.pin, first_stage=first)

    def pin_drift(self) -> List[str]:
        """How the code and engine module on disk differ from the ones this
        invocation started with, in the parts it must keep (`pin_parts`). A
        change of another part is logged and taken into the pin that later
        stage markers record."""
        if self.pin is None:
            return []
        now = pinning.refreshed(self.pin, THIS_DIR)
        kept = pinning.differences(self.pin, now, set(self.pin_parts) - {"config"})
        if kept:
            return kept
        changes = pinning.differences(self.pin, now, ("code", "engine"))
        for change in changes:
            ui.warn("pin", f"going on with a change this run does not pin: {change}")
        if changes:
            self.pin = now
        return []

    def retry_reason(
        self, key: str, args: List[str], rc: int, cause: str = ""
    ) -> Optional[str]:
        """Why a failed attempt of a stage is worth another (None: it is
        not): its GPU lease stayed busy (`cause` "busy"); the exit 2 of a
        stage that benches, which it gives only for trouble with its GPUs
        (crash blocks beyond tolerance, repeated quarantines, every GPU
        retired, a refused lease or platform check, a seed bench that failed
        so); a death by SIGHUP, SIGINT, SIGKILL or SIGTERM that this
        orchestrator did not send (the OOM killer's, for one). A lease that
        cannot be taken for another reason (`cause` "setup"), setup, config
        and identity errors (exit 1) and a death by another signal (a crash)
        are not retried."""
        if rc == 0 or cause == "setup":
            return None
        if cause == "busy":
            return "found a GPU lease busy"
        if cause == "health":
            return "found the run's GPUs failing their checks"
        sig = procs.killed_by(rc)
        if sig is not None:
            return f"was killed by {sig.name}"
        if rc == 2 and key in RETRIED_BENCH_STAGES and self.stage_devices(key, args):
            return "failed on its GPUs"
        return None

    def run_stage(
        self,
        owner_dir: Path,
        key: str,
        args: List[str],
        round_idx: Optional[int],
        out_dir: Optional[Path] = None,
        marker: Optional[Mapping[str, Any]] = None,
        timing_stage: Optional[str] = None,
    ) -> int:
        """Run one stage under the lease of the GPUs it benches on that no
        other process tree holds (lib/gpu_lease.py `hold_usable`), so that
        its sub-stages and benches share it, once the GPUs it leased pass
        their health checks (`wait_gpu_health`, `health_devices`); every GPU
        held for longer than runtime.gpu_lease_wait_s fails the attempt with
        2 before it starts.
        A failed attempt that `retry_reason` names is retried up to
        runtime.stage_retries times, the first time after
        runtime.stage_retry_backoff_s and then after twice the last wait,
        each attempt in a fresh output directory. The stage neither starts
        nor completes while the code or the engine module differ from the
        ones this invocation started with (`pin_drift`): the run stops with
        2. `marker` adds fields to its stage marker; the timing ledger
        names it `timing_stage` (default `key`)."""
        retries = int(self.get("runtime.stage_retries"))
        wait = float(self.get("runtime.stage_retry_backoff_s"))
        name = self.stage_name(key)
        attempt = 1
        while True:
            drift = self.pin_drift()
            if drift:
                self._report_drift(name, drift, "not started")
                return 2
            rc, cause = self._stage_attempt(
                owner_dir,
                key,
                args,
                round_idx,
                out_dir,
                attempt,
                marker,
                timing_stage,
            )
            if cause == "pin":
                return rc
            why = self.retry_reason(key, args, rc, cause)
            if rc == 0 or why is None:
                return rc
            if attempt > retries:
                if retries:
                    ui.err("retry", f"{name} {why} in each of {attempt} attempts")
                return rc
            ui.warn(
                "retry",
                f"{name} {why} (rc={rc}); attempt {attempt + 1} of {retries + 1} "
                f"starts in {ui.fmt_dur(wait)}",
            )
            time.sleep(wait)
            wait *= 2
            attempt += 1

    def _report_drift(self, key: str, drift: List[str], what: str) -> None:
        _print_problems(
            "pin",
            [
                f"{key} {what}: the code changed while this invocation ran",
                *(f"  {d}" for d in drift),
                "run the same command to go on with the new code, adding "
                "--allow-code-change",
            ],
            [],
        )

    def _stage_attempt(
        self,
        owner_dir: Path,
        key: str,
        args: List[str],
        round_idx: Optional[int],
        out_dir: Optional[Path],
        attempt: int,
        marker: Optional[Mapping[str, Any]] = None,
        timing_stage: Optional[str] = None,
    ) -> Tuple[int, str]:
        """(exit code, cause) of one attempt; the cause of a failure the
        stage did not report: "busy" (every GPU lease stayed busy), "setup"
        (a lease could not be taken otherwise, such as for devices the
        environment does not show), "health" (the run's GPUs failed their
        checks for longer than bench.health.max_wait_s) or "pin" (the code
        changed while it ran), else empty."""
        sdef = STAGE_DEFS[key]
        name = self.stage_name(key)
        out_dir = out_dir or owner_dir / name
        log = owner_dir / f"{name}.log"
        cmd = self.stage_cmd(key, args)
        title = f"Stage {sdef.number:g} -- {sdef.title}"
        ui.banner(title + (f" (attempt {attempt})" if attempt > 1 else ""), ui.C.BLUE)
        t0 = time.time()
        cause = ""
        label = f"{owner_dir.name} {name}"
        try:
            with gpu_lease.hold_usable(
                self.stage_devices(key, args),
                label,
                self.run_dir,
                wait_s=float(self.get("runtime.gpu_lease_wait_s")),
                report=lambda msg: ui.warn("lease", f"{label}: {msg}"),
            ) as leases:
                if not self.wait_gpu_health(
                    self.health_devices(key, leases.devices),
                    label,
                    per_gpu=key in GUARDED_STAGES,
                ):
                    ui.err(
                        "health",
                        f"{name} not started: the run's GPUs fail their checks",
                    )
                    rc, cause = 2, "health"
                else:
                    if leases.acquired:
                        where = gpu_lease.lease_dir()
                        ui.info("lease", f"GPUs {leases.acquired} leased ({where})")
                    move_aside(owner_dir, out_dir, log)
                    rc = run_logged(
                        cmd,
                        log,
                        env=self.stage_env(),
                        running=self.running,
                        label=label,
                        pass_fds=leases.fds,
                    )
        except gpu_lease.LeaseBusy as e:
            ui.err("lease", f"{name} not started: {e}")
            rc, cause = 2, "busy"
        except gpu_lease.LeaseError as e:
            ui.err("lease", f"{name} not started: {e}")
            rc, cause = 2, "setup"
        self.timing.add(round_idx, timing_stage or name, rc, time.time() - t0, attempt)
        if rc == 0:
            drift = self.pin_drift()
            if drift:
                self._report_drift(name, drift, "is not marked complete")
                return 2, "pin"
            self.mark_complete(out_dir, cmd, attempt=attempt, extra=marker)
        return rc, cause

    def _require_complete(self, rd: Path, key: str) -> bool:
        name = self.stage_name(key)
        if stage_complete(rd / name):
            ui.warn("stage", f"{name} skipped (already complete)")
            return True
        ui.err(
            "stage",
            f"cannot skip {name}: {rd / name} has no completed output; "
            f"resume from an earlier stage",
        )
        return False

    def _round_stage_keys(self, mode: str) -> List[str]:
        keys = ["stage01"]
        if self.planner:
            keys.append("stage02")
        keys.append("stage03")
        if self.rebench:
            keys.append("stage03b")
        if self.contexts:
            keys.append("stage03c")
        if self.sustained:
            keys.append("stage03d")
        keys.append("stage04")
        if mode == "active":
            keys.append("stage04b")
        keys.append("stage05")
        return keys

    def round_plans(self, rd: Path) -> List[Path]:
        """The plan.json files of a round's planner: its own, the seed
        bootstrap's and the context re-plans."""
        d = self.plan_dir(rd)
        found = [d / "plan.json", d / "seed" / "plan.json"]
        found += sorted((d / "contexts").glob("*/plan.json"))
        return [p for p in found if p.is_file()]

    def plan_identity_errors(self, rd: Path) -> List[str]:
        """Errors for the round's plans whose solution indices name other
        kernels in the library the round benches with
        (lib/kernel_identity.py): the hipBLASLt build changed since they were
        made, so their bench lines would run other kernels. A plan that
        records no kernels has its lists benched as they are, with a
        warning."""
        plans = self.round_plans(rd) if self.planner else []
        if not plans:
            return []
        try:
            current = bank_pool.config_identity(
                self.cfg, self.paths.library_dir, self.stem, self.devices[0]
            )
        except (OSError, ValueError, RuntimeError) as e:
            return [f"cannot read the kernel identities of the library: {e}"]
        out = []
        for p in plans:
            try:
                doc = json.loads(p.read_text())
                rec = ki.read_record(doc)
                if rec is None:
                    shapes = doc.get("shapes") if isinstance(doc, dict) else None
                    if any(e.get("kernels") is not None for e in shapes or []):
                        ki.warn_unrecorded(f"plan {p}")
                    continue
                ki.Remap.of(rec, current, f"plan {p}").require_same()
            except ki.BuildChangedError as e:
                out.append(
                    f"{e}; its bench lines would run other kernels: plan the "
                    f"round again (--from-stage 2)"
                )
            except (OSError, ValueError) as e:
                out.append(f"cannot read {p}: {e}")
        return out

    def first_incomplete_stage(self, rd: Path, mode: str) -> float:
        for key in self._round_stage_keys(mode):
            if not stage_complete(rd / self.stage_name(key)):
                return STAGE_DEFS[key].number
        return 6.0

    # ── argument builders ──

    def _common(self) -> List[str]:
        return ["--config-yaml", str(self.snapshot)]

    def args_stage01(self, rd: Path, step: RoundStep) -> List[str]:
        args = [
            "--mode",
            step.mode,
            "--out-dir",
            str(rd / "stage01"),
            *self._common(),
            "--seed",
            str(self.get("seed.seed")),
            "--round-index",
            str(step.index),
            "--n-strata",
            str(self.get("seed.n_strata")),
        ]
        if step.mode == "initial":
            args += ["--target-per-cell", str(self.get("seed.target_per_cell"))]
        else:
            for p in step.priors:
                args += ["--prior-round-dir", str(p)]
            args += [
                "--shapes-per-cell",
                str(self.shapes_per_cell(step)),
                "--min-cell-gemms",
                str(self.get("train.min_cell_gemms")),
            ]
            if step.targets is not None:
                args += ["--only-cells", ",".join(step.targets)]
        audit = auditlib.config_dir(self.cfg)
        if audit is not None:
            shapes = auditlib.shapes_yaml(audit)
            if shapes.is_file():
                args += ["--exclude-shapes-yaml", str(shapes)]
            else:
                ui.warn(
                    "audit",
                    f"{shapes} does not exist yet: {rd.name} cannot exclude the "
                    f"audit shapes (evaluations drop any it draws)",
                )
        return args

    def shapes_per_cell(self, step: Optional[RoundStep] = None) -> int:
        if step is not None and step.shapes_per_cell is not None:
            return int(step.shapes_per_cell)
        return int(self.get("seed.shapes_per_cell"))

    def shapes_dir(self, rd: Path) -> Path:
        """The stage whose shapes.yaml the round benches."""
        return self.plan_dir(rd) if self.planner else rd / "stage01"

    def args_stage03(
        self, rd: Path, in_dir: Optional[Path] = None, out_dir: Optional[Path] = None
    ) -> List[str]:
        return (
            [
                "--in-dir",
                str(in_dir or self.shapes_dir(rd)),
                "--out-dir",
                str(out_dir or rd / "stage03"),
                *self._common(),
                "--bench-binary",
                str(self.paths.bench_binary),
                "--devices",
                ",".join(str(d) for d in self.devices),
                "--blocks-per-gpu",
                str(self.get("bench.blocks_per_gpu")),
                "--stall-timeout-s",
                str(self.get("bench.stall_timeout_s")),
                "--startup-grace-s",
                str(self.get("bench.startup_grace_s")),
            ]
            + self._pool_only_args()
            + self.identity_args()
        )

    def identity_args(self) -> List[str]:
        """stage03 / stage03b options naming the library files whose
        solution indices the bench lines name (the pool's and the bank's),
        whose kernel identity the benches record for stage04."""
        args = ["--identity-dir", str(self.paths.library_dir)]
        for stem in [self.stem] + self.bank_stems():
            args += ["--identity-stem", stem]
        return args

    def _pool_only_args(self) -> List[str]:
        """stage03 / stage03c options of bench.pool_only_library."""
        if not self.get("bench.pool_only_library"):
            return []
        args = ["--library-dir", str(self.paths.library_dir)]
        stems = [self.stem] + self.bank_stems()
        for stem in stems:
            args += ["--pool-only-library", stem]
        return args

    def pool_digests(self) -> Dict[int, str]:
        """{solution: kernel digest} (lib/audit.py) of the pool the library
        and the bank give now, blocked kernels too, quarantined ones not
        (`config_pool_digests`)."""
        pool = bank_pool.config_pool(
            self.cfg,
            self.paths.library_dir,
            self.stem,
            self.devices[0],
            apply_blocklist=False,
        )
        return auditlib.pool_digests(hazlib.without_quarantined(self.cfg, pool.kernels))

    def bank_stems(self) -> List[str]:
        """Library files of the planner's bank besides the pool's own
        (`lib.bank_pool.bank_library_stems`), resolved once."""
        if self._bank_stems is None:
            self._bank_stems = bank_pool.bank_library_stems(
                self.cfg, self.paths.library_dir, self.stem, self.devices[0]
            )
        return list(self._bank_stems)

    def context_dir(self, rd: Path, ctx: "ctxlib.BenchContext") -> Path:
        return rd / "stage03c" / ctx.label

    def args_stage03c(self, rd: Path) -> List[str]:
        token = Path(stage03c_token())
        cmds: Dict[str, Any] = {
            "stage03": [self._script("stage03")]
            + _strip_pool_only(
                self.args_stage03(rd, in_dir=token, out_dir=token / "stage03")
            ),
            "stage04": [self._script("stage04")]
            + self.args_stage04(rd, log_dirs=[], out_dir=token / "stage04"),
        }
        args = []
        for y in self.context_shape_yamls(rd):
            args += ["--shapes-yaml", str(y)]
        args += [
            "--out-dir",
            str(rd / "stage03c"),
            *self._common(),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--cotenant-binary",
            str(context_cotenant(self.cfg, self.paths)),
            "--hardware-device",
            str(self.devices[0]),
            "--commands",
            json.dumps(cmds),
        ] + self._pool_only_args()
        if self.planner and self.get("contexts.replan"):
            args += ["--context-shapes-dir", str(self.plan_dir(rd) / "contexts")]
        for p in self.get("contexts.path_append") or [rocm_bin()]:
            args += ["--path-append", str(p)]
        return args

    def context_shape_yamls(self, rd: Path) -> List[Path]:
        """The bench lines every context draws from: the round's own, plus the
        seed bootstrap's (benched exhaustively in the exclusive context)."""
        yamls = [self.shapes_dir(rd) / "shapes.yaml"]
        if self.planner:
            yamls.append(self.plan_dir(rd) / "seed" / "shapes.yaml")
        return [y for y in yamls if y.is_file()] or yamls[:1]

    def context_log_args(self, rd: Path) -> List[str]:
        """stage04 options of the trained contexts' logs (and re-plans)."""
        args: List[str] = []
        for ctx in self.contexts:
            if ctx.holdout:
                continue
            cdir = self.context_dir(rd, ctx)
            budget, sched = str(ctx.cu_budget), ctx.schedule
            args += ["--context-log-dir", budget, sched, str(cdir / "stage03" / "logs")]
            replan = self.plan_dir(rd) / "contexts" / ctx.label / "plan.json"
            if self.planner and self.get("contexts.replan") and replan.is_file():
                args += ["--context-plan", budget, sched, str(replan)]
        return args

    def args_stage03b(self, rd: Path) -> List[str]:
        bench = self.args_stage03(rd)
        return [
            "--in-dir",
            bench[bench.index("--in-dir") + 1],
            "--stage03-dir",
            str(rd / "stage03"),
            "--out-dir",
            str(rd / "stage03b"),
            *self._common(),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--stall-timeout-s",
            str(self.get("bench.stall_timeout_s")),
            "--startup-grace-s",
            str(self.get("bench.startup_grace_s")),
        ] + self.identity_args()

    def args_stage03d(self, rd: Path) -> List[str]:
        """stage03d re-measures the finalists of the round's exclusive bench:
        stage03b's merged logs with the near-winner re-reads, else stage03's."""
        return [
            "--in-dir",
            str(self.shapes_dir(rd)),
            "--stage03-dir",
            str(rd / ("stage03b" if self.rebench else "stage03")),
            "--out-dir",
            str(rd / "stage03d"),
            *self._common(),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--stall-timeout-s",
            str(self.get("bench.stall_timeout_s")),
            "--startup-grace-s",
            str(self.get("bench.startup_grace_s")),
            *self._pool_only_args(),
            *self.identity_args(),
        ]

    @staticmethod
    def sustained_log_dirs(rd: Path) -> List[Path]:
        """The log directories of a round's stage03d runs."""
        return sorted((rd / "stage03d" / "runs").glob("run_*/stage03/logs"))

    def args_stage04(
        self,
        rd: Path,
        log_dirs: Optional[List[Path]] = None,
        out_dir: Optional[Path] = None,
        plan: Optional[Path] = None,
    ) -> List[str]:
        default_logs = log_dirs is None
        if log_dirs is None:
            log_dirs = [rd / ("stage03b" if self.rebench else "stage03") / "logs"]
            seed_logs = self.plan_dir(rd) / "seed" / "stage03" / "logs"
            if self.planner and seed_logs.is_dir():
                log_dirs.append(seed_logs)
        if plan is None and self.planner:
            plan = self.plan_dir(rd) / "plan.json"
        args: List[str] = []
        for d in log_dirs:
            args += ["--log-dir", str(d)]
        args += [
            "--out-dir",
            str(out_dir or rd / "stage04"),
            "--dat-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            *self._common(),
            "--progress-every",
            "60",
        ]
        if plan is not None:
            args += ["--plan", str(plan)]
        for bank in self.bank_stems():
            args += ["--extra-library-stem", bank]
        if default_logs and self.contexts:
            args += self.context_log_args(rd)
        if default_logs and self.sustained:
            for d in self.sustained_log_dirs(rd):
                args += ["--sustained-log-dir", str(d)]
        return args

    def planner_model_file(self, step: RoundStep) -> Optional[Path]:
        """The model planned rounds rank with: `planner.model_file`, else the
        model of the bundle the round builds on (`base_bundle`); None means a
        seed bootstrap."""
        explicit = self.get("planner.model_file")
        if explicit:
            return Path(explicit)
        base = self.base_bundle(step)
        return base / "stage05" / "models.pt" if base is not None else None

    def bundle_lineage(self, step: RoundStep) -> List[Path]:
        """The bundles before round `step`, in the order the run deploys
        them (`lib.run_bundles.bundle_key`): its prior rounds and the run's
        final retrain while that is a bundle of the run (lib/run_bundles.py)."""
        return [
            b
            for b in rb.bundles(self.run_dir, step.priors)
            if rb.bundle_key(b)[0] < step.index
        ]

    def base_bundle(self, step: RoundStep) -> Optional[Path]:
        """The bundle directory an active round builds on: the latest of
        `bundle_lineage` with a trained models.pt. stage04b measures its
        models on the round's new GEMMs, stage05 keeps every cell it does not
        retrain from it (`prior_bundles`) and the planner ranks with it. A
        round after a shipped final retrain thus starts from the final
        retrain, not from the last round's bundle. None for another step or
        when no prior has a models.pt."""
        if step.mode != "active" or not step.priors:
            return None
        models = latest_models_dir(self.bundle_lineage(step))
        return models.parent if models is not None else None

    def final_retrain_model_options(
        self, rounds: Sequence[Path]
    ) -> Optional[Dict[str, Any]]:
        """The model-shaping options a final retrain trains with now
        (`FinalRetrain.model_options`, on `rounds`, which they do not
        depend on). A final retrain trained its cells with
        full.final_retrain.recipe, so a round carries them unchanged while
        these options hold, rather than retraining them for differing from
        its own. None, with a warning, when they cannot be told."""
        if not rounds:
            return None
        try:
            return FinalRetrain(self, rounds).model_options()
        except (ConfigError, ValueError, OSError, yaml.YAMLError) as e:
            ui.warn(
                "round",
                f"cannot tell the options a final retrain trains with now ({e}); "
                f"the cells a final retrain trained are compared with this round's",
            )
            return None

    def prior_bundles(self, step: RoundStep) -> List[Path]:
        """stage05's --prior-round-dir of an active round: its prior rounds
        (their splits make the split tree), then its base bundle
        (`base_bundle`) when that is the final retrain, the last of them
        giving the models of the cells the round does not retrain."""
        priors = list(step.priors)
        base = self.base_bundle(step)
        if base is not None and base not in priors:
            priors.append(base)
        return priors

    def _script(self, key: str) -> str:
        return str(self.stages_dir / STAGE_DEFS[key].script)

    def seed_commands(self, rd: Path, step: RoundStep) -> Dict[str, Any]:
        """argv of the seed bootstrap's sub-stages, which the planner runs in
        `seed/` of its directory."""
        seed = self.plan_dir(rd) / "seed"
        cmds: Dict[str, Any] = {
            "stage03": [self._script("stage03")]
            + self.args_stage03(rd, in_dir=seed, out_dir=seed / "stage03"),
            "stage04": [self._script("stage04")]
            + self.args_stage04(
                rd,
                log_dirs=[seed / "stage03" / "logs"],
                out_dir=seed / "stage04",
                plan=seed / "plan.json",
            ),
        }
        if self.get("planner.ranker") == "stage05":
            seed_step = RoundStep(step.index, "initial", [], None, False)
            args = self.args_stage05(rd, seed_step, None)
            args = _replace_arg(args, "--enriched-csv-dir", str(seed / "stage04"))
            args = _replace_arg(args, "--output-dir", str(seed / "stage05"))
            args = _replace_arg(
                args,
                "--min-cell-gemms",
                str(
                    min(
                        int(self.get("train.min_cell_gemms")),
                        int(self.get("planner.seed_per_cell")),
                    )
                ),
            )
            cmds["stage05"] = [self._script("stage05")] + args
        return cmds

    def args_stage02(self, rd: Path, step: RoundStep) -> List[str]:
        args = [
            "--in-dir",
            str(rd / "stage01"),
            "--out-dir",
            str(self.plan_dir(rd)),
            *self._common(),
            "--dat-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--arch",
            self.arch,
            "--weight-dtype",
            self.weight_dtype,
            "--hardware-device",
            str(self.devices[0]),
        ]
        model = self.planner_model_file(step)
        if model is not None:
            args += ["--model-file", str(model)]
            if self.contexts and self.get("contexts.replan"):
                wanted = [
                    {
                        "name": c.label,
                        "cu_budget": c.cu_budget,
                        "schedule": c.schedule,
                        "fraction": c.fraction,
                    }
                    for c in self.contexts
                ]
                args += ["--contexts", json.dumps(wanted)]
        else:
            args += ["--seed-commands", json.dumps(self.seed_commands(rd, step))]
        return args

    def _model_eval_args(self) -> List[str]:
        return [
            *self._common(),
            "--arch",
            self.arch,
            "--hardware-device",
            str(self.devices[0]),
            "--library-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--weight-dtype",
            self.weight_dtype,
        ]

    def rebench_args(self, rd: Optional[Path] = None, bench: bool = True) -> List[str]:
        """Options of validate.rebench_misses for the stages that evaluate
        (stage04b, the stage06 gate, stage08): the run's bench, devices,
        watchdog, pool-only library and context bench, and with `rd` the
        round's bench lines. `bench` adds --bench-binary and --devices."""
        if not self.get("validate.rebench_misses"):
            return []
        args = [
            "--rebench-misses",
            "--rebench-runs",
            str(self.get("validate.rebench_runs")),
        ]
        if bench:
            args += ["--bench-binary", str(self.paths.bench_binary)]
            args += ["--devices", ",".join(str(d) for d in self.devices)]
        args += ["--stall-timeout-s", str(self.get("bench.stall_timeout_s"))]
        args += ["--startup-grace-s", str(self.get("bench.startup_grace_s"))]
        if self.get("bench.pool_only_library"):
            for stem in [self.stem] + self.bank_stems():
                args += ["--pool-only-library", stem]
        if self.contexts:
            args += ["--cotenant-binary", str(context_cotenant(self.cfg, self.paths))]
            for p in self.get("contexts.path_append") or [rocm_bin()]:
                args += ["--path-append", str(p)]
        for y in self.rebench_line_yamls(rd) if rd is not None else []:
            args += ["--rebench-lines", str(y)]
        return args

    def rebench_line_yamls(self, rd: Path) -> List[Path]:
        """The bench yamls of a round's shapes: the round's own and the seed
        bootstrap's."""
        out = [self.shapes_dir(rd) / "shapes.yaml"]
        if self.planner:
            out.append(self.plan_dir(rd) / "seed" / "shapes.yaml")
        return out

    def per_leaf_budget(self, step: Optional[RoundStep] = None) -> int:
        return max(self.shapes_per_cell(step), int(self.get("train.min_cell_gemms")))

    def args_stage04b(self, rd: Path, step: RoundStep, models_dir: Path) -> List[str]:
        args = [
            "--models-dir",
            str(models_dir),
            "--enriched-csv-dir",
            str(rd / "stage04"),
            "--out-dir",
            str(rd / "stage04b"),
            *self._model_eval_args(),
            "--sel-eff-threshold",
            str(self.get("validate.sel_eff_threshold")),
            "--split-floor",
            str(self.get("validate.split_floor")),
            "--split-min-improvement",
            str(self.get("validate.split_min_improvement")),
            "--split-after-attempts",
            str(self.get("validate.split_after_attempts")),
            "--max-split-depth",
            str(self.get("validate.max_split_depth")),
            "--structural-split-min-octaves",
            str(self.get("validate.structural_split_min_octaves")),
            "--structural-split-max-depth",
            str(self.get("validate.structural_split_max_depth")),
            "--round-idx",
            str(step.index),
            "--min-cell-gemms",
            str(self.get("validate.min_cell_gemms")),
            "--train-min-cell-gemms",
            str(self.get("train.min_cell_gemms")),
            "--per-leaf-budget",
            str(self.per_leaf_budget(step)),
            "--pick-measurement",
            str(self.get("validate.pick_measurement")),
            "--gate-frac",
            str(self.get("deploy.gate_frac")),
            "--seed",
            str(self.get("seed.seed")),
        ]
        if self.planner:
            args += ["--plan", str(self.plan_dir(rd) / "plan.json")]
        args += self.rebench_args(rd)
        if self.origami_reference:
            args.append("--origami-reference")
        if step.is_final:
            args.append("--final-round")
        if any(c.holdout for c in self.contexts):
            args += [
                "--context-holdout-dir",
                str(rd / "stage03c" / "holdout" / "stage04"),
            ]
        for p in step.priors:
            args += ["--prior-round-dir", str(p)]
            dec = p / "stage04b" / "decisions.json"
            if dec.is_file():
                args += ["--prior-decisions", str(dec)]
        evaluated = rb.bundle_key(Path(models_dir).parent)
        pre = latest_models_dir(
            b
            for b in rb.bundles(self.run_dir, step.priors)
            if rb.bundle_key(b) < evaluated
        )
        if pre is not None:
            args += ["--prior-models", str(pre / "models.pt")]
        return args + audit_args(self, rd)

    def _catalog_args(self) -> List[str]:
        """stage05 options of the feature catalog: v2 serves with the
        non-temporal rule unless train.non_temporal_rule is false; v3 with the
        soft rules (the family-wise non-temporal rule included) and launch
        keys train switches on, without the trained-key gate when
        train.trained_key_gate is false, with the whitelist tier of
        train.whitelist_tier_k, and trains on served candidates with
        train.served_candidates. The attribute spec reaches stage05 through
        the config file."""
        nt = self.get("train.non_temporal_rule")
        if self.get("train.catalog") != "v3":
            return ["--no-non-temporal-rule"] if nt is False else []
        args = ["--catalog", "v3", "--min-base-cell-gemms",
                str(self.get("train.min_base_cell_gemms"))]  # fmt: skip
        if nt:
            args.append("--non-temporal-rule")
        for key in (
            "non_temporal_family_rule",
            "small_batched_rule",
            "dot2_rule",
            "launch_keys",
            "served_candidates",
        ):
            if self.get(f"train.{key}"):
                args.append("--" + key.replace("_", "-"))
        if not self.get("train.trained_key_gate"):
            args.append("--no-trained-key-gate")
        if not self.get("train.clip_features"):
            args.append("--no-clip-features")
        if self.get("train.whitelist") is not None:
            args += ["--whitelist", str(self.get("train.whitelist"))]
        if self.get("train.whitelist_tier_k"):
            args += ["--whitelist-tier-k", str(self.get("train.whitelist_tier_k"))]
        if self.get("train.min_cu_fraction") is not None:
            args += ["--min-cu-fraction", str(self.get("train.min_cu_fraction"))]
        if self.get("train.schedules"):
            args += ["--schedules", ",".join(self.get("train.schedules"))]
        return args

    def args_stage05(
        self,
        rd: Path,
        step: RoundStep,
        only_cells: Optional[str],
        *,
        data_dirs: Optional[Sequence[Path]] = None,
        out_dir: Optional[Path] = None,
    ) -> List[str]:
        """stage05 of round `rd`: by default on the priors' enriched data and
        the round's own (`training_data_dir`), into `<rd>/stage05`."""
        if data_dirs is None:
            data_dirs = [p / "stage04" for p in step.priors if (p / "stage04").is_dir()]
            data_dirs.append(self.training_data_dir(rd))
        args: List[str] = []
        for d in data_dirs:
            args += ["--enriched-csv-dir", str(d)]
        for p in self.prior_bundles(step) if step.mode == "active" else step.priors:
            args += ["--prior-round-dir", str(p)]
        args += [
            "--output-dir",
            str(out_dir or rd / "stage05"),
            *self._model_eval_args(),
            "--device",
            str(self.get("train.device")),
            "--smart-k",
            str(self.get("train.smart_k")),
            "--epochs",
            str(self.get("train.epochs")),
            "--eval-sample-frac",
            str(self.get("train.eval_sample_frac")),
            "--min-cell-gemms",
            str(self.get("train.min_cell_gemms")),
            "--train-workers",
            str(self.get("train.workers")),
            "--train-threads-per-worker",
            str(self.get("train.threads_per_worker")),
            "--seed",
            str(self.get("seed.seed")),
            "--pick-measurement",
            str(self.get("validate.pick_measurement")),
        ]
        for key in STAGE05_OPTION_KEYS:
            value = self.get(f"train.{key}")
            if value is not None:
                args += ["--" + key.replace("_", "-"), str(value)]
        args += ["--repeat-aggregation", str(self.get("train.repeat_aggregation"))]
        args += self._catalog_args()
        cascade = self.get("train.cascade_k")
        if cascade is not None and (cascade or self.get("train.catalog") == "v3"):
            args += ["--cascade-k", str(cascade)]
        smax = self.get("train.smart_k_max")
        if smax is not None:
            args += [
                "--smart-k-max",
                str(smax),
                "--smart-k-min",
                str(self.get("train.smart_k_min")),
                "--smart-k-saturation-eps",
                str(self.get("train.smart_k_saturation_eps")),
            ]
        args += self.training_plan_args(dict.fromkeys([*step.priors, rd]))
        args += self.off_list_args(rd)
        if step.mode == "active":
            args += ["--current-round-dir", str(rd)]
        if only_cells is not None:
            args += ["--only-cells", only_cells]
        return args

    def with_train(self, overrides: Mapping[str, Any]) -> "Pipeline":
        """This pipeline with the train keys of `overrides` replaced."""
        if not overrides:
            return self
        out = copy.copy(self)
        out.cfg = {
            **self.cfg,
            "train": {**dict(self.cfg.get("train") or {}), **dict(overrides)},
        }
        return out

    def off_list_args(self, rd: Path) -> List[str]:
        """stage05's charge of an off-list pick: train.off_list_sel_eff, with
        auto measured by the round's stage04b re-bench of the previous
        model's misses (train.off_list_fallback without enough of them, as
        in round 0)."""
        value = self.get("train.off_list_sel_eff")
        if value is None:
            return []
        args = ["--off-list-sel-eff", str(value)]
        if value != "auto":
            return args
        from lib.miss_rebench import RECORD_FILE

        args += ["--off-list-min-misses", str(self.get("train.off_list_min_misses"))]
        fallback = self.get("train.off_list_fallback")
        if fallback is not None:
            args += ["--off-list-fallback", str(fallback)]
        return args + ["--off-list-rebench", str(rd / "stage04b" / RECORD_FILE)]

    def training_plan_args(self, rds: Iterable[Path]) -> List[str]:
        """stage05's --plan / --context-plan: the kernel lists the training
        rounds (and their seed benches) were benched from."""
        if not self.planner:
            return []
        args: List[str] = []
        for rd in rds:
            d = self.plan_dir(rd)
            for plan in (d / "plan.json", d / "seed" / "plan.json"):
                if plan.is_file():
                    args += ["--plan", str(plan)]
            if not self.get("contexts.replan"):
                continue
            for ctx in self.contexts:
                replan = d / "contexts" / ctx.label / "plan.json"
                if not ctx.holdout and replan.is_file():
                    args += [
                        "--context-plan",
                        str(ctx.cu_budget),
                        ctx.schedule,
                        str(replan),
                    ]
        return args

    @staticmethod
    def training_data_dir(rd: Path) -> Path:
        """The round's own enriched data stage05 trains on: stage04b's train/
        when it kept a gate slice out, else stage04."""
        train = rd / "stage04b" / "train"
        return train if train.is_dir() else rd / "stage04"

    @staticmethod
    def gate_dir(rd: Path) -> Optional[Path]:
        gate = rd / "stage04b" / "gate"
        return gate if any(gate.glob("chunk_*.csv")) else None

    def args_stage06(
        self,
        rd: Path,
        gate: Optional[GatePlan] = None,
        cell_gate: Optional[CellGatePlan] = None,
    ) -> List[str]:
        args = [
            "--train-dir",
            str(rd / "stage05"),
            *self._common(),
            "--out-dir",
            str(rd / "stage06"),
            "--library-stem",
            self.stem,
            "--weight-dtype",
            self.weight_dtype,
        ]
        if gate is not None:
            args += [
                "--gate-csv-dir",
                str(gate.csv_dir),
                "--hardware-device",
                str(self.devices[0]),
                "--library-dir",
                str(self.paths.library_dir),
                "--pick-measurement",
                str(self.get("validate.pick_measurement")),
            ]
            if gate.target is not None:
                args += ["--gate-target", str(gate.target)]
            if gate.fallback != "none":
                args += ["--gate-fallback", gate.fallback]
            if gate.baseline_train_dir is not None:
                args += ["--gate-baseline-train-dir", str(gate.baseline_train_dir)]
            if self.planner:
                args += ["--gate-plan", str(self.plan_dir(rd) / "plan.json")]
            args += self.rebench_args(rd)
        audit = audit_args(self, rd)
        if audit:
            if gate is None:
                args += [
                    "--hardware-device",
                    str(self.devices[0]),
                    "--library-dir",
                    str(self.paths.library_dir),
                    "--pick-measurement",
                    str(self.get("validate.pick_measurement")),
                ]
            args += audit
            if cell_gate is not None:
                st = cell_gate.settings
                args += [
                    "--cell-gate-baseline",
                    str(cell_gate.baseline_dir),
                    "--cell-tolerance",
                    str(st["cell_tolerance"]),
                    "--cell-floor",
                    str(st["cell_floor"]),
                    "--cell-gate-alpha",
                    str(st["alpha"]),
                    "--tail-floor",
                    str(st["tail_floor"]),
                    "--tail-collapse",
                    str(st["tail_collapse"]),
                    "--tail-budget",
                    str(st["tail_budget"]),
                    "--cell-gate-groups",
                    str(st["groups"]),
                ]
                if cell_gate.record_only:
                    args.append("--cell-gate-record-only")
        if self.origami_reference:
            args.append("--origami-reference")
        if self.paths.deploy_root is not None:
            args += ["--rocm-libraries-to-deploy", str(self.paths.deploy_root)]
            if self.get("deploy.apply_patch"):
                args.append("--apply-patch")
                if self.get("deploy.build_after_apply"):
                    args.append("--build-after-apply")
                    if self.paths.deploy_build_dir is not None:
                        args += ["--build-dir", str(self.paths.deploy_build_dir)]
                    jobs = self.get("deploy.build_jobs")
                    if jobs is not None:
                        args += ["--build-jobs", str(jobs)]
        return args

    def args_stage00(self) -> List[str]:
        devices = self.get("correctness.devices") or self.devices
        args = [
            *self._common(),
            "--out-dir",
            str(correctness_attempt_dir(self.run_dir)),
            "--dat-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--hardware-device",
            str(self.devices[0]),
            "--devices",
            ",".join(str(d) for d in devices),
            "--cache-dir",
            str(self.get("correctness.cache_dir")),
            "--build-dir",
            str(self.paths.build_dir),
            "--lease-wait-s",
            str(self.get("runtime.gpu_lease_wait_s")),
        ]
        if self.quarantine is not None:
            args += ["--quarantine", str(self.quarantine)]
        return args

    def args_stage00a(self) -> List[str]:
        devices = self.get("hazards.devices") or self.devices[:1]
        return [
            *self._common(),
            "--out-dir",
            str(hazard_canary_attempt_dir(self.run_dir)),
            "--dat-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--bench-binary",
            str(self.paths.bench_binary),
            "--build-dir",
            str(self.paths.build_dir),
            "--hardware-device",
            str(self.devices[0]),
            "--devices",
            ",".join(str(d) for d in devices),
            "--cache-dir",
            str(self.get("hazards.cache_dir")),
            "--lease-wait-s",
            str(self.get("runtime.gpu_lease_wait_s")),
            "--startup-grace-s",
            str(self.get("bench.startup_grace_s")),
        ]

    def hazard_canary(self) -> int:
        """The hazard canary (stage00a) of this invocation, before the
        correctness gate and any other stage that launches the pool's
        kernels, when `hazards.enabled`; it leases GPUs only for kernels
        without a verdict. It writes into `canary.new`, which replaces
        `canary` when it succeeds; its quarantine goes into the invocation's
        blocklist (`correctness_gate`). A pool that keeps quarantined
        kernels needs the planner: without it a bench line runs every kernel
        of the library. Without a kernel of a hazard class in the pool
        (`canary_needed`) it runs nothing."""
        if not self.canary:
            return 0
        final = hazard_canary_dir(self.run_dir)
        fresh = hazard_canary_attempt_dir(self.run_dir)
        rc = self.run_stage(
            final.parent, "stage00a", self.args_stage00a(), None, out_dir=fresh
        )
        if rc:
            if rc == 3:
                ui.err(
                    "canary",
                    f"the hazard canary stopped ({fresh / hazlib.SUMMARY_FILE}): "
                    f"its verdicts are recorded in {self.get('hazards.cache_dir')}; "
                    f"reset the device, then run the same command to go on",
                )
            return rc
        move_aside(final.parent, final)
        fresh.rename(final)
        self.quarantine = final / hazlib.QUARANTINE_FILE
        try:
            held = cxlib.read_list(self.quarantine)
        except ValueError as e:
            ui.err("canary", str(e))
            return 1
        if held and not self.planner:
            ui.err(
                "canary",
                f"{len(held)} kernel(s) of the pool are quarantined "
                f"({self.quarantine}), but without the planner a bench line runs "
                f"every kernel of the library: set planner.enabled",
            )
            return 2
        return 0

    def correctness_gate(self) -> int:
        """The correctness gate (stage00) of this invocation, before any
        stage reads the pool, when `correctness.enabled`; it leases GPUs only
        for kernels without a verdict, and leaves the hazard canary's
        quarantine unchecked (`--quarantine`), as a list it blocks. It writes
        into `gate.new`, which replaces `gate` only when it succeeds, so a
        gate that fails leaves the last one's outputs in place. The blocklist
        the invocation's config names (`correctness.blocklist`) is then a
        copy of `gate`'s, also after a failed gate, so that the snapshot
        stays readable, and the stage markers of the invocation record it
        (`mark_complete`). Without the gate that blocklist is a copy of the
        canary's quarantine."""
        if not self.get("correctness.enabled"):
            dst = self.get("correctness.blocklist")
            if not dst:
                return 0
            blocked = cxlib.Blocklist()
            if self.quarantine is not None:
                try:
                    blocked = cxlib.read_list(self.quarantine)
                except ValueError as e:
                    ui.err("canary", str(e))
                    return 1
            dst = Path(str(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            cxlib.write_blocklist(dst, blocked, source=hazlib.SOURCE)
            self.blocklist = cxlib.blocklist_record(dst)
            return 0
        gate = correctness_gate_dir(self.run_dir)
        fresh = correctness_attempt_dir(self.run_dir)
        rc = self.run_stage(
            gate.parent, "stage00", self.args_stage00(), None, out_dir=fresh
        )
        if rc == 0:
            move_aside(gate.parent, gate)
            fresh.rename(gate)
        src = gate / cxlib.BLOCKLIST_FILE
        dst = Path(str(self.get("correctness.blocklist")))
        if not src.is_file():
            if rc == 0:
                ui.err("gate", f"stage00 succeeded but wrote no {src}")
                return 1
            return rc
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        try:
            self.blocklist = cxlib.blocklist_record(dst)
        except ValueError as e:
            ui.err("gate", str(e))
            return rc or 1
        if rc:
            ui.warn(
                "gate",
                f"the gate failed; {dst} holds the blocklist of the last gate "
                f"that succeeded ({self.blocklist['kernels']} kernel(s))",
            )
        return rc

    # ── one training round ──

    def run_round(self, step: RoundStep) -> RoundResult:
        rd = self.run_dir / f"round_{step.index}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / ROUND_MARKER).unlink(missing_ok=True)
        from_stage = (
            step.from_stage
            if step.from_stage is not None
            else self.first_incomplete_stage(rd, step.mode)
        )
        ui.banner(
            f"round_{step.index}  mode={step.mode}  from_stage={from_stage:g}"
            + ("  (final round)" if step.is_final else ""),
            ui.C.HEAD,
        )
        if step.mode == "active":
            if not step.priors:
                ui.err("round", "an active round needs at least one prior round")
                return RoundResult(2)
            ui.info("round", f"priors: {[p.name for p in step.priors]}")
        if step.targets is not None:
            ui.info("round", f"enhance cells: {step.targets}")
            record = rd / ENHANCE_RECORD
            if not record.is_file():
                record.write_text(
                    json.dumps(_json_safe(enhance_record(step)), indent=2) + "\n"
                )
        t_round = time.time()
        result = self._run_round_stages(rd, step, from_stage)
        self.timing.add(step.index, "(round total)", result.rc, time.time() - t_round)
        if result.rc == 0:
            marker = {
                "round": step.index,
                "mode": "enhance" if step.targets is not None else step.mode,
                "deployed": result.deployed,
                "held_out_sel_eff": result.held_out_sel_eff,
                "gate_sel_eff": result.gate_sel_eff,
                "finished": datetime.now().isoformat(timespec="seconds"),
                "bytes": round_bytes(rd),
            }
            carried = _read_json(rd / "stage05" / CARRIED_CELLS) or {}
            if carried.get("cells"):
                marker["cells_carried"] = [c["cell"] for c in carried["cells"]]
            (rd / ROUND_MARKER).write_text(
                json.dumps(_json_safe(marker), indent=2) + "\n"
            )
            ui.ok("round", f"round_{step.index} complete")
        else:
            ui.err("round", f"round_{step.index} FAILED (rc={result.rc})")
        return result

    def _stage_or_skip(
        self,
        rd: Path,
        key: str,
        from_stage: float,
        args_fn: Callable[[], List[str]],
        round_idx: int,
    ) -> int:
        if STAGE_DEFS[key].number < from_stage:
            return 0 if self._require_complete(rd, key) else 2
        return self.run_stage(rd, key, args_fn(), round_idx)

    def _run_round_stages(
        self, rd: Path, step: RoundStep, from_stage: float
    ) -> RoundResult:
        idx = step.index
        rc = self._stage_or_skip(
            rd, "stage01", from_stage, lambda: self.args_stage01(rd, step), idx
        )
        if rc:
            return RoundResult(exit_code(rc))

        if self.planner:
            rc = self._stage_or_skip(
                rd, "stage02", from_stage, lambda: self.args_stage02(rd, step), idx
            )
            if rc:
                return RoundResult(exit_code(rc))
        elif from_stage <= STAGE_DEFS["stage02"].number:
            name = self.stage_name("stage02")
            move_aside(rd, rd / name, rd / f"{name}.log")

        if from_stage <= max(STAGE_DEFS[k].number for k in BENCH_STAGES):
            errors = self.plan_identity_errors(rd)
            if errors:
                _print_problems("identity", errors, [])
                return RoundResult(2)
        bench_stages = [("stage03", lambda: self.args_stage03(rd))]
        if self.rebench:
            bench_stages.append(("stage03b", lambda: self.args_stage03b(rd)))
        elif from_stage <= 3.2:
            move_aside(rd, rd / "stage03b", rd / "stage03b.log")
        if self.contexts:
            bench_stages.append(("stage03c", lambda: self.args_stage03c(rd)))
        elif from_stage <= 3.5:
            move_aside(rd, rd / "stage03c", rd / "stage03c.log")
        if self.sustained:
            bench_stages.append(("stage03d", lambda: self.args_stage03d(rd)))
        elif from_stage <= 3.7:
            move_aside(rd, rd / "stage03d", rd / "stage03d.log")
        bench_stages.append(("stage04", lambda: self.args_stage04(rd)))
        for key, fn in bench_stages:
            rc = self._stage_or_skip(rd, key, from_stage, fn, idx)
            if rc:
                return RoundResult(exit_code(rc))

        retrain: Optional[str] = None
        if step.mode == "active":
            base = self.base_bundle(step)
            if base is None:
                ui.err("round", "no prior round has a trained models.pt")
                return RoundResult(2)
            models_dir = base / "stage05"
            if round_index(base) is None:
                ui.info(
                    "round",
                    f"{rd.name} builds on the shipped {base.name} "
                    f"({base / 'stage05'}), not on {step.priors[-1].name}'s bundle",
                )
            rc = self._stage_or_skip(
                rd,
                "stage04b",
                from_stage,
                lambda: self.args_stage04b(rd, step, models_dir),
                idx,
            )
            if rc:
                return RoundResult(exit_code(rc))
            path = rd / "stage04b" / "retrain_cells.txt"
            retrain = path.read_text().strip() if path.is_file() else ""
            if from_stage <= 5.0:
                if step.targets is not None:
                    retrain = self._enhance_retrain(rd, step, retrain)
                retrain = self._retrain_carried(rd, step, retrain)

        if from_stage > 5.0:
            if not self._require_complete(rd, "stage05"):
                return RoundResult(2)
        elif retrain == "":
            rc = self._carry_forward(rd, step)
            if rc:
                return RoundResult(rc)
        else:
            args = self.args_stage05(rd, step, retrain)
            carry = self.get("train.on_cell_failure") == "carry"
            resumed = step.from_stage is None and from_stage == 5.0
            if carry and resumed and self._untrained_bundle_matches(rd, args):
                ui.info(
                    "carry",
                    f"{rd / 'stage05'} holds the bundle stage 5 wrote with these "
                    f"arguments when it could not train every cell; going on with it",
                )
                rc = STAGE05_CELLS_NOT_TRAINED
            else:
                rc = self.run_stage(rd, "stage05", args, idx)
                if rc == STAGE05_CELLS_NOT_TRAINED:
                    self._record_untrained_bundle(rd, args)
            if rc == STAGE05_CELLS_NOT_TRAINED:
                if carry:
                    rc = self._carry_untrained_cells(rd, args)
                else:
                    self._report_untrained_cells(rd)
            if rc:
                return RoundResult(exit_code(rc))
            self.record_origins(rd, step)

        return self._maybe_deploy(rd, step, from_stage)

    def record_origins(self, rd: Path, step: RoundStep) -> None:
        """`record_cell_origins` of the bundle an active round's stage05
        wrote, which carried the cells it did not train from the bundles
        before it (`bundle_lineage`)."""
        if step.mode == "active":
            record_cell_origins(rd / "stage05", self.bundle_lineage(step))

    def _untrained_bundle_key(self, args: List[str]) -> Dict[str, Any]:
        """What makes a stage05 bundle this invocation's: the arguments (the
        config snapshot aside, whose pinned keys the pin covers) and the
        pin."""
        return {
            "args": _replace_arg([str(a) for a in args], "--config-yaml", "<config>"),
            "pin": pinning.summary(self.pin) if self.pin is not None else None,
        }

    def _record_untrained_bundle(self, rd: Path, args: List[str]) -> None:
        rec = {
            **self._untrained_bundle_key(args),
            "finished": datetime.now().isoformat(timespec="seconds"),
        }
        (rd / "stage05" / UNTRAINED_BUNDLE).write_text(
            json.dumps(_json_safe(rec), indent=2) + "\n"
        )

    def _untrained_bundle_matches(self, rd: Path, args: List[str]) -> bool:
        """stage05 exited 3 with these arguments and this pin, and its
        outputs are still in place."""
        rec = _read_json(rd / "stage05" / UNTRAINED_BUNDLE)
        if not isinstance(rec, dict):
            return False
        key = self._untrained_bundle_key(args)
        return all(rec.get(k) == v for k, v in _json_safe(key).items())

    def _report_untrained_cells(self, rd: Path) -> None:
        metrics_path = rd / "stage05" / "metrics.json"
        metrics = _read_json(metrics_path) or {}
        failed = list(metrics.get("cells_failed") or [])
        missing = [str(r.get("cell")) for r in metrics.get("cells_not_trained") or []]
        ui.err(
            "round",
            f"stage05 did not train every requested cell (failed: {failed or '-'}; "
            f"not trained: {missing or '-'}; see {metrics_path}). {rd.name} stops "
            f"before deployment: fix the cause and run the same command to rerun "
            f"stage 5, add --from-stage 4.5 to redo the decisions, or set "
            f"train.on_cell_failure: carry and run the same command to go on "
            f"with the bundle stage 5 wrote.",
        )

    def _carry_untrained_cells(self, rd: Path, args: List[str]) -> int:
        """train.on_cell_failure carry, after stage05 exited 3 with every
        output written: the round goes on with that bundle, in which a cell
        it failed or could not train keeps the model an earlier round
        trained for it, or has none of its own. The cells are logged and
        recorded in stage05/cells_carried.json and in the round's marker.
        3 when stage05 wrote no bundle, trained none of the cells it was to
        train or left more than train.carry_max_fraction of them; 2 when
        the code changed meanwhile."""
        s5 = rd / "stage05"
        metrics = _read_json(s5 / "metrics.json")
        cells = _read_json(s5 / "cells.json")
        if not (s5 / "models.pt").is_file() or not metrics or cells is None:
            ui.err("carry", f"{s5} holds no complete bundle to go on with")
            self._report_untrained_cells(rd)
            return STAGE05_CELLS_NOT_TRAINED
        failures = dict(metrics.get("failures") or {})
        why = {
            str(c): failures.get(c) or "failed"
            for c in metrics.get("cells_failed") or []
        }
        for row in metrics.get("cells_not_trained") or []:
            why[str(row.get("cell"))] = f"not trained: {row.get('reason')}"
        n_trained = int(metrics.get("n_cells_trained") or 0)
        limit = float(self.get("train.carry_max_fraction"))
        if not n_trained or len(why) > limit * (n_trained + len(why)):
            ui.err(
                "carry",
                f"stage05 trained {n_trained} of the {n_trained + len(why)} cells "
                f"it was to train: train.on_cell_failure: carry goes on without "
                f"at most {limit:.0%} of them (train.carry_max_fraction), and "
                f"never without all",
            )
            self._report_untrained_cells(rd)
            return STAGE05_CELLS_NOT_TRAINED
        drift = self.pin_drift()
        if drift:
            self._report_drift("stage05", drift, "is not marked complete")
            return 2
        models = set(cells.get("model_labels") or [])
        carried = [
            {
                "cell": c,
                "why": why[c],
                "model": (
                    "the model of an earlier round"
                    if c in models
                    else "none of its own (served by its nearest trained cell)"
                ),
            }
            for c in sorted(why)
        ]
        record = {"policy": "carry", "n_cells_trained": n_trained, "cells": carried}
        (s5 / CARRIED_CELLS).write_text(json.dumps(record, indent=2) + "\n")
        for c in carried:
            ui.warn("carry", f"{c['cell']}: {c['why']}; keeps {c['model']}")
        n = len(carried)
        ui.warn(
            "carry",
            f"{rd.name} goes on with {n} cell{'s' if n != 1 else ''} stage 5 did "
            f"not train (train.on_cell_failure: carry; {s5 / CARRIED_CELLS})",
        )
        self.mark_complete(
            s5,
            self.stage_cmd("stage05", args),
            note=f"{n} cell(s) not trained, carried ({CARRIED_CELLS})",
        )
        return 0

    def _enhance_retrain(self, rd: Path, step: RoundStep, listed: str) -> str:
        """Cells an enhance round retrains: the requested leaves (children
        replacing a leaf stage04b split) that have enough training GEMMs,
        plus stage04b's retrain list inside them. stage04b entries outside
        the requested cells wait for an active round. Recorded in
        enhance.json."""
        min_gemms = int(self.get("train.min_cell_gemms"))
        n_train = self._training_gemms(rd)
        tree = sc.load_cumulative_split_tree([*step.priors, rd])
        leaves, _ = ce.requested_leaves(step.targets or [], tree)
        children = {
            c
            for rule in sc.load_cumulative_split_tree([rd]).values()
            for c in (rule.lo_label, rule.hi_label)
        }
        cells: List[str] = []
        too_few: List[str] = []
        for leaf in leaves:
            if leaf in children or n_train.get(leaf, 0) >= min_gemms:
                cells.append(leaf)
            else:
                too_few.append(leaf)
        deferred: List[str] = []
        for c in [c for c in listed.split(",") if c]:
            if not ce.under_any(c, step.targets or []):
                deferred.append(c)
            elif c not in cells:
                cells.append(c)
        if too_few:
            ui.warn(
                "enhance",
                f"fewer than {min_gemms} training GEMMs, not retrained: {too_few}",
            )
        if deferred:
            ui.warn(
                "enhance",
                f"stage04b also lists cells outside the enhance selection; an "
                f"active round retrains them: {deferred}",
            )
        ui.info("enhance", f"retraining {cells or 'nothing'}")
        record = rd / ENHANCE_RECORD
        data = _read_json(record) or enhance_record(step)
        data.update({"retrain": cells, "too_few_gemms": too_few, "deferred": deferred})
        record.write_text(json.dumps(_json_safe(data), indent=2) + "\n")
        return ",".join(cells)

    @staticmethod
    def _training_gemms(rd: Path) -> Dict[str, int]:
        """Training GEMMs per cell over every round's data, as the round's
        stage04b counted them."""
        decisions = _read_json(rd / "stage04b" / "decisions.json") or {}
        return {
            str(r.get("cell")): int(r.get("n_training_gemms") or 0)
            for r in decisions.get("per_cell") or []
        }

    def _retrain_carried(self, rd: Path, step: RoundStep, listed: str) -> Optional[str]:
        """`listed` plus the cells the round would carry forward that were
        trained with other model-shaping options than its stage05's
        (`option_changes`); None, every cell, when that is every carried
        cell. A carried cell with fewer than train.min_cell_gemms training
        GEMMs stays as it is."""
        cells = [c for c in listed.split(",") if c]
        tree = sc.load_cumulative_split_tree([*step.priors, rd])
        lineage = self.bundle_lineage(step)
        carried = carried_cells(lineage, cells, tree)
        s5 = stage05_module()
        options = s5.command_model_options(self.args_stage05(rd, step, listed))
        origins = rb.cell_origins(lineage)
        final = None
        if any(
            round_index(Path(origins[c]["trained_in"])) is None
            for c in carried
            if c in origins
        ):
            final = self.final_retrain_model_options(step.priors)
        changed = option_changes(lineage, carried, options, final)
        if len(changed) < len(carried):
            min_gemms = int(self.get("train.min_cell_gemms"))
            n_train = self._training_gemms(rd)
            too_few = [c for c in changed if n_train.get(c, 0) < min_gemms]
            if too_few:
                ui.warn(
                    "round",
                    f"fewer than {min_gemms} training GEMMs, not retrained "
                    f"although trained with other options: {too_few}",
                )
            changed = {c: d for c, d in changed.items() if c not in too_few}
        if not changed:
            return listed
        n = len(changed)
        ui.info(
            "round",
            f"{rd.name} retrains {n} carried cell{'s' if n != 1 else ''}: "
            f"{describe_option_changes(changed)}",
        )
        if n == len(carried):
            return None
        return ",".join(cells + list(changed))

    def _carry_forward(self, rd: Path, step: RoundStep) -> int:
        src = latest_models_dir(self.prior_bundles(step))
        if src is None:
            ui.err("carry", "every cell passed but no prior round has models.pt")
            return 2
        splits = _read_json(rd / "stage04b" / "splits.json") or {}
        if splits.get("splits"):
            ui.err(
                "carry",
                "stage04b split cells but listed none to retrain; refusing to "
                "carry the unsplit model forward",
            )
            return 2
        dst = rd / "stage05"
        move_aside(rd, dst, rd / "stage05.log")
        dst.mkdir(parents=True)
        s5 = stage05_module()
        for name in (
            "models.pt",
            "cells.json",
            s5.MODEL_OPTIONS_FILE,
            s5.MEMBER_ORDER_FILE,
        ):
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
        (dst / rb.CARRIED_FORWARD).write_text(
            json.dumps({"source": str(src)}, indent=2) + "\n"
        )
        self.mark_complete(dst, None, note=f"carried forward from {src}")
        self.record_origins(rd, step)
        ui.ok("carry", f"every cell passed; carried {src} forward to {dst}")
        return 0

    def deploy_gate(self, rd: Path, step: RoundStep) -> Tuple[bool, Optional[GatePlan]]:
        """(stage06 runs, its gate) for a round. A round ships the bundle it
        trained only after measuring it on its gate GEMMs (stage04b gate/):
        an active round deploys when the bundle reaches
        deploy.target_sel_eff there; the final round with
        deploy.on_final_round also deploys below the target when nothing was
        deployed before or the bundle is not worse than the deployed one
        (a round's, or the run's final retrain) on the same GEMMs. --deploy
        deploys every round, measuring when it can."""
        target = self.get("deploy.target_sel_eff")
        gate = self.gate_dir(rd) if step.mode == "active" else None
        if self.force_deploy:
            return True, GatePlan(gate) if gate is not None else None
        if step.mode == "initial":
            ui.info("deploy", "round_0 deploys only with --deploy")
            return False, None
        if step.is_final and self.get("deploy.on_final_round"):
            if gate is None:
                ui.warn(
                    "deploy",
                    f"{rd.name} has no gate GEMMs (deploy.gate_frac, or too few new "
                    f"GEMMs): the final bundle deploys without being measured",
                )
                return True, None
            prior = latest_deploy_record(rb.bundles(self.run_dir, step.priors))
            baseline = None
            if prior is not None:
                train = (_read_json(prior) or {}).get("train_dir")
                baseline = (
                    Path(str(train)) if train else prior.parent.parent / "stage05"
                )
            return True, GatePlan(gate, target, "not-worse", baseline)
        if step.targets is not None or target is None:
            return False, None
        if gate is None:
            ui.info(
                "deploy",
                f"{rd.name} has no gate GEMMs: its bundle cannot be measured and "
                f"is not deployed",
            )
            return False, None
        return True, GatePlan(gate, target)

    def cell_gate_plan(self, step: RoundStep) -> Optional[CellGatePlan]:
        """The per-cell gate of the round's stage06: against the bundle the
        run ships before the round (`shipped_round` of its priors and a
        final retrain that is a bundle of the run), on the audit set, with
        the config's gate settings; recording only with --deploy. None
        without audit.dir, with gate.per_cell off, or while the run ships
        no bundle."""
        settings = auditlib.cell_gate_settings(self.cfg)
        if settings is None or auditlib.config_dir(self.cfg) is None:
            return None
        ships = shipped_round(rb.bundles(self.run_dir, step.priors))
        if ships is None:
            return None
        return CellGatePlan(ships, settings, record_only=self.force_deploy)

    def _maybe_deploy(
        self, rd: Path, step: RoundStep, from_stage: float
    ) -> RoundResult:
        held = float("nan")
        if step.mode == "active":
            held = held_out_sel_eff(rd / "stage04b" / "decisions.json")
        target = self.get("deploy.target_sel_eff")
        self.audit_v2_round(rd, step)
        if not math.isnan(held):
            ui.info(
                "deploy",
                f"the previous bundle on this round's held-out GEMMs: "
                f"{held * 100:.2f}% (stage04b)",
            )
        should, gate = self.deploy_gate(rd, step)
        if not should:
            self.audit_candidate(rd)
            return RoundResult(0, held_out_sel_eff=held)
        cells = self.cell_gate_plan(step)
        ui.info(
            "deploy",
            (
                "forced"
                if self.force_deploy
                else (
                    "measuring this round's bundle on its gate GEMMs (target "
                    f"{'none' if target is None else f'{target * 100:.2f}%'})"
                    if gate is not None
                    else "final round"
                )
            ),
        )
        rc = self.run_stage(
            rd, "stage06", self.args_stage06(rd, gate, cells), step.index
        )
        if rc:
            return RoundResult(exit_code(rc), held_out_sel_eff=held)
        rec = _read_json(rd / "stage06" / "deploy_record.json") or {}
        deployed = rec.get("deployed", True) is not False
        gate_se = _f((rec.get("gate") or {}).get("sel_eff"))
        reached = (
            deployed
            and step.targets is None
            and target is not None
            and gate_se is not None
            and gate_se >= target
        )
        if not deployed:
            ui.warn("deploy", f"{rd.name} not deployed: {rec.get('not_deployed')}")
        return RoundResult(
            0,
            held_out_sel_eff=held,
            gate_sel_eff=float("nan") if gate_se is None else gate_se,
            deployed=deployed,
            target_reached=reached,
        )

    def args_audit_eval(self, rd: Path) -> List[str]:
        return [
            "--round-dir",
            str(rd),
            "--out",
            str(rd / "stage05" / auditlib.RESULT_FILE),
            *self.audit_eval_common(),
        ]

    def audit_eval_common(self) -> List[str]:
        """scripts/audit_eval.py options of the run: its audit set, config
        snapshot, library, device, weight dtype, target, bootstrap and
        official protocol."""
        st = auditlib.settings(self.cfg)
        return [
            "--audit-dir",
            str(auditlib.config_dir(self.cfg)),
            "--config",
            str(self.snapshot),
            "--library-dir",
            str(self.paths.library_dir),
            "--hardware-device",
            str(self.devices[0]),
            "--weight-dtype",
            self.weight_dtype,
            "--target",
            str(st["target"]),
            "--bootstrap",
            str(st["bootstrap"]),
            "--official",
            str(self.get("audit.official")),
        ]

    def run_tool(
        self,
        owner_dir: Path,
        name: str,
        cmd: List[str],
        round_idx: Optional[int],
        devices: Sequence[int] = (),
        per_gpu: bool = False,
    ) -> int:
        """Run a pipeline script that is not a stage (`cmd`, logged to
        `<owner_dir>/<name>.log`) under the lease of `devices`, as
        `run_stage` runs a stage; `per_gpu` when its benches run through
        stage 3 (`wait_gpu_health`)."""
        t0 = time.time()
        label = f"{owner_dir.name} {name}"
        try:
            with gpu_lease.hold_usable(
                list(devices),
                label,
                self.run_dir,
                wait_s=float(self.get("runtime.gpu_lease_wait_s")),
                report=lambda msg: ui.warn("lease", f"{label}: {msg}"),
            ) as leases:
                if not self.wait_gpu_health(leases.devices, label, per_gpu):
                    ui.err(
                        "health",
                        f"{name} not started: the run's GPUs fail their checks",
                    )
                    rc = 2
                else:
                    if leases.acquired:
                        where = gpu_lease.lease_dir()
                        ui.info("lease", f"GPUs {leases.acquired} leased ({where})")
                    rc = run_logged(
                        cmd,
                        owner_dir / f"{name}.log",
                        env=self.stage_env(),
                        running=self.running,
                        label=label,
                        pass_fds=leases.fds,
                    )
        except gpu_lease.LeaseError as e:
            ui.err("lease", f"{name} not started: {e}")
            rc = 2
        self.timing.add(round_idx, name, rc, time.time() - t0)
        return rc

    def v2_name(self, owner_dir: Path) -> str:
        """The protocol v2 name of the bundle trained in `owner_dir` (a
        round or the final retrain); audit_v2.py adds its sha256."""
        return f"{self.run_dir.name}.{owner_dir.name}"

    def audit_v2(
        self,
        owner_dir: Path,
        models: Sequence[Tuple[str, Path]],
        round_idx: Optional[int],
    ) -> Optional[int]:
        """Protocol v2 (lib/audit_v2.py) of `models` ((name, models.pt)) on
        the audit set: `scripts/audit_v2.py run` registers them and benches,
        on the run's devices under their leases, the audit GEMMs whose v2
        sets lack a pick of theirs (generation 0 when the audit set has no
        v2 measurement yet), then, with audit.sustained, the sustained
        composite's power-capped exclusive GEMMs that lack one
        (lib/audit_sustained.py). None when the audit set is not complete
        for the current pool (the end of a full run tries again); a failure
        is reported, never fatal."""
        d = auditlib.config_dir(self.cfg)
        if d is None or self.get("audit.v2") == "off" or not models:
            return 0
        names = ", ".join(name for name, _ in models)
        st, why = audit_status(self.cfg, self.paths)
        if st != "ok":
            ui.info("audit", f"protocol v2 of {names} waits for the audit set: {why}")
            return None
        wait = float(self.get("runtime.gpu_lease_wait_s"))
        cmd = [sys.executable, "-u", str(AUDIT_V2_SCRIPT), "run"]
        cmd += ["--audit-dir", str(d), "--config", str(self.snapshot)]
        cmd += ["--devices", ",".join(str(x) for x in self.devices)]
        cmd += ["--wait", str(wait), "--name-with-sha"]
        if not self.get("audit.sustained"):
            cmd.append("--no-sustained")
        for name, path in models:
            cmd += ["--model", f"{name}={path}"]
        ui.banner(f"audit v2 -- {names} on the audit set", ui.C.BLUE)
        rc = self.run_tool(
            owner_dir, "audit_v2", cmd, round_idx, self.devices, per_gpu=True
        )
        if rc:
            ui.warn(
                "audit",
                f"protocol v2 of {names} failed (rc={rc}); see "
                f"{owner_dir / 'audit_v2.log'}",
            )
        return rc

    def audit_v2_round(self, rd: Path, step: RoundStep) -> None:
        """With audit.v2 rounds, protocol v2 of the round's bundle before
        stage06 (or the stage05 audit evaluation) scores it, so that its
        audit record has v2."""
        models = rd / "stage05" / "models.pt"
        if self.get("audit.v2") == "rounds" and models.is_file():
            self.audit_v2(rd, [(self.v2_name(rd), models)], step.index)

    def round_audited(self, rd: Path, v2: bool) -> bool:
        """The round's bundle has a complete audit record (with a complete
        v2 record when `v2`, and then a complete sustained composite with
        audit.sustained): its stage06 (else stage05) audit.json, or the one
        the end of a run wrote for the same model file
        (audit_results/<round>.json)."""
        repeated = ["v2", "sustained"] if self.get("audit.sustained") else ["v2"]

        def complete(rec: Any) -> bool:
            return (
                isinstance(rec, dict)
                and rec.get("status") == "ok"
                and all(
                    auditlib.protocol_record(rec, p).get("status") == "ok"
                    for p in (repeated if v2 else [])
                )
            )

        own = None
        for st in ("stage06", "stage05"):
            if stage_complete(rd / st):
                own = _read_json(rd / st / auditlib.RESULT_FILE)
                if isinstance(own, dict):
                    break
        if complete(own):
            return True
        later = _read_json(self.run_dir / AUDIT_RESULTS_DIR / f"{rd.name}.json")
        if not complete(later):
            return False
        return not isinstance(own, dict) or (
            later.get("model_sha256") == own.get("model_sha256")
        )

    def rescore_rounds(self, rds: Sequence[Path]) -> int:
        """scripts/audit_eval.py of the bundles of `rds` on the audit set as
        it is now: rounds' candidates, and the final retrain's models.pt
        leaving out the shapes of the rounds it trained on; one record per
        bundle in audit_results/<round or final_retrain>.json (the report
        prefers the more complete record of a bundle)."""
        out_dir = self.run_dir / AUDIT_RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        scores = out_dir / "rescore.json"
        rounds = [rd for rd in rds if round_index(rd) is not None]
        finals = [d for d in rds if round_index(d) is None]
        cmd = [sys.executable, "-u", str(AUDIT_EVAL_SCRIPT)]
        for rd in rounds:
            cmd += ["--round-dir", str(rd)]
        for d in finals[:1]:
            cmd += ["--model-file", str(d / "stage05" / "models.pt")]
            for name in rb.trained_rounds(d):
                cmd += ["--trained-round-dir", str(self.run_dir / name)]
        cmd += ["--out", str(scores), *self.audit_eval_common()]
        names = [rd.name for rd in rounds] + [d.name for d in finals[:1]]
        ui.banner(
            f"audit -- {', '.join(names)} on the audit set as it is now", ui.C.BLUE
        )
        rc = self.run_tool(out_dir, "audit_eval", cmd, None)
        doc = _read_json(scores)
        if rc or not isinstance(doc, dict):
            ui.warn("audit", f"audit_eval failed (rc={rc}); see {out_dir}")
            return rc or 1
        recs = doc.get("models") if len(names) > 1 else [doc]
        for name, rec in zip(names, recs or []):
            if isinstance(rec, dict):
                path = out_dir / f"{name}.json"
                path.write_text(json.dumps(dict(rec, what=name), indent=2) + "\n")
        scores.unlink()
        return 0

    def audit_wrapup(self, rounds: Sequence[Path], final: Optional[Path]) -> None:
        """The end of a full run on a complete audit set: protocol v2 of the
        bundles that lack it (every round's with audit.v2 rounds, the final
        round's and the shipped one's with final, and unless audit.v2 is off
        a shipped final retrain's, whose v2 bench failed when it was
        decided) and of the undecided final retrain `final` (its models.pt),
        then the bundles whose records are not complete scored again
        (`rescore_rounds`)."""
        d = auditlib.config_dir(self.cfg)
        if d is None or not rounds:
            return
        st, why = audit_status(self.cfg, self.paths)
        if st != "ok":
            ui.warn(
                "audit",
                f"the audit set {d} is not complete ({why}): the rounds' audit "
                f"records stay as they are",
            )
            return
        mode = self.get("audit.v2")
        ends = {rounds[-1], shipped_round(rounds) or rounds[-1]}
        wanted = {
            rd: mode == "rounds" or (mode == "final" and rd in ends) for rd in rounds
        }
        todo = [rd for rd in rounds if not self.round_audited(rd, wanted[rd])]
        shipped = rb.final_retrain(self.run_dir)
        if shipped is not None and not self.round_audited(shipped, mode != "off"):
            wanted[shipped] = mode != "off"
            todo.append(shipped)
        models = [
            (self.v2_name(rd), rd / "stage05" / "models.pt")
            for rd in todo
            if wanted[rd] and (rd / "stage05" / "models.pt").is_file()
        ]
        if final is not None and mode != "off" and final.is_file():
            models.append((self.v2_name(final.parent.parent), final))
        if models:
            self.audit_v2(self.run_dir, models, None)
        if todo:
            self.rescore_rounds(todo)

    def end_of_run_waits(self, n_done: int, total: int) -> List[str]:
        """What waits for the planned rounds when a full run stops early
        (deploy.stop_when_target_reached) after `n_done` of `total`: the
        final retrain and the audit records the end of the run completes
        (`finish_run`)."""
        what = []
        if self.get("full.final_retrain.enabled"):
            what.append("the final retrain (full.final_retrain)")
        if auditlib.config_dir(self.cfg) is not None:
            what.append("the rounds' audit records completed at the end of the run")
        if not what:
            return []
        verb = "wait" if len(what) > 1 else "waits"
        return [
            f"{' and '.join(what)} {verb} for the planned rounds ({n_done} of "
            f"{total} complete): set full.total_rounds to {n_done} and run the "
            f"same command to end the run with them, or run it as it is for the "
            f"rounds left"
        ]

    def finish_run(self, build: "AuditBuild") -> Tuple[int, bool]:
        """The end of a full run whose planned rounds are all complete: the
        final retrain's training (full.final_retrain; `FinalRetrain.prepare`
        tells where), the join of a parallel audit set build, the audit
        records completed (`audit_wrapup`), then, when the final retrain
        trained, its gate and deploy. (the worst exit status, whether this
        invocation deployed the final retrain)."""
        rounds = complete_prefix(self.run_dir)
        final = None
        rc_train = 0
        if self.get("full.final_retrain.enabled") and rounds:
            final = FinalRetrain.prepare(self, rounds)
            rc_train = final.train()
        rc = build.join()
        undecided = (
            final is not None
            and not rc_train
            and not (final.dir / final.DECISION).is_file()
        )
        self.audit_wrapup(rounds, final.models if undecided else None)
        if final is None or rc_train:
            return max(rc, rc_train), False
        rc_final, deployed = final.decide_and_deploy()
        return max(rc, rc_final), deployed

    def audit_candidate(self, rd: Path) -> None:
        """The round's bundle on the audit set when stage06 does not run
        (`<round>/stage05/audit.json`); never fails the round."""
        if auditlib.config_dir(self.cfg) is None:
            return
        cmd = [sys.executable, "-u", str(AUDIT_EVAL_SCRIPT), *self.args_audit_eval(rd)]
        ui.banner("audit -- the round's bundle on the audit set", ui.C.BLUE)
        rc = run_logged(
            cmd,
            rd / "audit.log",
            env=self.stage_env(),
            running=self.running,
            label=f"{rd.name} audit",
        )
        if rc:
            ui.warn(
                "audit", f"audit evaluation failed (rc={rc}); see {rd / 'audit.log'}"
            )

    # ── validation (stages 7 and 8) ──

    def stage07_enabled(self) -> bool:
        return stage07_enabled(self.cfg)

    def stage08_enabled(self) -> bool:
        return stage08_enabled(self.cfg)

    def _section(self, name: str) -> Dict[str, Any]:
        spec = SCHEMA.children[name].children
        return {k: self.get(f"{name}.{k}") for k in spec}

    def validation_key(
        self, key: str, models_round: Path, rounds: Sequence[Path]
    ) -> Optional[Dict[str, Any]]:
        """What a pass of validation stage `key` vouches for: the model it
        validated and the settings it ran with. None for stage 7 when this
        run has deployed no model."""
        if key == "stage07":
            record = latest_deploy_record(rounds)
            if record is None:
                return None
            build = stage07_build(self.cfg, self.paths)
            return _plain(
                {
                    "deploy_record": str(record),
                    "model_sha256": (_read_json(record) or {}).get("sha256"),
                    "build_dir": str(build.build_dir) if build else None,
                    "libraries": build_libraries(build.build_dir) if build else [],
                    "stage07": self._section("stage07"),
                    "pick_measurement": self.get("validate.pick_measurement"),
                    "origami_reference": self.origami_reference,
                }
            )
        train = models_round / "stage05"
        evaluation = {
            k: self.get(f"validate.{k}")
            for k in (
                "pick_measurement",
                "rebench_misses",
                "rebench_runs",
                "min_cell_gemms",
            )
        }
        # stage08 merges a dataset's repeated rows by it (lib/repeats.py).
        evaluation["repeat_aggregation"] = self.get("train.repeat_aggregation")
        return _plain(
            {
                "train_dir": str(train),
                "models_sha256": _sha256_file(train / "models.pt"),
                "weight_dtype": self.weight_dtype,
                "stage08": self._section("stage08"),
                "evaluation": evaluation,
            }
        )

    def record_validation(self, key: str, status: str, **info: Any) -> None:
        self.validation[key] = {"status": status, **info}

    def run_validate(
        self,
        *,
        models_round: Path,
        rounds: List[Path],
        from_stage: float,
        require_deploy: bool = True,
        resume: bool = False,
    ) -> int:
        """Stages 7 and 8 of the enabled ones, from `from_stage`. With
        `resume` a stage whose marker records a pass that stands for the same
        model and settings (`validation_key`, `validation_covers`) is not run
        again. Stage 7 without a deployed model fails with 2 under
        `require_deploy`, else it is skipped with a warning. Returns the exit
        status: the worst of the stages that ran or failed to start."""
        vdir = self.run_dir / "validate"
        vdir.mkdir(parents=True, exist_ok=True)
        ui.banner(f"validate  models from {models_round.name}", ui.C.HEAD)
        rcs: List[int] = []
        for key, number, enabled in VALIDATION_STAGES:
            if not enabled(self.cfg):
                continue
            if number < from_stage:
                ui.warn(
                    "validate",
                    f"stage {number:g} skipped (--from-stage {from_stage:g})",
                )
                continue
            want = self.validation_key(key, models_round, rounds)
            if want is None:
                why = "this run has deployed no model"
                self.record_validation(key, "not_run", reason=why)
                if require_deploy:
                    ui.err(
                        "validate",
                        f"stage 7 needs a model deployed by this run (round_*/"
                        f"stage06/deploy_record.json): {why}",
                    )
                    rcs.append(2)
                else:
                    ui.warn("validate", f"stage 7 skipped: {why}")
                continue
            passed = validated_by(vdir / key)
            if resume and validation_covers(passed, want):
                ui.ok(
                    "validate",
                    f"stage {number:g} passed before for this model and settings "
                    f"({vdir / key / STAGE_MARKER})",
                )
                self.record_validation(key, "passed", validated=passed, earlier=True)
                continue
            if key == "stage07":
                rc = self._run_stage07(vdir, rounds, want)
            else:
                rc = self._run_stage08(vdir, models_round, want)
            self.record_validation(
                key, "passed" if rc == 0 else "failed", rc=rc, validated=want
            )
            rcs.append(rc)
        failed = [rc for rc in rcs if rc != 0]
        for rc in failed:
            ui.err("validate", f"a validation stage failed (rc={rc})")
        return max((exit_code(rc) for rc in failed), default=0)

    def _stage07_bench_yamls(self, vdir: Path) -> Tuple[List[Path], List[Path]]:
        """(stage07.parity_sample_trained shapes sampled from the training
        data, else stage07.bench_yamls; stage07.parity_sample_audit GEMMs
        sampled from the audit set, which no round trained on)."""
        out = [Path(p) for p in self.get("stage07.bench_yamls")]
        held_out: List[Path] = []
        seed = int(self.get("seed.seed"))
        n = int(self.get("stage07.parity_sample_trained"))
        if n > 0:
            path = vdir / f"stage07_trained_{n}.yaml"
            got, total = sample_trained_shapes(self.run_dir, n, self.cfg, path, seed)
            if got:
                ui.ok(
                    "validate",
                    f"stage 7: {got} of {total} trained shapes -> {path.name}",
                )
                out = [path]
            else:
                ui.warn("validate", "no trained shapes to sample for stage 7")
        n = int(self.get("stage07.parity_sample_audit"))
        audit = auditlib.config_dir(self.cfg)
        if n > 0 and audit is not None:
            path = vdir / f"stage07_audit_{n}.yaml"
            got, total = sample_audit_shapes(audit, n, self.cfg, path, seed)
            if got:
                ui.ok(
                    "validate",
                    f"stage 7: {got} of the audit set's {total} exclusive GEMMs "
                    f"(held out of training) -> {path.name}",
                )
                held_out.append(path)
            else:
                ui.warn("validate", f"the audit set {audit} has no GEMMs to sample")
        return out, held_out

    def _run_stage07(
        self, vdir: Path, rounds: List[Path], want: Optional[Mapping[str, Any]] = None
    ) -> int:
        record = latest_deploy_record(rounds)
        if record is None:
            ui.err(
                "validate",
                "stage 7 needs a model deployed by this run (round_*/stage06/"
                "deploy_record.json); none found",
            )
            return 2
        build = stage07_build(self.cfg, self.paths)
        if build is None:
            ui.err(
                "validate",
                "stage 7 needs stage07.build_dir or deploy.rocm_libraries_to_deploy",
            )
            return 2
        if build.library:
            try:
                self.build_stage07_library(vdir, record)
            except (OSError, ValueError, KeyError) as e:
                ui.err("validate", f"stage 7 cannot serve the staged model: {e}")
                return 2
        yamls, held_out = self._stage07_bench_yamls(vdir)
        if not yamls and not held_out:
            ui.err("validate", "stage 7 has no bench yaml to run")
            return 2
        return self.run_stage(
            vdir,
            "stage07",
            self.args_stage07(vdir, record, yamls, rounds, held_out),
            None,
            marker={VALIDATED: want} if want is not None else None,
        )

    def stage07_library_dir(self, vdir: Path) -> Path:
        return vdir / "stage07_lib" / arch_base(self.arch)

    def build_stage07_library(self, vdir: Path, record: Path) -> Path:
        """`stage07_library_dir`: links to the bench library's files with the
        staged model of the deploy record `record` in place of the
        library stem's weights and the blocklist sidecar staged with it in
        place of the stem's (lib/pool_libdir.py `build_serving_libdir`), so
        its kernels, solution indices and pool are the run's own."""
        from lib.pool_libdir import build_serving_libdir

        rec = _read_json(record) or {}
        arch_dir = arch_base(self.arch)
        target = next(
            (t for t in rec.get("targets") or [] if t.get("dir") == arch_dir), None
        )
        if target is None:
            raise ValueError(f"{record} has no deploy target {arch_dir}")
        model = Path(str(target.get("staged") or rec.get("staged_model") or ""))
        sidecar = (
            model.parent / str(target["blocklist"]) if target.get("blocklist") else None
        )
        if sidecar is not None and not sidecar.is_file():
            raise ValueError(
                f"{record} names the blocklist sidecar {sidecar.name}, which is "
                f"not next to the staged model {model}"
            )
        dst = self.stage07_library_dir(vdir)
        build_serving_libdir(
            self.paths.library_dir,
            dst,
            str(target["library_stem"]),
            str(target["weights_file"]),
            model,
            rec.get("sha256"),
            blocklist=sidecar,
        )
        ui.info(
            "validate",
            f"stage 7 serves {model.name} from {dst} (links to {self.paths.library_dir})",
        )
        return dst

    def args_stage07(
        self,
        vdir: Path,
        record: Path,
        yamls: List[Path],
        rounds: Iterable[Path] = (),
        held_out: Iterable[Path] = (),
    ) -> List[str]:
        build = stage07_build(self.cfg, self.paths)
        assert build is not None
        args = [
            *self._common(),
            "--build-dir",
            str(build.build_dir),
            "--deploy-record",
            str(record),
            "--out-dir",
            str(vdir / "stage07"),
            "--device",
            str(self.devices[0]),
        ]
        if build.library:
            args += ["--bench-binary", str(build.bench_binary)]
            args += ["--library-dir", str(self.stage07_library_dir(vdir))]
        if self.get("stage07.cpus"):
            args += ["--cpus", str(self.get("stage07.cpus"))]
        if not self.get("stage07.production_route"):
            args.append("--no-route")
        for y in yamls:
            args += ["--bench-yaml", str(y)]
        for y in held_out:
            args += ["--held-out-yaml", str(y)]
        for rd in rounds:
            if (rd / "stage04").is_dir():
                args += ["--timings-csv-dir", str(rd / "stage04")]
        audit = auditlib.config_dir(self.cfg)
        if self.get("stage07.parity_sample_audit") > 0 and audit is not None:
            for d in audit_exclusive_dirs(audit):
                args += ["--timings-csv-dir", str(d)]
        if self.origami_reference:
            args.append("--origami-reference")
        return args

    def _run_stage08(
        self, vdir: Path, models_round: Path, want: Optional[Mapping[str, Any]] = None
    ) -> int:
        return self.run_stage(
            vdir,
            "stage08",
            self.args_stage08(vdir, models_round),
            None,
            marker={VALIDATED: want} if want is not None else None,
        )

    def args_stage08(self, vdir: Path, models_round: Path) -> List[str]:
        args = [
            "--train-dir",
            str(models_round / "stage05"),
            *self._model_eval_args(),
            "--out-dir",
            str(vdir / "stage08"),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--blocks-per-gpu",
            str(self.get("bench.blocks_per_gpu")),
            "--min-cell-gemms",
            str(self.get("validate.min_cell_gemms")),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--pick-measurement",
            str(self.get("validate.pick_measurement")),
        ]
        args += self.rebench_args(bench=False)
        if self.origami_reference:
            args.append("--origami-reference")
        for flag, key in (
            ("--tie-tolerance", "stage08.tie_tolerance"),
            ("--bootstrap", "stage08.bootstrap"),
        ):
            if self.get(key) is not None:
                args += [flag, str(self.get(key))]
        hbl = self.cfg["hipblaslt"]
        for flag, key in (
            ("--filter-a-type", "a_type"),
            ("--filter-b-type", "b_type"),
            ("--filter-c-type", "c_type"),
            ("--filter-d-type", "d_type"),
            ("--filter-compute-type", "compute_type"),
            ("--filter-transA", "transA"),
            ("--filter-transB", "transB"),
        ):
            args += [flag, str(hbl[key])]
        for ds in self.get("stage08.datasets"):
            args += ["--dataset", f"{ds['name']}={ds['path']}"]
            if ds.get("rocm_libraries_used"):
                build = self.dataset_build_dir(str(ds["rocm_libraries_used"]))
                args += ["--dataset-build-dir", f"{ds['name']}={build}"]
        return args

    def dataset_build_dir(self, checkout: str) -> Path:
        """hipBLASLt build directory of the checkout a stage08 dataset was
        benched with: the bench build (runtime.build_dir) for the bench root,
        the checkout's default build directory otherwise."""
        root = Path(checkout)
        if root.resolve() == self.paths.bench_root.resolve():
            return self.paths.build_dir
        return _hipblaslt_build(root, None)


def sample_audit_shapes(
    audit_dir: Path, n: int, cfg: Mapping[str, Any], out_path: Path, seed: int
) -> Tuple[int, int]:
    """Write a bench yaml of up to `n` of the audit set's GEMMs (its stage01
    shapes, all benched in the exclusive context and never trained on).
    Returns (written, available)."""
    from lib.bench_yaml import write_bench_yaml
    from lib.shapes import Shape

    shapes = sorted(auditlib.read_shapes(auditlib.shapes_yaml(audit_dir)))
    random.Random(seed).shuffle(shapes)
    pick = shapes[: max(0, n)]
    if pick:
        write_bench_yaml(
            [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in pick], cfg, out_path
        )
    return len(pick), len(shapes)


def audit_exclusive_dirs(audit_dir: Path) -> List[Path]:
    """The enriched directories that hold the audit set's exclusive rows:
    generation 0's exclusive group (or its single stage04) and every
    increment's stage04; generation 0's exclusive group alone while the
    audit is still benched."""
    out: List[Path] = []
    for gen in (auditlib.manifest(audit_dir) or {}).get("generations") or []:
        dirs = gen.get("stage04")
        for d in [dirs] if isinstance(dirs, str) else dirs or []:
            parts = Path(str(d)).parts
            if (
                len(parts) > 2
                and parts[0] == "bench"
                and parts[1] != auditlib.EXCLUSIVE
            ):
                continue
            out.append(Path(audit_dir) / str(d))
    own = auditlib.group_stage04(Path(audit_dir), auditlib.EXCLUSIVE)
    if not out and (own / STAGE_MARKER).is_file():
        out.append(own)
    return out


def sample_trained_shapes(
    run_dir: Path, n: int, cfg: Mapping[str, Any], out_path: Path, seed: int
) -> Tuple[int, int]:
    """Write a bench yaml of up to `n` distinct (M, N, K, batch) problems that
    benched successfully in any round (problems present in the enriched
    stage04 CSVs; stage01 shapes when no CSV exists). Returns (written,
    available)."""
    from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml
    from lib.shapes import Shape

    seen: Dict[Tuple[int, int, int, int], None] = {}
    for rd in round_dirs(run_dir):
        for csv_path in sorted((rd / "stage04").glob("*.csv")):
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                cols = {c.lower(): c for c in (reader.fieldnames or [])}
                keys = [cols.get(k) for k in ("m", "n", "k")]
                bkey = cols.get("batch_count") or cols.get("batch")
                if None in keys:
                    continue
                for row in reader:
                    try:
                        t = (
                            int(row[keys[0]]),
                            int(row[keys[1]]),
                            int(row[keys[2]]),
                            int(row[bkey]) if bkey else 1,
                        )
                    except (TypeError, ValueError):
                        continue
                    seen.setdefault(t, None)
    if not seen:
        for rd in round_dirs(run_dir):
            y = rd / "stage01" / "shapes.yaml"
            if y.is_file():
                for t in parse_shapes_yaml(y):
                    seen.setdefault(tuple(int(v) for v in t), None)
    shapes = sorted(seen)
    random.Random(seed).shuffle(shapes)
    pick = shapes[: max(0, n)]
    if pick:
        write_bench_yaml(
            [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in pick], cfg, out_path
        )
    return len(pick), len(shapes)


# ── planning ────────────────────────────────────────────────────────────────


def plan_training(
    mode: str,
    run_dir: Path,
    total_rounds: int,
    from_round: Optional[int],
    from_stage: Optional[float],
) -> List[RoundStep]:
    """Round steps for initial/active/full. Raises ConfigError on an
    inconsistent request."""
    existing = {round_index(p): p for p in round_dirs(run_dir)}
    done = complete_prefix(run_dir)
    k = len(done)

    def priors(i: int) -> List[Path]:
        return [run_dir / f"round_{j}" for j in range(i)]

    if k in existing and (existing[k] / ENHANCE_RECORD).is_file():
        raise ConfigError(
            [
                f"round_{k} is an incomplete enhance round; finish it with "
                f"--mode enhance (it resumes with its recorded cells)"
            ]
        )

    if mode == "initial":
        if from_round not in (None, 0):
            raise ConfigError(["--mode initial only runs round_0"])
        if 0 in existing and round_complete(existing[0]):
            later = [i for i in existing if i and i > 0]
            if from_stage is None or later:
                raise ConfigError(
                    [
                        "round_0 is already complete"
                        + (" and later rounds depend on it" if later else "")
                        + "; use --mode active/full, or --from-stage to redo "
                        "round_0 of a run that has no later rounds"
                    ]
                )
        return [RoundStep(0, "initial", [], from_stage, True)]

    if mode == "active":
        if k == 0:
            raise ConfigError(
                [
                    f"--mode active needs a complete round_0 in {run_dir}; run "
                    f"--mode initial first"
                ]
            )
        if from_round not in (None, k):
            raise ConfigError(
                [f"--mode active runs round_{k} (the next round); got --from-round"]
            )
        stray = sorted(i for i in existing if i is not None and i > k)
        if stray:
            raise ConfigError(
                [
                    f"round_{k} is incomplete but later rounds exist "
                    f"({['round_%d' % i for i in stray]}); resume with --mode full "
                    f"--from-round {k}"
                ]
            )
        return [RoundStep(k, "active", priors(k), from_stage, k >= total_rounds - 1)]

    if from_round is not None:
        if from_round >= total_rounds:
            raise ConfigError(
                [f"--from-round {from_round} >= full.total_rounds ({total_rounds})"]
            )
        if from_round > k:
            raise ConfigError(
                [f"--from-round {from_round}: round_{k} is not complete yet"]
            )
        start = from_round
        start_stage = from_stage if from_stage is not None else 1.0
    else:
        if k >= total_rounds:
            return []
        start = k
        start_stage = from_stage
    steps = []
    for i in range(start, total_rounds):
        steps.append(
            RoundStep(
                i,
                "initial" if i == 0 else "active",
                priors(i),
                start_stage if i == start else 1.0,
                i == total_rounds - 1,
            )
        )
    return steps


def deploy_outlook(
    cfg: Mapping[str, Any], steps: Sequence[RoundStep], force_deploy: bool = False
) -> Tuple[str, str]:
    """(`always`, `maybe` or `never`, why) the rounds `steps` of a --mode full
    run, or its final retrain (full.final_retrain), deploy a bundle, which
    stage 7 validates; `force_deploy` is --deploy."""
    outlook, why = _rounds_deploy_outlook(cfg, steps, force_deploy)
    if outlook == "always" or not cfg_get(cfg, "full.final_retrain.enabled"):
        return outlook, why
    if cfg_get(cfg, "full.final_retrain.gate") == "none":
        return "always", "the final retrain deploys (full.final_retrain.gate none)"
    return "maybe", (
        f"{why}; the final retrain deploys when the audit set finds it not "
        f"worse (full.final_retrain)"
    )


def _rounds_deploy_outlook(
    cfg: Mapping[str, Any], steps: Sequence[RoundStep], force_deploy: bool
) -> Tuple[str, str]:
    if not steps:
        return "never", "every planned round is complete"
    if force_deploy:
        return "always", "--deploy deploys every round"
    if steps[-1].mode == "initial":
        return "never", "a run of one round deploys only with --deploy"
    if steps[-1].is_final and cfg_get(cfg, "deploy.on_final_round"):
        return "always", "the final round deploys (deploy.on_final_round)"
    rule = (
        "with deploy.on_final_round false only a round whose gate GEMMs reach "
        "deploy.target_sel_eff deploys"
    )
    if cfg_get(cfg, "deploy.target_sel_eff") is None:
        return "never", f"{rule}, and deploy.target_sel_eff is unset"
    if cfg_get(cfg, "deploy.gate_frac") == 0:
        return "never", f"{rule}, and deploy.gate_frac 0 sets no gate GEMMs aside"
    return "maybe", rule


def deploy_errors(
    cfg: Mapping[str, Any],
    run_dir: Path,
    steps: Sequence[RoundStep],
    force_deploy: bool,
) -> List[str]:
    """Stage 7 of a --mode full run validates a bundle the run deploys: an
    error when no round the run keeps has deployed one and none of `steps`
    will (`deploy_outlook`), a warning when only a round that reaches the
    target will."""
    if not stage07_enabled(cfg):
        return []
    done = complete_prefix(run_dir)
    first = steps[0].index if steps else len(done)
    kept = [rd for rd in done if (round_index(rd) or 0) < first]
    bundles = [b for b in rb.bundles(run_dir, kept) if rb.bundle_key(b)[0] < first]
    if latest_deploy_record(bundles) is not None:
        return []
    outlook, why = deploy_outlook(cfg, steps, force_deploy)
    if outlook == "maybe":
        unless = (
            "neither a round reaches it nor the final retrain ships"
            if cfg_get(cfg, "full.final_retrain.enabled")
            else "no round reaches it"
        )
        ui.warn(
            "deploy",
            f"stage 7 validates a bundle this run deploys: {why}, so the run ends "
            f"with stage 7 not run (exit 2) when {unless}",
        )
    if outlook != "never":
        return []
    if not steps:
        last = len(done) - 1
        fix = f"deploy round_{last} with --from-round {last} --from-stage 6 --deploy"
    elif steps[-1].mode == "initial":
        fix = "pass --deploy"
    else:
        fix = "pass --deploy, set deploy.on_final_round"
    return [
        f"stage 7 validates a bundle this run deploys, and the run deploys none: "
        f"{why}; {fix} or turn stage 7 off"
    ]


def enhance_record(step: RoundStep) -> Dict[str, Any]:
    return {
        "round": step.index,
        "targets": list(step.targets or []),
        "shapes_per_cell": step.shapes_per_cell,
        "is_final": step.is_final,
        "selection": step.selection or {},
    }


def select_enhance_cells(
    run_dir: Path,
    rounds: List[Path],
    *,
    cells: List[str],
    weakest: Optional[int],
    below: Optional[float],
    rank_by: str,
    target: float,
) -> Dict[str, Any]:
    """Leaves an enhance round works on: the leaves under `cells`, then the
    `weakest` cells below `below` (default `target`) and every cell below
    `below`, by the training-side evidence of the complete `rounds`
    (validation outputs are never read). Raises ConfigError on unknown
    labels or an empty selection."""
    tree = sc.load_cumulative_split_tree(rounds)
    leaves, unknown = ce.requested_leaves(cells, tree)
    if unknown:
        raise ConfigError(
            [f"--cells: unknown cell labels {unknown} (labels look like M|N|K|B)"]
        )
    picked: List[ce.CellEvidence] = []
    evidence: Dict[str, ce.CellEvidence] = {}
    if weakest is not None or below is not None:
        evidence = ce.collect(run_dir, rounds=rounds)
        picked = ce.select_weak_cells(
            evidence,
            weakest=weakest,
            below=below if below is not None else target,
            rank_by=rank_by,
            target=target,
        )
    targets = list(leaves)
    for e in picked:
        if e.cell not in targets:
            targets.append(e.cell)
    if not targets:
        raise ConfigError(
            [
                "the enhance selection is empty: no cell is below "
                f"{(below if below is not None else target) * 100:.2f}% on the "
                "training-side evidence; name cells with --cells"
            ]
        )
    return {
        "targets": targets,
        "selection": {
            "cells": list(cells),
            "weakest": weakest,
            "below": below,
            "rank_by": rank_by,
            "target": target,
            "evidence_rounds": [p.name for p in rounds],
            "picked": [e.to_json(target) for e in picked],
            "named": leaves,
        },
    }


def plan_enhance(
    run_dir: Path,
    *,
    n_rounds: int,
    from_stage: Optional[float],
    shapes_per_cell: Optional[int],
    select: Callable[[List[Path]], Dict[str, Any]],
) -> List[RoundStep]:
    """`n_rounds` enhance rounds after the complete rounds. An incomplete
    round left by an earlier enhance invocation resumes with the selection
    its enhance.json recorded (`select` is not called); it counts as the
    first of the `n_rounds`. Raises ConfigError on an inconsistent run."""
    if n_rounds < 1:
        raise ConfigError(["--enhance-rounds must be >= 1"])
    done = complete_prefix(run_dir)
    k = len(done)
    if k == 0:
        raise ConfigError(
            [
                f"--mode enhance needs a complete round_0 in {run_dir}; run "
                f"--mode initial first"
            ]
        )
    existing = {round_index(p): p for p in round_dirs(run_dir)}
    stray = sorted(i for i in existing if i is not None and i > k)
    if stray:
        raise ConfigError(
            [
                f"round_{k} is incomplete but later rounds exist "
                f"({['round_%d' % i for i in stray]}); finish the run with "
                f"--mode full --from-round {k} first"
            ]
        )
    record = _read_json(run_dir / f"round_{k}" / ENHANCE_RECORD)
    if k in existing and record is None:
        raise ConfigError(
            [
                f"round_{k} is an incomplete active round; finish it with "
                f"--mode active before enhancing"
            ]
        )
    if record is not None:
        chosen = {
            "targets": list(record.get("targets") or []),
            "selection": dict(record.get("selection") or {}),
        }
        if record.get("shapes_per_cell") is not None:
            shapes_per_cell = int(record["shapes_per_cell"])
        ui.info(
            "plan",
            f"round_{k} is an incomplete enhance round; resuming it with its "
            f"recorded cells {chosen['targets']}",
        )
    else:
        if from_stage not in (None, 1.0):
            raise ConfigError(
                [f"--from-stage resumes an enhance round; round_{k} has not started"]
            )
        chosen = select(done)
    return [
        RoundStep(
            k + i,
            "active",
            [run_dir / f"round_{j}" for j in range(k + i)],
            from_stage if i == 0 else 1.0,
            i == n_rounds - 1,
            targets=list(chosen["targets"]),
            shapes_per_cell=shapes_per_cell,
            selection=chosen["selection"],
        )
        for i in range(n_rounds)
    ]


# ── report ──────────────────────────────────────────────────────────────────


def emit_report(
    run_dir: Path, origami_reference: bool = False, timeout_s: Optional[float] = None
) -> Dict[str, Any]:
    """Best-effort HTML report, with its Weak spots data in
    `<run>/weak_spots/` (pool_gaps.json and .csv for kernel generation,
    weak_spots.json); never changes the exit status. make_report is
    stopped after `timeout_s` (report.timeout_s; None or 0: never).
    Returns how it went, for run_status.json."""
    script = THIS_DIR / "scripts" / "make_report.py"
    out = run_dir / "report.html"
    cmd = [sys.executable, str(script), str(run_dir), str(out)]
    cmd += ["--weak-spots-dir", str(run_dir / WEAK_SPOTS_DIR)]
    if origami_reference:
        cmd.append("--origami-reference")
    before = out.stat().st_mtime_ns if out.is_file() else None
    t0 = time.time()
    status: Dict[str, Any] = {"path": str(out), "origami_reference": origami_reference}
    try:
        proc = subprocess.run(cmd, check=False, timeout=timeout_s or None)
    except subprocess.TimeoutExpired:
        ui.warn(
            "report",
            f"report generation stopped after {timeout_s:g} s (report.timeout_s)",
        )
        return dict(status, status="timed_out", elapsed_s=time.time() - t0)
    except (OSError, subprocess.SubprocessError) as e:
        ui.warn("report", f"report generation failed: {e!r}")
        return dict(status, status="failed", reason=repr(e), elapsed_s=time.time() - t0)
    status["elapsed_s"] = time.time() - t0
    written = out.is_file() and out.stat().st_mtime_ns != before
    if proc.returncode == 0 and written:
        ui.ok("report", f"{out}")
        return dict(status, status="written")
    ui.warn("report", f"report generation failed (rc={proc.returncode})")
    return dict(status, status="failed", reason=f"rc={proc.returncode}")


def validation_bundles(run_dir: Path) -> List[Path]:
    """The directories a run's validation and run_status.json take the
    bundle from: stage 7 the latest deploy record among them, stage 8 the
    shipped one's models, else the last one's. The complete rounds and the
    final retrain while it is a bundle of the run, in the order the run
    deploys them (`run_bundles`), so that a shipped final retrain is the one
    validated, named in the validation markers and in run_status.json."""
    return run_bundles(run_dir)


def summary_failed(summary: Any) -> bool:
    """A validation stage's summary.json reports a failure: stage 7's `ok`
    false, a stage 8 dataset with a non-zero `rc`."""
    if not isinstance(summary, dict):
        return False
    datasets = summary.get("datasets")
    return summary.get("ok") is False or any(
        isinstance(d, dict) and d.get("rc") not in (0, None)
        for d in (datasets if isinstance(datasets, list) else [])
    )


def final_retrain_state(run_dir: Path, d: Path) -> Dict[str, Any]:
    """The state of the final retrain in `d` (final_retrain/, or a
    candidate in final_retrain_next/): its `status` is pending (not trained
    on the run's complete rounds yet: they are not all done, it has not
    trained, or the rounds changed since it did, so the next --mode full
    trains it), undecided (trained, its gate waiting for the audit set),
    accepted (its gate let it through and stage06 has not deployed it),
    rejected (its gate kept it out) or shipped (a bundle of the run,
    `run_bundles`); `reason` says why, `trained_on` names its rounds."""
    rounds = [rd.name for rd in complete_prefix(run_dir)]
    trained = rb.trained_rounds(d)
    dec = _read_json(d / FinalRetrain.DECISION)
    out: Dict[str, Any] = {
        "dir": str(d),
        "trained_on": list(trained),
        "reason": None,
    }
    changed = rb.changed_rounds(run_dir, trained) if trained else []
    if not trained or sorted(trained) != sorted(rounds) or changed:
        status = "pending"
        if changed or (trained and sorted(trained) != sorted(rounds)):
            out["reason"] = (
                f"trained on {', '.join(trained)}; the run's complete rounds "
                f"are {', '.join(rounds) or 'none'}"
            )
    elif not stage_complete(d / "stage05"):
        status = "pending"
    elif not isinstance(dec, dict):
        status = "undecided"
    elif not dec.get("accepted"):
        status, out["reason"] = "rejected", dec.get("reason")
    elif rb.final_retrain(run_dir) != d:
        status, out["reason"] = "accepted", dec.get("reason")
    else:
        status, out["reason"] = "shipped", dec.get("reason")
    out["status"] = status
    if isinstance(dec, dict):
        out["decision"] = str(d / FinalRetrain.DECISION)
        out["selected_by_audit"] = bool(dec.get("selected_by_audit"))
    return out


def final_retrain_status(run_dir: Path, enabled: bool) -> Dict[str, Any]:
    """run_status.json's account of full.final_retrain (`enabled`): `off`
    (disabled, and the run has none), else the state of the final retrain
    on the run's rounds (`final_retrain_state`): of the new one in
    final_retrain_next/ while there is one (`FinalRetrain.prepare`), with
    `replaces` naming the one in final_retrain/ while that stays a bundle of
    the run (run_status.json's `deployed` names the bundle that ships), else
    of the one in final_retrain/."""
    d = Path(run_dir) / FINAL_RETRAIN_DIR
    n = Path(run_dir) / rb.FINAL_RETRAIN_NEXT_DIR
    if not enabled and not d.is_dir() and not n.is_dir():
        return {"enabled": False, "status": "off"}
    if not n.is_dir():
        return {"enabled": enabled, **final_retrain_state(run_dir, d)}
    out: Dict[str, Any] = {"enabled": enabled, **final_retrain_state(run_dir, n)}
    if rb.final_retrain(run_dir) == d:
        out["replaces"] = {"dir": str(d), "trained_on": list(rb.trained_rounds(d))}
    return out


class RunStatus:
    """`<run_dir>/run_status.json`: the rounds, the deployed bundle (a
    round's, or the final retrain's), the final retrain, the audit set, each
    validation stage, the report and the invocation that wrote it,
    rewritten when the invocation starts, after each round and when it
    ends. `complete` holds once every planned round is complete, an enabled
    final retrain is decided (and deployed when it ships) for them, and
    every enabled validation stage has passed for the current model."""

    def __init__(self, run_dir: Path, mode: str) -> None:
        self.run_dir = run_dir
        self.invocation: Dict[str, Any] = {
            "mode": mode,
            "argv": list(sys.argv),
            "pid": os.getpid(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "finished": None,
            "exit_status": None,
        }
        self.cfg: Optional[Mapping[str, Any]] = None
        self.validation: Dict[str, Dict[str, Any]] = {}
        self.failed_round: Optional[Dict[str, Any]] = None
        self.report: Optional[Dict[str, Any]] = None

    def finish(self, exit_status: Optional[int]) -> None:
        self.invocation.update(
            finished=datetime.now().isoformat(timespec="seconds"),
            exit_status=exit_status,
        )

    def validation_entry(self, key: str, enabled: bool) -> Dict[str, Any]:
        """This invocation's outcome of `key`, else what the run directory
        holds: a pass of the bundle the run would validate now (`stale` when
        it vouches for another one, `passed_unrecorded` when its marker
        predates the VALIDATED record, so that the next --mode full runs it
        again), a summary that reports a failure, or nothing yet."""
        if not enabled:
            return {"enabled": False, "status": "off"}
        if key in self.validation:
            return {"enabled": True, **self.validation[key]}
        sdir = self.run_dir / "validate" / key
        marker = _read_json(sdir / STAGE_MARKER)
        passed = marker.get(VALIDATED) if isinstance(marker, dict) else None
        if isinstance(marker, dict) and not isinstance(passed, dict):
            return {
                "enabled": True,
                "status": "passed_unrecorded",
                "finished": marker.get("finished"),
                "marker": str(sdir / STAGE_MARKER),
            }
        if isinstance(passed, dict):
            rounds = validation_bundles(self.run_dir)
            if key == "stage07":
                record = latest_deploy_record(rounds)
                rec = (_read_json(record) or {}) if record else {}
                now = {
                    "deploy_record": str(record) if record else None,
                    "model_sha256": rec.get("sha256"),
                }
            else:
                models = shipped_round(rounds) or (rounds[-1] if rounds else None)
                now = {"train_dir": str(models / "stage05") if models else None}
            same = all(passed.get(k) == v for k, v in now.items())
            return {
                "enabled": True,
                "status": "passed" if same else "stale",
                "validated": passed,
            }
        if summary_failed(_read_json(sdir / "summary.json")):
            return {"enabled": True, "status": "failed", "summary": str(sdir)}
        return {"enabled": True, "status": "pending"}

    def doc(self) -> Dict[str, Any]:
        assert self.cfg is not None
        total = int(cfg_get(self.cfg, "full.total_rounds"))
        rounds = []
        for rd in round_dirs(self.run_dir):
            marker = _read_json(rd / ROUND_MARKER)
            entry: Dict[str, Any] = {"round": rd.name, "complete": False}
            if isinstance(marker, dict):
                entry["complete"] = True
                for k in ("mode", "deployed", "held_out_sel_eff", "gate_sel_eff"):
                    entry[k] = marker.get(k)
                entry["finished"] = marker.get("finished")
            rounds.append(entry)
        done = len(complete_prefix(self.run_dir))
        record = latest_deploy_record(validation_bundles(self.run_dir))
        deployed = None
        if record is not None:
            rec = _read_json(record) or {}
            deployed = {
                "round": record.parent.parent.name,
                "deploy_record": str(record),
                "sha256": rec.get("sha256"),
            }
        audit = auditlib.config_dir(self.cfg)
        audit_doc = None
        if audit is not None:
            state, why = auditlib.status(audit)
            audit_doc = {"dir": str(audit), "status": state, "reason": why}
        validation = {
            key: self.validation_entry(key, enabled(self.cfg))
            for key, _, enabled in VALIDATION_STAGES
        }
        final = final_retrain_status(
            self.run_dir, bool(cfg_get(self.cfg, "full.final_retrain.enabled"))
        )
        complete = (
            done >= total
            and all(v["status"] in ("off", "passed") for v in validation.values())
            and (not final["enabled"] or final["status"] in ("shipped", "rejected"))
        )
        return {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "complete": complete,
            "rounds": {
                "planned": total,
                "complete": done,
                "per_round": rounds,
                "failed": self.failed_round,
            },
            "deployed": deployed,
            "final_retrain": final,
            "audit": audit_doc,
            "validation": validation,
            "report": self.report,
            "invocation": self.invocation,
        }

    def write(self) -> None:
        if self.cfg is None:
            return
        from lib.fs import atomic_write_text

        try:
            atomic_write_text(
                self.run_dir / RUN_STATUS,
                json.dumps(_json_safe(self.doc()), indent=2) + "\n",
            )
        except OSError as e:
            ui.warn("status", f"could not write {RUN_STATUS}: {e!r}")


# ── audit set (lib/audit.py) ────────────────────────────────────────────────


class AuditRun:
    """`--mode audit`: build, resume or extend the audit set in `run_dir`
    with `pipeline` (built on `lib.audit.bench_config` of the training
    config)."""

    def __init__(
        self,
        pipeline: Pipeline,
        train_cfg: Mapping[str, Any],
        source_runs: Sequence[Path],
    ) -> None:
        self.p = pipeline
        self.dir = pipeline.run_dir
        self.train_cfg = train_cfg
        self.st = auditlib.settings(train_cfg)
        self.source_runs = [Path(s).resolve() for s in source_runs]
        self.started = time.time()
        self.shapes_this_run = 0

    # ── bookkeeping ──

    def binding(self) -> Dict[str, Any]:
        return audit_binding(self.train_cfg, self.p.cfg)

    def read_manifest(self) -> Optional[Dict[str, Any]]:
        return auditlib.manifest(self.dir)

    def write_manifest(self, m: Mapping[str, Any]) -> None:
        tmp = self.dir / (auditlib.MANIFEST + ".tmp")
        tmp.write_text(json.dumps(_json_safe(dict(m)), indent=2) + "\n")
        os.replace(tmp, self.dir / auditlib.MANIFEST)

    def manifest_errors(self, m: Mapping[str, Any]) -> List[str]:
        return audit_binding_errors(self.dir, self.binding(), m)

    def current_kernels(self) -> Dict[int, str]:
        return self.p.pool_digests()

    def current_identity(self) -> Optional["ki.LibraryIdentity"]:
        """The kernel identity of the pool's files (lib/kernel_identity.py);
        None when they cannot be read."""
        try:
            return bank_pool.config_identity(
                self.p.cfg, self.p.paths.library_dir, self.p.stem, self.p.devices[0]
            )
        except (OSError, ValueError, RuntimeError):
            return None

    def build_changed(
        self, what: str, changed: Sequence[int], n: int, then: Mapping[int, str]
    ) -> str:
        """The refusal of bench lines planned with another build: `changed`
        (of `n`) of their solution indices name other kernels now; `then`
        holds the kernel ids recorded when they were planned."""
        now = self.current_identity()
        sid = changed[0]
        was = then.get(sid) or "another kernel"
        is_now = (now.ids.get(sid) if now is not None else None) or "none"
        return (
            f"the hipBLASLt build changed since {what} was planned: {len(changed)} "
            f"of its {n} solution indices now name other kernels (e.g. solution "
            f"{sid} named {was} then and names {is_now} now), so its bench lines "
            f"would run them"
        )

    def planned_kernel_ids(self) -> Dict[int, str]:
        """{solution: kernel id} the audit's plan recorded (the planner's
        plan.json); empty for a plan written before identities were
        recorded."""
        try:
            doc = json.loads((self.p.plan_dir(self.dir) / "plan.json").read_text())
            return ki.recorded_ids(ki.read_record(doc)) or {}
        except (OSError, ValueError):
            return {}

    # ── shapes ──

    def source_shape_yamls(self) -> List[Path]:
        out: List[Path] = []
        for run in self.source_runs:
            for rd in round_dirs(run):
                y = rd / "stage01" / "shapes.yaml"
                if y.is_file():
                    out.append(y)
        return out

    def args_stage01(self) -> List[str]:
        args = [
            "--mode",
            "initial",
            "--out-dir",
            str(self.dir / "stage01"),
            *self.p._common(),
            "--seed",
            str(self.st["seed"]),
            "--round-index",
            "0",
            "--n-strata",
            str(self.p.get("seed.n_strata")),
            "--target-per-cell",
            str(self.st["shapes_per_cell"]),
            "--min-cell-gemms",
            "0",
        ]
        for y in self.source_shape_yamls():
            args += ["--prior-shapes-yaml", str(y)]
        return args

    def args_stage02(self) -> List[str]:
        return [
            "--in-dir",
            str(self.dir / "stage01"),
            "--out-dir",
            str(self.p.plan_dir(self.dir)),
            *self.p._common(),
            "--dat-dir",
            str(self.p.paths.library_dir),
            "--library-stem",
            self.p.stem,
            "--arch",
            self.p.arch,
            "--weight-dtype",
            self.p.weight_dtype,
            "--hardware-device",
            str(self.p.devices[0]),
        ]

    def listed_kernels(self) -> set:
        """Every solution index the exhaustive bench lines list."""
        from lib.bench_yaml import line_solution_list, read_bench_lines

        out: set = set()
        for ln in read_bench_lines(self.p.plan_dir(self.dir) / "shapes.yaml"):
            out.update(line_solution_list(ln) or [])
        return out

    # ── units and parts ──

    def units(self) -> Dict[str, Any]:
        """{label: {context, parts: [[shape, ...], ...]}}, fixed when first
        written to bench/units.json."""
        from lib.bench_yaml import read_bench_lines, shape_of_line

        path = self.dir / "bench" / "units.json"
        saved = _read_json(path)
        if isinstance(saved, dict):
            return saved
        lines = read_bench_lines(self.p.plan_dir(self.dir) / "shapes.yaml")
        shapes = list(dict.fromkeys(s for s in map(shape_of_line, lines) if s))
        size = int(self.st["part_shapes"])

        def parts(chosen: Sequence[Any]) -> List[List[List[int]]]:
            return [
                [list(s) for s in chosen[i : i + size]]
                for i in range(0, len(chosen), size)
            ]

        out: Dict[str, Any] = {
            auditlib.EXCLUSIVE: {"context": None, "parts": parts(shapes)}
        }
        for ctx in self.p.contexts:
            chosen = ctxlib.select_shapes(shapes, ctx, int(self.st["seed"]))
            out[ctx.label] = {"context": ctx.to_dict(), "parts": parts(chosen)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        return out

    @staticmethod
    def part_dir(root: Path, label: str, i: int) -> Path:
        return root / "bench" / label / f"part_{i:04d}"

    def wrapper(self, label: str, unit: Mapping[str, Any]) -> Optional[Path]:
        """The context bench wrapper of a non-exclusive unit (stage03c's)."""
        from lib import hardware as hwlib

        if unit.get("context") is None:
            return None
        ctx = ctxlib.BenchContext.from_mapping(unit["context"])
        cdir = self.dir / "bench" / label
        n_cu = int(hwlib.device_hardware(self.p.cfg, self.p.devices[0]).n_cu)
        library_path = None
        if self.p.get("bench.pool_only_library"):
            from lib.pool_libdir import build_pool_libdir

            dst = self.dir / "bench" / "libdir_pool"
            if not dst.is_dir():
                build_pool_libdir(
                    self.p.paths.library_dir, [self.p.stem] + self.p.bank_stems(), dst
                )
            library_path = str(dst)
        spec = ctxlib.bench_spec(
            ctx,
            n_cu,
            str(self.p.paths.bench_binary),
            str(context_cotenant(self.p.cfg, self.p.paths)),
            str(cdir / "cotenant_logs"),
            library_path=library_path,
            path_append=self.p.get("contexts.path_append") or [rocm_bin()],
        )
        script = THIS_DIR / "lib" / "context_bench.py"
        return ctxlib.write_wrapper(cdir, spec, sys.executable, script)

    def logs_of(self, part: Path) -> Path:
        return part / ("stage03b" if self.p.rebench else "stage03") / "logs"

    def bench_part(
        self, part: Path, lines: Sequence[str], wrapper: Optional[Path]
    ) -> int:
        """stage03 and stage03b of the bench lines `lines` in `part`; skipped
        when the part is complete, redone from its first incomplete stage
        otherwise."""
        from lib.fs import atomic_write_text

        if stage_complete(part):
            return 0
        part.mkdir(parents=True, exist_ok=True)
        if not (part / "shapes.yaml").is_file():
            atomic_write_text(part / "shapes.yaml", "".join(lines))
        p = self.p
        bench = p.args_stage03(self.dir, in_dir=part, out_dir=part / "stage03")
        if wrapper is not None:
            bench = _strip_pool_only(bench)
        steps: List[Tuple[str, List[str]]] = [("stage03", bench)]
        if p.rebench:
            rebench = [
                "--in-dir",
                str(part),
                "--stage03-dir",
                str(part / "stage03"),
                "--out-dir",
                str(part / "stage03b"),
                *p._common(),
                "--bench-binary",
                str(p.paths.bench_binary),
                "--devices",
                ",".join(str(d) for d in p.devices),
                "--stall-timeout-s",
                str(p.get("bench.stall_timeout_s")),
                "--startup-grace-s",
                str(p.get("bench.startup_grace_s")),
                *p.identity_args(),
            ]
            steps.append(("stage03b", rebench))
        redo = False
        for key, args in steps:
            if not redo and stage_complete(part / key):
                continue
            redo = True
            if wrapper is not None:
                args = _replace_arg(args, "--bench-binary", str(wrapper))
            rc = p.run_stage(part, key, args, None)
            if rc:
                return rc
        p.mark_complete(part, None, note=f"{len(lines)} shapes")
        return 0

    # ── progress ──

    def write_progress(self, root: Path, units: Mapping[str, Any], what: str) -> None:
        per: Dict[str, Any] = {}
        done = total = 0
        for label, unit in units.items():
            parts = unit["parts"]
            n_done = [
                len(s)
                for i, s in enumerate(parts)
                if stage_complete(self.part_dir(root, label, i))
            ]
            n_all = sum(len(s) for s in parts)
            per[label] = {
                "parts_done": len(n_done),
                "parts_total": len(parts),
                "shapes_done": sum(n_done),
                "shapes_total": n_all,
            }
            done += sum(n_done)
            total += n_all
        elapsed = time.time() - self.started
        rate = self.shapes_this_run / elapsed * 3600 if self.shapes_this_run else None
        eta = None
        if rate:
            eta = datetime.fromtimestamp(time.time() + (total - done) / rate * 3600)
        rec = {
            "what": what,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "shapes_done": done,
            "shapes_total": total,
            "gemms_per_hour_this_invocation": rate,
            "expected_finish": eta.isoformat(timespec="minutes") if eta else None,
            "units": per,
        }
        tmp = self.dir / (auditlib.PROGRESS + ".tmp")
        tmp.write_text(json.dumps(_json_safe(rec), indent=2) + "\n")
        os.replace(tmp, self.dir / auditlib.PROGRESS)
        ui.info(
            "audit",
            f"{what}: {done}/{total} GEMMs benched"
            + (
                f", {rate:.0f}/h, expected finish {rec['expected_finish']}"
                if rate
                else ""
            ),
        )

    def bench_units(
        self,
        root: Path,
        units: Mapping[str, Any],
        line_of: Callable[[str, int, Tuple[int, ...]], str],
        what: str,
        enrich_groups: bool = False,
    ) -> int:
        """Bench every part of `units` under `root`; with `enrich_groups`
        each group's rows are enriched once its parts are all benched, so a
        partial audit can evaluate it."""
        for label, unit in units.items():
            wrapper = self.wrapper(label, unit)
            for i, shapes in enumerate(unit["parts"]):
                part = self.part_dir(root, label, i)
                if stage_complete(part):
                    continue
                ui.banner(f"audit {what}  {label}  part {i + 1}/{len(unit['parts'])}")
                lines = [line_of(label, i, tuple(s)) for s in shapes]
                rc = self.bench_part(part, lines, wrapper)
                if rc:
                    return rc
                self.shapes_this_run += len(shapes)
                self.write_progress(root, units, what)
            if enrich_groups and not stage_complete(
                auditlib.group_stage04(root, label)
            ):
                rc = self.p.run_stage(
                    root / "bench" / label,
                    "stage04",
                    self.group_stage04_args(
                        root, label, unit, auditlib.group_stage04(root, label)
                    ),
                    None,
                )
                if rc:
                    return rc
        return 0

    def group_stage04_args(
        self, root: Path, label: str, unit: Mapping[str, Any], out_dir: Path
    ) -> List[str]:
        """stage04 of one group's parts: the exclusive logs as --log-dir, a
        context's as --context-log-dir."""
        logs = [
            self.logs_of(self.part_dir(root, label, i))
            for i in range(len(unit["parts"]))
        ]
        ctx = unit.get("context")
        args = self.p.args_stage04(
            self.dir,
            log_dirs=[] if ctx else logs,
            out_dir=out_dir,
            plan=self.p.plan_dir(self.dir) / "plan.json",
        )
        if ctx:
            for d in logs:
                args += [
                    "--context-log-dir",
                    str(ctx["cu_budget"]),
                    str(ctx["schedule"]),
                    str(d),
                ]
        return args

    def args_stage04(self, root: Path, units: Mapping[str, Any]) -> List[str]:
        excl = units[auditlib.EXCLUSIVE]["parts"]
        logs = [
            self.logs_of(self.part_dir(root, auditlib.EXCLUSIVE, i))
            for i in range(len(excl))
        ]
        args = self.p.args_stage04(
            self.dir,
            log_dirs=logs,
            out_dir=root / "stage04",
            plan=self.p.plan_dir(self.dir) / "plan.json",
        )
        for label, unit in units.items():
            if unit.get("context") is None:
                continue
            ctx = unit["context"]
            for i in range(len(unit["parts"])):
                args += [
                    "--context-log-dir",
                    str(ctx["cu_budget"]),
                    str(ctx["schedule"]),
                    str(self.logs_of(self.part_dir(root, label, i))),
                ]
        return args

    # ── the run ──

    def run(self) -> int:
        m = self.read_manifest()
        if m is not None:
            errs = self.manifest_errors(m)
            if errs:
                _print_problems("audit", errs, [])
                return 2
        else:
            m = {
                "schema": auditlib.SCHEMA_VERSION,
                "created": datetime.now().isoformat(timespec="seconds"),
                "binding": self.binding(),
                "settings": dict(self.st, dir=None),
                "source_runs": [str(s) for s in self.source_runs],
                "generations": [],
                "status": "benching",
            }
            self.write_manifest(m)
        if not stage_complete(self.dir / "stage01"):
            rc = self.p.run_stage(self.dir, "stage01", self.args_stage01(), None)
            if rc:
                return exit_code(rc)
        if not stage_complete(self.p.plan_dir(self.dir)):
            m.pop("planned_kernels", None)
            kernels = self.current_kernels()
            rc = self.p.run_stage(self.dir, "stage02", self.args_stage02(), None)
            if rc:
                return exit_code(rc)
        else:
            kernels = self.current_kernels()
        if "planned_kernels" not in m:
            listed = self.listed_kernels()
            m["planned_kernels"] = {
                str(k): v for k, v in kernels.items() if k in listed
            }
            self.write_manifest(m)
        elif not m["generations"]:
            changed = changed_kernels(m["planned_kernels"], kernels)
            if changed:
                text = self.build_changed(
                    "the audit set",
                    changed,
                    len(m["planned_kernels"]),
                    self.planned_kernel_ids(),
                )
                _print_problems(
                    "audit",
                    [
                        f"{text}. Bench it with the build it was planned with, or "
                        f"build a new audit set in another --run-dir"
                    ],
                    [],
                )
                return 2
        units = self.units()
        m["shapes"] = {
            label: sum(len(s) for s in u["parts"]) for label, u in units.items()
        }
        self.write_manifest(m)
        if not m["generations"]:
            rc = self.bench_generation0(m, units)
            if rc:
                return rc
        return self.extend(m, units)

    def bench_generation0(self, m: Dict[str, Any], units: Mapping[str, Any]) -> int:
        from lib.bench_yaml import read_bench_lines, shape_of_line

        by_shape = {
            shape_of_line(ln): ln
            for ln in read_bench_lines(self.p.plan_dir(self.dir) / "shapes.yaml")
        }
        rc = self.bench_units(
            self.dir,
            units,
            lambda _l, _i, s: by_shape[s],
            "generation 0",
            enrich_groups=True,
        )
        if rc:
            return exit_code(rc)
        identity = self.current_identity()
        m["generations"] = [
            {
                "index": 0,
                "stage04": [
                    str(auditlib.group_stage04(self.dir, label).relative_to(self.dir))
                    for label in units
                ],
                "kernels": dict(m.get("planned_kernels") or {}),
                "library_digest": identity.digest if identity is not None else None,
                "finished": datetime.now().isoformat(timespec="seconds"),
            }
        ]
        self.write_manifest(m)
        return 0

    def extend(self, m: Dict[str, Any], units: Mapping[str, Any]) -> int:
        """Bench the kernels of the current pool no generation benched (new
        ones, or changed under the same index) on every audit GEMM, on
        generation 0's bench lines."""
        from lib.bench_yaml import (
            format_solution_list,
            read_bench_lines,
            set_line_knob,
            shape_of_line,
        )

        current = self.current_kernels()
        pending = auditlib.unbenched(m, current)
        removed = auditlib.benched_digests(m) - set(current.values())
        if removed:
            ui.info(
                "audit",
                f"{len(removed)} benched kernel(s) left the pool; their rows are "
                f"ignored",
            )
        while pending:
            k = len(m["generations"])
            root = self.dir / "increments" / f"inc_{k:03d}"
            rec_path = root / "increment.json"
            rec = _read_json(rec_path)
            if not isinstance(rec, dict):
                rec = {"kernels": {str(s): current[s] for s in pending}}
                identity = self.current_identity()
                if identity is not None:
                    rec[ki.RECORD_KEY] = ki.record(identity, pending)
                root.mkdir(parents=True, exist_ok=True)
                rec_path.write_text(json.dumps(rec, indent=2) + "\n")
            changed = changed_kernels(rec["kernels"], current)
            if changed:
                try:
                    then = ki.recorded_ids(ki.read_record(rec)) or {}
                except ValueError:
                    then = {}
                text = self.build_changed(
                    f"increment {k} ({rec_path})", changed, len(rec["kernels"]), then
                )
                _print_problems(
                    "audit",
                    [
                        f"{text}; bench it with the build it was planned with, or "
                        f"move {root} aside to plan it again"
                    ],
                    [],
                )
                return 2
            ids = sorted(int(s) for s in rec["kernels"])
            ui.banner(f"audit increment {k}: {len(ids)} kernel(s)", ui.C.HEAD)
            ids_text = format_solution_list(ids)
            cache: Dict[Tuple[str, int], Dict[Any, str]] = {}

            def line_of(label: str, i: int, s: Tuple[int, ...]) -> str:
                key = (label, i)
                if key not in cache:
                    src = self.part_dir(self.dir, label, i) / "shapes.yaml"
                    cache[key] = {shape_of_line(ln): ln for ln in read_bench_lines(src)}
                return set_line_knob(cache[key][s], "solution_index_list", ids_text)

            rc = self.bench_units(root, units, line_of, f"increment {k}")
            if rc:
                return exit_code(rc)
            if not stage_complete(root / "stage04"):
                args = self.args_stage04(root, units)
                rc = self.p.run_stage(root, "stage04", args, None)
                if rc:
                    return exit_code(rc)
            m["generations"].append(
                {
                    "index": k,
                    "stage04": str((root / "stage04").relative_to(self.dir)),
                    "kernels": dict(rec["kernels"]),
                    "library_digest": (rec.get(ki.RECORD_KEY) or {}).get("digest"),
                    "finished": datetime.now().isoformat(timespec="seconds"),
                }
            )
            self.write_manifest(m)
            pending = auditlib.unbenched(m, current)
        m["status"] = "complete"
        m["completed"] = datetime.now().isoformat(timespec="seconds")
        m["n_kernels"] = len(current)
        self.write_manifest(m)
        (self.dir / auditlib.COMPLETE_MARKER).write_text(
            json.dumps(
                {
                    "finished": m["completed"],
                    "generations": len(m["generations"]),
                    "shapes": m.get("shapes"),
                },
                indent=2,
            )
            + "\n"
        )
        ui.ok(
            "audit",
            f"audit set complete: {sum((m.get('shapes') or {}).values())} GEMMs over "
            f"{len(m['generations'])} generation(s), {len(current)} kernels",
        )
        return 0


def changed_kernels(
    planned: Mapping[str, str], current: Mapping[int, str]
) -> List[int]:
    """Solution indices of `planned` ({index: kernel digest}, lib/audit.py)
    whose kernel digest differs in `current`."""
    return sorted(int(s) for s, d in planned.items() if current.get(int(s)) != d)


def enrich_audit_groups(audit_dir: Path, enrich_root: Path) -> Dict[str, str]:
    """Enrich under `enrich_root`, with the audit's own bench config and
    stage04, every group of a benching audit whose parts are all benched but
    whose rows the builder has not enriched (lib/audit.py `enriched_copy`);
    nothing is written into the audit directory. {label: problem} of the
    groups left unenriched."""
    audit_dir = Path(audit_dir).resolve()
    enrich_root = Path(enrich_root).resolve()
    now = auditlib.state(audit_dir, None, enrich_root)
    todo = [k for k, g in now.groups.items() if g.status == "not_enriched"]
    if not todo:
        return {}
    snaps = sorted(
        p
        for p in (audit_dir / "configs").glob("*_audit*.yaml")
        if not p.name.endswith(".source.yaml")
    )
    if not snaps:
        return {k: f"{audit_dir / 'configs'} holds no audit config" for k in todo}
    try:
        cfg, _ = load_config(snaps[-1])
    except ConfigError as e:
        return {k: "; ".join(e.errors) for k in todo}
    try:
        pipeline = Pipeline(
            cfg=cfg,
            snapshot=snaps[-1],
            run_dir=audit_dir,
            paths=derive_paths(cfg),
            stages_dir=THIS_DIR / "stages",
            timing=Timing(),
            force_deploy=False,
        )
    except ValueError as e:
        return {k: str(e) for k in todo}
    run = AuditRun(pipeline, cfg, [])
    units = auditlib.units(audit_dir) or {}
    problems: Dict[str, str] = {}
    for label in todo:
        unit = units[label]
        before = auditlib.logs_fingerprint(audit_dir, label, unit)
        owner = enrich_root / label
        out = owner / "stage04"
        if out.exists():
            shutil.rmtree(out)
        owner.mkdir(parents=True, exist_ok=True)
        script = THIS_DIR / "stages" / STAGE_DEFS["stage04"].script
        cmd = [
            sys.executable,
            "-u",
            str(script),
            *run.group_stage04_args(audit_dir, label, unit, out),
        ]
        ui.banner(f"audit -- enrich {label} into {out}", ui.C.BLUE)
        rc = run_logged(cmd, owner / "stage04.log", env=pipeline.stage_env())
        if rc:
            problems[label] = f"stage04 failed (rc={rc}); see {owner / 'stage04.log'}"
            continue
        if auditlib.logs_fingerprint(audit_dir, label, unit) != before:
            problems[label] = "the group's logs changed while it was enriched"
            continue
        (out / auditlib.GROUP_SOURCE).write_text(
            json.dumps(
                {"audit_dir": str(audit_dir), "label": label, "logs": before}, indent=2
            )
            + "\n"
        )
        mark_stage_complete(out, cmd)
    return problems


def audit_args(pipeline: Pipeline, rd: Path) -> List[str]:
    """stage04b / stage06 options of the audit set (none without audit.dir)."""
    d = auditlib.config_dir(pipeline.cfg)
    if d is None:
        return []
    st = auditlib.settings(pipeline.cfg)
    return [
        "--audit-dir",
        str(d),
        "--audit-target",
        str(st["target"]),
        "--audit-bootstrap",
        str(st["bootstrap"]),
        "--audit-official",
        str(pipeline.get("audit.official")),
    ]


class AuditBuild:
    """The audit set a training run builds itself (audit.build) when
    audit.dir is set and the set is not complete for the current kernel
    pool: `--mode audit` of the run's config snapshot in a child
    orchestrator, which holds the audit directory's lock, leaves out every
    shape of the run's rounds (`--source-run-dir`) and benches on
    audit.devices (default runtime.devices), in resumable parts.

    before    builds or extends it before the invocation's first round; a
              failure stops the run, and the same command resumes it unless
              the build refused a plan of another hipBLASLt build or a
              config key the set was produced with (its pin.json), which the
              same command with --allow-code-change accepts.
    parallel  starts it beside the rounds on audit.devices and waits only
              for its shapes (stage01), which every round leaves out; the
              end of the run joins it. Until then the rounds' audit records
              are partial and their protocol v2 waits for the end of the run.
    off       builds nothing: evaluations stay null or partial until
              `--mode audit` builds it.

    The child gets the invocation's --allow-code-change, and SIGTERM when
    the run's orchestrator dies; it is named in the run's .running.json, so
    that the next invocation stops it before it checks the audit directory's
    lock."""

    LABEL = "audit set build"

    def __init__(self, pipeline: Pipeline, args: argparse.Namespace) -> None:
        self.p = pipeline
        self.dir = auditlib.config_dir(pipeline.cfg)
        self.mode = str(pipeline.get("audit.build"))
        self.devices = audit_devices(pipeline.cfg)
        self.skip_preflight = bool(args.skip_preflight)
        self.allow_change = bool(args.allow_code_change)
        self.log = pipeline.run_dir / "audit_build.log"
        self.child: Optional[Started] = None
        self.started = 0.0

    @property
    def proc(self) -> Optional[subprocess.Popen]:
        return self.child.proc if self.child is not None else None

    def command(self) -> List[str]:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
        cmd += ["--config", str(self.p.snapshot), "--mode", "audit"]
        cmd += ["--run-dir", str(self.dir), "--source-run-dir", str(self.p.run_dir)]
        cmd += ["--devices", ",".join(str(d) for d in self.devices)]
        cmd += ["--stages-dir", str(self.p.stages_dir)]
        cmd += ["--allow-code-change"] if self.allow_change else []
        return cmd + (["--skip-preflight"] if self.skip_preflight else [])

    def prepare(self) -> int:
        """Build (before) or start (parallel) the audit set when it is not
        complete; 0, or the exit status that stops the run."""
        if self.dir is None:
            return 0
        st, why = audit_status(self.p.cfg, self.p.paths)
        if st == "ok":
            ui.ok("audit", f"the audit set {self.dir} is complete for the pool")
            return 0
        if self.mode == "off":
            ui.warn(
                "audit",
                f"audit set {self.dir}: {why}; audit.build is off, so its "
                f"evaluations stay null or partial until --mode audit builds it",
            )
            return 0
        where = "before the rounds" if self.mode == "before" else "beside the rounds"
        ui.banner(f"audit set -- build {self.dir} {where}", ui.C.HEAD)
        ui.info("audit", f"{why}; benching on devices {self.devices}")
        if self.mode == "before":
            t0 = time.time()
            rc = run_logged(
                self.command(),
                self.log,
                env=self.p.stage_env(),
                running=self.p.running,
                label=self.LABEL,
            )
            self.p.timing.add(None, "audit set", rc, time.time() - t0)
            if rc:
                ui.err("audit", self.failure(exit_code(rc)))
                return exit_code(rc)
            st, why = audit_status(self.p.cfg, self.p.paths)
            if st != "ok":
                ui.err("audit", f"the audit set is still not complete: {why}")
                return 2
            return 0
        self.child = start_logged(
            self.command(),
            self.log,
            env=self.p.stage_env(),
            running=self.p.running,
            label=self.LABEL,
        )
        self.started = time.time()
        shapes = self.dir / "stage01" / STAGE_MARKER
        while not shapes.is_file():
            if self.child.proc.poll() is not None and not shapes.is_file():
                rc = self.join()
                ui.err("audit", "the audit set build stopped before its shapes")
                return rc or 2
            time.sleep(0.5)
        ui.ok("audit", "the audit shapes are drawn; every round leaves them out")
        return 0

    def progress(self) -> str:
        assert self.dir is not None
        prog = auditlib.read_json(self.dir / auditlib.PROGRESS) or {}
        done, total = prog.get("shapes_done"), prog.get("shapes_total")
        if done is None or total is None:
            return "starting"
        return f"{done:,} of {total:,} GEMMs benched"

    def join(self) -> int:
        """Wait for a parallel build; its exit status (0 without one)."""
        child = self.child
        if child is None:
            return 0
        if child.proc.poll() is None:
            ui.info("audit", f"waiting for the audit set build ({self.progress()})")
            while True:
                try:
                    child.proc.wait(timeout=600)
                    break
                except subprocess.TimeoutExpired:
                    ui.info("audit", f"audit set build: {self.progress()}")
        rc = exit_code(child.proc.returncode)
        self.child = None
        child.end()
        self.p.timing.add(None, "audit set", rc, time.time() - self.started)
        if rc:
            ui.err("audit", self.failure(rc))
        else:
            ui.ok("audit", f"the audit set build finished ({self.progress()})")
        return rc

    def failure(self, rc: int) -> str:
        """What a build that exited with `rc` leaves to do."""
        assert self.dir is not None
        try:
            refusal = audit_build_refusal(self.dir, self.p.pool_digests())
        except (OSError, ValueError, RuntimeError, KeyError):
            refusal = None
        if refusal:
            return (
                f"building the audit set failed (rc={rc}; see {self.log}): "
                f"{refusal}, so the same command cannot resume it. Bench it "
                f"with that build, point audit.dir at a new directory, or set "
                f"audit.build: off"
            )
        changes = pin_refusal(self.log)
        if changes:
            return (
                f"building the audit set failed (rc={rc}; see {self.log}): "
                f"{self.dir} was produced with other values of the keys its "
                f"benches read ({'; '.join(changes)}). The same command with "
                f"--allow-code-change extends it with this run's, and its "
                f"pin.json records them; else restore them or point audit.dir "
                f"at a new directory"
            )
        return (
            f"building the audit set failed (rc={rc}); see {self.log}; the same "
            f"command resumes it"
        )

    def stop(self) -> None:
        """Stop a parallel build that still runs (its parts resume)."""
        child, self.child = self.child, None
        if child is None:
            return
        if child.proc.poll() is None:
            ui.warn(
                "audit", "stopping the audit set build; the same command resumes it"
            )
            stop_stage(child.proc)
        child.end()


_BRIEF_KEYS = (
    "cell_mean",
    "cell_mean_ci",
    "gemm_geomean",
    "n_gemms",
    "n_cells",
    "n_cells_below_target",
)


def _audit_brief(rec: Any, protocol: str) -> Dict[str, Any]:
    """The exclusive group of an audit record under `protocol`, with v2's
    alongside (`v2`) under the sustained composite."""
    if not isinstance(rec, dict):
        return {}
    g = auditlib.evaluated_group(auditlib.protocol_record(rec, protocol), "exclusive")
    g = g or {}
    out = {
        "model": rec.get("model"),
        "model_sha256": rec.get("model_sha256"),
        "status": rec.get("status"),
        "protocol": protocol,
        **{k: g.get(k) for k in _BRIEF_KEYS},
    }
    if protocol == "sustained":
        g2 = auditlib.evaluated_group(auditlib.protocol_record(rec, "v2"), "exclusive")
        out["v2"] = {k: g2.get(k) for k in _BRIEF_KEYS} if g2 else None
    return out


RELOCATED = ".relocated.json"


def relocate(src: Path, dst: Path) -> None:
    """Move the directory `src` to `dst`, which must not exist, and make the
    paths its JSON records hold (stage markers, deploy record, audit
    records, decision) name `dst`, both as given and resolved. A marker in
    it keeps the move until every record names `dst`, so that
    `finish_relocation` finishes an interrupted one."""
    paths = {
        str(src): str(dst),
        str(src.resolve()): str(dst.parent.resolve() / dst.name),
    }
    (src / RELOCATED).write_text(json.dumps({"paths": paths}, indent=2) + "\n")
    os.rename(src, dst)
    finish_relocation(dst)


def finish_relocation(d: Path) -> None:
    """Make the JSON records under `d` name it where they name the
    directory `relocate` moved it from, then drop its marker; nothing
    without one."""
    rec = _read_json(d / RELOCATED)
    paths = rec.get("paths") if isinstance(rec, dict) else None
    if not isinstance(paths, dict):
        return
    from lib.fs import atomic_write_text

    subs = []
    for old, new in sorted(paths.items(), key=lambda kv: -len(kv[0])):
        if old == new:
            continue
        for ascii_only in (True, False):
            a = json.dumps(old, ensure_ascii=ascii_only)[1:-1]
            b = json.dumps(new, ensure_ascii=ascii_only)[1:-1]
            subs.append((re.compile(re.escape(a) + r"(?![\w.\-])"), b))
    for path in sorted(d.rglob("*.json")):
        if path.name == RELOCATED or not path.is_file():
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        new_text = text
        for pattern, repl in subs:
            new_text = pattern.sub(lambda _m, r=repl: r, new_text)
        if new_text != text:
            atomic_write_text(path, new_text)
    (d / RELOCATED).unlink()


def installed_record(d: Path) -> Optional[Path]:
    """The deploy record of the stage06 in `d` when that stage06 installed
    its bundle into the deploy checkout (deploy.apply_patch)."""
    record = d / "stage06" / "deploy_record.json"
    return record if (_read_json(record) or {}).get("applied") else None


def shipped_text(run_dir: Path, but: Optional[Path] = None) -> str:
    """The bundle the run ships (`run_bundles`, leaving out `but`), as a
    phrase."""
    ships = shipped_round([b for b in run_bundles(run_dir) if b != but])
    if ships is None:
        return "no bundle (the run has deployed none)"
    if round_index(ships) is None:
        names = ", ".join(rb.trained_rounds(ships))
        return f"the final retrain in {ships} (trained on {names})"
    return f"{ships.name}'s bundle"


class FinalRetrain:
    """full.final_retrain, once every planned round of a full run is
    complete: stage05 retrains every cell on every round's enriched data
    (seed benches and trained contexts included, the final round's gate
    slice too) with the final options, the train section with the
    overrides of full.final_retrain.recipe (whose other keys are stage05
    options), into final_retrain/stage05. Protocol v2 registers it with the
    audit set. With gate not-worse, scripts/audit_eval.py scores it and the
    bundle the run ships when it is decided (`baseline`: a round's, or the
    final retrain it would replace; with nothing shipped, the final round's
    bundle, staged or not) on the audit set (audit.json) and it ships only
    when its paired equal-weight cell mean over the exclusive group is not
    below that bundle's (`lib.audit.not_worse`, under audit.official when
    both records have it; decision.json) and, when the run ships that
    bundle, the per-cell gate finds no cell or tail pick that regresses
    against it (`lib.audit.cell_gate` with the config's gate section; a
    decision whose per-cell gate ran with other settings, or that was made
    under another audit.official, is made again); with gate none
    it ships. A shipped final retrain is deployed by stage06 into
    final_retrain/stage06 and is the run's bundle (lib/run_bundles.py),
    which a later round builds on (`Pipeline.base_bundle`), until a later
    round deploys. final_retrain.json records the rounds it trained on and
    its stage05 options; when either changes, or its model-shaping options
    do, it is trained again (`prepare`): in its place when it is no bundle
    of the run, else beside it in final_retrain_next/ (a candidate), which
    replaces it only once its gate accepted it and stage06 deployed it
    (`promote`). Until then, and when the gate rejects it, cannot decide or
    its training fails, the final retrain in final_retrain/ stays."""

    RECORD = rb.RECORD
    DECISION = rb.DECISION
    AUDIT = rb.FINAL_AUDIT

    def __init__(
        self,
        pipeline: Pipeline,
        rounds: Sequence[Path],
        directory: Optional[Path] = None,
    ) -> None:
        self.p = pipeline
        self.rounds = list(rounds)
        self.dir = (
            Path(directory)
            if directory is not None
            else pipeline.run_dir / FINAL_RETRAIN_DIR
        )
        self.gate = str(pipeline.get("full.final_retrain.gate"))
        overrides, self.extra, errors = final_retrain_recipe(pipeline.cfg)
        if errors:
            raise ConfigError(errors)
        self.trainer = pipeline.with_train(overrides)

    @classmethod
    def prepare(cls, pipeline: Pipeline, rounds: Sequence[Path]) -> "FinalRetrain":
        """The final retrain a full run on `rounds` trains, decides and
        deploys. A candidate that an interrupted invocation left accepted
        and deployed is promoted first. Then final_retrain/ when it holds
        none, holds the current one (`stale` names no change), or holds one
        that is no bundle of the run (moved to stale/ to train it again);
        otherwise, while the final retrain there is a bundle of the run but
        out of date, a candidate in final_retrain_next/ (one there that is
        out of date moved to stale/ first), which replaces it only once its
        gate accepted it and stage06 deployed it."""
        run = pipeline.run_dir
        finish_promotion(pipeline)
        cur = cls(pipeline, rounds)
        nxt = cls(pipeline, rounds, run / rb.FINAL_RETRAIN_NEXT_DIR)
        why = cur.stale() if cur.dir.is_dir() else None
        if why is None:
            if nxt.dir.is_dir() and cur.dir.is_dir():
                ui.info(
                    "final",
                    f"the final retrain in {cur.dir} is the current one: the "
                    f"one trained beside it moves to stale/",
                )
            move_aside(run, nxt.dir)
            return cur
        if rb.final_retrain(run) != cur.dir:
            ui.warn("final", f"training the final retrain again: {why}")
            installed = installed_record(cur.dir)
            move_aside(run, cur.dir, nxt.dir)
            if installed is not None:
                ui.warn(
                    "final",
                    f"the final retrain moved to stale/ is no bundle of the run "
                    f"any more, but its stage06 installed its weights into "
                    f"{pipeline.paths.deploy_root}, which still holds them; the "
                    f"run ships {shipped_text(run)}",
                )
            return cur
        again = nxt.stale() if nxt.dir.is_dir() else None
        if again:
            ui.warn("final", f"training the new final retrain again: {again}")
            move_aside(run, nxt.dir)
        if nxt.dir.is_dir():
            ui.info(
                "final",
                f"the final retrain in {cur.dir} is out of date ({why}) and "
                f"stays in place beside the new one in {nxt.dir}",
            )
            return nxt
        ui.warn(
            "final",
            f"training the final retrain again beside the one in {cur.dir}, "
            f"which is out of date ({why}) and stays in place: the new one, in "
            f"{nxt.dir}, replaces it once its gate accepts it and stage06 "
            f"deployed it; until then the run ships {shipped_text(run)}",
        )
        return nxt

    @property
    def candidate(self) -> bool:
        """It trains beside a final retrain that is a bundle of the run
        (final_retrain_next/)."""
        return self.dir.name == rb.FINAL_RETRAIN_NEXT_DIR

    @property
    def name(self) -> str:
        return "the new final retrain" if self.candidate else "the final retrain"

    @property
    def models(self) -> Path:
        return self.dir / "stage05" / "models.pt"

    def args(self) -> List[str]:
        last = self.rounds[-1]
        step = RoundStep(len(self.rounds), "final", list(self.rounds), None, True)
        data = [rd / "stage04" for rd in self.rounds if (rd / "stage04").is_dir()]
        args = self.trainer.args_stage05(
            last, step, None, data_dirs=data, out_dir=self.dir / "stage05"
        )
        return args + self.extra

    def trained_on(self) -> Dict[str, Any]:
        """{round: its completion} of the rounds it trains on
        (`lib.run_bundles.round_stamps`); a round that runs again writes a
        new marker."""
        return rb.round_stamps(self.rounds)

    def training_options(self) -> Dict[str, Any]:
        """Its stage05 options but those that only place its inputs,
        outputs and work (FINAL_RETRAIN_PLACEMENT), as stage05 records them
        (`recorded_options`) and as JSON values: the optimizer's budget and
        schedule count, which a round's carried cells do not retrain for."""
        s5 = stage05_module()
        ns, _ = s5.arg_parser(s5._RaisingParser).parse_known_args(self.args())
        opts = {
            k: v
            for k, v in s5.recorded_options(ns).items()
            if k not in FINAL_RETRAIN_PLACEMENT
        }
        return json.loads(json.dumps(s5.jsonable(opts)))

    def model_options(self) -> Dict[str, Any]:
        """The model-shaping options (stage05 `model_options`) it trains
        with now."""
        return stage05_module().command_model_options(self.args())

    def stale(self) -> Optional[str]:
        """Why the final retrain in final_retrain/ is out of date, else
        None. Its rounds count as changed when they are other rounds, or one
        of them completed at another time (`lib.run_bundles.changed_rounds`,
        as the readers of its bundle judge it, so that a copy of the run
        directory keeps it). Its recorded stage05 options are compared with
        today's; an option the record does not name predates it and had
        stage05's value from before it was an option, else its default, and
        an option stage05 no longer defines is not compared."""
        rec = _read_json(self.dir / self.RECORD)
        if isinstance(rec, dict):
            recorded = rec.get("rounds")
            names = sorted(rd.name for rd in self.rounds)
            if (
                not isinstance(recorded, dict)
                or sorted(recorded) != names
                or rb.changed_rounds(self.p.run_dir, recorded)
            ):
                return "the rounds changed since it was trained"
        then = rec.get("options") if isinstance(rec, dict) else None
        if isinstance(then, dict):
            now = self.training_options()
            s5 = stage05_module()
            ap = s5.arg_parser()

            def was(key: str) -> Any:
                if key in then:
                    return then[key]
                if key in s5.PREDATING_OPTIONS:
                    return s5.PREDATING_OPTIONS[key]
                return json.loads(json.dumps(s5.jsonable(ap.get_default(key))))

            diff = {k: (was(k), now[k]) for k in sorted(now) if was(k) != now[k]}
            if diff:
                text = describe_option_changes({"final": diff})
                return f"it was trained with other stage05 options ({text})"
        if not stage_complete(self.dir / "stage05"):
            return None
        trained = trained_model_options(self.dir)
        if trained is None:
            return None
        s5 = stage05_module()
        changed = s5.changed_model_options(
            trained, s5.command_model_options(self.args())
        )
        if changed:
            text = describe_option_changes({"final": changed})
            return f"it was trained with other options ({text})"
        return None

    def lineage(self) -> List[Path]:
        """The bundles before it, in the order the run deploys them: the
        rounds it trains on and, when the run shipped a final retrain before
        the last of them, that one (later rounds built on it). Its stage05
        carries the cells it cannot train from the last round
        (`record_cell_origins`)."""
        last = round_index(self.rounds[-1]) if self.rounds else None
        if last is None:
            return []
        return [
            b
            for b in run_bundles(self.p.run_dir)
            if b != self.dir and rb.bundle_key(b) <= (last, 0)
        ]

    def train(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (self.dir / self.RECORD).is_file():
            rec = {
                "rounds": self.trained_on(),
                "recipe": dict(self.p.get("full.final_retrain.recipe") or {}),
                "options": self.training_options(),
                "started": datetime.now().isoformat(timespec="seconds"),
            }
            (self.dir / self.RECORD).write_text(json.dumps(rec, indent=2) + "\n")
        if stage_complete(self.dir / "stage05"):
            ui.ok("final", f"{self.name} is trained ({self.dir / 'stage05'})")
            return 0
        ui.banner(
            f"{'new ' if self.candidate else ''}final retrain -- every cell on "
            f"the data of {len(self.rounds)} rounds",
            ui.C.HEAD,
        )
        rc = self.p.run_stage(
            self.dir, "stage05", self.args(), None, timing_stage="final stage05"
        )
        if rc:
            self.report_failure(rc)
            return exit_code(rc)
        record_cell_origins(self.dir / "stage05", self.lineage())
        return 0

    def report_failure(self, rc: int) -> None:
        """Why the final retrain stops before its gate, what ships
        meanwhile, and what resumes it."""
        metrics_path = self.dir / "stage05" / "metrics.json"
        if rc == STAGE05_CELLS_NOT_TRAINED:
            metrics = _read_json(metrics_path) or {}
            failed = list(metrics.get("cells_failed") or [])
            missing = [
                str(r.get("cell")) for r in metrics.get("cells_not_trained") or []
            ]
            what = (
                f"stage05 did not train every cell of {self.name} (failed: "
                f"{failed or '-'}; not trained: {missing or '-'}; see {metrics_path})"
            )
        else:
            what = (
                f"{self.name}'s stage05 failed (rc={exit_code(rc)}; see "
                f"{self.dir / 'stage05.log'})"
            )
        ui.err(
            "final",
            f"{what}. It stops before its gate, and the run ships "
            f"{shipped_text(self.p.run_dir, self.dir)} meanwhile: fix the cause "
            f"and run the same command to train it again.",
        )

    def baseline(self) -> Tuple[Optional[Path], bool]:
        """(the bundle its gate compares it with, whether the run deployed
        that bundle): the bundle the run ships (`run_bundles`: a round's, or
        the final retrain a candidate would replace), else the final round's
        bundle, deployed or not."""
        ships = shipped_round([b for b in run_bundles(self.p.run_dir) if b != self.dir])
        if ships is not None:
            return ships, True
        return (self.rounds[-1] if self.rounds else None), False

    @staticmethod
    def bundle_text(base: Path, deployed: bool) -> str:
        if round_index(base) is None:
            names = ", ".join(rb.trained_rounds(base))
            return f"the shipped final retrain (trained on {names})"
        return f"{base.name}'s bundle" + ("" if deployed else " (not deployed)")

    def evaluate(
        self, baseline: Optional[Path]
    ) -> Optional[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """(the record of `baseline`'s bundle, the final retrain's) on the
        audit set from scripts/audit_eval.py (`audit.json`); None when it
        fails. The baseline is scored as its round dir (stage06's staged
        bundle, the bytes it deployed, else its models.pt), leaving out the
        shapes of the rounds it trained on."""
        out = self.dir / self.AUDIT
        cmd = [sys.executable, "-u", str(AUDIT_EVAL_SCRIPT)]
        if baseline is not None:
            cmd += ["--round-dir", str(baseline)]
        cmd += ["--model-file", str(self.models)]
        for rd in self.rounds:
            cmd += ["--trained-round-dir", str(rd)]
        cmd += ["--out", str(out), *self.p.audit_eval_common()]
        ui.banner(
            f"{'new ' if self.candidate else ''}final retrain -- on the audit set",
            ui.C.BLUE,
        )
        rc = self.p.run_tool(self.dir, "audit_eval", cmd, None)
        doc = _read_json(out)
        if rc or not isinstance(doc, dict):
            ui.err("final", f"audit_eval failed (rc={rc}); see {self.dir}")
            return None
        recs = doc.get("models") if baseline is not None else [doc]
        if not isinstance(recs, list) or len(recs) != (2 if baseline else 1):
            return None
        final = dict(recs[-1], what=FINAL_RETRAIN_DIR)
        before = recs[0] if baseline is not None else None
        if before is not None and round_index(baseline) is None:
            before = dict(before, what=f"shipped {baseline.name}")
        return before, final

    def decide(self) -> Optional[Dict[str, Any]]:
        """The gate's decision (see the class docstring); None when it
        cannot be made now: the audit set is not complete, or a model was
        not measured on it."""
        base, deployed = self.baseline()
        if not deployed and self.gate != "not-worse":
            base = None
        official = str(self.p.get("audit.official"))
        dec: Dict[str, Any] = {
            "gate": self.gate,
            "official": official,
            "rounds": [rd.name for rd in self.rounds],
            "models": str(self.models),
            "baseline": None,
            "final": None,
            "protocol": None,
            "change": None,
            "selected_by_audit": False,
        }
        what = self.bundle_text(base, deployed) if base is not None else ""
        needed = self.gate == "not-worse" and base is not None
        st, why = (
            audit_status(self.p.cfg, self.p.paths)
            if auditlib.config_dir(self.p.cfg) is not None
            else ("missing", "no audit set is configured")
        )
        if needed and st != "ok":
            ui.err(
                "final",
                f"{self.name} is compared with {what} on the audit set, which "
                f"is not complete ({why}); the same command decides once it is",
            )
            return None
        recs = self.evaluate(base) if st == "ok" else None
        if needed and (
            recs is None
            or not (auditlib.is_measured(recs[0]) and auditlib.is_measured(recs[1]))
        ):
            ui.err(
                "final",
                f"the audit set did not measure {self.name} and {what} (see "
                f"{self.dir / self.AUDIT}); the same command tries again",
            )
            return None
        if self.gate == "none" or base is None:
            accepted, reason = True, "full.final_retrain.gate is none"
        else:
            assert recs is not None
            verdict, text, change, proto = auditlib.not_worse(
                recs[0], recs[1], official=official
            )
            if proto != official:
                text += (
                    f"; the official protocol ({auditlib.PROTOCOLS[official]}) "
                    f"has not measured both"
                )
            dec.update(protocol=proto, change=change)
            if verdict is None:
                accepted = False
                reason = f"not comparable with {what}: {text}"
            else:
                accepted = verdict
                reason = (
                    f"{'not worse' if verdict else 'worse'} than {what} on the "
                    f"audit set: {text}"
                )
                dec.update(
                    selected_by_audit=True,
                    selection_note=(
                        f"the audit set chose which of the final retrain and "
                        f"{what} ships, so its numbers for the bundle that "
                        f"ships come from the GEMMs that chose it and carry a "
                        f"small optimistic bias"
                    ),
                )
                cells = self.cell_gate(recs, deployed)
                if cells is not None:
                    dec["cell_gate"] = cells
                    dec["cell_gate_settings"] = auditlib.cell_gate_settings(self.p.cfg)
                    if cells["passed"] is not True:
                        accepted = False
                        reason += (
                            f"; the per-cell gate keeps it out: {cells['reason']}"
                            if cells["passed"] is False
                            else f"; the per-cell gate cannot decide: {cells['reason']}"
                        )
        proto = dec["protocol"] or (
            auditlib.view(recs[1], official)[1] if recs is not None else "v1"
        )
        if base is not None:
            dec["baseline"] = {"round": base.name, "deployed": deployed}
            if round_index(base) is None:
                dec["baseline"]["trained_on"] = list(rb.trained_rounds(base))
        if recs is not None:
            dec["final"] = _audit_brief(recs[1], proto)
            if base is not None:
                dec["baseline"].update(_audit_brief(recs[0], proto))
        dec.update(
            accepted=accepted,
            reason=reason,
            decided=datetime.now().isoformat(timespec="seconds"),
        )
        return dec

    def cell_gate(
        self, recs: Tuple[Optional[Dict[str, Any]], Dict[str, Any]], deployed: bool
    ) -> Optional[Dict[str, Any]]:
        """The per-cell gate (`lib.audit.cell_gate`, the config's gate
        section) of the final retrain against the bundle the run ships;
        None when gate.per_cell is off or the run ships nothing, the
        baseline being then the final round's bundle, not deployed."""
        settings = auditlib.cell_gate_settings(self.p.cfg)
        if settings is None or not deployed or recs[0] is None:
            return None
        bootstrap = int(auditlib.settings(self.p.cfg)["bootstrap"])
        return auditlib.cell_gate(
            recs[0],
            recs[1],
            settings,
            bootstrap,
            official=str(self.p.get("audit.official")),
        )

    def official_changed(self, dec: Mapping[str, Any]) -> bool:
        """`dec` compared the bundles under another audit.official than the
        config's now; one that compared none (gate none, nothing to compare
        with) or was made before the setting existed stands."""
        then = dec.get("official")
        if then is None or dec.get("protocol") is None:
            return False
        return then != str(self.p.get("audit.official"))

    def cell_gate_changed(self, dec: Mapping[str, Any]) -> bool:
        """`dec` ran the per-cell gate with other settings than the
        config's now (or the config turns it off); a decision the per-cell
        gate did not take part in (none ran: gate none, nothing shipped,
        gate.per_cell off, or made before it existed) stands."""
        if not isinstance(dec.get("cell_gate"), dict):
            return False
        return dec.get("cell_gate_settings") != auditlib.cell_gate_settings(self.p.cfg)

    def print_decision(self, dec: Mapping[str, Any]) -> None:
        say = ui.ok if dec.get("accepted") else ui.warn
        verb = "ships" if dec.get("accepted") else "does not ship"
        say("final", f"{self.name} {verb}: {dec.get('reason')}")
        for key in ("baseline", "final"):
            b = dec.get(key) or {}
            if b.get("cell_mean") is None:
                continue
            label = (
                self.name
                if key == "final"
                else (
                    "the shipped final retrain"
                    if b.get("trained_on") is not None
                    else b.get("round")
                )
            )
            ci = b.get("cell_mean_ci")
            ui.info(
                "final",
                f"  {label}: audit cell mean "
                f"{b['cell_mean'] * 100:.2f}%"
                + (f" [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]" if ci else "")
                + f" over {b.get('n_cells')} cells, {b.get('n_cells_below_target')} "
                f"below the target (protocol {b.get('protocol')})",
            )
            v2 = b.get("v2") or {}
            if v2.get("cell_mean") is not None:
                ui.info(
                    "final",
                    f"  {label}: v2 alongside {v2['cell_mean'] * 100:.2f}%"
                    + _ci_text(v2.get("cell_mean_ci"))
                    + f", {v2.get('n_cells_below_target')} cells below the target",
                )
        cells = dec.get("cell_gate")
        if isinstance(cells, dict) and cells.get("passed"):
            ui.info("final", f"  per-cell gate: {cells.get('reason')}")

    def warn_installed(self, record: Path) -> None:
        """A gate change kept out the final retrain that the stage06 of
        `record` (now in stale/) installed into the deploy checkout
        (deploy.apply_patch): the checkout still holds its weights."""
        ships = latest_deploy_record(
            [b for b in run_bundles(self.p.run_dir) if b != self.dir]
        )
        what = (
            f"{shipped_text(self.p.run_dir, self.dir)} (its deploy record: {ships})"
            if ships is not None
            else "a bundle that ships"
        )
        ui.warn(
            "final",
            f"{self.name} no longer ships (full.final_retrain.gate "
            f"{self.gate}), but its earlier stage06 installed its weights into "
            f"{self.p.paths.deploy_root}, which still holds them: install "
            f"{what} there again",
        )

    def stage06_args(self) -> List[str]:
        args = self.p.args_stage06(self.dir, None)
        if auditlib.config_dir(self.p.cfg) is not None:
            for rd in self.rounds:
                args += ["--audit-trained-round-dir", str(rd)]
        return args

    def deployed(self) -> bool:
        """Its gate accepted it and its stage06 deployed it."""
        dec = _read_json(self.dir / self.DECISION)
        return (
            isinstance(dec, dict)
            and bool(dec.get("accepted"))
            and stage_complete(self.dir / "stage06")
            and (self.dir / "stage06" / "deploy_record.json").is_file()
        )

    def promote(self) -> None:
        """Put this candidate, accepted and deployed, in final_retrain/'s
        place: the final retrain there moves to stale/, and the paths the
        candidate's records hold name final_retrain/ (`relocate`)."""
        dest = self.p.run_dir / FINAL_RETRAIN_DIR
        move_aside(self.p.run_dir, dest)
        relocate(self.dir, dest)
        self.dir = dest
        ui.ok(
            "final",
            f"the new final retrain replaces the one before it in {dest} (moved "
            f"to {self.p.run_dir / 'stale'})",
        )

    def report_deploy_failure(self, rc: int) -> None:
        """A candidate that its gate accepted and whose stage06 failed."""
        installs = self.p.get("deploy.apply_patch") and self.p.paths.deploy_root
        ui.err(
            "final",
            f"{self.name} passed its gate, but its stage06 failed (rc="
            f"{exit_code(rc)}; see {self.dir / 'stage06.log'}): the run ships "
            f"{shipped_text(self.p.run_dir, self.dir)} meanwhile; fix the cause "
            f"and run the same command to deploy it"
            + (
                f". Its stage06 may have written into {self.p.paths.deploy_root} "
                f"before it failed"
                if installs
                else ""
            ),
        )

    def decide_and_deploy(self) -> Tuple[int, bool]:
        """(exit status, whether this call deployed it): the decision
        (decision.json, made again when the gate changed) and, when it
        ships, stage06 into its stage06/; a candidate then replaces the
        final retrain in final_retrain/ (`promote`)."""
        path = self.dir / self.DECISION
        dec = _read_json(path)
        installed = None
        if isinstance(dec, dict) and (
            dec.get("gate") != self.gate
            or self.cell_gate_changed(dec)
            or self.official_changed(dec)
        ):
            if dec.get("accepted"):
                installed = installed_record(self.dir)
            move_aside(self.dir, path, self.dir / "stage06", self.dir / "stage06.log")
            dec = None
        if not isinstance(dec, dict):
            dec = self.decide()
            if dec is None:
                return 1, False
            path.write_text(json.dumps(_json_safe(dec), indent=2) + "\n")
            if installed is not None and not dec.get("accepted"):
                self.warn_installed(installed)
        self.print_decision(dec)
        if not dec.get("accepted"):
            return 0, False
        deployed = False
        if not stage_complete(self.dir / "stage06"):
            rc = self.p.run_stage(
                self.dir,
                "stage06",
                self.stage06_args(),
                None,
                timing_stage="final stage06",
            )
            if rc:
                if self.candidate:
                    self.report_deploy_failure(rc)
                return exit_code(rc), False
            deployed = True
        if self.candidate:
            self.promote()
        return 0, deployed


def finish_promotion(pipeline: Pipeline) -> None:
    """What an interrupted invocation left of a new final retrain taking
    final_retrain/'s place: the rewrite of its records (`finish_relocation`)
    and, when its gate accepted it and stage06 deployed it, the move itself
    (`FinalRetrain.promote`)."""
    run = pipeline.run_dir
    finish_relocation(run / FINAL_RETRAIN_DIR)
    nxt = run / rb.FINAL_RETRAIN_NEXT_DIR
    if not nxt.is_dir():
        return
    pending = FinalRetrain(pipeline, complete_prefix(run), nxt)
    if pending.deployed():
        pending.promote()


def run_bundles(run_dir: Path) -> List[Path]:
    """The directories whose bundles the run may ship, in the order it
    deploys them: its complete rounds and its final retrain while that is
    one of its bundles (lib/run_bundles.py)."""
    return rb.bundles(run_dir, complete_prefix(run_dir))


# ── main ────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="tilewright training pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, required=True, help="pipeline config yaml")
    ap.add_argument(
        "--mode",
        choices=("initial", "active", "full", "validate", "enhance", "audit"),
        required=True,
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run directory (default: runs/<config stem>_<timestamp>)",
    )
    ap.add_argument(
        "--deploy",
        action="store_true",
        help="deploy after every round, round_0 included",
    )
    ap.add_argument("--from-round", type=int, default=None, metavar="R")
    ap.add_argument("--from-stage", default=None, metavar="N")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the checks of the bench, library and tilewright module",
    )
    ap.add_argument(
        "--stages-dir",
        type=Path,
        default=THIS_DIR / "stages",
        help="directory holding the stage scripts",
    )
    ap.add_argument("--no-report", action="store_true", help="do not write report.html")
    ap.add_argument(
        "--allow-code-change",
        action="store_true",
        help="go on although the pipeline code, the engine module or a pinned "
        "config key differ from what the run was produced with (<run-dir>/"
        "pin.json records the change)",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="also print Origami's numbers, as an external reference, in the "
        "stages and the summary, and add the report's reference section; "
        "they never enter a decision",
    )
    en = ap.add_argument_group("enhance mode")
    en.add_argument(
        "--cells",
        default="",
        metavar="L1,L2",
        help="cell labels to enhance (a split parent stands for its leaves)",
    )
    en.add_argument(
        "--weakest",
        type=int,
        default=None,
        metavar="N",
        help="also the N weakest cells below the target (or --below)",
    )
    en.add_argument(
        "--below",
        type=float,
        default=None,
        metavar="T",
        help="also every cell whose selection efficiency is below T (0-1)",
    )
    en.add_argument(
        "--rank-by",
        choices=ce.RANK_KEYS,
        default="sel_eff",
        help="order of --weakest: lowest sel_eff, or largest share x gap (gain)",
    )
    en.add_argument("--enhance-rounds", type=int, default=1, metavar="N")
    en.add_argument(
        "--shapes-per-cell",
        type=int,
        default=None,
        metavar="N",
        help="new shapes per enhanced leaf (default seed.shapes_per_cell)",
    )
    en.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selection and the stage commands; write nothing",
    )
    au = ap.add_argument_group("audit mode")
    au.add_argument(
        "--source-run-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="training run whose round shapes the audit set must not contain "
        "(repeatable)",
    )
    au.add_argument(
        "--devices",
        default=None,
        metavar="D1,D2",
        help="bench devices of the audit (default runtime.devices)",
    )
    args = ap.parse_args(argv)
    if args.from_stage is not None:
        try:
            args.from_stage, note = parse_stage(args.from_stage)
        except ValueError as e:
            ap.error(str(e))
        if note:
            ui.warn("args", note)
    if args.mode == "audit":
        bad = [
            flag
            for flag, given in (
                ("--deploy", args.deploy),
                ("--from-round", args.from_round is not None),
                ("--from-stage", args.from_stage is not None),
            )
            if given
        ]
        if bad:
            ap.error(f"{', '.join(bad)}: not with --mode audit")
        if args.run_dir is None:
            ap.error("--mode audit needs --run-dir (the audit directory)")
        if args.devices is not None:
            try:
                args.devices = [int(d) for d in args.devices.split(",") if d.strip()]
            except ValueError:
                ap.error("--devices is a comma list of device indices")
            if not args.devices:
                ap.error("--devices names no device")
        args.no_report = True
    elif args.source_run_dir or args.devices is not None:
        ap.error("--source-run-dir, --devices: only with --mode audit")
    args.cell_list = [c.strip() for c in args.cells.split(",") if c.strip()]
    enhance_opts = {
        "--cells": bool(args.cell_list),
        "--weakest": args.weakest is not None,
        "--below": args.below is not None,
        "--rank-by": args.rank_by != "sel_eff",
        "--enhance-rounds": args.enhance_rounds != 1,
        "--shapes-per-cell": args.shapes_per_cell is not None,
        "--dry-run": args.dry_run,
    }
    if args.mode != "enhance":
        given = [k for k, v in enhance_opts.items() if v]
        if given:
            ap.error(f"{', '.join(given)}: only with --mode enhance")
    else:
        if not (args.cell_list or args.weakest is not None or args.below is not None):
            ap.error("--mode enhance needs --cells, --weakest or --below")
        if args.weakest is not None and args.weakest < 1:
            ap.error("--weakest must be >= 1")
        if args.below is not None and not 0.0 < args.below <= 1.0:
            ap.error("--below is a selection efficiency in (0, 1]")
        if args.enhance_rounds < 1:
            ap.error("--enhance-rounds must be >= 1")
        if args.shapes_per_cell is not None and args.shapes_per_cell < 1:
            ap.error("--shapes-per-cell must be >= 1")
        if args.from_round is not None:
            ap.error("--mode enhance always runs the next round; drop --from-round")
    if args.mode == "validate":
        if args.from_stage is not None and args.from_stage not in (7.0, 8.0):
            ap.error("--mode validate takes --from-stage 7 or 8")
    elif args.from_stage is not None and args.from_stage not in TRAINING_STAGE_NUMBERS:
        ap.error(
            f"--mode {args.mode} takes --from-stage "
            + ", ".join(f"{n:g}" for n in TRAINING_STAGE_NUMBERS[:-1])
            + f" or {TRAINING_STAGE_NUMBERS[-1]:g}, or the stage's name"
        )
    return args


def _print_problems(tag: str, errors: List[str], warnings: List[str]) -> None:
    for w in warnings:
        ui.warn(tag, w)
    for e in errors:
        ui.err(tag, e)


def lock_run_dir(run_dir: Path) -> Optional[int]:
    """Descriptor holding an exclusive lock on `<run_dir>/.lock`, or None
    when another process holds it."""
    fd = os.open(run_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _audit_dir_errors(
    args: argparse.Namespace, cfg: Mapping[str, Any], run_dir: Path
) -> List[str]:
    """An audit directory is never a training run directory, nor inside
    one."""
    configured = auditlib.config_dir(cfg)
    if args.mode != "audit":
        if configured is not None and configured.resolve() == run_dir:
            return [f"audit.dir is this run directory {run_dir}"]
        if (run_dir / auditlib.MANIFEST).is_file():
            return [f"{run_dir} is an audit directory; train in another --run-dir"]
        if (
            configured is not None
            and args.mode != "validate"
            and cfg_get(cfg, "audit.build") != "off"
            and run_dir in configured.resolve().parents
        ):
            return [
                f"audit.dir {configured} lies inside this run directory; the "
                f"run builds its audit set, which needs a directory of its own"
            ]
        return []
    errs = audit_location_errors(run_dir)
    for src in args.source_run_dir:
        s = src.resolve()
        if not s.is_dir():
            errs.append(f"--source-run-dir {src} does not exist")
        elif s == run_dir:
            errs.append("--source-run-dir is the audit directory itself")
    return errs


def _engine_module_path() -> Optional[str]:
    try:
        from lib import evaluate

        return str(getattr(evaluate.tilewright_module(), "__file__", None))
    except Exception:  # noqa: BLE001
        return None


def _raise_interrupted(signum: int, _frame: Any) -> None:
    raise Interrupted(signum)


def stop_survivors(run_dir: Path) -> List[str]:
    """Stop what an earlier invocation that died left running
    (`attempts.stop_survivors`); errors when some of it does not stop, or
    when the orchestrator that started it still runs on this run directory's
    path without its lock (the directory was replaced while it ran)."""
    found = attempts.stop_survivors(run_dir)
    for label, where in found.elsewhere:
        ui.warn(
            "resume",
            f"{attempts.RUNNING_FILE} names {label} of {where}, another run "
            f"directory (this one is a copy of it?); left alone",
        )
    for label, pids in found.stopped:
        ui.warn(
            "resume",
            f"stopped {label} of an earlier invocation that died while it ran "
            f"(pids {pids})",
        )
    errs = [
        f"{label} of an earlier invocation is still running and does not stop "
        f"(pids {pids})"
        for label, pids in found.stuck
    ]
    errs += [
        f"the orchestrator that runs {label} (pid {pid}) is still alive and "
        f"writes into {run_dir}; stop it first"
        for label, pid in found.alive
    ]
    return errs


PIN_REFUSAL = "this invocation differs from what"


def pin_refusal(log: Path) -> List[str]:
    """The changes a pin refusal (`settle_run_pin`) names in `log`, the
    output of one invocation; empty when it holds none."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return []
    lines = _ANSI_RE.sub("", text).splitlines()
    tag = "[pin] "
    for i, line in enumerate(lines):
        if not (line.startswith(tag) and PIN_REFUSAL in line):
            continue
        changes = []
        for nxt in lines[i + 1 :]:
            body = nxt[len(tag) :] if nxt.startswith(tag) else ""
            if not body.startswith("  "):
                break
            changes.append(body.strip())
        return changes
    return []


def settle_run_pin(
    run_dir: Path,
    pin: Mapping[str, Any],
    allow_change: bool,
    parts: Sequence[str] = pinning.PARTS,
) -> List[str]:
    """Errors when `pin` differs from the pin the run was produced with
    (`<run_dir>/pin.json`, lib/pinning.py) in the `parts` the invocation
    must keep (`pin_parts`) and the change is not allowed. A change is
    recorded at once (an allowed one, and one of a part the invocation need
    not keep), except by an invocation that keeps no part (validation),
    which only names it; a run without a pin gets this one when its first
    stage completes."""
    recorded = pinning.read_run_pin(run_dir)
    if recorded is None:
        return []
    changes = pinning.differences(recorded["pin"], pin)
    if not changes:
        return []
    kept = pinning.differences(recorded["pin"], pin, parts)
    where = run_dir / pinning.RUN_PIN
    if kept and not allow_change:
        return [
            f"{PIN_REFUSAL} {run_dir} was produced with ({where}):",
            *(f"  {c}" for c in kept),
            "pass --allow-code-change to go on with them (the change is recorded "
            "there), or restore what the run recorded",
        ]
    if not parts:
        for c in changes:
            ui.warn("pin", f"validating with a change of what the run used: {c}")
        return []
    for c in changes:
        how = "accepted (--allow-code-change)" if c in kept else "recorded"
        ui.warn("pin", f"{how}: {c}")
    pinning.write_run_pin(run_dir, pin, previous=recorded, accepted=changes)
    return []


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg, raw = load_config(args.config)
    except ConfigError as e:
        _print_problems("cfg", e.errors, [])
        return 2
    errors, warnings = validate_config(cfg, args.mode)
    _print_problems("cfg", errors, warnings)
    if errors:
        return 2

    run_dir = (
        args.run_dir
        or RUNS / f"{args.config.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    if args.mode in ("active", "validate", "enhance") and not run_dir.is_dir():
        ui.err("cfg", f"--mode {args.mode} needs an existing --run-dir: {run_dir}")
        return 2
    if args.mode == "enhance" and args.dry_run:
        return enhance_dry_run(args, cfg, run_dir)
    errs = _audit_dir_errors(args, cfg, run_dir)
    if errs:
        _print_problems("cfg", errs, [])
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_run_dir(run_dir)
    if lock is None:
        ui.err(
            "cfg",
            f"another orchestrator is using {run_dir} (it holds "
            f"{run_dir / LOCK_NAME}); wait for it to finish or pick another "
            f"--run-dir",
        )
        return 2
    previous: Dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        previous = {
            s: signal.signal(s, _raise_interrupted)
            for s in (signal.SIGTERM, signal.SIGHUP)
        }
    try:
        return _main_locked(args, cfg, raw, run_dir)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        os.close(lock)


def plan_enhance_from_args(
    args: argparse.Namespace, cfg: Dict[str, Any], run_dir: Path
) -> List[RoundStep]:
    target = ce.absolute_target(cfg_get(cfg, "validate.sel_eff_threshold"))

    def select(rounds: List[Path]) -> Dict[str, Any]:
        return select_enhance_cells(
            run_dir,
            rounds,
            cells=args.cell_list,
            weakest=args.weakest,
            below=args.below,
            rank_by=args.rank_by,
            target=target,
        )

    return plan_enhance(
        run_dir,
        n_rounds=args.enhance_rounds,
        from_stage=args.from_stage,
        shapes_per_cell=args.shapes_per_cell,
        select=select,
    )


def _print_enhance_selection(steps: List[RoundStep]) -> None:
    sel = steps[0].selection or {}
    target = sel.get("target")
    ui.info("enhance", f"cells ({len(steps[0].targets or [])}):")
    for c in steps[0].targets or []:
        print(f"    {c}", flush=True)
    picked = sel.get("picked") or []
    if picked:
        ui.info(
            "enhance",
            f"picked by training-side evidence (rank by {sel.get('rank_by')}, "
            f"target {(target or 0) * 100:.2f}%):",
        )
        for e in picked:
            print(
                f"    {e['cell']:40s} sel_eff={e['sel_eff'] * 100:6.2f}% "
                f"({e['source']}, n_eval={e['n_eval']})  gemms={e['n_gemms']} "
                f"share={e['share'] * 100:5.2f}%  gain={e['gain'] * 100:.3f}%",
                flush=True,
            )


def enhance_dry_run(
    args: argparse.Namespace, cfg: Dict[str, Any], run_dir: Path
) -> int:
    """Print what --mode enhance would run; writes nothing."""
    try:
        cfg, resolved = resolve_contexts(cfg, int(cfg["runtime"]["devices"][0]))
        steps = plan_enhance_from_args(args, cfg, run_dir)
    except ConfigError as e:
        _print_problems("plan", e.errors, [])
        return 2
    for line in resolved:
        ui.info("contexts", line)
    try:
        pipeline = Pipeline(
            cfg=cfg,
            snapshot=args.config.resolve(),
            run_dir=run_dir,
            paths=derive_paths(cfg),
            stages_dir=args.stages_dir.resolve(),
            timing=Timing(),
            force_deploy=args.deploy,
            origami_reference=origami_reference_on(args, cfg),
        )
    except ValueError as e:
        _print_problems("plan", [str(e)], [])
        return 2
    ui.banner(f"enhance dry run  run_dir={run_dir}", ui.C.HEAD)
    _print_enhance_selection(steps)

    def show(key: str, stage_args: List[str]) -> None:
        script = pipeline.stages_dir / STAGE_DEFS[key].script
        print(f"  [{key}] " + shlex.join([sys.executable, str(script), *stage_args]))

    first = steps[0].index
    for step in steps:
        rd = run_dir / f"round_{step.index}"
        ui.info(
            "enhance",
            f"round_{step.index}" + ("  (final round)" if step.is_final else ""),
        )
        show("stage01", pipeline.args_stage01(rd, step))
        show("stage03", pipeline.args_stage03(rd))
        if pipeline.rebench:
            show("stage03b", pipeline.args_stage03b(rd))
        if pipeline.contexts:
            show("stage03c", pipeline.args_stage03c(rd))
        if pipeline.sustained:
            show("stage03d", pipeline.args_stage03d(rd))
        show("stage04", pipeline.args_stage04(rd))
        base = pipeline.base_bundle(step) if step.index == first else None
        models_dir = (base or step.priors[-1]) / "stage05"
        show("stage04b", pipeline.args_stage04b(rd, step, models_dir))
        gated = float(pipeline.get("deploy.gate_frac")) > 0
        stage05 = pipeline.args_stage05(rd, step, "<RETRAIN>")
        if gated:
            stage05 = [
                str(rd / "stage04b" / "train") if a == str(rd / "stage04") else a
                for a in stage05
            ]
        show("stage05", stage05)
        print(
            "    <RETRAIN>: the leaves above with at least "
            f"{pipeline.get('train.min_cell_gemms')} training GEMMs after "
            "stage04b, split children replacing a split leaf, and the cells "
            "carried forward that were trained with other model-shaping "
            "options",
            flush=True,
        )
        deploys = pipeline.force_deploy or (
            step.is_final and bool(pipeline.get("deploy.on_final_round"))
        )
        if deploys:
            gate = None
            if gated:
                gate = GatePlan(rd / "stage04b" / "gate")
                if not pipeline.force_deploy:
                    gate.target = pipeline.get("deploy.target_sel_eff")
                    gate.fallback = "not-worse"
            show("stage06", pipeline.args_stage06(rd, gate))
        else:
            print("  [stage06] not run (no --deploy, not a deploying round)")
    return 0


def origami_reference_on(args: argparse.Namespace, cfg: Mapping[str, Any]) -> bool:
    """--origami-reference or report.origami_reference."""
    return bool(args.origami_reference or cfg_get(cfg, "report.origami_reference"))


def _main_locked(
    args: argparse.Namespace, cfg: Dict[str, Any], raw: str, run_dir: Path
) -> int:
    global _ORIG_STDOUT
    log_fh = (run_dir / "orchestrator.log").open("a", buffering=1)
    log_fh.write(
        f"\n{'=' * 80}\n# {datetime.now().isoformat(timespec='seconds')}  "
        f"mode={args.mode}  config={args.config}\n# argv={sys.argv}\n{'=' * 80}\n"
    )
    _ORIG_STDOUT = sys.stdout
    orig_err = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    timing = Timing()
    status = None
    if args.mode != "audit":
        status = RunStatus(run_dir, args.mode)
        status.cfg = cfg
    rc: Optional[int] = None
    try:
        rc = _main_logged(args, cfg, raw, run_dir, timing, status)
        return rc
    except KeyboardInterrupt as e:
        signum = getattr(e, "signum", signal.SIGINT)
        ui.err(
            "interrupt",
            f"{signal.Signals(signum).name}: stopped; run the same command to "
            f"resume the interrupted round",
        )
        rc = 128 + signum
        return rc
    finally:
        timing.print_summary()
        try:
            timing.write(run_dir, args.mode)
        except OSError as e:
            ui.warn("timing", f"could not write timing json: {e!r}")
        if status is not None:
            status.finish(rc)
            status.report = {"status": "off" if args.no_report else "pending"}
            status.write()
        if not args.no_report:
            report = emit_report(
                run_dir,
                origami_reference_on(args, cfg),
                float(cfg_get(cfg, "report.timeout_s")),
            )
            if status is not None:
                status.report = report
                status.write()
        sys.stdout.flush()
        sys.stdout = _ORIG_STDOUT
        sys.stderr = orig_err
        _ORIG_STDOUT = None
        log_fh.close()


def _main_logged(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    raw: str,
    run_dir: Path,
    timing: Timing,
    status: Optional[RunStatus] = None,
) -> int:
    ui.banner(f"tilewright pipeline  config={args.config.name}  mode={args.mode}")
    errs = stop_survivors(run_dir)
    if errs:
        _print_problems("resume", errs, [])
        return 2
    cfg, audit_devs, notes, errs = exclude_gpus(
        cfg, args.devices if args.mode == "audit" else None
    )
    for note in notes:
        ui.warn("gpus", note)
    if errs:
        _print_problems("gpus", errs, [])
        return 2
    gone = gpu_health.release_excluded(
        gpu_health.Guard(
            run_dir / gpu_health.GUARD_DIR, "", by=f"orchestrator pid {os.getpid()}"
        ),
        cfg_get(cfg, "runtime.exclude_gpus") or [],
    )
    if gone:
        ui.warn(
            "health",
            f"the run's hold of {', '.join(gone)} is dropped: runtime.exclude_gpus "
            f"leaves them out",
        )
    if args.mode == "audit":
        args.devices = audit_devs
    device = int((args.devices or cfg["runtime"]["devices"])[0])
    held = cfg_get(cfg, "contexts.holdout")
    try:
        cfg, resolved = resolve_contexts(cfg, device)
    except ConfigError as e:
        _print_problems("contexts", e.errors, [])
        return 2
    for line in resolved:
        (ui.warn if line.endswith("left out") else ui.info)("contexts", line)
    if held and not cfg_get(cfg, "contexts.holdout"):
        ui.warn(
            "contexts",
            "contexts.holdout: every entry is left out on this device, so this "
            "run measures the models on no held-out context",
        )
    cfg, read = resolve_num_xcd(cfg, device)
    for line in read:
        ui.info("hardware", line)
    resolved += read
    try:
        cfg, read = resolve_roofline(cfg, device)
    except ConfigError as e:
        _print_problems("roofline", e.errors, [])
        return 2
    for line in read:
        (ui.warn if "unknown" in line else ui.info)("roofline", line)
    resolved += read
    train_cfg = cfg
    if args.mode == "audit":
        cfg = auditlib.bench_config(cfg, args.devices)
    cfg = resolve_bench_timing(cfg)
    invocation = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{args.mode}"
    canary = canary_needed(cfg, derive_paths(cfg))
    cfg = resolve_correctness(cfg, run_dir, invocation, canary)
    pin = pinning.make_pin(THIS_DIR, pinned_config(cfg, audit=args.mode == "audit"))
    parts = pin_parts(args.mode)
    errs = settle_run_pin(run_dir, pin, args.allow_code_change, parts)
    if errs:
        _print_problems("pin", errs, [])
        return 2
    snapshot = write_snapshot(run_dir, cfg, raw, args.mode, pin, resolved)
    paths = derive_paths(cfg)
    ui.info("cfg", f"config_id    : {cfg['config_id']}")
    ui.info("cfg", f"run_dir      : {run_dir}")
    ui.info("cfg", f"snapshot     : {snapshot}")
    ui.info("cfg", f"devices      : {cfg['runtime']['devices']}")
    ui.info("cfg", f"bench        : {paths.bench_binary}")
    ui.info("cfg", f"library      : {paths.library_dir}")
    ui.info("cfg", f"library stem : {cfg['hipblaslt']['library_stem']}")
    ui.info("cfg", f"bench timing : {bench_timing_text(cfg)}")
    ui.info("cfg", f"bench guard  : {guard_text(cfg)}")
    ui.info("cfg", f"produced with: {pinning.describe(pin)}")
    if benv.runtime_env_text(cfg):
        ui.info("cfg", f"runtime env  : {benv.runtime_env_text(cfg)}")
    for name, value in benv.runtime_env(cfg).items():
        if os.environ.get(name, value) != value:
            ui.warn(
                "cfg",
                f"runtime.env sets {name}={value} for every stage over this "
                f"environment's {name}={os.environ[name]}",
            )
    if not args.skip_preflight:
        record = snapshot.with_name(snapshot.stem + "." + fingerprint.PLATFORM_FILE)
        errs = preflight(cfg, paths, args.mode, run_dir, record)
        if errs:
            _print_problems("preflight", errs, [])
            return 2
        ui.ok("preflight", "bench, library and tilewright module found")
        ui.info("preflight", f"tilewright module: {_engine_module_path()}")
        print_pool(cfg, paths)
    try:
        layout = (
            run_layout.of(run_dir)
            if args.mode == "validate"
            else run_layout.settle(run_dir)
        )
    except (OSError, ValueError) as e:
        _print_problems("layout", [str(e)], [])
        return 2
    if layout.legacy:
        ui.info(
            "layout",
            f"{run_dir} was written before the planner became stage02, and "
            f"keeps it in {layout.planner}/",
        )

    pipeline = Pipeline(
        cfg=cfg,
        snapshot=snapshot,
        run_dir=run_dir,
        paths=paths,
        stages_dir=args.stages_dir.resolve(),
        timing=timing,
        force_deploy=args.deploy,
        origami_reference=origami_reference_on(args, cfg),
        pin=pin,
        pin_parts=parts,
        invocation=invocation,
    )
    pipeline.canary = canary
    if status is not None:
        pipeline.validation = status.validation
        status.invocation["tilewright_module"] = _engine_module_path()
        status.write()
    rc = pipeline.hazard_canary() or pipeline.correctness_gate()
    if rc:
        return rc

    if args.mode == "audit":
        return AuditRun(pipeline, train_cfg, args.source_run_dir).run()
    finish_promotion(pipeline)

    if args.mode == "validate":
        rounds = complete_prefix(run_dir)
        if not rounds:
            ui.err("validate", f"no complete round in {run_dir}")
            return 2
        bundles = run_bundles(run_dir)
        if args.from_round is not None:
            pick = [r for r in rounds if round_index(r) == args.from_round]
            if not pick:
                ui.err("validate", f"round_{args.from_round} is not a complete round")
                return 2
            models_round = pick[0]
        else:
            models_round = shipped_round(bundles) or rounds[-1]
        if not (pipeline.stage07_enabled() or pipeline.stage08_enabled()):
            ui.err(
                "validate",
                "nothing to validate: set stage07.parity_sample_trained, "
                "stage07.bench_yamls or stage08.datasets",
            )
            return 2
        considered = [
            r for r in bundles if rb.bundle_key(r) <= rb.bundle_key(models_round)
        ]
        return pipeline.run_validate(
            models_round=models_round,
            rounds=considered,
            from_stage=args.from_stage or 7.0,
        )

    try:
        if args.mode == "enhance":
            steps = plan_enhance_from_args(args, cfg, run_dir)
            _print_enhance_selection(steps)
        else:
            steps = plan_training(
                args.mode,
                run_dir,
                int(pipeline.get("full.total_rounds")),
                args.from_round,
                args.from_stage,
            )
    except ConfigError as e:
        _print_problems("plan", e.errors, [])
        return 2
    if args.mode == "full" and not args.skip_preflight:
        errs = deploy_errors(cfg, run_dir, steps, args.deploy)
        if errs:
            _print_problems("preflight", errs, [])
            pipeline.record_validation("stage07", "not_run", reason=errs[0])
            return 2
    if not steps:
        ui.ok("plan", "every planned round is already complete")
    build = AuditBuild(pipeline, args)
    end_rc, final_deployed, deployed_any = 0, False, False
    stopped_early = False
    try:
        rc = build.prepare()
        if rc:
            return rc
        for step in steps:
            errs, warns = disk_problems(run_dir, 1, pipeline.get("runtime.min_free_gb"))
            for w in warns:
                ui.warn("disk", w)
            if errs:
                _print_problems(
                    "disk", [f"round_{step.index} not started: {errs[0]}"], []
                )
                if status is not None:
                    status.failed_round = {
                        "round": step.index,
                        "rc": 2,
                        "reason": errs[0],
                    }
                return 2
            result = pipeline.run_round(step)
            if status is not None:
                if result.rc:
                    status.failed_round = {"round": step.index, "rc": result.rc}
                status.write()
            if result.rc:
                return result.rc
            deployed_any = deployed_any or result.deployed
            if (
                result.target_reached
                and pipeline.get("deploy.stop_when_target_reached")
                and not step.is_final
            ):
                ui.ok("round", "deploy target reached; stopping early")
                stopped_early = True
                break
        total = int(pipeline.get("full.total_rounds"))
        n_done = len(complete_prefix(run_dir))
        if args.mode == "full" and n_done >= total:
            end_rc, final_deployed = pipeline.finish_run(build)
        else:
            end_rc = build.join()
            if args.mode == "full" and stopped_early:
                for line in pipeline.end_of_run_waits(n_done, total):
                    ui.warn("done", line)
    finally:
        build.stop()

    shipped = shipped_bundle_numbers(run_dir)
    (run_dir / "held_out_sel_eff_breakdown.json").write_text(
        json.dumps(_json_safe(shipped), indent=2) + "\n"
    )
    print_shipped(shipped, pipeline.origami_reference)
    for line in untrained_cell_lines(run_dir):
        ui.warn("carry", line)

    rc = 0
    full = args.mode == "full"
    rounds = complete_prefix(run_dir)
    bundles = validation_bundles(run_dir)
    if (
        (steps or full)
        and rounds
        and (pipeline.stage07_enabled() or pipeline.stage08_enabled())
    ):
        if (
            steps
            and pipeline.stage07_enabled()
            and not (deployed_any or final_deployed)
            and latest_deploy_record(bundles) is not None
        ):
            ui.warn(
                "validate",
                "this invocation deployed nothing; stage 7 validates the "
                "latest model this run deployed",
            )
        rc = pipeline.run_validate(
            models_round=shipped_round(bundles) or rounds[-1],
            rounds=bundles,
            from_stage=7.0,
            require_deploy=full,
            resume=True,
        )
    for key, v in pipeline.validation.items():
        more = f" (rc={v['rc']})" if v.get("rc") else ""
        more += f": {v['reason']}" if v.get("reason") else ""
        (ui.ok if v["status"] == "passed" else ui.err)(
            "done", f"{key} {v['status'].replace('_', ' ')}{more}"
        )
    problems = []
    if end_rc:
        problems.append("the end of the run is incomplete; the same command resumes it")
    if rc:
        problems.append("validation has not passed")
    done = "done" + (f" ({'; '.join(problems)})" if problems else "")
    ui.banner(done, ui.C.RED if problems else ui.C.GREEN)
    ui.ok("done", f"run_dir: {run_dir} ({RUN_STATUS})")
    return end_rc or rc


if __name__ == "__main__":
    sys.exit(main())
