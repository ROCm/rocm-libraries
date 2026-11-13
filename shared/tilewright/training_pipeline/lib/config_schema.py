# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The schema of a run config: SCHEMA, the Spec type and the value
constructors it is written with, and the constants only it uses.
run_orchestrator.py checks a config against it (validate_config), fills in
its defaults and reads the pinned keys from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from . import audit as auditlib
from . import audit_v2 as v2lib
from . import bank_pool
from . import cell_merge as cm
from . import contexts as ctxlib
from . import correctness_gate as cgate
from . import gpu_health
from . import hazards as hazlib
from . import host_load
from . import library_pool as lp
from . import provenance as provlib
from . import rebench as rebenchlib
from . import repeats
from . import sustained as suslib
from . import weak_spots
from .bench_yaml import (
    ADAPTIVE_DEFAULTS,
    ADAPTIVE_LONG_DEFAULTS,
    CANARY_DEFAULTS,
    DEFAULT_ORDER_SEED,
    HEAVY_OPERAND_GIB,
    KERNEL_ORDERS,
    LONG_KNOB_KEYS,
    WARMUP_LINE_DEFAULTS,
)
from .planner import MAX_BENCH_LIST, RECALL_MIN_CEILING, PlanParams


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
# stage05 --whitelist-tier-contexts (lib/whitelist_tier.py CONTEXT_MODES).
WHITELIST_TIER_CONTEXTS = ("pooled", "balanced", "reserve")
# stage05's default --whitelist-tier-guard-keys (lib/whitelist_tier.py
# GUARD_KEYS): the keys the tail guard may add to a cell's tier.
WHITELIST_TIER_GUARD_KEYS = 8
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
        "enabled": _bool(default=True),
        "uniform_summation_order": _bool(default=True),
        "workspace_bytes": _int(
            default=ctxlib.FULL_WORKSPACE_BYTES,
            minimum=0,
            maximum=ctxlib.MAX_WORKSPACE_BYTES,
        ),
        "launch_keyed": _bool(default=False),
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
# stage01's --strong-cell-margin default (STRONG_CELL_MARGIN there; a test
# holds them equal).
STRONG_CELL_MARGIN = 0.03

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
                # The library's logic file name without its extension: the
                # shipped model is named after it.
                "logic_name": _str(pattern=r"^[A-Za-z0-9_+-]+$"),
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
                "targeted_per_tail_gemm": _int(default=4, minimum=0),
                "targeted_max_per_leaf": _int(default=32, minimum=0),
                "targeted_seed": _section(
                    {
                        "list": _str(nullable=True),
                        "per_target": _int(default=4, minimum=1),
                        "library": _str(nullable=True),
                    }
                ),
                "weak_cell_factor": _num(default=1.0, minimum=1.0),
                "strong_cell_factor": _num(
                    default=1.0, minimum=0.0, exclusive_minimum=True, maximum=1.0
                ),
                "strong_cell_margin": _num(default=STRONG_CELL_MARGIN, minimum=0.0),
                "large_k_factor": _num(default=1.0, minimum=1.0),
                "exclude": _list(
                    _section({key: _int(minimum=0) for key in EXCLUDE_KEYS})
                ),
            }
        ),
        "bench": _section(
            {
                "blocks_per_gpu": _int(default=20, minimum=1),
                "min_block_lines": _int(default=1, minimum=1),
                "heavy_operand_gib": _num(default=HEAVY_OPERAND_GIB, minimum=0),
                "stall_timeout_s": _num(default=600.0, minimum=0),
                "startup_grace_s": _num(default=900.0, minimum=0),
                # The fault policy of stage03 (lib/node_faults.py) and its
                # node store (None: node_faults.store_dir).
                "faults": _section(
                    {
                        "enabled": _bool(default=True),
                        "store_dir": _str(nullable=True),
                    }
                ),
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
                                "tolerance_us": _num(
                                    default=_HEALTH_CANARY["tolerance_us"],
                                    minimum=0,
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
                        "tolerance_us": _num(
                            default=_CANARY["tolerance_us"], minimum=0
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
                "granule": _int(default=ctxlib.CU_BUDGET_GRANULE, minimum=1),
                "workspaces": _list(
                    _int(minimum=0, maximum=ctxlib.MAX_WORKSPACE_BYTES),
                    default=list(ctxlib.DEFAULT_WORKSPACES),
                ),
                "full_workspace_bytes": _int(
                    default=ctxlib.FULL_WORKSPACE_BYTES,
                    minimum=1,
                    maximum=ctxlib.MAX_WORKSPACE_BYTES,
                ),
                ctxlib.ROUND_ORDER_KEY: _bool(nullable=True),
                "canaries": _section(
                    {
                        "enabled": _bool(default=ctxlib.CANARY_DEFAULTS["enabled"]),
                        "fast_check_samples": _int(
                            default=ctxlib.CANARY_DEFAULTS["fast_check_samples"],
                            minimum=0,
                        ),
                        "verify_lines": _int(
                            default=ctxlib.CANARY_DEFAULTS["verify_lines"], minimum=0
                        ),
                        "verify_iters": _int(
                            default=ctxlib.CANARY_DEFAULTS["verify_iters"], minimum=1
                        ),
                    }
                ),
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
                "whitelist_tier_contexts": _str(
                    default="pooled", choices=WHITELIST_TIER_CONTEXTS
                ),
                "whitelist_tier_guard_keys": _int(
                    default=WHITELIST_TIER_GUARD_KEYS, minimum=0
                ),
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
                "distill_lam": _num(minimum=0, maximum=1),
                "distill_T": _num(minimum=0, exclusive_minimum=True),
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
                "tail_floor": Spec("threshold", default=0.7),
                "collapse_retrain": _bool(default=True),
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
                        "repair": _bool(default=True),
                    }
                ),
            }
        ),
        "audit": _section(
            {
                "dir": _str(nullable=True),
                "held_out": _list(_str(), default=[]),
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
                "canary_tolerance_us": _num(default=0.0, minimum=0),
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
                "provenance": _str(
                    default=provlib.DEPLOY_PROVENANCE_MODES[0],
                    choices=provlib.DEPLOY_PROVENANCE_MODES,
                ),
                # The host-only launch parity gate of a model of launch rule
                # version 3 (lib/launch_parity.py).
                "launch_parity": _section(
                    {
                        "tests_binary": _str(nullable=True),
                        "hip_lib": _str(nullable=True),
                        "env": _list(_str(), default=[]),
                        "samples": _int(default=8, minimum=1),
                        "timeout_s": _num(default=3600.0, minimum=1),
                        "required": _bool(default=False),
                    }
                ),
                "alignment_veto": _section(
                    {
                        "k_min": _int(required=True, minimum=1, maximum=2**32 - 1),
                        "align_bytes": _int(required=True, minimum=1, maximum=2**28),
                    },
                    nullable=True,
                ),
                "drift_fallback": _section(
                    {
                        "keep": _num(
                            required=True,
                            minimum=0,
                            maximum=1,
                            exclusive_minimum=True,
                        ),
                    },
                    nullable=True,
                ),
                "standins": _section(
                    {"depth": _int(required=True, minimum=1, maximum=3)},
                    nullable=True,
                ),
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
                "repair": _bool(default=False),
                "repair_max_cells": _int(default=cm.DEFAULT_MAX_CELLS, minimum=1),
                "repair_max_attempts": _int(default=cm.DEFAULT_MAX_ATTEMPTS, minimum=1),
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
                "static_check": _bool(default=True),
                "custom_kernels": _str(nullable=True),
                "guard_unguarded_custom": _bool(default=False),
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
                "screen": _section(
                    {
                        "enabled": _bool(default=hazlib.SCREEN_DEFAULTS["enabled"]),
                        "shapes": _list(_list(_int(minimum=1), min_len=3), default=[]),
                        "per_cell": _int(
                            default=hazlib.SCREEN_DEFAULTS["per_cell"], minimum=0
                        ),
                        "seed": _int(default=hazlib.SCREEN_DEFAULTS["seed"]),
                        "stall_s": _num(
                            default=hazlib.SCREEN_DEFAULTS["stall_s"],
                            minimum=0,
                            exclusive_minimum=True,
                        ),
                    }
                ),
            }
        ),
    }
)
