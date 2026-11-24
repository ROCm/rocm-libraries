#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
comp_tdm_scheduled helpers for the MX-GEMM dispatcher (mx_gemm_utils).

A comp_tdm_scheduled kernel fixes its shape traits (MxTdmScheduledGemmTraits: the edge and
wide traits and tail_steps) at compile time, so a sweep fits each config to each problem:

  tdm_scheduled_for_problem(cfg, M, N, K)            : the config that runs one problem
  specialize_tdm_scheduled_configs(configs, problems): the unique configs and the ones each
                                                        problem runs
"""

import json
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from mx_gemm_utils import MxGemmKernelConfig, _mx_codegen

PIPELINE = "comp_tdm_scheduled"


def tdm_scheduled_for_problem(
    cfg: MxGemmKernelConfig, M: int, N: int, K: int
) -> Optional[MxGemmKernelConfig]:
    """The comp_tdm_scheduled config that runs an M x N x K problem: None if tile_k > K,
    else a copy whose shape traits are the problem's (mx_tdm_scheduled.shape_tuning).
    A tile the spec rejects, which the build reports, returns cfg."""
    if cfg.pipeline != PIPELINE:
        raise ValueError(
            f"tdm_scheduled_for_problem: pipeline {cfg.pipeline!r} is not {PIPELINE}"
        )
    sched = _mx_codegen().mx_tdm_scheduled
    try:
        spec = sched.spec_from_tile(
            cfg._tile_config(), cfg.tuning, cfg.datatype, M=M, N=N, K=K
        )
    except ValueError:
        return cfg
    tuning = sched.shape_tuning(spec, cfg.tuning)
    if tuning is None:
        return None
    return replace(cfg, tuning=tuning, _name_cache=None)


def _problem_mnk(problem) -> Tuple[int, int, int]:
    if isinstance(problem, dict):
        return problem["M"], problem["N"], problem["K"]
    return problem.M, problem.N, problem.K


def _config_key(cfg: MxGemmKernelConfig) -> str:
    """A key that two configs share iff they build the same kernel (cfg.name is a function of
    the fields, but costs a codegen call)."""
    fields = {k: v for k, v in vars(cfg).items() if k != "_name_cache"}
    return json.dumps(fields, sort_keys=True, default=list)


def specialize_tdm_scheduled_configs(
    configs: List[MxGemmKernelConfig], problems
) -> Tuple[List[MxGemmKernelConfig], List[List[int]], List[int]]:
    """Fit each comp_tdm_scheduled config to each problem (tdm_scheduled_for_problem): its shape
    traits become the problem's, and a tile_k above K drops it for that problem. Problems that
    need the same traits share one config. Configs of other pipelines raise ValueError.

    Returns (unique_configs, per_problem_indices, dropped): per_problem_indices[i] lists the
    indexes into unique_configs that run problems[i] (dicts or objects with M, N and K), and
    dropped[i] counts the configs dropped for it.
    """
    unique: List[MxGemmKernelConfig] = []
    index_of: Dict[str, int] = {}
    per_problem: List[List[int]] = []
    dropped: List[int] = []
    for problem in problems:
        M, N, K = _problem_mnk(problem)
        indices, n_dropped = [], 0
        for cfg in configs:
            fitted = tdm_scheduled_for_problem(cfg, M, N, K)
            if fitted is None:
                n_dropped += 1
                continue
            key = _config_key(fitted)
            if key not in index_of:
                index_of[key] = len(unique)
                unique.append(fitted)
            if index_of[key] not in indices:
                indices.append(index_of[key])
        per_problem.append(indices)
        dropped.append(n_dropped)
    return unique, per_problem, dropped
