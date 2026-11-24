#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Per-problem fitting of MX-GEMM configs whose pipeline has shape traits (mx_gemm_utils).

A pipeline extension with shape_traits fixes part of its tuning per problem shape at compile
time (comp_tdm_scheduled: the edge and wide traits and tail_steps), so a sweep fits each such
config to each problem:

  has_shape_traits(cfg)                 : whether cfg's pipeline has them
  fit_to_problem(cfg, M, N, K)          : the config that runs one problem
  specialize_configs(configs, problems) : the unique configs and the ones each problem runs
"""

import json
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from mx_gemm_utils import MxGemmKernelConfig, mx_pipeline_extension


def has_shape_traits(cfg: MxGemmKernelConfig) -> bool:
    extension = mx_pipeline_extension(cfg.pipeline)
    return extension is not None and extension.shape_traits


def fit_to_problem(
    cfg: MxGemmKernelConfig, M: int, N: int, K: int
) -> Optional[MxGemmKernelConfig]:
    """The config that runs an M x N x K problem: None if the instance cannot run it, else a
    copy with the problem's shape traits (the extension's fit_to_problem). Configs whose
    pipeline has no shape traits raise ValueError."""
    if not has_shape_traits(cfg):
        raise ValueError(
            f"fit_to_problem: pipeline {cfg.pipeline!r} has no shape traits"
        )
    tuning = mx_pipeline_extension(cfg.pipeline).fit_to_problem(
        cfg._tile_config(), cfg.tuning, cfg.datatype, M, N, K
    )
    if tuning is None:
        return None
    if tuning is cfg.tuning:
        return cfg
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


def specialize_configs(
    configs: List[MxGemmKernelConfig], problems
) -> Tuple[List[MxGemmKernelConfig], List[List[int]], List[int]]:
    """Fit each config to each problem (fit_to_problem); a config that cannot run a problem is
    dropped for it, and problems that need the same traits share one config. Configs whose
    pipeline has no shape traits raise ValueError.

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
            fitted = fit_to_problem(cfg, M, N, K)
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
