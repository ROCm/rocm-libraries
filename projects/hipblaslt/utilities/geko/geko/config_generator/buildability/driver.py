# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verdicts for every candidate MI of one GEMM type, in parallel: one fresh worker process per MI."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from geko.config_generator.buildability.codegen import CodegenChecker, TuningSpace, write_probe_config
from geko.config_generator.buildability.constants import LIST_SEARCH_SPACE
from geko.config_generator.buildability.search import SettingSearch
from geko.config_generator.buildability.verdict import Verdict, buildable_depthus, classify
from geko.config_generator.mi_candidates import MI, enumerate_candidate_mis


_WORKER: Tuple[Optional[CodegenChecker], Optional[SettingSearch]] = (None, None)


def _classify_in_worker(mi: MI) -> Verdict:
    checker, search = _WORKER
    v = classify(checker, search, mi)
    if v.buildable:
        v.depthu = buildable_depthus(checker, search, mi, v.witness)
    return v


class SubtileOnly(Exception):
    """The GEMM type is tuned only as subtile kernels, which MI Designer never filters with a list."""


def classify_candidates(
    config: Dict[str, Any],
    hipblaslt_path: Path,
    workers: int,
    progress: Callable[[int, int, int], None] = lambda done, total, buildable: None,
) -> Tuple[List[Verdict], TuningSpace, str]:
    """Verdict for every candidate MI of one GEMM type, in enumeration order; every buildable MI's
    verdict also lists the DepthUs it builds at (buildable_depthus).

    Returns (verdicts, tuning space, GEMM name).
    """
    global _WORKER
    from geko.config_generator.config_generator import mi_design_mx_options

    with tempfile.TemporaryDirectory(prefix="geko_buildable_") as tmp:
        config_path, cfg = write_probe_config(config, hipblaslt_path, Path(tmp))
        gemm = cfg["GemmProblems"][0]
        if mi_design_mx_options({**cfg, "GemmProblem": gemm})[1]:
            raise SubtileOnly(f"{gemm.name} on {cfg['ARCH']} is tuned as subtile kernels (gfx950 MX), "
                              "which do not read buildable-MI lists")
        checker = CodegenChecker(config_path, cfg["ARCH"])
    search = SettingSearch(checker.space, cfg)
    mis = enumerate_candidate_mis(cfg["ARCH"], gemm.gemm_type.data_type, LIST_SEARCH_SPACE)

    _WORKER = (checker, search)  # inherited by forked workers
    verdicts: Dict[MI, Verdict] = {}
    # A fresh worker per MI: code generation for a large kernel leaves several GB in the process, and long-lived
    # workers accumulate it (gfx1250 FP4 at 352 workers exhausted 3 TB). Verdicts do not depend on the process.
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for v in pool.imap_unordered(_classify_in_worker, mis, chunksize=1):
            verdicts[v.mi] = v
            progress(len(verdicts), len(mis), sum(x.buildable for x in verdicts.values()))
    _WORKER = (None, None)
    return [verdicts[mi] for mi in mis], checker.space, gemm.name
