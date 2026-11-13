# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch rule entries of an audit record (`entries`, lib/audit.py
`evaluate`): what launch rule version 3 changes on the audit set, and how
the twin's summation-order refusals (lib/hipblaslt_launch.py
`uso_obstacle`) meet the kernels its benches left out.

  pool          the pool's kernels and Stream-K kernels, those of them
                whose per-tile extras are 0 (`per_tile_extra_iters_zero`)
                or unknown (`..._absent`), and their row-uniform StaggerU
                groups (`row_uniform_stagger`: {value: kernels}, `absent`
                for a kernel without the attribute)
  plans_v2_v3   over every (GEMM, kernel it benched), each GEMM under its
                own execution context: how many pairs were planned
                (`pairs`), whose version 3 plan differs from version 2's
                (`differ`), in the grid (`grid`) and in the Stream-K
                reduction (`reduction`), and `unplanned` (a kernel the rule
                cannot plan)
  uso           over every (GEMM under the uniform summation order, pool
                kernel), planned under the rule version the arch trains
                on: kernels the twin refuses that the bench left out
                (`refused_and_missing`) or timed (`refused_but_benched`, a
                disagreement), kernels it admits that the bench timed
                (`admitted_and_benched`) or left out (`admitted_but_missing`:
                the size predicates and the clauses the attributes do not
                hold leave out kernels too), and the refusals by token
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from . import evaluate as ev
from . import features as fs
from . import hipblaslt_launch as hl
from .dat import PER_TILE_EXTRA_ITERS, ROW_UNIFORM_STAGGER

_STREAM_K = 2


def _attrs(kernel: Mapping[str, Any]) -> Mapping[str, Any]:
    return kernel.get("attributes") or {}


def pool_entries(kernels: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The `pool` entry of a pool's kernels."""
    sk = [
        k
        for k in kernels
        if int(_attrs(k).get("tile_processing_strategy", 0)) == _STREAM_K
    ]
    stagger = Counter(str(_attrs(k).get(ROW_UNIFORM_STAGGER, "absent")) for k in sk)
    return {
        "kernels": len(kernels),
        "stream_k": len(sk),
        "per_tile_extra_iters_zero": sum(
            1 for k in sk if _attrs(k).get(PER_TILE_EXTRA_ITERS) == 0
        ),
        "per_tile_extra_iters_absent": sum(
            1 for k in sk if PER_TILE_EXTRA_ITERS not in _attrs(k)
        ),
        "row_uniform_stagger": dict(sorted(stagger.items())),
    }


def entries(
    kernels: Sequence[Mapping[str, Any]],
    gemms: Iterable[Mapping[str, Any]],
    arch: str,
    n_cu: int,
    num_xcd: Optional[int] = None,
) -> Dict[str, Any]:
    """The launch rule entries (see the module docstring) of the pool
    `kernels` and the audit `gemms` (stage05 GEMM dicts) on `arch`, on a
    device of `n_cu` CUs and `num_xcd` XCDs."""
    by_sid = {int(k["sol_idx_global"]): k for k in kernels}
    v2 = hl.HipblasltLaunch(arch, num_xcd, version=2)
    v3 = hl.HipblasltLaunch(arch, num_xcd, version=3)
    trained = hl.HipblasltLaunch(arch, num_xcd)
    plans: Counter = Counter()
    uso: Counter = Counter()
    tokens: Counter = Counter()
    for g in gemms:
        problem = fs.problem_kwargs_from_row(g)
        ctx = ev.context_of(g)
        benched = {int(c["sol_idx_global"]) for c in g.get("candidates") or ()}
        for sid in sorted(benched):
            k = by_sid.get(sid)
            if k is None:
                continue
            try:
                a = v2(problem, k, _attrs(k), ctx, n_cu)
                b = v3(problem, k, _attrs(k), ctx, n_cu)
            except ValueError:
                plans["unplanned"] += 1
                continue
            plans["pairs"] += 1
            plans["differ"] += int(a != b)
            plans["grid"] += int(a.grid_wgs != b.grid_wgs)
            plans["reduction"] += int(a.partial_reduction != b.partial_reduction)
        if not ctx.uniform_summation_order:
            continue
        for sid, k in by_sid.items():
            try:
                plan = trained(problem, k, _attrs(k), ctx, n_cu)
                token = hl.uso_obstacle(problem, k, _attrs(k), ctx, plan)
            except ValueError:
                uso["unplanned"] += 1
                continue
            present = sid in benched
            if token is not None:
                tokens[token] += 1
                uso["refused_but_benched" if present else "refused_and_missing"] += 1
            else:
                uso["admitted_and_benched" if present else "admitted_but_missing"] += 1
    return {
        "pool": pool_entries(kernels),
        "plans_v2_v3": {
            k: int(plans[k]) for k in ("pairs", "differ", "grid", "reduction", "unplanned")
        },
        "uso": {
            **{
                k: int(uso[k])
                for k in (
                    "refused_and_missing",
                    "refused_but_benched",
                    "admitted_and_benched",
                    "admitted_but_missing",
                    "unplanned",
                )
            },
            "refusals": dict(sorted(tokens.items())),
        },
    }
