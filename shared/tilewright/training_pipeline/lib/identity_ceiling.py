# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Selection-efficiency ceiling of a kernel identity.

The engine scores groups of kernels it cannot tell apart, and for a group it
ranks first it runs the group's first member in pool order that the problem
admits. With perfect scoring of groups, a GEMM therefore costs the fastest
such first member over all groups; the ratio of the GEMM's fastest kernel to
that cost is the ceiling a model trained on that identity can reach.

A kernel counts as admitted by a GEMM when the GEMM measured it (a timed row,
else a warm-up row, as `lib.evaluate.measurements` reads them). On an
exhaustive bench that is every kernel the library's predicates accept; on a
planned bench unmeasured kernels count as not admitted, which makes the
ceiling optimistic.

Groupings are functions of a pool kernel (`lib.evaluate` pool entry:
`config_kwargs_from_row` fields, `sol_idx_global`, optional `attributes`):
`engine_grouping` is the engine's grouping for a v3 model ranked with a
launch model (pass the launch model only for a model that serves launch keys,
`ModelInfo.launch_keys`; the engine ignores the keys of any other),
`signature_grouping` the 8-field signature alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Dict, Hashable, Iterable, List, Mapping
from typing import Optional, Sequence, Tuple

from . import evaluate as ev
from . import features as fs

Grouping = Callable[[Mapping[str, Any]], Hashable]

# A GEMM whose ceiling is below 1 - LOSS_TOLERANCE counts as losing.
LOSS_TOLERANCE = 1e-3


def signature_grouping(kernel: Mapping[str, Any]) -> Hashable:
    return tuple(int(kernel[f]) for f in fs.SIGNATURE_FIELDS)


def engine_grouping(
    spec: fs.AttributeSpec = (), launch: Optional[Any] = None
) -> Grouping:
    """The engine's grouping (`features.v3_candidate_key`) for a v3 model with
    attribute spec `spec` ranked with launch model `launch` (an object with
    `launch_key(config_kwargs, attributes)`, such as
    `hipblaslt_launch.HipblasltLaunch`; None: every kernel of a view key
    launches alike)."""

    def key(kernel: Mapping[str, Any]) -> Hashable:
        return fs.v3_candidate_key(kernel, kernel.get("attributes"), spec, launch)

    return key


@dataclass
class Ceiling:
    """Perfect-scoring efficiency of one grouping. `per_gemm` maps
    `lib.evaluate.gemm_key` to winner / pick latency; GEMMs without a
    measured kernel are left out."""

    label: str
    n_kernels: int
    n_groups: int
    n_shared: int  # groups with more than one kernel
    max_group: int
    geomean: float
    n_losing: int
    per_gemm: Dict[Tuple[Any, ...], float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n_kernels": self.n_kernels,
            "n_groups": self.n_groups,
            "n_shared": self.n_shared,
            "max_group": self.max_group,
            "geomean": self.geomean,
            "n_gemms": len(self.per_gemm),
            "n_losing": self.n_losing,
        }


def _latencies(gemm: Mapping[str, Any]) -> Dict[int, float]:
    timed, screened = ev.measurements(gemm)
    out = dict(screened)
    out.update(timed)
    return out


def ceiling(
    gemms: Iterable[Mapping[str, Any]],
    grouping: Grouping,
    label: str = "",
    pools: Optional[Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]]] = None,
    kernels: Optional[Collection[int]] = None,
) -> Ceiling:
    """The ceiling of `grouping` over `gemms` (GEMM dicts with `candidates`,
    as stage05 loads them). `pools` maps `lib.evaluate.pool_key` to the pool
    in runtime order; by default each pool holds the measured kernels,
    ascending by solution index (`lib.evaluate.build_pools`). `kernels`
    (solution indices) ships only those kernels: groups and picks use them
    alone, while each GEMM's best stays its fastest measured kernel."""
    gemms = list(gemms)
    if pools is None:
        pools = ev.build_pools(gemms)
    if kernels is not None:
        shipped = set(int(sid) for sid in kernels)
        pools = {
            pk: [k for k in pool if int(k["sol_idx_global"]) in shipped]
            for pk, pool in pools.items()
        }
    # Per pool: each group's members' solution indices in pool order.
    members: Dict[Tuple[str, ...], List[List[int]]] = {}
    n_kernels = n_groups = n_shared = max_group = 0
    for pk, pool in pools.items():
        by_key: Dict[Hashable, List[int]] = {}
        for kernel in pool:
            by_key.setdefault(grouping(kernel), []).append(
                int(kernel["sol_idx_global"])
            )
        groups = list(by_key.values())
        members[pk] = groups
        n_kernels += len(pool)
        n_groups += len(groups)
        n_shared += sum(1 for g in groups if len(g) > 1)
        max_group = max([max_group] + [len(g) for g in groups])

    per_gemm: Dict[Tuple[Any, ...], float] = {}
    for g in gemms:
        lat = _latencies(g)
        if not lat:
            continue
        best = min(lat.values())
        pick = math.inf
        for group in members.get(ev.pool_key(g), ()):
            for sid in group:
                if sid in lat:
                    pick = min(pick, lat[sid])
                    break
        per_gemm[ev.gemm_key(g)] = best / pick if math.isfinite(pick) else 0.0
    values = list(per_gemm.values())
    geomean = (
        math.exp(sum(math.log(v) for v in values) / len(values))
        if values and all(v > 0 for v in values)
        else 0.0
    )
    return Ceiling(
        label=label,
        n_kernels=n_kernels,
        n_groups=n_groups,
        n_shared=n_shared,
        max_group=max_group,
        geomean=geomean,
        n_losing=sum(1 for v in values if v < 1.0 - LOSS_TOLERANCE),
        per_gemm=per_gemm,
    )


def engine_and_signature(
    gemms: Iterable[Mapping[str, Any]],
    spec: fs.AttributeSpec = (),
    launch: Optional[Any] = None,
    pools: Optional[Mapping[Tuple[str, ...], Sequence[Mapping[str, Any]]]] = None,
    kernels: Optional[Collection[int]] = None,
) -> Dict[str, Ceiling]:
    """{"engine": the engine's grouping, "signature": signature only}, for
    the shipped `kernels` as in `ceiling`."""
    gemms = list(gemms)
    if pools is None:
        pools = ev.build_pools(gemms)
    return {
        "engine": ceiling(
            gemms, engine_grouping(spec, launch), "engine", pools, kernels
        ),
        "signature": ceiling(gemms, signature_grouping, "signature", pools, kernels),
    }
