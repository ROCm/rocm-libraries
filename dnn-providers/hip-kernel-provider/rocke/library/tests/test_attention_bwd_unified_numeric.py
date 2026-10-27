# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the unified (paged) attention backward.

Every selected case of :mod:`tests.sdpa.bwd_unified_cases` runs in each of
its page modes through the unified host plan and run function, then:

* dQ matches the float64 oracle on the gathered contiguous THD problem, and
  dK / dV match it slot by slot in the gradient caches (the expected cache
  scatter-adds the gathered gradient through the block table, so a shared
  page sums every referencing sequence);
* every slot of every referenced page is written, slots of the last live page
  past ``len_kv`` are exact zeros, pages no live block references (spare
  pages and the trap page named by every block-table entry past the live
  blocks) keep the sentinel, tail canaries and stride gaps keep the sentinel,
  the NaN-poisoned workspace canary stays intact. Inputs hold NaN in every
  slot that must never be read (unreferenced pages, past-length slots, stride
  gaps), so a stray read poisons the result.

Further tests:

* ``test_direct_is_deterministic``: two direct-mode runs give bit-identical
  dK / dV caches;
* ``test_modes_agree``: on an injective table both page modes pass and agree
  with each other within the oracle bound (exact zeros past ``len_kv`` in
  both);
* ``test_negative_controls``: a plan without page_zero, without the dV page
  convert, or with the atomic-mode grid in direct mode (no zero-fill tiles
  when ``page > kN0``) fails the checks above (they are live).

Selection: ``ROCKE_BWD_UNIFIED_TIER`` = ``smoke`` (default) | ``full`` |
``exhaustive``; ``ROCKE_BWD_SHARD`` = ``i/n`` (stable hash of the case id).
Skipped unless the visible device is gfx942, gfx950, gfx1151 or gfx1201.
"""

from __future__ import annotations

import dataclasses
import os
import zlib
from types import MappingProxyType

import numpy as np
import pytest

from ._attention_bwd_harness import DEVICE_ARCHS
from ._attention_bwd_unified_harness import (
    ARCH,
    drop_launch,
    plan_unified_case,
    run_unified_case,
)
from .sdpa.bwd_cases import TOLERANCE_BWD
from .sdpa.bwd_unified_cases import get_unified_case, select_unified

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

TIER = os.environ.get("ROCKE_BWD_UNIFIED_TIER", "smoke")


def _in_shard(case_id: str) -> bool:
    raw = os.environ.get("ROCKE_BWD_SHARD", "")
    if not raw:
        return True
    i, n = (int(x) for x in raw.split("/"))
    return zlib.crc32(case_id.encode()) % n == i


def _params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(case, mode, id=f"{case.id}-{mode}")
        for case in select_unified(TIER)
        if _in_shard(case.id)
        for mode in case.modes
    ]


def _passes(run):
    assert not run.integrity, run.integrity
    assert not run.problems, run.problems


def _plannable(case, mode):
    """Skip a case whose forced knobs the arch cannot take (for example the
    ``"sub"`` tiles or the register-accumulator epilogues on RDNA, which runs
    LDS accumulators): the plan raises before any kernel exists. A case
    without forced knobs must always plan."""
    try:
        plan_unified_case(case, arch=ARCH, mode=mode)
    except ValueError as exc:
        if not case.knobs:
            raise
        pytest.skip(f"knobs {dict(case.knobs)} not legal on {ARCH}: {exc}")


@pytest.mark.parametrize("case,mode", _params())
def test_unified_case(case, mode):
    _plannable(case, mode)
    run = run_unified_case(case, arch=ARCH, mode=mode)
    _passes(run)
    stages = [s.stage for s in run.plan.launches]
    if mode == "direct":
        assert stages == ["prep", "main", "convert"]
    else:
        assert stages == [
            "prep", "page_zero", "main", "convert", "page_convert", "page_convert",
        ]  # fmt: skip


DETERMINISM_CASES = (
    "bwd_unified_page16_br",
    "bwd_unified_canary_page256_len100",
    "bwd_unified_gqa4",
)


@pytest.mark.parametrize("case_id", DETERMINISM_CASES)
def test_direct_is_deterministic(case_id):
    case = get_unified_case(case_id)
    a = run_unified_case(case, arch=ARCH, mode="direct")
    b = run_unified_case(case, arch=ARCH, mode="direct")
    _passes(a)
    _passes(b)
    assert np.array_equal(a.dk_cache, b.dk_cache)
    assert np.array_equal(a.dv_cache, b.dv_cache)


@pytest.mark.parametrize("case_id", DETERMINISM_CASES)
def test_modes_agree(case_id):
    case = get_unified_case(case_id)
    a = run_unified_case(case, arch=ARCH, mode="direct")
    b = run_unified_case(case, arch=ARCH, mode="atomic")
    _passes(a)
    _passes(b)
    # Same written slots (integrity), same exact zeros past len_kv, and
    # values within the oracle bound of each other.
    ref = a.inputs.referenced
    tol = TOLERANCE_BWD[case.dtype]
    for x, y in ((a.dk_cache, b.dk_cache), (a.dv_cache, b.dv_cache)):
        x, y = x[ref], y[ref]
        bound = tol["atol"] + tol["rtol"] * float(np.abs(x).max())
        assert float(np.abs(x - y).max()) <= bound
    past = a.inputs.past_len
    assert np.all(a.dk_cache[past] == 0.0) and np.all(b.dk_cache[past] == 0.0)


def _direct_grid_without_zero_fill(plan):
    """The direct plan's main launch with the atomic-mode grid (kv tiles
    cover the keys only): slots past len_kv beyond the last key tile stay
    unwritten when the page is larger than the tile."""
    steps = []
    for step in plan.launches:
        if step.stage == "main":
            kn = plan.specs["main"].block_n
            n = -(-step.scalars["max_seqlen_kv"] // kn)
            scal = dict(step.scalars, n_kv_tiles=n)
            step = dataclasses.replace(
                step, grid=(n,) + tuple(step.grid[1:]), scalars=MappingProxyType(scal)
            )
        steps.append(step)
    return dataclasses.replace(plan, launches=tuple(steps))


@pytest.mark.parametrize(
    "case_id,mode,mutate",
    [
        ("bwd_unified_page16_br", "atomic", lambda p: drop_launch(p, "page_zero")),
        (
            "bwd_unified_page16_br",
            "atomic",
            lambda p: drop_launch(p, "page_convert", 1),
        ),
        ("bwd_unified_canary_page256_len100", "direct", _direct_grid_without_zero_fill),
    ],
    ids=["no_page_zero", "no_dv_page_convert", "no_zero_fill_tiles"],
)
def test_negative_controls(case_id, mode, mutate):
    case = get_unified_case(case_id)
    run = run_unified_case(case, arch=ARCH, mode=mode, mutate_plan=mutate, cache={})
    assert run.integrity or run.problems, "the mutated plan passed every check"
