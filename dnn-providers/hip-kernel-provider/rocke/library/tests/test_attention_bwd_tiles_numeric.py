# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the small-tile / occupancy region of the backward.

Every set of ``tests/sdpa/bwd_tile_region_sets.py`` (replicated and
transposed G0 / G2 grids, the transposed G1 / G3 grid, the (2, W/2) grids on
tiles below 16 * W keys, occupancy hints 2 to 4) runs ``prep -> main ->
convert`` on the cases of its head size and passes the float64 oracle with
the harness integrity facts (``test_region_set``); the launched spec carries
every knob of the set.

* ``test_bitwise_groups``: sets of one tile that only spread the S / dP (or
  dK / dV) fragments differently over the waves give direct-mode dK / dV bit
  for bit equal to the group's first set (each fragment is the same matrix-op
  chain whichever wave computes it; a replicated grid must neither lose nor
  double a fragment).
* ``test_occupancy_hint_is_met``: each occupancy set compiles (production
  comgr, llvm22) with no scratch and no spill, and its code object holds at
  least ``waves_per_eu`` waves per SIMD in whole workgroups.

Selection: ``ROCKE_BWD_SHARD`` (``i/n``). Skipped unless the device is gfx942
or gfx950.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ._attention_bwd_test_utils import find_bwd_case
from ._attention_bwd_harness import ARCH, plan_case, run_case
from .sdpa.bwd_cases import compare_bwd
from .sdpa.bwd_kernel_cases import shard_of
from .sdpa.bwd_tile_region_sets import (
    BITWISE_GROUPS,
    REGION_CASES,
    REGION_SETS,
    region_knobs,
)

_CDNA = ("gfx942", "gfx950")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in _CDNA, reason=f"needs gfx942/gfx950; device is {ARCH}"
    ),
]


def _shard():
    return tuple(int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _set_params():
    if ARCH not in _CDNA:
        return [pytest.param(None, None, id="no-device")]
    i, n = _shard()
    out = []
    for name, row in REGION_SETS.items():
        for d in row[-1]:
            for cid in REGION_CASES[d]:
                if shard_of(f"{cid}-{name}", n) == i:
                    out.append(pytest.param(name, cid, id=f"{cid}-{name}"))
    return out or [pytest.param(None, None, id="empty-shard")]


_RUNS: dict = {}


def _refusal(name, case_id):
    """The validator's refusal of set ``name`` on ``case_id`` (a recorded
    resource refusal), or ``None`` when it plans."""
    try:
        plan_case(
            find_bwd_case(case_id),
            arch=ARCH,
            config="default",
            overrides=dict(region_knobs(name)),
        )
    except ValueError as exc:
        return str(exc)
    return None


def _run(name, case_id):
    key = (name, case_id)
    if key not in _RUNS:
        case = find_bwd_case(case_id)
        knobs = region_knobs(name)
        run = run_case(case, arch=ARCH, config="default", overrides=dict(knobs))
        spec = run.plan.specs["main"]
        for k, v in knobs.items():
            if v is not None:
                assert getattr(spec, k) == v, (k, getattr(spec, k), v)
        _passes(run)
        _RUNS[key] = (spec, run.dk, run.dv)
    return _RUNS[key]


@pytest.mark.parametrize("set_name,case_id", _set_params())
def test_region_set(set_name, case_id):
    if set_name is None:
        pytest.skip("no region set in this shard")
    refused = _refusal(set_name, case_id)
    if refused:
        pytest.skip(f"{set_name} is not legal for {case_id}: {refused}")
    _run(set_name, case_id)


def _group_params():
    if ARCH not in _CDNA:
        return [pytest.param(None, None, id="no-device")]
    i, n = _shard()
    out = []
    for g, group in enumerate(BITWISE_GROUPS):
        for d in REGION_SETS[group[0]][-1]:
            for cid in REGION_CASES[d]:
                if shard_of(f"{cid}-group{g}", n) == i:
                    out.append(pytest.param(g, cid, id=f"{cid}-group{g}"))
    return out or [pytest.param(None, None, id="empty-shard")]


@pytest.mark.parametrize("group,case_id", _group_params())
def test_bitwise_groups(group, case_id):
    if group is None:
        pytest.skip("no bitwise group in this shard")
    # members the validator refuses on this case (a recorded resource
    # refusal) drop out; the others still compare against each other
    names = [n for n in BITWISE_GROUPS[group] if _refusal(n, case_id) is None]
    if len(names) < 2:
        pytest.skip(
            f"fewer than two members of group {group} plan on {case_id}: "
            + "; ".join(
                f"{n}: {_refusal(n, case_id)}"
                for n in BITWISE_GROUPS[group]
                if n not in names
            )
        )
    spec0, dk0, dv0 = _run(names[0], case_id)
    if spec0.dkv_mode != "direct":
        # atomic dK / dV are summed by fp32 atomics in a run-dependent order;
        # every member still passes the oracle
        for name in names[1:]:
            _run(name, case_id)
        return
    for name in names[1:]:
        _spec, dk, dv = _run(name, case_id)
        assert np.array_equal(dk, dk0), f"dK of {name} differs from {names[0]}"
        assert np.array_equal(dv, dv0), f"dV of {name} differs from {names[0]}"


def _occ_params():
    if ARCH not in _CDNA:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = _shard()
    out = []
    for name, row in REGION_SETS.items():
        if row[8] is None or row[8] < 2:
            continue
        for d in row[-1]:
            for dt in ("fp16", "bf16"):
                if shard_of(f"occ-{name}-{d}-{dt}", n) == i:
                    out.append(pytest.param(name, d, dt, id=f"{name}-d{d}-{dt}"))
    return out or [pytest.param(None, None, None, id="empty-shard")]


@pytest.mark.parametrize("set_name,d,dtype", _occ_params())
def test_occupancy_hint_is_met(set_name, d, dtype):
    if set_name is None:
        pytest.skip("no occupancy set in this shard")
    from rocke.helpers.compile import compile_kernel

    from kernels.common.attention_bwd import (
        AttnBwdSpec,
        attn_bwd_arch_facts,
        build_attn_bwd_main,
        validate_attn_bwd_spec,
    )

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        occupancy_waves_per_simd,
        pin_production_comgr,
        read_code_object_resources,
    )

    pin_production_comgr()  # torch's bundled comgr, by path; torch not imported
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(f"needs {GATE_FLAVOR} and the production comgr ({flavor}, {comgr})")
    spec = AttnBwdSpec(head_size=d, dtype=dtype, **region_knobs(set_name))
    r = validate_attn_bwd_spec(spec, ARCH)
    art = compile_kernel(
        build_attn_bwd_main(spec, arch=ARCH), arch=ARCH, capture_ir_text=False,
        backend="python",
    )  # fmt: skip
    res = read_code_object_resources(art.hsaco, arch=ARCH)
    assert res.scratch_bytes == 0 and res.vgpr_spill == 0 and res.sgpr_spill == 0, res
    cap = attn_bwd_arch_facts(ARCH).lds_capacity_bytes
    occ = occupancy_waves_per_simd(res, lds_capacity_bytes=cap)
    assert occ >= r.waves_per_eu, (set_name, d, dtype, occ, res)
