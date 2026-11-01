# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P15 (the N3b router form of the fused MoE): ``m3_ept=2`` (the router
over every CTA), ``FusedKnobs.route="wave"`` (redundant per-wave top-10, no
route barrier) and the router mutants of
``kernels.gfx950.qwen38_mono.moe_fused``. CPU only.

* off: ``route="cta"`` is invisible (``repr``, so the stage functions' pair
  ids, and the suffix equal a build without the field); the byte identity of
  every existing build before / after the edit is
  ``hackathon-prep/qwen38/truemono/P15/p15_sha.py`` + the WP1 census.
* ``m3_ept=2``: 256 router tasks a row set at 512 experts (the shared gate is
  the last task's third router row), so ``m3_rows`` must equal the rows;
  every router row 0..512 is computed by exactly one task.
* ``route=wave`` refuses ``slots=table``, ``route_gate=pre_poll`` and
  ``spec=prev``.
* the bodies lower (llvm22 / llvm23), fit the mono's LDS arena and differ
  from their twins; every mutant differs from its base build.

GPU (bitwise vs the plain mono in situ, mutants, timing):
``hackathon-prep/qwen38/truemono/P15``.
"""

import json

import pytest
from kernels.gfx950.qwen38_mono import abi as A
from kernels.gfx950.qwen38_mono import moe as MO
from kernels.gfx950.qwen38_mono import moe_fused as MF
from kernels.gfx950.qwen38_mono import stage_moe_fused as SMF
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

MONO_LDS = {(1, False): 22640, (4, True): 38912, (4, False): 38912, (2, True): 38912, (3, True): 38912}


def _serving(rows, extra):
    return SMF.knobs_from_json(dict(SMF.BEST_KNOBS[1 if rows == 1 else 0], **extra))


def _n3b(rows, extra=None):
    d = {"m3_ept": 2, "m3_rows": rows, "route": "wave"}
    if rows > 1:
        d["slots"] = "wave"
    return dict(d, **(extra or {}))


def test_route_off_is_invisible():
    for rows in (1, 4):
        k0 = _serving(rows, {})
        assert k0.route == "cta" and " route=" not in repr(k0) and "routewave" not in k0.suffix()
        assert repr(kw0 := MF.FusedKnobs(**{**k0.__dict__, "route": "wave", "slots": "wave"})) == repr(MF.FusedKnobs(**{**k0.__dict__, "slots": "wave"})) and kw0.route == "wave"
        kw = _serving(rows, {"route": "wave", "slots": "wave"} if rows > 1 else {"route": "wave"})
        assert kw.suffix().endswith("_routewave")
        for v in ("r1",) if rows == 1 else ("r4", "r4s"):
            assert SMF.pair_id(v, kw) != SMF.pair_id(v, k0)
            assert SMF.pair_id(v, k0) == SMF.pair_id(v, MF.FusedKnobs(**{**k0.__dict__}))
    assert MO._ksum_order(object()) == list(range(16))


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_ept2_tasks_cover_every_router_row(rows):
    plan = MF.FusedPlan(rows, A.EXPERTS, _serving(rows, _n3b(rows)))
    plan.validate()
    assert plan.moe.m3_tasks == A.GRID
    nt = A.EXPERTS // 2
    got = []
    for t in range(nt):  # _emit_m3_n: rows 2t, 2t + 1 and, on the last task, row E (the shared gate)
        got += [2 * t, 2 * t + 1] + ([A.EXPERTS] if t == nt - 1 else [])
    assert sorted(got) == list(range(A.EXPERTS + 1))


@pytest.mark.parametrize("rows,extra", [
    (4, {"m3_ept": 2}),  # m3_rows 2: 512 tasks
    (4, {"m3_ept": 2, "m3_rows": 2}),
    (2, {"m3_ept": 2, "m3_rows": 1}),
])
def test_ept2_refusals(rows, extra):
    with pytest.raises(ValueError):
        MF.FusedPlan(rows, A.EXPERTS, _serving(rows, extra)).validate()


@pytest.mark.parametrize("extra", [
    {"route": "wave"},  # rows 4 serving set: slots=table
    {"route": "wave", "slots": "wave", "route_gate": "pre_poll"},
    {"route": "wave", "slots": "wave", "spec": "prev"},
    {"mutate": "route_wave7"},
    {"route": "wave2", "slots": "wave"},
])
def test_route_refusals(extra):
    with pytest.raises(ValueError):
        _serving(4, extra)


def _ir(rows, seq, knobs, flavor="llvm23"):
    kern, _p = SMF.build_function("m3", rows, seq, knobs, arena_words=MONO_LDS[(rows, seq)])
    return _lower_kernel_to_llvm_python(kern, arch="gfx950", llvm_flavor=flavor)


VARIANTS = [(1, False), (4, True), (4, False), (2, True), (3, True)]


@pytest.mark.parametrize("rows,seq", VARIANTS)
def test_n3b_bodies_lower_and_differ(rows, seq):
    base = _ir(rows, seq, _serving(rows, {}))
    arms = {
        "ept2": _n3b(rows, {"route": "cta", "slots": "arith" if rows == 1 else "table"}),
        "wave": {"route": "wave"} if rows == 1 else {"route": "wave", "slots": "wave"},
        "n3b": _n3b(rows),
    }
    seen = {"base": base}
    for name, ex in arms.items():
        k = _serving(rows, ex)
        MF.FusedPlan(rows, A.EXPERTS, k).validate()
        assert MF.lds_layout(MF.FusedPlan(rows, A.EXPERTS, k))[2] <= MONO_LDS[(rows, seq)]
        txt = _ir(rows, seq, k)
        assert txt not in seen.values(), name
        seen[name] = txt
    _ir(rows, seq, _serving(rows, _n3b(rows)), flavor="llvm22")


@pytest.mark.parametrize("rows,seq", [(1, False), (4, True)])
@pytest.mark.parametrize("base_extra,mut", [
    ({}, "router_ksum"), ({}, "router_kdrop"), ({}, "route_k11"),
    ("n3b", "router_ksum"), ("n3b", "router_kdrop"), ("n3b", "route_k11"),
], ids=lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
def test_router_mutants_differ(rows, seq, base_extra, mut):
    ex = _n3b(rows) if base_extra == "n3b" else dict(base_extra)
    good = _ir(rows, seq, _serving(rows, ex))
    bad = _ir(rows, seq, _serving(rows, dict(ex, mutate=mut)))
    assert good != bad


def test_no_single_wave_mutant():
    # a one-wave routing mutant is not a reliable must-fail mutant (shared LDS words) and deadlocks at rows > 1
    # (probe P15, hardware): it is refused
    for ex in ({"mutate": "route_wave7"}, {"mutate": "route_wave7", "route": "wave"}):
        with pytest.raises(ValueError):
            _serving(1, ex)
