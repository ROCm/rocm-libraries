# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Route choice of plan_tensor_gett for operands without a DTL free-mode run
(CPU only: planning, no device).

* An operand whose fastest mode is a K mode has no MN-layout plan of its own:
  the pack hybrid packs it (TENSOR_GETT_PACK_ANY_RUN) and leaves an operand
  with a 16-byte free-mode run in place; with the knob off such a problem
  takes the packed K-contiguous route.
* The K-contiguous GETT runs as a single launch on the user operands only when
  they already are the K-contiguous copies it reads; otherwise its route packs
  both operands.
* A problem with neither route is refused.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import rocke.dispatch.families.tensor_gett as TG
from rocke.dispatch.families.tensor_gett import (
    TENSOR_GETT_TUNED,
    TensorGettRequest,
    plan_tensor_gett,
)
from rocke.instances.common.tensor import (
    GettKnobs,
    TensorGettSpec,
    kc_operand_modes,
    launch_operands,
)
from rocke.instances.common.tensor.gett_plan import packed_strides

_EXT = {"K0": 24, "K1": 24, "M0": 72, "M1": 64, "N0": 60, "N1": 64}
_D = ["N1", "N0", "M1", "M0"]


def _req(a, b, d=_D, ext=_EXT, strides=None, dtype="bf16"):
    sa, sb = strides or (None, None)
    p = TensorGettSpec.make(
        a, b, d, ext, dtype=dtype, knobs=GettKnobs(), strides_a=sa, strides_b=sb
    )
    return TensorGettRequest(problem=p, arch="gfx950")


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    saved = dict(TENSOR_GETT_TUNED)
    TENSOR_GETT_TUNED.clear()
    monkeypatch.delenv("ROCKE_TENSOR_GETT_ROUTE", raising=False)
    yield
    TENSOR_GETT_TUNED.clear()
    TENSOR_GETT_TUNED.update(saved)


def test_k_fastest_operand_is_packed_by_the_hybrid():
    # A's fastest mode is K0: no 4-byte M run, so the unpacked GETT plan does not exist
    req = _req(["M1", "M0", "K1", "K0"], ["K1", "K0", "N1", "N0"])
    plan = TG._plan_mn(req)
    assert plan.kind == "pack_hybrid"
    assert [s.operand for s in plan.packs] == ["A"]
    assert not plan.gett.spec.kc_body
    # whichever route wins, the problem has a plan
    assert plan_tensor_gett(req).gett.spec is not None


def test_both_k_fastest_operands_are_packed():
    req = _req(["M1", "M0", "K1", "K0"], ["N1", "N0", "K1", "K0"])
    plan = TG._plan_mn(req)
    assert sorted(s.operand for s in plan.packs) == ["A", "B"]


def test_without_the_knob_the_k_contiguous_route_serves(monkeypatch):
    monkeypatch.setattr(TG, "TENSOR_GETT_PACK_ANY_RUN", False)
    req = _req(["M1", "M0", "K1", "K0"], ["K1", "K0", "N1", "N0"])
    with pytest.raises(ValueError):
        TG._plan_mn(req)
    plan = plan_tensor_gett(req)
    assert plan.gett.spec.kc_body
    assert len(plan.packs) == 2
    assert "no MN-layout route" in plan.explanation
    # the MN policy alone refuses it
    monkeypatch.setenv("ROCKE_TENSOR_GETT_ROUTE", "mn")
    with pytest.raises(ValueError):
        plan_tensor_gett(req)


def test_hybrid_roles_follow_the_gemm_role_order():
    # D's fastest mode M0 belongs to A: GEMM A is the tensor A
    roles = TG._hybrid_roles(
        _req(["M1", "M0", "K1", "K0"], ["K1", "K0", "N1", "N0"]).problem
    )
    assert roles == [("A", True), ("B", False)]
    d = ["M1", "M0", "N1", "N0"]
    roles = TG._hybrid_roles(
        _req(["M1", "M0", "K1", "K0"], ["K1", "K0", "N1", "N0"], d=d).problem
    )
    assert roles == [("B", False), ("A", True)]
    # a batch mode (in A, B and D) is not a plain contraction
    ext = dict(_EXT, L0=3)
    batched = _req(
        ["L0", "M1", "M0", "K0"], ["L0", "K0", "N1", "N0"], d=["L0"] + _D, ext=ext
    )
    assert TG._hybrid_roles(batched.problem) is None


def test_k_contiguous_single_launch_only_on_its_own_copies():
    req = _req(["K1", "K0", "M1", "M0"], ["K1", "K0", "N1", "N0"])
    sp = TG._spec_kc(req)
    assert sp is not None
    # the user operands are MN-contiguous: not the K-contiguous copies
    assert not TG._kc_in_place(req.problem, sp)
    assert TG._spec_kc_user(req) is None
    # user operands laid out exactly as the kernel's K-contiguous copies (a padded,
    # extent-1 mode does not matter)
    orders = dict(zip(launch_operands(sp), kc_operand_modes(sp)))
    st = {nm: packed_strides(orders[nm], _EXT) for nm in ("A", "B")}
    own = _req(
        ["K1", "K0", "M1", "M0"], ["K1", "K0", "N1", "N0"], strides=(st["A"], st["B"])
    )
    sp_own = TG._spec_kc(own)
    assert TG._kc_in_place(own.problem, sp_own)
    assert TG._spec_kc_user(own) is not None
    # one stride off: packs again
    bad_a = dict(st["A"])
    fast = min(bad_a, key=bad_a.get)
    bad_a = {m: (v * 2 if m != fast else v) for m, v in bad_a.items()}
    off = _req(
        ["K1", "K0", "M1", "M0"], ["K1", "K0", "N1", "N0"], strides=(bad_a, st["B"])
    )
    assert TG._spec_kc_user(off) is None


def test_kc_registry_candidate_never_reads_mn_operands():
    req = _req(["K1", "K0", "M1", "M0"], ["K1", "K0", "N1", "N0"])
    for r in TG.dispatch_tensor_gett_all(req):
        if r.spec.kc_body:
            assert TG._kc_in_place(req.problem, r.spec)


def test_unplannable_problem_is_refused():
    # a Hadamard mode (in A, B and D) with K fastest on both sides and no K mode: neither route
    ext = {"H0": 8, "M0": 72, "N0": 60}
    req = _req(["M0", "H0"], ["N0", "H0"], d=["H0", "N0", "M0"], ext=ext)
    with pytest.raises(ValueError):
        plan_tensor_gett(req)


@pytest.mark.parametrize("operand", ["A", "B"])
def test_mode_of_one_operand_only_is_refused(operand):
    # X0 is a mode of one operand only: no kernel sums it, so every route refuses the problem
    # (it is not dropped silently)
    from rocke.instances.common.tensor.gett_plan import plan_gett

    ext = dict(_EXT, X0=4)
    a = ["X0", "K1", "K0", "M1", "M0"] if operand == "A" else ["K1", "K0", "M1", "M0"]
    b = ["X0", "K1", "K0", "N1", "N0"] if operand == "B" else ["K1", "K0", "N1", "N0"]
    with pytest.raises(ValueError, match="one operand only"):
        plan_gett(a, b, _D, ext)
    req = _req(a, b, ext=ext)
    assert TG._hybrid_roles(req.problem) is None
    with pytest.raises(ValueError):
        TG._plan_mn(req)
    assert TG._plan_kc(req) is None
    with pytest.raises(ValueError):
        plan_tensor_gett(req)
    # the same problem without X0 has a route
    assert plan_tensor_gett(_req([m for m in a if m != "X0"], [m for m in b if m != "X0"]))
