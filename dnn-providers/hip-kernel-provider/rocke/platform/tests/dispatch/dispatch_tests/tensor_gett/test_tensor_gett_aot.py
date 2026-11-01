# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tensor GETT AOT route table (``tensor_gett_aot.json``; CPU only).

The table freezes the product route (:func:`plan_tensor_gett`) of fixed
problems for ahead-of-time kernel packs. These tests keep it equal to the
product route:

* every entry is :func:`tensor_gett_aot_entry` of its stored problem (the
  route and GEMM body the cost model picks today; regenerate with
  ``tools/gen_tensor_gett_aot.py --refresh`` when the dispatcher changes);
* :func:`aot_route_plan` rebuilds exactly :func:`plan_tensor_gett`'s launches
  (pack steps, their kernels and grids, the GETT spec, grid and workspace);
* the table and the tuned route table are disjoint, and the product route
  never reads the AOT table.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import rocke.dispatch.families.tensor_gett as TG
from rocke.dispatch.families.tensor_gett import (
    TENSOR_GETT_AOT,
    TensorGettRequest,
    aot_entry_problem,
    aot_route_plan,
    plan_tensor_gett,
    tensor_gett_aot_entry,
    tensor_gett_route_key,
)
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec

# the product route as shipped: the learned GEMM selectors on (tests/conftest.py pins them off
# for the rule-layer tests), no GETT route policy override
GEMM_SELECTOR_TESTS = True
GEMM_SKINNY_SELECT_TESTS = True


@pytest.fixture(autouse=True)
def _product_route_env(monkeypatch):
    monkeypatch.delenv(TG.TENSOR_GETT_ROUTE_ENV, raising=False)


_ENTRIES = sorted(
    (arch, key) for arch, table in TENSOR_GETT_AOT.items() for key in table
)


def _req(prob: dict, arch: str) -> TensorGettRequest:
    p = TensorGettSpec.make(
        prob["modes_a"],
        prob["modes_b"],
        prob["modes_d"],
        prob["extents"],
        dtype=prob["dtype"],
        knobs=GettKnobs(bilinear=prob["bilinear"]),
    )
    return TensorGettRequest(problem=p, arch=arch)


def _launches(plan) -> tuple:
    return (
        tuple(
            (s.operand, s.spec, tuple(s.grid), tuple(s.block), s.workspace_bytes)
            for s in plan.packs
        ),
        plan.gett.spec,
        tuple(plan.gett.grid),
        tuple(plan.gett.block),
        plan.workspace_bytes,
        plan.gett.candidate.name,
    )


def test_table_is_shipped_and_disjoint_from_the_tuned_table():
    assert _ENTRIES, "the AOT route table has no entries"
    for arch, table in TENSOR_GETT_AOT.items():
        tuned = TG._SHIPPED_TUNED.get(arch, {})
        assert not set(table) & set(tuned)
        for key, (route, _cfg) in table.items():
            assert route in ("mn", "kc"), key


@pytest.mark.parametrize("arch,key", _ENTRIES)
def test_entry_is_the_product_route(arch, key):
    prob = aot_entry_problem(arch, key)
    assert prob is not None
    req = _req(prob, arch)
    assert tensor_gett_route_key(req) == key
    got = tensor_gett_aot_entry(req)
    assert got is not None, "the product route no longer runs a GEMM body"
    k2, ent = got
    assert k2 == key
    route, cfg = TENSOR_GETT_AOT[arch][key]
    assert (ent["route"], ent["body"]) == (
        route,
        cfg,
    ), "stale AOT route table entry: run tools/gen_tensor_gett_aot.py --refresh"
    assert ent["problem"] == prob
    tbl = aot_route_plan(req)
    assert tbl is not None
    assert "AOT route table" in tbl.explanation
    assert _launches(tbl) == _launches(plan_tensor_gett(req))


def test_product_route_ignores_the_aot_table(monkeypatch):
    """A changed AOT entry changes aot_route_plan, never plan_tensor_gett."""
    arch, key = _ENTRIES[0]
    req = _req(aot_entry_problem(arch, key), arch)
    before = _launches(plan_tensor_gett(req))
    route, cfg = TENSOR_GETT_AOT[arch][key]
    other = (
        "t128x128x64_w2x4x1_a16x16x32|direct_to_lds=True,dtl_prefetch=True,dtl_stages=3"
    )
    monkeypatch.setitem(
        TENSOR_GETT_AOT, arch, {**TENSOR_GETT_AOT[arch], key: ("mn", other)}
    )
    assert _launches(plan_tensor_gett(req)) == before
    alt = aot_route_plan(req)
    assert alt is None or _launches(alt) != before


def test_strided_problems_have_no_entry():
    arch, key = _ENTRIES[0]
    req = _req(aot_entry_problem(arch, key), arch)
    p = req.problem
    st = {m: 1 for m in p.modes_a}
    strided = replace(req, problem=replace(p, strides_a=tuple(sorted(st.items()))))
    assert tensor_gett_aot_entry(strided) is None


def _parity_rows() -> tuple:
    import importlib.util
    import pathlib
    import sys

    parity = pathlib.Path(__file__).resolve().parents[3] / "instances" / "parity"
    sys.path.insert(0, str(parity))
    try:
        spec = importlib.util.spec_from_file_location(
            "_tensor_gett_aot_emit", parity / "tensor_gett_aot_emit.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(parity))
    return mod.ARCH, mod.ROWS


def test_parity_rows_cover_the_table():
    """The byte-identity rows (tests/instances/parity/tensor_gett_aot_emit) build
    every entry's GETT kernel and every distinct pack kernel of the table, so the
    C++ pack builder's kernels are all checked against the Python engine."""
    arch, rows = _parity_rows()
    table = TENSOR_GETT_AOT.get(arch, {})
    assert {k for k, s in rows if s < 0} == set(table)
    assert len(set(rows)) == len(rows)
    need, have = set(), set()
    for key in sorted(table):
        plan = aot_route_plan(_req(aot_entry_problem(arch, key), arch))
        assert plan is not None, key
        for j, st in enumerate(plan.packs):
            need.add(repr(st.spec))
            if (key, j) in rows:
                have.add(repr(st.spec))
    for key, s in rows:
        assert key in table, key
    assert need == have, f"pack kernels without a parity row: {len(need - have)}"
