# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the persistent hipTensor contraction monokernel (plans, hooks, builds).

These pin the host-side plan (canonical orders, packs, separable offset tables,
vector widths), the schedule validation and that every knob builds; the GPU
numerics are checked against an f64 reference outside the unit suite.
"""

import dataclasses
import random

import pytest
from kernels.gfx950.hiptensor_mono import (
    OperandPackSpec,
    PersistentContractionKnobs,
    PersistentContractionSpec,
    build_operand_pack,
    build_persistent_contraction,
    persistent_body_spec,
    persistent_contraction_hooks,
    plan_contraction,
)
from kernels.gfx950.hiptensor_mono.contraction_plan import (
    contiguous_run,
    offsets,
    packed_strides,
)

# hipTensor packed column-major mode lists (first mode fastest) -> row-major lists (reversed)
P1 = (["M0", "K0", "M1", "K1"], ["N0", "K0", "N1", "K1"], ["M0", "N0", "M1", "N1"])
P1_EXT = {"M0": 48, "K0": 24, "M1": 32, "K1": 24, "N0": 40, "N1": 36}
P1R3 = (
    ["M0", "K0", "M1", "K1", "M2", "K2"],
    ["N0", "K0", "N1", "K1", "N2", "K2"],
    ["M0", "N0", "M1", "N1", "M2", "N2"],
)
P1R3_EXT = {
    "K0": 16,
    "K1": 12,
    "K2": 8,
    "M0": 32,
    "M1": 20,
    "M2": 12,
    "N0": 24,
    "N1": 20,
    "N2": 14,
}
P1R5 = (
    ["M0", "K0", "M1", "K1", "M2", "K2", "M3", "K3", "M4", "K4"],
    ["N0", "K0", "N1", "K1", "N2", "K2", "N3", "K3", "N4", "K4"],
    ["M0", "N0", "M1", "N1", "M2", "N2", "M3", "N3", "M4", "N4"],
)
P1R5_EXT = {
    "K0": 4,
    "K1": 4,
    "K2": 3,
    "K3": 3,
    "K4": 2,
    "M0": 8,
    "M1": 6,
    "M2": 6,
    "M3": 4,
    "M4": 4,
    "N0": 8,
    "N1": 6,
    "N2": 4,
    "N3": 6,
    "N4": 4,
}


def _plan(modes=P1, ext=P1_EXT):
    a, b, d = modes
    return plan_contraction(a[::-1], b[::-1], d[::-1], ext)


def _flat(modes, ext, idx):
    st = packed_strides(modes, ext)
    return sum(idx[m] * st[m] for m in modes)


def test_packed_strides_offsets_and_runs():
    ext = {"a": 2, "b": 3, "c": 4}
    st = packed_strides(["a", "b", "c"], ext)
    assert st == {"a": 12, "b": 4, "c": 1}
    assert offsets(["b", "c"], ext, st) == [
        b * 4 + c for b in range(3) for c in range(4)
    ]
    assert contiguous_run(["a", "b", "c"], ext, st) == 24
    assert contiguous_run(["c", "b"], ext, st) == 1


def _digits(order, ext, lin):
    out = {}
    for m in reversed(order):
        out[m] = lin % ext[m]
        lin //= ext[m]
    return out


def test_plan_sizes_and_separable_offsets():
    plan = _plan()
    # D's fastest mode (M0) is an A mode: the GEMM computes D^T (operands swap)
    assert plan.swapped and (plan.M, plan.N, plan.K) == (40 * 36, 48 * 32, 24 * 24)
    md = P1[2][::-1]
    rng = random.Random(3)
    for _ in range(200):
        m_lin, n_lin = rng.randrange(plan.M), rng.randrange(plan.N)
        idx = {
            **_digits(plan.m_order, P1_EXT, m_lin),
            **_digits(plan.n_order, P1_EXT, n_lin),
        }
        assert plan.drow[m_lin] + plan.dcol[n_lin] == _flat(md, P1_EXT, idx)
    # every non-canonical operand is packed through separable tables
    assert {p.name for p in plan.packs} == {"A", "B"}
    for p in plan.packs:
        assert len(p.rowoff) == p.rows and len(p.koff) == plan.K
        src = P1[0][::-1] if p.name == "A" else P1[1][::-1]
        rows = plan.m_order if p.gemm_role == "A" else plan.n_order
        r_lin, k_lin = rng.randrange(p.rows), rng.randrange(plan.K)
        idx = {**_digits(rows, P1_EXT, r_lin), **_digits(plan.k_order, P1_EXT, k_lin)}
        assert p.rowoff[r_lin] + p.koff[k_lin] == _flat(src, P1_EXT, idx)


def test_plan_canonical_operands_read_in_place():
    m, n, k = ["M0", "M1"], ["N0", "N1"], ["K0", "K1"]
    ext = {"M0": 4, "M1": 8, "N0": 4, "N1": 8, "K0": 8, "K1": 8}
    plan = plan_contraction(m + k, n + k, m + n, ext)
    assert plan.packs == [] and plan.vec_d == 8 and not plan.swapped
    assert plan.describe().endswith("packs=none vecD=8")


def test_hooks_schedule_validates_and_sizes():
    plan = _plan()
    for kn in (
        PersistentContractionKnobs(),
        PersistentContractionKnobs(dep="panel", reduce="max"),
    ):
        hooks = persistent_contraction_hooks(
            PersistentContractionSpec(plan=plan, knobs=kn)
        )
        assert hooks.task_table().validate() == []
        assert hooks.n_flags >= hooks.n_pack_tasks + 1
        assert hooks.writes_operands == bool(plan.packs)


@pytest.mark.parametrize(
    "kn",
    [
        {},
        dict(dep="panel"),
        dict(bilinear=True, reduce="max"),
        dict(pre_tasks=False, dep="none"),
        dict(reg_transpose=False),
        dict(store_nt=False, epi_group=4),
        dict(store_d=False, reduce="max"),
    ],
)
def test_build_every_knob(kn):
    plan = _plan(P1R3, P1R3_EXT)
    k = build_persistent_contraction(
        PersistentContractionSpec(plan=plan, knobs=PersistentContractionKnobs(**kn)),
        arch="gfx950",
    )
    names = [p.name for p in k.params]
    assert names[:6] == ["A", "B", "C", "M", "N", "K"]
    assert "epoch" in names and "Flags" in names and "alpha" in names
    assert ("Cin" in names) == bool(kn.get("bilinear"))
    assert ("Red" in names) == bool(kn.get("reduce"))
    # in-launch pack tasks write the workspaces: A / B lose readonly
    a = k.params[0]
    writes = kn.get("pre_tasks", True) and bool(plan.packs)
    assert bool(a.attrs.get("readonly", False)) == (not writes)


def test_build_rejects_unordered_and_wrong_arch():
    plan = _plan()
    with pytest.raises(ValueError, match="dep='none'"):
        build_persistent_contraction(
            PersistentContractionSpec(
                plan=plan, knobs=PersistentContractionKnobs(dep="none")
            ),
            arch="gfx950",
        )
    with pytest.raises(ValueError, match="gfx950"):
        build_persistent_contraction(
            PersistentContractionSpec(plan=plan), arch="gfx942"
        )
    with pytest.raises(ValueError, match="dep must be"):
        persistent_contraction_hooks(
            PersistentContractionSpec(
                plan=plan, knobs=PersistentContractionKnobs(dep="tile")
            )
        )


def test_k_not_multiple_of_64_needs_register_transpose():
    plan = _plan(P1R5, P1R5_EXT)
    assert plan.K % 64 == 32
    k = build_persistent_contraction(
        PersistentContractionSpec(plan=plan), arch="gfx950"
    )
    assert k.params[0].name == "A"
    with pytest.raises(ValueError, match="register-transpose"):
        build_persistent_contraction(
            PersistentContractionSpec(
                plan=plan, knobs=PersistentContractionKnobs(reg_transpose=False)
            ),
            arch="gfx950",
        )


def test_operand_pack_kernel_and_body_traits():
    plan = _plan()
    kp = build_operand_pack(OperandPackSpec(plan=plan), arch="gfx950")
    assert [p.name for p in kp.params][-1] == "nwg"
    a = build_persistent_contraction(
        PersistentContractionSpec(plan=plan), arch="gfx950"
    )
    b = build_persistent_contraction(
        PersistentContractionSpec(plan=plan, body_traits=(("chiplet_chunk_size", 8),)),
        arch="gfx950",
    )
    assert a.name != b.name


def test_body_spec_without_hooks_keeps_readonly_operands():
    from rocke.instances.common.gemm_universal import build_universal_gemm

    body = dataclasses.replace(
        persistent_body_spec("f16", dtl_tight_rsrc=True), name="plain"
    )
    k = build_universal_gemm(body, arch="gfx950")
    assert k.params[0].attrs.get("readonly") and k.params[1].attrs.get("readonly")
