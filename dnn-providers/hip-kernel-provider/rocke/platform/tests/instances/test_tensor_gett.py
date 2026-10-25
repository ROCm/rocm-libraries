# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks of the tensor GETT family (``rocke.instances.common.tensor``): the planner, its DTL / LDS /
fragment address algebra, and the family contract (validity, kernel params vs signature, grid, body guard).
CPU only (the device numerics need a GPU)."""

import itertools

import pytest
from rocke.instances.common.tensor import (
    GettHooks,
    GettKnobs,
    TensorGettSpec,
    build_tensor_gett,
    is_valid_spec,
    launch_operands,
    plan_gett,
    tensor_gett_grid,
    tensor_gett_signature,
)
from rocke.instances.common.tensor.gett import body_spec, body_trait_errors
from rocke.instances.common.tensor.gett_plan import mixed_offset, packed_strides

# P1 / P3 / P4-style permuted contractions (column-major mode lists, reversed to row-major below)
CASES = {
    "P1_r3": (
        ["M0", "K0", "M1", "K1", "M2", "K2"],
        ["N0", "K0", "N1", "K1", "N2", "K2"],
        ["M0", "N0", "M1", "N1", "M2", "N2"],
        {
            "K0": 16,
            "K1": 12,
            "K2": 8,
            "M0": 32,
            "M1": 20,
            "M2": 12,
            "N0": 24,
            "N1": 20,
            "N2": 14,
        },
    ),
    "P1_r5": (
        ["M0", "K0", "M1", "K1", "M2", "K2", "M3", "K3", "M4", "K4"],
        ["N0", "K0", "N1", "K1", "N2", "K2", "N3", "K3", "N4", "K4"],
        ["M0", "N0", "M1", "N1", "M2", "N2", "M3", "N3", "M4", "N4"],
        {
            "K0": 6,
            "K1": 4,
            "K2": 4,
            "K3": 4,
            "K4": 2,
            "M0": 10,
            "M1": 8,
            "M2": 8,
            "M3": 6,
            "M4": 4,
            "N0": 10,
            "N1": 8,
            "N2": 6,
            "N3": 8,
            "N4": 4,
        },
    ),
    "P3_r4": (
        ["M0", "M1", "M2", "M3", "K0", "K1", "K2", "K3"],
        ["N0", "N1", "N2", "N3", "K0", "K1", "K2", "K3"],
        ["M0", "N0", "M1", "N1", "M2", "N2", "M3", "N3"],
        {
            "K0": 8,
            "K1": 8,
            "K2": 6,
            "K3": 4,
            "M0": 16,
            "M1": 12,
            "M2": 10,
            "M3": 8,
            "N0": 14,
            "N1": 10,
            "N2": 12,
            "N3": 8,
        },
    ),
    "P4_r3": (
        ["M0", "M1", "M2", "K0", "K1", "K2"],
        ["N0", "N1", "N2", "K2", "K1", "K0"],
        ["M0", "M1", "M2", "N0", "N1", "N2"],
        {
            "K0": 16,
            "K1": 12,
            "K2": 8,
            "M0": 32,
            "M1": 20,
            "M2": 12,
            "N0": 24,
            "N1": 20,
            "N2": 14,
        },
    ),
}


def _plan(name, tile_k=32):
    a, b, d, ext = CASES[name]
    return plan_gett(a[::-1], b[::-1], d[::-1], ext, tile_k=tile_k), ext


def _koff(plan, role, k):
    """Direct element offset of flat K index k (k_order inner first) for GEMM operand `role`."""
    st = plan.ops[role].strides
    off = 0
    for m in plan.k_order:
        e = plan.ext[m]
        off += (k % e) * st[m]
        k //= e
    return off


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("tile_k", [32, 64])
def test_k_split_is_separable(name, tile_k):
    plan, _ = _plan(name, tile_k)
    T = plan.tile_k
    assert plan.K % T == 0
    for role in ("A", "B"):
        for k in range(0, plan.K, max(1, plan.K // 997)):
            assert _koff(plan, role, k) == plan.ktile(k // T, role) + plan.klocal(
                k % T, role
            )


@pytest.mark.parametrize("name", sorted(CASES))
def test_row_and_d_offsets_match_strides(name):
    plan, ext = _plan(name)
    for role in ("A", "B"):
        op = plan.ops[role]
        st = op.strides
        for r in range(0, op.rows, max(1, op.rows // 509)):
            direct, x = 0, r
            for m in reversed(op.rows_order):
                direct += (x % ext[m]) * st[m]
                x //= ext[m]
            assert mixed_offset(r, op.row_digits) == direct
    sd = plan.strides_d
    for r in range(0, plan.M, max(1, plan.M // 211)):
        direct, x = 0, r
        for m in reversed(plan.ops["A"].rows_order):
            direct += (x % ext[m]) * sd[m]
            x //= ext[m]
        assert mixed_offset(r, plan.d_row_digits) == direct


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("pad", [32, 0])
def test_dtl_lanes_cover_the_tile_once_and_the_reader_finds_them(name, pad):
    """Every (k, mn-chunk) of a stage is written by exactly one (pass, lane), at the LDS element the
    ds_read_b64_tr_b16 reader addresses for that (k, mn)."""
    plan, _ = _plan(name)
    spec = body_spec(tile_k=plan.tile_k)
    h = GettHooks(plan, GettKnobs(pad=pad), spec)
    HB = h.block_bytes() // 2
    for role in ("A", "B"):
        lb = h.lane_bytes(role)
        v = lb // 2
        seen = {}
        for p in range(h.passes(role)):
            for tid in range(256):
                k, mn = h._lane_geom(role, p, tid)
                w = tid // 64
                lds_byte = (
                    p * h.pass_bytes(role) + w * h.wave_bytes(role) + (tid % 64) * lb
                )
                for e in range(v):
                    key = (k, mn + e)
                    assert key not in seen
                    seen[key] = lds_byte // 2 + e
        assert len(seen) == plan.tile_k * 256
        # reader: lane l, atom idx, K atom kk, read r -> element (k, col..col+3)
        for wv, idx, kk, r, lane in itertools.product(
            range(2), range(8), range(plan.tile_k // 32), range(2), range(64)
        ):
            kq, g = lane // 16, (lane % 16) // 4
            blk = g + 4 * (kq % 2) + 8 * (kq // 2)
            elem = (
                blk * HB + (lane % 4) * 4 + wv * 128 + 16 * kk * HB + r * 256 + 16 * idx
            )
            k = kk * 32 + kq * 8 + g + 4 * r
            col = wv * 128 + idx * 16 + (lane % 4) * 4
            for e in range(4):
                assert seen[(k, col + e)] == elem + e


@pytest.mark.parametrize("name", sorted(CASES))
def test_lane_split_is_additive(name):
    plan, _ = _plan(name)
    h = GettHooks(plan, GettKnobs(), body_spec(tile_k=plan.tile_k))
    for role in ("A", "B"):
        if not h.lane_split(role):
            continue
        for p in range(h.passes(role)):
            for tid in range(0, 256, 7):
                k, _mn = h._lane_geom(role, p, tid)
                k0, _ = h._lane_geom(role, 0, tid)
                assert 2 * plan.klocal(k, role) == 2 * plan.klocal(
                    k0, role
                ) + h.pass_soff_bytes(role, p)


def test_packed_strides_row_major():
    assert packed_strides(["a", "b", "c"], {"a": 2, "b": 3, "c": 4}) == {
        "a": 12,
        "b": 4,
        "c": 1,
    }


# ------------------------------------------------------------------------------------------------ family contract


def _spec(name="P3_r4", **kw):
    a, b, d, ext = CASES[name]
    return TensorGettSpec.make(a[::-1], b[::-1], d[::-1], ext, **kw)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_default_spec_valid_and_params_match_signature(name, dtype):
    for kn in (GettKnobs(), GettKnobs(bilinear=True)):
        sp = _spec(name, dtype=dtype, knobs=kn)
        ok, why = is_valid_spec(sp)
        assert ok, why
        kd = build_tensor_gett(sp)
        assert [p.name for p in kd.params] == [
            s["name"] for s in tensor_gett_signature(sp)
        ]
        plan = sp.plan()
        assert tensor_gett_grid(sp) == (-(-plan.N // 256), -(-plan.M // 256), 1)
        assert set(launch_operands(sp)) == {"A", "B"}


def test_persistent_grid_and_name_is_spec_unique():
    sp = _spec(persistent_ctas=256)
    assert is_valid_spec(sp)[0]
    assert tensor_gett_grid(sp) == (256, 1, 1)
    names = {
        _spec().kernel_name(),
        _spec(persistent_ctas=256).kernel_name(),
        _spec(knobs=GettKnobs(epi_pair=True)).kernel_name(),
        _spec(dtl_stages=2).kernel_name(),
        _spec(dtype="bf16").kernel_name(),
    }
    assert len(names) == 5


@pytest.mark.parametrize(
    "kw,why",
    [
        (dict(dtype="f32"), "dtype"),
        (dict(tile_k=16), "tile_k"),
        (dict(knobs=GettKnobs(pad=16)), "pad"),
        (dict(knobs=GettKnobs(epi_order="x")), "epi_order"),
        (dict(knobs=GettKnobs(acc_pin=3)), "acc_pin"),
        (dict(dtl_stages=5), "dtl_stages"),
    ],
)
def test_invalid_specs_rejected(kw, why):
    ok, msg = is_valid_spec(_spec(**kw))
    assert not ok and why in msg
    with pytest.raises(ValueError):
        build_tensor_gett(_spec(**kw))


def test_other_arches_and_unplannable_problems_rejected():
    assert not is_valid_spec(_spec(), arch="gfx942")[0]
    # batch (Hadamard) mode: in A, B and D
    sp = TensorGettSpec.make(
        ["L", "M", "K"],
        ["L", "N", "K"],
        ["L", "M", "N"],
        {"L": 4, "M": 64, "N": 64, "K": 64},
    )
    ok, msg = is_valid_spec(sp)
    assert not ok and "no GETT plan" in msg
    # odd fastest free run: no 4-byte DTL lane
    sp = TensorGettSpec.make(
        ["K", "M"], ["K", "N"], ["M", "N"], {"M": 33, "N": 64, "K": 64}
    )
    assert not is_valid_spec(sp)[0]


def test_body_guard_rejects_other_levers():
    assert body_trait_errors(body_spec().trait) == []
    assert (
        body_trait_errors(
            body_spec(persistent_ctas=256, dtl_stages=4, chiplet_wgm=16).trait
        )
        == []
    )
    for lever in (
        dict(lds_row_interleave=True),
        dict(hb_sched=True),
        dict(streamk=True),
        dict(split_k=2),
        dict(mfma_serpentine=True),
        dict(pad_m=True),
        dict(dtl_soff0=True),
    ):
        errs = body_trait_errors(body_spec(**lever).trait)
        assert errs and next(iter(lever)) in errs[0]
    plan, _ = _plan("P3_r4")
    with pytest.raises(ValueError, match="GETT body"):
        GettHooks(plan, GettKnobs(), body_spec(tile_k=plan.tile_k, hb_sched=True))
