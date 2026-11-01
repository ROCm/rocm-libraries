# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the f64 / cf64 GETT family: index maps, planning, spec checks
and the emitted kernel structure (no GPU; device numerics are in
``tests/numeric/test_gett_f64_numeric.py``)."""

from __future__ import annotations

import itertools
import math
import random

import pytest
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import ERROR, verify
from rocke.instances.common.gett_f64 import (
    F64_GETT_ARCHS,
    GettF64Problem,
    GettF64Spec,
    _load_plan,
    build_gett_f64,
    gemm_problem,
    gett_f64_grid,
    gett_f64_signature,
    is_valid_spec,
    k_split,
    mixed_offset,
)
from rocke.instances.common.gett_f64_plan import (
    merge_digits,
    packed_strides,
    plan_contraction,
)


# ------------------------------------------------------------------ k_split


@pytest.mark.parametrize("seed", range(40))
def test_k_split_is_exact_when_it_splits(seed):
    rnd = random.Random(seed)
    nd = rnd.randint(1, 3)
    digits = tuple(
        (rnd.choice([2, 3, 4, 6, 8, 12, 16, 24]), rnd.randint(1, 50)) for _ in range(nd)
    )
    tile = rnd.choice([4, 8, 12, 16, 24, 32])
    split = k_split(digits, tile)
    if nd == 1:
        assert split is not None
    if split is None:
        return
    local, tiled = split
    K = math.prod(e for e, _ in digits)
    for k in range(K):
        kt, kl = divmod(k, tile)
        assert mixed_offset(k, digits) == mixed_offset(kt, tiled) + mixed_offset(
            kl, local
        )


def test_k_split_rejects_a_straddling_tile():
    assert k_split(((6, 1), (8, 6)), 8) is None
    assert k_split(((6, 1), (8, 6)), 12) is not None


# ------------------------------------------------------------------ planner


def _true_offset(modes, strides, idx):
    return sum(idx[m] * strides[m] for m in modes)


def _random_contraction(rnd, ranks=(2, 2, 2, 0), padded=False):
    nm, nn, nk, nl = ranks
    names = [f"m{i}" for i in range(nm)] + [f"n{i}" for i in range(nn)]
    names += [f"k{i}" for i in range(nk)] + [f"l{i}" for i in range(nl)]
    ext = {m: rnd.choice([1, 2, 3, 4, 5, 6]) for m in names}
    ma = [m for m in names if m[0] in "mkl"]
    mb = [m for m in names if m[0] in "nkl"]
    md = [m for m in names if m[0] in "mnl"]
    for lst in (ma, mb, md):
        rnd.shuffle(lst)

    def strides(modes):
        st = packed_strides(modes, ext)
        if padded:  # leave a gap after a random mode
            cut = rnd.randrange(len(modes))
            st = {m: (s * 2 if i > cut else s) for i, (m, s) in enumerate(st.items())}
        return st

    return ma, mb, md, ext, strides(ma), strides(mb), strides(md), strides(md)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("padded", [False, True])
def test_plan_digits_reproduce_tensor_offsets(seed, padded):
    rnd = random.Random(seed * 7 + padded)
    ranks = (rnd.randint(1, 3), rnd.randint(1, 3), rnd.randint(1, 2), rnd.randint(0, 1))
    ma, mb, md, ext, sa, sb, sd, sc = _random_contraction(rnd, ranks, padded)
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext, strides_c=sc)
    assert len(plan.launches) == 1
    p = plan.launches[0].problem
    A, B = (mb, ma) if plan.swapped else (ma, mb)
    SA, SB = (sb, sa) if plan.swapped else (sa, sb)
    space = lambda order: [  # noqa: E731
        dict(zip(order, vals))
        for vals in itertools.product(*[range(ext[m]) for m in order])
    ]

    def lin(order, idx):  # fastest-first linear index of a mode assignment
        out, mul = 0, 1
        for m in order:
            out += idx[m] * mul
            mul *= ext[m]
        return out

    for lidx in space(plan.l_order)[:4]:
        for midx in space(plan.m_order)[:20]:
            for kidx in space(plan.k_order)[:20]:
                idx = {**lidx, **midx, **kidx}
                l, m, k = (
                    lin(plan.l_order, idx),
                    lin(plan.m_order, idx),
                    lin(plan.k_order, idx),
                )
                got = (
                    mixed_offset(l, p.a_l)
                    + mixed_offset(m, p.a_m)
                    + mixed_offset(k, p.a_k)
                )
                assert got == _true_offset(A, SA, idx)
            for nidx in space(plan.n_order)[:20]:
                idx = {**lidx, **midx, **nidx}
                l, m, n = (
                    lin(plan.l_order, idx),
                    lin(plan.m_order, idx),
                    lin(plan.n_order, idx),
                )
                got = (
                    mixed_offset(l, p.d_l)
                    + mixed_offset(m, p.d_m)
                    + mixed_offset(n, p.d_n)
                )
                assert got == _true_offset(md, sd, idx)
        for nidx in space(plan.n_order)[:20]:
            for kidx in space(plan.k_order)[:20]:
                idx = {**lidx, **nidx, **kidx}
                l, n, k = (
                    lin(plan.l_order, idx),
                    lin(plan.n_order, idx),
                    lin(plan.k_order, idx),
                )
                got = (
                    mixed_offset(l, p.b_l)
                    + mixed_offset(n, p.b_n)
                    + mixed_offset(k, p.b_k)
                )
                assert got == _true_offset(B, SB, idx)


def test_plan_roles_follow_d_fastest_mode():
    ext = {"m": 8, "n": 8, "k": 8}
    a, b = ["m", "k"], ["n", "k"]
    sa, sb = packed_strides(a, ext), packed_strides(b, ext)
    plan = plan_contraction(
        a, sa, b, sb, ["n", "m"], packed_strides(["n", "m"], ext), ext
    )
    assert not plan.swapped  # D fastest is n (hipTensor B) -> B is GEMM-B
    plan = plan_contraction(
        a, sa, b, sb, ["m", "n"], packed_strides(["m", "n"], ext), ext
    )
    assert plan.swapped and plan.n_order == ["m"]


def test_plan_merges_contiguous_modes():
    ext = {"a": 3, "b": 4, "c": 5}
    st = packed_strides(["a", "b", "c"], ext)
    assert merge_digits(["a", "b", "c"], ext, st) == ((60, 1),)
    assert merge_digits(["b", "a", "c"], ext, st) == ((4, 3), (3, 1), (5, 12))


def test_plan_splits_large_contractions_and_covers_every_output_once():
    ext = {"m0": 6, "m1": 4, "n0": 5, "n1": 4, "k0": 3}
    ma, mb, md = ["m0", "k0", "m1"], ["n0", "k0", "n1"], ["m0", "n0", "m1", "n1"]
    sa, sb, sd = (packed_strides(x, ext) for x in (ma, mb, md))
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext, max_span=8 * 150)
    assert len(plan.launches) > 1
    seen = {}
    for launch in plan.launches:
        p = launch.problem
        assert all(p.span_bytes(w) <= 8 * 150 for w in "ABCD")
        for m in range(p.M):
            for n in range(p.N):
                off = launch.d_offset + mixed_offset(m, p.d_m) + mixed_offset(n, p.d_n)
                seen[off] = seen.get(off, 0) + 1
    assert sorted(seen) == list(range(math.prod(ext[m] for m in md)))
    assert set(seen.values()) == {1}


def test_plan_rejects_lonely_modes():
    ext = {"m": 2, "n": 2, "k": 2, "x": 3}
    with pytest.raises(ValueError, match="only one tensor"):
        plan_contraction(
            ["m", "k", "x"],
            packed_strides(["m", "k", "x"], ext),
            ["n", "k"],
            packed_strides(["n", "k"], ext),
            ["m", "n"],
            packed_strides(["m", "n"], ext),
            ext,
        )


# --------------------------------------------------------------- the spec


def _rcr(M=256, N=256, K=128, **kw):
    return gemm_problem(
        M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1), **kw
    )


def test_load_plans_pick_the_contiguous_direction():
    spec = GettF64Spec(_rcr())
    p = spec.problem
    assert _load_plan(spec, p.a_m, p.a_k, spec.tile_m).along_k  # A is K-contiguous
    p2 = gemm_problem(
        256, 256, 128, a_strides=(1, 256), b_strides=(1, 256), d_strides=(256, 1)
    )
    s2 = GettF64Spec(p2)
    plan = _load_plan(s2, p2.a_m, p2.a_k, s2.tile_m)
    assert not plan.along_k and plan.vec == 2


@pytest.mark.parametrize(
    "kw,arch,needle",
    [
        ({}, "gfx1151", "has no 16x16x4 f64 MMA atom"),
        ({"tile_k": 6}, "gfx950", "tile_k must be"),
        ({"tile_m": 72}, "gfx950", "whole number of 16x16 atoms"),
        ({"tile_m": 256, "tile_n": 256}, "gfx950", "accumulator needs"),
        ({"tile_k": 32}, "gfx942", "LDS"),
        ({"warps_m": 3}, "gfx950", "warps_m * warps_n"),
        ({"lds_stages": 3}, "gfx950", "lds_stages must be"),
        ({"lds_swizzle_mode": 3}, "gfx950", "lds_swizzle_mode must be"),
        ({"tile_k": 32, "lds_stages": 2}, "gfx90a", "LDS"),
    ],
)
def test_invalid_specs_are_rejected(kw, arch, needle):
    ok, why = is_valid_spec(GettF64Spec(_rcr(), **kw), arch=arch)
    assert not ok and needle in why


def test_problem_checks_digit_extents():
    bad = GettF64Problem(
        M=8,
        N=8,
        K=8,
        a_m=((4, 1),),
        a_k=((8, 8),),
        b_n=((8, 1),),
        b_k=((8, 8),),
        d_m=((8, 8),),
        d_n=((8, 1),),
    )
    ok, why = bad.check()
    assert not ok and "M digits" in why
    with pytest.raises(ValueError):
        build_gett_f64(GettF64Spec(bad), arch="gfx950")


def test_spans_over_2gib_need_a_split():
    p = _rcr(M=65536, N=65536, K=64)
    ok, why = is_valid_spec(GettF64Spec(p), arch="gfx950")
    assert not ok and "split the launch" in why


# ------------------------------------------------------------- the kernel


def _kernel_text(spec, arch, flavor="llvm22"):
    kernel = build_gett_f64(spec, arch=arch)
    assert not [d for d in verify(kernel) if d.severity == ERROR]
    ir = serialize(kernel)
    assert serialize(parse(ir)) == ir
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


@pytest.mark.parametrize("arch", F64_GETT_ARCHS)
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_real_kernel_structure(arch, flavor):
    # 64 KiB of LDS on gfx90a / gfx942: two 8-row K stages there.
    tile_k = 16 if arch == "gfx950" else 8
    spec = GettF64Spec(_rcr(K=100), tile_k=tile_k)
    ll = _kernel_text(spec, arch, flavor)
    body = ll.split("define")[1]
    # 4x4 atoms per wave, tile_k / 4 k-steps per K tile; the loop body and the
    # last K tile after it.
    assert body.count("@llvm.amdgcn.mfma.f64.16x16x4f64(") == 2 * 16 * (tile_k // 4)
    assert " float" not in body.replace("<2 x float>", "").replace(
        "bitcast <2 x float>", ""
    )
    assert body.count("@llvm.amdgcn.s.barrier()") >= 2
    assert "fcmp oeq double %beta" in body  # beta == 0 skips C
    assert "fmul float" not in body and "fadd float" not in body


@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_complex_kernel_is_fused_4m(conj):
    p = _rcr(M=128, N=64, K=32, complex_=True, conj_a=conj[0], conj_b=conj[1])
    spec = GettF64Spec(p, tile_m=64, tile_n=64)
    ll = _kernel_text(spec, "gfx950")
    body = ll.split("define")[1]
    nk = 2  # K = 32, tile_k = 16
    # 2x2 atoms per wave, 4 real MFMAs per complex atom, 4 k-steps per tile.
    per_tile = 2 * 2 * 4 * 4
    assert body.count("@llvm.amdgcn.mfma.f64.16x16x4f64(") in (per_tile, per_tile * nk)
    n_neg = body.count("fneg double")
    # Negated fragments: A's imaginary part unless exactly one operand is
    # conjugated by B only; B's imaginary part when B is conjugated.
    assert n_neg > 0
    assert "%alpha_im" in body and "%beta_im" in body


def test_grid_and_signature():
    spec = GettF64Spec(_rcr(M=300, N=200, K=64, batch=3, batch_strides=(1, 1, 1)))
    assert gett_f64_grid(spec) == (math.ceil(300 / 128) * math.ceil(200 / 128), 3, 1)
    names = [a["name"] for a in gett_f64_signature(spec)]
    assert names == ["A", "B", "C", "D", "alpha", "beta"]
    names = [a["name"] for a in gett_f64_signature(GettF64Spec(_rcr(), bilinear=False))]
    assert names == ["A", "B", "D", "alpha"]


def test_kernel_names_are_unique_per_layout():
    a = GettF64Spec(_rcr()).kernel_name()
    b = GettF64Spec(
        gemm_problem(
            256, 256, 128, a_strides=(1, 256), b_strides=(128, 1), d_strides=(256, 1)
        )
    ).kernel_name()
    assert a != b


def test_f64_scalar_kernargs_pack_as_doubles():
    import ctypes
    import struct

    from rocke.runtime.packing import compile_packer, pack_args, pack_args_kernelparams

    sig = gett_f64_signature(GettF64Spec(_rcr()))
    vals = {"A": 16, "B": 32, "C": 48, "D": 64, "alpha": 1.0 + 2.0**-40, "beta": -0.25}
    want = struct.pack("<4Q2d", 16, 32, 48, 64, 1.0 + 2.0**-40, -0.25)
    assert pack_args(sig, vals) == want
    assert compile_packer(sig)(vals) == want
    params = pack_args_kernelparams(sig, vals)
    assert isinstance(params[4], ctypes.c_double) and params[4].value == 1.0 + 2.0**-40


# ------------------------------------------------------------ LDS swizzle


def _bank_load(addresses_bytes, width=8, banks=64):
    """Accesses per 4-byte bank for one LDS transaction of ``width``-byte lanes."""
    load = {}
    for a in addresses_bytes:
        for w in range(width // 4):
            bank = (a // 4 + w) % banks
            load[bank] = load.get(bank, 0) + 1
    return max(load.values())


@pytest.mark.parametrize("side", [32, 64, 128, 256])
@pytest.mark.parametrize("tile_k", [8, 16, 32])
def test_swizzle_reads_are_conflict_free_and_writes_spread(side, tile_k):
    from rocke.instances.common.gett_f64 import _swz_const

    spec = GettF64Spec(
        _rcr(), tile_m=side if side >= 32 else 32, tile_n=64, tile_k=tile_k
    )
    mask = spec.swizzle_mask(side)
    pitch = side * 8  # no padding with the swizzle
    for k_in in range(tile_k):
        cols = {c ^ _swz_const(k_in, mask) for c in range(side)}
        assert cols == set(range(side))  # a permutation within the row
    # ds_read_b64 half-wave: lanes 0-15 row 4ks + 0/2, lanes 16-31 the next row.
    for ks in range(tile_k // 4):
        for half in (0, 1):
            for c0 in range(0, side, 16):
                addrs = []
                for lane in range(32):
                    k = 4 * ks + 2 * half + lane // 16
                    col = (c0 + lane % 16) ^ _swz_const(k, mask)
                    addrs.append(k * pitch + col * 8)
                assert _bank_load(addrs) == 1
    # K-contiguous loader: one ds_write_b64 of 64 lanes covers tile_k/2 K rows
    # (every other row) times 128/tile_k columns.
    cpl = tile_k // 2
    lines = 64 // cpl
    if lines <= side:
        for j in (0, 1):
            addrs = []
            for lane in range(64):
                c, line = lane % cpl, lane // cpl
                k = 2 * c + j
                addrs.append(k * pitch + (line ^ _swz_const(k, mask)) * 8)
            # 512 bytes need two passes over 64 banks; allow 2x that.
            assert _bank_load(addrs) <= 4


@pytest.mark.parametrize("arch", F64_GETT_ARCHS)
def test_one_lds_stage_halves_the_lds_and_adds_a_barrier(arch):
    s2 = GettF64Spec(_rcr(K=100), tile_k=16)
    s1 = GettF64Spec(_rcr(K=100), tile_k=32, lds_stages=1)
    assert not is_valid_spec(GettF64Spec(_rcr(K=100), tile_k=32), arch="gfx90a")[0]
    assert is_valid_spec(s1, arch=arch)[0]
    assert "_s1_" in s1.kernel_name() and "_s2" not in s2.kernel_name()
    body1 = _kernel_text(s1, arch).split("define")[1]
    body2 = _kernel_text(s2, arch).split("define")[1]
    # same LDS bytes (one stage of 32 rows vs two of 16), a second barrier in the K loop
    assert (
        body1.count("@llvm.amdgcn.s.barrier()")
        == body2.count("@llvm.amdgcn.s.barrier()") + 1
    )
    assert body1.count("@llvm.amdgcn.mfma.f64.16x16x4f64(") == 2 * body2.count(
        "@llvm.amdgcn.mfma.f64.16x16x4f64("
    )


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("side", [32, 64, 128, 256])
@pytest.mark.parametrize("tile_k", [8, 16, 32])
def test_swizzle_modes_are_row_permutations_with_bounded_conflicts(mode, side, tile_k):
    from rocke.instances.common.gett_f64 import _swz_const

    spec = GettF64Spec(
        _rcr(), tile_m=side, tile_n=64, tile_k=tile_k, lds_swizzle_mode=mode
    )
    assert spec.uses_swizzle()
    mask = spec.swizzle_mask(side)
    pitch = side * 8
    for k_in in range(tile_k):
        assert {c ^ _swz_const(k_in, mask, mode) for c in range(side)} == set(
            range(side)
        )
    worst_read = 0
    for ks in range(tile_k // 4):
        for half in (0, 1):
            for c0 in range(0, side, 16):
                addrs = []
                for lane in range(32):
                    k = 4 * ks + 2 * half + lane // 16
                    col = (c0 + lane % 16) ^ _swz_const(k, mask, mode)
                    addrs.append(k * pitch + col * 8)
                worst_read = max(worst_read, _bank_load(addrs))
    # mode 1 keeps the odd-row flip (conflict-free reads); mode 2 trades it
    # for the spread of the K-pair stores.
    assert worst_read == (1 if mode == 1 else 2)
    cpl = tile_k // 2
    lines = 64 // cpl
    if lines <= side:
        for j in (0, 1):
            addrs = []
            for lane in range(64):
                c, line = lane % cpl, lane // cpl
                k = 2 * c + j
                addrs.append(k * pitch + (line ^ _swz_const(k, mask, mode)) * 8)
            load = _bank_load(addrs)
            assert load <= 4
            if mode == 2 and tile_k >= 16:
                assert load == 2  # two passes over 64 banks: the minimum for 512 bytes


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("arch", F64_GETT_ARCHS)
def test_swizzle_modes_and_stages_build(mode, stages, arch):
    p = gemm_problem(
        200, 136, 68, a_strides=(68, 1), b_strides=(68, 1), d_strides=(1, 200)
    )
    spec = GettF64Spec(p, tile_k=8, lds_swizzle_mode=mode, lds_stages=stages)
    assert is_valid_spec(spec, arch=arch)[0]
    k = build_gett_f64(spec, arch=arch)
    assert k.name == spec.kernel_name()
    assert (f"_z{mode}_" in k.name) == (mode != 0)
    assert ("_s1_" in k.name) == (stages == 1)
    assert not verify(k) or all(d.severity != ERROR for d in verify(k))


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"prefetch_depth": 3}, "prefetch_depth must be"),
        ({"prefetch_depth": 2, "prefetch": False}, "needs prefetch"),
        ({"complex_accs": 3}, "complex_accs must be"),
        ({"unroll2": True, "lds_stages": 1}, "unroll2 needs"),
        ({"unroll2": True, "prefetch_depth": 2}, "unroll2 needs"),
        ({"k_stagger": -1}, "k_stagger must be"),
        ({"k_block": -1}, "k_block must be"),
        ({"k_block": 3, "unroll2": True}, "k_block must be even"),
    ],
)
def test_schedule_knob_combinations_are_checked(kw, needle):
    ok, why = is_valid_spec(GettF64Spec(_rcr(), **kw), arch="gfx950")
    assert not ok and needle in why


@pytest.mark.parametrize("K", [64, 80, 16])
def test_unroll2_runs_every_k_tile_once(K):
    # tile_k 16: K / 16 tiles; the last pair (even counts) or the last tile
    # (odd counts) runs after the loop.
    spec = GettF64Spec(_rcr(K=K), tile_k=16, unroll2=True)
    body = _kernel_text(spec, "gfx950").split("define")[1]
    nk = -(-K // 16)
    per_tile = 16 * 4  # 4x4 atoms per wave, 4 k-steps
    in_loop = 2 * per_tile  # the loop body is emitted once (it may run zero times)
    tail = per_tile if nk % 2 else 2 * per_tile  # the last tile, or the last pair
    assert body.count("@llvm.amdgcn.mfma.f64.16x16x4f64(") == in_loop + tail
    assert spec.kernel_name().endswith("_u2")


def test_prefetch_depth_two_carries_the_next_tile():
    s1 = GettF64Spec(_rcr(K=128), tile_k=16, lds_stages=1)
    s2 = GettF64Spec(_rcr(K=128), tile_k=16, lds_stages=1, prefetch_depth=2)
    b1 = _kernel_text(s1, "gfx90a").split("define")[1]
    b2 = _kernel_text(s2, "gfx90a").split("define")[1]
    # the prologue loads two tiles; the loop carries the second in registers
    assert b2.count("buffer.load") > b1.count("buffer.load")
    assert "_d2_" in s2.kernel_name()


@pytest.mark.parametrize(
    "conj", [(False, False), (True, False), (False, True), (True, True)]
)
def test_four_complex_accumulators_have_no_negation_in_the_loop(conj):
    p = _rcr(M=128, N=64, K=32, complex_=True, conj_a=conj[0], conj_b=conj[1])

    def fnegs(**kw):
        spec = GettF64Spec(p, tile_m=64, tile_n=64, **kw)
        return _kernel_text(spec, "gfx950").split("define")[1].count("fneg double")

    # with four accumulators the negations live in the epilogue only, so they
    # do not grow with the k-steps of a K tile; with two they do
    assert fnegs(complex_accs=4, tile_k=8) == fnegs(complex_accs=4, tile_k=16)
    assert fnegs(complex_accs=2, tile_k=8) < fnegs(complex_accs=2, tile_k=16)
    assert "_a4_" in GettF64Spec(p, complex_accs=4).kernel_name()


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("nk", [2, 3, 7, 16])
def test_k_stagger_visits_every_k_tile_once(n, nk):
    # host model of the device remap: start ((bid % 8) * n) % nk, wrap
    for bid in range(16):
        start = ((bid % 8) * n) % nk
        order = [(kt + start) - (nk if kt + start >= nk else 0) for kt in range(nk)]
        assert sorted(order) == list(range(nk))
    spec = GettF64Spec(_rcr(K=16 * nk), tile_k=16, k_stagger=n)
    assert f"_k{n}_" in spec.kernel_name()
    body = _kernel_text(spec, "gfx942").split("define")[1]
    assert "select" in body


# ------------------------------------------- catalog, parameters, epilogue


def test_f64_archs_come_from_the_catalog():
    from rocke.core.arch import known_arches
    from rocke.instances.common.gett_f64 import f64_atom, f64_gett_archs

    assert F64_GETT_ARCHS == f64_gett_archs()
    assert {"gfx90a", "gfx942", "gfx950"} <= set(F64_GETT_ARCHS)
    for arch in known_arches():
        assert (arch in F64_GETT_ARCHS) == (f64_atom(arch) is not None)


def _params(spec, arch="gfx950"):
    head = _kernel_text(spec, arch).split("define")[1].split("{")[0]
    inner = head.split("(", 1)[1].rsplit(")", 1)[0]
    return {p.split()[-1]: p for p in inner.split(",")}


def test_c_and_d_may_alias_and_pointers_need_only_8_byte_alignment():
    params = _params(GettF64Spec(_rcr()))
    assert "noalias" in params["%A"] and "noalias" in params["%B"]
    assert "noalias" not in params["%C"] and "noalias" not in params["%D"]
    assert "readonly" in params["%C"]
    for name in ("%A", "%B", "%C", "%D"):
        assert "align 8" in params[name] and "align 16" not in params[name]


def test_epilogue_tests_alpha_zero_and_real_scalars():
    body = _kernel_text(GettF64Spec(_rcr()), "gfx950").split("define")[1]
    assert "fcmp oeq double %alpha," in body
    pc = _rcr(M=64, N=64, K=32, complex_=True)
    body = _kernel_text(GettF64Spec(pc, tile_m=64, tile_n=64), "gfx950")
    body = body.split("define")[1]
    for name in ("%alpha_re", "%alpha_im", "%beta_im"):
        assert f"fcmp oeq double {name}," in body


def test_complex_dtype_is_named_c128():
    pc = _rcr(M=64, N=64, K=32, complex_=True)
    assert "_c128_" in GettF64Spec(pc, tile_m=64, tile_n=64).kernel_name()
    assert "_f64_M" in GettF64Spec(_rcr()).kernel_name()


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


@pytest.mark.parametrize(
    "kw,syncs_after",
    [
        ({}, 0),
        ({"lds_stages": 1}, 0),
        ({"prefetch_depth": 2, "lds_stages": 1}, 0),
        ({"unroll2": True}, 1),
    ],
)
def test_the_last_k_tile_stores_nothing(kw, syncs_after):
    """The loop covers every K tile but the last, which runs after it with no
    LDS store and no barrier (unroll2: the last pair, whose second half stores
    nothing); the loop body has no branch."""
    spec = GettF64Spec(_rcr(K=96), tile_k=16, **kw)  # six K tiles
    kernel = build_gett_f64(spec, arch="gfx950")
    top = list(kernel.body.ops)
    last_loop = max(i for i, op in enumerate(top) if op.name == "scf.for")
    after = [op.name for op in top[last_loop + 1 :]]
    assert after.count("tile.sync") == syncs_after
    assert not [op for op in _walk(top) if op.name == "scf.if"]


def test_stagger_masks_tiles_past_the_end():
    """With k_stagger the prefetch of tile nk would wrap onto a real tile: the
    load also tests the unmapped tile index."""
    nk = 6
    stag = _kernel_text(GettF64Spec(_rcr(K=16 * nk), tile_k=16, k_stagger=1), "gfx950")
    plain = _kernel_text(GettF64Spec(_rcr(K=16 * nk), tile_k=16), "gfx950")
    assert stag.count("icmp slt i32") > plain.count("icmp slt i32")


# ------------------------------------------------------------- k_block


@pytest.mark.parametrize(
    "kw", [{}, {"unroll2": True}, {"prefetch_depth": 2, "lds_stages": 1}]
)
def test_k_block_adds_block_sums(kw):
    spec = GettF64Spec(_rcr(K=200), tile_k=8, k_block=4, **kw)
    assert "_b4_" in spec.kernel_name()
    body = _kernel_text(spec, "gfx950").split("define")[1]
    # one f64 vector add per accumulator: 4x4 atoms per wave
    assert body.count("fadd <4 x double>") == 16
    plain = GettF64Spec(_rcr(K=200), tile_k=8, **kw)
    assert "fadd <4 x double>" not in _kernel_text(plain, "gfx950")


def test_k_block_doubles_the_accumulator_budget():
    big = GettF64Spec(_rcr(), tile_m=256, tile_n=128, warps_m=2, warps_n=2)
    assert is_valid_spec(big, arch="gfx950")[0]
    ok, why = is_valid_spec(
        GettF64Spec(_rcr(), tile_m=256, tile_n=128, k_block=4), arch="gfx950"
    )
    assert not ok and "accumulator needs" in why


# ------------------------------------------------------------ runtime sizes


def test_runtime_template_is_canonical():
    from rocke.instances.common.gett_f64 import runtime_template

    a = _rcr(M=300, N=200, K=64)
    b = _rcr(M=1000, N=2, K=1024)
    c = _rcr(M=300, N=200, K=65)
    ta, tb, tc = (runtime_template(x) for x in (a, b, c))
    assert ta == tb and ta != tc  # K parity differs (A and B load along K)
    assert runtime_template(_rcr(M=301, N=199, K=64)) == ta  # M, N: no vector run
    assert runtime_template(ta) == ta
    multi = GettF64Problem(
        M=12,
        N=4,
        K=4,
        a_m=((3, 1), (4, 12)),
        a_k=((4, 3),),
        b_n=((4, 1),),
        b_k=((4, 4),),
        d_m=((12, 4),),
        d_n=((4, 1),),
    )
    assert runtime_template(multi) is None
    ok, why = is_valid_spec(GettF64Spec(multi, runtime_sizes=True), arch="gfx950")
    assert not ok and "one digit per index space" in why


def test_runtime_kernel_is_the_same_for_every_problem_of_a_layout():
    a = GettF64Spec(_rcr(M=300, N=200, K=64), runtime_sizes=True)
    b = GettF64Spec(_rcr(M=1000, N=2000, K=1024), runtime_sizes=True)
    assert a.kernel_name() == b.kernel_name() and "_rt_" in a.kernel_name()
    assert _kernel_text(a, "gfx942") == _kernel_text(b, "gfx942")
    names = [x["name"] for x in gett_f64_signature(a)]
    assert names[:6] == ["A", "B", "C", "D", "alpha", "beta"]
    assert names[6:10] == ["M", "N", "K", "L"]
    assert names[-1] == "c_sl"
    nb = GettF64Spec(_rcr(), bilinear=False, runtime_sizes=True)
    names = [x["name"] for x in gett_f64_signature(nb)]
    assert "c_sm" not in names and "c_sl" not in names


def test_runtime_args_grid_and_compatibility():
    from rocke.instances.common.gett_f64 import (
        gett_f64_runtime_args,
        runtime_compatible,
    )

    spec = GettF64Spec(_rcr(M=300, N=200, K=64), runtime_sizes=True)
    p = gemm_problem(
        500,
        400,
        96,
        a_strides=(96, 1),
        b_strides=(96, 1),
        d_strides=(400, 1),
        batch=3,
        batch_strides=(500 * 96, 400 * 96, 500 * 400),
    )
    assert runtime_compatible(spec, p)[0]
    args = gett_f64_runtime_args(spec, p)
    assert (args["M"], args["N"], args["K"], args["L"]) == (500, 400, 96, 3)
    assert (args["a_sm"], args["a_sk"], args["d_sm"], args["d_sl"]) == (
        96,
        1,
        400,
        200000,
    )
    assert gett_f64_grid(spec, p) == (4 * 4, 3, 1)
    with pytest.raises(ValueError):
        gett_f64_grid(spec)
    odd = gemm_problem(
        500, 400, 97, a_strides=(97, 1), b_strides=(97, 1), d_strides=(400, 1)
    )
    ok, why = runtime_compatible(spec, odd)
    assert not ok and "layout class" in why
    many = gemm_problem(
        8,
        8,
        8,
        a_strides=(8, 1),
        b_strides=(8, 1),
        d_strides=(8, 1),
        batch=70000,
        batch_strides=(64, 64, 64),
    )
    ok, why = runtime_compatible(
        GettF64Spec(_rcr(M=8, N=8, K=8), runtime_sizes=True), many
    )
    assert not ok and "grid limit" in why


# ------------------------------------------------------------ batch limit


def test_batch_beyond_the_grid_is_rejected_and_split():
    from rocke.instances.common.gett_f64 import MAX_GRID_Y

    p = gemm_problem(
        16,
        16,
        16,
        a_strides=(16, 1),
        b_strides=(16, 1),
        d_strides=(16, 1),
        batch=MAX_GRID_Y + 1,
        batch_strides=(256, 256, 256),
    )
    spec = GettF64Spec(p, tile_m=16, tile_n=16, warps_m=1, warps_n=1)
    ok, why = is_valid_spec(spec, arch="gfx950")
    assert not ok and "grid limit" in why
    with pytest.raises(ValueError):
        gett_f64_grid(spec)
    ext = {"m": 4, "n": 4, "k": 8, "l": 200003}
    ma, mb, md = ["m", "k", "l"], ["n", "k", "l"], ["m", "n", "l"]
    sa, sb, sd = (packed_strides(x, ext) for x in (ma, mb, md))
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext)
    assert all(ln.problem.L <= MAX_GRID_Y for ln in plan.launches)
    covered = set()
    for ln in plan.launches:
        first = ln.d_offset // sd["l"]
        assert ln.a_offset == first * sa["l"] and ln.b_offset == first * sb["l"]
        covered |= set(range(first, first + ln.problem.L))
    assert covered == set(range(ext["l"]))
    assert sum(ln.problem.L for ln in plan.launches) == ext["l"]
    # two batch modes: the product per launch fits
    ext2 = {"m": 4, "n": 4, "k": 8, "l": 300, "q": 1000}
    ma, mb, md = ["m", "k", "l", "q"], ["n", "k", "l", "q"], ["m", "n", "l", "q"]
    sa, sb, sd = (packed_strides(x, ext2) for x in (ma, mb, md))
    plan = plan_contraction(ma, sa, mb, sb, md, sd, ext2)
    assert all(ln.problem.L <= MAX_GRID_Y for ln in plan.launches)
    assert sum(ln.problem.L for ln in plan.launches) == 300 * 1000


# --------------------------------------------- byte-identity coverage


def test_parity_family_covers_every_tuned_configuration():
    import sys
    from pathlib import Path

    from rocke.instances.common.gett_f64 import TUNED_CONFIGS, arch_spec

    sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))
    import gett_f64_emit as E

    assert set(E._TUNED) == set(TUNED_CONFIGS)
    specs = E.all_specs()
    for j, key in enumerate(E._TUNED):
        spec, arch = specs[len(E._CONFIGS) + j]
        assert spec == arch_spec(E._tuned_problem(key), arch, **TUNED_CONFIGS[key])
    # the generated C cases are current
    c_path = Path(__file__).resolve().parent / "parity" / "gett_f64_emit.c"
    if not c_path.is_file():
        pytest.skip("parity/gett_f64_emit.c is not here (installed test tree)")
    c_src = c_path.read_text()
    assert E.c_cases() in c_src.replace("\r\n", "\n")


# ------------------------------------------------------------- split_k


def test_split_k_kernels_and_launch_shape():
    from rocke.instances.common.gett_f64 import (
        build_gett_f64_reduce,
        gett_f64_reduce_grid,
        gett_f64_reduce_signature,
        gett_f64_workspace_bytes,
    )

    p = _rcr(
        M=300, N=200, K=1024, batch=2, batch_strides=(300 * 1024, 200 * 1024, 300 * 200)
    )
    s = GettF64Spec(p, split_k=5)
    assert "_sk5_" in s.kernel_name()
    assert gett_f64_grid(s) == (3 * 2, 2, 5)
    assert gett_f64_workspace_bytes(s) == 5 * 2 * 300 * 200 * 8
    assert gett_f64_workspace_bytes(GettF64Spec(p)) == 0
    assert [a["name"] for a in gett_f64_signature(s)] == ["A", "B", "W"]
    assert [a["name"] for a in gett_f64_reduce_signature(s)] == [
        "W",
        "C",
        "D",
        "alpha",
        "beta",
    ]
    assert gett_f64_reduce_grid(s) == (-(-300 * 200 // 256), 2, 1)
    part = _kernel_text(s, "gfx950")
    assert "@llvm.amdgcn.workgroup.id.z" in part and "%C" not in part.split("{")[0]
    red = build_gett_f64_reduce(s, arch="gfx950")
    ll = _lower_kernel_to_llvm_python(red, arch="gfx950")
    body = ll.split("define")[1]
    assert body.count("fadd <1 x double>") == 4  # five partials, added in order
    assert "fcmp oeq double %beta" in body and "fcmp oeq double %alpha" in body
    with pytest.raises(ValueError):
        build_gett_f64_reduce(GettF64Spec(p), arch="gfx950")


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"split_k": 4, "unroll2": True}, "split_k does not combine"),
        ({"split_k": 4, "k_stagger": 1}, "split_k does not combine"),
        ({"split_k": 0}, "split_k must be in"),
        ({"split_k": 200}, "at least one K tile per split"),
    ],
)
def test_split_k_is_checked(kw, needle):
    ok, why = is_valid_spec(GettF64Spec(_rcr(K=1024), **kw), arch="gfx950")
    assert not ok and needle in why
    assert is_valid_spec(
        GettF64Spec(_rcr(K=1024), split_k=4, k_block=2), arch="gfx950"
    )[0]


# ------------------------------------------------- f32 / complex f32 words


def _rcr32(M=256, N=256, K=128, **kw):
    return _rcr(M, N, K, dtype="f32", **kw)


def test_f32_archs_come_from_the_catalog():
    from rocke.core.arch import known_arches
    from rocke.instances.common.gett_f64 import (
        F32_GETT_ARCHS,
        f32_atom,
        f32_gett_archs,
    )

    assert F32_GETT_ARCHS == f32_gett_archs()
    assert {"gfx942", "gfx950"} <= set(F32_GETT_ARCHS)
    for arch in known_arches():
        assert (arch in F32_GETT_ARCHS) == (f32_atom(arch) is not None)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_f32_kernel_structure(arch, flavor):
    tile_k = 16 if arch == "gfx950" else 8
    spec = GettF64Spec(_rcr32(K=100), tile_k=tile_k)
    ll = _kernel_text(spec, arch, flavor)
    body = ll.split("define")[1]
    # 4x4 f32 atoms per wave, tile_k / 4 k-steps per K tile; the loop body
    # and the last K tile after it; no f64 anywhere.
    assert body.count("@llvm.amdgcn.mfma.f32.16x16x4f32(") == 2 * 16 * (tile_k // 4)
    assert "double" not in body and "mfma.f64" not in body
    assert "fcmp oeq float %beta" in body and "fcmp oeq float %alpha," in body
    params = _params(spec, arch)
    for name in ("%A", "%B", "%C", "%D"):
        assert "align 4" in params[name]
    assert "float %alpha" in params["%alpha"]


def test_f32_complex_kernel_is_fused_4m():
    p = _rcr32(M=128, N=64, K=32, complex_=True, conj_a=True)
    spec = GettF64Spec(p, tile_m=64, tile_n=64)
    body = _kernel_text(spec, "gfx950").split("define")[1]
    per_tile = 2 * 2 * 4 * 4
    assert body.count("@llvm.amdgcn.mfma.f32.16x16x4f32(") in (per_tile, 2 * per_tile)
    assert "fneg float" in body and "double" not in body
    assert "_c64_" in spec.kernel_name()


def test_f32_names_and_signature():
    s32 = GettF64Spec(_rcr32())
    assert "_f32_M" in s32.kernel_name()
    assert s32.kernel_name() != GettF64Spec(_rcr()).kernel_name()
    sig = gett_f64_signature(s32)
    types = {a["name"]: a["type"] for a in sig}
    assert types["A"] == types["B"] == types["D"] == "ptr<f32, global>"
    assert types["alpha"] == types["beta"] == "f32"
    assert s32.problem.elem_bytes == 4
    assert _rcr32(complex_=True).elem_bytes == 8


def test_f32_loads_take_four_words_on_runs_of_four():
    for K, vec in ((128, 4), (130, 2), (129, 1)):
        p = _rcr32(K=K)
        s = GettF64Spec(p)
        plan = _load_plan(s, p.a_m, p.a_k, s.tile_m)
        assert plan.along_k and plan.vec == vec, (K, plan)
    pc = _rcr32(K=128, complex_=True)
    sc = GettF64Spec(pc, tile_m=64, tile_n=64)
    assert _load_plan(sc, pc.a_m, pc.a_k, sc.tile_m).vec == 1


def test_f32_runtime_template_keeps_runs_of_four():
    from rocke.instances.common.gett_f64 import runtime_template

    t = {K: runtime_template(_rcr32(K=K)) for K in (128, 130, 129, 64)}
    assert t[128] == t[64] and t[128].K == 4
    assert t[130].K == 2 and t[129].K == 3
    assert t[128].dtype == "f32"
    # f64 templates are unchanged (runs of four are not a class there)
    assert runtime_template(_rcr(K=128)).K == 2
    assert runtime_template(_rcr(K=128)) != runtime_template(_rcr32(K=128))


@pytest.mark.parametrize(
    "kw,needle",
    [
        ({"split_k": 2}, "split_k is not built for f32"),
        ({"acc_agpr": True}, "acc_agpr is not built for f32"),
        ({"k_inner": 2}, "k_inner is not built for f32"),
        ({"kmajor_lds": True}, "kmajor_lds is not built for f32"),
    ],
)
def test_f32_refuses_the_f64_only_levers(kw, needle):
    ok, why = is_valid_spec(GettF64Spec(_rcr32(K=1024), **kw), arch="gfx950")
    assert not ok and needle in why


def test_f32_needs_the_f32_atom():
    ok, why = is_valid_spec(GettF64Spec(_rcr32()), arch="gfx90a")
    assert not ok and "has no 16x16x4 f32 MMA atom" in why
    bad = _rcr32()
    from dataclasses import replace as _rp

    ok, why = _rp(bad, dtype="f16").check()
    assert not ok and "dtype must be one of" in why


def test_f32_lds_and_registers_halve():
    # a tile whose f64 form does not fit gfx942's LDS fits with f32 words
    assert not is_valid_spec(GettF64Spec(_rcr(), tile_k=32), arch="gfx942")[0]
    assert is_valid_spec(GettF64Spec(_rcr32(), tile_k=32), arch="gfx942")[0]
    # 256 x 256 tile: the f64 accumulator does not fit, the f32 one does
    assert not is_valid_spec(
        GettF64Spec(_rcr(), tile_m=256, tile_n=256), arch="gfx950"
    )[0]
    assert is_valid_spec(GettF64Spec(_rcr32(), tile_m=256, tile_n=256), arch="gfx950")[
        0
    ]


# ------------------------------------------------- tuned f32 classes

_F32_KNOBS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warps_m",
    "warps_n",
    "group_m",
    "xcds",
    "waves_per_eu",
    "lds_stages",
    "prefetch_depth",
    "unroll2",
    "sched_fence",
    "k_stagger",
    "lds_pad",
    "lds_swizzle",
    "lds_swizzle_mode",
)


def test_f32_tuned_classes_are_explicit_valid_pack_templates():
    """Every TUNED_CONFIGS_F32 row sets every knob the C table carries, and is a
    valid pack kernel (hipTensor's packed layout template, runtime sizes) on
    its arch for every D layout, scale and bilinear."""
    from dataclasses import replace

    from rocke.instances.common.gett_f64 import (
        TUNED_CONFIGS_F32,
        arch_spec,
        runtime_template,
    )

    assert TUNED_CONFIGS_F32
    for (arch, kind, cls), cfg in TUNED_CONFIGS_F32.items():
        assert set(cfg) == set(_F32_KNOBS), (arch, kind, cls)
        cplx = kind == "complex"
        for dclass, d_strides in (("m", (1, 64)), ("n", (64, 1)), ("x", (2, 128))):
            prob = gemm_problem(
                64,
                64,
                64,
                a_strides=(64, 1),
                b_strides=(64, 1),
                d_strides=d_strides,
                batch=2,
                batch_strides=(64 * 64, 64 * 64, 4 * 64 * 64),
                complex_=cplx,
                dtype="f32",
            )
            for bilinear in (False, True):
                spec = arch_spec(
                    prob, arch, bilinear=bilinear, runtime_sizes=True, **cfg
                )
                spec = replace(spec, problem=runtime_template(prob))
                ok, why = is_valid_spec(spec, arch=arch)
                assert ok, (arch, kind, cls, dclass, bilinear, why)


def test_tuned_configs_by_dtype():
    """f32 problems take the f32 rows where (arch, kind) has them; f64 problems
    and (arch, kind) without f32 rows take TUNED_CONFIGS."""
    from rocke.instances.common.gett_f64 import (
        TUNED_CONFIGS,
        TUNED_CONFIGS_F32,
        tuned_configs,
    )

    archs = {a for (a, _k, _c) in TUNED_CONFIGS} | {
        a for (a, _k, _c) in TUNED_CONFIGS_F32
    }
    for arch in sorted(archs):
        for kind in ("real", "complex"):
            f64 = {
                c: v
                for (a, k, c), v in TUNED_CONFIGS.items()
                if a == arch and k == kind
            }
            f32 = {
                c: v
                for (a, k, c), v in TUNED_CONFIGS_F32.items()
                if a == arch and k == kind
            }
            assert tuned_configs(arch, kind) == f64
            assert tuned_configs(arch, kind, "f64") == f64
            assert tuned_configs(arch, kind, "f32") == (f32 or f64)


_F32_CLASS_CASES = (
    (False, 100, 90, 64, 1, "tiny"),
    (False, 1024, 1024, 64, 1, "small"),
    (False, 1536, 1440, 576, 1, "medium"),
    (False, 2560, 2560, 128, 1, "large"),
    (False, 4608, 4608, 288, 1, "xlarge"),
    (False, 8192, 8192, 256, 1, "huge"),
    (False, 4096, 6144, 100, 1, "xlarge"),
    (False, 128, 8000, 4100, 1, "small"),
    (True, 200, 180, 100, 1, "small"),
    (True, 600, 900, 128, 1, "medium"),
    (True, 256, 512, 1000, 1, "longk"),
    (True, 2048, 2048, 64, 1, "medium"),
    (True, 8192, 8192, 200, 1, "large"),
    (True, 520, 512, 256, 1, "small"),
    (True, 1000, 300, 257, 1, "longk"),
)


@pytest.mark.parametrize("cplx,M,N,K,L,cls", _F32_CLASS_CASES)
def test_gett_f32_shape_class_gfx950(cplx, M, N, K, L, cls):
    """gfx950 (256 CUs) classes of packed problems (K padded the way hipTensor's
    planner pads it); hipTensor's RockeGett32Class test expects the same."""
    from rocke.instances.common.gett_f64 import gett_f32_shape_class

    Kp = -(-K // (2 if cplx else 4)) * (2 if cplx else 4)
    p = gemm_problem(
        M,
        N,
        Kp,
        a_strides=(Kp, 1),
        b_strides=(Kp, 1),
        d_strides=(N, 1),
        batch=L,
        batch_strides=(M * Kp, N * Kp, M * N),
        complex_=cplx,
        dtype="f32",
    )
    assert gett_f32_shape_class(p, "gfx950", 256) == cls


def test_gett_f32_shape_class_boundaries_gfx950():
    """The real rule's tile-count edges (128x128 tiles per CU 1/8, 1/2, 1, 4, 8)
    and the complex rule's (64x64 tiles per CU 4 and 32, K 256, min(M, N) 512)."""
    from rocke.instances.common.gett_f64 import gett_f32_shape_class

    def real(tiles_m, tiles_n):
        p = gemm_problem(
            128 * tiles_m,
            128 * tiles_n,
            64,
            a_strides=(64, 1),
            b_strides=(64, 1),
            d_strides=(128 * tiles_n, 1),
            dtype="f32",
        )
        return gett_f32_shape_class(p, "gfx950", 256)

    assert [real(1, 32), real(1, 33)] == ["tiny", "small"]
    assert [real(2, 64), real(2, 65)] == ["small", "medium"]
    assert [real(4, 64), real(4, 65)] == ["medium", "large"]
    assert [real(16, 64), real(16, 65)] == ["large", "xlarge"]
    assert [real(32, 64), real(32, 65)] == ["xlarge", "huge"]

    def cx(M, N, K):
        p = gemm_problem(
            M,
            N,
            K,
            a_strides=(K, 1),
            b_strides=(K, 1),
            d_strides=(N, 1),
            complex_=True,
            dtype="f32",
        )
        return gett_f32_shape_class(p, "gfx950", 256)

    assert [cx(512, 1024, 256), cx(576, 1024, 256), cx(512, 1024, 258)] == [
        "small",
        "medium",
        "longk",
    ]
    assert [cx(64 * 32, 64 * 32, 64), cx(64 * 32, 64 * 33, 64)] == ["medium", "medium"]
    assert [cx(64 * 64, 64 * 128, 64), cx(64 * 64, 64 * 129, 64)] == ["medium", "large"]


def test_gett_f32_shape_class_boundaries_gfx942():
    """gfx942 (304 CUs): real 128x128 tiles per CU 1/8, 1/4, 8 with min(M, N)
    4096 and K 1024 above; complex 64x64 tiles per CU 1/2, 1, 64."""
    from rocke.instances.common.gett_f64 import gett_f32_shape_class

    def real(M, N, K=64):
        p = gemm_problem(
            M, N, K, a_strides=(K, 1), b_strides=(K, 1), d_strides=(N, 1), dtype="f32"
        )
        return gett_f32_shape_class(p, "gfx942", 304)

    assert [real(128, 128 * 38), real(128, 128 * 39)] == ["tiny", "small"]
    assert [real(256, 128 * 38), real(256, 128 * 39)] == ["small", "medium"]
    assert [real(4096, 128 * 73), real(4224, 128 * 73)] == ["medium", "large"]
    assert [real(4224, 128 * 73), real(4224, 128 * 74)] == ["large", "xlarge"]
    assert [real(4224, 128 * 74, 1024), real(4224, 128 * 74, 1028)] == [
        "xlarge",
        "huge",
    ]

    def cx(M, N):
        p = gemm_problem(
            M,
            N,
            64,
            a_strides=(64, 1),
            b_strides=(64, 1),
            d_strides=(N, 1),
            complex_=True,
            dtype="f32",
        )
        return gett_f32_shape_class(p, "gfx942", 304)

    assert [cx(64, 64 * 152), cx(64, 64 * 153)] == ["small", "compact"]
    assert [cx(128, 64 * 152), cx(128, 64 * 153)] == ["compact", "medium"]
    assert [cx(64 * 152, 64 * 128), cx(64 * 152, 64 * 129)] == ["medium", "large"]


def test_tuned_spec_takes_the_f32_class():
    """tuned_spec of an f32 problem is its f32 class's configuration; the same
    problem with f64 words takes the f64 class."""
    from rocke.instances.common.gett_f64 import (
        TUNED_CONFIGS,
        TUNED_CONFIGS_F32,
        arch_spec,
        gett_f64_shape_class,
        tuned_spec,
    )

    for dtype in ("f32", "f64"):
        p = gemm_problem(
            1536,
            1440,
            576,
            a_strides=(576, 1),
            b_strides=(576, 1),
            d_strides=(1440, 1),
            dtype=dtype,
        )
        spec = tuned_spec(p, "gfx950", num_cus=256)
        if dtype == "f32":
            cfg = TUNED_CONFIGS_F32[("gfx950", "real", "medium")]
        else:
            cfg = TUNED_CONFIGS[
                ("gfx950", "real", gett_f64_shape_class(p, "gfx950", 256))
            ]
        assert spec == arch_spec(p, "gfx950", bilinear=True, **cfg)
