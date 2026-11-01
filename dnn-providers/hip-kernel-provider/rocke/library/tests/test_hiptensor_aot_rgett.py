# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numerics of the ring GETT classes (``kernels.common.hiptensor_aot.rgett``): rocKE's native
tensor-contraction body with its index maps as kernel arguments (the kernel-pack form) or as
constants (the JIT form), on gfx950.

Every case is checked against an independent float64 reference computed with torch on the CPU
from the problem's own strided views: integer-valued operands make every product and sum exact
in the f32 accumulator, so the result is compared bit for bit with the reference rounded once
to the output type. The JIT form must give the same bits as the class kernel. Arguments come
from :func:`kernels.common.hiptensor_aot.gett_host.rgett_plan`, the Python statement of
hipTensor's C++ planner; the CPU tests check its plans against rocKE's own GETT planner.
"""

from __future__ import annotations

import math
import random

import pytest

from kernels.common.hiptensor_aot.gett_host import contraction_groups, rgett_plan
from kernels.common.hiptensor_aot.rgett import (
    RgettSpec,
    build_rgett,
    geometry_names,
    rgett_classes,
)


def test_classes_build():
    names = set()
    for sp in rgett_classes(("f16",))[:4]:
        k = build_rgett(sp, arch="gfx950")
        params = [p.name for p in k.params]
        assert params[:6] == ["A", "B", "C", "M", "N", "K"]
        assert set(geometry_names()) <= set(params)
        names.add(k.name)
    assert len(names) == 4


def test_class_set():
    """Every lane / D width in each digit-slot bucket, the wide classes of packed operands, C
    prefetch on every bilinear class and no bilinear 16-byte-store class of every slot.
    """
    from kernels.common.hiptensor_aot.rgett import BILINEAR_C_PREFETCH, SLOT_BUCKETS

    cls = rgett_classes()
    assert len({c.kernel_name() for c in cls}) == len(cls)
    for c in cls:
        assert c.slots in SLOT_BUCKETS
        assert c.c_prefetch == (BILINEAR_C_PREFETCH if c.bilinear else 0)
        assert not (c.bilinear and c.epi_pair and c.slots == 0)
        assert not c.wide_d or c.vec_a == c.vec_b == 8
    narrow = {
        (c.dtype, c.bilinear, c.vec_a, c.vec_b, c.vec_d, c.epi_pair, c.slots)
        for c in cls
        if not c.wide_d
    }
    # 2 types x 2 variants x 4 lane pairs x 4 D widths x 3 buckets, less the 8 bilinear pair
    # classes of every slot
    assert len(narrow) == 2 * 2 * 4 * 4 * 3 - 8


def test_slots_check():
    with pytest.raises(ValueError):
        RgettSpec("f16", False, 8, 8, 4, True, slots=7).check()
    with pytest.raises(ValueError):
        RgettSpec("f16", False, 8, 8, 4, True, c_prefetch=2).check()


def test_jit_form_has_no_geometry_arguments():
    sp = RgettSpec("bf16", True, 8, 2, 2, False)
    geom = {n: 1 for n in geometry_names()}
    geom.update(kps=-1, kpl=0)
    k = build_rgett(sp.with_geom(geom), arch="gfx950")
    params = [p.name for p in k.params]
    assert params == ["A", "B", "C", "M", "N", "K", "alpha", "Cin", "beta"]


def _problem(rng, rank_m, rank_n, rank_k, run=8, kmul=32, cap=1 << 17):
    """Mode lists (row-major: last mode fastest) of a problem the ring body serves: A's and B's
    fastest modes are free modes with runs of ``run``, the K extents multiply into ``kmul``
    multiples."""
    ms = list(range(rank_m + rank_n + rank_k))
    mm, nn, kk = ms[:rank_m], ms[rank_m : rank_m + rank_n], ms[rank_m + rank_n :]
    ext = {m: rng.choice([2, 3, 4, 6, 8, 12, 16]) for m in ms}
    ext[mm[0]] = run * rng.choice([1, 2, 3, 8, 16])
    ext[nn[0]] = run * rng.choice([1, 2, 5, 8, 16])
    for m in kk:
        ext[m] = rng.choice([4, 8, 16, 32])
    while math.prod(ext[m] for m in kk) > 1024:
        big = max(kk, key=lambda x: ext[x])
        ext[big] //= 2
    if math.prod(ext[m] for m in kk) % kmul:
        # a K space the ring body splits at its K tile (padding is tested separately)
        ext[kk[0]] = kmul
        for m in kk[1:]:
            ext[m] = min(ext[m], 8)
    while math.prod(ext[m] for m in mm + nn) > cap:
        big = max(mm[1:] + nn[1:] or mm + nn, key=lambda x: ext[x])
        if ext[big] <= 2:
            break
        ext[big] //= 2
    A = [m for m in mm[1:] + kk] + [mm[0]]
    B = [m for m in nn[1:] + kk] + [nn[0]]
    rng.shuffle(A)
    rng.shuffle(B)
    A = [m for m in A if m != mm[0]] + [mm[0]]
    B = [m for m in B if m != nn[0]] + [nn[0]]
    D = mm + nn
    rng.shuffle(D)
    return ext, {"A": A, "B": B, "D": D}


def _packed(modes, ext):
    st, s = {}, 1
    for m in reversed(modes):
        st[m] = s
        s *= ext[m]
    return st


def test_plan_matches_rocke_gett_planner():
    """The class geometry walks the same offsets as rocKE's compile-time GETT plan."""
    from rocke.instances.common.tensor.gett_plan import plan_gett

    rng = random.Random(3)
    for _ in range(20):
        ext, md = _problem(rng, 2, 2, 2)
        strides = {t: [_packed(md[t], ext)[m] for m in md[t]] for t in "ABD"}
        groups = contraction_groups(md, ext, strides)
        p = rgett_plan(groups, 2, False)
        names = {m: f"x{m}" for m in ext}
        ref = plan_gett(
            [names[m] for m in md["A"]],
            [names[m] for m in md["B"]],
            [names[m] for m in md["D"]],
            {names[m]: e for m, e in ext.items()},
            tile_k=32,
        )
        assert p["swapped"] == ref.swapped
        assert (p["vals"]["M"], p["vals"]["N"], p["vals"]["K"]) == (ref.M, ref.N, ref.K)
        assert p["vec_d"] == ref.vec_d and p["vec_d8"] == ref.vec_d8
        assert (p["vec_a"], p["vec_b"]) == (ref.ops["A"].vec, ref.ops["B"].vec)


# ------------------------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(
    0
).gcnArchName.startswith("gfx950")
gpu = pytest.mark.skipif(not GPU, reason="needs a gfx950 GPU")
TD = {"f16": torch.float16, "bf16": torch.bfloat16}
_CACHE = {}


def _run(kernel, vals, grid):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if kernel.name not in _CACHE:
        art = compile_kernel(
            kernel, arch="gfx950", backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _CACHE[kernel.name] = KernelLauncher(
            hsaco=art.hsaco, kernel_name=kernel.name, signature=sig
        )
    _CACHE[kernel.name](
        {p.name: vals[p.name] for p in kernel.params},
        config=LaunchConfig(grid=grid, block=(256, 1, 1)),
    )
    torch.cuda.synchronize()


def _letters(modes):
    return "".join(chr(ord("a") + m) for m in modes)


def _make(md, ext, dt, rng, pad_inner=0):
    """Device tensor stored in the order of ``md`` (row-major: last mode fastest), with
    ``pad_inner`` extra elements after each run of the fastest mode (kept a multiple of 8 so
    that the operand stays DTL-aligned)."""
    shape = [ext[m] for m in md]
    stor = list(shape)
    stor[-1] += pad_inner
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    base = torch.randint(-3, 4, stor, generator=g).to(TD[dt]).cuda()
    view = base[..., : shape[-1]] if pad_inner else base
    return view


def run(
    rng,
    sp_dtype,
    ext,
    md,
    bilinear,
    alpha,
    beta,
    pad=0,
    jit=False,
    wide=False,
    slots=0,
    cpf=0,
):
    t = {
        x: _make(
            md[x if x != "C" else "D"], ext, sp_dtype, rng, pad if x in "AB" else 0
        )
        for x in "ABCD"
    }
    strides = {x: list(t[x].stride()) for x in "ABD"}
    groups = contraction_groups(md, ext, strides)
    p = rgett_plan(groups, 2, bilinear)
    spec = RgettSpec(
        sp_dtype,
        bilinear,
        p["vec_a"],
        p["vec_b"],
        p["vec_d"],
        p["vec_d8"] and p["vec_d"] == 4,
        wide_d=wide,
        slots=slots,
        c_prefetch=cpf if bilinear else 0,
    )
    assert not slots or p["digits"] <= slots
    geom = {n: p["vals"][n] for n in geometry_names()} if jit else None
    k = build_rgett(spec.with_geom(geom), arch="gfx950")
    vals = dict(p["vals"])
    ka, kb = ("B", "A") if p["swapped"] else ("A", "B")
    d0 = t["D"].clone()
    vals.update(A=t[ka], B=t[kb], C=t["D"], alpha=alpha)
    if bilinear:
        vals.update(Cin=t["C"], beta=beta)
    _run(k, vals, p["grid"])
    eq = f"{_letters(md['A'])},{_letters(md['B'])}->{_letters(md['D'])}"
    ref = alpha * torch.einsum(eq, t["A"].cpu().double(), t["B"].cpu().double())
    if bilinear:
        ref = ref + beta * t["C"].cpu().double()
    want = ref.to(torch.float32).to(TD[sp_dtype]).double()
    got = t["D"].cpu().double()
    assert torch.equal(got, want), (
        (got - want).abs().max(),
        p["vec_a"],
        p["vec_b"],
        p["vec_d"],
    )
    _ = d0
    return got, spec


CASES = []
_rng = random.Random(77)
for dt in ("f16", "bf16"):
    for rm, rn, rk in ((1, 1, 1), (2, 1, 2), (2, 2, 2), (3, 2, 3), (2, 3, 1)):
        for run_len in (8, 2):
            CASES.append(
                (dt, rm, rn, rk, run_len, _rng.random() < 0.5, _rng.randrange(1 << 20))
            )


@gpu
@pytest.mark.parametrize("dt,rm,rn,rk,run_len,bilinear,seed", CASES)
def test_rgett_exact(dt, rm, rn, rk, run_len, bilinear, seed):
    rng = random.Random(seed)
    ext, md = _problem(rng, rm, rn, rk, run=run_len)
    run(
        rng,
        dt,
        ext,
        md,
        bilinear,
        rng.choice([1.0, -2.0, 0.5]),
        rng.choice([1.0, -0.5]),
    )


@gpu
@pytest.mark.parametrize("dt", ["f16", "bf16"])
def test_rgett_jit_form_matches_class_kernel(dt):
    rng = random.Random(5)
    ext, md = _problem(rng, 2, 2, 2)
    seed = rng.randrange(1 << 20)
    got_aot, _ = run(random.Random(seed), dt, ext, md, True, 1.5, -0.5)
    got_jit, _ = run(random.Random(seed), dt, ext, md, True, 1.5, -0.5, jit=True)
    assert torch.equal(got_aot, got_jit)


@gpu
@pytest.mark.parametrize("dt", ["f16", "bf16"])
def test_rgett_wide_d_classes(dt):
    """The classes for D past 2^31 bytes (the summed offset masked) on problems with ragged
    tiles: the same bits as the reference."""
    rng = random.Random(21)
    for bil in (False, True):
        ext, md = _problem(rng, 2, 2, 2)
        run(rng, dt, ext, md, bil, 1.0, -1.0, wide=True)


@gpu
def test_rgett_padded_k_and_ragged_tiles():
    rng = random.Random(9)
    # K extents 3 x 5 x 4: no exact split at 32, one in-tile mode padded
    ext = {0: 8 * 37, 1: 5, 2: 8 * 29, 3: 3, 4: 5, 5: 4}
    md = {"A": [4, 3, 1, 5, 0], "B": [5, 3, 4, 2], "D": [1, 2, 0]}
    md["A"] = [m for m in md["A"] if m != 0] + [0]
    for bil in (False, True):
        run(rng, "f16", ext, md, bil, 1.0, 1.0)


@gpu
@pytest.mark.parametrize("dt", ["f16", "bf16"])
@pytest.mark.parametrize("ranks", [(1, 1, 1), (2, 2, 2), (3, 2, 3)])
def test_rgett_slot_buckets(dt, ranks):
    """The digit-slot bucket classes (the smallest bucket that holds the problem's digits, and
    every larger one), with and without C prefetch: the same bits as the reference."""
    from kernels.common.hiptensor_aot.rgett import SLOT_BUCKETS

    rng = random.Random(31 + sum(ranks))
    for bil in (False, True):
        ext, md = _problem(rng, *ranks)
        seed = rng.randrange(1 << 20)
        ref = None
        for n in SLOT_BUCKETS:
            for cpf in (0, 2) if bil else (0,):
                strides = {
                    x: [_packed(md[x], ext)[m] for m in md[x]] for x in ("A", "B", "D")
                }
                digits = rgett_plan(contraction_groups(md, ext, strides), 2, bil)[
                    "digits"
                ]
                if n and digits > n:
                    continue
                got, _sp = run(
                    random.Random(seed), dt, ext, md, bil, 1.5, -0.5, slots=n, cpf=cpf
                )
                if ref is None:
                    ref = got
                assert torch.equal(got, ref)
