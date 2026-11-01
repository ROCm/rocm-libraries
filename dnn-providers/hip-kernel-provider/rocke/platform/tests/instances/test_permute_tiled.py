# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled N-D permute: canonicalisation, planner, validity, IR shape, device numerics.

The CPU tests check the host-side algebra (axis merging against numpy, the
planner's tile choice, the validity refusals, the swizzle bijection and the
static bank model) and that the kernel lowers to the intended instruction mix
(16-byte global and LDS accesses, ``v_perm_b32`` for the 16-bit in-register
transpose). The device tests run every dtype and stage on random bit patterns
(NaN payloads included; a permute must move bits exactly), ragged thread
counts, a persistent tile loop with a remainder, and one tensor past 2^31
elements (64-bit tile bases).
"""

from __future__ import annotations

import dataclasses
import itertools
import random

import numpy as np
import pytest

from rocke.instances.common import permute_tiled as PT

DOC_ROWS = [
    ((8192, 8192), (1, 0)),
    ((8192, 8192), (0, 1)),
    ((512, 1024, 1024), (2, 0, 1)),
    ((512, 1024, 1024), (1, 0, 2)),
    ((64, 256, 112, 112), (0, 2, 3, 1)),
    ((64, 256, 112, 112), (2, 1, 0, 3)),
    ((32, 64, 64, 64, 32), (0, 4, 1, 3, 2)),
    ((32, 64, 64, 64, 32), (3, 2, 1, 0, 4)),
    ((16, 32, 32, 32, 16, 16), (0, 3, 1, 5, 2, 4)),
    ((16, 32, 32, 32, 16, 16), (4, 3, 2, 1, 0, 5)),
]
DTYPES = ("f16", "bf16", "f32", "f64")


# ---------------------------------------------------------------- canonicalisation


def _np_perm(shape, perm):
    x = np.arange(int(np.prod(shape))).reshape(shape)
    return np.ascontiguousarray(np.transpose(x, perm)).ravel()


@pytest.mark.parametrize("seed", range(40))
def test_canonicalize_moves_every_element_like_numpy(seed):
    rng = random.Random(seed)
    r = rng.randint(1, 6)
    shape = tuple(rng.choice((1, 2, 3, 4)) for _ in range(r))
    perm = list(range(r))
    rng.shuffle(perm)
    cs, cp = PT.canonicalize_permute(shape, perm)
    assert int(np.prod(cs)) == int(np.prod(shape))
    # adjacent output axes are never adjacent-in-order input axes after merging
    for d in range(len(cp) - 1):
        assert cp[d + 1] != cp[d] + 1
    assert all(e > 1 for e in cs) or cs == (1,)
    np.testing.assert_array_equal(_np_perm(shape, perm), _np_perm(cs, cp))


def test_canonicalize_examples():
    assert PT.canonicalize_permute((512, 1024, 1024), (2, 0, 1)) == (
        (524288, 1024),
        (1, 0),
    )
    assert PT.canonicalize_permute((64, 256, 112, 112), (0, 2, 3, 1)) == (
        (64, 256, 12544),
        (0, 2, 1),
    )
    assert PT.canonicalize_permute((8192, 8192), (0, 1)) == ((67108864,), (0,))
    assert PT.canonicalize_permute((1, 1), (1, 0)) == ((1,), (0,))
    with pytest.raises(ValueError):
        PT.canonicalize_permute((2, 3), (0, 0))


# ---------------------------------------------------------------- planner / validity


@pytest.mark.parametrize("shape,perm", DOC_ROWS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("arch", ("gfx950", "gfx942", "gfx1151"))
def test_plan_is_valid_and_vectorised_on_doc_rows(shape, perm, dtype, arch):
    sp = PT.plan_permute_tiled(shape, perm, dtype, arch)
    ok, why = PT.is_valid_spec(sp, arch)
    assert ok, why
    # every listed problem gets full 16-byte accesses on both sides
    assert sp.vec * sp.elem_bytes == 16
    from rocke.core.arch import ArchTarget

    assert ArchTarget.from_gfx(arch).fits_lds(sp.tile_elems * sp.elem_bytes)


@pytest.mark.parametrize("dtype", ("f16", "f32"))
def test_auto_stage_uses_gather_for_odd_extents(dtype):
    # innermost extents 10 / 6: the transpose path is limited to 4- or 8-byte
    # accesses, the gather stage moves whole 16-byte vectors along the runs
    sp = PT.plan_permute_tiled(
        (8, 10, 6, 10), (0, 2, 1, 3), dtype, "gfx950", run_bytes=512
    )
    assert sp.stage == "gather"
    assert sp.vec * sp.elem_bytes == 16
    assert sp.run_in % sp.vec == 0 and sp.run_out % sp.vec == 0
    lds = PT.plan_permute_tiled(
        (8, 10, 6, 10), (0, 2, 1, 3), dtype, "gfx950", stage="lds"
    )
    assert lds.vec * lds.elem_bytes < 16


def test_planner_defaults_and_overrides():
    sp = PT.plan_permute_tiled((8192, 8192), (1, 0), "f16", "gfx950", use_tuned=False)
    assert sp.nt == "both" and sp.grid_order in ("in", "out", "both")
    kn = PT.plan_permute_tiled((8192, 8192), (1, 0), "f16", "gfx950", run_bytes=512)
    assert (
        kn.grid_order == PT.DEFAULT_GRID_ORDER
    )  # knob-driven plans keep the fixed defaults
    assert PT.plan_permute_tiled((8192, 8192), (1, 0), "f16", "gfx1151").nt == "none"
    o = PT.plan_permute_tiled(
        (8192, 8192), (1, 0), "f16", "gfx950", nt="none", grid_order="in"
    )
    assert (o.nt, o.grid_order) == ("none", "in")


def test_tuned_table_entries_are_valid():
    for arch, table in PT.TUNED_CONFIGS.items():
        for (es, shape, perm), knobs in table.items():
            dt = {2: "f16", 4: "f32", 8: "f64"}[es]
            sp = PT.tuned_spec(shape, perm, dt, arch)
            assert sp is not None, (arch, shape, perm, knobs)
            assert PT.plan_permute_tiled(shape, perm, dt, arch) == sp


def test_default_tile_always_divides_the_extents():
    # an innermost-axis trim used to halve 5 -> 2 (f64, 2048-byte runs, 4 KiB tiles): an invalid tile
    shape, perm = PT.canonicalize_permute((5, 8, 1000, 1, 1, 1), (4, 2, 1, 3, 0, 5))
    t = PT.default_tile(shape, perm, "f64", 2048, 4096)
    assert all(shape[a] % t[a] == 0 for a in range(len(shape))), t
    PT.plan_permute_tiled(
        (5, 8, 1000, 1, 1, 1),
        (4, 2, 1, 3, 0, 5),
        "f64",
        "gfx950",
        run_bytes=2048,
        max_tile_bytes=4096,
    )
    rng = random.Random(5)
    ext = (1, 2, 3, 5, 6, 7, 9, 10, 12, 15, 24, 30, 56, 100, 112, 224, 1000)
    for _ in range(400):
        r = rng.randint(1, 6)
        sh = [rng.choice(ext) for _ in range(r)]
        pm = list(range(r))
        rng.shuffle(pm)
        cs, cp = PT.canonicalize_permute(sh, pm)
        for dt in ("f16", "f32", "f64"):
            for rb, mt in ((64, 4096), (2048, 4096), (512, 32768)):
                t = PT.default_tile(cs, cp, dt, rb, mt)
                assert all(cs[a] % t[a] == 0 for a in range(len(cs))), (
                    cs,
                    cp,
                    dt,
                    rb,
                    mt,
                    t,
                )


def test_validity_refusals():
    base = PT.PermuteTiledSpec(
        x_shape=(256, 512), perm=(1, 0), tile=(64, 64), dtype="f16", block_size=64
    )
    assert PT.is_valid_spec(base)[0]
    bad = {
        "tile does not divide": dataclasses.replace(base, tile=(48, 64)),
        "bad perm": dataclasses.replace(base, perm=(0, 0)),
        "dtype": dataclasses.replace(base, dtype="f8"),
        "block": dataclasses.replace(base, block_size=96),
        "regs": dataclasses.replace(base, tile=(256, 512), block_size=64, stage="reg"),
        "lds": dataclasses.replace(
            base, x_shape=(512, 1024), tile=(512, 512), block_size=1024
        ),
        "swizzle k": dataclasses.replace(base, swizzle_k=3, swizzle_shift=2),
        "swizzle shift": dataclasses.replace(base, swizzle_k=16, swizzle_shift=2),
        "gather swizzle": dataclasses.replace(
            base, stage="gather", swizzle_k=4, swizzle_shift=4
        ),
        "stage": dataclasses.replace(base, stage="dma"),
        "nt arch": dataclasses.replace(base, nt="both"),
        "tiles_per_cta": dataclasses.replace(base, tiles_per_cta=0),
    }
    for why, sp in bad.items():
        arch = "gfx1151" if why == "nt arch" else "gfx950"
        assert not PT.is_valid_spec(sp, arch)[0], why
    # a tile whose in-tile offsets pass 2^31 elements is refused
    big = PT.PermuteTiledSpec(
        x_shape=(2, 2**31), perm=(1, 0), tile=(2, 8), dtype="f16", block_size=64
    )
    assert not PT.is_valid_spec(big)[0]


@pytest.mark.parametrize("shape,perm", DOC_ROWS)
def test_swizzle_is_a_bijection(shape, perm):
    sp = PT.plan_permute_tiled(shape, perm, "f16", "gfx950")
    if sp.swizzle_k <= 1:
        return
    chunks = sp.tile_elems // sp.vec
    k, sh = sp.swizzle_k, sp.swizzle_shift
    img = {c ^ ((c >> sh) & (k - 1)) for c in range(chunks)}
    assert img == set(range(chunks))


def test_bank_model_prefers_swizzle_on_square_transpose():
    sp = PT.PermuteTiledSpec(
        x_shape=(8192, 8192), perm=(1, 0), tile=(64, 64), dtype="f16", block_size=64
    )
    plain = PT.lds_bank_cost(sp)
    k, sh = PT.best_swizzle(sp)
    assert k > 1
    assert (
        PT.lds_bank_cost(dataclasses.replace(sp, swizzle_k=k, swizzle_shift=sh)) < plain
    )


# ---------------------------------------------------------------- IR / ISA


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stage", ("lds", "reg"))
def test_build_ir(dtype, stage):
    sp = PT.plan_permute_tiled(
        (64, 128, 96), (2, 0, 1), dtype, "gfx950", stage=stage, tiles_per_cta=3
    )
    k = PT.build_permute_tiled(sp, "gfx950")
    assert k.name == sp.kernel_name()
    names = [p.name for p in k.params]
    assert names == ["X", "Y"]


def _objdump():
    import os
    import shutil

    for c in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if c and os.path.isfile(c):
            return c
    return None


def _isa(sp, arch="gfx950"):
    import subprocess
    import tempfile
    from pathlib import Path

    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    hsaco = compile_kernel(PT.build_permute_tiled(sp, arch), arch=arch).hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "perm.hsaco"
        path.write_bytes(hsaco)
        return subprocess.run(
            [objdump, "-d", f"--mcpu={arch}", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def test_isa_uses_16_byte_accesses_and_byte_permutes():
    sp = PT.plan_permute_tiled(
        (8192, 8192),
        (1, 0),
        "f16",
        "gfx950",
        run_bytes=128,
        max_tile_bytes=8192,
        block_size=64,
    )
    isa = _isa(sp)
    assert isa.count("global_load_dwordx4") == 8
    assert isa.count("global_store_dwordx4") == 8
    assert isa.count("ds_write_b128") == 8
    assert isa.count("ds_read_b128") == 8
    assert isa.count("v_perm_b32") >= 32


# ---------------------------------------------------------------- device numerics

try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001
    _ARCH = None

device = pytest.mark.skipif(_ARCH is None, reason="no HIP device")


def _run(sp, x, arch):
    import torch

    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    k = PT.build_permute_tiled(sp, arch)
    art = compile_kernel(k, arch=arch, capture_ir_text=False)
    sig = [
        {
            "name": p.name,
            "type": (
                p.type.name if p.type.name.startswith("ptr<") else f"ptr<{p.type.name}>"
            ),
        }
        for p in k.params
        if isinstance(p.type, PtrType)
    ]
    L = KernelLauncher(hsaco=art.hsaco, kernel_name=k.name, signature=sig)
    y = torch.full((x.numel(),), 0x55, dtype=x.dtype, device=x.device)
    L.bind(
        {"X": x.data_ptr(), "Y": y.data_ptr()},
        config=LaunchConfig(
            grid=PT.permute_tiled_grid(sp),
            block=(sp.block_size, 1, 1),
            stream=torch.cuda.current_stream().cuda_stream,
        ),
    )()
    torch.cuda.synchronize()
    return y


def _bits(shape, es, seed=0):
    import torch

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if es == 8:
        hi = torch.randint(
            -(2**31), 2**31, shape, dtype=torch.int64, device="cuda", generator=g
        )
        lo = torch.randint(
            0, 2**32, shape, dtype=torch.int64, device="cuda", generator=g
        )
        return (hi << 32) | lo
    lim = 2 ** (8 * es - 1)
    dt = {2: torch.int16, 4: torch.int32}[es]
    return torch.randint(
        -lim, lim, shape, dtype=torch.int64, device="cuda", generator=g
    ).to(dt)


SMALL = [
    ((256, 512), (1, 0)),
    ((64, 128, 96), (2, 0, 1)),
    ((32, 64, 64), (1, 0, 2)),
    ((4, 32, 16, 16, 32), (0, 4, 1, 3, 2)),
    ((8, 16, 16, 16, 32), (3, 2, 1, 0, 4)),
    ((2, 16, 8, 8, 16, 16), (0, 3, 1, 5, 2, 4)),
    ((2, 16, 8, 8, 16, 16), (4, 3, 2, 1, 0, 5)),
    ((4, 32, 56, 112), (0, 2, 3, 1)),
    ((4, 32, 14, 112), (2, 1, 0, 3)),
    ((3, 5, 7), (2, 1, 0)),
    ((8, 10, 6, 10), (0, 2, 1, 3)),
    ((4, 12, 6, 6, 8, 8), (1, 3, 5, 0, 2, 4)),
    ((6, 12, 4, 10), (2, 0, 3, 1)),
    ((5, 24, 40), (2, 1, 0)),
]


@device
@pytest.mark.parametrize("shape,perm", SMALL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stage", ("lds", "reg", "gather", None))
def test_device_bit_exact(shape, perm, dtype, stage):
    es = PT.ELEM_BYTES[dtype]
    x = _bits(shape, es)
    ref = x.permute(*perm).contiguous().reshape(-1)
    sp = PT.plan_permute_tiled(shape, perm, dtype, _ARCH, stage=stage)
    y = _run(sp, x, _ARCH)
    assert int((y != ref).sum()) == 0, sp.kernel_name()


@device
@pytest.mark.parametrize("tpc", (2, 3))
@pytest.mark.parametrize("go", ("in", "out"))
@pytest.mark.parametrize("tail", ("clamp", "guard"))
def test_device_persistent_loop_with_remainder(tpc, go, tail):
    shape, perm = (26, 64, 96), (2, 0, 1)
    x = _bits(shape, 2, seed=3)
    ref = x.permute(*perm).contiguous().reshape(-1)
    sp = PT.plan_permute_tiled(
        shape,
        perm,
        "f16",
        _ARCH,
        tiles_per_cta=tpc,
        grid_order=go,
        tile=(128, 96),
        block_size=1024 if _ARCH != "gfx1151" else 512,
    )
    sp = dataclasses.replace(sp, tail=tail)
    assert sp.n_tiles % tpc  # the last CTA runs a partial tile loop
    y = _run(sp, x, _ARCH)
    assert int((y != ref).sum()) == 0


@device
@pytest.mark.parametrize("tail", ("clamp", "guard"))
@pytest.mark.parametrize("stage", ("lds", "reg", "gather"))
def test_device_ragged_thread_count(tail, stage):
    # 56 phase-1 units with 64 threads: clamped lanes redo the last unit, guarded lanes skip it
    shape, perm = (4, 32, 14, 112), (2, 1, 0, 3)
    sp = PT.PermuteTiledSpec(
        x_shape=shape,
        perm=perm,
        tile=(4, 1, 1, 112),
        dtype="f16",
        block_size=64,
        stage=stage,
        tail=tail,
    )
    assert PT.is_valid_spec(sp, _ARCH)[0]
    assert sp.phase1_units % sp.block_size
    x = _bits(shape, 2, seed=5)
    ref = x.permute(*perm).contiguous().reshape(-1)
    y = _run(sp, x, _ARCH)
    assert int((y != ref).sum()) == 0


@device
@pytest.mark.skipif(
    _ARCH not in ("gfx950", "gfx942"), reason="needs a large-memory CDNA device"
)
@pytest.mark.parametrize("stage", ("lds", "reg", "gather"))
@pytest.mark.parametrize("big_side", ("out", "in"))
def test_device_beyond_2_pow_31_elements(stage, big_side):
    import torch

    # 2^31 + 2^17 f16 elements (4.3 GB per tensor): the output (big_side="out") or the input
    # (big_side="in") tile bases pass 2^31 bytes
    n = 2**27 + 2**13
    shape, perm = ((16, n), (1, 0)) if big_side == "out" else ((n, 16), (1, 0))
    sp = PT.plan_permute_tiled(
        shape, perm, "f16", _ARCH, stage=stage, run_bytes=256, max_tile_bytes=8192
    )
    assert sp.stage == stage
    x = _bits(shape, 2, seed=9)
    y = _run(sp, x, _ARCH)
    ref = x.permute(*perm).contiguous().reshape(-1)
    tail = 2**20
    assert int((y[-tail:] != ref[-tail:]).sum()) == 0
    assert int((y != ref).sum()) == 0
    del x, y, ref
    torch.cuda.empty_cache()


@device
@pytest.mark.parametrize("skew", ("none", "diag", "hash"))
@pytest.mark.parametrize("go", ("in", "out", "both"))
@pytest.mark.parametrize(
    "shape,perm",
    [
        ((8, 16, 16, 16, 32), (3, 2, 1, 0, 4)),
        ((64, 128, 96), (2, 0, 1)),
        ((2, 16, 8, 8, 16, 16), (0, 3, 1, 5, 2, 4)),
        ((26, 64, 96), (2, 0, 1)),
    ],
)
def test_device_grid_skew_is_a_bijection(skew, go, shape, perm):
    x = _bits(shape, 2, seed=11)
    ref = x.permute(*perm).contiguous().reshape(-1)
    sp = PT.plan_permute_tiled(
        shape,
        perm,
        "f16",
        _ARCH,
        grid_order=go,
        tiles_per_cta=3,
        run_bytes=256,
        max_tile_bytes=4096,
    )
    sp = dataclasses.replace(sp, skew=skew)
    if (
        len(shape) > 3 and skew != "none"
    ):  # 3-D cases canonicalise to a 2-D transpose, one grid axis
        assert PT.skew_axes(sp) is not None
    y = _run(sp, x, _ARCH)
    assert int((y != ref).sum()) == 0


def test_grid_walk_both_puts_both_fast_axes_last():
    sp = PT.PermuteTiledSpec(
        x_shape=(32, 64, 64, 64, 32),
        perm=(3, 2, 1, 0, 4),
        tile=(4, 1, 1, 4, 32),
        dtype="f16",
        block_size=64,
        grid_order="both",
    )
    # input-fastest grid axis 3 (D chunks), output-fastest grid axis 0 (A chunks)
    assert PT.grid_walk(sp)[-2:] == [0, 3]
    assert sorted(PT.grid_walk(sp)) == list(range(5))


def test_rank_above_eight_is_planned():
    # GETT operand packs of rank-5 contractions have 10 modes
    shape = (2, 4, 4, 6, 4, 8, 4, 8, 6, 10)
    perm = (1, 3, 5, 7, 9, 0, 2, 4, 6, 8)
    sp = PT.plan_permute_tiled(shape, perm, "f16", "gfx950")
    assert PT.is_valid_spec(sp)[0]
    assert sp.rank == 10


@device
def test_device_rank_ten():
    shape = (2, 2, 2, 3, 2, 4, 2, 4, 3, 10)
    perm = (1, 3, 5, 7, 9, 0, 2, 4, 6, 8)
    x = _bits(shape, 2, seed=13)
    ref = x.permute(*perm).contiguous().reshape(-1)
    y = _run(PT.plan_permute_tiled(shape, perm, "f16", _ARCH), x, _ARCH)
    assert int((y != ref).sum()) == 0
