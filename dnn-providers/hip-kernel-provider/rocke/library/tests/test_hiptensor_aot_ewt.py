# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numerics of the tiled transposing elementwise kernel (``kernels.common.hiptensor_aot.ewt``).

Permutations and elementwise binary / trinary operations whose inputs are fastest along another
mode than the output, against an independent float64 reference computed with torch on the CPU:
integer-valued inputs and exact operators (identity, negation, abs, ReLU; add, mul, max, min)
make every result exact, so the outputs are compared bit for bit with the reference rounded once
to the output type. Every input kind (read through LDS along the transposed mode, vector loads
along the output's mode, gathered), ragged tiles, zero scales (the input is not read: a NaN in
it does not reach D) and batch modes. Launch arguments come from :func:`ewt_plan`, the Python
statement of hipTensor's C++ planner.
"""

from __future__ import annotations

import math
import random

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.ewt import (
    KIND_G,
    KIND_T,
    KIND_V,
    THREADS,
    EwtSpec,
    build_ewt,
    ewt_fold,
    ewt_plan,
)
from kernels.common.hiptensor_aot.host import canonicalize


def test_plan_needs_a_transposed_input():
    spec = EwtSpec(1, "f16", "f16")
    # D[i, j] = A[i, j] (no transpose)
    assert ewt_plan(spec, [(64, [1, 1]), (64, [64, 64])]) is None
    # D[i, j] = A[j, i]
    vals, grid, kinds = ewt_plan(spec, [(64, [64, 1]), (48, [1, 64])])
    assert kinds == [KIND_T] and grid == (1, 1, 1)
    # D's extent along its fastest mode not a multiple of the vector: not served
    assert ewt_plan(spec, [(60, [64, 1]), (48, [1, 60])]) is None


def test_fold_plan():
    """Planes smaller than half a tile are folded: fo x ft planes of consecutive batch indices
    per CTA, a grid of batch groups; larger planes are not."""
    spec = EwtSpec(1, "f32", "f32", fold=True)
    assert ewt_fold(spec, 16, 16) == (4, 4)
    assert ewt_fold(spec, 40, 64) is None and ewt_fold(spec, 64, 64) is None
    assert ewt_fold(spec, 40, 24) == (1, 2)
    # D[i, j, b] = A[j, i, b]: 16 x 16 planes, 37 batch indices
    dims = [(16, [16, 1]), (16, [1, 16]), (37, [256, 256])]
    vals, grid, kinds = ewt_plan(spec, dims)
    assert grid == (1, 3, 1) and vals["nb"] == 3 and vals["nbt"] == 37
    assert (vals["fo"], vals["ft"]) == (4, 4) and kinds == [KIND_T]
    assert ewt_plan(spec, [(64, [64, 1]), (64, [1, 64])]) is None
    assert build_ewt(spec).name.endswith("_f")


def test_fold_plan_bounded_by_the_table():
    """A tile holds at most THREADS planes (the rows of the kernel's batch-offset table, the pack
    attribute ``max_planes`` hipTensor's planner checks): smaller planes are not folded.
    """
    f64 = EwtSpec(1, "f64", "f64", fold=True)  # tiles of 64 x 32
    assert ewt_fold(f64, 2, 2) is None  # 32 x 16 = 512 planes
    assert ewt_fold(f64, 2, 4) == (32, 8)  # 256 planes: the whole table
    assert ewt_fold(f64, 4, 2) == (16, 16)
    f32 = EwtSpec(1, "f32", "f32", fold=True)  # tiles of 64 x 64
    assert ewt_fold(f32, 4, 4) == (16, 16)
    assert ewt_fold(f32, 4, 2) is None and ewt_fold(f32, 2, 4) is None
    for spec in (f64, f32):
        for eo in range(1, 65):
            for et in range(1, spec.tile_t() + 1):
                f = ewt_fold(spec, eo, et)
                assert f is None or 2 <= f[0] * f[1] <= THREADS
    # D[i, j, b] = A[j, i, b], f64 2 x 2 planes: past the table, not served by the folded class
    dims = [(2, [2, 1]), (2, [1, 2]), (300, [4, 4])]
    assert ewt_plan(f64, dims) is None
    vals, grid, kinds = ewt_plan(EwtSpec(1, "f64", "f64"), dims)
    assert grid == (1, 300, 1) and kinds == [KIND_T]


def test_specs_build():
    for sp in (
        EwtSpec(1, "f16", "f32"),
        EwtSpec(2, "bf16", "bf16", c="bf16"),
        EwtSpec(3, "f32", "f32", "f32", "f32"),
        EwtSpec(3, "f64", "f64", "f64", "f64"),
    ):
        k = build_ewt(sp)
        names = [p.name for p in k.params]
        assert len(set(names)) == len(names)


torch = pytest.importorskip("torch")
GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not GPU, reason="needs a GPU")
TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
_CACHE = {}


def _arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _run(kernel, vals, grid):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if kernel.name not in _CACHE:
        art = compile_kernel(
            kernel, arch=_arch(), backend="python", capture_ir_text=False
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


def _tensor(shape, order, dt, rng, pad=0):
    """A device view of ``shape`` stored in mode order ``order`` (fastest last) with ``pad``
    elements after each run of the fastest mode; integer values in [-3, 3]."""
    stor = [shape[i] for i in order]
    stor[-1] += pad
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    base = torch.randint(-3, 4, stor, generator=g).to(TD[dt]).cuda()
    view = base[..., : shape[order[-1]]] if pad else base
    return view.permute([order.index(i) for i in range(len(shape))])


def _ref_op(x, op):
    return {
        C.OP_IDENTITY: lambda t: t,
        C.OP_NEG: torch.neg,
        C.OP_ABS: torch.abs,
        C.OP_RELU: torch.relu,
    }[op](x)


def _bin(a, c, op):
    return {
        C.OP_ADD: lambda: a + c,
        C.OP_MUL: lambda: a * c,
        C.OP_MAX: lambda: torch.maximum(a, c),
        C.OP_MIN: lambda: torch.minimum(a, c),
    }[op]()


def run_case(
    rng,
    arity,
    a,
    d,
    shape,
    orders,
    dorder,
    kinds_want=None,
    zero=None,
    pads=None,
    fold=False,
    transposed=True,
):
    names = {1: ["A"], 2: ["A", "C"], 3: ["A", "B", "C"]}[arity]
    pads = pads or {}
    ins = {n: _tensor(shape, orders[n], a, rng, pads.get(n, 0)) for n in names}
    if zero:
        # a NaN in the input whose scale is zero: it must not be read
        ins[zero].select(0, 0).fill_(float("nan"))
    D = torch.full(
        [shape[i] for i in dorder], float("nan"), dtype=TD[d], device="cuda"
    ).permute([dorder.index(i) for i in range(len(shape))])
    dims = [
        (shape[i], [ins[n].stride(i) for n in names] + [D.stride(i)])
        for i in range(len(shape))
    ]
    cdims = canonicalize(dims, len(names))
    spec = EwtSpec(
        arity,
        a,
        d,
        b=a if arity == 3 else None,
        c=a if arity >= 2 else None,
        fold=fold,
    )
    plan = ewt_plan(spec, cdims, transposed=transposed)
    assert plan is not None, cdims
    vals, grid, kinds = plan
    if kinds_want is not None:
        assert kinds == kinds_want, kinds
    ops = {n: rng.choice([C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]) for n in names}
    bops = [rng.choice(C.BINARY_OPS), rng.choice(C.BINARY_OPS)]
    scal = {n: rng.choice([1.0, 2.0, -1.0, 0.5]) for n in names}
    if zero:
        scal[zero] = 0.0
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    for n in names:
        vals[n] = ins[n]
        vals["op" + n] = ops[n]
        vals[sn[n]] = scal[n]
    vals["D"] = D
    vals["opAB"], vals["opABC"] = bops
    k = build_ewt(spec)
    _run(k, vals, grid)

    def term(n):
        if scal[n] == 0.0:
            return torch.zeros(shape, dtype=torch.float64)
        return scal[n] * _ref_op(ins[n].cpu().double(), ops[n])

    if arity == 1:
        ref = term("A")
    elif arity == 2:
        ref = _bin(term("A"), term("C"), bops[0])
    else:
        ref = _bin(_bin(term("A"), term("B"), bops[0]), term("C"), bops[1])
    want = ref.to(TD[d]).double()
    got = D.cpu().double()
    assert not got.isnan().any()
    # zeros: a zero scale adds +0 (as the generic kernel), so compare signs only where nonzero
    assert torch.equal(got, want), (got - want).abs().max()
    return kinds


CASES = []
_rng = random.Random(31)
for arity in (1, 2, 3):
    for a, d in (
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f32", "f32"),
        ("f64", "f64"),
        ("f16", "f32"),
        ("f32", "f16"),
    ):
        for rank in (2, 3, 4):
            CASES.append((arity, a, d, rank, _rng.randrange(1 << 20)))


@gpu
@pytest.mark.parametrize("arity,a,d,rank,seed", CASES)
def test_ewt_exact(arity, a, d, rank, seed):
    rng = random.Random(seed)
    shape = [rng.choice([3, 5, 16, 24, 40, 64, 72, 136]) for _ in range(rank)]
    shape[0] = 8 * rng.choice([1, 3, 9, 17])  # D's fastest mode: whole vectors
    shape[1] = 8 * rng.choice([2, 5, 13])  # the transposed mode: whole vectors
    while math.prod(shape) > 1 << 21:
        i = max(range(2, rank), key=lambda j: shape[j]) if rank > 2 else 0
        shape[i] = max(1, shape[i] // 2)
        if rank == 2:
            break
    dorder = list(range(rank))[::-1]  # D: mode 0 fastest
    rest = list(range(2, rank))
    names = {1: ["A"], 2: ["A", "C"], 3: ["A", "B", "C"]}[arity]
    orders = {}
    for i, n in enumerate(names):
        r = rest[:]
        rng.shuffle(r)
        if i == 0:
            orders[n] = r + [0, 1]  # A: mode 1 fastest (transposed)
        else:
            orders[n] = r + rng.choice([[0, 1], [1, 0], [1, 0]])
    run_case(rng, arity, a, d, shape, orders, dorder)


@gpu
@pytest.mark.parametrize("dt", ["f16", "f32"])
def test_ewt_every_kind_and_zero_scale(dt):
    rng = random.Random(3)
    shape = [72, 40, 6]
    dorder = [2, 1, 0]
    # A transposed (T), B along D's mode (V), C fastest along a batch mode (G)
    orders = {"A": [2, 0, 1], "B": [2, 1, 0], "C": [1, 0, 2]}
    kinds = run_case(
        rng, 3, dt, dt, shape, orders, dorder, kinds_want=[KIND_T, KIND_V, KIND_G]
    )
    assert kinds == [KIND_T, KIND_V, KIND_G]
    for z in ("A", "B", "C"):
        run_case(random.Random(4), 3, dt, dt, shape, orders, dorder, zero=z)


FOLD_CASES = []
_frng = random.Random(57)
for arity in (1, 2, 3):
    for a, d in (("f16", "f16"), ("f32", "f32"), ("f64", "f64"), ("bf16", "f32")):
        for eo, et in ((16, 16), (8, 24), (32, 32), (24, 8), (16, 48)):
            FOLD_CASES.append((arity, a, d, eo, et, _frng.randrange(1 << 20)))
    # the whole batch-offset table (256 planes per tile), and planes that do not divide the tile
    # (positions outside every plane)
    for a, d, eo, et in (
        ("f64", "f64", 2, 4),
        ("f32", "f32", 4, 4),
        ("f32", "f32", 12, 20),
    ):
        FOLD_CASES.append((arity, a, d, eo, et, _frng.randrange(1 << 20)))


@gpu
@pytest.mark.parametrize("arity,a,d,eo,et,seed", FOLD_CASES)
def test_ewt_fold_exact(arity, a, d, eo, et, seed):
    """Folded classes: small planes of several batch indices per tile, batch counts that are not
    multiples of the planes per tile, inputs of every kind, zero scales."""
    rng = random.Random(seed)
    vo = 16 // C.DTYPES[d][1]
    vt = 16 // C.DTYPES[a][1]
    eo = max(vo, eo // vo * vo)
    # a plane within one tile (8-byte inputs: tiles of 32 along t)
    et = min(max(vt, et // vt * vt), 32 if C.DTYPES[a][1] == 8 else 64)
    rank = rng.choice([3, 4])
    shape = [eo, et] + [rng.choice([3, 5, 7]) for _ in range(rank - 2)]
    dorder = list(range(rank))[::-1]
    rest = list(range(2, rank))
    names = {1: ["A"], 2: ["A", "C"], 3: ["A", "B", "C"]}[arity]
    orders = {}
    for i, n in enumerate(names):
        r = rest[:]
        rng.shuffle(r)
        orders[n] = r + [0, 1] if i == 0 else r + rng.choice([[0, 1], [1, 0]])
    zero = rng.choice([None, names[-1]]) if arity > 1 else None
    run_case(rng, arity, a, d, shape, orders, dorder, zero=zero, fold=True)


@gpu
@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("fold", [False, True])
def test_ewt_without_a_transposed_input(arity, fold):
    """Layouts whose inputs are fastest along D's mode (vector loads) or gathered, many modes that
    do not merge (hipTensor's contraction packs past the generic kernel's slots): the tile's
    other mode is D's next fastest mode."""
    rng = random.Random(71 + arity)
    shape = [8, 3, 2, 3, 2, 3, 2] if fold else [72, 40, 3, 2, 3, 2]
    rank = len(shape)
    dorder = list(range(rank))[::-1]
    names = {1: ["A"], 2: ["A", "C"]}[arity]
    orders = {}
    for n in names:
        r = list(range(1, rank))
        rng.shuffle(r)
        orders[n] = r + [0]  # fastest along D's fastest mode
    # every mode separate (no input fastest along another mode): only the relaxed plan
    ins = {n: _tensor(shape, orders[n], "f32", rng) for n in names}
    D = torch.empty([shape[i] for i in dorder], device="cuda").permute(
        [dorder.index(i) for i in range(rank)]
    )
    dims = [
        (shape[i], [ins[n].stride(i) for n in names] + [D.stride(i)])
        for i in range(rank)
    ]
    spec = EwtSpec(arity, "f32", "f32", c="f32" if arity == 2 else None, fold=fold)
    cd = canonicalize(dims, len(names))
    assert len(cd) > 2 and ewt_plan(spec, cd) is None
    kinds = run_case(
        rng, arity, "f32", "f32", shape, orders, dorder, fold=fold, transposed=False
    )
    assert KIND_T not in kinds
