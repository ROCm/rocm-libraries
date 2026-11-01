# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The flat elementwise kernel (``kernels.common.hiptensor_aot.ewf``).

CPU: the class set of hipTensor's kernel pack (one class per real row of the generic kernel, on
the streaming targets only), the kernel names and the kernel ABI (parameter order).

GPU: every class against the pack's generic elementwise kernel (``.ew``) of the same row, bit for
bit (a NaN equals a NaN of any sign and payload), and on integer data also against an
independent float64 torch reference: element counts with a partial last block and a single
vector, every arity and data-type row, (1) integer data with the exact operators, (2) data with
NaN, infinities, signed zeros, overflowing and tiny values with every basic operator and binary
operator, (3) a zero scale (that input, all NaN, is not read) and (4) a NaN scale (counts as
zero, as in the generic kernel). The classes stream (``nt``) on gfx942 / gfx950 and are built
without it elsewhere (the test runs anywhere; the pack has them on the streaming targets).
"""

from __future__ import annotations

import importlib.util
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew
from kernels.common.hiptensor_aot.ewf import (
    EWF_DTYPES,
    EWF_STREAMING_ARCHS,
    EWF_UNROLL,
    EwfSpec,
    build_ewf,
    ewf_classes,
)
from kernels.common.hiptensor_aot.host import ew_args


def _api_rows():
    """The real 32-bit rows of the pack's generic kernel (make_rocke_pack.ew_params, the rows of
    the API's elementwise operations)."""
    path = (
        Path(__file__).resolve().parents[1]
        / "builders"
        / "hiptensor"
        / "make_rocke_pack.py"
    )
    spec = importlib.util.spec_from_file_location("_ewf_make_rocke_pack", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ew_params(False, mixed=False)


ROWS = _api_rows()
CLASSES = ewf_classes(ROWS, "gfx950")
# every real row, for the numerics (the builder takes every row; the pack has the 16-bit ones)
ALL = [
    EwfSpec(arity=p["arity"], a=p["a"], d=p["d"], b=p.get("b"), c=p.get("c"), nt=True)
    for p in ROWS
    if not p.get("cplx") and not p.get("math")
]


def test_class_set():
    real = [
        p
        for p in ROWS
        if not p.get("cplx")
        and not p.get("math")
        and all(p.get(k) in EWF_DTYPES for k in ("a", "b", "c", "d") if p.get(k))
    ]
    assert len(CLASSES) == len(real) == 6
    assert ewf_classes(ROWS, "gfx942") == CLASSES
    assert ewf_classes(ROWS, "gfx90a") == []
    names = [c.kernel_name() for c in CLASSES]
    assert len(set(names)) == len(names)
    assert all(c.nt and c.block == 256 for c in CLASSES)
    assert "hiptensor_aot_ewf2_r_f16_f16_f16_b256_v8_x2_nt" in names


def test_kernel_abi():
    k = build_ewf(EwfSpec(arity=3, a="f16", b="f16", c="f16", d="f32", nt=True))
    assert [p.name for p in k.params] == [
        "A",
        "B",
        "C",
        "D",
        "alpha",
        "beta",
        "gamma",
        "opA",
        "opB",
        "opC",
        "opAB",
        "opABC",
        "n",
    ]
    k = build_ewf(EwfSpec(arity=1, a="f32", d="f16"))
    assert [p.name for p in k.params] == ["A", "D", "alpha", "opA", "n"]
    with pytest.raises(ValueError):
        EwfSpec(arity=2, a="f16", d="f16").check()  # no C dtype


torch = pytest.importorskip("torch")

TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
_BITS = {
    "f16": torch.int16,
    "bf16": torch.int16,
    "f32": torch.int32,
    "f64": torch.int64,
}
_SPECIALS = [math.nan, math.inf, -math.inf, -0.0, 0.0, 7.0e4, -3.0e38, 1.0e-30, 0.75]
_LAUNCHERS = {}


def _gpu():
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only runners
        pytest.skip("needs a GPU")
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _launch(kernel, vals, grid):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if kernel.name not in _LAUNCHERS:
        art = compile_kernel(
            kernel, arch=_gpu(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _LAUNCHERS[kernel.name] = (
            KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig),
            [p.name for p in kernel.params],
        )
    launcher, names = _LAUNCHERS[kernel.name]
    launcher(
        {n: vals[n] for n in names}, config=LaunchConfig(grid=grid, block=(256, 1, 1))
    )
    torch.cuda.synchronize()


def _binary(a, c, op):
    return {
        C.OP_ADD: lambda: a + c,
        C.OP_MUL: lambda: a * c,
        C.OP_MAX: lambda: torch.maximum(a, c),
        C.OP_MIN: lambda: torch.minimum(a, c),
    }[op]()


def _unary(x, op):
    return {
        C.OP_IDENTITY: lambda t: t,
        C.OP_NEG: torch.neg,
        C.OP_ABS: torch.abs,
        C.OP_RELU: torch.relu,
    }[op](x)


def _bits(t, dt):
    t = t.cpu().contiguous()
    return t.view(_BITS[dt]), torch.isnan(t)


@pytest.mark.parametrize("spec", ALL, ids=lambda s: s.kernel_name())
def test_ewf_matches_generic_kernel(spec):
    arch = _gpu()
    if arch not in EWF_STREAMING_ARCHS:
        spec = replace(spec, nt=False)
    base = spec.base()
    names = base.tensors()
    ins = names[:-1]
    dts = {"A": spec.a, "B": spec.b, "C": spec.c, "D": spec.d}
    rng = random.Random(sum(map(ord, spec.kernel_name())))
    vec = spec.vec
    per_block = 256 * EWF_UNROLL
    sizes = [vec, vec * (3 * per_block + 37), vec * (per_block - 1)]
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    exact_ops = [C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]
    for n_el in sizes:
        for case in ("reference", "special", "zero_scale", "nan_scale"):
            g = torch.Generator().manual_seed(rng.randrange(1 << 30))
            values = {}
            for n in ins:
                v = torch.randint(-6, 7, (n_el,), generator=g).double()
                if case == "special":
                    idx = torch.randperm(n_el, generator=g)[: max(1, n_el // 3)]
                    for k, j in enumerate(idx.tolist()):
                        v[j] = _SPECIALS[k % len(_SPECIALS)]
                values[n] = v
            pool = list(C.BASIC_UNARY_OPS) if case == "special" else exact_ops
            ops = {n: rng.choice(pool) for n in ins}
            scal = {n: rng.choice([1.0, 2.0, -1.0, 0.5]) for n in ins}
            if case in ("zero_scale", "nan_scale"):
                # that input is all NaN and is not read (a NaN scale counts as zero)
                off = rng.choice(ins)
                scal[off] = 0.0 if case == "zero_scale" else math.nan
                values[off] = torch.full((n_el,), math.nan, dtype=torch.float64)
            dev = {n: values[n].to(TD[dts[n]]).cuda() for n in ins}
            vals = {"op" + n: ops[n] for n in ins}
            vals["opAB"] = rng.choice(list(C.BINARY_OPS))
            vals["opABC"] = rng.choice(list(C.BINARY_OPS))
            run = dict(dev)
            run.update({sn[n]: scal[n] for n in ins})
            out_f = torch.full((n_el,), 5.0, dtype=TD[spec.d], device="cuda")
            out_g = out_f.clone()
            # the generic kernel over the same single mode
            gv, ggrid, idx64 = ew_args([(n_el, [1] * len(names))], names)
            assert not idx64
            _launch(build_ew(base), dict(gv, **vals, **run, D=out_g), ggrid)
            fgrid = (-(-(n_el // vec) // per_block), 1, 1)
            _launch(build_ewf(spec), dict(vals, **run, D=out_f, n=n_el // vec), fgrid)
            (gb, gnan), (fb, fnan) = _bits(out_g, spec.d), _bits(out_f, spec.d)
            diff = ((gb != fb) & ~(gnan & fnan)).nonzero().view(-1)
            assert diff.numel() == 0, (
                n_el,
                case,
                ops,
                vals["opAB"],
                scal,
                [
                    (i, out_g.cpu()[i].item(), out_f.cpu()[i].item())
                    for i in diff[:4].tolist()
                ],
            )
            if case == "special":
                continue

            def term(n):
                if not scal[n] == scal[n] or scal[n] == 0.0:
                    return torch.zeros(n_el, dtype=torch.float64)
                x = values[n].to(TD[dts[n]]).double()
                return scal[n] * _unary(x, ops[n])

            if spec.arity == 1:
                ref = term("A")
            elif spec.arity == 2:
                ref = _binary(term("A"), term("C"), vals["opAB"])
            else:
                ref = _binary(
                    _binary(term("A"), term("B"), vals["opAB"]),
                    term("C"),
                    vals["opABC"],
                )
            want = ref.to(TD[spec.d])
            assert torch.equal(out_f.cpu(), want), (spec.kernel_name(), n_el, case, ops)
