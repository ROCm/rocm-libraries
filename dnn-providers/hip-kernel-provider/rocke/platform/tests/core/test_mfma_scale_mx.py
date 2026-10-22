# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 ``tile.mfma_scale_mx`` and ``arith.cvt_scalef32_pk_{fp8,bf8}_f32``.

CPU-only, Python engine (both ops are in ``PYTHON_ONLY_OPCODES``): builder
validation, the emitted declaration and call at every LLVM flavor for both
shapes and every operand width pair, the op_sel / format immediates, the
convert's ``hi`` / ``bf8`` forms, target rejection, and the host lane maps of
``rocke.core.arch.mfma_scale_mx`` (bijective over each tile). The gfx950
numerics of both ops are in ``library/tests/test_mx_mfma.py`` (bitwise against
an exact reference and the catalog route) and
``tests/numeric/test_mx_quant.py`` (the convert, through a ``cvt`` callable).
"""

import re

import pytest
from rocke.core.arch import mfma_scale_mx as mx
from rocke.core.ir import (
    F32,
    I32,
    PYTHON_ONLY_OPCODES,
    IRBuilder,
    PtrType,
    VectorType,
    uses_python_only_ops,
)
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python


def _operand(b, words):
    return b.vec_bitcast(b.zero_vec_f32(words), VectorType(I32, words))


def _mfma_kernel(shape, fa, fb, opsel_a=0, opsel_b=0):
    sh = mx.shape(shape)
    b = IRBuilder(f"mx_{shape}_{fa}_{fb}")
    pc = b.param("C", PtrType(F32, "global"))
    ps = b.param("S", PtrType(I32, "global"))
    sa = b.global_load(ps, b.const_i32(0), I32, align=4)
    sb = b.global_load(ps, b.const_i32(1), I32, align=4)
    acc = b.mfma_scale_mx(
        _operand(b, mx.OPERAND_WORDS[mx.format_code(fa)]),
        _operand(b, mx.OPERAND_WORDS[mx.format_code(fb)]),
        b.zero_vec_f32(sh.c_len),
        sa,
        sb,
        shape=shape,
        fmt_a=fa,
        fmt_b=fb,
        opsel_a=opsel_a,
        opsel_b=opsel_b,
    )
    b.global_store(pc, b.const_i32(0), b.vec_extract(acc, 0), align=4)
    b.ret()
    return b.kernel


def test_ops_are_python_only():
    assert {
        "tile.mfma_scale_mx",
        "arith.cvt_scalef32_pk_fp8_f32",
        "arith.cvt_scalef32_pk_bf8_f32",
    } <= set(PYTHON_ONLY_OPCODES)
    assert uses_python_only_ops(_mfma_kernel("16x16x128", "fp8", "fp4")) == [
        "tile.mfma_scale_mx"
    ]
    assert uses_python_only_ops(_cvt_kernel(False, True)) == [
        "arith.cvt_scalef32_pk_bf8_f32"
    ]


def test_builder_validation():
    b = IRBuilder("v")
    v8, v6, v4 = _operand(b, 8), _operand(b, 6), _operand(b, 4)
    s = b.const_i32(127)
    acc4, acc16 = b.zero_vec_f32(4), b.zero_vec_f32(16)
    with pytest.raises(ValueError, match="expects operands"):
        b.mfma_scale_mx(v8, v8, acc4, s, s, fmt_b="fp4")
    with pytest.raises(ValueError, match="expects operands"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, shape="32x32x64")
    with pytest.raises(ValueError, match="expects operands"):
        b.mfma_scale_mx(v8, v4, acc4, b.const_f32(1.0), s)
    with pytest.raises(ValueError, match="opsel_b"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, opsel_b=-1)
    with pytest.raises(ValueError, match="opsel_a"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, opsel_a=True)
    with pytest.raises(ValueError, match="shape"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, shape="16x16x64")
    with pytest.raises(ValueError, match="format"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, fmt_a="int8")
    with pytest.raises(ValueError, match="0..4"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, fmt_a=5)
    with pytest.raises(TypeError):
        mx.format_code(True)
    r = b.mfma_scale_mx(v6, v6, acc16, s, s, shape="32x32x64", fmt_a=2, fmt_b="bf6")
    assert r.type.name == "vec<f32x16>"
    assert mx.format_code("fp8e4m3") == 0 and mx.format_code("fp4e2m1") == 4
    with pytest.raises(ValueError, match="must be f32"):
        b.cvt_scalef32_pk_fp8_f32(s, s, b.const_f32(1.0), b.const_f32(1.0))
    with pytest.raises(ValueError, match="old must be i32"):
        b.cvt_scalef32_pk_fp8_f32(b.const_f32(0.0), *([b.const_f32(1.0)] * 3))


_PAIRS = [
    ("fp8", "fp4"),
    ("fp8", "fp8"),
    ("fp4", "fp4"),
    ("bf8", "fp6"),
    ("bf6", "fp8"),
]


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("shape", sorted(mx.SHAPES))
@pytest.mark.parametrize("fmts", _PAIRS, ids=lambda p: f"{p[0]}x{p[1]}")
def test_lowering_per_flavor(flavor, shape, fmts):
    fa, fb = (mx.format_code(f) for f in fmts)
    wa, wb = mx.OPERAND_WORDS[fa], mx.OPERAND_WORDS[fb]
    c_len = mx.shape(shape).c_len
    ir = _lower_kernel_to_llvm_python(
        _mfma_kernel(shape, fa, fb, 2, 1), arch="gfx950", llvm_flavor=flavor
    )
    name = f"llvm.amdgcn.mfma.scale.f32.{shape}.f8f6f4.v{wa}i32.v{wb}i32"
    acc = f"<{c_len} x float>"
    decl = (
        f"declare {acc} @{name}(<{wa} x i32>, <{wb} x i32>, {acc}, i32 immarg, "
        f"i32 immarg, i32 immarg, i32, i32 immarg, i32)"
    )
    assert ir.count(decl) == 1, ir
    calls = re.findall(rf"call {re.escape(acc)} @{re.escape(name)}\((.*)\)", ir)
    assert len(calls) == 1
    assert re.search(
        rf"<{wa} x i32> %\S+, <{wb} x i32> %\S+, {re.escape(acc)} %\S+, "
        rf"i32 {fa}, i32 {fb}, i32 2, i32 %\S+, i32 1, i32 %\S+$",
        calls[0],
    ), calls[0]


def test_lowering_rejects_other_targets():
    with pytest.raises(NotImplementedError, match="gfx950"):
        _lower_kernel_to_llvm_python(
            _mfma_kernel("16x16x128", "fp8", "fp4"), arch="gfx942", llvm_flavor="llvm22"
        )
    with pytest.raises(NotImplementedError, match="gfx950"):
        _lower_kernel_to_llvm_python(
            _cvt_kernel(True, False), arch="gfx942", llvm_flavor="llvm22"
        )


def _cvt_kernel(hi, bf8):
    b = IRBuilder(f"cvt_{int(hi)}_{int(bf8)}")
    px = b.param("X", PtrType(F32, "global"))
    pq = b.param("Q", PtrType(I32, "global"))
    x = [b.global_load(px, b.const_i32(i), F32, align=4) for i in range(3)]
    old = b.global_load(pq, b.const_i32(1), I32, align=4)
    q = b.cvt_scalef32_pk_fp8_f32(old, x[0], x[1], x[2], hi=hi, bf8=bf8)
    b.global_store(pq, b.const_i32(0), q, align=4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("hi", [False, True])
@pytest.mark.parametrize("bf8", [False, True])
def test_cvt_lowering(flavor, hi, bf8):
    kind = "bf8" if bf8 else "fp8"
    ir = _lower_kernel_to_llvm_python(
        _cvt_kernel(hi, bf8), arch="gfx950", llvm_flavor=flavor
    )
    intrin = f"llvm.amdgcn.cvt.scalef32.pk.{kind}.f32"
    assert (
        ir.count(
            f"declare <2 x i16> @{intrin}(<2 x i16>, float, float, float, i1 immarg)"
        )
        == 1
    )
    call = re.search(
        rf"(%\S+) = call <2 x i16> @{re.escape(intrin)}\(<2 x i16> (%\S+), "
        rf"float %\S+, float %\S+, float %\S+, i1 (true|false)\)",
        ir,
    )
    assert call and call.group(3) == ("true" if hi else "false")
    # ``old`` is the i32 reinterpreted, and the result goes back to i32.
    assert re.search(rf"{re.escape(call.group(2))} = bitcast i32 %\S+ to <2 x i16>", ir)
    assert re.search(rf"= bitcast <2 x i16> {re.escape(call.group(1))} to i32", ir)


@pytest.mark.parametrize("shape", sorted(mx.SHAPES))
@pytest.mark.parametrize("fmt", ["fp8", "bf8", "fp6", "bf6", "fp4"])
def test_lane_maps_bijective(shape, fmt):
    sh = mx.shape(shape)
    ar = mx.HostArith
    a = {mx.a_coord(ar, shape, fmt, l, s) for l in range(64) for s in range(32)}
    b = {mx.b_coord(ar, shape, fmt, l, s) for l in range(64) for s in range(32)}
    assert a == {(r, k) for r in range(sh.m) for k in range(sh.k)}
    assert b == {(k, c) for k in range(sh.k) for c in range(sh.n)}
    c = [mx.c_coord(ar, shape, l, i) for l in range(64) for i in range(sh.c_len)]
    assert sorted(c) == [(r, col) for r in range(sh.m) for col in range(sh.n)]
    sa = {mx.scale_a_coord(ar, shape, l) for l in range(64)}
    sb = {mx.scale_b_coord(ar, shape, l) for l in range(64)}
    assert sa == {(r, kb) for r in range(sh.m) for kb in range(sh.k_blocks)}
    assert sb == {(kb, col) for kb in range(sh.k_blocks) for col in range(sh.n)}
