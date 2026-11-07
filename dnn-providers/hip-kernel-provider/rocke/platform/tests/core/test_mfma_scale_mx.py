# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 ``tile.mfma_scale_mx`` (free formats, 6-dword fp6 operands, op_sel).

CPU-only, Python engine (the op is in ``PYTHON_ONLY_OPCODES``): builder
validation, the emitted declaration and call at every LLVM flavor for both
shapes and every operand width pair, the op_sel / format immediates, target
rejection, and the host lane maps of ``rocke.core.arch.mfma_scale_mx``
(bijective over each tile). The gfx950 numerics of the fp6 x fp6 route are in
``library/tests/test_mxfp6_gemm_numeric.py`` (fp64 reference, must-fail
mutants).
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


def test_op_is_python_only():
    assert "tile.mfma_scale_mx" in PYTHON_ONLY_OPCODES
    assert uses_python_only_ops(_mfma_kernel("16x16x128", "fp6", "fp6")) == [
        "tile.mfma_scale_mx"
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
    with pytest.raises(ValueError, match="expects operands"):
        # fp6 is six dwords, not the eight of the catalog atom
        b.mfma_scale_mx(v8, v8, acc4, s, s, fmt_a="fp6", fmt_b="fp6")
    with pytest.raises(ValueError, match="opsel_b"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, opsel_b=-1)
    with pytest.raises(ValueError, match="opsel_b"):
        b.mfma_scale_mx(v8, v4, acc4, s, s, opsel_b=4)
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
    r = b.mfma_scale_mx(v6, v6, acc4, s, s, fmt_a="fp6", fmt_b="fp6", opsel_a=3)
    assert r.type.name == "vec<f32x4>"
    assert mx.format_code("fp8e4m3") == 0 and mx.format_code("fp4e2m1") == 4
    assert mx.format_code("fp6e2m3") == 2 and mx.format_code("fp6") == 2


_PAIRS = [
    ("fp6", "fp6"),
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


@pytest.mark.parametrize("shape", sorted(mx.SHAPES))
def test_fp6_lane_holds_one_scale_block(shape):
    """Every fp6 lane fragment is one contiguous 32-K block: one E8M0 scale."""
    ar = mx.HostArith
    for lane in range(64):
        ks = [mx.a_coord(ar, shape, "fp6", lane, s)[1] for s in range(32)]
        assert ks == list(range(ks[0], ks[0] + 32)) and ks[0] % 32 == 0
        assert mx.scale_a_coord(ar, shape, lane)[1] == ks[0] // 32
