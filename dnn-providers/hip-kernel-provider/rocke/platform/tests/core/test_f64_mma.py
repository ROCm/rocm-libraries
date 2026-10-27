# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 MFMA atom contracts and native parity, independent of GPU access."""
import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, F64, IRBuilder, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.gfx942.f64_mma_probe import (
    F64MmaProbeSpec,
    build_f64_mma_probe,
)

# m -> (op_id, c_frag_len, A/B/C shapes) for the gfx942 f64 atoms.
_ATOMS = {
    16: ("mfma_f64_16x16x4_f64", 4, ((16, 4), (4, 16), (16, 16))),
    4: ("mfma_f64_4x4x4_f64", 1, ((16, 4), (16, 4), (16, 4))),
}


class _Integers:
    const_i32 = staticmethod(int)
    mod = staticmethod(lambda x, y: x % y)
    div = staticmethod(lambda x, y: x // y)
    mul = staticmethod(lambda x, y: x * y)
    add = staticmethod(lambda x, y: x + y)


def _f64_atom(m):
    return ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp64", b_dtype="fp64", c_dtype="fp64", m=m, n=m, k=4
    )


@pytest.mark.parametrize("m", [16, 4])
def test_catalog_isolation(m):
    op_id, c_frag_len, _ = _ATOMS[m]
    atom = _f64_atom(m)
    assert atom.op_id == op_id
    assert atom.c_dtype == "fp64"
    assert (atom.a_frag_len, atom.b_frag_len, atom.c_frag_len) == (1, 1, c_frag_len)
    fp32 = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    assert fp32.op_id == "mfma_f32_16x16x4_f32"
    for gfx in ("gfx950", "gfx1151", "gfx1250"):
        assert not any(op.a_dtype == "fp64" for op in ArchTarget.from_gfx(gfx).mma.ops)


@pytest.mark.parametrize("m", [16, 4])
def test_layouts_are_bijective(m):
    atom = _f64_atom(m)
    _, c_frag_len, shapes = _ATOMS[m]
    calc = _Integers()
    for layout, shape, count in zip(
        (atom.a_layout(), atom.b_layout(), atom.c_layout()), shapes, (1, 1, c_frag_len)
    ):
        coords = [
            layout.coord(calc, lane, slot)
            for lane in range(64)
            for slot in range(count)
        ]
        assert len(set(coords)) == shape[0] * shape[1] == len(coords)


def test_layout_16x16x4():
    atom = _f64_atom(16)
    calc = _Integers()
    for lane in range(64):
        assert atom.a_layout().coord(calc, lane, 0) == (lane % 16, lane // 16)
        assert atom.b_layout().coord(calc, lane, 0) == (lane // 16, lane % 16)
        for slot in range(4):
            assert atom.c_layout().coord(calc, lane, slot) == (
                4 * slot + lane // 16,
                lane % 16,
            )


def test_layout_4x4x4():
    # Lane l = 16*k + 4*blk + (m or n) for A/B, 16*m + 4*blk + n for C/D, with
    # the four blocks stacked along rows.
    atom = _f64_atom(4)
    calc = _Integers()
    for lane in range(64):
        hi, blk, lo = lane // 16, (lane // 4) % 4, lane % 4
        assert atom.a_layout().coord(calc, lane, 0) == (4 * blk + lo, hi)
        assert atom.b_layout().coord(calc, lane, 0) == (4 * blk + hi, lo)
        assert atom.c_layout().coord(calc, lane, 0) == (4 * blk + hi, lo)


@pytest.mark.parametrize("m", [16, 4])
def test_mma_result_is_f64(m):
    op_id, c_frag_len, _ = _ATOMS[m]
    b = IRBuilder("f64_mma")
    a = b.param("a", F64)
    c = b.param("c", VectorType(F64, c_frag_len))
    assert b.mma(_f64_atom(m), a, a, c).type == VectorType(F64, c_frag_len)
    assert b.mma(op_id, a, a, c).type == VectorType(F64, c_frag_len)


def test_fp32_mma_result_is_unchanged():
    b = IRBuilder("f32_mma")
    f32 = b.param("f", F32)
    c = b.param("c", VectorType(F32, 4))
    assert b.mma("mfma_f32_16x16x4_f32", f32, f32, c).type == VectorType(F32, 4)


@pytest.mark.parametrize("m,k", [(16, 0), (16, 6), (8, 4)])
def test_probe_rejects_bad_spec(m, k):
    with pytest.raises(ValueError, match="f64 MMA probe requires"):
        F64MmaProbeSpec(m, k)


@pytest.mark.parametrize("m", [16, 4])
@pytest.mark.parametrize("k", [4, 8, 64])
def test_both_builders_and_lowerers(m, k):
    kernel = build_f64_mma_probe(F64MmaProbeSpec(m, k))
    ir = serialize(kernel)
    assert serialize(parse(ir)) == ir
    hip = lower_kernel_to_hip(kernel, arch="gfx942")
    assert f"__builtin_amdgcn_mfma_f64_{m}x{m}x4f64(" in hip
    call = (
        "call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64("
        if m == 16
        else "call double @llvm.amdgcn.mfma.f64.4x4x4f64("
    )
    native = pytest.importorskip("rocke_engine")
    assert native.f64_mma_probe_serialize_ir(m, k) == ir
    for flavor in ("llvm20", "llvm22", "llvm23"):
        ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
        assert native.lower_serialized_ir(ir, arch="gfx942", flavor=flavor) == ll
        assert ll.count(call) == k // 4
