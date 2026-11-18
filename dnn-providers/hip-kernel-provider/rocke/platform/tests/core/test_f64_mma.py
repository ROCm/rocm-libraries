# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 MFMA atom contracts and native parity, independent of GPU access."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from rocke.instances.gfx942 import f64_mma_probe as probe
from rocke.core.arch import ArchTarget, MmaCatalog, known_arches
from rocke.core.ir import F32, F64, IRBuilder, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python
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
    assert atom.srcs[2].frag_len == atom.dst.frag_len == c_frag_len
    fp32 = ArchTarget.from_gfx("gfx942").mma.op_for_shape(
        family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32", m=16, n=16, k=4
    )
    assert fp32.op_id == "mfma_f32_16x16x4_f32"
    # gfx90a and gfx950 list the 16x16x4 atom too; the four-block atom is
    # gfx942's alone, and no other target lists either.
    listed_on = {"gfx942"} if m == 4 else {"gfx90a", "gfx942", "gfx950"}
    for gfx in known_arches():
        lists = any(op.op_id == op_id for op in ArchTarget.from_gfx(gfx).mma.ops)
        assert lists == (gfx in listed_on), gfx


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
def test_python_builder_and_lowerers(m, k):
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
    for flavor in LLVM_FLAVORS:
        ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
        assert ll.count(call) == k // 4
        assert ll.count("store double ") == _ATOMS[m][1]


@pytest.mark.parametrize("m", [16, 4])
@pytest.mark.parametrize("k", [4, 8, 64])
def test_native_builder_and_lowerers_match_python(m, k):
    native = pytest.importorskip("rocke_engine")
    kernel = build_f64_mma_probe(F64MmaProbeSpec(m, k))
    ir = serialize(kernel)
    assert native.f64_mma_probe_serialize_ir(m, k) == ir
    for flavor in LLVM_FLAVORS:
        ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
        assert native.lower_serialized_ir(ir, arch="gfx942", flavor=flavor) == ll


_CONTRACT_CASES = [
    pytest.param(role, field, id=f"{name}-{field}")
    for role, name in enumerate(("src0", "src1", "src2", "dst"))
    for field in (
        "dtype",
        "width_short",
        "width_long",
        "width_float",
        "missing_map",
        "no_fn",
        "wrong_map",
        "map_width",
        "map_wave",
        "map_float",
    )
    if field != "wrong_map" or role in (2, 3)
] + [
    pytest.param(-1, field, id=field)
    for field in (
        "missing_atom",
        "family",
        "m",
        "n",
        "k",
        "shape_float",
        "wave",
        "wave_float",
        "both_width_short",
        "both_capacity",
    )
]


@pytest.mark.parametrize("m", [16, 4])
@pytest.mark.parametrize("role,field", _CONTRACT_CASES)
def test_probe_rejects_incompatible_contract_before_emission(m, role, field):
    atom = _f64_atom(m)
    srcs, dst = list(atom.srcs), atom.dst
    if role >= 0:
        desc = dst if role == 3 else srcs[role]
        if field == "dtype":
            desc = replace(desc, dtype="fp32")
        elif field.startswith("width_"):
            width = (
                float(desc.frag_len)
                if field == "width_float"
                else desc.frag_len + (1 if field == "width_long" else -1)
            )
            desc = replace(desc, frag_len=width)
        else:
            layout = desc.layout
            if field == "missing_map":
                layout = None
            elif field == "no_fn":
                layout = replace(layout, fn=None)
            elif field == "wrong_map":
                layout = replace(layout, fn=lambda b, lane, slot: (lane, slot))
            elif field == "map_width":
                layout = replace(layout, frag_len=layout.frag_len + 1)
            elif field == "map_float":
                layout = replace(layout, frag_len=float(layout.frag_len))
            else:
                layout = replace(layout, wave_size=32)
            desc = replace(desc, layout=layout)
        if role == 3:
            dst = desc
        else:
            srcs[role] = desc
        atom = replace(atom, srcs=tuple(srcs), dst=dst)
    elif field.startswith("both_"):
        width = 5 if field == "both_capacity" else dst.frag_len - 1
        srcs[2] = replace(srcs[2], frag_len=width)
        dst = replace(dst, frag_len=width)
        atom = replace(atom, srcs=tuple(srcs), dst=dst)
    elif field == "missing_atom":
        atom = None
    else:
        updates = {
            "family": {"family": "wrong"},
            "m": {"m": m + 1},
            "n": {"n": m + 1},
            "k": {"k": 8},
            "shape_float": {"m": float(m)},
            "wave": {"wave_size": 32},
            "wave_float": {"wave_size": 64.0},
        }
        atom = replace(atom, **updates[field])
    target = ArchTarget.from_gfx("gfx942")
    # Both shape and ID lookup see this catalog; leave the real query intact.
    catalog = MmaCatalog(
        [atom if op.op_id == _ATOMS[m][0] else op for op in target.mma.ops]
        if atom is not None
        else [op for op in target.mma.ops if op.op_id != _ATOMS[m][0]]
    )
    with (
        patch.object(ArchTarget, "from_gfx", return_value=replace(target, mma=catalog)),
        patch.object(probe, "IRBuilder", wraps=IRBuilder) as builder,
    ):
        with pytest.raises(ValueError) as error:
            build_f64_mma_probe(F64MmaProbeSpec(m))
        assert str(error.value) == (
            "F64 probe requires the fixed gfx942 instruction and layout contract"
        )
        builder.assert_not_called()


@pytest.mark.parametrize("m", [16, 4])
@pytest.mark.parametrize("forwarded", [False, True], ids=["copied", "shared-forwarder"])
def test_probe_accepts_copied_contract(m, forwarded):
    atom = _f64_atom(m)
    copied = replace(
        atom,
        srcs=tuple(replace(src, layout=replace(src.layout)) for src in atom.srcs),
        dst=replace(atom.dst, layout=replace(atom.dst.layout)),
    )
    if forwarded:
        original = atom.srcs[2].layout.fn

        def forward(b, lane, slot):
            return original(b, lane, slot)

        srcs = list(copied.srcs)
        srcs[2] = replace(srcs[2], layout=replace(srcs[2].layout, fn=forward))
        copied = replace(
            copied,
            srcs=tuple(srcs),
            dst=replace(copied.dst, layout=replace(copied.dst.layout, fn=forward)),
        )
    spec = F64MmaProbeSpec(m)
    expected = serialize(build_f64_mma_probe(spec))
    target = ArchTarget.from_gfx("gfx942")
    catalog = MmaCatalog(
        [copied if op.op_id == atom.op_id else op for op in target.mma.ops]
    )
    with patch.object(
        ArchTarget, "from_gfx", return_value=replace(target, mma=catalog)
    ):
        assert serialize(build_f64_mma_probe(spec)) == expected
