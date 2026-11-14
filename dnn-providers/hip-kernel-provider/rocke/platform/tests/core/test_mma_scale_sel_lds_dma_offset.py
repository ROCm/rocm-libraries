# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``scale_sel`` on a scaled ``tile.mma`` and ``offset`` on a buffer LDS DMA.

The direct compiler honours both attributes; the Python and C++ LLVM and HIP
lowerings emit them into the intrinsic's ``opsel_a`` / ``opsel_b`` and
instruction-offset operands. A kernel without them serializes and lowers
exactly as before. The builder refuses a bad value, and every lowering refuses
one that reaches it another way (an attribute set after building, or a
deserialized kernel).

The 16x16x128 MX atom (``mfma_scale_f32_16x16x128_f8f6f4``) takes the same
``scale_sel``; every lowering also refuses its bad operands, an ``a_format`` /
``b_format`` other than fp8e4m3, and a target other than gfx950.
"""

import re

import pytest
from rocke.core.ir import F16, F32, I8, I32, IRBuilder, PtrType, VectorType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.atoms import MFMA_ATOMS
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

FLAVORS = ["llvm20", "llvm22", "llvm23"]
MX16 = "mfma_scale_f32_16x16x128_f8f6f4"
# Not two exact ints in 0..3: out of range, wrong length, or bool / float / str.
BAD_SCALE_SELS = [
    [4, 0],
    [0, -1],
    [1],
    [1, 2, 3],
    2,
    True,
    "1",
    1.0,
    [True, 0],
    [1.0, 0],
    ["1", 0],
]
BAD_OFFSETS = [-4, 4096, "8", 1.5, True]
# The C++ IR parser holds only int lists and dict lists, so it refuses a list
# with a bool, float or str element before the lowering sees it.
SCALE_SEL_MESSAGE = "scale_sel must be two byte indices"
MIXED_LIST_MESSAGE = "list: element must be a dict"


def _scale_sel_engine_message(sel):
    if isinstance(sel, list) and any(type(e) is not int for e in sel):
        return MIXED_LIST_MESSAGE
    return SCALE_SEL_MESSAGE


def _ops(region):
    for op in region.ops:
        yield op
        for nested in op.regions:
            yield from _ops(nested)


def _set_attr(kernel, name, key, value):
    ops = [op for op in _ops(kernel.body) if op.name == name]
    assert ops
    for op in ops:
        op.attrs[key] = value
    return kernel


def _scaled_kernel(scale_sel=None):
    kernel = build_block_scaled_gemm(BlockScaledGemmSpec("opsel", 32, 32, 128))
    if scale_sel is not None:
        _set_attr(kernel, "tile.mma", "scale_sel", scale_sel)
    return kernel


def _mx16_operands(b, payload=I32):
    lanes = 32 if payload == I8 else 8
    zero = b.const_i32(0) if payload != I8 else b.trunc(b.const_i32(0), I8)
    a = b.vector_splat(zero, lanes)
    bb = b.vector_splat(zero, lanes)
    c = b.vector_splat(b.const_f32(0.0), 4)
    return a, bb, c, b.const_i32(3), b.const_i32(5)


def _mx16_kernel(scale_sel=None, payload=I32):
    b = IRBuilder("mx16")
    b.kernel.attrs["max_workgroup_size"] = 64
    b.mfma_scale_f32_16x16x128_f8f6f4(*_mx16_operands(b, payload), scale_sel=scale_sel)
    b.ret()
    return b.kernel


def _dma_kernel(variant, offset=0):
    b = IRBuilder(f"lds_dma_{variant}")
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F16, "global"))
    n = b.param("N_bytes", I32)
    lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
    rsrc = b.buffer_rsrc(x, n)
    zero = b.const_i32(0)
    if variant == "addr":
        b.async_buffer_load_lds_addr(
            rsrc, b.smem_addr_of(lds), zero, zero, dwords=1, offset=offset
        )
    else:
        b.async_buffer_load_lds(rsrc, lds, zero, zero, dwords=1, offset=offset)
    b.ret()
    return b.kernel


def _dma_op_name(variant):
    return {
        "addr": "tile.async_buffer_load_lds_addr",
        "typed": "tile.async_buffer_load_lds",
    }[variant]


def _engine():
    return pytest.importorskip("rocke_engine")


def _refused_everywhere(
    kernel, message, arch="gfx950", python_error=ValueError, engine_message=None
):
    """Every lowering (Python and C++, LLVM and HIP) refuses ``kernel``.

    ``engine_message`` overrides ``message`` for the C++ engine, for input its
    IR parser refuses before any lowering runs.
    """
    with pytest.raises(python_error, match=message):
        _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm23")
    with pytest.raises(python_error, match=message):
        lower_kernel_to_hip(kernel, arch=arch)
    engine = _engine()
    engine_message = engine_message or message
    with pytest.raises(RuntimeError, match=engine_message):
        engine.lower_serialized_ir(serialize(kernel), arch, "llvm23")
    with pytest.raises(RuntimeError, match=engine_message):
        engine.lower_serialized_ir_hip(serialize(kernel), arch)


# ----- scale_sel on a mfma_scaled catalog atom (32x32x64) -----


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("sel", [None, [0, 0], [2, 3], [3, 1]])
def test_scale_sel_reaches_opsel(flavor, sel):
    kernel = _scaled_kernel(sel)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    calls = [
        ln
        for ln in llvm.splitlines()
        if "@llvm.amdgcn.mfma.scale" in ln and "call" in ln
    ]
    sa, sb = sel or (0, 0)
    assert calls and all(
        re.search(rf", i32 {sa}, i32 %[\w.]+, i32 {sb}, i32 %[\w.]+\)$", ln)
        for ln in calls
    ), calls
    opsel = re.compile(rf"mfma_scale_f32_32x32x64_f8f6f4\(.*, {sa}, \w+, {sb}, \w+\);")
    assert opsel.search(lower_kernel_to_hip(kernel, arch="gfx950"))
    engine = _engine()
    assert engine.lower_serialized_ir(serialize(kernel), "gfx950", flavor) == llvm
    # The two HIP lowerings already print this kernel's float constants
    # differently, so the C++ one is checked for the operands alone.
    assert opsel.search(engine.lower_serialized_ir_hip(serialize(kernel), "gfx950"))


def test_scale_sel_absent_serializes_as_before():
    assert "scale_sel" not in serialize(_scaled_kernel())
    assert "scale_sel = l:[ i:2, i:3 ]" in serialize(_scaled_kernel([2, 3]))


@pytest.mark.parametrize("sel", BAD_SCALE_SELS)
def test_scale_sel_out_of_range_refused(sel):
    _refused_everywhere(
        _scaled_kernel(sel),
        SCALE_SEL_MESSAGE,
        engine_message=_scale_sel_engine_message(sel),
    )


def _unscaled_kernel(b):
    one = b.const_f32(1.0)
    return one, one, b.zero_vec_f32(16)


def test_scale_sel_on_an_unscaled_atom_refused():
    b = IRBuilder("unscaled_scale_sel")
    with pytest.raises(ValueError, match="scale_sel applies only to a scaled MFMA"):
        b.mma("mfma_f32_32x32x2_f32", *_unscaled_kernel(b), scale_sel=(1, 0))

    # A scale_sel set after building reaches the lowerings, which refuse it too.
    b = IRBuilder("unscaled_scale_sel")
    b.mma("mfma_f32_32x32x2_f32", *_unscaled_kernel(b))
    b.ret()
    kernel = _set_attr(b.kernel, "tile.mma", "scale_sel", [1, 0])
    _refused_everywhere(kernel, "scale_sel applies only to a scaled MFMA")


@pytest.mark.parametrize("sel", BAD_SCALE_SELS)
def test_the_builder_refuses_a_bad_scale_sel(sel):
    b = IRBuilder("bad_scale_sel")
    with pytest.raises(ValueError, match="scale_sel must be two byte indices"):
        b.mfma_scale_f32_16x16x128_f8f6f4(*_mx16_operands(b), scale_sel=sel)
    with pytest.raises(ValueError, match="scale_sel must be two byte indices"):
        MFMA_ATOMS[0].emit_scaled(b, *_mx16_operands(b), scale_sel=sel)
    atom = _scaled_kernel()
    (mma,) = {op.attrs["op_id"] for op in _ops(atom.body) if op.name == "tile.mma"}
    a = b.vector_splat(b.const_i32(0), 8)
    c = b.vector_splat(b.const_f32(0.0), 16)
    scale = b.const_i32(0)
    with pytest.raises(ValueError, match="scale_sel must be two byte indices"):
        b.mma(mma, a, a, c, scale, scale, scale_sel=sel)


# ----- the 16x16x128 MX atom -----


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("sel", [None, (0, 0), (1, 2), (3, 3)])
@pytest.mark.parametrize("payload", [I32, I8])
def test_mx16_scale_sel_reaches_opsel(flavor, sel, payload):
    kernel = _mx16_kernel(sel, payload)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    (call,) = [
        ln
        for ln in llvm.splitlines()
        if "call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4(" in ln
    ]
    sa, sb = sel or (0, 0)
    if flavor == "llvm23":
        tail = rf"i32 0, i32 0, i32 {sa}, i32 3, i32 {sb}, i32 5\)$"
    else:
        tail = rf"i32 0, i32 0, i32 0, i32 {sa}, i32 3, i32 {sb}, i32 5, i32 0\)$"
    assert re.search(tail, call), call
    if payload == I8:
        assert re.search(r"bitcast <32 x i8> %\S+ to <8 x i32>", llvm)
    hip = lower_kernel_to_hip(kernel, arch="gfx950")
    builtin = re.compile(
        rf"__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4\(.*, 0, 0, {sa}, \w+, {sb}, \w+\);"
    )
    assert builtin.search(hip), hip
    if payload == I8:
        assert "__builtin_bit_cast(i32x8, " in hip
    engine = _engine()
    assert engine.lower_serialized_ir(serialize(kernel), "gfx950", flavor) == llvm
    assert builtin.search(engine.lower_serialized_ir_hip(serialize(kernel), "gfx950"))


def test_mx16_scale_sel_absent_serializes_as_before():
    assert "scale_sel" not in serialize(_mx16_kernel())
    assert "scale_sel = l:[ i:1, i:2 ]" in serialize(_mx16_kernel((1, 2)))


def test_emit_scaled_passes_scale_sel_through():
    b = IRBuilder("emit_scaled")
    MFMA_ATOMS[0].emit_scaled(b, *_mx16_operands(b), scale_sel=(2, 1))
    (mma,) = [op for op in _ops(b.kernel.body) if op.name == "tile.mma"]
    assert mma.attrs == {"op_id": MX16, "scale_sel": [2, 1]}


@pytest.mark.parametrize("sel", BAD_SCALE_SELS)
def test_mx16_bad_scale_sel_refused_by_every_lowering(sel):
    kernel = _set_attr(_mx16_kernel(), "tile.mma", "scale_sel", sel)
    _refused_everywhere(
        kernel, SCALE_SEL_MESSAGE, engine_message=_scale_sel_engine_message(sel)
    )


@pytest.mark.parametrize("key", ["a_format", "b_format"])
@pytest.mark.parametrize("fmt", ["fp6e2m3", "fp4e2m1", "bf8e5m2"])
def test_mx16_format_other_than_fp8e4m3_refused(key, fmt):
    kernel = _set_attr(_mx16_kernel(), "tile.mma", key, fmt)
    _refused_everywhere(
        kernel, rf"{key} '{fmt}' is not lowered by the Python or C\+\+ engine"
    )


def _bad_mx16_operands(b, which):
    a, bb, c, sa, sb = _mx16_operands(b)
    if which == "a_f32x8":
        a = b.vector_splat(b.const_f32(0.0), 8)
    elif which == "b_short":
        bb = b.vector_splat(b.const_i32(0), 4)
    elif which == "c_f32x16":
        c = b.vector_splat(b.const_f32(0.0), 16)
    elif which == "scale_f32":
        sa = b.const_f32(1.0)
    return [a, bb, c, sa, sb]


BAD_MX16_OPERANDS = ["a_f32x8", "b_short", "c_f32x16", "scale_f32", "no_b_scale"]


@pytest.mark.parametrize("which", BAD_MX16_OPERANDS)
def test_mx16_bad_operands_refused(which):
    b = IRBuilder("mx16_bad_operands")
    operands = _bad_mx16_operands(b, which)
    if which == "no_b_scale":
        operands = operands[:4]
    message = rf"{MX16} requires A and B of 32 bytes"
    with pytest.raises(ValueError, match=message):
        b.mma(MX16, *operands)
    if which != "no_b_scale":
        with pytest.raises(ValueError, match=message):
            b.mfma_scale_f32_16x16x128_f8f6f4(*operands)

    # The same op emitted around the builder's check reaches the lowerings.
    b._op("tile.mma", operands, [VectorType(F32, 4)], attrs={"op_id": MX16})
    b.ret()
    _refused_everywhere(b.kernel, message)


def test_mx16_wrong_result_type_refused():
    b = IRBuilder("mx16_bad_result")
    b._op("tile.mma", _mx16_operands(b), [VectorType(F32, 16)], attrs={"op_id": MX16})
    b.ret()
    _refused_everywhere(b.kernel, rf"{MX16} result must be vec<f32x4>")


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_mx16_requires_gfx950(arch):
    _refused_everywhere(
        _mx16_kernel(),
        f"tile.mma {MX16} requires gfx950",
        arch=arch,
        python_error=NotImplementedError,
    )


# ----- offset on a buffer LDS DMA -----


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("variant", ["addr", "typed"])
@pytest.mark.parametrize("offset", [0, 64, 4095])
def test_lds_dma_offset_reaches_the_intrinsic(flavor, variant, offset):
    kernel = _dma_kernel(variant, offset)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    (call,) = [
        ln
        for ln in llvm.splitlines()
        if "call void @llvm.amdgcn.raw.ptr.buffer.load.lds(" in ln
    ]
    assert re.search(rf", i32 {offset}, i32 0\)", call), call
    hip = lower_kernel_to_hip(kernel, arch="gfx950")
    assert re.search(rf"_llvm_amdgcn_raw_ptr_buffer_load_lds\(.*, {offset}, 0\);", hip)
    engine = _engine()
    assert engine.lower_serialized_ir(serialize(kernel), "gfx950", flavor) == llvm
    assert engine.lower_serialized_ir_hip(serialize(kernel), "gfx950") == hip


@pytest.mark.parametrize("variant", ["addr", "typed"])
def test_lds_dma_offset_default_serializes_as_before(variant):
    assert "offset" not in serialize(_dma_kernel(variant))
    assert "offset = i:64" in serialize(_dma_kernel(variant, 64))


@pytest.mark.parametrize("variant", ["addr", "typed"])
@pytest.mark.parametrize("offset", BAD_OFFSETS)
def test_the_builder_refuses_a_bad_lds_dma_offset(variant, offset):
    with pytest.raises(ValueError, match="offset must be 0..4095 bytes"):
        _dma_kernel(variant, offset)


@pytest.mark.parametrize("variant", ["addr", "typed"])
@pytest.mark.parametrize("offset", BAD_OFFSETS)
def test_lds_dma_offset_out_of_range_refused(variant, offset):
    kernel = _set_attr(
        _dma_kernel(variant, 64), _dma_op_name(variant), "offset", offset
    )
    _refused_everywhere(kernel, "offset must be 0..4095 bytes")
