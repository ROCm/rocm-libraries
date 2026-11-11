# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``scale_sel`` on a scaled ``tile.mma`` and ``offset`` on a buffer LDS DMA.

The direct compiler honours both attributes; the Python and C++ LLVM and HIP
lowerings emit them into the intrinsic's ``opsel_a`` / ``opsel_b`` and
instruction-offset operands. A kernel without them serializes and lowers
exactly as before, and every lowering refuses an out-of-range value.
"""

import re

import pytest
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

FLAVORS = ["llvm20", "llvm22", "llvm23"]


def _ops(region):
    for op in region.ops:
        yield op
        for nested in op.regions:
            yield from _ops(nested)


def _scaled_kernel(scale_sel=None):
    kernel = build_block_scaled_gemm(BlockScaledGemmSpec("opsel", 32, 32, 128))
    mmas = [op for op in _ops(kernel.body) if op.name == "tile.mma"]
    assert mmas
    if scale_sel is not None:
        for op in mmas:
            op.attrs["scale_sel"] = scale_sel
    return kernel


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


def _engine():
    return pytest.importorskip("rocke_engine")


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("sel", [None, [0, 0], [2, 3], [3, 1]])
def test_scale_sel_reaches_opsel(flavor, sel):
    kernel = _scaled_kernel(sel)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    calls = [ln for ln in llvm.splitlines() if "@llvm.amdgcn.mfma.scale" in ln and "call" in ln]
    sa, sb = sel or (0, 0)
    assert calls and all(
        re.search(rf", i32 {sa}, i32 %[\w.]+, i32 {sb}, i32 %[\w.]+\)$", ln) for ln in calls
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


@pytest.mark.parametrize("sel", [[4, 0], [0, -1], [1], [1, 2, 3], 2])
def test_scale_sel_out_of_range_refused(sel):
    kernel = _scaled_kernel(sel)
    with pytest.raises(ValueError, match="scale_sel must be two byte indices"):
        _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm23")
    with pytest.raises(ValueError, match="scale_sel must be two byte indices"):
        lower_kernel_to_hip(kernel, arch="gfx950")
    engine = _engine()
    with pytest.raises(RuntimeError, match="scale_sel must be two byte indices"):
        engine.lower_serialized_ir(serialize(kernel), "gfx950", "llvm23")
    with pytest.raises(RuntimeError, match="scale_sel must be two byte indices"):
        engine.lower_serialized_ir_hip(serialize(kernel), "gfx950")


def test_scale_sel_on_an_unscaled_atom_refused():
    b = IRBuilder("unscaled_scale_sel")
    one = b.const_f32(1.0)
    b.mma("mfma_f32_32x32x2_f32", one, one, b.zero_vec_f32(16), scale_sel=(1, 0))
    b.ret()
    kernel = b.kernel
    with pytest.raises(ValueError, match="scale_sel applies only to a scaled MFMA"):
        _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm23")
    with pytest.raises(ValueError, match="scale_sel applies only to a scaled MFMA"):
        lower_kernel_to_hip(kernel, arch="gfx950")
    engine = _engine()
    with pytest.raises(RuntimeError, match="scale_sel applies only to a scaled MFMA"):
        engine.lower_serialized_ir(serialize(kernel), "gfx950", "llvm23")
    with pytest.raises(RuntimeError, match="scale_sel applies only to a scaled MFMA"):
        engine.lower_serialized_ir_hip(serialize(kernel), "gfx950")


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("variant", ["addr", "typed"])
@pytest.mark.parametrize("offset", [0, 64, 4095])
def test_lds_dma_offset_reaches_the_intrinsic(flavor, variant, offset):
    kernel = _dma_kernel(variant, offset)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    (call,) = [ln for ln in llvm.splitlines() if "call void @llvm.amdgcn.raw.ptr.buffer.load.lds(" in ln]
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
@pytest.mark.parametrize("offset", [-4, 4096])
def test_lds_dma_offset_out_of_range_refused(variant, offset):
    kernel = _dma_kernel(variant, offset)
    with pytest.raises(ValueError, match="offset must be 0..4095 bytes"):
        _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor="llvm23")
    with pytest.raises(ValueError, match="offset must be 0..4095 bytes"):
        lower_kernel_to_hip(kernel, arch="gfx950")
    engine = _engine()
    with pytest.raises(RuntimeError, match="offset must be 0..4095 bytes"):
        engine.lower_serialized_ir(serialize(kernel), "gfx950", "llvm23")
    with pytest.raises(RuntimeError, match="offset must be 0..4095 bytes"):
        engine.lower_serialized_ir_hip(serialize(kernel), "gfx950")
