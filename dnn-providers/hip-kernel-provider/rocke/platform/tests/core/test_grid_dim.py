# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``grid_dim`` op: workgroups launched along an axis.

``grid_dim(axis, block_size)`` reads the dispatch packet's grid size for the
axis (work-items, a u32 at byte 12, 16 or 20 of
``hsa_kernel_dispatch_packet_t``) through ``llvm.amdgcn.dispatch.ptr`` and
divides it by ``block_size``, the kernel's workgroup size along that axis, when
that is above 1. HIP source reads the same packet words through
``__builtin_amdgcn_dispatch_ptr``.

Covers the LLVM IR and HIP source of both engines (the same text at every LLVM
flavor), the purity both engines give the op, the refusals of the builder and
of every lowering with their exact texts, and, when a GPU is present, on the
LLVM and the hipcc route, the value for a three-dimensional launch.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rocke.core.ir import I32, PURE_OP_NAMES, IRBuilder, PtrType
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.helpers.spec import sig_param
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)
from rocke.tests.gpu import current_device_arch, launch_with_buffers

_BLOCK = 64


def _grid_dim_kernel(name):
    """Y[0..2] = the workgroups launched along x (block_size 64), y and z."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = _BLOCK
    y = b.param("Y", PtrType(I32, "global"))
    gx = b.grid_dim("x", _BLOCK)
    gy = b.grid_dim("y")
    gz = b.grid_dim("z", 1)
    b.global_store(y, b.const_i32(0), gx, align=4)
    b.global_store(y, b.const_i32(1), gy, align=4)
    b.global_store(y, b.const_i32(2), gz, align=4)
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------
# LLVM IR and HIP source
# --------------------------------------------------------------------------


def test_both_engines_read_the_dispatch_packet_grid_size():
    kernel = _grid_dim_kernel("grid_dim_llvm")
    for flavor in LLVM_FLAVORS:
        text = lower_with_both_engines(kernel, "gfx942", llvm_flavor=flavor)
        assert (
            text.count("declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()\n") == 1
        ), flavor
        assert (
            text.count(" = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()") == 3
        ), flavor
        for offset in (12, 16, 20):
            assert text.count(f".dp, i64 {offset}\n") == 1, (flavor, offset)
        # Only the x axis divides: y and z have block_size 1.
        assert text.count(" = udiv i32 ") == 1, flavor
        assert text.count(f".gs, {_BLOCK}\n") == 1, flavor


def test_both_engines_emit_the_hip_dispatch_packet_read():
    hip = lower_hip_with_both_engines(_grid_dim_kernel("grid_dim_hip"), "gfx942")
    packet = "((const unsigned int*)__builtin_amdgcn_dispatch_ptr())"
    assert hip.count(f"(int)({packet}[3] / {_BLOCK}u);") == 1
    assert hip.count(f"(int){packet}[4];") == 1
    assert hip.count(f"(int){packet}[5];") == 1


def test_grid_dim_is_pure():
    # The C++ purity table pins the same answer (tests/core/test_opcode_values.cpp).
    assert "gpu.grid_dim" in PURE_OP_NAMES


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis, block_size, message",
    [
        ("w", 1, "grid_dim axis must be x, y, or z, got 'w'"),
        (None, 1, "grid_dim axis must be x, y, or z, got None"),
        ("x", 0, "grid_dim block_size must be >= 1, got 0"),
    ],
    ids=["axis_w", "axis_none", "block_size_0"],
)
def test_builder_refuses_a_bad_axis_or_block_size(axis, block_size, message):
    b = IRBuilder("grid_dim_bad")
    with pytest.raises(ValueError, match=message):
        b.grid_dim(axis, block_size)


def _raw_grid_dim_kernel():
    """A gpu.grid_dim on axis 'w', written past the builder (as serialized IR
    can reach the lowerers)."""
    b = IRBuilder("grid_dim_raw")
    y = b.param("Y", PtrType(I32, "global"))
    r = b._op(
        "gpu.grid_dim", attrs={"axis": "w", "block_size": 1}, result_types=[I32]
    ).result
    b.global_store(y, b.const_i32(0), r, align=4)
    b.ret()
    return b.kernel


def test_every_lowering_refuses_a_bad_axis():
    kernel = _raw_grid_dim_kernel()
    message = "grid_dim axis must be x, y, or z, got 'w'"
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, message)


# --------------------------------------------------------------------------
# GPU: a three-dimensional launch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_reads_the_launched_grid(route):
    arch = current_device_arch()
    if route == "hipcc" and shutil.which("hipcc") is None:
        pytest.skip("hipcc not found")
    kernel = _grid_dim_kernel(f"grid_dim_gpu_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        artifact = compile_kernel(kernel, arch=arch)
    else:
        from rocke.helpers.compile import compile_kernel_via_hipcc

        artifact = compile_kernel_via_hipcc(kernel, arch=arch)
    grid = (5, 3, 2)
    got = np.zeros(3, dtype=np.int32)
    launch_with_buffers(
        artifact,
        [sig_param("Y", "i32")],
        {"Y": got},
        grid=grid,
        block=(_BLOCK, 1, 1),
    )
    assert got.tolist() == list(grid)
