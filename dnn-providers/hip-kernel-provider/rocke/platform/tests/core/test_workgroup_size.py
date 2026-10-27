# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The exact ``workgroup_size`` kernel attr.

A kernel states its block size in one of two attrs: ``workgroup_size`` is the
exact block size it must be launched with, ``max_workgroup_size`` an upper
bound it may be launched below. ``KernelDef.max_workgroup_size`` is the one
accessor both engines' lowerers read, and it returns ``workgroup_size`` when
set. So stating the exact size changes no emitted byte: both engines lower a
``workgroup_size=64`` kernel exactly as a ``max_workgroup_size=64`` one. Both
verifiers refuse a kernel that sets both attrs. A launch whose
``LaunchConfig`` leaves ``block`` unset runs one wave64 block. The C++ side of
these checks is ``test_workgroup_size.cpp``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.verify import verify
from rocke.helpers.tiling.kernels.tiling_gemm_manual_demo import build_manual_gemm
from rocke.runtime.launcher import LaunchConfig
from rocke.tests.engines import lower_hip_with_both_engines, lower_with_both_engines
from rocke.tests.gpu import current_device_arch

FLAVORS = ("llvm20", "llvm22", "llvm23")

BOTH_MESSAGE = (
    "kernel sets both workgroup_size and max_workgroup_size -- set one: "
    "workgroup_size is the exact launch block size, max_workgroup_size an upper "
    "bound on it"
)


def _thread_id_kernel(**attrs):
    """One block stores ``tid + 1`` to ``Y[tid]``, so an unwritten slot stays 0."""
    b = IRBuilder("workgroup_size_probe")
    b.kernel.attrs.update(attrs)
    y = b.param("Y", PtrType(I32, "global"))
    tid = b.thread_id_x()
    b.global_store(y, tid, b.add(tid, b.const_i32(1)))
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({}, 256),
        ({"max_workgroup_size": 128}, 128),
        ({"workgroup_size": 64}, 64),
        ({"workgroup_size": 64, "max_workgroup_size": 256}, 64),
    ],
)
def test_the_accessor_reads_the_exact_size_first(attrs, expected):
    assert _thread_id_kernel(**attrs).max_workgroup_size == expected


@pytest.mark.parametrize("arch, low", [("gfx942", 64), ("gfx1201", 32)])
def test_both_engines_lower_the_exact_size_like_the_bound(arch, low):
    exact = _thread_id_kernel(workgroup_size=64)
    bound = _thread_id_kernel(max_workgroup_size=64)
    for flavor in FLAVORS:
        text = lower_with_both_engines(exact, arch, llvm_flavor=flavor)
        assert f'"amdgpu-flat-work-group-size"="{low},64"' in text, flavor
        assert text == lower_with_both_engines(bound, arch, llvm_flavor=flavor), flavor
    hip = lower_hip_with_both_engines(exact, arch)
    assert "__launch_bounds__(64)" in hip
    assert hip == lower_hip_with_both_engines(bound, arch)


def test_a_tiling_demo_kernel_lowers_the_same_with_the_exact_size():
    """The manual tiling GEMM demo states ``max_workgroup_size`` equal to its
    launch block; stating the same size as ``workgroup_size`` instead keeps
    every byte both engines emit."""
    bound, _ = build_manual_gemm(64, 64, 64, arch="gfx942")
    exact, _ = build_manual_gemm(64, 64, 64, arch="gfx942")
    exact.attrs["workgroup_size"] = exact.attrs.pop("max_workgroup_size")
    assert exact.attrs["workgroup_size"] == 64
    for flavor in FLAVORS:
        assert lower_with_both_engines(
            exact, "gfx942", llvm_flavor=flavor
        ) == lower_with_both_engines(bound, "gfx942", llvm_flavor=flavor), flavor


def test_the_verifier_refuses_both_block_size_attrs():
    both = _thread_id_kernel(workgroup_size=64, max_workgroup_size=64)
    assert BOTH_MESSAGE in [d.message for d in verify(both)]
    for one in ({"workgroup_size": 64}, {"max_workgroup_size": 64}):
        assert BOTH_MESSAGE not in [d.message for d in verify(_thread_id_kernel(**one))]


def test_a_launch_config_leaves_the_block_unset_by_default():
    assert LaunchConfig().block is None


def test_gpu_an_unset_block_launches_one_wave64():
    """On the GPU, ``__call__`` and ``bind`` with ``block`` unset each launch
    one 64-lane block of a ``workgroup_size=64`` kernel: lanes 0..63 store,
    the slots past them stay 0."""
    arch = current_device_arch()
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import sig_param
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        synchronize_and_release,
    )

    artifact = compile_kernel(_thread_id_kernel(workgroup_size=64), arch=arch)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=[sig_param("Y", "i32")],
    )
    expected = np.zeros(128, dtype=np.int32)
    expected[:64] = np.arange(1, 65, dtype=np.int32)
    rt = Runtime()
    for launch in (
        lambda values, config: launcher(values, config=config),
        lambda values, config: launcher.bind(values, config=config)(),
    ):
        host = np.zeros(128, dtype=np.int32)
        device = DeviceMem(host.nbytes)
        rt.memcpy_h2d(device.ptr(), as_u8_buffer(host), host.nbytes)
        launch({"Y": device}, LaunchConfig(grid=(1, 1, 1)))
        synchronize_and_release()
        rt.memcpy_d2h(as_u8_buffer(host), device.ptr(), host.nbytes)
        np.testing.assert_array_equal(host, expected)
