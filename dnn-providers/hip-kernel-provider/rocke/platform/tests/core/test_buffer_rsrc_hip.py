# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The buffer descriptor word the HIP lowering writes, per target.

``tile.buffer_rsrc`` becomes ``__builtin_amdgcn_make_buffer_rsrc`` on the HIP
route. Its ``flags`` argument is DWORD3 of the descriptor, whose format and
out-of-bounds encoding is ISA-specific: the gfx9 word makes every raw buffer
access on gfx11 / gfx12 read 0 or drop the write. So the HIP lowering takes the
word from the target's ISA backend, the same source the LLVM lowering uses.

These tests pin that word per target family in the HIP text, check it equals
the word in the LLVM IR for the same target, compare the Python HIP text with
the C++ engine's (through the native ``rocke_buffer_rsrc_hip`` test binary
named by ``ROCKE_BUFFER_RSRC_HIP_TEST``), compile the gfx11 / gfx12 source with
hipcc, and run a bounds-checked copy through the HIP route on the current GPU,
lowered for that GPU's target (or its family's generic target when the GPU has
no ISA backend of its own), so gfx9, gfx11 and gfx12 GPUs all run it. Tests
needing hipcc, the native binary or a GPU skip when it is missing.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import F16, I32, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20, lower_kernel_to_llvm

_GFX9_WORD = 0x00027000
_GFX10_PLUS_WORD = 0x31014000

# Every target the HIP lowering accepts, with the word its ISA backend owns.
# gfx1250 currently inherits the gfx12 word from its backend.
_WORD_BY_ARCH = {
    "gfx90a": _GFX9_WORD,
    "gfx942": _GFX9_WORD,
    "gfx950": _GFX9_WORD,
    "gfx1151": _GFX10_PLUS_WORD,
    "gfx11-generic": _GFX10_PLUS_WORD,
    "gfx1201": _GFX10_PLUS_WORD,
    "gfx12-generic": _GFX10_PLUS_WORD,
    "gfx1250": _GFX10_PLUS_WORD,
}

_WAVE = 64


def _copy_kernel():
    """One wave copies S to D one dword (two halves) per lane, loading through
    a descriptor bounded by ``load_bytes`` and storing through one bounded by
    ``store_bytes``. Built in the same order as the native test binary so both
    engines assign the same value names."""
    b = IRBuilder("buffer_rsrc_copy")
    src = b.param("S", PtrType(F16, "global"))
    dst = b.param("D", PtrType(F16, "global"))
    load_bytes = b.param("load_bytes", I32)
    store_bytes = b.param("store_bytes", I32)
    voffset = b.mul(b.thread_id_x(), b.const_i32(4))
    soffset = b.const_i32(0)
    value = b.buffer_load_vN_f16(b.buffer_rsrc(src, load_bytes), voffset, soffset, 1)
    b.buffer_store_vN_f16(b.buffer_rsrc(dst, store_bytes), voffset, soffset, value, 1)
    b.ret()
    return b.kernel


def _hip_flags(hip: str) -> list[int]:
    return [
        int(word, 16)
        for word in re.findall(r"/\*flags=\*/\(int\)0x([0-9A-F]{8})\);", hip)
    ]


def _llvm_flags(ll: str) -> list[int]:
    return [
        int(word)
        for word in re.findall(
            r"@llvm\.amdgcn\.make\.buffer\.rsrc\.p1\(.*, i32 (\d+)\)$", ll, re.MULTILINE
        )
    ]


@pytest.mark.parametrize("arch", sorted(_WORD_BY_ARCH))
def test_hip_descriptor_carries_the_target_word(arch):
    hip = lower_kernel_to_hip(_copy_kernel(), arch=arch)
    assert hip.count("__builtin_amdgcn_make_buffer_rsrc(") == 2
    assert _hip_flags(hip) == [_WORD_BY_ARCH[arch]] * 2


@pytest.mark.parametrize("arch", sorted(_WORD_BY_ARCH))
def test_hip_and_llvm_routes_use_the_same_word(arch):
    kernel = _copy_kernel()
    hip = lower_kernel_to_hip(kernel, arch=arch)
    ll = lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=LLVM_FLAVOR_LLVM20)
    assert _llvm_flags(ll) == _hip_flags(hip)


def test_default_arch_keeps_the_gfx950_word():
    # An omitted arch lowers for the gfx950 baseline, so existing HIP output
    # does not change.
    assert _hip_flags(lower_kernel_to_hip(_copy_kernel())) == [_GFX9_WORD] * 2


@pytest.mark.parametrize("arch", sorted(_WORD_BY_ARCH))
def test_hip_source_matches_cpp_engine(arch):
    executable = os.environ.get("ROCKE_BUFFER_RSRC_HIP_TEST")
    if not executable:
        pytest.skip("set ROCKE_BUFFER_RSRC_HIP_TEST to the built rocke_buffer_rsrc_hip")
    assert Path(executable).is_file()
    native = subprocess.run(
        [executable, "--hip", arch], check=True, capture_output=True, text=True
    ).stdout
    assert native == lower_kernel_to_hip(_copy_kernel(), arch=arch)


def _require_hipcc():
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc not found on PATH")


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
def test_gfx11_gfx12_hip_source_compiles_with_the_target_word(arch):
    """No gfx11 / gfx12 GPU is needed: hipcc compiles the HIP source for the
    target, and the device IR it emits keeps the target word on both
    descriptors."""
    from rocke.helpers.compile import (
        compile_kernel_via_hipcc,
        emit_device_llvm_ir_via_hipcc,
    )

    _require_hipcc()
    kernel = _copy_kernel()
    assert compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    device_ir = emit_device_llvm_ir_via_hipcc(kernel, arch=arch)
    words = re.findall(
        r"@llvm\.amdgcn\.make\.buffer\.rsrc\.[\w.]+\(.*, i32 (-?\d+)\)", device_ir
    )
    assert [int(w) for w in words] == [_GFX10_PLUS_WORD] * 2, device_ir


def _lowering_target(device_arch):
    """The target to lower for on a ``device_arch`` GPU: the arch itself when
    it has an ISA backend, otherwise its family's generic target (a gfx1100
    runs gfx11-generic code); None when neither has one."""
    if device_arch in _WORD_BY_ARCH:
        return device_arch
    generic = f"{device_arch[:5]}-generic"
    return generic if generic in _WORD_BY_ARCH else None


def test_hip_route_copy_is_bounds_checked_on_the_gpu():
    """A copy through the HIP route on the current GPU: lanes inside both
    bounds copy S, lanes past the load bound read 0, and lanes past the store
    bound leave D untouched. A descriptor word that disables buffer access on
    the target (the gfx9 word on gfx11 and gfx12, for example) leaves all of D
    untouched and fails the check."""
    from rocke.helpers.compile import compile_kernel_via_hipcc
    from rocke.runtime.hip_module import Runtime
    from rocke.tests.gpu import current_device_arch

    # The Runtime launches on the current device, so compile for that one.
    device_arch = current_device_arch()
    arch = _lowering_target(device_arch)
    if arch is None:
        pytest.skip(f"no ISA backend for the device's {device_arch}")
    _require_hipcc()

    load_lanes, store_lanes = 40, 56
    src_values = np.arange(1, 2 * _WAVE + 1, dtype=np.float16)
    sentinel = np.float16(-7.0)
    nbytes = src_values.nbytes

    art = compile_kernel_via_hipcc(_copy_kernel(), arch=arch)
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    src = rt.alloc(nbytes)
    dst = rt.alloc(nbytes)
    try:
        host_src = (ctypes.c_uint8 * nbytes).from_buffer_copy(src_values.tobytes())
        rt.memcpy_h2d(src, host_src, nbytes)
        host_dst = np.full(2 * _WAVE, sentinel, dtype=np.float16)
        rt.memcpy_h2d(
            dst, (ctypes.c_uint8 * nbytes).from_buffer_copy(host_dst.tobytes()), nbytes
        )
        rt.launch(
            module.get_function(art.kernel_name),
            (1, 1, 1),
            (_WAVE, 1, 1),
            struct.pack("<QQii", src, dst, 4 * load_lanes, 4 * store_lanes),
        )
        rt.sync()
        out = (ctypes.c_uint8 * nbytes)()
        rt.memcpy_d2h(out, dst, nbytes)
        got = np.frombuffer(bytes(out), dtype=np.float16)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()

    expected = np.full(2 * _WAVE, sentinel, dtype=np.float16)
    expected[: 2 * load_lanes] = src_values[: 2 * load_lanes]
    expected[2 * load_lanes : 2 * store_lanes] = 0
    np.testing.assert_array_equal(got, expected)
