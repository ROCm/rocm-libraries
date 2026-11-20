# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the direct compiler's gfx12 code objects.

Each case builds an in-tree FP16 GEMM, compiles it with the direct compiler
(``ROCKE_DIRECT_STRICT=1``, so a refusal fails instead of falling back) and
with COMGR, launches both on the same small-integer inputs over an output
prefilled with NaN, and requires the direct output to equal the exact integer
reference and COMGR's output bit for bit.

The ``gfx1201`` cases need a gfx1201 device. The ``gfx12-generic`` cases load
a generic code object, which a gfx1200 or a gfx1201 device takes. Without the
direct library, COMGR, numpy or such a device the cases skip.
"""

from __future__ import annotations

import ctypes
import dataclasses
import struct

import pytest
from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.helpers import compile_kernel
from rocke.instances.common.gemm_universal import build_universal_gemm
from rocke.instances.gfx1201.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
from rocke.runtime import code_object
from rocke.runtime.code_object import CodeObjectError
from rocke.runtime.hip_module import HipError, Runtime, get_device_arch

# Devices that load each requested processor's code objects.
_DEVICES = {"gfx1201": ("gfx1201",), "gfx12-generic": ("gfx1200", "gfx1201")}

# Elements past the output that must stay as prefilled.
_GUARD = 64


@dataclasses.dataclass(frozen=True)
class _Case:
    arch: str
    shape: tuple[int, int, int]
    # "dispatch": the FP16 GEMM the dispatcher selects for the shape;
    # "universal": the dispatcher's gfx1201 spec built for ``arch``;
    # "wmma": the one-wave-per-tile gfx12 WMMA GEMM instance.
    kernel: str

    @property
    def id(self) -> str:
        return f"{self.arch}-{self.kernel}-" + "x".join(str(n) for n in self.shape)


_CASES = (
    _Case("gfx1201", (128, 128, 64), "dispatch"),
    _Case("gfx1201", (192, 160, 80), "dispatch"),
    # 32x32 tiles: the dispatcher's small candidate.
    _Case("gfx1201", (96, 96, 48), "dispatch"),
    _Case("gfx1201", (48, 64, 32), "wmma"),
    _Case("gfx12-generic", (128, 128, 64), "universal"),
    _Case("gfx12-generic", (48, 64, 32), "wmma"),
)


def _require_device(arch: str) -> None:
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    if arch not in code_object.direct_supported_targets():
        pytest.skip(f"the direct library does not compile for {arch}")
    try:
        device = get_device_arch()
    except (HipError, OSError) as error:
        pytest.skip(f"HIP device is unavailable: {error}")
    if device not in _DEVICES[arch]:
        pytest.skip(f"{arch} code objects need one of {_DEVICES[arch]}, found {device}")
    pytest.importorskip("numpy")


def _build(case: _Case):
    """The kernel of ``case`` with its launch grid and block."""
    m, n, _k = case.shape
    if case.kernel == "wmma":
        spec = WmmaGemmSpec(name=f"direct_gfx12_wmma_{case.arch.replace('-', '_')}")
        return build_wmma_gemm(spec, arch=case.arch), (n // 16, m // 16, 1), (32, 1, 1)
    selected = dispatch_gemm_fp16(GemmRequest(M=m, N=n, K=_k, arch="gfx1201"))
    block = (selected.spec.block_size, 1, 1)
    if case.kernel == "dispatch":
        return selected.build(), selected.grid, block
    return build_universal_gemm(selected.spec, arch=case.arch), selected.grid, block


def _launch(artifact, grid, block, a, b, shape):
    """``C`` of ``C = A @ B.T`` from ``artifact``, with its guard elements."""
    import numpy as np

    m, n, k = shape
    out = np.full(m * n + _GUARD, np.nan, dtype=np.float16)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    buffers = [rt.alloc(x.nbytes) for x in (a, b, out)]
    try:
        for dev, host in zip(buffers, (a, b, out)):
            staged = (ctypes.c_uint8 * host.nbytes).from_buffer_copy(host.tobytes())
            rt.memcpy_h2d(dev, staged, host.nbytes)
        rt.launch_blocking(
            module.get_function(artifact.kernel_name),
            grid,
            block,
            struct.pack("<QQQiii", *buffers, m, n, k),
        )
        back = (ctypes.c_uint8 * out.nbytes)()
        rt.memcpy_d2h(back, buffers[2], out.nbytes)
        return np.frombuffer(bytes(back), dtype=np.float16)
    finally:
        for dev in buffers:
            rt.free(dev)
        module.unload()


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
def test_direct_gfx12_gemm_matches_the_reference_and_comgr(case: _Case, monkeypatch) -> None:
    _require_device(case.arch)
    import numpy as np

    kernel, grid, block = _build(case)
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    direct = compile_kernel(kernel, arch=case.arch, code_object_backend="direct")
    assert direct.code_object_backend == "direct", direct.direct_fallback
    assert direct.isa == f"amdgcn-amd-amdhsa--{case.arch}"
    try:
        comgr = compile_kernel(kernel, arch=case.arch, code_object_backend="comgr")
    except (CodeObjectError, OSError) as error:
        pytest.skip(f"COMGR is unavailable: {error}")
    assert comgr.code_object_backend == "comgr"

    # Integers in -2..2: every partial sum is an integer below 2048, exact
    # in f32 and in f16, so the product has one right answer.
    m, n, k = case.shape
    rng = np.random.default_rng(1201)
    a = rng.integers(-2, 3, size=(m, k)).astype(np.float16)
    b = rng.integers(-2, 3, size=(n, k)).astype(np.float16)
    exact = a.astype(np.int64) @ b.astype(np.int64).T
    assert np.abs(exact).max() < 2048
    want = exact.astype(np.float16).reshape(-1).view(np.uint16)

    got = _launch(direct, grid, block, a, b, case.shape)
    reference = _launch(comgr, grid, block, a, b, case.shape)
    assert np.isnan(got[m * n :]).all(), "the direct kernel wrote past C"
    assert not np.isnan(got[: m * n]).any(), "the direct kernel left C unwritten"
    assert np.array_equal(got[: m * n].view(np.uint16), want)
    assert np.array_equal(got.view(np.uint16), reference.view(np.uint16))
