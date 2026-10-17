# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics for the E8M0 scale decoder and the MX GEMM that consumes it.

Two lanes run on a gfx942 or gfx950 device. Both lanes compile through COMGR and
through hipcc. The tests skip when no such device is visible. Set
``ROCKE_REQUIRE_GFX950=1`` to fail instead of skipping.

* Decoder: one-thread kernel, all 256 codes, checked bit for bit against the
  oracle in ``mx_scale_oracle.py``. Code 0 is the subnormal ``2^-127``. The test
  reads the f32 denormal mode from the compiled kernel descriptor and records it.
  A mode that flushes output denormals must give +0.0 for code 0.
* MX GEMM (``build_mx_gemm``): fp8e4m3 and bf8e5m2 mantissas, one to four
  K groups, scale codes 126..129 and NaN. The reference is an f32 GEMM from
  independently dequantized operands. Every mantissa is a small integer and
  every scale is a power of two, so each result is exact and the comparison is
  exact.

Scales are uniform across rows (A) and across columns (B) within a K group.
That isolates the decoder from a separate defect: the kernel loads the A scale
by the lane's input row, not by the output row of each accumulator. A test at
the end records that defect as an expected failure.

Mantissa bytes use the device's native format: OCP on gfx950, FNUZ on gfx942.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from rocke.helpers import compile_kernel
from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.helpers.spec import SignatureBuilder
from rocke.instances.common.mx_gemm import MxGemmSpec, build_mx_gemm, mx_gemm_grid
from rocke.instances.common.mx_gemm import mx_gemm_signature
from rocke.runtime.hip_module import Runtime, get_device_arch
from rocke.runtime.packing import pack_args

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mx_scale_oracle import (  # noqa: E402  (shared helper, not a test module)
    NAN_CODE,
    NUM_CODES,
    check_all_codes,
    dequantized_gemm,
    emitter,
)

SUPPORTED_ARCHES = ("gfx942", "gfx950")
ROUTES = ("comgr", "hip")
MANTISSAS = ("fp8e4m3", "bf8e5m2")
# ml_dtypes types that match the device's native fp8 / bf8 encodings.
ML_DTYPES = {
    "gfx950": {"fp8e4m3": "float8_e4m3fn", "bf8e5m2": "float8_e5m2"},
    "gfx942": {"fp8e4m3": "float8_e4m3fnuz", "bf8e5m2": "float8_e5m2fnuz"},
}
SENTINEL_BITS = 0xDEADBEEF  # a finite f32 that no test expects as a result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arch():
    target = get_device_arch(0)
    strict = os.environ.get("ROCKE_REQUIRE_GFX950") == "1"
    if target not in SUPPORTED_ARCHES:
        reason = f"needs a gfx942 or gfx950 device, found {target}"
        if strict:
            pytest.fail(reason)
        pytest.skip(reason)
    if strict and target != "gfx950":
        pytest.fail(f"ROCKE_REQUIRE_GFX950=1 needs gfx950, found {target}")
    return target


@pytest.fixture(scope="module", params=ROUTES)
def route(request):
    if request.param == "hip" and shutil.which("hipcc") is None:
        pytest.skip("the HIP route needs hipcc on PATH")
    return request.param


@pytest.fixture(scope="module")
def runtime(arch):
    return Runtime()


_ARTIFACTS: dict = {}


def compiled(kernel, arch: str, route: str):
    """Compile once per (kernel name, arch, route)."""
    key = (kernel.name, arch, route)
    if key not in _ARTIFACTS:
        compile_fn = compile_kernel if route == "comgr" else compile_kernel_via_hipcc
        _ARTIFACTS[key] = compile_fn(kernel, arch=arch)
    return _ARTIFACTS[key]


# ---------------------------------------------------------------------------
# Kernel descriptor: f32 denormal mode
# ---------------------------------------------------------------------------

_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def _device_elf(blob: bytes, arch: str) -> bytes | None:
    """The code object for ``arch``, unbundled when hipcc wrapped it."""
    if not blob.startswith(_BUNDLE_MAGIC):
        return blob
    count = struct.unpack_from("<Q", blob, len(_BUNDLE_MAGIC))[0]
    position = len(_BUNDLE_MAGIC) + 8
    for _ in range(count):
        offset, size, triple_size = struct.unpack_from("<QQQ", blob, position)
        position += 24
        triple = blob[position : position + triple_size].decode()
        position += triple_size
        if triple.endswith(arch):
            return blob[offset : offset + size]
    return None


def f32_denorm_mode(blob: bytes, kernel_name: str, arch: str) -> int | None:
    """``FLOAT_DENORM_MODE_32`` of a kernel, or ``None`` when it cannot be read.

    The mode sits in ``compute_pgm_rsrc1`` of the kernel descriptor. 3 keeps
    denormal inputs and outputs. 0 and 1 flush denormal outputs.
    """
    elf = _device_elf(blob, arch)
    if elf is None or elf[:4] != b"\x7fELF":
        return None
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize)
        for i in range(shnum)
    ]
    # (name, type, flags, addr, offset, size, link, info, align, entsize)
    for _, kind, _, _, offset, size, link, _, _, entsize in sections:
        if kind not in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
            continue
        strings = sections[link][4]
        for entry in range(size // entsize):
            name, _, _, index, value, _ = struct.unpack_from(
                "<IBBHQQ", elf, offset + entry * entsize
            )
            end = elf.index(b"\0", strings + name)
            if elf[strings + name : end].decode() != f"{kernel_name}.kd":
                continue
            section = sections[index]
            descriptor = section[4] + (value - section[3])
            rsrc1 = struct.unpack_from("<I", elf, descriptor + 48)[0]
            return (rsrc1 >> 16) & 0x3
    return None


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------


def _upload(rt, array: np.ndarray) -> int:
    array = np.ascontiguousarray(array)
    pointer = rt.alloc(array.nbytes)
    rt.memcpy_h2d(
        pointer, (ctypes.c_uint8 * array.nbytes).from_buffer_copy(array), array.nbytes
    )
    return pointer


def _download(rt, pointer: int, dtype, shape) -> np.ndarray:
    count = int(np.prod(shape))
    nbytes = count * np.dtype(dtype).itemsize
    host = (ctypes.c_uint8 * nbytes)()
    rt.memcpy_d2h(host, pointer, nbytes)
    return np.frombuffer(bytes(host), dtype=dtype).reshape(shape).copy()


def run_decoder(rt, art) -> np.ndarray:
    """Launch the one-thread decoder for each code. Return the f32 bits."""
    signature = SignatureBuilder().ptr("codes", "i8").ptr("out", "f32").build()
    module = rt.load_module(art.hsaco)
    pointers = []
    try:
        function = module.get_function(art.kernel_name)
        pointers = [_upload(rt, np.zeros(1, np.uint8)), _upload(rt, np.zeros(1, "<u4"))]
        args = pack_args(signature, {"codes": pointers[0], "out": pointers[1]})
        bits = np.empty(NUM_CODES, dtype=np.uint32)
        for code in range(NUM_CODES):
            rt.memcpy_h2d(pointers[0], (ctypes.c_uint8 * 1)(code), 1)
            rt.memcpy_h2d(pointers[1], (ctypes.c_uint32 * 1)(SENTINEL_BITS), 4)
            rt.launch(function, (1, 1, 1), (1, 1, 1), args)
            rt.sync()
            bits[code] = _download(rt, pointers[1], np.uint32, (1,))[0]
        return bits
    finally:
        for pointer in pointers:
            rt.free(pointer)
        module.unload()


def run_mx_gemm(rt, art, spec, a_bytes, a_codes, b_bytes, b_codes) -> np.ndarray:
    """Launch ``build_mx_gemm`` and return the (M, N) f32 result."""
    out = np.full((spec.M, spec.N), np.array(SENTINEL_BITS, "<u4").view("<f4"), "<f4")
    arrays = (a_bytes, a_codes.astype(np.uint8), b_bytes, b_codes.astype(np.uint8), out)
    module = rt.load_module(art.hsaco)
    pointers = []
    try:
        function = module.get_function(art.kernel_name)
        pointers = [_upload(rt, array) for array in arrays]
        values = dict(zip(("A", "AScale", "B", "BScale", "C"), pointers, strict=True))
        values.update(M=spec.M, N=spec.N, K=spec.K)
        args = pack_args(mx_gemm_signature(spec), values)
        rt.launch(function, mx_gemm_grid(spec), (spec.block_size, 1, 1), args)
        rt.sync()
        return _download(rt, pointers[-1], "<f4", out.shape)
    finally:
        for pointer in pointers:
            rt.free(pointer)
        module.unload()


def encode(arch: str, mantissa: str, values: np.ndarray):
    """Encode small integers as device fp8 bytes. Return (bytes, decoded f64)."""
    ml_dtypes = pytest.importorskip("ml_dtypes", reason="ml_dtypes gives the fp8 types")
    narrow = values.astype(getattr(ml_dtypes, ML_DTYPES[arch][mantissa]))
    decoded = narrow.astype(np.float64)
    assert np.array_equal(decoded, values), "a test value is not exact in this format"
    return narrow.view(np.uint8), decoded


def gemm_case(arch, mantissa, M, N, K, a_values, b_values, a_codes, b_codes):
    """Return the spec inputs plus the expected result for one case."""
    a_bytes, a_decoded = encode(arch, mantissa, a_values)
    b_bytes, b_decoded = encode(arch, mantissa, b_values)
    expected = dequantized_gemm(a_decoded, b_decoded, a_codes, b_codes)
    return a_bytes, b_bytes, expected


def group_codes(per_group, rows_or_cols: int, *, axis: int) -> np.ndarray:
    """Scale codes that are uniform across rows (axis 0) or columns (axis 1)."""
    codes = np.asarray(per_group, dtype=np.uint8)
    if axis == 0:  # A: (M, groups)
        return np.tile(codes, (rows_or_cols, 1))
    return np.tile(codes[:, None], (1, rows_or_cols))  # B: (groups, N)


def small_integers(rng, shape) -> np.ndarray:
    """Integers in [-4, 4]: exact in fp8e4m3 and bf8e5m2."""
    return rng.integers(-4, 5, size=shape).astype(np.float32)


def check_exact(got: np.ndarray, expected: np.ndarray) -> None:
    sentinel = np.array(SENTINEL_BITS, "<u4").view("<f4")
    assert not (got == sentinel).any(), "some outputs were never written"
    unequal = ~((got == expected) | (np.isnan(got) & np.isnan(expected)))
    assert not unequal.any(), (
        f"{int(unequal.sum())} of {got.size} outputs differ; first at "
        f"{tuple(np.argwhere(unequal)[0])}: got {got[unequal][0]!r}, "
        f"expected {expected[unequal][0]!r}"
    )


def spec_for(mantissa, M, N, K, per_input_row) -> MxGemmSpec:
    return MxGemmSpec(
        M=M, N=N, K=K, mantissa_dtype=mantissa, per_input_row=per_input_row
    )


def launch_case(rt, arch, route, mantissa, M, N, K, per_input_row, a, b, ca, cb):
    spec = spec_for(mantissa, M, N, K, per_input_row)
    art = compiled(build_mx_gemm(spec, arch=arch), arch, route)
    a_bytes, b_bytes, expected = gemm_case(arch, mantissa, M, N, K, a, b, ca, cb)
    got = run_mx_gemm(rt, art, spec, a_bytes, ca, b_bytes, cb)
    return got, expected


# ---------------------------------------------------------------------------
# Decoder kernel on the device
# ---------------------------------------------------------------------------


def test_decoder_kernel_all_codes(arch, route, runtime, record_property):
    art = compiled(emitter().build_mx_scale_decode(), arch, route)
    mode = f32_denorm_mode(art.hsaco, art.kernel_name, arch)
    record_property("f32_denorm_mode_32", mode)
    bits = run_decoder(runtime, art)
    # Mode 0 and 1 flush denormal outputs. When the mode is unreadable, accept
    # the flushed value for code 0 and record which one the device gave.
    flush = mode in (0, 1) if mode is not None else bits[0] == 0
    record_property("code0_flushed", bool(flush))
    check_all_codes(bits, flush_subnormals=flush)


# ---------------------------------------------------------------------------
# Production MX GEMM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.parametrize("mantissa", MANTISSAS)
def test_ticket_example_a128_b127_gives_64(
    arch, route, runtime, mantissa, per_input_row
):
    """All mantissas 1, A code 128, B code 127, K=32: every output is 64."""
    M = N = 16
    K = 32
    ones_a = np.ones((M, K), np.float32)
    ones_b = np.ones((K, N), np.float32)
    ca = group_codes([128], M, axis=0)
    cb = group_codes([127], N, axis=1)
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, ones_a, ones_b, ca, cb
    )
    assert (expected == 64.0).all()
    check_exact(got, expected)


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.parametrize("mantissa", MANTISSAS)
def test_uniform_scale_codes_126_to_129(arch, route, runtime, mantissa, per_input_row):
    """Every pair of A and B codes in 126..129 gives K * 2^(ca-127) * 2^(cb-127)."""
    M = N = 16
    K = 32
    ones_a = np.ones((M, K), np.float32)
    ones_b = np.ones((K, N), np.float32)
    for code_a in range(126, 130):
        for code_b in range(126, 130):
            ca = group_codes([code_a], M, axis=0)
            cb = group_codes([code_b], N, axis=1)
            got, expected = launch_case(
                runtime, arch, route, mantissa, M, N, K, per_input_row,
                ones_a, ones_b, ca, cb,
            )  # fmt: skip
            assert (expected == K * 2.0 ** (code_a - 127) * 2.0 ** (code_b - 127)).all()
            check_exact(got, expected)


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("mantissa", MANTISSAS)
def test_group_dependent_scales_over_several_k_groups(
    arch, route, runtime, mantissa, K, per_input_row
):
    """Each K group has its own A and B code. The groups use all of 126..129."""
    M = N = 32  # four CTAs
    rng = np.random.default_rng(1234 + K)
    groups = K // 32
    ca = group_codes([126, 127, 128, 129][:groups], M, axis=0)
    cb = group_codes([129, 128, 127, 126][:groups], N, axis=1)
    a = small_integers(rng, (M, K))
    b = small_integers(rng, (K, N))
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, a, b, ca, cb
    )
    check_exact(got, expected)


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.parametrize("mantissa", MANTISSAS)
def test_high_bit_codes_on_both_operands(arch, route, runtime, mantissa, per_input_row):
    """Codes 128..129 on A and on B in the same GEMM. Neither side may be zero."""
    M = N = 16
    K = 64
    rng = np.random.default_rng(99)
    ca = group_codes([129, 128], M, axis=0)
    cb = group_codes([128, 129], N, axis=1)
    a = small_integers(rng, (M, K))
    b = small_integers(rng, (K, N))
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, a, b, ca, cb
    )
    assert np.count_nonzero(expected) > 0 and np.isfinite(expected).all()
    check_exact(got, expected)


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.parametrize("mantissa", MANTISSAS)
def test_nan_scale_propagates(arch, route, runtime, mantissa, per_input_row):
    """Code 255 makes every output that sums over its group NaN."""
    M = N = 16
    K = 64
    rng = np.random.default_rng(7)
    a = small_integers(rng, (M, K))
    b = small_integers(rng, (K, N))

    # A scale NaN in one group, all rows: every output is NaN.
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, a, b,
        group_codes([NAN_CODE, 127], M, axis=0), group_codes([127, 127], N, axis=1),
    )  # fmt: skip
    assert np.isnan(expected).all()
    check_exact(got, expected)

    # B scale NaN in the second group, all columns: every output is NaN.
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, a, b,
        group_codes([127, 127], M, axis=0), group_codes([127, NAN_CODE], N, axis=1),
    )  # fmt: skip
    assert np.isnan(expected).all()
    check_exact(got, expected)

    # B scale NaN for one column only. The B scale follows the output column, so
    # exactly that column is NaN and the rest stay finite.
    cb = group_codes([127, 128], N, axis=1)
    cb[1, 5] = NAN_CODE
    got, expected = launch_case(
        runtime, arch, route, mantissa, M, N, K, per_input_row, a, b,
        group_codes([127, 127], M, axis=0), cb,
    )  # fmt: skip
    assert np.isnan(expected[:, 5]).all()
    assert np.isfinite(np.delete(expected, 5, axis=1)).all()
    check_exact(got, expected)


@pytest.mark.parametrize("per_input_row", [True, False])
@pytest.mark.xfail(
    reason=(
        "independent defect: the A scale is loaded by the lane's input row, "
        "not the output row of each accumulator, and per_input_row does not "
        "change the kernel"
    ),
    strict=False,
)
def test_row_dependent_a_scales_known_routing_defect(
    arch, route, runtime, per_input_row
):
    """Rows with different A scales. Not a decoder defect. Reported as xfail."""
    M = N = 16
    K = 32
    rng = np.random.default_rng(5)
    a = small_integers(rng, (M, K))
    b = small_integers(rng, (K, N))
    ca = (127 + np.arange(M, dtype=np.uint8) % 3)[:, None]
    cb = group_codes([127], N, axis=1)
    got, expected = launch_case(
        runtime, arch, route, "fp8e4m3", M, N, K, per_input_row, a, b, ca, cb
    )
    check_exact(got, expected)
