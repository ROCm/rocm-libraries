# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check, time and host-build the gfx1250 MX GEMM (``mx_gemm``).

Every configuration is the cartesian product of the ``--tile``,
``--wave-group``, ``--cluster``, ``--a-dtype`` / ``--b-dtype`` (``fp4``,
``fp6``, ``fp8``, ``bf8``) and tuning options (each takes a comma list). For each
one: compile, report code-object resources (VGPR, SGPR, spills, scratch,
LDS) and the WMMA count, require bit-exact output at ``beta = 0`` and
``beta = 0.5`` on two input sets (``--check-data``: 0 / +1 over all of K,
and many A/B values in a few K-steps), then time the ``beta = 0`` path, on
hipBLASLt's MAF benchmark data by default (``--timing-inputs``).
``batched_us`` and ``event_batch_us`` are per-launch times over ``--batch``
back-to-back launches: ``batched_us`` by the host clock around ``launch``
calls, as ``mxfp4_tdm_gemm_variants_bench`` does, and ``event_batch_us`` by
GPU events around launches whose arguments are built once, as CK Tile times
kernels. ``enqueue_us`` is the host time per enqueue of that batch:
``event_batch_us`` is a GPU time only where it is clearly above
``enqueue_us``; where the two are close, the GPU waited for the host.
``median_us`` is the median of ``--iters`` single event-timed launches.
TFLOP/s and ``speedup_vs_base`` come from ``event_batch_us``
(``median_us`` with ``--batch 0``). ``--base`` also times the Step 6.14
kernel of ``mxfp4_tdm_gemm`` on the same problem before and after the
configurations, and reports each configuration's speedup over the mean of
the two.

``--hold-seconds`` keeps the timed launch running back to back for that
long after the timing, so that clocks and power can be sampled from outside
the process; ``hold_start_unix`` and ``hold_end_unix`` (host clock, seconds
since the epoch) bound that window and ``hold_launches`` counts its launches.

``--host-only`` compiles and reports resources without touching a device;
``--check-only`` skips the timing. Absolute timings must not be committed
to the repo; use ``--out`` with a path outside the tree.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import functools
import itertools
import re
import statistics
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ....examples.common.scaled_gemm_reference import decode_e8m0, pack_fp4_codes
from ....helpers.compile import compile_kernel
from ....instances.gfx1250.mx_gemm import (
    LdsPad,
    MxGemmSpec,
    build_mx_gemm,
    mx_gemm_grid,
    validate,
)
from ....instances.gfx1250.mxfp4_tdm_gemm import (
    build_mxfp4_tdm_gemm,
    mxfp4_tdm_gemm_grid,
)
from ....instances.gfx1250.mxfp4_tdm_gemm_variants import variant_spec

# Private names: _Launcher.prepare_launch builds the cluster launch of
# Runtime._enqueue, as Runtime.prepare_launch has no cluster form.
from ....runtime.hip_module import (
    _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
    HIP_LAUNCH_PARAM_BUFFER_POINTER,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,
    HIP_LAUNCH_PARAM_END,
    HipError,
    _HipLaunchAttribute,
    _HipLaunchConfig,
    _hipDrvLaunchKernelEx,
    check_launch_cluster,
    current_device,
    device_supports_cluster_launch,
)
from ....runtime.hip_module import _check as _hip_check
from .mxfp4_tdm_gemm import (
    YAML_TILE,
    _u8_buffer,
    make_inputs,
    swizzle_mx_scales_gfx1250,
)
from .mxfp4_tdm_gemm_variants_bench import _resources, tensile_timing_inputs

BASE_VARIANT = "step6_14_mxfp4_gemm"
YAML_PROBLEM = (4096, 4096, 65536)
SMOKE_PROBLEM = (1024, 1024, 2048)
# hipBLASLt's MXFP8 benchmark problems (mxf8_tn_maf.yaml, mxf8_tn_mab.yaml).
PROBLEMS = {
    "smoke": SMOKE_PROBLEM,
    "yaml": YAML_PROBLEM,
    "maf": (4096, 4096, 65536),
    "mab": (65536, 16, 32768),
}
_FP8_MAX = 448.0
_E2M1 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)
# Mantissa bits of the 8-bit A/B types (E4M3, E5M2).
_MANTISSA_BITS = {"fp8": 3, "bf8": 2}
_WMMA_K = 128
CHECK_DATA = ("exact", "sparse")
# 128-K steps that hold data in the sparse check. FP4 products are multiples
# of 2^-6 (E2M1 0.5, E8M0 2^-2) up to 144 (6 * 6 * 2 * 2); 8-bit ones are
# multiples of 2^-10 (2^-4 values, E8M0 2^-1) up to 3.52 (1.875^2); FP6 ones
# (all 64 E2M3 codes, A scales 2^-1 or 2^0, B scales 2^0) multiples of 2^-7
# (1/8 * 1/8 * 2^-1) up to 56.25 (7.5^2). Every FP32 partial sum is exact
# while SPARSE_STEPS * 128 * max / multiple < 2^24 (FP4 12 * 128 * 144 * 2^6,
# 8-bit 12 * 128 * 3.52 * 2^10, FP6 12 * 128 * 56.25 * 2^7).
SPARSE_STEPS = 12
SPARSE_ALPHA = 2.0**-6

FIELDS = [
    "config",
    "kernel",
    "check",
    "vgpr",
    "sgpr",
    "spills",
    "scratch",
    "lds",
    "wmma",
    "loop_scratch",
    "batched_us",
    "event_batch_us",
    "enqueue_us",
    "median_us",
    "tflops",
    "speedup_vs_base",
    "hold_start_unix",
    "hold_end_unix",
    "hold_launches",
]
# Target device time per synchronized group of launches in the hold window.
_HOLD_GROUP_SECONDS = 0.02


def _out_numpy_dtype(out_dtype: str):
    import ml_dtypes

    return {
        "fp8e4m3": ml_dtypes.float8_e4m3fn,
        "fp16": np.float16,
        "bf16": ml_dtypes.bfloat16,
    }[out_dtype]


def _ml_dtype(dtype: str):
    """The ml_dtypes type of an FP6 or 8-bit A/B type; FP6 holds one code
    per byte, in its low six bits."""
    import ml_dtypes

    return {
        "fp6": ml_dtypes.float6_e2m3fn,
        "fp8": ml_dtypes.float8_e4m3fn,
        "bf8": ml_dtypes.float8_e5m2,
    }[dtype]


@functools.cache
def decode_table(dtype: str) -> np.ndarray:
    """FP32 value of every A/B code of ``dtype`` (``fp4``, ``fp6``, ``fp8``,
    ``bf8``)."""
    if dtype == "fp4":
        return _E2M1
    count = 64 if dtype == "fp6" else 256
    return np.arange(count, dtype=np.uint8).view(_ml_dtype(dtype)).astype(np.float32)


def _ml_codes(values: np.ndarray, dtype: str) -> np.ndarray:
    """Codes of exactly representable ``values``."""
    return values.astype(_ml_dtype(dtype)).view(np.uint8)


_PACK_ROWS = 1024


def pack_fp6_rows(codes: np.ndarray) -> np.ndarray:
    """Row-major ``[rows][K]`` six-bit codes as ``[rows][3K/4]`` bytes: code
    ``k`` at bit ``6k`` of its row, least significant bits first (four codes
    in three bytes). ``pack_fp6_codes``' layout, vectorized and in blocks of
    rows to bound the temporaries."""
    rows, k = codes.shape
    if codes.dtype != np.uint8 or k % 4 or (codes > 63).any():
        raise ValueError("FP6 codes must be uint8 below 64, K a multiple of 4")
    out = np.empty((rows, k // 4, 3), dtype=np.uint8)
    for first in range(0, rows, _PACK_ROWS):
        part = codes[first : first + _PACK_ROWS].reshape(-1, k // 4, 4)
        word = part[..., 0].astype(np.uint32)
        for i in (1, 2, 3):
            word |= part[..., i].astype(np.uint32) << (6 * i)
        for byte in range(3):
            out[first : first + _PACK_ROWS, :, byte] = word >> (8 * byte)
    return out.reshape(rows, k // 4 * 3)


def _device_operand(codes: np.ndarray, dtype: str) -> np.ndarray:
    """Row-major ``[rows][K]`` codes as the kernel reads them."""
    if dtype == "fp4":
        return pack_fp4_codes(codes)
    if dtype == "fp6":
        return pack_fp6_rows(codes)
    return np.ascontiguousarray(codes)


def check_inputs(spec: MxGemmSpec, *, signed: bool = False):
    """Exactly representable inputs (0 and +1, or +-1 with ``signed``) and C
    in ``spec.out_dtype``, in ``make_inputs``' canonical layout."""
    inputs = list(make_inputs(spec, signed=signed))
    inputs[4] = inputs[4].astype(np.float32).astype(_out_numpy_dtype(spec.out_dtype))
    for index, dtype in enumerate((spec.a_dtype, spec.b_dtype)):
        if dtype != "fp4":
            codes = _ml_codes(_E2M1[inputs[5 + index]], dtype)
            inputs[index] = _device_operand(codes, dtype)
            inputs[5 + index] = codes
    return tuple(inputs)


def sparse_steps(spec: MxGemmSpec) -> np.ndarray:
    """The 128-K steps of the sparse check: the first and last, both sides
    of the first and last tile boundaries, then random steps up to
    ``SPARSE_STEPS``."""
    steps = -(-spec.K // _WMMA_K)
    per_tile = spec.tile_k // _WMMA_K
    last_tile = (steps - 1) // per_tile * per_tile
    fixed = {0, steps - 1, per_tile - 1, per_tile, last_tile - 1, last_tile}
    chosen = sorted(s for s in fixed if 0 <= s < steps)
    rest = np.setdiff1d(np.arange(steps), chosen)
    extra = max(0, min(SPARSE_STEPS - len(chosen), len(rest)))
    rng = np.random.default_rng(0x5BA5)
    return np.union1d(chosen, rng.choice(rest, size=extra, replace=False))


def _sparse_codes(rng, shape, dtype: str) -> np.ndarray:
    """FP4: all 16 E2M1 codes. FP6: all 64 E2M3 codes. 8-bit: +-(1 + m /
    2^mantissa_bits) * 2^e with e in {-1, 0}, every mantissa: multiples of
    2^-4 up to 1.875."""
    if dtype == "fp4":
        return rng.integers(0, 16, shape, dtype=np.uint8)
    if dtype == "fp6":
        return rng.integers(0, 64, shape, dtype=np.uint8)
    bits = _MANTISSA_BITS[dtype]
    mantissa = rng.integers(0, 1 << bits, shape).astype(np.float32)
    values = (1 + mantissa / (1 << bits)) * np.exp2(rng.integers(-1, 1, shape))
    values *= rng.choice(np.array([-1.0, 1.0], dtype=np.float32), shape)
    return _ml_codes(values.astype(np.float32), dtype)


def sparse_inputs(spec: MxGemmSpec):
    """Random codes (``_sparse_codes``) and E8M0 scales (FP4 2^-2..2^1,
    8-bit 2^-1..2^0, FP6 A 2^-1..2^0 and B 2^0) in the ``sparse_steps``
    K-steps, zero elsewhere; C uniform in [-2, 2]. The FP32 sums are exact in
    any order, so the check stays bit-exact at any K. ``make_inputs``'
    layout: A and B as the kernel reads them, scales, C, then the codes."""
    rng = np.random.default_rng(0x5BA6)
    k_index = np.concatenate(
        [
            np.arange(s * _WMMA_K, min(spec.K, (s + 1) * _WMMA_K))
            for s in sparse_steps(spec)
        ]
    )
    codes = []
    for rows, dtype in ((spec.M, spec.a_dtype), (spec.N, spec.b_dtype)):
        full = np.zeros((rows, spec.K), dtype=np.uint8)
        full[:, k_index] = _sparse_codes(rng, (rows, len(k_index)), dtype)
        codes.append(full)
    groups = spec.K // spec.block_k
    low, high = (125, 129) if spec.a_dtype == "fp4" else (126, 128)
    a_scale = rng.integers(low, high, (spec.M, groups), dtype=np.uint8)
    if spec.b_dtype == "fp6":
        low, high = 127, 128
    b_scale = rng.integers(low, high, (groups, spec.N), dtype=np.uint8)
    c = rng.uniform(-2.0, 2.0, (spec.M, spec.N)).astype(np.float32)
    return (
        _device_operand(codes[0], spec.a_dtype),
        _device_operand(codes[1], spec.b_dtype),
        a_scale,
        b_scale,
        c.astype(_out_numpy_dtype(spec.out_dtype)),
        *codes,
    )


def timed_inputs(spec: MxGemmSpec, seed: int = 0x32F4):
    """A, B and scales of the timed launches: FP4 as the Tensile client's
    YAML run (``tensile_timing_inputs``); FP6 and 8-bit random codes with
    the exponent MSB clear (8-bit: finite and below 2 in magnitude; FP6:
    0 to 1.875) and E8M0 scales 2^-3..2^3."""
    if spec.a_dtype == "fp4":
        return tensile_timing_inputs(spec, seed)
    rng = np.random.default_rng(seed)
    fp6 = spec.a_dtype == "fp6"
    operands = []
    for rows in (spec.M, spec.N):
        codes = rng.integers(0, 64 if fp6 else 256, (rows, spec.K), dtype=np.uint8)
        np.bitwise_and(codes, 0x2F if fp6 else 0xBF, out=codes)
        operands.append(pack_fp6_rows(codes) if fp6 else codes)
    groups = spec.K // spec.block_k
    return (
        *operands,
        rng.integers(124, 131, (spec.M, groups), dtype=np.uint8),
        rng.integers(124, 131, (groups, spec.N), dtype=np.uint8),
    )


def _pad_scale_groups(scales_mn_k: np.ndarray) -> np.ndarray:
    """Zero-pad ``[MN][K/32]`` scales to whole 128-K steps, as hipBLASLt's
    ``preSwizzleScalesGFX1250`` does when 128 does not divide K."""
    pad = -scales_mn_k.shape[1] % 4
    return np.pad(scales_mn_k, ((0, 0), (0, pad))) if pad else scales_mn_k


def device_inputs(inputs) -> tuple[np.ndarray, ...]:
    """A, B, A_scale, B_scale, C with the scales swizzled ``[ceil(K/128)][MN][4]``."""
    a, b, a_scale, b_scale, c = inputs[:5]
    return (
        a,
        b,
        swizzle_mx_scales_gfx1250(_pad_scale_groups(a_scale)),
        swizzle_mx_scales_gfx1250(_pad_scale_groups(np.ascontiguousarray(b_scale.T))),
        c,
    )


def reference(
    spec: MxGemmSpec, inputs, *, alpha, beta, rows=None, cols=None
) -> np.ndarray:
    """FP32 grouped-scale reference rounded to ``spec.out_dtype`` the way the
    kernel stores it (FP8 saturating, FP16/BF16 round to nearest even). It is
    exact for ``check_inputs`` and ``sparse_inputs``: their FP32 partial sums
    are exact. All-zero K groups of A are skipped.

    ``rows`` and ``cols`` (index arrays) restrict it to that block of D."""
    _, _, a_scale, b_scale, c, a_codes, b_codes = inputs
    if rows is not None:
        a_codes, a_scale, c = a_codes[rows], a_scale[rows], c[rows]
    if cols is not None:
        b_codes, b_scale, c = b_codes[cols], b_scale[:, cols], c[:, cols]
    sa = decode_e8m0(a_scale).astype(np.float32)
    sb = decode_e8m0(b_scale).astype(np.float32)
    a_values, b_values = decode_table(spec.a_dtype), decode_table(spec.b_dtype)
    out = np.zeros((a_codes.shape[0], b_codes.shape[0]), dtype=np.float32)
    for group in range(spec.K // spec.block_k):
        begin, end = group * spec.block_k, (group + 1) * spec.block_k
        a_group = a_codes[:, begin:end]
        if not a_group.any():
            continue
        out += (a_values[a_group] * sa[:, group, None]) @ (
            b_values[b_codes[:, begin:end]] * sb[group, :, None]
        ).T
    out = np.float32(alpha) * out + np.float32(beta) * c.astype(np.float32)
    if spec.out_dtype == "fp8e4m3":
        out = np.where(np.isinf(out), np.float32(np.nan), out)
        out = np.clip(out, -_FP8_MAX, _FP8_MAX)
    return out.astype(_out_numpy_dtype(spec.out_dtype))


def block_indices(
    extent: int, tile: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """The first and last tile plus ``count`` random indices, sorted."""
    edges = np.concatenate(
        [np.arange(min(tile, extent)), np.arange(max(0, extent - tile), extent)]
    )
    rest = rng.choice(extent, size=min(count, extent), replace=False)
    return np.unique(np.concatenate([edges, rest]))


def verify_block(spec: MxGemmSpec, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of D that ``--verify block`` checks."""
    rng = np.random.default_rng(0xB10C)
    return (
        block_indices(spec.M, spec.tile_m, count, rng),
        block_indices(spec.N, spec.tile_n, count, rng),
    )


def _check(spec, got, expected, *, verify: str, samples: int) -> bool:
    if verify in ("full", "block"):
        return bool(np.array_equal(got, expected))
    rng = np.random.default_rng(0x32F4)
    count = min(samples, spec.M * spec.N)
    flat = rng.choice(spec.M * spec.N, size=count, replace=False)
    rows, cols = np.divmod(flat, spec.N)
    return bool(np.array_equal(got[rows, cols], expected[rows, cols]))


class _Launcher:
    """A compiled kernel with device buffers for A, B, scales, C and D."""

    # The workgroup shape of every launch, checked or timed.
    _BLOCK = (128, 1, 1)

    def __init__(self, runtime, artifact, grid, cluster, arrays, out_bytes: int):
        self.runtime = runtime
        self.module = runtime.load_module(artifact.hsaco)
        self.function = self.module.get_function(artifact.kernel_name)
        self.grid, self.cluster = grid, cluster
        self.out_bytes = out_bytes
        self.pointers = []
        for array in (*arrays, None):
            nbytes = out_bytes if array is None else array.nbytes
            pointer = runtime.alloc(nbytes)
            self.pointers.append(pointer)
            if array is not None:
                runtime.memcpy_h2d(pointer, _u8_buffer(array), nbytes)

    def upload(self, index: int, array: np.ndarray) -> None:
        self.runtime.memcpy_h2d(self.pointers[index], _u8_buffer(array), array.nbytes)

    def arguments(self, m, n, k, alpha, beta) -> bytes:
        return struct.pack(
            "<QQQQQQiiiff", *self.pointers, m, n, k, float(alpha), float(beta)
        )

    def launch(self, arguments: bytes) -> None:
        self.runtime.launch(
            self.function, self.grid, self._BLOCK, arguments, cluster=self.cluster
        )

    def prepare_launch(self, arguments: bytes) -> Callable[[], None]:
        """A zero-argument enqueue of this launch with every ctypes object
        built once and kept alive by the callable, for timed loops:
        ``launch`` rebuilds them on each call. It owns the argument buffer:
        hold it until its launches finish."""
        grid, block = self.grid, self._BLOCK
        if self.cluster is None:
            return self.runtime.prepare_launch(self.function, grid, block, arguments)
        # Runtime.prepare_launch has no cluster form. This is the cluster
        # branch of Runtime._enqueue (the same checks, attribute and config)
        # with every ctypes object built once; it must change with that branch.
        shape = check_launch_cluster(grid, self.cluster)
        device = current_device()
        if device_supports_cluster_launch(device) is False:
            raise HipError(
                f"device {device} does not support cluster launch (cluster {shape})"
            )
        args_buf = (ctypes.c_ubyte * len(arguments)).from_buffer_copy(arguments)
        size_buf = ctypes.c_size_t(len(arguments))
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(args_buf, ctypes.c_void_p),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(size_buf), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END,
        )
        attr = _HipLaunchAttribute()
        attr.id = _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.val.clusterDim.x, attr.val.clusterDim.y, attr.val.clusterDim.z = shape
        config = _HipLaunchConfig(
            gridDimX=grid[0],
            gridDimY=grid[1],
            gridDimZ=grid[2],
            blockDimX=block[0],
            blockDimY=block[1],
            blockDimZ=block[2],
            sharedMemBytes=0,
            hStream=0,
            attrs=ctypes.pointer(attr),
            numAttrs=1,
        )
        config_ptr = ctypes.pointer(config)
        function = self.function
        keep = (args_buf, size_buf, extra, attr, config, config_ptr)

        def enqueue(_keep=keep) -> None:
            _hip_check(
                _hipDrvLaunchKernelEx(config_ptr, function, None, extra),
                "hipDrvLaunchKernelEx",
            )

        return enqueue

    def output(self) -> bytes:
        self.runtime.sync()
        host = (ctypes.c_uint8 * self.out_bytes)()
        self.runtime.memcpy_d2h(host, self.pointers[-1], self.out_bytes)
        return bytes(host)

    def close(self) -> None:
        # hipFree synchronizes the device, so the module is unloaded only
        # after every launch that uses it has finished.
        for pointer in self.pointers:
            self.runtime.free(pointer)
        self.module.unload()


class _Times(NamedTuple):
    """Microseconds per launch; see the module docstring. ``batched``,
    ``event_batch`` and ``enqueue`` are ``None`` when ``--batch`` is 0,
    ``median`` when ``--iters`` is 0, and all of them when nothing was
    timed."""

    batched: float | None = None
    event_batch: float | None = None
    enqueue: float | None = None
    median: float | None = None


def _time(launcher: _Launcher, arguments: bytes, *, warmup, iters, batch) -> _Times:
    """Time the launch of ``arguments``; see the module docstring. When a
    launch or a timing call raises an ``Exception``, the launches already
    queued are waited for before it propagates, so the caller may unload the
    module and free the buffers."""
    runtime = launcher.runtime
    batched_us = event_batch_us = enqueue_us = None
    times = []
    try:
        for _ in range(max(warmup, 1)):
            launcher.launch(arguments)
        runtime.sync()
        if batch > 0:
            begin = time.perf_counter()
            for _ in range(batch):
                launcher.launch(arguments)
            runtime.sync()
            batched_us = (time.perf_counter() - begin) * 1.0e6 / batch
            enqueue = launcher.prepare_launch(arguments)
            start, end = runtime.event(), runtime.event()
            try:
                start.record()
                begin = time.perf_counter()
                for _ in range(batch):
                    enqueue()
                enqueue_us = (time.perf_counter() - begin) * 1.0e6 / batch
                end.record()
                end.synchronize()
                event_batch_us = start.elapsed_to(end) * 1.0e3 / batch
            finally:
                start.destroy()
                end.destroy()
        for _ in range(iters):
            start, end = runtime.event(), runtime.event()
            start.record()
            launcher.launch(arguments)
            end.record()
            runtime.sync()
            times.append(start.elapsed_to(end) * 1.0e3)
            start.destroy()
            end.destroy()
    except Exception as exc:
        # The queued launches still use the module and the buffers that the
        # caller releases next. A failing wait is not raised: the error that
        # propagates names the call that failed.
        try:
            runtime.sync()
        except Exception as drain_exc:  # noqa: BLE001 - reported on exc
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(f"waiting for the device after this error failed: {drain_exc}")
        raise
    median_us = statistics.median(times) if times else None
    return _Times(batched_us, event_batch_us, enqueue_us, median_us)


class _Hold(NamedTuple):
    """The ``--hold-seconds`` window: host clock at its start and end
    (seconds since the epoch) and the launches enqueued in it."""

    start: float | None = None
    end: float | None = None
    launches: int = 0


def _hold(launcher: _Launcher, arguments: bytes, seconds: float, launch_us) -> _Hold:
    """Enqueue the prepared launch of ``arguments`` back to back for
    ``seconds``, in groups of about ``_HOLD_GROUP_SECONDS`` of device time
    (``launch_us`` per launch), synchronizing after each group so the window
    ends on time. Nothing is timed."""
    if seconds <= 0:
        return _Hold()
    runtime = launcher.runtime
    group = max(1, int(_HOLD_GROUP_SECONDS * 1.0e6 / launch_us)) if launch_us else 1
    enqueue = launcher.prepare_launch(arguments)
    launches = 0
    start = time.time()
    try:
        while time.time() - start < seconds:
            for _ in range(group):
                enqueue()
            launches += group
            runtime.sync()
    except Exception:
        runtime.sync()
        raise
    return _Hold(start, time.time(), launches)


def _wmma_count(llvm_text: str | None) -> str:
    if not llvm_text:
        return ""
    return str(len(re.findall(r"call [^\n]*@llvm\.amdgcn\.wmma\.scale", llvm_text)))


_SYMBOL = re.compile(r"^([0-9a-f]+) <[^>]+>:$", re.MULTILINE)
_ADDRESS = re.compile(r"//\s*([0-9A-F]+):")
_TARGET = re.compile(r"<[^+>]+\+0x([0-9a-f]+)>")


def loop_scratch_ops(objdump_text: str) -> int | None:
    """Scratch instructions inside the longest backward branch (the main
    loop) of a disassembly, or ``None`` if it has no backward branch."""
    symbol = _SYMBOL.search(objdump_text)
    if not symbol:
        return None
    base = int(symbol.group(1), 16)
    loop = None
    addressed = []
    for line in objdump_text.splitlines():
        address = _ADDRESS.search(line)
        if not address:
            continue
        pc = int(address.group(1), 16)
        addressed.append((pc, line))
        target = _TARGET.search(line)
        if target and "branch" in line:
            begin = base + int(target.group(1), 16)
            if begin <= pc and (loop is None or pc - begin > loop[1] - loop[0]):
                loop = (begin, pc)
    if loop is None:
        return None
    return sum(
        1 for pc, line in addressed if loop[0] <= pc <= loop[1] and "scratch_" in line
    )


def _loop_scratch(hsaco: bytes) -> str:
    import tempfile

    from ....analysis.isa import analyze_hsaco

    try:
        with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
            handle.write(hsaco)
            handle.flush()
            text = analyze_hsaco(Path(handle.name), keep_text=True).objdump_text
    except Exception as exc:  # noqa: BLE001 - resources are best effort
        print(f"warning: disassembly failed: {exc}", file=sys.stderr)
        return ""
    count = loop_scratch_ops(text)
    return "" if count is None else str(count)


def _run(spec, artifact, grid, cluster, *, args, runtime, timing_inputs: bool):
    """Check at every ``--check-betas`` value, then time, then hold the timed
    launch for ``--hold-seconds``. Returns (check, times, hold)."""
    m, n, k = spec.M, spec.N, spec.K
    alpha = np.float32(1.0 / max(k, 256))
    inputs = check_inputs(spec)
    out_dtype = _out_numpy_dtype(spec.out_dtype)
    out_bytes = m * n * np.dtype(out_dtype).itemsize
    launcher = _Launcher(
        runtime, artifact, grid, cluster, device_inputs(inputs), out_bytes
    )
    rows = cols = None
    if args.verify == "block":
        rows, cols = verify_block(spec, args.samples)
    try:
        failed = []
        for kind in args.check_data:
            data, data_alpha = inputs, alpha
            if kind == "sparse":
                data, data_alpha = sparse_inputs(spec), np.float32(SPARSE_ALPHA)
                for index, array in enumerate(device_inputs(data)):
                    launcher.upload(index, array)
            for beta in args.check_betas:
                arguments = launcher.arguments(m, n, k, data_alpha, beta)
                # beta 0 goes through the enqueue event_batch_us times, held
                # until output() has synchronized; other betas through launch.
                if beta == 0:
                    enqueue = launcher.prepare_launch(arguments)
                else:
                    enqueue = functools.partial(launcher.launch, arguments)
                enqueue()
                got = np.frombuffer(launcher.output(), dtype=np.uint8)
                got = got.reshape(m, n, -1)
                expected = reference(
                    spec, data, alpha=data_alpha, beta=beta, rows=rows, cols=cols
                )
                if rows is not None:
                    got = got[np.ix_(rows, cols)]
                expected = expected.view(np.uint8).reshape(got.shape)
                if not _check(
                    spec, got, expected, verify=args.verify, samples=args.samples
                ):
                    failed.append(f"{kind} beta={beta:g}")
            if kind == "sparse":
                for index, array in enumerate(device_inputs(inputs)):
                    launcher.upload(index, array)
        if failed:
            return "FAIL:" + ",".join(failed), _Times(), _Hold()
        if args.check_only:
            return "PASS", _Times(), _Hold()
        timed = launcher.arguments(m, n, k, alpha, 0.0)
        if timing_inputs:
            timing = device_inputs((*timed_inputs(spec), inputs[4]))
            for index, array in enumerate(timing[:4]):
                launcher.upload(index, array)
            timed = launcher.arguments(m, n, k, 1.0, 0.0)
        times = _time(
            launcher, timed, warmup=args.warmup, iters=args.iters, batch=args.batch
        )
        launch_us = times.event_batch if times.event_batch is not None else times.median
        return "PASS", times, _hold(launcher, timed, args.hold_seconds, launch_us)
    finally:
        launcher.close()


def run_case(spec: MxGemmSpec, *, betas=(0.0, 0.5)) -> str:
    """Compile and launch ``spec`` on visible device 0 (gfx1250) and require
    bit-exact output on both ``--check-data`` input sets at each beta. Returns the kernel name.
    """
    from ....runtime.hip_module import Runtime, get_device_arch

    reason = validate(spec)
    if reason:
        raise ValueError(reason)
    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
    artifact = compile_kernel(build_mx_gemm(spec), arch=arch, backend="python")
    args = argparse.Namespace(
        check_betas=list(betas),
        check_data=CHECK_DATA,
        verify="full",
        samples=0,
        check_only=True,
        hold_seconds=0.0,
    )
    check, _, _ = _run(
        spec,
        artifact,
        mx_gemm_grid(spec),
        spec.launch_cluster,
        args=args,
        runtime=Runtime(),
        timing_inputs=False,
    )
    if check != "PASS":
        raise AssertionError(f"{artifact.kernel_name}: {check}")
    return artifact.kernel_name


def _pair(text: str) -> tuple[int, int]:
    x, y = (int(v) for v in text.lower().split("x"))
    return x, y


def _pad(text: str) -> LdsPad | None:
    """``derived``, ``none`` or ``<pad bytes>x<block bytes>``."""
    if text == "derived":
        return None
    if text == "none":
        return LdsPad()
    pad, block = _pair(text)
    return LdsPad(pad, block)


def _optional_int(text: str) -> int | None:
    return None if text == "derived" else int(text)


def _bool(text: str) -> bool:
    if text not in ("0", "1"):
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return text == "1"


_AXES = {
    # option: (spec field, parser, default)
    "tile": (
        "tile",
        lambda t: tuple(int(v) for v in t.lower().split("x")),
        "256x256x512",
    ),
    "wave_group": ("wave_group", _pair, "2x2"),
    "cluster": ("cluster_dim", lambda t: (*_pair(t), 1), "4x4"),
    "out_dtype": ("out_dtype", str, "fp8e4m3"),
    "gl2_prefetch_distance": ("gl2_prefetch_distance", int, "4"),
    "cluster_signal_wmmas": ("cluster_signal_wmmas", _optional_int, "derived"),
    "local_prefetch": ("local_prefetch", int, "1"),
    "half_plr": ("half_plr", str, "b"),
    "half_b_order": ("half_b_order", str, "ahead"),
    "mma": ("mma", str, "32x16"),
    "a_dtype": ("a_dtype", str, "fp4"),
    "b_dtype": ("b_dtype", str, "fp4"),
    "k_reads": ("k_reads", str, "chunked"),
    "wmma_reuse": ("wmma_reuse", str, "none"),
    "scales": ("scales", str, "lds"),
    "a_lds_pad": ("a_lds_pad", _pad, "derived"),
    "b_lds_pad": ("b_lds_pad", _pad, "derived"),
    "a_scale_lds_pad": ("a_scale_lds_pad", _pad, "derived"),
    "b_scale_lds_pad": ("b_scale_lds_pad", _pad, "derived"),
}


def configurations(args, m: int, n: int, k: int):
    """``(label, spec)`` for every combination of the axis options."""
    axes = []
    for option, (fieldname, parse, _) in _AXES.items():
        texts = [t.strip() for t in getattr(args, option).split(",") if t.strip()]
        axes.append([(option, t, fieldname, parse(t)) for t in texts])
    for combo in itertools.product(*axes):
        fields = {}
        label = []
        for option, text, fieldname, value in combo:
            if fieldname == "tile":
                fields.update(tile_m=value[0], tile_n=value[1], tile_k=value[2])
            else:
                fields[fieldname] = value
            if text != _AXES[option][2]:
                label.append(f"{option}={text}")
        name = "mxfp4_gemm" if fields["a_dtype"] == "fp4" else "mx_gemm"
        yield "|".join(label) or "(defaults)", MxGemmSpec(
            name=name, M=m, N=n, K=k, **fields
        )


def _row(config: str, **values) -> dict:
    row = {f: "" for f in FIELDS}
    row["config"] = config
    row.update(values)
    return row


def _fmt(value) -> str:
    return "" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        choices=tuple(PROBLEMS),
        default="smoke",
        help="smoke 1024x1024x2048; yaml and maf 4096x4096x65536; "
        "mab 65536x16x32768 (hipBLASLt's MXFP8 MAB)",
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    for option, (_, _, default) in _AXES.items():
        parser.add_argument(
            "--" + option.replace("_", "-"),
            dest=option,
            default=default,
            help=f"comma list (default {default})",
        )
    parser.add_argument("--base", action="store_true", help=f"time {BASE_VARIANT} too")
    parser.add_argument("--check-betas", default="0,0.5")
    parser.add_argument(
        "--check-data",
        default=",".join(CHECK_DATA),
        help="comma list: exact (0 / +1 over all of K), sparse (all E2M1 "
        "values or 8-bit values in [2^-1, 1.875], with small E8M0 scales, in "
        f"up to {SPARSE_STEPS} 128-K steps); both bit-exact",
    )
    parser.add_argument(
        "--verify",
        choices=("full", "sampled", "block"),
        default=None,
        help="block: rows x cols of the first and last tile plus --samples "
        "random rows and columns",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4096,
        help="sampled: elements checked; block: random rows and columns",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--batch",
        type=int,
        default=2000,
        help="back-to-back launches timed for batched_us, event_batch_us and "
        "enqueue_us (0 skips them)",
    )
    parser.add_argument(
        "--timing-inputs",
        choices=("tensile", "check"),
        default="tensile",
        help="data of the timed launches: tensile (FP4 as in "
        "mxfp4_tdm_gemm_variants_bench, 8-bit random) or check",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="after timing, keep the timed launch running back to back this long "
        "(for sampling clocks and power from outside; 0: no hold)",
    )
    parser.add_argument("--check-only", action="store_true", help="check, do not time")
    parser.add_argument(
        "--host-only", action="store_true", help="compile only; no device"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="CSV to create; must not exist"
    )
    parser.add_argument("--emit-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    m, n, k = PROBLEMS[args.shape]
    m, n, k = args.m or m, args.n or n, args.k or k
    if args.base and {args.a_dtype, args.b_dtype} != {"fp4"}:
        parser.error(f"--base times the FP4 {BASE_VARIANT}; use it with FP4 only")
    args.verify = args.verify or ("full" if args.shape == "smoke" else "sampled")
    args.check_betas = [float(x) for x in args.check_betas.split(",") if x.strip()]
    args.check_data = [x.strip() for x in args.check_data.split(",") if x.strip()]
    if not args.check_data or set(args.check_data) - set(CHECK_DATA):
        parser.error(f"--check-data must be a comma list of {CHECK_DATA}")

    runtime = None
    arch = "gfx1250"
    if not args.host_only:
        from ....runtime.hip_module import Runtime, get_device_arch

        arch = get_device_arch(0)
        if arch != "gfx1250":
            raise SystemExit(f"visible HIP device 0 must be gfx1250, got {arch!r}")
        runtime = Runtime()

    def compile_and_run(config, kernel, grid, cluster, run_spec):
        artifact = compile_kernel(kernel, arch=arch, backend="python")
        if args.emit_dir is not None:
            args.emit_dir.mkdir(parents=True, exist_ok=True)
            (args.emit_dir / f"{artifact.kernel_name}.hsaco").write_bytes(
                artifact.hsaco
            )
            if artifact.llvm_text:
                (args.emit_dir / f"{artifact.kernel_name}.ll").write_text(
                    artifact.llvm_text
                )
        resources = _resources(artifact.hsaco, args.emit_dir, artifact.kernel_name)
        values = dict(
            kernel=artifact.kernel_name,
            wmma=_wmma_count(artifact.llvm_text),
            loop_scratch=_loop_scratch(artifact.hsaco),
            **resources,
        )
        if args.host_only:
            return _row(config, check="HOST", **values), None
        check, times, hold = _run(
            run_spec,
            artifact,
            grid,
            cluster,
            args=args,
            runtime=runtime,
            timing_inputs=args.timing_inputs == "tensile",
        )
        # event_batch is None only when --batch is 0, and then so is batched.
        best = times.event_batch if times.event_batch is not None else times.median
        tflops = 2.0 * m * n * k / (best * 1.0e-6) / 1.0e12 if best else None
        row = _row(
            config,
            check=check,
            batched_us=_fmt(times.batched),
            event_batch_us=_fmt(times.event_batch),
            enqueue_us=_fmt(times.enqueue),
            median_us=_fmt(times.median),
            tflops=_fmt(tflops),
            hold_start_unix="" if hold.start is None else f"{hold.start:.3f}",
            hold_end_unix="" if hold.end is None else f"{hold.end:.3f}",
            hold_launches=str(hold.launches) if hold.start is not None else "",
            **values,
        )
        return row, best

    def run_base(label):
        spec = variant_spec(BASE_VARIANT, m, n, k, **YAML_TILE)
        run_spec = MxGemmSpec(name="base", M=m, N=n, K=k)
        return compile_and_run(
            label,
            build_mxfp4_tdm_gemm(spec, arch=arch),
            mxfp4_tdm_gemm_grid(spec),
            spec.cluster_dim,
            run_spec,
        )

    # --out gets each row as it finishes, so a run that dies keeps its rows.
    # It must not exist: opening it truncates it before the first row.
    handle = writer = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        handle = args.out.open("x", newline="")
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()

    def report(row):
        print("\t".join(str(row[f]) for f in FIELDS), flush=True)
        if writer is not None:
            writer.writerow(row)
            handle.flush()

    rows, base_times = [], []
    if args.base:
        row, best = run_base(f"{BASE_VARIANT} (before)")
        rows.append(row)
        base_times.append(best)
        report(row)
    timed = []
    for config, spec in configurations(args, m, n, k):
        reason = validate(spec, arch)
        if reason:
            row, best = _row(config, check=f"INVALID:{reason}"), None
        else:
            try:
                row, best = compile_and_run(
                    config,
                    build_mx_gemm(spec, arch=arch),
                    mx_gemm_grid(spec),
                    spec.launch_cluster,
                    spec,
                )
            except Exception as exc:  # noqa: BLE001 - one row per failed config
                row, best = (
                    _row(config, check=f"ERROR:{type(exc).__name__}: {exc}"),
                    None,
                )
        rows.append(row)
        timed.append((row, best))
        report(row)
    if args.base:
        row, best = run_base(f"{BASE_VARIANT} (after)")
        rows.append(row)
        base_times.append(best)
        report(row)
    base = [t for t in base_times if t]
    if base:
        base_mean = sum(base) / len(base)
        for row, best in timed:
            if best:
                row["speedup_vs_base"] = f"{base_mean / best:.3f}"

    print("\t".join(FIELDS))
    for row in rows:
        print("\t".join(str(row[f]) for f in FIELDS))
    if handle is not None:
        handle.close()
        if base:
            # The speedups exist only now; write the whole file again.
            with args.out.open("w", newline="") as final:
                final_writer = csv.DictWriter(final, fieldnames=FIELDS)
                final_writer.writeheader()
                final_writer.writerows(rows)
        print(f"wrote {args.out}", flush=True)
    bad = [r for r in rows if not r["check"].startswith(("PASS", "HOST", "INVALID"))]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
