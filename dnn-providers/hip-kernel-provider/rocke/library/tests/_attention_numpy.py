# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free helpers for attention GPU numeric tests.

Replaces the pieces of torch the numeric tests used: seeded inputs, fp16/bf16
device buffers, readback, and a strict comparison. Reference attention comes
from ``rocke.numeric.sdpa_reference`` (float64 NumPy), not from here.

Values are rounded to the kernel dtype before the reference sees them, so the
reference and the kernel read the same numbers.
"""

from __future__ import annotations

import numpy as np
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import Runtime

from ._attention_device_array import DeviceArray

DTYPES = ("fp16", "bf16")

_TOLERANCE = {"fp16": 2e-2, "bf16": 4e-2}


def tolerance(dtype: str) -> float:
    """Max-abs bound for attention outputs of ``dtype`` with unit-normal inputs."""
    return _TOLERANCE[dtype]


def encode(values, dtype: str) -> np.ndarray:
    """fp32 values as the kernel's host array: ``float16``, or bf16 ``uint16`` bits."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype(np.float16)
    if dtype == "bf16":
        return f32_to_bf16(values)
    raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")


def decode(host, dtype: str) -> np.ndarray:
    """The inverse of :func:`encode`, as fp32."""
    if dtype == "fp16":
        return np.asarray(host, dtype=np.float16).astype(np.float32)
    if dtype == "bf16":
        return bf16_to_f32(host)
    raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")


def round_to(values, dtype: str) -> np.ndarray:
    """fp32 ``values`` rounded to ``dtype`` and back: what the kernel will read."""
    return decode(encode(values, dtype), dtype)


def randn(rng: np.random.Generator, shape, dtype: str, *, scale: float = 1.0):
    """Unit-normal values (times ``scale``) already rounded to ``dtype``, as fp32."""
    return round_to(rng.standard_normal(shape) * scale, dtype)


def runtime() -> Runtime:
    return Runtime()


def to_device(rt, values, dtype: str) -> DeviceArray:
    """Upload fp32 ``values`` as a ``dtype`` device buffer."""
    return DeviceArray(
        rt,
        encode(values, dtype),
        torch_dtype="float16" if dtype == "fp16" else "bfloat16",
    )


def empty_device(rt, shape, dtype: str, *, fill=np.nan) -> DeviceArray:
    """A ``dtype`` output buffer pre-filled with ``fill`` (NaN by default), so an
    unwritten element cannot pass as a correct value."""
    return to_device(rt, np.full(shape, fill, dtype=np.float32), dtype)


def read(dev: DeviceArray, dtype: str) -> np.ndarray:
    """Device contents as fp32."""
    return decode(dev.read(), dtype)


def lse_device(rt, shape, *, fill=np.nan) -> DeviceArray:
    """An fp32 LSE output buffer pre-filled with ``fill``."""
    return DeviceArray(rt, np.full(shape, fill, dtype=np.float32))


def assert_close(got, want, dtype: str, what: str = "output", *, atol=None):
    """Fail unless ``got`` is finite wherever ``want`` is and within ``atol``.

    ``want`` may hold ``-inf``/``nan`` only where ``got`` must hold the same;
    elsewhere both must be finite. The failure names the worst element.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    atol = tolerance(dtype) if atol is None else atol
    special = ~np.isfinite(want)
    same_special = (
        (np.isnan(got) & np.isnan(want))
        | (np.isposinf(got) & np.isposinf(want))
        | (np.isneginf(got) & np.isneginf(want))
    )
    bad = special & ~same_special
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} elements should be {want[bad][0]} "
        f"but are {got[bad][0]} (first at {tuple(np.argwhere(bad)[0])})"
    )
    finite = ~special
    assert np.isfinite(got[finite]).all(), f"{what}: non-finite kernel output"
    if not finite.any():
        return
    err = np.abs(got - want)
    err[~finite] = 0.0
    worst = np.unravel_index(int(np.argmax(err)), err.shape)
    assert err[worst] < atol, (
        f"{what}: max_abs={err[worst]:.3e} >= {atol:.1e} at {tuple(worst)} "
        f"(got {got[worst]:.6g}, want {want[worst]:.6g})"
    )


def paged_attention_reference(
    q, k_cache, v_cache, block_tables, q_lens, kv_lens, *, scale=None, window=0
):
    """float64 paged attention with a bottom-right causal mask.

    ``q`` is ``[sum(q_lens), Hq, D]``; the caches are ``[blocks, block_size, Hkv, D]``
    and ``block_tables[b]`` lists sequence ``b``'s pages. ``window > 0`` keeps only the
    ``window`` newest keys at each query (the diagonal included). Returns ``O`` in
    the layout of ``q``.
    """
    from rocke.numeric.sdpa_reference import sdpa_reference

    hkv, d = k_cache.shape[2], k_cache.shape[3]
    keys, values = [], []
    for b, n in enumerate(kv_lens):
        pages = np.asarray(block_tables[b])[: -(-int(n) // k_cache.shape[1])]
        keys.append(np.asarray(k_cache[pages], np.float64).reshape(-1, hkv, d)[:n])
        values.append(np.asarray(v_cache[pages], np.float64).reshape(-1, hkv, d)[:n])

    def offsets(lens):
        return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    out, _ = sdpa_reference(
        q,
        np.concatenate(keys),
        np.concatenate(values),
        scale=scale,
        q_offsets=offsets(q_lens),
        kv_offsets=offsets(kv_lens),
        left_bound=window - 1 if window else None,
        right_bound=0,
        top_left=False,
    )
    return out


class LargeDeviceTensor:
    """A device tensor too large to mirror on the host, filled in chunks.

    Carries the attributes the launch helpers read. ``fill_random`` writes unit-normal
    values over the whole buffer a chunk at a time; ``write_rows`` overwrites whole
    leading-axis rows with chosen values.
    """

    device = None
    is_cuda = True

    def __init__(self, rt, shape, dtype: str):
        from rocke.runtime.launcher import DeviceMem

        self._rt = rt
        self._dtype = dtype
        self.shape = tuple(shape)
        self.dtype = "torch.float16" if dtype == "fp16" else "torch.bfloat16"
        self._row_elems = int(np.prod(self.shape[1:]))
        self._mem = DeviceMem(int(np.prod(self.shape)) * 2)

    def stride(self, dim):
        return int(np.prod(self.shape[dim + 1 :]))

    def data_ptr(self):
        return self._mem.ptr()

    def is_contiguous(self):
        return True

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 2

    def dim(self):
        return len(self.shape)

    def write_rows(self, start, values):
        import ctypes

        host = encode(values, self._dtype).reshape(-1, self._row_elems)
        host = np.ascontiguousarray(host.view(np.uint8))
        buf = (ctypes.c_uint8 * host.nbytes).from_buffer(host)
        self._rt.memcpy_h2d(
            self._mem.ptr() + start * self._row_elems * 2, buf, host.nbytes
        )

    def fill_random(self, rng, *, chunk_elems=1 << 26):
        rows = max(1, chunk_elems // self._row_elems)
        for start in range(0, self.shape[0], rows):
            n = min(rows, self.shape[0] - start)
            self.write_rows(
                start, rng.standard_normal((n, self._row_elems), dtype=np.float32)
            )


def scattered_paged_cache(rt, rng, num_blocks, block_size, hkv, d, dtype, pages):
    """Device K/V page caches of ``num_blocks`` pages, random everywhere, with the
    page ids in ``pages`` overwritten by known values.

    Returns ``(k_cache, v_cache, k_host, v_host)``; the host arrays are
    ``[len(pages), block_size, Hkv, D]`` fp32 in the order of ``pages``, so a
    reference can run on them with the table ``arange(len(pages))``. A kernel that
    reads a wrong page or a wrapped offset sees unrelated random data.
    """
    caches = []
    for _ in range(2):
        cache = LargeDeviceTensor(rt, (num_blocks, block_size, hkv, d), dtype)
        cache.fill_random(rng)
        caches.append(cache)
    hosts = []
    for cache in caches:
        host = randn(rng, (len(pages), block_size, hkv, d), dtype)
        for i, page in enumerate(pages):
            cache.write_rows(int(page), host[i : i + 1])
        hosts.append(host)
    return caches[0], caches[1], hosts[0], hosts[1]
