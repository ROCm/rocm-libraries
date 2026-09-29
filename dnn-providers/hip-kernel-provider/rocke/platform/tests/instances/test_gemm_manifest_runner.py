# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only regression tests for the GEMM manifest-runner verify path.

The verify callback inside ``run_gemm_manifest_problem`` decides whether
kernel output is correct.  These tests exercise it without a GPU by
injecting controlled output bytes directly through ``make_args`` / ``check``.

Cases covered
-------------
bf16 correct
    Seeded 24x128x3072 problem with exact kernel output: zero error, zero
    bad elements.  The accumulated value -871 must round to -872 (RNE, not
    truncation) and the output must be decoded as bf16, not fp16.

bf16 corruption
    Same problem with one element changed from 292 -> 300 in the raw bf16
    output bytes: absolute error = 8, exactly one bad element.

fp16 correct
    Unchanged fp16 path: zero error, zero bad elements.

args_signature dtype
    ``gemm_args_signature`` emits ``ptr<bf16, global>`` for dtype="bf16"
    and ``ptr<f16, global>`` for dtype="fp16", and rejects unknown dtypes.

``_gemm_is_bf16``
    Reads the A-pointer type from args_signature; returns True for bf16,
    False for fp16, False when the key is absent.

These tests are CPU-only, torch-free, and have no GPU dependency.
"""

from __future__ import annotations

import struct
import unittest
from typing import Optional, Tuple

import numpy as np

from rocke.dispatch.gemm.binding import _bf16_from_f32, _f32_from_bf16
from rocke.helpers.manifest import gemm_args_signature
from rocke.instances.common.manifest_runner.gemm import (
    _gemm_is_bf16,
    run_gemm_manifest_problem,
)


# ---------------------------------------------------------------------------
# Minimal fake Runtime that backs the check() callback with numpy arrays
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal Runtime substitute: h2d/d2h move bytes between numpy arrays."""

    def __init__(self):
        self._store: dict[int, bytearray] = {}
        self._next_ptr = 1 << 40  # above any realistic address

    def alloc(self, n: int) -> int:
        ptr = self._next_ptr
        self._store[ptr] = bytearray(n)
        self._next_ptr += n + 8  # small gap so addresses don't collide
        return ptr

    def memcpy_h2d(self, dst: int, src: memoryview, n: int) -> None:
        self._store[dst][:n] = bytes(src)[:n]

    def memcpy_d2h(self, dst: memoryview, src: int, n: int) -> None:
        dst[:n] = bytes(self._store[src])[:n]

    def memset(self, ptr: int, val: int, n: int) -> None:
        self._store[ptr][:n] = bytes([val & 0xFF]) * n


# ---------------------------------------------------------------------------
# Helper: build a minimal manifest dict, run the problem, capture check()
# ---------------------------------------------------------------------------

_SHAPE = (24, 128, 3072)  # M, N, K  (matches reviewer-cited problem)


def _make_manifest(dtype: str = "fp16") -> dict:
    M, N, K = _SHAPE
    return {
        "kind": "gemm_fp16",
        "block_m": M,
        "block_n": N,
        "block_k": K,
        "threads_per_block": 256,
        "default_shape": list(_SHAPE),
        "grid_order": "NM",
        "args_signature": gemm_args_signature(dtype=dtype),
    }


def _run_and_check(
    dtype: str,
    corrupt_fn=None,
    shape: Optional[Tuple[int, int, int]] = None,
):
    """Build the problem, optionally corrupt the device C buffer, run check().

    Returns (max_abs_diff, bad_count, total).
    """
    manifest = _make_manifest(dtype)
    make_args_fn, grid, block, flop, bw, check_fn = run_gemm_manifest_problem(
        manifest, shape or _SHAPE, verify=True
    )

    rt = _FakeRuntime()
    packed_args, ptrs = make_args_fn(rt)

    # Unpack C pointer from the struct: "<QQQiii" -> ptr[2] is C_dev
    c_ptr = struct.unpack_from("<QQQ", packed_args)[2]

    if corrupt_fn is not None:
        corrupt_fn(rt, c_ptr)

    return check_fn(rt, ptrs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGemmManifestRunnerBf16(unittest.TestCase):
    """bf16 verify path: RNE rounding and correct byte interpretation."""

    def test_bf16_correct_output_passes(self):
        """Exact kernel output (including -871 -> -872 RNE) gives zero error."""
        M, N, K = _SHAPE
        # Reproduce the seeded inputs exactly as the runner does.
        rng = np.random.default_rng(0xC0FFEE)
        A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
        B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)

        # Correct output: fp32 matmul -> RNE -> bf16 raw bytes
        ref_u16 = _bf16_from_f32(np, A_f32 @ B_f32.T)  # (M, N) uint16

        # Verify the reviewer-cited -871 -> -872 rounding is present.
        dot = A_f32 @ B_f32.T
        self.assertTrue(np.any(dot.ravel() == -871), "-871 not in seeded inputs")
        minus871_u16 = _bf16_from_f32(np, np.array([-871.0], dtype=np.float32))[0]
        minus872_f32 = _f32_from_bf16(np, np.array([minus871_u16], dtype=np.uint16))[0]
        self.assertEqual(float(minus872_f32), -872.0, "-871 must RNE to -872")

        def inject_correct(rt: _FakeRuntime, c_ptr: int) -> None:
            # Write bf16 raw bytes into the fake device C buffer.
            raw = ref_u16.view(np.uint8).tobytes()
            view = memoryview(bytearray(raw))
            rt.memcpy_h2d(c_ptr, view, len(raw))

        max_err, bad, total = _run_and_check("bf16", corrupt_fn=inject_correct)
        self.assertEqual(max_err, 0.0, f"expected zero error, got {max_err}")
        self.assertEqual(bad, 0, f"expected zero bad elements, got {bad}")
        self.assertEqual(total, M * N)

    def test_bf16_corrupted_output_detected(self):
        """One element changed 292 -> 300 gives abs_err=8, bad_count=1."""
        M, N, K = _SHAPE
        rng = np.random.default_rng(0xC0FFEE)
        A_f32 = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
        B_f32 = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)

        ref_u16 = _bf16_from_f32(np, A_f32 @ B_f32.T)  # (M, N) uint16
        # Confirm 292 is present in the reference (reviewer-cited).
        ref_vals = _f32_from_bf16(np, ref_u16)
        self.assertTrue(np.any(ref_vals == 292.0), "292 not in reference")

        corrupted_u16 = ref_u16.copy()
        idx = np.argwhere(ref_vals == 292.0)[0]  # first occurrence
        # Replace 292 with 300 (nearest representable bf16).
        corrupted_u16[tuple(idx)] = _bf16_from_f32(
            np, np.array([300.0], dtype=np.float32)
        )[0]

        def inject_corrupted(rt: _FakeRuntime, c_ptr: int) -> None:
            raw = corrupted_u16.view(np.uint8).tobytes()
            rt.memcpy_h2d(c_ptr, memoryview(bytearray(raw)), len(raw))

        max_err, bad, total = _run_and_check("bf16", corrupt_fn=inject_corrupted)
        self.assertAlmostEqual(max_err, 8.0, places=3, msg="expected abs_err=8")
        self.assertEqual(bad, 1, f"expected exactly 1 bad element, got {bad}")


class TestGemmManifestRunnerFp16(unittest.TestCase):
    """fp16 verify path is unchanged."""

    def test_fp16_correct_output_passes(self):
        M, N, K = _SHAPE
        rng = np.random.default_rng(0xC0FFEE)
        A = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float16)
        B = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float16)
        ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)

        def inject_correct(rt: _FakeRuntime, c_ptr: int) -> None:
            raw = ref.view(np.uint8).tobytes()
            rt.memcpy_h2d(c_ptr, memoryview(bytearray(raw)), len(raw))

        max_err, bad, total = _run_and_check("fp16", corrupt_fn=inject_correct)
        self.assertEqual(max_err, 0.0)
        self.assertEqual(bad, 0)


class TestGemmArgSignatureDtype(unittest.TestCase):
    """``gemm_args_signature`` emits correct ptr types."""

    def test_fp16_emits_f16_ptr(self):
        sig = gemm_args_signature(dtype="fp16")
        ptr_types = {a["name"]: a["type"] for a in sig}
        self.assertEqual(ptr_types["A"], "ptr<f16, global>")
        self.assertEqual(ptr_types["B"], "ptr<f16, global>")
        self.assertEqual(ptr_types["C"], "ptr<f16, global>")

    def test_bf16_emits_bf16_ptr(self):
        sig = gemm_args_signature(dtype="bf16")
        ptr_types = {a["name"]: a["type"] for a in sig}
        self.assertEqual(ptr_types["A"], "ptr<bf16, global>")
        self.assertEqual(ptr_types["B"], "ptr<bf16, global>")
        self.assertEqual(ptr_types["C"], "ptr<bf16, global>")

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            gemm_args_signature(dtype="fp32")


class TestGemmIsBf16(unittest.TestCase):
    """``_gemm_is_bf16`` reads element type from args_signature."""

    def test_bf16_ptr_returns_true(self):
        sig = gemm_args_signature(dtype="bf16")
        self.assertTrue(_gemm_is_bf16({"args_signature": sig}))

    def test_fp16_ptr_returns_false(self):
        sig = gemm_args_signature(dtype="fp16")
        self.assertFalse(_gemm_is_bf16({"args_signature": sig}))

    def test_missing_signature_returns_false(self):
        self.assertFalse(_gemm_is_bf16({}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
