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
    and ``ptr<f16, global>`` for dtype="fp16", rejects unknown dtypes, and
    -- for the 8-bit GEMMs -- carries a C dtype wider than A/B.  C defaults
    to the operand dtype, which every pre-existing fp16/bf16 caller relies
    on: a changed default would silently rewrite their manifests.

``_gemm_is_bf16``
    Reads the A-pointer type from args_signature; returns True for bf16,
    False for fp16, False when the key is absent.

e4m3 codec
    ``_fp8e4m3_decode_table`` / ``_fp8e4m3_encode`` checked against the OCP
    e4m3 (``fn``) format rather than against a transcription of themselves --
    a second hand-written table could drift from the hardware's in exactly
    the way the first one did.  The encoder must *refuse* inexact values
    rather than round them: a silent round would demote the runner's exact
    compare to a tolerance check without changing any output.

fp8 runner body
    ``run_gemm_fp8_manifest_problem`` over the same 24x128x3072 problem:
    the e4m3 bytes it pushes to the device decode back to the seeded
    integers exactly, exact output verifies clean, one corrupted element
    is caught, and the K guard that protects the exact-accumulate argument
    fires.  The fp8 path compares exactly rather than within a tolerance,
    so a round-off anywhere in it would surface as a nonzero bad count.

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
    _fp8e4m3_decode_table,
    _fp8e4m3_encode,
    _gemm_is_bf16,
    run_gemm_fp8_manifest_problem,
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


def _make_fp8_manifest(shape: Optional[Tuple[int, int, int]] = None) -> dict:
    """fp8e4m3 A/B -> bf16 C manifest, the gfx1250 Phase 0 shape."""
    M, N, K = shape or _SHAPE
    return {
        "kind": "gemm_fp8",
        "block_m": M,
        "block_n": N,
        "block_k": K,
        "threads_per_block": 256,
        "default_shape": [M, N, K],
        "grid_order": "NM",
        "args_signature": gemm_args_signature(dtype="fp8e4m3", c_dtype="bf16"),
    }


def _run_fp8(shape: Optional[Tuple[int, int, int]] = None):
    """Build the fp8 problem on a fake runtime.

    Returns (rt, ptrs, check_fn, a_ptr, b_ptr, c_ptr).  The device pointers
    come back so a test can read what ``make_args`` actually pushed.
    """
    manifest = _make_fp8_manifest(shape)
    make_args_fn, _grid, _block, _flop, _bw, check_fn = run_gemm_fp8_manifest_problem(
        manifest, shape or _SHAPE, verify=True
    )
    rt = _FakeRuntime()
    packed_args, ptrs = make_args_fn(rt)
    a_ptr, b_ptr, c_ptr = struct.unpack_from("<QQQ", packed_args)
    return rt, ptrs, check_fn, a_ptr, b_ptr, c_ptr


def _fp8_operands(rt: _FakeRuntime, a_ptr: int, b_ptr: int, shape):
    """Decode the e4m3 bytes sitting in the fake device buffers back to f32."""
    M, N, K = shape
    table = _fp8e4m3_decode_table(np)
    a_raw = np.frombuffer(bytes(rt._store[a_ptr]), dtype=np.uint8)[: M * K]
    b_raw = np.frombuffer(bytes(rt._store[b_ptr]), dtype=np.uint8)[: N * K]
    return table[a_raw.reshape(M, K)], table[b_raw.reshape(N, K)]


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

    def test_c_defaults_to_the_operand_dtype(self):
        # Every pre-existing fp16/bf16 caller omits c_dtype, so the default must
        # reproduce the old signature byte for byte or their manifests change.
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                sig = gemm_args_signature(dtype=dtype)
                types = [a["type"] for a in sig[:3]]
                self.assertEqual(len(set(types)), 1)

    def test_eight_bit_operands_can_carry_a_wider_c(self):
        # The 8-bit GEMMs read fp8 A/B but write bf16 C; the runner sniffs the A
        # pointer type to pick its reference dtype, so this signature is the
        # single place the mixed ABI is stated.
        sig = gemm_args_signature(dtype="fp8e4m3", c_dtype="bf16")
        self.assertEqual(sig[0]["type"], "ptr<fp8e4m3, global>")
        self.assertEqual(sig[1]["type"], "ptr<fp8e4m3, global>")
        self.assertEqual(sig[2]["type"], "ptr<bf16, global>")

    def test_unsupported_dtypes_are_named_in_the_error(self):
        for kwargs in ({"dtype": "fp4"}, {"dtype": "fp16", "c_dtype": "fp4"}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    gemm_args_signature(**kwargs)
                self.assertIn("fp4", str(ctx.exception))


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


class TestFp8E4M3Codec(unittest.TestCase):
    """The e4m3 codec the fp8 runner encodes its test inputs with.

    The runner verifies the device against a numpy reference *exactly* --
    ``bad = err > 0.0``, no tolerance -- which is only sound while the host
    encodes those inputs into e4m3 without rounding them.  An exact gate that
    silently stopped being exact would still report PASS.
    """

    def test_known_codes_decode_to_their_format_values(self):
        # Anchors spanning the encoding: both zeros, the implicit-leading-one
        # boundary, a mantissa step, and the largest finite magnitude. Hand-
        # computed from s.eeee.mmm / bias 7, not read back out of the table.
        table = _fp8e4m3_decode_table(np)
        for code, want in (
            (0x00, 0.0),
            (0x80, -0.0),
            (0x38, 1.0),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0xB8, -1.0),
            (0x7E, 448.0),
            (0xFE, -448.0),
        ):
            with self.subTest(code=hex(code)):
                self.assertEqual(float(table[code]), want)

    def test_subnormals_use_the_e_eq_0_rule(self):
        # e == 0 drops the implicit leading one and pins the exponent at 2^-6,
        # so the smallest step is 2^-6/8 == 2^-9 and the steps are uniform.
        table = _fp8e4m3_decode_table(np)
        self.assertEqual(float(table[0x01]), 2.0**-9)
        self.assertEqual(float(table[0x07]), 7 * 2.0**-9)
        # Subnormal-to-normal must be continuous: 0x08 is the first normal.
        self.assertEqual(float(table[0x08]), 8 * 2.0**-9)

    def test_nan_only_at_the_two_all_ones_codes(self):
        # e4m3 spends no codes on infinities, so 0x7f/0xff are the only NaNs
        # and every other code is finite -- which is what lets the encoder
        # treat the finite set as the whole representable set.
        table = _fp8e4m3_decode_table(np)
        nan_codes = set(np.flatnonzero(np.isnan(table)).tolist())
        self.assertEqual(nan_codes, {0x7F, 0xFF})
        self.assertFalse(np.isinf(table).any())

    def test_every_finite_code_round_trips(self):
        table = _fp8e4m3_decode_table(np)
        finite = np.flatnonzero(np.isfinite(table)).astype(np.uint8)
        got = _fp8e4m3_encode(np, table[finite])
        # -0.0 encodes to +0.0's code: they are numerically equal, so the
        # encoder cannot distinguish them and must not claim to. Compare the
        # decoded values instead of the codes.
        self.assertTrue(np.array_equal(_fp8e4m3_decode_table(np)[got], table[finite]))

    def test_the_verify_input_range_is_exactly_representable(self):
        # The runner draws integers in -5..5 and leans on every one being exact
        # in e4m3. If that ever stopped holding the gate would quietly compare
        # the device against numbers it was never given.
        vals = np.arange(-5, 6, dtype=np.float32)
        table = _fp8e4m3_decode_table(np)
        self.assertTrue(np.array_equal(table[_fp8e4m3_encode(np, vals)], vals))

    def test_inexact_values_are_refused_rather_than_rounded(self):
        # A silent round here is the failure mode that matters: it would demote
        # the exact compare to a tolerance check without changing any output.
        for bad in (0.3, 1.1, 2.0**-10, 449.0, 1e30):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    _fp8e4m3_encode(np, np.array([bad], dtype=np.float32))

    def test_encode_preserves_shape(self):
        x = np.zeros((3, 5), dtype=np.float32)
        self.assertEqual(_fp8e4m3_encode(np, x).shape, (3, 5))
        self.assertEqual(_fp8e4m3_encode(np, x).dtype, np.uint8)


class TestGemmFp8ManifestRunner(unittest.TestCase):
    """fp8e4m3 A/B -> bf16 C runner body, launch stubbed out."""

    def test_device_operands_decode_back_exactly(self):
        """The e4m3 bytes pushed to the device decode to the seeded integers.

        This is the load-bearing property behind the exact (tolerance-free)
        comparison: the encode must be lossless for inputs in -5..5, or the
        reference and the kernel would be multiplying different numbers.
        """
        M, N, K = _SHAPE
        rt, _ptrs, _check, a_ptr, b_ptr, _c = _run_fp8()
        A, B = _fp8_operands(rt, a_ptr, b_ptr, _SHAPE)

        rng = np.random.default_rng(0xC0FFEE)
        A_ref = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float32)
        B_ref = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float32)

        self.assertTrue(np.array_equal(A, A_ref), "A did not round-trip through e4m3")
        self.assertTrue(np.array_equal(B, B_ref), "B did not round-trip through e4m3")

    def test_fp8_correct_output_passes(self):
        """Exact kernel output (including -871 -> -872 RNE) gives zero error."""
        M, N, _K = _SHAPE
        rt, ptrs, check_fn, a_ptr, b_ptr, c_ptr = _run_fp8()
        A, B = _fp8_operands(rt, a_ptr, b_ptr, _SHAPE)

        dot = A @ B.T
        # Same RNE landmark the bf16 case uses; it survives the fp8 operands.
        self.assertEqual(float(dot[0, 0]), -871.0, "seeded fp8 dot[0,0] != -871")
        ref_u16 = _bf16_from_f32(np, dot)
        self.assertEqual(float(_f32_from_bf16(np, ref_u16)[0, 0]), -872.0)

        raw = ref_u16.view(np.uint8).tobytes()
        rt.memcpy_h2d(c_ptr, memoryview(bytearray(raw)), len(raw))

        max_err, bad, total = check_fn(rt, ptrs)
        self.assertEqual(max_err, 0.0, f"expected zero error, got {max_err}")
        self.assertEqual(bad, 0, f"expected zero bad elements, got {bad}")
        self.assertEqual(total, M * N)

    def test_fp8_corrupted_output_detected(self):
        """One element shifted by 8 gives abs_err=8 and exactly one bad."""
        rt, ptrs, check_fn, a_ptr, b_ptr, c_ptr = _run_fp8()
        A, B = _fp8_operands(rt, a_ptr, b_ptr, _SHAPE)

        corrupted = _bf16_from_f32(np, A @ B.T)
        orig = float(_f32_from_bf16(np, corrupted[:1, :1])[0, 0])
        corrupted[0, 0] = _bf16_from_f32(
            np, np.array([orig + 8.0], dtype=np.float32)
        )[0]

        raw = corrupted.view(np.uint8).tobytes()
        rt.memcpy_h2d(c_ptr, memoryview(bytearray(raw)), len(raw))

        max_err, bad, _total = check_fn(rt, ptrs)
        self.assertAlmostEqual(max_err, 8.0, places=3, msg="expected abs_err=8")
        self.assertEqual(bad, 1, f"expected exactly 1 bad element, got {bad}")

    def test_fp8_verify_disabled_short_circuits(self):
        """verify=False reports clean without reading the device buffer."""
        manifest = _make_fp8_manifest()
        make_args_fn, _g, _b, _f, _bw, check_fn = run_gemm_fp8_manifest_problem(
            manifest, _SHAPE, verify=False
        )
        rt = _FakeRuntime()
        _packed, ptrs = make_args_fn(rt)
        self.assertEqual(check_fn(rt, ptrs), (0.0, 0, _SHAPE[0] * _SHAPE[1]))

    def test_large_k_rejected(self):
        """K large enough to break the exact-accumulate argument is refused."""
        manifest = _make_fp8_manifest()
        with self.assertRaises(ValueError) as cm:
            run_gemm_fp8_manifest_problem(manifest, (8, 8, 1 << 20), verify=True)
        self.assertIn("25*K", str(cm.exception))

    def test_largest_exact_k_accepted(self):
        """The largest K the exact-accumulate argument allows is not refused.

        Pins the bound itself: a guard tightened past 25*K rejects this K,
        which is the mutation ``test_large_k_rejected`` alone cannot see.
        """
        k_max = ((1 << 24) - 1) // 25  # largest K with 25*K < 2**24
        self.assertLess(25 * k_max, 1 << 24)
        manifest = _make_fp8_manifest()
        run_gemm_fp8_manifest_problem(manifest, (8, 8, k_max), verify=False)

    def test_grid_order_flips_axes(self):
        """grid_order="MN" swaps gx/gy relative to the default "NM"."""
        shape = (48, 128, 256)
        manifest = _make_fp8_manifest(shape)
        manifest["block_m"], manifest["block_n"] = 16, 32

        _mk, grid_nm, *_ = run_gemm_fp8_manifest_problem(manifest, shape, verify=False)
        manifest["grid_order"] = "MN"
        _mk, grid_mn, *_ = run_gemm_fp8_manifest_problem(manifest, shape, verify=False)

        self.assertEqual(grid_nm, (4, 3, 1))  # gx over N, gy over M
        self.assertEqual(grid_mn, (3, 4, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
