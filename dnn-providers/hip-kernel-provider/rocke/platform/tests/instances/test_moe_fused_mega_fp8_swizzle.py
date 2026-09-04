# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the fp8 mega-kernel's coalesced B (weight) layout.

The swizzle is a contract between two pieces of code that never meet at
runtime: the host permutation ``swizzle_b_fp8_weights`` and the kernel-side
address math in ``_load_b_fp8_swizzled`` / ``_dtla_stage_b_fp8``. A mismatch
produces silently wrong numerics rather than an error, so the contract is
pinned here by re-deriving the kernel's addresses in numpy.
"""

from __future__ import annotations

import unittest

import numpy as np

WAVE = 64
ATOM_N = 16
ATOM_K = 128
CHUNK = 16
PER_LANE = ATOM_N * ATOM_K // WAVE  # 32 bytes of K per lane
CHUNKS = PER_LANE // CHUNK  # 2


def _kernel_addr(n_tile_base, k_tile_base, k_extent, lane, chunk):
    """The address ``_load_b_fp8_swizzled`` computes, in bytes from the expert base."""
    tile_base = n_tile_base * k_extent + k_tile_base * ATOM_N
    return tile_base + chunk * WAVE * CHUNK + lane * CHUNK


class TestBSwizzleLayout(unittest.TestCase):
    def _swizzle(self, w):
        from rocke.instances.common.moe_fused_mega_fp8 import swizzle_b_fp8_weights

        return swizzle_b_fp8_weights(
            w, atom_n=ATOM_N, atom_k=ATOM_K, wave_size=WAVE
        )

    def test_is_a_pure_permutation(self):
        rng = np.random.default_rng(0)
        w = rng.integers(0, 256, size=(3, 64, 256), dtype=np.uint8)
        swz = self._swizzle(w)
        self.assertEqual(swz.shape, w.shape)
        self.assertEqual(swz.dtype, w.dtype)
        self.assertTrue(swz.flags["C_CONTIGUOUS"])
        # Per expert, so a swizzle can never move bytes across the per-expert
        # stride the kernel signature carries.
        for e in range(w.shape[0]):
            self.assertTrue(
                np.array_equal(np.sort(swz[e].ravel()), np.sort(w[e].ravel()))
            )

    def test_kernel_addresses_recover_the_logical_fragment(self):
        """Every (tile, lane, chunk) the kernel reads must hold the right bytes."""
        rng = np.random.default_rng(1)
        for rows, k in ((64, 256), (48, 384)):
            w = rng.integers(0, 256, size=(2, rows, k), dtype=np.uint8)
            flat = self._swizzle(w).reshape(w.shape[0], rows * k)
            for e in range(w.shape[0]):
                for n_tile_base in range(0, rows, ATOM_N):
                    for k_tile_base in range(0, k, ATOM_K):
                        for lane in range(WAVE):
                            for c in range(CHUNKS):
                                addr = _kernel_addr(
                                    n_tile_base, k_tile_base, k, lane, c
                                )
                                got = flat[e, addr : addr + CHUNK]
                                row = n_tile_base + lane % ATOM_N
                                col = (
                                    k_tile_base
                                    + (lane // ATOM_N) * PER_LANE
                                    + c * CHUNK
                                )
                                np.testing.assert_array_equal(
                                    got,
                                    w[e, row, col : col + CHUNK],
                                    err_msg=(
                                        f"e={e} n={n_tile_base} k={k_tile_base} "
                                        f"lane={lane} chunk={c}"
                                    ),
                                )

    def test_a_chunk_instruction_is_fully_coalesced(self):
        """The point of the layout: 64 lanes x 16B == one contiguous 1024B run."""
        for c in range(CHUNKS):
            addrs = [_kernel_addr(16, 128, 2048, lane, c) for lane in range(WAVE)]
            self.assertEqual(sorted(addrs), addrs)
            self.assertEqual(max(addrs) + CHUNK - min(addrs), WAVE * CHUNK)
            # 1024 contiguous bytes touch 8 128B lines, against 16 for the
            # row-major form below.
            self.assertEqual(len({a // 128 for a in addrs}), 8)

    def test_row_major_form_touches_twice_the_cache_lines(self):
        """Pins the pathology the swizzle exists to remove."""
        k_extent = 2048  # gate/up row stride
        addrs = [
            (16 + lane % ATOM_N) * k_extent + 128 + (lane // ATOM_N) * PER_LANE
            for lane in range(WAVE)
        ]
        self.assertEqual(len({a // 128 for a in addrs}), 16)

    def test_rejects_untileable_shapes(self):
        from rocke.instances.common.moe_fused_mega_fp8 import swizzle_b_fp8_weights

        with self.assertRaises(ValueError):
            swizzle_b_fp8_weights(np.zeros((1, 24, 256), dtype=np.uint8))
        with self.assertRaises(ValueError):
            swizzle_b_fp8_weights(np.zeros((1, 64, 200), dtype=np.uint8))


class TestBSwizzleSpecWiring(unittest.TestCase):
    def test_defaults_are_off(self):
        from rocke.instances.common.moe_fused_mega_fp8 import FusedMegaKernelSpecFp8

        spec = FusedMegaKernelSpecFp8(name="t")
        self.assertFalse(spec.swizzle_gu)
        self.assertFalse(spec.swizzle_down)

    def test_supported_only_for_the_hero_atom(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            FusedMegaKernelSpecFp8,
            b_swizzle_supported,
        )

        spec = FusedMegaKernelSpecFp8(name="t")
        self.assertTrue(b_swizzle_supported(spec.gate_up_atom()))
        # The legacy 16x16x32 atom's 8-byte B fragment is narrower than the
        # 16-byte VMEM chunk the layout is built from.
        legacy = FusedMegaKernelSpecFp8(name="t", gate_up_k=32)
        self.assertFalse(b_swizzle_supported(legacy.gate_up_atom()))

    def test_builder_rejects_swizzle_on_unsupported_atom(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            FusedMegaKernelSpecFp8,
            build_moe_fused_mega_gemm_fp8,
        )

        spec = FusedMegaKernelSpecFp8(name="t", gate_up_k=32, swizzle_gu=True)
        with self.assertRaises(ValueError) as cm:
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        self.assertIn("swizzle_gu", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
