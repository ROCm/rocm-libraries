# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Spec-build tests for the gfx942 lightning indexer (CPU-only, no GPU).

Covers the validator gates, the LDS budget, the kernel-name geometry, and that a
valid spec builds a KernelDef and lowers to LLVM IR. HSACO resource-fit on a full
comgr toolchain is exercised by ``tools/run_checks.py``; this file stays green
without a device.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from kernels.common.lightning_indexer import (  # noqa: E402
    LDS_LIMIT,
    IndexerSpec,
    IndexerTileSpec,
    build_lightning_indexer,
    is_valid_spec,
    lightning_indexer_grid,
    lightning_indexer_signature,
)

_ARCH = "gfx942"


def _spec(**over) -> IndexerSpec:
    base = dict(
        n_index_heads=4,
        index_head_dim=16,
        seqlen_q=8,
        seqlen_k=32,
        tile=IndexerTileSpec(block_size=64),
    )
    base.update(over)
    return IndexerSpec(**base)


class TestValidator(unittest.TestCase):
    def test_accepts_gfx942(self):
        ok, why = is_valid_spec(_spec(), arch="gfx942")
        self.assertTrue(ok, why)

    def test_accepts_gfx950(self):
        ok, why = is_valid_spec(_spec(), arch="gfx950")
        self.assertTrue(ok, why)

    def test_rejects_unsupported_arch(self):
        # An RDNA wave32 part: no CDNA MFMA path.
        ok, why = is_valid_spec(_spec(), arch="gfx1151")
        self.assertFalse(ok)
        self.assertIn("gfx942/gfx950", why)

    def test_rejects_bad_dtype(self):
        ok, why = is_valid_spec(_spec(dtype="f16"), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("dtype", why)

    def test_rejects_block_size_not_wave_multiple(self):
        ok, why = is_valid_spec(_spec(tile=IndexerTileSpec(block_size=100)), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("wave_size", why)

    def test_rejects_oversize_block(self):
        ok, why = is_valid_spec(
            _spec(tile=IndexerTileSpec(block_size=2048)), arch=_ARCH
        )
        self.assertFalse(ok)
        self.assertIn("1024", why)

    def test_rejects_nonpositive_geometry(self):
        ok, why = is_valid_spec(_spec(n_index_heads=0), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("n_index_heads", why)


class TestBudget(unittest.TestCase):
    def test_scalar_v1_uses_no_lds(self):
        self.assertEqual(_spec().lds_bytes(), 0)
        self.assertLessEqual(_spec().lds_bytes(), LDS_LIMIT)


class TestKernelName(unittest.TestCase):
    def test_geometry_in_name(self):
        name = _spec(
            n_index_heads=32,
            index_head_dim=128,
            seqlen_q=8,
            seqlen_k=64,
            tile=IndexerTileSpec(block_size=256),
        ).kernel_name()
        self.assertIn("HI32", name)
        self.assertIn("D128", name)
        self.assertIn("bf16", name)
        self.assertIn("Q8", name)
        self.assertIn("K64", name)
        self.assertIn("b256", name)


class TestBuildAndLower(unittest.TestCase):
    def test_builds_and_lowers(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        kd = build_lightning_indexer(_spec(), arch=_ARCH)
        for flavor in ("llvm20", "llvm22"):
            ir = _lower_kernel_to_llvm_python(kd, arch=_ARCH, llvm_flavor=flavor)
            self.assertIn("define", ir, f"no define in IR for {flavor}")

    def test_grid_and_signature(self):
        spec = _spec(seqlen_q=13)
        self.assertEqual(lightning_indexer_grid(spec), (13, 1, 1))
        names = [s["name"] for s in lightning_indexer_signature(spec)]
        self.assertEqual(names, ["index_q", "index_k", "w", "scores", "q_pos_base"])

    def test_invalid_spec_raises(self):
        with self.assertRaises(ValueError):
            build_lightning_indexer(_spec(), arch="gfx1151")


class TestMfmaBody(unittest.TestCase):
    def _mfma(self, **over) -> IndexerSpec:
        base = dict(
            n_index_heads=32,
            index_head_dim=128,
            seqlen_q=16,
            seqlen_k=64,
            body="mfma",
            tile=IndexerTileSpec(block_size=64),
        )
        base.update(over)
        return IndexerSpec(**base)

    def test_accepts_aligned(self):
        ok, why = is_valid_spec(self._mfma(), arch=_ARCH)
        self.assertTrue(ok, why)

    def test_requires_wave_block(self):
        ok, why = is_valid_spec(
            self._mfma(tile=IndexerTileSpec(block_size=256)), arch=_ARCH
        )
        self.assertFalse(ok)
        self.assertIn("block_size", why)

    def test_requires_aligned_seqlen(self):
        ok, why = is_valid_spec(self._mfma(seqlen_q=15), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("multiples of 16", why)

    def test_requires_aligned_dim(self):
        ok, why = is_valid_spec(self._mfma(index_head_dim=24), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("index_head_dim", why)

    def test_name_has_mfma_tag(self):
        self.assertIn("mfma", self._mfma().kernel_name())

    def test_grid_is_query_by_key_tiles(self):
        # seqlen_q=32, seqlen_k=64 -> (32/16, 64/16, 1)
        self.assertEqual(lightning_indexer_grid(self._mfma(seqlen_q=32)), (2, 4, 1))

    def test_builds_and_lowers(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        kd = build_lightning_indexer(self._mfma(), arch=_ARCH)
        for flavor in ("llvm20", "llvm22"):
            ir = _lower_kernel_to_llvm_python(kd, arch=_ARCH, llvm_flavor=flavor)
            self.assertIn("define", ir, f"no define in MFMA IR for {flavor}")


if __name__ == "__main__":
    unittest.main()
