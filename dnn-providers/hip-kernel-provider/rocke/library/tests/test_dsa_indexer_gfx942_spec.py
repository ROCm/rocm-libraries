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


class TestOffsetOverflowGuard(unittest.TestCase):
    def test_rejects_index_key_i32_overflow(self):
        # index-key base is s*index_head_dim; 2**24 * 128 == 2**31 overflows i32.
        ok, why = is_valid_spec(_spec(seqlen_k=2**24, index_head_dim=128), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("index-key", why)

    def test_accepts_just_under_index_key_limit(self):
        # One tile below the limit: (2**24 - 16) * 128 < 2**31.
        ok, why = is_valid_spec(
            _spec(seqlen_k=2**24 - 16, index_head_dim=128), arch=_ARCH
        )
        self.assertTrue(ok, why)


class TestQPosBaseGuard(unittest.TestCase):
    # q_pos_base is a runtime i32; pos(q) = q_pos_base + q (q in [0, seqlen_q)) must
    # fit i32 or the causal compare wraps. The guard is at the host entry and is
    # body-independent (the same add is in both bodies).
    def _validate(self, q_pos_base, seqlen_q):
        from builders.common.dsa.manifest import _validate_q_pos_base

        _validate_q_pos_base(q_pos_base, seqlen_q)

    def test_rejects_wrapping_last_row(self):
        with self.assertRaises(ValueError):
            self._validate(2**31 - 1, seqlen_q=8)  # +7 wraps past INT32_MAX

    def test_accepts_max_safe_base(self):
        self._validate(2**31 - 8, seqlen_q=8)  # +7 == INT32_MAX, still fits

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            self._validate(-1, seqlen_q=8)


class TestRuntimeOffsetGuard(unittest.TestCase):
    # seqlen_q/seqlen_k are runtime kernel args now, so the i32 offset bound is
    # enforced at the manifest host entry against the actual launch shape.
    def _validate(self, q, k, h, d):
        from builders.common.dsa.manifest import _validate_offsets

        _validate_offsets(q, k, h, d)

    def test_rejects_index_key_overflow(self):
        # seqlen_k*index_head_dim: 2**24 * 128 == 2**31 overflows i32.
        with self.assertRaises(ValueError):
            self._validate(1, 2**24, 4, 128)

    def test_rejects_score_matrix_overflow(self):
        # seqlen_q*seqlen_k >= 2**31.
        with self.assertRaises(ValueError):
            self._validate(2**16, 2**15, 4, 16)

    def test_accepts_realistic_shape(self):
        self._validate(1, 131072, 64, 128)  # long-context decode, fits i32


class TestManifestShapeGuard(unittest.TestCase):
    # n_index_heads and index_head_dim are compiled into the kernel, so the runner
    # must reject a --shape that changes them (OOB), and must not default a missing
    # body (an MFMA binary launched on the scalar grid runs past the buffers).
    def _run(self, manifest, shape):
        from builders.common.dsa.manifest import (
            run_lightning_indexer_manifest_problem,
        )

        return run_lightning_indexer_manifest_problem(manifest, shape, verify=False)

    def _manifest(self, **over):
        m = dict(
            n_index_heads=4,
            index_head_dim=16,
            body="scalar",
            default_shape=[8, 64, 4],
            threads_per_block=64,
            q_pos_base=0,
            args_signature=[],
        )
        m.update(over)
        return m

    def test_rejects_shape_overriding_n_index_heads(self):
        with self.assertRaises(ValueError):
            self._run(self._manifest(), shape=(8, 64, 8))  # HI 8 != built 4

    def test_accepts_shape_changing_only_seqlens(self):
        # seqlen_q/seqlen_k are runtime; overriding them (HI unchanged) is allowed.
        make_args, grid, block, *_ = self._run(self._manifest(), shape=(16, 128, 4))
        self.assertEqual(grid, (16, 2, 1))  # scalar: (seqlen_q, ceil(128/64), 1)

    def test_rejects_missing_body(self):
        m = self._manifest()
        del m["body"]
        with self.assertRaises(ValueError):
            self._run(m, shape=None)

    def test_mfma_grid_from_helper(self):
        # The grid comes from lightning_indexer_grid, not a hand copy: mfma ceils
        # both axes. seqlen_q=24, seqlen_k=70 -> (ceil(24/16), ceil(70/16), 1).
        _, grid, *_ = self._run(
            self._manifest(body="mfma", threads_per_block=64), shape=(24, 70, 4)
        )
        self.assertEqual(grid, (2, 5, 1))


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
        self.assertIn("b256", name)
        # seqlen_q / seqlen_k are runtime args, so they must NOT be in the name
        # (one binary per (H_I, D_I, body) serves every sequence length).
        self.assertNotIn("Q8", name)
        self.assertNotIn("K64", name)

    def test_seqlens_do_not_change_name(self):
        # Two specs differing only in the sequence lengths share one kernel.
        a = _spec(n_index_heads=32, index_head_dim=128, seqlen_q=8, seqlen_k=64)
        b = _spec(n_index_heads=32, index_head_dim=128, seqlen_q=999, seqlen_k=7777)
        self.assertEqual(a.kernel_name(), b.kernel_name())


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
        # Keys parallelize across grid.y = ceil(seqlen_k/block_size); block_size 64
        # here, so seqlen_k=200 -> ceil(200/64) = 4.
        self.assertEqual(
            lightning_indexer_grid(_spec(seqlen_q=4, seqlen_k=200)), (4, 4, 1)
        )
        names = [s["name"] for s in lightning_indexer_signature(spec)]
        self.assertEqual(
            names,
            ["index_q", "index_k", "w", "scores", "q_pos_base", "seqlen_q", "seqlen_k"],
        )

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

    def test_accepts_nonaligned_seqlen(self):
        # The MFMA body tail-masks non-16-aligned seqlen_q / seqlen_k (ceil grid,
        # clamped loads, predicated stores), so they are valid now.
        ok, why = is_valid_spec(self._mfma(seqlen_q=24, seqlen_k=70), arch=_ARCH)
        self.assertTrue(ok, why)

    def test_requires_aligned_dim(self):
        ok, why = is_valid_spec(self._mfma(index_head_dim=24), arch=_ARCH)
        self.assertFalse(ok)
        self.assertIn("index_head_dim", why)

    def test_name_has_mfma_tag(self):
        self.assertIn("mfma", self._mfma().kernel_name())

    def test_grid_is_query_by_key_tiles(self):
        # seqlen_q=32, seqlen_k=64 -> (32/16, 64/16, 1)
        self.assertEqual(lightning_indexer_grid(self._mfma(seqlen_q=32)), (2, 4, 1))
        # non-aligned -> ceil: seqlen_q=24, seqlen_k=70 -> (ceil(24/16), ceil(70/16))
        self.assertEqual(
            lightning_indexer_grid(self._mfma(seqlen_q=24, seqlen_k=70)), (2, 5, 1)
        )

    def test_builds_and_lowers(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        kd = build_lightning_indexer(self._mfma(), arch=_ARCH)
        for flavor in ("llvm20", "llvm22"):
            ir = _lower_kernel_to_llvm_python(kd, arch=_ARCH, llvm_flavor=flavor)
            self.assertIn("define", ir, f"no define in MFMA IR for {flavor}")


if __name__ == "__main__":
    unittest.main()
