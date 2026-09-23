# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only coverage for the gfx942 MLA prefill score probe.

Everything here runs without a GPU: spec validation, the arch gate, launch
geometry, and a full lower of the emitted ``KernelDef`` to LLVM IR. The lowering
test is the cheap guard against the copied-helper failure class -- gfx942 has no
``ds_read_*_tr_*``, so a transpose-read helper lifted from a gfx950 kernel would
lower fine on paper and fault on the device. Asserting the instruction is absent
from the IR catches that here instead of in a numeric run.

Numeric correctness is NOT covered here; that is the on-GPU comparison against
:func:`builders.mla.ref_mla_attn.ref_mla_prefill_scores`.
"""

from __future__ import annotations

import unittest

from kernels.mla.mla_prefill_gfx942 import (
    MlaPrefillSpec,
    build_mla_prefill_fwd,
    build_mla_prefill_score_probe,
    mla_prefill_block,
    mla_prefill_fwd_grid,
    mla_prefill_fwd_signature,
    mla_prefill_score_probe_grid,
    mla_prefill_score_probe_signature,
    supports_mla_prefill,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _spec(**kw) -> MlaPrefillSpec:
    return MlaPrefillSpec(num_heads=kw.pop("num_heads", 16), **kw)


class TestMlaPrefillSpec(unittest.TestCase):
    """``__post_init__`` rejects what no arch could satisfy."""

    def test_defaults_are_consistent(self):
        spec = _spec()
        self.assertEqual(spec.head_dim_qk, 192)
        self.assertEqual(spec.w_uk_cols, 256)
        self.assertEqual(spec.threads, 256)

    def test_non_positive_head_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_heads must be positive"):
            _spec(num_heads=0)

    def test_non_bf16_rejected(self):
        with self.assertRaisesRegex(ValueError, "bf16 only"):
            _spec(dtype="fp16")

    def test_indivisible_r_kv_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of r_kv_tile"):
            _spec(r_kv=520)

    def test_page_block_size_must_match_block_k(self):
        with self.assertRaisesRegex(ValueError, "page_block_size must equal block_k"):
            _spec(page_block_size=32)

    def test_bad_warp_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_warps must be one of"):
            _spec(num_warps=3)


class TestMlaPrefillGate(unittest.TestCase):
    """``supports_*`` reports rather than raises, and only on arch/atom fit."""

    def test_default_spec_supported_on_gfx942(self):
        self.assertEqual(supports_mla_prefill(_spec(), arch="gfx942"), (True, ""))

    def test_other_arch_refused(self):
        ok, reason = supports_mla_prefill(_spec(), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("gfx942-only", reason)

    def test_head_dim_off_the_mfma_grid_refused(self):
        # 8 is positive and not a multiple of the 16-wide MFMA N, so it clears
        # __post_init__ and is caught by the arch gate instead.
        ok, reason = supports_mla_prefill(_spec(d_rope=8), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("d_rope", reason)

    def test_n_tiles_must_cover_every_wave(self):
        # d_nope/16 = 8 n-tiles cannot be split across 16 waves.
        ok, reason = supports_mla_prefill(_spec(num_warps=8, d_nope=64), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("n-tiles", reason)


class TestMlaPrefillLaunchGeometry(unittest.TestCase):
    def test_block_is_flat(self):
        self.assertEqual(mla_prefill_block(_spec()), (256, 1, 1))

    def test_grid_is_q_by_head_by_k(self):
        grid = mla_prefill_score_probe_grid(_spec(), num_q_blocks=7, num_k_tiles=3)
        self.assertEqual(grid, (7, 16, 3))

    def test_empty_grid_rejected(self):
        with self.assertRaises(ValueError):
            mla_prefill_score_probe_grid(_spec(), num_q_blocks=0, num_k_tiles=3)

    def test_fwd_grid_has_no_key_dimension(self):
        # The online softmax state (m, l, acc) cannot be split across
        # workgroups, so one workgroup owns a query tile's whole key range.
        self.assertEqual(mla_prefill_fwd_grid(_spec(), num_q_blocks=7), (7, 16, 1))

    def test_empty_fwd_grid_rejected(self):
        with self.assertRaises(ValueError):
            mla_prefill_fwd_grid(_spec(), num_q_blocks=0)


class TestMlaPrefillFwdSignature(unittest.TestCase):
    """13 args: the probe's pack with a second output pointer in front."""

    def test_input_tail_matches_the_probe(self):
        # The two differ only at the head -- the probe writes one f32 scores
        # buffer, the forward kernel writes out + lse. Everything from q_ptr on
        # is the same pack, so the driver stages both the same way.
        fwd = mla_prefill_fwd_signature(_spec())
        probe = mla_prefill_score_probe_signature(_spec())
        self.assertEqual(len(fwd), 13)
        self.assertEqual(fwd[2:], probe[1:])

    def test_scale_is_raw_f32(self):
        # The ABI carries the raw 1/sqrt(d) scale; log2(e) is folded inside the
        # kernel next to the exp2. A scale declared anything but f32 would mean
        # the driver's struct.pack format and the kernel disagree.
        (scale,) = [
            a for a in mla_prefill_fwd_signature(_spec()) if a["name"] == "scale"
        ]
        self.assertEqual(scale["type"], "f32")

    def test_outputs_lead_the_pack(self):
        fwd = mla_prefill_fwd_signature(_spec())
        self.assertEqual([a["name"] for a in fwd[:2]], ["out_ptr", "lse_ptr"])


class TestMlaPrefillLowering(unittest.TestCase):
    """The probe lowers, and lowers to instructions gfx942 actually has."""

    @classmethod
    def setUpClass(cls):
        cls.kernel = build_mla_prefill_score_probe(_spec(), arch="gfx942")
        cls.ir = lower_kernel_to_llvm(cls.kernel, arch="gfx942")

    def test_lowers_to_nonempty_ir(self):
        self.assertGreater(len(self.ir), 10_000)
        self.assertIn(self.kernel.name, self.ir)

    def test_no_transpose_read(self):
        # ds_read_*_tr_* is gfx950-only. Its presence would mean a helper was
        # copied from a gfx950 kernel.
        self.assertNotIn("ds_read_tr", self.ir)
        self.assertNotIn("ds.read.tr", self.ir)

    def test_uses_the_narrow_bf16_mfma(self):
        # CDNA3 has no wide-K bf16 atom; K-step must be 16.
        self.assertIn("mfma.f32.16x16x16bf16", self.ir)
        self.assertNotIn("16x16x32", self.ir)

    def test_workgroup_size_matches_the_spec(self):
        self.assertIn('"amdgpu-flat-work-group-size"="64,256"', self.ir)

    def test_build_refuses_unsupported_arch(self):
        with self.assertRaises(NotImplementedError):
            build_mla_prefill_score_probe(_spec(), arch="gfx950")


class TestMlaPrefillFwdLowering(unittest.TestCase):
    """The forward kernel lowers, and fits gfx942's 64 KiB of LDS."""

    @classmethod
    def setUpClass(cls):
        cls.kernel = build_mla_prefill_fwd(_spec(num_heads=128), arch="gfx942")
        cls.ir = lower_kernel_to_llvm(cls.kernel, arch="gfx942")

    def test_lowers_to_nonempty_ir(self):
        self.assertGreater(len(self.ir), 10_000)
        self.assertIn(self.kernel.name, self.ir)

    def test_lds_slot_is_shared(self):
        # Eight buffers, 115712 B if none aliased, packed by the lowerer's
        # liveness pass into 63104 B -- under gfx942's 65536 B workgroup
        # limit. This is a hard gate, not a size preference: the kernel
        # cannot launch above 65536.
        #
        # The achieved layout, by phase (bf16, so 2 B/elem):
        #
        #   offset  buffer                       shape            bytes
        #        0  q_lds / kv_lds / accl_lds    three phases alias the base
        #     6144  wq_lds                       [64, 128+8]      17408
        #    23552  qa_lds / wt_lds              [16, 576+4]      18560
        #    42112  ct_lds                       [512, 16+4]      20480
        #    62592  p_lds                        [16, 16]           512
        #                                                 total = 63104
        #
        # ``q_lds`` (6144) + ``wq_lds`` (17408) die at the end of the
        # prologue, and the packer folds ``kv_lds`` (18688) into the hole
        # they leave at the base. The epilogue pair starts from the base
        # again because nothing else is live by then. ``_fwd_lds_bytes``
        # models exactly this so the admission check can predict it without
        # lowering.
        self.assertIn(f"@smem_pool.{self.kernel.name}", self.ir)
        self.assertIn("[63104 x i8]", self.ir)

    def test_no_transpose_read(self):
        self.assertNotIn("ds_read_tr", self.ir)
        self.assertNotIn("ds.read.tr", self.ir)

    def test_uses_the_narrow_bf16_mfma(self):
        self.assertIn("mfma.f32.16x16x16bf16", self.ir)
        self.assertNotIn("16x16x32", self.ir)

    def test_no_scratch_spills(self):
        # A private-memory alloca would mean the accumulator vectors did not
        # stay in registers across the k loop.
        self.assertNotIn("scratch", self.ir)

    def test_build_refuses_unsupported_arch(self):
        with self.assertRaises(NotImplementedError):
            build_mla_prefill_fwd(_spec(), arch="gfx950")


if __name__ == "__main__":
    unittest.main()
