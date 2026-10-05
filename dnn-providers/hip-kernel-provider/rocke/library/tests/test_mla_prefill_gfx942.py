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

import tempfile
import unittest
from pathlib import Path

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
        # d_v/16 = 4 epilogue n-tiles cannot be split across 8 waves.
        ok, reason = supports_mla_prefill(_spec(num_warps=8, d_v=64), arch="gfx942")
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
        # The probe is single-tile: one MFMA_M-row query block.
        cls.kernel = build_mla_prefill_score_probe(_spec(block_q=16), arch="gfx942")
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
            build_mla_prefill_score_probe(_spec(block_q=16), arch="gfx950")


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
        # Six buffers, 71808 B if none aliased, packed by the lowerer's
        # liveness pass into 27392 B. That is under gfx942's 65536 B workgroup
        # limit -- a hard gate, the kernel cannot launch above it -- and, the
        # point of the layout, under 32768 B, so two workgroups fit per CU.
        #
        # The achieved layout at block_q=32, by phase (bf16 except s_part,
        # which is f32). qa_lds holds one 288-column window of the absorbed
        # query and accl_lds one wave's 128 latent columns, each for both
        # 16-row M tiles:
        #
        #   offset  buffer                       bytes
        #        0  wq_lds / s_part / accl_lds   8704 / 8192 / 8448
        #     8704  qa_lds / kv_lds / wt_lds     18688 / 18560 / 9216
        #                                 total = 27392 (prologue peak)
        #
        # Each phase packs from the base on its own: the prologue pair is dead
        # before the k-loop (the score A operand is read into registers
        # first), and everything is dead by the epilogue. The small buffer of
        # each later phase is allocated first so it takes the base slot and
        # the large one lands on the next dead slot. ``_fwd_lds_bytes`` models
        # exactly this so the admission check can predict it without lowering.
        self.assertIn(f"@smem_pool.{self.kernel.name}", self.ir)
        self.assertIn("[27392 x i8]", self.ir)

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


def _hpw_spec(**kw) -> MlaPrefillSpec:
    kw.setdefault("block_q", 16)
    kw.setdefault("heads_per_wg", 4)
    return _spec(num_heads=kw.pop("num_heads", 128), **kw)


class TestMlaPrefillHeadsPerWg(unittest.TestCase):
    """``heads_per_wg > 1``: one wave per head, the key tile shared by all."""

    def test_default_is_one_head(self):
        spec = _spec()
        self.assertEqual(spec.heads_per_wg, 1)
        self.assertNotIn("hpw", spec.fwd_kernel_name())

    def test_name_and_grid_carry_the_packing(self):
        spec = _hpw_spec()
        self.assertTrue(spec.fwd_kernel_name().endswith("_w4_hpw4_bf16"))
        self.assertEqual(mla_prefill_fwd_grid(spec, num_q_blocks=7), (7, 32, 1))

    def test_heads_must_divide(self):
        with self.assertRaisesRegex(ValueError, "multiple of heads_per_wg"):
            _hpw_spec(num_heads=6)

    def test_bad_packing_rejected(self):
        with self.assertRaisesRegex(ValueError, "heads_per_wg must be one of"):
            _hpw_spec(heads_per_wg=3)

    def test_one_wave_per_head(self):
        ok, reason = supports_mla_prefill(_hpw_spec(num_warps=8), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("one wave per head", reason)

    def test_one_m_tile(self):
        ok, reason = supports_mla_prefill(_hpw_spec(block_q=32), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("block_q=16", reason)

    def test_supported(self):
        self.assertEqual(supports_mla_prefill(_hpw_spec(), arch="gfx942"), (True, ""))

    def test_lowers_with_the_predicted_pool(self):
        spec = _hpw_spec()
        kernel = build_mla_prefill_fwd(spec, arch="gfx942")
        ir = lower_kernel_to_llvm(kernel, arch="gfx942")
        # The loop holds one shared kv_lds (18560 B) plus every head's parked
        # Q_rope (8704 B); the epilogue's W_UV block for all four heads
        # (17408 B) pools onto them. Under 32768 B: two workgroups per CU.
        self.assertIn(f"@smem_pool.{kernel.name}", ir)
        self.assertIn("[27264 x i8]", ir)
        self.assertNotIn("scratch", ir)
        self.assertNotIn("ds.read.tr", ir)


if __name__ == "__main__":
    unittest.main()


class _OccupancyGuard:
    """The compiled forward kernel still fits two workgroups per CU.

    ``analyze_hsaco`` reports resources, not occupancy, so it is derived here
    with the model from the case-study README (gfx942: 65536 B LDS per CU, 512
    VGPRs per SIMD, 4 SIMDs per CU, unified VGPR/AGPR file, so the reported
    VGPR count is the total):

        workgroups/CU = min( 65536 // lds_bytes,
                             (waves_per_simd_vgpr * 4) // waves_per_workgroup )
        waves_per_simd_vgpr = min( 512 // round_up(vgpr, 8), 8 )

    With 4 waves per workgroup, two workgroups per CU therefore needs
    LDS <= 32768 B *and* VGPR <= 256 at the same time. Losing either halves the
    resident workgroups, and neither the IR golden nor the LDS-pool assertion
    sees the VGPR half of it. This compiles through comgr (no GPU launch) and
    checks the code-object metadata the hardware schedules by.
    """

    LDS_PER_CU = 65536
    VGPRS_PER_SIMD = 512
    SIMDS_PER_CU = 4

    @staticmethod
    def make_spec():  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        # Import torch first when it is installed: it binds the torch-bundled
        # comgr, the one the benchmark and dispatch path compile with. The
        # system comgr is a different LLVM vintage and allocates a few more
        # VGPRs, so without this the guard would check a code object the
        # runtime never runs.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        try:
            from rocke.analysis.isa import analyze_hsaco
            from rocke.helpers.compile import compile_kernel
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"comgr/resource tools unavailable: {exc}")

        cls.spec = cls.make_spec()
        try:
            art = compile_kernel(
                build_mla_prefill_fwd(cls.spec, arch="gfx942"),
                arch="gfx942",
                capture_ir_text=False,
            )
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"comgr toolchain unavailable: {exc}")
        except RuntimeError as exc:
            # A forced ROCKE_LLVM_FLAVOR that the loaded comgr cannot take is an
            # environment mismatch, not a kernel defect. Any other compile
            # failure is real.
            if "vintage mismatch" not in str(exc):
                raise
            raise unittest.SkipTest(f"comgr cannot take this IR flavor: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mla_prefill_fwd.hsaco"
            path.write_bytes(bytes(art.hsaco))
            try:
                cls.res = analyze_hsaco(path).resources
            except (FileNotFoundError, RuntimeError) as exc:  # pragma: no cover
                raise unittest.SkipTest(f"HSACO introspection unavailable: {exc}")

    def test_lds_fits_two_workgroups(self):
        self.assertIsNotNone(self.res.lds_bytes)
        self.assertLessEqual(self.res.lds_bytes, self.LDS_PER_CU // 2)

    def test_vgpr_fits_two_waves_per_simd(self):
        self.assertIsNotNone(self.res.vgpr_count)
        self.assertLessEqual(self.res.vgpr_count, self.VGPRS_PER_SIMD // 2)

    def test_no_scratch(self):
        self.assertEqual(self.res.scratch_bytes, 0)

    def test_two_workgroups_per_cu(self):
        vgpr = -(-self.res.vgpr_count // 8) * 8
        waves_per_simd = min(self.VGPRS_PER_SIMD // vgpr, 8)
        by_vgpr = (waves_per_simd * self.SIMDS_PER_CU) // self.spec.num_warps
        by_lds = self.LDS_PER_CU // self.res.lds_bytes
        self.assertGreaterEqual(min(by_lds, by_vgpr), 2)


class TestShippedLayoutOccupancy(_OccupancyGuard, unittest.TestCase):
    """The layout dispatch ships for a DeepSeek-V3-shaped request."""

    @staticmethod
    def make_spec():
        from dispatch.mla import MLARequest, dispatch_mla

        return dispatch_mla(
            MLARequest(
                num_heads=128, total_q=64, num_seqs=2, max_seqlen_k=96, arch="gfx942"
            )
        ).spec


class TestDefaultLayoutOccupancy(_OccupancyGuard, unittest.TestCase):
    """The single-head default, which dispatch falls back to."""

    @staticmethod
    def make_spec():
        return _spec(num_heads=128)
