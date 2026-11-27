# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only tests of the batchnorm2d kernel families beyond the base forward kernel.

Validity, runtime admission, launch plans, grids, workspace sizes and kernel names of
every spec kind (flat inference, dynamic split, single-launch and split backward,
NHWC, multi-channel-per-wave, read-once), the shipped-set sizes, the per-row candidate
enumeration, fp32 acceptance, and that every shipped kernel builds verifier-clean IR.
No GPU, no compiler, no torch.
"""

from __future__ import annotations

import unittest

from rocke.core.verify import verify_or_raise
from rocke.dispatch.families import batchnorm as dsp
from rocke.instances.common import batchnorm2d as bn
from rocke.instances.common import batchnorm2d_ro as ro

ARCH = "gfx950"


def _req(n, c, h, w, **kw):
    kw.setdefault("arch", ARCH)
    return dsp.BatchNormRequest(n=n, c=c, h=h, w=w, **kw)


class TestShippedSets(unittest.TestCase):
    def test_mcw_shipped_set_has_58_candidates(self):
        sizes = {k: len(v) for k, v in dsp._MCW_SHIPPED.items()}
        self.assertEqual(sum(sizes.values()), 58)
        total = 0
        for mode in ("train", "bwd"):
            for dtype in ("fp16", "bf16", "fp32"):
                got = dsp.shipped_batchnorm_mcw_candidates(mode, dtype)
                self.assertEqual(len(got), sizes[(dtype, mode)], (mode, dtype))
                total += len(got)
        self.assertEqual(total, 58)

    def test_mcw_shipped_candidates_are_valid_specs(self):
        for mode in ("train", "bwd"):
            for dtype in ("f16", "bf16", "f32"):
                d = {"f16": "fp16", "f32": "fp32"}.get(dtype, dtype)
                for cand in dsp.shipped_batchnorm_mcw_candidates(mode, d):
                    ok, why = bn.is_valid_mcw_spec(cand.spec(dtype, ARCH), ARCH)
                    self.assertTrue(ok, f"{cand.name} {dtype}: {why}")

    def test_nhwc_shipped_set_has_68_kernels(self):
        kernels = 0
        for mode in ("infer", "train", "bwd"):
            cands = dsp.dynamic_batchnorm_nhwc_candidates(mode)
            self.assertEqual(len({c.name for c in cands}), len(cands))
            kernels += sum(len(c.kinds) for c in cands)
        self.assertEqual(kernels, 68)
        self.assertEqual(
            [
                len(dsp.dynamic_batchnorm_nhwc_candidates(m))
                for m in ("infer", "train", "bwd")
            ],
            [12, 14, 14],
        )

    def test_nhwc_shipped_geometries_are_valid(self):
        for mode in ("infer", "train", "bwd"):
            for cand in dsp.dynamic_batchnorm_nhwc_candidates(mode):
                for dtype in ("f16", "bf16"):
                    ok, why = bn.is_valid_nhwc_spec(cand.spec(dtype, ARCH), ARCH)
                    self.assertTrue(ok, f"{cand.name} {dtype}: {why}")

    def test_registered_set_sizes(self):
        self.assertEqual(len(dsp.flat_infer_batchnorm_candidates()), 84)
        self.assertEqual(len(dsp.dynamic_batchnorm_candidates()), 480 + 84)
        self.assertEqual(len(dsp.dynamic_batchnorm_bwd_candidates()), 168)
        self.assertEqual(len(dsp.dynamic_batchnorm_split_candidates()), 18)
        self.assertEqual(len(dsp.dynamic_batchnorm_bwd_split_candidates()), 18)

    def test_candidate_names_are_unique(self):
        for group in (
            dsp.dynamic_batchnorm_candidates(),
            dsp.dynamic_batchnorm_bwd_candidates(),
            dsp.dynamic_batchnorm_split_candidates(),
            dsp.dynamic_batchnorm_bwd_split_candidates(),
            dsp.dynamic_batchnorm_mcw_candidates("train"),
            dsp.dynamic_batchnorm_mcw_candidates("bwd"),
            dsp.dynamic_batchnorm_ro_candidates(False),
            dsp.dynamic_batchnorm_ro_candidates(True),
        ):
            names = [c.name for c in group]
            self.assertEqual(len(names), len(set(names)))
        # a forward and a backward candidate never share a kernel name
        fwd = {bn.BatchNorm2DSplitDynSpec().kernel_name("reduce")}
        self.assertNotIn(bn.BatchNorm2DBwdSplitSpec().kernel_name("reduce"), fwd)


class TestForwardSpecs(unittest.TestCase):
    def _dyn(self, **kw):
        base = dict(
            n=1, c=1, h=1, w=1, dynamic=True, training=True, save_mean_invstd=True
        )
        base.update(kw)
        return bn.BatchNorm2DSpec(**base)

    def test_f32_vec_rules(self):
        self.assertTrue(bn.is_valid_spec(self._dyn(dtype="f32", vec=4), ARCH)[0])
        self.assertTrue(bn.is_valid_spec(self._dyn(dtype="f32", vec=2), ARCH)[0])
        self.assertFalse(bn.is_valid_spec(self._dyn(dtype="f32", vec=8), ARCH)[0])
        self.assertTrue(bn.is_valid_spec(self._dyn(dtype="f16", vec=8), ARCH)[0])

    def test_saved_stats_need_training(self):
        ok, why = bn.is_valid_spec(self._dyn(training=False), ARCH)
        self.assertFalse(ok)
        self.assertIn("save_mean_invstd", why)

    def test_train_and_infer_kernel_names_differ(self):
        # one candidate serves both modes; the AOT cache is keyed by this name
        train = self._dyn().kernel_name()
        infer = self._dyn(training=False, save_mean_invstd=False).kernel_name()
        self.assertNotEqual(train, infer)
        self.assertIn("train", train)
        self.assertNotIn("train", infer)
        self.assertIn("smv", train)
        self.assertIn("dyn", train)

    def test_dtype_is_part_of_the_kernel_name(self):
        names = {self._dyn(dtype=d).kernel_name() for d in ("f16", "bf16", "f32")}
        self.assertEqual(len(names), 3)

    def test_dynamic_candidates_accept_f32(self):
        for training in (True, False):
            req = _req(4, 32, 14, 14, training=training, dtype="fp32")
            admitted = [
                c for c in dsp.dynamic_batchnorm_candidates() if c.admits(req)[0]
            ]
            self.assertTrue(admitted, f"no f32 candidate (training={training})")
            for cand in admitted:
                self.assertEqual(cand.select_spec(req).dtype, "f32")
                self.assertTrue(
                    bn.is_valid_spec(cand.select_spec(req), ARCH)[0]
                    or isinstance(cand.select_spec(req), bn.BatchNorm2DFlatInferSpec)
                )


class TestFlatInfer(unittest.TestCase):
    def test_spec_validity(self):
        S = bn.BatchNorm2DFlatInferSpec
        self.assertTrue(bn.is_valid_flat_infer_spec(S(), ARCH)[0])
        self.assertFalse(bn.is_valid_flat_infer_spec(S(dtype="f32"), ARCH)[0])
        self.assertFalse(bn.is_valid_flat_infer_spec(S(vec=1, aligned=False), ARCH)[0])
        self.assertFalse(bn.is_valid_flat_infer_spec(S(block_size=100), ARCH)[0])
        self.assertFalse(bn.is_valid_flat_infer_spec(S(unroll=3), ARCH)[0])

    def test_admission_aligned_vs_unaligned(self):
        al = bn.BatchNorm2DFlatInferSpec(vec=8, aligned=True)
        un = bn.BatchNorm2DFlatInferSpec(vec=8, aligned=False)
        self.assertTrue(bn.flat_infer_admits(al, 2, 4, 8, 8)[0])  # hw 64 % 8 == 0
        self.assertFalse(bn.flat_infer_admits(al, 2, 4, 3, 5)[0])
        self.assertTrue(
            bn.flat_infer_admits(un, 2, 8, 3, 5)[0]
        )  # hw 15, total 240 % 8 == 0
        self.assertFalse(bn.flat_infer_admits(un, 2, 4, 8, 8)[0])
        self.assertFalse(bn.flat_infer_admits(un, 1, 1, 1, 7)[0])  # hw < vec

    def test_grid_covers_the_tensor(self):
        spec = bn.BatchNorm2DFlatInferSpec(block_size=256, vec=4, unroll=2)
        self.assertEqual(spec.tile_elems, 2048)
        self.assertEqual(bn.batchnorm2d_flat_infer_grid(2, 3, 32, 32, spec), (3, 1, 1))
        self.assertEqual(bn.batchnorm2d_flat_infer_grid(1, 1, 1, 1, spec), (1, 1, 1))

    def test_inference_only_and_row_class_enumeration(self):
        req = _req(8, 64, 14, 14, training=False)
        flat = [c for c in dsp.flat_infer_batchnorm_candidates() if c.admits(req)[0]]
        self.assertTrue(flat)
        # hw 196: vec 8 only unaligned, vec 4 / 2 / 1 aligned
        for c in flat:
            self.assertEqual(
                c.spec_id.endswith("_al"), "_v8_" not in c.spec_id, c.spec_id
            )
        req_t = _req(8, 64, 14, 14, training=True)
        self.assertFalse(
            [c for c in dsp.flat_infer_batchnorm_candidates() if c.admits(req_t)[0]]
        )
        odd = _req(8, 64, 7, 7, training=False)  # hw 49: only vec 1 is aligned
        got = [c for c in dsp.flat_infer_batchnorm_candidates() if c.admits(odd)[0]]
        aligned = [c for c in got if c.spec_id.endswith("_al")]
        self.assertTrue(aligned)
        self.assertTrue(all("_v1_" in c.spec_id for c in aligned))
        self.assertTrue([c for c in got if not c.spec_id.endswith("_al")])


class TestSplit(unittest.TestCase):
    def test_validity_per_dtype(self):
        S = bn.BatchNorm2DSplitDynSpec
        self.assertTrue(bn.is_valid_split_dyn_spec(S(vec=8), ARCH)[0])
        self.assertFalse(bn.is_valid_split_dyn_spec(S(dtype="f32", vec=8), ARCH)[0])
        self.assertTrue(bn.is_valid_split_dyn_spec(S(dtype="f32", vec=4), ARCH)[0])
        self.assertFalse(bn.is_valid_split_dyn_spec(S(vec=3), ARCH)[0])
        self.assertTrue(
            bn.is_valid_split_dyn_spec(bn.BatchNorm2DBwdSplitSpec(), ARCH)[0]
        )

    def test_plan_geometry(self):
        spec = bn.BatchNorm2DSplitDynSpec(block_size=128, vec=4)
        plan = bn.batchnorm2d_split_plan(spec, 4, 32, 32, sn_req=2, sh_req=2)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.hw_chunk % (128 * 4), 0)
        self.assertEqual((plan.sn, plan.sh), (2, 2))
        self.assertEqual(plan.splits, 4)
        self.assertGreaterEqual(plan.n_chunk * plan.sn, 4)
        self.assertGreaterEqual(plan.hw_chunk * plan.sh, 32 * 32)
        # a single split is not a split plan
        self.assertIsNone(
            bn.batchnorm2d_split_plan(spec, 4, 32, 32, sn_req=1, sh_req=1)
        )
        # no more splits than threads
        self.assertIsNone(
            bn.batchnorm2d_split_plan(spec, 1024, 32, 32, sn_req=1024, sh_req=1)
        )

    def test_grid_and_workspace(self):
        spec = bn.BatchNorm2DSplitDynSpec(block_size=128, vec=4)
        plan = bn.batchnorm2d_split_plan(spec, 4, 32, 32, sn_req=2, sh_req=2)
        self.assertEqual(
            bn.batchnorm2d_split_dyn_grid(64, plan), (64, plan.sn, plan.sh)
        )
        fwd = bn.batchnorm2d_split_dyn_workspace_floats(64, plan)
        self.assertGreaterEqual(
            fwd, 64 * plan.splits * 3
        )  # (mean, M2, count) per split
        self.assertGreater(fwd, 0)
        self.assertEqual(
            bn.batchnorm2d_bwd_split_workspace_floats(64, plan), 64 * plan.splits * 2
        )

    def test_kernel_names(self):
        spec = bn.BatchNorm2DSplitDynSpec(fast_reduce=True)
        red, app = spec.kernel_name("reduce"), spec.kernel_name("apply")
        self.assertNotEqual(red, app)
        self.assertIn("reduce", red)
        self.assertIn("fr", red)
        self.assertNotEqual(
            red, bn.BatchNorm2DBwdSplitSpec(fast_reduce=True).kernel_name("reduce")
        )

    def test_candidate_plans_respect_the_cta_window(self):
        req = _req(32, 128, 56, 56, dtype="fp16")
        cands = [
            c for c in dsp.dynamic_batchnorm_split_candidates() if c.admits(req)[0]
        ]
        self.assertTrue(cands)
        for cand in cands:
            for plan in cand.plans(req):
                ctas = req.c * plan.splits
                self.assertTrue(dsp._SPLIT_MIN_CTAS <= ctas <= dsp._SPLIT_MAX_CTAS)
        # training forward only
        self.assertFalse(
            [
                c
                for c in dsp.dynamic_batchnorm_split_candidates()
                if c.plans(_req(32, 128, 56, 56, training=False))
            ]
        )

    def test_split_backward_candidates_follow_the_backward_flag(self):
        fwd = _req(32, 128, 56, 56)
        bwd = _req(32, 128, 56, 56, backward=True)
        self.assertTrue(
            any(c.plans(bwd) for c in dsp.dynamic_batchnorm_bwd_split_candidates())
        )
        self.assertFalse(
            any(c.plans(fwd) for c in dsp.dynamic_batchnorm_bwd_split_candidates())
        )
        self.assertFalse(
            any(c.plans(bwd) for c in dsp.dynamic_batchnorm_split_candidates())
        )


class TestBackward(unittest.TestCase):
    def test_validity(self):
        S = bn.BatchNorm2DBwdSpec
        self.assertTrue(bn.is_valid_bwd_spec(S(), ARCH)[0])
        self.assertTrue(bn.is_valid_bwd_spec(S(dtype="f32", vec=4), ARCH)[0])
        self.assertFalse(bn.is_valid_bwd_spec(S(dtype="f32", vec=8), ARCH)[0])
        self.assertFalse(bn.is_valid_bwd_spec(S(block_size=256, n_groups=3), ARCH)[0])
        self.assertFalse(
            bn.is_valid_bwd_spec(S(block_size=100, fast_reduce=True), ARCH)[0]
        )

    def test_grid_and_name(self):
        self.assertEqual(
            bn.batchnorm2d_bwd_grid(64, bn.BatchNorm2DBwdSpec(channels_per_block=4)),
            (16, 1, 1),
        )
        name = bn.BatchNorm2DBwdSpec(
            channels_per_block=2, n_groups=4, fast_reduce=True
        ).kernel_name()
        for token in ("bwd", "cpb2", "ng4", "fr", "dyn"):
            self.assertIn(token, name)

    def test_row_class_enumeration(self):
        # backward requests only reach backward candidates, admission follows cpb / ng divisibility
        bwd = _req(8, 64, 14, 14, backward=True)
        got = [c for c in dsp.dynamic_batchnorm_bwd_candidates() if c.admits(bwd)[0]]
        self.assertTrue(got)
        self.assertTrue(any("_ng8" in c.spec_id for c in got))
        self.assertTrue(any("_cpb8" in c.spec_id for c in got))
        odd = _req(
            3, 12, 14, 14, backward=True
        )  # n=3 admits ng 1 only, c=12 admits cpb 1, 2, 4
        names = [
            c.spec_id
            for c in dsp.dynamic_batchnorm_bwd_candidates()
            if c.admits(odd)[0]
        ]
        self.assertFalse(any("_ng" in n for n in names))
        self.assertFalse(any("_cpb8" in n for n in names))
        self.assertTrue(any("_cpb4" in n for n in names))
        fwd = _req(8, 64, 14, 14)
        self.assertFalse(
            [c for c in dsp.dynamic_batchnorm_bwd_candidates() if c.admits(fwd)[0]]
        )

    def test_backward_accepts_f32(self):
        req = _req(8, 64, 14, 14, backward=True, dtype="fp32")
        self.assertTrue(
            [c for c in dsp.dynamic_batchnorm_bwd_candidates() if c.admits(req)[0]]
        )


class TestNhwc(unittest.TestCase):
    def test_spec_validity(self):
        S = bn.BatchNorm2DNhwcSpec
        self.assertTrue(bn.is_valid_nhwc_spec(S(), ARCH)[0])
        self.assertFalse(bn.is_valid_nhwc_spec(S(dtype="f32"), ARCH)[0])
        self.assertFalse(bn.is_valid_nhwc_spec(S(tx=6), ARCH)[0])
        self.assertFalse(bn.is_valid_nhwc_spec(S(mode="bogus"), ARCH)[0])
        self.assertFalse(bn.is_valid_nhwc_spec(S(vec=3), ARCH)[0])

    def test_admission(self):
        spec = bn.BatchNorm2DNhwcSpec(tx=8, vec=8)  # 64-channel tile
        self.assertTrue(bn.nhwc_admits(spec, 4, 128, 8, 8)[0])
        self.assertFalse(bn.nhwc_admits(spec, 4, 100, 8, 8)[0])  # C % vec
        self.assertFalse(bn.nhwc_admits(spec, 4, 32, 8, 8)[0])  # tile more than twice C
        self.assertFalse(bn.nhwc_admits(spec, 0, 128, 8, 8)[0])

    def test_plans_grid_and_workspace(self):
        spec = bn.BatchNorm2DNhwcSpec(
            mode="train", block_size=256, tx=8, vec=4, unroll=4
        )
        n, c, h, w = 32, 64, 16, 16
        plans = bn.batchnorm2d_nhwc_plans(spec, "split", n, c, h, w)
        self.assertTrue(plans)
        rows = n * h * w
        for p in plans:
            self.assertEqual(p.splits, -(-rows // p.rows_per_cta))
            self.assertEqual(p.rows_per_cta % (spec.ty * spec.unroll), 0)
            self.assertEqual(bn.batchnorm2d_nhwc_grid(c, p), (p.splits, p.tiles, 1))
            self.assertEqual(
                bn.batchnorm2d_nhwc_workspace_floats(c, p), 2 * c * p.splits
            )
            self.assertEqual(p.tiles, -(-c // spec.tile_channels))
        self.assertEqual(
            bn.batchnorm2d_nhwc_plans(spec, "fused", 32, 8, 4, 4), ()
        )  # C not admitted

    def test_inference_has_no_split_variant(self):
        spec = bn.BatchNorm2DNhwcSpec(mode="infer")
        self.assertEqual(bn.batchnorm2d_nhwc_plans(spec, "split", 32, 64, 16, 16), ())
        self.assertTrue(bn.batchnorm2d_nhwc_plans(spec, "infer", 32, 64, 16, 16))

    def test_candidate_plans_per_row_class(self):
        for mode in ("infer", "train", "bwd"):
            cands = dsp.dynamic_batchnorm_nhwc_candidates(mode)
            big = sum(len(c.plans(32, 256, 28, 28, "f16", ARCH)) for c in cands)
            small = sum(len(c.plans(1, 4, 2, 2, "f16", ARCH)) for c in cands)
            self.assertGreater(big, 0, mode)
            self.assertEqual(small, 0, mode)
            # f32 NHWC is not part of the family
            self.assertEqual(
                sum(len(c.plans(32, 256, 28, 28, "f32", ARCH)) for c in cands), 0
            )

    def test_kernel_names_per_kind(self):
        spec = bn.BatchNorm2DNhwcSpec(mode="train")
        names = {spec.kernel_name(k) for k in ("fused", "reduce", "apply")}
        self.assertEqual(len(names), 3)
        self.assertNotEqual(
            spec.kernel_name("fused"),
            bn.BatchNorm2DNhwcSpec(mode="bwd").kernel_name("fused"),
        )


class TestMcw(unittest.TestCase):
    def test_validity_and_register_budget(self):
        S = bn.BatchNorm2DMcwSpec
        self.assertTrue(bn.is_valid_mcw_spec(S(), ARCH)[0])
        self.assertFalse(bn.is_valid_mcw_spec(S(waves=3), ARCH)[0])
        self.assertFalse(bn.is_valid_mcw_spec(S(cpw=3), ARCH)[0])
        self.assertFalse(bn.is_valid_mcw_spec(S(mode="x"), ARCH)[0])
        self.assertFalse(bn.is_valid_mcw_spec(S(npw=128, slots=16, vec=8), ARCH)[0])
        self.assertFalse(bn.is_valid_mcw_spec(S(vec=8, dtype="f32"), ARCH)[0])
        # backward keeps x and dy per element: twice the registers of training
        self.assertEqual(
            S(mode="bwd", npw=4, slots=2).regs, 2 * S(mode="train", npw=4, slots=2).regs
        )

    def test_admission_matches_geometry(self):
        spec = bn.BatchNorm2DMcwSpec(waves=4, cpw=2, npw=8, slots=2, vec=4)
        self.assertEqual(spec.lpc, 32)
        hw = 4 * spec.lpc * 2  # vec * lpc * slots
        self.assertTrue(bn.mcw_admits(spec, 32, 8, 1, hw)[0])
        self.assertTrue(bn.mcw_admits(spec, 31, 8, 1, hw - 4)[0])  # ragged n and hw
        self.assertFalse(bn.mcw_admits(spec, 16, 8, 1, hw)[0])  # npw would be 4
        self.assertFalse(bn.mcw_admits(spec, 32, 7, 1, hw)[0])  # C % cpw
        self.assertFalse(bn.mcw_admits(spec, 32, 8, 1, hw + 2)[0])  # hw % vec
        self.assertFalse(
            bn.mcw_admits(spec, 32, 8, 1, 4 * spec.lpc * 8)[0]
        )  # slots would be 8
        self.assertFalse(
            bn.mcw_admits(bn.BatchNorm2DMcwSpec(waves=16, npw=1), 4, 8, 1, 64)[0]
        )

    def test_grid_and_name(self):
        spec = bn.BatchNorm2DMcwSpec(waves=2, cpw=4)
        self.assertEqual(bn.batchnorm2d_mcw_grid(64, spec), (16, 1, 1))
        self.assertEqual(spec.block_size, 128)
        names = {
            bn.BatchNorm2DMcwSpec(mode=m, dtype=d).kernel_name()
            for m in ("train", "bwd")
            for d in ("f16", "bf16", "f32")
        }
        self.assertEqual(len(names), 6)

    def test_shipped_candidates_admit_their_own_geometry(self):
        for mode in ("train", "bwd"):
            for dtype in ("fp16", "bf16", "fp32"):
                d = {"fp16": "f16", "fp32": "f32"}.get(dtype, dtype)
                for cand in dsp.shipped_batchnorm_mcw_candidates(mode, dtype):
                    spec = cand.spec(d, ARCH)
                    shape = (
                        cand.waves * cand.npw,
                        2 * cand.cpw,
                        1,
                        cand.vec * spec.lpc * cand.slots,
                    )
                    self.assertTrue(cand.admits(*shape, d, ARCH), (cand.name, d))

    def test_row_class_enumeration(self):
        # a small NCHW row is covered by several shipped candidates, a huge one by none
        first = dsp.shipped_batchnorm_mcw_candidates("train", "fp16")[0]
        spec = first.spec("f16", ARCH)
        shape = (
            first.waves * first.npw,
            2 * first.cpw,
            1,
            first.vec * spec.lpc * first.slots,
        )
        cands = [
            c
            for c in dsp.shipped_batchnorm_mcw_candidates("train", "fp16")
            if c.admits(*shape, "f16", ARCH)
        ]
        self.assertIn(first, cands)
        self.assertFalse(
            [
                c
                for c in dsp.shipped_batchnorm_mcw_candidates("train", "fp16")
                if c.admits(64, 64, 112, 112, "f16", ARCH)
            ]
        )


class TestReadOnce(unittest.TestCase):
    CAP = 2048  # nominal co-resident CTAs

    def test_spec_validity(self):
        S = ro.BatchNorm2DRoSpec
        self.assertTrue(ro.is_valid_ro_spec(S(), ARCH)[0])
        self.assertTrue(ro.is_valid_ro_spec(S(dtype="f32", vec=4), ARCH)[0])
        self.assertFalse(ro.is_valid_ro_spec(S(dtype="f32", vec=8), ARCH)[0])
        self.assertFalse(ro.is_valid_ro_spec(S(block_size=100), ARCH)[0])
        self.assertFalse(ro.is_valid_ro_spec(S(dtype="fp8"), ARCH)[0])
        self.assertFalse(ro.is_valid_ro_spec(S(regs=0), ARCH)[0])

    def test_kernel_names_separate_train_and_backward(self):
        spec = ro.BatchNorm2DRoSpec()
        self.assertNotEqual(spec.kernel_name("train"), spec.kernel_name("bwd"))

    def test_plan_is_co_resident_and_whole_groups(self):
        spec = ro.BatchNorm2DRoSpec(block_size=256, vec=8, regs=32)
        plan = ro.batchnorm2d_ro_plan(
            spec, 32, 64, 56, 56, sn_req=2, sh_req=4, max_resident=self.CAP
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.grid % plan.splits, 0)
        self.assertLessEqual(plan.grid, self.CAP)
        self.assertLessEqual(plan.grid, 64 * plan.splits)
        self.assertLessEqual(plan.n_chunk * plan.hw_chunk, spec.cap_elems)
        # no whole group fits: refused rather than launched
        self.assertIsNone(
            ro.batchnorm2d_ro_plan(
                spec, 32, 64, 56, 56, sn_req=2, sh_req=4, max_resident=plan.splits - 1
            )
        )
        # the tile must fit the register budget
        small = ro.BatchNorm2DRoSpec(block_size=256, vec=8, regs=2)
        self.assertIsNone(
            ro.batchnorm2d_ro_plan(
                small, 32, 64, 56, 56, sn_req=1, sh_req=1, max_resident=self.CAP
            )
        )
        # H*W must be a multiple of vec
        self.assertIsNone(
            ro.batchnorm2d_ro_plan(
                spec, 32, 64, 7, 7, sn_req=2, sh_req=2, max_resident=self.CAP
            )
        )

    def test_sync_buffer_size_scales_with_channels(self):
        self.assertEqual(
            ro.batchnorm2d_ro_sync_ints(2 * 64), 2 * ro.batchnorm2d_ro_sync_ints(64)
        )

    def _plans(self, cand, dtype, n, c, h, w, rule=True):
        return cand.plans(dtype, n, c, h, w, self.CAP, rule=rule)

    def test_admission_rule_split_window(self):
        cand = dsp.BatchNormRoCandidate(256, 8, 32, False)
        plans = self._plans(cand, "fp16", 32, 64, 56, 56)
        self.assertTrue(plans)
        for p in plans:
            self.assertTrue(dsp.RO_MIN_SPLIT <= p.sn * p.sh <= dsp.RO_MAX_SPLIT, p)
        full = self._plans(cand, "fp16", 32, 64, 56, 56, rule=False)
        self.assertGreater(len(full), len(plans))
        self.assertTrue(set(plans) <= set(full))

    def test_admission_rule_size_floor(self):
        cand = dsp.BatchNormRoCandidate(256, 8, 32, False)
        # 7.8 MB of f16 is below the 8 MB floor; 8.4 MB is above it
        below = (4, 64, 125, 120)  # 7_680_000 B
        above = (4, 64, 128, 128)  # 8_388_608 B
        self.assertLess(2 * below[0] * below[1] * below[2] * below[3], dsp.RO_MIN_BYTES)
        self.assertGreaterEqual(
            2 * above[0] * above[1] * above[2] * above[3], dsp.RO_MIN_BYTES
        )
        self.assertEqual(self._plans(cand, "fp16", *below), [])
        self.assertTrue(self._plans(cand, "fp16", *above))
        # the sweep space (rule off) still has the small tensor
        self.assertTrue(self._plans(cand, "fp16", *below, rule=False))

    def test_admission_rule_dtypes(self):
        cand = dsp.BatchNormRoCandidate(256, 4, 32, False)
        self.assertEqual(
            self._plans(cand, "bf16", 32, 64, 56, 56), []
        )  # bf16 is not admitted by the rule
        self.assertTrue(
            self._plans(
                dsp.BatchNormRoCandidate(256, 8, 32, False), "fp16", 32, 64, 56, 56
            )
        )
        f32 = dsp.BatchNormRoCandidate(256, 4, 32, False)
        self.assertTrue(self._plans(f32, "fp32", 16, 64, 56, 56))  # 12.8 MB of f32

    def test_candidate_set_per_dtype(self):
        for backward in (False, True):
            f16 = dsp.dynamic_batchnorm_ro_candidates(backward, "f16")
            f32 = dsp.dynamic_batchnorm_ro_candidates(backward, "f32")
            self.assertEqual(len(f16), len(dsp.RO_BLOCKS) * 2 * len(dsp.RO_REGS))
            self.assertEqual({c.vec for c in f16}, {8, 4})
            self.assertEqual({c.vec for c in f32}, {4, 2})
            self.assertTrue(all(c.backward == backward for c in f16 + f32))


class TestShippedKernelsBuild(unittest.TestCase):
    """Every kind x dtype builds IR the verifier accepts (no compiler involved)."""

    def _verify(self, kernel):
        verify_or_raise(kernel)

    def test_dynamic_forward_and_flat(self):
        for dtype in ("f16", "bf16", "f32"):
            for training in (True, False):
                spec = bn.BatchNorm2DSpec(
                    n=1, c=1, h=1, w=1, dtype=dtype, vec=4, dynamic=True, training=training,
                    save_mean_invstd=training,
                )  # fmt: skip
                self._verify(bn.build_batchnorm2d(spec))
        for dtype in ("f16", "bf16"):
            for aligned, vec in ((True, 8), (False, 4)):
                self._verify(
                    bn.build_batchnorm2d_flat_infer(
                        bn.BatchNorm2DFlatInferSpec(
                            dtype=dtype, vec=vec, aligned=aligned
                        )
                    )
                )

    def test_split_and_backward(self):
        for dtype in ("f16", "bf16", "f32"):
            vec = 4
            for cls, red, app in (
                (
                    bn.BatchNorm2DSplitDynSpec,
                    bn.build_batchnorm2d_split_dyn_reduce,
                    bn.build_batchnorm2d_split_dyn_apply,
                ),
                (
                    bn.BatchNorm2DBwdSplitSpec,
                    bn.build_batchnorm2d_bwd_split_reduce,
                    bn.build_batchnorm2d_bwd_split_apply,
                ),
            ):
                for fr in (False, True):
                    spec = cls(block_size=128, vec=vec, dtype=dtype, fast_reduce=fr)
                    self._verify(red(spec))
                    self._verify(app(spec))
            for ng in (1, 4):
                self._verify(
                    bn.build_batchnorm2d_bwd_dynamic(
                        bn.BatchNorm2DBwdSpec(dtype=dtype, vec=vec, n_groups=ng)
                    )
                )

    def test_nhwc(self):
        for dtype in ("f16", "bf16"):
            for mode in ("infer", "train", "bwd"):
                for cand in dsp.dynamic_batchnorm_nhwc_candidates(mode)[:2]:
                    spec = cand.spec(dtype, ARCH)
                    for kind in cand.kinds:
                        self._verify(cand.build(kind, spec, ARCH))

    def test_mcw_shipped(self):
        for mode in ("train", "bwd"):
            for dtype in ("fp16", "bf16", "fp32"):
                d = {"fp16": "f16", "fp32": "f32"}.get(dtype, dtype)
                for cand in dsp.shipped_batchnorm_mcw_candidates(mode, dtype):
                    self._verify(bn.build_batchnorm2d_mcw(cand.spec(d, ARCH)))

    def test_read_once(self):
        for dtype, vec in (("f16", 8), ("bf16", 8), ("f32", 4)):
            spec = ro.BatchNorm2DRoSpec(vec=vec, regs=4, dtype=dtype)
            self._verify(ro.build_batchnorm2d_ro_train(spec))
            self._verify(ro.build_batchnorm2d_ro_bwd(spec))


if __name__ == "__main__":
    unittest.main()
