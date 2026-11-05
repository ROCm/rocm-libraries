# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the deterministic dgrad box split-K (DgradConvSpec.pp_box_det):
validation, names / labels, the host workspace layout (pp_box_det_pack) and
the stage-2 fold kernel's IR build."""

from __future__ import annotations

import math
import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_dgrad_box_det import (
    PBD_ENTRY_WORDS,
    DgradBoxDetReduceSpec,
    build_dgrad_box_det_reduce,
    pp_box_det_pack,
)
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
    _MAGIC_FIELDS,
    _RECORD_FIELDS,
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
    is_valid_dgrad_spec,
    pp_box_sub_gemms,
)

_ARCH = "gfx942"


def _p3(D=18, H=10, W=10, C=256, K=128, N=1):
    return ConvProblem(
        N=N, Di=D, Hi=H, Wi=W, C=C, K=K, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=0, pW=0
    )


def _spec(problem=None, **kw):
    base = dict(
        problem=problem if problem is not None else _p3(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        pipeline="mem", epilogue="cshuffle", lds_k_outer=False,
        tap_outer_k=True, tapk_prefetch=1, b_kpack=4,
        lds_xor=True, lds_k_pad=0,
        tapk_pingpong=4, tapk_pp_sched=2,
    )
    base.update(kw)
    return DgradConvSpec(**base)


class TestValidation(unittest.TestCase):
    def test_default_off(self):
        self.assertEqual(DgradConvSpec.__dataclass_fields__["pp_box_det"].default, 0)
        self.assertNotIn("pbd", _spec(split_k=8, pp_boxes=7).kernel_name())

    def test_valid(self):
        for v in (1, 2, 3, 4):
            s = _spec(split_k=8, pp_boxes=7, pp_box_det=v)
            ok, why = is_valid_dgrad_spec(s, _ARCH)
            self.assertTrue(ok, why)
            self.assertIn(f"pbd{v}", s.kernel_name())

    def test_invalid(self):
        for kw in (
            dict(pp_box_det=1),  # no boxes
            dict(pp_box_det=1, pp_boxes=7),  # no split
            dict(pp_box_det=5, pp_boxes=7, split_k=8),
        ):
            ok, _ = is_valid_dgrad_spec(_spec(**kw), _ARCH)
            self.assertFalse(ok, kw)

    def test_label(self):
        from kernels.conv.kernel_identity import KernelIdentity

        base = KernelIdentity(arch=_ARCH, direction="dgrad", algorithm="implicit_gemm",
                              dtype_a="bf16", dtype_b="bf16", dtype_d="bf16", pp_boxes=7, split_k=8,
                              tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4, warp_tile_m=16,
                              warp_tile_n=16, warp_tile_k=16, pipeline="mem", epilogue="cshuffle",
                              wave_size=64, vector_size_a=8, vector_size_b=8, vector_size_c=8)
        lab0 = base.label()
        from dataclasses import replace

        lab = replace(base, pp_box_det=2).label()
        self.assertNotIn("pbd", lab0)
        self.assertIn("pbd2", lab)


class TestPack(unittest.TestCase):
    def _check(self, p, nclass, tile_m=128, tile_n=256, order=0, slot_bytes=4):
        sgs = pp_box_sub_gemms(p, tile_m, tile_n, nclass, tile_k=64, max_split=8)
        buf, lay = pp_box_det_pack(sgs, p, tile_m=tile_m, tile_n=tile_n, pp_box_order=order,
                                   slot_bytes=slot_bytes)
        hdr = len(sgs) * (_RECORD_FIELDS + _MAGIC_FIELDS)
        self.assertEqual(buf[hdr], lay["ws_word"])
        self.assertEqual(buf[hdr + 1], lay["cnt_word"])
        self.assertEqual(lay["ws_word"] % 64, 0)
        self.assertGreaterEqual(lay["ws_word"], len(buf))
        nn = math.ceil(p.C / tile_n)
        # Every split tile's slots: S distinct blocks of its record, all slots disjoint.
        seen_slots, seen_pix = set(), set()
        n_tiles = 0
        for sg in sgs:
            S = max(1, sg.b_x_stride)
            if S > 1:
                n_tiles += math.ceil(sg.gemm_m / tile_m) * nn
        self.assertEqual(lay["n_tiles"], n_tiles)
        for t in range(lay["n_tiles"]):
            e = buf[lay["tab_word"] + PBD_ENTRY_WORDS * t: lay["tab_word"] + PBD_ENTRY_WORDS * (t + 1)]
            slot0, S, step, rows_word, n_off, m_valid = e[:6]
            for s in range(S):
                slot = slot0 + s * step
                self.assertNotIn(slot, seen_slots)
                seen_slots.add(slot)
                self.assertLess(slot, lay["blocks"])
            for r in range(m_valid):
                seen_pix.add((buf[rows_word + r], n_off))
        # the split tiles' rows are distinct dX pixels per column block
        self.assertEqual(len(seen_pix), sum(buf[lay["tab_word"] + PBD_ENTRY_WORDS * t + 5]
                                            for t in range(lay["n_tiles"])))
        npix = p.N * p.Di * p.Hi * p.Wi
        self.assertTrue(all(0 <= px < npix for px, _ in seen_pix))

    def test_pixel_lists(self):
        for order in (0, 1):
            self._check(_p3(), 7, order=order)
            self._check(_p3(D=34, H=18, W=18, C=512), 77, tile_m=256, tile_n=256, order=order)

    def test_boxes(self):
        self._check(_p3(), 5)
        self._check(_p3(), 113, slot_bytes=2)


class TestBuild(unittest.TestCase):
    def test_stage1_ir(self):
        for v in (1, 2, 3):
            k = build_implicit_gemm_conv_dgrad(_spec(split_k=8, pp_boxes=7, pp_box_det=v), arch=_ARCH)
            self.assertIsNotNone(k)

    def test_stage2_ir(self):
        for kw in (dict(), dict(msplit=4), dict(msplit=2, wave_wg=True), dict(slot_bytes=2, unroll=2)):
            s = DgradBoxDetReduceSpec(tile_m=128, tile_n=256, warp_m=2, warp_n=4, **kw)
            self.assertIsNotNone(build_dgrad_box_det_reduce(s, arch=_ARCH))


if __name__ == "__main__":
    unittest.main()
