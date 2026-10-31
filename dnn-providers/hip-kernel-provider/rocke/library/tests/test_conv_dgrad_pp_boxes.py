# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the dgrad ping-pong small-grid knobs (DgradConvSpec.pp_streamk,
pp_sk_mode, pp_zskip, pp_boxes, pp_box_cus): validation, names, identity labels,
the host box decomposition, IR build."""

from __future__ import annotations

import math
import unittest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_implicit_gemm import ConvDataSpec
from kernels.common.conv_implicit_gemm_dgrad import (
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


def _valid_taps(c, n_in, n_out, pad, filt):
    return {f for f in range(filt) if 0 <= c + pad - f <= n_out - 1}


class TestBoxDecomposition(unittest.TestCase):
    def _check(self, p, nclass, tile_m=128, tile_n=256, split=1):
        sgs = pp_box_sub_gemms(p, tile_m, tile_n, nclass, tile_k=64, max_split=split)
        seen = set()
        for sg in sgs:
            ds, db, zb, zs = sg.a_embed_h_coeff, sg.a_embed_w_coeff, sg.b_y_stride, sg.gemm_k_padded
            hb, hs = sg.h_tilde_slice_begin, sg.h_tilde_slice
            wb, ws = sg.w_tilde_slice_begin, sg.w_tilde_slice
            yb, ys, xb, xs = sg.b_y_offset, sg.y_dot_slice, sg.b_x_offset, sg.x_dot_slice
            self.assertEqual(sg.gemm_m, p.N * ds * hs * ws)
            self.assertEqual(
                sg.block_end - sg.block_start,
                math.ceil(sg.gemm_m / tile_m) * math.ceil(p.C / tile_n) * sg.b_x_stride,
            )
            for d in range(db, db + ds):
                for h in range(hb, hb + hs):
                    for w in range(wb, wb + ws):
                        self.assertNotIn((d, h, w), seen)
                        seen.add((d, h, w))
                        # every tap the pixel can use is inside the box's sub-filter
                        self.assertTrue(
                            _valid_taps(d, p.Di, p.Do, p.pD, p.Z) <= set(range(zb, zb + zs))
                        )
                        self.assertTrue(
                            _valid_taps(h, p.Hi, p.Ho, p.pH, p.Y) <= set(range(yb, yb + ys))
                        )
                        self.assertTrue(
                            _valid_taps(w, p.Wi, p.Wo, p.pW, p.X) <= set(range(xb, xb + xs))
                        )
        self.assertEqual(len(seen), p.Di * p.Hi * p.Wi)
        starts = [sg.block_start for sg in sgs]
        self.assertEqual(starts, sorted(starts))
        return sgs

    def test_classes(self):
        p = _p3()
        self.assertEqual(len(self._check(p, 3)), 9)
        self.assertEqual(len(self._check(p, 5)), 25)
        self.assertEqual(len(self._check(p, 33)), 27)
        self.assertEqual(len(self._check(p, 55)), 125)

    def test_split_budget(self):
        p = _p3(C=1024, K=1024)
        sgs = self._check(p, 3, split=8)
        self.assertLessEqual(sgs[-1].block_end, 228)
        self.assertGreater(max(sg.b_x_stride for sg in sgs), 1)
        sgs = pp_box_sub_gemms(p, 128, 256, 3, tile_k=64, max_split=8, cus=456)
        self.assertLessEqual(sgs[-1].block_end, 456)


class TestPixelLists(unittest.TestCase):
    """pp_boxes 7: every dX pixel in exactly one record's row list, and
    each record's offsets cover every tap its pixels can use."""

    def _check(self, p, tile_m=128, tile_n=256, split=1, cus=0, code=7):
        sgs = pp_box_sub_gemms(
            p, tile_m, tile_n, code, tile_k=64, max_split=split, cus=cus
        )
        hw = p.Hi * p.Wi
        seen = set()
        for sg in sgs:
            rows = sg.row_list
            self.assertEqual(sg.gemm_m, len(rows))
            self.assertEqual((sg.h_tilde_slice, sg.w_tilde_slice), (p.Hi, p.Wi))
            self.assertEqual(sg.a_embed_h_coeff, p.Di)
            self.assertEqual(
                sg.block_end - sg.block_start,
                math.ceil(len(rows) / tile_m) * math.ceil(p.C / tile_n) * sg.b_x_stride,
            )
            zb, zs = sg.b_y_stride, sg.gemm_k_padded
            yb, ys, xb, xs = sg.b_y_offset, sg.y_dot_slice, sg.b_x_offset, sg.x_dot_slice
            for pix in rows:
                self.assertNotIn(pix, seen)
                seen.add(pix)
                nd, r = divmod(pix, hw)
                h, w = divmod(r, p.Wi)
                d = nd % p.Di
                self.assertTrue(_valid_taps(d, p.Di, p.Do, p.pD, p.Z) <= set(range(zb, zb + zs)))
                self.assertTrue(_valid_taps(h, p.Hi, p.Ho, p.pH, p.Y) <= set(range(yb, yb + ys)))
                self.assertTrue(_valid_taps(w, p.Wi, p.Wo, p.pW, p.X) <= set(range(xb, xb + xs)))
        self.assertEqual(seen, set(range(p.N * p.Di * hw)))
        return sgs

    def test_cover_and_split(self):
        self._check(_p3())
        self._check(_p3(N=2, D=7, H=12, W=9))
        sgs = self._check(_p3(C=1024, K=1024), split=8)
        self.assertLessEqual(sgs[-1].block_end, 228)
        self.assertGreater(max(sg.b_x_stride for sg in sgs), 1)

    def test_cheap_tail(self):
        # pp_boxes 77: same cover; s10's 8-pixel partial tile runs 3 taps
        # (a depth-interior H/W corner column), not the packer's union box.
        for p in (_p3(), _p3(N=2, D=7, H=12, W=9), _p3(C=1024, K=1024)):
            self._check(p, code=77)
            self._check(p, split=8, code=77)
        sgs = self._check(_p3(C=1024, K=1024), split=8, code=77)
        part = [sg for sg in sgs if sg.gemm_m % 128]
        self.assertEqual([sg.gemm_m for sg in part], [8])
        sg = part[0]
        self.assertEqual(sg.gemm_k_padded * sg.y_dot_slice * sg.x_dot_slice, 3)
        work = lambda q: sum(  # noqa: E731
            math.ceil(sg.gemm_m / 128) * sg.gemm_k_padded * sg.y_dot_slice * sg.x_dot_slice
            for sg in q
        )
        old = pp_box_sub_gemms(_p3(C=1024, K=1024), 128, 256, 7, tile_k=64, max_split=8)
        self.assertLess(work(sgs), work(old))

    def test_buffer_layout(self):
        from kernels.common.conv_implicit_gemm_dgrad import (
            _MAGIC_FIELDS,
            _RECORD_FIELDS,
            pack_sub_gemm_buffer,
        )

        p = _p3()
        sgs = pp_box_sub_gemms(p, 128, 256, 7, tile_k=64)
        buf = pack_sub_gemm_buffer(sgs, 128, 256)
        n = len(sgs)
        tab = n * (_RECORD_FIELDS + _MAGIC_FIELDS)
        for i, sg in enumerate(sgs):
            base = tab + n + buf[tab + i]
            self.assertEqual(buf[base : base + sg.gemm_m], list(sg.row_list))


class TestSmallGridSpecs(unittest.TestCase):
    def test_defaults_off(self):
        s = _spec()
        for tag in ("psk", "pzs", "pbx"):
            self.assertNotIn(tag, s.kernel_name())

    def test_valid_named_and_built(self):
        for kw, tag in (
            (dict(pp_streamk=228), "psk228"),
            (dict(pp_streamk=228, pp_sk_mode=2), "pskm2"),
            (dict(pp_streamk=228, pp_sk_mode=1), "pskm1"),
            (dict(pp_zskip=True), "pzs"),
            (dict(pp_streamk=228, pp_zskip=True), "pzs"),
            (dict(pp_boxes=5), "pbx"),
            (dict(pp_boxes=3, split_k=8, pp_zskip=True), "pbx"),
            (dict(pp_boxes=5, tapk_pp_epi=2), "pbx"),
            (dict(pp_boxes=3, split_k=8, tile_m=256, tile_k=32, cshuffle_m_chunks=2), "pbx"),
            (dict(pp_boxes=113, split_k=8, pp_box_order=1), "pbo1"),
            (dict(pp_boxes=113, split_k=8, pp_box_order=1, tapk_pp_sched=7), "pps7"),
            (dict(pp_boxes=5, pp_box_order=2, xcd_remap=6, pp_zskip=True, tapk_pp_epi=2), "pbo2"),
            (dict(pp_boxes=5, split_k=8, pp_box_order=3, xcd_remap=6), "pbo3"),
            (dict(pp_boxes=7), "pbl"),
            (dict(pp_boxes=7, split_k=8, pp_box_order=1, tapk_pp_sched=7), "pbl"),
            (dict(pp_boxes=7, tapk_pp_epi=2, xcd_remap=6, pp_box_order=2), "pbl"),
            (dict(pp_boxes=77, split_k=8, pp_box_order=1, tapk_pp_sched=7), "pbl"),
            (dict(tapk_pp_sched=7), "pps7"),
        ):
            with self.subTest(**kw):
                s = _spec(**kw)
                ok, why = is_valid_dgrad_spec(s, _ARCH)
                self.assertTrue(ok, why)
                self.assertIn(tag, s.kernel_name())
                build_implicit_gemm_conv_dgrad(s, arch=_ARCH)

    def test_box_classes_share_a_binary_name(self):
        names = {_spec(pp_boxes=n, pp_box_cus=c).kernel_name() for n in (3, 5, 33) for c in (0, 456)}
        self.assertEqual(len(names), 1)

    def test_rejected(self):
        for kw in (
            dict(pp_sk_mode=1),  # needs pp_streamk
            dict(pp_streamk=228, split_k=2),
            dict(pp_streamk=228, tapk_pp_epi=2),
            dict(pp_boxes=4),
            dict(pp_boxes=3, tapk_pingpong=3, tapk_pp_sched=0),
            dict(pp_boxes=3, split_k=2, tapk_pp_epi=2),
            dict(pp_boxes=3, pp_streamk=228),
            dict(pp_boxes=3, pp_box_order=1),  # the order is of split-K blocks
            dict(pp_boxes=3, pp_box_order=2),  # the window order needs xcd_remap
            dict(pp_boxes=3, pp_box_order=4, xcd_remap=6),
            dict(pp_boxes=7, pp_zskip=True),  # the records carry exact depths
            dict(pp_boxes=77, pp_zskip=True),
            dict(pp_boxes=123),
            dict(tapk_pp_sched=7, tapk_pp_epi=2),
        ):
            with self.subTest(**kw):
                ok, _why = is_valid_dgrad_spec(_spec(**kw), _ARCH)
                self.assertFalse(ok)


class TestIdentity(unittest.TestCase):
    def _ident(self, **kw):
        from kernels.conv.kernel_identity import KernelIdentity

        base = dict(
            arch=_ARCH, direction="dgrad", algorithm="implicit_gemm",
            dtype_a="bf16", dtype_b="bf16", dtype_d="bf16",
            tile_m=128, tile_n=256, tile_k=64, warp_m=2, warp_n=4,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="mem", epilogue="cshuffle", wave_size=64,
            vector_size_a=8, vector_size_b=8, vector_size_c=8,
            lds_k_outer=False, grouped=True, is_3d=True,
            stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, max_sub_gemms=1024,
            tap_outer_k=True, tapk_prefetch=1, b_kpack=4, tapk_pingpong=4, tapk_pp_sched=2,
        )
        base.update(kw)
        return KernelIdentity(**base)

    def test_labels_round_trip(self):
        off = self._ident()
        for kw, tags in (
            (dict(pp_streamk=225, pp_sk_mode=2), ("psk225", "pskm2")),
            (dict(pp_zskip=True), ("pzs",)),
            (dict(split_k=8, pp_boxes=3, pp_box_cus=456), ("pbx3", "pbc456")),
        ):
            with self.subTest(**kw):
                on = self._ident(**kw)
                self.assertNotEqual(on.stable_hash(), off.stable_hash())
                for t in tags:
                    self.assertIn(t, on.label())
                self.assertEqual(type(on).from_label(on.label()), on)

    def test_box_records_and_support(self):
        from kernels.conv.kernel_identity import dgrad_sub_gemms, identity_supports_problem

        ident = self._ident(split_k=8, pp_boxes=3)
        p = _p3(C=1024, K=1024)
        self.assertEqual(len(dgrad_sub_gemms(ident, p)), 9)
        self.assertTrue(identity_supports_problem(ident, p)[0])
        self.assertFalse(identity_supports_problem(ident, ConvProblem(
            N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, groups=2
        ))[0])
        # padded (production-form) problems decompose too
        self.assertTrue(identity_supports_problem(ident, ConvProblem(
            N=1, Di=10, Hi=10, Wi=10, C=256, K=128, Z=3, Y=3, X=3, sD=1, pD=0, dD=1, pH=1, pW=1
        ))[0])


if __name__ == "__main__":
    unittest.main()
