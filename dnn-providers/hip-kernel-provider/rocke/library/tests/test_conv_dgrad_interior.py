# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of interior-only dX (KernelIdentity.dx_interior): the view-problem
algebra against a numpy dgrad, the pixel map, labels / hashes, support rules,
and the zero-fill kernel builds."""

import dataclasses
import unittest

import numpy as np

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_dgrad_interior import (
    DgradHaloZeroSpec,
    build_dgrad_halo_zero,
    halo_zero_launch,
    halo_zero_units,
    interior_mask_np,
    interior_view_dx,
    interior_view_problem,
    view_pixel_map,
    written_mask_np,
)


def _dgrad_np(dy, w, p):
    """dX of a stride-1 3-D conv, NDHWC / NDHWK / KZYXC, pads (pD, pH, pW)."""
    N, Do, Ho, Wo, K = dy.shape
    dx = np.zeros((N, p.Di, p.Hi, p.Wi, p.C))
    for z in range(p.Z):
        for y in range(p.Y):
            for x in range(p.X):
                for do in range(Do):
                    di = do + z - p.pD
                    if not 0 <= di < p.Di:
                        continue
                    for ho in range(Ho):
                        hi = ho + y - p.pH
                        if not 0 <= hi < p.Hi:
                            continue
                        for wo in range(Wo):
                            wi = wo + x - p.pW
                            if 0 <= wi < p.Wi:
                                dx[:, di, hi, wi, :] += dy[:, do, ho, wo, :] @ w[:, z, y, x, :]
    return dx


def _halo(D=6, H=5, W=7, C=2, K=3, N=1):
    return ConvProblem(N=N, Hi=H, Wi=W, C=C, K=K, Y=3, X=3, Di=D, Z=3, sD=1, pD=0, dD=1)


class ViewProblem(unittest.TestCase):
    def test_interior_equals_view(self):
        p = _halo()
        rng = np.random.default_rng(0)
        dy = rng.standard_normal((p.N, p.Do, p.Ho, p.Wo, p.K))
        w = rng.standard_normal((p.K, 3, 3, 3, p.C))
        full = _dgrad_np(dy, w, p)
        for keep_w in (False, True):
            v = interior_view_problem(p, keep_w)
            self.assertEqual((v.Do, v.Ho, v.Wo), (p.Do, p.Ho, p.Wo))
            sub = _dgrad_np(dy, w, v)
            off, st = interior_view_dx(p, keep_w)
            flat = full.reshape(-1)
            f = view_pixel_map(p, keep_w)
            for pix in range(v.N * v.Di * v.Hi * v.Wi):
                n, r = divmod(pix, v.Di * v.Hi * v.Wi)
                d, r = divmod(r, v.Hi * v.Wi)
                h, x = divmod(r, v.Wi)
                e = off + n * st["p_dX_stride_n"] + d * st["p_dX_stride_di"] + h * st["p_dX_stride_hi"] + x * st["p_dX_stride_wi"]
                self.assertEqual(e, f(pix) * p.C)
                np.testing.assert_allclose(flat[e:e + p.C], sub[n, d, h, x], rtol=1e-12, atol=1e-12)

    def test_masks(self):
        p = _halo()
        m = interior_mask_np(p)
        self.assertEqual(int(m.sum()), p.Do * p.Ho * p.Wo)
        self.assertEqual(int(written_mask_np(p, True).sum()), p.Do * p.Ho * p.Wi)


class Identity(unittest.TestCase):
    LAB = (
        "dgrad-implicit_gemm-bf16-3d-t128x256x64-w2x4-a16x16x16-v8x8x8-mem-cshuffle-grp-s1x1-d1x1-sk8-xcd6-lxor-"
        "tapk-tpf1-bkp4-bpp4-pps2-pbx7-pbo1-pbd1-msg1024-wave64-gfx942-llvm22"
    )

    def test_label_hash(self):
        from kernels.conv.kernel_label import decode

        base = decode(self.LAB)
        self.assertEqual(base.dx_interior, 0)
        for v in (1, 2):
            t = dataclasses.replace(base, dx_interior=v)
            self.assertIn(f"-dxi{v}-", t.label())
            self.assertEqual(decode(t.label()), t)
            self.assertNotEqual(t.stable_hash(), base.stable_hash())

    def test_support(self):
        from kernels.conv.kernel_identity import identity_supports_problem
        from kernels.conv.kernel_label import decode

        base = decode(self.LAB)
        halo = ConvProblem(N=1, Hi=10, Wi=10, C=1024, K=1024, Y=3, X=3, Di=18, Z=3, sD=1, pD=0, dD=1)
        same = dataclasses.replace(halo, pH=1, pW=1, pD=1)
        t = dataclasses.replace(base, dx_interior=1)
        self.assertTrue(identity_supports_problem(t, halo)[0])
        self.assertFalse(identity_supports_problem(t, same)[0])  # not halo form
        self.assertFalse(identity_supports_problem(dataclasses.replace(t, dx_interior=3), halo)[0])
        self.assertFalse(identity_supports_problem(dataclasses.replace(t, pp_box_det=0), halo)[0])


class ZeroFill(unittest.TestCase):
    def test_builds(self):
        k = build_dgrad_halo_zero(DgradHaloZeroSpec())
        self.assertTrue(k is not None)
        v, grid, block = halo_zero_launch(_halo(C=8), 2, 0, 1 << 20, DgradHaloZeroSpec())
        self.assertEqual(block, (256, 1, 1))
        self.assertEqual(v["u1"] + v["u2"] + v["u3"], int((~interior_mask_np(_halo(C=8))).sum()))

    def test_sections_cover_the_shell(self):
        """The kernel's per-thread decode (emulated) hits every shell unit once."""
        for p in (_halo(N=2, C=16), _halo(D=18, H=10, W=10, C=64)):
            u = halo_zero_units(p, 2)
            pu, plane = p.C * 2 // 16, u["Hi"] * u["ru"]
            got = []
            for t in range(u["u1"]):
                n, r = divmod(t, u["L1"])
                side, o = divmod(r, u["hd"] * plane)
                got.append(n * u["Di"] * plane + side * (u["Di"] - u["hd"]) * plane + o)
            for t in range(u["u2"]):
                g, r = divmod(t, u["L2"])
                side, o = divmod(r, u["hh"] * u["ru"])
                n, d = divmod(g, u["Do"])
                got.append((n * u["Di"] + d + u["hd"]) * plane + side * (u["Hi"] - u["hh"]) * u["ru"] + o)
            for t in range(u["u3"]):
                g, r = divmod(t, u["L3"])
                side, o = divmod(r, u["e"])
                nd, h = divmod(g, u["Ho"])
                n, d = divmod(nd, u["Do"])
                row = (n * u["Di"] + d + u["hd"]) * u["Hi"] + h + u["hh"]
                got.append(row * u["ru"] + side * (u["ru"] - u["e"]) + o)
            shell = np.flatnonzero(~interior_mask_np(p).reshape(-1))
            want = sorted(int(v) * pu + j for v in shell for j in range(pu))
            self.assertEqual(sorted(got), want)


class PpBoxInterior(unittest.TestCase):
    def test_records_cover_interior(self):
        from kernels.common.conv_dgrad_box_det import _record_pixels
        from kernels.common.conv_implicit_gemm_dgrad import pp_box_sub_gemms

        p = ConvProblem(N=1, Hi=10, Wi=12, C=64, K=64, Y=3, X=3, Di=9, Z=3, sD=1, pD=0, dD=1)
        want = set(np.flatnonzero(interior_mask_np(p).reshape(-1)).tolist())
        for nclass in (3, 5, 7, 77, 555):
            for split in (1, 4):
                sgs = pp_box_sub_gemms(p, 64, 64, nclass, tile_k=32, max_split=split, cus=64, interior=True)
                got = [q for sg in sgs for q in _record_pixels(sg, p)]
                self.assertEqual(len(got), len(want), nclass)
                self.assertEqual(set(got), want, nclass)
                full = pp_box_sub_gemms(p, 64, 64, nclass, tile_k=32, max_split=split, cus=64)
                self.assertEqual(sum(sg.gemm_m for sg in full), p.Di * p.Hi * p.Wi)


class XhWmask(unittest.TestCase):
    XH = (
        "dgrad-implicit_gemm-bf16-3d-t256x64x32-w4x1-a16x16x16-v8x8x8-mem-cshuffle-grp-f0x3-s1x1-d1x1-wpe2-xcd6-lxor-"
        "tapk-tpf3-xh-bkp2-ilv4-xhx-mch2-lda1-lsb-sp1-bw-bpad4-zt1-ak4-xo7-xdf4-xfa1-xep1-elb-nag-mb11-xsm-xyr130-"
        "xel-xtr-msg1024-wave64-gfx942-llvm22"
    )

    def test_support(self):
        from kernels.conv.kernel_identity import identity_supports_problem
        from kernels.conv.kernel_label import decode

        base = decode(self.XH)
        xwm = dataclasses.replace(base, xh_dx_wmask=True)
        self.assertIn("-xwm-", xwm.label())
        self.assertEqual(decode(xwm.label()), xwm)
        halo = ConvProblem(N=1, Hi=130, Wi=130, C=64, K=64, Y=3, X=3, Di=258, Z=3, sD=1, pD=0, dD=1)
        pad1 = dataclasses.replace(halo, Di=256, Hi=128, Wi=128, pD=1, pH=1, pW=1)
        self.assertTrue(identity_supports_problem(base, halo)[0])
        self.assertFalse(identity_supports_problem(base, pad1)[0])  # pW == 1
        self.assertFalse(identity_supports_problem(xwm, halo)[0])  # would skip the w columns
        self.assertTrue(identity_supports_problem(xwm, pad1)[0])  # W-extended view (Wi 130: xyr130)
        for v in (1, 2):
            self.assertTrue(identity_supports_problem(dataclasses.replace(xwm, dx_interior=v), halo)[0])


if __name__ == "__main__":
    unittest.main()
