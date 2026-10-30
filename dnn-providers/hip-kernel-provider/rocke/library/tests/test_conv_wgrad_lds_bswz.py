# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for wgrad ``lds_bswz`` (pad-free XOR-swizzled B tile).

No GPU: spec gates, the name tag, the LDS footprint, lowering, and the bank
behaviour of the swizzle map (``loads.xor8_params``) on a 32-bank model.
"""

from __future__ import annotations

import unittest
from dataclasses import replace


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=ConvProblem(N=1, Hi=18, Wi=18, C=64, K=64, Y=3, X=3, pH=0, pW=0),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=64,
        tile_n=256,
        tile_k=32,
        warp_m=1,
        warp_n=4,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=8,
        wave_size=64,
        pipeline="mem",
        epilogue="default",
        lds_k_pad=4,
        prefetch_regs=1,
        lds_kpack=2,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=1,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _addr(r, c, tile_cols):
    from rocke.helpers.loads import xor8_params

    sh, mask, ush = xor8_params(tile_cols)
    return r * tile_cols * 2 + (c ^ (((r >> sh) & mask) << ush)) * 2


def _cycles(addrs, width):
    grp = {4: 32, 8: 16}[width]
    tot = 0
    for g in range(0, 64, grp):
        cnt = {}
        for a in addrs[g : g + grp]:
            for d in range(width // 4):
                bank = (a // 4 + d) % 32
                cnt[bank] = cnt.get(bank, 0) + 1
        tot += max(cnt.values())
    return tot


class TestWgradLdsBswz(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertEqual(spec.lds_bswz, 0)
        self.assertNotIn("bswz", spec.kernel_name())
        self.assertIn("bswz", replace(spec, lds_bswz=1).kernel_name())

    def test_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        ok, why = is_valid_wgrad_spec(_spec(lds_bswz=1), arch="gfx942")
        self.assertTrue(ok, why)
        for bad in (
            dict(prefetch_regs=0, lds_kpack=1),
            dict(prefetch_regs=2),
            dict(lds_kpack=1),
            dict(lds_bswz=2),
        ):
            ok, _ = is_valid_wgrad_spec(_spec(**{**dict(lds_bswz=1), **bad}), arch="gfx942")
            self.assertFalse(ok, bad)

    def test_lowers_and_drops_the_b_pad(self):
        from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        ll = lower_kernel_to_llvm(
            build_implicit_gemm_conv_wgrad(_spec(lds_bswz=1), arch="gfx942"), arch="gfx942"
        )
        ll0 = lower_kernel_to_llvm(
            build_implicit_gemm_conv_wgrad(_spec(), arch="gfx942"), arch="gfx942"
        )
        # A 64 x (32 + 4) and B 256 x 32 (no pad) vs B 256 x (32 + 4), bf16.
        self.assertIn("[20992 x i8]", ll)
        self.assertIn("[23040 x i8]", ll0)
        self.assertIn(" xor ", ll)

    def test_map_is_a_bijection_and_conflict_free(self):
        for tk in (16, 32, 64):
            seen = {_addr(r, c, tk) for r in range(256) for c in range(tk)}
            self.assertEqual(len(seen), 256 * tk)
            # MFMA fragment reads: ds_read_b64 of 16/32 consecutive rows at one
            # 4-element unit -> every 16-lane group touches 32 distinct banks.
            for rm, kdiv, wk in ((32, 32, 8), (16, 16, 16)):
                for kk in range(tk // wk):
                    for base in (0, rm, 64, 128 + rm):
                        ad = [
                            _addr(base + l % rm, (l // kdiv) * 4 + kk * wk, tk)
                            for l in range(64)
                        ]
                        self.assertEqual(_cycles(ad, 8), 4, (tk, rm, kk, base))


class TestWgradKpLine(unittest.TestCase):
    """``kp_line``: line-major lane decode of the k-packed global reads."""

    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertEqual(spec.kp_line, 0)
        self.assertNotIn("kpl", spec.kernel_name())
        self.assertIn("kpl8", replace(spec, kp_line=8).kernel_name())

    def test_gates(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        ok, why = is_valid_wgrad_spec(_spec(kp_line=8), arch="gfx942")
        self.assertTrue(ok, why)
        for bad in (dict(kp_line=4), dict(lds_kpack=1), dict(prefetch_regs=0, lds_kpack=1)):
            ok, _ = is_valid_wgrad_spec(_spec(**{**dict(kp_line=8), **bad}), arch="gfx942")
            self.assertFalse(ok, bad)

    def test_decode_is_a_bijection_with_whole_lines(self):
        # Mirror of CoalescedTileLoader._chunk_row_col for kp_line.
        for rows, cols, kp, threads in ((256, 32, 2, 256), (64, 32, 2, 128), (256, 16, 2, 256)):
            vec, line = 8, 8
            cgl = 64 // line
            nb = (cols // kp) // cgl
            units = (rows // vec) * (cols // kp)
            seen, col_of_thread = set(), {}
            for u in range(units):
                lane, grp = u % 64, u // 64
                colg, rowc = lane % cgl, lane // cgl
                if nb > 1:
                    colg += (grp % nb) * cgl
                    grp //= nb
                rowc += grp * line
                seen.add((rowc, colg))
                col_of_thread.setdefault(u % threads, set()).add(colg)
            self.assertEqual(len(seen), units)
            self.assertTrue(all(len(c) == 1 for c in col_of_thread.values()))

    def test_lowers_with_the_same_memory_ops(self):
        import re

        from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        def count(spec):
            ll = lower_kernel_to_llvm(
                build_implicit_gemm_conv_wgrad(spec, arch="gfx942"), arch="gfx942"
            )
            return len(re.findall(r"buffer\.load", ll)), len(re.findall(r"addrspace\(3\)", ll))

        base = _spec(warp_n=2, tile_n=128)
        self.assertEqual(count(base)[0], count(replace(base, kp_line=8))[0])


if __name__ == "__main__":
    unittest.main()
