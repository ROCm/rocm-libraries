# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Forward split-K store / order knobs of the ping-pong family (round 4):
``split_atomic_pairs`` (sapl), ``split_atomic_agent`` (saa),
``split_atomic_fast`` (sapf), ``epi_store_fast`` (esf), ``split_xcd_remap``
(skx) and ``xh_subtile`` (xsub, incl. its split-K slices).

GPU: each against the fp32 reference (garbage-filled D, split-K at several
degrees). ``split_streamk`` / ``split_flag_store`` take launch-side state the
shared test launcher does not build; they are verified out of the cache
(``att_launch --check`` / ``benchmark_conv_compare --verify``). CPU:
rejections, name tags, label round trip, default hash, AOT axes.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
from unittest import mock

import pytest

from tests.test_conv_fwd_3d_kloop_knobs import _SKIP, _TOL, _problem, _run

_PP5 = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, lds_kpack_read=True,
            rpf_pingpong=5, vmem_interleave=4)
_T256 = (256, 128, 64, 4, 2, 16)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwdSplitStore(unittest.TestCase):
    def _check(self, problem, **kw):
        err = _run(problem, **kw)
        self.assertLess(err, _TOL, f"{kw}: rel err {err:.3e}")

    def test_split_atomic_pairs(self):
        p = _problem(C=128, K=128, pH=0, pW=0, pD=0)
        for knobs in (
            dict(split_atomic_pairs=True),
            dict(split_atomic_pairs=True, split_atomic_agent=True),
            dict(split_atomic_agent=True),
            dict(split_atomic_pairs=True, split_atomic_fast=True),
            dict(split_atomic_pairs=True, split_xcd_remap=6),
        ):
            for degree in (2, 3):
                with self.subTest(degree=degree, **knobs):
                    self._check(p, tile=_T256, degree=degree, **_PP5, uk_nopad=True,
                                split_k=2, **knobs)

    def test_split_atomic_pairs_chunked(self):
        p = _problem(C=128, K=128, pH=0, pW=0, pD=0)
        for tile, wn in (((512, 128, 32, 4, 2, 16), 2), ((256, 256, 32, 2, 4, 16), 4)):
            with self.subTest(tile=tile):
                self._check(p, tile=tile, degree=3, **_PP5, uk_nopad=True, split_k=2,
                            cshuffle_m_chunks=2, split_atomic_pairs=True)

    def test_epi_store_fast(self):
        # Partial tiles (the slow path) and whole tiles (the fast path).
        for p in (_problem(C=128, K=128, pH=0, pW=0, pD=0),
                  _problem(C=128, K=128, Hi=18, Wi=18, Di=6, pH=0, pW=0, pD=0)):
            with self.subTest(M=p.N * p.Do * p.Ho * p.Wo):
                self._check(p, tile=_T256, **_PP5, uk_nopad=True, epi_store_fast=True)

    def test_chunk_group_split(self):
        # uk_chunk_group on the split binaries: the grouped slice start
        # decode (C=128, tile_k 32: 4 chunks, groups of 2 / 4), and a cpg that
        # is not a whole number of groups (C=96: the plain fallback loop).
        for C in (128, 96):
            p = _problem(C=C, K=128, pH=0, pW=0, pD=0)
            for g in (2, 4):
                for degree in (2, 3, 5):
                    with self.subTest(C=C, g=g, degree=degree):
                        self._check(p, tile=(256, 256, 32, 4, 2, 16), degree=degree, split_k=2,
                                    **_PP5, uk_nopad=True, uk_tap_fast=True, uk_chunk_group=g,
                                    cshuffle_m_chunks=2, lds_xor_rowkey=True,
                                    split_atomic_pairs=True, pp_split_fetch=3,
                                    fb_skip_empty=True)

    def test_xh_subtile(self):
        # Wo=16, Wi=18: one 256-row tile spans 288 input pixels (uk_xhalo=32);
        # unsplit and split slices of the (chunk, z, y) steps.
        p = _problem(C=128, K=128, Hi=18, Wi=18, Di=4, pH=0, pW=0, pD=0)
        xs = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, lds_kpack_read=True,
                  rpf_pingpong=5, vmem_interleave=4, uk_xhalo=32, uk_tap_fast=True,
                  xh_subtile=True)
        self._check(p, tile=_T256, **xs)
        for degree in (2, 3, 5):
            with self.subTest(degree=degree):
                self._check(p, tile=_T256, degree=degree, split_k=2,
                            split_atomic_pairs=True, **xs)

    def test_pp_split_fetch_and_deep(self):
        # pp_split_fetch 1-5 / pp_fetch_deep on the pp5 loop, plain and with
        # pp_b_double, unsplit and split (odd tile counts run the tail loop).
        p = _problem(C=128, K=128, pH=0, pW=0, pD=0)
        for extra in (dict(), dict(pp_b_double=True)):
            for knobs in [dict(pp_split_fetch=n) for n in (1, 2, 3, 4, 5, 6, 7, 8, 10)] + [
                    dict(pp_fetch_deep=True), dict(pp_fetch_deep=True, pp_split_fetch=3),
                    dict(pp_split_fetch=3, fb_skip_empty=True, pp_end_guard=True)]:
                with self.subTest(**extra, **knobs):
                    self._check(p, tile=_T256, **_PP5, uk_nopad=True, uk_tap_fast=True,
                                **extra, **knobs)
                    self._check(p, tile=_T256, degree=3, split_k=2, split_atomic_pairs=True,
                                **_PP5, uk_nopad=True, uk_tap_fast=True, **extra, **knobs)


def _run_flags(problem, *, degree, tile, launches=3, full_tiles=None, **knobs):
    """split_zero_first / split_flag_store: D left as garbage (the kernel owns
    its zeroing / ordering), the sk_flags buffer zeroed once; launched
    ``launches`` times (the flags must reset themselves). Max rel error."""
    import torch

    from builders.common.conv_reference import conv_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from tests.test_conv_fwd_3d_kloop_knobs import _u8, get_device_arch

    arch = get_device_arch().split(":")[0]
    tm, tn, tk, wm, wn, wt = tile
    spec = ImplicitGemmConvSpec(
        problem=problem, name="test_fwd_split_flags",
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=tm, tile_n=tn, tile_k=tk, warp_m=wm, warp_n=wn, warp_tile_m=wt,
        warp_tile_n=wt, warp_tile_k=16, pipeline="mem", epilogue="cshuffle",
        groups=problem.groups, **knobs,
    )
    art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name,
        signature=conv_args_signature("bf16", is_3d=True, split_k=True,
                                      split_tail=spec.split_tail, split_flags=True),
    )
    p = problem
    torch.manual_seed(0)
    A = torch.empty(p.N, p.Di, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(torch.bfloat16)
    B = torch.empty(p.K, p.Z, p.Y, p.X, p.cpg).uniform_(-1, 1).to(torch.bfloat16)
    D = torch.empty(p.N, p.Do, p.Ho, p.Wo, p.K, dtype=torch.bfloat16)
    ref = conv_reference(A, B, p, out_dtype=torch.bfloat16)
    args = ConvArgs.from_problem(p, tile_m=tm, tile_n=tn, tile_k=tk)
    nflag = 8 * args.num_pid_m * args.num_pid_n
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D)]
    flags = rt.alloc(nflag)
    errs = []
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        rt.memset(flags, 0, nflag)
        values = args.to_launch_values(
            *ptrs, A.nbytes, B.nbytes, D.nbytes, split_k=degree, sk_flags=flags,
            full_tiles=full_tiles if spec.split_tail else None,
        )
        gx, gy, gz = args.grid()
        for _ in range(launches):
            rt.memset(ptrs[2], 0x7F, D.nbytes)  # garbage
            launcher(values, config=LaunchConfig(grid=(gx, gy, gz * degree),
                                                 block=(spec.block_size, 1, 1), fence=True))
            out = torch.empty_like(D)
            rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
            o = out.float().to(ref.device)
            errs.append(float((o - ref).abs().max() / ref.abs().max().clamp(min=1.0)))
    finally:
        for ptr in ptrs + [flags]:
            rt.free(ptr)
    return max(errs)


@pytest.mark.gpu
@unittest.skipIf(_SKIP, _SKIP)
class TestFwdSplitFlags(unittest.TestCase):
    def test_split_zero_first_and_flag_store(self):
        p = _problem(C=128, K=128, Hi=18, Wi=18, Di=6, pH=0, pW=0, pD=0)
        for knobs in (dict(split_zero_first=True), dict(split_flag_store=True)):
            for degree in (2, 3):
                with self.subTest(degree=degree, **knobs):
                    err = _run_flags(p, degree=degree, tile=_T256, **_PP5, uk_nopad=True,
                                     split_k=2, split_atomic_pairs=True, **knobs)
                    self.assertLess(err, _TOL)
        # split_tail + split_zero_first: the full tiles store, the tail zeroes itself
        with self.subTest(split_tail=True):
            err = _run_flags(p, degree=2, tile=_T256, full_tiles=2, **_PP5, uk_nopad=True,
                             split_k=2, split_tail=True, split_atomic_pairs=True,
                             split_zero_first=True)
            self.assertLess(err, _TOL)


class TestFwdSplitStoreSpec(unittest.TestCase):
    def _spec(self, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

        base = dict(
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            problem=_problem(C=128, K=128, pH=0, pW=0, pD=0), tile_m=256, tile_n=128,
            tile_k=64, warp_m=4, warp_n=2, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            pipeline="mem", epilogue="cshuffle", **_PP5,
        )
        base.update(kw)
        return ImplicitGemmConvSpec(**base)

    def test_rejections(self):
        from kernels.common.conv_implicit_gemm import fwd_split_k_conflict

        for kw in (
            dict(split_k=2, split_atomic_pairs=True),
            dict(split_k=2, split_atomic_pairs=True, split_atomic_fast=True),
            dict(split_k=2, split_streamk=True),
            dict(split_k=2, split_flag_store=True),
            dict(split_k=2, split_xcd_remap=6),
            dict(split_k=2, split_hybrid=24),
        ):
            with self.subTest(**kw):
                self.assertEqual(fwd_split_k_conflict(self._spec(**kw)), "")
        for kw in (
            dict(split_atomic_pairs=True),          # needs split_k
            dict(split_streamk=True),
            dict(split_flag_store=True),
            dict(split_xcd_remap=6),
            dict(split_k=2, split_atomic_fast=True),  # needs split_atomic_pairs
            dict(split_k=2, split_streamk=True, split_tail=True),
            dict(split_k=2, split_streamk=True, xcd_remap=6),
            dict(split_k=2, split_xcd_remap=6, xcd_remap=6),
            dict(split_k=2, split_flag_store=True, split_streamk=True),
            dict(split_hybrid=24),                     # needs split_k
            dict(split_k=2, split_hybrid=24, split_streamk=True),
            dict(split_k=2, split_hybrid=24, xcd_remap=6),
            dict(split_k=2, split_hybrid=24, split_tail=True),
        ):
            with self.subTest(**kw):
                self.assertNotEqual(fwd_split_k_conflict(self._spec(**kw)), "")

    def test_names(self):
        a = self._spec(split_k=2)
        b = self._spec(split_k=2, split_atomic_pairs=True, split_atomic_agent=True,
                       split_atomic_fast=True, epi_store_fast=True)
        for tag in ("_sapl", "_saa", "_sapf", "_esf", "_fsk", "_sfs", "_skx", "_xsub"):
            self.assertNotIn(tag, a.kernel_name())
        for tag in ("_sapl", "_saa", "_sapf", "_esf"):
            self.assertIn(tag, b.kernel_name())

    def test_label_roundtrip_and_hash(self):
        from kernels.conv import kernel_label as kl

        lab = ("fwd-implicit_gemm-bf16-3d-t256x128x64-w4x2-a16x16x16-v8x8x8-mem-cshuffle-grp-"
               "sk2-xcd6-ukd-rpf1-lxor-tapf-nopad-vmi4-kprd-pp5-ppbd-wave64-gfx942-llvm22")
        i = kl.decode(lab)
        on = dataclasses.replace(i, split_atomic_pairs=True, split_atomic_fast=True,
                                 split_atomic_agent=True, epi_store_fast=True)
        self.assertEqual(kl.decode(kl.encode(on)), on)
        self.assertIn("-saa-sapl-sapf-esf-", kl.encode(on))
        self.assertNotEqual(i.stable_hash(), on.stable_hash())
        off = dataclasses.replace(on, split_atomic_pairs=False, split_atomic_fast=False,
                                  split_atomic_agent=False, epi_store_fast=False)
        self.assertEqual(i.stable_hash(), off.stable_hash())
        for f, v in (("pp_split_fetch", 3), ("pp_fetch_deep", True), ("fb_skip_empty", True),
                     ("pp_end_guard", True), ("split_hybrid", 24), ("epi_lds_barriers", True),
                     ("sched_strategy", 1), ("split_tail_compact", True)):
            x = dataclasses.replace(i, **{f: v})
            self.assertEqual(kl.decode(kl.encode(x)), x)
            self.assertNotEqual(i.stable_hash(), x.stable_hash())
        for f in ("split_streamk", "split_flag_store", "xh_subtile"):
            x = dataclasses.replace(i, **{f: True})
            self.assertEqual(kl.decode(kl.encode(x)), x)

    def test_aot_axes(self):
        from kernels.conv import aot_jobs

        kl = dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True, rpf_pingpong=5)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SAA": ""}):
            self.assertEqual(aot_jobs._fwd_split_atomic_variants(kl, (2, False, False)), ({},))
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SAA": "p,pf"}):
            self.assertEqual(
                aot_jobs._fwd_split_atomic_variants(kl, (2, False, False)),
                ({}, dict(split_atomic_pairs=True),
                 dict(split_atomic_pairs=True, split_atomic_fast=True)),
            )
            # unsplit binaries and non-pp loops get none
            self.assertEqual(aot_jobs._fwd_split_atomic_variants(kl, (1, False, False)), ({},))
            self.assertEqual(
                aot_jobs._fwd_split_atomic_variants(dict(kl, rpf_pingpong=0), (2, False, False)),
                ({},),
            )
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_SAA": "x"}):
            with self.assertRaises(ValueError):
                aot_jobs._fwd_split_atomic_variants(kl, (2, False, False))
        # ROCKE_AOT_FWD_PSF: pp_split_fetch variants of the pp5 loops only
        pp5 = dict(kl, vmem_interleave=4)
        with mock.patch.dict(os.environ, {"ROCKE_AOT_FWD_PP": "p5v6", "ROCKE_AOT_FWD_PSF": "3"}):
            loops = aot_jobs._fwd_pingpong_k_loops(
                [dict(uniform_k_decode=True, reg_prefetch=1, lds_xor=True)], "mem")
            self.assertTrue(any(d.get("pp_split_fetch") == 3 for d in loops))
            self.assertTrue(all(d.get("rpf_pingpong") == 5 for d in loops
                                if d.get("pp_split_fetch")))
        del pp5
        # m-chunked pp loops only where the bf16 C tile does not fit the LDS
        self.assertFalse(aot_jobs._fwd_r2_in_scope(dict(kl, cshuffle_m_chunks=2),
                                                   256, 128, 64, 4, 2, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
