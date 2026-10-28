# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM K-tile schedule families.

The schedule families are separate K-tile bodies:

* early release ``ilv_early_release`` (+ ``ilv_release_at`` /
  ``ilv_acquire_at``, ``ilv_read_span`` / ``ilv_dtl_span``, ``acc_out_vgpr``;
  mirrored in C++),
* early stage free ``ilv_early_free`` / ``ilv_ef_spread`` / ``ilv_split_wait``
  (Python only),
* slot schedule ``hb_sched`` (+ slot knobs; Python only; covered in depth by
  test_gemm_hb_sched.py) with ``lds_row_interleave`` / ``rilv_vec_epilogue`` /
  ``rve_store_nt`` (mirrored in C++),
* wave-specialized ``pingpong`` (+ ``pp_*``; Python only).

These tests pin the composition rules: every lever is default-off (the default
trait keeps the historical emission), at most one schedule family per kernel,
the layout lever ``lds_row_interleave`` composes with the early-free /
early-release schedules but not with the ping-pong body, kernel-name flags are
unique, and the C-engine
converters refuse every Python-only lever instead of silently building another
kernel.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import fields, replace

from rocke.core import backend as _be
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
)

_T22 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_T24 = TileSpec(256, 256, 64, 2, 4, 1, 16, 16, 32)
# the A16S2 / K2 body: 4 waves of 128x128, 16x16x32, AGPR accumulators, 2-stage ring
_A16 = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    dtl_hoist=True,
)
_ER3 = replace(_A16, ilv_early_release=3, ilv_release_at=32, ilv_acquire_at=40)
_EF = replace(_A16, interwave_setprio=True, ilv_early_free=20, ilv_ef_spread=True)
_HB = replace(
    _A16,
    ilv_merge=False,
    lds_swizzle=False,
    lds_row_interleave=True,
    hb_sched=True,
    hb_read_every=2,
    rilv_vec_epilogue=True,
)
_PP7 = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    acc_agpr=True,
    dtl_stages=2,
    interwave_setprio=True,
    lds_swizzle_r=-1,
    pingpong=7,
    pp_mix_reads=3,
)
_RILV = dict(lds_row_interleave=True, lds_swizzle=False)


def _spec(trait, tile=_T22, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=tile, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _ok(trait, tile=_T22, dtype="fp16"):
    return is_valid_spec(_spec(trait, tile, dtype), "gfx950")


class TestDefaults(unittest.TestCase):
    _NEW = {
        "ilv_early_release": 0,
        "ilv_release_at": 0,
        "ilv_acquire_at": 0,
        "acc_out_vgpr": False,
        "ilv_read_span": 0,
        "ilv_dtl_span": 0,
        "ilv_early_free": 0,
        "ilv_ef_spread": False,
        "ilv_split_wait": 0,
        "lds_row_interleave": False,
        "hb_sched": False,
        "hb_rel_slot": -1,
        "hb_acq_slot": -1,
        "hb_read_every": 1,
        "hb_dtl_every": 0,
        "hb_acq_b_slot": -1,
        "hb_split_rel": False,
        "hb_rel_gap": 4,
        "rilv_vec_epilogue": False,
        "rve_store_nt": False,
        "pingpong": 0,
        "pp_wait_late": False,
        "pp_mix_reads": 0,
        "pp_spread_lo": 0,
    }

    def test_every_merged_lever_defaults_off(self):
        tr = TraitSpec()
        names = {f.name for f in fields(TraitSpec)}
        for k, v in self._NEW.items():
            self.assertIn(k, names)
            self.assertEqual(getattr(tr, k), v, k)

    def test_default_values_keep_the_emission(self):
        # spelling every merged lever out at its default must not change the IR
        base = _spec(_A16)
        spelled = _spec(replace(_A16, **self._NEW))
        self.assertEqual(_ir(base), _ir(spelled))

    def test_default_name_has_no_new_flag(self):
        toks = _spec(_A16).kernel_name().split("_")
        pat = re.compile(
            r"^(er\d|rl\d|aq\d|ef\d|efs$|sw\d|rilv$|hbs$|rve$|rvnt$|pp\d|ppm\d)"
        )
        self.assertEqual([t for t in toks if pat.match(t)], [])


def _ir(spec):
    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
    )


class TestScheduleBodiesValid(unittest.TestCase):
    def test_reference_bodies(self):
        for tr, tile in (
            (_ER3, _T22),
            (
                replace(
                    _ER3, persistent_ctas=256, chiplet_wgm=16, chiplet_chunk_size=8
                ),
                _T22,
            ),
            (_EF, _T22),
            (replace(_EF, ilv_barrier_after=1, persistent_ctas=256), _T22),
            (_HB, _T22),
            (replace(_HB, rve_store_nt=True, persistent_ctas=256), _T22),
            (_PP7, _T24),
        ):
            for dt in ("fp16", "bf16"):
                ok, why = _ok(tr, tile, dt)
                self.assertTrue(ok, f"{tr}: {why}")

    def test_rilv_composes_with_early_free_and_release_schedules(self):
        for tr in (_ER3, _EF, _A16):
            for extra in (
                _RILV,
                dict(_RILV, rilv_vec_epilogue=True),
                dict(_RILV, rilv_vec_epilogue=True, rve_store_nt=True),
            ):
                ok, why = _ok(replace(tr, **extra))
                self.assertTrue(ok, why)


class TestOneScheduleFamily(unittest.TestCase):
    def test_pairs_rejected(self):
        pairs = [
            replace(_ER3, ilv_early_free=20),
            replace(_ER3, ilv_split_wait=20),
            replace(_EF, ilv_early_release=3),
            replace(_HB, ilv_early_release=3),
            replace(_HB, ilv_early_free=20),
            replace(_HB, ilv_read_span=50),
            replace(_HB, ilv_dtl_span=50),
        ]
        for tr in pairs:
            ok, why = _ok(tr)
            self.assertFalse(ok, f"{tr} must be rejected")

    def test_pingpong_excludes_rilv(self):
        ok, why = _ok(replace(_PP7, **_RILV), _T24)
        self.assertFalse(ok)
        self.assertIn("lds_row_interleave", why)

    def test_pingpong_excludes_ilv_families(self):
        for extra in (
            dict(ilv_early_release=3),
            dict(ilv_early_free=20),
            dict(hb_sched=True),
        ):
            ok, _ = _ok(replace(_PP7, **extra), _T24)
            self.assertFalse(ok, extra)


class TestNameFlags(unittest.TestCase):
    def test_flags_distinguish_every_family(self):
        names = {
            _spec(t, tile).kernel_name()
            for t, tile in (
                (_A16, _T22),
                (_ER3, _T22),
                (_EF, _T22),
                (replace(_EF, ilv_ef_spread=False), _T22),
                (_HB, _T22),
                (replace(_HB, rve_store_nt=True), _T22),
                (replace(_ER3, **_RILV), _T22),
                (_PP7, _T24),
                (replace(_PP7, pp_mix_reads=0), _T24),
            )
        }
        self.assertEqual(len(names), 9)

    def test_specific_flags(self):
        self.assertIn("er3", _spec(_ER3).kernel_name())
        n = _spec(_EF).kernel_name()
        self.assertIn("ef20", n)
        self.assertIn("efs", n)
        self.assertIn("rilv", _spec(replace(_ER3, **_RILV)).kernel_name())
        self.assertIn("pp7", _spec(_PP7, _T24).kernel_name())


class TestEngineGuard(unittest.TestCase):
    def test_python_only_levers_refused(self):
        for tr, tile in (
            (_EF, _T22),
            (replace(_A16, ilv_split_wait=8, ilv_barrier_after=1), _T22),
            (_PP7, _T24),
        ):
            with self.assertRaises(NotImplementedError):
                _be.universal_gemm_spec_to_dict(_spec(tr, tile))

    def test_hb_sched_crosses_to_the_engine(self):
        # the hb_sched slot schedule is mirrored in C++ (parity: gemm_hb_emit)
        d = _be.universal_gemm_spec_to_dict(_spec(_HB, _T22))
        self.assertIs(d["hb_sched"], True)
        self.assertEqual(d["hb_read_every"], 2)

    def test_row_interleave_levers_cross_to_the_engine(self):
        # the row-interleave layout (with the early-release levers) is mirrored in C++
        d = _be.universal_gemm_spec_to_dict(_spec(replace(_ER3, **_RILV), _T22))
        self.assertIs(d["lds_row_interleave"], True)
        self.assertEqual(d["rilv_pad"], 16)

    def test_every_python_only_lever_is_listed(self):
        listed = {k for k, _ in _be._PYTHON_ONLY_GEMM_LEVERS}
        names = {f.name for f in fields(TraitSpec)}
        self.assertTrue(listed <= names, listed - names)
        for k, off in _be._PYTHON_ONLY_GEMM_LEVERS:
            self.assertEqual(getattr(TraitSpec(), k), off, k)

    def test_early_release_levers_cross_to_the_engine(self):
        d = _be.universal_gemm_spec_to_dict(_spec(_ER3))
        self.assertEqual(d["ilv_early_release"], 3)
        self.assertEqual(d["ilv_release_at"], 32)
        self.assertEqual(d["ilv_acquire_at"], 40)

    def test_wrapper_converter_refuses_too(self):
        tr = _be._gemm_lever_keys(_ER3)
        self.assertEqual(tr["ilv_early_release"], 3)
        with self.assertRaises(NotImplementedError):
            _be._gemm_lever_keys(_EF)


if __name__ == "__main__":
    unittest.main()
