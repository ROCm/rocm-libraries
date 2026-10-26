# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM validity gate, LDS plan and knobs.

Covers the gate's LDS accounting (the cshuffle C tile aliases the A/B bytes),
the explicit ``ab_double_buffer`` knob, the trait values whose emission is kept
historical (``persistent`` is an accepted no-op rename, ``scheduler="interwave"``
emits the intrawave body) next to the opt-in ``interwave_setprio`` knob, the two
LDS write-granularity rules (full DTL passes, swizzle group >= write chunk) and
their independence from the ``CK_SWZ_*`` debug env vars, the kernel-name flags
that keep lever variants distinguishable, the lever bundles of the config
enumerator, and that the reserved LDS equals what the lowered kernel allocates.
"""

from __future__ import annotations

import os
import re
import unittest
import warnings
from dataclasses import replace
from unittest import mock

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    DEFAULT_LEVER_BUNDLES,
    LEVER_BUNDLES,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    all_dispatcher_configs,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
    lds_swizzle_params,
    mono_data_spec,
    universal_lds_bytes,
)

_F_TILE = TileSpec(256, 256, 64, 4, 4, 1, 16, 16, 32)
_F_TRAIT = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
)
_PICK_TILE = TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16)


def _spec(tile, trait, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=tile, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _plan(s, cap=163840):
    t, tr = s.tile, s.trait
    return universal_lds_bytes(
        t.tile_m,
        t.tile_n,
        t.tile_k,
        pipeline=tr.pipeline,
        epilogue=tr.epilogue,
        direct_to_lds=tr.direct_to_lds,
        dtl_prefetch=tr.dtl_prefetch,
        ab_double_buffer=tr.ab_double_buffer,
        cshuffle_no_alias=tr.cshuffle_no_alias,
        lds_k_pad=tr.lds_k_pad,
        lds_cap=cap,
    )


def _ll(s, arch="gfx950"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_kernel_to_llvm(build_universal_gemm(s, arch=arch), arch=arch)


def _code(s, arch="gfx950"):
    """The lowered IR with the kernel symbol replaced (code identity)."""
    return _ll(s, arch).replace(s.kernel_name(), "KNAME")


def _ir_lds_bytes(s, arch="gfx950"):
    ll = lower_kernel_to_llvm(build_universal_gemm(s, arch=arch), arch=arch)
    return sum(int(n) for n in re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", ll))


class TestLdsAccounting(unittest.TestCase):
    def test_element_sizes(self):
        kw = {
            "pipeline": "compv4",
            "epilogue": "cshuffle",
            "direct_to_lds": False,
            "dtl_prefetch": False,
            "ab_double_buffer": False,
            "cshuffle_no_alias": True,
            "lds_k_pad": 0,
            "lds_cap": 163840,
        }
        self.assertEqual(universal_lds_bytes(128, 128, 64, **kw), (65536, 32768, 32768))
        self.assertEqual(
            universal_lds_bytes(
                128, 128, 64, elem_bytes_a=1, elem_bytes_b=1, elem_bytes_c=4, **kw
            ),
            (81920, 16384, 65536),
        )

    def test_cshuffle_c_aliases_ab(self):
        s = _spec(_F_TILE, replace(_F_TRAIT, epilogue="cshuffle"))
        self.assertEqual(_plan(s), (131072, 131072, 131072))
        ok, why = is_valid_spec(s, "gfx950")
        self.assertTrue(ok, why)

    def test_cshuffle_no_alias_is_additive(self):
        s = _spec(
            _F_TILE, replace(_F_TRAIT, epilogue="cshuffle", cshuffle_no_alias=True)
        )
        ok, why = is_valid_spec(s, "gfx950")
        self.assertFalse(ok)
        self.assertIn("LDS budget 262144", why)

    def test_gfx942_cap_still_binds(self):
        s = _spec(
            TileSpec(256, 256, 32, 4, 4, 1, 16, 16, 16),
            replace(_F_TRAIT, epilogue="cshuffle"),
        )
        ok, why = is_valid_spec(s, "gfx942")
        self.assertFalse(ok)
        self.assertIn("65536 cap", why)

    def test_reserved_equals_allocated(self):
        for s in (
            _spec(_F_TILE, _F_TRAIT),
            _spec(_F_TILE, replace(_F_TRAIT, epilogue="cshuffle")),
            _spec(_PICK_TILE, TraitSpec(pipeline="compv4", epilogue="cshuffle")),
            _spec(
                _PICK_TILE,
                TraitSpec(
                    pipeline="compv4", epilogue="cshuffle", ab_double_buffer=True
                ),
            ),
            _spec(
                _F_TILE,
                TraitSpec(pipeline="compv4", epilogue="default", ab_double_buffer=True),
            ),
            _spec(_F_TILE, TraitSpec(pipeline="compv4", epilogue="default")),
        ):
            with self.subTest(name=s.kernel_name()):
                self.assertTrue(is_valid_spec(s, "gfx950")[0])
                self.assertEqual(_ir_lds_bytes(s), _plan(s)[0])


class TestAbDoubleBuffer(unittest.TestCase):
    def test_auto_rule_unchanged(self):
        # 256x256x64 compv4: the doubled AB (128 KiB) does not fit 2 WG/CU, so the
        # automatic rule keeps the single buffer (historical emission).
        self.assertEqual(
            _plan(_spec(_F_TILE, TraitSpec(pipeline="compv4", epilogue="default")))[1],
            65536,
        )

    def test_forced_double_buffer(self):
        s = _spec(
            _F_TILE,
            TraitSpec(pipeline="compv4", epilogue="default", ab_double_buffer=True),
        )
        self.assertEqual(_plan(s)[1], 131072)
        self.assertTrue(s.kernel_name().endswith("_abdb"))

    def test_rejected_off_the_compv4_vgpr_body(self):
        for tr in (
            replace(_F_TRAIT, ab_double_buffer=True),
            TraitSpec(pipeline="compv3", epilogue="default", ab_double_buffer=False),
        ):
            with self.subTest(trait=tr):
                ok, why = is_valid_spec(_spec(_F_TILE, tr), "gfx950")
                self.assertFalse(ok)
                self.assertIn("ab_double_buffer applies only", why)


class TestTraitValues(unittest.TestCase):
    def test_persistent_is_an_accepted_noop_rename(self):
        tr = TraitSpec(pipeline="compv4", epilogue="default")
        plain = _spec(_PICK_TILE, tr)
        pers = _spec(_PICK_TILE, replace(tr, persistent=True))
        ok, why = is_valid_spec(pers, "gfx950")
        self.assertTrue(ok, why)
        self.assertTrue(pers.kernel_name().endswith("_pers"))
        with self.assertWarns(UserWarning):
            build_universal_gemm(pers, arch="gfx950")
        self.assertEqual(_code(plain), _code(pers))

    def test_scheduler_interwave_keeps_intrawave_emission(self):
        # Shipped tuned kernels (qwen3 decode, gemm_perf_skinny_decode) set
        # scheduler="interwave"; it must keep emitting the historical body.
        for tile, tr in (
            (_F_TILE, _F_TRAIT),
            (_PICK_TILE, TraitSpec(pipeline="compv4", epilogue="cshuffle")),
        ):
            a = _spec(tile, tr)
            b = _spec(tile, replace(tr, scheduler="interwave"))
            with self.subTest(name=a.kernel_name()):
                self.assertTrue(is_valid_spec(b, "gfx950")[0])
                self.assertEqual(_code(a), _code(b))

    def test_interwave_setprio_changes_emission(self):
        base = _spec(_F_TILE, _F_TRAIT)
        iw = _spec(_F_TILE, replace(_F_TRAIT, interwave_setprio=True))
        self.assertTrue(iw.kernel_name().endswith("_iwp"))
        self.assertEqual(_ll(base).count("call void @llvm.amdgcn.s.setprio"), 0)
        self.assertGreater(_ll(iw).count("call void @llvm.amdgcn.s.setprio"), 0)

    def test_interwave_setprio_rejected_on_wmma(self):
        s = UniversalGemmSpec(
            name="t",
            tile=TileSpec(64, 32, 16, 2, 1, 1, 16, 16, 16),
            trait=TraitSpec(pipeline="mem", epilogue="default", interwave_setprio=True),
            data=mono_data_spec("fp16"),
            wave_size=32,
        )
        ok, why = is_valid_spec(s, "gfx1151")
        self.assertFalse(ok)
        self.assertIn("interwave_setprio is MFMA-only", why)
        ok, why = is_valid_spec(
            replace(s, trait=replace(s.trait, interwave_setprio=False)), "gfx1151"
        )
        self.assertTrue(ok, why)

    def test_unknown_scheduler_rejected(self):
        ok, why = is_valid_spec(
            _spec(_PICK_TILE, TraitSpec(scheduler="bogus")), "gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("unknown scheduler", why)


class TestWriteGranularity(unittest.TestCase):
    def test_swizzle_params_geometry(self):
        self.assertEqual(lds_swizzle_params(8, 8, 32), (0, 2, 3))
        self.assertEqual(lds_swizzle_params(4, 4, 32), (0, 3, 2))
        self.assertEqual(lds_swizzle_params(8, 4, 32), (3, 1, 4))

    def test_gate_ignores_swizzle_debug_env(self):
        good = _spec(
            TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32),
            TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
        )
        bad = _spec(
            TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
            TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
        )
        for v in ("1", "5"):
            with self.subTest(CK_SWZ_L=v), mock.patch.dict(os.environ, {"CK_SWZ_L": v}):
                self.assertTrue(is_valid_spec(good, "gfx950")[0])
                self.assertFalse(is_valid_spec(bad, "gfx950")[0])

    def test_swizzle_group_narrower_than_store_rejected(self):
        for atom in ((16, 16, 16), (32, 32, 8)):
            s = _spec(
                TileSpec(128, 128, 32, 2, 2, 1, *atom),
                TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
            )
            with self.subTest(atom=atom):
                ok, why = is_valid_spec(s, "gfx950")
                self.assertFalse(ok)
                self.assertIn("lds_swizzle: XOR group", why)

    def test_swizzle_wide_atom_accepted(self):
        s = _spec(
            TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32),
            TraitSpec(pipeline="compv4", epilogue="default", lds_swizzle=True),
        )
        self.assertTrue(is_valid_spec(s, "gfx950")[0])

    def test_dtl_k_tail_is_outside_the_launch_contract(self):
        # DTL launches with a partial K tile compute wrong outputs even with
        # pad_k (the DRAM->LDS loads do not mask K), so they are refused
        s = _spec(_F_TILE, replace(_F_TRAIT, pad_m=True, pad_n=True, pad_k=True))
        self.assertTrue(is_valid_spec(s, arch="gfx950")[0])  # K-multiple launches
        self.assertIsNone(launch_shape_error(s, 1024, 1024, 4096))
        err = launch_shape_error(s, 1024, 1024, 1000)
        self.assertIsNotNone(err)
        self.assertIn("direct_to_lds", err)
        # ragged rows: the unbounded DMA would read rows past the operand
        err = launch_shape_error(s, 1000, 1000, 4096)
        self.assertIsNotNone(err)
        self.assertIn("dtl_tight_rsrc", err)
        # the global-load path masks the K tail: a plain pad_k body is fine
        # (ragged rows: pad_row_clamp keeps the row loads inside the operand)
        plain = _spec(
            _PICK_TILE, TraitSpec(pad_m=True, pad_n=True, pad_k=True, pad_row_clamp=True)
        )
        self.assertTrue(is_valid_spec(plain, arch="gfx950")[0])
        self.assertIsNone(launch_shape_error(plain, 1000, 1000, 1000))

    def test_partial_dtl_pass_rejected(self):
        s = _spec(
            TileSpec(128, 128, 32, 4, 4, 1, 16, 16, 32),
            replace(_F_TRAIT, lds_swizzle=False),
        )
        ok, why = is_valid_spec(s, "gfx950")
        self.assertFalse(ok)
        self.assertIn("not a multiple of block_size 1024", why)


class TestKernelNames(unittest.TestCase):
    def test_lever_variants_have_distinct_names(self):
        base = replace(
            _F_TRAIT, chiplet_swizzle=True, chiplet_wgm=8, chiplet_chunk_size=64
        )
        variants = (
            base,
            replace(base, chiplet_chunk_size=32),
            replace(base, chiplet_swizzle=False),
            replace(base, lds_swizzle=False),
            replace(base, dtl_cache_b=2),
            replace(base, dtl_cache_a=2),
            replace(base, interwave_setprio=True),
        )
        names = [_spec(_F_TILE, tr).kernel_name() for tr in variants]
        self.assertEqual(len(set(names)), len(names), names)
        self.assertTrue(names[0].endswith("_dtl_pref_swz_chip8x8c64_ca0cb0"))

    def test_unused_levers_keep_historical_name(self):
        s = _spec(_PICK_TILE, TraitSpec(pipeline="compv4", epilogue="cshuffle"))
        self.assertEqual(
            s.kernel_name(),
            "t_fp16_t128x128x32_w2x2x1_wt32x32x16_compv4_intrawave_cshuffle",
        )
        # dtl cache hints are only encoded when DTL is on and they differ from
        # the default (CACHE_ALL A, CACHE_STREAM B).
        s2 = _spec(_PICK_TILE, TraitSpec(pipeline="compv4", dtl_cache_b=0))
        self.assertNotIn("_ca", s2.kernel_name())


class TestLeverBundles(unittest.TestCase):
    def test_default_enumeration_is_historical(self):
        plain = list(all_dispatcher_configs(arch="gfx950"))
        self.assertEqual(
            len(plain),
            len(list(all_dispatcher_configs(arch="gfx950", levers=("none",)))),
        )
        self.assertTrue(
            all(not s.trait.direct_to_lds and not s.trait.lds_swizzle for s in plain)
        )
        self.assertTrue(all(s.name == "rocke_universal" for s in plain))

    def test_bundle_names_are_distinct(self):
        names = {
            s.name
            for s in all_dispatcher_configs(
                arch="gfx950",
                tile_m=(256,),
                tile_n=(256,),
                levers=DEFAULT_LEVER_BUNDLES,
            )
        }
        self.assertIn("rocke_universal_dtl_pingpong_chiplet", names)
        self.assertIn("rocke_universal", names)

    def test_pingpong_is_in_the_bundle_space(self):
        hits = [
            s
            for s in all_dispatcher_configs(
                arch="gfx950",
                pipeline=("compv4",),
                epilogue=("default",),
                levers=DEFAULT_LEVER_BUNDLES,
            )
            if s.tile == _F_TILE
            and s.trait.dtl_prefetch
            and s.trait.lds_swizzle
            and s.trait.dtl_cache_b == 0
        ]
        self.assertTrue(hits)

    def test_unknown_bundle(self):
        with self.assertRaises(ValueError):
            list(all_dispatcher_configs(levers=("nope",)))

    def test_every_bundle_is_trait_fields(self):
        fields = set(TraitSpec.__dataclass_fields__)
        for name, kw in LEVER_BUNDLES.items():
            with self.subTest(bundle=name):
                self.assertLessEqual(set(kw), fields)


class TestSweepDefaultSet(unittest.TestCase):
    def test_validates_for_the_isa_arch(self):
        from rocke import sweep

        seen = {}

        def fake_build(specs, **kw):
            seen[kw["isa"]] = list(specs)
            return []

        with mock.patch.object(sweep, "build_all_instances", fake_build):
            for isa in (
                "amdgcn-amd-amdhsa--gfx950",
                "amdgcn-amd-amdhsa--gfx942:sramecc+",
            ):
                sweep.build_default_dispatcher_set(cache_dir=None, isa=isa)
        s950 = seen["amdgcn-amd-amdhsa--gfx950"]
        s942 = seen["amdgcn-amd-amdhsa--gfx942:sramecc+"]
        self.assertTrue(s950 and s942)
        self.assertTrue(all(is_valid_spec(s, "gfx942")[0] for s in s942))
        # gfx950-only atoms (e.g. 32x32x16) must not reach a gfx942 build
        self.assertTrue(any(not is_valid_spec(s, "gfx942")[0] for s in s950))


if __name__ == "__main__":
    unittest.main()
