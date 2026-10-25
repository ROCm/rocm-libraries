# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the universal-GEMM alias-scoped DTL ring and its levers.

``dtl_stages`` (one A / B LDS allocation per stage + ``lds_alias_scopes`` + a
K-loop unrolled by the stage count), ``dtl_issue_spread``, ``dtl_frag_prefetch``,
``dtl_stream_barrier``, ``mfma_swap_ab`` (swapped-accumulator vector epilogue),
``iglp_opt`` and ``persistent_ctas`` (persistent overlapped-epilogue body):
validity (each lever only where its body implements it), LDS accounting, kernel
names, the emitted structure (per-stage allocations with alias scopes, the
static-stage waits, vector stores, the persistent loop) and that every lever is
default-off (the default trait keeps the historical emission).
"""

from __future__ import annotations

import re
import unittest
import warnings
from dataclasses import replace

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mono_data_spec,
    universal_lds_bytes,
)

_F_TILE = TileSpec(256, 256, 64, 4, 4, 1, 16, 16, 32)
_F_TILE32 = TileSpec(256, 256, 32, 4, 4, 1, 16, 16, 32)
_W24X32 = TileSpec(256, 256, 64, 2, 4, 1, 32, 32, 16)
_F942 = TileSpec(256, 256, 32, 4, 4, 1, 16, 16, 16)
_F_TRAIT = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
)


def _spec(tile=_F_TILE, dtype="fp16", **trait):
    return UniversalGemmSpec(
        name="t",
        tile=tile,
        trait=replace(_F_TRAIT, **trait),
        data=mono_data_spec(dtype),
        wave_size=64,
    )


def _ll(s, arch="gfx950"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_kernel_to_llvm(build_universal_gemm(s, arch=arch), arch=arch)


def _waits(ll):
    """vmcnt fields of the s_waitcnt calls the builder emitted, in order."""
    return [
        int(v)
        for v in re.findall(r"call void @llvm\.amdgcn\.s\.waitcnt\(i32 (\d+)\)", ll)
    ]


class TestRingValidity(unittest.TestCase):
    def test_default_off(self):
        tr = TraitSpec()
        self.assertEqual(
            (
                tr.dtl_stages,
                tr.dtl_issue_spread,
                tr.dtl_frag_prefetch,
                tr.dtl_stream_barrier,
                tr.mfma_swap_ab,
                tr.iglp_opt,
                tr.persistent_ctas,
            ),
            (0, False, False, False, False, None, 0),
        )

    def test_stage_values(self):
        for n, ok in (
            (0, True),
            (1, False),
            (2, True),
            (3, False),
            (4, False),
            (5, False),
        ):
            with self.subTest(stages=n):
                # 256x256x64 at 3 / 4 stages is over the 160 KiB budget
                self.assertEqual(is_valid_spec(_spec(dtl_stages=n), "gfx950")[0], ok)
        for n in (3, 4):
            self.assertTrue(is_valid_spec(_spec(_F_TILE32, dtl_stages=n), "gfx950")[0])

    def test_ring_requires_dtl_prefetch(self):
        ok, why = is_valid_spec(_spec(dtl_stages=2, dtl_prefetch=False), "gfx950")
        self.assertFalse(ok)
        self.assertIn("dtl_stages requires", why)

    def test_levers_require_ring(self):
        for kw in (
            {"dtl_issue_spread": True},
            {"dtl_frag_prefetch": True},
            {"dtl_stream_barrier": True},
            {"iglp_opt": 1},
        ):
            with self.subTest(**kw):
                ok, why = is_valid_spec(_spec(**kw), "gfx950")
                self.assertFalse(ok)
                self.assertIn("requires dtl_stages >= 2", why)
                self.assertTrue(is_valid_spec(_spec(dtl_stages=2, **kw), "gfx950")[0])

    def test_iglp_range(self):
        self.assertFalse(is_valid_spec(_spec(dtl_stages=2, iglp_opt=4), "gfx950")[0])

    def test_persistent_rules(self):
        self.assertTrue(
            is_valid_spec(_spec(dtl_stages=2, persistent_ctas=256), "gfx950")[0]
        )
        for kw, msg in (
            ({"persistent_ctas": 256}, "persistent_ctas requires dtl_stages"),
            (
                {"dtl_stages": 2, "persistent_ctas": 256, "epilogue": "cshuffle"},
                "default epilogue",
            ),
            ({"dtl_stages": 2, "persistent_ctas": -1}, "must be >= 0"),
            # the chiplet swizzle's XCD assignment needs P % #XCDs == 0
            ({"dtl_stages": 2, "persistent_ctas": 100}, "multiple of chiplet_num_xcds"),
        ):
            with self.subTest(**kw):
                ok, why = is_valid_spec(_spec(**kw), "gfx950")
                self.assertFalse(ok)
                self.assertIn(msg, why)

    def test_persistent_admits_the_in_stream_bodies(self):
        # the persistent loop runs the in-stream barrier and
        # the fragment-prefetch bodies too (checked by GPU numerics)
        for kw in ({"dtl_stream_barrier": True}, {"dtl_frag_prefetch": True}):
            with self.subTest(**kw):
                ok, why = is_valid_spec(
                    _spec(dtl_stages=2, persistent_ctas=256, **kw), "gfx950"
                )
                self.assertTrue(ok, why)

    def test_persistent_without_chiplet_takes_any_count(self):
        ok, why = is_valid_spec(
            _spec(dtl_stages=2, persistent_ctas=100, chiplet_swizzle=False), "gfx950"
        )
        self.assertTrue(ok, why)

    def test_swap_rules(self):
        self.assertTrue(is_valid_spec(_spec(mfma_swap_ab=True), "gfx950")[0])
        for kw in ({"epilogue": "cshuffle"}, {"pad_n": True}, {"split_k": 2}):
            with self.subTest(**kw):
                ok, why = is_valid_spec(_spec(mfma_swap_ab=True, **kw), "gfx950")
                self.assertFalse(ok)
                self.assertIn("mfma_swap_ab requires", why)

    def test_lds_accounting(self):
        kw = {
            "pipeline": "compv4",
            "epilogue": "default",
            "direct_to_lds": True,
            "dtl_prefetch": True,
            "ab_double_buffer": None,
            "cshuffle_no_alias": False,
            "lds_k_pad": 0,
            "lds_cap": 163840,
        }
        self.assertEqual(universal_lds_bytes(256, 256, 64, **kw)[1], 131072)
        self.assertEqual(
            universal_lds_bytes(256, 256, 64, dtl_stages=2, **kw)[1], 131072
        )
        self.assertEqual(
            universal_lds_bytes(256, 256, 32, dtl_stages=3, **kw)[1], 98304
        )
        self.assertEqual(
            universal_lds_bytes(256, 256, 32, dtl_stages=4, **kw)[1], 131072
        )

    def test_gfx942_ring_fits(self):
        self.assertTrue(is_valid_spec(_spec(_F942, dtl_stages=2), "gfx942")[0])
        self.assertFalse(is_valid_spec(_spec(_F942, dtl_stages=3), "gfx942")[0])

    def test_dispatch_support_query_sizes_the_ring(self):
        # the dispatcher's own hardware / LDS predicate sizes the ring like the
        # builder gate: 4 stages of 256x256x32 fill 128 KiB, 3 of 256x256x64
        # (192 KiB) are rejected by the LDS budget before the builder gate runs
        from rocke.dispatch.gemm.support import (
            _lds_bytes,
            gemm_config_supported,
            support_query_from_universal_spec,
        )

        q4 = support_query_from_universal_spec(
            _spec(_F_TILE32, dtl_stages=4), arch="gfx950"
        )
        self.assertEqual(_lds_bytes(q4, 163840)[1], 131072)
        self.assertTrue(gemm_config_supported(q4)[0])
        q3 = support_query_from_universal_spec(_spec(dtl_stages=3), arch="gfx950")
        self.assertEqual(_lds_bytes(q3, 163840)[1], 196608)
        ok, why = gemm_config_supported(q3)
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)

    def test_kernel_name_flags(self):
        names = {
            _spec().kernel_name(),
            _spec(dtl_stages=2).kernel_name(),
            _spec(dtl_stages=2, dtl_issue_spread=True).kernel_name(),
            _spec(dtl_stages=2, dtl_stream_barrier=True).kernel_name(),
            _spec(dtl_stages=2, dtl_frag_prefetch=True).kernel_name(),
            _spec(dtl_stages=2, iglp_opt=1).kernel_name(),
            _spec(dtl_stages=2, persistent_ctas=256).kernel_name(),
            _spec(mfma_swap_ab=True).kernel_name(),
        }
        self.assertEqual(len(names), 8)
        self.assertTrue(
            _spec(dtl_stages=2, persistent_ctas=256)
            .kernel_name()
            .endswith("_stg2_pc256")
        )


class TestRingEmission(unittest.TestCase):
    def test_per_stage_allocations_with_alias_scopes(self):
        base = _ll(_spec())
        ring = _ll(_spec(dtl_stages=2))
        self.assertNotIn("!alias.scope", base)
        self.assertIn("!alias.scope", ring)
        # every buffer_load...lds and every ds_read carries a scope
        lds_dma = re.findall(
            r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\([^\n]*", ring
        )
        self.assertTrue(lds_dma)
        self.assertTrue(all("!alias.scope" in x for x in lds_dma))
        reads = re.findall(r"load <8 x half>, ptr addrspace\(3\)[^\n]*", ring)
        self.assertTrue(reads)
        self.assertTrue(all("!alias.scope" in x for x in reads))
        # 4 distinct allocation scopes (A0, A1, B0, B1), and the DMA and the
        # reads of one stage share its scope
        dma_scopes = {re.search(r"!alias\.scope (!\d+)", x).group(1) for x in lds_dma}
        read_scopes = {re.search(r"!alias\.scope (!\d+)", x).group(1) for x in reads}
        self.assertEqual(len(dma_scopes), 4)
        self.assertEqual(dma_scopes, read_scopes)

    def test_partial_vmcnt_for_three_stages(self):
        # two tiles in flight: the in-loop wait keeps one tile's loads (L = 4)
        ll = _ll(_spec(_F_TILE32, dtl_stages=3))
        waits = _waits(ll)
        self.assertTrue(waits)
        # s_waitcnt encodes vmcnt in bits [3:0] + [15:14]; vmcnt(4), lgkmcnt(0)
        vm = [((w >> 10) & 0x30) | (w & 0xF) for w in waits]
        # L = 2 loads per tile (256x32 tiles, one pass per operand): every main-loop
        # wait keeps L loads (the next tile) in flight, the 2-tile tail waits
        # vmcnt(2) then vmcnt(0)
        self.assertEqual(vm, [2, 2, 2, 2, 0])

    def test_default_trait_emission_unchanged_by_new_fields(self):
        a = _ll(_spec())
        b = _ll(
            _spec(
                dtl_stages=0,
                dtl_issue_spread=False,
                mfma_swap_ab=False,
                iglp_opt=None,
                persistent_ctas=0,
            )
        )
        self.assertEqual(a, b)

    def test_swap_ab_vector_stores(self):
        scalar = _ll(_spec(dtl_stages=2))
        vec = _ll(_spec(dtl_stages=2, mfma_swap_ab=True))
        self.assertEqual(len(re.findall(r"store <4 x half>", scalar)), 0)
        self.assertEqual(len(re.findall(r"store <4 x half>", vec)), 16)
        self.assertEqual(len(re.findall(r"store half ", vec)), 0)

    def test_swap_ab_store_alignment_follows_the_dtype(self):
        # 4 x 2-byte elements per vector store -> align 8 (derived from the
        # storage dtype's width, not hard-coded)
        for dt, ty in (("fp16", "half"), ("bf16", "bfloat")):
            with self.subTest(dtype=dt):
                ll = _ll(_spec(dtype=dt, dtl_stages=2, mfma_swap_ab=True))
                st = re.findall(r"store <4 x %s> [^\n]*" % ty, ll)
                self.assertEqual(len(st), 16)
                self.assertTrue(all("align 8" in x for x in st), st[:2])

    def test_swap_ab_feeds_b_first(self):
        # mfma_swap_ab: every MFMA's first operand is a B-stage fragment (its
        # ds_read carries a B allocation's alias scope), A's without the lever
        def first_operand_allocs(ll):
            scope_name = {
                m.group(1): m.group(2)
                for m in re.finditer(
                    r'^!(\d+) = distinct !\{!\d+, !\d+, !"@([AB])_smem_s', ll, re.M
                )
            }
            list_of = {
                m.group(1): m.group(2)
                for m in re.finditer(r"^!(\d+) = !\{!(\d+)\}$", ll, re.M)
            }
            out = set()
            for m in re.finditer(
                r"@llvm\.amdgcn\.mfma[^(]*\(<8 x half> (%[\w.]+),", ll
            ):
                d = re.search(
                    re.escape(m.group(1))
                    + r" = load <8 x half>[^\n]*!alias\.scope !(\d+)",
                    ll,
                )
                out.add(scope_name[list_of[d.group(1)]])
            return out

        self.assertEqual(first_operand_allocs(_ll(_spec(dtl_stages=2))), {"A"})
        self.assertEqual(
            first_operand_allocs(_ll(_spec(dtl_stages=2, mfma_swap_ab=True))), {"B"}
        )

    def test_persistent_loop(self):
        ll = _ll(_spec(dtl_stages=2, persistent_ctas=256))
        self.assertIn("ptile", ll)
        # the epilogue (64 scalar stores) sits inside the persistent loop
        self.assertEqual(len(re.findall(r"store half ", ll)), 64)

    def test_every_lever_builds(self):
        for tile, kw in (
            (_F_TILE, {"dtl_stages": 2, "interwave_setprio": True}),
            (_F_TILE, {"dtl_stages": 2, "dtl_issue_spread": True}),
            (_F_TILE, {"dtl_stages": 2, "dtl_stream_barrier": True}),
            (_F_TILE, {"dtl_stages": 2, "iglp_opt": 0}),
            (
                _F_TILE,
                {"dtl_stages": 2, "persistent_ctas": 256, "chiplet_swizzle": False},
            ),
            (_W24X32, {"dtl_stages": 2, "dtl_frag_prefetch": True}),
            (_F_TILE32, {"dtl_stages": 4, "epilogue": "cshuffle"}),
        ):
            with self.subTest(**kw):
                s = _spec(tile, **kw)
                self.assertTrue(is_valid_spec(s, "gfx950")[0])
                self.assertIn("define amdgpu_kernel", _ll(s))
        self.assertIn("define amdgpu_kernel", _ll(_spec(_F942, dtl_stages=2), "gfx942"))


class TestDispatchPick(unittest.TestCase):
    def test_gfx950_pick_uses_the_ring(self):
        from rocke.dispatch.gemm.common import (
            dtl_pingpong_sweep_variants,
            dtl_pingpong_tile_trait,
        )

        _t, tr = dtl_pingpong_tile_trait("gfx950")
        self.assertEqual((tr.dtl_stages, tr.interwave_setprio), (2, True))
        _t, tr942 = dtl_pingpong_tile_trait("gfx942")
        self.assertEqual((tr942.dtl_stages, tr942.interwave_setprio), (0, False))
        _t, rf = dtl_pingpong_tile_trait("gfx950", ring=False)
        self.assertEqual(rf.dtl_stages, 0)
        spec = UniversalGemmSpec(
            name="d", tile=_t, trait=tr, data=mono_data_spec("fp16"), wave_size=64
        )
        variants = dtl_pingpong_sweep_variants(spec, "gfx950")
        self.assertEqual(variants[0], spec)
        stages = {v.trait.dtl_stages for v in variants}
        self.assertEqual(stages, {0, 2, 3})  # + the 3-stage AGPR interleave bodies
        self.assertTrue(any(v.trait.persistent_ctas == 256 for v in variants))
        self.assertTrue(all(is_valid_spec(v, "gfx950")[0] for v in variants))
        # the dispatcher-relevant ring variants are reachable by a sweep
        self.assertTrue(
            any(v.trait.persistent_ctas == 256 and v.trait.dtl_hoist for v in variants)
        )
        self.assertTrue(any(v.trait.iglp_opt == 1 for v in variants))
        self.assertTrue(
            any(
                v.trait.iglp_opt == 1
                and v.trait.chiplet_wgm == 4
                and not v.trait.acc_agpr
                for v in variants
            )
        )
        ilv = [
            v for v in variants if v.trait.acc_agpr and not v.trait.lds_row_interleave
        ]
        self.assertEqual(sorted(v.trait.chiplet_wgm for v in ilv), [4, 8])
        for v in ilv:
            self.assertEqual((v.tile.warp_m, v.tile.warp_n, v.tile.tile_k), (2, 2, 32))
            self.assertEqual(v.block_size, 256)
        # both row-interleave schedules (tile_k 64) are reachable
        rilv = [v for v in variants if v.trait.lds_row_interleave]
        self.assertTrue(any(v.trait.hb_sched for v in rilv))
        self.assertTrue(any(v.trait.ilv_merge for v in rilv))
        for v in rilv:
            self.assertEqual((v.tile.warp_m, v.tile.warp_n, v.tile.tile_k), (2, 2, 64))
        # gfx942 keeps its ping-pong neighbourhood (no AGPR interleave geometry there)
        _t4, tr4 = dtl_pingpong_tile_trait("gfx942")
        s4 = UniversalGemmSpec(
            name="d", tile=_t4, trait=tr4, data=mono_data_spec("fp16"), wave_size=64
        )
        self.assertFalse(
            any(v.trait.acc_agpr for v in dtl_pingpong_sweep_variants(s4, "gfx942"))
        )

    def test_persistent_launch_grid(self):
        from rocke.dispatch.gemm import bf16_rcr, fp16_rcr
        from rocke.dispatch.gemm.common import GemmRequest, persistent_grid

        s = _spec(dtl_stages=2, persistent_ctas=256)
        self.assertEqual(persistent_grid(s, 8192, 8192), (256, 1, 1))
        self.assertEqual(persistent_grid(s, 512, 512), (4, 1, 1))  # 4 tiles < P
        for mod, dt in ((fp16_rcr, "fp16"), (bf16_rcr, "bf16")):
            req = GemmRequest(M=8192, N=4096, K=1024, arch="gfx950", dtype=dt)
            self.assertEqual(mod._grid(s, req), (256, 1, 1))
            # non-persistent bodies keep the 2D tile grid
            self.assertEqual(mod._grid(_spec(dtl_stages=2), req), (16, 32, 1))

    def test_cost_models_keyed_by_body(self):
        from unittest import mock

        from rocke.dispatch.gemm.common import (
            DTL_ADMIT_ENV,
            DTL_PINGPONG_COST_MODELS,
            DTL_PINGPONG_COST_MODELS_BY_BODY,
            GemmRequest,
            dtl_pingpong_admit,
            dtl_pingpong_body,
            dtl_pingpong_tile_trait,
        )

        for dt in ("fp16", "bf16"):
            ring = DTL_PINGPONG_COST_MODELS_BY_BODY[("gfx950", dt, "ring")]
            rf = DTL_PINGPONG_COST_MODELS_BY_BODY[("gfx950", dt, "pingpong")]
            self.assertNotEqual(ring, rf)
            # the (arch, dtype) view describes the dispatched (ring) body
            self.assertEqual(DTL_PINGPONG_COST_MODELS[("gfx950", dt)], ring)
            self.assertEqual(
                DTL_PINGPONG_COST_MODELS[("gfx942", dt)],
                DTL_PINGPONG_COST_MODELS_BY_BODY[("gfx942", dt, "pingpong")],
            )
        t, tr = dtl_pingpong_tile_trait("gfx950")
        _t, rf_tr = dtl_pingpong_tile_trait("gfx950", ring=False)
        ring_spec = UniversalGemmSpec(
            name="r", tile=t, trait=tr, data=mono_data_spec("fp16"), wave_size=64
        )
        rf_spec = replace(ring_spec, trait=rf_tr)
        self.assertEqual(
            (dtl_pingpong_body(ring_spec), dtl_pingpong_body(rf_spec)),
            ("ring", "pingpong"),
        )
        # 1024x2048x4096: the ring's coefficients admit (at either body's margin),
        # the ping-pong coefficients do not -> the ping-pong body is judged with its own
        # coefficients
        req = GemmRequest(M=1024, N=2048, K=4096, arch="gfx950")
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "cost_model"}):
            self.assertTrue(dtl_pingpong_admit(req, ring_spec)[0])
            self.assertFalse(dtl_pingpong_admit(req, rf_spec)[0])
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "rule"}):
            ok, why = dtl_pingpong_admit(req, rf_spec)
            self.assertFalse(ok)
            self.assertIn("calibrated for the ring", why)
        with mock.patch.dict("os.environ", {DTL_ADMIT_ENV: "auto"}):
            # auto on the ping-pong body falls back to its cost model, not the rule
            big = GemmRequest(M=8192, N=8192, K=8192, arch="gfx950")
            ok, why = dtl_pingpong_admit(big, rf_spec)
            self.assertTrue(ok)
            self.assertNotIn("threshold rule", why)


if __name__ == "__main__":
    unittest.main()
