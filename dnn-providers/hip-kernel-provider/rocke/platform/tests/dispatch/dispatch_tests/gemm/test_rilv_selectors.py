# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the row-interleave body selectors and related gates.

* ``ROCKE_GEMM_RILV_BODY`` / ``ROCKE_GEMM_RILV_HB_LOOP`` select the retained
  alternative bodies without monkeypatching, default to the tuned IR body and
  reject unknown values.
* ``is_valid_spec`` rejects the bodies without a fused-epilogue hook when a
  ``helpers.fuse`` epilogue is attached, instead of the builder raising; the
  row-interleave vector epilogue takes one, and ``fused_epilogue_spec`` maps a
  dispatched body to the nearest one that does.
* The non-DTL dispatcher LDS budget keeps the CK-mirrored per-pipeline cap.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from rocke.dispatch import dispatch_gemm_fp16
from rocke.dispatch.gemm import common as C
from rocke.dispatch.gemm.common import GemmRequest
from rocke.dispatch.gemm.support import (
    gemm_config_supported,
    support_query_from_universal_spec,
)
from rocke.instances.common.gemm_universal import is_valid_spec

_REQ = GemmRequest(M=8192, N=8192, K=8192, arch="gfx950")


def _env(**kv):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROCKE_GEMM_RILV")}
    env.update(kv)
    return mock.patch.dict(os.environ, env, clear=True)


class TestRilvSelectors(unittest.TestCase):
    def test_defaults(self):
        with _env():
            self.assertEqual(C.rilv_body(), "tuned")
            self.assertEqual(C.rilv_hb_loop(), "ir")
            tr = dispatch_gemm_fp16(_REQ).spec.trait
        self.assertTrue(tr.lds_row_interleave and tr.hb_sched)
        self.assertEqual(tr.asm_loop, "")

    def test_env_selects_alternatives(self):
        with _env(ROCKE_GEMM_RILV_BODY="asm_loop"):
            tr = dispatch_gemm_fp16(_REQ).spec.trait
        self.assertEqual(tr.asm_loop, "tensile")
        with _env(ROCKE_GEMM_RILV_BODY="base"):
            tr = dispatch_gemm_fp16(_REQ).spec.trait
        self.assertTrue(tr.lds_row_interleave)
        self.assertFalse(tr.dtl_soff0)
        with _env(ROCKE_GEMM_RILV_HB_LOOP="asm_loop"):
            tr = dispatch_gemm_fp16(
                GemmRequest(M=16384, N=16384, K=16384, arch="gfx950")
            ).spec.trait
        self.assertEqual(tr.asm_loop, "tensile")
        self.assertEqual((tr.hb_tile_lgkm0, tr.das_late), (0, False))

    def test_env_rejects_unknown(self):
        with _env(ROCKE_GEMM_RILV_BODY="x4"):
            with self.assertRaises(ValueError):
                C.rilv_body()
        with _env(ROCKE_GEMM_RILV_HB_LOOP="tclone"):
            with self.assertRaises(ValueError):
                C.rilv_hb_loop()


class TestFusedEpilogueGate(unittest.TestCase):
    def _with_fused(self, spec):
        object.__setattr__(spec, "_fused_epilogue", object())
        return spec

    def test_rejects_bodies_without_fused_hook(self):
        with _env():
            spec = dispatch_gemm_fp16(_REQ).spec
        self.assertTrue(is_valid_spec(spec, arch="gfx950")[0])
        for lever in (
            dict(persistent_ctas=256),
            dict(split_k=2),
            dict(streamk=True),
        ):
            with self.subTest(**lever):
                s = replace(spec, trait=replace(spec.trait, **lever))
                ok, why = is_valid_spec(self._with_fused(s), arch="gfx950")
                self.assertFalse(ok)
                self.assertIn("fused epilogue", why)

    def test_row_interleave_vector_epilogue_takes_a_fused_epilogue(self):
        with _env():
            spec = dispatch_gemm_fp16(_REQ).spec
        self.assertTrue(spec.trait.rilv_vec_epilogue)
        body = C.fused_epilogue_spec(spec)
        self.assertFalse(body.trait.persistent_ctas)
        self.assertTrue(body.trait.rilv_vec_epilogue)
        self.assertTrue(
            is_valid_spec(self._with_fused(replace(body)), arch="gfx950")[0]
        )

    def test_fused_epilogue_spec_reroutes_or_refuses(self):
        with _env():
            spec = dispatch_gemm_fp16(_REQ).spec
        pers = replace(spec, trait=replace(spec.trait, persistent_ctas=256))
        self.assertEqual(C.fused_epilogue_spec(pers).trait.persistent_ctas, 0)
        self.assertIsNone(
            C.fused_epilogue_spec(replace(spec, trait=replace(spec.trait, split_k=2)))
        )

    def test_plain_body_keeps_fused_epilogue(self):
        spec = dispatch_gemm_fp16(
            GemmRequest(M=1024, N=1024, K=256, arch="gfx950")
        ).spec
        self.assertFalse(spec.trait.rilv_vec_epilogue)
        self.assertIs(C.fused_epilogue_spec(spec), spec)
        self.assertTrue(
            is_valid_spec(self._with_fused(replace(spec)), arch="gfx950")[0]
        )


class TestNonDtlLdsCap(unittest.TestCase):
    def test_ck_pipeline_cap_still_bounds_non_dtl_bodies(self):
        # a VGPR-staged (non-DTL) compv4 body
        from rocke.instances.common.gemm_universal import (
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            mono_data_spec,
        )

        spec = UniversalGemmSpec(
            name="non_dtl",
            tile=TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=mono_data_spec("fp16"),
        )
        self.assertFalse(spec.trait.direct_to_lds)
        q = support_query_from_universal_spec(spec, arch="gfx950")
        self.assertTrue(gemm_config_supported(q)[0])
        # one A + B tile above the 32 KiB compv4 budget (but inside 160 KiB)
        big = replace(q, cta_tile=(256, 256, 64), pipeline="compv4")
        ok, why = gemm_config_supported(big)
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)


if __name__ == "__main__":
    unittest.main()
