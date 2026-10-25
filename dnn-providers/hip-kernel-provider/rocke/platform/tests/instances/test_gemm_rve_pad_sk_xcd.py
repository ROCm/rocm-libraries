# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the vector-epilogue bounds and XCD-local stream-K levers.

* ``rve_pad`` (1 / 2): the row-interleave vector epilogue on ragged M / N
  (1 = control-flow guards, 2 = branch-free buffer stores whose out-of-range
  lanes get voffset = num_records), with pad + ``dtl_tight_rsrc``;
* ``sk_xcd_local``: XCD-major stream-K ranks (owner and partners on one L2;
  workgroup-scope fences for same-XCD hand-offs).

Both are Python-engine-only, default-off levers: the default trait keeps the
historical emission, the kernel name gets a flag only when they are on, the
C-engine converters refuse them, and the validity / launch rules hold.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rocke.core import backend as _be
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
    mono_data_spec,
)

_T22 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
# the merged-interleave row-interleave body (rilv + vector nt epilogue)
_RIM = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
_PAD = dict(pad_n=True, dtl_tight_rsrc=True, pad_fast_tile=True)
_SK = dict(persistent_ctas=256, streamk=True)


def _spec(trait, dtype="fp16"):
    return UniversalGemmSpec(
        name="t", tile=_T22, trait=trait, data=mono_data_spec(dtype), wave_size=64
    )


def _ir(spec):
    return lower_kernel_to_llvm(
        build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
    )


class TestDefaults(unittest.TestCase):
    def test_defaults_off(self):
        tr = TraitSpec()
        self.assertEqual(tr.rve_pad, 0)
        self.assertFalse(tr.sk_xcd_local)

    def test_spelled_defaults_keep_the_emission(self):
        for base in (_RIM, replace(_RIM, **_SK)):
            self.assertEqual(
                _ir(_spec(base)),
                _ir(_spec(replace(base, rve_pad=0, sk_xcd_local=False))),
            )

    def test_name_flags(self):
        toks = _spec(_RIM).kernel_name().split("_")
        self.assertFalse({"rvp1", "rvp2", "skx"} & set(toks))
        for v in (1, 2):
            n = _spec(replace(_RIM, rve_pad=v, **_PAD)).kernel_name().split("_")
            self.assertIn(f"rvp{v}", n)
        n = _spec(replace(_RIM, sk_xcd_local=True, **_SK)).kernel_name().split("_")
        self.assertIn("skx", n)


class TestValidity(unittest.TestCase):
    def test_rve_pad_bodies_valid(self):
        for v in (1, 2):
            for extra in (
                _PAD,
                dict(_PAD, pad_m=True),
                dict(_PAD, pad_fast_tile=False),
            ):
                for dt in ("fp16", "bf16"):
                    ok, why = is_valid_spec(
                        _spec(replace(_RIM, rve_pad=v, **extra), dt), "gfx950"
                    )
                    self.assertTrue(ok, why)

    def test_rve_still_rejects_pad_without_rve_pad(self):
        ok, why = is_valid_spec(_spec(replace(_RIM, **_PAD)), "gfx950")
        self.assertFalse(ok)
        self.assertIn("rve_pad", why)

    def test_rve_pad_requirements(self):
        bad = (
            replace(_RIM, rve_pad=2),  # no pad
            replace(_RIM, rve_pad=2, pad_n=True),  # no exact DTL bounds
            replace(_RIM, rve_pad=3, **_PAD),  # unknown mode
            replace(
                _RIM, rilv_vec_epilogue=False, rve_store_nt=False, rve_pad=1, **_PAD
            ),
        )
        for tr in bad:
            self.assertFalse(is_valid_spec(_spec(tr), "gfx950")[0], tr)

    def test_sk_xcd_local_requirements(self):
        self.assertTrue(
            is_valid_spec(_spec(replace(_RIM, sk_xcd_local=True, **_SK)), "gfx950")[0]
        )
        for tr in (
            replace(_RIM, sk_xcd_local=True),  # no stream-K
            replace(_RIM, sk_xcd_local=True, chiplet_swizzle=False, **_SK),
        ):
            self.assertFalse(is_valid_spec(_spec(tr), "gfx950")[0], tr)


class TestLaunchContract(unittest.TestCase):
    def test_rve_pad_needs_n_multiple_of_8(self):
        s = _spec(replace(_RIM, rve_pad=2, **_PAD))
        self.assertIsNone(launch_shape_error(s, 15360, 13440, 1536))
        self.assertIn("multiple of 8", launch_shape_error(s, 15360, 13444, 1536))
        s1 = _spec(replace(_RIM, rve_pad=1, **_PAD))
        self.assertIn("multiple of 8", launch_shape_error(s1, 15360, 13444, 1536))

    def test_rve_pad2_buffer_range(self):
        s = _spec(replace(_RIM, rve_pad=2, **_PAD))
        # 46336 x 46400 x 2 bytes >= 2^32: the C buffer range does not fit
        self.assertIn("u32 buffer range", launch_shape_error(s, 46336, 46400, 64))
        self.assertIsNone(launch_shape_error(s, 38400, 30240, 3072))

    def test_unpadded_dimension_still_checked(self):
        s = _spec(replace(_RIM, rve_pad=2, **_PAD))  # pad_n only
        self.assertIsNotNone(launch_shape_error(s, 15000, 13440, 1536))


class TestEmission(unittest.TestCase):
    def test_rve_pad2_uses_buffer_stores_no_branches_per_store(self):
        ir = _ir(_spec(replace(_RIM, rve_pad=2, **dict(_PAD, pad_fast_tile=False))))
        self.assertIn("llvm.amdgcn.raw.ptr.buffer.store.v4i32", ir)
        ir1 = _ir(_spec(replace(_RIM, rve_pad=1, **dict(_PAD, pad_fast_tile=False))))
        self.assertNotIn("llvm.amdgcn.raw.ptr.buffer.store.v4i32", ir1)

    def test_sk_xcd_local_keeps_agent_fences_for_cross_xcd(self):
        ir = _ir(_spec(replace(_RIM, sk_xcd_local=True, **_SK)))
        self.assertIn('fence syncscope("agent") release', ir)
        self.assertIn('fence syncscope("agent") acquire', ir)
        self.assertIn('fence syncscope("workgroup") release', ir)
        self.assertIn('fence syncscope("workgroup") acquire', ir)
        base = _ir(_spec(replace(_RIM, **_SK)))
        self.assertNotIn('fence syncscope("workgroup") release', base)


class TestEngineGuard(unittest.TestCase):
    def test_levers_listed_python_only(self):
        keys = dict(_be._PYTHON_ONLY_GEMM_LEVERS)
        self.assertEqual(keys.get("rve_pad"), 0)
        self.assertIs(keys.get("sk_xcd_local"), False)

    def test_converter_refuses_sk_xcd_local_on_a_c_engine_body(self):
        # the swizzled ring body with stream-K is C-mirrored; sk_xcd_local is not
        ring = replace(
            _RIM,
            lds_row_interleave=False,
            rilv_vec_epilogue=False,
            rve_store_nt=False,
            lds_swizzle=True,
        )
        _be.universal_gemm_spec_to_dict(_spec(replace(ring, **_SK)))  # accepted
        with self.assertRaises(NotImplementedError):
            _be.universal_gemm_spec_to_dict(
                _spec(replace(ring, sk_xcd_local=True, **_SK))
            )


if __name__ == "__main__":
    unittest.main()
