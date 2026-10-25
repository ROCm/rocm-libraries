# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the DTL issue-economy levers.

* ``rilv_pad``: bytes of LDS padding per 1 KiB block of the
  ``lds_row_interleave`` layout (16 = historical; 32 = conflict-free
  ds_read_b128 on gfx950),
* ``dtl_rsrc_word3``: DWORD3 of the DTL buffer resources (``IRBuilder.
  buffer_rsrc(word3=)``; mirrored in both LLVM and HIP lowerings),
* ``dtl_soff0``: per-tile origin in the DTL resource base, inline soffset 0,
* ``kloop_unroll``: staged main loop of ``kloop_unroll * dtl_stages`` tiles,
* ``dtl_addr_spread`` / ``hb_dtl_delay``: hb_sched DTL address math pinned to
  free MFMA slots / first DTL a few slots after the release barrier.

Every lever is default-off (the default trait keeps the historical
emission), has a unique kernel-name flag, is rejected outside its scope, and
is refused by the C-engine converters (Python-engine only).
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
    mono_data_spec,
)

_T22 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_HB = TraitSpec(
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
    dtl_hoist=True,
    lds_row_interleave=True,
    hb_sched=True,
    hb_read_every=2,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
_RM = replace(_HB, hb_sched=False, hb_read_every=1, ilv_merge=True)

_NEW = {
    "rilv_pad": (16, 32),
    "dtl_rsrc_word3": (0, 0x20000),
    "dtl_soff0": (False, True),
    "kloop_unroll": (1, 2),
    "dtl_addr_spread": (False, True),
    "hb_dtl_delay": (0, 1),
}


def _spec(tr, dt="bf16"):
    return UniversalGemmSpec(name="t", tile=_T22, trait=tr, data=mono_data_spec(dt))


def _ir(tr, dt="bf16"):
    return lower_kernel_to_llvm(
        build_universal_gemm(_spec(tr, dt), arch="gfx950"), arch="gfx950"
    )


class DefaultsTest(unittest.TestCase):
    def test_defaults_are_off(self):
        tr = TraitSpec()
        for k, (off, _on) in _NEW.items():
            self.assertEqual(getattr(tr, k), off, k)

    def test_spelled_out_defaults_emit_identical_ir(self):
        for base in (_HB, _RM):
            ref = _ir(base)
            spelled = replace(base, **{k: off for k, (off, _on) in _NEW.items()})
            self.assertEqual(_ir(spelled), ref)

    def test_name_flags_unique(self):
        names = {_spec(_HB).kernel_name()}
        for k, (_off, on) in _NEW.items():
            tr = replace(_HB, **{k: on})
            if k == "dtl_addr_spread":
                tr = replace(tr, dtl_soff0=True)
            ok, why = is_valid_spec(_spec(tr), arch="gfx950")
            self.assertTrue(ok, f"{k}: {why}")
            n = _spec(tr).kernel_name()
            self.assertNotIn(n, names, k)
            names.add(n)


class ValidityTest(unittest.TestCase):
    def _bad(self, tr, frag):
        ok, why = is_valid_spec(_spec(tr), arch="gfx950")
        self.assertFalse(ok)
        self.assertIn(frag, why)

    def test_rilv_pad_scope(self):
        self._bad(replace(_HB, rilv_pad=24), "rilv_pad")
        self._bad(
            replace(
                _HB,
                lds_row_interleave=False,
                rilv_vec_epilogue=False,
                rve_store_nt=False,
                hb_sched=False,
                rilv_pad=32,
            ),
            "rilv_pad",
        )
        for p in (32, 48, 64):
            self.assertTrue(is_valid_spec(_spec(replace(_HB, rilv_pad=p)), "gfx950")[0])

    def test_soff0_scope(self):
        self._bad(replace(_HB, dtl_soff0=True, dtl_hoist=False), "dtl_soff0")
        self._bad(replace(_HB, dtl_soff0=True, dtl_tight_rsrc=True), "dtl_soff0")

    def test_word3_scope(self):
        self._bad(replace(_HB, dtl_rsrc_word3=-1), "dtl_rsrc_word3")
        self._bad(replace(_HB, dtl_rsrc_word3=1 << 32), "dtl_rsrc_word3")

    def test_unroll_scope(self):
        self._bad(replace(_HB, kloop_unroll=3), "kloop_unroll")
        self._bad(replace(_HB, kloop_unroll=2, direct_to_lds=False), "")

    def test_addr_spread_scope(self):
        self._bad(replace(_RM, dtl_addr_spread=True), "dtl_addr_spread")
        self._bad(
            replace(_HB, hb_dtl_delay=1, hb_sched=False, ilv_merge=True), "hb_dtl_delay"
        )
        self._bad(replace(_HB, hb_dtl_delay=17), "hb_dtl_delay")


class EmissionTest(unittest.TestCase):
    def test_rilv_pad_changes_lds_rows(self):
        a, b = _ir(_HB), _ir(replace(_HB, rilv_pad=32))
        self.assertNotEqual(a, b)
        # 32 alloc rows of 8 x 64 bf16 + 16 (32 B) pad per operand stage
        self.assertIn(str(32 * 528), b)
        self.assertIn(str(32 * 520), a)

    def test_word3_reaches_llvm(self):
        b = _ir(replace(_HB, dtl_rsrc_word3=0x20000))
        self.assertIn("i32 131072)", b)
        self.assertIn("i32 159744)", _ir(_HB))

    def test_variants_build(self):
        for tr in (
            replace(_HB, rilv_pad=32, dtl_soff0=True),
            replace(
                _HB, rilv_pad=32, dtl_soff0=True, dtl_addr_spread=True, hb_dtl_delay=1
            ),
            replace(_HB, rilv_pad=32, dtl_soff0=True, kloop_unroll=2),
            replace(_RM, rilv_pad=32, kloop_unroll=2),
            replace(_RM, rilv_pad=32, dtl_soff0=True),
        ):
            ok, why = is_valid_spec(_spec(tr), arch="gfx950")
            self.assertTrue(ok, why)
            self.assertIn("define", _ir(tr))


class EngineGuardTest(unittest.TestCase):
    def test_python_only(self):
        keys = {k for k, _ in _be._PYTHON_ONLY_GEMM_LEVERS}
        for k in _NEW:
            self.assertIn(k, keys)
        for k, (_off, on) in _NEW.items():
            tr = replace(_HB, **{k: on})
            with self.assertRaises(NotImplementedError):
                _be._refuse_python_only_levers(tr)


if __name__ == "__main__":
    unittest.main()
