# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the MFMA issue-order levers.

``mfma_serpentine`` (order 1) and ``mfma_order`` (the alternative orders 2 and
3) reorder the MFMAs of one K atom inside the hb_sched / sia_sched /
mfma_interleave MFMA streams. Every read / DTL issue / wait / barrier keeps
its MFMA slot index and every accumulator still sums its K atoms in the same
order.

Default-off, IR-identical when spelled out, unique kernel-name flags, rejected
outside their scope, refused by the C-engine converters (Python-engine only).
Also pins which dispatched bodies take the serpentine order and the
admission of the opt-in XCD-local stream-K tail helpers.
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from dataclasses import replace

from rocke.core import backend as _be
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    mfma_order_perm,
    mono_data_spec,
)

_T22 = TileSpec(256, 256, 64, 2, 2, 1, 16, 16, 32)
_BASE = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=False,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
    chiplet_wgm=16,
    chiplet_chunk_size=8,
    acc_agpr=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_pad=32,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
)
# the long-K hb_sched body with its address / skew levers
_X4 = replace(
    _BASE,
    hb_sched=True,
    hb_read_every=2,
    dtl_soff0=True,
    dtl_addr_spread=True,
    hb_dtl_delay=1,
    kloop_unroll=4,
    hb_skew=1,
)
# the sia_sched body and the merged-interleave body
_SIA = replace(_BASE, sia_sched=True, sia_plr=True, sia_dtl_spread=True, sia_acq_gap=12)
_MRG = replace(_BASE, ilv_merge=True)
_ASM = replace(
    _BASE, hb_sched=True, hb_read_every=1, asm_loop="tensile", asm_loop_skew=1
)

# (lever kwargs, kernel-name flag) of every non-default order
_ORDERS = {
    1: (dict(mfma_serpentine=True), "_msp"),
    2: (dict(mfma_order=2), "_mo2"),
    3: (dict(mfma_order=3), "_mo3"),
}


def _spec(tr, dt="bf16"):
    return UniversalGemmSpec(name="t", tile=_T22, trait=tr, data=mono_data_spec(dt))


def _ok(tr):
    return is_valid_spec(_spec(tr), arch="gfx950")


def _ir(tr, dt="bf16"):
    return lower_kernel_to_llvm(
        build_universal_gemm(_spec(tr, dt), arch="gfx950"), arch="gfx950"
    )


def _nir(tr, dt="bf16"):
    """IR with the kernel name (which carries the name flag) replaced."""
    k = build_universal_gemm(_spec(tr, dt), arch="gfx950")
    return lower_kernel_to_llvm(k, arch="gfx950").replace(k.name, "KNAME")


def _calls(ir):
    return Counter(re.findall(r"call [^@]*@([A-Za-z0-9_.]+)\(", ir))


def _order(tr, mo):
    return replace(tr, **_ORDERS[mo][0])


class PermTest(unittest.TestCase):
    def test_every_mode_is_a_permutation(self):
        for mo in (0, 1, 2, 3):
            for m, n in ((8, 8), (4, 8), (8, 4), (2, 6), (1, 1)):
                even, odd = mfma_order_perm(mo, m, n)
                self.assertEqual(sorted(even), list(range(m * n)), (mo, m, n))
                self.assertEqual(sorted(odd), list(range(m * n)), (mo, m, n))

    def test_orders(self):
        e, o = mfma_order_perm(0, 2, 3)
        self.assertEqual((e, o), ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        e, o = mfma_order_perm(1, 2, 3)
        self.assertEqual((e, o), ([0, 1, 2, 5, 4, 3], [0, 1, 2, 5, 4, 3]))
        e, o = mfma_order_perm(2, 2, 3)
        self.assertEqual((e, o), ([0, 1, 2, 5, 4, 3], [3, 4, 5, 2, 1, 0]))
        e, o = mfma_order_perm(3, 2, 3)
        self.assertEqual((e, o), ([0, 3, 4, 1, 2, 5], [0, 3, 4, 1, 2, 5]))

    def test_one_operand_change_per_step(self):
        # serpentine orders: consecutive MFMAs of an atom share A or B
        for mo in (1, 3):
            even, _ = mfma_order_perm(mo, 8, 8)
            for a, b in zip(even, even[1:]):
                self.assertTrue(a // 8 == b // 8 or a % 8 == b % 8, (mo, a, b))
        # mode 2: the atom boundary repeats the accumulator
        even, odd = mfma_order_perm(2, 8, 8)
        self.assertEqual(even[-1], odd[0])
        self.assertEqual(odd[-1], even[0])


class MfmaOrderTest(unittest.TestCase):
    def test_default_off_and_spelled_out_identical(self):
        self.assertEqual(TraitSpec().mfma_order, 0)
        self.assertFalse(TraitSpec().mfma_serpentine)
        for tr in (_X4, _SIA, _MRG):
            self.assertEqual(
                _ir(tr), _ir(replace(tr, mfma_order=0, mfma_serpentine=False))
            )

    def test_name_flag(self):
        n0 = build_universal_gemm(_spec(_X4), arch="gfx950").name
        for _kw, flag in _ORDERS.values():
            self.assertNotIn(flag, n0)
        for mo, (_kw, flag) in _ORDERS.items():
            n = build_universal_gemm(_spec(_order(_X4, mo)), arch="gfx950").name
            self.assertIn(flag, n)

    def test_scope(self):
        for tr in (_X4, _SIA):
            for mo in (1, 2, 3):
                ok, why = _ok(_order(tr, mo))
                self.assertTrue(ok, why)
        for mo in (1, 3):
            self.assertTrue(_ok(_order(_MRG, mo))[0], mo)
        self.assertTrue(_ok(_ASM)[0], _ok(_ASM)[1])
        self.assertTrue(_ok(_order(_ASM, 1))[0])  # serpentine has an asm_loop form
        for bad in (
            replace(_X4, mfma_order=1),  # order 1 is spelled mfma_serpentine
            replace(_X4, mfma_order=4),
            replace(_X4, mfma_order=-1),
            replace(_X4, mfma_order=2, mfma_serpentine=True),
            replace(_MRG, mfma_order=2),  # no K-atom parity in the per-atom emitters
            replace(_ASM, mfma_order=3),
            replace(_BASE, mfma_interleave=False, mfma_serpentine=True),
            replace(_BASE, mfma_interleave=False, mfma_order=3),
        ):
            ok, _why = _ok(bad)
            self.assertFalse(ok, bad)

    def test_emission_reorders_only(self):
        for tr, modes in ((_X4, (1, 2, 3)), (_SIA, (1, 2, 3)), (_MRG, (1, 3))):
            ir0 = _nir(tr)
            for mo in modes:
                ir = _nir(_order(tr, mo))
                self.assertNotEqual(ir0, ir, (tr is _X4, mo))
                # same calls (MFMAs, reads, DTL issues, waits, barriers), same
                # number of IR lines: a pure reorder of the MFMA stream
                self.assertEqual(_calls(ir0), _calls(ir), mo)
                self.assertEqual(len(ir0.splitlines()), len(ir.splitlines()), mo)
        # mode 2 reverses the odd K atoms: differs from mode 1 on hb / sia
        for tr in (_X4, _SIA):
            self.assertNotEqual(_nir(_order(tr, 1)), _nir(_order(tr, 2)))


class DispatchTest(unittest.TestCase):
    def test_row_interleave_bodies_dispatch_serpentine(self):
        # every gfx950 row-interleave class (aligned hb_sched, persistent,
        # short-K merged interleave, pad) runs the serpentine order; the
        # small-tile / ring bodies and gfx942 do not
        from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
        from rocke.dispatch.gemm import GemmRequest
        from rocke.dispatch.gemm.common import RILV_SERPENTINE

        self.assertTrue(RILV_SERPENTINE)
        cases = (
            ((8192, 8192, 8192), True),
            ((23040, 24576, 12288), True),
            ((15360, 15360, 768), True),
            ((8192, 8192, 512), True),
            ((4096, 4096, 512), True),  # short-K merged interleave
            ((30720, 20160, 7680), True),
            ((7680, 6720, 1536), True),
            ((4608, 4608, 384), False),  # small-tile body
            ((4608, 3840, 576), False),  # small ring
        )
        for fn, dt in ((dispatch_gemm_fp16, "fp16"), (dispatch_gemm_bf16, "bf16")):
            for (M, N, K), serp in cases:
                with self.subTest(dtype=dt, shape=(M, N, K)):
                    r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
                    self.assertEqual(r.spec.trait.mfma_serpentine, serp)
                    self.assertEqual(r.spec.trait.mfma_order, 0)
                    self.assertTrue(is_valid_spec(r.spec, arch="gfx950")[0])
            r = fn(GemmRequest(M=8192, N=8192, K=8192, arch="gfx942", dtype=dt))
            self.assertFalse(r.spec.trait.mfma_serpentine)
            self.assertEqual(r.spec.trait.mfma_order, 0)


class StreamKTailTest(unittest.TestCase):
    def test_admission_boundary(self):
        from rocke.dispatch.gemm.common import rilv_streamk_tail_admits as adm

        cases = (
            ((12288, 17920, 12288), True),  # 13.1 waves, 32-tile last wave
            ((12288, 17920, 16384), True),
            ((16640, 16384, 12288), True),  # 64-tile last wave (fill 0.25)
            ((10240, 13312, 12288), False),  # 8.1 waves
            ((12288, 17920, 8192), False),  # K below the minimum
            ((23040, 24576, 12288), False),  # last wave 0.75 full
            ((16384, 16384, 12288), False),  # whole waves
            ((12288, 17936, 12288), False),  # ragged N
        )
        for (M, N, K), want in cases:
            self.assertEqual(adm(M, N, K, "gfx950"), want, (M, N, K))
        self.assertFalse(adm(12288, 17920, 12288, "gfx942"))

    def test_trait_valid_and_not_dispatched(self):
        from rocke.dispatch import dispatch_gemm_bf16
        from rocke.dispatch.gemm import GemmRequest
        from rocke.dispatch.gemm.common import rilv_streamk_tail_trait

        r = dispatch_gemm_bf16(
            GemmRequest(M=12288, N=17920, K=12288, arch="gfx950", dtype="bf16")
        )
        self.assertFalse(r.spec.trait.streamk)  # opt-in helpers only
        tr = rilv_streamk_tail_trait(r.spec.trait, "gfx950")
        self.assertTrue(tr.streamk and tr.sk_xcd_local)
        self.assertEqual((tr.persistent_ctas, tr.kloop_unroll, tr.hb_skew), (256, 1, 0))
        self.assertTrue(tr.mfma_serpentine)
        ok, why = is_valid_spec(replace(r.spec, trait=tr), arch="gfx950")
        self.assertTrue(ok, why)


class PythonOnlyTest(unittest.TestCase):
    def test_engine_refuses_mfma_order(self):
        self.assertIn(("mfma_order", 0), _be._PYTHON_ONLY_GEMM_LEVERS)
        self.assertIn(("mfma_serpentine", False), _be._PYTHON_ONLY_GEMM_LEVERS)
        with self.assertRaises(NotImplementedError) as cm:
            _be._refuse_python_only_levers(replace(TraitSpec(), mfma_order=3))
        self.assertIn("mfma_order", str(cm.exception))
        _be._refuse_python_only_levers(TraitSpec())  # the default passes


if __name__ == "__main__":
    unittest.main()
