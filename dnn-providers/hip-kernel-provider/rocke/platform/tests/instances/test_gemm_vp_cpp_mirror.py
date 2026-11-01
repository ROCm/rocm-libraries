# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ mirror of the register-staged global prefetch body (TraitSpec.vgpr_prefetch,
the K-contiguous RCR form, gemm_build_vp.cpp), its main-loop levers and the split-K
slice / reduction levers (split_k_aligned, split_k_serial; gemm_build_sks.cpp).

* CPU: the levers are off the Python-only list (the converters pass them through) while
  the MN-contiguous vp_mn_* levers stay on it; the generated C++ dispatcher table agrees.
* With the ``rocke_engine`` binding: both engines agree on validity (same reason on a
  reject), the kernel name and the lowered IR (``backend="both"``) of a spread of
  specs. The parity families gemm_vp_* compare the full configuration sets.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from rocke.core import backend
from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
)

_MIRRORED = {
    "vgpr_prefetch": 0,
    "vgpr_prefetch_depth": 1,
    "vp_unroll": 0,
    "vp_late_reads": 0,
    "vp_release_atoms": 1,
    "vp_tail_pack": 0,
    "vp_pair_reads": 0,
    "vp_mfma_first": 0,
    "split_k_aligned": False,
    "split_k_serial": 0,
    "lds_swizzle_l": -1,
}
_STILL_PY = (
    "vp_mn_kv",
    "vp_mn_order",
    "vp_mn_worder",
    "vp_mn_pipe",
    "vp_mn_wrap",
    "mn_vec",
)


def _engine():
    try:
        import rocke_engine  # type: ignore
    except ImportError:
        return None
    return rocke_engine


def _spec(tile=(256, 256, 64, 2, 2), dt="bf16", wt=(16, 16, 16), **kw):
    tm, tn, tk, wm, wn = tile
    t = TileSpec(tm, tn, tk, wm, wn, 1, *wt)
    mode = kw.get("vgpr_prefetch", 4)
    tr = TraitSpec(
        pipeline="compv4",
        epilogue="default",
        ab_double_buffer=mode != 4,
        chiplet_swizzle=True,
        lds_swizzle=True,
        lds_swizzle_r=0,
        acc_agpr=tm * tn // (64 * wm * wn) > 128,
        vgpr_prefetch=mode,
    )
    tr = replace(tr, **kw)
    return UniversalGemmSpec(
        name="vpm", tile=t, trait=tr, data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)
    )


_ML = dict(vp_pair_reads=1, vp_release_atoms=3, vp_tail_pack=3, vp_mfma_first=1)


def _cases():
    out = []
    for mode in (1, 2, 3, 4):
        out.append(_spec(vgpr_prefetch=mode))
        out.append(
            _spec(
                (128, 128, 64, 2, 2), "fp16", vgpr_prefetch=mode, vgpr_prefetch_depth=2
            )
        )
        out.append(
            _spec((64, 64, 128, 2, 2), vgpr_prefetch=mode, pad_m=True, pad_n=True)
        )
        out.append(_spec((128, 64, 64, 4, 2), "fp16", vgpr_prefetch=mode, pad_k=True))
        out.append(
            _spec(
                (256, 128, 64, 4, 2),
                vgpr_prefetch=mode,
                split_k=4,
                pad_k=True,
                split_k_aligned=True,
                split_k_serial=2,
            )
        )
        out.append(
            _spec((128, 128, 64, 2, 2), vgpr_prefetch=mode, split_k=2, split_k_serial=3)
        )
        out.append(_spec((128, 128, 64, 2, 2), wt=(32, 32, 8), vgpr_prefetch=mode))
    out.append(_spec(**_ML))
    out.append(_spec((256, 224, 64, 2, 2), "fp16", vp_unroll=1, **_ML))
    out.append(_spec(vp_late_reads=1))
    out.append(_spec(vp_tail_pack=2, vp_release_atoms=2))
    out.append(_spec(vp_pair_reads=2))
    out.append(
        _spec((128, 128, 64, 2, 2), vgpr_prefetch=3, vp_mfma_first=2, vp_unroll=1)
    )
    out.append(
        _spec((128, 128, 64, 2, 2), vgpr_prefetch=3, vgpr_prefetch_depth=3, vp_unroll=1)
    )
    # refused by both validity gates (same reason)
    out.append(_spec(vgpr_prefetch_depth=2))  # depth > 1 with AGPR accumulators
    out.append(_spec(vp_pair_reads=1, vp_late_reads=1))
    out.append(_spec(vgpr_prefetch=1, ab_double_buffer=False))
    out.append(_spec(split_k=2, split_k_serial=1))
    out.append(_spec(vp_release_atoms=4))
    return out


class TestMirroredLevers(unittest.TestCase):
    def test_off_the_python_only_list(self):
        listed = dict(backend._PYTHON_ONLY_GEMM_LEVERS)
        for k, off in _MIRRORED.items():
            with self.subTest(lever=k):
                self.assertNotIn(k, listed)
                self.assertEqual(getattr(TraitSpec(), k), off)
        for k in _STILL_PY:
            self.assertIn(k, listed)

    def test_converter_passes_them(self):
        s = _spec(
            (256, 128, 64, 4, 2),
            split_k=4,
            pad_k=True,
            split_k_aligned=True,
            split_k_serial=2,
            **_ML,
        )
        d = backend.universal_gemm_spec_to_dict(s)
        for k in _MIRRORED:
            self.assertEqual(d[k], getattr(s.trait, k), k)
        # the wrapper converters carry them too
        self.assertEqual(backend._gemm_lever_keys(s.trait)["vp_tail_pack"], 3)
        # the MN-contiguous loader stays Python-only
        with self.assertRaises(NotImplementedError):
            backend._refuse_python_only_levers(replace(s.trait, vp_mn_kv=8))

    def test_cpp_dispatcher_table(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "cpp"
            / "include"
            / "rocke"
            / "gemm_dispatch_fields.inc"
        )
        if not path.is_file():
            self.skipTest("no C++ engine sources here (the installed test tree)")
        inc = path.read_text()
        for k in _MIRRORED:
            self.assertNotIn(f"ROCKE_GD_PYONLY({k})", inc)
        for k in _STILL_PY:
            self.assertIn(f"ROCKE_GD_PYONLY({k})", inc)


class TestEngineMirror(unittest.TestCase):
    def setUp(self):
        self.eng = _engine()
        if self.eng is None:
            self.skipTest("rocke_engine binding not built")

    def test_same_validity_and_name(self):
        for s in _cases():
            for arch in ("gfx942", "gfx950"):
                with self.subTest(name=s.kernel_name(), arch=arch):
                    py_ok, py_why = is_valid_spec(s, arch)
                    d = backend.universal_gemm_spec_to_dict(s)
                    c_ok, c_why = self.eng.gemm_is_valid(d, arch)
                    self.assertEqual(bool(c_ok), py_ok, (py_why, c_why))
                    if not py_ok:
                        self.assertEqual(c_why, py_why)
                    self.assertEqual(self.eng.gemm_kernel_name(d), s.kernel_name())

    def test_same_ir(self):
        for s in _cases():
            if not is_valid_spec(s, "gfx942")[0]:
                continue
            with self.subTest(name=s.kernel_name()):
                r = backend.lower_universal_gemm(
                    s, arch="gfx942", backend="both", want_ir=True
                )
                self.assertIn(s.kernel_name(), r.llvm_text)

    def test_same_ir_with_the_out_f32_prologue(self):
        for mode in (1, 2, 3, 4):
            s = _spec((128, 128, 64, 2, 2), vgpr_prefetch=mode)
            attach_prologue(s, Prologue(out_f32=True))
            with self.subTest(mode=mode):
                backend.lower_universal_gemm(s, arch="gfx942", backend="both")

    def test_same_ir_with_the_f32_prologue_forms(self):
        # the f32 recipes (split modes, complex B / conj), batched, wide_addr and
        # lds_swizzle_l on the register-staged body
        x6 = dict(src="f32", recipe="x6", out_f32=True, split_at="use", psub=True)
        cases = []
        for mode in (3, 4):
            for sm in ("rne", "trunc", "rnef"):
                for cx in (
                    {},
                    dict(cplx_b=True),
                    dict(cplx_b=True, conj_a=True, conj_b=True),
                ):
                    cases.append(
                        (dict(vgpr_prefetch=mode), dict(x6, split_mode=sm, **cx), False)
                    )
        cases.append((dict(lds_swizzle_l=3), x6, False))
        cases.append((dict(wide_addr=True), dict(x6, cplx_b=True), True))
        cases.append((dict(wide_addr=True), dict(out_f32=True), False))
        cases.append(
            (dict(_ML, _tile=(64, 64, 128, 2, 2)), dict(x6, recipe="x3"), True)
        )
        for kw, pro, batched in cases:
            kw = dict(kw)
            s = _spec(kw.pop("_tile", (128, 64, 64, 4, 2)), **kw)
            s = replace(s, batched=batched)
            attach_prologue(s, Prologue(**pro))
            with self.subTest(kw=kw, pro=pro, batched=batched):
                self.assertTrue(is_valid_spec(s, "gfx942")[0])
                d = backend.universal_gemm_spec_to_dict(s)
                self.assertEqual(self.eng.gemm_kernel_name(d), s.kernel_name())
                backend.lower_universal_gemm(s, arch="gfx942", backend="both")


if __name__ == "__main__":
    unittest.main()
