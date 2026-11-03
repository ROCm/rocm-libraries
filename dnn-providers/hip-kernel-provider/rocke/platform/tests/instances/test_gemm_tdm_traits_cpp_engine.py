# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250 TDM K-loop traits through the C++ engine's Python seam.

``split_barrier``, ``tdm_alternate_issuers`` and ``tdm_early_refill`` -- and
``tdm`` / ``tdm_depth`` beneath them -- must reach the C++ engine through
``universal_gemm_spec_to_dict`` and the ``rocke_engine`` binding. A key that
either side drops does not fail: the engine fills in the default and lowers a
different kernel (a non-TDM one, when ``tdm`` itself is dropped) for the cpp
backend without a word. The byte-identity gate drives the C struct directly, so
it cannot see this seam; these tests do.

The first class needs nothing; the second needs the ``rocke_engine`` binding
and is skipped when it is not importable.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

# pytest.ini uses --import-mode=importlib, which does not put this directory on
# sys.path: without the insert the sibling import below only resolves when an
# earlier-collected module happened to add it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_gemm_split_barrier import _spec  # noqa: E402

ARCH = "gfx1250"

# Admitted configurations of the three traits on the shared probe shape.
_ARMS = {
    "tdm_d2": {},
    "split_barrier": dict(split_barrier=True),
    "alternate_issuers": dict(tdm_alternate_issuers=True),
    "split_alternate": dict(split_barrier=True, tdm_alternate_issuers=True),
    "early_refill": dict(
        split_barrier=True, tdm_alternate_issuers=True, tdm_early_refill=True
    ),
}

# Rejected configurations, one per gate; each must fail for the same reason in
# both engines (an admitted one waits on a barrier nothing signals: a hang).
# Split barrier with ``wmma_v1`` is Python-only: the C++ engine rejects every
# ``wmma_v1`` spec on its pipeline (test_gemm_split_barrier covers Python).
_REJECTS = {
    "split, depth 1": dict(split_barrier=True, tdm_depth=1),
    "alternate, depth 3": dict(tdm_alternate_issuers=True, tdm_depth=3),
    "early refill, no alternate": dict(tdm_early_refill=True, split_barrier=True),
    "early refill, no split": dict(tdm_early_refill=True, tdm_alternate_issuers=True),
}


class TestGemmTdmTraitsSpecDict(unittest.TestCase):
    def test_spec_dict_carries_every_tdm_trait(self):
        from rocke.core.backend import universal_gemm_spec_to_dict

        spec = _spec(
            tdm_depth=2,
            split_barrier=True,
            tdm_alternate_issuers=True,
            tdm_early_refill=True,
        )
        d = universal_gemm_spec_to_dict(spec)
        for key in (
            "tdm",
            "tdm_depth",
            "split_barrier",
            "tdm_alternate_issuers",
            "tdm_early_refill",
        ):
            with self.subTest(key):
                self.assertIn(key, d)
                self.assertEqual(d[key], getattr(spec.trait, key))


class TestGemmTdmTraitsKernelName(unittest.TestCase):
    """Kernels that differ only in a K-loop knob must not share a name."""

    def test_every_arm_has_its_own_name(self):
        names = {label: _spec(**kw).kernel_name() for label, kw in _ARMS.items()}
        self.assertEqual(len(set(names.values())), len(names), names)

    def test_tags_name_only_the_knobs_that_are_on(self):
        base = _spec().kernel_name()
        self.assertIn("_tdm2", base)  # the probe spec runs TDM at depth 2
        for tag in ("_spb", "_alti", "_erf"):
            self.assertNotIn(tag, base)
        er = _spec(**_ARMS["early_refill"]).kernel_name()
        for tag in ("_tdm2", "_spb", "_alti", "_erf"):
            self.assertIn(tag, er)

    def test_tdm_depth_is_in_the_name(self):
        self.assertNotEqual(
            _spec(tdm_depth=2).kernel_name(), _spec(tdm_depth=3).kernel_name()
        )

    def test_a_kernel_without_tdm_keeps_its_name(self):
        from dataclasses import replace

        spec = _spec()
        off = replace(spec, trait=replace(spec.trait, tdm=False))
        self.assertNotIn("tdm", off.kernel_name().replace(spec.name, ""))

    def test_c_dtype_is_named_only_when_it_differs_from_a(self):
        spec = _spec()  # fp8 A/B, bf16 C
        fp16_c = replace(spec, data=replace(spec.data, dtype_c="fp16"))
        self.assertTrue(spec.kernel_name().endswith("_cbf16"), spec.kernel_name())
        self.assertNotEqual(spec.kernel_name(), fp16_c.kernel_name())
        bf16 = replace(spec, data=replace(spec.data, dtype_a="bf16", dtype_b="bf16"))
        self.assertNotIn("_cbf16", bf16.kernel_name())


class TestGemmTdmTraitsCppEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rocke_engine  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"rocke_engine C++ binding not importable: {e}")

    def test_both_engines_emit_identical_kernels(self):
        from rocke.core.backend import lower_universal_gemm
        from rocke.core.lower_llvm import _resolve_llvm_flavor

        # lower_universal_gemm lowers at the environment's flavor, and both
        # engines refuse TDM below llvm23.
        flavor = _resolve_llvm_flavor()
        if flavor in ("llvm20", "llvm22"):
            self.skipTest(f"TDM kernels need an llvm23+ flavor, got {flavor}")

        for label, kw in _ARMS.items():
            with self.subTest(label):
                # Raises BackendMismatch on any .ll or serialized-IR difference.
                lower_universal_gemm(
                    _spec(**kw), arch=ARCH, backend="both", want_ir=True
                )
        with self.subTest("batched"):  # the i64 batch offset of the TDM fills
            spec = replace(_spec(), batched=True)
            lower_universal_gemm(spec, arch=ARCH, backend="both", want_ir=True)
        with self.subTest("fp16 C"):  # the C-dtype tag in the kernel name
            spec = _spec()
            spec = replace(spec, data=replace(spec.data, dtype_c="fp16"))
            lower_universal_gemm(spec, arch=ARCH, backend="both", want_ir=True)

    def test_both_engines_apply_the_same_storage_rule(self):
        """One fp8/bf8 storage rule per engine: same verdict and same reason."""
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _spec()
        cases = {
            "fp8 x fp8 -> bf16": ("fp8e4m3", "fp8e4m3", "bf16", True),
            "fp8 x fp8 -> fp16": ("fp8e4m3", "fp8e4m3", "fp16", True),
            "fp8 x bf8 (mixed)": ("fp8e4m3", "bf8e5m2", "bf16", False),
            "fp8 x fp8 -> fp32 C": ("fp8e4m3", "fp8e4m3", "fp32", False),
            "fp8 x fp8 -> fp8 C": ("fp8e4m3", "fp8e4m3", "fp8e4m3", False),
            "bf16 x fp8 (one side fp8)": ("bf16", "fp8e4m3", "bf16", False),
        }
        for label, (a, bb, c, admitted) in cases.items():
            with self.subTest(label):
                spec = replace(
                    base, data=replace(base.data, dtype_a=a, dtype_b=bb, dtype_c=c)
                )
                py = is_valid_spec(spec, arch=ARCH)
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), ARCH
                )
                self.assertEqual(tuple(cpp), tuple(py))
                if not admitted:
                    self.assertFalse(py[0], f"{label} admitted")

    def test_both_engines_bound_ab_load_elem_bytes(self):
        """Out-of-range widths, and any width under TDM, fail alike."""
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _spec()  # TDM
        no_tdm = replace(base, trait=replace(base.trait, tdm=False, tdm_depth=1))
        # Both bases are valid on their own, so a reject below is the width's.
        for ok_spec in (base, no_tdm):
            self.assertTrue(is_valid_spec(ok_spec, arch=ARCH)[0])
        cases = {
            "0": (no_tdm, 0),
            "-1": (no_tdm, -1),
            "32": (no_tdm, 32),
            "2 with tdm": (base, 2),
        }
        for label, (spec0, width) in cases.items():
            with self.subTest(label):
                spec = replace(
                    spec0, trait=replace(spec0.trait, ab_load_elem_bytes=width)
                )
                py = is_valid_spec(spec, arch=ARCH)
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), ARCH
                )
                self.assertFalse(py[0], label)
                self.assertIn("ab_load_elem_bytes", py[1])
                self.assertEqual(tuple(cpp), tuple(py))

    def test_cpp_rejects_wmma_v1(self):
        """The C++ engine does not emit the wmma_v1 schedule, so it must not
        build a kernel under that pipeline's name; Python keeps it."""
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _spec()
        spec = replace(base, trait=replace(base.trait, pipeline="wmma_v1"))
        self.assertTrue(is_valid_spec(spec, arch=ARCH)[0])
        ok, why = rocke_engine.gemm_is_valid(universal_gemm_spec_to_dict(spec), ARCH)
        self.assertFalse(ok)
        self.assertIn("'mem' pipeline", why)

    def test_both_engines_agree_on_target_scope(self):
        """The summed cshuffle budget on CDNA and the lds_swizzle reject on RDNA
        WMMA: both engines agree, verdict and reason."""
        import rocke_engine

        from test_gemm_validity_gaps import CDNA_CSHUFFLE, RDNA_WMMA, _bf16_spec

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, spec, arch in (
            (
                "gfx950 cshuffle",
                _bf16_spec(CDNA_CSHUFFLE, epilogue="cshuffle"),
                "gfx950",
            ),
            ("gfx1151 lds_swizzle", _bf16_spec(RDNA_WMMA, lds_swizzle=True), "gfx1151"),
        ):
            with self.subTest(label):
                py = is_valid_spec(spec, arch=arch)
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), arch
                )
                self.assertEqual(tuple(cpp), tuple(py))

    def test_both_engines_reject_for_the_same_reason(self):
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _spec()
        for label, kw in _REJECTS.items():
            with self.subTest(label):
                spec = replace(base, trait=replace(base.trait, **kw))
                py = is_valid_spec(spec, arch=ARCH)
                self.assertFalse(py[0], f"{label} admitted")
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), ARCH
                )
                self.assertEqual(tuple(cpp), tuple(py))


if __name__ == "__main__":
    unittest.main()
