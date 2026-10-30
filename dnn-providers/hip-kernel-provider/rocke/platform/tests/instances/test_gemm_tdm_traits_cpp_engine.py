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
_REJECTS = {
    "split, depth 1": dict(split_barrier=True, tdm_depth=1),
    "split, wmma_v1": dict(split_barrier=True, pipeline="wmma_v1"),
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
