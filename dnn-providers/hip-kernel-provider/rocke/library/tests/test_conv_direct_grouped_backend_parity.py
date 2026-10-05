# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dual-engine parity of the public ``lower_conv_direct_grouped`` entry point.

``rocke.core.backend.lower_conv_direct_grouped`` flattens a
``DirectConv16cSpec`` / ``DirectConv4cSpec`` into a dict
(``conv_direct_grouped_spec_to_dict``) for the C++ binding. Every spec field
that changes the emitted kernel must survive that flattening -- the problem
``dtype`` and the 4c fused-dgrad knobs (``dgrad_fused_weights``,
``dgrad_weights_lds``) included -- or the default (cpp) backend silently
returns a different kernel than the Python engine. The byte-identity gate
covers the ``_emit.c`` parity pair, not this binding path, so this test runs
``backend="both"`` (python vs cpp IR + ``.ll`` equality) over those specs.

CPU only (no kernel launch). Skips when the ``rocke_engine`` extension is not
importable.

Run:
  PYTHONPATH=<engine build>/cpp/bindings:rocke/platform/python:rocke/library \\
      <python> -m pytest rocke/library/tests/test_conv_direct_grouped_backend_parity.py -v
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.core.backend import (
    conv_direct_grouped_spec_to_dict,
    lower_conv_direct_grouped,
)

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    is_valid_spec_4c,
    make_dgrad_4c_spec,
)

_HAS_ENGINE = importlib.util.find_spec("rocke_engine") is not None
_SKIP = "" if _HAS_ENGINE else "rocke_engine extension not importable"


def _p4(dtype, N=2, H=13, W=17, groups=32, K=3):
    return DirectConvProblem(
        N=N, H=H, W=W, groups=groups, cpg=4, kpg=4, KH=K, KW=K,
        PAD=(K - 1) // 2, dtype=dtype,
    )  # fmt: skip


def _4c_cases():
    """(label, spec, arch) for the 4c forms this path must carry."""
    out = []
    for dtype in ("fp16", "bf16"):
        out.append(
            (f"4c_fprop_{dtype}", DirectConv4cSpec(problem=_p4(dtype)), "gfx950")
        )
        out.append(
            (
                f"4c_fprop_{dtype}_1x1",
                DirectConv4cSpec(problem=_p4(dtype, H=7, W=15, K=1)),
                "gfx942",
            )
        )
        for fused, lds in ((False, False), (True, False), (True, True)):
            spec = make_dgrad_4c_spec(
                _p4(dtype),
                dgrad_fused_weights=fused,
                dgrad_weights_lds=lds,
            )
            out.append(
                (f"4c_dgrad_{dtype}_fw{int(fused)}_lds{int(lds)}", spec, "gfx950")
            )
        # Fused without LDS staging is also legal on gfx942.
        spec = make_dgrad_4c_spec(_p4(dtype, N=1, H=7, W=13), dgrad_fused_weights=True)
        out.append((f"4c_dgrad_{dtype}_fw1_gfx942", spec, "gfx942"))
    return out


def _16c_cases():
    out = []
    for dtype in ("fp16", "bf16"):
        for fold in (False, True):
            p = DirectConvProblem(
                N=1, H=15, W=17, groups=8, cpg=16, kpg=16, dtype=dtype
            )
            out.append(
                (
                    f"16c_{dtype}_k32{int(fold)}",
                    DirectConv16cSpec(problem=p, fold_k32=fold),
                    "gfx950",
                )
            )
    return out


class TestSpecToDict(unittest.TestCase):
    """CPU-only: the flattened dict carries every kernel-shaping field."""

    def test_4c_fields_forwarded(self):
        spec = make_dgrad_4c_spec(
            _p4("bf16"), dgrad_fused_weights=True, dgrad_weights_lds=True
        )
        d = conv_direct_grouped_spec_to_dict(spec, "4c")
        self.assertEqual(d["problem"]["dtype"], "bf16")
        self.assertIs(d["dgrad_fused_weights"], True)
        self.assertIs(d["dgrad_weights_lds"], True)
        self.assertNotIn("fold_k32", d)

    def test_16c_fields_forwarded(self):
        p = DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16, dtype="bf16")
        d = conv_direct_grouped_spec_to_dict(
            DirectConv16cSpec(problem=p, fold_k32=False), "16c"
        )
        self.assertEqual(d["problem"]["dtype"], "bf16")
        self.assertIs(d["fold_k32"], False)
        self.assertNotIn("dgrad_fused_weights", d)


@unittest.skipIf(bool(_SKIP), _SKIP)
class TestBackendBoth(unittest.TestCase):
    """python vs cpp through the public backend entry (IR + .ll equality)."""

    def _check(self, cases, kind):
        for label, spec, arch in cases:
            with self.subTest(case=label, arch=arch):
                ok, why = (
                    is_valid_spec_4c(spec, arch=arch) if kind == "4c" else (True, "")
                )
                self.assertTrue(ok, why)
                r = lower_conv_direct_grouped(
                    spec, kind=kind, arch=arch, backend="both", want_ir=True
                )
                # Python-side name must appear in the cpp-checked .ll.
                self.assertIn(spec.kernel_name(), r.llvm_text)
                c = lower_conv_direct_grouped(spec, kind=kind, arch=arch, backend="cpp")
                self.assertEqual(c.llvm_text, r.llvm_text)

    def test_4c(self):
        self._check(_4c_cases(), "4c")

    def test_16c(self):
        self._check(_16c_cases(), "16c")


@unittest.skipIf(bool(_SKIP), _SKIP)
class Test4cTapBound(unittest.TestCase):
    """KH*KW > 16 is rejected with the same reason by both engines (the C++
    builder's per-tap arrays are bounded)."""

    def test_5x5_rejected_both_engines(self):
        spec = DirectConv4cSpec(problem=_p4("fp16", K=5))
        ok, why = is_valid_spec_4c(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("KH*KW <= 16", why)
        for backend in ("python", "cpp"):
            with self.subTest(backend=backend):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="4c", arch="gfx950", backend=backend
                    )
                self.assertIn("KH*KW <= 16", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
