# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dual-engine parity of the public ``lower_conv_direct_grouped`` entry point.

``rocke.core.backend.lower_conv_direct_grouped`` flattens a
``DirectConv16cSpec`` / ``DirectConv4cSpec`` into a dict
(``conv_direct_grouped_spec_to_dict``) for the C++ binding. Every spec field
that changes the emitted kernel must survive that flattening -- the problem
``dtype`` and the 4c dgrad knobs (``dgrad_fused_weights``,
``dgrad_weights_lds``, ``stage_rows``, ``waves_q``) included -- or the C++
backend silently returns a different kernel
than the Python engine. The byte-identity gate covers the ``_emit.c`` parity
pair, not this binding path, so this test lowers those specs through both
backends and compares the ``.ll``.

CPU only (no kernel launch). The binding half skips when the ``rocke_engine``
extension is not importable.

Run:
  PYTHONPATH=<engine build>/cpp/bindings:rocke/platform/python:rocke/library \\
      <python> -m pytest rocke/library/tests/test_conv_direct_grouped_backend_parity.py -v
"""

from __future__ import annotations

import dataclasses
import importlib.util
import unittest

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    is_valid_spec_4c,
    make_dgrad_4c_spec,
)
from rocke.core.backend import (
    conv_direct_grouped_spec_to_dict,
    lower_conv_direct_grouped,
)

_HAS_ENGINE = importlib.util.find_spec("rocke_engine") is not None
_SKIP = "" if _HAS_ENGINE else "rocke_engine extension not importable"


def _p4(dtype, N=2, H=13, W=17, groups=32, K=3):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=4,
        kpg=4,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2,
        dtype=dtype,
    )


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
                DirectConv4cSpec(problem=_p4(dtype, H=7, W=15, K=1), block_q=8),
                "gfx942",
            )
        )
        for fused, lds in ((False, False), (True, False), (True, True)):
            spec = make_dgrad_4c_spec(
                _p4(dtype), dgrad_fused_weights=fused, dgrad_weights_lds=lds
            )
            out.append(
                (f"4c_dgrad_{dtype}_fw{int(fused)}_lds{int(lds)}", spec, "gfx950")
            )
        # Fused without LDS staging is also legal on gfx942.
        spec = make_dgrad_4c_spec(_p4(dtype, N=1, H=7, W=13), dgrad_fused_weights=True)
        out.append((f"4c_dgrad_{dtype}_fw1_gfx942", spec, "gfx942"))
        # Row-staged form (stage_rows, waves_q = block_q / 4), 3x3 and 1x1.
        for bq, bg, K in ((4, 16, 3), (8, 32, 3), (4, 32, 1)):
            spec = make_dgrad_4c_spec(
                _p4(dtype, K=K),
                block_q=bq,
                block_groups=bg,
                dgrad_fused_weights=True,
                dgrad_weights_lds=True,
                stage_rows=True,
            )
            out.append((f"4c_dgrad_{dtype}_sr_bq{bq}_bg{bg}_k{K}", spec, "gfx950"))
    return out


class TestSpecToDict(unittest.TestCase):
    """CPU-only: the flattened dict carries every kernel-shaping field."""

    def test_4c_dgrad_fields_forwarded(self):
        spec = make_dgrad_4c_spec(
            _p4("bf16"),
            block_q=8,
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
            stage_rows=True,
        )
        d = conv_direct_grouped_spec_to_dict(spec, "4c")
        self.assertEqual(d["problem"]["dtype"], "bf16")
        self.assertIs(d["dgrad_fused_weights"], True)
        self.assertIs(d["dgrad_weights_lds"], True)
        self.assertIs(d["stage_rows"], True)
        self.assertEqual(d["waves_q"], 2)
        self.assertNotIn("fold_k32", d)

    def test_4c_every_kernel_field_forwarded(self):
        # Every DirectConv4cSpec field except the problem (nested) reaches the
        # binding, and the problem carries its dtype: a field that is not
        # forwarded would silently build the old kernel on the cpp path.
        for _label, spec, _arch in _4c_cases():
            d = conv_direct_grouped_spec_to_dict(spec, "4c")
            self.assertEqual(d["problem"]["dtype"], spec.problem.dtype)
            for f in dataclasses.fields(spec):
                if f.name == "problem":
                    continue
                with self.subTest(field=f.name):
                    self.assertIn(f.name, d)
                    self.assertEqual(d[f.name], getattr(spec, f.name))


@unittest.skipIf(bool(_SKIP), _SKIP)
class TestBackendBoth(unittest.TestCase):
    """python vs cpp through the public backend entry (``.ll`` equality).

    The two engines' serialized IR is not compared: the 4c and 16c loops
    disagree on a ``scf.for`` attribute (``elide_trailing_barrier``) that has
    no effect on a loop without a barrier, and the byte-identity gate is on
    the lowered ``.ll``."""

    def _check(self, cases, kind):
        for label, spec, arch in cases:
            with self.subTest(case=label, arch=arch):
                if kind == "4c":
                    ok, why = is_valid_spec_4c(spec, arch=arch)
                    self.assertTrue(ok, why)
                py = lower_conv_direct_grouped(
                    spec, kind=kind, arch=arch, backend="python"
                )
                self.assertIn(spec.kernel_name(), py.llvm_text)
                cpp = lower_conv_direct_grouped(
                    spec, kind=kind, arch=arch, backend="cpp"
                )
                self.assertEqual(cpp.llvm_text, py.llvm_text)

    def test_4c(self):
        self._check(_4c_cases(), "4c")

    def test_4c_rejects_match(self):
        # The dtype reject reason is identical in both engines (the C++ engine
        # prefixes its lowering context).
        spec = DirectConv4cSpec(problem=_p4("fp32"))
        msgs = []
        for backend in ("python", "cpp"):
            with self.assertRaises(Exception) as cm:
                lower_conv_direct_grouped(
                    spec, kind="4c", arch="gfx950", backend=backend
                )
            msgs.append(str(cm.exception))
        self.assertIn("unsupported dtype", msgs[0])
        self.assertTrue(msgs[1].endswith(msgs[0]), msgs)

    def test_stage_rows_rejects_match(self):
        # The stage_rows validator reasons are identical in both engines.
        p = _p4("bf16")
        base = make_dgrad_4c_spec(
            p, dgrad_fused_weights=True, dgrad_weights_lds=True, stage_rows=True
        )
        R = dataclasses.replace
        bad = [
            R(base, block_q=8),
            R(base, waves_q=0),
            R(base, dgrad_weights_lds=False),
            R(base, stage_rows=False, block_q=8, waves_q=2),
            R(base, problem=R(base.problem, PAD=0)),
            R(base, block_q=68, waves_q=17),
        ]
        for spec in bad:
            msgs = []
            for backend in ("python", "cpp"):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="4c", arch="gfx950", backend=backend
                    )
                msgs.append(str(cm.exception))
            with self.subTest(reason=msgs[0]):
                self.assertTrue(msgs[0].startswith("DirectConv4cSpec "))
                self.assertTrue(msgs[1].endswith(msgs[0]), msgs)

    def test_target_and_tap_rejects_match(self):
        # Target-dependent (no transpose LDS reads on gfx942) and the tap bound
        # (the C++ builder's per-tap arrays): the same reason in both engines.
        cases = [
            (
                make_dgrad_4c_spec(
                    _p4("bf16"), dgrad_fused_weights=True, dgrad_weights_lds=True
                ),
                "gfx942",
                "ds_read_b64_tr_b16",
            ),
            (DirectConv4cSpec(problem=_p4("fp16", K=5)), "gfx950", "KH*KW <= 16"),
        ]
        for spec, arch, needle in cases:
            msgs = []
            for backend in ("python", "cpp"):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="4c", arch=arch, backend=backend
                    )
                msgs.append(str(cm.exception))
            with self.subTest(arch=arch, needle=needle):
                self.assertIn(needle, msgs[0])
                self.assertTrue(msgs[1].endswith(msgs[0]), msgs)


if __name__ == "__main__":
    unittest.main()
