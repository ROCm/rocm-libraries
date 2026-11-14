# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the Toeplitz WMMA depthwise wgrad (``conv_direct_dw_wgrad_wmma``).

The spec validator's reasons, the n_split clamp, the per-axis padding the
cache's problem gate takes for this variant only, and the cross-engine
agreement of the validator, kernel name, grid, LDS size and serialized IR
(the ``rocke_library_engine`` binding; skipped when it is not importable). The GPU
numerics live in ``test_direct_conv_correctness.py``.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_dw_wgrad_wmma.py -v
"""

from __future__ import annotations

import unittest

import pytest

from kernels.common.conv_direct_dw_wgrad_wmma import (
    DwWgradWmmaSpec,
    build_dw_wgrad_wmma,
    dw_wgrad_wmma_reason,
    is_valid_dw_wgrad_wmma_spec,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

VARIANT = "direct_depthwise_wgrad_wmma"


def _problem(N=2, H=20, W=37, C=64, KH=7, KW=None, PAD=3, dtype="bf16", stride=1, **kw):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C,
        cpg=kw.pop("cpg", 1),
        kpg=kw.pop("kpg", 1),
        KH=KH,
        KW=KH if KW is None else KW,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        **kw,
    )


def _reason(spec, arch):
    ok, why = is_valid_dw_wgrad_wmma_spec(spec, arch)
    return None if ok else why


# (problem kwargs, spec knobs, Python's reason) of specs every arch refuses.
_REJECTED = (
    (dict(dtype="fp32"), {}, "unsupported dtype 'fp32'; expected fp16 or bf16"),
    (dict(C=32, cpg=2, kpg=2), {}, "requires cpg=kpg=1 (got 2, 2)"),
    (dict(stride=2), {}, "requires stride 1 (got 2)"),
    (
        dict(C=12),
        {},
        "channels 12 must be a multiple of 8 (16-byte NHWC reads)",
    ),
    (dict(KH=0, KW=3, PAD=0), {}, "filter 0x3 is not positive"),
    (
        dict(pad_w=-2),
        {},
        "pad_w must be -1 (PAD) or a column padding of at least 0 (got -2)",
    ),
    (dict(PAD=-1), {}, "degenerate geometry"),
    (dict(H=4, PAD=0), {}, "degenerate geometry"),
    (dict(dil_h=2, PAD=6), {}, "this direct conv kernel has no dilation (got 2x1)"),
    ({}, dict(wave_size=64), "the WMMA 16x16x32 lane maps are wave32"),
    ({}, dict(waves=0), "waves must be 1..16 (got 0)"),
    ({}, dict(waves=17), "waves must be 1..16 (got 17)"),
    ({}, dict(tile_q=6), "tile_q must be a positive multiple of 4 (got 6)"),
    ({}, dict(tile_q=0), "tile_q must be a positive multiple of 4 (got 0)"),
    ({}, dict(rows=3), "rows must be a positive even number (got 3)"),
    ({}, dict(rows=0), "rows must be a positive even number (got 0)"),
    ({}, dict(target_blocks=0), "target_blocks and ws_replicas must be >= 1"),
    ({}, dict(ws_replicas=0), "target_blocks and ws_replicas must be >= 1"),
    (
        dict(KW=31, pad_w=15, H=40, W=80),
        dict(rows=16, tile_q=64),
        "2048 unrolled WMMAs per item exceed 1024 (smaller rows / tile_q)",
    ),
    ({}, dict(waves=16, rows=16, tile_q=64), "LDS 703392 B > 327680 on gfx1250"),
)

# Specs both engines build that the AOT grid does not hold.
_EXTRA_OK = (
    (dict(KH=3, PAD=1, dtype="fp16"), {}),
    (dict(KH=5, PAD=2), dict(waves=1, tile_q=4, rows=2, prefetch=False)),
    (dict(KH=1, KW=7, PAD=0, pad_w=3, dtype="fp16"), dict(waves=2, tile_q=12, rows=6)),
    (dict(KH=3, KW=5, PAD=1, pad_w=2, C=24), dict(ws_replicas=1)),
    (dict(C=40, H=13, W=19, dtype="fp16"), dict(waves=16, tile_q=4, rows=2)),
    (dict(KH=2, KW=9, PAD=0), {}),
)


def _aot_specs():
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec

    direction, caps_list = DIRECT_CAPABILITIES[VARIANT]
    assert direction == "wgrad"
    specs = []
    for dtype in ("bf16", "fp16"):
        for caps in caps_list:
            probe = probe_problem(caps, dtype, VARIANT)
            for knobs in _knob_grid(VARIANT, "gfx1250", dtype, caps):
                specs.append(make_direct_spec(VARIANT, probe, knobs, 32))
    return specs


class TestDwWgradWmmaSpec(unittest.TestCase):
    def test_reasons(self):
        for pkw, knobs, want in _REJECTED:
            spec = DwWgradWmmaSpec(problem=_problem(**pkw), **knobs)
            self.assertEqual(_reason(spec, "gfx1250"), want, (pkw, knobs))
        for pkw, knobs in _EXTRA_OK:
            spec = DwWgradWmmaSpec(problem=_problem(**pkw), **knobs)
            self.assertIsNone(_reason(spec, "gfx1250"), (pkw, knobs))
            # The arch-independent rules alone: "" when the spec can be built.
            self.assertEqual(dw_wgrad_wmma_reason(spec), "")

    def test_arch_rules(self):
        spec = DwWgradWmmaSpec(problem=_problem())
        self.assertEqual(is_valid_dw_wgrad_wmma_spec(spec, "gfx1250"), (True, "ok"))
        self.assertEqual(_reason(spec, "gfx950"), "wave32 kernel (got gfx950, wave64)")
        self.assertIn("gfx0", _reason(spec, "gfx0"))

    def test_n_split_clamp(self):
        # grid x = ceil(64 / 32) * ceil(37 / 32) = 4; items = N * ceil(20 / 8).
        spec = DwWgradWmmaSpec(problem=_problem(N=2, PAD=3), target_blocks=1024)
        self.assertEqual(spec.grid(), (4, 6, 1))  # clamped to the 6 items
        spec = DwWgradWmmaSpec(problem=_problem(N=64, PAD=3), target_blocks=16)
        self.assertEqual(spec.grid(), (4, 4, 1))  # ceil(16 / 4)
        spec = DwWgradWmmaSpec(problem=_problem(N=64, PAD=3), target_blocks=1)
        self.assertEqual(spec.grid(), (4, 1, 1))  # at least one split

    def test_aot_grid(self):
        specs = _aot_specs()
        # 7x7 PAD 0 / 3: 2 waves x 3 tile_q x 2 rows x 2 tb x 2 prefetch;
        # 1x31: 2 waves x 3 tile_q x 1 rows x 2 tb x 2 prefetch; two dtypes.
        self.assertEqual(len(specs), 2 * (2 * 48 + 24))
        for spec in specs:
            self.assertIsNone(_reason(spec, "gfx1250"), spec.kernel_name())
        self.assertTrue(any(s.problem.PAD_W != s.problem.PAD for s in specs))

    def test_spec_for_problem_takes_per_axis_padding(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        problem = _problem(KH=1, KW=31, PAD=0, pad_w=15, H=6, W=80)
        knobs = dict(wave_size=32, waves=2, tile_q=16, rows=2, ws_replicas=64)
        spec, why = direct_spec_for_problem(VARIANT, problem, knobs, "gfx1250", 32)
        self.assertIsNotNone(spec, why)
        self.assertEqual(spec.pad_w, 15)
        # Another depthwise wgrad variant refuses a column padding of its own.
        spec, why = direct_spec_for_problem(
            "direct_depthwise_wgrad", problem, {}, "gfx950", 64
        )
        self.assertIsNone(spec)
        self.assertIn("one padding for both axes", why)
        # No variant takes a pad_w below -1.
        spec, why = direct_spec_for_problem(
            VARIANT, _problem(pad_w=-2), knobs, "gfx1250", 32
        )
        self.assertIsNone(spec)
        self.assertIn("pad_w must be -1", why)


class TestDwWgradWmmaEngines(unittest.TestCase):
    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over the AOT grid (the 1x31 per-axis rows included) and the
        rejected and extra specs above, on gfx1250 and gfx950."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_dw_wgrad_wmma_spec_to_dict

        specs = _aot_specs()
        for pkw, knobs, _ in _REJECTED:
            specs.append(DwWgradWmmaSpec(problem=_problem(**pkw), **knobs))
        for pkw, knobs in _EXTRA_OK:
            specs.append(DwWgradWmmaSpec(problem=_problem(**pkw), **knobs))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, why_c, name, grid, lds = eng.conv_direct_dw_wgrad_wmma_describe(
                    conv_direct_dw_wgrad_wmma_spec_to_dict(spec), arch
                )
                why = _reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_c))
                self.assertEqual(why_c, why or "ok", (arch, spec))
                self.assertEqual(name, spec.kernel_name())
                if spec.problem.KH >= 1 and spec.problem.KW >= 1:
                    self.assertEqual(lds, spec.lds_bytes)
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_engines_serialize_the_same_ir(self):
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_dw_wgrad_wmma_spec_to_dict
        from rocke.core.ir_serialize import serialize

        for pkw, knobs in (
            ({}, {}),
            (
                dict(PAD=0, dtype="fp16"),
                dict(waves=8, tile_q=8, rows=4, prefetch=False),
            ),
            (dict(KH=1, KW=31, PAD=0, pad_w=15, H=6, W=80), dict(waves=2, rows=2)),
            (dict(KH=3, KW=5, PAD=1, pad_w=2, C=24), dict(ws_replicas=1)),
        ):
            spec = DwWgradWmmaSpec(problem=_problem(**pkw), **knobs)
            want = serialize(build_dw_wgrad_wmma(spec, "gfx1250"))
            got = eng.conv_direct_dw_wgrad_wmma_serialize_ir(
                conv_direct_dw_wgrad_wmma_spec_to_dict(spec), "gfx1250"
            )
            self.assertEqual(got, want, spec.kernel_name())

    def test_engines_refuse_to_build_the_same_specs(self):
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_dw_wgrad_wmma_spec_to_dict

        for pkw, knobs, why in _REJECTED[:3]:
            spec = DwWgradWmmaSpec(problem=_problem(**pkw), **knobs)
            with self.assertRaises(ValueError) as py_err:
                build_dw_wgrad_wmma(spec, "gfx1250")
            self.assertEqual(
                str(py_err.exception), f"invalid DwWgradWmmaSpec for gfx1250: {why}"
            )
            with self.assertRaises(Exception) as cpp_err:
                eng.conv_direct_dw_wgrad_wmma_serialize_ir(
                    conv_direct_dw_wgrad_wmma_spec_to_dict(spec), "gfx1250"
                )
            self.assertIn(why, str(cpp_err.exception))


if __name__ == "__main__":
    unittest.main()
