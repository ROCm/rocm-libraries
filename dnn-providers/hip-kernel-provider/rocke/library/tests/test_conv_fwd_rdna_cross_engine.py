# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-engine checks for the RDNA4 forward implicit-GEMM knobs
(``uniform_k_decode``, ``regt_prefetch``, ``lds_swizzle``,
``group_fast_raster``).

* Host-only: the dict the C++ engine builds from carries every knob, so a
  binding that dropped one would build the plain kernel under the knob's name.
* With ``rocke_library_engine``: each knob, alone and combined, lowers to the
  same ``.ll`` through ``lower_conv_implicit_gemm`` with ``backend="python"``
  and ``backend="cpp"``, the kernel symbol carries the knob's name tag, and the
  specs the Python validator refuses are refused by the C++ engine too.

The byte-level parity of the same knobs is gated by the
``conv_implicit_gemm`` parity pair (``tests/parity/conv_implicit_gemm_emit``).
"""

import dataclasses
import importlib.util
import re
import unittest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec, is_valid_spec

_ARCH = "gfx1201"

# (label, spec overrides, kernel-name tags)
_KNOBS = (
    ("ukd", dict(uniform_k_decode=True), ("ukd",)),
    ("rtp1", dict(regt_prefetch=1), ("rtp1",)),
    (
        "ukd_rtp1_bf16",
        dict(uniform_k_decode=True, regt_prefetch=1, dtype="bf16"),
        ("ukd", "rtp1"),
    ),
    ("lxs", dict(lds_swizzle=True), ("lxs",)),
    ("lxs_tk128", dict(lds_swizzle=True, tile_k=128), ("lxs",)),
    ("gfr", dict(group_fast_raster=True, groups=2), ("gfr",)),
    ("gfr_ungrouped", dict(group_fast_raster=True), ("gfr",)),
    (
        "all",
        dict(
            uniform_k_decode=True,
            regt_prefetch=1,
            lds_swizzle=True,
            group_fast_raster=True,
            groups=2,
        ),
        ("ukd", "rtp1", "lxs", "gfr"),
    ),
)

# Specs both validators refuse: (label, spec overrides, arch).
_REFUSED = (
    ("ukd_pointwise", dict(uniform_k_decode=True, pointwise=True), _ARCH),
    ("rtp2", dict(regt_prefetch=2), _ARCH),
    ("rtp_gfx1100", dict(regt_prefetch=1), "gfx1100"),
    ("lxs_k_pad", dict(lds_swizzle=True, lds_k_pad=0), _ARCH),
    ("gfr_wavelet", dict(group_fast_raster=True, pipeline="wavelet"), _ARCH),
)


def _spec(*, dtype="fp16", pointwise=False, groups=1, **kw):
    fy = 1 if pointwise else 3
    base = dict(
        problem=ConvProblem(
            N=2,
            Hi=14,
            Wi=14,
            C=64 * groups,
            K=128,
            Y=fy,
            X=fy,
            pH=0 if pointwise else 1,
            pW=0 if pointwise else 1,
            groups=groups,
        ),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        pipeline="mem",
        epilogue="cshuffle",
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


class TestRdnaKnobSpecDict(unittest.TestCase):
    """Host-only: the C++ engine's spec dict carries each knob."""

    def test_defaults(self):
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        d = conv_implicit_gemm_spec_to_dict(_spec())
        self.assertIs(d["uniform_k_decode"], False)
        self.assertEqual(d["regt_prefetch"], 0)
        self.assertIs(d["lds_swizzle"], False)
        self.assertIs(d["group_fast_raster"], False)

    def test_knobs_reach_the_dict(self):
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        d = conv_implicit_gemm_spec_to_dict(_spec(**dict(_KNOBS[-1][1])))
        self.assertIs(d["uniform_k_decode"], True)
        self.assertEqual(d["regt_prefetch"], 1)
        self.assertIs(d["lds_swizzle"], True)
        self.assertIs(d["group_fast_raster"], True)


@unittest.skipIf(
    importlib.util.find_spec("rocke_library_engine") is None,
    "rocke_library_engine extension not importable",
)
class TestRdnaKnobCrossEngine(unittest.TestCase):
    """Both engines build the same kernel for every knob, under its name."""

    def test_python_and_cpp_agree(self):
        from kernels.conv.engine_bridge import lower_conv_implicit_gemm
        from rocke.core import backend as bk

        for label, kw, tags in _KNOBS:
            spec = _spec(**kw)
            with self.subTest(knob=label):
                ok, why = is_valid_spec(spec, _ARCH)
                self.assertTrue(ok, why)
                py = lower_conv_implicit_gemm(spec, arch=_ARCH, backend="python")
                cpp = lower_conv_implicit_gemm(spec, arch=_ARCH, backend="cpp")
                self.assertEqual(
                    (py.backend, cpp.backend), (bk.BACKEND_PYTHON, bk.BACKEND_CPP)
                )
                self.assertEqual(cpp.llvm_text, py.llvm_text)
                name = spec.kernel_name()
                self.assertTrue(all(f"_{t}" in name for t in tags), name)
                self.assertRegex(
                    cpp.llvm_text, r"define [^@\n]*@" + re.escape(name) + r"\("
                )

    def test_refused_by_both(self):
        from kernels.conv.engine_bridge import lower_conv_implicit_gemm

        for label, kw, arch in _REFUSED:
            spec = _spec(**kw)
            with self.subTest(case=label):
                self.assertFalse(is_valid_spec(spec, arch)[0])
                with self.assertRaises(Exception):
                    lower_conv_implicit_gemm(spec, arch=arch, backend="python")
                with self.assertRaises(Exception):
                    lower_conv_implicit_gemm(spec, arch=arch, backend="cpp")

    def test_defaults_are_inert(self):
        """The default-valued knobs build the historical kernel on both engines."""
        from kernels.conv.engine_bridge import lower_conv_implicit_gemm

        spec = _spec()
        explicit = dataclasses.replace(
            spec,
            uniform_k_decode=False,
            regt_prefetch=0,
            lds_swizzle=False,
            group_fast_raster=False,
        )
        a = lower_conv_implicit_gemm(spec, arch=_ARCH, backend="cpp")
        b = lower_conv_implicit_gemm(explicit, arch=_ARCH, backend="cpp")
        self.assertEqual(a.llvm_text, b.llvm_text)


if __name__ == "__main__":
    unittest.main()
