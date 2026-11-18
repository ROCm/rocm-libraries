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

# A 3-D and a depthwise problem for the ukd refusals.
_P3D = ConvProblem(
    N=2, Hi=8, Wi=8, C=64, K=128, Y=3, X=3, pH=1, pW=1, Di=8, Z=3, sD=1, pD=1, dD=1
)
_PDW = ConvProblem(N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1, groups=32)

# Specs both validators refuse: (label, spec overrides, arch, reason text).
# Each case is refused by its own knob's check (the reason pins which), not by
# an earlier gate.
_REFUSED = (
    (
        "ukd_pointwise",
        dict(uniform_k_decode=True, pointwise=True),
        _ARCH,
        "uniform_k_decode does not support the pointwise fast path",
    ),
    (
        "ukd_3d",
        dict(uniform_k_decode=True, problem=_P3D),
        _ARCH,
        "uniform_k_decode does not support 3-D convolution",
    ),
    (
        "ukd_group_merge",
        dict(
            uniform_k_decode=True,
            problem=_PDW,
            group_merge=2,
            vector_size_b=1,
            tile_n=32,
            warp_n=1,
        ),
        _ARCH,
        "uniform_k_decode does not support group_merge",
    ),
    ("rtp2", dict(regt_prefetch=2), _ARCH, "regt_prefetch (fwd) must be 0 or 1"),
    (
        "rtp_gfx1100",
        dict(regt_prefetch=1),
        "gfx1100",
        "regt_prefetch (fwd) is enabled only on the gfx12 RDNA targets",
    ),
    (
        "rtp_wavelet",
        dict(regt_prefetch=1, pipeline="wavelet"),
        _ARCH,
        "regt_prefetch is the 'mem' pipeline's K loop",
    ),
    (
        "rtp_split_k_cluster",
        dict(regt_prefetch=1, split_k_cluster=1),
        _ARCH,
        "regt_prefetch does not combine with split_k_cluster",
    ),
    (
        "lxs_k_pad",
        dict(lds_swizzle=True, lds_k_pad=0),
        _ARCH,
        "lds_swizzle sets the LDS layout itself",
    ),
    (
        "lxs_tile_k_bad",
        dict(lds_swizzle=True, tile_k=48),
        _ARCH,
        "lds_swizzle needs tile_k in 16/32/64/128 (got 48)",
    ),
    (
        "gfr_wavelet",
        dict(group_fast_raster=True, pipeline="wavelet"),
        _ARCH,
        "group_fast_raster does not support the wavelet pipeline",
    ),
    (
        "gfr_split_k",
        dict(group_fast_raster=True, groups=2, split_k=2),
        _ARCH,
        "group_fast_raster does not support split-K",
    ),
    (
        # Another target: the arch gate (the knob is gfx12 RDNA only) refuses it
        # before any knob-specific check.
        "gfr_gfx950",
        dict(
            group_fast_raster=True,
            groups=2,
            wave_size=64,
            tile_k=64,
            warp_tile_k=32,
        ),
        "gfx950",
        "group_fast_raster is enabled only on the gfx12 RDNA targets",
    ),
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

    def test_lxs_with_explicit_lds_layout_refused_by_bridge(self):
        """lxs + explicit lds_layout is refused at the bridge level.

        lds_layout is not forwarded to the C++ engine, so this combination
        would silently lose lds_layout.  The bridge catches it before the
        engine sees the dict."""
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict
        from rocke.helpers.layouts import LdsLayout

        import dataclasses

        spec = dataclasses.replace(
            _spec(lds_swizzle=True),
            lds_layout=LdsLayout(logical_cols=32, k_pad=0),
        )
        with self.assertRaises(
            ValueError, msg="lxs+lds_layout should raise ValueError"
        ):
            conv_implicit_gemm_spec_to_dict(spec)


class TestRdnaKnobRefusals(unittest.TestCase):
    """Python-only (runs in CI): each knob refusal fires with its own reason."""

    def test_each_refusal_hits_its_own_check(self):
        for label, kw, arch, why in _REFUSED:
            with self.subTest(case=label):
                ok, reason = is_valid_spec(_spec(**kw), arch)
                self.assertFalse(ok)
                self.assertIn(why, reason)

    def test_the_admitted_twins_are_valid(self):
        """The refused specs differ from valid ones only in the knob's trigger."""
        for label, kw in (
            ("ukd", dict(uniform_k_decode=True)),
            ("lxs_tk64", dict(lds_swizzle=True, tile_k=64)),
            ("gfr", dict(group_fast_raster=True, groups=2)),
        ):
            with self.subTest(case=label):
                ok, reason = is_valid_spec(_spec(**kw), _ARCH)
                self.assertTrue(ok, reason)


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

        for label, kw, arch, _why in _REFUSED:
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
