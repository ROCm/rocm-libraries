# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ dgrad validator decides as the Python one (no GPU).

``rocke_dgrad_conv_is_valid_spec`` is the only gate in front of the C++ dgrad
builder, so a spec Python refuses must never pass it: the builder would then
build a kernel whose preconditions do not hold, or divide by a knob it trusts
(b_kpack). For the dgrad parity configs, one knob or problem field changed at
a time, the C++ verdict equals Python's; where C++ refuses a spec Python
accepts, it is a knob the C++ engine does not implement (its "Python-only"
refusal). The rules the C++ validator mirrors give the same reason. The LDS
charge of both validators is the pool the C++ builder allocates.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import re
import sys
import unittest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity")
_PY_ONLY = "Python-only"


def _engine():
    from rocke.core.backend import BackendError, import_engine

    from kernels.conv.engine_bridge import ENGINE_MODULE

    try:
        return import_engine(ENGINE_MODULE)
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _cfg(idx: int):
    if _PARITY not in sys.path:
        sys.path.insert(0, _PARITY)
    return importlib.import_module("conv_implicit_gemm_dgrad_emit")._spec(idx)


def _dict(spec) -> dict:
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        and getattr(spec, f.name) is not None
    }
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    if spec.data.mma_dtype is not None:
        d["mma_dtype"] = spec.data.mma_dtype
    return d


def _mutate(spec, kw):
    kw = dict(kw)
    prob = kw.pop("problem", None)
    if prob:
        kw["problem"] = dataclasses.replace(spec.problem, **prob)
    return dataclasses.replace(spec, **kw)


# (parity config, change, the reason both validators give). Each is a spec
# Python refuses that the C++ validator used to accept (and then built, or
# crashed on: b_kpack 3 divided by zero in the B loader).
_REFUSED = (
    (0, dict(b_kpack=3), "b_kpack must be 0, 2, 4 or 8; got 3"),
    (0, dict(tapk_prefetch=1), "it needs tap_outer_k"),
    (51, dict(tapk_xhalo=True), "tapk_xhalo and dy_halo are exclusive"),
    (91, dict(problem=dict(pW=1)), "tapk_xhalo needs pW == 0"),
    (91, dict(xh_defer=128), "xh_defer needs tapk_xhalo"),
    (91, dict(tapk_setprio=12), "tapk_setprio is 0, 1 or 2"),
    (92, dict(xh_b_kpad=9), "xh_b_kpad (2/4/6/8) needs tapk_xhalo with lds_xor"),
    (91, dict(xcd_remap=-1), "xcd_remap must be >= 0"),
    (91, dict(lds_k_pad=8), "lds_xor needs tapk_xhalo, lds_k_pad 0"),
    (91, dict(problem=dict(Z=5)), "xh_fast_a is 0..2"),
    (91, dict(tapk_ilv=-1), "tapk_ilv needs tapk_xhalo"),
    (91, dict(tapk_lds_ahead=-2), "tapk_lds_ahead needs tapk_xhalo"),
    (91, dict(b_kpack=0), "xh_b_wide needs tapk_xhalo"),
    (91, dict(tile_n=256), "LDS budget 72896 bytes"),
    (95, dict(problem=dict(Y=1)), "a filter with Y == X"),
    (95, dict(xh_yring=40), "a filter with Y == X"),
    (71, dict(tapk_prefetch=2), "LDS budget 98304 bytes"),
    (59, dict(cshuffle_m_chunks=2), None),  # (TF32: C++ refuses it as Python-only)
    (81, dict(tapk_pingpong=7), "tapk_pingpong must be 0..5; got 7"),
    (81, dict(epi_lds_barriers=True, tapk_pingpong=0, tapk_xhalo=False), None),
)

# One-knob changes applied to every parity config (values at and past each
# knob's range, and toggles), plus problem changes the x-halo rules read.
_KNOBS = {
    "b_kpack": (0, 2, 3, 4, 8, -1),
    "tapk_prefetch": (0, 1, 2, 3, 4),
    "tapk_xhalo": (True, False),
    "tap_outer_k": (True, False),
    "xh_fast_a": (0, 1, 2, 3),
    "tapk_ilv": (0, 2, -1),
    "tapk_lds_ahead": (0, 1, -1),
    "lds_xor": (True, False),
    "lds_k_pad": (0, 8),
    "xh_b_kpad": (0, 4, 5),
    "xh_b_wide": (True, False),
    "xh_defer": (0, 2, 64),
    "tapk_setprio": (0, 1, 2, 3),
    "tapk_lds_barrier": (True, False),
    "xhalo_exact": (True, False),
    "xcd_remap": (0, 6, -1),
    "epi_lds_barriers": (True, False),
    "cshuffle_m_chunks": (1, 2, 4, 0),
    "xh_yring": (0, 66, 130),
    "tapk_pingpong": (0, 4, 6),
    "split_atomic_pairs": (True, False),
    "pp_box_det": (0, 1, 3),
    "split_k": (1, 2, 8),
    "epilogue": ("default", "cshuffle"),
    "tile_n": (128, 256),
}
_PROBLEM = ({"pW": 1}, {"pW": 0}, {"X": 1}, {"Y": 5}, {"Z": 5})


class TestDgradCppValidator(unittest.TestCase):
    def test_refuses_what_python_refuses(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        eng = _engine()
        for idx, kw, why in _REFUSED:
            spec, arch = _cfg(idx)
            s = _mutate(spec, kw)
            with self.subTest(idx=idx, kw=kw):
                py = is_valid_dgrad_spec(s, arch)
                cpp = tuple(eng.conv_dgrad_is_valid_spec(_dict(s), arch))
                self.assertFalse(py[0])
                self.assertFalse(cpp[0], py[1])
                if why is not None:
                    self.assertIn(why, py[1])
                    self.assertEqual(cpp[1], py[1])

    def test_one_knob_changes_agree(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        eng = _engine()
        n = 0
        for idx in range(200):
            try:
                spec, arch = _cfg(idx)
            except SystemExit:
                continue
            if arch not in ("gfx942", "gfx950"):
                continue
            changes = [{k: v} for k, vs in _KNOBS.items() for v in vs]
            changes += [{"problem": p} for p in _PROBLEM]
            for kw in changes:
                try:
                    s = _mutate(spec, kw)
                    py = is_valid_dgrad_spec(s, arch)
                except (ValueError, ZeroDivisionError):
                    continue  # (a problem the dataclasses refuse)
                cpp = tuple(eng.conv_dgrad_is_valid_spec(_dict(s), arch))
                n += 1
                if py[0] == cpp[0]:
                    continue
                with self.subTest(idx=idx, kw=kw):
                    # Only ever a knob the C++ engine does not implement.
                    self.assertTrue(py[0], f"C++ accepts what Python refuses: {py[1]}")
                    self.assertIn(_PY_ONLY, cpp[1])
        self.assertGreater(n, 1000)

    def test_lds_charge_is_the_cpp_pool(self):
        # The C++ charge has the x-halo / tapk_prefetch 2 terms: a spec it
        # admits allocates no more LDS than it was charged.
        from kernels.common.conv_implicit_gemm_dgrad import dgrad_lds_bytes

        eng = _engine()
        cases = [(91, {}), (93, {}), (102, {}), (126, {}), (71, {}), (75, {})]
        cases += [(91, dict(tile_n=128, warp_n=2)), (71, dict(tapk_prefetch=0))]
        for idx, kw in cases:
            spec, arch = _cfg(idx)
            s = _mutate(spec, kw)
            with self.subTest(idx=idx, kw=kw):
                ok, why = eng.conv_dgrad_is_valid_spec(_dict(s), arch)
                self.assertTrue(ok, why)
                ll = eng.conv_dgrad_lower_llvm(_dict(s), arch)
                pool = re.findall(r"addrspace\(3\) global \[(\d+) x i8\]", ll)
                self.assertEqual([int(x) for x in pool], [dgrad_lds_bytes(s)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
