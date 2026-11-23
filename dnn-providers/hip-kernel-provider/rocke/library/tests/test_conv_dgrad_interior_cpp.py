# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ twin of kernels/conv/conv_dgrad_interior.py agrees with Python.

No GPU needed. dx_interior / dx_interior_axes are launch-time fields (the
dgrad binary is unchanged): the C++ engine carries the launch algebra (view
problem, dX origin and stride kernargs, the W-masked view of xh_dx_wmask),
the launch rules with Python's reason strings, the shell zero-fill launch
values and the zero-fill kernel itself. Each is compared here with the Python
reference through the ``rocke_library_engine`` binding.
"""

from __future__ import annotations

import dataclasses
import itertools
import unittest
from types import SimpleNamespace

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.conv.conv_dgrad_interior import (
    DgradHaloZeroSpec,
    build_dgrad_halo_zero,
    dx_interior_launch_conflict,
    dx_interior_problem_ok,
    halo_zero_launch,
    interior_view_dx,
    interior_view_problem,
    wmask_view,
)


def _engine():
    from rocke.core.backend import BackendError, import_engine

    from kernels.conv.engine_bridge import ENGINE_MODULE

    try:
        return import_engine(ENGINE_MODULE)
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _p3(N, D, H, W, C, pad=(0, 0, 0), **kw):
    base = dict(
        N=N,
        Di=D,
        Hi=H,
        Wi=W,
        C=C,
        K=C,
        Z=3,
        Y=3,
        X=3,
        sD=1,
        sH=1,
        sW=1,
        dD=1,
        dH=1,
        dW=1,
        pD=pad[0],
        pH=pad[1],
        pW=pad[2],
    )
    base.update(kw)
    return ConvProblem(**base)


# Halo-form (pad 0) 3-D UNet-like shapes, a padded form, a 2-D problem and
# problems the interior launch refuses (pad, stride, even filter, groups).
_PROBLEMS = (
    _p3(1, 258, 130, 130, 64),
    _p3(1, 66, 34, 34, 256),
    _p3(1, 18, 10, 10, 1024),
    _p3(2, 10, 6, 8, 32),
    _p3(1, 16, 8, 8, 64, pad=(1, 1, 1)),
    _p3(1, 16, 8, 8, 64, pad=(0, 1, 1)),
    ConvProblem(N=2, Hi=12, Wi=14, C=48, K=48, Y=3, X=3),
    ConvProblem(N=1, Hi=12, Wi=14, C=48, K=48, Y=3, X=3, pH=1, pW=1),
    _p3(1, 10, 10, 10, 64, sH=2),
    _p3(1, 10, 10, 10, 64, Y=2),
    _p3(1, 10, 10, 10, 64, groups=2),
    _p3(1, 3, 3, 3, 64),
)


def _pd(p) -> dict:
    return {k: v for k, v in dataclasses.asdict(p).items() if v is not None}


class TestDgradInteriorHost(unittest.TestCase):
    def test_problem_ok(self):
        eng = _engine()
        for p, mask in itertools.product(_PROBLEMS, range(1, 8)):
            with self.subTest(p=p, mask=mask):
                want = dx_interior_problem_ok(p, mask)
                self.assertEqual(
                    tuple(eng.conv_dgrad_dx_interior_problem_ok(_pd(p), mask)), want
                )

    def test_view(self):
        eng = _engine()
        for p, keep_w, axes in itertools.product(
            _PROBLEMS[:4] + _PROBLEMS[6:7], (False, True), range(1, 8)
        ):
            with self.subTest(p=p, keep_w=keep_w, axes=axes):
                got = eng.conv_dgrad_interior_view(_pd(p), keep_w, axes)
                self.assertEqual(
                    got["problem"], _pd(interior_view_problem(p, keep_w, axes))
                )
                off, strides = interior_view_dx(p, keep_w, axes)
                self.assertEqual(got["dx_offset"], off)
                self.assertEqual(got["dx_strides"], strides)

    def test_wmask_view(self):
        eng = _engine()
        for p in (_PROBLEMS[4], _PROBLEMS[5], _PROBLEMS[7]):
            with self.subTest(p=p):
                got = eng.conv_dgrad_wmask_view(_pd(p))
                v, off, strides = wmask_view(p)
                self.assertEqual(got["problem"], _pd(v))
                self.assertEqual(got["dx_offset"], off)
                self.assertEqual(got["dx_strides"], strides)

    def test_launch_rules(self):
        eng = _engine()
        base = dict(
            tap_outer_k=True,
            tapk_xhalo=False,
            split_k=1,
            pp_boxes=0,
            pp_box_det=0,
            pp_streamk=0,
            tapk_pp_persist=0,
            xh_dx_wmask=False,
        )
        variants = (
            {},
            dict(tap_outer_k=False),
            dict(split_k=4),
            dict(split_k=4, pp_boxes=7, pp_box_det=1),
            dict(pp_streamk=2),
            dict(tapk_pp_persist=2),
            dict(tapk_xhalo=True, xh_dx_wmask=True),
            dict(tapk_xhalo=False, xh_dx_wmask=True),
        )
        for v, dxi, axes, p in itertools.product(
            variants, (0, 1, 2, 3), (0, 1, 3, 6, 7, 9), _PROBLEMS
        ):
            fields = dict(base, **v)
            ident = SimpleNamespace(dx_interior=dxi, dx_interior_axes=axes, **fields)
            with self.subTest(fields=v, dxi=dxi, axes=axes, p=p):
                why = dx_interior_launch_conflict(ident, p)
                got = tuple(
                    eng.conv_dgrad_dx_interior_launch_ok(fields, dxi, axes, _pd(p))
                )
                self.assertEqual(got, (not why, why or "ok"))

    def test_zero_fill_launch(self):
        eng = _engine()
        for p, axes, block in itertools.product(
            _PROBLEMS[:4] + _PROBLEMS[6:7], (1, 3, 6, 7), (128, 256)
        ):
            with self.subTest(p=p, axes=axes, block=block):
                spec = DgradHaloZeroSpec(block=block)
                values, grid, blk = halo_zero_launch(p, 2, 0, 0, spec, axes)
                got_v, got_grid, got_blk = eng.conv_dgrad_halo_zero_launch(
                    _pd(p), 2, axes, {"block": block}
                )
                del values["dx"], values["dx_bytes"]
                self.assertEqual(got_v, values)
                self.assertEqual(tuple(got_grid), grid)
                self.assertEqual(tuple(got_blk), blk)
        with self.assertRaises(ValueError):
            eng.conv_dgrad_halo_zero_launch(_pd(_p3(1, 6, 6, 6, 4)), 2, 7, {})

    def test_zero_fill_empty_shell(self):
        # 1-tap filters: no shell, yet block 0 runs (into the section-3
        # branch, every store out of bounds), so no divisor kernarg is 0.
        eng = _engine()
        p = _p3(1, 6, 6, 6, 64, Z=1, Y=1, X=1)
        values, grid, _blk = halo_zero_launch(p, 2, 0, 0, DgradHaloZeroSpec(), 7)
        self.assertEqual(grid, (1, 1, 1))
        self.assertEqual((values["u1"], values["u2"], values["u3"]), (0, 0, 0))
        for k in ("e", "L1", "L2", "L3", "Do", "Ho"):
            self.assertGreaterEqual(values[k], 1, k)
        got_v, got_grid, _ = eng.conv_dgrad_halo_zero_launch(_pd(p), 2, 7, {})
        del values["dx"], values["dx_bytes"]
        self.assertEqual(got_v, values)
        self.assertEqual(tuple(got_grid), grid)


class TestDgradHaloZeroKernel(unittest.TestCase):
    def test_lowered_ir_is_byte_identical(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        eng = _engine()
        for block, arch in ((256, "gfx942"), (64, "gfx942"), (512, "gfx950")):
            with self.subTest(block=block, arch=arch):
                py = _lower_kernel_to_llvm_python(
                    build_dgrad_halo_zero(DgradHaloZeroSpec(block=block)), arch=arch
                )
                self.assertEqual(
                    eng.conv_dgrad_halo_zero_lower_llvm({"block": block}, arch), py
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
