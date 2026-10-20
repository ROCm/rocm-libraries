# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Selection + support + grid tests for grouped forward dispatch.

CPU-only (no GPU / no comgr): asserts that the gfx1250 forward candidate
admits grouped and depthwise requests the way the gfx950 one does -- grid-per-
group, the group index on block_id_z -- and that the spec it hands the builder
passes the builder's own validator.
"""

from __future__ import annotations

import math
import unittest

from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    _fwd_grid,
    _problem,
    dispatch_conv_grouped,
)
from kernels.common.conv_implicit_gemm import is_valid_spec_for_problem


def _fwd(arch, **kw):
    base = dict(
        N=2,
        C=64,
        K=64,
        Hi=14,
        Wi=14,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        direction="fwd",
        dtype="bf16",
    )
    base.update(kw)
    return ConvGroupedRequest(**base)


# (C, K, G): grouped, cardinality-grouped, asymmetric and depthwise.
_GROUPED = ((64, 64, 4), (256, 256, 32), (48, 96, 2), (64, 64, 64), (192, 192, 192))


class TestGroupedFwdDispatch(unittest.TestCase):
    def test_gfx1250_admits_grouped_and_depthwise(self):
        for C, K, G in _GROUPED:
            for dtype in ("fp16", "bf16"):
                with self.subTest(C=C, K=K, G=G, dtype=dtype):
                    req = _fwd("gfx1250", C=C, K=K, G=G, dtype=dtype)
                    r = dispatch_conv_grouped(req)
                    self.assertEqual(r.spec.arch, "gfx1250")
                    p = _problem(req)
                    ok, why = is_valid_spec_for_problem(
                        r.spec.to_fwd_spec(p), p, "gfx1250"
                    )
                    self.assertTrue(ok, why)

    def test_grid_is_per_group(self):
        # One workgroup column per group on z, the per-group GEMM tiled on x/y.
        for arch in ("gfx950", "gfx1250"):
            for C, K, G in _GROUPED:
                with self.subTest(arch=arch, G=G):
                    req = _fwd(arch, C=C, K=K, G=G)
                    spec = dispatch_conv_grouped(req).spec
                    p = _problem(req)
                    gx, gy, gz = _fwd_grid(spec, req)
                    self.assertEqual(gz, G)
                    self.assertEqual(gy, math.ceil(p.M / spec.tile_m))
                    self.assertEqual(gx, math.ceil((K // G) / spec.tile_n))


if __name__ == "__main__":
    unittest.main()
