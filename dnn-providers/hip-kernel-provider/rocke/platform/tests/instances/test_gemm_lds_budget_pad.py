# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests that the AB LDS budget charges for ``lds_k_pad``.

The LDS row is ``tile_k + lds_k_pad`` elements wide, not ``tile_k``: the pad is
physically allocated on every row, which is how it breaks bank conflicts.
Charging only ``tile_k`` under-reserves by ``lds_k_pad / tile_k`` of the whole
AB region.

That is invisible at ``tdm_depth`` 1-2, where the slack absorbs it, and bites
at depth >= 3 where the region is multiplied by the ring. The specs it let
through did not fail cleanly either: ``is_valid_spec`` accepted them, the
backend then aborted, and an aborting build worker used to hang the sweep's
pool rather than reporting (see ``tests/test_sweep_build_pool.py``). So the
cost of this one missing term was a sweep that froze with no diagnostic.

The bound is asserted by *arithmetic identity* rather than against hardcoded
byte counts, so the tests keep meaning if the cap or a tile changes.
"""

from __future__ import annotations

import unittest

ARCH = "gfx1250"


def _spec(*, tile_m, tile_n, tile_k, depth, lds_k_pad, dtype="fp8e4m3"):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="lds_budget_probe",
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=64 if dtype.startswith(("fp8", "bf8")) else 32,
        ),
        trait=TraitSpec(
            pipeline="mem",
            scheduler="intrawave",
            epilogue="default",
            tdm=True,
            tdm_depth=depth,
            lds_k_pad=lds_k_pad,
        ),
        data=DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c="bf16" if dtype.startswith(("fp8", "bf8")) else dtype,
            dtype_acc="fp32",
            layout="RCR",
        ),
        wave_size=32,
    )


class TestGemmLdsBudgetPad(unittest.TestCase):
    def test_budget_scales_with_the_padded_row_not_the_bare_tile_k(self):
        """The reported AB figure must grow when only ``lds_k_pad`` grows.

        With the defect the figure is identical for every pad, which is the
        single clearest signature of the term being dropped.
        """
        from rocke.instances.common.gemm_universal import _ab_lds_plan

        base, _, _ = _ab_lds_plan(
            _spec(tile_m=128, tile_n=256, tile_k=256, depth=3, lds_k_pad=0), ARCH
        )
        padded, _, _ = _ab_lds_plan(
            _spec(tile_m=128, tile_n=256, tile_k=256, depth=3, lds_k_pad=48), ARCH
        )
        self.assertGreater(
            padded, base, "AB budget ignores lds_k_pad (the row is wider than tile_k)"
        )
        # Exact identity: the region is (tile_m + tile_n) * row_k * elem_bytes,
        # so a pad of P adds (tile_m + tile_n) * P * elem_bytes. fp8 is 1 byte.
        self.assertEqual(padded - base, (128 + 256) * 48 * 1)

    def test_depth3_specs_that_the_backend_rejects_are_rejected_here(self):
        """The two shapes that used to pass validation and then abort comgr."""
        from rocke.instances.common.gemm_universal import is_valid_spec

        for tile_m, tile_n in ((128, 256), (256, 128)):
            with self.subTest(tile=f"{tile_m}x{tile_n}x256"):
                ok, why = is_valid_spec(
                    _spec(
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=256,
                        depth=3,
                        lds_k_pad=48,
                    ),
                    arch=ARCH,
                )
                self.assertFalse(
                    ok,
                    "accepted a spec whose padded AB region exceeds the LDS cap; "
                    "the backend aborts on it",
                )
                self.assertIn("LDS budget", why)

    def test_legitimate_specs_are_not_over_rejected(self):
        """Tightening the bound must not cost specs that genuinely fit."""
        from rocke.instances.common.gemm_universal import is_valid_spec

        for tile_m, tile_n, tile_k, depth in (
            (128, 256, 256, 2),  # same shape, one fewer ring stage
            (128, 256, 128, 3),  # same depth, half the K tile
            (128, 128, 64, 4),  # deepest ring, small tile
        ):
            with self.subTest(tile=f"{tile_m}x{tile_n}x{tile_k}", depth=depth):
                ok, why = is_valid_spec(
                    _spec(
                        tile_m=tile_m,
                        tile_n=tile_n,
                        tile_k=tile_k,
                        depth=depth,
                        lds_k_pad=48,
                    ),
                    arch=ARCH,
                )
                self.assertTrue(ok, f"over-rejected a valid spec: {why}")

    def test_pad_is_charged_for_bf16_too(self):
        """The term is dtype-independent; only the element width differs."""
        from rocke.instances.common.gemm_universal import _ab_lds_plan

        kw = dict(tile_m=128, tile_n=128, tile_k=64, depth=2, dtype="bf16")
        base, _, _ = _ab_lds_plan(_spec(lds_k_pad=0, **kw), ARCH)
        padded, _, _ = _ab_lds_plan(_spec(lds_k_pad=8, **kw), ARCH)
        self.assertEqual(padded - base, (128 + 128) * 8 * 2)  # bf16 is 2 bytes


if __name__ == "__main__":
    unittest.main()
