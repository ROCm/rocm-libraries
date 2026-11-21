# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for forward ``addr64`` (large-tensor addressing). Phase 0:

No GPU and no comgr: the field is declared on ``ImplicitGemmConvSpec`` and
``KernelIdentity`` and threaded through ``kernel_name()`` / ``stable_hash()`` /
``label()`` -- the naming and cache-key plumbing a future addressing variant
needs -- but no builder reads it yet, so ``validate()`` refuses ``True``
rather than silently building the plain i32 kernel under an ``_a64`` name.
"""

from __future__ import annotations

import unittest
from dataclasses import replace


def _spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=64,
        pipeline="mem",
        epilogue="cshuffle",
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


class TestFwdAddr64Gate(unittest.TestCase):
    def test_default_is_off_and_untagged(self):
        spec = _spec()
        self.assertFalse(spec.addr64)
        self.assertNotIn("a64", spec.kernel_name())
        self.assertTrue(replace(spec, addr64=True).kernel_name().endswith("_a64"))

    def test_not_yet_implemented_is_refused(self):
        with self.assertRaisesRegex(ValueError, "addr64"):
            replace(_spec(), addr64=True).validate()
        # Off is unaffected: the existing validator keeps passing.
        _spec().validate()


class TestKernelIdentityAddr64(unittest.TestCase):
    def _ident(self, **kw):
        from kernels.conv.kernel_identity import KernelIdentity

        base = dict(
            arch="gfx950",
            direction="fwd",
            algorithm="implicit_gemm",
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_d="bf16",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=8,
            llvm_flavor="llvm20",
        )
        base.update(kw)
        return KernelIdentity(**base)

    def test_off_keeps_the_historical_hash(self):
        import hashlib
        import json
        from dataclasses import asdict

        ident = self._ident()
        fields = asdict(ident)
        for name in (
            "prefetch_addr",
            "prefetch_global_regs",
            "ws_replicas",
            "uniform_k_decode",
            "patch",
            "tap_outer_k",
            "dy_halo",
            "dy_halo_setprio",
            "dy_halo_kouter_pad",
            "halo_max_wo",
            "split_k_cluster",
            "addr64",
        ):
            del fields[name]
        del fields["streamk"], fields["streamk_reduction"]
        legacy = hashlib.sha1(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(ident.stable_hash(), legacy)

    def test_on_is_a_distinct_identity_and_round_trips(self):
        off = self._ident()
        on = self._ident(addr64=True)
        self.assertNotEqual(on.stable_hash(), off.stable_hash())
        self.assertIn("a64", on.label())
        from kernels.conv.kernel_identity import KernelIdentity

        self.assertEqual(KernelIdentity.from_label(on.label()), on)


if __name__ == "__main__":
    unittest.main(verbosity=2)
