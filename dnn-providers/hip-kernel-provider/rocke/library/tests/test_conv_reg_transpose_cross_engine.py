# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-engine describe test for the wgrad register-transpose knobs.

The C++ engine (``rocke_library_engine``) must build every knob from the
Python spec dict byte for byte and refuse the same specs with the same
reason. Skipped where the binding is not built. The fixed two-engine corpus
is configs 109 onward of the wgrad parity emitters.
"""

from __future__ import annotations

import dataclasses
import unittest


def _wgrad(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=32,
        reg_transpose=True,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _lower(spec, arch):
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
    )


class _CrossEngine(unittest.TestCase):
    def setUp(self):
        from rocke.core import BackendError, import_engine

        from kernels.conv.engine_bridge import ENGINE_MODULE

        try:
            self.eng = import_engine(ENGINE_MODULE)
        except BackendError as e:
            self.skipTest(f"C++ engine not importable: {e}")

    @staticmethod
    def _dict(spec):
        d = {
            f.name: getattr(spec, f.name)
            for f in dataclasses.fields(spec)
            if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        }
        d = {k: v for k, v in d.items() if v is not None}
        d["problem"] = {
            k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
        }
        d["dtype_a"] = spec.data.dtype_a
        d["dtype_b"] = spec.data.dtype_b
        d["dtype_d"] = spec.data.dtype_d
        return d

    def assert_same(self, cases, arch="gfx1201"):
        for kw in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _wgrad(**kw)
                self.assertEqual(
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), arch),
                    _lower(spec, arch),
                )

    def assert_rejected_alike(self, cases, arch="gfx1201"):
        for kw in cases:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                spec = _wgrad(**kw)
                with self.assertRaises(ValueError) as py:
                    _lower(spec, arch)
                with self.assertRaises(RuntimeError) as cc:
                    self.eng.conv_wgrad_lower_llvm(self._dict(spec), arch)
                # Same reason text from both engines.
                self.assertIn(str(py.exception).split(": ", 1)[-1], str(cc.exception))


class TestRegTransposeCrossEngine(_CrossEngine):
    def test_identical_ir(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        self.assert_same(
            (
                # Width 8: 32 micro-tiles over 128 threads (partial participation).
                dict(),
                # Width 2: every thread loads one micro-tile.
                dict(vector_size_a=2, vector_size_b=2),
                dict(data=bf16, vector_size_a=4, vector_size_b=2),
                dict(split_k=2, vector_size_a=1),
            )
        )

    def test_rejected_alike(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        self.assert_rejected_alike(
            (
                dict(prefetch_addr=True),
                dict(data=ConvDataSpec(dtype_a="fp32", dtype_b="fp32", dtype_d="fp32")),
                dict(prefetch_global_regs=True),
                # Width 8 splits 96 rows into 48 micro-tiles: no whole waves.
                dict(tile_m=96, vector_size_a=8),
            )
        )
        self.assert_rejected_alike(
            (dict(wave_size=64, warp_tile_m=32, warp_tile_n=32),), arch="gfx950"
        )


class TestRegtPrefetchCrossEngine(_CrossEngine):
    def test_identical_ir(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        self.assert_same(
            (
                dict(regt_prefetch=3),
                # Mode 3 on a full-participation tile (no guarded reads).
                dict(regt_prefetch=3, data=bf16, vector_size_a=2, vector_size_b=2),
                dict(regt_prefetch=3, split_k=2),
            )
        )

    def test_rejected_alike(self):
        self.assert_rejected_alike(
            (
                # Modes 1 / 2 are not built.
                dict(regt_prefetch=1),
                dict(regt_prefetch=2),
                dict(regt_prefetch=3, reg_transpose=False),
                dict(regt_prefetch=3, unroll_k=True),
            )
        )


class TestRegtAddrStepCrossEngine(_CrossEngine):
    def test_identical_ir(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem

        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        self.assert_same(
            (
                dict(regt_addr_step=True),
                dict(regt_addr_step=True, regt_prefetch=3),
                dict(
                    regt_addr_step=True,
                    data=bf16,
                    vector_size_a=2,
                    vector_size_b=2,
                    problem=ConvProblem(
                        N=1, Hi=16, Wi=16, C=32, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1
                    ),
                ),
            )
        )

    def test_rejected_alike(self):
        self.assert_rejected_alike((dict(regt_addr_step=True, reg_transpose=False),))


class TestWmmaAtomicPairCrossEngine(_CrossEngine):
    def test_identical_ir(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        self.assert_same(
            (
                dict(split_k=2, wmma_atomic_pair=True),
                dict(split_k=2, wmma_atomic_pair=True, data=bf16),
                # The epilogue does not need the register-transpose loader.
                dict(split_k=2, wmma_atomic_pair=True, reg_transpose=False),
            )
        )

    def test_rejected_alike(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        self.assert_rejected_alike(
            (
                dict(wmma_atomic_pair=True),
                dict(split_k=2, two_stage=True, wmma_atomic_pair=True),
                dict(
                    split_k=2,
                    wmma_atomic_pair=True,
                    data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp32"),
                ),
            )
        )


def _stem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    # An ungrouped stem (stride == filter, no padding): X * C = 12.
    base = dict(N=2, Hi=32, Wi=32, C=3, K=64, Y=4, X=4, sH=4, sW=4)
    base.update(kw)
    return ConvProblem(**base)


def _xrun_stem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    # The patch-stem problem of the x_run validity tests (stride == filter).
    base = dict(N=2, Hi=64, Wi=64, C=3, K=64, Y=16, X=16, sH=16, sW=16)
    base.update(kw)
    return ConvProblem(**base)


def _xrun_base():
    """x_run on the sync X loader (no register transpose): valid for the stem
    problem on its own, so a combination refusal can only be the x_run gate."""
    return dict(
        x_run=True,
        problem=_xrun_stem(),
        reg_transpose=False,
        tile_k=16,
        vector_size_a=8,
        vector_size_b=8,
    )


def _xrun_depthwise():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(N=2, Hi=64, Wi=64, C=8, K=8, Y=3, X=3, pH=1, pW=1, groups=8)


def _xrun_stem3d():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=2, Di=8, Hi=64, Wi=64, C=3, K=64, Z=3, Y=3, X=3,
        sD=1, sH=1, sW=1, pD=1, pH=1, pW=1, dD=1,
    )  # fmt: skip


class TestXRunCrossEngine(_CrossEngine):
    def test_identical_ir(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16 = ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16")
        self.assert_same(
            (
                dict(x_run=True, problem=_stem(), regt_prefetch=3),
                dict(x_run=True, problem=_stem(), vector_size_b=4),
                dict(x_run=True, problem=_stem(), data=bf16, reg_transpose=False),
                dict(
                    x_run=True,
                    problem=_stem(),
                    split_k=2,
                    wmma_atomic_pair=True,
                    regt_prefetch=3,
                ),
            )
        )

    def test_rejected_alike(self):
        self.assert_rejected_alike(
            (
                # Problem conditions.
                dict(x_run=True, problem=_stem(pW=1)),
                dict(x_run=True, problem=_stem(C=6, groups=2)),
                # Combination gate: each base is refused for x_run alone (it
                # is valid, or fails a later gate, without it).
                dict(_xrun_base(), lds_k_outer=True, tile_k=32, warp_tile_k=32),
                dict(
                    _xrun_base(),
                    lds_k_outer=True,
                    async_dma=True,
                    tile_k=32,
                    warp_tile_k=32,
                ),
                dict(_xrun_base(), problem=_xrun_stem3d()),
            )
        )
        # The wave64 combinations.
        self.assert_rejected_alike(
            (
                dict(
                    _xrun_base(),
                    problem=_xrun_depthwise(),
                    group_merge=2,
                    wave_size=64,
                    tile_k=32,
                    warp_tile_k=32,
                ),
                dict(
                    _xrun_base(),
                    streamk="dp_sk",
                    wave_size=64,
                    tile_k=32,
                    warp_tile_k=32,
                ),
            ),
            arch="gfx950",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
