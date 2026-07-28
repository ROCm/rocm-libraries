# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numeric-correctness tests for grouped forward implicit-GEMM conv.

Covers the AICK-1752 grouped Conv2d forward path on the MFMA (CDNA) arches
(gfx942 / gfx950): each group is a per-group GEMM placed on ``blockIdx.z``
(CK-style grid-per-group). The oracle is the shared grouped NumPy fp32-accum
reference in ``rocke.instances.common.manifest_runner.conv`` (per-group einsum),
so this test exercises the real descriptor + kernel path end to end and compares
against a trusted reference.

Cases include the g32/cpg8 cardinality-grouped hero shape (kpg < tile_n, so the
surplus N lanes are masked), asymmetric cpg != kpg, a partial K tail
(K_gemm = Y*X*cpg not a multiple of tile_k), and bf16 + fp16. groups == 1 is
included as a non-regression anchor.

Run on a CDNA ROCm runner (needs a real GPU; numpy required):
  HIP_VISIBLE_DEVICES=0 python -m pytest \
    platform/tests/instances/test_grouped_conv_numeric.py -v
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_GPU_ARCH = get_device_arch(0)
_IS_CDNA = _GPU_ARCH in ("gfx942", "gfx950", "gfx90a")
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_SKIP = (
    f"needs a CDNA (MFMA) ROCm GPU + numpy; detected {_GPU_ARCH!r}, "
    f"numpy={'yes' if _HAS_NUMPY else 'no'}"
)


def _run_grouped_case(problem, *, dtype="fp16", epilogue="default", pipeline="mem"):
    """Build, compile, launch one grouped conv and return ``(max_err, bad, size)``."""
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.conv_implicit_gemm import (
        ConvDataSpec,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
        implicit_gemm_conv_grid,
        is_valid_spec,
    )
    from rocke.instances.common.manifest_runner.conv import run_conv_manifest_problem
    from rocke.runtime.hip_module import Runtime

    arch = get_device_arch(0)
    spec = ImplicitGemmConvSpec(
        problem=problem,
        data=ConvDataSpec(dtype, dtype, dtype),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        # 16x16x16 f16/bf16 is valid on every CDNA arch in the catalog.
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        epilogue=epilogue,
        pipeline=pipeline,
        vector_size_c=(8 if epilogue == "cshuffle" else None),
    )
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise unittest.SkipTest(f"spec invalid on {arch}: {why}")

    kernel = build_implicit_gemm_conv(spec, arch=arch)
    # Use the default backend (the C++ engine in a normal install): it lowers the
    # Python-authored serialized IR family-agnostically, so grouped conv — built
    # from standard embed/unmerge/pad ops — lowers byte-identically without a
    # conv-specific C++ rebuild. This exercises the real shipping path.
    art = compile_kernel(kernel, arch=arch)
    grid = implicit_gemm_conv_grid(spec)

    p = problem
    manifest = {
        "conv": [
            p.N,
            p.Hi,
            p.Wi,
            p.C,
            p.K,
            p.Y,
            p.X,
            p.sH,
            p.sW,
            p.pH,
            p.pW,
            p.dH,
            p.dW,
        ],
        "groups": p.groups,
        "cpg": p.cpg,
        "kpg": p.kpg,
        "dtype": dtype,
        "grid_explicit": list(grid),
        "threads_per_block": spec.block_size,
        "sig_has_bytes": 1,
        "kernel_name": art.kernel_name,
    }
    make_args, _grid, block, _flop, _bytes, check = run_conv_manifest_problem(
        manifest, None, True
    )
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    args, ptrs = make_args(rt)
    rt.launch(fn, grid, block, args)
    result = check(rt, ptrs)
    module.unload()
    return result


@unittest.skipUnless(_IS_CDNA and _HAS_NUMPY, _SKIP)
class TestGroupedConvNumeric(unittest.TestCase):
    def _assert_pass(self, problem, **kw):
        max_err, bad, size = _run_grouped_case(problem, **kw)
        self.assertEqual(
            bad, 0, f"grouped conv mismatch: {bad}/{size} bad, max_err={max_err:.4g}"
        )

    def _p(self, **kw):
        from rocke.instances.common.conv_implicit_gemm import ConvProblem

        return ConvProblem(**kw)

    def test_groups1_nonregression(self):
        self._assert_pass(self._p(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1))

    def test_g4_fp16(self):
        self._assert_pass(
            self._p(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4)
        )

    def test_g4_bf16(self):
        self._assert_pass(
            self._p(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4),
            dtype="bf16",
        )

    def test_g32_cpg8_hero_fp16(self):
        # Cardinality-grouped: cpg=kpg=8 < tile_n=64 (masked N tail) and
        # K_gemm = 3*3*8 = 72 (partial K tile past tile_k=64).
        self._assert_pass(
            self._p(N=2, Hi=8, Wi=8, C=256, K=256, Y=3, X=3, pH=1, pW=1, groups=32)
        )

    def test_g32_cpg8_hero_bf16(self):
        self._assert_pass(
            self._p(N=2, Hi=8, Wi=8, C=256, K=256, Y=3, X=3, pH=1, pW=1, groups=32),
            dtype="bf16",
        )

    def test_g4_asymmetric_cpg_ne_kpg(self):
        # C=64 -> cpg=16, K=128 -> kpg=32.
        self._assert_pass(
            self._p(N=2, Hi=10, Wi=10, C=64, K=128, Y=3, X=3, pH=1, pW=1, groups=4)
        )

    def test_g8_1x1_stride2(self):
        self._assert_pass(
            self._p(N=2, Hi=16, Wi=16, C=128, K=128, Y=1, X=1, sH=2, sW=2, groups=8)
        )

    def test_g4_cshuffle_epilogue(self):
        self._assert_pass(
            self._p(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4),
            epilogue="cshuffle",
        )

    def test_g4_compv4_pipeline(self):
        self._assert_pass(
            self._p(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4),
            pipeline="compv4",
        )


def _run_via_dispatch(*, dtype="fp16", **conv_kw):
    """Route a request through ``dispatch_conv`` then build+launch the selected
    spec, returning ``(candidate_spec_id, grid, (max_err, bad, size))``.

    Exercises the real dispatcher selection + launch path (AICK-1752 acceptance:
    "direct runtime/dispatcher selection and launch tests").
    """
    from rocke.dispatch.families.conv import ConvRequest, dispatch_conv
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.conv_implicit_gemm import build_implicit_gemm_conv
    from rocke.instances.common.manifest_runner.conv import run_conv_manifest_problem
    from rocke.runtime.hip_module import Runtime

    arch = get_device_arch(0)
    base = dict(
        N=2,
        C=64,
        K=64,
        Hi=16,
        Wi=16,
        Y=3,
        X=3,
        pad_h=1,
        pad_w=1,
        arch=arch,
        dtype=dtype,
    )
    base.update(conv_kw)
    req = ConvRequest(**base)
    res = dispatch_conv(req)
    spec = res.spec
    p = spec.problem
    kernel = build_implicit_gemm_conv(spec, arch=arch)
    art = compile_kernel(kernel, arch=arch)
    manifest = {
        "conv": [
            p.N,
            p.Hi,
            p.Wi,
            p.C,
            p.K,
            p.Y,
            p.X,
            p.sH,
            p.sW,
            p.pH,
            p.pW,
            p.dH,
            p.dW,
        ],
        "groups": p.groups,
        "cpg": p.cpg,
        "kpg": p.kpg,
        "dtype": "bf16" if dtype == "bf16" else "fp16",
        "grid_explicit": list(res.grid),
        "threads_per_block": res.block[0],
        "sig_has_bytes": 1,
        "kernel_name": art.kernel_name,
    }
    make_args, _grid, _block, _flop, _bytes, check = run_conv_manifest_problem(
        manifest, None, True
    )
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)
    args, ptrs = make_args(rt)
    rt.launch(fn, res.grid, res.block, args)
    result = check(rt, ptrs)
    module.unload()
    return res.candidate.spec_id, res.grid, result


@unittest.skipUnless(_IS_CDNA and _HAS_NUMPY, _SKIP)
class TestGroupedConvDispatchLaunch(unittest.TestCase):
    """Dispatcher-selection + on-GPU launch correctness for grouped conv."""

    def test_dispatch_groups1(self):
        _sid, grid, (max_err, bad, size) = _run_via_dispatch()
        self.assertEqual(grid[2], 1)
        self.assertEqual(bad, 0, f"{bad}/{size} bad, max_err={max_err:.4g}")

    def test_dispatch_g4_bf16(self):
        _sid, grid, (max_err, bad, size) = _run_via_dispatch(G=4, dtype="bf16")
        self.assertEqual(grid[2], 4)
        self.assertEqual(bad, 0, f"{bad}/{size} bad, max_err={max_err:.4g}")

    def test_dispatch_g32_cpg8_bf16(self):
        _sid, grid, (max_err, bad, size) = _run_via_dispatch(
            N=2, C=256, K=256, Hi=8, Wi=8, G=32, dtype="bf16"
        )
        self.assertEqual(grid[2], 32)
        self.assertEqual(bad, 0, f"{bad}/{size} bad, max_err={max_err:.4g}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
