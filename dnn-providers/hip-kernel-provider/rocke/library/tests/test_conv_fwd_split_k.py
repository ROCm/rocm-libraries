# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K forward implicit-GEMM conv (``ImplicitGemmConvSpec.split_k``).

CPU: the validator gate, the ABI, and that ``split_k = 1`` builds the same
kernel as before. GPU (gfx1250): Stage 1 + the Stage 2 reduce against the
reference across degrees, tiles, pipelines, epilogues, groups, strides, a
partial last K tile and degrees past the K-tile count (empty slices).
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import conv_arg_names
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    is_valid_spec,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None

_WMMA = dict(
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
    wave_size=32,
    vector_size_a=8,
    vector_size_b=8,
)
_PROBE = ConvProblem(N=1, Hi=16, Wi=16, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=2)
_BF16 = ConvDataSpec("bf16", "bf16", "bf16")


def _spec(problem=_PROBE, data=_BF16, **kw):
    cfg = dict(
        _WMMA,
        tile_m=16,
        tile_n=16,
        tile_k=64,
        warp_m=1,
        warp_n=1,
        pipeline="wavelet",
        epilogue="cshuffle",
        vector_size_c=8,
        num_load_waves=1,
    )
    cfg.update(kw)
    return ImplicitGemmConvSpec(problem=problem, data=data, **cfg)


class TestSplitKGate(unittest.TestCase):
    def test_wmma_split_k_is_valid(self):
        ok, why = is_valid_spec(_spec(split_k=2), "gfx1250")
        self.assertTrue(ok, why)

    def test_mfma_is_rejected(self):
        spec = _spec(
            split_k=2,
            wave_size=64,
            pipeline="mem",
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            tile_m=64,
            tile_n=64,
        )
        ok, why = is_valid_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("WMMA-only", why)

    def test_group_merge_is_rejected(self):
        dw = ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64)
        ok, why = is_valid_spec(
            _spec(
                problem=dw,
                split_k=2,
                group_merge=4,
                vector_size_a=None,
                vector_size_b=None,
                vector_size_c=None,
            ),
            "gfx1250",
        )
        self.assertFalse(ok)

    def test_name_tags_the_variant(self):
        self.assertTrue(_spec(split_k=2).kernel_name().endswith("_splitk"))
        self.assertNotIn("splitk", _spec().kernel_name())

    def test_abi_appends_workspace_and_slice(self):
        base = [n for n, _ in conv_arg_names(direction="fwd")]
        split = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual(split[: len(base)], base)
        self.assertEqual(split[len(base) :], ["ws_ptr", "ws_bytes", "ks", "ks_count"])

    def test_kernel_params_follow_the_abi(self):
        kernel = build_implicit_gemm_conv(_spec(split_k=2), arch="gfx1250")
        names = [p.name for p in kernel.params]
        abi = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual(names, abi)


class TestClusterSplitKGate(unittest.TestCase):
    """``split_k_cluster``: the single-launch fold through a workgroup cluster."""

    def test_mem_cluster_is_valid_and_sets_the_cluster(self):
        spec = _spec(split_k=2, split_k_cluster=4, pipeline="mem")
        ok, why = is_valid_spec(spec, "gfx1250")
        self.assertTrue(ok, why)
        kernel = build_implicit_gemm_conv(spec, arch="gfx1250")
        self.assertEqual(kernel.cluster_dims, (1, 1, 4))
        self.assertTrue(spec.kernel_name().endswith("_splitk_skc4"))
        # The ABI is the split-K one: the degree still arrives as ks_count.
        abi = [n for n, _ in conv_arg_names(direction="fwd", two_stage=True)]
        self.assertEqual([p.name for p in kernel.params], abi)

    def test_rejections(self):
        for kw, why in (
            (dict(split_k=1, split_k_cluster=4, pipeline="mem"), "split_k > 1"),
            (dict(split_k=2, split_k_cluster=4), "mem pipeline"),
            (dict(split_k=2, split_k_cluster=16, pipeline="mem"), "2..15"),
        ):
            ok, reason = is_valid_spec(_spec(**kw), "gfx1250")
            self.assertFalse(ok, kw)
            self.assertIn(why, reason)

    def test_off_by_default(self):
        kernel = build_implicit_gemm_conv(_spec(split_k=2), arch="gfx1250")
        self.assertIsNone(kernel.cluster_dims)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _gpu_skip() -> str:
    if not _HAS_TORCH:
        return "torch not importable"
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch()
    except Exception as e:  # pragma: no cover - no device
        return f"no GPU ({e})"
    return (
        "" if arch == "gfx1250" else f"split-K fwd is WMMA (gfx1250); device is {arch}"
    )


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(problem, spec_kw, degrees, dtype="bf16", unrolled=True, cluster=False):
    """Max relative error per degree of one split-K binary on ``problem``."""
    import torch

    from rocke import compile_kernel
    from builders.common.conv_reference import conv_reference_gfx1250
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_fwd_split_k_reduce import (
        FwdSplitKReduceSpec,
        build_conv_fwd_split_k_reduce,
        fwd_split_k_reduce_grid,
        fwd_split_k_reduce_signature,
        fwd_split_k_reduce_vec,
        fwd_split_k_workspace_nbytes,
    )
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import (
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    data = ConvDataSpec(dtype, dtype, dtype)

    def _stage1(degree):
        # The cluster binary bakes its degree (one per degree); the plain one
        # serves every degree.
        spec = _spec(
            problem=problem,
            data=data,
            split_k=2,
            split_k_cluster=degree if cluster else 0,
            **spec_kw,
        )
        ok, why = is_valid_spec(spec, "gfx1250")
        assert ok, why
        art = compile_kernel(
            build_implicit_gemm_conv(spec, arch="gfx1250"), arch="gfx1250"
        )
        return spec, KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=conv_args_signature(dtype, two_stage=True),
            cluster_dims=(1, 1, degree) if cluster else None,
        )

    if not cluster:
        spec, s1 = _stage1(2)
    elems = problem.M * problem.K

    def _stage2(degree):
        # The degree-specialised fold (what the benchmark launches), or the
        # runtime-loop one.
        r_spec = FwdSplitKReduceSpec(
            dtype_d=dtype,
            vec=fwd_split_k_reduce_vec(elems),
            degree=degree if unrolled else 0,
        )
        r_art = compile_kernel(build_conv_fwd_split_k_reduce(r_spec), arch="gfx1250")
        return r_spec, KernelLauncher(
            hsaco=r_art.hsaco,
            kernel_name=r_art.kernel_name,
            signature=fwd_split_k_reduce_signature(r_spec),
        )

    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    torch.manual_seed(0)
    A = (
        torch.empty(problem.N, problem.Hi, problem.Wi, problem.C)
        .uniform_(-1, 1)
        .to(tdt)
    )
    B = (
        torch.empty(problem.K, problem.Y, problem.X, problem.cpg)
        .uniform_(-1, 1)
        .to(tdt)
    )
    ref = conv_reference_gfx1250(A, B, problem, out_dtype=tdt).float().cpu()
    D = torch.empty(problem.N, problem.Ho, problem.Wo, problem.K, dtype=tdt)

    rt = Runtime()
    bufs = [rt.alloc(A.nbytes), rt.alloc(B.nbytes), rt.alloc(D.nbytes)]
    rt.memcpy_h2d(bufs[0], _u8(A), A.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(B), B.nbytes)
    errs = {}
    try:
        for degree in degrees:
            if cluster:
                spec, s1 = _stage1(degree)
            r_spec, s2 = _stage2(degree)
            ws_bytes = fwd_split_k_workspace_nbytes(problem.M, problem.K, degree)
            ws = rt.alloc(ws_bytes)
            # Poison the workspace and the output: Stage 1 must write every slab
            # element, Stage 2 every output element.
            rt.memset(ws, 0x7F, ws_bytes)
            rt.memset(bufs[2], 0x7F, D.nbytes)
            args = ConvArgs.from_problem(
                problem, tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=spec.tile_k
            )
            values = args.to_launch_values(
                *[int(x) for x in bufs],
                A.nbytes,
                B.nbytes,
                D.nbytes,
                split_k=degree,
                ws_ptr=int(ws),
                ws_bytes=ws_bytes,
            )
            s1_cfg = LaunchConfig(
                grid=args.grid(degree),
                block=(spec.launch_block_size, 1, 1),
                fence=True,
            )
            if cluster:
                # One launch: the cluster folds the slabs into D itself. Run
                # it through bind() too, the path the benchmark times.
                s1(values, config=s1_cfg)
                out = torch.empty_like(D)
                rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
                errs[degree] = float(
                    (out.float() - ref).abs().max() / ref.abs().max().clamp(min=1.0)
                )
                rt.memset(bufs[2], 0x7F, D.nbytes)
                s1.bind(values, config=s1_cfg)()
                synchronize_and_release()
                rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
                rt.free(ws)
                diff = (out.float() - ref).abs().max()
                errs[degree] = max(
                    errs[degree], float(diff / ref.abs().max().clamp(min=1.0))
                )
                continue
            s1(values, config=s1_cfg)
            s2(
                dict(
                    ws_ptr=int(ws),
                    d_ptr=int(bufs[2]),
                    elems=elems,
                    ws_bytes=ws_bytes,
                    d_bytes=D.nbytes,
                    ks_count=degree,
                ),
                config=LaunchConfig(
                    grid=fwd_split_k_reduce_grid(r_spec, elems),
                    block=(r_spec.block_size, 1, 1),
                    fence=True,
                ),
            )
            out = torch.empty_like(D)
            rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
            rt.free(ws)
            diff = (out.float() - ref).abs().max()
            errs[degree] = float(diff / ref.abs().max().clamp(min=1.0))
    finally:
        for x in bufs:
            rt.free(x)
    return errs


_GPU_SKIP = _gpu_skip()

# (problem, spec knobs, degrees)
_CASES = [
    # small M, deep K; 9*256 = 2304 = 36 K tiles of 64
    (
        "small_m_wavelet",
        ConvProblem(N=1, Hi=7, Wi=7, C=256, K=64, Y=3, X=3, pH=1, pW=1),
        {},
        (1, 2, 4, 16, 64),
    ),
    (
        "mem_cshuffle_64",
        ConvProblem(N=2, Hi=12, Wi=12, C=128, K=96, Y=3, X=3, pH=1, pW=1),
        dict(tile_m=64, tile_n=64, warp_m=2, warp_n=2, pipeline="mem"),
        (1, 3, 8),
    ),
    (
        "mem_default",
        ConvProblem(N=1, Hi=9, Wi=9, C=64, K=40, Y=3, X=3, pH=1, pW=1),
        dict(
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=1,
            pipeline="mem",
            epilogue="default",
            vector_size_c=1,
        ),
        (2, 5),
    ),
    # partial last K tile: 9*24 = 216 is not a multiple of 64
    (
        "k_tail_stride2",
        ConvProblem(N=2, Hi=15, Wi=13, C=24, K=32, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
        {},
        (2, 3, 4),
    ),
    (
        "grouped",
        ConvProblem(N=1, Hi=10, Wi=10, C=128, K=64, Y=3, X=3, pH=1, pW=1, groups=2),
        {},
        (1, 4),
    ),
    (
        "pointwise_shape_2x2s2",
        ConvProblem(N=1, Hi=14, Wi=14, C=256, K=128, Y=2, X=2, sH=2, sW=2),
        {},
        (2, 8),
    ),
]


@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestSplitKNumerics(unittest.TestCase):
    def test_cases(self):
        for dtype in ("bf16", "fp16"):
            tol = 2e-2
            for name, problem, kw, degrees in _CASES:
                with self.subTest(case=name, dtype=dtype):
                    for unrolled in (True, False):
                        errs = _run(problem, kw, degrees, dtype, unrolled)
                        for degree, err in errs.items():
                            self.assertLess(
                                err,
                                tol,
                                f"{name} {dtype} degree {degree} "
                                f"unrolled={unrolled}: {err}",
                            )


# (problem, spec knobs, degrees) for the single-launch cluster fold (mem only).
_CLUSTER_CASES = [
    (
        "small_m",
        ConvProblem(N=1, Hi=7, Wi=7, C=256, K=64, Y=3, X=3, pH=1, pW=1),
        dict(pipeline="mem"),
        (2, 4, 15),
    ),
    (
        "mem_cshuffle_64",
        ConvProblem(N=2, Hi=12, Wi=12, C=128, K=96, Y=3, X=3, pH=1, pW=1),
        dict(tile_m=64, tile_n=64, warp_m=2, warp_n=2, pipeline="mem"),
        (3, 8),
    ),
    (
        "k_tail_stride2",
        ConvProblem(N=2, Hi=15, Wi=13, C=24, K=32, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
        dict(tile_m=32, tile_n=32, warp_m=2, warp_n=1, pipeline="mem"),
        (2, 4),
    ),
    (
        "grouped",
        ConvProblem(N=1, Hi=10, Wi=10, C=128, K=64, Y=3, X=3, pH=1, pW=1, groups=2),
        dict(tile_m=32, tile_n=32, warp_m=1, warp_n=2, pipeline="mem"),
        (2, 4),
    ),
]


@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestClusterSplitKNumerics(unittest.TestCase):
    def test_cases(self):
        for dtype in ("bf16", "fp16"):
            for name, problem, kw, degrees in _CLUSTER_CASES:
                with self.subTest(case=name, dtype=dtype):
                    errs = _run(problem, kw, degrees, dtype, cluster=True)
                    for degree, err in errs.items():
                        self.assertLess(err, 2e-2, f"{name} {dtype} S={degree}: {err}")


if __name__ == "__main__":
    unittest.main()
