# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics for stream-K implicit-GEMM backward-weight convolution.

Every case builds, compiles and launches a stream-K wgrad kernel, then checks
dW against a float32 ``torch.nn.grad.conv2d_weight`` reference computed on the
CPU. Before the tolerance check a row-coverage guard reports whole output
channels that were never written (a dropped group or tile reports as such
rather than as an opaque numeric miss).

The shapes are chosen to exercise the partitioner, not the GEMM: several
stream-K CTAs per tile (multi-round linear/tree fixups), a data-parallel plus
stream-K mix, a grouped problem whose last M tile is partial, and a small CTA
pool so the stream-K path is taken at these sizes. The linear and tree fixups
are also run repeatedly and must be bit-identical across launches -- the only
check that catches a missing release/acquire on the partial hand-off. Before
every launch the partial slots and dW are poisoned with NaN bytes (only the
flags are zeroed, as the contract requires), so a consumer that reads a slot
before its producer's release lands sees NaN rather than the previous
launch's bit-identical partial.

Requires a gfx942/gfx950 GPU and torch; skips cleanly otherwise. Host-side
coverage lives in ``test_conv_wgrad_streamk.py``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
GPU_ARCH = get_device_arch(0)


def _skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in ("gfx942", "gfx950"):
        return f"stream-K wgrad is gfx942/gfx950 only; got {GPU_ARCH}"
    return ""


_SKIP_REASON = _skip_reason()

# Relative max-error bound by dW dtype: the reference is fp32 over fp16/bf16
# inputs, so the bound is set by the output rounding.
_TOL = {"fp32": 1e-4, "fp16": 2e-3, "bf16": 2e-2}
_REDUCTIONS = (
    ("linear", "fp16"),
    ("tree", "fp16"),
    ("atomic", "fp32"),
    ("workspace", "bf16"),
)


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _reference(X, dY, p):
    import torch

    w = torch.nn.grad.conv2d_weight(
        X.permute(0, 3, 1, 2).contiguous(),
        weight_size=(p.K, p.C // p.groups, p.Y, p.X),
        grad_output=dY.permute(0, 3, 1, 2).contiguous(),
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return w.permute(0, 2, 3, 1).contiguous()


def _run(spec, *, launches: int = 1, problem=None, artifact=None):
    """Build, launch ``launches`` times, return (dW outputs, fp32 reference).

    ``problem`` is the launch shape (default ``spec.problem``): the kernel is
    AOT, so one binary -- passed in as ``artifact`` to reuse it -- serves any
    shape its spec admits.
    """
    import torch

    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from kernels.common.conv_implicit_gemm_wgrad import (
        build_implicit_gemm_conv_wgrad,
        wgrad_streamk_grid,
        wgrad_streamk_launch_values,
        wgrad_streamk_signature,
        wgrad_streamk_workspace_layout,
        wgrad_streamk_workspace_nbytes,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_grid,
        wgrad_reduce_signature,
    )

    arch = GPU_ARCH
    p = problem if problem is not None else spec.problem
    art = artifact or compile_kernel(
        build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
    )
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=wgrad_streamk_signature(spec),
    )
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch.manual_seed(0)
    X = torch.empty(p.N, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(dtypes[spec.data.dtype_b])
    dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1, 1).to(dtypes[spec.data.dtype_a])
    dW = torch.empty(p.K, p.Y, p.X, p.C // p.groups, dtype=dtypes[spec.data.dtype_d])
    ref = _reference(X.float(), dY.float(), p)

    rt = Runtime()
    ws_bytes = wgrad_streamk_workspace_nbytes(spec, arch=arch, problem=p)
    bufs = [rt.alloc(dY.nbytes), rt.alloc(X.nbytes), rt.alloc(dW.nbytes)]
    ws = rt.alloc(max(ws_bytes, 4))
    bufs.append(ws)
    rt.memcpy_h2d(bufs[0], _u8(dY), dY.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(X), X.nbytes)
    reduction = spec.streamk_reduction
    flags_bytes, partials_bytes = wgrad_streamk_workspace_layout(
        spec, arch=arch, problem=p
    )
    extra = {}
    if reduction == "workspace":
        extra = dict(ws_ptr=ws, ws_bytes=ws_bytes)
    elif reduction in ("linear", "tree"):
        extra = dict(sk_flags=ws, sk_partials=ws + flags_bytes)
    values = wgrad_streamk_launch_values(
        spec,
        bufs[0],
        bufs[1],
        bufs[2],
        dY.nbytes,
        X.nbytes,
        dW.nbytes,
        arch=arch,
        problem=p,
        **extra,
    )
    if reduction == "workspace":
        rspec = WgradReduceSpec(
            problem=p,
            dtype_d=spec.data.dtype_d,
            groups=p.groups,
            ws_replicas=spec.ws_replicas,
        )
        rart = compile_kernel(
            build_conv_wgrad_workspace_reduce(rspec, arch=arch), arch=arch
        )
        reducer = KernelLauncher(
            hsaco=rart.hsaco,
            kernel_name=rart.kernel_name,
            signature=wgrad_reduce_signature(rspec),
        )
        rvalues = {
            "ws_ptr": ws,
            "dw_ptr": bufs[2],
            "wg_M": p.K // p.groups,
            "wg_N": p.Y * p.X * (p.C // p.groups),
            "ws_bytes": ws_bytes,
            "dw_bytes": dW.nbytes,
            "groups": p.groups,
        }

    outs = []
    try:
        for _ in range(launches):
            # The documented zeroing contract, per reduction: atomic accumulates
            # into dW, workspace into the scratch, linear/tree only reset flags.
            # Everything the contract leaves alone is poisoned (0xFF bytes are
            # NaN in every float dtype): a stale value from the previous
            # launch would otherwise be bit-identical to the right one and
            # hide an early read of a partial or an unwritten dW element.
            if reduction == "atomic":
                rt.memset(bufs[2], 0, dW.nbytes)
            else:
                rt.memset(bufs[2], 0xFF, dW.nbytes)
            if reduction == "workspace":
                rt.memset(ws, 0, ws_bytes)
            elif reduction in ("linear", "tree"):
                if partials_bytes:
                    rt.memset(ws + flags_bytes, 0xFF, partials_bytes)
                if flags_bytes:
                    rt.memset(ws, 0, flags_bytes)
            launcher(
                values,
                config=LaunchConfig(
                    grid=wgrad_streamk_grid(spec, arch=arch, problem=p),
                    block=(spec.block_size, 1, 1),
                    fence=True,
                ),
            )
            if reduction == "workspace":
                reducer(
                    rvalues,
                    config=LaunchConfig(
                        grid=wgrad_reduce_grid(rspec),
                        block=(rspec.block_size, 1, 1),
                        fence=True,
                    ),
                )
            out = torch.empty_like(dW)
            rt.memcpy_d2h(_u8(out), bufs[2], dW.nbytes)
            outs.append(out)
    finally:
        for buf in bufs:
            rt.free(buf)
    return outs, ref


def _check(test, spec, *, launches: int = 1, problem=None, artifact=None):
    outs, ref = _run(spec, launches=launches, problem=problem, artifact=artifact)
    p = problem if problem is not None else spec.problem
    out = outs[0].float()
    written = out.reshape(p.K, -1).abs().sum(dim=1) > 0
    expected = ref.reshape(p.K, -1).abs().sum(dim=1) > 0
    missing = int((expected & ~written).sum())
    test.assertEqual(
        missing, 0, f"{missing}/{p.K} dW output-channel rows were never written"
    )
    rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    test.assertLess(rel, _TOL[spec.data.dtype_d], f"rel_err={rel:.3e}")
    return outs


def _spec(problem, *, mode, reduction, dtype_d, ctas):
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    ab = "bf16" if dtype_d == "bf16" else "fp16"
    edge = 32 if GPU_ARCH == "gfx950" else 16
    return WgradConvSpec(
        problem=problem,
        data=ConvDataSpec(dtype_a=ab, dtype_b=ab, dtype_d=dtype_d),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk=mode,
        streamk_reduction=reduction,
        streamk_ctas=ctas,
    )


def _problems():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return {
        # 6 tiles, 32 iterations each, over 16 CTAs: up to 3 partners per tile,
        # two tree rounds.
        "multi_partner": (
            ConvProblem(N=8, Hi=16, Wi=16, C=16, K=96, Y=3, X=3, pH=1, pW=1),
            16,
        ),
        # 25 tiles over 7 CTAs: 14 data-parallel tiles plus 11 stream-K tiles.
        "dp_plus_sk": (
            ConvProblem(N=2, Hi=8, Wi=8, C=32, K=320, Y=3, X=3, pH=1, pW=1),
            7,
        ),
        "strided": (
            ConvProblem(
                N=3, Hi=13, Wi=11, C=24, K=72, Y=3, X=3, sH=2, sW=2, pH=1, pW=1
            ),
            5,
        ),
    }


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKNumerics(unittest.TestCase):
    def test_modes_and_reductions(self):
        for name, (problem, ctas) in _problems().items():
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=ctas,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        _check(self, spec)

    def test_default_pool(self):
        # streamk_ctas=-1: one CTA per CU for linear/tree, the occupancy-sized
        # pool (several CTAs per CU) for the reductions that never wait.
        for name in ("dp_plus_sk", "multi_partner"):
            problem, _ = _problems()[name]
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=-1,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        _check(self, spec)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKGrouped(unittest.TestCase):
    """Groups folded into GEMM-M, including a partial last M tile per group."""

    def test_grouped(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        problems = {
            "g3_partial_m": (
                ConvProblem(
                    N=4, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3
                ),
                6,
            ),
            "g8": (
                ConvProblem(
                    N=2, Hi=12, Wi=12, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=8
                ),
                5,
            ),
        }
        for name, (problem, ctas) in problems.items():
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in _REDUCTIONS:
                    spec = _spec(
                        problem,
                        mode=mode,
                        reduction=reduction,
                        dtype_d=dtype_d,
                        ctas=ctas,
                    )
                    with self.subTest(shape=name, mode=mode, reduction=reduction):
                        _check(self, spec)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKAot(unittest.TestCase):
    """One binary per spec serves every launch shape.

    The partition is a kernarg block, so a kernel built for one problem must
    be correct on others -- with a different tile count, iteration count,
    data-parallel/stream-K split and group count.
    """

    def test_one_binary_many_shapes(self):
        from rocke import compile_kernel
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        build = ConvProblem(N=2, Hi=8, Wi=8, C=48, K=96, Y=3, X=3, pH=1, pW=1, groups=3)
        launches = [
            ConvProblem(N=4, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3),
            ConvProblem(N=8, Hi=16, Wi=16, C=24, K=96, Y=3, X=3, pH=1, pW=1, groups=3),
            ConvProblem(
                N=3,
                Hi=13,
                Wi=11,
                C=72,
                K=72,
                Y=3,
                X=3,
                sH=2,
                sW=2,
                pH=1,
                pW=1,
                groups=3,
            ),
        ]
        for mode in ("dp_sk", "persistent"):
            for reduction, dtype_d in _REDUCTIONS:
                spec = _spec(
                    build, mode=mode, reduction=reduction, dtype_d=dtype_d, ctas=6
                )
                art = compile_kernel(
                    build_implicit_gemm_conv_wgrad(spec, arch=GPU_ARCH), arch=GPU_ARCH
                )
                for i, problem in enumerate(launches):
                    with self.subTest(mode=mode, reduction=reduction, shape=i):
                        _check(self, spec, problem=problem, artifact=art)


@unittest.skipUnless(not _SKIP_REASON, _SKIP_REASON or "no GPU")
class TestConvWgradStreamKDeterminism(unittest.TestCase):
    """Linear and tree fold partials in a fixed order: dW is bit-exact."""

    def test_repeat_launches_are_bit_identical(self):
        import torch

        problem, ctas = _problems()["multi_partner"]
        for mode in ("dp_sk", "persistent"):
            for reduction in ("linear", "tree"):
                spec = _spec(
                    problem, mode=mode, reduction=reduction, dtype_d="fp16", ctas=ctas
                )
                with self.subTest(mode=mode, reduction=reduction):
                    outs = _check(self, spec, launches=20)
                    for i, out in enumerate(outs[1:], start=1):
                        self.assertTrue(
                            torch.equal(out, outs[0]),
                            f"launch {i} differs from launch 0",
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
