# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness for the NCHW direct-conv prototype (experimental spike).

:mod:`kernels.common.conv_direct_nongrouped_nchw_proto` is not a production
kernel family -- see its module docstring -- so this test only checks what a
spike needs: it compiles, it launches, and its output matches an IEEE conv2d
reference computed directly in NCHW (no host-side transpose on either side,
since NCHW is torch's native layout).

Run as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_direct_nongrouped_nchw_proto.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py / test_conv_direct_nongrouped_tf32.py).
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_direct_nongrouped_nchw_proto import (  # noqa: E402
    DirectNongroupedConvNchwProtoSpec,
    build_direct_conv_nongrouped_nchw_proto,
    is_valid_nchw_proto_spec,
)

_ARCHES = ("gfx950", "gfx942")
_TOL = {"fp16": 5e-3, "bf16": 1.5e-2}

GPU_ARCH = get_device_arch(0)


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in _ARCHES:
        return f"unsupported arch {GPU_ARCH!r}"
    return ""


_GPU_SKIP = _gpu_skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _reference(X_t, W_t, p: DirectConvProblem):
    """IEEE fp32 conv2d on the CPU, native NCHW (no permutes)."""
    import torch.nn.functional as F

    with torch.no_grad():
        out = F.conv2d(
            X_t.float(),
            W_t.float(),
            padding=p.PAD,
            stride=p.stride,
            dilation=(p.dil_h, p.dil_w),
        )
    return out


def run_nchw_proto_spec(
    arch: str, spec: DirectNongroupedConvNchwProtoSpec, artifact=None
):
    """Compile, launch and check one spec; returns ``(ok, rel_err)``.

    ``X`` is ``[N, C, H, W]``, ``W`` is ``[K, C, KH, KW]``, ``D`` is
    ``[N, K, Ho, Wo]`` -- NCHW / KCYX / NKHW, all native, no transpose.
    With ``spec.weight_layout == "krsc"`` the weights are pre-packed on the
    host into ``[K, KH, KW, C]`` before upload; the reference still uses KCYX.
    """
    from rocke import compile_kernel
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_nchw_proto_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    if artifact is None:
        artifact = compile_kernel(
            build_direct_conv_nongrouped_nchw_proto(spec, arch=arch), arch=arch
        )

    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    X_t = torch.empty(p.N, p.total_c, p.H, p.W, dtype=td).uniform_(-1.0, 1.0)
    W_t = torch.empty(p.total_k, p.cpg, p.KH, p.KW, dtype=td).uniform_(-1.0, 1.0)
    D_t = torch.empty(p.N, p.total_k, p.Ho, p.Wo, dtype=td)
    ref = _reference(X_t, W_t, p)
    W_dev = (
        W_t.permute(0, 2, 3, 1).contiguous() if spec.weight_layout == "krsc" else W_t
    )

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (X_t, W_dev, D_t)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(X_t), X_t.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(W_dev), W_dev.nbytes)
        # NaN fill: an output element the kernel never writes cannot pass as 0.
        D_t.fill_(float("nan"))
        rt.memcpy_h2d(bufs[2], _u8(D_t), D_t.nbytes)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(p.dtype),
        )
        launcher(
            ConvArgs.from_problem(p).to_launch_values(
                int(bufs[0]),
                int(bufs[1]),
                int(bufs[2]),
                X_t.nbytes,
                W_t.nbytes,
                D_t.nbytes,
            ),
            config=LaunchConfig(
                grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(D_t)
        rt.memcpy_d2h(_u8(out), bufs[2], D_t.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)
    if not torch.isfinite(out.float()).all():
        return False, float("inf")
    rel = float((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < _TOL[p.dtype], rel


def _spec(p: DirectConvProblem, arch: str, **kw) -> DirectNongroupedConvNchwProtoSpec:
    """A small spec, narrowed to fit ``arch``'s LDS if the defaults do not."""
    from dataclasses import replace

    from rocke.core.arch import ArchTarget

    base = dict(
        tile_h=8, tile_w=32, tile_k=32, ck=16, waves_m=1, waves_n=2, atom="32x32x16"
    )
    base.update(kw)
    spec = DirectNongroupedConvNchwProtoSpec(problem=p, **base)
    target = ArchTarget.from_gfx(arch)
    while not target.fits_lds(spec.lds_bytes) and spec.ck // 2 >= spec.atom_k:
        spec = replace(spec, ck=spec.ck // 2)
    return spec


class TestNchwProtoSpec(unittest.TestCase):
    """Host-side: validation accepts a plain shape and rejects bad dtype/groups."""

    def _problem(self, **kw):
        base = dict(N=1, H=16, W=16, groups=1, cpg=32, kpg=32, dtype="fp16")
        base.update(kw)
        return DirectConvProblem(**base)

    def test_valid_fp16_and_bf16(self):
        for dtype in ("fp16", "bf16"):
            p = self._problem(dtype=dtype)
            spec = DirectNongroupedConvNchwProtoSpec(
                problem=p, tile_h=8, tile_w=32, tile_k=32, ck=16, waves_m=1, waves_n=2
            )
            ok, why = is_valid_nchw_proto_spec(spec, "gfx950")
            self.assertTrue(ok, why)

    def test_fp32_storage_rejected(self):
        p = self._problem(dtype="fp32")
        spec = DirectNongroupedConvNchwProtoSpec(problem=p)
        ok, why = is_valid_nchw_proto_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("unsupported dtype", why)

    def test_grouped_rejected(self):
        p = self._problem(groups=2, cpg=16, kpg=16)
        spec = DirectNongroupedConvNchwProtoSpec(problem=p)
        ok, why = is_valid_nchw_proto_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("groups==1", why)

    def test_build_and_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        p = self._problem()
        spec = _spec(p, "gfx950")
        ll = lower_kernel_to_llvm(
            build_direct_conv_nongrouped_nchw_proto(spec, arch="gfx950"), arch="gfx950"
        )
        self.assertIn("llvm.amdgcn.mfma", ll)


@unittest.skipUnless(not _GPU_SKIP, _GPU_SKIP or "no GPU")
class TestNchwProtoGpu(unittest.TestCase):
    """On silicon: output against torch's native-NCHW conv2d reference."""

    def test_3x3_stride1_pad1(self):
        shapes = [
            (1, 16, 32, 32, 32),
            (2, 24, 24, 64, 64),
            (1, 14, 18, 32, 64),
        ]
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in shapes:
                p = DirectConvProblem(
                    N=N,
                    H=H,
                    W=W,
                    groups=1,
                    cpg=C,
                    kpg=K,
                    KH=3,
                    KW=3,
                    PAD=1,
                    stride=1,
                    dtype=dtype,
                )
                spec = _spec(p, GPU_ARCH)
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K)):
                    ok, rel = run_nchw_proto_spec(GPU_ARCH, spec)
                    print(
                        f"  {'PASS' if ok else 'FAIL'} {dtype} "
                        f"N{N}H{H}W{W}C{C}K{K} s1p1 rel={rel:.2e}",
                        flush=True,
                    )
                    self.assertTrue(
                        ok, f"N{N}H{H}W{W}C{C}K{K} {dtype}: rel_err {rel:.3e}"
                    )

    def test_3x3_stride2_pad1(self):
        shapes = [
            (1, 16, 32, 32, 32),
            (2, 24, 24, 64, 64),
            (1, 15, 19, 32, 64),
        ]
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in shapes:
                p = DirectConvProblem(
                    N=N,
                    H=H,
                    W=W,
                    groups=1,
                    cpg=C,
                    kpg=K,
                    KH=3,
                    KW=3,
                    PAD=1,
                    stride=2,
                    dtype=dtype,
                )
                spec = _spec(p, GPU_ARCH)
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K)):
                    ok, rel = run_nchw_proto_spec(GPU_ARCH, spec)
                    print(
                        f"  {'PASS' if ok else 'FAIL'} {dtype} "
                        f"N{N}H{H}W{W}C{C}K{K} s2p1 rel={rel:.2e}",
                        flush=True,
                    )
                    self.assertTrue(
                        ok, f"N{N}H{H}W{W}C{C}K{K} {dtype}: rel_err {rel:.3e}"
                    )

    def test_krsc_weights(self):
        """Pre-packed K,KH,KW,C weights: vector W staging, NCHW in / NKHW out."""
        shapes = [
            (1, 16, 32, 32, 32),
            (2, 24, 24, 64, 64),
            (1, 15, 19, 32, 64),
        ]
        for dtype in ("fp16", "bf16"):
            for stride in (1, 2):
                for N, H, W, C, K in shapes:
                    p = DirectConvProblem(
                        N=N,
                        H=H,
                        W=W,
                        groups=1,
                        cpg=C,
                        kpg=K,
                        KH=3,
                        KW=3,
                        PAD=1,
                        stride=stride,
                        dtype=dtype,
                    )
                    spec = _spec(p, GPU_ARCH, weight_layout="krsc")
                    with self.subTest(
                        dtype=dtype, stride=stride, shape=(N, H, W, C, K)
                    ):
                        ok, rel = run_nchw_proto_spec(GPU_ARCH, spec)
                        self.assertTrue(
                            ok,
                            f"N{N}H{H}W{W}C{C}K{K} s{stride} {dtype}: rel_err {rel:.3e}",
                        )

    def test_tr_read_and_direct_epilogue(self):
        """KRSC weights + chw_tr X staging (gfx950) and/or the operand-swapped
        direct epilogue, against the same NCHW reference."""
        variants = [("pos_c", "direct")]
        if GPU_ARCH == "gfx950":
            variants += [("chw_tr", "lds_transpose"), ("chw_tr", "direct")]
        cases = [
            (1, 16, 32, 32, 64, 3, 1, 1),
            (2, 15, 19, 32, 64, 3, 1, 1),
            (1, 16, 32, 32, 64, 3, 1, 2),
            (2, 15, 19, 32, 64, 3, 1, 2),
            (1, 13, 21, 32, 64, 5, 2, 1),
            (1, 13, 21, 32, 64, 5, 2, 2),
        ]
        for xl, ep in variants:
            for dtype in ("fp16", "bf16"):
                for N, H, W, C, K, F, P, stride in cases:
                    p = DirectConvProblem(
                        N=N,
                        H=H,
                        W=W,
                        groups=1,
                        cpg=C,
                        kpg=K,
                        KH=F,
                        KW=F,
                        PAD=P,
                        stride=stride,
                        dtype=dtype,
                    )
                    spec = _spec(
                        p,
                        GPU_ARCH,
                        weight_layout="krsc",
                        x_lds_layout=xl,
                        epilogue=ep,
                        tile_h=4,
                        tile_k=64,
                        waves_m=2,
                        waves_n=4,
                    )
                    tag = f"{xl}/{ep} {dtype} N{N}H{H}W{W}C{C}K{K} {F}x{F}s{stride}"
                    with self.subTest(tag):
                        ok, rel = run_nchw_proto_spec(GPU_ARCH, spec)
                        self.assertTrue(ok, f"{tag}: rel_err {rel:.3e}")

    def test_one_binary_serves_every_shape(self):
        """AOT: the compiled binary only depends on N/H/W/C/K through kernargs."""
        from rocke import compile_kernel

        p0 = DirectConvProblem(N=1, H=16, W=16, groups=1, cpg=32, kpg=32, dtype="fp16")
        spec = _spec(p0, GPU_ARCH)
        art = compile_kernel(
            build_direct_conv_nongrouped_nchw_proto(spec, arch=GPU_ARCH), arch=GPU_ARCH
        )
        for N, H, W, C, K in [
            (1, 16, 16, 32, 32),
            (2, 20, 24, 32, 64),
            (1, 17, 13, 32, 32),
        ]:
            p = DirectConvProblem(N=N, H=H, W=W, groups=1, cpg=C, kpg=K, dtype="fp16")
            case = type(spec)(**{**spec.__dict__, "problem": p})
            with self.subTest(shape=(N, H, W, C, K)):
                ok, rel = run_nchw_proto_spec(GPU_ARCH, case, artifact=art)
                self.assertTrue(ok, f"rel_err {rel:.3e}")


if __name__ == "__main__":
    unittest.main()
