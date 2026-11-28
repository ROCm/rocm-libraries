# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness for the native-NCHW forward direct convolution family.

:mod:`kernels.common.conv_direct_nchw_fwd` is AOT-buildable and C++-mirrored
(see its module docstring) but not yet wired into runtime dispatch. This test
checks that it compiles, launches and matches ``torch.nn.functional.conv2d``
computed directly in NCHW
(X NCHW, W KCYX, Y NKHW -- no host transpose of the activations; the weights
are repacked once on the host, as a plan-time step would). It also runs the
native-NCHW dgrad family (:mod:`kernels.common.conv_direct_nchw_dgrad`) as a
stride-1 forward conv (weights flipped and transposed, pad ``KH-1-PAD``), the
mapping the forward benchmark uses.

Tolerance: max |Y - ref| / max(|ref|, 1) below 5e-3 (fp16) / 1.5e-2 (bf16).

Run as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_direct_nchw_fwd.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_direct_nchw_fwd import (  # noqa: E402
    DirectNchwFwdSpec,
    build_direct_conv_nchw_fwd,
    is_valid_direct_nchw_fwd_spec,
    repack_weights_krsc,
    with_conflict_free_pad,
    x_read_conflict_ways,
)

_ARCHES = ("gfx950", "gfx942")
TOL = {"fp16": 5e-3, "bf16": 1.5e-2}

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


def _launch(artifact, p, block, grid, A, B, D_shape, td):
    """Launch a direct-ABI kernel on host tensors ``A``/``B``; returns D."""
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    D = torch.full(D_shape, float("nan"), dtype=td)  # unwritten != 0
    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (A, B, D)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(B), B.nbytes)
        rt.memcpy_h2d(bufs[2], _u8(D), D.nbytes)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=conv_direct_args_signature(p.dtype),
        )
        launcher(
            ConvArgs.from_problem(p).to_launch_values(
                int(bufs[0]), int(bufs[1]), int(bufs[2]), A.nbytes, B.nbytes, D.nbytes
            ),
            config=LaunchConfig(grid=grid, block=(block, 1, 1), fence=True),
        )
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), bufs[2], D.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)
    return out


def _data(p: DirectConvProblem):
    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    X = torch.empty(p.N, p.total_c, p.H, p.W, dtype=td).uniform_(-1.0, 1.0)
    W = torch.empty(p.total_k, p.total_c, p.KH, p.KW, dtype=td).uniform_(-1.0, 1.0)
    with torch.no_grad():
        ref = torch.nn.functional.conv2d(
            X.float(), W.float(), stride=p.stride, padding=p.PAD
        )
    return td, X, W, ref


def _rel(out, ref, dtype):
    if not torch.isfinite(out.float()).all():
        return False, float("inf")
    rel = float((out.float() - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < TOL[dtype], rel


def run_fwd_nchw_spec(arch: str, spec: DirectNchwFwdSpec, artifact=None):
    """Compile, launch and check one spec; returns ``(ok, rel_err)``."""
    from rocke import compile_kernel

    p = spec.problem
    ok, why = is_valid_direct_nchw_fwd_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    if artifact is None:
        artifact = compile_kernel(build_direct_conv_nchw_fwd(spec, arch=arch), arch=arch)
    td, X, W, ref = _data(p)
    out = _launch(
        artifact,
        p,
        spec.threads_per_block,
        spec.grid(),
        X,
        repack_weights_krsc(W),
        (p.N, p.total_k, p.Ho, p.Wo),
        td,
    )
    return _rel(out, ref, p.dtype)


def dgrad_as_fwd_problem(p: DirectConvProblem) -> DirectConvProblem:
    """The stride-1 forward ``p`` as the dgrad prototype's problem: the dgrad
    ``dX`` is ``Y`` (``K`` channels), ``dY`` is ``X`` (``C``), pad
    ``KH-1-PAD``."""
    return DirectConvProblem(
        N=p.N,
        H=p.Ho,
        W=p.Wo,
        groups=1,
        cpg=p.total_k,
        kpg=p.total_c,
        KH=p.KH,
        KW=p.KW,
        PAD=p.KH - 1 - p.PAD,
        stride=1,
        dtype=p.dtype,
    )


def dgrad_as_fwd_weights(W_kcyx):
    """Forward KCYX weights in the dgrad prototype's ``[C_d][KH][KW][K_d]`` order
    for :func:`dgrad_as_fwd_problem`: flipped taps, ``[K][KH][KW][C]``."""
    return W_kcyx.flip(2, 3).permute(0, 2, 3, 1).contiguous()


def _problem(N, H, W, C, K, F=3, P=1, stride=1, dtype="fp16"):
    return DirectConvProblem(
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


def _spec(p, **kw):
    base = dict(tile_i=8, tile_j=32, tile_k=64, ck=32, waves_m=2, waves_n=2)
    if p.stride == 2:
        base.update(tile_i=4)
    base.update(kw)
    return DirectNchwFwdSpec(problem=p, **base)


class TestFwdNchwProtoHost(unittest.TestCase):
    def test_validation(self):
        ok, _ = is_valid_direct_nchw_fwd_spec(_spec(_problem(1, 16, 16, 64, 64)))
        self.assertTrue(ok)
        ok, why = is_valid_direct_nchw_fwd_spec(_spec(_problem(1, 16, 12, 64, 64)))
        self.assertFalse(ok)
        self.assertIn("W 12", why)
        ok, _ = is_valid_direct_nchw_fwd_spec(
            _spec(_problem(1, 16, 16, 64, 64, dtype="fp32"))
        )
        self.assertFalse(ok)
        ok, why = is_valid_direct_nchw_fwd_spec(
            _spec(_problem(1, 16, 16, 64, 64), col_split=True)
        )
        self.assertFalse(ok)
        self.assertIn("col_split", why)

    def test_window_geometry(self):
        # 3x3 pad 1: stride 1 needs a 1-column halo on each side; stride 2
        # only the left one (the last tap of the last output lands in the body).
        s1 = _spec(_problem(1, 32, 32, 64, 64))
        self.assertEqual((s1.halo_l, s1.halo_r, s1.lds_w, s1.lds_h), (1, 1, 34, 10))
        s2 = _spec(_problem(1, 32, 64, 64, 64, stride=2))
        self.assertEqual((s2.halo_l, s2.halo_r, s2.lds_w, s2.lds_h), (1, 0, 65, 9))
        s5 = _spec(_problem(1, 32, 64, 64, 64, F=5, P=2, stride=2))
        self.assertEqual((s5.halo_l, s5.halo_r), (2, 1))

    def test_conflict_model(self):
        # ck 32 + pad 8 = 5 16-byte units per position: odd, conflict free at
        # stride 1. Stride 2 doubles the lane stride (2-way) unless the
        # columns are split into even/odd planes.
        s1 = _spec(_problem(1, 32, 32, 64, 64))
        self.assertEqual(x_read_conflict_ways(s1), 1)
        self.assertGreater(x_read_conflict_ways(replace(s1, lds_pad=0)), 1)
        p2 = _problem(1, 32, 64, 64, 64, stride=2)
        self.assertEqual(x_read_conflict_ways(_spec(p2)), 2)
        self.assertEqual(x_read_conflict_ways(_spec(p2, col_split=True)), 1)
        self.assertEqual(with_conflict_free_pad(replace(s1, lds_pad=0)).lds_pad, 8)

    def test_build_and_lower(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        for stride, kw in ((1, {}), (2, {}), (2, dict(col_split=True))):
            spec = _spec(
                _problem(1, 32, 32, 64, 64, stride=stride),
                waves_per_eu=2,
                pin_prefetch=True,
                **kw,
            )
            ll = lower_kernel_to_llvm(
                build_direct_conv_nchw_fwd(spec, "gfx950"), arch="gfx950"
            )
            self.assertIn("llvm.amdgcn.mfma", ll)
            self.assertIn("amdgpu-waves-per-eu", ll)
            self.assertIn("llvm.amdgcn.sched.barrier", ll)


@unittest.skipUnless(not _GPU_SKIP, _GPU_SKIP or "no GPU")
class TestFwdNchwProtoGpu(unittest.TestCase):
    """On silicon: Y against torch's native-NCHW ``conv2d``."""

    def _check(self, p, **kw):
        spec = _spec(p, **kw)
        ok, rel = run_fwd_nchw_spec(GPU_ARCH, spec)
        tag = f"{spec.kernel_name()} N{p.N}H{p.H}W{p.W}C{p.total_c}K{p.total_k}"
        print(f"  {'PASS' if ok else 'FAIL'} {tag} rel={rel:.2e}", flush=True)
        self.assertTrue(ok, f"{tag}: rel_err {rel:.3e}")

    def test_3x3_stride1(self):
        # Edge tiles in h (Ho % tile_i) and w (Wo % tile_j), C != K, N > 1.
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in [
                (1, 16, 32, 64, 64),
                (2, 20, 40, 64, 96),
                (1, 9, 24, 128, 32),
            ]:
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K)):
                    self._check(_problem(N, H, W, C, K, stride=1, dtype=dtype))

    def test_3x3_stride2(self):
        # Even and odd H, Wo not a multiple of the tile, both column layouts.
        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K in [
                (1, 16, 16, 64, 64),
                (2, 23, 48, 64, 96),
                (1, 32, 80, 128, 32),
            ]:
                for cs in (False, True):
                    with self.subTest(dtype=dtype, shape=(N, H, W, C, K), cs=cs):
                        self._check(
                            _problem(N, H, W, C, K, stride=2, dtype=dtype),
                            col_split=cs,
                        )

    def test_5x5_and_padding(self):
        # 5x5 pad 2 (2-column left halo, 1-column right halo at stride 2),
        # pad 0 (no left halo) and 1x1.
        cases = []
        for stride in (1, 2):
            for N, H, W, C, K in [(1, 16, 32, 64, 64), (2, 18, 48, 64, 32)]:
                cases.append((_problem(N, H, W, C, K, F=5, P=2, stride=stride), {}))
            cases.append((_problem(1, 17, 32, 64, 64, F=3, P=0, stride=stride), {}))
            cases.append((_problem(1, 16, 32, 64, 64, F=1, P=0, stride=stride), {}))
        cases.append(
            (_problem(1, 18, 48, 64, 32, F=5, P=2, stride=2), dict(col_split=True))
        )
        cases.append(
            (_problem(1, 16, 48, 64, 32, F=5, P=2, stride=2, dtype="bf16"), {})
        )
        for p, kw in cases:
            with self.subTest(F=p.KH, P=p.PAD, stride=p.stride, dtype=p.dtype, **kw):
                self._check(p, **kw)

    def test_knobs(self):
        """Every staging group, both atom sizes, pads and the occupancy levers."""
        cases = []
        for stride in (1, 2):
            p = _problem(2, 24, 64, 64, 64, stride=stride)
            for g in (1, 2, 4, 8):
                cases.append((p, dict(stage_group=g)))
            cases.append((p, dict(atom="16x16x32", tile_j=32, waves_m=2, waves_n=2)))
            cases.append((p, dict(atom="16x16x16", ck=16)))
            cases.append((p, dict(tile_j=64, tile_i=2, waves_n=1, waves_m=2, ck=16)))
            cases.append((p, dict(lds_pad=0, lds_wpad=3)))
            cases.append((p, dict(waves_per_eu=2, pin_prefetch=True)))
        for p, kw in cases:
            with self.subTest(stride=p.stride, **kw):
                self._check(p, **kw)

    def test_one_binary_serves_every_shape(self):
        """The binary depends on N/H/W/C/K only through kernargs."""
        from rocke import compile_kernel

        for stride in (1, 2):
            spec0 = _spec(_problem(1, 16, 32, 64, 64, stride=stride))
            art = compile_kernel(
                build_direct_conv_nchw_fwd(spec0, GPU_ARCH), arch=GPU_ARCH
            )
            for N, H, W, C, K in [(2, 32, 32, 128, 64), (1, 17, 48, 64, 128)]:
                case = replace(spec0, problem=_problem(N, H, W, C, K, stride=stride))
                with self.subTest(stride=stride, shape=(N, H, W, C, K)):
                    ok, rel = run_fwd_nchw_spec(GPU_ARCH, case, artifact=art)
                    self.assertTrue(ok, f"rel_err {rel:.3e}")

    def test_dgrad_kernel_as_stride1_forward(self):
        """The dgrad prototype computes a stride-1 forward conv given the
        flipped, transposed weights and pad ``KH-1-PAD``."""
        from kernels.common.conv_direct_nchw_dgrad import (
            DirectNchwDgradSpec,
            build_direct_conv_nchw_dgrad,
        )
        from rocke import compile_kernel

        for dtype in ("fp16", "bf16"):
            for N, H, W, C, K, F, P in [
                (1, 16, 32, 64, 64, 3, 1),
                (2, 20, 40, 64, 96, 3, 1),
                (1, 16, 32, 64, 32, 5, 2),
            ]:
                p = _problem(N, H, W, C, K, F=F, P=P, dtype=dtype)
                dp = dgrad_as_fwd_problem(p)
                spec = DirectNchwDgradSpec(
                    problem=dp, tile_i=8, tile_j=32, tile_c=64, ck=32, stage_group=4
                )
                art = compile_kernel(
                    build_direct_conv_nchw_dgrad(spec, GPU_ARCH), arch=GPU_ARCH
                )
                td, X, Wk, ref = _data(p)
                out = _launch(
                    art,
                    dp,
                    spec.threads_per_block,
                    spec.grid(),
                    X,
                    dgrad_as_fwd_weights(Wk),
                    (p.N, p.total_k, p.Ho, p.Wo),
                    td,
                )
                ok, rel = _rel(out, ref, dtype)
                with self.subTest(dtype=dtype, shape=(N, H, W, C, K, F, P)):
                    self.assertTrue(ok, f"rel_err {rel:.3e}")


if __name__ == "__main__":
    unittest.main()
