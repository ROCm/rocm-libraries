# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU correctness for the direct-style native-NCHW wgrad convolution.

:mod:`kernels.common.conv_direct_nchw_wgrad` is a productionized,
C++-mirrored, byte-identity-gated kernel family -- see its module docstring --
so this test checks GPU correctness: the knob space compiles, launches, and
matches ``torch.nn.grad.conv2d_weight`` computed directly in NCHW / NKHW / KCYX
(torch's native layouts, no permute on either side), including image-edge
tiles, partial spatial tiles and split degrees past the tile count.

Tolerance: max |dW - ref| / max(max |ref|, 1) below ``_TOL``, the metric and
fp16 / bf16 bound of ``test_conv_wgrad_correctness.py`` (each split rounds its
partial to 16 bits before the packed atomic add, as there).

The mutation check rebuilds the kernel with the image-edge masking of the X
window removed (``_MUTATIONS``) and requires the edge cases to fail, so the
shapes here do exercise the padding path.

Run as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_direct_nchw_wgrad.py
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

from kernels.common import conv_direct_nchw_wgrad as wgrad_mod  # noqa: E402
from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_direct_nchw_wgrad import (  # noqa: E402
    DirectNchwWgradSpec,
    build_direct_conv_nchw_wgrad,
    is_valid_direct_nchw_wgrad_spec,
    direct_nchw_wgrad_launch_values,
    direct_nchw_wgrad_signature,
)

_TOL = {"fp16": 5e-2, "bf16": 5e-2}  # test_conv_wgrad_correctness.py's _TOL

GPU_ARCH = get_device_arch(0)


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH != "gfx950":
        return f"this family targets gfx950 (got {GPU_ARCH!r})"
    return ""


_GPU_SKIP = _gpu_skip_reason()


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _problem(N, C, K, H, W, stride=1, k=3, dtype="fp16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=1,
        cpg=C,
        kpg=K,
        KH=k,
        KW=k,
        PAD=k // 2,
        stride=stride,
        dtype=dtype,
    )


def run_direct_nchw_wgrad(
    arch: str, spec: DirectNchwWgradSpec, split: int = 1, poison: bool = False
):
    """Compile, launch and check one spec; returns ``(ok, rel_err)``.

    ``poison`` fills dW with NaN instead of zero (plain-store epilogue only:
    an element the kernel misses then fails the check).
    """
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_direct_nchw_wgrad_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    art = compile_kernel(build_direct_conv_nchw_wgrad(spec, arch=arch), arch=arch)

    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    X_t = torch.empty(p.N, p.total_c, p.H, p.W).uniform_(-1.0, 1.0).to(td)
    dY_t = torch.empty(p.N, p.total_k, p.Ho, p.Wo).uniform_(-1.0, 1.0).to(td)
    ref = torch.nn.grad.conv2d_weight(
        X_t.float(),
        (p.total_k, p.cpg, p.KH, p.KW),
        dY_t.float(),
        stride=p.stride,
        padding=p.PAD,
    )
    dW_t = torch.zeros(p.total_k, p.cpg, p.KH, p.KW, dtype=td)

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (dY_t, X_t, dW_t)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(dY_t), dY_t.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(X_t), X_t.nbytes)
        # The atomic epilogue accumulates into zero; the store one overwrites.
        rt.memset(bufs[2], 0xFF if poison else 0, dW_t.nbytes)
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=direct_nchw_wgrad_signature(spec),
        )
        launcher(
            direct_nchw_wgrad_launch_values(
                bufs[0], bufs[1], bufs[2], dY_t.nbytes, X_t.nbytes, split
            ),
            config=LaunchConfig(
                grid=spec.launch_grid(split), block=(spec.threads, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(dW_t)
        rt.memcpy_d2h(_u8(out), bufs[2], dW_t.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)

    out = out.float()
    if not torch.isfinite(out).all():
        return False, float("inf")
    rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < _TOL[p.dtype], rel


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu ok")
class TestDirectNchwWgrad(unittest.TestCase):
    def _check(self, spec, split=1, poison=False):
        with self.subTest(spec=spec.kernel_name(), split=split):
            ok, rel = run_direct_nchw_wgrad(GPU_ARCH, spec, split, poison=poison)
            self.assertTrue(ok, f"{spec.kernel_name()} split{split}: rel_err={rel:.3e}")

    def test_fp16_3x3_shapes(self):
        """Default knobs over stride, batch, channel counts and image size."""
        for stride in (1, 2):
            for N in (1, 2):
                for C, K in ((32, 64), (64, 64)):
                    for HW in (16, 32):
                        p = _problem(N, C, K, HW, HW, stride=stride)
                        spec = DirectNchwWgradSpec(
                            problem=p, tile_m=64, tile_c=32, waves_m=2, waves_c=1
                        )
                        self._check(spec, split=1)

    def test_fp16_knobs(self):
        """Tiles, wave splits, stage group, LDS pad, MFMA order, prefetch and split degree."""
        for stride in (1, 2):
            p = _problem(2, 64, 64, 32, 32, stride=stride)
            variants = [
                dict(),
                dict(tile_h=2, tile_w=16, stage_group=8, lds_pad=0),
                dict(tile_h=1, tile_w=32, tile_m=32, tile_c=32, waves_m=1, waves_c=1),
                dict(tile_h=4, tile_w=8, waves_t=3, waves_c=1),
                dict(tile_h=8, tile_w=16, waves_t=9, waves_m=1, waves_c=1, tile_m=32),
                dict(
                    tile_m=32,
                    tile_c=32,
                    waves_m=1,
                    waves_c=2,
                    y_groups=3,
                    epilogue="direct",
                ),
                dict(
                    tile_h=2,
                    tile_w=64,
                    y_groups=3,
                    waves_t=3,
                    waves_c=1,
                    lds_pad=4,
                    epilogue="direct",
                ),
                dict(mfma_order="ks", waves_per_eu=2),
                dict(prefetch=False),
                dict(epilogue="direct"),
                dict(kmap="2row", lds_pad=16, lds_wpad=2),
                dict(lds_buffers=2, tile_h=2),
                dict(
                    lds_buffers=2,
                    kmap="2row",
                    tile_h=2,
                    lds_pad=8,
                    waves_t=3,
                    waves_c=1,
                ),
                dict(
                    kmap="2row",
                    tile_h=2,
                    tile_w=16,
                    lds_pad=8,
                    lds_wpad=3,
                    prefetch=False,
                ),
                dict(
                    kmap="2row", tile_h=4, tile_w=64, tile_c=32, waves_c=1, lds_wpad=5
                ),
                dict(xcd_remap=False, tile_m=32, waves_m=1, waves_t=3, waves_c=1),
                dict(
                    prefetch=False,
                    waves_per_eu=2,
                    tile_h=2,
                    tile_w=32,
                    waves_t=3,
                    waves_c=1,
                ),
            ]
            for kw in variants:
                spec = DirectNchwWgradSpec(problem=p, **kw)
                for sk in (1, 3, 8):
                    self._check(spec, split=sk)

    def test_fp16_store_epilogue(self):
        """Plain-store epilogue (split 1): dW is poisoned, every element written once."""
        for stride in (1, 2):
            for N, HW in ((1, 16), (2, 32)):
                p = _problem(N, 64, 32, HW, HW, stride=stride)
                for kw in (
                    dict(atomic=False),
                    dict(
                        atomic=False,
                        tile_m=32,
                        waves_m=1,
                        tile_c=32,
                        waves_c=2,
                        tile_h=2,
                    ),
                    dict(atomic=False, kmap="2row", tile_h=2, waves_t=3, waves_c=1),
                ):
                    spec = DirectNchwWgradSpec(
                        problem=p, **{"tile_m": 32, "waves_m": 1, **kw}
                    )
                    self._check(spec, split=1, poison=True)

    def test_fp16_tails_and_nonsquare(self):
        """Partial spatial tiles in h and w, H != W, split past the tile count."""
        for stride in (1, 2):
            # Ho = 12 / 6 rows over 8-row tiles, Wo = 24 / 16 over 32-wide tiles.
            p = _problem(1, 32, 64, 12 * stride, 24 * stride, stride=stride)
            spec = DirectNchwWgradSpec(
                problem=p,
                tile_m=64,
                tile_c=32,
                tile_h=8,
                tile_w=32,
                waves_m=2,
                waves_c=1,
            )
            for sk in (1, 2, 64):
                self._check(spec, split=sk)
        # Odd input height at stride 2: Ho = 17 over 4-row tiles.
        p = DirectConvProblem(
            N=2, H=33, W=48, groups=1, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=2
        )
        spec = DirectNchwWgradSpec(
            problem=p, tile_m=32, tile_c=32, tile_h=4, tile_w=16, waves_m=1, waves_c=2
        )
        self._check(spec, split=5)

    def test_fp16_unaligned_wo(self):
        """``W = 48`` at stride 2 (Wo 24): the right edge of a 32-wide tile."""
        for stride in (1, 2):
            p = _problem(2, 32, 32, 20, 24 * stride, stride=stride)
            spec = DirectNchwWgradSpec(
                problem=p,
                tile_m=32,
                tile_c=32,
                tile_h=2,
                tile_w=32,
                waves_m=1,
                waves_c=2,
            )
            self._check(spec, split=4)

    def test_fp16_5x5(self):
        for stride in (1, 2):
            p = _problem(1, 32, 32, 32, 32, stride=stride, k=5)
            for kw in (
                dict(tile_m=32, tile_c=32, waves_m=1, waves_c=1, waves_t=5),
                dict(tile_m=16, tile_c=16, waves_m=1, waves_c=1, tile_h=2, tile_w=16),
                dict(
                    tile_m=32,
                    tile_c=16,
                    waves_m=2,
                    waves_c=1,
                    y_groups=5,
                    epilogue="direct",
                ),
            ):
                spec = DirectNchwWgradSpec(problem=p, **kw)
                self._check(spec, split=4)

    def test_bf16(self):
        for stride in (1, 2):
            p = _problem(2, 64, 64, 32, 32, stride=stride, dtype="bf16")
            spec = DirectNchwWgradSpec(problem=p)
            self._check(spec, split=4)
            self._check(
                replace(
                    spec, tile_m=32, tile_c=32, waves_m=1, y_groups=3, epilogue="direct"
                ),
                split=1,
            )

    def test_mutation_edge_masking_is_exercised(self):
        """With the X window's image-edge masking removed the edge cases fail."""
        cases = []
        for stride in (1, 2):
            p = _problem(2, 32, 32, 16, 32, stride=stride)
            cases.append(
                DirectNchwWgradSpec(
                    problem=p,
                    tile_m=32,
                    tile_c=32,
                    tile_h=2,
                    tile_w=16,
                    waves_m=1,
                    waves_c=2,
                )
            )
        saved = wgrad_mod._MUTATIONS
        try:
            for mut in ("halo_unmasked", "rows_unmasked"):
                wgrad_mod._MUTATIONS = frozenset({mut})
                fails = 0
                for spec in cases:
                    ok, _rel = run_direct_nchw_wgrad(
                        GPU_ARCH,
                        replace(spec, name=f"wgrad_direct_nchw_mut_{mut}"),
                        split=2,
                    )
                    fails += not ok
                with self.subTest(mutation=mut):
                    self.assertGreater(fails, 0, f"mutation {mut} went undetected")
        finally:
            wgrad_mod._MUTATIONS = saved
        for spec in cases:  # and the unmutated build still passes
            self._check(spec, split=2)


if __name__ == "__main__":
    unittest.main()
