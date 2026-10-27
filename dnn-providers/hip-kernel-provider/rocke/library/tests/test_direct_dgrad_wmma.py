# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA direct dgrad (``conv_direct_dgrad_wmma``).

CPU: the spec validator, the AOT grid (gfx1250 only) and the shape filter of
``direct_spec_for_problem``. GPU (gfx1250): every spec branch -- edges and
ragged tiles, groups, PAD 0 / 5x5, several images per block, a 3-deep ring,
two WMMA k-steps per chunk, fragment prefetch, the upper VGPR banks -- checked
numerically against torch's dgrad.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_dgrad_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

import pytest
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_dgrad_wmma import (
    DirectDgradWmmaSpec,
    build_direct_dgrad_wmma,
    dgrad_wmma_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=13, W=21, C=64, K=96, groups=1, KH=3, PAD=1, dtype="bf16"):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=1,
        dtype=dtype,
    )


class TestDgradWmmaSpec(unittest.TestCase):
    def test_wave32_tdm_only(self):
        spec = DirectDgradWmmaSpec(problem=_problem())
        self.assertIsNone(dgrad_wmma_reason(spec, "gfx1250"))
        self.assertIn("wave32", dgrad_wmma_reason(spec, "gfx950"))

    def test_rejects(self):
        for problem, knobs, needle in (
            (_problem(), dict(ck=48), "ck 48"),
            (_problem(C=36), {}, "multiples of 8"),
            (_problem(), dict(tile_w=12), "tile_w"),
            (_problem(), dict(waves_c=1, waves_p=1), "wave"),
            (_problem(), dict(tile_w=8, tile_n=8), "tile_n"),
            (_problem(), dict(waves_c=1, waves_p=2, tiles_c=8, tiles_p=8), "VGPRs"),
            (_problem(), dict(tiles_c=8, tiles_p=4, waves_c=4, ck=64), "LDS"),
        ):
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)
        p = DirectConvProblem(
            N=1, H=8, W=8, groups=1, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=2
        )
        self.assertIn("stride", dgrad_wmma_reason(DirectDgradWmmaSpec(problem=p)))

    def test_rejects_bad_knobs_without_raising(self):
        """Bad knobs are reasons, not exceptions (validate_direct_spec only
        catches ValueError)."""
        for knobs, needle in (
            (dict(tile_n=0), "tile_n"),
            (dict(tile_w=0), "tile_w"),
            (dict(tile_w=-16), "tile_w"),
            (dict(sched="no-such-strategy"), "scheduler_strategy"),
            (dict(sched=""), "scheduler_strategy"),
        ):
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=_problem(), **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_rejects_i32_overflow(self):
        """Global byte offsets are i32: a tensor of 2 GiB or more is refused."""
        for pkw in (
            dict(N=8, H=768, W=768, C=256, K=256),  # dX and dY 2.4 GB
            dict(N=1, H=8, W=8, C=4096, K=32768),  # W 2.4 GB
        ):
            why = dgrad_wmma_reason(DirectDgradWmmaSpec(problem=_problem(**pkw)))
            self.assertIsNotNone(why, pkw)
            self.assertIn("i32", why)
        self.assertIsNone(
            dgrad_wmma_reason(
                DirectDgradWmmaSpec(problem=_problem(N=2, H=512, W=512, C=256, K=256))
            )
        )

    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over the AOT grid and a few rejected specs (the byte-identity
        gate only covers the IR of specs both engines build)."""
        eng = pytest.importorskip("rocke_engine")
        from kernels.conv.direct_aot_jobs import DGRAD_WMMA_KNOBS
        from rocke.core.backend import conv_direct_dgrad_wmma_spec_to_dict

        specs = []
        for wc, wp, tc, tp, tw, tn, ck, st, pf in DGRAD_WMMA_KNOBS:
            knobs = dict(
                waves_c=wc,
                waves_p=wp,
                tiles_c=tc,
                tiles_p=tp,
                tile_w=tw,
                tile_n=tn,
                ck=ck,
                stages=st,
            )
            if pf:
                knobs.update(prefetch=True, sched="iterative-ilp")
            specs.append(DirectDgradWmmaSpec(problem=_problem(), **knobs))
        for pkw, knobs in (
            ({}, dict(ck=48)),
            ({}, dict(tile_n=0)),
            ({}, dict(sched="bogus")),
            (dict(C=36), {}),
            (dict(N=8, H=768, W=768, C=256, K=256), {}),
            (dict(KH=3, PAD=3), {}),
        ):
            specs.append(DirectDgradWmmaSpec(problem=_problem(**pkw), **knobs))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, _, name, grid, lds = eng.conv_direct_dgrad_wmma_describe(
                    conv_direct_dgrad_wmma_spec_to_dict(spec), arch
                )
                why = dgrad_wmma_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed from its
        zero tile extents); a missing, mistyped or partial problem raises."""
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import conv_direct_dgrad_wmma_spec_to_dict

        for knobs in (dict(waves_c=0), dict(tiles_c=0), dict(waves_p=0)):
            sd = conv_direct_dgrad_wmma_spec_to_dict(
                DirectDgradWmmaSpec(problem=_problem(), **knobs)
            )
            ok, why, _, grid, _ = eng.conv_direct_dgrad_wmma_describe(sd, "gfx1250")
            self.assertFalse(ok, knobs)
            self.assertIn("waves and tiles", why)
            self.assertEqual(tuple(grid), (0, 0, 0))
        good = conv_direct_dgrad_wmma_spec_to_dict(
            DirectDgradWmmaSpec(problem=_problem())
        )
        partial = dict(good["problem"])
        del partial["kpg"]
        for problem in (None, "N2H13", partial):
            sd = dict(good)
            if problem is None:
                del sd["problem"]
            else:
                sd["problem"] = problem
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_dgrad_wmma_describe(sd, "gfx1250")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_dgrad_wmma_lower_llvm(sd, arch="gfx1250")

    def test_ir_is_shape_independent(self):
        """One binary serves every extent, group count and channel count."""
        from rocke.core.ir_serialize import serialize

        a = build_direct_dgrad_wmma(DirectDgradWmmaSpec(problem=_problem()))
        b = build_direct_dgrad_wmma(
            DirectDgradWmmaSpec(problem=_problem(N=7, H=40, W=9, C=256, K=32, groups=2))
        )
        self.assertEqual(serialize(a), serialize(b))

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DGRAD_WMMA_KNOBS,
            DIRECT_CAPABILITIES,
            _caps_jobs,
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_dgrad_wmma"]
        self.assertEqual(direction, "dgrad")
        for arch, wave, want in (
            ("gfx1250", 32, len(DGRAD_WMMA_KNOBS)),
            ("gfx950", 64, 0),
        ):
            for caps in caps_list:
                _, jobs = _caps_jobs(
                    arch, wave, "direct_dgrad_wmma", "dgrad", caps, "bf16"
                )
                self.assertEqual(len(jobs), want, (arch, caps))

    def test_spec_for_problem_filters_padding(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = dict(waves_c=2, waves_p=4, tiles_c=4, tiles_p=8, tile_w=16)
        spec, why = direct_spec_for_problem(
            "direct_dgrad_wmma", _problem(N=1, H=64, W=64, C=128), knobs, "gfx1250"
        )
        self.assertIsNotNone(spec, why)
        spec, why = direct_spec_for_problem(
            "direct_dgrad_wmma", _problem(N=1, H=8, W=8, C=32), knobs, "gfx1250"
        )
        self.assertIsNone(spec)
        self.assertIn("tile efficiency", why)


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_CASES = (
    # (problem kwargs, spec knobs)
    (dict(), dict(waves_c=1, waves_p=2, tiles_c=1, tiles_p=1)),
    (dict(dtype="fp16"), dict()),
    (dict(N=1, H=18, W=33, C=256, K=128), dict(tiles_c=2, tiles_p=2, stages=3)),
    (dict(H=8, W=8, groups=2), dict(tiles_c=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=32, K=40, PAD=0), dict(tiles_c=1, tiles_p=2, tile_w=8)),
    (dict(N=3, H=7, W=9, C=32, K=40, KH=5, PAD=2), dict(tiles_c=1, tiles_p=2)),
    (
        dict(N=5, H=7, W=9, C=64, K=64, dtype="fp16"),
        dict(tiles_c=2, tiles_p=2, tile_w=8, tile_n=2, stages=3),
    ),
    (
        dict(N=3, H=6, W=6, C=32, K=96),
        dict(waves_c=1, waves_p=8, tiles_c=1, tiles_p=2, tile_w=4, tile_n=4),
    ),
    (
        dict(N=1, H=20, W=40, C=128, K=160),
        dict(waves_c=1, waves_p=4, tiles_c=4, tiles_p=4, ck=64),
    ),
    # Grouped, cpg 48 over 32-channel blocks: the weight load and the dX
    # store clip at the group boundary.
    (dict(N=2, H=9, W=12, C=96, K=64, groups=2), dict(tiles_c=1, tiles_p=2)),
    # PAD = K - 1: no halo cell is left unwritten (no zero fill).
    (dict(N=2, H=9, W=12, C=32, K=32, PAD=2), dict(tiles_c=1, tiles_p=2)),
    (
        dict(N=1, H=30, W=40, C=256, K=64),
        dict(tiles_c=4, tiles_p=8, tile_w=32, prefetch=True, sched="iterative-ilp"),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDgradWmmaGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = DirectDgradWmmaSpec(problem=p, **knobs)
        self.assertIsNone(dgrad_wmma_reason(spec, GPU_ARCH))
        art = compile_kernel(build_direct_dgrad_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k).uniform_(-1, 1).to(td)
        w = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(td)
        dX = torch.zeros(p.N, p.H, p.W, p.total_c, dtype=td)
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        bufs = [rt.alloc(t.nbytes) for t in (dY, w, dX)]
        try:
            rt.memcpy_h2d(bufs[0], u8(dY), dY.nbytes)
            rt.memcpy_h2d(bufs[1], u8(w), w.nbytes)
            rt.memset(bufs[2], 0x7F, dX.nbytes)  # poison: every element is written
            vals = ConvArgs.from_problem(p, direction="dgrad").to_launch_values(
                *(int(x) for x in bufs), dY.nbytes, w.nbytes, dX.nbytes
            )
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="dgrad"),
            )(
                vals,
                config=LaunchConfig(
                    grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
                ),
            )
            rt.memcpy_d2h(u8(dX), bufs[2], dX.nbytes)
        finally:
            for x in bufs:
                rt.free(x)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.total_c, p.H, p.W),
            w.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            padding=p.PAD,
            groups=p.groups,
        ).permute(0, 2, 3, 1)
        err = float((dX.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{pkw} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, knobs in _CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)


if __name__ == "__main__":
    unittest.main()
