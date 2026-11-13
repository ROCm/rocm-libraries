# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the tap-outer WMMA direct forward (``conv_direct_fwd_tapouter_wmma``).

CPU: the spec validator, shape independence of the IR, the AOT grid and the
cross-engine agreement of the validator, kernel name, grid and LDS size. GPU
(gfx1250): edges on every side (windows shifted by the top / left padding,
taps skipped wholly), ragged tiles, groups, a channel tail, both dtypes,
several transfers per operand and a 3-deep ring, checked against torch's
conv2d.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_fwd_tapouter_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_fwd_tapouter_wmma import (
    DirectFwdTapOuterSpec,
    build_direct_fwd_tapouter_wmma,
    tapouter_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)
VARIANT = "direct_fwd_tapouter_wmma"


def _problem(N=2, H=29, W=37, C=64, K=96, groups=1, PAD=12, dil=12, dtype="bf16", **kw):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=3,
        KW=3,
        PAD=PAD,
        stride=1,
        dtype=dtype,
        dil_h=kw.pop("dil_h", dil),
        dil_w=dil,
        **kw,
    )


class TestTapOuterSpec(unittest.TestCase):
    def test_wave32_tdm_only(self):
        spec = DirectFwdTapOuterSpec(problem=_problem())
        self.assertIsNone(tapouter_reason(spec, "gfx1250"))
        self.assertIn("wave32", tapouter_reason(spec, "gfx950"))

    def test_rejects(self):
        p = _problem()
        for problem, knobs, needle in (
            (_problem(C=36), {}, "multiples of 8"),
            (p, dict(ck=48), "multiple of 32"),
            (p, dict(waves_k=1, waves_p=1), "two waves"),
            (p, dict(stages=1), "stages"),
            (p, dict(tile_w=12), "must divide"),
            (p, dict(tdm_split=8), "tdm_split"),
            (p, dict(tdm_split=3), "tdm_split"),
            (p, dict(tdm_split=2, stages=3), "2 stages"),
            (p, dict(sched="no-such-strategy"), "no-such-strategy"),
            (p, dict(tiles_k=8, tiles_p=8, waves_k=2, waves_p=4, ck=256), "LDS"),
            (_problem(pad_w=11), {}, "one padding"),
        ):
            with self.subTest(knobs=knobs):
                why = tapouter_reason(DirectFwdTapOuterSpec(problem=problem, **knobs))
                self.assertIsNotNone(why)
                self.assertIn(needle, why)
        strided = DirectConvProblem(
            N=1, H=32, W=32, groups=1, cpg=64, kpg=64, PAD=1, stride=2
        )
        self.assertIn("stride", tapouter_reason(DirectFwdTapOuterSpec(problem=strided)))

    def test_ir_is_shape_independent(self):
        """One binary serves every extent, group count and channel count."""
        from rocke.core.ir_serialize import serialize

        a = build_direct_fwd_tapouter_wmma(DirectFwdTapOuterSpec(problem=_problem()))
        b = build_direct_fwd_tapouter_wmma(
            DirectFwdTapOuterSpec(
                problem=_problem(N=7, H=65, W=9, C=256, K=32, groups=2)
            )
        )
        self.assertEqual(serialize(a), serialize(b))

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _caps_jobs,
            _knob_grid,
        )

        direction, caps_list = DIRECT_CAPABILITIES[VARIANT]
        self.assertEqual(direction, "fwd")
        for caps in caps_list:
            knobs = list(_knob_grid(VARIANT, "gfx1250", "bf16", caps))
            _, jobs = _caps_jobs("gfx1250", 32, VARIANT, "fwd", caps, "bf16")
            self.assertEqual(len(jobs), len(knobs), caps)  # every row builds
            self.assertEqual(list(_knob_grid(VARIANT, "gfx950", "bf16", caps)), [])

    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over the AOT grid and a set of rejected specs."""
        import pytest

        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec
        from kernels.conv.engine_bridge import (
            conv_direct_fwd_tapouter_wmma_spec_to_dict,
        )

        specs = []
        _, caps_list = DIRECT_CAPABILITIES[VARIANT]
        for dtype in ("bf16", "fp16"):
            for caps in caps_list:
                probe = probe_problem(caps, dtype, VARIANT)
                for knobs in _knob_grid(VARIANT, "gfx1250", dtype, caps):
                    specs.append(make_direct_spec(VARIANT, probe, knobs))
        p = _problem()
        for problem, knobs in (
            (_problem(dtype="fp32"), {}),
            (_problem(C=36), {}),
            (_problem(PAD=-1), {}),
            (_problem(H=1, W=1, PAD=0), {}),
            (_problem(pad_w=11), {}),
            (_problem(pad_w=-2), {}),
            (_problem(dil_h=0), {}),
            (p, dict(ck=48)),
            (p, dict(waves_k=0)),
            (p, dict(waves_k=1, waves_p=1)),
            (p, dict(stages=1)),
            (p, dict(tile_w=12)),
            (p, dict(tdm_split=8)),
            (p, dict(tdm_split=3)),
            (p, dict(waves_k=8, waves_p=8)),
            (p, dict(tiles_k=8, tiles_p=8, waves_k=2, waves_p=4, ck=256)),
            (p, dict(tiles_k=16, tiles_p=16, ck=32, stages=2)),
            (_problem(N=4096, H=128, W=128, C=512, K=512), {}),
            (p, dict(prefetch=True, tdm_split=2, stages=4)),
            (p, dict(prefetch=True, tdm_split=2, stages=2)),
            (p, dict(sched="bogus")),
            # 2^31 channels: total_c / total_k must not wrap in C++.
            (_problem(N=1, H=33, W=33, C=1 << 31, K=1 << 31, groups=1 << 25), {}),
        ):
            specs.append(DirectFwdTapOuterSpec(problem=problem, **knobs))
        strided = DirectConvProblem(
            N=1, H=32, W=32, groups=1, cpg=64, kpg=64, PAD=1, stride=2
        )
        specs.append(DirectFwdTapOuterSpec(problem=strided))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, why_cpp, name, grid, lds = (
                    eng.conv_direct_fwd_tapouter_wmma_describe(
                        conv_direct_fwd_tapouter_wmma_spec_to_dict(spec), arch
                    )
                )
                why = tapouter_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_cpp))
                if why is not None:
                    self.assertEqual(why_cpp, why, (arch, spec))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_spec_for_problem_filters_tiles(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = dict(waves_k=2, waves_p=2, tiles_k=4, tiles_p=4, tile_w=16)
        spec, why = direct_spec_for_problem(
            VARIANT, _problem(H=65, W=65), knobs, "gfx1250"
        )
        self.assertIsNotNone(spec, why)
        # A 1x1 output keeps 1 of the 128 pixels of a tile.
        spec, why = direct_spec_for_problem(
            VARIANT, _problem(H=1, W=1, PAD=12), knobs, "gfx1250"
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
    (dict(), dict()),
    (dict(dtype="fp16"), dict(waves_k=2, waves_p=2, tiles_k=4, tiles_p=4, ck=128)),
    # Dilation past the image: most taps of most blocks skipped.
    (dict(N=1, H=65, W=65, C=128, K=256, PAD=36, dil=36), dict(tile_w=8)),
    (dict(N=1, H=20, W=20, C=64, K=64), dict(tiles_p=1, tile_w=32)),
    # A channel tail (96 = 64 + 32) and K short of the block.
    (dict(C=96, K=40), dict(stages=3)),
    (dict(groups=2, C=128, K=64, PAD=2, dil=2), dict(tdm_split=2, stages=2)),
    (
        dict(N=3, H=7, W=9, C=64, K=32, PAD=0, dil=1),
        dict(tdm_split=4, tile_w=8, stages=2),
    ),
    (dict(H=33, W=17, PAD=4, dil=4), dict(waves_k=1, waves_p=4, tiles_k=8, tiles_p=2)),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestTapOuterGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = DirectFwdTapOuterSpec(problem=p, **knobs)
        self.assertIsNone(tapouter_reason(spec, GPU_ARCH))
        kernel = build_direct_fwd_tapouter_wmma(spec, GPU_ARCH)
        art = compile_kernel(kernel, arch=GPU_ARCH)
        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        x = torch.empty(p.N, p.H, p.W, p.total_c).uniform_(-1, 1).to(td)
        w = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(td)
        y = torch.zeros(p.N, p.Ho, p.Wo, p.total_k, dtype=td)
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        bufs = [rt.alloc(t.nbytes) for t in (x, w, y)]
        try:
            rt.memcpy_h2d(bufs[0], u8(x), x.nbytes)
            rt.memcpy_h2d(bufs[1], u8(w), w.nbytes)
            rt.memset(bufs[2], 0x7F, y.nbytes)  # poison: every element is written
            vals = ConvArgs.from_problem(p, direction="fwd").to_launch_values(
                *(int(v) for v in bufs), x.nbytes, w.nbytes, y.nbytes
            )
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="fwd"),
            )(
                vals,
                config=LaunchConfig(
                    grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
                ),
            )
            rt.memcpy_d2h(u8(y), bufs[2], y.nbytes)
        finally:
            for v in bufs:
                rt.free(v)
        ref = torch.nn.functional.conv2d(
            x.float().permute(0, 3, 1, 2),
            w.float().permute(0, 3, 1, 2),
            padding=p.PAD,
            dilation=(p.dil_h, p.dil_w),
            groups=p.groups,
        ).permute(0, 2, 3, 1)
        err = float((y.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{pkw} {knobs}: rel err {err:.3g}")

    def test_cases(self):
        for pkw, knobs in _CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                self._check(pkw, knobs)


if __name__ == "__main__":
    unittest.main()
