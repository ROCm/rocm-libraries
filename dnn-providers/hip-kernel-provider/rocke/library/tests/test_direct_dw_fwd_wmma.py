# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the Toeplitz WMMA depthwise forward (``conv_direct_dw_fwd_wmma``).

CPU: the spec validator, the AOT grid, the cache's problem gate and the
cross-engine agreement of the validator, kernel name, grid and LDS size. GPU
(gfx1250): 3x3 / 5x5 / 7x7 "same" and 7x7 "valid" layers with image edges on
every side, 1 to 4 folded images, 2 to 16 waves, one-row to whole-map tiles
and a partial channel block -- checked numerically against torch's forward
convolution.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_dw_fwd_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_dw_fwd_wmma import (
    DwFwdWmmaSpec,
    build_dw_fwd_wmma,
    dw_fwd_wmma_reason,
    is_valid_dw_fwd_wmma_spec,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=20, W=37, C=64, K=3, PAD=1, dtype="bf16", stride=1, **kw):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=C,
        cpg=1,
        kpg=1,
        KH=K,
        KW=K,
        PAD=PAD,
        stride=stride,
        dtype=dtype,
        **kw,
    )


def _reason(spec, arch):
    ok, why = is_valid_dw_fwd_wmma_spec(spec, arch)
    return None if ok else why


class TestDwFwdWmmaSpec(unittest.TestCase):
    def test_gfx1250_only(self):
        spec = DwFwdWmmaSpec(problem=_problem())
        self.assertEqual(is_valid_dw_fwd_wmma_spec(spec, "gfx1250"), (True, "ok"))
        ok, why = is_valid_dw_fwd_wmma_spec(spec, "gfx950")
        self.assertFalse(ok)
        self.assertIn("gfx1250", why)

    def test_rejects(self):
        for problem, knobs, needle in (
            (_problem(dtype="fp32"), {}, "dtype"),
            (DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=2, kpg=2), {}, "cpg=kpg=1"),
            (_problem(stride=2), {}, "stride=1"),
            (_problem(dil_h=2, PAD=2), {}, "dilation"),
            (_problem(pad_w=0), {}, "one padding"),
            (_problem(PAD=3), {}, "degenerate"),
            (_problem(C=12), {}, "groups % 8"),
            (_problem(C=0), {}, "groups must be >= 1 (got 0)"),
            (_problem(C=-16), {}, "groups must be >= 1 (got -16)"),
            (_problem(N=0), {}, "N must be >= 1 (got 0)"),
            (_problem(K=0, PAD=0), {}, "KH must be >= 1 (got 0)"),
            (_problem(), dict(block_h=0), "block_h"),
            (_problem(), dict(block_waves=17), "block_waves"),
            (_problem(), dict(w_fold=3), "w_fold"),
            (_problem(), dict(prefetch_rows=5), "prefetch_rows"),
            (_problem(K=9, PAD=4), {}, "VGPRs"),
            (_problem(K=5, PAD=2), dict(block_waves=16, prefetch_rows=4), "16 waves"),
            (_problem(K=7, PAD=3), dict(block_h=80), "unrolled"),
            (_problem(N=4096, H=256, W=256), {}, "sentinel"),
        ):
            why = dw_fwd_wmma_reason(DwFwdWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, (problem, knobs))
            self.assertIn(needle, why)

    def test_engines_refuse_non_positive_extents(self):
        """A zero or negative extent is refused before any other rule, with
        the same text in both engines: groups = 0 passed the divisibility and
        tensor-size rules (0 bytes) and gave a zero grid Y."""
        import dataclasses

        import pytest

        eng = pytest.importorskip("rocke_library_engine")
        from kernels.common.conv_direct_grouped import PROBLEM_DIMS
        from kernels.conv.engine_bridge import conv_direct_dw_fwd_wmma_spec_to_dict

        base = DwFwdWmmaSpec(problem=_problem())
        self.assertIsNone(dw_fwd_wmma_reason(base))
        for name in PROBLEM_DIMS:
            for value in (0, -1, -16):
                problem = dataclasses.replace(base.problem, **{name: value})
                spec = dataclasses.replace(base, problem=problem)
                want = f"{name} must be >= 1 (got {value})"
                self.assertEqual(_reason(spec, "gfx1250"), want, spec)
                ok, why_cpp, *_ = eng.conv_direct_dw_fwd_wmma_describe(
                    conv_direct_dw_fwd_wmma_spec_to_dict(spec), "gfx1250"
                )
                self.assertFalse(ok, spec)
                self.assertEqual(why_cpp, want, spec)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _caps_jobs,
            _knob_grid,
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_dw_fwd_wmma"]
        self.assertEqual(direction, "fwd")
        for caps in caps_list:
            self.assertEqual(
                list(_knob_grid("direct_dw_fwd_wmma", "gfx950", "bf16", caps)), []
            )
            knobs = list(_knob_grid("direct_dw_fwd_wmma", "gfx1250", "bf16", caps))
            _, jobs = _caps_jobs(
                "gfx1250", 32, "direct_dw_fwd_wmma", "fwd", caps, "bf16"
            )
            # The validator drops the rows past the fragment / unroll / LDS
            # budgets; every cap keeps some.
            self.assertGreater(len(jobs), 0, caps)
            self.assertLessEqual(len(jobs), len(knobs), caps)

    def test_spec_for_problem(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = dict(block_waves=4, block_h=8, w_fold=1, prefetch_rows=2)
        spec, why = direct_spec_for_problem(
            "direct_dw_fwd_wmma", _problem(N=8, H=56, W=56, C=128), knobs, "gfx1250", 32
        )
        self.assertIsNotNone(spec, why)
        self.assertEqual(spec.launch_geometry()[1], (128, 1, 1))
        spec, why = direct_spec_for_problem(
            "direct_dw_fwd_wmma", _problem(pad_w=0), knobs, "gfx1250", 32
        )
        self.assertIsNone(spec)
        self.assertIn("one padding", why)

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
        from kernels.conv.engine_bridge import conv_direct_dw_fwd_wmma_spec_to_dict

        variant = "direct_dw_fwd_wmma"
        specs = []
        _, caps_list = DIRECT_CAPABILITIES[variant]
        for dtype in ("bf16", "fp16"):
            for caps in caps_list:
                probe = probe_problem(caps, dtype, variant)
                for knobs in _knob_grid(variant, "gfx1250", dtype, caps):
                    specs.append(make_direct_spec(variant, probe, knobs))
        for problem, knobs in (
            (_problem(dtype="fp32"), {}),
            (_problem(stride=2), {}),
            (_problem(dil_w=2), {}),
            (_problem(pad_w=0), {}),
            (_problem(pad_w=-2), {}),
            (_problem(PAD=3), {}),
            (_problem(H=1, PAD=0), {}),
            (_problem(C=12), {}),
            # groups = 0 (and -16) passed the divisibility rule and gave an
            # empty grid; every non-positive extent is refused first.
            (_problem(C=0), {}),
            (_problem(C=-16), {}),
            (_problem(N=0), {}),
            (_problem(N=-1), {}),
            (_problem(), dict(block_h=0)),
            (_problem(), dict(block_waves=17)),
            (_problem(), dict(w_fold=3)),
            (_problem(), dict(prefetch_rows=0)),
            (_problem(), dict(prefetch_rows=5)),
            (_problem(K=9, PAD=4), {}),
            (_problem(K=5, PAD=2), dict(block_waves=16, prefetch_rows=4)),
            (_problem(K=7, PAD=3), dict(block_h=80)),
            (_problem(N=4096, H=256, W=256), {}),
            (_problem(C=24, H=13, W=19), dict(block_waves=8, w_fold=4)),
        ):
            specs.append(DwFwdWmmaSpec(problem=problem, **knobs))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, why_cpp, name, grid, lds = eng.conv_direct_dw_fwd_wmma_describe(
                    conv_direct_dw_fwd_wmma_spec_to_dict(spec), arch
                )
                why = _reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_cpp))
                if why is not None:
                    self.assertEqual(why_cpp, why, (arch, spec))
                self.assertEqual(name, spec.kernel_name())
                if spec.block_waves >= 1 and spec.w_fold in (1, 2, 4):
                    self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_CASES = (
    # (problem kwargs, spec knobs)
    (dict(), dict()),
    (dict(dtype="fp16"), dict(block_waves=2, block_h=16, w_fold=2, prefetch_rows=3)),
    (dict(K=5, PAD=2, N=3, H=17, W=23), dict(block_waves=8, block_h=4, w_fold=4)),
    (dict(K=7, PAD=3, H=29, W=31, dtype="fp16"), dict(block_h=7, prefetch_rows=4)),
    (dict(K=7, PAD=0, H=22, W=40), dict(block_h=8, w_fold=2)),
    (dict(C=128, H=9, W=11), dict(block_waves=16, block_h=1)),
    (dict(N=1, H=56, W=56, dtype="fp16"), dict(block_h=56, w_fold=1)),
    # A partial channel block (24 channels, 64 per block).
    (dict(C=24, H=13, W=19), dict(block_waves=8, block_h=2, w_fold=4, prefetch_rows=3)),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestDwFwdWmmaGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = DwFwdWmmaSpec(problem=p, **knobs)
        self.assertIsNone(_reason(spec, GPU_ARCH))
        art = compile_kernel(build_dw_fwd_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        x = torch.empty(p.N, p.H, p.W, p.groups).uniform_(-1, 1).to(td)
        w = torch.empty(p.groups, p.KH, p.KW, 1).uniform_(-1, 1).to(td)
        y = torch.zeros(p.N, p.Ho, p.Wo, p.groups, dtype=td)
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
            grid, block = spec.launch_geometry()
            KernelLauncher(
                hsaco=art.hsaco,
                kernel_name=art.kernel_name,
                signature=conv_direct_args_signature(p.dtype, direction="fwd"),
            )(vals, config=LaunchConfig(grid=grid, block=block, fence=True))
            rt.memcpy_d2h(u8(y), bufs[2], y.nbytes)
        finally:
            for v in bufs:
                rt.free(v)
        ref = torch.nn.functional.conv2d(
            x.float().permute(0, 3, 1, 2),
            w.float().permute(0, 3, 1, 2),
            padding=p.PAD,
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
