# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the small-group forward walking row bands
(``conv_direct_fwd_gstream_wmma``).

CPU: the spec validator, the AOT grid, the cache's problem gate and the
cross-engine agreement of the validator, kernel name, grid and LDS size. GPU
(gfx1250): 3x3 layers with 8 / 16 / 32 / 56 / 64 channels per group at
stride 1 and 2 (with and without the phase split), image edges on every side,
walks that run past the last band, 2 and 3 halo slots, prefetch depth 1 / 2
and none -- checked numerically against torch's grouped convolution.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_direct_fwd_gstream_wmma.py -v
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

from kernels.common.conv_direct_fwd_gstream_wmma import (
    FwdGstreamWmmaSpec,
    build_fwd_gstream_wmma,
    fwd_gstream_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(c=16, g=8, S=1, N=2, H=28, W=28, dtype="bf16", k=None, **kw):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=g,
        cpg=c,
        kpg=c if k is None else k,
        KH=kw.pop("KH", 3),
        KW=kw.pop("KW", 3),
        PAD=kw.pop("PAD", 1),
        stride=S,
        dtype=dtype,
        **kw,
    )


def _k(wk, wp, tk, tp, tw, walk, st=2, **kw):
    return dict(
        waves_k=wk,
        waves_p=wp,
        tiles_k=tk,
        tiles_p=tp,
        tile_w=tw,
        walk=walk,
        stages=st,
        **kw,
    )


class TestFwdGstreamSpec(unittest.TestCase):
    def test_wave32_tdm_only(self):
        spec = FwdGstreamWmmaSpec(problem=_problem())
        self.assertIsNone(fwd_gstream_reason(spec, "gfx1250"))
        self.assertIn("wave32", fwd_gstream_reason(spec, "gfx950"))

    def test_rejects(self):
        for problem, knobs, needle in (
            (_problem(S=3), {}, "stride 3"),
            (_problem(dil_h=2), {}, "dilation"),
            (_problem(KH=3, KW=1), {}, "not square"),
            (_problem(pad_w=0), {}, "one padding"),
            (_problem(PAD=3), {}, "PAD 3"),
            (_problem(k=32), {}, "!= kpg"),
            (_problem(c=24), {}, "cpg 24"),
            (_problem(c=48, g=4), {}, "LDS row skew"),
            (_problem(), dict(waves_k=0), "waves and tiles"),
            (_problem(c=32, g=8), _k(1, 2, 3, 2, 16, 4), "neither whole groups"),
            (_problem(c=16, g=3), _k(2, 2, 2, 2, 16, 4), "multiple of block_k"),
            (_problem(), dict(pf_depth=2, prefetch=False), "pf_depth"),
            (_problem(), dict(stages=1), "stages"),
            (_problem(), dict(tile_w=12), "tile_w"),
            (_problem(), dict(phase_split=True), "stride 2 only"),
            (_problem(), _k(4, 16, 1, 1, 16, 4), "threads per block"),
            (_problem(c=64, g=4), _k(1, 4, 4, 4, 16, 4), "VGPRs"),
            (_problem(), dict(sched="bogus"), "scheduler"),
            (_problem(N=4096, H=256, W=256), {}, "i32 byte offsets"),
        ):
            why = fwd_gstream_reason(FwdGstreamWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, (problem, knobs))
            self.assertIn(needle, why)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _caps_jobs,
            _knob_grid,
        )

        direction, caps_list = DIRECT_CAPABILITIES["direct_fwd_gstream_wmma"]
        self.assertEqual(direction, "fwd")
        for caps in caps_list:
            self.assertEqual(
                list(_knob_grid("direct_fwd_gstream_wmma", "gfx950", "bf16", caps)), []
            )
            knobs = list(_knob_grid("direct_fwd_gstream_wmma", "gfx1250", "bf16", caps))
            # phase_split rows only at stride 2.
            if caps.stride == 1:
                self.assertFalse(any(k.get("phase_split") for k in knobs))
            _, jobs = _caps_jobs(
                "gfx1250", 32, "direct_fwd_gstream_wmma", "fwd", caps, "bf16"
            )
            self.assertGreater(len(jobs), 0, caps)
            self.assertLessEqual(len(jobs), len(knobs), caps)

    def test_spec_for_problem(self):
        from kernels.conv.direct_variants import direct_spec_for_problem

        knobs = _k(2, 2, 2, 2, 16, 4)
        spec, why = direct_spec_for_problem(
            "direct_fwd_gstream_wmma",
            _problem(c=16, g=32, N=8, H=56, W=56),
            knobs,
            "gfx1250",
            32,
        )
        self.assertIsNotNone(spec, why)
        spec, why = direct_spec_for_problem(
            "direct_fwd_gstream_wmma",
            _problem(c=16, g=32, H=7, W=7),
            knobs,
            "gfx1250",
            32,
        )
        self.assertIsNone(spec)
        self.assertIn("tile efficiency", why)

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
        from kernels.conv.engine_bridge import conv_direct_fwd_gstream_wmma_spec_to_dict

        variant = "direct_fwd_gstream_wmma"
        specs = []
        _, caps_list = DIRECT_CAPABILITIES[variant]
        for dtype in ("bf16", "fp16"):
            for caps in caps_list:
                probe = probe_problem(caps, dtype, variant)
                for knobs in _knob_grid(variant, "gfx1250", dtype, caps):
                    specs.append(make_direct_spec(variant, probe, knobs))
        for problem, knobs in (
            (_problem(dtype="fp32"), {}),
            (_problem(S=3), {}),
            (_problem(dil_w=2), {}),
            (_problem(KH=3, KW=1), {}),
            (_problem(pad_w=0), {}),
            (_problem(PAD=3), {}),
            (_problem(H=1, PAD=0), {}),
            (_problem(k=32), {}),
            (_problem(c=24), {}),
            (_problem(c=48, g=4), {}),
            (_problem(c=32, g=8), _k(1, 2, 3, 2, 16, 4)),
            (_problem(c=16, g=3), _k(2, 2, 2, 2, 16, 4)),
            (_problem(), dict(pf_depth=2, prefetch=False)),
            (_problem(), dict(stages=1)),
            (_problem(), dict(walk=0)),
            (_problem(), dict(tile_w=12)),
            (_problem(), dict(phase_split=True)),
            (_problem(S=2), _k(1, 1, 1, 1, 16, 4, phase_split=True)),
            (_problem(), _k(4, 16, 1, 1, 16, 4)),
            (_problem(c=64, g=4), _k(1, 4, 4, 4, 16, 4)),
            (_problem(N=4096, H=256, W=256), {}),
            (_problem(c=56, g=4, S=2), _k(8, 1, 2, 4, 32, 4, phase_split=True)),
            (_problem(c=8, g=8, S=2), _k(2, 2, 2, 2, 16, 4, st=3, pf_depth=2)),
            (_problem(), dict(sched="bogus")),
            # 2^31 channels: total_c / total_k must not wrap in C++.
            (_problem(g=1 << 27, N=1, H=8, W=8), {}),
        ):
            specs.append(FwdGstreamWmmaSpec(problem=problem, **knobs))
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, why_cpp, name, grid, lds = (
                    eng.conv_direct_fwd_gstream_wmma_describe(
                        conv_direct_fwd_gstream_wmma_spec_to_dict(spec), arch
                    )
                )
                why = fwd_gstream_reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_cpp))
                if why is not None:
                    self.assertEqual(why_cpp, why, (arch, spec))
                self.assertEqual(name, spec.kernel_name())
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
    (dict(c=8, g=8), _k(1, 2, 2, 2, 8, 2)),
    (dict(c=8, g=16, S=2, H=29, W=31, dtype="fp16"), _k(2, 2, 2, 2, 16, 4)),
    (dict(c=16, g=8, N=3, H=17, W=23), _k(2, 2, 2, 2, 16, 4, st=3, pf_depth=2)),
    (dict(c=16, g=8, S=2, dtype="fp16"), _k(2, 2, 2, 2, 16, 4, phase_split=True)),
    (dict(c=32, g=8, H=19, W=21), _k(2, 2, 4, 1, 16, 12)),
    (dict(c=32, g=8), _k(4, 2, 2, 2, 16, 12, prefetch=False)),
    (
        dict(c=64, g=4, S=2, H=30, W=30),
        _k(2, 2, 2, 1, 16, 8, phase_split=True, pf_depth=2),
    ),
    (dict(c=56, g=4, H=14, W=14), _k(4, 1, 2, 4, 16, 4)),
    (
        dict(c=56, g=4, S=2, H=28, W=28, dtype="fp16"),
        _k(8, 1, 2, 4, 32, 4, phase_split=True),
    ),
    (dict(c=56, g=8, N=1, H=7, W=9), _k(8, 1, 2, 4, 16, 4)),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestFwdGstreamGpu(unittest.TestCase):
    def _check(self, pkw, knobs):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = _problem(**pkw)
        spec = FwdGstreamWmmaSpec(problem=p, **knobs)
        self.assertIsNone(fwd_gstream_reason(spec, GPU_ARCH))
        art = compile_kernel(build_fwd_gstream_wmma(spec, GPU_ARCH), arch=GPU_ARCH)
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
            stride=p.stride,
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
