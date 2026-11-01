# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA forward kernels of tiny channel counts
(``conv_fwd_small_wmma``: few input channels / few output channels).

CPU: the spec validators and the AOT grid. GPU (gfx1250): 3x3 stride 1 / 2 and
7x7 stride 2 small-C stems with image edges on every side, several tiles per
block, the TDM-store epilogue and a partial output-channel block; the small-K
kernel with edges, stride 2, several channel chunks and K 4 / 8 -- checked
numerically against torch's forward convolution.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_fwd_small_wmma.py -v
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

from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_fwd_small_wmma import (
    FwdSmallcWmmaSpec,
    FwdSmallkWmmaSpec,
    build_fwd_smallc_wmma,
    build_fwd_smallk_wmma,
    fwd_smallc_reason,
    fwd_smallk_reason,
)

GPU_ARCH = get_device_arch(0)


def _problem(
    N=2, H=32, W=48, C=3, K=64, KH=3, S=1, PAD=1, groups=1, dtype="bf16", KW=0, pad_w=-1
):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KW or KH,
        PAD=PAD,
        pad_w=pad_w,
        stride=S,
        dtype=dtype,
    )


class TestSmallSpecs(unittest.TestCase):
    def test_smallc_rejects(self):
        ok = FwdSmallcWmmaSpec(problem=_problem())
        self.assertIsNone(fwd_smallc_reason(ok))
        self.assertIn("wave32", fwd_smallc_reason(ok, "gfx950"))
        for problem, knobs, needle in (
            (_problem(C=8), {}, "C 8"),
            (_problem(K=60), {}, "multiple of 8"),
            (_problem(C=4, K=32, groups=2), {}, "ungrouped"),
            (_problem(dtype="fp32"), {}, "dtype"),
            (_problem(dtype=None), {}, "dtype"),
            (_problem(S=0), {}, "stride 0 < 1"),
            (_problem(KH=1, KW=7, PAD=0, pad_w=3), {}, "one padding"),
            (_problem(), dict(out_h=4), "GEMM pixels"),
            (_problem(), dict(tiles_k=8, tiles_p=8, out_h=16), "VGPRs"),
        ):
            why = fwd_smallc_reason(FwdSmallcWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_smallk_rejects(self):
        p = _problem(C=64, K=4)
        ok = FwdSmallkWmmaSpec(problem=p)
        self.assertIsNone(fwd_smallk_reason(ok))
        self.assertIn("wave32", fwd_smallk_reason(ok, "gfx950"))
        # Any K up to 8, rectangular filters with their own column padding.
        k3 = FwdSmallkWmmaSpec(
            problem=_problem(C=64, K=3), tiles_c=2, tiles_p=3, out_h=6, out_w=16
        )
        self.assertIsNone(fwd_smallk_reason(k3))
        row = _problem(H=1, W=300, C=32, K=1, KH=1, KW=7, PAD=0, pad_w=3)
        self.assertIsNone(
            fwd_smallk_reason(
                FwdSmallkWmmaSpec(problem=row, tiles_c=1, tiles_p=4, out_h=1, out_w=58)
            )
        )
        for problem, knobs, needle in (
            (_problem(C=64, K=16), {}, "K 16"),
            (_problem(C=64, K=3), {}, "block_c 48"),
            (_problem(KH=1, KW=7, PAD=0, pad_w=7, C=32, K=1), {}, "column PAD 7"),
            (_problem(C=60, K=4), {}, "multiple of 8"),
            (_problem(C=64, K=4, S=0), {}, "stride 0 < 1"),
            (p, dict(tiles_c=4), "block_c"),
            (p, dict(out_h=32, out_w=32), "input window"),
            (p, dict(ck=48), "ck 48"),
        ):
            why = fwd_smallk_reason(FwdSmallkWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

        for variant in ("direct_fwd_smallc_wmma", "direct_fwd_smallk_wmma"):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            self.assertEqual(
                list(_knob_grid(variant, "gfx950", "bf16", caps_list[0])), []
            )
            for caps in caps_list:
                probe = probe_problem(caps, "bf16", variant)
                n_ok = 0
                for knobs in _knob_grid(variant, "gfx1250", "bf16", caps):
                    spec = make_direct_spec(variant, probe, knobs)
                    n_ok += validate_direct_spec(variant, spec, "gfx1250")[0]
                self.assertGreater(n_ok, 0, (variant, caps))

    def test_engines_agree_on_specs(self):
        """The C++ engine's validator, kernel name, grid and LDS size match
        Python's over the AOT grid and a few rejected specs (the byte-identity
        gate only covers the IR of specs both engines build)."""
        import pytest

        eng = pytest.importorskip("rocke_engine")
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec
        from rocke.core.backend import (
            conv_direct_fwd_smallc_wmma_spec_to_dict,
            conv_direct_fwd_smallk_wmma_spec_to_dict,
        )

        sides = {
            "direct_fwd_smallc_wmma": (
                eng.conv_direct_fwd_smallc_wmma_describe,
                conv_direct_fwd_smallc_wmma_spec_to_dict,
                fwd_smallc_reason,
            ),
            "direct_fwd_smallk_wmma": (
                eng.conv_direct_fwd_smallk_wmma_describe,
                conv_direct_fwd_smallk_wmma_spec_to_dict,
                fwd_smallk_reason,
            ),
        }
        specs = []
        for variant in sides:
            _, caps_list = DIRECT_CAPABILITIES[variant]
            for dtype in ("bf16", "fp16"):
                for caps in caps_list:
                    probe = probe_problem(caps, dtype, variant)
                    for knobs in _knob_grid(variant, "gfx1250", dtype, caps):
                        specs.append((variant, make_direct_spec(variant, probe, knobs)))
        pk = _problem(C=64, K=4)
        for problem, knobs in (
            (_problem(C=8), {}),
            (_problem(K=60), {}),
            (_problem(C=4, K=32, groups=2), {}),
            (_problem(dtype="fp32"), {}),
            (_problem(KH=3, PAD=3), {}),
            (_problem(S=0), {}),
            (_problem(S=-1), {}),
            (_problem(N=64, H=1024, W=1024), {}),
            (_problem(), dict(out_h=4)),
            (_problem(), dict(tiles_k=8, tiles_p=8, out_h=16)),
            (_problem(), dict(tiles_per_block=0)),
            (_problem(), dict(waves_k=0)),
            (_problem(), dict(sched="bogus")),
            (_problem(), dict(tiles_per_block=3, tdm_store=True, sched="max-ilp")),
        ):
            specs.append(
                (
                    "direct_fwd_smallc_wmma",
                    FwdSmallcWmmaSpec(problem=problem, **knobs),
                )
            )
        for problem, knobs in (
            (_problem(C=64, K=16), {}),
            (_problem(C=64, K=3), {}),
            (_problem(C=60, K=4), {}),
            (_problem(C=64, K=4, S=0), {}),
            (pk, dict(tiles_c=4)),
            (pk, dict(out_h=32, out_w=32)),
            (pk, dict(ck=48)),
            (pk, dict(stages=1)),
            (pk, dict(waves_p=1)),
            (pk, dict(tiles_p=12, prefetch=True)),
            (pk, dict(stages=4, prefetch=True, sched="iterative-ilp")),
        ):
            specs.append(
                (
                    "direct_fwd_smallk_wmma",
                    FwdSmallkWmmaSpec(problem=problem, **knobs),
                )
            )
        for arch in ("gfx1250", "gfx950"):
            for variant, spec in specs:
                describe, to_dict, reason = sides[variant]
                ok, why_cpp, name, grid, lds = describe(to_dict(spec), arch)
                why = reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_cpp))
                if why is not None and "scheduler" not in why:
                    self.assertEqual(why_cpp, why, (arch, spec))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed); a
        missing, mistyped or partial problem raises."""
        import pytest

        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import (
            conv_direct_fwd_smallc_wmma_spec_to_dict,
            conv_direct_fwd_smallk_wmma_spec_to_dict,
        )

        for prefix, spec, to_dict in (
            (
                "conv_direct_fwd_smallc_wmma",
                FwdSmallcWmmaSpec(problem=_problem(), waves_k=0),
                conv_direct_fwd_smallc_wmma_spec_to_dict,
            ),
            (
                "conv_direct_fwd_smallk_wmma",
                FwdSmallkWmmaSpec(problem=_problem(C=64, K=4), waves_c=0),
                conv_direct_fwd_smallk_wmma_spec_to_dict,
            ),
        ):
            describe = getattr(eng, f"{prefix}_describe")
            lower = getattr(eng, f"{prefix}_lower_llvm")
            bad = to_dict(spec)
            ok, why, _, grid, _ = describe(bad, "gfx1250")
            self.assertFalse(ok)
            self.assertIn("waves and tiles", why)
            self.assertEqual(tuple(grid), (0, 0, 0))
            with self.assertRaises(Exception):
                lower(bad, arch="gfx1250")
            good = dict(bad)
            good.pop("waves_k", None)
            good.pop("waves_c", None)
            # A zero stride is rejected before the output extents divide by it.
            sd = dict(good, problem=dict(good["problem"], stride=0))
            ok, why, _, grid, _ = describe(sd, "gfx1250")
            self.assertFalse(ok)
            self.assertEqual(why, "stride 0 < 1")
            self.assertEqual(tuple(grid), (0, 0, 0))
            with self.assertRaises(Exception):
                lower(sd, arch="gfx1250")
            # An explicit dtype=None is rejected (Python rejects the spec
            # too), not read as the fp16 default.
            sd = dict(good, problem=dict(good["problem"], dtype=None))
            with self.subTest(prefix=prefix, dtype=None):
                with self.assertRaisesRegex(ValueError, "dtype"):
                    describe(sd, "gfx1250")
                with self.assertRaisesRegex(ValueError, "dtype"):
                    lower(sd, arch="gfx1250")
            partial = dict(good["problem"])
            del partial["kpg"]
            for problem in (None, "N2H13", partial):
                sd = dict(good)
                if problem is None:
                    del sd["problem"]
                else:
                    sd["problem"] = problem
                with self.subTest(prefix=prefix, problem=problem):
                    with self.assertRaises(ValueError):
                        describe(sd, "gfx1250")
                    with self.assertRaises(ValueError):
                        lower(sd, arch="gfx1250")


# ---------------------------------------------------------------------------
# The shipped path: a gfx1250 forward cache (placeholder binaries) and
# direct_plans, which matches capabilities, rebuilds each entry's spec from
# its identity and re-validates it on the request (tile efficiency included).
# ---------------------------------------------------------------------------
_CACHE_ARCH = "gfx1250"


def _wmma_fwd_cache(tmp_path):
    """The wave32-only WMMA forward entries of the gfx1250 cache, each with a
    placeholder binary (nothing is compiled)."""
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelCache
    from kernels.conv.direct_variants import (
        DIRECT_FWD_SMALLC_WMMA,
        DIRECT_FWD_SMALLK_WMMA,
        DIRECT_FWD_WMMA,
    )
    from rocke.core.arch import ArchTarget

    families = {DIRECT_FWD_WMMA, DIRECT_FWD_SMALLC_WMMA, DIRECT_FWD_SMALLK_WMMA}
    cache = KernelCache(tmp_path, _CACHE_ARCH)
    for job in dks.enumerate_direct_jobs(
        arch=_CACHE_ARCH,
        target=ArchTarget.from_gfx(_CACHE_ARCH),
        directions=("fwd",),
    ):
        if job.identity.algorithm in families:
            # One placeholder per entry, so each keeps its own symbol.
            label = job.identity.label()
            meta = {"kernel_name": label, "spec_wave_size": 32}
            cache.put(job.identity, label.encode(), meta)
    return cache


def _cpu_problem(N, H, W, C, K, KH, S, PAD):
    return DirectConvProblem(
        N=N, H=H, W=W, groups=1, cpg=C, kpg=K, KH=KH, KW=KH, PAD=PAD, stride=S
    )


def _check_small_plans(problem, variant, plans, rejected):
    """Every ``variant`` plan launches the spec its knobs make on ``problem``,
    with the forward operands; every rejection of it is a shape rule."""
    import json

    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.conv.direct_variants import make_direct_spec
    from rocke.runtime.packing import compile_packer

    mine = [pl for pl in plans if pl.identity.algorithm == variant]
    assert mine, [why for i, why in rejected if i.algorithm == variant]
    ptrs = {"x": 0x1000, "w": 0x2000, "y": 0x3000}
    sizes = {
        "x": 2 * problem.N * problem.H * problem.W * problem.total_c,
        "w": 2 * problem.total_k * problem.KH * problem.KW * problem.cpg,
        "y": 2 * problem.N * problem.Ho * problem.Wo * problem.total_k,
    }
    expect = ConvArgs.from_problem(problem, direction="fwd").to_launch_values(
        ptrs["x"], ptrs["w"], ptrs["y"], sizes["x"], sizes["w"], sizes["y"]
    )
    for plan in mine:
        ident = plan.identity
        assert (ident.filter_h, ident.pad_h, ident.stride_h, ident.wave_size) == (
            problem.KH,
            problem.PAD,
            problem.stride,
            32,
        )
        knobs = json.loads(ident.knobs)
        spec = make_direct_spec(variant, problem, knobs)
        (step,) = plan.steps
        assert step.kernel_name == ident.label()
        assert (step.grid, step.block) == spec.launch_geometry(), ident.label()
        assert step.buffers == ("x", "w", "y")
        assert step.signature == conv_direct_args_signature(
            problem.dtype, direction="fwd"
        )
        assert plan.workspaces == {}
        values = dks.launch_values(step, ptrs, sizes)
        assert values == expect
        compile_packer(step.signature)(values)
    for ident, why in rejected:
        if ident.algorithm == variant:
            assert "tile efficiency" in why or "block_k" in why, why
    return mine


def test_smallc_is_served_from_the_cache(tmp_path):
    """A 3-channel stem (and a 4-channel one) gets the small-C kernel from the
    cache, with the launch its knobs make on the request; a filter no binary
    bakes, or C past the small-C limit, gets none."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.conv.direct_variants import (
        DIRECT_FWD_SMALLC_WMMA,
        DIRECT_FWD_SMALLK_WMMA,
    )

    cache = _wmma_fwd_cache(tmp_path)
    for problem in (
        _cpu_problem(N=2, H=224, W=224, C=3, K=64, KH=7, S=2, PAD=3),
        _cpu_problem(N=3, H=64, W=96, C=4, K=32, KH=3, S=1, PAD=1),
        _cpu_problem(N=1, H=224, W=224, C=3, K=96, KH=4, S=4, PAD=0),
    ):
        plans, rejected = dks.direct_plans(cache, problem, "fwd", _CACHE_ARCH)
        mine = _check_small_plans(problem, DIRECT_FWD_SMALLC_WMMA, plans, rejected)
        assert {pl.identity.cpg for pl in mine} == {problem.cpg}
        assert DIRECT_FWD_SMALLK_WMMA not in {pl.identity.algorithm for pl in plans}
    # A tiny output: the filter rejects the knobs whose tiles are mostly
    # padding, and keeps the rest.
    tiny = _cpu_problem(N=1, H=12, W=12, C=3, K=64, KH=3, S=1, PAD=1)
    plans, rejected = dks.direct_plans(cache, tiny, "fwd", _CACHE_ARCH)
    _check_small_plans(tiny, DIRECT_FWD_SMALLC_WMMA, plans, rejected)
    assert any("tile efficiency" in why for _, why in rejected)
    for problem in (
        _cpu_problem(N=2, H=56, W=56, C=3, K=64, KH=5, S=1, PAD=2),
        _cpu_problem(N=2, H=56, W=56, C=8, K=64, KH=3, S=1, PAD=1),
    ):
        plans, _ = dks.direct_plans(cache, problem, "fwd", _CACHE_ARCH)
        assert DIRECT_FWD_SMALLC_WMMA not in {pl.identity.algorithm for pl in plans}


def test_smallk_is_served_from_the_cache(tmp_path):
    """A 4-output-channel layer gets the small-K kernel from the cache at any
    multiple-of-8 C, with the launch its knobs make on the request; K or a
    filter no binary bakes gets none."""
    from benchmarks.common import direct_kernel_sweep as dks
    from kernels.conv.direct_variants import (
        DIRECT_FWD_SMALLC_WMMA,
        DIRECT_FWD_SMALLK_WMMA,
    )

    cache = _wmma_fwd_cache(tmp_path)
    for problem in (
        _cpu_problem(N=2, H=56, W=56, C=64, K=4, KH=3, S=1, PAD=1),
        _cpu_problem(N=1, H=100, W=60, C=192, K=4, KH=3, S=1, PAD=1),
    ):
        plans, rejected = dks.direct_plans(cache, problem, "fwd", _CACHE_ARCH)
        mine = _check_small_plans(problem, DIRECT_FWD_SMALLK_WMMA, plans, rejected)
        assert {pl.identity.kpg for pl in mine} == {problem.kpg}
        assert DIRECT_FWD_SMALLC_WMMA not in {pl.identity.algorithm for pl in plans}
    for problem in (
        _cpu_problem(N=2, H=56, W=56, C=64, K=8, KH=3, S=1, PAD=1),
        _cpu_problem(N=2, H=56, W=56, C=64, K=4, KH=3, S=2, PAD=1),
    ):
        plans, _ = dks.direct_plans(cache, problem, "fwd", _CACHE_ARCH)
        assert DIRECT_FWD_SMALLK_WMMA not in {pl.identity.algorithm for pl in plans}


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_SMALLC_CASES = (
    (dict(H=37, W=29), dict()),
    (dict(H=37, W=29, K=24), dict(tdm_store=True)),
    (
        dict(N=2, H=50, W=70, K=24, S=2),
        dict(out_h=4, out_w=16, tiles_p=1, tdm_store=True),
    ),
    (
        dict(H=40, W=36, KH=7, S=2, PAD=3, dtype="fp16"),
        dict(tiles_k=2, out_h=16, out_w=16, waves_p=2, tiles_p=8),
    ),
    (
        dict(N=2, H=33, W=45, C=4, K=40),
        dict(waves_k=2, tiles_k=1, tiles_per_block=3, tdm_store=True),
    ),
    (
        dict(N=2, H=64, W=64, K=128),
        dict(waves_k=2, tiles_k=4, tiles_per_block=4),
    ),
)
_SMALLK_CASES = (
    (dict(C=64, K=4, H=33, W=27), dict()),
    (dict(N=2, C=320, K=4, H=40, W=44), dict(ck=64)),
    (
        dict(C=64, K=8, H=20, W=20, dtype="fp16"),
        dict(tiles_c=5, prefetch=True, sched="iterative-ilp"),
    ),
    (
        dict(C=64, K=4, H=30, W=30, S=2),
        dict(out_h=8, out_w=8, tiles_c=3, tiles_p=5),
    ),
    (
        dict(C=64, K=3, H=29, W=35, dtype="fp16"),
        dict(tiles_c=2, tiles_p=3, out_h=6, out_w=16),
    ),
    (
        dict(N=3, C=128, K=3, H=40, W=18),
        dict(waves_c=2, tiles_c=1, tiles_p=4, out_h=14, out_w=14, ck=32, stages=3),
    ),
    (
        dict(N=2, C=32, K=1, H=1, W=301, KH=1, KW=7, PAD=0, pad_w=3, dtype="fp16"),
        dict(waves_p=2, tiles_c=1, tiles_p=4, out_h=1, out_w=122, ck=32),
    ),  # K = 6: channel-by-channel Z reads and stores.
    (
        dict(C=64, K=6, H=20, W=23),
        dict(waves_p=3, tiles_c=4, tiles_p=3, out_h=6, out_w=16, ck=32),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestSmallGpu(unittest.TestCase):
    def _check(self, spec, build):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = spec.problem
        art = compile_kernel(build(spec, GPU_ARCH), arch=GPU_ARCH)
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
            padding=(p.PAD, p.PAD_W),
        ).permute(0, 2, 3, 1)
        err = float((y.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{spec}: rel err {err:.3g}")

    def test_smallc(self):
        for pkw, knobs in _SMALLC_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = FwdSmallcWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(fwd_smallc_reason(spec, GPU_ARCH))
                self._check(spec, build_fwd_smallc_wmma)

    def test_smallk(self):
        for pkw, knobs in _SMALLK_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = FwdSmallkWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(fwd_smallk_reason(spec, GPU_ARCH))
                self._check(spec, build_fwd_smallk_wmma)
