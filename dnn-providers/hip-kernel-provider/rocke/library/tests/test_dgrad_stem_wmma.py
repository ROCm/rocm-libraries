# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the wave32 WMMA stem dgrad kernels (``conv_dgrad_patch_wmma``,
``conv_dgrad_smallc_wmma``).

CPU: the spec validators and the AOT grid. GPU (gfx1250): patch sizes with and
without an uncovered border, padded GEMM columns, several patch-row blocks;
3x3 stride 1 / 2 and 7x7 stride 2 small-C stems with image edges -- checked
numerically against torch's dgrad.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_dgrad_stem_wmma.py -v
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib.util
import unittest

import pytest
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()

from kernels.common.conv_dgrad_patch_wmma import (
    DgradPatchWmmaSpec,
    build_dgrad_patch_wmma,
    dgrad_patch_reason,
)
from kernels.common.conv_dgrad_smallc_wmma import (
    DgradSmallcWmmaSpec,
    build_dgrad_smallc_wmma,
    dgrad_smallc_reason,
)
from kernels.common.conv_direct_grouped import DirectConvProblem

GPU_ARCH = get_device_arch(0)


def _problem(N=2, H=32, W=48, C=3, K=64, KH=4, S=4, PAD=0, groups=1, dtype="bf16"):
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
        stride=S,
        dtype=dtype,
    )


class TestStemSpecs(unittest.TestCase):
    def test_patch_rejects(self):
        ok = DgradPatchWmmaSpec(
            problem=_problem(), waves_c=3, waves_p=1, tiles_c=1, r_blk=4
        )
        self.assertIsNone(dgrad_patch_reason(ok))
        self.assertIn("wave32", dgrad_patch_reason(ok, "gfx950"))
        for problem, knobs, needle in (
            (_problem(PAD=1), {}, "patch embedding"),
            (_problem(S=2), {}, "patch embedding"),
            (_problem(K=60), dict(waves_c=3, tiles_c=1), "multiple of 8"),
            (_problem(), dict(waves_c=3, tiles_c=1, r_blk=3), "divide P"),
            (_problem(), dict(waves_c=4, tiles_c=1, r_blk=4), "block_c"),
            (_problem(), dict(waves_c=3, tiles_c=1, r_blk=4, ck=48), "ck 48"),
        ):
            why = dgrad_patch_reason(DgradPatchWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_smallc_rejects(self):
        p = _problem(KH=3, S=1, PAD=1)
        ok = DgradSmallcWmmaSpec(
            problem=p, waves_c=1, waves_p=4, tiles_c=2, tiles_p=6, out_h=16, out_w=16
        )
        self.assertIsNone(dgrad_smallc_reason(ok))
        for problem, knobs, needle in (
            (_problem(C=8, KH=3, S=1, PAD=1), {}, "C 8"),
            (_problem(), {}, "patch embedding"),
            (_problem(KH=3, S=2, PAD=1), dict(out_h=15), "multiples of S"),
            (p, dict(tiles_c=3), "block_c"),
            (p, dict(out_h=32, out_w=32), "dY window"),
            (_problem(KH=3, S=0, PAD=1), {}, "stride 0 must be >= 1"),
            (_problem(KH=3, S=-1, PAD=1), {}, "stride -1 must be >= 1"),
        ):
            why = dgrad_smallc_reason(DgradSmallcWmmaSpec(problem=problem, **knobs))
            self.assertIsNotNone(why, knobs)
            self.assertIn(needle, why)

    def test_aot_grid(self):
        from kernels.conv.direct_aot_jobs import (
            DIRECT_CAPABILITIES,
            _knob_grid,
            probe_problem,
        )
        from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

        for variant in ("direct_dgrad_patch_wmma", "direct_dgrad_smallc_wmma"):
            _, caps_list = DIRECT_CAPABILITIES[variant]
            self.assertEqual(
                list(_knob_grid(variant, "gfx950", "bf16", caps_list[0])), []
            )
            n_ok = 0
            for caps in caps_list:
                probe = probe_problem(caps, "bf16", variant)
                for knobs in _knob_grid(variant, "gfx1250", "bf16", caps):
                    spec = make_direct_spec(variant, probe, knobs)
                    n_ok += validate_direct_spec(variant, spec, "gfx1250")[0]
            self.assertGreater(n_ok, 0, variant)


def _aot_specs(variant: str) -> list:
    """Every spec of ``variant``'s AOT grid (all capabilities, both dtypes),
    valid or not."""
    from kernels.conv.direct_aot_jobs import (
        DIRECT_CAPABILITIES,
        _knob_grid,
        probe_problem,
    )
    from kernels.conv.direct_variants import make_direct_spec

    specs = []
    for dtype in ("bf16", "fp16"):
        for caps in DIRECT_CAPABILITIES[variant][1]:
            probe = probe_problem(caps, dtype, variant)
            for knobs in _knob_grid(variant, "gfx1250", dtype, caps):
                specs.append(make_direct_spec(variant, probe, knobs))
    return specs


# Rejected specs the engine-agreement tests add to the AOT grid.
_PATCH_REJECTS = (
    (dict(PAD=1), {}),
    (dict(S=2), {}),
    (dict(K=60), dict(waves_c=3, tiles_c=1)),
    (dict(), dict(waves_c=3, tiles_c=1, r_blk=3)),
    (dict(), dict(waves_c=4, tiles_c=1, r_blk=4)),
    (dict(), dict(waves_c=3, tiles_c=1, r_blk=4, ck=48)),
    (dict(), dict(waves_c=3, tiles_c=1, r_blk=4, tile_w=0)),
    (dict(), dict(waves_c=3, tiles_c=1, r_blk=4, sched="bogus")),
    (dict(dtype="fp32"), dict(waves_c=3, tiles_c=1, r_blk=4)),
    (dict(N=512, H=1024, W=1024), dict(waves_c=3, tiles_c=1, r_blk=4)),
    (dict(KH=16, S=16), dict(waves_c=8, waves_p=8, tiles_c=6, tiles_p=8, r_blk=16)),
    (dict(S=0), dict(waves_c=3, tiles_c=1, r_blk=4)),
)
_SMALLC_REJECTS = (
    (dict(C=8, KH=3, S=1, PAD=1), {}),
    (dict(), {}),
    (dict(KH=3, S=2, PAD=1), dict(out_h=15)),
    (dict(KH=3, S=1, PAD=1), dict(tiles_c=3)),
    (dict(KH=3, S=1, PAD=1), dict(out_h=32, out_w=32)),
    (dict(KH=3, S=1, PAD=3), {}),
    (dict(KH=3, S=1, PAD=1, groups=3, C=9, K=72), {}),
    (dict(KH=3, S=1, PAD=1), dict(sched="")),
    (dict(KH=3, S=1, PAD=1), dict(waves_c=0)),
    (dict(KH=7, S=2, PAD=3), dict(tiles_c=5, waves_c=2, ck=256, stages=4)),
    # A non-positive stride is refused before any stride division (the C++
    # validator used to divide by it).
    (dict(KH=3, S=0, PAD=1), {}),
    (dict(KH=3, S=-1, PAD=1), {}),
)


class TestStemEngines(unittest.TestCase):
    """The C++ engine's validator, reason, kernel name, grid and LDS size
    match Python's over the whole AOT grid and a set of rejected specs (the
    byte-identity gate only covers the IR of specs both engines build)."""

    def _agree(self, describe, to_dict, reason, specs):
        for arch in ("gfx1250", "gfx950"):
            for spec in specs:
                ok, why_c, name, grid, lds = describe(to_dict(spec), arch)
                why = reason(spec, arch)
                self.assertEqual(ok, why is None, (arch, spec, why, why_c))
                if why is not None:
                    # The C++ text of an unknown scheduler strategy is the
                    # first clause of Python's; every other reason is equal.
                    self.assertTrue(why.startswith(why_c), (why, why_c))
                self.assertEqual(name, spec.kernel_name())
                self.assertEqual(lds, spec.lds_bytes())
                if why is None:
                    self.assertEqual(tuple(grid), spec.grid())

    def test_patch_engines_agree_on_specs(self):
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import conv_dgrad_patch_wmma_spec_to_dict

        specs = _aot_specs("direct_dgrad_patch_wmma")
        self.assertTrue(any(dgrad_patch_reason(s) is None for s in specs))
        base = dict(waves_c=3, waves_p=1, tiles_c=1, r_blk=4)
        for pkw, knobs in _PATCH_REJECTS:
            spec = DgradPatchWmmaSpec(problem=_problem(**pkw), **{**base, **knobs})
            self.assertIsNotNone(dgrad_patch_reason(spec), (pkw, knobs))
            specs.append(spec)
        self._agree(
            eng.conv_dgrad_patch_wmma_describe,
            conv_dgrad_patch_wmma_spec_to_dict,
            dgrad_patch_reason,
            specs,
        )

    def test_smallc_engines_agree_on_specs(self):
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import conv_dgrad_smallc_wmma_spec_to_dict

        specs = _aot_specs("direct_dgrad_smallc_wmma")
        self.assertTrue(any(dgrad_smallc_reason(s) is None for s in specs))
        base = dict(waves_c=1, waves_p=4, tiles_c=2, tiles_p=6, out_h=16, out_w=16)
        for pkw, knobs in _SMALLC_REJECTS:
            spec = DgradSmallcWmmaSpec(problem=_problem(**pkw), **{**base, **knobs})
            self.assertIsNotNone(dgrad_smallc_reason(spec), (pkw, knobs))
            specs.append(spec)
        self._agree(
            eng.conv_dgrad_smallc_wmma_describe,
            conv_dgrad_smallc_wmma_spec_to_dict,
            dgrad_smallc_reason,
            specs,
        )

    def test_binding_rejects_bad_input(self):
        """A rejected spec returns the rejection (no grid is computed from its
        zero tile extents); a missing, mistyped or partial problem raises."""
        eng = pytest.importorskip("rocke_engine")
        from rocke.core.backend import (
            conv_dgrad_patch_wmma_spec_to_dict,
            conv_dgrad_smallc_wmma_spec_to_dict,
        )

        good = (
            (
                "conv_dgrad_patch_wmma",
                conv_dgrad_patch_wmma_spec_to_dict,
                DgradPatchWmmaSpec(
                    problem=_problem(), waves_c=3, waves_p=1, tiles_c=1, r_blk=4
                ),
            ),
            (
                "conv_dgrad_smallc_wmma",
                conv_dgrad_smallc_wmma_spec_to_dict,
                DgradSmallcWmmaSpec(
                    problem=_problem(KH=3, S=1, PAD=1),
                    tiles_c=2,
                    tiles_p=6,
                    out_h=16,
                    out_w=16,
                ),
            ),
        )
        reason_of = {
            "conv_dgrad_patch_wmma": dgrad_patch_reason,
            "conv_dgrad_smallc_wmma": dgrad_smallc_reason,
        }
        for fam, to_dict, spec in good:
            describe = getattr(eng, f"{fam}_describe")
            lower = getattr(eng, f"{fam}_lower_llvm")
            self.assertTrue(describe(to_dict(spec), "gfx1250")[0], fam)
            for knobs in (dict(waves_c=0), dict(tiles_c=0), dict(waves_p=0)):
                sd = to_dict(spec)
                sd.update(knobs)
                ok, why, _, grid, _ = describe(sd, "gfx1250")
                self.assertFalse(ok, (fam, knobs))
                self.assertIn("waves", why)
                self.assertEqual(tuple(grid), (0, 0, 0))
                with self.subTest(fam=fam, knobs=knobs), self.assertRaises(
                    RuntimeError
                ):
                    lower(sd, arch="gfx1250")
            partial = dict(to_dict(spec)["problem"])
            del partial["kpg"]
            for problem in (None, "N2H32", partial):
                sd = to_dict(spec)
                if problem is None:
                    del sd["problem"]
                else:
                    sd["problem"] = problem
                with self.subTest(fam=fam, problem=problem), self.assertRaises(
                    ValueError
                ):
                    describe(sd, "gfx1250")
                with self.subTest(fam=fam, problem=problem), self.assertRaises(
                    ValueError
                ):
                    lower(sd, arch="gfx1250")
            # An explicit None dtype is refused, as Python's validator does,
            # rather than read as the fp16 default; an omitted one is fp16.
            self.assertIn("dtype None", reason_of[fam](_with_dtype(spec, None)))
            sd = to_dict(spec)
            sd["problem"]["dtype"] = None
            with self.subTest(fam=fam, dtype=None), self.assertRaises(ValueError):
                describe(sd, "gfx1250")
            with self.subTest(fam=fam, dtype=None), self.assertRaises(ValueError):
                lower(sd, arch="gfx1250")
            del sd["problem"]["dtype"]
            ok, _, name, _, _ = describe(sd, "gfx1250")
            self.assertTrue(ok, fam)
            self.assertEqual(name, _with_dtype(spec, "fp16").kernel_name(), fam)


# Real problems served through the AOT cache, per family: (variant, dtype,
# problem, uncovered border). The 16x16 patch embedding covers 224 of 230 rows
# and 231 columns, so its plans add the border-zeroing workgroups.
_STEM_ARCH = "gfx1250"
_CACHED_CASES = (
    (
        "direct_dgrad_patch_wmma",
        "bf16",
        dict(N=2, H=230, W=231, K=64, KH=16, S=16),
        True,
    ),
    ("direct_dgrad_patch_wmma", "fp16", dict(N=2, H=64, W=72, K=96), False),
    (
        "direct_dgrad_smallc_wmma",
        "bf16",
        dict(N=2, H=50, W=70, K=24, KH=3, S=2, PAD=1),
        False,
    ),
    (
        "direct_dgrad_smallc_wmma",
        "fp16",
        dict(N=1, H=37, W=29, C=4, K=32, KH=3, S=1, PAD=1),
        False,
    ),
)


def _cached_jobs(variant: str, problem) -> list:
    """The AOT jobs of ``variant`` whose capability serves ``problem``."""
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelCache

    jobs = []
    for caps in dks.DIRECT_CAPABILITIES[variant][1]:
        for _, job in dks._caps_jobs(
            _STEM_ARCH, 32, variant, "dgrad", caps, problem.dtype
        )[1]:
            if KernelCache.supports_problem(job.identity, problem)[0]:
                jobs.append(job)
    return jobs


def _cached_plans(cache, variant: str, problem):
    """``variant``'s launch plans for ``problem`` from ``direct_plans``."""
    from benchmarks.common import direct_kernel_sweep as dks

    plans, rejected = dks.direct_plans(cache, problem, "dgrad", _STEM_ARCH)
    return [p for p in plans if p.identity.algorithm == variant], rejected


class TestStemCachedPlans(unittest.TestCase):
    """The shipped path: a real problem reaches each family through the AOT
    cache's capability match and per-shape re-validation (``direct_plans``),
    and each plan binds the kernel's own kernarg order, the grid (patch
    border workgroups included) and the block of its knobs. The cache holds
    placeholder binaries; ``TestStemGpu.test_cached_plans`` runs real ones."""

    def test_cached_plans(self):
        import json
        import tempfile
        from pathlib import Path

        from benchmarks.common import direct_kernel_sweep as dks
        from benchmarks.common.kernel_cache import KernelCache
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs

        builders = {
            "direct_dgrad_patch_wmma": build_dgrad_patch_wmma,
            "direct_dgrad_smallc_wmma": build_dgrad_smallc_wmma,
        }
        for variant, dtype, pkw, border in _CACHED_CASES:
            p = _problem(dtype=dtype, **pkw)
            with self.subTest(
                variant=variant, problem=pkw
            ), tempfile.TemporaryDirectory() as tmp:
                cache = KernelCache(Path(tmp), _STEM_ARCH)
                jobs = _cached_jobs(variant, p)
                self.assertTrue(jobs)
                for job in jobs:
                    cache.put(
                        job.identity,
                        b"placeholder",
                        {"kernel_name": job.identity.label()},
                    )
                plans, rejected = _cached_plans(cache, variant, p)
                self.assertTrue(plans, rejected)
                ptrs = {"dy": 1 << 32, "w": 2 << 32, "dx": 3 << 32}
                sizes = {
                    "dy": 2 * p.N * p.Ho * p.Wo * p.total_k,
                    "w": 2 * p.total_k * p.KH * p.KW * p.cpg,
                    "dx": 2 * p.N * p.H * p.W * p.total_c,
                }
                for plan in plans:
                    knobs = json.loads(plan.identity.knobs)
                    (step,) = plan.steps
                    self.assertEqual(step.buffers, ("dy", "w", "dx"))
                    self.assertEqual(
                        step.signature,
                        conv_direct_args_signature(dtype, direction="dgrad"),
                    )
                    self.assertEqual(
                        step.conv_args, ConvArgs.from_problem(p, direction="dgrad")
                    )
                    # The signature is the kernel's own kernarg order.
                    spec = dks.make_spec(variant, p, knobs, 32)
                    kernel = builders[variant](spec, _STEM_ARCH)
                    names = [e["name"] for e in step.signature]
                    self.assertEqual([a.name for a in kernel.params], names)
                    vals = dks.launch_values(step, ptrs, sizes)
                    self.assertEqual(set(vals), set(names))
                    self.assertEqual(
                        [vals[n] for n in names[:3]],
                        [ptrs["dy"], ptrs["w"], ptrs["dx"]],
                    )
                    # Grid and block from the knobs alone.
                    wc, wp = knobs["waves_c"], knobs["waves_p"]
                    self.assertEqual(step.block, (wc * wp * 32, 1, 1))
                    if variant == "direct_dgrad_patch_wmma":
                        P, tw = p.KH, knobs["tile_w"]
                        th = wp * knobs["tiles_p"] * 16 // tw
                        main = (
                            p.N
                            * -(-p.Ho // th)
                            * -(-p.Wo // tw)
                            * (P // knobs["r_blk"])
                        )
                        self.assertEqual(p.Ho * P < p.H or p.Wo * P < p.W, border, pkw)
                        self.assertEqual(
                            step.grid, (main + (p.N if border else 0), 1, 1)
                        )
                    else:
                        self.assertEqual(
                            step.grid,
                            (
                                p.N
                                * -(-p.H // knobs["out_h"])
                                * -(-p.W // knobs["out_w"]),
                                1,
                                1,
                            ),
                        )


def _with_dtype(spec, dtype):
    return dataclasses.replace(
        spec, problem=dataclasses.replace(spec.problem, dtype=dtype)
    )


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


_PATCH_CASES = (
    (dict(), dict(waves_c=3, waves_p=1, tiles_c=1, tiles_p=4, tile_w=8, r_blk=4)),
    (
        dict(dtype="fp16", K=40),
        dict(waves_c=1, waves_p=2, tiles_c=1, tiles_p=2, r_blk=1),
    ),
    # 14x14 over 45x50: an uncovered border; padded columns (2*42 -> 96).
    (
        dict(N=3, H=45, W=50, KH=14, S=14, K=96),
        dict(waves_c=2, waves_p=1, tiles_c=3, tiles_p=1, tile_w=4, r_blk=2),
    ),
    (
        dict(N=2, H=64, W=32, KH=16, S=16, K=72),
        dict(
            waves_c=2,
            waves_p=1,
            tiles_c=6,
            tiles_p=1,
            tile_w=4,
            r_blk=4,
            ck=64,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
)
_SMALLC_CASES = (
    (
        dict(KH=3, S=1, PAD=1, C=4, H=37, W=29),
        dict(waves_c=1, waves_p=4, tiles_c=3, tiles_p=6, out_h=16, out_w=16),
    ),
    (
        dict(KH=3, S=2, PAD=1, K=24, H=50, W=70),
        dict(waves_c=2, waves_p=4, tiles_c=1, tiles_p=5, out_h=32, out_w=32),
    ),
    (
        dict(KH=7, S=2, PAD=3, H=40, W=36, dtype="fp16"),
        dict(
            waves_c=2,
            waves_p=2,
            tiles_c=5,
            tiles_p=4,
            out_h=16,
            out_w=16,
            ck=64,
            prefetch=True,
            sched="iterative-ilp",
        ),
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestStemGpu(unittest.TestCase):
    def _run(self, p, launch, label):
        """Fill dY / W, poison dX, ``launch(ptrs, sizes)`` and check dX
        against torch's dgrad."""
        from rocke.runtime.hip_module import Runtime

        td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k).uniform_(-1, 1).to(td)
        w = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(td)
        dX = torch.zeros(p.N, p.H, p.W, p.total_c, dtype=td)
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        sizes = {"dy": dY.nbytes, "w": w.nbytes, "dx": dX.nbytes}
        ptrs = {name: rt.alloc(nb) for name, nb in sizes.items()}
        try:
            rt.memcpy_h2d(ptrs["dy"], u8(dY), dY.nbytes)
            rt.memcpy_h2d(ptrs["w"], u8(w), w.nbytes)
            rt.memset(ptrs["dx"], 0x7F, dX.nbytes)  # poison: every element is written
            launch({k: int(v) for k, v in ptrs.items()}, sizes)
            rt.memcpy_d2h(u8(dX), ptrs["dx"], dX.nbytes)
        finally:
            for x in ptrs.values():
                rt.free(x)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.total_c, p.H, p.W),
            w.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            stride=p.stride,
            padding=p.PAD,
        ).permute(0, 2, 3, 1)
        err = float((dX.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{label}: rel err {err:.3g}")

    def _check(self, spec, build):
        from kernels.common.conv_abi import conv_direct_args_signature
        from kernels.common.conv_args import ConvArgs
        from rocke import compile_kernel
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        p = spec.problem
        art = compile_kernel(build(spec, GPU_ARCH), arch=GPU_ARCH)

        def launch(ptrs, sizes):
            vals = ConvArgs.from_problem(p, direction="dgrad").to_launch_values(
                ptrs["dy"], ptrs["w"], ptrs["dx"], sizes["dy"], sizes["w"], sizes["dx"]
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

        self._run(p, launch, str(spec))

    def test_cached_plans(self):
        """The shipped path on the GPU: each case's AOT cell is compiled into
        a cache as --compile-all does, and every plan ``direct_plans`` offers
        runs with its own kernargs, grid and block."""
        import tempfile
        from pathlib import Path

        from benchmarks.common import direct_kernel_sweep as dks
        from benchmarks.common.kernel_cache import KernelCache
        from benchmarks.common.kernel_sweep import compile_jobs
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        for variant, dtype, pkw, _ in _CACHED_CASES:
            p = _problem(dtype=dtype, **pkw)
            with tempfile.TemporaryDirectory() as tmp:
                cache = KernelCache(Path(tmp), GPU_ARCH)
                compile_jobs(
                    cache=cache,
                    all_jobs=_cached_jobs(variant, p),
                    build=dks.direct_kernel_for_job,
                    arch=GPU_ARCH,
                    dtype=dtype,
                    directions=["direct_dgrad"],
                    jobs=8,
                    log=lambda *_: None,
                )
                plans, rejected = _cached_plans(cache, variant, p)
                self.assertTrue(plans, (variant, pkw, rejected[:2]))
                for plan in plans:
                    (step,) = plan.steps

                    def launch(ptrs, sizes, step=step):
                        KernelLauncher(
                            hsaco=step.hsaco_path.read_bytes(),
                            kernel_name=step.kernel_name,
                            signature=step.signature,
                        )(
                            dks.launch_values(step, ptrs, sizes),
                            config=LaunchConfig(
                                grid=step.grid, block=step.block, fence=True
                            ),
                        )

                    with self.subTest(variant=variant, kernel=plan.label):
                        self._run(p, launch, plan.label)

    def test_patch(self):
        for pkw, knobs in _PATCH_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DgradPatchWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_patch_reason(spec, GPU_ARCH))
                self._check(spec, build_dgrad_patch_wmma)

    def test_smallc(self):
        for pkw, knobs in _SMALLC_CASES:
            with self.subTest(problem=pkw, knobs=knobs):
                spec = DgradSmallcWmmaSpec(problem=_problem(**pkw), **knobs)
                self.assertIsNone(dgrad_smallc_reason(spec, GPU_ARCH))
                self._check(spec, build_dgrad_smallc_wmma)
