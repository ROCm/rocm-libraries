# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the Winograd K3 batched GEMM (``conv_winograd_gemm_wmma``).

CPU: the spec / problem validators, launch geometry, split resolution, the
launch plans of ``benchmarks.common.winograd_k3_wmma`` and IR builds of every
knob. GPU (gfx1250): numeric checks against a float32 torch matmul, with M
poisoned (0x7F bytes) first so an unwritten element fails -- tails in T, Kg
and Cg, bf16, both epilogues and LDS pads, a 3-deep ring, one k-step per
chunk, split-K with the reduce kernel, and a two-step batch-range plan.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \\
        rocke/library/tests/test_conv_winograd_gemm_wmma.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it.
    import torch

    torch.cuda.is_available()

from kernels.common.conv_winograd_gemm_wmma import (  # noqa: E402
    WinogradGemmProblem,
    WinogradGemmReduceSpec,
    WinogradGemmWmmaSpec,
    build_winograd_gemm_reduce,
    build_winograd_gemm_wmma,
    winograd_gemm_problem_reason,
    winograd_gemm_reason,
    winograd_gemm_signature,
)

GPU_ARCH = get_device_arch(0)
_TINY = dict(waves_m=1, waves_n=2, frags_m=2, frags_n=1, tile_k=32, stages=2)


def _spec(**kw) -> WinogradGemmWmmaSpec:
    knobs = dict(_TINY)
    knobs.update(kw)
    return WinogradGemmWmmaSpec(**knobs)


class TestSpec(unittest.TestCase):
    def test_default_is_valid(self):
        spec = WinogradGemmWmmaSpec()
        self.assertIsNone(winograd_gemm_reason(spec))
        self.assertEqual(
            (spec.tile_m, spec.tile_n, spec.threads_per_block), (256, 256, 256)
        )
        self.assertLessEqual(spec.lds_bytes(), 320 * 1024)

    def test_rejects(self):
        for kw, why in (
            (dict(dtype="fp32"), "dtype"),
            (dict(tile_k=48), "tile_k"),
            (dict(tile_k=96), "power-of-two"),  # lds_pad tdm
            (dict(stages=1), "stages"),
            (dict(epilogue="cshuffle"), "epilogue"),
            (dict(lds_pad="none"), "lds_pad"),
            (dict(frags_m=0), "fragments"),
            (dict(waves_m=8, waves_n=8), "threads"),
            (dict(waves_m=2, waves_n=4, frags_m=8, frags_n=5), "VGPR"),
            (
                dict(waves_m=2, waves_n=4, frags_m=8, frags_n=4, tile_k=128, stages=3),
                "LDS",
            ),
            (dict(sched="no-such-strategy"), "sched"),
        ):
            with self.subTest(kw=kw):
                why_got = winograd_gemm_reason(_spec(**kw))
                self.assertIsNotNone(why_got)
                self.assertIn(why.lower(), why_got.lower())
        self.assertIsNone(winograd_gemm_reason(_spec(tile_k=96, lds_pad="clip")))
        self.assertIsNotNone(winograd_gemm_reason(_spec(), "gfx950"))

    def test_problem_reason(self):
        spec = _spec()
        ok = WinogradGemmProblem(T=77, G=3, Cg=72, Kg=40)
        self.assertIsNone(winograd_gemm_problem_reason(spec, ok))
        for prob in (
            WinogradGemmProblem(T=8, G=2, Cg=12, Kg=16),
            WinogradGemmProblem(T=8, G=2, Cg=16, Kg=20),
            WinogradGemmProblem(T=0, G=2, Cg=16, Kg=16),
            WinogradGemmProblem(T=1 << 20, G=36, Cg=512, Kg=64),  # > 2 GiB V
        ):
            with self.subTest(prob=prob):
                self.assertIsNotNone(winograd_gemm_problem_reason(spec, prob))
        for batches in ((0, 0), (2, 2), (-1, 1)):
            with self.subTest(batches=batches):
                self.assertIsNotNone(
                    winograd_gemm_problem_reason(spec, ok, batches=batches)
                )
        self.assertIsNone(winograd_gemm_problem_reason(spec, ok, batches=(1, 2)))

    def test_grid_and_splits(self):
        prob = WinogradGemmProblem(T=77, G=3, Cg=72, Kg=40)
        spec = _spec()  # 32 x 32 tiles, 3 chunks of 32
        self.assertEqual(spec.tiles(prob), (3, 2, 3))
        self.assertEqual(spec.grid(prob), (18, 1, 1))
        self.assertEqual(spec.grid(prob, batches=(1, 2)), (12, 1, 1))
        self.assertEqual(spec.resolve_splits(prob, 4), (1, 3))  # no split-K
        sk = _spec(split_k=True)
        self.assertEqual(sk.resolve_splits(prob, 2), (2, 2))
        self.assertEqual(sk.resolve_splits(prob, 9), (3, 1))  # clamped
        self.assertEqual(sk.grid(prob, 2), (36, 1, 1))
        self.assertEqual(sk.workspace_bytes(prob, 2), 4 * 2 * 77 * 3 * 40)
        vals = sk.launch_values(prob, 1, 2, 3, 2, batches=(1, 2))
        self.assertEqual(
            (vals["splits"], vals["cps"], vals["g0"], vals["nb"]), (2, 2, 1, 2)
        )
        names = [p["name"] for p in winograd_gemm_signature(sk)]
        self.assertEqual(list(vals), names)

    def test_from_conv(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        p = ConvProblem(N=8, Hi=32, Wi=32, C=1280, K=1280, Y=3, X=3, pH=1, pW=1)
        self.assertEqual(
            WinogradGemmProblem.from_conv(p, 4),
            WinogradGemmProblem(T=512, G=36, Cg=1280, Kg=1280),
        )
        g = ConvProblem(
            N=1, Hi=64, Wi=64, C=2048, K=2048, Y=3, X=3, pH=1, pW=1, groups=8
        )
        self.assertEqual(
            WinogradGemmProblem.from_conv(g, 4),
            WinogradGemmProblem(T=256, G=288, Cg=256, Kg=256),
        )

    def test_ir_builds(self):
        for kw in (
            dict(),
            dict(dtype="bf16"),
            dict(split_k=True),
            dict(epilogue="direct"),
            dict(epilogue="direct", split_k=True, dtype="bf16"),
            dict(lds_pad="clip", tile_k=96),
            dict(stages=3, prefetch=False, sched=None, xcd_remap=False),
            dict(waves_m=1, waves_n=1, frags_m=1, frags_n=1),
            dict(waves_m=2, waves_n=4, frags_m=8, frags_n=4, tile_k=128),
        ):
            with self.subTest(kw=kw):
                kernel = build_winograd_gemm_wmma(_spec(**kw), arch="gfx1250")
                self.assertTrue(kernel.name.startswith("wino_gemm_wmma"))
        build_winograd_gemm_reduce(WinogradGemmReduceSpec("bf16"), arch="gfx1250")
        with self.assertRaises(ValueError):
            build_winograd_gemm_wmma(_spec(tile_k=48), arch="gfx1250")


class TestPlans(unittest.TestCase):
    def test_candidate_plans(self):
        from benchmarks.common import winograd_k3_wmma as K3W

        prob = WinogradGemmProblem(T=512, G=36, Cg=1280, Kg=1280)
        big = WinogradGemmWmmaSpec()  # 10 blocks per batch
        small = _spec()
        # 25 batches fill one round of 256 blocks.
        self.assertEqual(K3W.main_batches(big, prob, 256), 25)
        self.assertEqual(K3W.main_batches(big, prob, 512), 0)  # no full round
        built = [({}, big, None), ({}, small, None)]
        plans = K3W.candidate_plans(prob, built, 256)
        self.assertEqual(len(plans), 2 + 2)  # two singles, big + {big, small}
        for plan in plans:
            self.assertIsNone(plan.reason(prob), plan.label())
        two = plans[2]
        self.assertEqual([s.batches for s in two.steps], [(0, 25), (25, 11)])
        self.assertEqual(len(K3W.candidate_plans(prob, built, 256, tails=False)), 2)
        bad = K3W.K3Plan([K3W.K3Step(big, None, (0, 20))])
        self.assertIn("covers", bad.reason(prob))


def _gpu_skip() -> str:
    if GPU_ARCH != "gfx1250":
        return f"needs gfx1250 (got {GPU_ARCH!r})"
    if not _HAS_TORCH:
        return "torch not importable"
    return ""


# (problem (T, G, Cg, Kg), dtype, plan steps [(knobs, batches, splits)])
_ONE = [(dict(), None, 1)]
_CASES = (
    ((77, 3, 72, 40), "fp16", _ONE),
    ((130, 2, 104, 56), "bf16", _ONE),
    ((77, 3, 72, 40), "fp16", [(dict(epilogue="direct"), None, 1)]),
    ((77, 3, 72, 40), "fp16", [(dict(lds_pad="clip", tile_k=64), None, 1)]),
    (
        (70, 2, 200, 48),
        "fp16",
        [
            (
                dict(stages=3, tile_k=64, prefetch=False, sched=None, xcd_remap=False),
                None,
                1,
            )
        ],
    ),
    (
        (40, 3, 96, 24),
        "fp16",
        [(dict(waves_m=1, waves_n=1, frags_m=1, frags_n=1), None, 1)],
    ),
    ((77, 3, 72, 40), "fp16", [(dict(split_k=True), None, 2)]),
    ((50, 2, 168, 32), "bf16", [(dict(split_k=True, epilogue="direct"), None, 3)]),
    (
        (300, 5, 160, 136),
        "fp16",
        [
            (dict(waves_m=2, waves_n=4, frags_m=8, frags_n=4, tile_k=128), (0, 3), 1),
            (dict(waves_m=2, waves_n=2, frags_m=2, frags_n=2, tile_k=64), (3, 2), 1),
        ],
    ),
)


@unittest.skipUnless(not _gpu_skip(), _gpu_skip() or "no GPU")
class TestGemmGpu(unittest.TestCase):
    def _check(self, dims, dtype, steps):
        from benchmarks.common import winograd_k3_wmma as K3W
        from rocke import compile_kernel
        from rocke.runtime.hip_module import Runtime

        T, G, Cg, Kg = dims
        prob = WinogradGemmProblem(T=T, G=G, Cg=Cg, Kg=Kg)
        plan = K3W.K3Plan()
        for knobs, batches, splits in steps:
            spec = _spec(dtype=dtype, **knobs)
            self.assertIsNone(winograd_gemm_reason(spec, GPU_ARCH))
            art = compile_kernel(
                build_winograd_gemm_wmma(spec, arch=GPU_ARCH), arch=GPU_ARCH
            )
            plan.steps.append(K3W.K3Step(spec, art, batches or (0, G), splits))
        self.assertIsNone(plan.reason(prob))
        reduce_art = None
        if any(s.spec.split_k for s in plan.steps):
            reduce_art = K3W.build_reduce(dtype, GPU_ARCH)
        td = torch.bfloat16 if dtype == "bf16" else torch.float16
        torch.manual_seed(0)
        V = torch.empty(T, G, Cg).uniform_(-1, 1).to(td)
        U = torch.empty(G, Kg, Cg).uniform_(-1, 1).to(td)
        M = torch.empty(T, G, Kg, dtype=td)
        # Pinned host buffers: pageable copies near in-flight kernels hang gfx1250.
        V, U, M = (t.pin_memory() for t in (V, U, M))
        rt = Runtime()

        def u8(t):
            return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

        ws_bytes = plan.workspace_bytes(prob)
        bufs = [rt.alloc(t.nbytes) for t in (V, U, M)] + [rt.alloc(max(ws_bytes, 4))]
        try:
            rt.sync()
            rt.memcpy_h2d(bufs[0], u8(V), V.nbytes)
            rt.memcpy_h2d(bufs[1], u8(U), U.nbytes)
            rt.memset(bufs[2], 0x7F, M.nbytes)
            rt.memset(bufs[3], 0x7F, max(ws_bytes, 4))
            run = plan.bind(
                prob,
                bufs[0],
                bufs[1],
                bufs[2],
                ws=bufs[3] if ws_bytes else 0,
                reduce_art=reduce_art,
            )
            run()
            rt.sync()
            rt.memcpy_d2h(u8(M), bufs[2], M.nbytes)
        finally:
            from rocke.runtime import synchronize_and_release

            synchronize_and_release(0)
            for b in bufs:
                rt.free(b)
        ref = torch.einsum("tgc,gkc->tgk", V.float(), U.float())
        err = float((M.float() - ref).abs().max() / ref.abs().max())
        self.assertLess(err, 1e-2, f"{dims} {dtype} {plan.label()}: rel err {err:.3g}")

    def test_cases(self):
        for dims, dtype, steps in _CASES:
            with self.subTest(dims=dims, dtype=dtype, steps=steps):
                self._check(dims, dtype, steps)


if __name__ == "__main__":
    unittest.main()
