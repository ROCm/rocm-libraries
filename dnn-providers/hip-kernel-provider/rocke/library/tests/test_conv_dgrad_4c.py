# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""4c (cpg = kpg = 4) direct conv: bf16 support and the 4c dgrad entry.

Covers:

* CPU only: spec plumbing (``make_dgrad_4c_spec`` / ``dgrad_4c_spec_for_problem``
  / ``is_valid_dgrad_4c_problem`` / ``direct_4c_dgrad_launch``), the
  ``stage_rows`` knob's name, validator and defaults, and that the bf16 4c
  kernel emits only the ``mfma_f32_4x4x4_bf16`` (``_1k``) atom while the fp16
  kernel keeps the f16 atom;
* on-device numerics (gfx942 / gfx950 + torch): the 4c forward kernel in bf16
  and the 4c dgrad in its forms -- the two-kernel pipeline (weight transpose
  + 4c kernel on dY), the single kernel with fused weights (per-element
  gathers, or LDS staging + transpose reads with ``dgrad_weights_lds`` on
  gfx950) and the row-staged kernel (``stage_rows``, gfx950) -- in fp16 and
  bf16 over adversarial shapes (odd H / W, N = 1, H = 1, W not a multiple of
  block_q, 1x1 filters, every block_q / block_groups knob), plus one binary per
  form built on a probe problem and launched on several shapes (the AOT
  contract: extents are kernargs).

Every launch goes through the direct forward ABI (``ConvArgs`` /
``conv_direct_args_signature``) with the grid of ``direct_launch_geometry``;
the dgrad runs it over the transposed problem, like the AOT plan does.

Elementwise rule (``manifest_runner/conv.py``): an output element is bad when
``|out - ref| > tol + tol * |ref|`` with ``tol = 1e-2``; a case passes iff no
element is bad. The output is NaN-filled before each launch, so an element the
kernel never writes is caught as bad too.

Run (GPU part as a script, as the other numeric suites):
    PYTHONPATH=rocke/platform/python:rocke/library python \\
        rocke/library/tests/test_conv_dgrad_4c.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    build_direct_4c_dgrad,
    build_direct_conv_4c,
    dgrad_4c_spec_for_problem,
    dgrad_4c_transposed_problem,
    direct_4c_dgrad_launch,
    direct_launch_geometry,
    is_valid_dgrad_4c_problem,
    is_valid_spec_4c,
    make_dgrad_4c_spec,
)
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_GPU_SKIP = (
    ""
    if (GPU_ARCH in ("gfx942", "gfx950") and _HAS_TORCH)
    else f"needs gfx942/gfx950 + torch (arch={GPU_ARCH!r}, torch={_HAS_TORCH})"
)
_TOL = 1e-2

_F16_ATOM = "@llvm.amdgcn.mfma.f32.4x4x4f16("
_BF16_ATOM = "@llvm.amdgcn.mfma.f32.4x4x4bf16.1k("


def _problem(N, H, W, groups, dtype="bf16", K=3, pad=None, stride=1, cpg=4, kpg=4):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2 if pad is None else pad,
        stride=stride,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# CPU-only: emitted atom and naming
# ---------------------------------------------------------------------------


class Test4cBf16Spec(unittest.TestCase):
    def test_fp16_kernel_name_has_no_dtype_flag(self):
        p = _problem(N=1, H=8, W=8, groups=16, dtype="fp16")
        name = DirectConv4cSpec(problem=p).kernel_name()
        self.assertEqual(name, "direct_conv_4c_N1H8W8_g16_c4k4_bq4_bg16")
        bf = DirectConv4cSpec(problem=_problem(N=1, H=8, W=8, groups=16))
        self.assertEqual(bf.kernel_name(), name + "_bf16")

    def test_bf16_atom_emitted(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                p = _problem(N=1, H=8, W=8, groups=16)
                ll = lower_kernel_to_llvm(
                    build_direct_conv_4c(DirectConv4cSpec(problem=p), arch=arch),
                    arch=arch,
                )
                self.assertIn(_BF16_ATOM, ll)
                self.assertNotIn(_F16_ATOM, ll)
                self.assertIn("<4 x bfloat>", ll)
                ll16 = lower_kernel_to_llvm(
                    build_direct_conv_4c(
                        DirectConv4cSpec(problem=_problem(1, 8, 8, 16, dtype="fp16")),
                        arch=arch,
                    ),
                    arch=arch,
                )
                self.assertIn(_F16_ATOM, ll16)
                self.assertNotIn(_BF16_ATOM, ll16)

    def test_launch_geometry(self):
        p = _problem(N=3, H=13, W=17, groups=64)
        spec = DirectConv4cSpec(problem=p, block_q=8, block_groups=32)
        self.assertEqual(direct_launch_geometry(spec), ((3, 2, 3), (128, 1, 1)))


class TestDgrad4cSpec(unittest.TestCase):
    def test_make_dgrad_4c_spec_transposes_problem(self):
        p = _problem(N=3, H=13, W=17, groups=32)
        s = make_dgrad_4c_spec(p, block_q=8, block_groups=32)
        tp = s.problem
        self.assertEqual(tp, dgrad_4c_transposed_problem(p))
        self.assertEqual((tp.N, tp.H, tp.W), (3, 13, 17))
        self.assertEqual((tp.cpg, tp.kpg, tp.groups), (4, 4, 32))
        self.assertEqual((tp.KH, tp.KW, tp.PAD, tp.stride), (3, 3, 1, 1))
        self.assertEqual(tp.dtype, "bf16")
        self.assertEqual((s.block_q, s.block_groups), (8, 32))
        self.assertTrue(s.kernel_name().startswith("direct_conv_4c_dgrad_"))
        self.assertTrue(s.kernel_name().endswith("_bf16"))
        s.validate()

    def test_invalid_problems_rejected(self):
        cases = {
            "stride": _problem(N=1, H=8, W=8, groups=16, stride=2),
            "cpg": _problem(N=1, H=8, W=8, groups=16, cpg=8, kpg=8),
            "asym": _problem(N=1, H=8, W=8, groups=16, cpg=4, kpg=8),
            "5x5": _problem(N=1, H=8, W=8, groups=16, K=5),
            "pad0": _problem(N=1, H=8, W=8, groups=16, K=3, pad=0),
            "groups": _problem(N=1, H=8, W=8, groups=8),
            "fp32": _problem(N=1, H=8, W=8, groups=16, dtype="fp32"),
        }
        for tag, p in cases.items():
            with self.subTest(case=tag):
                ok, why = is_valid_dgrad_4c_problem(p)
                self.assertFalse(ok)
                self.assertTrue(why)
                self.assertIsNone(dgrad_4c_spec_for_problem(p))
                with self.assertRaises(ValueError):
                    make_dgrad_4c_spec(p)

    def test_tap_bound_and_lds_staging_passes(self):
        # Python and C++ share DCONV4C_MAX_TAPS (the C++ builder's per-tap
        # arrays are sized by it). With KH*KW <= 16 the LDS staging pass
        # count stays well under DCONV4C_MAX_WL_PASSES, so the tap bound is
        # the reject that fires for large filters.
        from kernels.common.conv_direct_grouped import (
            DCONV4C_MAX_TAPS,
            DCONV4C_MAX_WL_PASSES,
        )

        for K, want in ((1, True), (3, True), (5, False), (7, False), (9, False)):
            p = _problem(N=1, H=16, W=16, groups=16, K=K)
            s = DirectConv4cSpec(
                problem=p, dgrad_fused_weights=True, dgrad_weights_lds=True
            )
            ok, why = is_valid_spec_4c(s, arch="gfx950")
            with self.subTest(K=K):
                self.assertEqual(ok, want, why)
                if not want:
                    self.assertIn(f"KH*KW <= {DCONV4C_MAX_TAPS}", why)
        # Worst legal case (16 taps) vs the staging-pass bound.
        for bg in (16, 32, 64):
            threads = (bg // 16) * 64
            passes = -(-(bg * 4 * DCONV4C_MAX_TAPS * 4 // 8) // threads)
            self.assertLessEqual(passes, DCONV4C_MAX_WL_PASSES)

    @unittest.skipUnless(_HAS_TORCH, "needs torch")
    def test_check_counts_nan_as_bad(self):
        ref = torch.zeros(8)
        out = ref.clone()
        out[3] = float("nan")
        ok, msg = _check(out, ref)
        self.assertFalse(ok)
        self.assertIn("bad=1", msg)
        self.assertTrue(_check(ref.clone(), ref)[0])

    def test_spec_helper(self):
        p = _problem(N=6, H=30, W=26, groups=32)
        s = dgrad_4c_spec_for_problem(p)
        self.assertIsNotNone(s)
        self.assertTrue(is_valid_spec_4c(s, arch="gfx950")[0])
        # Single-kernel form by default: fused weights, LDS-staged where the
        # target has transpose reads (gfx950) and gathers otherwise (gfx942).
        self.assertTrue(s.dgrad_fused_weights and s.dgrad_weights_lds)
        s942 = dgrad_4c_spec_for_problem(p, arch="gfx942")
        self.assertTrue(s942.dgrad_fused_weights)
        self.assertFalse(s942.dgrad_weights_lds)
        pre = dgrad_4c_spec_for_problem(p, fused_weights=False)
        self.assertFalse(pre.dgrad_fused_weights or pre.dgrad_weights_lds)
        # An explicit knob that does not divide groups is rejected, not rewritten.
        self.assertIsNone(
            dgrad_4c_spec_for_problem(
                _problem(N=2, H=8, W=8, groups=16), block_groups=32
            )
        )
        self.assertIsNone(dgrad_4c_spec_for_problem(p, arch="gfx000"))

    def test_fused_spec_plumbing(self):
        p = _problem(N=2, H=13, W=13, groups=32)
        s = make_dgrad_4c_spec(p, block_q=8, block_groups=32, dgrad_fused_weights=True)
        self.assertTrue(s.kernel_name().endswith("_bf16_fw"))
        sl = replace(s, dgrad_weights_lds=True)
        self.assertTrue(sl.kernel_name().endswith("_bf16_fwl"))
        L = direct_4c_dgrad_launch(sl)
        self.assertIsNone(L["transpose_grid"])
        self.assertIsNone(L["transpose_block"])
        self.assertEqual(L["workspace_bytes"], 0)
        self.assertEqual(L["grid"], (2, 1, 2))
        kt, km = build_direct_4c_dgrad(sl, arch="gfx950")
        self.assertIsNone(kt)
        self.assertEqual(km.name, sl.kernel_name())
        # LDS staging needs the fused form and ds_read_b64_tr_b16 (gfx950).
        bad = make_dgrad_4c_spec(p, dgrad_weights_lds=True)
        with self.assertRaises(ValueError):
            bad.validate()
        self.assertFalse(is_valid_spec_4c(bad, arch="gfx950")[0])
        ok942, why = is_valid_spec_4c(sl, arch="gfx942")
        self.assertFalse(ok942)
        self.assertIn("ds_read_b64_tr_b16", why)
        self.assertTrue(is_valid_spec_4c(s, arch="gfx942")[0])

    def test_fused_weights_emission(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        p = _problem(N=1, H=8, W=8, groups=32)
        base = make_dgrad_4c_spec(p, block_groups=32)
        gather = replace(base, dgrad_fused_weights=True)
        lds = replace(gather, dgrad_weights_lds=True)
        ll_g = lower_kernel_to_llvm(
            build_direct_conv_4c(gather, arch="gfx950"), arch="gfx950"
        )
        ll_l = lower_kernel_to_llvm(
            build_direct_conv_4c(lds, arch="gfx950"), arch="gfx950"
        )
        self.assertNotIn("ds.read.tr16.b64", ll_g)
        self.assertIn("raw.ptr.buffer.load.i16", ll_g)
        # One transpose read per tap, one barrier for the staged slice.
        self.assertEqual(ll_l.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64("), 9)
        self.assertEqual(ll_l.count("call void @llvm.amdgcn.s.barrier()"), 1)

    def test_launch_geometry(self):
        p = _problem(N=2, H=13, W=13, groups=32)
        s = make_dgrad_4c_spec(p, block_q=8, block_groups=32)
        L = direct_4c_dgrad_launch(s)
        self.assertEqual(L["grid"], (2, 1, 2))
        self.assertEqual(L["block"], (128, 1, 1))
        # The pre-pass: one (group, r, s) per x, a 64-wide kpg tile per y,
        # one cpg per z; W_T holds 128 channels x 9 taps x kpg 4.
        self.assertEqual(L["transpose_grid"], (32 * 9, 1, 4))
        self.assertEqual(L["transpose_block"], (64, 1, 1))
        self.assertEqual(L["workspace_bytes"], 128 * 9 * 4 * 2)

    def test_dgrad_pipeline_builds(self):
        p = _problem(N=1, H=7, W=7, groups=16)
        s = make_dgrad_4c_spec(p)
        kt, km = build_direct_4c_dgrad(s, arch="gfx950")
        self.assertIn("transpose_weights_dgrad", kt.name)
        self.assertEqual(km.name, s.kernel_name())


class Test4cStageRowsSpec(unittest.TestCase):
    """CPU only: the ``stage_rows`` knob's name, validator and defaults."""

    def _staged(self, p, bq=4, bg=16, **kw):
        s = make_dgrad_4c_spec(
            p,
            block_q=bq,
            block_groups=bg,
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
            stage_rows=True,
        )
        return replace(s, **kw) if kw else s

    def test_name_and_threads(self):
        p = _problem(N=2, H=13, W=13, groups=32)
        s = self._staged(p)
        self.assertTrue(s.kernel_name().endswith("_bf16_fwl_sr1"))
        self.assertEqual(s.threads_per_block, 64)
        s2 = self._staged(p, bq=8, bg=32)
        self.assertEqual(s2.waves_q, 2)
        self.assertTrue(s2.kernel_name().endswith("_bq8_bg32_bf16_fwl_sr2"))
        self.assertEqual(s2.threads_per_block, 256)
        self.assertEqual(direct_4c_dgrad_launch(s2)["block"], (256, 1, 1))
        # Default off: the name of an unstaged spec is unchanged.
        plain = make_dgrad_4c_spec(p, dgrad_fused_weights=True, dgrad_weights_lds=True)
        self.assertFalse(plain.kernel_name().endswith("_sr1"))
        self.assertEqual(plain.threads_per_block, 64)
        s.validate()
        self.assertTrue(is_valid_spec_4c(s, arch="gfx950")[0])

    def test_rejects(self):
        p = _problem(N=2, H=13, W=13, groups=32)
        same = DirectConvProblem(
            N=1, H=8, W=8, groups=16, cpg=4, kpg=4, KH=3, KW=3, PAD=0
        )
        cases = {
            "no_lds": (
                replace(self._staged(p), dgrad_weights_lds=False),
                "stage_rows needs dgrad_fused_weights and dgrad_weights_lds",
            ),
            "block_q": (
                replace(self._staged(p), block_q=8),
                "stage_rows needs block_q == 4*waves_q",
            ),
            "waves_q0": (replace(self._staged(p), waves_q=0), "waves_q must be >= 1"),
            "waves_q_unstaged": (
                replace(self._staged(p, bq=8), stage_rows=False),
                "waves_q > 1 needs stage_rows",
            ),
            "not_same_pad": (
                DirectConv4cSpec(
                    problem=same,
                    dgrad_fused_weights=True,
                    dgrad_weights_lds=True,
                    stage_rows=True,
                ),
                "'same' padding",
            ),
            "threads": (
                self._staged(p, bq=64, bg=32),
                "stage_rows needs threads_per_block <= 1024",
            ),
        }
        for tag, (s, want) in cases.items():
            with self.subTest(case=tag):
                with self.assertRaises(ValueError) as cm:
                    s.validate()
                self.assertIn(want, str(cm.exception))
                ok, why = is_valid_spec_4c(s, arch="gfx950")
                self.assertFalse(ok)
                self.assertIn(want, why)
        # Transpose LDS reads are gfx950-only, so gfx942 rejects stage_rows.
        ok, why = is_valid_spec_4c(self._staged(p), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("ds_read_b64_tr_b16", why)

    def test_helper_default(self):
        p = _problem(N=2, H=13, W=13, groups=32)
        s = dgrad_4c_spec_for_problem(p)
        self.assertTrue(s.stage_rows)
        self.assertEqual(s.waves_q, 1)
        self.assertFalse(dgrad_4c_spec_for_problem(p, stage_rows=False).stage_rows)
        self.assertFalse(dgrad_4c_spec_for_problem(p, arch="gfx942").stage_rows)
        self.assertIsNone(dgrad_4c_spec_for_problem(p, arch="gfx942", stage_rows=True))
        # Auto falls back to the direct-load kernel past the thread limit.
        big = dgrad_4c_spec_for_problem(
            _problem(N=1, H=8, W=64, groups=64), block_q=32, block_groups=64
        )
        self.assertFalse(big.stage_rows)

    def test_fprop_unaffected(self):
        # The fprop 4c spec never takes the knob (it needs the dgrad fused
        # weight form), so its default name and launch width are unchanged.
        s = DirectConv4cSpec(problem=_problem(N=1, H=8, W=8, groups=16, dtype="fp16"))
        self.assertFalse(s.stage_rows)
        self.assertEqual(s.kernel_name(), "direct_conv_4c_N1H8W8_g16_c4k4_bq4_bg16")
        self.assertEqual(s.threads_per_block, 64)
        with self.assertRaises(ValueError):
            replace(s, stage_rows=True).validate()

    def test_staged_emission(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        p = _problem(N=1, H=8, W=8, groups=32)
        ll = lower_kernel_to_llvm(
            build_direct_conv_4c(self._staged(p, bg=32), arch="gfx950"), arch="gfx950"
        )
        # 16-byte row loads, ds_read_b64 fragments, one transpose read per tap.
        self.assertIn("raw.ptr.buffer.load.v4i32", ll)
        self.assertEqual(ll.count("call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64("), 9)
        self.assertNotIn("raw.ptr.buffer.load.v2i32", ll)

    def test_row_stream_is_a_runtime_loop(self):
        # The image height is a kernarg: the rows stream through one scf.for
        # whose body is unrolled lcm(KH, 2) rows, whatever the image height.
        p = _problem(N=1, H=8, W=8, groups=32)
        for K, unroll in ((3, 6), (1, 2)):
            k = build_direct_conv_4c(
                self._staged(_problem(1, 8, 8, 32, K=K), bg=32), arch="gfx950"
            )
            loops = [op for op in k.body.ops if op.name == "scf.for"]
            self.assertEqual(len(loops), 1)
            body = loops[0].regions[0].ops
            n_mma = sum(1 for op in body if op.name == "tile.mma")
            self.assertEqual(n_mma, unroll * K * K)
            self.assertEqual(sum(1 for op in body if op.name == "tile.sync"), unroll)
        self.assertTrue(p.H)


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _check(out, ref) -> tuple[bool, str]:
    out_f = out.float().cpu()
    ref_f = ref.float().cpu()
    diff = (out_f - ref_f).abs()
    # NaN-safe: a NaN (e.g. unwritten, NaN-filled) element fails `<=`, so it
    # counts as bad; `diff > tol` would let it pass.
    n_bad = int((~(diff <= _TOL + _TOL * ref_f.abs())).sum())
    max_abs = float(diff.nan_to_num(nan=float("inf")).max())
    return n_bad == 0, f"bad={n_bad} max_abs={max_abs:.3e}"


def _rand(shape, dtype):
    td = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.empty(*shape, dtype=td).uniform_(-1.0, 1.0)


def _compile(kernel):
    from rocke import compile_kernel

    return compile_kernel(kernel, arch=GPU_ARCH)


def _launch_direct(art, spec, run_problem, abi_problem, a, b, d, direction="fwd"):
    """Launch ``art`` (built from ``spec``'s shape class) on ``run_problem``.

    ``abi_problem`` is the problem the kernarg block describes (for a dgrad run
    on the transposed problem this is the transposed one); ``a`` / ``b`` / ``d``
    are host tensors bound to the A, B and D slots. Returns the D tensor.
    """
    from dataclasses import replace

    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (a, b, d)]
    rt.memcpy_h2d(bufs[0], _u8(a), a.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(b), b.nbytes)
    # NaN-fill the output so an unwritten element can never pass.
    rt.memset(bufs[2], 0xFF, d.nbytes)
    grid, block = direct_launch_geometry(replace(spec, problem=abi_problem))
    values = ConvArgs.from_problem(abi_problem, direction=direction).to_launch_values(
        int(bufs[0]), int(bufs[1]), int(bufs[2]), a.nbytes, b.nbytes, d.nbytes
    )
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(run_problem.dtype, direction=direction),
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
    rt.memcpy_d2h(_u8(d), bufs[2], d.nbytes)
    for buf in bufs:
        rt.free(buf)
    return d


def _run_4c_fprop(spec: DirectConv4cSpec, p: DirectConvProblem, art=None):
    """Run the 4c forward kernel of ``spec`` (built once, on any shape of its
    class) on ``p``; ``(ok, msg)``."""
    if art is None:
        art = _compile(build_direct_conv_4c(spec, arch=GPU_ARCH))
    torch.manual_seed(0)
    x = _rand((p.N, p.H, p.W, p.total_c), p.dtype)
    w = _rand((p.total_k, p.KH, p.KW, p.cpg), p.dtype)
    y = torch.zeros(p.N, p.Ho, p.Wo, p.total_k, dtype=x.dtype)
    ref = torch.nn.functional.conv2d(
        x.float().permute(0, 3, 1, 2),
        w.float().permute(0, 3, 1, 2),
        padding=p.PAD,
        groups=p.groups,
    ).permute(0, 2, 3, 1)
    y = _launch_direct(art, spec, p, p, x, w, y)
    return _check(y, ref)


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class Test4cBf16Fprop(unittest.TestCase):
    def test_bf16_fprop(self):
        cases = [
            (_problem(N=2, H=14, W=14, groups=16), 4, 16),
            (_problem(N=1, H=13, W=13, groups=32), 8, 32),
            (_problem(N=1, H=7, W=17, groups=64), 16, 64),
            (_problem(N=2, H=15, W=15, groups=32, K=1), 32, 16),
        ]
        for p, bq, bg in cases:
            with self.subTest(p=p.short(), K=p.KH, bq=bq, bg=bg):
                spec = DirectConv4cSpec(problem=p, block_q=bq, block_groups=bg)
                ok, msg = _run_4c_fprop(spec, p)
                self.assertTrue(ok, msg)

    def test_one_binary_serves_every_shape(self):
        # Built once on a probe problem, launched on unrelated extents.
        probe = _problem(N=1, H=64, W=64, groups=64)
        spec = DirectConv4cSpec(problem=probe, block_q=8, block_groups=16)
        art = _compile(build_direct_conv_4c(spec, arch=GPU_ARCH))
        for p in (
            _problem(N=2, H=5, W=11, groups=16),
            _problem(N=1, H=17, W=3, groups=48),
            _problem(N=3, H=9, W=24, groups=32),
        ):
            with self.subTest(p=p.short()):
                ok, msg = _run_4c_fprop(spec, p, art=art)
                self.assertTrue(ok, msg)


#: dgrad forms exercised on the device: the two-kernel pre-pass pipeline and
#: the fused-weight single kernel with gathers or (gfx950) LDS staging, the
#: latter also in its row-staged form (``stage_rows``).
_MODES = ("prepass", "gather") + (("lds", "staged") if GPU_ARCH == "gfx950" else ())


def _dgrad_spec(p, block_q=None, block_groups=None, mode="prepass"):
    spec = dgrad_4c_spec_for_problem(
        p,
        arch=GPU_ARCH,
        block_q=block_q,
        block_groups=block_groups,
        fused_weights=mode != "prepass",
        stage_rows=mode == "staged",
    )
    if spec is None or mode == "staged":
        return spec
    return replace(spec, dgrad_weights_lds=mode == "lds")


def _run_4c_dgrad(p: DirectConvProblem, spec: DirectConv4cSpec, arts=None):
    """dgrad of ``p`` through the 4c pipeline of ``spec`` (built on any problem
    of its shape class, or taken from ``arts``); ``(ok, msg)``."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if arts is None:
        kt, km = build_direct_4c_dgrad(spec, arch=GPU_ARCH)
        arts = (_compile(kt) if kt is not None else None, _compile(km))
    art_t, art_m = arts
    tp = dgrad_4c_transposed_problem(p)
    run_spec = replace(spec, problem=tp)

    torch.manual_seed(1)
    dy = _rand((p.N, p.Ho, p.Wo, p.total_k), p.dtype)
    w = _rand((p.total_k, p.KH, p.KW, p.cpg), p.dtype)
    dx = torch.zeros(p.N, p.H, p.W, p.total_c, dtype=dy.dtype)
    ref = torch.nn.grad.conv2d_input(
        (p.N, p.total_c, p.H, p.W),
        w.float().permute(0, 3, 1, 2),
        dy.float().permute(0, 3, 1, 2),
        stride=1,
        padding=p.PAD,
        groups=p.groups,
    ).permute(0, 2, 3, 1)
    b_host = w
    if art_t is not None:
        # The weight pre-pass writes W_T, which the 4c kernel reads as B.
        L = direct_4c_dgrad_launch(run_spec)
        rt = Runtime()
        d_w, d_ws = rt.alloc(w.nbytes), rt.alloc(L["workspace_bytes"])
        rt.memcpy_h2d(d_w, _u8(w), w.nbytes)
        elem = "bf16" if p.dtype == "bf16" else "f16"
        sig = [
            {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
            {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
            {"name": "A_bytes", "type": "i32", "size_bytes": 4},
            {"name": "D_bytes", "type": "i32", "size_bytes": 4},
        ]
        KernelLauncher(hsaco=art_t.hsaco, kernel_name=art_t.kernel_name, signature=sig)(
            {
                "A": int(d_w),
                "D": int(d_ws),
                "A_bytes": w.nbytes,
                "D_bytes": L["workspace_bytes"],
            },
            config=LaunchConfig(
                grid=L["transpose_grid"], block=L["transpose_block"], fence=True
            ),
        )
        b_host = torch.empty(L["workspace_bytes"] // 2, dtype=w.dtype)
        rt.memcpy_d2h(_u8(b_host), d_ws, L["workspace_bytes"])
        rt.free(d_w)
        rt.free(d_ws)
    dx = _launch_direct(art_m, run_spec, p, tp, dy, b_host, dx)
    return _check(dx, ref)


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class Test4cDgrad(unittest.TestCase):
    def _run(self, p, block_q=None, block_groups=None, mode="prepass"):
        spec = _dgrad_spec(p, block_q, block_groups, mode)
        if spec is None:
            return False, "no 4c dgrad spec"
        return _run_4c_dgrad(p, spec)

    def test_dgrad_default_knobs(self):
        for dtype in ("fp16", "bf16"):
            for N, H, W, g in (
                (1, 7, 7, 16),
                (2, 13, 13, 32),
                (1, 15, 17, 64),
                (2, 30, 26, 32),
            ):
                p = _problem(N=N, H=H, W=W, groups=g, dtype=dtype)
                for mode in _MODES:
                    with self.subTest(dtype=dtype, p=p.short(), mode=mode):
                        ok, msg = self._run(p, mode=mode)
                        self.assertTrue(ok, msg)

    def test_dgrad_knob_grid(self):
        # W=17 is not a multiple of any block_q, so every config has a ragged
        # right edge; N=1 and odd H cover the H-edge flushes.
        p = _problem(N=1, H=9, W=17, groups=64, dtype="bf16")
        for bq in (4, 8, 16, 32):
            for bg in (16, 32, 64):
                for mode in _MODES:
                    if mode == "staged" and (bg // 16) * (bq // 4) * 64 > 1024:
                        continue  # over the 1024-thread workgroup limit
                    with self.subTest(bq=bq, bg=bg, mode=mode):
                        ok, msg = self._run(p, block_q=bq, block_groups=bg, mode=mode)
                        self.assertTrue(ok, msg)

    def test_dgrad_1x1(self):
        for dtype in ("fp16", "bf16"):
            p = _problem(N=2, H=13, W=11, groups=16, dtype=dtype, K=1)
            for mode in _MODES:
                with self.subTest(dtype=dtype, mode=mode):
                    ok, msg = self._run(p, mode=mode)
                    self.assertTrue(ok, msg)

    def test_one_binary_serves_every_shape(self):
        # One build per form on a probe problem, launched on unrelated
        # extents: N, H, W and groups are kernargs.
        probe = _problem(N=1, H=64, W=64, groups=64, dtype="bf16")
        shapes = (
            _problem(N=2, H=5, W=11, groups=16, dtype="bf16"),
            _problem(N=1, H=17, W=3, groups=48, dtype="bf16"),
            _problem(N=3, H=9, W=24, groups=32, dtype="bf16"),
            _problem(N=1, H=1, W=6, groups=16, dtype="bf16"),
        )
        for mode in _MODES:
            spec = _dgrad_spec(probe, 8, 16, mode)
            kt, km = build_direct_4c_dgrad(spec, arch=GPU_ARCH)
            arts = (_compile(kt) if kt is not None else None, _compile(km))
            for p in shapes:
                with self.subTest(mode=mode, p=p.short()):
                    ok, msg = _run_4c_dgrad(p, spec, arts=arts)
                    self.assertTrue(ok, msg)


@unittest.skipIf(GPU_ARCH != "gfx950" or not _HAS_TORCH, "needs gfx950 + torch")
class Test4cStageRowsDgrad(unittest.TestCase):
    """On device: the row-staged kernel over adversarial shapes (NaN-filled
    dX, so an unwritten element fails)."""

    def test_adversarial_shapes(self):
        # (N, H, W, groups, K, block_q, block_groups)
        cases = [
            (1, 1, 6, 16, 3, 4, 16),  # H = 1: both halo rows outside the image
            (3, 7, 1, 16, 3, 4, 16),  # W = 1: one live column per tile
            (2, 5, 5, 32, 3, 4, 16),  # partial q tile
            (1, 2, 33, 64, 3, 8, 32),  # two q waves, ragged right edge
            (2, 9, 21, 64, 3, 16, 64),  # 4 q waves x 4 channel waves
            (3, 7, 10, 32, 1, 4, 16),  # 1x1 filter
            (1, 13, 11, 128, 1, 8, 64),  # 1x1, partial row-staging pass
            (4, 28, 28, 32, 3, 4, 32),
            (1, 6, 7, 16, 3, 4, 16),  # H a multiple of the unrolled rows
            (1, 5, 7, 16, 3, 4, 16),  # H + 1 rows: one full unrolled body
        ]
        for dtype in ("fp16", "bf16"):
            for N, H, W, g, K, bq, bg in cases:
                p = _problem(N=N, H=H, W=W, groups=g, dtype=dtype, K=K)
                with self.subTest(dtype=dtype, p=p.short(), K=K, bq=bq, bg=bg):
                    spec = _dgrad_spec(p, bq, bg, "staged")
                    self.assertIsNotNone(spec)
                    ok, msg = _run_4c_dgrad(p, spec)
                    self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
