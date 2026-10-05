# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""4c (cpg = kpg = 4) direct conv: bf16 support and the 4c dgrad entry.

Covers:

* spec plumbing (CPU only): ``make_dgrad_4c_spec`` / ``dgrad_4c_spec_for_problem``
  / ``is_valid_dgrad_4c_problem`` / ``direct_4c_dgrad_launch``, and that the
  bf16 4c kernel emits only the ``mfma_f32_4x4x4_bf16`` (``_1k``) atom while the
  fp16 kernel keeps the f16 atom;
* on-device numerics (gfx942 / gfx950 + torch): the 4c fprop kernel in bf16 and
  the 4c dgrad in its three forms -- the two-kernel pipeline (weight transpose
  + 4c kernel on dY) and the single kernel with the fused weight transform
  (``dgrad_fused_weights``: per-element gathers, or LDS staging + transpose
  reads with ``dgrad_weights_lds`` on gfx950) -- in fp16 and bf16 over
  adversarial shapes (odd H / W, N = 1, W not a multiple of block_q, 1x1
  filters, every block_q / block_groups knob).

Elementwise rule (``manifest_runner/conv.py``): an output element is bad when
``|out - ref| > tol + tol * |ref|`` with ``tol = 1e-2``; a case passes iff no
element is bad.

Run:
  PYTHONPATH=rocke/platform/python:rocke/library <torch-python> -m pytest \
      rocke/library/tests/test_conv_dgrad_4c.py -v
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

from rocke.runtime.hip_module import get_device_arch

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    build_direct_4c_dgrad,
    build_direct_conv_4c,
    dgrad_4c_spec_for_problem,
    direct_4c_dgrad_launch,
    is_valid_dgrad_4c_problem,
    is_valid_spec_4c,
    make_dgrad_4c_spec,
)

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
# CPU-only: spec plumbing and emitted atom
# ---------------------------------------------------------------------------


class TestDgrad4cSpec(unittest.TestCase):
    def test_make_dgrad_4c_spec_transposes_problem(self):
        p = _problem(N=3, H=13, W=17, groups=32)
        s = make_dgrad_4c_spec(p, block_q=8, block_groups=32)
        tp = s.problem
        self.assertEqual((tp.N, tp.H, tp.W), (3, 13, 17))
        self.assertEqual((tp.cpg, tp.kpg, tp.groups), (4, 4, 32))
        self.assertEqual((tp.KH, tp.KW, tp.PAD, tp.stride), (3, 3, 1, 1))
        self.assertEqual(tp.dtype, "bf16")
        self.assertEqual((s.block_q, s.block_groups), (8, 32))
        self.assertTrue(s.kernel_name().startswith("direct_conv_4c_dgrad_"))
        self.assertTrue(s.kernel_name().endswith("_bf16"))
        s.validate()

    def test_fp16_kernel_name_has_no_dtype_flag(self):
        p = _problem(N=1, H=8, W=8, groups=16, dtype="fp16")
        name = DirectConv4cSpec(problem=p).kernel_name()
        self.assertEqual(name, "direct_conv_4c_N1H8W8_g16_c4k4_bq4_bg16")

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

    def test_dispatch_hook(self):
        p = _problem(N=128, H=56, W=56, groups=32)
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
        # groups=16 cannot take a block_groups=32 default; hook falls back to 16.
        s16 = dgrad_4c_spec_for_problem(_problem(N=2, H=8, W=8, groups=16))
        self.assertIsNotNone(s16)
        self.assertEqual(s16.block_groups, 16)
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
        # One lane per W_T element (128 channels x 9 taps x kpg 4), 256 per block.
        self.assertEqual(L["transpose_grid"], (18, 1, 1))
        self.assertEqual(L["transpose_block"], (256, 1, 1))
        self.assertEqual(L["workspace_bytes"], 128 * 9 * 4 * 2)

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

    def test_dgrad_pipeline_builds(self):
        p = _problem(N=1, H=7, W=7, groups=16)
        s = make_dgrad_4c_spec(p)
        kt, km = build_direct_4c_dgrad(s, arch="gfx950")
        self.assertIn("transpose_weights_dgrad", kt.name)
        self.assertEqual(km.name, s.kernel_name())


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


def _launch(rt, art, sig, values, grid, block):
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
    )
    launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))


def _run_4c_fprop(
    p: DirectConvProblem, block_q: int, block_groups: int
) -> tuple[bool, str]:
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime.hip_module import Runtime

    spec = DirectConv4cSpec(problem=p, block_q=block_q, block_groups=block_groups)
    art = compile_kernel(build_direct_conv_4c(spec, arch=GPU_ARCH), arch=GPU_ARCH)
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
    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (x, w, y)]
    rt.memcpy_h2d(bufs[0], _u8(x), x.nbytes)
    rt.memcpy_h2d(bufs[1], _u8(w), w.nbytes)
    rt.memset(bufs[2], 0, y.nbytes)
    values = {
        "A": bufs[0],
        "B": bufs[1],
        "D": bufs[2],
        "A_bytes": x.nbytes,
        "B_bytes": w.nbytes,
        "D_bytes": y.nbytes,
    }
    grid = (-(-p.Wo // block_q), p.groups // block_groups, p.N)
    _launch(
        rt,
        art,
        conv_args_signature(p.dtype),
        values,
        grid,
        (spec.threads_per_block, 1, 1),
    )
    rt.memcpy_d2h(_u8(y), bufs[2], y.nbytes)
    for b in bufs:
        rt.free(b)
    return _check(y, ref)


#: dgrad forms exercised on the device: the two-kernel pre-pass pipeline and
#: the fused-weight single kernel with gathers or (gfx950) LDS staging.
_MODES = ("prepass", "gather") + (("lds",) if GPU_ARCH == "gfx950" else ())


def _run_4c_dgrad(
    p: DirectConvProblem,
    block_q: int | None = None,
    block_groups: int | None = None,
    mode: str = "prepass",
) -> tuple[bool, str]:
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime.hip_module import Runtime

    spec = dgrad_4c_spec_for_problem(
        p,
        arch=GPU_ARCH,
        block_q=block_q,
        block_groups=block_groups,
        fused_weights=mode != "prepass",
    )
    if spec is None:
        return False, "no 4c dgrad spec"
    spec = replace(spec, dgrad_weights_lds=mode == "lds")
    kt, km = build_direct_4c_dgrad(spec, arch=GPU_ARCH)
    art_t = compile_kernel(kt, arch=GPU_ARCH) if kt is not None else None
    art_m = compile_kernel(km, arch=GPU_ARCH)
    L = direct_4c_dgrad_launch(spec)

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

    rt = Runtime()
    d_dy, d_w, d_dx = (rt.alloc(t.nbytes) for t in (dy, w, dx))
    rt.memcpy_h2d(d_dy, _u8(dy), dy.nbytes)
    rt.memcpy_h2d(d_w, _u8(w), w.nbytes)
    # NaN-fill dX so an unwritten output element can never pass.
    rt.memset(d_dx, 0xFF, dx.nbytes)
    if art_t is not None:
        d_ws = rt.alloc(L["workspace_bytes"])
        wsig = [
            {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
            {"name": "A_bytes", "type": "i32", "size_bytes": 4},
            {"name": "D_bytes", "type": "i32", "size_bytes": 4},
        ]
        _launch(
            rt,
            art_t,
            wsig,
            {
                "A": d_w,
                "D": d_ws,
                "A_bytes": w.nbytes,
                "D_bytes": L["workspace_bytes"],
            },
            L["transpose_grid"],
            L["transpose_block"],
        )
        b_buf, b_bytes, extra = d_ws, L["workspace_bytes"], [d_ws]
    else:
        b_buf, b_bytes, extra = d_w, w.nbytes, []
    _launch(
        rt,
        art_m,
        conv_args_signature(p.dtype),
        {
            "A": d_dy,
            "B": b_buf,
            "D": d_dx,
            "A_bytes": dy.nbytes,
            "B_bytes": b_bytes,
            "D_bytes": dx.nbytes,
        },
        L["grid"],
        L["block"],
    )
    rt.memcpy_d2h(_u8(dx), d_dx, dx.nbytes)
    for b in (d_dy, d_w, d_dx, *extra):
        rt.free(b)
    return _check(dx, ref)


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
                ok, msg = _run_4c_fprop(p, bq, bg)
                self.assertTrue(ok, msg)


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class Test4cDgrad(unittest.TestCase):
    def test_dgrad_default_knobs(self):
        for dtype in ("fp16", "bf16"):
            for N, H, W, g in (
                (1, 7, 7, 16),
                (2, 13, 13, 32),
                (1, 15, 17, 64),
                (2, 56, 56, 32),
            ):
                p = _problem(N=N, H=H, W=W, groups=g, dtype=dtype)
                for mode in _MODES:
                    with self.subTest(dtype=dtype, p=p.short(), mode=mode):
                        ok, msg = _run_4c_dgrad(p, mode=mode)
                        self.assertTrue(ok, msg)

    def test_dgrad_knob_grid(self):
        # W=17 is not a multiple of any block_q, so every config has a ragged
        # right edge; N=1 and odd H cover the H-edge flushes.
        p = _problem(N=1, H=9, W=17, groups=64, dtype="bf16")
        for bq in (4, 8, 16, 32):
            for bg in (16, 32, 64):
                for mode in _MODES:
                    with self.subTest(bq=bq, bg=bg, mode=mode):
                        ok, msg = _run_4c_dgrad(
                            p, block_q=bq, block_groups=bg, mode=mode
                        )
                        self.assertTrue(ok, msg)

    def test_dgrad_1x1(self):
        for dtype in ("fp16", "bf16"):
            p = _problem(N=2, H=13, W=11, groups=16, dtype=dtype, K=1)
            for mode in _MODES:
                with self.subTest(dtype=dtype, mode=mode):
                    ok, msg = _run_4c_dgrad(p, mode=mode)
                    self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
