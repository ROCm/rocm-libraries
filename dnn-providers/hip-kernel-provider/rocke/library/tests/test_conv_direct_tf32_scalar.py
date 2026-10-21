# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp32 / tf32 storage for the scalar-FMA direct kernels and the direct wgrad.

Scalar-FMA kernels (``DirectConvDgradSpec`` and the depthwise family: preload,
spatial, tiled, column, dgrad, dgrad-stream) widen every element to f32 and
reduce with scalar FMA, so ``"fp32"`` and ``"tf32"`` run the same exact fp32
math: one binary, one name (the ``f32`` flag). ``DirectConvWgradSpec`` feeds an
MFMA whose K is the spatial axis; with fp32 storage it gathers its operand
fragments with ``ds_read_b32`` (no 32-bit transpose read exists) and issues them
through ``_mma_f32_logical`` -- exact fp32 atoms for ``"fp32"``, TF32 atoms
(gfx942 native XF32, gfx950 BF16x3) for ``"tf32"``.

CPU part: build + lower on gfx950 and gfx942, names, rejections, fp32 == tf32
IR for the scalar kernels, and an ISA instruction-count check of the wgrad
MFMAs (comgr cross-compile). GPU part (gfx950/gfx942 attached): correctness
against an IEEE fp32 torch reference computed on the CPU, relative to the
output peak -- 1e-5 for exact fp32 math, 1e-4 for the TF32 wgrad.

pytest forks per test on some setups, which leaves HIP without a device; run
the GPU part as ``python3 library/tests/test_conv_direct_tf32_scalar.py``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import re
import unittest
from typing import Callable, List, Tuple

from kernels.common import conv_direct_grouped as cdg
from kernels.common.conv_direct_grouped import DirectConvProblem as P

_ARCHS = ("gfx950", "gfx942")
_F32_DTYPES = ("fp32", "tf32")
_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py). The reference itself runs on the CPU.
    import torch

    torch.cuda.is_available()


def _gpu_arch() -> str:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Variant table: (label, spec factory(problem), builder, validator, is_dgrad)
# ---------------------------------------------------------------------------


def _dw(**kw) -> P:
    base = dict(N=2, H=14, W=14, groups=64, cpg=1, kpg=1)
    base.update(kw)
    return P(**base)


def _scalar_cases(dtype: str) -> List[Tuple[str, object, Callable, Callable, str]]:
    """(label, spec, builder, validator, direction) for every scalar variant."""
    out = []
    for i, p in enumerate([_dw(dtype=dtype), _dw(dtype=dtype, KH=7, KW=7, PAD=3)]):
        out.append(
            (
                f"dw{i}",
                cdg.DirectDepthwiseSpec(problem=p, name=f"t32_dw{i}"),
                cdg.build_direct_depthwise,
                cdg.is_valid_depthwise_spec,
                "fwd",
            )
        )
    out.append(
        (
            "dw_spatial",
            cdg.DirectDepthwiseSpatialSpec(
                problem=_dw(dtype=dtype, groups=16), name="t32_dwsp"
            ),
            cdg.build_direct_depthwise_spatial,
            cdg.is_valid_depthwise_spatial_spec,
            "fwd",
        )
    )
    for i, (p, kn) in enumerate(
        [
            (_dw(dtype=dtype), {}),
            (_dw(dtype=dtype, stride=2), {"unroll_rows": True}),
        ]
    ):
        out.append(
            (
                f"dw_tiled{i}",
                cdg.DirectDepthwiseTiledSpec(problem=p, name=f"t32_dwt{i}", **kn),
                cdg.build_direct_depthwise_tiled,
                cdg.is_valid_depthwise_tiled_spec,
                "fwd",
            )
        )
    for i, p in enumerate([_dw(), _dw(stride=2, H=13, W=11)]):
        out.append(
            (
                f"dw_col{i}",
                cdg.DirectDepthwiseColSpec(
                    problem=p, name=f"t32_dwc{i}", dtype=dtype, block_h=4, block_w=2
                ),
                cdg.build_direct_depthwise_col,
                cdg.is_valid_depthwise_col_spec,
                "fwd",
            )
        )
    for i, p in enumerate(
        [
            _dw(dtype=dtype),
            _dw(dtype=dtype, stride=2),
            _dw(dtype=dtype, KH=5, KW=5, PAD=4, stride=2),
        ]
    ):
        out.append(
            (
                f"dw_dgrad{i}",
                cdg.DirectDepthwiseDgradSpec(problem=p, name=f"t32_dwdg{i}"),
                cdg.build_direct_depthwise_dgrad,
                cdg.is_valid_depthwise_dgrad_spec,
                "dgrad",
            )
        )
        out.append(
            (
                f"dw_dgrad_stream{i}",
                cdg.DirectDepthwiseDgradStreamSpec(problem=p, name=f"t32_dwdgs{i}"),
                cdg.build_direct_depthwise_dgrad_streaming,
                cdg.is_valid_depthwise_dgrad_stream_spec,
                "dgrad",
            )
        )
    for i, p in enumerate(
        [
            P(N=2, H=8, W=9, groups=8, cpg=3, kpg=5, dtype=dtype),
            P(N=2, H=9, W=9, groups=8, cpg=16, kpg=8, stride=2, PAD=0, dtype=dtype),
        ]
    ):
        out.append(
            (
                f"dgrad{i}",
                cdg.DirectConvDgradSpec(problem=p, name=f"t32_dg{i}"),
                cdg.build_direct_conv_dgrad,
                cdg.is_valid_dgrad_spec,
                "dgrad",
            )
        )
    return out


# (label, problem kwargs, spec kwargs)
_WGRAD_CASES = [
    ("c16_mk32", dict(N=2, H=8, W=40, groups=2, cpg=16, kpg=16), dict(mfma_k=32)),
    ("c16_mk16", dict(N=2, H=8, W=40, groups=2, cpg=16, kpg=16), dict(mfma_k=16)),
    (
        "c32_k24_w2x2",
        dict(N=1, H=9, W=21, groups=2, cpg=32, kpg=24, PAD=0),
        dict(waves_k=2, waves_c=2, ho_per_block=3),
    ),
    (
        "k5_wq2",
        dict(N=1, H=7, W=70, groups=1, cpg=16, kpg=16, KH=5, KW=5, PAD=2),
        dict(waves_q=2),
    ),
]


def _wgrad_spec(label: str, pkw: dict, skw: dict, dtype: str):
    return cdg.DirectConvWgradSpec(
        problem=P(dtype=dtype, **pkw), name=f"t32_wg_{label}", **skw
    )


def _lower(kernel, arch: str) -> str:
    from rocke import lower_kernel_to_llvm

    return lower_kernel_to_llvm(kernel, arch=arch)


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


class TestScalarBuild(unittest.TestCase):
    def test_fp32_tf32_build_lower_same_ir(self):
        """Every scalar variant builds for fp32 and tf32 on both archs, with
        one name and byte-identical IR (one binary serves both)."""
        for arch in _ARCHS:
            fp = _scalar_cases("fp32")
            tf = _scalar_cases("tf32")
            for (label, s32, build, valid, _), (_, st, _, _, _) in zip(fp, tf):
                with self.subTest(arch=arch, variant=label):
                    ok, why = valid(s32, arch=arch)
                    self.assertTrue(ok, why)
                    ok, why = valid(st, arch=arch)
                    self.assertTrue(ok, why)
                    self.assertEqual(s32.kernel_name(), st.kernel_name())
                    self.assertIn("f32", s32.kernel_name())
                    ir32 = _lower(build(s32, arch=arch), arch)
                    irtf = _lower(build(st, arch=arch), arch)
                    self.assertEqual(ir32, irtf)
                    # fp32 storage: no 16-bit element access survives.
                    self.assertNotIn("half", ir32)
                    self.assertNotIn("bfloat", ir32)

    def test_16bit_names_unchanged(self):
        """fp16 / bf16 names carry no f32 flag and bf16 keeps its own."""
        for dtype in ("fp16", "bf16"):
            for label, spec, *_ in _scalar_cases(dtype):
                with self.subTest(dtype=dtype, variant=label):
                    name = spec.kernel_name()
                    self.assertNotRegex(name, r"_t?f32(_|$)")
                    if label.startswith("dw_col"):
                        continue  # the column kernel spells its dtype tag
                    self.assertEqual(name.endswith("_bf16"), dtype == "bf16")
        self.assertEqual(
            cdg.DirectDepthwiseDgradSpec(problem=_dw()).kernel_name(),
            "direct_depthwise_dgrad_N2H14W14_g64_c1k1_bw8_bw1wv",
        )
        self.assertEqual(
            cdg.DirectDepthwiseDgradSpec(problem=_dw(dtype="tf32")).kernel_name(),
            "direct_depthwise_dgrad_N2H14W14_g64_c1k1_bw8_bw1wv_f32",
        )

    def test_rejections(self):
        """Unknown dtypes are rejected with a reason naming the dtype."""
        for label, spec, build, valid, _ in _scalar_cases("fp8"):
            with self.subTest(variant=label):
                ok, why = valid(spec, arch="gfx950")
                self.assertFalse(ok)
                self.assertIn("fp8", why)
                with self.assertRaises(ValueError):
                    build(spec, arch="gfx950")


class TestWgradBuild(unittest.TestCase):
    def test_build_lower_both_archs(self):
        for arch in _ARCHS:
            for dtype in _F32_DTYPES:
                for label, pkw, skw in _WGRAD_CASES:
                    with self.subTest(arch=arch, dtype=dtype, case=label):
                        spec = _wgrad_spec(label, pkw, skw, dtype)
                        ok, why = cdg.is_valid_wgrad_spec(spec, arch=arch)
                        self.assertTrue(ok, why)
                        flag = "_f32" if dtype == "fp32" else "_tf32"
                        self.assertIn(flag, spec.kernel_name())
                        ir = _lower(cdg.build_direct_conv_wgrad(spec, arch=arch), arch)
                        self.assertNotIn("ds.read.tr", ir)

    def test_names_and_rejections(self):
        pkw, skw = _WGRAD_CASES[0][1], _WGRAD_CASES[0][2]
        names = {
            dt: _wgrad_spec("n", pkw, skw, dt).kernel_name()
            for dt in ("fp16", "bf16", "fp32", "tf32")
        }
        self.assertEqual(len(set(names.values())), 4)
        self.assertFalse(names["fp16"].endswith(("f32", "bf16")))
        # The 16-bit path still needs ds_read_tr16_b64 (gfx950+); mk16 so the
        # atom check (gfx942 has no 16x16x32 f16) does not answer first.
        ok, why = cdg.is_valid_wgrad_spec(
            _wgrad_spec("n", pkw, {"mfma_k": 16}, "fp16"), arch="gfx942"
        )
        self.assertFalse(ok)
        self.assertIn("ds_read_tr16_b64", why)
        ok, why = cdg.is_valid_wgrad_spec(
            _wgrad_spec("n", pkw, skw, "fp64"), arch="gfx950"
        )
        self.assertFalse(ok)
        self.assertIn("fp64", why)
        with self.assertRaises(ValueError):
            cdg.build_direct_conv_wgrad(_wgrad_spec("n", pkw, skw, "fp64"))

    def test_isa_mfma_counts(self):
        """KH*KW*HPB logical 16x16 MMAs: gfx950 tf32 = 3 bf16 MFMAs each,
        gfx942 tf32 = native xf32, fp32 = 16x16x4 f32 split."""
        import shutil
        import subprocess
        import tempfile

        from rocke import compile_kernel

        objdump = shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
        label, pkw, skw = _WGRAD_CASES[0]  # 3x3, HPB 4, mk 32 -> frag 8
        logical = 3 * 3 * 4
        expect = {
            ("gfx950", "tf32"): ("v_mfma_f32_16x16x32_bf16", 3 * logical),
            ("gfx942", "tf32"): ("v_mfma_f32_16x16x8_xf32", 4 * logical),
            ("gfx950", "fp32"): ("v_mfma_f32_16x16x4_f32", 8 * logical),
            ("gfx942", "fp32"): ("v_mfma_f32_16x16x4_f32", 8 * logical),
        }
        for (arch, dtype), (mnemonic, count) in expect.items():
            with self.subTest(arch=arch, dtype=dtype):
                spec = _wgrad_spec(label, pkw, skw, dtype)
                art = compile_kernel(
                    cdg.build_direct_conv_wgrad(spec, arch=arch), arch=arch
                )
                with tempfile.TemporaryDirectory() as d:
                    path = f"{d}/k.hsaco"
                    with open(path, "wb") as f:
                        f.write(art.hsaco)
                    isa = subprocess.run(
                        [objdump, "-d", f"--mcpu={arch}", path],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                mfmas = re.findall(r"v_mfma_\w+", isa)
                self.assertEqual(mfmas.count(mnemonic), count, set(mfmas))
                self.assertEqual(len(mfmas), count)
                self.assertNotIn("ds_read_b64_tr", isa)
                self.assertNotIn("scratch_", isa)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _launch(spec, build, arch: str, direction: str, A, B, D) -> "torch.Tensor":
    """Compile ``spec``, launch it on host tensors A/B/D, return D's result."""
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke import compile_kernel
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    art = compile_kernel(build(spec, arch=arch), arch=arch)
    rt = Runtime()
    devs = []
    for t in (A, B, D):
        d = rt.alloc(t.nbytes)
        devs.append(d)
    rt.memcpy_h2d(devs[0], _u8(A), A.nbytes)
    rt.memcpy_h2d(devs[1], _u8(B), B.nbytes)
    rt.memset(devs[2], 0, D.nbytes)
    try:
        # fp32 pointers either way: tf32 is fp32 storage.
        sig = conv_direct_args_signature("fp32", direction=direction)
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
        )
        values = ConvArgs.from_problem(spec.problem, direction=direction)
        values = values.to_launch_values(
            int(devs[0]), int(devs[1]), int(devs[2]), A.nbytes, B.nbytes, D.nbytes
        )
        grid, block = cdg.direct_launch_geometry(spec)
        launcher(values, config=LaunchConfig(grid=grid, block=block, fence=True))
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), devs[2], D.nbytes)
    finally:
        for d in devs:
            rt.free(d)
        synchronize_and_release(0)
    return out


def _rel_err(out, ref) -> float:
    return float((out - ref).abs().max() / ref.abs().max().clamp(min=1e-30))


def _ref_fwd(X, W, p):
    import torch.nn.functional as F

    y = F.conv2d(
        X.permute(0, 3, 1, 2),
        W.permute(0, 3, 1, 2),
        padding=p.PAD,
        stride=p.stride,
        groups=p.groups,
    )
    return y.permute(0, 2, 3, 1).contiguous()


def _ref_dgrad(dY, W, p):
    import torch.nn.functional as F

    hb = (p.Ho - 1) * p.stride - 2 * p.PAD + p.KH
    wb = (p.Wo - 1) * p.stride - 2 * p.PAD + p.KW
    dx = F.conv_transpose2d(
        dY.permute(0, 3, 1, 2),
        W.permute(0, 3, 1, 2),
        padding=p.PAD,
        stride=p.stride,
        groups=p.groups,
        output_padding=(p.H - hb, p.W - wb),
    )
    return dx.permute(0, 2, 3, 1).contiguous()


def _ref_wgrad(X, dY, p):
    import torch.nn.functional as F

    W = torch.zeros(p.total_k, p.cpg, p.KH, p.KW, requires_grad=True)
    y = F.conv2d(
        X.permute(0, 3, 1, 2), W, padding=p.PAD, stride=p.stride, groups=p.groups
    )
    y.backward(dY.permute(0, 3, 1, 2))
    return W.grad.permute(0, 2, 3, 1).contiguous()


_GPU = _gpu_arch()
_GPU_SKIP = (
    "torch not importable"
    if not _HAS_TORCH
    else ("" if _GPU in _ARCHS else f"no gfx950/gfx942 GPU (got {_GPU!r})")
)


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP)
class TestGpuCorrectness(unittest.TestCase):
    def setUp(self):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(0)

    def test_scalar_variants(self):
        ran = 0
        for dtype in _F32_DTYPES:
            for label, spec, build, valid, direction in _scalar_cases(dtype):
                with self.subTest(dtype=dtype, variant=label):
                    p = spec.problem
                    ok, why = valid(spec, arch=_GPU)
                    self.assertTrue(ok, why)
                    W = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1)
                    if direction == "fwd":
                        X = torch.empty(p.N, p.H, p.W, p.total_c).uniform_(-1, 1)
                        D = torch.empty(p.N, p.Ho, p.Wo, p.total_k)
                        out = _launch(spec, build, _GPU, "fwd", X, W, D)
                        ref = _ref_fwd(X, W, p)
                    else:
                        dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k).uniform_(-1, 1)
                        D = torch.empty(p.N, p.H, p.W, p.total_c)
                        out = _launch(spec, build, _GPU, "dgrad", dY, W, D)
                        ref = _ref_dgrad(dY, W, p)
                    err = _rel_err(out, ref)
                    print(f"  {dtype:5s} {label:18s} rel_err={err:.2e}", flush=True)
                    self.assertLess(err, 1e-5)
                    ran += 1
        self.assertGreater(ran, 0)

    def test_wgrad(self):
        for dtype in _F32_DTYPES:
            tol = 1e-5 if dtype == "fp32" else 1e-4
            for label, pkw, skw in _WGRAD_CASES:
                with self.subTest(dtype=dtype, case=label):
                    spec = _wgrad_spec(label, pkw, skw, dtype)
                    p = spec.problem
                    X = torch.empty(p.N, p.H, p.W, p.total_c).uniform_(-1, 1)
                    dY = torch.empty(p.N, p.Ho, p.Wo, p.total_k).uniform_(-1, 1)
                    dW = torch.empty(p.total_k, p.KH, p.KW, p.cpg)
                    out = _launch(
                        spec, cdg.build_direct_conv_wgrad, _GPU, "wgrad", dY, X, dW
                    )
                    err = _rel_err(out, _ref_wgrad(X, dY, p))
                    print(
                        f"  {dtype:5s} wgrad {label:14s} rel_err={err:.2e}", flush=True
                    )
                    self.assertLess(err, tol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
