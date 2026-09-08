# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric correctness tests for the conv backward-data (dgrad) implicit-GEMM kernel.

Builds one dgrad kernel per test case on the running GPU and compares the output
against a float32 torch reference (``torch.nn.grad.conv2d_input``).  Covers:

  - stride=1 (direct-store epilogue, no atomics)
  - stride=2 (tilde-decomposition, atomic epilogue)
  - split_k > 1 (atomic epilogue)
  - bf16 and fp32 data types
  - gfx1151 / gfx1201 via WMMA candidates
  - gfx1250 via WMMA wavelet pipeline (stride=1 mem + stride>1 / split_k wavelet)

Requires a ROCm GPU and torch (skip otherwise).

Run:
  PYTHONPATH=rocke/platform/python <torch-python> \
    rocke/platform/tests/instances/test_conv_dgrad_correctness.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import unittest

from rocke.runtime.hip_module import get_device_arch

_PYDIR = os.path.join(os.path.dirname(__file__), "..", "..", "python")

ARCH = get_device_arch(0)
_HAS_TORCH = importlib.util.find_spec("torch") is not None

_MFMA_ARCHES = ("gfx90a", "gfx942", "gfx950")
_WMMA_ARCHES = ("gfx1151", "gfx1201")
_WMMA_WAVELET_ARCHES = ("gfx1250",)
_SUPPORTED_ARCHES = _MFMA_ARCHES + _WMMA_ARCHES + _WMMA_WAVELET_ARCHES

_SKIP_REASON = (
    f"needs a supported ROCm GPU ({', '.join(_SUPPORTED_ARCHES)}) + torch; "
    f"detected arch={ARCH!r}, torch={'ok' if _HAS_TORCH else 'missing'}"
)


def _run_benchmark(*extra_args, timeout=600):
    """Run benchmark_implicit_gemm_conv in a subprocess and return (rc, output)."""
    import io

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _PYDIR}
    cmd = [
        sys.executable,
        "-m",
        "rocke.benchmark.benchmark_implicit_gemm_conv",
        "--arch",
        ARCH,
        "--direction",
        "dgrad",
        "--verify",
        "--sample",
        "0.05",
        "--warmup",
        "1",
        "--iters",
        "1",
        *extra_args,
    ]
    # Stream output to the terminal in real time and also collect it for
    # assertions.  Using Popen + readline avoids the buffering that hides
    # progress when capture_output=True is used with subprocess.run.
    buf = io.StringIO()
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            buf.write(line)
        proc.wait(timeout=timeout)
    return proc.returncode, buf.getvalue()


@unittest.skipUnless(ARCH in _SUPPORTED_ARCHES and _HAS_TORCH, _SKIP_REASON)
class TestConvDgradCorrectness(unittest.TestCase):
    """Build and verify dgrad kernels numerically on the running GPU."""

    def _verify(self, *extra_args, label="", timeout=600):
        rc, out = _run_benchmark(*extra_args, timeout=timeout)
        self.assertEqual(
            rc,
            0,
            f"dgrad benchmark failed{' (' + label + ')' if label else ''} "
            f"on {ARCH}:\n{out[-3000:]}",
        )
        self.assertNotIn(
            "FAIL",
            out,
            f"dgrad numeric FAIL{' (' + label + ')' if label else ''} "
            f"on {ARCH}:\n{out[-3000:]}",
        )

    # ---- stride=1 (direct store, no atomics) ---------------------------------

    def test_fp16_stride1(self):
        """fp16 dgrad, stride=1 — single sub-GEMM, direct-store epilogue."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "4",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="fp16 stride=1",
        )

    def test_bf16_stride1(self):
        """bf16 dgrad, stride=1."""
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "4",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="bf16 stride=1",
        )

    def test_fp32_stride1(self):
        """fp32 dgrad, stride=1."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"fp32 dgrad candidates require MFMA; running on {ARCH}")
        self._verify(
            "--dtype",
            "fp32",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "1",
            label="fp32 stride=1",
        )

    # ---- stride=2 (tilde decomposition, atomic epilogue) ---------------------

    def test_fp16_stride2(self):
        """fp16 dgrad, stride=2 — tilde decomposition with atomic epilogue."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--sH",
            "2",
            "--sW",
            "2",
            "--split-k",
            "1",
            label="fp16 stride=2",
        )

    def test_bf16_stride2(self):
        """bf16 dgrad, stride=2."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--sH",
            "2",
            "--sW",
            "2",
            "--split-k",
            "1",
            label="bf16 stride=2",
        )

    # ---- split_k > 1 (atomic epilogue) ---------------------------------------

    def test_fp16_split_k(self):
        """fp16 dgrad, split_k auto-selected — exercises atomic reduction path."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"split_k dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "4",
            "--Hi",
            "28",
            "--Wi",
            "28",
            "--C",
            "64",
            "--K",
            "128",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "-1",
            label="fp16 split_k=auto",
        )

    # ---- larger realistic shape ----------------------------------------------

    def test_fp16_resnet_shape(self):
        """fp16 dgrad, ResNet-style shape N8 H56 W56 C64 K64 R3 S3."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "8",
            "--Hi",
            "56",
            "--Wi",
            "56",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--split-k",
            "-1",
            label="fp16 resnet N8H56W56C64K64",
        )

    # ---- grouped (grid-per-group on blockIdx.y) ------------------------------

    def test_fp16_grouped_stride1(self):
        """fp16 grouped dgrad, groups=4 (cpg=kpg=16), stride=1 direct store."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--groups",
            "4",
            "--split-k",
            "1",
            label="fp16 grouped g4 stride=1",
        )

    def test_bf16_grouped_stride1(self):
        """bf16 grouped dgrad, groups=4, stride=1."""
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--groups",
            "4",
            "--split-k",
            "1",
            label="bf16 grouped g4 stride=1",
        )

    def test_fp16_grouped_stride2(self):
        """fp16 grouped dgrad, groups=4, stride=2 — tilde decomposition path."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"stride>1 dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--sH",
            "2",
            "--sW",
            "2",
            "--groups",
            "4",
            "--split-k",
            "1",
            label="fp16 grouped g4 stride=2",
        )

    def test_fp16_grouped_odd_kpg(self):
        """Non-power-of-two kpg (C=K=48, groups=8 -> cpg=kpg=6): guards against
        the k_sub decode-divisor trap (must divide by kpg, not total K)."""
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "16",
            "--Wi",
            "16",
            "--C",
            "48",
            "--K",
            "48",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--groups",
            "8",
            "--split-k",
            "1",
            label="fp16 grouped g8 cpg=kpg=6",
        )

    def test_fp16_grouped_split_k(self):
        """fp16 grouped dgrad with split_k>1 — group on y, split_k on z compose;
        even cpg (=16) keeps the packed <2 x f16> atomic pairs in-group."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"split_k dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "4",
            "--Hi",
            "28",
            "--Wi",
            "28",
            "--C",
            "64",
            "--K",
            "128",
            "--Y",
            "3",
            "--X",
            "3",
            "--pH",
            "1",
            "--pW",
            "1",
            "--groups",
            "4",
            "--split-k",
            "-1",
            label="fp16 grouped g4 split_k=auto",
        )


def _count_vector_buffer_loads(ll: str) -> int:
    """Number of vector-typed raw buffer loads in the lowered IR (dY free axis)."""
    return len(re.findall(r"amdgcn\.raw\.(?:ptr\.)?buffer\.load\.v\d+\w+", ll))


class TestConvDgradGfx1250Emit(unittest.TestCase):
    """gfx1250 (wave32 WMMA 16x16x32) grouped dgrad -- CPU-only emit check.

    Builds the kernel and lowers it with the *Python* engine (no GPU / comgr),
    so it runs in every CI lane including GPU-less ones.  A ROCKE_BACKEND=both
    dual-engine assertion is NOT available for dgrad: its weight (B) load is
    always scalar and emits the generic ``tile.buffer_load`` op, which the C++
    ``lower_serialized_ir`` does not implement (a pre-existing gap, independent
    of grouping -- it affects groups=1 dgrad too).  Numeric correctness of
    grouped dgrad is validated on gfx942/gfx950 above; this guards that the
    gfx1250 16x16x32 WMMA path builds and vectorises the dY loads.
    """

    def _lower_gfx1250_kouter(self, dtype: str) -> str:
        """Lower a K-outer (transpose-read) gfx1250 dgrad kernel, CPU-only."""
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
        from rocke.instances.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from rocke.instances.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
            is_valid_dgrad_spec,
        )

        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=1)
        spec = DgradConvSpec(
            problem=p,
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=32,
            pipeline="mem",
            epilogue="default",
            lds_k_outer=True,
        )
        ok, why = is_valid_dgrad_spec(spec, "gfx1250")
        self.assertTrue(ok, f"gfx1250 K-outer dgrad spec unexpectedly invalid: {why}")
        kernel = build_implicit_gemm_conv_dgrad(spec, arch="gfx1250")
        return _lower_kernel_to_llvm_python(kernel, arch="gfx1250")

    def test_gfx1250_kouter_rejects_wavelet_pipeline(self):
        """wavelet + lds_k_outer must be rejected, not silently miscompiled.

        build_wavelet_loaders pins the B tile to (block_n, block_k) and takes
        the unswapped descriptor, so it writes the tile M-outer while the
        compute phase reads it through _tr_frag. The allocation is K-outer, so
        the row stride is wrong for every element and the store runs past
        B_smem whenever tile_n > tile_k. Confirmed numerically wrong on
        gfx1250 before the gate went in, and the K-outer A/B pins
        pipeline="mem", so nothing else covers this pair.
        """
        import dataclasses

        from rocke.instances.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from rocke.instances.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=1)
        base = DgradConvSpec(
            problem=p,
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=1,
            warp_n=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=32,
            pipeline="mem",
            epilogue="cshuffle",
            lds_k_outer=True,
        )
        # mem + K-outer is the supported pair and must stay valid.
        ok, why = is_valid_dgrad_spec(base, "gfx1250")
        self.assertTrue(ok, f"mem + lds_k_outer should be valid: {why}")

        wavelet = dataclasses.replace(base, pipeline="wavelet")
        ok, why = is_valid_dgrad_spec(wavelet, "gfx1250")
        self.assertFalse(ok, "wavelet + lds_k_outer must be rejected")
        self.assertIn("wavelet", why)
        with self.assertRaises(ValueError):
            wavelet.validate()

        # And the selection policy must never hand out the broken pair.
        common = dict(
            arch="gfx1250", dtype_b="bf16", warp_tile_n=16, cpg=64, wave_size=32
        )
        self.assertTrue(DgradConvSpec.default_lds_k_outer(pipeline="mem", **common))
        self.assertFalse(
            DgradConvSpec.default_lds_k_outer(pipeline="wavelet", **common),
            "default_lds_k_outer must fall back to M-outer under wavelet",
        )

    def test_gfx1250_kouter_emits_ds_load_tr16_b128(self):
        """The K-outer B fetch must lower to ds_load_tr16_b128 of <8 x T>.

        Guards the wave32 transpose-read on GPU-less CI. The 16x16x32 atom has
        b_frag_len == 16 and the intrinsic returns 8 per lane, so the fragment is
        exactly **two** reads -- a count of 1 would mean half the K range is
        never fetched, and 4 would mean someone assumed the 4-element
        ds_read_tr16_b64 shape.
        """
        for dtype, vec in (("bf16", "<8 x bfloat>"), ("fp16", "<8 x half>")):
            with self.subTest(dtype=dtype):
                ll = self._lower_gfx1250_kouter(dtype)
                suffix = "v8bf16" if dtype == "bf16" else "v8f16"
                intrinsic = f"llvm.amdgcn.ds.load.tr16.b128.{suffix}"
                self.assertIn(
                    intrinsic,
                    ll,
                    f"expected the gfx1250 wide transpose-LDS read for {dtype}",
                )
                calls = re.findall(rf"call\s+[^\n]*@{re.escape(intrinsic)}\(", ll)
                self.assertEqual(
                    len(calls),
                    2,
                    f"expected 2 {intrinsic} calls (b_frag_len 16 / 8 per read), "
                    f"got {len(calls)}",
                )
                # Operand shape: the transpose read must be typed <8 x T> and
                # take an LDS (addrspace 3) pointer, not a generic one.
                self.assertRegex(
                    ll,
                    rf"call\s+{re.escape(vec)}\s+@{re.escape(intrinsic)}"
                    rf"\(ptr addrspace\(3\)",
                    f"expected {vec} from an addrspace(3) pointer for {dtype}",
                )
                # The K-outer path must not fall back to the wave64 b64 form.
                self.assertNotIn("llvm.amdgcn.ds.read.tr16.b64", ll)

    def _lower_gfx1250(self, groups: int) -> str:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
        from rocke.instances.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from rocke.instances.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
            is_valid_dgrad_spec,
        )

        p = ConvProblem(
            N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=groups
        )
        spec = DgradConvSpec(
            problem=p,
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=32,
            tile_n=32,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=32,
            pipeline="mem",
            epilogue="default",
        )
        ok, why = is_valid_dgrad_spec(spec, "gfx1250")
        self.assertTrue(ok, f"gfx1250 dgrad spec unexpectedly invalid: {why}")
        kernel = build_implicit_gemm_conv_dgrad(spec, arch="gfx1250")
        return _lower_kernel_to_llvm_python(kernel, arch="gfx1250")

    def test_gfx1250_grouped_dgrad_emits_wmma_16x16x32(self):
        # Grouped dgrad (grid-per-group, group on block_id_y) on gfx1250:
        # C=K=64, groups=4 -> cpg=kpg=16.
        ll = self._lower_gfx1250(groups=4)
        self.assertIn(
            "wmma.f32.16x16x32",
            ll,
            "expected the gfx1250 16x16x32 WMMA intrinsic in the grouped lowered IR",
        )
        self.assertGreater(
            _count_vector_buffer_loads(ll),
            0,
            "expected vectorised dY loads for gfx1250 grouped dgrad, got scalar only",
        )

    def test_gfx1250_ungrouped_dgrad_emits_wmma_16x16x32(self):
        # groups=1 must also build on the relaxed 16x16x32 WMMA atom gate.
        ll = self._lower_gfx1250(groups=1)
        self.assertIn("wmma.f32.16x16x32", ll)


# ---------------------------------------------------------------------------
# K-outer LDS (transpose-read) A/B
# ---------------------------------------------------------------------------
#
# Every other test in this file shells out to the benchmark driver and greps
# stdout, which cannot A/B one spec flag. These helpers build and launch a
# dgrad kernel in-process so the K-outer B tile can be compared against the
# M-outer default for the identical problem and tiling.


def _dgrad_run_inprocess(spec, dtype, seed=0):
    """Launch one dgrad kernel and return dX as a torch tensor (NHWC)."""
    import ctypes

    import torch

    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.instances.common.conv_implicit_gemm_dgrad import (
        build_implicit_gemm_conv_dgrad,
        pack_sub_gemm_buffer,
    )
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    def _u8(t):
        return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())

    artifact = compile_kernel(
        build_implicit_gemm_conv_dgrad(spec, arch=ARCH), arch=ARCH
    )
    td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    p = spec.problem
    torch.manual_seed(seed)
    dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).to(td)
    W = torch.empty(p.K, p.Y, p.X, p.cpg).uniform_(-1.0, 1.0).to(td)
    dX = torch.zeros(p.N, p.Hi, p.Wi, p.C, dtype=td)

    rt = Runtime()
    dY_d, W_d, dX_d = rt.alloc(dY.nbytes), rt.alloc(W.nbytes), rt.alloc(dX.nbytes)
    rt.memcpy_h2d(dY_d, _u8(dY), dY.nbytes)
    rt.memcpy_h2d(W_d, _u8(W), W.nbytes)
    rt.memset(dX_d, 0, dX.nbytes)  # split-K atomic-add needs a zeroed dX

    sub_gemms = spec.compute_sub_gemms()
    buf = pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n)
    raw = (ctypes.c_int32 * len(buf))(*buf)
    sg_d = rt.alloc(ctypes.sizeof(raw))
    rt.memcpy_h2d(sg_d, raw, ctypes.sizeof(raw))

    sig = conv_args_signature(dtype) + [
        {"name": "sub_gemm_buf", "type": "ptr<i32, global>", "size_bytes": 8},
        {"name": "num_sub_gemms", "type": "i32", "size_bytes": 4},
    ]
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
    )
    launcher(
        {
            "A": dY_d,
            "B": W_d,
            "D": dX_d,
            "A_bytes": dY.nbytes,
            "B_bytes": W.nbytes,
            "D_bytes": dX.nbytes,
            "sub_gemm_buf": sg_d,
            "num_sub_gemms": len(sub_gemms),
        },
        config=LaunchConfig(
            grid=(sub_gemms[-1].block_end, p.groups, spec.split_k),
            block=(spec.launch_block_size, 1, 1),
            fence=True,
        ),
    )
    out = torch.empty_like(dX)
    rt.memcpy_d2h(_u8(out), dX_d, dX.nbytes)
    for d in (dY_d, W_d, dX_d, sg_d):
        rt.free(d)
    return out


# The K-outer B tile runs in two lane-mapping regimes, and the A/B has to cover
# both: gfx950 is wave64 MFMA reading through ``ds_read_b64_tr_b16`` (4 elements
# per lane), gfx1250 is wave32 WMMA reading through ``ds_load_tr16_b128`` (8 per
# lane). ``_tr_frag`` in conv_implicit_gemm_dgrad.py branches on wave_size, so a
# green run on one arch says nothing about the other.
_KOUTER_WAVE = {"gfx950": 64, "gfx1250": 32}

# gfx1250 exposes exactly one usable atom here -- 16x16x32 -- so the gfx950
# tilings below have no one-to-one counterpart. Remap the cases that have an
# equivalent and skip the rest explicitly; silently running a *different* shape
# would turn "this atom is untested on wave32" into a false green.
_KOUTER_WAVE32_ATOM = {(32, 64): (16, 32)}  # (warp_tile_mn, tile_k) gfx950 -> wave32


@unittest.skipUnless(
    ARCH in _KOUTER_WAVE and _HAS_TORCH,
    f"K-outer dgrad needs {'/'.join(_KOUTER_WAVE)} + torch",
)
class TestConvDgradLdsKOuter(unittest.TestCase):
    """The K-outer B tile must be a pure re-layout of the M-outer default."""

    def _pair(
        self, dtype, *, warp_tile_mn, tile_k, epilogue, split_k, stride=1, Hi=14, K=64
    ):
        sys.path.insert(0, os.path.abspath(_PYDIR))
        from rocke.instances.common.conv_implicit_gemm import ConvDataSpec
        from rocke.instances.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )
        from rocke.core.arch import ArchTarget

        from rocke.benchmark.benchmark_implicit_gemm_conv import parse_miopen_cmd

        kw = "convbfp16" if dtype == "bf16" else "convfp16"
        problem, _dt, _f = parse_miopen_cmd(
            f"./MIOpenDriver {kw} -n 2 -c 64 -H {Hi} -W {Hi} -k {K} -y 3 -x 3 "
            f"-p 1 -q 1 -u {stride} -v {stride} -l 1 -j 1 -m conv -g 1 -F 2 -t 1"
        )
        wave = _KOUTER_WAVE[ARCH]
        if wave == 32:
            remapped = _KOUTER_WAVE32_ATOM.get((warp_tile_mn, tile_k))
            if remapped is None:
                self.skipTest(
                    f"no wave32 counterpart for the {warp_tile_mn}x{warp_tile_mn} "
                    f"k_max={tile_k} atom; {ARCH} has only 16x16x32"
                )
            warp_tile_mn, tile_k = remapped
        family = "wmma" if wave == 32 else "mma"

        tgt = ArchTarget.from_gfx(ARCH)
        atom = tgt.mma.select_largest_k(
            family=family,
            a_dtype=dtype,
            b_dtype=dtype,
            c_dtype="fp32",
            m=warp_tile_mn,
            n=warp_tile_mn,
            k_max=tile_k,
        )
        if atom is None:
            self.skipTest(f"no {family} atom for {warp_tile_mn} k_max={tile_k}")
        out = []
        for kouter in (False, True):
            spec = DgradConvSpec(
                problem=problem,
                name="rocke_test_dgrad",
                data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
                tile_m=2 * warp_tile_mn,
                tile_n=2 * warp_tile_mn,
                tile_k=tile_k,
                warp_m=1,
                warp_n=1,
                warp_tile_m=warp_tile_mn,
                warp_tile_n=warp_tile_mn,
                warp_tile_k=atom.k,
                wave_size=wave,
                pipeline="mem",
                epilogue=epilogue,
                split_k=split_k,
                lds_k_outer=kouter,
            )
            ok, reason = is_valid_dgrad_spec(spec, ARCH)
            if not ok:
                self.skipTest(f"invalid spec (lds_k_outer={kouter}): {reason}")
            spec.validate()
            out.append(_dgrad_run_inprocess(spec, dtype))
        return out

    def _assert_exact(self, dtype, **kw):
        import torch

        ref, got = self._pair(dtype, **kw)
        self.assertTrue(
            torch.equal(ref, got),
            "K-outer is a pure re-layout, so dX must match the M-outer path "
            "bit for bit; a difference means the transpose-read lane mapping "
            "is wrong",
        )

    def test_kouter_matches_default_bf16(self):
        self._assert_exact(
            "bf16", warp_tile_mn=32, tile_k=64, epilogue="cshuffle", split_k=1
        )

    def test_kouter_matches_default_fp16(self):
        self._assert_exact(
            "fp16", warp_tile_mn=32, tile_k=64, epilogue="cshuffle", split_k=1
        )

    def test_kouter_atom_16x16x16(self):
        # b_frag_len is 4 here, not 8: one ds_read_tr16_b64 per fragment. Pins
        # the per-atom fragment length -- hardcoding 8 reads past the tile end.
        self._assert_exact(
            "bf16", warp_tile_mn=16, tile_k=16, epilogue="cshuffle", split_k=1
        )

    def test_kouter_strided_tilde(self):
        # stride=2 exercises the tilde sub-GEMM decomposition.
        self._assert_exact(
            "bf16", warp_tile_mn=32, tile_k=64, epilogue="cshuffle", split_k=1, stride=2
        )

    def test_kouter_k_not_tile_aligned(self):
        # gemm_k not a multiple of tile_k: the last tile runs past real K and
        # relies on the buffer OOB clamp. Under K-outer the zero-fill becomes
        # zero rows rather than zero columns.
        self._assert_exact(
            "bf16",
            warp_tile_mn=32,
            tile_k=64,
            epilogue="cshuffle",
            split_k=1,
            Hi=13,
            K=48,
        )

    def test_kouter_split_k_matches_reference(self):
        # split_k > 1 uses the atomic epilogue, whose accumulation order is not
        # deterministic -- the M-outer kernel does not even reproduce itself
        # bitwise. Compare against torch within tolerance instead.
        import torch

        ref, got = self._pair(
            "bf16", warp_tile_mn=32, tile_k=64, epilogue="default", split_k=4
        )
        for out in (ref, got):
            self.assertFalse(torch.isnan(out.float()).any(), "dX contains NaN")
        delta = (ref.float() - got.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        self.assertLess(delta / max(scale, 1e-6), 5e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
