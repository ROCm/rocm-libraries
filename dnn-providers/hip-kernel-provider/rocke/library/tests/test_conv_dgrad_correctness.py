# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric correctness tests for the conv backward-data (dgrad) implicit-GEMM kernel.

Builds one dgrad kernel per test case on the running GPU and compares the output
against a float32 torch reference (``torch.nn.grad.conv2d_input``).  Covers:

  - stride=1 (direct-store epilogue, no atomics)
  - stride=2 (tilde-decomposition, atomic epilogue)
  - split_k > 1 (atomic epilogue)
  - pointwise 1x1/stride1/pad0 (flat-offset fast path, no tilde decomposition)
  - bf16 and fp32 data types
  - gfx1151 / gfx1201 via WMMA candidates
  - gfx1250 via WMMA wavelet pipeline (stride=1 mem + stride>1 / split_k wavelet)

Requires a ROCm GPU and torch (skip otherwise).

Run:
  PYTHONPATH=rocke/platform/python:rocke/library <torch-python> \
    rocke/library/tests/test_conv_dgrad_correctness.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import unittest

from rocke.assets import library_root, platform_root
from rocke.runtime.hip_module import get_device_arch

_PYDIR = str(platform_root() / "python")
_LIB_DIR = str(library_root())

# The assets roots describe the source checkout. In an installed test artifact
# the packages sit under tests/ and tests/library instead, so lead with this
# process's own sys.path -- whatever let pytest import rocke and kernels here
# is by definition enough for the child -- and keep the derived roots as the
# source-tree fallback.
_CHILD_PYTHONPATH = os.pathsep.join(
    dict.fromkeys([p for p in sys.path if p] + [_PYDIR, _LIB_DIR])
)

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

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": _CHILD_PYTHONPATH,
    }
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.common.benchmark_implicit_gemm_conv",
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
        "--jobs",
        "0",
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

    # ---- pointwise 1x1 (flat-offset fast path) -------------------------------

    def test_fp16_pointwise_1x1(self):
        """fp16 dgrad, 1x1/stride1/pad0 ungrouped — the flat-offset fast path.

        dy_descriptor/w_descriptor take a separate branch when
        ``p.is_pointwise and not grouped``: the tilde decomposition is skipped
        and the offsets collapse to ``m_sub*K + k_sub`` and ``k_sub*C + c_val``.
        The emitter parity config only byte-compares the Python and C++
        emitters, so a shared wrong offset would pass there; this is the
        reference-based check, and the offset arithmetic is what it pins down.

        N*Ho*Wo == 98 and gemm_k == K == 48 are each short of the widest
        candidate tile, so the last M and K tiles are partial and the tail lanes
        do take the branch.  What that deliberately does NOT buy is coverage of
        the branch's validity predicate: on a pointwise shape the offset is a
        flat index, so an out-of-range m_sub lands past the end of dY and an
        out-of-range k_sub lands past the end of W, and the descriptor's OOB
        clamp (see the dg_K_padded note in conv_implicit_gemm_dgrad) has already
        zeroed those lanes.  Deleting either clause leaves this test green.  Do
        not widen the dims hoping to change that — the redundancy is structural,
        not a property of these particular numbers.
        """
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "7",
            "--Wi",
            "7",
            "--C",
            "64",
            "--K",
            "48",
            "--Y",
            "1",
            "--X",
            "1",
            "--pH",
            "0",
            "--pW",
            "0",
            "--split-k",
            "1",
            label="fp16 pointwise 1x1",
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

    def test_bf16_patch_embedding_stride16(self):
        """bf16 dgrad of a 16x16 / stride-16 patch embedding.

        Its tilde decomposition has 16*16 = 256 sub-GEMMs, more than the 64 a
        kernel used to dispatch over: the CTA's search over the records runs
        as many steps as their count needs.
        """
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
            "32",
            "--Wi",
            "32",
            "--C",
            "3",
            "--K",
            "32",
            "--Y",
            "16",
            "--X",
            "16",
            "--pH",
            "0",
            "--pW",
            "0",
            "--sH",
            "16",
            "--sW",
            "16",
            "--split-k",
            "1",
            label="bf16 patch 16x16 stride=16",
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

    def test_bf16_depthwise_large_filter(self):
        """Depthwise (cpg=kpg=1) dgrad: a one-column GEMM per group, every
        operand scalar. A tall, asymmetric filter with per-axis "same"
        padding pins the separate Y/X and pH/pW decode."""
        self._verify(
            "--dtype",
            "bf16",
            "--N",
            "2",
            "--Hi",
            "14",
            "--Wi",
            "14",
            "--C",
            "32",
            "--K",
            "32",
            "--Y",
            "11",
            "--X",
            "3",
            "--pH",
            "5",
            "--pW",
            "1",
            "--groups",
            "32",
            "--split-k",
            "1",
            label="bf16 depthwise g32 11x3",
        )

    def test_fp16_depthwise_stride2(self):
        """Depthwise dgrad on the stride-2 tilde decomposition path."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"stride>1 dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "--dtype",
            "fp16",
            "--N",
            "2",
            "--Hi",
            "15",
            "--Wi",
            "13",
            "--C",
            "64",
            "--K",
            "64",
            "--Y",
            "5",
            "--X",
            "3",
            "--pH",
            "2",
            "--pW",
            "0",
            "--sH",
            "2",
            "--sW",
            "2",
            "--groups",
            "64",
            "--split-k",
            "1",
            label="fp16 depthwise g64 stride=2",
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
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
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

        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
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
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
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


def _dgrad_run_inprocess(spec, dtype, seed=0, poison=False, problem=None):
    """Launch one dgrad kernel and return dX as a torch tensor (NHWC).

    The kernel is built from ``spec`` (its problem is the build probe) and
    launched on ``problem`` (default: the spec's own), as an AOT binary is:
    the shape travels as kernargs and the sub-GEMM records are packed for the
    launched problem. ``poison=True`` pre-fills dX with 0xFF bytes (NaN in
    fp16/bf16) when the kernel stores rather than accumulates, so an output
    the kernel never writes shows up as NaN instead of passing as zero.
    """
    import ctypes
    from dataclasses import replace as dc_replace

    import torch

    from rocke import compile_kernel
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_implicit_gemm_dgrad import (
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
    p = spec.problem if problem is None else problem
    torch.manual_seed(seed)
    dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).to(td)
    W = torch.empty(p.K, p.Y, p.X, p.cpg).uniform_(-1.0, 1.0).to(td)
    dX = torch.zeros(p.N, p.Hi, p.Wi, p.C, dtype=td)

    rt = Runtime()
    dY_d, W_d, dX_d = rt.alloc(dY.nbytes), rt.alloc(W.nbytes), rt.alloc(dX.nbytes)
    rt.memcpy_h2d(dY_d, _u8(dY), dY.nbytes)
    rt.memcpy_h2d(W_d, _u8(W), W.nbytes)
    # split-K atomic-add needs a zeroed dX.
    rt.memset(dX_d, 0xFF if (poison and not spec.needs_atomic) else 0, dX.nbytes)

    sub_gemms = dc_replace(spec, problem=p).compute_sub_gemms()
    buf = pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n)
    raw = (ctypes.c_int32 * len(buf))(*buf)
    sg_d = rt.alloc(ctypes.sizeof(raw))
    rt.memcpy_h2d(sg_d, raw, ctypes.sizeof(raw))

    # The tilde record buffer is part of the dgrad ABI, not an extension.
    sig = conv_args_signature(dtype, direction="dgrad")
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=sig
    )
    launcher(
        ConvArgs.from_problem(
            p, direction="dgrad", tile_m=spec.tile_m, tile_n=spec.tile_n
        ).to_launch_values(
            int(dY_d),
            int(W_d),
            int(dX_d),
            dY.nbytes,
            W.nbytes,
            dX.nbytes,
            sub_gemm_buf=int(sg_d),
            num_sub_gemms=len(sub_gemms),
        ),
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


def _dgrad_reference(problem, dtype, seed=0):
    """fp32 torch reference dX (NHWC) for the inputs _dgrad_run_inprocess draws."""
    import torch

    td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    p = problem
    torch.manual_seed(seed)
    # Same draw order as _dgrad_run_inprocess, so the inputs match.
    dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).to(td)
    W = torch.empty(p.K, p.Y, p.X, p.cpg).uniform_(-1.0, 1.0).to(td)
    return torch.nn.grad.conv2d_input(
        (p.N, p.C, p.Hi, p.Wi),
        W.float().permute(0, 3, 1, 2).contiguous(),
        dY.float().permute(0, 3, 1, 2).contiguous(),
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    ).permute(0, 2, 3, 1)


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
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )
        from rocke.core.arch import ArchTarget

        from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

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


@unittest.skipUnless(
    ARCH in _MFMA_ARCHES and _HAS_TORCH, f"needs one of {_MFMA_ARCHES} + torch"
)
class TestConvDgradUniformKDecode(unittest.TestCase):
    """``uniform_k_decode`` must compute exactly what the plain K loop computes.

    Both loops visit the same k values in the same order and feed the same
    MFMA sequence, so with split_k == 1 dX must match bit for bit. The cases
    cover the per-tile decode loop (kpg a multiple of tile_k) -- including the
    k_out -> xdot -> ydot wraps of a multi-tap filter, the tilde sub-GEMMs,
    dilation, grouping and both B layouts -- and the fallback loop the same
    binary runs when kpg is not a multiple of tile_k.
    """

    def _run(
        self,
        dtype,
        *,
        ukd,
        tile_k=64,
        groups=1,
        K=64,
        C=64,
        Hi=14,
        Y=3,
        stride=1,
        dil=1,
        pad=1,
        kouter=True,
        epilogue="cshuffle",
        split_k=1,
    ):
        sys.path.insert(0, os.path.abspath(_PYDIR))
        from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        kw = "convbfp16" if dtype == "bf16" else "convfp16"
        problem, _dt, _f = parse_miopen_cmd(
            f"./MIOpenDriver {kw} -n 2 -c {C} -H {Hi} -W {Hi + 3} -k {K} -y {Y} "
            f"-x {Y} -p {pad} -q {pad} -u {stride} -v {stride} -l {dil} -j {dil} "
            f"-m conv -g {groups} -F 2 -t 1"
        )
        wt = 32
        spec = DgradConvSpec(
            problem=problem,
            name="rocke_test_dgrad_ukd",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=64,
            tile_n=64,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_tile_m=wt,
            warp_tile_n=wt,
            warp_tile_k=16,
            wave_size=64,
            pipeline="mem",
            epilogue=epilogue,
            split_k=split_k,
            lds_k_outer=kouter and ARCH == "gfx950",
            uniform_k_decode=ukd,
        )
        ok, reason = is_valid_dgrad_spec(spec, ARCH)
        if not ok:
            self.skipTest(f"invalid spec (uniform_k_decode={ukd}): {reason}")
        spec.validate()
        return _dgrad_run_inprocess(spec, dtype)

    def _assert_exact(self, dtype="bf16", **kw):
        import torch

        ref = self._run(dtype, ukd=False, **kw)
        got = self._run(dtype, ukd=True, **kw)
        self.assertFalse(torch.isnan(got.float()).any(), "dX contains NaN")
        self.assertTrue(
            torch.equal(ref, got),
            "uniform_k_decode reorders no work, so dX must match the plain K "
            f"loop bit for bit; max |diff| = "
            f"{(ref.float() - got.float()).abs().max().item()}",
        )

    def test_stride1_3x3(self):
        self._assert_exact()

    def test_several_k_tiles_per_tap(self):
        # kpg = 3 * tile_k: k_out wraps every third tile.
        self._assert_exact(K=96, tile_k=32)

    def test_tilde_4x4_stride2(self):
        # 4 sub-GEMMs, each with a 2x2 tap set: xdot and ydot both wrap.
        self._assert_exact(Y=4, stride=2, Hi=15)

    def test_dilation2(self):
        self._assert_exact(dil=2, pad=2)

    def test_mouter_default_epilogue_fp16(self):
        self._assert_exact("fp16", kouter=False, epilogue="default")

    def test_grouped(self):
        self._assert_exact(groups=2, C=128, K=128)

    def test_fallback_kpg_not_tile_multiple(self):
        # kpg = 48 is not a multiple of tile_k = 32: the binary takes its plain
        # loop, and the tail tile still relies on the buffer OOB clamp.
        self._assert_exact(K=48, tile_k=32, Hi=13)

    def _assert_close_split_k(self, **kw):
        # The atomic epilogue's accumulation order is not deterministic, so
        # compare within tolerance.
        import torch

        kw = dict(epilogue="default", split_k=2, tile_k=32, **kw)
        ref = self._run("bf16", ukd=False, **kw)
        got = self._run("bf16", ukd=True, **kw)
        self.assertFalse(torch.isnan(got.float()).any(), "dX contains NaN")
        delta = (ref.float() - got.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        self.assertLess(delta / max(scale, 1e-6), 5e-2)

    def test_split_k_matches_plain(self):
        # Exercises a non-zero k_lo decode.
        self._assert_close_split_k()

    def test_split_k_tiles_past_gemm_k(self):
        # gemm_k = 3*3*32 = 288 pads to 320 under split-K 2: the last slice
        # ends in a tile past gemm_k, which the B (y, x) test must zero.
        self._assert_close_split_k(K=32)

    def test_split_k_tilde_stride2(self):
        # 4x4 stride 2: the tilde sub-GEMMs have different tap slices, so
        # each one's slices start at a different decode position.
        self._assert_close_split_k(Y=4, stride=2, Hi=15)

    def test_grouped_fallback_kpg_not_tile_multiple(self):
        # K = 96 is a multiple of tile_k = 32 but kpg = 48 is not: the run-time
        # test must key on kpg, so the grouped binary takes its plain loop.
        self._assert_exact(groups=2, C=64, K=96, tile_k=32)

    def test_grouped_split_k(self):
        self._assert_close_split_k(groups=2, C=128, K=128)

    def test_matches_torch_reference(self):
        import torch

        got = self._run("bf16", ukd=True, Y=4, stride=2, Hi=15)
        # Same problem as _run builds: N=2, C=64, H=15, W=18, K=64, 4x4, s2, p1.
        torch.manual_seed(0)
        dY = torch.empty(2, 7, 9, 64).uniform_(-1.0, 1.0).to(torch.bfloat16)
        W = torch.empty(64, 4, 4, 64).uniform_(-1.0, 1.0).to(torch.bfloat16)
        ref = torch.nn.grad.conv2d_input(
            (2, 64, 15, 18),
            W.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            stride=2,
            padding=1,
        ).permute(0, 2, 3, 1)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        delta = (got.float() - ref).abs().max().item()
        self.assertLess(delta / ref.abs().max().item(), 2e-2)


@unittest.skipUnless(
    ARCH in _MFMA_ARCHES and _HAS_TORCH, f"needs one of {_MFMA_ARCHES} + torch"
)
class TestConvDgradPatch(unittest.TestCase):
    """The patch-embedding dgrad (``DgradConvSpec.patch``) against torch.

    The parity configs only show that both engines emit the same IR; these
    check the dX it stores: both store epilogues, partial M/N tiles, a K tail
    (K not a multiple of tile_k: the tail lands on y = Y and reads no W row),
    stride > filter (pixels no patch covers stay zero), dX runs of X*C that
    are odd or 2 mod 4 at narrow vector widths, and a width-folded ViT stem
    (``fold_width``), checked against the unfolded problem's reference.
    """

    def _check(
        self,
        problem,
        *,
        fold=False,
        epilogue="cshuffle",
        vec_bc=None,
        tile_m=128,
        tile_n=64,
        tile_k=64,
        dtype="bf16",
    ):
        import torch

        sys.path.insert(0, os.path.abspath(_PYDIR))
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            fold_width,
            is_valid_dgrad_spec,
        )
        from rocke.core.arch import ArchTarget

        p = problem
        run = fold_width(p) if fold else p
        if fold:
            self.assertNotEqual(run, p, "the case is meant to fold")
        atom = ArchTarget.from_gfx(ARCH).mma.select_largest_k(
            family="mma", a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=16, n=16
        )
        spec = DgradConvSpec(
            problem=run,
            name="rocke_test_dgrad_patch",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=atom.k,
            wave_size=64,
            pipeline="mem",
            epilogue=epilogue,
            vector_size_a=8,
            vector_size_b=vec_bc,
            vector_size_c=vec_bc if epilogue == "cshuffle" else None,
            patch=True,
        )
        ok, reason = is_valid_dgrad_spec(spec, ARCH)
        self.assertTrue(ok, reason)
        spec.validate()
        got = _dgrad_run_inprocess(spec, dtype, seed=0)

        # _dgrad_run_inprocess's inputs, regenerated: the folded tensors hold
        # the unfolded ones' bytes, so they are the same values reshaped.
        td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        torch.manual_seed(0)
        dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).to(td)
        W = torch.empty(p.K, p.Y, p.X, p.C).uniform_(-1.0, 1.0).to(td)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.C, p.Hi, p.Wi),
            W.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            stride=(p.sH, p.sW),
        ).permute(0, 2, 3, 1)
        got = got.float().reshape(ref.shape)
        self.assertFalse(torch.isnan(got).any(), "dX contains NaN")
        delta = (got - ref).abs().max().item()
        self.assertLess(
            delta / ref.abs().max().item(), 2e-2, f"max |dX - ref| = {delta}"
        )
        # Every pixel no patch covers is exactly zero (the caller's memset).
        uncovered = ref == 0
        if uncovered.any():
            self.assertEqual(got[uncovered].abs().max().item(), 0.0)

    @staticmethod
    def _problem(**kw):
        from kernels.common.conv_implicit_gemm import ConvProblem

        return ConvProblem(**kw)

    def test_stride_equals_filter_cshuffle(self):
        # Parity config 34's problem: M = 4*8*8 = 256 (two M tiles),
        # N = Y*X*C = 48 (a partial N tile), K = 64.
        self._check(
            self._problem(N=4, Hi=32, Wi=32, C=3, K=64, Y=4, X=4, sH=4, sW=4),
            vec_bc=4,
        )

    def test_stride_gaps_default_epilogue(self):
        # Parity config 35's problem (stride 3 > filter 2): a third of the dX
        # rows and columns take no tap; and a height that leaves an edge row.
        self._check(
            self._problem(N=2, Hi=31, Wi=30, C=4, K=32, Y=2, X=2, sH=3, sW=3),
            epilogue="default",
            tile_m=64,
            tile_n=64,
            tile_k=32,
        )

    def test_stride_gaps_cshuffle(self):
        self._check(
            self._problem(N=2, Hi=31, Wi=30, C=4, K=32, Y=2, X=2, sH=3, sW=3),
            vec_bc=8,
            tile_k=32,
        )

    def test_k_tail(self):
        # K = 72 is not a multiple of tile_k = 64: the tail of the second K
        # tile must read no W row (ydot = 1 lands on y = Y).
        for epilogue in ("cshuffle", "default"):
            with self.subTest(epilogue=epilogue):
                self._check(
                    self._problem(N=2, Hi=34, Wi=35, C=3, K=72, Y=4, X=4, sH=4, sW=4),
                    epilogue=epilogue,
                    vec_bc=4 if epilogue == "cshuffle" else None,
                )

    def test_odd_run_scalar_width(self):
        # X*C = 9 and Y*X*C = 27: odd runs, stored one element at a time.
        self._check(
            self._problem(N=3, Hi=27, Wi=30, C=3, K=40, Y=3, X=3, sH=3, sW=3),
            vec_bc=1,
        )

    def test_run_two_mod_four(self):
        # X*C = 18: a run of two elements per vector, then a 2-wide tail.
        self._check(
            self._problem(N=2, Hi=30, Wi=30, C=6, K=64, Y=3, X=3, sH=3, sW=3),
            vec_bc=2,
        )

    def test_folded_vit_stem(self):
        # 16x16/s16 at 224: folds to C = 48 with X = 1; one record of
        # 16*48 = 768 columns, against the unfolded reference.
        self._check(
            self._problem(N=2, Hi=224, Wi=224, C=3, K=768, Y=16, X=16, sH=16, sW=16),
            fold=True,
            tile_n=128,
            vec_bc=8,
        )

    def test_folded_partial_fold(self):
        # 14x14/s14 at 196: Wi/14 = 14, so the fold is the full 14; and a
        # K tail (K = 200).
        self._check(
            self._problem(N=1, Hi=196, Wi=196, C=3, K=200, Y=14, X=14, sH=14, sW=14),
            fold=True,
            vec_bc=2,
        )


# Stride-1 binaries (DgradConvSpec.folds_sub_gemm_record, and the tap-outer
# K loop of tap_outer_k), built as the AOT cache builds them: once, against a
# probe problem with the binary's capabilities (grouped, stride 1), and
# launched on every case below with the shape in the kernargs. Each is
# compared bit for bit with the tilde binary (built against a stride-2 probe:
# runtime record search and loads) on the same launch, and against the torch
# reference with dX poisoned. They all reduce in the same order -- tap-major,
# output-channel chunk, then the tile's MFMA sequence -- so they must agree
# exactly. The tap-outer binaries run where the cache would offer them. The cases are
# adversarial: odd and non-square H/W, N=1, 5x5 and 1x3 filters, pad 0/1/2
# (including pads that shrink the output), M not a multiple of the tile, C not
# a multiple of tile_n, kpg not a multiple of tile_k, grouped, dilation 2.
_STRIDE1_CASES = (
    # (label, dtype, N, Hi, Wi, C, K, Y, X, pad or (pH, pW), dil, groups)
    ("odd_7x13_n1", "bf16", 1, 7, 13, 64, 128, 3, 3, 1, 1, 1),
    ("odd_15x17", "fp16", 3, 15, 17, 96, 64, 3, 3, 1, 1, 1),
    ("5x5_pad2_c48", "bf16", 2, 13, 9, 48, 64, 5, 5, 2, 1, 1),
    ("pad0_shrinks", "fp16", 2, 11, 11, 64, 64, 3, 3, 0, 1, 1),
    ("pad2_grows", "bf16", 1, 9, 10, 64, 64, 3, 3, 2, 1, 1),
    ("k_not_tile_aligned", "bf16", 2, 14, 14, 64, 96, 3, 3, 1, 1, 1),
    ("grouped_g4", "bf16", 2, 12, 12, 128, 256, 3, 3, 1, 1, 4),
    ("grouped_g3_partial_n", "fp16", 1, 13, 11, 288, 480, 3, 3, 1, 1, 3),
    ("1x3_pad1", "fp16", 2, 6, 19, 64, 128, 1, 3, 1, 1, 1),
    ("dil2_pad2", "bf16", 2, 12, 15, 64, 128, 3, 3, 2, 2, 1),
    ("dil2_grouped_pad1", "fp16", 1, 11, 9, 128, 128, 3, 3, 1, 2, 2),
    ("asym_pad_0_2", "fp16", 2, 9, 13, 64, 64, 3, 3, (0, 2), 1, 1),
    # Grids the XCD tile order remaps (gfx950: many groups, one N tile, 3x3)
    # and keeps in launch order (one tap and one N tile).
    ("grouped_g12_xcd", "fp16", 2, 10, 10, 384, 768, 3, 3, 1, 1, 12),
    ("grouped_1x1_g6_launch_order", "bf16", 1, 9, 9, 384, 384, 1, 1, 0, 1, 6),
)


@unittest.skipUnless(
    ARCH in _MFMA_ARCHES and _HAS_TORCH, f"needs one of {_MFMA_ARCHES} + torch"
)
class TestConvDgradStride1Specializations(unittest.TestCase):
    """Stride-1 (folded record) AOT binaries vs the tilde binary."""

    def _problem(self, N, Hi, Wi, C, K, Y, X, pad, dil, groups, stride=1):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        pH, pW = pad if isinstance(pad, tuple) else (pad, pad)
        return ConvProblem(
            N=N,
            Hi=Hi,
            Wi=Wi,
            C=C,
            K=K,
            Y=Y,
            X=X,
            pH=pH,
            pW=pW,
            sH=stride,
            sW=stride,
            dH=dil,
            dW=dil,
            groups=groups,
        )

    def _probe(self, stride, dil=1):
        # The AOT dgrad probe: grouped, 3x3, pad = dilation.
        return self._problem(1, 16, 16, 128, 128, 3, 3, dil, dil, 2, stride=stride)

    def _spec(self, probe, dtype, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec
        from rocke.core.arch import ArchTarget

        wt = 32
        atom = ArchTarget.from_gfx(ARCH).mma.select_largest_k(
            family="mma", a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=wt, n=wt
        )
        base = {
            "problem": probe,
            "name": "rocke_test_dgrad_s1",
            "data": ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_tile_m": wt,
            "warp_tile_n": wt,
            "warp_tile_k": atom.k,
            "pipeline": "mem",
            "epilogue": "cshuffle",
            "vector_size_a": 8,
            "vector_size_b": 8,
            "vector_size_c": 8,
            "lds_k_outer": DgradConvSpec.default_lds_k_outer(
                arch=ARCH, dtype_b=dtype, warp_tile_n=wt, cpg=8
            ),
            "uniform_k_decode": DgradConvSpec.default_uniform_k_decode(arch=ARCH),
        }
        base.update(kw)
        return DgradConvSpec(**base)

    def _binaries(self, dtype):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        tapk = {"tap_outer_k": True, "uniform_k_decode": False}
        out = {
            "s1": self._spec(self._probe(1), dtype),
            "s1_d2": self._spec(self._probe(1, 2), dtype),
            "tilde": self._spec(self._probe(2), dtype),
            # The tap-outer K loop, with both B tile layouts.
            "tapk": self._spec(self._probe(1), dtype, **tapk),
            "tapk_mouter": self._spec(self._probe(1), dtype, lds_k_outer=False, **tapk),
            # The launch-order A/B build of the XCD tile order.
            "tapk_noxcd": self._spec(
                self._probe(1), dtype, xcd_tile_order=False, **tapk
            ),
        }
        for label, spec in out.items():
            ok, why = is_valid_dgrad_spec(spec, ARCH)
            self.assertTrue(ok, f"{label}: {why}")
        self.assertTrue(out["s1"].folds_sub_gemm_record)
        self.assertTrue(out["s1_d2"].folds_sub_gemm_record)
        self.assertFalse(out["tilde"].folds_sub_gemm_record)
        return out

    @staticmethod
    def _serves(spec, problem):
        """The tap-outer binaries serve stride 1 / dilation 1 with kpg a
        multiple of tile_k only (KernelCache.supports_problem)."""
        if not spec.tap_outer_k:
            return True
        return problem.dH == problem.dW == 1 and problem.kpg % spec.tile_k == 0

    def _check(self, label, problem, dtype, outs):
        import torch

        ref = _dgrad_reference(problem, dtype)
        names = list(outs)
        for name in names[1:]:
            self.assertTrue(
                torch.equal(outs[names[0]], outs[name]),
                f"{label}: {names[0]} vs {name}, max |diff| = "
                f"{(outs[names[0]].float() - outs[name].float()).abs().max().item()}",
            )
        got = outs[names[0]].float()
        self.assertFalse(torch.isnan(got).any(), f"{label}: NaN in dX")
        bad = (got - ref).abs() > 1e-2 + 1e-2 * ref.abs()
        self.assertEqual(int(bad.sum()), 0, f"{label}: tolerance violations")

    def test_folded_record_matches_the_tilde_binary(self):
        bins = {d: self._binaries(d) for d in ("fp16", "bf16")}
        for label, dtype, N, Hi, Wi, C, K, Y, X, pad, dil, g in _STRIDE1_CASES:
            with self.subTest(case=label):
                problem = self._problem(N, Hi, Wi, C, K, Y, X, pad, dil, g)
                b = bins[dtype]
                # The folded binary is offered for its own dilation; at
                # stride 1 the record (and the unit-stride epilogue) are the
                # same at any dilation, so the d1 binary is run at d2 too.
                outs = {
                    name: _dgrad_run_inprocess(
                        spec, dtype, poison=True, problem=problem
                    )
                    for name, spec in b.items()
                    if self._serves(spec, problem)
                }
                self._check(label, problem, dtype, outs)

    def test_folded_split_k(self):
        """The folded gemm_k_padded (dg_K rounded to tile_k * split_k)."""
        for label, K in (("k_aligned", 128), ("k_partial", 96)):
            with self.subTest(case=label):
                problem = self._problem(2, 9, 11, 64, K, 3, 3, 1, 1, 1)
                outs = {
                    name: _dgrad_run_inprocess(
                        self._spec(
                            self._probe(stride),
                            "fp16",
                            split_k=2,
                            uniform_k_decode=False,
                            vector_size_a=None,
                            vector_size_c=2,
                        ),
                        "fp16",
                        problem=problem,
                    )
                    for name, stride in (("s1", 1), ("tilde", 2))
                }
                # The split slices meet in atomic adds, whose order is not
                # fixed: compare each against the reference only.
                for name, out in outs.items():
                    self._check(f"{label}/{name}", problem, "fp16", {name: out})


# dY halo (DgradConvSpec.dy_halo) cases: Ho == Hi, Wo == Wi, kpg a multiple
# of tile_k. (label, dtype, N, Hi, Wi, C, K, Y, X, groups); pad (Y-1)/2,
# (X-1)/2.
_HALO_CASES = (
    ("n1_odd_13x17", "fp16", 1, 13, 17, 64, 256, 3, 3, 1),
    ("straddle_3x70", "fp16", 2, 3, 70, 64, 128, 3, 3, 1),
    ("w1_column", "fp16", 2, 33, 1, 64, 192, 3, 3, 1),
    ("1x1_image", "bf16", 1, 1, 1, 64, 64, 3, 3, 1),
    ("partial_n_c96", "bf16", 3, 9, 31, 96, 128, 3, 3, 1),
    ("partial_n_c200", "fp16", 5, 6, 32, 200, 320, 3, 3, 1),
    ("5x5_15x13", "fp16", 2, 15, 13, 128, 64, 5, 5, 1),
    ("7x7_w2", "bf16", 3, 9, 2, 64, 128, 7, 7, 1),
    ("7x7_8x32", "fp16", 4, 8, 32, 64, 64, 7, 7, 1),
    ("grouped_g2", "bf16", 2, 14, 14, 256, 256, 3, 3, 2),
    ("grouped_g4_5x5", "fp16", 1, 11, 9, 256, 512, 5, 5, 4),
    ("1x7_12x20", "bf16", 2, 12, 20, 64, 128, 1, 7, 1),
    ("7x1_21x9", "fp16", 1, 21, 9, 64, 64, 7, 1, 1),
    ("3x1_odd_11x5", "bf16", 3, 11, 5, 96, 128, 3, 1, 1),
    ("3x5_10x14", "bf16", 1, 10, 14, 128, 64, 3, 5, 1),
)
# (label, spec overrides) per tiling.
_HALO_TILINGS = (
    ("halo1_64x64", {"dy_halo": 1}),
    ("halo2_64x64", {"dy_halo": 2}),
    (
        "halo2_128x64_4x1_prio_kpad",
        {
            "dy_halo": 2,
            "tile_m": 128,
            "warp_m": 4,
            "warp_n": 1,
            "dy_halo_setprio": 1,
            "dy_halo_kouter_pad": 32,
        },
    ),
    ("halo2_256x64_4x1", {"dy_halo": 2, "tile_m": 256, "warp_m": 4, "warp_n": 1}),
    ("halo2_64x64_mouter", {"dy_halo": 2, "lds_k_outer": False}),
)
# The widest Wo the test binaries are sized for (every case fits).
_HALO_MAX_WO = 72


@unittest.skipUnless(
    ARCH == "gfx950" and _HAS_TORCH,
    "the dY halo dgrad loop is gfx950 wave64 MFMA + torch",
)
class TestConvDgradDyHalo(unittest.TestCase):
    """The dY halo loop against the reference on adversarial shapes.

    As the AOT cache builds it: one binary per (tiling, filter, halo_max_wo),
    built once against a grouped probe of that filter and width, launched
    through ConvArgs launch values on every case of the filter -- narrower
    images, odd widths, a one-pixel column or image, partial N tiles, groups.
    """

    def _probe(self, Y, X, max_wo):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        return ConvProblem(
            N=1,
            Hi=max(Y, 8),
            Wi=max_wo,
            C=128,
            K=128,
            Y=Y,
            X=X,
            pH=(Y - 1) // 2,
            pW=(X - 1) // 2,
            groups=2,
        )

    def _spec(self, Y, X, dtype, max_wo=_HALO_MAX_WO, **kw):
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        base = {
            "problem": self._probe(Y, X, max_wo),
            "name": "rocke_test_dgrad_halo",
            "data": ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 64,
            "warp_m": 2,
            "warp_n": 2,
            "warp_tile_m": 32,
            "warp_tile_n": 32,
            "warp_tile_k": 16,
            "pipeline": "mem",
            "epilogue": "cshuffle",
            "lds_k_outer": True,
            "tap_outer_k": True,
            "halo_max_wo": max_wo,
        }
        base.update(kw)
        spec = DgradConvSpec(**base)
        ok, why = is_valid_dgrad_spec(spec, ARCH)
        self.assertTrue(ok, why)
        return spec

    def _problem(self, N, Hi, Wi, C, K, Y, X, g):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        return ConvProblem(
            N=N,
            Hi=Hi,
            Wi=Wi,
            C=C,
            K=K,
            Y=Y,
            X=X,
            pH=(Y - 1) // 2,
            pW=(X - 1) // 2,
            groups=g,
        )

    def _check(self, label, out, problem, dtype):
        import torch

        out = out.float()
        ref = _dgrad_reference(problem, dtype)
        bad = ((out - ref).abs() > 1e-2 + 1e-2 * ref.abs()) | ~torch.isfinite(out)
        self.assertEqual(int(bad.sum()), 0, f"{label}: bad elements (NaN/Inf = bad)")

    def test_halo_tilings_match_reference(self):
        ran = 0
        for tlabel, kw in _HALO_TILINGS:
            for label, dtype, N, Hi, Wi, C, K, Y, X, g in _HALO_CASES:
                with self.subTest(case=label, tiling=tlabel):
                    spec = self._spec(Y, X, dtype, **kw)
                    p = self._problem(N, Hi, Wi, C, K, Y, X, g)
                    self.assertEqual((p.Ho, p.Wo), (p.Hi, p.Wi))
                    out = _dgrad_run_inprocess(spec, dtype, poison=True, problem=p)
                    self._check(f"{label}/{tlabel}", out, p, dtype)
                    ran += 1
        self.assertEqual(ran, len(_HALO_TILINGS) * len(_HALO_CASES))

    def test_halo_max_wo_does_not_change_the_result(self):
        # A binary sized for the problem's own width and one sized for a much
        # wider image compute the same dX bit for bit: the bucket changes
        # only the LDS the halo tile reserves and the rows the loader skips.
        import torch

        for label, dtype, N, Hi, Wi, C, K, Y, X, g in _HALO_CASES:
            with self.subTest(case=label):
                p = self._problem(N, Hi, Wi, C, K, Y, X, g)
                tight = self._spec(Y, X, dtype, max_wo=Wi, dy_halo=2)
                wide = self._spec(Y, X, dtype, dy_halo=2)
                a = _dgrad_run_inprocess(tight, dtype, poison=True, problem=p)
                b = _dgrad_run_inprocess(wide, dtype, poison=True, problem=p)
                self.assertTrue(torch.equal(a, b), label)
                self._check(label, a, p, dtype)


@unittest.skipUnless(
    ARCH in _MFMA_ARCHES and _HAS_TORCH, f"needs one of {_MFMA_ARCHES} + torch"
)
class TestConvDgradResolvedSplitK(unittest.TestCase):
    """dX is accurate at the degree the split-K resolver picks, and at its cap.

    A one-tile output over a long reduction (N*Hi*Wi = 64 rows, Y*X*K = 18432)
    is where the cost model splits deepest: on gfx950 it resolves to about a
    hundred slices for bf16, each rounding its partial into the 16-bit dX.
    The override is held to the 16-bit atomic's accuracy limit, so a forced
    degree past it must come back at the limit and still be accurate.
    """

    def _problem(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem

        return ConvProblem(N=1, Hi=8, Wi=8, C=64, K=2048, Y=3, X=3, pH=1, pW=1)

    def _spec(self, dtype, split_k):
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

        return DgradConvSpec(
            problem=self._problem(),
            name="rocke_test_dgrad_resolved_split_k",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            wave_size=64,
            epilogue="cshuffle",
            split_k=split_k,
        )

    def _resolve(self, dtype, **kw):
        from kernels.common.conv_split_k import resolve_conv_split_k

        return resolve_conv_split_k(
            "dgrad",
            self._problem(),
            arch=ARCH,
            dtype=dtype,
            tile_m=64,
            tile_n=64,
            tile_k=32,
            block_size=256,
            **kw,
        ).split_k

    def _assert_accurate(self, dtype, split_k):
        import torch

        p = self._problem()
        got = _dgrad_run_inprocess(self._spec(dtype, split_k), dtype)
        # Same draws as _dgrad_run_inprocess.
        td = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        torch.manual_seed(0)
        dY = torch.empty(p.N, p.Ho, p.Wo, p.K).uniform_(-1.0, 1.0).to(td)
        W = torch.empty(p.K, p.Y, p.X, p.cpg).uniform_(-1.0, 1.0).to(td)
        ref = torch.nn.grad.conv2d_input(
            (p.N, p.C, p.Hi, p.Wi),
            W.float().permute(0, 3, 1, 2),
            dY.float().permute(0, 3, 1, 2),
            padding=1,
        ).permute(0, 2, 3, 1)
        self.assertFalse(torch.isnan(got.float()).any(), "dX contains NaN")
        # The benchmark's verify gate: peak-normalised max error.
        delta = (got.float() - ref).abs().max().item()
        self.assertLess(
            delta / max(ref.abs().max().item(), 1.0),
            5e-2,
            f"{dtype} split_k={split_k}",
        )

    def test_resolved_degree(self):
        from rocke.helpers.split_k_planner import split_k_cost_model

        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                split_k = self._resolve(dtype)
                if split_k_cost_model(ARCH) is not None:
                    self.assertGreater(split_k, 1, "the shape is meant to split")
                self._assert_accurate(dtype, split_k)

    def test_override_past_the_16bit_atomic_limit(self):
        from kernels.common.conv_split_k import ATOMIC_16BIT_MAX_SPLIT_K

        for dtype in ("bf16", "fp16"):
            with self.subTest(dtype=dtype):
                split_k = self._resolve(dtype, override=2 * ATOMIC_16BIT_MAX_SPLIT_K)
                self.assertEqual(split_k, ATOMIC_16BIT_MAX_SPLIT_K)
                self._assert_accurate(dtype, split_k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
