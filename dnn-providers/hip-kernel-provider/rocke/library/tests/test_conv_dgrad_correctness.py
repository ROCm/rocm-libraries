# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numeric correctness tests for the conv backward-data (dgrad) implicit-GEMM kernel.

Builds dgrad kernels on the running GPU and compares each output against a
float64 data-gradient reference of the stored inputs
(``builders.common.conv_reference.conv_dgrad_reference``).
``TestConvDgradCorrectness`` checks, per problem, every candidate of a pinned
sample of the dgrad sweep (``_conv_dgrad_sweep_sample``) against the sweep's
``--verify`` bound; the other classes build the kernels they name.  Covers:

  - stride=1 (direct-store epilogue, no atomics)
  - stride=2 (tilde-decomposition, atomic epilogue)
  - split_k > 1 (atomic epilogue)
  - pointwise 1x1/stride1/pad0 (flat-offset fast path, no tilde decomposition)
  - bf16 and fp32 data types
  - gfx1151 / gfx1201 via WMMA candidates
  - gfx1250 via WMMA wavelet pipeline (stride=1 mem + stride>1 / split_k wavelet)

Requires a ROCm GPU (skip otherwise).

Run, from rocke/library:
  PYTHONPATH=../platform/python:. python -m pytest \
    tests/test_conv_dgrad_correctness.py
"""

from __future__ import annotations

import multiprocessing
import re
import unittest
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

import numpy as np
import pytest
from rocke import compile_kernel
from rocke.numeric import storage
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.host_cpus import available_cpus
from rocke.tests.lanes import arch_skip_reason

from benchmarks.common import conv_fill
from benchmarks.common.benchmark_implicit_gemm_conv import (
    compile_dgrad_sweep_candidate,
    dgrad_sweep_grid,
    sample_combos,
    verify_verdict,
)
from builders.common.conv_reference import (
    conv_dgrad_reference,
    tf32_reference_inputs,
    tf32_reference_tol,
)
from kernels.common.conv_implicit_gemm import ConvProblem
from kernels.common.conv_implicit_gemm_dgrad import build_implicit_gemm_conv_dgrad

from ._conv_dgrad_sweep_sample import (
    SWEEP_SAMPLE_FRACTION,
    SWEEP_SAMPLE_SEED,
    SWEEP_SAMPLE_TILINGS,
    sweep_sample,
)

ARCH = get_device_arch(0)

_MFMA_ARCHES = ("gfx90a", "gfx942", "gfx950")
_WMMA_ARCHES = ("gfx1151", "gfx1201")
_WMMA_WAVELET_ARCHES = ("gfx1250",)
_SUPPORTED_ARCHES = _MFMA_ARCHES + _WMMA_ARCHES + _WMMA_WAVELET_ARCHES

_SKIP_REASON = (
    f"needs a supported ROCm GPU ({', '.join(_SUPPORTED_ARCHES)}); "
    f"detected arch={ARCH!r}"
)

#: Most worker processes the sample compiles in: each worker holds its own
#: imports and COMGR in memory.
_MAX_SAMPLE_COMPILE_WORKERS = 32


# Each test below declares its rough gfx942 CI seconds, so the CI shards and
# their workers share these heavy keys (rocke.tests.ctest_shard).
@unittest.skipUnless(ARCH in _SUPPORTED_ARCHES, _SKIP_REASON)
class TestConvDgradCorrectness(unittest.TestCase):
    """Every candidate of the sampled dgrad sweep computes the right dX.

    For each problem, every tile configuration of ``SWEEP_SAMPLE_TILINGS``
    that is a sweep candidate with the test's split-K mode is built and
    compiled in worker processes (:func:`compile_dgrad_sweep_candidate`),
    launched here, and must pass the sweep's ``--verify``
    (:func:`verify_verdict`). A kernel that stores its output is launched on
    poisoned output, an atomic (split-K) one on zeroed output. A build or
    compile that raises fails the test, and so does a problem with no
    candidate.
    """

    @classmethod
    def _start_compile_pool(cls):
        # One pool for the class: its workers import the kernel modules once.
        cls._compile_pool = ProcessPoolExecutor(
            max_workers=min(available_cpus(), _MAX_SAMPLE_COMPILE_WORKERS),
            mp_context=multiprocessing.get_context("spawn"),
        )

    @classmethod
    def setUpClass(cls):
        cls._start_compile_pool()

    @classmethod
    def tearDownClass(cls):
        cls._compile_pool.shutdown(cancel_futures=True)

    def _verify(self, dtype: str, problem: ConvProblem, *, split_k: int) -> None:
        """Check the sample's candidates for ``problem`` in ``dtype``;
        ``split_k`` is 1, or -1 for the split-K resolver's pick."""
        combos = sweep_sample(split_k)
        futures = [
            self._compile_pool.submit(
                compile_dgrad_sweep_candidate,
                combo,
                problem=problem,
                dtype=dtype,
                arch=ARCH,
            )
            for combo in combos
        ]
        grad_output, weights = _dgrad_inputs(problem, dtype)
        ref = _dgrad_reference(problem, dtype)
        checked, failures = 0, []
        try:
            for combo, future in zip(combos, futures):
                try:
                    compiled = future.result()
                except BrokenExecutor as error:
                    # A worker died (a crash, out of memory): every pending
                    # candidate would report it, so stop here, and give the
                    # next test a working pool.
                    self._compile_pool.shutdown(wait=False, cancel_futures=True)
                    type(self)._start_compile_pool()
                    self.fail(f"the compile pool died at {combo}: {error!r}")
                except Exception as error:  # noqa: BLE001 - reported per candidate
                    failures.append(f"{combo}: build/compile raised {error!r}")
                    continue
                if compiled is None:
                    continue
                spec, artifact = compiled
                got = _dgrad_launch(
                    spec,
                    dtype,
                    grad_output,
                    weights,
                    hsaco=artifact.hsaco,
                    kernel_name=artifact.kernel_name,
                    poison=True,
                )
                checked += 1
                passed, error = verify_verdict(
                    got,
                    ref,
                    compute_dtype=dtype,
                    arch=ARCH,
                    ref_floor=conv_fill.ref_floor("dgrad"),
                )
                if not passed:
                    failures.append(f"{artifact.kernel_name}: rel_err={error:.2e}")
        finally:
            for future in futures:
                future.cancel()
        case = f"{dtype} {problem.short()} split_k={split_k} on {ARCH}"
        if failures:
            self.fail(
                f"{len(failures)} sampled dgrad candidates failed for {case}:\n"
                + "\n".join(failures)
            )
        self.assertGreater(checked, 0, f"no sampled dgrad candidate for {case}")

    # ---- stride=1 (direct store, no atomics) ---------------------------------

    @pytest.mark.rocke_shard_cost(11)
    def test_fp16_stride1(self):
        """fp16 dgrad, stride=1 — single sub-GEMM, direct-store epilogue."""
        self._verify(
            "fp16",
            ConvProblem(N=4, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(11)
    def test_bf16_stride1(self):
        """bf16 dgrad, stride=1."""
        self._verify(
            "bf16",
            ConvProblem(N=4, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(10)
    def test_fp32_stride1(self):
        """fp32 dgrad, stride=1."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"fp32 dgrad candidates require MFMA; running on {ARCH}")
        self._verify(
            "fp32",
            ConvProblem(N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1),
            split_k=1,
        )

    # ---- pointwise 1x1 (flat-offset fast path) -------------------------------

    @pytest.mark.rocke_shard_cost(13)
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
            "fp16",
            ConvProblem(N=2, Hi=7, Wi=7, C=64, K=48, Y=1, X=1, pH=0, pW=0),
            split_k=1,
        )

    # ---- stride=2 (tilde decomposition, atomic epilogue) ---------------------

    @pytest.mark.rocke_shard_cost(19)
    def test_fp16_stride2(self):
        """fp16 dgrad, stride=2 — tilde decomposition with atomic epilogue."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "fp16",
            ConvProblem(
                N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1, sH=2, sW=2
            ),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(16)
    def test_bf16_stride2(self):
        """bf16 dgrad, stride=2."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"stride>1 dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "bf16",
            ConvProblem(
                N=2, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1, sH=2, sW=2
            ),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(25)
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
            "bf16",
            ConvProblem(
                N=2, Hi=32, Wi=32, C=3, K=32, Y=16, X=16, pH=0, pW=0, sH=16, sW=16
            ),
            split_k=1,
        )

    # ---- split_k > 1 (atomic epilogue) ---------------------------------------

    @pytest.mark.rocke_shard_cost(31)
    def test_fp16_split_k(self):
        """fp16 dgrad, split_k auto-selected — exercises atomic reduction path."""
        if ARCH not in _MFMA_ARCHES + _WMMA_WAVELET_ARCHES:
            self.skipTest(
                f"split_k dgrad requires atomic-add or wavelet pipeline; running on {ARCH}"
            )
        self._verify(
            "fp16",
            ConvProblem(N=4, Hi=28, Wi=28, C=64, K=128, Y=3, X=3, pH=1, pW=1),
            split_k=-1,
        )

    # ---- larger realistic shape ----------------------------------------------

    @pytest.mark.rocke_shard_cost(28)
    def test_fp16_resnet_shape(self):
        """fp16 dgrad, ResNet-style shape N8 H56 W56 C64 K64 R3 S3."""
        self._verify(
            "fp16",
            ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
            split_k=-1,
        )

    # ---- grouped (grid-per-group on blockIdx.y) ------------------------------

    @pytest.mark.rocke_shard_cost(12)
    def test_fp16_grouped_stride1(self):
        """fp16 grouped dgrad, groups=4 (cpg=kpg=16), stride=1 direct store."""
        self._verify(
            "fp16",
            ConvProblem(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(16)
    def test_bf16_grouped_stride1(self):
        """bf16 grouped dgrad, groups=4, stride=1."""
        self._verify(
            "bf16",
            ConvProblem(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=4),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(15)
    def test_fp16_grouped_stride2(self):
        """fp16 grouped dgrad, groups=4, stride=2 — tilde decomposition path."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"stride>1 dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "fp16",
            ConvProblem(
                N=2,
                Hi=16,
                Wi=16,
                C=64,
                K=64,
                Y=3,
                X=3,
                pH=1,
                pW=1,
                sH=2,
                sW=2,
                groups=4,
            ),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(22)
    def test_fp16_grouped_odd_kpg(self):
        """Non-power-of-two kpg (C=K=48, groups=8 -> cpg=kpg=6): guards against
        the k_sub decode-divisor trap (must divide by kpg, not total K)."""
        self._verify(
            "fp16",
            ConvProblem(N=2, Hi=16, Wi=16, C=48, K=48, Y=3, X=3, pH=1, pW=1, groups=8),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(35)
    def test_fp16_grouped_split_k(self):
        """fp16 grouped dgrad with split_k>1 — group on y, split_k on z compose;
        even cpg (=16) keeps the packed <2 x f16> atomic pairs in-group."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"split_k dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "fp16",
            ConvProblem(N=4, Hi=28, Wi=28, C=64, K=128, Y=3, X=3, pH=1, pW=1, groups=4),
            split_k=-1,
        )

    @pytest.mark.rocke_shard_cost(33)
    def test_bf16_depthwise_large_filter(self):
        """Depthwise (cpg=kpg=1) dgrad: a one-column GEMM per group, every
        operand scalar. A tall, asymmetric filter with per-axis "same"
        padding pins the separate Y/X and pH/pW decode."""
        self._verify(
            "bf16",
            ConvProblem(
                N=2, Hi=14, Wi=14, C=32, K=32, Y=11, X=3, pH=5, pW=1, groups=32
            ),
            split_k=1,
        )

    @pytest.mark.rocke_shard_cost(39)
    def test_fp16_depthwise_stride2(self):
        """Depthwise dgrad on the stride-2 tilde decomposition path."""
        if ARCH not in _MFMA_ARCHES:
            self.skipTest(f"stride>1 dgrad requires CDNA atomic-add; running on {ARCH}")
        self._verify(
            "fp16",
            ConvProblem(
                N=2,
                Hi=15,
                Wi=13,
                C=64,
                K=64,
                Y=5,
                X=3,
                pH=2,
                pW=0,
                sH=2,
                sW=2,
                groups=64,
            ),
            split_k=1,
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
# In-process build and launch
# ---------------------------------------------------------------------------


def _dgrad_inputs(problem, dtype, seed=0):
    """The stored ``(dY, W)`` of ``problem`` an in-process launch reads:
    U(-1, 1) drawn from ``seed``, dY first, as ``dtype`` host data."""
    p = problem
    rng = np.random.default_rng(seed)
    dy = storage.encode(rng.uniform(-1.0, 1.0, (p.N, p.Ho, p.Wo, p.K)), dtype)
    w = storage.encode(rng.uniform(-1.0, 1.0, (p.K, p.Y, p.X, p.cpg)), dtype)
    return dy, w


def _dgrad_run_inprocess(spec, dtype, seed=0, poison=False, problem=None):
    """Build, compile and launch one dgrad kernel (:func:`_dgrad_launch`) on
    the inputs :func:`_dgrad_inputs` draws from ``seed`` for the launched
    problem, and return dX (NHWC) as float32."""
    artifact = compile_kernel(
        build_implicit_gemm_conv_dgrad(spec, arch=ARCH), arch=ARCH
    )
    p = spec.problem if problem is None else problem
    dy, w = _dgrad_inputs(p, dtype, seed)
    return _dgrad_launch(
        spec,
        dtype,
        dy,
        w,
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        poison=poison,
        problem=problem,
    )


def _dgrad_launch(
    spec, dtype, grad_output, weights, *, hsaco, kernel_name, poison=False, problem=None
):
    """Launch the compiled dgrad kernel of ``spec`` on the stored ``dtype``
    inputs ``grad_output`` (dY) and ``weights`` (W) and return dX (NHWC): the float32 values of the
    stored output.

    The kernel was built from ``spec`` (its problem is the build probe) and is
    launched on ``problem`` (default: the spec's own), as an AOT binary is:
    the shape travels as kernargs and the sub-GEMM records are packed for the
    launched problem. ``poison=True`` pre-fills dX with 0xFF bytes (NaN in
    every stored dtype) when the kernel stores rather than accumulates, so an
    output the kernel never writes shows up as NaN instead of passing as zero.
    """
    import ctypes
    from dataclasses import replace as dc_replace

    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_implicit_gemm_dgrad import pack_sub_gemm_buffer
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem if problem is None else problem
    dX = np.zeros((p.N, p.Hi, p.Wi, p.C), dtype=grad_output.dtype)

    rt = Runtime()
    dY_d, W_d, dX_d = (
        rt.alloc(grad_output.nbytes),
        rt.alloc(weights.nbytes),
        rt.alloc(dX.nbytes),
    )
    rt.memcpy_h2d(dY_d, as_u8_buffer(grad_output), grad_output.nbytes)
    rt.memcpy_h2d(W_d, as_u8_buffer(weights), weights.nbytes)
    # split-K atomic-add needs a zeroed dX.
    rt.memset(dX_d, 0xFF if (poison and not spec.needs_atomic) else 0, dX.nbytes)

    sub_gemms = dc_replace(spec, problem=p).compute_sub_gemms()
    buf = pack_sub_gemm_buffer(sub_gemms, spec.tile_m, spec.tile_n)
    raw = (ctypes.c_int32 * len(buf))(*buf)
    sg_d = rt.alloc(ctypes.sizeof(raw))
    rt.memcpy_h2d(sg_d, raw, ctypes.sizeof(raw))

    # The tilde record buffer is part of the dgrad ABI, not an extension.
    sig = conv_args_signature(dtype, direction="dgrad")
    launcher = KernelLauncher(hsaco=hsaco, kernel_name=kernel_name, signature=sig)
    launcher(
        ConvArgs.from_problem(
            p, direction="dgrad", tile_m=spec.tile_m, tile_n=spec.tile_n
        ).to_launch_values(
            int(dY_d),
            int(W_d),
            int(dX_d),
            grad_output.nbytes,
            weights.nbytes,
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
    rt.memcpy_d2h(as_u8_buffer(dX), dX_d, dX.nbytes)
    for d in (dY_d, W_d, dX_d, sg_d):
        rt.free(d)
    return storage.decode(dX, dtype)


def _dgrad_reference(problem, dtype, seed=0, *, tf32=False):
    """float64 reference dX (NHWC) of the inputs :func:`_dgrad_inputs` draws.

    ``tf32=True`` masks the fp32 inputs as ``ARCH``'s TF32 MMA reads them
    (:func:`tf32_reference_inputs`)."""
    dy, w = (storage.decode(x, dtype) for x in _dgrad_inputs(problem, dtype, seed))
    if tf32:
        dy, w = tf32_reference_inputs(dy, ARCH), tf32_reference_inputs(w, ARCH)
    return conv_dgrad_reference(dy, w, problem)


# ---------------------------------------------------------------------------
# K-outer LDS (transpose-read) A/B
# ---------------------------------------------------------------------------

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
    ARCH in _KOUTER_WAVE,
    arch_skip_reason(_KOUTER_WAVE, f"K-outer dgrad needs {'/'.join(_KOUTER_WAVE)}"),
)
class TestConvDgradLdsKOuter(unittest.TestCase):
    """The K-outer B tile must be a pure re-layout of the M-outer default."""

    def _pair(
        self, dtype, *, warp_tile_mn, tile_k, epilogue, split_k, stride=1, Hi=14, K=64
    ):
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
        ref, got = self._pair(dtype, **kw)
        self.assertTrue(
            np.array_equal(ref, got),
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
        # bitwise. Compare the two within tolerance instead.
        ref, got = self._pair(
            "bf16", warp_tile_mn=32, tile_k=64, epilogue="default", split_k=4
        )
        for out in (ref, got):
            self.assertFalse(np.isnan(out).any(), "dX contains NaN")
        delta = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max())
        self.assertLess(delta / max(scale, 1e-6), 5e-2)


@unittest.skipUnless(ARCH in _MFMA_ARCHES, f"needs one of {_MFMA_ARCHES}")
class TestConvDgradUniformKDecode(unittest.TestCase):
    """``uniform_k_decode`` must compute exactly what the plain K loop computes.

    Both loops visit the same k values in the same order and feed the same
    MFMA sequence, so with split_k == 1 dX must match bit for bit. The cases
    cover the per-tile decode loop (kpg a multiple of tile_k) -- including the
    k_out -> xdot -> ydot wraps of a multi-tap filter, the tilde sub-GEMMs,
    dilation, grouping and both B layouts -- and the fallback loop the same
    binary runs when kpg is not a multiple of tile_k.
    """

    @staticmethod
    def _problem(dtype, *, groups=1, K=64, C=64, Hi=14, Y=3, stride=1, dil=1, pad=1):
        """The problem :meth:`_run` launches: N=2, H=Hi, W=Hi+3, YxY filter."""
        from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

        kw = "convbfp16" if dtype == "bf16" else "convfp16"
        problem, _dt, _f = parse_miopen_cmd(
            f"./MIOpenDriver {kw} -n 2 -c {C} -H {Hi} -W {Hi + 3} -k {K} -y {Y} "
            f"-x {Y} -p {pad} -q {pad} -u {stride} -v {stride} -l {dil} -j {dil} "
            f"-m conv -g {groups} -F 2 -t 1"
        )
        return problem

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
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        problem = self._problem(
            dtype,
            groups=groups,
            K=K,
            C=C,
            Hi=Hi,
            Y=Y,
            stride=stride,
            dil=dil,
            pad=pad,
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
            # The case waits for the runner of an arch that takes the spec.
            self.skipTest(
                arch_skip_reason(
                    [
                        arch
                        for arch in _MFMA_ARCHES
                        if is_valid_dgrad_spec(spec, arch)[0]
                    ],
                    f"invalid spec (uniform_k_decode={ukd}): {reason}",
                )
            )
        spec.validate()
        return _dgrad_run_inprocess(spec, dtype)

    def _assert_exact(self, dtype="bf16", **kw):
        ref = self._run(dtype, ukd=False, **kw)
        got = self._run(dtype, ukd=True, **kw)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        self.assertTrue(
            np.array_equal(ref, got),
            "uniform_k_decode reorders no work, so dX must match the plain K "
            f"loop bit for bit; max |diff| = {float(np.abs(ref - got).max())}",
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
        kw = dict(epilogue="default", split_k=2, tile_k=32, **kw)
        ref = self._run("bf16", ukd=False, **kw)
        got = self._run("bf16", ukd=True, **kw)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        delta = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max())
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

    def test_matches_reference(self):
        got = self._run("bf16", ukd=True, Y=4, stride=2, Hi=15)
        ref = _dgrad_reference(self._problem("bf16", Y=4, stride=2, Hi=15), "bf16")
        self.assertEqual(got.shape, ref.shape)
        delta = float(np.abs(got - ref).max())
        self.assertLess(delta / float(np.abs(ref).max()), 2e-2)


@unittest.skipUnless(ARCH in _MFMA_ARCHES, f"needs one of {_MFMA_ARCHES}")
class TestConvDgradPatch(unittest.TestCase):
    """The patch-embedding dgrad (``DgradConvSpec.patch``) against the reference.

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

        # The unfolded problem's reference on the same draws: the folded
        # tensors hold the unfolded ones' bytes, so they are the same values
        # reshaped.
        ref = _dgrad_reference(p, dtype, seed=0)
        got = got.reshape(ref.shape)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        delta = float(np.abs(got - ref).max())
        self.assertLess(
            delta / float(np.abs(ref).max()), 2e-2, f"max |dX - ref| = {delta}"
        )
        # Every pixel no patch covers is exactly zero (the caller's memset).
        uncovered = ref == 0
        if uncovered.any():
            self.assertEqual(float(np.abs(got[uncovered]).max()), 0.0)

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
# runtime record search and loads) on the same launch, and against the float64
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


@unittest.skipUnless(ARCH in _MFMA_ARCHES, f"needs one of {_MFMA_ARCHES}")
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
        ref = _dgrad_reference(problem, dtype)
        names = list(outs)
        for name in names[1:]:
            self.assertTrue(
                np.array_equal(outs[names[0]], outs[name]),
                f"{label}: {names[0]} vs {name}, max |diff| = "
                f"{float(np.abs(outs[names[0]] - outs[name]).max())}",
            )
        got = outs[names[0]]
        self.assertFalse(np.isnan(got).any(), f"{label}: NaN in dX")
        bad = np.abs(got - ref) > 1e-2 + 1e-2 * np.abs(ref)
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
    ARCH == "gfx950",
    arch_skip_reason(("gfx950",), "the dY halo dgrad loop is gfx950 wave64 MFMA"),
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
        ref = _dgrad_reference(problem, dtype)
        bad = (np.abs(out - ref) > 1e-2 + 1e-2 * np.abs(ref)) | ~np.isfinite(out)
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
        for label, dtype, N, Hi, Wi, C, K, Y, X, g in _HALO_CASES:
            with self.subTest(case=label):
                p = self._problem(N, Hi, Wi, C, K, Y, X, g)
                tight = self._spec(Y, X, dtype, max_wo=Wi, dy_halo=2)
                wide = self._spec(Y, X, dtype, dy_halo=2)
                a = _dgrad_run_inprocess(tight, dtype, poison=True, problem=p)
                b = _dgrad_run_inprocess(wide, dtype, poison=True, problem=p)
                self.assertTrue(np.array_equal(a, b), label)
                self._check(label, a, p, dtype)


@unittest.skipUnless(ARCH in _MFMA_ARCHES, f"needs one of {_MFMA_ARCHES}")
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

    # The tile is the widest 32x32 atom the arch's MMA table offers for 16-bit
    # inputs (32x32x16 on gfx950, 32x32x8 on gfx90a/gfx942) and it must
    # validate. These tests are about the split-K degree and the 16-bit atomic.
    def _spec(self, dtype, split_k):
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )
        from rocke.core.arch import ArchTarget

        atom = ArchTarget.from_gfx(ARCH).mma.select_largest_k(
            family="mma", a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=32, n=32
        )
        spec = DgradConvSpec(
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
            warp_tile_k=atom.k,
            wave_size=64,
            epilogue="cshuffle",
            split_k=split_k,
        )
        ok, why = is_valid_dgrad_spec(spec, arch=ARCH)
        assert ok, why
        return spec

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
        p = self._problem()
        got = _dgrad_run_inprocess(self._spec(dtype, split_k), dtype)
        ref = _dgrad_reference(p, dtype)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        # The benchmark's verify gate: peak-normalised max error.
        delta = float(np.abs(got - ref).max())
        self.assertLess(
            delta / max(float(np.abs(ref).max()), 1.0),
            5e-2,
            f"{dtype} split_k={split_k}",
        )

    def test_resolved_degree(self):
        from kernels.conv.split_k_planner import split_k_cost_model

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


# TF32 is judged against conv_reference.tf32_reference_inputs (gfx942's native
# XF32 on masked inputs) with conv_reference.tf32_reference_tol, as in
# test_conv_fwd_correctness / test_conv_wgrad_correctness.

# Per arch, the two TF32 atoms as (warp_tile_m, warp_tile_n, warp_tile_k),
# 16x16 first.
_TF32_WARP_TILES = {
    "gfx950": ((16, 16, 32), (32, 32, 16)),
    "gfx942": ((16, 16, 8), (32, 32, 4)),
}


@unittest.skipUnless(
    ARCH in _TF32_WARP_TILES, "TF32 dgrad correctness needs gfx942 or gfx950"
)
class TestConvDgradTf32(unittest.TestCase):
    """fp32 storage, TF32 MMA (``mma_dtype="tf32"``) against a float64 reference.

    Built and launched in-process (``_dgrad_run_inprocess``), not through the
    benchmark driver, so every case pins exactly one spec.
    """

    def _check(
        self, *, warp_tile, tile_k, epilogue, split_k, stride, groups=1, Hi=14, K=64
    ):
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        C = 64 * groups
        K = K * groups
        problem = ConvProblem(
            N=2,
            Hi=Hi,
            Wi=Hi + 3,
            C=C,
            K=K,
            Y=3,
            X=3,
            sH=stride,
            sW=stride,
            pH=1,
            pW=1,
            groups=groups,
        )
        wm, wn, wk = warp_tile
        spec = DgradConvSpec(
            problem=problem,
            name="rocke_test_dgrad_tf32",
            data=ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32"),
            tile_m=64,
            tile_n=64,
            tile_k=tile_k,
            warp_m=2,
            warp_n=2,
            warp_tile_m=wm,
            warp_tile_n=wn,
            warp_tile_k=wk,
            wave_size=64,
            pipeline="mem",
            epilogue=epilogue,
            split_k=split_k,
        )
        ok, reason = is_valid_dgrad_spec(spec, ARCH)
        self.assertTrue(ok, reason)
        spec.validate()
        got = _dgrad_run_inprocess(spec, "fp32")
        ref = _dgrad_reference(spec.problem, "fp32", tf32=True)
        self.assertEqual(got.shape, ref.shape)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        rel = float(np.abs(got - ref).max()) / float(np.abs(ref).max())
        self.assertLess(
            rel, tf32_reference_tol(ARCH), f"relative-to-peak error {rel:.3e}"
        )

    def test_cases(self):
        for warp_tile, tile_k in zip(_TF32_WARP_TILES[ARCH], (64, 32)):
            for epilogue in ("default", "cshuffle"):
                for split_k in (1, 2):
                    for stride in (1, 2):
                        with self.subTest(
                            warp_tile=warp_tile,
                            epilogue=epilogue,
                            split_k=split_k,
                            stride=stride,
                        ):
                            self._check(
                                warp_tile=warp_tile,
                                tile_k=tile_k,
                                epilogue=epilogue,
                                split_k=split_k,
                                stride=stride,
                            )

    def test_grouped(self):
        for split_k in (1, 2):
            with self.subTest(split_k=split_k):
                self._check(
                    warp_tile=_TF32_WARP_TILES[ARCH][1],
                    tile_k=32,
                    epilogue="cshuffle",
                    split_k=split_k,
                    stride=2,
                    groups=2,
                )

    def test_k_not_tile_aligned(self):
        # gemm_k = 9 * 48 is not a multiple of tile_k = 64: the last tile reads
        # past real K and relies on the buffer OOB clamp returning zero.
        self._check(
            warp_tile=_TF32_WARP_TILES[ARCH][0],
            tile_k=64,
            epilogue="default",
            split_k=1,
            stride=1,
            Hi=13,
            K=48,
        )

    def test_stride1_and_patch_k_loops(self):
        # The tap-outer K loop, its dY halo forms and the patch (one-GEMM)
        # decomposition read their fragments through the same TF32 path as
        # the tilde K loop; pin that each still feeds the TF32 atom.
        from kernels.common._conv_implicit_gemm_common import (
            ConvDataSpec,
            ConvProblem,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        data = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")
        same = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        patch = ConvProblem(N=2, Hi=28, Wi=28, C=64, K=64, Y=2, X=2, sH=2, sW=2)
        cases = [
            ("tap_outer_k", dict(tap_outer_k=True), same),
            ("dy_halo=1", dict(tap_outer_k=True, dy_halo=1, halo_max_wo=14), same),
            ("dy_halo=2", dict(tap_outer_k=True, dy_halo=2, halo_max_wo=14), same),
            ("patch", dict(patch=True), patch),
        ]
        for warp_tile in _TF32_WARP_TILES[ARCH]:
            for label, knobs, p in cases:
                with self.subTest(warp_tile=warp_tile, knobs=label):
                    spec = DgradConvSpec(
                        problem=p,
                        data=data,
                        tile_m=64,
                        tile_n=64,
                        tile_k=64,
                        warp_m=2,
                        warp_n=2,
                        warp_tile_m=warp_tile[0],
                        warp_tile_n=warp_tile[1],
                        warp_tile_k=warp_tile[2],
                        epilogue="cshuffle",
                        **knobs,
                    )
                    ok, reason = is_valid_dgrad_spec(spec, ARCH)
                    self.assertTrue(ok, reason)
                    got = _dgrad_run_inprocess(spec, "fp32", seed=0)
                    ref = _dgrad_reference(p, "fp32", seed=0, tf32=True)
                    rel = float(np.abs(got - ref).max()) / float(np.abs(ref).max())
                    self.assertLess(
                        rel,
                        tf32_reference_tol(ARCH),
                        f"relative-to-peak error {rel:.3e}",
                    )


@unittest.skipUnless(
    ARCH == "gfx950",
    arch_skip_reason(("gfx950",), "the cached split-K cshuffle dgrad is gfx950"),
)
class TestConvDgradSplitKCshuffle(unittest.TestCase):
    """Split-K with the cshuffle epilogue: the LDS-staged tile is atomic-added
    into dX through the tilde address function (packed pairs for 16-bit dX).

    The geometry of the cached split-K kernels (16x16x32 atom, K-outer B, 8-wide
    loads); the atomic accumulation order is not deterministic, so dX is held
    to the float64 reference within tolerance. Each case is launched on a
    problem other than its build probe, as an AOT binary is.
    """

    def _check(
        self,
        dtype,
        *,
        split_k,
        vec_c=2,
        stride=1,
        groups=1,
        C=64,
        K=128,
        Hi=14,
        tile_m=32,
        tile_n=128,
        pipeline="mem",
    ):
        from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd
        from kernels.common.conv_implicit_gemm import ConvDataSpec
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        kw = "convbfp16" if dtype == "bf16" else "convfp16"

        def problem(n, h):
            return parse_miopen_cmd(
                f"./MIOpenDriver {kw} -n {n} -c {C} -H {h} -W {h + 3} -k {K} "
                f"-y 3 -x 3 -p 1 -q 1 -u {stride} -v {stride} -l 1 -j 1 -m conv "
                f"-g {groups} -F 2 -t 1"
            )[0]

        spec = DgradConvSpec(
            problem=problem(1, 8),
            name="rocke_test_dgrad_sk_cshuffle",
            data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=64,
            warp_m=2,
            warp_n=4,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            wave_size=64,
            pipeline=pipeline,
            epilogue="cshuffle",
            lds_k_outer=True,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=vec_c,
            split_k=split_k,
        )
        ok, reason = is_valid_dgrad_spec(spec, ARCH)
        self.assertTrue(ok, reason)
        spec.validate()
        p = problem(2, Hi)
        got = _dgrad_run_inprocess(spec, dtype, problem=p)
        ref = _dgrad_reference(p, dtype)
        self.assertFalse(np.isnan(got).any(), "dX contains NaN")
        delta = float(np.abs(got - ref).max())
        scale = float(np.abs(ref).max())
        self.assertLess(delta / max(scale, 1e-6), 2e-2)

    def test_fp16_stride1_split4(self):
        self._check("fp16", split_k=4)

    def test_bf16_stride2_split2(self):
        # The tilde sub-GEMMs: dX rows a sub-GEMM skips stay untouched.
        self._check("bf16", split_k=2, stride=2, Hi=15)

    def test_fp16_grouped_split8_compv3(self):
        # The group base in the dX address; kpg = 64 split eight ways.
        self._check(
            "fp16", split_k=8, groups=2, C=128, K=128, tile_m=64, pipeline="compv3"
        )

    def test_fp16_width8_split16(self):
        # Four packed pairs per lane; tiles past the channel run are masked.
        self._check("fp16", split_k=16, vec_c=8, C=72, tile_m=64, tile_n=64)


def test_the_pinned_sample_is_the_sweeps_draw():
    """``SWEEP_SAMPLE_TILINGS`` is what the dgrad sweep's ``--sample`` draws
    from its grid today, in order, so a change to the sweep's axes or its
    sampling rule fails here instead of silently changing what
    ``TestConvDgradCorrectness`` covers."""
    drawn = sample_combos(
        dgrad_sweep_grid((1,)), SWEEP_SAMPLE_FRACTION, SWEEP_SAMPLE_SEED
    )
    assert [tuple(combo)[:-1] for combo in drawn] == list(SWEEP_SAMPLE_TILINGS)
