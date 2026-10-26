# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side tests for TF32 implicit-GEMM backward-weight convolution.

A TF32 wgrad keeps fp32 in memory and in an M-outer LDS tile, and reinterprets
each fragment it reads from LDS as ``vec<tf32xN>`` for the catalog's TF32 atom:
native ``v_mfma_*_xf32`` on gfx942, three BF16 MFMAs (bf16x3) on gfx950.

No GPU needed. Covers, for every TF32 atom of both arches:

* build + lower over epilogue (default, cshuffle) x split-K (atomic, two-stage)
  x stream-K (linear, tree, atomic) x pipeline (mem, compv4, basic, unroll_k)
  x 2-D / 3-D / grouped, checking the IR the design rests on -- fp32 LDS tiles,
  one bitcast per fragment, the TF32 op_id, and in the lowered LLVM IR exactly
  three BF16 MFMAs per logical MMA on gfx950 and one xf32 MFMA on gfx942;
* a comgr ISA compile for one config per atom (skips without comgr);
* the validator's rejections (K-outer, async, WMMA, storage spellings, a
  misaligned LDS row stride, an atom of the other arch);
* that the kernel name is unchanged unless TF32 is set.

GPU numerics live in ``test_conv_wgrad_correctness.py`` (``TestConvWgradTf32``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
)

_TF32 = ConvDataSpec(dtype_a="fp32", dtype_b="fp32", dtype_d="fp32", mma_dtype="tf32")
_FP32 = ConvDataSpec(dtype_a="fp32", dtype_b="fp32", dtype_d="fp32")

_P2 = ConvProblem(N=2, Hi=12, Wi=12, C=32, K=96, Y=3, X=3, pH=1, pW=1)
_P3 = ConvProblem(
    N=1, Di=5, Hi=6, Wi=6, C=16, K=32, Z=3, Y=3, X=3, pD=1, sD=1, dD=1, pH=1, pW=1
)
# kpg = 40 leaves a partial M tile in every group.
_PG = ConvProblem(N=2, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3)

# (arch, m=n, k, logical op_id, LLVM intrinsic, physical MFMAs per logical MMA)
_ATOMS = (
    ("gfx950", 32, 16, "mfma_f32_32x32x16_xf32", "mfma.f32.32x32x16.bf16", 3),
    ("gfx950", 16, 32, "mfma_f32_16x16x32_xf32", "mfma.f32.16x16x32.bf16", 3),
    ("gfx942", 32, 4, "mfma_f32_32x32x4_xf32", "mfma.f32.32x32x4.xf32", 1),
    ("gfx942", 16, 8, "mfma_f32_16x16x8_xf32", "mfma.f32.16x16x8.xf32", 1),
)

# Every variant the TF32 path is admitted on. The epilogue, reduction and
# pipeline knobs are independent of the MMA dtype, so each is crossed with
# every atom and both spatial ranks below.
_VARIANTS = {
    "default": {},
    "cshuffle": dict(epilogue="cshuffle"),
    "compv4": dict(pipeline="compv4"),
    "basic": dict(pipeline="basic"),
    "unroll_k": dict(pipeline="compv4", unroll_k=True),
    "splitk_atomic": dict(split_k=4),
    "splitk_atomic_cshuffle": dict(split_k=4, epilogue="cshuffle"),
    "two_stage": dict(split_k=4, two_stage=True),
    "streamk_linear": dict(streamk="dp_sk", streamk_ctas=6),
    "streamk_tree": dict(streamk="dp_sk", streamk_ctas=6, streamk_reduction="tree"),
    "streamk_atomic": dict(streamk="dp_sk", streamk_ctas=6, streamk_reduction="atomic"),
    "streamk_persistent": dict(streamk="persistent", streamk_ctas=6),
    # Stride tile_k + 4: 16-byte aligned, not 32 -- legal because no TF32 LDS
    # load claims more than 16 bytes.
    "pad4": dict(lds_k_pad=4),
}


def _spec(arch_atom=_ATOMS[0], problem=_P2, data=_TF32, **kw) -> WgradConvSpec:
    _arch, edge, k, *_ = arch_atom
    base = dict(
        problem=problem,
        data=data,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=k,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _lower(spec: WgradConvSpec, arch: str):
    from rocke.core.ir_print import print_ir
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    return kernel, print_ir(kernel), _lower_kernel_to_llvm_python(kernel, arch=arch)


class TestTf32BuildMatrix(unittest.TestCase):
    """Every admitted variant builds, lowers, and carries the TF32 contract."""

    def _check(self, atom, problem, kw) -> None:
        arch, _edge, _k, op_id, intrinsic, per_logical = atom
        spec = _spec(atom, problem, **kw)
        ok, why = is_valid_wgrad_spec(spec, arch)
        self.assertTrue(ok, why)
        self.assertTrue(spec.kernel_name().endswith("_tf32"), spec.kernel_name())
        _k, ir, ll = _lower(spec, arch)

        # fp32 LDS tiles; no fragment is ever read as 16-bit.
        allocs = re.findall(r"tile\.smem_alloc : smem<(\w+),", ir)
        self.assertTrue(allocs)
        self.assertEqual(set(allocs), {"f32"}, allocs)
        self.assertNotRegex(ll, r"load <\d+ x (half|bfloat)>, ptr addrspace\(3\)")

        # Every logical MMA is the TF32 atom, fed by a bitcast of an fp32
        # fragment -- one per fragment, so never more bitcasts than reads.
        n = spec.atom.a_per_lane
        mmas = re.findall(r"tile\.mma [^\n]*\{op_id = \"(\w+)\"\}", ir)
        self.assertTrue(mmas)
        self.assertEqual(set(mmas), {op_id})
        # A fragment is read as loads of at most 4 fp32 (two <4 x float> on
        # gfx950, one <2 x float> on gfx942), joined and cast once.
        w = min(n, 4)
        part_reads = len(
            re.findall(
                rf"tile\.smem_load_vN %[AB]_smem\d+, [^\n]*vec = {w}\}} : vec<f32x{w}>",
                ir,
            )
        )
        casts = len(re.findall(rf"arith\.bitcast [^\n]*: vec<tf32x{n}>", ir))
        self.assertGreater(casts, 0)
        self.assertEqual(casts * (n // w), part_reads)

        # No LDS load claims more than the 16 bytes the LDS pool is aligned to.
        aligns = [
            int(a) for a in re.findall(r"ptr addrspace\(3\) [^\n]*, align (\d+)", ll)
        ]
        self.assertTrue(aligns)
        self.assertLessEqual(max(aligns), 16)

        # Lowered: exactly ``per_logical`` physical MFMAs per logical one, and
        # nothing but the expected intrinsic.
        calls = re.findall(r"= call [^\n]*@llvm\.amdgcn\.(mfma\.[\w.]+)\(", ll)
        self.assertEqual(set(calls), {intrinsic})
        self.assertEqual(len(calls), per_logical * len(mmas))

    def test_matrix(self):
        for atom in _ATOMS:
            for rank, problem in (("2d", _P2), ("3d", _P3)):
                for name, kw in _VARIANTS.items():
                    with self.subTest(arch=atom[0], atom=atom[3], rank=rank, v=name):
                        self._check(atom, problem, kw)

    def test_grouped(self):
        for atom in _ATOMS:
            for name in ("default", "cshuffle", "splitk_atomic", "two_stage"):
                with self.subTest(arch=atom[0], atom=atom[3], v=name):
                    self._check(atom, _PG, _VARIANTS[name])

    def test_split_k_uses_the_f32_atomic(self):
        # The epilogue is keyed on dW alone: an fp32 dW takes the scalar f32
        # atomic, with or without TF32 compute.
        for atom in _ATOMS:
            for name in ("splitk_atomic", "two_stage", "streamk_atomic"):
                with self.subTest(arch=atom[0], v=name):
                    _k, _ir, ll = _lower(_spec(atom, **_VARIANTS[name]), atom[0])
                    self.assertRegex(
                        ll, r"atomicrmw fadd ptr addrspace\(1\)[^\n]* float"
                    )

    def test_scheduling_hints_count_physical_work(self):
        # sched_group_barrier(DS_READ=0x100, n) / (MFMA=0x8, n) after each K
        # step: gfx950 reads an 8-float fragment as two ds_read_b128 and
        # issues three MFMAs per logical one; gfx942 one read, one MFMA.
        for atom in _ATOMS:
            arch, edge, k, *_ = atom
            spec = _spec(atom, pipeline="compv4")
            _k, _ir, ll = _lower(spec, arch)
            hints = re.findall(
                r"sched\.group\.barrier\(i32 (\d+), i32 (\d+), i32 0\)", ll
            )
            ds = {int(c) for m, c in hints if int(m) == 0x100}
            mf = {int(c) for m, c in hints if int(m) == 0x8}
            frags = spec.mfmas_per_warp_m + spec.mfmas_per_warp_n
            mmas = spec.mfmas_per_warp_m * spec.mfmas_per_warp_n
            reads, issues = (2, 3) if arch == "gfx950" else (1, 1)
            with self.subTest(arch=arch, atom=f"{edge}x{edge}x{k}"):
                self.assertEqual(ds, {frags * reads})
                self.assertEqual(mf, {mmas * issues})


class TestTf32Rejections(unittest.TestCase):
    """The paths TF32 cannot take are refused, by both gates, with a reason."""

    def _rejected(self, spec, arch, needle):
        ok, why = is_valid_wgrad_spec(spec, arch)
        self.assertFalse(ok)
        self.assertIn(needle, why)

    def _both(self, spec, arch, needle):
        self._rejected(spec, arch, needle)
        with self.assertRaisesRegex(ValueError, re.escape(needle)):
            spec.validate()

    def test_lds_k_outer(self):
        self._both(_spec(lds_k_outer=True), "gfx950", "lds_k_outer")

    def test_async_dma(self):
        self._both(_spec(async_dma=True), "gfx950", "async_dma")
        self._both(
            _spec(async_dma=True, lds_k_outer=True, pipeline="compv4"),
            "gfx950",
            "lds_k_outer",
        )

    def test_wmma(self):
        for arch in ("gfx1201", "gfx1250"):
            spec = _spec(
                tile_m=32, tile_n=32, warp_m=1, warp_n=1, warp_tile_m=16,
                warp_tile_n=16, warp_tile_k=32, wave_size=32,
            )  # fmt: skip
            with self.subTest(arch=arch):
                self._both(spec, arch, "MFMA-only")

    def test_tf32_storage_spelling(self):
        for bad in ("tf32", "xf32"):
            spec = _spec(data=ConvDataSpec(bad, bad, "fp32"))
            with self.subTest(dtype=bad):
                self._both(spec, "gfx950", "compute mode")

    def test_16_bit_storage(self):
        spec = _spec(data=ConvDataSpec("bf16", "bf16", "fp32", mma_dtype="tf32"))
        self._both(spec, "gfx950", "needs fp32 A and B storage")

    def test_16_bit_dw(self):
        spec = _spec(data=ConvDataSpec("fp32", "fp32", "bf16", mma_dtype="tf32"))
        self._both(spec, "gfx950", "needs an fp32 output")

    def test_misaligned_row_stride(self):
        # tile_k + pad must be a multiple of min(fragment, 4): 4 on gfx950
        # (the 8-wide fragment is two 4-wide loads), 2 on gfx942.
        self._both(_spec(lds_k_pad=2), "gfx950", "is not a multiple of 4")
        self._both(_spec(_ATOMS[2], lds_k_pad=1), "gfx942", "is not a multiple of 2")
        for atom, pad in ((_ATOMS[0], 4), (_ATOMS[1], 4), (_ATOMS[2], 2)):
            ok, why = is_valid_wgrad_spec(_spec(atom, lds_k_pad=pad), atom[0])
            self.assertTrue(ok, why)

    def test_atom_of_the_other_arch(self):
        self._rejected(_spec(_ATOMS[0]), "gfx942", "not a TF32 atom on gfx942")
        self._rejected(_spec(_ATOMS[2]), "gfx950", "not a TF32 atom on gfx950")

    def test_no_tf32_atom(self):
        self._both(_spec(warp_tile_k=2), "gfx950", "not a TF32 atom")


class TestTf32KernelName(unittest.TestCase):
    def test_default_name_is_unchanged(self):
        fp32 = _spec(data=_FP32, warp_tile_k=2)
        self.assertEqual(
            fp32.kernel_name(),
            "conv_igemm_wgrad_N2H12W12C32_K96Y3X3_t64x64x32_w2x2_a32x32x2_mem_default",
        )
        self.assertNotIn("tf32", _spec(data=ConvDataSpec()).kernel_name())

    def test_tf32_is_one_trailing_flag(self):
        for atom in _ATOMS:
            for name, kw in _VARIANTS.items():
                tf32 = _spec(atom, **kw)
                fp32 = replace(tf32, data=_FP32)
                with self.subTest(arch=atom[0], v=name):
                    self.assertEqual(tf32.kernel_name(), fp32.kernel_name() + "_tf32")

    def test_bf16_and_tf32_do_not_collide(self):
        # Same shape tag on gfx950 (a32x32x16); only the flag tells them apart.
        bf16 = _spec(data=ConvDataSpec("bf16", "bf16", "fp32"))
        self.assertNotEqual(bf16.kernel_name(), _spec().kernel_name())

    def test_storage_helpers_take_the_storage_dtype(self):
        # The storage dtype a TF32 caller passes keeps fp32's answers.
        self.assertEqual(WgradConvSpec.default_vector_sizes(32, 96, "fp32"), (4, 4, 4))
        self.assertFalse(
            WgradConvSpec.default_lds_k_outer(
                arch="gfx950", dtype_a="fp32", dtype_b="fp32",
                warp_tile_m=32, warp_tile_n=32,
            )  # fmt: skip
        )
        self.assertIsNone(
            WgradConvSpec.default_lds_k_pad(
                arch="gfx942", tile_k=32, dtype_a="fp32", dtype_b="fp32"
            )
        )


# ---------------------------------------------------------------------------
# ISA (comgr; no GPU)
# ---------------------------------------------------------------------------


def _objdump():
    """ROCm's ``llvm-objdump`` first: a system one may predate gfx950."""
    root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    cand = root / "llvm" / "bin" / "llvm-objdump"
    return str(cand) if cand.exists() else shutil.which("llvm-objdump")


class TestTf32Isa(unittest.TestCase):
    """One config per atom compiles to a code object with the expected MFMAs.

    The lowered-IR checks above prove the recipe is emitted; this proves it
    survives codegen: on gfx950 exactly three ``v_mfma_*_bf16`` per logical
    MMA plus the packed RNE convert, on gfx942 the native ``v_mfma_*_xf32``,
    and no scratch on either.
    """

    _MNEMONIC = {
        "mfma_f32_32x32x16_xf32": "v_mfma_f32_32x32x16_bf16",
        "mfma_f32_16x16x32_xf32": "v_mfma_f32_16x16x32_bf16",
        "mfma_f32_32x32x4_xf32": "v_mfma_f32_32x32x4_xf32",
        "mfma_f32_16x16x8_xf32": "v_mfma_f32_16x16x8_xf32",
    }

    def _mnemonics(self, arch: str, ll: str) -> list:
        try:
            from rocke.runtime import comgr

            isas = comgr.supported_isas()
        except Exception as exc:  # noqa: BLE001 -- environment-dependent
            self.skipTest(f"comgr unavailable: {exc}")
        isa = f"amdgcn-amd-amdhsa--{arch}"
        if isa not in isas:
            self.skipTest(f"the loaded comgr cannot target {arch}")
        hsaco, _ = comgr.build_hsaco_from_llvm_ir(ll, isa=isa)
        self.assertTrue(hsaco)
        tool = _objdump()
        if tool is None:
            self.skipTest("llvm-objdump not found")
        from rocke.analysis.isa import _extract_mnemonic, _instruction_lines

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "k.hsaco"
            path.write_bytes(hsaco)
            done = subprocess.run(
                [tool, "-d", str(path)], capture_output=True, text=True, check=True
            )
        return [_extract_mnemonic(x) for x in _instruction_lines(done.stdout)]

    def test_each_atom(self):
        for atom in _ATOMS:
            arch, _edge, _k, op_id, _intr, per_logical = atom
            with self.subTest(arch=arch, atom=op_id):
                spec = _spec(atom, pipeline="compv4", epilogue="cshuffle")
                _kern, ir, ll = _lower(spec, arch)
                logical = len(re.findall(r"tile\.mma ", ir))
                seen = self._mnemonics(arch, ll)
                want = self._MNEMONIC[op_id]
                mfma = [m for m in seen if m.startswith("v_mfma")]
                self.assertEqual(set(mfma), {want})
                self.assertEqual(len(mfma), per_logical * logical)
                if arch == "gfx950":
                    self.assertIn("v_cvt_pk_bf16_f32", seen)
                self.assertFalse([m for m in seen if m.startswith("scratch_")])


if __name__ == "__main__":
    unittest.main()
