# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for TF32 compute in the forward implicit-GEMM conv.

TF32 here means ``ConvDataSpec(dtype_a="fp32", dtype_b="fp32", mma_dtype="tf32")``:
fp32 storage everywhere, with each LDS fragment reinterpreted as ``vec<tf32xN>``
for the catalog's TF32 atom. On gfx942 that atom is native XF32. On gfx950 it is
emulated by three BF16 MFMAs.

No GPU is needed. Everything here is spec validation, the built kernel, and the
lowered LLVM IR. One comgr compile per atom checks the ISA. The numerical
counterpart is the ``fp32`` / ``tf32`` axis of ``test_conv_fwd_correctness.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    _build_implicit_gemm_conv_impl,
    build_implicit_gemm_conv,
    is_valid_spec,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm

_TF32 = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")

# (arch, warp tile m=n, k, physical MFMA intrinsic, is emulated)
_ATOMS = (
    ("gfx950", 16, 32, "llvm.amdgcn.mfma.f32.16x16x32.bf16", True),
    ("gfx950", 32, 16, "llvm.amdgcn.mfma.f32.32x32x16.bf16", True),
    ("gfx942", 16, 8, "llvm.amdgcn.mfma.f32.16x16x8.xf32", False),
    ("gfx942", 32, 4, "llvm.amdgcn.mfma.f32.32x32x4.xf32", False),
)

_PROBLEMS = {
    "2d": ConvProblem(N=2, Hi=14, Wi=14, C=32, K=64, Y=3, X=3, pH=1, pW=1),
    "3d": ConvProblem(
        N=1, Hi=6, Wi=6, C=32, K=32, Y=3, X=3, pH=1, pW=1, Di=4, Z=3, sD=1, pD=1, dD=1
    ),
    "pointwise": ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=1, X=1),
    "grouped": ConvProblem(N=2, Hi=8, Wi=8, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=2),
}

# Every MFMA K-loop: the plain scf.for body (mem/compv3/compv4), the split
# basic body, the hand-unrolled double buffer, and the async-DMA pipeline.
# ``basic`` + ``async_dma`` is not a legal pair for any dtype.
_PIPELINES = (
    ("mem", {}),
    ("compv3", {}),
    ("compv4", {}),
    ("basic", {}),
    ("mem", {"unroll_k": True}),
    ("compv4", {"unroll_k": True}),
    ("mem", {"async_dma": True}),
    ("compv3", {"async_dma": True}),
)
_EPILOGUES = ("default", "cshuffle")


def _spec(
    arch_m: int, k: int, *, problem="2d", pipeline="mem", epilogue="cshuffle", **kw
):
    p = _PROBLEMS[problem] if isinstance(problem, str) else problem
    base = dict(
        problem=p,
        name="tf32_gate",
        data=_TF32,
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=arch_m,
        warp_tile_n=arch_m,
        warp_tile_k=k,
        pipeline=pipeline,
        epilogue=epilogue,
        vector_size_c=1 if epilogue == "default" else None,
        groups=p.groups,
    )
    base.update(kw)
    return ImplicitGemmConvSpec(**base)


def _walk(region):
    for op in region.ops:
        yield op
        for sub in op.regions:
            yield from _walk(sub)


def _logical_mmas(kernel):
    return [op for op in _walk(kernel.body) if op.name == "tile.mma"]


def _tf32_bitcasts(kernel):
    return [
        op
        for op in _walk(kernel.body)
        if op.name == "arith.bitcast" and "tf32" in str(op.result.type)
    ]


class TestConvFwdTf32Build(unittest.TestCase):
    """Every pipeline x epilogue x problem builds and lowers on every TF32 atom."""

    def _check_one(self, arch, m, k, intrinsic, emulated, problem, pipe, extra, epi):
        spec = _spec(m, k, problem=problem, pipeline=pipe, epilogue=epi, **extra)
        ok, why = is_valid_spec(spec, arch)
        self.assertTrue(ok, why)
        kernel = build_implicit_gemm_conv(spec, arch=arch)
        ir = lower_kernel_to_llvm(kernel, arch=arch)

        mmas = _logical_mmas(kernel)
        self.assertTrue(mmas, "no tile.mma in the kernel")
        for op in mmas:
            for operand in op.operands[:2]:
                self.assertIn("tf32", str(operand.type), "MMA operand is not TF32")

        # One reinterpretation per LDS fragment, shared by every (mi, ni) MMA
        # that reads it -- never one per MMA.
        casts = _tf32_bitcasts(kernel)
        cast_results = {id(op.result) for op in casts}
        used = {id(v) for op in mmas for v in op.operands[:2]}
        self.assertEqual(used, cast_results)
        self.assertEqual(len({id(op.operands[0]) for op in casts}), len(casts))

        calls = re.findall(r"call [^@\n]*@(llvm\.amdgcn\.mfma[\w.]*)\(", ir)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {intrinsic}, "unexpected MFMA intrinsic")
        self.assertEqual(len(calls), (3 if emulated else 1) * len(mmas))
        if not emulated:
            self.assertNotIn("bfloat", ir)

        # LDS stays fp32: every operand load reads float, never bf16 or half,
        # and none claims more than the 16 bytes a ds_read_b128 needs.
        lds_loads = re.findall(
            r"load (?:volatile )?(<\d+ x [\w]+>|[\w]+), "
            r"ptr addrspace\(3\)[^\n]*align (\d+)",
            ir,
        )
        self.assertTrue(lds_loads)
        for ty, align in lds_loads:
            self.assertIn("float", ty, f"non-fp32 LDS load {ty}")
            self.assertLessEqual(int(align), 16, f"LDS load {ty} over-claims align")

    def test_matrix(self):
        n = 0
        for arch, m, k, intrinsic, emulated in _ATOMS:
            for problem in _PROBLEMS:
                for pipe, extra in _PIPELINES:
                    for epi in _EPILOGUES:
                        with self.subTest(
                            arch=arch,
                            atom=(m, m, k),
                            problem=problem,
                            pipeline=pipe,
                            extra=extra,
                            epilogue=epi,
                        ):
                            self._check_one(
                                arch,
                                m,
                                k,
                                intrinsic,
                                emulated,
                                problem,
                                pipe,
                                extra,
                                epi,
                            )
                            n += 1
        self.assertEqual(
            n, len(_ATOMS) * len(_PROBLEMS) * len(_PIPELINES) * len(_EPILOGUES)
        )

    def test_depthwise_group_merge(self):
        p = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1, groups=64)
        for arch, m, k, intrinsic, emulated in _ATOMS:
            if m != 16:
                continue
            with self.subTest(arch=arch):
                self._check_one(
                    arch,
                    m,
                    k,
                    intrinsic,
                    emulated,
                    p,
                    "mem",
                    {"group_merge": 4},
                    "cshuffle",
                )

    def test_scheduling_hints_count_physical_instructions(self):
        # compv4 64x64x32 w2x2 on 16x16 atoms: 2x2 MMAs per K atom.
        # gfx950: 4 fragments x 2 ds_read_b128, 4 MMAs x 3 bf16 MFMAs.
        # gfx942: 4 fragments x 1 ds_read_b64,  4 native MFMAs.
        for arch, k, ds, mfma in (("gfx950", 32, 8, 12), ("gfx942", 8, 4, 4)):
            with self.subTest(arch=arch):
                spec = _spec(16, k, pipeline="compv4")
                ir = lower_kernel_to_llvm(
                    build_implicit_gemm_conv(spec, arch=arch), arch=arch
                )
                hints = re.findall(r"sched\.group\.barrier\(i32 (\d+), i32 (\d+)", ir)
                self.assertTrue(hints)
                self.assertEqual(set(hints), {("256", str(ds)), ("8", str(mfma))})


class TestConvFwdTf32Rejections(unittest.TestCase):
    def _rejected(self, spec, arch, needle):
        ok, why = is_valid_spec(spec, arch)
        self.assertFalse(ok, f"{arch} admitted a spec it must reject")
        self.assertIn(needle, why)
        with self.assertRaises(ValueError):
            lower_kernel_to_llvm(build_implicit_gemm_conv(spec, arch=arch), arch=arch)

    def test_tf32_needs_fp32_storage(self):
        data = ConvDataSpec("fp16", "fp16", "fp16", mma_dtype="tf32")
        spec = _spec(16, 32, data=data)
        self._rejected(spec, "gfx950", "fp32 A and B storage")
        with self.assertRaises(ValueError):
            spec.validate()

    def test_tf32_is_not_a_storage_dtype(self):
        for spelling in ("tf32", "xf32"):
            with self.subTest(spelling=spelling):
                spec = _spec(16, 32, data=ConvDataSpec(spelling, spelling, "fp32"))
                self._rejected(spec, "gfx950", "compute mode")

    def test_unknown_mma_dtype(self):
        data = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="bf16")
        spec = _spec(16, 32, data=data)
        self._rejected(spec, "gfx950", "mma_dtype")

    def test_non_tf32_atom_shapes(self):
        for arch, m, k in (
            ("gfx950", 16, 4),  # the fp32 atom
            ("gfx950", 16, 8),  # gfx942's TF32 atom
            ("gfx942", 16, 32),  # gfx950's emulated atom
            ("gfx942", 32, 16),
        ):
            with self.subTest(arch=arch, atom=(m, m, k)):
                self._rejected(_spec(m, k), arch, "not a TF32 atom")

    def test_arch_without_tf32(self):
        self._rejected(_spec(16, 8), "gfx90a", "has no TF32 MFMA atom")

    def test_wmma_targets(self):
        for arch in ("gfx1151", "gfx1201", "gfx1250"):
            with self.subTest(arch=arch):
                spec = _spec(16, 16, warp_m=1, warp_n=1, wave_size=32)
                self._rejected(spec, arch, "MFMA-only")

    def test_misaligning_lds_row_stride(self):
        # 8-wide fragment (gfx950): row stride must be a multiple of 4 floats.
        self._rejected(_spec(16, 32, lds_k_pad=2), "gfx950", "LDS row stride")
        self._rejected(_spec(32, 16, lds_k_pad=6), "gfx950", "LDS row stride")
        # 2-wide fragment (gfx942): a multiple of 2.
        self._rejected(_spec(16, 8, lds_k_pad=1), "gfx942", "LDS row stride")

    def test_aligned_lds_row_strides_build(self):
        for arch, m, k, pad in (
            ("gfx950", 16, 32, 4),
            ("gfx950", 16, 32, 0),
            ("gfx950", 32, 16, 16),
            ("gfx942", 16, 8, 2),
            ("gfx942", 32, 4, 0),
        ):
            with self.subTest(arch=arch, pad=pad):
                spec = _spec(m, k, lds_k_pad=pad)
                ok, why = is_valid_spec(spec, arch)
                self.assertTrue(ok, why)
                ir = lower_kernel_to_llvm(
                    build_implicit_gemm_conv(spec, arch=arch), arch=arch
                )
                aligns = re.findall(r"ptr addrspace\(3\)[^\n]*align (\d+)", ir)
                self.assertTrue(aligns)
                self.assertLessEqual(max(int(a) for a in aligns), 16)

    def test_a_operand_override(self):
        def _override(*_args):  # pragma: no cover - never reached
            raise AssertionError("override called for a rejected spec")

        with self.assertRaisesRegex(ValueError, "a_operand_override"):
            _build_implicit_gemm_conv_impl(
                _spec(16, 32), arch="gfx950", a_operand_override=_override
            )


class TestConvFwdTf32Name(unittest.TestCase):
    def test_default_name_unchanged(self):
        spec = _spec(16, 32, data=ConvDataSpec())
        self.assertEqual(
            spec.kernel_name(),
            "tf32_gate_N2H14W14C32_K64Y3X3_t64x64x32_w2x2_a16x16x32_mem_cshuffle",
        )
        fp32 = _spec(16, 4, data=ConvDataSpec("fp32", "fp32", "fp32"))
        self.assertFalse(fp32.kernel_name().endswith("tf32"))

    def test_tf32_name_is_tagged_last(self):
        tf32 = _spec(16, 32, unroll_k=True)
        plain = replace(tf32, data=ConvDataSpec("fp32", "fp32", "fp32"))
        self.assertEqual(tf32.kernel_name(), plain.kernel_name() + "_tf32")


# ---------------------------------------------------------------------------
# ISA: one comgr compile per atom
# ---------------------------------------------------------------------------


def _llvm_tool(name: str):
    """ROCm's own LLVM tool first: a distro build may not know the gfx target."""
    candidate = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin" / name
    return str(candidate) if candidate.exists() else shutil.which(name)


def _compile(ir: str, arch: str) -> bytes:
    """comgr code object; skips only when comgr cannot target ``arch``."""
    try:
        from rocke.runtime import comgr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    isa = f"amdgcn-amd-amdhsa--{arch}"
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    if isa not in isas:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"the loaded comgr cannot target {arch}")
    hsaco, _ = comgr.build_hsaco_from_llvm_ir(ir, isa=isa)
    return hsaco


def _disassemble(hsaco: bytes):
    """``(mnemonics, notes)``; skips when llvm-objdump is missing."""
    from rocke.analysis.isa import _extract_mnemonic, _instruction_lines

    objdump = _llvm_tool("llvm-objdump")
    if objdump is None:
        raise unittest.SkipTest("llvm-objdump not found; cannot inspect the ISA")
    readelf = _llvm_tool("llvm-readelf")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(hsaco)
        dis = subprocess.run(
            [objdump, "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        notes = (
            subprocess.run(
                [readelf, "--notes", str(path)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            if readelf
            else ""
        )
    return [_extract_mnemonic(line) for line in _instruction_lines(dis)], notes


class TestConvFwdTf32Isa(unittest.TestCase):
    """The TF32 kernels compile to the instructions the design rests on."""

    def test_one_config_per_atom(self):
        for arch, m, k, _intrinsic, emulated in _ATOMS:
            with self.subTest(arch=arch, atom=(m, m, k)):
                spec = _spec(m, k, pipeline="compv4")
                kernel = build_implicit_gemm_conv(spec, arch=arch)
                n_mma = len(_logical_mmas(kernel))
                hsaco = _compile(lower_kernel_to_llvm(kernel, arch=arch), arch)
                self.assertTrue(hsaco)
                seen, notes = _disassemble(hsaco)
                mfmas = [s for s in seen if s.startswith("v_mfma")]
                if emulated:
                    phys = f"v_mfma_f32_{m}x{m}x{k}_bf16"
                    self.assertEqual(set(mfmas), {phys})
                    self.assertEqual(len(mfmas), 3 * n_mma)
                    self.assertIn("v_cvt_pk_bf16_f32", seen)
                    # 8-wide fp32 fragments: two 128-bit reads each.
                    self.assertIn("ds_read_b128", seen)
                else:
                    self.assertEqual(set(mfmas), {f"v_mfma_f32_{m}x{m}x{k}_xf32"})
                    self.assertEqual(len(mfmas), n_mma)
                    self.assertNotIn("v_cvt_pk_bf16_f32", seen)
                scratch = re.search(r"\.private_segment_fixed_size:\s+(\d+)", notes)
                if scratch:
                    self.assertEqual(int(scratch.group(1)), 0, "TF32 kernel spills")


if __name__ == "__main__":
    unittest.main()
