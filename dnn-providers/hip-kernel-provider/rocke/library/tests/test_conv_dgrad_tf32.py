# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side gate for TF32 compute in the dgrad implicit-GEMM kernel.

TF32 here is fp32 storage with ``ConvDataSpec(mma_dtype="tf32")``: A and B stay
fp32 in memory and LDS, and each fragment read from LDS is reinterpreted as
``vec<tf32xN>`` for the catalog's XF32 atom -- native on gfx942, a BF16x3
expansion on gfx950.

No GPU: spec validation, the lowered LLVM IR over the configuration grid, and
one comgr compile per atom whose ISA is read back. A comgr that cannot target
the arch skips the ISA checks; past that a compile error fails. The numerical
counterpart is ``TestConvDgradTf32`` in ``test_conv_dgrad_correctness.py``.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace

# (arch, warp tile, tile_k): every TF32 atom the catalog carries.
_ATOMS = (
    ("gfx950", (16, 16, 32), 64),
    ("gfx950", (32, 32, 16), 32),
    ("gfx942", (16, 16, 8), 32),
    ("gfx942", (32, 32, 4), 32),
)

# The physical MFMA intrinsic each logical TF32 atom must lower to, and how
# many of them one logical atom issues.
_PHYSICAL = {
    ("gfx950", (16, 16, 32)): ("mfma.f32.16x16x32.bf16", 3),
    ("gfx950", (32, 32, 16)): ("mfma.f32.32x32x16.bf16", 3),
    ("gfx942", (16, 16, 8)): ("mfma.f32.16x16x8.xf32", 1),
    ("gfx942", (32, 32, 4)): ("mfma.f32.32x32x4.xf32", 1),
}

# Width of one LDS load of a TF32 fragment: the 8-wide gfx950 fragment is read
# as two 4-wide halves (each claiming 16 B, what ds_read_b128 needs and the
# LDS pool guarantees); the 2-wide gfx942 fragment is one load.
_LDS_LOAD = {"gfx950": 4, "gfx942": 2}


def _problem(*, stride=1, groups=1):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=2,
        Hi=16,
        Wi=16,
        C=64 * groups,
        K=64 * groups,
        Y=3,
        X=3,
        sH=stride,
        sW=stride,
        pH=1,
        pW=1,
        groups=groups,
    )


def _tf32():
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    return ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")


def _spec(warp_tile=(16, 16, 32), tile_k=64, **kw):
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    base = {
        "problem": _problem(),
        "data": _tf32(),
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": tile_k,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": warp_tile[0],
        "warp_tile_n": warp_tile[1],
        "warp_tile_k": warp_tile[2],
        "wave_size": 64,
        "pipeline": "mem",
        "epilogue": "default",
    }
    base.update(kw)
    return DgradConvSpec(**base)


def _build(spec, arch):
    from kernels.common.conv_implicit_gemm_dgrad import build_implicit_gemm_conv_dgrad

    return build_implicit_gemm_conv_dgrad(spec, arch=arch)


def _lower(kernel, arch) -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(kernel, arch=arch)


def _logical_mmas(kernel) -> int:
    """``tile.mma`` ops in the kernel IR -- one per logical TF32 atom."""
    count = 0

    def walk(region):
        nonlocal count
        for op in region.ops:
            if op.name == "tile.mma":
                count += 1
            for r in op.regions or ():
                walk(r)

    walk(kernel.body)
    return count


def _mfma_calls(ll: str) -> list[str]:
    return re.findall(r"call <\d+ x float> @llvm\.amdgcn\.(mfma\.[\w.]+)\(", ll)


class TestTf32Lowering(unittest.TestCase):
    """Every atom x epilogue x split-K x stride x groups builds and lowers."""

    def test_grid(self):
        for (arch, wt, tk), epi, sk, stride, groups in itertools.product(
            _ATOMS, ("default", "cshuffle"), (1, 2), (1, 2), (1, 2)
        ):
            with self.subTest(
                arch=arch, wt=wt, epi=epi, split_k=sk, stride=stride, groups=groups
            ):
                spec = _spec(
                    wt,
                    tk,
                    problem=_problem(stride=stride, groups=groups),
                    epilogue=epi,
                    split_k=sk,
                )
                kernel = _build(spec, arch)
                self.assertTrue(kernel.name.endswith("_tf32"), kernel.name)
                ll = _lower(kernel, arch)
                n = _logical_mmas(kernel)
                self.assertGreater(n, 0)
                physical, issues = _PHYSICAL[(arch, wt)]
                calls = _mfma_calls(ll)
                self.assertEqual(calls.count(physical), issues * n)
                stray = [c for c in calls if c != physical]
                self.assertEqual(stray, [], "an MFMA other than the TF32 atom's")
                # The fragments are fp32 LDS reads claiming no more than
                # 16 B alignment; no 16-bit LDS read stands in for them.
                width = _LDS_LOAD[arch]
                self.assertTrue(
                    re.search(
                        rf"load <{width} x float>, ptr addrspace\(3\) [^\n]*align "
                        rf"{width * 4}\b",
                        ll,
                    ),
                    f"no <{width} x float> LDS fragment load",
                )
                self.assertIsNone(
                    re.search(r"load <8 x float>, ptr addrspace\(3\)", ll),
                    "an 8-wide fp32 LDS load over-claims the pool's alignment",
                )
                self.assertIsNone(
                    re.search(r"load <\d+ x (?:half|bfloat)>, ptr addrspace\(3\)", ll),
                    "a 16-bit LDS load in a TF32 kernel",
                )

    def test_lds_tiles_are_fp32(self):
        kernel = _build(_spec(), "gfx950")
        allocs = []

        def walk(region):
            for op in region.ops:
                if op.name == "tile.smem_alloc":
                    allocs.append(op.result.type)
                for r in op.regions or ():
                    walk(r)

        walk(kernel.body)
        self.assertTrue(allocs)
        for t in allocs:
            self.assertEqual(t.elem.name, "f32", t)

    def test_pipelines_and_schedule_hints(self):
        """compv3/compv4 build; their hint groups count TF32's real issues.

        gfx950: an 8-wide fp32 fragment is two ds_read_b128 and one logical
        atom is three bf16 MFMAs. gfx942: a 2-wide fragment is one read and the
        atom is native.
        """
        mfmas = 2 * 2  # 64x64 tile, 2x2 warps, 16x16 atom
        for arch, wt, tk, reads_per_frag, issues in (
            ("gfx950", (16, 16, 32), 64, 2, 3),
            ("gfx942", (16, 16, 8), 32, 1, 1),
        ):
            for pipeline in ("compv3", "compv4"):
                with self.subTest(arch=arch, pipeline=pipeline):
                    ll = _lower(_build(_spec(wt, tk, pipeline=pipeline), arch), arch)
                    groups = set(
                        re.findall(
                            r"sched\.group\.barrier\(i32 (\d+), i32 (\d+), i32 0\)",
                            ll,
                        )
                    )
                    self.assertEqual(
                        groups,
                        {
                            ("256", str((2 + 2) * reads_per_frag)),  # DS_READ
                            ("8", str(mfmas * issues)),  # MFMA
                        },
                    )


class TestTf32Rejections(unittest.TestCase):
    def _rejected(self, spec, arch, needle):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        ok, why = is_valid_dgrad_spec(spec, arch)
        self.assertFalse(ok)
        self.assertIn(needle, why)
        return why

    def test_lds_k_outer(self):
        spec = _spec(lds_k_outer=True)
        self._rejected(spec, "gfx950", "lds_k_outer")
        with self.assertRaisesRegex(ValueError, "tf32.*lds_k_outer"):
            spec.validate()

    def test_gfx90a_has_no_tf32_atom(self):
        self._rejected(_spec((32, 32, 4), 32), "gfx90a", "gfx90a has no TF32 MFMA atom")

    def test_gfx1250(self):
        # wave32: no TF32 matrix instruction at all.
        spec = _spec((16, 16, 32), 32, wave_size=32)
        self._rejected(spec, "gfx1250", "MFMA-only")
        with self.assertRaisesRegex(ValueError, "MFMA-only"):
            spec.validate()
        self._rejected(_spec((16, 16, 32), 32), "gfx1250", "MFMA-only")

    def test_atom_of_the_other_arch(self):
        self._rejected(_spec((16, 16, 8), 32), "gfx950", "not a TF32 atom on gfx950")
        self._rejected(_spec((16, 16, 32), 64), "gfx942", "not a TF32 atom on gfx942")

    def test_no_tf32_atom_for_the_shape(self):
        spec = _spec((16, 16, 16), 32)
        self._rejected(spec, "gfx950", "not a TF32 atom")
        with self.assertRaises(ValueError):
            spec.validate()

    def test_tf32_mma_needs_fp32_storage(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        spec = _spec(data=ConvDataSpec("bf16", "bf16", "bf16", mma_dtype="tf32"))
        self._rejected(spec, "gfx950", "fp32 A and B storage")
        with self.assertRaisesRegex(ValueError, "fp32 A and B storage"):
            spec.validate()

    def test_tf32_is_not_a_storage_dtype(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        for spelling in ("tf32", "xf32"):
            with self.subTest(spelling=spelling):
                spec = _spec(
                    (16, 16, 32), 64, data=ConvDataSpec(spelling, spelling, "fp32")
                )
                self._rejected(spec, "gfx950", "compute mode")
                with self.assertRaisesRegex(ValueError, "compute mode"):
                    spec.validate()
                with self.assertRaises(ValueError):
                    _build(spec, "gfx950")

    def test_unknown_mma_dtype(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        spec = _spec(data=ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="fp16"))
        self._rejected(spec, "gfx950", "mma_dtype")

    def test_lds_row_stride_must_hold_the_fragment(self):
        from kernels.common.conv_implicit_gemm_dgrad import is_valid_dgrad_spec

        # gfx950 reads its 8-wide fragment as two 16 B halves: the row stride
        # must be a multiple of 4 fp32. 64 + 2 = 66 is not.
        spec = _spec(lds_k_pad=2)
        self._rejected(spec, "gfx950", "row stride")
        with self.assertRaisesRegex(ValueError, "row stride"):
            spec.validate()
        ok, why = is_valid_dgrad_spec(_spec(lds_k_pad=4), "gfx950")
        self.assertTrue(ok, why)
        _build(_spec(lds_k_pad=4), "gfx950")
        # The 2-wide gfx942 fragment only needs an even stride.
        ok, why = is_valid_dgrad_spec(_spec((16, 16, 8), 32, lds_k_pad=2), "gfx942")
        self.assertTrue(ok, why)
        _build(_spec((16, 16, 8), 32, lds_k_pad=2), "gfx942")
        self._rejected(_spec((16, 16, 8), 32, lds_k_pad=3), "gfx942", "row stride")


class TestTf32Naming(unittest.TestCase):
    def test_default_name_is_unchanged(self):
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec

        bf16 = _spec(data=ConvDataSpec("bf16", "bf16", "bf16"))
        self.assertEqual(
            bf16.kernel_name(),
            "conv_igemm_dgrad_N2H16W16C64_K64Y3X3_t64x64x64_w2x2_a16x16x32_"
            "mem_default",
        )
        tf32 = _spec()
        self.assertEqual(tf32.kernel_name(), bf16.kernel_name() + "_tf32")
        fp32 = replace(tf32, data=ConvDataSpec("fp32", "fp32", "fp32"))
        self.assertNotIn("tf32", fp32.kernel_name())

    def test_flag_trails_the_other_flags(self):
        name = _spec(split_k=2, uniform_k_decode=True).kernel_name()
        self.assertTrue(name.endswith("_spk2_ukd_tf32"), name)


# ---------------------------------------------------------------------------
# ISA: one comgr compile per atom (pattern: platform/tests/core/
# test_xf32_comgr_compile.py). gfx942 is a cross-compile.
# ---------------------------------------------------------------------------


def _objdump() -> str | None:
    """ROCm's ``llvm-objdump`` first: a distro build may lack the AMDGPU target."""
    root = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidate = Path(root) / "llvm" / "bin" / "llvm-objdump"
    if candidate.exists():
        return str(candidate)
    return shutil.which("llvm-objdump")


def _comgr(isa: str):
    try:
        from rocke.runtime import comgr
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"comgr unavailable: {exc}") from exc
    if isa not in isas:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"the loaded comgr cannot target {isa}")
    return comgr.build_hsaco_from_llvm_ir


def _mnemonics(hsaco: bytes) -> list[str]:
    from rocke.analysis.isa import _extract_mnemonic, _instruction_lines

    tool = _objdump()
    if tool is None:
        raise unittest.SkipTest("llvm-objdump not found; cannot inspect the ISA")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(hsaco)
        done = subprocess.run(
            [tool, "-d", str(path)], capture_output=True, text=True, check=False
        )
    if done.returncode != 0:
        raise AssertionError(f"llvm-objdump failed: {done.stderr or done.stdout}")
    return [_extract_mnemonic(line) for line in _instruction_lines(done.stdout)]


class TestTf32Isa(unittest.TestCase):
    def test_one_config_per_atom(self):
        for arch, wt, tk in _ATOMS:
            with self.subTest(arch=arch, wt=wt):
                isa = f"amdgcn-amd-amdhsa--{arch}"
                build = _comgr(isa)
                kernel = _build(_spec(wt, tk, epilogue="cshuffle", split_k=2), arch)
                hsaco, _ = build(_lower(kernel, arch), isa=isa)
                self.assertTrue(hsaco, "comgr produced an empty code object")
                seen = _mnemonics(hsaco)
                n = _logical_mmas(kernel)
                physical, issues = _PHYSICAL[(arch, wt)]
                mnemonic = "v_" + physical.replace(".", "_")
                mfmas = [m for m in seen if m.startswith("v_mfma")]
                self.assertEqual(mfmas.count(mnemonic), issues * n)
                self.assertEqual([m for m in mfmas if m != mnemonic], [])
                if arch == "gfx950":
                    # The BF16x3 split narrows with the packed RNE convert.
                    self.assertIn("v_cvt_pk_bf16_f32", seen)
                else:
                    self.assertNotIn("v_cvt_pk_bf16_f32", seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
