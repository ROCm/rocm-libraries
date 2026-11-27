# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp32 storage and TF32 for the MFMA direct-conv kernels.

Covers the forward fixed-cpg kernels (``DirectConv16cSpec`` with and without
``fold_k32``, ``DirectConv8cSpec``, ``DirectConv32cSpec``), the generic
``DirectConvSpec`` (plain loops, ``fold_k32``, runtime K loop, ``waves_k``
preload, the ``waves_k == 1`` weight preload with ``waves_m``, ``stage_out``,
runtime channels), the storage-width register and LDS budgets, the MFMA dgrad
pipeline (``make_dgrad_fprop_spec``
plus the transpose / reorganize / coalesced weight copies) and the
``DirectConv4cSpec`` rejection.

``dtype="fp32"`` runs the exact fp32 atoms (16x16x4 / 32x32x2); ``"tf32"`` the
catalog TF32 atoms (gfx942 native XF32, gfx950 BF16x3 emulation). Both keep the
fp16 kernels' K-contiguous fragments; :func:`_mma_f32_logical` splits or pads
them to the atom, and :func:`_mma_f32_chain` pairs 4-wide fragments into the
8-wide gfx950 TF32 atom where a kernel chains several on one accumulator.

The CPU part (no GPU) builds and lowers every variant for both dtypes on gfx950
and gfx942, checks the MFMA intrinsics and their counts, the rejections, and
that fp16 / bf16 kernel names are unchanged; one comgr compile per arch checks
the ISA, including that the BF16x3 split of the loop-invariant weights is
hoisted out of the row loop. The GPU part runs every variant on the device
(gfx942 or gfx950) for both dtypes against a float64 numpy reference (no TF32
anywhere in it). On gfx942 the TF32 reference reads the inputs masked to what
the native XF32 MFMA consumes (``conv_reference.tf32_reference_inputs``).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.numeric import storage
from rocke.numeric.compare import peak_relative_error
from rocke.runtime import comgr
from rocke.tests.comgr import require_comgr_target
from rocke.tests.lanes import BUILD_VARIANT, skip_until, skip_until_reason
from rocke.tests.llc import find_llvm_tool

from builders.common.conv_reference import tf32_reference_inputs, tf32_reference_tol
from kernels.common.conv_direct_grouped import (
    DirectCoalescedWeightsDgradSpec,
    DirectConv4cSpec,
    DirectConv8cSpec,
    DirectConv16cSpec,
    DirectConv32cSpec,
    DirectConvProblem,
    DirectConvSpec,
    DirectReorganizeWeightsSpec,
    DirectTransposeWeightsDgradSpec,
    _f32_mma_plan,
    build_direct_coalesced_weights_dgrad,
    build_direct_conv,
    build_direct_conv_4c,
    build_direct_conv_8c,
    build_direct_conv_16c,
    build_direct_conv_32c,
    build_direct_reorganize_weights,
    build_direct_transpose_weights_dgrad,
    direct_conv_lds_bytes,
    direct_dgrad_coalesced_workspace_bytes,
    direct_dgrad_workspace_bytes,
    is_valid_spec,
    is_valid_spec_4c,
    is_valid_spec_8c,
    is_valid_spec_16c,
    is_valid_spec_32c,
    make_dgrad_fprop_spec,
    preload_weight_vgprs,
)

from ._direct_conv_numeric import (
    direct_dgrad_reference,
    direct_fwd_reference,
    launch_direct_kernel,
    poisoned,
)

_ARCHES = ("gfx950", "gfx942")
_F32_DTYPES = ("fp32", "tf32")


def _p(cpg, dtype, *, kpg=None, groups=8, N=2, H=14, W=14, KH=3, PAD=1):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=cpg if kpg is None else kpg,
        KH=KH,
        KW=KH,
        PAD=PAD,
        dtype=dtype,
    )


def _rtc(spec):
    return replace(
        spec,
        runtime_channels=True,
        k_atoms=spec.problem_k_atoms(),
        m_tiles=spec.problem_m_tiles(),
    )


# Forward variants: name -> (builder, validator, spec factory, MFMA m).
_FWD = {
    "16c_k32": (
        build_direct_conv_16c,
        is_valid_spec_16c,
        lambda dt: DirectConv16cSpec(problem=_p(16, dt), fold_k32=True),
        16,
    ),
    "16c": (
        build_direct_conv_16c,
        is_valid_spec_16c,
        lambda dt: DirectConv16cSpec(problem=_p(16, dt), fold_k32=False),
        16,
    ),
    "16c_1x1": (
        build_direct_conv_16c,
        is_valid_spec_16c,
        lambda dt: DirectConv16cSpec(
            problem=_p(16, dt, KH=1, PAD=0, H=16, W=16), fold_k32=False
        ),
        16,
    ),
    "8c": (
        build_direct_conv_8c,
        is_valid_spec_8c,
        lambda dt: DirectConv8cSpec(problem=_p(8, dt)),
        16,
    ),
    "32c": (
        build_direct_conv_32c,
        is_valid_spec_32c,
        lambda dt: DirectConv32cSpec(problem=_p(32, dt, H=8, W=40)),
        32,
    ),
    "gen": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(problem=_p(32, dt), block_h=8),
        16,
    ),
    "gen_k32": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(problem=_p(32, dt), block_h=8, fold_k32=True),
        16,
    ),
    "gen_partial": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(problem=_p(12, dt, kpg=20, W=21), block_h=8),
        16,
    ),
    "gen_rk": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(32, dt, groups=2),
            block_groups=1,
            block_q=32,
            waves_q=2,
            runtime_k_loop=True,
        ),
        16,
    ),
    "gen_wk2_k32": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(64, dt, kpg=32, groups=2),
            block_groups=1,
            block_h=8,
            waves_k=2,
            fold_k32=True,
        ),
        16,
    ),
    "gen_wq2_sb": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(16, dt, groups=2, W=40),
            block_groups=1,
            block_q=32,
            waves_q=2,
            block_h=8,
            double_buffer=False,
        ),
        16,
    ),
    "gen_persistent_wk2": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(32, dt, groups=1, N=1, H=8, W=16),
            block_groups=1,
            block_h=8,
            waves_k=2,
            persistent_grid=True,
        ),
        16,
    ),
    "gen_rtc": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: _rtc(
            DirectConvSpec(problem=_p(24, dt, kpg=40, groups=2), block_groups=1)
        ),
        16,
    ),
    # waves_k == 1 weight preload: the prologue loads every fragment at
    # storage width (fp32: dwordx4 pieces, not the 16-bit vN load).
    "gen_pw": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(16, dt, kpg=32, groups=4),
            block_groups=2,
            block_h=8,
            preload_weights=True,
        ),
        16,
    ),
    # waves_m splits the preloaded M-tiles; stage_out keeps the fp32
    # accumulators unrounded in the LDS output tile.
    "gen_pw_wm2_so": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(16, dt, kpg=32, groups=4),
            block_groups=1,
            block_h=8,
            preload_weights=True,
            waves_m=2,
            stage_out=True,
            lds_pad=8,
        ),
        16,
    ),
    "gen_so": (
        build_direct_conv,
        is_valid_spec,
        lambda dt: DirectConvSpec(
            problem=_p(32, dt, groups=4), block_groups=2, block_h=8, stage_out=True
        ),
        16,
    ),
}

# Physical MFMA intrinsic per (arch, dtype, m).
_PHYS = {
    ("gfx950", "tf32", 16): "f32.16x16x32.bf16",
    ("gfx950", "tf32", 32): "f32.32x32x16.bf16",
    ("gfx942", "tf32", 16): "f32.16x16x8.xf32",
    ("gfx942", "tf32", 32): "f32.32x32x4.xf32",
    ("gfx950", "fp32", 16): "f32.16x16x4f32",
    ("gfx950", "fp32", 32): "f32.32x32x2f32",
    ("gfx942", "fp32", 16): "f32.16x16x4f32",
    ("gfx942", "fp32", 32): "f32.32x32x2f32",
}

# Physical MFMAs per filter row of the fixed-cpg kernels (3x3 filter). The
# gfx950 TF32 counts are BF16 MFMAs (3 per emulated atom): non-fold 16c and 8c
# pair their 4-wide atoms, so 16c costs the same with or without fold_k32.
_PER_ROW = {
    "16c_k32": {("gfx950", "tf32"): 6, ("gfx942", "tf32"): 8, "fp32": 16},
    "16c": {("gfx950", "tf32"): 6, ("gfx942", "tf32"): 6, "fp32": 12},
    "8c": {("gfx950", "tf32"): 3, ("gfx942", "tf32"): 4, "fp32": 8},
    "32c": {("gfx950", "tf32"): 18, ("gfx942", "tf32"): 24, "fp32": 48},
}

_MFMA_CALL = re.compile(r"call [^@]*@llvm\.amdgcn\.mfma\.([\w.]+)\(")


def _mfma_calls(ll: str):
    out = {}
    for name in _MFMA_CALL.findall(ll):
        out[name] = out.get(name, 0) + 1
    return out


class TestDirectTf32Plan(unittest.TestCase):
    """The logical-to-physical MMA plan the kernels rely on."""

    def test_plans(self):
        expect = {
            ("gfx950", "tf32", 4): ("mfma_f32_16x16x32_xf32", 1, 4),
            ("gfx950", "tf32", 8): ("mfma_f32_16x16x32_xf32", 1, 0),
            ("gfx942", "tf32", 4): ("mfma_f32_16x16x8_xf32", 2, 0),
            ("gfx942", "tf32", 8): ("mfma_f32_16x16x8_xf32", 4, 0),
            ("gfx950", "fp32", 4): ("mfma_f32_16x16x4_f32", 4, 0),
            ("gfx942", "fp32", 8): ("mfma_f32_16x16x4_f32", 8, 0),
        }
        for (arch, dt, frag), (op_id, chunks, pad) in expect.items():
            with self.subTest(arch=arch, dtype=dt, frag=frag):
                op, n, p = _f32_mma_plan(arch, dt, 16, frag)
                self.assertEqual((op.op_id, n, p), (op_id, chunks, pad))

    def test_zero_padding_is_symmetric(self):
        # A padded K slot must be zero in A *and* B, or 0 * Inf / 0 * NaN
        # leaks into the sum: the helper pads both operands at the same K and
        # refuses fragments of different widths.
        from rocke.core.ir import F32, IRBuilder

        from kernels.common.conv_direct_grouped import _zero_pad_operands

        b = IRBuilder("pad_probe")
        a4 = b.vec_pack([b.const_f32(1.0)] * 4, F32)
        b4 = b.vec_pack([b.const_f32(2.0)] * 4, F32)
        pa, pb = _zero_pad_operands(b, a4, b4, 4)
        self.assertEqual((pa.type.count, pb.type.count), (8, 8))
        b2 = b.vec_pack([b.const_f32(2.0)] * 2, F32)
        with self.assertRaisesRegex(ValueError, "differ in width"):
            _zero_pad_operands(b, a4, b2, 4)


class TestDirectTf32Build(unittest.TestCase):
    """Every variant builds and lowers for fp32 and tf32 on both arches."""

    def test_forward_variants(self):
        for name, (build, valid, make, m) in _FWD.items():
            for arch in _ARCHES:
                for dt in _F32_DTYPES:
                    with self.subTest(variant=name, arch=arch, dtype=dt):
                        spec = make(dt)
                        ok, why = valid(spec, arch=arch)
                        self.assertTrue(ok, why)
                        kernel = build(spec, arch=arch)
                        flag = "_f32" if dt == "fp32" else "_tf32"
                        self.assertTrue(kernel.name.endswith(flag), kernel.name)
                        ll = lower_kernel_to_llvm(kernel, arch=arch)
                        calls = _mfma_calls(ll)
                        self.assertEqual(set(calls), {_PHYS[(arch, dt, m)]}, calls)
                        # fp32 storage end to end: no 16-bit load, LDS or store.
                        self.assertNotIn("half", ll)
                        per_row = _PER_ROW.get(name)
                        if per_row is not None:
                            n = per_row.get((arch, dt), per_row.get(dt))
                            # Row loop unrolled lcm(2, KH) = 6 times, KH taps.
                            self.assertEqual(sum(calls.values()), 6 * 3 * n)

    def test_dgrad_pipeline(self):
        problem_args = dict(N=2, H=14, W=14, groups=2, cpg=24, kpg=48)
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                with self.subTest(arch=arch, dtype=dt):
                    p = DirectConvProblem(dtype=dt, **problem_args)
                    fprop = make_dgrad_fprop_spec(p, block_groups=1, block_h=8)
                    ok, why = is_valid_spec(fprop, arch=arch)
                    self.assertTrue(ok, why)
                    ll = lower_kernel_to_llvm(
                        build_direct_conv(fprop, arch=arch), arch=arch
                    )
                    self.assertEqual(set(_mfma_calls(ll)), {_PHYS[(arch, dt, 16)]})
                    for kernel in (
                        build_direct_transpose_weights_dgrad(
                            DirectTransposeWeightsDgradSpec(problem=p), arch=arch
                        ),
                        build_direct_transpose_weights_dgrad(
                            DirectTransposeWeightsDgradSpec(
                                problem=p, runtime_channels=True
                            ),
                            arch=arch,
                        ),
                        build_direct_reorganize_weights(
                            DirectReorganizeWeightsSpec(problem=p), arch=arch
                        ),
                        build_direct_reorganize_weights(
                            DirectReorganizeWeightsSpec(
                                problem=replace(p, kpg=64),
                                fold_k32=True,
                                runtime_channels=True,
                            ),
                            arch=arch,
                        ),
                        build_direct_coalesced_weights_dgrad(
                            DirectCoalescedWeightsDgradSpec(problem=p), arch=arch
                        ),
                    ):
                        # A copy: fp32 and tf32 share one binary.
                        self.assertTrue(kernel.name.endswith("_f32"), kernel.name)
                        ll = lower_kernel_to_llvm(kernel, arch=arch)
                        self.assertNotIn("half", ll)
                        self.assertIn("raw.ptr.buffer.load", ll)
                    self.assertEqual(
                        direct_dgrad_workspace_bytes(p),
                        2 * direct_dgrad_workspace_bytes(replace(p, dtype="fp16")),
                    )
                    self.assertEqual(
                        direct_dgrad_coalesced_workspace_bytes(p),
                        2
                        * direct_dgrad_coalesced_workspace_bytes(
                            replace(p, dtype="fp16")
                        ),
                    )


class TestDirectTf32Rejections(unittest.TestCase):
    def test_4c_has_no_fp32_atom(self):
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                with self.subTest(arch=arch, dtype=dt):
                    spec = DirectConv4cSpec(problem=_p(4, dt, groups=16))
                    ok, why = is_valid_spec_4c(spec, arch=arch)
                    self.assertFalse(ok)
                    self.assertIn("no 4x4 TF32/fp32", why)
                    self.assertIn("DirectConvSpec", why)
                    with self.assertRaisesRegex(ValueError, "no 4x4 TF32/fp32"):
                        build_direct_conv_4c(spec, arch=arch)

    def test_fused_dgrad_weights_are_16bit_only(self):
        # The fused-weights fragment is gathered with 16-bit loads (or the
        # 16-bit LDS transpose read): fp32 storage must be refused, not built.
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                for lds in (False, True):
                    with self.subTest(arch=arch, dtype=dt, weights_lds=lds):
                        spec = DirectConvSpec(
                            problem=_p(16, dt),
                            block_h=8,
                            dgrad_fused_weights=True,
                            dgrad_weights_lds=lds,
                        )
                        ok, why = is_valid_spec(spec, arch=arch)
                        self.assertFalse(ok)
                        self.assertIn("dgrad_fused_weights is 16-bit only", why)

    def test_unknown_dtype(self):
        spec = DirectConv16cSpec(problem=_p(16, "fp64"))
        ok, why = is_valid_spec_16c(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("unsupported dtype", why)

    def test_fp32_preload_vgprs_at_storage_width(self):
        # One fp32 element per VGPR: this preload fits at 16-bit storage and
        # needs twice the registers at fp32, past the budget.
        gen = DirectConvSpec(
            problem=_p(32, "fp16", kpg=32, groups=4),
            block_groups=1,
            preload_weights=True,
        )
        self.assertEqual(preload_weight_vgprs(gen), 72)
        self.assertTrue(is_valid_spec(gen, "gfx950")[0])
        for dt in _F32_DTYPES:
            with self.subTest(dtype=dt):
                spec = replace(gen, problem=_p(32, dt, kpg=32, groups=4))
                self.assertEqual(preload_weight_vgprs(spec), 144)
                ok, why = is_valid_spec(spec, arch="gfx950")
                self.assertFalse(ok)
                self.assertIn("weight preload needs 144 VGPRs", why)

    def test_fp32_lds_bytes_at_storage_width(self):
        # Row buffers, the lds_pad column pads and the stage_out tiles all
        # hold storage elements: fp32 doubles every one of them.
        for knobs in (
            {"lds_pad": 8},
            {"stage_out": True},
            {"lds_pad": 8, "stage_out": True, "double_buffer": False},
        ):
            with self.subTest(**knobs):
                gen = DirectConvSpec(
                    problem=_p(32, "fp16", groups=4), block_groups=2, **knobs
                )
                for dt in _F32_DTYPES:
                    spec = replace(gen, problem=_p(32, dt, groups=4))
                    self.assertEqual(
                        direct_conv_lds_bytes(spec), 2 * direct_conv_lds_bytes(gen)
                    )
        # Fits gfx942's 64 KiB at 16-bit storage, not at fp32.
        gen = DirectConvSpec(
            problem=_p(64, "fp16", W=17, H=9, groups=8),
            block_q=64,
            block_groups=2,
            double_buffer=False,
            stage_out=True,
        )
        self.assertTrue(is_valid_spec(gen, "gfx942")[0])
        for dt in _F32_DTYPES:
            with self.subTest(dtype=dt):
                spec = replace(gen, problem=_p(64, dt, W=17, H=9, groups=8))
                ok, why = is_valid_spec(spec, arch="gfx942")
                self.assertFalse(ok)
                self.assertIn("LDS footprint 104448 B", why)

    def test_fp32_lds_budget(self):
        # The fp16 ring of this 32c spec fits; fp32 doubles it past gfx942's LDS.
        spec = DirectConv32cSpec(
            problem=_p(32, "tf32", W=64), block_q=64, block_groups=8
        )
        ok, why = is_valid_spec_32c(spec, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("LDS", why)
        self.assertTrue(is_valid_spec_32c(spec, arch="gfx950")[0])
        gen = DirectConvSpec(problem=_p(64, "fp32"), block_groups=8)
        self.assertTrue(
            is_valid_spec(replace(gen, problem=_p(64, "fp16")), "gfx942")[0]
        )
        ok, why = is_valid_spec(gen, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("LDS budget", why)


class TestDirectTf32Names(unittest.TestCase):
    """fp16 / bf16 names are unchanged; fp32 and tf32 get their own flag."""

    def test_names(self):
        p16 = _p(16, "fp16")
        cases = {
            DirectConv16cSpec(problem=p16).kernel_name(): (
                "direct_conv_16c_N2H14W14_g8_c16k16_bq16_bg8_db_k32"
            ),
            DirectConv16cSpec(problem=_p(16, "bf16")).kernel_name(): (
                "direct_conv_16c_N2H14W14_g8_c16k16_bq16_bg8_db_k32_bf16"
            ),
            DirectConv16cSpec(problem=_p(16, "tf32")).kernel_name(): (
                "direct_conv_16c_N2H14W14_g8_c16k16_bq16_bg8_db_k32_tf32"
            ),
            DirectConv8cSpec(problem=_p(8, "fp16")).kernel_name(): (
                "direct_conv_8c_N2H14W14_g8_c8k8_bq16_bg8_db"
            ),
            DirectConv8cSpec(problem=_p(8, "fp32")).kernel_name(): (
                "direct_conv_8c_N2H14W14_g8_c8k8_bq16_bg8_db_f32"
            ),
            DirectConv32cSpec(problem=_p(32, "bf16")).kernel_name(): (
                "direct_conv_32c_N2H14W14_g8_c32k32_bq32_bg4_db_bf16"
            ),
            DirectConvSpec(problem=_p(32, "fp16")).kernel_name(): (
                "direct_conv_N2H14W14_g8_c32k32_bq16_bg8_db_bh16"
            ),
            DirectConvSpec(problem=_p(32, "tf32")).kernel_name(): (
                "direct_conv_N2H14W14_g8_c32k32_bq16_bg8_db_bh16_tf32"
            ),
            build_direct_transpose_weights_dgrad(
                DirectTransposeWeightsDgradSpec(problem=p16)
            ).name: "direct_transpose_weights_dgrad_N2H14W14_g8_c16k16",
            build_direct_reorganize_weights(
                DirectReorganizeWeightsSpec(problem=_p(16, "bf16"))
            ).name: "direct_reorg_wt_N2H14W14_g8_c16k16",
        }
        for got, want in cases.items():
            self.assertEqual(got, want)


# ---------------------------------------------------------------------------
# ISA: one comgr compile per arch
# ---------------------------------------------------------------------------


def _disassemble(kernel, arch: str):
    """``(instructions, notes)`` with ``instructions = [(addr, mnemonic, text)]``."""
    require_comgr_target(arch)
    isa = f"amdgcn-amd-amdhsa--{arch}"
    objdump = find_llvm_tool("llvm-objdump")
    readelf = find_llvm_tool("llvm-readelf")
    if objdump is None or readelf is None:
        skip_until(BUILD_VARIANT, reason="llvm-objdump / llvm-readelf not found")
    hsaco, _ = comgr.build_hsaco_from_llvm_ir(
        lower_kernel_to_llvm(kernel, arch=arch), isa=isa
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(hsaco)
        dis = subprocess.run(
            [objdump, "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
        notes = subprocess.run(
            [readelf, "--notes", str(path)], capture_output=True, text=True
        ).stdout
    insts = []
    for line in dis.splitlines():
        m = re.match(r"\s+(\S+)\s+(.*?)\s*//\s*([0-9A-Fa-f]+):(.*)", line)
        if m:
            insts.append((int(m.group(3), 16), m.group(1), m.group(2) + m.group(4)))
    labels = {}
    for line in dis.splitlines():
        m = re.match(r"([0-9a-fA-F]+) <(.+)>:", line)
        if m:
            labels[m.group(2)] = int(m.group(1), 16)
    return insts, labels, notes


def _loops(insts, labels):
    """``[(start, end)]`` address ranges closed by a backward branch."""
    out = []
    for addr, mn, text in insts:
        if not mn.startswith(("s_cbranch", "s_branch")):
            continue
        m = re.search(r"<([^+>]+)(?:\+0x([0-9a-fA-F]+))?>", text)
        if m and m.group(1) in labels:
            target = labels[m.group(1)] + int(m.group(2) or "0", 16)
            if target < addr:
                out.append((target, addr))
    return out


class TestDirectTf32Isa(unittest.TestCase):
    def test_8c_isa(self):
        for arch in _ARCHES:
            with self.subTest(arch=arch):
                kernel = build_direct_conv_8c(
                    DirectConv8cSpec(problem=_p(8, "tf32")), arch=arch
                )
                insts, labels, notes = _disassemble(kernel, arch)
                mnems = [mn for _, mn, _ in insts]
                mfmas = {mn for mn in mnems if mn.startswith("v_mfma")}
                scratch = re.search(r"\.private_segment_fixed_size:\s+(\d+)", notes)
                if scratch:
                    self.assertEqual(int(scratch.group(1)), 0, "spills")
                if arch == "gfx942":
                    self.assertEqual(mfmas, {"v_mfma_f32_16x16x8_xf32"})
                    self.assertNotIn("v_cvt_pk_bf16_f32", mnems)
                    continue
                self.assertEqual(mfmas, {"v_mfma_f32_16x16x32_bf16"})
                # 3 BF16 MFMAs per filter row, 6 unrolled rows x 3 taps.
                self.assertEqual(sum(m.startswith("v_mfma") for m in mnems), 54)
                loops = _loops(insts, labels)
                self.assertTrue(loops)

                def inside(a):
                    return any(lo <= a <= hi for lo, hi in loops)

                cvt = [a for a, mn, _ in insts if mn == "v_cvt_pk_bf16_f32"]
                cvt_in = sum(1 for a in cvt if inside(a))
                cvt_out = len(cvt) - cvt_in
                # The weights are loop-invariant: their BF16x3 split is hoisted
                # out of the row loop (cvt_out > 0), and what stays inside is
                # the input split, CSE'd across the KH taps that share it.
                self.assertGreater(cvt_out, 0)
                self.assertLessEqual(cvt_in, 2 * cvt_out)


# ---------------------------------------------------------------------------
# GPU correctness (gfx942 / gfx950)
# ---------------------------------------------------------------------------


def _gpu_arch() -> str:
    from rocke.runtime.hip_module import get_device_arch

    return get_device_arch(0) or ""


_GPU_ARCH = _gpu_arch()


def _gpu_skip_reason() -> str:
    if _GPU_ARCH not in _ARCHES:
        return f"needs a gfx942 or gfx950 GPU (got {_GPU_ARCH!r})"
    return ""


_GPU_SKIP = _gpu_skip_reason()


def _tolerance(dtype: str, arch: str) -> float:
    """Max error relative to the output's peak.

    fp32 is exact fp32 MFMA arithmetic. gfx950 TF32 is the BF16x3 emulation,
    about fp32 accurate, and 1e-4 fails a single bf16 pass. gfx942 TF32 is
    the native XF32 MFMA, checked against masked inputs
    (``conv_reference.tf32_reference_inputs``) at 1e-3, as in test_conv_fwd_correctness.
    """
    if dtype == "tf32":
        return tf32_reference_tol(arch)
    return 2e-5


def _launch_geometry(spec):
    """``(grid, block)`` of ``spec``'s kernel."""
    from kernels.common.conv_direct_grouped import direct_launch_geometry

    if isinstance(spec, (DirectConv16cSpec, DirectConv8cSpec, DirectConv32cSpec)):
        # The fixed-cpg kernels decode (bx, by, bz) = (Q tile, group tile, n);
        # direct_launch_geometry does not cover them.
        p = spec.problem
        grid = (-(-p.Wo // spec.block_q), p.groups // spec.block_groups, p.N)
        return grid, (spec.threads_per_block, 1, 1)
    return direct_launch_geometry(spec)


def _launch_transform(art, sig, values, spec):
    """Launch a dgrad weight-transform kernel and wait for it."""
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    grid, block = _launch_geometry(spec)
    KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)(
        values, config=LaunchConfig(grid=grid, block=block, fence=True)
    )


def _run_fwd(spec, build, arch: str) -> float:
    from rocke import compile_kernel
    from rocke.runtime.device_array import DeviceArray

    p = spec.problem
    art = compile_kernel(build(spec, arch=arch), arch=arch)
    rng = np.random.default_rng(0)
    x = storage.encode(rng.uniform(-1.0, 1.0, (p.N, p.H, p.W, p.total_c)), "fp32")
    w = storage.encode(rng.uniform(-1.0, 1.0, (p.total_k, p.KH, p.KW, p.cpg)), "fp32")
    ref = direct_fwd_reference(
        p,
        tf32_reference_inputs(x, arch, p.dtype),
        tf32_reference_inputs(w, arch, p.dtype),
    )
    # 0x7F bytes: an unwritten output shows up huge.
    y = DeviceArray(poisoned((p.N, p.Ho, p.Wo, p.total_k), "fp32"))
    grid, block = _launch_geometry(spec)
    # fp32 and tf32 share the fp32 storage ABI.
    launch_direct_kernel(
        art, p, (DeviceArray(x), DeviceArray(w), y), grid=grid, block=block
    )
    return peak_relative_error(y.read(), ref, floor=1.0)


def _run_dgrad(problem, fprop, arch: str) -> float:
    """transpose [-> reorganize] -> fprop on fp32 storage, vs the float64 dgrad."""
    from rocke import compile_kernel
    from rocke.runtime.device_array import DeviceArray

    from kernels.common.conv_abi import conv_direct_transform_signature

    p = problem
    rtc = fprop.runtime_channels
    coalesced = fprop.waves_k > 1 or fprop.runtime_k_loop
    xform_sig = conv_direct_transform_signature("fp32", runtime_channels=rtc)
    scalars = {"p_cpg": p.cpg, "p_kpg": p.kpg} if rtc else {}

    t_spec = DirectTransposeWeightsDgradSpec(problem=p, runtime_channels=rtc)
    r_spec = DirectReorganizeWeightsSpec(
        problem=p, fold_k32=fprop.fold_k32, runtime_channels=rtc
    )
    t_art = compile_kernel(build_direct_transpose_weights_dgrad(t_spec), arch=arch)
    r_art = compile_kernel(build_direct_reorganize_weights(r_spec), arch=arch)
    f_art = compile_kernel(build_direct_conv(fprop, arch=arch), arch=arch)

    rng = np.random.default_rng(7)
    dy = storage.encode(rng.uniform(-1.0, 1.0, (p.N, p.Ho, p.Wo, p.total_k)), "fp32")
    w = storage.encode(rng.uniform(-1.0, 1.0, (p.total_k, p.KH, p.KW, p.cpg)), "fp32")
    ref = direct_dgrad_reference(
        p,
        tf32_reference_inputs(dy, arch, p.dtype),
        tf32_reference_inputs(w, arch, p.dtype),
    )

    ws_t = direct_dgrad_workspace_bytes(p)
    ws_coa = direct_dgrad_coalesced_workspace_bytes(p, fold_k32=fprop.fold_k32)
    dy_dev, w_dev = DeviceArray(dy), DeviceArray(w)
    # 0x7F bytes: an unwritten output or workspace element shows up huge.
    dx_dev = DeviceArray(poisoned((p.N, p.H, p.W, p.total_c), "fp32"))
    transposed = DeviceArray(poisoned((ws_t,), "u8"))
    coalesced_ws = DeviceArray(poisoned((ws_coa,), "u8"))
    _launch_transform(
        t_art,
        xform_sig,
        {
            "A": w_dev.data_ptr(),
            "D": transposed.data_ptr(),
            "A_bytes": w_dev.nbytes,
            "D_bytes": ws_t,
            **scalars,
        },
        t_spec,
    )
    weights = transposed
    if coalesced:
        _launch_transform(
            r_art,
            xform_sig,
            {
                "A": transposed.data_ptr(),
                "D": coalesced_ws.data_ptr(),
                "A_bytes": ws_t,
                "D_bytes": ws_coa,
                **scalars,
            },
            r_spec,
        )
        weights = coalesced_ws
    grid, block = _launch_geometry(fprop)
    launch_direct_kernel(
        f_art, fprop.problem, (dy_dev, weights, dx_dev), grid=grid, block=block
    )
    return peak_relative_error(dx_dev.read(), ref, floor=1.0)


# These read B in the reorganized (W_coa) layout the dgrad weight transforms
# write, so they run end to end in _DGRAD_CASES rather than on plain weights.
_COALESCED_ONLY = ("gen_rk", "gen_wk2_k32", "gen_persistent_wk2")

# (label, problem kwargs, make_dgrad_fprop_spec knobs, fold_k32, runtime channels)
_DGRAD_CASES = (
    (
        "plain_loop",
        dict(N=2, H=14, W=14, groups=1, cpg=24, kpg=24),
        dict(block_q=32, block_h=8),
        False,
        False,
    ),
    (
        "plain_loop_rtc",
        dict(N=1, H=9, W=20, groups=3, cpg=20, kpg=28),
        dict(block_q=32, block_h=8),
        False,
        True,
    ),
    (
        "wk4_preload",
        dict(N=1, H=12, W=18, groups=1, cpg=48, kpg=192),
        dict(block_h=8, waves_k=4),
        False,
        False,
    ),
    (
        "wk2_k32",
        dict(N=1, H=12, W=18, groups=1, cpg=48, kpg=192),
        dict(block_h=16, waves_k=2),
        True,
        False,
    ),
    (
        "wq2_rk_rtc",
        dict(N=2, H=10, W=21, groups=2, cpg=20, kpg=40),
        dict(block_q=32, waves_q=2, runtime_k_loop=True),
        False,
        True,
    ),
    (
        "bg2_plain_loop",
        dict(N=2, H=14, W=14, groups=4, cpg=16, kpg=16),
        dict(block_groups=2, block_h=8),
        False,
        False,
    ),
)


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP or "gpu")
class TestDirectTf32Gpu(unittest.TestCase):
    """Every variant, fp32 and tf32, on gfx942 / gfx950 against a float64 reference."""

    def test_forward(self):
        for name, (build, _valid, make, _m) in _FWD.items():
            if name in _COALESCED_ONLY:
                continue
            for dt in _F32_DTYPES:
                with self.subTest(variant=name, dtype=dt):
                    err = _run_fwd(make(dt), build, _GPU_ARCH)
                    print(f"  {name:12s} {dt}  rel_err={err:.2e}", flush=True)
                    self.assertLess(err, _tolerance(dt, _GPU_ARCH))

    def test_mfma_dgrad(self):
        for label, kw, knobs, fold, rtc in _DGRAD_CASES:
            for dt in _F32_DTYPES:
                with self.subTest(case=label, dtype=dt):
                    p = DirectConvProblem(dtype=dt, **kw)
                    fprop = replace(
                        make_dgrad_fprop_spec(p, **{"block_groups": 1, **knobs}),
                        fold_k32=fold,
                    )
                    if rtc:
                        fprop = _rtc(fprop)
                    ok, why = is_valid_spec(fprop, arch=_GPU_ARCH)
                    self.assertTrue(ok, why)
                    err = _run_dgrad(p, fprop, _GPU_ARCH)
                    print(f"  dgrad {label:16s} {dt}  rel_err={err:.2e}", flush=True)
                    self.assertLess(err, _tolerance(dt, _GPU_ARCH))


#: The public backend entry point's kind for each builder it lowers.
_BACKEND_KIND = {build_direct_conv_16c: "16c", build_direct_conv: "generic"}


@unittest.skipIf(
    importlib.util.find_spec("rocke_library_engine") is None,
    skip_until_reason(
        BUILD_VARIANT, reason="rocke_library_engine extension not importable"
    ),
)
class TestDirectF32CrossEngine(unittest.TestCase):
    """The C++ direct builders build the fp32 / tf32 kernels too: every 16c
    and generic variant above lowers to the same ``.ll`` through
    ``lower_conv_direct_grouped`` with ``backend="python"`` and
    ``backend="cpp"`` (named on each call, so the lane the suite runs in
    does not matter). The 8c and 32c kernels have no backend entry point;
    the ``conv_direct_grouped`` byte-identity family compares them."""

    def test_python_and_cpp_builders_agree(self):
        from rocke.core.backend import BACKEND_CPP, BACKEND_PYTHON

        from kernels.conv.engine_bridge import lower_conv_direct_grouped

        for name, (build, _valid, make, m) in _FWD.items():
            kind = _BACKEND_KIND.get(build)
            if kind is None:
                continue
            for arch in _ARCHES:
                for dt in _F32_DTYPES:
                    with self.subTest(variant=name, arch=arch, dtype=dt):
                        spec = make(dt)
                        py = lower_conv_direct_grouped(
                            spec, kind=kind, arch=arch, backend="python"
                        )
                        cpp = lower_conv_direct_grouped(
                            spec, kind=kind, arch=arch, backend="cpp"
                        )
                        self.assertEqual(
                            (py.backend, cpp.backend), (BACKEND_PYTHON, BACKEND_CPP)
                        )
                        self.assertEqual(
                            set(_mfma_calls(cpp.llvm_text)), {_PHYS[(arch, dt, m)]}
                        )
                        self.assertEqual(cpp.llvm_text, py.llvm_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
