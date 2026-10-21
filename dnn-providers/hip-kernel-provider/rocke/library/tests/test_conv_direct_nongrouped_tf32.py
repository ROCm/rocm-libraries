# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""fp32 storage and TF32 compute in the non-grouped (``groups == 1``) direct conv.

``DirectConvProblem.dtype`` ``"tf32"`` is fp32 tensors with TF32 MFMA operands
(gfx942: native XF32 atoms; gfx950: the BF16x3 emulation); ``"fp32"`` is fp32
tensors with the exact fp32 atoms. The kernel keeps its 16-bit addressing in
elements and issues each logical ``tile x tile`` atom through
``conv_direct_grouped._mma_f32_logical``.

Host-side (no GPU): spec validation and rejections, kernel names, LDS sizing,
the sweep knobs, the register estimate, built + lowered IR across atoms,
dilation, padding and stride on gfx950 and gfx942, the AOT shape invariance,
and -- for fp16 / bf16 -- that the Python IR still matches the C++ engine
byte for byte. One comgr compile per (arch, dtype, atom) checks the ISA.

On silicon (gfx942 / gfx950 + torch): output against an IEEE fp32
``conv2d`` reference computed on the CPU, relative to the output peak.

Run GPU tests as a script if pytest reports "No HIP GPUs are available"::

    python library/tests/test_conv_direct_nongrouped_tf32.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from kernels.common.conv_direct_grouped import DirectConvProblem  # noqa: E402
from kernels.common.conv_direct_nongrouped import (  # noqa: E402
    _ATOMS,
    DirectNongroupedConvSpec,
    _sweep_atoms,
    build_direct_conv_nongrouped,
    default_lds_pad,
    is_valid_nongrouped_spec,
    grouped_knobs,
    nongrouped_knobs,
    nongrouped_live_regs,
    nongrouped_specs,
    nongrouped_split_knobs,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402

_F32_DTYPES = ("fp32", "tf32")
_ARCHES = ("gfx950", "gfx942")

# Physical MFMA intrinsic each (arch, dtype, atom tile) issues, and how many of
# them one logical atom with an 8- / 4-element fragment costs.
#   gfx950 tf32: emulated 32x32x16 / 16x16x32 xf32 = 3 bf16 MFMAs per atom; a
#                4-element fragment is zero-padded to that atom.
#   gfx942 tf32: native 32x32x4 / 16x16x8 xf32, 2 elements per atom.
#   fp32:        exact 32x32x2 / 16x16x4 f32, 1 element per atom.
_PHYS = {
    ("gfx950", "tf32", 32): ("llvm.amdgcn.mfma.f32.32x32x16.bf16", {8: 3, 4: 3}),
    ("gfx950", "tf32", 16): ("llvm.amdgcn.mfma.f32.16x16x32.bf16", {8: 3, 4: 3}),
    ("gfx942", "tf32", 32): ("llvm.amdgcn.mfma.f32.32x32x4.xf32", {8: 4, 4: 2}),
    ("gfx942", "tf32", 16): ("llvm.amdgcn.mfma.f32.16x16x8.xf32", {8: 4, 4: 2}),
    ("gfx950", "fp32", 32): ("llvm.amdgcn.mfma.f32.32x32x2f32", {8: 8, 4: 4}),
    ("gfx950", "fp32", 16): ("llvm.amdgcn.mfma.f32.16x16x4f32", {8: 8, 4: 4}),
    ("gfx942", "fp32", 32): ("llvm.amdgcn.mfma.f32.32x32x2f32", {8: 8, 4: 4}),
    ("gfx942", "fp32", 16): ("llvm.amdgcn.mfma.f32.16x16x4f32", {8: 8, 4: 4}),
}


def _problem(dtype="tf32", **kw) -> DirectConvProblem:
    base = dict(N=1, H=16, W=32, groups=1, cpg=64, kpg=64, dtype=dtype)
    base.update(kw)
    return DirectConvProblem(**base)


def _spec(dtype="tf32", atom="32x32x16", problem=None, **kw):
    """A small spec that fits both arches' LDS for every atom in fp32."""
    t, k, _ = _ATOMS[atom]
    ck = max(16, k)
    base = dict(
        tile_h=4,
        tile_w=t,
        tile_k=64 if ck == 16 else 32,
        ck=ck,
        waves_m=2,
        waves_n=2,
        atom=atom,
        lds_pad=4,
    )
    base.update(kw)
    return DirectNongroupedConvSpec(problem=problem or _problem(dtype), **base)


def _n_mfma_calls(ll: str):
    """``{intrinsic: call count}`` in lowered LLVM IR."""
    out = {}
    for m in re.finditer(r"call [^@]*@(llvm\.amdgcn\.mfma\.[A-Za-z0-9_.]+)\(", ll):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def _logical_atoms(spec) -> int:
    """Logical MFMAs per channel chunk (the loop body is emitted once)."""
    p = spec.problem
    return spec.m_tiles_per_wave * spec.n_tiles_per_wave * spec.k_atoms * p.KH * p.KW


# ---------------------------------------------------------------------------
# Spec: validation, rejections, names, sizing, knobs
# ---------------------------------------------------------------------------


class TestNongroupedF32Spec(unittest.TestCase):
    def test_fp32_and_tf32_validate_on_both_arches(self):
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                for atom in _ATOMS:
                    with self.subTest(arch=arch, dtype=dt, atom=atom):
                        ok, why = is_valid_nongrouped_spec(_spec(dt, atom), arch)
                        self.assertTrue(ok, why)

    def test_unknown_dtype_rejected(self):
        ok, why = is_valid_nongrouped_spec(_spec("fp8"), "gfx950")
        self.assertFalse(ok)
        self.assertIn("unsupported dtype", why)

    def test_pad_alignment_is_16_bytes(self):
        # 4 floats is 16 bytes: legal for fp32, misaligned for fp16 (4 halves).
        self.assertTrue(is_valid_nongrouped_spec(_spec("tf32", lds_pad=4))[0])
        self.assertTrue(is_valid_nongrouped_spec(_spec("fp32", lds_pad=12))[0])
        for dt, pad in (("tf32", 2), ("fp32", 6), ("fp16", 4), ("bf16", 12)):
            with self.subTest(dtype=dt, pad=pad):
                ok, why = is_valid_nongrouped_spec(_spec(dt, lds_pad=pad))
                self.assertFalse(ok)
                self.assertIn("lds_pad", why)

    def test_default_pad(self):
        self.assertEqual(default_lds_pad("fp16"), 8)
        self.assertEqual(default_lds_pad("bf16"), 8)
        self.assertEqual(default_lds_pad("fp32"), 4)
        self.assertEqual(default_lds_pad("tf32"), 4)

    def test_tensor_bound_counts_four_byte_elements(self):
        # A is 2**30 elements: 2 GiB as fp16 is past the masked-access offset
        # either way; at 2**29 elements fp16 (1 GiB) fits and fp32 (2 GiB) not.
        p16 = _problem("fp16", N=2**29 // (16 * 32 * 64))
        self.assertTrue(
            is_valid_nongrouped_spec(_spec("fp16", problem=p16, lds_pad=8))[0]
        )
        ok, why = is_valid_nongrouped_spec(
            _spec("tf32", problem=replace(p16, dtype="tf32"))
        )
        self.assertFalse(ok)
        self.assertIn("masked-access", why)

    def test_kernel_names(self):
        base = _spec("fp16", lds_pad=8).kernel_name()
        self.assertFalse(base.endswith(("bf16", "fp32", "tf32")))
        self.assertEqual(_spec("bf16", lds_pad=8).kernel_name(), base + "_bf16")
        self.assertEqual(_spec("tf32").kernel_name(), base + "_tf32")
        self.assertEqual(_spec("fp32").kernel_name(), base + "_fp32")
        # A non-default pad is named for fp32 storage (16-bit names never were).
        self.assertEqual(_spec("tf32", lds_pad=8).kernel_name(), base + "_p8_tf32")
        self.assertEqual(_spec("fp16", lds_pad=16).kernel_name(), base)

    def test_lds_bytes_doubles_with_element_size(self):
        s16 = _spec("fp16", lds_pad=8)
        s32 = _spec("tf32", lds_pad=4)
        # Same geometry, 16-byte pad and staging vector either way.
        x16 = s16.lds_x_halves + 8 + s16.lds_w_halves + s16.frag
        x32 = s32.lds_x_halves + 4 + s32.lds_w_halves + s32.frag
        self.assertEqual(s16.lds_bytes, 2 * x16)
        self.assertEqual(s32.lds_bytes, 4 * x32)
        self.assertEqual(replace(s32, double_buffer=True).lds_bytes, 8 * x32)

    def test_sweep_prefers_wide_fragment_atoms(self):
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                with self.subTest(arch=arch, dtype=dt):
                    self.assertEqual(_sweep_atoms(arch, dt), ["32x32x16", "16x16x32"])
        # 16-bit is unchanged.
        self.assertEqual(_sweep_atoms("gfx950", "bf16"), ["32x32x16", "16x16x32"])
        self.assertEqual(_sweep_atoms("gfx942", "fp16"), ["32x32x8", "16x16x16"])

    def test_knobs(self):
        for arch in _ARCHES:
            f32 = list(nongrouped_knobs(arch, "tf32"))
            self.assertTrue(f32)
            self.assertEqual({k["lds_pad"] for k in f32}, {4})
            self.assertEqual({k["ck"] for k in f32}, {16, 32})
            f16 = list(nongrouped_knobs(arch, "fp16"))
            self.assertTrue(all("lds_pad" not in k for k in f16))
            self.assertEqual({k["ck"] for k in f16}, {16, 32, 64})

    def test_specs_valid_and_unique(self):
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                p = _problem(dt, N=2, W=48, kpg=128)
                specs = nongrouped_specs(p, arch=arch)
                with self.subTest(arch=arch, dtype=dt):
                    self.assertGreater(len(specs), 0)
                    names = [s.kernel_name() for s in specs]
                    self.assertEqual(len(names), len(set(names)))
                    for s in specs:
                        self.assertTrue(is_valid_nongrouped_spec(s, arch)[0])
                        self.assertTrue(s.kernel_name().endswith(dt))
                        self.assertEqual(s.lds_pad, 4)

    def test_live_regs(self):
        s16 = _spec("bf16", lds_pad=8)
        s32 = _spec("fp32")
        acc = s16.acc_vgprs
        # Fragments and weight prefetch double; the dwordx4 X prefetch does not.
        self.assertGreater(nongrouped_live_regs(s32), nongrouped_live_regs(s16))
        frag_part16 = nongrouped_live_regs(s16) - acc
        self.assertLess(nongrouped_live_regs(s32) - acc, 2 * frag_part16 + 1)
        # BF16x3 split temporaries only where TF32 is emulated (gfx950).
        t = _spec("tf32")
        self.assertEqual(nongrouped_live_regs(t, "gfx942"), nongrouped_live_regs(s32))
        self.assertGreater(nongrouped_live_regs(t, "gfx950"), nongrouped_live_regs(t))


# ---------------------------------------------------------------------------
# Built and lowered IR
# ---------------------------------------------------------------------------

_GEOMETRY = (
    ("plain", {}, {}),
    ("dil2x3_pad2", dict(PAD=2, dil_h=2, dil_w=3), {}),
    ("stride2", dict(stride=2, H=32, W=64), {}),
    ("pointwise", dict(KH=1, KW=1, PAD=0), {}),
    ("5x5_partial", dict(KH=5, KW=5, PAD=2, W=40, kpg=96), dict(tile_k=32, waves_m=1)),
    ("db_iglp", {}, dict(double_buffer=True, iglp=0, waves_per_eu=3)),
    (
        "no_swizzle_s2_dil",
        dict(stride=2, dil_h=3, H=29, W=33),
        dict(chiplet_swizzle=False),
    ),
)


class TestNongroupedF32Build(unittest.TestCase):
    def test_build_and_lower_matrix(self):
        for arch in _ARCHES:
            for dt in _F32_DTYPES:
                for atom in _ATOMS:
                    for gname, pkw, skw in _GEOMETRY:
                        spec = _spec(dt, atom, problem=_problem(dt, **pkw), **skw)
                        ok, why = is_valid_nongrouped_spec(spec, arch)
                        # Wide halos (5x5, dilation, stride 2) on gfx942's
                        # 64 KiB: narrow the channel tile until fp32 fits.
                        while not ok and "LDS" in why and spec.m_tiles_total > 1:
                            spec = replace(spec, tile_k=spec.tile_k // 2, waves_m=1)
                            ok, why = is_valid_nongrouped_spec(spec, arch)
                        while not ok and "LDS" in why and spec.tile_h > 2:
                            spec = replace(spec, tile_h=spec.tile_h // 2)
                            ok, why = is_valid_nongrouped_spec(spec, arch)
                        with self.subTest(arch=arch, dtype=dt, atom=atom, g=gname):
                            if not ok and arch == "gfx942" and gname == "5x5_partial":
                                # 25 fp32 taps of a 32-channel tile at ck=16
                                # do not fit 64 KiB: a real limit, not a bug.
                                self.assertIn("LDS", why)
                                continue
                            self.assertTrue(ok, why)
                            ll = lower_kernel_to_llvm(
                                build_direct_conv_nongrouped(spec, arch=arch),
                                arch=arch,
                            )
                            t, _, frag = _ATOMS[atom]
                            intr, per = _PHYS[(arch, dt, t)]
                            calls = _n_mfma_calls(ll)
                            self.assertEqual(set(calls), {intr})
                            self.assertEqual(
                                calls[intr], _logical_atoms(spec) * per[frag]
                            )
                            # fp32 storage only: no 16-bit tensor traffic.
                            self.assertNotIn("half", ll)
                            if dt == "fp32":
                                self.assertNotIn("bfloat", ll)

    def test_emitted_ir_is_shape_independent(self):
        """AOT: N/H/W/C/K are kernargs, so they do not reach the IR."""
        for dt in _F32_DTYPES:
            a = _spec(dt, problem=_problem(dt, N=1, H=16, W=32, cpg=64, kpg=64))
            b = replace(a, problem=_problem(dt, N=3, H=40, W=24, cpg=128, kpg=192))
            with self.subTest(dtype=dt):
                # Only the kernel name (which spells the probe shape) differs.
                self.assertEqual(
                    lower_kernel_to_llvm(
                        build_direct_conv_nongrouped(a), arch="gfx950"
                    ).replace(a.kernel_name(), "K"),
                    lower_kernel_to_llvm(
                        build_direct_conv_nongrouped(b), arch="gfx950"
                    ).replace(b.kernel_name(), "K"),
                )

    def test_fp32_staging_and_store_are_dwordx4(self):
        for atom in _ATOMS:
            ll = lower_kernel_to_llvm(
                build_direct_conv_nongrouped(_spec("tf32", atom)), arch="gfx950"
            )
            # Every LDS vector access is one 16-byte piece, aligned 16 -- all
            # the 16-byte pixel stride guarantees.
            lds = re.findall(
                r"(?:load|store) (?:volatile )?<(\d+) x float>[^\n]*addrspace\(3\)"
                r"[^\n]*align (\d+)",
                ll,
            )
            with self.subTest(atom=atom):
                self.assertTrue(lds)
                for n, align in lds:
                    self.assertEqual((int(n), int(align)), (4, 16))

    def test_16bit_python_matches_cpp_engine(self):
        """fp16 / bf16 emission is unchanged: the C++ port (not touched by the
        fp32 work) still produces the same LLVM IR bytes."""
        from rocke.core import backend

        try:
            backend._import_engine()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"rocke_engine unavailable: {exc}")
        for arch, atoms in (
            ("gfx950", ("32x32x16", "16x16x32")),
            ("gfx942", ("32x32x8", "16x16x16")),
        ):
            for dt in ("fp16", "bf16"):
                for atom in atoms:
                    for gname, pkw, skw in _GEOMETRY:
                        t, k, _ = _ATOMS[atom]
                        knobs = dict(
                            tile_h=8,
                            tile_w=2 * t,
                            tile_k=64,
                            ck=16 if k <= 16 else 32,
                            waves_m=2,
                            waves_n=2,
                            atom=atom,
                        )
                        knobs.update(skw)
                        spec = DirectNongroupedConvSpec(
                            problem=_problem(dt, **pkw), **knobs
                        )
                        if not is_valid_nongrouped_spec(spec, arch)[0]:
                            continue
                        with self.subTest(arch=arch, dtype=dt, atom=atom, g=gname):
                            # "both" raises when the two engines disagree.
                            backend.lower_conv_direct_nongrouped(
                                spec, arch=arch, backend="both"
                            )


# ---------------------------------------------------------------------------
# ISA: one comgr compile per (arch, dtype, atom tile)
# ---------------------------------------------------------------------------


def _llvm_tool(name: str):
    candidate = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin" / name
    return str(candidate) if candidate.exists() else shutil.which(name)


def _compile(ir: str, arch: str) -> bytes:
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


class TestNongroupedF32Isa(unittest.TestCase):
    """gfx950 tf32 -> 3x v_mfma_*_bf16 per atom + v_cvt_pk_bf16_f32; gfx942 tf32
    -> v_mfma_*_xf32; fp32 -> v_mfma_*_f32. 128-bit LDS and global traffic, no
    scratch."""

    _ISA = {
        ("gfx950", "tf32", 32): ("v_mfma_f32_32x32x16_bf16", 3),
        ("gfx950", "tf32", 16): ("v_mfma_f32_16x16x32_bf16", 3),
        ("gfx942", "tf32", 32): ("v_mfma_f32_32x32x4_xf32", 4),
        ("gfx942", "tf32", 16): ("v_mfma_f32_16x16x8_xf32", 4),
        ("gfx950", "fp32", 32): ("v_mfma_f32_32x32x2_f32", 8),
        ("gfx942", "fp32", 32): ("v_mfma_f32_32x32x2_f32", 8),
    }

    def test_isa(self):
        for (arch, dt, t), (mnem, per) in self._ISA.items():
            atom = "32x32x16" if t == 32 else "16x16x32"
            # One M tile per wave and ck = atom K keep the 16-wide compile quick.
            spec = _spec(
                dt,
                atom,
                tile_k=32 if t == 16 else 64,
                waves_m=2 if t == 32 else 1,
                ck=_ATOMS[atom][1],
                iglp=0,
            )
            with self.subTest(arch=arch, dtype=dt, atom=atom):
                ok, why = is_valid_nongrouped_spec(spec, arch)
                self.assertTrue(ok, why)
                ll = lower_kernel_to_llvm(
                    build_direct_conv_nongrouped(spec, arch=arch), arch=arch
                )
                seen, notes = _disassemble(_compile(ll, arch))
                mfmas = [s for s in seen if s.startswith("v_mfma")]
                self.assertEqual(set(mfmas), {mnem})
                self.assertEqual(len(mfmas), per * _logical_atoms(spec))
                if (arch, dt) == ("gfx950", "tf32"):
                    self.assertIn("v_cvt_pk_bf16_f32", seen)
                else:
                    self.assertNotIn("v_cvt_pk_bf16_f32", seen)
                self.assertIn("ds_read_b128", seen)
                self.assertIn("ds_write_b128", seen)
                self.assertIn("buffer_store_dwordx4", seen)
                self.assertNotIn("ds_read_b64", seen)
                scratch = re.search(r"\.private_segment_fixed_size:\s+(\d+)", notes)
                if scratch:
                    self.assertEqual(int(scratch.group(1)), 0, "kernel spills")


# ---------------------------------------------------------------------------
# On silicon
# ---------------------------------------------------------------------------

GPU_ARCH = get_device_arch(0)


def _gpu_skip_reason() -> str:
    if not GPU_ARCH:
        return "no ROCm GPU detected"
    if not _HAS_TORCH:
        return "torch not importable"
    if GPU_ARCH not in _ARCHES:
        return f"unsupported arch {GPU_ARCH!r}"
    return ""


_GPU_SKIP = _gpu_skip_reason()

# Max |error| relative to the output peak, against an IEEE fp32 reference.
# BF16x3 tracks the full fp32 payload (gfx950); native XF32 drops 13 mantissa
# bits of every operand (gfx942).
_TOL = {
    ("fp32", "gfx950"): 1e-5,
    ("fp32", "gfx942"): 1e-5,
    ("tf32", "gfx950"): 1e-4,
    ("tf32", "gfx942"): 5e-3,
}


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _reference(A_t, B_t, p):
    """IEEE fp32 conv2d on the CPU, NHWK."""
    import torch.nn.functional as F

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = F.conv2d(
        A_t.permute(0, 3, 1, 2),
        B_t.permute(0, 3, 1, 2),
        padding=p.PAD,
        stride=p.stride,
        dilation=(p.dil_h, p.dil_w),
    )
    return out.permute(0, 2, 3, 1).contiguous()


def run_f32_spec(arch: str, spec, artifact=None):
    """Compile, launch and check one fp32-storage spec; ``(ok, rel_err)``.

    The launch path is the one ``test_direct_conv_correctness`` uses for the
    16-bit kernels (``conv_direct_args_signature`` + ``ConvArgs``), with fp32
    tensors and an fp32 reference.
    """
    from rocke import compile_kernel
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = spec.problem
    ok, why = is_valid_nongrouped_spec(spec, arch=arch)
    if not ok:
        raise AssertionError(f"invalid spec {spec.kernel_name()}: {why}")
    if artifact is None:
        artifact = compile_kernel(
            build_direct_conv_nongrouped(spec, arch=arch), arch=arch
        )

    g = torch.Generator().manual_seed(0)
    A_t = torch.empty(p.N, p.H, p.W, p.cpg).uniform_(-1.0, 1.0, generator=g)
    B_t = torch.empty(p.kpg, p.KH, p.KW, p.cpg).uniform_(-1.0, 1.0, generator=g)
    D_t = torch.empty(p.N, p.Ho, p.Wo, p.kpg)
    ref = _reference(A_t, B_t, p)

    rt = Runtime()
    bufs = [rt.alloc(t.nbytes) for t in (A_t, B_t, D_t)]
    try:
        rt.memcpy_h2d(bufs[0], _u8(A_t), A_t.nbytes)
        rt.memcpy_h2d(bufs[1], _u8(B_t), B_t.nbytes)
        # NaN fill: an output the kernel never writes cannot pass as zero.
        D_t.fill_(float("nan"))
        rt.memcpy_h2d(bufs[2], _u8(D_t), D_t.nbytes)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            # fp32 pointers for both "fp32" and "tf32": tf32 is fp32 storage.
            signature=conv_direct_args_signature("fp32"),
        )
        launcher(
            ConvArgs.from_problem(p).to_launch_values(
                int(bufs[0]),
                int(bufs[1]),
                int(bufs[2]),
                A_t.nbytes,
                B_t.nbytes,
                D_t.nbytes,
            ),
            config=LaunchConfig(
                grid=spec.grid(), block=(spec.threads_per_block, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(D_t)
        rt.memcpy_d2h(_u8(out), bufs[2], D_t.nbytes)
    finally:
        for buf in bufs:
            rt.free(buf)
        synchronize_and_release(0)
    if not torch.isfinite(out).all():
        return False, float("inf")
    rel = float((out - ref).abs().max() / ref.abs().max().clamp(min=1.0))
    return rel < _TOL[(p.dtype, arch)], rel


def _ng_cases():
    """The 16-bit suite's shapes (test_direct_conv_correctness._NG_CASES)."""
    from tests.test_direct_conv_correctness import _NG_CASES

    return _NG_CASES


def _case_spec(case, dtype: str, arch: str):
    p = DirectConvProblem(
        N=case.N,
        H=case.H,
        W=case.W,
        groups=1,
        cpg=case.C,
        kpg=case.K,
        KH=case.KH,
        KW=case.KW,
        PAD=case.PAD,
        stride=case.stride,
        dtype=dtype,
        dil_h=case.dil_h,
        dil_w=case.dil_w,
    )
    spec = DirectNongroupedConvSpec(
        problem=p,
        tile_h=case.tile_h,
        tile_w=case.tile_w,
        tile_k=case.tile_k,
        ck=case.ck,
        waves_m=case.waves_m,
        waves_n=case.waves_n,
        atom=case.atom,
        double_buffer=case.double_buffer,
        iglp=case.iglp,
        waves_per_eu=case.waves_per_eu,
        chiplet_swizzle=case.chiplet_swizzle,
        swizzle_wgm=case.swizzle_wgm,
        lds_pad=default_lds_pad(dtype),
    )
    # fp32 doubles the staged bytes: halve ck, then tile_h, until LDS fits.
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    while not target.fits_lds(spec.lds_bytes) and spec.ck // 2 >= spec.atom_k:
        spec = replace(spec, ck=spec.ck // 2)
    while (
        not target.fits_lds(spec.lds_bytes)
        and spec.tile_h > spec.waves_n
        and (spec.tile_h // 2) % spec.waves_n == 0
    ):
        spec = replace(spec, tile_h=spec.tile_h // 2)
    return spec


@unittest.skipUnless(not _GPU_SKIP, _GPU_SKIP or "no GPU")
class TestNongroupedF32Gpu(unittest.TestCase):
    def test_cases(self):
        for dt in _F32_DTYPES:
            for case in _ng_cases():
                spec = _case_spec(case, dt, GPU_ARCH)
                with self.subTest(dtype=dt, case=case.id):
                    ok, rel = run_f32_spec(GPU_ARCH, spec)
                    print(
                        f"  {'PASS' if ok else 'FAIL'} {dt} {case.id} rel={rel:.2e}",
                        flush=True,
                    )
                    self.assertTrue(ok, f"{case.id} {dt}: rel_err {rel:.3e}")

    def test_sweep_candidates(self):
        for dt in _F32_DTYPES:
            p = _problem(dt, N=2, W=48, kpg=128)
            specs = nongrouped_specs(p, arch=GPU_ARCH)
            self.assertGreater(len(specs), 0)
            for spec in specs[:: max(1, len(specs) // 6)]:
                with self.subTest(spec=spec.kernel_name()):
                    ok, rel = run_f32_spec(GPU_ARCH, spec)
                    print(
                        f"  {'PASS' if ok else 'FAIL'} {spec.kernel_name()} "
                        f"rel={rel:.2e}",
                        flush=True,
                    )
                    self.assertTrue(ok, f"rel_err {rel:.3e}")

    def test_one_binary_serves_every_shape(self):
        from rocke import compile_kernel

        for dt in _F32_DTYPES:
            spec = _spec(dt, problem=_problem(dt), iglp=0)
            art = compile_kernel(
                build_direct_conv_nongrouped(spec, arch=GPU_ARCH), arch=GPU_ARCH
            )
            for shape in [
                (1, 16, 32, 64, 64),
                (3, 20, 40, 128, 96),
                (2, 33, 17, 32, 192),
            ]:
                N, H, W, C, K = shape
                case = replace(spec, problem=_problem(dt, N=N, H=H, W=W, cpg=C, kpg=K))
                with self.subTest(dtype=dt, shape=shape):
                    ok, rel = run_f32_spec(GPU_ARCH, case, artifact=art)
                    self.assertTrue(ok, f"rel_err {rel:.3e}")


class TestF32PlainKernelOnly(unittest.TestCase):
    """fp32 storage is wired through the plain kernel only: the split-C and
    grouped-halo variants stage and reduce 16-bit data."""

    def test_split_c_and_grouped_reject_f32(self):
        for dtype in ("fp32", "tf32"):
            with self.assertRaisesRegex(ValueError, "plain kernel only"):
                _spec(dtype=dtype, split_c=2).validate()
            grouped = _problem(dtype=dtype, groups=4, cpg=16, kpg=16)
            with self.assertRaisesRegex(ValueError, "plain kernel only"):
                _spec(dtype=dtype, problem=grouped, grouped=True).validate()

    def test_split_and_grouped_knobs_skip_f32(self):
        for arch in ("gfx950", "gfx942"):
            for dtype in ("fp32", "tf32"):
                self.assertEqual(list(nongrouped_split_knobs(arch, dtype)), [])
                self.assertEqual(list(grouped_knobs(arch, dtype)), [])
            # 16-bit keeps both grids.
            self.assertTrue(list(nongrouped_split_knobs(arch, "fp16")))
            self.assertTrue(list(grouped_knobs(arch, "fp16")))


if __name__ == "__main__":
    unittest.main()
