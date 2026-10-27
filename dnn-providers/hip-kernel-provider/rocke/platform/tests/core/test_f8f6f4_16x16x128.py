# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The K=128 f8f6f4 MFMA: the MX op and the unscaled fp8 hero atom.

Both ops call ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4``. The MX op passes
real E8M0 scales. The hero atom passes literal 0 for both scales, and the
backend then selects the unscaled instruction. LLVM 20, 22, 23 and 24 accept the
same nine-argument, mangled form, so the text does not depend on the flavor.

The MX op takes ``a_format`` and ``b_format``. A and B must use the same format
in this version. Every operand has 256 bits per lane for every format.

The unscaled op ``mfma_f32_16x16x128_fp4fp6fp8`` takes the same two arguments. It
always passes literal 0 for both scales, so the backend selects the unscaled
instruction for every format.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python

_NAME = "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32"
_DECLARATION = (
    f"declare <4 x float> @{_NAME}("
    "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
    "i32 immarg, i32, i32 immarg, i32)"
)
# The hero call passes literal 0 for cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b.
_HERO_CALL_TAIL = "i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)"
# The MX call passes cbsz, blgp, opsel_a, then scale_a, opsel_b, scale_b.
_MX_CALL_TAIL = "i32 0, i32 0, i32 0, i32 {sa}, i32 0, i32 {sb})"
# The cbsz and blgp value of each data type. A and B use one format in this version.
_FORMATS = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
_ONE = {  # the encoding of +1.0
    "fp8e4m3": 0x38,
    "bf8e5m2": 0x3C,
    "fp6e2m3": 0x08,
    "fp6e3m2": 0x0C,
    "fp4e2m1": 0x2,
}
_MX_CALL_TAIL_FMT = "i32 {n}, i32 {n}, i32 0, i32 {sa}, i32 0, i32 {sb})"


def _kernel(kind: str, scales=(127, 127), fmt=None) -> KernelDef:
    """A one-wave kernel: C = A * B with K=128. A and B hold 8 dwords per lane."""
    kw = {} if fmt is None else {"a_format": fmt, "b_format": fmt}
    b = IRBuilder(f"f8f6f4_{kind}")
    b.kernel.attrs["max_workgroup_size"] = 64
    pa = b.param("A", PtrType(I32, "global"), align=32)
    pb = b.param("B", PtrType(I32, "global"), align=32)
    pc = b.param("C", PtrType(F32, "global"), align=16)
    tid = b.thread_id_x()
    off8 = b.mul(tid, b.const_i32(8))
    a = b.global_load_vN(pa, off8, I32, 8, align=32)
    bb = b.global_load_vN(pb, off8, I32, 8, align=32)
    acc = b.vector_splat(b.const_f32(0.0), 4)
    if kind == "hero":
        r = b.mma("mfma_f32_16x16x128_fp8", a, bb, acc)
    elif kind == "mx":
        r = b.mfma_scale_f32_16x16x128_f8f6f4(
            a, bb, acc, b.const_i32(scales[0]), b.const_i32(scales[1]), **kw
        )
    elif kind == "both":
        r1 = b.mma("mfma_f32_16x16x128_fp8", a, bb, acc)
        r = b.mfma_scale_f32_16x16x128_f8f6f4(
            a, bb, r1, b.const_i32(scales[0]), b.const_i32(scales[1]), **kw
        )
    elif kind == "unscaled":
        r = b.mfma_f32_16x16x128_fp4fp6fp8(a, bb, acc, **kw)
    elif kind == "all":
        r1 = b.mma("mfma_f32_16x16x128_fp8", a, bb, acc)
        r2 = b.mfma_scale_f32_16x16x128_f8f6f4(
            a, bb, r1, b.const_i32(scales[0]), b.const_i32(scales[1]), **kw
        )
        r = b.mfma_f32_16x16x128_fp4fp6fp8(a, bb, r2, **kw)
    else:
        raise ValueError(kind)
    b.global_store_vN(pc, b.mul(tid, b.const_i32(4)), r, 4, align=16)
    b.ret()
    return b.kernel


def _raw_kernel(
    a_format: str, b_format: str, dwords: int = 8, op: str = "mx"
) -> KernelDef:
    """An MX or unscaled op built through the generic ``mma`` with raw attributes.

    The generic path skips the checks of the MX builder method. Serialized IR
    reaches both engines the same way, so the engines must check on their own.
    """
    b = IRBuilder("f8f6f4_raw")
    b.kernel.attrs["max_workgroup_size"] = 64
    a = b.vector_splat(b.const_i32(1), dwords)
    acc = b.vector_splat(b.const_f32(0.0), 4)
    attrs = {"a_format": a_format, "b_format": b_format}
    if op == "unscaled":
        b.mma("mfma_f32_16x16x128_fp4fp6fp8", a, a, acc, attrs=attrs)
    else:
        b.mma(
            "mfma_scale_f32_16x16x128_f8f6f4",
            a,
            a,
            acc,
            b.const_i32(127),
            b.const_i32(127),
            attrs=attrs,
        )
    b.ret()
    return b.kernel


def _lane_bytes(fmt: str) -> bytes:
    """One lane's operand: 32 K values of +1.0. The unused bytes hold 0xFF."""
    one = _ONE[fmt]
    if fmt in ("fp8e4m3", "bf8e5m2"):
        return bytes([one]) * 32
    if fmt in ("fp6e2m3", "fp6e3m2"):
        packed = sum(one << (6 * i) for i in range(32)).to_bytes(24, "little")
        return packed + bytes([0xFF]) * 8
    return bytes([one | (one << 4)]) * 16 + bytes([0xFF]) * 16


def _lower(kernel: KernelDef, flavor: str) -> str:
    return _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)


def _llvm_tool(name: str):
    """Find an LLVM tool: ROCKE_LLVM_BIN, PATH, then ROCm installs (newest first)."""
    exe = name + (".exe" if os.name == "nt" else "")
    roots = []
    if os.environ.get("ROCKE_LLVM_BIN"):
        roots.append(Path(os.environ["ROCKE_LLVM_BIN"]))
    for var in ("ROCM_PATH", "ROCM_HOME"):
        if os.environ.get(var):
            roots += [
                Path(os.environ[var]) / "llvm" / "bin",
                Path(os.environ[var]) / "lib" / "llvm" / "bin",
            ]
    for base in sorted(Path("/opt").glob("rocm*"), reverse=True):
        roots += [base / "llvm" / "bin", base / "lib" / "llvm" / "bin"]
    for root in roots:
        cand = root / exe
        if cand.is_file():
            return str(cand)
    return shutil.which(name)


def _call_lines(ll: str):
    return [ln for ln in ll.splitlines() if f"@{_NAME}(" in ln and " = call " in ln]


def _declare_lines(ll: str):
    return [
        ln for ln in ll.splitlines() if ln.startswith("declare ") and "f8f6f4" in ln
    ]


class TestEmittedText(unittest.TestCase):
    def test_hero_text_is_the_same_at_every_flavor(self):
        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                ll = _lower(_kernel("hero"), flavor)
                self.assertEqual(_declare_lines(ll), [_DECLARATION])
                (call,) = _call_lines(ll)
                self.assertTrue(call.endswith(_HERO_CALL_TAIL), call)

    def test_mx_text_is_the_same_at_every_flavor(self):
        tail = _MX_CALL_TAIL.format(sa=127, sb=127)
        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                ll = _lower(_kernel("mx"), flavor)
                self.assertEqual(_declare_lines(ll), [_DECLARATION])
                (call,) = _call_lines(ll)
                self.assertTrue(call.endswith(tail), call)

    def test_both_ops_in_one_kernel_share_one_declaration(self):
        for flavor in LLVM_FLAVORS:
            with self.subTest(flavor=flavor):
                ll = _lower(_kernel("both"), flavor)
                self.assertEqual(_declare_lines(ll), [_DECLARATION])
                self.assertEqual(len(_call_lines(ll)), 2)


class TestFormats(unittest.TestCase):
    def test_the_default_formats_are_fp8e4m3_and_add_no_attribute(self):
        default = _kernel("mx")
        explicit = _kernel("mx", fmt="fp8e4m3")
        self.assertEqual(_lower(default, "llvm23"), _lower(explicit, "llvm23"))
        self.assertNotIn("a_format", str(serialize(default)))
        self.assertNotIn("a_format", str(serialize(explicit)))

    def test_the_call_carries_the_format_selectors(self):
        for fmt, n in _FORMATS.items():
            for flavor in LLVM_FLAVORS:
                with self.subTest(fmt=fmt, flavor=flavor):
                    ll = _lower(_kernel("mx", fmt=fmt), flavor)
                    self.assertEqual(_declare_lines(ll), [_DECLARATION])
                    (call,) = _call_lines(ll)
                    tail = _MX_CALL_TAIL_FMT.format(n=n, sa=127, sb=127)
                    self.assertTrue(call.endswith(tail), call)

    def test_the_builder_rejects_different_formats(self):
        with self.assertRaisesRegex(ValueError, "same format"):
            _kernel_mixed("fp8e4m3", "fp4e2m1")
        with self.assertRaisesRegex(ValueError, "same format"):
            _kernel_mixed("fp8e4m3", "bf8e5m2")  # the same width is not enough

    def test_the_builder_rejects_an_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "unknown a_format 'fp8'"):
            _kernel_mixed("fp8", "fp8")

    def test_the_builder_rejects_an_operand_that_is_not_256_bits(self):
        b = IRBuilder("narrow")
        a = b.vector_splat(b.const_i32(1), 4)
        acc = b.vector_splat(b.const_f32(0.0), 4)
        with self.assertRaisesRegex(ValueError, "256 bits per lane"):
            b.mfma_scale_f32_16x16x128_f8f6f4(
                a,
                a,
                acc,
                b.const_i32(127),
                b.const_i32(127),
                a_format="fp4e2m1",
                b_format="fp4e2m1",
            )

    def test_the_python_engine_checks_serialized_attributes(self):
        for args, pattern in (
            (("fp8e4m3", "fp4e2m1"), "same format"),
            (("fp8", "fp8"), "unknown a_format"),
            (("fp6e2m3", "fp6e2m3", 6), "256 bits per lane"),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, pattern):
                _lower(_raw_kernel(*args), "llvm23")

    def test_the_atom_wrapper_passes_the_formats_through(self):
        from rocke.helpers.atoms import MfmaAtom

        b = IRBuilder("atom")
        a = b.vector_splat(b.const_i32(1), 8)
        acc = b.vector_splat(b.const_f32(0.0), 4)
        MfmaAtom.fp8_16x16x128().emit_scaled(
            b,
            a,
            a,
            acc,
            b.const_i32(127),
            b.const_i32(127),
            a_format="fp4e2m1",
            b_format="fp4e2m1",
        )
        b.ret()
        (call,) = _call_lines(_lower(b.kernel, "llvm23"))
        self.assertTrue(call.endswith(_MX_CALL_TAIL_FMT.format(n=4, sa=127, sb=127)))

    def test_the_hip_lowering_does_not_fake_the_mx_op(self):
        from rocke.core.lower_hip import lower_kernel_to_hip

        with self.assertRaises(NotImplementedError):
            lower_kernel_to_hip(_kernel("mx"), arch="gfx950")


class TestUnscaledOp(unittest.TestCase):
    """The op ``mfma_f32_16x16x128_fp4fp6fp8`` for every format."""

    def test_the_call_passes_literal_zero_scales_and_the_selectors(self):
        for fmt, n in _FORMATS.items():
            for flavor in LLVM_FLAVORS:
                with self.subTest(fmt=fmt, flavor=flavor):
                    ll = _lower(_kernel("unscaled", fmt=fmt), flavor)
                    self.assertEqual(_declare_lines(ll), [_DECLARATION])
                    (call,) = _call_lines(ll)
                    self.assertTrue(
                        call.endswith(f"i32 {n}, i32 {n}, i32 0, i32 0, i32 0, i32 0)"),
                        call,
                    )

    def test_all_three_ops_in_one_kernel_share_one_declaration(self):
        for fmt in _FORMATS:
            with self.subTest(fmt=fmt):
                ll = _lower(_kernel("all", fmt=fmt), "llvm23")
                self.assertEqual(_declare_lines(ll), [_DECLARATION])
                self.assertEqual(len(_call_lines(ll)), 3)

    def test_the_builder_rejects_different_formats_and_bad_input(self):
        def build(a_format, b_format, dwords=8):
            b = IRBuilder("bad")
            a = b.vector_splat(b.const_i32(1), dwords)
            acc = b.vector_splat(b.const_f32(0.0), 4)
            b.mfma_f32_16x16x128_fp4fp6fp8(
                a, a, acc, a_format=a_format, b_format=b_format
            )

        with self.assertRaisesRegex(ValueError, "same format"):
            build("fp8e4m3", "fp4e2m1")
        with self.assertRaisesRegex(ValueError, "unknown a_format 'fp8'"):
            build("fp8", "fp8")
        with self.assertRaisesRegex(ValueError, "256 bits per lane"):
            build("fp4e2m1", "fp4e2m1", dwords=4)

    def test_the_python_engine_checks_serialized_attributes(self):
        for args, pattern in (
            (("fp8e4m3", "fp4e2m1"), "same format"),
            (("fp8", "fp8"), "unknown a_format"),
            (("fp6e2m3", "fp6e2m3", 6), "256 bits per lane"),
        ):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, pattern):
                _lower(_raw_kernel(*args, op="unscaled"), "llvm23")

    def test_default_formats_add_no_attribute(self):
        self.assertNotIn("a_format", str(serialize(_kernel("unscaled"))))
        self.assertIn("a_format", str(serialize(_kernel("unscaled", fmt="fp4e2m1"))))

    def test_the_hip_lowering_does_not_fake_the_op(self):
        from rocke.core.lower_hip import lower_kernel_to_hip

        with self.assertRaises(NotImplementedError):
            lower_kernel_to_hip(_kernel("unscaled"), arch="gfx950")


def _kernel_mixed(a_format: str, b_format: str) -> KernelDef:
    b = IRBuilder("mixed")
    a = b.vector_splat(b.const_i32(1), 8)
    acc = b.vector_splat(b.const_f32(0.0), 4)
    b.mfma_scale_f32_16x16x128_f8f6f4(
        a,
        a,
        acc,
        b.const_i32(127),
        b.const_i32(127),
        a_format=a_format,
        b_format=b_format,
    )
    b.ret()
    return b.kernel


class TestEngineIdentity(unittest.TestCase):
    def test_python_and_cpp_emit_the_same_bytes(self):
        try:
            _lower_via_cpp_engine(serialize(_kernel("hero")), "gfx950", "llvm20")
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")
        cases = [(kind, None) for kind in ("hero", "mx", "both")]
        cases += [("mx", fmt) for fmt in _FORMATS]
        cases += [("unscaled", fmt) for fmt in _FORMATS]
        cases += [("all", "fp6e3m2")]
        for kind, fmt in cases:
            kernel = _kernel(kind, fmt=fmt)
            serialized = serialize(kernel)
            for flavor in LLVM_FLAVORS:
                with self.subTest(kind=kind, fmt=fmt, flavor=flavor):
                    self.assertEqual(
                        _lower_via_cpp_engine(serialized, "gfx950", flavor),
                        _lower(kernel, flavor),
                    )

    def test_the_cpp_engine_rejects_what_the_python_engine_rejects(self):
        try:
            _lower_via_cpp_engine(serialize(_kernel("hero")), "gfx950", "llvm23")
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")
        for args, text in (
            (("fp8e4m3", "fp4e2m1"), "same format"),
            (("fp8", "fp8"), "unknown a_format"),
            (("fp6e2m3", "fp6e2m3", 6), "256 bits per lane"),
        ):
            for op in ("mx", "unscaled"):
                with (
                    self.subTest(args=args, op=op),
                    self.assertRaises(RuntimeError) as ctx,
                ):
                    _lower_via_cpp_engine(
                        serialize(_raw_kernel(*args, op=op)), "gfx950", "llvm23"
                    )
                self.assertIn(text, str(ctx.exception))


class TestToolchain(unittest.TestCase):
    """Tests that need an LLVM toolchain. They skip when no tool exists."""

    def test_verifier_accepts_every_kernel(self):
        tool = _llvm_tool("llvm-as") or _llvm_tool("opt")
        if tool is None:
            self.skipTest("no llvm-as or opt found")
        cases = [(kind, None) for kind in ("hero", "mx", "both")]
        cases += [("mx", fmt) for fmt in _FORMATS]
        cases += [("unscaled", fmt) for fmt in _FORMATS]
        cases += [("all", "fp6e3m2")]
        for kind, fmt in cases:
            for flavor in LLVM_FLAVORS:
                with (
                    self.subTest(kind=kind, fmt=fmt, flavor=flavor),
                    tempfile.TemporaryDirectory() as td,
                ):
                    src = Path(td) / "k.ll"
                    src.write_text(_lower(_kernel(kind, fmt=fmt), flavor))
                    cmd = (
                        [tool, str(src), "-o", str(Path(td) / "k.bc")]
                        if Path(tool).stem == "llvm-as"
                        else [tool, "-passes=verify", "-disable-output", str(src)]
                    )
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)

    def _isa(self, kind: str, td: str, scales=(127, 127), fmt=None) -> str:
        clang = _llvm_tool("clang")
        if clang is None:
            self.skipTest("no clang found")
        src = Path(td) / "k.ll"
        src.write_text(_lower(_kernel(kind, scales, fmt), "llvm23"))
        proc = subprocess.run(
            [
                clang,
                "-x",
                "ir",
                "-O0",
                "-nogpulib",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcpu=gfx950",
                "-S",
                "-o",
                "-",
                str(src),
            ],
            capture_output=True,
            text=True,
            cwd=td,
            check=False,
        )
        if proc.returncode != 0:
            self.skipTest(
                f"clang cannot compile for gfx950: {proc.stderr.strip()[:120]}"
            )
        return proc.stdout

    def test_hero_selects_the_unscaled_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            asm = self._isa("hero", td)
        self.assertIn("v_mfma_f32_16x16x128_f8f6f4", asm)
        self.assertNotIn("v_mfma_scale_", asm)
        self.assertNotIn("v_mfma_ld_scale_", asm)

    def test_mx_selects_the_scaled_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            asm = self._isa("mx", td)
        self.assertIn("v_mfma_scale_f32_16x16x128_f8f6f4", asm)

    def test_literal_zero_scales_on_the_mx_op_select_the_unscaled_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            asm = self._isa("mx", td, scales=(0, 0))
        self.assertIn("v_mfma_f32_16x16x128_f8f6f4", asm)
        self.assertNotIn("v_mfma_scale_", asm)

    def test_the_instruction_carries_the_format_selectors(self):
        for fmt, n in _FORMATS.items():
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as td:
                asm = self._isa("mx", td, fmt=fmt)
            line = next(
                ln
                for ln in asm.splitlines()
                if "v_mfma_scale_f32_16x16x128_f8f6f4" in ln
            )
            if n == 0:
                self.assertNotIn("cbsz:", line)
                self.assertNotIn("blgp:", line)
            else:
                self.assertIn(f"cbsz:{n}", line)
                self.assertIn(f"blgp:{n}", line)

    def test_the_unscaled_op_selects_the_unscaled_instruction_for_every_format(self):
        for fmt, n in _FORMATS.items():
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as td:
                asm = self._isa("unscaled", td, fmt=fmt)
            self.assertNotIn("v_mfma_scale_", asm)
            self.assertNotIn("v_mfma_ld_scale_", asm)
            line = next(
                ln for ln in asm.splitlines() if "v_mfma_f32_16x16x128_f8f6f4" in ln
            )
            if n == 0:
                self.assertNotIn("cbsz:", line)
            else:
                self.assertIn(f"cbsz:{n}", line)
                self.assertIn(f"blgp:{n}", line)

    def test_one_literal_zero_scale_keeps_the_scaled_instruction(self):
        with tempfile.TemporaryDirectory() as td:
            asm = self._isa("mx", td, scales=(0, 127))
        self.assertIn("v_mfma_scale_f32_16x16x128_f8f6f4", asm)


def _gfx950_device() -> bool:
    try:
        from rocke.runtime import hip_module as hip

        return hip.get_device_count() > 0 and hip.get_device_arch(0) == "gfx950"
    except Exception:  # noqa: BLE001 - any failure means "no usable device"
        return False


class TestGfx950Numerics(unittest.TestCase):
    """A = B = 1.0 in the format under test and K = 128, so an unscaled result is 128.

    Each lane holds 8 dwords. fp6 and fp4 data fill the low dwords, and the
    unused bytes hold 0xFF, so a result of 128 shows that the instruction
    ignores them.
    """

    @classmethod
    def setUpClass(cls):
        if not _gfx950_device():
            raise unittest.SkipTest("no gfx950 device")
        if _llvm_tool("clang") is None:
            raise unittest.SkipTest("no clang found")

    def _run(self, kind: str, scales=(127, 127), fmt=None, data_fmt=None) -> float:
        from rocke.runtime.hip_module import Runtime

        rt = Runtime()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "k.ll"
            src.write_text(_lower(_kernel(kind, scales, fmt), "llvm23"))
            co = Path(td) / "k.co"
            proc = subprocess.run(
                [
                    _llvm_tool("clang"),
                    "-x",
                    "ir",
                    "-O3",
                    "-nogpulib",
                    "-target",
                    "amdgcn-amd-amdhsa",
                    "-mcpu=gfx950",
                    "-o",
                    str(co),
                    str(src),
                ],
                capture_output=True,
                text=True,
                cwd=td,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            mod = rt.load_module(co.read_bytes())
            fn = mod.get_function(f"f8f6f4_{kind}")
            ones = _lane_bytes(data_fmt or fmt or "fp8e4m3") * 64
            pa, pb, pc = rt.alloc(len(ones)), rt.alloc(len(ones)), rt.alloc(64 * 16)
            for p in (pa, pb):
                rt.memcpy_h2d(
                    p, (ctypes.c_ubyte * len(ones)).from_buffer_copy(ones), len(ones)
                )
            rt.memset(pc, 0xFF, 64 * 16)
            rt.launch(fn, (1, 1, 1), (64, 1, 1), struct.pack("<QQQ", pa, pb, pc))
            rt.sync()
            out = (ctypes.c_ubyte * (64 * 16))()
            rt.memcpy_d2h(out, pc, 64 * 16)
            vals = struct.unpack("<256f", bytes(out))
            for p in (pa, pb, pc):
                rt.free(p)
            mod.unload()
        self.assertEqual(min(vals), max(vals), "lanes disagree")
        return vals[0]

    def test_every_format_gives_128(self):
        for fmt in _FORMATS:
            with self.subTest(fmt=fmt):
                self.assertEqual(self._run("mx", (127, 127), fmt), 128.0)

    def test_the_unscaled_op_gives_128_for_every_format(self):
        for fmt in _FORMATS:
            with self.subTest(fmt=fmt):
                self.assertEqual(self._run("unscaled", fmt=fmt), 128.0)

    def test_the_scale_applies_to_fp4_and_fp6(self):
        for fmt in ("fp4e2m1", "fp6e2m3"):
            with self.subTest(fmt=fmt):
                self.assertEqual(self._run("mx", (128, 127), fmt), 256.0)

    def test_a_wrong_format_selector_changes_the_result(self):
        # fp8e4m3 1.0 (0x38) read as bf8e5m2 is 0.5, so 0.5 * 0.5 * 128 = 32.
        self.assertEqual(self._run("mx", (127, 127), "bf8e5m2", "fp8e4m3"), 32.0)

    def test_hero_gives_128(self):
        self.assertEqual(self._run("hero"), 128.0)

    def test_mx_with_unit_scales_gives_128(self):
        self.assertEqual(self._run("mx", (127, 127)), 128.0)

    def test_mx_applies_the_e8m0_scale(self):
        self.assertEqual(self._run("mx", (128, 127)), 256.0)
        self.assertEqual(self._run("mx", (126, 127)), 64.0)

    def test_a_literal_zero_scale_is_not_the_neutral_scale(self):
        # One literal 0 keeps the scaled instruction; e = 0 means 2^-127.
        self.assertAlmostEqual(self._run("mx", (0, 127)) / 2.0**-120, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
