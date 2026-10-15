# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lowering gate for the emulated TF32 BF16x3 MMA atoms.

An emulated ``tile.mma`` takes logical TF32 operands, like a native XF32 atom, and
expands before ISA dispatch into a Dekker split of the fp32 payload each carries
plus three BF16 MMAs threading one fp32 accumulator. What is pinned here is
everything a numerics reviewer would want to re-derive by hand:

* the payload is viewed as fp32 by a bitcast -- never a rounding;
* the split is ``big = cap(rne_bf16(x))``, ``small = rne_bf16(x - fp32(big))``,
  where ``cap`` holds ``big`` at the largest finite bf16 so that a payload that
  rounds past it still splits into finite parts, and the residual is taken from
  the payload itself;
* the three terms are ``small*big``, ``big*small``, ``big*big`` **in that
  order** -- fp32 accumulation is not associative, so the order is part of the
  answer, not a detail of how it is written;
* ``small*small`` is not computed (its magnitude is the same order as the
  split's own representation error);
* the physical atom is the catalog's ``physical_op_id``, and the final term --
  only the final term -- lands in the caller's SSA value.

and, separately, that both engines emit the same bytes for all of it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rocke.core.arch import ArchTarget
from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import F32, TF32, IRBuilder, KernelDef, VectorType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _flavor_for_llvm, _lower_kernel_to_llvm_python

_ARCH = "gfx950"
_FLAVORS = ("llvm20", "llvm22", "llvm23")

# Every emulated atom the catalog ships, paired with the bf16 atom it expands
# to. Kept as literals rather than read back out of the catalog: this file is
# the place a reviewer checks the catalog against, so deriving it from the
# catalog would make the test agree with itself.
_EMULATED = {
    "mfma_f32_16x16x32_xf32": "mfma_f32_16x16x32_bf16",
    "mfma_f32_32x32x16_xf32": "mfma_f32_32x32x16_bf16",
}


def _mma_kernel(op_id: str, *, bare: bool = False) -> KernelDef:
    """A kernel whose whole body is one ``tile.mma`` on the named atom.

    ``bare`` hands ``IRBuilder.mma`` the op_id string instead of the ``MmaOp``.
    """
    atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
    # The kernel name deliberately avoids the string "xf32" so the "nothing
    # downstream sees an xf32 atom" check cannot pass or fail on the name.
    builder = IRBuilder("emulated_mma_smoke")
    builder.kernel.attrs["max_workgroup_size"] = atom.wave_size
    a = builder.vector_splat(builder.const_f32(1.0), atom.a_frag_len)
    b = builder.vector_splat(builder.const_f32(2.0), atom.b_frag_len)
    if not atom.is_native:
        # An emulated atom takes the native atoms' ABI: fp32 payload bits wrapped
        # as logical TF32, not an fp32 vector.
        a = builder.bitcast(a, VectorType(TF32, atom.a_frag_len))
        b = builder.bitcast(b, VectorType(TF32, atom.b_frag_len))
    c = builder.vector_splat(builder.const_f32(0.0), atom.c_frag_len)
    builder.mma(op_id if bare else atom, a, b, c)
    builder.ret()
    return builder.kernel


def _lanes(value: int, n: int) -> str:
    """The ``<n x i16>`` constant the lowering spells for a splat of ``value``."""
    return "<" + ", ".join(f"i16 {value}" for _ in range(n)) + ">"


def _name(line: str) -> str:
    """The SSA name a line defines."""
    return line.split(" =")[0]


def _operand_split(body: list[str], side: str) -> dict[str, str]:
    """One operand's expansion, as the eleven lines it emits, by role.

    ``cast`` views the I32 carrier as fp32; ``rne`` .. ``sat`` build the capped
    leading part; ``big`` .. ``small`` are the split proper.
    """
    roles = (
        "cast",
        "rne",
        "bits",
        "sign",
        "cap",
        "over",
        "sat",
        "big",
        "big_f32",
        "diff",
        "small",
    )
    lines = [ln.strip() for ln in body if f"%xf32_{side}_" in ln.split("=")[0]]
    assert len(lines) == len(
        roles
    ), f"operand {side}: expected {len(roles)} lines, got {len(lines)}:\n" + "\n".join(
        lines
    )
    return dict(zip(roles, lines))


def _body(llvm: str) -> list[str]:
    """The lines inside ``define ... {``, which is where emission is pinned."""
    lines = llvm.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("define "))
    return [ln for ln in lines[start + 1 :] if not ln.startswith("}")]


class TestBf16x3Expansion(unittest.TestCase):
    """The recipe itself, read back out of the emitted IR."""

    def _lower(self, op_id: str) -> tuple[str, list[str]]:
        llvm = _lower_kernel_to_llvm_python(
            _mma_kernel(op_id), arch=_ARCH, llvm_flavor="llvm20"
        )
        return llvm, _body(llvm)

    def test_each_operand_is_split_into_big_and_small(self):
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
                n = atom.a_frag_len
                f32v, bf16v, i16v = (
                    f"<{n} x float>",
                    f"<{n} x bfloat>",
                    f"<{n} x i16>",
                )
                _, body = self._lower(op_id)
                for side in ("a", "b"):
                    ln = _operand_split(body, side)
                    # The I32 carrier is viewed as fp32 by a bitcast -- the whole
                    # difference from a native XF32 atom, which ignores the low 13
                    # payload bits; here every one of them reaches the split.
                    src = _name(ln["cast"])
                    self.assertRegex(
                        ln["cast"],
                        rf"^{re.escape(src)} = bitcast <{n} x i32> %\S+ to {re.escape(f32v)}$",
                    )
                    # big starts as rne_bf16(payload) ...
                    self.assertEqual(
                        ln["rne"],
                        f"{_name(ln['rne'])} = fptrunc {f32v} {src} to {bf16v}",
                    )
                    # ... is capped on its bit pattern (pinned in detail by
                    # TestTopBinadeSaturation) and handed back as a bf16,
                    self.assertEqual(
                        ln["bits"],
                        f"{_name(ln['bits'])} = bitcast {bf16v} {_name(ln['rne'])} to {i16v}",
                    )
                    self.assertEqual(
                        ln["big"],
                        f"{_name(ln['big'])} = bitcast {i16v} {_name(ln['sat'])} to {bf16v}",
                    )
                    # ... widened back so the rounding error can be named,
                    self.assertEqual(
                        ln["big_f32"],
                        f"{_name(ln['big_f32'])} = fpext {bf16v} {_name(ln['big'])} to {f32v}",
                    )
                    # ... which is exactly what the *payload* lost -- taken against
                    # the payload and not a capped copy, so a payload the cap
                    # clipped leaves its remainder in small, and a non-finite one
                    # leaves a non-finite small,
                    self.assertEqual(
                        ln["diff"],
                        f"{_name(ln['diff'])} = fsub {f32v} {src}, {_name(ln['big_f32'])}",
                    )
                    # ... itself rounded to bf16.
                    self.assertEqual(
                        ln["small"],
                        f"{_name(ln['small'])} = fptrunc {f32v} {_name(ln['diff'])} to {bf16v}",
                    )

    def test_rounding_is_round_to_nearest_even_not_truncation(self):
        """Every narrowing is an ``fptrunc``, which LLVM defines as RNE.

        The AMDGPU backend selects the packed ``v_cvt_pk_bf16_f32`` for the
        vector form on gfx950 -- CK's shipping packed path. CK's *scalar*
        fallback truncates instead, and is deliberately not the parity target:
        an RNE split is strictly the more accurate of the two.

        The cap added after the first narrowing works on that result's bit
        pattern, so it could not have replaced the rounding with a mask: the
        only integer arithmetic in the expansion is on ``i16`` lanes, never on
        the payload.
        """
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
                i16v = f"<{atom.a_frag_len} x i16>"
                _, body = self._lower(op_id)
                narrowing = [ln for ln in body if "fptrunc" in ln]
                self.assertEqual(
                    len(narrowing), 4, "big and small, for each of the two operands"
                )
                for line in narrowing:
                    # No intrinsic call standing in for the cast.
                    self.assertNotIn("call ", line)
                bit_ops = [ln for ln in body if re.search(r" = (and|or) ", ln)]
                self.assertEqual(len(bit_ops), 4, "sign and cap, per operand")
                for line in bit_ops:
                    self.assertIn(i16v, line)

    def test_three_terms_in_order_threading_one_accumulator(self):
        for op_id, physical in _EMULATED.items():
            with self.subTest(op_id=op_id):
                llvm, body = self._lower(op_id)
                intrinsic = "@llvm.amdgcn." + physical.replace(
                    "mfma_", "mfma."
                ).replace("_", ".")
                calls = [ln.strip() for ln in body if intrinsic in ln]
                self.assertEqual(len(calls), 3, "BF16x3 issues exactly three MMAs")

                def operands(call: str, intrinsic: str = intrinsic) -> list[str]:
                    args = call.split(intrinsic + "(")[1]
                    return re.findall(r"(%[\w.]+)", args)

                names = [call.split(" =")[0] for call in calls]
                a_big, a_small, b_big, b_small = (
                    next(
                        ln.strip().split(" =")[0] for ln in body if f"%xf32_{s}." in ln
                    )
                    for s in ("a_big", "a_small", "b_big", "b_small")
                )
                seed = operands(calls[0])[2]
                self.assertEqual(operands(calls[0])[:3], [a_small, b_big, seed])
                self.assertEqual(operands(calls[1])[:3], [a_big, b_small, names[0]])
                self.assertEqual(operands(calls[2])[:3], [a_big, b_big, names[1]])

    def test_small_times_small_is_not_computed(self):
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                _, body = self._lower(op_id)
                a_small, b_small = (
                    next(
                        ln.strip().split(" =")[0] for ln in body if f"%xf32_{s}." in ln
                    )
                    for s in ("a_small", "b_small")
                )
                for line in body:
                    if "mfma" not in line:
                        continue
                    args = line.split("(", 1)[1]
                    self.assertFalse(
                        a_small in args and b_small in args,
                        "small*small is deliberately dropped",
                    )

    def test_no_emulated_op_id_survives_into_the_emitted_ir(self):
        """The expansion is complete: nothing downstream sees an xf32 atom."""
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                llvm, _ = self._lower(op_id)
                self.assertNotIn(op_id, llvm)
                self.assertNotIn("xf32", llvm.replace("%xf32_", ""))

    def test_only_the_last_term_lands_in_the_callers_value(self):
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                _, body = self._lower(op_id)
                calls = [ln.strip() for ln in body if "mfma" in ln and "call" in ln]
                names = [call.split(" =")[0] for call in calls]
                self.assertTrue(
                    all(n.startswith("%xf32_acc.") for n in names[:2]),
                    "the first two terms thread through temporaries",
                )
                self.assertFalse(
                    names[2].startswith("%xf32_acc."),
                    "the last term writes the tile.mma result the caller named",
                )


class TestTopBinadeSaturation(unittest.TestCase):
    """The cap on ``big``, read back out of the emitted IR.

    bf16 round-to-nearest-even overflows to infinity for any fp32 payload of
    magnitude 0x7f7f8000 or more -- the top 2^-9 of the finite range, where TF32's
    largest finite values are. Uncapped, ``big`` is +-inf there and ``small`` the
    opposite infinity, and the ``inf * 0`` in the cross terms makes a finite
    product NaN. What the numbers do is checked on the host model
    (``test_xf32_bf16x3_numerics.py``) and on a device
    (``bf16x3_parity.py``); this pins that the IR is built the way those assume.
    """

    def _split(self, op_id: str, side: str):
        atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
        llvm = _lower_kernel_to_llvm_python(
            _mma_kernel(op_id), arch=_ARCH, llvm_flavor="llvm20"
        )
        return atom.a_frag_len, _operand_split(_body(llvm), side)

    def test_the_cap_is_the_largest_finite_bf16_carrying_the_values_sign(self):
        for op_id in _EMULATED:
            for side in ("a", "b"):
                with self.subTest(op_id=op_id, side=side):
                    n, ln = self._split(op_id, side)
                    i16v = f"<{n} x i16>"
                    bits = _name(ln["bits"])
                    # The sign bit, as the signed i16 literal for 0x8000 ...
                    self.assertEqual(
                        ln["sign"],
                        f"{_name(ln['sign'])} = and {i16v} {bits}, {_lanes(-0x8000, n)}",
                    )
                    # ... joined to 0x7f7f, the largest finite bf16 magnitude
                    # (0x7f80 is the infinity, so nothing finite is above it).
                    self.assertEqual(
                        ln["cap"],
                        f"{_name(ln['cap'])} = or {i16v} {_name(ln['sign'])}, {_lanes(0x7F7F, n)}",
                    )

    def test_only_what_is_past_the_cap_is_replaced(self):
        """An *unsigned* compare of the pattern against its own signed cap.

        Unsigned, because a negative value's pattern has the sign bit set and
        would otherwise compare below every positive one; and against a cap that
        carries the value's own sign, so +-inf and every NaN (0x7f80 and up in
        magnitude) are caught for both signs and no finite value is."""
        for op_id in _EMULATED:
            for side in ("a", "b"):
                with self.subTest(op_id=op_id, side=side):
                    n, ln = self._split(op_id, side)
                    i16v = f"<{n} x i16>"
                    bits, cap = _name(ln["bits"]), _name(ln["cap"])
                    self.assertEqual(
                        ln["over"],
                        f"{_name(ln['over'])} = icmp ugt {i16v} {bits}, {cap}",
                    )
                    self.assertEqual(
                        ln["sat"],
                        f"{_name(ln['sat'])} = select <{n} x i1> {_name(ln['over'])}, "
                        f"{i16v} {cap}, {i16v} {bits}",
                    )

    def test_the_capped_value_is_what_the_mmas_consume_as_big(self):
        """The MMAs must read the capped ``big``, not the raw rounding."""
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                llvm = _lower_kernel_to_llvm_python(
                    _mma_kernel(op_id), arch=_ARCH, llvm_flavor="llvm20"
                )
                body = _body(llvm)
                mma_args = " ".join(ln for ln in body if "mfma" in ln and "call" in ln)
                for side in ("a", "b"):
                    ln = _operand_split(body, side)
                    self.assertIn(_name(ln["big"]) + ",", mma_args)
                    self.assertNotIn(_name(ln["rne"]), mma_args)


# A power of two, so its own split is exact and it can sit opposite any payload.
_SMALLEST_NORMAL = 0x00800000


def _opt() -> str | None:
    """LLVM's ``opt``, which ROCm installs outside the default PATH."""
    found = shutil.which("opt")
    if found:
        return found
    candidate = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm" / "bin" / "opt"
    return str(candidate) if candidate.exists() else None


def _fold_kernel(op_id: str, a_bits: int, b_bits: int) -> KernelDef:
    """One emulated ``tile.mma`` over constant payloads."""
    atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)

    def payload(bits: int) -> float:
        return float(np.array([bits], dtype=np.uint32).view(np.float32)[0])

    builder = IRBuilder("fold_probe")
    builder.kernel.attrs["max_workgroup_size"] = atom.wave_size
    a = builder.vector_splat(builder.const_f32(payload(a_bits)), atom.a_frag_len)
    b = builder.vector_splat(builder.const_f32(payload(b_bits)), atom.b_frag_len)
    a = builder.bitcast(a, VectorType(TF32, atom.a_frag_len))
    b = builder.bitcast(b, VectorType(TF32, atom.b_frag_len))
    c = builder.vector_splat(builder.const_f32(0.0), atom.c_frag_len)
    builder.mma(atom, a, b, c)
    builder.ret()
    return builder.kernel


# The split parts are handed to this opaque external function as ``i16`` lanes, so
# that what LLVM folds them to is read back as plain integers. A bf16 constant is
# *printed* differently from one LLVM release to the next (hex, then decimal with
# ``+inf`` and ``+qnan`` spellings), and a test that parsed those spellings broke on
# the first toolchain that changed them; an integer is printed the same way in all.
_PROBE_SINK = "rocke_probe_sink"


def _with_probes(llvm: str) -> str:
    """``llvm`` with the four parts the first two MMAs consume routed to a sink.

    The parts are read from those MMA calls in the *unfolded* IR -- ``small*big``
    then ``big*small`` -- and each is bitcast to ``i16`` lanes and passed to
    ``@rocke_probe_sink`` just before the kernel returns, in the order
    ``a_small, b_big, a_big, b_small``. The sink is external, so no pass may drop
    or reorder the calls, and everything feeding them is folded.
    """
    calls = [
        ln for ln in llvm.splitlines() if "@llvm.amdgcn.mfma" in ln and " call " in ln
    ]
    parts: list[tuple[str, str]] = []
    for call in calls[:2]:
        found = re.findall(r"<(\d+) x bfloat> (%[\w.]+)", call)
        assert len(found) == 2, call
        parts.extend(found)
    assert len(parts) == 4, calls
    lines: list[str] = []
    inserted = False
    for ln in llvm.splitlines():
        if not inserted and re.fullmatch(r"\s*ret void\s*", ln):
            for i, (lanes, name) in enumerate(parts):
                lines.append(
                    f"  %probe{i} = bitcast <{lanes} x bfloat> {name} to <{lanes} x i16>"
                )
                lines.append(f"  call void @{_PROBE_SINK}(<{lanes} x i16> %probe{i})")
            inserted = True
        lines.append(ln)
    assert inserted, "no `ret void` to insert the probes before"
    lines.append(f"declare void @{_PROBE_SINK}(<{parts[0][0]} x i16>)")
    return "\n".join(lines) + "\n"


# payload bits -> (big, small) as bf16 bit patterns, with the arithmetic
# spelled out. M is 0x7f7f0000, the largest finite bf16 (as an fp32 payload).
_FOLDED_SPLITS = {
    # x - M = 0xe000 * 2^104 = 1.75 * 2^119, exact in bf16.
    0x7F7FE000: (0x7F7F, 0x7B60),
    0xFF7FE000: (0xFF7F, 0xFB60),
    # x - M = 2^119; the first payload that would have rounded to infinity.
    0x7F7F8000: (0x7F7F, 0x7B00),
    # Just below the threshold: rounds to M without any help from the cap, and
    # the remainder (0x7fff * 2^104, ~2^119) rounds to the same 2^119.
    0x7F7F7FFF: (0x7F7F, 0x7B00),
    # x - M = 0xffff * 2^104, which rounds up to 2^120.
    0x7F7FFFFF: (0x7F7F, 0x7B80),
    # Non-finite: big is capped to a finite value, small stays +-inf / NaN.
    0x7F800000: (0x7F7F, 0x7F80),
    0xFF800000: (0xFF7F, 0xFF80),
    0x7FC00000: (0x7F7F, 0x7FC0),
    # Nothing special: 1 + 2^-8 sits halfway between two bf16 values, ties to
    # even keeps 1.0 and leaves 2^-8 (exponent -8 -> 0x3b80) ...
    0x3F808000: (0x3F80, 0x3B80),
    # ... and 1 + 2^-11 + 2^-23 rounds to 1.0, leaving 2^-11 (-> 0x3a00).
    0x3F801001: (0x3F80, 0x3A00),
}


class TestLlvmFoldsTheSplit(unittest.TestCase):
    """LLVM's own constant folder, as an evaluator of the emitted IR.

    Everything above reads the IR as text, and the host model reads it as
    arithmetic; neither runs *it*. Here the lowered kernel is handed to ``opt``
    with constant payloads, which folds the whole split -- the narrowing, the cap,
    the widening, the subtraction -- down to constants, which are read back as
    bf16 bit patterns and compared with values worked out by hand below. No GPU
    is needed, and no model of mine sits between the IR and the answer.

    Both engines' output is evaluated -- the C++ engine's through the pybind
    module, so that lane skips where only the archive is built -- rather than the
    second being inferred from the byte-identity gate. Skips without an ``opt``;
    it uses the IR flavor that matches that ``opt``.
    """

    @classmethod
    def setUpClass(cls):
        tool = _opt()
        if tool is None:
            raise unittest.SkipTest("opt not found; cannot evaluate the emitted IR")
        version = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, check=False
        ).stdout
        match = re.search(r"LLVM version (\d+)", version)
        if match is None:
            raise unittest.SkipTest(f"cannot tell which LLVM this opt is: {version!r}")
        cls.tool = tool
        cls.flavor = _flavor_for_llvm(int(match.group(1)))

    @staticmethod
    def _patterns(operand: str) -> set[int]:
        """Every distinct bit pattern a folded ``<N x i16>`` constant carries."""
        if "zeroinitializer" in operand:
            return {0}
        # LLVM prints an integer in two's complement, so 0xfb60 reads as -1184.
        return {int(v) & 0xFFFF for v in re.findall(r"i16 (-?\d+)", operand)}

    def _lower(self, engine: str, kernel: KernelDef) -> str:
        if engine == "python":
            return _lower_kernel_to_llvm_python(
                kernel, arch=_ARCH, llvm_flavor=self.flavor
            )
        try:
            return _lower_via_cpp_engine(serialize(kernel), _ARCH, self.flavor)
        except BackendError as exc:
            self.skipTest(f"C++ engine unavailable: {exc}")

    def _folded_operands(
        self, op_id: str, a_bits: int, b_bits: int, engine: str = "python"
    ):
        """``(a_small, b_big, a_big, b_small)`` as sets of bf16 bit patterns."""
        llvm = self._lower(engine, _fold_kernel(op_id, a_bits, b_bits))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "split.ll"
            source.write_text(_with_probes(llvm))
            done = subprocess.run(
                [self.tool, "-S", "-passes=instcombine", str(source), "-o", "-"],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        sinks = [
            ln.split("(", 1)[1]
            for ln in done.stdout.splitlines()
            if ln.strip().startswith(f"call void @{_PROBE_SINK}(")
        ]
        self.assertEqual(len(sinks), 4, done.stdout)
        return tuple(self._patterns(operand) for operand in sinks)

    def _check_the_hand_derived_values(self, engine: str):
        for op_id in _EMULATED:
            for a_bits, (big, small) in _FOLDED_SPLITS.items():
                with self.subTest(engine=engine, op_id=op_id, payload=hex(a_bits)):
                    a_small, b_big, a_big, b_small = self._folded_operands(
                        op_id, a_bits, _SMALLEST_NORMAL, engine
                    )
                    self.assertEqual((a_big, a_small), ({big}, {small}))
                    # The other operand is a power of two: its split is exact, so
                    # its remainder is zero, and it is unaffected by the cap.
                    self.assertEqual((b_big, b_small), ({0x0080}, {0}))

    def _check_no_infinity_in_a_term(self, engine: str):
        """The failure the cap exists for, stated as a property of the folded IR."""
        for op_id in _EMULATED:
            for a_bits in (0x7F7F8000, 0x7F7FE000, 0x7F7FFFFF, 0xFF7FE000):
                with self.subTest(engine=engine, op_id=op_id, payload=hex(a_bits)):
                    parts = self._folded_operands(
                        op_id, a_bits, _SMALLEST_NORMAL, engine
                    )
                    for patterns in parts:
                        for pattern in patterns:
                            # Exponent field all ones is an infinity or a NaN.
                            self.assertNotEqual(pattern & 0x7F80, 0x7F80, hex(pattern))

    def test_the_folded_split_matches_the_hand_derived_values(self):
        self._check_the_hand_derived_values("python")

    def test_the_cpp_engines_folded_split_matches_too(self):
        self._check_the_hand_derived_values("cpp")

    def test_a_finite_payload_never_leaves_an_infinity_in_a_term(self):
        self._check_no_infinity_in_a_term("python")

    def test_the_cpp_engine_never_leaves_an_infinity_in_a_term_either(self):
        self._check_no_infinity_in_a_term("cpp")


class TestOperandContract(unittest.TestCase):
    """Emulated atoms share the native atoms' ABI: TF32 in, FP32 accumulator.

    One kernel body can then reach either through a catalog query. An fp32 vector
    is not a TF32 operand, and a TF32 vector of the wrong length is not this
    atom's fragment; the verifier says which, naming the atom's own lengths.
    """

    def _build(self, op_id: str, a_type, b_type):
        atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
        builder = IRBuilder("emulated_mma_bad_operands")
        a = builder.vector_splat(builder.const_f32(1.0), a_type.count)
        b = builder.vector_splat(builder.const_f32(1.0), b_type.count)
        if a_type.elem is TF32:
            a = builder.bitcast(a, a_type)
        if b_type.elem is TF32:
            b = builder.bitcast(b, b_type)
        c = builder.vector_splat(builder.const_f32(0.0), atom.c_frag_len)
        return builder, atom, a, b, c

    def test_fp32_fragments_are_rejected(self):
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                n = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id).a_frag_len
                builder, atom, a, b, c = self._build(
                    op_id, VectorType(F32, n), VectorType(F32, n)
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{op_id} requires vec<tf32x{n}>, vec<tf32x{n}> and vec<f32x",
                ):
                    builder.mma(atom, a, b, c)

    def test_wrong_length_tf32_fragments_are_rejected(self):
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                n = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id).a_frag_len
                short = VectorType(TF32, n // 2)
                builder, atom, a, b, c = self._build(op_id, short, short)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{op_id} requires vec<tf32x{n}>, vec<tf32x{n}> and vec<f32x",
                ):
                    builder.mma(atom, a, b, c)

    def test_ordinary_atoms_still_refuse_tf32_operands(self):
        """The new branch is keyed on the catalog, so it cannot widen the old rule."""
        physical = next(iter(_EMULATED.values()))
        atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(physical)
        builder = IRBuilder("native_mma_with_tf32")
        a = builder.bitcast(
            builder.vector_splat(builder.const_f32(1.0), atom.a_frag_len),
            VectorType(TF32, atom.a_frag_len),
        )
        c = builder.vector_splat(builder.const_f32(0.0), atom.c_frag_len)
        with self.assertRaisesRegex(ValueError, "TF32 operands require an XF32 MMA"):
            builder.mma(atom, a, a, c)


class TestHipBackend(unittest.TestCase):
    def test_hip_lowering_rejects_an_emulated_atom_by_name(self):
        """The expansion exists only in the LLVM lowerer.

        Without the explicit rejection the op reaches the generic dispatcher as
        ``tile.<op_id>`` and fails as though HIP had merely missed a hardware atom,
        which, with a name shaped like one, is what it would read as.
        """
        from rocke.core.lower_hip import lower_kernel_to_hip

        for op_id, physical in _EMULATED.items():
            with self.subTest(op_id=op_id):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"tile\.mma {op_id}: an emulated atom \(bf16x3 over {physical}\) "
                    r"has no HIP lowering",
                ):
                    lower_kernel_to_hip(_mma_kernel(op_id), arch=_ARCH)


class TestBareOpIdForm(unittest.TestCase):
    def test_the_bare_op_id_builds_the_same_kernel_as_the_atom(self):
        """``IRBuilder.mma`` documents a bare-op_id form; emulated atoms take it.

        The logical op_id has no fragment of its own, so this is the case that
        needs the lookup to resolve through the physical atom.
        """
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                self.assertEqual(
                    serialize(_mma_kernel(op_id, bare=True)),
                    serialize(_mma_kernel(op_id)),
                )


class TestNativeAtomsAreUntouched(unittest.TestCase):
    def test_a_native_bf16_mma_still_lowers_to_one_call(self):
        """The guard is confined to emulated atoms -- no native atom moved."""
        for physical in _EMULATED.values():
            with self.subTest(op_id=physical):
                body = _body(
                    _lower_kernel_to_llvm_python(
                        _mma_kernel(physical), arch=_ARCH, llvm_flavor="llvm20"
                    )
                )
                self.assertEqual(len([ln for ln in body if "mfma" in ln]), 1)
                self.assertEqual(len([ln for ln in body if "fptrunc" in ln]), 0)


class TestEngineParity(unittest.TestCase):
    """The #1 invariant: the two engines emit the same bytes for the expansion.

    Needs the ``rocke_engine`` pybind module and skips without it. Where only
    the archive is built, the same comparison is the ``xf32_bf16x3_mma`` parity
    family that ``tools/check_byte_identity.py`` runs.
    """

    def test_python_cpp_byte_identity_for_every_flavor(self):
        for op_id in _EMULATED:
            kernel = _mma_kernel(op_id)
            serialized = serialize(kernel)
            try:
                _lower_via_cpp_engine(serialized, _ARCH, "llvm20")
            except BackendError as exc:
                self.skipTest(f"C++ engine unavailable: {exc}")
            for flavor in _FLAVORS:
                with self.subTest(op_id=op_id, flavor=flavor):
                    self.assertEqual(
                        _lower_kernel_to_llvm_python(
                            kernel, arch=_ARCH, llvm_flavor=flavor
                        ),
                        _lower_via_cpp_engine(serialized, _ARCH, flavor),
                    )


if __name__ == "__main__":
    unittest.main()
