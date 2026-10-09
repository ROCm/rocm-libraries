# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lowering gate for the XF32 BF16x3 emulated MMA atoms (AICK-2208).

An emulated ``tile.mma`` expands, before ISA dispatch, into a Dekker split of
each fp32 operand plus three BF16 MMAs threading one fp32 accumulator. What is
pinned here is everything a numerics reviewer would want to re-derive by hand:

* the split is ``big = rne_bf16(x)``, ``small = rne_bf16(x - fp32(big))``;
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

import re
import unittest

from rocke.core.arch import ArchTarget
from rocke.core.backend import BackendError, _lower_via_cpp_engine
from rocke.core.ir import IRBuilder, KernelDef
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCH = "gfx950"
_FLAVORS = ("llvm20", "llvm22", "llvm23")

# Every emulated atom the catalog ships, paired with the bf16 atom it expands
# to. Kept as literals rather than read back out of the catalog: this file is
# the place a reviewer checks the catalog against, so deriving it from the
# catalog would make the test agree with itself.
_EMULATED = {
    "xf32_bf16x3_f32_16x16x32": "mfma_f32_16x16x32_bf16",
    "xf32_bf16x3_f32_32x32x16": "mfma_f32_32x32x16_bf16",
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
    c = builder.vector_splat(builder.const_f32(0.0), atom.c_frag_len)
    builder.mma(op_id if bare else atom, a, b, c)
    builder.ret()
    return builder.kernel


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
                f32v, bf16v = f"<{n} x float>", f"<{n} x bfloat>"
                _, body = self._lower(op_id)
                for side in ("a", "b"):
                    split = [
                        ln.strip()
                        for ln in body
                        if f"%xf32_{side}_" in ln.split("=")[0]
                    ]
                    self.assertEqual(
                        len(split), 4, f"operand {side}: expected a 4-line split"
                    )
                    big, big_f32, diff, small = split
                    src = big.split(f"fptrunc {f32v} ")[1].split(" to ")[0]
                    big_n = big.split(" =")[0]
                    # big = rne_bf16(src)
                    self.assertEqual(big, f"{big_n} = fptrunc {f32v} {src} to {bf16v}")
                    # ... widened back so the rounding error can be named,
                    bf_n = big_f32.split(" =")[0]
                    self.assertEqual(
                        big_f32, f"{bf_n} = fpext {bf16v} {big_n} to {f32v}"
                    )
                    # ... which is exactly what the source lost,
                    diff_n = diff.split(" =")[0]
                    self.assertEqual(diff, f"{diff_n} = fsub {f32v} {src}, {bf_n}")
                    # ... itself rounded to bf16.
                    small_n = small.split(" =")[0]
                    self.assertEqual(
                        small, f"{small_n} = fptrunc {f32v} {diff_n} to {bf16v}"
                    )

    def test_rounding_is_round_to_nearest_even_not_truncation(self):
        """Both narrowing steps are ``fptrunc``, which LLVM defines as RNE.

        The AMDGPU backend selects the packed ``v_cvt_pk_bf16_f32`` for the
        vector form on gfx950 -- CK's shipping packed path. CK's *scalar*
        fallback truncates instead, and is deliberately not the parity target:
        an RNE split is strictly the more accurate of the two.
        """
        for op_id in _EMULATED:
            with self.subTest(op_id=op_id):
                atom = ArchTarget.from_gfx(_ARCH).mma.by_op_id(op_id)
                bf16v = f"<{atom.a_frag_len} x bfloat>"
                _, body = self._lower(op_id)
                narrowing = [ln for ln in body if ln.endswith(f" to {bf16v}")]
                self.assertTrue(narrowing)
                for line in narrowing:
                    self.assertIn("fptrunc", line)
                    # No intrinsic call, and no bit-masking stand-in for a cast.
                    self.assertNotIn("call ", line)
                    self.assertNotIn(" and ", line)

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
    """The #1 invariant: the two engines emit the same bytes for the expansion."""

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
