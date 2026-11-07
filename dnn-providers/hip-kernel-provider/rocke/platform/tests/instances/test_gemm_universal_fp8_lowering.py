# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What the 8-bit universal GEMM actually emits on gfx1250.

The fp8 path through ``instances/common/gemm_universal.py`` differs from the
fp16/bf16 path in three places that all live in the emitted IR and nowhere else:
the WMMA intrinsic and its operand ABI, the element width of the global A/B
traffic, and the C store type. A spec can be well-formed, pass
``GemmPipelinePolicy.validate``, build without raising, and still be wrong in
any of them -- the result is a kernel that runs and returns numbers.

Two of the three are not symmetric with anything the fp16 path does, which is
why they are asserted here rather than trusted to the byte-identity gate:

* The K=64 fp8 intrinsic takes ``(<8 x i32>, <8 x i32>, i16, <8 x float>, i1,
  i1)`` -- **no** leading ``i1 immarg`` sign-extend flags, unlike the K=32 f16
  form ``(i1, <16 x half>, i1, <16 x half>, i16, <8 x float>, i1, i1)``. Copying
  the f16 call shape is the obvious mistake and produces a signature mismatch
  only at link time, or silently wrong operands if the arity happens to line up.
* A and B are fp8 while C is bf16. Every other GEMM in this family has one dtype
  across all three, so a C that came back ``i8`` would look locally consistent.

The third is the padding zero. ``_zero_storage_scalar`` cannot route +0.0
through ``cast_f32_to`` -- that helper has no 8-bit arm and used to raise -- so
it truncates a zero i32 instead. That arm is reached only when padding is on,
which the tests below exercise from both sides so the assertion is about the
padding path and not about something that is simply always there.

Host-only: ``lower_kernel_to_llvm`` runs the Python lowerer directly, so no
compiler, GPU, or built C++ engine is required.
"""

from __future__ import annotations

import re
import unittest

from rocke.core.arch import ArchTarget
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances import GemmPipelinePolicy
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)

_ARCH = "gfx1250"

# The gfx1250 WMMA path accepts only the 'mem' and 'wmma_v1' pipelines; 'compv3'
# is rejected by the policy before any IR exists.
_PIPELINE = "mem"

_IR_CACHE: dict[tuple, str] = {}


def _ir(
    dtype: str,
    *,
    c_dtype: str = "bf16",
    pad: bool = True,
    wtk: int = 64,
    ab_load_elem_bytes: int | None = None,
    lds_k_pad: int = 0,
) -> str:
    """Lower one universal GEMM and return its LLVM text.

    Defaults are the shape the fp8 verify harness builds: a 16x16x64 atom in a
    2x2 warp grid with padding on. ``wtk`` drops to 32 for the bf16 contrast,
    since that is the widest K its atom offers. ``ab_load_elem_bytes`` pins the
    global A/B load-vector element width; None resolves it from the dtype.

    Every argument must appear in ``key`` -- the cache is keyed on it, so a
    parameter left out would silently hand back IR built with a different value.
    """
    key = (dtype, c_dtype, pad, wtk, ab_load_elem_bytes, lds_k_pad)
    if key not in _IR_CACHE:
        target = ArchTarget.from_gfx(_ARCH)
        tile = TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=max(32, wtk),
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=wtk,
        )
        trait = TraitSpec(
            pipeline=_PIPELINE,
            scheduler="intrawave",
            epilogue="default",
            pad_m=pad,
            pad_n=pad,
            pad_k=pad,
            ab_load_elem_bytes=ab_load_elem_bytes,
            lds_k_pad=lds_k_pad,
        )
        data = DataSpec(
            dtype_a=dtype,
            dtype_b=dtype,
            dtype_c=c_dtype,
            dtype_acc="fp32",
            layout="RCR",
        )
        spec = UniversalGemmSpec(
            name=f"ugemm_{dtype}_{'pad' if pad else 'nopad'}",
            tile=tile,
            trait=trait,
            data=data,
            wave_size=target.wave_size,
        )
        res = GemmPipelinePolicy().validate(target, spec)
        if not res.ok:  # pragma: no cover - a spec change, not a lowering bug
            raise AssertionError(f"spec rejected before lowering: {res.reason}")
        _IR_CACHE[key] = lower_kernel_to_llvm(
            build_universal_gemm(spec, arch=_ARCH), arch=_ARCH
        )
    return _IR_CACHE[key]


class TestFp8Intrinsic(unittest.TestCase):
    def test_the_declared_signature_has_no_sign_extend_flags(self):
        # The whole reason this test exists: the K=64 low-bit form drops the two
        # leading i1 immargs the K=32 f16 form carries. Asserting the full
        # declare pins arity and operand order together -- a partial match on the
        # intrinsic name alone would pass for a call built from the f16 shape.
        self.assertIn(
            "declare <8 x float> "
            "@llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32("
            "<8 x i32>, <8 x i32>, i16 immarg, <8 x float>, "
            "i1 immarg, i1 immarg)",
            _ir("fp8e4m3"),
        )

    def test_the_call_passes_packed_i32_operands_and_an_f32_accumulator(self):
        # A/B arrive as <8 x i32> (32 bytes of fp8 per lane) and the accumulator
        # stays f32. If an operand were widened to <8 x float> somewhere upstream
        # the declare above would still match while the call would not.
        ll = _ir("fp8e4m3")
        calls = [
            line
            for line in ll.splitlines()
            if "@llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8" in line
            and not line.lstrip().startswith("declare")
        ]
        self.assertTrue(calls, "no call site for the fp8 WMMA intrinsic")
        for line in calls:
            self.assertEqual(line.count("<8 x i32> %"), 2, line)
            self.assertIn("i16 0, <8 x float> %", line)

    def test_bf8_selects_its_own_intrinsic(self):
        # bf8 has no numeric coverage anywhere, so the one thing asserted about
        # it is that it does not quietly reuse the fp8 opcode -- which would
        # reinterpret every operand byte under the wrong exponent width.
        ll = _ir("bf8e5m2")
        self.assertIn("@llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8", ll)
        self.assertNotIn("@llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8", ll)


class TestEightBitMemoryTraffic(unittest.TestCase):
    def test_global_a_b_traffic_is_byte_typed(self):
        # Padded tiles load A/B one guarded element at a time; unpadded tiles
        # take the whole run. Both are asserted because the element width is
        # the claim, not the vector width -- an i8 that had been widened to i16
        # would read the right addresses and the wrong bytes.
        self.assertIn("load i8, ptr addrspace(1)", _ir("fp8e4m3", pad=True))
        self.assertIn("load <16 x i8>, ptr addrspace(1)", _ir("fp8e4m3", pad=False))

    def test_a_one_byte_operand_uses_the_whole_sixteen_byte_load(self):
        # The picker caps the vector at 16/elem_bytes. Resolving the width from
        # the dtype is what spends all four dwords on a 1-byte operand; the
        # historical default of 2 capped it at 8 elements and wasted half the
        # load. Asserting the alignment too, because `align 16` is a claim to
        # LLVM about the address, not a formatting detail.
        ll = _ir("fp8e4m3", pad=False)
        self.assertIn("load <16 x i8>, ptr addrspace(1)", ll)
        self.assertIn("align 16", ll)
        self.assertNotIn("load <8 x i8>, ptr addrspace(1)", ll)

    def test_the_override_reproduces_the_narrow_load(self):
        # The counterfactual that makes the assertion above mean something: the
        # same spec with the width pinned to 2 goes back to 8 elements. Without
        # it, a picker that ignored elem_bytes entirely and always returned 16
        # would pass every positive case here.
        ll = _ir("fp8e4m3", pad=False, ab_load_elem_bytes=2)
        self.assertIn("load <8 x i8>, ptr addrspace(1)", ll)
        self.assertNotIn("load <16 x i8>, ptr addrspace(1)", ll)

    def test_two_byte_operands_are_unaffected_by_the_resolved_width(self):
        # bf16 resolves to elem_bytes=2, which is what the picker defaulted to
        # all along -- so the widening must leave every f16/bf16 kernel byte
        # for byte where it was. 8 bfloats is already a full 16 bytes.
        ll = _ir("bf16", wtk=32, pad=False)
        self.assertIn("load <8 x bfloat>, ptr addrspace(1)", ll)
        self.assertNotIn("load <16 x bfloat>", ll)

    def test_c_is_stored_as_bfloat_and_never_as_a_byte(self):
        # The asymmetric half of the ABI. An fp8 C would halve the store width
        # and silently requantize every output element.
        for pad in (True, False):
            with self.subTest(pad=pad):
                ll = _ir("fp8e4m3", pad=pad)
                self.assertIn("store bfloat", ll)
                self.assertNotIn("store i8, ptr addrspace(1)", ll)
                self.assertNotIn("store <8 x i8>, ptr addrspace(1)", ll)


class TestLdsAlignmentClaims(unittest.TestCase):
    """The ``align`` on an LDS access must not exceed what the address gives.

    An alignment attribute is an assertion to LLVM, not a formatting detail.
    Overstating it is undefined behaviour even where the hardware happens to
    tolerate the access, so these assert an upper bound rather than a value.
    """

    @staticmethod
    def _lds_aligns(ll: str, kind: str) -> set[int]:
        pat = rf"{kind} <\d+ x i8>[^\n]*addrspace\(3\)[^\n]*align (\d+)"
        return {int(m) for m in re.findall(pat, ll)}

    def test_a_sixteen_byte_access_never_claims_more_than_sixteen(self):
        # The load path's element-size map omitted the 8-bit floats, so they
        # fell to 2 bytes and a 16-element read claimed align 32 -- double the
        # bytes actually touched. No pad choice could avoid that one.
        ll = _ir("fp8e4m3", pad=False)
        for kind in ("load", "store"):
            with self.subTest(kind=kind):
                claims = self._lds_aligns(ll, kind)
                self.assertTrue(claims, f"no 16-byte LDS {kind} found")
                self.assertLessEqual(max(claims), 16)

    def test_an_unaligned_row_stride_lowers_the_claim(self):
        # lds_k_pad=56 on a 64-element tile gives a 120-element row stride, so
        # alternate rows start 8 bytes off a 16-byte boundary. This is the
        # counterfactual: without it every stride the vector width divides
        # would pass even if the claim were still derived from the vector
        # shape alone, which is exactly how the bug survived.
        ll = _ir("fp8e4m3", pad=False, lds_k_pad=56)
        for kind in ("load", "store"):
            with self.subTest(kind=kind):
                claims = self._lds_aligns(ll, kind)
                self.assertTrue(claims, f"no 16-byte LDS {kind} found")
                self.assertLessEqual(max(claims), 8)

    def test_a_divisible_row_stride_keeps_the_full_claim(self):
        # The bound must not be applied indiscriminately: a stride the vector
        # width divides still earns align 16, or the fix would cost instruction
        # selection on every well-padded config.
        ll = _ir("fp8e4m3", pad=False, lds_k_pad=80)
        self.assertIn(16, self._lds_aligns(ll, "store"))


class TestPaddingZero(unittest.TestCase):
    def test_the_padding_zero_is_a_truncated_integer(self):
        # _zero_storage_scalar's 8-bit arm. +0.0 is all-zero in both e4m3 and
        # e5m2 and both lower to i8, so the byte is produced by truncation rather
        # than by a conversion intrinsic -- and cast_f32_to, the route the other
        # dtypes take, has no 8-bit arm at all and raised here before the fix.
        self.assertIn("trunc i32 0 to i8", _ir("fp8e4m3", pad=True))
        self.assertIn("trunc i32 0 to i8", _ir("bf8e5m2", pad=True))

    def test_that_zero_comes_from_the_padding_path(self):
        # Without this the assertion above would still pass if an i8 zero were
        # emitted unconditionally somewhere else, and would prove nothing about
        # the arm it was written for.
        self.assertNotIn("trunc i32 0 to i8", _ir("fp8e4m3", pad=False))


class TestBf16PathIsUnchanged(unittest.TestCase):
    """Phase 0 widened shared dtype gates; the bf16 path must not have moved."""

    def test_bf16_keeps_the_k32_atom_and_two_byte_operands(self):
        ll = _ir("bf16", wtk=32)
        self.assertIn("@llvm.amdgcn.wmma.f32.16x16x32", ll)
        self.assertNotIn("16x16x64", ll)

    def test_bf16_emits_no_eight_bit_global_traffic_or_zero(self):
        # The gates widened in tensor_view.load_vec and _zero_storage_scalar are
        # shared with fp16/bf16. If either started matching on dtype too loosely,
        # byte-wide loads or an i8 zero would appear in a bf16 kernel.
        ll = _ir("bf16", wtk=32)
        self.assertNotIn("load i8, ptr addrspace(1)", ll)
        self.assertNotIn("load <8 x i8>, ptr addrspace(1)", ll)
        self.assertNotIn("trunc i32 0 to i8", ll)


if __name__ == "__main__":
    unittest.main()
