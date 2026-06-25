# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Round-trip + verifier tests for the ``ck.dsl.ir/v1`` serialization.

Covers:
  * round-trip idempotence (build -> serialize -> parse -> serialize is
    byte-identical) on several REAL kernels from the instances catalog;
  * lowering equality (the parsed KernelDef lowers to the same LLVM IR as the
    original);
  * canonical (semantic) equality across id renumbering;
  * verifier returns no errors on valid kernels and flags corrupted ones.
"""

from __future__ import annotations

import copy
import hashlib
import unittest

from rocke.core.ir import (
    F32,
    I32,
    PtrType,
    Value,
    VectorType,
)
from rocke.core.ir_serialize import (
    canonical_equal,
    canonicalize,
    parse,
    serialize,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import ERROR, verify, verify_or_raise


# --------------------------------------------------------------------------
# Build a corpus of REAL kernels from the instances catalog.
# --------------------------------------------------------------------------


def _build_corpus():
    kernels = {}

    from rocke.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    kernels["gemm_universal"] = build_universal_gemm(
        UniversalGemmSpec(
            name="vgemm",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
        )
    )

    from rocke.instances.common.rmsnorm2d import (
        RMSNorm2DSpec,
        build_rmsnorm2d,
    )

    kernels["rmsnorm2d"] = build_rmsnorm2d(RMSNorm2DSpec(n_per_block=1024))

    from rocke.instances.common.layernorm2d import (
        LayerNorm2DSpec,
        build_layernorm2d,
    )

    kernels["layernorm2d"] = build_layernorm2d(LayerNorm2DSpec(n_per_block=1024))

    from rocke.instances.common.elementwise import (
        ElementwiseSpec,
        build_elementwise,
    )

    kernels["elementwise"] = build_elementwise(ElementwiseSpec(op="add"))

    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    kernels["conv_implicit_gemm"] = build_implicit_gemm_conv(
        ImplicitGemmConvSpec(
            problem=ConvProblem(N=1, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)
        )
    )

    return kernels


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TestRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = _build_corpus()

    def test_roundtrip_byte_identical(self):
        """serialize -> parse -> serialize is byte-identical for real kernels."""
        for name, k in self.corpus.items():
            with self.subTest(kernel=name):
                s1 = serialize(k)
                k2 = parse(s1)
                s2 = serialize(k2)
                self.assertEqual(_sha(s1), _sha(s2), f"{name}: round-trip sha differs")
                self.assertEqual(s1, s2, f"{name}: round-trip text differs")

    def test_roundtrip_lowers_identically(self):
        """The parsed KernelDef lowers to the SAME LLVM IR as the original."""
        for name, k in self.corpus.items():
            with self.subTest(kernel=name):
                k2 = parse(serialize(k))
                ll1 = lower_kernel_to_llvm(k)
                ll2 = lower_kernel_to_llvm(k2)
                self.assertEqual(
                    _sha(ll1), _sha(ll2), f"{name}: lowered LLVM sha differs"
                )

    def test_serialization_is_versioned(self):
        for name, k in self.corpus.items():
            with self.subTest(kernel=name):
                self.assertTrue(serialize(k).startswith("ckdsl.ir v1\n"))

    def test_parse_rejects_unknown_version(self):
        s = serialize(self.corpus["rmsnorm2d"]).replace("ckdsl.ir v1", "ckdsl.ir v9", 1)
        with self.assertRaises(ValueError):
            parse(s)

    def test_canonical_equal_self(self):
        for name, k in self.corpus.items():
            with self.subTest(kernel=name):
                self.assertTrue(canonical_equal(k, parse(serialize(k))))

    def test_canonical_tolerates_id_gaps(self):
        """Two structurally identical kernels with different incidental SSA ids
        canonicalize to the same string (the drift-tolerant semantic diff)."""
        from rocke.core.ir import IRBuilder

        def build(start_counter: int):
            b = IRBuilder("twin")
            b._counter = start_counter  # perturb id numbering only
            x = b.param("X", PtrType(F32, "global"), noalias=True)
            i = b.const_i32(0)
            v = b.global_load_f32(x, i)
            one = b.const_f32(1.0)
            s = b.fadd(v, one)
            b.global_store(x, i, s)
            b.ret()
            return b.kernel

        k_a = build(0)
        k_b = build(100)  # every result id shifted by 100
        self.assertNotEqual(
            serialize(k_a), serialize(k_b), "ids should differ before canon"
        )
        self.assertEqual(
            canonicalize(k_a), canonicalize(k_b), "canonical form should match"
        )
        self.assertTrue(canonical_equal(k_a, k_b))


class TestAttrEncoding(unittest.TestCase):
    def test_inline_asm_with_embedded_specials(self):
        """Embedded newline + quote in a string attr round-trip exactly."""
        from rocke.core.ir import IRBuilder

        b = IRBuilder("asm_test")
        x = b.param("X", PtrType(F32, "global"), noalias=True, align=16)
        i = b.const_i32(0)
        v = b.global_load_f32(x, i)
        vi = b.bitcast(v, I32)
        r = b.inline_asm(
            'v_mov_b32 $0, $1\n  ; note "x"',
            "=v,v",
            [vi],
            result_type=I32,
            sideeffect=True,
        )
        rf = b.bitcast(r, F32)
        b.global_store(x, i, rf)
        b.ret()
        k = b.kernel
        s1 = serialize(k)
        k2 = parse(s1)
        self.assertEqual(s1, serialize(k2))
        self.assertEqual(_sha(lower_kernel_to_llvm(k)), _sha(lower_kernel_to_llvm(k2)))
        # the original attr text survives the round-trip
        op = next(o for o in k2.body.ops if o.name == "tile.inline_asm")
        self.assertEqual(op.attrs["template"], 'v_mov_b32 $0, $1\n  ; note "x"')
        self.assertIs(op.attrs["sideeffect"], True)

    def test_bool_not_confused_with_int(self):
        """A bool attr survives as a bool (not int 1) across round-trip."""
        from rocke.core.ir import IRBuilder

        b = IRBuilder("bool_test")
        x = b.param("X", PtrType(F32, "global"))
        i = b.const_i32(0)
        v = b.global_load_f32(x, i)
        b.global_store(x, i, v)
        b.ret()
        k = b.kernel
        # inject a known bool + float attr on an op
        k.body.ops[0].attrs["flag"] = True
        k.body.ops[0].attrs["scale"] = 0.5
        k2 = parse(serialize(k))
        self.assertIs(k2.body.ops[0].attrs["flag"], True)
        self.assertEqual(k2.body.ops[0].attrs["scale"], 0.5)
        self.assertIsInstance(k2.body.ops[0].attrs["scale"], float)


class TestVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = _build_corpus()

    def test_valid_kernels_have_no_errors(self):
        for name, k in self.corpus.items():
            with self.subTest(kernel=name):
                errors = [d for d in verify(k) if d.severity == ERROR]
                self.assertEqual(errors, [], f"{name}: {[str(e) for e in errors]}")
                verify_or_raise(k)  # must not raise

    def test_flags_use_before_def(self):
        k = copy.deepcopy(self.corpus["rmsnorm2d"])
        for op in k.body.ops:
            if op.operands:
                op.operands[0] = Value(
                    name="%ghost_undefined", type=op.operands[0].type
                )
                break
        errors = [d for d in verify(k) if d.severity == ERROR]
        self.assertTrue(errors, "expected a use-before-def error")
        self.assertTrue(
            any("used before definition" in e.message for e in errors),
            [str(e) for e in errors],
        )
        with self.assertRaises(ValueError):
            verify_or_raise(k)

    def test_flags_type_mismatch(self):
        k = copy.deepcopy(self.corpus["gemm_universal"])
        patched = False
        for op in k.body.ops:
            if op.name == "arith.add" and len(op.operands) == 2 and op.results:
                op.results[0].type = VectorType(F32, 4)
                patched = True
                break
        self.assertTrue(patched, "no arith.add to corrupt; adjust the test")
        errors = [d for d in verify(k) if d.severity == ERROR]
        self.assertTrue(
            any("result type" in e.message for e in errors),
            [str(e) for e in errors],
        )

    def test_flags_missing_required_attr(self):
        k = copy.deepcopy(self.corpus["gemm_universal"])
        patched = False
        for op in k.body.ops:
            if op.name == "tile.mma":
                op.attrs.pop("op_id", None)
                patched = True
                break

        def walk(region):
            nonlocal patched
            for op in region.ops:
                if op.name == "tile.mma" and "op_id" in op.attrs:
                    op.attrs.pop("op_id")
                    patched = True
                for r in op.regions:
                    walk(r)

        if not patched:
            walk(k.body)
        self.assertTrue(patched, "no tile.mma found to corrupt")
        errors = [d for d in verify(k) if d.severity == ERROR]
        self.assertTrue(
            any("missing required attr" in e.message for e in errors),
            [str(e) for e in errors],
        )


if __name__ == "__main__":
    unittest.main()
