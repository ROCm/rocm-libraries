# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 erratum guard and the ISA check: the text checker; the guarded gfx950
MFMA fixture passes; the same fixture with the guard forced off is rejected; gfx942 is
never guarded."""

import importlib.util
import sys
import unittest
from pathlib import Path

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    check_erratum_object,
    check_erratum_text,
    disassemble,
    isa_counts,
)
from generators._manifest import erratum_record  # noqa: E402

FIXTURE = TRITON_DIR / "tests" / "fixtures" / "mfma_fixture.py"


class IsaTextCheckerTest(unittest.TestCase):
    def test_pk_f32_beside_mfma_rejected(self):
        text = "\tv_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], 0\n\tv_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]\n"
        with self.assertRaises(GeneratorError) as raised:
            check_erratum_text(text, "text")
        self.assertTrue(str(raised.exception).startswith(C.ERR_ERRATUM))

    def test_each_alone_passes(self):
        check_erratum_text("\tv_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]\n", "text")
        check_erratum_text(
            "\tv_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], 0\n", "text"
        )

    def test_packed_f32_variants_counted(self):
        for op in ("v_pk_fma_f32", "v_pk_mul_f32", "v_pk_add_f32"):
            self.assertEqual(isa_counts(f"v_mfma_x\n{op} v0\n"), (1, 1))
        self.assertEqual(isa_counts("v_mfma_x\nv_pk_mov_b32 v0\n"), (1, 0))

    def test_smfmac_counts_as_mfma(self):
        text = "\tv_smfmac_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v12\n"
        self.assertEqual(isa_counts(text), (1, 0))
        with self.assertRaises(GeneratorError):
            check_erratum_text(text + "\tv_pk_mul_f32 v[0:1], v[2:3], v[4:5]\n", "t")


class GuardRuleTest(unittest.TestCase):
    def test_rule_is_gfx950_and_mfma_only(self):
        mfma_ir = "call @llvm.amdgcn.mfma.f32.16x16x32.bf16("
        self.assertEqual(
            env.guarded_features("gfx950", "a", mfma_ir),
            ("a," + env.GUARD_FEATURE, True),
        )
        self.assertEqual(
            env.guarded_features("gfx950", "", mfma_ir), (env.GUARD_FEATURE, True)
        )
        self.assertEqual(env.guarded_features("gfx942", "a", mfma_ir), ("a", False))
        self.assertEqual(
            env.guarded_features("gfx950", "a", "call @llvm.fma.f32("), ("a", False)
        )


class CompiledFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env.prepare()
        env.import_triton()
        spec = importlib.util.spec_from_file_location("tritongen_mfma_fixture", FIXTURE)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.fn = cls.module.mfma_fixture

    def compile(self, arch, guard=True):
        m = self.module
        return env.compile_instance(
            self.fn, m.SIGNATURE, m.CONSTEXPRS, None, arch, m.NUM_WARPS, guard=guard
        )

    def test_gfx950_guarded_passes(self):
        compiled = self.compile("gfx950")
        self.assertTrue(compiled.guard_applied)
        self.assertEqual(erratum_record(compiled.guard_applied)["applied"], True)
        mfma, pk = check_erratum_object(compiled.hsaco, "gfx950", "guarded")
        self.assertGreater(mfma, 0)
        self.assertEqual(pk, 0)

    def test_gfx950_forced_unguarded_rejected(self):
        compiled = self.compile("gfx950", guard=False)
        self.assertFalse(compiled.guard_applied)
        mfma, pk = isa_counts(disassemble(compiled.hsaco, "gfx950", "unguarded"))
        self.assertGreater(mfma, 0)
        self.assertGreater(pk, 0)
        with self.assertRaises(GeneratorError) as raised:
            check_erratum_object(compiled.hsaco, "gfx950", "unguarded")
        self.assertTrue(str(raised.exception).startswith(C.ERR_ERRATUM))

    def test_gfx942_not_guarded(self):
        compiled = self.compile("gfx942")
        self.assertFalse(compiled.guard_applied)
        self.assertGreater(
            isa_counts(disassemble(compiled.hsaco, "gfx942", "gfx942"))[0], 0
        )


if __name__ == "__main__":
    unittest.main()
