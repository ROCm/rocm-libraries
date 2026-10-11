# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The kernarg tuple (both scratch slots) is checked; a mutated tuple is rejected
with TRITONGEN-ABI."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators import gen_rmsnorm  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    describe,
    expected_signature,
    verify_abi,
)

INSTANCE = C.instances()[0]  # rmsnorm_bf16_bn1024


class KernargTupleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env.prepare()
        cls.compiled = gen_rmsnorm.compile_rmsnorm(
            env.load_kernel(), "gfx950", INSTANCE
        )
        cls.described = describe(cls.compiled.hsaco, "test")

    def test_real_object_matches_contract(self):
        gen_rmsnorm.verify_compiled(self.compiled, INSTANCE, "gfx950")
        self.assertEqual(self.described["signature"], expected_signature())
        self.assertEqual(len(self.described["signature"]), 7)

    def _mutated(self, signature):
        return {**self.described, "signature": tuple(signature)}

    def assert_abi_rejected(self, signature):
        with self.assertRaises(GeneratorError) as raised:
            verify_abi(self._mutated(signature), INSTANCE.name, "test")
        self.assertTrue(str(raised.exception).startswith(C.ERR_ABI), raised.exception)

    def test_changed_offset_rejected(self):
        sig = list(self.described["signature"])
        kind, size, offset = sig[3]
        sig[3] = (kind, size, offset + 4)
        self.assert_abi_rejected(sig)

    def test_changed_kind_rejected(self):
        sig = list(self.described["signature"])
        sig[4] = ("global_buffer", 8, 28)
        self.assert_abi_rejected(sig)

    def test_missing_scratch_slot_rejected(self):
        self.assert_abi_rejected(self.described["signature"][:6])

    def test_extra_non_hidden_argument_rejected(self):
        self.assert_abi_rejected(
            list(self.described["signature"]) + [("global_buffer", 8, 48)]
        )

    def test_wrong_kernel_symbol_rejected(self):
        described = {**self.described, "symbol": "other_kernel"}
        with self.assertRaises(GeneratorError) as raised:
            verify_abi(described, INSTANCE.name, "test")
        self.assertTrue(str(raised.exception).startswith(C.ERR_ABI), raised.exception)
        self.assertIn("other_kernel", str(raised.exception))

    def test_small_kernarg_segment_rejected(self):
        described = {**self.described, "kernarg_segment_size": 40}
        with self.assertRaises(GeneratorError) as raised:
            verify_abi(described, INSTANCE.name, "test")
        self.assertIn(C.ERR_ABI, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
