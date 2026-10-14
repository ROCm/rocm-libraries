# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Non-zero global/profile scratch is rejected, naming the instance and both sizes.
The sizes are injected into a committed object's otherwise real metadata."""

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_generator_contract import committed_compiled  # noqa: E402
from tritongen import _family as F  # noqa: E402
from tritongen import gen  # noqa: E402
from tritongen._family import GeneratorError  # noqa: E402
from tritongen.ops import OPS  # noqa: E402


class ScratchRejectionTest(unittest.TestCase):
    def assert_scratch_rejected(self, global_size, profile_size):
        for op in OPS.values():
            arch, instance = op.ARCHES[0], op.instances()[0]
            with self.subTest(op=op.OP):
                injected = dataclasses.replace(
                    committed_compiled(op, arch, instance),
                    global_scratch_size=global_size,
                    profile_scratch_size=profile_size,
                )
                with self.assertRaises(GeneratorError) as raised:
                    gen.verify_compiled(op, injected, instance, arch)
                message = str(raised.exception)
                self.assertTrue(message.startswith(F.ERR_SCRATCH), message)
                self.assertIn(instance.name, message)
                self.assertIn(f"global_scratch_size={global_size}", message)
                self.assertIn(f"profile_scratch_size={profile_size}", message)

    def test_global_scratch_rejected(self):
        self.assert_scratch_rejected(128, 0)

    def test_profile_scratch_rejected(self):
        self.assert_scratch_rejected(0, 64)

    def test_both_rejected(self):
        self.assert_scratch_rejected(128, 64)


if __name__ == "__main__":
    unittest.main()
