# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Non-zero global/profile scratch is rejected, naming the instance and both sizes.
The sizes are injected into otherwise real metadata."""

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators import _contract as C  # noqa: E402
from generators import _triton_env as env  # noqa: E402
from generators import gen_rmsnorm  # noqa: E402
from generators._codeobject import GeneratorError, check_scratch  # noqa: E402

INSTANCE = C.instances()[0]


class ScratchRejectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env.prepare()
        cls.compiled = gen_rmsnorm.compile_rmsnorm(
            env.load_kernel(), "gfx942", INSTANCE
        )

    def assert_scratch_rejected(self, global_size, profile_size):
        injected = dataclasses.replace(
            self.compiled,
            global_scratch_size=global_size,
            profile_scratch_size=profile_size,
        )
        with self.assertRaises(GeneratorError) as raised:
            gen_rmsnorm.verify_compiled(injected, INSTANCE, "gfx942")
        message = str(raised.exception)
        self.assertTrue(message.startswith(C.ERR_SCRATCH), message)
        self.assertIn(INSTANCE.name, message)
        self.assertIn(f"global_scratch_size={global_size}", message)
        self.assertIn(f"profile_scratch_size={profile_size}", message)

    def test_global_scratch_rejected(self):
        self.assert_scratch_rejected(128, 0)

    def test_profile_scratch_rejected(self):
        self.assert_scratch_rejected(0, 64)

    def test_both_rejected(self):
        self.assert_scratch_rejected(128, 64)

    def test_zero_scratch_accepted(self):
        check_scratch(INSTANCE.name, 0, 0)
        self.assertEqual(
            (self.compiled.global_scratch_size, self.compiled.profile_scratch_size),
            (0, 0),
        )


if __name__ == "__main__":
    unittest.main()
