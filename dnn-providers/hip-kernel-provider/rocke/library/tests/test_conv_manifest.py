# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The convolution vertical registers its own manifest runners.

The deep-fused conv/pool manifest kinds used to be built into
``rocke.run_manifest``. They moved out with the kernels: their buffer packing
is kernel knowledge, and the platform SDK must not import ``kernels``. The
platform's registry exposes ``register_manifest_runner`` for exactly this, and
``kernels.manifest`` uses it at ``import kernels`` time.

The platform-side counterpart (``platform/tests/test_manifest_runner_registry.py``)
asserts these kinds are NOT built in; this asserts importing the vertical adds
them back, so the pair pins both halves of the seam.
"""

from __future__ import annotations

import unittest

from rocke.run_manifest import registered_manifest_kinds

from kernels.manifest import CONV_MANIFEST_KINDS, register


class TestConvManifestRegistration(unittest.TestCase):
    def test_importing_kernels_registers_the_conv_kinds(self):
        # `import kernels` already ran via the import above.
        kinds = registered_manifest_kinds()
        for kind in CONV_MANIFEST_KINDS:
            self.assertIn(kind, kinds)

    def test_registration_is_idempotent(self):
        before = registered_manifest_kinds()
        register()
        register()
        self.assertEqual(before, registered_manifest_kinds())

    def test_platform_conv_kinds_are_still_platform_owned(self):
        # The plain conv_* manifest runner is pure buffer marshalling and
        # stays in the platform; only the kernel-coupled ones moved.
        kinds = registered_manifest_kinds()
        for kind in ("conv_fp16", "conv_bf16", "conv_fp32"):
            self.assertIn(kind, kinds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
