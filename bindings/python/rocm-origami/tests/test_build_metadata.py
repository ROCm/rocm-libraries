# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check SDK dependency and version metadata without compiling the extension."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "build_metadata", Path(__file__).resolve().parents[1] / "build_metadata.py"
)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


class TestBuildMetadata(unittest.TestCase):
    def test_system_build_has_no_sdk_dependency_or_device_extras(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(metadata.dynamic_metadata("version", {}), "0.1.0")
            self.assertEqual(metadata.dynamic_metadata("dependencies", {}), [])
            self.assertEqual(
                set(metadata.dynamic_metadata("optional-dependencies", {})),
                {"test", "torch"},
            )

    def test_sdk_build_pins_sdk_and_forwards_device_extras(self):
        with patch.dict(
            os.environ,
            {
                "ROCM_SDK_VERSION": "10.3.0a20261123+bkc.20261123",
                "ROCM_SDK_TARGETS": "gfx950,gfx942,gfx950",
            },
            clear=True,
        ):
            self.assertEqual(
                metadata.dynamic_metadata("version", {}),
                "0.1.0+rocm10.3.0a20261123.bkc.20261123",
            )
            self.assertEqual(
                metadata.dynamic_metadata("dependencies", {}),
                ["rocm[libraries]==10.3.0a20261123+bkc.20261123"],
            )
            extras = metadata.dynamic_metadata("optional-dependencies", {})
            self.assertEqual(
                extras["device-gfx942"],
                ["rocm[device-gfx942]==10.3.0a20261123+bkc.20261123"],
            )
            self.assertEqual(len(extras), 4)

    def test_invalid_sdk_version_fails(self):
        with patch.dict(os.environ, {"ROCM_SDK_VERSION": "unknown"}, clear=True):
            with self.assertRaises(ValueError):
                metadata.dynamic_metadata("version", {})

    def test_sdk_build_requires_valid_target_input(self):
        for target in ["", "../gfx942", "gfx942;gfx950"]:
            with self.subTest(target=target), patch.dict(
                os.environ,
                {"ROCM_SDK_VERSION": "10.3.0", "ROCM_SDK_TARGETS": target},
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    metadata.dynamic_metadata("optional-dependencies", {})


if __name__ == "__main__":
    unittest.main()
