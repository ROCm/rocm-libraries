#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocsparse_gentest.py."""

import unittest

import yaml
from yaml.constructor import ConstructorError

import rocsparse_gentest


class YamlLoaderTest(unittest.TestCase):
    def test_uses_safe_loader(self):
        safe_loaders = (yaml.SafeLoader,)
        if hasattr(yaml, "CSafeLoader"):
            safe_loaders += (yaml.CSafeLoader,)

        self.assertIn(rocsparse_gentest.Loader, safe_loaders)

    def test_rejects_python_object_tag(self):
        loader = rocsparse_gentest.Loader(
            "!!python/object/apply:builtins.str ['unsupported']"
        )
        with self.assertRaises(ConstructorError):
            loader.get_single_data()


if __name__ == "__main__":
    unittest.main()
