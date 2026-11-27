# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prefer the selected generator packages, including in spawned Python workers.

Editable-install finders precede Python's normal path lookup and can override
PYTHONPATH. This hook is added only to the generator command's Python path.
Unrelated dependencies continue to use the interpreter's normal environment.
"""

from importlib.machinery import PathFinder
import os
import sys


class GeneratorPackages:
    def __init__(self, python_root, native_root):
        self.roots = {
            "tensilelite": python_root,
            "_tensilelite_client_binding": python_root,
            "tensilelite_configure_client": python_root,
            "rocisa": native_root,
        }

    def find_spec(self, fullname, path=None, target=None):
        root = self.roots.get(fullname.partition(".")[0])
        if root is None:
            return None
        spec = PathFinder.find_spec(fullname, path if path is not None else [root])
        if spec is None:
            raise ModuleNotFoundError(f"Selected generator does not provide {fullname}")
        return spec


sys.meta_path.insert(
    0,
    GeneratorPackages(
        os.environ["TENSILELITE_GENERATOR_PYTHON_ROOT"],
        os.environ["TENSILELITE_GENERATOR_NATIVE_ROOT"],
    ),
)
