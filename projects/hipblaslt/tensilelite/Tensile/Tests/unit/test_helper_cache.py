################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest

pytestmark = pytest.mark.unit

import hashlib
import os
from collections import namedtuple
from pathlib import Path

MockVersion = namedtuple("MockVersion", ["major", "minor", "patch"])


class MockCompiler:
    def __init__(self, version=(6, 0, 0), rocm_version=(6, 0, 0), asan=False):
        self.version = MockVersion(*version)
        self.rocm_version = MockVersion(*rocm_version)
        self.default_args = ["amdclang++", "-O3"]
        if asan:
            self.default_args.append("-fsanitize=address")


def _write_test_files(tmp_path, cpp_content="void f(){}", h_content="#pragma once"):
    """Create minimal source + header files for cache key tests."""
    (tmp_path / "Kernels.cpp").write_text(cpp_content)
    (tmp_path / "Kernels.h").write_text(h_content)
    for name in [
        "KernelHeader.h", "TensileTypes.h", "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h", "ReductionTemplate.h", "memory_gfx.h",
    ]:
        (tmp_path / name).write_text(f"// {name}")
    return tmp_path / "Kernels.cpp"


class TestComputeCacheKey:
    def test_deterministic(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex digest

    def test_different_source_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path, cpp_content="void f(){}")
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "Kernels.cpp").write_text("void g(){}")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2

    def test_different_arch_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100"], compiler)
        assert k1 != k2

    def test_arch_order_irrelevant(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942", "gfx1100"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100", "gfx942"], compiler)
        assert k1 == k2

    def test_different_compiler_version_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 1, 0)))
        assert k1 != k2

    def test_different_rocm_version_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 1, 0)))
        assert k1 != k2

    def test_asan_changes_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=False))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=True))
        assert k1 != k2

    def test_static_header_change_changes_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "TensileTypes.h").write_text("// modified")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2
