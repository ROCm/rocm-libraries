################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""
Unit tests for the 64-bit byte offset fix in KernelWriterConversion.

Validates that buffer_store calls in the conversion kernel use a pre-computed
uint64_t byteOffsetD instead of inline 32-bit index multiplication, preventing
address overflow for large tensors (>2GB).
"""
import re
import os
import pytest


TENSILE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


class TestKernelWriterConversion64BitOffset:
    """Verify KernelWriterConversion uses uint64_t byteOffsetD for buffer_store."""

    @pytest.fixture(autouse=True)
    def load_source(self):
        src_path = os.path.join(TENSILE_ROOT, "KernelWriterConversion.py")
        with open(src_path, "r") as f:
            self.source = f.read()

    def test_byteOffsetD_declared_as_uint64(self):
        """byteOffsetD must be declared with self.uint64Str type."""
        pattern = (
            r'kStr\s*\+=\s*".*%s byteOffsetD\s*=\s*idxD\s*\*\s*sizeof\(%s\)'
            r'.*"\s*%\s*\(\s*self\.uint64Str\s*,\s*destTypeStr'
        )
        assert re.search(pattern, self.source), \
            "Expected byteOffsetD declaration to use self.uint64Str"

    def test_buffer_store_uses_byteOffsetD(self):
        """All buffer_store calls in the store section must use byteOffsetD."""
        store_lines = [
            line for line in self.source.splitlines()
            if "buffer_store" in line and "kStr" in line and "byteOffsetD" in line
        ]
        assert len(store_lines) >= 3, \
            f"Expected at least 3 buffer_store calls using byteOffsetD, found {len(store_lines)}"

    def test_no_inline_idxD_multiply_in_buffer_store(self):
        """buffer_store must not use inline 'idxD * sizeof(...)' anymore."""
        store_lines = [
            line for line in self.source.splitlines()
            if "buffer_store" in line and "kStr" in line
        ]
        for line in store_lines:
            assert not re.search(r"idxD\s*\*\s*sizeof", line), \
                f"Found deprecated inline idxD*sizeof in buffer_store: {line.strip()}"


class TestMemoryGfxSplitBufferOffset:
    """Verify memory_gfx.h contains the splitBufferOffset helper and 64-bit overloads."""

    @pytest.fixture(autouse=True)
    def load_header(self):
        header_path = os.path.join(TENSILE_ROOT, "Source", "memory_gfx.h")
        with open(header_path, "r") as f:
            self.header = f.read()

    def test_splitBufferOffset_exists(self):
        """splitBufferOffset helper function must exist."""
        assert "splitBufferOffset" in self.header

    def test_splitBufferOffset_signature(self):
        """splitBufferOffset must accept base_ptr, uint64_t voffset, and uint32_t& voffset_lo."""
        pattern = r'splitBufferOffset\s*\(\s*void\s+const\s*\*\s*base_ptr\s*,\s*uint64_t\s+voffset\s*,\s*uint32_t\s*&\s*voffset_lo\s*\)'
        assert re.search(pattern, self.header), \
            "splitBufferOffset signature mismatch"

    def test_splitBufferOffset_masks_high_bits(self):
        """splitBufferOffset must mask off upper 32 bits from voffset into base pointer."""
        assert "0xFFFFFFFF00000000ull" in self.header

    def test_buffer_store_uint64_overloads_exist(self):
        """buffer_store specializations must have uint64_t voffset overload."""
        overload_pattern = r'buffer_store\(.*?uint64_t\s+voffset'
        matches = re.findall(overload_pattern, self.header, re.DOTALL)
        assert len(matches) >= 5, \
            f"Expected at least 5 buffer_store overloads with uint64_t voffset, found {len(matches)}"

    def test_overloads_call_splitBufferOffset(self):
        """Each uint64_t overload must call splitBufferOffset to split the offset."""
        assert self.header.count("splitBufferOffset(base_ptr, voffset, voffset_lo)") >= 5
