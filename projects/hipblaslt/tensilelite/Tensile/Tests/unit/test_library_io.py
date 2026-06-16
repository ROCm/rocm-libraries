################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""Unit tests for Tensile.LibraryIO.writeMsgPack."""

import zlib
import msgpack
import pytest

from Tensile.LibraryIO import writeMsgPack


def test_writeMsgPack_produces_zlib_file(tmp_path):
    """writeMsgPack writes <filename>.zlib, not <filename>."""
    dest = str(tmp_path / "library.dat")
    data = {"key": "value", "count": 42}

    writeMsgPack(dest, data)

    assert not (tmp_path / "library.dat").exists()
    assert (tmp_path / "library.dat.zlib").exists()


def test_writeMsgPack_roundtrips_data(tmp_path):
    """Content decompresses and unpacks to the original data."""
    dest = str(tmp_path / "library.dat")
    data = {"kernels": ["k0", "k1", "k2"], "version": 3}

    writeMsgPack(dest, data)

    raw = zlib.decompress((tmp_path / "library.dat.zlib").read_bytes())
    assert msgpack.unpackb(raw) == data


def test_writeMsgPack_uses_zlib_compression(tmp_path):
    """Output is valid zlib (not raw msgpack)."""
    dest = str(tmp_path / "library.dat")
    writeMsgPack(dest, {"x": list(range(100))})

    gz_bytes = (tmp_path / "library.dat.zlib").read_bytes()
    # zlib.decompress raises if the bytes are not valid zlib
    decompressed = zlib.decompress(gz_bytes)
    assert len(decompressed) > 0
