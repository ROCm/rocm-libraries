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


def test_writeMsgPack_removes_stale_uncompressed(tmp_path):
    """A pre-existing uncompressed .dat is deleted so it cannot shadow the .zlib."""
    dat = tmp_path / "library.dat"
    dat.write_bytes(b"stale uncompressed payload from a previous build")

    writeMsgPack(str(dat), {"key": "value"})

    assert not dat.exists()
    assert (tmp_path / "library.dat.zlib").exists()


def test_writeMsgPack_missing_stale_uncompressed_is_noop(tmp_path):
    """Absence of an uncompressed sibling is not an error."""
    dest = str(tmp_path / "library.dat")

    writeMsgPack(dest, {"key": "value"})

    assert (tmp_path / "library.dat.zlib").exists()


def test_writeMsgPack_reader_contract_is_zlib_wrapped_msgpack(tmp_path):
    """The on-disk format the C++ loader depends on: zlib(msgpack(data)).

    This guards the producer side of the Python-writer -> C++-reader contract:
    the C++ ``readCompressedMsgObject`` inflates the file then msgpack-parses
    the result, so the writer must emit exactly that, for non-trivial nested
    data, with no extra framing.
    """
    dest = str(tmp_path / "library.dat")
    data = {
        "0": "TensileLibrary_gfx942_kernels_fallback_gfx942_0",
        "10": "TensileLibrary_gfx942_kernels_fallback_gfx942_10",
        "nested": {"a": [1, 2, 3], "b": {"c": "d"}},
    }

    writeMsgPack(dest, data)

    raw = zlib.decompress((tmp_path / "library.dat.zlib").read_bytes())
    assert msgpack.unpackb(raw, raw=False, strict_map_key=False) == data


def test_writeMsgPack_empty_mapping_roundtrips(tmp_path):
    """Corner case: an empty mapping still produces a loadable .zlib."""
    dest = str(tmp_path / "library.dat")

    writeMsgPack(dest, {})

    raw = zlib.decompress((tmp_path / "library.dat.zlib").read_bytes())
    assert msgpack.unpackb(raw, raw=False, strict_map_key=False) == {}
