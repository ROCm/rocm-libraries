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

import copy
import zlib
import msgpack
import pytest

from Tensile.LibraryIO import write, writeMsgPack, writeMsgPackIndexed


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


####################
# writeMsgPackIndexed
####################

def _masterLibraryState(count=5):
    """A MasterSolutionLibrary.state()-shaped dict with `count` solutions.

    Indices are deliberately unsorted and non-contiguous so tests catch a
    writer that assumes either.
    """
    order = [7, 2, 9, 0, 4][:count]
    return {
        "solutions": [
            {"index": i, "kernelName": f"kernel_{i}", "sizeMapping": {"depthU": i}}
            for i in order
        ],
        "library": {"type": "Matching", "table": [{"key": [1], "index": i} for i in order]},
    }


def _readIndexed(path):
    return msgpack.unpackb(zlib.decompress(path.read_bytes()),
                           raw=False, strict_map_key=False)


def test_writeMsgPackIndexed_produces_zlib_file(tmp_path):
    """Same artifact contract as writeMsgPack: .dat.zlib, no bare .dat."""
    dest = str(tmp_path / "library.dat")

    writeMsgPackIndexed(dest, _masterLibraryState())

    assert (tmp_path / "library.dat.zlib").exists()
    assert not (tmp_path / "library.dat").exists()


def test_writeMsgPackIndexed_removes_stale_uncompressed(tmp_path):
    """A leftover uncompressed .dat must not shadow the new artifact."""
    dat = tmp_path / "library.dat"
    dat.write_bytes(b"stale")

    writeMsgPackIndexed(str(dat), _masterLibraryState())

    assert not dat.exists()
    assert (tmp_path / "library.dat.zlib").exists()


def test_writeMsgPackIndexed_emits_expected_schema(tmp_path):
    """format_version 2 plus the index/blob pair, and no legacy solutions key."""
    dest = str(tmp_path / "library.dat")

    writeMsgPackIndexed(dest, _masterLibraryState())
    indexed = _readIndexed(tmp_path / "library.dat.zlib")

    assert indexed["format_version"] == 2
    assert "solutions" not in indexed
    assert isinstance(indexed["solutions_blob"], (bytes, bytearray))
    assert len(indexed["solutions_index"]) % 3 == 0


def test_writeMsgPackIndexed_key_order_is_fixed(tmp_path):
    """Fixed key order keeps the artifact reproducible across runs."""
    dest = str(tmp_path / "library.dat")

    data = _masterLibraryState()
    data["version"] = "5"
    writeMsgPackIndexed(dest, data)

    raw = zlib.decompress((tmp_path / "library.dat.zlib").read_bytes())
    keys = list(msgpack.unpackb(raw, raw=False, strict_map_key=False))
    assert keys == ["format_version", "version", "solutions_index",
                    "solutions_blob", "library"]


def test_writeMsgPackIndexed_roundtrips_every_solution(tmp_path):
    """Each slice decodes back to the solution the table says it holds."""
    dest = str(tmp_path / "library.dat")
    data = _masterLibraryState()

    writeMsgPackIndexed(dest, data)
    indexed = _readIndexed(tmp_path / "library.dat.zlib")

    table = indexed["solutions_index"]
    blob = indexed["solutions_blob"]
    rebuilt = {}
    for i in range(0, len(table), 3):
        index, offset, length = table[i], table[i + 1], table[i + 2]
        solution = msgpack.unpackb(blob[offset:offset + length],
                                   raw=False, strict_map_key=False)
        assert solution["index"] == index
        rebuilt[index] = solution

    assert rebuilt == {s["index"]: s for s in data["solutions"]}


def test_writeMsgPackIndexed_slices_tile_the_blob_exactly(tmp_path):
    """No gaps, overlaps, or trailing bytes -- a decoder can trust the table."""
    dest = str(tmp_path / "library.dat")

    writeMsgPackIndexed(dest, _masterLibraryState())
    indexed = _readIndexed(tmp_path / "library.dat.zlib")

    table = indexed["solutions_index"]
    spans = sorted((table[i + 1], table[i + 2]) for i in range(0, len(table), 3))
    cursor = 0
    for offset, length in spans:
        assert offset == cursor
        cursor += length
    assert cursor == len(indexed["solutions_blob"])


def test_writeMsgPackIndexed_table_is_sorted_by_index(tmp_path):
    """Sorted output makes the artifact byte-reproducible regardless of the
    order state() happened to emit."""
    dest = str(tmp_path / "library.dat")

    writeMsgPackIndexed(dest, _masterLibraryState())
    table = _readIndexed(tmp_path / "library.dat.zlib")["solutions_index"]

    indices = [table[i] for i in range(0, len(table), 3)]
    assert indices == sorted(indices)


def test_writeMsgPackIndexed_preserves_library_tree(tmp_path):
    """The decision tree is copied through untouched."""
    dest = str(tmp_path / "library.dat")
    data = _masterLibraryState()

    writeMsgPackIndexed(dest, data)
    indexed = _readIndexed(tmp_path / "library.dat.zlib")

    assert indexed["library"] == data["library"]


def test_writeMsgPackIndexed_omits_absent_version(tmp_path):
    """`version` is optional and must not be invented."""
    dest = str(tmp_path / "library.dat")

    writeMsgPackIndexed(dest, _masterLibraryState())

    assert "version" not in _readIndexed(tmp_path / "library.dat.zlib")


def test_writeMsgPackIndexed_does_not_mutate_caller_dict(tmp_path):
    """Run.py reuses the state dict, so splitting must not consume it."""
    dest = str(tmp_path / "library.dat")
    data = _masterLibraryState()
    before = copy.deepcopy(data)

    writeMsgPackIndexed(dest, data)

    assert data == before


def test_writeMsgPackIndexed_rejects_non_master_payload(tmp_path):
    """The per-arch lazy mapping file is a flat {index: name} dict. Writing it
    in indexed form would silently break lazy loading, so it must be refused
    rather than mangled."""
    dest = str(tmp_path / "mapping.dat")

    with pytest.raises(SystemExit):
        writeMsgPackIndexed(dest, {0: "shard_a", 1: "shard_b"})


def test_write_dispatches_msgpack_indexed(tmp_path):
    """write() routes the new format name to the indexed writer."""
    base = str(tmp_path / "library")

    write(base, _masterLibraryState(), format="msgpack-indexed")

    assert (tmp_path / "library.dat.zlib").exists()
    assert _readIndexed(tmp_path / "library.dat.zlib")["format_version"] == 2


def test_write_msgpack_stays_legacy(tmp_path):
    """The plain msgpack route must keep emitting the eager layout: mapping
    files and older runtimes depend on it."""
    base = str(tmp_path / "library")

    write(base, _masterLibraryState(), format="msgpack")

    written = _readIndexed(tmp_path / "library.dat.zlib")
    assert "solutions" in written
    assert "format_version" not in written
