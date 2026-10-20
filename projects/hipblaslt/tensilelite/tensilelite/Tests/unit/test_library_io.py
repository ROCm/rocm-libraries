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
"""Binary library writers and shared readers, including selective indexed decoding."""

import copy
import os
import subprocess
import sys
import zlib
from pathlib import Path

import msgpack
import pytest
from tensilelite.LibraryIO import read, write, writeMsgPack, writeMsgPackIndexed
from tensilelite.Utilities.BinaryLibraryIO import (
    iterSolutionRecords,
    readMessagePack,
    writeMessagePack,
)

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
def test_read_binary_library_preserves_its_schema(tmp_path, library_format):
    """General LibraryIO.read exposes raw metadata in both compressed and plain files."""
    base = tmp_path / "library"
    write(str(base), _masterLibraryState(), library_format)
    compressed = base.with_suffix(".dat.zlib")
    plain = base.with_suffix(".dat")
    plain.write_bytes(zlib.decompress(compressed.read_bytes()))
    expected = msgpack.unpackb(plain.read_bytes(), raw=False, strict_map_key=False)
    assert read(compressed) == expected
    assert read(plain) == expected


def test_indexed_filter_does_not_decode_unrequested_records(tmp_path):
    """A corrupt unrequested blob entry proves selection avoids eager record decoding."""
    path = tmp_path / "library.dat"
    writeMsgPackIndexed(str(path), _masterLibraryState())
    payload = readMessagePack(str(path) + ".zlib")
    table = payload["solutions_index"]
    assert table[0] == 0
    skipped_index, offset, length = table[3:6]
    blob = payload["solutions_blob"]
    payload["solutions_blob"] = blob[:offset] + b"\xc1" * length + blob[offset + length :]
    writeMsgPack(str(path), payload)

    records = dict(iterSolutionRecords(str(path) + ".zlib", [0]))
    assert set(records) == {0, 2, 4, 7, 9}
    assert records[0]["kernelName"] == "kernel_0"
    assert all(record is None for index, record in records.items() if index != 0)
    with pytest.raises(ValueError, match="invalid solution metadata"):
        list(iterSolutionRecords(str(path) + ".zlib", [skipped_index]))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"format_version": 3}, "unsupported library format_version"),
        ({"solutions_index": [17, 0]}, "triples"),
        ({"solutions_index": None}, "triples"),
        ({"solutions_blob": []}, "binary data"),
        ({"solutions_index": [True, 0, 1]}, "nonnegative integers"),
        ({"solutions_index": [-1, 0, 1]}, "nonnegative integers"),
        ({"solutions_index": [2**31, 0, 1]}, "invalid solution index"),
        ({"solutions_index": [17, -1, 1]}, "nonnegative integers"),
        ({"solutions_index": [17, 1_000_000, 1]}, "outside solutions_blob"),
        ({"solutions_index": [17, 0, 1_000_000]}, "outside solutions_blob"),
        ({"solutions_blob": msgpack.packb({"index": 18})}, "does not match"),
    ],
)
def test_indexed_metadata_rejects_invalid_containers(tmp_path, change, message):
    path = tmp_path / "library.dat"
    writeMsgPackIndexed(str(path), {"solutions": [{"index": 17}], "library": {}})
    payload = readMessagePack(str(path) + ".zlib")
    payload.update(change)
    writeMsgPack(str(path), payload)
    with pytest.raises(ValueError, match=message):
        list(iterSolutionRecords(str(path) + ".zlib"))


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
def test_duplicate_indices_are_checked_even_outside_selection(tmp_path, library_format):
    base = tmp_path / "library"
    write(str(base), {"solutions": [{"index": 17}, {"index": 17}], "library": {}}, library_format)
    with pytest.raises(ValueError, match="duplicate solution index 17"):
        list(iterSolutionRecords(base.with_suffix(".dat.zlib"), []))


@pytest.mark.parametrize("library_format", ["msgpack", "msgpack-indexed"])
def test_solution_reader_skips_heuristic_tables_without_decoding(tmp_path, library_format):
    base = tmp_path / "library"
    write(str(base), _masterLibraryState(), library_format)
    payload = readMessagePack(base.with_suffix(".dat.zlib"))
    packer = msgpack.Packer(use_bin_type=True)
    # Invalid UTF-8 inside a table fails if the reader materializes that table.
    data = packer.pack_map_header(len(payload)) + b"".join(
        packer.pack(key) + (b"\x91\xa1\xff" if key == "library" else packer.pack(value))
        for key, value in payload.items()
    )
    path = base.with_suffix(".dat")
    path.write_bytes(data)
    expected = {record["index"]: record for record in _masterLibraryState()["solutions"]}
    assert dict(iterSolutionRecords(path)) == expected
    with pytest.raises(ValueError, match="invalid MessagePack data"):
        readMessagePack(path)


@pytest.mark.parametrize("reader", [readMessagePack, iterSolutionRecords])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("corruption", ["truncated", "trailing", "invalid"])
def test_binary_readers_reject_corrupt_files(tmp_path, reader, compressed, corruption):
    path = tmp_path / ("library.dat.zlib" if compressed else "library.dat")
    data = msgpack.packb(_masterLibraryState(), use_bin_type=True)
    if compressed:
        data = zlib.compress(data)
    data = {"truncated": data[:-1], "trailing": data + b"\x00", "invalid": b"\xc1"}[corruption]
    path.write_bytes(data)
    with pytest.raises(ValueError, match="library.dat"):
        list(reader(path))


@pytest.mark.parametrize("reader", [readMessagePack, iterSolutionRecords])
def test_binary_readers_reject_zlib_checksum_mismatch(tmp_path, reader):
    """A complete stream with a damaged checksum must not supply unchecked metadata."""
    path = tmp_path / "library.dat.zlib"
    data = bytearray(zlib.compress(msgpack.packb(_masterLibraryState(), use_bin_type=True)))
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid zlib data"):
        list(reader(path))


@pytest.mark.parametrize(
    "reader, location",
    [
        (readMessagePack, "container"),
        (readMessagePack, "legacy"),
        (iterSolutionRecords, "container"),
        (iterSolutionRecords, "legacy"),
        (iterSolutionRecords, "indexed"),
    ],
)
@pytest.mark.parametrize("key", [[], {}])
def test_binary_readers_reject_unhashable_map_keys(tmp_path, reader, location, key):
    """Unsupported map keys must produce a filename-qualified metadata error."""
    packer = msgpack.Packer(use_bin_type=True)
    invalid_map = packer.pack_map_header(2) + (
        packer.pack("index") + packer.pack(0) + packer.pack(key) + packer.pack(0)
    )
    if location == "container":
        data = invalid_map
    elif location == "legacy":
        data = (
            packer.pack_map_header(2)
            + packer.pack("solutions")
            + packer.pack_array_header(1)
            + invalid_map
            + packer.pack("library")
            + packer.pack({})
        )
    else:
        data = packer.pack(
            {
                "format_version": 2,
                "solutions_index": [0, 0, len(invalid_map)],
                "solutions_blob": invalid_map,
                "library": {},
            }
        )
    path = tmp_path / "library.dat"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="library.dat: invalid"):
        list(reader(path))


@pytest.mark.parametrize("index", [True, -1, 2**31, 1.0, None])
def test_legacy_reader_validates_indices_outside_selection(tmp_path, index):
    path = tmp_path / "library.dat"
    writeMsgPack(str(path), {"solutions": [{"index": index}], "library": {}})
    with pytest.raises(ValueError, match="invalid solution index"):
        list(iterSolutionRecords(str(path) + ".zlib", []))


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"format_version": True, "solutions": []}, "unsupported library format_version"),
        ({"format_version": "2", "solutions": []}, "unsupported library format_version"),
        ({"format_version": 2, "solutions": [], "solutions_index": [],
          "solutions_blob": b""}, "indexed library contains a legacy solutions array"),
        ({"solutions_index": [], "solutions_blob": b""},
         "indexed solution fields require format_version 2"),
        ({"library": {}}, "library has no solution records"),
    ],
)
def test_solution_reader_rejects_inconsistent_schema(tmp_path, payload, message):
    """Reject incompatible field combinations even when no records are requested."""
    path = tmp_path / "library.dat"
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    with pytest.raises(ValueError, match=message):
        list(iterSolutionRecords(path, []))


@pytest.mark.parametrize("field, value", [("solutions", []), ("format_version", 0)])
def test_solution_reader_rejects_duplicate_schema_fields(tmp_path, field, value):
    """A Python dict cannot represent repeated keys; encode the ambiguous map directly."""
    packer = msgpack.Packer(use_bin_type=True)
    path = tmp_path / "library.dat"
    path.write_bytes(packer.pack_map_header(2) + (packer.pack(field) + packer.pack(value)) * 2)
    with pytest.raises(ValueError, match=f"duplicate {field} field"):
        list(iterSolutionRecords(path))


@pytest.mark.parametrize("version", [0, 1])
def test_solution_reader_accepts_legacy_versions_and_lazy_mapping(tmp_path, version):
    """Empty legacy libraries and record-free lazy mappings are valid containers."""
    path = tmp_path / "library.dat"
    for payload in ({"format_version": version, "solutions": [], "library": {}},
                    {"0": "shard_a", "17": "shard_b"}):
        path.write_bytes(msgpack.packb(payload, use_bin_type=True))
        assert list(iterSolutionRecords(path)) == []


@pytest.mark.parametrize("payload", [[], {"solutions": []}, {"library": {}}])
def test_shared_indexed_writer_rejects_incomplete_master_state(tmp_path, payload):
    """Standalone callers get ValueError and no file for a non-master state."""
    path = tmp_path / "library.dat"
    with pytest.raises(ValueError, match="requires a master-library state"):
        writeMessagePack(path, payload, indexed=True)
    assert not path.with_suffix(".dat.zlib").exists()


def test_shared_metadata_utilities_work_without_generator_imports(tmp_path):
    """Use both utilities in a fresh interpreter with generator imports forbidden."""
    # A fresh interpreter keeps pytest's already-imported generator modules from
    # hiding dependencies from the import guard. As in test_namespace_bridge.py,
    # pass a self-contained script to Python's -c option.
    script = """
import importlib.abc
import sys
from pathlib import Path

class NoGenerator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"rocisa", "Tensile"} or fullname in {
            "tensilelite.Common", "tensilelite.ClientWriter", "tensilelite.LibraryIO",
            "tensilelite.Contractions", "tensilelite.SolutionStructs",
        }:
            raise AssertionError(f"Unexpected generator import: {fullname}")

sys.meta_path.insert(0, NoGenerator())
from tensilelite.Utilities.DataTypes import TYPE_PROPERTIES
from tensilelite.Utilities.BinaryLibraryIO import iterSolutionRecords, readMessagePack, writeMessagePack

assert "msgpack" not in sys.modules
half = TYPE_PROPERTIES["Half"]
assert (half.char, half.reg, half.hip) == ("H", 0.5, "tensile_half")
try:
    TYPE_PROPERTIES["Half"] = half
except TypeError:
    pass
else:
    raise AssertionError("Datatype registry must be immutable")
try:
    half.reg = 1
except AttributeError:
    pass
else:
    raise AssertionError("Datatype properties must be immutable")

record = {"index": 17, "kernelName": "kernel_17"}
state = {"solutions": [record, {"index": 9}], "library": {"type": "Matching"}}
path = Path(sys.argv[1]) / "library.dat"
for indexed in (False, True):
    writeMessagePack(path, state, indexed=indexed)
    compressed = path.with_suffix(".dat.zlib")
    assert not path.exists()
    assert readMessagePack(compressed)["library"] == state["library"]
    assert dict(iterSolutionRecords(compressed, [17])) == {9: None, 17: record}
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
