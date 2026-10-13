# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LibraryIO MessagePack writes go through the C++ writer when it was built."""

import gzip
import importlib.util
import zlib
from pathlib import Path

import msgpack
import pytest

from Tensile.LibraryIO import writeMsgPack, writeMsgPackIndexed


def _writer():
    if importlib.util.find_spec("tensilelite.tensilelite_write") is None:
        pytest.skip("tensilelite_write extension was not built")
    from tensilelite import tensilelite_write

    return tensilelite_write


def _kernels_lite_state():
    fixture = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "configs"
        / "SolutionLibraries"
        / "KernelsLite.dat.gz"
    )
    raw = gzip.decompress(fixture.read_bytes())
    state = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    assert isinstance(state, dict)
    assert "solutions" in state and "library" in state
    return state


def _unpack_zlib(path):
    return zlib.decompress(path.read_bytes())


def test_library_io_msgpack_writer_documents_load(tmp_path):
    writer = _writer()
    state = _kernels_lite_state()

    eager_base = tmp_path / "eager.dat"
    writeMsgPack(str(eager_base), state)
    eager_raw = _unpack_zlib(tmp_path / "eager.dat.zlib")
    eager = msgpack.unpackb(eager_raw, raw=False, strict_map_key=False)
    assert "solutions" in eager
    assert "library" in eager
    assert "format_version" not in eager
    assert "solutions_index" not in eager
    # The module loads the document it just wrote.
    assert writer.write_library(eager_raw)

    indexed_base = tmp_path / "indexed.dat"
    writeMsgPackIndexed(str(indexed_base), state)
    indexed_raw = _unpack_zlib(tmp_path / "indexed.dat.zlib")
    indexed = msgpack.unpackb(indexed_raw, raw=False, strict_map_key=False)
    assert indexed["format_version"] == 2
    assert isinstance(indexed["solutions_blob"], (bytes, bytearray))
    assert len(indexed["solutions_index"]) % 3 == 0
    assert len(indexed["solutions_index"]) > 0
    assert "solutions" not in indexed
    keys = list(indexed)
    if "version" in indexed:
        assert keys == [
            "format_version",
            "version",
            "solutions_index",
            "solutions_blob",
            "library",
        ]
    else:
        assert keys == [
            "format_version",
            "solutions_index",
            "solutions_blob",
            "library",
        ]
    # Loading the indexed document is the C++ reader's job. write_library does
    # that load, then emits the eager form.
    rewritten = writer.write_library(indexed_raw)
    rewritten_doc = msgpack.unpackb(rewritten, raw=False, strict_map_key=False)
    assert "solutions" in rewritten_doc
    assert "format_version" not in rewritten_doc
    assert "solutions_blob" not in rewritten_doc
