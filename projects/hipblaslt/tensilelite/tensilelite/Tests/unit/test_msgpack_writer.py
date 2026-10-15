# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ writer module, when it has been built."""

import gzip
import zlib
from pathlib import Path

import msgpack
import pytest

from Tensile.LibraryIO import writeMsgPack, writeMsgPackIndexed


def _writer():
    return pytest.importorskip("tensilelite_write")


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


# Keys the C++ loader does not read. They are absent from the rewritten solution.
_DROPPED_KEYS = {"pythonOnlyMarker", "streamK", "CustomKernelName"}


def _compare_solution(before, after, path=""):
    """Fields present on both sides must match.

    Omitted fields come back as C++ defaults, so keys that exist only on the
    written solution are not errors. Keys in _DROPPED_KEYS are not written.
    threadTile may gain a trailing 0, and a TypesEqual list may gain "Float".
    """
    errors = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            if key in _DROPPED_KEYS:
                if key in after:
                    errors.append(f"{path}.{key} was written")
                continue
            if key not in after:
                errors.append(f"dropped {path}.{key}")
                continue
            errors.extend(_compare_solution(value, after[key], f"{path}.{key}"))
        return errors
    if isinstance(before, list) and isinstance(after, list):
        if path.endswith(".threadTile") and after == before + [0]:
            return errors
        if (
            before
            and isinstance(before[0], str)
            and after == before + ["Float"]
        ):
            return errors
        if len(before) != len(after):
            errors.append(f"{path}: length {len(before)} -> {len(after)}")
            return errors
        for index, (left, right) in enumerate(zip(before, after)):
            errors.extend(_compare_solution(left, right, f"{path}[{index}]"))
        return errors
    if before != after:
        errors.append(f"{path}: {before!r} -> {after!r}")
    return errors


def test_write_compares_solution_fields_and_drops_unknown_keys(tmp_path):
    """Each solution is compared with the input. A Python-only field is not written."""
    writer = _writer()
    state = _kernels_lite_state()
    state["solutions"][0]["pythonOnlyMarker"] = {"from": "python"}

    written = msgpack.unpackb(
        writer.write_library(msgpack.packb(state)), raw=False, strict_map_key=False
    )

    by_index = {solution["index"]: solution for solution in written["solutions"]}
    assert set(by_index) == {solution["index"] for solution in state["solutions"]}
    errors = []
    for solution in state["solutions"]:
        errors.extend(_compare_solution(solution, by_index[solution["index"]]))
    assert errors == []
    written_solution = by_index[state["solutions"][0]["index"]]
    assert "pythonOnlyMarker" not in written_solution
    assert "CustomKernelName" not in written_solution["sizeMapping"]
    assert "streamK" not in written_solution["sizeMapping"]
    assert written_solution["kernelName"] == state["solutions"][0]["kernelName"]
