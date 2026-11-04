# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Binary library I/O shared by generation and packaged-library inspection.

The module has no generator or GPU dependencies. MessagePack is imported only
by binary operations, so YAML-only users of the main LibraryIO need not load it.
"""

import zlib
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

# Keep the writer and readers on the same schema as the native runtime's
# Serialization/SolutionLibrary.hpp. Versions 0/1 use the solutions array.
INDEXED_FORMAT_VERSION = 2

# Native solution indices use a signed 32-bit int. Match the native reader's
# std::numeric_limits<int>::max() check in Serialization/SolutionLibrary.hpp
# so an index cannot overflow into a different solution's key.
_MAX_SOLUTION_INDEX = (1 << 31) - 1


def _readBytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".zlib":
        decoder = zlib.decompressobj()
        try:
            data = decoder.decompress(data) + decoder.flush()
        except zlib.error as error:
            raise ValueError(f"{path}: invalid zlib data: {error}") from error
        if not decoder.eof or decoder.unused_data:
            raise ValueError(f"{path}: incomplete or trailing zlib data")
    return data


def readMessagePack(path: str | Path) -> Any:
    """Read one ordinary or zlib-compressed MessagePack value, without converting its schema.

    Invalid encoded data raises ``ValueError`` with the input path.
    """
    import msgpack

    path = Path(path)
    data = _readBytes(path)
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    except (msgpack.UnpackException, ValueError, TypeError) as error:
        raise ValueError(f"{path}: invalid MessagePack data: {error}") from error


def writeMessagePack(filename: str | Path, data: Any, *, indexed: bool = False) -> None:
    """Write ``filename + '.zlib'`` and remove any stale uncompressed sibling.

    ``indexed=True`` requires a master-library state. Each solution is packed
    separately into a binary blob with an index/offset/length table, allowing a
    reader to decode only requested solutions. The caller's state is unchanged.
    """
    import msgpack

    if indexed:
        if not isinstance(data, dict) or "solutions" not in data or "library" not in data:
            raise ValueError("Indexed MessagePack requires a master-library state")
        packer = msgpack.Packer(use_bin_type=True)
        table, chunks, offset = [], [], 0
        for solution in sorted(data["solutions"], key=lambda s: s["index"]):
            chunk = packer.pack(solution)
            table.extend((solution["index"], offset, len(chunk)))
            chunks.append(chunk)
            offset += len(chunk)
        payload = {"format_version": INDEXED_FORMAT_VERSION}
        if "version" in data:
            payload["version"] = data["version"]
        payload["solutions_index"] = table
        payload["solutions_blob"] = b"".join(chunks)
        payload["library"] = data["library"]
        data = payload

    path = Path(filename)
    compressed = zlib.compress(msgpack.packb(data, use_bin_type=True), 9)
    Path(str(path) + ".zlib").write_bytes(compressed)
    path.unlink(missing_ok=True)


def _solutionIndex(record: dict) -> int:
    index = record.get("index") if isinstance(record, dict) else None
    if type(index) is not int or not 0 <= index <= _MAX_SOLUTION_INDEX:
        raise ValueError(f"invalid solution index {index!r}")
    return index


def _indexedRecords(table, blob, wanted):
    import msgpack

    if not isinstance(table, list) or len(table) % 3:
        raise ValueError("solutions_index must contain index/offset/length triples")
    if not isinstance(blob, bytes):
        raise ValueError("solutions_blob must be binary data")
    view = memoryview(blob)
    for position in range(0, len(table), 3):
        index, offset, length = table[position : position + 3]
        if any(type(value) is not int or value < 0 for value in (index, offset, length)):
            raise ValueError("solutions_index fields must be nonnegative integers")
        if index > _MAX_SOLUTION_INDEX:
            raise ValueError(f"invalid solution index {index!r}")
        if offset > len(blob) or length > len(blob) - offset:
            raise ValueError(f"solution {index}: slice lies outside solutions_blob")
        record = None
        if wanted is None or index in wanted:
            record = msgpack.unpackb(
                view[offset : offset + length], raw=False, strict_map_key=False
            )
            if _solutionIndex(record) != index:
                raise ValueError(f"solution {index}: index does not match its blob record")
        yield index, record


def _solutionRecords(data: bytes, wanted):
    import msgpack

    unpacker = msgpack.Unpacker(raw=False, strict_map_key=False, max_buffer_size=0)
    unpacker.feed(data)
    fields = {}
    has_library = False
    for _ in range(unpacker.read_map_header()):
        key = unpacker.unpack()
        # Match unpackb's rejection of keys that cannot belong to a Python dict.
        hash(key)
        if key in ("solutions", "format_version", "solutions_index", "solutions_blob"):
            if key in fields:
                raise ValueError(f"duplicate {key} field")
            if key == "solutions":
                fields[key] = True
                for _ in range(unpacker.read_array_header()):
                    record = unpacker.unpack()
                    index = _solutionIndex(record)
                    yield index, record if wanted is None or index in wanted else None
            else:
                fields[key] = unpacker.unpack()
        else:
            has_library |= key == "library"
            # Heuristic tables can dwarf the records. Traversing their packed
            # bytes avoids constructing (or decoding) those Python objects.
            unpacker.skip()
    if unpacker.tell() != len(data):
        raise ValueError("trailing MessagePack data")

    version = fields.get("format_version", 0)
    if type(version) is not int or version not in (0, 1, INDEXED_FORMAT_VERSION):
        raise ValueError(f"unsupported library format_version: {version!r}")
    if version == INDEXED_FORMAT_VERSION:
        if "solutions" in fields:
            raise ValueError("indexed library contains a legacy solutions array")
        yield from _indexedRecords(
            fields.get("solutions_index"), fields.get("solutions_blob"), wanted
        )
    elif "solutions_index" in fields or "solutions_blob" in fields:
        raise ValueError("indexed solution fields require format_version 2")
    elif has_library and "solutions" not in fields:
        raise ValueError("library has no solution records")
    # Lazy-loading mapping files contain neither records nor a library tree.


def iterSolutionRecords(
    path: str | Path, solution_indices: Collection[int] | None = None
) -> Iterator[tuple[int, dict | None]]:
    """Yield every index and the selected records from either binary library format.

    Unrequested indices have a ``None`` record. Exposing their indices lets a
    catalog detect collisions across shards without decoding their indexed blob
    entries. Legacy arrays still require decoding each record. At most one shard
    is decompressed at a time; heuristic tables are never materialized. The whole
    file is still read and decompressed, even when selecting a single index.

    This checks container structure and index consistency, not native solution
    predicates. Legacy container errors may appear after records have been yielded.
    Consume the iterator fully to validate the whole container.
    Indexed records outside the selection remain undecoded.
    Invalid encoded data raises ``ValueError`` with the input path.
    """
    import msgpack

    path = Path(path)
    # Catalogs pass the same set to every shard. Reuse its constant-time
    # membership instead of copying the complete selection for each file.
    wanted = solution_indices
    if wanted is not None and not isinstance(wanted, (set, frozenset)):
        wanted = frozenset(wanted)
    data = _readBytes(path)
    seen = set()
    try:
        for index, record in _solutionRecords(data, wanted):
            if index in seen:
                raise ValueError(f"duplicate solution index {index}")
            seen.add(index)
            yield index, record
    except (msgpack.UnpackException, ValueError, TypeError) as error:
        raise ValueError(f"{path}: invalid solution metadata: {error}") from error
