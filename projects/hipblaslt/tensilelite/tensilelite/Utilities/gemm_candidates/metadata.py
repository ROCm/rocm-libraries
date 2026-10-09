# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read packaged solution records once, without constructing heuristic tables."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import msgpack

from ..LibraryIO import iterSolutionRecords


@dataclass(frozen=True, slots=True)
class _Solution:
    index: int
    name: str
    kernel_name: str
    metadata_file: Path
    problem: dict
    predicate: dict
    mapping: dict
    code_objects: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class LibraryMetadata:
    """An artifact's solution lookup, reusable across calls to ``candidate_gemms``.

    ``indices`` contains the loaded indices in numeric order. Loading a subset
    retains only those records; querying another index raises ``KeyError``.
    No GPU, code objects or kernel-generator extension is needed to load it.
    """

    directory: Path
    target: str
    indices: tuple[int, ...]
    _solutions: Mapping[int, _Solution] = field(repr=False)


def load_library_metadata(
    library_dir: str | Path,
    *,
    target: str,
    solution_indices: Iterable[int] | None = None,
) -> LibraryMetadata:
    """Load an index-to-solution lookup from ``*.dat`` / ``*.dat.zlib`` files.

    ``library_dir`` may be the architecture directory or its parent. Compressed
    shards take precedence over an uncompressed copy, matching native loading.
    Indices are local to this library: aliases with distinct indices are kept,
    while duplicate indices across shards are errors.

    Pass ``solution_indices`` when probing a small part of a large library to
    reduce retained memory. All shards are scanned once; subsequent candidate
    calls perform no file I/O. An empty iterable loads an empty lookup.
    """
    directory = Path(library_dir).resolve()
    if (directory / target).is_dir():
        directory /= target
    wanted = None if solution_indices is None else set(solution_indices)
    if wanted is not None and any(type(index) is not int or index < 0 for index in wanted):
        raise ValueError("Solution indices must be nonnegative integers")

    shards = {
        path.name.removesuffix(".zlib"): path
        for path in sorted(directory.glob(f"*_{target}.dat*"))
        if path.name.endswith((".dat", ".dat.zlib"))
    }
    if not shards:
        raise ValueError(f"No {target} solution metadata in {directory}")

    # Flat installations may contain several architectures. Untagged helper
    # objects are shared; tagged objects must match the requested architecture.
    code_objects = tuple(
        path
        for path in sorted((*directory.glob("*.co"), *directory.glob("*.hsaco")))
        if not (arches := re.findall(r"gfx[0-9a-f]+", path.name)) or target in arches
    )
    object_set = set(code_objects)
    helpers = tuple(path for path in code_objects if path.suffix == ".hsaco")
    records, seen = {}, set()
    shared = {}

    def intern(value):
        # Many indices share a problem type and predicates. These private
        # records are read-only; keeping one copy reduces catalog memory.
        return shared.setdefault(msgpack.packb(value, use_bin_type=True), value)

    for path in shards.values():
        main_object = path.with_name(path.name.removesuffix(".zlib").removesuffix(".dat") + ".co")
        # Lazy shards have a matching code object; monolithic metadata may need
        # all objects in its architecture directory. Metadata-only use is valid.
        objects = (main_object, *helpers) if main_object in object_set else code_objects
        for index, record in iterSolutionRecords(path, wanted):
            if index in seen:
                raise ValueError(f"{path}: duplicate solution index {index}")
            seen.add(index)
            if record is not None:
                mapping = record["sizeMapping"]
                records[index] = _Solution(
                    index=index,
                    name=record["name"],
                    kernel_name=record["kernelName"],
                    metadata_file=path,
                    problem=intern(record["problemType"]),
                    predicate=intern(record["problemPredicate"]),
                    mapping={
                        "macroTile": mapping["macroTile"],
                        "depthU": mapping["depthU"],
                        "globalSplitU": mapping["globalSplitU"],
                        "tileProcessingStrategy": mapping.get("tileProcessingStrategy"),
                    },
                    code_objects=objects,
                )
    if wanted is not None and wanted - seen:
        raise KeyError(f"Solution indices not found: {sorted(wanted - seen)}")
    return LibraryMetadata(directory, target, tuple(sorted(records)), MappingProxyType(records))
