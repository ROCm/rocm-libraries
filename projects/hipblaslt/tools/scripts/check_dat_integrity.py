#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Post-build integrity checks for installed hipBLASLt Tensile .dat/.dat.zlib files."""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path
from typing import List

try:
    import msgpack
except ImportError:
    msgpack = None

_MSGPACK_ERRORS = (msgpack.exceptions.UnpackException,) if msgpack is not None else ()

_MASTER_RE = re.compile(r"^TensileLibrary_lazy_(?P<arch>[A-Za-z0-9]+)\.dat(?:\.zlib)?$")
_MAPPING_RE = re.compile(
    r"^TensileLiteLibrary_lazy_(?P<arch>[A-Za-z0-9]+)_Mapping\.dat(?:\.zlib)?$"
)


class _MappingLoadError(Exception):
    pass


def _scanSubtrees(libDir: Path):
    """Every gfx* subtree, with the arch tokens its masters and Mappings carry.
    Keyed on the directory, not the token: library/gfx1250/ and library/gfx1250v0/
    each hold a complete set of files all named for gfx1250 (shared compiler
    target), so merging by token would check one subtree twice and the other not
    at all.
    """
    subtrees = {}
    for sub in sorted(libDir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("gfx"):
            continue
        masters, mappings = set(), set()
        for f in sub.iterdir():
            if m := _MASTER_RE.match(f.name):
                masters.add(m.group("arch"))
            elif m := _MAPPING_RE.match(f.name):
                mappings.add(m.group("arch"))
        if masters or mappings:
            subtrees[sub] = (masters, mappings)
    return subtrees


def _strictDecompress(data: bytes) -> bytes:
    """Decompress a single zlib stream, rejecting any trailing bytes.

    Mirrors the C++ loader (readCompressedMsgObject), which treats leftover
    input after the zlib stream as corruption. One-shot zlib.decompress
    silently ignores such trailing bytes, so the integrity check would
    otherwise pass files the runtime loader rejects.
    """
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(data)
    raw += decompressor.flush()
    if not decompressor.eof:
        raise zlib.error("incomplete zlib stream")
    if decompressor.unused_data:
        raise zlib.error(
            f"trailing bytes after zlib stream: {decompressor.unused_data!r}"
        )
    return raw


def _loadMapping(archDir: Path, arch: str, where: str):
    base = archDir / f"TensileLiteLibrary_lazy_{arch}_Mapping.dat"
    gz_path = Path(str(base) + ".zlib")
    src = gz_path if gz_path.is_file() else base
    try:
        if src == gz_path:
            raw = _strictDecompress(src.read_bytes())
            return msgpack.unpackb(raw, raw=False, strict_map_key=False)
        with open(src, "rb") as f:
            return msgpack.unpack(f, raw=False, strict_map_key=False)
    except (OSError, ValueError, zlib.error, *_MSGPACK_ERRORS) as exc:
        raise _MappingLoadError(
            f"{where}: failed to read/decode Mapping ({src.name}): {exc}"
        ) from exc


def _validateSubtree(archDir: Path, masters, mappings) -> List[str]:
    """Check one subtree against itself. Shards are resolved inside this
    directory, which is what the runtime does once it has selected it."""
    violations: List[str] = []
    where = archDir.name
    archs = masters & mappings

    for a in sorted(masters - archs):
        violations.append(f"{where}: master without a per-arch Mapping: {a}")
    for a in sorted(mappings - archs):
        violations.append(f"{where}: per-arch Mapping without a master: {a}")

    fallback_dat_files = list(archDir.glob("*_fallback_*.dat")) + list(
        archDir.glob("*_fallback_*.dat.zlib")
    )
    for arch in sorted(archs):
        try:
            mapping = _loadMapping(archDir, arch, where)
        except _MappingLoadError as exc:
            violations.append(str(exc))
            continue

        for idx, kernelName in mapping.items():
            if not (archDir / f"{kernelName}.dat").is_file() and not (
                archDir / f"{kernelName}.dat.zlib"
            ).is_file():
                violations.append(
                    f"{where}: arch={arch}: Mapping[{idx}] -> {kernelName}.dat(.zlib) "
                    "is not on disk"
                )
            if not kernelName.endswith("_" + arch):
                violations.append(
                    f"{where}: arch={arch}: Mapping[{idx}] -> {kernelName} is not "
                    "arch-scoped (kpack overlay collision risk)"
                )

        arch_has_fallback_files = any(
            f.name.endswith(f"_fallback_{arch}.dat")
            or f.name.endswith(f"_fallback_{arch}.dat.zlib")
            for f in fallback_dat_files
        )
        mapping_has_fallback = any(
            "_fallback_" in n and n.endswith("_" + arch) for n in mapping.values()
        )
        if arch_has_fallback_files and not mapping_has_fallback:
            violations.append(
                f"{where}: arch={arch}: per-arch Mapping is missing fallback entries "
                "(runtime may report 'NO solution found!' for fallback-only dtypes)"
            )

    return violations


def validate(libDir: Path) -> List[str]:
    libDir = Path(libDir).resolve()
    if not libDir.is_dir():
        return [f"library dir does not exist or is not a directory: {libDir}"]
    if msgpack is None:
        return ["msgpack is required to read Tensile .dat files but is not installed"]

    subtrees = _scanSubtrees(libDir)
    if not any(masters & mappings for masters, mappings in subtrees.values()):
        return [
            f"{libDir} contains no matched (master, Mapping) pair; runtime "
            "cannot resolve lazy lookups"
        ]

    violations: List[str] = []
    for archDir, (masters, mappings) in subtrees.items():
        violations.extend(_validateSubtree(archDir, masters, mappings))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "library_dir",
        type=Path,
        help="Installed library dir containing <base-arch>/ subdirs of .dat/.dat.zlib files",
    )
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args(argv)

    violations = validate(args.library_dir)
    if violations:
        print(
            f"[check_dat_integrity] {len(violations)} violation(s) in {args.library_dir}:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"[check_dat_integrity] OK: {args.library_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
