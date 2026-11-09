#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check that Windows PE binaries carry an embedded VERSIONINFO resource.

Scans a directory tree (for example flattened TheRock Windows artifacts) for
.dll/.exe files matching the given glob patterns and reports the version
strings each one embeds. Binaries matching --require patterns fail the check
when they have no VERSIONINFO resource; binaries matching --report-only
patterns are listed but never fail it. Every --require pattern must match at
least one file, so a layout change cannot turn the check into a silent pass.

Uses only the standard library, so it runs on any host OS.

Example:
  python .github/scripts/check_pe_versioninfo.py build/artifacts \\
    --require 'bin/*hipblaslt*.dll' 'bin/hipblaslt-*.exe' \\
    --report-only 'libexec/hipblaslt-samples/*.exe'
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

RT_VERSION = 16
RESOURCE_DIRECTORY_INDEX = 2
VS_FIXEDFILEINFO_SIGNATURE = 0xFEEF04BD
PE_SUFFIXES = (".dll", ".exe")


class PEFormatError(ValueError):
    """Raised when a file is not a well-formed PE image."""


@dataclass
class VersionInfo:
    file_version: str
    strings: dict[str, str] = field(default_factory=dict)


def _align4(offset: int) -> int:
    return (offset + 3) & ~3


def _read_utf16z(data: bytes, offset: int, end: int) -> tuple[str, int]:
    """Read a NUL-terminated UTF-16LE string; return it and the offset past the NUL."""
    pos = offset
    while pos + 1 < end and data[pos : pos + 2] != b"\0\0":
        pos += 2
    return data[offset:pos].decode("utf-16-le", errors="replace"), pos + 2


def _rva_to_offset(rva: int, sections: list[tuple[int, int, int, int]]) -> int:
    for virtual_address, virtual_size, raw_size, raw_pointer in sections:
        if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
            return rva - virtual_address + raw_pointer
    raise PEFormatError(f"RVA 0x{rva:x} is not inside any section")


def _resource_entries(data: bytes, directory: int) -> list[tuple[int, int, bool]]:
    """Return (id_or_name, offset, is_subdirectory) for one IMAGE_RESOURCE_DIRECTORY."""
    named, ids = struct.unpack_from("<HH", data, directory + 12)
    entries = []
    for index in range(named + ids):
        name, target = struct.unpack_from("<II", data, directory + 16 + 8 * index)
        entries.append((name, target & 0x7FFFFFFF, bool(target & 0x80000000)))
    return entries


def _find_version_resource(data: bytes) -> bytes | None:
    """Return the raw RT_VERSION resource bytes, or None when the image has none."""
    if data[:2] != b"MZ":
        raise PEFormatError("missing MZ header")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise PEFormatError("missing PE signature")
    coff = pe_offset + 4
    section_count, optional_size = struct.unpack_from("<H12xH", data, coff + 2)
    optional = coff + 20
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x10B:
        rva_count_offset, directories_offset = 92, 96
    elif magic == 0x20B:
        rva_count_offset, directories_offset = 108, 112
    else:
        raise PEFormatError(f"unknown optional header magic 0x{magic:x}")
    rva_count = struct.unpack_from("<I", data, optional + rva_count_offset)[0]
    if rva_count <= RESOURCE_DIRECTORY_INDEX:
        return None
    resource_rva, resource_size = struct.unpack_from(
        "<II", data, optional + directories_offset + 8 * RESOURCE_DIRECTORY_INDEX
    )
    if resource_rva == 0 or resource_size == 0:
        return None

    sections = []
    section_table = optional + optional_size
    for index in range(section_count):
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
            "<IIII", data, section_table + 40 * index + 8
        )
        sections.append((virtual_address, virtual_size, raw_size, raw_pointer))
    root = _rva_to_offset(resource_rva, sections)

    for type_id, type_offset, type_is_dir in _resource_entries(data, root):
        if type_id != RT_VERSION or not type_is_dir:
            continue
        for _, name_offset, name_is_dir in _resource_entries(data, root + type_offset):
            if not name_is_dir:
                continue
            for _, lang_offset, lang_is_dir in _resource_entries(
                data, root + name_offset
            ):
                if lang_is_dir:
                    continue
                data_rva, data_size = struct.unpack_from(
                    "<II", data, root + lang_offset
                )
                start = _rva_to_offset(data_rva, sections)
                return data[start : start + data_size]
    return None


def _parse_block(data: bytes, offset: int) -> tuple[str, int, int, int, int]:
    """Parse a version block header; return (key, value_offset, value_length, value_type, end)."""
    length, value_length, value_type = struct.unpack_from("<HHH", data, offset)
    end = min(offset + length, len(data))
    key, after_key = _read_utf16z(data, offset + 6, end)
    return key, _align4(after_key), value_length, value_type, end


def _children(data: bytes, offset: int, end: int):
    while offset + 6 <= end:
        length = struct.unpack_from("<H", data, offset)[0]
        if length == 0:
            return
        yield offset
        offset = _align4(offset + length)


def parse_version_resource(resource: bytes) -> VersionInfo:
    """Decode a VS_VERSIONINFO blob into its FILEVERSION and StringFileInfo values."""
    key, value_offset, value_length, _, end = _parse_block(resource, 0)
    if key != "VS_VERSION_INFO" or value_length < 52:
        raise PEFormatError("malformed VS_VERSIONINFO header")
    signature, _, ms, ls = struct.unpack_from("<IIII", resource, value_offset)
    if signature != VS_FIXEDFILEINFO_SIGNATURE:
        raise PEFormatError("bad VS_FIXEDFILEINFO signature")
    info = VersionInfo(f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}")

    for child in _children(resource, _align4(value_offset + value_length), end):
        child_key, child_value, _, _, child_end = _parse_block(resource, child)
        if child_key != "StringFileInfo":
            continue
        for table in _children(resource, child_value, child_end):
            _, table_value, _, _, table_end = _parse_block(resource, table)
            for entry in _children(resource, table_value, table_end):
                name, value_start, _, _, entry_end = _parse_block(resource, entry)
                value = resource[value_start:entry_end].decode(
                    "utf-16-le", errors="replace"
                )
                info.strings[name] = value.split("\0", 1)[0]
    return info


def read_version_info(path: Path) -> VersionInfo | None:
    """Return the VERSIONINFO embedded in a PE file, or None when it has none."""
    data = path.read_bytes()
    try:
        resource = _find_version_resource(data)
    except struct.error as error:
        raise PEFormatError(f"truncated PE image: {error}") from error
    if resource is None:
        return None
    try:
        return parse_version_resource(resource)
    except struct.error as error:
        raise PEFormatError(f"truncated VERSIONINFO: {error}") from error


def _matches(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def run(root: Path, require: list[str], report_only: list[str]) -> tuple[int, str]:
    """Scan root and return (exit code, Markdown report)."""
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in PE_SUFFIXES
    )
    failures = []
    rows = []
    matched_patterns = set()
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        required = _matches(relative, require)
        if not required and not _matches(relative, report_only):
            continue
        matched_patterns.update(p for p in require if fnmatch.fnmatchcase(relative, p))
        mode = "required" if required else "report-only"
        try:
            info = read_version_info(path)
            status = "ok" if info else "MISSING"
            detail = (
                f"{info.file_version}; {info.strings.get('Comments', '(no Comments)')}"
                if info
                else ""
            )
        except PEFormatError as error:
            info = None
            status = "UNREADABLE"
            detail = str(error)
        if info is None and required:
            failures.append(relative)
        rows.append(f"| `{relative}` | {mode} | {status} | {detail} |")

    unmatched = [p for p in require if p not in matched_patterns]
    lines = [
        "## Windows VERSIONINFO check",
        "",
        "| Binary | Mode | VERSIONINFO | Version; Comments |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
    ]
    if unmatched:
        lines.append(
            "Required patterns that matched no file: "
            + ", ".join(f"`{p}`" for p in unmatched)
        )
    if failures:
        lines.append(
            "Required binaries without VERSIONINFO: "
            + ", ".join(f"`{f}`" for f in failures)
        )
    if not unmatched and not failures:
        lines.append("All required binaries carry VERSIONINFO.")
    return (1 if unmatched or failures else 0), "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="Directory tree to scan")
    parser.add_argument(
        "--require",
        nargs="+",
        default=[],
        help="Glob patterns (relative to root, '/'-separated) for binaries that must carry VERSIONINFO",
    )
    parser.add_argument(
        "--report-only",
        nargs="+",
        default=[],
        help="Glob patterns for binaries that are reported but never fail the check",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"{args.root} is not a directory")

    exit_code, report = run(args.root, args.require, args.report_only)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
