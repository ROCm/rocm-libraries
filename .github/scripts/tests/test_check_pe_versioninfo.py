# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import check_pe_versioninfo as checker

SECTION_RVA = 0x1000
SECTION_FILE_OFFSET = 0x200


def _pad4(blob: bytes) -> bytes:
    return blob + b"\0" * (-len(blob) % 4)


def _block(
    key: str, value: bytes, value_length: int, value_type: int, children=()
) -> bytes:
    """Encode one VS_VERSIONINFO-style block the way rc.exe lays it out."""
    body = _pad4(b"\0" * 6 + key.encode("utf-16-le") + b"\0\0")
    if value:
        body = _pad4(body + value)
    for child in children:
        body = _pad4(body + child)
    return struct.pack("<HHH", len(body), value_length, value_type) + body[6:]


def _string(key: str, value: str) -> bytes:
    text = value.encode("utf-16-le") + b"\0\0"
    return _block(key, text, len(text) // 2, 1)


def version_resource(strings: dict[str, str], version=(1, 5, 0, 0)) -> bytes:
    major, minor, patch, build = version
    fixed = struct.pack(
        "<13I",
        checker.VS_FIXEDFILEINFO_SIGNATURE,
        0x00010000,
        (major << 16) | minor,
        (patch << 16) | build,
        (major << 16) | minor,
        (patch << 16) | build,
        0x3F,
        0,
        0x40004,
        2,
        0,
        0,
        0,
    )
    table = _block("040904B0", b"", 0, 1, [_string(k, v) for k, v in strings.items()])
    string_info = _block("StringFileInfo", b"", 0, 1, [table])
    var_info = _block(
        "VarFileInfo",
        b"",
        0,
        1,
        [_block("Translation", struct.pack("<HH", 0x409, 1200), 4, 0)],
    )
    return _block("VS_VERSION_INFO", fixed, len(fixed), 0, [string_info, var_info])


def resource_section(version_blob: bytes) -> bytes:
    """Build a .rsrc section with RT_VERSION -> id 1 -> language 0x409 -> data."""

    def directory(entries: list[tuple[int, int]]) -> bytes:
        header = struct.pack("<IIHHHH", 0, 0, 0, 0, 0, len(entries))
        return header + b"".join(struct.pack("<II", i, o) for i, o in entries)

    name_dir_offset = 24
    lang_dir_offset = 48
    data_entry_offset = 72
    blob_offset = 88
    section = (
        directory([(checker.RT_VERSION, 0x80000000 | name_dir_offset)])
        + directory([(1, 0x80000000 | lang_dir_offset)])
        + directory([(0x409, data_entry_offset)])
        + struct.pack("<IIII", SECTION_RVA + blob_offset, len(version_blob), 0, 0)
        + version_blob
    )
    assert section.index(version_blob) == blob_offset
    return section


def pe_image(resources: bytes | None, pe32_plus: bool = True) -> bytes:
    """Build a minimal PE image with one section holding the resource directory."""
    pe_offset = 0x40
    dos = b"MZ" + b"\0" * 0x3A + struct.pack("<I", pe_offset)
    directories = [(0, 0)] * 16
    section_data = resources or b"\0" * 16
    if resources is not None:
        directories[checker.RESOURCE_DIRECTORY_INDEX] = (SECTION_RVA, len(resources))
    data_dirs = b"".join(struct.pack("<II", rva, size) for rva, size in directories)
    if pe32_plus:
        optional = (
            struct.pack("<H", 0x20B) + b"\0" * 106 + struct.pack("<I", 16) + data_dirs
        )
    else:
        optional = (
            struct.pack("<H", 0x10B) + b"\0" * 90 + struct.pack("<I", 16) + data_dirs
        )
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, len(optional), 0x2022)
    section_header = b".rsrc\0\0\0" + struct.pack(
        "<IIIIIIHHI",
        len(section_data),
        SECTION_RVA,
        len(section_data),
        SECTION_FILE_OFFSET,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    headers = dos + b"PE\0\0" + coff + optional + section_header
    assert len(headers) <= SECTION_FILE_OFFSET
    return headers.ljust(SECTION_FILE_OFFSET, b"\0") + section_data


STRINGS = {
    "CompanyName": "Advanced Micro Devices, Inc.",
    "FileDescription": "AMD hipBLASLt Benchmark Client",
    "Comments": "Git commit: 0123abcd",
}


class _TreeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path


class ReadVersionInfoTest(_TreeTestCase):
    def test_reads_version_and_strings(self):
        path = self.write(
            "bin/a.dll", pe_image(resource_section(version_resource(STRINGS)))
        )
        info = checker.read_version_info(path)
        self.assertEqual(info.file_version, "1.5.0.0")
        self.assertEqual(info.strings, STRINGS)

    def test_reads_pe32_image(self):
        blob = resource_section(version_resource(STRINGS, version=(2, 3, 4, 5)))
        path = self.write("bin/a.exe", pe_image(blob, pe32_plus=False))
        self.assertEqual(checker.read_version_info(path).file_version, "2.3.4.5")

    def test_image_without_resources_has_no_version_info(self):
        path = self.write("bin/a.dll", pe_image(None))
        self.assertIsNone(checker.read_version_info(path))

    def test_non_pe_file_is_rejected(self):
        path = self.write("bin/a.dll", b"not a PE file at all" * 8)
        with self.assertRaises(checker.PEFormatError):
            checker.read_version_info(path)


class RunTest(_TreeTestCase):
    def test_passes_when_required_binaries_have_version_info(self):
        self.write(
            "bin/libhipblaslt.dll",
            pe_image(resource_section(version_resource(STRINGS))),
        )
        exit_code, report = checker.run(self.root, ["bin/*hipblaslt*.dll"], [])
        self.assertEqual(exit_code, 0)
        self.assertIn("Git commit: 0123abcd", report)

    def test_fails_when_required_binary_lacks_version_info(self):
        self.write(
            "bin/libhipblaslt.dll",
            pe_image(resource_section(version_resource(STRINGS))),
        )
        self.write("bin/hipblaslt-bench.exe", pe_image(None))
        exit_code, report = checker.run(
            self.root, ["bin/*hipblaslt*.dll", "bin/hipblaslt-*.exe"], []
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("without VERSIONINFO: `bin/hipblaslt-bench.exe`", report)

    def test_report_only_binaries_never_fail(self):
        self.write(
            "bin/libhipblaslt.dll",
            pe_image(resource_section(version_resource(STRINGS))),
        )
        self.write("libexec/hipblaslt-samples/sample.exe", pe_image(None))
        exit_code, report = checker.run(
            self.root, ["bin/*hipblaslt*.dll"], ["libexec/hipblaslt-samples/*.exe"]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn(
            "| `libexec/hipblaslt-samples/sample.exe` | report-only | MISSING |", report
        )

    def test_required_pattern_matching_nothing_fails(self):
        self.write(
            "bin/libhipblaslt.dll",
            pe_image(resource_section(version_resource(STRINGS))),
        )
        exit_code, report = checker.run(
            self.root, ["bin/*hipblaslt*.dll", "bin/origami*.dll"], []
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("matched no file: `bin/origami*.dll`", report)

    def test_unmatched_binaries_are_ignored(self):
        self.write("bin/librocblas.dll", pe_image(None))
        self.write(
            "bin/libhipblaslt.dll",
            pe_image(resource_section(version_resource(STRINGS))),
        )
        exit_code, report = checker.run(self.root, ["bin/*hipblaslt*.dll"], [])
        self.assertEqual(exit_code, 0)
        self.assertNotIn("rocblas", report)


if __name__ == "__main__":
    unittest.main()
