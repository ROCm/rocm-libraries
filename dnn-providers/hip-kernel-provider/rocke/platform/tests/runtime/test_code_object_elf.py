# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Loader-visible HSACO semantic signatures."""

from __future__ import annotations

import struct

import pytest
from rocke.runtime.code_object_elf import (
    CodeObjectInspectionError,
    inspect_hsaco_semantics,
)


def _put(fmt: str, data: bytearray, offset: int, *values) -> None:
    struct.pack_into(fmt, data, offset, *values)


def _synthetic_hsaco(kernel_name: str = "copy") -> bytes:
    msgpack = pytest.importorskip("msgpack")
    metadata = msgpack.packb(
        {
            "amdhsa.version": [1, 2],
            "amdhsa.kernels": [
                {
                    ".name": kernel_name,
                    ".symbol": f"{kernel_name}.kd",
                    ".kernarg_segment_size": 24,
                    ".kernarg_segment_align": 8,
                    ".group_segment_fixed_size": 0,
                    ".private_segment_fixed_size": 0,
                    ".wavefront_size": 64,
                    ".sgpr_count": 8,
                    ".vgpr_count": 4,
                    ".max_flat_workgroup_size": 64,
                    ".args": [
                        {
                            ".name": "X",
                            ".size": 8,
                            ".offset": 0,
                            ".value_kind": "global_buffer",
                            ".address_space": "global",
                        },
                        {
                            ".name": "N",
                            ".size": 4,
                            ".offset": 16,
                            ".value_kind": "by_value",
                            ".value_type": "i32",
                        },
                    ],
                }
            ],
        },
        use_bin_type=True,
    )
    name = b"AMDGPU\0"
    note_size = 12 + ((len(name) + 3) & ~3) + ((len(metadata) + 3) & ~3)
    note_offset = 64
    section_offset = note_offset + note_size
    elf = bytearray(section_offset + 2 * 64)
    elf[:4] = b"\x7fELF"
    elf[4:9] = bytes([2, 1, 1, 0, 5])
    _put("<HH", elf, 16, 3, 224)
    _put("<Q", elf, 0x28, section_offset)
    _put("<I", elf, 0x30, 0x42)
    _put("<HH", elf, 0x3A, 64, 2)
    _put("<III", elf, note_offset, len(name), len(metadata), 32)
    cursor = note_offset + 12
    elf[cursor : cursor + len(name)] = name
    cursor += (len(name) + 3) & ~3
    elf[cursor : cursor + len(metadata)] = metadata
    note_header = section_offset + 64
    _put("<I", elf, note_header + 4, 7)
    _put("<QQ", elf, note_header + 0x18, note_offset, note_size)
    return bytes(elf)


def test_semantic_signature_covers_target_kernel_and_arguments() -> None:
    semantics = inspect_hsaco_semantics(_synthetic_hsaco())
    assert semantics.abi_version == 5
    assert semantics.flags == 0x42
    assert semantics.metadata_version == (1, 2)
    kernel = dict(semantics.kernels[0])
    assert kernel[".name"] == "copy"
    assert dict(kernel[".args"][0])[".value_kind"] == "global_buffer"
    assert dict(kernel[".args"][1])[".offset"] == 16


def test_semantic_signature_detects_kernel_abi_change() -> None:
    assert inspect_hsaco_semantics(
        _synthetic_hsaco("first")
    ) != inspect_hsaco_semantics(_synthetic_hsaco("second"))


def test_semantic_signature_rejects_non_amdgpu_elf() -> None:
    blob = bytearray(_synthetic_hsaco())
    struct.pack_into("<H", blob, 18, 62)
    with pytest.raises(CodeObjectInspectionError, match="type/machine"):
        inspect_hsaco_semantics(bytes(blob))
