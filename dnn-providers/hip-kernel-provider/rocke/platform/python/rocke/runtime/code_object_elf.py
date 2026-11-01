# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Semantic inspection of AMDGPU HSA code objects.

Compiler backends need not emit byte-identical ELF files. This module extracts
the loader-visible target and kernel ABI fields used by differential gates.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = [
    "CodeObjectInspectionError",
    "CodeObjectSemantics",
    "inspect_hsaco_semantics",
]

_ET_DYN = 3
_EM_AMDGPU = 224
_SHT_SYMTAB = 2
_SHT_NOTE = 7
_SHT_DYNSYM = 11
_NT_AMDGPU_METADATA = 32

_KERNEL_FIELDS = (
    ".name",
    ".symbol",
    ".kernarg_segment_size",
    ".kernarg_segment_align",
    ".group_segment_fixed_size",
    ".private_segment_fixed_size",
    ".wavefront_size",
    ".sgpr_count",
    ".vgpr_count",
    ".agpr_count",
    ".max_flat_workgroup_size",
    ".uses_dynamic_stack",
)
_ARG_FIELDS = (
    ".name",
    ".type_name",
    ".size",
    ".offset",
    ".value_kind",
    ".value_type",
    ".address_space",
    ".access",
    ".actual_access",
    ".is_const",
    ".is_restrict",
    ".is_volatile",
)


class CodeObjectInspectionError(ValueError):
    """The bytes are not a supported loadable AMDGPU code object."""


@dataclass(frozen=True)
class CodeObjectSemantics:
    """Target and kernel ABI facts that must agree across compiler backends."""

    abi_version: int
    flags: int
    metadata_version: tuple[int, ...]
    kernels: tuple[tuple[tuple[str, object], ...], ...]
    exported_symbols: tuple[str, ...]


def _unpack(fmt: str, data: bytes, offset: int, what: str):
    try:
        return struct.unpack_from(fmt, data, offset)
    except struct.error as error:
        raise CodeObjectInspectionError(f"{what} is truncated: {error}") from error


def _metadata_payloads(elf: bytes) -> list[bytes]:
    if len(elf) < 64 or elf[:4] != b"\x7fELF":
        raise CodeObjectInspectionError("HSACO is not an ELF image")
    if elf[4] != 2 or elf[5] != 1:
        raise CodeObjectInspectionError(
            f"HSACO class/data is {(elf[4], elf[5])}, expected ELF64 little-endian"
        )
    elf_type, machine = _unpack("<HH", elf, 16, "ELF header")
    if elf_type != _ET_DYN or machine != _EM_AMDGPU:
        raise CodeObjectInspectionError(
            f"HSACO type/machine is {(elf_type, machine)}, "
            f"expected {(_ET_DYN, _EM_AMDGPU)}"
        )
    (section_offset,) = _unpack("<Q", elf, 0x28, "ELF section table")
    section_size, section_count = _unpack("<HH", elf, 0x3A, "ELF section table")
    if section_size < 64:
        raise CodeObjectInspectionError(
            f"ELF section header size is {section_size}, expected at least 64"
        )
    payloads = []
    for index in range(section_count):
        header = section_offset + index * section_size
        (section_type,) = _unpack("<I", elf, header + 4, "ELF section header")
        if section_type != _SHT_NOTE:
            continue
        offset, size = _unpack("<QQ", elf, header + 0x18, "ELF note section")
        end = offset + size
        if end > len(elf):
            raise CodeObjectInspectionError("ELF note section extends past the image")
        cursor = offset
        while cursor < end:
            namesz, descsz, note_type = _unpack("<III", elf, cursor, "ELF note header")
            cursor += 12
            name = elf[cursor : cursor + namesz].rstrip(b"\0")
            cursor += (namesz + 3) & ~3
            payload = elf[cursor : cursor + descsz]
            if len(payload) != descsz:
                raise CodeObjectInspectionError("ELF note payload is truncated")
            cursor += (descsz + 3) & ~3
            if note_type == _NT_AMDGPU_METADATA and name == b"AMDGPU":
                payloads.append(payload)
    return payloads


def _exported_symbols(elf: bytes) -> tuple[str, ...]:
    (section_offset,) = _unpack("<Q", elf, 0x28, "ELF section table")
    section_size, section_count = _unpack("<HH", elf, 0x3A, "ELF section table")
    names = set()
    for index in range(section_count):
        header = section_offset + index * section_size
        (section_type,) = _unpack("<I", elf, header + 4, "ELF section header")
        if section_type not in (_SHT_SYMTAB, _SHT_DYNSYM):
            continue
        offset, size = _unpack("<QQ", elf, header + 0x18, "ELF symbol table")
        (string_index,) = _unpack("<I", elf, header + 0x28, "ELF symbol strings")
        (entry_size,) = _unpack("<Q", elf, header + 0x38, "ELF symbol table")
        if entry_size < 24 or size % entry_size:
            raise CodeObjectInspectionError(
                f"ELF symbol table has invalid size/stride {(size, entry_size)}"
            )
        string_header = section_offset + string_index * section_size
        string_offset, string_size = _unpack(
            "<QQ", elf, string_header + 0x18, "ELF string table"
        )
        strings = elf[string_offset : string_offset + string_size]
        if len(strings) != string_size:
            raise CodeObjectInspectionError("ELF string table is truncated")
        for symbol_offset in range(offset, offset + size, entry_size):
            (name_offset,) = _unpack("<I", elf, symbol_offset, "ELF symbol")
            info = elf[symbol_offset + 4] if symbol_offset + 4 < len(elf) else 0
            (section_index,) = _unpack("<H", elf, symbol_offset + 6, "ELF symbol")
            if not name_offset or info >> 4 == 0 or section_index == 0:
                continue
            if name_offset >= len(strings):
                raise CodeObjectInspectionError(
                    f"ELF symbol string offset {name_offset} is out of range"
                )
            raw_name = strings[name_offset:].split(b"\0", 1)[0]
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CodeObjectInspectionError(
                    f"ELF symbol is not UTF-8: {error}"
                ) from error
            if name:
                names.add(name)
    return tuple(sorted(names))


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _kernel_signature(kernel: dict) -> tuple[tuple[str, object], ...]:
    fields = [
        (field, _freeze(kernel[field])) for field in _KERNEL_FIELDS if field in kernel
    ]
    args = []
    for argument in kernel.get(".args", ()):
        args.append(
            tuple(
                (field, _freeze(argument[field]))
                for field in _ARG_FIELDS
                if field in argument
            )
        )
    fields.append((".args", tuple(args)))
    return tuple(fields)


def inspect_hsaco_semantics(elf: bytes) -> CodeObjectSemantics:
    """Return loader-visible target and kernel ABI facts from one HSACO."""
    payloads = _metadata_payloads(elf)
    if len(payloads) != 1:
        raise CodeObjectInspectionError(
            f"HSACO has {len(payloads)} AMDGPU metadata notes, expected exactly 1"
        )
    try:
        import msgpack
    except ImportError as error:
        raise CodeObjectInspectionError(
            "msgpack is required to inspect AMDGPU metadata"
        ) from error
    try:
        metadata = msgpack.unpackb(payloads[0], raw=False, strict_map_key=False)
    except (ValueError, TypeError) as error:
        raise CodeObjectInspectionError(
            f"AMDGPU metadata is not valid MessagePack: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise CodeObjectInspectionError("AMDGPU metadata root is not a map")
    kernels = metadata.get("amdhsa.kernels")
    if not isinstance(kernels, list):
        raise CodeObjectInspectionError("AMDGPU metadata has no amdhsa.kernels list")
    signatures = tuple(
        sorted(
            (_kernel_signature(kernel) for kernel in kernels),
            key=lambda signature: dict(signature).get(".name", ""),
        )
    )
    version = metadata.get("amdhsa.version", ())
    if not isinstance(version, (list, tuple)):
        raise CodeObjectInspectionError(
            f"AMDGPU metadata version has unexpected shape {version!r}"
        )
    (flags,) = _unpack("<I", elf, 0x30, "ELF flags")
    return CodeObjectSemantics(
        abi_version=elf[8],
        flags=flags,
        metadata_version=tuple(int(part) for part in version),
        kernels=signatures,
        exported_symbols=_exported_symbols(elf),
    )
