# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reads kernel metadata from an AMDGPU code object (.co).

A code object stores the metadata of its kernels in an ELF note of type
NT_AMDGPU_METADATA, encoded as MessagePack. On ROCm, hipFuncGetAttribute
reports a kernel's HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK straight from its
.max_flat_workgroup_size entry, so codegen.py reads that value here and the
engine can check a launch block without loading the kernel on a GPU.

Only the standard library is used: codegen.py runs before any Python package
the build fetches is on the path.
"""

import struct

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_SHT_NOTE = 7
_NT_AMDGPU_METADATA = 32
_AMDGPU_NOTE_NAME = b"AMDGPU\0"


class CodeObjectError(Exception):
    """The file is not a code object this module can read."""


class _MsgpackReader:
    """Decodes the MessagePack subset AMDGPU metadata uses (no extension types)."""

    def __init__(self, data):
        self._data = data
        self._pos = 0

    def _take(self, size):
        end = self._pos + size
        if end > len(self._data):
            raise CodeObjectError("truncated MessagePack metadata")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def _unpack(self, fmt):
        return struct.unpack(">" + fmt, self._take(struct.calcsize(">" + fmt)))[0]

    def _map(self, count):
        return {self.read(): self.read() for _ in range(count)}

    def _array(self, count):
        return [self.read() for _ in range(count)]

    def _str(self, size):
        return self._take(size).decode("utf-8")

    def read(self):
        tag = self._take(1)[0]
        if tag <= 0x7F:
            return tag
        if tag >= 0xE0:
            return tag - 0x100
        if 0x80 <= tag <= 0x8F:
            return self._map(tag & 0x0F)
        if 0x90 <= tag <= 0x9F:
            return self._array(tag & 0x0F)
        if 0xA0 <= tag <= 0xBF:
            return self._str(tag & 0x1F)
        fixed = {
            0xC0: lambda: None,
            0xC2: lambda: False,
            0xC3: lambda: True,
            0xC4: lambda: self._take(self._unpack("B")),
            0xC5: lambda: self._take(self._unpack("H")),
            0xC6: lambda: self._take(self._unpack("I")),
            0xCA: lambda: self._unpack("f"),
            0xCB: lambda: self._unpack("d"),
            0xCC: lambda: self._unpack("B"),
            0xCD: lambda: self._unpack("H"),
            0xCE: lambda: self._unpack("I"),
            0xCF: lambda: self._unpack("Q"),
            0xD0: lambda: self._unpack("b"),
            0xD1: lambda: self._unpack("h"),
            0xD2: lambda: self._unpack("i"),
            0xD3: lambda: self._unpack("q"),
            0xD9: lambda: self._str(self._unpack("B")),
            0xDA: lambda: self._str(self._unpack("H")),
            0xDB: lambda: self._str(self._unpack("I")),
            0xDC: lambda: self._array(self._unpack("H")),
            0xDD: lambda: self._array(self._unpack("I")),
            0xDE: lambda: self._map(self._unpack("H")),
            0xDF: lambda: self._map(self._unpack("I")),
        }
        if tag not in fixed:
            raise CodeObjectError(f"unsupported MessagePack type 0x{tag:02x}")
        return fixed[tag]()


def _align4(value):
    return (value + 3) & ~3


def _amdgpu_metadata(data):
    """Returns the decoded NT_AMDGPU_METADATA note of an ELF64 code object."""
    if data[:4] != _ELF_MAGIC:
        raise CodeObjectError("not an ELF file")
    if data[4] != _ELFCLASS64 or data[5] != _ELFDATA2LSB:
        raise CodeObjectError("not a little-endian ELF64 file")

    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    for index in range(shnum):
        header = shoff + index * shentsize
        sh_type = struct.unpack_from("<I", data, header + 4)[0]
        if sh_type != _SHT_NOTE:
            continue
        offset, size = struct.unpack_from("<QQ", data, header + 0x18)
        pos = offset
        while pos + 12 <= offset + size:
            namesz, descsz, note_type = struct.unpack_from("<III", data, pos)
            name_start = pos + 12
            desc_start = name_start + _align4(namesz)
            name = data[name_start : name_start + namesz]
            if note_type == _NT_AMDGPU_METADATA and name == _AMDGPU_NOTE_NAME:
                return _MsgpackReader(data[desc_start : desc_start + descsz]).read()
            pos = desc_start + _align4(descsz)
    raise CodeObjectError("no NT_AMDGPU_METADATA note")


def max_flat_workgroup_size(co_path, kernel_name):
    """Largest block, in threads, that kernel_name in co_path can be launched with."""
    with open(co_path, "rb") as f:
        data = f.read()
    try:
        metadata = _amdgpu_metadata(data)
    except (CodeObjectError, struct.error, UnicodeDecodeError) as e:
        raise CodeObjectError(f"{co_path}: {e}") from e

    for kernel in metadata.get("amdhsa.kernels", []):
        if kernel.get(".name") == kernel_name:
            value = kernel.get(".max_flat_workgroup_size")
            if not isinstance(value, int) or value <= 0:
                raise CodeObjectError(
                    f"{co_path}: kernel {kernel_name} has no valid .max_flat_workgroup_size"
                )
            return value
    raise CodeObjectError(f"{co_path}: no kernel named {kernel_name}")
