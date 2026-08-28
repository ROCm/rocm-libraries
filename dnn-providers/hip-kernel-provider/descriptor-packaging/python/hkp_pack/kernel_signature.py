"""Kernel argument signatures, read out of the compiled code object.

The signature a descriptor records is derived from the object the packer just
compiled, never typed by a kernel author. A hand-authored arity restates the
same assumption that drifted -- the author edits the kernel and updates the
descriptor, or does neither -- so it catches typos and nothing else.

What the producer stores differs by kind, and both shapes appear in this tree:
hipcc `--genco` emits a clang offload bundle wrapping the device ELF, while the
ASM producers store a bare ELF. Sniff, do not assume.

Argument NAMES are producer-dependent: clang emits `.args[].name` for OpenCL,
where names are part of the ABI, and omits it for HIP `extern "C" __global__`
kernels, so every kernel on the `hip` path here has `name: None`. The ASM/aiter
objects do carry them. They are recorded when present and omitted when not.
"""

import struct

from .errors import HkpPackError

try:
    import msgpack
except ImportError:  # pragma: no cover - environment without the packer's deps
    msgpack = None

_BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
_ELF_MAGIC = b"\x7fELF"

# ELF64 little-endian header offsets. Both are asserted rather than assumed:
# an amdgcn code object is always ELFCLASS64/ELFDATA2LSB, and anything else
# reaching here is a producer change this parser has not been taught about.
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_SHT_NOTE = 7

# NT_AMDGPU_METADATA, owner "AMDGPU": the msgpack document describing every
# kernel in the object. Code-object v3 and later; v2 used a YAML note under a
# different type and is not produced by any toolchain this packer supports.
_NT_AMDGPU_METADATA = 32
_AMDGPU_NOTE_OWNER = "AMDGPU"

# Arguments the compiler appends rather than the host marshalling them. Keeping
# them would make the recorded signature structurally incomparable with the
# pack's argument list.
_HIDDEN_KIND_PREFIX = "hidden_"

_AMDGCN_TRIPLE_MARKER = "-amdgcn-"


def _require_msgpack(where):
    if msgpack is None:
        raise HkpPackError(
            f"{where}: reading a kernel signature needs the 'msgpack' module, which is "
            "not importable. It ships as a dependency of rocm_kpack; install it, or "
            "pass --kpack-python-dir pointing at an environment that has it."
        )
    return msgpack


def _align4(value):
    return (value + 3) & ~3


def amdgcn_object(data, where):
    """The amdgcn ELF inside `data`, unwrapping a clang offload bundle if present.

    A bundle holds one entry per offload target plus a host entry, which for a
    `--genco` compile is present but zero-length. Entries are selected on the
    triple carrying `-amdgcn-` rather than on position or on the arch name: the
    host entry's offset can coincide with the device entry's, so an index-based
    pick silently returns host bytes.
    """
    if data[: len(_ELF_MAGIC)] == _ELF_MAGIC:
        return data
    if data[: len(_BUNDLE_MAGIC)] != _BUNDLE_MAGIC:
        raise HkpPackError(
            f"{where}: code object is neither an ELF nor a clang offload bundle "
            f"(leading bytes {data[:8]!r})"
        )

    cursor = len(_BUNDLE_MAGIC)
    (count,) = struct.unpack_from("<Q", data, cursor)
    cursor += 8

    found = []
    for _ in range(count):
        offset, size, triple_len = struct.unpack_from("<QQQ", data, cursor)
        cursor += 24
        triple = data[cursor : cursor + triple_len].decode("utf-8", "replace")
        cursor += triple_len
        if _AMDGCN_TRIPLE_MARKER in triple:
            found.append((triple, offset, size))

    if not found:
        raise HkpPackError(
            f"{where}: clang offload bundle holds no amdgcn entry "
            f"({count} entries present)"
        )
    if len(found) > 1:
        triples = ", ".join(sorted(t for t, _, _ in found))
        raise HkpPackError(
            f"{where}: clang offload bundle holds {len(found)} amdgcn entries "
            f"({triples}); this packer compiles one arch per object"
        )

    _, offset, size = found[0]
    inner = data[offset : offset + size]
    if inner[: len(_ELF_MAGIC)] != _ELF_MAGIC:
        raise HkpPackError(
            f"{where}: clang offload bundle's amdgcn entry is not an ELF "
            f"(leading bytes {inner[:8]!r})"
        )
    return inner


def _metadata_document(elf, where):
    """The msgpack-decoded NT_AMDGPU_METADATA note, or raise."""
    unpacker = _require_msgpack(where)

    if elf[4] != _ELFCLASS64 or elf[5] != _ELFDATA2LSB:
        raise HkpPackError(
            f"{where}: code object is not a little-endian 64-bit ELF "
            f"(class {elf[4]}, data {elf[5]})"
        )

    (section_offset,) = struct.unpack_from("<Q", elf, 0x28)
    entry_size, section_count = struct.unpack_from("<HH", elf, 0x3A)

    for index in range(section_count):
        _, kind, _, _, offset, size, _, _, _, _ = struct.unpack_from(
            "<IIQQQQIIQQ", elf, section_offset + index * entry_size
        )
        if kind != _SHT_NOTE:
            continue
        cursor = offset
        end = offset + size
        while cursor < end:
            name_size, desc_size, note_type = struct.unpack_from("<III", elf, cursor)
            cursor += 12
            owner_bytes = elf[cursor : cursor + name_size].rstrip(b"\x00")
            owner = owner_bytes.decode("utf-8", "replace")
            cursor += _align4(name_size)
            descriptor = elf[cursor : cursor + desc_size]
            cursor += _align4(desc_size)
            if owner == _AMDGPU_NOTE_OWNER and note_type == _NT_AMDGPU_METADATA:
                return unpacker.unpackb(descriptor, raw=False, strict_map_key=False)

    raise HkpPackError(
        f"{where}: code object carries no AMDGPU metadata note, so its kernel "
        "signatures cannot be read"
    )


def kernel_signature(code_object, symbol, where):
    """The marshalled argument list of `symbol` within `code_object`.

    Returns a list of dicts in declaration order, one per argument the host is
    expected to pass: `kind`, `size`, `offset`, and `name` when the producer
    emitted one. Hidden arguments are excluded -- see the module docstring.

    An empty list is a legitimate answer for a kernel that takes no arguments
    and is distinct from a missing signature, which raises.
    """
    document = _metadata_document(amdgcn_object(code_object, where), where)

    kernels = document.get("amdhsa.kernels")
    if not kernels:
        raise HkpPackError(f"{where}: AMDGPU metadata note lists no kernels")

    for kernel in kernels:
        if kernel.get(".name") != symbol:
            continue
        arguments = []
        for argument in kernel.get(".args") or []:
            kind = argument[".value_kind"]
            if kind.startswith(_HIDDEN_KIND_PREFIX):
                continue
            recorded = {
                "kind": kind,
                "size": argument[".size"],
                "offset": argument[".offset"],
            }
            name = argument.get(".name")
            if name is not None:
                recorded["name"] = name
            arguments.append(recorded)
        return arguments

    present = sorted(str(k.get(".name")) for k in kernels)
    raise HkpPackError(
        f"{where}: code object has no kernel named '{symbol}'; it holds "
        f"{', '.join(present)}"
    )
