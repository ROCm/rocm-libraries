"""Synthesised code objects.

Shared by the signature parser tests and the rocke producer tests, whose stubbed
compiler has to hand back something the packer can read a signature out of. A
real comgr object is a bare ELF carrying this note, so a stub that is not one
exercises a path no producer takes.

A plain module rather than conftest content: `tests/` and `tests/rocke/` each
carry a conftest, both import as `conftest`, and whichever loads last shadows the
other for every `from conftest import`. A uniquely named module has no such
collision, and it is importable from both directories because pytest puts
`tests/` on `sys.path` when it loads the conftest there.
"""

import struct

import pytest

# Imported here rather than at conftest load: msgpack ships as a rocm_kpack
# dependency, so on a host without it installed system-wide it becomes importable
# only once conftest has put HIPKERNELPROVIDER_ROCM_KPACK_DIR on the path. Test
# modules, and so this one, are imported only after that wiring has run.
try:
    import msgpack
except ImportError:
    msgpack = None


def requires_msgpack():
    """Fail the calling test when the note cannot be packed.

    Synthesising an object needs msgpack; most tests in these modules do not, so
    callers reaching one down a single branch call this at that branch rather
    than marking the module.

    Fails rather than skips, matching the hipcc and rocke fixtures: msgpack ships
    as a rocm_kpack dependency, so it cannot be legitimately absent where the
    packer itself is importable, and a skip here would quietly retire the
    signature-parsing coverage instead of reporting a broken environment.
    """
    if msgpack is None:
        pytest.fail(
            "msgpack is not importable, but it ships as a rocm_kpack dependency"
        )


def _note(owner, note_type, payload):
    """One ELF note: namesz/descsz/type, then name and desc each padded to 4."""
    name = owner.encode() + b"\x00"
    pad = lambda b: b + b"\x00" * (-len(b) % 4)
    return (
        struct.pack("<III", len(name), len(payload), note_type)
        + pad(name)
        + pad(payload)
    )


def _elf(note_bytes):
    """A minimal ELF64 LSB object holding `note_bytes` in one SHT_NOTE section.

    Two section headers -- the mandatory null entry and the note -- and no
    string table: the parser selects on section type, never on section name, so
    naming the section would test nothing the real objects exercise.
    """
    header_size = 64
    entry_size = 64
    note_offset = header_size
    sh_offset = note_offset + len(note_bytes)

    header = bytearray(header_size)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # ELFCLASS64
    header[5] = 1  # ELFDATA2LSB
    header[6] = 1  # EV_CURRENT
    struct.pack_into("<Q", header, 0x28, sh_offset)
    struct.pack_into("<HHH", header, 0x3A, entry_size, 2, 0)

    null_section = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    note_section = struct.pack(
        "<IIQQQQIIQQ", 0, 7, 0, 0, note_offset, len(note_bytes), 0, 0, 4, 0
    )
    return bytes(header) + note_bytes + null_section + note_section


def _metadata(kernels):
    return msgpack.packb({"amdhsa.kernels": kernels}, use_bin_type=True)


def _arg(kind, size, offset, name=None):
    argument = {".value_kind": kind, ".size": size, ".offset": offset}
    if name is not None:
        argument[".name"] = name
    return argument


def _kernel(name, args):
    return {".name": name, ".symbol": name + ".kd", ".args": args}


def _object(kernels):
    return _elf(_note("AMDGPU", 32, _metadata(kernels)))


def _bundle(entries):
    """A clang offload bundle over `entries`, each (triple, payload)."""
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    header = magic + struct.pack("<Q", len(entries))
    for triple, _ in entries:
        header += struct.pack("<QQQ", 0, 0, len(triple)) + triple.encode()

    body = b""
    offsets = []
    for _, payload in entries:
        offsets.append(len(header) + len(body))
        body += payload

    rebuilt = magic + struct.pack("<Q", len(entries))
    for (triple, payload), offset in zip(entries, offsets):
        rebuilt += struct.pack("<QQQ", offset, len(payload), len(triple))
        rebuilt += triple.encode()
    return rebuilt + body
