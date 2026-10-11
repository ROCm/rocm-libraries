# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/_codeobject.py @ 03911535bd0 (origin/users/brpepers/hipdnn-flydsl-provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reading and verifying a compiled code object.

The generator compiles an instance, then checks the object it got back against
what it asked for -- arch, kernarg tuple, scratch, erratum pattern --
*before* writing it. Reading goes through ``hkp_pack.kernel_signature`` (the
packer parses the very same AMDGPU note), which needs ``msgpack`` in the running
interpreter. Every failure carries a stable ``TRITONGEN-*`` prefix.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from . import _contract as C

HKP_DIR = Path(__file__).resolve().parents[2]
PACKAGING_DIR = HKP_DIR / "descriptor-packaging" / "python"


def objdump_path() -> str:
    """``llvm-objdump``: ``$ROCM_PATH/llvm/bin``, then ``/opt/rocm/llvm/bin``, then PATH."""
    candidates = []
    rocm = os.environ.get("ROCM_PATH")
    if rocm:
        candidates.append(Path(rocm) / "llvm" / "bin" / "llvm-objdump")
    candidates.append(Path("/opt/rocm/llvm/bin/llvm-objdump"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("llvm-objdump")
    if found is None:
        raise GeneratorError(
            f"{C.ERR_ENV}: llvm-objdump not found in "
            f"{', '.join(map(str, candidates))} or on PATH"
        )
    return found


class GeneratorError(RuntimeError):
    """An instance could not be built, or the object built is not the one asked for."""


def hkp_pack_module():
    """Import ``hkp_pack.kernel_signature`` from the sibling packaging tree."""
    if not (PACKAGING_DIR / "hkp_pack").is_dir():
        raise GeneratorError(f"hkp_pack not found under {PACKAGING_DIR}")
    if str(PACKAGING_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGING_DIR))
    from hkp_pack import kernel_signature as module  # noqa: PLC0415

    return module


def describe(blob: bytes, where: str, symbol: str | None = None) -> dict:
    """The AMDGPU metadata facts we verify and record, for one code object.

    ``signature`` lists the explicit kernargs only (``hidden_*`` arguments are
    dropped, as the packer drops them).
    """
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)

    kernels = document.get("amdhsa.kernels") or []
    if symbol is not None:
        named = [k for k in kernels if k.get(".name") == symbol]
        if len(named) != 1:
            names = ", ".join(str(k.get(".name")) for k in kernels)
            raise GeneratorError(f"{where}: no single kernel {symbol!r} in [{names}]")
        kernels = named
    if len(kernels) != 1:
        names = ", ".join(str(k.get(".name")) for k in kernels)
        raise GeneratorError(
            f"{where}: expected exactly one kernel in the object, found "
            f"{len(kernels)} [{names}]"
        )
    kernel = kernels[0]

    args = [
        arg
        for arg in (kernel.get(".args") or [])
        if not str(arg.get(".value_kind", "")).startswith(hkp._HIDDEN_KIND_PREFIX)
    ]
    return {
        "symbol": kernel.get(".name"),
        "kernarg_segment_size": kernel.get(".kernarg_segment_size"),
        "targets": document.get("amdhsa.target"),
        "signature": tuple(
            (arg[".value_kind"], arg[".size"], arg[".offset"]) for arg in args
        ),
    }


# The processor at the end of an `amdhsa.target` triple, after dropping `:feature`
# suffixes (a feature can itself end in a dash, `xnack-`).
_TARGET_ARCH = re.compile(r"(gfx[0-9a-z]+(?:-[0-9]+)*-generic|gfx[0-9a-f]+)$")


def target_arch(targets, where: str) -> str:
    """The processor out of ``amdhsa.target``, ignoring feature suffixes."""
    if isinstance(targets, (list, tuple)):
        if len(targets) != 1:
            raise GeneratorError(
                f"{C.ERR_ARCH}: {where}: expected one target, got {targets!r}"
            )
        target = targets[0]
    else:
        target = targets
    if not isinstance(target, str):
        raise GeneratorError(
            f"{C.ERR_ARCH}: {where}: unreadable amdhsa.target {target!r}"
        )
    match = _TARGET_ARCH.search(target.split(":", 1)[0])
    if match is None:
        raise GeneratorError(
            f"{C.ERR_ARCH}: {where}: amdhsa.target {target!r} does not end in a gfx name"
        )
    return match.group(1)


def verify_arch(described: dict, arch: str, where: str) -> None:
    """Fail before writing an object filed under an arch it was not built for."""
    built_for = target_arch(described["targets"], where)
    if built_for != arch:
        raise GeneratorError(
            f"{C.ERR_ARCH}: {where}: object targets {built_for!r} but is being "
            f"filed under {arch!r}"
        )


def expected_signature() -> tuple:
    return tuple((a.value_kind, a.size, a.offset) for a in C.KERNARGS)


def verify_abi(described: dict, name: str, where: str) -> None:
    """The full explicit kernarg tuple (both scratch slots included).

    Any argument after slot 6 must be ``hidden_*``; ``describe`` drops those, so
    a non-hidden extra argument shows up here as a tuple mismatch.
    """
    if described["signature"] != expected_signature():
        raise GeneratorError(
            f"{C.ERR_ABI}: {where}: kernarg tuple for {name!r} is\n"
            f"  {described['signature']}\nbut the contract declares\n"
            f"  {expected_signature()}"
        )
    segment = described["kernarg_segment_size"]
    if segment is None or segment < C.MIN_KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{C.ERR_ABI}: {where}: kernarg_segment_size {segment} < "
            f"{C.MIN_KERNARG_SEGMENT_SIZE}"
        )
    if described["symbol"] != C.KERNEL_SYMBOL:
        raise GeneratorError(
            f"{C.ERR_ABI}: {where}: kernel symbol {described['symbol']!r}, "
            f"expected {C.KERNEL_SYMBOL!r}"
        )


def check_scratch(name: str, global_scratch_size, profile_scratch_size) -> None:
    """Reject any non-zero scratch, naming the instance and both sizes."""
    if global_scratch_size != 0 or profile_scratch_size != 0:
        raise GeneratorError(
            f"{C.ERR_SCRATCH}: {name} global_scratch_size={global_scratch_size} "
            f"profile_scratch_size={profile_scratch_size}"
        )


def arg_records(signature: tuple) -> list[dict]:
    """The manifest's per-argument records: semantic name beside the real layout."""
    return [
        {"name": arg.name, "kind": kind, "size": size, "offset": offset}
        for arg, (kind, size, offset) in zip(C.KERNARGS, signature)
    ]


# --- ELF .comment ----------------------------------------------------------

_LLVM_COMMIT = re.compile(r"\(\S*llvm-project\s+([0-9a-f]{40})\)")


def _comment_section(elf: bytes, where: str) -> str:
    if len(elf) < 0x40 or elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
        raise GeneratorError(f"{where}: not a little-endian ELF64 code object")
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)

    def header(index: int):
        return struct.unpack_from("<IIQQQQIIQQ", elf, shoff + index * shentsize)

    _, _, _, _, str_off, str_size, *_ = header(shstrndx)
    names = elf[str_off : str_off + str_size]
    for index in range(shnum):
        name, _, _, _, offset, size, *_ = header(index)
        end = names.index(b"\0", name)
        if names[name:end] == b".comment":
            return elf[offset : offset + size].decode("utf-8", "replace")
    raise GeneratorError(f"{where}: object has no .comment section")


def llvm_commit(blob: bytes, where: str) -> str:
    """The LLVM commit from the object's ``.comment`` (``... llvm-project <sha>``)."""
    elf = hkp_pack_module().amdgcn_object(blob, where)
    match = _LLVM_COMMIT.search(_comment_section(elf, where))
    if match is None:
        raise GeneratorError(f"{where}: .comment carries no llvm-project commit")
    return match.group(1)


# --- Erratum ISA check -------------------------------------------------------

_MFMA = re.compile(r"\bv_s?mfma")
_PK_F32 = re.compile(r"\bv_pk_(?:fma|mul|add)_f32\b")


def isa_counts(disassembly: str) -> tuple[int, int]:
    """``(v_mfma*/v_smfmac* count, v_pk_(fma|mul|add)_f32 count)`` in a disassembly."""
    return len(_MFMA.findall(disassembly)), len(_PK_F32.findall(disassembly))


def check_erratum_text(disassembly: str, where: str) -> None:
    """Reject packed-FP32 instructions beside MFMA (the gfx950 erratum pattern)."""
    mfma, pk = isa_counts(disassembly)
    if mfma and pk:
        raise GeneratorError(
            f"{C.ERR_ERRATUM}: {where}: {pk} v_pk_*_f32 beside {mfma} v_(s)mfma"
        )


def disassemble(blob: bytes, arch: str, where: str) -> str:
    elf = hkp_pack_module().amdgcn_object(blob, where)
    objdump = objdump_path()
    with tempfile.TemporaryDirectory(prefix="tritongen-objdump-") as tmp:
        path = Path(tmp) / "object.hsaco"
        path.write_bytes(elf)
        try:
            result = subprocess.run(
                [objdump, "-d", f"--mcpu={arch}", str(path)],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise GeneratorError(f"{C.ERR_ENV}: {objdump} not runnable: {exc}") from exc
    if result.returncode != 0:
        raise GeneratorError(
            f"{where}: {objdump} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def check_erratum_object(blob: bytes, arch: str, where: str) -> tuple[int, int]:
    """Disassemble the object and apply :func:`check_erratum_text`; returns the counts."""
    text = disassemble(blob, arch, where)
    check_erratum_text(text, where)
    return isa_counts(text)
