# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reading and verifying a compiled code object, shared by every op's generator.

Every generator compiles an instance, then checks the object it got back against
what it asked for -- arch, kernarg layout, kernarg segment size -- *before*
writing it. The checks differ per op only in the expected values, so the reading
and the arch parse live here once.
"""

from __future__ import annotations

import re
import sys

from . import _arch_families as families
from . import _flydsl_env as env


class GeneratorError(RuntimeError):
    """An instance could not be built, or the object built is not the one asked for."""


def hkp_pack_module():
    """Import ``hkp_pack.kernel_signature`` from the sibling packaging tree.

    We reach for two of its helpers rather than re-deriving them: ``amdgcn_object``
    unwraps the clang offload bundle by *selecting on the ``-amdgcn-`` triple*
    instead of by position, and ``_metadata_document`` walks the ELF's SHT_NOTE
    sections for ``NT_AMDGPU_METADATA``. A second copy of either in this
    directory would be a second thing to keep correct, and the packer these
    generators feed parses the very same note.
    """
    packaging = env.PROVIDER_DIR.parent / "descriptor-packaging" / "python"
    if not (packaging / "hkp_pack").is_dir():
        raise GeneratorError(f"hkp_pack not found under {packaging}")
    if str(packaging) not in sys.path:
        sys.path.insert(0, str(packaging))
    from hkp_pack import kernel_signature as module  # noqa: PLC0415

    return module


def describe(blob: bytes, where: str, symbol: str | None = None) -> dict:
    """The AMDGPU metadata facts we verify and record, for one code object.

    An object holds one kernel unless @p symbol names which of several to describe:
    the decode objects carry their merge kernel beside the main one.
    """
    hkp = hkp_pack_module()
    # `_metadata_document` is private to hkp_pack, but `kernel_signature()` -- its
    # public entry point -- returns only the argument list, and we also need the
    # kernarg segment size and the target triple to verify the object at all.
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
            f"{len(kernels)} [{names}]. The packer keys on (toc_key, symbol), so a "
            "multi-kernel object needs a TOC entry per symbol, not one per file."
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
        "named_args": sum(1 for arg in args if arg.get(".name") is not None),
        "group_segment_fixed_size": kernel.get(".group_segment_fixed_size"),
        "vgpr_count": kernel.get(".vgpr_count"),
        "vgpr_spill_count": kernel.get(".vgpr_spill_count"),
    }


# The processor at the end of an `amdhsa.target` triple: a concrete gfx name, or
# an LLVM generic target (`gfx11-generic`, `gfx9-4-generic`), whose name itself
# contains dashes.
_TARGET_ARCH = re.compile(r"(gfx[0-9a-z]+(?:-[0-9]+)*-generic|gfx[0-9a-f]+)$")

# ELF header e_flags (ELF64: 16-byte ident, then type/machine/version, then
# entry/phoff/shoff), and the two AMDGPU fields read out of it.
_E_FLAGS_OFFSET = 48
_EF_AMDGPU_MACH = 0x0FF
_EF_AMDGPU_GENERIC_VERSION_SHIFT = 24


def kernel_names(blob: bytes, where: str) -> list[str]:
    """Every kernel symbol the object defines."""
    hkp = hkp_pack_module()
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)
    return [str(k.get(".name")) for k in document.get("amdhsa.kernels") or []]


def target_arch(targets, where: str) -> str:
    """The processor out of ``amdhsa.target``, ignoring feature suffixes.

    Three spellings are in circulation and all have to parse: the AITER ASM
    objects carry an *empty* environment field,
    ``amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-``; FlyDSL emits it as
    ``unknown``, ``amdgcn-amd-amdhsa-unknown-gfx1151``; and a generic target
    ends in a dashed name, ``amdgcn-amd-amdhsa-unknown-gfx11-generic``. So: drop
    the ``:feature`` suffixes, then match the processor at the end. The suffixes
    come off *first* because a feature can itself end in a dash (``xnack-``).

    We compare the processor alone -- feature flags describe how the object was
    built, while the directory claims which devices it runs on.
    """
    if isinstance(targets, (list, tuple)):
        if len(targets) != 1:
            raise GeneratorError(f"{where}: expected one target, got {targets!r}")
        target = targets[0]
    else:
        target = targets
    if not isinstance(target, str):
        raise GeneratorError(f"{where}: unreadable amdhsa.target {target!r}")
    match = _TARGET_ARCH.search(target.split(":", 1)[0])
    if match is None:
        raise GeneratorError(
            f"{where}: amdhsa.target {target!r} does not end in a gfx name"
        )
    return match.group(1)


def verify_arch(described: dict, arch: str, where: str) -> None:
    """Fail before writing an object filed under an arch it was not built for."""
    built_for = target_arch(described["targets"], where)
    if built_for != arch:
        raise GeneratorError(
            f"{where}: object targets {built_for!r} but is being filed under "
            f"{arch!r}. FlyDSL reads ARCH from the environment at each compile, so "
            "this means the env moved under the run -- not that the file is "
            "misnamed. Regenerate one arch per invocation."
        )


def elf_flags(blob: bytes, where: str) -> tuple[int, int]:
    """``(EF_AMDGPU_MACH, EF_AMDGPU_GENERIC_VERSION)`` from the object's ELF header."""
    elf = hkp_pack_module().amdgcn_object(blob, where)
    if len(elf) < _E_FLAGS_OFFSET + 4 or elf[:4] != b"\x7fELF":
        raise GeneratorError(f"{where}: not an ELF code object")
    flags = int.from_bytes(elf[_E_FLAGS_OFFSET : _E_FLAGS_OFFSET + 4], "little")
    return flags & _EF_AMDGPU_MACH, flags >> _EF_AMDGPU_GENERIC_VERSION_SHIFT


def verify_generic(blob: bytes, arch: str, where: str) -> None:
    """For a generic target, the ELF header must say so too.

    ``amdhsa.target`` is metadata the loader does not consult; ROCr decides
    compatibility from ``e_flags``. An object whose metadata says
    ``gfx11-generic`` but whose header carries a concrete machine would load on
    one member and be refused by the other seven.
    """
    if not families.is_generic(arch):
        return
    row = families.family(arch)
    mach, version = elf_flags(blob, where)
    if mach != row["elf_mach"] or version < row["min_generic_version"]:
        raise GeneratorError(
            f"{where}: ELF e_flags carry machine {mach:#05x} generic version "
            f"{version}; {arch} needs machine {row['elf_mach']:#05x} and generic "
            f"version >= {row['min_generic_version']}"
        )


def arg_records(names: tuple[str, ...], signature: tuple) -> list[dict]:
    """The manifest's per-argument records: semantic name beside the real layout."""
    return [
        {"name": name, "kind": kind, "size": size, "offset": offset}
        for name, (kind, size, offset) in zip(names, signature)
    ]
