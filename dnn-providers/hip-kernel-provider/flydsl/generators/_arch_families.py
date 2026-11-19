# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The LLVM generic targets FlyDSL objects are built for, and who each one covers.

A *generic* code object (``gfx11-generic``) is compiled once for the
intersection of a family's ISAs and loads on every member: ROCr's
``Isa::IsCompatible`` accepts it on any processor whose generic is that family,
provided the object's generic version is at least the one the processor joined
at. So the content tree holds one object set per family, and its descriptors
list every member: the shared packer copies those bytes into each member's own
shard, and each device finds its shard by its own ``gcnArchName``.

The table is ``arch_families.json`` beside this file, read here and by
``flydsl/CMakeLists.txt``, so the generators and the build agree on which arches
a family dir ships. Membership is an explicit list, never derived from the
digits of a name: gfx1170/gfx1171 are gfx11 parts but belong to
``gfx11-7-generic``, not ``gfx11-generic``. The lists come from LLVM's
AMDGPUUsage "Generic Processors" table and ROCr's ``isa.cpp``; re-check them when
the pinned toolchain moves.

Per family:

* ``members`` -- every processor the generic object runs on.
* ``elf_mach`` -- ``EF_AMDGPU_MACH`` the object's ELF header must carry
  (``EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC`` = 0x054).
* ``min_generic_version`` -- the lowest ``EF_AMDGPU_GENERIC_VERSION`` every
  member accepts.
* ``mlir_chipset`` -- the concrete chipset FlyDSL's MLIR lowering is told
  (``_generic_targets.py``); the lowest member, whose feature gates assume least.
* ``flydsl_gpu_arch`` (optional) -- the arch FlyDSL's Python-side checks are told
  (``FLYDSL_GPU_ARCH``) when they cannot classify the generic name itself; see
  ``_flydsl_env.prepare``. Absent, they are told the generic name.
"""

from __future__ import annotations

import json
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent / "arch_families.json"

_TABLE: dict | None = None


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    return _TABLE


def is_generic(arch: str) -> bool:
    return arch.endswith("-generic")


def family(arch: str) -> dict:
    """The table row for a generic target; an unknown generic name is an error."""
    row = table().get(arch)
    if row is None:
        raise KeyError(
            f"generic target {arch!r} is not in {TABLE_PATH.name}; known: "
            f"{', '.join(sorted(table()))}"
        )
    return row


def members(arch: str) -> tuple[str, ...]:
    """Every arch an object built for `arch` runs on: itself if concrete."""
    if not is_generic(arch):
        return (arch,)
    return tuple(family(arch)["members"])


def covers(target: str, arch: str) -> bool:
    """Whether an object built for `target` runs on processor `arch`."""
    return arch in members(target)
