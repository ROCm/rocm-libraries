# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Let the pinned FlyDSL emit AMDGPU *generic* code objects.

FlyDSL 0.3.4 hands the target arch verbatim to MLIR's ``convert-gpu-to-rocdl``
``chipset=`` option, and ``amdgpu::Chipset::parse`` accepts only concrete
``gfxNNNN`` names, so ``ARCH=gfx11-generic`` fails before codegen. The LLVM
backend and ``rocdl-attach-target chip=`` take generic targets as they are. This
shim rewrites the one ``chipset=`` fragment to the family's ``mlir_chipset``
(``arch_families.json``) and leaves the attach-target alone, so the object's
``amdhsa.target`` and ELF ``e_flags`` still name the generic target.

The lowest member is the safe chipset: MLIR's chipset gates are monotone
"at least chip X" checks and the generic ISA is the intersection of its
members, so it never enables an op a sibling lacks -- and LLVM's own codegen
for the generic target rejects anything that slipped through.

Pinned to the FlyDSL release it was validated against and to the families
validated under it. FlyDSL 0.3.4's Python-side arch checks are prefix compares
on the arch string, which ``gfx11-generic`` passes (``startswith("gfx11")``) and
``gfx12-generic`` does not (``startswith("gfx120")``): told ``gfx12-generic``,
FlyDSL picks no WMMA atom and CDNA buffer-descriptor flags. That family carries a
``flydsl_gpu_arch`` (its lowest member) that ``_flydsl_env.prepare`` gives those
checks instead, while the target the object is built for stays the generic one.
Delete this file once the pinned FlyDSL parses generic targets itself.
"""

from __future__ import annotations

from . import _arch_families as families

VALIDATED_FLYDSL = "0.3.4"
VALIDATED_FAMILIES = ("gfx11-generic", "gfx12-generic")

_installed: str | None = None


def install(arch: str) -> None:
    """Patch FlyDSL's lowering pipeline for generic `arch`. Idempotent per arch."""
    global _installed
    if arch not in VALIDATED_FAMILIES:
        raise RuntimeError(
            f"generic target {arch!r} is not validated for the FlyDSL generic-target "
            f"shim (validated: {', '.join(VALIDATED_FAMILIES)})"
        )
    if _installed is not None:
        if _installed != arch:
            raise RuntimeError(
                f"generic-target shim already installed for {_installed}, not {arch}"
            )
        return

    import flydsl  # noqa: PLC0415
    from flydsl.compiler.backends import rocm  # noqa: PLC0415

    if flydsl.__version__ != VALIDATED_FLYDSL:
        raise RuntimeError(
            f"the generic-target shim was validated against flydsl "
            f"{VALIDATED_FLYDSL}, found {flydsl.__version__}: drop the shim if this "
            "release parses generic targets, otherwise re-validate it"
        )

    chipset = families.family(arch)["mlir_chipset"]
    original = rocm.RocmBackend._pipeline_parts

    def _pipeline_parts(self, *, compile_hints):
        pre, binary = original(self, compile_hints=compile_hints)
        chip = self.target.arch
        if chip != arch:
            return pre, binary
        needle = f"convert-gpu-to-rocdl{{chipset={chip} "
        hits = [index for index, fragment in enumerate(pre) if needle in fragment]
        if len(hits) != 1:
            raise RuntimeError(
                f"generic-target shim: expected one '{needle}' fragment in the "
                f"lowering pipeline, found {len(hits)}"
            )
        pre = list(pre)
        pre[hits[0]] = pre[hits[0]].replace(
            needle, f"convert-gpu-to-rocdl{{chipset={chipset} "
        )
        return pre, binary

    rocm.RocmBackend._pipeline_parts = _pipeline_parts
    _installed = arch


def record(arch: str) -> dict | None:
    """Provenance for an object built through the shim, or None if it was not."""
    if not families.is_generic(arch):
        return None
    row = families.family(arch)
    shim = f"chipset={row['mlir_chipset']}"
    if "flydsl_gpu_arch" in row:
        shim += f" flydsl_gpu_arch={row['flydsl_gpu_arch']}"
    return {"generic_target_shim": shim}
