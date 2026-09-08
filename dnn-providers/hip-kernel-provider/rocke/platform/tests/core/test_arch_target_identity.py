# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from rocke.core.arch import (
    ArchTarget,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_from_target_id,
    known_arches,
    target_id_from_isa,
)


@pytest.mark.parametrize("arch", known_arches())
def test_known_target_ids_normalize_to_themselves(arch: str) -> None:
    assert base_arch_from_target_id(arch) == arch
    assert arch_from_isa(f"amdgcn-amd-amdhsa--{arch}") == arch


@pytest.mark.parametrize(
    ("target_id", "base_arch"),
    [
        ("gfx1250-strict", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx1300-strict", "gfx1300"),
    ],
)
def test_profiled_target_ids_have_an_explicit_base_arch(
    target_id: str, base_arch: str
) -> None:
    isa = f"amdgcn-amd-amdhsa--{target_id}"
    assert target_id_from_isa(isa) == target_id
    assert arch_from_isa(isa) == base_arch
    assert base_arch_from_target_id(target_id) == base_arch


@pytest.mark.parametrize(
    ("isa", "target_id"),
    [
        ("amdgcn-amd-amdhsa--gfx1250-strict", "gfx1250-strict"),
        ("amdgcn-amd-amdhsa-opencl-gfx1250-strict", "gfx1250-strict"),
        (
            "vendor-prefix-without-triple-fields-gfx942:sramecc+:xnack-",
            "gfx942:sramecc+:xnack-",
        ),
        (
            "gfx-named-prefix-gfx942:sramecc+:xnack-",
            "gfx942:sramecc+:xnack-",
        ),
    ],
)
def test_target_id_extraction_accepts_all_triple_field_forms(
    isa: str, target_id: str
) -> None:
    assert target_id_from_isa(isa) == target_id


def test_arch_target_uses_base_architecture_rows() -> None:
    assert ArchTarget.from_gfx("gfx1250").gfx == "gfx1250"
    with pytest.raises(KeyError, match="unknown gfx target"):
        ArchTarget.from_gfx("gfx1250-strict")


@pytest.mark.parametrize(
    ("target_id", "compiler_target"),
    [
        ("gfx1250", "gfx1250"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-"),
        ("gfx11-generic", "gfx11-generic"),
    ],
)
def test_runtime_profiles_are_removed_from_compiler_targets(
    target_id: str, compiler_target: str
) -> None:
    assert compiler_target_from_target_id(target_id) == compiler_target


def test_non_gfx_target_is_preserved_for_validation() -> None:
    assert target_id_from_isa("unexpected-target") == "unexpected-target"
    assert base_arch_from_target_id("unexpected-target") == "unexpected-target"
