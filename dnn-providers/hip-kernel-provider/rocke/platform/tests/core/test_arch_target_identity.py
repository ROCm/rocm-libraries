# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess

import pytest

from rocke.core.arch import (
    ArchTarget,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_from_target_id,
    generic_arch_from_target_id,
    known_arches,
    target_id_from_isa,
)


@pytest.mark.parametrize("arch", known_arches())
def test_known_target_ids_normalize_to_themselves(arch: str) -> None:
    assert base_arch_from_target_id(arch) == arch
    assert compiler_target_from_target_id(arch) == arch
    assert arch_from_isa(f"amdgcn-amd-amdhsa--{arch}") == arch


@pytest.mark.parametrize(
    ("target_id", "base_arch", "compiler_target"),
    [
        ("gfx1250-strict", "gfx1250", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942", "gfx942:sramecc+:xnack-"),
        ("unexpected-target", "unexpected-target", "unexpected-target"),
    ],
)
def test_target_identity_forms(
    target_id: str, base_arch: str, compiler_target: str
) -> None:
    assert target_id_from_isa(target_id) == target_id
    assert arch_from_isa(target_id) == base_arch
    assert base_arch_from_target_id(target_id) == base_arch
    assert compiler_target_from_target_id(target_id) == compiler_target


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
    ("target_id", "generic"),
    [
        ("gfx1100", "gfx11-generic"),
        ("gfx1103:xnack-", "gfx11-generic"),
        ("gfx1151", "gfx11-generic"),
        ("gfx1153-strict", "gfx11-generic"),
        ("gfx11-generic", "gfx11-generic"),
        ("gfx1010", None),
        ("gfx942", None),
        ("gfx11000", None),
        ("", None),
    ],
)
def test_generic_arch_covers_family_members(target_id: str, generic) -> None:
    assert generic_arch_from_target_id(target_id) == generic


def test_generic_targets_are_catalogued_and_disjoint() -> None:
    from rocke.core.arch.target import _load_generic_targets

    seen = set()
    for generic, members in _load_generic_targets().items():
        target = ArchTarget.from_gfx(generic)
        assert members and seen.isdisjoint(members)
        seen.update(members)
        for member in members:
            if member in known_arches():
                member_target = ArchTarget.from_gfx(member)
                assert member_target.target_family == target.target_family
                assert member_target.wave_size == target.wave_size


def test_cpp_target_identity_matches_python() -> None:
    """Compare the native CTest executable with Python when a build is supplied."""
    executable = os.environ.get("ROCKE_ARCH_TARGET_TEST_EXE")
    if not executable:
        pytest.skip("set ROCKE_ARCH_TARGET_TEST_EXE to the native target identity test")
    targets = [
        *known_arches(),
        "gfx00a",
        "gfx1100",
        "gfx1102",
        "gfx1010",
        "unexpected-target",
        "",
        "gfx",
        "gfx-",
        "gfx942_bad",
    ]
    inputs = [
        f"{prefix}{target}{profile}{features}"
        for target in targets
        for prefix in ("", "amdgcn-amd-amdhsa--", "gfx-named-prefix-")
        for profile in ("", "-strict")
        for features in ("", ":", ":sramecc+:xnack-", ":unknown+")
    ]
    result = subprocess.run(
        [executable, *inputs], capture_output=True, text=True, check=True, timeout=30
    )
    actual = [tuple(line.split("\t")) for line in result.stdout.splitlines()]
    expected = []
    for isa in inputs:
        target = target_id_from_isa(isa)
        expected.append(
            (
                target,
                base_arch_from_target_id(target),
                compiler_target_from_target_id(target),
                arch_from_isa(isa),
                generic_arch_from_target_id(target) or "-",
            )
        )
    assert actual == expected
