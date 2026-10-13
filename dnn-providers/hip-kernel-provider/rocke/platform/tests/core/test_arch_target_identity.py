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


@pytest.mark.parametrize(
    "target_id",
    [
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1152",
        "gfx1153",
        "gfx1154",
    ],
)
def test_gfx11_processors_use_the_generic_catalog_row(target_id: str) -> None:
    assert base_arch_from_target_id(target_id) == "gfx11-generic"
    assert compiler_target_from_target_id(target_id) == target_id


@pytest.mark.parametrize(
    ("target_id", "compiler_target"),
    [
        ("gfx1101:unknown+", "gfx1101:unknown+"),
        ("gfx1101-strict", "gfx1101"),
        ("gfx1101-strict:unknown+", "gfx1101:unknown+"),
    ],
)
def test_gfx11_generic_catalog_selection_is_independent_of_target_suffixes(
    target_id: str, compiler_target: str
) -> None:
    assert base_arch_from_target_id(target_id) == "gfx11-generic"
    assert compiler_target_from_target_id(target_id) == compiler_target


@pytest.mark.parametrize("target_id", ["gfx1170", "gfx1200", "gfx1101_bad"])
def test_nonmembers_do_not_use_the_gfx11_generic_catalog_row(
    target_id: str,
) -> None:
    expected = "gfx1101" if target_id == "gfx1101_bad" else target_id
    assert base_arch_from_target_id(target_id) == expected


@pytest.mark.parametrize(
    ("target_id", "compiler_target"),
    [
        ("gfx1151", "gfx1151"),
        ("gfx1151-strict:unknown+", "gfx1151:unknown+"),
    ],
)
def test_exact_catalog_row_takes_precedence_over_generic_domain(
    target_id: str, compiler_target: str
) -> None:
    assert base_arch_from_target_id(target_id) == "gfx1151"
    assert compiler_target_from_target_id(target_id) == compiler_target


def test_arch_target_uses_base_architecture_rows() -> None:
    assert ArchTarget.from_gfx("gfx1250").gfx == "gfx1250"
    with pytest.raises(KeyError, match="unknown gfx target"):
        ArchTarget.from_gfx("gfx1250-strict")


def test_cpp_target_identity_matches_python() -> None:
    """Compare the native CTest executable with Python when a build is supplied."""
    executable = os.environ.get("ROCKE_ARCH_TARGET_TEST_EXE")
    if not executable:
        pytest.skip("set ROCKE_ARCH_TARGET_TEST_EXE to the native target identity test")
    targets = [
        *known_arches(),
        "gfx1101",
        "gfx1170",
        "gfx1101_bad",
        "gfx00a",
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
            )
        )
    assert actual == expected
