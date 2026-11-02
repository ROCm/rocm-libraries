# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import subprocess

import pytest

from rocke.core.arch import (
    ArchTarget,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_from_target_id,
    generic_arch_from_target_id,
    known_arches,
    parse_target_id,
    processor_from_target_id,
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
    ("value", "target_id"),
    [
        # Compilers write the triple's environment field empty or as "unknown".
        ("amdgcn-amd-amdhsa--gfx950", "gfx950"),
        ("amdgcn-amd-amdhsa-unknown-gfx1201", "gfx1201"),
        ("gfx942", "gfx942"),
        ("amdgcn-amd-amdhsa--gfx950:xnack+", "gfx950:xnack+"),
        ("amdgcn-amd-amdhsa-unknown-gfx950:xnack-", "gfx950:xnack-"),
        ("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-"),
        ("amdgcn-amd-amdhsa--gfx11-generic", "gfx11-generic"),
        ("amdgcn-amd-amdhsa-unknown-gfx9-4-generic:xnack+", "gfx9-4-generic:xnack+"),
        ("gfx1250-strict", "gfx1250-strict"),
    ],
)
def test_parse_target_id_accepts_isa_names_and_bare_target_ids(
    value: str, target_id: str
) -> None:
    assert parse_target_id(value) == target_id


@pytest.mark.parametrize(
    "value",
    [
        "",
        "''",
        '""',
        # A processor name in the wrong place, which target_id_from_isa would
        # still pick out of the value.
        "notgfx950",
        "amdgcn-amd-amdhsa--notgfx950",
        "amdgcn-amd-amdhsa-unknown-:gfx950",
        "gfx-named-prefix-gfx942",
        # The triple with no processor, or another environment field.
        "amdgcn-amd-amdhsa-unknown-",
        "amdgcn-amd-amdhsa--",
        "amdgcn-amd-amdhsa-linux-gfx950",
        "amdgcn-amd-amdhsa-opencl-gfx950",
        # A feature without its sign, an empty feature, a trailing separator.
        "gfx950:xnack",
        "gfx950:",
        "gfx950::xnack+",
        "gfx950-",
        "gfx",
        "GFX950",
        " gfx950",
        "gfx950 ",
    ],
)
def test_parse_target_id_refuses_any_other_form(value: str) -> None:
    with pytest.raises(ValueError, match=f"^'{re.escape(value)}' is not a target ID"):
        parse_target_id(value)


@pytest.mark.parametrize(
    ("target_id", "processor"),
    [
        ("gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx11-generic:xnack-", "gfx11-generic"),
        # A concrete processor stays itself; base_arch gives its catalog row.
        ("gfx1100", "gfx1100"),
    ],
)
def test_processor_from_target_id(target_id: str, processor: str) -> None:
    assert processor_from_target_id(target_id) == processor


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


@pytest.mark.parametrize("target_id", ["gfx1170", "gfx1101_bad"])
def test_nonmembers_do_not_use_a_generic_catalog_row(
    target_id: str,
) -> None:
    expected = "gfx1101" if target_id == "gfx1101_bad" else target_id
    assert base_arch_from_target_id(target_id) == expected


@pytest.mark.parametrize(
    ("target_id", "base", "compiler_target"),
    [
        ("gfx1200", "gfx12-generic", "gfx1200"),
        ("gfx1200-strict:xnack-", "gfx12-generic", "gfx1200:xnack-"),
        ("gfx1201", "gfx1201", "gfx1201"),
    ],
)
def test_gfx12_processors_without_a_row_use_the_generic_catalog_row(
    target_id: str, base: str, compiler_target: str
) -> None:
    assert base_arch_from_target_id(target_id) == base
    assert compiler_target_from_target_id(target_id) == compiler_target


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


# Windows' CreateProcess refuses a command line longer than this many characters
# (WinError 206); Linux allows far more, so only the Windows lanes would see it.
_WINDOWS_COMMAND_LINE_LIMIT = 32767


def _native_comparison_inputs() -> list[str]:
    """Every value the native target identity test is compared on.

    The C++ engine keeps its own copy of the generic target table, so every
    generic processor and every member read from arch_specs.json is an input:
    a member missing from, or misspelled in, the C++ copy resolves to a
    different catalog row there and fails the comparison.
    """
    from rocke.core.arch.target import _load_generic_targets

    generic_targets = _load_generic_targets()
    targets = [
        *known_arches(),
        *generic_targets,
        *(member for members in generic_targets.values() for member in members),
        "gfx1101",
        "gfx1170",
        "gfx1101_bad",
        "gfx00a",
        "gfx1100",
        "gfx1102",
        "gfx1154",
        "gfx1200",
        "gfx1010",
        "unexpected-target",
        "",
        "gfx",
        "gfx-",
        "gfx942_bad",
        "notgfx950",
        "GFX950",
    ]
    inputs = [
        f"{prefix}{target}{profile}{features}"
        for target in targets
        for prefix in (
            "",
            "amdgcn-amd-amdhsa--",
            "amdgcn-amd-amdhsa-unknown-",
            "amdgcn-amd-amdhsa-linux-",
            "gfx-named-prefix-",
        )
        for profile in ("", "-strict", "-")
        for features in ("", ":", ":sramecc+:xnack-", ":unknown+", ":xnack", "::xnack+")
    ]
    return inputs + [
        "amdgcn-amd-amdhsa-unknown-",
        "amdgcn-amd-amdhsa-unknown-:gfx950",
        "''",
    ]


def _command_line_batches(executable: str, inputs: list[str]) -> list[list[str]]:
    """Split ``inputs`` so each run of ``executable`` fits a Windows command line.

    The batches keep the order of ``inputs``. Each command line is measured as
    Windows spells it (``subprocess.list2cmdline``, quotes included) and kept
    to half the limit, which leaves room for a longer executable path.
    """
    budget = _WINDOWS_COMMAND_LINE_LIMIT // 2
    batches: list[list[str]] = []
    batch: list[str] = []
    for value in inputs:
        if batch and len(subprocess.list2cmdline([executable, *batch, value])) > budget:
            batches.append(batch)
            batch = []
        batch.append(value)
    if batch:
        batches.append(batch)
    return batches


def test_native_comparison_command_lines_fit_on_windows() -> None:
    """Each run of the native comparison stays under the Windows limit.

    The comparison passes thousands of values on the command line; in one run
    they overflow Windows' limit and the test cannot even start the executable
    there. A long executable path stands in for a deep build directory.
    """
    executable = "C:/" + "build/" * 50 + "rocke_arch_target_identity.exe"
    inputs = _native_comparison_inputs()
    batches = _command_line_batches(executable, inputs)
    assert [value for batch in batches for value in batch] == inputs
    for batch in batches:
        assert (
            len(subprocess.list2cmdline([executable, *batch]))
            < _WINDOWS_COMMAND_LINE_LIMIT
        )


def test_cpp_target_identity_matches_python() -> None:
    """Compare the native CTest executable with Python when a build is supplied.

    Every target identity helper is compared, including the strict parser's
    accepted target IDs and its refusal text, on every value
    ``_native_comparison_inputs`` lists. The values go to the executable in
    batches that fit a Windows command line.
    """
    executable = os.environ.get("ROCKE_ARCH_TARGET_TEST_EXE")
    if not executable:
        pytest.skip("set ROCKE_ARCH_TARGET_TEST_EXE to the native target identity test")
    inputs = _native_comparison_inputs()
    actual = []
    for batch in _command_line_batches(executable, inputs):
        result = subprocess.run(
            [executable, *batch], capture_output=True, text=True, check=True, timeout=30
        )
        actual += [tuple(line.split("\t")) for line in result.stdout.splitlines()]
    expected = []
    for isa in inputs:
        target = target_id_from_isa(isa)
        try:
            parsed = parse_target_id(isa)
        except ValueError as refusal:
            parsed = f"!{refusal}"
        expected.append(
            (
                target,
                base_arch_from_target_id(target),
                compiler_target_from_target_id(target),
                arch_from_isa(isa),
                generic_arch_from_target_id(target) or "-",
                processor_from_target_id(target),
                parsed,
            )
        )
    assert actual == expected
