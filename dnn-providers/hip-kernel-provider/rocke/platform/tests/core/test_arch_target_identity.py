# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import subprocess
import warnings

import pytest

from rocke.core.arch import (
    ArchTarget,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_for_flavor,
    compiler_target_for_llvm_major,
    compiler_target_from_target_id,
    generic_arch_from_target_id,
    known_arches,
    parse_target_id,
    processor_from_target_id,
    target_id_from_isa,
)
from rocke.core.arch.target import _load_generic_targets
from rocke.core.lower_llvm import LLVM_FLAVORS


@pytest.mark.parametrize("arch", known_arches())
def test_known_target_ids_normalize_to_themselves(arch: str) -> None:
    assert base_arch_from_target_id(arch) == arch
    assert compiler_target_from_target_id(arch) == arch
    assert arch_from_isa(f"amdgcn-amd-amdhsa--{arch}") == arch


# What each LLVM flavor's compiler was measured to reject by name: llvm20 knows
# neither gfx1154 nor gfx1250, llvm22 and llvm23 do not know gfx1154, and llvm24
# accepts both.
_REJECTED = {
    "llvm20": {"gfx1154", "gfx1250"},
    "llvm22": {"gfx1154"},
    "llvm23": {"gfx1154"},
    "llvm24": set(),
}


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("arch", known_arches())
def test_each_target_compiles_for_itself_where_its_flavor_accepts_it(
    arch: str, flavor: str
) -> None:
    if arch not in _REJECTED[flavor]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert compiler_target_for_flavor(arch, llvm_flavor=flavor) == arch
        return
    generic = generic_arch_from_target_id(arch)
    if arch == "gfx1250":
        with pytest.raises(ValueError):
            compiler_target_for_flavor(arch, llvm_flavor=flavor)
        return
    with pytest.warns(UserWarning, match=f"compiling for '{generic}' instead"):
        assert compiler_target_for_flavor(arch, llvm_flavor=flavor) == generic


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_gfx1154_compiles_as_gfx11_generic(flavor: str) -> None:
    with pytest.warns(UserWarning) as record:
        target = compiler_target_for_flavor("gfx1154-strict:xnack-", llvm_flavor=flavor)
    assert target == "gfx11-generic:xnack-"
    assert [str(w.message) for w in record] == [
        f"the {flavor} compiler does not accept the processor 'gfx1154'; compiling "
        "for 'gfx11-generic' instead, the generic target its arch_specs.json row "
        "points at, whose code objects load on 'gfx1154'"
    ]


@pytest.mark.parametrize(
    ("llvm_major", "target"),
    [
        (20, "gfx11-generic:xnack-"),
        (23, "gfx11-generic:xnack-"),
        (24, "gfx1154:xnack-"),
        (25, "gfx1154:xnack-"),
        (None, "gfx1154:xnack-"),
    ],
    ids=["llvm20", "llvm23", "llvm24", "newer_than_every_lane", "unknown"],
)
def test_a_compiler_major_routes_through_its_lane_or_keeps_the_processor(
    llvm_major: int | None, target: str
) -> None:
    """A major in a qualified lane routes as that lane's column says (LLVM 20
    and 23 reject gfx1154, LLVM 24 takes it); a newer or unknown compiler was
    never measured, so the processor keeps its name, with no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if target.startswith("gfx11-") else "error")
        assert (
            compiler_target_for_llvm_major("gfx1154-strict:xnack-", llvm_major)
            == target
        )


def test_a_target_no_compiler_can_take_is_refused() -> None:
    with pytest.raises(ValueError) as err:
        compiler_target_for_flavor("gfx1250", llvm_flavor="llvm20")
    assert str(err.value) == (
        "the llvm20 compiler does not accept the processor 'gfx1250', and its "
        "arch_specs.json row points at no generic target that compiler accepts -- "
        "compile with an LLVM flavor whose compiler accepts 'gfx1250'"
    )


@pytest.mark.parametrize("target_id", ["gfx00a", "unexpected-target"])
def test_a_name_no_column_measured_is_left_to_the_compiler(target_id: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compiler_target_for_flavor(target_id, llvm_flavor="llvm20") == target_id


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
        # A generic-target member stays itself, not its generic.
        ("gfx1100", "gfx1100"),
        # Any other name: its leading gfx token, or up to its first ':'.
        ("gfx942_bad:xnack-", "gfx942"),
        ("unexpected-target:xnack-", "unexpected-target"),
    ],
)
def test_processor_from_target_id(target_id: str, processor: str) -> None:
    assert processor_from_target_id(target_id) == processor


@pytest.mark.parametrize(
    ("target_id", "generic"),
    [
        (member, generic)
        for generic, members in _load_generic_targets().items()
        for member in members
    ],
)
def test_generic_members_are_catalog_rows_of_their_own(
    target_id: str, generic: str
) -> None:
    # Every member of a supported generic target has its own row, so catalog
    # lookup keeps the processor's name; its generic is still found by name.
    # The compiler-target normaliser never replaces the processor either.
    assert base_arch_from_target_id(target_id) == target_id
    assert compiler_target_from_target_id(target_id) == target_id
    assert generic_arch_from_target_id(target_id) == generic


@pytest.mark.parametrize(
    ("target_id", "base", "compiler_target"),
    [
        ("gfx1101:unknown+", "gfx1101", "gfx1101:unknown+"),
        ("gfx1101-strict", "gfx1101", "gfx1101"),
        ("gfx1101-strict:unknown+", "gfx1101", "gfx1101:unknown+"),
        ("gfx1200-strict:xnack-", "gfx1200", "gfx1200:xnack-"),
    ],
)
def test_member_catalog_selection_is_independent_of_target_suffixes(
    target_id: str, base: str, compiler_target: str
) -> None:
    assert base_arch_from_target_id(target_id) == base
    assert compiler_target_from_target_id(target_id) == compiler_target


@pytest.mark.parametrize("target_id", ["gfx1170", "gfx1101_bad"])
def test_nonmembers_do_not_use_a_generic_catalog_row(
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


def _native_comparison_inputs() -> list[str]:
    """Every value the native target identity test is compared on.

    The C++ engine keeps its own copy of the generic target table, so every
    generic processor and every member read from arch_specs.json is an input:
    a member missing from, or misspelled in, the C++ copy resolves to a
    different catalog row there and fails the comparison. The prefixes, profiles
    and features include forms the strict parser refuses, so its refusal text is
    compared too.
    """
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


def test_cpp_target_identity_matches_python() -> None:
    """Compare the native CTest executable with Python when a build is supplied.

    Every target identity helper is compared, including the strict parser's
    accepted target IDs and its refusal text and the compile target at every
    LLVM flavor, on every value ``_native_comparison_inputs`` lists.
    """
    executable = os.environ.get("ROCKE_ARCH_TARGET_TEST_EXE")
    if not executable:
        pytest.skip("set ROCKE_ARCH_TARGET_TEST_EXE to the native target identity test")
    inputs = _native_comparison_inputs()
    # The inputs go through standard input, one per line: as arguments they
    # would outgrow the command-line length Windows allows.
    result = subprocess.run(
        [executable, "--query"],
        input="".join(f"{isa}\n" for isa in inputs),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    actual = [tuple(line.split("\t")) for line in result.stdout.splitlines()]
    expected = []
    for isa in inputs:
        target = target_id_from_isa(isa)
        try:
            parsed = parse_target_id(isa)
        except ValueError as refusal:
            parsed = f"!{refusal}"
        row = [
            target,
            base_arch_from_target_id(target),
            compiler_target_from_target_id(target),
            arch_from_isa(isa),
            generic_arch_from_target_id(target) or "-",
            processor_from_target_id(target),
            parsed,
        ]
        for flavor in LLVM_FLAVORS:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                try:
                    row.append(compiler_target_for_flavor(target, llvm_flavor=flavor))
                except ValueError as exc:
                    row.append("!")
                    row.append(str(exc))
                    continue
            row.append(str(record[0].message) if record else "")
        expected.append(tuple(row))
    assert actual == expected
