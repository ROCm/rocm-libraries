# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from lib.arch import (
    STRICT_SWITCH,
    build_alias_of,
    canonical_arch,
    device_requirement,
    library_subtree,
    mode_dependent,
    reported_arch,
)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gfx950", "gfx950"),
        ("gfx942", "gfx942"),
        ("gfx1201", "gfx1201"),
        ("gfx90a", "gfx90a"),
        ("gfx1250", "gfx1250"),
        ("gfx1250v0", "gfx1250"),
        ("gfx1250v12", "gfx1250"),
        ("gfx1250-strict", "gfx1250"),
        ("gfx1250v0:xnack+", "gfx1250"),
        ("gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx942-xnack+", "gfx942"),
        ("gfx942-sramecc+-xnack-", "gfx942"),
        ("  gfx950 ", "gfx950"),
    ],
)
def test_canonical_arch(name, expected):
    assert canonical_arch(name) == expected


@pytest.mark.parametrize("empty", ["", None])
def test_canonical_arch_passes_empty_through(empty):
    assert canonical_arch(empty) == empty


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gfx1250v0", "gfx1250-strict"),
        ("gfx1250v0:xnack+", "gfx1250-strict"),
        ("gfx1250-strict", None),
        ("gfx1250", None),
        ("gfx950", None),
        ("", None),
        (None, None),
    ],
)
def test_build_alias_of(name, expected):
    assert build_alias_of(name) == expected


@pytest.mark.parametrize(
    "reported, revision, subtree",
    [
        ("gfx1250", 0, "gfx1250v0"),
        ("gfx1250-strict", 0, "gfx1250-strict"),
        ("gfx1250", 1, "gfx1250"),
        ("gfx1250", 2, "gfx1250"),
        ("gfx1250", None, "gfx1250"),
        ("gfx942", 0, "gfx942"),
        ("gfx950", 0, "gfx950"),
        ("gfx1250v0", 0, "gfx1250v0"),
        ("gfx1250-strict", 1, "gfx1250-strict"),
    ],
)
def test_library_subtree_is_hipblaslts(reported, revision, subtree):
    assert library_subtree(reported, revision) == subtree


@pytest.mark.parametrize(
    "target, revision, switch, reported",
    [
        ("gfx1250", 0, None, "gfx1250"),
        ("gfx1250", 0, "0", "gfx1250-strict"),
        ("gfx1250", 0, " 0 ", "gfx1250-strict"),
        ("gfx1250", 0, "1", "gfx1250"),
        ("gfx1250", 1, "0", "gfx1250"),
        ("gfx1250", None, "0", "gfx1250"),
        ("gfx950", 0, "0", "gfx950"),
    ],
)
def test_the_strict_name_needs_revision_0_and_the_switch(
    target, revision, switch, reported
):
    environ = {} if switch is None else {STRICT_SWITCH: switch}
    assert reported_arch(target, revision, environ) == reported


def test_only_mode_dependent_arches_have_a_device_requirement():
    for arch in ("gfx950", "gfx942:sramecc+:xnack-", "gfx90a"):
        assert not mode_dependent(arch) and device_requirement(arch) is None
    strict = device_requirement("gfx1250-strict")
    assert mode_dependent("gfx1250-strict") and f"{STRICT_SWITCH}=0" in strict
    assert "of ASIC revision 0" in strict
    base = device_requirement("gfx1250")
    assert mode_dependent("gfx1250") and "other than 0" in base
