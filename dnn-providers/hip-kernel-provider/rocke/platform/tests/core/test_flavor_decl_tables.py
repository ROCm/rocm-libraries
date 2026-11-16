# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hold the intrinsic declaration rule in one place.

A compiler lane's declaration table is the base table plus the overrides of
every declaration ABI generation up to and including the one its emission
profile selects, oldest first in ``_LlvmDeclarationAbi`` order, then the
declarations the matrix-call table derives for that lane. Each
generation's override table holds only the entries whose text changed at that
generation. These tests state the rule declaratively against the live tables,
so a second spelling of it (an override table that copies an older one, a
reader that merges differently) fails here rather than as golden-hash churn.

The C++ engine resolves the same chain (``rocke_ll_decl_text`` /
``rocke_ll_decl_at``); ``tests/core/flavor_decl_chain.cpp`` holds its half and
the byte-identity gate holds the two engines together.
"""

from __future__ import annotations

import pytest
from rocke.core.isa.matrix_calls import MATRIX_CALLS
from rocke.core.llvm_profile import _emission_profile, _LlvmDeclarationAbi
from rocke.core.lower_llvm import (
    _INTRINSIC_DECL_OVERRIDES_BY_ABI,
    _INTRINSIC_DECLS,
    _INTRINSIC_DECLS_LLVM22_OVERRIDES,
    LLVM_FLAVOR_LLVM23,
    LLVM_FLAVOR_LLVM24,
    LLVM_FLAVORS,
    _decls_for_flavor,
)

_GENERATIONS = list(_LlvmDeclarationAbi)


def test_every_generation_has_exactly_one_override_table():
    """A generation without an entry would make resolution raise at lowering
    time; an entry for a generation rocke does not know would never be read."""
    assert set(_INTRINSIC_DECL_OVERRIDES_BY_ABI) == set(_GENERATIONS)


def test_overrides_only_change_the_text_of_existing_keys():
    """Call sites name a key, never a flavor; a key that exists only in an
    override table would be a declaration some flavors cannot resolve."""
    for abi, overrides in _INTRINSIC_DECL_OVERRIDES_BY_ABI.items():
        assert sorted(set(overrides) - set(_INTRINSIC_DECLS)) == [], abi


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_a_flavor_resolves_the_newest_override_at_or_below_its_generation(flavor):
    """Each base key takes the text of the newest generation, up to the one
    ``flavor``'s profile selects, that overrides it, else the base text; keys
    keep the base table's order. The matrix-call declarations follow, each
    derived for ``flavor``."""
    abi = _emission_profile(flavor).declaration_abi
    inherited = _GENERATIONS[: _GENERATIONS.index(abi) + 1]
    resolved = _decls_for_flavor(flavor)
    base_count = len(_INTRINSIC_DECLS)

    assert list(resolved)[:base_count] == list(_INTRINSIC_DECLS)
    derived = {call.key: call.declaration(flavor) for call in MATRIX_CALLS.values()}
    assert list(resolved.items())[base_count:] == [
        (key, text) for key, text in derived.items() if key not in _INTRINSIC_DECLS
    ]
    for key, text in list(resolved.items())[:base_count]:
        owners = [g for g in inherited if key in _INTRINSIC_DECL_OVERRIDES_BY_ABI[g]]
        want = (
            _INTRINSIC_DECL_OVERRIDES_BY_ABI[owners[-1]][key]
            if owners
            else _INTRINSIC_DECLS[key]
        )
        assert text == want, (flavor, key)


@pytest.mark.parametrize("abi", _GENERATIONS[1:])
def test_an_override_table_holds_only_its_own_changes(abi):
    """An entry equal to what the previous generation already resolves is a
    copy of an inherited entry: a second home that drifts when the older one
    is edited."""
    parent = _GENERATIONS[_GENERATIONS.index(abi) - 1]
    parent_table = dict(_INTRINSIC_DECLS)
    for older in _GENERATIONS[: _GENERATIONS.index(parent) + 1]:
        parent_table.update(_INTRINSIC_DECL_OVERRIDES_BY_ABI[older])
    copies = sorted(
        key
        for key, text in _INTRINSIC_DECL_OVERRIDES_BY_ABI[abi].items()
        if parent_table[key] == text
    )
    assert copies == [], f"{abi} re-lists entries it inherits: {copies}"


@pytest.mark.parametrize("flavor", [LLVM_FLAVOR_LLVM23, LLVM_FLAVOR_LLVM24])
def test_newer_generations_inherit_the_llvm22_signatures(flavor):
    """The concrete case the rule replaced: the LLVM 23 and LLVM 24
    generations once copied LLVM 22's table to get these entries; they now
    reach them through inheritance."""
    resolved = _decls_for_flavor(flavor)
    for key, text in _INTRINSIC_DECLS_LLVM22_OVERRIDES.items():
        assert resolved[key] == text, key


def test_lanes_that_share_a_profile_share_one_table():
    """Two compiler lanes with the same declaration generation resolve the
    same table, items and order."""
    for flavor in LLVM_FLAVORS:
        for other in LLVM_FLAVORS:
            same = (
                _emission_profile(flavor).declaration_abi
                is _emission_profile(other).declaration_abi
            )
            if same:
                assert list(_decls_for_flavor(flavor).items()) == list(
                    _decls_for_flavor(other).items()
                ), (flavor, other)


def test_an_unknown_flavor_is_refused():
    """No silent fallback to the base table for a misspelled flavor."""
    with pytest.raises(
        ValueError,
        match=r"^unknown LLVM flavor 'llvm99'; expected one of ",
    ):
        _decls_for_flavor("llvm99")
