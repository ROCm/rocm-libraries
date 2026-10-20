# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hold the per-flavor intrinsic declaration rule in one place.

A flavor's declaration table is the base table plus the overrides of every
flavor up to and including it, oldest first in ``LLVM_FLAVORS`` order. Each
flavor's override table holds only the entries whose text changed at that
flavor. These tests state the rule declaratively against the live tables, so a
second spelling of it (an override table that copies an older one, a reader that
merges differently) fails here rather than as golden-hash churn.

The C++ engine resolves the same chain (``rocke_ll_decl_text`` /
``rocke_ll_decl_at``); ``tests/core/flavor_decl_chain.cpp`` holds its half and
the byte-identity gate holds the two engines together.
"""

from __future__ import annotations

import pytest
from rocke.core.lower_llvm import (
    _INTRINSIC_DECL_OVERRIDES_BY_FLAVOR,
    _INTRINSIC_DECLS,
    _INTRINSIC_DECLS_LLVM22_OVERRIDES,
    LLVM_FLAVOR_LLVM23,
    LLVM_FLAVORS,
    _decls_for_flavor,
)


def test_every_flavor_has_exactly_one_override_table():
    """A flavor without an entry would make resolution raise at lowering time;
    an entry for a flavor rocke does not know would never be read."""
    assert set(_INTRINSIC_DECL_OVERRIDES_BY_FLAVOR) == set(LLVM_FLAVORS)


def test_overrides_only_change_the_text_of_existing_keys():
    """Call sites name a key, never a flavor; a key that exists only in an
    override table would be a declaration some flavors cannot resolve."""
    for flavor, overrides in _INTRINSIC_DECL_OVERRIDES_BY_FLAVOR.items():
        assert sorted(set(overrides) - set(_INTRINSIC_DECLS)) == [], flavor


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_a_flavor_resolves_the_newest_override_at_or_below_it(flavor):
    """Each key takes the text of the newest flavor, up to ``flavor``, that
    overrides it, else the base text; keys keep the base table's order."""
    inherited = LLVM_FLAVORS[: LLVM_FLAVORS.index(flavor) + 1]
    resolved = _decls_for_flavor(flavor)

    assert list(resolved) == list(_INTRINSIC_DECLS)
    for key, text in resolved.items():
        owners = [f for f in inherited if key in _INTRINSIC_DECL_OVERRIDES_BY_FLAVOR[f]]
        want = (
            _INTRINSIC_DECL_OVERRIDES_BY_FLAVOR[owners[-1]][key]
            if owners
            else _INTRINSIC_DECLS[key]
        )
        assert text == want, (flavor, key)


@pytest.mark.parametrize("flavor", LLVM_FLAVORS[1:])
def test_an_override_table_holds_only_its_own_changes(flavor):
    """An entry equal to what the previous flavor already resolves is a copy
    of an inherited entry: a second home that drifts when the older one is
    edited."""
    parent = LLVM_FLAVORS[LLVM_FLAVORS.index(flavor) - 1]
    parent_table = _decls_for_flavor(parent)
    copies = sorted(
        key
        for key, text in _INTRINSIC_DECL_OVERRIDES_BY_FLAVOR[flavor].items()
        if parent_table[key] == text
    )
    assert copies == [], f"{flavor} re-lists entries it inherits: {copies}"


def test_llvm23_inherits_the_llvm22_signatures():
    """The concrete case the rule replaced: llvm23 once copied llvm22's table
    to get these entries; it now reaches them through inheritance."""
    resolved = _decls_for_flavor(LLVM_FLAVOR_LLVM23)
    for key, text in _INTRINSIC_DECLS_LLVM22_OVERRIDES.items():
        assert resolved[key] == text, key


def test_an_unknown_flavor_is_refused():
    """No silent fallback to the base table for a misspelled flavor."""
    with pytest.raises(
        ValueError,
        match=r"^unknown LLVM flavor 'llvm99'; expected one of "
        r"\('llvm20', 'llvm22', 'llvm23'\)$",
    ):
        _decls_for_flavor("llvm99")
