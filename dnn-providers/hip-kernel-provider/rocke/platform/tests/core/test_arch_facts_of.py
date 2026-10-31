# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A target row may take its facts from the generic target that lists it.

``arch_specs.json`` keeps a row for every member of a supported generic target,
even when every fact it records equals its generic target's (gfx1151, gfx1201).
Such a row is a pointer, ``{"facts_of": <generic>}``, so each fact has one home.
These tests hold the pointer contract: the loader resolves every pointer before
any reader sees it, the resolved target is the generic's target under the
processor's own name, and a malformed pointer is refused with an exact error.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.arch import domain as arch_domain
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.core.arch.target import (
    _load_doc,
    _load_generic_targets,
    _load_specs,
    _op_id_c_dtype,
    _op_id_family,
    _resolve_row,
    normalize_dtype,
)

# Every member of a generic target rocKE supports has a row of its own; these
# members' recorded facts are all their generic's, so each row is a pointer to
# the generic target that lists it (read from generic_targets, not restated).
POINTERS = {
    member: generic
    for generic, members in _load_generic_targets().items()
    for member in members
}

_GENERIC_TARGETS = {
    "gfx11-generic": ("gfx1100", "gfx1151"),
    "gfx12-generic": ("gfx1200", "gfx1201"),
}


def _fact_row(**extra) -> dict:
    """A minimal fact row; _resolve_row only distinguishes facts from pointers."""
    return {"family": "rdna", "wave_size": 32, "mma": [], **extra}


def test_shipped_pointer_rows():
    """The file's pointers are exactly the generic members, each naming its generic."""
    raw = _load_doc()["arches"]
    pointers = {gfx: row["facts_of"] for gfx, row in raw.items() if "facts_of" in row}
    assert pointers == POINTERS
    for gfx in POINTERS:
        assert set(raw[gfx]) <= {"facts_of", "_comment"}


def test_no_resolved_row_is_a_pointer():
    """Every reader of _load_specs() sees fact rows only."""
    for gfx, row in _load_specs().items():
        assert "facts_of" not in row, gfx
        assert {"family", "wave_size", "mma", "memory", "limits"} <= set(row), gfx


@pytest.mark.parametrize("gfx,generic", sorted(POINTERS.items()))
def test_pointer_target_is_the_generic_target_under_its_own_name(gfx, generic):
    """Field by field, the processor's target equals its generic's, except gfx."""
    target = ArchTarget.from_gfx(gfx)
    generic_target = ArchTarget.from_gfx(generic)
    assert gfx in known_arches()
    assert target.gfx == gfx
    for field in dataclasses.fields(ArchTarget):
        if field.name in ("gfx", "mma"):
            continue
        assert getattr(target, field.name) == getattr(
            generic_target, field.name
        ), field.name
    assert target.mma.ops == generic_target.mma.ops


@pytest.mark.parametrize("gfx,generic", sorted(POINTERS.items()))
def test_resolved_pointer_row_is_an_independent_copy(gfx, generic):
    """Editing one target's resolved row never edits another target's."""
    specs = _load_specs()
    assert specs[gfx] is not specs[generic]
    assert specs[gfx]["mma"] is not specs[generic]["mma"]
    assert specs[gfx]["mma"] == specs[generic]["mma"]
    assert specs[gfx]["_comment"] == _load_doc()["arches"][gfx]["_comment"]


def test_pointers_add_nothing_to_the_op_id_lookups():
    """The bare-op_id lookups answer exactly what the fact rows alone answer."""
    expected_c: dict = {}
    expected_family: dict = {}
    for row in _load_specs().values():
        for op in row["mma"]:
            expected_c.setdefault(op["op_id"], normalize_dtype(op["c"]))
            expected_family.setdefault(op["op_id"], op["family"])
    fact_rows_only = {
        op["op_id"]
        for gfx, row in _load_specs().items()
        if gfx not in POINTERS
        for op in row["mma"]
    }
    assert set(expected_c) == fact_rows_only
    assert _op_id_c_dtype() == expected_c
    assert _op_id_family() == expected_family


def test_every_generic_target_member_has_a_row():
    """Support is explicit: a supported generic's members are all known arches."""
    for generic, members in _load_generic_targets().items():
        assert generic in known_arches()
        for member in members:
            assert member in known_arches(), (generic, member)


def _measured_cells(column, gfx):
    return {key: column.lookup(key, gfx) for key in column.keys()}


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("gfx,generic", sorted(POINTERS.items()))
def test_pointer_has_its_generics_measured_compiler_data(gfx, generic, flavor):
    """The compiler agrees with the pointer: in every committed arch-domain
    column, each declaration the member was probed for links or fails exactly as
    it does for its generic. Where the flavor's compiler does not accept the
    member's name, every probed cell says so and none answers for it.
    """
    column = arch_domain.load(flavor)
    if column is None:
        pytest.skip(f"no committed {flavor} arch-domain column")
    member = _measured_cells(column, gfx)
    reference = _measured_cells(column, generic)
    assert None not in member.values()
    unsupported = {
        key
        for key, cell in member.items()
        if cell.status == arch_domain.STATUS_TARGET_UNSUPPORTED
    }
    if not unsupported:
        assert member == reference
        return
    for key, cell in member.items():
        assert cell.status != arch_domain.STATUS_OK, key
        assert (
            cell == reference[key]
            or cell.status == arch_domain.STATUS_TARGET_UNSUPPORTED
        ), key


def test_fact_row_resolves_to_itself():
    arches = {"gfx11-generic": _fact_row()}
    assert (
        _resolve_row(arches, _GENERIC_TARGETS, "gfx11-generic")
        is arches["gfx11-generic"]
    )


def test_pointer_resolves_to_its_generic_row():
    arches = {
        "gfx11-generic": _fact_row(),
        "gfx1151": {"_comment": "a member", "facts_of": "gfx11-generic"},
    }
    assert _resolve_row(arches, _GENERIC_TARGETS, "gfx1151") is arches["gfx11-generic"]


def test_refuses_a_pointer_to_a_missing_row():
    arches = {
        "gfx11-generic": _fact_row(),
        "gfx1151": {"facts_of": "gfx11-genric"},
    }
    with pytest.raises(ValueError) as err:
        _resolve_row(arches, _GENERIC_TARGETS, "gfx1151")
    assert str(err.value) == (
        "arch_specs.json row 'gfx1151' takes its facts from 'gfx11-genric', "
        "which is not a row"
    )


def test_refuses_a_chained_pointer():
    arches = {
        "gfx11-generic": {"facts_of": "gfx12-generic"},
        "gfx12-generic": _fact_row(),
        "gfx1151": {"facts_of": "gfx11-generic"},
    }
    with pytest.raises(ValueError) as err:
        _resolve_row(arches, _GENERIC_TARGETS, "gfx1151")
    assert str(err.value) == (
        "arch_specs.json row 'gfx1151' takes its facts from 'gfx11-generic', "
        "which itself takes them from 'gfx12-generic' -- reference the row that "
        "holds the facts"
    )


def test_refuses_a_pointer_with_its_own_facts():
    arches = {
        "gfx11-generic": _fact_row(),
        "gfx1151": {"facts_of": "gfx11-generic", "lds_capacity_bytes": 65536},
    }
    with pytest.raises(ValueError) as err:
        _resolve_row(arches, _GENERIC_TARGETS, "gfx1151")
    assert str(err.value) == (
        "arch_specs.json row 'gfx1151' has 'facts_of' and its own facts "
        "['lds_capacity_bytes'] -- a row either holds facts or names the row "
        "that does"
    )


def test_refuses_a_pointer_to_another_generic():
    arches = {
        "gfx11-generic": _fact_row(),
        "gfx12-generic": _fact_row(),
        "gfx1201": {"facts_of": "gfx11-generic"},
    }
    with pytest.raises(ValueError) as err:
        _resolve_row(arches, _GENERIC_TARGETS, "gfx1201")
    assert str(err.value) == (
        "arch_specs.json row 'gfx1201' takes its facts from 'gfx11-generic', but "
        "generic_targets lists 'gfx1201' under 'gfx12-generic' -- a row may take "
        "its facts only from the generic target that lists it"
    )


def test_refuses_a_pointer_from_a_processor_no_generic_lists():
    arches = {"gfx11-generic": _fact_row(), "gfx950": {"facts_of": "gfx11-generic"}}
    with pytest.raises(ValueError) as err:
        _resolve_row(arches, _GENERIC_TARGETS, "gfx950")
    assert str(err.value) == (
        "arch_specs.json row 'gfx950' takes its facts from 'gfx11-generic', but "
        "generic_targets lists 'gfx950' under no generic target -- a row may take "
        "its facts only from the generic target that lists it"
    )


def test_refuses_a_pointer_to_a_processor_row():
    """A pointer names the generic, never a sibling processor that holds facts."""
    arches = {
        "gfx11-generic": _fact_row(),
        "gfx1100": _fact_row(),
        "gfx1151": {"facts_of": "gfx1100"},
    }
    with pytest.raises(ValueError, match="under 'gfx11-generic'"):
        _resolve_row(arches, _GENERIC_TARGETS, "gfx1151")
