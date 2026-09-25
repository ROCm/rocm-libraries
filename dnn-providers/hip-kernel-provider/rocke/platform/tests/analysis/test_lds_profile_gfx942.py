# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only golden cases for the explicit gfx942 LDS profile."""

import pytest

from rocke.analysis.lds.model import AccessClassification, GroupKind, LdsAccess
from rocke.analysis.lds.predict import predict_lds_conflicts
from rocke.analysis.lds.registry import resolve_profile


def _access(access_id: int, lane: int, address: int, *, width: int = 4) -> LdsAccess:
    return LdsAccess(
        access_id=access_id,
        lane=lane,
        lds_byte_address=address,
        access_width_bytes=width,
    )


def _predict(target: str, opcode: str, accesses: list[LdsAccess]):
    return predict_lds_conflicts(
        target=target, opcode=opcode, wave_size=64, accesses=accesses
    )


def test_gfx942_has_distinct_profile_identity():
    gfx90a = resolve_profile("gfx90a")
    gfx942 = resolve_profile("gfx942")

    assert gfx942 is not gfx90a
    assert gfx942.identity.target == "gfx942"
    assert gfx942.identity != gfx90a.identity


@pytest.mark.parametrize("multiplicity", [4, 8])
def test_explicit_multiway_conflict_group(multiplicity):
    accesses = [
        _access(access_id=lane, lane=lane, address=lane * 128)
        for lane in range(multiplicity)
    ]

    result = _predict("gfx942", "ds_read_b32", accesses)

    assert len(result.conflict_groups) == 1
    group = result.conflict_groups[0]
    assert group.kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert group.access_ids == tuple(range(multiplicity))
    assert group.multiplicity == multiplicity
    assert result.summary.maximum_multiplicity == multiplicity
    assert all(
        access.classification is AccessClassification.CONFLICT
        for access in result.accesses
    )


def test_eight_distinct_address_classes_are_conflict_free():
    accesses = [
        _access(access_id=lane, lane=lane, address=lane * 4) for lane in range(8)
    ]

    result = _predict("gfx942", "ds_read_b32", accesses)

    assert not result.conflict_groups
    assert result.summary.conflict_group_count == 0
    assert result.summary.maximum_multiplicity == 0
    assert all(
        access.classification is AccessClassification.NORMAL
        for access in result.accesses
    )


def test_wave_half_separation_remains_conflict_free():
    result = _predict(
        "gfx942",
        "ds_read_b32",
        [_access(0, 0, 0), _access(1, 32, 128)],
    )

    assert not result.conflict_groups


@pytest.mark.parametrize(
    ("opcode", "width"),
    [
        ("ds_read_b32", 4),
        ("ds_write_b32", 4),
        ("ds_read_b64", 8),
        ("ds_write_b64", 8),
        ("ds_read_b128", 16),
        ("ds_write_b128", 16),
    ],
)
def test_reviewed_scope_agrees_with_gfx90a(opcode, width):
    accesses = [_access(0, 0, 0, width=width), _access(1, 1, 128, width=width)]

    gfx942 = _predict("gfx942", opcode, accesses)
    gfx90a = _predict("gfx90a", opcode, accesses)

    assert gfx942.profile.target == "gfx942"
    assert gfx90a.profile.target == "gfx90a"
    assert gfx942.request == gfx90a.request
    assert gfx942.accesses == gfx90a.accesses
    assert gfx942.conflict_groups == gfx90a.conflict_groups
    assert gfx942.summary == gfx90a.summary
