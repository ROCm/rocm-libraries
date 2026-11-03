# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only LDS profile cases that the gfx942 and gfx950 profiles share.

Both profiles give the same answer for b32 accesses and for the wave-half
separation. Cases that differ between the profiles are in
`test_lds_profile_gfx942.py` and `test_lds_profile_gfx950.py`.
"""

import pytest
from rocke.analysis.lds.model import AccessClassification, GroupKind, LdsAccess
from rocke.analysis.lds.predict import predict_lds_conflicts

_CDNA_TARGETS = ["gfx942", "gfx950"]


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


def _classes(result) -> list[AccessClassification]:
    return [access.classification for access in result.accesses]


@pytest.mark.parametrize("target", _CDNA_TARGETS)
@pytest.mark.parametrize("multiplicity", [4, 8])
def test_explicit_multiway_conflict_group(target, multiplicity):
    accesses = [
        _access(access_id=lane, lane=lane, address=lane * 128)
        for lane in range(multiplicity)
    ]

    result = _predict(target, "ds_read_b32", accesses)

    assert len(result.conflict_groups) == 1
    group = result.conflict_groups[0]
    assert group.kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert group.access_ids == tuple(range(multiplicity))
    assert group.multiplicity == multiplicity
    assert result.summary.maximum_multiplicity == multiplicity
    assert all(c is AccessClassification.CONFLICT for c in _classes(result))


@pytest.mark.parametrize("target", _CDNA_TARGETS)
def test_eight_distinct_address_classes_are_conflict_free(target):
    accesses = [
        _access(access_id=lane, lane=lane, address=lane * 4) for lane in range(8)
    ]

    result = _predict(target, "ds_read_b32", accesses)

    assert not result.conflict_groups
    assert result.summary.conflict_group_count == 0
    assert result.summary.maximum_multiplicity == 0
    assert all(c is AccessClassification.NORMAL for c in _classes(result))


@pytest.mark.parametrize("target", _CDNA_TARGETS)
def test_b32_conflict_and_no_conflict_cases(target):
    conflict = _predict(target, "ds_read_b32", [_access(0, 0, 0), _access(1, 1, 128)])
    no_conflict = _predict(target, "ds_read_b32", [_access(0, 0, 0), _access(1, 1, 4)])

    assert _classes(conflict) == [AccessClassification.CONFLICT] * 2
    assert conflict.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert conflict.conflict_groups[0].access_ids == (0, 1)
    assert not no_conflict.conflict_groups
    assert _classes(no_conflict) == [AccessClassification.NORMAL] * 2


@pytest.mark.parametrize("target", _CDNA_TARGETS)
def test_same_address_read_is_broadcast(target):
    result = _predict(target, "ds_read_b32", [_access(0, 0, 0), _access(1, 1, 0)])

    assert _classes(result) == [AccessClassification.BROADCAST] * 2
    assert result.conflict_groups[0].kind is GroupKind.BROADCAST


@pytest.mark.parametrize("target", _CDNA_TARGETS)
def test_b32_wave_half_separation_remains_conflict_free(target):
    result = _predict(target, "ds_read_b32", [_access(0, 0, 0), _access(1, 32, 128)])

    assert not result.conflict_groups


@pytest.mark.parametrize("target", _CDNA_TARGETS)
def test_b64_wave_half_separation_prevents_conflict(target):
    result = _predict(
        target,
        "ds_read_b64",
        [_access(0, 0, 0, width=8), _access(1, 32, 256, width=8)],
    )

    assert not result.conflict_groups
    assert _classes(result) == [AccessClassification.NORMAL] * 2
