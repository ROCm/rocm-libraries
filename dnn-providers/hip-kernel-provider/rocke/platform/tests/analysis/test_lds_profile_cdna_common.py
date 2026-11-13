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
from rocke.tests.arch_marks import COVERED_ARCHES


def _classes(result) -> list[AccessClassification]:
    return [access.classification for access in result.accesses]


@pytest.mark.parametrize("target", COVERED_ARCHES)
@pytest.mark.parametrize("multiplicity", [4, 8])
def test_explicit_multiway_conflict_group(target, multiplicity):
    accesses = [LdsAccess(lane, lane, lane * 128, 4) for lane in range(multiplicity)]

    result = predict_lds_conflicts(
        target=target, opcode="ds_read_b32", wave_size=64, accesses=accesses
    )

    assert len(result.conflict_groups) == 1
    group = result.conflict_groups[0]
    assert group.kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert group.access_ids == tuple(range(multiplicity))
    assert group.multiplicity == multiplicity
    assert result.summary.maximum_multiplicity == multiplicity
    assert all(c is AccessClassification.CONFLICT for c in _classes(result))


@pytest.mark.parametrize("target", COVERED_ARCHES)
def test_eight_distinct_address_classes_are_conflict_free(target):
    accesses = [LdsAccess(lane, lane, lane * 4, 4) for lane in range(8)]

    result = predict_lds_conflicts(
        target=target, opcode="ds_read_b32", wave_size=64, accesses=accesses
    )

    assert not result.conflict_groups
    assert result.summary.conflict_group_count == 0
    assert result.summary.maximum_multiplicity == 0
    assert all(c is AccessClassification.NORMAL for c in _classes(result))


@pytest.mark.parametrize("target", COVERED_ARCHES)
def test_b32_conflict_and_no_conflict_cases(target):
    conflict = predict_lds_conflicts(
        target=target,
        opcode="ds_read_b32",
        wave_size=64,
        accesses=[LdsAccess(0, 0, 0, 4), LdsAccess(1, 1, 128, 4)],
    )
    no_conflict = predict_lds_conflicts(
        target=target,
        opcode="ds_read_b32",
        wave_size=64,
        accesses=[LdsAccess(0, 0, 0, 4), LdsAccess(1, 1, 4, 4)],
    )

    assert _classes(conflict) == [AccessClassification.CONFLICT] * 2
    assert conflict.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert conflict.conflict_groups[0].access_ids == (0, 1)
    assert not no_conflict.conflict_groups
    assert _classes(no_conflict) == [AccessClassification.NORMAL] * 2


@pytest.mark.parametrize("target", COVERED_ARCHES)
def test_same_address_read_is_broadcast(target):
    result = predict_lds_conflicts(
        target=target,
        opcode="ds_read_b32",
        wave_size=64,
        accesses=[LdsAccess(0, 0, 0, 4), LdsAccess(1, 1, 0, 4)],
    )

    assert _classes(result) == [AccessClassification.BROADCAST] * 2
    assert result.conflict_groups[0].kind is GroupKind.BROADCAST


@pytest.mark.parametrize("target", COVERED_ARCHES)
def test_b32_wave_half_separation_remains_conflict_free(target):
    result = predict_lds_conflicts(
        target=target,
        opcode="ds_read_b32",
        wave_size=64,
        accesses=[LdsAccess(0, 0, 0, 4), LdsAccess(1, 32, 128, 4)],
    )

    assert not result.conflict_groups


@pytest.mark.parametrize("target", COVERED_ARCHES)
def test_b64_wave_half_separation_prevents_conflict(target):
    result = predict_lds_conflicts(
        target=target,
        opcode="ds_read_b64",
        wave_size=64,
        accesses=[LdsAccess(0, 0, 0, 8), LdsAccess(1, 32, 256, 8)],
    )

    assert not result.conflict_groups
    assert _classes(result) == [AccessClassification.NORMAL] * 2
