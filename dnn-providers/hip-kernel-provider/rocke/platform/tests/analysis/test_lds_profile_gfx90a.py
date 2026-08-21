# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only golden cases for the explicit gfx90a LDS profile."""

import pytest

from rocke.analysis.lds.model import AccessClassification, GroupKind, LdsAccess
from rocke.analysis.lds.opcodes import UnsupportedLdsOpcodeError
from rocke.analysis.lds.predict import LdsPredictionError, predict_lds_conflicts


def _access(
    access_id: int,
    lane: int,
    address: int,
    *,
    width: int = 4,
    active: bool = True,
) -> LdsAccess:
    return LdsAccess(
        access_id=access_id,
        lane=lane,
        lds_byte_address=address,
        access_width_bytes=width,
        active=active,
    )


def _predict(opcode: str, accesses: list[LdsAccess]):
    return predict_lds_conflicts(
        target="gfx90a", opcode=opcode, wave_size=64, accesses=accesses
    )


def test_distinct_addresses_in_one_phase_conflict():
    result = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 128)])

    assert [access.classification for access in result.accesses] == [
        AccessClassification.CONFLICT,
        AccessClassification.CONFLICT,
    ]
    assert result.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert result.conflict_groups[0].access_ids == (0, 1)
    assert result.summary.maximum_multiplicity == 2


def test_distinct_address_classes_do_not_conflict():
    result = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 4)])

    assert not result.conflict_groups
    assert all(
        access.classification is AccessClassification.NORMAL
        for access in result.accesses
    )


def test_same_address_is_broadcast_not_conflict():
    result = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 0)])

    assert [access.classification for access in result.accesses] == [
        AccessClassification.BROADCAST,
        AccessClassification.BROADCAST,
    ]
    assert result.conflict_groups[0].kind is GroupKind.BROADCAST
    assert result.summary.conflict_group_count == 0
    assert result.summary.broadcast_access_count == 2


def test_same_address_writes_are_not_classified_as_broadcasts():
    result = _predict("ds_write_b32", [_access(0, 0, 0), _access(1, 1, 0)])

    assert not result.conflict_groups
    assert all(
        access.classification is AccessClassification.NORMAL
        for access in result.accesses
    )


def test_wave_half_separation_prevents_conflict():
    result = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 32, 128)])

    assert not result.conflict_groups
    assert all(
        access.classification is AccessClassification.NORMAL
        for access in result.accesses
    )


@pytest.mark.parametrize("opcode", ["ds_read_b64", "ds_write_b64"])
def test_b64_lane_phase_separation_prevents_conflict(opcode):
    result = _predict(
        opcode,
        [_access(0, 0, 0, width=8), _access(1, 16, 128, width=8)],
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
def test_supported_operations_conflict_within_one_lane_phase(opcode, width):
    result = _predict(
        opcode,
        [_access(0, 0, 0, width=width), _access(1, 1, 128, width=width)],
    )

    assert result.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT


@pytest.mark.parametrize("opcode", ["ds_read_b128", "ds_write_b128"])
def test_b128_lane_phase_separation_prevents_conflict(opcode):
    result = _predict(
        opcode,
        [_access(0, 0, 0, width=16), _access(1, 8, 128, width=16)],
    )

    assert not result.conflict_groups


def test_b128_read_and_write_have_distinct_lane_phases():
    accesses = [_access(0, 0, 0, width=16), _access(1, 4, 128, width=16)]

    read_result = _predict("ds_read_b128", accesses)
    write_result = _predict("ds_write_b128", accesses)

    assert not read_result.conflict_groups
    assert write_result.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT


def test_multiway_grouping_uses_access_multiplicity():
    result = _predict(
        "ds_read_b32",
        [_access(access_id, access_id, access_id * 128) for access_id in range(4)],
    )

    assert len(result.conflict_groups) == 1
    assert result.conflict_groups[0].access_ids == (0, 1, 2, 3)
    assert result.conflict_groups[0].multiplicity == 4
    assert result.summary.maximum_multiplicity == 4


def test_inactive_access_is_reported_but_excluded_from_groups():
    result = _predict(
        "ds_read_b32",
        [_access(0, 0, 0), _access(1, 1, 128), _access(2, 2, 256, active=False)],
    )

    assert result.accesses[2].classification is AccessClassification.INACTIVE
    assert result.conflict_groups[0].access_ids == (0, 1)
    assert result.request.active_lanes == (0, 1)
    assert result.summary.inactive_access_count == 1


def test_prediction_is_deterministic_for_unsorted_inputs():
    accesses = [_access(9, 1, 128), _access(3, 0, 0)]

    first = _predict("DS_READ_B32", accesses)
    second = _predict("ds_read_b32", list(reversed(accesses)))

    assert first == second
    assert tuple(access.access_id for access in first.accesses) == (3, 9)
    assert first.request.opcode == "ds_read_b32"


@pytest.mark.parametrize(
    ("opcode", "accesses", "message"),
    [
        ("ds_read_b32", [_access(0, 64, 0)], "lane 64"),
        ("ds_read_b64", [_access(0, 0, 0)], "does not match"),
        ("ds_read_b32", [_access(0, 0, 2)], "dword aligned"),
        ("ds_read_b32", [_access(0, 0, 0), _access(0, 1, 128)], "must be unique"),
    ],
)
def test_invalid_accesses_are_rejected(opcode, accesses, message):
    with pytest.raises(LdsPredictionError, match=message):
        _predict(opcode, accesses)


def test_unsupported_wave_size_is_rejected():
    with pytest.raises(LdsPredictionError, match="unsupported wave_size"):
        predict_lds_conflicts(
            target="gfx90a",
            opcode="ds_read_b32",
            wave_size=32,
            accesses=[_access(0, 0, 0)],
        )


def test_unsupported_opcode_is_rejected():
    with pytest.raises(UnsupportedLdsOpcodeError, match="unsupported LDS opcode"):
        _predict("ds_read_b96", [_access(0, 0, 0, width=12)])


def test_coordinate_axes_must_be_a_string_sequence():
    with pytest.raises(TypeError, match="sequence of strings"):
        predict_lds_conflicts(
            target="gfx90a",
            opcode="ds_read_b32",
            wave_size=64,
            accesses=[_access(0, 0, 0)],
            coordinate_axes="row",
        )
