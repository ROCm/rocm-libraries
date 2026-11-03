# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only golden cases for the explicit gfx950 LDS profile.

The b32 and wave-half cases run for gfx942 and gfx950: both profiles agree.
"""

import pytest

from rocke.analysis.lds import opcodes
from rocke.analysis.lds.model import AccessClassification, GroupKind, LdsAccess
from rocke.analysis.lds.predict import LdsPredictionError, predict_lds_conflicts
from rocke.analysis.lds.profiles import Gfx950Profile
from rocke.tests.arch_marks import arch_specific


def _access(access_id: int, lane: int, address: int, *, width: int = 4) -> LdsAccess:
    return LdsAccess(
        access_id=access_id,
        lane=lane,
        lds_byte_address=address,
        access_width_bytes=width,
    )


def _predict(opcode: str, accesses: list[LdsAccess], target: str = "gfx950"):
    return predict_lds_conflicts(
        target=target, opcode=opcode, wave_size=64, accesses=accesses
    )


def _assert_conflict(result) -> None:
    assert [access.classification for access in result.accesses] == [
        AccessClassification.CONFLICT,
        AccessClassification.CONFLICT,
    ]
    assert result.conflict_groups[0].kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
    assert result.conflict_groups[0].access_ids == (0, 1)


def _assert_normal(result) -> None:
    assert not result.conflict_groups
    assert all(
        access.classification is AccessClassification.NORMAL
        for access in result.accesses
    )


@arch_specific(("gfx950",), reason="pins the gfx950 profile identity and version")
def test_profile_identity_is_explicit():
    result = _predict("ds_read_b32", [_access(0, 0, 0)])

    assert result.profile.target == "gfx950"
    assert result.profile.profile_version == 1


@pytest.mark.parametrize("target", ["gfx942", "gfx950"])
def test_common_b32_conflict_and_no_conflict_cases(target):
    conflict = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 128)], target)
    no_conflict = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 4)], target)

    _assert_conflict(conflict)
    _assert_normal(no_conflict)


@pytest.mark.parametrize("target", ["gfx942", "gfx950"])
def test_common_same_address_read_is_broadcast(target):
    result = _predict("ds_read_b32", [_access(0, 0, 0), _access(1, 1, 0)], target)

    assert [access.classification for access in result.accesses] == [
        AccessClassification.BROADCAST,
        AccessClassification.BROADCAST,
    ]
    assert result.conflict_groups[0].kind is GroupKind.BROADCAST


@arch_specific(
    ("gfx950",),
    reason="gfx950 b64/b128 reads and writes use distinct collision periods; gfx942 follows gfx90a",
)
@pytest.mark.parametrize(
    ("read_opcode", "write_opcode", "width"),
    [
        ("ds_read_b64", "ds_write_b64", 8),
        ("ds_read_b128", "ds_write_b128", 16),
    ],
)
def test_wide_read_and_write_use_distinct_collision_periods(
    read_opcode, write_opcode, width
):
    accesses = [_access(0, 0, 0, width=width), _access(1, 1, 128, width=width)]

    _assert_normal(_predict(read_opcode, accesses))
    _assert_conflict(_predict(write_opcode, accesses))


@arch_specific(
    ("gfx950",),
    reason="gfx950 wide reads conflict at their own repeating address class",
)
@pytest.mark.parametrize(
    ("opcode", "width"),
    [("ds_read_b64", 8), ("ds_read_b128", 16)],
)
def test_wide_reads_conflict_at_their_repeating_address_class(opcode, width):
    result = _predict(
        opcode,
        [_access(0, 0, 0, width=width), _access(1, 1, 256, width=width)],
    )

    _assert_conflict(result)


@arch_specific(
    ("gfx950",),
    reason="gfx950 b64 reads and writes use distinct lane phases; gfx942 follows gfx90a",
)
def test_b64_read_and_write_use_distinct_lane_phases():
    read_accesses = [_access(0, 0, 0, width=8), _access(1, 16, 256, width=8)]
    write_accesses = [_access(0, 0, 0, width=8), _access(1, 16, 128, width=8)]

    _assert_conflict(_predict("ds_read_b64", read_accesses))
    _assert_normal(_predict("ds_write_b64", write_accesses))


@arch_specific(
    ("gfx950",),
    reason="gfx950 b128 reads and writes use distinct lane phases; gfx942 follows gfx90a",
)
def test_b128_read_and_write_use_distinct_lane_phases():
    read_accesses = [_access(0, 0, 0, width=16), _access(1, 4, 256, width=16)]
    write_accesses = [_access(0, 0, 0, width=16), _access(1, 4, 128, width=16)]

    _assert_normal(_predict("ds_read_b128", read_accesses))
    _assert_conflict(_predict("ds_write_b128", write_accesses))


@arch_specific(
    ("gfx950",), reason="pins the gfx950 b128 read phase, which differs from gfx90a"
)
def test_b128_read_phase_differs_from_gfx90a():
    accesses = [_access(0, 0, 0, width=16), _access(1, 12, 256, width=16)]

    gfx950 = _predict("ds_read_b128", accesses)
    gfx90a = predict_lds_conflicts(
        target="gfx90a", opcode="ds_read_b128", wave_size=64, accesses=accesses
    )

    _assert_conflict(gfx950)
    _assert_normal(gfx90a)


@pytest.mark.parametrize("target", ["gfx942", "gfx950"])
def test_wave_half_separation_prevents_conflict(target):
    result = _predict(
        "ds_read_b64",
        [_access(0, 0, 0, width=8), _access(1, 32, 256, width=8)],
        target,
    )

    _assert_normal(result)


@arch_specific(("gfx950",), reason="pins the gfx950 profile's supported opcode set")
@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("method", ["phase_key", "collision_key"])
def test_global_wide_read_addition_does_not_expand_gfx950_support(
    monkeypatch, width, method
):
    opcode = f"ds_read_test_b{width * 8}"
    monkeypatch.setitem(
        opcodes._OPCODE_SPECS, opcode, opcodes.OpcodeSpec(opcode, "read", width)
    )
    profile = Gfx950Profile()
    assert opcode not in profile.supported_opcodes
    with pytest.raises(LdsPredictionError, match="not supported by gfx950"):
        _predict(opcode, [_access(0, 0, 0, width=width)])
    with pytest.raises(ValueError, match="not supported by gfx950"):
        getattr(profile, method)(opcode, 0)
