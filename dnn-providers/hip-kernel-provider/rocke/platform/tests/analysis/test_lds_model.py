# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import FrozenInstanceError, replace

import pytest

from rocke.analysis.lds.model import (
    AccessClassification,
    AccessResult,
    ConflictGroup,
    ConflictSummary,
    Diagnostic,
    DiagnosticSeverity,
    GroupKind,
    LdsAccess,
    LdsConflictResult,
    ModelValidationError,
    NormalizedRequest,
    ProfileIdentity,
)


def _result() -> LdsConflictResult:
    accesses = (
        AccessResult(
            access_id=5,
            lane=1,
            lds_byte_address=128,
            access_width_bytes=4,
            coordinate=(0, 1),
            classification=AccessClassification.CONFLICT,
            conflict_group_ids=(7,),
        ),
        AccessResult(
            access_id=2,
            lane=0,
            lds_byte_address=0,
            access_width_bytes=4,
            coordinate=(0, 0),
            classification=AccessClassification.CONFLICT,
            conflict_group_ids=(7,),
        ),
        AccessResult(
            access_id=9,
            lane=2,
            lds_byte_address=4,
            access_width_bytes=4,
            coordinate=(0, 2),
            classification=AccessClassification.INACTIVE,
        ),
    )
    groups = (
        ConflictGroup(
            group_id=7,
            kind=GroupKind.DISTINCT_ADDRESS_CONFLICT,
            multiplicity=2,
            access_ids=(5, 2),
        ),
    )
    return LdsConflictResult(
        profile=ProfileIdentity(target="gfx90a", profile_version=1),
        request=NormalizedRequest(
            opcode="ds_read_b32",
            direction="read",
            access_width_bytes=4,
            wave_size=64,
            active_lanes=(1, 0),
        ),
        coordinate_axes=("row", "column"),
        accesses=accesses,
        conflict_groups=groups,
        summary=ConflictSummary.from_results(accesses, groups),
        diagnostics=(
            Diagnostic(
                code="example",
                message="Example public diagnostic.",
                severity=DiagnosticSeverity.INFO,
                access_ids=(5, 2),
            ),
        ),
    )


def test_input_access_is_immutable_and_json_primitive_compatible():
    access = LdsAccess(
        access_id=3,
        lane=1,
        lds_byte_address=64,
        access_width_bytes=4,
        coordinate=[2, 4],
        active=False,
    )

    assert access.coordinate == (2, 4)
    assert LdsAccess.from_dict(access.as_dict()) == access
    with pytest.raises(FrozenInstanceError):
        access.lane = 4


def test_input_access_from_minimal_dictionary_uses_public_defaults():
    access = LdsAccess.from_dict(
        {
            "access_id": 3,
            "lane": 1,
            "lds_byte_address": 64,
            "access_width_bytes": 4,
        }
    )

    assert access.coordinate is None
    assert access.active is True


def test_input_access_from_dictionary_rejects_unknown_fields():
    with pytest.raises(ModelValidationError, match="unknown fields: color"):
        LdsAccess.from_dict(
            {
                "access_id": 3,
                "lane": 1,
                "lds_byte_address": 64,
                "access_width_bytes": 4,
                "color": "red",
            }
        )


def test_result_orders_identifier_based_collections_deterministically():
    result = _result()

    assert [access.access_id for access in result.accesses] == [2, 5, 9]
    assert result.conflict_groups[0].access_ids == (2, 5)
    assert result.request.active_lanes == (0, 1)
    assert result.diagnostics[0].access_ids == (2, 5)
    assert result.summary == ConflictSummary(
        active_access_count=2,
        conflicted_access_count=2,
        broadcast_access_count=0,
        inactive_access_count=1,
        conflict_group_count=1,
        maximum_multiplicity=2,
    )


def test_result_dictionary_round_trip():
    result = _result()

    assert LdsConflictResult.from_dict(result.as_dict()) == result


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LdsAccess(True, 0, 0, 4),
        lambda: LdsAccess(0, -1, 0, 4),
        lambda: LdsAccess(0, 0, 0, 0),
        lambda: ProfileIdentity("", 1),
        lambda: NormalizedRequest("ds_read_b32", "load", 4, 64, (0,)),
        lambda: NormalizedRequest("ds_read_b32", "read", 4, 64, (64,)),
        lambda: ConflictGroup(0, GroupKind.BROADCAST, 3, (0, 1)),
    ],
)
def test_component_validation_rejects_invalid_values(factory):
    with pytest.raises(ModelValidationError):
        factory()


def test_result_rejects_inconsistent_group_membership():
    access = AccessResult(
        access_id=0,
        lane=0,
        lds_byte_address=0,
        access_width_bytes=4,
        coordinate=None,
        classification=AccessClassification.CONFLICT,
        conflict_group_ids=(1,),
    )
    group = ConflictGroup(
        group_id=1,
        kind=GroupKind.DISTINCT_ADDRESS_CONFLICT,
        multiplicity=2,
        access_ids=(0, 2),
    )

    with pytest.raises(ModelValidationError, match="unknown access"):
        LdsConflictResult(
            profile=ProfileIdentity("gfx90a", 1),
            request=NormalizedRequest("ds_read_b32", "read", 4, 64, (0,)),
            coordinate_axes=(),
            accesses=(access,),
            conflict_groups=(group,),
            summary=ConflictSummary.from_results((access,), (group,)),
        )


def test_result_rejects_access_width_that_differs_from_request():
    result = _result()
    accesses = (replace(result.accesses[0], access_width_bytes=8),) + result.accesses[
        1:
    ]

    with pytest.raises(ModelValidationError, match="result access width must match"):
        replace(result, accesses=accesses)


def test_result_rejects_broadcast_group_with_distinct_addresses():
    result = _result()
    accesses = tuple(
        (
            replace(access, classification=AccessClassification.BROADCAST)
            if access.classification is AccessClassification.CONFLICT
            else access
        )
        for access in result.accesses
    )
    groups = (replace(result.conflict_groups[0], kind=GroupKind.BROADCAST),)

    with pytest.raises(
        ModelValidationError, match="broadcast group accesses must share"
    ):
        replace(
            result,
            accesses=accesses,
            conflict_groups=groups,
            summary=ConflictSummary.from_results(accesses, groups),
        )


def test_result_rejects_distinct_address_group_with_one_address():
    result = _result()
    accesses = (
        result.accesses[0],
        replace(
            result.accesses[1],
            lds_byte_address=result.accesses[0].lds_byte_address,
        ),
        result.accesses[2],
    )

    with pytest.raises(
        ModelValidationError, match="distinct-address conflict group must contain"
    ):
        replace(result, accesses=accesses)


def test_result_rejects_diagnostic_that_references_unknown_access():
    result = _result()
    diagnostic = replace(result.diagnostics[0], access_ids=(2, 99))

    with pytest.raises(ModelValidationError, match="diagnostic references an unknown"):
        replace(result, diagnostics=(diagnostic,))


def test_result_rejects_summary_that_does_not_match_semantics():
    result = _result()
    bad_summary = ConflictSummary(
        active_access_count=2,
        conflicted_access_count=0,
        broadcast_access_count=0,
        inactive_access_count=1,
        conflict_group_count=0,
        maximum_multiplicity=0,
    )

    with pytest.raises(ModelValidationError, match="summary does not match"):
        LdsConflictResult(
            profile=result.profile,
            request=result.request,
            coordinate_axes=result.coordinate_axes,
            accesses=result.accesses,
            conflict_groups=result.conflict_groups,
            summary=bad_summary,
        )


def test_from_dict_rejects_missing_and_unknown_fields():
    document = _result().as_dict()
    del document["summary"]
    document["presentation"] = {}

    with pytest.raises(ModelValidationError, match="missing required fields: summary"):
        LdsConflictResult.from_dict(document)

    document = _result().as_dict()
    document["presentation"] = {}
    with pytest.raises(ModelValidationError, match="unknown fields: presentation"):
        LdsConflictResult.from_dict(document)
