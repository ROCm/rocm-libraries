# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deterministic, renderer-neutral LDS conflict prediction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .model import (
    AccessClassification,
    AccessResult,
    ConflictGroup,
    ConflictSummary,
    GroupKind,
    LdsAccess,
    LdsConflictResult,
    NormalizedRequest,
)
from .opcodes import get_opcode_spec
from .registry import LdsProfile, resolve_profile


__all__ = ["LdsPredictionError", "predict_lds_conflicts"]


class LdsPredictionError(ValueError):
    """Raised when a request cannot be evaluated by its selected profile."""


def _validate_accesses(
    *,
    profile: LdsProfile,
    opcode: str,
    wave_size: int,
    accesses: Sequence[LdsAccess],
) -> tuple[LdsAccess, ...]:
    if not isinstance(wave_size, int) or isinstance(wave_size, bool):
        raise TypeError("wave_size must be an integer")
    if wave_size not in profile.supported_wave_sizes:
        choices = ", ".join(
            str(value) for value in sorted(profile.supported_wave_sizes)
        )
        raise LdsPredictionError(
            f"unsupported wave_size {wave_size} for {profile.identity.target}; "
            f"supported wave sizes: {choices}"
        )
    if opcode not in profile.supported_opcodes:
        raise LdsPredictionError(
            f"opcode {opcode!r} is not supported by {profile.identity.target}"
        )
    if isinstance(accesses, (str, bytes)) or not isinstance(accesses, Sequence):
        raise TypeError("accesses must be a sequence of LdsAccess values")
    if not all(isinstance(access, LdsAccess) for access in accesses):
        raise TypeError("accesses must contain only LdsAccess values")

    ordered = tuple(sorted(accesses, key=lambda access: access.access_id))
    access_ids = [access.access_id for access in ordered]
    if len(set(access_ids)) != len(access_ids):
        raise LdsPredictionError("access_id values must be unique")

    spec = get_opcode_spec(opcode)
    for access in ordered:
        if access.lane >= wave_size:
            raise LdsPredictionError(
                f"access {access.access_id} lane {access.lane} must be smaller than "
                f"wave_size {wave_size}"
            )
        if access.access_width_bytes != spec.access_width_bytes:
            raise LdsPredictionError(
                f"access {access.access_id} width {access.access_width_bytes} does not "
                f"match {opcode} width {spec.access_width_bytes}"
            )
        if access.lds_byte_address % 4:
            raise LdsPredictionError(
                f"access {access.access_id} address {access.lds_byte_address} must be "
                "dword aligned"
            )
    return ordered


def _build_groups(
    profile: LdsProfile, opcode: str, accesses: Sequence[LdsAccess]
) -> tuple[tuple[ConflictGroup, ...], dict[int, int]]:
    spec = get_opcode_spec(opcode)
    buckets: dict[tuple[tuple[int, ...], int], list[LdsAccess]] = defaultdict(list)
    for access in accesses:
        if not access.active:
            continue
        key = (
            profile.phase_key(opcode, access.lane),
            profile.collision_key(opcode, access.lds_byte_address),
        )
        buckets[key].append(access)

    group_specs: list[
        tuple[tuple[tuple[int, ...], int], GroupKind, tuple[int, ...]]
    ] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        access_ids = tuple(sorted(member.access_id for member in members))
        addresses = {member.lds_byte_address for member in members}
        if len(addresses) == 1:
            if spec.direction == "write":
                continue
            kind = GroupKind.BROADCAST
        else:
            kind = GroupKind.DISTINCT_ADDRESS_CONFLICT
        group_specs.append((key, kind, access_ids))
    group_specs.sort(key=lambda item: (item[0], item[1].value, item[2]))

    groups = tuple(
        ConflictGroup(
            group_id=group_id,
            kind=kind,
            multiplicity=len(access_ids),
            access_ids=access_ids,
        )
        for group_id, (_, kind, access_ids) in enumerate(group_specs)
    )
    group_for_access = {
        access_id: group.group_id for group in groups for access_id in group.access_ids
    }
    return groups, group_for_access


def predict_lds_conflicts(
    *,
    target: str,
    opcode: str,
    wave_size: int,
    accesses: Sequence[LdsAccess],
    coordinate_axes: Sequence[str] = (),
) -> LdsConflictResult:
    """Predict broadcasts and distinct-address conflicts for one LDS operation."""

    if isinstance(coordinate_axes, (str, bytes)) or not isinstance(
        coordinate_axes, Sequence
    ):
        raise TypeError("coordinate_axes must be a sequence of strings")
    if not all(isinstance(axis, str) for axis in coordinate_axes):
        raise TypeError("coordinate_axes must contain only strings")

    profile = resolve_profile(target)
    spec = get_opcode_spec(opcode)
    ordered = _validate_accesses(
        profile=profile,
        opcode=spec.opcode,
        wave_size=wave_size,
        accesses=accesses,
    )
    groups, group_for_access = _build_groups(profile, spec.opcode, ordered)
    group_by_id = {group.group_id: group for group in groups}

    results: list[AccessResult] = []
    for access in ordered:
        group_id = group_for_access.get(access.access_id)
        if not access.active:
            classification = AccessClassification.INACTIVE
            group_ids = ()
        elif group_id is None:
            classification = AccessClassification.NORMAL
            group_ids = ()
        else:
            classification = (
                AccessClassification.BROADCAST
                if group_by_id[group_id].kind is GroupKind.BROADCAST
                else AccessClassification.CONFLICT
            )
            group_ids = (group_id,)
        results.append(
            AccessResult(
                access_id=access.access_id,
                lane=access.lane,
                lds_byte_address=access.lds_byte_address,
                access_width_bytes=access.access_width_bytes,
                coordinate=access.coordinate,
                classification=classification,
                conflict_group_ids=group_ids,
            )
        )

    active_lanes = tuple(sorted({access.lane for access in ordered if access.active}))
    request = NormalizedRequest(
        opcode=spec.opcode,
        direction=spec.direction,
        access_width_bytes=spec.access_width_bytes,
        wave_size=wave_size,
        active_lanes=active_lanes,
    )
    result_accesses = tuple(results)
    return LdsConflictResult(
        profile=profile.identity,
        request=request,
        coordinate_axes=tuple(coordinate_axes),
        accesses=result_accesses,
        conflict_groups=groups,
        summary=ConflictSummary.from_results(result_accesses, groups),
    )
