# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Renderer-neutral semantic model for LDS conflict analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


class ModelValidationError(ValueError):
    """Raised when an LDS semantic document is malformed or inconsistent."""


class AccessClassification(str, Enum):
    """Stable per-access states used by profiles and serialized results.

    A profile assigns one state to each access. The string values are part of the
    renderer-neutral schema and must remain stable for result consumers.
    """

    NORMAL = "normal"
    CONFLICT = "conflict"
    BROADCAST = "broadcast"
    INACTIVE = "inactive"


class GroupKind(str, Enum):
    """Semantic group types emitted by an LDS profile.

    The values distinguish a predicted distinct-address conflict from a broadcast
    without requiring a renderer to infer the meaning from access data.
    """

    DISTINCT_ADDRESS_CONFLICT = "distinct-address-conflict"
    BROADCAST = "broadcast"


class DiagnosticSeverity(str, Enum):
    """Stable severity levels for diagnostics in serialized results.

    Severity lets a renderer choose how to present a diagnostic. It does not change
    the access classifications or conflict summary.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise ModelValidationError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ModelValidationError(f"{name} must be at least {minimum}")
    return result


def _require_str(name: str, value: object, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ModelValidationError(f"{name} must be a string")
    if nonempty and not value:
        raise ModelValidationError(f"{name} must not be empty")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ModelValidationError(f"{name} must be a boolean")
    return value


def _tuple_of_ints(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    unique: bool = False,
    sorted_values: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{name} must be an array of integers")
    result = tuple(
        _require_int(f"{name}[{index}]", item, minimum=minimum)
        for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        raise ModelValidationError(f"{name} must not contain duplicates")
    if sorted_values:
        result = tuple(sorted(result))
    return result


def _tuple_of_strings(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ModelValidationError(f"{name} must be an array of strings")
    result = tuple(
        _require_str(f"{name}[{index}]", item) for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ModelValidationError(f"{name} must not contain duplicates")
    return result


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ModelValidationError(f"{name} keys must be strings")
    return value


def _fields(
    name: str,
    value: object,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    data = _mapping(name, value)
    optional = optional or set()
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise ModelValidationError(
            f"{name} is missing required fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise ModelValidationError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return data


def _enum(name: str, enum_type: type[Enum], value: object) -> Enum:
    if not isinstance(value, str):
        raise ModelValidationError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ModelValidationError(f"{name} must be one of: {choices}") from exc


@dataclass(frozen=True)
class LdsAccess:
    """One normalized LDS access supplied to a profile for classification.

    The LDS address and access width are measured in bytes. A coordinate may link
    the access to caller-defined axes, while ``active`` controls whether the profile
    includes it in the prediction.
    """

    access_id: int
    lane: int
    lds_byte_address: int
    access_width_bytes: int
    coordinate: tuple[int, ...] | None = None
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "access_id", _require_int("access_id", self.access_id, minimum=0)
        )
        object.__setattr__(self, "lane", _require_int("lane", self.lane, minimum=0))
        object.__setattr__(
            self,
            "lds_byte_address",
            _require_int("lds_byte_address", self.lds_byte_address, minimum=0),
        )
        object.__setattr__(
            self,
            "access_width_bytes",
            _require_int("access_width_bytes", self.access_width_bytes, minimum=1),
        )
        if self.coordinate is not None:
            object.__setattr__(
                self, "coordinate", _tuple_of_ints("coordinate", self.coordinate)
            )
        object.__setattr__(self, "active", _require_bool("active", self.active))

    def as_dict(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "lane": self.lane,
            "lds_byte_address": self.lds_byte_address,
            "access_width_bytes": self.access_width_bytes,
            "coordinate": (
                list(self.coordinate) if self.coordinate is not None else None
            ),
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, value: object) -> "LdsAccess":
        data = _fields(
            "access",
            value,
            required={"access_id", "lane", "lds_byte_address", "access_width_bytes"},
            optional={"coordinate", "active"},
        )
        coordinate = data.get("coordinate")
        if coordinate is not None:
            coordinate = _tuple_of_ints("access.coordinate", coordinate)
        return cls(
            access_id=data["access_id"],
            lane=data["lane"],
            lds_byte_address=data["lds_byte_address"],
            access_width_bytes=data["access_width_bytes"],
            coordinate=coordinate,
            active=data.get("active", True),
        )


@dataclass(frozen=True)
class NormalizedRequest:
    """Canonical operation metadata stored with every prediction result.

    The opcode and direction describe the operation interpreted by the profile.
    Access width is measured in bytes, and active lanes are indexes within the
    declared wave size.
    """

    opcode: str
    direction: str
    access_width_bytes: int
    wave_size: int
    active_lanes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "opcode", _require_str("opcode", self.opcode))
        object.__setattr__(self, "direction", _require_str("direction", self.direction))
        if self.direction not in {"read", "write"}:
            raise ModelValidationError("direction must be one of: read, write")
        object.__setattr__(
            self,
            "access_width_bytes",
            _require_int("access_width_bytes", self.access_width_bytes, minimum=1),
        )
        object.__setattr__(
            self, "wave_size", _require_int("wave_size", self.wave_size, minimum=1)
        )
        active_lanes = _tuple_of_ints(
            "active_lanes",
            self.active_lanes,
            minimum=0,
            unique=True,
            sorted_values=True,
        )
        if any(lane >= self.wave_size for lane in active_lanes):
            raise ModelValidationError("active_lanes must be smaller than wave_size")
        object.__setattr__(self, "active_lanes", active_lanes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode,
            "direction": self.direction,
            "access_width_bytes": self.access_width_bytes,
            "wave_size": self.wave_size,
            "active_lanes": list(self.active_lanes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "NormalizedRequest":
        data = _fields(
            "request",
            value,
            required={
                "opcode",
                "direction",
                "access_width_bytes",
                "wave_size",
                "active_lanes",
            },
        )
        return cls(
            opcode=data["opcode"],
            direction=data["direction"],
            access_width_bytes=data["access_width_bytes"],
            wave_size=data["wave_size"],
            active_lanes=_tuple_of_ints("request.active_lanes", data["active_lanes"]),
        )


@dataclass(frozen=True)
class ProfileIdentity:
    """Identify the exact target rules used to produce a prediction.

    ``profile_version`` versions the rules independently of the result schema. The
    target is the selected profile target, not a fallback or a compatible alias.
    """

    target: str
    profile_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _require_str("target", self.target))
        object.__setattr__(
            self,
            "profile_version",
            _require_int("profile_version", self.profile_version, minimum=1),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"target": self.target, "profile_version": self.profile_version}

    @classmethod
    def from_dict(cls, value: object) -> "ProfileIdentity":
        data = _fields("profile", value, required={"target", "profile_version"})
        return cls(target=data["target"], profile_version=data["profile_version"])


@dataclass(frozen=True)
class AccessResult:
    """One profile-classified access in the renderer-neutral result.

    The LDS address and access width remain in bytes. Conflict group IDs connect the
    access to the semantic groups that explain a conflict or broadcast state.
    """

    access_id: int
    lane: int
    lds_byte_address: int
    access_width_bytes: int
    coordinate: tuple[int, ...] | None
    classification: AccessClassification
    conflict_group_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "access_id", _require_int("access_id", self.access_id, minimum=0)
        )
        object.__setattr__(self, "lane", _require_int("lane", self.lane, minimum=0))
        object.__setattr__(
            self,
            "lds_byte_address",
            _require_int("lds_byte_address", self.lds_byte_address, minimum=0),
        )
        object.__setattr__(
            self,
            "access_width_bytes",
            _require_int("access_width_bytes", self.access_width_bytes, minimum=1),
        )
        if self.coordinate is not None:
            object.__setattr__(
                self, "coordinate", _tuple_of_ints("coordinate", self.coordinate)
            )
        if not isinstance(self.classification, AccessClassification):
            raise ModelValidationError("classification must be an AccessClassification")
        object.__setattr__(
            self,
            "conflict_group_ids",
            _tuple_of_ints(
                "conflict_group_ids",
                self.conflict_group_ids,
                minimum=0,
                unique=True,
                sorted_values=True,
            ),
        )
        if (
            self.classification
            in {AccessClassification.NORMAL, AccessClassification.INACTIVE}
            and self.conflict_group_ids
        ):
            raise ModelValidationError(
                "normal and inactive accesses cannot reference conflict groups"
            )
        if (
            self.classification
            in {AccessClassification.CONFLICT, AccessClassification.BROADCAST}
            and not self.conflict_group_ids
        ):
            raise ModelValidationError(
                "conflict and broadcast accesses must reference a conflict group"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "lane": self.lane,
            "lds_byte_address": self.lds_byte_address,
            "access_width_bytes": self.access_width_bytes,
            "coordinate": (
                list(self.coordinate) if self.coordinate is not None else None
            ),
            "classification": self.classification.value,
            "conflict_group_ids": list(self.conflict_group_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "AccessResult":
        data = _fields(
            "access_result",
            value,
            required={
                "access_id",
                "lane",
                "lds_byte_address",
                "access_width_bytes",
                "coordinate",
                "classification",
                "conflict_group_ids",
            },
        )
        coordinate = data["coordinate"]
        if coordinate is not None:
            coordinate = _tuple_of_ints("access_result.coordinate", coordinate)
        return cls(
            access_id=data["access_id"],
            lane=data["lane"],
            lds_byte_address=data["lds_byte_address"],
            access_width_bytes=data["access_width_bytes"],
            coordinate=coordinate,
            classification=_enum(
                "access_result.classification",
                AccessClassification,
                data["classification"],
            ),
            conflict_group_ids=_tuple_of_ints(
                "access_result.conflict_group_ids", data["conflict_group_ids"]
            ),
        )


@dataclass(frozen=True)
class ConflictGroup:
    """A semantic access group produced by an LDS profile.

    The kind records whether the profile predicted a distinct-address conflict or a
    broadcast. ``multiplicity`` is the number of access records in the group, not a
    measured performance value.
    """

    group_id: int
    kind: GroupKind
    multiplicity: int
    access_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "group_id", _require_int("group_id", self.group_id, minimum=0)
        )
        if not isinstance(self.kind, GroupKind):
            raise ModelValidationError("kind must be a GroupKind")
        object.__setattr__(
            self,
            "multiplicity",
            _require_int("multiplicity", self.multiplicity, minimum=2),
        )
        access_ids = _tuple_of_ints(
            "access_ids", self.access_ids, minimum=0, unique=True, sorted_values=True
        )
        object.__setattr__(self, "access_ids", access_ids)
        if self.multiplicity != len(access_ids):
            raise ModelValidationError(
                "multiplicity must equal the number of access_ids"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "kind": self.kind.value,
            "multiplicity": self.multiplicity,
            "access_ids": list(self.access_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictGroup":
        data = _fields(
            "conflict_group",
            value,
            required={"group_id", "kind", "multiplicity", "access_ids"},
        )
        return cls(
            group_id=data["group_id"],
            kind=_enum("conflict_group.kind", GroupKind, data["kind"]),
            multiplicity=data["multiplicity"],
            access_ids=_tuple_of_ints("conflict_group.access_ids", data["access_ids"]),
        )


@dataclass(frozen=True)
class ConflictSummary:
    """A compact index derived from classified accesses and conflict groups.

    The counts support filtering and presentation without reprocessing every access.
    They describe the prediction result and are not profiler or performance data.
    """

    active_access_count: int
    conflicted_access_count: int
    broadcast_access_count: int
    inactive_access_count: int
    conflict_group_count: int
    maximum_multiplicity: int

    def __post_init__(self) -> None:
        for field_name in (
            "active_access_count",
            "conflicted_access_count",
            "broadcast_access_count",
            "inactive_access_count",
            "conflict_group_count",
            "maximum_multiplicity",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_int(field_name, getattr(self, field_name), minimum=0),
            )
        if (
            self.conflicted_access_count + self.broadcast_access_count
            > self.active_access_count
        ):
            raise ModelValidationError(
                "classified active access counts exceed active_access_count"
            )

    @classmethod
    def from_results(
        cls, accesses: Sequence[AccessResult], groups: Sequence[ConflictGroup]
    ) -> "ConflictSummary":
        active = sum(
            access.classification is not AccessClassification.INACTIVE
            for access in accesses
        )
        conflicted = sum(
            access.classification is AccessClassification.CONFLICT
            for access in accesses
        )
        broadcast = sum(
            access.classification is AccessClassification.BROADCAST
            for access in accesses
        )
        conflict_groups = [
            group
            for group in groups
            if group.kind is GroupKind.DISTINCT_ADDRESS_CONFLICT
        ]
        return cls(
            active_access_count=active,
            conflicted_access_count=conflicted,
            broadcast_access_count=broadcast,
            inactive_access_count=len(accesses) - active,
            conflict_group_count=len(conflict_groups),
            maximum_multiplicity=max(
                (group.multiplicity for group in conflict_groups), default=0
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_access_count": self.active_access_count,
            "conflicted_access_count": self.conflicted_access_count,
            "broadcast_access_count": self.broadcast_access_count,
            "inactive_access_count": self.inactive_access_count,
            "conflict_group_count": self.conflict_group_count,
            "maximum_multiplicity": self.maximum_multiplicity,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSummary":
        required = {
            "active_access_count",
            "conflicted_access_count",
            "broadcast_access_count",
            "inactive_access_count",
            "conflict_group_count",
            "maximum_multiplicity",
        }
        data = _fields("summary", value, required=required)
        return cls(**{field_name: data[field_name] for field_name in required})


@dataclass(frozen=True)
class Diagnostic:
    """A machine-readable prediction note with optional access references.

    ``code`` is the stable identifier for result consumers. ``message`` is public
    text for people, and ``access_ids`` identifies the related access records.
    """

    code: str
    message: str
    severity: DiagnosticSeverity
    access_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_str("code", self.code))
        object.__setattr__(self, "message", _require_str("message", self.message))
        if not isinstance(self.severity, DiagnosticSeverity):
            raise ModelValidationError("severity must be a DiagnosticSeverity")
        object.__setattr__(
            self,
            "access_ids",
            _tuple_of_ints(
                "diagnostic.access_ids",
                self.access_ids,
                minimum=0,
                unique=True,
                sorted_values=True,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "access_ids": list(self.access_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "Diagnostic":
        data = _fields(
            "diagnostic", value, required={"code", "message", "severity", "access_ids"}
        )
        return cls(
            code=data["code"],
            message=data["message"],
            severity=_enum("diagnostic.severity", DiagnosticSeverity, data["severity"]),
            access_ids=_tuple_of_ints("diagnostic.access_ids", data["access_ids"]),
        )


@dataclass(frozen=True)
class LdsConflictResult:
    """Versioned, renderer-neutral output of LDS conflict prediction.

    This is the boundary between profile evaluation and later presentation backends.
    It validates cross-references and derived counts, and it can be serialized without
    any renderer-specific state.
    """

    profile: ProfileIdentity
    request: NormalizedRequest
    coordinate_axes: tuple[str, ...]
    accesses: tuple[AccessResult, ...]
    conflict_groups: tuple[ConflictGroup, ...]
    summary: ConflictSummary
    diagnostics: tuple[Diagnostic, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ProfileIdentity):
            raise ModelValidationError("profile must be a ProfileIdentity")
        if not isinstance(self.request, NormalizedRequest):
            raise ModelValidationError("request must be a NormalizedRequest")
        if not isinstance(self.summary, ConflictSummary):
            raise ModelValidationError("summary must be a ConflictSummary")
        schema_version = _require_int("schema_version", self.schema_version, minimum=1)
        if schema_version != SCHEMA_VERSION:
            raise ModelValidationError(f"unsupported schema_version: {schema_version}")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(
            self,
            "coordinate_axes",
            _tuple_of_strings("coordinate_axes", self.coordinate_axes),
        )

        if not isinstance(self.accesses, (list, tuple)) or not all(
            isinstance(access, AccessResult) for access in self.accesses
        ):
            raise ModelValidationError("accesses must contain only AccessResult values")
        if not isinstance(self.conflict_groups, (list, tuple)) or not all(
            isinstance(group, ConflictGroup) for group in self.conflict_groups
        ):
            raise ModelValidationError(
                "conflict_groups must contain only ConflictGroup values"
            )
        if not isinstance(self.diagnostics, (list, tuple)) or not all(
            isinstance(diagnostic, Diagnostic) for diagnostic in self.diagnostics
        ):
            raise ModelValidationError(
                "diagnostics must contain only Diagnostic values"
            )

        accesses = tuple(sorted(self.accesses, key=lambda access: access.access_id))
        groups = tuple(sorted(self.conflict_groups, key=lambda group: group.group_id))
        diagnostics = tuple(
            sorted(
                self.diagnostics,
                key=lambda item: (
                    item.code,
                    item.severity.value,
                    item.access_ids,
                    item.message,
                ),
            )
        )
        object.__setattr__(self, "accesses", accesses)
        object.__setattr__(self, "conflict_groups", groups)
        object.__setattr__(self, "diagnostics", diagnostics)

        access_by_id = {access.access_id: access for access in accesses}
        group_by_id = {group.group_id: group for group in groups}
        if len(access_by_id) != len(accesses):
            raise ModelValidationError("access_id values must be unique")
        if len(group_by_id) != len(groups):
            raise ModelValidationError("group_id values must be unique")
        if any(access.lane >= self.request.wave_size for access in accesses):
            raise ModelValidationError(
                "access lane must be smaller than request.wave_size"
            )
        active_lanes = {
            access.lane
            for access in accesses
            if access.classification is not AccessClassification.INACTIVE
        }
        if active_lanes != set(self.request.active_lanes):
            raise ModelValidationError(
                "request.active_lanes must match active result accesses"
            )

        coordinate_rank = len(self.coordinate_axes)
        for access in accesses:
            if access.access_width_bytes != self.request.access_width_bytes:
                raise ModelValidationError(
                    "result access width must match request.access_width_bytes"
                )
            if (
                access.coordinate is not None
                and len(access.coordinate) != coordinate_rank
            ):
                raise ModelValidationError(
                    "access coordinate rank must match coordinate_axes"
                )
            for group_id in access.conflict_group_ids:
                group = group_by_id.get(group_id)
                if group is None or access.access_id not in group.access_ids:
                    raise ModelValidationError(
                        "access references an unknown or inconsistent conflict group"
                    )
            expected_kind = {
                AccessClassification.CONFLICT: GroupKind.DISTINCT_ADDRESS_CONFLICT,
                AccessClassification.BROADCAST: GroupKind.BROADCAST,
            }.get(access.classification)
            if expected_kind is not None and any(
                group_by_id[group_id].kind is not expected_kind
                for group_id in access.conflict_group_ids
            ):
                raise ModelValidationError(
                    "access classification does not match its conflict group kind"
                )

        for group in groups:
            if any(access_id not in access_by_id for access_id in group.access_ids):
                raise ModelValidationError(
                    "conflict group references an unknown access"
                )
            if any(
                group.group_id not in access_by_id[access_id].conflict_group_ids
                for access_id in group.access_ids
            ):
                raise ModelValidationError(
                    "conflict group membership must be referenced by each access"
                )
            addresses = {
                access_by_id[access_id].lds_byte_address
                for access_id in group.access_ids
            }
            if group.kind is GroupKind.BROADCAST and len(addresses) != 1:
                raise ModelValidationError(
                    "broadcast group accesses must share one LDS byte address"
                )
            if group.kind is GroupKind.DISTINCT_ADDRESS_CONFLICT and len(addresses) < 2:
                raise ModelValidationError(
                    "distinct-address conflict group must contain distinct LDS byte "
                    "addresses"
                )

        if any(
            access_id not in access_by_id
            for diagnostic in diagnostics
            for access_id in diagnostic.access_ids
        ):
            raise ModelValidationError("diagnostic references an unknown access")

        expected_summary = ConflictSummary.from_results(accesses, groups)
        if self.summary != expected_summary:
            raise ModelValidationError(
                "summary does not match accesses and conflict_groups"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.as_dict(),
            "request": self.request.as_dict(),
            "coordinate_axes": list(self.coordinate_axes),
            "accesses": [access.as_dict() for access in self.accesses],
            "conflict_groups": [group.as_dict() for group in self.conflict_groups],
            "summary": self.summary.as_dict(),
            "diagnostics": [diagnostic.as_dict() for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: object) -> "LdsConflictResult":
        data = _fields(
            "result",
            value,
            required={
                "schema_version",
                "profile",
                "request",
                "coordinate_axes",
                "accesses",
                "conflict_groups",
                "summary",
                "diagnostics",
            },
        )
        schema_version = _require_int(
            "result.schema_version", data["schema_version"], minimum=1
        )
        if schema_version != SCHEMA_VERSION:
            raise ModelValidationError(f"unsupported schema_version: {schema_version}")
        for array_name in ("accesses", "conflict_groups", "diagnostics"):
            if not isinstance(data[array_name], list):
                raise ModelValidationError(f"result.{array_name} must be an array")
        return cls(
            schema_version=schema_version,
            profile=ProfileIdentity.from_dict(data["profile"]),
            request=NormalizedRequest.from_dict(data["request"]),
            coordinate_axes=_tuple_of_strings(
                "result.coordinate_axes", data["coordinate_axes"]
            ),
            accesses=tuple(AccessResult.from_dict(item) for item in data["accesses"]),
            conflict_groups=tuple(
                ConflictGroup.from_dict(item) for item in data["conflict_groups"]
            ),
            summary=ConflictSummary.from_dict(data["summary"]),
            diagnostics=tuple(
                Diagnostic.from_dict(item) for item in data["diagnostics"]
            ),
        )
