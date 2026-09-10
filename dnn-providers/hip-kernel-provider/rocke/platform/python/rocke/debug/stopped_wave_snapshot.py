# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collect, validate, and persist presentation-neutral stopped-wave snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA = "rocke-value-snapshot/v1"
MANIFEST_SCHEMA = "rocke-debug-manifest/v1"
CAPTURE_STATUSES = ("available", "optimized_out", "location_unavailable")


def _uint32(value: Any, context: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= 0xFFFFFFFF
    ):
        raise ValueError(f"{context} must be an unsigned 32-bit integer")
    return value


def _optional_triplet(value: Any, context: str) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(axis, int) or isinstance(axis, bool) or axis < 0
            for axis in value
        )
    ):
        raise ValueError(f"{context} must be three non-negative integers or null")
    return tuple(value)


def _optional_hex(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{context} must be a hexadecimal string or null")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} must be a hexadecimal string or null") from error
    return value


@dataclass(frozen=True)
class CapturedLocation:
    expression: str
    raw_words: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.expression:
            raise ValueError("captured location expression must not be empty")
        if not self.raw_words:
            raise ValueError(f"captured location {self.expression!r} has no lane words")
        for lane, word in enumerate(self.raw_words):
            _uint32(word, f"{self.expression} lane {lane}")

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "raw_words": list(self.raw_words)}

    @classmethod
    def from_dict(cls, record: Any) -> CapturedLocation:
        if not isinstance(record, dict):
            raise TypeError("captured location must be an object")
        expression = record.get("expression")
        raw_words = record.get("raw_words")
        if not isinstance(expression, str):
            raise TypeError("captured location expression must be a string")
        if not isinstance(raw_words, list):
            raise TypeError("captured location raw_words must be a list")
        return cls(expression, tuple(raw_words))


@dataclass(frozen=True)
class CapturedValue:
    name: str
    status: str
    locations: tuple[CapturedLocation, ...] = ()
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("captured value name must not be empty")
        if self.status not in CAPTURE_STATUSES:
            raise ValueError(f"unsupported capture status {self.status!r}")
        if self.status == "available" and not self.locations:
            raise ValueError("available captured value must have locations")
        if self.status != "available" and self.locations:
            raise ValueError("unavailable captured value cannot have locations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "locations": [location.to_dict() for location in self.locations],
        }

    @classmethod
    def from_dict(cls, record: Any) -> CapturedValue:
        if not isinstance(record, dict):
            raise TypeError("captured value must be an object")
        name = record.get("name")
        status = record.get("status")
        detail = record.get("detail")
        locations = record.get("locations")
        if not isinstance(name, str) or not isinstance(status, str):
            raise TypeError("captured value name and status must be strings")
        if detail is not None and not isinstance(detail, str):
            raise TypeError("captured value detail must be a string or null")
        if not isinstance(locations, list):
            raise TypeError("captured value locations must be a list")
        return cls(
            name=name,
            status=status,
            detail=detail,
            locations=tuple(CapturedLocation.from_dict(item) for item in locations),
        )


@dataclass(frozen=True)
class WaveCapture:
    thread_id: str
    pc: str | None
    exec: str | None
    status: str
    values: tuple[CapturedValue, ...]
    dispatch_id: str | None = None
    workgroup: tuple[int, int, int] | None = None
    wave_position: tuple[int, int, int] | None = None
    kernel_pc_offset: str | None = None

    def __post_init__(self) -> None:
        if not self.thread_id:
            raise ValueError("wave thread_id must not be empty")
        if self.status not in ("available", "partial"):
            raise ValueError(f"unsupported wave status {self.status!r}")
        _optional_hex(self.pc, "wave pc")
        _optional_hex(self.exec, "wave exec")
        _optional_hex(self.kernel_pc_offset, "wave kernel_pc_offset")
        if len({value.name for value in self.values}) != len(self.values):
            raise ValueError("captured value names must be unique within a wave")
        expected = (
            "available"
            if self.values and all(value.status == "available" for value in self.values)
            else "partial"
        )
        if self.status != expected:
            raise ValueError(
                f"wave status {self.status!r} does not match captured values "
                f"({expected})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "dispatch_id": self.dispatch_id,
            "workgroup": None if self.workgroup is None else list(self.workgroup),
            "wave_position": (
                None if self.wave_position is None else list(self.wave_position)
            ),
            "pc": self.pc,
            "kernel_pc_offset": self.kernel_pc_offset,
            "exec": self.exec,
            "status": self.status,
            "values": [value.to_dict() for value in self.values],
        }

    @classmethod
    def from_dict(cls, record: Any) -> WaveCapture:
        if not isinstance(record, dict):
            raise TypeError("wave capture must be an object")
        thread_id = record.get("thread_id")
        dispatch_id = record.get("dispatch_id")
        status = record.get("status")
        values = record.get("values")
        if not isinstance(thread_id, str) or not isinstance(status, str):
            raise TypeError("wave thread_id and status must be strings")
        if dispatch_id is not None and not isinstance(dispatch_id, str):
            raise TypeError("wave dispatch_id must be a string or null")
        if not isinstance(values, list):
            raise TypeError("wave values must be a list")
        return cls(
            thread_id=thread_id,
            dispatch_id=dispatch_id,
            workgroup=_optional_triplet(record.get("workgroup"), "wave workgroup"),
            wave_position=_optional_triplet(
                record.get("wave_position"), "wave position"
            ),
            pc=_optional_hex(record.get("pc"), "wave pc"),
            kernel_pc_offset=_optional_hex(
                record.get("kernel_pc_offset"), "wave kernel_pc_offset"
            ),
            exec=_optional_hex(record.get("exec"), "wave exec"),
            status=status,
            values=tuple(CapturedValue.from_dict(item) for item in values),
        )


@dataclass(frozen=True)
class ValueSnapshot:
    capture: Mapping[str, Any]
    target: Mapping[str, Any]
    values: tuple[dict[str, Any], ...]
    waves: tuple[WaveCapture, ...]

    def __post_init__(self) -> None:
        if self.capture.get("scope") not in ("wave", "block"):
            raise ValueError("snapshot capture scope must be 'wave' or 'block'")
        if self.capture.get("stop_mode") not in ("all-stop", "non-stop"):
            raise ValueError(
                "snapshot capture stop_mode must be 'all-stop' or 'non-stop'"
            )
        if not isinstance(self.capture.get("complete"), bool):
            raise TypeError("snapshot capture complete must be a boolean")
        names = []
        wave_sizes: dict[str, int] = {}
        for value in self.values:
            try:
                name = value["logical"]["name"]
                wave_size = value["logical"]["layout"]["wave_size"]
                locations = value["binding"]["locations"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "snapshot contains an invalid value specification"
                ) from error
            if not isinstance(name, str) or not name:
                raise ValueError("snapshot logical value name must not be empty")
            if not isinstance(wave_size, int) or wave_size <= 0:
                raise ValueError(f"snapshot value {name!r} has invalid wave size")
            if not isinstance(locations, list) or not locations:
                raise ValueError(f"snapshot value {name!r} has no binding locations")
            names.append(name)
            wave_sizes[name] = wave_size
        if len(set(names)) != len(names):
            raise ValueError("snapshot logical value names must be unique")
        expected_names = set(names)
        for wave in self.waves:
            captured_names = {value.name for value in wave.values}
            if captured_names != expected_names:
                raise ValueError(
                    f"wave {wave.thread_id!r} captured values "
                    f"{sorted(captured_names)!r}; "
                    f"expected {sorted(expected_names)!r}"
                )
            for value in wave.values:
                if value.status != "available":
                    continue
                expected_expressions = next(
                    spec["binding"]["locations"]
                    for spec in self.values
                    if spec["logical"]["name"] == value.name
                )
                expressions = [location.expression for location in value.locations]
                if expressions != expected_expressions:
                    raise ValueError(
                        f"captured locations for {value.name!r} do not match "
                        "its binding"
                    )
                if any(
                    len(location.raw_words) != wave_sizes[value.name]
                    for location in value.locations
                ):
                    raise ValueError(
                        f"captured locations for {value.name!r} do not match wave size"
                    )
        complete = bool(self.waves) and all(
            wave.status == "available" for wave in self.waves
        )
        if self.capture["complete"] != complete:
            raise ValueError(
                "snapshot capture completeness does not match captured wave status"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "collector": {"name": "rocke rocGDB", "version": 1},
            "capture": deepcopy(dict(self.capture)),
            "target": deepcopy(dict(self.target)),
            "values": deepcopy(list(self.values)),
            "waves": [wave.to_dict() for wave in self.waves],
        }

    @classmethod
    def from_dict(cls, record: Any) -> ValueSnapshot:
        if not isinstance(record, dict):
            raise TypeError("snapshot must be an object")
        if record.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError(
                f"unsupported snapshot schema {record.get('schema')!r}; "
                f"expected {SNAPSHOT_SCHEMA!r}"
            )
        capture = record.get("capture")
        target = record.get("target")
        values = record.get("values")
        waves = record.get("waves")
        if not isinstance(capture, dict) or not isinstance(target, dict):
            raise TypeError("snapshot capture and target must be objects")
        if not isinstance(values, list) or not isinstance(waves, list):
            raise TypeError("snapshot values and waves must be lists")
        return cls(
            capture=deepcopy(capture),
            target=deepcopy(target),
            values=tuple(deepcopy(values)),
            waves=tuple(WaveCapture.from_dict(wave) for wave in waves),
        )


def _manifest_values(
    manifest: Mapping[str, Any], names: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported debug manifest schema {manifest.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    available = manifest.get("values")
    if not isinstance(available, list):
        raise TypeError("debug manifest 'values' must be a list")
    selected = []
    for name in names:
        matches = [
            value
            for value in available
            if isinstance(value, dict) and value.get("logical", {}).get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"debug manifest must contain exactly one value named {name!r}; "
                f"found {len(matches)}"
            )
        selected.append(deepcopy(matches[0]))
    return tuple(selected)


def collect_selected_wave(
    manifest: Mapping[str, Any],
    names: Sequence[str],
    *,
    read_words: Callable[[str], Sequence[int]],
    thread_id: str,
    pc: int | None,
    exec_mask: int | None,
    architecture: str | None,
    kernel: str | None,
    stop_mode: str,
    float8_format: str = "ocp",
) -> ValueSnapshot:
    """Capture selected-wave expressions through an injected debugger reader."""
    if not names:
        raise ValueError("at least one logical value name is required")
    specs = _manifest_values(manifest, names)
    captures = []
    issues = []
    for spec in specs:
        name = spec["logical"]["name"]
        wave_size = spec["logical"]["layout"]["wave_size"]
        locations = []
        try:
            for expression in spec["binding"]["locations"]:
                words = tuple(int(word) & 0xFFFFFFFF for word in read_words(expression))
                if len(words) != wave_size:
                    raise ValueError(
                        f"{expression} returned {len(words)} lane words; "
                        f"expected {wave_size}"
                    )
                locations.append(CapturedLocation(expression, words))
        except Exception as error:
            message = str(error)
            status = (
                "optimized_out"
                if "optimized out" in message.lower()
                else "location_unavailable"
            )
            captures.append(CapturedValue(name=name, status=status, detail=message))
            issues.append({"value": name, "status": status, "detail": message})
        else:
            captures.append(
                CapturedValue(
                    name=name, status="available", locations=tuple(locations)
                )
            )
    wave_status = (
        "available"
        if all(value.status == "available" for value in captures)
        else "partial"
    )
    wave = WaveCapture(
        thread_id=thread_id,
        pc=None if pc is None else f"0x{pc:x}",
        exec=None if exec_mask is None else f"0x{exec_mask:x}",
        status=wave_status,
        values=tuple(captures),
    )
    return ValueSnapshot(
        capture={
            "scope": "wave",
            "stop_mode": stop_mode,
            "complete": wave_status == "available",
            "float8_format": float8_format,
            "issues": issues,
        },
        target={"architecture": architecture, "kernel": kernel},
        values=specs,
        waves=(wave,),
    )


def dumps_snapshot(snapshot: ValueSnapshot) -> str:
    return json.dumps(
        snapshot.to_dict(), allow_nan=False, indent=2, sort_keys=True
    ) + "\n"


def dump_snapshot(snapshot: ValueSnapshot, path: str | Path) -> None:
    """Write one snapshot atomically without overwriting an existing file."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(dumps_snapshot(snapshot))
        # Linking publishes the fully written inode atomically and fails if a
        # concurrent writer created the requested destination in the meantime.
        os.link(temporary_name, destination)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_snapshot(path: str | Path) -> ValueSnapshot:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load snapshot {str(path)!r}: {error}") from error
    return ValueSnapshot.from_dict(record)
