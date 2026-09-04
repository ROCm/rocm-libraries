#!/usr/bin/env python3
"""Validate complete API policy coverage and append-only enum evolution."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


VALID_CLASSES = {"protocol", "facade", "edge_local", "bespoke_cpp", "source_only"}


@dataclass(frozen=True)
class EnumShape:
    underlying_type: str
    values: dict[str, str]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def declarations(document: dict[str, object], path: Path) -> list[dict[str, object]]:
    value = document.get("declarations")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: declarations must be an array of objects")
    return value


def enum_shapes(document: dict[str, object], path: Path) -> dict[str, EnumShape]:
    result: dict[str, EnumShape] = {}
    for declaration in declarations(document, path):
        if declaration.get("kind") != "enum":
            continue
        name = declaration.get("name")
        underlying = declaration.get("underlying_type")
        raw_values = declaration.get("values")
        if (
            not isinstance(name, str)
            or not isinstance(underlying, str)
            or not isinstance(raw_values, list)
        ):
            raise ValueError(f"{path}: malformed enum declaration")
        values: dict[str, str] = {}
        for item in raw_values:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("value"), str)
            ):
                raise ValueError(f"{path}: malformed enumerator in {name}")
            values[item["name"]] = item["value"]
        result[name] = EnumShape(underlying, values)
    return result


def check_policy(snapshot_path: Path, policy_path: Path) -> None:
    snapshot = load_json(snapshot_path)
    policy = load_json(policy_path)
    raw_entries = policy.get("declarations")
    if not isinstance(raw_entries, dict):
        raise ValueError(f"{policy_path}: declarations must be an object")
    errors: list[str] = []
    for declaration in declarations(snapshot, snapshot_path):
        name = declaration.get("name")
        if not isinstance(name, str):
            errors.append("snapshot declaration has no string name")
            continue
        entry = raw_entries.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{name}: missing policy")
            continue
        classification = entry.get("classification")
        if classification not in VALID_CLASSES:
            errors.append(f"{name}: invalid classification {classification!r}")
        if declaration.get("kind") == "function" and classification in {
            "protocol",
            "facade",
        }:
            if not isinstance(entry.get("cluster"), str) or not entry["cluster"]:
                errors.append(f"{name}: callable policy requires a cluster")
    if errors:
        raise ValueError("API policy validation failed:\n" + "\n".join(errors))


def check_enum_evolution(baseline_path: Path, current_path: Path) -> None:
    baseline = enum_shapes(load_json(baseline_path), baseline_path)
    current = enum_shapes(load_json(current_path), current_path)
    errors: list[str] = []
    for name, old in baseline.items():
        new = current.get(name)
        if new is None:
            errors.append(f"{name}: public enum was removed")
            continue
        if old.underlying_type != new.underlying_type:
            errors.append(f"{name}: underlying type changed")
        for enumerator, value in old.values.items():
            if new.values.get(enumerator) != value:
                errors.append(f"{name}.{enumerator}: value changed or was removed")
    if errors:
        raise ValueError(
            "public enum compatibility check failed:\n" + "\n".join(errors)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--enum-baseline", type=Path)
    args = parser.parse_args()
    check_policy(args.snapshot, args.policy)
    if args.enum_baseline is not None:
        check_enum_evolution(args.enum_baseline, args.snapshot)


if __name__ == "__main__":
    main()
