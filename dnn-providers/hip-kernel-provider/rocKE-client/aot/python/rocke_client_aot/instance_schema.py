# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common parser for checked-in rocKE AOT instance descriptions."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSTANCE_SCHEMA = "rocke.aot.instance/v1"
_INSTANCE_FIELDS = frozenset(
    {
        "schema",
        "name",
        "op",
        "family",
        "arch",
        "compile_spec",
        "selection",
        "test_profiles",
    }
)
_HANDLER_FILENAME = "aot_instance.py"
_SUPPORTED_ATTRIBUTE_CONSTRAINTS = frozenset({"equals", "not_equals", "one_of"})


class InstanceError(ValueError):
    """Raised when a checked-in AOT instance is invalid."""


@dataclass(frozen=True)
class KernelInstanceActions:
    """Operation-specific actions supplied by one kernel directory."""

    build_kernel: Callable[..., Any]
    emit_sidecar: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ParsedInstance:
    """Normalized checked-in instance plus operation-specific build actions."""

    path: Path
    data: Mapping[str, Any]
    compile_spec: Mapping[str, Any]
    selection: Mapping[str, Any]
    test_profiles: Sequence[Mapping[str, Any]]
    spec: Any
    validation_reason: str
    actions: KernelInstanceActions


def parse_instance(
    path: str | Path, *, kernel_dir: str | Path | None = None
) -> ParsedInstance:
    """Load one ``.instance.json`` and delegate operation-specific fields."""

    instance_path = Path(path)
    data = _load_instance(instance_path)
    instance = _validate_instance(data)
    handler = _load_handler(_resolve_kernel_dir(instance_path, kernel_dir))
    _validate_handler_id(handler, instance)

    parse_instance_fields = getattr(handler, "parse_instance_fields", None)
    if not callable(parse_instance_fields):
        raise InstanceError(
            f"{handler.__file__} must define callable parse_instance_fields"
        )
    normalized_fields, spec, reason = parse_instance_fields(instance, instance_path)
    normalized_fields = dict(require_mapping(normalized_fields, "normalized fields"))
    normalized = dict(instance)
    normalized.update(_validate_normalized_fields(normalized_fields))

    build_kernel = getattr(handler, "build_kernel", None)
    emit_sidecar = getattr(handler, "emit_sidecar", None)
    if not callable(build_kernel):
        raise InstanceError(f"{handler.__file__} must define callable build_kernel")
    if not callable(emit_sidecar):
        raise InstanceError(f"{handler.__file__} must define callable emit_sidecar")

    return ParsedInstance(
        path=instance_path,
        data=normalized,
        compile_spec=normalized["compile_spec"],
        selection=normalized["selection"],
        test_profiles=normalized["test_profiles"],
        spec=spec,
        validation_reason=require_string(reason, "validation reason"),
        actions=KernelInstanceActions(
            build_kernel=build_kernel,
            emit_sidecar=emit_sidecar,
        ),
    )


def attributes_match_constraints(
    attributes: Mapping[str, Any], constraints: Mapping[str, Any]
) -> bool:
    """Return whether runtime attributes satisfy normalized attribute constraints."""

    normalized = normalize_attribute_constraints(constraints)
    for name, rule in normalized.items():
        if name not in attributes:
            return False
        value = attributes[name]
        if "equals" in rule and value != rule["equals"]:
            return False
        if "not_equals" in rule and value == rule["not_equals"]:
            return False
        if "one_of" in rule and value not in rule["one_of"]:
            return False
    return True


def normalize_attribute_constraints(constraints: Any) -> dict[str, dict[str, Any]]:
    """Validate and copy generic selection attribute constraints."""

    data = require_mapping(constraints, "selection.attribute_constraints")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_rule in data.items():
        name = require_string(raw_name, "selection.attribute_constraints key")
        rule = dict(
            require_mapping(raw_rule, f"selection.attribute_constraints.{name}")
        )
        if not rule:
            raise InstanceError(
                f"selection.attribute_constraints.{name} must not be empty"
            )
        unsupported = sorted(set(rule) - _SUPPORTED_ATTRIBUTE_CONSTRAINTS)
        if unsupported:
            raise InstanceError(
                f"selection.attribute_constraints.{name} has unsupported operators: "
                + ", ".join(unsupported)
            )
        if "one_of" in rule:
            options = rule["one_of"]
            if not isinstance(options, list) or not options:
                raise InstanceError(
                    f"selection.attribute_constraints.{name}.one_of must be a non-empty array"
                )
            rule["one_of"] = list(options)
        normalized[name] = rule
    return normalized


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return a mapping value or raise an instance validation error."""

    if not isinstance(value, Mapping):
        raise InstanceError(f"{context} must be an object")
    return value


def require_string(value: Any, context: str) -> str:
    """Return a non-empty string value or raise an instance validation error."""

    if not isinstance(value, str) or not value:
        raise InstanceError(f"{context} must be a non-empty string")
    return value


def require_int(value: Any, context: str) -> int:
    """Return an integer value or raise an instance validation error."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InstanceError(f"{context} must be an integer")
    return value


def _load_instance(path: Path) -> dict[str, Any]:
    """Load and parse a checked-in instance JSON file."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise InstanceError(f"failed to read instance {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InstanceError(f"failed to parse instance {path}: {exc}") from exc
    return dict(require_mapping(value, "instance file"))


def _validate_instance(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate top-level instance fields and return normalized common data."""

    extras = sorted(set(data) - _INSTANCE_FIELDS)
    if extras:
        raise InstanceError(
            "instance contains unsupported top-level fields: " + ", ".join(extras)
        )
    schema = data.get("schema")
    if schema != INSTANCE_SCHEMA:
        raise InstanceError(
            f"instance schema must be {INSTANCE_SCHEMA!r}, got {schema!r}"
        )
    test_profiles = data.get("test_profiles", [])
    if not isinstance(test_profiles, list):
        raise InstanceError("test_profiles must be an array")
    return {
        "schema": schema,
        "name": require_string(data.get("name"), "instance name"),
        "op": require_string(data.get("op"), "instance op"),
        "family": require_string(data.get("family"), "instance family"),
        "arch": require_string(data.get("arch"), "instance arch"),
        "compile_spec": require_mapping(data.get("compile_spec"), "compile_spec"),
        "selection": _normalize_selection(data.get("selection", {})),
        "test_profiles": list(test_profiles),
    }


def _validate_normalized_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Validate handler-normalized fields and return normalized common data."""

    extras = sorted(set(fields) - {"compile_spec", "selection", "test_profiles"})
    if extras:
        raise InstanceError(
            "normalized fields contain unsupported entries: " + ", ".join(extras)
        )
    test_profiles = fields.get("test_profiles", [])
    if not isinstance(test_profiles, list):
        raise InstanceError("normalized test_profiles must be an array")
    return {
        "compile_spec": require_mapping(fields.get("compile_spec"), "compile_spec"),
        "selection": _normalize_selection(fields.get("selection", {})),
        "test_profiles": list(test_profiles),
    }


def _normalize_selection(selection: Any) -> dict[str, Any]:
    """Normalize selection fields that are shared by all instance kinds."""

    data = dict(require_mapping(selection, "selection"))
    constraints = data.get("attribute_constraints", {})
    data["attribute_constraints"] = normalize_attribute_constraints(constraints)
    return data


def _resolve_kernel_dir(instance_path: Path, kernel_dir: str | Path | None) -> Path:
    """Resolve the operation kernel directory for an instance file."""

    if kernel_dir is not None:
        return Path(kernel_dir)
    parent = instance_path.parent
    if parent.parent.name == "instances":
        return parent.parent.parent
    raise InstanceError(
        f"kernel_dir is required for copied instance outside a kernel instances tree: {instance_path}"
    )


def _load_handler(kernel_dir: Path) -> Any:
    """Load the operation-specific AOT instance handler module."""

    handler_path = kernel_dir / _HANDLER_FILENAME
    if not handler_path.is_file():
        raise InstanceError(
            f"kernel directory {kernel_dir} is missing {_HANDLER_FILENAME}"
        )
    module_name = f"_rocke_client_aot_{abs(hash(handler_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    if spec is None or spec.loader is None:
        raise InstanceError(f"failed to load kernel handler {handler_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_handler_id(handler: Any, instance: Mapping[str, Any]) -> None:
    """Ensure the handler matches the instance operation and family."""

    op = getattr(handler, "OP", None)
    family = getattr(handler, "FAMILY", None)
    if op != instance["op"] or family != instance["family"]:
        raise InstanceError(
            f"kernel handler {handler.__file__} supports {op!r}/{family!r}, "
            f"not {instance['op']!r}/{instance['family']!r}"
        )


__all__ = [
    "INSTANCE_SCHEMA",
    "InstanceError",
    "KernelInstanceActions",
    "ParsedInstance",
    "attributes_match_constraints",
    "normalize_attribute_constraints",
    "parse_instance",
    "require_int",
    "require_mapping",
    "require_string",
]
