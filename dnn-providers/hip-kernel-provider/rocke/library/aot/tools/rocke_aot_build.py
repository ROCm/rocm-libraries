#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Build loose rocKE client AOT artifacts from checked-in instances."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from rocke.helpers import compile_kernel
from rocke_client_aot.instance_schema import AOT_LIST_FILENAME, parse_instance_list
from rocke_client_aot.json_schema import load_json_schema, validate_json_schema

_STALE_OUTPUT_PATTERNS = ("*.hsaco", "*.sidecar.json")
_HIPCC_ENV_KEYS = frozenset(
    {
        "ROCKE_AOT_BACKEND",
        "ROCKE_AOT_COMPILE_BACKEND",
        "ROCKE_COMPILE_BACKEND",
        "ROCKE_USE_HIPCC",
    }
)
_TRUTHY = frozenset({"1", "true", "yes", "on", "hipcc"})


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser for rocKE client AOT artifact builds."""

    parser = argparse.ArgumentParser(
        description="Build rocKE client AOT HSACO and sidecar artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Directory containing the copied aot_list.json instance array.",
    )
    parser.add_argument(
        "--handler",
        required=True,
        type=Path,
        help="Path to the family AOT handler module (kernels/common/<family>_aot.py).",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=None,
        help="Optional family schema overlay dir; falls back to the shared AOT schemas.",
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Architecture every instance in the aot_list.json must target.",
    )
    return parser


def _reject_hipcc_env() -> None:
    """Reject environment settings that would route compilation through hipcc."""

    for key, value in os.environ.items():
        lowered = value.strip().lower()
        if (key in _HIPCC_ENV_KEYS and lowered in _TRUTHY) or (
            key.startswith("ROCKE_")
            and (lowered == "hipcc" or ("HIPCC" in key and lowered in _TRUTHY))
        ):
            raise ValueError(
                f"{key}={value!r} would request a hipcc compile path; "
                "rocKE client AOT always uses compile_kernel(..., backend='python')"
            )


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return a value as a mapping or fail with contextual type information."""

    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{context} must be a mapping")


def _parsed_instance_data(parsed: Any) -> Mapping[str, Any]:
    """Extract the parsed instance document from a parsed instance result."""

    if isinstance(parsed, Mapping):
        return parsed
    data = getattr(parsed, "data", None)
    if data is None:
        data = getattr(parsed, "instance", None)
    return _as_mapping(data, "parsed instance data")


def _parsed_spec(parsed: Any) -> Any:
    """Extract the rocKE kernel spec from a parsed instance result."""

    spec = getattr(parsed, "spec", None)
    if spec is None:
        spec = getattr(parsed, "fmha_spec", None)
    if spec is None:
        raise TypeError("parsed instance result must expose spec or fmha_spec")
    return spec


def _clean_stale_outputs(artifact_dir: Path) -> None:
    """Remove generated AOT outputs before rebuilding an artifact directory."""

    for pattern in _STALE_OUTPUT_PATTERNS:
        for path in sorted(artifact_dir.glob(pattern)):
            if path.is_file() or path.is_symlink():
                path.unlink()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write a JSON document using the repository's stable formatting."""

    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _schema_path(schema_dir: Path | None, name: str) -> Path:
    """Resolve a family schema overlay before falling back to shared AOT schemas."""

    if schema_dir is not None:
        family_schema = schema_dir / f"{name}.schema.json"
        if family_schema.is_file():
            return family_schema
    return Path(__file__).resolve().parents[1] / "schemas" / f"{name}.schema.json"


def _load_json(path: Path) -> Any:
    """Load a JSON document from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_json_file(path: Path, schema_path: Path) -> None:
    """Validate a JSON file against the selected schema."""

    validate_json_schema(
        _load_json(path), load_json_schema(schema_path), schema_path=schema_path
    )


def _validate_json_value(value: Mapping[str, Any], schema_path: Path) -> None:
    """Validate an in-memory JSON document against the selected schema."""

    validate_json_schema(value, load_json_schema(schema_path), schema_path=schema_path)


def _temporary_artifact_path(final_path: Path) -> Path:
    """Create a closed temporary artifact path beside its final destination."""

    handle, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent
    )
    os.close(handle)
    return Path(temp_name)


def _build_one(
    parsed: Any,
    artifact_dir: Path,
    sidecar_schema_path: Path,
) -> tuple[Path, Path]:
    """Build the HSACO and sidecar artifacts for one parsed instance."""

    instance_data = _parsed_instance_data(parsed)
    spec = _parsed_spec(parsed)

    name = instance_data.get("name")
    arch = instance_data.get("arch")
    if not isinstance(name, str) or not name:
        raise ValueError("instance name must be a non-empty string")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{name}: instance arch must be a non-empty string")

    kernel = parsed.actions.build_kernel(spec, arch=arch)
    artifact = compile_kernel(
        kernel,
        arch=arch,
        backend="python",
        capture_ir_text=False,
    )

    hsaco_path = artifact_dir / f"{name}.hsaco"
    sidecar_path = artifact_dir / f"{name}.sidecar.json"
    hsaco_temp_path: Path | None = None
    sidecar_temp_path: Path | None = None
    try:
        hsaco_temp_path = _temporary_artifact_path(hsaco_path)
        sidecar_temp_path = _temporary_artifact_path(sidecar_path)

        hsaco_temp_path.write_bytes(artifact.hsaco)
        sidecar = parsed.actions.emit_sidecar(parsed, spec, artifact, hsaco_path.name)
        _validate_json_value(sidecar, sidecar_schema_path)
        _write_json(sidecar_temp_path, sidecar)
        _validate_json_file(sidecar_temp_path, sidecar_schema_path)

        os.replace(hsaco_temp_path, hsaco_path)
        hsaco_temp_path = None
        os.replace(sidecar_temp_path, sidecar_path)
        sidecar_temp_path = None
    finally:
        for temp_path in (hsaco_temp_path, sidecar_temp_path):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return hsaco_path, sidecar_path


def main(argv: Sequence[str] | None = None) -> int:
    """Build all AOT artifacts described by the command-line arguments."""

    try:
        args = _parser().parse_args(argv)
        _reject_hipcc_env()
        artifact_dir = args.artifact_dir
        handler_path = args.handler
        schema_dir = args.schema_dir
        arch = args.arch
        if not artifact_dir.is_dir():
            raise ValueError(f"artifact directory does not exist: {artifact_dir}")
        if not handler_path.is_file():
            raise ValueError(f"handler module does not exist: {handler_path}")
        if schema_dir is not None and not schema_dir.is_dir():
            raise ValueError(f"schema directory does not exist: {schema_dir}")

        _clean_stale_outputs(artifact_dir)
        aot_list_path = artifact_dir / AOT_LIST_FILENAME
        if not aot_list_path.is_file():
            raise ValueError(f"no {AOT_LIST_FILENAME} found in {artifact_dir}")

        instance_schema_path = _schema_path(schema_dir, "instance")
        sidecar_schema_path = _schema_path(schema_dir, "sidecar")
        for schema_path in (instance_schema_path, sidecar_schema_path):
            if not schema_path.is_file():
                raise ValueError(f"schema file does not exist: {schema_path}")

        raw_instances = _load_json(aot_list_path)
        if not isinstance(raw_instances, list) or not raw_instances:
            raise ValueError(f"{aot_list_path} must be a non-empty JSON array")
        for raw_instance in raw_instances:
            _validate_json_value(raw_instance, instance_schema_path)

        for parsed in parse_instance_list(
            aot_list_path, handler_path=handler_path, expected_arch=arch
        ):
            _build_one(parsed, artifact_dir, sidecar_schema_path)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    except Exception as exc:
        print(f"rocke_aot_build: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
