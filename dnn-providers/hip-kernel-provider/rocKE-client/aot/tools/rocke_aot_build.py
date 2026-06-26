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
from rocke_client_aot.instance_schema import parse_instance
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
        help="Directory containing checked-in or copied .instance.json files.",
    )
    parser.add_argument(
        "--kernel-dir",
        required=True,
        type=Path,
        help="Source kernel directory containing the operation-specific aot_instance.py.",
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
    """Extract the parsed instance document from a parse_instance result."""

    if isinstance(parsed, Mapping):
        return parsed
    data = getattr(parsed, "data", None)
    if data is None:
        data = getattr(parsed, "instance", None)
    return _as_mapping(data, "parsed instance data")


def _parsed_spec(parsed: Any) -> Any:
    """Extract the rocKE kernel spec from a parse_instance result."""

    spec = getattr(parsed, "spec", None)
    if spec is None:
        spec = getattr(parsed, "fmha_spec", None)
    if spec is None:
        raise TypeError("parse_instance result must expose spec or fmha_spec")
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


def _schema_path(kernel_dir: Path, name: str) -> Path:
    """Resolve a kernel-local schema before falling back to shared AOT schemas."""

    kernel_schema = kernel_dir / "schemas" / f"{name}.schema.json"
    if kernel_schema.is_file():
        return kernel_schema
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
    instance_path: Path,
    kernel_dir: Path,
    instance_schema_path: Path,
    sidecar_schema_path: Path,
) -> tuple[Path, Path]:
    """Build the HSACO and sidecar artifacts for one instance file."""

    _validate_json_file(instance_path, instance_schema_path)
    parsed = parse_instance(instance_path, kernel_dir=kernel_dir)
    instance_data = _parsed_instance_data(parsed)
    spec = _parsed_spec(parsed)

    name = instance_data.get("name")
    arch = instance_data.get("arch")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{instance_path}: instance name must be a non-empty string")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{instance_path}: instance arch must be a non-empty string")

    kernel = parsed.actions.build_kernel(spec, arch=arch)
    artifact = compile_kernel(
        kernel,
        arch=arch,
        backend="python",
        capture_ir_text=False,
    )

    hsaco_path = instance_path.with_name(f"{name}.hsaco")
    sidecar_path = instance_path.with_name(f"{name}.sidecar.json")
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
        kernel_dir = args.kernel_dir
        if not artifact_dir.is_dir():
            raise ValueError(f"artifact directory does not exist: {artifact_dir}")
        if not kernel_dir.is_dir():
            raise ValueError(f"kernel directory does not exist: {kernel_dir}")

        _clean_stale_outputs(artifact_dir)
        instance_paths = sorted(artifact_dir.glob("*.instance.json"))
        if not instance_paths:
            raise ValueError(f"no .instance.json files found in {artifact_dir}")

        instance_schema_path = _schema_path(kernel_dir, "instance")
        sidecar_schema_path = _schema_path(kernel_dir, "sidecar")
        for schema_path in (instance_schema_path, sidecar_schema_path):
            if not schema_path.is_file():
                raise ValueError(f"schema file does not exist: {schema_path}")

        for instance_path in instance_paths:
            _build_one(
                instance_path, kernel_dir, instance_schema_path, sidecar_schema_path
            )
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    except Exception as exc:
        print(f"rocke_aot_build: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
