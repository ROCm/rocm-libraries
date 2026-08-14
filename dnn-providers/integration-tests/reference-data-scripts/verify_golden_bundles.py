#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify hipDNN golden bundle directories."""

import argparse
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ALLOWED_TIERS = {"quick", "standard", "comprehensive", "full"}
BUNDLE_SIZE_WARNING_BYTES = 1024 * 1024
BUNDLE_SIZE_ERROR_BYTES = 2 * 1024 * 1024

CASE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
PLACEHOLDER_PATTERN = re.compile(r"^\$\{case\.([A-Za-z0-9_.]+)\}$")
TEMPLATE_TENSOR_FIELDS = ("dims", "strides", "data_type")

# Node types whose real hipDNN JSON schema declares output tensor uids as flat
# scalar fields directly on the node, instead of nesting them under an `outputs`
# object like every other attribute type. See
# flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/utilities/json/ReductionAttributes.hpp
# to_json/from_json, backed by reduction_attributes.fbs (`in_tensor_uid`/
# `out_tensor_uid` are plain scalar fields on ReductionAttributes, not a map).
FLAT_OUTPUT_UID_FIELDS: dict[str, tuple[str, ...]] = {
    "ReductionAttributes": ("out_tensor_uid",),
}


DTYPE_BYTE_SIZE = {
    "float": 4,
    "float32": 4,
    "fp32": 4,
    "double": 8,
    "float64": 8,
    "fp64": 8,
    "half": 2,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "bfp16": 2,
    "uint8": 1,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "boolean": 1,
    "bool": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_e8m0": 1,
    "fp8_e4m3_fnuz": 1,
    "fp8_e5m2_fnuz": 1,
}

FLOAT_DTYPES = {
    "float",
    "float32",
    "fp32",
    "double",
    "float64",
    "fp64",
    "half",
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
    "bfp16",
}


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    path: Path
    message: str
    tensor_uid: int | None = None


@dataclass(frozen=True)
class Advisory:
    graph_path: Path
    canonical_path: str
    test_suite: str
    test_case: str
    full_test_name: str


@dataclass
class VerificationResult:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    advisories: list[Advisory] = field(default_factory=list)

    def error(self, path: Path, message: str, tensor_uid: int | None = None) -> None:
        self.diagnostics.append(Diagnostic("error", path, message, tensor_uid))

    def warning(self, path: Path, message: str, tensor_uid: int | None = None) -> None:
        self.diagnostics.append(Diagnostic("warning", path, message, tensor_uid))

    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)

    def print_diagnostics(self) -> None:
        for diagnostic in self.diagnostics:
            label = diagnostic.severity.upper()
            prefix = f"{label}: {diagnostic.path}: "
            if diagnostic.tensor_uid is not None:
                prefix += f"tensor uid {diagnostic.tensor_uid}: "
            print(prefix + diagnostic.message, file=sys.stderr)

        error_count = sum(
            1 for diagnostic in self.diagnostics if diagnostic.severity == "error"
        )
        warning_count = len(self.diagnostics) - error_count
        print(
            f"SUMMARY: {error_count} error(s), {warning_count} warning(s), "
            f"{len(self.advisories)} advisory/advisories",
            file=sys.stderr,
        )

    def print_advisories(self) -> None:
        for advisory in self.advisories:
            print(f"ADVISORY: {advisory.graph_path}")
            print(f"  canonical_path: {advisory.canonical_path}")
            print(f"  test_suite: {advisory.test_suite}")
            print(f"  test_case: {advisory.test_case}")
            print(f"  full_test_name: {advisory.full_test_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify hipDNN golden bundle directories"
    )
    parser.add_argument(
        "--default-tier",
        default="quick",
        choices=sorted(ALLOWED_TIERS),
        help="Fallback tier for advisory output when no tier directory is present",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help=(
            "Treat missing tensor payloads as errors instead of warnings. "
            "Enable in CI after `dvc pull`; leave off for local structure-only "
            "checks before pulling golden tensor data."
        ),
    )
    parser.add_argument(
        "roots",
        metavar="ROOT",
        nargs="+",
        type=Path,
        help="Bundle root directories to verify",
    )
    return parser.parse_args()


def is_metadata_sidecar(path: Path) -> bool:
    return path.name == "meta.json" or path.name.endswith(".meta.json")


def is_graph_candidate(path: Path) -> bool:
    return (
        path.suffix == ".json"
        and not is_metadata_sidecar(path)
        and path.stem == path.parent.name
    )


def is_template_file(path: Path) -> bool:
    return path.name == "graph.template.json"


def is_sweep_file(path: Path) -> bool:
    return path.name == "sweep.json"


def bundle_has_tensor_manifest(path: Path) -> bool:
    return path.with_suffix(".tensors.dvc").is_file()


def iter_json_files(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix != ".json":
            return []
        return [root]

    return sorted(path for path in root.rglob("*.json") if path.is_file())


def find_sweep_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    dirs = {path.parent for path in root.rglob("sweep.json") if path.is_file()}
    dirs.update(
        path.parent for path in root.rglob("graph.template.json") if path.is_file()
    )
    return sorted(dirs)


def warn_unexpected_top_level_directories(
    root: Path, result: VerificationResult
) -> None:
    if not root.is_dir() or root.name in ALLOWED_TIERS:
        return

    child_directories = sorted(path for path in root.iterdir() if path.is_dir())
    has_tier_children = any(path.name in ALLOWED_TIERS for path in child_directories)
    if root.name != "integration-test-bundles" and not has_tier_children:
        return

    for path in child_directories:
        if path.name not in ALLOWED_TIERS:
            result.warning(path, "unexpected top-level directory")


def validate_metadata_fields(
    metadata: dict, path: Path, result: VerificationResult, prefix: str = ""
) -> None:
    for key in ("generator", "reference_source"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            result.error(
                path, f"{prefix}{key} is required and must be a non-empty string"
            )


def validate_metadata(path: Path, result: VerificationResult) -> None:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(path, f"metadata JSON is not parseable: {error}")
        return

    if not isinstance(metadata, dict):
        result.error(path, "metadata JSON is not an object")
        return

    validate_metadata_fields(metadata, path, result)


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_integer_list(value: object) -> bool:
    return isinstance(value, list) and all(is_integer(item) for item in value)


def element_space(dims: list[int], strides: list[int]) -> int:
    return 1 + sum((dim - 1) * stride for dim, stride in zip(dims, strides))


# TODO(ALMIOPEN-1971 follow-up): RFC 0011 4.3 also calls for an aggregate DVC
# payload budget check (fail when the whole tree's committed golden tensor
# payload exceeds 800 MB) in addition to the per-bundle checks below. Track
# and implement as an opt-in flag (CI-only by default; a full local run
# shouldn't force scanning every pulled tensor) in a follow-up story.
def validate_directory_size(
    scan_root: Path, report_path: Path, result: VerificationResult
) -> None:
    total_bytes = 0
    try:
        for child in scan_root.rglob("*"):
            if child.is_file():
                total_bytes += child.stat().st_size
    except OSError as error:
        result.error(report_path, f"could not stat bundle contents: {error}")
        return

    if total_bytes > BUNDLE_SIZE_ERROR_BYTES:
        result.error(
            report_path,
            "bundle totals "
            f"{total_bytes} bytes (> {BUNDLE_SIZE_ERROR_BYTES} bytes); "
            "cannot have bundles larger than 2 MiB because they would quickly "
            "explode our test artifact sizes",
        )
        return

    if total_bytes > BUNDLE_SIZE_WARNING_BYTES:
        result.warning(
            report_path,
            "bundle totals "
            f"{total_bytes} bytes (> {BUNDLE_SIZE_WARNING_BYTES} bytes); "
            "keep bundles at or below 1 MiB when possible",
        )


def extract_output_tensor_uids(
    nodes: list[object], path: Path, result: VerificationResult
) -> set[int]:
    output_tensor_uids: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            result.error(path, f"node {index} is not an object")
            continue

        flat_fields = FLAT_OUTPUT_UID_FIELDS.get(node.get("type"))
        if flat_fields is not None:
            for field_name in flat_fields:
                value = node.get(field_name)
                if value is None:
                    continue
                if not is_integer(value):
                    result.error(path, f"'{field_name}' must be an integer tensor uid")
                    continue
                output_tensor_uids.add(value)
            continue

        outputs = node.get("outputs")
        if not isinstance(outputs, dict):
            result.error(
                path, f"node {index} outputs is required and must be an object"
            )
            continue

        for name, value in outputs.items():
            if "_tensor_uid" not in name or value is None:
                continue
            if not is_integer(value):
                result.error(path, f"output '{name}' must be an integer tensor uid")
                continue
            output_tensor_uids.add(value)

    return output_tensor_uids


def find_nonfinite_index(dtype_key: str, data: bytes) -> int | None:
    if dtype_key in {"float", "float32", "fp32"}:
        for index, (value,) in enumerate(struct.iter_unpack("<f", data)):
            if not math.isfinite(value):
                return index
        return None

    if dtype_key in {"double", "float64", "fp64"}:
        for index, (value,) in enumerate(struct.iter_unpack("<d", data)):
            if not math.isfinite(value):
                return index
        return None

    if dtype_key in {"half", "float16", "fp16"}:
        for index, (word,) in enumerate(struct.iter_unpack("<H", data)):
            if ((word >> 10) & 0x1F) == 0x1F:
                return index
        return None

    if dtype_key in {"bfloat16", "bf16", "bfp16"}:
        for index, (word,) in enumerate(struct.iter_unpack("<H", data)):
            if ((word >> 7) & 0xFF) == 0xFF:
                return index
        return None

    return None


def sanitize_gtest_name(value: str) -> str:
    sanitized: list[str] = []
    for character in value:
        is_ascii_alnum = (
            "A" <= character <= "Z"
            or "a" <= character <= "z"
            or "0" <= character <= "9"
        )
        sanitized.append(character if is_ascii_alnum or character == "_" else "_")
    return "".join(sanitized)


def validate_tensor_payloads(
    tensor_specs: dict[int, dict[str, object]],
    tensor_path_for: Callable[[int], Path],
    bundle_has_manifest: bool,
    require_data: bool,
    output_tensor_uids: set[int],
    report_path: Path,
    result: VerificationResult,
) -> None:
    for uid, tensor_spec in tensor_specs.items():
        dims = tensor_spec["dims"]
        strides = tensor_spec["strides"]
        data_type = tensor_spec["data_type"]
        dtype_key = data_type.lower()
        element_size = DTYPE_BYTE_SIZE.get(dtype_key)
        tensor_path = tensor_path_for(uid)

        if element_size is None:
            result.error(
                report_path,
                f"unsupported data_type '{data_type}' for byte-size validation",
                uid,
            )
            continue

        if not tensor_path.exists():
            if bundle_has_manifest:
                if require_data:
                    result.error(
                        tensor_path,
                        f"missing tensor file; expected {tensor_path}",
                        uid,
                    )
                else:
                    result.warning(
                        tensor_path,
                        "tensor data not pulled locally; expected "
                        f"{tensor_path} (run `dvc pull` or pass --require-data "
                        "to enforce in CI)",
                        uid,
                    )
            continue

        try:
            actual_size = tensor_path.stat().st_size
        except OSError as error:
            result.error(tensor_path, f"could not stat tensor file: {error}", uid)
            continue

        expected_size = element_space(dims, strides) * element_size
        if actual_size != expected_size:
            result.error(
                tensor_path,
                "file has "
                f"{actual_size} bytes but graph expects {expected_size} bytes "
                f"(element_space={element_space(dims, strides)}, element_size={element_size})",
                uid,
            )
            continue

        if dtype_key not in FLOAT_DTYPES:
            continue

        try:
            data = tensor_path.read_bytes()
        except OSError as error:
            result.error(tensor_path, f"could not read tensor file: {error}", uid)
            continue

        bad_index = find_nonfinite_index(dtype_key, data)
        if bad_index is not None:
            tensor_role = "output" if uid in output_tensor_uids else "input"
            result.error(
                tensor_path,
                f"{tensor_role} tensor contains NaN/Inf at element index {bad_index}",
                uid,
            )


def derive_advisory(
    path: Path, result: VerificationResult, default_tier: str
) -> Advisory | None:
    if path.stem != path.parent.name:
        result.error(path, "graph files must be named <BundleName>/<BundleName>.json")
        return None

    parts = path.parts
    tier_index = next(
        (index for index, part in enumerate(parts) if part in ALLOWED_TIERS), None
    )

    if tier_index is None:
        result.warning(
            path,
            f"no tier directory found; using default tier '{default_tier}' for advisory output",
        )
        if len(parts) < 5:
            result.error(
                path,
                "cannot derive advisory path; expected {Tier}/{Operation}/{Layout}/{DataType}/{Name}/{Name}.json",
            )
            return None
        tier = default_tier
        operation, layout, data_type, name, file_name = parts[-5:]
    else:
        trailing_parts = parts[tier_index:]
        if len(trailing_parts) < 6:
            result.error(
                path,
                "cannot derive advisory path; expected {Tier}/{Operation}/{Layout}/{DataType}/{Name}/{Name}.json",
            )
            return None
        tier, operation, layout, data_type, name, file_name = trailing_parts[:6]

    if file_name != f"{name}.json":
        result.error(path, "graph files must be named <BundleName>/<BundleName>.json")
        return None

    canonical_path = f"{tier}/{operation}/{layout}/{data_type}/{name}/"
    test_suite = sanitize_gtest_name(
        "_".join((tier, operation, layout, data_type, name))
    )
    test_case = sanitize_gtest_name(name)
    return Advisory(
        path, canonical_path, test_suite, test_case, f"{test_suite}.{test_case}"
    )


def derive_sweep_advisory(
    sweep_path: Path, case_id: str, result: VerificationResult, default_tier: str
) -> Advisory | None:
    parts = sweep_path.parent.parts
    tier_index = next(
        (index for index, part in enumerate(parts) if part in ALLOWED_TIERS), None
    )

    if tier_index is None:
        result.warning(
            sweep_path,
            f"no tier directory found; using default tier '{default_tier}' for advisory output",
        )
        if len(parts) < 2:
            result.error(
                sweep_path,
                "cannot derive advisory path; expected {Tier}/{Operation}/{TopologyName}/sweep.json",
            )
            return None
        tier = default_tier
        operation, topology = parts[-2:]
    else:
        trailing_parts = parts[tier_index:]
        if len(trailing_parts) < 3:
            result.error(
                sweep_path,
                "cannot derive advisory path; expected {Tier}/{Operation}/{TopologyName}/sweep.json",
            )
            return None
        tier, operation, topology = trailing_parts[:3]

    canonical_path = f"{tier}/{operation}/{topology}/sweep.json"
    test_suite = sanitize_gtest_name("_".join((tier, operation, topology)))
    test_case = sanitize_gtest_name(case_id)
    return Advisory(
        sweep_path, canonical_path, test_suite, test_case, f"{test_suite}.{test_case}"
    )


def validate_graph_bundle(
    path: Path, result: VerificationResult, default_tier: str, require_data: bool
) -> None:
    advisory = derive_advisory(path, result, default_tier)
    if advisory is not None:
        result.advisories.append(advisory)
    validate_directory_size(path.parent, path, result)

    try:
        graph = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(path, f"graph JSON is not parseable: {error}")
        return

    if not isinstance(graph, dict):
        result.error(path, "graph JSON is not an object")
        return

    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        result.error(path, "top-level nodes must be a non-empty list")
        return

    tensors = graph.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        result.error(path, "top-level tensors must be a non-empty list")
        return

    output_tensor_uids = extract_output_tensor_uids(nodes, path, result)
    tensor_specs: dict[int, dict[str, object]] = {}

    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            result.error(path, f"tensor entry {index} is not an object")
            continue

        uid_value = tensor.get("uid")
        uid = uid_value if is_integer(uid_value) else None
        if uid is None:
            result.error(path, "uid is required and must be an integer")

        dims_value = tensor.get("dims")
        dims = list(dims_value) if is_integer_list(dims_value) else None
        if dims is None:
            result.error(path, "dims is required and must be a list of integers", uid)

        strides_value = tensor.get("strides")
        strides = list(strides_value) if is_integer_list(strides_value) else None
        if strides is None:
            result.error(
                path, "strides is required and must be a list of integers", uid
            )

        if dims is not None and strides is not None and len(dims) != len(strides):
            result.error(path, "dims and strides must have the same length", uid)

        if dims is not None and any(dim <= 0 for dim in dims):
            result.error(path, "dims must contain only positive integers", uid)

        if strides is not None and any(stride < 0 for stride in strides):
            result.error(path, "strides must contain only non-negative integers", uid)

        data_type = tensor.get("data_type")
        if not isinstance(data_type, str):
            result.error(path, "data_type is required and must be a string", uid)

        if uid is None:
            continue

        if uid in tensor_specs:
            result.error(path, "duplicate tensor uid declared", uid)
            continue

        if dims is None or strides is None or not isinstance(data_type, str):
            continue

        tensor_specs[uid] = {
            "dims": dims,
            "strides": strides,
            "data_type": data_type,
        }

    bundle_has_manifest = bundle_has_tensor_manifest(path)
    base_path = path.with_suffix("")
    validate_tensor_payloads(
        tensor_specs,
        lambda uid: Path(f"{base_path}.tensor{uid}.bin"),
        bundle_has_manifest,
        require_data,
        output_tensor_uids,
        path,
        result,
    )

    for output_tensor_uid in output_tensor_uids:
        if output_tensor_uid not in tensor_specs:
            result.error(
                path, "output tensor uid is not declared in tensors", output_tensor_uid
            )


def extract_placeholder(value: object) -> str | None:
    if isinstance(value, str):
        match = PLACEHOLDER_PATTERN.match(value)
        if match:
            return match.group(1)
    return None


def collect_placeholders(
    value: object, scalar_fields: set[str], attribute_fields: set[str]
) -> None:
    if isinstance(value, dict):
        for item in value.values():
            collect_placeholders(item, scalar_fields, attribute_fields)
        return
    if isinstance(value, list):
        for item in value:
            collect_placeholders(item, scalar_fields, attribute_fields)
        return

    placeholder = extract_placeholder(value)
    if placeholder is None:
        return
    if placeholder.startswith("attributes."):
        attribute_fields.add(placeholder[len("attributes.") :])
    elif placeholder != "tensors" and not placeholder.startswith("tensors."):
        scalar_fields.add(placeholder)


def parse_template_tensors(
    tensors: list[object], template_path: Path, result: VerificationResult
) -> tuple[dict[int, set[str]], dict[int, dict[str, object]]]:
    tensor_placeholder_fields: dict[int, set[str]] = {}
    literal_tensor_specs: dict[int, dict[str, object]] = {}
    seen_uids: set[int] = set()

    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            result.error(template_path, f"tensor entry {index} is not an object")
            continue

        uid_value = tensor.get("uid")
        uid = uid_value if is_integer(uid_value) else None
        if uid is None:
            result.error(template_path, "uid is required and must be an integer")
            continue

        if uid in seen_uids:
            result.error(template_path, "duplicate tensor uid declared", uid)
            continue
        seen_uids.add(uid)

        placeholder_fields: set[str] = set()
        literal_spec: dict[str, object] = {}
        for field_name in TEMPLATE_TENSOR_FIELDS:
            field_value = tensor.get(field_name)
            placeholder = extract_placeholder(field_value)
            if placeholder is not None:
                if placeholder != field_name:
                    result.error(
                        template_path,
                        f"tensor placeholder '${{case.{placeholder}}}' does not match field '{field_name}'",
                        uid,
                    )
                placeholder_fields.add(field_name)
            else:
                literal_spec[field_name] = field_value

        tensor_placeholder_fields[uid] = placeholder_fields
        literal_tensor_specs[uid] = literal_spec

    return tensor_placeholder_fields, literal_tensor_specs


def resolve_case_tensor_specs(
    case_tensors: object,
    template_tensor_uids: set[int],
    tensor_placeholder_fields: dict[int, set[str]],
    literal_tensor_specs: dict[int, dict[str, object]],
    sweep_path: Path,
    case_label: str,
    result: VerificationResult,
) -> dict[int, dict[str, object]]:
    if not isinstance(case_tensors, list):
        result.error(
            sweep_path,
            f"case '{case_label}' values.tensors is required and must be a list",
        )
        case_tensors = []

    case_tensor_specs: dict[int, dict[str, object]] = {}
    for index, tensor in enumerate(case_tensors):
        if not isinstance(tensor, dict):
            result.error(
                sweep_path, f"case '{case_label}' tensor entry {index} is not an object"
            )
            continue

        uid_value = tensor.get("uid")
        uid = uid_value if is_integer(uid_value) else None
        if uid is None:
            result.error(
                sweep_path,
                f"case '{case_label}' tensor entry {index} uid is required and must be an integer",
            )
            continue

        if uid in case_tensor_specs:
            result.error(
                sweep_path, f"case '{case_label}' duplicate tensor uid declared", uid
            )
            continue

        if uid not in template_tensor_uids:
            result.error(
                sweep_path,
                f"case '{case_label}' tensor uid is not present in template graph",
                uid,
            )
            continue

        required_fields = tensor_placeholder_fields.get(uid, set())
        spec: dict[str, object] = dict(literal_tensor_specs.get(uid, {}))
        for field_name in TEMPLATE_TENSOR_FIELDS:
            if field_name in required_fields:
                if field_name not in tensor:
                    result.error(
                        sweep_path,
                        f"case '{case_label}' is missing placeholder value for tensor '{field_name}'",
                        uid,
                    )
                    continue
                spec[field_name] = tensor[field_name]
            elif field_name in tensor and tensor[field_name] != spec.get(field_name):
                result.warning(
                    sweep_path,
                    f"case '{case_label}' overrides non-placeholder tensor field '{field_name}'",
                    uid,
                )

        case_tensor_specs[uid] = spec

    missing_uids = template_tensor_uids - set(case_tensor_specs)
    if missing_uids:
        result.error(
            sweep_path,
            f"case '{case_label}' is missing tensor uid(s) {sorted(missing_uids)} present in template graph",
        )

    return case_tensor_specs


def validate_case_tensor_shapes(
    case_tensor_specs: dict[int, dict[str, object]],
    sweep_path: Path,
    case_label: str,
    result: VerificationResult,
) -> dict[int, dict[str, object]]:
    validated_specs: dict[int, dict[str, object]] = {}
    for uid, spec in case_tensor_specs.items():
        dims_value = spec.get("dims")
        dims = list(dims_value) if is_integer_list(dims_value) else None
        if dims is None:
            result.error(
                sweep_path,
                f"case '{case_label}' dims is required and must be a list of integers",
                uid,
            )

        strides_value = spec.get("strides")
        strides = list(strides_value) if is_integer_list(strides_value) else None
        if strides is None:
            result.error(
                sweep_path,
                f"case '{case_label}' strides is required and must be a list of integers",
                uid,
            )

        if dims is not None and strides is not None and len(dims) != len(strides):
            result.error(
                sweep_path,
                f"case '{case_label}' dims and strides must have the same length",
                uid,
            )

        if dims is not None and any(dim <= 0 for dim in dims):
            result.error(
                sweep_path,
                f"case '{case_label}' dims must contain only positive integers",
                uid,
            )

        if strides is not None and any(stride < 0 for stride in strides):
            result.error(
                sweep_path,
                f"case '{case_label}' strides must contain only non-negative integers",
                uid,
            )

        data_type = spec.get("data_type")
        if not isinstance(data_type, str):
            result.error(
                sweep_path,
                f"case '{case_label}' data_type is required and must be a string",
                uid,
            )
            continue

        if dims is None or strides is None:
            continue

        validated_specs[uid] = {
            "dims": dims,
            "strides": strides,
            "data_type": data_type,
        }

    return validated_specs


def validate_sweep_case(
    case: object,
    index: int,
    sweep_path: Path,
    bundle_dir: Path,
    template_tensor_uids: set[int],
    tensor_placeholder_fields: dict[int, set[str]],
    literal_tensor_specs: dict[int, dict[str, object]],
    scalar_fields: set[str],
    attribute_fields: set[str],
    output_tensor_uids: set[int],
    seen_case_ids: set[str],
    default_tier: str,
    require_data: bool,
    result: VerificationResult,
) -> None:
    if not isinstance(case, dict):
        result.error(sweep_path, f"case {index} is not an object")
        return

    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        result.error(
            sweep_path, f"case {index} id is required and must be a non-empty string"
        )
        case_id = None
    elif not CASE_ID_PATTERN.match(case_id):
        result.error(sweep_path, f"case id '{case_id}' must be lowercase_snake_case")

    if case_id is not None:
        if case_id in seen_case_ids:
            result.error(sweep_path, f"duplicate case id '{case_id}'")
        seen_case_ids.add(case_id)

    case_label = case_id if case_id is not None else f"#{index}"

    values = case.get("values")
    if not isinstance(values, dict):
        result.error(
            sweep_path, f"case '{case_label}' values is required and must be an object"
        )
        values = {}

    for field_name in sorted(scalar_fields):
        if field_name not in values:
            result.error(
                sweep_path,
                f"case '{case_label}' is missing placeholder value for '{field_name}'",
            )
    for key in values:
        if key in ("tensors", "attributes"):
            continue
        if key not in scalar_fields:
            result.warning(
                sweep_path, f"case '{case_label}' has unused values entry '{key}'"
            )

    attributes = values.get("attributes", {})
    if attribute_fields and not isinstance(attributes, dict):
        result.error(
            sweep_path,
            f"case '{case_label}' values.attributes is required and must be an object",
        )
        attributes = {}
    if isinstance(attributes, dict):
        for field_name in sorted(attribute_fields):
            if field_name not in attributes:
                result.error(
                    sweep_path,
                    f"case '{case_label}' is missing placeholder value for 'attributes.{field_name}'",
                )
        for key in attributes:
            if key not in attribute_fields:
                result.warning(
                    sweep_path,
                    f"case '{case_label}' has unused values.attributes entry '{key}'",
                )

    case_tensor_specs = resolve_case_tensor_specs(
        values.get("tensors"),
        template_tensor_uids,
        tensor_placeholder_fields,
        literal_tensor_specs,
        sweep_path,
        case_label,
        result,
    )
    validated_specs = validate_case_tensor_shapes(
        case_tensor_specs, sweep_path, case_label, result
    )

    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        result.error(
            sweep_path,
            f"case '{case_label}' metadata is required and must be an object",
        )
    else:
        validate_metadata_fields(
            metadata, sweep_path, result, prefix=f"case '{case_label}' metadata "
        )

    golden = case.get("golden")
    golden_dir: Path | None = None
    if golden is not None:
        if not isinstance(golden, dict):
            result.error(
                sweep_path, f"case '{case_label}' golden must be an object or null"
            )
        else:
            golden_path_value = golden.get("path")
            if not isinstance(golden_path_value, str) or not golden_path_value:
                result.error(
                    sweep_path,
                    f"case '{case_label}' golden.path is required and must be a non-empty string",
                )
            else:
                golden_path = bundle_dir / golden_path_value
                if golden_path.name != "tensors.dvc":
                    result.error(
                        sweep_path,
                        f"case '{case_label}' golden.path must reference a tensors.dvc file",
                    )
                else:
                    golden_dir = golden_path.parent

    if case_id is not None:
        advisory = derive_sweep_advisory(sweep_path, case_id, result, default_tier)
        if advisory is not None:
            result.advisories.append(advisory)

    if golden_dir is not None:
        validate_directory_size(golden_dir, golden_dir, result)
        bundle_has_manifest = (golden_dir / "tensors.dvc").is_file()
        validate_tensor_payloads(
            validated_specs,
            lambda uid, golden_dir=golden_dir: golden_dir / f"tensor{uid}.bin",
            bundle_has_manifest,
            require_data,
            output_tensor_uids,
            sweep_path,
            result,
        )


def validate_sweep_bundle(
    template_path: Path,
    sweep_path: Path,
    result: VerificationResult,
    default_tier: str,
    require_data: bool,
) -> None:
    try:
        template = json.loads(template_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(template_path, f"template JSON is not parseable: {error}")
        return

    try:
        sweep = json.loads(sweep_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(sweep_path, f"sweep JSON is not parseable: {error}")
        return

    if not isinstance(template, dict):
        result.error(template_path, "template JSON is not an object")
        return

    if not isinstance(sweep, dict):
        result.error(sweep_path, "sweep JSON is not an object")
        return

    nodes = template.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        result.error(template_path, "top-level nodes must be a non-empty list")
        return

    tensors = template.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        result.error(template_path, "top-level tensors must be a non-empty list")
        return

    output_tensor_uids = extract_output_tensor_uids(nodes, template_path, result)
    tensor_placeholder_fields, literal_tensor_specs = parse_template_tensors(
        tensors, template_path, result
    )
    template_tensor_uids = set(tensor_placeholder_fields)

    scalar_fields: set[str] = set()
    attribute_fields: set[str] = set()
    collect_placeholders(nodes, scalar_fields, attribute_fields)
    for key, value in template.items():
        if key in ("tensors", "nodes"):
            continue
        collect_placeholders(value, scalar_fields, attribute_fields)

    version = sweep.get("version")
    if not is_integer(version) or version < 1:
        result.error(sweep_path, "version is required and must be a positive integer")

    cases = sweep.get("cases")
    if not isinstance(cases, list) or not cases:
        result.error(sweep_path, "top-level cases must be a non-empty list")
        return

    bundle_dir = sweep_path.parent
    seen_case_ids: set[str] = set()

    for index, case in enumerate(cases):
        validate_sweep_case(
            case,
            index,
            sweep_path,
            bundle_dir,
            template_tensor_uids,
            tensor_placeholder_fields,
            literal_tensor_specs,
            scalar_fields,
            attribute_fields,
            output_tensor_uids,
            seen_case_ids,
            default_tier,
            require_data,
            result,
        )


def verify_root(
    root: Path, default_tier: str, require_data: bool
) -> VerificationResult:
    result = VerificationResult()

    if not root.exists():
        result.error(root, "input root does not exist")
        return result

    warn_unexpected_top_level_directories(root, result)

    handled_paths: set[Path] = set()
    for sweep_dir in find_sweep_dirs(root):
        template_path = sweep_dir / "graph.template.json"
        sweep_path = sweep_dir / "sweep.json"
        has_template = template_path.is_file()
        has_sweep = sweep_path.is_file()

        if has_template:
            handled_paths.add(template_path)
        if has_sweep:
            handled_paths.add(sweep_path)

        if has_template and has_sweep:
            validate_sweep_bundle(
                template_path, sweep_path, result, default_tier, require_data
            )
        elif has_template:
            result.error(template_path, "template-sweep bundle is missing sweep.json")
        elif has_sweep:
            result.error(
                sweep_path, "template-sweep bundle is missing graph.template.json"
            )

    for path in iter_json_files(root):
        if path in handled_paths:
            continue

        if is_metadata_sidecar(path):
            validate_metadata(path, result)
            continue

        if is_graph_candidate(path):
            validate_graph_bundle(path, result, default_tier, require_data)
            continue

        result.warning(
            path,
            "non-graph JSON ignored; graph files must be named <BundleName>/<BundleName>.json",
        )

    return result


def verify_roots(
    roots: list[Path], default_tier: str, require_data: bool
) -> VerificationResult:
    result = VerificationResult()
    for root in roots:
        root_result = verify_root(root, default_tier, require_data)
        result.diagnostics.extend(root_result.diagnostics)
        result.advisories.extend(root_result.advisories)
    return result


def main() -> int:
    args = parse_args()
    result = verify_roots(args.roots, args.default_tier, args.require_data)
    result.print_advisories()
    result.print_diagnostics()
    return 1 if result.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
