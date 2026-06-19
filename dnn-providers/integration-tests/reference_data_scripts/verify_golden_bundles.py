#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify hipDNN golden bundle directories."""

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_TIERS = {"quick", "standard", "comprehensive", "full"}

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


def iter_json_files(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix != ".json":
            return []
        return [root]

    return sorted(path for path in root.rglob("*.json") if path.is_file())


def warn_unexpected_top_level_directories(
    root: Path, result: VerificationResult
) -> None:
    if not root.is_dir() or root.name in ALLOWED_TIERS:
        return

    child_directories = sorted(path for path in root.iterdir() if path.is_dir())
    has_tier_children = any(path.name in ALLOWED_TIERS for path in child_directories)
    if root.name != "integration_test_bundles" and not has_tier_children:
        return

    for path in child_directories:
        if path.name not in ALLOWED_TIERS:
            result.warning(path, "unexpected top-level directory")


def validate_metadata(path: Path, result: VerificationResult) -> None:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(path, f"metadata JSON is not parseable: {error}")
        return

    if not isinstance(metadata, dict):
        result.error(path, "metadata JSON is not an object")
        return

    for key in ("generator", "reference_source"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            result.error(path, f"{key} is required and must be a non-empty string")


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_integer_list(value: object) -> bool:
    return isinstance(value, list) and all(is_integer(item) for item in value)


def element_space(dims: list[int], strides: list[int]) -> int:
    return 1 + sum((dim - 1) * stride for dim, stride in zip(dims, strides))


def extract_output_tensor_uids(
    nodes: list[object], path: Path, result: VerificationResult
) -> set[int]:
    output_tensor_uids: set[int] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            result.error(path, f"node {index} is not an object")
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


def validate_graph_bundle(
    path: Path, result: VerificationResult, default_tier: str
) -> None:
    advisory = derive_advisory(path, result, default_tier)
    if advisory is not None:
        result.advisories.append(advisory)

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

    base_path = path.with_suffix("")
    for uid, tensor_spec in tensor_specs.items():
        dims = tensor_spec["dims"]
        strides = tensor_spec["strides"]
        data_type = tensor_spec["data_type"]
        dtype_key = data_type.lower()
        element_size = DTYPE_BYTE_SIZE.get(dtype_key)
        tensor_path = Path(f"{base_path}.tensor{uid}.bin")

        if element_size is None:
            result.error(
                path,
                f"unsupported data_type '{data_type}' for byte-size validation",
                uid,
            )
            continue

        if not tensor_path.exists():
            result.error(
                tensor_path, f"missing tensor file; expected {tensor_path}", uid
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

        if uid not in output_tensor_uids or dtype_key not in FLOAT_DTYPES:
            continue

        try:
            data = tensor_path.read_bytes()
        except OSError as error:
            result.error(tensor_path, f"could not read tensor file: {error}", uid)
            continue

        bad_index = find_nonfinite_index(dtype_key, data)
        if bad_index is not None:
            result.error(
                tensor_path,
                f"output tensor contains NaN/Inf at element index {bad_index}",
                uid,
            )

    for output_tensor_uid in output_tensor_uids:
        if output_tensor_uid not in tensor_specs:
            result.error(
                path, "output tensor uid is not declared in tensors", output_tensor_uid
            )


def verify_root(root: Path, default_tier: str) -> VerificationResult:
    result = VerificationResult()

    if not root.exists():
        result.error(root, "input root does not exist")
        return result

    warn_unexpected_top_level_directories(root, result)

    for path in iter_json_files(root):
        if is_metadata_sidecar(path):
            validate_metadata(path, result)
            continue

        if is_graph_candidate(path):
            validate_graph_bundle(path, result, default_tier)
            continue

        result.warning(
            path,
            "non-graph JSON ignored; graph files must be named <BundleName>/<BundleName>.json",
        )

    return result


def verify_roots(roots: list[Path], default_tier: str) -> VerificationResult:
    result = VerificationResult()
    for root in roots:
        root_result = verify_root(root, default_tier)
        result.diagnostics.extend(root_result.diagnostics)
        result.advisories.extend(root_result.advisories)
    return result


def main() -> int:
    args = parse_args()
    result = verify_roots(args.roots, args.default_tier)
    result.print_advisories()
    result.print_diagnostics()
    return 1 if result.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
