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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "migration-scripts"))
from bundle_utils import (  # noqa: E402
    ABSENT_TENSORS_KEY,
    UNRESOLVED,
    absent_case_problems,
    walk_dotted_path,
)

ALLOWED_TIERS = {"quick", "standard", "comprehensive", "full"}
BUNDLE_SIZE_WARNING_BYTES = 1024 * 1024
BUNDLE_SIZE_ERROR_BYTES = 2 * 1024 * 1024
REFERENCE_DISABLE_FLAGS = ("no_cpu_reference", "no_gpu_reference")
BOOLEAN_METADATA_FLAGS = (*REFERENCE_DISABLE_FLAGS, "expected_rejection")

MIN_TIERED_BUNDLE_SEGMENTS = 6

ADVISORY_LAYOUT_ERROR = (
    "cannot derive advisory path; expected "
    "{Tier}/{Operation}/{Layout}/{DataType}/[{Variant}/...]/{Name}/{Name}.json"
)

CASE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
PLACEHOLDER_PATTERN = re.compile(r"^\$\{case\.([A-Za-z0-9_.]+)\}$")
TEMPLATE_TENSOR_FIELDS = ("dims", "strides", "data_type")
# Fields a template tensor may also carry, as a literal or a placeholder: the ragged
# binding that sizes the tensor's payload (RFC 0014).
OPTIONAL_TEMPLATE_TENSOR_FIELDS = (
    "ragged_offset_tensor_uid",
    "ragged_offset_multiplier",
)

# Node types whose real hipDNN JSON schema declares output tensor uids as flat
# scalar fields directly on the node, instead of nesting them under an `outputs`
# object like every other attribute type. See
# flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/utilities/json/ReductionAttributes.hpp
# to_json/from_json, backed by reduction_attributes.fbs (`in_tensor_uid`/
# `out_tensor_uid` are plain scalar fields on ReductionAttributes, not a map).
FLAT_OUTPUT_UID_FIELDS: dict[str, tuple[str, ...]] = {
    "ReductionAttributes": ("out_tensor_uid",),
}

# SDPA stats (LSE) tensors, as (uid map, field) per node type: the forward node's
# stats output and the backward node's stats input. A padded or fully masked row
# has LSE = -inf by contract (SEM-010 in projects/hipdnn/docs/SdpaSemanticsRegister.md,
# and the harness accepts a matching -inf), so -inf is a valid value in these
# tensors only; NaN and +inf stay defects.
SDPA_STATS_UID_FIELDS: dict[str, tuple[str, str]] = {
    "SdpaAttributes": ("outputs", "stats_tensor_uid"),
    "SdpaBackwardAttributes": ("inputs", "stats_tensor_uid"),
}

# The SDPA tensors whose ragged batches take their lengths from a node's seq_len
# (RFC 0014), per node type: seq_len field (in `inputs`) -> the (uid map, field) of each
# tensor it sizes. A node without the seq_len leaves these tensors to offset deltas.
SDPA_RAGGED_LENGTH_ROLES: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "SdpaAttributes": {
        "seq_len_q_tensor_uid": (
            ("inputs", "q_tensor_uid"),
            ("outputs", "o_tensor_uid"),
            ("outputs", "stats_tensor_uid"),
            ("outputs", "max_tensor_uid"),
            ("outputs", "sum_exp_tensor_uid"),
        ),
        "seq_len_kv_tensor_uid": (
            ("inputs", "k_tensor_uid"),
            ("inputs", "v_tensor_uid"),
        ),
    },
    "SdpaBackwardAttributes": {
        "seq_len_q_tensor_uid": (
            ("inputs", "q_tensor_uid"),
            ("inputs", "o_tensor_uid"),
            ("inputs", "do_tensor_uid"),
            ("outputs", "dq_tensor_uid"),
            ("inputs", "stats_tensor_uid"),
        ),
        "seq_len_kv_tensor_uid": (
            ("inputs", "k_tensor_uid"),
            ("inputs", "v_tensor_uid"),
            ("outputs", "dk_tensor_uid"),
            ("outputs", "dv_tensor_uid"),
        ),
    },
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

RAGGED_OFFSET_FORMATS = {"int32": "<i", "int64": "<q"}

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


# RFC 0011 4.2 ("Forward-Backward Generation Constraint"): a backward bundle's
# O and LSE come from a forward pass, and if that forward pass is regenerated on
# its own the backward bundle stays internally consistent with its stale inputs
# while silently describing a different reference implementation. forward_source
# is what makes that detectable, so it is mandatory on backward bundles -- the
# one case where RFC 0011 requires a .meta.json at all rather than allowing the
# tolerance fallback.
FORWARD_SOURCE_FIELDS = ("method", "version", "generated_at")


def validate_forward_source(
    metadata: dict,
    path: Path,
    result: VerificationResult,
    prefix: str,
    base_dir: Path | None,
) -> None:
    forward_source = metadata.get("forward_source")
    if forward_source is None:
        result.error(
            path,
            f"{prefix}forward_source is required on backward bundles "
            "(RFC 0011 4.2): it names the forward pass whose O and LSE this "
            "bundle's gradients were computed from",
        )
        return
    if not isinstance(forward_source, dict):
        result.error(path, f"{prefix}forward_source must be an object")
        return

    for key in FORWARD_SOURCE_FIELDS:
        value = forward_source.get(key)
        if not isinstance(value, str) or not value.strip():
            result.error(
                path,
                f"{prefix}forward_source.{key} is required and must be a "
                "non-empty string",
            )

    validate_forward_source_pairing(forward_source, path, result, prefix, base_dir)


def validate_forward_source_pairing(
    forward_source: dict,
    path: Path,
    result: VerificationResult,
    prefix: str,
    base_dir: Path | None,
) -> None:
    """Cross-check forward_source against the forward bundle it names.

    RFC 0011 4.2 asks for this comparison: a difference in method or generation
    timestamp means the forward bundle was regenerated without its backward
    partner, which no test would otherwise catch. The version field is required
    to be present but is not compared.

    The `bundle` field is optional, because RFC 0011's own example omits it; the
    check is skipped, rather than failed, when it is absent or when the forward
    bundle is not on disk (a backward bundle can legitimately be verified
    without its partner pulled).
    """
    bundle = forward_source.get("bundle")
    if bundle is None or base_dir is None:
        return
    if not isinstance(bundle, str) or not bundle.strip():
        result.error(path, f"{prefix}forward_source.bundle must be a non-empty string")
        return

    forward_graph = (base_dir / bundle).resolve()
    forward_meta = forward_graph.with_suffix(".meta.json")
    if forward_graph.suffix != ".json" or forward_graph.name.endswith(".meta.json"):
        result.error(
            path,
            f"{prefix}forward_source.bundle must point at the forward bundle's "
            f"graph JSON (got '{bundle}')",
        )
        return
    if not forward_meta.is_file():
        # Verifying one subtree at a time is normal, and a backward bundle's
        # forward partner is in a sibling directory, so an absent partner is not
        # itself a defect. The pairing is cross-checked whenever both are in
        # scope, which is the case for a whole-tree run.
        return

    try:
        metadata = json.loads(forward_meta.read_text())
    except (OSError, json.JSONDecodeError) as error:
        result.error(
            path, f"{prefix}forward_source bundle metadata is not parseable: {error}"
        )
        return
    if not isinstance(metadata, dict):
        result.error(path, f"{prefix}forward_source bundle metadata is not an object")
        return

    if metadata.get("direction") not in (None, "forward"):
        result.error(
            path,
            f"{prefix}forward_source names '{bundle}', which is not a forward "
            f"bundle (direction '{metadata.get('direction')}')",
        )

    # The forward bundle records its generation timestamp as generated_at and
    # the method that produced it as reference_backend; forward_source restates
    # both. Drift between them is the stale-pairing this check exists to find.
    for source_key, forward_key in (
        ("method", "reference_backend"),
        ("generated_at", "generated_at"),
    ):
        expected = metadata.get(forward_key)
        actual = forward_source.get(source_key)
        if expected is None or actual is None:
            continue
        if expected != actual:
            result.error(
                path,
                f"{prefix}forward_source.{source_key} is '{actual}' but "
                f"'{bundle}' records {forward_key} '{expected}'; the forward "
                "bundle was regenerated without its backward partner",
            )


def validate_metadata_fields(
    metadata: dict,
    path: Path,
    result: VerificationResult,
    prefix: str = "",
    base_dir: Path | None = None,
) -> None:
    for key in ("generator", "reference_source"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            result.error(
                path, f"{prefix}{key} is required and must be a non-empty string"
            )
    for key in BOOLEAN_METADATA_FLAGS:
        if key in metadata and not isinstance(metadata[key], bool):
            result.error(path, f"{prefix}{key} must be a boolean")

    direction = metadata.get("direction")
    if direction is not None and direction not in ("forward", "backward"):
        result.error(
            path,
            f"{prefix}direction must be 'forward' or 'backward' when present "
            f"(got {direction!r})",
        )

    strategy = metadata.get("reference_strategy")
    strategy_present = isinstance(strategy, str) and strategy.strip()
    if direction == "backward":
        if not strategy_present:
            # RFC 0011 4.1 lists reference_strategy as mandatory whenever a
            # .meta.json is present. It is enforced here only for backward
            # bundles, which RFC 0011 requires to carry metadata at all; making
            # it a hard error everywhere would reject the forward bundles
            # migrated from the old test system, which the RFC explicitly
            # allows to have no metadata.
            result.error(
                path,
                f"{prefix}reference_strategy is required on backward bundles "
                "(RFC 0011 4.1)",
            )
        validate_forward_source(metadata, path, result, prefix, base_dir)
    elif not strategy_present and strategy is not None:
        result.error(path, f"{prefix}reference_strategy must be a non-empty string")


def validate_reference_flags_require_golden(
    metadata: dict,
    golden_declared: bool,
    path: Path,
    result: VerificationResult,
    prefix: str = "",
) -> None:
    # An expected rejection ignores the reference flags: nothing is ever compared.
    if metadata.get("expected_rejection") is True:
        return
    if (
        all(metadata.get(key) is True for key in REFERENCE_DISABLE_FLAGS)
        and not golden_declared
    ):
        result.error(
            path,
            f"{prefix}disables both references (no_cpu_reference and "
            "no_gpu_reference) but declares no golden data; add golden data or "
            "enable a reference",
        )


def validate_expected_rejection_has_no_golden(
    metadata: dict,
    golden_declared: bool,
    path: Path,
    result: VerificationResult,
    prefix: str = "",
) -> None:
    if metadata.get("expected_rejection") is True and golden_declared:
        result.error(
            path,
            f"{prefix}is an expected_rejection bundle but declares golden data; "
            "it is never executed, so there is nothing to compare against",
        )


def single_graph_golden_declared(directory: Path, stem: str) -> bool:
    """Mirror the C++ loader: the `.tensors.dvc` pointer, or a blob for every output.

    Input-only or partial output blobs do not declare golden data. A graph that
    cannot be read declares only through its pointer (the graph check reports it).
    """
    if (directory / f"{stem}.tensors.dvc").is_file():
        return True
    try:
        graph = json.loads((directory / f"{stem}.json").read_text())
        nodes = graph["nodes"]
        virtual = {
            tensor["uid"]
            for tensor in graph["tensors"]
            if isinstance(tensor, dict) and tensor.get("virtual") is True
        }
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(nodes, list):
        return False
    output_uids = extract_output_tensor_uids(nodes, directory, VerificationResult())
    output_uids -= virtual
    return bool(output_uids) and all(
        (directory / f"{stem}.tensor{uid}.bin").is_file() for uid in output_uids
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

    validate_metadata_fields(metadata, path, result, base_dir=path.parent)

    if path.name.endswith(".meta.json") and path.name != "meta.json":
        stem = path.name[: -len(".meta.json")]
        golden_declared = single_graph_golden_declared(path.parent, stem)
        validate_reference_flags_require_golden(metadata, golden_declared, path, result)
        validate_expected_rejection_has_no_golden(
            metadata, golden_declared, path, result
        )


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


def extract_sdpa_stats_tensor_uids(nodes: list[object]) -> set[int]:
    """Uids of the SDPA stats (LSE) tensors, where -inf is allowed (SEM-010)."""
    stats_tensor_uids: set[int] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        location = SDPA_STATS_UID_FIELDS.get(node.get("type"))
        if location is None:
            continue
        uid_map_name, field_name = location
        uid_map = node.get(uid_map_name)
        if not isinstance(uid_map, dict):
            continue
        value = uid_map.get(field_name)
        if is_integer(value):
            stats_tensor_uids.add(value)
    return stats_tensor_uids


def extract_ragged_length_uids(nodes: list[object]) -> dict[int, int]:
    """Ragged tensor uid -> the uid of the seq_len tensor that sizes its batches.
    Raises ValueError when a ragged tensor is bound to two different seq_len tensors."""
    length_uids: dict[int, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        roles = SDPA_RAGGED_LENGTH_ROLES.get(node.get("type"))
        inputs = node.get("inputs")
        if roles is None or not isinstance(inputs, dict):
            continue
        for length_field, tensors in roles.items():
            length_uid = inputs.get(length_field)
            if not is_integer(length_uid):
                continue
            for uid_map_name, field_name in tensors:
                uid_map = node.get(uid_map_name)
                if isinstance(uid_map, dict) and is_integer(uid_map.get(field_name)):
                    ragged_uid = uid_map[field_name]
                    if length_uids.setdefault(ragged_uid, length_uid) != length_uid:
                        raise ValueError(
                            f"ragged tensor {ragged_uid} is bound to two seq_len "
                            f"tensors, {length_uids[ragged_uid]} and {length_uid}"
                        )
    return length_uids


def iter_float_values(dtype_key: str, data: bytes):
    """Decoded values of a float tensor payload; nothing for other dtypes."""
    if dtype_key in {"float", "float32", "fp32"}:
        return (value for (value,) in struct.iter_unpack("<f", data))
    if dtype_key in {"double", "float64", "fp64"}:
        return (value for (value,) in struct.iter_unpack("<d", data))
    if dtype_key in {"half", "float16", "fp16"}:
        return (value for (value,) in struct.iter_unpack("<e", data))
    if dtype_key in {"bfloat16", "bf16", "bfp16"}:
        return (
            struct.unpack("<f", struct.pack("<I", word << 16))[0]
            for (word,) in struct.iter_unpack("<H", data)
        )
    return iter(())


def find_invalid_stats_index(
    dtype_key: str, data: bytes, allow_nan: bool
) -> int | None:
    """First +inf (or NaN, unless allowed) in an SDPA stats tensor; -inf is valid."""
    for index, value in enumerate(iter_float_values(dtype_key, data)):
        if value == math.inf or (math.isnan(value) and not allow_nan):
            return index
    return None


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


def find_infinite_index(dtype_key: str, data: bytes) -> int | None:
    if dtype_key in {"float", "float32", "fp32"}:
        for index, (value,) in enumerate(struct.iter_unpack("<f", data)):
            if math.isinf(value):
                return index
        return None

    if dtype_key in {"double", "float64", "fp64"}:
        for index, (value,) in enumerate(struct.iter_unpack("<d", data)):
            if math.isinf(value):
                return index
        return None

    if dtype_key in {"half", "float16", "fp16"}:
        for index, (word,) in enumerate(struct.iter_unpack("<H", data)):
            if ((word >> 10) & 0x1F) == 0x1F and (word & 0x3FF) == 0:
                return index
        return None

    if dtype_key in {"bfloat16", "bf16", "bfp16"}:
        for index, (word,) in enumerate(struct.iter_unpack("<H", data)):
            if ((word >> 7) & 0xFF) == 0xFF and (word & 0x7F) == 0:
                return index
        return None

    return None


def read_ragged_values(path: Path, value_format: str) -> list[int]:
    data = path.read_bytes()
    width = struct.calcsize(value_format)
    if not data or len(data) % width != 0:
        raise ValueError(
            f"tensor has {len(data)} bytes, not a non-empty multiple of {width}"
        )
    return [value for (value,) in struct.iter_unpack(value_format, data)]


def report_missing_tensor_file(
    tensor_path: Path,
    uid: int,
    bundle_has_manifest: bool,
    require_data: bool,
    result: VerificationResult,
) -> None:
    if not bundle_has_manifest:
        return
    if require_data:
        result.error(tensor_path, f"missing tensor file; expected {tensor_path}", uid)
    else:
        result.warning(
            tensor_path,
            "tensor data not pulled locally; expected "
            f"{tensor_path} (run `dvc pull` or pass --require-data "
            "to enforce in CI)",
            uid,
        )


def ragged_element_space(
    dims: list[int],
    strides: list[int],
    multiplier: int,
    offsets: list[int],
    lengths: list[int] | None,
) -> int:
    """elementSpace of a ragged tensor (RFC 0014 §4.5): 1 + sum over the non-batch
    axes of (extent - 1) * stride, with maxRow + 1 (the furthest row any sequence
    addresses, plus one) as the sequence axis extent; 0 when no sequence has a row or
    a non-sequence axis has extent 0. Batch b starts at
    offsets[b] * multiplier and holds lengths[b] token rows; without lengths the
    offsets must be monotonic and the blocks give the lengths. Raises ValueError
    naming the violated rule."""
    batches, s_max, seq_stride = dims[0], dims[2], strides[2]
    if seq_stride <= 0 or any(stride < 0 for stride in strides[1:]):
        raise ValueError(
            "ragged strides must be non-negative with a positive sequence stride; "
            "the batch-axis stride is unused "
            f"(negative-stride rule; got {strides})"
        )
    if len(offsets) != batches + 1:
        raise ValueError(
            f"ragged_offset tensor holds {len(offsets)} values; it must hold exactly "
            f"B+1 = {batches + 1} (offset-count rule)"
        )
    if lengths is not None and len(lengths) != batches:
        raise ValueError(
            f"seq_len tensor holds {len(lengths)} values; it must hold exactly "
            f"B = {batches} (seq_len-count rule)"
        )
    origins = [offset * multiplier for offset in offsets]
    if lengths is not None:
        origins = origins[:batches]
    if any(origin < 0 or origin % seq_stride for origin in origins):
        raise ValueError(
            f"origins {origins} are not whole tokens >= 0 of {seq_stride} elements "
            "(whole-token origin rule)"
        )
    if lengths is None:
        if any(later < earlier for earlier, later in zip(origins, origins[1:])):
            raise ValueError(
                f"offsets {origins} are not monotonic without seq_len (monotonic rule)"
            )
        lengths = [
            min((later - earlier) // seq_stride, s_max)
            for earlier, later in zip(origins, origins[1:])
        ]
    if any(length < 0 or length > s_max for length in lengths):
        raise ValueError(
            f"lengths {lengths} must lie in [0, S_max = {s_max}] (length range rule)"
        )
    live = sorted(
        (origin // seq_stride, length)
        for origin, length in zip(origins, lengths)
        if length > 0
    )
    if not live:
        return 0
    for (start, length), (next_start, _) in zip(live, live[1:]):
        if next_start < start + length:
            raise ValueError(
                f"ragged sequences overlap: rows from {next_start} start inside rows "
                f"{start}..{start + length - 1} (overlap rule)"
            )
    if 0 in (dims[1], dims[3]):
        return 0
    max_row = max(start + length - 1 for start, length in live)
    span = 1
    for stride, extent in sorted(
        (strides[axis], extent)
        for axis, extent in ((2, max_row + 1), (1, dims[1]), (3, dims[3]))
        if extent > 1 and strides[axis] > 0
    ):
        if stride < span:
            raise ValueError(
                f"ragged strides {strides} alias: stride {stride} is below the span "
                f"{span} of the axes under it (sorted-stride rule)"
            )
        span += (extent - 1) * stride
    return (
        1
        + max_row * seq_stride
        + (dims[1] - 1) * strides[1]
        + (dims[3] - 1) * strides[3]
    )


def read_ragged_tensor_values(
    uid: int,
    owner_uid: int,
    role: str,
    count: int,
    count_label: str,
    tensor_specs: dict[int, dict[str, object]],
    tensor_path_for: Callable[[int], Path],
    bundle_has_manifest: bool,
    require_data: bool,
    result: VerificationResult,
) -> list[int] | None:
    """Integer values of the offset or seq_len tensor `uid` of ragged tensor
    `owner_uid`, or None if unknown. The tensor must have shape [count, 1, 1, 1];
    `count_label` is that count as written in the message (B + 1 or B)."""
    spec = tensor_specs.get(uid)
    if spec is None:
        result.error(
            tensor_path_for(owner_uid),
            f"{role} {uid} is not a declared tensor",
            owner_uid,
        )
        return None
    if len(spec["dims"]) != 4:
        result.error(
            tensor_path_for(owner_uid),
            f"{role} {uid} must be rank 4 (has dims {spec['dims']})",
            owner_uid,
        )
        return None
    if list(spec["dims"]) != [count, 1, 1, 1]:
        result.error(
            tensor_path_for(owner_uid),
            f"{role} {uid} must have shape [{count_label}, 1, 1, 1] "
            f"(has dims {spec['dims']})",
            owner_uid,
        )
        return None
    data_type = spec["data_type"]
    value_format = RAGGED_OFFSET_FORMATS.get(data_type.lower())
    if value_format is None:
        result.error(
            tensor_path_for(owner_uid),
            f"{role} {uid} has data_type '{data_type}'; expected int32 or int64",
            owner_uid,
        )
        return None
    path = tensor_path_for(uid)
    if not path.exists():
        report_missing_tensor_file(path, uid, bundle_has_manifest, require_data, result)
        return None
    try:
        values = read_ragged_values(path, value_format)
    except (OSError, ValueError) as error:
        result.error(path, f"could not read {role} values: {error}", uid)
        return None
    # Value i sits at element i * strides[0]; the file spans the tensor's element space.
    stride = spec["strides"][0]
    if count > 1 and stride < 1:
        result.error(
            path,
            f"{role} {uid} must have a positive batch stride (has {stride})",
            uid,
        )
        return None
    if stride > 1:
        span = 1 + (count - 1) * stride
        if len(values) != span:
            result.error(
                path,
                f"{role} {uid} file holds {len(values)} values but its batch "
                f"stride {stride} spans {span}",
                uid,
            )
            return None
        values = values[::stride]
    return values


def ragged_tensor_element_space(
    tensor_spec: dict[str, object],
    tensor_specs: dict[int, dict[str, object]],
    length_uids: dict[int, int],
    tensor_path_for: Callable[[int], Path],
    bundle_has_manifest: bool,
    require_data: bool,
    uid: int,
    result: VerificationResult,
) -> int | None:
    """The RFC 0014 §4.5 elementSpace of a ragged tensor from its offset tensor and,
    when its SDPA node names the role's seq_len, that tensor's lengths; None if it
    cannot be determined."""
    if len(tensor_spec["dims"]) != 4 or len(tensor_spec["strides"]) != 4:
        result.error(
            tensor_path_for(uid),
            "invalid ragged layout: ragged tensors must have rank 4 "
            f"(got dims {tensor_spec['dims']}, strides {tensor_spec['strides']})",
            uid,
        )
        return None
    offset_uid = tensor_spec["ragged_offset_tensor_uid"]
    offsets = read_ragged_tensor_values(
        offset_uid,
        uid,
        "ragged_offset_tensor_uid",
        tensor_spec["dims"][0] + 1,
        "B + 1",
        tensor_specs,
        tensor_path_for,
        bundle_has_manifest,
        require_data,
        result,
    )
    if offsets is None:
        return None
    lengths = None
    length_uid = length_uids.get(uid)
    if length_uid is not None:
        lengths = read_ragged_tensor_values(
            length_uid,
            uid,
            "seq_len_tensor_uid",
            tensor_spec["dims"][0],
            "B",
            tensor_specs,
            tensor_path_for,
            bundle_has_manifest,
            require_data,
            result,
        )
        if lengths is None:
            return None
    try:
        return ragged_element_space(
            tensor_spec["dims"],
            tensor_spec["strides"],
            tensor_spec.get("ragged_offset_multiplier", 1),
            offsets,
            lengths,
        )
    except ValueError as error:
        result.error(tensor_path_for(uid), f"invalid ragged layout: {error}", uid)
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
    stats_tensor_uids: set[int] | None = None,
    length_uids: dict[int, int] | None = None,
) -> None:
    stats_tensor_uids = stats_tensor_uids or set()
    length_uids = length_uids or {}
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
            report_missing_tensor_file(
                tensor_path, uid, bundle_has_manifest, require_data, result
            )
            continue

        try:
            actual_size = tensor_path.stat().st_size
        except OSError as error:
            result.error(tensor_path, f"could not stat tensor file: {error}", uid)
            continue

        is_ragged = tensor_spec.get("ragged_offset_tensor_uid") is not None
        if is_ragged:
            element_count = ragged_tensor_element_space(
                tensor_spec,
                tensor_specs,
                length_uids,
                tensor_path_for,
                bundle_has_manifest,
                require_data,
                uid,
                result,
            )
            if element_count is None:
                continue
            size_basis = f"ragged elementSpace={element_count}"
        else:
            element_count = element_space(dims, strides)
            size_basis = f"element_space={element_count}"

        expected_size = element_count * element_size
        if actual_size != expected_size:
            result.error(
                tensor_path,
                "file has "
                f"{actual_size} bytes but graph expects {expected_size} bytes "
                f"({size_basis}, element_size={element_size})",
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

        is_output = uid in output_tensor_uids
        if uid in stats_tensor_uids:
            # Padded and fully masked rows carry LSE = -inf (SEM-010); a ragged
            # stats output may also carry the NaN pad sentinel (RFC 0014 §4.11.4).
            allow_nan = is_ragged and is_output
            bad_index = find_invalid_stats_index(dtype_key, data, allow_nan)
            if bad_index is not None:
                rejected = "+Inf" if allow_nan else "NaN/+Inf"
                result.error(
                    tensor_path,
                    f"SDPA stats tensor contains {rejected} at element index "
                    f"{bad_index} (only -inf is allowed, SEM-010)",
                    uid,
                )
            continue

        if is_ragged and is_output:
            # Ragged outputs carry the NaN sentinel in in-block pad rows
            # (RFC 0014 §4.11.4), so only Inf is a defect there.
            bad_index = find_infinite_index(dtype_key, data)
            if bad_index is not None:
                result.error(
                    tensor_path,
                    f"ragged output tensor contains Inf at element index {bad_index}",
                    uid,
                )
            continue

        bad_index = find_nonfinite_index(dtype_key, data)
        if bad_index is not None:
            tensor_role = "output" if is_output else "input"
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
    # Scan backwards from the deepest position the minimal layout allows: variant and
    # bundle directories may be named after a tier, and so may an enclosing directory.
    last_possible_tier_index = len(parts) - MIN_TIERED_BUNDLE_SEGMENTS
    tier_index = next(
        (
            index
            for index in range(last_possible_tier_index, -1, -1)
            if parts[index] in ALLOWED_TIERS
        ),
        None,
    )

    if tier_index is None and any(part in ALLOWED_TIERS for part in parts[:-1]):
        result.error(path, ADVISORY_LAYOUT_ERROR)
        return None

    if tier_index is None:
        result.warning(
            path,
            f"no tier directory found; using default tier '{default_tier}' for advisory"
            " output, and any variant directories cannot be recovered",
        )
        if len(parts) < 5:
            result.error(path, ADVISORY_LAYOUT_ERROR)
            return None
        tier = default_tier
        bundle_parts = parts[-5:]
    else:
        tier = parts[tier_index]
        bundle_parts = parts[tier_index + 1 :]

    operation, layout, data_type = bundle_parts[:3]
    *variants, name, _ = bundle_parts[3:]

    suite_segments = (tier, operation, layout, data_type, *variants, name)
    canonical_path = "/".join(suite_segments) + "/"
    test_suite = sanitize_gtest_name("_".join(suite_segments))
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

        ragged_offset_uid = tensor.get("ragged_offset_tensor_uid")
        ragged_fields_valid = True
        if ragged_offset_uid is not None and not is_integer(ragged_offset_uid):
            result.error(path, "ragged_offset_tensor_uid must be an integer", uid)
            ragged_fields_valid = False

        ragged_offset_multiplier = tensor.get("ragged_offset_multiplier", 1)
        if not is_integer(ragged_offset_multiplier) or ragged_offset_multiplier < 1:
            result.error(path, "ragged_offset_multiplier must be an integer >= 1", uid)
            ragged_fields_valid = False

        if uid is None:
            continue

        if uid in tensor_specs:
            result.error(path, "duplicate tensor uid declared", uid)
            continue

        if (
            dims is None
            or strides is None
            or not isinstance(data_type, str)
            or not ragged_fields_valid
        ):
            continue

        tensor_specs[uid] = {
            "dims": dims,
            "strides": strides,
            "data_type": data_type,
            "ragged_offset_tensor_uid": ragged_offset_uid,
            "ragged_offset_multiplier": ragged_offset_multiplier,
        }

    try:
        length_uids = extract_ragged_length_uids(nodes)
    except ValueError as error:
        result.error(path, f"invalid ragged seq_len binding: {error}")
        return

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
        extract_sdpa_stats_tensor_uids(nodes),
        length_uids,
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


def first_path_token(path: str) -> str:
    """The first dotted component of a placeholder path: ``a.b`` -> ``a``."""
    return path.split(".", 1)[0]


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
        for field_name in TEMPLATE_TENSOR_FIELDS + OPTIONAL_TEMPLATE_TENSOR_FIELDS:
            field_value = tensor.get(field_name)
            if field_value is None and field_name in OPTIONAL_TEMPLATE_TENSOR_FIELDS:
                continue
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
    absent_uids: set[int] = frozenset(),
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

        if uid in absent_uids:
            continue

        required_fields = tensor_placeholder_fields.get(uid, set())
        spec: dict[str, object] = dict(literal_tensor_specs.get(uid, {}))
        for field_name in TEMPLATE_TENSOR_FIELDS + OPTIONAL_TEMPLATE_TENSOR_FIELDS:
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

    missing_uids = template_tensor_uids - absent_uids - set(case_tensor_specs)
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
        for field_name in OPTIONAL_TEMPLATE_TENSOR_FIELDS:
            if spec.get(field_name) is not None:
                validated_specs[uid][field_name] = spec[field_name]
        multiplier = spec.get("ragged_offset_multiplier", 1)
        if not is_integer(multiplier) or multiplier < 1:
            result.error(
                sweep_path,
                f"case '{case_label}' ragged_offset_multiplier must be an integer >= 1",
                uid,
            )

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
    stats_tensor_uids: set[int] | None = None,
    length_uids: dict[int, int] | None = None,
    absent_uids: set[int] = frozenset(),
    tensor_scoped_fields: set[tuple[int, str]] | None = None,
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

    # An absent tensor leaves the template before placeholder expansion, so its
    # placeholders need no values.
    tensor_scoped_fields = {
        (uid, f) for uid, f in tensor_scoped_fields or () if uid not in absent_uids
    }

    for field_name in sorted(scalar_fields):
        if first_path_token(field_name) not in values:
            result.error(
                sweep_path,
                f"case '{case_label}' is missing placeholder value for '{field_name}'",
            )
    # A placeholder on a tensor's own template object resolves the way the C++
    # loader's resolvePlaceholder() does: first by walking the full (possibly
    # dotted) path through that tensor's own values.tensors[] entry, then -- if
    # that walk fails -- by walking the same path through the top-level values
    # instead. There is no restriction on which field names may be looked up
    # per-tensor.
    raw_case_tensors = {}
    if isinstance(values.get("tensors"), list):
        for tensor in values["tensors"]:
            if isinstance(tensor, dict) and is_integer(tensor.get("uid")):
                raw_case_tensors.setdefault(tensor["uid"], tensor)
    for uid, field_name in sorted(tensor_scoped_fields):
        tensor_entry = raw_case_tensors.get(uid)
        resolved = (
            walk_dotted_path(tensor_entry, field_name)
            if tensor_entry is not None
            else UNRESOLVED
        )
        if resolved is UNRESOLVED:
            resolved = walk_dotted_path(values, field_name)
        if resolved is UNRESOLVED:
            result.error(
                sweep_path,
                f"case '{case_label}' is missing placeholder value for '{field_name}'",
                uid,
            )
    used_keys = {first_path_token(f) for f in scalar_fields} | {
        first_path_token(f) for _, f in tensor_scoped_fields
    }
    for key in values:
        if key in ("tensors", "attributes"):
            continue
        if key not in used_keys:
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
        absent_uids,
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
        # A sweep case's forward_source path is written relative to the case's
        # own golden directory, which is where its tensors live.
        golden_spec = case.get("golden")
        case_base_dir = None
        if isinstance(golden_spec, dict) and isinstance(golden_spec.get("path"), str):
            case_base_dir = (sweep_path.parent / golden_spec["path"]).parent
        validate_metadata_fields(
            metadata,
            sweep_path,
            result,
            prefix=f"case '{case_label}' metadata ",
            base_dir=case_base_dir,
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

    if isinstance(metadata, dict):
        golden_path_declared = (
            isinstance(golden, dict)
            and isinstance(golden.get("path"), str)
            and bool(golden["path"])
        )
        validate_reference_flags_require_golden(
            metadata,
            golden_path_declared,
            sweep_path,
            result,
            prefix=f"case '{case_label}' ",
        )
        validate_expected_rejection_has_no_golden(
            metadata,
            golden_path_declared,
            sweep_path,
            result,
            prefix=f"case '{case_label}' ",
        )

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
            stats_tensor_uids,
            length_uids,
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
    stats_tensor_uids = extract_sdpa_stats_tensor_uids(nodes)
    try:
        length_uids = extract_ragged_length_uids(nodes)
    except ValueError as error:
        result.error(template_path, f"invalid ragged seq_len binding: {error}")
        return
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
    tensor_scoped_fields: set[tuple[int, str]] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict) or not is_integer(tensor.get("uid")):
            continue
        fields: set[str] = set()
        for key, value in tensor.items():
            if key not in TEMPLATE_TENSOR_FIELDS:
                collect_placeholders(value, fields, attribute_fields)
        tensor_scoped_fields.update((tensor["uid"], f) for f in fields)

    version = sweep.get("version")
    if not is_integer(version) or version not in (1, 2):
        result.error(sweep_path, "version is required and must be 1 or 2")

    cases = sweep.get("cases")
    if not isinstance(cases, list) or not cases:
        result.error(sweep_path, "top-level cases must be a non-empty list")
        return

    bundle_dir = sweep_path.parent
    seen_case_ids: set[str] = set()

    for index, case in enumerate(cases):
        absent_uids: set[int] = set()
        if isinstance(case, dict) and ABSENT_TENSORS_KEY in case:
            case_id = case.get("id")
            case_label = (
                case_id if isinstance(case_id, str) and case_id else f"#{index}"
            )
            for problem in absent_case_problems(sweep, template, case):
                result.error(
                    sweep_path, f"case '{case_label}' absent_tensors: {problem}"
                )
            absent = case[ABSENT_TENSORS_KEY]
            if isinstance(absent, list):
                absent_uids = {uid for uid in absent if is_integer(uid)}
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
            stats_tensor_uids,
            length_uids,
            absent_uids,
            tensor_scoped_fields,
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
