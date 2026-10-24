#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Extract the cuDNN backend C-API surface used by cudnn-frontend v0.x and PyTorch.

Writes the committed fixtures next to this script (or to --output):

- cudnn_v8_identifiers.inc: every C identifier the shim must declare, classified.
- cudnn_v8_enum_values.inc: the shim-assigned enumerator values. Values are
  alphabetical within each enum on first assignment and append-only afterwards:
  an existing value never changes, a new enumerator takes the next free value.

Run manually; this is not part of the build. Nothing is downloaded: --pytorch
points at a checkout, or at a directory holding the pinned files at their
repository paths.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS_FILE = HERE / "pins.json"
IDENTIFIERS_FILE = "cudnn_v8_identifiers.inc"
ENUM_VALUES_FILE = "cudnn_v8_enum_values.inc"

# v0.x code only names CUDNN_KNOB_TYPE_COUNTS; the enumerators themselves come
# from the v1 knob mapping, which the shim's KnobType_t mirrors.
EXTRA_FRONTEND_INPUTS = ("include/cudnn_frontend/knobs.h",)

ENUMERATOR = "ENUMERATOR"
TYPE = "TYPE"
FUNCTION = "FUNCTION"
MACRO = "MACRO"
ALREADY_DECLARED = "ALREADY_DECLARED"
KIND_ORDER = (ENUMERATOR, TYPE, FUNCTION, MACRO, ALREADY_DECLARED)

# Names the scan finds that the shim must not declare. Every entry must still
# be found by the scan, so the table cannot go stale silently.
EXCLUDED = {
    "cudnnException": "cudnn-frontend class, ported with the frontend",
    "cudnnFindPlan": "cudnn-frontend function template",
    "cudnnFindPlanAndCache": "cudnn-frontend function template",
    "cudnnGetPlan": "cudnn-frontend function",
    "cudnnReorderFilterAndBiasInt8x32": "cudnn-frontend helper over the v7 reorder API",
    "CUDNN_FIND_SAMPLE_ONCE": "cudnn-frontend scoped enumerator",
    "CUDNN_FIND_SAMPLE_TILL_STABLE": "cudnn-frontend scoped enumerator",
    "CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE": "cudnn-frontend scoped enumerator",
    "CUDNN_FRONTEND_SKIP_JSON_LIB": (
        "cudnn-frontend configuration macro; defined by the shim when "
        "HIPDNN_FRONTEND_SKIP_JSON_LIB is"
    ),
    "cudnnFilterDescriptor_t": "v7 filter API, out of scope",
    "cudnnCreateFilterDescriptor": "v7 filter API, out of scope",
    "cudnnSetFilterNdDescriptor": "v7 filter API, out of scope",
    "cudnnDestroyFilterDescriptor": "v7 filter API, out of scope",
    "cudnnReorderFilterAndBias": "v7 reorder API, out of scope",
    "cudnnReorderType_t": "v7 reorder API, out of scope",
    "CUDNN_DEFAULT_REORDER": "v7 reorder API, out of scope",
    "cudnnBackendPopulateCudaGraph": "CUDA-graph API, out of scope",
    "cudnnBackendUpdateCudaGraph": "CUDA-graph API, out of scope",
    "cudnnCausalConv1dForward": "causal-conv1d C API, out of scope",
    "cudnnCausalConv1dBackward": "causal-conv1d C API, out of scope",
    "cudnnCausalConv1dActivation_t": "causal-conv1d C API, out of scope",
    "cudnnGetExecutionPlanWorkspaceSize": "v1-only entry point, out of scope",
    "cudnnBackendCreateDescriptor": "raw descriptor C API removed; only cudnnBackendExecute stays",
    "cudnnBackendDestroyDescriptor": "raw descriptor C API removed; only cudnnBackendExecute stays",
    "cudnnBackendSetAttribute": "raw descriptor C API removed; only cudnnBackendExecute stays",
    "cudnnBackendGetAttribute": "raw descriptor C API removed; only cudnnBackendExecute stays",
    "cudnnBackendFinalize": "raw descriptor C API removed; only cudnnBackendExecute stays",
}

# Whole enums the scan finds that the shim no longer declares: no shim code names
# them (the raw descriptor C API is gone) or only type converters did. Their
# enumerators are dropped with them; their pinned values are retired.
EXCLUDED_ENUMS = {
    "cudnnBackendAttributeName_t": "raw descriptor C API removed",
    "cudnnBackendAttributeType_t": "raw descriptor C API removed",
    "cudnnMoeGroupedMatmulMode_t": "named only by type converters",
}

# The v9 shim surface: emitted under ALREADY_DECLARED so the test still checks
# that each name exists, but never redeclared.
ALREADY_DECLARED_NAMES = {
    "CUDNN_VERSION": MACRO,
    "CUDNN_MAJOR": MACRO,
    "CUDNN_MINOR": MACRO,
    "CUDNN_STATUS_SUCCESS": ENUMERATOR,
    "CUDNN_STATUS_BAD_PARAM": ENUMERATOR,
    "CUDNN_STATUS_INVALID_VALUE": ENUMERATOR,
    "CUDNN_STATUS_NOT_SUPPORTED": ENUMERATOR,
    "CUDNN_STATUS_ARCH_MISMATCH": ENUMERATOR,
    "CUDNN_STATUS_VERSION_MISMATCH": ENUMERATOR,
    "cudnnStatus_t": TYPE,
    "cudnnHandle_t": TYPE,
    "cudnnCreate": FUNCTION,
    "cudnnDestroy": FUNCTION,
    "cudnnSetStream": FUNCTION,
    "cudnnGetStream": FUNCTION,
    "cudnnGetErrorString": FUNCTION,
    "cudnnGetVersion": FUNCTION,
}

# Ordered: the first matching pattern classifies a name. A name that matches
# nothing is an error, so every new upstream name forces a decision here.
CLASSIFIERS = (
    (r"CUDNN_ATTR_\w+", ENUMERATOR, "cudnnBackendAttributeName_t"),
    (r"CUDNN_TYPE_\w+", ENUMERATOR, "cudnnBackendAttributeType_t"),
    (r"CUDNN_BACKEND_\w+_DESCRIPTOR", ENUMERATOR, "cudnnBackendDescriptorType_t"),
    (r"CUDNN_BEHAVIOR_NOTE_\w+", ENUMERATOR, "cudnnBackendBehaviorNote_t"),
    (r"CUDNN_BN_FINALIZE_STATISTICS_\w+", ENUMERATOR, "cudnnBnFinalizeStatsMode_t"),
    (
        r"CUDNN_CONVOLUTION|CUDNN_CROSS_CORRELATION",
        ENUMERATOR,
        "cudnnConvolutionMode_t",
    ),
    (r"CUDNN_DATA_\w+", ENUMERATOR, "cudnnDataType_t"),
    (r"CUDNN_GENSTATS_\w+", ENUMERATOR, "cudnnGenStatsMode_t"),
    (r"CUDNN_HEUR_MODES?_\w+", ENUMERATOR, "cudnnBackendHeurMode_t"),
    (r"CUDNN_KNOB_TYPE_\w+", ENUMERATOR, "cudnnBackendKnobType_t"),
    (
        r"CUDNN_MOE_GROUPED_MATMUL_MODE_\w+",
        ENUMERATOR,
        "cudnnMoeGroupedMatmulMode_t",
    ),
    (r"CUDNN_NORM_FWD_\w+", ENUMERATOR, "cudnnBackendNormFwdPhase_t"),
    (r"CUDNN_\w+_NORM", ENUMERATOR, "cudnnBackendNormMode_t"),
    (r"CUDNN_NUMERICAL_NOTE_\w+", ENUMERATOR, "cudnnBackendNumericalNote_t"),
    (r"CUDNN_(?:ZERO|NEG_INF|EDGE_VAL)_PAD", ENUMERATOR, "cudnnPaddingMode_t"),
    (r"CUDNN_POINTWISE_\w+", ENUMERATOR, "cudnnPointwiseMode_t"),
    (r"CUDNN_(?:NOT_)?PROPAGATE_NAN", ENUMERATOR, "cudnnNanPropagation_t"),
    (r"CUDNN_REDUCE_TENSOR_\w+", ENUMERATOR, "cudnnReduceTensorOp_t"),
    (r"CUDNN_RESAMPLE_\w+", ENUMERATOR, "cudnnResampleMode_t"),
    (r"CUDNN_RESHAPE_\w+", ENUMERATOR, "cudnnBackendReshapeMode_t"),
    (r"CUDNN_RNG_DISTRIBUTION_\w+", ENUMERATOR, "cudnnRngDistribution_t"),
    (
        r"CUDNN_TENSOR_REORDERING_\w+",
        ENUMERATOR,
        "cudnnBackendTensorReordering_t",
    ),
    (r"CUDNN_TENSOR_N\w+", ENUMERATOR, "cudnnTensorFormat_t"),
    (r"CUDNN_DIM_MAX", MACRO, None),
    (r"cudnn\w+_t", TYPE, None),
    (r"cudnnBackend[A-Za-z]+|cudnnGetLastErrorString", FUNCTION, None),
)

# TYPE names that are not enums; c_types.h declares them by hand.
NON_ENUM_TYPES = ("cudnnBackendDescriptor_t", "cudnnFraction_t")

# Initial value order where the alphabetical rule does not apply. Upstream
# Tensor.h sets the FE TensorReordering_t bytes as CUDNN_TYPE_TENSOR_REORDERING_MODE,
# so the C values must equal the shim's TensorReordering_t values.
SEED_ORDER = {
    "cudnnBackendTensorReordering_t": (
        "CUDNN_TENSOR_REORDERING_NONE",
        "CUDNN_TENSOR_REORDERING_INT8x32",
        "CUDNN_TENSOR_REORDERING_F16x16",
        "CUDNN_TENSOR_REORDERING_F8_128x4",
    ),
}

# Sentinels that upstream uses to size arrays; each must equal last + 1. Not a
# pattern: real enumerators such as CUDNN_KNOB_TYPE_CTA_COUNT also end in COUNT.
COUNT_SENTINELS = {
    "CUDNN_BEHAVIOR_NOTE_TYPE_COUNT",
    "CUDNN_HEUR_MODES_COUNT",
    "CUDNN_KNOB_TYPE_COUNTS",
    "CUDNN_NUMERICAL_NOTE_TYPE_COUNT",
}
TOKEN_PATTERN = re.compile(r"\b(CUDNN_[A-Za-z0-9_]+|cudnn[A-Z][A-Za-z0-9_]*)\b")
DEFINE_PATTERN = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
RAW_STRING_PATTERN = re.compile(r'(?:u8|u|U|L)?R"([^()\\\s]{0,16})\(')


class ExtractionError(Exception):
    pass


@dataclass(frozen=True)
class Identifier:
    kind: str
    name: str
    enum: str | None = None
    declared_kind: str | None = None


def strip_comments_and_literals(text: str) -> str:
    """Blank out comments and string/char/raw-string literals, keeping line breaks."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = i
            while True:
                j = text.find("\n", j)
                if j < 0:
                    j = n
                    break
                if text[j - 1] != "\\":
                    break
                j += 1
            out.append(" " + "\n" * text.count("\n", i, j))
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(" " + "\n" * text.count("\n", i, j))
            i = j
            continue
        if c in "uULR" and (i == 0 or not _is_identifier_char(text[i - 1])):
            raw = RAW_STRING_PATTERN.match(text, i)
            if raw:
                terminator = ")" + raw.group(1) + '"'
                j = text.find(terminator, raw.end())
                j = n if j < 0 else j + len(terminator)
                out.append('""' + "\n" * text.count("\n", i, j))
                i = j
                continue
        if c == '"' or (c == "'" and not _is_digit_separator(text, i)):
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(c + c)
            i = j + 1 if j < n and text[j] == c else j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _is_identifier_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _is_digit_separator(text: str, i: int) -> bool:
    k = i - 1
    while k >= 0 and (_is_identifier_char(text[k]) or text[k] in ".'"):
        k -= 1
    return k + 1 < i and text[k + 1].isdigit()


def load_pins() -> dict:
    return json.loads(PINS_FILE.read_text(encoding="utf-8"))


def check_frontend_version(frontend: Path, pins: dict) -> None:
    version_header = frontend / "include" / "cudnn_frontend_version.h"
    if not version_header.is_file():
        raise ExtractionError(f"{version_header} not found")
    text = version_header.read_text(encoding="utf-8")
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define\s+CUDNN_FRONTEND_{part}_VERSION\s+(\d+)", text)
        if not match:
            raise ExtractionError(f"CUDNN_FRONTEND_{part}_VERSION not found")
        parts.append(int(match.group(1)))
    version = parts[0] * 10000 + parts[1] * 100 + parts[2]
    expected = pins["cudnn_frontend"]["frontend_version"]
    if version != expected:
        raise ExtractionError(
            f"cudnn-frontend checkout is {version}, pins.json requires {expected}"
        )


def input_files(frontend: Path, pytorch: Path, pins: dict) -> list[Path]:
    include = frontend / "include"
    legacy_samples = frontend / "samples" / "legacy_samples"
    files = [include / "cudnn_frontend.h"]
    files += sorted(include.glob("cudnn_frontend_*.h"))
    files.append(include / "cudnn_backend_base.h")
    files += sorted(p for p in legacy_samples.rglob("*") if p.suffix in (".h", ".cpp"))
    files += [frontend / p for p in EXTRA_FRONTEND_INPUTS]
    files += [pytorch / p for p in pins["pytorch"]["files"]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise ExtractionError("missing inputs:\n  " + "\n  ".join(missing))
    return files


@dataclass
class Scan:
    tokens: set[str]
    defines: set[str]


def scan(files: list[Path]) -> Scan:
    result = Scan(set(), set())
    for path in files:
        text = strip_comments_and_literals(
            path.read_text(encoding="utf-8", errors="replace")
        )
        result.defines.update(DEFINE_PATTERN.findall(text))
        result.tokens.update(TOKEN_PATTERN.findall(text))
    return result


def classify(scanned: Scan) -> list[Identifier]:
    names = scanned.tokens - scanned.defines
    stale = sorted(
        (
            set(EXCLUDED)
            | set(EXCLUDED_ENUMS)
            | set(ALREADY_DECLARED_NAMES)
            | COUNT_SENTINELS
        )
        - names
    )
    if stale:
        raise ExtractionError(
            "EXCLUDED/EXCLUDED_ENUMS/ALREADY_DECLARED/COUNT_SENTINELS entries"
            " no longer found: " + ", ".join(stale)
        )
    identifiers = []
    unclassified = []
    for name in sorted(names - set(EXCLUDED)):
        if name in ALREADY_DECLARED_NAMES:
            identifiers.append(
                Identifier(
                    ALREADY_DECLARED,
                    name,
                    declared_kind=ALREADY_DECLARED_NAMES[name],
                )
            )
            continue
        for pattern, kind, enum in CLASSIFIERS:
            if re.fullmatch(pattern, name):
                identifiers.append(Identifier(kind, name, enum))
                break
        else:
            unclassified.append(name)
    if unclassified:
        raise ExtractionError("unclassified names: " + ", ".join(unclassified))
    identifiers = [
        i
        for i in identifiers
        if i.enum not in EXCLUDED_ENUMS and i.name not in EXCLUDED_ENUMS
    ]

    types = {i.name for i in identifiers if i.kind == TYPE}
    enums = {i.enum for i in identifiers if i.kind == ENUMERATOR}
    problems = [f"enum {e} is never named" for e in sorted(enums - types)]
    problems += [
        f"type {t} has no enumerators and is not a known non-enum type"
        for t in sorted(types - enums - set(NON_ENUM_TYPES))
    ]
    for enum in sorted(enums):
        members = [i.name for i in identifiers if i.enum == enum]
        counts = [m for m in members if m in COUNT_SENTINELS]
        if len(counts) > 1 or (counts and len(counts) == len(members)):
            problems.append(f"enum {enum} has an unusable COUNT shape: {members}")
    if problems:
        raise ExtractionError("\n".join(problems))
    return identifiers


ENUM_VALUE_LINE = re.compile(
    r"^HIPDNN_CUDNN_SHIM_ENUM_VALUE\((\w+), (\w+), (\d+)\)$", re.MULTILINE
)


def parse_enum_values(text: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for enum, name, value in ENUM_VALUE_LINE.findall(text):
        members = table.setdefault(enum, {})
        if name in members:
            raise ExtractionError(f"{name} pinned twice")
        members[name] = int(value)
    return table


def assign_enum_values(
    identifiers: list[Identifier], pinned: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Keep every pinned value; give new enumerators the next free values.

    Pinned enumerators that upstream stopped using keep their values, so a
    retired name never frees its number for reuse.
    """
    members_by_enum: dict[str, set[str]] = {}
    for identifier in identifiers:
        if identifier.kind == ENUMERATOR:
            members_by_enum.setdefault(identifier.enum, set()).add(identifier.name)
    for enum, members in pinned.items():
        members_by_enum.setdefault(enum, set()).update(members)

    table = {}
    for enum in sorted(members_by_enum):
        members = members_by_enum[enum]
        counts = sorted(m for m in members if m in COUNT_SENTINELS)
        real = members - set(counts)
        previous = {
            name: value
            for name, value in pinned.get(enum, {}).items()
            if name not in COUNT_SENTINELS
        }
        if sorted(previous.values()) != list(range(len(previous))):
            raise ExtractionError(f"pinned values of {enum} are not 0..N-1")
        seed = SEED_ORDER.get(enum, ())
        new = [m for m in seed if m in real and m not in previous]
        new += sorted(real - set(previous) - set(new))
        values = dict(previous)
        for name in new:
            values[name] = len(values)
        for name in counts:
            values[name] = len(previous) + len(new)
        table[enum] = values
    return table


def header(pins: dict, *extra: str) -> str:
    frontend = pins["cudnn_frontend"]
    pytorch = pins["pytorch"]
    lines = [
        "// Copyright © Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier:  MIT",
        "//",
        "// Generated by extract_cudnn_v8_identifiers.py; rerun it instead of editing.",
        f"// cudnn-frontend {frontend['tag']} ({frontend['frontend_version']}),"
        f" PyTorch {pytorch['tag']} ({pytorch['commit']}).",
    ]
    lines += [f"// {line}" if line else "//" for line in extra]
    return "\n".join(lines) + "\n\n"


def render_identifiers(pins: dict, identifiers: list[Identifier]) -> str:
    def key(identifier: Identifier):
        return (
            KIND_ORDER.index(identifier.kind),
            identifier.enum or "",
            identifier.name,
        )

    body = []
    for identifier in sorted(identifiers, key=key):
        if identifier.kind == ENUMERATOR:
            body.append(
                f"HIPDNN_CUDNN_SHIM_ENUMERATOR({identifier.enum}, {identifier.name})"
            )
        elif identifier.kind == ALREADY_DECLARED:
            body.append(
                "HIPDNN_CUDNN_SHIM_ALREADY_DECLARED("
                f"{identifier.declared_kind}, {identifier.name})"
            )
        else:
            body.append(f"HIPDNN_CUDNN_SHIM_{identifier.kind}({identifier.name})")
    return header(pins) + "\n".join(body) + "\n"


def render_enum_values(pins: dict, table: dict[str, dict[str, int]]) -> str:
    body = []
    for enum, values in table.items():
        for name, value in sorted(values.items(), key=lambda item: (item[1], item[0])):
            body.append(f"HIPDNN_CUDNN_SHIM_ENUM_VALUE({enum}, {name}, {value})")
    text = header(
        pins,
        "Append-only: the script keeps every value listed here and gives new",
        "enumerators the next free value. *_COUNT(S) is always last + 1.",
    )
    return text + "\n".join(body) + "\n"


C_TYPES_PROLOGUE = """\
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Declares identifiers matching the NVIDIA cuDNN C API, for source compatibility.

/**
 * @file c_types.h
 * @brief cuDNN backend C-API types for the hipDNN cuDNN v8 compatibility shim.
 *
 * Enumerator values are shim-assigned, not NVIDIA's, and append-only; they are
 * pinned by tests/data/cudnn_v8/cudnn_v8_enum_values.inc.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hipdnn_backend.h>
#include <hipdnn_compatibility/cudnn/cudnn_status.h>

/// @brief cuDNN handle, aliased directly to the hipDNN handle type.
///
/// `hipdnnHandle_t` is the global C typedef from `<hipdnn_backend.h>`, not a
/// member of namespace `hipdnn_frontend`.
using cudnnHandle_t = ::hipdnnHandle_t;

static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type");

// Upstream sizes fixed arrays with CUDNN_DIM_MAX + 1.
#define CUDNN_DIM_MAX 8 // NOLINT(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)

// NOLINTBEGIN(readability-identifier-naming)

/// @brief Backend descriptor, aliased to the hipDNN descriptor type so plans and variant packs
/// pass straight to hipDNN.
using cudnnBackendDescriptor_t = ::hipdnnBackendDescriptor_t;

/// @brief Rational number attribute value. Mirrors NVIDIA `cudnnFraction_t`.
typedef struct
{
    std::int64_t numerator;
    std::int64_t denominator;
} cudnnFraction_t;
"""


def render_c_types(table: dict[str, dict[str, int]]) -> str:
    parts = [C_TYPES_PROLOGUE]
    for enum, values in table.items():
        ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
        lines = [
            "",
            f"/// @brief Mirrors NVIDIA `{enum}`.",
            "typedef enum",
            "{",
        ]
        lines += [f"    {name} = {value}," for name, value in ordered]
        lines += [
            f"}} {enum};",
            f'static_assert(sizeof({enum}) == 4, "{enum} must be 4 bytes");',
        ]
        counts = [n for n, _ in ordered if n in COUNT_SENTINELS]
        real = [n for n, _ in ordered if n not in COUNT_SENTINELS]
        for count in counts:
            lines.append(
                f'static_assert({count} == {real[-1]} + 1, "{count} must be last + 1");'
            )
        parts.append("\n".join(lines) + "\n")
    parts.append("\n// NOLINTEND(readability-identifier-naming)\n")
    return "".join(parts)


def run(arguments: argparse.Namespace) -> int:
    pins = load_pins()
    frontend = Path(arguments.cudnn_frontend)
    check_frontend_version(frontend, pins)
    scanned = scan(input_files(frontend, Path(arguments.pytorch), pins))
    identifiers = classify(scanned)

    output = Path(arguments.output)
    values_path = output / ENUM_VALUES_FILE
    pinned = (
        parse_enum_values(values_path.read_text(encoding="utf-8"))
        if values_path.is_file()
        else {}
    )
    pinned = {
        enum: members for enum, members in pinned.items() if enum not in EXCLUDED_ENUMS
    }
    table = assign_enum_values(identifiers, pinned)
    for enum, members in pinned.items():
        for name, value in members.items():
            if name not in COUNT_SENTINELS and table[enum][name] != value:
                raise ExtractionError(f"{name} would change from {value}")

    outputs = {
        IDENTIFIERS_FILE: render_identifiers(pins, identifiers),
        ENUM_VALUES_FILE: render_enum_values(pins, table),
    }
    if arguments.check:
        drift = []
        for name, text in outputs.items():
            path = output / name
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                drift.append(name)
        if drift:
            print("drift: " + ", ".join(drift), file=sys.stderr)
            return 1
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name, text in outputs.items():
            (output / name).write_text(text, encoding="utf-8", newline="\n")
    if arguments.emit_c_types:
        Path(arguments.emit_c_types).write_text(
            render_c_types(table), encoding="utf-8", newline="\n"
        )

    counts = {kind: 0 for kind in KIND_ORDER}
    for identifier in identifiers:
        counts[identifier.kind] += 1
    summary = ", ".join(f"{kind}={count}" for kind, count in counts.items())
    print(f"{summary}, EXCLUDED={len(EXCLUDED)}, EXCLUDED_ENUMS={len(EXCLUDED_ENUMS)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cudnn-frontend", required=True, help="cudnn-frontend checkout"
    )
    parser.add_argument(
        "--pytorch",
        required=True,
        help="PyTorch checkout, or a directory holding the pinned files",
    )
    parser.add_argument("--output", default=str(HERE), help="fixture directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and exit 1 if the committed fixtures differ",
    )
    parser.add_argument(
        "--emit-c-types",
        metavar="PATH",
        help="also write a c_types.h draft from the enum value table",
    )
    try:
        return run(parser.parse_args())
    except ExtractionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
