# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare list-format and dict-format Tensile library logic files for equivalence.

Uses the same loader as ``TensileCreateLibrary`` (``LibraryIO.parseLibraryLogicData``),
so a match here means both files produce identical parsed solutions and matching
tables before any kernel codegen runs.

Full ``TensileCreateLibrary`` builds add post-kernel metadata (for example
``CUOccupancy``) that depends on the build environment; that step is optional
and not required to verify list-vs-dict format equivalence.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from Tensile import LibraryIO
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Utilities import state
from Tensile.CustomYamlLoader import load_yaml_stream
from Tensile.SolutionStructs.Naming import getKernelNameMin, getSolutionNameMin
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain


# Keys copied onto every solution during parse; exclude from per-solution diffs.
_SOLUTION_COMPARE_IGNORE = frozenset({"ProblemType"})


@dataclass
class ComparisonReport:
    """Structured result of a list-vs-dict library logic comparison.

    Attributes:
        equivalent: True when no semantic differences were found.
        differences: Human-readable diff lines (empty when equivalent).
    """

    equivalent: bool
    differences: list[str] = field(default_factory=list)


def _make_parse_context() -> tuple[Any, dict[str, Any]]:
    """Build assembler and ISA capability map for ``parseLibraryLogicData``.

    Returns:
        Tuple of (assembler, isaInfoMap).

    Raises:
        RuntimeError: If the ROCm toolchain cannot be validated.
    """
    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    isa_info_map = makeIsaInfoMap(SUPPORTED_ISA, cxx)
    return assembler, isa_info_map


def _normalize_problem_type(problem_type: ProblemType) -> dict[str, Any]:
    """Serialize a ``ProblemType`` to a JSON-comparable dict.

    Args:
        problem_type: Parsed problem type from library logic.

    Returns:
        Normalized problem-type mapping with enum values flattened.

    Raises:
        None.
    """
    pt_state = deepcopy(problem_type.state)
    for key in ("DataType", "MacDataTypeA", "MacDataTypeB", "DataTypeA", "DataTypeB",
                "DataTypeE", "DataTypeAmaxD", "DestDataType", "ComputeDataType",
                "ActivationComputeDataType", "ActivationType", "F32XdlMathOp"):
        if key in pt_state and hasattr(pt_state[key], "value"):
            pt_state[key] = pt_state[key].value
    if "BiasDataTypeList" in pt_state:
        pt_state["BiasDataTypeList"] = [
            b.value if hasattr(b, "value") else b for b in pt_state["BiasDataTypeList"]
        ]
    if "GateResidualDataTypeList" in pt_state:
        pt_state["GateResidualDataTypeList"] = [
            b.value if hasattr(b, "value") else b
            for b in pt_state["GateResidualDataTypeList"]
        ]
    for opt in ("DataTypeMetadata", "DataTypeMXSA", "DataTypeMXSB"):
        if opt in pt_state and hasattr(pt_state[opt], "value"):
            pt_state[opt] = pt_state[opt].value
    return dict(sorted(pt_state.items()))


def _normalize_exact_logic(exact_logic: list[Any] | None) -> list[tuple[tuple[Any, ...], int, float]]:
    """Normalize ``ExactLogic`` entries for stable comparison.

    Args:
        exact_logic: Size-to-solution mapping from parsed library logic.

    Returns:
        Sorted list of ``(size_tuple, solution_index, efficiency)`` records.

    Raises:
        None.
    """
    if not exact_logic:
        return []
    normalized: list[tuple[tuple[Any, ...], int, float]] = []
    for entry in exact_logic:
        size = tuple(entry[0])
        sol_index = int(entry[1][0])
        efficiency = float(entry[1][1])
        normalized.append((size, sol_index, efficiency))
    return sorted(normalized, key=lambda row: (row[0], row[1], row[2]))


def _normalize_solution_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Convert a solution parameter dict to a JSON-comparable form.

    Args:
        raw_state: Solution ``_state`` mapping from ``getAttributes()``.

    Returns:
        Normalized mapping with ignored keys removed and nested values flattened.

    Raises:
        None.
    """
    out: dict[str, Any] = {}
    for key, value in raw_state.items():
        if key in _SOLUTION_COMPARE_IGNORE:
            continue
        if key == "InternalSupportParams" and isinstance(value, dict):
            out[key] = dict(sorted(value.items()))
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        elif hasattr(value, "value"):
            out[key] = value.value
        else:
            out[key] = value
    return dict(sorted(out.items()))


def _solution_identity(solution: Any) -> dict[str, Any]:
    """Return naming metadata used by ``TensileCreateLibrary`` for a solution.

    Args:
        solution: Parsed ``Solution`` object.

    Returns:
        Dict with ``SolutionIndex``, ``SolutionNameMin``, and ``KernelNameMin``.

    Raises:
        None.
    """
    sol_state = solution.getAttributes()
    return {
        "SolutionIndex": sol_state.get("SolutionIndex"),
        "SolutionNameMin": getSolutionNameMin(sol_state, splitGSU=False),
        "KernelNameMin": getKernelNameMin(sol_state, splitGSU=False),
    }


def _parse_logic_file(
    path: str,
    assembler: Any,
    isa_info_map: dict[str, Any],
) -> LibraryIO.LibraryLogic:
    """Load and parse one library logic YAML file.

    Args:
        path: Path to list- or dict-format library logic YAML.
        assembler: ROCm assembler handle for ``parseLibraryLogicData``.
        isa_info_map: Per-ISA capability map.

    Returns:
        Parsed ``LibraryLogic`` named tuple.

    Raises:
        SystemExit: If ``parseLibraryLogicData`` rejects the file.
    """
    raw = load_yaml_stream(path, yaml.CSafeLoader)
    return LibraryIO.parseLibraryLogicData(
        raw,
        path,
        assembler,
        splitGSU=False,
        printSolutionRejectionReason=False,
        printIndexAssignmentInfo=False,
        isaInfoMap=isa_info_map,
        lazyLibraryLoading=False,
    )


def _diff_values(label: str, left: Any, right: Any, diffs: list[str]) -> None:
    """Append a diff line when *left* and *right* differ.

    Args:
        label: Field name for the report.
        left: Value from the list-format parse.
        right: Value from the dict-format parse.
        diffs: Mutable list collecting diff messages.

    Returns:
        None.

    Raises:
        None.
    """
    if left != right:
        diffs.append(f"{label}: list={left!r} dict={right!r}")


def _compare_solution_sets(
    list_solutions: Iterable[Any],
    dict_solutions: Iterable[Any],
    diffs: list[str],
    *,
    full_state: bool,
) -> None:
    """Compare parsed solutions from list and dict inputs.

    Args:
        list_solutions: Solutions parsed from the list-format file.
        dict_solutions: Solutions parsed from the dict-format file.
        diffs: Mutable list collecting diff messages.
        full_state: When True, compare full normalized parameter dicts.

    Returns:
        None.

    Raises:
        None.
    """
    list_by_index = {s["SolutionIndex"]: s for s in list_solutions}
    dict_by_index = {s["SolutionIndex"]: s for s in dict_solutions}

    list_indices = set(list_by_index)
    dict_indices = set(dict_by_index)
    if list_indices != dict_indices:
        only_list = sorted(list_indices - dict_indices)
        only_dict = sorted(dict_indices - list_indices)
        if only_list:
            diffs.append(f"solution indices only in list file: {only_list}")
        if only_dict:
            diffs.append(f"solution indices only in dict file: {only_dict}")

    for index in sorted(list_indices & dict_indices):
        left = list_by_index[index]
        right = dict_by_index[index]
        left_id = _solution_identity(left)
        right_id = _solution_identity(right)
        for key in left_id:
            _diff_values(f"solution[{index}].{key}", left_id[key], right_id[key], diffs)

        if not full_state:
            continue

        left_state = _normalize_solution_state(left.getAttributes())
        right_state = _normalize_solution_state(right.getAttributes())
        if left_state == right_state:
            continue

        all_keys = sorted(set(left_state) | set(right_state))
        param_diffs = 0
        for key in all_keys:
            if left_state.get(key) != right_state.get(key):
                param_diffs += 1
                if param_diffs <= 5:
                    diffs.append(
                        f"solution[{index}].{key}: "
                        f"list={left_state.get(key)!r} dict={right_state.get(key)!r}"
                    )
        if param_diffs > 5:
            diffs.append(
                f"solution[{index}]: {param_diffs} parameter differences total "
                f"(showing first 5 above)"
            )


def compare_library_logic_files(
    list_path: str,
    dict_path: str,
    *,
    full_state: bool = True,
    assembler: Any | None = None,
    isa_info_map: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Compare list-format and dict-format library logic for Tensile equivalence.

    Equivalence means ``parseLibraryLogicData`` produces the same problem type,
    exact-logic table, solution naming, and (optionally) full per-solution params,
    plus the same serialized master library structure. This is the input
    ``TensileCreateLibrary`` consumes before kernel builds.

    Comparing solution params alone is necessary but not sufficient: also verify
    ``ExactLogic`` (size-to-kernel mapping) and ``ProblemType``, otherwise runtime
    kernel selection can differ even when individual solutions match.

    Args:
        list_path: Path to legacy list-format YAML.
        dict_path: Path to dict-format YAML (typically converted from *list_path*).
        full_state: When True, diff every normalized solution parameter.
        assembler: Optional pre-built assembler; created when omitted.
        isa_info_map: Optional ISA map; created when omitted.

    Returns:
        ``ComparisonReport`` with ``equivalent`` and human-readable ``differences``.

    Raises:
        RuntimeError: If the ROCm toolchain cannot be validated.
        SystemExit: If either file fails to parse.
    """
    if assembler is None or isa_info_map is None:
        built_assembler, built_map = _make_parse_context()
        assembler = assembler or built_assembler
        isa_info_map = isa_info_map or built_map

    list_logic = _parse_logic_file(list_path, assembler, isa_info_map)
    dict_logic = _parse_logic_file(dict_path, assembler, isa_info_map)

    diffs: list[str] = []

    _diff_values("schedule", list_logic.schedule, dict_logic.schedule, diffs)
    _diff_values("architecture", list_logic.architecture, dict_logic.architecture, diffs)

    list_pt = _normalize_problem_type(list_logic.problemType)
    dict_pt = _normalize_problem_type(dict_logic.problemType)
    if list_pt != dict_pt:
        pt_keys = sorted(set(list_pt) | set(dict_pt))
        pt_diffs = 0
        for key in pt_keys:
            if list_pt.get(key) != dict_pt.get(key):
                pt_diffs += 1
                if pt_diffs <= 5:
                    diffs.append(
                        f"ProblemType.{key}: list={list_pt.get(key)!r} "
                        f"dict={dict_pt.get(key)!r}"
                    )
        if pt_diffs > 5:
            diffs.append(f"ProblemType: {pt_diffs} field differences total")

    list_exact = _normalize_exact_logic(list_logic.exactLogic)
    dict_exact = _normalize_exact_logic(dict_logic.exactLogic)
    if list_exact != dict_exact:
        diffs.append(
            f"ExactLogic differs: {len(list_exact)} list entries vs "
            f"{len(dict_exact)} dict entries"
        )

    _diff_values(
        "solution_count",
        len(list_logic.solutions),
        len(dict_logic.solutions),
        diffs,
    )
    _compare_solution_sets(
        list_logic.solutions,
        dict_logic.solutions,
        diffs,
        full_state=full_state,
    )

    list_library_state = state(list_logic.library)
    dict_library_state = state(dict_logic.library)
    if list_library_state != dict_library_state:
        diffs.append("MasterSolutionLibrary serialized state differs")

    return ComparisonReport(equivalent=not diffs, differences=diffs)


def _format_report(
    list_path: str,
    dict_path: str,
    report: ComparisonReport,
    *,
    json_output: bool,
) -> str:
    """Format a comparison report for stdout.

    Args:
        list_path: List-format input path.
        dict_path: Dict-format input path.
        report: Comparison outcome.
        json_output: When True, emit JSON instead of plain text.

    Returns:
        Rendered report string.

    Raises:
        None.
    """
    if json_output:
        payload = {
            "equivalent": report.equivalent,
            "list_path": list_path,
            "dict_path": dict_path,
            "differences": report.differences,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    lines = [
        f"List: {list_path}",
        f"Dict: {dict_path}",
        f"Equivalent: {report.equivalent}",
    ]
    if report.differences:
        lines.append("Differences:")
        lines.extend(f"  - {line}" for line in report.differences)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: compare list and dict library logic YAML files.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code 0 when equivalent, 1 when differences exist, 2 on error.

    Raises:
        None.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare list-format and dict-format library logic YAML using the "
            "same parser as TensileCreateLibrary."
        ),
    )
    parser.add_argument("list_path", help="Legacy list-format library logic YAML")
    parser.add_argument("dict_path", help="Dict-format library logic YAML")
    parser.add_argument(
        "--naming-only",
        action="store_true",
        help="Compare only SolutionIndex / SolutionNameMin / KernelNameMin per solution",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args(argv)

    list_path = str(Path(args.list_path).resolve())
    dict_path = str(Path(args.dict_path).resolve())

    try:
        report = compare_library_logic_files(
            list_path,
            dict_path,
            full_state=not args.naming_only,
        )
    except (OSError, RuntimeError, SystemExit, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    print(_format_report(list_path, dict_path, report, json_output=args.json))
    return 0 if report.equivalent else 1


if __name__ == "__main__":
    sys.exit(main())
