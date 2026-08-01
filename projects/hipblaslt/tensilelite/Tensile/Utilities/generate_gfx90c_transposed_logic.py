# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate the transposed gfx90c Equality packages from classic Vega logic.

The classic catalogs are used only as transpose-specific solution pools.  This
utility drops source kernels and vector-width combinations which do not satisfy
the TensileLite gfx90c contract, retargets the remaining assembly kernels, and
rebuilds contiguous solution and exact-logic references.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml


LAYOUTS = ("Ailk_Bjlk", "Alik_Bljk", "Alik_Bjlk")
HALF_SUFFIXES = ("HB", "HHS_BH")
FLOAT_SUFFIXES = ("SBIc", "SBIIc")
LICENSE = (
    "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier: MIT\n\n"
)


def _compute_sha256(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _is_compatible_assembly(solution: dict) -> bool:
    if solution.get("KernelLanguage") != "Assembly":
        return False
    width = solution.get("VectorWidth", 1)
    width_a = solution.get(
        "GlobalReadVectorWidthA",
        solution.get("GlobalLoadVectorWidthA", solution.get("GlobalReadVectorWidth", 1)),
    )
    width_b = solution.get(
        "GlobalReadVectorWidthB",
        solution.get("GlobalLoadVectorWidthB", solution.get("GlobalReadVectorWidth", 1)),
    )
    return width_a in (1, width) and width_b in (1, width)


def _normalize_solution(
    source: dict, problem_type: dict, solution_index: int, old_suffix: str, suffix: str
) -> tuple[dict, dict]:
    solution = copy.deepcopy(source)
    diff_log = {}

    if "AssertMinApproxSize" in solution:
        diff_log["AssertMinApproxSize"] = {
            "source": solution.pop("AssertMinApproxSize"),
            "action": "dropped_unsupported_assertion",
        }

    read_w_a = solution.pop(
        "GlobalLoadVectorWidthA", solution.get("GlobalReadVectorWidth", 1)
    )
    read_w_b = solution.pop(
        "GlobalLoadVectorWidthB", solution.get("GlobalReadVectorWidth", 1)
    )
    if solution.get("GlobalReadVectorWidthA") != read_w_a:
        diff_log["GlobalReadVectorWidthA"] = {"source": solution.get("GlobalReadVectorWidthA"), "new": read_w_a}
        solution["GlobalReadVectorWidthA"] = read_w_a

    if solution.get("GlobalReadVectorWidthB") != read_w_b:
        diff_log["GlobalReadVectorWidthB"] = {"source": solution.get("GlobalReadVectorWidthB"), "new": read_w_b}
        solution["GlobalReadVectorWidthB"] = read_w_b

    vec_w = solution.get("VectorWidth", 1)
    solution["VectorWidthA"] = vec_w
    solution["VectorWidthB"] = vec_w

    if solution.get("ISA") != [9, 0, 12]:
        diff_log["ISA"] = {"source": solution.get("ISA"), "new": [9, 0, 12]}
        solution["ISA"] = [9, 0, 12]

    solution["EnableMatrixInstruction"] = False
    solution["MatrixInstruction"] = []

    if solution.get("ScheduleIterAlg") != 1:
        diff_log["ScheduleIterAlg"] = {"source": solution.get("ScheduleIterAlg"), "new": 1}
        solution["ScheduleIterAlg"] = 1

    solution["CustomKernelName"] = ""
    solution["SolutionIndex"] = solution_index
    solution["ProblemType"] = copy.deepcopy(problem_type)

    old_name = solution["SolutionNameMin"]
    new_name = old_name.replace(f"_{old_suffix}_", f"_{suffix}_", 1)
    if old_name != new_name:
        diff_log["SolutionNameMin"] = {"source": old_name, "new": new_name}
        solution["SolutionNameMin"] = new_name

    int_fields = (
        "DirectToLds",
        "DirectToLdsA",
        "DirectToLdsB",
        "PrefetchGlobalRead",
        "PrefetchLocalRead",
        "UseSgprForGRO",
        "VectorStore",
        "ScheduleGlobalRead",
        "ScheduleLocalWrite",
        "WaveSeparateGlobalReadA",
        "WaveSeparateGlobalReadB",
        "UnrollLoopSwapGlobalReadOrder",
        "SwapGlobalReadOrder",
    )
    for field in int_fields:
        if field in solution and isinstance(solution[field], bool):
            diff_log[field] = {"source": solution[field], "new": int(solution[field])}
            solution[field] = int(solution[field])

    return solution, diff_log


def _remap_exact_logic(
    exact_logic: list, source_solutions: list, retained_positions: list[int]
) -> tuple[list, set[int], list[dict]]:
    position_to_index = {
        source_position: index
        for index, source_position in enumerate(retained_positions)
    }
    remapped = []
    selected = set()
    dropped_entries = []
    for entry in exact_logic:
        old_position = entry[1][0]
        if old_position in position_to_index:
            new_index = position_to_index[old_position]
            updated = copy.deepcopy(entry)
            updated[1][0] = new_index
            remapped.append(updated)
            selected.add(new_index)
        else:
            dropped_entries.append({
                "exact_problem_key": copy.deepcopy(entry[0]),
                "source_solution_position": old_position,
                "reason": f"Source solution position {old_position} was not in retained compatible assembly solution set",
            })
    return remapped, selected, dropped_entries


def _generate_package(
    source_data: list,
    template_data: list,
    source_suffix: str,
    suffix: str,
    max_solutions: int | None,
    source_path: Path,
    template_path: Path,
) -> tuple[list, dict]:
    output = copy.deepcopy(template_data)
    output[0] = {"MinimumRequiredVersion": "5.0.0"}
    output[1:3] = ["gfx90c", "gfx90c"]

    problem_type = copy.deepcopy(template_data[4])
    source_problem_type = source_data[4]
    for key in (
        "IndexAssignmentsA",
        "IndexAssignmentsB",
        "IndexUnrollA",
        "IndexUnrollB",
        "TLUA",
        "TLUB",
        "TransposeA",
        "TransposeB",
    ):
        problem_type[key] = copy.deepcopy(source_problem_type[key])

    source_solutions = source_data[5]
    compatible_positions = [
        position
        for position, solution in enumerate(source_solutions)
        if _is_compatible_assembly(solution)
    ]
    if max_solutions is not None:
        frequency = Counter(entry[1][0] for entry in source_data[7])
        compatible_positions = sorted(
            compatible_positions,
            key=lambda position: (-frequency[position], position),
        )[:max_solutions]
        compatible_positions.sort()

    exact_logic, selected, dropped_entries = _remap_exact_logic(
        source_data[7], source_solutions, compatible_positions
    )

    # Exact logic is the source of reachability. Prune candidates which no
    # measured point can select.
    reachable_positions = [
        position
        for index, position in enumerate(compatible_positions)
        if index in selected
    ]
    if len(reachable_positions) != len(compatible_positions):
        compatible_positions = reachable_positions
        exact_logic, selected, dropped_entries = _remap_exact_logic(
            source_data[7], source_solutions, compatible_positions
        )

    normalized_solutions = []
    transformation_diff_logs = []
    retained_provenance = []

    for index, position in enumerate(compatible_positions):
        source_sol = source_solutions[position]
        norm_sol, diff_log = _normalize_solution(
            source_sol,
            problem_type,
            index,
            source_suffix,
            suffix,
        )
        normalized_solutions.append(norm_sol)
        transformation_diff_logs.append(diff_log)
        retained_provenance.append({
            "generated_solution_index": index,
            "source_solution_position": position,
            "source_solution_index": source_sol.get("SolutionIndex"),
            "source_solution_name": source_sol.get("SolutionNameMin"),
            "generated_solution_name": norm_sol.get("SolutionNameMin"),
        })

    output[4] = problem_type
    output[5] = normalized_solutions
    output[6] = copy.deepcopy(template_data[6])
    output[7] = exact_logic

    provenance_report = {
        "source_package": str(source_path),
        "source_package_sha256": _compute_sha256(source_path.read_bytes()),
        "template_package": str(template_path),
        "template_package_sha256": _compute_sha256(template_path.read_bytes()),
        "retained_solutions_count": len(retained_provenance),
        "retained_solutions": retained_provenance,
        "exact_entries_retained_count": len(exact_logic),
        "dropped_exact_entries_count": len(dropped_entries),
        "dropped_exact_entries": dropped_entries,
        "transformation_diff_logs": transformation_diff_logs,
    }

    return output, provenance_report


HEADER = (
    "- MinimumRequiredVersion: 5.0.0\n"
    "- gfx90c\n"
    "- gfx90c\n"
    "- [Device 6863, Device 6862, Device 687f, Device 6860, Device 6861, 'Vega 10 XTX [Radeon Vega Frontier Edition]', 'Vega [Radeon RX Vega]', Vega, Device 6864, Device 686c, 'Vega 10 [Radeon Instinct MI25 MxGPU]']\n"
    "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier: MIT\n\n"
)

def generate(repo_root: Path, provenance_root: Path | None = None) -> None:
    repo_root = repo_root.resolve()
    if provenance_root is not None:
        provenance_root = provenance_root.resolve()
        provenance_root.mkdir(parents=True, exist_ok=True)

    classic_root = (
        repo_root
        / "projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/vega10"
    )
    output_root = (
        repo_root
        / "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic"
        / "asm_full/gfx90c/Equality"
    )

    for layout in LAYOUTS:
        for suffix in HALF_SUFFIXES:
            source_path = classic_root / f"vega10_Cijk_{layout}_{suffix}.yaml"
            template_path = output_root / f"vega10_Cijk_Ailk_Bljk_{suffix}.yaml"
            source = yaml.safe_load(source_path.read_text())
            template = yaml.safe_load(template_path.read_text())
            output, provenance = _generate_package(
                source, template, suffix, suffix, None, source_path, template_path
            )
            output_path = output_root / f"vega10_Cijk_{layout}_{suffix}.yaml"
            yaml_content = HEADER + yaml.safe_dump(output[4:], sort_keys=False, width=1000)
            output_path.write_text(yaml_content)

            if provenance_root is not None:
                provenance["source_package"] = str(source_path.relative_to(repo_root))
                provenance["template_package"] = str(template_path.relative_to(repo_root))
                provenance["generated_package"] = str(output_path.relative_to(repo_root))
                provenance["generated_package_sha256"] = _compute_sha256(yaml_content)
                provenance_path = provenance_root / f"vega10_Cijk_{layout}_{suffix}.provenance.json"
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

        source_path = classic_root / f"vega10_Cijk_{layout}_SB.yaml"
        source = yaml.safe_load(source_path.read_text())
        for suffix in FLOAT_SUFFIXES:
            template_path = output_root / f"vega10_Cijk_Ailk_Bljk_{suffix}.yaml"
            template = yaml.safe_load(template_path.read_text())
            output, provenance = _generate_package(
                source, template, "SB", suffix, 5, source_path, template_path
            )
            output_path = output_root / f"vega10_Cijk_{layout}_{suffix}.yaml"
            yaml_content = HEADER + yaml.safe_dump(output[4:], sort_keys=False, width=1000)
            output_path.write_text(yaml_content)

            if provenance_root is not None:
                provenance["source_package"] = str(source_path.relative_to(repo_root))
                provenance["template_package"] = str(template_path.relative_to(repo_root))
                provenance["generated_package"] = str(output_path.relative_to(repo_root))
                provenance["generated_package_sha256"] = _compute_sha256(yaml_content)
                provenance_path = provenance_root / f"vega10_Cijk_{layout}_{suffix}.provenance.json"
                provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[5],
    )
    parser.add_argument(
        "--provenance-root",
        type=Path,
        help="Optional directory outside the source logic tree for provenance reports.",
    )
    args = parser.parse_args()
    generate(args.repo_root, args.provenance_root)


if __name__ == "__main__":
    main()
