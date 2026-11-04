# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI to regenerate ``SolutionUID`` for one solution in a logic YAML file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import LibraryIO
from .Common.SolutionIdGen import encode_solution_uid, regenerate_solution_uid


def find_solution_by_index(
    data: dict[str, Any],
    solution_index: int,
) -> dict[str, Any]:
    """Return the solution dict with the given local ``SolutionIndex``.

    Args:
        data: Parsed library logic YAML data.
        solution_index: Local ``SolutionIndex`` value to locate.

    Returns:
        Matching solution dictionary from ``data["Solutions"]``.

    Raises:
        ValueError: If ``Solutions`` is missing or no entry matches ``solution_index``.
    """
    solutions = data.get("Solutions")
    if not isinstance(solutions, list):
        raise ValueError("Logic YAML does not contain a Solutions list")

    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        if int(solution.get("SolutionIndex", -1)) == solution_index:
            return solution

    raise ValueError(
        f"No solution with SolutionIndex={solution_index} found in logic YAML"
    )


def regenerate_uid_for_solution(
    yaml_path: Path,
    solution_index: int,
    *,
    inplace: bool = False,
) -> int:
    """Regenerate ``SolutionUID`` for one solution in a logic YAML file.

    Args:
        yaml_path: Path to the logic YAML file.
        solution_index: Local ``SolutionIndex`` of the target solution.
        inplace: When True, write the updated YAML back to ``yaml_path``.

    Returns:
        Newly generated 64-bit ``SolutionUID``.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        ValueError: If the target solution cannot be found.
    """
    yaml_path = yaml_path.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Logic YAML not found: {yaml_path}")

    data = LibraryIO.readYAML(str(yaml_path))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-format logic YAML: {yaml_path}")

    solution = find_solution_by_index(data, solution_index)
    new_uid = regenerate_solution_uid(solution)
    LibraryIO.reorderSolutionsParams(data)

    if inplace:
        LibraryIO.writeYAML(str(yaml_path), data)

    return new_uid


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate SolutionUID for one solution in a logic YAML file. "
            "Uses a fresh timestamp + random draw; no full-repo scan is required."
        ),
    )
    parser.add_argument(
        "yaml_file",
        help="Path to a logic YAML file (dict format with Solutions list)",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        dest="solution_index",
        help="Local SolutionIndex of the solution to update",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write the new SolutionUID back to the YAML file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``TensileGenerateUID.py``.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 on error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        new_uid = regenerate_uid_for_solution(
            Path(args.yaml_file),
            args.solution_index,
            inplace=args.inplace,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(encode_solution_uid(new_uid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
