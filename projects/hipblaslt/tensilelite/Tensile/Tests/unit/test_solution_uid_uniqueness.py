# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verify SolutionUID uniqueness across shipped logic YAML files."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from Tensile.Common.SolutionIdGen import decode_solution_uid, encode_solution_uid

pytestmark = pytest.mark.unit

DEFAULT_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library"
    / "src"
    / "amd_detail"
    / "rocblaslt"
    / "src"
    / "Tensile"
    / "Logic"
)

STRICT_ENV = "HIPBLASLT_REQUIRE_SOLUTION_UID"


def _collect_yaml_files(logic_root: Path) -> List[Path]:
    """Return logic YAML files that contain solution definitions.

    Args:
        logic_root: Root directory containing arch/datatype logic trees.

    Returns:
        Sorted list of YAML file paths.
    """
    asm_full = logic_root / "asm_full"
    if asm_full.is_dir():
        return sorted(asm_full.rglob("*.yaml"))
    return sorted(logic_root.rglob("*.yaml"))


def test_collect_yaml_files_includes_all_yaml_names(tmp_path: Path) -> None:
    """Verify discovery does not require the ``_UserArgs.yaml`` suffix.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If any YAML file under ``asm_full`` is omitted.
    """
    asm_full = tmp_path / "asm_full"
    asm_full.mkdir()
    user_args = asm_full / "logic_UserArgs.yaml"
    conventional = asm_full / "logic.yaml"
    ignored = asm_full / "README.txt"
    user_args.touch()
    conventional.touch()
    ignored.touch()

    assert _collect_yaml_files(tmp_path) == [conventional, user_args]


def _scan_yaml_file(yaml_path: Path) -> Tuple[List[Tuple[int, str, int]], int]:
    """Extract SolutionUID entries from one logic YAML.

    Args:
        yaml_path: Path to a logic YAML file.

    Returns:
        Tuple of (entries, missing_count) where each entry is
        ``(SolutionUID, yaml_path, SolutionIndex)`` and
        ``missing_count`` counts solutions without the field.

    Raises:
        TypeError: If a UID value is not a string.
        ValueError: If a UID is malformed, zero, out of range, or an index
            value is not an integer.
    """
    entries: List[Tuple[int, str, int]] = []
    missing = 0
    current_index: int | None = None
    current_uid: int | None = None

    def flush_solution() -> None:
        """Record the current solution before scanning the next one.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        nonlocal missing, current_index, current_uid
        if current_index is None:
            return
        if current_uid is None:
            missing += 1
        else:
            entries.append((current_uid, str(yaml_path), current_index))
        current_index = None
        current_uid = None

    with yaml_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("- SolutionIndex:") or stripped.startswith("SolutionIndex:"):
                flush_solution()
                current_index = int(stripped.split(":", 1)[1].strip())
            elif current_index is not None and stripped.startswith("SolutionUID:"):
                encoded_uid = stripped.split(":", 1)[1].strip()
                current_uid = decode_solution_uid(encoded_uid)
                if current_uid == 0:
                    raise ValueError(
                        f"Stored SolutionUID must not be zero: {yaml_path} "
                        f"(SolutionIndex={current_index})"
                    )

    flush_solution()
    return entries, missing


def collect_solution_uids(
    logic_root: Path,
    *,
    process_count: int,
) -> Tuple[List[Tuple[int, str, int]], int]:
    """Collect SolutionUID values using multiple processes.

    Args:
        logic_root: Root directory containing logic YAML files.
        process_count: Number of worker processes to use.

    Returns:
        Tuple of all ``(uid, yaml_path, local_index)`` entries and total missing count.
    """
    yaml_files = _collect_yaml_files(logic_root)
    if not yaml_files:
        return [], 0

    process_count = max(1, min(process_count, len(yaml_files)))
    ctx = mp.get_context("spawn")
    all_entries: List[Tuple[int, str, int]] = []
    total_missing = 0
    with ctx.Pool(process_count) as pool:
        for entries, missing in pool.imap(_scan_yaml_file, yaml_files):
            all_entries.extend(entries)
            total_missing += missing
    return all_entries, total_missing


def find_duplicate_uids(
    entries: List[Tuple[int, str, int]],
) -> Dict[int, List[Tuple[str, int]]]:
    """Find duplicate SolutionUID values.

    Args:
        entries: Output from :func:`collect_solution_uids`.

    Returns:
        Mapping from duplicate UID to list of ``(yaml_path, local SolutionIndex)``.
    """
    seen: Dict[int, List[Tuple[str, int]]] = {}
    for uid, yaml_path, local_index in entries:
        seen.setdefault(uid, []).append((yaml_path, local_index))
    return {uid: locations for uid, locations in seen.items() if len(locations) > 1}


@pytest.fixture(name="logic_root")
def fixture_logic_root() -> Path:
    """Resolve the logic YAML root, overridable via env var.

    Returns:
        Path to ``Logic/`` under the hipBLASLt library tree.
    """
    override = os.environ.get("HIPBLASLT_LOGIC_ROOT")
    if override:
        return Path(override)
    return DEFAULT_LOGIC_ROOT


def test_solution_uid_unique_across_logic_files(logic_root: Path) -> None:
    """All present SolutionUID values must be unique repo-wide."""
    process_count = int(os.environ.get("HIPBLASLT_UID_TEST_PROCESSES", "8"))
    entries, missing = collect_solution_uids(logic_root, process_count=process_count)
    if not entries and missing == 0:
        pytest.fail(f"No solution definitions found under {logic_root}")

    duplicates = find_duplicate_uids(entries)
    if duplicates:
        lines = ["Duplicate SolutionUID values detected:"]
        for uid, locations in sorted(duplicates.items()):
            lines.append(f"  uid {encode_solution_uid(uid)}:")
            for yaml_path, local_index in locations:
                lines.append(f"    {yaml_path} (SolutionIndex={local_index})")
        pytest.fail("\n".join(lines))

    strict = os.environ.get(STRICT_ENV, "").lower() in {"1", "true", "yes"}
    if strict and missing > 0:
        pytest.fail(
            f"{missing} solution(s) under {logic_root} are missing SolutionUID "
            f"(set {STRICT_ENV}=0 to warn only during migration)"
        )
