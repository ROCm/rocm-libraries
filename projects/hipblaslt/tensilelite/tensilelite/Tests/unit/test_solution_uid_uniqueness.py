# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Verify SolutionUID uniqueness across shipped logic YAML files."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import pytest


def _load_solution_id_gen() -> ModuleType:
    """Load SolutionIdGen without importing ``tensilelite.Common``.

    The ``library-uniqueness`` tox env skips installing tensilelite and rocisa.
    ``from tensilelite.Common.SolutionIdGen import ...`` still executes
    ``tensilelite.Common.__init__``, which imports rocisa.

    Args:
        None.

    Returns:
        The loaded ``SolutionIdGen`` module.

    Raises:
        FileNotFoundError: If ``SolutionIdGen.py`` is missing.
        ImportError: If the module spec or loader cannot be created.
    """
    module_path = Path(__file__).resolve().parents[2] / "Common" / "SolutionIdGen.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"SolutionIdGen.py not found: {module_path}")
    spec = importlib.util.spec_from_file_location("tensile_solution_id_gen", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SolutionIdGen from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_solution_id_gen = _load_solution_id_gen()
decode_solution_uid = _solution_id_gen.decode_solution_uid
encode_solution_uid = _solution_id_gen.encode_solution_uid

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

# The shipped logic corpus lives in the hipBLASLt product tree, not in
# tensilelite, so it is absent whenever this file is run from installed
# tensilelite test artifacts. The enforcement point is the library-uniqueness
# component job, which checks out the source tree; this copy is a
# convenience/local-dev signal otherwise -- hence ``skipif`` (an unmet
# precondition), not ``xfail`` (an expected failure).
_LOGIC_ROOT = Path(os.environ.get("HIPBLASLT_LOGIC_ROOT") or DEFAULT_LOGIC_ROOT)

_needs_logic_dir = pytest.mark.skipif(
    not _LOGIC_ROOT.is_dir(),
    reason="Logic files not found: https://github.com/ROCm/rocm-libraries/issues/7481",
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


_SOLUTION_KEYS = ("SolutionIndex", "SolutionUID")


def _strip_comment(line: str) -> str:
    """Remove an unquoted YAML comment from one line.

    A ``#`` starts a comment at the start of the line or after whitespace,
    outside single or double quotes.

    Args:
        line: One line of a logic YAML file.

    Returns:
        The line without its comment.
    """
    if "#" not in line:
        return line
    quote = ""
    for position, character in enumerate(line):
        if quote:
            if character == quote:
                quote = ""
        elif character in "'\"":
            quote = character
        elif character == "#" and (position == 0 or line[position - 1] in " \t"):
            return line[:position]
    return line


class _SolutionItem:
    """``SolutionIndex`` and ``SolutionUID`` text seen in one sequence item."""

    __slots__ = ("key_column", "index", "uid")

    def __init__(self, key_column: int) -> None:
        """Start an item whose mapping keys begin at ``key_column``.

        Args:
            key_column: Column of the item's first mapping key.

        Returns:
            None.
        """
        self.key_column = key_column
        self.index: Optional[str] = None
        self.uid: Optional[str] = None

    def record(self, key: str, value: str, yaml_path: Path) -> None:
        """Store one ``SolutionIndex`` or ``SolutionUID`` value.

        Args:
            key: ``SolutionIndex`` or ``SolutionUID``.
            value: The value text after the colon.
            yaml_path: File being scanned, for error messages.

        Returns:
            None.

        Raises:
            ValueError: If the item already has a value for ``key``.
        """
        if key == "SolutionIndex":
            if self.index is not None:
                raise ValueError(f"{yaml_path}: solution has two SolutionIndex keys")
            self.index = value
        else:
            if self.uid is not None:
                raise ValueError(
                    f"{yaml_path}: solution has two SolutionUID keys "
                    f"(SolutionIndex={self.index})"
                )
            self.uid = value


def _scan_yaml_file(yaml_path: Path) -> Tuple[List[Tuple[int, str, int]], int]:
    """Extract SolutionUID entries from one logic YAML.

    The file is read once, line by line, without a YAML parser. Unquoted
    comments are removed. ``SolutionIndex`` and ``SolutionUID`` are collected
    per sequence item, in either order, from block keys at the item's key
    column. This covers dict-format files, where solutions are top-level items
    under ``Solutions:``, and list-format files, where solutions are nested one
    level deeper. Logic files do not write solutions as flow mappings.

    Args:
        yaml_path: Path to a logic YAML file.

    Returns:
        Tuple of (entries, missing_count) where each entry is
        ``(SolutionUID, yaml_path, SolutionIndex)`` and
        ``missing_count`` counts solutions without the field.

    Raises:
        ValueError: If a UID is malformed, zero, or out of range, an index is
            not an integer, a solution repeats either key, or a UID has no
            ``SolutionIndex`` in the same item.
    """
    entries: List[Tuple[int, str, int]] = []
    missing = 0
    stack: List[_SolutionItem] = []

    def close(item: _SolutionItem) -> None:
        """Record a finished item that names a solution.

        Args:
            item: Item whose last line has been read.

        Returns:
            None.

        Raises:
            ValueError: If the item's index or UID is invalid.
        """
        nonlocal missing
        if item.index is None and item.uid is None:
            return
        if item.index is None:
            raise ValueError(f"{yaml_path}: SolutionUID {item.uid} has no SolutionIndex")
        try:
            local_index = int(item.index)
        except ValueError as error:
            raise ValueError(f"{yaml_path}: bad SolutionIndex {item.index!r}") from error
        if item.uid is None:
            missing += 1
            return
        try:
            uid = decode_solution_uid(item.uid)
        except ValueError as error:
            raise ValueError(
                f"{yaml_path} (SolutionIndex={local_index}): {error}"
            ) from error
        if uid == 0:
            raise ValueError(
                f"Stored SolutionUID must not be zero: {yaml_path} "
                f"(SolutionIndex={local_index})"
            )
        entries.append((uid, str(yaml_path), local_index))

    top_column = -1
    with yaml_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            content = line.lstrip(" ")
            column = len(line) - len(content)
            if (
                column >= top_column
                and not content.startswith("- ")
                and not content.startswith("Solution")
            ):
                continue
            if "#" in line:
                line = _strip_comment(line)
                content = line.lstrip(" ")
                column = len(line) - len(content)
            if not content or content.isspace():
                continue
            if column < top_column:
                while stack and stack[-1].key_column > column:
                    close(stack.pop())
                top_column = stack[-1].key_column if stack else -1

            if content.startswith("- "):
                if ":" not in content:
                    continue
                while content.startswith("- "):
                    content = content[2:]
                    stripped = content.lstrip(" ")
                    column += 2 + len(content) - len(stripped)
                    content = stripped
                content = content.rstrip()
                if ": " in content or content.endswith(":"):
                    stack.append(_SolutionItem(column))
                    top_column = column
            elif content.startswith("Solution"):
                content = content.rstrip()
            else:
                continue

            if column != top_column or not content.startswith("Solution"):
                continue
            key, _, value = content.partition(":")
            key = key.rstrip()
            if key in _SOLUTION_KEYS:
                stack[-1].record(key, value.strip(), yaml_path)

    while stack:
        close(stack.pop())
    return entries, missing


def test_scan_handles_reordered_keys_and_comments(tmp_path: Path) -> None:
    """Verify the scan groups keys by item and ignores comments.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If any solution is attributed to the wrong UID or index.
    """
    logic = tmp_path / "dict.yaml"
    logic.write_text(
        "ProblemType:\n"
        "  SolutionIndex: 99\n"
        "Solutions:\n"
        "- SolutionUID: 0u3E  # written before the index\n"
        "  SolutionIndex: 0\n"
        "  Nested:\n"
        "    SolutionUID: 0u5\n"
        "- SolutionIndex: 1\n"
        "  # SolutionUID: 0u3E\n"
        "  SolutionUID: 0u3F\n"
        "- SolutionIndex: 2\n"
        "ExactLogic:\n"
        "- - [64, 64, 1, 64]\n"
        "  - [0, 1.0]\n",
        encoding="utf-8",
    )

    entries, missing = _scan_yaml_file(logic)

    path = str(logic)
    assert entries == [
        (decode_solution_uid("0u3E"), path, 0),
        (decode_solution_uid("0u3F"), path, 1),
    ]
    assert missing == 1


def test_scan_reads_list_format_solutions(tmp_path: Path) -> None:
    """Verify nested list-format solutions are read at their own key column.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If a list-format solution is missed or misattributed.
    """
    logic = tmp_path / "list.yaml"
    logic.write_text(
        "- {MinimumRequiredVersion: 5.0.0}\n"
        "- gfx950\n"
        "- Activation: true\n"
        "  DataType: 0\n"
        "- - 1LDSBuffer: 0\n"
        "    SFCWGM:\n"
        "    - [1, 1]\n"
        "    SolutionIndex: 0\n"
        "    SolutionUID: 0u3E\n"
        "  - 1LDSBuffer: 1\n"
        "    SolutionUID: 0u3F\n"
        "    SolutionIndex: 1\n"
        "- [2, 3, 0, 1]\n"
        "- - - [64, 64, 1, 64]\n"
        "    - [0, 1.0]\n",
        encoding="utf-8",
    )

    entries, missing = _scan_yaml_file(logic)

    path = str(logic)
    assert entries == [
        (decode_solution_uid("0u3E"), path, 0),
        (decode_solution_uid("0u3F"), path, 1),
    ]
    assert missing == 0


@pytest.mark.parametrize(
    "solutions",
    [
        "- SolutionIndex: 0\n  SolutionUID: 0u0\n",
        "- SolutionIndex: 0\n  SolutionUID: 0u3E # comment\n  SolutionUID: 0u3F\n",
        "- SolutionUID: 0u3E\n  KernelNameMin: k\n",
        "- SolutionIndex: 0\n  SolutionUID: 0u03E\n",
    ],
    ids=["zero", "repeated-key", "no-index", "leading-zero"],
)
def test_scan_rejects_invalid_solutions(tmp_path: Path, solutions: str) -> None:
    """Verify invalid UID entries still fail the scan.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        solutions: ``Solutions`` block containing one invalid entry.

    Returns:
        None.

    Raises:
        AssertionError: If the scan accepts the invalid entry.
    """
    logic = tmp_path / "bad.yaml"
    logic.write_text("Solutions:\n" + solutions, encoding="utf-8")

    with pytest.raises(ValueError):
        _scan_yaml_file(logic)


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
    return _LOGIC_ROOT


@_needs_logic_dir
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
