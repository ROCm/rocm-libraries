# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused tests for list-format introduction and TuningDriver solution rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from audit_hipblaslt_libraries import (
    DEFAULT_CHECK_IDS,
    FileAuditContext,
    format_check_list,
    main,
    normalize_library,
    parse_name_status,
    resolve_enabled_checks,
    run_enabled_checks,
)

LOGIC_PREFIX = "projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/"


def _errors(
    record: Any,
    checks: tuple[str, ...],
    *,
    added: bool = False,
    base_format: Optional[str] = None,
) -> list[tuple[str, str]]:
    """Run selected checks and collect ``(code, message)`` pairs.

    Args:
        record: Parsed library record.
        checks: Check ids to run.
        added: Whether the path is a new file.
        base_format: Root format at the base revision.

    Returns:
        Errors emitted by the selected checks.

    Raises:
        No exceptions are raised.
    """
    found: list[tuple[str, str]] = []

    def sink(path: str, code: str, message: str) -> None:
        del path
        found.append((code, message))

    run_enabled_checks(
        record,
        sink,
        {},
        FileAuditContext(added=added, base_format=base_format),
        checks,
    )
    return found


def _dict_library(
    solutions: list[dict[str, Any]],
    *,
    default_solution: Optional[dict[str, Any]] = None,
    filename: str = "Equality_gfx950_Cijk_Ailk_Bjlk_sample.yaml",
) -> Any:
    """Build a minimal dict-format library record.

    Args:
        solutions: Solution dictionaries.
        default_solution: Optional ``DefaultSolution`` map.
        filename: Basename stored on the record path.

    Returns:
        Normalized library record.

    Raises:
        ValueError: If the constructed document is not a dict library.
    """
    data: dict[str, Any] = {
        "ArchitectureName": "gfx950",
        "ScheduleName": "gfx950",
        "DeviceNames": ["Device 75a0"],
        "ProblemType": {"TransposeA": False, "TransposeB": True},
        "Solutions": solutions,
        "LibraryType": "Equality",
        "ExactLogic": [],
    }
    if default_solution is not None:
        data["DefaultSolution"] = default_solution
    return normalize_library(data, Path("gfx950/gfx950/Equality") / filename)


def _list_library() -> Any:
    """Build a minimal list-format library record.

    Args:
        None.

    Returns:
        Normalized library record whose format is ``list``.

    Raises:
        ValueError: If the constructed document is not a list library.
    """
    data: list[Any] = [None] * 12
    data[1] = "gfx950"
    data[2] = "gfx950"
    data[3] = ["Device 75a0"]
    data[4] = {}
    data[5] = []
    data[6] = []
    data[7] = []
    data[8] = []
    data[10] = "DeviceEfficiency"
    data[11] = "Equality"
    return normalize_library(data, Path("legacy.yaml"))


def test_list_format_added_fails_for_new_files() -> None:
    """A newly added list-format library fails.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the format check does not fail.
    """
    errors = _errors(_list_library(), ("list_format_added",), added=True)
    assert [code for code, _ in errors] == ["list_format_added"]


def test_list_format_edit_is_allowed() -> None:
    """Editing an existing list-format library does not fail the format check.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If an edited list library is rejected.
    """
    errors = _errors(
        _list_library(),
        ("list_format_added",),
        added=False,
        base_format="list",
    )
    assert errors == []


def test_dict_to_list_fails() -> None:
    """Changing a dict-format library into list-format fails.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a dict-to-list change is accepted.
    """
    errors = _errors(
        _list_library(),
        ("list_format_added",),
        added=False,
        base_format="dict",
    )
    assert [code for code, _ in errors] == ["list_format_added"]


def test_new_dict_library_passes_format_check() -> None:
    """A newly added dict-format library passes the format check.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a new dict library fails the format check.
    """
    record = _dict_library([{"SolutionIndex": 0}])
    assert record.format == "dict"
    errors = _errors(record, ("list_format_added",), added=True)
    assert errors == []


def test_streamk_requires_gsu_zero_only_when_streamk_is_3() -> None:
    """StreamK 3 requires GlobalSplitU 0; other StreamK values do not.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the StreamK/GlobalSplitU rule does not match TuningDriver.
    """
    bad = _dict_library([{"SolutionIndex": 1, "StreamK": 3, "GlobalSplitU": 4}])
    good = _dict_library([{"SolutionIndex": 1, "StreamK": 3, "GlobalSplitU": 0}])
    other = _dict_library([{"SolutionIndex": 1, "StreamK": 1, "GlobalSplitU": 4}])
    assert _errors(bad, ("streamk_gsu",))
    assert _errors(good, ("streamk_gsu",)) == []
    assert _errors(other, ("streamk_gsu",)) == []


def test_use_custom_main_loop_schedule_must_be_one() -> None:
    """UseCustomMainLoopSchedule must coerce to 1, including when it is missing.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a non-1 or missing value is accepted.
    """
    bad = _dict_library([{"SolutionIndex": 2, "UseCustomMainLoopSchedule": 0}])
    missing = _dict_library([{"SolutionIndex": 2}])
    good = _dict_library([{"SolutionIndex": 2, "UseCustomMainLoopSchedule": 1}])
    assert _errors(bad, ("use_custom_main_loop_schedule",))
    assert _errors(missing, ("use_custom_main_loop_schedule",))
    assert _errors(good, ("use_custom_main_loop_schedule",)) == []


def test_default_solution_fills_use_custom_main_loop_schedule() -> None:
    """A missing solution key is taken from DefaultSolution before the check.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If DefaultSolution does not satisfy the check.
    """
    record = _dict_library(
        [{"SolutionIndex": 3, "StreamK": 0}],
        default_solution={"UseCustomMainLoopSchedule": 1},
    )
    assert _errors(record, ("use_custom_main_loop_schedule",)) == []


def test_check_registry_lists_defaults_and_skip() -> None:
    """The registry lists every default check and honors --skip.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a default check is missing or --skip is ignored.
    """
    text = format_check_list()
    for check_id in (
        "list_format_added",
        "streamk_gsu",
        "use_custom_main_loop_schedule",
        "activation_func_call",
        "filename_policy",
        "kernel_name_matches_filename",
    ):
        assert check_id in text
    assert "use_custom_main_loop_schedule" not in resolve_enabled_checks(
        None, ["use_custom_main_loop_schedule"]
    )
    assert resolve_enabled_checks(["streamk_gsu"], None) == ("streamk_gsu",)
    assert "dict_structure" in DEFAULT_CHECK_IDS


def test_list_checks_cli_exits_zero() -> None:
    """``--list-checks`` prints the registry and does not audit files.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the CLI does not exit 0.
    """
    assert main(["--list-checks"]) == 0


def test_parse_name_status_marks_adds_and_renames() -> None:
    """Additions and rename destinations are list-format candidates.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If adds, renames, or unrelated paths are classified wrong.
    """
    logic = LOGIC_PREFIX
    selected = parse_name_status(
        [
            f"M\t{logic}gfx950/keep.yaml",
            f"A\t{logic}gfx950/new.yaml",
            f"R100\t{logic}gfx950/old.yaml\t{logic}gfx950/renamed.yaml",
            f"M\tprojects/hipblaslt/tensilelite/readme.md",
            f"D\t{logic}gfx950/removed.yaml",
        ]
    )
    assert selected == [
        (f"{logic}gfx950/keep.yaml", False),
        (f"{logic}gfx950/new.yaml", True),
        (f"{logic}gfx950/renamed.yaml", True),
    ]
