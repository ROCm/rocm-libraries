# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Checks that compare a changed logic file with its base revision.

These do not belong in ``TensileLogic --check-all``. That command sees one
tree. ``scripts/run_tensile_logic_check.py`` calls them for the paths in
``--changed-files``, using the base text the caller read (``git show HEAD^``
in the logic-corpus lane).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

try:
    import yaml

    try:
        from yaml import CSafeLoader as SafeLoader
    except ImportError:  # pragma: no cover
        from yaml import SafeLoader
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_LOGIC_MARK = "src/amd_detail/rocblaslt/src/Tensile/Logic/"

_LEAD_KEYS = ("SolutionIndex", "SolutionUID", "KernelNameMin", "SolutionNameMin")


def _load(text: str) -> Any:
    """Parse logic YAML text.

    Args:
        text: File contents.

    Returns:
        The parsed document.

    Raises:
        RuntimeError: If PyYAML is not installed.
        yaml.YAMLError: If ``text`` is not valid YAML.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    return yaml.load(text, Loader=SafeLoader)


def _root_format(data: Any) -> str:
    """Classify a logic document root.

    Args:
        data: Parsed YAML document.

    Returns:
        ``list``, ``dict``, or ``other``.

    Raises:
        No exceptions are raised.
    """
    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        return "dict"
    return "other"


def _solutions(data: Any) -> list[Any]:
    """Return the solution list from either logic format.

    Args:
        data: Parsed logic document.

    Returns:
        The solutions. A short list document contributes nothing.

    Raises:
        No exceptions are raised.
    """
    if isinstance(data, dict):
        solutions = data.get("Solutions") or []
        return list(solutions) if isinstance(solutions, list) else []
    if isinstance(data, list) and len(data) > 5 and isinstance(data[5], list):
        return list(data[5])
    return []


def _solution_identity(solution: Any, position: int) -> str:
    """Pair a solution with the same solution in the base revision.

    Args:
        solution: One solution value.
        position: Zero-based index in the solutions list.

    Returns:
        ``index:`` plus ``SolutionIndex`` when that field is present, otherwise
        ``position:`` plus ``position``.

    Raises:
        No exceptions are raised.
    """
    if isinstance(solution, dict) and "SolutionIndex" in solution:
        return f"index:{solution['SolutionIndex']!r}"
    return f"position:{position}"


def _uid(solution: Any) -> Optional[str]:
    """Return a solution's non-empty ``SolutionUID``.

    Args:
        solution: One solution value.

    Returns:
        The stripped UID, or None when the solution has none.

    Raises:
        No exceptions are raised.
    """
    if not isinstance(solution, dict):
        return None
    uid = solution.get("SolutionUID")
    if isinstance(uid, str) and uid.strip():
        return uid.strip()
    return None


def list_format_errors(new_data: Any, base_data: Any, added: bool) -> list[str]:
    """Reject a new list-format library and a dict that becomes a list.

    Args:
        new_data: Parsed changed file.
        base_data: Parsed base revision, or None when the file is new.
        added: True when the base revision has no such file.

    Returns:
        Error messages. Empty when the change is allowed.

    Raises:
        No exceptions are raised.
    """
    introduced = added or _root_format(base_data) == "dict"
    if _root_format(new_data) == "list" and introduced:
        return [
            "List-format library logic was added; dict format is required for new libraries."
        ]
    return []


def solution_uid_errors(new_data: Any, base_data: Any) -> list[str]:
    """Reject a removed ``SolutionUID`` and one stored on ``DefaultSolution``.

    A solution that never had a ``SolutionUID`` is allowed. The same solution
    is the one with the same ``SolutionIndex``, or the same list position when
    it has no index.

    Args:
        new_data: Parsed changed file.
        base_data: Parsed base revision, or None when the file is new.

    Returns:
        Error messages.

    Raises:
        No exceptions are raised.
    """
    errors: list[str] = []
    if isinstance(new_data, dict) and isinstance(new_data.get("DefaultSolution"), dict):
        if "SolutionUID" in new_data["DefaultSolution"]:
            errors.append("DefaultSolution must not contain SolutionUID.")
    base_uids = {}
    for position, solution in enumerate(_solutions(base_data)):
        uid = _uid(solution)
        if uid is not None:
            base_uids[_solution_identity(solution, position)] = uid
    for position, solution in enumerate(_solutions(new_data)):
        if _uid(solution) is not None:
            continue
        previous = base_uids.get(_solution_identity(solution, position))
        if previous is None:
            continue
        if isinstance(solution, dict) and "SolutionIndex" in solution:
            label = f"SolutionIndex {solution['SolutionIndex']}"
        else:
            label = f"Solution at position {position}"
        errors.append(f"{label}: SolutionUID {previous} was removed.")
    return errors


def _keys_in_order(solution: Any, lead: bool) -> bool:
    """Tell whether one solution's keys are in the order the writers emit.

    Args:
        solution: One solution value.
        lead: True for dict format, where the ``_LEAD_KEYS`` that are present
            come first in that order. False for list format, where every key
            is sorted.

    Returns:
        True when the remaining keys follow ``sorted()`` and a dict
        ``InternalSupportParams`` is sorted too. A non-dict solution passes.

    Raises:
        No exceptions are raised.
    """
    if not isinstance(solution, dict):
        return True
    keys = list(solution)
    head = [key for key in _LEAD_KEYS if key in solution] if lead else []
    if keys != head + sorted(key for key in keys if key not in head):
        return False
    params = solution.get("InternalSupportParams")
    return not isinstance(params, dict) or list(params) == sorted(params)


def _first_unordered(data: Any) -> Optional[int]:
    """Find the first solution whose keys are out of order.

    Args:
        data: Parsed logic document.

    Returns:
        Zero-based position in the solutions list, or None when all are ordered.

    Raises:
        No exceptions are raised.
    """
    lead = _root_format(data) == "dict"
    for position, solution in enumerate(_solutions(data)):
        if not _keys_in_order(solution, lead):
            return position
    return None


def key_order_errors(new_data: Any, base_data: Any, added: bool) -> list[str]:
    """Reject solution keys that fall out of the order the writers emit.

    A dict-format solution starts with ``SolutionIndex``, ``SolutionUID`` when
    present, ``KernelNameMin``, and ``SolutionNameMin``. A list-format solution
    has no lead keys. The rest, and ``InternalSupportParams``, follow
    ``sorted()``: digits, capital letters, underscore, then small letters. A
    modified file that was already out of order is not reported.

    Args:
        new_data: Parsed changed file.
        base_data: Parsed base revision, or None when the file is new.
        added: True when the base revision has no such file.

    Returns:
        At most one error message.

    Raises:
        No exceptions are raised.
    """
    position = _first_unordered(new_data)
    if position is None or (not added and _first_unordered(base_data) is not None):
        return []
    return [
        f"Solution keys are not in canonical order, first at solution position {position}."
    ]


def changed_logic_paths(repo_root: Path, list_file: Path) -> list[tuple[str, Path]]:
    """Return the added or modified logic YAML files named in ``list_file``.

    Args:
        repo_root: Repository root. Paths in ``list_file`` are relative to it,
            or absolute.
        list_file: Newline-separated changed paths, as ``git diff --name-only``
            prints them.

    Returns:
        ``(repo-relative posix path, path)`` pairs. A deleted file is left out.

    Raises:
        OSError: If ``list_file`` cannot be read.
    """
    paths: list[tuple[str, Path]] = []
    for raw in list_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if _LOGIC_MARK not in line.replace("\\", "/") or not line.endswith((".yaml", ".yml")):
            continue
        path = Path(line) if Path(line).is_absolute() else repo_root / line
        if path.is_file():
            paths.append((path.resolve().relative_to(repo_root.resolve()).as_posix(), path))
    return paths


def logic_change_errors(rel: str, new_data: Any, base_data: Any, added: bool) -> list[str]:
    """Run every change check on one added or modified logic file.

    Args:
        rel: Repository-relative posix path.
        new_data: Parsed changed file.
        base_data: Parsed base revision, or None when the file is new.
        added: True when the base revision has no such file.

    Returns:
        Error messages.

    Raises:
        No exceptions are raised.
    """
    del rel
    return (
        list_format_errors(new_data, base_data, added)
        + solution_uid_errors(new_data, base_data)
        + key_order_errors(new_data, base_data, added)
    )


def check_changed_logic(
    repo_root: Path,
    list_file: Path,
    show_text: Callable[[str], Optional[str]],
) -> list[str]:
    """Check each changed logic file in ``list_file`` against its base text.

    Args:
        repo_root: Repository root. Paths in ``list_file`` are relative to it,
            or absolute.
        list_file: Newline-separated changed paths, as ``git diff --name-only``
            prints them.
        show_text: ``(repo-relative posix path) -> base text``. None means the
            path is not in the base revision.

    Returns:
        ``path: message`` lines. Empty when every changed logic file passes.

    Raises:
        RuntimeError: If PyYAML is not installed.
        OSError: If ``list_file`` or a changed file cannot be read.
        yaml.YAMLError: If a changed file or its base text is not valid YAML.
    """
    errors: list[str] = []
    for rel, path in changed_logic_paths(repo_root, list_file):
        base_text = show_text(rel)
        new_data = _load(path.read_text(encoding="utf-8"))
        base_data = _load(base_text) if base_text is not None else None
        for message in logic_change_errors(rel, new_data, base_data, base_text is None):
            errors.append(f"{rel}: {message}")
    return errors
