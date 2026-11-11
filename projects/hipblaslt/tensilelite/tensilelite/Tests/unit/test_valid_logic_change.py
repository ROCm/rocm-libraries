# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic tests for logic changes compared with a base revision.

The files live under ``tmp_path``. Nothing here reads the shipped corpus.
"""

from pathlib import Path

import pytest

import tensilelite.TensileLogic.ValidLogicChange as change

pytestmark = pytest.mark.unit

_LOGIC = change._LOGIC_MARK


def _place(repo: Path, name: str, text: str) -> str:
    """Write one logic file and return its repository-relative path.

    Args:
        repo: Temporary repository root.
        name: File name.
        text: YAML contents.

    Returns:
        Repository-relative posix path.

    Raises:
        OSError: If the file cannot be written.
    """
    rel = f"projects/hipblaslt/library/{_LOGIC}gfx950/{name}"
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel


def _run(repo: Path, rels: list[str], base: dict[str, str]) -> list[str]:
    """Run the change checks with a stand-in for ``git show``.

    Args:
        repo: Temporary repository root.
        rels: Changed paths, one per line.
        base: Base text keyed by repository-relative path.

    Returns:
        Error lines.

    Raises:
        OSError: If the path list cannot be written.
    """
    listed = repo / "changed.txt"
    listed.write_text("\n".join(rels) + "\n", encoding="utf-8")
    return change.check_changed_logic(repo, listed, base.get)


def test_new_list_file_fails_and_edited_list_file_passes(tmp_path: Path) -> None:
    """A new list file fails. The same text, already in the base, passes.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the two cases are classified the same way.
    """
    text = "- [a]\n"
    rel = _place(tmp_path, "new.yaml", text)
    assert _run(tmp_path, [rel], {})
    assert _run(tmp_path, [rel], {rel: text}) == []


def test_dict_converted_to_list_fails(tmp_path: Path) -> None:
    """Changing a dict library into a list fails.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the conversion is accepted.
    """
    rel = _place(tmp_path, "was-dict.yaml", "- [a]\n")
    errors = _run(tmp_path, [rel], {rel: "LibraryType: Equality\n"})
    assert len(errors) == 1
    assert "dict format is required" in errors[0]


def test_removed_solution_uid_fails_and_default_solution_cannot_hold_one(
    tmp_path: Path,
) -> None:
    """Dropping a base UID fails. A UID on DefaultSolution fails too.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If either case is accepted.
    """
    rel = _place(
        tmp_path,
        "uids.yaml",
        "DefaultSolution: {SolutionUID: hidden}\n"
        "Solutions:\n"
        "- {SolutionIndex: 1}\n"
        "- {SolutionIndex: 2, SolutionUID: keep}\n",
    )
    base = (
        "Solutions:\n"
        "- {SolutionIndex: 1, SolutionUID: gone}\n"
        "- {SolutionIndex: 2, SolutionUID: keep}\n"
    )
    errors = _run(tmp_path, [rel], {rel: base})
    assert any("SolutionUID gone was removed" in error for error in errors)
    assert any("DefaultSolution must not contain SolutionUID" in error for error in errors)


def test_removed_uid_is_matched_by_position_without_solution_index(tmp_path: Path) -> None:
    """A list file pairs solutions by position when they carry no index.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the removed UID is missed or a non-dict entry fails.
    """
    rel = _place(tmp_path, "list.yaml", "- [v]\n- s\n- a\n- d\n- {}\n- [{}, plain]\n")
    base = "- [v]\n- s\n- a\n- d\n- {}\n- [{SolutionUID: u0}, plain]\n"
    assert _run(tmp_path, [rel], {rel: base}) == [
        f"{rel}: Solution at position 0: SolutionUID u0 was removed."
    ]


_ORDERED = (
    "Solutions:\n"
    "- {SolutionIndex: 0, SolutionUID: u0, KernelNameMin: k, SolutionNameMin: s,"
    " 1LDSBuffer: 0, InternalSupportParams: {SupportUserGSU: 1, UseUniversalArgs: 0},"
    " MT: 1, _Flag: 0, a: 1}\n"
)


@pytest.mark.parametrize(
    "text",
    [
        _ORDERED.replace("_Flag: 0, a: 1", "a: 1, _Flag: 0"),
        _ORDERED.replace("SolutionIndex: 0, SolutionUID: u0", "SolutionUID: u0, SolutionIndex: 0"),
        _ORDERED.replace(
            "SupportUserGSU: 1, UseUniversalArgs: 0", "UseUniversalArgs: 0, SupportUserGSU: 1"
        ),
        _ORDERED.replace("KernelNameMin: k, ", "") + "- {a: 1, MT: 1}\n",
    ],
    ids=["tail", "lead", "internal-params", "second-solution"],
)
def test_added_file_with_keys_out_of_order_fails(tmp_path: Path, text: str) -> None:
    """A new dict file fails once, at the first solution out of order.

    Args:
        tmp_path: Pytest temporary directory.
        text: Logic file with one ordering mistake.

    Returns:
        None.

    Raises:
        AssertionError: If the ordered file fails or the mistake is missed.
    """
    rel = _place(tmp_path, "ordered.yaml", _ORDERED)
    assert _run(tmp_path, [rel], {}) == []
    _place(tmp_path, "ordered.yaml", text)
    errors = _run(tmp_path, [rel], {})
    assert len(errors) == 1
    assert "not in canonical order" in errors[0]
    position = 1 if "MT: 1}" in text else 0
    assert errors[0].endswith(f"solution position {position}.")


def test_list_file_sorts_every_key() -> None:
    """A list-format solution has no lead keys, so all keys are sorted.

    Returns:
        None.

    Raises:
        AssertionError: If list-format keys are judged with the lead keys.
    """
    head = ["v", "s", "a", "d", {}]
    assert change.key_order_errors(head + [[{"KernelNameMin": "k", "MT": 1}]], None, True) == []
    assert change.key_order_errors(head + [[{"SolutionIndex": 0, "MT": 1}]], None, True)
    assert change.key_order_errors(head + [["plain"]], None, True) == []


def test_modified_file_is_reported_only_when_it_loses_its_order(tmp_path: Path) -> None:
    """An edit fails when it breaks a sorted file, not when the base was unsorted.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If either case is classified the wrong way.
    """
    unordered = _ORDERED.replace("_Flag: 0, a: 1", "a: 1, _Flag: 0")
    rel = _place(tmp_path, "edited.yaml", unordered)
    assert len(_run(tmp_path, [rel], {rel: _ORDERED})) == 1
    assert _run(tmp_path, [rel], {rel: unordered}) == []


def test_other_roots_and_short_lists_have_no_solutions() -> None:
    """A scalar root is neither format, and a short list holds no solutions.

    Returns:
        None.

    Raises:
        AssertionError: If either shape is misread.
    """
    assert change._root_format("text") == "other"
    assert change.list_format_errors("text", None, True) == []
    assert change._solutions(["only", "five", "fields", "in", "here"]) == []


def test_loading_needs_pyyaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PyYAML the checks stop with a clear error.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the load does not raise.
    """
    monkeypatch.setattr(change, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML is required"):
        change._load("a: 1\n")


def test_unrelated_path_is_ignored(tmp_path: Path) -> None:
    """A changed file outside the logic tree is not a logic change.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the unrelated path is checked.
    """
    listed = tmp_path / "changed.txt"
    listed.write_text("README.md\n", encoding="utf-8")
    assert change.check_changed_logic(tmp_path, listed, lambda _rel: None) == []
