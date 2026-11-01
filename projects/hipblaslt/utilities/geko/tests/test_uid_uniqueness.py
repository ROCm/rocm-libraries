# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for loading SolutionIdGen from the hipBLASLt checkout used by Geko."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geko.library.uid_uniqueness import assign_fresh_solution_uids, uniquify_generated_logic
from geko.paths import HIPBLASLT_PATH_ENV_VAR, bind_hipblaslt


def _write_solution_id_gen(directory: Path, source: str) -> Path:
    """Write a stand-in SolutionIdGen.py.

    Args:
        directory: Directory that will contain the stand-in module.
        source: Python source of the stand-in module.

    Returns:
        Path of the written module.

    Raises:
        OSError: If the file cannot be created.
    """
    module_path = directory / "SolutionIdGen.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(source, encoding="utf-8")
    return module_path


def test_assign_fresh_solution_uids_skips_an_empty_library(tmp_path: Path) -> None:
    """An empty library list does not look for SolutionIdGen.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call loads the codec or returns a count.
    """
    assert assign_fresh_solution_uids([]) == 0


def test_assign_fresh_solution_uids_rejects_a_missing_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound checkout without SolutionIdGen.py fails and does not fall through.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the missing codec is not reported or the env checkout is used.
    """
    _write_solution_id_gen(
        tmp_path / "env" / "tensilelite" / "tensilelite" / "Common",
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uENV'\n"
        "    seen.add('0uENV')\n"
        "    return '0uENV'\n",
    )
    monkeypatch.setenv(HIPBLASLT_PATH_ENV_VAR, str(tmp_path / "env"))
    bind_hipblaslt(tmp_path / "missing")
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        assign_fresh_solution_uids([library])

    assert library.solutions[0]["SolutionUID"] == "0u5"


def test_assign_fresh_solution_uids_uses_the_bound_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound hipBLASLt checkout supplies the codec and wins over the env var.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the bound codec is not used.
    """
    _write_solution_id_gen(
        tmp_path / "bound" / "tensilelite" / "tensilelite" / "Common",
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uBOUND'\n"
        "    seen.add('0uBOUND')\n"
        "    return '0uBOUND'\n",
    )
    _write_solution_id_gen(
        tmp_path / "env" / "tensilelite" / "tensilelite" / "Common",
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uENV'\n"
        "    seen.add('0uENV')\n"
        "    return '0uENV'\n",
    )
    monkeypatch.setenv(HIPBLASLT_PATH_ENV_VAR, str(tmp_path / "env"))
    bind_hipblaslt(tmp_path / "bound")
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    assert assign_fresh_solution_uids([library]) == 1
    assert library.solutions[0]["SolutionUID"] == "0uBOUND"


def test_assign_fresh_solution_uids_uses_the_loaded_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present SolutionIdGen.py supplies the UID draw.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in UID is not stored.
    """
    module_path = _write_solution_id_gen(
        tmp_path,
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uTEST'\n"
        "    seen.add('0uTEST')\n"
        "    return '0uTEST'\n",
    )
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    count = assign_fresh_solution_uids([library])

    assert count == 1
    assert library.solutions[0]["SolutionUID"] == "0uTEST"


def test_load_rejects_a_module_without_a_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module spec without a loader is an import error.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the missing loader is not reported.
    """
    module_path = _write_solution_id_gen(tmp_path, "x = 1\n")
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    monkeypatch.setattr(
        "geko.library.uid_uniqueness.importlib.util.spec_from_file_location",
        lambda *_args, **_kwargs: SimpleNamespace(loader=None),
    )
    library = SimpleNamespace(solutions=[{}])
    with pytest.raises(ImportError, match="Could not load SolutionIdGen"):
        assign_fresh_solution_uids([library])


def test_uniquify_generated_logic_skips_empty_paths() -> None:
    """No logic files means no SolutionIdGen load.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call returns a rewrite count.
    """
    assert uniquify_generated_logic([]) == 0


def test_uniquify_generated_logic_rejects_a_missing_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env checkout without SolutionIdGen.py fails and does not fall through.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the logic file changes or no error is raised.
    """
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")
    monkeypatch.setenv(HIPBLASLT_PATH_ENV_VAR, str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        uniquify_generated_logic([logic])

    assert logic.read_text(encoding="utf-8") == "SolutionUID: 0u5\n"


def test_uniquify_generated_logic_uses_the_loaded_rewriter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collision rewriting uses the loaded SolutionIdGen.py.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in rewriter is not called.
    """
    module_path = _write_solution_id_gen(
        tmp_path,
        "def rewrite_colliding_solution_uids(paths):\n" "    return len(paths)\n",
    )
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    rewritten = uniquify_generated_logic([logic])

    assert rewritten == 1


def test_assign_fresh_solution_uids_rejects_a_missing_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No bound checkout, env var, or ancestor walk leaves the copied UID unchanged.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the missing checkout is not reported.
    """
    monkeypatch.setattr(
        "geko.library.uid_uniqueness.locate_hipblaslt_root",
        lambda **_kwargs: None,
    )
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        assign_fresh_solution_uids([library])

    assert library.solutions[0]["SolutionUID"] == "0u5"


def test_assign_fresh_solution_uids_uses_the_env_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GEKO_HIPBLASLT_PATH supplies the codec when nothing is bound.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the env checkout is not used.
    """
    _write_solution_id_gen(
        tmp_path / "tensilelite" / "tensilelite" / "Common",
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uENV'\n"
        "    seen.add('0uENV')\n"
        "    return '0uENV'\n",
    )
    monkeypatch.setenv(HIPBLASLT_PATH_ENV_VAR, str(tmp_path))
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    assert assign_fresh_solution_uids([library]) == 1
    assert library.solutions[0]["SolutionUID"] == "0uENV"


def test_assign_fresh_solution_uids_uses_the_containing_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no binding and no env var, the codec is the checkout containing Geko.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the source-tree codec does not replace a malformed UID.
    """
    monkeypatch.delenv(HIPBLASLT_PATH_ENV_VAR, raising=False)
    library = SimpleNamespace(solutions=[{"SolutionUID": "not-a-uid"}])

    assert assign_fresh_solution_uids([library]) == 1
    assert library.solutions[0]["SolutionUID"].startswith("0u")
    assert library.solutions[0]["SolutionUID"] != "not-a-uid"
