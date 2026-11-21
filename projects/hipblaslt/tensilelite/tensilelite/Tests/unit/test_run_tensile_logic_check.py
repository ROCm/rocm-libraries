# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``--changed-files`` scoping in hipBLASLt's
``scripts/run_tensile_logic_check.py``.

The logic-corpus component lane passes a pull request's changed paths, and the
script turns them into the set of architectures ``TensileLogic --check-all``
validates. An empty set skips the check, so a mistake here quietly turns the
lane into a no-op. Logic-file cases use a synthetic tree under ``tmp_path``;
validator cases need no files, because a validator path decides the answer
before any logic is read.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "run_tensile_logic_check.py"
_LOGIC_REL = Path("library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full")


@pytest.fixture(scope="module")
def script():
    if not _SCRIPT.is_file():
        pytest.skip("hipBLASLt scripts/ is not in this tree")
    spec = importlib.util.spec_from_file_location("run_tensile_logic_check", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    """A hipBLASLt project root two levels below a repository root."""
    root = tmp_path / "repo" / "projects" / "hipblaslt"
    (root / "tensilelite" / "tensilelite" / "TensileLogic").mkdir(parents=True)
    return root


def _write_logic(root, rel, arch):
    path = root / _LOGIC_REL / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"MinimumRequiredVersion: 5.0.0\nScheduleName: {arch}\nArchitectureName: {arch}\n"
        "CUCount: null\nDeviceNames: []\nLibraryType: Equality\n"
    )
    return "projects/hipblaslt" / _LOGIC_REL / rel


def _changed(tmp_path, *lines):
    listFile = tmp_path / "changed_files.txt"
    listFile.write_text("".join(f"{line}\n" for line in lines))
    return listFile


def test_changed_logic_files_select_their_declared_architectures(script, project, tmp_path):
    a = _write_logic(project, "gfx1151/Equality/a.yaml", "gfx1151")
    b = _write_logic(project, "aldebaran/GridBased/b.yaml", "gfx90a")
    listFile = _changed(tmp_path, a, "", b)
    assert script._changed_architectures(project, listFile) == {"gfx1151", "gfx90a"}


def test_absolute_paths_are_accepted(script, project, tmp_path):
    rel = _write_logic(project, "gfx1151/Equality/a.yaml", "gfx1151")
    listFile = _changed(tmp_path, project.parents[1] / rel)
    assert script._changed_architectures(project, listFile) == {"gfx1151"}


def test_deleted_logic_file_takes_the_architecture_of_its_directory(script, project, tmp_path):
    _write_logic(project, "gfx1151/Equality/kept.yaml", "gfx1151")
    deleted = "projects/hipblaslt" / _LOGIC_REL / "gfx1151/Equality/deleted.yaml"
    assert script._changed_architectures(project, _changed(tmp_path, deleted)) == {"gfx1151"}


def test_deleted_logic_directory_checks_everything(script, project, tmp_path):
    deleted = "projects/hipblaslt" / _LOGIC_REL / "gfx1151/Equality/deleted.yaml"
    assert script._changed_architectures(project, _changed(tmp_path, deleted)) is None


def test_paths_outside_logic_and_validators_select_nothing(script, project, tmp_path):
    listFile = _changed(
        tmp_path,
        "projects/hipblaslt/tensilelite/tox.ini",
        "projects/hipblaslt/tensilelite/tensilelite/Tests/unit/logic_corpus.py",
        ".github/workflows/component-ci-logic-corpus.yml",
    )
    assert script._changed_architectures(project, listFile) == set()


@pytest.mark.parametrize(
    "validator",
    [
        "projects/hipblaslt/tensilelite/tensilelite/logic/run.py",
        "projects/hipblaslt/tensilelite/tensilelite/logic/known_bugs.yaml",
        "projects/hipblaslt/tensilelite/tensilelite/CustomYamlLoader.py",
        "projects/hipblaslt/scripts/run_tensile_logic_check.py",
    ],
)
def test_a_changed_validator_checks_everything(script, tmp_path, validator):
    # The real project root: the script compares against its own location.
    root = _SCRIPT.parents[1]
    assert script._changed_architectures(root, _changed(tmp_path, validator)) is None


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--changed-files=list.txt"], "list.txt"),
        (["--changed-files="], ""),
        ([], None),
    ],
)
def test_parse_argv_reads_changed_files(script, monkeypatch, argv, expected):
    monkeypatch.setattr(sys, "argv", ["run_tensile_logic_check.py", *argv])
    _, changedFiles, passthrough = script._parse_argv(_SCRIPT.parents[1])
    assert changedFiles == expected
    assert passthrough == []


def test_parse_argv_rejects_a_separate_changed_files_value(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_tensile_logic_check.py", "--changed-files", "list.txt"])
    with pytest.raises(SystemExit):
        script._parse_argv(_SCRIPT.parents[1])


def _fake_git(script, monkeypatch, has_parent: bool, base: dict) -> list:
    """Replace ``git`` in the script with an in-memory stand-in.

    Args:
        script: The loaded check script.
        monkeypatch: Pytest monkeypatch fixture.
        has_parent: Whether ``git rev-parse HEAD^`` succeeds.
        base: Base-revision text keyed by repository-relative path.

    Returns:
        The list that records each git argument vector.

    Raises:
        None.
    """
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "rev-parse":
            return subprocess.CompletedProcess(args, 0 if has_parent else 1, b"", b"")
        rel = args[2].split(":", 1)[1]
        if rel not in base:
            return subprocess.CompletedProcess(args, 128, b"", b"fatal")
        return subprocess.CompletedProcess(args, 0, base[rel].encode(), b"")

    monkeypatch.setattr(script.subprocess, "run", run)
    return calls


def test_base_revision_text_reads_the_parent_commit(script, monkeypatch, tmp_path):
    """``git show HEAD^:path`` gives the text; a missing blob gives None.

    Args:
        script: The loaded check script.
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the text or the git arguments are wrong.
    """
    calls = _fake_git(script, monkeypatch, True, {"a.yaml": "old\n"})
    assert script._base_revision_text(tmp_path, "a.yaml") == "old\n"
    assert script._base_revision_text(tmp_path, "b.yaml") is None
    assert calls[0] == ["git", "show", "HEAD^:a.yaml"]


def test_check_logic_changes_fails_a_new_list_file(script, project, tmp_path, monkeypatch, capsys):
    """A new list-format logic file exits 1 and names the file.

    Args:
        script: The loaded check script.
        project: Synthetic hipBLASLt project root.
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.
        capsys: Pytest output capture.

    Returns:
        None.

    Raises:
        AssertionError: If the new list file passes.
    """
    rel = "projects/hipblaslt" / _LOGIC_REL / "gfx1151/Equality/new.yaml"
    (project.parents[1] / rel).parent.mkdir(parents=True)
    (project.parents[1] / rel).write_text("- [a]\n")
    _fake_git(script, monkeypatch, True, {})
    with pytest.raises(SystemExit) as raised:
        script._check_logic_changes(project, _changed(tmp_path, rel))
    assert raised.value.code == 1
    assert f"{rel.as_posix()}: List-format library logic was added" in capsys.readouterr().err


def test_check_logic_changes_passes_an_unchanged_rule(script, project, tmp_path, monkeypatch):
    """An edited list file that was already a list passes.

    Args:
        script: The loaded check script.
        project: Synthetic hipBLASLt project root.
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the edit is rejected.
    """
    rel = "projects/hipblaslt" / _LOGIC_REL / "gfx1151/Equality/old.yaml"
    (project.parents[1] / rel).parent.mkdir(parents=True)
    (project.parents[1] / rel).write_text("- [a]\n- [b]\n")
    _fake_git(script, monkeypatch, True, {rel.as_posix(): "- [a]\n"})
    assert script._check_logic_changes(project, _changed(tmp_path, rel)) is None


def test_check_logic_changes_needs_the_parent_commit(script, project, tmp_path, monkeypatch):
    """Without ``HEAD^`` a logic change cannot be compared, so the run stops.

    Args:
        script: The loaded check script.
        project: Synthetic hipBLASLt project root.
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the run continues.
    """
    rel = _write_logic(project, "gfx1151/Equality/a.yaml", "gfx1151")
    _fake_git(script, monkeypatch, False, {})
    with pytest.raises(SystemExit, match="HEAD\\^ is not available"):
        script._check_logic_changes(project, _changed(tmp_path, rel))


def test_check_logic_changes_skips_git_without_logic_files(script, project, tmp_path, monkeypatch):
    """A change with no logic file does not call git.

    Args:
        script: The loaded check script.
        project: Synthetic hipBLASLt project root.
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If git is called.
    """
    calls = _fake_git(script, monkeypatch, False, {})
    script._check_logic_changes(project, _changed(tmp_path, "README.md"))
    assert calls == []


@pytest.mark.parametrize("extra,policy", [([], False), (["--policy-checks"], True)])
def test_main_compares_changed_logic_before_scoping(
    script, monkeypatch, tmp_path, capsys, extra, policy
):
    """``main`` runs the base-revision comparison on a ``--changed-files`` run.

    Args:
        script: The loaded check script.
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.
        capsys: Pytest output capture.
        extra: Extra script arguments.
        policy: Whether the policy checks are expected.

    Returns:
        None.

    Raises:
        AssertionError: If the comparison is skipped or the flag is misread.
    """
    listFile = _changed(tmp_path, "README.md")
    compared = []
    argv = ["run_tensile_logic_check.py", *extra, f"--changed-files={listFile}"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(script, "_ensure_paths", lambda *args: None)
    monkeypatch.setattr(script, "_check_logic_changes", lambda *args: compared.append(args[1:]))
    monkeypatch.setattr(script, "_changed_architectures", lambda root, path: set())
    script.main()
    assert compared == [(listFile, policy)]
    assert "--policy-checks" not in sys.argv
    assert "nothing to check" in capsys.readouterr().out
