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
