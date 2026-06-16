# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for pytest dnn plugin path option parsing."""

from pathlib import Path

import pytest

from tests import conftest as project_conftest


def _plugin_dir(root: Path, relative: str) -> Path:
    path = root / relative
    path.mkdir(parents=True)
    (path / "engine.so").write_bytes(b"")
    return Path(relative)


def _absolute_plugin_dir(root: Path, relative: str) -> Path:
    path = root / relative
    path.mkdir(parents=True)
    (path / "engine.so").write_bytes(b"")
    return path


def test_parse_dnn_plugin_paths_accepts_relative_comma_list(
    tmp_path: Path, monkeypatch
) -> None:
    """Relative --dnn-plugin-paths entries are validated from pytest cwd."""
    first = _plugin_dir(tmp_path, "plugins/a")
    second = _plugin_dir(tmp_path, "plugins/b")

    monkeypatch.chdir(tmp_path)

    assert project_conftest._parse_plugin_paths("plugins/a, plugins/b") == [
        first,
        second,
    ]


def test_parse_dnn_plugin_paths_accepts_absolute_path(tmp_path: Path) -> None:
    plugin = _absolute_plugin_dir(tmp_path, "plugins/absolute")

    assert project_conftest._parse_plugin_paths(str(plugin)) == [plugin]


def test_parse_dnn_plugin_paths_accepts_mixed_absolute_and_relative_paths(
    tmp_path: Path, monkeypatch
) -> None:
    relative = _plugin_dir(tmp_path, "plugins/relative")
    absolute = _absolute_plugin_dir(tmp_path, "plugins/absolute")

    monkeypatch.chdir(tmp_path)

    assert project_conftest._parse_plugin_paths(f"plugins/relative, {absolute}") == [
        relative,
        absolute,
    ]


def test_parse_dnn_plugin_paths_ignores_blank_comma_entries(
    tmp_path: Path, monkeypatch
) -> None:
    first = _plugin_dir(tmp_path, "plugins/a")
    second = _plugin_dir(tmp_path, "plugins/b")

    monkeypatch.chdir(tmp_path)

    assert project_conftest._parse_plugin_paths(" plugins/a, ,plugins/b, ") == [
        first,
        second,
    ]


def test_parse_dnn_plugin_paths_rejects_empty_input() -> None:
    with pytest.raises(pytest.UsageError, match="requires at least one path"):
        project_conftest._parse_plugin_paths(" , ")


def test_parse_dnn_plugin_paths_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(pytest.UsageError) as excinfo:
        project_conftest._parse_plugin_paths(str(missing))

    message = str(excinfo.value)
    assert "entries must be directories containing at least one .so file" in message
    assert str(missing) in message


def test_parse_dnn_plugin_paths_rejects_directory_without_shared_library(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(pytest.UsageError) as excinfo:
        project_conftest._parse_plugin_paths(str(empty))

    message = str(excinfo.value)
    assert "entries must be directories containing at least one .so file" in message
    assert str(empty) in message


def test_parse_dnn_plugin_paths_reports_all_invalid_entries(tmp_path: Path) -> None:
    valid = _absolute_plugin_dir(tmp_path, "plugins/valid")
    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(pytest.UsageError) as excinfo:
        project_conftest._parse_plugin_paths(f"{valid},{missing},{empty}")

    message = str(excinfo.value)
    assert str(missing) in message
    assert str(empty) in message
    assert str(valid) not in message
