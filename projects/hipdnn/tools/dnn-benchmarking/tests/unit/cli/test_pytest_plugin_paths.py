# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for pytest dnn plugin path option parsing."""

from pathlib import Path

from tests import conftest as project_conftest


def _plugin_dir(root: Path, relative: str) -> Path:
    path = root / relative
    path.mkdir(parents=True)
    (path / "engine.so").write_bytes(b"")
    return Path(relative)


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
