# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for tasks._rocisa_install_status (the three-way rocisa detection
that drives auto-enabling HIPBLASLT_BUNDLE_PYTHON_DEPS in build_client)."""

import importlib.util
from importlib import metadata
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# tasks.py lives at the tensilelite root (toxinidir), four levels up from this
# file: Tensile/Tests/unit/<this file>. Load it by path so the test does not
# depend on the root being importable via sys.path.
_TASKS_PY = Path(__file__).parents[3] / "tasks.py"


def _load_tasks():
    spec = importlib.util.spec_from_file_location("tensilelite_tasks_under_test", _TASKS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tasks = _load_tasks()


class _FakeDist:
    def __init__(self, direct_url_text):
        self._direct_url_text = direct_url_text

    def read_text(self, name):
        assert name == "direct_url.json"
        return self._direct_url_text


def _patch_distribution(monkeypatch, result):
    """Patch importlib.metadata.distribution. `result` is either an exception
    instance to raise or a _FakeDist to return."""

    def fake_distribution(name):
        assert name == "rocisa"
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(metadata, "distribution", fake_distribution)


def test_absent_when_package_not_found(monkeypatch):
    _patch_distribution(monkeypatch, metadata.PackageNotFoundError("rocisa"))
    assert tasks._rocisa_install_status() == "absent"


def test_non_editable_when_no_direct_url(monkeypatch):
    # A normal pip/wheel install has no direct_url.json (read_text returns None).
    _patch_distribution(monkeypatch, _FakeDist(None))
    assert tasks._rocisa_install_status() == "non-editable"


def test_editable_when_dir_info_editable_true(monkeypatch):
    _patch_distribution(
        monkeypatch,
        _FakeDist('{"url": "file:///src/rocisa", "dir_info": {"editable": true}}'),
    )
    assert tasks._rocisa_install_status() == "editable"


def test_non_editable_when_dir_info_editable_false(monkeypatch):
    _patch_distribution(
        monkeypatch,
        _FakeDist('{"url": "file:///src/rocisa", "dir_info": {"editable": false}}'),
    )
    assert tasks._rocisa_install_status() == "non-editable"


def test_non_editable_when_dir_info_missing(monkeypatch):
    # direct_url.json present (e.g. VCS/archive install) but no dir_info block.
    _patch_distribution(monkeypatch, _FakeDist('{"url": "https://example/rocisa.whl"}'))
    assert tasks._rocisa_install_status() == "non-editable"


def test_non_editable_on_malformed_direct_url(monkeypatch):
    _patch_distribution(monkeypatch, _FakeDist("not-valid-json"))
    assert tasks._rocisa_install_status() == "non-editable"
