# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.RocisaStatus._rocisa_install_status (the three-way
rocisa detection that drives auto-enabling HIPBLASLT_BUNDLE_PYTHON_DEPS in
tasks.build_client)."""

from importlib import metadata

import pytest

# _rocisa_install_status ships inside the Tensile package, so it imports the
# same way in the source tree and in the installed test artifacts (the Tests
# conftest puts the tensilelite root on sys.path). Importing it here avoids the
# `invoke` dependency that loading tasks.py by path used to pull in.
from Tensile.RocisaStatus import _rocisa_install_status

pytestmark = pytest.mark.unit


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
    assert _rocisa_install_status() == "absent"


def test_non_editable_when_no_direct_url(monkeypatch):
    # A normal pip/wheel install has no direct_url.json (read_text returns None).
    _patch_distribution(monkeypatch, _FakeDist(None))
    assert _rocisa_install_status() == "non-editable"


def test_editable_when_dir_info_editable_true(monkeypatch):
    _patch_distribution(
        monkeypatch,
        _FakeDist('{"url": "file:///src/rocisa", "dir_info": {"editable": true}}'),
    )
    assert _rocisa_install_status() == "editable"


def test_non_editable_when_dir_info_editable_false(monkeypatch):
    _patch_distribution(
        monkeypatch,
        _FakeDist('{"url": "file:///src/rocisa", "dir_info": {"editable": false}}'),
    )
    assert _rocisa_install_status() == "non-editable"


def test_non_editable_when_dir_info_missing(monkeypatch):
    # direct_url.json present (e.g. VCS/archive install) but no dir_info block.
    _patch_distribution(monkeypatch, _FakeDist('{"url": "https://example/rocisa.whl"}'))
    assert _rocisa_install_status() == "non-editable"


def test_non_editable_on_malformed_direct_url(monkeypatch):
    _patch_distribution(monkeypatch, _FakeDist("not-valid-json"))
    assert _rocisa_install_status() == "non-editable"
