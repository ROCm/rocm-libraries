# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from geko import paths


def test_looks_like_and_is_built(tmp_path: Path) -> None:
    root = tmp_path / "hip"
    root.mkdir()
    assert paths.looks_like_hipblaslt_root(root) is False
    assert paths.is_hipblaslt_built(root) is False

    (root / "tensilelite").mkdir()
    assert paths.looks_like_hipblaslt_root(root) is True

    (root / "build/release").mkdir(parents=True)
    assert paths.is_hipblaslt_built(root) is True


def test_resolve_explicit_and_env_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    env_root.mkdir()
    (env_root / "tensilelite").mkdir()

    exp_root = tmp_path / "exp"
    exp_root.mkdir()
    (exp_root / "tensilelite").mkdir()

    monkeypatch.setenv(paths.HIPBLASLT_PATH_ENV_VAR, str(env_root))
    out = paths.resolve_hipblaslt_path(explicit=str(exp_root), anchor=None)
    assert out == exp_root.resolve()


def test_resolve_from_env_and_anchor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_root = tmp_path / "env"
    env_root.mkdir()
    (env_root / "tensilelite").mkdir()
    monkeypatch.setenv(paths.HIPBLASLT_PATH_ENV_VAR, str(env_root))
    out = paths.resolve_hipblaslt_path(anchor=None)
    assert out == env_root.resolve()

    monkeypatch.delenv(paths.HIPBLASLT_PATH_ENV_VAR, raising=False)
    anchor_root = tmp_path / "repo"
    anchor_file = anchor_root / "a/b/c.py"
    anchor_file.parent.mkdir(parents=True)
    anchor_file.write_text("x", encoding="utf-8")
    (anchor_root / "tensilelite").mkdir()
    out2 = paths.resolve_hipblaslt_path(anchor=anchor_file)
    assert out2 == anchor_root.resolve()


def test_resolve_errors_and_require_built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(SystemExit):
        paths.resolve_hipblaslt_path(explicit=bad)

    monkeypatch.delenv(paths.HIPBLASLT_PATH_ENV_VAR, raising=False)
    with pytest.raises(SystemExit):
        # anchor=None (not anchor=tmp_path/...) because on Windows CI tox sets
        # basetemp inside the repo; the parent walk would find projects/hipblaslt/tensilelite
        # and resolve successfully instead of raising.
        paths.resolve_hipblaslt_path(anchor=None)

    not_built = tmp_path / "nb"
    not_built.mkdir()
    (not_built / "tensilelite").mkdir()
    with pytest.raises(SystemExit):
        paths.resolve_hipblaslt_path(explicit=not_built, require_built=True)
