# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from geko import utils


def test_get_utc_timestamp_and_sha256(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("abc")
    ts = utils.get_utc_timestamp()
    assert "T" in ts
    assert utils.compute_file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_run_silent_command_raises_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Proc:
        returncode = 1

        def communicate(self):
            return "", "bad"

    monkeypatch.setattr(utils.subprocess, "Popen", lambda *_a, **_k: _Proc())
    with pytest.raises(ValueError, match="bad"):
        utils.run_silent_command(["x"])


def test_parse_devices_variants_and_errors() -> None:
    out = utils.parse_devices("0,1,1")
    assert set(out) == {0, 1}

    out2 = utils.parse_devices([2, 3])
    assert out2 == [2, 3]

    with pytest.raises(ValueError, match="Error parsing devices"):
        utils.parse_devices("x,y")

    with pytest.raises(ValueError, match="not supported"):
        utils.parse_devices(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Need at least 1 device"):
        utils.parse_devices([])


def test_build_tensilelite_client_missing_hip_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        utils.build_tensilelite_client(tmp_path / "missing")


def test_build_tensilelite_client_raises_without_invoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    (hip / "tensilelite").mkdir(parents=True)

    monkeypatch.setattr(utils, "find_spec", lambda _n: None)
    monkeypatch.setattr(utils, "run_silent_command", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="invoke"):
        utils.build_tensilelite_client(hip, build_dir=tmp_path / "b")


def test_build_tensilelite_client_build_and_cached_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    tensile = hip / "tensilelite"
    tensile.mkdir(parents=True)

    build_dir = tmp_path / "build"
    client = build_dir / "tensilelite/client/tensilelite-client"
    hash_file = build_dir / "hash.txt"
    client.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(utils, "find_spec", lambda _n: object())

    built = {"n": 0}

    def _fake_run(_cmd, cwd=None):
        built["n"] += 1
        client.parent.mkdir(parents=True, exist_ok=True)
        client.write_text("bin\n")

    monkeypatch.setattr(utils, "run_silent_command", _fake_run)

    out1 = utils.build_tensilelite_client(hip, build_dir=build_dir)
    assert out1 == client
    assert built["n"] == 1
    assert hash_file.is_file()

    out2 = utils.build_tensilelite_client(hip, build_dir=build_dir)
    assert out2 == client
    assert built["n"] == 1
