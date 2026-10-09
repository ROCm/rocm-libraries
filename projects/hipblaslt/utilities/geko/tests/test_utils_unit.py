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


def test_rocm_subprocess_env_adds_rocm_bins_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm-custom")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = utils._rocm_subprocess_env()

    assert env["ROCM_PATH"] == "/opt/rocm-custom"
    assert env["PATH"].startswith("/opt/rocm-custom/bin:/opt/rocm-custom/hip/bin:/opt/rocm-custom/llvm/bin")
    assert "/usr/bin" in env["PATH"]


def test_rocm_subprocess_env_defaults_rocm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROCM_PATH", raising=False)

    env = utils._rocm_subprocess_env()

    assert env["ROCM_PATH"] == "/opt/rocm"
    assert "/opt/rocm/bin" in env["PATH"]


def test_run_silent_command_uses_rocm_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _Proc:
        returncode = 0

        def communicate(self):
            return "", ""

    def _fake_popen(cmd, stdout=None, stderr=None, cwd=None, text=None, env=None):
        captured["env"] = env
        return _Proc()

    monkeypatch.setattr(utils.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(utils, "_rocm_subprocess_env", lambda: {"SENTINEL": "1"})

    utils.run_silent_command(["x"])

    assert captured["env"] == {"SENTINEL": "1"}


def test_run_silent_command_uses_explicit_env_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _Proc:
        returncode = 0

        def communicate(self):
            return "", ""

    def _fake_popen(cmd, stdout=None, stderr=None, cwd=None, text=None, env=None):
        captured["env"] = env
        return _Proc()

    monkeypatch.setattr(utils.subprocess, "Popen", _fake_popen)

    utils.run_silent_command(["x"], env={"CUSTOM": "1"})

    assert captured["env"] == {"CUSTOM": "1"}


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


def test_build_tensilelite_client_passes_gpu_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    tensile = hip / "tensilelite"
    tensile.mkdir(parents=True)

    build_dir = tmp_path / "build"
    client = build_dir / "tensilelite/client/tensilelite-client"

    monkeypatch.setattr(utils, "find_spec", lambda _n: object())

    captured = {}

    def _fake_run(cmd, cwd=None):
        captured["cmd"] = cmd
        client.parent.mkdir(parents=True, exist_ok=True)
        client.write_text("bin\n")

    monkeypatch.setattr(utils, "run_silent_command", _fake_run)

    utils.build_tensilelite_client(hip, build_dir=build_dir, gpu_targets="gfx950")

    assert "--gpu-targets" in captured["cmd"]
    assert "gfx950" in captured["cmd"]


def test_build_tensilelite_client_omits_gpu_targets_when_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    tensile = hip / "tensilelite"
    tensile.mkdir(parents=True)

    build_dir = tmp_path / "build"
    client = build_dir / "tensilelite/client/tensilelite-client"

    monkeypatch.setattr(utils, "find_spec", lambda _n: object())

    captured = {}

    def _fake_run(cmd, cwd=None):
        captured["cmd"] = cmd
        client.parent.mkdir(parents=True, exist_ok=True)
        client.write_text("bin\n")

    monkeypatch.setattr(utils, "run_silent_command", _fake_run)

    utils.build_tensilelite_client(hip, build_dir=build_dir)

    assert "--gpu-targets" not in captured["cmd"]
