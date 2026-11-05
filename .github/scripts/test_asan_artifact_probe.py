# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import io
import json
import shutil
import sys
import tarfile
import time
import zipfile

import pytest

import asan_artifact_probe as probe


@pytest.mark.parametrize("mutation", ["truncate", "same_length_corruption"])
def test_rejects_corrupt_archive(tmp_path, mutation):
    original = b"fixed archive bytes"
    payload = tmp_path / "payload.zip"
    payload.write_bytes(
        original[:-1] if mutation == "truncate" else b"X" + original[1:]
    )
    with pytest.raises(ValueError, match="mismatch"):
        probe.checked_archive(
            payload, len(original), hashlib.sha256(original).hexdigest()
        )


@pytest.mark.parametrize("mutation", ["content", "mode", "symlink", "missing", "extra"])
def test_tree_inventory_detects_nonidentical_install(tmp_path, mutation):
    binary = tmp_path / "test"
    binary.write_bytes(b"test binary")
    binary.chmod(0o755)
    link = tmp_path / "link"
    link.symlink_to("test")
    original = probe.inventory(tmp_path)
    if mutation == "content":
        binary.write_bytes(b"other bytes")
    elif mutation == "mode":
        binary.chmod(0o644)
    elif mutation == "symlink":
        link.unlink()
        link.symlink_to("missing")
    elif mutation == "missing":
        binary.unlink()
    else:
        (tmp_path / "extra").write_text("unselected")
    with pytest.raises(ValueError, match="Tree mismatch"):
        probe.compare_inventory(original, probe.inventory(tmp_path))


def test_redirect_keeps_credentials_out_of_results(tmp_path, monkeypatch, capsys):
    work = tmp_path / "work" / "B"
    work.mkdir(parents=True)
    (work / "headers").write_text(
        "HTTP/1.1 302 Found\r\nLocation: https://example.com/archive?signature=secret\r\n\r\n"
    )
    (work / "api.curl").write_text("private-token")
    monkeypatch.setattr(
        sys, "argv", ["probe", "redirect", "--root", str(tmp_path), "--arm", "B"]
    )
    probe.main()
    assert "signature=secret" in (work / "blob.curl").read_text()
    assert (work / "blob.curl").stat().st_mode & 0o777 == 0o600
    assert not (work / "headers").exists()
    assert not (work / "api.curl").exists()
    assert not (tmp_path / "results").exists()
    assert capsys.readouterr().out == ""


def test_failure_is_recorded_and_summary_fails(tmp_path, monkeypatch):
    for arm in ["A", "B", "C"]:
        probe.write_json(
            tmp_path / "results" / arm / "result.json",
            {"arm": arm, "verified": arm != "B"},
        )
    monkeypatch.setattr(sys, "argv", ["probe", "summary", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 1
    states = json.loads((tmp_path / "results" / "summary.json").read_text())["arms"]
    assert len(states) == 3
    assert next(s for s in states if s["arm"] == "B")["verified"] is False


def test_successful_summary_preserves_all_measurements(tmp_path, monkeypatch):
    for arm in ["A", "B", "C"]:
        probe.write_json(
            tmp_path / "results" / arm / "result.json",
            {"arm": arm, "verified": True, "ready_seconds": 12},
        )
    monkeypatch.setattr(sys, "argv", ["probe", "summary", "--root", str(tmp_path)])
    probe.main()
    states = json.loads((tmp_path / "results" / "summary.json").read_text())["arms"]
    assert [s["arm"] for s in states] == ["A", "B", "C"]
    assert [s["ready_seconds"] for s in states] == [12, 12, 12]


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_seed_and_consumer_verify_the_same_tree(tmp_path, monkeypatch, arm):
    seed = tmp_path / "work" / "seed"
    (seed / "trees").mkdir(parents=True)
    contents = {
        "bin/hipblaslt-test": b"#!/bin/sh\necho ExampleSuite.ExampleTest\n",
        "lib/llvm/bin/llvm-symbolizer": b"#!/bin/sh\nexit 0\n",
        "lib/llvm/lib/clang/23/lib/linux/libclang_rt.asan-x86_64.so": b"runtime",
        "lib/hipblaslt/library/kernel.co": b"kernel",
    }
    for member in probe.MEMBERS:
        with tarfile.open(seed / "trees" / member, "w:gz") as archive:
            for name, data in contents.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(seed / "payload.zip", "w") as archive:
        for member in probe.MEMBERS:
            archive.write(seed / "trees" / member, member)
    monkeypatch.setattr(probe, "ARTIFACT_SIZE", (seed / "payload.zip").stat().st_size)
    monkeypatch.setattr(probe, "ARTIFACT_SHA256", probe.sha256(seed / "payload.zip"))
    for name in ["seed", arm]:
        probe.write_json(
            tmp_path / "results" / name / "result.json",
            {
                "arm": name,
                "download_status": "success",
                "monotonic_start": time.monotonic(),
            },
        )
    monkeypatch.setattr(
        sys, "argv", ["probe", "verify", "--root", str(tmp_path), "--seed"]
    )
    probe.main()
    consumer = tmp_path / "work" / arm
    (consumer / "trees").mkdir(parents=True)
    if arm == "A":
        for member in probe.MEMBERS:
            shutil.copyfile(seed / "trees" / member, consumer / "trees" / member)
    else:
        shutil.copyfile(seed / "payload.zip", consumer / "payload.zip")
    monkeypatch.setenv("REFERENCE_SHA256", probe.sha256(tmp_path / "reference.json"))
    monkeypatch.setattr(
        sys, "argv", ["probe", "verify", "--root", str(tmp_path), "--arm", arm]
    )
    probe.main()
    result = json.loads((tmp_path / "results" / arm / "result.json").read_text())
    assert result["verified"] is True
    assert (
        tmp_path / "results" / arm / "test-discovery.txt"
    ).read_text() == "ExampleSuite.ExampleTest\n"
    assert (
        consumer / "installed" / "lib/hipblaslt/library/kernel.co"
    ).read_bytes() == b"kernel"
