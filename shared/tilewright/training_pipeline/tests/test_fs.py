# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import stat

import pytest

from lib import fs


@pytest.fixture
def umask():
    old = os.umask(0o022)
    yield
    os.umask(old)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_atomic_write_uses_umask_permissions(tmp_path, umask):
    target = tmp_path / "sub" / "out.txt"
    fs.atomic_write_text(target, "hello\n")
    assert target.read_text() == "hello\n"
    assert _mode(target) == 0o644
    os.umask(0o077)
    fs.atomic_write_bytes(tmp_path / "private.bin", b"\x00\x01")
    assert _mode(tmp_path / "private.bin") == 0o600


def test_atomic_write_replaces_and_leaves_no_temp_files(tmp_path, umask):
    target = tmp_path / "out.txt"
    target.write_text("old")
    fs.atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_atomic_write_fsyncs_file_and_directory(tmp_path, monkeypatch, umask):
    kinds = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        kinds.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(fs.os, "fsync", recording_fsync)
    fs.atomic_write_text(tmp_path / "out.txt", "x")
    assert kinds == ["file", "dir"]


def test_failed_write_keeps_old_file_and_cleans_up(tmp_path, monkeypatch, umask):
    target = tmp_path / "out.txt"
    target.write_text("intact")

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(fs.os, "replace", failing_replace)
    with pytest.raises(OSError):
        fs.atomic_write_text(target, "new")
    assert target.read_text() == "intact"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_json_round_trip(tmp_path, umask):
    obj = {"a": [1, 2, {"b": None}], "c": "d"}
    fs.write_json(tmp_path / "x.json", obj)
    assert fs.read_json(tmp_path / "x.json") == obj
    assert (tmp_path / "x.json").read_text().endswith("\n")
