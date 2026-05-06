# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for graph file resolution and tarball extraction."""

import io
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph.resolver import (
    _is_safe_member,
    extract_tarball,
    is_tarball,
    resolve_graph_files,
)


def _make_tarball(dest: Path, members: dict) -> Path:
    """Create a real tarball at dest containing the given {name: content} members."""
    with tarfile.open(str(dest), "w:gz") as tf:
        for name, content in members.items():
            data = content.encode() if isinstance(content, str) else content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return dest


class TestIsTarball:
    """Tests for is_tarball()."""

    def test_recognizes_tar_gz(self) -> None:
        assert is_tarball("graphs.tar.gz") is True

    def test_recognizes_tgz(self) -> None:
        assert is_tarball("graphs.tgz") is True

    def test_recognizes_tar_bz2(self) -> None:
        assert is_tarball("graphs.tar.bz2") is True

    def test_recognizes_tar(self) -> None:
        assert is_tarball("graphs.tar") is True

    def test_recognizes_tar_xz(self) -> None:
        assert is_tarball("graphs.tar.xz") is True

    def test_rejects_json(self) -> None:
        assert is_tarball("graph.json") is False

    def test_rejects_txt(self) -> None:
        assert is_tarball("shapes.txt") is False

    def test_case_insensitive(self) -> None:
        assert is_tarball("GRAPHS.TAR.GZ") is True


class TestIsSafeMember:
    """Tests for _is_safe_member()."""

    def test_safe_relative_path(self) -> None:
        m = tarfile.TarInfo(name="subdir/graph.json")
        assert _is_safe_member(m) is True

    def test_rejects_absolute_path(self) -> None:
        m = tarfile.TarInfo(name="/etc/evil.json")
        assert _is_safe_member(m) is False

    def test_rejects_dotdot_component(self) -> None:
        m = tarfile.TarInfo(name="../../evil.json")
        assert _is_safe_member(m) is False

    def test_rejects_embedded_dotdot(self) -> None:
        m = tarfile.TarInfo(name="subdir/../../../evil.json")
        assert _is_safe_member(m) is False

    def test_safe_plain_name(self) -> None:
        m = tarfile.TarInfo(name="graph.json")
        assert _is_safe_member(m) is True


class TestExtractTarball:
    """Tests for extract_tarball()."""

    def test_extracts_json_files(self, tmp_path: Path) -> None:
        graph = json.dumps({"name": "g", "nodes": [], "tensors": []})
        tb = _make_tarball(tmp_path / "graphs.tar.gz", {"graph.json": graph})

        tmpdir, extracted = extract_tarball(str(tb))
        try:
            assert len(extracted) == 1
            assert extracted[0].endswith(".json")
        finally:
            tmpdir.cleanup()

    def test_nonexistent_path_raises_graph_load_error(self) -> None:
        with pytest.raises(GraphLoadError, match="not found"):
            extract_tarball("/nonexistent/path/graphs.tar.gz")

    def test_not_a_tarball_raises_graph_load_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "fake.tar.gz"
        bad_file.write_text("this is not a tarball")
        with pytest.raises(GraphLoadError, match="Not a valid tarball"):
            extract_tarball(str(bad_file))

    def test_no_json_in_tarball_raises_graph_load_error(self, tmp_path: Path) -> None:
        tb = _make_tarball(tmp_path / "empty.tar.gz", {"readme.txt": "hello"})
        with pytest.raises(GraphLoadError, match="No .json files"):
            extract_tarball(str(tb))

    def test_path_traversal_member_is_filtered(self, tmp_path: Path) -> None:
        # A tarball that contains a path-traversal entry alongside a safe one.
        # The traversal member should be silently skipped.
        graph = json.dumps({"name": "g", "nodes": [], "tensors": []})
        tb = tmp_path / "graphs.tar.gz"
        with tarfile.open(str(tb), "w:gz") as tf:
            # Safe member
            data = graph.encode()
            info = tarfile.TarInfo(name="safe.json")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            # Traversal member — should be silently skipped
            evil = b'{"evil": true}'
            info2 = tarfile.TarInfo(name="../../evil.json")
            info2.size = len(evil)
            tf.addfile(info2, io.BytesIO(evil))

        tmpdir, extracted = extract_tarball(str(tb))
        try:
            # Only the safe member should have been extracted
            assert len(extracted) == 1
            assert "evil" not in extracted[0]
        finally:
            tmpdir.cleanup()


class TestResolveGraphFiles:
    """Tests for resolve_graph_files()."""

    def test_single_json_file(self, tmp_path: Path) -> None:
        f = tmp_path / "graph.json"
        f.write_text(json.dumps({"name": "g", "nodes": [], "tensors": []}))

        tmpdirs, files, tarball_source = resolve_graph_files(str(f))
        assert tmpdirs == []
        assert len(files) == 1
        assert files[0] == str(f)
        assert tarball_source is None

    def test_glob_matches_multiple_json_files(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"g{i}.json").write_text(
                json.dumps({"name": f"g{i}", "nodes": [], "tensors": []})
            )
        pattern = str(tmp_path / "*.json")

        tmpdirs, files, _ = resolve_graph_files(pattern)
        assert len(files) == 3
        assert all(f.endswith(".json") for f in files)
        for td in tmpdirs:
            td.cleanup()

    def test_directory_collects_json_recursively(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        for name, parent in [("a.json", tmp_path), ("b.json", subdir)]:
            (parent / name).write_text(
                json.dumps({"name": name, "nodes": [], "tensors": []})
            )
        (tmp_path / "readme.txt").write_text("not a graph")

        tmpdirs, files, tarball_source = resolve_graph_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".json") for f in files)
        assert tarball_source is None
        assert tmpdirs == []

    def test_directory_collects_tarballs_and_json(self, tmp_path: Path) -> None:
        (tmp_path / "loose.json").write_text(
            json.dumps({"name": "loose", "nodes": [], "tensors": []})
        )
        graph = json.dumps({"name": "packed", "nodes": [], "tensors": []})
        _make_tarball(tmp_path / "packed.tar.gz", {"packed.json": graph})

        tmpdirs, files, tarball_source = resolve_graph_files(str(tmp_path))
        try:
            assert len(files) == 2
            assert any("loose.json" in f for f in files)
            assert any("packed.json" in f for f in files)
            assert tarball_source == str(tmp_path)
        finally:
            for td in tmpdirs:
                td.cleanup()

    def test_empty_directory_returns_no_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        tmpdirs, files, tarball_source = resolve_graph_files(str(empty))
        assert files == []
        assert tmpdirs == []

    def test_tarball_path_extracts_and_returns_json(self, tmp_path: Path) -> None:
        graph = json.dumps({"name": "g", "nodes": [], "tensors": []})
        tb = _make_tarball(tmp_path / "graphs.tar.gz", {"graph.json": graph})

        tmpdirs, files, tarball_source = resolve_graph_files(str(tb))
        try:
            assert len(files) == 1
            assert files[0].endswith(".json")
            assert tarball_source == str(tb)
        finally:
            for td in tmpdirs:
                td.cleanup()
