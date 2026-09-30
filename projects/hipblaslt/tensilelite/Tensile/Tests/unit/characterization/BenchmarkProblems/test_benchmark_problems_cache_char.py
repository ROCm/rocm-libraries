################################################################################
# Characterization tests for Tensile.BenchmarkProblems — cache helper layer.
#
# ADD-ONLY. BenchmarkProblems.py is a benchmark build/run orchestrator that pulls
# in KernelWriter / ClientWriter / KernelWriterAssembly / Assembler (codegen +
# GPU), which is out of scope for this effort. This suite pins the pure cache
# helper layer (_cacheDataMatches / _computeCacheKey / _readCacheIfValid /
# _loadCacheIfMatches / _loadLegacyCacheIfMatches / _resetCacheDir). The build/
# run path (writeBenchmarkFiles, _benchmarkProblemType, _generate*Solutions,
# main) is documented resistance — see target.md.
################################################################################
import importlib
import os
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.BenchmarkProblems")


def _step(**over):
    base = dict(
        constantParams={"a": 1},
        forkParams={"b": 2},
        paramGroups=["g"],
        customKernels=[],
        internalSupportParams={"x": True},
        customKernelWildcard=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _cache_dict(step, codeObjects=("k.co",), libraryFile="lib.dat"):
    d = {f: getattr(step, attr) for f, attr in M._CACHE_FIELDS.items()}
    d["CodeObjectFiles"] = list(codeObjects)
    d["LibraryFile"] = libraryFile
    return d


# ---------------------------------------------------------------------------
# _cacheDataMatches
# ---------------------------------------------------------------------------
def test_cache_data_matches_true():
    step = _step()
    assert M._cacheDataMatches(_cache_dict(step), step) is True


def test_cache_data_matches_false():
    step = _step()
    other = _cache_dict(_step(forkParams={"b": 999}))
    assert M._cacheDataMatches(other, step) is False


# ---------------------------------------------------------------------------
# _computeCacheKey
# ---------------------------------------------------------------------------
def test_compute_cache_key_deterministic_and_len():
    step = _step()
    k1 = M._computeCacheKey(step)
    k2 = M._computeCacheKey(step)
    assert k1 == k2
    assert len(k1) == M._CACHE_KEY_LEN
    assert all(c in "0123456789abcdef" for c in k1)


def test_compute_cache_key_changes_with_params():
    assert M._computeCacheKey(_step()) != M._computeCacheKey(_step(constantParams={"a": 2}))


# ---------------------------------------------------------------------------
# _readCacheIfValid
# ---------------------------------------------------------------------------
def test_read_cache_missing_file_returns_none(tmp_path):
    assert M._readCacheIfValid(str(tmp_path / "nope.yaml"), _step(), "m") is None


def test_read_cache_match_returns_code_objects(tmp_path, monkeypatch):
    step = _step()
    path = tmp_path / "cache.yaml"
    path.write_text("placeholder")  # must exist
    monkeypatch.setattr(M.LibraryIO, "read", lambda p: _cache_dict(step, ["a.co", "b.co"]))
    assert M._readCacheIfValid(str(path), step, "m") == {
        "CodeObjectFiles": ["a.co", "b.co"],
        "LibraryFile": "lib.dat",
    }


def test_read_cache_unreadable_returns_none(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cache.yaml"
    path.write_text("x")

    def boom(p):
        raise OSError("bad")

    monkeypatch.setattr(M.LibraryIO, "read", boom)
    assert M._readCacheIfValid(str(path), _step(), "m") is None
    assert "unreadable cache" in capsys.readouterr().out


def test_read_cache_missing_field_returns_none(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cache.yaml"
    path.write_text("x")
    monkeypatch.setattr(M.LibraryIO, "read", lambda p: {"CodeObjectFiles": []})  # missing fields
    assert M._readCacheIfValid(str(path), _step(), "m") is None
    assert "incompatible cache" in capsys.readouterr().out


def test_read_cache_mismatch_returns_none(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cache.yaml"
    path.write_text("x")
    step = _step()
    monkeypatch.setattr(M.LibraryIO, "read", lambda p: _cache_dict(_step(paramGroups=["other"])))
    assert M._readCacheIfValid(str(path), step, "MISMATCH {path}") is None
    assert "MISMATCH" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _loadCacheIfMatches / _loadLegacyCacheIfMatches
# ---------------------------------------------------------------------------
def test_load_cache_if_matches(tmp_path, monkeypatch):
    step = _step()
    (tmp_path / "cache.yaml").write_text("x")
    monkeypatch.setattr(M.LibraryIO, "read", lambda p: _cache_dict(step, ["z.co"]))
    assert M._loadCacheIfMatches(str(tmp_path), step) == {
        "CodeObjectFiles": ["z.co"],
        "LibraryFile": "lib.dat",
    }


def test_load_legacy_cache_if_matches(tmp_path, monkeypatch):
    step = _step()
    (tmp_path / "cache.yaml").write_text("x")
    monkeypatch.setattr(M.LibraryIO, "read", lambda p: _cache_dict(step, ["legacy.co"]))
    assert M._loadLegacyCacheIfMatches(str(tmp_path), step) == {
        "CodeObjectFiles": ["legacy.co"],
        "LibraryFile": "lib.dat",
    }


# ---------------------------------------------------------------------------
# _resetCacheDir
# ---------------------------------------------------------------------------
def test_reset_cache_dir_creates(tmp_path):
    d = tmp_path / "cache"
    M._resetCacheDir(str(d))
    assert d.is_dir()


def test_reset_cache_dir_wipes_existing(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "stale.txt").write_text("old")
    M._resetCacheDir(str(d))
    assert d.is_dir()
    assert not (d / "stale.txt").exists()  # wiped
