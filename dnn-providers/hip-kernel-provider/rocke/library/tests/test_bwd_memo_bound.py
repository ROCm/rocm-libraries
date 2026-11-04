# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pins the backward-test memo bound in ``tests/conftest.py`` (no GPU, no kernel).

Fake modules and items drive the knob parser, the module match, the grouping
by parameter set and the teardown clear, so a change to any of them shows up
here rather than as an unbounded or reordered run.
"""

import functools
import os
import sys
import types
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent


def _conftest():
    """The conftest module pytest loaded for this directory."""
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None) or ""
        if path and Path(path).resolve() == _TESTS / "conftest.py":
            return mod
    raise RuntimeError("tests/conftest.py is not loaded")


C = _conftest()


def _module(name: str, directory: Path = _TESTS, memo_entries: int = 0):
    """A fake test module in ``directory`` with one memo holding
    ``memo_entries`` entries (no memo when ``memo_entries`` is None)."""
    mod = types.ModuleType(f"fake_{name}")
    mod.__file__ = str(directory / f"{name}.py")
    if memo_entries is not None:

        def build(key):
            return key

        build.__module__ = mod.__name__
        mod.build = functools.cache(build)
        for k in range(memo_entries):
            mod.build(k)
    return mod


class _Item:
    def __init__(self, mod, func: str, param=None):
        self.module = mod
        stem = Path(mod.__file__).name
        self.nodeid = f"tests/{stem}::{func}" + (f"[{param}]" if param else "")
        if param is not None:
            self.callspec = types.SimpleNamespace(id=param)


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(C, "_memo_modules", {})
    monkeypatch.setattr(C, "_BWD_MEMO_MAX", 8)
    return C


@pytest.mark.parametrize(
    "raw, want",
    [(None, 8), ("", 8), ("  ", 8), ("0", 0), ("-1", -1), ("16", 16), (" 3 ", 3)],
)
def test_memo_max_parses_the_knob(raw, want):
    assert C._memo_max(raw) == want


@pytest.mark.parametrize("raw", ["off", "8.5", "eight"])
def test_memo_max_refuses_a_non_integer_with_the_knob_name(raw):
    with pytest.raises(pytest.UsageError, match="ROCKE_BWD_MEMO_MAX"):
        C._memo_max(raw)


def test_bwd_module_matches_by_prefix_and_directory(tmp_path):
    assert C._is_bwd_test_module(_module("test_attention_bwd_x"))
    assert C._is_bwd_test_module(_module("test_attention_dense_bwd_x"))
    assert not C._is_bwd_test_module(_module("test_attention_fwd_x"))
    assert not C._is_bwd_test_module(_module("test_attention_bwd_x", tmp_path))
    assert not C._is_bwd_test_module(types.ModuleType("no_file"))
    assert not C._is_bwd_test_module(None)


def test_bwd_module_matches_through_a_symlinked_checkout(tmp_path):
    link = tmp_path / "link"
    try:
        os.symlink(_TESTS, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    assert C._is_bwd_test_module(_module("test_attention_bwd_x", link))


def test_grouping_runs_one_parameter_set_back_to_back(fresh):
    bwd = _module("test_attention_bwd_x")
    plain = _module("test_attention_fwd_x")
    items = [
        _Item(bwd, "test_a", "p1"),
        _Item(bwd, "test_a", "p2"),
        _Item(plain, "test_c", "p1"),
        _Item(plain, "test_c", "p2"),
        _Item(bwd, "test_b", "p1"),
        _Item(plain, "test_d", "p1"),
        _Item(bwd, "test_b", "p2"),
    ]
    C._group_by_parameter_set(items)
    assert [i.nodeid for i in items] == [
        "tests/test_attention_bwd_x.py::test_a[p1]",
        "tests/test_attention_bwd_x.py::test_b[p1]",
        "tests/test_attention_bwd_x.py::test_a[p2]",
        "tests/test_attention_bwd_x.py::test_b[p2]",
        "tests/test_attention_fwd_x.py::test_c[p1]",
        "tests/test_attention_fwd_x.py::test_c[p2]",
        "tests/test_attention_fwd_x.py::test_d[p1]",
    ]
    assert list(C._memo_modules) == [bwd]


def test_grouping_leaves_a_bwd_module_without_memo_alone(fresh):
    nomemo = _module("test_attention_bwd_y", memo_entries=None)
    items = [
        _Item(nomemo, "test_a", "p1"),
        _Item(nomemo, "test_b", "p2"),
        _Item(nomemo, "test_a", "p2"),
    ]
    before = [i.nodeid for i in items]
    C._group_by_parameter_set(items)
    assert [i.nodeid for i in items] == before
    assert C._memo_modules == {}


def test_zero_turns_off_grouping_and_recording(fresh, monkeypatch):
    monkeypatch.setattr(C, "_BWD_MEMO_MAX", 0)
    bwd = _module("test_attention_bwd_x")
    items = [
        _Item(bwd, "test_a", "p1"),
        _Item(bwd, "test_a", "p2"),
        _Item(bwd, "test_b", "p1"),
    ]
    before = [i.nodeid for i in items]
    C._group_by_parameter_set(items)
    assert [i.nodeid for i in items] == before
    assert C._memo_modules == {}


def test_teardown_clears_only_at_the_end_of_a_parameter_set(fresh):
    bwd = _module("test_attention_bwd_x", memo_entries=12)
    small = _module("test_attention_bwd_y", memo_entries=3)
    a1, b1, a2 = (
        _Item(bwd, "test_a", "p1"),
        _Item(bwd, "test_b", "p1"),
        _Item(bwd, "test_a", "p2"),
    )
    C._group_by_parameter_set([a1, b1, a2, _Item(small, "test_s", "p1")])

    C.pytest_runtest_teardown(a1, b1)  # same parameter set: keep
    assert bwd.build.cache_info().currsize == 12
    C.pytest_runtest_teardown(b1, a2)  # set ends: clear the memo over the bound
    assert bwd.build.cache_info().currsize == 0
    assert small.build.cache_info().currsize == 3  # at or under it: kept


def test_teardown_clears_after_the_last_test(fresh):
    bwd = _module("test_attention_bwd_x", memo_entries=12)
    last = _Item(bwd, "test_a", "p1")
    C._group_by_parameter_set([last])
    C.pytest_runtest_teardown(last, None)
    assert bwd.build.cache_info().currsize == 0


def test_teardown_ignores_unrecorded_modules(fresh):
    bwd = _module("test_attention_bwd_x", memo_entries=12)
    other = _module("test_attention_fwd_x")
    C._group_by_parameter_set([_Item(bwd, "test_a", "p1")])
    C.pytest_runtest_teardown(_Item(other, "test_c", "p1"), None)
    assert bwd.build.cache_info().currsize == 12
