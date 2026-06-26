################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Component``: the pure ``PartialMatch``
matcher and the ``Component`` registry/search surface
(``matches``/``findAll``/``find``/``componentPath``/``commentHeader``).

An isolated ``_CharBase`` hierarchy (private, under ``Component``) is defined so
``find``/``findAll``/``matches``/``versions`` can be driven deterministically
(single-match, >1-match RuntimeError, version filter). It is benign: production
code searches via ``Component.<RealSubtype>.find``, never ``Component.findAll``,
so these private impls never appear in a real search. See DECISIONS.md (D3)."""

import types

import pytest

from Tensile.Component import Component, PartialMatch

# NOTE: Component.py ends with `from .Components import *`, which rebinds the
# module-level name `LocalRead` to the Components.LocalRead submodule. The real
# base class is the `Component.LocalRead` attribute set by the metaclass.
LocalRead = Component.LocalRead

pytestmark = pytest.mark.unit


def _writer(version=(9, 4, 2), asmCaps=None, archCaps=None, kernel=None):
    states = types.SimpleNamespace(asmCaps=asmCaps or {}, archCaps=archCaps or {},
                                   kernel=kernel or {})
    return types.SimpleNamespace(version=version, states=states)


# --- isolated test hierarchy (registered once at import) --------------------

class _CharBase(Component):  # abstract (no __call__) -> a fresh implementations registry
    pass


class _CharImplCap(_CharBase):
    asmCaps = {"capX": True}

    def __call__(self):
        return "cap"


class _CharImplVersioned(_CharBase):
    versions = [(9, 4, 2)]

    def __call__(self):
        return "ver"


# A separate hierarchy with a nested *abstract* level, to drive findAll's
# "recurse into abstract impl" branch in isolation from the _CharBase tests.
class _CharNestBase(Component):
    pass


class _CharNestMid(_CharNestBase):  # abstract (no __call__)
    pass


class _CharNestLeaf(_CharNestMid):
    asmCaps = {"capZ": True}

    def __call__(self):
        return "leaf"


# ===========================================================================
# PartialMatch
# ===========================================================================

def test_partial_match_callable_true_false():
    assert PartialMatch(lambda o: o > 0, 5) is True
    assert PartialMatch(lambda o: o > 0, -1) is False


def test_partial_match_mapping():
    assert PartialMatch({"a": 1}, {"a": 1, "b": 2}) is True       # subset match
    assert PartialMatch({"a": 1}, {"b": 2}) is False              # key missing
    assert PartialMatch({"a": 1}, {"a": 2}) is False              # value mismatch


def test_partial_match_nested_and_scalar():
    assert PartialMatch({"x": {"y": 1}}, {"x": {"y": 1, "z": 2}}) is True
    assert PartialMatch(5, 5) is True
    assert PartialMatch(5, 6) is False


def test_partial_match_debug_paths():
    # debug=True walks the print branches; behaviour is unchanged.
    assert PartialMatch({"a": 1}, {"a": 1}, debug=True) is True
    assert PartialMatch({"a": 1}, {"b": 2}, debug=True) is False
    assert PartialMatch(lambda o: False, 1, debug=True) is False
    assert PartialMatch(1, 2, debug=True) is False


# ===========================================================================
# matches / versions
# ===========================================================================

def test_matches_capability():
    # asmCaps pattern present in the writer -> match; absent -> mismatch.
    assert _CharImplCap.matches(_writer(asmCaps={"capX": True})) is True
    assert _CharImplCap.matches(_writer(asmCaps={"capX": False})) is False


def test_matches_versions():
    assert _CharImplVersioned.matches(_writer(version=(9, 4, 2))) is True
    assert _CharImplVersioned.matches(_writer(version=(9, 0, 10))) is False


def test_component_base_matches_true():
    # Component itself has no version/caps attrs -> always matches.
    assert Component.matches(_writer()) is True


# ===========================================================================
# findAll / find
# ===========================================================================

def test_find_single(snapshot):
    # Only _CharImplCap matches (wrong version excludes the versioned impl).
    w = _writer(version=(0, 0, 0), asmCaps={"capX": True})
    found = _CharBase.findAll(w)
    inst = _CharBase.find(w)
    assert {"num_found": len(found), "instance_call": inst()} == snapshot


def test_find_none():
    # Nothing matches -> find returns None.
    w = _writer(version=(0, 0, 0), asmCaps={"capX": False})
    assert _CharBase.find(w) is None
    assert _CharBase.findAll(w) == []


def test_find_multiple_raises():
    # Both impls match -> find raises RuntimeError.
    w = _writer(version=(9, 4, 2), asmCaps={"capX": True})
    assert len(_CharBase.findAll(w)) == 2
    with pytest.raises(RuntimeError):
        _CharBase.find(w)


def test_find_debug(snapshot):
    w = _writer(version=(0, 0, 0), asmCaps={"capX": True})
    assert len(_CharBase.findAll(w, debug=True)) == snapshot


def test_findall_recurses_into_abstract(snapshot):
    # _CharNestBase.implementations holds _CharNestMid (abstract) -> findAll
    # recurses into it and finds _CharNestLeaf.
    w = _writer(version=(0, 0, 0), asmCaps={"capZ": True})
    found = _CharNestBase.findAll(w, debug=True)  # debug -> the recursion print branch
    assert {"num_found": len(found), "leaf_call": _CharNestBase.find(w)()} == snapshot


# ===========================================================================
# LocalRead helpers (called unbound with stubs — no registry pollution)
# ===========================================================================

def test_localread_get_lds_read_mem_token():
    writer = types.SimpleNamespace(states=types.SimpleNamespace(ldsReadTokenIdx=3))
    token, idx = LocalRead._getLdsReadMemToken(types.SimpleNamespace(), writer, None, None)
    assert idx == 3


def test_localread_emit_lds_read():
    added = []

    class _Inst:
        def __init__(self, **kw):
            self.kw = kw
            self.tok = None

        def setMemToken(self, t):
            self.tok = t

    module = types.SimpleNamespace(add=added.append)
    fake_self = types.SimpleNamespace(
        _getLdsReadMemToken=lambda w, k, t: ("TOKEN", 5)
    )
    # comment set -> the "%s sync LDS%u" branch.
    LocalRead._emitLdsRead(fake_self, None, None, None, _Inst, "dst", "src", "ds",
                           module, comment="cmt")
    # comment empty -> the "sync LDS%u" branch.
    LocalRead._emitLdsRead(fake_self, None, None, None, _Inst, "dst", "src", "ds", module)
    assert len(added) == 2
    assert added[0].tok == "TOKEN" and added[0].kw["comment"] == "cmt sync LDS5"
    assert added[1].kw["comment"] == "sync LDS5"


# ===========================================================================
# componentPath / commentHeader
# ===========================================================================

def test_component_path_and_comment_header(snapshot):
    inst = _CharImplCap()
    assert {"path": _CharImplCap.componentPath(), "comment": inst.commentHeader()} == snapshot
