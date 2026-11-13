# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for ``composition`` — pure-Python port of ``mosaic::composition``.

Three independent ground-truth checks per case:

1. **Header golden** (``golden/composition_golden.json``) — the result's
   ``(shape, stride)`` must match the *static* C++ ``composition`` structurally.
   mosaic's compile-time algebra is the source of truth for nesting.
2. **Defining invariant** — ``composition(A, B)(c) == A(B(c))`` for every
   ``c`` in ``domain(B)``. The load-bearing semantic check (mirrors
   ``test_algebra.cpp``'s ``expect_composition_preserves``). Evaluated via the
   JIT, so it requires the C++ toolchain.

A drift guard recompiles the generator and confirms the committed JSON is
current (skips cleanly if no C++ compiler is present).
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile

import pytest

import pymosaic as pm
from pymosaic import Layout

_HERE = pathlib.Path(__file__).resolve().parent
_GOLDEN_DIR = _HERE / "golden"
_GOLDEN_JSON = _GOLDEN_DIR / "composition_golden.json"
_GENERATOR_CPP = _GOLDEN_DIR / "generate_composition_golden.cpp"
_INCLUDE_DIR = _HERE.parents[2] / "include"

_GOLDEN = json.loads(_GOLDEN_JSON.read_text())
_IDS = [rec["name"] for rec in _GOLDEN]


def _to_tuple(x):
    """JSON arrays -> tuples; bare ints stay ints (mosaic scalar-mode form)."""
    if isinstance(x, list):
        return tuple(_to_tuple(e) for e in x)
    return x


@pytest.fixture(params=_GOLDEN, ids=_IDS)
def rec(request):
    return request.param


def _layouts(rec):
    A = Layout(shape=_to_tuple(rec["lhs_shape"]), stride=_to_tuple(rec["lhs_stride"]))
    B = Layout(shape=_to_tuple(rec["rhs_shape"]), stride=_to_tuple(rec["rhs_stride"]))
    return A, B


def test_composition_matches_header(rec):
    """Structural parity with the static C++ ``composition``."""
    A, B = _layouts(rec)
    C = A.composition(B)
    assert C.shape == _to_tuple(
        rec["composition_shape"]
    ), f"shape mismatch for {rec['name']}"
    assert C.stride == _to_tuple(
        rec["composition_stride"]
    ), f"stride mismatch for {rec['name']}"


def test_composition_preserves_map(rec):
    """``composition(A, B)(c) == A(B(c))`` over all of ``domain(B)``."""
    A, B = _layouts(rec)
    C = A.composition(B)
    assert len(C) == len(B)
    for c in range(len(B)):
        assert C[c] == A[B[c]], f"composition map broke at c={c} for {rec['name']}"


def test_composition_free_function_matches_method(rec):
    A, B = _layouts(rec)
    fn = pm.composition(A, B)
    m = A.composition(B)
    assert fn.shape == m.shape and fn.stride == m.stride


def test_composition_underscore_is_identity():
    """``composition(L, _)`` returns ``L`` unchanged (underscore tiler)."""
    L = Layout(shape=(8, 4), stride=(4, 1))
    for under in (None, slice(None), Ellipsis):
        C = L.composition(under)
        assert C.shape == L.shape and C.stride == L.stride


def test_composition_integer_tiler():
    """``composition(L, N)`` treats N as the contiguous tiler ``(N):(1)``."""
    L = Layout(shape=(8,), stride=(1,))
    C = L.composition(4)
    assert len(C) == 4
    assert [C[i] for i in range(4)] == [0, 1, 2, 3]


def test_composition_golden_matches_cpp_source():
    """Drift guard: recompile the generator and confirm the committed JSON is
    current. Skips cleanly if no C++ compiler is present."""
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler not found; cannot regenerate composition golden")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        exe = tmp_path / "gen_composition.bin"
        fresh = tmp_path / "fresh.json"
        compile_cmd = [
            "g++",
            "-std=c++20",
            "-O2",
            f"-I{_INCLUDE_DIR}",
            str(_GENERATOR_CPP),
            "-o",
            str(exe),
        ]
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            pytest.fail(
                f"composition golden generator failed to compile:\n{exc.stderr}"
            )
        subprocess.run(
            [str(exe), str(fresh)], check=True, capture_output=True, text=True
        )
        assert json.loads(fresh.read_text()) == json.loads(_GOLDEN_JSON.read_text()), (
            "committed composition_golden.json is stale; regenerate it from "
            "generate_composition_golden.cpp"
        )
