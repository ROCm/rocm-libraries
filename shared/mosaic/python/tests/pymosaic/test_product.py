# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the logical_product family — ``logical_product``,
``zipped_product``, ``tiled_product``, ``flat_product`` (pure-Python ports of
``mosaic::*_product``).

Ground-truth checks per case:

1. **Header golden** (``golden/product_golden.json``) — structural parity with
   the static C++ result. This is the *primary* oracle for products: unlike
   composition/divide, mosaic's per-mode tuple-tiler product is a deliberately
   non-canonical construction that diverges from pycute, so the C++ headers
   (not pycute) are authoritative for structure.
2. **Size postcondition** — ``size(logical_product(A, T)) == size(A)*size(T)``
   (the only guarantee for tuple tilers; mirrors ``test_algebra.cpp``).
3. **Bijection** — for the cases C++ marks bijective (rank-1 layouts, integer
   tilers, and cleanly-factoring multi-mode layout tilers), every offset is hit
   exactly once. Requires the JIT.

Drift guard recompiles the generator and confirms the committed JSON is current.
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
_GOLDEN_JSON = _GOLDEN_DIR / "product_golden.json"
_GENERATOR_CPP = _GOLDEN_DIR / "generate_product_golden.cpp"
_INCLUDE_DIR = _HERE.parents[2] / "include"

_GOLDEN = json.loads(_GOLDEN_JSON.read_text())
_IDS = [rec["name"] for rec in _GOLDEN]

_OPS = {
    "logical_product": lambda L, t: L.logical_product(t),
    "zipped_product": lambda L, t: L.zipped_product(t),
    "tiled_product": lambda L, t: L.tiled_product(t),
    "flat_product": lambda L, t: L.flat_product(t),
}

# Cases C++ guarantees bijective (see test_algebra.cpp comments).
_BIJECTIVE = {"lp_flat_2x3", "lp_nonunit", "lp_multi", "lp_int_2_3"}


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(e) for e in x)
    return x


def _make_tiler(tj):
    if "int" in tj:
        return tj["int"]
    if "layout" in tj:
        sh, st = tj["layout"]
        return Layout(shape=_to_tuple(sh), stride=_to_tuple(st))
    if "tuple" in tj:
        return tuple(tj["tuple"])
    raise ValueError(f"unrecognised tiler encoding: {tj!r}")


def _tiler_size(tj):
    if "int" in tj:
        return tj["int"]
    if "layout" in tj:
        sh, _ = tj["layout"]
        return _prod(_to_tuple(sh))
    if "tuple" in tj:
        out = 1
        for n in tj["tuple"]:
            out *= n
        return out
    raise ValueError(tj)


def _prod(t):
    if isinstance(t, tuple):
        out = 1
        for x in t:
            out *= _prod(x)
        return out
    return t


@pytest.fixture(params=_GOLDEN, ids=_IDS)
def rec(request):
    return request.param


def _block(rec):
    return Layout(
        shape=_to_tuple(rec["block_shape"]), stride=_to_tuple(rec["block_stride"])
    )


def test_product_matches_header(rec):
    """Structural parity with the static C++ product-family result."""
    A = _block(rec)
    t = _make_tiler(rec["tiler"])
    R = _OPS[rec["op"]](A, t)
    assert R.shape == _to_tuple(
        rec["result_shape"]
    ), f"shape mismatch for {rec['name']}"
    assert R.stride == _to_tuple(
        rec["result_stride"]
    ), f"stride mismatch for {rec['name']}"


def test_product_size_postcondition(rec):
    """``size(result) == size(block) * size(tiler)`` (the universal guarantee)."""
    A = _block(rec)
    t = _make_tiler(rec["tiler"])
    R = _OPS[rec["op"]](A, t)
    assert len(R) == len(A) * _tiler_size(
        rec["tiler"]
    ), f"size postcondition for {rec['name']}"


def test_product_bijection(rec):
    """For the C++-guaranteed-bijective cases, every offset is hit once."""
    if rec["name"] not in _BIJECTIVE or rec["op"] != "logical_product":
        pytest.skip("not a bijective fixture")
    A = _block(rec)
    t = _make_tiler(rec["tiler"])
    R = _OPS[rec["op"]](A, t)
    img = R.enumerate()
    assert sorted(img) == list(range(len(R))), f"not a bijection for {rec['name']}"


def test_product_free_function_matches_method(rec):
    A = _block(rec)
    t = _make_tiler(rec["tiler"])
    free = getattr(pm, rec["op"])(A, t)
    m = _OPS[rec["op"]](A, t)
    assert free.shape == m.shape and free.stride == m.stride


def test_logical_product_underscore_is_identity():
    A = Layout(shape=(2, 3), stride=(1, 2))
    R = A.logical_product(slice(None))
    assert R.shape == A.shape and R.stride == A.stride


def test_product_golden_matches_cpp_source():
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler not found; cannot regenerate product golden")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        exe = tmp_path / "gen_product.bin"
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
            pytest.fail(f"product golden generator failed to compile:\n{exc.stderr}")
        subprocess.run(
            [str(exe), str(fresh)], check=True, capture_output=True, text=True
        )
        assert json.loads(fresh.read_text()) == json.loads(_GOLDEN_JSON.read_text()), (
            "committed product_golden.json is stale; regenerate it from "
            "generate_product_golden.cpp"
        )
