# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the logical_divide family — ``logical_divide``, ``zipped_divide``,
``tiled_divide``, ``flat_divide`` (pure-Python ports of ``mosaic::*_divide``).

Ground-truth checks per case:

1. **Header golden** (``golden/divide_golden.json``) — structural parity with
   the static C++ result.
2. **Image-set invariant** — a divide is a re-indexing of the layout: the set
   of offsets it produces equals the set the original layout produces (mirrors
   ``test_algebra.cpp``'s ``expect_logical_divide_preserves``). Requires the JIT.

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
_GOLDEN_JSON = _GOLDEN_DIR / "divide_golden.json"
_GENERATOR_CPP = _GOLDEN_DIR / "generate_divide_golden.cpp"
_INCLUDE_DIR = _HERE.parents[2] / "include"

_GOLDEN = json.loads(_GOLDEN_JSON.read_text())
_IDS = [rec["name"] for rec in _GOLDEN]

# Method dispatch by op name.
_OPS = {
    "logical_divide": lambda L, t: L.logical_divide(t),
    "zipped_divide": lambda L, t: L.zipped_divide(t),
    "tiled_divide": lambda L, t: L.tiled_divide(t),
    "flat_divide": lambda L, t: L.flat_divide(t),
}


def _to_tuple(x):
    if isinstance(x, list):
        return tuple(_to_tuple(e) for e in x)
    return x


def _make_tiler(tj):
    """Reconstruct a public-API tiler from the golden's JSON encoding."""
    if "int" in tj:
        return tj["int"]
    if "layout" in tj:
        sh, st = tj["layout"]
        return Layout(shape=_to_tuple(sh), stride=_to_tuple(st))
    if "tuple" in tj:
        return tuple(tj["tuple"])
    raise ValueError(f"unrecognised tiler encoding: {tj!r}")


@pytest.fixture(params=_GOLDEN, ids=_IDS)
def rec(request):
    return request.param


def _layout(rec):
    return Layout(
        shape=_to_tuple(rec["layout_shape"]), stride=_to_tuple(rec["layout_stride"])
    )


def test_divide_matches_header(rec):
    """Structural parity with the static C++ divide-family result."""
    L = _layout(rec)
    t = _make_tiler(rec["tiler"])
    R = _OPS[rec["op"]](L, t)
    assert R.shape == _to_tuple(
        rec["result_shape"]
    ), f"shape mismatch for {rec['name']}"
    assert R.stride == _to_tuple(
        rec["result_stride"]
    ), f"stride mismatch for {rec['name']}"


def test_divide_preserves_image(rec):
    """A divide re-indexes the layout: same offset *set*, same size."""
    L = _layout(rec)
    t = _make_tiler(rec["tiler"])
    R = _OPS[rec["op"]](L, t)
    assert len(R) == len(L)
    assert set(R.enumerate()) == set(
        L.enumerate()
    ), f"image set changed for {rec['name']}"


def test_divide_free_function_matches_method(rec):
    L = _layout(rec)
    t = _make_tiler(rec["tiler"])
    free = getattr(pm, rec["op"])(L, t)
    m = _OPS[rec["op"]](L, t)
    assert free.shape == m.shape and free.stride == m.stride


def test_logical_divide_underscore_is_identity():
    L = Layout(shape=(8, 4), stride=(4, 1))
    R = L.logical_divide(slice(None))
    assert R.shape == L.shape and R.stride == L.stride


def test_divide_golden_matches_cpp_source():
    if shutil.which("g++") is None:
        pytest.skip("C++ compiler not found; cannot regenerate divide golden")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        exe = tmp_path / "gen_divide.bin"
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
            pytest.fail(f"divide golden generator failed to compile:\n{exc.stderr}")
        subprocess.run(
            [str(exe), str(fresh)], check=True, capture_output=True, text=True
        )
        assert json.loads(fresh.read_text()) == json.loads(_GOLDEN_JSON.read_text()), (
            "committed divide_golden.json is stale; regenerate it from "
            "generate_divide_golden.cpp"
        )
