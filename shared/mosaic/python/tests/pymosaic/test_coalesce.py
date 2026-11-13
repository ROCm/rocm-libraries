# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for ``Layout.coalesce`` (and the ``coalesce`` free function).

``coalesce`` is JIT-backed: the merge fold runs in the compiled C++ module
(``coalesce_flat`` in ``_template.cpp.in``). It is validated three ways:

1. **Header golden** (``golden/coalesce_golden.json``) — reference results
   computed directly from ``mosaic::coalesce`` on *static* (``C<v>``)
   layouts. mosaic::coalesce is compile-time only (try_merge's runtime
   branch returns two different tuple types under a non-constexpr ``if``,
   so it won't instantiate on runtime ``size_t`` leaves — which is exactly
   why the runtime fold exists). Building the reference from constants lets
   the header act as the source of truth; the cases avoid size-1 modes so
   the static path (which absorbs them) and the runtime fold (which does
   not) apply the identical ``prev_shape*prev_stride == stride`` rule.

2. **Map-preservation invariant** — coalesce must never change the
   coordinate->offset map. Checked across many layouts, including the
   flat-rank > 4 ones the compile-time header can't even instantiate.

3. **Exact canonical-form cases** — including high-rank ones, to pin that
   the fold actually merges (not just preserves the map).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

import pytest

from pymosaic.layout import Layout, coalesce

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
_GOLDEN_JSON = _GOLDEN_DIR / "coalesce_golden.json"
_GENERATOR_CPP = _GOLDEN_DIR / "generate_coalesce_golden.cpp"
_REPO_ROOT = _GOLDEN_DIR.parent.parent.parent.parent
_INCLUDE_DIR = _REPO_ROOT / "include"


def _to_tuple(v):
    """JSON value -> the nested-tuple form Layout uses (lists -> tuples)."""
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    return int(v)


def _flat(v) -> tuple:
    """Normalise a coalesced shape/stride to a flat tuple.

    The header returns a scalar-shape layout for a fully-collapsed result
    (``make_layout(scalar, scalar)`` -> ``Layout<size_t, size_t>``), so the
    golden value may be a bare int ``32`` where pymosaic consistently uses
    the rank-1 form ``(32,)``. Normalising both sides makes the comparison
    representation-independent while still pinning the leaf values."""
    t = _to_tuple(v)
    return t if isinstance(t, tuple) else (t,)


_GOLDEN = json.loads(_GOLDEN_JSON.read_text())
_IDS = [rec["name"] for rec in _GOLDEN]


# ---------------------------------------------------------------------------
# 1. Header golden: runtime fold must match mosaic::coalesce exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_coalesce_matches_header(rec: dict) -> None:
    L = Layout(shape=_to_tuple(rec["shape"]), stride=_to_tuple(rec["stride"]))
    C = L.coalesce()
    assert C.shape == _flat(rec["coalesce_shape"]), f"shape mismatch for {rec['name']}"
    assert C.stride == _flat(
        rec["coalesce_stride"]
    ), f"stride mismatch for {rec['name']}"
    # The map must be preserved regardless.
    assert C.enumerate() == L.enumerate(), f"map changed for {rec['name']}"


# ---------------------------------------------------------------------------
# 2. Map-preservation invariant (semantic correctness, any rank/nesting).
# ---------------------------------------------------------------------------

_INVARIANT_LAYOUTS = [
    Layout.column_major(4, 8),
    Layout.row_major(4, 8),
    Layout(shape=(2, 4), stride=(1, 2)),
    Layout(shape=(4,), stride=(2,)),
    Layout(shape=(2, 3, 4), stride=(1, 2, 6)),
    Layout(shape=(2, 3, 4), stride=(12, 4, 1)),
    Layout(shape=((2, 2), 3), stride=((1, 2), 4)),
    Layout(shape=((2, 2), (3, 3))),  # auto col-major
    Layout(shape=(2, 4, 8), stride=(1, 2, 16)),
    # flat rank > 4: the compile-time header can't instantiate these, the
    # runtime fold can — fully contiguous, so it collapses to a single mode.
    Layout(shape=(2, 2, 2, 2, 2), stride=(1, 2, 4, 8, 16)),
    Layout(shape=(2, 2, 2, 2, 2, 2), stride=(1, 2, 4, 8, 16, 32)),
]
_INV_IDS = [f"{tuple(L.shape)}:{tuple(L.stride)}" for L in _INVARIANT_LAYOUTS]


@pytest.mark.parametrize("L", _INVARIANT_LAYOUTS, ids=_INV_IDS)
def test_coalesce_preserves_map(L: Layout) -> None:
    C = L.coalesce()
    assert C.enumerate() == L.enumerate()
    assert len(C) == len(L)
    assert C.cosize == L.cosize
    # The result is always flat (depth 1) and minimal.
    assert C.depth == 1


@pytest.mark.parametrize("L", _INVARIANT_LAYOUTS, ids=_INV_IDS)
def test_coalesce_is_idempotent(L: Layout) -> None:
    C = L.coalesce()
    CC = C.coalesce()
    assert CC.shape == C.shape
    assert CC.stride == C.stride


# ---------------------------------------------------------------------------
# 3. Exact canonical-form cases (pin that merging actually happens).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, stride, want_shape, want_stride",
    [
        ((4, 8), (1, 4), (32,), (1,)),  # col-major -> collapse
        ((4, 8), (8, 1), (4, 8), (8, 1)),  # row-major -> no merge
        ((2, 4), (1, 2), (8,), (1,)),  # collapse
        ((2, 3, 4), (1, 2, 6), (24,), (1,)),  # col-major -> collapse
        ((2, 4, 8), (1, 2, 16), (8, 8), (1, 16)),  # partial merge
        (((2, 2), 3), ((1, 2), 4), (12,), (1,)),  # nested -> collapse
        ((4,), (2,), (4,), (2,)),  # single strided mode
        # high rank, contiguous -> single mode (beyond the header's reach)
        ((2, 2, 2, 2, 2), (1, 2, 4, 8, 16), (32,), (1,)),
        # high rank, contiguous block then a gap
        ((2, 2, 2, 3), (1, 2, 4, 16), (8, 3), (1, 16)),
    ],
)
def test_coalesce_exact(shape, stride, want_shape, want_stride) -> None:
    C = Layout(shape=shape, stride=stride).coalesce()
    assert C.shape == want_shape
    assert C.stride == want_stride


# ---------------------------------------------------------------------------
# Free-function parity and error handling.
# ---------------------------------------------------------------------------


def test_free_function_matches_method() -> None:
    L = Layout(shape=(2, 4, 8), stride=(1, 2, 16))
    a = coalesce(L)
    b = L.coalesce()
    assert (a.shape, a.stride) == (b.shape, b.stride)


def test_free_function_rejects_non_layout() -> None:
    with pytest.raises(TypeError):
        coalesce((2, 4))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Drift guard: recompile the generator and confirm the committed JSON still
# matches the current headers. Skips cleanly if no C++ compiler is present.
# ---------------------------------------------------------------------------


def test_coalesce_golden_matches_cpp_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        exe = tmp_path / "gen_coalesce.bin"
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
        except FileNotFoundError:
            pytest.skip("C++ compiler not found; cannot regenerate coalesce golden")
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"coalesce golden generator failed to compile:\n{exc.stderr}")
        subprocess.run(
            [str(exe), str(fresh)], check=True, capture_output=True, text=True
        )
        assert json.loads(fresh.read_text()) == json.loads(_GOLDEN_JSON.read_text()), (
            "committed coalesce_golden.json is stale; regenerate it from "
            "generate_coalesce_golden.cpp"
        )
