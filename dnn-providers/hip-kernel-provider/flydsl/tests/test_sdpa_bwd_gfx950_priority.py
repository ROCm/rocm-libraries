# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The dropout helpers of the gfx950 backward kernel tree evaluate in float32, as the native engine does.

``dropout_threshold`` and ``dropout_scale`` follow ``dropoutThreshold`` and ``dropoutScale`` of
``FlydslSdpaDropout.hpp``: ``(p - 0.5F) * 2^32`` truncated to i32 and ``1.0F / (1.0F - p)``, every operation in float32.
They live in ``fmha_abi_gfx1201``, whose module imports ``flydsl``; where it is not installed (the quick tier's system Python)
the pins run on the three function definitions read from that file, so they run in every environment.
"""

from __future__ import annotations

import ast
import ctypes
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "kernels_src"))
ABI = FLYDSL / "kernels_src" / "kernels" / "attention_bwd_cdna" / "fmha_abi_gfx1201.py"
HELPERS = ("_f32", "dropout_threshold", "dropout_scale")


def _flydsl_installed() -> bool:
    """Whether the real FlyDSL package is importable. The provider's own `flydsl/` source directory is a namespace package
    on `sys.path` when pytest runs from the provider root, so the package's presence is not enough: the installed package
    has `flydsl.compiler`, the source directory does not."""
    try:
        return importlib.util.find_spec("flydsl.compiler") is not None
    except ModuleNotFoundError:
        return False


def _host_helpers() -> dict:
    """The module's own helpers: imported where flydsl is installed, else their definitions executed from the file."""
    if _flydsl_installed():
        # flydsl is installed, so a failing import here is a broken sibling module, not a missing dependency.
        from kernels.attention_bwd_cdna import fmha_abi_gfx1201 as abi  # noqa: PLC0415

        return {name: getattr(abi, name) for name in HELPERS}
    tree = ast.parse(ABI.read_text(encoding="utf-8"))
    defs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS
    ]
    assert {d.name for d in defs} == set(HELPERS)
    namespace = {"struct": struct}
    exec(
        compile(ast.Module(body=defs, type_ignores=[]), str(ABI), "exec"), namespace
    )  # noqa: S102
    return namespace


_helpers = _host_helpers()
dropout_threshold = _helpers["dropout_threshold"]
dropout_scale = _helpers["dropout_scale"]

I32_MIN, I32_MAX = -(2**31), 2**31 - 1


def f32(x: float) -> float:
    """`x` rounded to float32 by the C runtime's conversion, not by the module's own `struct` round trip."""
    return ctypes.c_float(x).value


# The sweep is the C++ operation sequence in float32: each operation of two float32 operands is computed in double (exact
# enough that one rounding to float32 equals the float32 operation). 0.1, 0.2 and 0.22 are the probabilities whose
# double-precision threshold differs from the float32 one.
PROBABILITIES = sorted(
    {0.0, 0.1, 0.2, 0.22, 0.25, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999999, 1.0}
    | {i / 64.0 for i in range(65)}
    | {f32(f32(i) / f32(97)) for i in range(97)}
)


def cxx_threshold(p: float) -> int:
    scaled = f32(f32(f32(p) - 0.5) * 4294967296.0)
    if not scaled < 2147483648.0:
        return I32_MAX
    if not scaled > -2147483648.0:
        return I32_MIN
    return int(scaled)


def cxx_scale(p: float) -> float:
    return f32(1.0 / f32(1.0 - f32(p)))


@pytest.mark.parametrize("p", PROBABILITIES)
def test_threshold_is_the_float32_formula(p: float) -> None:
    assert dropout_threshold(p) == cxx_threshold(p)


@pytest.mark.parametrize("p", [q for q in PROBABILITIES if q < 1.0])
def test_scale_is_the_float32_formula(p: float) -> None:
    assert dropout_scale(p) == cxx_scale(p)


@pytest.mark.parametrize("p", [0.1, 0.2, 0.22])
def test_threshold_differs_from_the_double_formula(p: float) -> None:
    """The cases the float32 form exists for: the double product lands on another i32."""
    double = int((p - 0.5) * 0xFFFFFFFF)
    assert dropout_threshold(p) != double


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_threshold_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError):
        dropout_threshold(p)


# --- Ranking of the split rows over the installed gfx950 tables -------------------------------------------------------------
#
# The rows are the instance tables as the KMD spells them (``sdpa_bwd_model.table_rows``) and a row serves a graph by
# ``sdpa_bwd_model.serves``, the model of ``flydslSdpaBwdSplitKernelMatches`` and ``servesHeadDims`` in
# ``FlydslSdpaBwdSplitNative.cpp``: head dims, class, dropout, ``bias_f32``, ``u2_role`` and the dQ bias stage. A graph is a
# head-dim pair, a class (plain, causal, bias), a dropout state, and the bias width, staging and V heads the model takes; GQA
# packing is off (a grouped graph's packed rows resolve a tile only the built manifest knows).

sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
from generators import _sdpa_bwd_nondet as nondet  # noqa: E402
from generators._instances import SDPA_BWD_TARGET, instances_for  # noqa: E402
from generators._sdpa_bwd_spec import to_spec  # noqa: E402
from sdpa_bwd_model import serves, table_rows  # noqa: E402

CLASSES = ("plain", "causal", "bias")
HEAD_DIMS = sorted({*range(1, 513), *(8 * i for i in range(1, 65))})
MIXED_HEAD_DIMS = [
    (qk, v)
    for qk in (24, 32, 64, 100, 128, 200, 256, 300)
    for v in (32, 64, 128, 256)
    if qk != v
]


def graph_shapes(dtype: str, cls: str):
    """(staging, FLOAT bias, Hk != Hv) of a class: the bias axes only on a bias graph, V heads only without one, neither on fp32."""
    if cls == "bias":
        return [
            (staging, f32, False)
            for staging in (True, False)
            for f32 in ((False, True) if dtype != "fp32" else (False,))
        ]
    return [(True, False, False)] + ([(True, False, True)] if dtype != "fp32" else [])


def graphs(kind: str, dtype: str) -> list[dict]:
    """Every graph of a kind and dtype, as ``serves`` reads it; prep is class- and dropout-free."""
    pairs = [(d, d) for d in HEAD_DIMS] + MIXED_HEAD_DIMS
    shared = dict(dtype=dtype, reduces=False, gqa=False, staging=True)
    if kind == "prep":
        return [
            dict(shared, qk=qk, v=v, cls="plain", dropout=False, dbias=False)
            for qk, v in pairs
        ]
    return [
        dict(
            shared,
            qk=qk,
            v=v,
            cls=cls,
            dropout=dropout,
            dbias=dbias,
            staging=staging,
            bias_f32=f32,
            value_heads=value_heads,
        )
        for qk, v in pairs
        for cls in CLASSES
        for staging, f32, value_heads in graph_shapes(dtype, cls)
        for dropout in (False, True)
        for dbias in ((False, True) if cls == "bias" else (False,))
    ]


LISTED = nondet.read(arch="gfx950").keys()


def policy_default_hashes(kind: str) -> set[str]:
    """The Spec hashes of the rows of no knob that the nondeterministic list does not name."""
    hashes = set()
    for row in instances_for(f"sdpa_bwd_{kind}", SDPA_BWD_TARGET):
        spec = to_spec(row)
        if not spec.knobs and spec.hash() not in LISTED:
            hashes.add(spec.hash())
    return hashes


def serving(kind: str, rows_by_tile, graph: dict) -> list[dict]:
    high = max(graph["qk"], graph["v"])
    return [
        row
        for tile, rows in rows_by_tile
        if tile >= high
        for row in rows
        if serves(f"sdpa_bwd_{kind}", row, graph)
    ]


@pytest.fixture(scope="module", params=["prep", "dkdv", "dq"])
def tables(request):
    kind = request.param
    by_dtype = {}
    for row in table_rows(kind, SDPA_BWD_TARGET):
        by_dtype.setdefault(row["dtype"], {}).setdefault(
            row["knobs"]["head_dim"], []
        ).append(row)
    return (
        kind,
        {dtype: sorted(tiles.items()) for dtype, tiles in by_dtype.items()},
        policy_default_hashes(kind),
    )


def test_the_exact_tile_row_ranks_first(tables) -> None:
    """Where an exact padded-0 row serves the graph, a row of that tile and padded 0 outranks every other serving row."""
    kind, by_dtype, _ = tables
    checked = 0
    for dtype, rows_by_tile in by_dtype.items():
        for graph in graphs(kind, dtype):
            qk = graph["qk"]
            if qk != graph["v"] or qk not in dict(rows_by_tile):
                continue
            rows = serving(kind, rows_by_tile, graph)
            exact = [r for r in rows if r["knobs"]["padded"] == 0]
            if not exact:
                continue
            top = max(r["priority"] for r in rows)
            best = [r for r in rows if r["priority"] == top]
            assert all(
                r["knobs"]["padded"] == 0 and r["knobs"]["head_dim"] == qk for r in best
            ), (kind, dtype, graph)
            checked += 1
    assert checked


def test_no_two_policy_default_rows_serving_one_graph_tie(tables) -> None:
    kind, by_dtype, defaults = tables
    checked = 0
    for dtype, rows_by_tile in by_dtype.items():
        for graph in graphs(kind, dtype):
            rows = serving(kind, rows_by_tile, graph)
            priorities = [r["priority"] for r in rows if r["hash"] in defaults]
            assert len(priorities) == len(set(priorities)), (kind, dtype, graph)
            checked += bool(priorities)
    assert checked
