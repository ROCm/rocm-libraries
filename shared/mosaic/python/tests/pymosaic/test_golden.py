# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden-vector tests: validate the JIT-backed Python ``Layout`` against
reference outputs computed directly from the mosaic C++ headers.

The reference vectors live in ``golden/golden_vectors.json``, produced by
``golden/generate_golden.cpp`` (which uses the C++ headers as the source
of truth — see that file's header for the format). For each layout the
golden file records, for every flat index ``i`` in ``[0, size)``:

  - ``offsets[i]``    — ``L(i)``                (flat eval)
  - ``top_coords[i]`` — one scalar per mode     (eval_coord input)
  - ``idx2crd[i]``    — the nested coordinate   (idx2crd output / crd2idx input)

plus ``size``, ``cosize``, ``rank`` and the ``(shape):(stride)`` repr.

Each ``Layout`` method is exercised against the corresponding golden
field, so the whole binding stack (``_render`` -> ``_jit`` -> nanobind ->
``Layout``) is checked end-to-end against the C++ semantics.

A separate test (``test_golden_matches_cpp_source``) recompiles the
generator and asserts the committed JSON still matches what the current
C++ headers produce — catching silent drift if the headers change.

The same layout set powers a small benchmark; run this file directly::

    PYTHONPATH=python python python/tests/pymosaic/test_golden.py --bench
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

from pymosaic.layout import Layout, crd2idx, idx2crd

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
_GOLDEN_JSON = _GOLDEN_DIR / "golden_vectors.json"
_GENERATOR_CPP = _GOLDEN_DIR / "generate_golden.cpp"
_REPO_ROOT = _GOLDEN_DIR.parent.parent.parent.parent
_INCLUDE_DIR = _REPO_ROOT / "include"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tuple(v):
    """Convert a JSON value into the nested-tuple form the Python ``Layout``
    uses: lists become tuples, numbers stay ints."""
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    return int(v)


def _load_golden() -> list[dict]:
    if not _GOLDEN_JSON.is_file():
        pytest.fail(
            f"golden vectors missing at {_GOLDEN_JSON}; regenerate with "
            f"`g++ -std=c++20 -I include {_GENERATOR_CPP} -o /tmp/gg && "
            f"/tmp/gg {_GOLDEN_JSON}`"
        )
    return json.loads(_GOLDEN_JSON.read_text())


_GOLDEN = _load_golden()
_IDS = [rec["name"] for rec in _GOLDEN]


def _layout_from_record(rec: dict) -> Layout:
    return Layout(shape=_to_tuple(rec["shape"]), stride=_to_tuple(rec["stride"]))


# ---------------------------------------------------------------------------
# Scalar metadata: rank / size / cosize / repr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_scalar_metadata(rec: dict) -> None:
    L = _layout_from_record(rec)
    assert L.rank == rec["rank"], f"rank mismatch for {rec['name']}"
    assert len(L) == rec["size"], f"size/len mismatch for {rec['name']}"
    assert L.cosize == rec["cosize"], f"cosize mismatch for {rec['name']}"
    assert str(L) == rec["repr"], f"repr mismatch for {rec['name']}"


# ---------------------------------------------------------------------------
# Flat eval: L[i] and the bulk enumerate()/iter paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_flat_eval(rec: dict) -> None:
    L = _layout_from_record(rec)
    expected = rec["offsets"]

    # Per-element flat eval.
    for i, want in enumerate(expected):
        assert L[i] == want, f"{rec['name']}: L[{i}] == {L[i]}, want {want}"

    # Bulk paths must agree with the per-element offsets.
    assert L.enumerate() == expected, f"{rec['name']}: enumerate() mismatch"
    assert list(L) == expected, f"{rec['name']}: list(L) mismatch"


# ---------------------------------------------------------------------------
# Coordinate eval: L[c0, c1, ...] (one scalar per top-level mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_coord_eval(rec: dict) -> None:
    L = _layout_from_record(rec)
    offsets = rec["offsets"]
    for i, coord in enumerate(rec["top_coords"]):
        got = L[tuple(coord)]
        assert (
            got == offsets[i]
        ), f"{rec['name']}: L[{tuple(coord)}] == {got}, want {offsets[i]}"


# ---------------------------------------------------------------------------
# idx2crd: flat index -> nested coordinate (and the free-function form)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_idx2crd(rec: dict) -> None:
    L = _layout_from_record(rec)
    for i, crd in enumerate(rec["idx2crd"]):
        want = _to_tuple(crd)
        got = L.idx2crd(i)
        assert got == want, f"{rec['name']}: idx2crd({i}) == {got}, want {want}"
        # Free-function form mirrors the method.
        assert idx2crd(i, L) == want


# ---------------------------------------------------------------------------
# crd2idx: nested coordinate -> offset. The composition identity
#   L.crd2idx(L.idx2crd(i)) == L[i]
# is the bijection round-trip, checked against the C++ offsets.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", _GOLDEN, ids=_IDS)
def test_crd2idx_roundtrip(rec: dict) -> None:
    L = _layout_from_record(rec)
    offsets = rec["offsets"]
    for i, crd in enumerate(rec["idx2crd"]):
        coord = _to_tuple(crd)
        got = L.crd2idx(coord)
        assert (
            got == offsets[i]
        ), f"{rec['name']}: crd2idx({coord}) == {got}, want {offsets[i]}"
        # Free-function form mirrors the method.
        assert crd2idx(coord, L) == offsets[i]


# ---------------------------------------------------------------------------
# Coverage guard: make sure we actually exercise every composition family.
# ---------------------------------------------------------------------------


def test_families_covered() -> None:
    families = {rec["family"] for rec in _GOLDEN}
    assert {
        "flat",
        "hierarchical",
        "mixed",
    } <= families, f"golden set is missing a composition family; have {families}"


# ---------------------------------------------------------------------------
# Drift guard: recompile the C++ generator and confirm the committed JSON
# still matches. Skips cleanly if no C++ compiler is available.
# ---------------------------------------------------------------------------


def _compile_and_run_generator(out_path: pathlib.Path) -> None:
    cxx = os.environ.get("CXX", "g++")
    exe = out_path.with_suffix(".bin")
    compile_cmd = [
        cxx,
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
        pytest.skip(f"C++ compiler {cxx!r} not found; cannot regenerate golden vectors")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"golden generator failed to compile:\n{exc.stderr}")
    subprocess.run(
        [str(exe), str(out_path)], check=True, capture_output=True, text=True
    )


def test_golden_matches_cpp_source() -> None:
    """The committed golden JSON must equal a fresh run of the C++ generator.

    This is what keeps the reference honest: if someone changes the layout
    headers (strides, idx2crd, cosize, ...) without regenerating, this fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fresh = pathlib.Path(tmp) / "fresh_golden.json"
        _compile_and_run_generator(fresh)
        fresh_data = json.loads(fresh.read_text())
    committed = json.loads(_GOLDEN_JSON.read_text())
    assert fresh_data == committed, (
        "committed golden_vectors.json is stale relative to the C++ headers; "
        f"regenerate with: g++ -std=c++20 -I include {_GENERATOR_CPP} "
        f"-o /tmp/gg && /tmp/gg {_GOLDEN_JSON}"
    )


# ---------------------------------------------------------------------------
# Benchmark mode (not a test): reuse the golden layouts to time compile and
# eval per composition family.  `python python/tests/pymosaic/test_golden.py --bench`
# ---------------------------------------------------------------------------


def _benchmark() -> int:
    import statistics
    import time

    # Fresh cache so compile timings are honest.
    cache = tempfile.mkdtemp(prefix="mosaic-golden-bench-")
    os.environ["MOSAIC_JIT_CACHE"] = cache
    from pymosaic import _jit

    _jit.cache_root = lambda: pathlib.Path(cache)  # type: ignore[assignment]

    print(f"bench cache : {cache}\n")
    print(
        f"{'layout':24s} {'family':13s} {'size':>5s} {'compile':>10s} {'eval µs':>9s}"
    )
    print("-" * 66)

    seen_struct: dict[str, float] = {}
    for rec in _GOLDEN:
        L = _layout_from_record(rec)
        sig = L.shape, L.stride  # display only; module is keyed on structure
        t0 = time.perf_counter()
        L._ensure_module()
        compile_s = time.perf_counter() - t0

        n = len(L)
        ITERS = max(2000, n * 50)
        t0 = time.perf_counter()
        for i in range(ITERS):
            _ = L[i % n]
        eval_us = (time.perf_counter() - t0) / ITERS * 1e6

        compiled = "compiled" if compile_s > 0.05 else "cache-hit"
        print(
            f"{rec['name']:24s} {rec['family']:13s} {n:>5d} "
            f"{compile_s * 1000:>8.1f}ms {eval_us:>8.3f} ({compiled})"
        )

    return 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        raise SystemExit(_benchmark())
    raise SystemExit(pytest.main([__file__, "-v"]))
