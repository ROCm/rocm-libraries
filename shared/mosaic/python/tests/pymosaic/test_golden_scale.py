# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scale golden tests: validate tens of thousands of random layouts against
ground-truth offset hashes computed by the mosaic C++ headers.

Unlike ``test_golden.py`` (a small, fully-materialised, human-readable
reference), this path is generative:

  1. ``scale_spec.generate_specs(N, seed)`` makes N random layouts over a
     FIXED set of structures (nesting profiles).
  2. ``bench_golden.cpp`` computes, from the C++ headers, a 64-bit FNV-1a
     hash of every layout's full offset enumeration (the ground truth).
  3. The Python ``Layout`` reproduces the enumeration via ``enumerate()``
     and must match the hash, plus size / cosize / repr.

Because the JIT is keyed on *structure*, all N layouts share a handful of
compiled modules — so validating tens of thousands costs only a few
compiles plus eval throughput. That's the whole point, and the benchmark
mode quantifies it.

Default N is modest so the suite stays fast; override with
``MOSAIC_SCALE_N``. Skips cleanly if no C++ compiler is available.

Benchmark (not a test)::

    PYTHONPATH=python python python/tests/pymosaic/test_golden_scale.py --bench 50000
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

from pymosaic.layout import Layout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "golden"))
import scale_spec  # noqa: E402

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
_BENCH_CPP = _GOLDEN_DIR / "bench_golden.cpp"
_REPO_ROOT = _GOLDEN_DIR.parent.parent.parent.parent
_INCLUDE_DIR = _REPO_ROOT / "include"

_DEFAULT_N = int(os.environ.get("MOSAIC_SCALE_N", "2000"))
_SEED = 0xC0FFEE


def _compile_bench(workdir: pathlib.Path) -> pathlib.Path | None:
    """Compile bench_golden.cpp; return the exe path, or None if no compiler."""
    cxx = os.environ.get("CXX", "g++")
    exe = workdir / "bench_golden.bin"
    cmd = [
        cxx,
        "-std=c++20",
        "-O2",
        f"-I{_INCLUDE_DIR}",
        str(_BENCH_CPP),
        "-o",
        str(exe),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"bench_golden.cpp failed to compile:\n{exc.stderr}")
    return exe


def _run_cpp_truth(
    exe: pathlib.Path, specs: list[dict], workdir: pathlib.Path
) -> list[dict]:
    """Run the C++ generator over ``specs`` and parse its golden output."""
    infile = workdir / "specs.txt"
    outfile = workdir / "golden.txt"
    scale_spec.write_cpp_input(specs, infile)
    subprocess.run(
        [str(exe), str(infile), str(outfile)],
        check=True,
        capture_output=True,
        text=True,
    )

    golden: list[dict] = []
    for line in outfile.read_text().splitlines():
        if not line.strip():
            continue
        # sid size cosize rank h_eval h_idx2crd repr  (repr is the final token)
        sid, size, cosize, rank, h_eval, h_idx2crd, repr_str = line.split(maxsplit=6)
        golden.append(
            {
                "sid": int(sid),
                "size": int(size),
                "cosize": int(cosize),
                "rank": int(rank),
                "h_eval": int(h_eval),
                "h_idx2crd": int(h_idx2crd),
                "repr": repr_str,
            }
        )
    return golden


def _profile(t):
    """Nesting profile of a tuple (values blanked) — used to count structures."""
    if isinstance(t, tuple):
        return tuple(_profile(x) for x in t)
    return None


def _mode_sizes(shape: tuple) -> list[int]:
    """Product of leaves for each top-level mode (the per-mode extent used to
    colex-decompose a flat index into one scalar coordinate per mode)."""
    from pymosaic import _render

    return [_render.size(mode) for mode in shape]


def _colex_coord(i: int, mode_size: list[int]) -> tuple:
    """Decompose flat index ``i`` into one scalar per mode, column-major —
    the eval_coord input. Mirrors the C++/idx2crd colex convention."""
    coord = []
    rem = i
    for sz in mode_size:
        if sz == 0:
            coord.append(0)
        else:
            coord.append(rem % sz)
            rem //= sz
    return tuple(coord)


def _validate(specs: list[dict], golden: list[dict], *, deep: bool = True) -> dict:
    """Check every Python Layout against the C++ ground truth across ALL
    function paths. Returns stats.

    Functions covered:
      - rank / size / cosize / str(repr)             (scalar metadata)
      - enumerate()                                   (bulk flat eval -> h_eval)
      - L[c0, c1, ...]                                (eval_coord -> must equal h_eval)
      - L.idx2crd(i)                                  (-> h_idx2crd)
      - L.crd2idx(L.idx2crd(i))                       (round-trip -> must equal h_eval)

    ``deep`` controls the per-index paths (eval_coord / idx2crd / crd2idx).
    They are exact and always checked; the flag exists only so the benchmark
    can time them separately.
    """
    from pymosaic import _render

    assert len(specs) == len(
        golden
    ), f"spec/golden count mismatch: {len(specs)} vs {len(golden)}"
    structures: set[tuple] = set()
    total_elems = 0
    for i, (spec, g) in enumerate(zip(specs, golden)):
        tag = f"[{i}] {spec['shape']}:{spec['stride']}"
        L = Layout(shape=spec["shape"], stride=spec["stride"])
        structures.add(_profile(L.shape))

        # --- scalar metadata ---
        assert L.rank == g["rank"], f"{tag} rank {L.rank} != {g['rank']}"
        assert L.cosize == g["cosize"], f"{tag} cosize {L.cosize} != {g['cosize']}"
        assert str(L) == g["repr"], f"{tag} repr {str(L)!r} != {g['repr']!r}"

        # --- flat eval via enumerate() (bulk; per-index L[i] is covered
        #     exhaustively by the curated test_golden.py) ---
        offsets = L.enumerate()
        total_elems += len(offsets)
        assert len(offsets) == g["size"], f"{tag} size {len(offsets)} != {g['size']}"
        assert (
            scale_spec.fnv1a_offsets(offsets) == g["h_eval"]
        ), f"{tag} enumerate() hash diverges from C++"

        if not deep:
            continue

        n = len(offsets)
        mode_size = _mode_sizes(L.shape)

        # --- eval_coord: L[c0, c1, ...] over colex top-level coords ---
        coord_offsets = [L[_colex_coord(j, mode_size)] for j in range(n)]
        assert (
            scale_spec.fnv1a_offsets(coord_offsets) == g["h_eval"]
        ), f"{tag} eval_coord path diverges from flat eval"

        # --- idx2crd: nested coordinate per flat index ---
        crds = [L.idx2crd(j) for j in range(n)]
        leaves: list[int] = []
        for c in crds:
            leaves.extend(_render.flatten(c) if isinstance(c, tuple) else (c,))
        assert (
            scale_spec.fnv1a_offsets(leaves) == g["h_idx2crd"]
        ), f"{tag} idx2crd path diverges from C++"

        # --- crd2idx round-trip: L.crd2idx(L.idx2crd(i)) == L[i] ---
        round_offsets = [L.crd2idx(c) for c in crds]
        assert (
            scale_spec.fnv1a_offsets(round_offsets) == g["h_eval"]
        ), f"{tag} crd2idx round-trip diverges from flat eval"

    return {
        "layouts": len(specs),
        "structures": len(structures),
        "elements": total_elems,
    }


def test_scale_against_cpp_golden() -> None:
    """Validate ``MOSAIC_SCALE_N`` random layouts against C++ offset hashes."""
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        exe = _compile_bench(workdir)
        if exe is None:
            pytest.skip(
                "no C++ compiler available; cannot generate ground-truth hashes"
            )
        specs = scale_spec.generate_specs(_DEFAULT_N, seed=_SEED)
        golden = _run_cpp_truth(exe, specs, workdir)
        stats = _validate(specs, golden)
    assert stats["layouts"] == _DEFAULT_N
    # All N layouts must fold onto the small fixed structure set — that's the
    # invariant that makes "tens of thousands" cheap.
    assert stats["structures"] <= len(scale_spec.STRUCTURES)


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------


def _benchmark(n: int) -> int:
    import statistics

    print(f"\nScale benchmark: {n:,} random layouts, seed={hex(_SEED)}\n")
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)

        t0 = time.perf_counter()
        exe = _compile_bench(workdir)
        if exe is None:
            print("no C++ compiler; aborting")
            return 1
        cpp_build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        specs = scale_spec.generate_specs(n, seed=_SEED)
        gen_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        golden = _run_cpp_truth(exe, specs, workdir)
        cpp_truth_s = time.perf_counter() - t0

        # Build every Python Layout + enumerate (this is where the JIT compiles
        # the handful of structures on first touch, then it's pure eval).
        from pymosaic import _render

        t0 = time.perf_counter()
        layouts = [Layout(shape=s["shape"], stride=s["stride"]) for s in specs]
        build_s = time.perf_counter() - t0

        structures: set[tuple] = set()
        total_elems = 0
        mism = 0

        # Per-function timing. Each block runs the real Python binding path and
        # checks its FNV hash against the C++ ground truth.
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        for L, g in zip(layouts, golden):
            offs = L.enumerate()
            total_elems += len(offs)
            structures.add(_profile(L.shape))
            mism += scale_spec.fnv1a_offsets(offs) != g["h_eval"]
        timings["enumerate (flat eval)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for L, g in zip(layouts, golden):
            ms = _mode_sizes(L.shape)
            co = [L[_colex_coord(j, ms)] for j in range(g["size"])]
            mism += scale_spec.fnv1a_offsets(co) != g["h_eval"]
        timings["eval_coord L[c0,c1,..]"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx_cache: list[list] = []
        for L, g in zip(layouts, golden):
            crds = [L.idx2crd(j) for j in range(g["size"])]
            idx_cache.append(crds)
            leaves: list[int] = []
            for c in crds:
                leaves.extend(_render.flatten(c) if isinstance(c, tuple) else (c,))
            mism += scale_spec.fnv1a_offsets(leaves) != g["h_idx2crd"]
        timings["idx2crd"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for L, g, crds in zip(layouts, golden, idx_cache):
            ro = [L.crd2idx(c) for c in crds]
            mism += scale_spec.fnv1a_offsets(ro) != g["h_eval"]
        timings["crd2idx round-trip"] = time.perf_counter() - t0

    print(f"  distinct structures (= JIT compiles) : {len(structures)}")
    print(f"  C++ generator build                  : {cpp_build_s * 1000:8.1f} ms")
    print(f"  Python spec generation               : {gen_s * 1000:8.1f} ms")
    print(f"  C++ ground-truth ({n:,} layouts)     : {cpp_truth_s * 1000:8.1f} ms")
    print(f"  Python Layout build + first-touch JIT: {build_s * 1000:8.1f} ms")
    print(f"  total elements per layout-pass       : {total_elems:,}")
    print("\n  per-function validation (Python binding vs C++ ground truth):")
    for name, secs in timings.items():
        print(
            f"    {name:24s} : {secs * 1000:8.1f} ms  "
            f"({n / secs:,.0f} layouts/s, {total_elems / secs / 1e6:.2f} M elems/s)"
        )
    print(f"\n  mismatches vs C++                    : {mism}")
    if mism:
        print("  !! MISMATCH — Python diverges from C++ ground truth")
        return 1
    print("  All functions validated against C++ ground truth.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--bench"]
    n = int(args[0]) if args else 50_000
    raise SystemExit(_benchmark(n))
