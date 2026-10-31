# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark the dynamic JIT route across all layout examples.

Answers two questions:

  1. *How often do we compile?* The dynamic route keys the compiled
     ``.so`` on layout **structure** (nesting profile), passing
     shape/stride **values** at call time. So compilation happens once
     per distinct structure — NOT once per distinct ``(shape, stride)``
     value tuple. This script drives every example through one shared
     cache and counts the real compiles, then contrasts:
         distinct value-layouts requested   (what the OLD static route
                                              would have compiled)
         distinct structures compiled        (what the dynamic route does)

  2. *What does it cost?* Cold compile (per structure), warm dlopen
     (fresh process, .so already on disk), and hot eval (per call).

Run::

    PYTHONPATH=python/pymosaic python examples/python/pymosaic/benchmark_all.py

Excluded from the driven set: ``benchmark.py`` (self-contained timing),
``05_cache_inspection.py`` (calls clear_cache — would wipe the shared
cache mid-run), and the ``visualize.py`` scripts (need matplotlib and
emit PNGs). Those exercise no layout structures the others don't.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import runpy
import statistics
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent

# Workload examples to drive (relative to this dir). One representative
# per layout pattern; see module docstring for what's excluded and why.
_EXAMPLES = [
    "01_basics.py",
    "02_hierarchical.py",
    "03_slicing.py",
    "basic_tiling/check_tiling.py",
    "slice/slice.py",
    "mfma/mfma_layout.py",
    "gemm_grid/grid_mapping.py",
]


# ---------------------------------------------------------------------------
# Instrumentation — funnel every example into one shared cache and record
# each module request + actual compile.
# ---------------------------------------------------------------------------

_SHARED_CACHE = pathlib.Path(tempfile.mkdtemp(prefix="mosaic-jit-bench-all-"))

from pymosaic import _jit, _render  # noqa: E402

_orig_get_module = _jit.get_module
_orig_compile = _jit._compile

# Requests: (example, shape, stride, structure_key) for every get_module call.
_requests: list[tuple[str, tuple, tuple, str]] = []
# Compiles: (key, seconds) for every actual _compile invocation.
_compiles: list[tuple[str, float]] = []
_current_example = "<startup>"


def _patched_cache_root() -> pathlib.Path:
    _SHARED_CACHE.mkdir(parents=True, exist_ok=True)
    return _SHARED_CACHE


def _patched_get_module(shape, stride):
    key = _jit.cache_key(shape, stride)
    _requests.append((_current_example, shape, stride, key))
    return _orig_get_module(shape, stride)


def _patched_compile(*, cache_dir, key, shape, stride):
    t0 = time.perf_counter()
    _orig_compile(cache_dir=cache_dir, key=key, shape=shape, stride=stride)
    _compiles.append((key, time.perf_counter() - t0))


def _install_patches() -> None:
    _jit.cache_root = _patched_cache_root
    _jit.get_module = _patched_get_module
    _jit._compile = _patched_compile


def _drop_loaded_jit_modules() -> None:
    """Forget already-dlopen'd JIT modules so a rerun pays the dlopen
    cost again (simulating a fresh process), while the on-disk .so cache
    stays warm."""
    for name in [m for m in sys.modules if m.startswith("mosaic_jit_")]:
        del sys.modules[name]


def _run_example(rel_path: str) -> str | None:
    """Execute one example with stdout/stderr suppressed. Returns an
    error string on failure, else None."""
    global _current_example
    _current_example = rel_path
    path = _HERE / rel_path
    saved_path = list(sys.path)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            runpy.run_path(str(path), run_name="__main__")
        return None
    except SystemExit:
        return None  # examples that `raise SystemExit(main())`
    except Exception as exc:  # noqa: BLE001 — report, don't abort the sweep
        return f"{type(exc).__name__}: {exc}"
    finally:
        sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.1f} ms"


def _fmt_us(s: float) -> str:
    return f"{s * 1_000_000:.3f} µs"


def main() -> int:
    _install_patches()
    print(f"shared cache : {_SHARED_CACHE}\n")

    # ------------------------------------------------------------------
    # Pass 1 — COLD. Empty shared cache: each new structure compiles once.
    # ------------------------------------------------------------------
    print("Pass 1 — cold (empty cache)")
    print("-" * 60)
    cold_wall_t0 = time.perf_counter()
    for rel in _EXAMPLES:
        before = len(_compiles)
        err = _run_example(rel)
        n_compiles = len(_compiles) - before
        status = "OK " if err is None else "ERR"
        note = "" if err is None else f"  <-- {err}"
        print(f"  [{status}] {rel:34s} compiles: {n_compiles}{note}")
    cold_wall = time.perf_counter() - cold_wall_t0

    # Aggregate structure / value statistics from the recorded requests.
    ok_requests = [r for r in _requests]
    distinct_values = {(s, d) for (_, s, d, _) in ok_requests}
    distinct_structs = {k for (_, _, _, k) in ok_requests}
    compile_time = sum(s for _, s in _compiles)

    print("\nCompile frequency")
    print("-" * 60)
    print(f"  module requests (Layout first-uses)      : {len(ok_requests)}")
    print(f"  distinct value-layouts (shape, stride)   : {len(distinct_values)}")
    print(f"  distinct STRUCTURES  → actual compiles   : {len(distinct_structs)}")
    print(
        f"  cache entries on disk                    : "
        f"{sum(1 for p in _SHARED_CACHE.iterdir() if p.is_dir())}"
    )
    if distinct_structs:
        ratio = len(distinct_values) / len(distinct_structs)
        print(
            f"  dynamic vs static compiles               : "
            f"{len(distinct_structs)} vs {len(distinct_values)}  ({ratio:.1f}x fewer)"
        )

    # Per-structure breakdown: how many value-layouts collapse onto each.
    print("\n  structures compiled (one .so each):")
    by_struct: dict[str, set] = {}
    for _, s, d, k in ok_requests:
        by_struct.setdefault(k, set()).add((s, d))
    # Show a readable signature for each: render the structure once.
    for k, values in sorted(by_struct.items(), key=lambda kv: -len(kv[1])):
        any_s, any_d = next(iter(values))
        sig = _render.structure_signature(any_s, any_d)
        # Compact the signature: strip namespace noise for display.
        compact = (
            sig.replace("::mosaic::", "")
            .replace("::std::size_t", "_")
            .replace("Layout", "L")
            .replace("Shape", "S")
            .replace("Stride", "D")
        )
        print(f"    {len(values):>2d} value-layouts  ->  {compact}")

    print(f"\n  cold compile wall time (all examples)    : {_fmt_ms(cold_wall)}")
    print(
        f"  pure compile time (sum of {len(_compiles)} compiles) : {_fmt_ms(compile_time)}"
    )
    if _compiles:
        per = [s for _, s in _compiles]
        print(
            f"  per-structure compile                    : "
            f"mean {_fmt_ms(statistics.mean(per))}, max {_fmt_ms(max(per))}"
        )

    # ------------------------------------------------------------------
    # Pass 2 — WARM. Cache populated; drop loaded modules to pay dlopen.
    # ------------------------------------------------------------------
    print("\nPass 2 — warm rerun (.so cached, modules dropped)")
    print("-" * 60)
    _drop_loaded_jit_modules()
    compiles_before = len(_compiles)
    warm_t0 = time.perf_counter()
    for rel in _EXAMPLES:
        _run_example(rel)
    warm_wall = time.perf_counter() - warm_t0
    warm_compiles = len(_compiles) - compiles_before
    print(f"  rerun wall time                          : {_fmt_ms(warm_wall)}")
    print(f"  compiles triggered                       : {warm_compiles}  (expect 0)")
    print(f"  speedup vs cold                          : {cold_wall / warm_wall:.1f}x")

    # ------------------------------------------------------------------
    # Hot eval — per-call cost once the module is loaded.
    # ------------------------------------------------------------------
    print("\nHot eval (module loaded, repeated calls)")
    print("-" * 60)
    from pymosaic.layout import Layout

    L = Layout.row_major(128, 128)
    n = len(L)
    _ = L[0]  # ensure loaded
    ITERS, REPEATS = 50_000, 5
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for i in range(ITERS):
            _ = L[i % n]
        samples.append((time.perf_counter() - t0) / ITERS)
    print(
        f"  flat eval L[i]   : mean {_fmt_us(statistics.mean(samples))}, "
        f"fastest {_fmt_us(min(samples))}"
    )

    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for i in range(ITERS):
            _ = L[i % 128, i % 128]
        samples.append((time.perf_counter() - t0) / ITERS)
    print(
        f"  coord eval L[r,c]: mean {_fmt_us(statistics.mean(samples))}, "
        f"fastest {_fmt_us(min(samples))}"
    )

    print("\nTakeaway")
    print("-" * 60)
    print("  Compilation is amortised over STRUCTURE, not value. Once the")
    print("  handful of nesting profiles your program uses are compiled,")
    print("  every later layout — any shape, any stride — is a cache hit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
