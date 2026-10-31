# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Micro-benchmark the JIT compile vs the per-call eval cost for the
basic_tiling layout.

Three numbers are isolated:

  1. Cold JIT compile — ``ninja + g++ + link`` for a layout never
     before built on this machine. Happens once per unique
     ``(shape, stride)``. Measured against a fresh temp cache so the
     timing is honest regardless of your real ``~/.cache/mosaic/jit/``.

  2. Warm dlopen — the .so is already on disk (or in this process'
     ``sys.modules`` cache after step 1). A *fresh* ``Layout`` proxy
     with the same ``(shape, stride)`` skips the compile but still
     pays the dlopen + nanobind module-setup cost on first use. This
     is what every program startup pays.

  3. Hot eval — ``l2_layout[i]`` once the proxy already holds its
     module reference. Pure C++ call, nanobind dispatch overhead +
     constant-folded coord arithmetic.

Run::

    PYTHONPATH=python/pymosaic python examples/python/pymosaic/basic_tiling/benchmark.py
"""

from __future__ import annotations

import os
import pathlib
import statistics
import sys
import tempfile
import time

# Force a fresh JIT cache BEFORE importing mosaic so the cold timing
# below is honest. The env var is re-read on every cache_root() call,
# so importing first and setting later would still work, but this is
# clearer.
_TMP_CACHE = tempfile.mkdtemp(prefix="mosaic-jit-tiling-bench-")
os.environ["MOSAIC_JIT_CACHE"] = _TMP_CACHE

# Local import — shares the blocked_product helper with check_tiling.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_tiling import blocked_product  # noqa: E402

from pymosaic._jit import cache_key  # noqa: E402
from pymosaic.layout import Layout  # noqa: E402

HEIGHT, WIDTH = 8, 8
TOTAL = HEIGHT * WIDTH
L2_Y, L2_X = 2, 2


def build_l2_layout() -> Layout:
    """Build the same layout check_tiling.py uses. Pure construction
    so we can repeat it cheaply to get a fresh proxy."""
    inner = Layout.row_major(L2_Y, L2_X)
    outer = Layout.column_major(HEIGHT // L2_Y, WIDTH // L2_X)
    return blocked_product(inner, outer)


def time_eval_loop(L: Layout, iters: int) -> tuple[float, float]:
    """Run ``L[i % TOTAL]`` ``iters`` times. Returns (total_s, per_call_s)."""
    t0 = time.perf_counter()
    for i in range(iters):
        _ = L[i % TOTAL]
    total = time.perf_counter() - t0
    return total, total / iters


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:>10.2f} ms"


def fmt_us(seconds: float) -> str:
    return f"{seconds * 1_000_000:>10.2f} µs"


def main() -> int:
    print(f"Bench cache : {_TMP_CACHE}\n")

    # ------------------------------------------------------------------
    # Stage A — Cold JIT compile (ninja + g++ + link)
    # ------------------------------------------------------------------
    print("Stage A — cold JIT compile (fresh cache, never built)")
    print("------------------------------------------------------")
    L = build_l2_layout()
    # NB: use repr (pure Python) — print(f"... {L}") would call __str__
    # which goes through _ensure_module() and steals the compile time
    # from the cold-path timer below.
    print(f"  layout = {L!r}")
    assert (
        L._module is None
    ), "blocked_product must return a fresh proxy with no module loaded"

    # _ensure_module() does the work of "fetch .so from cache or build
    # it now". Since the cache is empty, this is the full ninja run.
    t0 = time.perf_counter()
    L._ensure_module()
    cold_compile_s = time.perf_counter() - t0
    key = cache_key(L.shape, L.stride)
    so_path = pathlib.Path(_TMP_CACHE) / key / f"mosaic_jit_{key}.so"
    so_kb = so_path.stat().st_size / 1024 if so_path.is_file() else float("nan")
    print(
        f"  cold compile          : {fmt_ms(cold_compile_s)}    (.so = {so_kb:.0f} KB)"
    )

    # First eval *after* the compile — separated out so we don't double-count.
    t0 = time.perf_counter()
    _ = L[0]
    first_eval_after_compile_s = time.perf_counter() - t0
    print(
        f"  first eval (post-compile, hot proxy)   : {fmt_us(first_eval_after_compile_s)}"
    )

    # ------------------------------------------------------------------
    # Stage B — Warm dlopen (.so on disk, fresh Layout proxy)
    # ------------------------------------------------------------------
    print("\nStage B — warm dlopen (.so cached, fresh Layout proxy)")
    print("-------------------------------------------------------")
    # Drop the in-process module reference so _load_module() re-imports
    # the .so. This simulates "second run of the program" or "fresh
    # Layout created later in the same process".
    module_name = f"mosaic_jit_{key}"
    if module_name in sys.modules:
        del sys.modules[module_name]

    L2 = build_l2_layout()
    assert L2._module is None, "fresh Layout should not have a module loaded"

    t0 = time.perf_counter()
    L2._ensure_module()
    warm_dlopen_s = time.perf_counter() - t0
    print(f"  dlopen + module setup : {fmt_ms(warm_dlopen_s)}")
    print(f"  speedup vs cold       : {cold_compile_s / warm_dlopen_s:>10.0f}x")

    t0 = time.perf_counter()
    _ = L2[0]
    first_eval_post_dlopen_s = time.perf_counter() - t0
    print(
        f"  first eval (post-dlopen)               : {fmt_us(first_eval_post_dlopen_s)}"
    )

    # ------------------------------------------------------------------
    # Stage C — Hot eval cost (same proxy, repeated calls)
    # ------------------------------------------------------------------
    print("\nStage C — hot eval (same proxy, repeated calls)")
    print("------------------------------------------------")
    # A short loop, repeated several times for jitter estimation.
    ITERS = 10_000
    REPEATS = 5
    samples: list[float] = []
    for _ in range(REPEATS):
        _, per_call = time_eval_loop(L2, ITERS)
        samples.append(per_call)
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    fastest = min(samples)
    print(f"  per-call eval (avg of {REPEATS}x {ITERS} iters):")
    print(f"    mean ± stdev        : {fmt_us(mean)} ± {fmt_us(stdev).strip()}")
    print(f"    fastest run         : {fmt_us(fastest)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nSummary")
    print("-------")
    print(
        f"  cold compile : {fmt_ms(cold_compile_s)}    one-time, per unique (shape, stride)"
    )
    print(f"  warm dlopen  : {fmt_ms(warm_dlopen_s)}    per fresh process / proxy")
    print(f"  hot eval     : {fmt_us(mean)}    per call once module is loaded")
    print(
        f"  ratios       : compile is {cold_compile_s / warm_dlopen_s:.0f}x slower than dlopen, "
        f"and {cold_compile_s / mean:.0e}x slower than a hot eval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
