# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example 5 — Inspect the on-disk JIT cache.

What this shows:
  * Point the JIT cache at a fresh per-run temp directory so the
    example is reproducible and doesn't churn your real cache.
  * Build a few Layouts and trigger evaluation to populate the cache.
  * List cache entries, read manifest.json, peek at the generated
    module.cpp and compile.log.
  * Demonstrate cache_info() and clear_cache() from the public
    ``mosaic.jit`` module.

Environment knobs (just for reference; this script sets the first one
for itself):
  MOSAIC_JIT_CACHE  override the cache root (default ~/.cache/mosaic/jit)
  MOSAIC_JIT_DEBUG  if "1", emit preprocessed source + assembly per entry
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    # Point the cache at a temp dir BEFORE importing mosaic.layout so
    # that the first cache_root() call resolves to it. Strictly the env
    # var is re-read on each call so post-import would also work, but
    # this matches how a fresh process would behave.
    tmp = tempfile.mkdtemp(prefix="mosaic-jit-example-")
    os.environ["MOSAIC_JIT_CACHE"] = tmp

    from pymosaic import jit as mosaic_jit
    from pymosaic.layout import Layout

    print(f"  cache root  : {mosaic_jit.cache_root()}")

    section("Populate the cache — modules are keyed on STRUCTURE")
    # The dynamic route compiles one .so per nesting profile and passes
    # shape/stride values at call time. So the first two layouts below
    # (both flat rank-2) SHARE a single cache entry despite different
    # values/strides; only the hierarchical layout adds a second entry.
    layouts = [
        Layout.row_major(4, 8),
        Layout.column_major(4, 8),  # same structure as above
        Layout(shape=((2, 2), 3), stride=((1, 2), 4)),  # distinct structure
    ]
    for L in layouts:
        _ = L[0]  # trigger JIT compile
        print(f"  built  {L!r}  ->  {L}")
    print("  (expect 2 cache entries, not 3 — the two flat rank-2 layouts share one)")

    section("cache_info() — summary stats")
    info = mosaic_jit.cache_info()
    print(f"  entries     : {info['entries']}")
    print(f"  size_bytes  : {info['size_bytes']:,} bytes")
    print(f"  path        : {info['path']}")
    print(f"  schema ver  : {info['version']}")

    section("Inspect one cache entry on disk")
    cache_root = pathlib.Path(info["path"])
    entry = next(p for p in cache_root.iterdir() if p.is_dir())
    print(f"  entry dir   : {entry}")
    for name in sorted(p.name for p in entry.iterdir()):
        size = (entry / name).stat().st_size
        print(f"    {name:30s}  {size:>10,} bytes")

    section("manifest.json — full metadata (cache invalidation key)")
    manifest = json.loads((entry / "manifest.json").read_text())
    for k, v in manifest.items():
        print(f"  {k:14s}: {v}")

    section("python_repr.txt — the originating Python call")
    print((entry / "python_repr.txt").read_text().strip())

    section("module.cpp — first 30 lines of the generated source")
    src = (entry / "module.cpp").read_text().splitlines()
    for line in src[:30]:
        print(f"  {line}")
    print(f"  ... ({len(src)} lines total)")

    section("compile.log — first 10 lines (always preserved, hot or cold)")
    log = (entry / "compile.log").read_text().splitlines()
    for line in log[:10]:
        print(f"  {line}")

    section("clear_cache() — wipe a single entry")
    removed = mosaic_jit.clear_cache(key=entry.name)
    print(f"  removed {removed} entry (key={entry.name})")
    print(f"  remaining   : {mosaic_jit.cache_info()['entries']}")

    section("clear_cache() — wipe everything")
    removed = mosaic_jit.clear_cache()
    print(f"  removed {removed} entries")
    print(f"  remaining   : {mosaic_jit.cache_info()['entries']}")

    print(f"\n  (temp cache lives at {tmp}; safe to delete manually if you want)")


if __name__ == "__main__":
    main()
