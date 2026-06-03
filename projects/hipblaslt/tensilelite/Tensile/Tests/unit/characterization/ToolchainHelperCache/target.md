# Characterization target — `Tensile/Toolchain/HelperKernelCache.py`

Part of the master-plan remaining-module sweep. **Before 90.8% → after ~97.7%
line** (87 stmts, 2 miss). Drives `_computeCacheKey` (determinism + arch/asan
variance), `_checkCache` (missing/empty/valid/zero-size), `_populateCache`
(normal + already-exists + rename-failure cleanup), `_evictStale`
(missing-dir / stale-evict / fresh-keep / .tmp-skip), and `HelperKernelCache`
(disabled / miss→store→hit / copy-failure-falls-through), with a fake compiler +
tmp dirs (no real compilation). Residual: L104, L150 — two `except OSError`
defensive arms (stat/unlink mid-failure) not forced; line coverage ≥95%.
