# Characterization target — `Tensile/Toolchain/Source.py`

Part of the master-plan remaining-module sweep. **Before 41.5% → after ~95.5%
line** (66 stmts, 3 miss). Drives `_computeSourceCodeObjectFilename` (fallback /
TensileLibrary-variant / other), `makeSourceToolchain`, and
`buildSourceCodeObjectFiles` via stub compiler/bundler with the helper cache
disabled (cache-miss orchestration: compile -> unbundle -> move). 5 tests.
Residual: L113-115 (cache-HIT early return — needs a populated cache) + 2
branches. The real compiler/bundler subprocess invocation is injected (stubbed),
not run (D8).
