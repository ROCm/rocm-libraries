# Characterization target — `Tensile/Common/Parallel.py`

Part of the master-plan remaining-module sweep. **Before 39.5% → after ~80.6%
line** (124 stmts, 24 miss). Drives the pure helpers (`joblibParallelSupports
Generator`, `CPUThreadCount` enable/cap/all-cores, `OverwriteGlobalParameters`,
`pcallWithGlobalParams*`, `apply_print_exception` all 3 paths) and the
single-threaded / `n_jobs=1` **in-process** paths of `ParallelMap` /
`ParallelMap2` (progress-bar + dummy-pool + joblib-n_jobs=1).

**Accepted <95% — see DECISIONS D4.** The residual lines are the real
fork/spawn parallel paths (`ProcessingPool` multiprocessing.Pool,
`ParallelMapReturnAsGenerator` ProcessPoolExecutor, the joblib generator-return
branch) + the Windows-only `os.name=="nt"` branch. These fork OS processes and
are flaky/slow/nondeterministic to unit-test (they exercise the OS scheduler,
not our logic) — the same rationale that excludes the codegen surface.

NOTE: `import Tensile.Common.Parallel as P` yields joblib's `Parallel` class
(package shadowing); the module is loaded via `importlib.import_module`.
