# BenchmarkProblems.py — characterization target (cache layer)

BenchmarkProblems.py is a benchmark build/run orchestrator importing
KernelWriter / ClientWriter / KernelWriterAssembly / Assembler (codegen + GPU),
out of scope for this effort. This suite pins the pure **cache helper layer**:
_cacheDataMatches, _computeCacheKey, _readCacheIfValid (all branches: missing
file / unreadable / match / missing-field / mismatch), _loadCacheIfMatches,
_loadLegacyCacheIfMatches, _resetCacheDir.

Those helpers (lines ~80-141) are now fully covered. The build/run path
(writeBenchmarkFiles, _benchmarkProblemType, _generate*Solutions, main) requires
a real assembler, derived Solutions and codegen and is documented resistance;
it remains covered only by the existing integration tests.
