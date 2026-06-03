# TensileBenchmarkCluster.py — characterization target

Pins the SLURM cluster-benchmark orchestrator: argv parsing, full config
initialization, workflow-step derivation, backend delegation, results merging,
and the docker/script builders. Subprocess/docker/gzip/template work is stubbed
at the module seams (`subprocess`, `gzip`, `ScriptWriter`, `BenchmarkSplitter`,
`mergePartialLogics`).

Coverage: 192 stmts, 1 miss → 99.51% (line 120 = bare-except dir-exists guard).

Pinned quirks:
- `__parseArgs` ignores its `cmdlineArgs` param and reads `sys.argv` directly.
- `--results-only` alone raises `AssertionError` at construction — a real bug in
  `ExpressionEvaluator` BoolOp handling (only evaluates the first two operands of
  a 3-way `or`). See DECISIONS D12.
