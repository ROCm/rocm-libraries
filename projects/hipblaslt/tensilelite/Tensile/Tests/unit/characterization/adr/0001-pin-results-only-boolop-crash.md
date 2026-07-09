# ADR 0001: Pin the `--results-only` constraint crash in TensileBenchmarkCluster

Status:  Accepted
Defect:  AIHPBLAS-XXXX (to file)

## Context
While characterizing `TensileBenchmarkCluster`, the `--results-only` flag (alone) raises `AssertionError: Constraint evaluation failed: RunDeployStep or RunBenchmarkStep or RunResultsStep` during construction.

Root cause: `ExpressionEvaluator`'s `BoolOp` handler (`Configuration.py:651-652`) only evaluates `node.values[0]` and `node.values[1]`, ignoring `values[2:]`. Python parses `a or b or c` as a single `BoolOp(Or, values=[a, b, c])`, so the constraint collapses to `a or b`. With `--results-only` only the third operand (`RunResultsStep`) is True, so the constraint evaluates `False or False` and fails. `--deploy-only`, `--run-only`, and `--run-and-results-only` happen to leave one of the first two operands True, so they survive. This is a real, user-facing bug: any boolean constraint with three or more operands whose truth depends on the 3rd+ operand is mis-evaluated.

## Decision
Pin the actual behavior — a test asserting `--results-only` raises `AssertionError` — instead of asserting the intended-but-unreachable workflow tuple `(False, False, True)`. Do not change `Configuration.py` (add-only).

## Consequences
The golden encodes a real bug on purpose, so the suite stays an honest record of current behavior and the regression net holds rather than masking the defect behind a green test. The fix belongs in a separate change (fold `values[2:]` into the `BoolOp` evaluator) with its own regression coverage; when that lands, flip this golden and supersede this ADR.

Residual coverage for the module: 192 stmts, 1 miss (line 120, the bare-`except` swallow when a task subdir already exists) → 99.51%.
