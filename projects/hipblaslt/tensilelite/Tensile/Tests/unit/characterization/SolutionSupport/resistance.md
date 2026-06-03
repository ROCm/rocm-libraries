# Resisting items — `Solution.py` support slice (L165-439)

The targeted slice reaches **100% line and branch** (0 missing lines, 0 partial
branches in L165-439). Nothing in-slice resisted. This file records the scoping
boundary and the one import gotcha. New file in the per-target dir per the
add-only rule.

## Out of slice (not resistance — explicitly deferred)

| Region | Why deferred |
|---|---|
| `_deriveAndValidateMXScaleLayoutAndTransport` (L57-162) | Takes `asmCaps`/`archCaps`; cap-coupled. Belongs to a later cap-driven slice. |
| `Solution` class (L444-5230) | The bulk of the file (~3000 stmts), heavily coupled to `asmCaps`/`archCaps`/`assembler`/`isaInfoMap`. The next campaign slice(s), to be driven with the established real `isaInfoMap`/`assembler` fixtures. |

## Import gotcha (resolved)

`SolutionStructs/__init__.py` re-exports the `Solution` *class*
(`from .Solution import Solution`), which **shadows the submodule attribute** on
the package. As a result `import Tensile.SolutionStructs.Solution as S` binds
`S` to the class (whose `Fbs`/`isExtractableIndex`/... are absent), not the
module. The suite loads the module robustly via
`importlib.import_module("Tensile.SolutionStructs.Solution")`. (The `Problem` /
`Naming` submodules are not re-exported, which is why their suites could use a
plain `import ... as`.)

## Branch reachability notes (all reached)

- `validateParameterTypes` existing-collector-key branch (L275 false side) is
  reached by `test_validate_mismatch_accumulates` (same `(param, actual,
  expected)` twice).
- The `ActivationArgs` "specific type" else branch (neither `none` nor
  `all`/`hipblaslt_all`) is unreachable through a real `ProblemType` (which only
  yields those three), so it is driven with a dict stand-in
  (`{"ActivationType": ActivationType("relu")}`) — a documented technique, the
  branch is genuine module behaviour.
- `printExit` (`sys.exit(-1)`) paths in `BiasTypeArgs` / `ActivationArgs` are
  pinned via `pytest.raises(SystemExit)`.

## Determinism technique (not a gap)

- The module-global `_typeMismatchCollector` is isolated per test
  (clear → run → capture delta → restore) so snapshots are minimal and the
  shared `-m unit` session stays green (verified: 1800 passed).
- `ProblemType` inputs are built minimal (mirroring the ProblemType suite); the
  arg-dataclass snapshots capture structured fields (`factorDims`, dtype names,
  setting counts), not live objects.
