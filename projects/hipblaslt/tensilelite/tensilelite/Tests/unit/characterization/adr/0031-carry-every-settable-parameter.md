# ADR 0031: Carry every settable parameter through `TensileLibLogicToYaml`

Status:  Accepted
Defect:  none — behavior is intended

## Context
`TensileLibLogicToYaml` turns one solution of a logic file into a tuning config
that should rebuild the same kernel. It chose what to emit from two defaults
tables: a solution key was written only if
`GlobalParameters.defaultBenchmarkCommonParameters` had an entry for it, and a
problem type key only if `Problem._defaultProblemType` did. Neither table is the
list of what a config can set — `ValidParameters.validParameters` is — so every
settable parameter without a default entry was dropped: `PrefetchGlobalReadA`
and `PrefetchGlobalReadB` (set together or not at all, so there is no default to
list), `TDMFuse`, and on the problem type `ActivationType` and
`BiasDataTypeList`. Comparing each problem type value with a registry default
was also wrong on its own terms, since ProblemType derives many defaults from
other keys.

Three more gaps had the same effect:
- Dict-format logic stores every value equal to the file's `DefaultSolution`
  only there. The converter never merged it back, so most dict-format solutions
  raised `KeyError` (`MIArchVgpr`, `WorkGroup`) and the rest silently took
  today's defaults (`MaxOccupancy` 64 where the file records 40).
- The build target was not emitted, so Tensile built for whatever GPU it
  detected, and a stepping's solution (gfx1250-strict) built as its base
  architecture.
- Values recorded in a form the config validator rejects (0/1 for a bool
  parameter), or a `MinimumRequiredVersion` from another major version, made
  the emitted config unloadable.

## Decision
Take every choice from Tensile's own registries and code, so no parameter is
named in the converter:
- **Solution parameters.** Fill the solution the way `LibraryIO` reads it
  (`fillSolutionDefaults`, now shared), then carry every `validParameters` key it
  records. Values are converted to the registry's types when lossless, left out
  when they equal `defaultSolution` (what `Solution` fills in for an omitted
  key), and left out with a warning when Tensile's validator rejects them for the
  target architecture.
- **Problem type.** Start from every key ProblemType reads (an AST scan of
  `Problem.py`) and drop a key only if building ProblemType without it gives the
  same state as building the recorded block the way `LibraryIO` does
  (`normalizeLogicProblemType`, now shared).
- **Keys with a home elsewhere in the config.** `MatrixInstruction` and
  `WorkGroup` go in `Groups`. The target — the logic's `ArchitectureName`, which
  TensileCreateLibrary matches logic files by — goes in `GlobalParameters.ISA`,
  plus `Architecture` when it is a stepping (`steppingArchOf`), the one thing the
  ISA cannot express. A handwritten custom kernel becomes a
  `CustomKernels` entry carrying its `InternalSupportParams`, with no fork
  parameters. A generated kernel's `CustomKernel` block is the stamp
  `KernelWriter` writes during codegen, so it is rebuilt rather than carried:
  pinning it would force the kernel name to the stamped, possibly stale, string.
- **Version.** Keep the recorded `MinimumRequiredVersion` when
  `versionIsCompatible` accepts it; otherwise require the running version, with
  a warning.

Three existing expectations change: `test_form_fork_params_includes_nondefault_fork_key`
no longer patches the removed `defaultBenchmarkCommonParameters` lookup,
`test_set_global_params_non_i8` records a version this Tensile accepts, and the
gfx950 golden in `Tests/unit/test_TensileLibLogicToYaml.py` gains `ISA`, loses the
problem type keys the rest of its block implies, and writes bools as bools. The
old golden is rejected by today's validator (`ExpandPointerSwap = 0 (int);
expected bool`).

## Consequences
- A parameter Tensile adds to `validParameters`, or a problem type option
  ProblemType starts reading, is carried with no change to the converter.
- Across two solutions from each of the 2,802 shipped logic files (5,213), the
  config is rebuilt through the config-driven path and compared with the logic
  entry. Before this change 1,685 configs could not be produced or loaded and
  643 rebuilt the same kernel and solution names; now 5,142 do. Of the rest, 36
  carry a generated `CustomKernel` stamp whose name an older Tensile computed,
  13 record `StoreVectorWidth` 16, which the validator rejects as input, and 22
  are handwritten custom kernels whose own `custom.config` the `CustomKernels`
  path does not accept.
- What no config can express is reported instead of lost silently:
  `InternalSupportParams` of a generated kernel (a config accepts them only
  beside `CustomKernels`) and validator-rejected values print a warning.
- `test_parameter_carry_char.py` rebuilds four fixtures' solutions from the
  emitted config through the config-driven path and compares kernel and
  solution names, so a parameter the converter stops carrying fails a test.

**Rejected alternatives:**
- Add the missing keys to the defaults tables, or keep a list in the converter —
  rejected: the list has to grow with every parameter and silently drops the
  next one it misses, and some keys (the PrefetchGlobalRead pair) have no
  default to add.
- Emit every recorded key unconditionally — rejected: derived values outside the
  validator's domain make the config unloadable, and the generated
  `CustomKernel` stamp pins stale names.
