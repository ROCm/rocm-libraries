# Strict Type Validation for TensileLite Input YAMLs — Plan (revised)

## 1. Goal & scope

**Goal.** Malformed parameter types in TensileLite *input* YAMLs (the
test/benchmark configs read at the start of a Tensile run) must fail
loudly at the earliest validator that already owns the key, with a single
structured error message naming the file, the key path, the offending
value, its actual Python type, and the expected type(s). This closes the
source-side gap that `fix_yaml_types.py` patches downstream by regex.

**Preconditions.**

- All YAML reads use `LibraryIO.read` / `readYAML`, which default to
  `StrictTypeLoader` (see `tensilelite/Tensile/LibraryIO.py`,
  `StrictTypeLoader` class and `readYAML`). Without `StrictTypeLoader`,
  PyYAML collapses `0`/`1` to `False`/`True` at parse time and validators
  never see the original type. Any reader that bypasses
  `StrictTypeLoader` is a hole in the plan and must be converted before
  the strict gate goes live — see Step 0.

**Entry points covered (v1).** The following entry points read input-style
YAMLs that feed `assignGlobalParameters`, `ProblemType.__init__`, or
`checkParametersAreValid`:

- `Tensile.Tensile()` (full benchmark pipeline) — primary path.
- `TensileClientConfig` (`tensilelite/Tensile/TensileClientConfig.py`):
  parses an input config, builds a `ProblemType`, and calls
  `assignGlobalParameters`. Currently constructs `ProblemType(problemDict)`
  with one positional arg — a latent arity bug since the constructor
  requires `printIndexAssignmentInfo: bool` (see `Problem.py`,
  `ProblemType.__init__`). Filed as blocker B1 on Step 5 below.
- `BenchmarkSplitter` (`tensilelite/Tensile/BenchmarkSplitter.py`):
  reads its config via bare `yaml.safe_load` — bypasses
  `StrictTypeLoader`. Switched to `LibraryIO.read` as part of Step 0.

**Entry points NOT covered (v1).**

- `TensileRetuneLibrary`, `TensileUpdateLibrary`, `GenerateSummations` —
  spot-checked: each reads *library logic* YAMLs (Tensile-generated
  output), not input-style configs. Out of scope. `GenerateSummations`
  uses `yaml.load(stream, yaml.SafeLoader)` on tensile library output and
  is not converted by Step 0.
- Library-logic strictness in general (see deferral note on
  `validateParameterTypes` below).
- `CustomKernels` directory — text/INI-style files, not validated
  through this path; flagged for a one-shot `fix_yaml_types.py` sweep in
  Step 0.

**In-scope validators (v1).** Every YAML block that has an existing
validator gets that validator extended in place. One validator per
section. No new central dispatcher.

- `GlobalParameters:` — extend `assignGlobalParameters` in
  `tensilelite/Tensile/Common/GlobalParameters.py`.
- `BenchmarkProblems[*][0]` (ProblemType) — promote
  `validateProblemTypeParameterTypes` in
  `tensilelite/Tensile/SolutionStructs/Problem.py` (called from
  `ProblemType.__init__`) from warning-collector to raise-on-mismatch.
- `BenchmarkProblems[*][1].BenchmarkCommonParameters`, `ForkParameters`,
  and `ForkParameters.Groups[*][*]` — extend `checkParametersAreValid` in
  `tensilelite/Tensile/Common/ValidParameters.py`, invoked from
  `BenchmarkStructs.getConfigParameters`. This is the **primary new
  strict gate** for the bulk of input-YAML keys.
- `InternalSupportParams:` — today `checkParametersAreValid` has an
  early `return` on this key that is dead code (see B3 below). Add a
  **sibling validator** rather than folding into
  `checkParametersAreValid`.
- `LibraryLogic:` — at the per-key merge loop in `generateLogic()`
  (in `tensilelite/Tensile/LibraryLogic.py`; `main()` simply delegates
  to `generateLogic()`), today silently drops unknown keys (it iterates
  `defaultAnalysisParameters`, not the user dict). Add an unknown-key +
  type check at that merge site. Numeric-range validation for
  `SolutionImportanceMin` is added inline at the same site. **No enum
  check for `LibraryType`** — it is a distance-metric label, not a
  fixed enum (see B2 below).

**Explicitly out of scope (v1).**

- `LibraryClient:` block. Only one in-tree YAML
  (`Tests/common/gemm/xfp32.yaml`) uses it and its body is empty. The
  three keys `ClientWriter` actually consumes (`ActivationArgs`,
  `FactorDimArgs`, `ICacheFlush`) have their real home in
  `BenchmarkFinalParameters` inside `BenchmarkProblems`, which is
  already validated. Cost > benefit.

- `ProblemSizes`, `Range`, `Exact`, `MatrixInstruction`, `WorkGroup`,
  `ThreadTile`, the various `IndexAssignments*`, `SetConstStride*`,
  `MirrorDims*` free-form list payloads. Structural lists with their own
  shape-checkers (e.g. `checkSpaceFillAlgoIsValid` /
  `checkSpaceFillAlgoWGMIsValid` in `ValidParameters.py`); already
  skipped naturally because their `validParameters` entry is `-1`. (See
  §3 — `[[]]` placeholder claim removed; the spot check found only `-1`.)

- Tuning UI YAMLs under `utilities/QuickTune/`.

- The `TestParameters:` block at the top of a test config. Consumed by
  the pytest harness, not by Tensile core, and not seen by any of the
  in-scope validators — no special-case skip needed.

- Cached/pickled YAMLs read by `BenchmarkProblems._readCache` and
  Tensile-generated YAMLs read by `TensileRetuneLibrary` — program
  output, not human-authored.

**Library-logic strictness deferral.** The post-construction
`validateParameterTypes` call in `Solution.__init__` also fires on the
library-logic load path (`LibraryIO.py`). For library-logic YAMLs this
remains the ONLY type validator, and library-logic strictness is out of
v1 scope. On the input-YAML path the call becomes redundant once Step 4
fires; Step 8 removes it on that path via a `strictTypeValidation`
kwarg (see B4). The library-logic-path invocation stays.

## 2. Where validation hooks in — one validator per section, extended in place

The previous draft proposed a new central module called once near
`Tensile.Tensile()` after `LibraryIO.read`. That approach is rejected:
`checkParametersAreValid`, `assignGlobalParameters`, and
`validateProblemTypeParameterTypes` each already own dispatch for their
section. A central module would route around them, duplicate
section-detection, and leave the existing validators as a parallel
half-dead pathway. The "no tactical fixes" rule applies: extend the
existing validators.

| Block | Validator (extended in place) | What the extension adds |
|---|---|---|
| `GlobalParameters` | `assignGlobalParameters` in `GlobalParameters.py` | Per-key type check against the override table + `type(globalParameters[key])`; the existing `printWarning` for unknown keys is promoted to raise. |
| `ProblemType` | `validateProblemTypeParameterTypes` in `Problem.py` (called from `ProblemType.__init__`) | Promoted to raise after collecting all mismatches within one `ProblemType` instance. |
| `BenchmarkCommonParameters` / `ForkParameters` / `ForkParameters.Groups` | `checkParametersAreValid` in `ValidParameters.py` (called from `BenchmarkStructs.getConfigParameters`) | Add a per-element type check alongside the existing name + value-membership check. |
| `InternalSupportParams` | New **sibling validator** in `ValidParameters.py`, called explicitly from `BenchmarkStructs.getConfigParameters` after `self.internalSupportParams` is populated. The existing early `return` in `checkParametersAreValid` is deleted as dead code in the same step. | Each key must exist in `defaultInternalSupportParams` and have `type(default_value)`. |
| `LibraryLogic` | `generateLogic()` in `LibraryLogic.py` (merge site `for parameter in defaultAnalysisParameters`) | Pre-check pass over the user-supplied `config` dict: every key must be in `defaultAnalysisParameters` and of the matching `type(default)`. `SolutionImportanceMin` numeric in `[0.0, 1.0]`. |

**Why this shape.**

- Each section's existing validator is the natural single owner. Putting
  the type check next to the existing membership/unknown-key check keeps
  the logic discoverable and removes the temptation to add a third
  validation pass elsewhere.
- The type-registry maps live in the right modules:
  `_expectedParamTypes` (the validParameters-derived map) and
  `_skipTypeCheck` are moved into `Tensile/Common/ValidParameters.py`
  alongside `validParameters` (see M1 below — avoids the new
  `ValidParameters → Solution` reverse import). `Solution.py` then
  imports them from there, matching the existing import direction
  `Common → Solution`.
- The shared exception type and key-path formatter (see §4) live in a
  small shared module. That is a one-screen utility, not a dispatcher.

## 3. How to derive the "expected type" for each parameter

Per registry; each rule lives next to the validator that owns it.

**`validParameters` (lists of allowed values).** Implemented in
`_getExpectedTypes` (moved into `ValidParameters.py`):

- Skip entries whose value is the `-1` sentinel (free-form). Examples:
  `MatrixInstruction`, `WorkGroup`, `ThreadTile`, `MIWaveGroup`,
  `MIWaveTile`, `SpaceFillingAlgo`, `SFCWGM`.
- Otherwise: expected type set = `{type(v) for v in allowed_values}`.
  Using `type()` not `isinstance()` is mandatory; see §5.

A `[[]]` placeholder pattern was claimed in the prior draft. **Verified
absent**: spot-checked `ValidParameters.py` — `MatrixInstruction`,
`SpaceFillingAlgo`, `SFCWGM`, `ClusterDim`, `WorkGroup`, `ThreadTile`,
`MIWaveGroup`, `MIWaveTile` all use `-1`. Rule removed.

Sentinel `-1` inside an otherwise-numeric list (e.g.
`LocalWritePerMfma: [i/100 for i in range(1, 3200)] + [-1]` in
`ValidParameters.py`) is handled automatically by the set-union: the
allowed type set becomes `{float, int}` and either is accepted. Other
sentinel-bearing parameters in `ValidParameters.py` (e.g. `MaxVgprNumber`,
`LdsBlockSizePerPad*`, `LdsPad*`) follow the same union-handles-it rule;
the test list enumerates one per sentinel class (§7).

The `_skipTypeCheck` set covers `DataType*` / `MacDataType*` /
`F32XdlMathOp` / `ISA` keys that get post-processed into typed objects
downstream. Reused unchanged.

The `checkParametersAreValid` extension iterates the existing `values`
list and applies the type check to each element. `ForkParameters.Groups`
already enters `checkParametersAreValid` per-key via `BenchmarkStructs`
with a single-element list wrap, so it inherits the same logic for free.

**`_defaultProblemType` (single defaults).** Existing rule in `Problem.py`:
expected type = `{type(default_value)}`. Inherits the same `_skipTypeCheck`
set for `DataType*` keys.

**`globalParameters` (single defaults), with override table.** Derivation
inside `assignGlobalParameters`: `expected = {type(default_value)}`. Two
classes of special-case:

1. `ignoreKeys` (existing list inside `assignGlobalParameters`): not
   user-settable globals; skip in the type loop.
2. `MinimumRequiredVersion`: checked separately at the top of
   `assignGlobalParameters`; skip in the type loop.

A new **override table** lives next to `globalParameters` in
`GlobalParameters.py` and annotates expected types for entries where the
default is `None` (M2). Skipping `None` defaults wholesale is a coverage
gap: `RocProfCounter: 42` (int) would pass silently if not annotated. The
table is a dict literal, e.g.:

```python
globalParameterTypeOverrides = {
    "ClientExecutionLockPath": {type(None), str},   # path or unset
    "ROCmSMIPath":             {type(None), str},   # path, populated at startup
    "CmakeCxxCompiler":        {type(None), str},   # path, populated at startup
    "RocProfCounter":          {type(None), str},   # counter spec or None
}
```

The validator consults the override table for any key whose default is
`None`; if the key has no entry, raise at validator init time so missing
annotations are caught immediately, not silently skipped. The verified
set of `None` defaults in `GlobalParameters.py` (confirmed by grep) is
the four entries above; the table is complete on day one. Annotations for
new `None`-defaulted entries are required by the validator.

Tuple-defaulted globals: **verified absent** (review M1 / §9.5 of prior
draft). Every `globalParameters["X"] = (\n    value  # comment\n)` line in
`GlobalParameters.py:40-340` is a single value in line-continuation
parens with no comma inside. No tuple handling needed. Rule deleted.

**`defaultInternalSupportParams`.** Single defaults → expected =
`{type(default_value)}`. All current entries are bool or int literals; no
special cases. `KernArgsVersion` is int (verified) — confirm no in-tree
custom kernel sets it as a string by running `fix_yaml_types.py` over the
custom-kernel directory in Step 0.

**`defaultAnalysisParameters` (LibraryLogic block).** Single defaults →
expected = `{type(default_value)}`. Plus the `SolutionImportanceMin`
range check `[0.0, 1.0]` inline at the same call site. **No
`LibraryType` enum check** — see B2 below.

**Parameter-name collisions across registries (M4).** Some names appear
in multiple registries with different semantics:

- `ISA`: `globalParameters["ISA"] = []` (list of arch strings, populated
  at startup); `validParameters["ISA"]` is set inside
  `assignGlobalParameters` to a list of `IsaVersion` objects. The
  `_skipTypeCheck` set already lists `"ISA"` because the user-supplied
  string is converted downstream.
- `CodeObjectVersion`: in `globalParameters` (`str`, e.g. `"4"`) and
  consumed by Solution-side logic. No collision with `validParameters`.

**Ownership rule (section-scoped).** A key under `GlobalParameters:` in
YAML is validated against `globalParameters` only; a key under
`ProblemType:` is validated against `_defaultProblemType` only; a key
under `BenchmarkCommonParameters` / `ForkParameters` is validated against
`validParameters` only; a key under `LibraryLogic:` is validated against
`defaultAnalysisParameters` only. Same name under different sections is
validated by the section's owner — they are independent registries.
Cross-section consistency is out of v1 scope (see Finding 1.4 in the
review).

A property test (promoted from optional to required, see §7) asserts
that every key in the four registries appears in exactly one
expected-types map or in an explicit skip set, and fails on accidental
collision.

## 4. Error reporting design

**One shared exception class + one keypath helper.** Lives in a small
new module `tensilelite/Tensile/Common/TypeValidationErrors.py`:

```
class ConfigTypeError(Exception):
    pass

def formatMismatch(srcFile, keyPath, value, expectedTypes): ...
def getStrictMode(): ...   # reads TENSILE_STRICT_TYPE_CHECK
```

Not a dispatcher; it owns the exception type and a single formatter so
every extended validator emits identical-shape messages. Distinct from
generic `Exception` so tests can catch it specifically and existing
`printExit` flows aren't intercepted.

**Per-mismatch record.** Each emitted message carries:

- `srcFile` — absolute path of the YAML being loaded (or `"<inline>"`
  for programmatic dicts).
- `keyPath` — dotted/bracketed path. The producing validator owns
  composition because it knows its own scope:
  - `assignGlobalParameters` → `GlobalParameters.<Key>`.
  - `validateProblemTypeParameterTypes` → `ProblemType.<Key>` (or
    `BenchmarkProblems[<i>][0].<Key>` if the caller passes an optional
    `keyPathPrefix`).
  - `checkParametersAreValid` (called from `BenchmarkStructs`) →
    `BenchmarkProblems[<i>][1].ForkParameters.<Key>[<j>]` /
    `BenchmarkCommonParameters.<Key>[<j>]` /
    `ForkParameters.Groups[<g>][<e>].<Key>` — the `BenchmarkStructs`
    caller already knows the section; pass the prefix down.
  - `generateLogic()` → `LibraryLogic.<Key>`.
  - Sibling `InternalSupportParams` validator → `InternalSupportParams.<Key>`.
- `value` — `repr(value)` so `True` vs `'true'` vs `1` are unambiguous.
- `actualType` — `type(value).__name__`.
- `expectedTypes` — sorted `__name__`s joined with " or ".
- `lineNo` — optional. See below.

**Line numbers.** PyYAML's default loader discards position info. Two
options: (a) subclass the loader to attach `mark.line` to constructed
nodes (cost on every parse), or (b) on validation failure re-parse the
file with `yaml.compose()` to recover line numbers for the failing keys
(paid only on the error path). Recommendation: (b). Validators that
encounter a mismatch may pass `srcFile` to a helper that re-parses for
lines; if `srcFile` is empty/missing the line is omitted.

**Fail-fast vs collect.** Each validator collects all mismatches in its
own scope before raising (e.g. `assignGlobalParameters` collects all
bad-typed global keys and raises once; `BenchmarkStructs` aggregates
across its iterated `checkParametersAreValid` calls before raising). The
first validator to fire aborts the run; we do not catch and continue
across blocks.

**Worker-process exception passthrough (B1).** Verified: the worker
`_generate_single_solution` in `BenchmarkProblems.py` wraps the full
`Solution(...)` body in `try: ... except Exception as e: print(...);
return None`. `Solution.__init__` constructs a `ProblemType` inside the
worker (which runs `validateProblemTypeParameterTypes`) and also runs
`validateParameterTypes`. A `ConfigTypeError` raised inside would be
caught and turned into `None`, producing a silent count drop and an
unstructured one-line print — not the structured fail-fast the plan
promises.

Fix (local, smaller change than restructuring): the worker's broad
`except Exception` is replaced with a typed except that re-raises
`ConfigTypeError`:

```python
try:
    ...
except ConfigTypeError:
    raise
except Exception as e:
    print(f"Error processing permutation {perm}: {e}")
    return None
```

This belongs in the same step that flips `validateProblemTypeParameterTypes`
to raise (Step 4). A regression test asserts that a YAML with a bad
ProblemType key produces a `ConfigTypeError` visible to the user, not a
silent `None`.

Note: per the review, the inner `ProblemType` re-construction inside the
worker is fed a state that has already been validated by the outer
`ProblemType` construction in `BenchmarkProcess.__init__`. The inner
revalidation is therefore expected to be a no-op for input YAMLs. The
typed-except fix is still required as a backstop: programmatic callers
or future code paths that construct the worker without an outer
revalidation must surface the error rather than swallow it.

**Exit code.** `ConfigTypeError` propagates out of `Tensile.Tensile()`
naturally; the CLI wrapper already converts unhandled exceptions to a
non-zero exit. Validators do not call `sys.exit`.

## 5. Bool/int trap — explicit treatment

`bool` is a subclass of `int` in Python, so `isinstance(True, int)` is
`True`. Every extended validator MUST use `type(value) is T` /
`type(value) in expectedTypeSet`, not `isinstance`. This is already the
rule applied in `Solution.py` `_getExpectedTypes` /
`validateParameterTypes` and in `Problem.py`
`validateProblemTypeParameterTypes`; the extensions follow the same
convention.

**Numeric strictness rule.** `int` is **not** accepted where `float` is
expected, nor vice versa, for `globalParameters` and
`defaultAnalysisParameters` single-default-typed entries. Rationale: the
collapse is exactly the bug class this work targets, and YAML `0` vs
`0.0` is a writeable distinction. The natural union over allowed values
for `validParameters`-typed parameters (e.g. `LocalWritePerMfma` becomes
`{float, int}`) is preserved — that's a registry-level signal that both
are intentional.

Tests cover both directions:

- `bool` rejected where `int` expected (`BoundsCheck: False`).
- `int` rejected where `bool` expected (`UseBeta: 1`,
  `PrefetchGlobalRead: [1]` where the validParameters list is `[False, True]`).
- `int` rejected where `float` expected (e.g. `SolutionImportanceMin: 0`).
- Legitimate `-1` sentinel in a float list (`LocalWritePerMfma: [-1]`)
  does NOT raise.

## 6. Migration / rollout strategy

**The principled fix and the tree cleanup ship in the same PR.** The
warning-collector has not driven the tree to cleanliness because
warnings are ignored; that is exactly why `fix_yaml_types.py` exists.
Repeating the warn-then-tighten cycle would perpetuate the design
problem. So:

1. The extended validators run in **strict (raise)** mode from the first
   commit.
2. Every input YAML in the tree is fixed in the same PR so the tree is
   green at HEAD.

**Squash-merge.** The PR is merged with squash. The development branch
retains the per-step commits for review, but mainline gets one commit —
so no commit in mainline is ever bisect-broken. State this explicitly in
the PR description.

**Migration mechanics:**

1. **Step 0** below performs the tree-clean preconditions (generalise
   `fix_yaml_types.py`, run it, convert `BenchmarkSplitter` to
   `LibraryIO.read`, sweep custom-kernel directory).
2. Add the shared exception module (`TypeValidationErrors.py`).
3. Extend each validator (one step per validator, see §8). Each
   extension is independently reviewable; each ships its tests in the
   same commit.
4. **Delete obsolete code in the same commit that supersedes it.** Per
   the "delete hacks immediately" rule:
   - Step 4 (typed except in worker) ships with Step 4
     (`validateProblemTypeParameterTypes` → raise).
   - Step 8 introduces a `strictTypeValidation: bool = True` kwarg on
     `Solution.__init__`, threads it through input-YAML callers
     (`BenchmarkProblems._generate_single_solution`,
     `_getCustomKernelSolutionObj`) as `False`, and keeps the default
     `True` for library-logic callers in `LibraryIO.py`. In the same
     commit it deletes `printTypeMismatchSummary()` and its import from
     `BenchmarkProblems.py` because the call becomes provably dead on
     the input-YAML path. The collector machinery
     (`_typeMismatchCollector`, `getTypeMismatchCollector`,
     `mergeTypeMismatchCollector`, `printTypeMismatchSummary`) stays
     intact for the library-logic path (still called from
     `TensileCreateLibrary/Run.py`); it is the last consumer until v2.

**Opt-out.** A single env var `TENSILE_STRICT_TYPE_CHECK` honoured by
every extended validator: `strict` (default), `warn` (downgrade to a
`printWarning` listing the same records and continue), `off` (skip
entirely). Read centrally via `getStrictMode()` in
`TypeValidationErrors.py`.

**Env-var removal trigger.** `TENSILE_STRICT_TYPE_CHECK` is transitional.
Removed in the first release cycle after the PR merges (one minor version
bump), or 90 days after merge, whichever comes first. Final cleanup step
filed as Step 9 in §8 with the removal date set when the merge commit
lands. After removal: `strict` is unconditional; downstream YAML hygiene
is a hard requirement, not a knob.

**CI interaction.** CI runs without the env var → strict mode. Any PR
introducing a bad-typed value fails at the first relevant validator
with a clear error. The new **corpus-clean** CI test (B5) walks every
YAML under the in-tree input-YAML roots and asserts each loads without
raising `ConfigTypeError`; it is part of the unit test env (`tox -e
unit` per `tensilelite/CLAUDE.md`). No additional CI plumbing.

**Out-of-tree YAMLs.** Downstream consumers get a one-time break. The
error message names `utilities/fix_yaml_types.py` as the bulk fixer.
`TENSILE_STRICT_TYPE_CHECK=warn` covers the transition window until the
env var is removed.

**Library-logic downstream consumers.** hipBLASLt's C++ runtime reads
library-logic YAMLs / msgpack (out of v1 scope). Strict validation on
input YAMLs produces *cleaner* library-logic outputs, which can only
help the C++ side — `fix_yaml_types.py` exists because the consumer
already wants strict typing. No behavioural risk for them.

## 7. Test strategy

Tests live next to the validator they cover. No new central test file.

**Extension to `checkParametersAreValid`.** Either extend an existing
test file or create
`tensilelite/Tensile/Tests/unit/test_checkParametersAreValid_types.py`:

- Bool-vs-int trap in both directions (§5).
- `LocalWritePerMfma: [-1]` accepted (sentinel-in-float-list).
- One test per sentinel class enumerated in §3 (`MaxVgprNumber`,
  `LdsBlockSizePerPad*`, `LdsPad*` patterns).
- `MatrixInstruction`, `SpaceFillingAlgo`, `SFCWGM`, `ClusterDim`,
  `WorkGroup`, `ThreadTile`, `MIWaveGroup`, `MIWaveTile` free-form list
  values accepted (skipped via `-1` rule; assert this).
- `DataType*` / `MacDataType*` / `F32XdlMathOp` / `ISA` strings accepted
  (skip set).
- Groups path: bad type inside `ForkParameters.Groups[0][1]` produces a
  message containing `Groups[0][1].<Key>`.
- `TENSILE_STRICT_TYPE_CHECK=warn` downgrades to a `printWarning` and
  does not raise.
- `TENSILE_STRICT_TYPE_CHECK=off` skips entirely.

**Sibling `InternalSupportParams` validator.** New test file or extend:

- Bad type rejected (e.g. `KernArgsVersion: "two"`).
- Unknown key rejected.
- Clean dict passes.

**Extension to `assignGlobalParameters`.** Add to the existing
GlobalParameters test file (locate at impl time; create
`tensilelite/Tensile/Tests/unit/test_assignGlobalParameters_types.py`
if none):

- Bool-vs-int trap on a known int default (`BoundsCheck: False`) and on
  a known bool default (`PinClocks: 1`).
- Unknown global key RAISES (today is a `printWarning`).
- `ignoreKeys` entries pass through cleanly.
- Override-table entries: `RocProfCounter: 42` raises (int where
  `{None, str}` expected); `RocProfCounter: None` passes;
  `RocProfCounter: "WAVES"` passes.
- Override-table coverage assertion: every `None`-defaulted key in
  `globalParameters` has an entry in `globalParameterTypeOverrides` (the
  validator's init-time check is exercised by a test that mutates the
  defaults dict to add an unannotated `None` and asserts a startup
  error).

**Extension to `validateProblemTypeParameterTypes`.** Tests live next
to existing ProblemType validator tests at
`tensilelite/Tensile/Tests/unit/test_validateParameterTypes.py` (verify
filename and extend):

- `UseBeta: 1` raises (int where bool expected).
- `UseBeta: True` passes.
- Per-file aggregation: multiple bad keys in one ProblemType yield one
  raised exception listing all.
- `DataType*` skip-set values are not touched.

**Worker-process passthrough regression test (B1).** Build a minimal
forked-solutions input with a bad-typed key in
`BenchmarkCommonParameters` or under `ProblemType`. Drive
`_generateForkedSolutions` (or the higher-level `Tensile.Tensile()`)
and assert that the user sees `ConfigTypeError` with the expected key
path — not a silent solution-count drop. This test guards the typed
`except ConfigTypeError: raise` in the worker.

**Extension to `generateLogic()` merge.** Tests live next to existing
LibraryLogic tests:

- Unknown analysis key raises (today silently dropped).
- `SolutionImportanceMin: 1.5` raises.
- `SolutionImportanceMin: -0.1` raises.
- `SolutionImportanceMin: "0.5"` raises (string where float expected).
- `SolutionImportanceMin: 0` raises (int where float expected, per §5
  numeric strictness rule).
- `LibraryType: "Equality"` passes (distance-metric label, B2).
- `LibraryType: 42` raises (int where str expected).
- Clean config passes.

**Corpus-clean CI test (B5).** New parametrized test in
`tensilelite/Tensile/Tests/unit/test_input_yaml_corpus_clean.py`:

- Walks every YAML under `tensilelite/Tensile/Tests/common/` and
  `tensilelite/Tensile/Configs/` (and any other in-tree
  input-YAML root identified at impl time).
- For each: load via the full `Tensile.Tensile()` entry-point's
  parse-and-validate prefix (do not run kernel generation), assert no
  `ConfigTypeError`.
- Runs in `tox -e unit`.
- Enforces tree-cleanliness invariant going forward: any new test YAML
  with a typo'd type fails this gate.

**Integration tests.** One end-to-end test in
`tensilelite/Tensile/Tests/unit/test_input_yaml_validation_integration.py`:

- Load a known-good real test config from `tensilelite/Tensile/Tests/common/`
  via the full `Tensile.Tensile()` entry path on a `tmp_path` copy and
  assert no exception.
- Mutate one well-defined key to a bad type, assert the run aborts with
  `ConfigTypeError` and the expected key path in the message.

Plus a **validator-ordering** integration test (review Finding 5.1):
construct a YAML with a bad `GlobalParameter` AND a bad `ProblemType`
key. Assert `assignGlobalParameters` fires first (closer to load) and
the error message is the `GlobalParameters` one. Guards against future
refactors that reorder validators.

Plus a **bool/int integration trap** (review Finding 5.3): run a config
with bool-where-int and assert the failure identifies the right
validator, key, and key path.

Per project policy, all pytest invocations pass
`--ignore=...test_MatrixInstructionConversion.py`. The new tests are
fast and add no new slowness.

**Property test (required).** A property test asserting that every key
in `globalParameters`, `_defaultProblemType`, `validParameters`, and
`defaultAnalysisParameters` either appears in the derived expected-types
map for its validator, in an explicit skip set, or in
`globalParameterTypeOverrides` — and that no key appears in more than
one expected-types map. Fails loudly if a new param is added without
considering type validation or accidentally collides with another
registry. Promoted from optional to required per review Finding 1.5.

## 8. Phased implementation steps

Each step is one atomic commit on the development branch; the PR merges
as squash. Each step ships its tests in the same commit as the code.

**Step 0 — Tree-clean preconditions.** Files:

- `utilities/fix_yaml_types.py`: extend file discovery to cover input
  YAML roots; add a `--mode {logic,input,both}` flag defaulting to
  `both`.
- Run it across `tensilelite/Tensile/Tests/common/`,
  `tensilelite/Tensile/Configs/`, and the custom-kernel directory.
  Commit the YAML churn together with the fixer change.
- `tensilelite/Tensile/BenchmarkSplitter.py`: switch `yaml.safe_load`
  (line ~42) to `LibraryIO.read` so the loader respects
  `StrictTypeLoader`. Per "delete hacks immediately": this conversion
  ships in Step 0, not deferred — the new strict gate is invalid if any
  reader bypasses the strict loader.
- No production validator code touched yet. Prerequisite for all
  subsequent steps that flip raise behaviour.

**Step 1 — Add the shared exception + keypath helper module.** New
file `tensilelite/Tensile/Common/TypeValidationErrors.py` exporting
`ConfigTypeError`, `formatMismatch(...)`, `getStrictMode()`. Minimal
unit tests. No callers yet.

**Step 2 — Move `_getExpectedTypes` and `_skipTypeCheck` into
`ValidParameters.py` (M1).** Files:
`tensilelite/Tensile/Common/ValidParameters.py` (add the functions and
the pre-computed `_expectedParamTypes`),
`tensilelite/Tensile/SolutionStructs/Solution.py` (import them from
the new location, remove local definitions). Pure refactor — no
behaviour change. Avoids the
`ValidParameters → Solution` reverse import that Step 3 would otherwise
need.

**Step 3 — Extend `checkParametersAreValid` with type check.** Files:
`tensilelite/Tensile/Common/ValidParameters.py` (extend
`checkParametersAreValid`); `tensilelite/Tensile/BenchmarkStructs.py`
(pass the section/index/groups prefix down so the validator can build
the keypath). Use the now-local `_expectedParamTypes` and
`_skipTypeCheck`. Honour `TENSILE_STRICT_TYPE_CHECK`. Tests per §7.

**Step 4 — Promote `validateProblemTypeParameterTypes` to raise + fix
worker swallow (B1).** Files:
`tensilelite/Tensile/SolutionStructs/Problem.py` — change the validator
to accumulate per-instance mismatches and raise `ConfigTypeError` at
the end of `ProblemType.__init__` if any. Keep the function callable
in collector mode for the library-logic path (out-of-scope).
`tensilelite/Tensile/BenchmarkProblems.py` — replace the broad
`except Exception` in `_generate_single_solution` with a typed except
that re-raises `ConfigTypeError`. Tests per §7 including the
worker-process regression test.

**Step 5 — Extend `assignGlobalParameters` + raise on unknown keys +
override table (M2) + fix `TensileClientConfig` arity bug (M5/B1
follow-up).** Files: `tensilelite/Tensile/Common/GlobalParameters.py`
(add per-key type check in the `for key in globalParameters` loop,
convert the existing `printWarning` for unknown keys into a raise,
add `globalParameterTypeOverrides` dict, add the init-time assertion
that every `None`-defaulted key has an override entry, skip
`ignoreKeys` and `MinimumRequiredVersion`).
`tensilelite/Tensile/TensileClientConfig.py` — fix
`ProblemType(problemDict)` to pass `printIndexAssignmentInfo`
(threaded from args or set to a sensible default; verify call-site
intent at impl). Per "delete hacks immediately" this is fixed in the
same step that promotes `validateProblemTypeParameterTypes` to raise's
caller-side coverage; otherwise the new validator would crash this
tool on its first invocation with a TypeError on the missing arg
rather than a clean `ConfigTypeError`. Tests per §7.

**Step 6 — Add LibraryLogic validation at the merge site.** Files:
`tensilelite/Tensile/LibraryLogic.py`. Add a pre-check pass over
`config` before the `for parameter in defaultAnalysisParameters:` loop
in `generateLogic()`: unknown-key check, type check (`type(default)`),
`SolutionImportanceMin` range `[0.0, 1.0]`. **No `LibraryType` enum
check** — verified open-ended (B2). Tests per §7.

**Step 7 — Sibling `InternalSupportParams` validator (B3).** Files:
`tensilelite/Tensile/Common/ValidParameters.py` (delete the dead
`elif name == "InternalSupportParams": return` early-return as dead
code per "delete hacks immediately"; add a new sibling validator
function `validateInternalSupportParams(d, srcFile="")` that iterates
the dict against `defaultInternalSupportParams`, checking key
membership and `type(default)`).
`tensilelite/Tensile/BenchmarkStructs.py` (call the sibling explicitly
from `getConfigParameters` right after `self.internalSupportParams` is
populated). Sibling chosen over fold because `checkParametersAreValid`
has a `(name, list)` contract and `InternalSupportParams` is a dict
— folding would break the function's signature. Tests per §7.

**Step 8 — Thread `strictTypeValidation` kwarg + delete now-dead
input-path `validateParameterTypes` machinery (B4).** Files:
`tensilelite/Tensile/SolutionStructs/Solution.py` (add
`strictTypeValidation: bool = True` kwarg on `Solution.__init__`;
inside the constructor, only call `validateParameterTypes(self._state,
srcFile=srcName)` when `strictTypeValidation` is True);
`tensilelite/Tensile/BenchmarkProblems.py` (pass
`strictTypeValidation=False` from `_generate_single_solution` and
`_getCustomKernelSolutionObj` — input-YAML path is already covered
upstream by Steps 3 and 4; same commit deletes the
`printTypeMismatchSummary()` call site and its import from this file,
because the collector is no longer populated on the input-YAML path).
Library-logic callers in `tensilelite/Tensile/LibraryIO.py` use the
default `True`; the
`printTypeMismatchSummary` call in `TensileCreateLibrary/Run.py` stays
as the only consumer on the library-logic path. Tests confirm an
input-YAML run never invokes `validateParameterTypes` (assertion in
the worker-process test).

**Step 9 — Remove `TENSILE_STRICT_TYPE_CHECK` env var.** Scheduled
*after* merge: file as a follow-up issue with concrete trigger (first
minor version bump after merge, or 90 days, whichever comes first).
At trigger time: delete `getStrictMode()` (or hard-code `strict`),
delete `warn`/`off` branches in each extended validator, delete the
related tests. The plan calls this out so the loophole has an explicit
expiration rather than living forever.

Steps are listed in dependency order. Step 0 must land before any of
Steps 3-7 flips raise behaviour. Step 2 (registry move) must land
before Step 3 (which imports the moved symbols). Step 4 worker fix
must land in the same commit as Step 4 validator promotion. Step 8
depends on Steps 3 and 4 being in place.

## 9. Decisions (formerly open questions)

The prior draft left eight questions open. The review resolved most.
Documented decisions:

1. **`Solution.__init__` `strictTypeValidation` kwarg.** Decided:
   thread the kwarg (Step 8). Input-YAML callers pass `False`,
   library-logic callers default to `True`. Resolves prior §9.1; the
   "avoid threading a strict flag" rationale was a tactical-fix
   instance.

2. **Unknown global key → raise.** Decided: raise (Step 5). Matches
   "no tactical fixes." The env var covers transitional downstream
   YAMLs with extra keys.

3. **`LibraryType` is NOT an enum.** Verified: `LibraryIO.py`
   `rawLibraryLogicForMatchingHeader` (around the read of
   `data[11]`) shows that any value not in `{"FreeSize", "Prediction"}`
   becomes the `distance` label of a `Matching` library. The default
   `"GridBased"` is itself just an arbitrary distance label. **Drop the
   enum check** (Step 6) — keep only type-is-`str`. Resolves prior §9.3.

4. **`InternalSupportParams` validation site.** Decided: sibling
   validator (Step 7). `checkParametersAreValid`'s `(name, list)`
   contract would break under a dict-mode fold. Resolves prior §9.4
   in favour of option (b) — the prior draft's "fold" recommendation
   was wrong.

5. **Tuple-typed `globalParameters` defaults.** Verified absent. No
   tuple defaults exist in `GlobalParameters.py` between lines 40 and
   340; all `( ... )` are line-continuation parens with no comma.
   Question deleted.

6. **Env var only, no CLI flag.** Confirmed (§6).

7. **One PR with squash-merge.** Confirmed (§6). Within-PR commits
   are reviewable; squash leaves no bisect-broken intermediate state
   on mainline.

8. **Cached YAMLs.** Confirmed: the cache-read path (`_readCache`)
   bypasses the in-scope validators. Verified at impl by tracing —
   if not, file as a blocker on Step 3 (per "no deferred discoveries").

## 10. Blockers filed during this revision

Per "no deferred discoveries", new work surfaced during this revision
is filed as blockers on the step that depends on it:

- **B1 (Step 4).** Worker-process `try/except Exception` in
  `_generate_single_solution` swallows `ConfigTypeError`. Step 4
  ships the typed-except fix in the same commit as the
  ProblemType-validator promotion. Without this, Step 4 produces
  silent solution-count drops, not fail-fast errors.
- **B2 (Step 6).** `LibraryType` is not a fixed enum — drop the enum
  check from the prior plan; keep only the type check.
- **B3 (Step 7).** `InternalSupportParams` is dict-typed; sibling
  validator chosen over fold-into-`checkParametersAreValid`.
- **B4 (Step 8).** `strictTypeValidation` kwarg threaded through
  `Solution.__init__` to delete the now-dead input-path
  `validateParameterTypes` invocation; the warn-only-collector
  recommendation in the prior draft was a tactical fix.
- **B5 (§7, gated by all of Steps 3-7).** Corpus-clean CI test added
  to enforce tree-cleanliness invariant going forward.
- **M5a (Step 5).** `TensileClientConfig.py` calls
  `ProblemType(problemDict)` with a missing positional arg
  (`printIndexAssignmentInfo`). Verified bug. Fixed in the same step
  that promotes the new global-parameter validator, so the tool
  surfaces clean errors rather than `TypeError`.
- **M5b — verified false.** The reviewer claimed `LibraryLogic.py`
  references dead keys `SmoothOutliers` and `BranchPenalty` that
  would `KeyError`. Verified: both references are inside
  triple-quoted comment blocks (`LibraryLogic.py` around line 173
  and lines 843-866), i.e. commented-out code. No live `KeyError`
  risk. Filed nothing.

## 11. Reviewer claims verified false

- **`SmoothOutliers` / `BranchPenalty` live KeyError risk** —
  false. Both inside `"""..."""` comment blocks.
- **`[[]]` placeholder in `validParameters`** — claim in prior draft
  was unverified; spot check confirms no `[[]]` entries exist (only
  `-1`). Rule removed from §3.
- **Tuple-defaulted `globalParameters`** — false. No tuples exist;
  all `( ... )` are line-continuation parens. §9.5 of prior draft
  deleted.

All other reviewer claims (worker swallow, `LibraryType` open-ended,
`InternalSupportParams` dict-typed, `TensileClientConfig` arity bug,
`BenchmarkSplitter` bypassing `StrictTypeLoader`, `None`-default
coverage gap, the cross-section ownership question) were verified true
and incorporated above.
