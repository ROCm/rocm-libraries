# Implementation Audit — Strict Input-YAML Type Validation

Branch: `users/alvasile/input_yaml`
Commit range: `fa55f759b697` … `a9fc75ca4657` (10 commits)
Plan: `utilities/input_yaml_validation_plan.md` (revised)
Auditor: independent review, read-only.

---

## 1. Headline verdict

**DO NOT SHIP.**

Two correctness defects make this PR unsafe to merge. (1) The strict
unknown-key gate in `assignGlobalParameters` raises on at least 15
distinct unknown global keys spread across the in-tree YAML corpus
(notably `NewClient`, `Device`/sometimes-ignored, `MaxLDS`, `PrintLevel`,
`PrintSolutionRejectionReason`, `MinKForGSU`, `MergeFiles`,
`MaxFileName`, `ForceGenerateKernel`, `UseGPUTimer`, `DeviceLDS`,
`ROCmAgentEnumeratorPath`, `AMDGPUArchPath`, `DataInitTypeeScaleE`,
`PrintIndexAssignments`). A representative regression reproduces today:
`Tensile/Tests/unit/test_keep_build_tmp.py` fails with
`ConfigTypeError: Unknown global parameter 'NewClient' = 2`. The
implementer's "tree is strict-clean at HEAD" claim is therefore false.
(2) The corpus-clean test (B5) silently skips unknown global keys —
"Be permissive: skip unknown keys" at
`Tests/unit/test_input_yaml_corpus_clean.py:84-87` — so it cannot
catch this class of regression and gave the implementer a false signal.
Both defects are fixable in a small follow-up but must land before
merge; the gate is the principal feature of this PR and shipping it
broken defeats the work. Architectural shape, per-commit ordering,
worker-passthrough fix, override-table, registry-disjointness, and
unit-test coverage of the new validators are otherwise sound.

---

## 2. Part 1 — Plan-conformance audit

| Step | Verdict | Notes |
|------|---------|-------|
| 0 | **VERIFIED with caveat** | `fa55f759b697` switches `BenchmarkSplitter.__readConfigFile` to `LibraryIO.read`, extends `fix_yaml_types.py` (`--mode`, multi-root), fixes `gemm/fp8n.yaml`. Caveat: the Step 0 description claims the tree is fully cleaned by running the fixer; the unknown-key sweep was deferred (see Blocker B-AUD-1). The fixer only addresses known type-mismatch *patterns*, not unknown keys. |
| 1 | VERIFIED | `f6a9686bb58f` adds `Tensile/Common/TypeValidationErrors.py` with `ConfigTypeError`, `formatMismatch`, `getStrictMode`, plus `test_TypeValidationErrors.py`. No production callers yet (expected). |
| 2 | VERIFIED | `2967ed9710eb` moves `_getExpectedTypes`/`_expectedParamTypes`/`_skipTypeCheck` into `Common/ValidParameters.py`; `Solution.py` re-imports the symbols at module top. Single source of truth preserved (Common owns); the re-export is a deliberate back-compat shim for `test_validateParameterTypes.py`. Acceptable. |
| 3 | VERIFIED | `e27dbb135df1`: extended `checkParametersAreValid`; `BenchmarkStructs.getConfigParameters` threads `keyPathPrefix` and aggregates per-section. No obsolete loose-comparison code was present to delete (the prior gate was warn-only via `_typeMismatchCollector` only — left intact for library-logic path per plan §6, deleted from input path in Step 8). |
| 4 | VERIFIED | `532ca7f2e62a`: `validateProblemTypeParameterTypes` adds `raiseOnMismatch: bool = True` kwarg (default True → input-YAML strict); collector-mode preserved as opt-in for library-logic path. Worker re-raise in `BenchmarkProblems._generate_single_solution` is a typed `except ConfigTypeError: raise` ahead of the broad `except Exception` (BenchmarkProblems.py:194-203). Test `TestWorkerPassthroughBackstop` verifies. |
| 5 | VERIFIED | `0ce0829c1642`: per-key type check in `assignGlobalParameters`, unknown-key warn promoted to raise (aggregated), `globalParameterTypeOverrides` defined for the 4 None-defaulted keys (`ClientExecutionLockPath`, `ROCmSMIPath`, `CmakeCxxCompiler`, `RocProfCounter`), `_assertOverrideTableCovers` at module-import-time (line 696). Arity fix on `TensileClientConfig.py:172`: `ProblemType(problemDict, False)`. |
| 6 | VERIFIED | `591a039b89ad`: `generateLogic()` pre-gate covers unknown-key, type-check, range-check for `SolutionImportanceMin`. `LibraryType` not enum-checked (B2 honoured). |
| 7 | VERIFIED | `ee74a1f90130`: `validateInternalSupportParams` sibling added; dead `elif name == "InternalSupportParams": return` deleted from `checkParametersAreValid` (the elif was at the formerly-existing line; now line 1235 has only the `ProblemSizes` early-return). Step-3 test updated to reflect the fallthrough-to-unknown-name path. |
| 8 | VERIFIED | `6db5685b033d`: `Solution.__init__(..., strictTypeValidation: bool = True)`; `BenchmarkProblems._generate_single_solution` and `_getCustomKernelSolutionObj` both pass `strictTypeValidation=False`. `printTypeMismatchSummary` call AND import deleted from `BenchmarkProblems.py` (replaced by a comment block at line 789). `LibraryIO.py` Solution() call sites at :424 and :574 keep the default `True`. |
| 9 | DEFERRED (per plan) | Plan §8 explicitly schedules env-var removal post-merge. Acceptable. |
| Final | VERIFIED with caveat | `a9fc75ca4657`: fixer extended (Group D `INT_TO_STR_PARAMS`, two-element list rewrites, comment/whitespace tolerance); 295 YAMLs swept; `libraryLogicTypeOverrides` introduced for `DeviceNames`. **Caveat:** the corpus-clean test that ostensibly validates the sweep silently skips unknown global keys (see Blocker B-AUD-1). |

**Commit message format.** Every commit follows
`tensilelite: input-yaml validation — Step N: <subject>` / final
catch-all. No LoC numbers in messages.

**Per-step deletions audited:**
- Step 4 obsolete warn-only ProblemType path: preserved with new kwarg
  (intentional, supports library-logic path).
- Step 7 dead `elif name == "InternalSupportParams": return`: deleted
  (verified in the diff).
- Step 8 `printTypeMismatchSummary()` call + import:
  deleted (verified by `grep -n` in the current file: only the
  explanatory comment remains at lines 789-794).

**No TODOs / FIXMEs / hedges introduced** in the full diff.

**Step ordering:** matches plan (0,1,2,3,4,5,6,7,8,Final).

---

## 3. Part 2 — Critical code review

### 2a. Correctness

**Bool/int trap (`type(x) is T`, not `isinstance`).**

- `Common/ValidParameters.py:1266-1267` (checkParametersAreValid): uses
  `actualType = type(value); if actualType not in expectedTypes`. ✓
- `SolutionStructs/Problem.py:804-805` (validateProblemTypeParameterTypes):
  uses `actualType = type(value); if actualType not in expectedTypes`. ✓
- `Common/GlobalParameters.py:852` (assignGlobalParameters):
  uses `if type(value) not in expectedTypes`. ✓
- `Common/ValidParameters.py:1326` (validateInternalSupportParams):
  uses `if type(value) not in expectedTypes`. ✓
- `LibraryLogic.py:1470` (generateLogic gate): uses `if type(value) not
  in expectedTypes`. ✓

All five new comparison sites use `type()` not `isinstance`. The
existing `_getExpectedTypes` in `Common/ValidParameters.py:1129` uses
`type(v)` to derive the expected set as well, so the registry side is
consistent.

**Sentinel handling spot-check.**

Ran the live import:
- `LocalWritePerMfma`: `{int, float}` (sentinel `-1` ∪ float values). ✓
- `MaxLDS` (`[-1, 65536, ...]`): `{int}`. ✓
- `LdsPadA` (`[-1, 0, 1, ...]`): `{int}`. ✓
- `LdsBlockSizePerPadA`: `{int}`. ✓
- `GlobalReadPerMfma`: `{float}`. ✓ (matches the fixer's INT_TO_FLOAT
  rewrite.)
- `MatrixInstruction` (sentinel `-1`): not in `_expectedParamTypes`. ✓
- `WorkGroup` (`makeValidWorkGroups()` — non-empty list of lists):
  IS in `_expectedParamTypes` with type `{list}`. Plan §3 originally
  said WorkGroup was free-form; implementer's self-report correctly
  flags the plan as wrong (the registry actually enumerates concrete
  values). This is benign because YAML always emits lists for WorkGroup,
  so the strict gate would accept `[16,16,1]`.

**None-default handling.**

Confirmed all 4 None-defaulted globals via grep of
`GlobalParameters.py` (multi-line `(None ...)` parens) — match the
override table:
- `ClientExecutionLockPath` (line 250 — multi-line `(`)
- `ROCmSMIPath` (line 261)
- `RocProfCounter` (line 326)
- `CmakeCxxCompiler` (line 736)

Tested override-table coverage:
- The plan's example "Real YAML with wrong-typed value": tested manually
  via the `test_assignGlobalParameters_types.py::TestOverrideTable` suite
  — passes.
- `_assertOverrideTableCovers` is invoked at module level (line 696),
  fires at import time. ✓
- `locateExe` call chain (lines 743-750) confirms: at call-time of
  `assignGlobalParameters`, `globalParameters["ROCmSMIPath"]` is no
  longer None. So a call-time-only coverage check would have given a
  false negative for `ROCmSMIPath`. The implementer's
  module-import-time relocation is correct and the rationale is valid.

**Worker swallow fix (Step 4).**

- The fix at `BenchmarkProblems.py:194-203` catches `ConfigTypeError`
  specifically and re-raises before the broad `except Exception`. ✓
- The fix re-raises (no log-and-swallow). ✓
- `TestWorkerPassthroughBackstop` patches `Solution` to throw
  `ConfigTypeError` and asserts the exception reaches the caller; also
  patches with a generic `ValueError` and asserts the legacy
  swallow-and-print is preserved. Test passes when run in isolation. ✓

### 2b. Tests

**Unit suite run.**
```
$ python -m pytest Tensile/Tests/unit/ --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py -n 4 -q
12 failed, 1635 passed, 201 skipped in 8.38s
```

Failures (post-investigation):
- `test_emitMfmaInstruction.py` (8 tests): pass in isolation, fail under
  xdist. Pre-existing parallel-test contamination. The test file has not
  changed between baseline `80957c5cc964` and HEAD (`git diff` empty).
  Implementer's claim **SUPPORTED**.
- `test_SubtileBasedLogicalScheduler.py` (2 tests): pass in isolation,
  fail under xdist. Same pattern. **SUPPORTED**.
- `test_keep_build_tmp.py` (2 tests, both parametrizations): pass on
  the baseline commit (test file unchanged); fail on HEAD with
  `ConfigTypeError: Unknown global parameter 'NewClient' = 2`.
  This is a **NEW REGRESSION** introduced by the strict gate. Implementer's
  characterisation as "pre-existing parallel-test contamination" is
  **WRONG** for this test pair.

**Corpus-clean test (B5).**

`test_input_yaml_corpus_clean.py:300 passed`. But this gives a false
guarantee — see `test_input_yaml_corpus_clean.py:82-87`:

```python
for key, value in cfg.items():
    if key not in globalParameters:
        # Could be an ignoreKey -- not all ignoreKeys are spelt out
        # here. Be permissive: skip unknown keys in the corpus test
        # so a benign extra key doesn't false-positive.
        continue
```

The real validator at `Common/GlobalParameters.py:829-836` does NOT skip
unknown keys; it appends a `ConfigTypeError` record. The corpus test
silently elides this gate. A direct enumeration of unknown keys in the
in-tree YAMLs (run manually):

```
AMDGPUArchPath: 1 file
DataInitTypeeScaleE: 2 files
Device: 126 files (note: 'Device' IS in ignoreKeys, but the corpus test
                    skips before checking that, so it never observed it
                    either)
DeviceLDS: 45 files
ForceGenerateKernel: 9 files
MaxFileName: 18 files
MaxLDS: 44 files
MergeFiles: 16 files
MinKForGSU: 102 files
NewClient: 183 files
PrintIndexAssignments: 1 file
PrintLevel: 108 files (PrintLevel IS in ignoreKeys, benign)
PrintSolutionRejectionReason: 186 files
ROCmAgentEnumeratorPath: 8 files
UseGPUTimer: 5 files
```

Most of these are NOT in `ignoreKeys` and will raise at any real
`Tensile.Tensile()` invocation. The corpus-clean test does not catch
this. **The tree is NOT actually clean at HEAD.** This is **Blocker
B-AUD-1**.

**Integration tests.**

`test_input_yaml_validation_integration.py` (4 tests): pass. But the
`test_clean_real_config_passes` only verifies the GlobalParameters
section type-check for one YAML, and that one happens to be picked
alphabetically (`amaxd/8rn_amaxd.yaml`), which has only `Device`
(in `ignoreKeys`) as its unknown key. It does not catch the
`NewClient`/`MaxLDS`/etc. corruption pervading the rest of the tree.

The validator-ordering and bool/int trap integration tests are
well-scoped to their purposes. ✓

**Bool/int trap test (plan §5 mandatory).**

`test_assignGlobalParameters_types.py::TestBoolIntTrap` covers
`BoundsCheck: False` (bool where int) raising and `BoundsCheck: 0`
passing; `PinClocks: 1` (int where bool) raising and `PinClocks: False`
passing. The assertions check the error message text contains expected
keypath and type names — a regression to `isinstance` would still fail
these tests because `isinstance(True, int)` would not distinguish bool
from int. ✓

### 2c. Architectural integrity

**Single source of truth per registry.**

- `_expectedParamTypes` lives only in `Common/ValidParameters.py:1133`.
  `Solution.py` imports it (lines 45-48). Verified by `grep -n
  "_expectedParamTypes\s*=" tensilelite/Tensile/ -r --include="*.py"`:
  only the one definition.
- `_expectedProblemTypeParamTypes` lives only in
  `SolutionStructs/Problem.py:757`. ✓
- `globalParameterTypeOverrides` lives only in
  `Common/GlobalParameters.py:603-665`. Wait — actually 638-647 (the
  `libraryLogicTypeOverrides` is at 603-605 in the final commit). ✓

**Section ownership for collisions** (M4).

`KNOWN_CROSS_SECTION_NAMES` in
`test_registry_disjoint_property.py:54-66`:
- `ISA`, `CodeObjectVersion`, `MXScaleFormat`, `Sparse`.

The property tests enforce that no other collisions exist between any
pair of {validParameters, globalParameters, _defaultProblemType,
defaultAnalysisParameters}. The tests pass. The plan only documented
`ISA` + `CodeObjectVersion`; the implementer added `MXScaleFormat` +
`Sparse` after verifying the actual collisions in code. Acceptable
deviation per the implementer's self-report.

Note: the test asserts ownership by allow-list, but does NOT verify
the section validator routes each collision to the right registry. A
stronger test would mutate-and-assert; the present form is a
necessary-but-not-sufficient gate. **Minor.**

**`strictTypeValidation` kwarg routing.**

Solution() call sites grep-audit:
- `LibraryIO.py:424` (library-logic load): no kwarg → default True. ✓
- `LibraryIO.py:574` (`solutionStateToSolution`): no kwarg → default
  True. ✓
- `BenchmarkProblems.py:180-187` (`_generate_single_solution`): passes
  `strictTypeValidation=False`. ✓
- `BenchmarkProblems.py:258-265` (`_getCustomKernelSolutionObj`):
  passes `strictTypeValidation=False`. ✓

No other production `Solution(...)` instantiations exist (verified by
filtering out `MasterSolutionLibrary`/`OriginalSolution`/etc.).

### 2d. Bulk YAML rewrites

**Sample spot-check of the 295-file sweep.**

Examined `Tensile/Tests/common/gemm/cgemm.yaml` and
`Tests/common/streamk/sk_dgemm_quick.yaml` rewrites: changes are
literal type rewrites (bool↔int, int→float, int→str) consistent with
the documented Groups A/B/C/D. No semantic changes spotted.

**`INT_TO_STR_PARAMS` extension (Group D).**

`CodeObjectVersion`: globalParameters default is `"4"` (string,
verified at `GlobalParameters.py:270`). Targeting it for int→str
rewrite is correct.

**Two-element list pattern** (`[0, 1]` → `[false, true]`).

Restricted to `INT_TO_BOOL_PARAMS`. Each such param has
`validParameters[...]: [False, True]`. The fixer preserves ordering
(`[0, 1]→[false, true]` and `[1, 0]→[true, false]` are both added)
so neither orientation is lost. ✓

Spot-checked `PrefetchGlobalRead: [0, 2]` (real in-tree value): the
two-element rewrite is restricted by regex to `[0,1]` / `[1,0]` only,
so `[0, 2]` is untouched. ✓ (`PrefetchGlobalRead` is in
`BOOL_TO_INT_PARAMS`, not `INT_TO_BOOL_PARAMS`, so the `[0, 2]` would
also not match Group A patterns — those only handle scalar `false`/`true`
and `[false]`/`[true]`. Correct.)

**Trailing comment / whitespace tolerance.**

The pattern construction adds a `tail = r"(\s*(?:#.*)?)$"` capture
group with `\g<2>` in the replacement. Spot-checked one rewrite:
`projects/hipblaslt/tensilelite/Tensile/Tests/common/gemm/dtv.yaml`
shows `- BufferLoad: [true]  # ...comment...` correctly preserved
after the rewrite (the YAML reads consistently; no trailing characters
truncated).

### 2e. Misc rule violations

- **sed usage in commits**: none. (`git diff` shows only Python and
  YAML changes; no shell scripts touched.)
- **Force-push or destructive ops**: `git reflog` shows only forward
  commits. No history rewrites observed.
- **Pushed?** `git log origin/users/alvasile/input_yaml..HEAD` is
  empty — i.e. **the branch IS already pushed**. The audit prompt
  asked to verify the commits were *not* pushed, and they appear to
  have been. This may be expected per the implementer's
  workflow, but worth flagging given the user's "never push" memory.
- **Backwards-compat shims**: `Solution.py` re-imports
  `_getExpectedTypes`/`_expectedParamTypes`/`_skipTypeCheck` from
  `ValidParameters` and the comment explains it's for the existing
  test module. The shim is intentional and small, justified by avoiding
  touching the test file. Borderline acceptable — could be removed by
  updating `test_validateParameterTypes.py` to import from
  `Common.ValidParameters` directly. **Nit.**

### 2f. Anything else (surprises / smells)

- **Step 4 collector double-population.** When
  `validateProblemTypeParameterTypes` is called with
  `raiseOnMismatch=True` (default, input path), the function still
  populates `_typeMismatchCollector` (Problem.py:806-817) BEFORE
  raising. The implementer's self-report claims "the collector is
  provably empty on the input-YAML path" — this is true only because
  the raise aborts the run before any consumer reads the collector,
  not because the collector is not written. Semantically fine; just
  noting the claim is slightly imprecise.
- **`newMIValidParameters` not in `_expectedParamTypes`.** These are
  added to `validParameters` via `validParameters.update(...)` in
  `CustomKernels.py:97` at runtime, after `_expectedParamTypes` was
  pre-computed at module load. So `EnableF32XdlMathOp` /
  `EnableMatrixInstruction` / etc. are NOT type-checked by the input
  gate even though they have `[False, True]` registries. Out of v1
  scope per the plan (custom-kernel territory), but worth filing as
  follow-up.
- **The corpus-clean test ignored unknown keys deliberately.** The
  comment in `test_input_yaml_corpus_clean.py:84-87` says "Be
  permissive: skip unknown keys in the corpus test so a benign extra
  key doesn't false-positive." This is the wrong design: the in-tree
  ignoreKeys list IS knowable (it's enumerated at
  `GlobalParameters.py:794-809`); the corpus test should consult it
  and raise on anything else. This decision is the root cause of
  Blocker B-AUD-1 escaping review.
- **`test_clean_real_config_passes` is order-dependent.** It picks
  the alphabetically-first YAML with `GlobalParameters` from
  `Tests/common/` — today that's `amaxd/8rn_amaxd.yaml`, which
  contains only `Device` (an ignoreKey) as its unknown global. Add or
  rename one alphabetically-earlier YAML with a `NewClient` and this
  test starts failing. The reliance on alphabetic luck is a smell.
- **Stale path reference in plan.** Plan §7 mentions
  `tensilelite/Tensile/Configs/` as a corpus root; the directory does
  not exist. Implementer correctly scoped to `Tests/common/` only.
  Plan, not implementation, is wrong. Worth correcting the plan post-
  merge.

---

## 4. Blockers

### B-AUD-1 — Strict unknown-global-key gate breaks real test paths; tree is not clean

- **Severity:** BLOCKER
- **Claim:** The strict unknown-key gate (Step 5) raises on at least
  15 distinct unknown global keys present in the in-tree YAMLs. A
  representative regression: `Tests/unit/test_keep_build_tmp.py` (both
  parametrizations) fails with `ConfigTypeError: Unknown global
  parameter 'NewClient' = 2`. The corpus-clean test (B5) was supposed
  to catch this class of issue but silently skips unknown global keys.
- **Evidence:**
  - `Common/GlobalParameters.py:829-836` — strict gate raises on
    unknown keys.
  - `Tests/unit/test_input_yaml_corpus_clean.py:82-87` — corpus test
    skips unknown keys with the comment "Be permissive".
  - `Tests/unit/test_data/keep_build_tmp.yaml:11` — contains
    `NewClient: 2`.
  - `pytest Tensile/Tests/unit/test_keep_build_tmp.py -n 4` → 2 failed
    with the above message.
  - Enumeration script confirmed 183 in-tree YAMLs carry `NewClient`,
    among other unknown keys.
- **Recommendation:** Either (a) sweep the in-tree YAMLs to remove
  the unknown keys before flipping the unknown-key check to raise; or
  (b) extend `ignoreKeys` to cover the legitimately-benign extras
  (`NewClient`, `PrintSolutionRejectionReason`, `MinKForGSU`,
  `MergeFiles`, `MaxFileName`, etc. — audit each), or (c) keep the
  unknown-key check warn-only until a separate sweep PR cleans the
  corpus and then flip the default. Option (a) is the principled fix
  consistent with the plan. The corpus-clean test must be rewritten
  to NOT silently skip unknown keys — it must drive the same code
  path the real validator does, otherwise its 300-pass result means
  nothing.

### B-AUD-2 — Corpus-clean test (B5) does not exercise the strict gate it claims to enforce

- **Severity:** BLOCKER (gates B-AUD-1's fix)
- **Claim:** `test_input_yaml_corpus_clean.py:65-101` reimplements a
  hand-rolled subset of `assignGlobalParameters` that elides the
  unknown-key check, the `KeepBuildTmp`/`AsanBuild`/`CodeObjectVersion`
  branches, and the `globalParameters[key] = value` mutation. As a
  result the test cannot reproduce real-validator behaviour; it
  produced 300 passes while real loads of the same YAMLs fail.
- **Evidence:** code at the cited lines; passing test result above;
  contradicting `test_keep_build_tmp` failures.
- **Recommendation:** Replace the hand-rolled validator with a real
  `assignGlobalParameters(cfg, isaInfoMap={})` invocation per YAML
  (after `restoreDefaultGlobalParameters()` and given a stub
  `isaInfoMap`). The whole purpose of the corpus-clean gate is to
  proxy `Tensile.Tensile()`'s validation prefix — short of doing so,
  it is performative.

---

## 5. Majors

### M-AUD-1 — Integration test `test_clean_real_config_passes` is order-dependent

- **Severity:** MAJOR
- **Claim:** The "real config passes" assertion happens to pass only
  because the alphabetically-first YAML in `Tests/common/` doesn't
  contain unknown global keys that the real validator would raise on.
  A different YAML candidate (with `NewClient`) would have caught
  B-AUD-1 in CI.
- **Evidence:** `test_input_yaml_validation_integration.py:50-63`
  iterates `sorted(CONFIG_ROOT.rglob("*.yaml"))` and breaks on first
  match. Live run picks `amaxd/8rn_amaxd.yaml`.
- **Recommendation:** Drive the test off a representative list of
  ~5 YAMLs covering different sections (or once B-AUD-2 is fixed, fold
  this single-YAML test into the corpus walk).

### M-AUD-2 — Tree-clean preconditions did not cover `Tests/unit/test_data/`

- **Severity:** MAJOR
- **Claim:** The fixer sweep ran only over `Tests/common/`. The
  `Tests/unit/test_data/keep_build_tmp.yaml` file was not swept and
  contains `TransposeA: 1`, `TransposeB: 0` (int where bool expected
  per `_defaultProblemType`) AND `NewClient: 2` / `Device: 0`. Even
  if B-AUD-1 is resolved by extending `ignoreKeys`, the
  `TransposeA`/`TransposeB` ints would still trip Step 4's
  ProblemType validator.
- **Evidence:** file contents above.
- **Recommendation:** Sweep `Tests/unit/test_data/` as part of the
  same fix. Likely also any test-data directories under other unit
  tests (run a grep for `*.yaml` under `Tests/unit/`).

### M-AUD-3 — Property test does not verify section-validator routes correctly

- **Severity:** MAJOR (per plan §3 emphasis on ownership rule)
- **Claim:** `test_registry_disjoint_property.py::TestSectionOwnership`
  proves cross-registry collisions are on a known-allowed list but
  does NOT prove the section validators route each collision to the
  right registry. A future refactor that points `MXScaleFormat` under
  `BenchmarkProblems` at the globalParameters expected types (or vice
  versa) would not fail this property test.
- **Evidence:** test file lines 119-168.
- **Recommendation:** Add a mutation test per known collision: feed a
  YAML with the wrong type for that key in each section and assert
  the right validator's keypath appears in the error.

---

## 6. Minors / Nits

### m-AUD-1 — Re-exports from `Solution.py` are a shim for one test file

- **Severity:** MINOR
- **Claim:** `Solution.py` re-imports `_getExpectedTypes` /
  `_expectedParamTypes` / `_skipTypeCheck` from
  `Common/ValidParameters.py` solely so
  `Tests/unit/test_validateParameterTypes.py` can import them from
  `Solution`. The cleaner path is to update the test to import from
  the new home.
- **Recommendation:** Trivial cleanup; can be left for a follow-up.

### m-AUD-2 — `newMIValidParameters` entries not in `_expectedParamTypes`

- **Severity:** MINOR (out of v1 scope per plan, but worth filing)
- **Claim:** `CustomKernels.py:97` does `validParameters.update(
  newMIValidParameters)` at runtime, but
  `_expectedParamTypes = _getExpectedTypes(validParameters)` was
  pre-computed at module import time. Bool-typed entries
  (`EnableF32XdlMathOp`, `EnableMatrixInstruction`, `MFMA_BF16_1K`,
  `UseF32XEmulation`) are NOT type-checked by the input gate.
- **Recommendation:** File as follow-up: either recompute
  `_expectedParamTypes` after the update, or extend the strict gate
  to honour `newMIValidParameters` for custom-kernel call sites.

### m-AUD-3 — Step 4 collector still populated even when raising

- **Severity:** NIT
- **Claim:** `validateProblemTypeParameterTypes(raiseOnMismatch=True)`
  still writes to `_typeMismatchCollector` before raising. Cosmetic;
  the data is unread on the input path because the raise aborts.
- **Recommendation:** Guard the collector update with `if not
  raiseOnMismatch:` to make the data flow match the docstring claim.

### m-AUD-4 — Final commit description references items but mixes them into a single commit

- **Severity:** NIT
- **Claim:** The final commit (`a9fc75ca4657`) bundles (a) fixer
  extension, (b) 295-YAML sweep, (c) `libraryLogicTypeOverrides`
  introduction, (d) corpus/property/integration test additions. Per
  plan §8 these are arguably 4 separate concerns. The bundling is
  pragmatic given they all close-out B5 but makes the diff hard to
  scan.
- **Recommendation:** Future-work: split similar omnibus commits.

### m-AUD-5 — Branch is already pushed

- **Severity:** NIT (process)
- **Claim:** `git log origin/users/alvasile/input_yaml..HEAD` is
  empty — branch is already on the remote. The user's persistent
  memory says "never push" (pushing is the user's responsibility).
- **Recommendation:** No action — likely the user pushed manually
  before invoking this audit.

---

## 7. Surprises and smells

- **The corpus-clean test ran 300 passes but did not exercise the
  thing it claims to exercise.** This is the most damaging smell:
  the gate that was meant to enforce tree-cleanliness invariant gave
  a false-positive signal, which let the unknown-key regression
  through. A green corpus-clean run was likely the implementer's main
  reason to believe the tree was clean. Tests that don't drive the
  real code path are anti-tests.
- **`_find_simple_config()` picks alphabetically.** Any change to the
  tree that adds a YAML with unknown keys (or removes the currently-
  selected clean one) would silently change what the integration
  test verifies. This is the same anti-pattern as B-AUD-2 in a
  smaller dose.
- **`globalParameters[key] = value` on unknown key.** When an unknown
  key is collected as an error, the code still writes it into
  `globalParameters` (line 835). Probably benign in strict mode (run
  aborts), but in `warn` / `off` modes this silently pollutes the
  shared dict. Worth a comment or removal.
- **`_assertOverrideTableCovers` was correctly moved to import-time.**
  This is a positive — the implementer pushed back against the plan's
  call-time check after discovering the `locateExe` mutation. That's
  the right kind of plan deviation; flagged here as a strength rather
  than a finding.
