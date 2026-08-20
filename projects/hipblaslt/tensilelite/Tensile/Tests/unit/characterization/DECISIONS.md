# Decision log — TensileLite characterization

The durable record of *why* the characterization suite looks the way it does:
which modules are deliberately left below the coverage bar, which behaviors are
pinned as latent bugs rather than "fixed" in a characterization test, which
mutants are accepted as equivalent, and the few places this departed from the
add-only rule. A reviewer of any future TensileLite change should be able to
read this file and understand the intent behind the tests next to it.

One entry per non-trivial decision: the choice made, why, and why the
alternatives were rejected. Routine "wrote tests, hit the bar, committed" steps
are **not** logged — only genuine forks.

See `README.md` (this directory) for the per-module protocol and how to run the
suite. Additional background is tracked under AIHPBLAS-3871.

---

## D0 — Scope of "every remaining module"
**Decision:** Characterize the pure / table / IO / config / toolchain-helper
Python surface; **exclude** the codegen/asm/GPU modules (KernelWriter*,
Components/*, Asm*, GenerateSummations, verify_stinky*, ClientWriter) and
**defer** Solution.py slice-3b (derivation config matrix).
**Why:** the excluded set (~38k stmts) emits GPU assembly / drives the full
build; it is not unit-characterizable without a GPU + toolchain pipeline (rated
★ lowest fit in the original MODULE MAP). Snapshotting their structure would be
brittle and low-value.
**Alternatives rejected:** (a) attempt the kernel writers with heavy mocking —
rejected: the mocks would assert our own scaffolding, not real behaviour, and
would break on every codegen change; (b) include them as 0%-and-documented
stubs — rejected: adds empty dirs with no value. These are listed OUT of scope
in the plan, not silently skipped.

## D1 — Per-module coverage measurement (suite-alone) vs full `-m unit` each time
**Decision:** Measure each module with a fast **suite-alone** `--cov` run; run
the full `-m unit` no-regression gate **once per batch** (and capture a fresh
baseline) rather than once per module.
**Why:** the full suite is ~110s; doing it per module across ~44 modules would
add hours with no extra signal (a new add-only test dir cannot reduce another
module's coverage). Per-batch is enough to catch any accidental import-time
regression.
**Alternative rejected:** full run per module — rejected on cost; the add-only
constraint makes per-module regression practically impossible.

## D2 — Trivial-module doc overhead
**Decision:** For small modules that hit ~100% cleanly, write a single compact
`target.md` (before→after + any resistance inline) and skip separate
`resistance.md`/`recommendations.md`; commit one atomic commit per module.
**Why:** keeps the per-module-commit requirement without ceremony that adds no
information for a trivial 100% module.
**Alternative rejected:** the full 5-file deliverable per module (as used for the
large standalone targets) — rejected as disproportionate for 9-60 stmt modules.

## D3 — Testing `Component` find/match without polluting the global registry
**Decision:** Define private isolated `_CharBase`/`_CharNest*` Component
subclasses in the test module to drive `matches`/`findAll`/`find`/`versions`
(single-match, >1-match RuntimeError, nested-abstract recursion) deterministically;
test `LocalRead._getLdsReadMemToken`/`_emitLdsRead` by calling them **unbound**
with stub self/writer/module (no subclass needed).
**Why:** `ComponentMeta` auto-registers every subclass into its base's
`implementations`. Production searches always start at a real subtype
(`Component.<RealSubtype>.find`), never at `Component.findAll`, so private impls
parented at `Component`/`_CharBase` never appear in a real search — the mutation
is additive and inert. The unbound-call trick covers the codegen LocalRead
helpers without registering a concrete LocalRead (which would join the real
LocalRead search set).
**Alternatives rejected:** (a) exercise the real registered components with a
fake writer — rejected: match results are nondeterministic across environments
and the >1-match error can't be forced reliably; (b) register a concrete
LocalRead subclass — rejected: it would pollute the real `LocalRead`
implementations set used by the kernel writers. Also note the
`from .Components import *` at the end of `Component.py` shadows the module-level
`LocalRead` name, so the real class is reached via the `Component.LocalRead`
attribute.

## D4 — `Common/Parallel.py`: accept <95% (fork/process-pool paths)
**Decision:** Characterize the pure helpers + single-threaded + `n_jobs=1`
in-process paths of `Parallel.py` (→ ~81% line) and **document the rest as
resistance**, accepting this module below the 95% bar.
**Why:** the uncovered lines are the real parallel-execution paths —
`ProcessingPool` (multiprocessing.Pool), `ParallelMapReturnAsGenerator`
(ProcessPoolExecutor), the joblib generator-return branch, and the Windows-only
`os.name=="nt"` branch. These fork/spawn OS processes; exercising them in a unit
test is flaky (pickling, fork-in-pytest, CI nondeterminism, slow) and tests the
OS scheduler more than our code. joblib `n_jobs=1` and `multiprocessing.dummy`
(threads) ARE covered because they run in-process.
**Alternatives rejected:** (a) run real `multiprocessing.Pool(2)` /
`ProcessPoolExecutor` with module-level picklable funcs — covers the lines but
is flaky and slow; rejected (same rationale as excluding the codegen surface);
(b) deep-monkeypatch multiprocessing — would assert our mocks, not real
behaviour; rejected. Net: Parallel.py is an honest <95% module, like the
out-of-scope codegen set.

## D5 — recurring submodule-shadowing gotcha
**Observation (not a fork, but recorded):** several `Tensile` packages re-export
a class that shadows a same-named submodule attribute, so
`import Tensile.X.Foo as F` binds the *class*, not the module. Hit for
`SolutionStructs.Solution`, `Component` (LocalRead), and `Common.Parallel`
(joblib `Parallel`). **Standard fix applied everywhere:**
`F = importlib.import_module("Tensile.X.Foo")`.

## D6 — `KernelHelperNaming.py`: cover the naming half, accept <95%
**Decision:** Characterize the pure naming/orchestration surface
(`KernelHelperEnum`, `kernelObjectNameCallables`, the five `*Names` functions)
and **document the `init*` object-construction functions (L110-240) as
out-of-scope codegen** — accepting the module at ~34% line.
**Why:** the `init*` functions construct `KernelWriter{BetaOnly,Conversion,
ActivationEnumHeader,ActivationFunction,Reduction}` instances — the GPU
code-emit classes excluded by D0. They are ~half the module and are not
unit-characterizable without the full kernel-writer machinery.
**Alternatives rejected:** (a) construct the KernelWriter* objects — pulls the
out-of-scope codegen surface into the unit tests; rejected; (b) drop the module
entirely — rejected: the `*Names` functions encode the real kernel-naming
contract and are worth pinning. Net: a partial module like `Parallel`.

## D9 — `Configuration.py`: operators/ProjectConfig covered; AST evaluator deferred
**Decision:** Cover the `Parameter` operator surface, `ReadWriteTransformDict`,
and `ProjectConfig` (sections/dotted-get/defaults/constraints); **document** (a)
the reflected-operator `isinstance(lhs, Parameter)` branches as DEAD and (b) the
`ExpressionEvaluator` AST walker + `CallableParameter`/`createBinaryOp` as a
deferred expression-machinery slice. Accept Configuration <95% combined.
**Why (a):** Python only dispatches `__radd__`/`__rlt__`/... when the LEFT
operand is not a `Parameter`, so inside those methods `lhs` is never a
`Parameter` — that branch is unreachable via real operators (the reflected
*comparison* dunders aren't auto-called at all; Python uses the opposite
operator). They are pinned by explicit calls where meaningful, else dead.
**Why (b):** `ExpressionEvaluator.evaluate` is a ~70-line `ast` node walker;
exhaustive coverage needs an AST-node matrix (BinOp/BoolOp/Compare/Name/Num/…)
— a focused slice, disproportionate to this sweep's per-module budget.
**Alternatives rejected:** force the dead reflected branches via `__radd__`
internals — impossible without a Parameter left operand; build the full AST
matrix now — deferred as Configuration-slice-2. Net: a partial module.

## D10 — `Contractions.py`: predicate/serialization matrix deferred (~86%)
**Decision:** Cover the index value classes + `ProblemType` (indexNames/
operationIdentifier/placeholderStr/predicates) + `SizeMapping`/
`InternalArgsSupport`/`ProblemPredicate.CompoundPredicates` from the one vendored
gfx942-HSS fixture; accept ~86% combined and document the rest.
**Why:** the remaining branches are `ProblemPredicate.FromOriginalKeyPair`/
`CompoundPredicates` + `Solution`/`SizeMapping.FromOriginalState` arms that fire
only for *other* problem configurations (sparse, activation, bias variants,
batched, double/complex dtypes, GSU algorithms, ...). Exercising them needs a
MATRIX of varied logic fixtures; only one is vendored, and hand-authoring
derived-solution states that match the exact serialized format is brittle.
**Alternatives rejected:** vendor many more logic YAMLs — large/out of proportion
and add-only-risky; synthesize derived states by hand — fragile (must match the
full post-derivation key set). Net: a partial like the Solution.py slices; a
"Contractions matrix" slice could finish it given more fixtures.

## D11 — `BenchmarkStructs.py`: BenchmarkProcess builder deferred
**Decision:** Cover the pure helpers (getDefaultsForMissingParameters,
separateParameters, checkCDBufferAndStrides), the fork-permutation cartesian
product (constructForkPermutations/constructLazyForkPermutations), and
BenchmarkStep; document `BenchmarkProcess` (the config->benchmark-steps
integration builder, L83-235) as needing full benchmark configs.
**Why:** `BenchmarkProcess.__init__`/`getConfigParameters`/
`convertParametersToSteps` consume a complete benchmark config (problemType +
problemSizeGroup with BenchmarkCommonParameters/ForkParameters/ProblemSizes/...)
and build ProblemType/ProblemSizes/steps — an integration path better covered by
an end-to-end benchmark-config fixture than hand-built dicts.
**Alternatives rejected:** hand-author a full benchmark config — large/brittle;
out of proportion to the per-module budget. Net: a partial; an integration
fixture would finish it.

## D12 — TensileBenchmarkCluster: pin the `--results-only` constraint crash rather than asserting clean workflow steps

**ADR:** [`adr/0001-pin-results-only-boolop-crash.md`](adr/0001-pin-results-only-boolop-crash.md) — the per-decision record for this pinned bug.

**Defect:** [`AIHPBLAS-4298`](https://amd-hub.atlassian.net/browse/AIHPBLAS-4298).

**Context:** While characterizing `TensileBenchmarkCluster`, the `--results-only`
flag (alone) raises `AssertionError: Constraint evaluation failed: RunDeployStep
or RunBenchmarkStep or RunResultsStep` during construction.

**Root cause (real latent bug):** `ExpressionEvaluator`'s `BoolOp` handler
(`Configuration.py:651-652`) only evaluates `node.values[0]` and
`node.values[1]`, ignoring `values[2:]`. Python parses `a or b or c` as a single
`BoolOp(Or, values=[a,b,c])`, so the constraint collapses to `a or b`. With
`--results-only` only the *third* operand (`RunResultsStep`) is True, so the
constraint evaluates `False or False` → fails. `--deploy-only`, `--run-only`,
and `--run-and-results-only` happen to leave one of the first two operands True,
so they survive.

**Decision:** Pin the actual behavior — a test asserting `--results-only` raises
`AssertionError` — instead of asserting the (intended-but-unreachable) workflow
tuple `(False, False, True)`.

**Why:** Characterization tests must encode what the code *does today*, not what
it should do. Flagging this as a real bug (3+ operand boolean constraints whose
truth depends on the 3rd+ operand are mis-evaluated) is more valuable than a
green test that hides it. ADD-ONLY constraint forbids fixing `Configuration.py`
here.

**Rejected alternatives:**
- *Assert the clean tuple* — would fail (construction raises) and misrepresent
  behavior.
- *Skip the flag entirely* — loses the documentation of a real, user-facing bug.
- *Fix the BoolOp evaluator* — out of scope (ADD-ONLY) and belongs in a separate
  change with its own regression coverage.

**Residual coverage:** 192 stmts, 1 miss (line 120, the bare-`except` swallow
when a task subdir already exists) → 99.51%. Line 120 is a defensive
already-exists guard not worth a dedicated fixture.

## D13 — Activation.py: pin the pure config/type/numeric layer only; asm codegen is out of scope

**Context:** Activation.py is ~1037 statements. After pinning the pure surface,
line coverage is 34.1% (up from 16.8%). The remaining ~660 lines are rocisa
**assembly codegen**: the getXModule emitters (getExp/getGelu/getSigmoid/getTanh/
getDGelu/getSilu/getSwish/...), CombineInstructions/FuseInstruction and their
iter helpers, replaceInst/removeOldInst, ConvertCoeffToHex/HolderToGpr/
createVgprIdxList, and ActivationInline.

**Decision:** Characterize only the pure layer + the asm entry-points that run
cleanly with dummy vgprs. Do NOT attempt to drive the full asm codegen.

**What is pinned (48 tests):** ActivationAvailable, ActivationTypeRegister.
typeAvailable, the full ActivationType API (construct/passActivation/
getAdditionalArgNum/arg-strings/fitSupported/getEnumIndex/getEnumStrList/
state/repr/str/eq/lt/toEnum), actCacheInfo.isSame, getMagic/getMagicStr/
HexToStr/addSpace, and ActivationModule defaults/setters/counters/vgprPrefix +
the working getModule paths (abs/relu/none/clippedrelu/leakyrelu/clamp/drelu)
and getAllGprUsage for a single type.

**Why:** (a) The codegen/asm/GPU layer is explicitly excluded from this
characterization effort's scope. (b) In this environment most emitters raise
immediately — `NameError: 'SelectBit'`/`'VMaxF16'` (half paths for sigmoid/exp/
gelu/tanh/silu/swish/clamp) and `KeyError: 'TransOpWait'` (single paths for
gelu/sigmoid/exp/tanh/silu/swish/dgelu/geluscaling). These are missing-symbol /
ISA-map-dependent codegen paths that cannot be exercised without the full
KernelWriter/ISA context, so they can be neither run nor meaningfully pinned
here. Verifying emitted assembly would require exactly the codegen harness the
scope excludes.

**Rejected alternatives:**
- *Smoke-call every getModule type* — most raise (see above); would just assert
  the raises, which pins environment breakage, not behavior.
- *Build a full rocisa register/ISA context and snapshot emitted asm* — that is
  codegen characterization, out of scope and high-maintenance.

**Result:** 1037 stmts, 683 missed → 34.1% line. Documented ceiling.

## D14 — TensileLibLogicToYaml: pin the formGroups("None") crash on the skipMI / MI-disabled path

**Context:** `formForkParams(sol, skipMI=True)` (or any solution with
`EnableMatrixInstruction` falsy) sets `temp = "None"` (a *string*) and then calls
`forkData.append(formGroups(temp))`. `formGroups` does `temp.items()`, which on a
str raises `AttributeError`. So the entire skipMI / MI-disabled code path is
currently broken, and `TensileLibLogicToYaml(..., skipMI=True)` crashes too.

**Decision:** Pin the crash (assert `AttributeError`) instead of asserting a
"None"-sentinel Group, and drive the orchestrator / fork tests through the
MI-enabled (`skipMI=False` + `EnableMatrixInstruction=True`) path which works.

**Why:** Characterization records present behavior; this is a real, user-facing
bug (the `--skipMI` CLI flag is unusable). ADD-ONLY forbids fixing
`formGroups`/`formForkParams`.

**Rejected alternatives:**
- *Assert a "None" group is produced* — fails; misrepresents behavior.
- *Skip the path* — loses documentation of a real bug on a public CLI flag.

**Residual:** 199 stmts, 4 missed → 98% line. Misses are two yaml-representer
callbacks (representNone/flowSeq, registered but not invoked by these tests) and
two orchestrator RuntimeError guards (empty solutionIndex / missing solution).

## D15 — TensileClientConfig: dead code, REMOVED (final)

**Final verdict (2026-06-03, with the user):** `TensileClientConfig` is dead
code and has been removed. The earlier two readings in this entry were both
wrong on the conclusion; this records the corrected reasoning and the outcome.

**What was removed:**
- `Tensile/TensileClientConfig.py` (the module)
- `Tensile/bin/TensileClientConfig` (the launcher)
- the `"TensileClientConfig"` entry in `cmake/tensilelite_auto_build.cmake`
  `VALID_BINS`

**Why it is dead (evidence):**
- *No in-tree caller.* Following `invoke` / the build / QuickTune / the tuning
  docs, the client-config writing done during tuning goes through
  `ClientWriter.writeClientConfig` / `writeClientConfigIni` (driven by
  `bin/Tensile` → `Tensile.py` → `BenchmarkProblems.py`). Nothing calls the
  standalone `TensileClientConfig.main()` / `bin/TensileClientConfig`. The two
  share the "ClientConfig" name but are different code paths — the source of the
  earlier "it's used in tuning" confusion.
- *Not shipped.* `MANIFEST.in` packages only `bin/Tensile` and
  `bin/TensileCreateLibrary`; `[project.scripts]` registers only `Tensile`.
- *Unimportable anyway.* `TensileClientConfig.py:29` still did
  `from .Common import ... assignGlobalParameters, restoreDefaultGlobalParameters`,
  the pre-refactor flat path. After `Tensile.Common` became a package those
  funcs live in `Common/GlobalParameters.py` and are not re-exported by
  `Common/__init__.py` (which only star-imports Constants/Parallel/Types/
  Utilities), so the import raised `ImportError`. (Sibling entrypoints —
  `Tensile.py`, `GenerateSummations`, `TensileUpdateLibrary`,
  `TensileRetuneLibrary` — were migrated to `.Common.GlobalParameters`; this one
  was missed.) A second latent break existed too: `:176` called
  `assignGlobalParameters(globalParams)` with one arg against the current
  two-arg `(config, isaInfoMap)` signature.

**Validation:** full `-m unit` (`Tensile/Tests/unit`, in `tensilelite-char:repro`)
= **2466 passed / 201 skipped both before and after** the removal — no
regression. This is a real source deletion (departs from the ADD-ONLY rule of
the characterization pass) committed separately as a cleanup, at the user's
explicit direction.

**History of this entry (do not repeat):**
- v1 — "dead module, skip; assert nothing." WRONG reasoning (called it dead only
  because the import failed, without checking callers/packaging).
- v2 — "live tuning entrypoint, broken by refactor, restore it (~2 lines)." Also
  WRONG: there is no caller and it is not shipped, so there was nothing live to
  restore. The `writeClientConfig*` path (which *is* live) was conflated with it.
- v3 (this) — dead code, verified by caller/packaging/import analysis, removed
  with a green suite on both sides.

**Not touched:** `shared/tensile/Tensile/TensileClientConfig.py` — a separate
vendored full-Tensile tree (different `ClientWriter` signatures), out of scope.

---

## Mutation testing — accepted equivalents & `# pragma: no mutate`

A mutant counts as killed only if the suite passes clean, fails on the mutant,
and reverts cleanly. A survivor is accepted (marked
`# pragma: no mutate`, or recorded here as equivalent) only with a one-line
justification. The mutation config lives in `[tool.mutmut]` in
`pyproject.toml`; see the **Mutation testing** section of `README.md` for how to
run it.

**M0 — pilot slice (report-only).** The first slice mutates five files only:
`Common/Utilities.py`, and the four `TensileLogic/Valid{ChipId,MatrixInstruction,
WorkGroup,WorkGroupMappingXCC}.py`. It is intentionally narrow so the workflow
(triage → kill → re-certify) is proven before widening to the critical modules.
Survivors on covered lines are killed with focused `test_mut_*_char.py`; only the
genuinely-unkillable ones below are accepted.

**M1 — accepted `# pragma: no mutate` (display-only string mutations).** Three
lines in `Common/Utilities.py` carry the pragma because the mutant only alters a
user-facing string with no observable control-flow or return-value effect, so no
characterization assertion can distinguish mutant from original:
- `:219` — `sys.stdout.write("\b" + self.chars[...])`, the progress-spinner
  animation frame (cosmetic terminal output).
- `:362` — `print("ERROR: Can't have a negative register value")`, a diagnostic
  message string.
- `:367` — `print("ERROR: Divide by 0")`, a diagnostic message string.

**M2 — accepted `# pragma: no mutate` (expanded mutation run).** These
equivalent source forms are fenced so mutmut does not keep reporting them:
- `Tensile.Common.ValidParameters.checkSpaceFillAlgoIsValid` — the
  `range(0, maxOrderID + 1)` membership check carries `# pragma: no mutate`
  because `range(0, n)` and `range(n)` produce the same values; the explicit
  lower bound documents the valid OrderID interval.
- `Tensile.Common.ValidParameters.checkSpaceFillAlgoWGMIsValid` — the
  `range(0, 256)` membership check carries `# pragma: no mutate` because
  `range(0, n)` and `range(n)` produce the same values; the explicit lower bound
  documents the half-open GridDim interval `[0, 256)`.

**M3 — accepted equivalent mutants (expanded mutation run).** These survivors
are behaviorally equivalent on the specific public surface under test:
- `Tensile.TensileLogic.ValidWorkGroupMappingXCC.x__cu_count_from_path__mutmut_9` —
  changing `cu` to `CU` inside the regex literal is equivalent because the search
  uses `re.IGNORECASE`.

Two former survivors are intentionally no longer accepted equivalents:
`Tensile.TensileLogic.ValidWorkGroupMappingXCC.x__validateWorkGroupMappingXCC__mutmut_14`
is avoided by making the missing-key / `-1` sentinel branch explicit before
reading the fixed `WorkGroupMappingXCC` value, and
`Tensile.Common.Utilities.xǁSpinnyThingǁincrement__mutmut_1` is killable because
`SpinnyThing.increment` now uses its `value` parameter to advance by caller
selected steps.

**M4 — widened mutation slice.** The `only_mutate` set in `[tool.mutmut]` was
extended past the original five files to add `Common/DataType.py`,
`Common/Types.py`, `Common/ValidParameters.py`, `SolutionStructs/Naming.py`, and
`SolutionStructs/Utilities.py`, with matching characterization directories added
to `pytest_add_cli_args_test_selection`. Source-path mapping for the widened slice:
DataType → `Tensile/Common/DataType.py`; CommonTypes → `Tensile/Common/Types.py`;
ValidParameters → `Tensile/Common/ValidParameters.py`; Naming →
`Tensile/SolutionStructs/Naming.py`; SolutionStructsUtils →
`Tensile/SolutionStructs/Utilities.py`.

**M5 — SolutionStructs Naming/Utilities mutation outcome.** `Naming.py`: 455
generated, 453 killed, 2 accepted equivalents, 0 no-test mutants → 100% covered
non-equivalent score (99.56% raw). `Utilities.py`: 131 generated, 131 killed, 0
survivors, 0 no-test mutants → 100% covered score. The full run had 30 no-test
mutants outside these two modules. Because `mutate_only_covered_lines = false`,
mutmut enumerates every source-line mutation; these scores exclude only the
explicit no-test entries and accepted equivalents.

**Pinned equivalent (Naming).**
`Tensile.SolutionStructs.Naming.x__getName__mutmut_{70,71}` changes the masked
`state["GlobalSplitU"] = "M"` expression at `Naming.py:172`; every string form
reaches the same pinned string-versus-integer `TypeError` before it can affect a
name. [`AIHPBLAS-4297`](https://amd-hub.atlassian.net/browse/AIHPBLAS-4297)
tracks this pinned defect.

The former WGMXCC and unreachable-abbreviation equivalents were removed as
redundant/dead source instead of being fenced, unlike M2's documented pragmas.
No new `# pragma: no mutate` fences are accepted in this round.

## D16 — BufferLoad/BufferStore promoted to Required Parameters
**Context** kernel basename hash changes across all archs; assembly verified unchanged/correct; no err or kernel-count changes."

## D17 — StreamKWorkStealing added to the required (min-naming) parameter set
**Decision:** Promote `StreamKWorkStealing` to the required (min-naming) parameter set in
`Common/RequiredParameters.py` and accept the regenerated `_codegen` / SolutionClass /
ValidParameters goldens.
**Why:** without it, two solutions differing only in `StreamKWorkStealing` would collide on the
same kernel identity name/hash.
**Verification:** only `basename` hashes + the `SKWS0` name token + the roster/valid-values entry
change (`num_keys` 334→335); no `err`, instruction-count, or emitted-assembly changes.

## D18 — LibraryIO dict-format raw logic snapshot addition
**Context:** `rawLibraryLogic` historically unpacked only list-format logic. A
dict-format input path was added to preserve the legacy tuple contract used by
older call sites (`versionString`, `scheduleName`, `architectureName`,
`deviceNames`, `problemTypeState`, `solutionStates`, `indexOrder`,
`exactLogic`, `rangeLogic`, `otherFields`). Characterization gained a new test
(`test_raw_library_logic_dict_format`) to pin this behavior.

**Decision:** Add a new syrupy golden node for the new test in
`LibraryIO/__snapshots__/test_logiccontract_char.ambr`.

**Why:** This is a **new characterization case**, not a rewrite of an existing
golden's meaning. The snapshot records the expected dict-format-to-legacy-tuple
mapping (including optional-field ordering in `otherFields`) so future refactors
cannot silently break backward compatibility.

**Alternatives rejected:**
- Avoid snapshot and assert piecemeal fields manually — rejected: weaker
  protection for tuple ordering/shape regressions.
- Update all snapshots wholesale — rejected by governance; only the single new
  node was generated.

## D19 — test_create_library_logic_dict_arch golden changed from list-shape to dict-shape
**Context:** Diff vs `develop` shows the snapshot node
`test_create_library_logic_dict_arch` in
`LibraryIO/__snapshots__/test_logiccontract_char.ambr` moved from a legacy
matching-table list representation to canonical dict-format library logic.

**Decision:** Keep the new dict-format golden and document it as intentional.

**Why:** The serialization contract being characterized is now dict-first for
`createLibraryLogic`, with explicit root keys (`ArchitectureName`, `CUCount`,
`DefaultSolution`, `Solutions`, `LibraryType`, etc.). The old list-shape golden
encoded the prior format and would now mask the intended migration. The new
golden also captures that for the gfx942 + CUCount!=304 branch, architecture is
materialized as `ArchitectureName` + `CUCount` in dict logic rather than a list
field tuple position.

**Alternatives rejected:**
- Revert to list-shaped snapshot for compatibility optics — rejected: it would
  assert obsolete output and fight the dict-format migration.
- Keep both shapes in one test — rejected: conflates two contracts; list-format
  coverage is already pinned separately via parse-list roundtrip tests.

## D20 — KnownBugs keyed on solution_name (intended behavior change)

**ADR:** [`adr/0002-knownbugs-key-on-solution-name.md`](adr/0002-knownbugs-key-on-solution-name.md).

**Decision:** `TensileLogic.KnownBugs` now keys documented `--check-all` skips on
`(path, solution_name)` (the solution's stable `SolutionNameMin`) instead of the
positional `(path, solution_index)`; `solution_index` support is dropped. The
`test_knownbugs_char.py` goldens for `test_is_known_bug_hit_and_miss` and
`test_load_roundtrip_multi` were re-recorded to match. This is an intended
behavior change (not a pinned bug): positional indices shift on re-tuning and
forced manual edits to `known_bugs.yaml`, whereas the content-derived name is
stable and self-invalidating. Motivating context: ROCM-7144.

**Note:** the two golden nodes were hand-edited to match syrupy's amber format
and must be confirmed byte-identical via `--snapshot-update` in a build
environment; the `-m unit` lane needs the compiled rocisa module, which is not
available where this change was authored.

## D21 — CustomKernels: re-target at `_readEmbeddedYaml` after Gemm-From-Anywhere removed `getCustomKernelConfigAndAssembly`

**ADR:** [`adr/0002-custom-kernels-embedded-yaml-parsing.md`](adr/0002-custom-kernels-embedded-yaml-parsing.md).

**Context:** `test_custom_kernels_char.py` (added on `develop` by #7989) pinned
`getCustomKernelConfigAndAssembly`, a raw `---`/`...` line-splitter returning
`(config_text, assembly_text)`. The Gemm-From-Anywhere branch's rewrite of
`CustomKernels.py` (proper `.amdgpu_metadata` YAML parsing for the external-
kernel `custom.config` schema) dropped that function in favor of a private
`_readEmbeddedYaml` returning a parsed dict — the two branches diverged before
#7989 merged to `develop`, so this was never reconciled. The stale import
caused a pytest **collection** error, which (per pytest's default behavior)
aborted the entire `-m unit` run before any test executed — silently hiding
every other test in the suite behind this one file, not just this module.

**Decision:** Point the test at the real replacement (`_readEmbeddedYaml`,
pinning its parsed-dict return) instead of restoring the removed function.
Also gave `test_get_custom_kernel_config_ok` its own fixture with a minimal
`amdhsa.kernels` entry (`_VALID_S_WITH_KERNEL_META`), matching the convention
in `Tests/unit/test_CustomKernelMetadata.py::write_kernel`, since
`getCustomKernelConfig`'s new no-explicit-`CustomKernel` auto-infer path
requires one (real kernel `.s` files always have one; only the bare-minimum
test fixture didn't).

**Why:** Same-purpose replacement (pin how the module reads its embedded
YAML), not a scope cut; add-only (no production code touched); private-helper
characterization already has precedent in this same branch
(`test_CustomKernelMetadata.py` imports `_parse_tensile_yaml`/`_read_asm_file`
directly).

**Rejected alternatives:**
- *Restore `getCustomKernelConfigAndAssembly`* — rejected per explicit
  direction: don't reintroduce what Gemm-From-Anywhere deliberately removed.
- *Delete the test instead of replacing it* — rejected: loses real coverage of
  the embedded-YAML parsing path with no offsetting gain.

**Validation:** `test_custom_kernels_char.py` — 11 passed, byte-identical
`.ambr` across two additional `--snapshot-update`-free re-runs.

**Residual scope (not fixed here):** unblocking collection let the full
`-m unit` suite actually run for the first time on this branch's diff, and it
surfaced 12 pre-existing failures unrelated to this file. Triaged and closed
in D22 below.

## D22 — Triage of the 12 failures D21 unblocked: 2 real regressions fixed, 10 stale-fixture goldens/asserts updated

**Context:** D21 fixed a pytest *collection* error that had aborted the entire
`-m unit` run before any test executed, on this branch's diff, since it
diverged from `develop`. With collection fixed, the suite ran for the first
time and surfaced 12 failures across 6 files, all in code this branch itself
touched. Each was triaged individually per the "did you intend to change this
behavior?" protocol in `README.md` — no blanket `--snapshot-update`, no
fixing-via-the-test of anything that was a real code bug.

**Two were real regressions; fixed the source, not the tests/goldens:**

- **`TensileLogic/HandleCustomKernel.hasCustomKernel`** — the line-scanner's
  marker pattern was changed from `CustomKernelName:` (legacy flat key) to
  `name:` (matching the new `CustomKernel:` mapping's nested name field), but
  `handleCustomKernel()` itself still explicitly accepts *either* shape
  (`sol["CustomKernel"]["name"]` or `sol.get("CustomKernelName", "")`). The
  narrowed scanner is reachable from a live gating call site
  (`TensileLogic/Run.py:105`, `if check.OnlyCustomKernels and
  hasCustomKernel(file): ...`) that decides whether a logic file's solutions
  get loaded at all under `--only-custom-kernels`-style checks — so a legacy
  `CustomKernelName:`-keyed logic file would have its custom-kernel solutions
  silently dropped from that check. Fixed by matching *both*
  `CustomKernelName:` and `CustomKernel:` (the distinctive parent keys for
  each schema — not the generic, collision-prone bare `name:`), restoring the
  original (unchanged) golden's expectation. Added
  `test_has_custom_kernel_true_new_style_mapping` since the new-style path had
  zero prior coverage.
- **`Toolchain.Component.Assembler._retargetAssemblySource`** — new,
  unconditional preprocessing step on every single assembly compile (rewrites
  a mismatched `.amdgcn_target`/`amdhsa.target` directive in the source to
  match the actual build target — the mechanism the `CustomKernels/README.md`
  Triton section describes). It called `path.read_text()` with no handling
  for a missing/unreadable file (only `UnicodeDecodeError` was caught), so any
  `srcPath` that doesn't exist yet crashes `Assembler.__call__` with a
  confusing traceback instead of reaching the actual compiler invocation
  right after, which would raise its own clear "no such file" error through
  the already-exercised `_invoke`/`CalledProcessError` path. Widened the catch
  to `(UnicodeDecodeError, OSError)`. Also added
  `test_retarget_assembly_source_rewrites_mismatched_target` and
  `..._leaves_matching_target_untouched` — the regex rewrite itself had zero
  test coverage anywhere in the repo before this.

**Everything else was a stale test double, not a code bug — production
behavior is correct; the fixtures just don't auto-default new fields the way
their real counterparts do:**

- **`ValidParameters::test_valid_parameters_{key_roster,structure}`** — clean,
  intentional, single-line-diff changes in `Common/ValidParameters.py`:
  `CustomKernelName` renamed to `CustomKernel` (matches the mapping-typed
  parameter everywhere else in this branch), and
  `AssertFree0/1ElementMultiple` / `AssertSummationElementMultiple` extended
  with `64`/`128`/`256` (larger custom-kernel tile sizes). Updated both
  goldens; reviewed the diff line-by-line (see the `.ambr` diff in the PR).
- **`TensileMain::test_arg_updated_global_parameters_*`** and
  **`PublicInputSurface::test_platform_*_branch_*`** — two independently
  hand-rolled fake-`args` builders (`_args()` / `_make_args()`, one using
  `SimpleNamespace`, one using `argparse.Namespace` directly) both predate the
  new `--validate-metadata` flag (`Tensile.py`'s `argUpdatedGlobalParameters`
  now reads `args.ValidateMetadata`). A *real* `argparse` parser always
  supplies a `store_true` flag's default (`False`), so this can't happen
  outside a test; added `ValidateMetadata=False` to both builders. Extended
  `TensileMain`'s "all overrides" test to actually cover
  `ValidateMetadata=True` (previously untested) and added a explicit
  default-omitted case; `PublicInputSurface`'s file is narrowly scoped to the
  unrelated `platform` predicate per its own docstring, so left it at the
  minimal fixture fix.
- **`TensileCreateLibraryRun::test_pass_post_kernel_info_to_solution`** — same
  shape of issue: `KernelCodeGenResult` gained a `customKernelDef:
  Optional[dict] = None` field (a real instance always has it via the
  `NamedTuple` default), but the test's `SimpleNamespace` stand-in doesn't
  auto-default missing attributes the way a `NamedTuple` does. Added
  `customKernelDef=None`/`=<a dict>` to the two cases and a new test,
  `..._carries_custom_kernel_def`, pinning the previously-uncovered
  `solution._state["CustomKernel"] = result.customKernelDef` assignment.

**Validation:** every touched file re-run individually (all green) plus two
full, `--snapshot-update`-free `-m unit` runs of the whole suite (see the PR
description for the exact before/after counts).

**Rejected alternatives:**
- *Blanket `--snapshot-update` across the whole suite* — forbidden by this
  file's own cardinal rule; would have silently accepted the two real
  regressions above instead of fixing them.
- *Leave `hasCustomKernel`/`_retargetAssemblySource` as-is and just fix the
  two tests* — rejected: both are reachable from real call sites, and
  "fixing" the test to assert the buggy behavior would have pinned a real
  regression as if it were intended, exactly what this suite exists to catch.
