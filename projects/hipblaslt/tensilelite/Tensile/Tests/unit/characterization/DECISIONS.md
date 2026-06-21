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

No other equivalent mutants are accepted yet; widening rounds append their
accepted equivalents/pragmas here, each with its one-line reason.
