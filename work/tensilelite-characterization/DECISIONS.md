# Decision log — remaining-module characterization sweep

Autonomous execution of `master-plan-remaining.md`. One entry per non-trivial
decision point: the choice made, why, and why the alternatives were rejected.
(Routine "wrote tests, hit ≥95%, committed" steps are not logged — only genuine
forks.)

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

## D7 — pre-existing unstaged `env/Dockerfile` rewrite (left untouched)
**Decision:** Leave the modified `work/tensilelite-characterization/env/Dockerfile`
unstaged/uncommitted; do not commit it and do not discard it.
**Why:** it is a substantial content rewrite (TheRock CI run base, reproducible
image) that this characterization sweep did not author and that is unrelated to
any test module. Committing it would bundle unrelated work into a per-module
commit; discarding it could destroy someone's WIP. My commits stage only
specific test-suite paths, so it will not be swept in accidentally.
**Alternatives rejected:** (a) commit it — out of scope, violates the
per-module-commit intent; (b) `git checkout` it — risks losing authored work I
can't attribute. Surfaced to the user instead.

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
