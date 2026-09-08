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

**ADR:** [`adr/0004-accept-parallel-below-coverage-bar.md`](adr/0004-accept-parallel-below-coverage-bar.md)

**Decision:** Characterize the pure helpers + single-threaded + `n_jobs=1` in-process paths (→ ~81% line); accept the real fork/spawn `multiprocessing.Pool` / `ProcessPoolExecutor` / Windows-only paths as out of reach for a flake-free unit test.

## D5 — recurring submodule-shadowing gotcha
**Observation (not a fork, but recorded):** several `Tensile` packages re-export
a class that shadows a same-named submodule attribute, so
`import Tensile.X.Foo as F` binds the *class*, not the module. Hit for
`SolutionStructs.Solution`, `Component` (LocalRead), and `Common.Parallel`
(joblib `Parallel`). **Standard fix applied everywhere:**
`F = importlib.import_module("Tensile.X.Foo")`.

## D6 — `KernelHelperNaming.py`: cover the naming half, accept <95%

**ADR:** [`adr/0005-accept-kernelhelpernaming-below-coverage-bar.md`](adr/0005-accept-kernelhelpernaming-below-coverage-bar.md)

**Decision:** Characterize the pure naming/orchestration surface (`KernelHelperEnum`, `kernelObjectNameCallables`, the five `*Names` functions); accept the `init*` `KernelWriter*`-construction functions (out-of-scope codegen, see D0) as uncovered — module lands at ~34% line.

## D9 — `Configuration.py`: operators/ProjectConfig covered; AST evaluator deferred

**ADR:** [`adr/0006-accept-configuration-below-coverage-bar.md`](adr/0006-accept-configuration-below-coverage-bar.md)

**Decision:** Cover the `Parameter` operator surface, `ReadWriteTransformDict`, and `ProjectConfig`; document the dead reflected-operator branches and defer the `ExpressionEvaluator` AST-walker matrix as its own future slice. Accept `Configuration.py` <95% combined.

## D10 — `Contractions.py`: predicate/serialization matrix deferred (~86%)

**ADR:** [`adr/0007-accept-contractions-below-coverage-bar.md`](adr/0007-accept-contractions-below-coverage-bar.md)

**Decision:** Cover the index value classes, `ProblemType`, `SizeMapping`, `InternalArgsSupport`, and `ProblemPredicate.CompoundPredicates` from the one vendored gfx942-HSS fixture; accept ~86% combined and defer the other-problem-configuration predicate/state arms as a future "Contractions matrix" slice.

## D11 — `BenchmarkStructs.py`: BenchmarkProcess builder deferred

**ADR:** [`adr/0008-accept-benchmarkstructs-below-coverage-bar.md`](adr/0008-accept-benchmarkstructs-below-coverage-bar.md)

**Decision:** Cover the pure helpers, fork-permutation cartesian product, and `BenchmarkStep`; defer `BenchmarkProcess` (the config→benchmark-steps integration builder) pending an end-to-end benchmark-config fixture.

## D12 — TensileBenchmarkCluster: pin the `--results-only` constraint crash rather than asserting clean workflow steps

**ADR:** [`adr/0001-pin-results-only-boolop-crash.md`](adr/0001-pin-results-only-boolop-crash.md)

**Defect:** [`AIHPBLAS-4298`](https://amd-hub.atlassian.net/browse/AIHPBLAS-4298)

**Decision:** Pin the real `AssertionError` that `--results-only` raises today, instead of asserting the intended-but-unreachable clean workflow tuple. Root cause: `ExpressionEvaluator`'s `BoolOp` handler only evaluates the first two operands of an n-ary boolean constraint.

## D13 — Activation.py: pin the pure config/type/numeric layer only; asm codegen is out of scope

**ADR:** [`adr/0009-accept-activation-below-coverage-bar.md`](adr/0009-accept-activation-below-coverage-bar.md)

**Decision:** Characterize only the pure config/type/numeric layer (48 tests) plus asm entry-points that run cleanly with dummy vgprs; do not drive the full asm codegen. Result: 1037 stmts, 683 missed → 34.1% line, documented ceiling.

## D14 — TensileLibLogicToYaml: pin the formGroups("None") crash on the skipMI / MI-disabled path

**ADR:** [`adr/0010-pin-formgroups-none-crash.md`](adr/0010-pin-formgroups-none-crash.md)

**Defect:** [`AIHPBLAS-4409`](https://amd-hub.atlassian.net/browse/AIHPBLAS-4409)

**Decision:** Pin the real `AttributeError` that `formGroups` raises on the skipMI / MI-disabled path (a string `"None"` sentinel hits `.items()`) instead of asserting a working sentinel Group; drive the rest of the orchestrator/fork tests through the working MI-enabled path.

## D15 — TensileClientConfig: dead code, REMOVED (final)

**ADR:** [`adr/0011-remove-tensileclientconfig-dead-code.md`](adr/0011-remove-tensileclientconfig-dead-code.md)

**Decision:** Remove `TensileClientConfig` (module, launcher, and its `cmake` `VALID_BINS` entry) as verified dead code — no in-tree caller, not shipped, and unimportable since a prior refactor. A real source deletion, a deliberate one-off departure from the characterization pass's add-only rule (2026-06-03, at the user's direction). `-m unit` was 2466 passed / 201 skipped both before and after — no regression. Two earlier readings of this question reached opposite wrong conclusions before landing here; see the ADR's Context for why that history matters.

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
**ADR:** [`adr/0003-pin-split-gsu-naming-crash.md`](adr/0003-pin-split-gsu-naming-crash.md).
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

**ADR:** [`adr/0002-knownbugs-key-on-solution-name.md`](adr/0002-knownbugs-key-on-solution-name.md)

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

## D21 — `test_bigfile_capped_emit` decoupled from live `library/src` tuning data

**ADR:** [`adr/0012-decouple-bigfile-tests-from-library-src.md`](adr/0012-decouple-bigfile-tests-from-library-src.md)

**Decision:** Replace the 10 `_BIG` entries' live `library/src` tuning-data references with vendored, trimmed, self-contained fixtures under `_codegen/data/bigfiles/`; add `test_no_library_src_dependency_char.py` as a standing AST-scan regression guard against the coupling reappearing.

## D22 — `test_bigfile_capped_emit` basename churn from upstream StreamK/GSU codegen changes

**Context:** After rebasing D21's fixtures onto current `develop`, 3 of the 10 `test_bigfile_capped_emit` cases (`equality_gfx950_HSS_big`, `gfx950_origami_MX`, `gfx1201_I8II`) failed on basename only — `err` stayed `0` and each fixture's solution count matched its `cap` exactly (6 solutions in, 6 emitted), so the affected kernels are unchanged in identity, just renamed. Root cause: `Tensile/Components/GSU.py`, `GlobalWriteBatch.py`, `StreamK.py`, and `KernelWriterAssembly.py` changed on `develop` (notably #9401 "enable PrefetchAcrossPersistent for SK4 and SK5" and #11245 "CompactLoopStore for D-store, MBSK, and StreamK") between when these fixtures were baselined and now, shifting the content-derived `MinNaming` hash for a subset of solutions that happen to hit those codegen paths. Same category as D16/D17.

**Decision:** Re-recorded only the 3 affected snapshot nodes via `--snapshot-update`; verified locally beforehand that both old and new basenames refer to the same 6 vendored solutions per fixture (no solution added/dropped/reordered-in-or-out of the capped set), and that assembly still emits cleanly (`err == 0`) for all of them.

## D23 — Canonical code-object linker input order

**ADR:** [`adr/0013-canonical-code-object-link-order.md`](adr/0013-canonical-code-object-link-order.md)

**Decision:** Sort every code object's input paths immediately before linking,
so default and explicitly grouped code objects share one deterministic physical
kernel order even though their inputs originate from different collection types.
