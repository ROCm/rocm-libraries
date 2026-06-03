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
