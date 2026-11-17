# AMD LLVM capability churn report

This report assesses compiler-capability churn across the selected public weekly
wheel snapshots in `historical_study.md`. Its scope is the focused behavior
vector emitted by the standalone probe for `gfx942`; it is not a compiler
qualification matrix and contains no workload performance results.

## Executive result

The study has 43 weekly columns from W06 through W46, including the separately
tracked W26 and W29 cuts. Across the 42 adjacent boundaries, the observed
capability vector changes at seven boundaries. Most compiler source revisions
therefore do not change a behavior rocKE currently knows it needs to navigate.

This supports a small behavior manifest rather than a release or commit-hash
lookup. It also shows why the manifest cannot be one generation label: the seven
boundaries affect independent subsets of the vector.

## Observed change boundaries

| Boundary | Changed observations | Interpretation |
| --- | --- | --- |
| W11 → W12 | FP16 builtin source ABI | Frontend source ABI changes independently of LLVM-IR declarations. |
| W13 → W14 | Module flags | Module preamble metadata can change while layout and intrinsic ABIs remain stable. |
| W20 → W21 | Optimization-level codegen | Pipeline behavior can regress at one optimization level while normal optimized codegen remains accepted. |
| W28 → W29a | Module flags | A second preamble change adds unwind-table metadata without changing the canonical layout. |
| W34 → W35 | Legacy `p1` / i32 candidate; Buffer-resource accepted ABIs | One compiler begins accepting the legacy spelling and retaining both indexed count-width overloads. |
| W37 → W38 | Canonical target triple | Target spelling changes independently of data layout. |
| W40 → W41 | Complete data layout; Pointer address spaces 10–15; Layout policy L0/L1/L2/L3; Named-barrier pointer AS; Optimization-level codegen | Several facts move at the same sampled boundary, but remain separate capabilities because no dependency has been established. |

The reported upstream-development LLVM major also changes in the sampled
history, but it is provenance rather than a member of the capability vector.
Nearby wheels can retain an older major label after the associated candidate
branch has moved, reinforcing that distinction.

## Stable observations

The following measured contracts remain stable throughout all weekly columns:

- kernel calling convention;
- FP8/BF8 MFMA packed operand type;
- scaled-MFMA nine-argument call ABI;
- requested and generic processor-name acceptance for the selected target;
- acceptance of the three sampled backend controls; and
- minimal codegen acceptance of the sampled waves-per-EU and zero-AGPR
  attributes.

Stability is useful evidence, but option or attribute acceptance does not prove
that a request changes final ISA or resources. Those effect checks remain an
explicit probe-coverage gap.

## Churn and toil

### Probe execution

The probe process is bounded, host-only, and cacheable. One command selects a
wheel environment, runs the same candidate corpus, validates the resulting JSON,
and prints a compact summary. Compiler, target, probe-source, and schema-source
changes all invalidate the cache automatically.

### Manual interpretation

The principal toil is no longer collecting version strings or hand-editing a
major-to-profile ladder. It is classifying a newly changed cell:

1. confirm the transition with adjacent wheel snapshots;
2. identify the earliest compiler stage that differs;
3. decide whether the result is a portable focused capability or
   workload-specific qualification evidence;
4. add a bounded reproducer when the current corpus does not expose the change;
5. replay the corpus and update the matrix key.

The seven observed boundaries collapse many compiler source revisions into a
small number of reviewable behavior changes. The W40-to-W41 boundary also shows
why grouping coincident changes into one profile would create future toil: a
later compiler can move any one of those facts independently.

## Steady-state schema size

The current manifest has twelve focused capability IDs. The observation model
covers canonical values, candidate acceptance/rejection, stage-specific
outcomes, crashes, timeouts, and compact output evidence without adding a new
schema file for each probe expansion.

The expected steady-state rule is:

- add a probe only for a source-derived compiler behavior or a bounded
  historical regression class that rocKE must distinguish;
- keep complete-kernel schedules, resource counts, and performance results in
  qualification evidence keyed by compiler and workload identity;
- retire a focused probe only after its behavior is settled outside the
  supported compiler window and no active consumer needs the distinction; and
- update the single checkout-local schema, probe, consumers, and study together.

## Remaining coverage work

The source and public project-history regression archaeology identifies the
next closure work:

1. generate complete frontend builtin and LLVM declaration coverage from the
   source registries;
2. integrate operation-by-processor lowerability and normalized fatal-stage
   outcomes;
3. extend optimization-level reproducers into parser, optimizer, codegen, link,
   and assembly stage matrices;
4. prove backend-option and function-attribute effects in final ISA/resources;
5. add focused pass-transformation, scheduling, wait, and resource sentinels;
   and
6. preserve compile-route context so Clang, hipcc, and loaded COMGR evidence are
   never substituted for one another.

These are probe-family and registry-completeness tasks, not requests for a new
compiler-version taxonomy.
