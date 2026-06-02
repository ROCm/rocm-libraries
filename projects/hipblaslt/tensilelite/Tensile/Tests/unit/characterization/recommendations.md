# Recommendations — scaling characterization across tensilelite

## Verdict: **GO** (conditional)

Characterizing `Tensile/SolutionStructs/Validators/` was high-value and
low-cost: the suite reached **96.65% line / 95.08% branch** on the module
with the remainder being provably-unreachable defensive code, added zero
risk (63 new tests, **add-only**, full `-m unit` suite unchanged at
1249 passed / 201 skipped vs 1186 baseline + 63), and produced an
executable spec for a module slated for consolidation. Recommend scaling
the pattern to the next tiers — **with eyes open about which modules are
cheap and which are not**, and after resolving two environment frictions
below so future targets don't re-pay the setup cost.

## What worked (keep doing)

- **syrupy snapshots of structured returns.** For value-returning functions
  (`matrixInstructionToMIParameters` → MI-param dict; validators → bool +
  `state["Valid"]`) snapshots pin behaviour while leaving the implementation
  free to change. `--snapshot-update` makes regeneration trivial.
- **Build consistent inputs via the code under test.** Running the
  conversion function to produce a self-consistent solution, then feeding it
  to the assertion-heavy validator, sidesteps hand-deriving ~25 interlocked
  fields. This was the single biggest effort-saver for `validateMIParameters`.
- **Minimal dicts for early-exit / reject branches.** Reaching a specific
  reject needs only the keys read before that point — far cheaper than a
  full solution.
- **Deterministic side effects.** Pure target + `printRejectionReason=False`
  meant no RNG/time/stdout to manage. Pick pure-ish targets first.

## Cost model (per future target)

Calibrated from this target (1 module, 3 files, 239 stmts, ~63 tests):

| Target profile | Examples (see module map) | Effort to ≥95% line | Notes |
|---|---|---|---|
| **Pure validators / table logic** | TensileLogic validators (group 8) | ~0.5–1 day | Same shape as this target; cheapest. |
| **Structured-output, pure-ish** | `LibraryIO` round-trips, `CustomYamlLoader`, Utilities (groups 5/10) | ~1–2 days | Snapshot the structured form; add path/timestamp normalisation. |
| **Core types & params** | `Common/{DataType,ValidParameters,...}` (group 1) | ~2–4 days | Large surface, but pure; high leverage (everything depends on it). |
| **Solution/problem model** | `Problem.py`, `Solution.py` (group 2) | ~1–2 weeks | `Solution.py` is 5.2k LOC and deeply stateful; characterize in slices, expect resistance. |
| **Emitters / orchestration** | KernelWriter*, benchmark/client (groups 4/6) | not recommended as line-coverage targets | 45k LOC stateful asm emit + GPU/subprocess; snapshot only narrow structured slices, accept partial. |

Rule of thumb: effort scales with **statefulness and input-derivation cost**,
not raw LOC. A 5k-LOC pure module is cheaper than a 1k-LOC stateful one.

## Blockers / frictions to fix before scaling

1. **Coverage `--cov` must be a path, never a dotted module.** A dotted
   `--cov=Tensile.X` makes coverage import the package, re-initialising the
   `_rocisa` nanobind extension → SIGABRT. Use `--cov=Tensile/X` (path).
   The project's own tox `coverage`/`coverage-unit` envs use
   `--cov=Tensile --cov=rocisa` from the tensilelite dir where both resolve
   as directories, so they are safe — but anyone scoping coverage to a
   sub-package by dotted name will hit the abort. **Recommend** a short note
   in the dev docs (a *new* doc, per add-only) and, when the add-only
   constraint is lifted, a wrapper or `coverage` config that pins path-mode.

2. **`branch=True` in `pyproject.toml` makes the headline % stricter than
   the goal's "line coverage".** Line coverage here is 96.65%; the blended
   figure is 95.08%. Both clear 95%, but future pure-defensive code (dead
   branches that can't be pragma'd under add-only) will drag the blended
   number. **Recommend** reporting line coverage explicitly for the
   characterization gate, or (post add-only) a `# pragma: no cover` on the
   genuinely-dead lines documented in resistance.md.

3. **Dead/defensive code surfaces during characterization.** This target
   exposed ~7 unreachable lines (940/941 remap, symmetric-table dtype
   fallback, a label fallback). Characterization is a good *forcing function*
   to find and delete dead code — **recommend** filing cleanups (separate
   from this add-only effort) for the items in resistance.md.

4. **Snapshots can couple to the toolchain.** `asmCaps` come from the live
   assembler. Stable in the pinned dev image; could drift across compiler
   versions. **Recommend** standardising on the dev container
   (`tensilelite-char:dev`) for snapshot generation/CI, or injecting a
   synthetic `isaInfoMap` for targets where caps matter.

## Suggested next targets (in order)

1. **TensileLogic validators (group 8)** — same cheap shape, table-driven,
   partial tests already exist; fast ≥95%.
2. **`LibraryIO` (group 5)** — highest value: it is the solution-library
   (de)serialisation contract; snapshot the structured YAML/dict round-trip
   with path/timestamp normalisation. This is the "snapshot the structured
   solution form" the original goal pointed at.
3. **`Common` core types (group 1)** — high leverage, pure.

Defer groups 2/4/6 (stateful/large/GPU) until the cheap, high-value tiers
are done; characterize them only in narrow structured slices.
