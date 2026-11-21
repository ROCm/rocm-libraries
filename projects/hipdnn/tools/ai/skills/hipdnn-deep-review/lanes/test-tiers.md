# Lane: test tier placement (`TIER`)

Judge which tier each test the change adds or moves lands in, and what runtime it adds to that tier. CI time is limited: `quick`
is paid for in every run of every tier, so it holds only ultra-fast tests; `standard` is the default home. Coverage gaps are
`TESTB`'s and test quality is `TESTQ`'s; this lane owns placement and cost only. It runs no test: the follow-up `TIER-f1` adds
times from `RUN`.

## Tiers and budgets

Read these at `$HEAD`; the YAML wins over this table. Sources: `projects/hipdnn/docs/TESTING.md` *Test tiers*,
`dnn-providers/integration-tests/docs/running-tests.md` *Test tiers*, `dnn-providers/integration-tests/docs/adding-tests.md`
*Choosing a tier*, and the `execution_settings.category_timeouts` of each `*test_categories*.yaml`.

| Tier | Holds | Budget | Runs in PR CI |
|---|---|---|---|
| `quick` | Ultra-fast CPU-only unit tests, and the smallest gross-breakage enablement set | Whole tier under 5 min | Yes: `standard` includes it; alone on run-all fallbacks and dispatches |
| `standard` | The default: slower enablement and high-priority regression cases | Whole tier (quick included) under 30 min | Yes, every PR touching hipDNN |
| `comprehensive` | Nightly breadth | Under 2 h | No |
| `full` | Stress, exhaustive, cases redundant with lower-tier coverage, release testing | None (2 h or more) | No |

Tiers cascade: a test in `quick` also runs in `standard`, `comprehensive` and `full`. Every CI test step is limited to 30 minutes
per shard; a tier that runs longer reaches its budget by sharding, and `quick` is never sharded. Job-level timeouts and shard
arithmetic are `ROCK-f1`'s ([therock.md](therock.md) section 6); do not repeat them here.

## Thresholds

One table, so a retune is one edit. A case is one GTest instance (each parametrized instance counts) or one bundle case (each
row of a sweep counts).

| Name | Value | Meaning |
|---|---|---|
| `QUICK_CASE_MAX` | 2 s | Longest a basic unit test added to `quick` may take; a case that is not a basic unit test is flagged whatever its time |
| `QUICK_ADDED_MAX` | 30 s | Most the change may add to `quick` in total (10% of its budget), even when every case is under `QUICK_CASE_MAX` |
| `STANDARD_CASE_SLOW` | 30 s | A single `standard` case longer than this belongs in `comprehensive` or `full` unless it is the cheapest case that covers its behavior |

## Method

1. **Enumerate.** From `git -C "$CLONE" diff -M "$BASE...$HEAD"` (`$BASE` is the merge base, [common.md](common.md) *Variables*; when
   `git merge-base --is-ancestor "$BASE" "$HEAD"` fails, use `git merge-base` instead, so target-side additions never appear as
   deletions), list every added, moved or renamed test: bundle files (and `<bundle>.support.json` or `support.json` sidecars, which add
   no case), GTest `TEST*`/`INSTANTIATE_TEST_SUITE_P` instances, pytest tests, and each change to a `*test_categories*.yaml` pattern,
   label or `category_timeouts`. Count cases: expand every parametrized instantiation and every sweep row; read the sweep file, do not
   count files.
   - A rename that leaves a test in its tier is not an addition; a rename or edit that changes its tier is a move.
   - A case that skipped on an arch (an `onlyOn` or arch gate) and executes there at `$HEAD` counts as added for that arch. Record the
     count per arch.
   - Rows inside the body of one test (a table it iterates) add no instance. Record `rows +<added> -<removed>` for that test; for any other
     change in the work a test does (an added assertion, a row that now executes on one more arch) record `work: <note>`. `TIER-f1` prices
     the test as its time at `$HEAD` minus `$BASE`. With no measurement it is `UNVERIFIED`, never zero.
2. **Resolve the effective tier** of each test at `$HEAD`. Never take the directory or prefix on trust: apply the owning binary's
   YAML patterns (`BUILD` owns whether the YAML is correct; this lane reads it). Where a build exists, `ctest -N -L '^<tier>$'` on the
   `RUN` tree lists what a tier selects (labels are regexes: anchor them, and check `Total Tests:`). Rules that decide the answer:
   - **Bundle:** its top-level directory under `integration-test-bundles/` (`quick/`, `standard/`, `comprehensive/`, `full/`) is the
     suite prefix `quick_`, `standard_`, ...
   - **C++ test in an explicit-include provider YAML:** its instantiation prefix (`Smoke/`, `Quick/`, `Standard/`, `Comprehensive/`,
     `Full/`). No prefix: the lane selects it only from `comprehensive` up (the miopen lane through `Integration*`), so it never runs in
     PR CI.
   - **A provider's own GTest binary** (for example `hip_kernel_provider_tests`): find the YAML through the link graph, never the lane
     name: source file, CMake target, its `apply_test_category_labels` call, the YAML variable it passes. A catch-all `quick` (`*` minus
     `Comprehensive*` and `Full*`) puts an unprefixed test and `Smoke/` in `quick`; the "no prefix runs from `comprehensive` up" rule above
     holds only for an explicit-include YAML.
   - **Test in the project's own binaries** (`hipdnn_integration_tests_unit_tests`, `hipdnn_golden_data_tests`, `hipdnn_gpu_ref_tests`):
     `quick` is a catch-all, so a test with no tier prefix lands in `quick`. A tier missing from either filter spelling
     (`Standard*` and `standard_*`) leaves its cases in `quick`. In `hipdnn_gpu_ref_tests`, reduced-precision, fp8, int8, TF32 and
     mixed-type cases belong in `standard` (each extra datatype is a separate hipRTC compile).
   - **hipDNN core and samples** (`projects/hipdnn/test_categories.yaml`): every test is in `quick` by design, because core tests are
     fast. The same `QUICK_CASE_MAX` limit applies to each added case; a core test that needs a device or a large shape does not
     belong there.
   - **Lane narrowing:** a lane YAML can narrow a tier's patterns (ASM_SDPA selects `*Sdpa*`; the gfx950 attention lanes select
     `*_SdpaFwd_Default.*` in every tier). A test the narrowed pattern does not select runs in no tier of that lane.
   - **Split check, every YAML the change touches.** A YAML is touched when a changed test source (an edit counts, an added test is not
     needed) belongs to a binary it categorizes. Compare its `quick` and `standard` patterns and excludes. When `standard` selects exactly
     what `quick` selects (the hip-kernel-provider `HIP_MLOPS_ENGINE_test_categories.yaml` says "No distinct Standard* prefix exists, so
     standard runs the same tests as quick"), the tier split does not exist: a `Standard` prefix or directory runs in `quick` too, so
     nothing can be moved out of `quick` without leaving PR CI. Record `split: missing` once for the YAML, with every binary it
     categorizes (a YAML shared by several binaries is one record). A binary that holds tests belonging in `standard` has those tests made
     standard tier: suggest it, and leave how (rename a suite, a `Standard` prefix, a parametrized wrapper) to the developer. The fix
     depends on the YAML's shape:
     - catch-all `quick` (`*` minus tier excludes): add `Standard*` to `quick`'s exclude list; `standard` already selects `*`.
     - explicit-include `quick`: list `Standard/*` only under `standard` (reference: miopen-provider's `test_categories_integration.yaml`,
       `quick` lines 28-32, `standard` lines 42-46). Check what the unprefixed unit tests need first: an explicit-include `quick` drops them.
3. **Classify each `quick` case** as a basic unit test (host-only, no device, no hipRTC compile, no production-sized graph, no file or
   golden-data load beyond a few KB) or not. A GPU test, a bundle, or a case that builds or compiles a kernel is not a basic unit test; nor
   is a `TEST_F` whose fixture constructor or `SetUp` touches a device, even when the body does not. A `TEST` that only reads generated
   config tables or calls pure functions is basic. Read the test body and its fixture, and follow calls one level into production code: a
   call that queries or uses the device (`isApplicable` with a device check, a plan build) makes it not basic, whether or not the test
   guards with `SKIP_IF_NO_DEVICES()`. When unsure, it is not: classify from the source, never from the case's name or the suite it sits
   in.
4. **Cost.** Without times, give the static facts the cost follows from (case count, shapes, device need, datatype count) and mark
   the cost `[INFERENCE]`. `TIER-f1` replaces it with `RUN`'s measured times. A case whose time is unknown is never counted as free.
5. **Recommend a tier** for each finding: the cheapest tier that still finds the failures the test exists to find
   (`adding-tests.md` *Choosing a tier*). `quick` keeps at most one basic (host-only) test for the feature the change adds; everything
   else that is not basic moves to `standard`, which runs on every PR. Never recommend keeping a representative of each family, op or
   datatype in `quick`: one per family adds up across the tier. A move to `standard` leaves PR CI coverage intact; a move to
   `comprehensive` or `full` takes the behavior out of PR CI and is a `TESTB` handoff, not a clean pass
   ([test-breadth.md](test-breadth.md) *Dimensions*, *Tiers*).

## Findings

Kind is `source-claim` ([../findings.md](../findings.md) *Proof per kind*); `Proof` cites the YAML pattern or directory that places the
test and, once `TIER-f1` ran, the `RUN` log row for the time. Tiers follow [../policy.md](../policy.md) *Review rules*.

- A case added to `quick` that is not a basic unit test: `IMPORTANT`, whatever its measured time and without waiting for `TIER-f1`.
  The reviewer must look at it and either move it (`Fix`: the `standard/` bundle directory, or any way of making the test standard tier;
  the developer chooses how) or accept it. Add one **Needs human decision** entry per group of cases that share a prefix and a reason
  (`<ID>: Keep <tests> in quick? Options: move to standard, accept in quick. Recommendation:
  move to standard, because quick is paid for in every run of every tier.`). Acceptance is a human `ACCEPTED-RISK` in the ledger; the lane
  never downgrades it, and a time under `QUICK_CASE_MAX` does not close it. When the YAML is `split: missing`, "move to standard" is
  not available: the options are instead (A) the split is in place by the end of this change (here, or in a change merged before it):
  implement it as the YAML shape says (*Method* step 2, split check) and move the test; and (B) the test stays in `quick`: accept it with
  a tracker that names the split. A follow-up that implements the split later is (B) until it lands; the ledger records it as
  `ACCEPTED-RISK` with that tracker. Recommend (A); never recommend moving the test to `comprehensive` as the default, because that
  leaves the behavior out of PR CI (`TESTB` handoff). The split is a missing capability, not a choice: do not offer "accept with no
  split" as a clean outcome.
- `split: missing` for a YAML the change touches: file one finding per YAML (not per binary or test), listing its binaries. It is always
  `FUTURE WORK`: the gap exists at `$BASE` ([common.md](common.md) *Rules*). The `IMPORTANT` finding above cites it and carries the
  closure gate. `Fix`: the YAML change for its shape, and making the tests that belong in `standard` standard tier in each binary (name
  them only where the change adds a non-basic test; the developer chooses how). Write `BUILD: implement the tier split for <YAML>`
  under **Handoffs**.
- A basic unit test added to `quick` whose measured time exceeds `QUICK_CASE_MAX`: `IMPORTANT`; same `Fix`.
- A change that adds more than `QUICK_ADDED_MAX` to `quick` in total: `IMPORTANT`.
- A test with no tier prefix that lands in `quick` through a catch-all: the two bullets above apply to it (not a basic unit test, or over
  `QUICK_CASE_MAX`); name the prefix that fixes it.
- A `quick` tier whose projected total (`TIER-f1` only: the baseline from `ROCK`'s `Timing-base:` line for the `quick` entry plus the
  added time) exceeds 5 min: `IMPORTANT`. A baseline that is `UNVERIFIED (no CI log)` carries through as `UNVERIFIED`, never as a pass.
- A `standard` tier whose projected total exceeds 30 min, or a single `standard` case over `STANDARD_CASE_SLOW` with no cheaper
  covering case: `SUGGESTION`; `IMPORTANT` when the projection exceeds 30 min.
- A new test that runs in no PR tier (no prefix in a provider lane, a narrowed pattern that skips it) and has no stated reason:
  `IMPORTANT` when it is the only test of new behavior, else `SUGGESTION`.
- A basic unit test in `quick` at or under `QUICK_CASE_MAX` is not a finding. A slow case already in the right tier is not a finding. A
  test the change moves out of `quick` to `standard` is a credit: record it, with the time it takes off the tier.

A YAML change that alters what a tier selects is a finding only through the cost it adds; whether the pattern list is correct (cascade,
both spellings, labels) is `BUILD`'s: write `BUILD: <one line>` under **Handoffs**.

## Report lines

Write under **Checked, no finding**, with no `|` ([../findings.md](../findings.md) *Lane report*), one line per test group (a bundle
directory, a GTest suite or an instantiation prefix):

`Tier: <path or suite> <n> cases [(<arch>=<n> ...)] [rows +<added> -<removed>] [work: <note>] class <basic|not basic|n/a> declared <tier> effective <tier> cost <s>s (<source>) recommend <tier|keep>`

Write an unmeasured cost as `cost UNMEASURED (static [INFERENCE])`. Split a group that spans binaries or datatypes into one line each.
`<n>` and each `<arch>=<n>` count the cases the change adds or un-gates, not every case in the group; give group totals in prose.
`recommend standard (after split)` is the form when the recommendation depends on a missing split landing. For each `split: missing`
YAML write one line: `split: missing <YAML> binaries <list>`.
`<source>` is `static [INFERENCE]` in the main pass. `TIER-f1` also writes, for `quick` (budget `300s`) and for `standard` (budget
`1800s`), one line each:

`Tier-total: <tier> added <s>s baseline <s>s budget <s>s`

`baseline` is `ROCK`'s `Timing-base:` max-shard value for the tier's entry, or `UNVERIFIED (no CI log)`. A budget applies per CTest entry
and per arch job: an un-gated case adds to that arch's `quick` entry only. Added time counts in every entry that selects the test; where
`standard` selects what `quick` selects, both entries run it, so compare each entry with its own budget and never sum the two.

## Reconcile with `RUN` (`TIER-f1`)

Runs after `RUN` returns, and only when the `TIER` report has a `Tier:` line. Read the `TIER`, `RUN` and (when it ran) `ROCK` reports and
the `RUN` logs they cite; write the `TIER-f1` lane report and do no other check.

1. For each `Tier:` group take the time of each case from the `[       OK ] <case> (<ms> ms)` lines of the logs `RUN`'s suite rows cite,
   at `$HEAD` (and at `$BASE` for a moved or altered case). Count only cases that execute; a skipped case counts 0. A case on another arch is
   `estimate (<arch>)`. A group with no case time is `UNVERIFIED (no case times)`: write no cost for it, and add `RUN: time <suite or
   bundle>` under **Handoffs** so a re-dispatched `RUN` measures it (a device run needs the user's approval).
2. Rewrite each `Tier:` line with the measured cost, `<source>` naming the `RUN` log rows, and write the `Tier-total:` lines from the
   `ROCK` report's `Timing-base:` lines (none: `UNVERIFIED (no CI log)`).
3. Apply *Findings* to the measured times. Drop an `[INFERENCE]` finding that the measurement refutes; keep it `UNVERIFIED` when no
   time exists.
