# Support Claims Failure-Mode Runbook

Every `--enforce-support-claims` finding ends with a stable anchor into
this file. The anchor names match the literals in
`src/harness/SupportClaimsVerifier.hpp` — renaming an anchor without
updating the verifier breaks the CI-log link.

## Rule A — claim broken

**Anchor:** `#rule-a-claim-broken`

**What triggered it.** A test ran and recorded an observation whose
`(op_chain, io_dtype, layout)` tuple lies in some `[[supported.matchers]]`
cross-product for the current `(arch, platform)`, but the engine returned
no support for the graph. The TOML claims support; runtime disagrees.

**What the message names.** The full GTest name of the offending test,
the observed `op_chain`, `io_dtype`, and `layout`, and the matcher
location is implicit (the verifier lists the rule once per failing
record — find the matcher by searching the sidecar for the op_chain).

**Likely root causes, ranked.**

1. **Engine regression.** A solver dropped support for a shape; the
   matcher was correct yesterday. → Fix the engine or roll back.
2. **Matcher too wide.** The matcher includes a tuple the engine never
   supported. → Narrow `op_chains`, `io_dtypes`, or `layouts`.
3. **Genuine known-broken combo.** The engine *would* dispatch but
   results are wrong → add a `[[test_skips]]` entry in the main TOML
   with a reason (issue link), and tighten the matcher to exclude the
   tuple so the skip and the claim don't shadow each other.

**Remediation steps.**

- Run the offending test in isolation with `--gtest_filter='SuiteName.TestName/Param'`
  to confirm the regression reproduces.
- If the engine genuinely regressed, fix it; a passing verifier run is
  the merge gate.
- If the matcher was over-broad, hand-edit the sidecar to remove the
  tuple OR regenerate via `--write-support-claims` on hardware. The
  auto-gen tool's safety check (RFC 0012 §7) won't reintroduce a U
  tuple into any matcher.

## Rule B — issue before the test runs

**Anchor:** `#rule-b-issue-before-the-test-runs`

**What triggered it.** The harness registered the test in `SetUp()` but
no `recordGraphSupport` entry exists for it. Either `SetUp()` errored
before reaching the first `verifyGraph()` call, or the test code calls
`verifyGraph()` from a code path that never runs (an `ASSERT_*` fired
before it). The verifier cannot evaluate any claim for this test because
it never observed a graph.

**Likely root causes.**

1. `SetUp()` failed an `ASSERT_EQ` (e.g., `hipInit` returned non-success).
2. The test built the graph but bailed out before calling `verifyGraph()`.
3. A test class derives from the harness but uses its own verification
   path that never records.

**Remediation steps.**

- Fix the underlying test failure first — Rule B is always a symptom of
  another problem, never the bug itself.
- If the test genuinely doesn't use `verifyGraph()`, it should not
  inherit from `IntegrationGraphVerificationHarness`. Move it to a
  plain `::testing::Test` so it isn't in the harness registry.

## Rule C — zero-coverage matcher

**Anchor:** `#rule-c-zero-coverage-matcher`

**What triggered it.** A matcher's `op_chains × io_dtypes × layouts`
cross-product matched zero observed records.

- In full CI mode (no `--gtest_filter`, no `GTEST_TOTAL_SHARDS`,
  `--enforce-support-claims` is in effect): hard **FAIL**.
- In partial runs (any filter or shard env set): downgraded to a
  **note** because a missing tuple may simply have been filtered out.

**Likely root causes.**

1. The catalog shrank — an `INSTANTIATE_TEST_SUITE_P` was deleted and
   the matcher's tuples no longer exist. The matcher is stale.
2. The matcher's `op_chain` string was hand-edited and now doesn't
   match `describeGraph`'s actual output (typo, casing).
3. A `describeGraph` format change shipped without a coordinated
   sidecar regen (this would be a process failure — RFC 0012 §12).

**Remediation steps.**

- Regenerate via `--write-support-claims` on hardware. The condensation
  pass only emits matchers backed by observed tuples, so a stale
  matcher will drop out of the new file.
- If you don't want to regenerate, hand-remove the stale matcher.
- If the issue is a `describeGraph` format change, the fix is broader:
  bump `[meta].version`, regenerate every engine's sidecar, ship as
  one coordinated PR.

## Rule D — engine over-claim

**Anchor:** `#rule-d-engine-over-claim`

**What triggered it.** A test FAILED, the engine returned support for
its graph, but no matcher covers the observation. This is a note tacked
onto the existing test failure — not a separate verifier failure.

**Likely root causes.**

1. The engine's `get_ranked_engine_ids` is too permissive: it claims a
   shape it can dispatch to but doesn't execute correctly.
2. The test was added recently and exercises a combination that's a
   real gain — but the TOML hasn't been updated yet (also covered by
   Rule E for passing tests).

**Remediation steps.**

- Fix the test failure itself first. Once the engine either stops
  claiming support OR starts producing correct results, Rule D
  disappears.
- If the engine should be claiming support, add the tuple to a
  `[[supported]]` block (regenerate, or hand-edit) so the next CI run
  reports it under Rule A if it regresses.
- If the engine should not be claiming, tighten the engine's
  applicability logic. This is a per-provider concern.

## Rule E — unclaimed gain

**Anchor:** `#rule-e-unclaimed-gain`

**What triggered it.** A test PASSED, the engine returned support, and
no matcher covers the observation. The engine supports something the
sidecar doesn't claim — this is silent capability drift.

This is a **warning**, not a failure. RFC 0012 §3.2 deliberately keeps
"engine starts supporting something new" out of the fail list: failing
it would force TOML updates on every new test family and create PR
churn that discourages adding tests.

**Remediation steps.**

- If the gain is intentional: regenerate on hardware via
  `--write-support-claims` and commit the diff. The new tuple is now
  protected against future regressions (Rule A).
- If the gain is accidental (e.g., the engine started dispatching to
  shapes it shouldn't), tighten `get_ranked_engine_ids`.
- If you ignore the warning, the only consequence is that a future
  regression on this tuple will be invisible to the verifier. You're
  not blocked.

## Pre-run refusals

These aren't rules — they're preconditions checked before
`RUN_ALL_TESTS` so the verifier never runs against a misconfigured
session. Each prints `Error: ...` and returns non-zero immediately.

| Refusal | Why |
|---|---|
| `--enforce-support-claims` + `--write-support-claims` together | Conflicting modes; pick one. |
| Either flag without `--test-engine` | Verifier/condenser are per-engine. |
| More than one plugin loaded | RFC 0012 §13: multi-engine attribution is deferred to v2. |
| `GTEST_TOTAL_SHARDS > 1` or `GTEST_SHARD_INDEX` set | Sharding deferred to v2 (§9). |
| `--gtest_break_on_failure` set | Aborts before the verifier runs. |
| `--gtest_repeat > 1` | Duplicate records confuse the verifier. |
| Debug build (`NDEBUG` not defined) | Param strings may differ from release; auto-gen and verifier must agree byte-for-byte. |
| `--write-support-claims` with non-trivial `--gtest_filter` | Partial baseline would drop existing valid claims. |

For each refusal, the message names which flag/env to remove. They are
intentionally fatal: a "warn and continue" path would let a CI job
appear green while silently not enforcing.
