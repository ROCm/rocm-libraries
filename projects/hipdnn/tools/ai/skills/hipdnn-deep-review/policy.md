# Review policy

## Precedence
On conflict, the first source wins: the repository's machine gate, the Libraries PR Bot
(`tools/libraries_pr_bot`); then the M-rules; then the rest of this file. No waiver
overrides the gate. A result on the merged tree (`DRIFT`) outranks the same result at `$HEAD`, because the merge is what lands.

## Tiers

| Tier | Use for |
|---|---|
| `BLOCKING` | Must fix before merge: a correctness or logic error, a security issue, a leak or crash in normal use, an ABI or API break without a migration path, missing tests for new functionality, an untested broad-blast-radius change to a default or shipping path, incomplete cleanup of code the change modifies (dead parameters, helpers, compatibility aliases for internal symbols) |
| `IMPORTANT` | Should fix: real behavioral risk, an unvalidated likely edge case, a meaningful test gap, missing required device or architecture coverage, a maintainability issue likely to cause defects soon |
| `SUGGESTION` | An improvement to code the change already touches |
| `FUTURE WORK` | Code the change does not touch, larger refactors, follow-ups, pre-existing problems. Does not block |

Never mark an unrelated improvement `BLOCKING` or soften incomplete cleanup.

## M-rules

| Rule | Requirement | Waiver |
|---|---|---|
| M1 | A defect fix includes a regression test that fails on the unpatched build and passes with the fix, or a tracked two-PR plan (a time-boxed, tracked expected-fail test PR, then the fix PR that removes it) | None without the plan |
| M2 | Product-code changes carry tests, a safe-default flag or a written waiver | Written reason |
| M3 | Never disable, skip or weaken a test only to green CI | None |
| M4 | A non-trivial PR carries work tracking (ticket, public issue or a credible no-tracker reason) | Written reason |
| M5 | The PR links its work item, the defect it fixes and related PRs, and the links resolve | Written reason |

Without a PR (a range or branch review), M4 and M5 are `N/A (no PR)`; the line notes that they are settled at PR time.
An unwaived M-rule `FAIL` is `BLOCKING`. An `UNVERIFIED` M-rule becomes a finding, and blocks closure only at `BLOCKING` or `IMPORTANT`
([SKILL.md](SKILL.md) *Completion*). A waiver is a code plus a one-to-two-sentence reason; a bare "N/A" is not one. Codes:
`W-DOC` (docs or comments only), `W-BUILD` (build or CI only), `W-REVERT` (the linked
reverted PR), `W-HOTFIX` (test deferred to a linked follow-up; needs a named approver),
`W-FLAG` (new behavior dark behind a safe-default flag, with a tracker), `W-NOTICKET`
(credible reason plus a commitment to link). Docs-only changes, named tool-driven
mechanical changes, reverts and one-line obviously correct edits need no tracker or new
tests; M3 always applies.

## Test smells

A hit on a test the change adds or edits is `BLOCKING`. For each sampled test, ask what one
change to the source would make it fail; no clear answer means coverage padding. Smells:

- It asserts only that a callable exists, a mock was called, or `is not None`; a `hasattr`
  guard lets the body skip silently.
- Copy-pasted tests that should be one parametrized test.
- A mock-only assertion passes against a no-op; it mocks the thing under test.
- It names a method absent from the source, or a non-obvious tolerance has no recorded reason.

Closer-read signals: many new test files for a small change; comments that narrate the
diff; reformat churn mixed into a functional change.

## Review rules

- A merge with the target's tip that fails to build, fails a selected test or breaks a contract the change depends on is
  `BLOCKING`. A conflict whose resolution needs an author decision is `IMPORTANT` (kind `process`). A conflict that resolves
  mechanically and passes is no finding. A failure that also occurs at `$HEAD` belongs to `RUN`.
- An undescribed public API change goes to **Needs human decision**, whatever its tier.
- A change that can affect TheRock (built code, tests, install, build, packaging or CI
  files) that TheRock CI neither builds nor tests (flag-guarded, skip-listed or
  test-excluded) is at least `IMPORTANT` unless the author reports verification. A change
  whose files are all documentation is exempt.
- A host output of a `target-specific` TheRock artifact (source list, compile definitions, host header, installed file or CTest
  entry) that varies with the GPU targets is `BLOCKING` when a two-arch configure confirms it and TheRock builds it: unguarded, or behind
  a flag that is ON at the TheRock pin or on TheRock main (the next pin bump builds it). It is `IMPORTANT` when only a static trace
  shows it or the flag is OFF at both. An install into a component the artifact does not request (hipkernelprovider `share/doc`) is
  `SUGGESTION`.
- A CTest entry whose projected per-shard time (CI baseline plus the change's measured delta) exceeds its tier timeout, or a job whose
  summed entry times plus the delta share exceed its `timeout_minutes`, is `BLOCKING` when presubmit runs the entry (run-all `quick`,
  scoped `standard`) on a presubmit family, a case behind a flag that main turns ON included; `IMPORTANT` with only nightly
  (`comprehensive`) exposure.
- A Windows path over 259 characters that a test or loader passes to a file API is `BLOCKING` when a required set provably fails to
  load, else `IMPORTANT`; a caller that uses a `\\?\` path is exempt.
- Documentation is prose that no tool, build, install, packaging or CI step reads:
  markdown, reStructuredText, plain-text docs, license files (`LICENSE`, `CONTRIBUTING`,
  `SECURITY`, `README`) and skill prompt text. Config files (`.gitignore`,
  `.clang-format`) are not.
- A required (architecture, suite) pair with no executed run is at least `IMPORTANT`. Green
  is not coverage: a suite counts only for cases that executed, on the engine and
  architecture under review. Skipped, filtered-out or other-engine runs are no evidence.
- Blast radius comes from what the diff changes, not its path: plumbing needs standard CI;
  a change to defaults or dispatch needs a multi-architecture run; an architecture-scoped
  change needs only those devices; an expanded support surface needs a full sweep. Do not
  over-escalate; "passing CI suffices" is valid.
- Read every CI run on the PR and compare with the target branch's latest: a lane green on
  the base and failing now is `BLOCKING`; a skipped lane, a masking retry or a missing
  required check is a finding.
- A branch-added leak is a `source-claim`, confirmed when the cited published line holds the
  value. It is `BLOCKING` for private details (credentials, machine or cluster names, code
  words, internal-only links or data, local workspace details) and `IMPORTANT` for content
  unfit for an open-source repository that is not private (planning chatter, process
  references).
