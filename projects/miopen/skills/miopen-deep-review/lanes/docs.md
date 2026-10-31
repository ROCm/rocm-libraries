# Lane: docs and comments (`DOCS`)

Docs and comments should be accurate, current and as short as they can be while complete.

When your packet lists owned files (docs-only change), you are the correctness owner for them and list each under Coverage.

## Accuracy

- Every statement matches the code at `$HEAD`: names, defaults, paths, commands, supported types and archs. Run or read what
  you can to confirm.
- Docs the change makes stale are updated or removed (`projects/miopen/CONTRIBUTING.md`, reviewer checklist item 4): the
  `projects/miopen/docs/` pages for find, the databases, the cache and environment variables, `projects/miopen/CHANGELOG.md`,
  and `projects/miopen/test/gtest/README.md` for test conventions.
- Existing public headers the change edits (`projects/miopen/include/miopen/miopen.h`) keep accurate Doxygen documentation.

## PR text (PR targets only)

The description explains why more than how, and every factual claim in it is checked against the diff. Read `pr/pr-text.txt` and
`facts.md` (the description-against-commits fact), and verify each remaining claim, recording the check under Coverage:

- Scope: the files, solvers, archs, data types, layouts and backends the text names match what the diff touches; nothing the diff
  changes is missing from the text (a changed default, a removed fallback, a db edit).
- Risk and compatibility statements (find-db and perf-db compatibility, ABI or public API impact, behavior change on existing
  configurations) match the code.
- Test results and counts (passed, skipped, architecture, build flags) match the evidence available: CI logs, the `RUN` lane's
  rows. A result the author states and nothing supports is `UNVERIFIED`, not accepted.
- Linked tickets, defects and PRs are described accurately. Existence and resolution of the links belong to `AUTH`: hand off.

A false or unsupported claim is `IMPORTANT` when a reader would rely on it to judge risk (kind `process`), else `SUGGESTION`.

## Concision

Concision, hedging, filler, marketing and table-versus-prose points are findings only when the bloat hides a requirement,
contradicts another document or makes an instruction ambiguous (`SUGGESTION`; tiers per [../policy.md](../policy.md) *Tiers*); otherwise they are checked, no finding. When one is a finding, propose a shorter version. Look for:

- Sentences that restate the code, the heading or the previous sentence; preamble, recaps and closing summaries.
- Hedging, filler and marketing ("simply", "robust", "powerful", "it is important to note").
- Prose that would be clearer as a table, list or example, and the reverse.
- Comments that explain what a line does when its name says it; keep comments that explain why. Comments describe the current
  state, never history.
- The same explanation in more than one file (keep one, link it); new documents that duplicate an existing one.
