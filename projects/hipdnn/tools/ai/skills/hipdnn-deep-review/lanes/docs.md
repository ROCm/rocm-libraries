# Lane: docs and comments (`DOCS`)

Docs and comments should be accurate, current and as short as they can be while complete.

When your packet lists owned files (docs-only change), you are the correctness owner for them and list each under Coverage.

## Accuracy

- Every statement matches the code at `$HEAD`: names, defaults, paths, commands, supported types and archs. Run or read what
  you can to confirm.
- Docs the change makes stale are updated or removed (`projects/hipdnn/CONTRIBUTING.md`).
- Existing public headers the change edits keep accurate documentation per `projects/hipdnn/docs/doxygen-guidelines.md`.

## Concision

Concision, hedging, filler, marketing and table-versus-prose points are findings only when the bloat hides a requirement,
contradicts another document or makes an instruction ambiguous (`SUGGESTION`; tiers per [../policy.md](../policy.md) *Tiers*); otherwise they are checked, no finding. When one is a finding, propose a shorter version. Look for:

- Sentences that restate the code, the heading or the previous sentence; preamble, recaps and closing summaries.
- Hedging, filler and marketing ("simply", "robust", "powerful", "it is important to note").
- Prose that would be clearer as a table, list or example, and the reverse.
- Comments that explain what a line does when its name says it; keep comments that explain why. Comments describe the current
  state, never history.
- The same explanation in more than one file (keep one, link it); new documents that duplicate an existing one.
