# Lane: design (`DESIGN`)

Judge the contract the change adds or alters, whether it sits in the right layer with the right owner, and the extensions it
forecloses. Defects, public-surface classification and duplication belong to other lanes (see **Handoffs**).

## Contract and history

Read these before judging; cite what you rely on.

- Related open PRs: the stack parent and children, and up to 10 other open PRs that touch the same files or symbols
  (`gh pr list -R <owner>/<repo> --state open --search "<path or symbol>" --json number,title,url`). Read their descriptions
  for contracts the change must keep or can now conflict with.
- Scoped history: `git -C "$CLONE" log --format='%h %s' -n 20 "$BASE" -- <changed path>` for the paths the change edits; read the
  commit bodies that explain why the code is shaped as it is.
- Discussions on merged PRs that introduced or changed that code (`gh pr view <n> --comments`, found from the commit subjects).
  An RFC, decision or discussion that makes the design intentional ends the finding.

## Checks

- Obligations: what the change now requires of implementers (a provider, a plugin, a test) or callers. Does every existing one
  meet it, and does the contract say so where they will read it?
- Layer and owner: the code depends only on layers below it and sits with the component that owns the concept; a caller does
  not work around a wrong-layer dependency.
- Defaults: each default is deliberate and does not imply support that does not exist.
- Minimal surface: the change exposes no more than its callers need; nothing is public only for one internal consumer.
- Policy out of mechanical code: a decision (selection, fallback, tolerance, naming of a mode) lives in the layer that owns the
  policy, not in code that only moves or converts data.
- Existing patterns: the change follows how the surrounding code solves the same problem, or says why it departs.
- Unsupported paths: an input the change does not handle is rejected explicitly, not left to fail further away.

## Extensibility

Report a likely extension the change makes harder or impossible only with evidence that the extension is likely: a linked
issue or RFC, an open PR, a documented roadmap or a pattern the codebase already repeats. Without evidence it is not a finding.
File it as `FUTURE WORK`.

## Filing

Every finding has Kind `design` and a tier per [../policy.md](../policy.md) *Design findings*: `IMPORTANT` only for a cost the
change already imposes, shown by the cited contract, history or discussion; `SUGGESTION` otherwise; a foreclosed extension is always `FUTURE WORK`; never `BLOCKING`. A concern
that must block a merge is a defect: hand it to `CORR`. Name the caller or implementer who pays and what they must do.

For each `IMPORTANT` design finding, add a **Needs human decision** entry
([../findings.md](../findings.md) *Lane report*): the question, the options (fix, or accept the risk) and your recommendation.

## Handoffs

- A wrong result, crash or leak: `CORR: <location>`.
- A public surface changed or undescribed: `API: <declaration>`.
- Code the change repeats or could reuse: `DUP: <locations>`.
