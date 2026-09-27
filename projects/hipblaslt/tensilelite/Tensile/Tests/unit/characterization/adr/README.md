# Architecture Decision Records (ADRs)

This directory holds the **Architecture Decision Records** for the TensileLite characterization suite: one short file per genuine decision, in [Michael Nygard's](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) format. An ADR captures *why* a decision was made, so a future reader — or a reviewer of a PR that touches the tests or goldens next to it — can recover the intent without archaeology.

See the characterization [`README.md`](../README.md) for the suite protocol and the snapshot/golden discipline; the content below covers only the ADR mechanics.

## When to write an ADR

Write one for a genuine decision fork, e.g.:
- **Pinning a known-wrong behavior** — assert the current crash/wrong result instead of "fixing" it in a characterization test (the suite is add-only). The ADR records the bug and links a filed defect that tracks the real fix.
- **Accepting a module below the coverage bar** for a structural reason (fork/IPC paths, GPU/asm emit, integration-only builders).
- **A departure from the add-only rule** (e.g. deleting verified-dead source).

Routine "wrote tests, hit the bar, committed" steps are **not** ADRs. Neither are accepted-equivalent mutants — a genuinely unkillable mutation survivor gets a one-line `# pragma: no mutate` plus a line in `DECISIONS.md`'s Mutation testing section, not a full ADR (see the characterization `README.md`'s ADR section for why).

The running catalog of pinned behaviors, coverage ceilings, and accepted mutants lives in [`../DECISIONS.md`](../DECISIONS.md) (the registry); an ADR is the per-decision rationale behind a catalog entry, and **only** the rationale — the catalog row itself should stay to a title, an `ADR:`/`Defect:` link, and a one-to-two sentence summary, not a restatement of the ADR's Context/Decision/Consequences.

## Format

- One file per decision, named `NNNN-short-slug.md` (zero-padded, monotonic) — e.g. `0001-pin-results-only-boolop-crash.md`.
- Nygard sections: **Status**, **Context**, **Decision**, **Consequences**. Add a `Defect:` line when the decision pins a behavior that looks wrong.
- **Status** is one of `Proposed`, `Accepted`, or `Superseded by adr/NNNN`.
- **Append-only.** Never rewrite an accepted ADR's Context, Decision, or Consequences prose. Two metadata fields are the narrow, expected exceptions: `Status` (flip to `Superseded by adr/NNNN` when a *new* ADR — the thing actually being appended — supersedes this one; this file's own prose is untouched) and `Defect` (fill in a tracker ID once one is filed, if it wasn't known when the ADR was written).
- **`Commit:` is a backfill-only field, not a template requirement.** An ADR written in the same PR as its decision doesn't need one — that PR's own commit(s) already are the provenance, and you can't know your own commit's SHA before it exists anyway. It exists only for retrofitted ADRs written after the fact (like ADRs 0004-0011, written well after their decisions), where the ADR's authoring commit and the decision's actual commit are two different things and that gap needs to be spelled out explicitly. When present, point it at the commit that made the real change (test/code), not the commit that wrote the ADR file — check with `git log -S` / `git blame` on `DECISIONS.md`, and if the repo squash-merges PRs, fetch the PR's pre-squash history (`git fetch origin refs/pull/<n>/head`) to find the real per-decision commit instead of citing the one shared squash commit for everything that PR touched.

## Template

```markdown
# ADR NNNN: <short title>

Status:  Accepted
Defect:  <TRACKER-ID, or "none — behavior is intended">
Commit:  <only for a retrofitted ADR — omit this line entirely otherwise>

## Context
<What forced a decision? The behavior observed, why it is ambiguous or
costly, and the constraints (e.g. add-only).>

## Decision
<What we decided to do, in one or two sentences.>

## Consequences
<What this costs and enables, and what a future change must do to revisit
it — e.g. "if the defect is fixed, flip the golden and supersede this ADR".>
```
