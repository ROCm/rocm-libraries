# Architecture Decision Records (ADRs)

This directory holds the **Architecture Decision Records** for the TensileLite characterization suite: one short file per genuine decision, in [Michael Nygard's](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) format. An ADR captures *why* a decision was made, so a future reader — or a reviewer of a PR that touches the tests or goldens next to it — can recover the intent without archaeology.

See the characterization [`README.md`](../README.md) for the suite protocol and the snapshot/golden discipline; this file covers only the ADR mechanics.

## When to write an ADR

Write one for a genuine decision fork, e.g.:
- **Pinning a known-wrong behavior** — assert the current crash/wrong result instead of "fixing" it in a characterization test (the suite is add-only). The ADR records the bug and links a filed defect that tracks the real fix.
- **Accepting a module below the coverage bar** for a structural reason (fork/IPC paths, GPU/asm emit, integration-only builders).
- **A departure from the add-only rule** (e.g. deleting verified-dead source).
- **An accepted-equivalent-mutant policy call** (why a survivor is genuinely unkillable).

Routine "wrote tests, hit the bar, committed" steps are **not** ADRs. The running catalog of pinned behaviors, coverage ceilings, and accepted mutants lives in [`../DECISIONS.md`](../DECISIONS.md) (the registry); an ADR is the per-decision rationale behind a catalog entry.

## Format

- One file per decision, named `NNNN-short-slug.md` (zero-padded, monotonic) — e.g. `0001-pin-results-only-boolop-crash.md`.
- Nygard sections: **Status**, **Context**, **Decision**, **Consequences**. Add a `Defect:` line when the decision pins a behavior that looks wrong.
- **Status** is one of `Proposed`, `Accepted`, or `Superseded by adr/NNNN`.
- **Append-only.** Never rewrite an accepted ADR's decision. If the world changes (e.g. a pinned bug is fixed and its golden flips), write a *new* ADR that supersedes the old one and set the old one's status to `Superseded by adr/NNNN`.

## Template

```markdown
# ADR NNNN: <short title>

Status:  Accepted
Defect:  <TRACKER-ID, or "none — behavior is intended">

## Context
<What forced a decision? The behavior observed, why it is ambiguous or
costly, and the constraints (e.g. add-only).>

## Decision
<What we decided to do, in one or two sentences.>

## Consequences
<What this costs and enables, and what a future change must do to revisit
it — e.g. "if the defect is fixed, flip the golden and supersede this ADR".>
```
