---
name: rfc-review
description: Review a technical RFC or design document across a structured set of perspectives (framing, design correctness, change management, operational concerns, sustainability) and produce a written review report. Use when reviewing RFCs, design docs, ADRs, or proposal documents — not for reviewing implementation PRs (use /review for that).
argument-hint: "[<path-or-pr>] [--focus <lens>] [--brief]"
allowed-tools: Read, Grep, Glob, Bash, WebFetch
---

# RFC Review Skill

Review a technical RFC across structured perspectives and produce a written review report. The aim is a coherent, opinionated review that catches the issues a careful reviewer would, not a generic checklist pass.

## Usage

```
/rfc-review                              # Find the RFC on the current branch and review it
/rfc-review docs/rfcs/my-rfc.md          # Review a specific RFC file
/rfc-review --focus compat               # Drill into one lens (see lens list below)
/rfc-review --brief                      # Top-line verdict + major concerns only
```

## When to invoke this skill

- The user asks for an RFC, design doc, ADR, or proposal review.
- The user is on a branch whose primary content is an RFC document (e.g. recent commit titled "RFC for ...").
- The user asks "what do you think of this design" about a document, not code.

Do NOT invoke for implementation PR review — use `/review` instead. RFCs propose, code implements; the lenses differ.

## Workflow

### Step 1 — Locate the RFC

In order of preference:
1. Argument path, if provided.
2. PR/branch context: run `git log -1 --name-only` and look for added/modified `.md` files under `docs/`, `rfcs/`, `design/`, or paths matching `*rfc*`.
3. If still ambiguous, ask the user for the path. Do not guess across multiple candidates.

### Step 2 — Read the RFC and its context

- Read the full RFC document. Do not skim.
- Follow links to referenced specs, prior RFCs, and code paths it claims to touch. If a path is named, confirm it exists with Glob/Grep.
- Note the author, status (draft/proposed/accepted), and any prior review comments visible on the PR.

### Step 3 — Walk the lenses

See `prompt.md` for the full per-lens checklist. The umbrella lens set:

1. **Framing** — Problem clarity, alternatives considered, non-goals.
2. **Design correctness** — Technical soundness, interface design, architectural fit.
3. **Change management** — Compatibility, migration/rollout, blast radius.
4. **Operational** — Performance, testing strategy, build/packaging/CI, observability, security.
5. **Sustainability** — Maintenance cost, ownership, documentation.
6. **RFC as document** — Internal consistency, named risks, stakeholder coverage.

For each lens, produce: **strengths**, **concerns** (graded blocking / non-blocking), and **questions for the author**. If a lens does not apply (e.g. no public API surface → skip API ergonomics), say so explicitly rather than padding.

### Step 4 — Ask the user how to deliver the report

Before writing, ask the user how they want the report delivered. Offer these options:

1. **On disk next to the RFC** — saved as `<rfc-dir>/review-<YYYY-MM-DD>-<reviewer>.md`. Best for sharing or attaching to a PR.
2. **Inline in chat only** — printed in the conversation, nothing written to disk. Best for quick iteration.
3. **Both** — written to disk and printed inline.

Skip this prompt only if the user already stated a preference in this turn (e.g. "just print it" or "save the review to ..."), in which case respect that and note the choice in your reply.

### Step 5 — Synthesize the report

Use `assets/templates/review-report.md` as the structure. Lead with the verdict and blocking concerns; put per-lens detail underneath so the author can skim or drill in.

If writing to disk, use the git user name (`git config user.name`) as `<reviewer>` and tell the user the path when done.

### Step 6 — Recommend follow-up reviews

If any lens warrants deeper specialized review, name it explicitly at the end of the report:

- Cross-version / ABI concerns → `/rfc-review-compatibility` (focused skill)
- Build, CI, packaging, observability → `/rfc-review-ops` (focused skill)
- Threat model, attack surface → `/rfc-review-security` (focused skill)

Recommend a focused pass only when the umbrella surfaced enough signal to warrant it — don't reflexively list all three on every review.

## Output rules

- **No padding.** A lens with nothing to say should be one line: "Framing — no concerns." Do not invent issues to fill a section.
- **Cite the RFC.** Every concern should quote or reference the specific section/line it pertains to, so the author can find it.
- **Separate blocking from non-blocking.** A blocking concern means "I would not approve without this resolved." Use the label sparingly.
- **Ask, don't assert.** When a section is ambiguous, ask a question rather than guess what the author meant and critique the guess.
- **Verdict at the top.** One of: `approve`, `approve-with-comments`, `needs-revision`, `block`. Justify in one sentence.

## Examples

```
/rfc-review
# → finds MIOpen-hipDNN shim RFC on current branch, produces full review

/rfc-review docs/rfcs/0042-new-allocator.md --brief
# → verdict + blocking concerns only, no per-lens detail

/rfc-review --focus compat
# → only the compatibility/migration lens, in depth
```
