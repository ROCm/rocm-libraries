---
name: rfc-review-compatibility
description: Deep-dive review of a technical RFC focused on change-management concerns — API/ABI/wire/on-disk compatibility, version skew, migration tooling, rollout mechanics, reversibility, and consumer impact. Produces a written review report. Use when an RFC touches a stable surface (public API, plugin ABI, on-disk or wire format), or as a follow-up pass after `/rfc-review` flags compatibility as a deeper concern.
argument-hint: "[<path-or-pr>]"
allowed-tools: Read, Grep, Glob, Bash, WebFetch
---

# RFC Review — Compatibility & Change-Management Deep Dive

Focused lens on a single RFC. Where the umbrella `/rfc-review` gives change management one paragraph among five, this skill spends the entire pass on it. A reader of the report should be able to answer:

- If we ship this and an N-1 consumer hits the new code, what happens?
- If we ship it and discover it's wrong tomorrow, what is the rollback?
- Who depends on the surface we're changing, and have they been consulted?

## Usage

```
/rfc-review-compatibility                       # Find the RFC on the current branch and review it
/rfc-review-compatibility docs/rfcs/my-rfc.md   # Review a specific RFC file
```

Can be invoked standalone, or as a follow-up when `/rfc-review` recommends a deeper compat pass.

## When to invoke this skill

- The RFC proposes a change to a stable surface: public API headers, plugin ABI, exported symbols, wire protocol, on-disk format, configuration schema.
- The RFC introduces or modifies a shim, compatibility layer, or version-bridging code.
- The RFC describes a deprecation, replacement, or migration of an existing feature.
- The umbrella `/rfc-review` ran and flagged compatibility as the area needing a deeper pass.

Do NOT invoke for greenfield code with no existing consumers, or for pure internal refactors with no surface exposed to downstreams — the lens has nothing to bite on.

## Workflow

### Step 1 — Locate the RFC

In order of preference:
1. Argument path, if provided.
2. PR/branch context: `git log -1 --name-only` and look for added/modified `.md` files under `docs/`, `rfcs/`, `design/`, or paths matching `*rfc*`.
3. If still ambiguous, ask the user for the path. Do not guess across multiple candidates.

### Step 2 — Read the RFC and surrounding surface

- Read the full RFC end-to-end. Do not skim — compat issues hide in the parts that look like boilerplate.
- Identify the **surface(s) being changed**: which headers, which exported symbols, which plugin entry points, which formats. Use Grep/Glob to confirm what currently exists.
- Identify **the current consumers** of that surface where possible. For a shim, that means: who calls the old API today? For a plugin interface, who ships plugins against it?
- Note the RFC's own claims about compatibility, migration, rollout, and reversibility — these are what you are evaluating.

### Step 3 — Walk the focused checklist

See `prompt.md` for the full per-area checklist. The sub-areas:

1. **Compatibility kinds** — API, ABI, behavior, wire/on-disk format.
2. **Version skew** — client/server, plugin/host, build-time vs runtime.
3. **Migration** — tooling, deprecation timeline, communication.
4. **Rollout** — feature flags, staged rollout, dark-launch.
5. **Reversibility** — rollback story, forward-compat for the rollback.
6. **Consumer/dependency impact** — who is affected, have they been consulted.
7. **Domain hooks** — C++ ABI, plugin interface stability, CUDA/HIP/cuDNN-style API evolution.

For each sub-area produce: **strengths** (skip if none), **concerns** (graded `blocking` / `non-blocking`), **questions for the author**. A sub-area with nothing to say is one line — do not pad.

### Step 4 — Ask the user how to deliver the report

Before writing, ask the user how they want the report delivered. Offer these options:

1. **On disk next to the RFC** — saved as `<rfc-dir>/review-compat-<YYYY-MM-DD>-<reviewer>.md`.
2. **Inline in chat only** — printed in the conversation, nothing written to disk.
3. **Both** — written to disk and printed inline.

Skip this prompt only if the user already stated a preference in the current turn (e.g. "just print it" or "save the review to ..."), in which case respect that and note the choice in your reply.

### Step 5 — Synthesize the report

Use `assets/templates/review-report.md` as the structure. Lead with the verdict and blocking concerns; put per-area detail underneath.

If writing to disk, use the git user name (`git config user.name`) as `<reviewer>` and tell the user the path when done.

### Step 6 — Recommend deeper passes

If the compat review surfaces concerns that need a different specialist (e.g. operational/build concerns, security/threat model), name them at the end of the report. This skill does not try to be the umbrella — it owns one lens deeply.

## Output rules

- **No padding.** A sub-area with nothing to say is one line: "Reversibility — no concerns." Do not invent issues.
- **Cite the RFC.** Every concern references the specific section/line, so the author can find it.
- **Separate blocking from non-blocking.** Blocking means "I would not approve without this resolved." Use sparingly.
- **Ask, don't assert.** When a section is ambiguous, ask a question rather than critique a guess.
- **Verdict at the top.** One of: `approve`, `approve-with-comments`, `needs-revision`, `block`. One-sentence justification.
- **Be specific to the surface.** "ABI concerns" is not a finding; "adding a virtual method to `cudnnHandle` breaks vtable layout for existing plugins" is.

## Examples

```
/rfc-review-compatibility
# → finds the RFC on the current branch, runs the compat deep dive

/rfc-review-compatibility docs/rfcs/0042-plugin-v2.md
# → reviews a specific RFC with the compat lens
```
