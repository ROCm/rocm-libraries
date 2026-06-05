# Compatibility Review: <RFC title>

- **RFC**: `<relative path to RFC>`
- **Reviewer**: <reviewer name>
- **Date**: <YYYY-MM-DD>
- **Lens**: change management (compatibility, version skew, migration, rollout, reversibility, consumer impact)
- **Verdict**: `approve` | `approve-with-comments` | `needs-revision` | `block`

<One-sentence justification of the verdict.>

---

## Blocking concerns

<Issues the reviewer would not approve without resolving. Omit this section entirely if none — do not write "none" as a placeholder.>

### <Short concern title>
**Where**: <section / line reference in the RFC>
**Concern**: <what is wrong and why it matters — be specific to the surface, not generic>
**Suggested resolution**: <a question, an option, or a direction>

---

## Per-area findings

For each sub-area: strengths (skip if none), concerns (graded `blocking` / `non-blocking`, with RFC citations), questions for the author. Sub-areas with nothing to say get one line — do not pad.

### Compatibility kinds
<API source / ABI binary / behavioral / wire-or-on-disk. State which flavors the RFC claims and whether the proposed change actually preserves them.>

### Version skew
<Walk the matrix: {old, new} consumer × {old, new} provider; plugin built against host vN running on host vN±1; build-time vs runtime version.>

### Migration
<Tooling, deprecation timeline, coexistence window, communication, discoverability.>

### Rollout
<Feature flag presence and default, staged rollout plan, dark-launch option, instrumentation to detect a bad rollout.>

### Reversibility
<Rollback procedure, forward-compat of any new state for the rolled-back version, down-migration for data changes.>

### Consumer / dependency impact
<Named consumer list, consultation evidence, soname/major-version bump, packaging, toolchain/dependency requirement bumps.>

### Domain hooks
<Apply only what is relevant. Skip the rest with one line.>
- **C++ ABI** — mangling, vtables, struct layout, inline functions, default arguments, templates in public headers.
- **Plugin interface** — C ABI boundary, version negotiation, discovery/load order, fallback, symbol visibility.
- **CUDA/HIP/cuDNN-style API evolution** — opaque handles, versioned structs with size field, opt-in flags, `_v2` suffix pattern.
- **Build-time vs runtime version** — header version is what was used; library version is what exists; failure modes when they disagree.

---

## Strengths

<Brief — 3–5 bullets max. The value of the review is what the author missed, not what they got right.>

- ...

---

## Recommended deeper passes

<Compatibility issues sometimes point at concerns this lens doesn't own. Name them here if so.>

- [ ] Operational / build / observability deep-dive (`/rfc-review-ops`)
- [ ] Security threat model (`/rfc-review-security`)
- [ ] Return to umbrella `/rfc-review` if other lenses (framing, sustainability) weren't yet covered
- [ ] Other: ...
