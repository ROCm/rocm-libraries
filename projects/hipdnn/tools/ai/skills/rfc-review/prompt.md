# RFC Review — Detailed Lens Checklist

You are reviewing a technical RFC. Your job is to give the author the review a thoughtful, experienced engineer would give: catch the real issues, name the unstated assumptions, and ask the questions that surface hidden weaknesses. Avoid generic checklist energy.

## Reviewer stance

- Read the RFC end-to-end before forming opinions. First impressions from the title or intro are often wrong.
- Assume the author is competent and the design has reasons you haven't spotted yet. Ask before critiquing.
- A short, sharp review beats a long, hedged one. If you have nothing to say about a lens, say "no concerns" and move on.
- Distinguish what the RFC *says* from what it *implies*. Implied behavior is where bugs hide.
- Calibrate severity honestly. If everything is "critical," nothing is.

## The lenses

For each lens below, produce three things in the report:
- **Strengths** — what the RFC does well on this axis (skip if nothing notable).
- **Concerns** — issues, graded as `blocking` or `non-blocking`.
- **Questions** — clarifications you'd want from the author before signing off.

---

### 1. Framing

**What to look for**
- Is the problem stated in terms the audience can verify, not just asserted? ("Users complain about X" is weak; "X causes Y measurable regression in Z workflow" is strong.)
- Are the affected users/systems named? Is the cost of *not* solving this articulated?
- Are alternatives considered, or only the chosen design presented as if inevitable?
- Are non-goals stated? Absence of non-goals usually means scope will creep in review.

**Common failure modes**
- Solution-shaped problem statement (problem framed to fit the chosen solution).
- "Do nothing" is missing from alternatives — often it's the right answer.
- Conflating two problems that should be separate RFCs.

**Questions to ask**
- Who has this problem today, and how do they work around it?
- What changes if we ship the design vs. ship nothing?
- Why was [alternative X] rejected?

---

### 2. Design correctness

**What to look for**
- Walk the design end-to-end on a concrete example. Does it actually work?
- Failure modes: what happens on partial failure, on retries, on concurrent access, on version skew?
- Edge cases: zero, one, many; empty input; max size; non-ASCII; cycles.
- Interface design: naming consistent with surroundings; error model coherent; nothing hard to use correctly or easy to misuse.
- Architectural fit: does it work *with* the system's grain, or fight it?

**Common failure modes**
- Happy path described in detail, failure path waved away.
- New abstraction introduced where an existing one would do.
- Error handling delegated to "the caller" without specifying contract.
- Names that overlap or conflict with existing concepts in the codebase.

**Questions to ask**
- What happens if [step N] fails partway through?
- How does this interact with [existing system X]?
- Why introduce [new concept Y] instead of extending [existing concept Z]?

---

### 3. Change management

**What to look for**
- **Compatibility**: ABI, API, behavior, on-disk/wire format. What breaks? For whom?
- **Migration**: How do existing users get from current state to proposed state? Is there a transition period? Tooling?
- **Rollout**: Feature flag? Staged? Reversible if it goes wrong?
- **Blast radius**: Who is affected intentionally, and who is affected incidentally?

**Common failure modes**
- "We'll deprecate X" with no timeline, no migration tooling, no consumer list.
- Silent behavior change masquerading as a refactor.
- Version skew handling ignored (client old + server new, or vice versa).
- No rollback plan; "we'll fix forward" treated as a strategy.

**Questions to ask**
- How does a downstream consumer at version N-1 behave when this lands?
- If we ship this and discover it's wrong in production, how do we back out?
- Who currently depends on the behavior being changed? Have they been consulted?

---

### 4. Operational concerns

**What to look for**
- **Performance**: Hot-path cost. Memory footprint. Are claims backed by benchmarks or only by intuition?
- **Testing strategy**: What proves it works? What's the regression surface? Are integration tests proposed, or only unit?
- **Build / packaging / CI**: New dependencies, new build targets, new platforms? Impact on build time and binary size?
- **Observability**: Can you tell when it's broken in production? Are metrics/logs proposed?
- **Security**: Threat model. New attack surface. Trust boundaries crossed.

**Common failure modes**
- Performance claims with no numbers ("this will be fast").
- Tests proposed but no statement of what coverage means done.
- A dependency added with no discussion of supply-chain or licensing impact.
- Security treated as a separate concern to be addressed "later."

**Questions to ask**
- What's the expected overhead on [hot path]? How will we measure it?
- What test would fail if [key property] were violated?
- How would oncall notice this is misbehaving?

---

### 5. Sustainability

**What to look for**
- Long-term maintenance cost. Does this lock the project into supporting something complex forever?
- Ownership: who is the on-call for this code? Is that owner consulted and bought in?
- Documentation: will downstream users be able to adopt this without reading the code?
- Complexity budget: is the added complexity proportional to the value delivered?

**Common failure modes**
- New subsystem with no named owner.
- "We'll document it later" — almost always means never.
- A flexible/configurable design that pushes the hard choices onto users.

**Questions to ask**
- Who owns this code five years from now?
- What documentation lands with the implementation, not after?
- If this is configurable, what's the recommended default and why?

---

### 6. RFC as document

**What to look for**
- Internal consistency: do the stated goals, the design, and the migration plan agree with each other?
- Named risks: are unknowns acknowledged or hand-waved? An RFC with no risks section is suspicious.
- Stakeholder coverage: are the affected teams listed and have they reviewed?
- Open questions section: is there one, and does it reflect the actually unresolved decisions?

**Common failure modes**
- Goals say "minimal change" but design proposes a new subsystem.
- Risks section is boilerplate ("performance, security, compatibility") with no specifics.
- No stakeholder list, or a list that omits the team that owns the affected code.

**Questions to ask**
- The goals say X but the design does Y — which is the binding constraint?
- What's the riskiest assumption in this design? What would convince you it's wrong?
- Has [obviously affected team] reviewed this?

---

## Domain-specific lenses (apply when relevant)

These are not a standard part of every review — apply them when the RFC's nature warrants:

- **Shim / compatibility layers**: symbol/namespace pollution, version-matrix coverage, indirection overhead, behavior under partial backend support.
- **Plugin / extension systems**: ABI stability of the plugin interface, sandboxing, plugin discovery and loading order, versioning across plugin and host.
- **Build / packaging changes**: impact on superbuild, cross-platform support, third-party CI.
- **GPU / kernel work**: precision, determinism, occupancy, register pressure, multi-arch coverage.

## Producing the report

Use `assets/templates/review-report.md`. Fill sections in this order:

1. Verdict + one-sentence justification.
2. Blocking concerns (if any) — these are the bar for approval.
3. Non-blocking concerns and questions, grouped by lens.
4. Strengths (brief — the author knows what they got right; the value of the review is what they missed).
5. Recommended follow-up reviews.

**Delivery**: ask the user (via `AskUserQuestion`) whether to write the report on disk next to the RFC, print inline only, or both — unless they already stated a preference this turn. If writing to disk, save as `<rfc-dir>/review-<YYYY-MM-DD>-<reviewer>.md` and tell the user the path when done.
