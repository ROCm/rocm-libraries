# Ops Review: <RFC title>

- **RFC**: `<relative path to RFC>`
- **Reviewer**: <reviewer name>
- **Date**: <YYYY-MM-DD>
- **Lens**: operational concerns (deep pass)
- **Verdict**: `approve` | `approve-with-comments` | `needs-revision` | `block`

<One-sentence justification of the verdict, framed in operational terms — e.g. "Design is sound but the perf claim on the hot path has no measurement plan and the new dependency has no license audit.">

---

## Blocking concerns

<Operational issues the reviewer would not approve without resolving. Omit this section entirely if none — do not write "none" as a placeholder.>

### <Short concern title>
**Where**: <section / line reference in the RFC>
**Concern**: <what is wrong operationally and why it matters in production>
**Suggested resolution**: <what would address it — a question, an option, or a direction>

---

## Non-blocking concerns and questions

Grouped by operational sub-area. Skip any sub-area with nothing to say (one line: "Sub-area — no concerns" or "Sub-area — N/A").

### Build system impact
<Dependencies, CMake target shape, build-time, binary size, toolchain, cross-platform, multi-arch.>
- ...

### Packaging & distribution
<Install layout, package metadata, SONAME, public headers, downstream consumer impact.>
- ...

### CI
<New jobs, runtime cost, platform/arch coverage, cache invalidation, parallelism, flakiness risk.>
- ...

### Observability
<For a library: error codes, log verbosity, env vars, version surfacing — what the user can put in a bug report. For a service: metrics, logs, traces, dashboards.>
- ...

### Performance in production
<Hot-path overhead, memory footprint, warm-up cost, tail latency vs throughput, claims-vs-benchmarks, regression surface.>
- ...

### Deployment & rollout mechanics
<Release vehicle, version pinning, default on/off, kill switch, dark launch, reversibility.>
- ...

### Capacity & scaling
<Often N/A for library code — one line is fine. Apply if the RFC adds a daemon, a growing cache, or a resource-shape change.>
- ...

### Failure modes in production
<What does broken look like, blast radius, recovery story, silent-wrong-answer risk.>
- ...

### Domain-specific
<Only if applicable — shim / plugin / GPU-specific / superbuild-specific items not covered above.>
- ...

---

## Strengths

<Brief — 3–5 bullets max. The author already knows what they got right; the value of the review is what they missed. Skip the section if the RFC is operationally thin and there's nothing notable to flag positively.>

- ...

---

## Recommended deeper passes

<List any operational areas that warrant their own follow-up beyond this review. Omit if none.>

- [ ] Benchmark plan review (perf claims need numbers before approval)
- [ ] Build-system / packaging deep-dive with the release engineering owner
- [ ] CI capacity / cost review with the infra owner
- [ ] Security threat model (`/rfc-review-security`) — if the new dep / new surface raises trust questions
- [ ] Compatibility / ABI deep-dive (`/rfc-review-compatibility`) — if SONAME or public-header changes are non-trivial
- [ ] Other: ...
