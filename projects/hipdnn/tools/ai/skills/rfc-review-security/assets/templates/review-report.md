# Security Review: <RFC title>

- **RFC**: `<relative path to RFC>`
- **Reviewer**: <reviewer name>
- **Date**: <YYYY-MM-DD>
- **Verdict**: `approve` | `approve-with-comments` | `needs-revision` | `block`

<One-sentence justification of the verdict, in security terms.>

---

## Threat model summary

<This is the lens of the whole review. Lead with it.>

- **Stated in RFC?** Yes / No / Partial.
  <If No or Partial, the model below is `(inferred)` — and that itself is a finding worth raising in the report.>
- **Realistic attacker**: <who, with what capability, after what asset>
- **Deployment context**: <single-user workstation / shared host / multi-tenant cluster / CI runner / downstream OEM shipped binary / other>
- **In scope**: <threats this design must prevent or detect>
- **Out of scope**: <threats explicitly accepted, and why that is defensible>
- **Trust assumptions**: <what this design treats as trusted and on what basis>

---

## Blocking concerns

<Issues the reviewer would not approve without resolving. Omit this section entirely if none — do not write "none" as a placeholder.>

### <Short concern title>
**Where**: <section / line reference in the RFC>
**Concern**: <what is wrong and why it matters in the threat model above>
**Suggested resolution**: <a direction, an option, or a question — not a finished design>

---

## Non-blocking concerns and questions

Grouped by area. Skip any area with nothing to say — do not pad.

### Trust boundaries
- ...

### Attack surface
- ...

### Input validation & injection
- ...

### Cryptography
<Only if crypto is in scope. Otherwise: "N/A — no crypto in scope.">
- ...

### Authentication & authorization
<Only if there is a privilege boundary. Otherwise one line.>
- ...

### Secrets handling
- ...

### Supply chain
- ...

### Privacy
- ...

### Audit & forensics
<Only if relevant.>
- ...

---

## Strengths

<Brief — 3–5 bullets max. The author already knows what they got right; the value of the review is what they missed. Skip entirely if there is nothing notable rather than padding.>

- ...

---

## Recommended deeper passes

<Only if warranted by what this review found. Do not over-recommend.>

- [ ] Formal threat modeling session with the security team
- [ ] Dedicated cryptography review (if a non-standard construction is proposed)
- [ ] Fuzzing strategy for the new parser/deserializer surface
- [ ] Third-party penetration test before GA
- [ ] Supply-chain / SBOM review of new dependencies
- [ ] Other: ...
