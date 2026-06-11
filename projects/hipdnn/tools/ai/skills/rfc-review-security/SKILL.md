---
name: rfc-review-security
description: Deep-dive security review of a technical RFC or design document — threat model, trust boundaries, attack surface, input validation, crypto, authN/authZ, secrets, supply chain, privacy, and audit. Use when reviewing an RFC and security warrants its own focused pass, or as a follow-up from /rfc-review. NOT for reviewing implementation diffs — use /security-review for that.
argument-hint: "[<path-or-pr>]"
allowed-tools: Read, Grep, Glob, Bash, WebFetch
---

# RFC Security Review Skill

Deep-dive security review of an RFC. This is the focused, opinionated sibling of the umbrella `/rfc-review` skill — same locate/read/walk/deliver workflow, but the entire review is the security lens. Aim for the review a security-minded engineer with codebase context would give: catch the realistic threats, name the unstated trust assumptions, and ask the questions that surface hidden exposure.

This skill reviews **design proposals**. For review of pending code changes on the current branch, use `/security-review` instead.

## Usage

```
/rfc-review-security                              # Find the RFC on the current branch and security-review it
/rfc-review-security docs/rfcs/my-rfc.md          # Security-review a specific RFC file
```

## When to invoke this skill

- The user explicitly asks for a security review of an RFC, design doc, ADR, or proposal.
- `/rfc-review` flagged security as needing a deeper pass and the user wants it now.
- The RFC introduces any of: a new parser/deserializer, a new network or IPC endpoint, a new file-load path, a loadable plugin, a new dependency, a new daemon/service, a credential/secret, a change to a privilege boundary, telemetry that leaves the host.

Do NOT invoke for:
- Review of implementation diffs on the current branch → `/security-review`.
- General RFC review where security is one of many concerns → `/rfc-review` (it covers security at one-paragraph depth).

## Workflow

### Step 1 — Locate the RFC

In order of preference:
1. Argument path, if provided.
2. PR/branch context: `git log -1 --name-only` and look for added/modified `.md` files under `docs/`, `rfcs/`, `design/`, or paths matching `*rfc*`.
3. If still ambiguous, ask the user. Do not guess across multiple candidates.

### Step 2 — Read the RFC and its context

- Read the full RFC end-to-end before forming opinions.
- Follow links to referenced specs, prior RFCs, and code paths it claims to touch. If a path is named, confirm it exists with Glob/Grep — a "security boundary" in code that doesn't exist is a red flag.
- Note the author, status, and any prior security comments visible on the PR.
- Identify the domain. The realistic threat model differs sharply between, say, a numerical GPU library (most-likely attacker = a malicious model file or a co-tenant on shared hardware) and a network service (most-likely attacker = anyone on the internet). Calibrate accordingly — see the domain hooks in `prompt.md`.

### Step 3 — Walk the security checklist

See `prompt.md` for the full per-area checklist. The areas:

1. **Threat model** — who is the realistic attacker, what asset, what capability. Most important section; an RFC with no threat model is the most common defect.
2. **Trust boundaries** — where untrusted input becomes trusted; process / network / user / privilege boundaries crossed.
3. **Attack surface** — new inputs, endpoints, parsers, deserializers, file reads, IPC, env-var consumers.
4. **Input validation & injection** — every parser is a vuln; every concatenation into a command/query/path is an injection vector.
5. **Cryptography** — algorithm choice, key management, RNG sources, rolling-your-own.
6. **Authentication & authorization** — who can do what, how it is enforced, failure modes.
7. **Secrets handling** — how secrets enter, are stored, are logged, are rotated.
8. **Supply chain** — new direct and transitive deps, build provenance, update policy, binary trust.
9. **Privacy** — what data flows where, what is retained, what leaks via logs / telemetry / errors.
10. **Audit & forensics** — can you tell after the fact who did what.

For each area, produce: **strengths**, **concerns** (graded `blocking` vs `non-blocking`), and **questions for the author**. A sub-area with nothing to say is one line, not invented filler.

### Step 4 — Ask the user how to deliver the report

Before writing, ask the user how they want the report delivered. Offer these options:

1. **On disk next to the RFC** — saved as `<rfc-dir>/review-security-<YYYY-MM-DD>-<reviewer>.md`. Best for sharing or attaching to a PR.
2. **Inline in chat only** — printed in the conversation, nothing written to disk. Best for quick iteration.
3. **Both** — written to disk and printed inline.

Skip this prompt only if the user already stated a preference in this turn (e.g. "just print it" or "save the review to ..."), in which case respect that and note the choice in your reply.

### Step 5 — Synthesize the report

Use `assets/templates/review-report.md` as the structure. Order:

1. **Verdict** + one-sentence justification.
2. **Threat model summary** at the top — this is the lens of the whole review; the rest of the report makes sense only in its context.
3. **Blocking concerns** (if any).
4. Per-area concerns and questions.
5. Strengths.
6. Recommended deeper passes.

If writing to disk, use the git user name (`git config user.name`) as `<reviewer>` and tell the user the path when done.

### Step 6 — Recommend deeper passes

If you find concerns that need expertise beyond an RFC review — formal threat modeling session, dedicated crypto review, fuzzing strategy, third-party pen test — name them at the end. Don't over-recommend; only call them out if the RFC actually justifies it.

## Output rules

- **Verdict at the top.** One of: `approve`, `approve-with-comments`, `needs-revision`, `block`.
- **Threat model first.** A security review without a stated threat model is just vibes. If the RFC doesn't state one, infer the most realistic one yourself and label it as inferred — then judge the design against it.
- **No padding.** A sub-area with nothing to say is one line: "Crypto — N/A, no crypto in scope." Do not invent threats. Resist "what if the attacker had quantum capability" energy.
- **Cite the RFC.** Every concern references the specific section or line.
- **Severity, calibrated.** `blocking` = "I would not approve without this resolved." Use sparingly. If everything is blocking, nothing is.
- **Ask, don't assert.** When a section is ambiguous on a security property, ask. Don't critique a guess.
- **Domain-aware.** A GPU numerical library has a different realistic threat surface than a web service. Spend your words where the real risk is, not where a generic OWASP checklist would tell you to.

## Examples

```
/rfc-review-security
# → finds RFC on current branch, produces a security-focused review

/rfc-review-security docs/rfcs/0042-plugin-loader.md
# → deep security review of plugin-loader RFC (plugin trust model, ABI, supply chain)
```
