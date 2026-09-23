---
name: rfc-review-ops
description: Deep operational-concerns review of a technical RFC — build system impact, packaging, CI, observability, performance-in-production, deployment, capacity, and failure modes. Use as a focused follow-up to /rfc-review when the operational lens deserves more than the umbrella's one-paragraph pass, or standalone when the RFC is primarily ops-shaped (new dependency, build/CI change, perf-sensitive subsystem). Not a substitute for /review on implementation PRs.
argument-hint: "[<path-or-pr>]"
allowed-tools: Read, Grep, Glob, Bash, WebFetch
---

# RFC Review — Operational Concerns (Focused)

A deep pass on a single RFC focused entirely on what matters once the code is built, shipped, and running. The umbrella `/rfc-review` covers operational concerns in one paragraph; this skill goes the rest of the way.

For a C++ library like hipDNN / MIOpen, "production" means *downstream applications' production*. There is no dashboard. Debugging happens through user-submitted bug reports. The questions are: when this breaks, will the user know it broke? Will their bug report contain enough information to act on? Will the maintainer who reads that report be able to reproduce, diagnose, and fix without a multi-week back-and-forth?

## Usage

```
/rfc-review-ops                              # Find the RFC on the current branch and run the ops pass
/rfc-review-ops docs/rfcs/my-rfc.md          # Review a specific RFC file
/rfc-review-ops https://github.com/.../pull/123   # Review the RFC in a PR
```

## When to invoke this skill

- `/rfc-review` recommended a follow-up ops pass.
- The RFC is operationally heavy on its face: new third-party dependency, new build target or toolchain requirement, new CI job, new optional runtime feature with a perf claim, change to packaging or install layout, change to logging/error-surfacing.
- The user asks "what's the build/CI/perf/oncall cost of this design."
- The user is reviewing a shim layer, a plugin system, or anything that ships in downstream binaries.

Do NOT invoke for general RFC review — use `/rfc-review` first. Do NOT use for implementation PR review — use `/review`.

This skill can be invoked standalone OR as a follow-up from `/rfc-review`. If invoked as a follow-up, assume the user has already seen the umbrella verdict and does not need it repeated — go deep on ops only.

## Workflow

### Step 1 — Locate the RFC

In order of preference:
1. Argument path or PR URL, if provided.
2. PR/branch context: `git log -1 --name-only` and look for added/modified `.md` files under `docs/`, `rfcs/`, `design/`, or paths matching `*rfc*`.
3. If still ambiguous, ask the user for the path. Do not guess across candidates.

### Step 2 — Read the RFC and gather operational context

- Read the full RFC.
- Identify what the RFC *touches* operationally. Use Glob/Grep to confirm:
  - CMake targets it modifies or adds (`CMakeLists.txt`, `*.cmake`, presets).
  - CI configs it would affect (`.github/workflows/`, `.gitlab-ci.yml`, Jenkinsfiles).
  - Packaging metadata (`*.spec`, `*.deb`, `CPackConfig*`, install rules).
  - Logging / env-var conventions in the affected area.
- If the RFC names a new dependency, look it up: size, build-time cost, license, platform coverage, release cadence. WebFetch the project page if needed.
- For ROCm/GPU work: check what arch list the affected subsystem currently builds for (`AMDGPU_TARGETS`, `GPU_TARGETS`) and whether the RFC expands or narrows it.

### Step 3 — Walk the operational sub-areas

See `prompt.md` for the full checklist. The sub-areas:

1. **Build system impact** — dependencies, CMake target shape, build time, binary size, toolchain.
2. **Packaging & distribution** — artifact shape, install layout, downstream install commands.
3. **CI** — new jobs, runtime, platform coverage, flakiness, cache invalidation.
4. **Observability** — what oncall sees when this breaks (for a library: what the user can put in a bug report).
5. **Performance in production** — hot-path overhead, memory, warm-up, tail latency, claims vs benchmarks.
6. **Deployment & rollout mechanics** — release vehicle, version pinning, gradual rollout, kill switches.
7. **Capacity & scaling** — added load, resource shape changes, upper bounds.
8. **Failure modes in production** — what does broken look like, blast radius, recovery.

For each sub-area, produce **strengths** (only if notable), **concerns** (graded `blocking` / `non-blocking`), and **questions for the author**. A sub-area with nothing to say should be one line: "Capacity — N/A (library code, no capacity model)." Do not invent filler.

### Step 4 — Ask the user how to deliver the report

Before writing, ask the user how they want the report delivered. Offer these options:

1. **On disk next to the RFC** — saved as `<rfc-dir>/review-ops-<YYYY-MM-DD>-<reviewer>.md`.
2. **Inline in chat only** — printed in the conversation, nothing written to disk.
3. **Both** — written to disk and printed inline.

Skip this prompt only if the user already stated a preference in the current turn (e.g. "just print it" or "save it next to the RFC"). If invoked as a follow-up from `/rfc-review` and the user already answered the same question for the umbrella, ask again — disk paths differ (`review-ops-...md` vs `review-...md`).

### Step 5 — Synthesize the report

Use `assets/templates/review-report.md` as the structure. Lead with the verdict and blocking concerns; the per-sub-area detail goes underneath.

If writing to disk, use `git config user.name` as `<reviewer>` and tell the user the path when done.

## Output rules

- **No padding.** A sub-area with nothing to say is one line. The whole report can be short if the RFC has few operational implications — that's a feature, not a failure.
- **Cite the RFC.** Every concern quotes or references the specific section/line, so the author can find it.
- **Separate blocking from non-blocking.** A blocking concern means "I would not approve without this resolved." Use sparingly. Examples of legitimate blockers: a new mandatory third-party dep with no license check, a perf claim with no measurement plan on a hot path, a behavior change with no user-visible diagnostic when it fails.
- **Ask, don't assert.** When a section is ambiguous about ops behavior, ask. Don't critique a guess.
- **Verdict at the top.** One of: `approve`, `approve-with-comments`, `needs-revision`, `block`. Justify in one sentence.
- **Engineering judgment, not SRE platitudes.** "Add metrics" is not advice. "Expose a `HIPDNN_LOG_LEVEL` env var so users can capture trace output in bug reports, since there's no in-process metrics infra" is.

## Examples

```
/rfc-review-ops
# → finds RFC on current branch, deep ops pass

/rfc-review-ops docs/rfcs/0042-new-allocator.md
# → ops-only review of a perf-sensitive RFC

/rfc-review-ops --focus build
# (not supported — this skill is already focused; use a specific question in chat instead)
```
