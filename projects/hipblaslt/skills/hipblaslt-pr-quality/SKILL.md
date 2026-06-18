---
name: hipblaslt-pr-quality
description: "hipBLASLt supplements to the ROCm PR quality base skill. Use for hipBLASLt PR author, review, or pre-merge gating (target branch develop; product paths under projects/hipblaslt/**, including tensilelite/). Adds and tightens base rules; never relaxes a base MUST."
argument-hint: "[author | review | pre-merge] [PR URL | branch:<name> | local]"
extends: rocm-pr-quality
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# hipBLASLt PR Quality (overlay)

## Dependency (mandatory — do this first)

Read and apply the `rocm-pr-quality` base skill before anything below. It lives in `ROCm/TheRock`
at `skills/rocm-pr-quality/` (`SKILL.md` + `reference.md`). Since `rocm-libraries` is a submodule of
TheRock, a normal TheRock checkout already has the base present alongside this overlay.

The supplements here only **ADD** rules or **TIGHTEN** thresholds. They never relax a base MUST-rule.
On any conflict, the base MUST-rule wins.

---

## Scope

- **Target branch:** `develop`.
- **Product paths:** `projects/hipblaslt/**` (the hipBLASLt component root, including `tensilelite/`).
- Paths below are written relative to the repo root; in a standalone hipBLASLt checkout they are relative to `projects/hipblaslt/`.
- Changes outside these (docs, repo tooling) follow the base bar only.

---

## Supplements

### Scoping of base rules (adds)
Bind the base change-classes and scope buckets to hipBLASLt paths:
- Frontend / API: `projects/hipblaslt/library/include/hipblaslt/**` (public headers), `projects/hipblaslt/library/src/**`.
- Tensile / kernel generation: `projects/hipblaslt/tensilelite/**`, especially `KernelWriter*.py`.
- Tests/clients: `projects/hipblaslt/clients/**`.

### Tightens M1 (defect-fix regression test)
The regression test for a defect fix must run in a **shared CI lane** (TheRock GitHub Actions or
Math CI; not local-only) for the affected gfx arch. A local-only repro does not satisfy M1 for
hipBLASLt.

### Adds H1 — known-bug two-PR flow
Known-bug entries live in `projects/hipblaslt/clients/tests/data/known_bugs.yaml` with a tracker
id and a time-box. This is the concrete implementation of the base "track and time-box
quarantines" requirement. Waiver code **`W-KNOWN-BUG`** declares a tracked two-PR plan.

### Adds — Tensile test levels (maps the base test-level SHOULD to real lanes)
C++ unit → Tensile pytest → client/API → integration → perf. Pick the lowest level that fails on
the regression. Map the base test-level table onto these lanes when advising or reviewing.

### Adds — gfx CI labels
Component CI vocabulary: `ci:gpu:<gfx>`, `ci:extended`, `ci:performance`. Labels select coverage;
they never waive the base test/flag policy. Discover the live label/gfx set from the repo's CI
config rather than hardcoding it.

### Tightens — device/architecture coverage
Kernel-generation and assembly changes must show coverage on the affected gfx arch(s) in CI, not
just host-side tests. Treat a build-only-but-untested arch as uncovered.

### Tightens — stale-base on high-coupling files
Make the base stale-base check concrete and stricter. High-coupling files:
`KernelWriter*.py`, `KernelWriterAssembly.py`, register/SGPR-lifetime code, shared `Components/*`.
Overlap with the base branch on any of these since the PR diverged → **mandatory** rebase + re-run
(base default is strong-recommend).

### Tightens — approvals
Changes to the high-coupling files above need **≥ 2 hipBLASLt code-owner approvals** after the
local team review (stricter than a generic base approval count).

### Adds — `W-TUNE`
Component-specific waiver for tuning-only PRs, on top of the base waiver set.

### Tightens M5 — tracker linking
A Jira key (`AIHPBLAS-` or `ROCM-`) in the branch name or PR title triggers Jira's dev-panel
auto-linking, creating the reverse edge automatically. Prefer this so M5's links resolve in both
directions.

### Adds — risky-moment region (configure the base timing gate)
hipBLASLt spans a Taiwan-based team (MI300 focus) and a North-American team (MI350 focus). The
pre-merge timing gate should weigh the **owning team's** region/timezone for "going into the
weekend / end of day," not just the author's local clock. (Set the concrete regions/hours here as
the team confirms them.)

---

## What the overlay cannot do
Drop the regression-test-on-defect rule, allow disabling tests to green CI (M3), or skip work
tracking on a non-trivial PR. Those are base MUSTs; the overlay can only make them stricter.

---

## Worked example this overlay is designed to catch
PR #7796 (StaggerU for TDM) was green ~2.5h before #7750 (SGPR release for wave-separated TDM)
landed in `KernelWriterAssembly.py`, then merged ~3 days later without re-test — producing a
gfx1250 compile fault on develop that neither PR's own CI could have seen. That is exactly the
stale-base-on-high-coupling-file case the pre-merge gate flags as **mandatory rebase + re-run**.
