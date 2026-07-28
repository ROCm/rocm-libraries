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

### Adds — characterization snapshot (`.ambr` golden) discipline
Scope: `projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/` — characterization tests
that pin TensileLite's *current* Python behavior as syrupy `.ambr` goldens, run in the `-m unit`
lane. Green means "behavior unchanged," not "correct"; a red is information about the PR's own
change. When a PR's diff touches any `.ambr` file, gate on all of the below.
- **Classify the red first.** The author must say which of three a golden change is: (a) an intended
  behavior change, (b) a real bug the golden just caught — fix the source, leave the golden alone, or
  (c) a fragile/non-deterministic test — make it deterministic (e.g. the `{basename, err}` digest /
  canonicalization). Never re-record to mask churn, never weaken production source to stabilize a
  golden, and never delete a failing test to go green.
- **Surgical, never blanket.** Update goldens only on the smallest affected node
  (`pytest <node-id> --snapshot-update`) and review every `.ambr` diff line. A blanket
  `pytest -m unit --snapshot-update` is forbidden: while CI stays green it blesses real regressions,
  re-records vacuous goldens nobody read, un-pins deliberately-pinned bugs, and bakes in
  non-determinism. A PR whose `.ambr` diff spans many unrelated nodes/files is rejected and re-scoped
  to the nodes the behavior change actually affects.
- **ADR required for pinned or changed behavior.** A golden that pins known-wrong behavior, or any
  non-obvious golden change, MUST land with an Architecture Decision Record under
  `.../characterization/adr/`: one short, append-only file per decision in Nygard form
  (`Status` / `Context` / `Decision` / `Consequences`), carrying a `Defect:` tracker link when a real
  bug is pinned. ADRs are *superseded*, not edited in place — if a later fix flips the golden, update
  the golden and supersede the ADR. (The running catalog of pinned behaviors / accepted mutants stays
  a registry, e.g. `DECISIONS.md`, not one ADR each.) A behavior-changing `.ambr` diff with no
  matching ADR does not pass review — the concrete hipBLASLt form of the base "document intentional
  behavior changes" rule.
- **Stable-arch goldens are a regression signal.** On stable archs (gfx908 / gfx90a / gfx942) a
  codegen-digest golden change is a suspected compiler/codegen regression: require explicit
  root-cause and sign-off (ADR + PR description) before accepting the new golden. Newer, still-churning
  archs may keep a few compiler generations side by side.
- **Re-run before push.** After recording, the node is byte-identical on two further
  `--snapshot-update`-free runs, and the full `-m unit` suite stays green.

References: the characterization `README.md` ("Snapshot / golden discipline") and the `adr/` records.

### Adds — gfx CI labels
Component CI vocabulary: `ci:gpu:<gfx>`, `ci:extended`, `ci:performance`. Labels select coverage;
they never waive the base test/flag policy. Discover the live label/gfx set from the repo's CI
config rather than hardcoding it.

### Tightens — device/architecture coverage
Kernel-generation and assembly changes must show coverage on the affected gfx arch(s) in CI, not
just host-side tests. Treat a build-only-but-untested arch as uncovered.

### Tightens — stale-base on high-coupling files
Make the base stale-base check concrete and stricter. High-coupling files:
`KernelWriter*.py` (which already covers `KernelWriterAssembly.py`), register/SGPR-lifetime code, shared `Components/*`.
Overlap with the base branch on any of these since the PR diverged → **mandatory** rebase + re-run
(base default is strong-recommend).

### Tightens — stale-base on validator vs. validated-data (no file overlap)
Some merge-time breakage has **no file overlap at all**: one PR changes a *validator or allow-list*
while another adds *data the validator checks*. Treat these as a coupled pair even though they touch
different paths. The concrete hipBLASLt instance:
- Validator / allow-list: `projects/hipblaslt/tensilelite/Tensile/Common/GlobalParameters.py` (the
  global-parameter registry and the `_assertGlobalParametersAreValid` ignored-key allow-list), plus
  the enforcing test `projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_input_yaml_corpus_clean.py`.
- Validated data: YAML fixtures under `projects/hipblaslt/tensilelite/Tensile/Tests/**` (e.g.
  `common/gemm/**`).

If, since the PR diverged, the base branch changed the validator/allow-list **and** this PR adds or
edits validated data (or vice-versa), neither PR's own CI can see the collision → **mandatory**
rebase + re-run the strict-corpus test (`test_input_yaml_corpus_is_strict_clean`) before merge.
Generalize the check: when a PR *narrows* an allow-list/schema, scan the base for in-flight PRs
adding data the new rule would reject; when a PR *adds validated data*, confirm the allow-list/schema
on the merge-target still accepts it.

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
weekend / end of day," not just the author's local clock. Concrete weekend windows: Taiwan team
UTC+8 Fri 17:00 – Mon 09:00; NA team UTC-7 Fri 17:00 – Mon 09:00. A merge touching a high-coupling
file inside the owning team's window should defer or require explicit owner sign-off.

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

PR #7781 added gfx950 MXFP8 YAML fixtures carrying dead `MergeFiles` / `DeviceLDS` keys, while #8714
removed those keys from the `GlobalParameters` allow-list. The two PRs shared **no file**, so both
went green; once both were on `develop` the strict-corpus unit test
(`test_input_yaml_corpus_is_strict_clean`) rejected the fixtures, and the break only surfaced
downstream in the ROCm/TheRock rocm-libraries bump (TheRock #6133, tracked by #8810). That is the
validator-vs-validated-data collision the no-file-overlap gate above is designed to catch.
